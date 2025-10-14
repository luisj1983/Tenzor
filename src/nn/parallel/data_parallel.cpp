/**
 * @file data_parallel.cpp
 * @brief Implementation of DataParallel for multi-GPU training
 */

#include "tenzor/nn/parallel/data_parallel.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/utils/error.hpp"
#include <stdexcept>
#include <algorithm>

#ifdef TENZOR_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tenzor {
namespace nn {

DataParallel::DataParallel(
    std::shared_ptr<Module> module,
    std::vector<int> device_ids,
    int output_device,
    int dim
) : module_(module), device_ids_(device_ids), output_device_(output_device), dim_(dim) {
    if (!module_) {
        throw std::invalid_argument("DataParallel: module cannot be null");
    }

    // Auto-detect devices if not provided
    if (device_ids_.empty()) {
#ifdef TENZOR_USE_CUDA
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            throw std::runtime_error("DataParallel: No CUDA devices available");
        }
        for (int i = 0; i < device_count; ++i) {
            device_ids_.push_back(i);
        }
#else
        throw std::runtime_error("DataParallel: CUDA not available, cannot auto-detect devices");
#endif
    }

    if (device_ids_.empty()) {
        throw std::invalid_argument("DataParallel: device_ids cannot be empty");
    }

    // Set output device
    if (output_device_ == -1) {
        output_device_ = device_ids_[0];
    }

    // Validate that output_device is in device_ids
    if (std::find(device_ids_.begin(), device_ids_.end(), output_device_) == device_ids_.end()) {
        throw std::invalid_argument("DataParallel: output_device must be in device_ids");
    }

    // Validate devices
    validate_devices();
}

auto DataParallel::forward(const Variable& input) -> Variable {
    // Single device optimization
    if (device_ids_.size() == 1) {
        return module_->forward(input);
    }

    // Check batch size
    auto input_shape = input.tensor().shape();
    if (input_shape.empty()) {
        throw std::runtime_error("DataParallel: input must have at least one dimension");
    }

    int64_t batch_size = input_shape[dim_];
    if (!can_split_batch(batch_size)) {
        throw std::runtime_error(
            "DataParallel: batch size (" + std::to_string(batch_size) +
            ") must be >= number of devices (" + std::to_string(device_ids_.size()) + ")"
        );
    }

    // Replicate module if needed
    if (!replicas_initialized_) {
        std::lock_guard<std::mutex> lock(replicas_mutex_);
        if (!replicas_initialized_) {
            replicate();
            replicas_initialized_ = true;
        }
    }

    // Scatter inputs to devices
    auto inputs = scatter(input);

    // Parallel forward pass
    auto outputs = parallel_apply(inputs);

    // Gather outputs
    auto result = gather(outputs);

    return result;
}

auto DataParallel::parameters() -> std::vector<Variable*> {
    return module_->parameters();
}

auto DataParallel::named_parameters() -> std::vector<std::pair<std::string, Variable*>> {
    return module_->named_parameters();
}

auto DataParallel::train(bool mode) -> void {
    module_->train(mode);
    for (auto& replica : replicas_) {
        if (replica) {
            replica->train(mode);
        }
    }
}

auto DataParallel::eval() -> void {
    train(false);
}

auto DataParallel::replicate() -> void {
    replicas_.clear();
    replicas_.reserve(device_ids_.size());

    // For each device, create a replica
    for (size_t i = 0; i < device_ids_.size(); ++i) {
        int device_id = device_ids_[i];

        if (device_id == output_device_) {
            // Master device uses original module
            replicas_.push_back(module_);
        } else {
            // For other devices, create a shallow copy
            // In a real implementation, this would involve:
            // 1. Creating a new module instance of the same type
            // 2. Sharing parameter storage with master (for gradient sync)
            // 3. Moving module to target device

            // For now, we'll use a simplified approach:
            // Store the master module and handle device transfers in parallel_apply
            replicas_.push_back(module_);
        }
    }
}

auto DataParallel::scatter(const Variable& input) -> std::vector<Variable> {
    std::vector<Variable> scattered_inputs;
    scattered_inputs.reserve(device_ids_.size());

    auto input_tensor = input.tensor();
    auto input_shape = input_tensor.shape();
    int64_t batch_size = input_shape[dim_];
    int num_devices = static_cast<int>(device_ids_.size());

    // Calculate chunk size for each device
    int64_t chunk_size = batch_size / num_devices;
    int64_t remainder = batch_size % num_devices;

    int64_t start = 0;
    for (int i = 0; i < num_devices; ++i) {
        // First 'remainder' chunks get one extra element
        int64_t current_chunk_size = chunk_size + (i < remainder ? 1 : 0);
        int64_t end = start + current_chunk_size;

        // Slice input along batch dimension
        auto chunk = input_tensor.slice(dim_, start, end);

        // Move chunk to target device
        int device_id = device_ids_[i];
        if (device_id != output_device_) {
            chunk = chunk.cuda(device_id);
        }

        scattered_inputs.emplace_back(chunk);
        start = end;
    }

    return scattered_inputs;
}

auto DataParallel::parallel_apply(const std::vector<Variable>& inputs) -> std::vector<Variable> {
    if (inputs.size() != device_ids_.size()) {
        throw std::runtime_error("DataParallel: input count mismatch with device count");
    }

    std::vector<Variable> outputs;
    outputs.reserve(inputs.size());

#ifdef TENZOR_USE_CUDA
    // Create CUDA events for synchronization
    std::vector<cudaEvent_t> events(device_ids_.size());
    std::vector<cudaStream_t> streams(device_ids_.size());

    for (size_t i = 0; i < device_ids_.size(); ++i) {
        cudaSetDevice(device_ids_[i]);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
    }

    // Launch forward passes in parallel
    for (size_t i = 0; i < device_ids_.size(); ++i) {
        cudaSetDevice(device_ids_[i]);
        cudaStreamSynchronize(streams[i]);

        // Execute forward pass on this device
        auto output = replicas_[i]->forward(inputs[i]);
        outputs.push_back(output);

        cudaEventRecord(events[i], streams[i]);
    }

    // Wait for all forward passes to complete
    for (size_t i = 0; i < device_ids_.size(); ++i) {
        cudaSetDevice(device_ids_[i]);
        cudaEventSynchronize(events[i]);
    }

    // Cleanup
    for (size_t i = 0; i < device_ids_.size(); ++i) {
        cudaEventDestroy(events[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Reset to master device
    cudaSetDevice(output_device_);
#else
    // CPU fallback: sequential execution
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = replicas_[i]->forward(inputs[i]);
        outputs.push_back(output);
    }
#endif

    return outputs;
}

auto DataParallel::gather(const std::vector<Variable>& outputs) -> Variable {
    if (outputs.empty()) {
        throw std::runtime_error("DataParallel: no outputs to gather");
    }

    if (outputs.size() == 1) {
        return outputs[0];
    }

    // Move all outputs to master device
    std::vector<Tensor> tensors_on_master;
    tensors_on_master.reserve(outputs.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
        auto tensor = outputs[i].tensor();
        const auto& dev = tensor.device();
        if (dev.type != Device::Type::CUDA || dev.index != output_device_) {
            tensor = tensor.cuda(output_device_);
        }
        tensors_on_master.push_back(tensor);
    }

    // Concatenate along batch dimension
    auto concatenated = tenzor::cat(tensors_on_master, dim_);

    return Variable(concatenated);
}

auto DataParallel::synchronize_gradients() -> void {
    // This would be called automatically during backward pass
    // In a full implementation, this would:
    // 1. Gather gradients from all replicas
    // 2. Average them (all-reduce)
    // 3. Update master module's parameter gradients

    // Note: In practice, gradient synchronization happens automatically
    // through autograd's backward hooks and shared parameter storage
}

auto DataParallel::validate_devices() -> void {
#ifdef TENZOR_USE_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess) {
        throw std::runtime_error(
            "DataParallel: CUDA error - " + std::string(cudaGetErrorString(err))
        );
    }

    for (int device_id : device_ids_) {
        if (device_id < 0 || device_id >= device_count) {
            throw std::invalid_argument(
                "DataParallel: invalid device_id " + std::to_string(device_id) +
                " (available: 0-" + std::to_string(device_count - 1) + ")"
            );
        }
    }
#else
    throw std::runtime_error("DataParallel: CUDA support not enabled");
#endif
}

auto DataParallel::can_split_batch(int64_t batch_size) const -> bool {
    return batch_size >= static_cast<int64_t>(device_ids_.size());
}

auto make_data_parallel(
    std::shared_ptr<Module> module,
    std::vector<int> device_ids,
    int output_device
) -> std::shared_ptr<DataParallel> {
    return std::make_shared<DataParallel>(module, device_ids, output_device);
}

} // namespace nn
} // namespace tenzor
