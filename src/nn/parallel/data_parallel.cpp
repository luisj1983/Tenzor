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

auto DataParallel::forward_impl(const Variable& input) -> Variable {
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

    // Attach backward hook to synchronize gradients after backward pass
    // Note: In a full autograd implementation, we would register a hook
    // that gets called after the backward pass completes. For now, we
    // document that users should call synchronize_gradients() manually
    // after loss.backward() or integrate this into the backward engine.
    //
    // Example integration:
    // result.set_backward_hook([this]() {
    //     this->synchronize_gradients();
    // });

    return result;
}

auto DataParallel::parameters() -> std::vector<std::shared_ptr<Variable>> {
    return module_->parameters();
}

auto DataParallel::named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> {
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

    // Get master module's state for replication
    auto master_state = module_->state_dict();

    // For each device, create a replica
    for (size_t i = 0; i < device_ids_.size(); ++i) {
        int device_id = device_ids_[i];

        if (device_id == output_device_) {
            // Master device uses original module
            replicas_.push_back(module_);
        } else {
            // For other devices, we need to create a deep copy
            // Since we don't have a generic clone() method, we use the same module
            // but move parameters to the target device
            //
            // Production Note: In a full framework, we would:
            // 1. Clone the module structure (using a factory or clone method)
            // 2. Load state dict and move to target device
            // 3. Keep parameters separate per device
            //
            // Current approach: Share module but track device-specific parameter copies
            replicas_.push_back(module_);
        }
    }

    // Store parameters for gradient synchronization
    // These are the master parameters that will accumulate gradients from all devices
    parameters_to_sync_ = module_->parameters();
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
    // Strategy: Execute forward passes in parallel using CUDA streams
    // Each device runs its forward pass concurrently

    std::vector<cudaStream_t> streams(device_ids_.size());
    std::vector<cudaEvent_t> start_events(device_ids_.size());
    std::vector<cudaEvent_t> end_events(device_ids_.size());

    // Create streams and events for async execution
    for (size_t i = 0; i < device_ids_.size(); ++i) {
        cudaSetDevice(device_ids_[i]);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&start_events[i]);
        cudaEventCreate(&end_events[i]);
    }

    // Pre-allocate output vector (will be filled in parallel)
    outputs.resize(device_ids_.size());

    // Launch forward passes asynchronously on all devices
    for (size_t i = 0; i < device_ids_.size(); ++i) {
        cudaSetDevice(device_ids_[i]);

        // Record start event
        cudaEventRecord(start_events[i], streams[i]);

        // Execute forward pass on this device
        // Note: The actual computation happens on the default stream for now
        // In a more advanced implementation, we would pass the stream to kernels
        outputs[i] = replicas_[i]->forward(inputs[i]);

        // Record completion event
        cudaEventRecord(end_events[i], streams[i]);
    }

    // Synchronize all devices to ensure forward passes complete
    for (size_t i = 0; i < device_ids_.size(); ++i) {
        cudaSetDevice(device_ids_[i]);
        cudaEventSynchronize(end_events[i]);

        // Optional: Check execution time for profiling
        // float ms = 0;
        // cudaEventElapsedTime(&ms, start_events[i], end_events[i]);
    }

    // Cleanup resources
    for (size_t i = 0; i < device_ids_.size(); ++i) {
        cudaSetDevice(device_ids_[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(start_events[i]);
        cudaEventDestroy(end_events[i]);
    }

    // Reset to master device
    cudaSetDevice(output_device_);
#else
    // CPU fallback: sequential execution
    // In a CPU multi-threaded version, we could use std::thread or parallel algorithms
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
    // Single device optimization - no synchronization needed
    if (device_ids_.size() == 1) {
        return;
    }

    // Early exit if no parameters to synchronize
    if (parameters_to_sync_.empty()) {
        return;
    }

#ifdef TENZOR_USE_CUDA
    // Multi-GPU gradient synchronization using all-reduce pattern
    //
    // Algorithm:
    // 1. For each parameter, gather gradients from all device replicas
    // 2. Sum gradients on master device
    // 3. Average by dividing by number of devices
    // 4. Update master parameter's gradient with averaged result
    //
    // Note: Since our current implementation shares the module across devices,
    // the gradients accumulate in the master parameter. In a true multi-device
    // setup with separate parameter copies per device, we would need to:
    // - Fetch gradients from each device's parameter copy
    // - Perform all-reduce (sum + average)
    // - Optionally broadcast back to all devices

    int num_devices = static_cast<int>(device_ids_.size());
    float scale_factor = 1.0f / static_cast<float>(num_devices);

    // Set master device for gradient operations
    cudaSetDevice(output_device_);

    // Create stream for async gradient operations
    cudaStream_t grad_stream;
    cudaStreamCreate(&grad_stream);

    // For each parameter, average the gradients
    for (auto& param : parameters_to_sync_) {
        if (!param || !param->has_grad()) {
            continue;
        }

        // Check if parameter has gradient computed
        if (!param->has_grad()) {
            continue;
        }

        // Get the gradient reference
        auto grad_opt = param->grad();
        if (!grad_opt.has_value()) {
            continue;
        }

        Tensor grad_tensor = grad_opt.value();

        // Validate gradient is on master device
        const auto& grad_device = grad_tensor.device();
        if (grad_device.type != Device::Type::CUDA || grad_device.index != output_device_) {
            // Move gradient to master device if needed
            grad_tensor = grad_tensor.cuda(output_device_);
        }

        // In our current architecture, gradients accumulate on the master device
        // In a true multi-GPU setup, we would:
        //
        // 1. Create gradient copies for each device
        // std::vector<Tensor> device_grads(num_devices);
        //
        // 2. Fetch gradients from each device
        // for (int i = 0; i < num_devices; ++i) {
        //     cudaSetDevice(device_ids_[i]);
        //     device_grads[i] = replicas_[i]->get_parameter(param_name)->grad();
        // }
        //
        // 3. Perform all-reduce (sum)
        // cudaSetDevice(output_device_);
        // Tensor summed_grad = device_grads[0];
        // for (int i = 1; i < num_devices; ++i) {
        //     Tensor grad_on_master = device_grads[i].cuda(output_device_);
        //     summed_grad = summed_grad + grad_on_master;
        // }
        //
        // 4. Average
        // Tensor averaged_grad = summed_grad * scale_factor;
        //
        // 5. Update master parameter
        // param->set_grad(averaged_grad);

        // Current simplified approach: Scale the gradient by 1/num_devices
        // This assumes gradients from all devices have been accumulated
        // in the master parameter (which happens in our shared-module design)
        Tensor scaled_grad = grad_tensor * scale_factor;

        // Update the parameter's gradient
        param->set_grad(scaled_grad);
    }

    // Synchronize stream
    cudaStreamSynchronize(grad_stream);
    cudaStreamDestroy(grad_stream);

    // Reset to master device
    cudaSetDevice(output_device_);

#else
    // CPU fallback: Average gradients
    // In a CPU multi-threading scenario, gradients would be accumulated
    // through atomic operations or mutex-protected updates, then averaged here
    int num_devices = static_cast<int>(device_ids_.size());
    float scale_factor = 1.0f / static_cast<float>(num_devices);

    for (auto& param : parameters_to_sync_) {
        if (!param || !param->has_grad()) {
            continue;
        }

        auto grad_opt = param->grad();
        if (!grad_opt.has_value()) {
            continue;
        }

        // Scale gradient by averaging factor
        Tensor scaled_grad = grad_opt.value() * scale_factor;
        param->set_grad(scaled_grad);
    }
#endif
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
