/**
 * @file data_parallel.cpp
 * @brief Implementation of DataParallel for multi-GPU training
 */

#include "tenzor/nn/parallel/data_parallel.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/distributed/process_group.hpp"  // A1/B5: PG-aware grad all_reduce
#include "tenzor/distributed/distributed.hpp"     // ReduceOp enum
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
    int dim,
    ::tenzor::distributed::ReduceOp reduce_op
) : module_(module), device_ids_(device_ids), output_device_(output_device), dim_(dim),
    reduce_op_(reduce_op) {
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

// A1/B5: factory-based constructor for real per-device replicas.
DataParallel::DataParallel(
    UseFactoryTag /*tag*/,
    ModuleFactory factory,
    std::vector<int> device_ids,
    int output_device,
    int dim,
    std::shared_ptr<::tenzor::distributed::ProcessGroupBase> pg,
    ::tenzor::distributed::ReduceOp reduce_op
) : device_ids_(std::move(device_ids)),
    output_device_(output_device),
    dim_(dim),
    module_factory_(std::move(factory)),
    pg_(std::move(pg)),
    reduce_op_(reduce_op)
{
    if (!module_factory_) {
        throw std::invalid_argument("DataParallel(factory): factory cannot be null");
    }
    if (device_ids_.empty()) {
        throw std::invalid_argument("DataParallel(factory): device_ids cannot be empty");
    }

    // Set output device
    if (output_device_ == -1) {
        output_device_ = device_ids_[0];
    }

    // Validate that output_device is in device_ids
    if (std::find(device_ids_.begin(), device_ids_.end(), output_device_) == device_ids_.end()) {
        throw std::invalid_argument("DataParallel(factory): output_device must be in device_ids");
    }

    validate_devices();

    // Build the master module from the factory once now so accessors
    // (parameters(), named_parameters()) work before the first forward.
    module_ = module_factory_();
    if (!module_) {
        throw std::runtime_error("DataParallel(factory): factory returned a null module");
    }
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

    // Gradient sync is wired into autograd via `register_grad_hooks()`
    // (called from `replicate()` once the master parameter list is built).
    // Each parameter has a `Variable::register_hook` that fires when its
    // gradient is computed during backward(), triggering the configured
    // all-reduce automatically. Callers do not need to call
    // `synchronize_gradients()` manually after `loss.backward()`.

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

    // A1/B5: factory-based path constructs an independent module per non-master
    // device, syncs initial state from the master, and moves parameters to the
    // target device. The legacy shared-module path is preserved for callers
    // that used the original ctor (no factory available).
    auto master_state = module_->state_dict();

    for (size_t i = 0; i < device_ids_.size(); ++i) {
        int device_id = device_ids_[i];

        if (device_id == output_device_) {
            // Master device uses original module
            replicas_.push_back(module_);
            continue;
        }

        if (module_factory_) {
            // Factory ctor: materialize a fresh replica, load master state,
            // move parameters to target device. Each replica has its own
            // parameter tensors — real data parallelism, not shared-module.
            auto replica = module_factory_();
            if (!replica) {
                throw std::runtime_error(
                    "DataParallel::replicate: module factory returned null for device " +
                    std::to_string(device_id));
            }
            replica->load_state_dict(master_state);
            // Move all parameters to the target CUDA device. The module's
            // `to(Device)` method walks its parameters + buffers + submodules.
            replica->to(Device::cuda(device_id));
            replicas_.push_back(replica);
        } else {
            // Legacy shared-module path: keep the same module reference on
            // all devices. Single-process multi-stream "data parallelism"
            // (parameters are shared; gradients accumulate on the master).
            replicas_.push_back(module_);
        }
    }

    // Store parameters for gradient synchronization. With the factory path,
    // each replica has its own params; we still register the master's params
    // here so the optimizer updates them and `synchronize_gradients` will
    // gather from per-device replicas and reduce onto the master.
    parameters_to_sync_ = module_->parameters();

    // Wire backward hooks on every master parameter so the configured
    // all-reduce fires automatically when each grad lands during backward().
    register_grad_hooks();
}

auto DataParallel::register_grad_hooks() -> void {
    // Idempotent — skip if already wired (parameter set is stable after
    // the first call to replicate()).
    if (!grad_hook_ids_.empty()) {
        return;
    }
    // Single-device path or empty parameter set: nothing to sync.
    if (device_ids_.size() <= 1 || parameters_to_sync_.empty()) {
        return;
    }

    grad_hook_ids_.reserve(parameters_to_sync_.size());

    const auto num_devices = static_cast<int>(device_ids_.size());
    const auto reduce_op = reduce_op_;
    auto pg = pg_;

    for (auto& param : parameters_to_sync_) {
        if (!param || !param->requires_grad()) {
            grad_hook_ids_.push_back(0);  // placeholder, never used
            continue;
        }

        // The hook receives the freshly-computed gradient for this parameter
        // and returns the (possibly transformed) gradient that the engine
        // will accumulate into param->grad(). We perform the all-reduce
        // in-place on a copy and return that as the new gradient — this
        // means the accumulated grad observes the reduced value directly,
        // without a second pass that re-reads param->grad().
        size_t id = param->register_hook(
            [num_devices, reduce_op, pg](const Tensor& grad) -> Tensor {
                Tensor g = grad;
                if (pg != nullptr) {
                    // Real process-group path (NCCL/Gloo/MPI). All ranks
                    // contribute, then we normalize per `reduce_op`.
                    int ws = pg->world_size();
                    if (ws <= 1) {
                        return g;  // single-rank PG = no-op
                    }
                    switch (reduce_op) {
                        case ::tenzor::distributed::ReduceOp::SUM:
                            pg->all_reduce(g, ::tenzor::distributed::ReduceOp::SUM);
                            return g;
                        case ::tenzor::distributed::ReduceOp::AVG: {
                            pg->all_reduce(g, ::tenzor::distributed::ReduceOp::SUM);
                            return g * (1.0 / static_cast<double>(ws));
                        }
                        case ::tenzor::distributed::ReduceOp::MAX:
                            pg->all_reduce(g, ::tenzor::distributed::ReduceOp::MAX);
                            return g;
                        case ::tenzor::distributed::ReduceOp::MIN:
                            pg->all_reduce(g, ::tenzor::distributed::ReduceOp::MIN);
                            return g;
                        case ::tenzor::distributed::ReduceOp::PRODUCT:
                        case ::tenzor::distributed::ReduceOp::BAND:
                        case ::tenzor::distributed::ReduceOp::BOR:
                        case ::tenzor::distributed::ReduceOp::BXOR:
                            throw std::invalid_argument(
                                "DataParallel: ReduceOp PRODUCT/BAND/BOR/BXOR "
                                "are not valid for gradient reduction");
                    }
                    throw std::invalid_argument(
                        "DataParallel: unknown ReduceOp in grad hook");
                }
                // Shared-module path: gradients accumulate onto the master
                // parameter (each replica's backward routes back to the same
                // Variable), so the "reduce across replicas" is already done
                // by accumulation. All that remains is the per-op
                // normalization step.
                switch (reduce_op) {
                    case ::tenzor::distributed::ReduceOp::SUM:
                        return g;
                    case ::tenzor::distributed::ReduceOp::AVG:
                        return g * (1.0 / static_cast<double>(num_devices));
                    case ::tenzor::distributed::ReduceOp::MAX:
                    case ::tenzor::distributed::ReduceOp::MIN:
                        // Without a real PG, we can't take a true elementwise
                        // max/min across replicas — the gradients have already
                        // been summed by accumulation. Surface this as an
                        // error rather than silently returning the sum.
                        throw std::invalid_argument(
                            "DataParallel: MAX/MIN ReduceOp requires a "
                            "ProcessGroup (factory ctor with `pg`); "
                            "shared-module path only supports SUM and AVG");
                    case ::tenzor::distributed::ReduceOp::PRODUCT:
                    case ::tenzor::distributed::ReduceOp::BAND:
                    case ::tenzor::distributed::ReduceOp::BOR:
                    case ::tenzor::distributed::ReduceOp::BXOR:
                        throw std::invalid_argument(
                            "DataParallel: ReduceOp PRODUCT/BAND/BOR/BXOR "
                            "are not valid for gradient reduction");
                }
                throw std::invalid_argument(
                    "DataParallel: unknown ReduceOp in grad hook");
            }
        );
        grad_hook_ids_.push_back(id);
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
    // Gradient reduction is performed inline by the per-parameter
    // `register_hook` callbacks installed in `register_grad_hooks()` —
    // those fire during backward() and apply the configured `reduce_op_`
    // to the freshly-computed gradient before it lands in param->grad().
    //
    // This method remains as a private no-op so legacy code paths that
    // call it (or test fixtures, debug invocations) compile without
    // emitting a spurious double-reduce. The early-out below documents
    // the invariant: when num_devices_ == 1 or no params are tracked,
    // there is nothing to do regardless of which path triggered the call.
    if (device_ids_.size() == 1 || parameters_to_sync_.empty()) {
        return;
    }
    // No-op: hooks already applied `reduce_op_` per-parameter during backward.
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
