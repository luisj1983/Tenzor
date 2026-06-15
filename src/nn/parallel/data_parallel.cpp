/**
 * @file data_parallel.cpp
 * @brief Implementation of DataParallel for multi-GPU training
 */

#include "tenzor/nn/parallel/data_parallel.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/utils/autograd_wrap.hpp"
#include "tenzor/autograd/ops.hpp"  // autograd-aware tenzor::cat for gather()
#include "tenzor/distributed/process_group.hpp"  // A1/B5: PG-aware grad all_reduce
#include "tenzor/distributed/distributed.hpp"     // ReduceOp enum
#include <stdexcept>
#include <algorithm>
#include <string>
#include <functional>

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

    // No process group on the legacy ctor: only SUM/AVG are realizable via
    // gradient accumulation across the shared module. Reject MAX/MIN/etc. up
    // front (they would otherwise compile but throw on the first backward()).
    validate_reduce_op();
}

DataParallel::DataParallel(
    std::shared_ptr<Module> module,
    ModuleFactory module_factory,
    std::shared_ptr<::tenzor::distributed::ProcessGroupBase> pg,
    std::vector<int> device_ids,
    int output_device,
    int dim,
    ::tenzor::distributed::ReduceOp reduce_op
) : module_(std::move(module)), device_ids_(std::move(device_ids)),
    output_device_(output_device), dim_(dim), reduce_op_(reduce_op),
    module_factory_(std::move(module_factory)), pg_(std::move(pg)) {
    if (!module_) {
        throw std::invalid_argument("DataParallel: module cannot be null");
    }
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
    if (output_device_ == -1) {
        output_device_ = device_ids_[0];
    }
    if (std::find(device_ids_.begin(), device_ids_.end(), output_device_) == device_ids_.end()) {
        throw std::invalid_argument("DataParallel: output_device must be in device_ids");
    }
    validate_devices();
    // With pg_ set, MAX/MIN are realizable via real all_reduce; validation
    // accounts for that.
    validate_reduce_op();
}

auto DataParallel::validate_reduce_op() const -> void {
    if (pg_) {
        // Real process group: SUM/AVG/MAX/MIN are all valid all_reduce ops;
        // only the non-gradient reductions are rejected.
        switch (reduce_op_) {
            case ::tenzor::distributed::ReduceOp::SUM:
            case ::tenzor::distributed::ReduceOp::AVG:
            case ::tenzor::distributed::ReduceOp::MAX:
            case ::tenzor::distributed::ReduceOp::MIN:
                return;
            default:
                throw std::invalid_argument(
                    "DataParallel: reduce_op PRODUCT/BAND/BOR/BXOR are not valid "
                    "for gradient reduction");
        }
    }
    switch (reduce_op_) {
        case ::tenzor::distributed::ReduceOp::SUM:
        case ::tenzor::distributed::ReduceOp::AVG:
            return;
        default:
            throw std::invalid_argument(
                "DataParallel: reduce_op MAX/MIN/PRODUCT/bitwise require a "
                "ProcessGroup (use the factory/ProcessGroup ctor); the "
                "shared-module path only supports SUM and AVG");
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

    // Normalize the (possibly negative) batch dimension against the input rank
    // and bounds-check it. dim_ is stored verbatim; a negative dim_ (-1 is a
    // valid PyTorch convention) or dim_ >= rank would index input_shape (a span
    // indexed by size_t) out of bounds.
    const int64_t rank = static_cast<int64_t>(input_shape.size());
    int64_t norm_dim = (dim_ < 0) ? dim_ + rank : dim_;
    if (norm_dim < 0 || norm_dim >= rank) {
        throw std::out_of_range(
            "DataParallel: batch dim " + std::to_string(dim_) +
            " out of range for input rank " + std::to_string(rank));
    }

    int64_t batch_size = input_shape[norm_dim];
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

    auto input_shape = input.tensor().shape();
    // Normalize and bounds-check the batch dim (see forward_impl). Use the
    // normalized value for both the batch-size read and the narrow below.
    const int64_t in_rank = static_cast<int64_t>(input_shape.size());
    const int64_t norm_dim = (dim_ < 0) ? dim_ + in_rank : dim_;
    if (norm_dim < 0 || norm_dim >= in_rank) {
        throw std::out_of_range(
            "DataParallel::scatter: batch dim " + std::to_string(dim_) +
            " out of range for input rank " + std::to_string(in_rank));
    }
    int64_t batch_size = input_shape[norm_dim];
    int num_devices = static_cast<int>(device_ids_.size());

    // Calculate chunk size for each device
    int64_t chunk_size = batch_size / num_devices;
    int64_t remainder = batch_size % num_devices;

    int64_t start = 0;
    for (int i = 0; i < num_devices; ++i) {
        // First 'remainder' chunks get one extra element
        int64_t current_chunk_size = chunk_size + (i < remainder ? 1 : 0);

        // Slice the input *Variable* along the batch dimension with the
        // autograd-aware narrow (registers NarrowBackward) so gradients flow
        // back into the original input. The previous implementation sliced the
        // raw Tensor and re-wrapped each chunk via the leaf Variable(Tensor)
        // ctor — a disconnected leaf, so input.grad() was silently zero after
        // backward. This mirrors gather()'s wrap_preserving_grad round-trip.
        Variable chunk = ::tenzor::narrow(input, norm_dim, start, current_chunk_size);

        // Move chunk to target device, preserving grad_fn/requires_grad/hooks.
        int device_id = device_ids_[i];
        if (device_id != output_device_) {
            utils::wrap_preserving_grad(chunk, chunk.tensor().cuda(device_id));
        }

        scattered_inputs.push_back(std::move(chunk));
        start += current_chunk_size;
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

    // Check every CUDA return code and throw via the project error convention
    // instead of silently using invalid handles.
    auto cuda_check = [](cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("DataParallel::parallel_apply: ") + what + " - " +
                cudaGetErrorString(err));
        }
    };

    std::vector<cudaStream_t> streams(device_ids_.size(), nullptr);
    std::vector<cudaEvent_t> start_events(device_ids_.size(), nullptr);
    std::vector<cudaEvent_t> end_events(device_ids_.size(), nullptr);

    // RAII cleanup so a throw anywhere below (including replicas_[i]->forward)
    // still destroys every stream/event instead of leaking them. Destroy errors
    // are ignored during unwinding.
    auto destroy_all = [&]() noexcept {
        for (size_t i = 0; i < device_ids_.size(); ++i) {
            cudaSetDevice(device_ids_[i]);
            if (streams[i]) cudaStreamDestroy(streams[i]);
            if (start_events[i]) cudaEventDestroy(start_events[i]);
            if (end_events[i]) cudaEventDestroy(end_events[i]);
        }
    };
    struct ScopeGuard {
        std::function<void()> fn;
        ~ScopeGuard() { if (fn) fn(); }
    } guard{destroy_all};

    // Create streams and events for async execution
    for (size_t i = 0; i < device_ids_.size(); ++i) {
        cuda_check(cudaSetDevice(device_ids_[i]), "cudaSetDevice");
        cuda_check(cudaStreamCreate(&streams[i]), "cudaStreamCreate");
        cuda_check(cudaEventCreate(&start_events[i]), "cudaEventCreate(start)");
        cuda_check(cudaEventCreate(&end_events[i]), "cudaEventCreate(end)");
    }

    // Pre-allocate output vector (will be filled in parallel)
    outputs.resize(device_ids_.size());

    // Launch forward passes asynchronously on all devices
    for (size_t i = 0; i < device_ids_.size(); ++i) {
        cuda_check(cudaSetDevice(device_ids_[i]), "cudaSetDevice");

        // Record start event
        cuda_check(cudaEventRecord(start_events[i], streams[i]),
                   "cudaEventRecord(start)");

        // Execute forward pass on this device. If this throws, the ScopeGuard
        // above tears down all streams/events.
        outputs[i] = replicas_[i]->forward(inputs[i]);

        // Record completion event
        cuda_check(cudaEventRecord(end_events[i], streams[i]),
                   "cudaEventRecord(end)");
    }

    // Synchronize all devices to ensure forward passes complete
    for (size_t i = 0; i < device_ids_.size(); ++i) {
        cuda_check(cudaSetDevice(device_ids_[i]), "cudaSetDevice");
        cuda_check(cudaEventSynchronize(end_events[i]), "cudaEventSynchronize");
    }

    // Reset to master device (cleanup of streams/events runs via ScopeGuard).
    cuda_check(cudaSetDevice(output_device_), "cudaSetDevice(output)");
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

    // Move each replica output to the master device WITHOUT severing the
    // grad_fn chain. The previous implementation extracted .tensor() and
    // wrapped the cat result with the leaf Variable(Tensor) ctor — that
    // discards every replica's forward graph and makes backward dead-end
    // at the gather, silently zeroing all upstream gradients.
    //
    // Variables are kept in-place: we only swap the underlying data tensor
    // when a cross-device transfer is required (using wrap_preserving_grad
    // so grad_fn is preserved across the move). The autograd-aware
    // tenzor::cat(vector<Variable>, dim) then registers a CatBackward node
    // whose inputs are these per-replica Variables, chaining the gather
    // through each replica's forward pass exactly as PyTorch's gather does.
    std::vector<Variable> outputs_on_master;
    outputs_on_master.reserve(outputs.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
        Variable v = outputs[i];
        const auto& dev = v.tensor().device();
        if (dev.type != Device::Type::CUDA || dev.index != output_device_) {
            // Transfer the underlying tensor to the master device but keep
            // this Variable's grad_fn/requires_grad/hooks intact so the
            // replica's forward graph remains reachable from cat's backward.
            utils::wrap_preserving_grad(v, v.tensor().cuda(output_device_));
        }
        outputs_on_master.push_back(std::move(v));
    }

    // Concatenate along batch dimension via the autograd-aware variant.
    // Returns a Variable with grad_fn = CatBackward whose saved inputs are
    // outputs_on_master — backward propagates per-replica gradients.
    return tenzor::cat(outputs_on_master, dim_);
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
