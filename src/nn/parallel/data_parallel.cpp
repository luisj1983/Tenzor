/**
 * @file data_parallel.cpp
 * @brief Implementation of DataParallel for multi-GPU training
 */

#include "tenzor/nn/parallel/data_parallel.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/core/device_guard.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/loader_fwd.hpp"  // try_get_backend()
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

    device_type_ = detect_device_type();

    // Auto-detect devices if not provided
    if (device_ids_.empty()) {
        auto* backend = ::tenzor::try_get_backend(device_type_);
        int32_t device_count = backend ? backend->device_count() : 0;
        if (device_count == 0) {
            throw std::runtime_error(
                "DataParallel: no " + Device{device_type_, 0}.to_string() +
                " devices available");
        }
        for (int i = 0; i < device_count; ++i) {
            device_ids_.push_back(i);
        }
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
    output_device_(output_device), dim_(dim),
    module_factory_(std::move(module_factory)), pg_(std::move(pg)),
    reduce_op_(reduce_op) {
    if (!module_) {
        throw std::invalid_argument("DataParallel: module cannot be null");
    }
    device_type_ = detect_device_type();
    if (device_ids_.empty()) {
        auto* backend = ::tenzor::try_get_backend(device_type_);
        int32_t device_count = backend ? backend->device_count() : 0;
        if (device_count == 0) {
            throw std::runtime_error(
                "DataParallel: no " + Device{device_type_, 0}.to_string() +
                " devices available");
        }
        for (int i = 0; i < device_count; ++i) {
            device_ids_.push_back(i);
        }
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

auto DataParallel::detect_device_type() const -> Device::Type {
    // Prefer the device type of the module's own parameters (matches the
    // convention already used by DistributedDataParallel::init_comm_resources
    // and ZeROStage1Optimizer's detect_comm_device_type): a module already
    // resident on ROCm/Vulkan/OneAPI/MPS should drive DataParallel onto that
    // same backend rather than assuming CUDA.
    for (const auto& p : module_->parameters()) {
        if (p && p->tensor().device().type != Device::Type::CPU) {
            return p->tensor().device().type;
        }
    }
    // No GPU-resident parameter found (e.g. a freshly constructed module):
    // fall back to whichever GPU backend is actually loaded and has devices,
    // preferring CUDA to preserve the historical default.
    for (auto type : {Device::Type::CUDA, Device::Type::ROCm,
                       Device::Type::Vulkan, Device::Type::OneAPI,
                       Device::Type::MPS}) {
        if (auto* backend = ::tenzor::try_get_backend(type)) {
            if (backend->device_count() > 0) {
                return type;
            }
        }
    }
    return Device::Type::CUDA;
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

    // Replicate module if needed.
    //
    // Concurrency: forward() may be invoked from multiple host threads
    // concurrently. replicas_initialized_ is a plain (non-atomic) bool, so the
    // previous double-checked-locking pattern performed an unguarded read of it
    // outside replicas_mutex_ while another thread wrote it inside the lock --
    // a data race / UB under the C++ memory model, with no happens-before
    // edge guaranteeing that the populated replicas_/parameter state is visible
    // to a thread that observes the flag as true.
    //
    // We close the race by always taking the mutex before touching the flag.
    // The lock both serializes the read/check/write and establishes the
    // acquire/release ordering needed so that a thread which sees
    // replicas_initialized_ == true is guaranteed to see the fully populated
    // replica state written under the same lock. The mutex acquisition is
    // negligible relative to a multi-device forward pass.
    {
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

    // Gradient sync wiring depends on the path:
    //  - Shared-module path: replica backwards route into the same master
    //    Variables, so accumulation already reduces; the per-parameter hooks
    //    (register_grad_hooks()) apply only the AVG normalization. Automatic.
    //  - ProcessGroup path: the hooks all_reduce each grad as it lands during
    //    backward(). Automatic.
    //  - Factory/independent-replica path WITHOUT a ProcessGroup: each replica
    //    owns its own parameter Variables, so their grads are invisible to the
    //    master after backward() and cannot be reduced from a per-parameter
    //    hook. For this path the caller MUST call `synchronize_gradients()`
    //    after `loss.backward()` and before `optimizer.step()`; it sums each
    //    replica's grad onto the master and applies `reduce_op_`.

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
            // Move all parameters to the target device. The module's
            // `to(Device)` method walks its parameters + buffers + submodules.
            replica->to(Device{device_type_, device_id});
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

    // Determine whether this is the factory/independent-replica path that needs
    // an explicit cross-replica gradient reduction. This is true exactly when a
    // module factory built genuinely separate replicas AND there is no real
    // ProcessGroup to all-reduce through. In that case each non-master replica
    // owns its own parameter Variables whose gradients are *not* visible to the
    // master's parameters after backward(); they must be summed onto the master
    // explicitly (see synchronize_gradients()).
    independent_replica_sync_ =
        static_cast<bool>(module_factory_) && !pg_ && device_ids_.size() > 1;

    if (independent_replica_sync_) {
        // Cache each non-master replica's parameter list so synchronize_gradients()
        // can reduce per-parameter by positional correspondence. Replicas are all
        // built from the same factory and share the master's architecture (loaded
        // via load_state_dict), so parameters() ordering is stable and aligns
        // 1:1 with parameters_to_sync_. Validate that invariant up front rather
        // than silently mis-reducing if a replica's parameter count diverges.
        replica_parameters_.clear();
        replica_parameters_.reserve(replicas_.size());
        for (size_t i = 0; i < replicas_.size(); ++i) {
            auto& replica = replicas_[i];
            // The master device's slot aliases module_ (same Variables as
            // parameters_to_sync_), so it carries no separate gradient to add;
            // skip it by storing an empty list, keeping indices aligned with
            // replicas_/device_ids_.
            if (replica == module_) {
                replica_parameters_.emplace_back();
                continue;
            }
            auto rparams = replica->parameters();
            if (rparams.size() != parameters_to_sync_.size()) {
                throw std::runtime_error(
                    "DataParallel::replicate: replica parameter count (" +
                    std::to_string(rparams.size()) + ") does not match master (" +
                    std::to_string(parameters_to_sync_.size()) +
                    "); cannot establish replica->master gradient correspondence");
            }
            replica_parameters_.push_back(std::move(rparams));
        }
    }

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

    // Factory/independent-replica path without a ProcessGroup: each replica owns
    // its own parameter Variables, so a per-master-parameter hook cannot perform
    // the cross-replica reduction (the other replicas' gradients are not even
    // computed yet when an individual master grad lands during backward, and the
    // hook only sees the master shard's gradient). Worse, the shared-module
    // normalization below would divide the *master-only* gradient by N, which is
    // exactly the silent-wrong-gradient bug. For this path we leave the master
    // grads untouched during backward() and perform the real summation +
    // normalization in synchronize_gradients(), which the caller invokes after
    // loss.backward(). Skip hook installation entirely.
    if (independent_replica_sync_) {
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
                        // Single-graph shared-module design: scatter(narrow) ->
                        // per-replica forward -> gather(cat) form ONE autograd
                        // graph over the shared master parameters, so backward
                        // already accumulates each sample's contribution exactly
                        // once and `g` is the correct full-batch gradient. There
                        // are no N independent replica gradients left to average;
                        // dividing by num_devices here would under-scale every
                        // gradient by N. AVG is therefore a no-op relative to the
                        // already-reduced gradient (identical to SUM).
                        (void)num_devices;
                        return g;
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
        // This must happen for EVERY device, including the master slot
        // (device_id == output_device_): a CPU input against a GPU-resident
        // master module would otherwise leave the master chunk on the CPU,
        // producing a device mismatch in the replica forward. Mirror gather()'s
        // device check so we only pay for a transfer when actually needed
        // (like PyTorch, which relocates the first chunk too).
        int device_id = device_ids_[i];
        const auto& chunk_dev = chunk.tensor().device();
        if (chunk_dev.type != device_type_ || chunk_dev.index != device_id) {
            utils::wrap_preserving_grad(chunk, chunk.tensor().to(Device{device_type_, device_id}));
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

    // Strategy: run each replica's forward on its own device, sequentially.
    //
    // The op-dispatch layer does not expose a per-call stream binding, so
    // replicas_[i]->forward() necessarily issues its kernels on the active
    // device's default stream. The previous implementation created per-device
    // streams and recorded start/end events on them, but the forward was never
    // bound to those streams (no StreamGuard / set_stream). Consequently the
    // launches were already sequential, the events tracked *empty* streams, and
    // an event-based sync could complete before the forward's real work on the
    // default stream finished — a no-op barrier that risked gather() reading
    // incomplete results. We replace that misleading machinery with an honest
    // per-device synchronize on the stream the forward actually used (the
    // default stream), via the backend's own synchronize(device_id) — the same
    // abstraction DeviceGuard uses, so this works identically on CUDA, ROCm,
    // Vulkan, OneAPI, and MPS instead of only CUDA.
    auto* backend = ::tenzor::try_get_backend(device_type_);
    if (backend == nullptr) {
        throw std::runtime_error(
            "DataParallel::parallel_apply: no backend loaded for " +
            Device{device_type_, 0}.to_string());
    }

    outputs.resize(device_ids_.size());

    for (size_t i = 0; i < device_ids_.size(); ++i) {
        // Make device i current so its forward dispatches there, then block
        // until that device's default-stream work completes before moving on.
        DeviceGuard guard(Device{device_type_, device_ids_[i]});
        outputs[i] = replicas_[i]->forward(inputs[i]);
        backend->synchronize(device_ids_[i]);
    }

    // DeviceGuard restores the previously-active device as each guard above
    // goes out of scope; explicitly (re)select the master device here so the
    // subsequent gather() dispatches its transfers/cat against it.
    backend->set_device(output_device_);

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
        if (dev.type != device_type_ || dev.index != output_device_) {
            // Transfer the underlying tensor to the master device but keep
            // this Variable's grad_fn/requires_grad/hooks intact so the
            // replica's forward graph remains reachable from cat's backward.
            utils::wrap_preserving_grad(v, v.tensor().to(Device{device_type_, output_device_}));
        }
        outputs_on_master.push_back(std::move(v));
    }

    // Concatenate along batch dimension via the autograd-aware variant.
    // Returns a Variable with grad_fn = CatBackward whose saved inputs are
    // outputs_on_master — backward propagates per-replica gradients.
    return tenzor::cat(outputs_on_master, dim_);
}

auto DataParallel::synchronize_gradients() -> void {
    if (device_ids_.size() == 1 || parameters_to_sync_.empty()) {
        return;
    }

    if (!independent_replica_sync_) {
        // Shared-module and ProcessGroup paths: gradient reduction is performed
        // inline by the per-parameter `register_hook` callbacks installed in
        // `register_grad_hooks()` — those fire during backward() and apply the
        // configured `reduce_op_` before the grad lands in param->grad(). Calling
        // this method on those paths is a deliberate no-op (no double-reduce).
        return;
    }

    // Factory/independent-replica path (module_factory_ set, no ProcessGroup):
    // each non-master replica computed gradients on its OWN parameter Variables
    // during backward(). Those gradients are invisible to the master's
    // parameters, so we explicitly reduce them onto the master here. After this
    // call the master parameter's gradient equals the SUM (reduce_op_ == SUM) or
    // AVG (reduce_op_ == AVG, sum divided by the number of replicas) of all
    // replicas' gradients for that parameter — the correct full-batch gradient
    // the optimizer must step on.
    const size_t num_params = parameters_to_sync_.size();
    const auto num_devices = static_cast<double>(device_ids_.size());

    for (size_t p = 0; p < num_params; ++p) {
        auto& master = parameters_to_sync_[p];
        if (!master || !master->requires_grad()) {
            continue;
        }

        // Accumulate every replica's gradient for this parameter onto the
        // master. Start from the master's own shard gradient (if any), then add
        // each non-master replica's gradient, moving cross-device tensors onto
        // the master parameter's device first so the elementwise add is valid.
        const Device& master_dev = master->tensor().device();
        std::optional<Tensor> summed;
        if (master->has_grad() && master->grad().has_value()) {
            summed = *master->grad();
        }

        for (size_t i = 0; i < replica_parameters_.size(); ++i) {
            const auto& rparams = replica_parameters_[i];
            if (rparams.empty()) {
                // Master-device slot (aliases module_): its gradient is already
                // the `summed` seed above; do not double-count.
                continue;
            }
            const auto& rparam = rparams[p];
            if (!rparam || !rparam->has_grad() || !rparam->grad().has_value()) {
                continue;
            }
            Tensor g = *rparam->grad();
            if (g.device() != master_dev) {
                g = g.to(master_dev);
            }
            summed = summed.has_value() ? (*summed + g) : g;
        }

        if (!summed.has_value()) {
            // No replica produced a gradient for this parameter (e.g. unused in
            // this batch) — leave the master grad untouched.
            continue;
        }

        switch (reduce_op_) {
            case ::tenzor::distributed::ReduceOp::SUM:
                master->set_grad(*summed);
                break;
            case ::tenzor::distributed::ReduceOp::AVG:
                // Single-graph design: `summed` already holds the full-batch
                // gradient (each replica contributed the gradient for its own
                // scatter chunk, i.e. each sample exactly once). Dividing by
                // num_devices would under-scale by N, so AVG is a no-op relative
                // to the summed gradient (identical to SUM here).
                (void)num_devices;
                master->set_grad(*summed);
                break;
            default:
                // validate_reduce_op() already rejects everything but SUM/AVG on
                // the no-ProcessGroup path; this is defensive only.
                throw std::invalid_argument(
                    "DataParallel::synchronize_gradients: reduce_op MAX/MIN/"
                    "PRODUCT/bitwise require a ProcessGroup; the factory path "
                    "without a ProcessGroup only supports SUM and AVG");
        }
    }
}

auto DataParallel::validate_devices() -> void {
    auto* backend = ::tenzor::try_get_backend(device_type_);
    if (backend == nullptr) {
        throw std::runtime_error(
            "DataParallel: no backend loaded for " +
            Device{device_type_, 0}.to_string());
    }

    int32_t device_count = backend->device_count();
    for (int device_id : device_ids_) {
        if (device_id < 0 || device_id >= device_count) {
            throw std::invalid_argument(
                "DataParallel: invalid device_id " + std::to_string(device_id) +
                " (available: 0-" + std::to_string(device_count - 1) + ")"
            );
        }
    }
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
