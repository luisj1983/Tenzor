/**
 * @file fsdp.cpp
 * @brief Implementation of Fully Sharded Data Parallel (FSDP)
 *
 * Implements ZeRO-3 style parameter sharding. Parameters are flattened into
 * contiguous buffers per FSDP unit, sharded across ranks, and all-gathered
 * on demand for forward/backward computation.
 *
 * Memory lifecycle per FSDP unit (FULL_SHARD):
 *   idle:      local_shard_ only (1/N of params)
 *   forward:   flat_param_ (full) + local_shard_
 *   backward:  flat_param_ (full) + flat_grad_ (full) + local_shard_
 *   post-bwd:  local_shard_ + grad shard (1/N of grads)
 */

#include "tenzor/distributed/fsdp.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/utils/logging.hpp"
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <cstring>

// FINDING 60 (resolved): see the equivalent comment in ddp.cpp -- stream/
// event creation used to be picked at compile time via #if
// defined(TENZOR_USE_CUDA) #elif defined(TENZOR_USE_ROCM), which could never
// reach the ROCm branch on this project's combined CUDA+ROCm build.
// gpu_stream_ops.hpp dispatches at runtime instead, on comm_device_type_.
#include "tenzor/core/gpu_stream_ops.hpp"

namespace tenzor::distributed {

// ============================================================================
// FSDPUnit Implementation
// ============================================================================

FSDPUnit::FSDPUnit(nn::Module& module, ProcessGroup& pg, const FSDPConfig& config,
                   std::vector<std::shared_ptr<Variable>> explicit_params)
    : module_(module), pg_(&pg), config_(config),
      explicit_params_(std::move(explicit_params)) {

    use_gpu_comm_ = pg_->supports_async_stream();

    // Flatten all parameters into a single contiguous buffer
    flatten_params();

    // Shard the flat buffer across ranks
    shard_params();

    // Initialize communication resources for GPU backends
    init_comm_resources();
}

FSDPUnit::~FSDPUnit() {
    destroy_comm_resources();
}

// ============================================================================
// Parameter flattening and sharding
// ============================================================================

auto FSDPUnit::flatten_params() -> void {
    // The recursive auto-wrap policy assigns each unit an explicit, disjoint set
    // of parameters covering its slice of the module tree (the module's own
    // parameters plus any descendant parameters not claimed by a deeper unit).
    // When no explicit set was provided, fall back to this module's own (direct)
    // parameters.
    original_params_ = explicit_params_.empty() ? module_.own_parameters()
                                                : explicit_params_;

    if (original_params_.empty()) {
        return;
    }

    // Record shapes and compute total size
    total_numel_ = 0;
    for (const auto& param : original_params_) {
        if (!param) continue;
        auto shape = param->tensor().shape();
        param_shapes_.push_back(std::vector<int64_t>(shape.begin(), shape.end()));
        size_t numel = param->tensor().numel();
        param_numels_.push_back(numel);
        total_numel_ += numel;
    }

    if (total_numel_ == 0) {
        return;
    }

    // Determine device and dtype from the first parameter
    auto device = original_params_[0]->tensor().device();
    auto dtype = original_params_[0]->tensor().dtype();

    // The whole unit is flattened into ONE buffer of this (dtype, device). A
    // mixed-dtype/device unit (e.g. fp16 weights + fp32 norms) would be silently
    // mis-cast/mis-placed by slice_scatter, corrupting the gathered weights and
    // gradients. Require homogeneity (DDP groups by (dtype,device) instead).
    for (const auto& param : original_params_) {
        if (!param) continue;
        if (param->tensor().dtype() != dtype ||
            param->tensor().device() != device) {
            throw std::runtime_error(
                "FSDP: all parameters in a unit must share dtype and device; "
                "mixed-precision / multi-device units are not supported");
        }
    }

    // Allocate flat buffer and copy parameter data into it using slice_scatter
    // — device-agnostic, unlike std::memcpy which is invalid for GPU pointers.
    flat_param_ = zeros({static_cast<int64_t>(total_numel_)}, dtype, device);

    size_t offset = 0;
    for (const auto& param : original_params_) {
        if (!param) continue;
        size_t numel = param->tensor().numel();
        Tensor param_flat = param->tensor().reshape({static_cast<int64_t>(numel)}).contiguous();

        flat_param_ = slice_scatter(flat_param_, param_flat, /*dim=*/0,
                                    /*start=*/static_cast<int64_t>(offset),
                                    /*end=*/static_cast<int64_t>(offset + numel));
        offset += numel;
    }

    params_full_ = true;
}

auto FSDPUnit::shard_params() -> void {
    if (total_numel_ == 0) {
        return;
    }

    int ws = pg_->world_size();
    int rank = pg_->rank();

    // Compute shard size (ceil division; trailing ranks zero-padded). A high
    // rank's shard_offset_ can land at or beyond total_numel_ (e.g.
    // total_numel_=5, ws=4 -> shard_numel_=2, rank 3 -> shard_offset_=6); that
    // rank holds only padding. compute_shard_range clamps the valid length so
    // the unsigned subtraction below cannot underflow into a ~1.8e19 slice.
    const auto range = FSDPUnit::compute_shard_range(total_numel_, ws, rank);
    shard_numel_ = range.shard_numel;
    shard_offset_ = range.shard_offset;
    size_t actual_shard_numel = range.valid_numel;

    // Extract this rank's shard from the flat buffer
    auto dtype = flat_param_.dtype();
    auto device = flat_param_.device();

    // Start from zeros so the padded tail (when shard extends beyond total_numel)
    // is already zero without touching raw device memory.
    local_shard_ = zeros({static_cast<int64_t>(shard_numel_)}, dtype, device);

    // Copy the valid portion of the shard via slice_scatter (device-agnostic).
    if (actual_shard_numel > 0) {
        Tensor valid_src = slice(flat_param_, /*dim=*/0,
                                 static_cast<int64_t>(shard_offset_),
                                 static_cast<int64_t>(shard_offset_ + actual_shard_numel))
                               .contiguous();
        local_shard_ = slice_scatter(local_shard_, valid_src, /*dim=*/0,
                                     /*start=*/0,
                                     /*end=*/static_cast<int64_t>(actual_shard_numel));
    }

    if (config_.strategy == ShardingStrategy::FULL_SHARD) {
        // Free the full buffer -- we only keep the local shard
        flat_param_ = Tensor{};
        params_full_ = false;
    }

    // Offload to CPU if configured
    if (config_.cpu_offload) {
        offload_to_cpu();
    }
}

auto FSDPUnit::reshard_from_params() -> void {
    // Refresh local_shard_ (the persistent sharded source of truth) from the
    // current param Variables. After backward the optimizer updates the full
    // param Variables (original_params_) in place, but nothing else writes those
    // updates back into local_shard_; since all_gather_params() rebuilds the
    // full params from local_shard_ every step, without this the next forward
    // would restore the pre-step weights and FULL_SHARD training would make zero
    // progress. On the first step this reproduces the init shard (no-op).
    //
    // Unlike flatten_params(), this does NOT touch original_params_/param_shapes_
    // /param_numels_/total_numel_ (which flatten_params appends to), so it is
    // safe to call every step.
    if (total_numel_ == 0 || original_params_.empty()) {
        return;
    }
    auto dtype = original_params_[0]->tensor().dtype();
    auto device = original_params_[0]->tensor().device();

    // Rebuild a flat buffer from the current (optimizer-updated) params.
    Tensor flat = zeros({static_cast<int64_t>(total_numel_)}, dtype, device);
    size_t offset = 0;
    for (const auto& param : original_params_) {
        if (!param) continue;
        size_t numel = param->tensor().numel();
        Tensor param_flat =
            param->tensor().reshape({static_cast<int64_t>(numel)}).contiguous();
        flat = slice_scatter(flat, param_flat, /*dim=*/0,
                             static_cast<int64_t>(offset),
                             static_cast<int64_t>(offset + numel));
        offset += numel;
    }

    // Extract this rank's shard (shard_numel_/shard_offset_ were set by
    // shard_params() at construction and do not change).
    const auto range =
        FSDPUnit::compute_shard_range(total_numel_, pg_->world_size(), pg_->rank());
    size_t actual = range.valid_numel;
    Tensor new_shard = zeros({static_cast<int64_t>(shard_numel_)}, dtype, device);
    if (actual > 0) {
        Tensor valid_src = slice(flat, /*dim=*/0,
                                 static_cast<int64_t>(shard_offset_),
                                 static_cast<int64_t>(shard_offset_ + actual))
                               .contiguous();
        new_shard = slice_scatter(new_shard, valid_src, /*dim=*/0, /*start=*/0,
                                  static_cast<int64_t>(actual));
    }
    // Match local_shard_'s current device (e.g. CPU when cpu_offload is on and
    // reload hasn't run); the caller reloads before gather when needed.
    if (local_shard_.is_valid() && new_shard.device() != local_shard_.device()) {
        new_shard = new_shard.to(local_shard_.device());
    }
    local_shard_ = new_shard;
}

auto FSDPUnit::unflatten_params() -> void {
    if (!params_full_ || total_numel_ == 0) {
        return;
    }

    // Write values from flat_param_ back into original parameter Variables
    size_t offset = 0;
    for (size_t i = 0; i < original_params_.size(); ++i) {
        auto& param = original_params_[i];
        if (!param) continue;

        size_t numel = param_numels_[i];
        const auto& shape = param_shapes_[i];

        // Slice the flat buffer and reshape to the parameter's shape (device-agnostic).
        Tensor param_data =
            slice(flat_param_, /*dim=*/0,
                  static_cast<int64_t>(offset),
                  static_cast<int64_t>(offset + numel))
                .contiguous()
                .reshape(shape);

        // R.18: preserve the parameter's TensorImpl/Storage handle by writing
        // the new bytes in place. A bare `param->tensor() = param_data` swaps
        // the impl pointer, so any saved-for-backward activation that captured
        // the parameter Tensor before this call would see a stale TensorImpl
        // (frozen on the previous step's values) when the backward kernel
        // later reads it. zero_() + add_() is the codebase's public copy_
        // surrogate (cf. zero_optimizer.cpp:2834-2835) and keeps the same
        // TensorImpl/Storage alive.
        auto& dst = param->tensor();
        bool same_layout = dst.is_valid() &&
                           dst.dtype() == param_data.dtype() &&
                           dst.device() == param_data.device() &&
                           dst.numel() == param_data.numel() &&
                           std::vector<int64_t>(dst.shape().begin(), dst.shape().end()) == shape;
        if (same_layout) {
            dst.zero_();
            add_(dst, param_data);
        } else {
            // First-ever materialization (or shape changed): no outstanding
            // saved-activation references can exist yet, so reassignment is
            // safe.
            dst = param_data;
        }
        offset += numel;
    }
}

auto FSDPUnit::collect_grads() -> void {
    if (total_numel_ == 0) {
        return;
    }

    auto dtype = original_params_[0]->tensor().dtype();
    auto device = original_params_[0]->tensor().device();

    // Allocate flat gradient buffer with padded size
    int ws = pg_->world_size();
    size_t padded_numel = shard_numel_ * ws;
    flat_grad_ = zeros({static_cast<int64_t>(padded_numel)}, dtype, device);

    // Copy each parameter's gradient into the flat buffer
    size_t offset = 0;
    for (size_t i = 0; i < original_params_.size(); ++i) {
        auto& param = original_params_[i];
        if (!param) continue;

        size_t numel = param_numels_[i];

        if (param->has_grad()) {
            Tensor grad = param->grad().value();
            Tensor grad_flat = grad.reshape({static_cast<int64_t>(numel)}).contiguous();

            flat_grad_ = slice_scatter(flat_grad_, grad_flat, /*dim=*/0,
                                       static_cast<int64_t>(offset),
                                       static_cast<int64_t>(offset + numel));
        }
        offset += numel;
    }
}

auto FSDPUnit::scatter_grads_to_params() -> void {
    if (total_numel_ == 0) {
        return;
    }

    // After reduce-scatter, flat_grad_ contains only this rank's shard
    // of the averaged gradient. Write it back to the corresponding parameters.
    auto dtype = original_params_[0]->tensor().dtype();

    // Determine which parameters actually produced a gradient. collect_grads()
    // zero-fills the flat buffer and only writes a real gradient where
    // param->has_grad() was true, so a parameter never used in the forward
    // (e.g. an inactive MoE expert or a conditional branch) contributes only
    // zeros to the reduce-scatter. PyTorch leaves such parameters with
    // grad=None so the optimizer skips them (no weight-decay / momentum
    // update). If we unconditionally set_grad() below, we would hand those
    // parameters an explicit zero gradient and change optimizer dynamics.
    //
    // Because reduce-scatter sums across ranks, a parameter may be unused on
    // this rank yet have a real gradient on another rank; in that case the
    // reduced shard genuinely contains its contribution and we MUST set its
    // grad here. We therefore reconcile the per-parameter "has a real grad"
    // mask across ranks with an all-reduce MAX so the decision is globally
    // consistent: a parameter is materialized iff at least one rank produced a
    // gradient for it. At this point param->has_grad() still reflects the
    // original (pre-scatter) local gradients, since only flat_grad_ has been
    // modified so far.
    const size_t num_params = original_params_.size();
    std::vector<int64_t> local_has_grad(num_params, 0);
    for (size_t i = 0; i < num_params; ++i) {
        const auto& param = original_params_[i];
        if (param && param->has_grad()) {
            local_has_grad[i] = 1;
        }
    }

    std::vector<int64_t> global_has_grad = local_has_grad;
    if (pg_->world_size() > 1 && num_params > 0) {
        Tensor mask = zeros({static_cast<int64_t>(num_params)}, DType::Float32,
                            flat_grad_.device());
        {
            Tensor mask_cpu = zeros({static_cast<int64_t>(num_params)},
                                    DType::Float32, Device::cpu());
            float* mptr = mask_cpu.data<float>();
            for (size_t i = 0; i < num_params; ++i) {
                mptr[i] = static_cast<float>(local_has_grad[i]);
            }
            mask = mask_cpu.to(flat_grad_.device());
        }
        // A parameter has a grad globally iff any rank produced one.
        pg_->all_reduce(mask, ReduceOp::MAX);
        Tensor mask_cpu = mask.to(Device::cpu());
        const float* mptr = mask_cpu.data<float>();
        for (size_t i = 0; i < num_params; ++i) {
            global_has_grad[i] = (mptr[i] > 0.5f) ? 1 : 0;
        }
    }

    // The gradient shard corresponds to flat buffer offset [shard_offset_, shard_offset_ + shard_numel_)
    // We need to figure out which parameters overlap with this range and write partial grads
    size_t shard_start = shard_offset_;
    size_t shard_end = shard_offset_ + shard_numel_;

    size_t param_offset = 0;
    for (size_t i = 0; i < original_params_.size(); ++i) {
        auto& param = original_params_[i];
        if (!param) continue;

        size_t numel = param_numels_[i];
        size_t param_start = param_offset;
        size_t param_end = param_offset + numel;

        // Check overlap between this parameter and our shard
        size_t overlap_start = std::max(param_start, shard_start);
        size_t overlap_end = std::min(param_end, shard_end);

        // Leave never-used parameters with grad=None (PyTorch parity): if no
        // rank produced a gradient for this parameter, the reduced shard holds
        // only zeros and we must not synthesize a spurious zero gradient.
        if (overlap_start < overlap_end && global_has_grad[i]) {
            size_t overlap_numel = overlap_end - overlap_start;
            size_t grad_buf_offset = overlap_start - shard_start;
            size_t param_sub_offset = overlap_start - param_start;

            // Create a gradient tensor for this parameter; only the overlapping
            // portion has valid data. Use slice_scatter for device safety.
            const auto& shape = param_shapes_[i];
            Tensor grad_flat = zeros({static_cast<int64_t>(numel)}, dtype, flat_grad_.device());
            Tensor overlap_src =
                slice(flat_grad_, /*dim=*/0,
                      static_cast<int64_t>(grad_buf_offset),
                      static_cast<int64_t>(grad_buf_offset + overlap_numel))
                    .contiguous();
            grad_flat = slice_scatter(grad_flat, overlap_src, /*dim=*/0,
                                      static_cast<int64_t>(param_sub_offset),
                                      static_cast<int64_t>(param_sub_offset + overlap_numel));

            param->set_grad(grad_flat.reshape(shape));
        }

        param_offset += numel;
    }
}

// ============================================================================
// All-gather and reduce-scatter
// ============================================================================

auto FSDPUnit::all_gather_params() -> void {
    if (total_numel_ == 0) {
        return;
    }
    // FULL_SHARD skips a redundant re-gather when the full params are already
    // materialized. SHARD_GRAD_OP (ZeRO-2) keeps params resident (params_full_
    // stays true), but after the optimizer step each rank has updated only its
    // own shard slice of the full param, so the ranks have diverged and must be
    // reconciled by re-gathering every step — hence it does NOT early-return.
    if (params_full_ && config_.strategy != ShardingStrategy::SHARD_GRAD_OP) {
        return;
    }

    // Reload from CPU if offloaded
    if (offloaded_) {
        reload_from_cpu();
    }

    // Writeback: capture any optimizer update to the full param Variables into
    // local_shard_ before rebuilding the full params from it. For FULL_SHARD
    // this is the difference between training and a no-op; for SHARD_GRAD_OP it
    // extracts this rank's post-step shard slice so the subsequent gather
    // reconciles the divergent ranks. (No-op on the first step.)
    if (config_.strategy == ShardingStrategy::FULL_SHARD ||
        config_.strategy == ShardingStrategy::SHARD_GRAD_OP) {
        reshard_from_params();
    }

    int ws = pg_->world_size();
    auto dtype = local_shard_.dtype();
    auto device = local_shard_.device();

    // Mixed precision: cast shard to config_.comm_dtype before communication
    // for bandwidth savings. Audit-4 W.11: the comm dtype is now
    // configurable (default BFloat16) instead of being hardcoded to Float16
    // — BF16 has the same exponent range as F32 and is the standard AMP
    // dtype for transformer-scale training.
    //
    // Audit-5 Z.13: previously gated on `dtype == Float32 && comm_dtype !=
    // Float32`, which silently skipped compression for a native-F16 model
    // with the default `comm_dtype=BFloat16` (same size, but the F16->BF16
    // up-cast would still have been pointless and lossy). Widen the gate to
    // "down-cast whenever the param dtype is strictly wider than the comm
    // dtype" so any narrower-than-param comm dtype compresses regardless of
    // the source. Equal-size or wider comm dtypes are no-ops and we emit a
    // one-time warning so the user knows their comm_dtype request was
    // ignored.
    Tensor comm_shard = local_shard_;
    DType comm_dtype = dtype;
    const bool comm_narrower = ::tenzor::dtype_size(config_.comm_dtype) <
                               ::tenzor::dtype_size(dtype);
    const bool use_mixed_precision = config_.mixed_precision &&
                                     config_.comm_dtype != dtype &&
                                     comm_narrower;
    // audit-7 FF.17: the equal-size-but-different-family case (e.g. param
    // dtype F16 with comm_dtype BF16, both 2 bytes) falls under the generic
    // "no narrower" warning above, but the message is misleading — it isn't
    // wider, it's the same width but a different bit layout.  Emit a more
    // specific warning so the user knows their compression request produced
    // no bandwidth savings and they should pick a strictly smaller dtype.
    if (config_.mixed_precision &&
        ::tenzor::dtype_size(dtype) == ::tenzor::dtype_size(config_.comm_dtype) &&
        config_.comm_dtype != dtype) {
        TENZOR_WARN_ONCE(
            "FSDP: comm_dtype is the same size as the parameter dtype but a "
            "different family; no compression applied — consider setting "
            "comm_dtype to a smaller dtype (e.g. F8) for actual bandwidth savings.");
    } else if (config_.mixed_precision && config_.comm_dtype != dtype && !comm_narrower) {
        TENZOR_WARN_ONCE(
            "FSDP: comm_dtype is no narrower than the parameter dtype; "
            "skipping the mixed-precision down-cast to avoid wasted bandwidth "
            "and lossy F16->BF16 (or wider) reinterpretation.");
    }
    if (use_mixed_precision) {
        comm_shard = local_shard_.to(config_.comm_dtype);
        comm_dtype = config_.comm_dtype;
    }

    // Allocate output tensors for all-gather (one per rank)
    std::vector<Tensor> gathered(ws);
    for (int i = 0; i < ws; ++i) {
        gathered[i] = empty({static_cast<int64_t>(shard_numel_)}, comm_dtype, device);
    }

    // All-gather: each rank sends its shard, receives all shards
    pg_->all_gather(comm_shard, gathered);

    // Cast back to original dtype after communication
    if (use_mixed_precision) {
        for (auto& g : gathered) {
            g = g.to(dtype);
        }
    }

    // Concatenate all shards into the flat parameter buffer
    // Total size is shard_numel_ * ws (may be padded beyond total_numel_)
    size_t padded_numel = shard_numel_ * ws;
    flat_param_ = zeros({static_cast<int64_t>(padded_numel)}, dtype, device);

    for (int i = 0; i < ws; ++i) {
        int64_t off = static_cast<int64_t>(i) * static_cast<int64_t>(shard_numel_);
        flat_param_ = slice_scatter(flat_param_, gathered[i].contiguous(),
                                    /*dim=*/0, /*start=*/off,
                                    /*end=*/off + static_cast<int64_t>(shard_numel_));
    }

    params_full_ = true;

    // Write full parameters back to original parameter Variables
    unflatten_params();
}

auto FSDPUnit::free_full_params() -> void {
    if (!params_full_ || config_.strategy != ShardingStrategy::FULL_SHARD) {
        return;
    }

    // Free the full parameter buffer
    flat_param_ = Tensor{};
    params_full_ = false;

    // Offload local shard to CPU if configured
    if (config_.cpu_offload) {
        offload_to_cpu();
    }
}

auto FSDPUnit::reduce_scatter_grads() -> void {
    if (total_numel_ == 0) {
        return;
    }

    int ws = pg_->world_size();

    // Collect gradients from parameters into flat buffer
    collect_grads();

    auto dtype = flat_grad_.dtype();
    auto device = flat_grad_.device();

    // Mixed precision: compress gradients to config_.comm_dtype before the
    // reduce-scatter, symmetric with the forward all_gather_params() path so
    // the backward collective enjoys the same bandwidth savings. Use the
    // identical gate ("comm_dtype strictly narrower than the grad dtype") so we
    // never up-cast or reinterpret across same-size dtype families. The
    // diagnostic warnings for the no-op / same-size cases are already emitted
    // (once) by all_gather_params(), which runs earlier in the same step for
    // the same config, so we intentionally do not duplicate them here.
    const bool comm_narrower = ::tenzor::dtype_size(config_.comm_dtype) <
                               ::tenzor::dtype_size(dtype);
    const bool use_mixed_precision = config_.mixed_precision &&
                                     config_.comm_dtype != dtype &&
                                     comm_narrower;
    DType comm_dtype = use_mixed_precision ? config_.comm_dtype : dtype;

    Tensor comm_grad = use_mixed_precision ? flat_grad_.to(comm_dtype) : flat_grad_;

    // Split flat gradient into world_size chunks for reduce-scatter input
    std::vector<Tensor> grad_chunks(ws);
    for (int i = 0; i < ws; ++i) {
        int64_t off = static_cast<int64_t>(i) * static_cast<int64_t>(shard_numel_);
        grad_chunks[i] = slice(comm_grad, /*dim=*/0, off,
                               off + static_cast<int64_t>(shard_numel_))
                             .contiguous();
    }

    // Output: this rank's reduced gradient shard (in the comm dtype)
    Tensor grad_shard = empty({static_cast<int64_t>(shard_numel_)}, comm_dtype, device);

    // Reduce-scatter: sum across ranks, each rank gets its shard
    pg_->reduce_scatter(grad_chunks, grad_shard, ReduceOp::SUM);

    // Cast the reduced shard back to the full gradient dtype before averaging
    // so the division (and everything downstream) runs in full precision.
    if (use_mixed_precision) {
        grad_shard = grad_shard.to(dtype);
    }

    // Divide by world_size to compute average
    if (ws > 1) {
        grad_shard = grad_shard / static_cast<float>(ws);
    }

    // Store reduced shard back into flat_grad_ for scatter_grads_to_params
    flat_grad_ = grad_shard;

    // Write gradient shards back to parameters
    scatter_grads_to_params();

    // Re-shard parameters (they were all-gathered for backward)
    if (config_.strategy == ShardingStrategy::FULL_SHARD) {
        free_full_params();
    }
}

// ============================================================================
// CPU offloading
// ============================================================================

auto FSDPUnit::offload_to_cpu() -> void {
    if (offloaded_ || total_numel_ == 0) {
        return;
    }

    // Use Tensor::to() for a device-correct host transfer (handles any backend).
    cpu_shard_ = local_shard_.to(Device::cpu());

    // Free GPU shard
    local_shard_ = Tensor{};
    offloaded_ = true;
}

auto FSDPUnit::reload_from_cpu() -> void {
    if (!offloaded_ || total_numel_ == 0) {
        return;
    }

    // Determine target device from original parameters
    auto device = original_params_[0]->tensor().device();

    local_shard_ = cpu_shard_.to(device);

    // Free CPU copy
    cpu_shard_ = Tensor{};
    offloaded_ = false;
}

// ============================================================================
// Communication resource management
// ============================================================================

auto FSDPUnit::init_comm_resources() -> void {
    if (!use_gpu_comm_) {
        return;
    }

    // Determine the actual GPU vendor from this unit's own sharded parameter
    // buffer at runtime (FINDING 60) -- comm_stream_/comm_event_ are only
    // ever valid for this device type.
    comm_device_type_ = local_shard_.is_valid() ? local_shard_.device().type
                                                 : Device::Type::CUDA;

    comm_stream_ = core::gpu_stream::create_stream(comm_device_type_);
    comm_event_ = core::gpu_stream::create_event(comm_device_type_);
}

auto FSDPUnit::destroy_comm_resources() -> void {
    if (!use_gpu_comm_) {
        return;
    }

    if (comm_event_) {
        core::gpu_stream::destroy_event(comm_event_, comm_device_type_);
        comm_event_ = nullptr;
    }

    if (comm_stream_) {
        core::gpu_stream::destroy_stream(comm_stream_, comm_device_type_);
        comm_stream_ = nullptr;
    }
}

// ============================================================================
// FullyShardedDataParallel Implementation
// ============================================================================

FullyShardedDataParallel::FullyShardedDataParallel(
    nn::Module& module,
    ProcessGroup& pg,
    const FSDPConfig& config
) : module_(module), pg_(&pg), config_(config) {

    // Apply auto-wrap policy to identify FSDP units
    apply_auto_wrap();

    // Register forward hooks for automatic all-gather/free
    register_hooks();
}

FullyShardedDataParallel::~FullyShardedDataParallel() = default;

auto FullyShardedDataParallel::apply_auto_wrap() -> void {
    // Recursively partition the whole module tree so that every parameter is
    // sharded by exactly one unit. collect_units() creates a unit for each
    // qualifying subtree and returns the parameters left over at the root.
    auto root_remaining = collect_units(module_);

    // Always wrap the leftover root parameters (the root module's own parameters
    // and any small descendants that did not reach the threshold). Without this,
    // those parameters would never be sharded.
    if (!root_remaining.empty()) {
        units_.push_back(std::make_unique<FSDPUnit>(
            module_, *pg_, config_, std::move(root_remaining)));
    }
}

auto FullyShardedDataParallel::collect_units(nn::Module& module)
    -> std::vector<std::shared_ptr<Variable>> {
    // Start with this module's own (direct) parameters.
    std::vector<std::shared_ptr<Variable>> remaining = module.own_parameters();

    for (auto& [name, submodule] : module.get_submodules()) {
        if (!submodule) continue;

        // Bottom-up: first carve out any deeper units within this child.
        auto child_remaining = collect_units(*submodule);

        // Count trainable parameters left in the child subtree.
        size_t grad_numel = 0;
        for (const auto& p : child_remaining) {
            if (p && p->requires_grad()) {
                grad_numel += p->tensor().numel();
            }
        }

        if (!child_remaining.empty() &&
            grad_numel >= config_.auto_wrap_min_params) {
            // Child subtree is large enough to be its own unit.
            units_.push_back(std::make_unique<FSDPUnit>(
                *submodule, *pg_, config_, std::move(child_remaining)));
        } else {
            // Too small: bubble its parameters up to be wrapped by an ancestor.
            for (auto& p : child_remaining) {
                remaining.push_back(std::move(p));
            }
        }
    }

    return remaining;
}

auto FullyShardedDataParallel::wrap_module(nn::Module& module) -> void {
    units_.push_back(std::make_unique<FSDPUnit>(module, *pg_, config_));
}

auto FullyShardedDataParallel::count_params(nn::Module& module) const -> size_t {
    size_t count = 0;
    auto params = module.own_parameters();
    for (const auto& p : params) {
        if (p && p->requires_grad()) {
            count += p->tensor().numel();
        }
    }
    return count;
}

auto FullyShardedDataParallel::forward(const Variable& input) -> Variable {
    // All-gather parameters for all units before forward
    for (auto& unit : units_) {
        unit->all_gather_params();
    }

    // Execute the wrapped module's forward pass
    auto output = module_.forward(input);

    // Free non-local shards after forward (FULL_SHARD only)
    if (config_.strategy == ShardingStrategy::FULL_SHARD) {
        for (auto& unit : units_) {
            unit->free_full_params();
        }
    }

    return output;
}

auto FullyShardedDataParallel::finalize_backward() -> void {
    // For FULL_SHARD: all-gather params again (needed for gradient computation
    // with respect to weights, which requires the full weight values)
    if (config_.strategy == ShardingStrategy::FULL_SHARD) {
        for (auto& unit : units_) {
            unit->all_gather_params();
        }
    }

    // Reduce-scatter gradients for each unit
    for (auto& unit : units_) {
        unit->reduce_scatter_grads();
    }
}

auto FullyShardedDataParallel::summon_full_params() -> void {
    // NN.18: forward post-hook (see register_hooks) would otherwise free the
    // shards we just all-gathered the moment any wrapped module's forward()
    // runs inside the summon window.  Each unit carries a re-entry counter
    // (summon_depth_) — bump it here, the post-hook sees it non-zero and
    // skips the free, and release_full_params() drops it.
    for (auto& unit : units_) {
        unit->enter_summon();
        unit->all_gather_params();
    }
}

auto FullyShardedDataParallel::release_full_params() -> void {
    // NN.18: pair with summon_full_params() — drop the re-entry counter
    // *before* freeing so the actual free can proceed unblocked.
    for (auto& unit : units_) {
        unit->exit_summon();
        unit->free_full_params();
    }
}

auto FullyShardedDataParallel::total_params() const -> size_t {
    size_t total = 0;
    for (const auto& unit : units_) {
        total += unit->total_numel();
    }
    return total;
}

auto FullyShardedDataParallel::sharded_param_bytes() const -> size_t {
    size_t total = 0;
    for (const auto& unit : units_) {
        if (unit->total_numel() > 0) {
            // Each rank holds shard_numel elements
            // flat_param may be empty if sharded; use module params for dtype
            auto own_params = unit->module().own_parameters();
            DType dtype = own_params.empty() ? DType::Float32
                                              : own_params[0]->tensor().dtype();
            total += unit->shard_numel() * dtype_size(dtype);
        }
    }
    return total;
}

auto FullyShardedDataParallel::register_hooks() -> void {
    // Register forward pre-hooks on each FSDP unit's module for automatic
    // all-gather when the module's forward is called
    for (size_t i = 0; i < units_.size(); ++i) {
        auto* unit_ptr = units_[i].get();

        unit_ptr->module().register_forward_pre_hook(
            [unit_ptr](nn::Module* /*module*/, const Variable& /*input*/) {
                unit_ptr->all_gather_params();
            }
        );

        // Post-hook to free full params after forward
        if (config_.strategy == ShardingStrategy::FULL_SHARD) {
            unit_ptr->module().register_forward_post_hook(
                [unit_ptr](nn::Module* /*module*/, const Variable& /*input*/,
                          const Variable& /*output*/) {
                    // NN.18: skip the free when we're inside a summon window.
                    // The user has explicitly asked for full (unsharded) params
                    // for the duration of summon_full_params() / release_full_params();
                    // letting a normal forward post-hook tear them down mid-window
                    // turns the very next layer's forward into a sharded read.
                    if (unit_ptr->in_summon()) return;
                    unit_ptr->free_full_params();
                }
            );
        }
    }
}

// ============================================================================
// audit-9 JJ.4: state_dict / load_state_dict implementations
// ============================================================================

auto FSDPUnit::param_names() const -> std::vector<std::string> {
    // Mirror Module::own_parameters() iteration order so indices align with
    // param_shapes_ / param_numels_.  We can't query `original_params_` for
    // names directly (they're stored as Variable*); reconstruct from
    // module_.named_parameters() filtered to those present in
    // original_params_.
    std::vector<std::string> out;
    out.reserve(original_params_.size());
    auto named = const_cast<nn::Module&>(module_).named_parameters();
    // Build a Variable*-to-name map then iterate original_params_ in their
    // own order.
    std::unordered_map<const Variable*, std::string> by_ptr;
    for (auto& [name, p] : named) {
        if (p) by_ptr[p.get()] = name;
    }
    for (const auto& p : original_params_) {
        auto it = by_ptr.find(p.get());
        out.push_back(it != by_ptr.end() ? it->second : std::string{"<unknown>"});
    }
    return out;
}

auto FSDPUnit::copy_local_shard_from(const Tensor& src) -> void {
    if (src.numel() != local_shard_.numel()) {
        throw std::runtime_error(
            "FSDPUnit::copy_local_shard_from: shard numel mismatch (expected " +
            std::to_string(local_shard_.numel()) + ", got " +
            std::to_string(src.numel()) + ")");
    }
    if (src.dtype() != local_shard_.dtype()) {
        throw std::runtime_error(
            "FSDPUnit::copy_local_shard_from: dtype mismatch");
    }
    // Copy device-agnostically; src may live on CPU (typical checkpoint
    // restore) while the live shard is on GPU.
    Tensor src_on_device = (src.device() == local_shard_.device())
        ? src
        : src.to(local_shard_.device());
    // Tenzor's Tensor type doesn't expose ``copy_``; rebind the shard handle
    // to a contiguous copy of the source instead. local_shard_ retains the
    // same shape / dtype / device because src_on_device was already
    // validated against them above.
    local_shard_ = src_on_device.contiguous();
}

auto FullyShardedDataParallel::state_dict() const
    -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> out;

    const int ws = pg_->world_size();
    const int rank = pg_->rank();

    // Metadata: world_size, rank as scalar tensors so the dict is uniform-type.
    out["__world_size__"] = full({}, static_cast<double>(ws), DType::Int64, Device::cpu());
    out["__rank__"]       = full({}, static_cast<double>(rank), DType::Int64, Device::cpu());
    out["__num_units__"]  = full({}, static_cast<double>(units_.size()),
                                 DType::Int64, Device::cpu());

    for (size_t ui = 0; ui < units_.size(); ++ui) {
        const auto& unit = *units_[ui];
        const std::string up = "unit_" + std::to_string(ui) + "/";

        out[up + "__total_numel__"] =
            full({}, static_cast<double>(unit.total_numel()), DType::Int64, Device::cpu());
        out[up + "__shard_numel__"] =
            full({}, static_cast<double>(unit.shard_numel()), DType::Int64, Device::cpu());
        out[up + "__shard_offset__"] =
            full({}, static_cast<double>(unit.shard_offset()), DType::Int64, Device::cpu());

        // Per-parameter metadata: name, original numel, shape (as 1-D Int64).
        auto names = unit.param_names();
        const auto& shapes = unit.param_shapes();
        const auto& numels = unit.param_numels();
        for (size_t pi = 0; pi < names.size(); ++pi) {
            const std::string pp = up + "params/" + names[pi];
            out[pp + "/numel"] =
                full({}, static_cast<double>(numels[pi]), DType::Int64, Device::cpu());
            // Encode shape as a 1-D Int64 tensor.
            Tensor shape_t = zeros({static_cast<int64_t>(shapes[pi].size())},
                                   DType::Int64, Device::cpu());
            for (int64_t k = 0; k < static_cast<int64_t>(shapes[pi].size()); ++k) {
                shape_t.data<int64_t>()[k] = shapes[pi][k];
            }
            out[pp + "/shape"] = shape_t;
        }

        // The actual shard payload: this rank's slice of flat_param_.
        // Move to CPU for checkpoint portability.
        out[up + "shard"] = unit.local_shard().to(Device::cpu());
    }

    return out;
}

auto FullyShardedDataParallel::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state) -> void {
    auto get_or_throw = [&](const std::string& key) -> const Tensor& {
        auto it = state.find(key);
        if (it == state.end()) {
            throw std::runtime_error("FSDP::load_state_dict: missing key '" + key + "'");
        }
        return it->second;
    };

    // Validate world_size and rank.
    const int ws_now = pg_->world_size();
    const int rank_now = pg_->rank();
    int64_t ws_saved = get_or_throw("__world_size__").data<int64_t>()[0];
    int64_t rank_saved = get_or_throw("__rank__").data<int64_t>()[0];
    int64_t nu_saved = get_or_throw("__num_units__").data<int64_t>()[0];

    if (ws_saved != ws_now) {
        throw std::runtime_error(
            "FSDP::load_state_dict: world_size mismatch (saved " +
            std::to_string(ws_saved) + ", current " + std::to_string(ws_now) +
            ").  Cross-world-size resharding is not supported by this API; "
            "consult `summon_full_params` for a single-rank full-state save.");
    }
    if (rank_saved != rank_now) {
        throw std::runtime_error(
            "FSDP::load_state_dict: rank mismatch (saved " +
            std::to_string(rank_saved) + ", current " + std::to_string(rank_now) + ")");
    }
    if (static_cast<size_t>(nu_saved) != units_.size()) {
        throw std::runtime_error(
            "FSDP::load_state_dict: num_units mismatch (saved " +
            std::to_string(nu_saved) + ", current " + std::to_string(units_.size()) + ")");
    }

    for (size_t ui = 0; ui < units_.size(); ++ui) {
        auto& unit = *units_[ui];
        const std::string up = "unit_" + std::to_string(ui) + "/";

        int64_t total_saved = get_or_throw(up + "__total_numel__").data<int64_t>()[0];
        int64_t shard_saved = get_or_throw(up + "__shard_numel__").data<int64_t>()[0];
        if (static_cast<size_t>(total_saved) != unit.total_numel()) {
            throw std::runtime_error(
                "FSDP::load_state_dict: " + up +
                "total_numel mismatch (saved " + std::to_string(total_saved) +
                ", current " + std::to_string(unit.total_numel()) + ")");
        }
        if (static_cast<size_t>(shard_saved) != unit.shard_numel()) {
            throw std::runtime_error(
                "FSDP::load_state_dict: " + up +
                "shard_numel mismatch (saved " + std::to_string(shard_saved) +
                ", current " + std::to_string(unit.shard_numel()) + ")");
        }

        // Validate per-parameter shapes.
        auto names = unit.param_names();
        const auto& shapes_cur = unit.param_shapes();
        for (size_t pi = 0; pi < names.size(); ++pi) {
            const std::string pp = up + "params/" + names[pi];
            const Tensor& shape_saved = get_or_throw(pp + "/shape");
            if (shape_saved.numel() != static_cast<int64_t>(shapes_cur[pi].size())) {
                throw std::runtime_error(
                    "FSDP::load_state_dict: " + pp + " rank mismatch");
            }
            for (int64_t k = 0; k < shape_saved.numel(); ++k) {
                if (shape_saved.data<int64_t>()[k] != shapes_cur[pi][k]) {
                    throw std::runtime_error(
                        "FSDP::load_state_dict: " + pp + " shape[" +
                        std::to_string(k) + "] mismatch");
                }
            }
        }

        // Copy the shard payload into the live local shard.
        unit.copy_local_shard_from(get_or_throw(up + "shard"));
    }
}

} // namespace tenzor::distributed
