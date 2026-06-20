/**
 * @file fsdp2.cpp
 * @brief Implementation of FSDP2 (per-parameter DTensor sharding)
 *
 * Uses DTensor placements to shard each parameter independently along the
 * configured mesh dimension. All-gather reconstructs full parameters for
 * computation; reduce-scatter distributes gradients back to shards.
 *
 * In single-process mode (world_size == 1 on the shard dimension), all
 * operations are no-ops that pass tensors through unchanged.
 */

#include "tenzor/distributed/fsdp2.hpp"
#include "tenzor/autograd/checkpoint.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace tenzor::distributed {

// ============================================================================
// FSDP2 Implementation
// ============================================================================

FSDP2::FSDP2(std::shared_ptr<nn::Module> module, FSDP2Config config)
    : module_(std::move(module)), config_(std::move(config)) {

    if (!module_) {
        throw std::invalid_argument("FSDP2: module must not be null");
    }

    if (!config_.mesh) {
        throw std::invalid_argument("FSDP2: config.mesh must not be null");
    }

    // Validate that the shard mesh dimension exists
    // get_dim throws std::invalid_argument if not found
    auto dim_idx = config_.mesh->get_dim(config_.shard_mesh_dim);

    shard_world_size_ = config_.mesh->shape()[dim_idx];

    // Determine this rank's position along the shard dimension
    auto local_rank = config_.mesh->get_local_rank();
    auto coord = config_.mesh->get_coordinate(local_rank);
    shard_rank_ = coord[dim_idx];
}

FSDP2::~FSDP2() = default;

auto FSDP2::shard_parameters() -> void {
    // HH.15: previously this early-returned when is_sharded_ was already true,
    // which meant parameters added mid-training via add_param_group never got
    // a DTensor wrapper and silently fell out of all-gather / reduce-scatter.
    // Replace with a diff loop that appends any named_parameter not already
    // present in sharded_params_. Existing slots stay put (EE.13 invariant —
    // saved-for-backward activations share their TensorImpl).
    auto named_params = module_->named_parameters();
    if (!is_sharded_) {
        sharded_params_.clear();
        sharded_params_.reserve(named_params.size());
    }

    // Build a lookup over already-sharded names so the diff is O(N) over
    // current params instead of O(N*M) string compares.
    std::unordered_set<std::string> already_sharded;
    already_sharded.reserve(sharded_params_.size());
    for (auto& [n, _dt] : sharded_params_) already_sharded.insert(n);

    for (auto& [name, param] : named_params) {
        if (!param) {
            continue;
        }
        if (already_sharded.contains(name)) {
            continue;  // Preserve existing DTensor slot.
        }

        auto& tensor = param->tensor();

        if (shard_world_size_ <= 1) {
            // Single-process mode: wrap as DTensor with Replicate placement
            // but no actual communication needed
            std::vector<Placement> placements;
            for (int64_t d = 0; d < config_.mesh->ndim(); ++d) {
                placements.emplace_back(Replicate{});
            }
            DTensor dt(tensor, config_.mesh, std::move(placements));
            sharded_params_.emplace_back(name, std::move(dt));
        } else {
            // Multi-process: shard along dim 0 on the dp mesh dimension
            // Build placements: Shard(0) for the shard dim, Replicate for others
            std::vector<Placement> placements;
            auto shard_dim_idx = config_.mesh->get_dim(config_.shard_mesh_dim);

            for (int64_t d = 0; d < config_.mesh->ndim(); ++d) {
                if (d == shard_dim_idx) {
                    placements.emplace_back(Shard{0});
                } else {
                    placements.emplace_back(Replicate{});
                }
            }

            // Create DTensor from the full parameter, which internally
            // slices to keep only this rank's shard
            auto dt = DTensor::from_global(tensor, config_.mesh, placements);

            // Cast to mixed precision param dtype if needed
            if (config_.mixed_precision.param_dtype != tensor.dtype()) {
                auto& local = dt.local_tensor();
                local = maybe_cast(local, config_.mixed_precision.param_dtype);
            }

            sharded_params_.emplace_back(name, std::move(dt));
        }
    }

    // BB.15: do NOT pre-allocate the persistent unsharded destinations
    // here. Y.22 reserved a full_shape buffer per parameter at
    // shard_parameters() time to force the in-place zero_/add_ path on
    // cycle 0, but the pre-allocation doubles memory permanently — the
    // shard plus the unsharded slot are both alive until the next
    // shard_parameters() call. unshard_params() already handles cycle 0
    // via the slot_valid=false branch (it adopts the freshly allgathered
    // tensor's TensorImpl); subsequent cycles reuse that slot via the
    // in-place add path. The R.18 invariant (saved-for-backward activations
    // share the param's TensorImpl) holds from cycle 1 onward, and cycle 0
    // can't have prior activations to orphan because forward has not run
    // yet at shard time.
    //
    // EE.13: do NOT clear unsharded_dst_/sharded_dst_ here. shard_parameters()
    // may be called mid-training (e.g. after add_param_group) and the
    // persistent slots from prior cycles already have stable TensorImpls
    // captured by saved-for-backward activations. Clearing them forces the
    // next unshard_params() into the !slot_valid adoption branch, which
    // swaps param->tensor() to a new TensorImpl and orphans those captures.
    // unshard_params() itself handles dtype/shape transitions (e.g. mixed
    // precision reconfig) via the slot_valid check and falls through to
    // the adoption path only when the existing slot is genuinely incompatible.

    is_sharded_ = true;
}

auto FSDP2::forward(const Variable& input) -> Variable {
    // Unshard parameters for forward computation
    unshard_params();

    // Run the module's forward pass with full parameters. With activation_checkpointing,
    // wrap in autograd::checkpoint so intermediates between input and output are dropped
    // after forward and recomputed on backward — trades ~33% backward compute for typically
    // 4-8× activation-memory reduction on transformer blocks.
    Variable output;
    if (config_.activation_checkpointing) {
        nn::Module* mod = module_.get();
        output = autograd::checkpoint(
            [mod](const Variable& x) -> Variable { return mod->forward(x); },
            input
        );
    } else {
        output = module_->forward(input);
    }

    // Optionally reshard to free memory
    if (config_.reshard_after_forward && shard_world_size_ > 1) {
        reshard_params();
    }

    return output;
}

auto FSDP2::backward_hook() -> void {
    if (shard_world_size_ <= 1) {
        return;  // Single-process: nothing to reduce
    }

    // Ensure params are unsharded (needed for gradient computation)
    // In the typical case with reshard_after_forward=true, backward will
    // have already re-unsharded via autograd hooks. But we make sure here.
    if (!params_unsharded_) {
        unshard_params();
    }

    auto named_params = module_->named_parameters();

    for (auto& [name, param] : named_params) {
        if (!param || !param->requires_grad()) {
            continue;
        }

        auto& grad_opt = param->mutable_grad();
        if (!grad_opt.has_value() || grad_opt->numel() == 0) {
            continue;
        }
        auto& grad = grad_opt.value();

        // Cast gradient to reduction dtype if needed
        auto reduce_grad = maybe_cast(grad, config_.mixed_precision.reduce_dtype);

        // S.16: Use Partial{Mean} so the Partial -> Shard redistribute
        // averages across the shard group internally (all_reduce(AVG) +
        // narrow). The previous implementation used Partial{Sum} (native
        // reduce_scatter) and then divided the local shard by
        // shard_world_size_ — semantically equivalent on a SUM reduce-
        // scatter today, but a quiet double-average if a future backend
        // ever wires reduce_scatter to honour the placement's reduce op
        // directly. Picking Mean here keeps the averaging concern inside
        // the placement layer instead of duplicating it on the caller.
        auto shard_dim_idx = config_.mesh->get_dim(config_.shard_mesh_dim);
        std::vector<Placement> partial_placements;
        for (int64_t d = 0; d < config_.mesh->ndim(); ++d) {
            if (d == shard_dim_idx) {
                partial_placements.emplace_back(Partial{DTensorReduceOp::Mean});
            } else {
                partial_placements.emplace_back(Replicate{});
            }
        }

        DTensor grad_dt(reduce_grad, config_.mesh, std::move(partial_placements));

        // Redistribute: Partial(Mean) -> Shard(0). For Mean the redistribute
        // takes the all_reduce(AVG)+narrow path which averages across the
        // shard group internally — no further divide needed.
        std::vector<Placement> shard_placements;
        for (int64_t d = 0; d < config_.mesh->ndim(); ++d) {
            if (d == shard_dim_idx) {
                shard_placements.emplace_back(Shard{0});
            } else {
                shard_placements.emplace_back(Replicate{});
            }
        }

        auto sharded_grad = grad_dt.redistribute(shard_placements);
        auto local_grad = sharded_grad.local_tensor();

        // Write back the sharded gradient
        param->set_grad(maybe_cast(local_grad, param->tensor().dtype()));
    }

    // Reshard parameters after backward
    reshard_params();
}

auto FSDP2::sharded_parameters() -> std::vector<DTensor> {
    std::vector<DTensor> result;
    result.reserve(sharded_params_.size());
    for (auto& [name, dt] : sharded_params_) {
        result.push_back(dt);
    }
    return result;
}

auto FSDP2::summon_full_params() -> void {
    unshard_params();
}

auto FSDP2::release_full_params() -> void {
    reshard_params();
}

auto FSDP2::unshard_params() -> void {
    if (params_unsharded_ || shard_world_size_ <= 1) {
        params_unsharded_ = true;
        return;
    }

    // Build a name -> Variable index once (O(P)) instead of rescanning
    // named_parameters() for every sharded param (which was O(P^2) per call).
    auto named_params = module_->named_parameters();
    std::unordered_map<std::string, std::shared_ptr<Variable>> param_index;
    param_index.reserve(named_params.size());
    for (auto& [pname, p] : named_params) {
        param_index.emplace(pname, p);
    }

    for (auto& [name, dt] : sharded_params_) {
        // Redistribute from Shard(0) -> Replicate (all-gather)
        auto full = dt.full_tensor();

        // Find and update the corresponding module parameter (O(1)).
        auto pit = param_index.find(name);
        if (pit != param_index.end() && pit->second) {
            auto& param = pit->second;
            {
                // R.18 / V.22: preserve param's TensorImpl/Storage so any
                // saved-for-backward activation that captured the parameter
                // sees the updated bytes rather than a stale frozen view.
                //
                // V.22: we now keep two persistent destinations per
                // parameter — an "unsharded" full-sized slot and a "sharded"
                // local-sized slot.  Each cycle reuses whichever slot
                // matches the requested layout, so the in-place copy path
                // runs every cycle (previously only on the first one,
                // because round 1's dst was shard-sized after reshard and
                // round 2's unshard would fall through to a TensorImpl
                // swap).
                std::vector<int64_t> full_shape(full.shape().begin(),
                                                full.shape().end());

                auto slot_it = unsharded_dst_.find(name);
                bool slot_valid = (slot_it != unsharded_dst_.end()) &&
                                  slot_it->second.is_valid() &&
                                  slot_it->second.dtype() == full.dtype() &&
                                  slot_it->second.device() == full.device() &&
                                  slot_it->second.numel() == full.numel() &&
                                  std::vector<int64_t>(slot_it->second.shape().begin(),
                                                       slot_it->second.shape().end()) == full_shape;

                if (!slot_valid) {
                    // First unshard for this parameter (or layout/dtype change
                    // due to mixed-precision reconfiguration).  Take ownership
                    // of the freshly-allgathered tensor; subsequent cycles
                    // will copy into this slot in-place.
                    //
                    // Note: slot_valid already encodes the full predicate set
                    // (exists && valid && dtype/device/numel/shape match), so a
                    // separate "shape_compatible" in-place-reuse branch here
                    // would be unreachable — when !slot_valid, the slot bytes
                    // can never be reused.  full() is freshly all-gathered
                    // storage, independent of any DTensor internal buffer, so
                    // adopting it directly does not alias.
                    unsharded_dst_[name] = full;
                } else {
                    Tensor& slot = slot_it->second;
                    slot.zero_();
                    add_(slot, full);
                }

                // Point the live parameter at the unsharded slot.  The slot's
                // TensorImpl is stable across cycles, so activations that
                // captured the unsharded layout remain valid through the
                // next forward/backward.
                param->tensor() = unsharded_dst_[name];
            }
        }
    }

    params_unsharded_ = true;
}

auto FSDP2::reshard_params() -> void {
    if (!params_unsharded_ || shard_world_size_ <= 1) {
        return;
    }

    // Build a name -> Variable index once (O(P)) instead of rescanning
    // named_parameters() for every sharded param (previously O(P^2) per call).
    auto named_params = module_->named_parameters();
    std::unordered_map<std::string, std::shared_ptr<Variable>> param_index;
    param_index.reserve(named_params.size());
    for (auto& [pname, p] : named_params) {
        param_index.emplace(pname, p);
    }

    for (auto& [name, dt] : sharded_params_) {
        // Write the local shard back to the module parameter (O(1) lookup).
        auto pit = param_index.find(name);
        if (pit != param_index.end() && pit->second) {
            auto& param = pit->second;
            {
                // R.18 / V.22: mirror unshard_params -- write into a
                // persistent sharded-sized slot so the in-place copy path
                // is reachable on every cycle, not just the first one.
                auto local = dt.local_tensor();
                std::vector<int64_t> local_shape(local.shape().begin(),
                                                 local.shape().end());

                auto slot_it = sharded_dst_.find(name);
                bool slot_valid = (slot_it != sharded_dst_.end()) &&
                                  slot_it->second.is_valid() &&
                                  slot_it->second.dtype() == local.dtype() &&
                                  slot_it->second.device() == local.device() &&
                                  slot_it->second.numel() == local.numel() &&
                                  std::vector<int64_t>(slot_it->second.shape().begin(),
                                                       slot_it->second.shape().end()) == local_shape;

                if (!slot_valid) {
                    // local is a shallow copy of dt's internal local_tensor_
                    // (DTensor::local_tensor() returns a const Tensor& sharing
                    // the same TensorImpl/Storage).  Nothing rebinds
                    // local_tensor_ across cycles, so adopting `local` directly
                    // would make the persistent slot alias dt.local_tensor_.
                    // On the next reshard the in-place `slot.zero_(); add_(slot,
                    // local)` would then operate on a single self-aliased
                    // buffer (the dispatch alias guard does not fire for
                    // same-impl references), zeroing the parameter shard.
                    // clone() gives the slot independent storage so subsequent
                    // in-place copies are well-defined.
                    sharded_dst_[name] = local.clone();
                } else {
                    Tensor& slot = slot_it->second;
                    slot.zero_();
                    add_(slot, local);
                }

                param->tensor() = sharded_dst_[name];
            }
        }
    }

    params_unsharded_ = false;
}

auto FSDP2::maybe_cast(const Tensor& tensor, DType target) -> Tensor {
    if (tensor.dtype() == target) {
        return tensor;
    }
    return tensor.to(target);
}

} // namespace tenzor::distributed
