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
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include <algorithm>
#include <stdexcept>

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
    if (is_sharded_) {
        return;  // Already sharded
    }

    auto named_params = module_->named_parameters();
    sharded_params_.clear();
    sharded_params_.reserve(named_params.size());

    for (auto& [name, param] : named_params) {
        if (!param) {
            continue;
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

    is_sharded_ = true;
}

auto FSDP2::forward(const Variable& input) -> Variable {
    // Unshard parameters for forward computation
    unshard_params();

    // Run the module's forward pass with full parameters
    auto output = module_->forward(input);

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

        // Reduce-scatter the gradient: sum across shard group, each rank
        // gets its shard of the averaged gradient.
        // Build a DTensor with Partial(Sum) placement, then redistribute
        // to Shard(0) to get the local gradient shard.
        auto shard_dim_idx = config_.mesh->get_dim(config_.shard_mesh_dim);
        std::vector<Placement> partial_placements;
        for (int64_t d = 0; d < config_.mesh->ndim(); ++d) {
            if (d == shard_dim_idx) {
                partial_placements.emplace_back(Partial{DTensorReduceOp::Sum});
            } else {
                partial_placements.emplace_back(Replicate{});
            }
        }

        DTensor grad_dt(reduce_grad, config_.mesh, std::move(partial_placements));

        // Redistribute: Partial -> Shard(0) is a reduce-scatter
        std::vector<Placement> shard_placements;
        for (int64_t d = 0; d < config_.mesh->ndim(); ++d) {
            if (d == shard_dim_idx) {
                shard_placements.emplace_back(Shard{0});
            } else {
                shard_placements.emplace_back(Replicate{});
            }
        }

        auto sharded_grad = grad_dt.redistribute(shard_placements);

        // Divide by world size to average
        auto local_grad = sharded_grad.local_tensor();
        local_grad = local_grad / static_cast<float>(shard_world_size_);

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

    auto named_params = module_->named_parameters();

    for (auto& [name, dt] : sharded_params_) {
        // Redistribute from Shard(0) -> Replicate (all-gather)
        auto full = dt.full_tensor();

        // Find and update the corresponding module parameter
        for (auto& [pname, param] : named_params) {
            if (pname == name && param) {
                param->tensor() = full;
                break;
            }
        }
    }

    params_unsharded_ = true;
}

auto FSDP2::reshard_params() -> void {
    if (!params_unsharded_ || shard_world_size_ <= 1) {
        return;
    }

    auto named_params = module_->named_parameters();

    for (auto& [name, dt] : sharded_params_) {
        // Write the local shard back to the module parameter
        for (auto& [pname, param] : named_params) {
            if (pname == name && param) {
                param->tensor() = dt.local_tensor();
                break;
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
