/**
 * @file dtensor.cpp
 * @brief DTensor implementation
 */

#include "tenzor/distributed/dtensor.hpp"
#include "tenzor/ops/transform.hpp"

#include <stdexcept>
#include <format>

namespace tenzor::distributed {

DTensor::DTensor(Tensor local_tensor,
                 std::shared_ptr<DeviceMesh> mesh,
                 std::vector<Placement> placements)
    : local_tensor_(std::move(local_tensor)),
      mesh_(std::move(mesh)),
      placements_(std::move(placements)) {

    if (static_cast<int64_t>(placements_.size()) != mesh_->ndim()) {
        throw std::invalid_argument(std::format(
            "DTensor: expected {} placements (one per mesh dimension), got {}",
            mesh_->ndim(), placements_.size()));
    }
}

auto DTensor::local_tensor() const -> const Tensor& {
    return local_tensor_;
}

auto DTensor::local_tensor() -> Tensor& {
    return local_tensor_;
}

auto DTensor::shape() const -> std::vector<int64_t> {
    auto local_shape = local_tensor_.shape();
    std::vector<int64_t> global_shape(local_shape.begin(), local_shape.end());

    for (int64_t mesh_dim = 0; mesh_dim < mesh_->ndim(); ++mesh_dim) {
        if (auto* shard = std::get_if<Shard>(&placements_[mesh_dim])) {
            auto tensor_dim = shard->dim;
            if (tensor_dim < 0 || tensor_dim >= static_cast<int64_t>(global_shape.size())) {
                throw std::out_of_range(std::format(
                    "DTensor::shape: Shard dim {} is out of range for tensor with {} dimensions",
                    tensor_dim, global_shape.size()));
            }
            global_shape[tensor_dim] *= mesh_->shape()[mesh_dim];
        }
        // Replicate and Partial don't change the global shape
    }

    return global_shape;
}

auto DTensor::mesh() const -> const DeviceMesh& {
    return *mesh_;
}

auto DTensor::placements() const -> const std::vector<Placement>& {
    return placements_;
}

auto DTensor::redistribute(const std::vector<Placement>& new_placements) -> DTensor {
    if (static_cast<int64_t>(new_placements.size()) != mesh_->ndim()) {
        throw std::invalid_argument(std::format(
            "DTensor::redistribute: expected {} placements, got {}",
            mesh_->ndim(), new_placements.size()));
    }

    Tensor current = local_tensor_;

    for (int64_t mesh_dim = 0; mesh_dim < mesh_->ndim(); ++mesh_dim) {
        const auto& old_p = placements_[mesh_dim];
        const auto& new_p = new_placements[mesh_dim];
        auto mesh_size = mesh_->shape()[mesh_dim];

        // Same placement — no-op
        if (old_p.index() == new_p.index()) {
            // Check if both are Shard with same dim
            if (auto* old_s = std::get_if<Shard>(&old_p)) {
                auto* new_s = std::get_if<Shard>(&new_p);
                if (old_s->dim == new_s->dim) {
                    continue;  // Same shard dim, no-op
                }
                // Shard(dim_a) -> Shard(dim_b): all-to-all needed
                throw std::runtime_error(std::format(
                    "DTensor::redistribute: Shard({}) -> Shard({}) requires "
                    "all-to-all, which is not yet implemented",
                    old_s->dim, new_s->dim));
            }
            continue;  // Replicate->Replicate or Partial->Partial: no-op
        }

        // Shard -> Replicate: all-gather
        if (std::holds_alternative<Shard>(old_p) &&
            std::holds_alternative<Replicate>(new_p)) {
            [[maybe_unused]] auto shard_dim = std::get<Shard>(old_p).dim;

            // In single-process mode (world_size=1), all-gather is a no-op:
            // the local tensor already contains all the data.
            // Multi-process: would call all_gather and concatenate along
            // shard_dim via the process group.
            continue;
        }

        // Replicate -> Shard: local slice (no communication)
        if (std::holds_alternative<Replicate>(old_p) &&
            std::holds_alternative<Shard>(new_p)) {
            auto shard_dim = std::get<Shard>(new_p).dim;
            auto dim_size = current.size(shard_dim);

            if (dim_size % mesh_size != 0) {
                throw std::runtime_error(std::format(
                    "DTensor::redistribute: cannot shard dimension {} of size {} "
                    "across {} devices evenly",
                    shard_dim, dim_size, mesh_size));
            }

            auto chunk_size = dim_size / mesh_size;
            // Determine our index along this mesh dimension
            auto my_coord = mesh_->get_coordinate(mesh_->get_local_rank());
            auto my_idx = my_coord[mesh_dim];

            current = current.narrow(shard_dim, my_idx * chunk_size, chunk_size);
            continue;
        }

        // Partial -> Replicate: all-reduce
        if (std::holds_alternative<Partial>(old_p) &&
            std::holds_alternative<Replicate>(new_p)) {
            // In single-process mode, all-reduce is a no-op:
            // the local tensor is already the full result
            if (mesh_size == 1) {
                continue;
            }

            // Multi-process: would call all_reduce with the appropriate op.
            // For now, single-process stub.
            continue;
        }

        // Partial -> Shard: reduce-scatter (not yet implemented)
        if (std::holds_alternative<Partial>(old_p) &&
            std::holds_alternative<Shard>(new_p)) {
            throw std::runtime_error(
                "DTensor::redistribute: Partial -> Shard (reduce-scatter) "
                "is not yet implemented");
        }

        // Shard -> Partial: not a valid transition
        if (std::holds_alternative<Shard>(old_p) &&
            std::holds_alternative<Partial>(new_p)) {
            throw std::runtime_error(
                "DTensor::redistribute: Shard -> Partial is not a valid "
                "placement transition");
        }

        // Replicate -> Partial: not a valid transition
        if (std::holds_alternative<Replicate>(old_p) &&
            std::holds_alternative<Partial>(new_p)) {
            throw std::runtime_error(
                "DTensor::redistribute: Replicate -> Partial is not a valid "
                "placement transition");
        }
    }

    return DTensor(std::move(current), mesh_, new_placements);
}

auto DTensor::full_tensor() const -> Tensor {
    // Build all-Replicate placements
    std::vector<Placement> all_replicate(mesh_->ndim(), Replicate{});

    // Create a mutable copy and redistribute
    DTensor mutable_copy(local_tensor_, mesh_, placements_);
    auto gathered = mutable_copy.redistribute(all_replicate);
    return gathered.local_tensor();
}

auto DTensor::from_global(const Tensor& tensor,
                           std::shared_ptr<DeviceMesh> mesh,
                           const std::vector<Placement>& placements) -> DTensor {
    if (static_cast<int64_t>(placements.size()) != mesh->ndim()) {
        throw std::invalid_argument(std::format(
            "DTensor::from_global: expected {} placements, got {}",
            mesh->ndim(), placements.size()));
    }

    Tensor local = tensor;

    for (int64_t mesh_dim = 0; mesh_dim < mesh->ndim(); ++mesh_dim) {
        if (auto* shard = std::get_if<Shard>(&placements[mesh_dim])) {
            auto shard_dim = shard->dim;
            auto mesh_size = mesh->shape()[mesh_dim];
            auto dim_size = local.size(shard_dim);

            if (dim_size % mesh_size != 0) {
                throw std::runtime_error(std::format(
                    "DTensor::from_global: cannot shard dimension {} of size {} "
                    "across {} devices evenly",
                    shard_dim, dim_size, mesh_size));
            }

            auto chunk_size = dim_size / mesh_size;
            auto my_coord = mesh->get_coordinate(mesh->get_local_rank());
            auto my_idx = my_coord[mesh_dim];

            local = local.narrow(shard_dim, my_idx * chunk_size, chunk_size);
        }
        // Replicate: keep the full tensor (no-op)
        // Partial: the caller is responsible for providing partial data
    }

    return DTensor(std::move(local), std::move(mesh), placements);
}

} // namespace tenzor::distributed
