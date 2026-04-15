/**
 * @file dtensor.hpp
 * @brief Distributed tensor with placement strategies
 *
 * Provides DTensor, a tensor distributed across a DeviceMesh with explicit
 * placement strategies (Shard, Replicate, Partial) per mesh dimension.
 * Supports redistribution between placements via collective communication.
 */

#pragma once

#include "../core/tensor.hpp"
#include "device_mesh.hpp"
#include <variant>
#include <memory>

namespace tenzor::distributed {

/**
 * @brief Shard placement: tensor is partitioned along a tensor dimension
 *        across the corresponding mesh dimension.
 */
struct Shard {
    int64_t dim;  ///< Tensor dimension along which to shard
    explicit Shard(int64_t d) : dim(d) {}
};

/**
 * @brief Replicate placement: tensor is fully replicated across the
 *        corresponding mesh dimension.
 */
struct Replicate {};

/**
 * @brief Reduction operations for Partial placement.
 */
enum class DTensorReduceOp { Sum, Mean, Max, Min };

/**
 * @brief Partial placement: each rank holds a partial value that must be
 *        reduced (e.g., summed) to produce the correct result.
 */
struct Partial {
    DTensorReduceOp op;
    explicit Partial(DTensorReduceOp o = DTensorReduceOp::Sum) : op(o) {}
};

/** @brief A placement strategy for one mesh dimension */
using Placement = std::variant<Shard, Replicate, Partial>;

/**
 * @brief Distributed tensor with explicit placement semantics.
 *
 * A DTensor wraps a local tensor shard together with a DeviceMesh and a
 * vector of Placement strategies — one per mesh dimension. The placements
 * describe how the global tensor is distributed:
 *
 *   - Shard(dim): the tensor is split along tensor dimension `dim` across
 *     that mesh dimension.
 *   - Replicate: the tensor is fully replicated on every rank in that
 *     mesh dimension.
 *   - Partial(op): each rank holds a partial contribution that requires
 *     reduction with `op` to produce the correct global tensor.
 *
 * Example:
 * @code
 * auto mesh = std::make_shared<DeviceMesh>(
 *     Device::Type::CUDA, {2, 4}, {"dp", "tp"});
 *
 * // Global tensor [8, 1024] sharded along dim 1 across "tp",
 * // replicated across "dp"
 * auto dt = DTensor::from_global(
 *     global_tensor, mesh, {Replicate{}, Shard{1}});
 *
 * // Redistribute to all-replicate to gather the full tensor
 * auto gathered = dt.redistribute({Replicate{}, Replicate{}});
 * @endcode
 */
class DTensor {
public:
    /**
     * @brief Construct a DTensor from a local tensor shard.
     *
     * @param local_tensor The local tensor shard on this rank
     * @param mesh The device mesh
     * @param placements Placement strategy per mesh dimension
     * @throws std::invalid_argument if placements.size() != mesh.ndim()
     */
    DTensor(Tensor local_tensor,
            std::shared_ptr<DeviceMesh> mesh,
            std::vector<Placement> placements);

    /** @brief Get the local tensor shard on this rank (const) */
    auto local_tensor() const -> const Tensor&;

    /** @brief Get the local tensor shard on this rank (mutable) */
    auto local_tensor() -> Tensor&;

    /**
     * @brief Compute the full (global) tensor shape.
     *
     * For each Shard(dim) placement, the global shape along that tensor
     * dimension is local_shape[dim] * mesh_size_along_that_mesh_dim.
     *
     * @return Global tensor shape
     */
    auto shape() const -> std::vector<int64_t>;

    /** @brief Get the device mesh */
    auto mesh() const -> const DeviceMesh&;

    /** @brief Get the placement strategies */
    auto placements() const -> const std::vector<Placement>&;

    /**
     * @brief Redistribute to new placements.
     *
     * Triggers collective communication as needed:
     *   - Shard -> Replicate: all-gather
     *   - Replicate -> Shard: local slice (no communication)
     *   - Partial -> Replicate: all-reduce
     *   - Shard(dim_a) -> Shard(dim_b): all-to-all (not yet implemented)
     *
     * @param new_placements Target placements (one per mesh dimension)
     * @return DTensor with the new placements
     * @throws std::runtime_error for unsupported placement transitions
     */
    auto redistribute(const std::vector<Placement>& new_placements) -> DTensor;

    /**
     * @brief Gather all shards and return the full global tensor.
     *
     * Equivalent to redistributing to all-Replicate placements and
     * returning the local tensor.
     *
     * @return The full (un-sharded) tensor
     */
    auto full_tensor() const -> Tensor;

    /**
     * @brief Distribute a global tensor across a mesh.
     *
     * Creates a DTensor by sharding or replicating the global tensor
     * according to the given placements.
     *
     * @param tensor The full global tensor
     * @param mesh The device mesh
     * @param placements Placement strategy per mesh dimension
     * @return DTensor with local shards on each rank
     */
    static auto from_global(const Tensor& tensor,
                            std::shared_ptr<DeviceMesh> mesh,
                            const std::vector<Placement>& placements) -> DTensor;

private:
    Tensor local_tensor_;
    std::shared_ptr<DeviceMesh> mesh_;
    std::vector<Placement> placements_;
};

} // namespace tenzor::distributed
