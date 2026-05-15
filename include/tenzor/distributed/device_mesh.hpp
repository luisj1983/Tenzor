/**
 * @file device_mesh.hpp
 * @brief N-dimensional device mesh for multi-dimensional parallelism
 *
 * Provides DeviceMesh, a logical n-dimensional grid of devices with named
 * dimensions. Used to express hybrid parallelism strategies (e.g., data
 * parallel x tensor parallel) by assigning a parallelism type to each
 * mesh dimension.
 */

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <optional>
#include "../core/tensor.hpp"

namespace tenzor::distributed {

// Forward declaration
class ProcessGroup;
class ProcessGroupBase;

/**
 * @brief N-dimensional grid of devices with named dimensions.
 *
 * DeviceMesh represents a logical arrangement of devices into an
 * n-dimensional grid where each dimension can correspond to a different
 * parallelism strategy (data parallel, tensor parallel, pipeline parallel,
 * etc.).
 *
 * Example:
 * @code
 * // 2x4 mesh: 2-way data parallel, 4-way tensor parallel
 * DeviceMesh mesh(Device::Type::CUDA, {2, 4}, {"dp", "tp"});
 *
 * // Get the tensor-parallel group for the current rank
 * auto tp_ranks = mesh.get_submesh("tp");
 * @endcode
 */
class DeviceMesh {
public:
    /**
     * @brief Construct a device mesh.
     *
     * @param device_type Type of devices in the mesh
     * @param mesh_shape Shape of the n-dimensional grid (e.g., {2, 4})
     * @param mesh_dim_names Names for each dimension (e.g., {"dp", "tp"})
     * @throws std::invalid_argument if mesh_shape and mesh_dim_names have
     *         different sizes, or if dimension names are not unique
     */
    DeviceMesh(Device::Type device_type,
               std::vector<int64_t> mesh_shape,
               std::vector<std::string> mesh_dim_names);

    /**
     * @brief Construct a device mesh with an explicit mesh rank.
     *
     * Used for tests and for sub-mesh constructions where the mesh rank is
     * not derivable from the global RANK env var. The mesh rank must be in
     * the half-open range [0, prod(mesh_shape)).
     *
     * @param mesh_rank This process's position in the mesh (0..size()-1)
     * @throws std::out_of_range if mesh_rank is outside [0, size())
     */
    DeviceMesh(Device::Type device_type,
               std::vector<int64_t> mesh_shape,
               std::vector<std::string> mesh_dim_names,
               int64_t mesh_rank);

    /** @brief Get the shape of the mesh */
    auto shape() const -> const std::vector<int64_t>&;

    /** @brief Get the dimension names */
    auto dim_names() const -> const std::vector<std::string>&;

    /** @brief Get the number of dimensions */
    auto ndim() const -> int64_t;

    /** @brief Get the total number of devices in the mesh */
    auto size() const -> int64_t;

    /** @brief Get the device type */
    auto device_type() const -> Device::Type;

    /**
     * @brief Convert multi-dimensional coordinates to a flat device ID.
     *
     * Uses row-major (C-order) indexing: the last dimension varies fastest.
     *
     * @param coord Multi-dimensional coordinate
     * @return Flat device ID
     * @throws std::out_of_range if coordinates are out of bounds
     */
    auto get_device_id(const std::vector<int64_t>& coord) const -> int64_t;

    /**
     * @brief Convert a flat device ID to multi-dimensional coordinates.
     *
     * Inverse of get_device_id().
     *
     * @param flat_id Flat device ID
     * @return Multi-dimensional coordinate
     * @throws std::out_of_range if flat_id >= size()
     */
    auto get_coordinate(int64_t flat_id) const -> std::vector<int64_t>;

    /**
     * @brief Get the dimension index for a named dimension.
     *
     * @param name Dimension name
     * @return Index of the dimension
     * @throws std::invalid_argument if name is not found
     */
    auto get_dim(const std::string& name) const -> int64_t;

    /**
     * @brief Get the submesh (group of ranks) along a named dimension.
     *
     * Returns all flat device IDs that share the same coordinates as the
     * current rank on every dimension except the named one. This gives the
     * set of ranks that form a communication group for that parallelism type.
     *
     * Example: for a 2x4 mesh with local rank at [1, 2]:
     *   get_submesh("tp") returns ranks at [1,0], [1,1], [1,2], [1,3]
     *   get_submesh("dp") returns ranks at [0,2], [1,2]
     *
     * @param dim_name Name of the dimension to vary
     * @return Vector of flat device IDs in the submesh
     */
    auto get_submesh(const std::string& dim_name) const -> std::vector<int64_t>;

    /**
     * @brief Get the current rank's flat device ID in the mesh.
     *
     * This is the "mesh rank" — this process's position in the mesh's
     * row-major flat enumeration. In the common case of a single mesh that
     * spans all distributed processes, this equals the global rank
     * (i.e. the `RANK` env var). Use get_host_local_rank() to obtain the
     * conventional per-host LOCAL_RANK (for GPU device binding).
     */
    auto get_local_rank() const -> int64_t;

    /** @brief Alias for get_local_rank() — preferred for new code. */
    auto get_mesh_rank() const -> int64_t { return get_local_rank(); }

    /**
     * @brief This process's per-host local rank (the `LOCAL_RANK` env var).
     *
     * Used to bind to a local GPU device (`cudaSetDevice(local_rank)`).
     * Distinct from get_mesh_rank() which is the process's position
     * inside the mesh enumeration.
     */
    static auto get_host_local_rank() -> int64_t;

    /**
     * @brief Attach a process group to be used by mesh-aware ops (DTensor).
     *
     * The PG is consulted by collective-using consumers like
     * `DTensor::redistribute`. Pass `nullptr` to detach.
     *
     * Single-mesh setups typically pass a freshly constructed
     * `NCCLProcessGroup` / `GlooProcessGroup`.
     */
    auto set_process_group(std::shared_ptr<ProcessGroupBase> pg) -> void;

    /**
     * @brief Get the process group bound to this mesh.
     *
     * Returns the most recently `set_process_group`'d PG, or nullptr if
     * none has been attached. Callers should null-check before invoking
     * collectives — a mesh with no PG is single-process.
     */
    auto process_group() const -> std::shared_ptr<ProcessGroupBase>;

    /**
     * @brief Get the process group for a specific named mesh dimension.
     *
     * Logically, the per-dim process group communicates between processes
     * that share all *other* mesh coordinates with this rank. Used by
     * tensor-parallel / sequence-parallel layers that need to issue
     * collectives along a single axis of the mesh.
     *
     * @note When the mesh has only one dimension or only one PG has been
     *       attached, this returns the same PG as `process_group()`.
     *       Per-dim sub-groups (true axis-local PGs) require
     *       `ProcessGroupBase::split(ranks)`, which is tracked as A3-extended.
     *
     * @param dim_name Name of the mesh dimension.
     */
    auto process_group_for_dim(const std::string& dim_name) const
        -> std::shared_ptr<ProcessGroupBase>;

private:
    Device::Type device_type_;
    std::vector<int64_t> mesh_shape_;
    std::vector<std::string> mesh_dim_names_;
    std::unordered_map<std::string, int64_t> dim_name_to_idx_;
    int64_t local_rank_ = 0;
    std::shared_ptr<ProcessGroupBase> pg_;
    // Lazy per-dim sub-PG cache (A3-extended). Populated on first
    // `process_group_for_dim(name)` call; the call is collective so every
    // rank must invoke it together.
    mutable std::unordered_map<std::string, std::shared_ptr<ProcessGroupBase>>
        sub_pg_cache_;
};

} // namespace tenzor::distributed
