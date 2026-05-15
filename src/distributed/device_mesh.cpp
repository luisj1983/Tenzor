/**
 * @file device_mesh.cpp
 * @brief DeviceMesh implementation
 */

#include "tenzor/distributed/device_mesh.hpp"
#include "tenzor/distributed/process_group.hpp"

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <format>
#include <unordered_set>

namespace tenzor::distributed {

namespace {

// Read an integer env var, returning std::nullopt if unset or unparseable.
auto read_env_int(const char* name) -> std::optional<int64_t> {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return std::nullopt;
    char* end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end == v || *end != '\0') return std::nullopt;
    return static_cast<int64_t>(parsed);
}

// Initialize common fields shared by both constructors.
auto validate_and_init(std::vector<int64_t>& mesh_shape,
                       std::vector<std::string>& mesh_dim_names,
                       std::unordered_map<std::string, int64_t>& dim_name_to_idx) -> int64_t {
    if (mesh_shape.size() != mesh_dim_names.size()) {
        throw std::invalid_argument(std::format(
            "DeviceMesh: mesh_shape has {} dimensions but {} names were provided",
            mesh_shape.size(), mesh_dim_names.size()));
    }

    if (mesh_shape.empty()) {
        throw std::invalid_argument("DeviceMesh: mesh_shape must not be empty");
    }

    int64_t size = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(mesh_shape.size()); ++i) {
        if (mesh_shape[i] <= 0) {
            throw std::invalid_argument(std::format(
                "DeviceMesh: mesh_shape[{}] = {} must be positive",
                i, mesh_shape[i]));
        }
        size *= mesh_shape[i];
    }

    std::unordered_set<std::string> seen;
    for (int64_t i = 0; i < static_cast<int64_t>(mesh_dim_names.size()); ++i) {
        if (!seen.insert(mesh_dim_names[i]).second) {
            throw std::invalid_argument(std::format(
                "DeviceMesh: duplicate dimension name '{}'",
                mesh_dim_names[i]));
        }
        dim_name_to_idx[mesh_dim_names[i]] = i;
    }
    return size;
}

} // namespace

DeviceMesh::DeviceMesh(Device::Type device_type,
                       std::vector<int64_t> mesh_shape,
                       std::vector<std::string> mesh_dim_names)
    : device_type_(device_type),
      mesh_shape_(std::move(mesh_shape)),
      mesh_dim_names_(std::move(mesh_dim_names)) {

    auto size = validate_and_init(mesh_shape_, mesh_dim_names_, dim_name_to_idx_);

    // Resolve the mesh rank from the environment. Prefer RANK (the global
    // rank under PyTorch / torchrun convention) since DeviceMesh's mesh-rank
    // matches the global rank when one mesh spans all processes. Fall back
    // to LOCAL_RANK for single-host setups, then to 0 (single-process).
    int64_t resolved = 0;
    if (auto r = read_env_int("RANK"); r.has_value()) {
        resolved = *r;
    } else if (auto lr = read_env_int("LOCAL_RANK"); lr.has_value()) {
        resolved = *lr;
    }

    if (resolved < 0 || resolved >= size) {
        // Out-of-range env values are silently clamped to 0 rather than
        // throwing: many tests construct meshes without a full distributed
        // env, and a hard throw here would be surprising. Multi-process
        // launches always set RANK consistent with WORLD_SIZE.
        resolved = 0;
    }
    local_rank_ = resolved;
}

DeviceMesh::DeviceMesh(Device::Type device_type,
                       std::vector<int64_t> mesh_shape,
                       std::vector<std::string> mesh_dim_names,
                       int64_t mesh_rank)
    : device_type_(device_type),
      mesh_shape_(std::move(mesh_shape)),
      mesh_dim_names_(std::move(mesh_dim_names)) {

    auto size = validate_and_init(mesh_shape_, mesh_dim_names_, dim_name_to_idx_);

    if (mesh_rank < 0 || mesh_rank >= size) {
        throw std::out_of_range(std::format(
            "DeviceMesh: mesh_rank {} is outside the half-open range [0, {})",
            mesh_rank, size));
    }
    local_rank_ = mesh_rank;
}

auto DeviceMesh::set_process_group(std::shared_ptr<ProcessGroupBase> pg) -> void {
    pg_ = std::move(pg);
}

auto DeviceMesh::process_group() const -> std::shared_ptr<ProcessGroupBase> {
    return pg_;
}

auto DeviceMesh::process_group_for_dim(const std::string& dim_name) const
    -> std::shared_ptr<ProcessGroupBase> {
    const int64_t dim_idx = get_dim(dim_name);

    // Single-axis fast path: the only axis IS the whole mesh.
    if (ndim() == 1) {
        return pg_;
    }
    // If the parent PG has world_size 1, every sub-PG is trivially the
    // single rank's own group — no split needed.
    if (pg_ && pg_->world_size() == 1) {
        return pg_;
    }
    // If the named axis spans the entire mesh world, the sub-PG equals the
    // parent PG (every other axis has size 1, so nothing to split out).
    if (pg_ && mesh_shape_[dim_idx] == pg_->world_size()) {
        return pg_;
    }
    if (pg_ == nullptr) {
        return nullptr;
    }

    // Lazy cache.
    if (auto it = sub_pg_cache_.find(dim_name); it != sub_pg_cache_.end()) {
        return it->second;
    }

    // Compute the color/key pair that places this rank into the sub-PG that
    // contains the axis-local submesh. Ranks that share all coordinates on
    // axes OTHER than `dim_idx` are in the same submesh — encode those
    // other-axis coordinates as a single integer (row-major).
    const auto my_coord = get_coordinate(local_rank_);
    int color = 0;
    for (int64_t d = 0; d < ndim(); ++d) {
        if (d == dim_idx) continue;
        color = color * static_cast<int>(mesh_shape_[d]) +
                static_cast<int>(my_coord[d]);
    }
    const int key = static_cast<int>(my_coord[dim_idx]);

    // ncclCommSplit is collective — every rank in the parent PG must invoke
    // this together. DeviceMesh users follow the convention that mesh
    // construction + first sub-PG access happen in lockstep across all
    // ranks (matches PyTorch's DeviceMesh semantics).
    auto sub = pg_->split(color, key);
    sub_pg_cache_.emplace(dim_name, sub);
    return sub;
}

auto DeviceMesh::get_host_local_rank() -> int64_t {
    if (auto lr = read_env_int("LOCAL_RANK"); lr.has_value() && *lr >= 0) {
        return *lr;
    }
    return 0;
}

auto DeviceMesh::shape() const -> const std::vector<int64_t>& {
    return mesh_shape_;
}

auto DeviceMesh::dim_names() const -> const std::vector<std::string>& {
    return mesh_dim_names_;
}

auto DeviceMesh::ndim() const -> int64_t {
    return static_cast<int64_t>(mesh_shape_.size());
}

auto DeviceMesh::size() const -> int64_t {
    return std::accumulate(mesh_shape_.begin(), mesh_shape_.end(),
                           int64_t{1}, std::multiplies<>{});
}

auto DeviceMesh::device_type() const -> Device::Type {
    return device_type_;
}

auto DeviceMesh::get_device_id(const std::vector<int64_t>& coord) const -> int64_t {
    if (static_cast<int64_t>(coord.size()) != ndim()) {
        throw std::out_of_range(std::format(
            "DeviceMesh::get_device_id: expected {} coordinates, got {}",
            ndim(), coord.size()));
    }

    // Row-major (C-order): last dimension varies fastest
    int64_t flat_id = 0;
    int64_t stride = 1;
    for (int64_t i = ndim() - 1; i >= 0; --i) {
        if (coord[i] < 0 || coord[i] >= mesh_shape_[i]) {
            throw std::out_of_range(std::format(
                "DeviceMesh::get_device_id: coordinate[{}] = {} is out of range [0, {})",
                i, coord[i], mesh_shape_[i]));
        }
        flat_id += coord[i] * stride;
        stride *= mesh_shape_[i];
    }
    return flat_id;
}

auto DeviceMesh::get_coordinate(int64_t flat_id) const -> std::vector<int64_t> {
    if (flat_id < 0 || flat_id >= size()) {
        throw std::out_of_range(std::format(
            "DeviceMesh::get_coordinate: flat_id {} is out of range [0, {})",
            flat_id, size()));
    }

    std::vector<int64_t> coord(ndim());
    int64_t remaining = flat_id;
    for (int64_t i = ndim() - 1; i >= 0; --i) {
        coord[i] = remaining % mesh_shape_[i];
        remaining /= mesh_shape_[i];
    }
    return coord;
}

auto DeviceMesh::get_dim(const std::string& name) const -> int64_t {
    auto it = dim_name_to_idx_.find(name);
    if (it == dim_name_to_idx_.end()) {
        throw std::invalid_argument(std::format(
            "DeviceMesh::get_dim: dimension name '{}' not found", name));
    }
    return it->second;
}

auto DeviceMesh::get_submesh(const std::string& dim_name) const -> std::vector<int64_t> {
    auto dim_idx = get_dim(dim_name);
    auto my_coord = get_coordinate(local_rank_);

    std::vector<int64_t> submesh_ranks;
    submesh_ranks.reserve(mesh_shape_[dim_idx]);

    // Iterate over all values along the target dimension, keeping other
    // dimensions fixed to our coordinate
    auto coord = my_coord;
    for (int64_t i = 0; i < mesh_shape_[dim_idx]; ++i) {
        coord[dim_idx] = i;
        submesh_ranks.push_back(get_device_id(coord));
    }

    return submesh_ranks;
}

auto DeviceMesh::get_local_rank() const -> int64_t {
    return local_rank_;
}

} // namespace tenzor::distributed
