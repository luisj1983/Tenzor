/**
 * @file device_mesh.cpp
 * @brief DeviceMesh implementation
 */

#include "tenzor/distributed/device_mesh.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <format>
#include <unordered_set>

namespace tenzor::distributed {

DeviceMesh::DeviceMesh(Device::Type device_type,
                       std::vector<int64_t> mesh_shape,
                       std::vector<std::string> mesh_dim_names)
    : device_type_(device_type),
      mesh_shape_(std::move(mesh_shape)),
      mesh_dim_names_(std::move(mesh_dim_names)) {

    if (mesh_shape_.size() != mesh_dim_names_.size()) {
        throw std::invalid_argument(std::format(
            "DeviceMesh: mesh_shape has {} dimensions but {} names were provided",
            mesh_shape_.size(), mesh_dim_names_.size()));
    }

    if (mesh_shape_.empty()) {
        throw std::invalid_argument("DeviceMesh: mesh_shape must not be empty");
    }

    // Validate shape values are positive
    for (int64_t i = 0; i < static_cast<int64_t>(mesh_shape_.size()); ++i) {
        if (mesh_shape_[i] <= 0) {
            throw std::invalid_argument(std::format(
                "DeviceMesh: mesh_shape[{}] = {} must be positive",
                i, mesh_shape_[i]));
        }
    }

    // Validate unique dimension names
    std::unordered_set<std::string> seen;
    for (int64_t i = 0; i < static_cast<int64_t>(mesh_dim_names_.size()); ++i) {
        if (!seen.insert(mesh_dim_names_[i]).second) {
            throw std::invalid_argument(std::format(
                "DeviceMesh: duplicate dimension name '{}'",
                mesh_dim_names_[i]));
        }
        dim_name_to_idx_[mesh_dim_names_[i]] = i;
    }

    // In single-process mode, local rank is 0
    local_rank_ = 0;
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
