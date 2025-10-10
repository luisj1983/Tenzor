#pragma once

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace tenzor {
namespace cpu {
namespace detail {

// Check if two shapes are broadcastable
inline bool are_broadcastable(std::span<const int64_t> shape_a,
                               std::span<const int64_t> shape_b) {
    size_t max_ndim = std::max(shape_a.size(), shape_b.size());

    for (size_t i = 0; i < max_ndim; ++i) {
        int64_t dim_a = i < shape_a.size() ? shape_a[shape_a.size() - 1 - i] : 1;
        int64_t dim_b = i < shape_b.size() ? shape_b[shape_b.size() - 1 - i] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }

    return true;
}

// Compute the broadcasted output shape
inline std::vector<int64_t> compute_broadcast_shape(std::span<const int64_t> shape_a,
                                                     std::span<const int64_t> shape_b) {
    size_t max_ndim = std::max(shape_a.size(), shape_b.size());
    std::vector<int64_t> result(max_ndim);

    for (size_t i = 0; i < max_ndim; ++i) {
        int64_t dim_a = i < shape_a.size() ? shape_a[shape_a.size() - 1 - i] : 1;
        int64_t dim_b = i < shape_b.size() ? shape_b[shape_b.size() - 1 - i] : 1;

        if (dim_a == dim_b || dim_a == 1 || dim_b == 1) {
            result[max_ndim - 1 - i] = std::max(dim_a, dim_b);
        } else {
            throw std::runtime_error("Shapes are not broadcastable");
        }
    }

    return result;
}

// Compute strides for broadcasting
inline std::vector<int64_t> compute_broadcast_strides(std::span<const int64_t> shape,
                                                       std::span<const int64_t> broadcast_shape) {
    std::vector<int64_t> strides(broadcast_shape.size(), 0);

    // Compute normal strides for the original shape
    std::vector<int64_t> original_strides(shape.size());
    if (!shape.empty()) {
        original_strides.back() = 1;
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            original_strides[i] = original_strides[i + 1] * shape[i + 1];
        }
    }

    // Map to broadcast strides
    int64_t offset = static_cast<int64_t>(broadcast_shape.size()) - static_cast<int64_t>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 1) {
            strides[offset + i] = 0;  // Broadcasting dimension
        } else {
            strides[offset + i] = original_strides[i];
        }
    }

    return strides;
}

// Generic broadcast operation template
template<typename T, typename Op>
void broadcast_op(const T* a_data, const T* b_data, T* c_data,
                  std::span<const int64_t> shape_a,
                  std::span<const int64_t> shape_b,
                  std::span<const int64_t> output_shape,
                  Op op) {

    auto strides_a = compute_broadcast_strides(shape_a, output_shape);
    auto strides_b = compute_broadcast_strides(shape_b, output_shape);

    int64_t total_elements = 1;
    for (auto dim : output_shape) {
        total_elements *= dim;
    }

    // Iterate over all output elements
    for (int64_t out_idx = 0; out_idx < total_elements; ++out_idx) {
        // Convert flat index to multi-dimensional index
        int64_t idx_a = 0;
        int64_t idx_b = 0;
        int64_t tmp = out_idx;

        for (size_t i = 0; i < output_shape.size(); ++i) {
            int64_t coord = tmp % output_shape[output_shape.size() - 1 - i];
            tmp /= output_shape[output_shape.size() - 1 - i];

            int64_t stride_idx = output_shape.size() - 1 - i;
            idx_a += coord * strides_a[stride_idx];
            idx_b += coord * strides_b[stride_idx];
        }

        c_data[out_idx] = op(a_data[idx_a], b_data[idx_b]);
    }
}

} // namespace detail
} // namespace cpu
} // namespace tenzor
