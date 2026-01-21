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

// Check if shape_b can be broadcast to shape_a (for in-place operations)
// Returns true if shape_b can be broadcast to match shape_a
inline bool can_broadcast_to(std::span<const int64_t> shape_a,
                              std::span<const int64_t> shape_b) {
    // shape_b must have <= dimensions than shape_a (or extra dims must be 1)
    // Each dimension of shape_b must either match shape_a or be 1

    size_t ndim_a = shape_a.size();
    size_t ndim_b = shape_b.size();

    // Check from the rightmost (trailing) dimension
    for (size_t i = 0; i < std::max(ndim_a, ndim_b); ++i) {
        int64_t dim_a = i < ndim_a ? shape_a[ndim_a - 1 - i] : 1;
        int64_t dim_b = i < ndim_b ? shape_b[ndim_b - 1 - i] : 1;

        // For in-place broadcast: dim_b must be 1 or equal to dim_a
        if (dim_b != 1 && dim_b != dim_a) {
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

// Generic broadcast operation template with different output type (for comparisons)
template<typename TIn, typename TOut, typename Op>
void broadcast_op(const TIn* a_data, const TIn* b_data, TOut* c_data,
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

// In-place broadcast operation template
// Applies op(a, b) and stores result back in a
// b is broadcast to match a's shape
template<typename T, typename Op>
void broadcast_op_inplace(T* a_data, const T* b_data,
                          std::span<const int64_t> shape_a,
                          std::span<const int64_t> shape_b,
                          Op op) {
    // For in-place, output shape is always shape_a
    // b must be broadcastable to a's shape
    auto strides_b = compute_broadcast_strides(shape_b, shape_a);

    int64_t total_elements = 1;
    for (auto dim : shape_a) {
        total_elements *= dim;
    }

    // Iterate over all elements of a
    for (int64_t out_idx = 0; out_idx < total_elements; ++out_idx) {
        // Convert flat index to multi-dimensional index for b
        int64_t idx_b = 0;
        int64_t tmp = out_idx;

        for (size_t i = 0; i < shape_a.size(); ++i) {
            int64_t coord = tmp % shape_a[shape_a.size() - 1 - i];
            tmp /= shape_a[shape_a.size() - 1 - i];

            int64_t stride_idx = shape_a.size() - 1 - i;
            idx_b += coord * strides_b[stride_idx];
        }

        a_data[out_idx] = op(a_data[out_idx], b_data[idx_b]);
    }
}

} // namespace detail
} // namespace cpu
} // namespace tenzor
