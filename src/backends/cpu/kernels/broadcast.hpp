#pragma once

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include <type_traits>
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

// SIMD fast path for scalar-broadcast: one operand is a single element,
// the other is contiguous. Avoids per-element index computation.
template<typename T, typename Op>
bool try_scalar_broadcast(const T* a_data, const T* b_data, T* c_data,
                           std::span<const int64_t> shape_a,
                           std::span<const int64_t> shape_b,
                           int64_t total_elements, Op op) {
    // Check if b is a scalar (size-1 tensor) and a is the full tensor
    auto is_scalar = [](std::span<const int64_t> shape) {
        int64_t numel = 1;
        for (auto d : shape) numel *= d;
        return numel == 1;
    };

    const T* vec_data = nullptr;
    T scalar_val{};
    bool scalar_is_rhs = false;

    if (is_scalar(shape_b)) {
        vec_data = a_data;
        scalar_val = b_data[0];
        scalar_is_rhs = true;
    } else if (is_scalar(shape_a)) {
        vec_data = b_data;
        scalar_val = a_data[0];
        scalar_is_rhs = false;
    } else {
        return false;  // Neither operand is scalar
    }

    // Simple loop — the compiler auto-vectorizes for basic ops (add, mul, etc.)
    // and we avoid the expensive per-element index computation entirely.
    if (scalar_is_rhs) {
        for (int64_t i = 0; i < total_elements; ++i) {
            c_data[i] = op(vec_data[i], scalar_val);
        }
    } else {
        for (int64_t i = 0; i < total_elements; ++i) {
            c_data[i] = op(scalar_val, vec_data[i]);
        }
    }
    return true;
}

// Generic broadcast operation template
// Precondition: c_data must point to contiguous memory of size product(output_shape).
// This is always satisfied because the dispatcher allocates a fresh contiguous output tensor.
template<typename T, typename Op>
void broadcast_op(const T* a_data, const T* b_data, T* c_data,
                  std::span<const int64_t> shape_a,
                  std::span<const int64_t> shape_b,
                  std::span<const int64_t> output_shape,
                  Op op) {

    int64_t total_elements = 1;
    for (auto dim : output_shape) {
        total_elements *= dim;
    }

    // Early return for zero-element outputs (e.g. broadcasting with empty tensors)
    if (total_elements == 0) {
        return;
    }

    // Fast path: scalar broadcast with contiguous memory (avoids index computation)
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
        if (try_scalar_broadcast(a_data, b_data, c_data, shape_a, shape_b, total_elements, op)) {
            return;
        }
    }

    auto strides_a = compute_broadcast_strides(shape_a, output_shape);
    auto strides_b = compute_broadcast_strides(shape_b, output_shape);
    const size_t ndim = output_shape.size();

    // Specialized loops for common dimensions (avoid per-element coordinate tracking)
    if (ndim == 1) {
        const int64_t sa0 = strides_a[0], sb0 = strides_b[0];
        for (int64_t i0 = 0; i0 < output_shape[0]; ++i0)
            c_data[i0] = op(a_data[i0 * sa0], b_data[i0 * sb0]);
        return;
    }
    if (ndim == 2) {
        const int64_t d0 = output_shape[0], d1 = output_shape[1];
        const int64_t sa0 = strides_a[0], sa1 = strides_a[1];
        const int64_t sb0 = strides_b[0], sb1 = strides_b[1];
        int64_t out_idx = 0;
        for (int64_t i0 = 0; i0 < d0; ++i0) {
            int64_t ba = i0 * sa0, bb = i0 * sb0;
            for (int64_t i1 = 0; i1 < d1; ++i1, ++out_idx)
                c_data[out_idx] = op(a_data[ba + i1 * sa1], b_data[bb + i1 * sb1]);
        }
        return;
    }
    if (ndim == 3) {
        const int64_t d0 = output_shape[0], d1 = output_shape[1], d2 = output_shape[2];
        const int64_t sa0 = strides_a[0], sa1 = strides_a[1], sa2 = strides_a[2];
        const int64_t sb0 = strides_b[0], sb1 = strides_b[1], sb2 = strides_b[2];
        int64_t out_idx = 0;
        for (int64_t i0 = 0; i0 < d0; ++i0) {
            int64_t ba0 = i0 * sa0, bb0 = i0 * sb0;
            for (int64_t i1 = 0; i1 < d1; ++i1) {
                int64_t ba1 = ba0 + i1 * sa1, bb1 = bb0 + i1 * sb1;
                for (int64_t i2 = 0; i2 < d2; ++i2, ++out_idx)
                    c_data[out_idx] = op(a_data[ba1 + i2 * sa2], b_data[bb1 + i2 * sb2]);
            }
        }
        return;
    }
    if (ndim == 4) {
        const int64_t d0 = output_shape[0], d1 = output_shape[1];
        const int64_t d2 = output_shape[2], d3 = output_shape[3];
        const int64_t sa0 = strides_a[0], sa1 = strides_a[1], sa2 = strides_a[2], sa3 = strides_a[3];
        const int64_t sb0 = strides_b[0], sb1 = strides_b[1], sb2 = strides_b[2], sb3 = strides_b[3];
        int64_t out_idx = 0;
        for (int64_t i0 = 0; i0 < d0; ++i0) {
            int64_t ba0 = i0 * sa0, bb0 = i0 * sb0;
            for (int64_t i1 = 0; i1 < d1; ++i1) {
                int64_t ba1 = ba0 + i1 * sa1, bb1 = bb0 + i1 * sb1;
                for (int64_t i2 = 0; i2 < d2; ++i2) {
                    int64_t ba2 = ba1 + i2 * sa2, bb2 = bb1 + i2 * sb2;
                    for (int64_t i3 = 0; i3 < d3; ++i3, ++out_idx)
                        c_data[out_idx] = op(a_data[ba2 + i3 * sa3], b_data[bb2 + i3 * sb3]);
                }
            }
        }
        return;
    }

    // Generic fallback for ndim > 4: carry-based coordinate tracking
    std::vector<int64_t> coords(ndim, 0);

    for (int64_t out_idx = 0; out_idx < total_elements; ++out_idx) {
        int64_t idx_a = 0, idx_b = 0;
        for (size_t d = 0; d < ndim; ++d) {
            idx_a += coords[d] * strides_a[d];
            idx_b += coords[d] * strides_b[d];
        }
        c_data[out_idx] = op(a_data[idx_a], b_data[idx_b]);

        // Increment coordinates with carry (replaces modulo/divide)
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++coords[d] < output_shape[d]) break;
            coords[d] = 0;
        }
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

    const size_t ndim = output_shape.size();

    // Specialized loops for common dimensions
    if (ndim == 1) {
        const int64_t sa0 = strides_a[0], sb0 = strides_b[0];
        for (int64_t i0 = 0; i0 < output_shape[0]; ++i0)
            c_data[i0] = op(a_data[i0 * sa0], b_data[i0 * sb0]);
        return;
    }
    if (ndim == 2) {
        const int64_t d0 = output_shape[0], d1 = output_shape[1];
        const int64_t sa0 = strides_a[0], sa1 = strides_a[1];
        const int64_t sb0 = strides_b[0], sb1 = strides_b[1];
        int64_t out_idx = 0;
        for (int64_t i0 = 0; i0 < d0; ++i0) {
            int64_t ba = i0 * sa0, bb = i0 * sb0;
            for (int64_t i1 = 0; i1 < d1; ++i1, ++out_idx)
                c_data[out_idx] = op(a_data[ba + i1 * sa1], b_data[bb + i1 * sb1]);
        }
        return;
    }
    if (ndim == 3) {
        const int64_t d0 = output_shape[0], d1 = output_shape[1], d2 = output_shape[2];
        const int64_t sa0 = strides_a[0], sa1 = strides_a[1], sa2 = strides_a[2];
        const int64_t sb0 = strides_b[0], sb1 = strides_b[1], sb2 = strides_b[2];
        int64_t out_idx = 0;
        for (int64_t i0 = 0; i0 < d0; ++i0) {
            int64_t ba0 = i0 * sa0, bb0 = i0 * sb0;
            for (int64_t i1 = 0; i1 < d1; ++i1) {
                int64_t ba1 = ba0 + i1 * sa1, bb1 = bb0 + i1 * sb1;
                for (int64_t i2 = 0; i2 < d2; ++i2, ++out_idx)
                    c_data[out_idx] = op(a_data[ba1 + i2 * sa2], b_data[bb1 + i2 * sb2]);
            }
        }
        return;
    }
    if (ndim == 4) {
        const int64_t d0 = output_shape[0], d1 = output_shape[1];
        const int64_t d2 = output_shape[2], d3 = output_shape[3];
        const int64_t sa0 = strides_a[0], sa1 = strides_a[1], sa2 = strides_a[2], sa3 = strides_a[3];
        const int64_t sb0 = strides_b[0], sb1 = strides_b[1], sb2 = strides_b[2], sb3 = strides_b[3];
        int64_t out_idx = 0;
        for (int64_t i0 = 0; i0 < d0; ++i0) {
            int64_t ba0 = i0 * sa0, bb0 = i0 * sb0;
            for (int64_t i1 = 0; i1 < d1; ++i1) {
                int64_t ba1 = ba0 + i1 * sa1, bb1 = bb0 + i1 * sb1;
                for (int64_t i2 = 0; i2 < d2; ++i2) {
                    int64_t ba2 = ba1 + i2 * sa2, bb2 = bb1 + i2 * sb2;
                    for (int64_t i3 = 0; i3 < d3; ++i3, ++out_idx)
                        c_data[out_idx] = op(a_data[ba2 + i3 * sa3], b_data[bb2 + i3 * sb3]);
                }
            }
        }
        return;
    }

    // Generic fallback for ndim > 4
    std::vector<int64_t> coords(ndim, 0);

    for (int64_t out_idx = 0; out_idx < total_elements; ++out_idx) {
        int64_t idx_a = 0, idx_b = 0;
        for (size_t d = 0; d < ndim; ++d) {
            idx_a += coords[d] * strides_a[d];
            idx_b += coords[d] * strides_b[d];
        }
        c_data[out_idx] = op(a_data[idx_a], b_data[idx_b]);

        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++coords[d] < output_shape[d]) break;
            coords[d] = 0;
        }
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

    const size_t ndim = shape_a.size();

    // Specialized loops for common dimensions
    if (ndim == 1) {
        const int64_t sb0 = strides_b[0];
        for (int64_t i0 = 0; i0 < shape_a[0]; ++i0)
            a_data[i0] = op(a_data[i0], b_data[i0 * sb0]);
        return;
    }
    if (ndim == 2) {
        const int64_t d0 = shape_a[0], d1 = shape_a[1];
        const int64_t sb0 = strides_b[0], sb1 = strides_b[1];
        int64_t out_idx = 0;
        for (int64_t i0 = 0; i0 < d0; ++i0) {
            int64_t bb = i0 * sb0;
            for (int64_t i1 = 0; i1 < d1; ++i1, ++out_idx)
                a_data[out_idx] = op(a_data[out_idx], b_data[bb + i1 * sb1]);
        }
        return;
    }
    if (ndim == 3) {
        const int64_t d0 = shape_a[0], d1 = shape_a[1], d2 = shape_a[2];
        const int64_t sb0 = strides_b[0], sb1 = strides_b[1], sb2 = strides_b[2];
        int64_t out_idx = 0;
        for (int64_t i0 = 0; i0 < d0; ++i0) {
            int64_t bb0 = i0 * sb0;
            for (int64_t i1 = 0; i1 < d1; ++i1) {
                int64_t bb1 = bb0 + i1 * sb1;
                for (int64_t i2 = 0; i2 < d2; ++i2, ++out_idx)
                    a_data[out_idx] = op(a_data[out_idx], b_data[bb1 + i2 * sb2]);
            }
        }
        return;
    }
    if (ndim == 4) {
        const int64_t d0 = shape_a[0], d1 = shape_a[1];
        const int64_t d2 = shape_a[2], d3 = shape_a[3];
        const int64_t sb0 = strides_b[0], sb1 = strides_b[1], sb2 = strides_b[2], sb3 = strides_b[3];
        int64_t out_idx = 0;
        for (int64_t i0 = 0; i0 < d0; ++i0) {
            int64_t bb0 = i0 * sb0;
            for (int64_t i1 = 0; i1 < d1; ++i1) {
                int64_t bb1 = bb0 + i1 * sb1;
                for (int64_t i2 = 0; i2 < d2; ++i2) {
                    int64_t bb2 = bb1 + i2 * sb2;
                    for (int64_t i3 = 0; i3 < d3; ++i3, ++out_idx)
                        a_data[out_idx] = op(a_data[out_idx], b_data[bb2 + i3 * sb3]);
                }
            }
        }
        return;
    }

    // Generic fallback for ndim > 4
    std::vector<int64_t> coords(ndim, 0);

    for (int64_t out_idx = 0; out_idx < total_elements; ++out_idx) {
        int64_t idx_b = 0;
        for (size_t d = 0; d < ndim; ++d) {
            idx_b += coords[d] * strides_b[d];
        }
        a_data[out_idx] = op(a_data[out_idx], b_data[idx_b]);

        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++coords[d] < shape_a[d]) break;
            coords[d] = 0;
        }
    }
}

} // namespace detail
} // namespace cpu
} // namespace tenzor
