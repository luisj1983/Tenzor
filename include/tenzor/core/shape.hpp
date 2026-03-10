/**
 * @file shape.hpp
 * @brief Shape and stride utilities for tensor operations
 *
 * Provides shape representation, stride computation, and broadcasting
 * utilities for multi-dimensional tensor operations.
 */

#pragma once

#include <vector>
#include <cstdint>
#include <span>
#include <string>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <limits>
#include "../utils/safe_math.hpp"

namespace tenzor {

namespace detail {
inline auto checked_mul(int64_t a, int64_t b) -> int64_t {
    if (b != 0 && safe_abs(a) > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / safe_abs(b)) {
        throw std::overflow_error("Shape stride/numel computation overflow");
    }
    return a * b;
}
} // namespace detail

/**
 * @brief Represents the shape (dimensions) of a tensor.
 *
 * Encapsulates tensor dimensions and provides utilities for shape
 * manipulation, element counting, and comparison. Dimensions are
 * stored as signed 64-bit integers to support negative indexing.
 *
 * @code
 * Shape shape({2, 3, 4});  // 3D tensor: 2x3x4
 * auto total_elements = shape.numel();  // Returns 24
 * @endcode
 */
class Shape {
public:
    using size_type = int64_t;  ///< Dimension size type (signed for negative indexing)

    /**
     * @brief Default constructor creates empty shape.
     */
    Shape() = default;

    /**
     * @brief Construct shape from dimension vector.
     *
     * @param dims Vector of dimensions (e.g., {2, 3, 4})
     *
     * @code
     * Shape shape({32, 64, 128});
     * @endcode
     */
    explicit Shape(std::vector<size_type> dims) : dims_(std::move(dims)) {
        for (size_t i = 0; i < dims_.size(); ++i) {
            if (dims_[i] < 0) {
                throw std::invalid_argument(
                    "Shape dimensions must be non-negative, got " +
                    std::to_string(dims_[i]) + " at index " + std::to_string(i));
            }
        }
    }

    /**
     * @brief Access dimension by index (unchecked).
     *
     * @param idx Dimension index (0-based)
     * @return Dimension size at index
     *
     * @warning No bounds checking. Use at() for checked access.
     */
    auto operator[](size_t idx) const -> size_type { return dims_[idx]; }

    /**
     * @brief Access dimension by index (checked).
     *
     * @param idx Dimension index (0-based)
     * @return Dimension size at index
     * @throws std::out_of_range if index is out of bounds
     */
    auto at(size_t idx) const -> size_type { return dims_.at(idx); }

    /**
     * @brief Get number of dimensions (rank).
     *
     * @return Number of dimensions
     *
     * @code
     * Shape shape({2, 3, 4});
     * auto rank = shape.size();  // Returns 3
     * @endcode
     */
    auto size() const -> size_t { return dims_.size(); }

    /**
     * @brief Get raw pointer to dimension data.
     *
     * @return Pointer to dimension array
     */
    auto data() const -> const size_type* { return dims_.data(); }

    /**
     * @brief Get iterator to beginning of dimensions.
     */
    auto begin() const { return dims_.begin(); }

    /**
     * @brief Get iterator to end of dimensions.
     */
    auto end() const { return dims_.end(); }

    /**
     * @brief Append a dimension to the shape.
     *
     * @param dim Dimension size to append
     *
     * @code
     * Shape shape({2, 3});
     * shape.push_back(4);  // Now {2, 3, 4}
     * @endcode
     */
    auto push_back(size_type dim) -> void {
        if (dim < 0) {
            throw std::invalid_argument(
                "Shape dimensions must be non-negative, got " + std::to_string(dim));
        }
        dims_.push_back(dim);
    }

    /**
     * @brief Resize number of dimensions.
     *
     * @param n New number of dimensions
     *
     * @note New dimensions are zero-initialized.
     */
    auto resize(size_t n) -> void { dims_.resize(n); }

    /**
     * @brief Compute total number of elements.
     *
     * Multiplies all dimensions to get total element count.
     * Returns 1 for empty shapes.
     *
     * @return Total number of elements in tensor
     *
     * @code
     * Shape shape({2, 3, 4});
     * auto count = shape.numel();  // Returns 24
     * @endcode
     */
    auto numel() const -> size_type {
        size_type result = 1;
        for (auto dim : dims_) {
            result = detail::checked_mul(result, dim);
        }
        return result;
    }

    /**
     * @brief Compare shapes for equality.
     *
     * @param other Shape to compare with
     * @return true if all dimensions are equal
     */
    auto operator==(const Shape& other) const -> bool {
        return dims_ == other.dims_;
    }

private:
    std::vector<size_type> dims_;  ///< Dimension storage
};

/**
 * @brief Compute row-major strides from shape.
 *
 * Calculates strides for contiguous row-major (C-style) layout.
 * The last dimension has stride 1, each previous dimension's stride
 * is the product of all subsequent dimensions.
 *
 * @param shape Tensor dimensions
 * @return Stride vector for each dimension
 *
 * @code
 * // For shape {2, 3, 4}
 * auto strides = compute_strides({2, 3, 4});
 * // Returns {12, 4, 1}
 * @endcode
 *
 * @note Time complexity: O(n) where n is number of dimensions
 */
inline auto compute_strides(std::span<const int64_t> shape) -> std::vector<int64_t> {
    std::vector<int64_t> strides(shape.size());
    if (shape.empty()) return strides;

    strides.back() = 1;
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = detail::checked_mul(strides[i + 1], shape[i + 1]);
    }
    return strides;
}

/**
 * @brief Compute channels-last (NHWC) strides from 4D shape.
 *
 * For a 4D tensor with logical shape [N, C, H, W], computes strides
 * that arrange the data in NHWC memory layout. This enables better
 * Tensor Core utilization on NVIDIA GPUs.
 *
 * @param shape Tensor dimensions [N, C, H, W]
 * @return Strides for NHWC layout: [H*W*C, 1, W*C, C]
 *
 * @code
 * // For shape {1, 64, 32, 32} (NCHW)
 * auto strides = compute_channels_last_strides({1, 64, 32, 32});
 * // Returns {65536, 1, 2048, 64} for NHWC layout
 * @endcode
 *
 * @note Returns standard row-major strides for non-4D shapes.
 */
inline auto compute_channels_last_strides(std::span<const int64_t> shape)
    -> std::vector<int64_t> {
    if (shape.size() != 4) {
        return compute_strides(shape);
    }
    // Shape is [N, C, H, W], compute NHWC strides
    int64_t C = shape[1], H = shape[2], W = shape[3];
    int64_t WC = detail::checked_mul(W, C);
    int64_t HWC = detail::checked_mul(H, WC);
    return {HWC, 1, WC, C};
}

/**
 * @brief Compute channels-last-3d (NDHWC) strides from 5D shape.
 *
 * For a 5D tensor with logical shape [N, C, D, H, W], computes strides
 * that arrange the data in NDHWC memory layout.
 *
 * @param shape Tensor dimensions [N, C, D, H, W]
 * @return Strides for NDHWC layout: [D*H*W*C, 1, H*W*C, W*C, C]
 *
 * @note Returns standard row-major strides for non-5D shapes.
 */
inline auto compute_channels_last_3d_strides(std::span<const int64_t> shape)
    -> std::vector<int64_t> {
    if (shape.size() != 5) {
        return compute_strides(shape);
    }
    // Shape is [N, C, D, H, W], compute NDHWC strides
    int64_t C = shape[1], D = shape[2], H = shape[3], W = shape[4];
    int64_t WC = detail::checked_mul(W, C);
    int64_t HWC = detail::checked_mul(H, WC);
    int64_t DHWC = detail::checked_mul(D, HWC);
    return {DHWC, 1, HWC, WC, C};
}

/**
 * @brief Compute broadcasted shape from two input shapes.
 *
 * Applies NumPy-style broadcasting rules to determine the result shape
 * when operating on tensors with different shapes. Two dimensions are
 * compatible when:
 * - They are equal, or
 * - One of them is 1
 *
 * Shapes are aligned from the right (trailing dimensions).
 *
 * @param shape1 First tensor shape
 * @param shape2 Second tensor shape
 * @return Broadcasted result shape
 * @throws std::runtime_error if shapes are not broadcastable
 *
 * @code
 * // Broadcasting examples
 * auto result1 = broadcast_shapes({3, 1, 4}, {2, 1});
 * // Returns {3, 2, 4}
 *
 * auto result2 = broadcast_shapes({5, 1, 3}, {1, 8, 3});
 * // Returns {5, 8, 3}
 * @endcode
 *
 * @see https://numpy.org/doc/stable/user/basics.broadcasting.html
 */
inline auto broadcast_shapes(std::span<const int64_t> shape1,
                            std::span<const int64_t> shape2)
    -> std::vector<int64_t> {
    size_t ndim = std::max(shape1.size(), shape2.size());
    std::vector<int64_t> result(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        int64_t dim1 = i < shape1.size() ? shape1[shape1.size() - 1 - i] : 1;
        int64_t dim2 = i < shape2.size() ? shape2[shape2.size() - 1 - i] : 1;

        if (dim1 == dim2 || dim1 == 1 || dim2 == 1) {
            result[ndim - 1 - i] = std::max(dim1, dim2);
        } else {
            throw std::runtime_error("Shapes are not broadcastable");
        }
    }

    return result;
}

} // namespace tenzor
