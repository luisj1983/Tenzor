/**
 * @file padding.hpp
 * @brief Padding layers for neural networks
 *
 * Provides constant, reflection, and replication padding layers
 * for 1D, 2D, and 3D inputs.
 */

#pragma once

#include "../module.hpp"
#include <vector>

namespace tenzor {
namespace nn {

// ============================================================================
// Constant Padding
// ============================================================================

/**
 * @brief 1D constant padding layer.
 *
 * Pads the last dimension of a 3D input tensor with a constant value.
 *
 * @code
 * ConstantPad1d pad(2, 3, 0.0);  // pad_left=2, pad_right=3, value=0
 * // Input: (N, C, W) -> Output: (N, C, W + 5)
 * @endcode
 */
class ConstantPad1d : public Module {
public:
    /**
     * @brief Construct 1D constant padding layer.
     *
     * @param padding_left Number of elements to pad on the left
     * @param padding_right Number of elements to pad on the right
     * @param value Constant fill value (default: 0)
     */
    explicit ConstantPad1d(int64_t padding_left, int64_t padding_right, double value = 0.0);

    /// Symmetric padding convenience constructor
    explicit ConstantPad1d(int64_t padding, double value = 0.0);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t padding_left_;
    int64_t padding_right_;
    double value_;
};

/**
 * @brief 2D constant padding layer.
 *
 * Pads the last two dimensions of a 4D input tensor with a constant value.
 *
 * @code
 * ConstantPad2d pad(1, 1, 1, 1, 0.0);  // left, right, top, bottom
 * // Input: (N, C, H, W) -> Output: (N, C, H+2, W+2)
 * @endcode
 */
class ConstantPad2d : public Module {
public:
    /**
     * @brief Construct 2D constant padding layer.
     *
     * @param padding_left Padding on the left of the W dimension
     * @param padding_right Padding on the right of the W dimension
     * @param padding_top Padding on the top of the H dimension
     * @param padding_bottom Padding on the bottom of the H dimension
     * @param value Constant fill value (default: 0)
     */
    ConstantPad2d(int64_t padding_left, int64_t padding_right,
                  int64_t padding_top, int64_t padding_bottom, double value = 0.0);

    /// Symmetric padding convenience constructor (same padding on all sides)
    explicit ConstantPad2d(int64_t padding, double value = 0.0);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t padding_left_;
    int64_t padding_right_;
    int64_t padding_top_;
    int64_t padding_bottom_;
    double value_;
};

/**
 * @brief 3D constant padding layer.
 *
 * Pads the last three dimensions of a 5D input tensor with a constant value.
 *
 * @code
 * ConstantPad3d pad(1, 0.0);  // 1 on all sides
 * // Input: (N, C, D, H, W) -> Output: (N, C, D+2, H+2, W+2)
 * @endcode
 */
class ConstantPad3d : public Module {
public:
    /**
     * @brief Construct 3D constant padding layer.
     *
     * @param padding Six-element array: (left, right, top, bottom, front, back)
     * @param value Constant fill value (default: 0)
     */
    ConstantPad3d(std::vector<int64_t> padding, double value = 0.0);

    /// Symmetric padding convenience constructor (same padding on all sides)
    explicit ConstantPad3d(int64_t padding, double value = 0.0);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::vector<int64_t> padding_;  ///< [left, right, top, bottom, front, back]
    double value_;
};

// ============================================================================
// Reflection Padding
// ============================================================================

/**
 * @brief 1D reflection padding layer.
 *
 * Pads using reflection of the input boundary.
 * Padding size must be less than the input size.
 *
 * @code
 * ReflectionPad1d pad(2, 2);
 * // Input [a, b, c, d] -> [c, b, a, b, c, d, c, b]
 * @endcode
 */
class ReflectionPad1d : public Module {
public:
    ReflectionPad1d(int64_t padding_left, int64_t padding_right);
    explicit ReflectionPad1d(int64_t padding);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t padding_left_;
    int64_t padding_right_;
};

/**
 * @brief 2D reflection padding layer.
 *
 * Pads using reflection of the input boundary on the last two dimensions.
 *
 * @code
 * ReflectionPad2d pad(1, 1, 1, 1);  // left, right, top, bottom
 * @endcode
 */
class ReflectionPad2d : public Module {
public:
    ReflectionPad2d(int64_t padding_left, int64_t padding_right,
                    int64_t padding_top, int64_t padding_bottom);
    explicit ReflectionPad2d(int64_t padding);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t padding_left_;
    int64_t padding_right_;
    int64_t padding_top_;
    int64_t padding_bottom_;
};

// ============================================================================
// Replication Padding
// ============================================================================

/**
 * @brief 1D replication padding layer.
 *
 * Pads using replication of the edge values.
 *
 * @code
 * ReplicationPad1d pad(2, 2);
 * // Input [a, b, c, d] -> [a, a, a, b, c, d, d, d]
 * @endcode
 */
class ReplicationPad1d : public Module {
public:
    ReplicationPad1d(int64_t padding_left, int64_t padding_right);
    explicit ReplicationPad1d(int64_t padding);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t padding_left_;
    int64_t padding_right_;
};

/**
 * @brief 2D replication padding layer.
 *
 * Pads using replication of the edge values on the last two dimensions.
 */
class ReplicationPad2d : public Module {
public:
    ReplicationPad2d(int64_t padding_left, int64_t padding_right,
                     int64_t padding_top, int64_t padding_bottom);
    explicit ReplicationPad2d(int64_t padding);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t padding_left_;
    int64_t padding_right_;
    int64_t padding_top_;
    int64_t padding_bottom_;
};

/**
 * @brief 3D replication padding layer.
 *
 * Pads using replication of the edge values on the last three dimensions.
 */
class ReplicationPad3d : public Module {
public:
    /**
     * @param padding Six-element array: (left, right, top, bottom, front, back)
     */
    explicit ReplicationPad3d(std::vector<int64_t> padding);

    /// Symmetric padding convenience constructor
    explicit ReplicationPad3d(int64_t padding);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::vector<int64_t> padding_;  ///< [left, right, top, bottom, front, back]
};

// ============================================================================
// Zero Padding
// ============================================================================

/**
 * @brief 2D zero padding layer.
 *
 * Thin wrapper around ConstantPad2d with value=0.
 *
 * @code
 * ZeroPad2d pad(1);  // Pad 1 on all sides with zeros
 * // Input: (N, C, H, W) -> Output: (N, C, H+2, W+2)
 * @endcode
 */
class ZeroPad2d : public Module {
public:
    ZeroPad2d(int64_t padding_left, int64_t padding_right,
              int64_t padding_top, int64_t padding_bottom);
    explicit ZeroPad2d(int64_t padding);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    ConstantPad2d impl_;
};

} // namespace nn
} // namespace tenzor
