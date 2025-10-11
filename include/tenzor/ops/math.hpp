/**
 * @file math.hpp
 * @brief Mathematical operations for tensors
 *
 * Provides arithmetic, matrix, trigonometric, exponential, and element-wise
 * mathematical operations. All operations support broadcasting where applicable.
 */

#pragma once

#include "../core/tensor.hpp"

namespace tenzor {

/**
 * @defgroup tensor_math Mathematical Operations
 * @brief Arithmetic and mathematical functions for tensors
 * @{
 */

/// @name Arithmetic Operations
/// @{

/** @brief Element-wise addition with broadcasting. */
auto add(const Tensor& a, const Tensor& b) -> Tensor;

/** @brief Element-wise subtraction with broadcasting. */
auto sub(const Tensor& a, const Tensor& b) -> Tensor;

/** @brief Element-wise multiplication with broadcasting. */
auto mul(const Tensor& a, const Tensor& b) -> Tensor;

/** @brief Element-wise division with broadcasting. */
auto div(const Tensor& a, const Tensor& b) -> Tensor;

/// @}
/// @name Matrix Operations
/// @{

/**
 * @brief Matrix multiplication.
 *
 * Supports batched matrix multiplication for tensors with rank > 2.
 * @param a Left matrix (..., M, K)
 * @param b Right matrix (..., K, N)
 * @return Result matrix (..., M, N)
 */
auto matmul(const Tensor& a, const Tensor& b) -> Tensor;

/** @brief Dot product (1D tensors) or matrix-vector product. */
auto dot(const Tensor& a, const Tensor& b) -> Tensor;

/// @}

/// @name Power and Exponential
/// @{

/** @brief Element-wise power: input^exponent */
auto pow(const Tensor& input, float exponent) -> Tensor;

/** @brief Element-wise exponential: e^input */
auto exp(const Tensor& input) -> Tensor;

/** @brief Element-wise natural logarithm */
auto log(const Tensor& input) -> Tensor;

/** @brief Element-wise square root */
auto sqrt(const Tensor& input) -> Tensor;

/// @}
/// @name Trigonometric Functions
/// @{

/** @brief Element-wise sine */
auto sin(const Tensor& input) -> Tensor;

/** @brief Element-wise cosine */
auto cos(const Tensor& input) -> Tensor;

/** @brief Element-wise tangent */
auto tan(const Tensor& input) -> Tensor;

/// @}
/// @name Hyperbolic Functions
/// @{

/** @brief Element-wise hyperbolic sine */
auto sinh(const Tensor& input) -> Tensor;

/** @brief Element-wise hyperbolic cosine */
auto cosh(const Tensor& input) -> Tensor;

/** @brief Element-wise hyperbolic tangent */
auto tanh(const Tensor& input) -> Tensor;

/// @}
/// @name Element-wise Operations
/// @{

/** @brief Element-wise absolute value */
auto abs(const Tensor& input) -> Tensor;

/** @brief Element-wise negation */
auto neg(const Tensor& input) -> Tensor;

/** @brief Element-wise reciprocal: 1/input */
auto reciprocal(const Tensor& input) -> Tensor;

/** @brief Element-wise sign function (-1, 0, 1) */
auto sign(const Tensor& input) -> Tensor;

/// @}
/// @name Rounding Functions
/// @{

/** @brief Round down to nearest integer */
auto floor(const Tensor& input) -> Tensor;

/** @brief Round up to nearest integer */
auto ceil(const Tensor& input) -> Tensor;

/** @brief Round to nearest integer */
auto round(const Tensor& input) -> Tensor;

/// @}
/// @name Clamping Operations
/// @{

/**
 * @brief Clamp values to range [min, max].
 * @param input Input tensor
 * @param min Minimum value
 * @param max Maximum value
 * @return Clamped tensor
 */
auto clamp(const Tensor& input, float min, float max) -> Tensor;

/** @brief Clamp values to minimum */
auto clamp_min(const Tensor& input, float min) -> Tensor;

/** @brief Clamp values to maximum */
auto clamp_max(const Tensor& input, float max) -> Tensor;

/// @}
/** @} */ // end of tensor_math group

} // namespace tenzor
