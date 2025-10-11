/**
 * @file ops.hpp
 * @brief Gradient-aware tensor operations for automatic differentiation
 *
 * Provides operation functions that work with Variables to preserve
 * the autograd computation graph. All operations automatically track
 * gradients when applied to Variables with requires_grad=true.
 */

#pragma once

#include "variable.hpp"
#include "function.hpp"
#include <optional>

namespace tenzor {

// ============================================================================
// Reduction Operations
// ============================================================================

/**
 * @brief Sum reduction with gradient tracking.
 *
 * Computes sum of tensor elements along specified dimension.
 * Gradients are broadcast back to original shape during backpropagation.
 *
 * @param input Input variable
 * @param dim Optional dimension to reduce (nullopt = all dimensions)
 * @param keepdim If true, keep reduced dimension with size 1
 * @return Variable containing sum with gradient function
 *
 * @code
 * Variable x(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * Variable row_sums = sum(x, 1);        // Shape: {3}
 * Variable col_sums = sum(x, 0, true);  // Shape: {1, 4}
 * Variable total = sum(x);              // Shape: {}
 * @endcode
 *
 * @see SumBackward for gradient implementation
 */
auto sum(const Variable& input,
         std::optional<int64_t> dim = std::nullopt,
         bool keepdim = false) -> Variable;

/**
 * @brief Mean reduction with gradient tracking.
 *
 * Computes mean of tensor elements along specified dimension.
 * Gradients are divided by element count and broadcast during backpropagation.
 *
 * @param input Input variable
 * @param dim Optional dimension to reduce (nullopt = all dimensions)
 * @param keepdim If true, keep reduced dimension with size 1
 * @return Variable containing mean with gradient function
 *
 * @code
 * Variable x(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * Variable avg = mean(x);  // Scalar mean
 * @endcode
 *
 * @see MeanBackward for gradient implementation
 */
auto mean(const Variable& input,
          std::optional<int64_t> dim = std::nullopt,
          bool keepdim = false) -> Variable;

/**
 * @brief Max reduction with gradient tracking.
 *
 * Computes maximum of tensor elements along specified dimension.
 * Gradients flow only to maximum elements during backpropagation.
 *
 * @param input Input variable
 * @param dim Optional dimension to reduce (nullopt = all dimensions)
 * @param keepdim If true, keep reduced dimension with size 1
 * @return Variable containing max values with gradient function
 *
 * @note If multiple elements are tied for max, gradient is distributed.
 *
 * @see MaxBackward for gradient implementation
 */
auto max(const Variable& input,
         std::optional<int64_t> dim = std::nullopt,
         bool keepdim = false) -> Variable;

// ============================================================================
// Element-wise Mathematical Operations
// ============================================================================

/**
 * @brief Natural logarithm with gradient tracking.
 *
 * Computes element-wise natural logarithm.
 * Gradient: d(log(x))/dx = 1/x
 *
 * @param input Input variable
 * @return Variable containing log(input) with gradient function
 *
 * @note Undefined for input <= 0
 *
 * @code
 * Variable x(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * Variable y = log(x);
 * @endcode
 *
 * @see LogBackward for gradient implementation
 */
auto log(const Variable& input) -> Variable;

/**
 * @brief Exponential function with gradient tracking.
 *
 * Computes element-wise exponential (e^x).
 * Gradient: d(exp(x))/dx = exp(x)
 *
 * @param input Input variable
 * @return Variable containing exp(input) with gradient function
 *
 * @code
 * Variable x(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * Variable y = exp(x);
 * @endcode
 *
 * @see ExpBackward for gradient implementation
 */
auto exp(const Variable& input) -> Variable;

/**
 * @brief Negation with gradient tracking.
 *
 * Computes element-wise negation (-x).
 * Gradient: d(-x)/dx = -1
 *
 * @param input Input variable
 * @return Variable containing -input with gradient function
 *
 * @see NegBackward for gradient implementation
 */
auto neg(const Variable& input) -> Variable;

/**
 * @brief Absolute value with gradient tracking.
 *
 * Computes element-wise absolute value.
 * Gradient: d(|x|)/dx = sign(x)
 *
 * @param input Input variable
 * @return Variable containing |input| with gradient function
 *
 * @note Gradient is undefined at x=0
 *
 * @see AbsBackward for gradient implementation
 */
auto abs(const Variable& input) -> Variable;

/**
 * @brief Clamp values with gradient tracking.
 *
 * Clamps each element to the range [min, max].
 * Gradients pass through only for elements within bounds.
 *
 * @param input Input variable
 * @param min Minimum value
 * @param max Maximum value
 * @return Variable containing clamped values with gradient function
 *
 * @code
 * Variable x(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * Variable y = clamp(x, -1.0f, 1.0f);  // Clip to [-1, 1]
 * @endcode
 *
 * @see ClampBackward for gradient implementation
 */
auto clamp(const Variable& input, float min, float max) -> Variable;

// ============================================================================
// Activation Functions
// ============================================================================

/**
 * @brief Log-softmax activation with gradient tracking.
 *
 * Computes numerically stable log-softmax along specified dimension.
 * More stable than log(softmax(x)) for numerical computation.
 *
 * Formula: log_softmax(x_i) = x_i - log(sum(exp(x_j)))
 *
 * @param input Input variable
 * @param dim Dimension to compute softmax along
 * @return Variable containing log-softmax values with gradient function
 *
 * @code
 * Variable logits(Tensor({batch, classes}, DType::Float32, Device::cpu()), true);
 * Variable log_probs = log_softmax(logits, 1);  // Along class dimension
 * @endcode
 *
 * @see LogSoftmaxBackward for gradient implementation
 */
auto log_softmax(const Variable& input, int64_t dim) -> Variable;

} // namespace tenzor
