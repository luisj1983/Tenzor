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
 * @brief Softmax activation with gradient tracking.
 *
 * Computes softmax activation along specified dimension.
 * Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))
 *
 * @param input Input variable
 * @param dim Dimension to compute softmax along
 * @return Variable containing softmax probabilities with gradient function
 *
 * @code
 * Variable logits(Tensor({batch, classes}, DType::Float32, Device::cpu()), true);
 * Variable probs = softmax(logits, 1);  // Along class dimension
 * @endcode
 *
 * @see SoftmaxBackward for gradient implementation
 */
auto softmax(const Variable& input, int64_t dim) -> Variable;

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

// ============================================================================
// Shape Transformation Operations
// ============================================================================

/**
 * @brief Reshape tensor with gradient tracking.
 *
 * Reshapes input variable to the specified shape while preserving gradients.
 * Gradients are reshaped back to original input shape during backpropagation.
 *
 * @param input Input variable
 * @param shape New shape for the tensor
 * @return Variable containing reshaped tensor with gradient function
 *
 * @code
 * Variable x(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * Variable y = reshape(x, {12});  // Shape: {12}
 * Variable z = reshape(x, {2, 6}); // Shape: {2, 6}
 * @endcode
 *
 * @see ReshapeBackward for gradient implementation
 */
auto reshape(const Variable& input, const std::vector<int64_t>& shape) -> Variable;

/**
 * @brief Permute dimensions with gradient tracking.
 *
 * Permutes the dimensions of input variable according to specified order.
 * Gradients are permuted back using inverse permutation during backpropagation.
 *
 * @param input Input variable
 * @param dims New order of dimensions
 * @return Variable containing permuted tensor with gradient function
 *
 * @code
 * Variable x(Tensor({2, 3, 4}, DType::Float32, Device::cpu()), true);
 * Variable y = permute(x, {2, 0, 1});  // Shape: {4, 2, 3}
 * @endcode
 *
 * @see PermuteBackward for gradient implementation
 */
auto permute(const Variable& input, const std::vector<int64_t>& dims) -> Variable;

/**
 * @brief Transpose two dimensions with gradient tracking.
 *
 * Transposes two dimensions of input variable.
 * Gradients are transposed back during backpropagation.
 *
 * @param input Input variable
 * @param dim0 First dimension to transpose
 * @param dim1 Second dimension to transpose
 * @return Variable containing transposed tensor with gradient function
 *
 * @code
 * Variable x(Tensor({2, 3, 4}, DType::Float32, Device::cpu()), true);
 * Variable y = transpose(x, 0, 2);  // Shape: {4, 3, 2}
 * @endcode
 *
 * @see TransposeBackward for gradient implementation
 */
auto transpose(const Variable& input, int64_t dim0, int64_t dim1) -> Variable;

/**
 * @brief Remove dimensions of size 1 with gradient tracking.
 *
 * Removes a dimension of size 1 from input variable.
 * Gradients are unsqueezed back during backpropagation.
 *
 * @param input Input variable
 * @param dim Dimension to remove (must have size 1)
 * @return Variable containing squeezed tensor with gradient function
 *
 * @code
 * Variable x(Tensor({2, 1, 3}, DType::Float32, Device::cpu()), true);
 * Variable y = squeeze(x, 1);  // Shape: {2, 3}
 * @endcode
 *
 * @see SqueezeBackward for gradient implementation
 */
auto squeeze(const Variable& input, int64_t dim) -> Variable;

/**
 * @brief Roll tensor elements along dimension with gradient tracking.
 *
 * Rolls elements of input variable along the specified dimension.
 * Gradients are rolled back in opposite direction during backpropagation.
 *
 * @param input Input variable
 * @param shifts Number of positions to roll (positive = forward, negative = backward)
 * @param dim Dimension along which to roll
 * @return Variable containing rolled tensor with gradient function
 *
 * @code
 * Variable x(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * Variable y = roll(x, 2, 1);  // Roll along dim 1 by 2 positions
 * @endcode
 *
 * @see RollBackward for gradient implementation
 */
auto roll(const Variable& input, int64_t shifts, int64_t dim) -> Variable;

/**
 * @brief Concatenate variables along dimension with gradient tracking.
 *
 * Concatenates a sequence of variables along the specified dimension.
 * Gradients are split back to individual tensors during backpropagation.
 *
 * @param inputs Vector of input variables to concatenate
 * @param dim Dimension along which to concatenate
 * @return Variable containing concatenated tensor with gradient function
 *
 * @code
 * Variable x1(Tensor({2, 3}, DType::Float32, Device::cpu()), true);
 * Variable x2(Tensor({2, 5}, DType::Float32, Device::cpu()), true);
 * Variable y = cat({x1, x2}, 1);  // Shape: {2, 8}
 * @endcode
 *
 * @see CatBackward for gradient implementation
 */
auto cat(const std::vector<Variable>& inputs, int64_t dim) -> Variable;

/**
 * @brief Slice variable along dimension with gradient tracking.
 *
 * Extracts a slice from the input variable along the specified dimension.
 * Gradients are scattered back to original positions during backpropagation.
 *
 * @param input Input variable to slice
 * @param dim Dimension along which to slice
 * @param start Start index (inclusive)
 * @param end End index (exclusive)
 * @param step Step size (default: 1)
 * @return Variable containing sliced tensor with gradient function
 *
 * @code
 * Variable x(Tensor({10, 20}, DType::Float32, Device::cpu()), true);
 * Variable y = slice(x, 1, 5, 15, 2);  // Shape: {10, 5} - every 2nd element from index 5 to 15
 * // Backward: gradient scattered back to positions [5, 7, 9, 11, 13] in dimension 1
 * @endcode
 *
 * @see SliceBackward for gradient implementation
 */
auto slice(const Variable& input, int64_t dim, int64_t start, int64_t end, int64_t step = 1) -> Variable;

// ============================================================================
// Matrix Operations
// ============================================================================

/**
 * @brief Batch matrix multiplication with gradient tracking.
 *
 * Computes batched matrix multiplication of two 3D tensors.
 * For inputs (batch, n, m) and (batch, m, p), outputs (batch, n, p).
 * Gradients are computed using matrix multiplication chain rule.
 *
 * @param a First input variable (batch, n, m)
 * @param b Second input variable (batch, m, p)
 * @return Variable containing bmm(a, b) with gradient function
 *
 * @code
 * Variable a(Tensor({32, 10, 20}, DType::Float32, Device::cpu()), true);
 * Variable b(Tensor({32, 20, 30}, DType::Float32, Device::cpu()), true);
 * Variable c = bmm(a, b);  // Shape: {32, 10, 30}
 * @endcode
 *
 * @see BmmBackward for gradient implementation
 */
auto bmm(const Variable& a, const Variable& b) -> Variable;

/**
 * @brief Matrix multiplication with gradient tracking.
 *
 * Computes matrix multiplication of two 2D tensors.
 * For inputs (n, m) and (m, p), outputs (n, p).
 * Gradients are computed using matrix multiplication chain rule.
 *
 * @param a First input variable (n, m)
 * @param b Second input variable (m, p)
 * @return Variable containing matmul(a, b) with gradient function
 *
 * @code
 * Variable a(Tensor({10, 20}, DType::Float32, Device::cpu()), true);
 * Variable b(Tensor({20, 30}, DType::Float32, Device::cpu()), true);
 * Variable c = matmul(a, b);  // Shape: {10, 30}
 * @endcode
 *
 * @see MatMulBackward for gradient implementation
 */
auto matmul(const Variable& a, const Variable& b) -> Variable;

/**
 * @brief Fused linear layer with gradient tracking.
 *
 * Computes y = x @ W.T + b with automatic differentiation.
 * More efficient than separate matmul + add operations as it:
 * - Avoids intermediate tensor allocation
 * - Uses a single optimized backward function
 * - Leverages MKL BLAS kernels directly
 *
 * @param x Input variable (batch_size, in_features)
 * @param w Weight variable (out_features, in_features)
 * @param b Bias variable (out_features)
 * @return Variable containing linear(x, w, b) with gradient function
 *
 * @code
 * Variable x(Tensor({32, 784}, DType::Float32, Device::cpu()), true);
 * Variable w(Tensor({256, 784}, DType::Float32, Device::cpu()), true);
 * Variable b(Tensor({256}, DType::Float32, Device::cpu()), true);
 * Variable y = linear(x, w, b);  // Shape: {32, 256}
 * @endcode
 *
 * @see LinearBackward for gradient implementation
 */
auto linear(const Variable& x, const Variable& w, const Variable& b) -> Variable;

// ============================================================================
// Activation Functions (Variable wrappers)
// ============================================================================

/// Sigmoid activation: σ(x) = 1/(1+exp(-x)). Grad: σ(x)*(1-σ(x))
auto sigmoid(const Variable& input) -> Variable;

/// Tanh activation. Grad: 1 - tanh²(x)
auto tanh(const Variable& input) -> Variable;

/// GELU activation. Grad: 0.5*(1+erf(x/√2)) + x*φ(x)
auto gelu(const Variable& input) -> Variable;

/// ELU activation with alpha parameter. Grad: x>0 ? 1 : alpha*exp(x)
auto elu(const Variable& input, float alpha = 1.0f) -> Variable;

/// SELU activation (self-normalizing). Grad: x>0 ? λ : λ*α*exp(x)
auto selu(const Variable& input) -> Variable;

/// Mish activation: x*tanh(softplus(x))
auto mish(const Variable& input) -> Variable;

/// Leaky ReLU with negative slope. Grad: x>0 ? 1 : negative_slope
auto leaky_relu(const Variable& input, float negative_slope = 0.01f) -> Variable;

/// Softplus activation: log(1 + exp(β*x))/β. Grad: σ(β*x)
auto softplus(const Variable& input, float beta = 1.0f) -> Variable;

// ============================================================================
// Element-wise Math Operations (Variable wrappers)
// ============================================================================

/// Square root. Grad: 1/(2*sqrt(x))
auto sqrt(const Variable& input) -> Variable;

/// Power with scalar exponent. Grad: n*x^(n-1)
auto pow(const Variable& input, float exponent) -> Variable;

/// Reciprocal: 1/x. Grad: -1/x²
auto reciprocal(const Variable& input) -> Variable;

/// Sine. Grad: cos(x)
auto sin(const Variable& input) -> Variable;

/// Cosine. Grad: -sin(x)
auto cos(const Variable& input) -> Variable;

/// Tangent. Grad: 1/cos²(x)
auto tan(const Variable& input) -> Variable;

/// Arcsine. Grad: 1/sqrt(1-x²)
auto asin(const Variable& input) -> Variable;

/// Arccosine. Grad: -1/sqrt(1-x²)
auto acos(const Variable& input) -> Variable;

/// Arctangent. Grad: 1/(1+x²)
auto atan(const Variable& input) -> Variable;

/// Hyperbolic sine. Grad: cosh(x)
auto sinh(const Variable& input) -> Variable;

/// Hyperbolic cosine. Grad: sinh(x)
auto cosh(const Variable& input) -> Variable;

// ============================================================================
// Extended Math Operations (Variable wrappers)
// ============================================================================

/// Error function. Grad: (2/√π)*exp(-x²)
auto erf(const Variable& input) -> Variable;

/// Complementary error function. Grad: -(2/√π)*exp(-x²)
auto erfc(const Variable& input) -> Variable;

/// Log base 2. Grad: 1/(x*ln(2))
auto log2(const Variable& input) -> Variable;

/// Log base 10. Grad: 1/(x*ln(10))
auto log10(const Variable& input) -> Variable;

/// Log(1+x). Grad: 1/(1+x)
auto log1p(const Variable& input) -> Variable;

/// 2^x. Grad: 2^x * ln(2)
auto exp2(const Variable& input) -> Variable;

/// exp(x)-1. Grad: exp(x)
auto expm1(const Variable& input) -> Variable;

/// Two-argument arctangent. Grad_y: x/(x²+y²), Grad_x: -y/(x²+y²)
auto atan2(const Variable& y, const Variable& x) -> Variable;

// ============================================================================
// Additional Reduction Operations (Variable wrappers)
// ============================================================================

/// Min reduction along dimension
auto min(const Variable& input, std::optional<int64_t> dim = std::nullopt, bool keepdim = false) -> Variable;

/// Standard deviation. Grad: (x-mean)/(N*std)
auto std(const Variable& input, std::optional<int64_t> dim = std::nullopt, bool keepdim = false) -> Variable;

/// Variance. Grad: 2*(x-mean)/N
auto var(const Variable& input, std::optional<int64_t> dim = std::nullopt, bool keepdim = false) -> Variable;

/// Product reduction. Grad: prod/x (with zero handling)
auto prod(const Variable& input, std::optional<int64_t> dim = std::nullopt, bool keepdim = false) -> Variable;

/// Log-sum-exp (numerically stable). Grad: softmax(x, dim)
auto logsumexp(const Variable& input, int64_t dim, bool keepdim = false) -> Variable;

// ============================================================================
// Shape/Indexing Operations (Variable wrappers)
// ============================================================================

/// Insert dimension of size 1. Grad: squeeze(grad, dim)
auto unsqueeze(const Variable& input, int64_t dim) -> Variable;

/// Expand tensor to larger shape. Grad: sum along expanded dims
auto expand(const Variable& input, const std::vector<int64_t>& shape) -> Variable;

/// Flatten dimensions [start, end]. Grad: reshape to original
auto flatten(const Variable& input, int64_t start_dim = 0, int64_t end_dim = -1) -> Variable;

/// Element-wise conditional selection. Grad: grad*cond / grad*!cond
auto where(const Variable& condition, const Variable& x, const Variable& y) -> Variable;

/// Gather elements along dim. Grad: scatter_add
auto gather(const Variable& input, int64_t dim, const Tensor& index) -> Variable;

/// Scatter src into self along dim. Grad: scatter zeros / gather
auto scatter(const Variable& input, int64_t dim, const Tensor& index, const Variable& src) -> Variable;

/// Select elements along dim by index. Grad: index_add
auto index_select(const Variable& input, int64_t dim, const Tensor& index) -> Variable;

/// Narrow (alias for contiguous slice). Grad: zero-padded
auto narrow(const Variable& input, int64_t dim, int64_t start, int64_t length) -> Variable;

/// Flip along dimensions. Grad: flip(grad, dims)
auto flip(const Variable& input, const std::vector<int64_t>& dims) -> Variable;

/// Repeat tensor along dimensions. Grad: sum over repeated dims
auto repeat(const Variable& input, const std::vector<int64_t>& repeats) -> Variable;

} // namespace tenzor

namespace tenzor {
namespace ops {
// Convenience namespace alias for autograd operations
using tenzor::softmax;
using tenzor::log_softmax;
} // namespace ops
} // namespace tenzor
