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
#include <limits>
#include <optional>
#include <utility>

namespace tenzor {

// Forward declaration for sparse autograd ops
class SparseTensor;

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

/**
 * @brief Median reduction with gradient tracking.
 *
 * Computes median of tensor elements along specified dimension.
 * Gradients flow only to median elements during backpropagation.
 *
 * @param input Input variable
 * @param dim Optional dimension to reduce (nullopt = all dimensions)
 * @param keepdim If true, keep reduced dimension with size 1
 * @return Variable containing median values with gradient function
 *
 * @note If multiple elements are tied for median, gradient is distributed.
 *
 * @see MedianBackward for gradient implementation
 */
auto median(const Variable& input,
            std::optional<int64_t> dim = std::nullopt,
            bool keepdim = false) -> Variable;

/**
 * @brief Mode reduction with gradient tracking.
 *
 * Computes mode (most frequent value) of tensor elements along specified dimension.
 * Gradients flow only to mode elements during backpropagation.
 *
 * @param input Input variable
 * @param dim Optional dimension to reduce (nullopt = all dimensions)
 * @param keepdim If true, keep reduced dimension with size 1
 * @return Variable containing mode values with gradient function
 *
 * @note If multiple elements are tied for mode, gradient is distributed.
 *
 * @see ModeBackward for gradient implementation
 */
auto mode(const Variable& input,
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
auto clamp(const Variable& input, double min, double max) -> Variable;

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

/**
 * @brief Split a Variable into a fixed number of (nearly) equal chunks along
 *        a dimension with gradient tracking.
 *
 * Returns up to `chunks` Variables. The split is performed as repeated
 * `autograd::slice` calls, so the result naturally carries grad_fn chains
 * (SliceBackward per output) and supports `create_graph` end-to-end.
 *
 * V.10: previously callers had to chunk the raw `Tensor` via
 * `tenzor::chunk(var.tensor(), …)`, which severed the autograd graph; this
 * Variable-level overload restores the chain.
 *
 * @param input Input variable to split
 * @param chunks Number of chunks (positive integer)
 * @param dim Dimension along which to split (negative dims allowed)
 * @return Vector of Variables; the last entry may be shorter when
 *         `chunks` does not divide the dimension evenly. May contain
 *         fewer than `chunks` entries if the dim is smaller than chunks.
 */
auto chunk(const Variable& input, int64_t chunks, int64_t dim = 0) -> std::vector<Variable>;

/**
 * @brief Split a Variable along a dimension into fixed-size pieces with
 *        gradient tracking.
 *
 * Implemented as repeated `autograd::slice`; see `chunk` for the rationale.
 *
 * @param input Input variable to split
 * @param split_size Size of each output chunk along `dim`
 *        (the final chunk may be shorter if `split_size` doesn't divide
 *        the dimension evenly)
 * @param dim Dimension along which to split (negative dims allowed)
 * @return Vector of Variables that concatenate back to `input`.
 */
auto split(const Variable& input, int64_t split_size, int64_t dim = 0) -> std::vector<Variable>;

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
auto leaky_relu(const Variable& input, double negative_slope = 0.01) -> Variable;

/// Softplus activation: log(1 + exp(β*x))/β. Grad: σ(β*x)
auto softplus(const Variable& input, float beta = 1.0f) -> Variable;

// ============================================================================
// Element-wise Math Operations (Variable wrappers)
// ============================================================================

/// Square root. Grad: 1/(2*sqrt(x))
auto sqrt(const Variable& input) -> Variable;

/// Power with scalar exponent. Grad: n*x^(n-1)
auto pow(const Variable& input, double exponent) -> Variable;

/// Power with Variable exponent: a^b = exp(b * log(a)). Both inputs
/// participate in autograd; gradient w.r.t. `a` is `b * a^(b-1)` and
/// gradient w.r.t. `b` is `a^b * log(a)` (audit-5 Z.20).
auto pow(const Variable& base, const Variable& exponent) -> Variable;

/// Scalar base, Variable exponent: c^b = exp(b * log(c)). Only the
/// exponent is differentiable; gradient is `c^b * log(c)`. This is the
/// Variable-aware analogue of Python's ``scalar ** var`` (audit-5 Z.20).
auto pow(double base, const Variable& exponent) -> Variable;

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

/// Inverse error function. Grad: √π/2 * exp(erfinv(x)²)
auto erfinv(const Variable& input) -> Variable;

/// Gamma function Γ(x). Grad: Γ(x)*ψ(x)
auto gamma(const Variable& input) -> Variable;

/// Log-gamma ln|Γ(x)|. Grad: ψ(x) (digamma)
auto lgamma(const Variable& input) -> Variable;

/// Digamma ψ(x) = d/dx ln Γ(x). Grad: ψ¹(x) (trigamma)
auto digamma(const Variable& input) -> Variable;

/// Polygamma ψⁿ(x), n-th derivative of digamma
auto polygamma(int64_t n, const Variable& input) -> Variable;

/// Modified Bessel I₀(x). Grad: I₁(x)
auto bessel_i0(const Variable& input) -> Variable;

/// Modified Bessel I₁(x). Grad: I₀(x) - I₁(x)/x
auto bessel_i1(const Variable& input) -> Variable;

/// Normalized sinc: sin(πx)/(πx). Grad handled via chain rule
auto sinc(const Variable& input) -> Variable;

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

/// Standard deviation. Grad: grad * (x-mean) / ((N-1)*std) when unbiased=true; / (N*std) when unbiased=false.
auto std(const Variable& input, std::optional<int64_t> dim = std::nullopt, bool keepdim = false,
         bool unbiased = true) -> Variable;

/// Variance. Grad: 2 * grad * (x-mean) / (N-1) when unbiased=true; / N when unbiased=false.
auto var(const Variable& input, std::optional<int64_t> dim = std::nullopt, bool keepdim = false,
         bool unbiased = true) -> Variable;

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

/// Transfer tensor to target device. Grad: transfer back to source device
auto to_device(const Variable& input, Device target) -> Variable;

/// Flatten dimensions [start, end]. Grad: reshape to original
auto flatten(const Variable& input, int64_t start_dim = 0, int64_t end_dim = -1) -> Variable;

/// Element-wise conditional selection. Grad: grad*cond / grad*!cond
auto where(const Variable& condition, const Variable& x, const Variable& y) -> Variable;

/// Gather elements along dim. Grad: scatter_add
auto gather(const Variable& input, int64_t dim, const Tensor& index) -> Variable;

/// Scatter src into self along dim. Grad: scatter zeros / gather
auto scatter(const Variable& input, int64_t dim, const Tensor& index, const Variable& src) -> Variable;

/// Scatter-add src into input along dim. Grad: identity for input, gather for src
auto scatter_add(const Variable& input, int64_t dim, const Tensor& index, const Variable& src) -> Variable;

/// Select elements along dim by index. Grad: index_add
auto index_select(const Variable& input, int64_t dim, const Tensor& index) -> Variable;

/// Narrow (alias for contiguous slice). Grad: zero-padded
auto narrow(const Variable& input, int64_t dim, int64_t start, int64_t length) -> Variable;

/// Flip along dimensions. Grad: flip(grad, dims)
auto flip(const Variable& input, const std::vector<int64_t>& dims) -> Variable;

/// Repeat tensor along dimensions. Grad: sum over repeated dims
auto repeat(const Variable& input, const std::vector<int64_t>& repeats) -> Variable;

// ============================================================================
// Cumulative, Sorting, and Triangular Operations
// ============================================================================

/// Cumulative sum. Grad: reverse cumulative sum
auto cumsum(const Variable& input, int64_t dim) -> Variable;

/// Cumulative product. Grad: reverse cumsum(output*grad)/input (zero-safe)
auto cumprod(const Variable& input, int64_t dim) -> Variable;

/// TopK values along dimension. Grad: scatter grad to original positions
auto topk(const Variable& input, int64_t k, int64_t dim = -1,
          bool largest = true, bool sorted = true) -> std::pair<Variable, Tensor>;

/// Sort along dimension. Grad: scatter grad using inverse permutation
auto sort(const Variable& input, int64_t dim = -1,
          bool descending = false) -> std::pair<Variable, Tensor>;

/// Diagonal extraction/construction. Grad: inverse diag operation
auto diag(const Variable& input, int64_t diagonal = 0) -> Variable;

/// Matrix trace. Grad: grad * eye(n)
auto trace(const Variable& input) -> Variable;

/// Upper triangular. Grad: triu(grad, k)
auto triu(const Variable& input, int64_t diagonal = 0) -> Variable;

/// Lower triangular. Grad: tril(grad, k)
auto tril(const Variable& input, int64_t diagonal = 0) -> Variable;

// ============================================================================
// FFT Operations
// ============================================================================

namespace fft_autograd {

/// FFT with gradient tracking. Grad: ifft(grad)
auto fft(const Variable& input,
         std::optional<int64_t> n = std::nullopt,
         int64_t dim = -1,
         const std::string& norm = "backward") -> Variable;

/// Inverse FFT with gradient tracking. Grad: fft(grad)
auto ifft(const Variable& input,
          std::optional<int64_t> n = std::nullopt,
          int64_t dim = -1,
          const std::string& norm = "backward") -> Variable;

/// Real FFT with gradient tracking. Grad: irfft(grad, signal_length)
auto rfft(const Variable& input,
          std::optional<int64_t> n = std::nullopt,
          int64_t dim = -1,
          const std::string& norm = "backward") -> Variable;

/// Inverse real FFT with gradient tracking. Grad: rfft(grad)
auto irfft(const Variable& input,
           std::optional<int64_t> n = std::nullopt,
           int64_t dim = -1,
           const std::string& norm = "backward") -> Variable;

/// Short-time Fourier transform with gradient tracking.
/// Grad: istft(grad, ...) with the same params (mutual adjoint).
auto stft(const Variable& input,
          int64_t n_fft,
          int64_t hop_length = -1,
          int64_t win_length = -1,
          const Tensor& window = Tensor{},
          bool center = true,
          bool normalized = false,
          bool onesided = true) -> Variable;

/// Inverse short-time Fourier transform with gradient tracking.
/// Grad: stft(grad, ...) with the same params.
auto istft(const Variable& input,
           int64_t n_fft,
           int64_t hop_length = -1,
           int64_t win_length = -1,
           const Tensor& window = Tensor{},
           bool center = true,
           bool normalized = false,
           bool onesided = true,
           std::optional<int64_t> length = std::nullopt) -> Variable;

} // namespace fft_autograd

// ============================================================================
// Linear Algebra Operations
// ============================================================================

/**
 * @brief Matrix determinant with gradient tracking.
 *
 * Computes the determinant of a square matrix.
 * Gradient: dL/dA = dL/dy * det(A) * A^{-T}
 *
 * @param input Input variable containing square matrix (..., N, N)
 * @return Variable containing determinant with gradient function
 *
 * @see DetBackward for gradient implementation
 */
auto det(const Variable& input) -> Variable;

/**
 * @brief Matrix inverse with gradient tracking.
 *
 * Computes the inverse of a square matrix.
 * Gradient: dL/dA = -Y^T @ dL/dY @ Y^T where Y = A^{-1}
 *
 * @param input Input variable containing square matrix (..., N, N)
 * @return Variable containing inverse with gradient function
 *
 * @see InvBackward for gradient implementation
 */
auto inv(const Variable& input) -> Variable;

/**
 * @brief Linear system solve with gradient tracking.
 *
 * Solves AX = B for X.
 * Gradient for B: solve(A^T, dL/dX)
 * Gradient for A: -grad_B @ X^T
 *
 * @param A Coefficient matrix variable (..., N, N)
 * @param B Right-hand side variable (..., N, K)
 * @return Variable containing solution X with gradient function
 *
 * @see SolveBackward for gradient implementation
 */
auto solve(const Variable& A, const Variable& B) -> Variable;

/**
 * @brief Solve A @ X = B using pre-computed LU factors (audit-2026-05-03 Phase 8).
 * LU and pivots are non-Variable inputs (typically not differentiated).
 * Differentiates only w.r.t. B.
 */
auto lu_solve(const Tensor& LU_data, const Tensor& pivots,
              const Variable& B) -> Variable;

/**
 * @brief LU decomposition with pivots (audit-2026-05-03 Phase 8).
 * Returns (L, U, pivots) Variables. pivots is integer non-differentiable.
 */
auto lu(const Variable& A) -> std::tuple<Variable, Variable, Variable>;

/**
 * @brief Non-symmetric eigendecomposition (audit-2026-05-03 Phase 8).
 * Returns (W_real, W_imag, V) Variables. Backward only supports the
 * real-eigenvalue path (eigenvector grad is ill-posed for finite-diff).
 */
auto eig(const Variable& A) -> std::tuple<Variable, Variable, Variable>;

/**
 * @brief 2-D complex FFT (audit-2026-05-03 Phase 11).
 * Composed from fft along each of the two dimensions.
 */
auto fft2(const Variable& input,
          std::optional<std::vector<int64_t>> s = std::nullopt,
          std::vector<int64_t> dim = {-2, -1},
          const std::string& norm = "backward") -> Variable;

auto ifft2(const Variable& input,
           std::optional<std::vector<int64_t>> s = std::nullopt,
           std::vector<int64_t> dim = {-2, -1},
           const std::string& norm = "backward") -> Variable;

/**
 * @brief N-D complex FFT (audit-2026-05-03 Phase 11).
 * Composed from fft along each dim in turn.
 */
auto fftn(const Variable& input,
          std::optional<std::vector<int64_t>> s = std::nullopt,
          std::optional<std::vector<int64_t>> dim = std::nullopt,
          const std::string& norm = "backward") -> Variable;

auto ifftn(const Variable& input,
           std::optional<std::vector<int64_t>> s = std::nullopt,
           std::optional<std::vector<int64_t>> dim = std::nullopt,
           const std::string& norm = "backward") -> Variable;

/**
 * @brief Cholesky decomposition with gradient tracking.
 *
 * Computes lower-triangular L such that A = L @ L^T.
 *
 * @param input Input variable containing symmetric positive-definite matrix (..., N, N)
 * @param upper If true, return upper-triangular factor (default: false)
 * @return Variable containing Cholesky factor with gradient function
 *
 * @see CholeskyBackward for gradient implementation
 */
auto cholesky(const Variable& input, bool upper = false) -> Variable;

/**
 * @brief Solve a linear system using a Cholesky factor, with gradient tracking.
 *
 * Solves A @ X = B given the Cholesky factor L of A (A = L @ L^T).
 *
 * @param B Right-hand side variable (..., N, K)
 * @param L Cholesky factor variable (..., N, N)
 * @param upper If true, L is upper-triangular (default: false)
 * @return Solution variable X with same shape as B
 */
auto cholesky_solve(const Variable& B, const Variable& L, bool upper = false) -> Variable;

/**
 * @brief Singular Value Decomposition with gradient tracking.
 *
 * Factorizes A = U @ diag(S) @ Vh.
 * Returns tuple of three Variables (U, S, Vh).
 *
 * @param input Input variable (..., M, N)
 * @param full_matrices If true, compute full U and Vh
 * @return Tuple of (U, S, Vh) Variables with gradient function
 *
 * @see SvdBackward for gradient implementation
 */
auto svd(const Variable& input, bool full_matrices = true) -> std::tuple<Variable, Variable, Variable>;

/**
 * @brief QR decomposition with gradient tracking.
 *
 * Factorizes A = Q @ R.
 * Returns tuple of two Variables (Q, R).
 *
 * @param input Input variable (..., M, N)
 * @return Tuple of (Q, R) Variables with gradient function
 *
 * @see QrBackward for gradient implementation
 */
auto qr(const Variable& input) -> std::tuple<Variable, Variable>;

/**
 * @brief Symmetric eigendecomposition with gradient tracking.
 *
 * Computes eigenvalues and eigenvectors of a symmetric matrix.
 * Returns tuple of (eigenvalues, eigenvectors).
 *
 * @param input Input variable containing symmetric matrix (..., N, N)
 * @return Tuple of (W, V) Variables with gradient function
 *
 * @see EighBackward for gradient implementation
 */
auto eigh(const Variable& input) -> std::tuple<Variable, Variable>;

/**
 * @brief Symmetric eigenvalues with gradient tracking.
 *
 * Computes eigenvalues of a symmetric matrix.
 * Gradient: dL/dA = V @ diag(dL/dW) @ V^T
 *
 * @param input Input variable containing symmetric matrix (..., N, N)
 * @return Variable containing eigenvalues with gradient function
 *
 * @see EigvalshBackward for gradient implementation
 */
auto eigvalsh(const Variable& input) -> Variable;

/**
 * @brief Matrix norm with gradient tracking.
 *
 * Computes the Frobenius (or other) norm of a matrix.
 * Gradient (Frobenius): dL/dA = dL/dy * A / norm(A)
 *
 * @param input Input variable
 * @param ord Norm order (default: "fro" for Frobenius)
 * @return Variable containing norm value with gradient function
 *
 * @see NormBackward_Linalg for gradient implementation
 */
auto linalg_norm(const Variable& input, const std::string& ord = "fro") -> Variable;

/**
 * @brief Sign and log-absolute-determinant with gradient tracking.
 *
 * More numerically stable than det for large matrices.
 * Gradient flows only through logabsdet: dL/dA = dL/d(logabsdet) * A^{-T}
 *
 * @param input Input variable containing square matrix (..., N, N)
 * @return Tuple of (sign, logabsdet) Variables with gradient function
 *
 * @see SlogdetBackward for gradient implementation
 */
auto slogdet(const Variable& input) -> std::tuple<Variable, Variable>;

// ============================================================================
// Sparse Operations
// ============================================================================

/**
 * @brief Sparse-dense matrix multiplication with gradient tracking.
 *
 * Computes Y = S @ D where S is a sparse matrix and D is a dense matrix.
 * Only the dense input D receives a gradient during backpropagation:
 *   grad_D = S^T @ grad_Y
 *
 * The sparse matrix S is treated as a constant (no gradient computed).
 *
 * @param sparse Sparse matrix (M, K) -- not differentiated
 * @param dense Dense matrix variable (K, N) -- receives gradient
 * @return Variable containing the dense result (M, N) with gradient function
 *
 * @see SpMMBackward for gradient implementation
 */
auto spmm(const SparseTensor& sparse, const Variable& dense) -> Variable;

/**
 * @brief Sparse-dense matrix-vector multiplication with gradient tracking.
 *
 * Computes y = S @ v where S is a sparse matrix and v is a dense vector.
 * Only the dense input v receives a gradient during backpropagation:
 *   grad_v = S^T @ grad_y
 *
 * The sparse matrix S is treated as a constant (no gradient computed).
 *
 * @param sparse Sparse matrix (M, K) -- not differentiated
 * @param vec Dense vector variable (K,) -- receives gradient
 * @return Variable containing the dense result (M,) with gradient function
 *
 * @see SpMVBackward for gradient implementation
 */
auto spmv(const SparseTensor& sparse, const Variable& vec) -> Variable;

/**
 * @brief Sparse-dense addition with gradient tracking.
 *
 * Computes Y = S + D where S is a sparse tensor and D is a dense tensor.
 * Only the dense input D receives a gradient during backpropagation:
 *   grad_D = grad_Y  (gradient passes through directly)
 *
 * The sparse tensor S is treated as a constant (no gradient computed).
 *
 * @param sparse Sparse tensor -- not differentiated
 * @param dense Dense tensor variable -- receives gradient
 * @return Variable containing the dense result with gradient function
 *
 * @see SparseAddBackward for gradient implementation
 */
auto sparse_add(const SparseTensor& sparse, const Variable& dense) -> Variable;

/**
 * @brief Sparse triangular solve with gradient tracking.
 *
 * Solves L @ x = b for x where L is a sparse triangular matrix (constant)
 * and b is a dense Variable. Gradient flows through to b only — the
 * sparse matrix L is not differentiated (sparsity is structural).
 *
 * @param L Sparse triangular matrix (constant)
 * @param b Right-hand side dense Variable
 * @param upper If true, treat L as upper triangular (default lower)
 */
auto sparse_triangular_solve(const SparseTensor& L,
                              const Variable& b,
                              bool upper = false) -> Variable;

// ============================================================================
// New Op Autograd Wrappers (Phase 7)
// ============================================================================

/// LogAddExp: log(exp(a) + exp(b)), numerically stable. Grad: sigmoid decomposition
auto logaddexp(const Variable& a, const Variable& b) -> Variable;

/// LogAddExp2: log2(2^a + 2^b), numerically stable. Grad: base-2 softmax decomposition
auto logaddexp2(const Variable& a, const Variable& b) -> Variable;

/// XLogY: x * log(y), with 0*log(y) = 0. Grad: log(y) for x, x/y for y (0-safe)
auto xlogy(const Variable& x, const Variable& y) -> Variable;

/// Scaled modified Bessel I0e: exp(-|x|) * I0(x). Grad: i1e(x) - sign(x)*i0e(x)
auto i0e(const Variable& input) -> Variable;

/// Scaled modified Bessel I1e: exp(-|x|) * I1(x). Grad: see I1eBackward
auto i1e(const Variable& input) -> Variable;

/// Element-wise entropy: -x*log(x). Grad: -(1 + log(x)) for x > 0
auto entr(const Variable& input) -> Variable;

/// Spherical Bessel j0: sin(x)/x. Grad: cos(x)/x - sin(x)/x^2
auto spherical_bessel_j0(const Variable& input) -> Variable;


// Phase 12 (audit-2026-05-03) — Bessel J/Y and Zeta autograd wrappers.
auto bessel_j0(const Variable& input) -> Variable;
auto bessel_j1(const Variable& input) -> Variable;
auto bessel_y0(const Variable& input) -> Variable;
auto bessel_y1(const Variable& input) -> Variable;
auto zeta(const Variable& s, const Variable& q) -> Variable;

auto betainc(const Variable& a, const Variable& b, const Variable& x) -> Variable;

/// Regularised lower incomplete gamma P(a, x). Differentiable wrt x via
/// dP/dx = x^(a-1) exp(-x) / Gamma(a); a receives zero grad (no
/// elementary closed form for dP/da; PyTorch parity).
auto igamma(const Variable& a, const Variable& x) -> Variable;

/// Regularised upper incomplete gamma Q(a, x). Differentiable wrt x via
/// dQ/dx = -x^(a-1) exp(-x) / Gamma(a); a receives zero grad.
auto igammac(const Variable& a, const Variable& x) -> Variable;

/// Beta function B(a, b) = Gamma(a) Gamma(b) / Gamma(a+b). Closed-form
/// gradient via digamma: dB/da = B*(psi(a) - psi(a+b)).
auto beta(const Variable& a, const Variable& b) -> Variable;

/// Per-row L_p distance between x1 and x2, both shape (B, D). Output (B,).
/// Closed-form backward via sign/|d|^(p-1) and y^(1-p) scaling.
auto pairwise_distance(const Variable& x1, const Variable& x2, double p = 2.0) -> Variable;

/// All pairwise L_p distances within a single batch; shape (N, D) -> (N*(N-1)/2,).
/// NON-DIFFERENTIABLE — backward() throws NonDifferentiable until a
/// dedicated `pdist_backward` kernel lands; see PdistBackward docs.
auto pdist(const Variable& input, double p = 2.0) -> Variable;

/// Cross-pairwise L_p distance between two sets x1: (N, D), x2: (M, D).
/// Output (N, M). NON-DIFFERENTIABLE — same kernel-gap caveat as pdist.
auto cdist(const Variable& x1, const Variable& x2, double p = 2.0) -> Variable;

/// NumPy-style multi-tensor advanced indexing: y = x[indices].
/// NON-DIFFERENTIABLE — the scatter-add backward needs an accumulating
/// multi-dim scatter kernel that the project does not yet expose;
/// backward() throws NonDifferentiable. Use gather/index_select for the
/// single-dim case if you need autograd today.
auto index(const Variable& input,
           const std::vector<std::optional<Tensor>>& indices) -> Variable;

/// One-hot encoding of integer index tensor.
/// NON-DIFFERENTIABLE — the input is integer indices.
auto one_hot(const Variable& input, int64_t num_classes = -1) -> Variable;

/// Linear interpolation: lerp(start, end, weight) = start + weight*(end - start).
/// All three inputs differentiable in the tensor-weight overload.
auto lerp(const Variable& start, const Variable& end, const Variable& weight) -> Variable;

/// Linear interpolation with scalar weight. Only start, end are differentiable.
auto lerp(const Variable& start, const Variable& end, double weight) -> Variable;

/// 3-vector cross product along dim (length 3). grad_a = cross(b, grad),
/// grad_b = cross(grad, a).
auto cross(const Variable& a, const Variable& b, int64_t dim = -1) -> Variable;

/// Normal CDF. Grad: standard normal PDF
auto ndtr(const Variable& input) -> Variable;

/// Log normal CDF. Grad: pdf / ndtr (numerically stable)
auto log_ndtr(const Variable& input) -> Variable;

/// Multivariate log-gamma. Grad: sum of digamma at shifted args
auto multigammaln(const Variable& input, int64_t p) -> Variable;

/// Cosine similarity along dim. Grad: chain rule through norm/dot decomposition
auto cosine_similarity(const Variable& x1, const Variable& x2,
                       int64_t dim = 1, double eps = 1e-8) -> Variable;

/// Renormalize tensor slices. Grad: scale by renorm factor
auto renorm(const Variable& input, double p, int64_t dim, double maxnorm) -> Variable;

/// Inverse via Cholesky factors. Grad: through triangular structure
auto cholesky_inverse(const Variable& input, bool upper = false) -> Variable;

/// LDL^T factorization. Grad: structured (via LDLBackward in function_new_ops).
auto ldl_factor(const Variable& input) -> std::tuple<Variable, Variable>;

/// Solve using LDL^T factors. Grad: re-solve with grad.
auto ldl_solve(const Variable& LD, const Tensor& pivots, const Variable& B) -> Variable;

/// Householder product. Grad: implicit-Q + structured chain rule
/// (see HouseholderProductBackward in function_new_ops).
auto householder_product(const Variable& input, const Variable& tau) -> Variable;

/// Generalized tensor inverse. Grad: -Y @ grad @ Y (reshaped)
auto tensorinv(const Variable& input, int64_t ind = 2) -> Variable;

/// Generalized tensor solve. Grad: through linear system
auto tensorsolve(const Variable& A, const Variable& B) -> Variable;

/// Vector p-norm along dims. Grad: through p-norm formula
auto vector_norm(const Variable& input, double ord = 2.0,
                 std::vector<int64_t> dim = {}, bool keepdim = false) -> Variable;

/// Matrix norm. Grad: depends on norm type (Frobenius supported)
auto matrix_norm(const Variable& input, double ord = 2.0) -> Variable;

/// Dot product along dim. Grad: grad.unsqueeze(dim) * other
auto vecdot(const Variable& a, const Variable& b, int64_t dim = -1) -> Variable;

/// Zero-copy view with custom strides. Grad: scatter_add inverse mapping
auto as_strided(const Variable& input, std::span<const int64_t> size,
                std::span<const int64_t> stride,
                std::optional<int64_t> storage_offset = std::nullopt) -> Variable;

// ============================================================================
// Vision Operations (Variable-level wrappers for grid_sample / affine_grid)
// ============================================================================

/**
 * @brief View a complex tensor as a real tensor with trailing dim 2
 *        (real, imag). Variable wrapper around tenzor::view_as_real.
 *
 * Backward dispatches via ViewAsRealBackward (linear; backward = view_as_complex).
 * Useful for reducing complex-valued outputs (e.g., FFT) to real for gradcheck.
 */
auto view_as_real(const Variable& input) -> Variable;

/**
 * @brief View a real tensor with trailing dim 2 as a complex tensor.
 *        Variable wrapper around tenzor::view_as_complex.
 */
auto view_as_complex(const Variable& input) -> Variable;

/**
 * @brief Complex conjugate. For real tensors this is the identity.
 *
 * Variable-level wrapper around tenzor::conj routed through ConjBackward
 * so the chain stays intact. Required by the complex Wirtinger paths in
 * MulBackward / DivBackward / MatMulBackward (audit-5 X.5 / Y.9) — the
 * raw `Variable(conj(saved_b.tensor()), false)` rewrap severed grad_fn
 * on saved_variables_ Variables.
 */
auto conj(const Variable& input) -> Variable;

/**
 * @brief Sample input at non-integer grid locations with gradient tracking.
 *
 * Variable wrapper around tenzor::grid_sample. Backward via GridSampleBackward
 * computes gradients w.r.t. both `input` and `grid`.
 */
auto grid_sample(const Variable& input,
                 const Variable& grid,
                 const std::string& mode = "bilinear",
                 const std::string& padding_mode = "zeros",
                 bool align_corners = false) -> Variable;

/**
 * @brief Generate a 2D affine grid with gradient tracking.
 *
 * Variable wrapper around tenzor::affine_grid. Backward via AffineGridBackward
 * computes the gradient w.r.t. theta.
 */
auto affine_grid(const Variable& theta,
                 const std::vector<int64_t>& size,
                 bool align_corners = false) -> Variable;

// ============================================================================
// Attention apply helpers
// ============================================================================
//
// Wrap dispatch<OpId::Flash/Fused/Flex Attention>() in autograd Functions so
// the resulting Variable has a grad_fn — enabling training. Direct dispatch
// returns Variables with severed grad chains (audit C1).
// Per docs/internals/attention-contract.md.
//
// Conditional attach: if no input requires grad AND grad mode is disabled,
// dispatch is called raw and a no-grad Variable is returned (no Function alloc).

/**
 * @brief FlashAttention with autograd. Q/K/V are 4D [B, H, S, D].
 *
 * GPU backends that internally collapse to 3D [B*H, S, D] (CUDA/ROCm/OneAPI/
 * Vulkan) handle that reshape inside the apply helper.
 *
 * @param Q Query Variable [B, H, S_q, D]
 * @param K Key Variable [B, H_kv, S_k, D] (H_kv may differ from H for GQA — backend dependent)
 * @param V Value Variable [B, H_kv, S_k, D_v]
 * @param scale Multiplicative scale (typically 1/sqrt(D))
 * @param causal Apply causal (lower-triangular) mask
 * @param dropout_p Dropout probability (training-time only)
 * @param is_training Whether to apply dropout
 * @return Variable [B, H, S_q, D_v] with grad_fn = FlashAttentionBackward
 */
auto flash_attention(const Variable& Q,
                     const Variable& K,
                     const Variable& V,
                     float scale,
                     bool causal = false,
                     float dropout_p = 0.0f,
                     bool is_training = false) -> Variable;

/**
 * @brief FusedAttention with autograd (cuDNN-SDPA-friendly variant; no dropout).
 *        Same shape contract as flash_attention.
 */
auto fused_attention(const Variable& Q,
                     const Variable& K,
                     const Variable& V,
                     float scale,
                     bool causal = false,
                     bool use_cudnn_sdpa = false) -> Variable;

/**
 * @brief FlexAttention with autograd. score_mod_id selects a built-in score
 *        modification op (0 = none); see attention-contract.md for the registry.
 */
auto flex_attention(const Variable& Q,
                    const Variable& K,
                    const Variable& V,
                    float scale,
                    int64_t score_mod_id = 0,
                    const Tensor& block_mask = Tensor{}) -> Variable;

// ============================================================================
// Audit E.7 continuation: autograd wrappers for additional OpIds.
//
// The differentiable ones below build a real grad_fn chain so backward()
// flows. The non-differentiable ones still build a Function wrapper so the
// graph remains structurally valid, but the wrapper's backward() throws
// tenzor::NonDifferentiable with a descriptive message.
// ============================================================================

/** @brief square(x) = x*x. d/dx = 2*x. */
auto square(const Variable& input) -> Variable;

/** @brief rsqrt(x) = 1/sqrt(x). d/dx = -0.5*y^3 where y = rsqrt(x). */
auto rsqrt(const Variable& input) -> Variable;

/** @brief deg2rad(x) = x * (pi/180). d/dx = pi/180. */
auto deg2rad(const Variable& input) -> Variable;

/** @brief rad2deg(x) = x * (180/pi). d/dx = 180/pi. */
auto rad2deg(const Variable& input) -> Variable;

/** @brief logit(x) = log(x/(1-x)). d/dx = 1/(x*(1-x)). Undefined outside (0,1). */
auto logit(const Variable& input, double eps = -1.0) -> Variable;

/** @brief nan_to_num: identity on finite, constant on NaN/Inf.
 *         d/dx = isfinite(x) (1 on finite, 0 on NaN/Inf). */
auto nan_to_num(const Variable& input,
                double nan = 0.0,
                double posinf = std::numeric_limits<double>::max(),
                double neginf = std::numeric_limits<double>::lowest()) -> Variable;

/** @brief Heaviside step function. NON-DIFFERENTIABLE — backward() throws. */
auto heaviside(const Variable& input, const Variable& values) -> Variable;

/** @brief signbit returns Bool. NON-DIFFERENTIABLE — backward() throws. */
auto signbit(const Variable& input) -> Variable;

/** @brief frexp returns (mantissa, exponent). NON-DIFFERENTIABLE — backward() throws. */
auto frexp(const Variable& input) -> std::pair<Variable, Variable>;

/** @brief histogram returns (counts, edges). NON-DIFFERENTIABLE — backward() throws. */
auto histogram(const Variable& input, int64_t bins = 10,
               double min = 0.0, double max = 0.0)
    -> std::pair<Variable, Variable>;

// ============================================================================
// Audit E.7 continuation (batch 2): autograd wrappers for another 10 OpIds.
// Differentiable ones build a real grad_fn chain; non-differentiable ones
// still attach a Function whose backward() throws tenzor::NonDifferentiable.
// ============================================================================

/** @brief sign(x). Gradient is zero almost everywhere — backward returns a
 *         zero tensor of the input's shape/dtype (PyTorch/JAX convention). */
auto sign(const Variable& input) -> Variable;

/** @brief hypot(x, y) = sqrt(x*x + y*y). d/dx = x/h, d/dy = y/h. */
auto hypot(const Variable& x, const Variable& y) -> Variable;

/** @brief copysign(magnitude, sign_src). d/d(mag) = sign(sign_src), d/d(sign_src) = 0. */
auto copysign(const Variable& magnitude, const Variable& sign_src) -> Variable;

/** @brief xlog1py(x, y) = x * log1p(y), 0*log1p(y)=0.
 *         d/dx = log1p(y), d/dy = x/(1+y). */
auto xlog1py(const Variable& x, const Variable& y) -> Variable;

/** @brief addcmul(a, b, c, value) = a + value*b*c.
 *         d/da = 1, d/db = value*c, d/dc = value*b. */
auto addcmul(const Variable& a, const Variable& b, const Variable& c,
             double value = 1.0) -> Variable;

/** @brief addcdiv(a, b, c, value) = a + value*b/c.
 *         d/da = 1, d/db = value/c, d/dc = -value*b/c^2. */
auto addcdiv(const Variable& a, const Variable& b, const Variable& c,
             double value = 1.0) -> Variable;

/** @brief floor(x). NON-DIFFERENTIABLE — backward() throws. */
auto floor(const Variable& input) -> Variable;

/** @brief ceil(x). NON-DIFFERENTIABLE — backward() throws. */
auto ceil(const Variable& input) -> Variable;

/** @brief isnan(x). NON-DIFFERENTIABLE — backward() throws. */
auto isnan(const Variable& input) -> Variable;

/** @brief logical_and(a, b). NON-DIFFERENTIABLE — backward() throws. */
auto logical_and(const Variable& a, const Variable& b) -> Variable;

// ============================================================================
// Audit E.7 continuation (batch 3): autograd wrappers for another 10 OpIds.
// Differentiable wrappers build a real grad_fn chain; non-differentiable
// ones still attach a Function whose backward() throws
// tenzor::NonDifferentiable.
// ============================================================================

/** @brief addmm(input, mat1, mat2, beta, alpha) = beta*input + alpha*(mat1 @ mat2).
 *         d/d(input)=beta*grad, d/d(mat1)=alpha*grad@mat2^T,
 *         d/d(mat2)=alpha*mat1^T@grad. Broadcast-reduced as needed. */
auto addmm(const Variable& input, const Variable& mat1, const Variable& mat2,
           double beta = 1.0, double alpha = 1.0) -> Variable;

/** @brief addmv(input, mat, vec, beta, alpha) = beta*input + alpha*(mat @ vec).
 *         d/d(input)=beta*grad, d/d(mat)=alpha*outer(grad, vec),
 *         d/d(vec)=alpha*mat^T@grad. */
auto addmv(const Variable& input, const Variable& mat, const Variable& vec,
           double beta = 1.0, double alpha = 1.0) -> Variable;

/** @brief baddbmm(input, batch1, batch2, beta, alpha)
 *         = beta*input + alpha*(batch1 @ batch2). Batched analogue of addmm. */
auto baddbmm(const Variable& input, const Variable& batch1, const Variable& batch2,
             double beta = 1.0, double alpha = 1.0) -> Variable;

/** @brief nansum(x, dim, keepdim): sum with NaN treated as 0. Backward
 *         broadcasts grad back to input shape with NaN positions zeroed. */
auto nansum(const Variable& input,
            std::optional<int64_t> dim = std::nullopt,
            bool keepdim = false) -> Variable;

/** @brief tile(input, reps): tile input along each dim. Backward splits
 *         tiled dims and sums over the repetition axes. */
auto tile(const Variable& input, std::vector<int64_t> reps) -> Variable;

/** @brief count_nonzero(x, dim). NON-DIFFERENTIABLE — backward() throws. */
auto count_nonzero(const Variable& input,
                   std::optional<int64_t> dim = std::nullopt) -> Variable;

/** @brief isinf(x). NON-DIFFERENTIABLE — backward() throws. */
auto isinf(const Variable& input) -> Variable;

/** @brief bitwise_and(a, b). NON-DIFFERENTIABLE — backward() throws. */
auto bitwise_and(const Variable& a, const Variable& b) -> Variable;

/** @brief round(x). NON-DIFFERENTIABLE — backward() throws. */
auto round(const Variable& input) -> Variable;

/** @brief eq(a, b). NON-DIFFERENTIABLE — backward() throws. */
auto eq(const Variable& a, const Variable& b) -> Variable;

// ============================================================================
// Audit E.7 continuation (batch 4): autograd wrappers for another 10 OpIds.
// Closes out the boolean-comparison family plus bitwise_or / xor / not,
// isfinite, and one differentiable op (logcumsumexp).
// ============================================================================

/** @brief ne(a, b). NON-DIFFERENTIABLE — backward() throws. */
auto ne(const Variable& a, const Variable& b) -> Variable;

/** @brief lt(a, b). NON-DIFFERENTIABLE — backward() throws. */
auto lt(const Variable& a, const Variable& b) -> Variable;

/** @brief le(a, b). NON-DIFFERENTIABLE — backward() throws. */
auto le(const Variable& a, const Variable& b) -> Variable;

/** @brief gt(a, b). NON-DIFFERENTIABLE — backward() throws. */
auto gt(const Variable& a, const Variable& b) -> Variable;

/** @brief ge(a, b). NON-DIFFERENTIABLE — backward() throws. */
auto ge(const Variable& a, const Variable& b) -> Variable;

/** @brief bitwise_or(a, b). NON-DIFFERENTIABLE — backward() throws. */
auto bitwise_or(const Variable& a, const Variable& b) -> Variable;

/** @brief bitwise_xor(a, b). NON-DIFFERENTIABLE — backward() throws. */
auto bitwise_xor(const Variable& a, const Variable& b) -> Variable;

/** @brief bitwise_not(x). NON-DIFFERENTIABLE — backward() throws. */
auto bitwise_not(const Variable& input) -> Variable;

/** @brief isfinite(x). NON-DIFFERENTIABLE — backward() throws. */
auto isfinite(const Variable& input) -> Variable;

/** @brief logcumsumexp(x, dim) = log(cumsum(exp(x), dim)).
 *         Backward: grad_x = exp(x) * reverse_cumsum(grad_y * exp(-y), dim). */
auto logcumsumexp(const Variable& input, int64_t dim) -> Variable;

// ============================================================================
// Audit E.7 continuation (batch 5): 10 more wrappers — 6 non-diff, 4 diff.
// ============================================================================

/** @brief isposinf(x). NON-DIFFERENTIABLE — backward() throws. */
auto isposinf(const Variable& input) -> Variable;

/** @brief isneginf(x). NON-DIFFERENTIABLE — backward() throws. */
auto isneginf(const Variable& input) -> Variable;

/** @brief trunc(x): round toward zero. NON-DIFFERENTIABLE — backward() throws. */
auto trunc(const Variable& input) -> Variable;

/** @brief any(x, dim, keepdim). NON-DIFFERENTIABLE — backward() throws. */
auto any(const Variable& input,
         std::optional<int64_t> dim = std::nullopt,
         bool keepdim = false) -> Variable;

/** @brief all(x, dim, keepdim). NON-DIFFERENTIABLE — backward() throws. */
auto all(const Variable& input,
         std::optional<int64_t> dim = std::nullopt,
         bool keepdim = false) -> Variable;

/** @brief has_inf_nan(x). NON-DIFFERENTIABLE — backward() throws. */
auto has_inf_nan(const Variable& input) -> Variable;

/** @brief nanmean(x, dim, keepdim): mean with NaN treated as missing.
 *         Backward: grad_x = where(isnan(x), 0, grad_y_broadcast / count_non_nan). */
auto nanmean(const Variable& input,
             std::optional<int64_t> dim = std::nullopt,
             bool keepdim = false) -> Variable;

/** @brief masked_fill(x, mask, value): overwrite x[mask] with scalar value.
 *         Backward: grad_x = where(mask, 0, grad_y); value is non-diff scalar. */
auto masked_fill(const Variable& input, const Tensor& mask, float value) -> Variable;

/** @brief masked_select(x, mask): flat gather at mask=true positions.
 *         Backward scatters grad_y back into zeros_like(x). */
auto masked_select(const Variable& input, const Tensor& mask) -> Variable;

/** @brief masked_scatter(x, mask, source): overwrite x[mask] with leading
 *         source elements. Backward routes grad to x (where !mask) and to
 *         source (padded with zeros beyond the masked count). */
auto masked_scatter(const Variable& input, const Tensor& mask,
                    const Variable& source) -> Variable;

// ---- Audit E.7 batch 7 — index/scatter/view ops --------------------------

/** @brief index_add(input, dim, index, source): self with source[i] added at
 *         index[i] along dim. Backward: grad_input=grad_y;
 *         grad_source=index_select(grad_y, dim, index). `index` is non-diff. */
auto index_add(const Variable& input, int64_t dim, const Tensor& index,
               const Variable& source) -> Variable;

/** @brief index_copy(input, dim, index, source): self with source[i] copied at
 *         index[i] along dim. Backward: grad_input=index_fill(grad_y, dim, index, 0);
 *         grad_source=index_select(grad_y, dim, index). */
auto index_copy(const Variable& input, int64_t dim, const Tensor& index,
                const Variable& source) -> Variable;

/** @brief index_fill(input, dim, index, value): self with scalar value placed at
 *         index positions along dim. Backward: grad_input=index_fill(grad_y, dim, index, 0). */
auto index_fill(const Variable& input, int64_t dim, const Tensor& index,
                float value) -> Variable;

/** @brief select_scatter(input, src, dim, index): copy of input with src placed at
 *         input.select(dim, index). Backward routes grad to input (slice zeroed)
 *         and to src (the gathered slice). */
auto select_scatter(const Variable& input, const Variable& src,
                    int64_t dim, int64_t index) -> Variable;

/** @brief slice_scatter(input, src, dim, start, end, step): copy of input with src
 *         placed in the slice region. Backward routes grad to input (slice zeroed)
 *         and to src (= slice of grad_y). */
auto slice_scatter(const Variable& input, const Variable& src, int64_t dim,
                   int64_t start = 0, int64_t end = -1, int64_t step = 1) -> Variable;

/** @brief diagonal_scatter(input, src, offset, dim1, dim2): copy of input with src
 *         placed on a diagonal. NON-DIFFERENTIABLE — needs a general N-D
 *         `diagonal(offset, dim1, dim2)` extractor that the project does not yet
 *         ship; see DiagonalScatterBackward. */
auto diagonal_scatter(const Variable& input, const Variable& src,
                      int64_t offset = 0, int64_t dim1 = 0, int64_t dim2 = 1) -> Variable;

/** @brief repeat_interleave(input, repeats, dim) — uniform integer repeats overload.
 *         Backward: split repeated axis as (orig, repeats), sum over repeats axis,
 *         reshape to input shape. */
auto repeat_interleave(const Variable& input, int64_t repeats,
                       std::optional<int64_t> dim = std::nullopt) -> Variable;

/** @brief repeat_interleave(input, repeats: Tensor, dim) — per-element repeats.
 *         NON-DIFFERENTIABLE: variable-length expansion requires an accumulating
 *         scatter that we don't have a clean closed form for at Variable level. */
auto repeat_interleave(const Variable& input, const Tensor& repeats,
                       std::optional<int64_t> dim = std::nullopt) -> Variable;

/** @brief unfold(input, kernel_size, stride, padding, dilation) — im2col patch
 *         extraction. Backward = fold(grad_y, (H, W), kernel, stride, padding, dilation). */
auto unfold(const Variable& input, int64_t kernel_size, int64_t stride = 1,
            int64_t padding = 0, int64_t dilation = 1) -> Variable;

/** @brief nonzero(x): Int64 indices of nonzero entries. NON-DIFFERENTIABLE. */
auto nonzero(const Variable& input) -> Variable;

/** @brief unique(x, sorted, return_inverse, return_counts): sorted unique values
 *         plus optional inverse / counts. NON-DIFFERENTIABLE (discontinuous in x).
 *         Returns only the unique-values Variable; inverse/counts are integer-typed
 *         and discarded here for the autograd surface. */
auto unique(const Variable& input,
            bool sorted = true,
            bool return_inverse = false,
            bool return_counts = false) -> Variable;

// ---- Audit E.7 batch 8 — order stats, integration, segment ops ----------

/** @brief aminmax(x, dim, keepdim): simultaneous (min, max) along dim.
 *         Backward sums grad_min scatter-to-argmin and grad_max scatter-
 *         to-argmax (tie-normalised). Returns the pair as a vector to
 *         match the multi-output convention. */
auto aminmax(const Variable& input,
             std::optional<int64_t> dim = std::nullopt,
             bool keepdim = false) -> std::pair<Variable, Variable>;

/** @brief kthvalue(x, k, dim, keepdim): k-th smallest along dim. Backward
 *         scatters grad onto the k-th-value position (tie-normalised).
 *         Returns only the value Variable; the index is integer-typed. */
auto kthvalue(const Variable& input, int64_t k,
              int64_t dim = -1, bool keepdim = false) -> Variable;

/** @brief quantile(x, q, dim, keepdim): interpolated q-th quantile.
 *         NON-DIFFERENTIABLE — backward needs a stable per-row argsort
 *         with interpolation weights; see QuantileBackward. */
auto quantile(const Variable& input, double q,
              std::optional<int64_t> dim = std::nullopt,
              bool keepdim = false) -> Variable;

/** @brief nanmedian(x, dim): median along dim, NaN entries skipped.
 *         Backward: scatter grad onto positions equal to the median value,
 *         excluding NaN positions; tie-normalised. */
auto nanmedian(const Variable& input,
               std::optional<int64_t> dim = std::nullopt) -> Variable;

/** @brief trapezoid(y, dx, dim): trapezoidal integration with uniform dx.
 *         Backward applies linear weights along dim (dx interior, dx/2 at
 *         endpoints). */
auto trapezoid(const Variable& y, double dx = 1.0, int64_t dim = -1) -> Variable;

/** @brief trapezoid(y, x, dim): trapezoidal integration with non-uniform x.
 *         Backward routes grad only to y (x is non-diff, matching PyTorch). */
auto trapezoid(const Variable& y, const Variable& x, int64_t dim = -1) -> Variable;

/** @brief cumulative_trapezoid(y, dx, dim): cumulative trapezoidal
 *         integration with uniform dx. Backward is the reverse-cumsum of
 *         dx-weighted halves; see CumulativeTrapezoidBackward. */
auto cumulative_trapezoid(const Variable& y, double dx = 1.0,
                          int64_t dim = -1) -> Variable;

/** @brief cumulative_trapezoid(y, x, dim): cumulative trapezoidal
 *         integration with non-uniform x. Backward routes grad only to y. */
auto cumulative_trapezoid(const Variable& y, const Variable& x,
                          int64_t dim = -1) -> Variable;

/** @brief segment_reduce(data, offsets, reduce, axis): segmented reduction.
 *         NON-DIFFERENTIABLE — the kernel does not return per-segment
 *         argmax/argmin indices needed for max/min, and prod is
 *         numerically unsafe through zeros. See SegmentReduceBackward. */
auto segment_reduce(const Variable& data, const Tensor& offsets,
                    const std::string& reduce = "sum",
                    int64_t axis = 0) -> Variable;

/** @brief gumbel_softmax(logits, tau, hard, dim): Gumbel-softmax sampling.
 *         NON-DIFFERENTIABLE — forward doesn't save the drawn Gumbel noise,
 *         so the soft+STE backward cannot be reconstructed. See
 *         GumbelSoftmaxBackward. */
auto gumbel_softmax(const Variable& logits, double tau = 1.0,
                    bool hard = false, int64_t dim = -1) -> Variable;

/** @brief cummax(x, dim): cumulative max returning (values, indices).
 *         Backward: scatter_add of grad_values along dim using saved
 *         indices. Returns the pair; indices are non-differentiable. */
auto cummax(const Variable& input, int64_t dim) -> std::pair<Variable, Variable>;

/** @brief cummin(x, dim): cumulative min returning (values, indices). */
auto cummin(const Variable& input, int64_t dim) -> std::pair<Variable, Variable>;

// ---- Audit E.7 batch 9 — indexing reductions ----------------------------

/** @brief scatter_reduce(input, dim, index, src, reduce, include_self):
 *         scatter src into input at index positions along `dim`, combining
 *         colliding writes by `reduce` ("sum" | "mean" | "amax" | "amin" |
 *         "prod"). Differentiable for "sum" / "mean" (grad_src is
 *         gather(grad_out) optionally divided by per-position write count;
 *         grad_input is grad_out itself or zero depending on include_self).
 *         NonDifferentiable for "amax" / "amin" / "prod" — see
 *         ScatterReduceBackward. */
auto scatter_reduce(const Variable& input, int64_t dim, const Tensor& index,
                    const Variable& src, const std::string& reduce,
                    bool include_self = true) -> Variable;

/** @brief index_reduce(input, dim, index, src, reduce, include_self):
 *         1-D-index sibling of scatter_reduce. Differentiable for "sum" /
 *         "mean"; NonDifferentiable for "amax" / "amin" / "prod" — see
 *         IndexReduceBackward. */
auto index_reduce(const Variable& input, int64_t dim, const Tensor& index,
                  const Variable& src, const std::string& reduce,
                  bool include_self = true) -> Variable;

} // namespace tenzor

namespace tenzor {
namespace ops {
// Convenience namespace alias for autograd operations
using tenzor::softmax;
using tenzor::log_softmax;
} // namespace ops
} // namespace tenzor
