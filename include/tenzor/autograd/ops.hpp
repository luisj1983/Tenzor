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

/// LDL^T factorization. Grad: structured (stub)
auto ldl_factor(const Variable& input) -> std::tuple<Variable, Variable>;

/// Solve using LDL^T factors. Grad: re-solve with grad
auto ldl_solve(const Variable& LD, const Tensor& pivots, const Variable& B) -> Variable;

/// Householder product. Grad: complex (stub)
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

} // namespace tenzor

namespace tenzor {
namespace ops {
// Convenience namespace alias for autograd operations
using tenzor::softmax;
using tenzor::log_softmax;
} // namespace ops
} // namespace tenzor
