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

/**
 * @brief Batch matrix multiplication.
 *
 * Performs matrix multiplication for each corresponding pair of matrices in a batch.
 * @param a Left batch of matrices (batch, n, m)
 * @param b Right batch of matrices (batch, m, p)
 * @return Result batch of matrices (batch, n, p)
 * @throws std::runtime_error if inputs are not 3D or dimensions don't match
 */
auto bmm(const Tensor& a, const Tensor& b) -> Tensor;

/** @brief Dot product (1D tensors) or matrix-vector product. */
auto dot(const Tensor& a, const Tensor& b) -> Tensor;

/**
 * @brief Fused multiply-add for matrices: beta * input + alpha * (mat1 @ mat2).
 *
 * Uses a single BLAS GEMM call with alpha/beta parameters for optimal performance.
 * Avoids separate matmul, scale, and add operations.
 *
 * @param input Bias tensor (M, N) or broadcastable
 * @param mat1 Left matrix (M, K)
 * @param mat2 Right matrix (K, N)
 * @param beta Scalar multiplier for input (default: 1.0)
 * @param alpha Scalar multiplier for mat1 @ mat2 (default: 1.0)
 * @return Result tensor (M, N)
 */
auto addmm(const Tensor& input, const Tensor& mat1, const Tensor& mat2,
           double beta = 1.0, double alpha = 1.0) -> Tensor;

/**
 * @brief Fused matrix-vector multiply-add: beta * input + alpha * (mat @ vec).
 *
 * Uses a single BLAS GEMV call with alpha/beta parameters for optimal performance.
 *
 * @param input Bias vector (M,) or broadcastable
 * @param mat Matrix (M, K)
 * @param vec Vector (K,)
 * @param beta Scalar multiplier for input (default: 1.0)
 * @param alpha Scalar multiplier for mat @ vec (default: 1.0)
 * @return Result vector (M,)
 */
auto addmv(const Tensor& input, const Tensor& mat, const Tensor& vec,
           double beta = 1.0, double alpha = 1.0) -> Tensor;

/**
 * @brief Batched fused multiply-add: beta * input + alpha * (batch1 @ batch2).
 *
 * Uses batched BLAS GEMM with alpha/beta parameters for optimal performance.
 *
 * @param input Bias tensor (B, M, N) or broadcastable
 * @param batch1 Left batch of matrices (B, M, K)
 * @param batch2 Right batch of matrices (B, K, N)
 * @param beta Scalar multiplier for input (default: 1.0)
 * @param alpha Scalar multiplier for batch1 @ batch2 (default: 1.0)
 * @return Result tensor (B, M, N)
 */
auto baddbmm(const Tensor& input, const Tensor& batch1, const Tensor& batch2,
             double beta = 1.0, double alpha = 1.0) -> Tensor;

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

/** @brief Element-wise arctangent (inverse tangent) */
auto atan(const Tensor& input) -> Tensor;

/** @brief Element-wise arcsine (inverse sine) */
auto asin(const Tensor& input) -> Tensor;

/** @brief Element-wise arccosine (inverse cosine) */
auto acos(const Tensor& input) -> Tensor;

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

/** @brief Element-wise sigmoid function: 1 / (1 + exp(-x)) */
auto sigmoid(const Tensor& input) -> Tensor;

/** @brief Element-wise minimum of two tensors */
auto minimum(const Tensor& a, const Tensor& b) -> Tensor;

/** @brief Element-wise maximum of two tensors */
auto maximum(const Tensor& a, const Tensor& b) -> Tensor;

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
/// @name Comparison Operations
/// @{

/**
 * @brief Element-wise equality comparison.
 * @param a First tensor
 * @param b Second tensor (must be broadcastable with a)
 * @return Boolean tensor with true where a == b
 */
auto eq(const Tensor& a, const Tensor& b) -> Tensor;

/**
 * @brief Element-wise inequality comparison.
 * @param a First tensor
 * @param b Second tensor (must be broadcastable with a)
 * @return Boolean tensor with true where a != b
 */
auto ne(const Tensor& a, const Tensor& b) -> Tensor;

/**
 * @brief Element-wise less-than comparison.
 * @param a First tensor
 * @param b Second tensor (must be broadcastable with a)
 * @return Boolean tensor with true where a < b
 */
auto lt(const Tensor& a, const Tensor& b) -> Tensor;

/**
 * @brief Element-wise less-than-or-equal comparison.
 * @param a First tensor
 * @param b Second tensor (must be broadcastable with a)
 * @return Boolean tensor with true where a <= b
 */
auto le(const Tensor& a, const Tensor& b) -> Tensor;

/**
 * @brief Element-wise greater-than comparison.
 * @param a First tensor
 * @param b Second tensor (must be broadcastable with a)
 * @return Boolean tensor with true where a > b
 */
auto gt(const Tensor& a, const Tensor& b) -> Tensor;

/**
 * @brief Element-wise greater-than-or-equal comparison.
 * @param a First tensor
 * @param b Second tensor (must be broadcastable with a)
 * @return Boolean tensor with true where a >= b
 */
auto ge(const Tensor& a, const Tensor& b) -> Tensor;

/// @}
/// @name Scalar Arithmetic Operations
/// @brief Optimized overloads that avoid creating temporary scalar tensors
/// @{

/** @brief Element-wise addition with scalar (avoids temporary tensor). */
auto add(const Tensor& a, double scalar) -> Tensor;

/** @brief Element-wise subtraction with scalar (avoids temporary tensor). */
auto sub(const Tensor& a, double scalar) -> Tensor;

/** @brief Element-wise multiplication by scalar (avoids temporary tensor). */
auto mul(const Tensor& a, double scalar) -> Tensor;

/** @brief Element-wise division by scalar (avoids temporary tensor). */
auto div(const Tensor& a, double scalar) -> Tensor;

/// @}
/// @name In-Place Operations
/// @{

/**
 * @brief In-place element-wise addition with broadcasting.
 * @param self Tensor to modify in-place
 * @param other Tensor to add
 * @return Reference to modified tensor
 * @note More efficient than out-of-place version - modifies tensor without allocation
 */
auto add_(Tensor& self, const Tensor& other) -> Tensor&;

/**
 * @brief In-place element-wise multiplication with broadcasting.
 * @param self Tensor to modify in-place
 * @param other Tensor to multiply
 * @return Reference to modified tensor
 * @note More efficient than out-of-place version - modifies tensor without allocation
 */
auto mul_(Tensor& self, const Tensor& other) -> Tensor&;

/**
 * @brief In-place element-wise subtraction with broadcasting.
 * @param self Tensor to modify in-place
 * @param other Tensor to subtract
 * @return Reference to modified tensor
 * @note More efficient than out-of-place version - modifies tensor without allocation
 */
auto sub_(Tensor& self, const Tensor& other) -> Tensor&;

/**
 * @brief In-place element-wise division with broadcasting.
 * @param self Tensor to modify in-place
 * @param other Tensor to divide by
 * @return Reference to modified tensor
 * @note More efficient than out-of-place version - modifies tensor without allocation
 */
auto div_(Tensor& self, const Tensor& other) -> Tensor&;

/// @}

// =========================================================================
// Extended Math Operations
// =========================================================================

/// @name Logarithmic Functions
/// @{
auto log2(const Tensor& input) -> Tensor;
auto log10(const Tensor& input) -> Tensor;
auto log1p(const Tensor& input) -> Tensor;
/// @}

/// @name Exponential Functions
/// @{
auto exp2(const Tensor& input) -> Tensor;
auto expm1(const Tensor& input) -> Tensor;
/// @}

/// @name Special Functions
/// @{
auto erf(const Tensor& input) -> Tensor;
auto erfc(const Tensor& input) -> Tensor;
auto erfinv(const Tensor& input) -> Tensor;

auto gamma(const Tensor& input) -> Tensor;
auto lgamma(const Tensor& input) -> Tensor;
auto digamma(const Tensor& input) -> Tensor;
auto polygamma(int64_t n, const Tensor& input) -> Tensor;

auto beta(const Tensor& a, const Tensor& b) -> Tensor;
auto betainc(const Tensor& a, const Tensor& b, const Tensor& x) -> Tensor;

auto bessel_j0(const Tensor& input) -> Tensor;
auto bessel_j1(const Tensor& input) -> Tensor;
auto bessel_y0(const Tensor& input) -> Tensor;
auto bessel_y1(const Tensor& input) -> Tensor;
auto bessel_i0(const Tensor& input) -> Tensor;
auto bessel_i1(const Tensor& input) -> Tensor;

auto sinc(const Tensor& input) -> Tensor;
auto zeta(const Tensor& x, const Tensor& q) -> Tensor;
/// @}

/// @name Classification Functions (return Bool tensors)
/// @{
auto isnan(const Tensor& input) -> Tensor;
auto isinf(const Tensor& input) -> Tensor;
auto isfinite(const Tensor& input) -> Tensor;
/// @}

/// @name Binary Math Functions
/// @{
auto atan2(const Tensor& y, const Tensor& x) -> Tensor;
auto fmod(const Tensor& a, const Tensor& b) -> Tensor;
auto remainder(const Tensor& a, const Tensor& b) -> Tensor;
auto lerp(const Tensor& start, const Tensor& end, const Tensor& weight) -> Tensor;
auto lerp(const Tensor& start, const Tensor& end, double weight) -> Tensor;

/** @brief Fractional part: x - floor(x) */
auto frac(const Tensor& input) -> Tensor;

/** @brief Heaviside step function. Returns 0 where x<0, values where x==0, 1 where x>0. */
auto heaviside(const Tensor& input, const Tensor& values) -> Tensor;

/**
 * @brief Replace NaN, positive infinity, and negative infinity with specified values.
 * @param input Input tensor
 * @param nan Replacement for NaN (default: 0.0)
 * @param posinf Replacement for +Inf (default: largest finite value of dtype)
 * @param neginf Replacement for -Inf (default: smallest finite value of dtype)
 */
auto nan_to_num(const Tensor& input, double nan = 0.0,
                double posinf = std::numeric_limits<double>::max(),
                double neginf = std::numeric_limits<double>::lowest()) -> Tensor;

/// @}

/// @name Bitwise Operations (integer types)
/// @{
auto bitwise_and(const Tensor& a, const Tensor& b) -> Tensor;
auto bitwise_or(const Tensor& a, const Tensor& b) -> Tensor;
auto bitwise_xor(const Tensor& a, const Tensor& b) -> Tensor;
auto bitwise_not(const Tensor& input) -> Tensor;
auto bitwise_left_shift(const Tensor& input, const Tensor& shift) -> Tensor;
auto bitwise_right_shift(const Tensor& input, const Tensor& shift) -> Tensor;

/// @}

/// @name Logical Operations (return Bool tensors)
/// @{
auto logical_and(const Tensor& a, const Tensor& b) -> Tensor;
auto logical_or(const Tensor& a, const Tensor& b) -> Tensor;
auto logical_not(const Tensor& input) -> Tensor;
auto logical_xor(const Tensor& a, const Tensor& b) -> Tensor;
/// @}

/// @name Vector Operations
/// @{
auto cross(const Tensor& input, const Tensor& other, int64_t dim = -1) -> Tensor;
/// @}

/// @name Search Operations
/// @{

/**
 * @brief Find indices where elements should be inserted to maintain sorted order.
 *
 * @param sorted_sequence 1-D sorted tensor to search in
 * @param values Values to search for
 * @param right If false (default), use lower_bound; if true, use upper_bound
 * @return Int64 tensor with same shape as values containing insertion indices
 */
auto searchsorted(const Tensor& sorted_sequence, const Tensor& values, bool right = false) -> Tensor;
/// @}

/// @name Sampling Operations
/// @{

/**
 * @brief Sample from a categorical distribution using the Gumbel-Softmax trick.
 *
 * @param logits Unnormalized log probabilities
 * @param tau Temperature parameter (default 1.0, lower = more discrete)
 * @param hard If true, use straight-through estimator for hard one-hot
 * @param dim Dimension along which softmax is computed (default -1)
 * @return Soft (or hard) samples from the categorical distribution
 */
auto gumbel_softmax(const Tensor& logits, double tau = 1.0, bool hard = false, int64_t dim = -1) -> Tensor;
/// @}

/// @name Complex Number Operations
/// @{

/** @brief Complex conjugate. For real tensors, returns a copy. */
auto conj(const Tensor& input) -> Tensor;

/** @brief Extract real part. Returns a Float32/Float64 tensor. */
auto real(const Tensor& input) -> Tensor;

/** @brief Extract imaginary part. Returns a Float32/Float64 tensor (zeros for real inputs). */
auto imag(const Tensor& input) -> Tensor;

/** @brief Phase angle (argument) of complex numbers. Returns Float32/Float64 tensor. */
auto angle(const Tensor& input) -> Tensor;

/** @brief Construct complex tensor from magnitude and phase angle tensors. */
auto polar(const Tensor& abs, const Tensor& angle) -> Tensor;

/// @}

/**
 * @brief Compute pairwise distance between two sets of vectors.
 *
 * Computes batched pairwise p-norm distance: output[b][i][j] = ||x1[b][i] - x2[b][j]||_p
 *
 * @param x1 Input tensor (B, P, M) or (P, M)
 * @param x2 Input tensor (B, R, M) or (R, M)
 * @param p p-norm value (default: 2.0)
 * @return Distance tensor (B, P, R) or (P, R)
 */
auto cdist(const Tensor& x1, const Tensor& x2, double p = 2.0) -> Tensor;

/// @name Composed Math Operations
/// @{

/**
 * @brief Finite differences along a dimension (like numpy.diff).
 *
 * Computes the n-th discrete difference along the given dimension.
 * For n=1: output[i] = input[i+1] - input[i] along dim.
 *
 * @param input Input tensor
 * @param n Number of times to recursively compute differences (default: 1)
 * @param dim Dimension along which to compute differences (default: -1)
 * @return Tensor with size reduced by n along dim
 */
auto diff(const Tensor& input, int64_t n = 1, int64_t dim = -1) -> Tensor;

/**
 * @brief Numerically stable log(exp(a) + exp(b)).
 *
 * Computed as max(a, b) + log1p(exp(-abs(a - b))).
 *
 * @param a First tensor
 * @param b Second tensor (must be broadcastable with a)
 * @return Element-wise logaddexp result
 */
auto logaddexp(const Tensor& a, const Tensor& b) -> Tensor;

/**
 * @brief Numerically stable log2(2^a + 2^b).
 *
 * Computed as max(a, b) + log2(1 + exp2(-abs(a - b))).
 *
 * @param a First tensor
 * @param b Second tensor (must be broadcastable with a)
 * @return Element-wise logaddexp2 result
 */
auto logaddexp2(const Tensor& a, const Tensor& b) -> Tensor;

/**
 * @brief Compute x * log(y) with the convention that 0 * log(y) = 0.
 *
 * Handles the case where x == 0 correctly, even if y is NaN or 0.
 * Implementation: where(x == 0, zeros_like(x), x * log(y)).
 *
 * @param x First tensor
 * @param y Second tensor (must be broadcastable with x)
 * @return Element-wise xlogy result
 */
auto xlogy(const Tensor& x, const Tensor& y) -> Tensor;

/// @}

/// @name Comparison Utilities
/// @{

/**
 * @brief Element-wise check if two tensors are close within tolerance.
 *
 * Returns a Bool tensor where each element is true if:
 *   |a - b| <= atol + rtol * |b|
 *
 * @param a First tensor
 * @param b Second tensor
 * @param rtol Relative tolerance (default: 1e-5)
 * @param atol Absolute tolerance (default: 1e-8)
 * @return Bool tensor of same shape
 */
auto isclose(const Tensor& a, const Tensor& b, double rtol = 1e-5, double atol = 1e-8) -> Tensor;

/**
 * @brief Check if all elements of two tensors are close within tolerance.
 *
 * Returns true if isclose(a, b, rtol, atol) is true for all elements.
 *
 * @param a First tensor
 * @param b Second tensor
 * @param rtol Relative tolerance (default: 1e-5)
 * @param atol Absolute tolerance (default: 1e-8)
 * @return true if all elements are close
 */
auto allclose(const Tensor& a, const Tensor& b, double rtol = 1e-5, double atol = 1e-8) -> bool;

/// @}

// =========================================================================
// New element-wise math operations for PyTorch parity
// =========================================================================

/// Reciprocal square root: 1/sqrt(x)
auto rsqrt(const Tensor& input) -> Tensor;

/// Element-wise square: x*x
auto square(const Tensor& input) -> Tensor;

/// Inverse hyperbolic sine
auto asinh(const Tensor& input) -> Tensor;

/// Inverse hyperbolic cosine
auto acosh(const Tensor& input) -> Tensor;

/// Inverse hyperbolic tangent
auto atanh(const Tensor& input) -> Tensor;

/// Overflow-safe hypotenuse: sqrt(x*x + y*y)
auto hypot(const Tensor& x, const Tensor& y) -> Tensor;

/// Copy sign of second tensor to magnitude of first
auto copysign(const Tensor& magnitude, const Tensor& sign) -> Tensor;

/// Next representable floating-point value from 'from' toward 'to'
auto nextafter(const Tensor& from, const Tensor& to) -> Tensor;

/// Greatest common divisor (integer tensors only)
auto gcd(const Tensor& a, const Tensor& b) -> Tensor;

/// Least common multiple (integer tensors only)
auto lcm(const Tensor& a, const Tensor& b) -> Tensor;

/// Fused add-multiply: input + value * tensor1 * tensor2
auto addcmul(const Tensor& input, const Tensor& tensor1, const Tensor& tensor2,
             double value = 1.0) -> Tensor;

/// Fused add-divide: input + value * tensor1 / tensor2
auto addcdiv(const Tensor& input, const Tensor& tensor1, const Tensor& tensor2,
             double value = 1.0) -> Tensor;

/// Lower regularized incomplete gamma function
auto igamma(const Tensor& a, const Tensor& x) -> Tensor;

/// Upper regularized incomplete gamma function (1 - igamma)
auto igammac(const Tensor& a, const Tensor& x) -> Tensor;

/** @} */ // end of tensor_math group

} // namespace tenzor
