/**
 * @file simd.hpp
 * @brief SIMD optimizations for CPU backend with runtime feature detection
 *
 * Provides vectorized implementations of common operations using:
 * - AVX-512 (512-bit vectors, 16 floats)
 * - AVX2 (256-bit vectors, 8 floats)
 * - SSE4.2 (128-bit vectors, 4 floats)
 * - Scalar fallback
 *
 * Features automatic runtime CPU detection and dispatch to best available implementation.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace tenzor {
namespace cpu {

// Legacy CPU feature detection (CPUInfo / CPUFeature) was removed in
// Stream 17 of the release-prep audit. Use `tenzor::backend::SIMDFeatures`
// / `tenzor::backend::get_simd_features()` from
// `tenzor/backend/runtime_simd.hpp` instead — that API performs the OS-level
// XCR0 check needed to avoid SIGILL on CPUs whose silicon supports AVX-512
// but whose kernel does not save ZMM register state.

/**
 * @brief SIMD dispatch for vectorized operations
 *
 * Provides runtime dispatch to best available SIMD implementation
 * based on detected CPU features.
 */
namespace simd {

/**
 * @brief Vectorized element-wise add: out[i] = a[i] + b[i]
 *
 * @param a First input array
 * @param b Second input array
 * @param out Output array
 * @param size Number of elements
 */
auto add(const float* a, const float* b, float* out, size_t size) -> void;

/**
 * @brief Vectorized element-wise subtract: out[i] = a[i] - b[i]
 */
auto sub(const float* a, const float* b, float* out, size_t size) -> void;

/**
 * @brief Vectorized element-wise multiply: out[i] = a[i] * b[i]
 */
auto mul(const float* a, const float* b, float* out, size_t size) -> void;

/**
 * @brief Vectorized element-wise divide: out[i] = a[i] / b[i]
 */
auto div(const float* a, const float* b, float* out, size_t size) -> void;

/**
 * @brief Vectorized square root: out[i] = sqrt(a[i])
 */
auto sqrt(const float* a, float* out, size_t size) -> void;

/**
 * @brief Vectorized exponential: out[i] = exp(a[i])
 */
auto exp(const float* a, float* out, size_t size) -> void;

/**
 * @brief Vectorized natural logarithm: out[i] = log(a[i])
 */
auto log(const float* a, float* out, size_t size) -> void;

/**
 * @brief Vectorized ReLU: out[i] = max(0, a[i])
 */
auto relu(const float* a, float* out, size_t size) -> void;

/**
 * @brief Vectorized sigmoid: out[i] = 1 / (1 + exp(-a[i]))
 */
auto sigmoid(const float* a, float* out, size_t size) -> void;

/**
 * @brief Vectorized tanh: out[i] = tanh(a[i])
 */
auto tanh(const float* a, float* out, size_t size) -> void;

/**
 * @brief Vectorized GELU: out[i] = a[i] * Phi(a[i])
 *
 * where Phi(x) is the cumulative distribution function of the standard normal distribution.
 * Uses approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 */
auto gelu(const float* a, float* out, size_t size) -> void;

/**
 * @brief Vectorized fused multiply-add: out[i] = a[i] * b[i] + c[i]
 */
auto fma(const float* a, const float* b, const float* c, float* out, size_t size) -> void;

} // namespace simd

/**
 * @brief Scalar implementations (fallback)
 */
namespace scalar {

auto add(const float* a, const float* b, float* out, size_t size) -> void;
auto sub(const float* a, const float* b, float* out, size_t size) -> void;
auto mul(const float* a, const float* b, float* out, size_t size) -> void;
auto div(const float* a, const float* b, float* out, size_t size) -> void;
auto sqrt(const float* a, float* out, size_t size) -> void;
auto exp(const float* a, float* out, size_t size) -> void;
auto log(const float* a, float* out, size_t size) -> void;
auto relu(const float* a, float* out, size_t size) -> void;
auto sigmoid(const float* a, float* out, size_t size) -> void;
auto tanh(const float* a, float* out, size_t size) -> void;
auto gelu(const float* a, float* out, size_t size) -> void;
auto fma(const float* a, const float* b, const float* c, float* out, size_t size) -> void;

} // namespace scalar

/**
 * @brief AVX2 implementations (256-bit vectors)
 */
namespace avx2 {

auto add(const float* a, const float* b, float* out, size_t size) -> void;
auto sub(const float* a, const float* b, float* out, size_t size) -> void;
auto mul(const float* a, const float* b, float* out, size_t size) -> void;
auto div(const float* a, const float* b, float* out, size_t size) -> void;
auto sqrt(const float* a, float* out, size_t size) -> void;
auto exp(const float* a, float* out, size_t size) -> void;
auto log(const float* a, float* out, size_t size) -> void;
auto relu(const float* a, float* out, size_t size) -> void;
auto sigmoid(const float* a, float* out, size_t size) -> void;
auto tanh(const float* a, float* out, size_t size) -> void;
auto gelu(const float* a, float* out, size_t size) -> void;
auto fma(const float* a, const float* b, const float* c, float* out, size_t size) -> void;

} // namespace avx2

/**
 * @brief AVX-512 implementations (512-bit vectors)
 */
namespace avx512 {

auto add(const float* a, const float* b, float* out, size_t size) -> void;
auto sub(const float* a, const float* b, float* out, size_t size) -> void;
auto mul(const float* a, const float* b, float* out, size_t size) -> void;
auto div(const float* a, const float* b, float* out, size_t size) -> void;
auto sqrt(const float* a, float* out, size_t size) -> void;
auto exp(const float* a, float* out, size_t size) -> void;
auto log(const float* a, float* out, size_t size) -> void;
auto relu(const float* a, float* out, size_t size) -> void;
auto sigmoid(const float* a, float* out, size_t size) -> void;
auto tanh(const float* a, float* out, size_t size) -> void;
auto gelu(const float* a, float* out, size_t size) -> void;
auto fma(const float* a, const float* b, const float* c, float* out, size_t size) -> void;

} // namespace avx512

} // namespace cpu
} // namespace tenzor
