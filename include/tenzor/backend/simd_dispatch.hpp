/**
 * @file simd_dispatch.hpp
 * @brief Runtime SIMD dispatch system for optimal kernel selection
 *
 * This module provides runtime CPU feature detection and automatic selection
 * of the best available SIMD implementation for critical operations.
 * Supports x86/x64 (AVX-512, AVX2, SSE4.2) and ARM (NEON).
 */

#ifndef TENZOR_BACKEND_SIMD_DISPATCH_HPP
#define TENZOR_BACKEND_SIMD_DISPATCH_HPP

#include <cstddef>
#include <functional>

namespace tenzor {
namespace backend {

/**
 * @brief Function pointer type for SIMD kernel implementations
 *
 * Kernel functions operate on contiguous arrays of float data.
 * @param dst Destination array
 * @param src1 First source array
 * @param src2 Second source array (may be nullptr for unary ops)
 * @param size Number of elements to process
 */
using KernelFunc = void (*)(float* dst, const float* src1, const float* src2, size_t size);

/**
 * @brief CPU Feature Detection
 *
 * These functions detect available SIMD instruction sets at runtime.
 * Results are cached after first call for performance.
 */

/**
 * @brief Check if CPU supports AVX-512 instructions
 * @return true if AVX-512F is available
 * @note x86/x64 only, always returns false on other architectures
 */
bool cpu_supports_avx512();

/**
 * @brief Check if CPU supports AVX2 instructions
 * @return true if AVX2 is available
 * @note x86/x64 only, always returns false on other architectures
 */
bool cpu_supports_avx2();

/**
 * @brief Check if CPU supports SSE4.2 instructions
 * @return true if SSE4.2 is available
 * @note x86/x64 only, always returns false on other architectures
 */
bool cpu_supports_sse42();

/**
 * @brief Check if CPU supports NEON instructions
 * @return true if NEON is available
 * @note ARM only, always returns false on other architectures
 */
bool cpu_supports_neon();

/**
 * @brief Get CPU feature summary string
 * @return Human-readable string describing available SIMD features
 */
const char* get_cpu_features();

/**
 * @brief Kernel Selection Functions
 *
 * These functions return the optimal kernel implementation based on
 * runtime CPU feature detection. The selection is performed once at
 * first call and cached for subsequent calls.
 */

/**
 * @brief Get optimal element-wise addition kernel
 * @return Function pointer to best available add implementation
 *
 * Selection priority (highest to lowest):
 * - AVX-512: 16 floats per operation
 * - AVX2: 8 floats per operation
 * - SSE4.2: 4 floats per operation
 * - NEON: 4 floats per operation (ARM)
 * - Scalar: 1 float per operation (fallback)
 */
KernelFunc get_optimal_add_kernel();

/**
 * @brief Get optimal element-wise multiplication kernel
 * @return Function pointer to best available multiply implementation
 *
 * Selection priority same as get_optimal_add_kernel()
 */
KernelFunc get_optimal_mul_kernel();

/**
 * @brief Get optimal matrix multiplication kernel
 * @return Function pointer to best available matmul implementation
 *
 * Matrix multiplication kernels use blocked algorithms optimized
 * for cache locality and SIMD vectorization.
 *
 * @note This returns a simplified kernel. For production matmul,
 *       use specialized BLAS libraries (MKL, OpenBLAS, etc.)
 */
KernelFunc get_optimal_matmul_kernel();

/**
 * @brief Get optimal ReLU activation kernel
 * @return Function pointer to best available ReLU implementation
 *
 * ReLU(x) = max(0, x)
 * Applies element-wise rectified linear unit activation.
 */
KernelFunc get_optimal_relu_kernel();

/**
 * @brief Get optimal Sigmoid activation kernel
 * @return Function pointer to best available sigmoid implementation
 *
 * Sigmoid(x) = 1 / (1 + exp(-x))
 * Applies element-wise sigmoid activation.
 */
KernelFunc get_optimal_sigmoid_kernel();

/**
 * @brief Get optimal Tanh activation kernel
 * @return Function pointer to best available tanh implementation
 *
 * Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 * Applies element-wise hyperbolic tangent activation.
 */
KernelFunc get_optimal_tanh_kernel();

/**
 * @brief Function pointer type for reduction operations
 *
 * Reduction operations compute a single value from an array.
 * @param src Source array
 * @param size Number of elements
 * @return Reduced scalar value
 */
using ReductionFunc = float (*)(const float* src, size_t size);

/**
 * @brief Get optimal sum reduction kernel
 * @return Function pointer to best available sum reduction implementation
 *
 * Computes the sum of all elements in the array.
 */
ReductionFunc get_optimal_reduce_sum_kernel();

/**
 * @brief Get optimal max reduction kernel
 * @return Function pointer to best available max reduction implementation
 *
 * Finds the maximum value in the array.
 */
ReductionFunc get_optimal_reduce_max_kernel();

/**
 * @brief Initialize SIMD dispatch system
 *
 * Performs CPU feature detection and initializes kernel function pointers.
 * Called automatically on first kernel request, but can be called explicitly
 * for eager initialization.
 *
 * Thread-safe and idempotent.
 */
void initialize_simd_dispatch();

/**
 * @brief SIMD Kernel Implementations
 *
 * Direct access to specific SIMD implementations.
 * Normally you should use get_optimal_*_kernel() instead.
 */
namespace kernels {

// Scalar fallback implementations
void add_scalar(float* dst, const float* src1, const float* src2, size_t size);
void mul_scalar(float* dst, const float* src1, const float* src2, size_t size);
void matmul_scalar(float* dst, const float* src1, const float* src2, size_t size);
void relu_scalar(float* dst, const float* src1, const float* src2, size_t size);
void sigmoid_scalar(float* dst, const float* src1, const float* src2, size_t size);
void tanh_scalar(float* dst, const float* src1, const float* src2, size_t size);
float reduce_sum_scalar(const float* src, size_t size);
float reduce_max_scalar(const float* src, size_t size);

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

// SSE4.2 implementations
void add_sse42(float* dst, const float* src1, const float* src2, size_t size);
void mul_sse42(float* dst, const float* src1, const float* src2, size_t size);
void matmul_sse42(float* dst, const float* src1, const float* src2, size_t size);
void relu_sse42(float* dst, const float* src1, const float* src2, size_t size);
void sigmoid_sse42(float* dst, const float* src1, const float* src2, size_t size);
void tanh_sse42(float* dst, const float* src1, const float* src2, size_t size);
float reduce_sum_sse42(const float* src, size_t size);
float reduce_max_sse42(const float* src, size_t size);

// AVX2 implementations
void add_avx2(float* dst, const float* src1, const float* src2, size_t size);
void mul_avx2(float* dst, const float* src1, const float* src2, size_t size);
void matmul_avx2(float* dst, const float* src1, const float* src2, size_t size);
void relu_avx2(float* dst, const float* src1, const float* src2, size_t size);
void sigmoid_avx2(float* dst, const float* src1, const float* src2, size_t size);
void tanh_avx2(float* dst, const float* src1, const float* src2, size_t size);
float reduce_sum_avx2(const float* src, size_t size);
float reduce_max_avx2(const float* src, size_t size);

// AVX-512 implementations
void add_avx512(float* dst, const float* src1, const float* src2, size_t size);
void mul_avx512(float* dst, const float* src1, const float* src2, size_t size);
void matmul_avx512(float* dst, const float* src1, const float* src2, size_t size);
void relu_avx512(float* dst, const float* src1, const float* src2, size_t size);
void sigmoid_avx512(float* dst, const float* src1, const float* src2, size_t size);
void tanh_avx512(float* dst, const float* src1, const float* src2, size_t size);
float reduce_sum_avx512(const float* src, size_t size);
float reduce_max_avx512(const float* src, size_t size);

#endif // x86/x64

#if defined(__ARM_NEON) || defined(__aarch64__)

// NEON implementations
void add_neon(float* dst, const float* src1, const float* src2, size_t size);
void mul_neon(float* dst, const float* src1, const float* src2, size_t size);
void matmul_neon(float* dst, const float* src1, const float* src2, size_t size);
void relu_neon(float* dst, const float* src1, const float* src2, size_t size);
void sigmoid_neon(float* dst, const float* src1, const float* src2, size_t size);
void tanh_neon(float* dst, const float* src1, const float* src2, size_t size);
float reduce_sum_neon(const float* src, size_t size);
float reduce_max_neon(const float* src, size_t size);

#endif // ARM NEON

} // namespace kernels

} // namespace backend
} // namespace tenzor

#endif // TENZOR_BACKEND_SIMD_DISPATCH_HPP
