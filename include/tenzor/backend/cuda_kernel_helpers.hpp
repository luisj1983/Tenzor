/**
 * @file cuda_kernel_helpers.hpp
 * @brief CUDA kernel dispatch helpers for optimized stream extraction and dtype dispatch
 *
 * This header provides utilities to eliminate wrapper overhead in the CUDA backend:
 * - Optimized stream extraction with fast path for default stream
 * - Dtype dispatch macros to replace repetitive if-else chains
 * - Runtime cuDNN availability checking
 */

#pragma once

#include <charconv>
#include <stdexcept>
#include <atomic>
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/backend.hpp"

// Forward declare CUDA stream type to avoid including CUDA headers in this header
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

namespace tenzor::cuda {

// ============================================================================
// Optimized Stream Extraction
// ============================================================================

/**
 * @brief Extract CUDA stream from operation attributes with optimized fast path
 *
 * Most kernel calls use the default stream (nullptr). This function provides
 * a fast path that avoids hash lookups when attrs is empty or lacks a "stream" key.
 *
 * Performance: ~5ns for default stream vs ~50ns with hash lookup
 *
 * @param attrs Operation attributes map
 * @return CUDA stream handle (nullptr for default stream)
 */
inline cudaStream_t extract_stream(const OpAttributes& attrs) noexcept {
    // Fast path: most calls use default stream (empty attrs)
    if (attrs.empty()) [[likely]] {
        return nullptr;
    }

    // Check for stream key
    auto it = attrs.find("stream");
    if (it == attrs.end()) [[likely]] {
        return nullptr;
    }

    // Parse stream pointer from string (only reached for custom streams)
    uint64_t val = 0;
    auto [ptr, ec] = std::from_chars(
        it->second.data(),
        it->second.data() + it->second.size(),
        val
    );

    return (ec == std::errc{})
        ? reinterpret_cast<cudaStream_t>(val)
        : nullptr;
}

// ============================================================================
// Dtype Dispatch Macros
// ============================================================================

/**
 * @brief Dispatch to a templated kernel based on tensor dtype
 *
 * Replaces repetitive if-else chains with a single switch statement.
 * Usage: CUDA_DTYPE_DISPATCH(tensor.dtype(), kernel_template, args...)
 *
 * Example:
 * @code
 * template<typename T>
 * void launch_add(const T* a, const T* b, T* c, int64_t n, cudaStream_t stream);
 *
 * // Instead of 10+ if-else branches:
 * CUDA_DTYPE_DISPATCH(dtype, launch_add, a_ptr, b_ptr, c_ptr, n, stream);
 * @endcode
 */
#define CUDA_DTYPE_DISPATCH(dtype, kernel, ...)                                \
    do {                                                                        \
        switch (dtype) {                                                        \
            case DType::Float32:                                                \
                kernel<float>(__VA_ARGS__); break;                              \
            case DType::Float64:                                                \
                kernel<double>(__VA_ARGS__); break;                             \
            case DType::Int32:                                                  \
                kernel<int32_t>(__VA_ARGS__); break;                            \
            case DType::Int64:                                                  \
                kernel<int64_t>(__VA_ARGS__); break;                            \
            case DType::Int8:                                                   \
                kernel<int8_t>(__VA_ARGS__); break;                             \
            case DType::UInt8:                                                  \
                kernel<uint8_t>(__VA_ARGS__); break;                            \
            case DType::Int16:                                                  \
                kernel<int16_t>(__VA_ARGS__); break;                            \
            case DType::UInt16:                                                 \
                kernel<uint16_t>(__VA_ARGS__); break;                           \
            case DType::UInt32:                                                 \
                kernel<uint32_t>(__VA_ARGS__); break;                           \
            case DType::UInt64:                                                 \
                kernel<uint64_t>(__VA_ARGS__); break;                           \
            case DType::Bool:                                                   \
                kernel<bool>(__VA_ARGS__); break;                               \
            default:                                                            \
                throw std::runtime_error("Unsupported dtype for CUDA kernel");  \
        }                                                                       \
    } while (0)

/**
 * @brief Dispatch for floating-point types only (Float32, Float64, Float16, BFloat16)
 *
 * For operations that require floating-point (activations, reductions with division, etc.)
 * Float16/BFloat16 use specialized suffixed kernels (kernel_f16, kernel_bf16)
 */
#define CUDA_DTYPE_DISPATCH_FLOATING(dtype, kernel, ...)                       \
    do {                                                                        \
        switch (dtype) {                                                        \
            case DType::Float32:                                                \
                kernel<float>(__VA_ARGS__); break;                              \
            case DType::Float64:                                                \
                kernel<double>(__VA_ARGS__); break;                             \
            case DType::Float16:                                                \
                kernel##_f16(__VA_ARGS__); break;                               \
            case DType::BFloat16:                                               \
                kernel##_bf16(__VA_ARGS__); break;                              \
            default:                                                            \
                throw std::runtime_error("Floating-point dtype required");      \
        }                                                                       \
    } while (0)

/**
 * @brief Dispatch with return value for single-output kernels
 *
 * Similar to CUDA_DTYPE_DISPATCH but captures and returns the result.
 */
#define CUDA_DTYPE_DISPATCH_RETURN(dtype, result_var, kernel, ...)             \
    do {                                                                        \
        switch (dtype) {                                                        \
            case DType::Float32:                                                \
                result_var = kernel<float>(__VA_ARGS__); break;                 \
            case DType::Float64:                                                \
                result_var = kernel<double>(__VA_ARGS__); break;                \
            case DType::Int32:                                                  \
                result_var = kernel<int32_t>(__VA_ARGS__); break;               \
            case DType::Int64:                                                  \
                result_var = kernel<int64_t>(__VA_ARGS__); break;               \
            case DType::Int8:                                                   \
                result_var = kernel<int8_t>(__VA_ARGS__); break;                \
            case DType::UInt8:                                                  \
                result_var = kernel<uint8_t>(__VA_ARGS__); break;               \
            default:                                                            \
                throw std::runtime_error("Unsupported dtype for CUDA kernel");  \
        }                                                                       \
    } while (0)

// ============================================================================
// Runtime cuDNN Availability
// ============================================================================

/**
 * @brief Check if cuDNN is available at runtime
 *
 * Result is cached after first call for subsequent fast lookups.
 * This allows unified code paths instead of compile-time #ifdef blocks.
 *
 * @return true if cuDNN is linked and available
 */
bool is_cudnn_available() noexcept;

/**
 * @brief Check if cuDNN Frontend API is available (for SDPA/Flash Attention)
 *
 * @return true if cuDNN Frontend is available
 */
bool is_cudnn_frontend_available() noexcept;

} // namespace tenzor::cuda
