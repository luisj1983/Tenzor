/**
 * @file cuda_fp16.hpp
 * @brief CUDA-specific half-precision type conversions and utilities
 *
 * Provides conversion between tenzor::Float16/BFloat16 and CUDA native types
 * __half and __nv_bfloat16 for efficient GPU computation.
 */

#pragma once

#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

#include "tenzor/core/dtype.hpp"

namespace tenzor {
namespace cuda {

#ifdef __CUDACC__

// ============================================================================
// CUDA Float16 (__half) conversions
// ============================================================================

/**
 * @brief Convert tenzor::Float16 to CUDA __half
 */
__device__ __host__ inline auto to_cuda_half(Float16 f16) -> __half {
    __half_raw raw;
    raw.x = f16.bits;
    return __half(raw);
}

/**
 * @brief Convert CUDA __half to tenzor::Float16
 */
__device__ __host__ inline auto from_cuda_half(__half h) -> Float16 {
    __half_raw raw(h);
    return Float16(raw.x);
}

/**
 * @brief Convert float to CUDA __half
 */
__device__ __host__ inline auto float_to_half(float f) -> __half {
    return __float2half(f);
}

/**
 * @brief Convert CUDA __half to float
 */
__device__ __host__ inline auto half_to_float(__half h) -> float {
    return __half2float(h);
}

// ============================================================================
// CUDA BFloat16 (__nv_bfloat16) conversions
// ============================================================================

/**
 * @brief Convert tenzor::BFloat16 to CUDA __nv_bfloat16
 */
__device__ __host__ inline auto to_cuda_bfloat16(BFloat16 bf16) -> __nv_bfloat16 {
    __nv_bfloat16_raw raw;
    raw.x = bf16.bits;
    return __nv_bfloat16(raw);
}

/**
 * @brief Convert CUDA __nv_bfloat16 to tenzor::BFloat16
 */
__device__ __host__ inline auto from_cuda_bfloat16(__nv_bfloat16 bf) -> BFloat16 {
    __nv_bfloat16_raw raw(bf);
    return BFloat16(raw.x);
}

/**
 * @brief Convert float to CUDA __nv_bfloat16
 */
__device__ __host__ inline auto float_to_bfloat16(float f) -> __nv_bfloat16 {
    return __float2bfloat16(f);
}

/**
 * @brief Convert CUDA __nv_bfloat16 to float
 */
__device__ __host__ inline auto bfloat16_to_float(__nv_bfloat16 bf) -> float {
    return __bfloat162float(bf);
}

// ============================================================================
// Vectorized conversions for performance
// ============================================================================

/**
 * @brief Convert two Float16 to __half2 (vectorized)
 */
__device__ __host__ inline auto to_cuda_half2(Float16 f16_a, Float16 f16_b) -> __half2 {
    return __halves2half2(to_cuda_half(f16_a), to_cuda_half(f16_b));
}

/**
 * @brief Convert two BFloat16 to __nv_bfloat162 (vectorized)
 */
__device__ __host__ inline auto to_cuda_bfloat162(BFloat16 bf16_a, BFloat16 bf16_b) -> __nv_bfloat162 {
    return __halves2bfloat162(to_cuda_bfloat16(bf16_a), to_cuda_bfloat16(bf16_b));
}

// ============================================================================
// Arithmetic operations (use CUDA intrinsics)
// ============================================================================

/**
 * @brief Add two Float16 values using CUDA intrinsics
 */
__device__ inline auto add_fp16(Float16 a, Float16 b) -> Float16 {
    return from_cuda_half(__hadd(to_cuda_half(a), to_cuda_half(b)));
}

/**
 * @brief Multiply two Float16 values using CUDA intrinsics
 */
__device__ inline auto mul_fp16(Float16 a, Float16 b) -> Float16 {
    return from_cuda_half(__hmul(to_cuda_half(a), to_cuda_half(b)));
}

/**
 * @brief Add two BFloat16 values using CUDA intrinsics
 */
__device__ inline auto add_bf16(BFloat16 a, BFloat16 b) -> BFloat16 {
    return from_cuda_bfloat16(__hadd(to_cuda_bfloat16(a), to_cuda_bfloat16(b)));
}

/**
 * @brief Multiply two BFloat16 values using CUDA intrinsics
 */
__device__ inline auto mul_bf16(BFloat16 a, BFloat16 b) -> BFloat16 {
    return from_cuda_bfloat16(__hmul(to_cuda_bfloat16(a), to_cuda_bfloat16(b)));
}

// ============================================================================
// Tensor Core compatibility
// ============================================================================

/**
 * @brief Check if current GPU supports Tensor Core operations
 */
inline auto supports_tensor_cores() -> bool {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Tensor Cores require compute capability >= 7.0 (Volta+)
    return (prop.major >= 7);
}

/**
 * @brief Check if FP16 Tensor Cores are available
 */
inline auto supports_fp16_tensor_cores() -> bool {
    return supports_tensor_cores();  // Available since Volta (7.0+)
}

/**
 * @brief Check if BF16 Tensor Cores are available
 */
inline auto supports_bf16_tensor_cores() -> bool {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // BF16 Tensor Cores require compute capability >= 8.0 (Ampere+)
    return (prop.major >= 8);
}

#endif // __CUDACC__

} // namespace cuda
} // namespace tenzor
