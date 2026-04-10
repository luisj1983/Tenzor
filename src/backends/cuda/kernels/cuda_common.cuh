#pragma once

/**
 * @file cuda_common.cuh
 * @brief Shared CUDA error-checking and library-status macros.
 *
 * Consolidates CUDA_CHECK, CUBLAS_CHECK, CUSOLVER_CHECK, CUDNN_CHECK
 * and common grid-stride loop utilities into a single header for use
 * across all CUDA kernel files. Each macro includes __FILE__:__LINE__
 * context for easier debugging.
 *
 * All CUDA kernel files include this header instead of defining
 * local error-checking macros.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <string>

namespace tenzor {
namespace cuda {

// ============================================================================
// CUDA Runtime Error Checking
// ============================================================================

#ifndef TENZOR_CUDA_CHECK
#define TENZOR_CUDA_CHECK(call)                                                \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(                                          \
                std::string("CUDA error at ") + __FILE__ + ":" +              \
                std::to_string(__LINE__) + " - " + cudaGetErrorString(err));   \
        }                                                                      \
    } while (0)
#endif

/// Check for errors from the most recent kernel launch.
#ifndef TENZOR_CUDA_POST_LAUNCH_CHECK
#define TENZOR_CUDA_POST_LAUNCH_CHECK() TENZOR_CUDA_CHECK(cudaGetLastError())
#endif

// ============================================================================
// cuBLAS Error Checking
// ============================================================================

#ifndef TENZOR_CUBLAS_CHECK
#define TENZOR_CUBLAS_CHECK(call)                                              \
    do {                                                                        \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            throw std::runtime_error(                                          \
                std::string("cuBLAS error at ") + __FILE__ + ":" +            \
                std::to_string(__LINE__) + " - status code " +                \
                std::to_string(static_cast<int>(status)));                     \
        }                                                                      \
    } while (0)
#endif

// ============================================================================
// cuSOLVER Error Checking
// ============================================================================

#ifndef TENZOR_CUSOLVER_CHECK
#define TENZOR_CUSOLVER_CHECK(call)                                            \
    do {                                                                        \
        cusolverStatus_t status = (call);                                      \
        if (status != CUSOLVER_STATUS_SUCCESS) {                               \
            throw std::runtime_error(                                          \
                std::string("cuSOLVER error at ") + __FILE__ + ":" +          \
                std::to_string(__LINE__) + " - status code " +                \
                std::to_string(static_cast<int>(status)));                     \
        }                                                                      \
    } while (0)
#endif

// ============================================================================
// cuDNN Error Checking
// ============================================================================

#ifndef TENZOR_CUDNN_CHECK
#ifdef TENZOR_HAS_CUDNN
#include <cudnn.h>
#define TENZOR_CUDNN_CHECK(call)                                               \
    do {                                                                        \
        cudnnStatus_t status = (call);                                         \
        if (status != CUDNN_STATUS_SUCCESS) {                                  \
            throw std::runtime_error(                                          \
                std::string("cuDNN error at ") + __FILE__ + ":" +             \
                std::to_string(__LINE__) + " - " + cudnnGetErrorString(status));\
        }                                                                      \
    } while (0)
#endif
#endif

// ============================================================================
// Type Safety Guards
// ============================================================================

// Ensure int and int32_t are the same size — atomicAdd(int*, int) must be
// compatible with int32_t scatter_add operations.
static_assert(sizeof(int) == sizeof(int32_t),
              "atomicAdd requires int == int32_t on this platform");

// ============================================================================
// Grid-Stride Loop Macro
// ============================================================================

/// Standard grid-stride loop for 1-D element-wise kernels.
#ifndef TENZOR_CUDA_KERNEL_LOOP
#define TENZOR_CUDA_KERNEL_LOOP(i, n)                                          \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;                   \
         i < (n);                                                              \
         i += blockDim.x * gridDim.x)
#endif

/**
 * @brief Saturating float-to-half conversion.
 * Clamps finite input to the valid FP16 range [-65504, 65504] before
 * conversion, preventing overflow.  NaN and Inf are preserved.
 */
__device__ __forceinline__ __half float2half_sat(float x) {
    // Preserve NaN and Inf (isinf/isnan would be destroyed by the clamp)
    if (::isnan(x) || ::isinf(x)) {
        return __float2half(x);
    }
    constexpr float kHalfMax = 65504.0f;
    x = fminf(fmaxf(x, -kHalfMax), kHalfMax);
    return __float2half(x);
}

// ============================================================================
// Warp and Block Reduction Primitives
// ============================================================================

/// Warp-level sum reduction using shuffle instructions (full warp, 32 threads).
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/// Specialization for __half using native half-precision add.
template<>
__device__ __forceinline__ __half warp_reduce_sum(__half val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = __hadd(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/// Specialization for __nv_bfloat16 using native bfloat16 add.
template<>
__device__ __forceinline__ __nv_bfloat16 warp_reduce_sum(__nv_bfloat16 val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = __hadd(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/// Warp-level max reduction using shuffle instructions.
template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = (val > other) ? val : other;
    }
    return val;
}

/// Specialization for __half using intrinsic comparison.
template<>
__device__ __forceinline__ __half warp_reduce_max(__half val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        __half other = __shfl_down_sync(0xffffffff, val, offset);
        val = __hgt(val, other) ? val : other;
    }
    return val;
}

/// Specialization for __nv_bfloat16 using intrinsic comparison.
template<>
__device__ __forceinline__ __nv_bfloat16 warp_reduce_max(__nv_bfloat16 val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        __nv_bfloat16 other = __shfl_down_sync(0xffffffff, val, offset);
        val = __hgt(val, other) ? val : other;
    }
    return val;
}

/// Warp-level min reduction using shuffle instructions.
template<typename T>
__device__ __forceinline__ T warp_reduce_min(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = (val < other) ? val : other;
    }
    return val;
}

/// Specialization for __half using intrinsic comparison.
template<>
__device__ __forceinline__ __half warp_reduce_min(__half val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        __half other = __shfl_down_sync(0xffffffff, val, offset);
        val = __hlt(val, other) ? val : other;
    }
    return val;
}

/// Specialization for __nv_bfloat16 using intrinsic comparison.
template<>
__device__ __forceinline__ __nv_bfloat16 warp_reduce_min(__nv_bfloat16 val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        __nv_bfloat16 other = __shfl_down_sync(0xffffffff, val, offset);
        val = __hlt(val, other) ? val : other;
    }
    return val;
}

/**
 * @brief Block-level sum reduction using shared memory.
 * @param val    Per-thread input value
 * @param shared Shared memory array, must have at least (blockDim.x / 32) elements
 * @return       Reduced sum (valid only in thread 0)
 */
template<typename T>
__device__ T block_reduce_sum(T val, T* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[threadIdx.x] : T(0);
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

/**
 * @brief Block-level max reduction using shared memory.
 * @param val    Per-thread input value
 * @param shared Shared memory array, must have at least (blockDim.x / 32) elements
 * @param init   Identity element for max (e.g., -FLT_MAX)
 * @return       Reduced max (valid only in thread 0)
 */
template<typename T>
__device__ T block_reduce_max(T val, T* shared, T init) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[threadIdx.x] : init;
    if (wid == 0) {
        val = warp_reduce_max(val);
    }

    return val;
}

} // namespace cuda
} // namespace tenzor
