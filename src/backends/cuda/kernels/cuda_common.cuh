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

} // namespace cuda
} // namespace tenzor
