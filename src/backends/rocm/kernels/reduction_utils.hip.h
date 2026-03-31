/**
 * @file reduction_utils.hip.h
 * @brief Shared GPU reduction primitives for ROCm backend
 *
 * Provides wavefront-level and block-level reduction kernels that can be
 * reused across multiple ROCm kernel translation units (e.g., reduction,
 * fused_ops, normalization).
 */

#pragma once

#include <hip/hip_runtime.h>
#include <algorithm>
#include <stdexcept>

namespace tenzor {
namespace rocm {

// ============================================================================
// Constants
// ============================================================================

#ifndef REDUCTION_BLOCK_SIZE
#define REDUCTION_BLOCK_SIZE 256
#endif

// ============================================================================
// Wavefront-level reduction primitives
// ============================================================================

/**
 * @brief Wavefront-level sum reduction using AMD GPU warp shuffle
 * Uses __shfl_down for efficient intra-wavefront communication.
 * Handles both wave32 (RDNA) and wave64 (CDNA) via warpSize built-in.
 */
template<typename T>
__device__ __forceinline__ T wavefront_reduce_sum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// ============================================================================
// Block-level sum reduction kernel
// ============================================================================

/**
 * @brief Sum reduction kernel using LDS (Local Data Share)
 *
 * Two-level reduction:
 * 1. Grid-stride loop accumulates values per thread
 * 2. Block-level reduction in LDS
 * 3. Wavefront-level reduction for final sum
 */
template<typename T>
__global__ void sum_reduce_kernel(const T* input, T* output, int64_t n) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Grid-stride loop for better occupancy on AMD GPUs
    T thread_sum = 0;
    for (int64_t i = idx; i < n; i += grid_size) {
        thread_sum += input[i];
    }

    shared[tid] = thread_sum;
    __syncthreads();

    // Block-level reduction in LDS
    for (int stride = blockDim.x / 2; stride >= warpSize; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // Wavefront-level reduction for final stages
    if (tid < warpSize) {
        T val = shared[tid];
        val = wavefront_reduce_sum(val);

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// ============================================================================
// Single-element division kernel (for mean computation)
// ============================================================================

template<typename T>
__global__ void divide_scalar_kernel(T* val, T divisor) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *val /= divisor;
    }
}

// ============================================================================
// Launch helpers
// ============================================================================

#define REDUCTION_UTILS_HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
    } \
} while(0)

/**
 * @brief Two-phase full tensor sum reduction on GPU.
 * Phase 1: Reduce to one value per block.
 * Phase 2: Reduce block results to a single scalar.
 */
template<typename T>
inline void launch_full_reduction_sum(const T* d_input, T* d_output, int64_t n, hipStream_t stream) {
    if (n == 0) {
        T zero = 0;
        REDUCTION_UTILS_HIP_CHECK(hipMemcpyAsync(d_output, &zero, sizeof(T), hipMemcpyHostToDevice, stream));
        return;
    }

    if (n == 1) {
        REDUCTION_UTILS_HIP_CHECK(hipMemcpyAsync(d_output, d_input, sizeof(T), hipMemcpyDeviceToDevice, stream));
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        hipLaunchKernelGGL(sum_reduce_kernel<T>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_input, d_output, n);
    } else {
        // Phase 1: Reduce to num_blocks intermediate results
        T* d_temp;
        REDUCTION_UTILS_HIP_CHECK(hipMalloc(&d_temp, num_blocks * sizeof(T)));
        hipLaunchKernelGGL(sum_reduce_kernel<T>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_input, d_temp, n);

        // Phase 2: Final reduction
        hipLaunchKernelGGL(sum_reduce_kernel<T>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_temp, d_output, num_blocks);

        REDUCTION_UTILS_HIP_CHECK(hipStreamSynchronize(stream));
        REDUCTION_UTILS_HIP_CHECK(hipFree(d_temp));
    }
    REDUCTION_UTILS_HIP_CHECK(hipStreamSynchronize(stream));
}

/**
 * @brief Full tensor mean reduction on GPU (sum / n).
 */
template<typename T>
inline void launch_full_reduction_mean(const T* d_input, T* d_output, int64_t n, hipStream_t stream) {
    launch_full_reduction_sum(d_input, d_output, n, stream);
    if (n > 0) {
        hipLaunchKernelGGL(divide_scalar_kernel<T>, dim3(1), dim3(1), 0, stream,
            d_output, static_cast<T>(n));
        REDUCTION_UTILS_HIP_CHECK(hipStreamSynchronize(stream));
    }
}

} // namespace rocm
} // namespace tenzor
