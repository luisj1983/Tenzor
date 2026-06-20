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
// Accumulator-precision selection
// ============================================================================

/**
 * @brief Maps an element type to the accumulator type used by the sum reduction.
 *
 * The CPU reference (src/backends/cpu/kernels/reduction.cpp simd_sum_f32_avx512)
 * accumulates float32 sums with Kahan compensation to avoid losing ULPs over
 * long sequences. The GPU reduction is tree-structured across and within blocks,
 * but the per-thread grid-stride loop and the cross-block partial sums are still
 * a sequential, uncompensated accumulation. To match the CPU contract (and keep
 * cross-backend parity) we accumulate the float32 reduction in double precision
 * end to end, which removes the ULP drift without the divergence cost of an
 * in-kernel Kahan loop on the GPU.
 *
 * Integer and double element types keep their own type as the accumulator: a
 * double accumulator would silently lose precision for large Int64/UInt64
 * values (>2^53), so those paths must accumulate exactly in their native type.
 */
template<typename T>
struct reduction_accum { using type = T; };

template<>
struct reduction_accum<float> { using type = double; };

template<typename T>
using reduction_accum_t = typename reduction_accum<T>::type;

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
 *
 * @tparam InT  Element type of the input buffer.
 * @tparam AccT Accumulator type. The grid-stride loop, LDS tree, and wavefront
 *              reduction all run in AccT, which the launch helper selects as a
 *              higher-precision type for float32 input (double; see
 *              reduction_accum) so the per-thread and cross-block partial sums
 *              are not prematurely rounded.
 * @tparam OutT Element type of the output buffer. Defaults to AccT (used for the
 *              high-precision cross-block scratch buffer); the final stage sets
 *              OutT to the requested element type to narrow the result on store.
 */
template<typename InT, typename AccT, typename OutT = AccT>
__global__ void sum_reduce_kernel(const InT* input, OutT* output, int64_t n) {
    __shared__ AccT shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Grid-stride loop for better occupancy on AMD GPUs. Accumulate in AccT,
    // which is a higher-precision type for float32 input (see reduction_accum).
    AccT thread_sum = 0;
    for (int64_t i = idx; i < n; i += grid_size) {
        thread_sum += static_cast<AccT>(input[i]);
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
        AccT val = shared[tid];
        val = wavefront_reduce_sum(val);

        if (tid == 0) {
            // Narrow to the output element type on store (no-op when OutT == AccT).
            output[blockIdx.x] = static_cast<OutT>(val);
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
    // Accumulator type: double for float32 (matches the Kahan-compensated CPU
    // reference contract), the element type itself for everything else (integer
    // types must accumulate exactly; double would lose precision above 2^53).
    using Acc = reduction_accum_t<T>;

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
        // Single block: accumulate in Acc, narrow to T on the final store.
        hipLaunchKernelGGL((sum_reduce_kernel<T, Acc, T>), dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_input, d_output, n);
    } else {
        // Phase 1: Reduce to num_blocks intermediate results.
        // Use a stream-ordered scratch allocation so the temp buffer's lifetime
        // is tied to the stream and no host barrier is required. This keeps the
        // reduction safe to capture into a HIP graph (no hipMalloc/hipFree and no
        // host-device synchronization during capture).
        //
        // The scratch buffer holds the high-precision accumulator type (Acc) so
        // the cross-block partial sums are not rounded to T between phases.
        Acc* d_temp = nullptr;
#if HIP_VERSION >= 50300000
        REDUCTION_UTILS_HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&d_temp), num_blocks * sizeof(Acc), stream));
#else
        REDUCTION_UTILS_HIP_CHECK(hipMalloc(&d_temp, num_blocks * sizeof(Acc)));
#endif
        // Phase 1: const T* -> Acc* (accumulate and store in Acc precision).
        hipLaunchKernelGGL((sum_reduce_kernel<T, Acc, Acc>), dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_input, d_temp, n);

        // Phase 2: const Acc* -> T* (accumulate in Acc, narrow to T on store).
        hipLaunchKernelGGL((sum_reduce_kernel<Acc, Acc, T>), dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_temp, d_output, static_cast<int64_t>(num_blocks));

#if HIP_VERSION >= 50300000
        REDUCTION_UTILS_HIP_CHECK(hipFreeAsync(d_temp, stream));
#else
        REDUCTION_UTILS_HIP_CHECK(hipStreamSynchronize(stream));
        REDUCTION_UTILS_HIP_CHECK(hipFree(d_temp));
#endif
    }
}

/**
 * @brief Full tensor mean reduction on GPU (sum / n).
 */
template<typename T>
inline void launch_full_reduction_mean(const T* d_input, T* d_output, int64_t n, hipStream_t stream) {
    launch_full_reduction_sum(d_input, d_output, n, stream);
    if (n > 0) {
        // Divide kernel is stream-ordered; no host barrier needed (keeps the
        // path HIP-graph capture-safe).
        hipLaunchKernelGGL(divide_scalar_kernel<T>, dim3(1), dim3(1), 0, stream,
            d_output, static_cast<T>(n));
    }
}

} // namespace rocm
} // namespace tenzor
