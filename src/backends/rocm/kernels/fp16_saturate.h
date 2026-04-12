/**
 * @file fp16_saturate.h
 * @brief FP16 saturation utility for ROCm kernels.
 *
 * Clamps ±Inf values to ±65504 (max finite Float16) to prevent NaN propagation
 * in deep networks. This matches the CUDA backend's fp16_saturate behavior.
 *
 * IEEE 754 FP32→FP16 conversion produces ±Inf for values outside [-65504, 65504].
 * When Inf interacts with other values (e.g., Inf - Inf, 0 * Inf), NaN results.
 * Saturating to finite bounds prevents this cascade.
 */
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "tenzor/core/dtype.hpp"

namespace tenzor {
namespace rocm {

static __global__ void fp16_saturate_kernel(__half* data, int64_t n) {
    constexpr float kHalfMax = 65504.0f;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += blockDim.x * gridDim.x) {
        float val = __half2float(data[idx]);
        if (val > kHalfMax || val < -kHalfMax) {
            data[idx] = __float2half(fminf(fmaxf(val, -kHalfMax), kHalfMax));
        }
    }
}

inline void fp16_saturate(void* data, int64_t n, hipStream_t stream = 0) {
    if (n <= 0) return;
    constexpr int kBlock = 256;
    int grid = static_cast<int>((n + kBlock - 1) / kBlock);
    hipLaunchKernelGGL(fp16_saturate_kernel,
        dim3(grid), dim3(kBlock), 0, stream,
        reinterpret_cast<__half*>(data), n);
}

} // namespace rocm
} // namespace tenzor
