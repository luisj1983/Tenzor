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

#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "tenzor/core/dtype.hpp"

namespace tenzor {
namespace rocm {

// Y.15: NaN-preserving FP16 saturation.
//
// Per feedback_rocm_intrinsic_nan.md, __half2float(NaN_F16) can silently
// canonicalise NaN to 0 / +Inf on certain ROCm builds under fast-math.
// Inspect the raw F16 bit pattern (IEEE-754: exp == 0x1F and mantissa != 0
// is NaN) BEFORE converting to float; on NaN, leave the F16 bits untouched
// so the NaN payload propagates intact through downstream ops.
//
// Matches the bit-pattern approach in rocm_nan_helpers.hip.h::is_nan_bits.
static __global__ void fp16_saturate_kernel(__half* data, int64_t n) {
    constexpr float kHalfMax = 65504.0f;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += blockDim.x * gridDim.x) {
        // Read raw F16 bits via __half_raw (canonical HIP layout: ushort x).
        // Avoid `__half2float` for the NaN-detection path because some ROCm
        // builds canonicalise NaN to 0/Inf on cast.
        uint16_t bits = *reinterpret_cast<const uint16_t*>(&data[idx]);
        uint16_t exp  = (bits >> 10) & 0x1Fu;
        uint16_t mant =  bits        & 0x3FFu;
        bool is_nan = (exp == 0x1Fu) && (mant != 0u);
        if (is_nan) {
            // Preserve NaN bits untouched — do not clamp.
            continue;
        }
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
