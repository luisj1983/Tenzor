#pragma once

// ARM NEON quantized (INT8) kernels for dot product and GEMV.
// Uses native vdotq_s32 on ARMv8.4+ or widening-multiply fallback.

#ifdef __ARM_NEON
#include <arm_neon.h>
#include <cstdint>

namespace tenzor::cpu::neon {

// ============================================================================
// INT8 dot product: accumulate a[i]*b[i] into int32
// ============================================================================

inline int32_t qdot_s8(const int8_t* a, const int8_t* b, int64_t n) {
    int32x4_t acc = vdupq_n_s32(0);
    int64_t i = 0;

#if defined(__ARM_FEATURE_DOTPROD)
    // Native 4-way dot product instruction (ARMv8.4-A / ARMv8.2+DotProd)
    for (; i + 15 < n; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);
        acc = vdotq_s32(acc, va, vb);
    }
#else
    // Fallback: widening multiply to int16, pairwise add to int32
    for (; i + 15 < n; i += 16) {
        int8x8_t a_lo = vld1_s8(a + i);
        int8x8_t a_hi = vld1_s8(a + i + 8);
        int8x8_t b_lo = vld1_s8(b + i);
        int8x8_t b_hi = vld1_s8(b + i + 8);

        // Widen multiply: int8 x int8 -> int16
        int16x8_t prod_lo = vmull_s8(a_lo, b_lo);
        int16x8_t prod_hi = vmull_s8(a_hi, b_hi);

        // Pairwise add int16 -> int32
        int32x4_t sum_lo = vpaddlq_s16(prod_lo);
        int32x4_t sum_hi = vpaddlq_s16(prod_hi);

        acc = vaddq_s32(acc, sum_lo);
        acc = vaddq_s32(acc, sum_hi);
    }
    // Handle 8-element chunk
    if (i + 7 < n) {
        int8x8_t va = vld1_s8(a + i);
        int8x8_t vb = vld1_s8(b + i);
        int16x8_t prod = vmull_s8(va, vb);
        acc = vaddq_s32(acc, vpaddlq_s16(prod));
        i += 8;
    }
#endif

    // Horizontal sum of accumulator
    int32_t result = vaddvq_s32(acc);

    // Scalar tail
    for (; i < n; ++i) {
        result += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    return result;
}

// ============================================================================
// Quantized INT8 GEMV: y[m] = sum_k(A[m][k] * x[k])
// A is [M x K] row-major int8, x is [K] int8, y is [M] int32
// ============================================================================

inline void qgemv_s8(const int8_t* A, const int8_t* x, int32_t* y,
                     int64_t M, int64_t K) {
    for (int64_t m = 0; m < M; ++m) {
        y[m] = qdot_s8(A + m * K, x, K);
    }
}

} // namespace tenzor::cpu::neon

#endif // __ARM_NEON
