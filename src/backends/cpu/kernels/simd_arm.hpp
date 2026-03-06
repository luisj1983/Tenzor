#pragma once

// ARM NEON SIMD intrinsics for element-wise ops, reductions, and FMA.
// Used alongside x86 SSE2/AVX2/AVX512 paths (guarded by __ARM_NEON).

#ifdef __ARM_NEON
#include <arm_neon.h>

namespace tenzor::cpu::neon {

// ============================================================================
// Element-wise operations (float32, 4-wide)
// ============================================================================

inline void add_f32(const float* a, const float* b, float* out, int64_t n) {
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(va, vb));
    }
    for (; i < n; ++i) out[i] = a[i] + b[i];
}

inline void mul_f32(const float* a, const float* b, float* out, int64_t n) {
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vmulq_f32(va, vb));
    }
    for (; i < n; ++i) out[i] = a[i] * b[i];
}

inline void sub_f32(const float* a, const float* b, float* out, int64_t n) {
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vsubq_f32(va, vb));
    }
    for (; i < n; ++i) out[i] = a[i] - b[i];
}

// ============================================================================
// Activation functions (ReLU, approximate GELU via polynomial)
// ============================================================================

inline void relu_f32(const float* in, float* out, int64_t n) {
    int64_t i = 0;
    float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        vst1q_f32(out + i, vmaxq_f32(v, zero));
    }
    for (; i < n; ++i) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

// ============================================================================
// Reductions (sum, max)
// ============================================================================

inline float sum_f32(const float* data, int64_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        acc = vaddq_f32(acc, vld1q_f32(data + i));
    }
    // Horizontal add: sum all 4 lanes
    float result = vaddvq_f32(acc);
    for (; i < n; ++i) result += data[i];
    return result;
}

inline float max_f32(const float* data, int64_t n) {
    if (n == 0) return -__FLT_MAX__;
    float32x4_t vmax = vdupq_n_f32(data[0]);
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        vmax = vmaxq_f32(vmax, vld1q_f32(data + i));
    }
    float result = vmaxvq_f32(vmax);
    for (; i < n; ++i) {
        if (data[i] > result) result = data[i];
    }
    return result;
}

// ============================================================================
// FMA for matmul inner loop: out[i] += a[i] * b[i]
// ============================================================================

inline void fma_f32(const float* a, const float* b, float* out, int64_t n) {
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vo = vld1q_f32(out + i);
        vst1q_f32(out + i, vfmaq_f32(vo, va, vb));
    }
    for (; i < n; ++i) out[i] += a[i] * b[i];
}

// Dot product: sum(a[i] * b[i])
inline float dot_f32(const float* a, const float* b, int64_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float result = vaddvq_f32(acc);
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}

} // namespace tenzor::cpu::neon

#endif // __ARM_NEON
