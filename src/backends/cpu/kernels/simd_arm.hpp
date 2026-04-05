#pragma once

// ARM NEON SIMD intrinsics for element-wise ops, reductions, and FMA.
// Used alongside x86 SSE2/AVX2/AVX512 paths (guarded by __ARM_NEON).

#ifdef __ARM_NEON
#include <arm_neon.h>
#include <cmath>

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

// ============================================================================
// Element-wise unary operations (float32, 4-wide)
// ============================================================================

inline void neg_f32(const float* src, float* dst, int64_t n) {
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        vst1q_f32(dst + i, vnegq_f32(v));
    }
    for (; i < n; ++i) dst[i] = -src[i];
}

inline void abs_f32(const float* src, float* dst, int64_t n) {
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        vst1q_f32(dst + i, vabsq_f32(v));
    }
    for (; i < n; ++i) dst[i] = src[i] < 0.0f ? -src[i] : src[i];
}

// ============================================================================
// Reduction: min (float32)
// ============================================================================

inline float min_f32(const float* data, int64_t n) {
    if (n == 0) return __FLT_MAX__;
    float32x4_t vmin = vdupq_n_f32(data[0]);
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        vmin = vminq_f32(vmin, vld1q_f32(data + i));
    }
    float result = vminvq_f32(vmin);
    for (; i < n; ++i) {
        if (data[i] < result) result = data[i];
    }
    return result;
}

// ============================================================================
// NEON exp(x) approximation (Cephes-style polynomial, float32x4_t)
//
// Clamps input to [-87, 87] to avoid overflow/underflow, then computes:
//   exp(x) = 2^n * (1 + polynomial(r))
// where x = n*ln(2) + r, with |r| <= ln(2)/2.
// ============================================================================

namespace detail {

inline float32x4_t exp_f32_neon(float32x4_t x) {
    const float32x4_t c_min   = vdupq_n_f32(-87.3365f);
    const float32x4_t c_max   = vdupq_n_f32(88.3762f);
    const float32x4_t c_log2e = vdupq_n_f32(1.44269504088896341f);
    const float32x4_t c_ln2_h = vdupq_n_f32(0.693359375f);
    const float32x4_t c_ln2_l = vdupq_n_f32(-2.12194440e-4f);
    const float32x4_t c_one   = vdupq_n_f32(1.0f);

    // Cephes polynomial coefficients for exp on [-0.5*ln2, 0.5*ln2]
    const float32x4_t p0 = vdupq_n_f32(1.9875691500e-4f);
    const float32x4_t p1 = vdupq_n_f32(1.3981999507e-3f);
    const float32x4_t p2 = vdupq_n_f32(8.3334519073e-3f);
    const float32x4_t p3 = vdupq_n_f32(4.1665795894e-2f);
    const float32x4_t p4 = vdupq_n_f32(1.6666665459e-1f);
    const float32x4_t p5 = vdupq_n_f32(5.0000001201e-1f);

    // Clamp input
    x = vmaxq_f32(vminq_f32(x, c_max), c_min);

    // Express exp(x) = exp(n*ln2 + r) = 2^n * exp(r)
    // n = round(x / ln2)
    float32x4_t n = vrndnq_f32(vmulq_f32(x, c_log2e));

    // r = x - n * ln2  (Cahan reduction for precision)
    float32x4_t r = vfmsq_f32(x, n, c_ln2_h);
    r = vfmsq_f32(r, n, c_ln2_l);

    // Polynomial approximation of exp(r) - 1
    // Using Horner's method: ((((p0*r + p1)*r + p2)*r + p3)*r + p4)*r + p5
    float32x4_t y = vfmaq_f32(p1, p0, r);
    y = vfmaq_f32(p2, y, r);
    y = vfmaq_f32(p3, y, r);
    y = vfmaq_f32(p4, y, r);
    y = vfmaq_f32(p5, y, r);
    y = vfmaq_f32(c_one, y, vmulq_f32(r, r));

    // Construct 2^n by bit-shifting the integer n into the exponent field
    int32x4_t ni = vcvtq_s32_f32(n);
    ni = vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(ni);

    return vmulq_f32(y, pow2n);
}

} // namespace detail

// ============================================================================
// Activation functions: sigmoid, tanh, GELU, SiLU (float32, 4-wide)
// ============================================================================

inline void sigmoid_f32(const float* src, float* dst, int64_t n) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(src + i);
        // sigmoid(x) = 1 / (1 + exp(-x))
        float32x4_t exp_neg_x = detail::exp_f32_neon(vnegq_f32(x));
        float32x4_t denom = vaddq_f32(one, exp_neg_x);
        // Approximate reciprocal with Newton-Raphson refinement
        float32x4_t recip = vrecpeq_f32(denom);
        recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));
        recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));
        vst1q_f32(dst + i, recip);
    }
    for (; i < n; ++i) {
        float ex = std::exp(-src[i]);
        dst[i] = 1.0f / (1.0f + ex);
    }
}

inline void tanh_f32(const float* src, float* dst, int64_t n) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    const float32x4_t two = vdupq_n_f32(2.0f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(src + i);
        float32x4_t x2 = vmulq_f32(two, x);
        // sigmoid(2x)
        float32x4_t exp_neg = detail::exp_f32_neon(vnegq_f32(x2));
        float32x4_t denom = vaddq_f32(one, exp_neg);
        float32x4_t recip = vrecpeq_f32(denom);
        recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));
        recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));
        // 2 * sigmoid(2x) - 1
        float32x4_t result = vsubq_f32(vmulq_f32(two, recip), one);
        vst1q_f32(dst + i, result);
    }
    for (; i < n; ++i) {
        float s = 1.0f / (1.0f + std::exp(-2.0f * src[i]));
        dst[i] = 2.0f * s - 1.0f;
    }
}

inline void gelu_f32(const float* src, float* dst, int64_t n) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float32x4_t half    = vdupq_n_f32(0.5f);
    const float32x4_t one     = vdupq_n_f32(1.0f);
    const float32x4_t two     = vdupq_n_f32(2.0f);
    const float32x4_t coeff   = vdupq_n_f32(0.044715f);
    const float32x4_t sqrt2pi = vdupq_n_f32(0.7978845608028654f); // sqrt(2/pi)

    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(src + i);

        // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
        float32x4_t x3 = vmulq_f32(x, vmulq_f32(x, x));
        float32x4_t inner = vmulq_f32(sqrt2pi, vfmaq_f32(x, coeff, x3));

        // tanh(inner) via 2*sigmoid(2*inner) - 1
        float32x4_t inner2 = vmulq_f32(two, inner);
        float32x4_t exp_neg = detail::exp_f32_neon(vnegq_f32(inner2));
        float32x4_t denom = vaddq_f32(one, exp_neg);
        float32x4_t recip = vrecpeq_f32(denom);
        recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));
        recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));
        float32x4_t tanh_val = vsubq_f32(vmulq_f32(two, recip), one);

        // 0.5 * x * (1 + tanh_val)
        float32x4_t result = vmulq_f32(half, vmulq_f32(x, vaddq_f32(one, tanh_val)));
        vst1q_f32(dst + i, result);
    }
    for (; i < n; ++i) {
        float x = src[i];
        float x3 = x * x * x;
        float inner = 0.7978845608028654f * (x + 0.044715f * x3);
        float s = 1.0f / (1.0f + std::exp(-2.0f * inner));
        float t = 2.0f * s - 1.0f;
        dst[i] = 0.5f * x * (1.0f + t);
    }
}

inline void silu_f32(const float* src, float* dst, int64_t n) {
    // SiLU(x) = x * sigmoid(x)
    const float32x4_t one = vdupq_n_f32(1.0f);
    int64_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(src + i);
        float32x4_t exp_neg_x = detail::exp_f32_neon(vnegq_f32(x));
        float32x4_t denom = vaddq_f32(one, exp_neg_x);
        float32x4_t recip = vrecpeq_f32(denom);
        recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));
        recip = vmulq_f32(recip, vrecpsq_f32(denom, recip));
        vst1q_f32(dst + i, vmulq_f32(x, recip));
    }
    for (; i < n; ++i) {
        float s = 1.0f / (1.0f + std::exp(-src[i]));
        dst[i] = src[i] * s;
    }
}

// ============================================================================
// Matrix operations: GEMV, GEMM micro-kernel (float32)
// ============================================================================

/// Matrix-vector multiply: y = A * x, where A is [M x N] row-major.
inline void gemv_f32(const float* A, const float* x, float* y,
                     int64_t M, int64_t N) {
    int64_t m = 0;
    // Process 4 rows at a time, each accumulating a dot product over N
    for (; m + 3 < M; m += 4) {
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        const float* row0 = A + (m + 0) * N;
        const float* row1 = A + (m + 1) * N;
        const float* row2 = A + (m + 2) * N;
        const float* row3 = A + (m + 3) * N;

        int64_t k = 0;
        for (; k + 3 < N; k += 4) {
            float32x4_t vx = vld1q_f32(x + k);
            acc0 = vfmaq_f32(acc0, vld1q_f32(row0 + k), vx);
            acc1 = vfmaq_f32(acc1, vld1q_f32(row1 + k), vx);
            acc2 = vfmaq_f32(acc2, vld1q_f32(row2 + k), vx);
            acc3 = vfmaq_f32(acc3, vld1q_f32(row3 + k), vx);
        }
        // Horizontal reduce each accumulator and handle scalar tail
        float s0 = vaddvq_f32(acc0);
        float s1 = vaddvq_f32(acc1);
        float s2 = vaddvq_f32(acc2);
        float s3 = vaddvq_f32(acc3);
        for (; k < N; ++k) {
            s0 += row0[k] * x[k];
            s1 += row1[k] * x[k];
            s2 += row2[k] * x[k];
            s3 += row3[k] * x[k];
        }
        y[m + 0] = s0;
        y[m + 1] = s1;
        y[m + 2] = s2;
        y[m + 3] = s3;
    }
    // Scalar tail for remaining rows
    for (; m < M; ++m) {
        const float* row = A + m * N;
        float32x4_t acc = vdupq_n_f32(0.0f);
        int64_t k = 0;
        for (; k + 3 < N; k += 4) {
            acc = vfmaq_f32(acc, vld1q_f32(row + k), vld1q_f32(x + k));
        }
        float s = vaddvq_f32(acc);
        for (; k < N; ++k) s += row[k] * x[k];
        y[m] = s;
    }
}

/// Small GEMM micro-kernel with 4x4 register blocking: C += A * B.
/// A is [M x K] row-major, B is [K x N] row-major, C is [M x N] row-major.
/// Processes 4 rows of A and 4 columns of B at a time using FMA.
inline void gemm_4x4_f32(const float* A, const float* B, float* C,
                          int64_t M, int64_t N, int64_t K) {
    // Tile over M in steps of 4, N in steps of 4
    int64_t m = 0;
    for (; m + 3 < M; m += 4) {
        int64_t n = 0;
        for (; n + 3 < N; n += 4) {
            // 4x4 accumulator tile
            float32x4_t c0 = vld1q_f32(C + (m + 0) * N + n);
            float32x4_t c1 = vld1q_f32(C + (m + 1) * N + n);
            float32x4_t c2 = vld1q_f32(C + (m + 2) * N + n);
            float32x4_t c3 = vld1q_f32(C + (m + 3) * N + n);

            for (int64_t k = 0; k < K; ++k) {
                // Broadcast A[m+i][k] for each of the 4 rows
                float32x4_t a0 = vdupq_n_f32(A[(m + 0) * K + k]);
                float32x4_t a1 = vdupq_n_f32(A[(m + 1) * K + k]);
                float32x4_t a2 = vdupq_n_f32(A[(m + 2) * K + k]);
                float32x4_t a3 = vdupq_n_f32(A[(m + 3) * K + k]);

                // Load B[k][n..n+3]
                float32x4_t b_row = vld1q_f32(B + k * N + n);

                c0 = vfmaq_f32(c0, a0, b_row);
                c1 = vfmaq_f32(c1, a1, b_row);
                c2 = vfmaq_f32(c2, a2, b_row);
                c3 = vfmaq_f32(c3, a3, b_row);
            }

            vst1q_f32(C + (m + 0) * N + n, c0);
            vst1q_f32(C + (m + 1) * N + n, c1);
            vst1q_f32(C + (m + 2) * N + n, c2);
            vst1q_f32(C + (m + 3) * N + n, c3);
        }
        // Handle remaining columns (N not multiple of 4)
        for (; n < N; ++n) {
            for (int64_t mi = 0; mi < 4; ++mi) {
                float sum = C[(m + mi) * N + n];
                for (int64_t k = 0; k < K; ++k) {
                    sum += A[(m + mi) * K + k] * B[k * N + n];
                }
                C[(m + mi) * N + n] = sum;
            }
        }
    }
    // Handle remaining rows (M not multiple of 4)
    for (; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = C[m * N + n];
            for (int64_t k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

} // namespace tenzor::cpu::neon

// ============================================================================
// Float16 NEON kernels (8-wide, requires FP16 vector arithmetic)
// ============================================================================

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

namespace tenzor::cpu::neon {

inline void add_f16(const __fp16* a, const __fp16* b, __fp16* dst, int64_t n) {
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        float16x8_t va = vld1q_f16(a + i);
        float16x8_t vb = vld1q_f16(b + i);
        vst1q_f16(dst + i, vaddq_f16(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] + b[i];
}

inline void mul_f16(const __fp16* a, const __fp16* b, __fp16* dst, int64_t n) {
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        float16x8_t va = vld1q_f16(a + i);
        float16x8_t vb = vld1q_f16(b + i);
        vst1q_f16(dst + i, vmulq_f16(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] * b[i];
}

inline void relu_f16(const __fp16* src, __fp16* dst, int64_t n) {
    int64_t i = 0;
    float16x8_t zero = vdupq_n_f16((__fp16)0.0f);
    for (; i + 7 < n; i += 8) {
        float16x8_t v = vld1q_f16(src + i);
        vst1q_f16(dst + i, vmaxq_f16(v, zero));
    }
    for (; i < n; ++i) dst[i] = src[i] > (__fp16)0.0f ? src[i] : (__fp16)0.0f;
}

} // namespace tenzor::cpu::neon

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#endif // __ARM_NEON
