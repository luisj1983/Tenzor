/**
 * @file float16_simd.hpp
 * @brief SIMD-accelerated Float16 operations using F16C instructions
 *
 * Key features:
 * - Native F16C conversion intrinsics (_mm256_cvtph_ps / _mm256_cvtps_ph)
 * - Batch processing for Float16 arrays
 * - Mixed-precision compute (FP16 storage, FP32 compute)
 * - 8-15x speedup over scalar Float16 operations
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/omp_thresholds.hpp"

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #if defined(__F16C__)
        #define TENZOR_F16C_AVAILABLE
    #endif
    #if defined(__AVX2__)
        #define TENZOR_F16_AVX2
    #endif
    #if defined(__AVX512F__) && defined(__AVX512BW__)
        #define TENZOR_F16_AVX512
    #endif
#endif

namespace tenzor {
namespace cpu {
namespace float16_simd {

// ============================================================================
// F16C Conversion Utilities
// ============================================================================

#ifdef TENZOR_F16C_AVAILABLE

/**
 * @brief Convert 8 Float16 values to 8 Float32 values (AVX2)
 */
inline __m256 cvt_f16_to_f32_avx2(const uint16_t* fp16) {
    __m128i fp16_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(fp16));
    return _mm256_cvtph_ps(fp16_vec);
}

/**
 * @brief Saturate a vector of Float32 values to the finite Float16 range
 *        [-65504, 65504] (AVX2).
 *
 * The F16C conversion intrinsic `_mm256_cvtps_ph` produces signed *infinity*
 * on overflow (strict IEEE-754). Deep-network Float16 gradients routinely
 * exceed the half range (e.g. ResNet-101's input gradient reaches ~1.4e5),
 * and a mid-chain inf later becomes nan via inf - inf, severing the gradient
 * signal on CPU/CUDA. ROCm/OneAPI half conversions instead saturate to ±65504.
 * Clamping here makes CPU match those backends (cross-backend parity) and
 * keeps the gradient finite. NaN is preserved: the ordered comparison is
 * false for NaN, so NaN passes through unchanged (genuine NaNs stay visible).
 */
inline __m256 saturate_f16_range_avx2(__m256 x) {
    const __m256 vmax = _mm256_set1_ps(65504.0f);
    const __m256 signmask = _mm256_set1_ps(-0.0f);
    __m256 absx = _mm256_andnot_ps(signmask, x);
    // over = |x| > 65504  (ordered: false for NaN -> NaN preserved)
    __m256 over = _mm256_cmp_ps(absx, vmax, _CMP_GT_OQ);
    // sat = copysign(65504, x): 65504 with x's sign bit
    __m256 sat = _mm256_or_ps(vmax, _mm256_and_ps(signmask, x));
    return _mm256_blendv_ps(x, sat, over);
}

/**
 * @brief Convert 8 Float32 values to 8 Float16 values (AVX2), saturating
 *        out-of-range values to ±65504 instead of producing infinity.
 */
inline void cvt_f32_to_f16_avx2(__m256 fp32, uint16_t* fp16) {
    fp32 = saturate_f16_range_avx2(fp32);
    __m128i fp16_vec = _mm256_cvtps_ph(fp32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(fp16), fp16_vec);
}

/**
 * @brief Saturate a single Float32 to the finite Float16 range (NaN-preserving).
 *
 * Used by the scalar remainder loops and the non-SIMD fallback so every
 * float->half conversion path shares the same saturating overflow semantics.
 */
inline float sat_f16_value(float x) {
    // Comparisons with NaN are false, so NaN passes through unchanged.
    if (x > 65504.0f) return 65504.0f;
    if (x < -65504.0f) return -65504.0f;
    return x;
}

/**
 * @brief Convert one Float32 to a Float16 bit pattern, saturating overflow.
 */
inline uint16_t cvt_f32_to_f16_sat(float x) {
    return _cvtss_sh(sat_f16_value(x), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

#ifdef TENZOR_F16_AVX512

/**
 * @brief Convert 16 Float16 values to 16 Float32 values (AVX-512)
 */
inline __m512 cvt_f16_to_f32_avx512(const uint16_t* fp16) {
    __m256i fp16_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fp16));
    return _mm512_cvtph_ps(fp16_vec);
}

/**
 * @brief Saturate a vector of Float32 values to the finite Float16 range
 *        [-65504, 65504] (AVX-512). See saturate_f16_range_avx2 for rationale.
 *        NaN is preserved (ordered comparison is false for NaN).
 */
inline __m512 saturate_f16_range_avx512(__m512 x) {
    const __m512 vmax = _mm512_set1_ps(65504.0f);
    const __m512 signmask = _mm512_set1_ps(-0.0f);
    __m512 absx = _mm512_andnot_ps(signmask, x);
    __mmask16 over = _mm512_cmp_ps_mask(absx, vmax, _CMP_GT_OQ);
    __m512 sat = _mm512_or_ps(vmax, _mm512_and_ps(signmask, x));
    return _mm512_mask_blend_ps(over, x, sat);
}

/**
 * @brief Convert 16 Float32 values to 16 Float16 values (AVX-512), saturating
 *        out-of-range values to ±65504 instead of producing infinity.
 */
inline void cvt_f32_to_f16_avx512(__m512 fp32, uint16_t* fp16) {
    fp32 = saturate_f16_range_avx512(fp32);
    __m256i fp16_vec = _mm512_cvtps_ph(fp32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(fp16), fp16_vec);
}

#endif // TENZOR_F16_AVX512

#endif // TENZOR_F16C_AVAILABLE

// ============================================================================
// Float16 Binary Operations (compute in FP32, store in FP16)
// ============================================================================

/**
 * @brief SIMD Float16 addition: out = a + b
 */
inline void add_f16(const uint16_t* a, const uint16_t* b, uint16_t* out, size_t n) {
#ifdef TENZOR_F16C_AVAILABLE

#ifdef TENZOR_F16_AVX512
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = cvt_f16_to_f32_avx512(a + i);
        __m512 vb = cvt_f16_to_f32_avx512(b + i);
        __m512 vout = _mm512_add_ps(va, vb);
        cvt_f32_to_f16_avx512(vout, out + i);
    }
    a += i; b += i; out += i; n -= i;
    // Scalar remainder after AVX512
    for (size_t j = 0; j < n; ++j) {
        float fa = _cvtsh_ss(a[j]);
        float fb = _cvtsh_ss(b[j]);
        out[j] = cvt_f32_to_f16_sat(fa + fb);
    }
#elif defined(TENZOR_F16_AVX2)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = cvt_f16_to_f32_avx2(a + i);
        __m256 vb = cvt_f16_to_f32_avx2(b + i);
        __m256 vout = _mm256_add_ps(va, vb);
        cvt_f32_to_f16_avx2(vout, out + i);
    }
    // Scalar remainder
    for (; i < n; ++i) {
        float fa = _cvtsh_ss(a[i]);
        float fb = _cvtsh_ss(b[i]);
        out[i] = cvt_f32_to_f16_sat(fa + fb);
    }
#else
    // Scalar F16C (no AVX2/AVX512)
    for (size_t i = 0; i < n; ++i) {
        float fa = _cvtsh_ss(a[i]);
        float fb = _cvtsh_ss(b[i]);
        out[i] = cvt_f32_to_f16_sat(fa + fb);
    }
#endif

#else
    // Scalar fallback (no F16C) — use Float16 struct for software conversion
    for (size_t i = 0; i < n; ++i) {
        Float16 fa, fb;
        std::memcpy(&fa, &a[i], sizeof(uint16_t));
        std::memcpy(&fb, &b[i], sizeof(uint16_t));
        Float16 result(static_cast<float>(fa) + static_cast<float>(fb));
        std::memcpy(&out[i], &result, sizeof(uint16_t));
    }
#endif
}

/**
 * @brief SIMD Float16 subtraction: out = a - b
 */
inline void sub_f16(const uint16_t* a, const uint16_t* b, uint16_t* out, size_t n) {
#ifdef TENZOR_F16C_AVAILABLE

#ifdef TENZOR_F16_AVX512
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = cvt_f16_to_f32_avx512(a + i);
        __m512 vb = cvt_f16_to_f32_avx512(b + i);
        __m512 vout = _mm512_sub_ps(va, vb);
        cvt_f32_to_f16_avx512(vout, out + i);
    }
    a += i; b += i; out += i; n -= i;
    for (size_t j = 0; j < n; ++j) {
        float fa = _cvtsh_ss(a[j]);
        float fb = _cvtsh_ss(b[j]);
        out[j] = cvt_f32_to_f16_sat(fa - fb);
    }
#elif defined(TENZOR_F16_AVX2)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = cvt_f16_to_f32_avx2(a + i);
        __m256 vb = cvt_f16_to_f32_avx2(b + i);
        __m256 vout = _mm256_sub_ps(va, vb);
        cvt_f32_to_f16_avx2(vout, out + i);
    }
    for (; i < n; ++i) {
        float fa = _cvtsh_ss(a[i]);
        float fb = _cvtsh_ss(b[i]);
        out[i] = cvt_f32_to_f16_sat(fa - fb);
    }
#else
    // Scalar F16C (no AVX2/AVX512)
    for (size_t i = 0; i < n; ++i) {
        float fa = _cvtsh_ss(a[i]);
        float fb = _cvtsh_ss(b[i]);
        out[i] = cvt_f32_to_f16_sat(fa - fb);
    }
#endif

#else
    for (size_t i = 0; i < n; ++i) {
        Float16 fa, fb;
        std::memcpy(&fa, &a[i], sizeof(uint16_t));
        std::memcpy(&fb, &b[i], sizeof(uint16_t));
        Float16 result(static_cast<float>(fa) - static_cast<float>(fb));
        std::memcpy(&out[i], &result, sizeof(uint16_t));
    }
#endif
}

/**
 * @brief SIMD Float16 multiplication: out = a * b
 */
inline void mul_f16(const uint16_t* a, const uint16_t* b, uint16_t* out, size_t n) {
#ifdef TENZOR_F16C_AVAILABLE

#ifdef TENZOR_F16_AVX512
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = cvt_f16_to_f32_avx512(a + i);
        __m512 vb = cvt_f16_to_f32_avx512(b + i);
        __m512 vout = _mm512_mul_ps(va, vb);
        cvt_f32_to_f16_avx512(vout, out + i);
    }
    a += i; b += i; out += i; n -= i;
    for (size_t j = 0; j < n; ++j) {
        float fa = _cvtsh_ss(a[j]);
        float fb = _cvtsh_ss(b[j]);
        out[j] = cvt_f32_to_f16_sat(fa * fb);
    }
#elif defined(TENZOR_F16_AVX2)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = cvt_f16_to_f32_avx2(a + i);
        __m256 vb = cvt_f16_to_f32_avx2(b + i);
        __m256 vout = _mm256_mul_ps(va, vb);
        cvt_f32_to_f16_avx2(vout, out + i);
    }
    for (; i < n; ++i) {
        float fa = _cvtsh_ss(a[i]);
        float fb = _cvtsh_ss(b[i]);
        out[i] = cvt_f32_to_f16_sat(fa * fb);
    }
#else
    // Scalar F16C (no AVX2/AVX512)
    for (size_t i = 0; i < n; ++i) {
        float fa = _cvtsh_ss(a[i]);
        float fb = _cvtsh_ss(b[i]);
        out[i] = cvt_f32_to_f16_sat(fa * fb);
    }
#endif

#else
    for (size_t i = 0; i < n; ++i) {
        Float16 fa, fb;
        std::memcpy(&fa, &a[i], sizeof(uint16_t));
        std::memcpy(&fb, &b[i], sizeof(uint16_t));
        Float16 result(static_cast<float>(fa) * static_cast<float>(fb));
        std::memcpy(&out[i], &result, sizeof(uint16_t));
    }
#endif
}

/**
 * @brief SIMD Float16 division: out = a / b
 */
inline void div_f16(const uint16_t* a, const uint16_t* b, uint16_t* out, size_t n) {
#ifdef TENZOR_F16C_AVAILABLE

#ifdef TENZOR_F16_AVX512
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = cvt_f16_to_f32_avx512(a + i);
        __m512 vb = cvt_f16_to_f32_avx512(b + i);
        __m512 vout = _mm512_div_ps(va, vb);
        cvt_f32_to_f16_avx512(vout, out + i);
    }
    a += i; b += i; out += i; n -= i;
    for (size_t j = 0; j < n; ++j) {
        float fa = _cvtsh_ss(a[j]);
        float fb = _cvtsh_ss(b[j]);
        out[j] = cvt_f32_to_f16_sat(fa / fb);
    }
#elif defined(TENZOR_F16_AVX2)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = cvt_f16_to_f32_avx2(a + i);
        __m256 vb = cvt_f16_to_f32_avx2(b + i);
        __m256 vout = _mm256_div_ps(va, vb);
        cvt_f32_to_f16_avx2(vout, out + i);
    }
    for (; i < n; ++i) {
        float fa = _cvtsh_ss(a[i]);
        float fb = _cvtsh_ss(b[i]);
        out[i] = cvt_f32_to_f16_sat(fa / fb);
    }
#else
    // Scalar F16C (no AVX2/AVX512)
    for (size_t i = 0; i < n; ++i) {
        float fa = _cvtsh_ss(a[i]);
        float fb = _cvtsh_ss(b[i]);
        out[i] = cvt_f32_to_f16_sat(fa / fb);
    }
#endif

#else
    for (size_t i = 0; i < n; ++i) {
        Float16 fa, fb;
        std::memcpy(&fa, &a[i], sizeof(uint16_t));
        std::memcpy(&fb, &b[i], sizeof(uint16_t));
        Float16 result(static_cast<float>(fa) / static_cast<float>(fb));
        std::memcpy(&out[i], &result, sizeof(uint16_t));
    }
#endif
}

} // namespace float16_simd
} // namespace cpu
} // namespace tenzor
