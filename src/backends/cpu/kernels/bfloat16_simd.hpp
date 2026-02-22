/**
 * @file bfloat16_simd.hpp
 * @brief SIMD-accelerated BFloat16 operations using AVX2/AVX512
 *
 * BFloat16 conversion is simpler than Float16 — no dedicated hardware
 * instructions needed. BF16 is just the upper 16 bits of Float32:
 * - BF16 to F32: zero-extend to 32-bit, shift left 16
 * - F32 to BF16: round-to-nearest-even, shift right 16, pack to 16-bit
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "tenzor/core/dtype.hpp"

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #if defined(__AVX2__)
        #define TENZOR_BF16_AVX2
    #endif
    #if defined(__AVX512F__) && defined(__AVX512BW__)
        #define TENZOR_BF16_AVX512
    #endif
#endif

namespace tenzor {
namespace cpu {
namespace bfloat16_simd {

// ============================================================================
// BFloat16 <-> Float32 Conversion Utilities
// ============================================================================

// Scalar conversions
inline float bf16_to_f32_scalar(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

inline uint16_t f32_to_bf16_scalar(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(uint32_t));
    // Round-to-nearest-even: add rounding bias based on lsb and round bit
    uint32_t rounding_bias = (bits & 0x00010000u) ? 0x00007FFFu : 0x00007FFEu;
    // Handle NaN: if NaN, force quiet NaN
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0) {
        return static_cast<uint16_t>((bits >> 16) | 0x0040u);
    }
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

#ifdef TENZOR_BF16_AVX2

/**
 * @brief Convert 8 BFloat16 values to 8 Float32 values (AVX2)
 */
inline __m256 cvt_bf16_to_f32_avx2(const uint16_t* bf16) {
    __m128i bf16_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(bf16));
    __m256i extended = _mm256_cvtepu16_epi32(bf16_vec);
    __m256i shifted = _mm256_slli_epi32(extended, 16);
    return _mm256_castsi256_ps(shifted);
}

/**
 * @brief Convert 8 Float32 values to 8 BFloat16 values (AVX2) with round-to-nearest-even
 */
inline void cvt_f32_to_bf16_avx2(__m256 fp32, uint16_t* bf16) {
    __m256i bits = _mm256_castps_si256(fp32);
    // Round-to-nearest-even bias
    __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
    __m256i rounding_bias = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb);
    // Handle NaN: detect inf/nan exponent and non-zero mantissa
    __m256i exp_mask = _mm256_set1_epi32(0x7F800000);
    __m256i mantissa_mask = _mm256_set1_epi32(0x007FFFFF);
    __m256i is_inf_exp = _mm256_cmpeq_epi32(_mm256_and_si256(bits, exp_mask), exp_mask);
    __m256i has_mantissa = _mm256_cmpeq_epi32(
        _mm256_and_si256(bits, mantissa_mask), _mm256_setzero_si256());
    __m256i is_nan = _mm256_andnot_si256(has_mantissa, is_inf_exp);
    // For NaN: set quiet NaN bit; for normal: add rounding bias
    __m256i nan_result = _mm256_or_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(0x0040));
    __m256i normal_result = _mm256_srli_epi32(_mm256_add_epi32(bits, rounding_bias), 16);
    // _mm256_cmpeq_epi32 produces all-1s or all-0s per 32-bit element, so every byte
    // within each element has the same high bit. _mm256_blendv_epi8 (which selects
    // per byte based on the high bit) therefore operates correctly per 32-bit element.
    __m256i result32 = _mm256_blendv_epi8(normal_result, nan_result, is_nan);
    // Pack 32-bit results to 16-bit
    // Shuffle bytes to extract lower 16 bits of each 32-bit element
    __m256i shuffled = _mm256_shuffle_epi8(result32,
        _mm256_setr_epi8(
            0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1));
    // Extract lower 64 bits from each 128-bit lane and combine
    __m128i lo = _mm256_castsi256_si128(shuffled);
    __m128i hi = _mm256_extracti128_si256(shuffled, 1);
    __m128i packed = _mm_unpacklo_epi64(lo, hi);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(bf16), packed);
}

#endif // TENZOR_BF16_AVX2

#ifdef TENZOR_BF16_AVX512

/**
 * @brief Convert 16 BFloat16 values to 16 Float32 values (AVX-512)
 */
inline __m512 cvt_bf16_to_f32_avx512(const uint16_t* bf16) {
    __m256i bf16_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bf16));
    __m512i extended = _mm512_cvtepu16_epi32(bf16_vec);
    __m512i shifted = _mm512_slli_epi32(extended, 16);
    return _mm512_castsi512_ps(shifted);
}

/**
 * @brief Convert 16 Float32 values to 16 BFloat16 values (AVX-512) with round-to-nearest-even
 */
inline void cvt_f32_to_bf16_avx512(__m512 fp32, uint16_t* bf16) {
    __m512i bits = _mm512_castps_si512(fp32);
    // Round-to-nearest-even
    __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
    __m512i rounding_bias = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
    // NaN detection
    __m512i exp_mask = _mm512_set1_epi32(0x7F800000);
    __m512i mantissa_mask = _mm512_set1_epi32(0x007FFFFF);
    __mmask16 is_inf_exp = _mm512_cmpeq_epi32_mask(_mm512_and_si512(bits, exp_mask), exp_mask);
    __mmask16 has_mantissa = ~_mm512_cmpeq_epi32_mask(
        _mm512_and_si512(bits, mantissa_mask), _mm512_setzero_si512());
    __mmask16 is_nan = is_inf_exp & has_mantissa;
    // Normal path: add rounding bias and shift
    __m512i normal_result = _mm512_srli_epi32(_mm512_add_epi32(bits, rounding_bias), 16);
    // NaN path: quiet NaN
    __m512i nan_result = _mm512_or_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(0x0040));
    // Blend
    __m512i result32 = _mm512_mask_blend_epi32(is_nan, normal_result, nan_result);
    // Pack 16x32-bit to 16x16-bit using pmovdw
    __m256i packed = _mm512_cvtepi32_epi16(result32);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(bf16), packed);
}

#endif // TENZOR_BF16_AVX512

// ============================================================================
// BFloat16 Binary Operations (compute in FP32, store in BF16)
// ============================================================================

inline void add_bf16(const uint16_t* a, const uint16_t* b, uint16_t* out, size_t n) {
#ifdef TENZOR_BF16_AVX512
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = cvt_bf16_to_f32_avx512(a + i);
        __m512 vb = cvt_bf16_to_f32_avx512(b + i);
        __m512 vout = _mm512_add_ps(va, vb);
        cvt_f32_to_bf16_avx512(vout, out + i);
    }
    for (; i < n; ++i) {
        out[i] = f32_to_bf16_scalar(bf16_to_f32_scalar(a[i]) + bf16_to_f32_scalar(b[i]));
    }
#elif defined(TENZOR_BF16_AVX2)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = cvt_bf16_to_f32_avx2(a + i);
        __m256 vb = cvt_bf16_to_f32_avx2(b + i);
        __m256 vout = _mm256_add_ps(va, vb);
        cvt_f32_to_bf16_avx2(vout, out + i);
    }
    for (; i < n; ++i) {
        out[i] = f32_to_bf16_scalar(bf16_to_f32_scalar(a[i]) + bf16_to_f32_scalar(b[i]));
    }
#else
    for (size_t i = 0; i < n; ++i) {
        BFloat16 fa, fb;
        std::memcpy(&fa, &a[i], sizeof(uint16_t));
        std::memcpy(&fb, &b[i], sizeof(uint16_t));
        BFloat16 result(static_cast<float>(fa) + static_cast<float>(fb));
        std::memcpy(&out[i], &result, sizeof(uint16_t));
    }
#endif
}

inline void sub_bf16(const uint16_t* a, const uint16_t* b, uint16_t* out, size_t n) {
#ifdef TENZOR_BF16_AVX512
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = cvt_bf16_to_f32_avx512(a + i);
        __m512 vb = cvt_bf16_to_f32_avx512(b + i);
        __m512 vout = _mm512_sub_ps(va, vb);
        cvt_f32_to_bf16_avx512(vout, out + i);
    }
    for (; i < n; ++i) {
        out[i] = f32_to_bf16_scalar(bf16_to_f32_scalar(a[i]) - bf16_to_f32_scalar(b[i]));
    }
#elif defined(TENZOR_BF16_AVX2)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = cvt_bf16_to_f32_avx2(a + i);
        __m256 vb = cvt_bf16_to_f32_avx2(b + i);
        __m256 vout = _mm256_sub_ps(va, vb);
        cvt_f32_to_bf16_avx2(vout, out + i);
    }
    for (; i < n; ++i) {
        out[i] = f32_to_bf16_scalar(bf16_to_f32_scalar(a[i]) - bf16_to_f32_scalar(b[i]));
    }
#else
    for (size_t i = 0; i < n; ++i) {
        BFloat16 fa, fb;
        std::memcpy(&fa, &a[i], sizeof(uint16_t));
        std::memcpy(&fb, &b[i], sizeof(uint16_t));
        BFloat16 result(static_cast<float>(fa) - static_cast<float>(fb));
        std::memcpy(&out[i], &result, sizeof(uint16_t));
    }
#endif
}

inline void mul_bf16(const uint16_t* a, const uint16_t* b, uint16_t* out, size_t n) {
#ifdef TENZOR_BF16_AVX512
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = cvt_bf16_to_f32_avx512(a + i);
        __m512 vb = cvt_bf16_to_f32_avx512(b + i);
        __m512 vout = _mm512_mul_ps(va, vb);
        cvt_f32_to_bf16_avx512(vout, out + i);
    }
    for (; i < n; ++i) {
        out[i] = f32_to_bf16_scalar(bf16_to_f32_scalar(a[i]) * bf16_to_f32_scalar(b[i]));
    }
#elif defined(TENZOR_BF16_AVX2)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = cvt_bf16_to_f32_avx2(a + i);
        __m256 vb = cvt_bf16_to_f32_avx2(b + i);
        __m256 vout = _mm256_mul_ps(va, vb);
        cvt_f32_to_bf16_avx2(vout, out + i);
    }
    for (; i < n; ++i) {
        out[i] = f32_to_bf16_scalar(bf16_to_f32_scalar(a[i]) * bf16_to_f32_scalar(b[i]));
    }
#else
    for (size_t i = 0; i < n; ++i) {
        BFloat16 fa, fb;
        std::memcpy(&fa, &a[i], sizeof(uint16_t));
        std::memcpy(&fb, &b[i], sizeof(uint16_t));
        BFloat16 result(static_cast<float>(fa) * static_cast<float>(fb));
        std::memcpy(&out[i], &result, sizeof(uint16_t));
    }
#endif
}

inline void div_bf16(const uint16_t* a, const uint16_t* b, uint16_t* out, size_t n) {
#ifdef TENZOR_BF16_AVX512
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = cvt_bf16_to_f32_avx512(a + i);
        __m512 vb = cvt_bf16_to_f32_avx512(b + i);
        __m512 vout = _mm512_div_ps(va, vb);
        cvt_f32_to_bf16_avx512(vout, out + i);
    }
    for (; i < n; ++i) {
        out[i] = f32_to_bf16_scalar(bf16_to_f32_scalar(a[i]) / bf16_to_f32_scalar(b[i]));
    }
#elif defined(TENZOR_BF16_AVX2)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = cvt_bf16_to_f32_avx2(a + i);
        __m256 vb = cvt_bf16_to_f32_avx2(b + i);
        __m256 vout = _mm256_div_ps(va, vb);
        cvt_f32_to_bf16_avx2(vout, out + i);
    }
    for (; i < n; ++i) {
        out[i] = f32_to_bf16_scalar(bf16_to_f32_scalar(a[i]) / bf16_to_f32_scalar(b[i]));
    }
#else
    for (size_t i = 0; i < n; ++i) {
        BFloat16 fa, fb;
        std::memcpy(&fa, &a[i], sizeof(uint16_t));
        std::memcpy(&fb, &b[i], sizeof(uint16_t));
        BFloat16 result(static_cast<float>(fa) / static_cast<float>(fb));
        std::memcpy(&out[i], &result, sizeof(uint16_t));
    }
#endif
}

inline void fma_bf16(const uint16_t* a, const uint16_t* b, const uint16_t* c,
                     uint16_t* out, size_t n) {
#ifdef TENZOR_BF16_AVX512
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = cvt_bf16_to_f32_avx512(a + i);
        __m512 vb = cvt_bf16_to_f32_avx512(b + i);
        __m512 vc = cvt_bf16_to_f32_avx512(c + i);
        __m512 vout = _mm512_fmadd_ps(va, vb, vc);
        cvt_f32_to_bf16_avx512(vout, out + i);
    }
    for (; i < n; ++i) {
        float fa = bf16_to_f32_scalar(a[i]);
        float fb = bf16_to_f32_scalar(b[i]);
        float fc = bf16_to_f32_scalar(c[i]);
        out[i] = f32_to_bf16_scalar(fa * fb + fc);
    }
#elif defined(TENZOR_BF16_AVX2)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = cvt_bf16_to_f32_avx2(a + i);
        __m256 vb = cvt_bf16_to_f32_avx2(b + i);
        __m256 vc = cvt_bf16_to_f32_avx2(c + i);
        __m256 vout = _mm256_fmadd_ps(va, vb, vc);
        cvt_f32_to_bf16_avx2(vout, out + i);
    }
    for (; i < n; ++i) {
        float fa = bf16_to_f32_scalar(a[i]);
        float fb = bf16_to_f32_scalar(b[i]);
        float fc = bf16_to_f32_scalar(c[i]);
        out[i] = f32_to_bf16_scalar(fa * fb + fc);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        BFloat16 fa, fb, fc;
        std::memcpy(&fa, &a[i], sizeof(uint16_t));
        std::memcpy(&fb, &b[i], sizeof(uint16_t));
        std::memcpy(&fc, &c[i], sizeof(uint16_t));
        BFloat16 result(static_cast<float>(fa) * static_cast<float>(fb) + static_cast<float>(fc));
        std::memcpy(&out[i], &result, sizeof(uint16_t));
    }
#endif
}

// ============================================================================
// BFloat16 Unary Operations
// ============================================================================

inline void scale_bf16(const uint16_t* a, float scalar, uint16_t* out, size_t n) {
#ifdef TENZOR_BF16_AVX512
    size_t i = 0;
    __m512 vscalar = _mm512_set1_ps(scalar);
    for (; i + 16 <= n; i += 16) {
        __m512 va = cvt_bf16_to_f32_avx512(a + i);
        __m512 vout = _mm512_mul_ps(va, vscalar);
        cvt_f32_to_bf16_avx512(vout, out + i);
    }
    for (; i < n; ++i) {
        out[i] = f32_to_bf16_scalar(bf16_to_f32_scalar(a[i]) * scalar);
    }
#elif defined(TENZOR_BF16_AVX2)
    size_t i = 0;
    __m256 vscalar = _mm256_set1_ps(scalar);
    for (; i + 8 <= n; i += 8) {
        __m256 va = cvt_bf16_to_f32_avx2(a + i);
        __m256 vout = _mm256_mul_ps(va, vscalar);
        cvt_f32_to_bf16_avx2(vout, out + i);
    }
    for (; i < n; ++i) {
        out[i] = f32_to_bf16_scalar(bf16_to_f32_scalar(a[i]) * scalar);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        BFloat16 fa;
        std::memcpy(&fa, &a[i], sizeof(uint16_t));
        BFloat16 result(static_cast<float>(fa) * scalar);
        std::memcpy(&out[i], &result, sizeof(uint16_t));
    }
#endif
}

// ============================================================================
// BFloat16 Reduction Operations (compute in FP32)
// ============================================================================

inline float sum_bf16(const uint16_t* a, size_t n) {
    float sum = 0.0f;

#ifdef TENZOR_BF16_AVX512
    size_t i = 0;
    __m512 vsum = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        __m512 va = cvt_bf16_to_f32_avx512(a + i);
        vsum = _mm512_add_ps(vsum, va);
    }
    sum += _mm512_reduce_add_ps(vsum);
    for (; i < n; ++i) {
        sum += bf16_to_f32_scalar(a[i]);
    }
#elif defined(TENZOR_BF16_AVX2)
    size_t i = 0;
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = cvt_bf16_to_f32_avx2(a + i);
        vsum = _mm256_add_ps(vsum, va);
    }
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum += _mm_cvtss_f32(sum128);
    for (; i < n; ++i) {
        sum += bf16_to_f32_scalar(a[i]);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        BFloat16 fa;
        std::memcpy(&fa, &a[i], sizeof(uint16_t));
        sum += static_cast<float>(fa);
    }
#endif

    return sum;
}

inline float dot_bf16(const uint16_t* a, const uint16_t* b, size_t n) {
    float sum = 0.0f;

#ifdef TENZOR_BF16_AVX512
    size_t i = 0;
    __m512 vsum = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        __m512 va = cvt_bf16_to_f32_avx512(a + i);
        __m512 vb = cvt_bf16_to_f32_avx512(b + i);
        vsum = _mm512_fmadd_ps(va, vb, vsum);
    }
    sum += _mm512_reduce_add_ps(vsum);
    for (; i < n; ++i) {
        sum += bf16_to_f32_scalar(a[i]) * bf16_to_f32_scalar(b[i]);
    }
#elif defined(TENZOR_BF16_AVX2)
    size_t i = 0;
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = cvt_bf16_to_f32_avx2(a + i);
        __m256 vb = cvt_bf16_to_f32_avx2(b + i);
        vsum = _mm256_fmadd_ps(va, vb, vsum);
    }
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum += _mm_cvtss_f32(sum128);
    for (; i < n; ++i) {
        sum += bf16_to_f32_scalar(a[i]) * bf16_to_f32_scalar(b[i]);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        BFloat16 fa, fb;
        std::memcpy(&fa, &a[i], sizeof(uint16_t));
        std::memcpy(&fb, &b[i], sizeof(uint16_t));
        sum += static_cast<float>(fa) * static_cast<float>(fb);
    }
#endif

    return sum;
}

} // namespace bfloat16_simd
} // namespace cpu
} // namespace tenzor
