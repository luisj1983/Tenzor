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
    #if defined(__AVX512BF16__) && defined(__AVX512F__)
        #define TENZOR_BF16_AVX512_NATIVE
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
    // Round-to-nearest-even: bias = 0x7FFF + lsb16 (matches AVX2/AVX512 converters).
    // At an exact tie (low 16 bits == 0x8000), lsb16==1 yields bias 0x8000 which
    // carries into bit 16, rounding up to make the retained mantissa even.
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounding_bias = 0x00007FFFu + lsb;
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
// Batch BFloat16 <-> Float32 Conversion Helpers
// ============================================================================

/**
 * @brief Batch convert BFloat16 array to Float32 array using SIMD bit-shift
 *
 * BFloat16 is the upper 16 bits of Float32, so conversion is a zero-extend
 * and shift left by 16. Processes 16 elements at a time (AVX-512) or 8 at
 * a time (AVX2) with scalar fallback for remainder and non-SIMD targets.
 *
 * @param src  Source BFloat16 array
 * @param dst  Destination float array (must hold at least n elements)
 * @param n    Number of elements to convert
 */
inline void convert_bf16_to_f32_batch(const BFloat16* src, float* dst, size_t n) {
    const uint16_t* raw = reinterpret_cast<const uint16_t*>(src);
    size_t i = 0;

#ifdef TENZOR_BF16_AVX512
    for (; i + 16 <= n; i += 16) {
        __m512 vals = cvt_bf16_to_f32_avx512(raw + i);
        _mm512_storeu_ps(dst + i, vals);
    }
#endif // TENZOR_BF16_AVX512

#ifdef TENZOR_BF16_AVX2
    for (; i + 8 <= n; i += 8) {
        // Load 8 BF16 values as 16-bit integers
        __m128i bf16_vals = _mm_loadu_si128(reinterpret_cast<const __m128i*>(raw + i));
        // Zero-extend to 32-bit and shift left by 16 to reconstruct FP32
        __m256i extended = _mm256_cvtepu16_epi32(bf16_vals);
        __m256i shifted = _mm256_slli_epi32(extended, 16);
        __m256 fp32_vals = _mm256_castsi256_ps(shifted);
        _mm256_storeu_ps(dst + i, fp32_vals);
    }
#endif // TENZOR_BF16_AVX2

    for (; i < n; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

/**
 * @brief Batch convert Float32 array to BFloat16 array with round-to-nearest-even
 *
 * Uses SIMD rounding bias to achieve correct round-to-nearest-even behavior.
 * Processes 16 elements at a time (AVX-512) or 8 at a time (AVX2) with scalar
 * fallback for remainder and non-SIMD targets.
 *
 * @param src  Source float array
 * @param dst  Destination BFloat16 array (must hold at least n elements)
 * @param n    Number of elements to convert
 */
inline void convert_f32_to_bf16_batch(const float* src, BFloat16* dst, size_t n) {
    uint16_t* raw = reinterpret_cast<uint16_t*>(dst);
    size_t i = 0;

#ifdef TENZOR_BF16_AVX512
    for (; i + 16 <= n; i += 16) {
        __m512 fp32_vals = _mm512_loadu_ps(src + i);
        cvt_f32_to_bf16_avx512(fp32_vals, raw + i);
    }
#endif // TENZOR_BF16_AVX512

#ifdef TENZOR_BF16_AVX2
    for (; i + 8 <= n; i += 8) {
        __m256 fp32_vals = _mm256_loadu_ps(src + i);
        cvt_f32_to_bf16_avx2(fp32_vals, raw + i);
    }
#endif // TENZOR_BF16_AVX2

    for (; i < n; ++i) {
        dst[i] = BFloat16(src[i]);
    }
}

} // namespace bfloat16_simd
} // namespace cpu
} // namespace tenzor
