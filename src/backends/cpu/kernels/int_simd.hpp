/**
 * @file int_simd.hpp
 * @brief AVX2 SIMD intrinsics for integer arithmetic (Int8, Int16, Int32)
 *
 * Provides vectorized add/sub/mul for integer types using AVX2.
 * Falls back to scalar loops when AVX2 is not available.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <algorithm>

#ifdef TENZOR_HAS_AVX2
#include <immintrin.h>
#endif

namespace tenzor {
namespace cpu {
namespace int_simd {

// ============================================================================
// Int32 AVX2
// ============================================================================

inline void add_i32(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
#ifdef TENZOR_HAS_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i), _mm256_add_epi32(va, vb));
    }
    for (; i < n; ++i) c[i] = a[i] + b[i];
#else
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
#endif
}


// ============================================================================
// Int16 AVX2
// ============================================================================

inline void add_i16(const int16_t* a, const int16_t* b, int16_t* c, size_t n) {
#ifdef TENZOR_HAS_AVX2
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i), _mm256_add_epi16(va, vb));
    }
    for (; i < n; ++i) c[i] = a[i] + b[i];
#else
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
#endif
}



// ============================================================================
// Int8 AVX2
// ============================================================================

inline void add_i8(const int8_t* a, const int8_t* b, int8_t* c, size_t n) {
#ifdef TENZOR_HAS_AVX2
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i), _mm256_add_epi8(va, vb));
    }
    for (; i < n; ++i) c[i] = a[i] + b[i];
#else
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
#endif
}


inline void mul_i8(const int8_t* a, const int8_t* b, int8_t* c, size_t n) {
    // Int8 multiply: widen to 16-bit, multiply, take low byte (wrap semantics).
    // This matches PyTorch: int8(100)*int8(2) = -56 (two's-complement wrap).
#ifdef TENZOR_HAS_AVX2
    // Mask to extract the low byte of each 16-bit lane.
    const __m256i low_byte_mask = _mm256_set1_epi16(static_cast<short>(0x00FF));
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));

        // Unpack to 16-bit: low and high halves
        __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));

        // Multiply as 16-bit (keep low 16 bits = wrap on overflow)
        __m256i prod_lo = _mm256_mullo_epi16(va_lo, vb_lo);
        __m256i prod_hi = _mm256_mullo_epi16(va_hi, vb_hi);

        // Keep only the low byte of each 16-bit lane (wrapping truncation)
        prod_lo = _mm256_and_si256(prod_lo, low_byte_mask);
        prod_hi = _mm256_and_si256(prod_hi, low_byte_mask);

        // Pack 16-bit lanes back to 8-bit using unsigned pack (no saturation),
        // then fix the lane crossing: packs interleaves 128-bit lanes.
        __m256i packed = _mm256_packus_epi16(prod_lo, prod_hi);
        // _mm256_packus_epi16 produces {lo128_lo, hi128_lo, lo128_hi, hi128_hi};
        // permute to restore sequential order: {0,2,1,3} -> {0,1,2,3}
        packed = _mm256_permute4x64_epi64(packed, 0xD8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i), packed);
    }
    for (; i < n; ++i) {
        c[i] = static_cast<int8_t>(static_cast<int>(a[i]) * static_cast<int>(b[i]));
    }
#else
    for (size_t i = 0; i < n; ++i) {
        c[i] = static_cast<int8_t>(static_cast<int>(a[i]) * static_cast<int>(b[i]));
    }
#endif
}

// ============================================================================
// UInt8 AVX2
// ============================================================================

inline void add_u8(const uint8_t* a, const uint8_t* b, uint8_t* c, size_t n) {
#ifdef TENZOR_HAS_AVX2
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i), _mm256_add_epi8(va, vb));
    }
    for (; i < n; ++i) c[i] = a[i] + b[i];
#else
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
#endif
}


inline void mul_u8(const uint8_t* a, const uint8_t* b, uint8_t* c, size_t n) {
    // UInt8 multiply: widen to 16-bit, multiply, take low byte (wrap semantics).
    // This matches PyTorch: uint8(200)*uint8(2) = 144 (400 mod 256, wrap).
#ifdef TENZOR_HAS_AVX2
    const __m256i low_byte_mask = _mm256_set1_epi16(static_cast<short>(0x00FF));
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));

        // Unpack to 16-bit (zero-extend for unsigned)
        __m256i va_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));
        __m256i vb_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vb));
        __m256i va_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 1));
        __m256i vb_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vb, 1));

        // Multiply as 16-bit
        __m256i prod_lo = _mm256_mullo_epi16(va_lo, vb_lo);
        __m256i prod_hi = _mm256_mullo_epi16(va_hi, vb_hi);

        // Keep only the low byte of each 16-bit lane (wrapping truncation)
        prod_lo = _mm256_and_si256(prod_lo, low_byte_mask);
        prod_hi = _mm256_and_si256(prod_hi, low_byte_mask);

        // Pack back to 8-bit; permute to fix lane crossing
        __m256i packed = _mm256_packus_epi16(prod_lo, prod_hi);
        packed = _mm256_permute4x64_epi64(packed, 0xD8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i), packed);
    }
    for (; i < n; ++i) {
        c[i] = static_cast<uint8_t>(static_cast<unsigned>(a[i]) * static_cast<unsigned>(b[i]));
    }
#else
    for (size_t i = 0; i < n; ++i) {
        c[i] = static_cast<uint8_t>(static_cast<unsigned>(a[i]) * static_cast<unsigned>(b[i]));
    }
#endif
}

} // namespace int_simd
} // namespace cpu
} // namespace tenzor
