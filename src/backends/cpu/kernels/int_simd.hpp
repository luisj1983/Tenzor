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

inline void sub_i32(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
#ifdef TENZOR_HAS_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i), _mm256_sub_epi32(va, vb));
    }
    for (; i < n; ++i) c[i] = a[i] - b[i];
#else
    for (size_t i = 0; i < n; ++i) c[i] = a[i] - b[i];
#endif
}

inline void mul_i32(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
#ifdef TENZOR_HAS_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i), _mm256_mullo_epi32(va, vb));
    }
    for (; i < n; ++i) c[i] = a[i] * b[i];
#else
    for (size_t i = 0; i < n; ++i) c[i] = a[i] * b[i];
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

inline void sub_i16(const int16_t* a, const int16_t* b, int16_t* c, size_t n) {
#ifdef TENZOR_HAS_AVX2
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i), _mm256_sub_epi16(va, vb));
    }
    for (; i < n; ++i) c[i] = a[i] - b[i];
#else
    for (size_t i = 0; i < n; ++i) c[i] = a[i] - b[i];
#endif
}

inline void mul_i16(const int16_t* a, const int16_t* b, int16_t* c, size_t n) {
#ifdef TENZOR_HAS_AVX2
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i), _mm256_mullo_epi16(va, vb));
    }
    for (; i < n; ++i) c[i] = a[i] * b[i];
#else
    for (size_t i = 0; i < n; ++i) c[i] = a[i] * b[i];
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

inline void sub_i8(const int8_t* a, const int8_t* b, int8_t* c, size_t n) {
#ifdef TENZOR_HAS_AVX2
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i), _mm256_sub_epi8(va, vb));
    }
    for (; i < n; ++i) c[i] = a[i] - b[i];
#else
    for (size_t i = 0; i < n; ++i) c[i] = a[i] - b[i];
#endif
}

inline void mul_i8(const int8_t* a, const int8_t* b, int8_t* c, size_t n) {
    // Int8 multiply needs widening: unpack to 16-bit, multiply, saturate back to [-128,127]
#ifdef TENZOR_HAS_AVX2
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));

        // Unpack to 16-bit: low and high halves
        __m256i va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        __m256i vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        __m256i va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        __m256i vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));

        // Multiply as 16-bit
        __m256i prod_lo = _mm256_mullo_epi16(va_lo, vb_lo);
        __m256i prod_hi = _mm256_mullo_epi16(va_hi, vb_hi);

        // Saturate back to signed 8-bit [-128, 127] using packs_epi16
        __m256i packed = _mm256_packs_epi16(prod_lo, prod_hi);
        // Fix lane crossing from packs: {0,2,1,3} -> {0,1,2,3}
        packed = _mm256_permute4x64_epi64(packed, 0xD8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i), packed);
    }
    for (; i < n; ++i) {
        int result = static_cast<int>(a[i]) * static_cast<int>(b[i]);
        c[i] = static_cast<int8_t>(std::clamp(result, -128, 127));
    }
#else
    for (size_t i = 0; i < n; ++i) {
        int result = static_cast<int>(a[i]) * static_cast<int>(b[i]);
        c[i] = static_cast<int8_t>(std::clamp(result, -128, 127));
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

inline void sub_u8(const uint8_t* a, const uint8_t* b, uint8_t* c, size_t n) {
#ifdef TENZOR_HAS_AVX2
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i), _mm256_sub_epi8(va, vb));
    }
    for (; i < n; ++i) c[i] = a[i] - b[i];
#else
    for (size_t i = 0; i < n; ++i) c[i] = a[i] - b[i];
#endif
}

} // namespace int_simd
} // namespace cpu
} // namespace tenzor
