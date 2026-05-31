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
 * @brief Convert 8 Float32 values to 8 Float16 values (AVX2)
 */
inline void cvt_f32_to_f16_avx2(__m256 fp32, uint16_t* fp16) {
    __m128i fp16_vec = _mm256_cvtps_ph(fp32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(fp16), fp16_vec);
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
 * @brief Convert 16 Float32 values to 16 Float16 values (AVX-512)
 */
inline void cvt_f32_to_f16_avx512(__m512 fp32, uint16_t* fp16) {
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
        out[j] = _cvtss_sh(fa + fb, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
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
        out[i] = _cvtss_sh(fa + fb, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
#else
    // Scalar F16C (no AVX2/AVX512)
    for (size_t i = 0; i < n; ++i) {
        float fa = _cvtsh_ss(a[i]);
        float fb = _cvtsh_ss(b[i]);
        out[i] = _cvtss_sh(fa + fb, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
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
        out[j] = _cvtss_sh(fa - fb, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
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
        out[i] = _cvtss_sh(fa - fb, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
#else
    // Scalar F16C (no AVX2/AVX512)
    for (size_t i = 0; i < n; ++i) {
        float fa = _cvtsh_ss(a[i]);
        float fb = _cvtsh_ss(b[i]);
        out[i] = _cvtss_sh(fa - fb, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
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
        out[j] = _cvtss_sh(fa * fb, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
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
        out[i] = _cvtss_sh(fa * fb, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
#else
    // Scalar F16C (no AVX2/AVX512)
    for (size_t i = 0; i < n; ++i) {
        float fa = _cvtsh_ss(a[i]);
        float fb = _cvtsh_ss(b[i]);
        out[i] = _cvtss_sh(fa * fb, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
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
        out[j] = _cvtss_sh(fa / fb, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
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
        out[i] = _cvtss_sh(fa / fb, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
#else
    // Scalar F16C (no AVX2/AVX512)
    for (size_t i = 0; i < n; ++i) {
        float fa = _cvtsh_ss(a[i]);
        float fb = _cvtsh_ss(b[i]);
        out[i] = _cvtss_sh(fa / fb, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
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
