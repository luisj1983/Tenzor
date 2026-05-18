/**
 * @file simd_kernels_avx512_f64.cpp
 * @brief AVX-512 implementations for float64 binary/unary ops and
 *        float32 neg/abs used by the runtime dispatch table.
 *
 * Compiled with -mavx512f (per set_source_files_properties in CMake).
 * Falls back to AVX2 when AVX-512 is not compiled in
 * (i.e., when this TU is compiled without -mavx512f).
 *
 * All symbols live in tenzor::cpu::avx512_f64_impl and are referenced
 * as forward declarations in simd_dispatch.cpp.
 */

#include "tenzor/backends/cpu/simd.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX2__) && !defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace tenzor {
namespace cpu {
namespace avx512_f64_impl {

// ============================================================================
// Float64 binary ops
// ============================================================================

void add_f64(const double* a, const double* b, double* out, size_t size) {
#if defined(__AVX512F__)
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        _mm512_storeu_pd(out + i, _mm512_add_pd(va, vb));
    }
    for (; i < size; ++i) out[i] = a[i] + b[i];
#elif defined(__AVX2__)
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_add_pd(va, vb));
    }
    for (; i < size; ++i) out[i] = a[i] + b[i];
#else
    for (size_t i = 0; i < size; ++i) out[i] = a[i] + b[i];
#endif
}

void sub_f64(const double* a, const double* b, double* out, size_t size) {
#if defined(__AVX512F__)
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        _mm512_storeu_pd(out + i, _mm512_sub_pd(va, vb));
    }
    for (; i < size; ++i) out[i] = a[i] - b[i];
#elif defined(__AVX2__)
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_sub_pd(va, vb));
    }
    for (; i < size; ++i) out[i] = a[i] - b[i];
#else
    for (size_t i = 0; i < size; ++i) out[i] = a[i] - b[i];
#endif
}

void mul_f64(const double* a, const double* b, double* out, size_t size) {
#if defined(__AVX512F__)
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        _mm512_storeu_pd(out + i, _mm512_mul_pd(va, vb));
    }
    for (; i < size; ++i) out[i] = a[i] * b[i];
#elif defined(__AVX2__)
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_mul_pd(va, vb));
    }
    for (; i < size; ++i) out[i] = a[i] * b[i];
#else
    for (size_t i = 0; i < size; ++i) out[i] = a[i] * b[i];
#endif
}

void div_f64(const double* a, const double* b, double* out, size_t size) {
#if defined(__AVX512F__)
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        _mm512_storeu_pd(out + i, _mm512_div_pd(va, vb));
    }
    for (; i < size; ++i) out[i] = a[i] / b[i];
#elif defined(__AVX2__)
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_div_pd(va, vb));
    }
    for (; i < size; ++i) out[i] = a[i] / b[i];
#else
    for (size_t i = 0; i < size; ++i) out[i] = a[i] / b[i];
#endif
}

// ============================================================================
// Float64 unary ops
// ============================================================================

void sqrt_f64(const double* a, double* out, size_t size) {
#if defined(__AVX512F__)
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m512d va = _mm512_loadu_pd(a + i);
        _mm512_storeu_pd(out + i, _mm512_sqrt_pd(va));
    }
    for (; i < size; ++i) out[i] = std::sqrt(a[i]);
#elif defined(__AVX2__)
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        _mm256_storeu_pd(out + i, _mm256_sqrt_pd(va));
    }
    for (; i < size; ++i) out[i] = std::sqrt(a[i]);
#else
    for (size_t i = 0; i < size; ++i) out[i] = std::sqrt(a[i]);
#endif
}

void neg_f64(const double* a, double* out, size_t size) {
#if defined(__AVX512F__)
    size_t i = 0;
    // Negate via XOR with sign bit mask
    __m512d sign_bit = _mm512_castsi512_pd(
        _mm512_set1_epi64(static_cast<int64_t>(0x8000000000000000ll)));
    for (; i + 8 <= size; i += 8) {
        __m512d va = _mm512_loadu_pd(a + i);
        _mm512_storeu_pd(out + i, _mm512_xor_pd(va, sign_bit));
    }
    for (; i < size; ++i) out[i] = -a[i];
#elif defined(__AVX2__)
    size_t i = 0;
    __m256d zero = _mm256_setzero_pd();
    for (; i + 4 <= size; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        _mm256_storeu_pd(out + i, _mm256_sub_pd(zero, va));
    }
    for (; i < size; ++i) out[i] = -a[i];
#else
    for (size_t i = 0; i < size; ++i) out[i] = -a[i];
#endif
}

void abs_f64(const double* a, double* out, size_t size) {
#if defined(__AVX512F__)
    size_t i = 0;
    // Clear sign bit via AND with 0x7FFFFFFFFFFFFFFF
    __m512d sign_mask = _mm512_castsi512_pd(
        _mm512_set1_epi64(static_cast<int64_t>(0x7FFFFFFFFFFFFFFFll)));
    for (; i + 8 <= size; i += 8) {
        __m512d va = _mm512_loadu_pd(a + i);
        _mm512_storeu_pd(out + i, _mm512_and_pd(va, sign_mask));
    }
    for (; i < size; ++i) out[i] = std::abs(a[i]);
#elif defined(__AVX2__)
    size_t i = 0;
    __m256d sign_mask = _mm256_castsi256_pd(
        _mm256_set1_epi64x(static_cast<int64_t>(0x7FFFFFFFFFFFFFFFll)));
    for (; i + 4 <= size; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        _mm256_storeu_pd(out + i, _mm256_and_pd(va, sign_mask));
    }
    for (; i < size; ++i) out[i] = std::abs(a[i]);
#else
    for (size_t i = 0; i < size; ++i) out[i] = std::abs(a[i]);
#endif
}

// ============================================================================
// Float32 unary ops (neg, abs) — AVX-512 versions
// ============================================================================

void neg_f32(const float* a, float* out, size_t size) {
#if defined(__AVX512F__)
    size_t i = 0;
    // XOR with sign bit mask
    __m512 sign_bit = _mm512_castsi512_ps(_mm512_set1_epi32(static_cast<int>(0x80000000u)));
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        _mm512_storeu_ps(out + i, _mm512_xor_ps(va, sign_bit));
    }
    for (; i < size; ++i) out[i] = -a[i];
#elif defined(__AVX2__)
    size_t i = 0;
    __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(out + i, _mm256_sub_ps(zero, va));
    }
    for (; i < size; ++i) out[i] = -a[i];
#else
    for (size_t i = 0; i < size; ++i) out[i] = -a[i];
#endif
}

void abs_f32(const float* a, float* out, size_t size) {
#if defined(__AVX512F__)
    size_t i = 0;
    __m512 sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        _mm512_storeu_ps(out + i, _mm512_and_ps(va, sign_mask));
    }
    for (; i < size; ++i) out[i] = std::abs(a[i]);
#elif defined(__AVX2__)
    size_t i = 0;
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(out + i, _mm256_and_ps(va, sign_mask));
    }
    for (; i < size; ++i) out[i] = std::abs(a[i]);
#else
    for (size_t i = 0; i < size; ++i) out[i] = std::abs(a[i]);
#endif
}

} // namespace avx512_f64_impl
} // namespace cpu
} // namespace tenzor
