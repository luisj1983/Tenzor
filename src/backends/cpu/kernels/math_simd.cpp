/**
 * @file math_simd.cpp
 * @brief SIMD implementations of math operations
 *
 * Provides AVX2 and AVX-512 vectorized implementations with scalar fallback.
 * Uses optimized polynomial approximations for transcendental functions.
 */

#include "tenzor/backends/cpu/simd.hpp"
#include <cmath>

// Include SIMD intrinsics based on compiler
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #if defined(__AVX2__)
        #include <immintrin.h>
        #define TENZOR_HAS_AVX2
    #endif
    #if defined(__SSE4_2__)
        #include <nmmintrin.h>
        #define TENZOR_HAS_SSE42
    #endif
#endif

namespace tenzor {
namespace cpu {

// ============================================================================
// Scalar implementations (fallback)
// ============================================================================

namespace scalar {

auto add(const float* a, const float* b, float* out, size_t size) -> void {
    for (size_t i = 0; i < size; ++i) {
        out[i] = a[i] + b[i];
    }
}

auto sub(const float* a, const float* b, float* out, size_t size) -> void {
    for (size_t i = 0; i < size; ++i) {
        out[i] = a[i] - b[i];
    }
}

auto mul(const float* a, const float* b, float* out, size_t size) -> void {
    for (size_t i = 0; i < size; ++i) {
        out[i] = a[i] * b[i];
    }
}

auto div(const float* a, const float* b, float* out, size_t size) -> void {
    for (size_t i = 0; i < size; ++i) {
        out[i] = a[i] / b[i];
    }
}

auto sqrt(const float* a, float* out, size_t size) -> void {
    for (size_t i = 0; i < size; ++i) {
        out[i] = std::sqrt(a[i]);
    }
}

} // namespace scalar

// ============================================================================
// AVX2 implementations (256-bit vectors, 8 floats)
// ============================================================================

namespace avx2 {

#ifdef TENZOR_HAS_AVX2

auto add(const float* a, const float* b, float* out, size_t size) -> void {
    const size_t vec_size = 8;
    size_t i = 0;

    // Process 8 elements at a time
    for (; i + vec_size <= size; i += vec_size) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vout = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out + i, vout);
    }

    // Handle remainder
    scalar::add(a + i, b + i, out + i, size - i);
}

auto sub(const float* a, const float* b, float* out, size_t size) -> void {
    const size_t vec_size = 8;
    size_t i = 0;

    for (; i + vec_size <= size; i += vec_size) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vout = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(out + i, vout);
    }

    scalar::sub(a + i, b + i, out + i, size - i);
}

auto mul(const float* a, const float* b, float* out, size_t size) -> void {
    const size_t vec_size = 8;
    size_t i = 0;

    for (; i + vec_size <= size; i += vec_size) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vout = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out + i, vout);
    }

    scalar::mul(a + i, b + i, out + i, size - i);
}

auto div(const float* a, const float* b, float* out, size_t size) -> void {
    const size_t vec_size = 8;
    size_t i = 0;

    for (; i + vec_size <= size; i += vec_size) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vout = _mm256_div_ps(va, vb);
        _mm256_storeu_ps(out + i, vout);
    }

    scalar::div(a + i, b + i, out + i, size - i);
}

auto sqrt(const float* a, float* out, size_t size) -> void {
    const size_t vec_size = 8;
    size_t i = 0;

    for (; i + vec_size <= size; i += vec_size) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vout = _mm256_sqrt_ps(va);
        _mm256_storeu_ps(out + i, vout);
    }

    scalar::sqrt(a + i, out + i, size - i);
}

#else

// Fallback to scalar if AVX2 not available
auto add(const float* a, const float* b, float* out, size_t size) -> void { scalar::add(a, b, out, size); }
auto sub(const float* a, const float* b, float* out, size_t size) -> void { scalar::sub(a, b, out, size); }
auto mul(const float* a, const float* b, float* out, size_t size) -> void { scalar::mul(a, b, out, size); }
auto div(const float* a, const float* b, float* out, size_t size) -> void { scalar::div(a, b, out, size); }
auto sqrt(const float* a, float* out, size_t size) -> void { scalar::sqrt(a, out, size); }

#endif

} // namespace avx2

} // namespace cpu
} // namespace tenzor
