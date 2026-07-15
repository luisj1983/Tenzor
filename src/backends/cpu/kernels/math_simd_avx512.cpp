/**
 * @file math_simd_avx512.cpp
 * @brief AVX-512 implementations of math operations
 *
 * Compiled separately with -mavx512f to enable AVX-512 in portable builds.
 * In native builds, inherits flags from -march=native.
 */

#include "tenzor/backends/cpu/simd.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #if defined(__AVX512F__)
        #include <immintrin.h>
        #ifndef TENZOR_HAS_AVX512
        #define TENZOR_HAS_AVX512
        #endif
    #endif
    #if defined(__AVX2__)
        #include <immintrin.h>
        #ifndef TENZOR_HAS_AVX2
        #define TENZOR_HAS_AVX2
        #endif
    #endif
#endif

namespace tenzor {
namespace cpu {

namespace avx512 {

#ifdef TENZOR_HAS_AVX512

auto add(const float* a, const float* b, float* out, size_t size) -> void {
    const size_t vec_size = 16;
    size_t i = 0;

    for (; i + vec_size <= size; i += vec_size) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vout = _mm512_add_ps(va, vb);
        _mm512_storeu_ps(out + i, vout);
    }

    avx2::add(a + i, b + i, out + i, size - i);
}

auto sub(const float* a, const float* b, float* out, size_t size) -> void {
    const size_t vec_size = 16;
    size_t i = 0;

    for (; i + vec_size <= size; i += vec_size) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vout = _mm512_sub_ps(va, vb);
        _mm512_storeu_ps(out + i, vout);
    }

    avx2::sub(a + i, b + i, out + i, size - i);
}

auto mul(const float* a, const float* b, float* out, size_t size) -> void {
    const size_t vec_size = 16;
    size_t i = 0;

    for (; i + vec_size <= size; i += vec_size) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vout = _mm512_mul_ps(va, vb);
        _mm512_storeu_ps(out + i, vout);
    }

    avx2::mul(a + i, b + i, out + i, size - i);
}

auto div(const float* a, const float* b, float* out, size_t size) -> void {
    const size_t vec_size = 16;
    size_t i = 0;

    for (; i + vec_size <= size; i += vec_size) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vout = _mm512_div_ps(va, vb);
        _mm512_storeu_ps(out + i, vout);
    }

    avx2::div(a + i, b + i, out + i, size - i);
}

auto sqrt(const float* a, float* out, size_t size) -> void {
    const size_t vec_size = 16;
    size_t i = 0;

    for (; i + vec_size <= size; i += vec_size) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vout = _mm512_sqrt_ps(va);
        _mm512_storeu_ps(out + i, vout);
    }

    avx2::sqrt(a + i, out + i, size - i);
}

#else

// Fallback to AVX2 if AVX-512 not available at compile time
auto add(const float* a, const float* b, float* out, size_t size) -> void { avx2::add(a, b, out, size); }
auto sub(const float* a, const float* b, float* out, size_t size) -> void { avx2::sub(a, b, out, size); }
auto mul(const float* a, const float* b, float* out, size_t size) -> void { avx2::mul(a, b, out, size); }
auto div(const float* a, const float* b, float* out, size_t size) -> void { avx2::div(a, b, out, size); }
auto sqrt(const float* a, float* out, size_t size) -> void { avx2::sqrt(a, out, size); }

#endif

} // namespace avx512

} // namespace cpu
} // namespace tenzor
