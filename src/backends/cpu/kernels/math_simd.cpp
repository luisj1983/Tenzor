/**
 * @file math_simd.cpp
 * @brief SIMD implementations of math operations
 *
 * Provides AVX2 and AVX-512 vectorized implementations with scalar fallback.
 */

#include "tenzor/backends/cpu/simd.hpp"
#include <cmath>
#include <algorithm>

// Include SIMD intrinsics based on compiler
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #if defined(__AVX512F__)
        #include <immintrin.h>
        #define TENZOR_HAS_AVX512
    #endif
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

auto exp(const float* a, float* out, size_t size) -> void {
    for (size_t i = 0; i < size; ++i) {
        out[i] = std::exp(a[i]);
    }
}

auto log(const float* a, float* out, size_t size) -> void {
    for (size_t i = 0; i < size; ++i) {
        out[i] = std::log(a[i]);
    }
}

auto fma(const float* a, const float* b, const float* c, float* out, size_t size) -> void {
    for (size_t i = 0; i < size; ++i) {
        out[i] = a[i] * b[i] + c[i];
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

auto exp(const float* a, float* out, size_t size) -> void {
    // Use scalar exp for now (vectorized exp is complex)
    // Can be optimized with polynomial approximation
    scalar::exp(a, out, size);
}

auto log(const float* a, float* out, size_t size) -> void {
    // Use scalar log for now (vectorized log is complex)
    // Can be optimized with polynomial approximation
    scalar::log(a, out, size);
}

auto fma(const float* a, const float* b, const float* c, float* out, size_t size) -> void {
    const size_t vec_size = 8;
    size_t i = 0;

    for (; i + vec_size <= size; i += vec_size) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
        __m256 vout = _mm256_fmadd_ps(va, vb, vc);
        _mm256_storeu_ps(out + i, vout);
    }

    scalar::fma(a + i, b + i, c + i, out + i, size - i);
}

#else

// Fallback to scalar if AVX2 not available
auto add(const float* a, const float* b, float* out, size_t size) -> void { scalar::add(a, b, out, size); }
auto sub(const float* a, const float* b, float* out, size_t size) -> void { scalar::sub(a, b, out, size); }
auto mul(const float* a, const float* b, float* out, size_t size) -> void { scalar::mul(a, b, out, size); }
auto div(const float* a, const float* b, float* out, size_t size) -> void { scalar::div(a, b, out, size); }
auto sqrt(const float* a, float* out, size_t size) -> void { scalar::sqrt(a, out, size); }
auto exp(const float* a, float* out, size_t size) -> void { scalar::exp(a, out, size); }
auto log(const float* a, float* out, size_t size) -> void { scalar::log(a, out, size); }
auto fma(const float* a, const float* b, const float* c, float* out, size_t size) -> void { scalar::fma(a, b, c, out, size); }

#endif

} // namespace avx2

// ============================================================================
// AVX-512 implementations (512-bit vectors, 16 floats)
// ============================================================================

namespace avx512 {

#ifdef TENZOR_HAS_AVX512

auto add(const float* a, const float* b, float* out, size_t size) -> void {
    const size_t vec_size = 16;
    size_t i = 0;

    // Process 16 elements at a time
    for (; i + vec_size <= size; i += vec_size) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vout = _mm512_add_ps(va, vb);
        _mm512_storeu_ps(out + i, vout);
    }

    // Handle remainder with AVX2 or scalar
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

auto exp(const float* a, float* out, size_t size) -> void {
    // Use scalar exp for now
    scalar::exp(a, out, size);
}

auto log(const float* a, float* out, size_t size) -> void {
    // Use scalar log for now
    scalar::log(a, out, size);
}

auto fma(const float* a, const float* b, const float* c, float* out, size_t size) -> void {
    const size_t vec_size = 16;
    size_t i = 0;

    for (; i + vec_size <= size; i += vec_size) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_loadu_ps(c + i);
        __m512 vout = _mm512_fmadd_ps(va, vb, vc);
        _mm512_storeu_ps(out + i, vout);
    }

    avx2::fma(a + i, b + i, c + i, out + i, size - i);
}

#else

// Fallback to AVX2 if AVX-512 not available
auto add(const float* a, const float* b, float* out, size_t size) -> void { avx2::add(a, b, out, size); }
auto sub(const float* a, const float* b, float* out, size_t size) -> void { avx2::sub(a, b, out, size); }
auto mul(const float* a, const float* b, float* out, size_t size) -> void { avx2::mul(a, b, out, size); }
auto div(const float* a, const float* b, float* out, size_t size) -> void { avx2::div(a, b, out, size); }
auto sqrt(const float* a, float* out, size_t size) -> void { avx2::sqrt(a, out, size); }
auto exp(const float* a, float* out, size_t size) -> void { avx2::exp(a, out, size); }
auto log(const float* a, float* out, size_t size) -> void { avx2::log(a, out, size); }
auto fma(const float* a, const float* b, const float* c, float* out, size_t size) -> void { avx2::fma(a, b, c, out, size); }

#endif

} // namespace avx512

// ============================================================================
// Runtime dispatch based on CPU features
// ============================================================================

namespace simd {

auto add(const float* a, const float* b, float* out, size_t size) -> void {
    const auto& cpu = CPUInfo::get();
    if (cpu.has_avx512()) {
        avx512::add(a, b, out, size);
    } else if (cpu.has_avx2()) {
        avx2::add(a, b, out, size);
    } else {
        scalar::add(a, b, out, size);
    }
}

auto sub(const float* a, const float* b, float* out, size_t size) -> void {
    const auto& cpu = CPUInfo::get();
    if (cpu.has_avx512()) {
        avx512::sub(a, b, out, size);
    } else if (cpu.has_avx2()) {
        avx2::sub(a, b, out, size);
    } else {
        scalar::sub(a, b, out, size);
    }
}

auto mul(const float* a, const float* b, float* out, size_t size) -> void {
    const auto& cpu = CPUInfo::get();
    if (cpu.has_avx512()) {
        avx512::mul(a, b, out, size);
    } else if (cpu.has_avx2()) {
        avx2::mul(a, b, out, size);
    } else {
        scalar::mul(a, b, out, size);
    }
}

auto div(const float* a, const float* b, float* out, size_t size) -> void {
    const auto& cpu = CPUInfo::get();
    if (cpu.has_avx512()) {
        avx512::div(a, b, out, size);
    } else if (cpu.has_avx2()) {
        avx2::div(a, b, out, size);
    } else {
        scalar::div(a, b, out, size);
    }
}

auto sqrt(const float* a, float* out, size_t size) -> void {
    const auto& cpu = CPUInfo::get();
    if (cpu.has_avx512()) {
        avx512::sqrt(a, out, size);
    } else if (cpu.has_avx2()) {
        avx2::sqrt(a, out, size);
    } else {
        scalar::sqrt(a, out, size);
    }
}

auto exp(const float* a, float* out, size_t size) -> void {
    scalar::exp(a, out, size);
}

auto log(const float* a, float* out, size_t size) -> void {
    scalar::log(a, out, size);
}

auto fma(const float* a, const float* b, const float* c, float* out, size_t size) -> void {
    const auto& cpu = CPUInfo::get();
    if (cpu.has_avx512()) {
        avx512::fma(a, b, c, out, size);
    } else if (cpu.has_avx2()) {
        avx2::fma(a, b, c, out, size);
    } else {
        scalar::fma(a, b, c, out, size);
    }
}

} // namespace simd

} // namespace cpu
} // namespace tenzor
