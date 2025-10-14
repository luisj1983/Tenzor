/**
 * @file activations_simd.cpp
 * @brief SIMD implementations of activation functions
 */

#include "tenzor/backends/cpu/simd.hpp"
#include <cmath>
#include <algorithm>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #if defined(__AVX512F__)
        #include <immintrin.h>
        #define TENZOR_HAS_AVX512
    #endif
    #if defined(__AVX2__)
        #include <immintrin.h>
        #define TENZOR_HAS_AVX2
    #endif
#endif

namespace tenzor {
namespace cpu {

// ============================================================================
// Scalar implementations
// ============================================================================

namespace scalar {

auto relu(const float* a, float* out, size_t size) -> void {
    for (size_t i = 0; i < size; ++i) {
        out[i] = std::max(0.0f, a[i]);
    }
}

auto sigmoid(const float* a, float* out, size_t size) -> void {
    for (size_t i = 0; i < size; ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-a[i]));
    }
}

auto tanh(const float* a, float* out, size_t size) -> void {
    for (size_t i = 0; i < size; ++i) {
        out[i] = std::tanh(a[i]);
    }
}

auto gelu(const float* a, float* out, size_t size) -> void {
    constexpr float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/pi)
    constexpr float coeff = 0.044715f;

    for (size_t i = 0; i < size; ++i) {
        float x = a[i];
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        out[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

} // namespace scalar

// ============================================================================
// AVX2 implementations
// ============================================================================

namespace avx2 {

#ifdef TENZOR_HAS_AVX2

auto relu(const float* a, float* out, size_t size) -> void {
    const size_t vec_size = 8;
    size_t i = 0;

    __m256 zero = _mm256_setzero_ps();

    for (; i + vec_size <= size; i += vec_size) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vout = _mm256_max_ps(zero, va);
        _mm256_storeu_ps(out + i, vout);
    }

    scalar::relu(a + i, out + i, size - i);
}

auto sigmoid(const float* a, float* out, size_t size) -> void {
    // Sigmoid requires exp, which is complex to vectorize efficiently
    // Use scalar for now
    scalar::sigmoid(a, out, size);
}

auto tanh(const float* a, float* out, size_t size) -> void {
    // Tanh can be approximated, but scalar version is simpler
    scalar::tanh(a, out, size);
}

auto gelu(const float* a, float* out, size_t size) -> void {
    // GELU requires tanh and complex arithmetic
    // Use scalar for now
    scalar::gelu(a, out, size);
}

#else

auto relu(const float* a, float* out, size_t size) -> void { scalar::relu(a, out, size); }
auto sigmoid(const float* a, float* out, size_t size) -> void { scalar::sigmoid(a, out, size); }
auto tanh(const float* a, float* out, size_t size) -> void { scalar::tanh(a, out, size); }
auto gelu(const float* a, float* out, size_t size) -> void { scalar::gelu(a, out, size); }

#endif

} // namespace avx2

// ============================================================================
// AVX-512 implementations
// ============================================================================

namespace avx512 {

#ifdef TENZOR_HAS_AVX512

auto relu(const float* a, float* out, size_t size) -> void {
    const size_t vec_size = 16;
    size_t i = 0;

    __m512 zero = _mm512_setzero_ps();

    for (; i + vec_size <= size; i += vec_size) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vout = _mm512_max_ps(zero, va);
        _mm512_storeu_ps(out + i, vout);
    }

    avx2::relu(a + i, out + i, size - i);
}

auto sigmoid(const float* a, float* out, size_t size) -> void {
    scalar::sigmoid(a, out, size);
}

auto tanh(const float* a, float* out, size_t size) -> void {
    scalar::tanh(a, out, size);
}

auto gelu(const float* a, float* out, size_t size) -> void {
    scalar::gelu(a, out, size);
}

#else

auto relu(const float* a, float* out, size_t size) -> void { avx2::relu(a, out, size); }
auto sigmoid(const float* a, float* out, size_t size) -> void { avx2::sigmoid(a, out, size); }
auto tanh(const float* a, float* out, size_t size) -> void { avx2::tanh(a, out, size); }
auto gelu(const float* a, float* out, size_t size) -> void { avx2::gelu(a, out, size); }

#endif

} // namespace avx512

// ============================================================================
// Runtime dispatch
// ============================================================================

namespace simd {

auto relu(const float* a, float* out, size_t size) -> void {
    const auto& cpu = CPUInfo::get();
    if (cpu.has_avx512()) {
        avx512::relu(a, out, size);
    } else if (cpu.has_avx2()) {
        avx2::relu(a, out, size);
    } else {
        scalar::relu(a, out, size);
    }
}

auto sigmoid(const float* a, float* out, size_t size) -> void {
    scalar::sigmoid(a, out, size);
}

auto tanh(const float* a, float* out, size_t size) -> void {
    scalar::tanh(a, out, size);
}

auto gelu(const float* a, float* out, size_t size) -> void {
    scalar::gelu(a, out, size);
}

} // namespace simd

} // namespace cpu
} // namespace tenzor
