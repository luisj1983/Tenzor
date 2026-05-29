/**
 * @file activations_simd.cpp
 * @brief SIMD implementations of activation functions
 *
 * Uses optimized polynomial approximations for sigmoid, tanh, and GELU.
 */

#include "tenzor/backends/cpu/simd.hpp"
#include "tenzor/backend/runtime_simd.hpp"
#include "simd_fast_math.hpp"
#include <cmath>
#include <algorithm>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
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
    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))) — matches the canonical
    // gelu_kernel and PyTorch default (approximate='none').
    constexpr float inv_sqrt2 = 0.70710678f;
    for (size_t i = 0; i < size; ++i) {
        float x = a[i];
        out[i] = 0.5f * x * (1.0f + std::erf(x * inv_sqrt2));
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
    // Use optimized vectorized sigmoid
    fast_math::sigmoid_batch_avx2(a, out, size);
}

auto tanh(const float* a, float* out, size_t size) -> void {
    // Use optimized vectorized tanh
    fast_math::tanh_batch_avx2(a, out, size);
}

auto gelu(const float* a, float* out, size_t size) -> void {
    // Use optimized vectorized GELU
    fast_math::gelu_batch_avx2(a, out, size);
}

#else

auto relu(const float* a, float* out, size_t size) -> void { scalar::relu(a, out, size); }
auto sigmoid(const float* a, float* out, size_t size) -> void { scalar::sigmoid(a, out, size); }
auto tanh(const float* a, float* out, size_t size) -> void { scalar::tanh(a, out, size); }
auto gelu(const float* a, float* out, size_t size) -> void { scalar::gelu(a, out, size); }

#endif

} // namespace avx2

// ============================================================================
// Runtime dispatch
// AVX-512 implementations are in activations_simd_avx512.cpp (separate TU for
// portable builds that compile with -mavx512f per-file).
// ============================================================================

namespace simd {

auto relu(const float* a, float* out, size_t size) -> void {
    const auto& cpu = ::tenzor::backend::get_simd_features();
    if (cpu.avx512f) {
        avx512::relu(a, out, size);
    } else if (cpu.avx2) {
        avx2::relu(a, out, size);
    } else {
        scalar::relu(a, out, size);
    }
}

auto sigmoid(const float* a, float* out, size_t size) -> void {
    const auto& cpu = ::tenzor::backend::get_simd_features();
    if (cpu.avx512f) {
        avx512::sigmoid(a, out, size);
    } else if (cpu.avx2) {
        avx2::sigmoid(a, out, size);
    } else {
        scalar::sigmoid(a, out, size);
    }
}

auto tanh(const float* a, float* out, size_t size) -> void {
    const auto& cpu = ::tenzor::backend::get_simd_features();
    if (cpu.avx512f) {
        avx512::tanh(a, out, size);
    } else if (cpu.avx2) {
        avx2::tanh(a, out, size);
    } else {
        scalar::tanh(a, out, size);
    }
}

auto gelu(const float* a, float* out, size_t size) -> void {
    const auto& cpu = ::tenzor::backend::get_simd_features();
    if (cpu.avx512f) {
        avx512::gelu(a, out, size);
    } else if (cpu.avx2) {
        avx2::gelu(a, out, size);
    } else {
        scalar::gelu(a, out, size);
    }
}

} // namespace simd

} // namespace cpu
} // namespace tenzor
