/**
 * @file activations_simd_avx512.cpp
 * @brief AVX-512 implementations of activation functions
 *
 * Compiled separately with -mavx512f to enable AVX-512 in portable builds.
 * In native builds, inherits flags from -march=native.
 */

#include "tenzor/backends/cpu/simd.hpp"
#include "simd_fast_math.hpp"

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
    fast_math::sigmoid_batch_avx512(a, out, size);
}

auto tanh(const float* a, float* out, size_t size) -> void {
    fast_math::tanh_batch_avx512(a, out, size);
}

auto gelu(const float* a, float* out, size_t size) -> void {
    fast_math::gelu_batch_avx512(a, out, size);
}

#else

// Fallback to AVX2 if AVX-512 not available at compile time
auto relu(const float* a, float* out, size_t size) -> void { avx2::relu(a, out, size); }
auto sigmoid(const float* a, float* out, size_t size) -> void { avx2::sigmoid(a, out, size); }
auto tanh(const float* a, float* out, size_t size) -> void { avx2::tanh(a, out, size); }
auto gelu(const float* a, float* out, size_t size) -> void { avx2::gelu(a, out, size); }

#endif

} // namespace avx512

} // namespace cpu
} // namespace tenzor
