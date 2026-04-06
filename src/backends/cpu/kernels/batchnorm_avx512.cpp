/**
 * @file batchnorm_avx512.cpp
 * @brief AVX-512 implementations of BatchNorm operations
 *
 * Compiled separately with -mavx512f to enable AVX-512 in portable builds.
 * Provides 16-wide vectorized mean/variance and affine normalization.
 */

#include "tenzor/backends/cpu/simd.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #if defined(__AVX512F__)
        #include <immintrin.h>
        #define TENZOR_BN_HAS_AVX512
    #endif
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor {
namespace cpu {
namespace avx512 {

#ifdef TENZOR_BN_HAS_AVX512

void batchnorm_mean_var_f32(
    const float* __restrict input,
    float* __restrict mean,
    float* __restrict variance,
    int64_t N, int64_t C, int64_t H, int64_t W)
{
    int64_t spatial_size = H * W;
    int64_t total_elements = N * spatial_size;
    float inv_total = 1.0f / static_cast<float>(total_elements);

    constexpr int MIN_CHANNELS_PER_THREAD = 4;
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    int max_useful_threads = std::max(1, static_cast<int>(C / MIN_CHANNELS_PER_THREAD));
    int final_threads = std::min(nthreads, max_useful_threads);
#else
    int final_threads = 1;
#endif

    // Single pass: compute sum and sum-of-squares simultaneously,
    // then derive variance as E[X^2] - E[X]^2. Halves memory bandwidth.
    #pragma omp parallel for num_threads(final_threads) if(C > 1)
    for (int64_t c = 0; c < C; c++) {
        __m512 vsum = _mm512_setzero_ps();
        __m512 vsum_sq = _mm512_setzero_ps();
        float scalar_tail = 0.0f;
        float scalar_sq_tail = 0.0f;

        for (int64_t n = 0; n < N; n++) {
            const float* ch_ptr = input + (n * C + c) * spatial_size;
            int64_t i = 0;

            for (; i + 16 <= spatial_size; i += 16) {
                __m512 v = _mm512_loadu_ps(ch_ptr + i);
                vsum = _mm512_add_ps(vsum, v);
                vsum_sq = _mm512_fmadd_ps(v, v, vsum_sq);
            }
            for (; i < spatial_size; i++) {
                float val = ch_ptr[i];
                scalar_tail += val;
                scalar_sq_tail += val * val;
            }
        }

        float channel_sum = _mm512_reduce_add_ps(vsum) + scalar_tail;
        float channel_mean = channel_sum * inv_total;
        mean[c] = channel_mean;

        float channel_sum_sq = _mm512_reduce_add_ps(vsum_sq) + scalar_sq_tail;
        variance[c] = channel_sum_sq * inv_total - channel_mean * channel_mean;
    }
}

void batchnorm_forward_affine_f32(
    const float* __restrict input,
    float* __restrict output,
    const float* mean,
    const float* variance,
    const float* gamma,
    const float* beta,
    float epsilon,
    int64_t N, int64_t C, int64_t H, int64_t W)
{
    int64_t spatial_size = H * W;
    int64_t total_size = N * C * spatial_size;

    // Precompute per-channel scale and bias
    std::vector<float> scale(C), bias_vec(C);
    for (int64_t c = 0; c < C; c++) {
        float invstd = 1.0f / std::sqrt(variance[c] + epsilon);
        scale[c] = gamma[c] * invstd;
        bias_vec[c] = beta[c] - mean[c] * scale[c];
    }

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    int effective_threads = std::min({nthreads, static_cast<int>(total_size / 65536), 4});
    int final_threads = std::max(1, effective_threads);
#else
    int final_threads = 1;
#endif

    #pragma omp parallel for collapse(2) num_threads(final_threads) if(total_size > 10000)
    for (int64_t n = 0; n < N; n++) {
        for (int64_t c = 0; c < C; c++) {
            __m512 vscale = _mm512_set1_ps(scale[c]);
            __m512 vbias = _mm512_set1_ps(bias_vec[c]);

            const float* in_ptr = input + (n * C + c) * spatial_size;
            float* out_ptr = output + (n * C + c) * spatial_size;

            int64_t hw = 0;
            // Process 16 floats at a time with AVX-512 FMA
            for (; hw + 16 <= spatial_size; hw += 16) {
                __m512 vin = _mm512_loadu_ps(in_ptr + hw);
                __m512 vout = _mm512_fmadd_ps(vscale, vin, vbias);
                _mm512_storeu_ps(out_ptr + hw, vout);
            }
            // Handle remainder
            for (; hw < spatial_size; hw++) {
                out_ptr[hw] = scale[c] * in_ptr[hw] + bias_vec[c];
            }
        }
    }
}

void batchnorm_normalize_f32(
    const float* __restrict input,
    float* __restrict output,
    const float* mean,
    const float* variance,
    float epsilon,
    int64_t N, int64_t C, int64_t H, int64_t W)
{
    int64_t spatial_size = H * W;

    // Precompute per-channel inverse standard deviation
    std::vector<float> invstd(C);
    for (int64_t c = 0; c < C; c++) {
        invstd[c] = 1.0f / std::sqrt(variance[c] + epsilon);
    }

    #pragma omp parallel for collapse(2) if(N * C > 4)
    for (int64_t n = 0; n < N; n++) {
        for (int64_t c = 0; c < C; c++) {
            __m512 vmean = _mm512_set1_ps(mean[c]);
            __m512 vinvstd = _mm512_set1_ps(invstd[c]);

            const float* in_ptr = input + (n * C + c) * spatial_size;
            float* out_ptr = output + (n * C + c) * spatial_size;

            int64_t hw = 0;
            for (; hw + 16 <= spatial_size; hw += 16) {
                __m512 vin = _mm512_loadu_ps(in_ptr + hw);
                __m512 vout = _mm512_mul_ps(_mm512_sub_ps(vin, vmean), vinvstd);
                _mm512_storeu_ps(out_ptr + hw, vout);
            }
            for (; hw < spatial_size; hw++) {
                out_ptr[hw] = (in_ptr[hw] - mean[c]) * invstd[c];
            }
        }
    }
}

#else

// Stubs when AVX-512 is not available (should never be called)
void batchnorm_mean_var_f32(const float*, float*, float*, int64_t, int64_t, int64_t, int64_t) {}
void batchnorm_forward_affine_f32(const float*, float*, const float*, const float*, const float*, const float*, float, int64_t, int64_t, int64_t, int64_t) {}
void batchnorm_normalize_f32(const float*, float*, const float*, const float*, float, int64_t, int64_t, int64_t, int64_t) {}

#endif

} // namespace avx512
} // namespace cpu
} // namespace tenzor
