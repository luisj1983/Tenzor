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
#include "tenzor/backend/omp_thresholds.hpp"
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
    constexpr int MIN_CHANNELS_PER_THREAD = 4;
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    int max_useful_threads = std::max(1, static_cast<int>(C / MIN_CHANNELS_PER_THREAD));
    int final_threads = std::min(nthreads, max_useful_threads);
#else
    int final_threads = 1;
#endif

    // Welford's online algorithm for numerically stable variance computation.
    // Each of the 16 AVX-512 lanes maintains independent (mean, m2) accumulators,
    // merged at the end using the parallel combination formula.
    #pragma omp parallel for num_threads(final_threads) if(C > 1)
    for (int64_t c = 0; c < C; c++) {
        __m512 vmean = _mm512_setzero_ps();
        __m512 vm2 = _mm512_setzero_ps();
        int64_t lane_count = 0;

        for (int64_t n = 0; n < N; n++) {
            const float* ch_ptr = input + (n * C + c) * spatial_size;
            int64_t i = 0;

            for (; i + 16 <= spatial_size; i += 16) {
                __m512 v = _mm512_loadu_ps(ch_ptr + i);
                lane_count++;
                __m512 vcount = _mm512_set1_ps(static_cast<float>(lane_count));
                __m512 delta = _mm512_sub_ps(v, vmean);
                vmean = _mm512_add_ps(vmean, _mm512_div_ps(delta, vcount));
                __m512 delta2 = _mm512_sub_ps(v, vmean);
                vm2 = _mm512_fmadd_ps(delta, delta2, vm2);
            }
        }

        // Horizontally merge the 16 SIMD lanes using parallel Welford merge
        alignas(64) float lane_means[16];
        alignas(64) float lane_m2s[16];
        _mm512_store_ps(lane_means, vmean);
        _mm512_store_ps(lane_m2s, vm2);

        float combined_mean = lane_means[0];
        float combined_m2 = lane_m2s[0];
        float combined_n = static_cast<float>(lane_count);

        for (int lane = 1; lane < 16; lane++) {
            float n_b = static_cast<float>(lane_count);
            float total_n = combined_n + n_b;
            if (total_n == 0.0f) continue;
            float delta = lane_means[lane] - combined_mean;
            combined_mean = (combined_mean * combined_n + lane_means[lane] * n_b) / total_n;
            combined_m2 = combined_m2 + lane_m2s[lane] + delta * delta * combined_n * n_b / total_n;
            combined_n = total_n;
        }

        // Fold in scalar remainder elements
        int64_t simd_covered = (spatial_size / 16) * 16;
        for (int64_t n = 0; n < N; n++) {
            const float* ch_ptr = input + (n * C + c) * spatial_size;
            for (int64_t i = simd_covered; i < spatial_size; i++) {
                combined_n += 1.0f;
                float delta = ch_ptr[i] - combined_mean;
                combined_mean += delta / combined_n;
                float delta2 = ch_ptr[i] - combined_mean;
                combined_m2 += delta * delta2;
            }
        }

        mean[c] = combined_mean;
        variance[c] = (total_elements > 0) ? combined_m2 / static_cast<float>(total_elements) : 0.0f;
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

    #pragma omp parallel for collapse(2) num_threads(final_threads) if(total_size > ::tenzor::OmpThresholds::medium())
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
