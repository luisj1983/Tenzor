/**
 * @file batchnorm_avx512.cpp
 * @brief AVX-512 implementations of BatchNorm operations
 *
 * Compiled separately with -mavx512f to enable AVX-512 in portable builds.
 * Provides 16-wide vectorized mean/variance and affine normalization.
 */

#include "tenzor/backends/cpu/simd.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    // MSVC never predefines __AVX512F__ (unlike GCC/Clang), even when the
    // project's own TENZOR_HAS_AVX512 (set by CMake based on /arch: and CPU
    // detection) is enabled; fall back to that so this file's AVX-512 kernels
    // aren't silently dropped -- with call sites elsewhere unconditionally
    // referencing them -- under MSVC.
    #if defined(__AVX512F__) || defined(TENZOR_HAS_AVX512)
        #include <immintrin.h>
        #define TENZOR_BN_HAS_AVX512
    #endif
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
        // Accumulate per lane in DOUBLE (a float mean/count stalls past 2^24
        // per-channel elements). 16 float lanes -> two __m512d halves.
        __m512d vmean_lo = _mm512_setzero_pd();
        __m512d vmean_hi = _mm512_setzero_pd();
        __m512d vm2_lo = _mm512_setzero_pd();
        __m512d vm2_hi = _mm512_setzero_pd();
        double lane_count = 0.0;

        // Single scalar Welford state for the (spatial_size % 16) tail of every
        // batch, accumulated in the SAME n-loop so the input is streamed once.
        const int64_t simd_covered = (spatial_size / 16) * 16;
        double rem_n = 0.0, rem_mean = 0.0, rem_m2 = 0.0;

        for (int64_t n = 0; n < N; n++) {
            const float* ch_ptr = input + (n * C + c) * spatial_size;
            int64_t i = 0;

            for (; i + 16 <= spatial_size; i += 16) {
                __m512 v = _mm512_loadu_ps(ch_ptr + i);
                __m512d v_lo = _mm512_cvtps_pd(_mm512_castps512_ps256(v));
                __m512d v_hi = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v, 1));
                lane_count += 1.0;
                __m512d vcount = _mm512_set1_pd(lane_count);
                __m512d delta_lo = _mm512_sub_pd(v_lo, vmean_lo);
                __m512d delta_hi = _mm512_sub_pd(v_hi, vmean_hi);
                vmean_lo = _mm512_add_pd(vmean_lo, _mm512_div_pd(delta_lo, vcount));
                vmean_hi = _mm512_add_pd(vmean_hi, _mm512_div_pd(delta_hi, vcount));
                __m512d delta2_lo = _mm512_sub_pd(v_lo, vmean_lo);
                __m512d delta2_hi = _mm512_sub_pd(v_hi, vmean_hi);
                vm2_lo = _mm512_fmadd_pd(delta_lo, delta2_lo, vm2_lo);
                vm2_hi = _mm512_fmadd_pd(delta_hi, delta2_hi, vm2_hi);
            }

            // Fold this batch's tail into the running scalar state (single pass).
            for (; i < spatial_size; i++) {
                rem_n += 1.0;
                double d = static_cast<double>(ch_ptr[i]) - rem_mean;
                rem_mean += d / rem_n;
                double d2 = static_cast<double>(ch_ptr[i]) - rem_mean;
                rem_m2 += d * d2;
            }
        }

        // Horizontally merge the 16 SIMD lanes using parallel Welford merge
        alignas(64) double lane_means[16];
        alignas(64) double lane_m2s[16];
        _mm512_store_pd(lane_means, vmean_lo);
        _mm512_store_pd(lane_means + 8, vmean_hi);
        _mm512_store_pd(lane_m2s, vm2_lo);
        _mm512_store_pd(lane_m2s + 8, vm2_hi);

        // The horizontal merge and final division run once per channel, so
        // accumulate in double: float combined_n would round once the
        // per-channel element count exceeds 2^24 and float division of the
        // population variance would diverge from the scalar/double path.
        double combined_mean = lane_means[0];
        double combined_m2 = lane_m2s[0];
        double combined_n = static_cast<double>(lane_count);

        for (int lane = 1; lane < 16; lane++) {
            double n_b = static_cast<double>(lane_count);
            double total_n = combined_n + n_b;
            if (total_n == 0.0) continue;
            double delta = static_cast<double>(lane_means[lane]) - combined_mean;
            combined_mean = (combined_mean * combined_n + static_cast<double>(lane_means[lane]) * n_b) / total_n;
            combined_m2 = combined_m2 + static_cast<double>(lane_m2s[lane]) + delta * delta * combined_n * n_b / total_n;
            combined_n = total_n;
        }

        // Merge the scalar tail Welford state (accumulated in the single n-loop
        // above) into the combined SIMD state once, via the parallel merge.
        (void)simd_covered;
        if (rem_n > 0.0) {
            double total_n = combined_n + rem_n;
            double delta = rem_mean - combined_mean;
            combined_mean = (combined_mean * combined_n + rem_mean * rem_n) / total_n;
            combined_m2 = combined_m2 + rem_m2 + delta * delta * combined_n * rem_n / total_n;
            combined_n = total_n;
        }

        mean[c] = static_cast<float>(combined_mean);
        variance[c] = (total_elements > 0) ? static_cast<float>(combined_m2 / static_cast<double>(total_elements)) : 0.0f;
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

    // Precompute per-channel scale only. Do NOT precompute bias = beta -
    // mean*scale and then compute scale*x + bias: when mean is large and
    // scale is large (tiny variance), scale*x and bias are each huge and
    // nearly cancel, so the result inherits ~ulp(scale*mean) of rounding
    // noise instead of the true near-zero answer -- catastrophic
    // cancellation. Compute (x - mean) * scale + beta directly instead,
    // which subtracts the raw (similar-magnitude) x and mean first, matching
    // the numerically stable form the CUDA kernel already uses.
    std::vector<float> scale(C);
    for (int64_t c = 0; c < C; c++) {
        float invstd = 1.0f / std::sqrt(variance[c] + epsilon);
        scale[c] = gamma[c] * invstd;
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
            __m512 vmean = _mm512_set1_ps(mean[c]);
            __m512 vbeta = _mm512_set1_ps(beta[c]);

            const float* in_ptr = input + (n * C + c) * spatial_size;
            float* out_ptr = output + (n * C + c) * spatial_size;

            int64_t hw = 0;
            // Process 16 floats at a time with AVX-512 FMA
            for (; hw + 16 <= spatial_size; hw += 16) {
                __m512 vin = _mm512_loadu_ps(in_ptr + hw);
                __m512 vdiff = _mm512_sub_ps(vin, vmean);
                __m512 vout = _mm512_fmadd_ps(vscale, vdiff, vbeta);
                _mm512_storeu_ps(out_ptr + hw, vout);
            }
            // Handle remainder
            for (; hw < spatial_size; hw++) {
                out_ptr[hw] = scale[c] * (in_ptr[hw] - mean[c]) + beta[c];
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

// Audit item C.1: non-AVX-512 builds must NOT silently provide empty
// AVX-512 stubs that write nothing to the output buffer.  The comment
// admitted "should never be called" but a CMake misconfiguration or
// future refactor could route into them.  Make the failure mode loud.
[[noreturn]] static void avx512_unavailable(const char* func) {
    std::fprintf(stderr,
        "[Tenzor] %s called on a build without AVX-512 support — "
        "rebuild with -DTENZOR_HAVE_AVX512=ON or use the scalar / AVX2 "
        "fallback path.\n", func);
    std::abort();
}

void batchnorm_mean_var_f32(const float*, float*, float*, int64_t, int64_t, int64_t, int64_t) {
    avx512_unavailable("batchnorm_mean_var_f32");
}
void batchnorm_forward_affine_f32(const float*, float*, const float*, const float*, const float*, const float*, float, int64_t, int64_t, int64_t, int64_t) {
    avx512_unavailable("batchnorm_forward_affine_f32");
}
void batchnorm_normalize_f32(const float*, float*, const float*, const float*, float, int64_t, int64_t, int64_t, int64_t) {
    avx512_unavailable("batchnorm_normalize_f32");
}

#endif

} // namespace avx512
} // namespace cpu
} // namespace tenzor
