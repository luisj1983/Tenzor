#include "tenzor/core/tensor.hpp"
#include "tenzor/utils/config.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD intrinsics for vectorized reductions
#if defined(__AVX512F__)
#include <immintrin.h>
#define TENZOR_REDUCTION_AVX512 1
#elif defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#define TENZOR_REDUCTION_AVX2 1
#elif defined(__SSE2__)
#include <emmintrin.h>
#define TENZOR_REDUCTION_SSE2 1
#endif

// Audit item F.5: include OmpThresholds at file scope (NOT inside the
// `namespace tenzor::cpu { ... }` below — doing so would nest the
// struct under tenzor::cpu::tenzor::OmpThresholds and break the
// `::tenzor::OmpThresholds` references in the macro below).
#include "tenzor/backend/omp_thresholds.hpp"

namespace tenzor {
namespace cpu {

// Use adaptive OpenMP thresholds scaled to thread count
#define REDUCTION_OMP_THRESHOLD static_cast<int64_t>(::tenzor::OmpThresholds::medium())

// ============================================================================
// SIMD Horizontal Reduction Helpers
// ============================================================================

#ifdef TENZOR_REDUCTION_AVX512

// Horizontal sum of 16 floats in AVX-512 register -> single float
__attribute__((target("avx512f")))
static inline float hsum_avx512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

// Horizontal max of 16 floats in AVX-512 register -> single float
__attribute__((target("avx512f")))
static inline float hmax_avx512(__m512 v) {
    return _mm512_reduce_max_ps(v);
}

// Horizontal min of 16 floats in AVX-512 register -> single float
__attribute__((target("avx512f")))
static inline float hmin_avx512(__m512 v) {
    return _mm512_reduce_min_ps(v);
}

// AVX-512 vectorized sum for float32 with Kahan compensation
__attribute__((target("avx512f")))
static float simd_sum_f32_avx512(const float* data, int64_t n) {
    constexpr int64_t VEC_SIZE = 16;
    const int64_t vec_end = (n / VEC_SIZE) * VEC_SIZE;

    __m512 vsum = _mm512_setzero_ps();
    __m512 vcomp = _mm512_setzero_ps();  // Kahan compensation

    // Main vectorized loop with Kahan summation
    for (int64_t i = 0; i < vec_end; i += VEC_SIZE) {
        __m512 v = _mm512_loadu_ps(data + i);
        __m512 y = _mm512_sub_ps(v, vcomp);
        __m512 t = _mm512_add_ps(vsum, y);
        vcomp = _mm512_sub_ps(_mm512_sub_ps(t, vsum), y);
        vsum = t;
    }

    // Horizontal sum of vector accumulator
    float sum = hsum_avx512(vsum);
    float comp = hsum_avx512(vcomp);

    // Handle remaining elements with scalar Kahan
    for (int64_t i = vec_end; i < n; i++) {
        float y = data[i] - comp;
        float t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }

    return sum;
}

// AVX-512 vectorized max for float32
__attribute__((target("avx512f")))
static float simd_max_f32_avx512(const float* data, int64_t n) {
    if (n == 0) return -std::numeric_limits<float>::infinity();

    constexpr int64_t VEC_SIZE = 16;
    const int64_t vec_end = (n / VEC_SIZE) * VEC_SIZE;

    __m512 vmax = _mm512_set1_ps(-std::numeric_limits<float>::infinity());

    for (int64_t i = 0; i < vec_end; i += VEC_SIZE) {
        __m512 v = _mm512_loadu_ps(data + i);
        vmax = _mm512_max_ps(vmax, v);
    }

    float max_val = hmax_avx512(vmax);

    for (int64_t i = vec_end; i < n; i++) {
        if (std::isnan(data[i]) || data[i] > max_val) max_val = data[i];
    }

    return max_val;
}

// AVX-512 vectorized min for float32
__attribute__((target("avx512f")))
static float simd_min_f32_avx512(const float* data, int64_t n) {
    if (n == 0) return std::numeric_limits<float>::infinity();

    constexpr int64_t VEC_SIZE = 16;
    const int64_t vec_end = (n / VEC_SIZE) * VEC_SIZE;

    __m512 vmin = _mm512_set1_ps(std::numeric_limits<float>::infinity());

    for (int64_t i = 0; i < vec_end; i += VEC_SIZE) {
        __m512 v = _mm512_loadu_ps(data + i);
        vmin = _mm512_min_ps(vmin, v);
    }

    float min_val = hmin_avx512(vmin);

    for (int64_t i = vec_end; i < n; i++) {
        if (std::isnan(data[i]) || data[i] < min_val) min_val = data[i];
    }

    return min_val;
}

#endif // TENZOR_REDUCTION_AVX512

#ifdef TENZOR_REDUCTION_AVX2

// Horizontal sum of 8 floats in AVX register -> single float
__attribute__((target("avx2")))
static inline float hsum_avx2(__m256 v) {
    // Reduce 256 bits -> 128 bits
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);

    // Horizontal add within 128 bits: [a,b,c,d] -> [a+b, c+d, a+b, c+d]
    sum128 = _mm_hadd_ps(sum128, sum128);
    // -> [a+b+c+d, a+b+c+d, ...]
    sum128 = _mm_hadd_ps(sum128, sum128);

    return _mm_cvtss_f32(sum128);
}

// Horizontal max of 8 floats in AVX register -> single float
__attribute__((target("avx2")))
static inline float hmax_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 max128 = _mm_max_ps(lo, hi);

    // Shuffle and max to reduce 4 floats to 1
    __m128 shuf = _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1));
    max128 = _mm_max_ps(max128, shuf);
    shuf = _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2));
    max128 = _mm_max_ps(max128, shuf);

    return _mm_cvtss_f32(max128);
}

// Horizontal min of 8 floats in AVX register -> single float
__attribute__((target("avx2")))
static inline float hmin_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 min128 = _mm_min_ps(lo, hi);

    __m128 shuf = _mm_shuffle_ps(min128, min128, _MM_SHUFFLE(2, 3, 0, 1));
    min128 = _mm_min_ps(min128, shuf);
    shuf = _mm_shuffle_ps(min128, min128, _MM_SHUFFLE(1, 0, 3, 2));
    min128 = _mm_min_ps(min128, shuf);

    return _mm_cvtss_f32(min128);
}

// AVX2 vectorized sum for float32 with Kahan compensation
__attribute__((target("avx2")))
static float simd_sum_f32_avx2(const float* data, int64_t n) {
    constexpr int64_t VEC_SIZE = 8;
    const int64_t vec_end = (n / VEC_SIZE) * VEC_SIZE;

    __m256 vsum = _mm256_setzero_ps();
    __m256 vcomp = _mm256_setzero_ps();  // Kahan compensation

    // Main vectorized loop with Kahan summation
    for (int64_t i = 0; i < vec_end; i += VEC_SIZE) {
        __m256 v = _mm256_loadu_ps(data + i);
        __m256 y = _mm256_sub_ps(v, vcomp);
        __m256 t = _mm256_add_ps(vsum, y);
        vcomp = _mm256_sub_ps(_mm256_sub_ps(t, vsum), y);
        vsum = t;
    }

    // Horizontal sum of vector accumulator
    float sum = hsum_avx2(vsum);
    float comp = hsum_avx2(vcomp);

    // Handle remaining elements with scalar Kahan
    for (int64_t i = vec_end; i < n; i++) {
        float y = data[i] - comp;
        float t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }

    return sum;
}

// AVX2 vectorized max for float32
__attribute__((target("avx2")))
static float simd_max_f32_avx2(const float* data, int64_t n) {
    if (n == 0) return -std::numeric_limits<float>::infinity();

    constexpr int64_t VEC_SIZE = 8;
    const int64_t vec_end = (n / VEC_SIZE) * VEC_SIZE;

    __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

    for (int64_t i = 0; i < vec_end; i += VEC_SIZE) {
        __m256 v = _mm256_loadu_ps(data + i);
        vmax = _mm256_max_ps(vmax, v);
    }

    float max_val = hmax_avx2(vmax);

    for (int64_t i = vec_end; i < n; i++) {
        if (std::isnan(data[i]) || data[i] > max_val) max_val = data[i];
    }

    return max_val;
}

// AVX2 vectorized min for float32
__attribute__((target("avx2")))
static float simd_min_f32_avx2(const float* data, int64_t n) {
    if (n == 0) return std::numeric_limits<float>::infinity();

    constexpr int64_t VEC_SIZE = 8;
    const int64_t vec_end = (n / VEC_SIZE) * VEC_SIZE;

    __m256 vmin = _mm256_set1_ps(std::numeric_limits<float>::infinity());

    for (int64_t i = 0; i < vec_end; i += VEC_SIZE) {
        __m256 v = _mm256_loadu_ps(data + i);
        vmin = _mm256_min_ps(vmin, v);
    }

    float min_val = hmin_avx2(vmin);

    for (int64_t i = vec_end; i < n; i++) {
        if (std::isnan(data[i]) || data[i] < min_val) min_val = data[i];
    }

    return min_val;
}

#endif // TENZOR_REDUCTION_AVX2

// ============================================================================
// Float64 SIMD Reductions
// ============================================================================

#ifdef TENZOR_REDUCTION_AVX512

// Horizontal sum of 8 doubles in AVX-512 register
__attribute__((target("avx512f")))
static inline double hsum_avx512_f64(__m512d v) {
    __m256d lo = _mm512_castpd512_pd256(v);
    __m256d hi = _mm512_extractf64x4_pd(v, 1);
    __m256d sum256 = _mm256_add_pd(lo, hi);
    __m128d lo128 = _mm256_castpd256_pd128(sum256);
    __m128d hi128 = _mm256_extractf128_pd(sum256, 1);
    __m128d sum128 = _mm_add_pd(lo128, hi128);
    sum128 = _mm_hadd_pd(sum128, sum128);
    return _mm_cvtsd_f64(sum128);
}

__attribute__((target("avx512f")))
static double simd_sum_f64_avx512(const double* data, int64_t n) {
    constexpr int64_t VEC_SIZE = 8;
    const int64_t vec_end = (n / VEC_SIZE) * VEC_SIZE;
    __m512d vsum = _mm512_setzero_pd();
    __m512d vcomp = _mm512_setzero_pd();  // Kahan compensation
    for (int64_t i = 0; i < vec_end; i += VEC_SIZE) {
        __m512d v = _mm512_loadu_pd(data + i);
        __m512d y = _mm512_sub_pd(v, vcomp);
        __m512d t = _mm512_add_pd(vsum, y);
        vcomp = _mm512_sub_pd(_mm512_sub_pd(t, vsum), y);
        vsum = t;
    }
    double sum = hsum_avx512_f64(vsum);
    double comp = hsum_avx512_f64(vcomp);
    for (int64_t i = vec_end; i < n; i++) {
        double y = data[i] - comp;
        double t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    return sum;
}

__attribute__((target("avx512f")))
static double simd_max_f64_avx512(const double* data, int64_t n) {
    if (n == 0) return -std::numeric_limits<double>::infinity();
    constexpr int64_t VEC_SIZE = 8;
    const int64_t vec_end = (n / VEC_SIZE) * VEC_SIZE;
    __m512d vmax = _mm512_set1_pd(-std::numeric_limits<double>::infinity());
    for (int64_t i = 0; i < vec_end; i += VEC_SIZE) {
        __m512d v = _mm512_loadu_pd(data + i);
        vmax = _mm512_max_pd(vmax, v);
    }
    double max_val = _mm512_reduce_max_pd(vmax);
    for (int64_t i = vec_end; i < n; i++) {
        if (std::isnan(data[i]) || data[i] > max_val) max_val = data[i];
    }
    return max_val;
}

__attribute__((target("avx512f")))
static double simd_min_f64_avx512(const double* data, int64_t n) {
    if (n == 0) return std::numeric_limits<double>::infinity();
    constexpr int64_t VEC_SIZE = 8;
    const int64_t vec_end = (n / VEC_SIZE) * VEC_SIZE;
    __m512d vmin = _mm512_set1_pd(std::numeric_limits<double>::infinity());
    for (int64_t i = 0; i < vec_end; i += VEC_SIZE) {
        __m512d v = _mm512_loadu_pd(data + i);
        vmin = _mm512_min_pd(vmin, v);
    }
    double min_val = _mm512_reduce_min_pd(vmin);
    for (int64_t i = vec_end; i < n; i++) {
        if (std::isnan(data[i]) || data[i] < min_val) min_val = data[i];
    }
    return min_val;
}

#endif // TENZOR_REDUCTION_AVX512 (Float64)

#ifdef TENZOR_REDUCTION_AVX2

// Horizontal sum of 4 doubles in AVX register
__attribute__((target("avx2")))
static inline double hsum_avx2_f64(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d sum128 = _mm_add_pd(lo, hi);
    sum128 = _mm_hadd_pd(sum128, sum128);
    return _mm_cvtsd_f64(sum128);
}

// Horizontal max of 4 doubles in AVX register
__attribute__((target("avx2")))
static inline double hmax_avx2_f64(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d max128 = _mm_max_pd(lo, hi);
    __m128d shuf = _mm_shuffle_pd(max128, max128, 1);
    max128 = _mm_max_pd(max128, shuf);
    return _mm_cvtsd_f64(max128);
}

// Horizontal min of 4 doubles in AVX register
__attribute__((target("avx2")))
static inline double hmin_avx2_f64(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d min128 = _mm_min_pd(lo, hi);
    __m128d shuf = _mm_shuffle_pd(min128, min128, 1);
    min128 = _mm_min_pd(min128, shuf);
    return _mm_cvtsd_f64(min128);
}

__attribute__((target("avx2")))
static double simd_sum_f64_avx2(const double* data, int64_t n) {
    constexpr int64_t VEC_SIZE = 4;
    const int64_t vec_end = (n / VEC_SIZE) * VEC_SIZE;
    __m256d vsum = _mm256_setzero_pd();
    __m256d vcomp = _mm256_setzero_pd();  // Kahan compensation
    for (int64_t i = 0; i < vec_end; i += VEC_SIZE) {
        __m256d v = _mm256_loadu_pd(data + i);
        __m256d y = _mm256_sub_pd(v, vcomp);
        __m256d t = _mm256_add_pd(vsum, y);
        vcomp = _mm256_sub_pd(_mm256_sub_pd(t, vsum), y);
        vsum = t;
    }
    double sum = hsum_avx2_f64(vsum);
    double comp = hsum_avx2_f64(vcomp);
    for (int64_t i = vec_end; i < n; i++) {
        double y = data[i] - comp;
        double t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    return sum;
}

__attribute__((target("avx2")))
static double simd_max_f64_avx2(const double* data, int64_t n) {
    if (n == 0) return -std::numeric_limits<double>::infinity();
    constexpr int64_t VEC_SIZE = 4;
    const int64_t vec_end = (n / VEC_SIZE) * VEC_SIZE;
    __m256d vmax = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
    for (int64_t i = 0; i < vec_end; i += VEC_SIZE) {
        __m256d v = _mm256_loadu_pd(data + i);
        vmax = _mm256_max_pd(vmax, v);
    }
    double max_val = hmax_avx2_f64(vmax);
    for (int64_t i = vec_end; i < n; i++) {
        if (std::isnan(data[i]) || data[i] > max_val) max_val = data[i];
    }
    return max_val;
}

__attribute__((target("avx2")))
static double simd_min_f64_avx2(const double* data, int64_t n) {
    if (n == 0) return std::numeric_limits<double>::infinity();
    constexpr int64_t VEC_SIZE = 4;
    const int64_t vec_end = (n / VEC_SIZE) * VEC_SIZE;
    __m256d vmin = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    for (int64_t i = 0; i < vec_end; i += VEC_SIZE) {
        __m256d v = _mm256_loadu_pd(data + i);
        vmin = _mm256_min_pd(vmin, v);
    }
    double min_val = hmin_avx2_f64(vmin);
    for (int64_t i = vec_end; i < n; i++) {
        if (std::isnan(data[i]) || data[i] < min_val) min_val = data[i];
    }
    return min_val;
}

#endif // TENZOR_REDUCTION_AVX2 (Float64)

// Parallel SIMD sum for Float64
static double parallel_simd_sum_f64(const double* data, int64_t n) {
    if (n == 0) return 0.0;
    // Deterministic mode: force the single-threaded path for bit-identical
    // results regardless of OMP_NUM_THREADS. Parallel floating-point reductions
    // are non-associative; different thread partitionings produce different
    // rounding (matches parallel_simd_sum_f32).
    if (n < REDUCTION_OMP_THRESHOLD || ::tenzor::is_deterministic()) {
#ifdef TENZOR_REDUCTION_AVX512
        return simd_sum_f64_avx512(data, n);
#elif defined(TENZOR_REDUCTION_AVX2)
        return simd_sum_f64_avx2(data, n);
#else
        double sum = 0.0;
        for (int64_t i = 0; i < n; i++) sum += data[i];
        return sum;
#endif
    }
    double total_sum = 0.0;
    #pragma omp parallel reduction(+:total_sum)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int64_t chunk_size = (n + nthreads - 1) / nthreads;
        int64_t start = tid * chunk_size;
        int64_t end = std::min(start + chunk_size, n);
        if (start < end) {
#ifdef TENZOR_REDUCTION_AVX512
            total_sum = simd_sum_f64_avx512(data + start, end - start);
#elif defined(TENZOR_REDUCTION_AVX2)
            total_sum = simd_sum_f64_avx2(data + start, end - start);
#else
            for (int64_t i = start; i < end; i++) total_sum += data[i];
#endif
        }
    }
    return total_sum;
}

// Parallel SIMD max for Float64.
// NaN propagation: if any element is NaN, returns NaN (PyTorch/NumPy semantics).
// The SIMD/scalar max ignores NaN (unordered compares), so we fold an any_nan
// scan into the reduction — mirroring parallel_simd_max_f32.
static double parallel_simd_max_f64(const double* data, int64_t n) {
    if (n == 0) return -std::numeric_limits<double>::infinity();
    if (n < REDUCTION_OMP_THRESHOLD) {
        double result;
#ifdef TENZOR_REDUCTION_AVX512
        result = simd_max_f64_avx512(data, n);
#elif defined(TENZOR_REDUCTION_AVX2)
        result = simd_max_f64_avx2(data, n);
#else
        result = data[0];
        for (int64_t i = 1; i < n; i++) if (data[i] > result) result = data[i];
#endif
        if (std::isnan(result)) return std::numeric_limits<double>::quiet_NaN();
        for (int64_t i = 0; i < n; ++i) {
            if (std::isnan(data[i])) return std::numeric_limits<double>::quiet_NaN();
        }
        return result;
    }
    bool any_nan = false;
    double global_max = -std::numeric_limits<double>::infinity();
    #pragma omp parallel reduction(max:global_max) reduction(||:any_nan)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int64_t chunk_size = (n + nthreads - 1) / nthreads;
        int64_t start = tid * chunk_size;
        int64_t end = std::min(start + chunk_size, n);
        if (start < end) {
#ifdef TENZOR_REDUCTION_AVX512
            global_max = simd_max_f64_avx512(data + start, end - start);
#elif defined(TENZOR_REDUCTION_AVX2)
            global_max = simd_max_f64_avx2(data + start, end - start);
#else
            global_max = data[start];
            for (int64_t i = start + 1; i < end; i++) if (data[i] > global_max) global_max = data[i];
#endif
            for (int64_t i = start; i < end; i++) {
                if (std::isnan(data[i])) { any_nan = true; break; }
            }
        }
    }
    if (any_nan || std::isnan(global_max)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return global_max;
}

// Parallel SIMD min for Float64.
// NaN propagation: if any element is NaN, returns NaN (PyTorch/NumPy semantics).
// See parallel_simd_max_f64 for the NaN-handling rationale.
static double parallel_simd_min_f64(const double* data, int64_t n) {
    if (n == 0) return std::numeric_limits<double>::infinity();
    if (n < REDUCTION_OMP_THRESHOLD) {
        double result;
#ifdef TENZOR_REDUCTION_AVX512
        result = simd_min_f64_avx512(data, n);
#elif defined(TENZOR_REDUCTION_AVX2)
        result = simd_min_f64_avx2(data, n);
#else
        result = data[0];
        for (int64_t i = 1; i < n; i++) if (data[i] < result) result = data[i];
#endif
        if (std::isnan(result)) return std::numeric_limits<double>::quiet_NaN();
        for (int64_t i = 0; i < n; ++i) {
            if (std::isnan(data[i])) return std::numeric_limits<double>::quiet_NaN();
        }
        return result;
    }
    bool any_nan = false;
    double global_min = std::numeric_limits<double>::infinity();
    #pragma omp parallel reduction(min:global_min) reduction(||:any_nan)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int64_t chunk_size = (n + nthreads - 1) / nthreads;
        int64_t start = tid * chunk_size;
        int64_t end = std::min(start + chunk_size, n);
        if (start < end) {
#ifdef TENZOR_REDUCTION_AVX512
            global_min = simd_min_f64_avx512(data + start, end - start);
#elif defined(TENZOR_REDUCTION_AVX2)
            global_min = simd_min_f64_avx2(data + start, end - start);
#else
            global_min = data[start];
            for (int64_t i = start + 1; i < end; i++) if (data[i] < global_min) global_min = data[i];
#endif
            for (int64_t i = start; i < end; i++) {
                if (std::isnan(data[i])) { any_nan = true; break; }
            }
        }
    }
    if (any_nan || std::isnan(global_min)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return global_min;
}

// ============================================================================
// Parallel SIMD Reduction - combines OpenMP with SIMD for best performance
// ============================================================================

// Parallel SIMD sum for float32 - uses thread-local SIMD accumulators
static float parallel_simd_sum_f32(const float* data, int64_t n) {
    if (n == 0) return 0.0f;

    // Deterministic mode: force single-threaded path for bit-identical results
    // regardless of OMP_NUM_THREADS. Parallel floating-point reductions are
    // non-associative; different thread partitionings produce different rounding.
    if (n < REDUCTION_OMP_THRESHOLD || ::tenzor::is_deterministic()) {
#ifdef TENZOR_REDUCTION_AVX512
        return simd_sum_f32_avx512(data, n);
#elif defined(TENZOR_REDUCTION_AVX2)
        return simd_sum_f32_avx2(data, n);
#else
        float sum = 0.0f;
        for (int64_t i = 0; i < n; i++) sum += data[i];
        return sum;
#endif
    }

    // For large arrays, use parallel reduction with per-thread partial sums.
    // Each thread uses Kahan-compensated SIMD summation on its chunk.
    // We collect partial sums into an array and combine with Kahan summation
    // to preserve compensation across threads (plain OMP reduction(+:) would
    // lose the Kahan compensation at the inter-thread combination step).
    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif
    // Pad each slot to a full cache line (64 bytes) to prevent false sharing:
    // without padding 16 adjacent float slots share one 64-byte cache line and
    // every store from one thread invalidates every other thread's line.
    struct alignas(64) PaddedFloat { float v = 0.0f; char _pad[60]; };
    std::vector<PaddedFloat> partial_sums(max_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        // Compute chunk for this thread
        int64_t chunk_size = (n + nthreads - 1) / nthreads;
        int64_t start = tid * chunk_size;
        int64_t end = std::min(start + chunk_size, n);

        if (start < end) {
#ifdef TENZOR_REDUCTION_AVX512
            partial_sums[tid].v = simd_sum_f32_avx512(data + start, end - start);
#elif defined(TENZOR_REDUCTION_AVX2)
            partial_sums[tid].v = simd_sum_f32_avx2(data + start, end - start);
#else
            float local_sum = 0.0f;
            float local_comp = 0.0f;
            for (int64_t i = start; i < end; i++) {
                float y = data[i] - local_comp;
                float t = local_sum + y;
                local_comp = (t - local_sum) - y;
                local_sum = t;
            }
            partial_sums[tid].v = local_sum;
#endif
        }
    }

    // Combine partial sums with Kahan summation
    float total_sum = 0.0f;
    float comp = 0.0f;
    for (int i = 0; i < max_threads; ++i) {
        float y = partial_sums[i].v - comp;
        float t = total_sum + y;
        comp = (t - total_sum) - y;
        total_sum = t;
    }

    return total_sum;
}

// Parallel SIMD max for float32.
// NaN propagation: if any element is NaN, returns NaN (PyTorch/NumPy semantics).
//
// The SIMD/scalar max ignores NaN (unordered compares), so we cannot rely on
// the reduction result alone to surface NaNs. Instead of paying a separate full
// scalar NaN pre-scan (an extra non-SIMD, non-parallel memory pass that
// dominated large-tensor cost), we fold a fast result check + targeted fallback:
// take the SIMD/parallel max first, and only when that result is NaN — or, for
// the common case where the max is finite, do a parallelised NaN check that
// short-circuits — return NaN. The NaN check shares the same parallel structure
// (and SIMD chunking via the max helpers) rather than a serial pre-pass.
static float parallel_simd_max_f32(const float* data, int64_t n) {
    if (n == 0) return -std::numeric_limits<float>::infinity();

    if (n < REDUCTION_OMP_THRESHOLD) {
        float result;
#ifdef TENZOR_REDUCTION_AVX512
        result = simd_max_f32_avx512(data, n);
#elif defined(TENZOR_REDUCTION_AVX2)
        result = simd_max_f32_avx2(data, n);
#else
        result = data[0];
        for (int64_t i = 1; i < n; i++) {
            if (data[i] > result) result = data[i];
        }
#endif
        if (std::isnan(result)) return std::numeric_limits<float>::quiet_NaN();
        for (int64_t i = 0; i < n; ++i) {
            if (std::isnan(data[i])) return std::numeric_limits<float>::quiet_NaN();
        }
        return result;
    }

    bool any_nan = false;
    float global_max = -std::numeric_limits<float>::infinity();

    #pragma omp parallel reduction(max:global_max) reduction(||:any_nan)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        int64_t chunk_size = (n + nthreads - 1) / nthreads;
        int64_t start = tid * chunk_size;
        int64_t end = std::min(start + chunk_size, n);

        if (start < end) {
#ifdef TENZOR_REDUCTION_AVX512
            global_max = simd_max_f32_avx512(data + start, end - start);
#elif defined(TENZOR_REDUCTION_AVX2)
            global_max = simd_max_f32_avx2(data + start, end - start);
#else
            global_max = data[start];
            for (int64_t i = start + 1; i < end; i++) {
                if (data[i] > global_max) global_max = data[i];
            }
#endif
            // NaN detection folded into the same parallel pass (no separate
            // serial scan). Each thread checks only its own chunk.
            for (int64_t i = start; i < end; i++) {
                if (std::isnan(data[i])) { any_nan = true; break; }
            }
        }
    }

    if (any_nan || std::isnan(global_max)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return global_max;
}

// Parallel SIMD min for float32.
// NaN propagation: if any element is NaN, returns NaN (PyTorch/NumPy semantics).
// See parallel_simd_max_f32 for the NaN-handling rationale.
static float parallel_simd_min_f32(const float* data, int64_t n) {
    if (n == 0) return std::numeric_limits<float>::infinity();

    if (n < REDUCTION_OMP_THRESHOLD) {
        float result;
#ifdef TENZOR_REDUCTION_AVX512
        result = simd_min_f32_avx512(data, n);
#elif defined(TENZOR_REDUCTION_AVX2)
        result = simd_min_f32_avx2(data, n);
#else
        result = data[0];
        for (int64_t i = 1; i < n; i++) {
            if (data[i] < result) result = data[i];
        }
#endif
        if (std::isnan(result)) return std::numeric_limits<float>::quiet_NaN();
        for (int64_t i = 0; i < n; ++i) {
            if (std::isnan(data[i])) return std::numeric_limits<float>::quiet_NaN();
        }
        return result;
    }

    bool any_nan = false;
    float global_min = std::numeric_limits<float>::infinity();

    #pragma omp parallel reduction(min:global_min) reduction(||:any_nan)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        int64_t chunk_size = (n + nthreads - 1) / nthreads;
        int64_t start = tid * chunk_size;
        int64_t end = std::min(start + chunk_size, n);

        if (start < end) {
#ifdef TENZOR_REDUCTION_AVX512
            global_min = simd_min_f32_avx512(data + start, end - start);
#elif defined(TENZOR_REDUCTION_AVX2)
            global_min = simd_min_f32_avx2(data + start, end - start);
#else
            global_min = data[start];
            for (int64_t i = start + 1; i < end; i++) {
                if (data[i] < global_min) global_min = data[i];
            }
#endif
            for (int64_t i = start; i < end; i++) {
                if (std::isnan(data[i])) { any_nan = true; break; }
            }
        }
    }

    if (any_nan || std::isnan(global_min)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return global_min;
}

// Sentinel value for full reduction across all dimensions
static constexpr int64_t REDUCE_ALL = INT64_MIN;

// Helper to normalize negative dimension index
static auto normalize_dim(int64_t dim, int64_t ndim) -> int64_t {
    if (dim == REDUCE_ALL) {
        return REDUCE_ALL;
    }
    if (dim < 0) {
        dim += ndim;
    }
    // Validate dimension is within bounds
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Dimension " + std::to_string(dim) +
            " out of range for tensor with " + std::to_string(ndim) + " dimensions");
    }
    return dim;
}

// Helper to compute output shape for reduction
static auto compute_reduction_shape(const std::vector<int64_t>& input_shape,
                                    int64_t dim,
                                    bool keepdim) -> std::vector<int64_t> {
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize negative dimension
    dim = normalize_dim(dim, ndim);

    if (dim == REDUCE_ALL) {
        // Full reduction - return scalar or [1,1,...] if keepdim
        if (keepdim) {
            return std::vector<int64_t>(input_shape.size(), 1);
        }
        return {};  // Scalar (0D tensor)
    }

    std::vector<int64_t> output_shape = input_shape;
    if (keepdim) {
        output_shape[dim] = 1;
    } else {
        output_shape.erase(output_shape.begin() + dim);
        // Keep empty shape for scalar result
    }
    return output_shape;
}

// Template for sum reduction - uses SIMD + OpenMP for maximum performance
// Specialization for float uses parallel SIMD reductions
template<typename T>
auto sum_impl(const T* input_data, int64_t n) -> T {
    if (n == 0) return T(0);

    // Accumulate integer reductions in a 64-bit accumulator. With a narrow
    // accumulator (int8/int16/int32) the running sum overflows after as few as
    // 128 elements, and for *signed* integers that overflow is undefined
    // behavior (not merely a wrapped value the compiler may keep). Widening to
    // int64_t/uint64_t makes the accumulation well-defined; the final cast back
    // to T preserves the documented kernel contract (output dtype == input
    // dtype — the public `tenzor::sum()` op promotes narrow ints to Int64
    // before dispatch, so any narrowing here only affects direct kernel calls).
    using Acc = std::conditional_t<
        std::is_integral_v<T>,
        std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>,
        T>;
    Acc sum = Acc(0);

    #pragma omp parallel for reduction(+:sum) if(n > REDUCTION_OMP_THRESHOLD)
    for (int64_t i = 0; i < n; i++) {
        sum += static_cast<Acc>(input_data[i]);
    }

    return static_cast<T>(sum);
}

// Specialization for float - uses SIMD vectorized reduction
template<>
auto sum_impl<float>(const float* input_data, int64_t n) -> float {
    return parallel_simd_sum_f32(input_data, n);
}

// Specialization for double - uses SIMD vectorized reduction
template<>
auto sum_impl<double>(const double* input_data, int64_t n) -> double {
    return parallel_simd_sum_f64(input_data, n);
}

// Complex specializations avoid OpenMP user-defined reduction — OpenMP's
// `reduction(+:sum)` clause is not guaranteed to work on std::complex<T>
// across compiler versions, and the sum path from gradcheck only exercises
// modest sizes (FFT outputs of a few K elements) so the single-thread sum
// has no meaningful perf cost here.
template<>
auto sum_impl<std::complex<float>>(const std::complex<float>* input_data,
                                   int64_t n) -> std::complex<float> {
    std::complex<float> sum{0.0f, 0.0f};
    for (int64_t i = 0; i < n; ++i) sum += input_data[i];
    return sum;
}

template<>
auto sum_impl<std::complex<double>>(const std::complex<double>* input_data,
                                    int64_t n) -> std::complex<double> {
    std::complex<double> sum{0.0, 0.0};
    for (int64_t i = 0; i < n; ++i) sum += input_data[i];
    return sum;
}

// Sum along a specific dimension
template<typename T>
void sum_along_dim(const T* input_data,
                   T* output_data,
                   const std::vector<int64_t>& input_shape,
                   const std::vector<int64_t>& input_strides,
                   int64_t dim) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    // Compute output size
    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    // Integer reductions accumulate in a 64-bit accumulator to avoid signed
    // overflow UB / narrow-type wraparound; floating-point uses Kahan
    // compensation in T. (See sum_impl for the rationale and the output-dtype
    // contract.)
    using Acc = std::conditional_t<
        std::is_integral_v<T>,
        std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>,
        T>;
    constexpr bool kUseKahan = std::is_floating_point_v<T>;

    // Reduction along dimension - each output element is independent
    // Use higher threshold since inner loop work is proportional to dim_size
    const int64_t total_work = output_size * dim_size;
    #pragma omp parallel for if(total_work > REDUCTION_OMP_THRESHOLD)
    for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
        // Use stack allocation for common case (<=16 dims), heap for rare large dims
        std::vector<int64_t> indices_vec;
        int64_t indices_stack[16] = {};
        int64_t* indices = (ndim <= 16) ? indices_stack : (indices_vec.resize(ndim, 0), indices_vec.data());
        int64_t tmp = out_idx;

        for (int64_t d = ndim - 1; d >= 0; --d) {
            if (d == dim) continue;
            indices[d] = tmp % input_shape[d];
            tmp /= input_shape[d];
        }

        Acc sum = Acc(0);
        Acc compensation = Acc(0);
        for (int64_t i = 0; i < dim_size; i++) {
            indices[dim] = i;
            int64_t in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * input_strides[d];
            }
            if constexpr (kUseKahan) {
                // Kahan compensated sum for improved precision on Float32/Float64.
                Acc y = static_cast<Acc>(input_data[in_idx]) - compensation;
                Acc t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            } else {
                // Integer / complex: plain accumulation in the (possibly widened)
                // accumulator. Kahan is meaningless for these types and its
                // compensation arithmetic would itself overflow narrow ints.
                sum += static_cast<Acc>(input_data[in_idx]);
            }
        }
        output_data[out_idx] = static_cast<T>(sum);
    }
}

auto sum_kernel(const Tensor& input_raw, int64_t dim, bool keepdim) -> Tensor {
    // audit-2026-05-03 bug #2 root cause: this kernel reads input.data<T>()[i]
    // assuming contiguous layout. For a non-contiguous view (e.g. slice with
    // non-zero offset, expand with stride 0), that flat-pointer iteration
    // skips logical elements and reads off the end of the view's footprint.
    // Symptom: sum(slice(x, dim=last, start=2, end=6)) on shape {2,8} returns
    // 44 instead of 60 — it sums positions [2..9] of the underlying storage
    // instead of [2..5, 10..13]. Materializing once at the entry point fixes
    // every dtype/dim path inside this kernel without per-branch changes.
    auto input = input_raw.contiguous();

    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize negative dimension
    dim = normalize_dim(dim, ndim);

    // Compute output shape
    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    // Dispatch based on dtype
    switch (dtype) {
        case DType::Float16: {
            auto* input_data = input.data<Float16>();
            auto* output_data = output.data<Float16>();

            if (dim == REDUCE_ALL) {
                // Full reduction - Kahan compensated summation in Float32
                const int64_t n = input.numel();
                float sum = 0.0f;
                float compensation = 0.0f;

                for (int64_t i = 0; i < n; i++) {
                    float y = static_cast<float>(input_data[i]) - compensation;
                    float t = sum + y;
                    compensation = (t - sum) - y;
                    sum = t;
                }
                output_data[0] = Float16(sum);
            } else {
                // Dimensional reduction - Kahan compensated in Float32
                const int64_t shape_ndim = static_cast<int64_t>(input_shape.size());
                const int64_t dim_size = input_shape[dim];

                int64_t output_size = 1;
                for (int64_t i = 0; i < shape_ndim; i++) {
                    if (i != dim) {
                        output_size *= input_shape[i];
                    }
                }

                const int64_t total_work = output_size * dim_size;
                #pragma omp parallel for if(total_work > REDUCTION_OMP_THRESHOLD)
                for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
                    std::vector<int64_t> indices(shape_ndim, 0);
                    int64_t tmp = out_idx;

                    for (int64_t d = shape_ndim - 1; d >= 0; --d) {
                        if (d == dim) continue;
                        indices[d] = tmp % input_shape[d];
                        tmp /= input_shape[d];
                    }

                    // Kahan compensated sum along dimension
                    float sum = 0.0f;
                    float compensation = 0.0f;
                    for (int64_t i = 0; i < dim_size; i++) {
                        indices[dim] = i;
                        int64_t in_idx = 0;
                        for (int64_t d = 0; d < shape_ndim; d++) {
                            in_idx += indices[d] * input_strides[d];
                        }
                        float y = static_cast<float>(input_data[in_idx]) - compensation;
                        float t = sum + y;
                        compensation = (t - sum) - y;
                        sum = t;
                    }
                    output_data[out_idx] = Float16(sum);
                }
            }
            break;
        }
        case DType::BFloat16: {
            auto* input_data = input.data<BFloat16>();
            auto* output_data = output.data<BFloat16>();

            if (dim == REDUCE_ALL) {
                // Full reduction - Kahan compensated summation in Float32
                const int64_t n = input.numel();
                float sum = 0.0f;
                float compensation = 0.0f;

                for (int64_t i = 0; i < n; i++) {
                    float y = static_cast<float>(input_data[i]) - compensation;
                    float t = sum + y;
                    compensation = (t - sum) - y;
                    sum = t;
                }
                output_data[0] = BFloat16(sum);
            } else {
                // Dimensional reduction - Kahan compensated in Float32
                const int64_t shape_ndim = static_cast<int64_t>(input_shape.size());
                const int64_t dim_size = input_shape[dim];

                int64_t output_size = 1;
                for (int64_t i = 0; i < shape_ndim; i++) {
                    if (i != dim) {
                        output_size *= input_shape[i];
                    }
                }

                const int64_t total_work = output_size * dim_size;
                #pragma omp parallel for if(total_work > REDUCTION_OMP_THRESHOLD)
                for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
                    std::vector<int64_t> indices(shape_ndim, 0);
                    int64_t tmp = out_idx;

                    for (int64_t d = shape_ndim - 1; d >= 0; --d) {
                        if (d == dim) continue;
                        indices[d] = tmp % input_shape[d];
                        tmp /= input_shape[d];
                    }

                    // Kahan compensated sum along dimension
                    float sum = 0.0f;
                    float compensation = 0.0f;
                    for (int64_t i = 0; i < dim_size; i++) {
                        indices[dim] = i;
                        int64_t in_idx = 0;
                        for (int64_t d = 0; d < shape_ndim; d++) {
                            in_idx += indices[d] * input_strides[d];
                        }
                        float y = static_cast<float>(input_data[in_idx]) - compensation;
                        float t = sum + y;
                        compensation = (t - sum) - y;
                        sum = t;
                    }
                    output_data[out_idx] = BFloat16(sum);
                }
            }
            break;
        }
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            if (dim == REDUCE_ALL) {
                // Full reduction
                output_data[0] = sum_impl(input_data, input.numel());
            } else {
                sum_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (dim == REDUCE_ALL) {
                output_data[0] = sum_impl(input_data, input.numel());
            } else {
                sum_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int32_t>();

            if (dim == REDUCE_ALL) {
                output_data[0] = sum_impl(input_data, input.numel());
            } else {
                sum_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (dim == REDUCE_ALL) {
                output_data[0] = sum_impl(input_data, input.numel());
            } else {
                sum_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Complex64: {
            auto* input_data = input.data<std::complex<float>>();
            auto* output_data = output.data<std::complex<float>>();
            if (dim == REDUCE_ALL) {
                output_data[0] = sum_impl(input_data, input.numel());
            } else {
                sum_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Complex128: {
            auto* input_data = input.data<std::complex<double>>();
            auto* output_data = output.data<std::complex<double>>();
            if (dim == REDUCE_ALL) {
                output_data[0] = sum_impl(input_data, input.numel());
            } else {
                sum_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        // S14: small integer types are routed through the generic sum_impl<T>
        // and sum_along_dim<T> templates. The public `tenzor::sum()` op
        // promotes Int8/UInt8/Int16/UInt16/Int32/UInt32/Bool to Int64
        // *before* dispatching here, so these branches only fire on
        // direct kernel calls / backend-parity probes. Output dtype
        // matches input dtype for kernel-level consistency with the
        // existing Int32 / Int64 contract; overflow safety is the
        // responsibility of the public-op promotion layer.
        case DType::Int8: {
            auto* input_data = input.data<int8_t>();
            auto* output_data = output.data<int8_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = sum_impl(input_data, input.numel());
            } else {
                sum_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::UInt8: {
            auto* input_data = input.data<uint8_t>();
            auto* output_data = output.data<uint8_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = sum_impl(input_data, input.numel());
            } else {
                sum_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Int16: {
            auto* input_data = input.data<int16_t>();
            auto* output_data = output.data<int16_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = sum_impl(input_data, input.numel());
            } else {
                sum_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::UInt16: {
            auto* input_data = input.data<uint16_t>();
            auto* output_data = output.data<uint16_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = sum_impl(input_data, input.numel());
            } else {
                sum_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::UInt32: {
            auto* input_data = input.data<uint32_t>();
            auto* output_data = output.data<uint32_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = sum_impl(input_data, input.numel());
            } else {
                sum_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::UInt64: {
            auto* input_data = input.data<uint64_t>();
            auto* output_data = output.data<uint64_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = sum_impl(input_data, input.numel());
            } else {
                sum_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Bool: {
            // For Bool input, `bool` accumulators are semantically broken
            // (`bool sum=0; sum += bool[i]` collapses to logical-or).
            // Use a wider int64_t accumulator internally but write back
            // as bool (output = any non-zero contribution), preserving
            // the input-dtype-equals-output-dtype contract. The public
            // `tenzor::sum()` op promotes Bool→Int64 before dispatch,
            // so kernel-level Bool sum is only reached via direct calls
            // and yields a numerically-defensible "any" semantics.
            auto* input_data = input.data<bool>();
            auto* output_data = output.data<bool>();
            const int64_t shape_ndim = static_cast<int64_t>(input_shape.size());
            if (dim == REDUCE_ALL) {
                int64_t count = 0;
                const int64_t n = input.numel();
                for (int64_t i = 0; i < n; ++i) {
                    count += input_data[i] ? 1 : 0;
                }
                output_data[0] = (count != 0);
            } else {
                const int64_t dim_size = input_shape[dim];
                int64_t output_size = 1;
                for (int64_t i = 0; i < shape_ndim; i++) {
                    if (i != dim) output_size *= input_shape[i];
                }
                const int64_t total_work = output_size * dim_size;
                #pragma omp parallel for if(total_work > REDUCTION_OMP_THRESHOLD)
                for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
                    std::vector<int64_t> indices(shape_ndim, 0);
                    int64_t tmp = out_idx;
                    for (int64_t d = shape_ndim - 1; d >= 0; --d) {
                        if (d == dim) continue;
                        indices[d] = tmp % input_shape[d];
                        tmp /= input_shape[d];
                    }
                    int64_t count = 0;
                    for (int64_t i = 0; i < dim_size; i++) {
                        indices[dim] = i;
                        int64_t in_idx = 0;
                        for (int64_t d = 0; d < shape_ndim; d++) {
                            in_idx += indices[d] * input_strides[d];
                        }
                        count += input_data[in_idx] ? 1 : 0;
                    }
                    output_data[out_idx] = (count != 0);
                }
            }
            break;
        }
        default:
            throw std::runtime_error("sum: unsupported dtype");
    }

    return output;
}

auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    const auto dtype = input.dtype();
    const int64_t ndim = input.ndim();

    // Integer and Bool dtypes: widen to Float32, compute mean, return Float32
    // (PyTorch convention: integer mean returns Float32)
    if (dtype == DType::Int8 || dtype == DType::Int16 || dtype == DType::Int32 ||
        dtype == DType::Int64 || dtype == DType::UInt8 || dtype == DType::UInt16 ||
        dtype == DType::UInt32 || dtype == DType::UInt64 || dtype == DType::Bool) {
        auto f32_input = input.to(DType::Float32);
        return mean_kernel(f32_input, dim, keepdim);
    }

    // Float16/BFloat16: widen to Float32, compute the mean, then narrow the
    // RESULT back. Computing the half mean via sum_kernel narrows the Float32
    // accumulator to the half type BEFORE the division, so a sum exceeding the
    // half range becomes inf even when the mean itself is representable (F004).
    // Widening the whole computation keeps the intermediate sum in Float32.
    if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        auto f32_input = input.to(DType::Float32);
        auto f32_result = mean_kernel(f32_input, dim, keepdim);
        return f32_result.to(dtype);
    }

    // Complex dtypes: sum then divide by count in complex domain
    if (dtype == DType::Complex64 || dtype == DType::Complex128) {
        dim = normalize_dim(dim, ndim);
        auto sum_result = sum_kernel(input, dim, keepdim);
        int64_t count;
        if (dim == REDUCE_ALL) {
            count = input.numel();
        } else {
            count = input.shape()[dim];
        }
        const int64_t n = sum_result.numel();
        if (dtype == DType::Complex64) {
            auto* data = sum_result.data<std::complex<float>>();
            const float scale = 1.0f / static_cast<float>(count);
            for (int64_t i = 0; i < n; i++) {
                data[i] *= scale;
            }
        } else {
            auto* data = sum_result.data<std::complex<double>>();
            const double scale = 1.0 / static_cast<double>(count);
            for (int64_t i = 0; i < n; i++) {
                data[i] *= scale;
            }
        }
        return sum_result;
    }

    // Float-only path (Float16, BFloat16, Float32, Float64)
    if (dtype != DType::Float16 && dtype != DType::BFloat16 && dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("mean: unsupported dtype");
    }

    // Normalize negative dimension
    dim = normalize_dim(dim, ndim);

    // Compute sum first
    auto sum_result = sum_kernel(input, dim, keepdim);

    // Compute the count for averaging
    int64_t count;
    if (dim == REDUCE_ALL) {
        count = input.numel();
    } else {
        count = input.shape()[dim];
    }

    // Mean of zero elements is mathematically 0/0 -- produce NaN explicitly
    // and deterministically as a documented choice, rather than relying on
    // scale = 1/count overflowing to +inf and an (assumed-zero) sum_result
    // collapsing 0 * inf to NaN by accident. Matches CUDA's mean_kernel.
    if (count == 0) {
        const int64_t n = sum_result.numel();
        if (dtype == DType::Float16) {
            auto* data = sum_result.data<Float16>();
            for (int64_t i = 0; i < n; i++) data[i] = Float16(std::numeric_limits<float>::quiet_NaN());
        } else if (dtype == DType::BFloat16) {
            auto* data = sum_result.data<BFloat16>();
            for (int64_t i = 0; i < n; i++) data[i] = BFloat16(std::numeric_limits<float>::quiet_NaN());
        } else if (dtype == DType::Float32) {
            auto* data = sum_result.data<float>();
            for (int64_t i = 0; i < n; i++) data[i] = std::numeric_limits<float>::quiet_NaN();
        } else {  // Float64
            auto* data = sum_result.data<double>();
            for (int64_t i = 0; i < n; i++) data[i] = std::numeric_limits<double>::quiet_NaN();
        }
        return sum_result;
    }

    // Divide sum by count
    if (dtype == DType::Float16) {
        auto* data = sum_result.data<Float16>();
        const float scale = 1.0f / static_cast<float>(count);
        const int64_t n = sum_result.numel();

        #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
        for (int64_t i = 0; i < n; i++) {
            data[i] = Float16(static_cast<float>(data[i]) * scale);
        }
    } else if (dtype == DType::BFloat16) {
        auto* data = sum_result.data<BFloat16>();
        const float scale = 1.0f / static_cast<float>(count);
        const int64_t n = sum_result.numel();

        #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
        for (int64_t i = 0; i < n; i++) {
            data[i] = BFloat16(static_cast<float>(data[i]) * scale);
        }
    } else if (dtype == DType::Float32) {
        auto* data = sum_result.data<float>();
        const float scale = 1.0f / static_cast<float>(count);
        const int64_t n = sum_result.numel();

        #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
        for (int64_t i = 0; i < n; i++) {
            data[i] *= scale;
        }
    } else {  // Float64
        auto* data = sum_result.data<double>();
        const double scale = 1.0 / static_cast<double>(count);
        const int64_t n = sum_result.numel();

        #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
        for (int64_t i = 0; i < n; i++) {
            data[i] *= scale;
        }
    }

    return sum_result;
}

// Template for max reduction - uses OpenMP reduction(max:) for efficiency
template<typename T>
auto max_impl(const T* input_data, int64_t n) -> T {
    if (n == 0) throw std::runtime_error("max: input tensor is empty");

    T max_val = input_data[0];

    // OpenMP 3.1+ supports reduction(max:) for built-in types
    #pragma omp parallel for reduction(max:max_val) if(n > REDUCTION_OMP_THRESHOLD)
    for (int64_t i = 1; i < n; i++) {
        if (input_data[i] > max_val) {
            max_val = input_data[i];
        }
    }

    return max_val;
}

// Specialization for float - uses SIMD vectorized reduction
template<>
auto max_impl<float>(const float* input_data, int64_t n) -> float {
    if (n == 0) throw std::runtime_error("max: input tensor is empty");
    return parallel_simd_max_f32(input_data, n);
}

template<>
auto max_impl<double>(const double* input_data, int64_t n) -> double {
    if (n == 0) throw std::runtime_error("max: input tensor is empty");
    return parallel_simd_max_f64(input_data, n);
}

// Max along a specific dimension
template<typename T>
void max_along_dim(const T* input_data,
                   T* output_data,
                   const std::vector<int64_t>& input_shape,
                   const std::vector<int64_t>& input_strides,
                   int64_t dim) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    // Compute output size
    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    // Reduction along dimension
    const int64_t total_work = output_size * dim_size;
    #pragma omp parallel for if(total_work > REDUCTION_OMP_THRESHOLD)
    for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
        std::vector<int64_t> indices(ndim, 0);
        int64_t tmp = out_idx;

        for (int64_t d = ndim - 1; d >= 0; --d) {
            if (d == dim) continue;
            indices[d] = tmp % input_shape[d];
            tmp /= input_shape[d];
        }

        // Find max along the reduction dimension
        indices[dim] = 0;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }
        T max_val = input_data[in_idx];
        // For floating-point types propagate NaN to match the full-reduction
        // and Float16/BFloat16 dimensional paths; `x > max_val` is false for
        // any NaN so a bare comparison would silently drop NaN elements.
        if constexpr (std::is_floating_point_v<T>) {
            bool saw_nan = std::isnan(max_val);
            for (int64_t i = 1; i < dim_size && !saw_nan; i++) {
                indices[dim] = i;
                in_idx = 0;
                for (int64_t d = 0; d < ndim; d++) {
                    in_idx += indices[d] * input_strides[d];
                }
                const T v = input_data[in_idx];
                if (std::isnan(v)) { saw_nan = true; break; }
                if (v > max_val) {
                    max_val = v;
                }
            }
            output_data[out_idx] = saw_nan ? std::numeric_limits<T>::quiet_NaN() : max_val;
        } else {
            for (int64_t i = 1; i < dim_size; i++) {
                indices[dim] = i;
                in_idx = 0;
                for (int64_t d = 0; d < ndim; d++) {
                    in_idx += indices[d] * input_strides[d];
                }
                if (input_data[in_idx] > max_val) {
                    max_val = input_data[in_idx];
                }
            }
            output_data[out_idx] = max_val;
        }
    }
}

auto max_kernel(const Tensor& input_raw, int64_t dim, bool keepdim) -> Tensor {
    // Mirror sum_kernel: the REDUCE_ALL path reads input.data<T>()[i] assuming
    // contiguous layout, which reads off a non-contiguous view's footprint.
    // Materialize a contiguous copy once at entry so every dtype/dim path is safe.
    auto input = input_raw.contiguous();
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize negative dimension
    dim = normalize_dim(dim, ndim);

    // max has no identity element: a zero-size reduction is undefined. Guard
    // BOTH the full-reduction (numel==0) and per-dim (dim_size==0) empty cases
    // here at kernel entry — BEFORE any OpenMP region — because the downstream
    // paths seed an accumulator from element [0] and would read out of bounds
    // (throwing inside an OpenMP region would call std::terminate).
    if (dim == REDUCE_ALL) {
        if (input.numel() == 0)
            throw std::invalid_argument("max: cannot reduce over a zero-size dimension");
    } else if (input_shape[dim] == 0) {
        throw std::invalid_argument("max: cannot reduce over a zero-size dimension");
    }

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            if (dim == REDUCE_ALL) {
                output_data[0] = max_impl(input_data, input.numel());
            } else {
                max_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Float16: {
            auto* input_data = input.data<Float16>();
            auto* output_data = output.data<Float16>();

            if (dim == REDUCE_ALL) {
                // Compute max in Float32; propagate NaN on first encounter.
                const int64_t n = input.numel();
                float max_val = static_cast<float>(input_data[0]);
                bool saw_nan = std::isnan(max_val);
                for (int64_t i = 1; i < n && !saw_nan; i++) {
                    float val = static_cast<float>(input_data[i]);
                    if (std::isnan(val)) { saw_nan = true; break; }
                    if (val > max_val) max_val = val;
                }
                output_data[0] = Float16(saw_nan ? std::numeric_limits<float>::quiet_NaN() : max_val);
            } else {
                // Dimensional reduction - compute in Float32
                const int64_t dim_size = input_shape[dim];

                int64_t output_size = 1;
                for (int64_t d = 0; d < ndim; d++) {
                    if (d != dim) output_size *= input_shape[d];
                }

                const int64_t total_work = output_size * dim_size;
                #pragma omp parallel for if(total_work > REDUCTION_OMP_THRESHOLD)
                for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
                    std::vector<int64_t> indices(ndim, 0);
                    int64_t tmp = out_idx;
                    for (int64_t d = ndim - 1; d >= 0; --d) {
                        if (d == dim) continue;
                        indices[d] = tmp % input_shape[d];
                        tmp /= input_shape[d];
                    }

                    indices[dim] = 0;
                    int64_t in_idx0 = 0;
                    for (int64_t d = 0; d < ndim; d++) in_idx0 += indices[d] * input_strides[d];
                    float max_val = static_cast<float>(input_data[in_idx0]);
                    bool saw_nan = std::isnan(max_val);
                    for (int64_t i = 1; i < dim_size && !saw_nan; i++) {
                        indices[dim] = i;
                        int64_t in_idx = 0;
                        for (int64_t d = 0; d < ndim; d++) in_idx += indices[d] * input_strides[d];
                        float val = static_cast<float>(input_data[in_idx]);
                        if (std::isnan(val)) { saw_nan = true; break; }
                        if (val > max_val) max_val = val;
                    }
                    output_data[out_idx] = Float16(saw_nan ? std::numeric_limits<float>::quiet_NaN() : max_val);
                }
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (dim == REDUCE_ALL) {
                output_data[0] = max_impl(input_data, input.numel());
            } else {
                max_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int32_t>();

            if (dim == REDUCE_ALL) {
                output_data[0] = max_impl(input_data, input.numel());
            } else {
                max_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (dim == REDUCE_ALL) {
                output_data[0] = max_impl(input_data, input.numel());
            } else {
                max_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::BFloat16: {
            auto* input_data = input.data<BFloat16>();
            auto* output_data = output.data<BFloat16>();

            if (dim == REDUCE_ALL) {
                // Compute max in Float32; propagate NaN on first encounter.
                const int64_t n = input.numel();
                float max_val = static_cast<float>(input_data[0]);
                bool saw_nan = std::isnan(max_val);
                for (int64_t i = 1; i < n && !saw_nan; i++) {
                    float val = static_cast<float>(input_data[i]);
                    if (std::isnan(val)) { saw_nan = true; break; }
                    if (val > max_val) max_val = val;
                }
                output_data[0] = BFloat16(saw_nan ? std::numeric_limits<float>::quiet_NaN() : max_val);
            } else {
                // Dimensional reduction - compute in Float32
                const int64_t dim_size = input_shape[dim];

                int64_t output_size = 1;
                for (int64_t d = 0; d < ndim; d++) {
                    if (d != dim) output_size *= input_shape[d];
                }

                const int64_t total_work = output_size * dim_size;
                #pragma omp parallel for if(total_work > REDUCTION_OMP_THRESHOLD)
                for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
                    std::vector<int64_t> indices(ndim, 0);
                    int64_t tmp = out_idx;
                    for (int64_t d = ndim - 1; d >= 0; --d) {
                        if (d == dim) continue;
                        indices[d] = tmp % input_shape[d];
                        tmp /= input_shape[d];
                    }

                    indices[dim] = 0;
                    int64_t in_idx0 = 0;
                    for (int64_t d = 0; d < ndim; d++) in_idx0 += indices[d] * input_strides[d];
                    float max_val = static_cast<float>(input_data[in_idx0]);
                    bool saw_nan = std::isnan(max_val);
                    for (int64_t i = 1; i < dim_size && !saw_nan; i++) {
                        indices[dim] = i;
                        int64_t in_idx = 0;
                        for (int64_t d = 0; d < ndim; d++) in_idx += indices[d] * input_strides[d];
                        float val = static_cast<float>(input_data[in_idx]);
                        if (std::isnan(val)) { saw_nan = true; break; }
                        if (val > max_val) max_val = val;
                    }
                    output_data[out_idx] = BFloat16(saw_nan ? std::numeric_limits<float>::quiet_NaN() : max_val);
                }
            }
            break;
        }
        default:
            throw std::runtime_error("max: unsupported dtype");
    }

    return output;
}

// Template for min reduction - uses OpenMP reduction(min:) for efficiency
template<typename T>
auto min_impl(const T* input_data, int64_t n) -> T {
    if (n == 0) throw std::runtime_error("min: input tensor is empty");

    T min_val = input_data[0];

    // OpenMP 3.1+ supports reduction(min:) for built-in types
    #pragma omp parallel for reduction(min:min_val) if(n > REDUCTION_OMP_THRESHOLD)
    for (int64_t i = 1; i < n; i++) {
        if (input_data[i] < min_val) {
            min_val = input_data[i];
        }
    }

    return min_val;
}

// Specialization for float - uses SIMD vectorized reduction
template<>
auto min_impl<float>(const float* input_data, int64_t n) -> float {
    if (n == 0) throw std::runtime_error("min: input tensor is empty");
    return parallel_simd_min_f32(input_data, n);
}

template<>
auto min_impl<double>(const double* input_data, int64_t n) -> double {
    if (n == 0) throw std::runtime_error("min: input tensor is empty");
    return parallel_simd_min_f64(input_data, n);
}

// Min along a specific dimension
template<typename T>
void min_along_dim(const T* input_data,
                   T* output_data,
                   const std::vector<int64_t>& input_shape,
                   const std::vector<int64_t>& input_strides,
                   int64_t dim) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    // Compute output size
    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    // Reduction along dimension
    const int64_t total_work = output_size * dim_size;
    #pragma omp parallel for if(total_work > REDUCTION_OMP_THRESHOLD)
    for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
        std::vector<int64_t> indices(ndim, 0);
        int64_t tmp = out_idx;

        for (int64_t d = ndim - 1; d >= 0; --d) {
            if (d == dim) continue;
            indices[d] = tmp % input_shape[d];
            tmp /= input_shape[d];
        }

        // Find min along the reduction dimension
        indices[dim] = 0;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }
        T min_val = input_data[in_idx];
        // For floating-point types propagate NaN to match the full-reduction
        // and Float16/BFloat16 dimensional paths; `x < min_val` is false for
        // any NaN so a bare comparison would silently drop NaN elements.
        if constexpr (std::is_floating_point_v<T>) {
            bool saw_nan = std::isnan(min_val);
            for (int64_t i = 1; i < dim_size && !saw_nan; i++) {
                indices[dim] = i;
                in_idx = 0;
                for (int64_t d = 0; d < ndim; d++) {
                    in_idx += indices[d] * input_strides[d];
                }
                const T v = input_data[in_idx];
                if (std::isnan(v)) { saw_nan = true; break; }
                if (v < min_val) {
                    min_val = v;
                }
            }
            output_data[out_idx] = saw_nan ? std::numeric_limits<T>::quiet_NaN() : min_val;
        } else {
            for (int64_t i = 1; i < dim_size; i++) {
                indices[dim] = i;
                in_idx = 0;
                for (int64_t d = 0; d < ndim; d++) {
                    in_idx += indices[d] * input_strides[d];
                }
                if (input_data[in_idx] < min_val) {
                    min_val = input_data[in_idx];
                }
            }
            output_data[out_idx] = min_val;
        }
    }
}

auto min_kernel(const Tensor& input_raw, int64_t dim, bool keepdim) -> Tensor {
    // Mirror sum_kernel: contiguify at entry so the REDUCE_ALL flat-pointer
    // path does not read off a non-contiguous view's footprint.
    auto input = input_raw.contiguous();
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize negative dimension
    dim = normalize_dim(dim, ndim);

    // min has no identity element: a zero-size reduction is undefined. Guard
    // BOTH the full-reduction (numel==0) and per-dim (dim_size==0) empty cases
    // here at kernel entry — BEFORE any OpenMP region — because the downstream
    // paths seed an accumulator from element [0] and would read out of bounds.
    if (dim == REDUCE_ALL) {
        if (input.numel() == 0)
            throw std::invalid_argument("min: cannot reduce over a zero-size dimension");
    } else if (input_shape[dim] == 0) {
        throw std::invalid_argument("min: cannot reduce over a zero-size dimension");
    }

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            if (dim == REDUCE_ALL) {
                output_data[0] = min_impl(input_data, input.numel());
            } else {
                min_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Float16: {
            auto* input_data = input.data<Float16>();
            auto* output_data = output.data<Float16>();

            if (dim == REDUCE_ALL) {
                // Compute min in Float32; propagate NaN on first encounter.
                const int64_t n = input.numel();
                float min_val = static_cast<float>(input_data[0]);
                bool saw_nan = std::isnan(min_val);
                for (int64_t i = 1; i < n && !saw_nan; i++) {
                    float val = static_cast<float>(input_data[i]);
                    if (std::isnan(val)) { saw_nan = true; break; }
                    if (val < min_val) min_val = val;
                }
                output_data[0] = Float16(saw_nan ? std::numeric_limits<float>::quiet_NaN() : min_val);
            } else {
                // Dimensional reduction - compute in Float32
                const int64_t dim_size = input_shape[dim];

                int64_t output_size = 1;
                for (int64_t d = 0; d < ndim; d++) {
                    if (d != dim) output_size *= input_shape[d];
                }

                const int64_t total_work = output_size * dim_size;
                #pragma omp parallel for if(total_work > REDUCTION_OMP_THRESHOLD)
                for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
                    std::vector<int64_t> indices(ndim, 0);
                    int64_t tmp = out_idx;
                    for (int64_t d = ndim - 1; d >= 0; --d) {
                        if (d == dim) continue;
                        indices[d] = tmp % input_shape[d];
                        tmp /= input_shape[d];
                    }

                    indices[dim] = 0;
                    int64_t in_idx0 = 0;
                    for (int64_t d = 0; d < ndim; d++) in_idx0 += indices[d] * input_strides[d];
                    float min_val = static_cast<float>(input_data[in_idx0]);
                    bool saw_nan = std::isnan(min_val);
                    for (int64_t i = 1; i < dim_size && !saw_nan; i++) {
                        indices[dim] = i;
                        int64_t in_idx = 0;
                        for (int64_t d = 0; d < ndim; d++) in_idx += indices[d] * input_strides[d];
                        float val = static_cast<float>(input_data[in_idx]);
                        if (std::isnan(val)) { saw_nan = true; break; }
                        if (val < min_val) min_val = val;
                    }
                    output_data[out_idx] = Float16(saw_nan ? std::numeric_limits<float>::quiet_NaN() : min_val);
                }
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (dim == REDUCE_ALL) {
                output_data[0] = min_impl(input_data, input.numel());
            } else {
                min_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int32_t>();

            if (dim == REDUCE_ALL) {
                output_data[0] = min_impl(input_data, input.numel());
            } else {
                min_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (dim == REDUCE_ALL) {
                output_data[0] = min_impl(input_data, input.numel());
            } else {
                min_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::BFloat16: {
            auto* input_data = input.data<BFloat16>();
            auto* output_data = output.data<BFloat16>();

            if (dim == REDUCE_ALL) {
                // Compute min in Float32; propagate NaN on first encounter.
                const int64_t n = input.numel();
                float min_val = static_cast<float>(input_data[0]);
                bool saw_nan = std::isnan(min_val);
                for (int64_t i = 1; i < n && !saw_nan; i++) {
                    float val = static_cast<float>(input_data[i]);
                    if (std::isnan(val)) { saw_nan = true; break; }
                    if (val < min_val) min_val = val;
                }
                output_data[0] = BFloat16(saw_nan ? std::numeric_limits<float>::quiet_NaN() : min_val);
            } else {
                // Dimensional reduction - compute in Float32
                const int64_t dim_size = input_shape[dim];

                int64_t output_size = 1;
                for (int64_t d = 0; d < ndim; d++) {
                    if (d != dim) output_size *= input_shape[d];
                }

                const int64_t total_work = output_size * dim_size;
                #pragma omp parallel for if(total_work > REDUCTION_OMP_THRESHOLD)
                for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
                    std::vector<int64_t> indices(ndim, 0);
                    int64_t tmp = out_idx;
                    for (int64_t d = ndim - 1; d >= 0; --d) {
                        if (d == dim) continue;
                        indices[d] = tmp % input_shape[d];
                        tmp /= input_shape[d];
                    }

                    indices[dim] = 0;
                    int64_t in_idx0 = 0;
                    for (int64_t d = 0; d < ndim; d++) in_idx0 += indices[d] * input_strides[d];
                    float min_val = static_cast<float>(input_data[in_idx0]);
                    bool saw_nan = std::isnan(min_val);
                    for (int64_t i = 1; i < dim_size && !saw_nan; i++) {
                        indices[dim] = i;
                        int64_t in_idx = 0;
                        for (int64_t d = 0; d < ndim; d++) in_idx += indices[d] * input_strides[d];
                        float val = static_cast<float>(input_data[in_idx]);
                        if (std::isnan(val)) { saw_nan = true; break; }
                        if (val < min_val) min_val = val;
                    }
                    output_data[out_idx] = BFloat16(saw_nan ? std::numeric_limits<float>::quiet_NaN() : min_val);
                }
            }
            break;
        }
        default:
            throw std::runtime_error("min: unsupported dtype");
    }

    return output;
}

// NaN test that is a no-op for non-floating types (integers have no NaN).
template<typename T>
inline bool arg_isnan(T v) {
    if constexpr (std::is_floating_point_v<T>) return std::isnan(v);
    else return false;
}
// NaN-aware argmax/argmin "should (cand) replace (best)?" PyTorch returns the
// index of the FIRST NaN (NaN sorts as the extreme). Iterating ascending and
// never replacing once `best` is NaN keeps that first-NaN index; strict
// comparison keeps the lowest index on value ties.
template<typename T>
inline bool arg_takes_max(T cand, T best) {
    return !arg_isnan(best) && (arg_isnan(cand) || cand > best);
}
template<typename T>
inline bool arg_takes_min(T cand, T best) {
    return !arg_isnan(best) && (arg_isnan(cand) || cand < best);
}

// Template for argmax reduction - returns index of maximum value
template<typename T>
auto argmax_impl(const T* input_data, int64_t n) -> int64_t {
    if (n == 0) throw std::runtime_error("argmax: input tensor is empty");

    // Small arrays: stay single-threaded (OMP overhead not worth it).
    if (n < REDUCTION_OMP_THRESHOLD) {
        int64_t max_idx = 0;
        T max_val = input_data[0];
        for (int64_t i = 1; i < n; i++) {
            if (arg_takes_max(input_data[i], max_val)) {
                max_val = input_data[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    // Large arrays: parallel per-thread argmax, then single-threaded reduce.
    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif
    // Pad to cache line to avoid false sharing.
    struct alignas(64) LocalMax {
        T   val;
        int64_t idx;
        char _pad[64 - sizeof(T) - sizeof(int64_t) < 0 ? 0
                                                        : 64 - sizeof(T) - sizeof(int64_t)];
    };
    std::vector<LocalMax> thread_max(max_threads, {input_data[0], 0, {}});

    #pragma omp parallel
    {
        int tid = 0;
        int nthreads = 1;
#ifdef _OPENMP
        tid      = omp_get_thread_num();
        nthreads = omp_get_num_threads();
#endif
        int64_t chunk = (n + nthreads - 1) / nthreads;
        int64_t start = tid * chunk;
        int64_t end   = std::min(start + chunk, n);

        T   local_val = (start < end) ? input_data[start] : input_data[0];
        int64_t local_idx = (start < end) ? start : 0;

        for (int64_t i = start + 1; i < end; i++) {
            if (arg_takes_max(input_data[i], local_val)) {
                local_val = input_data[i];
                local_idx = i;
            }
        }
        thread_max[tid].val = local_val;
        thread_max[tid].idx = local_idx;
    }

    // Reduce across threads (sequential — tiny, ≤ max_threads iterations).
    int64_t max_idx = thread_max[0].idx;
    T       max_val = thread_max[0].val;
    for (int t = 1; t < max_threads; t++) {
        if (arg_takes_max(thread_max[t].val, max_val)) {
            max_val = thread_max[t].val;
            max_idx = thread_max[t].idx;
        }
    }
    return max_idx;
}

// Argmax along a specific dimension
template<typename T>
void argmax_along_dim(const T* input_data,
                      int64_t* output_data,
                      const std::vector<int64_t>& input_shape,
                      const std::vector<int64_t>& input_strides,
                      int64_t dim) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    // Compute output size
    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    // Find argmax along dimension
    const int64_t total_work = output_size * dim_size;
    #pragma omp parallel for if(total_work > REDUCTION_OMP_THRESHOLD)
    for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
        std::vector<int64_t> indices(ndim, 0);
        int64_t tmp = out_idx;

        for (int64_t d = ndim - 1; d >= 0; --d) {
            if (d == dim) continue;
            indices[d] = tmp % input_shape[d];
            tmp /= input_shape[d];
        }

        // Find index of max along the reduction dimension
        indices[dim] = 0;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }

        T max_val = input_data[in_idx];
        int64_t max_idx = 0;

        for (int64_t i = 1; i < dim_size; i++) {
            indices[dim] = i;
            in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * input_strides[d];
            }

            if (arg_takes_max(input_data[in_idx], max_val)) {
                max_val = input_data[in_idx];
                max_idx = i;
            }
        }

        output_data[out_idx] = max_idx;
    }
}

auto argmax_kernel(const Tensor& input_raw, int64_t dim, bool keepdim) -> Tensor {
    // Mirror sum_kernel: contiguify at entry. The REDUCE_ALL flat-pointer path
    // (Float32/Float64/Int32/Int64/Float16/BFloat16 branches) previously read
    // off a non-contiguous view's footprint; only some integer branches called
    // .contiguous(). Doing it once here unifies every dtype/dim path.
    auto input = input_raw.contiguous();
    // Argmax always returns Int64 indices
    auto input_shape_span = input.shape();
    std::vector<int64_t> input_shape_vec(input_shape_span.begin(), input_shape_span.end());
    const int64_t ndim = static_cast<int64_t>(input_shape_vec.size());

    // Normalize negative dimension
    dim = normalize_dim(dim, ndim);

    // argmax has no identity element: the index of the max over an empty range
    // is undefined. Guard BOTH the full-reduction (numel==0) and per-dim
    // (dim_size==0) empty cases here at kernel entry — BEFORE any OpenMP region —
    // because the downstream paths seed from element [0] and read out of bounds.
    if (dim == REDUCE_ALL) {
        if (input.numel() == 0)
            throw std::invalid_argument("argmax: cannot reduce over a zero-size dimension");
    } else if (input_shape_vec[dim] == 0) {
        throw std::invalid_argument("argmax: cannot reduce over a zero-size dimension");
    }

    auto output_shape = compute_reduction_shape(input_shape_vec, dim, keepdim);
    auto output = Tensor::empty_uninitialized(output_shape, DType::Int64, input.device());

    auto input_shape = input.shape();
    auto input_strides = input.strides();

    // Output is always int64
    auto* output_data = output.data<int64_t>();

    // Handle different input dtypes
    switch (input.dtype()) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmax_impl(input_data, input.numel());
            } else {
                argmax_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Float16:
        case DType::BFloat16: {
            // Widen to Float32 for a NaN-aware comparison: argmax_impl /
            // argmax_along_dim propagate NaN like PyTorch (return the index of
            // the first NaN). The half types have no std::isnan and the prior
            // inline loops silently skipped NaN. (This also adds the
            // previously-missing BFloat16 argmax path.)
            auto input_f32 = input.to(DType::Float32);
            auto* input_data = input_f32.data<float>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmax_impl(input_data, input_f32.numel());
            } else {
                auto f32_strides = input_f32.strides();
                argmax_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(f32_strides.begin(), f32_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmax_impl(input_data, input.numel());
            } else {
                argmax_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmax_impl(input_data, input.numel());
            } else {
                argmax_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmax_impl(input_data, input.numel());
            } else {
                argmax_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Int8: {
            auto input_c = input.contiguous();
            auto* input_data = input_c.data<int8_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmax_impl(input_data, input_c.numel());
            } else {
                argmax_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Int16: {
            auto input_c = input.contiguous();
            auto* input_data = input_c.data<int16_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmax_impl(input_data, input_c.numel());
            } else {
                argmax_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::UInt8: {
            auto input_c = input.contiguous();
            auto* input_data = input_c.data<uint8_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmax_impl(input_data, input_c.numel());
            } else {
                argmax_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::UInt16: {
            auto input_c = input.contiguous();
            auto* input_data = input_c.data<uint16_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmax_impl(input_data, input_c.numel());
            } else {
                argmax_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::UInt32: {
            auto input_c = input.contiguous();
            auto* input_data = input_c.data<uint32_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmax_impl(input_data, input_c.numel());
            } else {
                argmax_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::UInt64: {
            auto input_c = input.contiguous();
            auto* input_data = input_c.data<uint64_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmax_impl(input_data, input_c.numel());
            } else {
                argmax_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Bool: {
            // Bool: false < true, so argmax returns index of first true
            auto input_c = input.contiguous();
            auto* input_data = input_c.data<bool>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmax_impl(input_data, input_c.numel());
            } else {
                argmax_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        default:
            throw std::runtime_error("argmax: unsupported dtype");
    }

    return output;
}

// Argmin implementation - find index of minimum value
template<typename T>
auto argmin_impl(const T* input_data, int64_t n) -> int64_t {
    if (n == 0) throw std::runtime_error("argmin: input tensor is empty");

    // Small arrays: stay single-threaded (OMP overhead not worth it).
    if (n < REDUCTION_OMP_THRESHOLD) {
        int64_t min_idx = 0;
        T min_val = input_data[0];
        for (int64_t i = 1; i < n; i++) {
            if (arg_takes_min(input_data[i], min_val)) {
                min_val = input_data[i];
                min_idx = i;
            }
        }
        return min_idx;
    }

    // Large arrays: parallel per-thread argmin, then single-threaded reduce.
    // Mirrors argmax_impl so the two siblings share throughput characteristics.
    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif
    // Pad to cache line to avoid false sharing.
    struct alignas(64) LocalMin {
        T   val;
        int64_t idx;
        char _pad[64 - sizeof(T) - sizeof(int64_t) < 0 ? 0
                                                        : 64 - sizeof(T) - sizeof(int64_t)];
    };
    std::vector<LocalMin> thread_min(max_threads, {input_data[0], 0, {}});

    #pragma omp parallel
    {
        int tid = 0;
        int nthreads = 1;
#ifdef _OPENMP
        tid      = omp_get_thread_num();
        nthreads = omp_get_num_threads();
#endif
        int64_t chunk = (n + nthreads - 1) / nthreads;
        int64_t start = tid * chunk;
        int64_t end   = std::min(start + chunk, n);

        T   local_val = (start < end) ? input_data[start] : input_data[0];
        int64_t local_idx = (start < end) ? start : 0;

        for (int64_t i = start + 1; i < end; i++) {
            if (arg_takes_min(input_data[i], local_val)) {
                local_val = input_data[i];
                local_idx = i;
            }
        }
        thread_min[tid].val = local_val;
        thread_min[tid].idx = local_idx;
    }

    // Reduce across threads sequentially. Threads are merged in ascending tid
    // order and arg_takes_min only replaces on a strict improvement, so the
    // lowest index wins ties — matching PyTorch's first-occurrence semantics.
    int64_t min_idx = thread_min[0].idx;
    T       min_val = thread_min[0].val;
    for (int t = 1; t < max_threads; t++) {
        if (arg_takes_min(thread_min[t].val, min_val)) {
            min_val = thread_min[t].val;
            min_idx = thread_min[t].idx;
        }
    }
    return min_idx;
}

// Argmin along a specific dimension
template<typename T>
void argmin_along_dim(const T* input_data,
                      int64_t* output_data,
                      const std::vector<int64_t>& input_shape,
                      const std::vector<int64_t>& input_strides,
                      int64_t dim) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    // Compute output size
    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    // Find argmin along dimension
    const int64_t total_work = output_size * dim_size;
    #pragma omp parallel for if(total_work > REDUCTION_OMP_THRESHOLD)
    for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
        std::vector<int64_t> indices(ndim, 0);
        int64_t tmp = out_idx;

        for (int64_t d = ndim - 1; d >= 0; --d) {
            if (d == dim) continue;
            indices[d] = tmp % input_shape[d];
            tmp /= input_shape[d];
        }

        // Find index of min along the reduction dimension
        indices[dim] = 0;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }

        T min_val = input_data[in_idx];
        int64_t min_idx = 0;

        for (int64_t i = 1; i < dim_size; i++) {
            indices[dim] = i;
            in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * input_strides[d];
            }

            if (arg_takes_min(input_data[in_idx], min_val)) {
                min_val = input_data[in_idx];
                min_idx = i;
            }
        }

        output_data[out_idx] = min_idx;
    }
}

// Argmin kernel - returns indices of minimum values
auto argmin_kernel(const Tensor& input_raw, int64_t dim, bool keepdim) -> Tensor {
    // Mirror sum_kernel: contiguify at entry so every dtype/dim path (including
    // the REDUCE_ALL flat-pointer branches) is safe on non-contiguous views.
    auto input = input_raw.contiguous();
    // Argmin always returns Int64 indices
    auto input_shape_span = input.shape();
    std::vector<int64_t> input_shape_vec(input_shape_span.begin(), input_shape_span.end());
    const int64_t ndim = static_cast<int64_t>(input_shape_vec.size());

    // Normalize negative dimension
    dim = normalize_dim(dim, ndim);

    // argmin has no identity element: the index of the min over an empty range
    // is undefined. Guard BOTH the full-reduction (numel==0) and per-dim
    // (dim_size==0) empty cases here at kernel entry — BEFORE any OpenMP region —
    // because the downstream paths seed from element [0] and read out of bounds.
    if (dim == REDUCE_ALL) {
        if (input.numel() == 0)
            throw std::invalid_argument("argmin: cannot reduce over a zero-size dimension");
    } else if (input_shape_vec[dim] == 0) {
        throw std::invalid_argument("argmin: cannot reduce over a zero-size dimension");
    }

    auto output_shape = compute_reduction_shape(input_shape_vec, dim, keepdim);
    auto output = Tensor::empty_uninitialized(output_shape, DType::Int64, input.device());

    auto input_shape = input.shape();
    auto input_strides = input.strides();

    // Output is always int64
    auto* output_data = output.data<int64_t>();

    // Handle different input dtypes
    switch (input.dtype()) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmin_impl(input_data, input.numel());
            } else {
                argmin_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Float16:
        case DType::BFloat16: {
            // Widen to Float32 for a NaN-aware comparison: argmin_impl /
            // argmin_along_dim propagate NaN like PyTorch (return the index of
            // the first NaN). The half types have no std::isnan and the prior
            // inline loops silently skipped NaN.
            auto input_f32 = input.to(DType::Float32);
            auto* input_data = input_f32.data<float>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmin_impl(input_data, input_f32.numel());
            } else {
                auto f32_strides = input_f32.strides();
                argmin_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(f32_strides.begin(), f32_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmin_impl(input_data, input.numel());
            } else {
                argmin_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmin_impl(input_data, input.numel());
            } else {
                argmin_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmin_impl(input_data, input.numel());
            } else {
                argmin_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Int8: {
            auto input_c = input.contiguous();
            auto* input_data = input_c.data<int8_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmin_impl(input_data, input_c.numel());
            } else {
                argmin_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Int16: {
            auto input_c = input.contiguous();
            auto* input_data = input_c.data<int16_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmin_impl(input_data, input_c.numel());
            } else {
                argmin_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::UInt8: {
            auto input_c = input.contiguous();
            auto* input_data = input_c.data<uint8_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmin_impl(input_data, input_c.numel());
            } else {
                argmin_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::UInt16: {
            auto input_c = input.contiguous();
            auto* input_data = input_c.data<uint16_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmin_impl(input_data, input_c.numel());
            } else {
                argmin_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::UInt32: {
            auto input_c = input.contiguous();
            auto* input_data = input_c.data<uint32_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmin_impl(input_data, input_c.numel());
            } else {
                argmin_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::UInt64: {
            auto input_c = input.contiguous();
            auto* input_data = input_c.data<uint64_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmin_impl(input_data, input_c.numel());
            } else {
                argmin_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Bool: {
            // Bool: false < true, so argmin returns index of first false
            auto input_c = input.contiguous();
            auto* input_data = input_c.data<bool>();
            if (dim == REDUCE_ALL) {
                output_data[0] = argmin_impl(input_data, input_c.numel());
            } else {
                argmin_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        default:
            throw std::runtime_error("argmin: unsupported dtype");
    }

    return output;
}

// Argsort kernel - returns indices that would sort the input
// Uses parallel merge sort for large arrays via OpenMP task-based parallelism

namespace {

// Parallel merge sort threshold: below this, use std::sort
constexpr int64_t PARALLEL_SORT_THRESHOLD = 32768;

template<typename T, typename Comp>
void merge_halves(int64_t* indices, int64_t* buffer, int64_t left, int64_t mid, int64_t right, Comp comp) {
    int64_t i = left, j = mid, k = left;
    while (i < mid && j < right) {
        if (comp(indices[i], indices[j])) {
            buffer[k++] = indices[i++];
        } else {
            buffer[k++] = indices[j++];
        }
    }
    while (i < mid) buffer[k++] = indices[i++];
    while (j < right) buffer[k++] = indices[j++];
    std::copy(buffer + left, buffer + right, indices + left);
}

template<typename T, typename Comp>
void parallel_merge_sort(int64_t* indices, int64_t* buffer, int64_t left, int64_t right, Comp comp, int depth) {
    int64_t n = right - left;
    if (n <= PARALLEL_SORT_THRESHOLD || depth <= 0) {
        std::sort(indices + left, indices + right, comp);
        return;
    }

    int64_t mid = left + n / 2;

    #pragma omp task shared(indices, buffer) if(depth > 0)
    parallel_merge_sort<T>(indices, buffer, left, mid, comp, depth - 1);

    #pragma omp task shared(indices, buffer) if(depth > 0)
    parallel_merge_sort<T>(indices, buffer, mid, right, comp, depth - 1);

    #pragma omp taskwait

    merge_halves<T>(indices, buffer, left, mid, right, comp);
}

} // anonymous namespace

// F065: NaN-aware comparison for argsort. NaN is the LARGEST value (sorts last
// ascending / first descending) and NaN == NaN so ties (including NaN-vs-NaN)
// break by original index — matching the CUDA argsort radix NaN→+inf remap and
// PyTorch. NaN is detected from the IEEE-754 bit pattern so it is correct even
// under -ffast-math/-ffinite-math-only (where std::isnan folds to false).
namespace {
inline bool as_isnan(float x) {
    uint32_t u; std::memcpy(&u, &x, sizeof(u));
    return (u & 0x7fffffffu) > 0x7f800000u;
}
inline bool as_isnan(double x) {
    uint64_t u; std::memcpy(&u, &x, sizeof(u));
    return (u & 0x7fffffffffffffffull) > 0x7ff0000000000000ull;
}
template <typename T> inline bool as_nan_lt(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        bool na = as_isnan(a), nb = as_isnan(b);
        if (na || nb) return !na && nb;
        return a < b;
    } else { return a < b; }
}
template <typename T> inline bool as_nan_gt(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        bool na = as_isnan(a), nb = as_isnan(b);
        if (na || nb) return na && !nb;
        return a > b;
    } else { return a > b; }
}
template <typename T> inline bool as_nan_eq(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        bool na = as_isnan(a), nb = as_isnan(b);
        if (na || nb) return na && nb;
        return a == b;
    } else { return a == b; }
}
}  // namespace

template<typename T>
auto argsort_impl(const T* data, int64_t n, bool descending) -> std::vector<int64_t> {
    // Create index array
    std::vector<int64_t> indices(n);
    for (int64_t i = 0; i < n; ++i) {
        indices[i] = i;
    }

    if (n <= PARALLEL_SORT_THRESHOLD) {
        // Small array: use standard sort
        if (descending) {
            std::sort(indices.begin(), indices.end(),
                     [data](int64_t a, int64_t b) { return as_nan_gt(data[a], data[b]) || (as_nan_eq(data[a], data[b]) && a < b); });
        } else {
            std::sort(indices.begin(), indices.end(),
                     [data](int64_t a, int64_t b) { return as_nan_lt(data[a], data[b]) || (as_nan_eq(data[a], data[b]) && a < b); });
        }
    } else {
        // Large array: use parallel merge sort
        std::vector<int64_t> buffer(n);
        // Depth limits parallelism to ~num_threads levels
        int depth = 0;
        #ifdef _OPENMP
        depth = static_cast<int>(std::log2(omp_get_max_threads())) + 1;
        #endif

        if (descending) {
            auto comp = [data](int64_t a, int64_t b) { return as_nan_gt(data[a], data[b]) || (as_nan_eq(data[a], data[b]) && a < b); };
            #pragma omp parallel
            {
                #pragma omp single
                parallel_merge_sort<T>(indices.data(), buffer.data(), 0, n, comp, depth);
            }
        } else {
            auto comp = [data](int64_t a, int64_t b) { return as_nan_lt(data[a], data[b]) || (as_nan_eq(data[a], data[b]) && a < b); };
            #pragma omp parallel
            {
                #pragma omp single
                parallel_merge_sort<T>(indices.data(), buffer.data(), 0, n, comp, depth);
            }
        }
    }

    return indices;
}

template<typename T>
void argsort_along_dim(const T* input_data, int64_t* output_data,
                      const std::vector<int64_t>& shape,
                      [[maybe_unused]] const std::vector<int64_t>& strides,
                      int64_t dim, bool descending) {
    const int64_t ndim = shape.size();
    const int64_t dim_size = shape[dim];

    // Compute total number of elements
    int64_t total_elems = 1;
    for (auto s : shape) total_elems *= s;

    // Compute size of inner dimensions (after dim)
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
        inner_size *= shape[d];
    }

    // Compute size of outer dimensions (before dim)
    int64_t outer_size = total_elems / (dim_size * inner_size);

    // For each outer x inner combination, sort along dim
    #pragma omp parallel for if(outer_size * inner_size > ::tenzor::OmpThresholds::matmul())
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            // Collect values along the dimension
            std::vector<T> values(dim_size);
            for (int64_t i = 0; i < dim_size; ++i) {
                int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
                values[i] = input_data[offset];
            }

            // Get sorted indices
            auto sorted_indices = argsort_impl(values.data(), dim_size, descending);

            // Write sorted indices to output
            for (int64_t i = 0; i < dim_size; ++i) {
                int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
                output_data[offset] = sorted_indices[i];
            }
        }
    }
}

auto argsort_kernel(const Tensor& input_raw, int64_t dim, bool descending) -> Tensor {
    // Materialize a contiguous copy: argsort_along_dim computes flat element
    // offsets purely from shape (it ignores strides), so a non-contiguous view
    // (transpose, sliced view with offset, expand with stride 0) would otherwise
    // read the wrong storage elements. Mirrors sum_kernel/max_kernel.
    auto input = input_raw.contiguous();
    const int64_t ndim = input.ndim();

    // Normalize dimension
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("argsort: dimension out of range");
    }

    // Output has same shape as input but with Int64 dtype
    std::vector<int64_t> output_shape(input.shape().begin(), input.shape().end());
    Tensor output(output_shape, DType::Int64, input.device());
    int64_t* output_data = output.data<int64_t>();

    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();

    // Dispatch based on input dtype
    switch (input.dtype()) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            argsort_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim, descending);
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            argsort_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim, descending);
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            argsort_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim, descending);
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            argsort_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim, descending);
            break;
        }
        case DType::Float16: {
            // Convert Float16 to Float32 for sorting
            auto input_f32 = input.to(DType::Float32);
            auto* input_data = input_f32.data<float>();
            argsort_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim, descending);
            break;
        }
        default:
            throw std::runtime_error("argsort: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Product, Variance, and Standard Deviation operations
// ============================================================================

// Template for product reduction
template<typename T>
auto prod_impl(const T* input_data, int64_t n) -> T {
    if (n == 0) return T(1);  // Empty product is 1

    T result = T(1);
    #pragma omp parallel for reduction(*:result) if(n > ::tenzor::OmpThresholds::medium())
    for (int64_t i = 0; i < n; i++) {
        result *= input_data[i];
    }
    return result;
}

// Complex specializations: OpenMP has no built-in reduction(*) for
// std::complex, so use a plain scalar loop (mirrors sum_impl's complex spec).
template<>
inline auto prod_impl<std::complex<float>>(const std::complex<float>* input_data,
                                           int64_t n) -> std::complex<float> {
    std::complex<float> result{1.0f, 0.0f};
    for (int64_t i = 0; i < n; ++i) result *= input_data[i];
    return result;
}

template<>
inline auto prod_impl<std::complex<double>>(const std::complex<double>* input_data,
                                            int64_t n) -> std::complex<double> {
    std::complex<double> result{1.0, 0.0};
    for (int64_t i = 0; i < n; ++i) result *= input_data[i];
    return result;
}

// Product along a specific dimension
template<typename T>
void prod_along_dim(const T* input_data,
                    T* output_data,
                    const std::vector<int64_t>& input_shape,
                    const std::vector<int64_t>& input_strides,
                    int64_t dim) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    // Initialize output to 1 (identity for multiplication)
    std::fill(output_data, output_data + output_size, T(1));

    #pragma omp parallel for if(output_size > ::tenzor::OmpThresholds::matmul())
    for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
        // Compute indices for this output position
        std::vector<int64_t> indices(ndim);
        int64_t tmp = out_idx;
        for (int64_t d = ndim - 1; d >= 0; d--) {
            if (d == dim) continue;
            indices[d] = tmp % input_shape[d];
            tmp /= input_shape[d];
        }

        // Product along dimension
        T prod_val = T(1);
        for (int64_t i = 0; i < dim_size; i++) {
            indices[dim] = i;
            int64_t in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * input_strides[d];
            }
            prod_val *= input_data[in_idx];
        }
        output_data[out_idx] = prod_val;
    }
}

// Public API for product
auto prod_kernel(const Tensor& input_raw, int64_t dim, bool keepdim) -> Tensor {
    // Mirror sum_kernel: contiguify at entry so the REDUCE_ALL flat-pointer
    // path does not read off a non-contiguous view's footprint.
    auto input = input_raw.contiguous();
    auto shape_span = input.shape();
    std::vector<int64_t> input_shape(shape_span.begin(), shape_span.end());
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize negative dimension
    dim = normalize_dim(dim, ndim);

    auto output_shape = compute_reduction_shape(input_shape, dim, keepdim);

    Tensor output(output_shape, input.dtype(), input.device());

    switch (input.dtype()) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();
            if (dim == REDUCE_ALL) {
                output_data[0] = prod_impl(input_data, input.numel());
            } else {
                auto strides_span = input.strides();
                std::vector<int64_t> input_strides(strides_span.begin(), strides_span.end());
                prod_along_dim(input_data, output_data, input_shape, input_strides, dim);
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();
            if (dim == REDUCE_ALL) {
                output_data[0] = prod_impl(input_data, input.numel());
            } else {
                auto strides_span = input.strides();
                std::vector<int64_t> input_strides(strides_span.begin(), strides_span.end());
                prod_along_dim(input_data, output_data, input_shape, input_strides, dim);
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int32_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = prod_impl(input_data, input.numel());
            } else {
                auto strides_span = input.strides();
                std::vector<int64_t> input_strides(strides_span.begin(), strides_span.end());
                prod_along_dim(input_data, output_data, input_shape, input_strides, dim);
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();
            if (dim == REDUCE_ALL) {
                output_data[0] = prod_impl(input_data, input.numel());
            } else {
                auto strides_span = input.strides();
                std::vector<int64_t> input_strides(strides_span.begin(), strides_span.end());
                prod_along_dim(input_data, output_data, input_shape, input_strides, dim);
            }
            break;
        }
        case DType::Float16:
        case DType::BFloat16: {
            // Compute the product in Float32 for numerical headroom
            // (half-dtype products overflow trivially), then narrow back
            // to the input dtype so the caller's expected output dtype
            // is preserved (audit item E.2 — previous code left the
            // output as Float32, silently changing the dtype contract).
            const DType orig = input.dtype();
            Tensor input_f32 = input.to(DType::Float32);
            auto* input_data = input_f32.data<float>();

            Tensor out_f32(output_shape, DType::Float32, input.device());
            auto* out_data = out_f32.data<float>();
            if (dim == REDUCE_ALL) {
                out_data[0] = prod_impl(input_data, input.numel());
            } else {
                auto strides_span = input_f32.strides();
                std::vector<int64_t> input_strides(strides_span.begin(), strides_span.end());
                prod_along_dim(input_data, out_data,
                              std::vector<int64_t>(input_f32.shape().begin(), input_f32.shape().end()),
                              input_strides, dim);
            }
            output = out_f32.to(orig);
            break;
        }
        case DType::Complex64: {
            auto* input_data = input.data<std::complex<float>>();
            auto* output_data = output.data<std::complex<float>>();
            if (dim == REDUCE_ALL) {
                output_data[0] = prod_impl(input_data, input.numel());
            } else {
                auto strides_span = input.strides();
                std::vector<int64_t> input_strides(strides_span.begin(), strides_span.end());
                prod_along_dim(input_data, output_data, input_shape, input_strides, dim);
            }
            break;
        }
        case DType::Complex128: {
            auto* input_data = input.data<std::complex<double>>();
            auto* output_data = output.data<std::complex<double>>();
            if (dim == REDUCE_ALL) {
                output_data[0] = prod_impl(input_data, input.numel());
            } else {
                auto strides_span = input.strides();
                std::vector<int64_t> input_strides(strides_span.begin(), strides_span.end());
                prod_along_dim(input_data, output_data, input_shape, input_strides, dim);
            }
            break;
        }
        default:
            throw std::runtime_error("prod: unsupported dtype");
    }

    return output;
}

// Variance along a specific dimension using two-pass algorithm
template<typename T>
void var_along_dim(const T* input_data,
                   T* output_data,
                   const std::vector<int64_t>& input_shape,
                   const std::vector<int64_t>& input_strides,
                   int64_t dim,
                   int64_t correction) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) output_size *= input_shape[i];
    }

    int64_t divisor = dim_size - correction;
    if (divisor <= 0) {
        // Unbiased variance with n <= correction is undefined — fill with NaN
        for (int64_t i = 0; i < output_size; i++) {
            output_data[i] = std::numeric_limits<T>::quiet_NaN();
        }
        return;
    }

    const int64_t total_work = output_size * dim_size;
    #pragma omp parallel for if(total_work > REDUCTION_OMP_THRESHOLD)
    for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
        std::vector<int64_t> indices(ndim, 0);
        int64_t tmp = out_idx;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            if (d == dim) continue;
            indices[d] = tmp % input_shape[d];
            tmp /= input_shape[d];
        }

        // Single-pass Welford's algorithm for numerically stable variance
        T mean = T(0);
        T M2 = T(0);
        for (int64_t i = 0; i < dim_size; i++) {
            indices[dim] = i;
            int64_t in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * input_strides[d];
            }
            T x = input_data[in_idx];
            T delta = x - mean;
            mean += delta / static_cast<T>(i + 1);
            T delta2 = x - mean;
            M2 += delta * delta2;
        }
        output_data[out_idx] = M2 / static_cast<T>(divisor);
    }
}

// Variance using two-pass algorithm for numerical stability
auto var_kernel(const Tensor& input_raw, int64_t dim, bool keepdim, int64_t correction) -> Tensor {
    // Mirror sum_kernel: contiguify at entry so the REDUCE_ALL full-reduction
    // loop does not read off a non-contiguous view's footprint.
    auto input = input_raw.contiguous();
    auto shape_span = input.shape();
    std::vector<int64_t> input_shape(shape_span.begin(), shape_span.end());
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize negative dimension
    dim = normalize_dim(dim, ndim);

    auto output_shape = compute_reduction_shape(input_shape, dim, keepdim);

    // For Float16/BFloat16, compute and store in Float32
    DType output_dtype = input.dtype();
    if (output_dtype == DType::Float16 || output_dtype == DType::BFloat16) {
        output_dtype = DType::Float32;
    }

    Tensor output(output_shape, output_dtype, input.device());

    const int64_t n = input.numel();
    if (n == 0) {
        throw std::runtime_error("var: input tensor is empty");
    }

    if (dim == REDUCE_ALL) {
        switch (input.dtype()) {
            case DType::Float32: {
                auto* input_data = input.data<float>();
                auto* output_data = output.data<float>();
                int64_t divisor = n - correction;
                if (divisor <= 0) {
                    // Unbiased variance with n <= correction is undefined — return NaN
                    output_data[0] = std::numeric_limits<float>::quiet_NaN();
                    break;
                }
                float mean = sum_impl(input_data, n) / static_cast<float>(n);
                float var_sum = 0.0f;
                #pragma omp parallel for reduction(+:var_sum) if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    float diff = input_data[i] - mean;
                    var_sum += diff * diff;
                }
                output_data[0] = var_sum / static_cast<float>(divisor);
                break;
            }
            case DType::Float64: {
                auto* input_data = input.data<double>();
                auto* output_data = output.data<double>();
                int64_t divisor = n - correction;
                if (divisor <= 0) {
                    // Unbiased variance with n <= correction is undefined — return NaN
                    output_data[0] = std::numeric_limits<double>::quiet_NaN();
                    break;
                }
                double mean = sum_impl(input_data, n) / static_cast<double>(n);
                double var_sum = 0.0;
                #pragma omp parallel for reduction(+:var_sum) if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    double diff = input_data[i] - mean;
                    var_sum += diff * diff;
                }
                output_data[0] = var_sum / static_cast<double>(divisor);
                break;
            }
            case DType::Float16: {
                // Welford's single-pass algorithm for numerically stable variance
                // Float16 has limited precision; Welford avoids catastrophic cancellation
                auto* input_data = input.data<Float16>();
                auto* output_data = output.data<float>();
                int64_t divisor = n - correction;
                if (divisor <= 0) {
                    // Unbiased variance with n <= correction is undefined — return NaN
                    output_data[0] = std::numeric_limits<float>::quiet_NaN();
                    break;
                }
                float mean = 0.0f;
                float M2 = 0.0f;
                for (int64_t i = 0; i < n; i++) {
                    float x = static_cast<float>(input_data[i]);
                    float delta = x - mean;
                    mean += delta / static_cast<float>(i + 1);
                    float delta2 = x - mean;
                    M2 += delta * delta2;
                }
                output_data[0] = M2 / static_cast<float>(divisor);
                break;
            }
            case DType::BFloat16: {
                // Welford's single-pass algorithm for numerically stable variance
                // BFloat16 has only 8 mantissa bits; Welford avoids catastrophic cancellation
                auto* input_data = input.data<BFloat16>();
                auto* output_data = output.data<float>();
                int64_t divisor = n - correction;
                if (divisor <= 0) {
                    // Unbiased variance with n <= correction is undefined — return NaN
                    output_data[0] = std::numeric_limits<float>::quiet_NaN();
                    break;
                }
                float mean = 0.0f;
                float M2 = 0.0f;
                for (int64_t i = 0; i < n; i++) {
                    float x = static_cast<float>(input_data[i]);
                    float delta = x - mean;
                    mean += delta / static_cast<float>(i + 1);
                    float delta2 = x - mean;
                    M2 += delta * delta2;
                }
                output_data[0] = M2 / static_cast<float>(divisor);
                break;
            }
            default:
                throw std::runtime_error("var: unsupported dtype");
        }
    } else {
        // Dimensional reduction
        auto strides_span = input.strides();
        std::vector<int64_t> input_strides(strides_span.begin(), strides_span.end());

        switch (input.dtype()) {
            case DType::Float32: {
                var_along_dim<float>(input.data<float>(), output.data<float>(),
                                    input_shape, input_strides, dim, correction);
                break;
            }
            case DType::Float64: {
                var_along_dim<double>(input.data<double>(), output.data<double>(),
                                     input_shape, input_strides, dim, correction);
                break;
            }
            case DType::Float16: {
                // Welford's single-pass algorithm for numerically stable variance
                // Float16 has limited precision; single-pass avoids catastrophic cancellation
                const int64_t dim_size = input_shape[dim];
                const auto* input_data = input.data<Float16>();
                auto* output_data = output.data<float>();

                int64_t out_size = 1;
                for (int64_t d = 0; d < ndim; d++) {
                    if (d != dim) out_size *= input_shape[d];
                }

                int64_t divisor = dim_size - correction;
                if (divisor <= 0) {
                    // Unbiased variance with n <= correction is undefined — fill with NaN
                    for (int64_t i = 0; i < out_size; i++) {
                        output_data[i] = std::numeric_limits<float>::quiet_NaN();
                    }
                    break;
                }

                #pragma omp parallel for if(out_size * dim_size > REDUCTION_OMP_THRESHOLD)
                for (int64_t out_idx = 0; out_idx < out_size; out_idx++) {
                    std::vector<int64_t> indices(ndim, 0);
                    int64_t tmp = out_idx;
                    for (int64_t d = ndim - 1; d >= 0; --d) {
                        if (d == dim) continue;
                        indices[d] = tmp % input_shape[d];
                        tmp /= input_shape[d];
                    }

                    // Welford's algorithm with Float32 accumulation
                    float mean = 0.0f;
                    float M2 = 0.0f;
                    for (int64_t i = 0; i < dim_size; i++) {
                        indices[dim] = i;
                        int64_t in_idx = 0;
                        for (int64_t d = 0; d < ndim; d++) in_idx += indices[d] * input_strides[d];
                        float x = static_cast<float>(input_data[in_idx]);
                        float delta = x - mean;
                        mean += delta / static_cast<float>(i + 1);
                        float delta2 = x - mean;
                        M2 += delta * delta2;
                    }
                    output_data[out_idx] = M2 / static_cast<float>(divisor);
                }
                break;
            }
            case DType::BFloat16: {
                // Welford's single-pass algorithm for numerically stable variance
                // BFloat16 has only 8 mantissa bits; single-pass avoids catastrophic cancellation
                const int64_t dim_size = input_shape[dim];
                const auto* input_data = input.data<BFloat16>();
                auto* output_data = output.data<float>();

                int64_t out_size = 1;
                for (int64_t d = 0; d < ndim; d++) {
                    if (d != dim) out_size *= input_shape[d];
                }

                int64_t divisor = dim_size - correction;
                if (divisor <= 0) {
                    // Unbiased variance with n <= correction is undefined — fill with NaN
                    for (int64_t i = 0; i < out_size; i++) {
                        output_data[i] = std::numeric_limits<float>::quiet_NaN();
                    }
                    break;
                }

                #pragma omp parallel for if(out_size * dim_size > REDUCTION_OMP_THRESHOLD)
                for (int64_t out_idx = 0; out_idx < out_size; out_idx++) {
                    std::vector<int64_t> indices(ndim, 0);
                    int64_t tmp = out_idx;
                    for (int64_t d = ndim - 1; d >= 0; --d) {
                        if (d == dim) continue;
                        indices[d] = tmp % input_shape[d];
                        tmp /= input_shape[d];
                    }

                    // Welford's algorithm with Float32 accumulation
                    float mean = 0.0f;
                    float M2 = 0.0f;
                    for (int64_t i = 0; i < dim_size; i++) {
                        indices[dim] = i;
                        int64_t in_idx = 0;
                        for (int64_t d = 0; d < ndim; d++) in_idx += indices[d] * input_strides[d];
                        float x = static_cast<float>(input_data[in_idx]);
                        float delta = x - mean;
                        mean += delta / static_cast<float>(i + 1);
                        float delta2 = x - mean;
                        M2 += delta * delta2;
                    }
                    output_data[out_idx] = M2 / static_cast<float>(divisor);
                }
                break;
            }
            default:
                throw std::runtime_error("var: unsupported dtype");
        }
    }

    return output;
}

// Standard deviation (sqrt of variance)
auto std_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction) -> Tensor {
    auto var_result = var_kernel(input, dim, keepdim, correction);

    // Apply sqrt element-wise
    auto shape_span = var_result.shape();
    std::vector<int64_t> output_shape(shape_span.begin(), shape_span.end());
    Tensor output(output_shape, var_result.dtype(), var_result.device());

    const int64_t n = var_result.numel();

    switch (var_result.dtype()) {
        case DType::Float32: {
            auto* var_data = var_result.data<float>();
            auto* output_data = output.data<float>();
            #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
            for (int64_t i = 0; i < n; i++) {
                output_data[i] = std::sqrt(var_data[i]);
            }
            break;
        }
        case DType::Float64: {
            auto* var_data = var_result.data<double>();
            auto* output_data = output.data<double>();
            #pragma omp parallel for if(n > ::tenzor::OmpThresholds::medium())
            for (int64_t i = 0; i < n; i++) {
                output_data[i] = std::sqrt(var_data[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("std: unsupported dtype (got " + std::string(dtype_name(var_result.dtype())) + ")");
    }

    return output;
}

// Per-slice Lp norm over a single dimension. Returns a float32 output.
// Used as a fallback when norm_kernel is called with an explicit dim —
// the in-place vectorized "full reduction" path below does not generalize
// to partial reductions without reshaping, and correctness matters more
// than peak throughput for the dim-reduced case.
// Templatized per-dim norm kernel: T is the element type (float or double).
// Accumulation is always done in double for both paths so that Float64 inputs
// get full precision end-to-end without any F32 round-trip.
template <typename T>
static auto norm_kernel_dim(const Tensor& input, float p,
                            int64_t dim, bool keepdim) -> Tensor {
    auto shape_span = input.shape();
    std::vector<int64_t> input_shape(shape_span.begin(), shape_span.end());
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Work on a contiguous copy so we can index linearly. This is not the
    // hottest path in the library, so the extra copy is acceptable.
    Tensor cont = input.contiguous();

    // Compute outer (product of dims before `dim`), reduce (size of `dim`),
    // and inner (product of dims after `dim`). A flat-index iteration over
    // (outer, inner) gives a trivial O(n) implementation that matches the
    // semantics of torch.linalg.vector_norm(x, p, dim).
    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= input_shape[i];
    int64_t reduce_sz = input_shape[dim];
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner *= input_shape[i];

    // Output dtype matches input — Float32 → Float32, Float64 → Float64.
    constexpr DType out_dtype = std::is_same_v<T, double> ? DType::Float64 : DType::Float32;
    auto output_shape = compute_reduction_shape(input_shape, dim, keepdim);
    Tensor output(output_shape, out_dtype, input.device());
    auto* out_data = output.data<T>();
    const auto* in_data = cont.data<T>();

    auto reduce_slice = [&](int64_t o, int64_t i) -> T {
        double acc = 0.0;
        if (std::isinf(p)) {
            double m = 0.0;
            for (int64_t k = 0; k < reduce_sz; ++k) {
                double v = std::abs(static_cast<double>(in_data[(o * reduce_sz + k) * inner + i]));
                if (v > m) m = v;
            }
            return static_cast<T>(m);
        }
        if (p == 1.0f) {
            for (int64_t k = 0; k < reduce_sz; ++k) {
                acc += std::abs(static_cast<double>(in_data[(o * reduce_sz + k) * inner + i]));
            }
            return static_cast<T>(acc);
        }
        // p >= 2 / general: max-normalize the slice so squaring/powering large
        // magnitudes cannot overflow even the double accumulator (matters for
        // Float64 inputs with |x| > ~1e154; the double accumulator already
        // covers Float32). norm = max * (sum (|x|/max)^p)^(1/p).
        double m = 0.0;
        for (int64_t k = 0; k < reduce_sz; ++k) {
            double v = std::abs(static_cast<double>(in_data[(o * reduce_sz + k) * inner + i]));
            if (v > m) m = v;
        }
        if (m <= 0.0) return static_cast<T>(0);
        const double inv = 1.0 / m;
        if (p == 2.0f) {
            for (int64_t k = 0; k < reduce_sz; ++k) {
                double r = std::abs(static_cast<double>(in_data[(o * reduce_sz + k) * inner + i])) * inv;
                acc += r * r;
            }
            return static_cast<T>(m * std::sqrt(acc));
        }
        for (int64_t k = 0; k < reduce_sz; ++k) {
            double r = std::abs(static_cast<double>(in_data[(o * reduce_sz + k) * inner + i])) * inv;
            acc += std::pow(r, static_cast<double>(p));
        }
        return static_cast<T>(m * std::pow(acc, 1.0 / static_cast<double>(p)));
    };

    const int64_t slices = outer * inner;
    #pragma omp parallel for if(slices > 1024)
    for (int64_t s = 0; s < slices; ++s) {
        int64_t o = s / inner;
        int64_t i = s % inner;
        out_data[o * inner + i] = reduce_slice(o, i);
    }
    return output;
}

// Keep old name as thin wrapper for backward compatibility with any direct callers.
static auto norm_kernel_dim_float32(const Tensor& input, float p,
                                    int64_t dim, bool keepdim) -> Tensor {
    return norm_kernel_dim<float>(input, p, dim, keepdim);
}

// Norm operation - compute Lp norm
auto norm_kernel(const Tensor& input, float p, int64_t dim, bool keepdim) -> Tensor {
    auto shape_span = input.shape();
    std::vector<int64_t> input_shape(shape_span.begin(), shape_span.end());
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize negative dimension
    dim = normalize_dim(dim, ndim);

    if (dim != REDUCE_ALL) {
        // Per-dim reduction — Float32 / Float64 have native templates; for
        // Float16 / BFloat16 widen to Float32, compute, and narrow back so
        // half-dtype users get a working `norm(dim=...)` instead of a hard
        // throw (audit item E.1, feedback_float16_widen_narrow pattern).
        if (input.dtype() == DType::Float32) {
            return norm_kernel_dim_float32(input, p, dim, keepdim);
        }
        if (input.dtype() == DType::Float64) {
            // Dispatch directly to the double-precision template; no F32
            // round-trip, no precision loss.
            return norm_kernel_dim<double>(input, p, dim, keepdim);
        }
        if (input.dtype() == DType::Float16 ||
            input.dtype() == DType::BFloat16) {
            const DType orig = input.dtype();
            Tensor input_f32 = input.to(DType::Float32);
            Tensor out_f32 = norm_kernel_dim_float32(input_f32, p, dim, keepdim);
            return out_f32.to(orig);
        }
        throw std::runtime_error(
            "norm: dim reduction not supported for dtype " +
            std::string(dtype_name(input.dtype())) +
            " on CPU — only Float32 / Float64 / Float16 / BFloat16 are valid.");
    }

    auto output_shape = compute_reduction_shape(input_shape, dim, keepdim);
    Tensor output(output_shape, input.dtype(), input.device());

    const int64_t n = input.numel();
    if (n == 0) {
        throw std::runtime_error("norm: input tensor is empty");
    }

    // Lp norm for p >= 2 (and general p) is computed with max-normalization
    // (norm = max * (sum (|x|/max)^p)^(1/p)) so that squaring/powering large
    // magnitudes cannot overflow the accumulator to +inf. L1 and L-inf need no
    // scaling. This lambda is templated on the accumulator type (float for
    // F32/F16/BF16, double for F64) and a per-element |x| reader.
    auto scaled_lp = [&](auto zero, auto&& abs_at) {
        using Acc = decltype(zero);
        Acc max_val = zero;
        #pragma omp parallel for reduction(max:max_val) if(n > ::tenzor::OmpThresholds::medium())
        for (int64_t i = 0; i < n; i++) {
            Acc a = abs_at(i);
            if (a > max_val) max_val = a;
        }
        if (max_val <= zero) return zero;  // all-zero input
        Acc inv_max = Acc(1) / max_val;
        Acc acc = zero;
        const bool is_l2 = (p == static_cast<float>(2));
        #pragma omp parallel for reduction(+:acc) if(n > ::tenzor::OmpThresholds::medium())
        for (int64_t i = 0; i < n; i++) {
            Acc r = abs_at(i) * inv_max;
            acc += is_l2 ? (r * r) : std::pow(r, static_cast<Acc>(p));
        }
        return is_l2 ? (max_val * std::sqrt(acc))
                     : (max_val * std::pow(acc, Acc(1) / static_cast<Acc>(p)));
    };

    switch (input.dtype()) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();
            // Accumulate in double — matching the per-dim path norm_kernel_dim<float>
            // (which reduces in double) — then narrow to float only on store, so the
            // full-reduction norm(x) has the same precision as norm(x, dim=...) (F005).
            double norm_value = 0.0;
            if (p == 1.0f) {
                #pragma omp parallel for reduction(+:norm_value) if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) norm_value += std::abs(static_cast<double>(input_data[i]));
            } else if (std::isinf(p)) {
                #pragma omp parallel for reduction(max:norm_value) if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    double abs_val = std::abs(static_cast<double>(input_data[i]));
                    if (abs_val > norm_value) norm_value = abs_val;
                }
            } else {
                norm_value = scaled_lp(0.0, [&](int64_t i) { return std::abs(static_cast<double>(input_data[i])); });
            }
            output_data[0] = static_cast<float>(norm_value);
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();
            double norm_value = 0.0;
            if (p == 1.0f) {
                #pragma omp parallel for reduction(+:norm_value) if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) norm_value += std::abs(input_data[i]);
            } else if (std::isinf(p)) {
                #pragma omp parallel for reduction(max:norm_value) if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    double abs_val = std::abs(input_data[i]);
                    if (abs_val > norm_value) norm_value = abs_val;
                }
            } else {
                norm_value = scaled_lp(0.0, [&](int64_t i) { return std::abs(input_data[i]); });
            }
            output_data[0] = norm_value;
            break;
        }
        case DType::Float16: {
            auto* input_data = input.data<Float16>();
            output = Tensor(output_shape, DType::Float32, input.device());
            auto* output_data = output.data<float>();
            float norm_value = 0.0f;
            if (p == 1.0f) {
                #pragma omp parallel for reduction(+:norm_value) if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) norm_value += std::abs(static_cast<float>(input_data[i]));
            } else if (std::isinf(p)) {
                #pragma omp parallel for reduction(max:norm_value) if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    float abs_val = std::abs(static_cast<float>(input_data[i]));
                    if (abs_val > norm_value) norm_value = abs_val;
                }
            } else {
                norm_value = scaled_lp(0.0f, [&](int64_t i) { return std::abs(static_cast<float>(input_data[i])); });
            }
            output_data[0] = norm_value;
            break;
        }
        case DType::BFloat16: {
            auto* input_data = input.data<BFloat16>();
            output = Tensor(output_shape, DType::Float32, input.device());
            auto* output_data = output.data<float>();
            float norm_value = 0.0f;
            if (p == 1.0f) {
                #pragma omp parallel for reduction(+:norm_value) if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) norm_value += std::abs(static_cast<float>(input_data[i]));
            } else if (std::isinf(p)) {
                #pragma omp parallel for reduction(max:norm_value) if(n > ::tenzor::OmpThresholds::medium())
                for (int64_t i = 0; i < n; i++) {
                    float abs_val = std::abs(static_cast<float>(input_data[i]));
                    if (abs_val > norm_value) norm_value = abs_val;
                }
            } else {
                norm_value = scaled_lp(0.0f, [&](int64_t i) { return std::abs(static_cast<float>(input_data[i])); });
            }
            output_data[0] = norm_value;
            break;
        }
        default:
            throw std::runtime_error("norm: unsupported dtype");
    }

    return output;
}

// any() reduction - returns true if any element is nonzero
auto any_kernel(const Tensor& input_raw, int64_t dim, bool keepdim) -> Tensor {
    // The REDUCE_ALL branch iterates input.data<T>()[i] flatly, which is only
    // valid for a contiguous tensor. Materialize once at entry so both the
    // flat full-reduction path and the strided per-dim path read logical
    // elements of a non-contiguous view correctly.
    auto input = input_raw.contiguous();
    const auto& input_shape = input.shape();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());
    dim = normalize_dim(dim, ndim);

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, DType::Bool, input.device());
    auto* out = output.data<bool>();

    if (dim == REDUCE_ALL) {
        const int64_t n = input.numel();
        bool found = false;
        switch (input.dtype()) {
            case DType::Float32: {
                auto* d = input.data<float>();
                #pragma omp parallel for reduction(||:found) if(n > REDUCTION_OMP_THRESHOLD)
                for (int64_t i = 0; i < n; i++) found = found || (d[i] != 0.0f);
                break;
            }
            case DType::Float64: {
                auto* d = input.data<double>();
                #pragma omp parallel for reduction(||:found) if(n > REDUCTION_OMP_THRESHOLD)
                for (int64_t i = 0; i < n; i++) found = found || (d[i] != 0.0);
                break;
            }
            case DType::Int32: {
                auto* d = input.data<int32_t>();
                #pragma omp parallel for reduction(||:found) if(n > REDUCTION_OMP_THRESHOLD)
                for (int64_t i = 0; i < n; i++) found = found || (d[i] != 0);
                break;
            }
            case DType::Int64: {
                auto* d = input.data<int64_t>();
                #pragma omp parallel for reduction(||:found) if(n > REDUCTION_OMP_THRESHOLD)
                for (int64_t i = 0; i < n; i++) found = found || (d[i] != 0);
                break;
            }
            case DType::Bool: {
                auto* d = input.data<bool>();
                for (int64_t i = 0; i < n; i++) { if (d[i]) { found = true; break; } }
                break;
            }
            case DType::Float16: {
                auto* d = input.data<Float16>();
                for (int64_t i = 0; i < n; i++) {
                    if (static_cast<float>(d[i]) != 0.0f) { found = true; break; }
                }
                break;
            }
            case DType::BFloat16: {
                auto* d = input.data<BFloat16>();
                for (int64_t i = 0; i < n; i++) {
                    if (static_cast<float>(d[i]) != 0.0f) { found = true; break; }
                }
                break;
            }
            case DType::Int8: {
                auto* d = input.data<int8_t>();
                for (int64_t i = 0; i < n; i++) { if (d[i]) { found = true; break; } }
                break;
            }
            case DType::UInt8: {
                auto* d = input.data<uint8_t>();
                for (int64_t i = 0; i < n; i++) { if (d[i]) { found = true; break; } }
                break;
            }
            case DType::Int16: {
                auto* d = input.data<int16_t>();
                for (int64_t i = 0; i < n; i++) { if (d[i]) { found = true; break; } }
                break;
            }
            default:
                throw std::runtime_error("any: unsupported dtype");
        }
        out[0] = found;
    } else {
        const int64_t dim_size = input_shape[dim];
        int64_t output_size = output.numel();
        const int64_t shape_ndim = ndim;

        #pragma omp parallel for if(output_size > ::tenzor::OmpThresholds::matmul())
        for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
            // Use stack allocation for common case (<=16 dims), heap for rare large dims
            std::vector<int64_t> indices_vec;
            int64_t indices_stack[16] = {};
            int64_t* indices = (shape_ndim <= 16) ? indices_stack : (indices_vec.resize(shape_ndim, 0), indices_vec.data());
            int64_t tmp = out_idx;
            for (int64_t d = shape_ndim - 1; d >= 0; --d) {
                if (d == dim) continue;
                indices[d] = tmp % input_shape[d];
                tmp /= input_shape[d];
            }

            bool found = false;
            for (int64_t i = 0; i < dim_size && !found; i++) {
                indices[dim] = i;
                int64_t flat = 0;
                for (int64_t d = 0; d < shape_ndim; d++) {
                    flat += indices[d] * input.strides()[d];
                }
                switch (input.dtype()) {
                    case DType::Float32: found = input.data<float>()[flat] != 0.0f; break;
                    case DType::Float64: found = input.data<double>()[flat] != 0.0; break;
                    case DType::Float16:  found = static_cast<float>(input.data<Float16>()[flat]) != 0.0f; break;
                    case DType::BFloat16: found = static_cast<float>(input.data<BFloat16>()[flat]) != 0.0f; break;
                    case DType::Int8:  found = input.data<int8_t>()[flat] != 0; break;
                    case DType::Int16: found = input.data<int16_t>()[flat] != 0; break;
                    case DType::Int32: found = input.data<int32_t>()[flat] != 0; break;
                    case DType::Int64: found = input.data<int64_t>()[flat] != 0; break;
                    case DType::UInt8: found = input.data<uint8_t>()[flat] != 0; break;
                    case DType::Bool: found = input.data<bool>()[flat]; break;
                    default: break;
                }
            }
            out[out_idx] = found;
        }
    }
    return output;
}

// all() reduction - returns true if all elements are nonzero
auto all_kernel(const Tensor& input_raw, int64_t dim, bool keepdim) -> Tensor {
    // The REDUCE_ALL branch iterates input.data<T>()[i] flatly, which is only
    // valid for a contiguous tensor. Materialize once at entry so both the
    // flat full-reduction path and the strided per-dim path read logical
    // elements of a non-contiguous view correctly.
    auto input = input_raw.contiguous();
    const auto& input_shape = input.shape();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());
    dim = normalize_dim(dim, ndim);

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, DType::Bool, input.device());
    auto* out = output.data<bool>();

    if (dim == REDUCE_ALL) {
        const int64_t n = input.numel();
        bool result = true;
        switch (input.dtype()) {
            case DType::Float32: {
                auto* d = input.data<float>();
                #pragma omp parallel for reduction(&&:result) if(n > REDUCTION_OMP_THRESHOLD)
                for (int64_t i = 0; i < n; i++) result = result && (d[i] != 0.0f);
                break;
            }
            case DType::Float64: {
                auto* d = input.data<double>();
                #pragma omp parallel for reduction(&&:result) if(n > REDUCTION_OMP_THRESHOLD)
                for (int64_t i = 0; i < n; i++) result = result && (d[i] != 0.0);
                break;
            }
            case DType::Int32: {
                auto* d = input.data<int32_t>();
                #pragma omp parallel for reduction(&&:result) if(n > REDUCTION_OMP_THRESHOLD)
                for (int64_t i = 0; i < n; i++) result = result && (d[i] != 0);
                break;
            }
            case DType::Int64: {
                auto* d = input.data<int64_t>();
                #pragma omp parallel for reduction(&&:result) if(n > REDUCTION_OMP_THRESHOLD)
                for (int64_t i = 0; i < n; i++) result = result && (d[i] != 0);
                break;
            }
            case DType::Bool: {
                auto* d = input.data<bool>();
                for (int64_t i = 0; i < n; i++) { if (!d[i]) { result = false; break; } }
                break;
            }
            case DType::Float16: {
                auto* d = input.data<Float16>();
                for (int64_t i = 0; i < n; i++) {
                    if (static_cast<float>(d[i]) == 0.0f) { result = false; break; }
                }
                break;
            }
            case DType::BFloat16: {
                auto* d = input.data<BFloat16>();
                for (int64_t i = 0; i < n; i++) {
                    if (static_cast<float>(d[i]) == 0.0f) { result = false; break; }
                }
                break;
            }
            case DType::Int8: {
                auto* d = input.data<int8_t>();
                for (int64_t i = 0; i < n; i++) { if (!d[i]) { result = false; break; } }
                break;
            }
            case DType::UInt8: {
                auto* d = input.data<uint8_t>();
                for (int64_t i = 0; i < n; i++) { if (!d[i]) { result = false; break; } }
                break;
            }
            case DType::Int16: {
                auto* d = input.data<int16_t>();
                for (int64_t i = 0; i < n; i++) { if (!d[i]) { result = false; break; } }
                break;
            }
            default:
                throw std::runtime_error("all: unsupported dtype");
        }
        out[0] = result;
    } else {
        const int64_t dim_size = input_shape[dim];
        int64_t output_size = output.numel();
        const int64_t shape_ndim = ndim;

        #pragma omp parallel for if(output_size > ::tenzor::OmpThresholds::matmul())
        for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
            // Use stack allocation for common case (<=16 dims), heap for rare large dims
            std::vector<int64_t> indices_vec;
            int64_t indices_stack[16] = {};
            int64_t* indices = (shape_ndim <= 16) ? indices_stack : (indices_vec.resize(shape_ndim, 0), indices_vec.data());
            int64_t tmp = out_idx;
            for (int64_t d = shape_ndim - 1; d >= 0; --d) {
                if (d == dim) continue;
                indices[d] = tmp % input_shape[d];
                tmp /= input_shape[d];
            }

            bool result = true;
            for (int64_t i = 0; i < dim_size && result; i++) {
                indices[dim] = i;
                int64_t flat = 0;
                for (int64_t d = 0; d < shape_ndim; d++) {
                    flat += indices[d] * input.strides()[d];
                }
                switch (input.dtype()) {
                    case DType::Float32: result = input.data<float>()[flat] != 0.0f; break;
                    case DType::Float64: result = input.data<double>()[flat] != 0.0; break;
                    case DType::Float16:  result = static_cast<float>(input.data<Float16>()[flat]) != 0.0f; break;
                    case DType::BFloat16: result = static_cast<float>(input.data<BFloat16>()[flat]) != 0.0f; break;
                    case DType::Int8:  result = input.data<int8_t>()[flat] != 0; break;
                    case DType::Int16: result = input.data<int16_t>()[flat] != 0; break;
                    case DType::Int32: result = input.data<int32_t>()[flat] != 0; break;
                    case DType::Int64: result = input.data<int64_t>()[flat] != 0; break;
                    case DType::UInt8: result = input.data<uint8_t>()[flat] != 0; break;
                    case DType::Bool: result = input.data<bool>()[flat]; break;
                    default: break;
                }
            }
            out[out_idx] = result;
        }
    }
    return output;
}

// has_inf_nan() reduction - returns true if any element is inf or nan
auto has_inf_nan_kernel(const Tensor& input, int64_t /*dim*/, bool /*keepdim*/) -> Tensor {
    const int64_t numel = input.numel();

    // Empty tensor has no inf/nan
    if (numel == 0) {
        Tensor result({}, DType::Bool, input.device());
        result.data<bool>()[0] = false;
        return result;
    }

    bool found = false;

    // Handle BFloat16/Float16 by converting to Float32 first
    Tensor scan = input;
    if (scan.dtype() == DType::BFloat16 || scan.dtype() == DType::Float16) {
        scan = scan.to(DType::Float32);
    }

    switch (scan.dtype()) {
        case DType::Float32: {
            const auto* data = scan.data<float>();
            #pragma omp parallel for reduction(||:found) if(numel > REDUCTION_OMP_THRESHOLD)
            for (int64_t i = 0; i < numel; ++i) {
                if (std::isinf(data[i]) || std::isnan(data[i])) {
                    found = true;
                }
            }
            break;
        }
        case DType::Float64: {
            const auto* data = scan.data<double>();
            #pragma omp parallel for reduction(||:found) if(numel > REDUCTION_OMP_THRESHOLD)
            for (int64_t i = 0; i < numel; ++i) {
                if (std::isinf(data[i]) || std::isnan(data[i])) {
                    found = true;
                }
            }
            break;
        }
        default:
            // Integer types don't have inf/nan
            break;
    }

    Tensor result({}, DType::Bool, input.device());
    result.data<bool>()[0] = found;
    return result;
}

// ============================================================================
// LogSumExp - Numerically stable log(sum(exp(x)))
// Algorithm: result = max + log(sum(exp(x - max)))
// Uses Kahan summation for the exp-sum accumulation
// ============================================================================

template<typename T, typename Acc = T>
static Acc logsumexp_full_impl(const T* data, int64_t n) {
    if (n == 0) return Acc(-std::numeric_limits<Acc>::infinity());

    // Step 1: Find max
    Acc m = Acc(data[0]);
    for (int64_t i = 1; i < n; i++) {
        Acc val = Acc(data[i]);
        if (val > m) m = val;
    }

    // Handle inf
    if (std::isinf(m)) return m;

    // Step 2: Sum exp(x - max) with Kahan summation
    Acc sum = Acc(0);
    Acc compensation = Acc(0);
    for (int64_t i = 0; i < n; i++) {
        Acc y = std::exp(Acc(data[i]) - m) - compensation;
        Acc t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    return m + std::log(sum);
}

template<typename T, typename Acc = T>
static void logsumexp_along_dim(
    const T* input_data,
    T* output_data,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim
) {
    const int64_t ndim = static_cast<int64_t>(input_shape.size());
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) output_size *= input_shape[i];
    }

    const int64_t total_work = output_size * dim_size;
    #pragma omp parallel for if(total_work > REDUCTION_OMP_THRESHOLD)
    for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
        std::vector<int64_t> indices(ndim, 0);
        int64_t tmp = out_idx;

        for (int64_t d = ndim - 1; d >= 0; --d) {
            if (d == dim) continue;
            indices[d] = tmp % input_shape[d];
            tmp /= input_shape[d];
        }

        // Step 1: Find max along dim
        Acc m = Acc(-std::numeric_limits<Acc>::infinity());
        for (int64_t i = 0; i < dim_size; i++) {
            indices[dim] = i;
            int64_t in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * input_strides[d];
            }
            Acc val = Acc(input_data[in_idx]);
            if (val > m) m = val;
        }

        // Handle inf
        if (std::isinf(m)) {  // test on the accumulator type; casting double->float spuriously overflows finite large Float64
            output_data[out_idx] = T(m);
            continue;
        }

        // Step 2: Sum exp(x - max) with Kahan summation
        Acc sum = Acc(0);
        Acc compensation = Acc(0);
        for (int64_t i = 0; i < dim_size; i++) {
            indices[dim] = i;
            int64_t in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * input_strides[d];
            }
            Acc y = std::exp(Acc(input_data[in_idx]) - m) - compensation;
            Acc t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }

        output_data[out_idx] = T(m + std::log(sum));
    }
}

auto logsumexp_kernel(const Tensor& input_raw, int64_t dim, bool keepdim) -> Tensor {
    // The REDUCE_ALL path iterates input.data<T>()[i] flatly (logsumexp_full_impl
    // and the Float16/BFloat16 inline loops), which is only valid for a contiguous
    // tensor. Materialize once at entry so both the flat full-reduction path and
    // the strided per-dim path read logical elements of a non-contiguous view.
    auto input = input_raw.contiguous();
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    if (dtype != DType::Float32 && dtype != DType::Float64 &&
        dtype != DType::Float16 && dtype != DType::BFloat16) {
        throw std::runtime_error("logsumexp: only Float32, Float64, Float16, and BFloat16 are supported");
    }

    dim = normalize_dim(dim, ndim);

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float16: {
            auto* input_data = input.data<Float16>();
            auto* output_data = output.data<Float16>();

            if (dim == REDUCE_ALL) {
                // Upcast to float for computation
                const int64_t n = input.numel();
                if (n == 0) {
                    output_data[0] = Float16(-std::numeric_limits<float>::infinity());
                } else {
                    // Find max
                    float m = static_cast<float>(input_data[0]);
                    for (int64_t i = 1; i < n; i++) {
                        float val = static_cast<float>(input_data[i]);
                        if (val > m) m = val;
                    }
                    if (std::isinf(m)) {
                        output_data[0] = Float16(m);
                    } else {
                        float sum = 0.0f, compensation = 0.0f;
                        for (int64_t i = 0; i < n; i++) {
                            float y = std::exp(static_cast<float>(input_data[i]) - m) - compensation;
                            float t = sum + y;
                            compensation = (t - sum) - y;
                            sum = t;
                        }
                        output_data[0] = Float16(m + std::log(sum));
                    }
                }
            } else {
                logsumexp_along_dim<Float16, float>(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim);
            }
            break;
        }
        case DType::BFloat16: {
            auto* input_data = input.data<BFloat16>();
            auto* output_data = output.data<BFloat16>();

            if (dim == REDUCE_ALL) {
                const int64_t n = input.numel();
                if (n == 0) {
                    output_data[0] = BFloat16(-std::numeric_limits<float>::infinity());
                } else {
                    float m = static_cast<float>(input_data[0]);
                    for (int64_t i = 1; i < n; i++) {
                        float val = static_cast<float>(input_data[i]);
                        if (val > m) m = val;
                    }
                    if (std::isinf(m)) {
                        output_data[0] = BFloat16(m);
                    } else {
                        float sum = 0.0f, compensation = 0.0f;
                        for (int64_t i = 0; i < n; i++) {
                            float y = std::exp(static_cast<float>(input_data[i]) - m) - compensation;
                            float t = sum + y;
                            compensation = (t - sum) - y;
                            sum = t;
                        }
                        output_data[0] = BFloat16(m + std::log(sum));
                    }
                }
            } else {
                logsumexp_along_dim<BFloat16, float>(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim);
            }
            break;
        }
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            if (dim == REDUCE_ALL) {
                output_data[0] = logsumexp_full_impl<float>(input_data, input.numel());
            } else {
                logsumexp_along_dim<float>(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim);
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (dim == REDUCE_ALL) {
                output_data[0] = logsumexp_full_impl<double>(input_data, input.numel());
            } else {
                logsumexp_along_dim<double>(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim);
            }
            break;
        }
        default:
            throw std::runtime_error("logsumexp: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Median kernel — O(n) via nth_element
// ============================================================================

auto median_kernel(const Tensor& input, int64_t dim, bool keepdim) -> std::vector<Tensor> {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    dim = normalize_dim(dim, ndim);

    // median has no identity element: it selects an actual element, which does
    // not exist for an empty reduction. Guard BOTH the full-reduction (numel==0)
    // and per-dim (dim_size==0) empty cases here at kernel entry — BEFORE any
    // OpenMP region — because the reduction reads elems[mid] out of bounds on an
    // empty slice.
    if (dim == REDUCE_ALL) {
        if (input.numel() == 0)
            throw std::invalid_argument("median: cannot reduce over a zero-size dimension");
    } else if (input_shape[dim] == 0) {
        throw std::invalid_argument("median: cannot reduce over a zero-size dimension");
    }

    // Float16/BFloat16: no native comparison path. Widen to Float32, compute,
    // then narrow the value tensor back. median selects an actual element, so
    // the Float32->half cast reproduces the original half value exactly; the
    // index tensor is dtype-independent (Int64).
    if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        auto widened = median_kernel(input.to(DType::Float32), dim, keepdim);
        return {widened[0].to(dtype), widened[1]};
    }

    // For full reduction, flatten to 1D then reduce along dim 0
    if (dim == REDUCE_ALL) {
        Tensor flat = input.contiguous().reshape({input.numel()});
        auto result = median_kernel(flat, 0, false);
        if (keepdim) {
            std::vector<int64_t> kshape(ndim, 1);
            result[0] = result[0].reshape(kshape);
            result[1] = result[1].reshape(kshape);
        }
        return result;
    }

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    // Shape without the reduced dim (for indexing)
    std::vector<int64_t> reduced_shape;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != dim) reduced_shape.push_back(input_shape[d]);
    }

    int64_t dim_size = input_shape[dim];
    int64_t mid = (dim_size - 1) / 2; // lower median for even sizes

    int64_t outer_size = 1;
    for (auto s : reduced_shape) outer_size *= s;

    Tensor values(output_shape, dtype, device);
    Tensor indices(output_shape, DType::Int64, device);

    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();

    auto process = [&]<typename T>(T*) {
        const T* in_data = input_cont.data<T>();
        T* val_data = values.data<T>();
        int64_t* idx_data = indices.data<int64_t>();

        // Compute strides for the input
        std::vector<int64_t> in_strides(ndim);
        int64_t stride = 1;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            in_strides[d] = stride;
            stride *= input_shape[d];
        }

        #pragma omp parallel for if(outer_size > REDUCTION_OMP_THRESHOLD)
        for (int64_t o = 0; o < outer_size; ++o) {
            // Compute multi-dimensional index for output position
            std::vector<int64_t> out_idx(ndim, 0);
            int64_t tmp = o;
            for (int64_t d = ndim - 1; d >= 0; --d) {
                if (d == dim) continue;
                out_idx[d] = tmp % input_shape[d];
                tmp /= input_shape[d];
            }

            // Build array of (value, original_index) pairs along dim
            std::vector<std::pair<T, int64_t>> elems(dim_size);
            for (int64_t i = 0; i < dim_size; ++i) {
                out_idx[dim] = i;
                int64_t flat_idx = 0;
                for (int64_t d = 0; d < ndim; ++d) {
                    flat_idx += out_idx[d] * in_strides[d];
                }
                elems[i] = {in_data[flat_idx], i};
            }

            // nth_element for O(n) median, with an index tie-break so a duplicated
            // median value returns the lowest original index deterministically
            // (matching the CUDA median, which takes sorted_idx[mid] from a stable sort).
            // F065: NaN-aware selection (NaN is the LARGEST value → sorts last,
            // matching the CUDA median's sort and the sort/argsort/kthvalue paths).
            std::nth_element(elems.begin(), elems.begin() + mid, elems.end(),
                [](const auto& a, const auto& b) {
                    return as_nan_lt(a.first, b.first) || (as_nan_eq(a.first, b.first) && a.second < b.second);
                });

            val_data[o] = elems[mid].first;
            idx_data[o] = elems[mid].second;
        }
    };

    switch (dtype) {
        case DType::Float32: process(static_cast<float*>(nullptr)); break;
        case DType::Float64: process(static_cast<double*>(nullptr)); break;
        case DType::Int32:   process(static_cast<int32_t*>(nullptr)); break;
        case DType::Int64:   process(static_cast<int64_t*>(nullptr)); break;
        default: throw std::runtime_error("median_kernel: unsupported dtype");
    }

    return {values, indices};
}

// ============================================================================
// Mode kernel — sort then find longest run
// ============================================================================

auto mode_kernel(const Tensor& input, int64_t dim, bool keepdim) -> std::vector<Tensor> {
    // Float16/BFloat16: widen to Float32 (exact for half values, so value
    // groupings/counts are preserved), compute mode, then narrow the VALUES back;
    // indices are dtype-independent. Matches the CUDA backend's launch_half path.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto res = mode_kernel(input.to(DType::Float32), dim, keepdim);
        res[0] = res[0].to(input.dtype());
        return res;
    }
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    dim = normalize_dim(dim, ndim);

    // mode has no identity element: it selects an actual element, which does not
    // exist for an empty reduction. Guard BOTH the full-reduction (numel==0) and
    // per-dim (dim_size==0) empty cases here at kernel entry — BEFORE any OpenMP
    // region — because the reduction seeds from elems[0] on an empty slice.
    if (dim == REDUCE_ALL) {
        if (input.numel() == 0)
            throw std::invalid_argument("mode: cannot reduce over a zero-size dimension");
    } else if (input_shape[dim] == 0) {
        throw std::invalid_argument("mode: cannot reduce over a zero-size dimension");
    }

    // For full reduction, flatten to 1D then reduce along dim 0
    if (dim == REDUCE_ALL) {
        Tensor flat = input.contiguous().reshape({input.numel()});
        auto result = mode_kernel(flat, 0, false);
        if (keepdim) {
            std::vector<int64_t> kshape(ndim, 1);
            result[0] = result[0].reshape(kshape);
            result[1] = result[1].reshape(kshape);
        }
        return result;
    }

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    std::vector<int64_t> reduced_shape;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != dim) reduced_shape.push_back(input_shape[d]);
    }

    int64_t dim_size = input_shape[dim];
    int64_t outer_size = 1;
    for (auto s : reduced_shape) outer_size *= s;

    Tensor values(output_shape, dtype, device);
    Tensor indices(output_shape, DType::Int64, device);

    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();

    auto process = [&]<typename T>(T*) {
        const T* in_data = input_cont.data<T>();
        T* val_data = values.data<T>();
        int64_t* idx_data = indices.data<int64_t>();

        std::vector<int64_t> in_strides(ndim);
        int64_t stride = 1;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            in_strides[d] = stride;
            stride *= input_shape[d];
        }

        #pragma omp parallel for if(outer_size > REDUCTION_OMP_THRESHOLD)
        for (int64_t o = 0; o < outer_size; ++o) {
            std::vector<int64_t> out_idx(ndim, 0);
            int64_t tmp = o;
            for (int64_t d = ndim - 1; d >= 0; --d) {
                if (d == dim) continue;
                out_idx[d] = tmp % input_shape[d];
                tmp /= input_shape[d];
            }

            // Collect (value, original_index) pairs along dim
            std::vector<std::pair<T, int64_t>> elems(dim_size);
            for (int64_t i = 0; i < dim_size; ++i) {
                out_idx[dim] = i;
                int64_t flat_idx = 0;
                for (int64_t d = 0; d < ndim; ++d) {
                    flat_idx += out_idx[d] * in_strides[d];
                }
                elems[i] = {in_data[flat_idx], i};
            }

            // Sort by value with an index tie-break (ascending index within each
            // equal-value run) so the "last element of the longest run" is the
            // highest original index deterministically — matching the CUDA mode,
            // which uses a stable sort.
            std::sort(elems.begin(), elems.end(),
                [](const auto& a, const auto& b) {
                    return a.first < b.first || (a.first == b.first && a.second < b.second);
                });

            T best_val = elems[0].first;
            int64_t best_idx = elems[0].second;
            int64_t best_count = 1;
            int64_t cur_count = 1;

            for (int64_t i = 1; i < dim_size; ++i) {
                if (elems[i].first == elems[i - 1].first) {
                    cur_count++;
                } else {
                    cur_count = 1;
                }
                if (cur_count > best_count) {
                    best_count = cur_count;
                    best_val = elems[i].first;
                    best_idx = elems[i].second;
                }
            }

            val_data[o] = best_val;
            idx_data[o] = best_idx;
        }
    };

    switch (dtype) {
        case DType::Float32: process(static_cast<float*>(nullptr)); break;
        case DType::Float64: process(static_cast<double*>(nullptr)); break;
        case DType::Int32:   process(static_cast<int32_t*>(nullptr)); break;
        case DType::Int64:   process(static_cast<int64_t*>(nullptr)); break;
        default: throw std::runtime_error("mode_kernel: unsupported dtype");
    }

    return {values, indices};
}

// ============================================================================
// Histogram Kernel - Compute histogram of tensor values
// ============================================================================

auto histogram_kernel(const Tensor& input, int64_t bins, double min_val, double max_val)
    -> std::pair<Tensor, Tensor> {
    if (bins <= 0) {
        throw std::runtime_error("histogram: bins must be > 0");
    }

    // Compute at the input's native float precision. Previously everything was
    // forced to Float32, which truncated bin assignment for Float64 tensors
    // (histogramdd already keeps Float64 — this makes histogram consistent).
    const DType ctype = (input.dtype() == DType::Float64) ? DType::Float64 : DType::Float32;
    // input.to() already returns a contiguous tensor; when no dtype conversion is
    // needed we must still contiguify, otherwise the flat data[i] iteration below
    // would walk the storage footprint of a non-contiguous view (transpose/slice).
    Tensor in_c = (input.dtype() == ctype) ? input.contiguous() : input.to(ctype);
    const int64_t n = in_c.numel();

    Tensor edges({bins + 1}, ctype, input.device());
    Tensor counts({bins}, DType::Int64, input.device());
    int64_t* count_data = counts.data<int64_t>();
    std::memset(count_data, 0, static_cast<size_t>(bins) * sizeof(int64_t));

    auto run = [&](auto* tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const T* data = in_c.data<T>();
        T tmin = static_cast<T>(min_val);
        T tmax = static_cast<T>(max_val);
        if (tmin >= tmax) {  // auto-detect range
            if (n == 0) {
                tmin = T(0); tmax = T(1);
            } else {
                tmin = data[0]; tmax = data[0];
                for (int64_t i = 1; i < n; ++i) {
                    if (data[i] < tmin) tmin = data[i];
                    if (data[i] > tmax) tmax = data[i];
                }
                if (tmin == tmax) { tmin -= T(0.5); tmax += T(0.5); }
            }
        }
        T* edge_data = edges.data<T>();
        T step = (tmax - tmin) / static_cast<T>(bins);
        for (int64_t i = 0; i <= bins; ++i) {
            edge_data[i] = tmin + static_cast<T>(i) * step;
        }
        for (int64_t i = 0; i < n; ++i) {
            T v = data[i];
            if (v < tmin || v > tmax) continue;
            int64_t bin = static_cast<int64_t>((v - tmin) / step);
            // Last bin is closed on the right: [edge[bins-1], edge[bins]].
            if (bin >= bins) bin = bins - 1;
            if (bin < 0) bin = 0;
            count_data[bin]++;
        }
    };
    if (ctype == DType::Float64) run(static_cast<double*>(nullptr));
    else                        run(static_cast<float*>(nullptr));

    return {counts, edges};
}

// ============================================================================
// Histogramdd Kernel - Multi-dimensional histogram
// ============================================================================

auto histogramdd_kernel(const Tensor& input, std::vector<int64_t> bins,
                        std::vector<std::pair<double,double>> ranges,
                        bool density)
    -> std::pair<Tensor, std::vector<Tensor>> {

    if (input.dim() != 2) {
        throw std::runtime_error("histogramdd_kernel: input must be 2-D (N, D)");
    }

    const int64_t N = input.shape()[0];
    const int64_t D = input.shape()[1];

    if (static_cast<int64_t>(bins.size()) != D) {
        throw std::runtime_error("histogramdd_kernel: bins length must equal D");
    }

    // Determine compute dtype — work in Float64 for precision, or Float32
    const auto orig_dtype = input.dtype();
    const bool use_f64 = (orig_dtype == DType::Float64);
    const auto compute_dtype = use_f64 ? DType::Float64 : DType::Float32;
    // input.to() already returns a contiguous tensor; when no dtype conversion is
    // needed we must still contiguify, otherwise the data[i*D+d] row-major indexing
    // below would read the wrong elements of a non-contiguous (N,D) view.
    Tensor inp = (input.dtype() != compute_dtype) ? input.to(compute_dtype) : input.contiguous();
    const auto& device = input.device();

    // Auto-detect ranges from data if not provided
    bool auto_range = ranges.empty();
    if (auto_range) {
        ranges.resize(static_cast<size_t>(D));
    }

    auto process = [&]<typename T>(T*) {
        const T* data = inp.data<T>();

        if (auto_range) {
            for (int64_t d = 0; d < D; ++d) {
                if (N == 0) {
                    ranges[static_cast<size_t>(d)] = {0.0, 1.0};
                } else {
                    T vmin = data[0 * D + d];
                    T vmax = data[0 * D + d];
                    for (int64_t i = 1; i < N; ++i) {
                        T v = data[i * D + d];
                        if (v < vmin) vmin = v;
                        if (v > vmax) vmax = v;
                    }
                    if (vmin == vmax) {
                        vmin -= static_cast<T>(0.5);
                        vmax += static_cast<T>(0.5);
                    }
                    ranges[static_cast<size_t>(d)] = {static_cast<double>(vmin),
                                                       static_cast<double>(vmax)};
                }
            }
        }

        // Build edge tensors per dimension
        std::vector<Tensor> edges_vec;
        edges_vec.reserve(static_cast<size_t>(D));
        std::vector<T> dim_min(static_cast<size_t>(D));
        std::vector<T> dim_step(static_cast<size_t>(D));
        std::vector<T> dim_max(static_cast<size_t>(D));

        for (int64_t d = 0; d < D; ++d) {
            auto sd = static_cast<size_t>(d);
            int64_t nb = bins[sd];
            T fmin = static_cast<T>(ranges[sd].first);
            T fmax = static_cast<T>(ranges[sd].second);
            // A caller-supplied degenerate range (fmin >= fmax) yields step<=0, so
            // (v - fmin)/step at bin time is 0/0 = NaN (F072). The auto-range path
            // already widens an equal-bounds interval by ±0.5 (:vmin/vmax fixup
            // above); reuse that logic here so step stays strictly positive.
            if (fmin >= fmax) {
                fmin -= static_cast<T>(0.5);
                fmax += static_cast<T>(0.5);
            }
            T step = (fmax - fmin) / static_cast<T>(nb);
            dim_min[sd] = fmin;
            dim_step[sd] = step;

            Tensor edge({nb + 1}, compute_dtype, device);
            T* edata = edge.data<T>();
            for (int64_t i = 0; i <= nb; ++i) {
                edata[i] = fmin + static_cast<T>(i) * step;
            }
            // Record the actual computed upper edge so the in-range test matches
            // the edges that were emitted. Recomputing fmin + nb*step at test time
            // is not bit-identical to edata[nb] (and neither equals fmax exactly
            // once step is rounded), which could drop a sample equal to the max.
            dim_max[sd] = edata[nb];
            edges_vec.push_back(std::move(edge));
        }

        // Compute strides for the output (row-major)
        std::vector<int64_t> out_shape(bins.begin(), bins.end());
        std::vector<int64_t> out_strides(static_cast<size_t>(D));
        int64_t stride = 1;
        for (int64_t d = D - 1; d >= 0; --d) {
            out_strides[static_cast<size_t>(d)] = stride;
            stride *= bins[static_cast<size_t>(d)];
        }
        int64_t total_bins = stride;

        // Allocate counts
        Tensor counts(out_shape, DType::Int64, device);
        int64_t* count_data = counts.data<int64_t>();
        std::memset(count_data, 0, static_cast<size_t>(total_bins) * sizeof(int64_t));

        // Bin each sample
        for (int64_t i = 0; i < N; ++i) {
            int64_t flat = 0;
            bool in_range = true;
            for (int64_t d = 0; d < D; ++d) {
                auto sd = static_cast<size_t>(d);
                T v = data[i * D + d];
                T fmin_d = dim_min[sd];
                T step_d = dim_step[sd];
                int64_t nb = bins[sd];

                // A non-finite sample (NaN/±inf) compares false against both range
                // bounds, so without this guard it would slip through and
                // static_cast<int64_t>(NaN) is UB (F072). Treat it as out of range.
                if (!std::isfinite(static_cast<double>(v)) ||
                    v < fmin_d || v > dim_max[sd]) {
                    in_range = false;
                    break;
                }

                int64_t b = static_cast<int64_t>((v - fmin_d) / step_d);
                if (b >= nb) b = nb - 1;
                if (b < 0) b = 0;
                flat += b * out_strides[sd];
            }
            if (in_range) {
                count_data[flat]++;
            }
        }

        // Density normalization: counts / (N * bin_volume)
        Tensor result = counts;
        if (density && N > 0) {
            // Compute bin volume (product of all bin widths)
            double bin_volume = 1.0;
            for (int64_t d = 0; d < D; ++d) {
                bin_volume *= static_cast<double>(dim_step[static_cast<size_t>(d)]);
            }
            double norm = static_cast<double>(N) * bin_volume;

            Tensor density_out(out_shape, compute_dtype, device);
            T* ddata = density_out.data<T>();
            for (int64_t i = 0; i < total_bins; ++i) {
                ddata[i] = static_cast<T>(static_cast<double>(count_data[i]) / norm);
            }
            result = density_out;
        }

        return std::make_pair(std::move(result), std::move(edges_vec));
    };

    if (use_f64) {
        return process(static_cast<double*>(nullptr));
    } else {
        return process(static_cast<float*>(nullptr));
    }
}

// ============================================================================
// logcumsumexp — Log-Cumulative-Sum-Exp (numerically stable)
// ============================================================================

auto logcumsumexp_kernel(const Tensor& input, int64_t dim) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& shape = input.shape();
    const int64_t ndim = static_cast<int64_t>(shape.size());

    // Float16/BFloat16: widen to Float32, compute, narrow the result back
    // (matches the CUDA backend, which upcasts these dtypes).
    if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        return logcumsumexp_kernel(input.to(DType::Float32), dim).to(dtype);
    }
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("logcumsumexp_kernel: only Float32, Float64, Float16, BFloat16 supported");
    }

    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("logcumsumexp_kernel: dimension out of range");
    }

    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), dtype, device);

    const int64_t dim_size = shape[dim];

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    auto process = [&]<typename T>(T*) {
        const T* in = input_cont.data<T>();
        T* out = output.data<T>();

        #ifdef _OPENMP
        #pragma omp parallel for if (outer_size * inner_size > 64)
        #endif
        for (int64_t oi = 0; oi < outer_size * inner_size; ++oi) {
            int64_t outer = oi / inner_size;
            int64_t inner = oi % inner_size;

            T running_max = -std::numeric_limits<T>::infinity();
            T running_lse = -std::numeric_limits<T>::infinity();

            for (int64_t i = 0; i < dim_size; ++i) {
                int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
                T x = in[offset];
                T new_max = std::max(running_max, x);

                if (std::isinf(new_max) && new_max < T(0)) {
                    running_lse = -std::numeric_limits<T>::infinity();
                } else {
                    running_lse = new_max + std::log(
                        std::exp(running_lse - new_max) + std::exp(x - new_max));
                }
                running_max = new_max;
                out[offset] = running_lse;
            }
        }
    };

    if (dtype == DType::Float32) {
        process(static_cast<float*>(nullptr));
    } else {
        process(static_cast<double*>(nullptr));
    }

    return output;
}

} // namespace cpu
} // namespace tenzor
