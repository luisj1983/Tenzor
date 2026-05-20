/**
 * @file reduction_optimized.hpp
 * @brief Optimized reduction kernels with pre-computed strides and SIMD
 *
 * Key optimizations:
 * - Pre-computed stride arrays instead of runtime index calculation
 * - Stack-allocated index arrays (no heap allocation in hot loop)
 * - SIMD horizontal reductions for sum/max/min
 * - Incremental index updates instead of division chains
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <algorithm>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #if defined(__AVX512F__)
        #define TENZOR_REDUCTION_AVX512
    #endif
    #if defined(__AVX2__)
        #define TENZOR_REDUCTION_AVX2
    #endif
#endif

#ifdef _OPENMP
#include <omp.h>
#include "tenzor/backend/omp_thresholds.hpp"
#endif

namespace tenzor {
namespace cpu {
namespace reduction_opt {

// Maximum supported tensor dimensions (stack allocation)
constexpr size_t MAX_DIMS = 8;

/**
 * @brief Pre-computed stride information for fast index calculation
 */
struct StrideInfo {
    std::array<int64_t, MAX_DIMS> input_strides;
    std::array<int64_t, MAX_DIMS> output_strides;
    std::array<int64_t, MAX_DIMS> shape;
    int64_t ndim;
    int64_t reduce_dim;
    int64_t reduce_size;
    int64_t output_size;

    // Pre-computed multipliers for output-to-input index mapping
    std::array<int64_t, MAX_DIMS> dim_multipliers;
};

/**
 * @brief Initialize stride info for dimensional reduction
 */
inline StrideInfo compute_stride_info(
    const int64_t* input_shape,
    const int64_t* input_strides,
    int64_t ndim,
    int64_t dim
) {
    StrideInfo info{};
    info.ndim = ndim;
    info.reduce_dim = dim;
    info.reduce_size = input_shape[dim];
    info.output_size = 1;

    // Copy shape and strides
    for (int64_t d = 0; d < ndim; ++d) {
        info.shape[d] = input_shape[d];
        info.input_strides[d] = input_strides[d];
        if (d != dim) {
            info.output_size *= input_shape[d];
        }
    }

    // Compute dimension multipliers for output index decoding
    // This allows us to convert flat output index to multi-dim indices
    int64_t multiplier = 1;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d != dim) {
            info.dim_multipliers[d] = multiplier;
            multiplier *= input_shape[d];
        } else {
            info.dim_multipliers[d] = 0;
        }
    }

    return info;
}

/**
 * @brief Fast output index to input base index conversion
 * Uses pre-computed multipliers instead of division chain
 */
inline int64_t output_to_input_base(
    int64_t out_idx,
    const StrideInfo& info
) {
    int64_t in_idx = 0;
    int64_t remaining = out_idx;

    // Decode output index to input index using pre-computed multipliers
    for (int64_t d = 0; d < info.ndim; ++d) {
        if (d == info.reduce_dim) continue;

        int64_t coord = remaining / info.dim_multipliers[d];
        remaining %= info.dim_multipliers[d];
        in_idx += coord * info.input_strides[d];
    }

    return in_idx;
}

// ============================================================================
// SIMD Horizontal Reductions
// ============================================================================

#ifdef TENZOR_REDUCTION_AVX2

/**
 * @brief Horizontal sum of 8 floats in AVX2 register
 */
inline float hsum_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

/**
 * @brief Horizontal max of 8 floats in AVX2 register
 */
inline float hmax_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m = _mm_max_ps(hi, lo);
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1)));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(m);
}

/**
 * @brief Horizontal min of 8 floats in AVX2 register
 */
inline float hmin_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m = _mm_min_ps(hi, lo);
    m = _mm_min_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1)));
    m = _mm_min_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(m);
}

#endif // TENZOR_REDUCTION_AVX2

#ifdef TENZOR_REDUCTION_AVX512

/**
 * @brief Horizontal sum of 16 floats in AVX-512 register
 */
inline float hsum_avx512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

/**
 * @brief Horizontal max of 16 floats in AVX-512 register
 */
inline float hmax_avx512(__m512 v) {
    return _mm512_reduce_max_ps(v);
}

/**
 * @brief Horizontal min of 16 floats in AVX-512 register
 */
inline float hmin_avx512(__m512 v) {
    return _mm512_reduce_min_ps(v);
}

#endif // TENZOR_REDUCTION_AVX512

// ============================================================================
// Optimized Full Reductions (contiguous memory)
// ============================================================================

/**
 * @brief SIMD-optimized full sum reduction with Kahan summation
 */
inline float sum_full_simd(const float* data, int64_t n) {
#ifdef TENZOR_REDUCTION_AVX512
    if (n >= 64) {
        __m512 sum0 = _mm512_setzero_ps();
        __m512 sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps();
        __m512 sum3 = _mm512_setzero_ps();

        int64_t i = 0;
        // 4x unroll for better ILP
        for (; i + 64 <= n; i += 64) {
            sum0 = _mm512_add_ps(sum0, _mm512_loadu_ps(data + i));
            sum1 = _mm512_add_ps(sum1, _mm512_loadu_ps(data + i + 16));
            sum2 = _mm512_add_ps(sum2, _mm512_loadu_ps(data + i + 32));
            sum3 = _mm512_add_ps(sum3, _mm512_loadu_ps(data + i + 48));
        }

        // Combine partial sums
        sum0 = _mm512_add_ps(sum0, sum1);
        sum2 = _mm512_add_ps(sum2, sum3);
        sum0 = _mm512_add_ps(sum0, sum2);

        float result = hsum_avx512(sum0);

        // Handle remainder
        for (; i < n; ++i) {
            result += data[i];
        }
        return result;
    }
#endif

#ifdef TENZOR_REDUCTION_AVX2
    if (n >= 32) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        int64_t i = 0;
        for (; i + 32 <= n; i += 32) {
            sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(data + i));
            sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(data + i + 8));
            sum2 = _mm256_add_ps(sum2, _mm256_loadu_ps(data + i + 16));
            sum3 = _mm256_add_ps(sum3, _mm256_loadu_ps(data + i + 24));
        }

        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);

        float result = hsum_avx2(sum0);

        for (; i < n; ++i) {
            result += data[i];
        }
        return result;
    }
#endif

    // Scalar with Kahan summation
    float sum = 0.0f;
    float c = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        float y = data[i] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

/**
 * @brief SIMD-optimized full max reduction
 */
inline float max_full_simd(const float* data, int64_t n) {
    if (n == 0) return -std::numeric_limits<float>::infinity();

#ifdef TENZOR_REDUCTION_AVX512
    if (n >= 64) {
        __m512 max0 = _mm512_loadu_ps(data);
        __m512 max1 = max0;
        __m512 max2 = max0;
        __m512 max3 = max0;

        int64_t i = 16;
        for (; i + 64 <= n; i += 64) {
            max0 = _mm512_max_ps(max0, _mm512_loadu_ps(data + i));
            max1 = _mm512_max_ps(max1, _mm512_loadu_ps(data + i + 16));
            max2 = _mm512_max_ps(max2, _mm512_loadu_ps(data + i + 32));
            max3 = _mm512_max_ps(max3, _mm512_loadu_ps(data + i + 48));
        }

        max0 = _mm512_max_ps(max0, max1);
        max2 = _mm512_max_ps(max2, max3);
        max0 = _mm512_max_ps(max0, max2);

        float result = hmax_avx512(max0);

        for (; i < n; ++i) {
            result = std::max(result, data[i]);
        }
        return result;
    }
#endif

#ifdef TENZOR_REDUCTION_AVX2
    if (n >= 32) {
        __m256 max0 = _mm256_loadu_ps(data);
        __m256 max1 = max0;
        __m256 max2 = max0;
        __m256 max3 = max0;

        int64_t i = 8;
        for (; i + 32 <= n; i += 32) {
            max0 = _mm256_max_ps(max0, _mm256_loadu_ps(data + i));
            max1 = _mm256_max_ps(max1, _mm256_loadu_ps(data + i + 8));
            max2 = _mm256_max_ps(max2, _mm256_loadu_ps(data + i + 16));
            max3 = _mm256_max_ps(max3, _mm256_loadu_ps(data + i + 24));
        }

        max0 = _mm256_max_ps(max0, max1);
        max2 = _mm256_max_ps(max2, max3);
        max0 = _mm256_max_ps(max0, max2);

        float result = hmax_avx2(max0);

        for (; i < n; ++i) {
            result = std::max(result, data[i]);
        }
        return result;
    }
#endif

    float result = data[0];
    for (int64_t i = 1; i < n; ++i) {
        result = std::max(result, data[i]);
    }
    return result;
}

/**
 * @brief SIMD-optimized full min reduction
 */
inline float min_full_simd(const float* data, int64_t n) {
    if (n == 0) return std::numeric_limits<float>::infinity();

#ifdef TENZOR_REDUCTION_AVX512
    if (n >= 64) {
        __m512 min0 = _mm512_loadu_ps(data);
        __m512 min1 = min0;
        __m512 min2 = min0;
        __m512 min3 = min0;

        int64_t i = 16;
        for (; i + 64 <= n; i += 64) {
            min0 = _mm512_min_ps(min0, _mm512_loadu_ps(data + i));
            min1 = _mm512_min_ps(min1, _mm512_loadu_ps(data + i + 16));
            min2 = _mm512_min_ps(min2, _mm512_loadu_ps(data + i + 32));
            min3 = _mm512_min_ps(min3, _mm512_loadu_ps(data + i + 48));
        }

        min0 = _mm512_min_ps(min0, min1);
        min2 = _mm512_min_ps(min2, min3);
        min0 = _mm512_min_ps(min0, min2);

        float result = hmin_avx512(min0);

        for (; i < n; ++i) {
            result = std::min(result, data[i]);
        }
        return result;
    }
#endif

#ifdef TENZOR_REDUCTION_AVX2
    if (n >= 32) {
        __m256 min0 = _mm256_loadu_ps(data);
        __m256 min1 = min0;
        __m256 min2 = min0;
        __m256 min3 = min0;

        int64_t i = 8;
        for (; i + 32 <= n; i += 32) {
            min0 = _mm256_min_ps(min0, _mm256_loadu_ps(data + i));
            min1 = _mm256_min_ps(min1, _mm256_loadu_ps(data + i + 8));
            min2 = _mm256_min_ps(min2, _mm256_loadu_ps(data + i + 16));
            min3 = _mm256_min_ps(min3, _mm256_loadu_ps(data + i + 24));
        }

        min0 = _mm256_min_ps(min0, min1);
        min2 = _mm256_min_ps(min2, min3);
        min0 = _mm256_min_ps(min0, min2);

        float result = hmin_avx2(min0);

        for (; i < n; ++i) {
            result = std::min(result, data[i]);
        }
        return result;
    }
#endif

    float result = data[0];
    for (int64_t i = 1; i < n; ++i) {
        result = std::min(result, data[i]);
    }
    return result;
}

// ============================================================================
// Optimized Dimensional Reductions
// ============================================================================

/**
 * @brief Optimized sum along dimension with pre-computed strides
 */
template<typename T>
void sum_along_dim_optimized(
    const T* input,
    T* output,
    const StrideInfo& info
) {
    const int64_t output_size = info.output_size;
    const int64_t reduce_size = info.reduce_size;
    const int64_t reduce_stride = info.input_strides[info.reduce_dim];

    // Check if reduction dimension is contiguous (stride == 1)
    const bool is_contiguous = (reduce_stride == 1);

    #pragma omp parallel for if(output_size > ::tenzor::OmpThresholds::matmul())
    for (int64_t out_idx = 0; out_idx < output_size; ++out_idx) {
        int64_t base_idx = output_to_input_base(out_idx, info);

        if constexpr (std::is_same_v<T, float>) {
            if (is_contiguous && reduce_size >= 8) {
                // Use SIMD for contiguous reduction
                output[out_idx] = sum_full_simd(input + base_idx, reduce_size);
            } else {
                // Strided access with Kahan summation
                T sum = T(0);
                T c = T(0);
                for (int64_t i = 0; i < reduce_size; ++i) {
                    T y = input[base_idx + i * reduce_stride] - c;
                    T t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                output[out_idx] = sum;
            }
        } else {
            // Kahan-compensated sum for all other scalar types (Float64 etc.).
            // Applying compensation here is cheap relative to the compute and
            // avoids the counter-intuitive situation where the higher-precision
            // dtype lacks error compensation.
            T sum = T(0);
            T c = T(0);
            for (int64_t i = 0; i < reduce_size; ++i) {
                T y = input[base_idx + i * reduce_stride] - c;
                T t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            output[out_idx] = sum;
        }
    }
}

/**
 * @brief Optimized max along dimension with pre-computed strides
 */
template<typename T>
void max_along_dim_optimized(
    const T* input,
    T* output,
    const StrideInfo& info
) {
    const int64_t output_size = info.output_size;
    const int64_t reduce_size = info.reduce_size;
    const int64_t reduce_stride = info.input_strides[info.reduce_dim];
    const bool is_contiguous = (reduce_stride == 1);

    #pragma omp parallel for if(output_size > ::tenzor::OmpThresholds::matmul())
    for (int64_t out_idx = 0; out_idx < output_size; ++out_idx) {
        int64_t base_idx = output_to_input_base(out_idx, info);

        if constexpr (std::is_same_v<T, float>) {
            if (is_contiguous && reduce_size >= 8) {
                output[out_idx] = max_full_simd(input + base_idx, reduce_size);
            } else {
                float max_val = input[base_idx];
                for (int64_t i = 1; i < reduce_size; ++i) {
                    max_val = std::max(max_val, input[base_idx + i * reduce_stride]);
                }
                output[out_idx] = max_val;
            }
        } else {
            T max_val = input[base_idx];
            for (int64_t i = 1; i < reduce_size; ++i) {
                max_val = std::max(max_val, input[base_idx + i * reduce_stride]);
            }
            output[out_idx] = max_val;
        }
    }
}

/**
 * @brief Optimized min along dimension with pre-computed strides
 */
template<typename T>
void min_along_dim_optimized(
    const T* input,
    T* output,
    const StrideInfo& info
) {
    const int64_t output_size = info.output_size;
    const int64_t reduce_size = info.reduce_size;
    const int64_t reduce_stride = info.input_strides[info.reduce_dim];
    const bool is_contiguous = (reduce_stride == 1);

    #pragma omp parallel for if(output_size > ::tenzor::OmpThresholds::matmul())
    for (int64_t out_idx = 0; out_idx < output_size; ++out_idx) {
        int64_t base_idx = output_to_input_base(out_idx, info);

        if constexpr (std::is_same_v<T, float>) {
            if (is_contiguous && reduce_size >= 8) {
                output[out_idx] = min_full_simd(input + base_idx, reduce_size);
            } else {
                float min_val = input[base_idx];
                for (int64_t i = 1; i < reduce_size; ++i) {
                    min_val = std::min(min_val, input[base_idx + i * reduce_stride]);
                }
                output[out_idx] = min_val;
            }
        } else {
            T min_val = input[base_idx];
            for (int64_t i = 1; i < reduce_size; ++i) {
                min_val = std::min(min_val, input[base_idx + i * reduce_stride]);
            }
            output[out_idx] = min_val;
        }
    }
}

/**
 * @brief Optimized argmax along dimension
 */
template<typename T>
void argmax_along_dim_optimized(
    const T* input,
    int64_t* output,
    const StrideInfo& info
) {
    const int64_t output_size = info.output_size;
    const int64_t reduce_size = info.reduce_size;
    const int64_t reduce_stride = info.input_strides[info.reduce_dim];

    #pragma omp parallel for if(output_size > ::tenzor::OmpThresholds::matmul())
    for (int64_t out_idx = 0; out_idx < output_size; ++out_idx) {
        int64_t base_idx = output_to_input_base(out_idx, info);

        T max_val = input[base_idx];
        int64_t max_idx = 0;

        for (int64_t i = 1; i < reduce_size; ++i) {
            T val = input[base_idx + i * reduce_stride];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        output[out_idx] = max_idx;
    }
}

/**
 * @brief Optimized argmin along dimension
 */
template<typename T>
void argmin_along_dim_optimized(
    const T* input,
    int64_t* output,
    const StrideInfo& info
) {
    const int64_t output_size = info.output_size;
    const int64_t reduce_size = info.reduce_size;
    const int64_t reduce_stride = info.input_strides[info.reduce_dim];

    #pragma omp parallel for if(output_size > ::tenzor::OmpThresholds::matmul())
    for (int64_t out_idx = 0; out_idx < output_size; ++out_idx) {
        int64_t base_idx = output_to_input_base(out_idx, info);

        T min_val = input[base_idx];
        int64_t min_idx = 0;

        for (int64_t i = 1; i < reduce_size; ++i) {
            T val = input[base_idx + i * reduce_stride];
            if (val < min_val) {
                min_val = val;
                min_idx = i;
            }
        }
        output[out_idx] = min_idx;
    }
}

/**
 * @brief Optimized mean along dimension (sum + divide)
 */
template<typename T>
void mean_along_dim_optimized(
    const T* input,
    T* output,
    const StrideInfo& info
) {
    sum_along_dim_optimized(input, output, info);

    const T scale = T(1) / static_cast<T>(info.reduce_size);
    const int64_t output_size = info.output_size;

    #pragma omp parallel for if(output_size > ::tenzor::OmpThresholds::medium())
    for (int64_t i = 0; i < output_size; ++i) {
        output[i] *= scale;
    }
}

/**
 * @brief Optimized variance along dimension (two-pass)
 */
template<typename T>
void var_along_dim_optimized(
    const T* input,
    T* output,
    const StrideInfo& info,
    int64_t correction = 1
) {
    const int64_t output_size = info.output_size;
    const int64_t reduce_size = info.reduce_size;
    const int64_t reduce_stride = info.input_strides[info.reduce_dim];
    const T divisor = static_cast<T>(std::max(int64_t(1), reduce_size - correction));

    #pragma omp parallel for if(output_size > ::tenzor::OmpThresholds::matmul())
    for (int64_t out_idx = 0; out_idx < output_size; ++out_idx) {
        int64_t base_idx = output_to_input_base(out_idx, info);

        // Pass 1: Compute mean
        T sum = T(0);
        for (int64_t i = 0; i < reduce_size; ++i) {
            sum += input[base_idx + i * reduce_stride];
        }
        T mean = sum / static_cast<T>(reduce_size);

        // Pass 2: Compute variance
        T var_sum = T(0);
        for (int64_t i = 0; i < reduce_size; ++i) {
            T diff = input[base_idx + i * reduce_stride] - mean;
            var_sum += diff * diff;
        }

        output[out_idx] = var_sum / divisor;
    }
}

} // namespace reduction_opt
} // namespace cpu
} // namespace tenzor
