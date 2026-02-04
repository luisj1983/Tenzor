#include "tenzor/core/tensor.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cmath>

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

namespace tenzor {
namespace cpu {

// OpenMP threshold for reductions - needs enough work to amortize thread overhead
// Defined early so SIMD functions can use it
constexpr int64_t REDUCTION_OMP_THRESHOLD = 65536;  // 64K elements

// ============================================================================
// SIMD Horizontal Reduction Helpers
// ============================================================================

#ifdef TENZOR_REDUCTION_AVX512

// Horizontal sum of 16 floats in AVX-512 register -> single float
__attribute__((target("avx512f")))
static inline float hsum_avx512(__m512 v) {
    // Reduce 512 bits -> 256 bits
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 sum256 = _mm256_add_ps(lo, hi);

    // Reduce 256 bits -> 128 bits
    __m128 lo128 = _mm256_castps256_ps128(sum256);
    __m128 hi128 = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(lo128, hi128);

    // Horizontal add within 128 bits
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    return _mm_cvtss_f32(sum128);
}

// Horizontal max of 16 floats in AVX-512 register -> single float
__attribute__((target("avx512f")))
static inline float hmax_avx512(__m512 v) {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 max256 = _mm256_max_ps(lo, hi);

    __m128 lo128 = _mm256_castps256_ps128(max256);
    __m128 hi128 = _mm256_extractf128_ps(max256, 1);
    __m128 max128 = _mm_max_ps(lo128, hi128);

    // Shuffle and max to reduce 4 floats to 1
    __m128 shuf = _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1));
    max128 = _mm_max_ps(max128, shuf);
    shuf = _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2));
    max128 = _mm_max_ps(max128, shuf);

    return _mm_cvtss_f32(max128);
}

// Horizontal min of 16 floats in AVX-512 register -> single float
__attribute__((target("avx512f")))
static inline float hmin_avx512(__m512 v) {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 min256 = _mm256_min_ps(lo, hi);

    __m128 lo128 = _mm256_castps256_ps128(min256);
    __m128 hi128 = _mm256_extractf128_ps(min256, 1);
    __m128 min128 = _mm_min_ps(lo128, hi128);

    __m128 shuf = _mm_shuffle_ps(min128, min128, _MM_SHUFFLE(2, 3, 0, 1));
    min128 = _mm_min_ps(min128, shuf);
    shuf = _mm_shuffle_ps(min128, min128, _MM_SHUFFLE(1, 0, 3, 2));
    min128 = _mm_min_ps(min128, shuf);

    return _mm_cvtss_f32(min128);
}

// AVX-512 vectorized sum for float32
__attribute__((target("avx512f")))
static float simd_sum_f32_avx512(const float* data, int64_t n) {
    constexpr int64_t VEC_SIZE = 16;
    const int64_t vec_end = (n / VEC_SIZE) * VEC_SIZE;

    __m512 vsum = _mm512_setzero_ps();

    // Main vectorized loop - process 16 floats at a time
    for (int64_t i = 0; i < vec_end; i += VEC_SIZE) {
        __m512 v = _mm512_loadu_ps(data + i);
        vsum = _mm512_add_ps(vsum, v);
    }

    // Horizontal sum of vector accumulator
    float sum = hsum_avx512(vsum);

    // Handle remaining elements
    for (int64_t i = vec_end; i < n; i++) {
        sum += data[i];
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
        if (data[i] > max_val) max_val = data[i];
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
        if (data[i] < min_val) min_val = data[i];
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

// AVX2 vectorized sum for float32
__attribute__((target("avx2")))
static float simd_sum_f32_avx2(const float* data, int64_t n) {
    constexpr int64_t VEC_SIZE = 8;
    const int64_t vec_end = (n / VEC_SIZE) * VEC_SIZE;

    __m256 vsum = _mm256_setzero_ps();

    // Main vectorized loop - process 8 floats at a time
    for (int64_t i = 0; i < vec_end; i += VEC_SIZE) {
        __m256 v = _mm256_loadu_ps(data + i);
        vsum = _mm256_add_ps(vsum, v);
    }

    // Horizontal sum of vector accumulator
    float sum = hsum_avx2(vsum);

    // Handle remaining elements
    for (int64_t i = vec_end; i < n; i++) {
        sum += data[i];
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
        if (data[i] > max_val) max_val = data[i];
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
        if (data[i] < min_val) min_val = data[i];
    }

    return min_val;
}

#endif // TENZOR_REDUCTION_AVX2

// ============================================================================
// Parallel SIMD Reduction - combines OpenMP with SIMD for best performance
// ============================================================================

// Parallel SIMD sum for float32 - uses thread-local SIMD accumulators
static float parallel_simd_sum_f32(const float* data, int64_t n) {
    if (n == 0) return 0.0f;

    // For small arrays, use single-threaded SIMD
    if (n < REDUCTION_OMP_THRESHOLD) {
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

    // For large arrays, use parallel reduction with thread-local SIMD
    float total_sum = 0.0f;

    #pragma omp parallel reduction(+:total_sum)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        // Compute chunk for this thread
        int64_t chunk_size = (n + nthreads - 1) / nthreads;
        int64_t start = tid * chunk_size;
        int64_t end = std::min(start + chunk_size, n);

        if (start < end) {
#ifdef TENZOR_REDUCTION_AVX512
            total_sum = simd_sum_f32_avx512(data + start, end - start);
#elif defined(TENZOR_REDUCTION_AVX2)
            total_sum = simd_sum_f32_avx2(data + start, end - start);
#else
            for (int64_t i = start; i < end; i++) {
                total_sum += data[i];
            }
#endif
        }
    }

    return total_sum;
}

// Parallel SIMD max for float32
static float parallel_simd_max_f32(const float* data, int64_t n) {
    if (n == 0) return -std::numeric_limits<float>::infinity();

    if (n < REDUCTION_OMP_THRESHOLD) {
#ifdef TENZOR_REDUCTION_AVX512
        return simd_max_f32_avx512(data, n);
#elif defined(TENZOR_REDUCTION_AVX2)
        return simd_max_f32_avx2(data, n);
#else
        float max_val = data[0];
        for (int64_t i = 1; i < n; i++) {
            if (data[i] > max_val) max_val = data[i];
        }
        return max_val;
#endif
    }

    float global_max = -std::numeric_limits<float>::infinity();

    #pragma omp parallel reduction(max:global_max)
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
        }
    }

    return global_max;
}

// Parallel SIMD min for float32
static float parallel_simd_min_f32(const float* data, int64_t n) {
    if (n == 0) return std::numeric_limits<float>::infinity();

    if (n < REDUCTION_OMP_THRESHOLD) {
#ifdef TENZOR_REDUCTION_AVX512
        return simd_min_f32_avx512(data, n);
#elif defined(TENZOR_REDUCTION_AVX2)
        return simd_min_f32_avx2(data, n);
#else
        float min_val = data[0];
        for (int64_t i = 1; i < n; i++) {
            if (data[i] < min_val) min_val = data[i];
        }
        return min_val;
#endif
    }

    float global_min = std::numeric_limits<float>::infinity();

    #pragma omp parallel reduction(min:global_min)
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
        }
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
        throw std::runtime_error("Dimension " + std::to_string(dim) +
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

    T sum = 0;

    // Use OpenMP reduction clause for non-float types
    #pragma omp parallel for reduction(+:sum) if(n > REDUCTION_OMP_THRESHOLD)
    for (int64_t i = 0; i < n; i++) {
        sum += input_data[i];
    }

    return sum;
}

// Specialization for float - uses SIMD vectorized reduction
template<>
auto sum_impl<float>(const float* input_data, int64_t n) -> float {
    return parallel_simd_sum_f32(input_data, n);
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

    // Reduction along dimension - each output element is independent
    // Use higher threshold since inner loop work is proportional to dim_size
    const int64_t total_work = output_size * dim_size;
    #pragma omp parallel for if(total_work > REDUCTION_OMP_THRESHOLD)
    for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
        // Compute multi-dimensional index for output
        std::vector<int64_t> indices(ndim, 0);
        int64_t tmp = out_idx;

        for (int64_t d = 0; d < ndim; d++) {
            if (d == dim) continue;
            int64_t size = input_shape[d];
            indices[d] = tmp % size;
            tmp /= size;
        }

        // Sum along the reduction dimension - simple accumulation
        T sum = 0;
        for (int64_t i = 0; i < dim_size; i++) {
            indices[dim] = i;
            int64_t in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * input_strides[d];
            }
            sum += input_data[in_idx];
        }
        output_data[out_idx] = sum;
    }
}

auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
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
                // Full reduction - compute in Float32 for precision
                const int64_t n = input.numel();
                float sum = 0.0f;

                // Use OpenMP reduction - fast and efficient
                #pragma omp parallel for reduction(+:sum) if(n > REDUCTION_OMP_THRESHOLD)
                for (int64_t i = 0; i < n; i++) {
                    sum += static_cast<float>(input_data[i]);
                }
                output_data[0] = Float16(sum);
            } else {
                // Dimensional reduction - compute in Float32
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

                    for (int64_t d = 0; d < shape_ndim; d++) {
                        if (d == dim) continue;
                        int64_t size = input_shape[d];
                        indices[d] = tmp % size;
                        tmp /= size;
                    }

                    // Sum along dimension - simple accumulation in Float32
                    float sum = 0.0f;
                    for (int64_t i = 0; i < dim_size; i++) {
                        indices[dim] = i;
                        int64_t in_idx = 0;
                        for (int64_t d = 0; d < shape_ndim; d++) {
                            in_idx += indices[d] * input_strides[d];
                        }
                        sum += static_cast<float>(input_data[in_idx]);
                    }
                    output_data[out_idx] = Float16(sum);
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
        default:
            throw std::runtime_error("sum: unsupported dtype");
    }

    return output;
}

auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    const auto dtype = input.dtype();
    const int64_t ndim = input.ndim();

    // Mean only supports floating point types
    if (dtype != DType::Float16 && dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("mean: only Float16, Float32, and Float64 are supported");
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

    // Divide sum by count
    if (dtype == DType::Float16) {
        auto* data = sum_result.data<Float16>();
        const float scale = 1.0f / static_cast<float>(count);
        const int64_t n = sum_result.numel();

        #pragma omp parallel for if(n > 10000)
        for (int64_t i = 0; i < n; i++) {
            data[i] = Float16(static_cast<float>(data[i]) * scale);
        }
    } else if (dtype == DType::Float32) {
        auto* data = sum_result.data<float>();
        const float scale = 1.0f / static_cast<float>(count);
        const int64_t n = sum_result.numel();

        #pragma omp parallel for if(n > 10000)
        for (int64_t i = 0; i < n; i++) {
            data[i] *= scale;
        }
    } else {  // Float64
        auto* data = sum_result.data<double>();
        const double scale = 1.0 / static_cast<double>(count);
        const int64_t n = sum_result.numel();

        #pragma omp parallel for if(n > 10000)
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

        for (int64_t d = 0; d < ndim; d++) {
            if (d == dim) continue;
            int64_t size = input_shape[d];
            indices[d] = tmp % size;
            tmp /= size;
        }

        // Find max along the reduction dimension
        indices[dim] = 0;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }
        T max_val = input_data[in_idx];

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

auto max_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize negative dimension
    dim = normalize_dim(dim, ndim);

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
                // Compute max in Float32
                float max_val = std::numeric_limits<float>::lowest();
                const int64_t n = input.numel();
                #pragma omp parallel for reduction(max:max_val) if(n > REDUCTION_OMP_THRESHOLD)
                for (int64_t i = 0; i < n; i++) {
                    float val = static_cast<float>(input_data[i]);
                    if (val > max_val) {
                        max_val = val;
                    }
                }
                output_data[0] = Float16(max_val);
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
                    for (int64_t d = 0; d < ndim; d++) {
                        if (d == dim) continue;
                        indices[d] = tmp % input_shape[d];
                        tmp /= input_shape[d];
                    }

                    float max_val = std::numeric_limits<float>::lowest();
                    for (int64_t i = 0; i < dim_size; i++) {
                        indices[dim] = i;
                        int64_t in_idx = 0;
                        for (int64_t d = 0; d < ndim; d++) {
                            in_idx += indices[d] * input_strides[d];
                        }
                        float val = static_cast<float>(input_data[in_idx]);
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                    output_data[out_idx] = Float16(max_val);
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

        for (int64_t d = 0; d < ndim; d++) {
            if (d == dim) continue;
            int64_t size = input_shape[d];
            indices[d] = tmp % size;
            tmp /= size;
        }

        // Find min along the reduction dimension
        indices[dim] = 0;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }
        T min_val = input_data[in_idx];

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

auto min_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize negative dimension
    dim = normalize_dim(dim, ndim);

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
                // Compute min in Float32
                float min_val = std::numeric_limits<float>::max();
                const int64_t n = input.numel();
                #pragma omp parallel for reduction(min:min_val) if(n > REDUCTION_OMP_THRESHOLD)
                for (int64_t i = 0; i < n; i++) {
                    float val = static_cast<float>(input_data[i]);
                    if (val < min_val) {
                        min_val = val;
                    }
                }
                output_data[0] = Float16(min_val);
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
                    for (int64_t d = 0; d < ndim; d++) {
                        if (d == dim) continue;
                        indices[d] = tmp % input_shape[d];
                        tmp /= input_shape[d];
                    }

                    float min_val = std::numeric_limits<float>::max();
                    for (int64_t i = 0; i < dim_size; i++) {
                        indices[dim] = i;
                        int64_t in_idx = 0;
                        for (int64_t d = 0; d < ndim; d++) {
                            in_idx += indices[d] * input_strides[d];
                        }
                        float val = static_cast<float>(input_data[in_idx]);
                        if (val < min_val) {
                            min_val = val;
                        }
                    }
                    output_data[out_idx] = Float16(min_val);
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
        default:
            throw std::runtime_error("min: unsupported dtype");
    }

    return output;
}

// Template for argmax reduction - returns index of maximum value
template<typename T>
auto argmax_impl(const T* input_data, int64_t n) -> int64_t {
    if (n == 0) throw std::runtime_error("argmax: input tensor is empty");

    int64_t max_idx = 0;
    T max_val = input_data[0];

    for (int64_t i = 1; i < n; i++) {
        if (input_data[i] > max_val) {
            max_val = input_data[i];
            max_idx = i;
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

        for (int64_t d = 0; d < ndim; d++) {
            if (d == dim) continue;
            int64_t size = input_shape[d];
            indices[d] = tmp % size;
            tmp /= size;
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

            if (input_data[in_idx] > max_val) {
                max_val = input_data[in_idx];
                max_idx = i;
            }
        }

        output_data[out_idx] = max_idx;
    }
}

auto argmax_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    // Argmax always returns Int64 indices
    auto input_shape_span = input.shape();
    std::vector<int64_t> input_shape_vec(input_shape_span.begin(), input_shape_span.end());
    const int64_t ndim = static_cast<int64_t>(input_shape_vec.size());

    // Normalize negative dimension
    dim = normalize_dim(dim, ndim);

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
        case DType::Float16: {
            auto* input_data = input.data<Float16>();
            if (dim == REDUCE_ALL) {
                // Convert to Float32 comparison with OpenMP
                const int64_t n = input.numel();
                if (n == 0) throw std::runtime_error("argmax: input tensor is empty");
                int64_t max_idx = 0;
                float max_val = static_cast<float>(input_data[0]);

                #ifdef _OPENMP
                if (n > REDUCTION_OMP_THRESHOLD) {
                    // Parallel reduction: each thread finds local max, then combine
                    #pragma omp parallel
                    {
                        int64_t local_idx = 0;
                        float local_max = static_cast<float>(input_data[0]);
                        #pragma omp for nowait
                        for (int64_t i = 1; i < n; i++) {
                            float val = static_cast<float>(input_data[i]);
                            if (val > local_max) {
                                local_max = val;
                                local_idx = i;
                            }
                        }
                        #pragma omp critical
                        {
                            if (local_max > max_val || (local_max == max_val && local_idx < max_idx)) {
                                max_val = local_max;
                                max_idx = local_idx;
                            }
                        }
                    }
                } else
                #endif
                {
                    for (int64_t i = 1; i < n; i++) {
                        float val = static_cast<float>(input_data[i]);
                        if (val > max_val) {
                            max_val = val;
                            max_idx = i;
                        }
                    }
                }
                output_data[0] = max_idx;
            } else {
                // Dimensional argmax
                const int64_t dim_size = input_shape[dim];
                const int64_t output_size = output.numel();

                #pragma omp parallel for if(output_size > 1000)
                for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
                    std::vector<int64_t> indices(ndim, 0);
                    int64_t tmp = out_idx;
                    for (int64_t d = 0; d < ndim; d++) {
                        if (d == dim) continue;
                        int64_t size = input_shape[d];
                        indices[d] = tmp % size;
                        tmp /= size;
                    }

                    indices[dim] = 0;
                    int64_t in_idx = 0;
                    for (int64_t d = 0; d < ndim; d++) {
                        in_idx += indices[d] * input_strides[d];
                    }

                    float max_val = static_cast<float>(input_data[in_idx]);
                    int64_t max_idx = 0;

                    for (int64_t i = 1; i < dim_size; i++) {
                        indices[dim] = i;
                        in_idx = 0;
                        for (int64_t d = 0; d < ndim; d++) {
                            in_idx += indices[d] * input_strides[d];
                        }
                        float val = static_cast<float>(input_data[in_idx]);
                        if (val > max_val) {
                            max_val = val;
                            max_idx = i;
                        }
                    }
                    output_data[out_idx] = max_idx;
                }
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
        default:
            throw std::runtime_error("argmax: unsupported dtype");
    }

    return output;
}

// Argmin implementation - find index of minimum value
template<typename T>
auto argmin_impl(const T* input_data, int64_t n) -> int64_t {
    if (n == 0) throw std::runtime_error("argmin: input tensor is empty");

    int64_t min_idx = 0;
    T min_val = input_data[0];

    for (int64_t i = 1; i < n; i++) {
        if (input_data[i] < min_val) {
            min_val = input_data[i];
            min_idx = i;
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

        for (int64_t d = 0; d < ndim; d++) {
            if (d == dim) continue;
            int64_t size = input_shape[d];
            indices[d] = tmp % size;
            tmp /= size;
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

            if (input_data[in_idx] < min_val) {
                min_val = input_data[in_idx];
                min_idx = i;
            }
        }

        output_data[out_idx] = min_idx;
    }
}

// Argmin kernel - returns indices of minimum values
auto argmin_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    // Argmin always returns Int64 indices
    auto input_shape_span = input.shape();
    std::vector<int64_t> input_shape_vec(input_shape_span.begin(), input_shape_span.end());
    const int64_t ndim = static_cast<int64_t>(input_shape_vec.size());

    // Normalize negative dimension
    dim = normalize_dim(dim, ndim);

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
                     [data](int64_t a, int64_t b) { return data[a] > data[b]; });
        } else {
            std::sort(indices.begin(), indices.end(),
                     [data](int64_t a, int64_t b) { return data[a] < data[b]; });
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
            auto comp = [data](int64_t a, int64_t b) { return data[a] > data[b]; };
            #pragma omp parallel
            {
                #pragma omp single
                parallel_merge_sort<T>(indices.data(), buffer.data(), 0, n, comp, depth);
            }
        } else {
            auto comp = [data](int64_t a, int64_t b) { return data[a] < data[b]; };
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
                      const std::vector<int64_t>& strides,
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
    #pragma omp parallel for if(outer_size * inner_size > 1000)
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

auto argsort_kernel(const Tensor& input, int64_t dim, bool descending) -> Tensor {
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
    #pragma omp parallel for reduction(*:result) if(n > 10000)
    for (int64_t i = 0; i < n; i++) {
        result *= input_data[i];
    }
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

    #pragma omp parallel for if(output_size > 1000)
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
auto prod_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
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
        case DType::Float16: {
            // Compute product in Float32, store result as Float32
            // (Float16 product overflows easily, so output stays Float32)
            auto input_f32 = input.to(DType::Float32);
            auto* input_data = input_f32.data<float>();

            // Reallocate output as Float32
            output = Tensor(output_shape, DType::Float32, input.device());
            auto* output_data = output.data<float>();
            if (dim == REDUCE_ALL) {
                output_data[0] = prod_impl(input_data, input.numel());
            } else {
                auto strides_span = input_f32.strides();
                std::vector<int64_t> input_strides(strides_span.begin(), strides_span.end());
                prod_along_dim(input_data, output_data,
                              std::vector<int64_t>(input_f32.shape().begin(), input_f32.shape().end()),
                              input_strides, dim);
            }
            break;
        }
        case DType::BFloat16: {
            // Compute product in Float32, store result as Float32
            auto input_f32 = input.to(DType::Float32);
            auto* input_data = input_f32.data<float>();

            output = Tensor(output_shape, DType::Float32, input.device());
            auto* output_data = output.data<float>();
            if (dim == REDUCE_ALL) {
                output_data[0] = prod_impl(input_data, input.numel());
            } else {
                auto strides_span = input_f32.strides();
                std::vector<int64_t> input_strides(strides_span.begin(), strides_span.end());
                prod_along_dim(input_data, output_data,
                              std::vector<int64_t>(input_f32.shape().begin(), input_f32.shape().end()),
                              input_strides, dim);
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
    if (divisor <= 0) divisor = 1;

    const int64_t total_work = output_size * dim_size;
    #pragma omp parallel for if(total_work > REDUCTION_OMP_THRESHOLD)
    for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
        std::vector<int64_t> indices(ndim, 0);
        int64_t tmp = out_idx;
        for (int64_t d = 0; d < ndim; d++) {
            if (d == dim) continue;
            indices[d] = tmp % input_shape[d];
            tmp /= input_shape[d];
        }

        // Pass 1: compute mean
        T sum = T(0);
        for (int64_t i = 0; i < dim_size; i++) {
            indices[dim] = i;
            int64_t in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * input_strides[d];
            }
            sum += input_data[in_idx];
        }
        T mean = sum / static_cast<T>(dim_size);

        // Pass 2: compute variance
        T var_sum = T(0);
        for (int64_t i = 0; i < dim_size; i++) {
            indices[dim] = i;
            int64_t in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * input_strides[d];
            }
            T diff = input_data[in_idx] - mean;
            var_sum += diff * diff;
        }
        output_data[out_idx] = var_sum / static_cast<T>(divisor);
    }
}

// Variance using two-pass algorithm for numerical stability
auto var_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction) -> Tensor {
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
                float mean = sum_impl(input_data, n) / static_cast<float>(n);
                float var_sum = 0.0f;
                #pragma omp parallel for reduction(+:var_sum) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    float diff = input_data[i] - mean;
                    var_sum += diff * diff;
                }
                int64_t divisor = n - correction;
                if (divisor <= 0) divisor = 1;
                output_data[0] = var_sum / static_cast<float>(divisor);
                break;
            }
            case DType::Float64: {
                auto* input_data = input.data<double>();
                auto* output_data = output.data<double>();
                double mean = sum_impl(input_data, n) / static_cast<double>(n);
                double var_sum = 0.0;
                #pragma omp parallel for reduction(+:var_sum) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    double diff = input_data[i] - mean;
                    var_sum += diff * diff;
                }
                int64_t divisor = n - correction;
                if (divisor <= 0) divisor = 1;
                output_data[0] = var_sum / static_cast<double>(divisor);
                break;
            }
            case DType::Float16: {
                auto* input_data = input.data<Float16>();
                auto* output_data = output.data<float>();
                float sum = 0.0f;
                #pragma omp parallel for reduction(+:sum) if(n > REDUCTION_OMP_THRESHOLD)
                for (int64_t i = 0; i < n; i++) sum += static_cast<float>(input_data[i]);
                float mean = sum / static_cast<float>(n);
                float var_sum = 0.0f;
                #pragma omp parallel for reduction(+:var_sum) if(n > REDUCTION_OMP_THRESHOLD)
                for (int64_t i = 0; i < n; i++) {
                    float diff = static_cast<float>(input_data[i]) - mean;
                    var_sum += diff * diff;
                }
                int64_t divisor = n - correction;
                if (divisor <= 0) divisor = 1;
                output_data[0] = var_sum / static_cast<float>(divisor);
                break;
            }
            case DType::BFloat16: {
                auto* input_data = input.data<BFloat16>();
                auto* output_data = output.data<float>();
                float sum = 0.0f;
                #pragma omp parallel for reduction(+:sum) if(n > REDUCTION_OMP_THRESHOLD)
                for (int64_t i = 0; i < n; i++) sum += static_cast<float>(input_data[i]);
                float mean = sum / static_cast<float>(n);
                float var_sum = 0.0f;
                #pragma omp parallel for reduction(+:var_sum) if(n > REDUCTION_OMP_THRESHOLD)
                for (int64_t i = 0; i < n; i++) {
                    float diff = static_cast<float>(input_data[i]) - mean;
                    var_sum += diff * diff;
                }
                int64_t divisor = n - correction;
                if (divisor <= 0) divisor = 1;
                output_data[0] = var_sum / static_cast<float>(divisor);
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
                // Compute in Float32
                int64_t output_size = output.numel();
                const int64_t dim_size = input_shape[dim];
                const auto* input_data = input.data<Float16>();
                auto* output_data = output.data<float>();

                int64_t out_size = 1;
                for (int64_t d = 0; d < ndim; d++) {
                    if (d != dim) out_size *= input_shape[d];
                }

                int64_t divisor = dim_size - correction;
                if (divisor <= 0) divisor = 1;

                #pragma omp parallel for if(out_size * dim_size > REDUCTION_OMP_THRESHOLD)
                for (int64_t out_idx = 0; out_idx < out_size; out_idx++) {
                    std::vector<int64_t> indices(ndim, 0);
                    int64_t tmp = out_idx;
                    for (int64_t d = 0; d < ndim; d++) {
                        if (d == dim) continue;
                        indices[d] = tmp % input_shape[d];
                        tmp /= input_shape[d];
                    }

                    float sum = 0.0f;
                    for (int64_t i = 0; i < dim_size; i++) {
                        indices[dim] = i;
                        int64_t in_idx = 0;
                        for (int64_t d = 0; d < ndim; d++) in_idx += indices[d] * input_strides[d];
                        sum += static_cast<float>(input_data[in_idx]);
                    }
                    float mean = sum / static_cast<float>(dim_size);

                    float var_sum = 0.0f;
                    for (int64_t i = 0; i < dim_size; i++) {
                        indices[dim] = i;
                        int64_t in_idx = 0;
                        for (int64_t d = 0; d < ndim; d++) in_idx += indices[d] * input_strides[d];
                        float diff = static_cast<float>(input_data[in_idx]) - mean;
                        var_sum += diff * diff;
                    }
                    output_data[out_idx] = var_sum / static_cast<float>(divisor);
                }
                break;
            }
            case DType::BFloat16: {
                int64_t output_size = output.numel();
                const int64_t dim_size = input_shape[dim];
                const auto* input_data = input.data<BFloat16>();
                auto* output_data = output.data<float>();

                int64_t out_size = 1;
                for (int64_t d = 0; d < ndim; d++) {
                    if (d != dim) out_size *= input_shape[d];
                }

                int64_t divisor = dim_size - correction;
                if (divisor <= 0) divisor = 1;

                #pragma omp parallel for if(out_size * dim_size > REDUCTION_OMP_THRESHOLD)
                for (int64_t out_idx = 0; out_idx < out_size; out_idx++) {
                    std::vector<int64_t> indices(ndim, 0);
                    int64_t tmp = out_idx;
                    for (int64_t d = 0; d < ndim; d++) {
                        if (d == dim) continue;
                        indices[d] = tmp % input_shape[d];
                        tmp /= input_shape[d];
                    }

                    float sum = 0.0f;
                    for (int64_t i = 0; i < dim_size; i++) {
                        indices[dim] = i;
                        int64_t in_idx = 0;
                        for (int64_t d = 0; d < ndim; d++) in_idx += indices[d] * input_strides[d];
                        sum += static_cast<float>(input_data[in_idx]);
                    }
                    float mean = sum / static_cast<float>(dim_size);

                    float var_sum = 0.0f;
                    for (int64_t i = 0; i < dim_size; i++) {
                        indices[dim] = i;
                        int64_t in_idx = 0;
                        for (int64_t d = 0; d < ndim; d++) in_idx += indices[d] * input_strides[d];
                        float diff = static_cast<float>(input_data[in_idx]) - mean;
                        var_sum += diff * diff;
                    }
                    output_data[out_idx] = var_sum / static_cast<float>(divisor);
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
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                output_data[i] = std::sqrt(var_data[i]);
            }
            break;
        }
        case DType::Float64: {
            auto* var_data = var_result.data<double>();
            auto* output_data = output.data<double>();
            #pragma omp parallel for if(n > 10000)
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

// Norm operation - compute Lp norm
auto norm_kernel(const Tensor& input, float p, int64_t dim, bool keepdim) -> Tensor {
    auto shape_span = input.shape();
    std::vector<int64_t> input_shape(shape_span.begin(), shape_span.end());
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize negative dimension
    dim = normalize_dim(dim, ndim);

    if (dim != REDUCE_ALL) {
        throw std::runtime_error("norm: only full reduction is currently supported for CPU");
    }

    auto output_shape = compute_reduction_shape(input_shape, dim, keepdim);
    Tensor output(output_shape, input.dtype(), input.device());

    const int64_t n = input.numel();
    if (n == 0) {
        throw std::runtime_error("norm: input tensor is empty");
    }

    switch (input.dtype()) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            float norm_value = 0.0f;

            if (p == 1.0f) {
                // L1 norm: sum of absolute values
                #pragma omp parallel for reduction(+:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    norm_value += std::abs(input_data[i]);
                }
            } else if (p == 2.0f) {
                // L2 norm: sqrt of sum of squares
                #pragma omp parallel for reduction(+:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    norm_value += input_data[i] * input_data[i];
                }
                norm_value = std::sqrt(norm_value);
            } else if (std::isinf(p)) {
                // L-inf norm: max absolute value (parallelized)
                #pragma omp parallel for reduction(max:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    float abs_val = std::abs(input_data[i]);
                    if (abs_val > norm_value) {
                        norm_value = abs_val;
                    }
                }
            } else {
                // General Lp norm: (sum(|x|^p))^(1/p)
                #pragma omp parallel for reduction(+:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    norm_value += std::pow(std::abs(input_data[i]), p);
                }
                norm_value = std::pow(norm_value, 1.0f / p);
            }

            output_data[0] = norm_value;
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            double norm_value = 0.0;

            if (p == 1.0) {
                // L1 norm: sum of absolute values
                #pragma omp parallel for reduction(+:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    norm_value += std::abs(input_data[i]);
                }
            } else if (p == 2.0) {
                // L2 norm: sqrt of sum of squares
                #pragma omp parallel for reduction(+:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    norm_value += input_data[i] * input_data[i];
                }
                norm_value = std::sqrt(norm_value);
            } else if (std::isinf(p)) {
                // L-inf norm: max absolute value (parallelized)
                #pragma omp parallel for reduction(max:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    double abs_val = std::abs(input_data[i]);
                    if (abs_val > norm_value) {
                        norm_value = abs_val;
                    }
                }
            } else {
                // General Lp norm: (sum(|x|^p))^(1/p)
                #pragma omp parallel for reduction(+:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    norm_value += std::pow(std::abs(input_data[i]), p);
                }
                norm_value = std::pow(norm_value, 1.0 / p);
            }

            output_data[0] = norm_value;
            break;
        }
        case DType::Float16: {
            // Compute norm in Float32
            auto* input_data = input.data<Float16>();
            output = Tensor(output_shape, DType::Float32, input.device());
            auto* output_data = output.data<float>();

            float norm_value = 0.0f;

            if (p == 1.0f) {
                #pragma omp parallel for reduction(+:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    norm_value += std::abs(static_cast<float>(input_data[i]));
                }
            } else if (p == 2.0f) {
                #pragma omp parallel for reduction(+:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    float v = static_cast<float>(input_data[i]);
                    norm_value += v * v;
                }
                norm_value = std::sqrt(norm_value);
            } else if (std::isinf(p)) {
                #pragma omp parallel for reduction(max:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    float abs_val = std::abs(static_cast<float>(input_data[i]));
                    if (abs_val > norm_value) {
                        norm_value = abs_val;
                    }
                }
            } else {
                #pragma omp parallel for reduction(+:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    norm_value += std::pow(std::abs(static_cast<float>(input_data[i])), p);
                }
                norm_value = std::pow(norm_value, 1.0f / p);
            }

            output_data[0] = norm_value;
            break;
        }
        case DType::BFloat16: {
            // Compute norm in Float32
            auto* input_data = input.data<BFloat16>();
            output = Tensor(output_shape, DType::Float32, input.device());
            auto* output_data = output.data<float>();

            float norm_value = 0.0f;

            if (p == 1.0f) {
                #pragma omp parallel for reduction(+:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    norm_value += std::abs(static_cast<float>(input_data[i]));
                }
            } else if (p == 2.0f) {
                #pragma omp parallel for reduction(+:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    float v = static_cast<float>(input_data[i]);
                    norm_value += v * v;
                }
                norm_value = std::sqrt(norm_value);
            } else if (std::isinf(p)) {
                #pragma omp parallel for reduction(max:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    float abs_val = std::abs(static_cast<float>(input_data[i]));
                    if (abs_val > norm_value) {
                        norm_value = abs_val;
                    }
                }
            } else {
                #pragma omp parallel for reduction(+:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    norm_value += std::pow(std::abs(static_cast<float>(input_data[i])), p);
                }
                norm_value = std::pow(norm_value, 1.0f / p);
            }

            output_data[0] = norm_value;
            break;
        }
        default:
            throw std::runtime_error("norm: unsupported dtype");
    }

    return output;
}

} // namespace cpu
} // namespace tenzor
