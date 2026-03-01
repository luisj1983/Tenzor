#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <cmath>
#include <vector>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD headers for optimized BatchNorm
#ifdef __AVX2__
#include <immintrin.h>
#endif

// Intel oneDNN for optimized batch normalization (2-3x faster)
#ifdef TENZOR_USE_ONEDNN
#include <dnnl.hpp>
#include "onednn_cache.hpp"
#include <list>
#include <unordered_map>
#endif

namespace tenzor {
namespace cpu {

// ============================================================================
// Float16 Arithmetic Helper Functions
// ============================================================================
// These inline helpers allow Float16 to work with template code that uses
// arithmetic operators. Operations are performed in Float32 precision.

inline Float16 operator+(const Float16& a, const Float16& b) {
    return Float16(static_cast<float>(a) + static_cast<float>(b));
}

inline Float16 operator-(const Float16& a, const Float16& b) {
    return Float16(static_cast<float>(a) - static_cast<float>(b));
}

inline Float16 operator*(const Float16& a, const Float16& b) {
    return Float16(static_cast<float>(a) * static_cast<float>(b));
}

inline Float16 operator/(const Float16& a, const Float16& b) {
    return Float16(static_cast<float>(a) / static_cast<float>(b));
}

inline Float16& operator+=(Float16& a, const Float16& b) {
    a = Float16(static_cast<float>(a) + static_cast<float>(b));
    return a;
}

inline Float16& operator-=(Float16& a, const Float16& b) {
    a = Float16(static_cast<float>(a) - static_cast<float>(b));
    return a;
}

inline Float16& operator*=(Float16& a, const Float16& b) {
    a = Float16(static_cast<float>(a) * static_cast<float>(b));
    return a;
}

inline Float16& operator/=(Float16& a, const Float16& b) {
    a = Float16(static_cast<float>(a) / static_cast<float>(b));
    return a;
}

// ============================================================================
// BFloat16 Arithmetic Helper Functions
// ============================================================================
inline BFloat16 operator+(const BFloat16& a, const BFloat16& b) {
    return BFloat16(static_cast<float>(a) + static_cast<float>(b));
}

inline BFloat16 operator-(const BFloat16& a, const BFloat16& b) {
    return BFloat16(static_cast<float>(a) - static_cast<float>(b));
}

inline BFloat16 operator*(const BFloat16& a, const BFloat16& b) {
    return BFloat16(static_cast<float>(a) * static_cast<float>(b));
}

inline BFloat16 operator/(const BFloat16& a, const BFloat16& b) {
    return BFloat16(static_cast<float>(a) / static_cast<float>(b));
}

inline BFloat16& operator+=(BFloat16& a, const BFloat16& b) {
    a = BFloat16(static_cast<float>(a) + static_cast<float>(b));
    return a;
}

inline BFloat16& operator-=(BFloat16& a, const BFloat16& b) {
    a = BFloat16(static_cast<float>(a) - static_cast<float>(b));
    return a;
}

inline BFloat16& operator*=(BFloat16& a, const BFloat16& b) {
    a = BFloat16(static_cast<float>(a) * static_cast<float>(b));
    return a;
}

inline BFloat16& operator/=(BFloat16& a, const BFloat16& b) {
    a = BFloat16(static_cast<float>(a) / static_cast<float>(b));
    return a;
}

// Math helper templates for Float16/BFloat16 support
template<typename T>
inline T safe_sqrt(const T& x) {
    return std::sqrt(x);
}

template<>
inline Float16 safe_sqrt<Float16>(const Float16& x) {
    return Float16(safe_sqrt(static_cast<float>(x)));
}

template<>
inline BFloat16 safe_sqrt<BFloat16>(const BFloat16& x) {
    return BFloat16(std::sqrt(static_cast<float>(x)));
}

// ============================================================================
// BatchNorm2d Mean/Variance Computation
// ============================================================================

#ifdef __AVX2__
// SIMD-optimized mean/variance computation for Float32
// Uses AVX2 for 8-wide parallel reduction
static void batchnorm_mean_var_simd_f32(
    const float* __restrict input,
    float* __restrict mean,
    float* __restrict variance,
    int64_t N, int64_t C, int64_t H, int64_t W)
{
    int64_t spatial_size = H * W;
    int64_t total_elements = N * spatial_size;
    float inv_total = 1.0f / static_cast<float>(total_elements);

    // Use fewer threads to avoid OpenMP overhead
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    int effective_threads = std::min({nthreads, static_cast<int>(C), 4});
    int final_threads = std::max(1, effective_threads);
#else
    int final_threads = 1;
#endif

    #pragma omp parallel for num_threads(final_threads) if(C > 1)
    for (int64_t c = 0; c < C; c++) {
        // First pass: compute sum using AVX2
        __m256 vsum = _mm256_setzero_ps();

        for (int64_t n = 0; n < N; n++) {
            const float* ch_ptr = input + (n * C + c) * spatial_size;
            int64_t i = 0;

            // Process 8 floats at a time
            for (; i + 8 <= spatial_size; i += 8) {
                __m256 v = _mm256_loadu_ps(ch_ptr + i);
                vsum = _mm256_add_ps(vsum, v);
            }
            // Handle remainder with scalar accumulator to avoid 8x overcounting
            float scalar_remainder_sum = 0.0f;
            for (; i < spatial_size; i++) {
                scalar_remainder_sum += ch_ptr[i];
            }
            // Add scalar to only lane 0 of vsum
            __m256 remainder_vec = _mm256_setzero_ps();
            remainder_vec = _mm256_blend_ps(remainder_vec, _mm256_set1_ps(scalar_remainder_sum), 0x01);
            vsum = _mm256_add_ps(vsum, remainder_vec);
        }

        // Horizontal sum of vsum
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 sum128 = _mm_add_ps(hi, lo);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float channel_mean = _mm_cvtss_f32(sum128) * inv_total;
        mean[c] = channel_mean;

        // Second pass: compute variance
        __m256 vmean = _mm256_set1_ps(channel_mean);
        __m256 vvar = _mm256_setzero_ps();

        for (int64_t n = 0; n < N; n++) {
            const float* ch_ptr = input + (n * C + c) * spatial_size;
            int64_t i = 0;

            // Process 8 floats at a time
            for (; i + 8 <= spatial_size; i += 8) {
                __m256 v = _mm256_loadu_ps(ch_ptr + i);
                __m256 diff = _mm256_sub_ps(v, vmean);
                vvar = _mm256_fmadd_ps(diff, diff, vvar);
            }
            // Handle remainder with scalar accumulator to avoid 8x overcounting
            float scalar_remainder_var = 0.0f;
            for (; i < spatial_size; i++) {
                float diff = ch_ptr[i] - channel_mean;
                scalar_remainder_var += diff * diff;
            }
            // Add scalar to only lane 0 of vvar
            __m256 var_remainder_vec = _mm256_setzero_ps();
            var_remainder_vec = _mm256_blend_ps(var_remainder_vec, _mm256_set1_ps(scalar_remainder_var), 0x01);
            vvar = _mm256_add_ps(vvar, var_remainder_vec);
        }

        // Horizontal sum of vvar
        hi = _mm256_extractf128_ps(vvar, 1);
        lo = _mm256_castps256_ps128(vvar);
        __m128 var128 = _mm_add_ps(hi, lo);
        var128 = _mm_hadd_ps(var128, var128);
        var128 = _mm_hadd_ps(var128, var128);
        variance[c] = _mm_cvtss_f32(var128) * inv_total;
    }
}
#endif

// Compute per-channel mean and variance
// Input: [N, C, H, W] - NCHW format
// Output: mean[C], variance[C]
template<typename T>
void batchnorm_mean_var_impl(const T* input,
                             T* mean,
                             T* variance,
                             int64_t N,
                             int64_t C,
                             int64_t H,
                             int64_t W) {
    int64_t spatial_size = H * W;
    int64_t total_elements = N * spatial_size;

    // Check for division by zero
    if (total_elements == 0) {
        throw std::runtime_error("BatchNorm2d: Cannot compute mean/variance for empty tensor (total_elements = 0)");
    }

    // Use fewer threads to avoid OpenMP overhead
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    int effective_threads = std::min({nthreads, static_cast<int>(C), 4});
    int final_threads = std::max(1, effective_threads);
#else
    int final_threads = 1;
#endif

    // Compute mean and variance for each channel
    #pragma omp parallel for num_threads(final_threads) if(C > 1)
    for (int64_t c = 0; c < C; c++) {
        // Compute mean - simple sum (Kahan adds too much overhead)
        T sum = T(0.0f);
        for (int64_t n = 0; n < N; n++) {
            const T* ch_ptr = input + (n * C + c) * spatial_size;
            for (int64_t hw = 0; hw < spatial_size; hw++) {
                sum += ch_ptr[hw];
            }
        }
        T channel_mean = sum / T(static_cast<float>(total_elements));
        mean[c] = channel_mean;

        // Compute variance
        T sum_sq_diff = T(0.0f);
        for (int64_t n = 0; n < N; n++) {
            const T* ch_ptr = input + (n * C + c) * spatial_size;
            for (int64_t hw = 0; hw < spatial_size; hw++) {
                T diff = ch_ptr[hw] - channel_mean;
                sum_sq_diff += diff * diff;
            }
        }
        variance[c] = sum_sq_diff / T(static_cast<float>(total_elements));
    }
}

// Specialized Float16 version that accumulates in Float32 to prevent overflow
// Float16 has limited range (~65504 max), so summing many values can overflow
template<>
void batchnorm_mean_var_impl<Float16>(const Float16* input,
                                       Float16* mean,
                                       Float16* variance,
                                       int64_t N,
                                       int64_t C,
                                       int64_t H,
                                       int64_t W) {
    int64_t spatial_size = H * W;
    int64_t total_elements = N * spatial_size;

    if (total_elements == 0) {
        throw std::runtime_error("BatchNorm2d: Cannot compute mean/variance for empty tensor (total_elements = 0)");
    }

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    int effective_threads = std::min({nthreads, static_cast<int>(C), 4});
    int final_threads = std::max(1, effective_threads);
#else
    int final_threads = 1;
#endif

    // Accumulate in Float32 to prevent overflow
    #pragma omp parallel for num_threads(final_threads) if(C > 1)
    for (int64_t c = 0; c < C; c++) {
        // Compute mean with Float32 accumulation
        float sum = 0.0f;
        for (int64_t n = 0; n < N; n++) {
            const Float16* ch_ptr = input + (n * C + c) * spatial_size;
            for (int64_t hw = 0; hw < spatial_size; hw++) {
                sum += static_cast<float>(ch_ptr[hw]);
            }
        }
        float channel_mean = sum / static_cast<float>(total_elements);
        mean[c] = Float16(channel_mean);

        // Compute variance with Float32 accumulation
        float sum_sq_diff = 0.0f;
        for (int64_t n = 0; n < N; n++) {
            const Float16* ch_ptr = input + (n * C + c) * spatial_size;
            for (int64_t hw = 0; hw < spatial_size; hw++) {
                float diff = static_cast<float>(ch_ptr[hw]) - channel_mean;
                sum_sq_diff += diff * diff;
            }
        }
        variance[c] = Float16(sum_sq_diff / static_cast<float>(total_elements));
    }
}

auto batchnorm2d_mean_var_kernel(const Tensor& input) -> std::vector<Tensor> {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("batchnorm2d_mean_var expects 4D input (NCHW)");
    }

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    // Allocate output tensors
    Tensor mean({C}, input.dtype(), input.device());
    Tensor variance({C}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
#ifdef __AVX2__
        // Use SIMD-optimized version for Float32
        batchnorm_mean_var_simd_f32(
            input.data<float>(),
            mean.data<float>(),
            variance.data<float>(),
            N, C, H, W
        );
#else
        batchnorm_mean_var_impl<float>(
            input.data<float>(),
            mean.data<float>(),
            variance.data<float>(),
            N, C, H, W
        );
#endif
    } else if (input.dtype() == DType::Float64) {
        batchnorm_mean_var_impl<double>(
            input.data<double>(),
            mean.data<double>(),
            variance.data<double>(),
            N, C, H, W
        );
    } else if (input.dtype() == DType::Float16) {
        batchnorm_mean_var_impl<Float16>(
            input.data<Float16>(),
            mean.data<Float16>(),
            variance.data<Float16>(),
            N, C, H, W
        );
    } else if (input.dtype() == DType::BFloat16) {
        batchnorm_mean_var_impl<BFloat16>(
            input.data<BFloat16>(),
            mean.data<BFloat16>(),
            variance.data<BFloat16>(),
            N, C, H, W
        );
    } else {
        throw std::runtime_error("BatchNorm2d only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    return {mean, variance};
}

// ============================================================================
// BatchNorm2d Normalization Kernel
// ============================================================================

// Normalize: (x - mean) / sqrt(variance + epsilon)
template<typename T>
void batchnorm_forward_impl(const T* input,
                            T* output,
                            const T* mean,
                            const T* variance,
                            T epsilon,
                            int64_t N,
                            int64_t C,
                            int64_t H,
                            int64_t W) {
    int64_t spatial_size = H * W;
    int64_t total_size = N * C * spatial_size;

    // Use nested loops to avoid expensive modulo operations for index decoding
    // Also increase parallelization threshold - OpenMP overhead exceeds benefit for small tensors
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    // Only parallelize for large tensors where threading benefit exceeds overhead
    int effective_threads = std::min({nthreads, static_cast<int>(total_size / 200000), 4});
    int final_threads = std::max(1, effective_threads);
    bool should_parallelize = total_size > 200000;
#else
    int final_threads = 1;
    bool should_parallelize = false;
#endif

    #pragma omp parallel for collapse(2) num_threads(final_threads) if(should_parallelize)
    for (int64_t n = 0; n < N; n++) {
        for (int64_t c = 0; c < C; c++) {
            T channel_mean = mean[c];
            T channel_var = variance[c];
            T invstd = T(1.0f) / safe_sqrt(channel_var + epsilon);

            int64_t base_idx = (n * C + c) * spatial_size;
            for (int64_t hw = 0; hw < spatial_size; hw++) {
                output[base_idx + hw] = (input[base_idx + hw] - channel_mean) * invstd;
            }
        }
    }
}

auto batchnorm2d_forward_kernel(const Tensor& input,
                               const Tensor& mean,
                               const Tensor& variance,
                               float epsilon) -> Tensor {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    if (input.dtype() == DType::Float32) {
        batchnorm_forward_impl<float>(
            input.data<float>(),
            output.data<float>(),
            mean.data<float>(),
            variance.data<float>(),
            epsilon,
            N, C, H, W
        );
    } else if (input.dtype() == DType::Float64) {
        batchnorm_forward_impl<double>(
            input.data<double>(),
            output.data<double>(),
            mean.data<double>(),
            variance.data<double>(),
            static_cast<double>(epsilon),
            N, C, H, W
        );
    } else if (input.dtype() == DType::Float16) {
        batchnorm_forward_impl<Float16>(
            input.data<Float16>(),
            output.data<Float16>(),
            mean.data<Float16>(),
            variance.data<Float16>(),
            Float16(static_cast<float>(epsilon)),
            N, C, H, W
        );
    } else if (input.dtype() == DType::BFloat16) {
        batchnorm_forward_impl<BFloat16>(
            input.data<BFloat16>(),
            output.data<BFloat16>(),
            mean.data<BFloat16>(),
            variance.data<BFloat16>(),
            BFloat16(static_cast<float>(epsilon)),
            N, C, H, W
        );
    } else {
        throw std::runtime_error("BatchNorm2d only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    return output;
}

// ============================================================================
// BatchNorm2d Affine Transform Kernel
// ============================================================================

#ifdef TENZOR_USE_ONEDNN
// Use shared lazy-init accessors from onednn_cache.hpp to avoid static
// thread_local initialization issues in dlopen'd libraries.

// --------------------------------------------------------------------------
// BatchNorm Primitive Caching (eliminates ~1-5ms primitive creation overhead)
// --------------------------------------------------------------------------
struct BatchNormCacheKey {
    int64_t N, C, H, W;

    bool operator==(const BatchNormCacheKey& other) const {
        return N == other.N && C == other.C && H == other.H && W == other.W;
    }
};

struct BatchNormCacheKeyHash {
    size_t operator()(const BatchNormCacheKey& k) const {
        size_t h = std::hash<int64_t>{}(k.N);
        h ^= std::hash<int64_t>{}(k.C) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(k.H) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(k.W) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct BatchNormCachedPrimitive {
    dnnl::batch_normalization_forward prim;
    dnnl::memory::desc src_md, sc_md;
};

static constexpr size_t BATCHNORM_CACHE_SIZE = 32;

class BatchNormPrimitiveCache {
public:
    std::shared_ptr<BatchNormCachedPrimitive> get(const BatchNormCacheKey& key) {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            lru_list_.remove(key);
            lru_list_.push_front(key);
            return it->second;
        }
        return nullptr;
    }

    void put(const BatchNormCacheKey& key, std::shared_ptr<BatchNormCachedPrimitive> value) {
        if (cache_.size() >= BATCHNORM_CACHE_SIZE) {
            auto evict_key = lru_list_.back();
            lru_list_.pop_back();
            cache_.erase(evict_key);
        }
        cache_[key] = value;
        lru_list_.push_front(key);
    }

private:
    std::unordered_map<BatchNormCacheKey, std::shared_ptr<BatchNormCachedPrimitive>, BatchNormCacheKeyHash> cache_;
    std::list<BatchNormCacheKey> lru_list_;
};

static thread_local BatchNormPrimitiveCache g_batchnorm_cache;

// oneDNN-accelerated BatchNorm2d Forward with Affine Transform (Float32 only)
// Provides 2-3x speedup over scalar implementation with primitive caching
// NOTE: Only used for very large tensors (>10M elements) due to high overhead
static bool batchnorm2d_forward_affine_onednn(
    const Tensor& input,
    Tensor& output,
    const Tensor& mean,
    const Tensor& variance,
    const Tensor& gamma,
    const Tensor& beta,
    float epsilon
) {
    // oneDNN only supports Float32 for now
    if (input.dtype() != DType::Float32) {
        return false;
    }

    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    try {
        auto& engine = get_onednn_engine();
        auto& stream = get_onednn_stream();

        // Create cache key
        BatchNormCacheKey cache_key{N, C, H, W};

        // Try to get cached primitive
        auto cached = g_batchnorm_cache.get(cache_key);

        if (!cached) {
            // Cache miss - create new primitive and cache it
            cached = std::make_shared<BatchNormCachedPrimitive>();

            // Memory descriptors
            dnnl::memory::dims src_dims = {N, C, H, W};
            cached->src_md = dnnl::memory::desc(src_dims, dnnl::memory::data_type::f32,
                                                  dnnl::memory::format_tag::nchw);

            // Scale/shift memory descriptor
            dnnl::memory::dims sc_dims = {C};
            cached->sc_md = dnnl::memory::desc(sc_dims, dnnl::memory::data_type::f32,
                                                 dnnl::memory::format_tag::a);

            // Create batch normalization primitive descriptor for inference
            auto bn_pd = dnnl::batch_normalization_forward::primitive_desc(
                engine,
                dnnl::prop_kind::forward_inference,
                cached->src_md,
                cached->src_md,
                epsilon,
                dnnl::normalization_flags::use_global_stats |
                dnnl::normalization_flags::use_scale |
                dnnl::normalization_flags::use_shift
            );
            cached->prim = dnnl::batch_normalization_forward(bn_pd);

            g_batchnorm_cache.put(cache_key, cached);
        }

        // Create memory objects with user data (fast - just wraps pointers)
        auto src_mem = dnnl::memory(cached->src_md, engine, const_cast<float*>(input.data<float>()));
        auto dst_mem = dnnl::memory(cached->src_md, engine, output.data<float>());
        auto scale_mem = dnnl::memory(cached->sc_md, engine, const_cast<float*>(gamma.data<float>()));
        auto shift_mem = dnnl::memory(cached->sc_md, engine, const_cast<float*>(beta.data<float>()));
        auto mean_mem = dnnl::memory(cached->sc_md, engine, const_cast<float*>(mean.data<float>()));
        auto var_mem = dnnl::memory(cached->sc_md, engine, const_cast<float*>(variance.data<float>()));

        // Execute cached primitive
        cached->prim.execute(stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DST, dst_mem},
            {DNNL_ARG_SCALE, scale_mem},
            {DNNL_ARG_SHIFT, shift_mem},
            {DNNL_ARG_MEAN, mean_mem},
            {DNNL_ARG_VARIANCE, var_mem}
        });

        stream.wait();
        return true;

    } catch (const dnnl::error& e) {
        // oneDNN error, fall back to scalar implementation
        return false;
    }
}
#endif

// SIMD-optimized BatchNorm forward for Float32
// Uses AVX2 FMA for maximum throughput
#ifdef __AVX2__
static void batchnorm_forward_affine_simd_f32(
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

    // Use fewer threads for memory-bound operations
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    int effective_threads = std::min({nthreads, static_cast<int>(total_size / 65536), 4});
    int final_threads = std::max(1, effective_threads);
#else
    int final_threads = 1;
#endif

    // Process by batch and channel
    #pragma omp parallel for collapse(2) num_threads(final_threads) if(total_size > 10000)
    for (int64_t n = 0; n < N; n++) {
        for (int64_t c = 0; c < C; c++) {
            __m256 vscale = _mm256_set1_ps(scale[c]);
            __m256 vbias = _mm256_set1_ps(bias_vec[c]);

            const float* in_ptr = input + (n * C + c) * spatial_size;
            float* out_ptr = output + (n * C + c) * spatial_size;

            int64_t hw = 0;
            // Process 8 floats at a time with AVX2 FMA
            for (; hw + 8 <= spatial_size; hw += 8) {
                __m256 vin = _mm256_loadu_ps(in_ptr + hw);
                __m256 vout = _mm256_fmadd_ps(vscale, vin, vbias);
                _mm256_storeu_ps(out_ptr + hw, vout);
            }
            // Handle remainder
            for (; hw < spatial_size; hw++) {
                out_ptr[hw] = scale[c] * in_ptr[hw] + bias_vec[c];
            }
        }
    }
}
#endif

// Combined normalization + affine: y = gamma * normalized + beta
// Optimized for cache efficiency and minimal redundant computation
template<typename T>
void batchnorm_forward_affine_impl(const T* input,
                                   T* output,
                                   const T* mean,
                                   const T* variance,
                                   const T* gamma,
                                   const T* beta,
                                   T epsilon,
                                   int64_t N,
                                   int64_t C,
                                   int64_t H,
                                   int64_t W) {
    int64_t spatial_size = H * W;
    int64_t total_size = N * C * spatial_size;

    // Precompute per-channel scale and bias: y = scale * x + bias
    // where scale = gamma / sqrt(var + eps), bias = beta - mean * scale
    std::vector<T> scale(C), bias(C);
    for (int64_t c = 0; c < C; c++) {
        T invstd = T(1.0f) / safe_sqrt(variance[c] + epsilon);
        scale[c] = gamma[c] * invstd;
        bias[c] = beta[c] - mean[c] * scale[c];
    }

    // Use fewer threads for memory-bound operations
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    int effective_threads = std::min({nthreads, static_cast<int>(total_size / 65536), 4});
    int final_threads = std::max(1, effective_threads);
#else
    int final_threads = 1;
#endif

    // Process by batch and channel for better cache locality
    // Each (n, c) slice is contiguous in memory
    #pragma omp parallel for collapse(2) num_threads(final_threads) if(total_size > 10000)
    for (int64_t n = 0; n < N; n++) {
        for (int64_t c = 0; c < C; c++) {
            T sc = scale[c];
            T bi = bias[c];
            int64_t base_idx = (n * C + c) * spatial_size;

            // Simple fused multiply-add over spatial dimensions
            for (int64_t hw = 0; hw < spatial_size; hw++) {
                output[base_idx + hw] = sc * input[base_idx + hw] + bi;
            }
        }
    }
}

auto batchnorm2d_forward_affine_kernel(const Tensor& input,
                                       const Tensor& mean,
                                       const Tensor& variance,
                                       const Tensor& gamma,
                                       const Tensor& beta,
                                       float epsilon) -> Tensor {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

#ifdef TENZOR_USE_ONEDNN
    // Only use oneDNN for very large tensors where it's actually beneficial
    // oneDNN has ~6ms overhead per call, so only use when compute > overhead
    int64_t total_elements = N * C * H * W;
    bool use_onednn = total_elements > 10000000;  // > 10M elements

    if (use_onednn && batchnorm2d_forward_affine_onednn(input, output, mean, variance, gamma, beta, epsilon)) {
        return output;
    }
    // Fall through to scalar implementation for smaller tensors
#endif

    if (input.dtype() == DType::Float32) {
#ifdef __AVX2__
        // Use SIMD-optimized version for Float32
        batchnorm_forward_affine_simd_f32(
            input.data<float>(),
            output.data<float>(),
            mean.data<float>(),
            variance.data<float>(),
            gamma.data<float>(),
            beta.data<float>(),
            epsilon,
            N, C, H, W
        );
#else
        batchnorm_forward_affine_impl<float>(
            input.data<float>(),
            output.data<float>(),
            mean.data<float>(),
            variance.data<float>(),
            gamma.data<float>(),
            beta.data<float>(),
            epsilon,
            N, C, H, W
        );
#endif
    } else if (input.dtype() == DType::Float64) {
        batchnorm_forward_affine_impl<double>(
            input.data<double>(),
            output.data<double>(),
            mean.data<double>(),
            variance.data<double>(),
            gamma.data<double>(),
            beta.data<double>(),
            static_cast<double>(epsilon),
            N, C, H, W
        );
    } else if (input.dtype() == DType::Float16) {
        batchnorm_forward_affine_impl<Float16>(
            input.data<Float16>(),
            output.data<Float16>(),
            mean.data<Float16>(),
            variance.data<Float16>(),
            gamma.data<Float16>(),
            beta.data<Float16>(),
            Float16(static_cast<float>(epsilon)),
            N, C, H, W
        );
    } else if (input.dtype() == DType::BFloat16) {
        batchnorm_forward_affine_impl<BFloat16>(
            input.data<BFloat16>(),
            output.data<BFloat16>(),
            mean.data<BFloat16>(),
            variance.data<BFloat16>(),
            gamma.data<BFloat16>(),
            beta.data<BFloat16>(),
            BFloat16(static_cast<float>(epsilon)),
            N, C, H, W
        );
    } else {
        throw std::runtime_error("BatchNorm2d only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    return output;
}

// ============================================================================
// BatchNorm2d Running Statistics Update Kernel
// ============================================================================

// Update running statistics: running = (1 - momentum) * running + momentum * batch
template<typename T>
void batchnorm_update_running_stats_impl(T* running_mean,
                                         T* running_var,
                                         const T* batch_mean,
                                         const T* batch_var,
                                         T momentum,
                                         int64_t C) {
    // Running stats update is very lightweight - no parallelization needed
    // Parallelizing would add more overhead than benefit
    for (int64_t c = 0; c < C; c++) {
        running_mean[c] = (T(1.0f) - momentum) * running_mean[c] + momentum * batch_mean[c];
        running_var[c] = (T(1.0f) - momentum) * running_var[c] + momentum * batch_var[c];
    }
}

auto batchnorm2d_update_running_stats_kernel(Tensor& running_mean,
                                             Tensor& running_var,
                                             const Tensor& batch_mean,
                                             const Tensor& batch_var,
                                             float momentum) -> void {
    int64_t C = batch_mean.shape()[0];

    if (running_mean.dtype() == DType::Float32) {
        batchnorm_update_running_stats_impl<float>(
            running_mean.data<float>(),
            running_var.data<float>(),
            batch_mean.data<float>(),
            batch_var.data<float>(),
            momentum,
            C
        );
    } else if (running_mean.dtype() == DType::Float64) {
        batchnorm_update_running_stats_impl<double>(
            running_mean.data<double>(),
            running_var.data<double>(),
            batch_mean.data<double>(),
            batch_var.data<double>(),
            static_cast<double>(momentum),
            C
        );
    } else if (running_mean.dtype() == DType::Float16) {
        batchnorm_update_running_stats_impl<Float16>(
            running_mean.data<Float16>(),
            running_var.data<Float16>(),
            batch_mean.data<Float16>(),
            batch_var.data<Float16>(),
            Float16(static_cast<float>(momentum)),
            C
        );
    } else if (running_mean.dtype() == DType::BFloat16) {
        batchnorm_update_running_stats_impl<BFloat16>(
            running_mean.data<BFloat16>(),
            running_var.data<BFloat16>(),
            batch_mean.data<BFloat16>(),
            batch_var.data<BFloat16>(),
            BFloat16(static_cast<float>(momentum)),
            C
        );
    } else {
        throw std::runtime_error("BatchNorm2d only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }
}

// ============================================================================
// BatchNorm2d Backward Kernels
// ============================================================================

// Compute gradients w.r.t input, gamma, and beta
template<typename T>
void batchnorm_backward_impl(const T* grad_output,
                            const T* input,
                            T* grad_input,
                            T* grad_gamma,
                            T* grad_beta,
                            const T* mean,
                            const T* variance,
                            const T* gamma,
                            T epsilon,
                            int64_t N,
                            int64_t C,
                            int64_t H,
                            int64_t W) {
    int64_t spatial_size = H * W;
    int64_t total_elements = N * spatial_size;

    // Check for division by zero
    if (total_elements == 0) {
        throw std::runtime_error("BatchNorm2d backward: Cannot compute gradients for empty tensor (total_elements = 0)");
    }

    // Use fewer threads to avoid OpenMP overhead
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    int effective_threads = std::min({nthreads, static_cast<int>(C), 4});
    int final_threads = std::max(1, effective_threads);
#else
    int final_threads = 1;
#endif

    // Compute grad_gamma and grad_beta for each channel
    #pragma omp parallel for num_threads(final_threads) if(C > 1)
    for (int64_t c = 0; c < C; c++) {
        T channel_mean = mean[c];
        T channel_var = variance[c];
        T invstd = T(1.0f) / safe_sqrt(channel_var + epsilon);

        // Compute grad_gamma = sum(grad_output * normalized)
        // Compute grad_beta = sum(grad_output)
        T sum_grad_gamma = T(0.0f);
        T sum_grad_beta = T(0.0f);

        for (int64_t n = 0; n < N; n++) {
            for (int64_t h = 0; h < H; h++) {
                for (int64_t w = 0; w < W; w++) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    T grad_out = grad_output[idx];
                    T normalized = (input[idx] - channel_mean) * invstd;

                    sum_grad_gamma += grad_out * normalized;
                    sum_grad_beta += grad_out;
                }
            }
        }

        grad_gamma[c] = sum_grad_gamma;
        grad_beta[c] = sum_grad_beta;
    }

    // Compute grad_input
    // Efficient formulation: grad_input = gamma * invstd * (grad_output - mean(grad_output) - normalized * mean(grad_output * normalized))
    #pragma omp parallel for num_threads(final_threads) if(C > 1)
    for (int64_t c = 0; c < C; c++) {
        T channel_mean = mean[c];
        T channel_var = variance[c];
        T invstd = T(1.0f) / safe_sqrt(channel_var + epsilon);
        T channel_gamma = gamma[c];

        // Compute auxiliary statistics
        T sum_grad = T(0.0f);
        T sum_grad_norm = T(0.0f);

        for (int64_t n = 0; n < N; n++) {
            for (int64_t h = 0; h < H; h++) {
                for (int64_t w = 0; w < W; w++) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    T grad_out = grad_output[idx];
                    T normalized = (input[idx] - channel_mean) * invstd;

                    sum_grad += grad_out;
                    sum_grad_norm += grad_out * normalized;
                }
            }
        }

        T mean_grad = sum_grad / T(static_cast<float>(total_elements));
        T mean_grad_norm = sum_grad_norm / T(static_cast<float>(total_elements));

        // Compute gradient w.r.t input
        for (int64_t n = 0; n < N; n++) {
            for (int64_t h = 0; h < H; h++) {
                for (int64_t w = 0; w < W; w++) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    T grad_out = grad_output[idx];
                    T normalized = (input[idx] - channel_mean) * invstd;

                    // Efficient backward formulation
                    T grad_normalized = grad_out - mean_grad - normalized * mean_grad_norm;
                    grad_input[idx] = channel_gamma * invstd * grad_normalized;
                }
            }
        }
    }
}

auto batchnorm2d_backward_kernel(const Tensor& grad_output,
                                 const Tensor& input,
                                 const Tensor& mean,
                                 const Tensor& variance,
                                 const Tensor& gamma,
                                 float epsilon) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    // Allocate output gradients
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor grad_input(shape_vec, input.dtype(), input.device());
    Tensor grad_gamma({C}, input.dtype(), input.device());
    Tensor grad_beta({C}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        batchnorm_backward_impl<float>(
            grad_output.data<float>(),
            input.data<float>(),
            grad_input.data<float>(),
            grad_gamma.data<float>(),
            grad_beta.data<float>(),
            mean.data<float>(),
            variance.data<float>(),
            gamma.data<float>(),
            epsilon,
            N, C, H, W
        );
    } else if (input.dtype() == DType::Float64) {
        batchnorm_backward_impl<double>(
            grad_output.data<double>(),
            input.data<double>(),
            grad_input.data<double>(),
            grad_gamma.data<double>(),
            grad_beta.data<double>(),
            mean.data<double>(),
            variance.data<double>(),
            gamma.data<double>(),
            static_cast<double>(epsilon),
            N, C, H, W
        );
    } else if (input.dtype() == DType::Float16) {
        batchnorm_backward_impl<Float16>(
            grad_output.data<Float16>(),
            input.data<Float16>(),
            grad_input.data<Float16>(),
            grad_gamma.data<Float16>(),
            grad_beta.data<Float16>(),
            mean.data<Float16>(),
            variance.data<Float16>(),
            gamma.data<Float16>(),
            Float16(static_cast<float>(epsilon)),
            N, C, H, W
        );
    } else if (input.dtype() == DType::BFloat16) {
        batchnorm_backward_impl<BFloat16>(
            grad_output.data<BFloat16>(),
            input.data<BFloat16>(),
            grad_input.data<BFloat16>(),
            grad_gamma.data<BFloat16>(),
            grad_beta.data<BFloat16>(),
            mean.data<BFloat16>(),
            variance.data<BFloat16>(),
            gamma.data<BFloat16>(),
            BFloat16(static_cast<float>(epsilon)),
            N, C, H, W
        );
    } else {
        throw std::runtime_error("BatchNorm2d only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    return {grad_input, grad_gamma, grad_beta};
}

} // namespace cpu
} // namespace tenzor
