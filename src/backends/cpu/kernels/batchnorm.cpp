#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/backends/cpu/simd.hpp"
#include "tenzor/backend/runtime_simd.hpp"
#include "half_operators.hpp"
#include <bit>
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

// Import shared Float16/BFloat16 operator overloads and safe_sqrt helpers
#include "half_operators.hpp"
#include "tenzor/utils/log.hpp"
#include <cstdlib>
#include "tenzor/backend/omp_thresholds.hpp"

namespace tenzor {
namespace cpu {

// AVX-512 forward declarations (defined in batchnorm_avx512.cpp)
namespace avx512 {
void batchnorm_mean_var_f32(const float*, float*, float*, int64_t, int64_t, int64_t, int64_t);
void batchnorm_forward_affine_f32(const float*, float*, const float*, const float*, const float*, const float*, float, int64_t, int64_t, int64_t, int64_t);
void batchnorm_normalize_f32(const float*, float*, const float*, const float*, float, int64_t, int64_t, int64_t, int64_t);
} // namespace avx512

// ============================================================================
// BatchNorm2d Mean/Variance Computation
// ============================================================================

#ifdef __AVX2__
// SIMD-optimized mean/variance computation for Float32
// Uses Welford's online algorithm for numerically stable variance computation.
// Each of the 8 AVX2 lanes maintains independent (mean, m2) accumulators,
// which are merged at the end using the parallel Welford combination formula.
// This avoids the catastrophic cancellation of E[X^2]-E[X]^2 when |mean| >> stddev.
static void batchnorm_mean_var_simd_f32(
    const float* __restrict input,
    float* __restrict mean,
    float* __restrict variance,
    int64_t N, int64_t C, int64_t H, int64_t W)
{
    int64_t spatial_size = H * W;
    int64_t total_elements = N * spatial_size;

    // Cap threads: each thread should have enough work to amortize overhead.
    constexpr int MIN_CHANNELS_PER_THREAD = 4;
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    int max_useful_threads = std::max(1, static_cast<int>(C / MIN_CHANNELS_PER_THREAD));
    int final_threads = std::min(nthreads, max_useful_threads);
#else
    int final_threads = 1;
#endif

    #pragma omp parallel for num_threads(final_threads) if(C > 1)
    for (int64_t c = 0; c < C; c++) {
        // Welford accumulators: each SIMD lane tracks its own (mean, m2, count).
        // All 8 lanes share the same count since they advance in lockstep.
        __m256 vmean = _mm256_setzero_ps();
        __m256 vm2 = _mm256_setzero_ps();
        int64_t lane_count = 0;

        // Single scalar Welford state for the (spatial_size % 8) tail of every
        // batch, accumulated in the SAME n-loop as the SIMD body so the input is
        // streamed exactly once (no second pass over all N batches).
        const int64_t simd_covered = (spatial_size / 8) * 8;
        double rem_n = 0.0, rem_mean = 0.0, rem_m2 = 0.0;

        for (int64_t n = 0; n < N; n++) {
            const float* ch_ptr = input + (n * C + c) * spatial_size;
            int64_t i = 0;

            // Process 8 consecutive floats at a time with Welford's update
            for (; i + 8 <= spatial_size; i += 8) {
                __m256 v = _mm256_loadu_ps(ch_ptr + i);
                lane_count++;
                __m256 vcount = _mm256_set1_ps(static_cast<float>(lane_count));
                __m256 delta = _mm256_sub_ps(v, vmean);
                vmean = _mm256_add_ps(vmean, _mm256_div_ps(delta, vcount));
                __m256 delta2 = _mm256_sub_ps(v, vmean);
                vm2 = _mm256_fmadd_ps(delta, delta2, vm2);
            }

            // Fold this batch's tail elements into the running scalar Welford
            // state using the already-loaded ch_ptr (single pass over the input).
            for (; i < spatial_size; i++) {
                rem_n += 1.0;
                double d = static_cast<double>(ch_ptr[i]) - rem_mean;
                rem_mean += d / rem_n;
                double d2 = static_cast<double>(ch_ptr[i]) - rem_mean;
                rem_m2 += d * d2;
            }
        }

        // Horizontally merge the 8 SIMD lanes using the parallel merge formula:
        //   combined_mean = (mean_a * n_a + mean_b * n_b) / (n_a + n_b)
        //   combined_m2 = m2_a + m2_b + delta^2 * n_a * n_b / (n_a + n_b)
        // All lanes have the same count (lane_count).
        alignas(32) float lane_means[8];
        alignas(32) float lane_m2s[8];
        _mm256_store_ps(lane_means, vmean);
        _mm256_store_ps(lane_m2s, vm2);

        // The horizontal merge and the final division run once per channel, so
        // accumulate them in double: float combined_n would round once the
        // per-channel element count exceeds 2^24 and float division of the
        // population variance would diverge from the scalar/double path.
        double combined_mean = lane_means[0];
        double combined_m2 = lane_m2s[0];
        double combined_n = static_cast<double>(lane_count);

        for (int lane = 1; lane < 8; lane++) {
            double n_b = static_cast<double>(lane_count);
            double total_n = combined_n + n_b;
            if (total_n == 0.0) continue;
            double delta = static_cast<double>(lane_means[lane]) - combined_mean;
            combined_mean = (combined_mean * combined_n + static_cast<double>(lane_means[lane]) * n_b) / total_n;
            combined_m2 = combined_m2 + static_cast<double>(lane_m2s[lane]) + delta * delta * combined_n * n_b / total_n;
            combined_n = total_n;
        }

        // Merge the scalar tail Welford state (accumulated in the single n-loop
        // above) into the combined SIMD state once, via the same parallel merge
        // formula. (void) silences an unused warning when spatial_size % 8 == 0.
        (void)simd_covered;
        if (rem_n > 0.0) {
            double total_n = combined_n + rem_n;
            double delta = rem_mean - combined_mean;
            combined_mean = (combined_mean * combined_n + rem_mean * rem_n) / total_n;
            combined_m2 = combined_m2 + rem_m2 + delta * delta * combined_n * rem_n / total_n;
            combined_n = total_n;
        }

        mean[c] = static_cast<float>(combined_mean);
        // BatchNorm uses population variance (divide by N, not N-1)
        variance[c] = (total_elements > 0) ? static_cast<float>(combined_m2 / static_cast<double>(total_elements)) : 0.0f;
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

    // Cap threads: each thread should have enough work to amortize overhead.
    // Minimum work-per-thread threshold: at least 4 channels per thread.
    constexpr int MIN_CHANNELS_PER_THREAD = 4;
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    int max_useful_threads = std::max(1, static_cast<int>(C / MIN_CHANNELS_PER_THREAD));
    int final_threads = std::min(nthreads, max_useful_threads);
#else
    int final_threads = 1;
#endif

    // Compute mean and variance for each channel using Welford's online algorithm.
    // Single-pass, numerically stable (no catastrophic cancellation for large N*H*W).
    // Accumulate Welford's stats in a float accumulator for half types
    // (Float16/BFloat16); running the mean/variance recurrence in half loses
    // precision and can overflow (Float16 max ~65504), diverging from the CPU
    // float reference. For float/double, Acc == T (behavior unchanged).
    using Acc = std::conditional_t<
        std::is_same_v<T, Float16> || std::is_same_v<T, BFloat16>, float, T>;

    #pragma omp parallel for num_threads(final_threads) if(C > 1)
    for (int64_t c = 0; c < C; c++) {
        Acc channel_mean = Acc(0);
        Acc m2 = Acc(0);  // sum of squared deviations from running mean
        int64_t count = 0;

        for (int64_t n = 0; n < N; n++) {
            const T* ch_ptr = input + (n * C + c) * spatial_size;
            for (int64_t hw = 0; hw < spatial_size; hw++) {
                ++count;
                Acc delta = static_cast<Acc>(ch_ptr[hw]) - channel_mean;
                // Divide in Acc precision; casting count through float first
                // rounds integer counts > 2^24, silently degrading the
                // Float64 (T=double) path on large per-channel element counts.
                channel_mean += delta / static_cast<Acc>(count);
                Acc delta2 = static_cast<Acc>(ch_ptr[hw]) - channel_mean;
                m2 += delta * delta2;
            }
        }

        mean[c] = static_cast<T>(channel_mean);
        variance[c] = static_cast<T>(m2 / static_cast<Acc>(total_elements));
    }
}

// Specialized Float16 version that accumulates in Float32 with Kahan summation
// Float16 has limited range (~65504 max), so summing many values can overflow.
// Kahan summation compensates for floating-point rounding errors in the sum.
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

    // Accumulate in Float32 with Kahan summation for numerical stability
    #pragma omp parallel for num_threads(final_threads) if(C > 1)
    for (int64_t c = 0; c < C; c++) {
        // Compute mean with Kahan summation in Float32
        float sum = 0.0f;
        float compensation = 0.0f;
        for (int64_t n = 0; n < N; n++) {
            const Float16* ch_ptr = input + (n * C + c) * spatial_size;
            for (int64_t hw = 0; hw < spatial_size; hw++) {
                float y = static_cast<float>(ch_ptr[hw]) - compensation;
                float t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
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

// Specialized BFloat16 version that accumulates in Float32 with Kahan summation.
// BFloat16 has only 8 mantissa bits — Kahan summation is essential for accurate mean.
template<>
void batchnorm_mean_var_impl<BFloat16>(const BFloat16* input,
                                        BFloat16* mean,
                                        BFloat16* variance,
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

    // Accumulate in Float32 with Kahan summation for numerical stability
    #pragma omp parallel for num_threads(final_threads) if(C > 1)
    for (int64_t c = 0; c < C; c++) {
        // Compute mean with Kahan summation in Float32
        float sum = 0.0f;
        float compensation = 0.0f;
        for (int64_t n = 0; n < N; n++) {
            const BFloat16* ch_ptr = input + (n * C + c) * spatial_size;
            for (int64_t hw = 0; hw < spatial_size; hw++) {
                float y = static_cast<float>(ch_ptr[hw]) - compensation;
                float t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            }
        }
        float channel_mean = sum / static_cast<float>(total_elements);
        mean[c] = BFloat16(channel_mean);

        // Compute variance with Float32 accumulation
        float sum_sq_diff = 0.0f;
        for (int64_t n = 0; n < N; n++) {
            const BFloat16* ch_ptr = input + (n * C + c) * spatial_size;
            for (int64_t hw = 0; hw < spatial_size; hw++) {
                float diff = static_cast<float>(ch_ptr[hw]) - channel_mean;
                sum_sq_diff += diff * diff;
            }
        }
        variance[c] = BFloat16(sum_sq_diff / static_cast<float>(total_elements));
    }
}

auto batchnorm2d_mean_var_kernel(const Tensor& input_orig) -> std::vector<Tensor> {
    // The mean/variance kernels index the raw NCHW buffer via (n*C+c)*H*W, so
    // the input must be contiguous for the linear offsets to be valid.
    Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("batchnorm2d_mean_var expects 4D input (NCHW)");
    }

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    // Guard the empty-tensor case here so every dtype/SIMD path behaves
    // identically. The scalar batchnorm_mean_var_impl<T> paths throw on
    // total_elements == 0, but the AVX2 (batchnorm_mean_var_simd_f32) and
    // AVX512 (avx512::batchnorm_mean_var_f32) Float32 paths have no such guard
    // and would silently return all-zero mean/variance, a backend-config
    // dependent behavioral divergence. Hoisting the check makes the kernel
    // dtype/SIMD-agnostic.
    if (N * H * W == 0) {
        throw std::runtime_error("BatchNorm2d: Cannot compute mean/variance for empty tensor (total_elements = 0)");
    }

    // Allocate output tensors
    Tensor mean({C}, input.dtype(), input.device());
    Tensor variance({C}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        if (::tenzor::backend::get_simd_features().avx512f) {
            avx512::batchnorm_mean_var_f32(
                input.data<float>(), mean.data<float>(), variance.data<float>(),
                N, C, H, W);
        } else {
#ifdef __AVX2__
            batchnorm_mean_var_simd_f32(
                input.data<float>(), mean.data<float>(), variance.data<float>(),
                N, C, H, W);
#else
            batchnorm_mean_var_impl<float>(
                input.data<float>(), mean.data<float>(), variance.data<float>(),
                N, C, H, W);
#endif
        }
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
            // Compute invstd and normalize in a float accumulator for half types
            // (Float16/BFloat16); narrowing invstd to half diverges from the CPU
            // float reference. For float/double, Acc == T (behavior unchanged).
            using Acc = std::conditional_t<
                std::is_same_v<T, Float16> || std::is_same_v<T, BFloat16>, float, T>;
            Acc channel_mean = static_cast<Acc>(mean[c]);
            Acc channel_var = static_cast<Acc>(variance[c]);
            Acc invstd = Acc(1.0) / safe_sqrt(channel_var + static_cast<Acc>(epsilon));

            int64_t base_idx = (n * C + c) * spatial_size;
            for (int64_t hw = 0; hw < spatial_size; hw++) {
                output[base_idx + hw] = static_cast<T>(
                    (static_cast<Acc>(input[base_idx + hw]) - channel_mean) * invstd);
            }
        }
    }
}

auto batchnorm2d_forward_kernel(const Tensor& input_orig,
                               const Tensor& mean_orig,
                               const Tensor& variance_orig,
                               float epsilon) -> Tensor {
    Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
    Tensor mean = mean_orig.is_contiguous() ? mean_orig : mean_orig.contiguous();
    Tensor variance = variance_orig.is_contiguous() ? variance_orig : variance_orig.contiguous();
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
    DType dtype;    // src/dst element type (gamma/beta/mean/var are always F32)
    uint32_t eps_bits;  // bit-cast of float epsilon — avoids NaN-equality concerns

    bool operator==(const BatchNormCacheKey& other) const {
        return N == other.N && C == other.C && H == other.H && W == other.W
            && dtype    == other.dtype
            && eps_bits == other.eps_bits;
    }
};

struct BatchNormCacheKeyHash {
    size_t operator()(const BatchNormCacheKey& k) const {
        size_t h = std::hash<int64_t>{}(k.N);
        h ^= std::hash<int64_t>{}(k.C)                                 + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(k.H)                                 + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(k.W)                                 + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(static_cast<int64_t>(k.dtype))       + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(k.eps_bits)                         + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct BatchNormCachedPrimitive {
    dnnl::batch_normalization_forward prim;
    dnnl::memory::desc src_md, sc_md;
};

static constexpr size_t BATCHNORM_CACHE_SIZE = 32;

using BatchNormPrimitiveCache = OneDNNPrimitiveCache<BatchNormCacheKey, BatchNormCachedPrimitive, BatchNormCacheKeyHash, BATCHNORM_CACHE_SIZE>;

static thread_local BatchNormPrimitiveCache g_batchnorm_cache;

// W.6: register a thread-local clear-callback. Invoked by clear_dnnl_cache().
namespace {
void clear_local_batchnorm_cache() { g_batchnorm_cache.clear(); }
struct BatchNormCacheClearRegistrar {
    BatchNormCacheClearRegistrar() {
        ::tenzor::cpu::register_dnnl_cache_clear_callback(&clear_local_batchnorm_cache);
    }
};
static BatchNormCacheClearRegistrar g_batchnorm_cache_clear_registrar;
}

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
    // Map src/dst dtype to oneDNN. F32/F16/BF16 are natively supported; other
    // dtypes fall through to the templated generic path. Gamma/beta/mean/var
    // stay F32 (oneDNN's convention for scale/shift/stats).
    dnnl::memory::data_type src_dt;
    switch (input.dtype()) {
        case DType::Float32:  src_dt = dnnl::memory::data_type::f32; break;
        case DType::Float16:  src_dt = dnnl::memory::data_type::f16; break;
        case DType::BFloat16: src_dt = dnnl::memory::data_type::bf16; break;
        default: return false;
    }

    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    try {
        auto& engine = get_onednn_engine();
        auto& stream = get_onednn_stream();

        // Cache key includes dtype so F32/F16/BF16 primitives don't collide.
        BatchNormCacheKey cache_key{N, C, H, W, input.dtype(),
                                    std::bit_cast<uint32_t>(epsilon)};

        // Try to get cached primitive
        auto cached = g_batchnorm_cache.get(cache_key);

        if (!cached) {
            // Cache miss - create new primitive and cache it
            cached = std::make_shared<BatchNormCachedPrimitive>();

            // Memory descriptors
            dnnl::memory::dims src_dims = {N, C, H, W};
            cached->src_md = dnnl::memory::desc(src_dims, src_dt,
                                                  dnnl::memory::format_tag::nchw);

            // Scale/shift memory descriptor — always F32
            dnnl::memory::dims sc_dims = {C};
            cached->sc_md = dnnl::memory::desc(sc_dims, dnnl::memory::data_type::f32,
                                                 dnnl::memory::format_tag::a);

            // Create batch normalization primitive descriptor for inference.
            // If oneDNN can't synthesize this on the current CPU (e.g. no AVX2
            // for BF16), construction throws dnnl::error and we fall through.
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

        // Create memory objects with user data (fast - just wraps pointers).
        // src/dst use raw void* via data_ptr() because element type may be F16/BF16.
        auto src_mem = dnnl::memory(cached->src_md, engine,
                                    const_cast<void*>(input.storage()->data()));
        auto dst_mem = dnnl::memory(cached->src_md, engine, output.storage()->data());
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
        // TENZOR_STRICT_BACKEND=1 makes oneDNN failures unrecoverable so silent
        // fallback to the scalar implementation cannot mask a real bug.
        if (const char* s = std::getenv("TENZOR_STRICT_BACKEND"); s && *s && *s != '0') {
            throw std::runtime_error(
                std::string("[BatchNorm] oneDNN forward failed "
                            "(TENZOR_STRICT_BACKEND=1): ") +
                e.what());
        }
        TENZOR_LOG_WARN("[BatchNorm] oneDNN forward failed ({}); using scalar fallback",
                        e.what());
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
    #pragma omp parallel for collapse(2) num_threads(final_threads) if(total_size > ::tenzor::OmpThresholds::medium())
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
    // Compute scale/bias in a float accumulator for half types (Float16/BFloat16);
    // half-precision invstd/scale/bias diverge from the CPU float reference.
    using Acc = std::conditional_t<
        std::is_same_v<T, Float16> || std::is_same_v<T, BFloat16>, float, T>;
    std::vector<Acc> scale(C), bias(C);
    for (int64_t c = 0; c < C; c++) {
        Acc invstd = Acc(1.0) / safe_sqrt(static_cast<Acc>(variance[c]) + static_cast<Acc>(epsilon));
        scale[c] = static_cast<Acc>(gamma[c]) * invstd;
        bias[c] = static_cast<Acc>(beta[c]) - static_cast<Acc>(mean[c]) * scale[c];
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
    #pragma omp parallel for collapse(2) num_threads(final_threads) if(total_size > ::tenzor::OmpThresholds::medium())
    for (int64_t n = 0; n < N; n++) {
        for (int64_t c = 0; c < C; c++) {
            Acc sc = scale[c];
            Acc bi = bias[c];
            int64_t base_idx = (n * C + c) * spatial_size;

            // Simple fused multiply-add over spatial dimensions
            for (int64_t hw = 0; hw < spatial_size; hw++) {
                output[base_idx + hw] = static_cast<T>(sc * static_cast<Acc>(input[base_idx + hw]) + bi);
            }
        }
    }
}

auto batchnorm2d_forward_affine_kernel(const Tensor& input_orig,
                                       const Tensor& mean_orig,
                                       const Tensor& variance_orig,
                                       const Tensor& gamma_orig,
                                       const Tensor& beta_orig,
                                       float epsilon) -> Tensor {
    // Both the oneDNN path (which builds an nchw descriptor over the raw
    // storage) and the scalar/SIMD paths index linear NCHW offsets, so all
    // inputs must be contiguous.
    Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
    Tensor mean = mean_orig.is_contiguous() ? mean_orig : mean_orig.contiguous();
    Tensor variance = variance_orig.is_contiguous() ? variance_orig : variance_orig.contiguous();
    Tensor gamma = gamma_orig.is_contiguous() ? gamma_orig : gamma_orig.contiguous();
    Tensor beta = beta_orig.is_contiguous() ? beta_orig : beta_orig.contiguous();
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
        if (::tenzor::backend::get_simd_features().avx512f) {
            avx512::batchnorm_forward_affine_f32(
                input.data<float>(), output.data<float>(),
                mean.data<float>(), variance.data<float>(),
                gamma.data<float>(), beta.data<float>(),
                epsilon, N, C, H, W);
        } else {
#ifdef __AVX2__
            batchnorm_forward_affine_simd_f32(
                input.data<float>(), output.data<float>(),
                mean.data<float>(), variance.data<float>(),
                gamma.data<float>(), beta.data<float>(),
                epsilon, N, C, H, W);
#else
            batchnorm_forward_affine_impl<float>(
                input.data<float>(), output.data<float>(),
                mean.data<float>(), variance.data<float>(),
                gamma.data<float>(), beta.data<float>(),
                epsilon, N, C, H, W);
#endif
        }
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
    // Parallelizing would add more overhead than benefit.
    // Compute the EMA blend in a float accumulator for half types
    // (Float16/BFloat16): performing the multiplies/adds in half precision
    // rounds every step (BFloat16 has only 8 mantissa bits), drifts from the
    // float reference, and can saturate large running_var toward Float16 inf.
    // For float/double, Acc == T (behavior unchanged).
    using Acc = std::conditional_t<
        std::is_same_v<T, Float16> || std::is_same_v<T, BFloat16>, float, T>;
    const Acc m = static_cast<Acc>(momentum);
    const Acc one_minus_m = Acc(1) - m;
    for (int64_t c = 0; c < C; c++) {
        running_mean[c] = static_cast<T>(
            one_minus_m * static_cast<Acc>(running_mean[c]) + m * static_cast<Acc>(batch_mean[c]));
        running_var[c] = static_cast<T>(
            one_minus_m * static_cast<Acc>(running_var[c]) + m * static_cast<Acc>(batch_var[c]));
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

// Accumulation type trait: Float16/BFloat16 accumulate in float for precision.
template<typename T> struct bn_acc { using type = T; };
template<> struct bn_acc<Float16> { using type = float; };
template<> struct bn_acc<BFloat16> { using type = float; };

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
    using Acc = typename bn_acc<T>::type;

    int64_t spatial_size = H * W;
    int64_t total_elements = N * spatial_size;

    // Check for division by zero
    if (total_elements == 0) {
        throw std::runtime_error("BatchNorm2d backward: Cannot compute gradients for empty tensor (total_elements = 0)");
    }

    // Cap threads: each thread should have enough work to amortize overhead.
    // Minimum work-per-thread threshold: at least 4 channels per thread.
    constexpr int MIN_CHANNELS_PER_THREAD = 4;
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    int max_useful_threads = std::max(1, static_cast<int>(C / MIN_CHANNELS_PER_THREAD));
    int final_threads = std::min(nthreads, max_useful_threads);
#else
    int final_threads = 1;
#endif

    // Compute grad_gamma and grad_beta for each channel
    #pragma omp parallel for num_threads(final_threads) if(C > 1)
    for (int64_t c = 0; c < C; c++) {
        Acc channel_mean = static_cast<Acc>(mean[c]);
        Acc channel_var = static_cast<Acc>(variance[c]);
        Acc invstd = Acc(1.0) / std::sqrt(channel_var + static_cast<Acc>(epsilon));

        // Compute grad_gamma = sum(grad_output * normalized)
        // Compute grad_beta = sum(grad_output)
        // Accumulate in Acc (float for FP16/BF16) to avoid precision loss
        Acc sum_grad_gamma = Acc(0);
        Acc sum_grad_beta = Acc(0);

        for (int64_t n = 0; n < N; n++) {
            for (int64_t h = 0; h < H; h++) {
                for (int64_t w = 0; w < W; w++) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    Acc grad_out = static_cast<Acc>(grad_output[idx]);
                    Acc normalized = (static_cast<Acc>(input[idx]) - channel_mean) * invstd;

                    sum_grad_gamma += grad_out * normalized;
                    sum_grad_beta += grad_out;
                }
            }
        }

        grad_gamma[c] = static_cast<T>(sum_grad_gamma);
        grad_beta[c] = static_cast<T>(sum_grad_beta);
    }

    // Compute grad_input
    // Efficient formulation: grad_input = gamma * invstd * (grad_output - mean(grad_output) - normalized * mean(grad_output * normalized))
    #pragma omp parallel for num_threads(final_threads) if(C > 1)
    for (int64_t c = 0; c < C; c++) {
        Acc channel_mean = static_cast<Acc>(mean[c]);
        Acc channel_var = static_cast<Acc>(variance[c]);
        Acc invstd = Acc(1.0) / std::sqrt(channel_var + static_cast<Acc>(epsilon));
        Acc channel_gamma = static_cast<Acc>(gamma[c]);

        // Compute auxiliary statistics in Acc precision
        Acc sum_grad = Acc(0);
        Acc sum_grad_norm = Acc(0);

        for (int64_t n = 0; n < N; n++) {
            for (int64_t h = 0; h < H; h++) {
                for (int64_t w = 0; w < W; w++) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    Acc grad_out = static_cast<Acc>(grad_output[idx]);
                    Acc normalized = (static_cast<Acc>(input[idx]) - channel_mean) * invstd;

                    sum_grad += grad_out;
                    sum_grad_norm += grad_out * normalized;
                }
            }
        }

        Acc mean_grad = sum_grad / static_cast<Acc>(total_elements);
        Acc mean_grad_norm = sum_grad_norm / static_cast<Acc>(total_elements);

        // Compute gradient w.r.t input
        for (int64_t n = 0; n < N; n++) {
            for (int64_t h = 0; h < H; h++) {
                for (int64_t w = 0; w < W; w++) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    Acc grad_out = static_cast<Acc>(grad_output[idx]);
                    Acc normalized = (static_cast<Acc>(input[idx]) - channel_mean) * invstd;

                    // Efficient backward formulation
                    Acc grad_normalized = grad_out - mean_grad - normalized * mean_grad_norm;
                    grad_input[idx] = static_cast<T>(channel_gamma * invstd * grad_normalized);
                }
            }
        }
    }
}

auto batchnorm2d_backward_kernel(const Tensor& grad_output_orig,
                                 const Tensor& input_orig,
                                 const Tensor& mean_orig,
                                 const Tensor& variance_orig,
                                 const Tensor& gamma_orig,
                                 float epsilon) -> std::vector<Tensor> {
    // Backward indexes grad_output/input as linear NCHW buffers and reads the
    // per-channel stats by [c], so all operands must be contiguous.
    Tensor grad_output = grad_output_orig.is_contiguous() ? grad_output_orig : grad_output_orig.contiguous();
    Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
    Tensor mean = mean_orig.is_contiguous() ? mean_orig : mean_orig.contiguous();
    Tensor variance = variance_orig.is_contiguous() ? variance_orig : variance_orig.contiguous();
    Tensor gamma = gamma_orig.is_contiguous() ? gamma_orig : gamma_orig.contiguous();
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
