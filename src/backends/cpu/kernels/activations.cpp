#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/backend/dtype_dispatch.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/utils/log.hpp"   // TENZOR_LOG_WARN (F.4)
#include "tenzor/ops/creation.hpp"
#include <cstdlib>  // std::getenv for TENZOR_STRICT_BACKEND
#include "simd_elementwise.hpp"
#include "simd_fast_math.hpp"
#include "float16_simd.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <omp.h>

// Intel oneDNN for optimized activations (5-20x faster for large tensors)
#ifdef TENZOR_USE_ONEDNN
#include <dnnl.hpp>
#include "onednn_cache.hpp"
#include <unordered_map>
#include <list>
#include <mutex>
#endif

// Use adaptive OpenMP thresholds scaled to thread count
#include "tenzor/backend/omp_thresholds.hpp"  // unified (F.5)
#define ACTIVATION_OMP_THRESHOLD (::tenzor::OmpThresholds::simple())

// oneDNN threshold - use oneDNN for tensors larger than this.
// Lowered from 256K to 64K — primitive caching amortizes setup cost.
constexpr size_t ONEDNN_ACTIVATION_THRESHOLD = 65536;  // 64K elements

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#define TENZOR_HAS_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define TENZOR_HAS_AVX2 1
#elif defined(__SSE2__)
#include <emmintrin.h>
#define TENZOR_HAS_SSE2 1
#endif

namespace tenzor {
namespace cpu {

// ============================================================================
// oneDNN Engine and Stream — use shared lazy-init accessors from onednn_cache.hpp
// to avoid static thread_local initialization issues in dlopen'd libraries.
// ============================================================================
#ifdef TENZOR_USE_ONEDNN

// --------------------------------------------------------------------------
// Eltwise Primitive Caching (eliminates ~1-5ms primitive creation overhead)
// --------------------------------------------------------------------------
struct EltwiseCacheKey {
    dnnl::algorithm algo;
    size_t n;
    float alpha, beta;
    bool is_backward;
    DType dtype;

    bool operator==(const EltwiseCacheKey& other) const {
        return algo == other.algo && n == other.n &&
               alpha == other.alpha && beta == other.beta &&
               is_backward == other.is_backward && dtype == other.dtype;
    }
};

struct EltwiseCacheKeyHash {
    size_t operator()(const EltwiseCacheKey& k) const {
        size_t h = std::hash<int>{}(static_cast<int>(k.algo));
        h ^= std::hash<size_t>{}(k.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<float>{}(k.alpha) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<float>{}(k.beta) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(k.is_backward) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(static_cast<int>(k.dtype)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct EltwiseCachedPrimitive {
    dnnl::eltwise_forward fwd_prim;
    dnnl::eltwise_backward bwd_prim;
    dnnl::memory::desc data_md;
    bool is_backward;
};

static constexpr size_t ELTWISE_CACHE_SIZE = 64;

using EltwisePrimitiveCache = OneDNNPrimitiveCache<EltwiseCacheKey, EltwiseCachedPrimitive, EltwiseCacheKeyHash, ELTWISE_CACHE_SIZE>;

static thread_local EltwisePrimitiveCache g_eltwise_cache;

// --------------------------------------------------------------------------
// Softmax Primitive Caching
// --------------------------------------------------------------------------
struct SoftmaxCacheKey {
    std::vector<int64_t> dims;
    int64_t axis;
    DType dtype;

    bool operator==(const SoftmaxCacheKey& other) const {
        return dims == other.dims && axis == other.axis && dtype == other.dtype;
    }
};

struct SoftmaxCacheKeyHash {
    size_t operator()(const SoftmaxCacheKey& k) const {
        size_t h = std::hash<int64_t>{}(k.axis);
        for (auto d : k.dims) {
            h ^= std::hash<int64_t>{}(d) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        h ^= std::hash<int>{}(static_cast<int>(k.dtype)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct SoftmaxCachedPrimitive {
    dnnl::softmax_forward prim;
    dnnl::memory::desc data_md;
};

static constexpr size_t SOFTMAX_CACHE_SIZE = 32;

using SoftmaxPrimitiveCache = OneDNNPrimitiveCache<SoftmaxCacheKey, SoftmaxCachedPrimitive, SoftmaxCacheKeyHash, SOFTMAX_CACHE_SIZE>;

static thread_local SoftmaxPrimitiveCache g_softmax_cache;

// Register thread-local clear-callbacks so clear_dnnl_cache() actually reclaims
// these primitive caches (audit C1 — previously only conv2d/batchnorm did).
namespace {
void clear_local_eltwise_cache() { g_eltwise_cache.clear(); }
void clear_local_softmax_cache() { g_softmax_cache.clear(); }
struct ActivationsCacheClearRegistrar {
    ActivationsCacheClearRegistrar() {
        ::tenzor::cpu::register_dnnl_cache_clear_callback(&clear_local_eltwise_cache);
        ::tenzor::cpu::register_dnnl_cache_clear_callback(&clear_local_softmax_cache);
    }
};
static ActivationsCacheClearRegistrar g_activations_cache_clear_registrar;
}

// Helper: Execute oneDNN eltwise forward operation with caching
// Returns true if successful, false if should fall back to SIMD
static bool onednn_eltwise_forward(
    const float* input, float* output, size_t n,
    dnnl::algorithm alg, float alpha = 0.0f, float beta = 0.0f) {

    if (n < ONEDNN_ACTIVATION_THRESHOLD) {
        return false;  // Fall back to SIMD for small tensors
    }

    try {
        auto& engine = get_onednn_engine();
        auto& stream = get_onednn_stream();

        // Create cache key
        EltwiseCacheKey cache_key{alg, n, alpha, beta, false, DType::Float32};

        // Try to get cached primitive
        auto cached = g_eltwise_cache.get(cache_key);

        if (!cached) {
            // Cache miss - create new primitive and cache it
            cached = std::make_shared<EltwiseCachedPrimitive>();
            cached->is_backward = false;

            dnnl::memory::dims dims = {static_cast<dnnl::memory::dim>(n)};
            cached->data_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::a);

            auto eltwise_pd = dnnl::eltwise_forward::primitive_desc(
                engine, dnnl::prop_kind::forward_inference, alg,
                cached->data_md, cached->data_md, alpha, beta);
            cached->fwd_prim = dnnl::eltwise_forward(eltwise_pd);

            g_eltwise_cache.put(cache_key, cached);
        }

        // Create memory objects with user data (fast - just wraps pointers)
        auto src_mem = dnnl::memory(cached->data_md, engine, const_cast<float*>(input));
        auto dst_mem = dnnl::memory(cached->data_md, engine, output);

        // Execute cached primitive
        cached->fwd_prim.execute(stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DST, dst_mem}
        });
        stream.wait();

        return true;
    } catch (const dnnl::error& e) {
        // Audit item F.4: log + honour TENZOR_STRICT_BACKEND so a real
        // oneDNN failure cannot be masked by a silent SIMD fallback.
        if (const char* s = std::getenv("TENZOR_STRICT_BACKEND"); s && *s && *s != '0') {
            throw std::runtime_error(
                std::string("[Activation] oneDNN primitive failed "
                            "(TENZOR_STRICT_BACKEND=1): ") + e.what());
        }
        TENZOR_LOG_WARN("[Activation] oneDNN primitive failed ({}); using SIMD fallback",
                        e.what());
        return false;
    }
}

// Helper: Execute oneDNN softmax forward with caching
static bool onednn_softmax_forward(
    const float* input, float* output,
    const std::vector<int64_t>& shape, int64_t axis) {

    size_t n = 1;
    for (auto s : shape) n *= s;
    if (n < ONEDNN_ACTIVATION_THRESHOLD) {
        return false;
    }

    try {
        auto& engine = get_onednn_engine();
        auto& stream = get_onednn_stream();

        // Create cache key
        SoftmaxCacheKey cache_key{shape, axis, DType::Float32};

        auto cached = g_softmax_cache.get(cache_key);

        if (!cached) {
            cached = std::make_shared<SoftmaxCachedPrimitive>();

            // Convert shape to dnnl dims
            dnnl::memory::dims dims(shape.begin(), shape.end());

            // Use plain format
            dnnl::memory::format_tag tag;
            switch (shape.size()) {
                case 1: tag = dnnl::memory::format_tag::a; break;
                case 2: tag = dnnl::memory::format_tag::ab; break;
                case 3: tag = dnnl::memory::format_tag::abc; break;
                case 4: tag = dnnl::memory::format_tag::abcd; break;
                default: return false;  // Unsupported
            }

            cached->data_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, tag);

            auto softmax_pd = dnnl::softmax_forward::primitive_desc(
                engine, dnnl::prop_kind::forward_inference, dnnl::algorithm::softmax_accurate,
                cached->data_md, cached->data_md, static_cast<int>(axis));
            cached->prim = dnnl::softmax_forward(softmax_pd);

            g_softmax_cache.put(cache_key, cached);
        }

        auto src_mem = dnnl::memory(cached->data_md, engine, const_cast<float*>(input));
        auto dst_mem = dnnl::memory(cached->data_md, engine, output);

        cached->prim.execute(stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DST, dst_mem}
        });
        stream.wait();

        return true;
    } catch (const dnnl::error& e) {
        if (const char* s = std::getenv("TENZOR_STRICT_BACKEND"); s && *s && *s != '0') {
            throw std::runtime_error(
                std::string("[Softmax] oneDNN forward failed "
                            "(TENZOR_STRICT_BACKEND=1): ") + e.what());
        }
        TENZOR_LOG_WARN("[Softmax] oneDNN forward failed ({}); using SIMD fallback",
                        e.what());
        return false;
    }
}
#endif // TENZOR_USE_ONEDNN

// ============================================================================
// ReLU Activation
// ============================================================================

// Forward: max(0, x)
// Refactored to use TENZOR_DISPATCH_FLOAT_AND_HALF + elementwise_unary helper.
// Float32 path retains oneDNN fast path via if constexpr.
auto relu_kernel(const Tensor& input_raw) -> Tensor {
    // Flat-pointer iteration requires contiguous input; a non-contiguous
    // slice/transpose/expand view would otherwise read the wrong elements.
    auto input = input_raw.contiguous();
    auto output = Tensor(std::vector<int64_t>(input.shape().begin(), input.shape().end()), input.dtype(), input.device());
    size_t n = input.numel();

    // Try oneDNN fast path for Float32 (5-10x faster for large inputs)
#ifdef TENZOR_USE_ONEDNN
    if (input.dtype() == DType::Float32) {
        if (onednn_eltwise_forward(input.data<float>(), output.data<float>(), n,
                                    dnnl::algorithm::eltwise_relu)) {
            return output;
        }
    }
#endif
    // Generic dtype-dispatched path using elementwise_unary helper
    TENZOR_DISPATCH_FLOAT_AND_HALF(input.dtype(), "relu", [&]() {
        const scalar_t* in_data = input.data<scalar_t>();
        scalar_t* out_data = output.data<scalar_t>();
        cpu::elementwise_unary<scalar_t>(in_data, out_data, n,
            [](auto x) { return std::max(decltype(x)(0), x); });
    });

    return output;
}

// Backward: grad_out * (x > 0)
auto relu_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw) -> Tensor {
    // Flat-pointer iteration requires contiguous inputs.
    auto grad_output = grad_output_raw.contiguous();
    auto input = input_raw.contiguous();
    auto grad_input = zeros_like(input);

    if (input.dtype() == DType::Float32) {
        const float* grad_out_data = grad_output.data<float>();
        const float* in_data = input.data<float>();
        float* grad_in_data = grad_input.data<float>();
        size_t n = input.numel();

#ifdef TENZOR_HAS_AVX512
        const size_t simd_width = 16;
        const size_t simd_end = (n / simd_width) * simd_width;
        __m512 zero = _mm512_setzero_ps();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m512 x = _mm512_loadu_ps(in_data + i);
            __m512 grad_out = _mm512_loadu_ps(grad_out_data + i);
            __mmask16 mask = _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ);
            __m512 grad_in = _mm512_mask_blend_ps(mask, zero, grad_out);
            _mm512_storeu_ps(grad_in_data + i, grad_in);
        }

        for (size_t i = simd_end; i < n; ++i) {
            grad_in_data[i] = grad_out_data[i] * (in_data[i] > 0.0f ? 1.0f : 0.0f);
        }
#elif defined(TENZOR_HAS_AVX2)
        const size_t simd_width = 8;
        const size_t simd_end = (n / simd_width) * simd_width;
        __m256 zero = _mm256_setzero_ps();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256 x = _mm256_loadu_ps(in_data + i);
            __m256 grad_out = _mm256_loadu_ps(grad_out_data + i);
            __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
            __m256 grad_in = _mm256_and_ps(mask, grad_out);
            _mm256_storeu_ps(grad_in_data + i, grad_in);
        }

        for (size_t i = simd_end; i < n; ++i) {
            grad_in_data[i] = grad_out_data[i] * (in_data[i] > 0.0f ? 1.0f : 0.0f);
        }
#else
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            grad_in_data[i] = grad_out_data[i] * (in_data[i] > 0.0f ? 1.0f : 0.0f);
        }
#endif
    } else if (input.dtype() == DType::Float64) {
        const double* grad_out_data = grad_output.data<double>();
        const double* in_data = input.data<double>();
        double* grad_in_data = grad_input.data<double>();
        size_t n = input.numel();

#ifdef TENZOR_HAS_AVX512
        const size_t simd_width = 8; // AVX-512: 8 doubles
        const size_t simd_end = (n / simd_width) * simd_width;
        __m512d zero_d = _mm512_setzero_pd();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m512d x = _mm512_loadu_pd(in_data + i);
            __m512d grad = _mm512_loadu_pd(grad_out_data + i);
            __mmask8 mask = _mm512_cmp_pd_mask(x, zero_d, _CMP_GT_OQ);
            __m512d result = _mm512_maskz_mov_pd(mask, grad);
            _mm512_storeu_pd(grad_in_data + i, result);
        }
        for (size_t i = simd_end; i < n; ++i) {
            grad_in_data[i] = grad_out_data[i] * (in_data[i] > 0.0 ? 1.0 : 0.0);
        }
#elif defined(TENZOR_HAS_AVX2)
        const size_t simd_width = 4; // AVX2: 4 doubles
        const size_t simd_end = (n / simd_width) * simd_width;
        __m256d zero_d = _mm256_setzero_pd();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256d x = _mm256_loadu_pd(in_data + i);
            __m256d grad = _mm256_loadu_pd(grad_out_data + i);
            __m256d mask = _mm256_cmp_pd(x, zero_d, _CMP_GT_OQ);
            __m256d result = _mm256_and_pd(mask, grad);
            _mm256_storeu_pd(grad_in_data + i, result);
        }
        for (size_t i = simd_end; i < n; ++i) {
            grad_in_data[i] = grad_out_data[i] * (in_data[i] > 0.0 ? 1.0 : 0.0);
        }
#else
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            grad_in_data[i] = grad_out_data[i] * (in_data[i] > 0.0 ? 1.0 : 0.0);
        }
#endif
    } else if (input.dtype() == DType::Float16) {
        const Float16* grad_out_data = grad_output.data<Float16>();
        const Float16* in_data = input.data<Float16>();
        Float16* grad_in_data = grad_input.data<Float16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float in_val = static_cast<float>(in_data[i]);
            float grad_out_val = static_cast<float>(grad_out_data[i]);
            grad_in_data[i] = Float16(grad_out_val * (in_val > 0.0f ? 1.0f : 0.0f));
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* grad_out_data = grad_output.data<BFloat16>();
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* grad_in_data = grad_input.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float in_val = static_cast<float>(in_data[i]);
            float grad_out_val = static_cast<float>(grad_out_data[i]);
            grad_in_data[i] = BFloat16(grad_out_val * (in_val > 0.0f ? 1.0f : 0.0f));
        }
    } else {
        throw std::runtime_error("ReLU backward only supports Float32, Float64, Float16, and BFloat16");
    }

    return grad_input;
}

// ============================================================================
// Sigmoid Activation
// ============================================================================

// Forward: 1 / (1 + exp(-x))
// SIMD + OpenMP optimized using fast_math sigmoid approximation
auto sigmoid_kernel(const Tensor& input_raw) -> Tensor {
    // audit-2026-05-03 bug #15 mirror: ensure contiguous input. Same root
    // cause as the sum_kernel fix — flat-pointer iteration over a
    // non-contiguous slice/expand view reads the wrong logical elements.
    auto input = input_raw.contiguous();
    auto output = Tensor(std::vector<int64_t>(input.shape().begin(), input.shape().end()), input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();
        size_t n = input.numel();

#ifdef TENZOR_USE_ONEDNN
        // Try oneDNN for large tensors (significantly faster than SIMD for sigmoid)
        if (onednn_eltwise_forward(in_data, out_data, n, dnnl::algorithm::eltwise_logistic)) {
            return output;
        }
#endif
        // Fall back to SIMD implementation
#ifdef TENZOR_HAS_AVX512
        const size_t simd_width = 16;
        const size_t simd_end = (n / simd_width) * simd_width;

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m512 x = _mm512_loadu_ps(in_data + i);
            __m512 result = fast_math::sigmoid_avx512(x);
            _mm512_storeu_ps(out_data + i, result);
        }

        // Scalar remainder
        for (size_t i = simd_end; i < n; ++i) {
            float x = in_data[i];
            out_data[i] = 1.0f / (1.0f + std::exp(-x));
        }
#elif defined(TENZOR_HAS_AVX2)
        const size_t simd_width = 8;
        const size_t simd_end = (n / simd_width) * simd_width;

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256 x = _mm256_loadu_ps(in_data + i);
            __m256 result = fast_math::sigmoid_avx2(x);
            _mm256_storeu_ps(out_data + i, result);
        }

        // Scalar remainder
        for (size_t i = simd_end; i < n; ++i) {
            float x = in_data[i];
            out_data[i] = 1.0f / (1.0f + std::exp(-x));
        }
#else
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i];
            out_data[i] = 1.0f / (1.0f + std::exp(-x));
        }
#endif

    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            double x = in_data[i];
            out_data[i] = 1.0 / (1.0 + std::exp(-x));
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = output.data<Float16>();
        size_t n = input.numel();

#if defined(TENZOR_HAS_AVX2)
        // Convert to float, process with SIMD, convert back
        std::vector<float> in_f32(n), out_f32(n);

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            in_f32[i] = static_cast<float>(in_data[i]);
        }

        const size_t simd_width = 8;
        const size_t simd_end = (n / simd_width) * simd_width;

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256 x = _mm256_loadu_ps(in_f32.data() + i);
            __m256 result = fast_math::sigmoid_avx2(x);
            _mm256_storeu_ps(out_f32.data() + i, result);
        }

        for (size_t i = simd_end; i < n; ++i) {
            float x = in_f32[i];
            out_f32[i] = 1.0f / (1.0f + std::exp(-x));
        }

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = Float16(out_f32[i]);
        }
#else
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(in_data[i]);
            out_data[i] = Float16(1.0f / (1.0f + std::exp(-val)));
        }
#endif
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = output.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(in_data[i]);
            out_data[i] = BFloat16(1.0f / (1.0f + std::exp(-val)));
        }
    } else {
        throw std::runtime_error("Sigmoid only supports Float32, Float64, Float16, and BFloat16");
    }

    return output;
}

// Backward: grad_out * sigmoid(x) * (1 - sigmoid(x))
auto sigmoid_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw) -> Tensor {
    auto grad_output = grad_output_raw.contiguous();
    auto input = input_raw.contiguous();
    auto grad_input = zeros_like(input);

    if (input.dtype() == DType::Float32) {
        const float* grad_out_data = grad_output.data<float>();
        const float* in_data = input.data<float>();
        float* grad_in_data = grad_input.data<float>();
        size_t n = input.numel();

#ifdef TENZOR_HAS_AVX512
        __m512 one = _mm512_set1_ps(1.0f);
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t ii = 0; ii < n / 16; ++ii) {
            size_t offset = ii * 16;
            __m512 x = _mm512_loadu_ps(in_data + offset);
            __m512 grad = _mm512_loadu_ps(grad_out_data + offset);
            __m512 sig = fast_math::sigmoid_avx512(x);
            __m512 result = _mm512_mul_ps(_mm512_mul_ps(grad, sig), _mm512_sub_ps(one, sig));
            _mm512_storeu_ps(grad_in_data + offset, result);
        }
        // Scalar remainder
        for (size_t j = (n / 16) * 16; j < n; ++j) {
            float sigmoid_x = 1.0f / (1.0f + std::exp(-in_data[j]));
            grad_in_data[j] = grad_out_data[j] * sigmoid_x * (1.0f - sigmoid_x);
        }
#elif defined(TENZOR_HAS_AVX2)
        __m256 one = _mm256_set1_ps(1.0f);
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t ii = 0; ii < n / 8; ++ii) {
            size_t offset = ii * 8;
            __m256 x = _mm256_loadu_ps(in_data + offset);
            __m256 grad = _mm256_loadu_ps(grad_out_data + offset);
            __m256 sig = fast_math::sigmoid_avx2(x);
            __m256 result = _mm256_mul_ps(_mm256_mul_ps(grad, sig), _mm256_sub_ps(one, sig));
            _mm256_storeu_ps(grad_in_data + offset, result);
        }
        // Scalar remainder
        for (size_t j = (n / 8) * 8; j < n; ++j) {
            float sigmoid_x = 1.0f / (1.0f + std::exp(-in_data[j]));
            grad_in_data[j] = grad_out_data[j] * sigmoid_x * (1.0f - sigmoid_x);
        }
#else
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float sigmoid_x = 1.0f / (1.0f + std::exp(-in_data[i]));
            grad_in_data[i] = grad_out_data[i] * sigmoid_x * (1.0f - sigmoid_x);
        }
#endif
    } else if (input.dtype() == DType::Float64) {
        const double* grad_out_data = grad_output.data<double>();
        const double* in_data = input.data<double>();
        double* grad_in_data = grad_input.data<double>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            double sigmoid_x = 1.0 / (1.0 + std::exp(-in_data[i]));
            grad_in_data[i] = grad_out_data[i] * sigmoid_x * (1.0 - sigmoid_x);
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* grad_out_data = grad_output.data<Float16>();
        const Float16* in_data = input.data<Float16>();
        Float16* grad_in_data = grad_input.data<Float16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
            grad_in_data[i] = Float16(static_cast<float>(grad_out_data[i]) * sigmoid_x * (1.0f - sigmoid_x));
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* grad_out_data = grad_output.data<BFloat16>();
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* grad_in_data = grad_input.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
            grad_in_data[i] = BFloat16(static_cast<float>(grad_out_data[i]) * sigmoid_x * (1.0f - sigmoid_x));
        }
    } else {
        throw std::runtime_error("Sigmoid backward only supports Float32, Float64, Float16, and BFloat16");
    }

    return grad_input;
}

// ============================================================================
// Tanh Activation
// ============================================================================

// Forward: (exp(x) - exp(-x)) / (exp(x) + exp(-x)) - OpenMP + SIMD optimized
auto tanh_kernel(const Tensor& input_raw) -> Tensor {
    // audit-2026-05-03 bug #15 mirror: ensure contiguous input.
    auto input = input_raw.contiguous();
    auto output = Tensor(std::vector<int64_t>(input.shape().begin(), input.shape().end()), input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();
        size_t n = input.numel();

#ifdef TENZOR_USE_ONEDNN
        // Try oneDNN for large tensors (faster than SIMD for tanh)
        if (onednn_eltwise_forward(in_data, out_data, n, dnnl::algorithm::eltwise_tanh)) {
            return output;
        }
#endif
        // Fall back to SIMD implementation
        // For small arrays, use single-threaded SIMD
        if (n < ACTIVATION_OMP_THRESHOLD) {
#ifdef TENZOR_HAS_AVX512
            fast_math::tanh_batch_avx512(in_data, out_data, n);
#elif defined(TENZOR_HAS_AVX2)
            fast_math::tanh_batch_avx2(in_data, out_data, n);
#else
            for (size_t i = 0; i < n; ++i) {
                out_data[i] = std::tanh(in_data[i]);
            }
#endif
        } else {
            // For large arrays, use OpenMP with thread-local SIMD
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nthreads = omp_get_num_threads();

                size_t chunk_size = (n + nthreads - 1) / nthreads;
                size_t start = tid * chunk_size;
                size_t end = std::min(start + chunk_size, n);

                if (start < end) {
#ifdef TENZOR_HAS_AVX512
                    fast_math::tanh_batch_avx512(in_data + start, out_data + start, end - start);
#elif defined(TENZOR_HAS_AVX2)
                    fast_math::tanh_batch_avx2(in_data + start, out_data + start, end - start);
#else
                    for (size_t i = start; i < end; ++i) {
                        out_data[i] = std::tanh(in_data[i]);
                    }
#endif
                }
            }
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();
        size_t n = input.numel();

        #pragma omp parallel for if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = std::tanh(in_data[i]);
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = output.data<Float16>();
        size_t n = input.numel();

        if (n < ACTIVATION_OMP_THRESHOLD) {
#ifdef __F16C__
            size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in_data + i));
                __m256 fp32 = _mm256_cvtph_ps(packed);
                __m256 result = fast_math::tanh_avx2(fp32);
                __m128i out_packed = _mm256_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out_data + i), out_packed);
            }
            for (; i < n; ++i) {
                float val = static_cast<float>(in_data[i]);
                out_data[i] = Float16(std::tanh(val));
            }
#else
            for (size_t i = 0; i < n; ++i) {
                float val = static_cast<float>(in_data[i]);
                out_data[i] = Float16(std::tanh(val));
            }
#endif
        } else {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nthreads = omp_get_num_threads();

                size_t chunk_size = (n + nthreads - 1) / nthreads;
                size_t start = tid * chunk_size;
                size_t end = std::min(start + chunk_size, n);

#ifdef __F16C__
                size_t i = start;
                for (; i + 8 <= end; i += 8) {
                    __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in_data + i));
                    __m256 fp32 = _mm256_cvtph_ps(packed);
                    __m256 result = fast_math::tanh_avx2(fp32);
                    __m128i out_packed = _mm256_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_data + i), out_packed);
                }
                for (; i < end; ++i) {
                    float val = static_cast<float>(in_data[i]);
                    out_data[i] = Float16(std::tanh(val));
                }
#else
                for (size_t i = start; i < end; ++i) {
                    float val = static_cast<float>(in_data[i]);
                    out_data[i] = Float16(std::tanh(val));
                }
#endif
            }
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = output.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(in_data[i]);
            out_data[i] = BFloat16(std::tanh(val));
        }
    } else {
        throw std::runtime_error("Tanh only supports Float32, Float64, Float16, and BFloat16");
    }

    return output;
}

// Backward: grad_out * (1 - tanh(x)^2)
auto tanh_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw) -> Tensor {
    auto grad_output = grad_output_raw.contiguous();
    auto input = input_raw.contiguous();
    auto grad_input = zeros_like(input);

    if (input.dtype() == DType::Float32) {
        const float* grad_out_data = grad_output.data<float>();
        const float* in_data = input.data<float>();
        float* grad_in_data = grad_input.data<float>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float tanh_x = std::tanh(in_data[i]);
            grad_in_data[i] = grad_out_data[i] * (1.0f - tanh_x * tanh_x);
        }
    } else if (input.dtype() == DType::Float64) {
        const double* grad_out_data = grad_output.data<double>();
        const double* in_data = input.data<double>();
        double* grad_in_data = grad_input.data<double>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            double tanh_x = std::tanh(in_data[i]);
            grad_in_data[i] = grad_out_data[i] * (1.0 - tanh_x * tanh_x);
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* grad_out_data = grad_output.data<Float16>();
        const Float16* in_data = input.data<Float16>();
        Float16* grad_in_data = grad_input.data<Float16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float in_val = static_cast<float>(in_data[i]);
            float grad_out_val = static_cast<float>(grad_out_data[i]);
            float tanh_x = std::tanh(in_val);
            grad_in_data[i] = Float16(grad_out_val * (1.0f - tanh_x * tanh_x));
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* grad_out_data = grad_output.data<BFloat16>();
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* grad_in_data = grad_input.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float in_val = static_cast<float>(in_data[i]);
            float grad_out_val = static_cast<float>(grad_out_data[i]);
            float tanh_x = std::tanh(in_val);
            grad_in_data[i] = BFloat16(grad_out_val * (1.0f - tanh_x * tanh_x));
        }
    } else {
        throw std::runtime_error("Tanh backward only supports Float32, Float64, Float16, and BFloat16");
    }

    return grad_input;
}

// ============================================================================
// GELU Activation
// ============================================================================

// Forward: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// GELU (Gaussian Error Linear Unit) - exact erf form: 0.5*x*(1+erf(x/sqrt(2)))
auto gelu_kernel(const Tensor& input_raw) -> Tensor {
    auto input = input_raw.contiguous();  // flat-pointer iteration requires contiguity
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()), input.dtype(), input.device());

    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))) — matches PyTorch's default
    // (approximate='none'). The previous tanh approximation is ~1e-3 off and
    // diverged between dtypes; erf is exact and consistent across all dtypes.
    constexpr double INV_SQRT2_D = 0.7071067811865475244;
    constexpr float  INV_SQRT2_F = 0.70710678f;

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();
        size_t n = input.numel();

#ifdef TENZOR_USE_ONEDNN
        // oneDNN exact-erf GELU for large tensors (matches the scalar path).
        if (onednn_eltwise_forward(in_data, out_data, n, dnnl::algorithm::eltwise_gelu_erf)) {
            return output;
        }
#endif
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i];
            out_data[i] = 0.5f * x * (1.0f + std::erf(x * INV_SQRT2_F));
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            double x = in_data[i];
            out_data[i] = 0.5 * x * (1.0 + std::erf(x * INV_SQRT2_D));
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = output.data<Float16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            out_data[i] = Float16(0.5f * x * (1.0f + std::erf(x * INV_SQRT2_F)));
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = output.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            out_data[i] = BFloat16(0.5f * x * (1.0f + std::erf(x * INV_SQRT2_F)));
        }
    } else {
        throw std::runtime_error("GELU only supports Float32/Float64/Float16/BFloat16");
    }

    return output;
}

// Backward: derivative of exact erf GELU: 0.5*(1+erf(x/sqrt2)) + x*(1/sqrt(2pi))*e^(-x^2/2)
// gelu(x) = 0.5 * x * (1 + tanh(u)), where u = sqrt(2/pi) * (x + 0.044715 * x^3)
// gelu'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * du/dx
//          = 0.5 * (1 + tanh(u)) + 0.5 * x * (1 - tanh(u)^2) * sqrt(2/pi) * (1 + 3*0.044715*x^2)
auto gelu_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw) -> Tensor {
    auto grad_output = grad_output_raw.contiguous();  // flat-pointer iteration requires contiguity
    auto input = input_raw.contiguous();
    auto grad_input = zeros_like(input);

    // Exact GELU derivative (matches the erf forward):
    //   gelu(x)  = 0.5 * x * (1 + erf(x / sqrt(2)))
    //   gelu'(x) = 0.5 * (1 + erf(x / sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2)
    constexpr double INV_SQRT2_D   = 0.7071067811865475244;
    constexpr double PDF_COEF_D    = 0.3989422804014326779;  // 1/sqrt(2*pi)
    constexpr float  INV_SQRT2_F   = 0.70710678f;
    constexpr float  PDF_COEF_F    = 0.39894228f;

    if (input.dtype() == DType::Float32) {
        const float* grad_out_data = grad_output.data<float>();
        const float* in_data = input.data<float>();
        float* grad_in_data = grad_input.data<float>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i];
            float cdf = 0.5f * (1.0f + std::erf(x * INV_SQRT2_F));
            float pdf = PDF_COEF_F * std::exp(-0.5f * x * x);
            grad_in_data[i] = grad_out_data[i] * (cdf + x * pdf);
        }
    } else if (input.dtype() == DType::Float64) {
        const double* grad_out_data = grad_output.data<double>();
        const double* in_data = input.data<double>();
        double* grad_in_data = grad_input.data<double>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            double x = in_data[i];
            double cdf = 0.5 * (1.0 + std::erf(x * INV_SQRT2_D));
            double pdf = PDF_COEF_D * std::exp(-0.5 * x * x);
            grad_in_data[i] = grad_out_data[i] * (cdf + x * pdf);
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* grad_out_data = grad_output.data<Float16>();
        const Float16* in_data = input.data<Float16>();
        Float16* grad_in_data = grad_input.data<Float16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float cdf = 0.5f * (1.0f + std::erf(x * INV_SQRT2_F));
            float pdf = PDF_COEF_F * std::exp(-0.5f * x * x);
            grad_in_data[i] = Float16(static_cast<float>(grad_out_data[i]) * (cdf + x * pdf));
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* grad_out_data = grad_output.data<BFloat16>();
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* grad_in_data = grad_input.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float cdf = 0.5f * (1.0f + std::erf(x * INV_SQRT2_F));
            float pdf = PDF_COEF_F * std::exp(-0.5f * x * x);
            grad_in_data[i] = BFloat16(static_cast<float>(grad_out_data[i]) * (cdf + x * pdf));
        }
    } else {
        throw std::runtime_error("GELU backward only supports Float32/Float64/Float16/BFloat16");
    }

    return grad_input;
}

// ============================================================================
// Swish/SiLU Activation
// ============================================================================

// Forward: x * sigmoid(x)
// Swish (also known as SiLU - Sigmoid Linear Unit)
auto swish_kernel(const Tensor& input_raw) -> Tensor {
    auto input = input_raw.contiguous();  // flat-pointer iteration requires contiguity
    auto output = Tensor(std::vector<int64_t>(input.shape().begin(), input.shape().end()), input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i];
            float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
            out_data[i] = x * sigmoid_x;
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            double x = in_data[i];
            double sigmoid_x = 1.0 / (1.0 + std::exp(-x));
            out_data[i] = x * sigmoid_x;
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = output.data<Float16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
            out_data[i] = Float16(x * sigmoid_x);
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = output.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
            out_data[i] = BFloat16(x * sigmoid_x);
        }
    } else {
        throw std::runtime_error("Swish only supports Float32, Float64, Float16, and BFloat16");
    }

    return output;
}

// Backward: grad_out * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
// Simplified: grad_out * (sigmoid(x) * (1 + x * (1 - sigmoid(x))))
auto swish_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw) -> Tensor {
    auto grad_output = grad_output_raw.contiguous();  // flat-pointer iteration requires contiguity
    auto input = input_raw.contiguous();
    auto grad_input = zeros_like(input);

    if (input.dtype() == DType::Float32) {
        const float* grad_out_data = grad_output.data<float>();
        const float* in_data = input.data<float>();
        float* grad_in_data = grad_input.data<float>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i];
            float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
            grad_in_data[i] = grad_out_data[i] * (sigmoid_x * (1.0f + x * (1.0f - sigmoid_x)));
        }
    } else if (input.dtype() == DType::Float64) {
        const double* grad_out_data = grad_output.data<double>();
        const double* in_data = input.data<double>();
        double* grad_in_data = grad_input.data<double>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            double x = in_data[i];
            double sigmoid_x = 1.0 / (1.0 + std::exp(-x));
            grad_in_data[i] = grad_out_data[i] * (sigmoid_x * (1.0 + x * (1.0 - sigmoid_x)));
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* grad_out_data = grad_output.data<Float16>();
        const Float16* in_data = input.data<Float16>();
        Float16* grad_in_data = grad_input.data<Float16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
            float grad = static_cast<float>(grad_out_data[i]) * (sigmoid_x * (1.0f + x * (1.0f - sigmoid_x)));
            grad_in_data[i] = Float16(grad);
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* grad_out_data = grad_output.data<BFloat16>();
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* grad_in_data = grad_input.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
            float grad = static_cast<float>(grad_out_data[i]) * (sigmoid_x * (1.0f + x * (1.0f - sigmoid_x)));
            grad_in_data[i] = BFloat16(grad);
        }
    } else {
        throw std::runtime_error("Swish backward only supports Float32, Float64, Float16, and BFloat16");
    }

    return grad_input;
}

// ============================================================================
// Leaky ReLU Activation
// ============================================================================

// Forward: x if x > 0 else alpha * x
auto leaky_relu_kernel(const Tensor& input_raw, double alpha) -> Tensor {
    // Flat-pointer iteration requires contiguous input; a non-contiguous
    // slice/transpose/expand view would otherwise read the wrong elements.
    auto input = input_raw.contiguous();
    auto output = Tensor(std::vector<int64_t>(input.shape().begin(), input.shape().end()), input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();
        size_t n = input.numel();

        const float alpha_f = static_cast<float>(alpha);
#ifdef TENZOR_HAS_AVX512
        const size_t simd_width = 16;
        const size_t simd_end = (n / simd_width) * simd_width;
        const __m512 alpha_vec = _mm512_set1_ps(alpha_f);
        const __m512 zero = _mm512_setzero_ps();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m512 x = _mm512_loadu_ps(in_data + i);
            __mmask16 mask = _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ);
            __m512 negative_part = _mm512_mul_ps(x, alpha_vec);
            __m512 result = _mm512_mask_blend_ps(mask, negative_part, x);
            _mm512_storeu_ps(out_data + i, result);
        }

        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = in_data[i] > 0.0f ? in_data[i] : alpha_f * in_data[i];
        }
#elif defined(TENZOR_HAS_AVX2)
        const size_t simd_width = 8;
        const size_t simd_end = (n / simd_width) * simd_width;
        const __m256 alpha_vec = _mm256_set1_ps(alpha_f);
        const __m256 zero = _mm256_setzero_ps();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256 x = _mm256_loadu_ps(in_data + i);
            __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
            __m256 negative_part = _mm256_mul_ps(x, alpha_vec);
            __m256 result = _mm256_blendv_ps(negative_part, x, mask);
            _mm256_storeu_ps(out_data + i, result);
        }

        for (size_t i = simd_end; i < n; ++i) {
            out_data[i] = in_data[i] > 0.0f ? in_data[i] : alpha_f * in_data[i];
        }
#else
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = in_data[i] > 0.0f ? in_data[i] : alpha_f * in_data[i];
        }
#endif
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();
        size_t n = input.numel();
        double alpha_d = static_cast<double>(alpha);

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            out_data[i] = in_data[i] > 0.0 ? in_data[i] : alpha_d * in_data[i];
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = output.data<Float16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(in_data[i]);
            out_data[i] = Float16(static_cast<float>(val > 0.0f ? val : alpha * val));
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = output.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(in_data[i]);
            out_data[i] = BFloat16(static_cast<float>(val > 0.0f ? val : alpha * val));
        }
    } else {
        throw std::runtime_error("Leaky ReLU only supports Float32, Float64, Float16, and BFloat16");
    }

    return output;
}

// Backward: grad_out * (1 if x > 0 else alpha)
auto leaky_relu_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw, double alpha) -> Tensor {
    auto grad_output = grad_output_raw.contiguous();
    auto input = input_raw.contiguous();
    auto grad_input = zeros_like(input);

    if (input.dtype() == DType::Float32) {
        const float* grad_out_data = grad_output.data<float>();
        const float* in_data = input.data<float>();
        float* grad_in_data = grad_input.data<float>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            grad_in_data[i] = grad_out_data[i] * (in_data[i] > 0.0f ? 1.0f : alpha);
        }
    } else if (input.dtype() == DType::Float64) {
        const double* grad_out_data = grad_output.data<double>();
        const double* in_data = input.data<double>();
        double* grad_in_data = grad_input.data<double>();
        size_t n = input.numel();
        double alpha_d = static_cast<double>(alpha);

#ifdef TENZOR_HAS_AVX512
        const size_t simd_width = 8;
        const size_t simd_end = (n / simd_width) * simd_width;
        __m512d zero_d = _mm512_setzero_pd();
        __m512d one_d = _mm512_set1_pd(1.0);
        __m512d vslope_d = _mm512_set1_pd(alpha_d);

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m512d x = _mm512_loadu_pd(in_data + i);
            __m512d grad = _mm512_loadu_pd(grad_out_data + i);
            __mmask8 mask = _mm512_cmp_pd_mask(x, zero_d, _CMP_GT_OQ);
            __m512d multiplier = _mm512_mask_blend_pd(mask, vslope_d, one_d);
            __m512d result = _mm512_mul_pd(grad, multiplier);
            _mm512_storeu_pd(grad_in_data + i, result);
        }
        for (size_t i = simd_end; i < n; ++i) {
            grad_in_data[i] = grad_out_data[i] * (in_data[i] > 0.0 ? 1.0 : alpha_d);
        }
#elif defined(TENZOR_HAS_AVX2)
        const size_t simd_width = 4;
        const size_t simd_end = (n / simd_width) * simd_width;
        __m256d zero_d = _mm256_setzero_pd();
        __m256d one_d = _mm256_set1_pd(1.0);
        __m256d vslope_d = _mm256_set1_pd(alpha_d);

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < simd_end; i += simd_width) {
            __m256d x = _mm256_loadu_pd(in_data + i);
            __m256d grad = _mm256_loadu_pd(grad_out_data + i);
            __m256d mask = _mm256_cmp_pd(x, zero_d, _CMP_GT_OQ);
            __m256d multiplier = _mm256_blendv_pd(vslope_d, one_d, mask);
            __m256d result = _mm256_mul_pd(grad, multiplier);
            _mm256_storeu_pd(grad_in_data + i, result);
        }
        for (size_t i = simd_end; i < n; ++i) {
            grad_in_data[i] = grad_out_data[i] * (in_data[i] > 0.0 ? 1.0 : alpha_d);
        }
#else
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            grad_in_data[i] = grad_out_data[i] * (in_data[i] > 0.0 ? 1.0 : alpha_d);
        }
#endif
    } else if (input.dtype() == DType::Float16) {
        const Float16* grad_out_data = grad_output.data<Float16>();
        const Float16* in_data = input.data<Float16>();
        Float16* grad_in_data = grad_input.data<Float16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float in_val = static_cast<float>(in_data[i]);
            float grad_out_val = static_cast<float>(grad_out_data[i]);
            grad_in_data[i] = Float16(static_cast<float>(grad_out_val * (in_val > 0.0f ? 1.0f : alpha)));
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* grad_out_data = grad_output.data<BFloat16>();
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* grad_in_data = grad_input.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float in_val = static_cast<float>(in_data[i]);
            float grad_out_val = static_cast<float>(grad_out_data[i]);
            grad_in_data[i] = BFloat16(static_cast<float>(grad_out_val * (in_val > 0.0f ? 1.0f : alpha)));
        }
    } else {
        throw std::runtime_error("Leaky ReLU backward only supports Float32, Float64, Float16, and BFloat16");
    }

    return grad_input;
}

// ============================================================================
// Softmax Activation
// ============================================================================

// Helper: Compute max along dimension
static auto compute_max_along_dim(const float* data, const std::vector<int64_t>& shape, int64_t dim) -> std::vector<float> {
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= shape[i];
    }

    int64_t dim_size = shape[dim];

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
        inner_size *= shape[i];
    }

    std::vector<float> max_vals(outer_size * inner_size, -std::numeric_limits<float>::infinity());

    // Rows (outer index i) are independent; each writes disjoint max_idx slots.
    #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
    for (int64_t i = 0; i < outer_size; ++i) {
        for (int64_t k = 0; k < inner_size; ++k) {
            for (int64_t j = 0; j < dim_size; ++j) {
                int64_t idx = (i * dim_size + j) * inner_size + k;
                int64_t max_idx = i * inner_size + k;
                max_vals[max_idx] = std::max(max_vals[max_idx], data[idx]);
            }
        }
    }

    return max_vals;
}

// Forward: exp(x_i - max) / sum(exp(x_j - max))
auto softmax_kernel(const Tensor& input, int64_t dim) -> Tensor {
    if (input.numel() == 0) return input.clone();

    // Materialise a contiguous copy so all dtype branches can index with
    // simple row-major arithmetic. Zero-cost when caller already passed a
    // contiguous tensor (contiguous() short-circuits to a no-op).
    auto cont_input = input.contiguous();

    auto output = Tensor(std::vector<int64_t>(cont_input.shape().begin(), cont_input.shape().end()), cont_input.dtype(), cont_input.device());

    // Handle negative dimension
    if (dim < 0) {
        dim += cont_input.ndim();
    }

    if (dim < 0 || dim >= cont_input.ndim()) {
        throw std::runtime_error("Softmax dimension out of range");
    }

    if (cont_input.dtype() == DType::Float16) {
        const Float16* in_data = cont_input.data<Float16>();
        Float16* out_data = output.data<Float16>();

        auto shape_span = cont_input.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Use Float32 accumulation for numerical stability. Rows (outer i) are
        // independent and write disjoint output slots -> safe to parallelize.
        #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                // Find max (Float32 accumulation)
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    max_val = std::max(max_val, static_cast<float>(in_data[idx]));
                }

                // Compute exp and Kahan-compensated sum (matches logsumexp's
                // rigor; plain float summation lost precision over long dims).
                float sum = 0.0f, comp = 0.0f;
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    float exp_val = std::exp(static_cast<float>(in_data[idx]) - max_val);
                    out_data[idx] = Float16(exp_val);
                    float y = exp_val - comp;
                    float t = sum + y;
                    comp = (t - sum) - y;
                    sum = t;
                }

                // Normalize (convert back to Float16)
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    out_data[idx] = Float16(static_cast<float>(out_data[idx]) / sum);
                }
            }
        }
    } else if (cont_input.dtype() == DType::Float32) {
        const float* in_data = cont_input.data<float>();
        float* out_data = output.data<float>();

        auto shape_span = cont_input.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

#ifdef TENZOR_USE_ONEDNN
        // Try oneDNN softmax for large tensors
        if (onednn_softmax_forward(in_data, out_data, shape, dim)) {
            return output;
        }
#endif
        // Fall back to scalar implementation
        // Compute max values for numerical stability
        auto max_vals = compute_max_along_dim(in_data, shape, dim);

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Compute exp(x - max) and sum (Kahan-compensated to match the
        // Float16/BFloat16 paths' precision over long reduction dims).
        // Rows (outer i) are independent -> safe to parallelize.
        std::vector<float> sum_exp(outer_size * inner_size, 0.0f);

        #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                int64_t max_idx = i * inner_size + k;
                float max_val = max_vals[max_idx];

                float sum = 0.0f, comp = 0.0f;
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    float exp_val = std::exp(in_data[idx] - max_val);
                    out_data[idx] = exp_val;
                    float y = exp_val - comp;
                    float t = sum + y;
                    comp = (t - sum) - y;
                    sum = t;
                }
                sum_exp[max_idx] = sum;
            }
        }

        // Normalize
        #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                int64_t max_idx = i * inner_size + k;
                float sum = sum_exp[max_idx];

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    out_data[idx] /= sum;
                }
            }
        }
    } else if (cont_input.dtype() == DType::Float64) {
        const double* in_data = cont_input.data<double>();
        double* out_data = output.data<double>();

        auto shape_span = cont_input.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Simple implementation for double. Rows (outer i) are independent.
        #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                // Find max
                double max_val = -std::numeric_limits<double>::infinity();
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    max_val = std::max(max_val, in_data[idx]);
                }

                // Compute exp and sum
                double sum = 0.0;
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    double exp_val = std::exp(in_data[idx] - max_val);
                    out_data[idx] = exp_val;
                    sum += exp_val;
                }

                // Normalize
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    out_data[idx] /= sum;
                }
            }
        }
    } else if (cont_input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = cont_input.data<BFloat16>();
        BFloat16* out_data = output.data<BFloat16>();

        auto shape_span = cont_input.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Use Float32 accumulation for numerical stability. Rows independent.
        #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    max_val = std::max(max_val, static_cast<float>(in_data[idx]));
                }

                float sum = 0.0f, comp = 0.0f;  // Kahan-compensated
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    float exp_val = std::exp(static_cast<float>(in_data[idx]) - max_val);
                    out_data[idx] = BFloat16(exp_val);
                    float y = exp_val - comp;
                    float t = sum + y;
                    comp = (t - sum) - y;
                    sum = t;
                }

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    out_data[idx] = BFloat16(static_cast<float>(out_data[idx]) / sum);
                }
            }
        }
    } else {
        throw std::runtime_error("Softmax only supports Float16, Float32, Float64, and BFloat16");
    }

    return output;
}

// Backward: Jacobian-vector product
// grad_input[i] = softmax[i] * (grad_output[i] - sum(grad_output * softmax))
auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim) -> Tensor {
    // Materialise contiguous copies so row-major index arithmetic is correct
    // for any non-contiguous (e.g. transposed) input tensors.
    auto cont_grad_output = grad_output.contiguous();
    auto cont_output      = output.contiguous();

    if (cont_output.numel() == 0) return cont_output.clone();

    auto grad_input = zeros_like(cont_output);

    // Handle negative dimension and validate range (mirrors the forward kernel)
    int64_t ndim = cont_output.ndim();
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("softmax_backward: dim out of range");
    }

    if (cont_output.dtype() == DType::Float16) {
        const Float16* grad_out_data = cont_grad_output.data<Float16>();
        const Float16* out_data = cont_output.data<Float16>();
        Float16* grad_in_data = grad_input.data<Float16>();

        auto shape_span = cont_output.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Use Float32 accumulation for numerical stability. Rows independent.
        #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                float sum = 0.0f;

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    sum += static_cast<float>(grad_out_data[idx]) * static_cast<float>(out_data[idx]);
                }

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    grad_in_data[idx] = Float16(static_cast<float>(out_data[idx]) * (static_cast<float>(grad_out_data[idx]) - sum));
                }
            }
        }
    } else if (cont_output.dtype() == DType::Float32) {
        const float* grad_out_data = cont_grad_output.data<float>();
        const float* out_data = cont_output.data<float>();
        float* grad_in_data = grad_input.data<float>();

        auto shape_span = cont_output.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Compute sum(grad_output * softmax) for each position. Rows independent.
        #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                float sum = 0.0f;

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    sum += grad_out_data[idx] * out_data[idx];
                }

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    grad_in_data[idx] = out_data[idx] * (grad_out_data[idx] - sum);
                }
            }
        }
    } else if (cont_output.dtype() == DType::Float64) {
        const double* grad_out_data = cont_grad_output.data<double>();
        const double* out_data = cont_output.data<double>();
        double* grad_in_data = grad_input.data<double>();

        auto shape_span = cont_output.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Rows (outer i) are independent.
        #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                double sum = 0.0;

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    sum += grad_out_data[idx] * out_data[idx];
                }

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    grad_in_data[idx] = out_data[idx] * (grad_out_data[idx] - sum);
                }
            }
        }
    } else if (cont_output.dtype() == DType::BFloat16) {
        const BFloat16* grad_out_data = cont_grad_output.data<BFloat16>();
        const BFloat16* out_data = cont_output.data<BFloat16>();
        BFloat16* grad_in_data = grad_input.data<BFloat16>();

        auto shape_span = cont_output.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Rows (outer i) are independent.
        #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                float sum = 0.0f;
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    sum += static_cast<float>(grad_out_data[idx]) * static_cast<float>(out_data[idx]);
                }
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    grad_in_data[idx] = BFloat16(static_cast<float>(out_data[idx]) * (static_cast<float>(grad_out_data[idx]) - sum));
                }
            }
        }
    } else {
        throw std::runtime_error("Softmax backward only supports Float16, Float32, Float64, and BFloat16");
    }

    return grad_input;
}

// ============================================================================
// LogSoftmax Activation
// ============================================================================

// Forward: log(softmax(x, dim)) = x - max(x) - log(sum(exp(x - max(x))))
// More numerically stable than computing softmax then taking log
auto log_softmax_kernel(const Tensor& input, int64_t dim) -> Tensor {
    if (input.numel() == 0) return input.clone();

    // Materialise a contiguous copy so all dtype branches can index with
    // simple row-major arithmetic. Zero-cost when already contiguous.
    auto cont_input = input.contiguous();

    auto output = Tensor(std::vector<int64_t>(cont_input.shape().begin(), cont_input.shape().end()), cont_input.dtype(), cont_input.device());

    // Handle negative dimension
    if (dim < 0) {
        dim += cont_input.ndim();
    }

    if (dim < 0 || dim >= cont_input.ndim()) {
        throw std::runtime_error("LogSoftmax dimension out of range");
    }

    if (cont_input.dtype() == DType::Float16) {
        const Float16* in_data = cont_input.data<Float16>();
        Float16* out_data = output.data<Float16>();

        auto shape_span = cont_input.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Use Float32 accumulation for numerical stability. Rows independent.
        #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                // Find max (Float32 accumulation)
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    max_val = std::max(max_val, static_cast<float>(in_data[idx]));
                }

                // Compute log(sum(exp(x - max))) — Kahan-compensated sum.
                float sum_exp = 0.0f, comp = 0.0f;
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    float e = std::exp(static_cast<float>(in_data[idx]) - max_val);
                    float y = e - comp;
                    float t = sum_exp + y;
                    comp = (t - sum_exp) - y;
                    sum_exp = t;
                }
                float log_sum_exp = std::log(sum_exp);

                // Compute log_softmax and convert to Float16
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    out_data[idx] = Float16(static_cast<float>(in_data[idx]) - max_val - log_sum_exp);
                }
            }
        }
    } else if (cont_input.dtype() == DType::Float32) {
        const float* in_data = cont_input.data<float>();
        float* out_data = output.data<float>();

        auto shape_span = cont_input.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        // Compute max values for numerical stability
        auto max_vals = compute_max_along_dim(in_data, shape, dim);

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Compute log(sum(exp(x - max))). Rows (outer i) are independent.
        std::vector<float> log_sum_exp(outer_size * inner_size, 0.0f);

        #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                int64_t max_idx = i * inner_size + k;
                float max_val = max_vals[max_idx];
                float sum_exp = 0.0f, comp = 0.0f;  // Kahan-compensated

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    float e = std::exp(in_data[idx] - max_val);
                    float y = e - comp;
                    float t = sum_exp + y;
                    comp = (t - sum_exp) - y;
                    sum_exp = t;
                }

                log_sum_exp[max_idx] = std::log(sum_exp);
            }
        }

        // Compute log_softmax = x - max - log_sum_exp. Rows independent.
        #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                int64_t max_idx = i * inner_size + k;
                float max_val = max_vals[max_idx];
                float lse = log_sum_exp[max_idx];

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    out_data[idx] = in_data[idx] - max_val - lse;
                }
            }
        }
    } else if (cont_input.dtype() == DType::Float64) {
        const double* in_data = cont_input.data<double>();
        double* out_data = output.data<double>();

        auto shape_span = cont_input.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Rows (outer i) are independent.
        #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                // Find max
                double max_val = -std::numeric_limits<double>::infinity();
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    max_val = std::max(max_val, in_data[idx]);
                }

                // Compute log(sum(exp(x - max)))
                double sum_exp = 0.0;
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    sum_exp += std::exp(in_data[idx] - max_val);
                }
                double log_sum_exp = std::log(sum_exp);

                // Compute log_softmax
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    out_data[idx] = in_data[idx] - max_val - log_sum_exp;
                }
            }
        }
    } else if (cont_input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = cont_input.data<BFloat16>();
        BFloat16* out_data = output.data<BFloat16>();

        auto shape_span = cont_input.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Rows (outer i) are independent.
        #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    max_val = std::max(max_val, static_cast<float>(in_data[idx]));
                }

                float sum_exp = 0.0f, comp = 0.0f;  // Kahan-compensated
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    float e = std::exp(static_cast<float>(in_data[idx]) - max_val);
                    float y = e - comp;
                    float t = sum_exp + y;
                    comp = (t - sum_exp) - y;
                    sum_exp = t;
                }
                float log_sum_exp = std::log(sum_exp);

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    out_data[idx] = BFloat16(static_cast<float>(in_data[idx]) - max_val - log_sum_exp);
                }
            }
        }
    } else {
        throw std::runtime_error("LogSoftmax only supports Float16, Float32, Float64, and BFloat16");
    }

    return output;
}

// Backward: grad_input = grad_output - exp(log_softmax) * sum(grad_output)
auto log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim) -> Tensor {
    // Materialise contiguous copies so row-major index arithmetic is correct
    // for any non-contiguous (e.g. transposed) input tensors.
    auto cont_grad_output = grad_output.contiguous();
    auto cont_output      = output.contiguous();

    if (cont_output.numel() == 0) return cont_output.clone();

    auto grad_input = zeros_like(cont_output);

    // Handle negative dimension and validate range (mirrors the forward kernel)
    int64_t ndim = cont_output.ndim();
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("log_softmax_backward: dim out of range");
    }

    if (cont_output.dtype() == DType::Float16) {
        const Float16* grad_out_data = cont_grad_output.data<Float16>();
        const Float16* out_data = cont_output.data<Float16>();
        Float16* grad_in_data = grad_input.data<Float16>();

        auto shape_span = cont_output.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Use Float32 accumulation for numerical stability. Rows independent.
        #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                float sum_grad = 0.0f;

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    sum_grad += static_cast<float>(grad_out_data[idx]);
                }

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    grad_in_data[idx] = Float16(static_cast<float>(grad_out_data[idx]) - std::exp(static_cast<float>(out_data[idx])) * sum_grad);
                }
            }
        }
    } else if (cont_output.dtype() == DType::Float32) {
        const float* grad_out_data = cont_grad_output.data<float>();
        const float* out_data = cont_output.data<float>();
        float* grad_in_data = grad_input.data<float>();

        auto shape_span = cont_output.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Rows (outer i) are independent.
        #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                float sum_grad = 0.0f;

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    sum_grad += grad_out_data[idx];
                }

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    grad_in_data[idx] = grad_out_data[idx] - std::exp(out_data[idx]) * sum_grad;
                }
            }
        }
    } else if (cont_output.dtype() == DType::Float64) {
        const double* grad_out_data = cont_grad_output.data<double>();
        const double* out_data = cont_output.data<double>();
        double* grad_in_data = grad_input.data<double>();

        auto shape_span = cont_output.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Rows (outer i) are independent.
        #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                double sum_grad = 0.0;

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    sum_grad += grad_out_data[idx];
                }

                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    grad_in_data[idx] = grad_out_data[idx] - std::exp(out_data[idx]) * sum_grad;
                }
            }
        }
    } else if (cont_output.dtype() == DType::BFloat16) {
        const BFloat16* grad_out_data = cont_grad_output.data<BFloat16>();
        const BFloat16* out_data = cont_output.data<BFloat16>();
        BFloat16* grad_in_data = grad_input.data<BFloat16>();

        auto shape_span = cont_output.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }

        int64_t dim_size = shape[dim];

        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Rows (outer i) are independent.
        #pragma omp parallel for schedule(static) if(outer_size * inner_size * dim_size > ACTIVATION_OMP_THRESHOLD)
        for (int64_t i = 0; i < outer_size; ++i) {
            for (int64_t k = 0; k < inner_size; ++k) {
                float sum_grad = 0.0f;
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    sum_grad += static_cast<float>(grad_out_data[idx]);
                }
                for (int64_t j = 0; j < dim_size; ++j) {
                    int64_t idx = (i * dim_size + j) * inner_size + k;
                    grad_in_data[idx] = BFloat16(static_cast<float>(grad_out_data[idx]) - std::exp(static_cast<float>(out_data[idx])) * sum_grad);
                }
            }
        }
    } else {
        throw std::runtime_error("LogSoftmax backward only supports Float16, Float32, Float64, and BFloat16");
    }

    return grad_input;
}

// ============================================================================
// ELU Activation
// ============================================================================

// Forward: x if x > 0 else alpha * (exp(x) - 1)
auto elu_kernel(const Tensor& input_raw, float alpha) -> Tensor {
    auto input = input_raw.contiguous();  // flat-pointer iteration requires contiguity
    auto output = Tensor(std::vector<int64_t>(input.shape().begin(), input.shape().end()), input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();
        size_t n = input.numel();

        #pragma omp parallel for if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i];
            out_data[i] = (x > 0.0f) ? x : alpha * (std::exp(x) - 1.0f);
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();
        size_t n = input.numel();

        #pragma omp parallel for if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            double x = in_data[i];
            out_data[i] = (x > 0.0) ? x : static_cast<double>(alpha) * (std::exp(x) - 1.0);
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = output.data<Float16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float result = (x > 0.0f) ? x : alpha * (std::exp(x) - 1.0f);
            out_data[i] = Float16(result);
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = output.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float result = (x > 0.0f) ? x : alpha * (std::exp(x) - 1.0f);
            out_data[i] = BFloat16(result);
        }
    } else {
        throw std::runtime_error("ELU only supports Float32, Float64, Float16, and BFloat16");
    }

    return output;
}

// Backward: grad_out * (1 if x > 0 else alpha * exp(x))
auto elu_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw, float alpha) -> Tensor {
    auto grad_output = grad_output_raw.contiguous();  // flat-pointer iteration requires contiguity
    auto input = input_raw.contiguous();
    auto grad_input = zeros_like(input);

    if (input.dtype() == DType::Float32) {
        const float* grad_out_data = grad_output.data<float>();
        const float* in_data = input.data<float>();
        float* grad_in_data = grad_input.data<float>();
        size_t n = input.numel();

        #pragma omp parallel for if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i];
            grad_in_data[i] = grad_out_data[i] * ((x > 0.0f) ? 1.0f : alpha * std::exp(x));
        }
    } else if (input.dtype() == DType::Float64) {
        const double* grad_out_data = grad_output.data<double>();
        const double* in_data = input.data<double>();
        double* grad_in_data = grad_input.data<double>();
        size_t n = input.numel();

        #pragma omp parallel for if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            double x = in_data[i];
            grad_in_data[i] = grad_out_data[i] * ((x > 0.0) ? 1.0 : static_cast<double>(alpha) * std::exp(x));
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* grad_out_data = grad_output.data<Float16>();
        const Float16* in_data = input.data<Float16>();
        Float16* grad_in_data = grad_input.data<Float16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float grad_out = static_cast<float>(grad_out_data[i]);
            float result = grad_out * ((x > 0.0f) ? 1.0f : alpha * std::exp(x));
            grad_in_data[i] = Float16(result);
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* grad_out_data = grad_output.data<BFloat16>();
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* grad_in_data = grad_input.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float grad_out = static_cast<float>(grad_out_data[i]);
            float result = grad_out * ((x > 0.0f) ? 1.0f : alpha * std::exp(x));
            grad_in_data[i] = BFloat16(result);
        }
    } else {
        throw std::runtime_error("ELU backward only supports Float32, Float64, Float16, and BFloat16");
    }

    return grad_input;
}

// ============================================================================
// SELU Activation
// ============================================================================

// SELU constants from the original paper
constexpr float SELU_ALPHA = 1.6732632423543772848170429916717f;
constexpr float SELU_SCALE = 1.0507009873554804934193349852946f;
// Full double-precision variants for the Float64 path. The `float` constants
// above are truncated to ~7 significant digits; casting them to double (as the
// F64 branches previously did) diverged from the JIT/GPU codegen kernel, which
// bakes full-precision literals — a ~3e-8 relative error that fails f64 gradcheck.
constexpr double SELU_ALPHA_D = 1.6732632423543772848170429916717;
constexpr double SELU_SCALE_D = 1.0507009873554804934193349852946;

// Forward: scale * (x if x > 0 else alpha * (exp(x) - 1))
auto selu_kernel(const Tensor& input_raw) -> Tensor {
    auto input = input_raw.contiguous();  // flat-pointer iteration requires contiguity
    auto output = Tensor(std::vector<int64_t>(input.shape().begin(), input.shape().end()), input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();
        size_t n = input.numel();

        #pragma omp parallel for if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i];
            out_data[i] = SELU_SCALE * ((x > 0.0f) ? x : SELU_ALPHA * (std::exp(x) - 1.0f));
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();
        size_t n = input.numel();

        #pragma omp parallel for if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            double x = in_data[i];
            out_data[i] = SELU_SCALE_D *
                ((x > 0.0) ? x : SELU_ALPHA_D * (std::exp(x) - 1.0));
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = output.data<Float16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float result = SELU_SCALE * ((x > 0.0f) ? x : SELU_ALPHA * (std::exp(x) - 1.0f));
            out_data[i] = Float16(result);
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = output.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float result = SELU_SCALE * ((x > 0.0f) ? x : SELU_ALPHA * (std::exp(x) - 1.0f));
            out_data[i] = BFloat16(result);
        }
    } else {
        throw std::runtime_error("SELU only supports Float32, Float64, Float16, and BFloat16");
    }

    return output;
}

// Backward: grad_out * scale * (1 if x > 0 else alpha * exp(x))
auto selu_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw) -> Tensor {
    auto grad_output = grad_output_raw.contiguous();  // flat-pointer iteration requires contiguity
    auto input = input_raw.contiguous();
    auto grad_input = zeros_like(input);

    if (input.dtype() == DType::Float32) {
        const float* grad_out_data = grad_output.data<float>();
        const float* in_data = input.data<float>();
        float* grad_in_data = grad_input.data<float>();
        size_t n = input.numel();

        #pragma omp parallel for if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i];
            grad_in_data[i] = grad_out_data[i] * SELU_SCALE *
                ((x > 0.0f) ? 1.0f : SELU_ALPHA * std::exp(x));
        }
    } else if (input.dtype() == DType::Float64) {
        const double* grad_out_data = grad_output.data<double>();
        const double* in_data = input.data<double>();
        double* grad_in_data = grad_input.data<double>();
        size_t n = input.numel();

        #pragma omp parallel for if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            double x = in_data[i];
            grad_in_data[i] = grad_out_data[i] * SELU_SCALE_D *
                ((x > 0.0) ? 1.0 : SELU_ALPHA_D * std::exp(x));
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* grad_out_data = grad_output.data<Float16>();
        const Float16* in_data = input.data<Float16>();
        Float16* grad_in_data = grad_input.data<Float16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float grad_out = static_cast<float>(grad_out_data[i]);
            float result = grad_out * SELU_SCALE * ((x > 0.0f) ? 1.0f : SELU_ALPHA * std::exp(x));
            grad_in_data[i] = Float16(result);
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* grad_out_data = grad_output.data<BFloat16>();
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* grad_in_data = grad_input.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float grad_out = static_cast<float>(grad_out_data[i]);
            float result = grad_out * SELU_SCALE * ((x > 0.0f) ? 1.0f : SELU_ALPHA * std::exp(x));
            grad_in_data[i] = BFloat16(result);
        }
    } else {
        throw std::runtime_error("SELU backward only supports Float32, Float64, Float16, and BFloat16");
    }

    return grad_input;
}

// ============================================================================
// Mish Activation
// ============================================================================

// Forward: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
auto mish_kernel(const Tensor& input_raw) -> Tensor {
    auto input = input_raw.contiguous();  // flat-pointer iteration requires contiguity
    auto output = Tensor(std::vector<int64_t>(input.shape().begin(), input.shape().end()), input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();
        size_t n = input.numel();

        #pragma omp parallel for if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i];
            // Numerically stable softplus: log(1 + exp(x))
            // For large x: softplus(x) ≈ x
            // For small x: use standard formula
            float softplus;
            if (x > 20.0f) {
                softplus = x;
            } else if (x < -20.0f) {
                softplus = std::exp(x);
            } else {
                softplus = std::log1p(std::exp(x));
            }
            out_data[i] = x * std::tanh(softplus);
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();
        size_t n = input.numel();

        #pragma omp parallel for if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            double x = in_data[i];
            double softplus;
            if (x > 20.0) {
                softplus = x;
            } else if (x < -20.0) {
                softplus = std::exp(x);
            } else {
                softplus = std::log1p(std::exp(x));
            }
            out_data[i] = x * std::tanh(softplus);
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = output.data<Float16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float softplus;
            if (x > 20.0f) {
                softplus = x;
            } else if (x < -20.0f) {
                softplus = std::exp(x);
            } else {
                softplus = std::log1p(std::exp(x));
            }
            out_data[i] = Float16(x * std::tanh(softplus));
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = output.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float softplus;
            if (x > 20.0f) {
                softplus = x;
            } else if (x < -20.0f) {
                softplus = std::exp(x);
            } else {
                softplus = std::log1p(std::exp(x));
            }
            out_data[i] = BFloat16(x * std::tanh(softplus));
        }
    } else {
        throw std::runtime_error("Mish only supports Float32, Float64, Float16, and BFloat16");
    }

    return output;
}

// Backward: d/dx[x * tanh(softplus(x))]
// = tanh(softplus(x)) + x * sech^2(softplus(x)) * sigmoid(x)
auto mish_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw) -> Tensor {
    auto grad_output = grad_output_raw.contiguous();  // flat-pointer iteration requires contiguity
    auto input = input_raw.contiguous();
    auto grad_input = zeros_like(input);

    if (input.dtype() == DType::Float32) {
        const float* grad_out_data = grad_output.data<float>();
        const float* in_data = input.data<float>();
        float* grad_in_data = grad_input.data<float>();
        size_t n = input.numel();

        #pragma omp parallel for if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i];
            float softplus;
            if (x > 20.0f) {
                softplus = x;
            } else if (x < -20.0f) {
                softplus = std::exp(x);
            } else {
                softplus = std::log1p(std::exp(x));
            }
            float tanh_sp = std::tanh(softplus);
            float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
            float sech2 = 1.0f - tanh_sp * tanh_sp;
            grad_in_data[i] = grad_out_data[i] * (tanh_sp + x * sech2 * sigmoid_x);
        }
    } else if (input.dtype() == DType::Float64) {
        const double* grad_out_data = grad_output.data<double>();
        const double* in_data = input.data<double>();
        double* grad_in_data = grad_input.data<double>();
        size_t n = input.numel();

        #pragma omp parallel for if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            double x = in_data[i];
            double softplus;
            if (x > 20.0) {
                softplus = x;
            } else if (x < -20.0) {
                softplus = std::exp(x);
            } else {
                softplus = std::log1p(std::exp(x));
            }
            double tanh_sp = std::tanh(softplus);
            double sigmoid_x = 1.0 / (1.0 + std::exp(-x));
            double sech2 = 1.0 - tanh_sp * tanh_sp;
            grad_in_data[i] = grad_out_data[i] * (tanh_sp + x * sech2 * sigmoid_x);
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* grad_out_data = grad_output.data<Float16>();
        const Float16* in_data = input.data<Float16>();
        Float16* grad_in_data = grad_input.data<Float16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float grad_out = static_cast<float>(grad_out_data[i]);
            float softplus;
            if (x > 20.0f) {
                softplus = x;
            } else if (x < -20.0f) {
                softplus = std::exp(x);
            } else {
                softplus = std::log1p(std::exp(x));
            }
            float tanh_sp = std::tanh(softplus);
            float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
            float sech2 = 1.0f - tanh_sp * tanh_sp;
            grad_in_data[i] = Float16(grad_out * (tanh_sp + x * sech2 * sigmoid_x));
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* grad_out_data = grad_output.data<BFloat16>();
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* grad_in_data = grad_input.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float grad_out = static_cast<float>(grad_out_data[i]);
            float softplus;
            if (x > 20.0f) {
                softplus = x;
            } else if (x < -20.0f) {
                softplus = std::exp(x);
            } else {
                softplus = std::log1p(std::exp(x));
            }
            float tanh_sp = std::tanh(softplus);
            float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
            float sech2 = 1.0f - tanh_sp * tanh_sp;
            grad_in_data[i] = BFloat16(grad_out * (tanh_sp + x * sech2 * sigmoid_x));
        }
    } else {
        throw std::runtime_error("Mish backward only supports Float32, Float64, Float16, and BFloat16");
    }

    return grad_input;
}

// ============================================================================
// Softplus Activation: softplus(x) = ln(1 + exp(x))
// ============================================================================

auto softplus_kernel(const Tensor& input_raw, float beta, float threshold) -> Tensor {
    // Flat-pointer iteration requires contiguous input; a non-contiguous
    // slice/transpose/expand view would otherwise read the wrong elements.
    auto input = input_raw.contiguous();
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor output = empty(shape_vec, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();
        size_t n = input.numel();

        #pragma omp parallel for if(n > ::tenzor::OmpThresholds::matmul())
        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i] * beta;
            // Numerically stable softplus
            if (x > threshold) {
                out_data[i] = in_data[i];  // softplus(x) ≈ x for large x
            } else if (x < -threshold) {
                out_data[i] = std::exp(x) / beta;  // softplus(x) ≈ exp(x)/beta for very negative x
            } else {
                out_data[i] = std::log1p(std::exp(x)) / beta;
            }
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();
        size_t n = input.numel();

        #pragma omp parallel for if(n > ::tenzor::OmpThresholds::matmul())
        for (size_t i = 0; i < n; ++i) {
            double x = in_data[i] * static_cast<double>(beta);
            double thresh = static_cast<double>(threshold);
            if (x > thresh) {
                out_data[i] = in_data[i];
            } else if (x < -thresh) {
                out_data[i] = std::exp(x) / static_cast<double>(beta);
            } else {
                out_data[i] = std::log1p(std::exp(x)) / static_cast<double>(beta);
            }
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = output.data<Float16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ::tenzor::OmpThresholds::matmul())
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]) * beta;
            float result;
            if (x > threshold) {
                result = static_cast<float>(in_data[i]);
            } else if (x < -threshold) {
                result = std::exp(x) / beta;
            } else {
                result = std::log1p(std::exp(x)) / beta;
            }
            out_data[i] = Float16(result);
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = output.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ::tenzor::OmpThresholds::matmul())
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]) * beta;
            float result;
            if (x > threshold) {
                result = static_cast<float>(in_data[i]);
            } else if (x < -threshold) {
                result = std::exp(x) / beta;
            } else {
                result = std::log1p(std::exp(x)) / beta;
            }
            out_data[i] = BFloat16(result);
        }
    } else {
        throw std::runtime_error("Softplus only supports Float32, Float64, Float16, and BFloat16");
    }

    return output;
}

auto softplus_backward_kernel(const Tensor& grad_output_raw, const Tensor& input_raw, float beta, float threshold) -> Tensor {
    auto grad_output = grad_output_raw.contiguous();
    auto input = input_raw.contiguous();
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor grad_input = empty(shape_vec, input.dtype(), input.device());

    // d(softplus)/dx = sigmoid(beta * x)
    if (input.dtype() == DType::Float32) {
        const float* grad_out_data = grad_output.data<float>();
        const float* in_data = input.data<float>();
        float* grad_in_data = grad_input.data<float>();
        size_t n = input.numel();

        #pragma omp parallel for if(n > ::tenzor::OmpThresholds::matmul())
        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i] * beta;
            float sigmoid_x;
            if (x > threshold) {
                sigmoid_x = 1.0f;
            } else if (x < -threshold) {
                sigmoid_x = std::exp(x);
            } else {
                sigmoid_x = 1.0f / (1.0f + std::exp(-x));
            }
            grad_in_data[i] = grad_out_data[i] * sigmoid_x;
        }
    } else if (input.dtype() == DType::Float64) {
        const double* grad_out_data = grad_output.data<double>();
        const double* in_data = input.data<double>();
        double* grad_in_data = grad_input.data<double>();
        size_t n = input.numel();

        #pragma omp parallel for if(n > ::tenzor::OmpThresholds::matmul())
        for (size_t i = 0; i < n; ++i) {
            double x = in_data[i] * static_cast<double>(beta);
            double thresh = static_cast<double>(threshold);
            double sigmoid_x;
            if (x > thresh) {
                sigmoid_x = 1.0;
            } else if (x < -thresh) {
                sigmoid_x = std::exp(x);
            } else {
                sigmoid_x = 1.0 / (1.0 + std::exp(-x));
            }
            grad_in_data[i] = grad_out_data[i] * sigmoid_x;
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* grad_out_data = grad_output.data<Float16>();
        const Float16* in_data = input.data<Float16>();
        Float16* grad_in_data = grad_input.data<Float16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ::tenzor::OmpThresholds::matmul())
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]) * beta;
            float sigmoid_x;
            if (x > threshold) {
                sigmoid_x = 1.0f;
            } else if (x < -threshold) {
                sigmoid_x = std::exp(x);
            } else {
                sigmoid_x = 1.0f / (1.0f + std::exp(-x));
            }
            grad_in_data[i] = Float16(static_cast<float>(grad_out_data[i]) * sigmoid_x);
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* grad_out_data = grad_output.data<BFloat16>();
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* grad_in_data = grad_input.data<BFloat16>();
        size_t n = input.numel();

        #pragma omp parallel for schedule(static) if(n > ::tenzor::OmpThresholds::matmul())
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]) * beta;
            float sigmoid_x;
            if (x > threshold) {
                sigmoid_x = 1.0f;
            } else if (x < -threshold) {
                sigmoid_x = std::exp(x);
            } else {
                sigmoid_x = 1.0f / (1.0f + std::exp(-x));
            }
            grad_in_data[i] = BFloat16(static_cast<float>(grad_out_data[i]) * sigmoid_x);
        }
    } else {
        throw std::runtime_error("Softplus backward only supports Float32, Float64, Float16, and BFloat16");
    }

    return grad_input;
}

// ============================================================================
// In-place activation kernels
// ============================================================================

auto relu_inplace_kernel(Tensor& input) -> void {
    // In-place kernels iterate with flat data[i] and cannot materialise a
    // contiguous copy (the result must be written back into `input` itself),
    // so a non-contiguous view (transpose/permute/slice) would read/write the
    // wrong logical elements. Reject it explicitly; callers must contiguify.
    if (!input.is_contiguous()) {
        throw std::runtime_error("relu_inplace: input must be contiguous");
    }
    if (input.dtype() == DType::Float32) {
        float* data = input.data<float>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            data[i] = data[i] > 0.0f ? data[i] : 0.0f;
        }
    } else if (input.dtype() == DType::Float64) {
        double* data = input.data<double>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            data[i] = data[i] > 0.0 ? data[i] : 0.0;
        }
    } else if (input.dtype() == DType::Float16) {
        Float16* data = input.data<Float16>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(data[i]);
            data[i] = Float16(val > 0.0f ? val : 0.0f);
        }
    } else if (input.dtype() == DType::BFloat16) {
        BFloat16* data = input.data<BFloat16>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(data[i]);
            data[i] = BFloat16(val > 0.0f ? val : 0.0f);
        }
    } else {
        throw std::runtime_error("relu_inplace: Unsupported dtype");
    }
}

auto sigmoid_inplace_kernel(Tensor& input) -> void {
    if (!input.is_contiguous()) {
        throw std::runtime_error("sigmoid_inplace: input must be contiguous");
    }
    if (input.dtype() == DType::Float32) {
        float* data = input.data<float>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            data[i] = 1.0f / (1.0f + std::exp(-data[i]));
        }
    } else if (input.dtype() == DType::Float64) {
        double* data = input.data<double>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            data[i] = 1.0 / (1.0 + std::exp(-data[i]));
        }
    } else if (input.dtype() == DType::Float16) {
        Float16* data = input.data<Float16>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(data[i]);
            data[i] = Float16(1.0f / (1.0f + std::exp(-val)));
        }
    } else if (input.dtype() == DType::BFloat16) {
        BFloat16* data = input.data<BFloat16>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(data[i]);
            data[i] = BFloat16(1.0f / (1.0f + std::exp(-val)));
        }
    } else {
        throw std::runtime_error("sigmoid_inplace: Unsupported dtype");
    }
}

auto tanh_inplace_kernel(Tensor& input) -> void {
    if (!input.is_contiguous()) {
        throw std::runtime_error("tanh_inplace: input must be contiguous");
    }
    if (input.dtype() == DType::Float32) {
        float* data = input.data<float>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            data[i] = std::tanh(data[i]);
        }
    } else if (input.dtype() == DType::Float64) {
        double* data = input.data<double>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            data[i] = std::tanh(data[i]);
        }
    } else if (input.dtype() == DType::Float16) {
        Float16* data = input.data<Float16>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(data[i]);
            data[i] = Float16(std::tanh(val));
        }
    } else if (input.dtype() == DType::BFloat16) {
        BFloat16* data = input.data<BFloat16>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(data[i]);
            data[i] = BFloat16(std::tanh(val));
        }
    } else {
        throw std::runtime_error("tanh_inplace: Unsupported dtype");
    }
}

auto leaky_relu_inplace_kernel(Tensor& input, double alpha) -> void {
    if (!input.is_contiguous()) {
        throw std::runtime_error("leaky_relu_inplace: input must be contiguous");
    }
    if (input.dtype() == DType::Float32) {
        float* data = input.data<float>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            data[i] = data[i] > 0.0f ? data[i] : alpha * data[i];
        }
    } else if (input.dtype() == DType::Float64) {
        double* data = input.data<double>();
        size_t n = input.numel();
        double alpha_d = static_cast<double>(alpha);
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            data[i] = data[i] > 0.0 ? data[i] : alpha_d * data[i];
        }
    } else if (input.dtype() == DType::Float16) {
        Float16* data = input.data<Float16>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(data[i]);
            data[i] = Float16(static_cast<float>(val > 0.0f ? val : alpha * val));
        }
    } else if (input.dtype() == DType::BFloat16) {
        BFloat16* data = input.data<BFloat16>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float val = static_cast<float>(data[i]);
            data[i] = BFloat16(static_cast<float>(val > 0.0f ? val : alpha * val));
        }
    } else {
        throw std::runtime_error("leaky_relu_inplace: Unsupported dtype");
    }
}

auto gelu_inplace_kernel(Tensor& input) -> void {
    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))) — matches PyTorch default
    // and the (non-inplace) gelu_kernel.
    constexpr double INV_SQRT2_D = 0.7071067811865475244;
    constexpr float  INV_SQRT2_F = 0.70710678f;

    if (!input.is_contiguous()) {
        throw std::runtime_error("gelu_inplace: input must be contiguous");
    }
    if (input.dtype() == DType::Float32) {
        float* data = input.data<float>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = data[i];
            data[i] = 0.5f * x * (1.0f + std::erf(x * INV_SQRT2_F));
        }
    } else if (input.dtype() == DType::Float64) {
        double* data = input.data<double>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            double x = data[i];
            data[i] = 0.5 * x * (1.0 + std::erf(x * INV_SQRT2_D));
        }
    } else if (input.dtype() == DType::Float16) {
        Float16* data = input.data<Float16>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(data[i]);
            data[i] = Float16(0.5f * x * (1.0f + std::erf(x * INV_SQRT2_F)));
        }
    } else if (input.dtype() == DType::BFloat16) {
        BFloat16* data = input.data<BFloat16>();
        size_t n = input.numel();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(data[i]);
            data[i] = BFloat16(0.5f * x * (1.0f + std::erf(x * INV_SQRT2_F)));
        }
    } else {
        throw std::runtime_error("gelu_inplace: Unsupported dtype");
    }
}

// ============================================================================
// Hardswish Activation
// ============================================================================
//
// Hardswish(x) = x * clamp(x + 3, 0, 6) / 6
//
// Piecewise:
//   x <= -3:  0
//   -3 < x < 3:  x * (x + 3) / 6
//   x >= 3:  x
//
// Float16 / BFloat16 follow the widen-narrow pattern (compute in float).

auto hardswish_kernel(const Tensor& input_raw) -> Tensor {
    // Flat-pointer iteration requires contiguous input; a non-contiguous
    // slice/transpose/expand view would otherwise read the wrong elements.
    auto input = input_raw.contiguous();
    auto output = Tensor(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                         input.dtype(), input.device());
    auto compute_f32 = [](float x) -> float {
        float t = x + 3.0f;
        if (t < 0.0f) t = 0.0f;
        else if (t > 6.0f) t = 6.0f;
        return x * t * (1.0f / 6.0f);
    };
    auto compute_f64 = [](double x) -> double {
        double t = x + 3.0;
        if (t < 0.0) t = 0.0;
        else if (t > 6.0) t = 6.0;
        return x * t * (1.0 / 6.0);
    };
    size_t n = input.numel();
    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) out_data[i] = compute_f32(in_data[i]);
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) out_data[i] = compute_f64(in_data[i]);
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = output.data<Float16>();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) out_data[i] = Float16(compute_f32(static_cast<float>(in_data[i])));
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = output.data<BFloat16>();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) out_data[i] = BFloat16(compute_f32(static_cast<float>(in_data[i])));
    } else {
        throw std::runtime_error("hardswish_kernel: Unsupported dtype");
    }
    return output;
}

// ============================================================================
// Hardsigmoid Activation
// ============================================================================
//
// Hardsigmoid(x) = clamp(x + 3, 0, 6) / 6
//
// Piecewise:
//   x <= -3:  0
//   -3 < x < 3:  (x + 3) / 6
//   x >= 3:  1

auto hardsigmoid_kernel(const Tensor& input_raw) -> Tensor {
    // Flat-pointer iteration requires contiguous input; a non-contiguous
    // slice/transpose/expand view would otherwise read the wrong elements.
    auto input = input_raw.contiguous();
    auto output = Tensor(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                         input.dtype(), input.device());
    auto compute_f32 = [](float x) -> float {
        float t = x + 3.0f;
        if (t < 0.0f) t = 0.0f;
        else if (t > 6.0f) t = 6.0f;
        return t * (1.0f / 6.0f);
    };
    auto compute_f64 = [](double x) -> double {
        double t = x + 3.0;
        if (t < 0.0) t = 0.0;
        else if (t > 6.0) t = 6.0;
        return t * (1.0 / 6.0);
    };
    size_t n = input.numel();
    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) out_data[i] = compute_f32(in_data[i]);
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) out_data[i] = compute_f64(in_data[i]);
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = output.data<Float16>();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) out_data[i] = Float16(compute_f32(static_cast<float>(in_data[i])));
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = output.data<BFloat16>();
        #pragma omp parallel for schedule(static) if(n > ACTIVATION_OMP_THRESHOLD)
        for (size_t i = 0; i < n; ++i) out_data[i] = BFloat16(compute_f32(static_cast<float>(in_data[i])));
    } else {
        throw std::runtime_error("hardsigmoid_kernel: Unsupported dtype");
    }
    return output;
}

} // namespace cpu
} // namespace tenzor
