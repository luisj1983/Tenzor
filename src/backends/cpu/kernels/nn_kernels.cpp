/**
 * @file nn_kernels.cpp
 * @brief CPU neural network kernel implementations (linear, dropout, embedding, etc.)
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include <tuple>
#include <fstream>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

// Intel oneDNN for optimized layer operations
#ifdef TENZOR_USE_ONEDNN
#include <dnnl.hpp>
#include <list>
#include <unordered_map>
#endif

// Intel MKL for optimized BLAS operations
#ifdef TENZOR_USE_MKL
#include <mkl.h>
#endif

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#define HAS_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define HAS_AVX2 1
#endif

namespace tenzor {
namespace cpu {

// Thread-local RNG for dropout
static thread_local std::mt19937 tl_rng(std::random_device{}());

// Configure optimal thread count once per thread
// Uses physical cores (not hyperthreaded) to avoid contention
inline void configure_threads() {
    static thread_local bool configured = false;
    if (configured) return;
    configured = true;

#ifdef _OPENMP
    unsigned int logical_cores = std::thread::hardware_concurrency();
    unsigned int physical_cores = logical_cores;

    // Detect physical cores via Linux sysfs
    std::ifstream siblings("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list");
    if (siblings.good()) {
        std::string line;
        if (std::getline(siblings, line)) {
            int threads_per_core = 1;
            for (char c : line) {
                if (c == ',') threads_per_core++;
            }
            if (threads_per_core > 1) {
                physical_cores = logical_cores / threads_per_core;
            }
        }
    }

    int num_threads = std::max(1u, physical_cores);
    omp_set_num_threads(num_threads);
#endif
}

#ifdef TENZOR_USE_ONEDNN
// Thread-local oneDNN engine and stream with proper thread configuration
inline dnnl::engine& get_nn_engine() {
    static thread_local std::unique_ptr<dnnl::engine> engine;

    // Ensure threads are configured before using oneDNN
    configure_threads();

    if (!engine) {
        engine = std::make_unique<dnnl::engine>(dnnl::engine::kind::cpu, 0);
    }
    return *engine;
}

inline dnnl::stream& get_nn_stream() {
    static thread_local std::unique_ptr<dnnl::stream> stream;
    if (!stream) {
        stream = std::make_unique<dnnl::stream>(get_nn_engine());
    }
    return *stream;
}

// --------------------------------------------------------------------------
// LayerNorm Primitive Caching (eliminates ~1-5ms primitive creation overhead)
// --------------------------------------------------------------------------
struct LayerNormCacheKey {
    int64_t batch_size;
    int64_t norm_size;

    bool operator==(const LayerNormCacheKey& other) const {
        return batch_size == other.batch_size && norm_size == other.norm_size;
    }
};

struct LayerNormCacheKeyHash {
    size_t operator()(const LayerNormCacheKey& k) const {
        size_t h = std::hash<int64_t>{}(k.batch_size);
        h ^= std::hash<int64_t>{}(k.norm_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct LayerNormCachedPrimitive {
    dnnl::layer_normalization_forward prim;
    dnnl::memory::desc src_md, dst_md, stat_md;
};

static constexpr size_t LAYERNORM_CACHE_SIZE = 32;

class LayerNormPrimitiveCache {
public:
    std::shared_ptr<LayerNormCachedPrimitive> get(const LayerNormCacheKey& key) {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            lru_list_.remove(key);
            lru_list_.push_front(key);
            return it->second;
        }
        return nullptr;
    }

    void put(const LayerNormCacheKey& key, std::shared_ptr<LayerNormCachedPrimitive> value) {
        if (cache_.size() >= LAYERNORM_CACHE_SIZE) {
            auto evict_key = lru_list_.back();
            lru_list_.pop_back();
            cache_.erase(evict_key);
        }
        cache_[key] = value;
        lru_list_.push_front(key);
    }

private:
    std::unordered_map<LayerNormCacheKey, std::shared_ptr<LayerNormCachedPrimitive>, LayerNormCacheKeyHash> cache_;
    std::list<LayerNormCacheKey> lru_list_;
};

static thread_local LayerNormPrimitiveCache g_layernorm_cache;
#endif

#ifdef TENZOR_USE_ONEDNN
// oneDNN-accelerated Linear (inner product) - provides 10-50x speedup
static bool linear_onednn(
    const float* input, const float* weight, const float* bias,
    float* output,
    int64_t batch_size, int64_t in_features, int64_t out_features
) {
    try {
        auto& engine = get_nn_engine();
        auto& stream = get_nn_stream();

        // Create memory descriptors
        dnnl::memory::dims src_dims = {batch_size, in_features};
        dnnl::memory::dims weights_dims = {out_features, in_features};
        dnnl::memory::dims dst_dims = {batch_size, out_features};

        auto src_md = dnnl::memory::desc(src_dims, dnnl::memory::data_type::f32,
                                          dnnl::memory::format_tag::nc);
        auto weights_md = dnnl::memory::desc(weights_dims, dnnl::memory::data_type::f32,
                                              dnnl::memory::format_tag::oi);
        auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32,
                                          dnnl::memory::format_tag::nc);

        // Create inner product primitive descriptor
        dnnl::inner_product_forward::primitive_desc ip_pd;

        if (bias != nullptr) {
            dnnl::memory::dims bias_dims = {out_features};
            auto bias_md = dnnl::memory::desc(bias_dims, dnnl::memory::data_type::f32,
                                               dnnl::memory::format_tag::a);
            ip_pd = dnnl::inner_product_forward::primitive_desc(
                engine,
                dnnl::prop_kind::forward_inference,
                src_md, weights_md, bias_md, dst_md
            );
        } else {
            ip_pd = dnnl::inner_product_forward::primitive_desc(
                engine,
                dnnl::prop_kind::forward_inference,
                src_md, weights_md, dst_md
            );
        }

        // Create memory objects
        auto src_mem = dnnl::memory(src_md, engine, const_cast<float*>(input));
        auto weights_mem = dnnl::memory(weights_md, engine, const_cast<float*>(weight));
        auto dst_mem = dnnl::memory(dst_md, engine, output);

        // Create and execute primitive
        auto ip_prim = dnnl::inner_product_forward(ip_pd);

        if (bias != nullptr) {
            dnnl::memory::dims bias_dims = {out_features};
            auto bias_md = dnnl::memory::desc(bias_dims, dnnl::memory::data_type::f32,
                                               dnnl::memory::format_tag::a);
            auto bias_mem = dnnl::memory(bias_md, engine, const_cast<float*>(bias));
            ip_prim.execute(stream, {
                {DNNL_ARG_SRC, src_mem},
                {DNNL_ARG_WEIGHTS, weights_mem},
                {DNNL_ARG_BIAS, bias_mem},
                {DNNL_ARG_DST, dst_mem}
            });
        } else {
            ip_prim.execute(stream, {
                {DNNL_ARG_SRC, src_mem},
                {DNNL_ARG_WEIGHTS, weights_mem},
                {DNNL_ARG_DST, dst_mem}
            });
        }
        stream.wait();
        return true;
    } catch (...) {
        return false;
    }
}
#endif

#ifdef TENZOR_USE_MKL
// MKL-accelerated Linear using SGEMM
// Computes: output = input @ weight^T + bias
// Input layout:  [batch_size, in_features], stride = in_features
// Weight layout: [out_features, in_features], stride = in_features
// Output layout: [batch_size, out_features], stride = out_features
static bool linear_mkl_float32(
    const float* input, const float* weight, const float* bias,
    float* output,
    int64_t batch_size, int64_t in_features, int64_t out_features
) {
    // Only use MKL for larger matrices where the overhead is worth it
    const int64_t ops = batch_size * in_features * out_features;
    if (ops < 4096) {
        return false;  // Let SIMD fallback handle small matrices
    }

    // C = alpha * A @ B^T + beta * C
    // A = input[batch, in_features], lda = in_features
    // B = weight[out_features, in_features], ldb = in_features (NOT out_features!)
    // C = output[batch, out_features], ldc = out_features
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,     // A is not transposed
        CblasTrans,       // B is transposed (weight^T)
        static_cast<MKL_INT>(batch_size),     // M = rows of output
        static_cast<MKL_INT>(out_features),   // N = cols of output
        static_cast<MKL_INT>(in_features),    // K = reduction dimension
        1.0f,                                  // alpha
        input,                                 // A
        static_cast<MKL_INT>(in_features),    // lda = stride of input rows
        weight,                                // B
        static_cast<MKL_INT>(in_features),    // ldb = stride of weight rows (NOT out_features!)
        0.0f,                                  // beta
        output,                                // C
        static_cast<MKL_INT>(out_features)    // ldc = stride of output rows
    );

    // Add bias if present
    // NOTE: Using simple serial loop instead of OpenMP to avoid thread overhead
    // for small batch sizes. OpenMP parallelization added ~5ms overhead for
    // batch_size * out_features < 100k elements, far exceeding the ~3µs computation.
    if (bias) {
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t o = 0; o < out_features; ++o) {
                output[b * out_features + o] += bias[o];
            }
        }
    }

    return true;
}

// MKL-accelerated Linear for Float64
static bool linear_mkl_float64(
    const double* input, const double* weight, const double* bias,
    double* output,
    int64_t batch_size, int64_t in_features, int64_t out_features
) {
    const int64_t ops = batch_size * in_features * out_features;
    if (ops < 4096) {
        return false;
    }

    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        static_cast<MKL_INT>(batch_size),
        static_cast<MKL_INT>(out_features),
        static_cast<MKL_INT>(in_features),
        1.0,
        input,
        static_cast<MKL_INT>(in_features),
        weight,
        static_cast<MKL_INT>(in_features),
        0.0,
        output,
        static_cast<MKL_INT>(out_features)
    );

    // Add bias if present (serial loop to avoid OpenMP overhead)
    if (bias) {
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t o = 0; o < out_features; ++o) {
                output[b * out_features + o] += bias[o];
            }
        }
    }

    return true;
}
#endif

auto linear_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias) -> Tensor {
    // input: [*, in_features] or [batch, in_features]
    // weight: [out_features, in_features]
    // output: [*, out_features]

    // Ensure tensors are contiguous for optimal MKL/BLAS performance
    // Non-contiguous tensors would produce incorrect results with raw pointer access
    bool input_was_contiguous = input.is_contiguous();
    bool weight_was_contiguous = weight.is_contiguous();

    Tensor input_c = input_was_contiguous ? input : input.contiguous();
    Tensor weight_c = weight_was_contiguous ? weight : weight.contiguous();

    auto in_shape = input_c.shape();
    auto w_shape = weight_c.shape();
    int64_t out_features = w_shape[0];
    int64_t in_features = w_shape[1];

    // Handle batched input
    int64_t batch_size = 1;
    for (size_t i = 0; i < in_shape.size() - 1; ++i) {
        batch_size *= in_shape[i];
    }

    // Build output shape
    std::vector<int64_t> out_shape(in_shape.begin(), in_shape.end() - 1);
    out_shape.push_back(out_features);

    auto output = Tensor::empty_uninitialized(out_shape, input_c.dtype(), input_c.device());

    // Dispatch based on dtype with optimized backends
    if (input_c.dtype() == DType::Float32) {
        const float* in_data = input_c.data<float>();
        const float* w_data = weight_c.data<float>();
        float* out_data = output.data<float>();
        const float* b_data = bias ? bias->data<float>() : nullptr;

#ifdef TENZOR_USE_MKL
        // Try MKL SGEMM first for maximum performance
        if (linear_mkl_float32(in_data, w_data, b_data, out_data, batch_size, in_features, out_features)) {
            return output;
        }
#endif

#ifdef TENZOR_USE_ONEDNN
        // Try oneDNN as fallback
        if (linear_onednn(in_data, w_data, b_data, out_data, batch_size, in_features, out_features)) {
            return output;
        }
#endif

        // Scalar fallback: Y = X @ W^T with OpenMP parallelization
        #pragma omp parallel for collapse(2)
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t o = 0; o < out_features; ++o) {
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int64_t i = 0; i < in_features; ++i) {
                    sum += in_data[b * in_features + i] * w_data[o * in_features + i];
                }
                out_data[b * out_features + o] = sum + (b_data ? b_data[o] : 0.0f);
            }
        }
    } else if (input_c.dtype() == DType::Float64) {
        const double* in_data = input_c.data<double>();
        const double* w_data = weight_c.data<double>();
        double* out_data = output.data<double>();
        const double* b_data = bias ? bias->data<double>() : nullptr;

#ifdef TENZOR_USE_MKL
        // Try MKL DGEMM first
        if (linear_mkl_float64(in_data, w_data, b_data, out_data, batch_size, in_features, out_features)) {
            return output;
        }
#endif

        // Scalar fallback for Float64
        #pragma omp parallel for collapse(2)
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t o = 0; o < out_features; ++o) {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int64_t i = 0; i < in_features; ++i) {
                    sum += in_data[b * in_features + i] * w_data[o * in_features + i];
                }
                out_data[b * out_features + o] = sum + (b_data ? b_data[o] : 0.0);
            }
        }
    } else if (input_c.dtype() == DType::Float16) {
        // Float16: Convert to Float32, compute, convert back
        // This provides correct results at the cost of some overhead
        auto input_f32 = input_c.to(DType::Float32);
        auto weight_f32 = weight_c.to(DType::Float32);
        auto output_f32 = Tensor::empty_uninitialized(out_shape, DType::Float32, input_c.device());
        Tensor bias_f32;
        if (bias) {
            bias_f32 = bias->to(DType::Float32);
        }

        const float* in_data = input_f32.data<float>();
        const float* w_data = weight_f32.data<float>();
        float* out_data = output_f32.data<float>();
        const float* b_data = bias ? bias_f32.data<float>() : nullptr;

        #pragma omp parallel for collapse(2)
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t o = 0; o < out_features; ++o) {
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int64_t i = 0; i < in_features; ++i) {
                    sum += in_data[b * in_features + i] * w_data[o * in_features + i];
                }
                out_data[b * out_features + o] = sum + (b_data ? b_data[o] : 0.0f);
            }
        }

        // Convert result back to Float16
        return output_f32.to(DType::Float16);
    } else {
        // For other dtypes, throw an error rather than crash
        throw std::runtime_error("linear_kernel: Unsupported dtype " +
                                 std::to_string(static_cast<int>(input_c.dtype())));
    }

    return output;
}

// Template for linear backward implementation
template<typename T>
static void linear_backward_impl(
    const T* grad_out_data, const T* in_data, const T* w_data,
    T* grad_in_data, T* grad_w_data, T* grad_b_data,
    int64_t batch_size, int64_t in_features, int64_t out_features
) {
    // grad_input = grad_output @ weight
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < in_features; ++i) {
            T sum = T(0);
            for (int64_t o = 0; o < out_features; ++o) {
                sum += grad_out_data[b * out_features + o] * w_data[o * in_features + i];
            }
            grad_in_data[b * in_features + i] = sum;
        }
    }

    // grad_weight = grad_output^T @ input
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t o = 0; o < out_features; ++o) {
            for (int64_t i = 0; i < in_features; ++i) {
                grad_w_data[o * in_features + i] +=
                    grad_out_data[b * out_features + o] * in_data[b * in_features + i];
            }
        }
    }

    // grad_bias = sum(grad_output, dim=0)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t o = 0; o < out_features; ++o) {
            grad_b_data[o] += grad_out_data[b * out_features + o];
        }
    }
}

auto linear_backward_kernel(const Tensor& grad_output, const Tensor& input,
                             const Tensor& weight) -> std::vector<Tensor> {
    auto in_shape = input.shape();
    auto w_shape = weight.shape();

    int64_t out_features = w_shape[0];
    int64_t in_features = w_shape[1];

    int64_t batch_size = 1;
    for (size_t i = 0; i < in_shape.size() - 1; ++i) {
        batch_size *= in_shape[i];
    }

    // Allocate output tensors with correct dtypes
    auto grad_input = Tensor::empty_uninitialized(
        std::vector<int64_t>(in_shape.begin(), in_shape.end()),
        input.dtype(), input.device());
    auto grad_weight = zeros(
        std::vector<int64_t>(w_shape.begin(), w_shape.end()),
        weight.dtype(), weight.device());
    auto grad_bias = zeros({out_features}, grad_output.dtype(), grad_output.device());

    const int64_t ops = batch_size * in_features * out_features;
    // Use MKL for large enough matrices
    const bool use_mkl = ops > 4096;

    // Dispatch based on dtype
    if (grad_output.dtype() == DType::Float32) {
        const float* grad_out_data = grad_output.data<float>();
        const float* w_data = weight.data<float>();
        const float* in_data = input.data<float>();
        float* grad_in_data = grad_input.data<float>();
        float* grad_w_data = grad_weight.data<float>();
        float* grad_b_data = grad_bias.data<float>();

#ifdef TENZOR_USE_MKL
        if (use_mkl) {
            // grad_input = grad_output @ weight
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<MKL_INT>(batch_size),
                static_cast<MKL_INT>(in_features),
                static_cast<MKL_INT>(out_features),
                1.0f, grad_out_data, static_cast<MKL_INT>(out_features),
                w_data, static_cast<MKL_INT>(in_features),
                0.0f, grad_in_data, static_cast<MKL_INT>(in_features)
            );

            // grad_weight = grad_output^T @ input
            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                static_cast<MKL_INT>(out_features),
                static_cast<MKL_INT>(in_features),
                static_cast<MKL_INT>(batch_size),
                1.0f, grad_out_data, static_cast<MKL_INT>(out_features),
                in_data, static_cast<MKL_INT>(in_features),
                0.0f, grad_w_data, static_cast<MKL_INT>(in_features)
            );

            // grad_bias = sum(grad_output, dim=0)
            // Only use OpenMP for large out_features to avoid thread overhead
            #pragma omp parallel for if(out_features > 1000)
            for (int64_t o = 0; o < out_features; ++o) {
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int64_t b = 0; b < batch_size; ++b) {
                    sum += grad_out_data[b * out_features + o];
                }
                grad_b_data[o] = sum;
            }

            return {grad_input, grad_weight, grad_bias};
        }
#endif
        linear_backward_impl<float>(grad_out_data, in_data, w_data,
                                     grad_in_data, grad_w_data, grad_b_data,
                                     batch_size, in_features, out_features);

    } else if (grad_output.dtype() == DType::Float64) {
        const double* grad_out_data = grad_output.data<double>();
        const double* w_data = weight.data<double>();
        const double* in_data = input.data<double>();
        double* grad_in_data = grad_input.data<double>();
        double* grad_w_data = grad_weight.data<double>();
        double* grad_b_data = grad_bias.data<double>();

#ifdef TENZOR_USE_MKL
        if (use_mkl) {
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<MKL_INT>(batch_size),
                static_cast<MKL_INT>(in_features),
                static_cast<MKL_INT>(out_features),
                1.0, grad_out_data, static_cast<MKL_INT>(out_features),
                w_data, static_cast<MKL_INT>(in_features),
                0.0, grad_in_data, static_cast<MKL_INT>(in_features)
            );

            cblas_dgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                static_cast<MKL_INT>(out_features),
                static_cast<MKL_INT>(in_features),
                static_cast<MKL_INT>(batch_size),
                1.0, grad_out_data, static_cast<MKL_INT>(out_features),
                in_data, static_cast<MKL_INT>(in_features),
                0.0, grad_w_data, static_cast<MKL_INT>(in_features)
            );

            // Only use OpenMP for large out_features to avoid thread overhead
            #pragma omp parallel for if(out_features > 1000)
            for (int64_t o = 0; o < out_features; ++o) {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int64_t b = 0; b < batch_size; ++b) {
                    sum += grad_out_data[b * out_features + o];
                }
                grad_b_data[o] = sum;
            }

            return {grad_input, grad_weight, grad_bias};
        }
#endif
        linear_backward_impl<double>(grad_out_data, in_data, w_data,
                                      grad_in_data, grad_w_data, grad_b_data,
                                      batch_size, in_features, out_features);

    } else {
        // Float16 and other dtypes: use Float32 scalar fallback
        // (MKL doesn't support Float16 directly)
        const float* grad_out_data = grad_output.data<float>();
        const float* w_data = weight.data<float>();
        const float* in_data = input.data<float>();
        float* grad_in_data = grad_input.data<float>();
        float* grad_w_data = grad_weight.data<float>();
        float* grad_b_data = grad_bias.data<float>();

        linear_backward_impl<float>(grad_out_data, in_data, w_data,
                                     grad_in_data, grad_w_data, grad_b_data,
                                     batch_size, in_features, out_features);
    }

    return {grad_input, grad_weight, grad_bias};
}

auto dropout_kernel(const Tensor& input, float p, bool training)
    -> std::pair<Tensor, Tensor> {
    if (!training || p == 0.0f) {
        // During inference or p=0, just return input and empty mask
        return {input, Tensor()};
    }

    auto output = Tensor::empty_uninitialized(
        std::vector<int64_t>(input.shape().begin(), input.shape().end()),
        input.dtype(), input.device());
    auto mask = Tensor::empty_uninitialized(
        std::vector<int64_t>(input.shape().begin(), input.shape().end()),
        DType::Float32, input.device());

    int64_t n = input.numel();
    const float* in_data = input.data<float>();
    float* out_data = output.data<float>();
    float* mask_data = mask.data<float>();

    float scale = 1.0f / (1.0f - p);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    #pragma omp parallel
    {
        std::mt19937 local_rng(tl_rng());
        #pragma omp for
        for (int64_t i = 0; i < n; ++i) {
            float r = dist(local_rng);
            if (r < p) {
                mask_data[i] = 0.0f;
                out_data[i] = 0.0f;
            } else {
                mask_data[i] = scale;
                out_data[i] = in_data[i] * scale;
            }
        }
    }

    return {output, mask};
}

auto dropout_backward_kernel(const Tensor& grad_output, const Tensor& mask, float p) -> Tensor {
    if (!mask.impl() || p == 0.0f) {
        return grad_output;
    }

    auto grad_input = Tensor::empty_uninitialized(
        std::vector<int64_t>(grad_output.shape().begin(), grad_output.shape().end()),
        grad_output.dtype(), grad_output.device());

    int64_t n = grad_output.numel();
    const float* grad_data = grad_output.data<float>();
    const float* mask_data = mask.data<float>();
    float* grad_in_data = grad_input.data<float>();

    #pragma omp parallel for if(n > 10000)
    for (int64_t i = 0; i < n; ++i) {
        grad_in_data[i] = grad_data[i] * mask_data[i];
    }

    return grad_input;
}

auto embedding_kernel(const Tensor& weight, const Tensor& indices) -> Tensor {
    // weight: [num_embeddings, embedding_dim]
    // indices: [*] (any shape of int64 indices)
    // output: [*, embedding_dim]

    auto w_shape = weight.shape();
    auto idx_shape = indices.shape();

    int64_t embedding_dim = w_shape[1];

    std::vector<int64_t> out_shape(idx_shape.begin(), idx_shape.end());
    out_shape.push_back(embedding_dim);

    auto output = Tensor::empty_uninitialized(out_shape, weight.dtype(), weight.device());

    int64_t num_indices = indices.numel();
    const float* w_data = weight.data<float>();
    const int64_t* idx_data = indices.data<int64_t>();
    float* out_data = output.data<float>();

    #pragma omp parallel for if(num_indices * embedding_dim > 10000)
    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = idx_data[i];
        for (int64_t j = 0; j < embedding_dim; ++j) {
            out_data[i * embedding_dim + j] = w_data[idx * embedding_dim + j];
        }
    }

    return output;
}

auto embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                int64_t num_embeddings) -> Tensor {
    auto grad_shape = grad_output.shape();
    int64_t embedding_dim = grad_shape[grad_shape.size() - 1];

    auto grad_weight = zeros({num_embeddings, embedding_dim},
                             grad_output.dtype(), grad_output.device());

    int64_t num_indices = indices.numel();
    const float* grad_data = grad_output.data<float>();
    const int64_t* idx_data = indices.data<int64_t>();
    float* grad_w_data = grad_weight.data<float>();

    // Accumulate gradients (no parallel due to potential race conditions)
    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = idx_data[i];
        for (int64_t j = 0; j < embedding_dim; ++j) {
            grad_w_data[idx * embedding_dim + j] += grad_data[i * embedding_dim + j];
        }
    }

    return grad_weight;
}

#ifdef TENZOR_USE_ONEDNN
// oneDNN-accelerated LayerNorm with primitive caching - provides 10-50x speedup
static bool layer_norm_onednn(
    const float* input, float* output,
    const float* weight, const float* bias,
    int64_t batch_size, int64_t norm_size, float eps
) {
    try {
        auto& engine = get_nn_engine();
        auto& stream = get_nn_stream();

        // Create cache key
        LayerNormCacheKey cache_key{batch_size, norm_size};

        // Try to get cached primitive
        auto cached = g_layernorm_cache.get(cache_key);

        if (!cached) {
            // Cache miss - create new primitive and cache it
            cached = std::make_shared<LayerNormCachedPrimitive>();

            // Create memory descriptors for 2D layout [batch, norm_size]
            dnnl::memory::dims src_dims = {batch_size, norm_size};
            cached->src_md = dnnl::memory::desc(src_dims, dnnl::memory::data_type::f32,
                                                  dnnl::memory::format_tag::nc);
            cached->dst_md = dnnl::memory::desc(src_dims, dnnl::memory::data_type::f32,
                                                  dnnl::memory::format_tag::nc);

            // Weight and bias are 1D [norm_size]
            dnnl::memory::dims stat_dims = {norm_size};
            cached->stat_md = dnnl::memory::desc(stat_dims, dnnl::memory::data_type::f32,
                                                   dnnl::memory::format_tag::a);

            // Create layer normalization primitive descriptor and primitive
            auto lnorm_pd = dnnl::layer_normalization_forward::primitive_desc(
                engine,
                dnnl::prop_kind::forward_inference,
                cached->src_md, cached->dst_md, eps,
                dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift
            );
            cached->prim = dnnl::layer_normalization_forward(lnorm_pd);

            g_layernorm_cache.put(cache_key, cached);
        }

        // Create memory objects with user data (fast - just wraps pointers)
        auto src_mem = dnnl::memory(cached->src_md, engine, const_cast<float*>(input));
        auto dst_mem = dnnl::memory(cached->dst_md, engine, output);
        auto scale_mem = dnnl::memory(cached->stat_md, engine, const_cast<float*>(weight));
        auto shift_mem = dnnl::memory(cached->stat_md, engine, const_cast<float*>(bias));

        // Execute cached primitive
        cached->prim.execute(stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DST, dst_mem},
            {DNNL_ARG_SCALE, scale_mem},
            {DNNL_ARG_SHIFT, shift_mem}
        });
        stream.wait();
        return true;
    } catch (const std::exception& e) {
#ifndef NDEBUG
        std::cerr << "[TENZOR] oneDNN LayerNorm failed: " << e.what() << std::endl;
#endif
        return false;
    } catch (...) {
#ifndef NDEBUG
        std::cerr << "[TENZOR] oneDNN LayerNorm failed with unknown exception" << std::endl;
#endif
        return false;
    }
}
#endif

// SIMD-optimized LayerNorm with statistics output for backward pass
// This version outputs mean and rstd for use in autograd
static void layer_norm_simd_with_stats(
    const float* input, float* output,
    const float* weight, const float* bias,
    float* mean_out, float* rstd_out,
    int64_t batch_size, int64_t norm_size, float eps
) {
    // Configure optimal thread count (no-op after first call)
    configure_threads();

    // Prefetch distance (cache lines ahead)
    constexpr int64_t PREFETCH_DISTANCE = 64;

    // Only parallelize for large total work to avoid thread overhead
    #pragma omp parallel for if(batch_size * norm_size > 10000)
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* in_ptr = input + b * norm_size;
        float* out_ptr = output + b * norm_size;

        // Single-pass mean and variance computation with SIMD
        float sum = 0.0f;
        float sum_sq = 0.0f;

#ifdef HAS_AVX512
        int64_t simd_size = 16;
        __m512 vsum = _mm512_setzero_ps();
        __m512 vsum_sq = _mm512_setzero_ps();
        int64_t i = 0;
        for (; i + simd_size <= norm_size; i += simd_size) {
            if (i + PREFETCH_DISTANCE < norm_size) {
                _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + PREFETCH_DISTANCE), _MM_HINT_T0);
            }
            __m512 v = _mm512_loadu_ps(in_ptr + i);
            vsum = _mm512_add_ps(vsum, v);
            vsum_sq = _mm512_fmadd_ps(v, v, vsum_sq);
        }
        sum = _mm512_reduce_add_ps(vsum);
        sum_sq = _mm512_reduce_add_ps(vsum_sq);
        for (; i < norm_size; ++i) {
            sum += in_ptr[i];
            sum_sq += in_ptr[i] * in_ptr[i];
        }
#elif defined(HAS_AVX2)
        int64_t simd_size = 8;
        __m256 vsum = _mm256_setzero_ps();
        __m256 vsum_sq = _mm256_setzero_ps();
        int64_t i = 0;
        for (; i + simd_size <= norm_size; i += simd_size) {
            if (i + PREFETCH_DISTANCE < norm_size) {
                _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + PREFETCH_DISTANCE), _MM_HINT_T0);
            }
            __m256 v = _mm256_loadu_ps(in_ptr + i);
            vsum = _mm256_add_ps(vsum, v);
            vsum_sq = _mm256_fmadd_ps(v, v, vsum_sq);
        }
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        sum = _mm_cvtss_f32(lo);

        lo = _mm256_castps256_ps128(vsum_sq);
        hi = _mm256_extractf128_ps(vsum_sq, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        sum_sq = _mm_cvtss_f32(lo);

        for (; i < norm_size; ++i) {
            sum += in_ptr[i];
            sum_sq += in_ptr[i] * in_ptr[i];
        }
#else
        for (int64_t i = 0; i < norm_size; ++i) {
            sum += in_ptr[i];
            sum_sq += in_ptr[i] * in_ptr[i];
        }
#endif

        float mean = sum / norm_size;
        float var = (sum_sq / norm_size) - (mean * mean);
        float inv_std = 1.0f / std::sqrt(var + eps);

        // Save statistics for backward pass
        mean_out[b] = mean;
        rstd_out[b] = inv_std;

        // Normalize with SIMD
#ifdef HAS_AVX512
        __m512 vmean = _mm512_set1_ps(mean);
        __m512 vinv_std = _mm512_set1_ps(inv_std);
        i = 0;
        for (; i + 16 <= norm_size; i += 16) {
            if (i + PREFETCH_DISTANCE < norm_size) {
                _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(weight + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(bias + i + PREFETCH_DISTANCE), _MM_HINT_T0);
            }
            __m512 v = _mm512_loadu_ps(in_ptr + i);
            __m512 w = _mm512_loadu_ps(weight + i);
            __m512 b = _mm512_loadu_ps(bias + i);
            __m512 normalized = _mm512_mul_ps(_mm512_sub_ps(v, vmean), vinv_std);
            __m512 result = _mm512_fmadd_ps(normalized, w, b);
            _mm512_storeu_ps(out_ptr + i, result);
        }
        for (; i < norm_size; ++i) {
            float normalized = (in_ptr[i] - mean) * inv_std;
            out_ptr[i] = normalized * weight[i] + bias[i];
        }
#elif defined(HAS_AVX2)
        __m256 vmean = _mm256_set1_ps(mean);
        __m256 vinv_std = _mm256_set1_ps(inv_std);
        i = 0;
        for (; i + 8 <= norm_size; i += 8) {
            if (i + PREFETCH_DISTANCE < norm_size) {
                _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(weight + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(bias + i + PREFETCH_DISTANCE), _MM_HINT_T0);
            }
            __m256 v = _mm256_loadu_ps(in_ptr + i);
            __m256 w = _mm256_loadu_ps(weight + i);
            __m256 b = _mm256_loadu_ps(bias + i);
            __m256 normalized = _mm256_mul_ps(_mm256_sub_ps(v, vmean), vinv_std);
            __m256 result = _mm256_fmadd_ps(normalized, w, b);
            _mm256_storeu_ps(out_ptr + i, result);
        }
        for (; i < norm_size; ++i) {
            float normalized = (in_ptr[i] - mean) * inv_std;
            out_ptr[i] = normalized * weight[i] + bias[i];
        }
#else
        for (int64_t i = 0; i < norm_size; ++i) {
            float normalized = (in_ptr[i] - mean) * inv_std;
            out_ptr[i] = normalized * weight[i] + bias[i];
        }
#endif
    }
}

// SIMD-optimized LayerNorm for when oneDNN is not available or fails
static void layer_norm_simd(
    const float* input, float* output,
    const float* weight, const float* bias,
    int64_t batch_size, int64_t norm_size, float eps
) {
    // Configure optimal thread count (no-op after first call)
    configure_threads();

    // Prefetch distance (cache lines ahead)
    constexpr int64_t PREFETCH_DISTANCE = 64;

    // Only parallelize for large total work to avoid thread overhead
    #pragma omp parallel for if(batch_size * norm_size > 10000)
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* in_ptr = input + b * norm_size;
        float* out_ptr = output + b * norm_size;

        // Single-pass mean and variance computation with SIMD
        float sum = 0.0f;
        float sum_sq = 0.0f;

#ifdef HAS_AVX512
        int64_t simd_size = 16;
        __m512 vsum = _mm512_setzero_ps();
        __m512 vsum_sq = _mm512_setzero_ps();
        int64_t i = 0;
        for (; i + simd_size <= norm_size; i += simd_size) {
            // Prefetch ahead for next iterations
            if (i + PREFETCH_DISTANCE < norm_size) {
                _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + PREFETCH_DISTANCE), _MM_HINT_T0);
            }
            __m512 v = _mm512_loadu_ps(in_ptr + i);
            vsum = _mm512_add_ps(vsum, v);
            vsum_sq = _mm512_fmadd_ps(v, v, vsum_sq);
        }
        sum = _mm512_reduce_add_ps(vsum);
        sum_sq = _mm512_reduce_add_ps(vsum_sq);
        for (; i < norm_size; ++i) {
            sum += in_ptr[i];
            sum_sq += in_ptr[i] * in_ptr[i];
        }
#elif defined(HAS_AVX2)
        int64_t simd_size = 8;
        __m256 vsum = _mm256_setzero_ps();
        __m256 vsum_sq = _mm256_setzero_ps();
        int64_t i = 0;
        for (; i + simd_size <= norm_size; i += simd_size) {
            // Prefetch ahead for next iterations
            if (i + PREFETCH_DISTANCE < norm_size) {
                _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + PREFETCH_DISTANCE), _MM_HINT_T0);
            }
            __m256 v = _mm256_loadu_ps(in_ptr + i);
            vsum = _mm256_add_ps(vsum, v);
            vsum_sq = _mm256_fmadd_ps(v, v, vsum_sq);
        }
        // Horizontal sum
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        sum = _mm_cvtss_f32(lo);

        lo = _mm256_castps256_ps128(vsum_sq);
        hi = _mm256_extractf128_ps(vsum_sq, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        sum_sq = _mm_cvtss_f32(lo);

        for (; i < norm_size; ++i) {
            sum += in_ptr[i];
            sum_sq += in_ptr[i] * in_ptr[i];
        }
#else
        for (int64_t i = 0; i < norm_size; ++i) {
            sum += in_ptr[i];
            sum_sq += in_ptr[i] * in_ptr[i];
        }
#endif

        float mean = sum / norm_size;
        float var = (sum_sq / norm_size) - (mean * mean);
        float inv_std = 1.0f / std::sqrt(var + eps);

        // Normalize with SIMD
#ifdef HAS_AVX512
        __m512 vmean = _mm512_set1_ps(mean);
        __m512 vinv_std = _mm512_set1_ps(inv_std);
        i = 0;
        for (; i + 16 <= norm_size; i += 16) {
            // Prefetch input, weight, bias for next iteration
            if (i + PREFETCH_DISTANCE < norm_size) {
                _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(weight + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(bias + i + PREFETCH_DISTANCE), _MM_HINT_T0);
            }
            __m512 v = _mm512_loadu_ps(in_ptr + i);
            __m512 w = _mm512_loadu_ps(weight + i);
            __m512 b = _mm512_loadu_ps(bias + i);
            __m512 normalized = _mm512_mul_ps(_mm512_sub_ps(v, vmean), vinv_std);
            __m512 result = _mm512_fmadd_ps(normalized, w, b);
            _mm512_storeu_ps(out_ptr + i, result);
        }
        for (; i < norm_size; ++i) {
            float normalized = (in_ptr[i] - mean) * inv_std;
            out_ptr[i] = normalized * weight[i] + bias[i];
        }
#elif defined(HAS_AVX2)
        __m256 vmean = _mm256_set1_ps(mean);
        __m256 vinv_std = _mm256_set1_ps(inv_std);
        i = 0;
        for (; i + 8 <= norm_size; i += 8) {
            // Prefetch input, weight, bias for next iteration
            if (i + PREFETCH_DISTANCE < norm_size) {
                _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(weight + i + PREFETCH_DISTANCE), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(bias + i + PREFETCH_DISTANCE), _MM_HINT_T0);
            }
            __m256 v = _mm256_loadu_ps(in_ptr + i);
            __m256 w = _mm256_loadu_ps(weight + i);
            __m256 b = _mm256_loadu_ps(bias + i);
            __m256 normalized = _mm256_mul_ps(_mm256_sub_ps(v, vmean), vinv_std);
            __m256 result = _mm256_fmadd_ps(normalized, w, b);
            _mm256_storeu_ps(out_ptr + i, result);
        }
        for (; i < norm_size; ++i) {
            float normalized = (in_ptr[i] - mean) * inv_std;
            out_ptr[i] = normalized * weight[i] + bias[i];
        }
#else
        for (int64_t i = 0; i < norm_size; ++i) {
            float normalized = (in_ptr[i] - mean) * inv_std;
            out_ptr[i] = normalized * weight[i] + bias[i];
        }
#endif
    }
}

auto layer_norm_kernel(const Tensor& input, const std::vector<int64_t>& normalized_shape,
                        const Tensor& weight, const Tensor& bias, float eps) -> Tensor {
    // Ensure input is contiguous for optimal memory access patterns
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();

    auto in_shape = input_cont.shape();
    int64_t norm_size = 1;
    for (auto s : normalized_shape) {
        norm_size *= s;
    }
    int64_t batch_size = input_cont.numel() / norm_size;

    auto output = Tensor::empty_uninitialized(
        std::vector<int64_t>(in_shape.begin(), in_shape.end()),
        input_cont.dtype(), input_cont.device());

    const float* in_data = input_cont.data<float>();
    const float* w_data = weight.data<float>();
    const float* b_data = bias.data<float>();
    float* out_data = output.data<float>();

#ifdef TENZOR_USE_ONEDNN
    // Use oneDNN only for very large inputs - SIMD implementation is highly optimized
    // and faster for typical transformer sizes (batch*seq*hidden < 8M)
    // oneDNN primitive cache helps but doesn't beat well-tuned AVX-512 code
    const int64_t total_elements = batch_size * norm_size;
    const bool use_onednn = total_elements >= 8 * 1024 * 1024;  // 8M threshold (original)

    if (use_onednn && input_cont.dtype() == DType::Float32 &&
        layer_norm_onednn(in_data, out_data, w_data, b_data, batch_size, norm_size, eps)) {
        return output;
    }
#endif

    // Use SIMD-optimized implementation for small inputs or as fallback
    layer_norm_simd(in_data, out_data, w_data, b_data, batch_size, norm_size, eps);

    return output;
}

// LayerNorm kernel that also returns mean and rstd for backward pass
// This avoids the double computation issue where forward computes stats,
// then backward re-computes them again
auto layer_norm_kernel_with_stats(const Tensor& input, const std::vector<int64_t>& normalized_shape,
                                   const Tensor& weight, const Tensor& bias, float eps)
    -> std::tuple<Tensor, Tensor, Tensor> {
    // Ensure input is contiguous for optimal memory access patterns
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();

    auto in_shape = input_cont.shape();
    int64_t norm_size = 1;
    for (auto s : normalized_shape) {
        norm_size *= s;
    }
    int64_t batch_size = input_cont.numel() / norm_size;

    // Create output tensor
    auto output = Tensor::empty_uninitialized(
        std::vector<int64_t>(in_shape.begin(), in_shape.end()),
        input_cont.dtype(), input_cont.device());

    // Create mean and rstd tensors (one per batch element)
    auto mean = Tensor::empty_uninitialized({batch_size}, DType::Float32, input_cont.device());
    auto rstd = Tensor::empty_uninitialized({batch_size}, DType::Float32, input_cont.device());

    const float* in_data = input_cont.data<float>();
    const float* w_data = weight.data<float>();
    const float* b_data = bias.data<float>();
    float* out_data = output.data<float>();
    float* mean_data = mean.data<float>();
    float* rstd_data = rstd.data<float>();

    // Use SIMD-optimized implementation that saves statistics
    layer_norm_simd_with_stats(in_data, out_data, w_data, b_data, mean_data, rstd_data,
                                batch_size, norm_size, eps);

    return {output, mean, rstd};
}

auto group_norm_kernel(const Tensor& input, int64_t num_groups,
                        const Tensor& weight, const Tensor& bias, float eps) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < shape.size(); ++i) {
        spatial_size *= shape[i];
    }

    int64_t channels_per_group = C / num_groups;

    auto output = Tensor::empty_uninitialized(
        std::vector<int64_t>(shape.begin(), shape.end()),
        input.dtype(), input.device());

    const float* in_data = input.data<float>();
    const float* w_data = weight.data<float>();
    const float* b_data = bias.data<float>();
    float* out_data = output.data<float>();

    // Only parallelize for large total work to avoid thread overhead
    #pragma omp parallel for collapse(2) if(N * num_groups > 16)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < num_groups; ++g) {
            int64_t c_start = g * channels_per_group;
            int64_t group_size = channels_per_group * spatial_size;

            // Compute mean
            float mean = 0.0f;
            for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                for (int64_t s = 0; s < spatial_size; ++s) {
                    mean += in_data[(n * C + c) * spatial_size + s];
                }
            }
            mean /= group_size;

            // Compute variance
            float var = 0.0f;
            for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                for (int64_t s = 0; s < spatial_size; ++s) {
                    float diff = in_data[(n * C + c) * spatial_size + s] - mean;
                    var += diff * diff;
                }
            }
            var /= group_size;

            float inv_std = 1.0f / std::sqrt(var + eps);

            // Normalize
            for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                for (int64_t s = 0; s < spatial_size; ++s) {
                    int64_t idx = (n * C + c) * spatial_size + s;
                    float normalized = (in_data[idx] - mean) * inv_std;
                    out_data[idx] = normalized * w_data[c] + b_data[c];
                }
            }
        }
    }

    return output;
}

auto instance_norm_kernel(const Tensor& input, const Tensor& weight,
                           const Tensor& bias, float eps) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < shape.size(); ++i) {
        spatial_size *= shape[i];
    }

    auto output = Tensor::empty_uninitialized(
        std::vector<int64_t>(shape.begin(), shape.end()),
        input.dtype(), input.device());

    const float* in_data = input.data<float>();
    const float* w_data = weight.impl() ? weight.data<float>() : nullptr;
    const float* b_data = bias.impl() ? bias.data<float>() : nullptr;
    float* out_data = output.data<float>();

    // Only parallelize for large total work to avoid thread overhead
    #pragma omp parallel for collapse(2) if(N * C > 16)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            // Compute mean
            float mean = 0.0f;
            for (int64_t s = 0; s < spatial_size; ++s) {
                mean += in_data[(n * C + c) * spatial_size + s];
            }
            mean /= spatial_size;

            // Compute variance
            float var = 0.0f;
            for (int64_t s = 0; s < spatial_size; ++s) {
                float diff = in_data[(n * C + c) * spatial_size + s] - mean;
                var += diff * diff;
            }
            var /= spatial_size;

            float inv_std = 1.0f / std::sqrt(var + eps);

            // Normalize
            float w = w_data ? w_data[c] : 1.0f;
            float b = b_data ? b_data[c] : 0.0f;

            for (int64_t s = 0; s < spatial_size; ++s) {
                int64_t idx = (n * C + c) * spatial_size + s;
                float normalized = (in_data[idx] - mean) * inv_std;
                out_data[idx] = normalized * w + b;
            }
        }
    }

    return output;
}

} // namespace cpu
} // namespace tenzor
