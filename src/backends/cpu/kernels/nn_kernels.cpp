/**
 * @file nn_kernels.cpp
 * @brief CPU neural network kernel implementations (linear, dropout, embedding, etc.)
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <bit>
#include <random>
#include <cmath>
#include <iostream>
#include <tuple>
#include <fstream>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif
#include "../cpu_thread_config.hpp"

// Intel oneDNN for optimized layer operations
#ifdef TENZOR_USE_ONEDNN
#include <dnnl.hpp>
#include "onednn_cache.hpp"
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

// Delegate to single source of truth for OMP thread count.
inline void configure_threads() {
    tenzor::backends::cpu::configure_omp_threads();
}

#ifdef TENZOR_USE_ONEDNN
// Engine/stream accessors — delegate to single per-thread instance in onednn_cache.hpp.
inline dnnl::engine& get_nn_engine() {
    // Ensure OMP thread count is configured before first use.
    static thread_local bool threads_configured = false;
    if (!threads_configured) {
        threads_configured = true;
        configure_threads();
    }
    return tenzor::cpu::get_onednn_engine();
}

inline dnnl::stream& get_nn_stream() {
    return tenzor::cpu::get_onednn_stream();
}

// --------------------------------------------------------------------------
// LayerNorm Primitive Caching (eliminates ~1-5ms primitive creation overhead)
// --------------------------------------------------------------------------
struct LayerNormCacheKey {
    int64_t batch_size;
    int64_t norm_size;
    uint32_t eps_bits;  // bit-cast of float eps — avoids NaN-equality concerns
    int32_t  dtype_id;  // static_cast<int>(DType) — future-proofs multi-dtype LN

    bool operator==(const LayerNormCacheKey& other) const {
        return batch_size == other.batch_size
            && norm_size  == other.norm_size
            && eps_bits   == other.eps_bits
            && dtype_id   == other.dtype_id;
    }
};

struct LayerNormCacheKeyHash {
    size_t operator()(const LayerNormCacheKey& k) const {
        size_t h = std::hash<int64_t>{}(k.batch_size);
        h ^= std::hash<int64_t>{}(k.norm_size)  + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(k.eps_bits)  + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int32_t>{}(k.dtype_id)   + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct LayerNormCachedPrimitive {
    dnnl::layer_normalization_forward prim;
    dnnl::memory::desc src_md, dst_md, stat_md;
};

static constexpr size_t LAYERNORM_CACHE_SIZE = 32;

using LayerNormPrimitiveCache = OneDNNPrimitiveCache<LayerNormCacheKey, LayerNormCachedPrimitive, LayerNormCacheKeyHash, LAYERNORM_CACHE_SIZE>;

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
    } else if (input_c.dtype() == DType::BFloat16) {
        // BFloat16: Convert to Float32, compute, convert back
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

#ifdef TENZOR_USE_MKL
        if (linear_mkl_float32(in_data, w_data, b_data, out_data, batch_size, in_features, out_features)) {
            return output_f32.to(DType::BFloat16);
        }
#endif

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

        return output_f32.to(DType::BFloat16);
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

    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        // Float16/BFloat16: convert to Float32, compute, convert back
        DType orig_dtype = grad_output.dtype();
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);

        auto grad_input_f32 = Tensor::empty_uninitialized(
            std::vector<int64_t>(in_shape.begin(), in_shape.end()),
            DType::Float32, input.device());
        auto grad_weight_f32 = zeros(
            std::vector<int64_t>(w_shape.begin(), w_shape.end()),
            DType::Float32, weight.device());
        auto grad_bias_f32 = zeros({out_features}, DType::Float32, grad_output.device());

        const float* grad_out_data = grad_output_f32.data<float>();
        const float* w_data = weight_f32.data<float>();
        const float* in_data = input_f32.data<float>();
        float* grad_in_data = grad_input_f32.data<float>();
        float* grad_w_data = grad_weight_f32.data<float>();
        float* grad_b_data = grad_bias_f32.data<float>();

        linear_backward_impl<float>(grad_out_data, in_data, w_data,
                                     grad_in_data, grad_w_data, grad_b_data,
                                     batch_size, in_features, out_features);

        return {grad_input_f32.to(orig_dtype), grad_weight_f32.to(orig_dtype), grad_bias_f32.to(orig_dtype)};
    } else {
        throw std::runtime_error("linear_backward_kernel: Unsupported dtype " +
                                 std::to_string(static_cast<int>(grad_output.dtype())));
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
    float* mask_data = mask.data<float>();

    float scale = 1.0f / (1.0f - p);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    auto apply_dropout_float = [&](auto* in_data, auto* out_data) {
        using T = std::remove_pointer_t<decltype(in_data)>;
        #pragma omp parallel
        {
            std::mt19937 local_rng(tl_rng());
            #pragma omp for
            for (int64_t i = 0; i < n; ++i) {
                float r = dist(local_rng);
                if (r < p) {
                    mask_data[i] = 0.0f;
                    out_data[i] = T(0);
                } else {
                    mask_data[i] = scale;
                    out_data[i] = in_data[i] * static_cast<T>(scale);
                }
            }
        }
    };

    auto apply_dropout_half = [&](auto* in_data, auto* out_data) {
        using T = std::remove_pointer_t<decltype(in_data)>;
        #pragma omp parallel
        {
            std::mt19937 local_rng(tl_rng());
            #pragma omp for
            for (int64_t i = 0; i < n; ++i) {
                float r = dist(local_rng);
                if (r < p) {
                    mask_data[i] = 0.0f;
                    out_data[i] = T(0.0f);
                } else {
                    mask_data[i] = scale;
                    out_data[i] = T(static_cast<float>(in_data[i]) * scale);
                }
            }
        }
    };

    if (input.dtype() == DType::Float32) {
        apply_dropout_float(input.data<float>(), output.data<float>());
    } else if (input.dtype() == DType::Float64) {
        apply_dropout_float(input.data<double>(), output.data<double>());
    } else if (input.dtype() == DType::Float16) {
        apply_dropout_half(input.data<Float16>(), output.data<Float16>());
    } else if (input.dtype() == DType::BFloat16) {
        apply_dropout_half(input.data<BFloat16>(), output.data<BFloat16>());
    } else {
        throw std::runtime_error("dropout: unsupported dtype");
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
    const float* mask_data = mask.data<float>();

    if (grad_output.dtype() == DType::Float32) {
        const float* grad_data = grad_output.data<float>();
        float* grad_in_data = grad_input.data<float>();
        #pragma omp parallel for if(n > 10000)
        for (int64_t i = 0; i < n; ++i) {
            grad_in_data[i] = grad_data[i] * mask_data[i];
        }
    } else if (grad_output.dtype() == DType::Float64) {
        const double* grad_data = grad_output.data<double>();
        double* grad_in_data = grad_input.data<double>();
        #pragma omp parallel for if(n > 10000)
        for (int64_t i = 0; i < n; ++i) {
            grad_in_data[i] = grad_data[i] * static_cast<double>(mask_data[i]);
        }
    } else if (grad_output.dtype() == DType::Float16) {
        const Float16* grad_data = grad_output.data<Float16>();
        Float16* grad_in_data = grad_input.data<Float16>();
        #pragma omp parallel for if(n > 10000)
        for (int64_t i = 0; i < n; ++i) {
            grad_in_data[i] = Float16(static_cast<float>(grad_data[i]) * mask_data[i]);
        }
    } else if (grad_output.dtype() == DType::BFloat16) {
        const BFloat16* grad_data = grad_output.data<BFloat16>();
        BFloat16* grad_in_data = grad_input.data<BFloat16>();
        #pragma omp parallel for if(n > 10000)
        for (int64_t i = 0; i < n; ++i) {
            grad_in_data[i] = BFloat16(static_cast<float>(grad_data[i]) * mask_data[i]);
        }
    } else {
        throw std::runtime_error("dropout_backward: unsupported dtype");
    }

    return grad_input;
}

auto embedding_kernel(const Tensor& weight, const Tensor& indices, int64_t padding_idx = -1) -> Tensor {
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
    const int64_t* idx_data = indices.data<int64_t>();

    int64_t num_embeddings = w_shape[0];

    // Pre-validate all indices sequentially — throwing inside an OMP parallel
    // region is undefined behavior, so we validate before entering the loop.
    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = idx_data[i];
        if (idx < 0) idx += num_embeddings;
        if (idx < 0 || idx >= num_embeddings) {
            throw std::out_of_range("Embedding index " + std::to_string(idx_data[i]) +
                " out of range [0, " + std::to_string(num_embeddings) + ")");
        }
    }

    auto do_embedding = [&](auto* w_data, auto* out_data) {
        using elem_t = std::remove_pointer_t<decltype(out_data)>;
        #pragma omp parallel for if(num_indices * embedding_dim > 10000)
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = idx_data[i];
            if (idx < 0) idx += num_embeddings;
            if (padding_idx >= 0 && idx == padding_idx) {
                // PaddingIdx: zero the output row (matches PyTorch semantics).
                for (int64_t j = 0; j < embedding_dim; ++j) {
                    out_data[i * embedding_dim + j] = elem_t{};
                }
            } else {
                for (int64_t j = 0; j < embedding_dim; ++j) {
                    out_data[i * embedding_dim + j] = w_data[idx * embedding_dim + j];
                }
            }
        }
    };

    if (weight.dtype() == DType::Float32) {
        do_embedding(weight.data<float>(), output.data<float>());
    } else if (weight.dtype() == DType::Float64) {
        do_embedding(weight.data<double>(), output.data<double>());
    } else if (weight.dtype() == DType::Float16) {
        do_embedding(weight.data<Float16>(), output.data<Float16>());
    } else if (weight.dtype() == DType::BFloat16) {
        do_embedding(weight.data<BFloat16>(), output.data<BFloat16>());
    } else {
        throw std::runtime_error("embedding: unsupported dtype");
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
    const int64_t* idx_data = indices.data<int64_t>();

    // Validate all indices before accumulation (bounds check)
    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = idx_data[i];
        if (idx < 0) idx += num_embeddings;
        if (idx < 0 || idx >= num_embeddings) {
            throw std::out_of_range("Embedding backward: index " + std::to_string(idx_data[i]) +
                " out of range [0, " + std::to_string(num_embeddings) + ")");
        }
    }

    // Note: gradient accumulation is single-threaded to avoid races on duplicate indices.
    // If parallelism is needed, use per-thread buffers or atomic operations.
    if (grad_output.dtype() == DType::Float32) {
        const float* grad_data = grad_output.data<float>();
        float* grad_w_data = grad_weight.data<float>();
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = idx_data[i];
            if (idx < 0) idx += num_embeddings;
            for (int64_t j = 0; j < embedding_dim; ++j) {
                grad_w_data[idx * embedding_dim + j] += grad_data[i * embedding_dim + j];
            }
        }
    } else if (grad_output.dtype() == DType::Float64) {
        const double* grad_data = grad_output.data<double>();
        double* grad_w_data = grad_weight.data<double>();
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = idx_data[i];
            if (idx < 0) idx += num_embeddings;
            for (int64_t j = 0; j < embedding_dim; ++j) {
                grad_w_data[idx * embedding_dim + j] += grad_data[i * embedding_dim + j];
            }
        }
    } else if (grad_output.dtype() == DType::Float16) {
        const Float16* grad_data = grad_output.data<Float16>();
        Float16* grad_w_data = grad_weight.data<Float16>();
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = idx_data[i];
            if (idx < 0) idx += num_embeddings;
            for (int64_t j = 0; j < embedding_dim; ++j) {
                float val = static_cast<float>(grad_w_data[idx * embedding_dim + j]) +
                            static_cast<float>(grad_data[i * embedding_dim + j]);
                grad_w_data[idx * embedding_dim + j] = Float16(val);
            }
        }
    } else if (grad_output.dtype() == DType::BFloat16) {
        const BFloat16* grad_data = grad_output.data<BFloat16>();
        BFloat16* grad_w_data = grad_weight.data<BFloat16>();
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = idx_data[i];
            if (idx < 0) idx += num_embeddings;
            for (int64_t j = 0; j < embedding_dim; ++j) {
                float val = static_cast<float>(grad_w_data[idx * embedding_dim + j]) +
                            static_cast<float>(grad_data[i * embedding_dim + j]);
                grad_w_data[idx * embedding_dim + j] = BFloat16(val);
            }
        }
    } else {
        throw std::runtime_error("embedding_backward: unsupported dtype");
    }

    return grad_weight;
}

// EmbeddingBag forward aggregation kernel
// inputs[0] = embedded tensor [total_elements, embedding_dim]
// inputs[1] = offsets tensor [num_bags] (int64_t)
// attrs: Mode ("sum"/"mean"/"max"), EmbeddingDim, IncludeLastOffset
auto embedding_bag_forward_kernel(std::span<const Tensor> inputs,
                                   const OpAttributes& attrs) -> Tensor {
    const auto& embeddings = inputs[0];
    const auto& offsets = inputs[1];

    int64_t embedding_dim = attrs.get_int(AttrKey::EmbeddingDim, 0);
    std::string mode{attrs.get_string(AttrKey::Mode, "sum")};
    bool include_last_offset = attrs.get_bool(AttrKey::IncludeLastOffset, false);

    int64_t total_elements = embeddings.shape()[0];
    int64_t num_bags = offsets.numel();
    const int64_t* offsets_ptr = offsets.data<int64_t>();

    // If include_last_offset, last element of offsets is the end sentinel
    if (include_last_offset && num_bags > 0) {
        num_bags -= 1;
    }

    auto output = zeros({num_bags, embedding_dim}, embeddings.dtype(), embeddings.device());

    if (embeddings.dtype() == DType::Float32) {
        const float* emb_ptr = embeddings.data<float>();
        float* out_ptr = output.data<float>();

        #pragma omp parallel for if(num_bags > 16)
        for (int64_t bag = 0; bag < num_bags; ++bag) {
            int64_t start = offsets_ptr[bag];
            int64_t end;
            if (bag + 1 < offsets.numel()) {
                end = offsets_ptr[bag + 1];
            } else {
                end = total_elements;
            }
            int64_t bag_size = end - start;
            if (bag_size <= 0) continue;

            float* bag_out = out_ptr + bag * embedding_dim;

            if (mode == "sum" || mode == "mean") {
                for (int64_t i = start; i < end; ++i) {
                    const float* row = emb_ptr + i * embedding_dim;
                    for (int64_t j = 0; j < embedding_dim; ++j) {
                        bag_out[j] += row[j];
                    }
                }
                if (mode == "mean") {
                    for (int64_t j = 0; j < embedding_dim; ++j) {
                        bag_out[j] /= bag_size;
                    }
                }
            } else { // max
                const float* first = emb_ptr + start * embedding_dim;
                for (int64_t j = 0; j < embedding_dim; ++j) {
                    bag_out[j] = first[j];
                }
                for (int64_t i = start + 1; i < end; ++i) {
                    const float* row = emb_ptr + i * embedding_dim;
                    for (int64_t j = 0; j < embedding_dim; ++j) {
                        if (row[j] > bag_out[j]) bag_out[j] = row[j];
                    }
                }
            }
        }
    } else if (embeddings.dtype() == DType::Float64) {
        const double* emb_ptr = embeddings.data<double>();
        double* out_ptr = output.data<double>();

        #pragma omp parallel for if(num_bags > 16)
        for (int64_t bag = 0; bag < num_bags; ++bag) {
            int64_t start = offsets_ptr[bag];
            int64_t end = (bag + 1 < offsets.numel()) ? offsets_ptr[bag + 1] : total_elements;
            int64_t bag_size = end - start;
            if (bag_size <= 0) continue;

            double* bag_out = out_ptr + bag * embedding_dim;
            if (mode == "sum" || mode == "mean") {
                for (int64_t i = start; i < end; ++i) {
                    const double* row = emb_ptr + i * embedding_dim;
                    for (int64_t j = 0; j < embedding_dim; ++j) {
                        bag_out[j] += row[j];
                    }
                }
                if (mode == "mean") {
                    for (int64_t j = 0; j < embedding_dim; ++j) {
                        bag_out[j] /= bag_size;
                    }
                }
            } else {
                const double* first = emb_ptr + start * embedding_dim;
                for (int64_t j = 0; j < embedding_dim; ++j) bag_out[j] = first[j];
                for (int64_t i = start + 1; i < end; ++i) {
                    const double* row = emb_ptr + i * embedding_dim;
                    for (int64_t j = 0; j < embedding_dim; ++j) {
                        if (row[j] > bag_out[j]) bag_out[j] = row[j];
                    }
                }
            }
        }
    } else {
        // Float16/BFloat16: upcast to Float32, compute, downcast
        auto emb_f32 = embeddings.to(DType::Float32);
        std::array<Tensor, 2> f32_inputs = {emb_f32, offsets};
        auto result = embedding_bag_forward_kernel(f32_inputs, attrs);
        return result.to(embeddings.dtype());
    }

    return output;
}

// ============================================================================
// EmbeddingBag Backward — scatter-add gradients back to embedding weight matrix
//
// Inputs:
//   grad_output: [num_bags, embedding_dim]       — upstream gradient
//   indices:     [total_elements]  (Int64)       — original vocabulary indices
//                                                   that produced the bag lookups
//   offsets:     [num_bags] or [num_bags+1]      — bag boundaries (Int64)
//
// Output:
//   grad_weight: [num_embeddings, embedding_dim] — scatter-add into rows
//                                                   selected by `indices`
// ============================================================================
auto embedding_bag_backward_kernel(const Tensor& grad_output,
                                   const Tensor& indices,
                                   const Tensor& offsets,
                                   const OpAttributes& attrs) -> Tensor {
    int64_t num_embeddings = attrs.get_int(AttrKey::NumEmbeddings, 0);
    int64_t embedding_dim = attrs.get_int(AttrKey::EmbeddingDim, 0);
    std::string mode{attrs.get_string(AttrKey::Mode, "sum")};
    bool include_last_offset = attrs.get_bool(AttrKey::IncludeLastOffset, false);

    if (indices.dtype() != DType::Int64) {
        throw std::runtime_error(
            "embedding_bag_backward: indices must be Int64, got " +
            std::string(dtype_name(indices.dtype())));
    }

    int64_t total_elements = indices.numel();
    int64_t num_bags = offsets.numel();
    const int64_t* offsets_ptr = offsets.data<int64_t>();
    const int64_t* indices_ptr = indices.data<int64_t>();

    if (include_last_offset && num_bags > 0) {
        num_bags -= 1;
    }

    // grad_output: [num_bags, embedding_dim]
    // grad_weight: [num_embeddings, embedding_dim] — scatter-add from grad_output

    if (grad_output.dtype() == DType::Float32 || grad_output.dtype() == DType::Float64) {
        auto grad_weight = zeros({num_embeddings, embedding_dim},
                                 grad_output.dtype(), grad_output.device());

        auto scatter_add = [&](auto* gw_data, const auto* go_data) {
            for (int64_t bag = 0; bag < num_bags; ++bag) {
                int64_t start = offsets_ptr[bag];
                int64_t end = (bag + 1 < num_bags + (include_last_offset ? 1 : 0))
                    ? offsets_ptr[bag + 1] : total_elements;
                int64_t bag_size = end - start;
                if (bag_size <= 0) continue;

                for (int64_t i = start; i < end; ++i) {
                    int64_t row = indices_ptr[i];
                    if (row < 0 || row >= num_embeddings) {
                        throw std::runtime_error(
                            "embedding_bag_backward: index " +
                            std::to_string(row) +
                            " out of range [0, " +
                            std::to_string(num_embeddings) + ")");
                    }
                    for (int64_t j = 0; j < embedding_dim; ++j) {
                        auto grad_val = go_data[bag * embedding_dim + j];
                        if (mode == "mean") {
                            grad_val /= static_cast<std::remove_const_t<std::remove_pointer_t<decltype(go_data)>>>(bag_size);
                        }
                        gw_data[row * embedding_dim + j] += grad_val;
                    }
                }
            }
        };

        if (grad_output.dtype() == DType::Float32) {
            scatter_add(grad_weight.data<float>(), grad_output.data<float>());
        } else {
            scatter_add(grad_weight.data<double>(), grad_output.data<double>());
        }

        return grad_weight;
    } else if (grad_output.dtype() == DType::Float16 ||
               grad_output.dtype() == DType::BFloat16) {
        // Float16/BFloat16: upcast to Float32 (indices stays Int64)
        auto go_f32 = grad_output.to(DType::Float32);
        auto result = embedding_bag_backward_kernel(go_f32, indices, offsets, attrs);
        return result.to(grad_output.dtype());
    } else {
        throw std::runtime_error(
            "embedding_bag_backward: unsupported grad_output dtype " +
            std::string(dtype_name(grad_output.dtype())));
    }
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
        LayerNormCacheKey cache_key{
            batch_size,
            norm_size,
            std::bit_cast<uint32_t>(eps),
            static_cast<int32_t>(DType::Float32)  // oneDNN LN path is float32-only
        };

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

        // Phase P0 / Numerical-precision fix: two-pass mean/variance.
        // The previous single-pass form `var = sum_sq/n - mean*mean`
        // catastrophically cancels for inputs with large mean (the
        // standard LayerNorm-on-pre-trained-embedding scenario), often
        // producing negative variance and a NaN inv_std. The two-pass
        // `var = E[(x - mean)^2]` is unconditionally stable.
        // Accumulators are double for precision; final mean / inv_std are
        // float (matches the saved-stats Float32 contract).

        // -------- Pass 1: sum -> mean --------
        double sum_d = 0.0;
#ifdef HAS_AVX512
        int64_t simd_size = 16;
        __m512 vsum = _mm512_setzero_ps();
        int64_t i = 0;
        for (; i + simd_size <= norm_size; i += simd_size) {
            if (i + PREFETCH_DISTANCE < norm_size) {
                _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + PREFETCH_DISTANCE), _MM_HINT_T0);
            }
            __m512 v = _mm512_loadu_ps(in_ptr + i);
            vsum = _mm512_add_ps(vsum, v);
        }
        sum_d = static_cast<double>(_mm512_reduce_add_ps(vsum));
        for (; i < norm_size; ++i) sum_d += static_cast<double>(in_ptr[i]);
#elif defined(HAS_AVX2)
        int64_t simd_size = 8;
        __m256 vsum = _mm256_setzero_ps();
        int64_t i = 0;
        for (; i + simd_size <= norm_size; i += simd_size) {
            if (i + PREFETCH_DISTANCE < norm_size) {
                _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + PREFETCH_DISTANCE), _MM_HINT_T0);
            }
            __m256 v = _mm256_loadu_ps(in_ptr + i);
            vsum = _mm256_add_ps(vsum, v);
        }
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        sum_d = static_cast<double>(_mm_cvtss_f32(lo));
        for (; i < norm_size; ++i) sum_d += static_cast<double>(in_ptr[i]);
#else
        for (int64_t i = 0; i < norm_size; ++i) sum_d += static_cast<double>(in_ptr[i]);
#endif
        const double mean_d = sum_d / static_cast<double>(norm_size);

        // -------- Pass 2: sum((x - mean)^2) -> var --------
        double sum_sq_d = 0.0;
#ifdef HAS_AVX512
        const __m512 vmean512 = _mm512_set1_ps(static_cast<float>(mean_d));
        __m512 vss = _mm512_setzero_ps();
        int64_t j = 0;
        for (; j + simd_size <= norm_size; j += simd_size) {
            __m512 v = _mm512_loadu_ps(in_ptr + j);
            __m512 d = _mm512_sub_ps(v, vmean512);
            vss = _mm512_fmadd_ps(d, d, vss);
        }
        sum_sq_d = static_cast<double>(_mm512_reduce_add_ps(vss));
        for (; j < norm_size; ++j) {
            const double d = static_cast<double>(in_ptr[j]) - mean_d;
            sum_sq_d += d * d;
        }
#elif defined(HAS_AVX2)
        const __m256 vmean256 = _mm256_set1_ps(static_cast<float>(mean_d));
        __m256 vss = _mm256_setzero_ps();
        int64_t j = 0;
        for (; j + simd_size <= norm_size; j += simd_size) {
            __m256 v = _mm256_loadu_ps(in_ptr + j);
            __m256 d = _mm256_sub_ps(v, vmean256);
            vss = _mm256_fmadd_ps(d, d, vss);
        }
        {
            __m128 lo2 = _mm256_castps256_ps128(vss);
            __m128 hi2 = _mm256_extractf128_ps(vss, 1);
            lo2 = _mm_add_ps(lo2, hi2);
            lo2 = _mm_hadd_ps(lo2, lo2);
            lo2 = _mm_hadd_ps(lo2, lo2);
            sum_sq_d = static_cast<double>(_mm_cvtss_f32(lo2));
        }
        for (; j < norm_size; ++j) {
            const double d = static_cast<double>(in_ptr[j]) - mean_d;
            sum_sq_d += d * d;
        }
#else
        for (int64_t j = 0; j < norm_size; ++j) {
            const double d = static_cast<double>(in_ptr[j]) - mean_d;
            sum_sq_d += d * d;
        }
#endif

        const double var_d = sum_sq_d / static_cast<double>(norm_size);
        const float mean = static_cast<float>(mean_d);
        const float inv_std = static_cast<float>(
            1.0 / std::sqrt(var_d + static_cast<double>(eps)));

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

        // Phase P0 / Numerical-precision fix: two-pass mean/variance.
        // See layer_norm_simd_with_stats above for rationale — one-pass
        // `sum_sq/n - mean*mean` catastrophically cancels for inputs with
        // large mean. Two-pass `E[(x-mean)^2]` is unconditionally stable.

        // -------- Pass 1: sum -> mean --------
        double sum_d = 0.0;
#ifdef HAS_AVX512
        int64_t simd_size = 16;
        __m512 vsum = _mm512_setzero_ps();
        int64_t i = 0;
        for (; i + simd_size <= norm_size; i += simd_size) {
            if (i + PREFETCH_DISTANCE < norm_size) {
                _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + PREFETCH_DISTANCE), _MM_HINT_T0);
            }
            __m512 v = _mm512_loadu_ps(in_ptr + i);
            vsum = _mm512_add_ps(vsum, v);
        }
        sum_d = static_cast<double>(_mm512_reduce_add_ps(vsum));
        for (; i < norm_size; ++i) sum_d += static_cast<double>(in_ptr[i]);
#elif defined(HAS_AVX2)
        int64_t simd_size = 8;
        __m256 vsum = _mm256_setzero_ps();
        int64_t i = 0;
        for (; i + simd_size <= norm_size; i += simd_size) {
            if (i + PREFETCH_DISTANCE < norm_size) {
                _mm_prefetch(reinterpret_cast<const char*>(in_ptr + i + PREFETCH_DISTANCE), _MM_HINT_T0);
            }
            __m256 v = _mm256_loadu_ps(in_ptr + i);
            vsum = _mm256_add_ps(vsum, v);
        }
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        sum_d = static_cast<double>(_mm_cvtss_f32(lo));
        for (; i < norm_size; ++i) sum_d += static_cast<double>(in_ptr[i]);
#else
        for (int64_t i = 0; i < norm_size; ++i) sum_d += static_cast<double>(in_ptr[i]);
#endif
        const double mean_d = sum_d / static_cast<double>(norm_size);

        // -------- Pass 2: sum((x - mean)^2) -> var --------
        double sum_sq_d = 0.0;
#ifdef HAS_AVX512
        const __m512 vmean512 = _mm512_set1_ps(static_cast<float>(mean_d));
        __m512 vss = _mm512_setzero_ps();
        int64_t j = 0;
        for (; j + simd_size <= norm_size; j += simd_size) {
            __m512 v = _mm512_loadu_ps(in_ptr + j);
            __m512 d = _mm512_sub_ps(v, vmean512);
            vss = _mm512_fmadd_ps(d, d, vss);
        }
        sum_sq_d = static_cast<double>(_mm512_reduce_add_ps(vss));
        for (; j < norm_size; ++j) {
            const double d = static_cast<double>(in_ptr[j]) - mean_d;
            sum_sq_d += d * d;
        }
#elif defined(HAS_AVX2)
        const __m256 vmean256 = _mm256_set1_ps(static_cast<float>(mean_d));
        __m256 vss = _mm256_setzero_ps();
        int64_t j = 0;
        for (; j + simd_size <= norm_size; j += simd_size) {
            __m256 v = _mm256_loadu_ps(in_ptr + j);
            __m256 d = _mm256_sub_ps(v, vmean256);
            vss = _mm256_fmadd_ps(d, d, vss);
        }
        {
            __m128 lo2 = _mm256_castps256_ps128(vss);
            __m128 hi2 = _mm256_extractf128_ps(vss, 1);
            lo2 = _mm_add_ps(lo2, hi2);
            lo2 = _mm_hadd_ps(lo2, lo2);
            lo2 = _mm_hadd_ps(lo2, lo2);
            sum_sq_d = static_cast<double>(_mm_cvtss_f32(lo2));
        }
        for (; j < norm_size; ++j) {
            const double d = static_cast<double>(in_ptr[j]) - mean_d;
            sum_sq_d += d * d;
        }
#else
        for (int64_t j = 0; j < norm_size; ++j) {
            const double d = static_cast<double>(in_ptr[j]) - mean_d;
            sum_sq_d += d * d;
        }
#endif

        const double var_d = sum_sq_d / static_cast<double>(norm_size);
        const float mean = static_cast<float>(mean_d);
        const float inv_std = static_cast<float>(
            1.0 / std::sqrt(var_d + static_cast<double>(eps)));

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

// Promote a value of the half-precision or double dtype family to `double`
// for accumulation. Direct `static_cast<double>` works for `double` but not
// for `Float16`/`BFloat16` (those define implicit conversion to `float`,
// not to `double`); going through `float` is lossless for the half types
// since `float` is a strict superset of both.
template<typename T>
inline auto ln_to_double(const T& v) -> double {
    if constexpr (std::is_same_v<T, double>) {
        return v;
    } else {
        return static_cast<double>(static_cast<float>(v));
    }
}

// Scalar layer norm implementation for non-Float32 dtypes.
//
// Accumulators are `double` regardless of T: Float32 inputs gain numerical
// stability for free, and Float64 inputs no longer silently truncate to
// Float32 precision (the MEMORY.md "Float32-accumulator-inside-template<T>"
// pattern — Phase P0 / Fix 3 of the audit cleanup).
template<typename T>
void layer_norm_scalar(const T* in_data, T* out_data, const T* w_data, const T* b_data,
                       int64_t batch_size, int64_t norm_size, float eps) {
    #pragma omp parallel for if(batch_size > 16)
    for (int64_t b = 0; b < batch_size; ++b) {
        const T* in_ptr = in_data + b * norm_size;
        T* out_ptr = out_data + b * norm_size;

        // Two-pass mean/variance, double accumulation.
        double sum = 0.0;
        for (int64_t i = 0; i < norm_size; ++i) {
            sum += ln_to_double(in_ptr[i]);
        }
        const double mean = sum / static_cast<double>(norm_size);

        double var = 0.0;
        for (int64_t i = 0; i < norm_size; ++i) {
            const double diff = ln_to_double(in_ptr[i]) - mean;
            var += diff * diff;
        }
        var /= static_cast<double>(norm_size);
        const double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));

        for (int64_t i = 0; i < norm_size; ++i) {
            const double normalized = (ln_to_double(in_ptr[i]) - mean) * inv_std;
            const double result =
                normalized * ln_to_double(w_data[i]) + ln_to_double(b_data[i]);
            // T = double: assign directly. T = Float16/BFloat16: narrow via
            // float (their single-argument ctors take float and are
            // ambiguous from double).
            if constexpr (std::is_same_v<T, double>) {
                out_ptr[i] = result;
            } else {
                out_ptr[i] = static_cast<T>(static_cast<float>(result));
            }
        }
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

    if (input_cont.dtype() == DType::Float32) {
        const float* in_data = input_cont.data<float>();
        const float* w_data = weight.data<float>();
        const float* b_data = bias.data<float>();
        float* out_data = output.data<float>();

#ifdef TENZOR_USE_ONEDNN
        const int64_t total_elements = batch_size * norm_size;
        const bool use_onednn = total_elements >= 8 * 1024 * 1024;

        if (use_onednn &&
            layer_norm_onednn(in_data, out_data, w_data, b_data, batch_size, norm_size, eps)) {
            return output;
        }
#endif

        layer_norm_simd(in_data, out_data, w_data, b_data, batch_size, norm_size, eps);
    } else if (input_cont.dtype() == DType::Float64) {
        layer_norm_scalar<double>(input_cont.data<double>(), output.data<double>(),
                                  weight.data<double>(), bias.data<double>(),
                                  batch_size, norm_size, eps);
    } else if (input_cont.dtype() == DType::Float16) {
        layer_norm_scalar<Float16>(input_cont.data<Float16>(), output.data<Float16>(),
                                   weight.data<Float16>(), bias.data<Float16>(),
                                   batch_size, norm_size, eps);
    } else if (input_cont.dtype() == DType::BFloat16) {
        layer_norm_scalar<BFloat16>(input_cont.data<BFloat16>(), output.data<BFloat16>(),
                                    weight.data<BFloat16>(), bias.data<BFloat16>(),
                                    batch_size, norm_size, eps);
    } else {
        throw std::runtime_error("Unsupported dtype for layer_norm");
    }

    return output;
}

// Same fix as layer_norm_scalar plus saves mean/rstd for backward. Saved
// stats are stored Float32 (external buffer type), but accumulation +
// normalization remain double-precision throughout — matching PyTorch's
// LayerNorm reference implementation.
template<typename T>
void layer_norm_scalar_with_stats(const T* in_data, T* out_data, const T* w_data, const T* b_data,
                                   float* mean_data, float* rstd_data,
                                   int64_t batch_size, int64_t norm_size, float eps) {
    #pragma omp parallel for if(batch_size > 16)
    for (int64_t b = 0; b < batch_size; ++b) {
        const T* in_ptr = in_data + b * norm_size;
        T* out_ptr = out_data + b * norm_size;

        double sum = 0.0;
        for (int64_t i = 0; i < norm_size; ++i) {
            sum += ln_to_double(in_ptr[i]);
        }
        const double mean = sum / static_cast<double>(norm_size);

        double var = 0.0;
        for (int64_t i = 0; i < norm_size; ++i) {
            const double diff = ln_to_double(in_ptr[i]) - mean;
            var += diff * diff;
        }
        var /= static_cast<double>(norm_size);
        const double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));

        // Saved stats are Float32 buffers per the kernel-registry contract;
        // narrow at the boundary only.
        mean_data[b] = static_cast<float>(mean);
        rstd_data[b] = static_cast<float>(inv_std);

        for (int64_t i = 0; i < norm_size; ++i) {
            const double normalized = (ln_to_double(in_ptr[i]) - mean) * inv_std;
            const double result =
                normalized * ln_to_double(w_data[i]) + ln_to_double(b_data[i]);
            // T = double: assign directly. T = Float16/BFloat16: narrow via
            // float (their single-argument ctors take float and are
            // ambiguous from double).
            if constexpr (std::is_same_v<T, double>) {
                out_ptr[i] = result;
            } else {
                out_ptr[i] = static_cast<T>(static_cast<float>(result));
            }
        }
    }
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

    // Create mean and rstd tensors (one per batch element) - always Float32
    auto mean = Tensor::empty_uninitialized({batch_size}, DType::Float32, input_cont.device());
    auto rstd = Tensor::empty_uninitialized({batch_size}, DType::Float32, input_cont.device());
    float* mean_data = mean.data<float>();
    float* rstd_data = rstd.data<float>();

    if (input_cont.dtype() == DType::Float32) {
        const float* in_data = input_cont.data<float>();
        const float* w_data = weight.data<float>();
        const float* b_data = bias.data<float>();
        float* out_data = output.data<float>();

        layer_norm_simd_with_stats(in_data, out_data, w_data, b_data, mean_data, rstd_data,
                                    batch_size, norm_size, eps);
    } else if (input_cont.dtype() == DType::Float64) {
        layer_norm_scalar_with_stats<double>(input_cont.data<double>(), output.data<double>(),
                                             weight.data<double>(), bias.data<double>(),
                                             mean_data, rstd_data, batch_size, norm_size, eps);
    } else if (input_cont.dtype() == DType::Float16) {
        layer_norm_scalar_with_stats<Float16>(input_cont.data<Float16>(), output.data<Float16>(),
                                              weight.data<Float16>(), bias.data<Float16>(),
                                              mean_data, rstd_data, batch_size, norm_size, eps);
    } else if (input_cont.dtype() == DType::BFloat16) {
        layer_norm_scalar_with_stats<BFloat16>(input_cont.data<BFloat16>(), output.data<BFloat16>(),
                                               weight.data<BFloat16>(), bias.data<BFloat16>(),
                                               mean_data, rstd_data, batch_size, norm_size, eps);
    } else {
        throw std::runtime_error("Unsupported dtype for layer_norm_with_stats");
    }

    return {output, mean, rstd};
}

// Phase P0 / Numerical-precision fix: double-precision accumulation regardless
// of T. Same bug class + fix as the layer_norm_scalar accumulator (see commit
// 3a713e1a). Variance uses the two-pass form E[(X-μ)²] which is stable for
// arbitrary mean magnitude; the one-pass E[X²] - E[X]² form would
// catastrophically cancel for large means.
template<typename T>
void group_norm_impl(const T* in_data, T* out_data, const T* w_data, const T* b_data,
                     int64_t N, int64_t C, int64_t spatial_size, int64_t num_groups, float eps) {
    int64_t channels_per_group = C / num_groups;

    #pragma omp parallel for collapse(2) if(N * num_groups > 16)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < num_groups; ++g) {
            int64_t c_start = g * channels_per_group;
            int64_t group_size = channels_per_group * spatial_size;

            double sum = 0.0;
            for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                for (int64_t s = 0; s < spatial_size; ++s) {
                    sum += ln_to_double(in_data[(n * C + c) * spatial_size + s]);
                }
            }
            const double mean = sum / static_cast<double>(group_size);

            double var = 0.0;
            for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                for (int64_t s = 0; s < spatial_size; ++s) {
                    const double diff =
                        ln_to_double(in_data[(n * C + c) * spatial_size + s]) - mean;
                    var += diff * diff;
                }
            }
            var /= static_cast<double>(group_size);
            const double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));

            for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                for (int64_t s = 0; s < spatial_size; ++s) {
                    int64_t idx = (n * C + c) * spatial_size + s;
                    const double normalized = (ln_to_double(in_data[idx]) - mean) * inv_std;
                    const double result =
                        normalized * ln_to_double(w_data[c]) + ln_to_double(b_data[c]);
                    if constexpr (std::is_same_v<T, double>) {
                        out_data[idx] = result;
                    } else {
                        out_data[idx] = static_cast<T>(static_cast<float>(result));
                    }
                }
            }
        }
    }
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

    auto output = Tensor::empty_uninitialized(
        std::vector<int64_t>(shape.begin(), shape.end()),
        input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        group_norm_impl<float>(input.data<float>(), output.data<float>(),
                               weight.data<float>(), bias.data<float>(),
                               N, C, spatial_size, num_groups, eps);
    } else if (input.dtype() == DType::Float64) {
        group_norm_impl<double>(input.data<double>(), output.data<double>(),
                                weight.data<double>(), bias.data<double>(),
                                N, C, spatial_size, num_groups, eps);
    } else if (input.dtype() == DType::Float16) {
        group_norm_impl<Float16>(input.data<Float16>(), output.data<Float16>(),
                                 weight.data<Float16>(), bias.data<Float16>(),
                                 N, C, spatial_size, num_groups, eps);
    } else if (input.dtype() == DType::BFloat16) {
        group_norm_impl<BFloat16>(input.data<BFloat16>(), output.data<BFloat16>(),
                                  weight.data<BFloat16>(), bias.data<BFloat16>(),
                                  N, C, spatial_size, num_groups, eps);
    } else {
        throw std::runtime_error("Unsupported dtype for group_norm");
    }

    return output;
}

template<typename T>
void instance_norm_impl(const T* in_data, T* out_data, const T* w_data, const T* b_data,
                         int64_t N, int64_t C, int64_t spatial_size, float eps) {
    // Phase P0 / Numerical-precision fix: double-precision accumulation +
    // two-pass variance. Same fix class as group_norm_impl above.
    #pragma omp parallel for collapse(2) if(N * C > 16)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            double sum = 0.0;
            for (int64_t s = 0; s < spatial_size; ++s) {
                sum += ln_to_double(in_data[(n * C + c) * spatial_size + s]);
            }
            const double mean = sum / static_cast<double>(spatial_size);

            double var = 0.0;
            for (int64_t s = 0; s < spatial_size; ++s) {
                const double diff =
                    ln_to_double(in_data[(n * C + c) * spatial_size + s]) - mean;
                var += diff * diff;
            }
            var /= static_cast<double>(spatial_size);
            const double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));

            const double w = w_data ? ln_to_double(w_data[c]) : 1.0;
            const double b = b_data ? ln_to_double(b_data[c]) : 0.0;

            for (int64_t s = 0; s < spatial_size; ++s) {
                int64_t idx = (n * C + c) * spatial_size + s;
                const double normalized = (ln_to_double(in_data[idx]) - mean) * inv_std;
                const double result = normalized * w + b;
                if constexpr (std::is_same_v<T, double>) {
                    out_data[idx] = result;
                } else {
                    out_data[idx] = static_cast<T>(static_cast<float>(result));
                }
            }
        }
    }
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

    if (input.dtype() == DType::Float32) {
        instance_norm_impl<float>(input.data<float>(), output.data<float>(),
                                  weight.impl() ? weight.data<float>() : nullptr,
                                  bias.impl() ? bias.data<float>() : nullptr,
                                  N, C, spatial_size, eps);
    } else if (input.dtype() == DType::Float64) {
        instance_norm_impl<double>(input.data<double>(), output.data<double>(),
                                   weight.impl() ? weight.data<double>() : nullptr,
                                   bias.impl() ? bias.data<double>() : nullptr,
                                   N, C, spatial_size, eps);
    } else if (input.dtype() == DType::Float16) {
        instance_norm_impl<Float16>(input.data<Float16>(), output.data<Float16>(),
                                    weight.impl() ? weight.data<Float16>() : nullptr,
                                    bias.impl() ? bias.data<Float16>() : nullptr,
                                    N, C, spatial_size, eps);
    } else if (input.dtype() == DType::BFloat16) {
        instance_norm_impl<BFloat16>(input.data<BFloat16>(), output.data<BFloat16>(),
                                     weight.impl() ? weight.data<BFloat16>() : nullptr,
                                     bias.impl() ? bias.data<BFloat16>() : nullptr,
                                     N, C, spatial_size, eps);
    } else {
        throw std::runtime_error("Unsupported dtype for instance_norm");
    }

    return output;
}

// ============================================================================
// GroupNorm / InstanceNorm with saved stats (for backward pass)
// ============================================================================

// audit-2026-05-03 — same Stats-precision fix as instance_norm_impl_with_stats.
// Stats == double when T is Float64 (preserves Float64 gradient precision
// through the autograd path).
template<typename T2, typename Stats>
void group_norm_impl_with_stats(const T2* in_data, T2* out_data, const T2* w_data, const T2* b_data,
                                 Stats* mean_out, Stats* inv_std_out,
                                 int64_t N, int64_t C, int64_t spatial_size, int64_t num_groups, double eps) {
    int64_t channels_per_group = C / num_groups;

    #pragma omp parallel for collapse(2) if(N * num_groups > 16)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < num_groups; ++g) {
            int64_t c_start = g * channels_per_group;
            int64_t group_size = channels_per_group * spatial_size;

            Stats mean = Stats(0);
            for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                for (int64_t s = 0; s < spatial_size; ++s) {
                    mean += static_cast<Stats>(in_data[(n * C + c) * spatial_size + s]);
                }
            }
            mean /= static_cast<Stats>(group_size);

            Stats var = Stats(0);
            for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                for (int64_t s = 0; s < spatial_size; ++s) {
                    Stats diff = static_cast<Stats>(in_data[(n * C + c) * spatial_size + s]) - mean;
                    var += diff * diff;
                }
            }
            var /= static_cast<Stats>(group_size);

            Stats inv_std = Stats(1) / std::sqrt(var + static_cast<Stats>(eps));

            // Save stats
            mean_out[n * num_groups + g] = mean;
            inv_std_out[n * num_groups + g] = inv_std;

            for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                for (int64_t s = 0; s < spatial_size; ++s) {
                    int64_t idx = (n * C + c) * spatial_size + s;
                    Stats normalized = (static_cast<Stats>(in_data[idx]) - mean) * inv_std;
                    out_data[idx] = static_cast<T2>(normalized * static_cast<Stats>(w_data[c]) + static_cast<Stats>(b_data[c]));
                }
            }
        }
    }
}

auto group_norm_kernel_with_stats(const Tensor& input, int64_t num_groups,
                                    const Tensor& weight, const Tensor& bias, float eps) -> std::vector<Tensor> {
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
    DType stats_dtype = (input.dtype() == DType::Float64) ? DType::Float64 : DType::Float32;
    auto mean = Tensor::empty_uninitialized({N, num_groups}, stats_dtype, input.device());
    auto inv_std = Tensor::empty_uninitialized({N, num_groups}, stats_dtype, input.device());

    if (input.dtype() == DType::Float32) {
        group_norm_impl_with_stats<float, float>(input.data<float>(), output.data<float>(),
                                           weight.data<float>(), bias.data<float>(),
                                           mean.data<float>(), inv_std.data<float>(),
                                           N, C, spatial_size, num_groups, eps);
    } else if (input.dtype() == DType::Float64) {
        group_norm_impl_with_stats<double, double>(input.data<double>(), output.data<double>(),
                                            weight.data<double>(), bias.data<double>(),
                                            mean.data<double>(), inv_std.data<double>(),
                                            N, C, spatial_size, num_groups, eps);
    } else if (input.dtype() == DType::Float16) {
        group_norm_impl_with_stats<Float16, float>(input.data<Float16>(), output.data<Float16>(),
                                             weight.data<Float16>(), bias.data<Float16>(),
                                             mean.data<float>(), inv_std.data<float>(),
                                             N, C, spatial_size, num_groups, eps);
    } else if (input.dtype() == DType::BFloat16) {
        group_norm_impl_with_stats<BFloat16, float>(input.data<BFloat16>(), output.data<BFloat16>(),
                                              weight.data<BFloat16>(), bias.data<BFloat16>(),
                                              mean.data<float>(), inv_std.data<float>(),
                                              N, C, spatial_size, num_groups, eps);
    } else {
        throw std::runtime_error("Unsupported dtype for group_norm_with_stats");
    }

    return {output, mean, inv_std};
}

// audit-2026-05-03 Phase 10 — use Stats type for mean/inv_std accumulators.
// Stats == double when T is Float64, otherwise float. Stats type matches the
// dtype of the saved mean/rstd tensors, so the backward kernel reads them
// with the correct precision (previously hardcoded float, dropping Float64
// precision on the autograd path).
template<typename T, typename Stats>
void instance_norm_impl_with_stats(const T* in_data, T* out_data, const T* w_data, const T* b_data,
                                    Stats* mean_out, Stats* inv_std_out,
                                    int64_t N, int64_t C, int64_t spatial_size, double eps) {
    #pragma omp parallel for collapse(2) if(N * C > 16)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            Stats mean = Stats(0);
            for (int64_t s = 0; s < spatial_size; ++s) {
                mean += static_cast<Stats>(in_data[(n * C + c) * spatial_size + s]);
            }
            mean /= static_cast<Stats>(spatial_size);

            Stats var = Stats(0);
            for (int64_t s = 0; s < spatial_size; ++s) {
                Stats diff = static_cast<Stats>(in_data[(n * C + c) * spatial_size + s]) - mean;
                var += diff * diff;
            }
            var /= static_cast<Stats>(spatial_size);

            Stats inv_std = Stats(1) / std::sqrt(var + static_cast<Stats>(eps));

            // Save stats
            mean_out[n * C + c] = mean;
            inv_std_out[n * C + c] = inv_std;

            Stats w = w_data ? static_cast<Stats>(w_data[c]) : Stats(1);
            Stats b = b_data ? static_cast<Stats>(b_data[c]) : Stats(0);

            for (int64_t s = 0; s < spatial_size; ++s) {
                int64_t idx = (n * C + c) * spatial_size + s;
                Stats normalized = (static_cast<Stats>(in_data[idx]) - mean) * inv_std;
                out_data[idx] = static_cast<T>(normalized * w + b);
            }
        }
    }
}

auto instance_norm_kernel_with_stats(const Tensor& input, const Tensor& weight,
                                       const Tensor& bias, float eps) -> std::vector<Tensor> {
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
    // audit-2026-05-03 Phase 10 — store stats at Float64 precision when the
    // input is Float64; previously hardcoded Float32 dropped ~30 mantissa
    // bits, breaking Float64 gradcheck on the autograd path.
    DType stats_dtype = (input.dtype() == DType::Float64) ? DType::Float64 : DType::Float32;
    auto mean = Tensor::empty_uninitialized({N, C}, stats_dtype, input.device());
    auto inv_std = Tensor::empty_uninitialized({N, C}, stats_dtype, input.device());

    if (input.dtype() == DType::Float32) {
        instance_norm_impl_with_stats<float, float>(input.data<float>(), output.data<float>(),
                                              weight.impl() ? weight.data<float>() : nullptr,
                                              bias.impl() ? bias.data<float>() : nullptr,
                                              mean.data<float>(), inv_std.data<float>(),
                                              N, C, spatial_size, eps);
    } else if (input.dtype() == DType::Float64) {
        instance_norm_impl_with_stats<double, double>(input.data<double>(), output.data<double>(),
                                               weight.impl() ? weight.data<double>() : nullptr,
                                               bias.impl() ? bias.data<double>() : nullptr,
                                               mean.data<double>(), inv_std.data<double>(),
                                               N, C, spatial_size, eps);
    } else if (input.dtype() == DType::Float16) {
        instance_norm_impl_with_stats<Float16, float>(input.data<Float16>(), output.data<Float16>(),
                                                weight.impl() ? weight.data<Float16>() : nullptr,
                                                bias.impl() ? bias.data<Float16>() : nullptr,
                                                mean.data<float>(), inv_std.data<float>(),
                                                N, C, spatial_size, eps);
    } else if (input.dtype() == DType::BFloat16) {
        instance_norm_impl_with_stats<BFloat16, float>(input.data<BFloat16>(), output.data<BFloat16>(),
                                                 weight.impl() ? weight.data<BFloat16>() : nullptr,
                                                 bias.impl() ? bias.data<BFloat16>() : nullptr,
                                                 mean.data<float>(), inv_std.data<float>(),
                                                 N, C, spatial_size, eps);
    } else {
        throw std::runtime_error("Unsupported dtype for instance_norm_with_stats");
    }

    return {output, mean, inv_std};
}

// ============================================================================
// LayerNorm Backward
// ============================================================================

auto layer_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                 const std::vector<int64_t>& normalized_shape,
                                 const Tensor& mean, const Tensor& rstd,
                                 const Tensor& weight) -> std::vector<Tensor> {
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    Tensor grad_cont = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();

    auto in_shape = input_cont.shape();
    int64_t norm_size = 1;
    for (auto s : normalized_shape) {
        norm_size *= s;
    }
    int64_t batch_size = input_cont.numel() / norm_size;

    auto grad_input = Tensor::empty_uninitialized(
        std::vector<int64_t>(in_shape.begin(), in_shape.end()),
        input_cont.dtype(), input_cont.device());
    auto grad_weight = zeros(std::vector<int64_t>(normalized_shape.begin(), normalized_shape.end()),
                             weight.dtype(), weight.device());
    auto grad_bias = zeros(std::vector<int64_t>(normalized_shape.begin(), normalized_shape.end()),
                           weight.dtype(), weight.device());

    const float* mean_data = mean.data<float>();
    const float* rstd_data = rstd.data<float>();

    // Dispatch based on input dtype - compute always in float for precision
    if (input_cont.dtype() == DType::Float32) {
        const float* in_data = input_cont.data<float>();
        const float* grad_out_data = grad_cont.data<float>();
        const float* w_data = weight.data<float>();
        float* grad_in_data = grad_input.data<float>();
        float* grad_w_data = grad_weight.data<float>();
        float* grad_b_data = grad_bias.data<float>();

        // Accumulate grad_weight and grad_bias across batch (not thread-safe, so use local buffers)
        std::vector<float> local_grad_w(norm_size, 0.0f);
        std::vector<float> local_grad_b(norm_size, 0.0f);

        #pragma omp parallel if(batch_size > 16)
        {
            std::vector<float> thread_grad_w(norm_size, 0.0f);
            std::vector<float> thread_grad_b(norm_size, 0.0f);

            #pragma omp for
            for (int64_t b = 0; b < batch_size; ++b) {
                const float* in_ptr = in_data + b * norm_size;
                const float* grad_out_ptr = grad_out_data + b * norm_size;
                float* grad_in_ptr = grad_in_data + b * norm_size;

                float m = mean_data[b];
                float r = rstd_data[b];

                // Accumulate ds = sum(dy * w * x_hat) and db = sum(dy * w) over normalized dims
                float ds = 0.0f;
                float db = 0.0f;
                for (int64_t i = 0; i < norm_size; ++i) {
                    float dy = grad_out_ptr[i];
                    float x_hat = (in_ptr[i] - m) * r;
                    float dy_w = dy * w_data[i];
                    ds += dy_w * x_hat;
                    db += dy_w;
                }

                // Compute grad_input
                float inv_n = 1.0f / static_cast<float>(norm_size);
                for (int64_t i = 0; i < norm_size; ++i) {
                    float dy = grad_out_ptr[i];
                    float x_hat = (in_ptr[i] - m) * r;
                    grad_in_ptr[i] = r * (dy * w_data[i] - inv_n * (db + x_hat * ds));
                }

                // Accumulate weight/bias gradients
                for (int64_t i = 0; i < norm_size; ++i) {
                    float x_hat = (in_ptr[i] - m) * r;
                    thread_grad_w[i] += grad_out_ptr[i] * x_hat;
                    thread_grad_b[i] += grad_out_ptr[i];
                }
            }

            #pragma omp critical
            {
                for (int64_t i = 0; i < norm_size; ++i) {
                    grad_w_data[i] += thread_grad_w[i];
                    grad_b_data[i] += thread_grad_b[i];
                }
            }
        }
    } else if (input_cont.dtype() == DType::Float64) {
        const double* in_data = input_cont.data<double>();
        const double* grad_out_data = grad_cont.data<double>();
        const double* w_data = weight.data<double>();
        double* grad_in_data = grad_input.data<double>();
        double* grad_w_data = grad_weight.data<double>();
        double* grad_b_data = grad_bias.data<double>();

        #pragma omp parallel if(batch_size > 16)
        {
            std::vector<double> thread_grad_w(norm_size, 0.0);
            std::vector<double> thread_grad_b(norm_size, 0.0);

            #pragma omp for
            for (int64_t b = 0; b < batch_size; ++b) {
                const double* in_ptr = in_data + b * norm_size;
                const double* grad_out_ptr = grad_out_data + b * norm_size;
                double* grad_in_ptr = grad_in_data + b * norm_size;

                double m = static_cast<double>(mean_data[b]);
                double r = static_cast<double>(rstd_data[b]);

                double ds = 0.0;
                double db = 0.0;
                for (int64_t i = 0; i < norm_size; ++i) {
                    double dy = grad_out_ptr[i];
                    double x_hat = (in_ptr[i] - m) * r;
                    double dy_w = dy * w_data[i];
                    ds += dy_w * x_hat;
                    db += dy_w;
                }

                double inv_n = 1.0 / static_cast<double>(norm_size);
                for (int64_t i = 0; i < norm_size; ++i) {
                    double dy = grad_out_ptr[i];
                    double x_hat = (in_ptr[i] - m) * r;
                    grad_in_ptr[i] = r * (dy * w_data[i] - inv_n * (db + x_hat * ds));
                }

                for (int64_t i = 0; i < norm_size; ++i) {
                    double x_hat = (in_ptr[i] - m) * r;
                    thread_grad_w[i] += grad_out_ptr[i] * x_hat;
                    thread_grad_b[i] += grad_out_ptr[i];
                }
            }

            #pragma omp critical
            {
                for (int64_t i = 0; i < norm_size; ++i) {
                    grad_w_data[i] += thread_grad_w[i];
                    grad_b_data[i] += thread_grad_b[i];
                }
            }
        }
    } else if (input_cont.dtype() == DType::Float16 || input_cont.dtype() == DType::BFloat16) {
        // Compute in Float32, read/write in native dtype
        // Use Float32 intermediate tensors
        auto grad_input_f32 = Tensor::empty_uninitialized(
            std::vector<int64_t>(in_shape.begin(), in_shape.end()), DType::Float32, input_cont.device());
        auto grad_weight_f32 = zeros({norm_size}, DType::Float32, weight.device());
        auto grad_bias_f32 = zeros({norm_size}, DType::Float32, weight.device());

        // Convert input and grad_output to float32
        int64_t total = input_cont.numel();
        std::vector<float> in_f32(total);
        std::vector<float> grad_out_f32(total);
        std::vector<float> w_f32(norm_size);

        if (input_cont.dtype() == DType::Float16) {
            const Float16* in_raw = input_cont.data<Float16>();
            const Float16* grad_raw = grad_cont.data<Float16>();
            const Float16* w_raw = weight.data<Float16>();
            for (int64_t i = 0; i < total; ++i) in_f32[i] = static_cast<float>(in_raw[i]);
            for (int64_t i = 0; i < total; ++i) grad_out_f32[i] = static_cast<float>(grad_raw[i]);
            for (int64_t i = 0; i < norm_size; ++i) w_f32[i] = static_cast<float>(w_raw[i]);
        } else {
            const BFloat16* in_raw = input_cont.data<BFloat16>();
            const BFloat16* grad_raw = grad_cont.data<BFloat16>();
            const BFloat16* w_raw = weight.data<BFloat16>();
            for (int64_t i = 0; i < total; ++i) in_f32[i] = static_cast<float>(in_raw[i]);
            for (int64_t i = 0; i < total; ++i) grad_out_f32[i] = static_cast<float>(grad_raw[i]);
            for (int64_t i = 0; i < norm_size; ++i) w_f32[i] = static_cast<float>(w_raw[i]);
        }

        float* grad_in_f32 = grad_input_f32.data<float>();
        float* grad_w_f32 = grad_weight_f32.data<float>();
        float* grad_b_f32 = grad_bias_f32.data<float>();

        #pragma omp parallel if(batch_size > 16)
        {
            std::vector<float> thread_grad_w(norm_size, 0.0f);
            std::vector<float> thread_grad_b(norm_size, 0.0f);

            #pragma omp for
            for (int64_t b = 0; b < batch_size; ++b) {
                const float* in_ptr = in_f32.data() + b * norm_size;
                const float* grad_out_ptr = grad_out_f32.data() + b * norm_size;
                float* grad_in_ptr = grad_in_f32 + b * norm_size;

                float m = mean_data[b];
                float r = rstd_data[b];

                float ds = 0.0f;
                float db = 0.0f;
                for (int64_t i = 0; i < norm_size; ++i) {
                    float dy = grad_out_ptr[i];
                    float x_hat = (in_ptr[i] - m) * r;
                    float dy_w = dy * w_f32[i];
                    ds += dy_w * x_hat;
                    db += dy_w;
                }

                float inv_n = 1.0f / static_cast<float>(norm_size);
                for (int64_t i = 0; i < norm_size; ++i) {
                    float dy = grad_out_ptr[i];
                    float x_hat = (in_ptr[i] - m) * r;
                    grad_in_ptr[i] = r * (dy * w_f32[i] - inv_n * (db + x_hat * ds));
                }

                for (int64_t i = 0; i < norm_size; ++i) {
                    float x_hat = (in_ptr[i] - m) * r;
                    thread_grad_w[i] += grad_out_ptr[i] * x_hat;
                    thread_grad_b[i] += grad_out_ptr[i];
                }
            }

            #pragma omp critical
            {
                for (int64_t i = 0; i < norm_size; ++i) {
                    grad_w_f32[i] += thread_grad_w[i];
                    grad_b_f32[i] += thread_grad_b[i];
                }
            }
        }

        // Convert back to native dtype
        if (input_cont.dtype() == DType::Float16) {
            Float16* gi = grad_input.data<Float16>();
            Float16* gw = grad_weight.data<Float16>();
            Float16* gb = grad_bias.data<Float16>();
            for (int64_t i = 0; i < total; ++i) gi[i] = Float16(grad_in_f32[i]);
            for (int64_t i = 0; i < norm_size; ++i) gw[i] = Float16(grad_w_f32[i]);
            for (int64_t i = 0; i < norm_size; ++i) gb[i] = Float16(grad_b_f32[i]);
        } else {
            BFloat16* gi = grad_input.data<BFloat16>();
            BFloat16* gw = grad_weight.data<BFloat16>();
            BFloat16* gb = grad_bias.data<BFloat16>();
            for (int64_t i = 0; i < total; ++i) gi[i] = BFloat16(grad_in_f32[i]);
            for (int64_t i = 0; i < norm_size; ++i) gw[i] = BFloat16(grad_w_f32[i]);
            for (int64_t i = 0; i < norm_size; ++i) gb[i] = BFloat16(grad_b_f32[i]);
        }
    } else {
        throw std::runtime_error("Unsupported dtype for layer_norm_backward");
    }

    return {grad_input, grad_weight, grad_bias};
}

// ============================================================================
// GroupNorm Backward
// ============================================================================

auto group_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                 int64_t num_groups, const Tensor& mean,
                                 const Tensor& rstd, const Tensor& weight) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < shape.size(); ++i) {
        spatial_size *= shape[i];
    }
    int64_t channels_per_group = C / num_groups;
    int64_t group_size = channels_per_group * spatial_size;

    auto grad_input = Tensor::empty_uninitialized(
        std::vector<int64_t>(shape.begin(), shape.end()),
        input.dtype(), input.device());
    auto grad_weight = zeros({C}, weight.dtype(), weight.device());
    auto grad_bias = zeros({C}, weight.dtype(), weight.device());

    // audit-2026-05-03 — mean/rstd are stored at input dtype now (was hardcoded
    // Float32, dropping Float64 precision through the autograd path).
    const float* mean_data_f32 = nullptr;
    const float* rstd_data_f32 = nullptr;
    const double* mean_data_f64 = nullptr;
    const double* rstd_data_f64 = nullptr;
    if (mean.dtype() == DType::Float64) {
        mean_data_f64 = mean.data<double>();
        rstd_data_f64 = rstd.data<double>();
    } else {
        mean_data_f32 = mean.data<float>();
        rstd_data_f32 = rstd.data<float>();
    }
    const float* mean_data = mean_data_f32;
    const float* rstd_data = rstd_data_f32;

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        const float* grad_out_data = grad_output.data<float>();
        const float* w_data = weight.data<float>();
        float* grad_in_data = grad_input.data<float>();
        float* grad_w_data = grad_weight.data<float>();
        float* grad_b_data = grad_bias.data<float>();

        // Accumulate grad_weight/grad_bias per channel across batch
        // Use per-channel atomic-free accumulation
        std::vector<float> gw_accum(C, 0.0f);
        std::vector<float> gb_accum(C, 0.0f);

        #pragma omp parallel if(N * num_groups > 16)
        {
            std::vector<float> t_gw(C, 0.0f);
            std::vector<float> t_gb(C, 0.0f);

            #pragma omp for collapse(2)
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t g = 0; g < num_groups; ++g) {
                    int64_t c_start = g * channels_per_group;
                    float m = mean_data[n * num_groups + g];
                    float r = rstd_data[n * num_groups + g];

                    // Pass 1: accumulate ds, db
                    float ds = 0.0f;
                    float db = 0.0f;
                    for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                        for (int64_t s = 0; s < spatial_size; ++s) {
                            int64_t idx = (n * C + c) * spatial_size + s;
                            float dy = grad_out_data[idx];
                            float x_hat = (in_data[idx] - m) * r;
                            float dy_w = dy * w_data[c];
                            ds += dy_w * x_hat;
                            db += dy_w;
                        }
                    }

                    // Pass 2: compute grad_input
                    float inv_gs = 1.0f / static_cast<float>(group_size);
                    for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                        for (int64_t s = 0; s < spatial_size; ++s) {
                            int64_t idx = (n * C + c) * spatial_size + s;
                            float dy = grad_out_data[idx];
                            float x_hat = (in_data[idx] - m) * r;
                            grad_in_data[idx] = r * (dy * w_data[c] - inv_gs * (db + x_hat * ds));
                        }
                    }

                    // Accumulate grad_weight/grad_bias
                    for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                        for (int64_t s = 0; s < spatial_size; ++s) {
                            int64_t idx = (n * C + c) * spatial_size + s;
                            float x_hat = (in_data[idx] - m) * r;
                            t_gw[c] += grad_out_data[idx] * x_hat;
                            t_gb[c] += grad_out_data[idx];
                        }
                    }
                }
            }

            #pragma omp critical
            {
                for (int64_t c = 0; c < C; ++c) {
                    grad_w_data[c] += t_gw[c];
                    grad_b_data[c] += t_gb[c];
                }
            }
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        const double* grad_out_data = grad_output.data<double>();
        const double* w_data = weight.data<double>();
        double* grad_in_data = grad_input.data<double>();
        double* grad_w_data = grad_weight.data<double>();
        double* grad_b_data = grad_bias.data<double>();

        #pragma omp parallel if(N * num_groups > 16)
        {
            std::vector<double> t_gw(C, 0.0);
            std::vector<double> t_gb(C, 0.0);

            #pragma omp for collapse(2)
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t g = 0; g < num_groups; ++g) {
                    int64_t c_start = g * channels_per_group;
                    double m = mean_data_f64
                        ? mean_data_f64[n * num_groups + g]
                        : static_cast<double>(mean_data_f32[n * num_groups + g]);
                    double r = rstd_data_f64
                        ? rstd_data_f64[n * num_groups + g]
                        : static_cast<double>(rstd_data_f32[n * num_groups + g]);

                    double ds = 0.0;
                    double db = 0.0;
                    for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                        for (int64_t s = 0; s < spatial_size; ++s) {
                            int64_t idx = (n * C + c) * spatial_size + s;
                            double dy = grad_out_data[idx];
                            double x_hat = (in_data[idx] - m) * r;
                            double dy_w = dy * w_data[c];
                            ds += dy_w * x_hat;
                            db += dy_w;
                        }
                    }

                    double inv_gs = 1.0 / static_cast<double>(group_size);
                    for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                        for (int64_t s = 0; s < spatial_size; ++s) {
                            int64_t idx = (n * C + c) * spatial_size + s;
                            double dy = grad_out_data[idx];
                            double x_hat = (in_data[idx] - m) * r;
                            grad_in_data[idx] = r * (dy * w_data[c] - inv_gs * (db + x_hat * ds));
                        }
                    }

                    for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                        for (int64_t s = 0; s < spatial_size; ++s) {
                            int64_t idx = (n * C + c) * spatial_size + s;
                            double x_hat = (in_data[idx] - m) * r;
                            t_gw[c] += grad_out_data[idx] * x_hat;
                            t_gb[c] += grad_out_data[idx];
                        }
                    }
                }
            }

            #pragma omp critical
            {
                for (int64_t c = 0; c < C; ++c) {
                    grad_w_data[c] += t_gw[c];
                    grad_b_data[c] += t_gb[c];
                }
            }
        }
    } else {
        // Float16/BFloat16: compute entirely in Float32
        int64_t total = input.numel();
        std::vector<float> in_f32(total), grad_out_f32(total);
        std::vector<float> w_f32(C);

        if (input.dtype() == DType::Float16) {
            const Float16* ir = input.data<Float16>();
            const Float16* gr = grad_output.data<Float16>();
            const Float16* wr = weight.data<Float16>();
            for (int64_t i = 0; i < total; ++i) { in_f32[i] = static_cast<float>(ir[i]); grad_out_f32[i] = static_cast<float>(gr[i]); }
            for (int64_t i = 0; i < C; ++i) w_f32[i] = static_cast<float>(wr[i]);
        } else {
            const BFloat16* ir = input.data<BFloat16>();
            const BFloat16* gr = grad_output.data<BFloat16>();
            const BFloat16* wr = weight.data<BFloat16>();
            for (int64_t i = 0; i < total; ++i) { in_f32[i] = static_cast<float>(ir[i]); grad_out_f32[i] = static_cast<float>(gr[i]); }
            for (int64_t i = 0; i < C; ++i) w_f32[i] = static_cast<float>(wr[i]);
        }

        std::vector<float> grad_in_f32(total, 0.0f);
        std::vector<float> grad_w_f32(C, 0.0f);
        std::vector<float> grad_b_f32(C, 0.0f);

        #pragma omp parallel if(N * num_groups > 16)
        {
            std::vector<float> t_gw(C, 0.0f);
            std::vector<float> t_gb(C, 0.0f);

            #pragma omp for collapse(2)
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t g = 0; g < num_groups; ++g) {
                    int64_t c_start = g * channels_per_group;
                    float m = mean_data[n * num_groups + g];
                    float r = rstd_data[n * num_groups + g];

                    float ds = 0.0f, db = 0.0f;
                    for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                        for (int64_t s = 0; s < spatial_size; ++s) {
                            int64_t idx = (n * C + c) * spatial_size + s;
                            float dy = grad_out_f32[idx];
                            float x_hat = (in_f32[idx] - m) * r;
                            ds += dy * w_f32[c] * x_hat;
                            db += dy * w_f32[c];
                        }
                    }

                    float inv_gs = 1.0f / static_cast<float>(group_size);
                    for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                        for (int64_t s = 0; s < spatial_size; ++s) {
                            int64_t idx = (n * C + c) * spatial_size + s;
                            float dy = grad_out_f32[idx];
                            float x_hat = (in_f32[idx] - m) * r;
                            grad_in_f32[idx] = r * (dy * w_f32[c] - inv_gs * (db + x_hat * ds));
                        }
                    }

                    for (int64_t c = c_start; c < c_start + channels_per_group; ++c) {
                        for (int64_t s = 0; s < spatial_size; ++s) {
                            int64_t idx = (n * C + c) * spatial_size + s;
                            float x_hat = (in_f32[idx] - m) * r;
                            t_gw[c] += grad_out_f32[idx] * x_hat;
                            t_gb[c] += grad_out_f32[idx];
                        }
                    }
                }
            }

            #pragma omp critical
            {
                for (int64_t c = 0; c < C; ++c) {
                    grad_w_f32[c] += t_gw[c];
                    grad_b_f32[c] += t_gb[c];
                }
            }
        }

        // Convert back
        if (input.dtype() == DType::Float16) {
            Float16* gi = grad_input.data<Float16>();
            Float16* gw = grad_weight.data<Float16>();
            Float16* gb = grad_bias.data<Float16>();
            for (int64_t i = 0; i < total; ++i) gi[i] = Float16(grad_in_f32[i]);
            for (int64_t i = 0; i < C; ++i) { gw[i] = Float16(grad_w_f32[i]); gb[i] = Float16(grad_b_f32[i]); }
        } else {
            BFloat16* gi = grad_input.data<BFloat16>();
            BFloat16* gw = grad_weight.data<BFloat16>();
            BFloat16* gb = grad_bias.data<BFloat16>();
            for (int64_t i = 0; i < total; ++i) gi[i] = BFloat16(grad_in_f32[i]);
            for (int64_t i = 0; i < C; ++i) { gw[i] = BFloat16(grad_w_f32[i]); gb[i] = BFloat16(grad_b_f32[i]); }
        }
    }

    return {grad_input, grad_weight, grad_bias};
}

// ============================================================================
// InstanceNorm Backward
// ============================================================================

auto instance_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                    const Tensor& mean, const Tensor& rstd,
                                    const Tensor& weight) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < shape.size(); ++i) {
        spatial_size *= shape[i];
    }

    auto grad_input = Tensor::empty_uninitialized(
        std::vector<int64_t>(shape.begin(), shape.end()),
        input.dtype(), input.device());
    auto grad_weight = zeros({C}, weight.dtype(), weight.device());
    auto grad_bias = zeros({C}, weight.dtype(), weight.device());

    // audit-2026-05-03 Phase 10 — mean/rstd dtype now matches input dtype:
    // Float32 for Float32/Float16/BFloat16, Float64 for Float64. Previously
    // hardcoded float, dropping Float64 precision through the autograd path.
    const float* mean_data_f32 = nullptr;
    const float* rstd_data_f32 = nullptr;
    const double* mean_data_f64 = nullptr;
    const double* rstd_data_f64 = nullptr;
    if (mean.dtype() == DType::Float64) {
        mean_data_f64 = mean.data<double>();
        rstd_data_f64 = rstd.data<double>();
    } else {
        mean_data_f32 = mean.data<float>();
        rstd_data_f32 = rstd.data<float>();
    }
    // Float32 helpers used by the F32 / F16 / BF16 branches below.
    const float* mean_data = mean_data_f32;
    const float* rstd_data = rstd_data_f32;

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        const float* grad_out_data = grad_output.data<float>();
        const float* w_data = weight.impl() ? weight.data<float>() : nullptr;
        float* grad_in_data = grad_input.data<float>();
        float* grad_w_data = grad_weight.data<float>();
        float* grad_b_data = grad_bias.data<float>();

        #pragma omp parallel if(N * C > 16)
        {
            std::vector<float> t_gw(C, 0.0f);
            std::vector<float> t_gb(C, 0.0f);

            #pragma omp for collapse(2)
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    float m = mean_data[n * C + c];
                    float r = rstd_data[n * C + c];
                    float w = w_data ? w_data[c] : 1.0f;

                    float ds = 0.0f;
                    float db = 0.0f;
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        int64_t idx = (n * C + c) * spatial_size + s;
                        float dy = grad_out_data[idx];
                        float x_hat = (in_data[idx] - m) * r;
                        ds += dy * w * x_hat;
                        db += dy * w;
                    }

                    float inv_ss = 1.0f / static_cast<float>(spatial_size);
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        int64_t idx = (n * C + c) * spatial_size + s;
                        float dy = grad_out_data[idx];
                        float x_hat = (in_data[idx] - m) * r;
                        grad_in_data[idx] = r * (dy * w - inv_ss * (db + x_hat * ds));
                    }

                    for (int64_t s = 0; s < spatial_size; ++s) {
                        int64_t idx = (n * C + c) * spatial_size + s;
                        float x_hat = (in_data[idx] - m) * r;
                        t_gw[c] += grad_out_data[idx] * x_hat;
                        t_gb[c] += grad_out_data[idx];
                    }
                }
            }

            #pragma omp critical
            {
                for (int64_t c = 0; c < C; ++c) {
                    grad_w_data[c] += t_gw[c];
                    grad_b_data[c] += t_gb[c];
                }
            }
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        const double* grad_out_data = grad_output.data<double>();
        const double* w_data = weight.impl() ? weight.data<double>() : nullptr;
        double* grad_in_data = grad_input.data<double>();
        double* grad_w_data = grad_weight.data<double>();
        double* grad_b_data = grad_bias.data<double>();

        #pragma omp parallel if(N * C > 16)
        {
            std::vector<double> t_gw(C, 0.0);
            std::vector<double> t_gb(C, 0.0);

            #pragma omp for collapse(2)
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    // Read at the dtype the forward saved the stats in
                    // (Float64 when input is Float64).
                    double m = mean_data_f64
                        ? mean_data_f64[n * C + c]
                        : static_cast<double>(mean_data_f32[n * C + c]);
                    double r = rstd_data_f64
                        ? rstd_data_f64[n * C + c]
                        : static_cast<double>(rstd_data_f32[n * C + c]);
                    double w = w_data ? w_data[c] : 1.0;

                    double ds = 0.0, db = 0.0;
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        int64_t idx = (n * C + c) * spatial_size + s;
                        double dy = grad_out_data[idx];
                        double x_hat = (in_data[idx] - m) * r;
                        ds += dy * w * x_hat;
                        db += dy * w;
                    }

                    double inv_ss = 1.0 / static_cast<double>(spatial_size);
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        int64_t idx = (n * C + c) * spatial_size + s;
                        double dy = grad_out_data[idx];
                        double x_hat = (in_data[idx] - m) * r;
                        grad_in_data[idx] = r * (dy * w - inv_ss * (db + x_hat * ds));
                    }

                    for (int64_t s = 0; s < spatial_size; ++s) {
                        int64_t idx = (n * C + c) * spatial_size + s;
                        double x_hat = (in_data[idx] - m) * r;
                        t_gw[c] += grad_out_data[idx] * x_hat;
                        t_gb[c] += grad_out_data[idx];
                    }
                }
            }

            #pragma omp critical
            {
                for (int64_t c = 0; c < C; ++c) {
                    grad_w_data[c] += t_gw[c];
                    grad_b_data[c] += t_gb[c];
                }
            }
        }
    } else {
        // Float16/BFloat16: compute in Float32
        int64_t total = input.numel();
        std::vector<float> in_f32(total), grad_out_f32(total);
        std::vector<float> w_f32(C, 1.0f);

        if (input.dtype() == DType::Float16) {
            const Float16* ir = input.data<Float16>();
            const Float16* gr = grad_output.data<Float16>();
            for (int64_t i = 0; i < total; ++i) { in_f32[i] = static_cast<float>(ir[i]); grad_out_f32[i] = static_cast<float>(gr[i]); }
            if (weight.impl()) { const Float16* wr = weight.data<Float16>(); for (int64_t i = 0; i < C; ++i) w_f32[i] = static_cast<float>(wr[i]); }
        } else {
            const BFloat16* ir = input.data<BFloat16>();
            const BFloat16* gr = grad_output.data<BFloat16>();
            for (int64_t i = 0; i < total; ++i) { in_f32[i] = static_cast<float>(ir[i]); grad_out_f32[i] = static_cast<float>(gr[i]); }
            if (weight.impl()) { const BFloat16* wr = weight.data<BFloat16>(); for (int64_t i = 0; i < C; ++i) w_f32[i] = static_cast<float>(wr[i]); }
        }

        std::vector<float> grad_in_f32(total, 0.0f);
        std::vector<float> grad_w_f32(C, 0.0f);
        std::vector<float> grad_b_f32(C, 0.0f);

        #pragma omp parallel if(N * C > 16)
        {
            std::vector<float> t_gw(C, 0.0f);
            std::vector<float> t_gb(C, 0.0f);

            #pragma omp for collapse(2)
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C; ++c) {
                    float m = mean_data[n * C + c];
                    float r = rstd_data[n * C + c];
                    float w = w_f32[c];

                    float ds = 0.0f, db = 0.0f;
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        int64_t idx = (n * C + c) * spatial_size + s;
                        float dy = grad_out_f32[idx];
                        float x_hat = (in_f32[idx] - m) * r;
                        ds += dy * w * x_hat;
                        db += dy * w;
                    }

                    float inv_ss = 1.0f / static_cast<float>(spatial_size);
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        int64_t idx = (n * C + c) * spatial_size + s;
                        float dy = grad_out_f32[idx];
                        float x_hat = (in_f32[idx] - m) * r;
                        grad_in_f32[idx] = r * (dy * w - inv_ss * (db + x_hat * ds));
                    }

                    for (int64_t s = 0; s < spatial_size; ++s) {
                        int64_t idx = (n * C + c) * spatial_size + s;
                        float x_hat = (in_f32[idx] - m) * r;
                        t_gw[c] += grad_out_f32[idx] * x_hat;
                        t_gb[c] += grad_out_f32[idx];
                    }
                }
            }

            #pragma omp critical
            {
                for (int64_t c = 0; c < C; ++c) {
                    grad_w_f32[c] += t_gw[c];
                    grad_b_f32[c] += t_gb[c];
                }
            }
        }

        if (input.dtype() == DType::Float16) {
            Float16* gi = grad_input.data<Float16>();
            Float16* gw = grad_weight.data<Float16>();
            Float16* gb = grad_bias.data<Float16>();
            for (int64_t i = 0; i < total; ++i) gi[i] = Float16(grad_in_f32[i]);
            for (int64_t i = 0; i < C; ++i) { gw[i] = Float16(grad_w_f32[i]); gb[i] = Float16(grad_b_f32[i]); }
        } else {
            BFloat16* gi = grad_input.data<BFloat16>();
            BFloat16* gw = grad_weight.data<BFloat16>();
            BFloat16* gb = grad_bias.data<BFloat16>();
            for (int64_t i = 0; i < total; ++i) gi[i] = BFloat16(grad_in_f32[i]);
            for (int64_t i = 0; i < C; ++i) { gw[i] = BFloat16(grad_w_f32[i]); gb[i] = BFloat16(grad_b_f32[i]); }
        }
    }

    return {grad_input, grad_weight, grad_bias};
}

} // namespace cpu
} // namespace tenzor
