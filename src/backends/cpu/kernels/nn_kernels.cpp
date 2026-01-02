/**
 * @file nn_kernels.cpp
 * @brief CPU neural network kernel implementations (linear, dropout, embedding, etc.)
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <random>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

// Intel oneDNN for optimized layer operations
#ifdef TENZOR_USE_ONEDNN
#include <dnnl.hpp>
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

#ifdef TENZOR_USE_ONEDNN
// Thread-local oneDNN engine and stream for reuse
static thread_local dnnl::engine g_nn_engine(dnnl::engine::kind::cpu, 0);
static thread_local dnnl::stream g_nn_stream(g_nn_engine);
#endif

#ifdef TENZOR_USE_ONEDNN
// oneDNN-accelerated Linear (inner product) - provides 10-50x speedup
static bool linear_onednn(
    const float* input, const float* weight, const float* bias,
    float* output,
    int64_t batch_size, int64_t in_features, int64_t out_features
) {
    try {
        auto& engine = g_nn_engine;
        auto& stream = g_nn_stream;

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

auto linear_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias) -> Tensor {
    // input: [*, in_features] or [batch, in_features]
    // weight: [out_features, in_features]
    // output: [*, out_features]

    auto in_shape = input.shape();
    auto w_shape = weight.shape();
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

    auto output = Tensor::empty_uninitialized(out_shape, input.dtype(), input.device());

    const float* in_data = input.data<float>();
    const float* w_data = weight.data<float>();
    float* out_data = output.data<float>();
    const float* b_data = bias ? bias->data<float>() : nullptr;

#ifdef TENZOR_USE_ONEDNN
    // Try oneDNN first for maximum performance
    if (input.dtype() == DType::Float32 &&
        linear_onednn(in_data, w_data, b_data, out_data, batch_size, in_features, out_features)) {
        return output;
    }
#endif

    // Fallback: Y = X @ W^T with OpenMP parallelization
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

    return output;
}

auto linear_backward_kernel(const Tensor& grad_output, const Tensor& input,
                             const Tensor& weight) -> std::vector<Tensor> {
    auto grad_shape = grad_output.shape();
    auto in_shape = input.shape();
    auto w_shape = weight.shape();

    int64_t out_features = w_shape[0];
    int64_t in_features = w_shape[1];

    int64_t batch_size = 1;
    for (size_t i = 0; i < in_shape.size() - 1; ++i) {
        batch_size *= in_shape[i];
    }

    // grad_input = grad_output @ weight
    auto grad_input = Tensor::empty_uninitialized(
        std::vector<int64_t>(in_shape.begin(), in_shape.end()),
        input.dtype(), input.device());

    const float* grad_out_data = grad_output.data<float>();
    const float* w_data = weight.data<float>();
    float* grad_in_data = grad_input.data<float>();

    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < in_features; ++i) {
            float sum = 0.0f;
            for (int64_t o = 0; o < out_features; ++o) {
                sum += grad_out_data[b * out_features + o] * w_data[o * in_features + i];
            }
            grad_in_data[b * in_features + i] = sum;
        }
    }

    // grad_weight = grad_output^T @ input
    auto grad_weight = zeros(
        std::vector<int64_t>(w_shape.begin(), w_shape.end()),
        weight.dtype(), weight.device());

    const float* in_data = input.data<float>();
    float* grad_w_data = grad_weight.data<float>();

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t o = 0; o < out_features; ++o) {
            for (int64_t i = 0; i < in_features; ++i) {
                grad_w_data[o * in_features + i] +=
                    grad_out_data[b * out_features + o] * in_data[b * in_features + i];
            }
        }
    }

    // grad_bias = sum(grad_output, dim=0)
    auto grad_bias = zeros({out_features}, grad_output.dtype(), grad_output.device());
    float* grad_b_data = grad_bias.data<float>();

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t o = 0; o < out_features; ++o) {
            grad_b_data[o] += grad_out_data[b * out_features + o];
        }
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

    #pragma omp parallel for
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

    #pragma omp parallel for
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
// oneDNN-accelerated LayerNorm - provides 10-50x speedup
static bool layer_norm_onednn(
    const float* input, float* output,
    const float* weight, const float* bias,
    int64_t batch_size, int64_t norm_size, float eps
) {
    try {
        auto& engine = g_nn_engine;
        auto& stream = g_nn_stream;

        // Create memory descriptors for 2D layout [batch, norm_size]
        dnnl::memory::dims src_dims = {batch_size, norm_size};
        auto src_md = dnnl::memory::desc(src_dims, dnnl::memory::data_type::f32,
                                          dnnl::memory::format_tag::nc);
        auto dst_md = dnnl::memory::desc(src_dims, dnnl::memory::data_type::f32,
                                          dnnl::memory::format_tag::nc);

        // Weight and bias are 1D [norm_size]
        dnnl::memory::dims stat_dims = {norm_size};
        auto stat_md = dnnl::memory::desc(stat_dims, dnnl::memory::data_type::f32,
                                           dnnl::memory::format_tag::a);

        // Create layer normalization primitive descriptor
        auto lnorm_pd = dnnl::layer_normalization_forward::primitive_desc(
            engine,
            dnnl::prop_kind::forward_inference,
            src_md, dst_md, eps,
            dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift
        );

        // Create memory objects
        auto src_mem = dnnl::memory(src_md, engine, const_cast<float*>(input));
        auto dst_mem = dnnl::memory(dst_md, engine, output);
        auto scale_mem = dnnl::memory(stat_md, engine, const_cast<float*>(weight));
        auto shift_mem = dnnl::memory(stat_md, engine, const_cast<float*>(bias));

        // Create and execute primitive
        auto lnorm_prim = dnnl::layer_normalization_forward(lnorm_pd);
        lnorm_prim.execute(stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DST, dst_mem},
            {DNNL_ARG_SCALE, scale_mem},
            {DNNL_ARG_SHIFT, shift_mem}
        });
        stream.wait();
        return true;
    } catch (...) {
        return false;
    }
}
#endif

// SIMD-optimized LayerNorm for when oneDNN is not available or fails
static void layer_norm_simd(
    const float* input, float* output,
    const float* weight, const float* bias,
    int64_t batch_size, int64_t norm_size, float eps
) {
    #pragma omp parallel for
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
    auto in_shape = input.shape();
    int64_t norm_size = 1;
    for (auto s : normalized_shape) {
        norm_size *= s;
    }
    int64_t batch_size = input.numel() / norm_size;

    auto output = Tensor::empty_uninitialized(
        std::vector<int64_t>(in_shape.begin(), in_shape.end()),
        input.dtype(), input.device());

    const float* in_data = input.data<float>();
    const float* w_data = weight.data<float>();
    const float* b_data = bias.data<float>();
    float* out_data = output.data<float>();

#ifdef TENZOR_USE_ONEDNN
    // Try oneDNN first for maximum performance
    if (input.dtype() == DType::Float32 &&
        layer_norm_onednn(in_data, out_data, w_data, b_data, batch_size, norm_size, eps)) {
        return output;
    }
#endif

    // Fall back to SIMD-optimized implementation
    layer_norm_simd(in_data, out_data, w_data, b_data, batch_size, norm_size, eps);

    return output;
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

    #pragma omp parallel for collapse(2)
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

    #pragma omp parallel for collapse(2)
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
