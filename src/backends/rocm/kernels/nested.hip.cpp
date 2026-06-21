/**
 * @file nested.hip.cpp
 * @brief ROCm/HIP kernels for nested (ragged) tensor operations.
 *
 * Mirrors the CUDA nested kernels using HIP API. One block per batch
 * element; threads cooperate on the inner dimension D.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace tenzor {
namespace rocm {

#ifndef HIP_CHECK_NESTED
#define HIP_CHECK_NESTED(call)                                                 \
    do {                                                                        \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            throw std::runtime_error(                                          \
                std::string("HIP error in nested at ") + __FILE__ + ":" +     \
                std::to_string(__LINE__) + " - " + hipGetErrorString(err));    \
        }                                                                      \
    } while (0)
#endif

// ============================================================================
// Segmented Softmax
// ============================================================================

template <typename T>
__global__ void nested_softmax_kernel(
    const T* __restrict__ values,
    T* __restrict__ output,
    const int64_t* __restrict__ offsets,
    int64_t D, int64_t B)
{
    int64_t b = blockIdx.x;
    if (b >= B) return;

    int64_t start = offsets[b];
    int64_t end = offsets[b + 1];
    int64_t len = end - start;
    if (len <= 0) return;

    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        T max_val = values[start * D + d];
        for (int64_t s = 1; s < len; ++s) {
            T v = values[(start + s) * D + d];
            if (v > max_val) max_val = v;
        }

        T sum = T(0);
        for (int64_t s = 0; s < len; ++s) {
            T v = exp(values[(start + s) * D + d] - max_val);
            output[(start + s) * D + d] = v;
            sum += v;
        }

        T inv_sum = T(1) / sum;
        for (int64_t s = 0; s < len; ++s) {
            output[(start + s) * D + d] *= inv_sum;
        }
    }
}

auto nested_softmax_hip(const Tensor& values_in, const Tensor& offsets_in,
                         int64_t dim, hipStream_t stream) -> Tensor {
    if (values_in.dtype() == DType::Float16 || values_in.dtype() == DType::BFloat16) {
        auto out32 = nested_softmax_hip(values_in.to(DType::Float32), offsets_in, dim, stream);
        return out32.to(values_in.dtype());
    }
    // Kernels index values/offsets with dense [row*D+d] offsets, so a
    // non-contiguous ragged buffer would be misread. Materialize contiguous.
    Tensor values  = values_in.is_contiguous()  ? values_in  : values_in.contiguous();
    Tensor offsets = offsets_in.is_contiguous() ? offsets_in : offsets_in.contiguous();
    auto shape = values.shape();
    int64_t D = (shape.size() > 1) ? shape[1] : 1;
    int64_t B = offsets.numel() - 1;

    auto output = tenzor::empty(std::vector<int64_t>(shape.begin(), shape.end()), values.dtype(), values.device());
    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (values.dtype() == DType::Float32) {
        hipLaunchKernelGGL(nested_softmax_kernel<float>,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            values.data<float>(), output.data<float>(),
            offsets.data<int64_t>(), D, B);
    } else if (values.dtype() == DType::Float64) {
        hipLaunchKernelGGL(nested_softmax_kernel<double>,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            values.data<double>(), output.data<double>(),
            offsets.data<int64_t>(), D, B);
    } else {
        throw std::runtime_error("nested_softmax_hip: unsupported dtype");
    }

    HIP_CHECK_NESTED(hipGetLastError());
    return output;
}

// ============================================================================
// Segmented Log-Softmax
// ============================================================================

template <typename T>
__global__ void nested_log_softmax_kernel(
    const T* __restrict__ values,
    T* __restrict__ output,
    const int64_t* __restrict__ offsets,
    int64_t D, int64_t B)
{
    int64_t b = blockIdx.x;
    if (b >= B) return;

    int64_t start = offsets[b];
    int64_t end = offsets[b + 1];
    int64_t len = end - start;
    if (len <= 0) return;

    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        T max_val = values[start * D + d];
        for (int64_t s = 1; s < len; ++s) {
            T v = values[(start + s) * D + d];
            if (v > max_val) max_val = v;
        }

        T sum = T(0);
        for (int64_t s = 0; s < len; ++s) {
            sum += exp(values[(start + s) * D + d] - max_val);
        }

        T log_sum = log(sum);
        for (int64_t s = 0; s < len; ++s) {
            output[(start + s) * D + d] = (values[(start + s) * D + d] - max_val) - log_sum;
        }
    }
}

auto nested_log_softmax_hip(const Tensor& values_in, const Tensor& offsets_in,
                             int64_t dim, hipStream_t stream) -> Tensor {
    if (values_in.dtype() == DType::Float16 || values_in.dtype() == DType::BFloat16) {
        auto out32 = nested_log_softmax_hip(values_in.to(DType::Float32), offsets_in, dim, stream);
        return out32.to(values_in.dtype());
    }
    Tensor values  = values_in.is_contiguous()  ? values_in  : values_in.contiguous();
    Tensor offsets = offsets_in.is_contiguous() ? offsets_in : offsets_in.contiguous();
    auto shape = values.shape();
    int64_t D = (shape.size() > 1) ? shape[1] : 1;
    int64_t B = offsets.numel() - 1;

    auto output = tenzor::empty(std::vector<int64_t>(shape.begin(), shape.end()), values.dtype(), values.device());
    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (values.dtype() == DType::Float32) {
        hipLaunchKernelGGL(nested_log_softmax_kernel<float>,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            values.data<float>(), output.data<float>(),
            offsets.data<int64_t>(), D, B);
    } else if (values.dtype() == DType::Float64) {
        hipLaunchKernelGGL(nested_log_softmax_kernel<double>,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            values.data<double>(), output.data<double>(),
            offsets.data<int64_t>(), D, B);
    } else {
        throw std::runtime_error("nested_log_softmax_hip: unsupported dtype");
    }

    HIP_CHECK_NESTED(hipGetLastError());
    return output;
}

// ============================================================================
// Segmented Sum Reduction
// ============================================================================

template <typename T>
__global__ void nested_sum_kernel(
    const T* __restrict__ values,
    T* __restrict__ output,
    const int64_t* __restrict__ offsets,
    int64_t D, int64_t B)
{
    int64_t b = blockIdx.x;
    if (b >= B) return;

    int64_t start = offsets[b];
    int64_t end = offsets[b + 1];

    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        T sum = T(0);
        for (int64_t s = start; s < end; ++s) {
            sum += values[s * D + d];
        }
        output[b * D + d] = sum;
    }
}

auto nested_sum_hip(const Tensor& values_in, const Tensor& offsets_in,
                     hipStream_t stream) -> Tensor {
    if (values_in.dtype() == DType::Float16 || values_in.dtype() == DType::BFloat16) {
        auto out32 = nested_sum_hip(values_in.to(DType::Float32), offsets_in, stream);
        return out32.to(values_in.dtype());
    }
    Tensor values  = values_in.is_contiguous()  ? values_in  : values_in.contiguous();
    Tensor offsets = offsets_in.is_contiguous() ? offsets_in : offsets_in.contiguous();
    auto shape = values.shape();
    int64_t D = (shape.size() > 1) ? shape[1] : 1;
    int64_t B = offsets.numel() - 1;

    auto output = tenzor::zeros({B, D}, values.dtype(), values.device());
    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (values.dtype() == DType::Float32) {
        hipLaunchKernelGGL(nested_sum_kernel<float>,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            values.data<float>(), output.data<float>(),
            offsets.data<int64_t>(), D, B);
    } else if (values.dtype() == DType::Float64) {
        hipLaunchKernelGGL(nested_sum_kernel<double>,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            values.data<double>(), output.data<double>(),
            offsets.data<int64_t>(), D, B);
    } else {
        throw std::runtime_error("nested_sum_hip: unsupported dtype");
    }

    HIP_CHECK_NESTED(hipGetLastError());
    return output;
}

// ============================================================================
// Segmented Mean Reduction
// ============================================================================

template <typename T>
__global__ void nested_mean_kernel(
    const T* __restrict__ values,
    T* __restrict__ output,
    const int64_t* __restrict__ offsets,
    int64_t D, int64_t B)
{
    int64_t b = blockIdx.x;
    if (b >= B) return;

    int64_t start = offsets[b];
    int64_t end = offsets[b + 1];
    int64_t len = end - start;
    if (len <= 0) return;

    T inv_len = T(1) / static_cast<T>(len);

    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        T sum = T(0);
        for (int64_t s = start; s < end; ++s) {
            sum += values[s * D + d];
        }
        output[b * D + d] = sum * inv_len;
    }
}

auto nested_mean_hip(const Tensor& values_in, const Tensor& offsets_in,
                      hipStream_t stream) -> Tensor {
    if (values_in.dtype() == DType::Float16 || values_in.dtype() == DType::BFloat16) {
        auto out32 = nested_mean_hip(values_in.to(DType::Float32), offsets_in, stream);
        return out32.to(values_in.dtype());
    }
    Tensor values  = values_in.is_contiguous()  ? values_in  : values_in.contiguous();
    Tensor offsets = offsets_in.is_contiguous() ? offsets_in : offsets_in.contiguous();
    auto shape = values.shape();
    int64_t D = (shape.size() > 1) ? shape[1] : 1;
    int64_t B = offsets.numel() - 1;

    auto output = tenzor::zeros({B, D}, values.dtype(), values.device());
    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (values.dtype() == DType::Float32) {
        hipLaunchKernelGGL(nested_mean_kernel<float>,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            values.data<float>(), output.data<float>(),
            offsets.data<int64_t>(), D, B);
    } else if (values.dtype() == DType::Float64) {
        hipLaunchKernelGGL(nested_mean_kernel<double>,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            values.data<double>(), output.data<double>(),
            offsets.data<int64_t>(), D, B);
    } else {
        throw std::runtime_error("nested_mean_hip: unsupported dtype");
    }

    HIP_CHECK_NESTED(hipGetLastError());
    return output;
}

// ============================================================================
// Nested Attention (variable-length scaled dot-product attention)
// ============================================================================

template <typename T>
__global__ void nested_attention_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ output,
    const int64_t* __restrict__ q_offsets,
    const int64_t* __restrict__ kv_offsets,
    T scale, int64_t head_dim, int64_t B, bool causal)
{
    int64_t b = blockIdx.x;
    if (b >= B) return;

    int64_t q_start = q_offsets[b], q_end = q_offsets[b + 1];
    int64_t kv_start = kv_offsets[b], kv_end = kv_offsets[b + 1];
    int64_t Lq = q_end - q_start;
    int64_t Lkv = kv_end - kv_start;
    if (Lq <= 0 || Lkv <= 0) return;

    for (int64_t qi = threadIdx.x; qi < Lq; qi += blockDim.x) {
        const T* q_row = Q + (q_start + qi) * head_dim;
        T* out_row = output + (q_start + qi) * head_dim;

        T max_score = -FLT_MAX;
        T sum_exp = T(0);

        for (int64_t hd = 0; hd < head_dim; ++hd) {
            out_row[hd] = T(0);
        }

        for (int64_t ki = 0; ki < Lkv; ++ki) {
            if (causal && ki > qi) break;

            const T* k_row = K + (kv_start + ki) * head_dim;
            const T* v_row = V + (kv_start + ki) * head_dim;

            T score = T(0);
            for (int64_t hd = 0; hd < head_dim; ++hd) {
                score += q_row[hd] * k_row[hd];
            }
            score *= scale;

            if (score > max_score) {
                T correction = exp(max_score - score);
                sum_exp = sum_exp * correction + T(1);
                for (int64_t hd = 0; hd < head_dim; ++hd) {
                    out_row[hd] = out_row[hd] * correction + v_row[hd];
                }
                max_score = score;
            } else {
                T w = exp(score - max_score);
                sum_exp += w;
                for (int64_t hd = 0; hd < head_dim; ++hd) {
                    out_row[hd] += w * v_row[hd];
                }
            }
        }

        if (sum_exp > T(0)) {
            T inv_sum = T(1) / sum_exp;
            for (int64_t hd = 0; hd < head_dim; ++hd) {
                out_row[hd] *= inv_sum;
            }
        }
    }
}

auto nested_attention_hip(const Tensor& Q_in, const Tensor& K_in, const Tensor& V_in,
                           const Tensor& q_offsets_in, const Tensor& kv_offsets_in,
                           float scale, bool causal, hipStream_t stream) -> Tensor {
    if (Q_in.dtype() == DType::Float16 || Q_in.dtype() == DType::BFloat16) {
        auto out32 = nested_attention_hip(Q_in.to(DType::Float32), K_in.to(DType::Float32),
                                          V_in.to(DType::Float32), q_offsets_in, kv_offsets_in,
                                          scale, causal, stream);
        return out32.to(Q_in.dtype());
    }
    // Dense [(start+s)*D+d] indexing requires contiguous Q/K/V/offsets.
    auto mc = [](const Tensor& t) { return t.is_contiguous() ? t : t.contiguous(); };
    Tensor Q = mc(Q_in), K = mc(K_in), V = mc(V_in);
    Tensor q_offsets = mc(q_offsets_in), kv_offsets = mc(kv_offsets_in);
    int64_t head_dim = Q.shape().back();
    int64_t total_q_len = Q.shape()[0];
    int64_t B = q_offsets.numel() - 1;

    auto output = tenzor::zeros({total_q_len, head_dim}, Q.dtype(), Q.device());
    int threads = static_cast<int>(std::min(int64_t(256), total_q_len));

    if (Q.dtype() == DType::Float32) {
        hipLaunchKernelGGL(nested_attention_kernel<float>,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            Q.data<float>(), K.data<float>(), V.data<float>(),
            output.data<float>(),
            q_offsets.data<int64_t>(), kv_offsets.data<int64_t>(),
            scale, head_dim, B, causal);
    } else if (Q.dtype() == DType::Float64) {
        hipLaunchKernelGGL(nested_attention_kernel<double>,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            Q.data<double>(), K.data<double>(), V.data<double>(),
            output.data<double>(),
            q_offsets.data<int64_t>(), kv_offsets.data<int64_t>(),
            static_cast<double>(scale), head_dim, B, causal);
    } else {
        throw std::runtime_error("nested_attention_hip: unsupported dtype");
    }

    HIP_CHECK_NESTED(hipGetLastError());
    return output;
}

// ============================================================================
// Nested Attention Backward
// ============================================================================

template <typename T>
__global__ void nested_attention_backward_kernel(
    const T* __restrict__ grad_out,
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ grad_Q,
    T* __restrict__ grad_K,
    T* __restrict__ grad_V,
    const int64_t* __restrict__ q_offsets,
    const int64_t* __restrict__ kv_offsets,
    T scale,
    int64_t head_dim,
    int64_t B,
    bool causal)
{
    int64_t b = blockIdx.x;
    if (b >= B) return;

    int64_t q_start = q_offsets[b], q_end = q_offsets[b + 1];
    int64_t kv_start = kv_offsets[b], kv_end = kv_offsets[b + 1];
    int64_t Lq = q_end - q_start;
    int64_t Lkv = kv_end - kv_start;
    if (Lq <= 0 || Lkv <= 0) return;

    for (int64_t qi = threadIdx.x; qi < Lq; qi += blockDim.x) {
        const T* q_row = Q + (q_start + qi) * head_dim;
        const T* do_row = grad_out + (q_start + qi) * head_dim;
        T* gq_row = grad_Q + (q_start + qi) * head_dim;

        int64_t ki_end = causal ? min(Lkv, qi + 1) : Lkv;

        // Find max score
        T max_score = -FLT_MAX;
        for (int64_t ki = 0; ki < ki_end; ++ki) {
            const T* k_row = K + (kv_start + ki) * head_dim;
            T score = T(0);
            for (int64_t d = 0; d < head_dim; ++d) score += q_row[d] * k_row[d];
            score *= scale;
            if (score > max_score) max_score = score;
        }

        // Compute sum_exp
        T sum_exp = T(0);
        for (int64_t ki = 0; ki < ki_end; ++ki) {
            const T* k_row = K + (kv_start + ki) * head_dim;
            T score = T(0);
            for (int64_t d = 0; d < head_dim; ++d) score += q_row[d] * k_row[d];
            sum_exp += exp(score * scale - max_score);
        }
        T inv_sum = (sum_exp > T(0)) ? T(1) / sum_exp : T(0);

        // Compute softmax_dot
        T softmax_dot = T(0);
        for (int64_t ki = 0; ki < ki_end; ++ki) {
            const T* k_row = K + (kv_start + ki) * head_dim;
            const T* v_row = V + (kv_start + ki) * head_dim;
            T score = T(0);
            for (int64_t d = 0; d < head_dim; ++d) score += q_row[d] * k_row[d];
            T w = exp(score * scale - max_score) * inv_sum;
            T dot_dov = T(0);
            for (int64_t d = 0; d < head_dim; ++d) dot_dov += do_row[d] * v_row[d];
            softmax_dot += w * dot_dov;
        }

        // Compute gradients
        for (int64_t ki = 0; ki < ki_end; ++ki) {
            const T* k_row = K + (kv_start + ki) * head_dim;
            const T* v_row = V + (kv_start + ki) * head_dim;
            T* gk_row = grad_K + (kv_start + ki) * head_dim;
            T* gv_row = grad_V + (kv_start + ki) * head_dim;

            T score = T(0);
            for (int64_t d = 0; d < head_dim; ++d) score += q_row[d] * k_row[d];
            T w = exp(score * scale - max_score) * inv_sum;

            for (int64_t d = 0; d < head_dim; ++d) atomicAdd(&gv_row[d], w * do_row[d]);

            T dot_dov = T(0);
            for (int64_t d = 0; d < head_dim; ++d) dot_dov += do_row[d] * v_row[d];
            T ds = w * (dot_dov - softmax_dot) * scale;

            for (int64_t d = 0; d < head_dim; ++d) gq_row[d] += ds * k_row[d];
            for (int64_t d = 0; d < head_dim; ++d) atomicAdd(&gk_row[d], ds * q_row[d]);
        }
    }
}

auto nested_attention_backward_hip(const Tensor& grad_out, const Tensor& Q,
                                    const Tensor& K, const Tensor& V,
                                    const Tensor& attn_out,
                                    const Tensor& q_offsets, const Tensor& kv_offsets,
                                    float scale, bool causal, hipStream_t stream)
    -> std::vector<Tensor> {
    if (Q.dtype() == DType::Float16 || Q.dtype() == DType::BFloat16) {
        auto g = nested_attention_backward_hip(
            grad_out.to(DType::Float32), Q.to(DType::Float32), K.to(DType::Float32),
            V.to(DType::Float32), attn_out.to(DType::Float32), q_offsets, kv_offsets,
            scale, causal, stream);
        return {g[0].to(Q.dtype()), g[1].to(K.dtype()), g[2].to(V.dtype())};
    }
    int64_t head_dim = Q.shape().back();
    int64_t total_q_len = Q.shape()[0];
    int64_t total_kv_len = K.shape()[0];
    int64_t B = q_offsets.numel() - 1;

    auto grad_Q = tenzor::zeros({total_q_len, head_dim}, Q.dtype(), Q.device());
    auto grad_K = tenzor::zeros({total_kv_len, head_dim}, K.dtype(), K.device());
    auto grad_V = tenzor::zeros({total_kv_len, head_dim}, V.dtype(), V.device());

    int threads = static_cast<int>(std::min(int64_t(256), total_q_len));

    if (Q.dtype() == DType::Float32) {
        hipLaunchKernelGGL(nested_attention_backward_kernel<float>,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            grad_out.data<float>(), Q.data<float>(), K.data<float>(), V.data<float>(),
            grad_Q.data<float>(), grad_K.data<float>(), grad_V.data<float>(),
            q_offsets.data<int64_t>(), kv_offsets.data<int64_t>(),
            scale, head_dim, B, causal);
    } else if (Q.dtype() == DType::Float64) {
        hipLaunchKernelGGL(nested_attention_backward_kernel<double>,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            grad_out.data<double>(), Q.data<double>(), K.data<double>(), V.data<double>(),
            grad_Q.data<double>(), grad_K.data<double>(), grad_V.data<double>(),
            q_offsets.data<int64_t>(), kv_offsets.data<int64_t>(),
            static_cast<double>(scale), head_dim, B, causal);
    } else {
        throw std::runtime_error("nested_attention_backward_hip: unsupported dtype");
    }

    HIP_CHECK_NESTED(hipGetLastError());
    return {grad_Q, grad_K, grad_V};
}

// ============================================================================
// Nested to Padded
// ============================================================================

template <typename T>
__global__ void nested_to_padded_kernel(
    const T* __restrict__ values,
    T* __restrict__ padded,
    const int64_t* __restrict__ offsets,
    int64_t max_len, int64_t D, int64_t B, T padding_value)
{
    int64_t b = blockIdx.x;
    if (b >= B) return;

    int64_t start = offsets[b];
    int64_t end = offsets[b + 1];
    int64_t len = end - start;

    for (int64_t pos = 0; pos < max_len; ++pos) {
        for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
            int64_t out_idx = (b * max_len + pos) * D + d;
            padded[out_idx] = (pos < len) ? values[(start + pos) * D + d] : padding_value;
        }
    }
}

auto nested_to_padded_hip(const Tensor& values_in, const Tensor& offsets_in,
                           int64_t max_len, float padding_value,
                           hipStream_t stream) -> Tensor {
    if (values_in.dtype() == DType::Float16 || values_in.dtype() == DType::BFloat16) {
        auto out32 = nested_to_padded_hip(values_in.to(DType::Float32), offsets_in, max_len, padding_value, stream);
        return out32.to(values_in.dtype());
    }
    Tensor values  = values_in.is_contiguous()  ? values_in  : values_in.contiguous();
    Tensor offsets = offsets_in.is_contiguous() ? offsets_in : offsets_in.contiguous();
    auto shape = values.shape();
    int64_t D = (shape.size() > 1) ? shape[1] : 1;
    int64_t B = offsets.numel() - 1;

    auto padded = tenzor::full({B, max_len, D}, padding_value, values.dtype(), values.device());
    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (values.dtype() == DType::Float32) {
        hipLaunchKernelGGL(nested_to_padded_kernel<float>,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            values.data<float>(), padded.data<float>(),
            offsets.data<int64_t>(), max_len, D, B, padding_value);
    } else if (values.dtype() == DType::Float64) {
        hipLaunchKernelGGL(nested_to_padded_kernel<double>,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            values.data<double>(), padded.data<double>(),
            offsets.data<int64_t>(), max_len, D, B, static_cast<double>(padding_value));
    } else {
        throw std::runtime_error("nested_to_padded_hip: unsupported dtype");
    }

    HIP_CHECK_NESTED(hipGetLastError());
    return padded;
}

// ============================================================================
// Nested from Padded
// ============================================================================

template <typename T>
__global__ void nested_from_padded_kernel(
    const T* __restrict__ padded,
    T* __restrict__ values,
    const int64_t* __restrict__ offsets,
    int64_t max_len, int64_t D, int64_t B)
{
    int64_t b = blockIdx.x;
    if (b >= B) return;

    int64_t start = offsets[b];
    int64_t end = offsets[b + 1];
    int64_t len = end - start;

    for (int64_t pos = 0; pos < len; ++pos) {
        for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
            values[(start + pos) * D + d] = padded[(b * max_len + pos) * D + d];
        }
    }
}

auto nested_from_padded_hip(const Tensor& padded_in, const Tensor& offsets_in,
                              hipStream_t stream) -> Tensor {
    if (padded_in.dtype() == DType::Float16 || padded_in.dtype() == DType::BFloat16) {
        auto out32 = nested_from_padded_hip(padded_in.to(DType::Float32), offsets_in, stream);
        return out32.to(padded_in.dtype());
    }
    Tensor padded  = padded_in.is_contiguous()  ? padded_in  : padded_in.contiguous();
    Tensor offsets = offsets_in.is_contiguous() ? offsets_in : offsets_in.contiguous();
    auto pad_shape = padded.shape();
    int64_t B = pad_shape[0];
    int64_t max_len = pad_shape[1];
    int64_t D = (pad_shape.size() > 2) ? pad_shape[2] : 1;

    auto offsets_cpu = (offsets.device().type == Device::Type::CPU) ? offsets : offsets.to(Device::cpu());
    int64_t total_len = offsets_cpu.data<int64_t>()[B];

    auto values = tenzor::empty({total_len, D}, padded.dtype(), padded.device());
    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (padded.dtype() == DType::Float32) {
        hipLaunchKernelGGL(nested_from_padded_kernel<float>,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            padded.data<float>(), values.data<float>(),
            offsets.data<int64_t>(), max_len, D, B);
    } else if (padded.dtype() == DType::Float64) {
        hipLaunchKernelGGL(nested_from_padded_kernel<double>,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            padded.data<double>(), values.data<double>(),
            offsets.data<int64_t>(), max_len, D, B);
    } else {
        throw std::runtime_error("nested_from_padded_hip: unsupported dtype");
    }

    HIP_CHECK_NESTED(hipGetLastError());
    return values;
}

// ============================================================================
// Nested Layer Norm
// ============================================================================

template <typename T>
__global__ void nested_layer_norm_kernel(
    const T* __restrict__ values,
    T* __restrict__ output,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const int64_t* __restrict__ offsets,
    int64_t D, int64_t B, T eps)
{
    int64_t b = blockIdx.x;
    if (b >= B) return;

    int64_t start = offsets[b];
    int64_t end = offsets[b + 1];
    if (start >= end) return;

    // Block is launched with at most 256 threads (min(D, 256)). Each thread
    // computes the row's mean and variance independently over the full O(D)
    // range (no cross-thread reduction), then threads cooperatively write the
    // normalized output. This mirrors the CUDA reference
    // (nested_layer_norm_kernel_t) and is always correct regardless of whether
    // blockDim.x is a power of two. A shared-memory tree reduction was avoided
    // here because the halving recurrence double-counts elements for non-pow2
    // block sizes (e.g. D = 96, 100, 192, 200, 250), silently corrupting both
    // mean and variance. The two-pass formulation (sum, then sum of squared
    // diffs about the mean) matches CPU/other backends.
    for (int64_t row = start; row < end; ++row) {
        // Pass 1: mean (T-precision accumulator, full-range per thread).
        T mean = T(0);
        for (int64_t d = 0; d < D; ++d) {
            mean += values[row * D + d];
        }
        mean /= static_cast<T>(D);

        // Pass 2: variance (sum of squared deviations, full-range per thread).
        T var = T(0);
        for (int64_t d = 0; d < D; ++d) {
            T diff = values[row * D + d] - mean;
            var += diff * diff;
        }
        var /= static_cast<T>(D);

        T inv_std = T(1) / sqrt(var + eps);

        // Normalize and apply affine cooperatively across threads.
        for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
            T normalized = (values[row * D + d] - mean) * inv_std;
            output[row * D + d] = normalized * weight[d] + bias[d];
        }
    }
}

auto nested_layer_norm_hip(const Tensor& values_in, const Tensor& offsets_in,
                            const Tensor& weight_in, const Tensor& bias_in,
                            float eps, hipStream_t stream) -> Tensor {
    if (values_in.dtype() == DType::Float16 || values_in.dtype() == DType::BFloat16) {
        auto out32 = nested_layer_norm_hip(values_in.to(DType::Float32), offsets_in,
                                           weight_in.to(DType::Float32), bias_in.to(DType::Float32),
                                           eps, stream);
        return out32.to(values_in.dtype());
    }
    // Dense [row*D+d] indexing requires contiguous values/offsets/weight/bias.
    auto mc = [](const Tensor& t) { return t.is_contiguous() ? t : t.contiguous(); };
    Tensor values = mc(values_in), offsets = mc(offsets_in);
    Tensor weight = mc(weight_in), bias = mc(bias_in);
    auto shape = values.shape();
    int64_t D = (shape.size() > 1) ? shape[1] : 1;
    int64_t B = offsets.numel() - 1;

    auto output = tenzor::empty(std::vector<int64_t>(shape.begin(), shape.end()), values.dtype(), values.device());
    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (values.dtype() == DType::Float32) {
        hipLaunchKernelGGL(nested_layer_norm_kernel<float>,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            values.data<float>(), output.data<float>(),
            weight.data<float>(), bias.data<float>(),
            offsets.data<int64_t>(), D, B, eps);
    } else if (values.dtype() == DType::Float64) {
        hipLaunchKernelGGL(nested_layer_norm_kernel<double>,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            values.data<double>(), output.data<double>(),
            weight.data<double>(), bias.data<double>(),
            offsets.data<int64_t>(), D, B, static_cast<double>(eps));
    } else {
        throw std::runtime_error("nested_layer_norm_hip: unsupported dtype");
    }

    HIP_CHECK_NESTED(hipGetLastError());
    return output;
}

// ============================================================================
// Nested Linear (thin wrapper over matmul)
// ============================================================================

auto nested_linear_hip(const Tensor& values, const Tensor& weight,
                        const Tensor* bias, hipStream_t /*stream*/) -> Tensor {
    std::vector<Tensor> mm_inputs = {values, weight.transpose(0, 1)};
    auto result = dispatch_single<OpId::MatMul>(mm_inputs);
    if (bias != nullptr) {
        std::vector<Tensor> add_inputs = {result, *bias};
        result = dispatch_single<OpId::Add>(add_inputs);
    }
    return result;
}

} // namespace rocm
} // namespace tenzor
