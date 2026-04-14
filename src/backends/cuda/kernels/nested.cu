/**
 * @file nested.cu
 * @brief CUDA kernels for nested (ragged) tensor operations.
 *
 * All kernels use a one-block-per-batch-element strategy: blockIdx.x selects
 * the batch element and threads within the block cooperate on the inner
 * dimension D.  Shared memory is used for per-segment reductions where
 * needed (softmax max/sum).
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace tenzor {
namespace cuda {

// ============================================================================
// Error checking
// ============================================================================

#ifndef CUDA_CHECK_NESTED
#define CUDA_CHECK_NESTED(call)                                                \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(                                          \
                std::string("CUDA error in nested at ") + __FILE__ + ":" +    \
                std::to_string(__LINE__) + " - " + cudaGetErrorString(err));   \
        }                                                                      \
    } while (0)
#endif

// ============================================================================
// Segmented Softmax
// ============================================================================

__global__ void nested_softmax_kernel(
    const float* __restrict__ values,   // [total_len, D]
    float* __restrict__ output,         // [total_len, D]
    const int64_t* __restrict__ offsets, // [B+1]
    int64_t D,
    int64_t B)
{
    int64_t b = blockIdx.x;
    if (b >= B) return;

    int64_t start = offsets[b];
    int64_t end = offsets[b + 1];
    int64_t len = end - start;
    if (len <= 0) return;

    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        // Find max in segment for numerical stability
        float max_val = -FLT_MAX;
        for (int64_t s = 0; s < len; ++s) {
            float v = values[(start + s) * D + d];
            if (v > max_val) max_val = v;
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int64_t s = 0; s < len; ++s) {
            float v = expf(values[(start + s) * D + d] - max_val);
            output[(start + s) * D + d] = v;
            sum += v;
        }

        // Normalize
        float inv_sum = 1.0f / sum;
        for (int64_t s = 0; s < len; ++s) {
            output[(start + s) * D + d] *= inv_sum;
        }
    }
}

__global__ void nested_softmax_kernel_f64(
    const double* __restrict__ values,
    double* __restrict__ output,
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
        double max_val = -DBL_MAX;
        for (int64_t s = 0; s < len; ++s) {
            double v = values[(start + s) * D + d];
            if (v > max_val) max_val = v;
        }

        double sum = 0.0;
        for (int64_t s = 0; s < len; ++s) {
            double v = exp(values[(start + s) * D + d] - max_val);
            output[(start + s) * D + d] = v;
            sum += v;
        }

        double inv_sum = 1.0 / sum;
        for (int64_t s = 0; s < len; ++s) {
            output[(start + s) * D + d] *= inv_sum;
        }
    }
}

auto nested_softmax_cuda(const Tensor& values, const Tensor& offsets,
                          int64_t dim, cudaStream_t stream) -> Tensor {
    auto shape = values.shape();
    int64_t total_len = shape[0];
    int64_t D = (shape.size() > 1) ? shape[1] : 1;
    int64_t B = offsets.numel() - 1;

    auto output = tenzor::empty(std::vector<int64_t>(shape.begin(), shape.end()), values.dtype(), values.device());

    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (values.dtype() == DType::Float32) {
        nested_softmax_kernel<<<static_cast<unsigned>(B), threads, 0, stream>>>(
            values.data<float>(), output.data<float>(),
            offsets.data<int64_t>(), D, B);
    } else if (values.dtype() == DType::Float64) {
        nested_softmax_kernel_f64<<<static_cast<unsigned>(B), threads, 0, stream>>>(
            values.data<double>(), output.data<double>(),
            offsets.data<int64_t>(), D, B);
    } else {
        throw std::runtime_error("nested_softmax_cuda: unsupported dtype");
    }

    CUDA_CHECK_NESTED(cudaGetLastError());
    return output;
}

// ============================================================================
// Segmented Log-Softmax
// ============================================================================

__global__ void nested_log_softmax_kernel(
    const float* __restrict__ values,   // [total_len, D]
    float* __restrict__ output,         // [total_len, D]
    const int64_t* __restrict__ offsets, // [B+1]
    int64_t D,
    int64_t B)
{
    int64_t b = blockIdx.x;
    if (b >= B) return;

    int64_t start = offsets[b];
    int64_t end = offsets[b + 1];
    int64_t len = end - start;
    if (len <= 0) return;

    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        // Find max in segment for numerical stability
        float max_val = -FLT_MAX;
        for (int64_t s = 0; s < len; ++s) {
            float v = values[(start + s) * D + d];
            if (v > max_val) max_val = v;
        }

        // Compute sum of exp(x - max)
        float sum = 0.0f;
        for (int64_t s = 0; s < len; ++s) {
            sum += expf(values[(start + s) * D + d] - max_val);
        }

        // Write log-softmax: (x - max) - log(sum)
        float log_sum = logf(sum);
        for (int64_t s = 0; s < len; ++s) {
            output[(start + s) * D + d] = (values[(start + s) * D + d] - max_val) - log_sum;
        }
    }
}

__global__ void nested_log_softmax_kernel_f64(
    const double* __restrict__ values,
    double* __restrict__ output,
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
        double max_val = -DBL_MAX;
        for (int64_t s = 0; s < len; ++s) {
            double v = values[(start + s) * D + d];
            if (v > max_val) max_val = v;
        }

        double sum = 0.0;
        for (int64_t s = 0; s < len; ++s) {
            sum += exp(values[(start + s) * D + d] - max_val);
        }

        double log_sum = log(sum);
        for (int64_t s = 0; s < len; ++s) {
            output[(start + s) * D + d] = (values[(start + s) * D + d] - max_val) - log_sum;
        }
    }
}

auto nested_log_softmax_cuda(const Tensor& values, const Tensor& offsets,
                              int64_t dim, cudaStream_t stream) -> Tensor {
    auto shape = values.shape();
    int64_t total_len = shape[0];
    int64_t D = (shape.size() > 1) ? shape[1] : 1;
    int64_t B = offsets.numel() - 1;

    auto output = tenzor::empty(std::vector<int64_t>(shape.begin(), shape.end()), values.dtype(), values.device());

    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (values.dtype() == DType::Float32) {
        nested_log_softmax_kernel<<<static_cast<unsigned>(B), threads, 0, stream>>>(
            values.data<float>(), output.data<float>(),
            offsets.data<int64_t>(), D, B);
    } else if (values.dtype() == DType::Float64) {
        nested_log_softmax_kernel_f64<<<static_cast<unsigned>(B), threads, 0, stream>>>(
            values.data<double>(), output.data<double>(),
            offsets.data<int64_t>(), D, B);
    } else {
        throw std::runtime_error("nested_log_softmax_cuda: unsupported dtype");
    }

    CUDA_CHECK_NESTED(cudaGetLastError());
    return output;
}

// ============================================================================
// Segmented Sum Reduction
// ============================================================================

__global__ void nested_sum_kernel(
    const float* __restrict__ values,   // [total_len, D]
    float* __restrict__ output,         // [B, D]
    const int64_t* __restrict__ offsets, // [B+1]
    int64_t D, int64_t B)
{
    int64_t b = blockIdx.x;
    if (b >= B) return;

    int64_t start = offsets[b];
    int64_t end = offsets[b + 1];

    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        float sum = 0.0f;
        for (int64_t s = start; s < end; ++s) {
            sum += values[s * D + d];
        }
        output[b * D + d] = sum;
    }
}

__global__ void nested_sum_kernel_f64(
    const double* __restrict__ values,
    double* __restrict__ output,
    const int64_t* __restrict__ offsets,
    int64_t D, int64_t B)
{
    int64_t b = blockIdx.x;
    if (b >= B) return;

    int64_t start = offsets[b];
    int64_t end = offsets[b + 1];

    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        double sum = 0.0;
        for (int64_t s = start; s < end; ++s) {
            sum += values[s * D + d];
        }
        output[b * D + d] = sum;
    }
}

auto nested_sum_cuda(const Tensor& values, const Tensor& offsets,
                      cudaStream_t stream) -> Tensor {
    auto shape = values.shape();
    int64_t D = (shape.size() > 1) ? shape[1] : 1;
    int64_t B = offsets.numel() - 1;

    auto output = tenzor::zeros({B, D}, values.dtype(), values.device());

    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (values.dtype() == DType::Float32) {
        nested_sum_kernel<<<static_cast<unsigned>(B), threads, 0, stream>>>(
            values.data<float>(), output.data<float>(),
            offsets.data<int64_t>(), D, B);
    } else if (values.dtype() == DType::Float64) {
        nested_sum_kernel_f64<<<static_cast<unsigned>(B), threads, 0, stream>>>(
            values.data<double>(), output.data<double>(),
            offsets.data<int64_t>(), D, B);
    } else {
        throw std::runtime_error("nested_sum_cuda: unsupported dtype");
    }

    CUDA_CHECK_NESTED(cudaGetLastError());
    return output;
}

// ============================================================================
// Segmented Mean Reduction
// ============================================================================

__global__ void nested_mean_kernel(
    const float* __restrict__ values,
    float* __restrict__ output,
    const int64_t* __restrict__ offsets,
    int64_t D, int64_t B)
{
    int64_t b = blockIdx.x;
    if (b >= B) return;

    int64_t start = offsets[b];
    int64_t end = offsets[b + 1];
    int64_t len = end - start;
    if (len <= 0) return;

    float inv_len = 1.0f / static_cast<float>(len);

    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        float sum = 0.0f;
        for (int64_t s = start; s < end; ++s) {
            sum += values[s * D + d];
        }
        output[b * D + d] = sum * inv_len;
    }
}

auto nested_mean_cuda(const Tensor& values, const Tensor& offsets,
                       cudaStream_t stream) -> Tensor {
    auto shape = values.shape();
    int64_t D = (shape.size() > 1) ? shape[1] : 1;
    int64_t B = offsets.numel() - 1;

    auto output = tenzor::zeros({B, D}, values.dtype(), values.device());

    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (values.dtype() == DType::Float32) {
        nested_mean_kernel<<<static_cast<unsigned>(B), threads, 0, stream>>>(
            values.data<float>(), output.data<float>(),
            offsets.data<int64_t>(), D, B);
    } else {
        throw std::runtime_error("nested_mean_cuda: only Float32 currently supported");
    }

    CUDA_CHECK_NESTED(cudaGetLastError());
    return output;
}

// ============================================================================
// Nested Attention (variable-length scaled dot-product attention)
// ============================================================================

__global__ void nested_attention_kernel(
    const float* __restrict__ Q,            // [total_q_len, head_dim]
    const float* __restrict__ K,            // [total_kv_len, head_dim]
    const float* __restrict__ V,            // [total_kv_len, head_dim]
    float* __restrict__ output,             // [total_q_len, head_dim]
    const int64_t* __restrict__ q_offsets,  // [B+1]
    const int64_t* __restrict__ kv_offsets, // [B+1]
    float scale,
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

    // Each thread handles one query row using online softmax attention
    for (int64_t qi = threadIdx.x; qi < Lq; qi += blockDim.x) {
        const float* q_row = Q + (q_start + qi) * head_dim;
        float* out_row = output + (q_start + qi) * head_dim;

        // Initialize online softmax accumulators
        float max_score = -FLT_MAX;
        float sum_exp = 0.0f;

        // Zero output accumulator
        for (int64_t hd = 0; hd < head_dim; ++hd) {
            out_row[hd] = 0.0f;
        }

        for (int64_t ki = 0; ki < Lkv; ++ki) {
            // Causal mask: skip future positions
            if (causal && ki > qi) break;

            const float* k_row = K + (kv_start + ki) * head_dim;
            const float* v_row = V + (kv_start + ki) * head_dim;

            // Compute dot(q, k) * scale
            float score = 0.0f;
            for (int64_t hd = 0; hd < head_dim; ++hd) {
                score += q_row[hd] * k_row[hd];
            }
            score *= scale;

            // Online softmax update
            if (score > max_score) {
                float correction = expf(max_score - score);
                sum_exp = sum_exp * correction + 1.0f;
                for (int64_t hd = 0; hd < head_dim; ++hd) {
                    out_row[hd] = out_row[hd] * correction + v_row[hd];
                }
                max_score = score;
            } else {
                float w = expf(score - max_score);
                sum_exp += w;
                for (int64_t hd = 0; hd < head_dim; ++hd) {
                    out_row[hd] += w * v_row[hd];
                }
            }
        }

        // Normalize
        if (sum_exp > 0.0f) {
            float inv_sum = 1.0f / sum_exp;
            for (int64_t hd = 0; hd < head_dim; ++hd) {
                out_row[hd] *= inv_sum;
            }
        }
    }
}

auto nested_attention_cuda(const Tensor& Q, const Tensor& K, const Tensor& V,
                            const Tensor& q_offsets, const Tensor& kv_offsets,
                            float scale, bool causal, cudaStream_t stream) -> Tensor {
    int64_t head_dim = Q.shape().back();
    int64_t total_q_len = Q.shape()[0];
    int64_t B = q_offsets.numel() - 1;

    auto output = tenzor::zeros({total_q_len, head_dim}, Q.dtype(), Q.device());

    // Use enough threads to cover query rows (one thread per query row)
    int threads = static_cast<int>(std::min(int64_t(256), total_q_len));

    if (Q.dtype() == DType::Float32) {
        nested_attention_kernel<<<static_cast<unsigned>(B), threads, 0, stream>>>(
            Q.data<float>(), K.data<float>(), V.data<float>(),
            output.data<float>(),
            q_offsets.data<int64_t>(), kv_offsets.data<int64_t>(),
            scale, head_dim, B, causal);
    } else {
        throw std::runtime_error("nested_attention_cuda: only Float32 currently supported");
    }

    CUDA_CHECK_NESTED(cudaGetLastError());
    return output;
}

// ============================================================================
// Nested Attention Backward
// ============================================================================

__global__ void nested_attention_backward_kernel(
    const float* __restrict__ grad_out,     // [total_q_len, head_dim]
    const float* __restrict__ Q,            // [total_q_len, head_dim]
    const float* __restrict__ K,            // [total_kv_len, head_dim]
    const float* __restrict__ V,            // [total_kv_len, head_dim]
    float* __restrict__ grad_Q,             // [total_q_len, head_dim]
    float* __restrict__ grad_K,             // [total_kv_len, head_dim]
    float* __restrict__ grad_V,             // [total_kv_len, head_dim]
    const int64_t* __restrict__ q_offsets,  // [B+1]
    const int64_t* __restrict__ kv_offsets, // [B+1]
    float scale,
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

    // Each thread handles one query row
    for (int64_t qi = threadIdx.x; qi < Lq; qi += blockDim.x) {
        const float* q_row = Q + (q_start + qi) * head_dim;
        const float* do_row = grad_out + (q_start + qi) * head_dim;
        float* gq_row = grad_Q + (q_start + qi) * head_dim;

        int64_t ki_end = causal ? min(Lkv, qi + 1) : Lkv;

        // Step 1: Recompute attention weights via online softmax
        // First pass: find max score
        float max_score = -FLT_MAX;
        for (int64_t ki = 0; ki < ki_end; ++ki) {
            const float* k_row = K + (kv_start + ki) * head_dim;
            float score = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) {
                score += q_row[d] * k_row[d];
            }
            score *= scale;
            if (score > max_score) max_score = score;
        }

        // Second pass: compute softmax weights and d_attn dot products simultaneously
        float sum_exp = 0.0f;
        for (int64_t ki = 0; ki < ki_end; ++ki) {
            const float* k_row = K + (kv_start + ki) * head_dim;
            float score = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) {
                score += q_row[d] * k_row[d];
            }
            score *= scale;
            sum_exp += expf(score - max_score);
        }
        float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;

        // Step 2: Compute softmax_dot = sum_ki(attn_w[ki] * d_attn[ki])
        // where d_attn[ki] = dot(do_row, v_row[ki])
        float softmax_dot = 0.0f;
        for (int64_t ki = 0; ki < ki_end; ++ki) {
            const float* k_row = K + (kv_start + ki) * head_dim;
            const float* v_row = V + (kv_start + ki) * head_dim;
            float score = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) {
                score += q_row[d] * k_row[d];
            }
            score *= scale;
            float w = expf(score - max_score) * inv_sum;

            float dot_dov = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) {
                dot_dov += do_row[d] * v_row[d];
            }
            softmax_dot += w * dot_dov;
        }

        // Step 3: Compute gradients
        for (int64_t ki = 0; ki < ki_end; ++ki) {
            const float* k_row = K + (kv_start + ki) * head_dim;
            const float* v_row = V + (kv_start + ki) * head_dim;
            float* gk_row = grad_K + (kv_start + ki) * head_dim;
            float* gv_row = grad_V + (kv_start + ki) * head_dim;

            float score = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) {
                score += q_row[d] * k_row[d];
            }
            score *= scale;
            float w = expf(score - max_score) * inv_sum;

            // grad_V += w * do_row
            for (int64_t d = 0; d < head_dim; ++d) {
                atomicAdd(&gv_row[d], w * do_row[d]);
            }

            // d_attn = dot(do_row, v_row)
            float dot_dov = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) {
                dot_dov += do_row[d] * v_row[d];
            }

            // d_score = w * (d_attn - softmax_dot) * scale
            float ds = w * (dot_dov - softmax_dot) * scale;

            // grad_Q += ds * k_row
            for (int64_t d = 0; d < head_dim; ++d) {
                gq_row[d] += ds * k_row[d];
            }

            // grad_K += ds * q_row
            for (int64_t d = 0; d < head_dim; ++d) {
                atomicAdd(&gk_row[d], ds * q_row[d]);
            }
        }
    }
}

auto nested_attention_backward_cuda(const Tensor& grad_out, const Tensor& Q,
                                     const Tensor& K, const Tensor& V,
                                     const Tensor& attn_out,
                                     const Tensor& q_offsets, const Tensor& kv_offsets,
                                     float scale, bool causal, cudaStream_t stream)
    -> std::vector<Tensor> {
    int64_t head_dim = Q.shape().back();
    int64_t total_q_len = Q.shape()[0];
    int64_t total_kv_len = K.shape()[0];
    int64_t B = q_offsets.numel() - 1;

    auto grad_Q = tenzor::zeros({total_q_len, head_dim}, Q.dtype(), Q.device());
    auto grad_K = tenzor::zeros({total_kv_len, head_dim}, K.dtype(), K.device());
    auto grad_V = tenzor::zeros({total_kv_len, head_dim}, V.dtype(), V.device());

    int threads = static_cast<int>(std::min(int64_t(256), total_q_len));

    if (Q.dtype() == DType::Float32) {
        nested_attention_backward_kernel<<<static_cast<unsigned>(B), threads, 0, stream>>>(
            grad_out.data<float>(), Q.data<float>(), K.data<float>(), V.data<float>(),
            grad_Q.data<float>(), grad_K.data<float>(), grad_V.data<float>(),
            q_offsets.data<int64_t>(), kv_offsets.data<int64_t>(),
            scale, head_dim, B, causal);
    } else {
        throw std::runtime_error("nested_attention_backward_cuda: only Float32 supported");
    }

    CUDA_CHECK_NESTED(cudaGetLastError());
    return {grad_Q, grad_K, grad_V};
}

// ============================================================================
// Nested to Padded (convert ragged -> dense padded tensor)
// ============================================================================

__global__ void nested_to_padded_kernel(
    const float* __restrict__ values,   // [total_len, D]
    float* __restrict__ padded,         // [B, max_len, D]
    const int64_t* __restrict__ offsets, // [B+1]
    int64_t max_len, int64_t D, int64_t B,
    float padding_value)
{
    int64_t b = blockIdx.x;
    if (b >= B) return;

    int64_t start = offsets[b];
    int64_t end = offsets[b + 1];
    int64_t len = end - start;

    for (int64_t pos = 0; pos < max_len; ++pos) {
        for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
            int64_t out_idx = (b * max_len + pos) * D + d;
            if (pos < len) {
                padded[out_idx] = values[(start + pos) * D + d];
            } else {
                padded[out_idx] = padding_value;
            }
        }
    }
}

auto nested_to_padded_cuda(const Tensor& values, const Tensor& offsets,
                            int64_t max_len, float padding_value,
                            cudaStream_t stream) -> Tensor {
    auto shape = values.shape();
    int64_t D = (shape.size() > 1) ? shape[1] : 1;
    int64_t B = offsets.numel() - 1;

    auto padded = tenzor::full({B, max_len, D}, padding_value,
                                values.dtype(), values.device());

    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (values.dtype() == DType::Float32) {
        nested_to_padded_kernel<<<static_cast<unsigned>(B), threads, 0, stream>>>(
            values.data<float>(), padded.data<float>(),
            offsets.data<int64_t>(), max_len, D, B, padding_value);
    } else {
        throw std::runtime_error("nested_to_padded_cuda: only Float32 currently supported");
    }

    CUDA_CHECK_NESTED(cudaGetLastError());
    return padded;
}

// ============================================================================
// Nested from Padded (convert dense padded -> ragged values)
// ============================================================================

__global__ void nested_from_padded_kernel(
    const float* __restrict__ padded,   // [B, max_len, D]
    float* __restrict__ values,         // [total_len, D]
    const int64_t* __restrict__ offsets, // [B+1]
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

auto nested_from_padded_cuda(const Tensor& padded, const Tensor& offsets,
                              cudaStream_t stream) -> Tensor {
    auto pad_shape = padded.shape();
    int64_t B = pad_shape[0];
    int64_t max_len = pad_shape[1];
    int64_t D = (pad_shape.size() > 2) ? pad_shape[2] : 1;

    // Compute total_len from offsets
    auto offsets_cpu = (offsets.device().type == Device::Type::CPU) ? offsets : offsets.to(Device::cpu());
    int64_t total_len = offsets_cpu.data<int64_t>()[B];

    auto values = tenzor::empty({total_len, D}, padded.dtype(), padded.device());

    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (padded.dtype() == DType::Float32) {
        nested_from_padded_kernel<<<static_cast<unsigned>(B), threads, 0, stream>>>(
            padded.data<float>(), values.data<float>(),
            offsets.data<int64_t>(), max_len, D, B);
    } else {
        throw std::runtime_error("nested_from_padded_cuda: only Float32 currently supported");
    }

    CUDA_CHECK_NESTED(cudaGetLastError());
    return values;
}

// ============================================================================
// Nested Layer Norm
// ============================================================================

__global__ void nested_layer_norm_kernel(
    const float* __restrict__ values,   // [total_len, D]
    float* __restrict__ output,         // [total_len, D]
    const float* __restrict__ weight,   // [D]
    const float* __restrict__ bias,     // [D]
    const int64_t* __restrict__ offsets, // [B+1]
    int64_t D, int64_t B, float eps)
{
    int64_t b = blockIdx.x;
    if (b >= B) return;

    int64_t start = offsets[b];
    int64_t end = offsets[b + 1];
    int64_t len = end - start;
    if (len <= 0) return;

    // For each row in the segment, compute LN across the D dimension
    for (int64_t row = start; row < end; ++row) {
        // Compute mean
        float mean = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            mean += values[row * D + d];
        }
        mean /= static_cast<float>(D);

        // Compute variance
        float var = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            float diff = values[row * D + d] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(D);

        float inv_std = rsqrtf(var + eps);

        // Normalize and apply affine
        for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
            float normalized = (values[row * D + d] - mean) * inv_std;
            output[row * D + d] = normalized * weight[d] + bias[d];
        }
    }
}

auto nested_layer_norm_cuda(const Tensor& values, const Tensor& offsets,
                             const Tensor& weight, const Tensor& bias,
                             float eps, cudaStream_t stream) -> Tensor {
    auto shape = values.shape();
    int64_t D = (shape.size() > 1) ? shape[1] : 1;
    int64_t B = offsets.numel() - 1;

    auto output = tenzor::empty(std::vector<int64_t>(shape.begin(), shape.end()), values.dtype(), values.device());

    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (values.dtype() == DType::Float32) {
        nested_layer_norm_kernel<<<static_cast<unsigned>(B), threads, 0, stream>>>(
            values.data<float>(), output.data<float>(),
            weight.data<float>(), bias.data<float>(),
            offsets.data<int64_t>(), D, B, eps);
    } else {
        throw std::runtime_error("nested_layer_norm_cuda: only Float32 currently supported");
    }

    CUDA_CHECK_NESTED(cudaGetLastError());
    return output;
}

// ============================================================================
// Nested Linear (just matmul on packed values; thin wrapper)
// ============================================================================

auto nested_linear_cuda(const Tensor& values, const Tensor& weight,
                         const Tensor* bias, cudaStream_t stream) -> Tensor {
    // values: [total_len, D_in], weight: [D_out, D_in]
    // result = values @ weight^T + bias
    std::vector<Tensor> mm_inputs = {values, weight.transpose(0, 1)};
    auto result = dispatch_single<OpId::MatMul>(mm_inputs);
    if (bias != nullptr) {
        std::vector<Tensor> add_inputs = {result, *bias};
        result = dispatch_single<OpId::Add>(add_inputs);
    }
    return result;
}

} // namespace cuda
} // namespace tenzor
