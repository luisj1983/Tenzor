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

__global__ void nested_softmax_kernel(
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

    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        float max_val = -FLT_MAX;
        for (int64_t s = 0; s < len; ++s) {
            float v = values[(start + s) * D + d];
            if (v > max_val) max_val = v;
        }

        float sum = 0.0f;
        for (int64_t s = 0; s < len; ++s) {
            float v = expf(values[(start + s) * D + d] - max_val);
            output[(start + s) * D + d] = v;
            sum += v;
        }

        float inv_sum = 1.0f / sum;
        for (int64_t s = 0; s < len; ++s) {
            output[(start + s) * D + d] *= inv_sum;
        }
    }
}

auto nested_softmax_hip(const Tensor& values, const Tensor& offsets,
                         int64_t dim, hipStream_t stream) -> Tensor {
    auto shape = values.shape();
    int64_t D = (shape.size() > 1) ? shape[1] : 1;
    int64_t B = offsets.numel() - 1;

    auto output = tenzor::empty(std::vector<int64_t>(shape.begin(), shape.end()), values.dtype(), values.device());
    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (values.dtype() == DType::Float32) {
        hipLaunchKernelGGL(nested_softmax_kernel,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            values.data<float>(), output.data<float>(),
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

__global__ void nested_log_softmax_kernel(
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

    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        float max_val = -FLT_MAX;
        for (int64_t s = 0; s < len; ++s) {
            float v = values[(start + s) * D + d];
            if (v > max_val) max_val = v;
        }

        float sum = 0.0f;
        for (int64_t s = 0; s < len; ++s) {
            sum += expf(values[(start + s) * D + d] - max_val);
        }

        float log_sum = logf(sum);
        for (int64_t s = 0; s < len; ++s) {
            output[(start + s) * D + d] = (values[(start + s) * D + d] - max_val) - log_sum;
        }
    }
}

auto nested_log_softmax_hip(const Tensor& values, const Tensor& offsets,
                             int64_t dim, hipStream_t stream) -> Tensor {
    auto shape = values.shape();
    int64_t D = (shape.size() > 1) ? shape[1] : 1;
    int64_t B = offsets.numel() - 1;

    auto output = tenzor::empty(std::vector<int64_t>(shape.begin(), shape.end()), values.dtype(), values.device());
    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (values.dtype() == DType::Float32) {
        hipLaunchKernelGGL(nested_log_softmax_kernel,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            values.data<float>(), output.data<float>(),
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

__global__ void nested_sum_kernel(
    const float* __restrict__ values,
    float* __restrict__ output,
    const int64_t* __restrict__ offsets,
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

auto nested_sum_hip(const Tensor& values, const Tensor& offsets,
                     hipStream_t stream) -> Tensor {
    auto shape = values.shape();
    int64_t D = (shape.size() > 1) ? shape[1] : 1;
    int64_t B = offsets.numel() - 1;

    auto output = tenzor::zeros({B, D}, values.dtype(), values.device());
    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (values.dtype() == DType::Float32) {
        hipLaunchKernelGGL(nested_sum_kernel,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            values.data<float>(), output.data<float>(),
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

auto nested_mean_hip(const Tensor& values, const Tensor& offsets,
                      hipStream_t stream) -> Tensor {
    auto shape = values.shape();
    int64_t D = (shape.size() > 1) ? shape[1] : 1;
    int64_t B = offsets.numel() - 1;

    auto output = tenzor::zeros({B, D}, values.dtype(), values.device());
    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (values.dtype() == DType::Float32) {
        hipLaunchKernelGGL(nested_mean_kernel,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            values.data<float>(), output.data<float>(),
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

__global__ void nested_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    const int64_t* __restrict__ q_offsets,
    const int64_t* __restrict__ kv_offsets,
    float scale, int64_t head_dim, int64_t B, bool causal)
{
    int64_t b = blockIdx.x;
    if (b >= B) return;

    int64_t q_start = q_offsets[b], q_end = q_offsets[b + 1];
    int64_t kv_start = kv_offsets[b], kv_end = kv_offsets[b + 1];
    int64_t Lq = q_end - q_start;
    int64_t Lkv = kv_end - kv_start;
    if (Lq <= 0 || Lkv <= 0) return;

    for (int64_t qi = threadIdx.x; qi < Lq; qi += blockDim.x) {
        const float* q_row = Q + (q_start + qi) * head_dim;
        float* out_row = output + (q_start + qi) * head_dim;

        float max_score = -FLT_MAX;
        float sum_exp = 0.0f;

        for (int64_t hd = 0; hd < head_dim; ++hd) {
            out_row[hd] = 0.0f;
        }

        for (int64_t ki = 0; ki < Lkv; ++ki) {
            if (causal && ki > qi) break;

            const float* k_row = K + (kv_start + ki) * head_dim;
            const float* v_row = V + (kv_start + ki) * head_dim;

            float score = 0.0f;
            for (int64_t hd = 0; hd < head_dim; ++hd) {
                score += q_row[hd] * k_row[hd];
            }
            score *= scale;

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

        if (sum_exp > 0.0f) {
            float inv_sum = 1.0f / sum_exp;
            for (int64_t hd = 0; hd < head_dim; ++hd) {
                out_row[hd] *= inv_sum;
            }
        }
    }
}

auto nested_attention_hip(const Tensor& Q, const Tensor& K, const Tensor& V,
                           const Tensor& q_offsets, const Tensor& kv_offsets,
                           float scale, bool causal, hipStream_t stream) -> Tensor {
    int64_t head_dim = Q.shape().back();
    int64_t total_q_len = Q.shape()[0];
    int64_t B = q_offsets.numel() - 1;

    auto output = tenzor::zeros({total_q_len, head_dim}, Q.dtype(), Q.device());
    int threads = static_cast<int>(std::min(int64_t(256), total_q_len));

    if (Q.dtype() == DType::Float32) {
        hipLaunchKernelGGL(nested_attention_kernel,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            Q.data<float>(), K.data<float>(), V.data<float>(),
            output.data<float>(),
            q_offsets.data<int64_t>(), kv_offsets.data<int64_t>(),
            scale, head_dim, B, causal);
    } else {
        throw std::runtime_error("nested_attention_hip: only Float32 supported");
    }

    HIP_CHECK_NESTED(hipGetLastError());
    return output;
}

// ============================================================================
// Nested to Padded
// ============================================================================

__global__ void nested_to_padded_kernel(
    const float* __restrict__ values,
    float* __restrict__ padded,
    const int64_t* __restrict__ offsets,
    int64_t max_len, int64_t D, int64_t B, float padding_value)
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

auto nested_to_padded_hip(const Tensor& values, const Tensor& offsets,
                           int64_t max_len, float padding_value,
                           hipStream_t stream) -> Tensor {
    auto shape = values.shape();
    int64_t D = (shape.size() > 1) ? shape[1] : 1;
    int64_t B = offsets.numel() - 1;

    auto padded = tenzor::full({B, max_len, D}, padding_value, values.dtype(), values.device());
    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (values.dtype() == DType::Float32) {
        hipLaunchKernelGGL(nested_to_padded_kernel,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            values.data<float>(), padded.data<float>(),
            offsets.data<int64_t>(), max_len, D, B, padding_value);
    } else {
        throw std::runtime_error("nested_to_padded_hip: only Float32 supported");
    }

    HIP_CHECK_NESTED(hipGetLastError());
    return padded;
}

// ============================================================================
// Nested from Padded
// ============================================================================

__global__ void nested_from_padded_kernel(
    const float* __restrict__ padded,
    float* __restrict__ values,
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

auto nested_from_padded_hip(const Tensor& padded, const Tensor& offsets,
                              hipStream_t stream) -> Tensor {
    auto pad_shape = padded.shape();
    int64_t B = pad_shape[0];
    int64_t max_len = pad_shape[1];
    int64_t D = (pad_shape.size() > 2) ? pad_shape[2] : 1;

    auto offsets_cpu = (offsets.device().type == Device::Type::CPU) ? offsets : offsets.to(Device::cpu());
    int64_t total_len = offsets_cpu.data<int64_t>()[B];

    auto values = tenzor::empty({total_len, D}, padded.dtype(), padded.device());
    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (padded.dtype() == DType::Float32) {
        hipLaunchKernelGGL(nested_from_padded_kernel,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            padded.data<float>(), values.data<float>(),
            offsets.data<int64_t>(), max_len, D, B);
    } else {
        throw std::runtime_error("nested_from_padded_hip: only Float32 supported");
    }

    HIP_CHECK_NESTED(hipGetLastError());
    return values;
}

// ============================================================================
// Nested Layer Norm
// ============================================================================

__global__ void nested_layer_norm_kernel(
    const float* __restrict__ values,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int64_t* __restrict__ offsets,
    int64_t D, int64_t B, float eps)
{
    int64_t b = blockIdx.x;
    if (b >= B) return;

    int64_t start = offsets[b];
    int64_t end = offsets[b + 1];
    if (start >= end) return;

    for (int64_t row = start; row < end; ++row) {
        float mean = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            mean += values[row * D + d];
        }
        mean /= static_cast<float>(D);

        float var = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            float diff = values[row * D + d] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(D);

        float inv_std = rsqrtf(var + eps);

        for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
            float normalized = (values[row * D + d] - mean) * inv_std;
            output[row * D + d] = normalized * weight[d] + bias[d];
        }
    }
}

auto nested_layer_norm_hip(const Tensor& values, const Tensor& offsets,
                            const Tensor& weight, const Tensor& bias,
                            float eps, hipStream_t stream) -> Tensor {
    auto shape = values.shape();
    int64_t D = (shape.size() > 1) ? shape[1] : 1;
    int64_t B = offsets.numel() - 1;

    auto output = tenzor::empty(std::vector<int64_t>(shape.begin(), shape.end()), values.dtype(), values.device());
    int threads = static_cast<int>(std::min(D, int64_t(256)));

    if (values.dtype() == DType::Float32) {
        hipLaunchKernelGGL(nested_layer_norm_kernel,
            dim3(static_cast<unsigned>(B)), dim3(threads), 0, stream,
            values.data<float>(), output.data<float>(),
            weight.data<float>(), bias.data<float>(),
            offsets.data<int64_t>(), D, B, eps);
    } else {
        throw std::runtime_error("nested_layer_norm_hip: only Float32 supported");
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
