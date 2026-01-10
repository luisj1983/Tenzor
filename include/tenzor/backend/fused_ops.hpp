#pragma once

#include "tenzor/core/tensor.hpp"
#include <tuple>
#include <vector>

namespace tenzor {
namespace cuda {

/**
 * @brief Fused LayerNorm forward pass CUDA kernel
 *
 * Computes LayerNorm in a single fused kernel launch for maximum performance.
 * Avoids multiple kernel launches and intermediate memory allocations.
 *
 * @param input Input tensor
 * @param normalized_shape Shape of normalized dimensions
 * @param weight Weight (gamma) parameter
 * @param bias Bias (beta) parameter
 * @param eps Epsilon for numerical stability
 * @return Tuple of (output, mean, inv_std) where mean and inv_std are saved for backward
 */
auto fused_layer_norm_cuda(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    float eps
) -> std::tuple<Tensor, Tensor, Tensor>;

/**
 * @brief Fused LayerNorm backward pass CUDA kernel
 */
auto fused_layer_norm_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& mean,
    const Tensor& inv_std,
    const std::vector<int64_t>& normalized_shape
) -> std::tuple<Tensor, Tensor, Tensor>;

/**
 * @brief Fused Linear + ReLU CUDA kernel
 */
auto fused_linear_relu_cuda(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias
) -> Tensor;

/**
 * @brief Fused BatchNorm + ReLU CUDA kernel
 */
auto fused_batchnorm_relu_cuda(
    const Tensor& input,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& weight,
    const Tensor& bias,
    float eps
) -> Tensor;

/**
 * @brief Fused Add + ReLU CUDA kernel
 */
auto fused_add_relu_cuda(const Tensor& a, const Tensor& b) -> Tensor;

/**
 * @brief Fused GELU activation CUDA kernel
 */
auto fused_gelu_cuda(const Tensor& input) -> Tensor;

/**
 * @brief Fused RMSNorm forward pass CUDA kernel
 *
 * RMSNorm: output = x * weight / sqrt(mean(x^2) + eps)
 * Commonly used in LLaMA, Mistral, and other modern LLMs.
 *
 * @param input Input tensor
 * @param weight Weight (gamma) parameter
 * @param eps Epsilon for numerical stability
 * @return Tuple of (output, rrms) where rrms is saved for backward
 */
auto fused_rms_norm_cuda(
    const Tensor& input,
    const Tensor& weight,
    float eps
) -> std::tuple<Tensor, Tensor>;

/**
 * @brief Fused RMSNorm backward pass CUDA kernel
 */
auto fused_rms_norm_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& rrms
) -> std::tuple<Tensor, Tensor>;

/**
 * @brief Fused multi-head attention CUDA kernel
 *
 * Computes scaled dot-product attention in a single fused kernel:
 * output = softmax(Q @ K.T / sqrt(head_dim)) @ V
 *
 * @param Q Query tensor (batch_heads, seq_len_q, head_dim)
 * @param K Key tensor (batch_heads, seq_len_k, head_dim)
 * @param V Value tensor (batch_heads, seq_len_k, head_dim)
 * @param scale Scaling factor (typically 1/sqrt(head_dim))
 * @return Output tensor (batch_heads, seq_len_q, head_dim)
 */
auto fused_attention_cuda(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    float scale
) -> Tensor;

} // namespace cuda
} // namespace tenzor
