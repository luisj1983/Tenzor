#pragma once

// CUDA-specific fused-kernel declarations (namespace tenzor::cuda below).
// Despite living in the shared include/tenzor/backend/ directory, this is
// NOT a generic cross-backend fused-ops interface — ROCm has its own
// equivalent fused kernels (fused_layer_norm_hip, fused_attention_hip,
// fused_adam_step_hip, etc.) declared locally inside
// src/backends/rocm/rocm_kernel_registry.cpp instead of in a shared
// `namespace rocm` here, so there is no common interface between the two,
// only a CUDA-named one that happens to be reachable from generic code
// (src/ops/fused_ops.cpp, src/ops/fusion_optimizer.cpp). A real shared
// interface would need each backend's fused kernels declared under its own
// namespace in a header like this one, or a single backend-parameterized
// declaration set — out of scope for this comment; flagging so a future
// change doesn't mistake this for already being cross-backend.

#include "tenzor/core/tensor.hpp"
#include <tuple>
#include <utility>
#include <vector>

// Forward declare CUDA stream type to avoid including CUDA headers
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

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
 * Implements the FusedAttention contract from docs/internals/attention-contract.md.
 *
 * @param Q Query tensor (batch_heads, seq_len_q, head_dim)
 * @param K Key tensor (batch_heads, seq_len_k, head_dim)
 * @param V Value tensor (batch_heads, seq_len_k, head_dim)
 * @param scale Scaling factor (multiplicative; typically 1/sqrt(head_dim))
 * @param causal If true, apply causal (lower-triangular) mask before softmax
 * @return {output, logsumexp} — output (batch_heads, seq_len_q, head_dim);
 *         logsumexp (batch_heads, seq_len_q) is always Float32
 */
auto fused_attention_cuda(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    float scale,
    bool causal = false,
    float dropout_p = 0.0f,
    uint32_t rng_seed = 0u
) -> std::pair<Tensor, Tensor>;

/**
 * @brief Fused Flash Attention backward pass (tiled, memory-efficient)
 *
 * Recomputes attention in tiles using saved logsumexp, avoiding O(N^2) memory.
 * Falls back to composed-ops for unsupported head dimensions.
 *
 * Implements the FlashAttentionBackward contract from
 * docs/internals/attention-contract.md.
 *
 * @param dO Gradient of output (batch_heads, seq_len, head_dim)
 * @param Q Query tensor (batch_heads, seq_len, head_dim)
 * @param K Key tensor (batch_heads, seq_len, head_dim)
 * @param V Value tensor (batch_heads, seq_len, head_dim)
 * @param O Forward output (batch_heads, seq_len, head_dim)
 * @param L Row-wise logsumexp from forward (batch_heads, seq_len) — always Float32
 * @param scale Scaling factor (multiplicative; typically 1/sqrt(head_dim))
 * @param causal Whether to apply causal masking
 * @param dropout_p Dropout probability used in forward (0 disables dropout)
 * @param philox_seed Int64 scalar Tensor — Philox seed used in forward (empty if dropout_p == 0)
 * @param philox_offset Int64 scalar Tensor — Philox offset used in forward (empty if dropout_p == 0)
 * @return {dQ, dK, dV}
 */
auto flash_attention_backward_cuda(
    const Tensor& dO,
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    const Tensor& O,
    const Tensor& L,
    float scale,
    bool causal,
    float dropout_p = 0.0f,
    const Tensor& philox_seed = Tensor{},
    const Tensor& philox_offset = Tensor{}
) -> std::vector<Tensor>;

/**
 * @brief cuDNN SDPA (Scaled Dot-Product Attention) with Flash Attention
 *
 * Uses cuDNN Graph API for optimized fused attention with Tensor Core support.
 * Provides up to 2x speedup over separate BMM operations.
 *
 * Implements the FusedAttention contract from docs/internals/attention-contract.md.
 * Per docs/internals/attention-contract.md, Outputs=(output, logsumexp) —
 * the cuDNN graph is configured with set_generate_stats(true) so the Stats
 * tensor (LSE, FP32) is materialized alongside the output, matching every
 * other backend's FusedAttention contract instead of silently disabling the
 * fast fused-kernel backward path.
 *
 * @param Q Query tensor [batch, num_heads, seq_len_q, head_dim] (forced contiguous)
 * @param K Key tensor [batch, num_heads, seq_len_k, head_dim] (forced contiguous)
 * @param V Value tensor [batch, num_heads, seq_len_k, head_dim] (forced contiguous)
 * @param scale Scaling factor (multiplicative; typically 1/sqrt(head_dim))
 * @param causal If true, apply causal mask via cuDNN's set_causal_mask_bottom_right(true)
 * @param stream CUDA stream to bind to the cuDNN handle before graph execution.
 *               If nullptr, the default stream is used. Must match the stream on
 *               which the output is subsequently consumed to avoid a read-before-
 *               write race on the returned tensor.
 * @return {output, logsumexp} — output has same shape as Q; logsumexp is
 *         Float32 shaped [batch, num_heads, seq_len_q].
 */
auto cudnn_sdpa_forward(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    float scale,
    bool causal = false,
    cudaStream_t stream = nullptr
) -> std::pair<Tensor, Tensor>;

/**
 * @brief Check if cuDNN SDPA is supported for the given configuration
 *
 * @param batch Batch size
 * @param num_heads Number of attention heads
 * @param seq_len_q Query sequence length
 * @param seq_len_k Key/Value sequence length
 * @param head_dim Head dimension (must be 32, 64, 128, or 256)
 * @param device_index CUDA device the tensors live on. Pass the operand's
 *        device().index so the arch check targets the right GPU on a
 *        heterogeneous multi-GPU host. -1 (default) means "current device".
 * @return true if supported, false otherwise
 */
auto cudnn_sdpa_supported(
    int64_t batch,
    int64_t num_heads,
    int64_t seq_len_q,
    int64_t seq_len_k,
    int64_t head_dim,
    int device_index = -1
) -> bool;

/**
 * @brief Fused Adam optimizer step CUDA kernel
 *
 * Performs the complete Adam update in a single kernel launch:
 * - First moment update: m = beta1 * m + (1-beta1) * grad
 * - Second moment update: v = beta2 * v + (1-beta2) * grad^2
 * - Parameter update with bias correction
 *
 * @param param Parameter tensor (modified in-place)
 * @param grad Gradient tensor
 * @param exp_avg First moment buffer (modified in-place)
 * @param exp_avg_sq Second moment buffer (modified in-place)
 * @param lr Learning rate
 * @param beta1 First moment decay rate (default 0.9)
 * @param beta2 Second moment decay rate (default 0.999)
 * @param eps Epsilon for numerical stability (default 1e-8)
 * @param weight_decay Weight decay coefficient (default 0)
 * @param step Current step count (for bias correction)
 * @param decoupled_weight_decay True for AdamW, false for L2 regularization
 * @param stream CUDA stream for async execution
 */
auto fused_adam_step_cuda(
    Tensor& param,
    const Tensor& grad,
    Tensor& exp_avg,
    Tensor& exp_avg_sq,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    int64_t step,
    bool decoupled_weight_decay,
    cudaStream_t stream = nullptr,
    Tensor* max_exp_avg_sq = nullptr,
    bool amsgrad = false
) -> void;

/**
 * @brief Fused SGD optimizer step CUDA kernel
 *
 * Performs SGD with momentum in a single kernel launch.
 *
 * @param param Parameter tensor (modified in-place)
 * @param grad Gradient tensor
 * @param momentum_buffer Momentum buffer (optional, modified in-place)
 * @param lr Learning rate
 * @param momentum Momentum coefficient
 * @param weight_decay Weight decay coefficient
 * @param dampening Dampening for momentum
 * @param nesterov Use Nesterov momentum
 * @param stream CUDA stream for async execution
 */
auto fused_sgd_step_cuda(
    Tensor& param,
    const Tensor& grad,
    Tensor* momentum_buffer,
    float lr,
    float momentum,
    float weight_decay,
    float dampening,
    bool nesterov,
    bool first_step,
    cudaStream_t stream = nullptr
) -> void;

} // namespace cuda
} // namespace tenzor
