/**
 * @file attention.hpp
 * @brief Multi-head attention mechanisms for transformer models
 *
 * Implements multi-head self-attention and cross-attention layers
 * following the "Attention Is All You Need" paper (Vaswani et al., 2017).
 */

#pragma once

#include <memory>
#include <optional>
#include "../module.hpp"
#include "linear.hpp"
#include "dropout.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Multi-Head Attention layer.
 *
 * Implements multi-head attention mechanism with scaled dot-product attention.
 * This is the core building block of transformer architectures.
 *
 * The attention mechanism computes:
 * - Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
 *
 * Multi-head attention applies multiple attention mechanisms in parallel,
 * then concatenates and projects the results:
 * - MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
 * - where head_i = Attention(Q*W_Q^i, K*W_K^i, V*W_V^i)
 *
 * Shape transformations:
 * - Query: (N, L, E) or (L, N, E) if batch_first=false
 * - Key: (N, S, E) or (S, N, E) if batch_first=false
 * - Value: (N, S, E) or (S, N, E) if batch_first=false
 * - Output: (N, L, E) or (L, N, E) if batch_first=false
 * - Attention weights: (N, num_heads, L, S)
 *
 * Where:
 * - N = batch size
 * - L = target sequence length
 * - S = source sequence length
 * - E = embedding dimension
 * - num_heads = number of attention heads
 *
 * Computational Complexity:
 * - Time: O(L*S*E + L*S*num_heads)
 * - Memory: O(L*S*num_heads) for attention weights
 *
 * @code
 * // Create 8-head attention with 512-dimensional embeddings
 * MultiheadAttention attn(512, 8);
 *
 * // Self-attention
 * Variable query(Tensor({batch, seq_len, 512}, DType::Float32, Device::cpu()), true);
 * auto [output, attn_weights] = attn.forward(query, query, query);
 *
 * // Cross-attention with masking
 * Variable key(Tensor({batch, src_len, 512}, DType::Float32, Device::cpu()), true);
 * Variable value(Tensor({batch, src_len, 512}, DType::Float32, Device::cpu()), true);
 * Variable mask(Tensor({batch, seq_len, src_len}, DType::Float32, Device::cpu()));
 * auto [out, weights] = attn.forward(query, key, value, Tensor{}, mask, true);
 * @endcode
 *
 * @see TransformerEncoderLayer for usage in encoder
 * @see TransformerDecoderLayer for usage in decoder
 *
 * @note For best performance, use embed_dim divisible by num_heads
 */
class MultiheadAttention : public Module {
public:
    // Bring base class forward into scope (avoid hiding by multi-param forward)
    using Module::forward;

    /**
     * @brief Construct multi-head attention layer.
     *
     * @param embed_dim Total dimension of the model
     * @param num_heads Number of parallel attention heads
     * @param dropout Dropout probability on attention weights (default: 0.0)
     * @param bias If true, add bias to input/output projections (default: true)
     * @param add_bias_kv If true, add bias to key/value projections (default: false)
     * @param add_zero_attn If true, add zero attention weight (default: false)
     * @param kdim Key dimension (default: 0, uses embed_dim)
     * @param vdim Value dimension (default: 0, uses embed_dim)
     * @param batch_first If true, input/output tensors are (batch, seq, feature)
     *                    otherwise (seq, batch, feature) (default: false)
     * @param is_causal If true, applies causal masking inside fused attention
     *                  kernels instead of requiring an explicit mask (default: false)
     *
     * @throws std::invalid_argument if embed_dim not divisible by num_heads
     * @throws std::invalid_argument if dropout not in [0, 1]
     *
     * @code
     * MultiheadAttention attn1(512, 8);                    // Standard setup
     * MultiheadAttention attn2(512, 8, 0.1, true, false,
     *                          false, 0, 0, true);         // Batch-first with dropout
     * MultiheadAttention attn3(512, 8, 0.1, true, false,
     *                          false, 256, 256, false);    // Different K/V dims
     * MultiheadAttention attn4(512, 8, 0.0, true, false,
     *                          false, 0, 0, true, true);   // Causal attention
     * @endcode
     */
    MultiheadAttention(int64_t embed_dim,
                      int64_t num_heads,
                      double dropout = 0.0,
                      bool bias = true,
                      bool add_bias_kv = false,
                      bool add_zero_attn = false,
                      int64_t kdim = 0,
                      int64_t vdim = 0,
                      bool batch_first = false,
                      bool is_causal = false);

    /**
     * @brief Forward pass through multi-head attention.
     *
     * Computes multi-head attention output and optionally returns attention weights.
     *
     * @param query Query tensor of shape (N, L, E) or (L, N, E)
     * @param key Key tensor of shape (N, S, E) or (S, N, E)
     * @param value Value tensor of shape (N, S, E) or (S, N, E)
     * @param key_padding_mask Binary mask (N, S) where True/1.0 indicates padding
     * @param attn_mask Attention mask (L, S) or (N*num_heads, L, S) for custom patterns
     * @param need_weights If true, return attention weights (default: true).
     *        When false, fused/flash attention paths may be used for better
     *        performance (these paths cannot return intermediate attention weights).
     *        When true, the standard (non-fused) attention path is always used.
     *
     * @return Pair of (output, attention_weights)
     *         - output: (N, L, E) or (L, N, E) depending on batch_first
     *         - attention_weights: (N, num_heads, L, S) if need_weights=true, else empty
     *
     * @throws std::runtime_error if input shapes are incompatible
     * @throws std::runtime_error if mask shapes don't match expected dimensions
     *
     * Masking behavior:
     * - key_padding_mask: Masks out padding positions in the key sequence
     * - attn_mask: Custom attention pattern (e.g., causal mask for autoregressive models)
     * - Both masks are additive: -inf indicates positions to mask out
     *
     * @code
     * // Self-attention without masks
     * auto [out1, weights1] = attn.forward(x, x, x);
     *
     * // Cross-attention with padding mask
     * Tensor pad_mask({batch, src_len}, DType::Float32, Device::cpu());
     * auto [out2, weights2] = attn.forward(query, key, value, pad_mask);
     *
     * // Causal self-attention (for autoregressive models)
     * Tensor causal_mask = create_causal_mask(seq_len);
     * auto [out3, weights3] = attn.forward(x, x, x, Tensor{}, causal_mask);
     *
     * // Without attention weights (faster)
     * auto [out4, _] = attn.forward(x, x, x, Tensor{}, Tensor{}, false);
     * @endcode
     */
    auto forward(const Variable& query,
                const Variable& key,
                const Variable& value,
                const Tensor& key_padding_mask = Tensor{},
                const Tensor& attn_mask = Tensor{},
                bool need_weights = true) -> std::pair<Variable, Variable>;

    /**
     * @brief Default forward for Module interface (uses self-attention).
     *
     * This is a convenience method for self-attention only.
     * For cross-attention, use the full forward() signature.
     *
     * @param input Input variable for self-attention
     * @return Output variable (attention weights not returned)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Get embedding dimension.
     */
    auto embed_dim() const -> int64_t { return embed_dim_; }

    /**
     * @brief Get number of attention heads.
     */
    auto num_heads() const -> int64_t { return num_heads_; }

    /**
     * @brief Get dimension per head.
     */
    auto head_dim() const -> int64_t { return head_dim_; }

    /**
     * @brief Get whether causal masking is enabled.
     */
    auto is_causal() const -> bool { return is_causal_; }

private:
    int64_t embed_dim_;      ///< Total embedding dimension
    int64_t num_heads_;      ///< Number of attention heads
    int64_t head_dim_;       ///< Dimension per head (embed_dim / num_heads)
    int64_t kdim_;           ///< Key dimension
    int64_t vdim_;           ///< Value dimension
    double dropout_;         ///< Dropout probability
    bool batch_first_;       ///< Whether batch dimension is first
    bool add_zero_attn_;     ///< Whether to add zero attention
    bool is_causal_;         ///< Whether to apply causal masking in fused kernels

    // Projection layers
    std::shared_ptr<Linear> q_proj_;    ///< Query projection
    std::shared_ptr<Linear> k_proj_;    ///< Key projection
    std::shared_ptr<Linear> v_proj_;    ///< Value projection
    std::shared_ptr<Linear> out_proj_;  ///< Output projection

    std::shared_ptr<Dropout> dropout_layer_;  ///< Dropout on attention weights

    /**
     * @brief Compute scaled dot-product attention.
     *
     * Implements: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
     *
     * @param query Query tensor (N, num_heads, L, head_dim)
     * @param key Key tensor (N, num_heads, S, head_dim)
     * @param value Value tensor (N, num_heads, S, head_dim)
     * @param attn_mask Optional attention mask (L, S) or (N*num_heads, L, S)
     * @param dropout_p Dropout probability
     *
     * @return Pair of (attended_values, attention_weights)
     *         - attended_values: (N, num_heads, L, head_dim)
     *         - attention_weights: (N, num_heads, L, S)
     */
    auto scaled_dot_product_attention(const Variable& query,
                                     const Variable& key,
                                     const Variable& value,
                                     const Tensor& attn_mask = Tensor{},
                                     double dropout_p = 0.0,
                                     bool need_weights = false) const
        -> std::pair<Variable, Variable>;

    /**
     * @brief Transpose tensor for multi-head attention computation.
     *
     * Reshapes from (batch, seq_len, embed_dim) to (batch, num_heads, seq_len, head_dim)
     *
     * @param x Input variable
     * @return Transposed variable
     */
    auto transpose_for_scores(const Variable& x) const -> Variable;

    /**
     * @brief Merge multi-head outputs.
     *
     * Reshapes from (batch, num_heads, seq_len, head_dim) to (batch, seq_len, embed_dim)
     *
     * @param x Input variable
     * @return Merged variable
     */
    auto merge_heads(const Variable& x) const -> Variable;
};

/**
 * @brief Helper function to create causal attention mask.
 *
 * Creates a triangular mask that prevents attending to future positions.
 * Used for autoregressive models (e.g., GPT, language modeling).
 *
 * The mask is upper triangular with -inf above the diagonal:
 * @code
 *     [[  0, -inf, -inf],
 *      [  0,    0, -inf],
 *      [  0,    0,    0]]
 * @endcode
 *
 * @param seq_len Sequence length
 * @param device Device to create mask on
 * @return Causal mask tensor of shape (seq_len, seq_len)
 *
 * @code
 * Tensor mask = create_causal_mask(10, Device::cpu());
 * auto [output, _] = attn.forward(x, x, x, Tensor{}, mask);
 * @endcode
 */
auto create_causal_mask(int64_t seq_len, Device device = Device::cpu(), DType dtype = DType::Float32) -> Tensor;

} // namespace nn
} // namespace tenzor
