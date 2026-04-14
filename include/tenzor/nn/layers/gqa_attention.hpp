/**
 * @file gqa_attention.hpp
 * @brief Grouped Query Attention (GQA / MQA) for transformer models
 *
 * Implements Grouped Query Attention as described in "GQA: Training Generalized
 * Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al., 2023).
 *
 * GQA generalizes Multi-Head Attention (MHA) and Multi-Query Attention (MQA)
 * by allowing a configurable number of key/value heads that is less than the
 * number of query heads. K/V heads are repeated via expand() to match query heads.
 *
 * Special cases:
 * - num_kv_heads == num_heads: Standard Multi-Head Attention (MHA)
 * - num_kv_heads == 1: Multi-Query Attention (MQA)
 * - 1 < num_kv_heads < num_heads: Grouped Query Attention (GQA)
 */

#pragma once

#include <memory>
#include <optional>
#include "../module.hpp"
#include "linear.hpp"
#include "dropout.hpp"
#include "rope.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Grouped Query Attention (GQA / MQA) layer.
 *
 * Implements grouped query attention where K/V heads are shared across groups
 * of query heads. This reduces the KV cache size and computation while
 * maintaining most of the quality of standard multi-head attention.
 *
 * Shape transformations:
 * - Query: (N, L, embed_dim)
 * - Key:   (N, S, embed_dim)
 * - Value: (N, S, embed_dim)
 * - Output: (N, L, embed_dim)
 *
 * Where:
 * - N = batch size
 * - L = target sequence length
 * - S = source sequence length
 * - embed_dim = total embedding dimension
 *
 * @code
 * // GQA with 32 query heads and 8 KV heads (4 groups)
 * GroupedQueryAttention gqa(4096, 32, 8);
 *
 * // MQA with 1 KV head
 * GroupedQueryAttention mqa(4096, 32, 1);
 *
 * // GQA with RoPE integration
 * auto rope = std::make_shared<RoPE>(128, 4096);
 * GroupedQueryAttention gqa_rope(4096, 32, 8, 0.0, true, false, rope);
 * @endcode
 *
 * @see MultiheadAttention for standard multi-head attention
 */
class GroupedQueryAttention : public Module {
public:
    // Bring base class forward into scope
    using Module::forward;

    /**
     * @brief Construct grouped query attention layer.
     *
     * @param embed_dim Total dimension of the model
     * @param num_heads Number of query attention heads
     * @param num_kv_heads Number of key/value heads (must divide num_heads evenly)
     * @param dropout Dropout probability on attention weights (default: 0.0)
     * @param bias If true, add bias to input/output projections (default: true)
     * @param is_causal If true, applies causal masking (default: false)
     * @param rope Optional RoPE module for rotary position embeddings
     *
     * @throws std::invalid_argument if embed_dim not divisible by num_heads
     * @throws std::invalid_argument if num_heads not divisible by num_kv_heads
     * @throws std::invalid_argument if dropout not in [0, 1]
     */
    GroupedQueryAttention(int64_t embed_dim,
                         int64_t num_heads,
                         int64_t num_kv_heads,
                         double dropout = 0.0,
                         bool bias = true,
                         bool is_causal = false,
                         std::shared_ptr<RoPE> rope = nullptr,
                         int64_t window_size = -1);

    /**
     * @brief Forward pass through grouped query attention.
     *
     * @param query Query tensor of shape (N, L, embed_dim)
     * @param key Key tensor of shape (N, S, embed_dim)
     * @param value Value tensor of shape (N, S, embed_dim)
     * @param attn_mask Optional attention mask (L, S) or (N*num_heads, L, S)
     * @param need_weights If true, return attention weights (default: false)
     *
     * @return Pair of (output, attention_weights)
     *         - output: (N, L, embed_dim)
     *         - attention_weights: (N, num_heads, L, S) if need_weights=true, else empty
     */
    auto forward(const Variable& query,
                const Variable& key,
                const Variable& value,
                const Tensor& attn_mask = Tensor{},
                bool need_weights = false) -> std::pair<Variable, Variable>;

    /**
     * @brief Default forward for Module interface (self-attention).
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /** @brief Get embedding dimension. */
    auto embed_dim() const -> int64_t { return embed_dim_; }

    /** @brief Get number of query heads. */
    auto num_heads() const -> int64_t { return num_heads_; }

    /** @brief Get number of key/value heads. */
    auto num_kv_heads() const -> int64_t { return num_kv_heads_; }

    /** @brief Get number of query heads per KV head group. */
    auto num_heads_per_group() const -> int64_t { return num_heads_per_group_; }

    /** @brief Get dimension per head. */
    auto head_dim() const -> int64_t { return head_dim_; }

    /** @brief Get whether causal masking is enabled. */
    auto is_causal() const -> bool { return is_causal_; }

    /** @brief Get sliding window size (-1 means full attention). */
    auto window_size() const -> int64_t { return window_size_; }

private:
    int64_t embed_dim_;           ///< Total embedding dimension
    int64_t num_heads_;           ///< Number of query heads
    int64_t num_kv_heads_;        ///< Number of key/value heads
    int64_t num_heads_per_group_; ///< Query heads per KV group
    int64_t head_dim_;            ///< Dimension per head (embed_dim / num_heads)
    double dropout_;              ///< Dropout probability
    bool is_causal_;              ///< Whether to apply causal masking
    int64_t window_size_;         ///< Sliding window size (-1 = full attention)

    // Projection layers
    std::shared_ptr<Linear> q_proj_;    ///< Query projection: embed_dim -> embed_dim
    std::shared_ptr<Linear> k_proj_;    ///< Key projection: embed_dim -> num_kv_heads * head_dim
    std::shared_ptr<Linear> v_proj_;    ///< Value projection: embed_dim -> num_kv_heads * head_dim
    std::shared_ptr<Linear> out_proj_;  ///< Output projection: embed_dim -> embed_dim

    std::shared_ptr<Dropout> dropout_layer_;  ///< Dropout on attention weights
    std::shared_ptr<RoPE> rope_;              ///< Optional rotary position embeddings

    /**
     * @brief Repeat KV heads to match query head count via expand.
     *
     * Takes (batch, num_kv_heads, seq_len, head_dim) and returns
     * (batch, num_heads, seq_len, head_dim) by expanding groups.
     */
    auto repeat_kv(const Variable& x) const -> Variable;

    /**
     * @brief Compute scaled dot-product attention.
     */
    auto scaled_dot_product_attention(const Variable& query,
                                     const Variable& key,
                                     const Variable& value,
                                     const Tensor& attn_mask = Tensor{},
                                     double dropout_p = 0.0) const
        -> std::pair<Variable, Variable>;
};

} // namespace nn
} // namespace tenzor
