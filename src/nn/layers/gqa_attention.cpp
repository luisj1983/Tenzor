/**
 * @file gqa_attention.cpp
 * @brief Implementation of Grouped Query Attention (GQA / MQA)
 */

#include "tenzor/nn/layers/gqa_attention.hpp"
#include "tenzor/nn/layers/attention.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/autograd/ops.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>

namespace tenzor {
namespace nn {

// Namespace alias for autograd operations
namespace autograd = tenzor;

// ============================================================================
// GroupedQueryAttention Implementation
// ============================================================================

GroupedQueryAttention::GroupedQueryAttention(int64_t embed_dim,
                                           int64_t num_heads,
                                           int64_t num_kv_heads,
                                           double dropout,
                                           bool bias,
                                           bool is_causal,
                                           std::shared_ptr<RoPE> rope,
                                           int64_t window_size)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      dropout_(dropout),
      is_causal_(is_causal),
      rope_(std::move(rope)),
      window_size_(window_size) {

    // Validate parameters
    if (embed_dim_ % num_heads_ != 0) {
        throw std::invalid_argument(
            "embed_dim must be divisible by num_heads. Got embed_dim=" +
            std::to_string(embed_dim_) + ", num_heads=" + std::to_string(num_heads_));
    }

    if (num_heads_ % num_kv_heads_ != 0) {
        throw std::invalid_argument(
            "num_heads must be divisible by num_kv_heads. Got num_heads=" +
            std::to_string(num_heads_) + ", num_kv_heads=" + std::to_string(num_kv_heads_));
    }

    if (dropout_ < 0.0 || dropout_ > 1.0) {
        throw std::invalid_argument(
            "dropout probability must be in [0, 1]. Got " + std::to_string(dropout_));
    }

    head_dim_ = embed_dim_ / num_heads_;
    num_heads_per_group_ = num_heads_ / num_kv_heads_;

    int64_t kv_dim = num_kv_heads_ * head_dim_;

    // Create projection layers
    // Q projects to full embed_dim (num_heads * head_dim)
    q_proj_ = std::make_shared<Linear>(embed_dim_, embed_dim_, bias);
    // K/V project to reduced dim (num_kv_heads * head_dim)
    k_proj_ = std::make_shared<Linear>(embed_dim_, kv_dim, bias);
    v_proj_ = std::make_shared<Linear>(embed_dim_, kv_dim, bias);
    out_proj_ = std::make_shared<Linear>(embed_dim_, embed_dim_, bias);

    // Create dropout layer
    dropout_layer_ = std::make_shared<Dropout>(dropout_);

    // Register submodules
    register_module("q_proj", q_proj_);
    register_module("k_proj", k_proj_);
    register_module("v_proj", v_proj_);
    register_module("out_proj", out_proj_);
    register_module("dropout", dropout_layer_);

    if (rope_) {
        register_module("rope", rope_);
    }
}

auto GroupedQueryAttention::repeat_kv(const Variable& x) const -> Variable {
    // Input: (batch, num_kv_heads, seq_len, head_dim)
    // Output: (batch, num_heads, seq_len, head_dim)

    if (num_heads_per_group_ == 1) {
        // No repetition needed (MHA case)
        return x;
    }

    auto shape = x.shape();
    int64_t batch_size = shape[0];
    int64_t n_kv_heads = shape[1];
    int64_t seq_len = shape[2];
    int64_t h_dim = shape[3];

    // Unsqueeze to (batch, num_kv_heads, 1, seq_len, head_dim)
    // Then expand to (batch, num_kv_heads, num_heads_per_group, seq_len, head_dim)
    // Then reshape to (batch, num_heads, seq_len, head_dim)
    Tensor unsqueezed = tenzor::unsqueeze(x.tensor(), 2);
    std::vector<int64_t> expand_shape = {batch_size, n_kv_heads, num_heads_per_group_, seq_len, h_dim};
    Tensor expanded = tenzor::expand(unsqueezed, expand_shape);
    // Make contiguous after expand (expand creates a view with stride=0)
    Tensor expanded_contig = expanded.is_contiguous() ? expanded : expanded.contiguous();
    std::vector<int64_t> final_shape = {batch_size, num_heads_, seq_len, h_dim};
    Tensor reshaped = tenzor::reshape(expanded_contig, final_shape);

    return Variable(reshaped, x.requires_grad());
}

auto GroupedQueryAttention::scaled_dot_product_attention(
    const Variable& query,
    const Variable& key,
    const Variable& value,
    const Tensor& attn_mask,
    double dropout_p) const -> std::pair<Variable, Variable> {

    // Query: (batch, num_heads, seq_len_q, head_dim)
    // Key:   (batch, num_heads, seq_len_k, head_dim)
    // Value: (batch, num_heads, seq_len_k, head_dim)

    auto q_shape = query.shape();
    int64_t batch_size = q_shape[0];
    int64_t n_heads = q_shape[1];
    int64_t seq_len_q = q_shape[2];
    int64_t h_dim = q_shape[3];

    auto k_shape = key.shape();
    int64_t seq_len_k = k_shape[2];

    // Compute scaling factor
    double scale = 1.0 / std::sqrt(static_cast<double>(h_dim));

    // Reshape to 3D for bmm: (batch*num_heads, seq_len, head_dim)
    auto query_3d = autograd::reshape(query, {batch_size * n_heads, seq_len_q, h_dim});
    auto key_3d = autograd::reshape(key, {batch_size * n_heads, seq_len_k, h_dim});
    auto value_3d = autograd::reshape(value, {batch_size * n_heads, seq_len_k, h_dim});

    // Transpose key for matmul
    auto key_transposed = autograd::permute(key_3d, {0, 2, 1});

    // QK^T
    auto scores = autograd::bmm(query_3d, key_transposed);

    // Reshape back to 4D for masking
    scores = autograd::reshape(scores, {batch_size, n_heads, seq_len_q, seq_len_k});

    // Scale
    Tensor scale_tensor = full({1}, static_cast<float>(scale), query.dtype(), query.device());
    Variable scale_var(scale_tensor, false);
    scores = scores * scale_var;

    // Apply causal mask if requested
    if (is_causal_) {
        Tensor causal = create_causal_mask(seq_len_q, query.device(), query.dtype());
        // Slice if seq_len_k != seq_len_q (cross-attention case)
        if (seq_len_k != seq_len_q) {
            causal = tenzor::slice(causal, 1, seq_len_k - seq_len_q, seq_len_k);
        }
        Variable mask_var(causal, false);
        scores = scores + mask_var;
    }

    // Apply sliding window mask if window_size > 0
    if (window_size_ > 0) {
        // Create band mask: positions where |i - j| > window_size/2 get -inf
        auto row_idx = tenzor::arange(0, seq_len_q, 1, DType::Int64, query.device())
                       .reshape({seq_len_q, 1});
        auto col_idx = tenzor::arange(0, seq_len_k, 1, DType::Int64, query.device())
                       .reshape({1, seq_len_k});
        auto dist = tenzor::abs(row_idx.to(DType::Float32) - col_idx.to(DType::Float32));
        float half_window = static_cast<float>(window_size_ / 2);
        // Use `where` to avoid the 0 * (-1e9 in Float16) = 0 * -inf = NaN
        // pattern that the previous boolean-cast-multiply produced. For
        // Float16, -1e9 overflows to -inf, so the mask had NaN at
        // positions where outside=False (0 * -inf = NaN). where() picks
        // the right operand directly without arithmetic on the mask.
        auto threshold = tenzor::full(std::vector<int64_t>(dist.shape().begin(), dist.shape().end()),
                                     static_cast<double>(half_window), DType::Float32, query.device());
        auto outside = tenzor::gt(dist, threshold);
        auto neg_large = full(std::vector<int64_t>(dist.shape().begin(), dist.shape().end()),
                              -1e9, query.dtype(), query.device());
        auto zero_mask = zeros(std::vector<int64_t>(dist.shape().begin(), dist.shape().end()),
                               query.dtype(), query.device());
        auto window_mask = where(outside, neg_large, zero_mask);
        // window_mask is [seq_len_q, seq_len_k], broadcasts to [batch, heads, L, S]
        Variable window_var(window_mask, false);
        scores = scores + window_var;
    }

    // Apply explicit attention mask if provided
    if (attn_mask.is_valid() && attn_mask.shape().size() > 0) {
        Variable mask_var(attn_mask, false);
        scores = scores + mask_var;
    }

    // Softmax
    Variable attn_weights = autograd::softmax(scores, -1);

    // Dropout
    if (dropout_p > 0.0 && is_training()) {
        attn_weights = dropout_layer_->forward(attn_weights);
    }

    // Reshape for bmm
    auto attn_weights_3d = autograd::reshape(attn_weights, {batch_size * n_heads, seq_len_q, seq_len_k});

    // Weighted sum
    auto attended_3d = autograd::bmm(attn_weights_3d, value_3d);

    // Reshape back to 4D
    auto attended = autograd::reshape(attended_3d, {batch_size, n_heads, seq_len_q, h_dim});

    return {attended, attn_weights};
}

auto GroupedQueryAttention::forward(const Variable& query,
                                   const Variable& key,
                                   const Variable& value,
                                   const Tensor& attn_mask,
                                   bool need_weights) -> std::pair<Variable, Variable> {
    // Call forward pre-hooks
    call_forward_pre_hooks();

    auto q_shape = query.shape();
    int64_t batch_size = q_shape[0];
    int64_t seq_len_q = q_shape[1];

    auto k_shape = key.shape();
    int64_t seq_len_k = k_shape[1];

    // Project Q, K, V
    Variable Q = q_proj_->forward(query);
    Variable K = k_proj_->forward(key);
    Variable V = v_proj_->forward(value);

    // Reshape Q: (batch, seq, embed_dim) -> (batch, num_heads, seq, head_dim)
    Q = autograd::reshape(Q, {batch_size, seq_len_q, num_heads_, head_dim_});
    Q = autograd::permute(Q, {0, 2, 1, 3});

    // Reshape K: (batch, seq, kv_dim) -> (batch, num_kv_heads, seq, head_dim)
    K = autograd::reshape(K, {batch_size, seq_len_k, num_kv_heads_, head_dim_});
    K = autograd::permute(K, {0, 2, 1, 3});

    // Reshape V: same as K
    V = autograd::reshape(V, {batch_size, seq_len_k, num_kv_heads_, head_dim_});
    V = autograd::permute(V, {0, 2, 1, 3});

    // Apply RoPE if configured
    if (rope_) {
        // RoPE expects (..., seq_len, head_dim) with position offset.
        // Q is (batch, num_heads, seq_len_q, head_dim), K is (batch, num_kv_heads, seq_len_k, head_dim).
        // Previously wrapped the output in Variable(result.tensor(), requires_grad)
        // which discarded RoPE's grad_fn and prevented gradients from flowing
        // through the rotary embedding — rope_->forward returns a Variable
        // that already carries the correct grad_fn, so use it directly.
        Q = rope_->forward(Q, 0);
        K = rope_->forward(K, 0);
    }

    // Repeat KV heads to match query head count
    K = repeat_kv(K);  // (batch, num_heads, seq_len_k, head_dim)
    V = repeat_kv(V);  // (batch, num_heads, seq_len_k, head_dim)

    // Compute attention
    auto [attended, attn_weights] = scaled_dot_product_attention(Q, K, V, attn_mask, dropout_);

    // Merge heads: (batch, num_heads, seq_len_q, head_dim) -> (batch, seq_len_q, embed_dim)
    auto output = autograd::permute(attended, {0, 2, 1, 3});
    output = autograd::reshape(output, {batch_size, seq_len_q, embed_dim_});

    // Output projection
    output = out_proj_->forward(output);

    // Call forward post-hooks
    call_forward_post_hooks();

    if (need_weights) {
        return {output, attn_weights};
    } else {
        Tensor empty_tensor = zeros({}, DType::Float32, output.device());
        Variable empty_var(empty_tensor, false);
        return {output, empty_var};
    }
}

auto GroupedQueryAttention::forward_impl(const Variable& input) -> Variable {
    // Self-attention
    auto [output, _] = forward(input, input, input, Tensor{}, false);
    return output;
}

} // namespace nn
} // namespace tenzor
