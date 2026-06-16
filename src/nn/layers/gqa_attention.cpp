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
#include "tenzor/nn/utils/variable_cast.hpp"
#include "attention_mask_utils.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>

namespace tenzor {
namespace nn {

// Namespace alias for autograd operations
namespace autograd = tenzor;

// Y.19 / EE.12: normalise a Bool or integer attention mask to a float additive
// mask (-inf where True/non-zero, 0 elsewhere). PyTorch's attention layers
// accept Bool masks where True = "ignore"; without this widening the downstream
// `scores + mask` adds 1.0 (Bool→float of true) at masked positions instead of
// -inf, leaking attention. Mirrors the helper in attention.cpp.

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
    // Then reshape to (batch, num_heads, seq_len, head_dim).
    //
    // Use autograd:: ops (not raw tenzor::) so the grad_fn chain is preserved
    // back through to k_proj_/v_proj_. The previous implementation called
    // raw Tensor ops on x.tensor() and rewrapped as Variable(reshaped, x.requires_grad())
    // — discarding the grad_fn. That zeroed K/V projection gradients for every
    // GQA model where num_heads_per_group_ > 1 (memory: feedback_raw_tensor_op_bug).
    Variable unsqueezed = autograd::unsqueeze(x, 2);
    std::vector<int64_t> expand_shape = {batch_size, n_kv_heads, num_heads_per_group_, seq_len, h_dim};
    Variable expanded = autograd::expand(unsqueezed, expand_shape);
    // reshape requires contiguous. autograd::reshape handles the contiguity
    // pass-through correctly while keeping the chain intact.
    std::vector<int64_t> final_shape = {batch_size, num_heads_, seq_len, h_dim};
    return autograd::reshape(expanded, final_shape);
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

    // For Float16/BFloat16, upcast Q/K/V to Float32 for the entire
    // bmm -> scale -> mask -> softmax -> bmm chain, then cast the attended
    // output and attention weights back to the original dtype before
    // returning. Half-precision softmax/scores overflow and lose precision;
    // running the chain in Float32 mirrors MultiheadAttention::
    // scaled_dot_product_attention and PyTorch's SDPA. All mask/scale tensors
    // below are built in `compute_dtype` so the additive masks stay in the
    // same (upcast) dtype as the scores.
    DType orig_dtype = query.dtype();
    bool needs_attn_upcast =
        (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    DType compute_dtype = needs_attn_upcast ? DType::Float32 : orig_dtype;
    Variable query_c = needs_attn_upcast ? variable_cast(query, DType::Float32) : query;
    Variable key_c   = needs_attn_upcast ? variable_cast(key,   DType::Float32) : key;
    Variable value_c = needs_attn_upcast ? variable_cast(value, DType::Float32) : value;

    // Compute scaling factor
    double scale = 1.0 / std::sqrt(static_cast<double>(h_dim));

    // Reshape to 3D for bmm: (batch*num_heads, seq_len, head_dim)
    auto query_3d = autograd::reshape(query_c, {batch_size * n_heads, seq_len_q, h_dim});
    auto key_3d = autograd::reshape(key_c, {batch_size * n_heads, seq_len_k, h_dim});
    auto value_3d = autograd::reshape(value_c, {batch_size * n_heads, seq_len_k, h_dim});

    // Transpose key for matmul
    auto key_transposed = autograd::permute(key_3d, {0, 2, 1});

    // QK^T
    auto scores = autograd::bmm(query_3d, key_transposed);

    // Reshape back to 4D for masking
    scores = autograd::reshape(scores, {batch_size, n_heads, seq_len_q, seq_len_k});

    // Scale
    Tensor scale_tensor = full({1}, static_cast<float>(scale), compute_dtype, query.device());
    Variable scale_var(scale_tensor, false);
    scores = scores * scale_var;

    // Apply causal mask if requested.
    //
    // Bottom-right aligned causal mask. Query position i (0-based) may attend
    // to KV positions j with  j <= i + offset,  where offset = seq_k - seq_q:
    //   * self-attention (seq_q == seq_k): offset = 0, standard lower-triangular.
    //   * KV-cache cross-attention (seq_k > seq_q): offset > 0, each query also
    //     attends to all cached tokens (the bottom-right diagonal).
    //   * seq_k < seq_q (offset < 0): the top queries have i + offset < 0 and
    //     would attend to NO key — a fully-masked row whose softmax over all
    //     -inf is NaN. We clamp the per-row allowed boundary to at least key 0
    //     (boundary b_i = max(i + offset, 0)), so every query attends to at
    //     least its aligned key. This keeps the mask well-defined and NaN-free
    //     for arbitrary seq_q/seq_k.
    //
    // Built from explicit row/col index tensors (rather than triu with a
    // diagonal offset that underflows when seq_k < seq_q) so the alignment and
    // the clamp are exact.
    if (is_causal_) {
        const int64_t offset = seq_len_k - seq_len_q;
        // row_idx[i,j] = i, col_idx[i,j] = j  (Int64 broadcast grids)
        Tensor row_idx = tenzor::arange(0, seq_len_q, 1, DType::Int64, query.device())
                             .reshape({seq_len_q, 1});
        Tensor col_idx = tenzor::arange(0, seq_len_k, 1, DType::Int64, query.device())
                             .reshape({1, seq_len_k});
        Tensor row_exp = expand(row_idx, std::vector<int64_t>{seq_len_q, seq_len_k});
        Tensor col_exp = expand(col_idx, std::vector<int64_t>{seq_len_q, seq_len_k});
        // Per-row allowed boundary b_i = max(i + offset, 0): queries may attend
        // to columns j <= b_i. The clamp_min(0) is what prevents a fully-masked
        // row when offset < 0 (seq_k < seq_q) — without it the top queries would
        // have i + offset < 0 and mask every column, giving softmax(all -inf) =
        // NaN. An upper clamp to seq_k-1 is unnecessary: col is always < seq_k,
        // so `col > b_i` already handles the upper side.
        Tensor boundary = tenzor::clamp_min(
            tenzor::add(row_exp, static_cast<double>(offset)), 0.0);
        // masked_off: true where col > boundary (future / disallowed key).
        Tensor masked_off = gt(col_exp, boundary);
        Tensor neg_inf = full({seq_len_q, seq_len_k},
                              -std::numeric_limits<float>::infinity(),
                              compute_dtype, query.device());
        Tensor zero_mask = zeros({seq_len_q, seq_len_k},
                                 compute_dtype, query.device());
        Tensor causal = where(masked_off, neg_inf, zero_mask);
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
        // The mask is built in compute_dtype (Float32 when Q/K/V were upcast),
        // matching the scores' dtype for the additive combine. -1e9 overflows
        // to -inf in Float16/BFloat16 mask tensors and a fully-masked row then
        // softmaxes to NaN; use the -1e4 sentinel only when compute_dtype is
        // actually half (i.e. no upcast happened), true -inf otherwise.
        double mask_fill =
            (compute_dtype == DType::Float16 || compute_dtype == DType::BFloat16)
                ? -1e4
                : -std::numeric_limits<double>::infinity();
        auto neg_large = full(std::vector<int64_t>(dist.shape().begin(), dist.shape().end()),
                              mask_fill, compute_dtype, query.device());
        auto zero_mask = zeros(std::vector<int64_t>(dist.shape().begin(), dist.shape().end()),
                               compute_dtype, query.device());
        auto window_mask = where(outside, neg_large, zero_mask);
        // window_mask is [seq_len_q, seq_len_k], broadcasts to [batch, heads, L, S]
        Variable window_var(window_mask, false);
        scores = scores + window_var;
    }

    // Apply explicit attention mask if provided
    if (attn_mask.is_valid() && attn_mask.shape().size() > 0) {
        // Y.19 / EE.12: Bool/integer masks (PyTorch convention: True = ignore)
        // must be widened to a float -inf/0 mask before the additive combine.
        // Without this, `float scores + Bool mask` adds 1.0 at masked positions
        // instead of -inf, leaking attention. Mirrors V.28/AA.1/EE.11.
        Tensor normalised_mask = normalize_attn_mask(attn_mask);
        // Match the (possibly upcast) score dtype so the additive combine stays
        // in compute_dtype; a half float mask added to Float32 scores would
        // otherwise mismatch.
        if (normalised_mask.dtype() != compute_dtype) {
            normalised_mask = normalised_mask.to(compute_dtype);
        }
        Variable mask_var(normalised_mask, false);
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

    // Cast the attended output and attention weights back to the caller's
    // original dtype (Float16/BFloat16) after running the chain in Float32.
    if (needs_attn_upcast) {
        attended = variable_cast(attended, orig_dtype);
        attn_weights = variable_cast(attn_weights, orig_dtype);
    }

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
