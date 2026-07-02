/**
 * @file attention.cpp
 * @brief Implementation of multi-head attention mechanisms
 */

#include "tenzor/nn/layers/attention.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include "attention_mask_utils.hpp"
#include "tenzor/nn/utils/variable_cast.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>

// Include fused attention kernel for CPU optimization
// Note: cpu::attention::scaled_dot_product_attention (formerly in
// src/backends/cpu/kernels/fused_attention.hpp) was deleted as part of the
// attention contract consolidation — see docs/internals/attention-contract.md
// and audit S1. All CPU attention dispatch now flows through OpId::FusedAttention
// or OpId::FlashAttention via the autograd helpers in include/tenzor/autograd/ops.hpp.

// Include dispatch for fused attention
#include "tenzor/backend/fast_dispatch.hpp"

// Include cuDNN SDPA for optimized CUDA attention
#include "tenzor/backend/fused_ops.hpp"

namespace tenzor {
namespace nn {

// Namespace alias for autograd operations
namespace autograd = tenzor;


// ============================================================================
// MultiheadAttention Implementation
// ============================================================================

MultiheadAttention::MultiheadAttention(int64_t embed_dim,
                                     int64_t num_heads,
                                     double dropout,
                                     bool bias,
                                     bool add_bias_kv,
                                     bool add_zero_attn,
                                     int64_t kdim,
                                     int64_t vdim,
                                     bool batch_first,
                                     bool is_causal)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      kdim_(kdim > 0 ? kdim : embed_dim),
      vdim_(vdim > 0 ? vdim : embed_dim),
      dropout_(dropout),
      batch_first_(batch_first),
      add_zero_attn_(add_zero_attn),
      is_causal_(is_causal) {

    // Validate parameters
    if (embed_dim_ <= 0) {
        throw std::invalid_argument(
            "embed_dim must be positive. Got embed_dim=" + std::to_string(embed_dim_));
    }
    if (num_heads_ <= 0) {
        throw std::invalid_argument(
            "num_heads must be positive. Got num_heads=" + std::to_string(num_heads_));
    }
    if (embed_dim_ % num_heads_ != 0) {
        throw std::invalid_argument(
            "embed_dim must be divisible by num_heads. Got embed_dim=" +
            std::to_string(embed_dim_) + ", num_heads=" + std::to_string(num_heads_));
    }

    if (dropout_ < 0.0 || dropout_ > 1.0) {
        throw std::invalid_argument(
            "dropout probability must be in [0, 1]. Got " + std::to_string(dropout_));
    }

    add_bias_kv_ = add_bias_kv;

    if (add_bias_kv) {
        // Initialize bias_k and bias_v as learnable parameters [1, 1, embed_dim].
        // Stored at construction-time dtype (Float32); Module::to(DType) will
        // cast in-place via the registered_parameters map, which the local
        // shared_ptr members alias.
        register_parameter("bias_k", Variable(tenzor::zeros({1, 1, embed_dim_}, DType::Float32), true));
        register_parameter("bias_v", Variable(tenzor::zeros({1, 1, embed_dim_}, DType::Float32), true));
        bias_k_ = get_parameter("bias_k");
        bias_v_ = get_parameter("bias_v");
    }

    head_dim_ = embed_dim_ / num_heads_;

    // Create projection layers
    q_proj_ = std::make_shared<Linear>(embed_dim_, embed_dim_, bias);
    k_proj_ = std::make_shared<Linear>(kdim_, embed_dim_, bias);
    v_proj_ = std::make_shared<Linear>(vdim_, embed_dim_, bias);
    out_proj_ = std::make_shared<Linear>(embed_dim_, embed_dim_, bias);

    // Create dropout layer
    dropout_layer_ = std::make_shared<Dropout>(dropout_);

    // Register submodules
    register_module("q_proj", q_proj_);
    register_module("k_proj", k_proj_);
    register_module("v_proj", v_proj_);
    register_module("out_proj", out_proj_);
    register_module("dropout", dropout_layer_);
}

auto MultiheadAttention::transpose_for_scores(const Variable& x) const -> Variable {
    // Input: (batch, seq_len, embed_dim)
    // Output: (batch, num_heads, seq_len, head_dim)

    auto shape = x.shape();
    int64_t batch_size = shape[0];
    int64_t seq_len = shape[1];

    // Reshape to (batch, seq_len, num_heads, head_dim)
    std::vector<int64_t> new_shape = {batch_size, seq_len, num_heads_, head_dim_};
    auto reshaped = autograd::reshape(x, new_shape);

    // Permute to (batch, num_heads, seq_len, head_dim)
    std::vector<int64_t> perm = {0, 2, 1, 3};
    auto result = autograd::permute(reshaped, perm);

    return result;
}

auto MultiheadAttention::merge_heads(const Variable& x) const -> Variable {
    // Input: (batch, num_heads, seq_len, head_dim)
    // Output: (batch, seq_len, embed_dim)

    auto shape = x.shape();
    int64_t batch_size = shape[0];
    int64_t seq_len = shape[2];

    // Permute to (batch, seq_len, num_heads, head_dim)
    std::vector<int64_t> perm = {0, 2, 1, 3};
    auto permuted = autograd::permute(x, perm);

    // Reshape to (batch, seq_len, embed_dim)
    std::vector<int64_t> new_shape = {batch_size, seq_len, embed_dim_};
    auto result = autograd::reshape(permuted, new_shape);

    return result;
}

auto MultiheadAttention::scaled_dot_product_attention(
    const Variable& query,
    const Variable& key,
    const Variable& value,
    const Tensor& attn_mask,
    double dropout_p,
    bool need_weights,
    const Tensor& position_bias) const -> std::pair<Variable, Variable> {

    // Query: (batch, num_heads, seq_len_q, head_dim)
    // Key: (batch, num_heads, seq_len_k, head_dim)
    // Value: (batch, num_heads, seq_len_k, head_dim)

    // NOTE: The is_causal / explicit-attn_mask mutual-exclusion check lives in
    // MultiheadAttention::forward(), gated on the USER-supplied attn_mask
    // argument. It cannot live here: `attn_mask` at this level is the COMBINED
    // mask, which legitimately carries a folded key_padding_mask alongside
    // is_causal. The manual BMM path below applies the causal triangular mask
    // and this additive mask together (both additive), so a padding-only mask
    // coexists correctly with causal masking.
    auto q_shape = query.shape();
    int64_t batch_size = q_shape[0];
    int64_t num_heads = q_shape[1];
    int64_t seq_len_q = q_shape[2];
    int64_t head_dim = q_shape[3];

    auto k_shape = key.shape();
    int64_t seq_len_k = k_shape[2];

    // Use fused CPU kernel for inference when conditions allow:
    // - CPU device, Float32, no mask (or is_causal handles masking), no dropout (or eval mode)
    // - need_weights must be false (fused path doesn't compute attention weights)
    //
    // Previously this path called cpu::attention::scaled_dot_product_attention
    // directly from src/backends/cpu/kernels/fused_attention.hpp — bypassing
    // dispatch. That created two parallel CPU attention implementations
    // (audit S1) with different mask sentinels (-1e9 in the header version vs
    // -INFINITY in the dispatch version). Routing through the dispatch
    // (OpId::FusedAttention) ensures one source of truth and lets the autograd
    // Function attach a grad_fn when needed.
    bool can_use_fused_cpu = !need_weights &&
                             query.device().type == Device::Type::CPU &&
                             query.dtype() == DType::Float32 &&
                             (is_causal_ || !attn_mask.is_valid() || attn_mask.shape().size() == 0) &&
                             // The fused kernel has no additive-bias input; fall through to the
                             // manual BMM path (which applies position_bias) when one is supplied,
                             // mirroring the cuDNN SDPA guard below.
                             !(position_bias.is_valid() && position_bias.shape().size() > 0) &&
                             (dropout_p <= 0.0 || !is_training());

    if (can_use_fused_cpu && !is_training()) {
        try {
            float scale_f = 1.0f / std::sqrt(static_cast<float>(head_dim));
            // Route through the autograd FusedAttention Function so the
            // resulting Variable carries a grad_fn when any input needs grad
            // (eval-with-grad). For pure inference the apply helper skips
            // the Function allocation and returns a no-grad Variable.
            Variable attended = ::tenzor::fused_attention(
                query, key, value, scale_f, /*causal=*/is_causal_,
                /*use_cudnn_sdpa=*/false);
            Tensor empty_weights({0}, attended.tensor().dtype(), attended.tensor().device());
            return {attended, Variable(empty_weights, false)};
        } catch (const std::exception&) {
            // Fall through to the manual path below if dispatch fails.
        }
    }

    // CUDA Fast path: cuDNN SDPA with Flash Attention and Tensor Cores
    // Uses cuDNN Graph API for fused attention via dispatch.
    // Requirements: cuDNN 9.0+, Ampere+ GPU, head_dim in {32, 64, 128, 256},
    // FP16 / BF16 / FP32 input. cuDNN SDPA accumulates in FP32 regardless of
    // I/O dtype, so the FP32 path runs the same fused kernel as FP16/BF16
    // (just with more bandwidth). If cuDNN can't handle the configuration on
    // the current device (e.g. a brand new GPU arch with limited SDPA tuning)
    // the kernel internally falls back to the custom flash kernel via the
    // capability cache in cudnn_sdpa.cpp — see SDPACapCache.
    // Note: Only enabled in inference mode — cuDNN SDPA doesn't build the
    // autograd graph. need_weights must be false (no attention weights out).
    bool dtype_supported = query.dtype() == DType::Float16 ||
                           query.dtype() == DType::BFloat16 ||
                           query.dtype() == DType::Float32;
    // ROCm's fused_attention_hip kernel doesn't accept a causal flag (#46) — if
    // is_causal is requested, fall through to the manual BMM path which builds
    // an explicit triu mask. CUDA cuDNN SDPA handles causal natively.
    bool device_supports_causal = (query.device().type != Device::Type::ROCm) || !is_causal_;
    // Only take the cuDNN SDPA fast path when no gradient is needed downstream.
    // This dispatch returns a Variable without a grad_fn, so any caller that
    // backwards through it would get zero gradients (audit C1, H1). The manual
    // BMM path below is autograd-correct. Eval mode without grad-enabled
    // (the common inference case) still gets the fast cuDNN call.
    bool any_input_needs_grad = query.requires_grad() || key.requires_grad() || value.requires_grad();
    bool grad_path_safe = !is_grad_enabled() && !any_input_needs_grad;
    // The cuDNN SDPA dispatch below passes only {q,k,v} — it never forwards
    // attn_mask/key_padding_mask or position_bias to the fused kernel. So if
    // either is present, this fast path would silently drop it and produce an
    // unmasked / unbiased result. Exclude those calls here (mirroring the
    // fused-CPU gate at ~202 and the flash gate at ~324, which carve themselves
    // out when a mask is present) so they fall through to the manual BMM path
    // below, which applies position_bias (~413) and attn_mask (~441). A
    // user-supplied attn_mask is rejected together with is_causal in forward();
    // a folded padding-only mask may still reach here with is_causal, so gating
    // this fast path on mask/bias presence (below) keeps it correct.
    bool has_attn_mask = attn_mask.is_valid() && attn_mask.shape().size() > 0;
    bool has_position_bias = position_bias.is_valid() && position_bias.shape().size() > 0;
    bool can_use_cudnn_sdpa = !need_weights &&
        dtype_supported &&
        is_op_supported(OpId::FusedAttention, query.device().type) &&
        (head_dim == 32 || head_dim == 64 || head_dim == 128 || head_dim == 256) &&
        device_supports_causal &&
        !has_attn_mask &&     // fused dispatch can't apply an additive mask
        !has_position_bias && // fused dispatch can't apply position bias
        !is_training() &&     // dropout/training BMM-only
        grad_path_safe;       // gradient correctness

    if (can_use_cudnn_sdpa) {
        try {
            float scale_f = 1.0f / std::sqrt(static_cast<float>(head_dim));

            // Make tensors contiguous for cuDNN
            Tensor q_contig = query.tensor().is_contiguous() ? query.tensor() : query.tensor().contiguous();
            Tensor k_contig = key.tensor().is_contiguous() ? key.tensor() : key.tensor().contiguous();
            Tensor v_contig = value.tensor().is_contiguous() ? value.tensor() : value.tensor().contiguous();

            // ROCm's fused_attention_hip kernel expects 3D [B*H, L, E]; CUDA's
            // cuDNN SDPA accepts 4D directly. Collapse batch+heads on ROCm so
            // it doesn't reinterpret shape[0]/shape[1] as B/H. Output reshaped
            // back to 4D after dispatch. Without this the kernel returns shape
            // [B, H, L] (E dropped), causing merge_heads permute to throw.
            bool needs_3d_collapse = (query.device().type == Device::Type::ROCm);
            if (needs_3d_collapse) {
                q_contig = tenzor::reshape(q_contig, {batch_size * num_heads, seq_len_q, head_dim});
                k_contig = tenzor::reshape(k_contig, {batch_size * num_heads, seq_len_k, head_dim});
                v_contig = tenzor::reshape(v_contig, {batch_size * num_heads, seq_len_k, head_dim});
            }

            // Use dispatch to call cuDNN SDPA - pass 4D tensors directly.
            // Per docs/internals/attention-contract.md, Causal must be set so the
            // backend (cuDNN graph or custom kernel) applies the mask before
            // softmax. Previously this path silently dropped is_causal_ on
            // CUDA, producing a non-causal output for any caller that asked
            // for causal MHA in eval mode.
            OpAttributes attrs;
            attrs.set(AttrKey::Scale, static_cast<double>(scale_f));
            attrs.set(AttrKey::Causal, is_causal_);
            attrs.set(AttrKey::UseCudnnSdpa, true);
            std::vector<Tensor> fused_inputs = {q_contig, k_contig, v_contig};
            Tensor output = dispatch<OpId::FusedAttention>(fused_inputs, attrs)[0];

            if (needs_3d_collapse) {
                output = tenzor::reshape(output, {batch_size, num_heads, seq_len_q, head_dim});
            }

            Variable attended(output, false);
            Tensor empty_weights({0}, output.dtype(), output.device());
            Variable attn_weights_empty(empty_weights, false);

            return {attended, attn_weights_empty};
        } catch (const std::exception& e) {
            // Fall through to BMM path if cuDNN SDPA fails
            // This can happen if GPU doesn't support SDPA or other issues
        }
    }

    // CPU Flash Attention: O(N) memory via tiled online softmax with fused dropout
    // Uses OpId::FlashAttention for 4D [batch, num_heads, seq_len, head_dim] tensors
    // Conditions: CPU (tiled flash kernel) or Vulkan (tiled flash shader),
    // Float32, head_dim within the kernel's supported range, no explicit
    // attention mask.
    // Vulkan path needs head_dim <= 128 (flash_attention.comp constraint).
    // The Vulkan tiled shader now natively supports causal masking inline
    // (audit C2 Vulkan fix); the prior stale `vulkan_causal_supported=false`
    // gate forced causal inference through the slower composed path.
    // Dropout > 0 on Vulkan flows through the dispatch's composed-ops
    // fallback (Philox-keyed Bernoulli mask via dispatchPhiloxDropoutMask).
    // CPU path supports head_dim <= 256 and causal handled internally.
    // need_weights must be false (neither path computes attention weights).
    bool dev_cpu = (query.device().type == Device::Type::CPU);
    bool dev_vulkan = (query.device().type == Device::Type::Vulkan);
    int64_t flash_max_head_dim = dev_vulkan ? 128 : 256;
    bool can_use_flash_attention = !need_weights &&
                                   (dev_cpu || dev_vulkan) &&
                                   query.dtype() == DType::Float32 &&
                                   head_dim <= flash_max_head_dim &&
                                   // Flash kernel takes no additive bias; defer to the manual
                                   // BMM path when a position_bias is present.
                                   !has_position_bias &&
                                   (is_causal_ || !attn_mask.is_valid() || attn_mask.shape().size() == 0);

    // Same grad correctness gate as cuDNN SDPA above: this fast path also
    // returns Variable(output, false) without grad_fn — only safe when no
    // gradient is needed downstream. Re-uses grad_path_safe computed above.
    if (can_use_flash_attention && !is_training() && grad_path_safe) {
        try {
            float scale_f = 1.0f / std::sqrt(static_cast<float>(head_dim));

            // Make tensors contiguous for the kernel
            Tensor q_contig = query.tensor().is_contiguous() ? query.tensor() : query.tensor().contiguous();
            Tensor k_contig = key.tensor().is_contiguous() ? key.tensor() : key.tensor().contiguous();
            Tensor v_contig = value.tensor().is_contiguous() ? value.tensor() : value.tensor().contiguous();

            // Call Flash Attention via dispatch system (4D tensors directly)
            // When is_causal_ is true, the kernel applies causal masking internally,
            // which is more efficient than building and applying an explicit mask tensor
            // Dropout is fused into the kernel using Philox counter-based RNG
            OpAttributes attrs;
            attrs.set(AttrKey::Scale, static_cast<double>(scale_f));
            attrs.set(AttrKey::Causal, is_causal_);
            attrs.set(AttrKey::DropoutP, static_cast<double>(dropout_p));
            attrs.set(AttrKey::IsTraining, is_training());
            std::vector<Tensor> flash_inputs = {q_contig, k_contig, v_contig};
            Tensor output = dispatch<OpId::FlashAttention>(flash_inputs, attrs)[0];

            Variable attended(output, false);
            Tensor empty_weights({0}, output.dtype(), output.device());
            Variable attn_weights_empty(empty_weights, false);

            return {attended, attn_weights_empty};
        } catch (const std::exception& e) {
            // Fall through to standard BMM path if flash attention fails
        }
    }

    // Standard path: Use cuBLAS bmm operations (fast for all cases)

    // For Float16/BFloat16, upcast Q, K, V to Float32 for the full attention computation
    // to prevent gradient overflow. This matches PyTorch's scaled_dot_product_attention behavior.
    DType orig_dtype = query.dtype();
    bool needs_attn_upcast = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    Variable query_compute = needs_attn_upcast ? variable_cast(query, DType::Float32) : query;
    Variable key_compute = needs_attn_upcast ? variable_cast(key, DType::Float32) : key;
    Variable value_compute = needs_attn_upcast ? variable_cast(value, DType::Float32) : value;

    // Compute scaling factor
    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

    // Reshape Q, K, V from (batch, num_heads, seq_len, head_dim) to (batch*num_heads, seq_len, head_dim)
    // This allows us to use bmm for efficient batched matrix multiplication
    std::vector<int64_t> reshaped_q_shape = {batch_size * num_heads, seq_len_q, head_dim};
    std::vector<int64_t> reshaped_k_shape = {batch_size * num_heads, seq_len_k, head_dim};
    std::vector<int64_t> reshaped_v_shape = {batch_size * num_heads, seq_len_k, head_dim};

    auto query_3d = autograd::reshape(query_compute, reshaped_q_shape);
    auto key_3d = autograd::reshape(key_compute, reshaped_k_shape);
    auto value_3d = autograd::reshape(value_compute, reshaped_v_shape);

    // Fold the 1/sqrt(head_dim) scale into Q here rather than scaling the
    // materialized scores matrix below: Q is (B*H, Sq, D) while scores is
    // (B*H, Sq, Sk) — for seq>>head_dim that's ~Sk/D fewer elements to touch
    // (e.g. 8x at seq=512, head_dim=64), and it drops a full scores-sized
    // intermediate from the autograd graph. Mathematically identical:
    // (scale*Q) @ K^T == scale * (Q @ K^T).
    {
        Tensor q_scale = full({1}, static_cast<float>(scale),
                              query_3d.dtype(), query.device());
        query_3d = query_3d * Variable(q_scale, false);
    }

    // Transpose key: (batch*num_heads, seq_len_k, head_dim) -> (batch*num_heads, head_dim, seq_len_k)
    std::vector<int64_t> key_perm = {0, 2, 1};
    auto key_transposed = autograd::permute(key_3d, key_perm);

    // Compute attention scores: QK^T using batch matrix multiplication
    // (batch*num_heads, seq_len_q, head_dim) @ (batch*num_heads, head_dim, seq_len_k)
    // Result: (batch*num_heads, seq_len_q, seq_len_k)
    auto scores = autograd::bmm(query_3d, key_transposed);

    // Reshape scores back to (batch, num_heads, seq_len_q, seq_len_k)
    std::vector<int64_t> scores_4d_shape = {batch_size, num_heads, seq_len_q, seq_len_k};
    scores = autograd::reshape(scores, scores_4d_shape);

    // Scale already folded into Q above (scores now carry the 1/sqrt(d) factor).
    DType score_dtype = needs_attn_upcast ? DType::Float32 : orig_dtype;

    // Apply relative position bias if provided
    // position_bias shape: (num_heads, seq_len_q, seq_len_k) or (1, num_heads, seq_len_q, seq_len_k)
    if (position_bias.is_valid() && position_bias.shape().size() > 0) {
        Variable bias_var(position_bias, false);
        scores = scores + bias_var;
    }

    // Apply causal mask if is_causal_ is set (masks future tokens)
    if (is_causal_) {
        // Bottom-right alignment per docs/internals/attention-contract.md
        // (NestedAttention "Causal-with-cache convention"): when seq_q != seq_k
        // (cross-attention with a KV cache) the causal boundary is shifted by
        // offset = seq_k - seq_q so the last query attends to the last key and
        // existing cache always remains visible. This matches GQA's mask
        // (src/nn/layers/gqa_attention.cpp ~206-233). The previous
        // `triu(..., 1)` was top-left aligned (query i could only see keys
        // <= i), which diverged from GQA/the contract whenever seq_q != seq_k.
        //
        // Masking always runs at Float32 in the half case (needs_attn_upcast
        // makes score_dtype Float32 for Float16/BFloat16), so -inf is
        // representable and softmax(-inf)=0 with no overflow.
        const int64_t offset = seq_len_k - seq_len_q;
        // row_idx[i,j] = i, col_idx[i,j] = j  (Int64 broadcast grids)
        Tensor row_idx = tenzor::arange(0, seq_len_q, 1, DType::Int64, query.device())
                             .reshape({seq_len_q, 1});
        Tensor col_idx = tenzor::arange(0, seq_len_k, 1, DType::Int64, query.device())
                             .reshape({1, seq_len_k});
        Tensor row_exp = tenzor::expand(row_idx, std::vector<int64_t>{seq_len_q, seq_len_k});
        Tensor col_exp = tenzor::expand(col_idx, std::vector<int64_t>{seq_len_q, seq_len_k});
        // Per-row allowed boundary b_i = max(i + offset, 0): queries may attend
        // to columns j <= b_i. The clamp_min(0) prevents a fully-masked row
        // when offset < 0 (seq_k < seq_q) — without it the top queries would
        // have i + offset < 0, masking every column and giving softmax(all
        // -inf) = NaN.
        Tensor boundary = tenzor::clamp_min(
            tenzor::add(row_exp, static_cast<double>(offset)), 0.0);
        // masked_off: true where col > boundary (future / disallowed key).
        Tensor masked_off = gt(col_exp, boundary);
        Tensor neg_inf = full({seq_len_q, seq_len_k},
                              -std::numeric_limits<float>::infinity(),
                              score_dtype, query.device());
        Tensor zero_mask = zeros({seq_len_q, seq_len_k},
                                 score_dtype, query.device());
        Tensor causal = where(masked_off, neg_inf, zero_mask);
        Variable causal_var(causal, false);
        scores = scores + causal_var;
    }

    // Apply attention mask if provided
    if (attn_mask.is_valid() && attn_mask.shape().size() > 0) {
        // Validate mask shape is broadcastable to scores (batch, num_heads, seq_q, seq_k)
        auto mask_shape = attn_mask.shape();
        auto scores_shape = scores.shape();
        // Mask must be 2D (seq_q, seq_k), 3D (num_heads, seq_q, seq_k),
        // or 4D (batch, num_heads, seq_q, seq_k) and broadcastable
        if (mask_shape.size() > 4) {
            throw std::runtime_error(
                "Attention mask must be 2D, 3D, or 4D, got " +
                std::to_string(mask_shape.size()) + "D");
        }
        // Validate trailing dimensions match
        int64_t mask_ndim = static_cast<int64_t>(mask_shape.size());
        int64_t scores_ndim = static_cast<int64_t>(scores_shape.size());
        for (int64_t i = 1; i <= std::min(mask_ndim, scores_ndim); ++i) {
            int64_t mask_dim = mask_shape[mask_ndim - i];
            int64_t scores_dim = scores_shape[scores_ndim - i];
            if (mask_dim != 1 && mask_dim != scores_dim) {
                throw std::runtime_error(
                    "Attention mask shape is not broadcastable to scores shape. "
                    "Mask dim " + std::to_string(mask_ndim - i) + " is " +
                    std::to_string(mask_dim) + " but scores dim is " +
                    std::to_string(scores_dim));
            }
        }
        // Y.19 / EE.11: Bool/integer masks (PyTorch convention: True = ignore)
        // must be widened to a float -inf/0 mask before the additive combine.
        // Without this, `float scores + Bool mask` adds 1.0 at masked positions
        // instead of -inf, leaking attention through. Mirrors the normalisation
        // block in MultiheadAttention::forward (lines ~643-664).
        Tensor normalised_mask = normalize_attn_mask(attn_mask);
        // Add mask (mask should have -inf for positions to mask out)
        Variable mask_var(normalised_mask, false);
        scores = scores + mask_var;
    }

    // Apply softmax to get attention weights
    // Softmax over last dimension (seq_len_k)
    // For Float16/BFloat16, scores are already in Float32 from upcast above
    Variable attn_weights = autograd::softmax(scores, -1);

    // Apply dropout if in training mode and dropout > 0
    if (dropout_p > 0.0 && is_training()) {
        attn_weights = dropout_layer_->forward(attn_weights);
    }

    // Reshape attention weights from (batch, num_heads, seq_len_q, seq_len_k) to (batch*num_heads, seq_len_q, seq_len_k)
    auto attn_weights_3d = autograd::reshape(attn_weights, {batch_size * num_heads, seq_len_q, seq_len_k});

    // Compute weighted sum of values using batch matrix multiplication
    // (batch*num_heads, seq_len_q, seq_len_k) @ (batch*num_heads, seq_len_k, head_dim)
    // Result: (batch*num_heads, seq_len_q, head_dim)
    auto attended_3d = autograd::bmm(attn_weights_3d, value_3d);

    // Reshape back to (batch, num_heads, seq_len_q, head_dim)
    std::vector<int64_t> attended_4d_shape = {batch_size, num_heads, seq_len_q, head_dim};
    auto attended = autograd::reshape(attended_3d, attended_4d_shape);

    // Downcast attention output back to original dtype
    if (needs_attn_upcast) {
        attended = variable_cast(attended, orig_dtype);
        attn_weights = variable_cast(attn_weights, orig_dtype);
    }

    return {attended, attn_weights};
}

auto MultiheadAttention::forward(const Variable& query,
                                const Variable& key,
                                const Variable& value,
                                const Tensor& key_padding_mask,
                                const Tensor& attn_mask,
                                bool need_weights,
                                const Tensor& position_bias) -> std::pair<Variable, Variable> {
    // Call forward pre-hooks (enables CPU-start offloading)
    // NOTE: This is necessary because this multi-argument forward bypasses Module::forward()
    call_forward_pre_hooks();
    // Multi-input pre-hooks receive the actual (query, key, value) inputs. This
    // multi-arg forward is the canonical multi-input entry point, so without
    // this call the register_forward_pre_hook_multi API would never fire.
    call_forward_pre_hooks_multi({query, key, value});

    // Handle batch_first parameter - only transform if needed
    const Variable* q_ptr = &query;
    const Variable* k_ptr = &key;
    const Variable* v_ptr = &value;

    Variable q_permuted, k_permuted, v_permuted;
    if (!batch_first_) {
        // Convert from (seq, batch, embed) to (batch, seq, embed)
        q_permuted = autograd::permute(query, {1, 0, 2});
        k_permuted = autograd::permute(key, {1, 0, 2});
        v_permuted = autograd::permute(value, {1, 0, 2});
        q_ptr = &q_permuted;
        k_ptr = &k_permuted;
        v_ptr = &v_permuted;
    }

    auto q_shape = q_ptr->shape();
    if (q_shape.size() != 3) {
        throw std::invalid_argument("MultiheadAttention: query must be 3D [batch, seq, embed], got " +
            std::to_string(q_shape.size()) + "D");
    }
    auto k_shape = k_ptr->shape();
    if (k_shape.size() != 3 || v_ptr->shape().size() != 3) {
        throw std::invalid_argument("MultiheadAttention: key/value must be 3D");
    }
    if (q_shape[2] != embed_dim_) {
        throw std::invalid_argument("MultiheadAttention: query embed_dim (" +
            std::to_string(q_shape[2]) + ") != expected (" + std::to_string(embed_dim_) + ")");
    }
    // Per docs/internals/attention-contract.md: passing both is_causal and an
    // explicit USER attn_mask is ambiguous (PyTorch errors here). Reject it at
    // this level, where the user's attn_mask argument is distinguishable from a
    // key_padding_mask. A key_padding_mask MAY be combined with is_causal — it
    // is folded into a padding-only additive mask below and applied alongside
    // the causal triangular mask in the manual BMM path.
    if (is_causal_ && attn_mask.is_valid() && attn_mask.shape().size() > 0) {
        throw std::invalid_argument(
            "MultiheadAttention: is_causal=true and an explicit attn_mask are "
            "mutually exclusive. Pass exactly one. (Use is_causal for triangular "
            "masking; use attn_mask for arbitrary additive masks. A "
            "key_padding_mask may be combined with is_causal.)");
    }

    int64_t batch_size = q_shape[0];
    int64_t seq_len_q = q_shape[1];
    int64_t seq_len_k = k_shape[1];
    // Number of key/value rows appended by add_bias_kv / add_zero_attn. These
    // extend seq_len_k beyond the user-supplied key length, so a user-provided
    // key_padding_mask (which has the original column count) must be padded with
    // this many "keep" columns before it can be reshaped against seq_len_k.
    int64_t num_appended_keys = 0;

    // Ensure projection weights are on the same device as input
    auto weight_device = q_proj_->own_parameters()[0]->tensor().device();
    auto input_device = q_ptr->tensor().device();

    if (weight_device != input_device) {
        // Move projection layers to input device to preserve autograd chain.
        // The bias_k_/bias_v_ shared_ptrs alias the registered_parameters map
        // entries, so calling Module::to(input_device) on this module updates
        // their tensors in-place (per src/nn/module.cpp:90-115). We don't
        // rebuild local Variables here — that would detach the shared_ptr
        // and leave the registered_parameters map pointing at a stale tensor
        // (the dangling-leaf bug from feedback_raw_tensor_op_bug).
        q_proj_->to(input_device);
        k_proj_->to(input_device);
        v_proj_->to(input_device);
        out_proj_->to(input_device);
        if (add_bias_kv_) {
            this->to(input_device);  // updates parameters_["bias_k"], aliased by bias_k_
        }
    }

    const Variable& q_compat = *q_ptr;
    const Variable& k_compat = *k_ptr;
    const Variable& v_compat = *v_ptr;

    // Project inputs
    Variable Q = q_proj_->forward(q_compat);
    Variable K = k_proj_->forward(k_compat);
    Variable V = v_proj_->forward(v_compat);

    // add_bias_kv: concatenate bias_k/bias_v to key/value sequences
    if (add_bias_kv_) {
        // bias_k_ is [1, 1, embed_dim], expand to [batch_size, 1, embed_dim].
        // Dereference shared_ptr to get the live Variable (kept current by
        // Module::to(); see member declaration in attention.hpp).
        auto bk = tenzor::expand(*bias_k_, {batch_size, 1, embed_dim_});
        auto bv = tenzor::expand(*bias_v_, {batch_size, 1, embed_dim_});
        K = tenzor::cat({K, bk}, 1);  // [batch, seq_k+1, embed]
        V = tenzor::cat({V, bv}, 1);  // [batch, seq_k+1, embed]
        seq_len_k += 1;
        num_appended_keys += 1;
    }

    // add_zero_attn: append zero row to key/value sequences
    if (add_zero_attn_) {
        auto zero_row = Variable(tenzor::zeros({batch_size, 1, embed_dim_},
                                                K.tensor().dtype(), K.tensor().device()), false);
        K = tenzor::cat({K, zero_row}, 1);
        V = tenzor::cat({V, zero_row}, 1);
        seq_len_k += 1;
        num_appended_keys += 1;
    }

    // Reshape for multi-head attention
    Q = transpose_for_scores(Q);  // (batch, num_heads, seq_len_q, head_dim)
    K = transpose_for_scores(K);  // (batch, num_heads, seq_len_k, head_dim)
    V = transpose_for_scores(V);  // (batch, num_heads, seq_len_k, head_dim)

    // Prepare attention mask
    Tensor combined_mask;

    if (attn_mask.is_valid() && attn_mask.shape().size() > 0) {
        // R.21: normalise attn_mask to a 4D shape that broadcasts cleanly
        // against key_padding_mask's [N, 1, 1, Sk] form. PyTorch accepts:
        //   - 2D [Sq, Sk]            -> [1, 1, Sq, Sk]
        //   - 3D [N*H, Sq, Sk]       -> [N, H, Sq, Sk]
        //   - 4D [N, H, Sq, Sk]      (used as-is)
        // Previously a 3D mask was passed through verbatim; the subsequent
        // additive combine with the 4D padding mask then crashed on shape
        // mismatch (the 3D leading dim N*H cannot broadcast to N).
        //
        // Y.19: a Bool attn_mask (PyTorch convention: True = ignore) must be
        // widened to a float mask with -inf at True positions before the
        // additive combine. Mirrors V.28's key_padding_mask handling; without
        // this, the downstream `add(combined_mask, broadcastable_mask)`
        // computes `float + bool` (treated as 0/1), so masked positions get
        // an additive `1.0` instead of `-inf`, leaking attention to them.
        Tensor normalised_mask = normalize_attn_mask(attn_mask);

        auto am_shape = normalised_mask.shape();
        if (am_shape.size() == 2) {
            std::vector<int64_t> new_shape = {1, 1, am_shape[0], am_shape[1]};
            combined_mask = reshape(normalised_mask, new_shape);
        } else if (am_shape.size() == 3) {
            if (am_shape[0] != batch_size * num_heads_) {
                throw std::invalid_argument(
                    "MultiheadAttention: 3D attn_mask leading dim (" +
                    std::to_string(am_shape[0]) + ") must equal batch_size*num_heads (" +
                    std::to_string(batch_size * num_heads_) + ")");
            }
            std::vector<int64_t> new_shape = {batch_size, num_heads_, am_shape[1], am_shape[2]};
            combined_mask = reshape(normalised_mask, new_shape);
        } else if (am_shape.size() == 4) {
            combined_mask = normalised_mask;
        } else {
            throw std::invalid_argument(
                "MultiheadAttention: attn_mask must be 2D, 3D, or 4D, got " +
                std::to_string(am_shape.size()) + "D");
        }
    }

    // Add key padding mask if provided
    if (key_padding_mask.is_valid() && key_padding_mask.shape().size() > 0) {
        // key_padding_mask: (batch, seq_len_k). Reshape to a broadcastable
        // (batch, 1, 1, seq_len_k) and rely on downstream `add` to broadcast
        // along (num_heads, seq_q). Previously this code did an explicit
        // expand to [batch, num_heads, seq_q, seq_k] producing a stride-0
        // view that some backends mishandle (memory: feedback_stride_bugs).
        // add_bias_kv / add_zero_attn appended num_appended_keys extra key rows,
        // bumping seq_len_k past the user mask's column count. PyTorch pads the
        // key_padding_mask with "keep" entries for those positions; mirror that
        // here so the appended bias/zero keys are always attendable. The pad
        // value 0 is correct under both conventions handled below: 0.0 is the
        // additive-keep value for float masks, and 0/False means "not ignored"
        // for bool/integer indicator masks.
        Tensor source_mask = key_padding_mask;
        if (num_appended_keys > 0) {
            auto src_shape = source_mask.shape();
            std::vector<int64_t> pad_shape(src_shape.begin(), src_shape.end());
            pad_shape.back() = num_appended_keys;
            Tensor keep_cols = zeros(pad_shape, source_mask.dtype(), source_mask.device());
            source_mask = tenzor::cat(std::vector<Tensor>{source_mask, keep_cols}, 1);
        }

        std::vector<int64_t> mask_shape = {batch_size, 1, 1, seq_len_k};
        Tensor padding_mask = reshape(source_mask, mask_shape);

        // V.28 / AA.1: branch on dtype to honour PyTorch's two distinct
        // key_padding_mask conventions:
        //   - Bool / integer mask: indicator semantics, True = ignore.
        //     Convert via `gt(mask, 0.5)` + `where(-inf, 0)`.
        //   - Floating mask: ADDITIVE semantics (0.0 = keep, -inf = mask).
        //     Add the mask directly to combined_mask, without thresholding.
        //
        // The previous code unconditionally took the threshold path, which
        // silently inverted float additive masks: a finite mask value like
        // -inf was treated as "0.5? no" → keep, while finite 0.0 was also
        // treated as "0.5? no" → keep — so masked positions were never
        // actually masked under the additive convention.
        const DType mask_dtype = padding_mask.dtype();
        const bool is_floating_mask =
            (mask_dtype == DType::Float32 || mask_dtype == DType::Float64 ||
             mask_dtype == DType::Float16 || mask_dtype == DType::BFloat16);

        Tensor broadcastable_mask;
        if (is_floating_mask) {
            // Additive float mask: pass through unchanged. Downstream `add`
            // broadcasts [N,1,1,Sk] across [N,H,Sq,Sk].
            broadcastable_mask = padding_mask;
        } else {
            // Bool or integer indicator mask: widen to Float32 so the
            // threshold/where path is well-defined (Bool truncates -inf
            // to 1, integers can't represent -inf at all).
            if (mask_dtype != DType::Float32) {
                padding_mask = padding_mask.to(DType::Float32);
            }

            auto pm_shape = std::vector<int64_t>(padding_mask.shape().begin(), padding_mask.shape().end());
            Tensor neg_inf_tensor = full(pm_shape, -std::numeric_limits<float>::infinity(),
                                         padding_mask.dtype(), padding_mask.device());
            Tensor zero_tensor = zeros(pm_shape, padding_mask.dtype(), padding_mask.device());
            Tensor threshold = full(pm_shape, 0.5f, padding_mask.dtype(), padding_mask.device());

            Tensor mask_gt = Tensor(gt(padding_mask, threshold));
            broadcastable_mask = Tensor(where(mask_gt, neg_inf_tensor, zero_tensor));
        }

        if (combined_mask.shape().size() > 0) {
            combined_mask = Tensor(add(combined_mask, broadcastable_mask));
        } else {
            combined_mask = broadcastable_mask;
        }
    }

    // Compute attention
    auto [attended_values, attn_weights] = scaled_dot_product_attention(Q, K, V, combined_mask, dropout_, need_weights, position_bias);

    // Merge heads
    Variable output = merge_heads(attended_values);

    // Final output projection
    output = out_proj_->forward(output);

    // Convert back to (seq, batch, embed) if needed
    if (!batch_first_) {
        output = autograd::permute(output, {1, 0, 2});
    }

    // Call forward post-hooks (enables CPU-start offloading)
    call_forward_post_hooks();
    // Multi-input post-hooks receive the (query, key, value) inputs and the
    // final output, completing the multi-input hook API wiring.
    call_forward_post_hooks_multi({query, key, value}, {output});

    // Return attention weights based on need_weights flag
    if (need_weights) {
        // add_bias_kv / add_zero_attn appended num_appended_keys extra key
        // columns (last dim of attn_weights). PyTorch trims these synthetic
        // columns from the returned weights so the shape stays (N, H, L, S)
        // against the user's original key length S, rather than the widened
        // S + num_appended_keys. Without trimming, a caller indexing the
        // weights against the original S misaligns or reads out of bounds.
        if (num_appended_keys > 0) {
            const auto aw_shape = attn_weights.shape();
            const int64_t last_dim = static_cast<int64_t>(aw_shape.size()) - 1;
            const int64_t orig_len = aw_shape[last_dim] - num_appended_keys;
            attn_weights = tenzor::slice(attn_weights, last_dim, 0, orig_len);
        }
        return {output, attn_weights};
    } else {
        // Create empty Variable with no shape when weights not needed
        // Use zeros with empty shape to ensure proper initialization
        Tensor empty_tensor = zeros({}, DType::Float32, output.device());
        Variable empty_var(empty_tensor, false);  // requires_grad=false
        return {output, empty_var};
    }
}

auto MultiheadAttention::forward_impl(const Variable& input) -> Variable {
    // Self-attention: Q = K = V = input
    auto [output, _] = forward(input, input, input, Tensor{}, Tensor{}, false);
    return output;
}

// ============================================================================
// Helper Functions
// ============================================================================

auto create_causal_mask(int64_t seq_len, Device device, DType dtype) -> Tensor {
    // Create upper triangular matrix with -inf above diagonal
    // Always create on CPU first as Float32 to avoid dereferencing device pointers
    Tensor mask = zeros({seq_len, seq_len}, DType::Float32, Device::cpu());
    auto* data = mask.data<float>();

    for (int64_t i = 0; i < seq_len; ++i) {
        for (int64_t j = 0; j < seq_len; ++j) {
            if (j > i) {
                data[i * seq_len + j] = -std::numeric_limits<float>::infinity();
            } else {
                data[i * seq_len + j] = 0.0f;
            }
        }
    }

    // Convert to target dtype if needed
    if (dtype != DType::Float32) {
        mask = mask.to(dtype);
    }

    // Move to target device if needed
    if (device != Device::cpu()) {
        return mask.to(device);
    }

    return mask;
}

} // namespace nn
} // namespace tenzor
