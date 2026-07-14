// Autograd integration for FlashAttention / FusedAttention / FlexAttention.
//
// Per docs/internals/attention-contract.md, every public attention call must
// route through one of these Functions so the resulting Variable carries a
// grad_fn — direct dispatch<OpId::FlashAttention>() returns an autograd-severed
// Variable, which silently zeros Q/K/V gradients (audit C1).
//
// The Function classes are declared at the bottom of include/tenzor/autograd/function.hpp.
// User-facing apply helpers (flash_attention, fused_attention, flex_attention)
// are declared in include/tenzor/autograd/ops.hpp.
//
// Backend dtype/contract gaps:
//   - Today most backends still return only `output` from FlashAttention forward
//     (the audit's recurring "returns 1 tensor not 2" bug). M3-M7 wire the full
//     4-tuple; this Function is defensive about that and falls through to a
//     composed-ops backward when L is not saved.
//   - FlashAttentionBackward composed-ops fallback: recompute scores/softmax
//     from Q/K/V; correct but more memory (O(N^2)).
//   - Dropout reproducibility requires (seed, offset) round-trip; until M3-M7
//     produce them, dropout > 0 in training mode falls back to composed-ops
//     (which itself needs to use the same Philox seed — limitation flagged
//     until M3 lands the seed plumbing on CPU).

#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/anomaly_mode.hpp"
#include <iostream>
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/philox_dropout.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/utils/log.hpp"

#include <mutex>
#include <stdexcept>
#include <vector>

namespace tenzor {

namespace {

// NOTE (audit A-04): the composed backward replays the forward dropout mask
// via the shared `philox_dropout_mask` op (see composed_attention_backward
// below), which uses the counter convention (bh, qi, ki, 0). The CPU
// FlashAttention forward kernel uses the SAME convention — counter key
// (b*num_heads + h, qi, ki, 0) == the (bh, qi, ki, 0) of philox_dropout_mask
// (src/backends/cpu/kernels/flash_attention.cpp) — so the composed backward is
// bit-exact against the CPU forward, not merely against the GPU kernels. The
// former hand-rolled host Philox helper (host_philox_uniform / host_philox_round)
// was removed as dead code; philox_dropout_mask is now the single source of
// truth for the replay, on every backend.

// Composed-ops fallback for FlashAttention backward when the fused dispatch
// path is unavailable. Mathematically equivalent to the fused kernel; uses
// O(N^2) memory because P = softmax(scores) is materialized.
//
// Dropout replay: when dropout_p > 0 and philox_seed != 0, applies the
// inverted-scaled Philox mask to P before computing dV/dQ/dK so the gradient
// matches what the forward actually computed (audit contract requirement).
//
// Math:
//   S = Q @ K^T * scale  + (causal mask if applicable)
//   P = softmax(S, dim=-1)
//   if dropout: P_drop[i,j] = (philox(b,h,i,j) < p) ? 0 : P[i,j] / (1 - p)
//   dV = P_drop^T @ dO
//   dP = dO @ V^T
//   dS = P_drop * (dP - sum(dP * P_drop, dim=-1, keepdim))
//   dQ = dS @ K * scale
//   dK = dS^T @ Q * scale
auto composed_attention_backward(const Tensor& dO,
                                 const Tensor& Q,
                                 const Tensor& K,
                                 const Tensor& V,
                                 double scale,
                                 bool causal,
                                 double dropout_p = 0.0,
                                 uint64_t rng_seed = 0ull,
                                 uint64_t philox_offset = 0ull) -> std::vector<Tensor> {
    // V.8: double scale end-to-end; audit-3 R.6 had widened only the
    // Variable form. dropout_p is also widened to keep parity with the
    // Variable path; the only downstream effect is the threshold used
    // when constructing the dropout mask (precision-irrelevant for the
    // mask itself but consistent with the rest of the signature).
    // Q, K, V, dO all have shape [B, H, S, D] or [B*H, S, D]. We work on
    // 3D for bmm convenience by collapsing leading dims if 4D. shape() returns
    // a span; convert to vector for the reshape API.
    auto Q_shape_span = Q.shape();
    std::vector<int64_t> orig_shape(Q_shape_span.begin(), Q_shape_span.end());
    std::vector<int64_t> K_shape(K.shape().begin(), K.shape().end());
    std::vector<int64_t> V_shape(V.shape().begin(), V.shape().end());
    bool is_4d = (orig_shape.size() == 4);

    // The composed path collapses [B,H,S,D] -> [B*H,S,D] for Q,K,V identically
    // and does bmm(Q3, K3^T); this is only valid when Q and K/V share the same
    // head count H (standard MHA). Grouped/multi-query attention is materialised
    // to full H by repeat_kv BEFORE any flash/composed path in this library
    // (see src/nn/layers/gqa_attention.cpp), so equal heads is an invariant
    // here. Assert it loudly rather than silently computing a wrong-grouping
    // gradient if a caller ever feeds raw GQA-shaped K/V directly.
    if (is_4d) {
        if (K_shape.size() == 4 && K_shape[1] != orig_shape[1]) {
            throw std::runtime_error(
                "composed_attention_backward: Q and K have different head "
                "counts (GQA/MQA not supported on the composed fallback; "
                "expand K/V to the query head count before calling)");
        }
        if (V_shape.size() == 4 && V_shape[1] != orig_shape[1]) {
            throw std::runtime_error(
                "composed_attention_backward: Q and V have different head "
                "counts (GQA/MQA not supported on the composed fallback)");
        }
    }

    auto reshape_3d = [&](const Tensor& t) -> Tensor {
        if (!is_4d) return t;
        auto s = t.shape();
        return tenzor::reshape(t, std::vector<int64_t>{s[0] * s[1], s[2], s[3]});
    };

    Tensor Q3 = reshape_3d(Q);
    Tensor K3 = reshape_3d(K);
    Tensor V3 = reshape_3d(V);
    Tensor dO3 = reshape_3d(dO);

    // S = Q @ K^T * scale
    Tensor Kt = tenzor::transpose(K3, -1, -2);
    Tensor S = tenzor::bmm(Q3, Kt);
    {
        Tensor scale_t = tenzor::full({1}, static_cast<double>(scale), S.dtype(), S.device());
        S = S * scale_t;
    }

    if (causal) {
        // Apply -INFINITY mask above diagonal via where(mask>0, -inf, S) —
        // replaces the previous additive form `S + (mask * -inf)` which
        // computes 0 * -inf = NaN at unmasked positions on backends that
        // don't short-circuit the multiply. See attention-contract.md
        // (sentinel must be -INFINITY; never -1e9 / -1e30 — FP16 saturates
        // and leaks gradient mass).
        int64_t S_q = S.shape()[1];
        int64_t S_k = S.shape()[2];
        // F026: BOTTOM-RIGHT causal alignment — MUST match the forward, which
        // masks ki > qi + (S_k - S_q) (flash_attention forward causal_offset =
        // K.shape[2] - seq_len; GQA offset = seq_k - seq_q). The previous top-left
        // offset (1) differentiated a DIFFERENT masked softmax than the forward
        // computed whenever S_q != S_k (causal cross-attention / cached-KV
        // decode), yielding wrong dQ/dK/dV. triu offset 1 + (S_k - S_q) masks
        // col > row + (S_k - S_q), i.e. ki > qi + (S_k - S_q).
        Tensor mask = tenzor::triu(tenzor::ones({S_q, S_k}, S.dtype(), S.device()),
                                   1 + (S_k - S_q));
        Tensor neg_inf = tenzor::full({1}, -std::numeric_limits<float>::infinity(),
                                      S.dtype(), S.device());
        S = tenzor::where(mask, neg_inf, S);
    }

    // P = softmax(S, dim=-1) — inlined stable softmax (no Tensor overload exists;
    // wrapping in Variable would needlessly allocate autograd nodes for a no-grad
    // recomputation in a backward pass).
    Tensor S_max = tenzor::max(S, /*dim=*/std::optional<int64_t>{-1}, /*keepdim=*/true);
    Tensor S_centered = S - S_max;
    Tensor S_exp = tenzor::exp(S_centered);
    Tensor S_exp_sum = tenzor::sum(S_exp, /*dim=*/std::optional<int64_t>{-1}, /*keepdim=*/true);
    Tensor P = S_exp / S_exp_sum;

    // Replay forward Philox dropout (audit M4-rem follow-up, A.11):
    // Build the same dropout mask the forward applied using the shared
    // `philox_dropout_mask` op (same counter convention `(bh, qi, ki, 0)`
    // used by every backend's flash forward, including the CPU kernel's
    // `(b*num_heads + h, qi, ki, 0)` key — so this replay is bit-exact against
    // the forward on CPU as well as on GPU). The mask is generated on
    // CPU then ferried to P's device — replaces the previous host triple-
    // for-loop, which was effectively a hand-rolled re-implementation of
    // philox_dropout_mask. The mask op returns scale=1/(1-p) for kept
    // positions and 0 for dropped, the multiplicative correction for the
    // inverted-dropout forward (Pd = P_soft * mask, O = Pd @ V).
    //
    // CRITICAL (audit core-25): the softmax Jacobian must use the *unmasked*
    // softmax P_soft, not Pd. The dropout mask sits between P_soft and the
    // output, so its only effect on the softmax backward is to scale the
    // incoming gradient: dL/dP_soft = dP * mask. Folding the mask into both
    // the outer P factor and the row-sum (dS = Pd*(dP - rowsum(dP*Pd)))
    // double-applies it to the row-correction term, over-scaling it by
    // 1/(1-p) at every kept position and corrupting dQ/dK/dV whenever
    // dropout_p > 0. Keep P (= P_soft) for the Jacobian; track Pd separately
    // for dV and for masking the incoming gradient.
    Tensor Pd = P;          // Pd = P_soft * mask (== P_soft when no dropout)
    bool has_dropout = (dropout_p > 0.0 && rng_seed != 0ull);
    Tensor mask_dev;        // valid only when has_dropout
    if (has_dropout) {
        auto P_shape = P.shape();
        std::vector<int64_t> mask_shape(P_shape.begin(), P_shape.end());
        Tensor mask_cpu = philox_dropout_mask(
            mask_shape,
            static_cast<double>(dropout_p),
            rng_seed,
            /*offset=*/philox_offset,
            DType::Float32);
        mask_dev = mask_cpu.to(P.device()).to(P.dtype());
        Pd = P * mask_dev;
    }

    // dV = Pd^T @ dO   (forward used Pd, so dV depends on the masked product)
    Tensor Pt = tenzor::transpose(Pd, -1, -2);
    Tensor dV3 = tenzor::bmm(Pt, dO3);

    // dP = dO @ V^T   (= dL/dPd)
    Tensor Vt = tenzor::transpose(V3, -1, -2);
    Tensor dP = tenzor::bmm(dO3, Vt);

    // Gradient flowing into the softmax output: dL/dP_soft = dP * mask.
    // Without dropout this is just dP.
    Tensor g = has_dropout ? (dP * mask_dev) : dP;

    // Softmax backward on the UNMASKED softmax P (= P_soft):
    //   dS = P * (g - sum(g * P, dim=-1, keepdim))
    Tensor gP = g * P;
    Tensor row_sum = tenzor::sum(gP, /*dim=*/std::optional<int64_t>{-1}, /*keepdim=*/true);
    Tensor dS = P * (g - row_sum);

    // dQ = dS @ K * scale
    Tensor dQ3 = tenzor::bmm(dS, K3);
    {
        Tensor scale_t = tenzor::full({1}, static_cast<double>(scale), dQ3.dtype(), dQ3.device());
        dQ3 = dQ3 * scale_t;
    }

    // dK = dS^T @ Q * scale
    Tensor dSt = tenzor::transpose(dS, -1, -2);
    Tensor dK3 = tenzor::bmm(dSt, Q3);
    {
        Tensor scale_t = tenzor::full({1}, static_cast<double>(scale), dK3.dtype(), dK3.device());
        dK3 = dK3 * scale_t;
    }

    if (is_4d) {
        dQ3 = tenzor::reshape(dQ3, orig_shape);
        dK3 = tenzor::reshape(dK3, K_shape);
        dV3 = tenzor::reshape(dV3, V_shape);
    }
    return {dQ3, dK3, dV3};
}

// Try the fused FlashAttentionBackward dispatch first; fall through to
// composed-ops on any backend-side rejection (unsupported head_dim, missing L,
// etc.). Mirrors the policy the CUDA kernel registry already implements at
// cuda_kernel_registry.cpp:1685 — this is the cross-backend version.
auto try_fused_or_compose_backward(const Tensor& dO,
                                   const Tensor& Q,
                                   const Tensor& K,
                                   const Tensor& V,
                                   const Tensor& O,
                                   const Tensor& L,
                                   double scale,
                                   bool causal,
                                   double dropout_p,
                                   const Tensor& philox_seed,
                                   const Tensor& philox_offset) -> std::vector<Tensor> {
    // V.8: double scale/dropout_p; the callers (FlashAttentionBackward,
    // FusedAttentionBackward) already hold `double scale_` and were
    // narrowing through the float parameter at every call.
    // Read seed (if any) for the composed-fallback dropout replay.
    // Audit-5 X.6: previously narrowed Int64 → uint32, discarding the upper
    // 32 bits of the saved Philox seed. When the forward used the full 64
    // bits (manual_seed past 2^32) the replay produced a different mask
    // than the forward → corrupted dQ/dK/dV. philox_dropout_mask accepts
    // uint64_t natively; thread the full width through.
    uint64_t seed_for_replay = 0ull;
    if (philox_seed.is_valid() && philox_seed.numel() > 0) {
        // Move to CPU to read scalar; tiny single-int copy.
        Tensor seed_cpu = philox_seed.cpu();
        if (seed_cpu.dtype() == DType::Int64) {
            seed_for_replay = static_cast<uint64_t>(seed_cpu.data<int64_t>()[0]);
        } else if (seed_cpu.dtype() == DType::Int32) {
            seed_for_replay = static_cast<uint64_t>(static_cast<uint32_t>(seed_cpu.data<int32_t>()[0]));
        }
    }
    // Read the saved Philox OFFSET as well: the forward may have advanced the
    // global Philox counter (offset != 0), and the composed dropout replay must
    // regenerate the mask at that exact offset. Hardcoding offset 0 produced a
    // different mask than the forward whenever the forward ran with a nonzero
    // offset, corrupting dQ/dK/dV under dropout.
    uint64_t offset_for_replay = 0ull;
    if (philox_offset.is_valid() && philox_offset.numel() > 0) {
        Tensor offset_cpu = philox_offset.cpu();
        if (offset_cpu.dtype() == DType::Int64) {
            offset_for_replay = static_cast<uint64_t>(offset_cpu.data<int64_t>()[0]);
        } else if (offset_cpu.dtype() == DType::Int32) {
            offset_for_replay = static_cast<uint64_t>(static_cast<uint32_t>(offset_cpu.data<int32_t>()[0]));
        }
    }

    if (dropout_p > 0.0 || !L.is_valid() || L.shape().size() == 0) {
        // Use the composed backward when (a) there is no saved logsumexp, or
        // (b) dropout was applied: the fused FlashAttentionBackward kernel does
        // not replay the forward dropout mask, so routing a dropout backward
        // through it yields the wrong (and wrong-shaped) gradient. The composed
        // path rebuilds the exact mask from the saved Philox seed, so dV/dQ/dK
        // match the masked attention pattern and are deterministic per seed.
        return composed_attention_backward(dO, Q, K, V, scale, causal,
                                            dropout_p, seed_for_replay,
                                            offset_for_replay);
    }
    try {
        // A-03: canonical fused-backward input order is
        //   [dO, Q, K, V, O, L, (seed?), (offset?)].
        // The trailing optionals are disambiguated upstream by save-time boolean
        // predicates (has_lse, and dropout_p_ && is_training_ for the seed/offset
        // dropout-state pair) in FlashAttentionBackward::backward — NOT by vector
        // size or slot position. See the A-02 decode there.
        std::vector<Tensor> bwd_inputs = {dO, Q, K, V, O, L};
        if (philox_seed.is_valid() && philox_offset.is_valid()) {
            bwd_inputs.push_back(philox_seed);
            bwd_inputs.push_back(philox_offset);
        }
        OpAttributes bwd_attrs;
        bwd_attrs.set(AttrKey::Scale, static_cast<double>(scale));
        bwd_attrs.set(AttrKey::Causal, causal);
        bwd_attrs.set(AttrKey::DropoutP, static_cast<double>(dropout_p));
        return dispatch<OpId::FlashAttentionBackward>(bwd_inputs, bwd_attrs);
    } catch (const std::exception& e) {
        // The fused FlashAttentionBackward kernel does not support this
        // dtype/head-dim combination on this backend — fall back to the
        // mathematically-equivalent composed backward. Surface the underlying
        // reason under anomaly mode so a genuine backend error is not silently
        // masked as a routing decision (audit autograd-attn-01).
        if (is_anomaly_detection_enabled()) {
            std::cerr << "[tenzor] FlashAttentionBackward fused path unavailable; "
                         "using composed backward. Reason: " << e.what() << std::endl;
        }
        return composed_attention_backward(dO, Q, K, V, scale, causal,
                                            dropout_p, seed_for_replay,
                                            offset_for_replay);
    }
}

// Same defensive dispatch path for FlexAttention backward.
auto try_flex_backward_or_throw(const Tensor& dO,
                                const Tensor& Q,
                                const Tensor& K,
                                const Tensor& V,
                                const Tensor& O,
                                const Tensor& L,
                                double scale,
                                int64_t score_mod_id,
                                const Tensor& block_mask,
                                int64_t window_size,
                                int64_t prefix_length,
                                const Tensor& relpos_bias) -> std::vector<Tensor> {
    // V.8: double scale; the FlexAttentionBackward holds `double scale_`.
    // A-03: canonical flex-backward input order is
    //   [dO, Q, K, V, O, (L?), (block_mask?), (relpos_bias?)].
    // The trailing optionals are disambiguated upstream by save-time boolean
    // predicates (has_lse, and has_block_mask_) in FlexAttentionBackward::backward
    // — NOT by vector size or slot position. See the A-01 decode there.
    const bool has_block_mask = block_mask.is_valid() && block_mask.shape().size() > 0;
    const bool has_relpos_bias = relpos_bias.is_valid() && relpos_bias.shape().size() > 0;
    std::vector<Tensor> bwd_inputs = {dO, Q, K, V, O};
    if (L.is_valid() && L.shape().size() > 0) bwd_inputs.push_back(L);
    if (has_block_mask) bwd_inputs.push_back(block_mask);
    // R8 fix: relpos_bias (id 3) is appended LAST, matching exactly what
    // flex_attention_score_mod_backward reads (inputs.back() for id 3). The
    // builder previously never appended a bias, so the backward helper aliased
    // L / block_mask as the bias and produced wrong gradients.
    if (has_relpos_bias) bwd_inputs.push_back(relpos_bias);
    OpAttributes bwd_attrs;
    bwd_attrs.set(AttrKey::Scale, static_cast<double>(scale));
    bwd_attrs.set(AttrKey::ScoreModId, static_cast<int64_t>(score_mod_id));
    // AUTOGRAD-R027: explicit presence flags, NOT inferred from bwd_inputs.size()
    // (independently-optional trailing slots make size ambiguous — see the A-03
    // comment above). Backend wrappers use these to decide whether their
    // score_mod_id∈{0,1} fast path (FlashAttentionBackward's wrapper, which reads
    // inputs[5]=L, inputs[6]=philox_seed, inputs[7]=philox_offset by fixed
    // position) is safe to take without misinterpreting block_mask/relpos_bias
    // as philox seed/offset state.
    bwd_attrs.set(AttrKey::HasBlockMask, has_block_mask);
    bwd_attrs.set(AttrKey::HasRelposBias, has_relpos_bias);
    // Same built-in score-mod parameters as the forward so the replayed scores
    // (and thus the gradients) match the forward exactly.
    if (window_size > 0) bwd_attrs.set(AttrKey::WindowSize, static_cast<int64_t>(window_size));
    if (prefix_length > 0) bwd_attrs.set(AttrKey::PrefixLength, static_cast<int64_t>(prefix_length));
    return dispatch<OpId::FlexAttentionBackward>(bwd_inputs, bwd_attrs);
}

} // anonymous namespace

// ============================================================================
// FlashAttentionBackward
// ============================================================================

auto FlashAttentionBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    // Standalone forward helper; the apply path uses flash_attention() in ops.hpp.
    // This implementation exists for symmetry with other Function classes.
    TENZOR_CHECK(inputs.size() >= 3, "FlashAttentionBackward::forward expects at least 3 inputs (Q, K, V)");
    return {flash_attention(inputs[0], inputs[1], inputs[2],
                            scale_, causal_, dropout_p_, is_training_)};
}

auto FlashAttentionBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // grad_outputs[0] = dL/d(output) — only the output tensor is differentiable;
    // lse/seed/offset (saved auxiliaries) don't receive gradients.
    TENZOR_CHECK(!grad_outputs.empty(), "FlashAttentionBackward::backward needs >=1 grad_output");
    const Tensor& dO = grad_outputs[0];

    const auto& saved = saved_tensors();
    // Saved layout (set by apply): [Q, K, V, O, (L?), (seed?), (offset?)]
    TENZOR_CHECK(saved.size() >= 4, "FlashAttentionBackward: not enough saved tensors");
    const Tensor& Q = saved[0];
    const Tensor& K = saved[1];
    const Tensor& V = saved[2];
    const Tensor& O = saved[3];
    // A-02: L, seed and offset are each INDEPENDENTLY optional at save time, so
    // their saved positions must not be inferred from raw indices — an absent L
    // with a present seed/offset pair would otherwise alias seed into L. Decode
    // by a running index driven by the same predicates the forward used to push:
    //   * seed+offset are saved as a dropout-state PAIR (both or neither), gated
    //     on whether the forward actually sampled a Philox mask == exactly
    //     (dropout_p_ > 0 && is_training_) — the predicate run_flash_dispatch
    //     uses to request a seed. This is a persisted flag, so save/read cannot
    //     diverge.
    //   * whether L (logsumexp) was saved is then the only remaining degree of
    //     freedom and is recovered from the trailing count.
    const bool has_dropout_state = (dropout_p_ > 0.0f && is_training_);
    const bool has_lse = saved.size() > (4u + (has_dropout_state ? 2u : 0u));
    size_t idx = 4;
    Tensor L      = has_lse           ? saved[idx++] : Tensor{};
    Tensor seed   = has_dropout_state ? saved[idx++] : Tensor{};
    Tensor offset = has_dropout_state ? saved[idx++] : Tensor{};

    return try_fused_or_compose_backward(dO, Q, K, V, O, L,
                                          scale_, causal_, dropout_p_, seed, offset);
}

// ============================================================================
// FusedAttentionBackward
// ============================================================================

auto FusedAttentionBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    TENZOR_CHECK(inputs.size() >= 3, "FusedAttentionBackward::forward expects at least 3 inputs (Q, K, V)");
    return {fused_attention(inputs[0], inputs[1], inputs[2], scale_, causal_, use_cudnn_sdpa_)};
}

auto FusedAttentionBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    TENZOR_CHECK(!grad_outputs.empty(), "FusedAttentionBackward::backward needs >=1 grad_output");
    const Tensor& dO = grad_outputs[0];

    const auto& saved = saved_tensors();
    TENZOR_CHECK(saved.size() >= 4, "FusedAttentionBackward: not enough saved tensors");
    const Tensor& Q = saved[0];
    const Tensor& K = saved[1];
    const Tensor& V = saved[2];
    const Tensor& O = saved[3];
    Tensor L = (saved.size() > 4) ? saved[4] : Tensor{};

    // FusedAttention has no dropout — pass empty seed/offset.
    return try_fused_or_compose_backward(dO, Q, K, V, O, L,
                                          scale_, causal_, /*dropout_p=*/0.0f,
                                          /*seed=*/Tensor{}, /*offset=*/Tensor{});
}

// ============================================================================
// FlexAttentionBackward
// ============================================================================

auto FlexAttentionBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    TENZOR_CHECK(inputs.size() >= 3, "FlexAttentionBackward::forward expects at least 3 inputs (Q, K, V)");
    Tensor block_mask = has_block_mask_ && inputs.size() > 3 ? inputs[3].tensor() : Tensor{};
    // AUTOGRAD-R036: pass the saved score_mod parameters through — omitting
    // them silently degraded a recomputed sliding-window/prefix-lm/relpos_bias
    // forward to defaults (window_size=0, prefix_length=0, no bias), unlike
    // the sibling FlashAttentionBackward/FusedAttentionBackward::forward(),
    // which recompute with their full saved parameter set.
    return {flex_attention(inputs[0], inputs[1], inputs[2], scale_, score_mod_id_,
                           block_mask, window_size_, prefix_length_, relpos_bias_)};
}

auto FlexAttentionBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    TENZOR_CHECK(!grad_outputs.empty(), "FlexAttentionBackward::backward needs >=1 grad_output");
    const Tensor& dO = grad_outputs[0];

    const auto& saved = saved_tensors();
    TENZOR_CHECK(saved.size() >= 4, "FlexAttentionBackward: not enough saved tensors");
    const Tensor& Q = saved[0];
    const Tensor& K = saved[1];
    const Tensor& V = saved[2];
    const Tensor& O = saved[3];
    // A-01: L (logsumexp) and block_mask are each INDEPENDENTLY optional at save
    // time, so their saved positions must not be inferred from raw indices — an
    // absent L with a present block_mask would otherwise alias block_mask into L.
    // Decode by a running index driven by the same predicate the forward used to
    // push: has_block_mask_ is the persisted flag recording whether block_mask
    // was saved, so save/read cannot diverge. Whether L was saved is then the
    // only remaining degree of freedom and is recovered from the trailing count.
    const bool has_block_mask = has_block_mask_;
    const bool has_lse = saved.size() > (4u + (has_block_mask ? 1u : 0u));
    size_t idx = 4;
    Tensor L          = has_lse        ? saved[idx++] : Tensor{};
    Tensor block_mask = has_block_mask ? saved[idx++] : Tensor{};

    return try_flex_backward_or_throw(dO, Q, K, V, O, L, scale_, score_mod_id_,
                                      block_mask, window_size_, prefix_length_,
                                      relpos_bias_);
}

namespace {

// Audit B.3 — Variable-level composed attention backward.
//
// Mirrors the Tensor-level `composed_attention_backward()` math above
// (S = Q K^T scale, P = softmax(S), dV = P^T dO, dP = dO V^T,
//  dS = P * (dP - rowsum(dP*P)), dQ = dS K * scale, dK = dS^T Q * scale)
// but every operation is a Variable autograd op. Building the backward
// from differentiable Variable ops lets reverse-mode autograd compute
// the Hessian when this Function is itself reached in a higher-order
// backward call, with no need for a custom 2nd-order kernel.
//
// Dropout is incompatible with higher-order autograd (the stochastic
// mask isn't differentiable), so callers gate this path on dropout_p
// == 0. score_mod is similarly user-supplied at OpId granularity, so
// only the identity score-mod path is exposed for FlexAttention's
// higher-order entry.
auto composed_attention_backward_variable(const Variable& dO,
                                          const Variable& Q,
                                          const Variable& K,
                                          const Variable& V,
                                          double scale,
                                          bool causal) -> std::vector<Variable> {
    // V.8: double scale (the Variable form was the Tensor form's twin;
    // both now take `double` end-to-end, matching the `double scale_`
    // member on FlashAttentionBackward/FusedAttentionBackward/FlexAttentionBackward).
    auto Q_shape_span = Q.tensor().shape();
    std::vector<int64_t> orig_shape(Q_shape_span.begin(), Q_shape_span.end());
    std::vector<int64_t> K_shape(K.tensor().shape().begin(), K.tensor().shape().end());
    std::vector<int64_t> V_shape(V.tensor().shape().begin(), V.tensor().shape().end());
    bool is_4d = (orig_shape.size() == 4);

    auto reshape_3d = [&](const Variable& v) -> Variable {
        if (!is_4d) return v;
        auto s = v.tensor().shape();
        return tenzor::reshape(v, std::vector<int64_t>{s[0] * s[1], s[2], s[3]});
    };

    auto Q3 = reshape_3d(Q);
    auto K3 = reshape_3d(K);
    auto V3 = reshape_3d(V);
    auto dO3 = reshape_3d(dO);

    // S.1 — defensive negative-dim normalisation. Q3/K3/V3 are always rank-3
    // (post reshape_3d). `transpose(-1, -2)` and `sum(dim=-1)` would underflow
    // if any caller ever fed a < 2D tensor through the higher-order path; the
    // explicit normalised dims below match what the underlying op does and
    // make the intent obvious at the call site.
    auto Q3_ndim = static_cast<int64_t>(Q3.tensor().shape().size());
    auto K3_ndim = static_cast<int64_t>(K3.tensor().shape().size());
    auto V3_ndim = static_cast<int64_t>(V3.tensor().shape().size());
    if (Q3_ndim < 2 || K3_ndim < 2 || V3_ndim < 2) {
        throw std::runtime_error(
            "composed_attention_backward_variable: Q/K/V must be at least 2D after reshape_3d");
    }
    int64_t K_t_dim0 = K3_ndim - 1;
    int64_t K_t_dim1 = K3_ndim - 2;

    auto Kt = tenzor::transpose(K3, K_t_dim0, K_t_dim1);
    auto S = tenzor::bmm(Q3, Kt);
    {
        Variable scale_v(::tenzor::full({1}, static_cast<double>(scale),
                                         S.tensor().dtype(), S.tensor().device()),
                          false);
        S = S * scale_v;
    }

    if (causal) {
        // Use where(mask>0, -inf, S) instead of S + (mask * -inf).
        // The additive form computes 0 * -inf = NaN at unmasked positions
        // on backends that don't short-circuit the multiply, breaking
        // softmax. See attention-contract.md (sentinel must be -INFINITY).
        auto S_shape = S.tensor().shape();
        int64_t S_q = S_shape[1];
        int64_t S_k = S_shape[2];
        // F026: BOTTOM-RIGHT causal alignment (mask ki > qi + (S_k - S_q)) to
        // match the forward; see the Tensor-composed twin above.
        auto mask_t = ::tenzor::triu(
            ::tenzor::ones({S_q, S_k}, S.tensor().dtype(), S.tensor().device()),
            1 + (S_k - S_q));
        auto neg_inf_t = ::tenzor::full({1}, -std::numeric_limits<double>::infinity(),
                                         S.tensor().dtype(), S.tensor().device());
        Variable mask_v(mask_t, false);
        Variable neg_inf_v(neg_inf_t, false);
        S = tenzor::where(mask_v, neg_inf_v, S);
    }

    auto S_ndim = static_cast<int64_t>(S.tensor().shape().size());
    int64_t softmax_dim = S_ndim - 1;
    auto P = tenzor::softmax(S, softmax_dim);

    auto P_ndim = static_cast<int64_t>(P.tensor().shape().size());
    auto Pt = tenzor::transpose(P, P_ndim - 1, P_ndim - 2);
    auto dV3 = tenzor::bmm(Pt, dO3);

    auto Vt = tenzor::transpose(V3, V3_ndim - 1, V3_ndim - 2);
    auto dP = tenzor::bmm(dO3, Vt);

    auto dPP = dP * P;
    auto dPP_ndim = static_cast<int64_t>(dPP.tensor().shape().size());
    int64_t row_sum_dim = dPP_ndim - 1;
    auto row_sum = tenzor::sum(dPP, std::optional<int64_t>{row_sum_dim}, /*keepdim=*/true);
    auto dS = P * (dP - row_sum);

    Variable scale_v(::tenzor::full({1}, static_cast<double>(scale),
                                     dS.tensor().dtype(), dS.tensor().device()),
                      false);
    auto dQ3 = tenzor::bmm(dS, K3) * scale_v;
    auto dS_ndim = static_cast<int64_t>(dS.tensor().shape().size());
    auto dSt = tenzor::transpose(dS, dS_ndim - 1, dS_ndim - 2);
    auto dK3 = tenzor::bmm(dSt, Q3) * scale_v;

    if (is_4d) {
        dQ3 = tenzor::reshape(dQ3, orig_shape);
        dK3 = tenzor::reshape(dK3, K_shape);
        dV3 = tenzor::reshape(dV3, V_shape);
    }
    return {dQ3, dK3, dV3};
}

} // anonymous namespace

// Audit B.3 — real higher-order backward for FlashAttention.
// Falls through to the Tensor-level stub when dropout > 0 because the
// stochastic dropout mask is non-differentiable; for dropout == 0 the
// composed Variable-level backward is mathematically identical to the
// fused kernel's first-order backward and provides a real 2nd-order
// gradient via reverse-mode autograd.
auto FlashAttentionBackward::backward_with_variables(std::vector<Variable> grad_outputs)
    -> std::vector<Variable> {
    TENZOR_CHECK(!grad_outputs.empty(), "FlashAttentionBackward::backward_with_variables needs >=1 grad_output");
    const auto& saved = saved_tensors();
    TENZOR_CHECK(saved.size() >= 4,
                 "FlashAttentionBackward::backward_with_variables: not enough saved tensors");
    if (dropout_p_ > 0.0f) {
        // P.5: higher-order backward through FlashAttention with dropout>0
        // is currently a stub — the composed-ops path can't reproduce the
        // exact per-element dropout mask that the fused forward sampled
        // (Philox seed/offset aren't yet plumbed back into the Variable
        // graph), so a real 2nd-derivative would silently disagree with
        // the 1st-order kernel. Warn once and raise the structural-zero
        // error so the caller sees the unsupported configuration loudly.
        static std::once_flag warned;
        std::call_once(warned, []{
            TENZOR_LOG_WARN("FlashAttention higher-order backward with dropout>0 is currently a stub — "
                            "raising structural-zero error so this is not silently miscomputed");
        });
        TENZOR_CHECK(false,
                     "FlashAttentionBackward::backward_with_variables: higher-order backward "
                     "with dropout_p > 0 is not supported (structural zero stub).");
    }
    Variable dO = grad_outputs[0];
    // A-02: this higher-order path reads only the FIXED leading slots Q,K,V
    // (saved[0..2]); it never touches the independently-optional trailing tensors
    // (L, seed, offset), so the running-index decode used in ::backward is not
    // needed here — the positional collision cannot occur for slots 0..2.
    // audit-10 NN.1: prefer saved Variables (with grad_fn) when available so
    // the second derivative path through Q/K/V projections stays connected.
    Variable Q = has_saved_variables() ? saved_variables()[0] : Variable(saved[0], false);
    Variable K = has_saved_variables() ? saved_variables()[1] : Variable(saved[1], false);
    Variable V = has_saved_variables() ? saved_variables()[2] : Variable(saved[2], false);
    return composed_attention_backward_variable(dO, Q, K, V, scale_, causal_);
}

// Audit B.3 — real higher-order backward for FusedAttention.
// FusedAttention has no dropout and the math is identical to
// FlashAttention's no-dropout path, so we reuse the same composed
// Variable-level backward.
auto FusedAttentionBackward::backward_with_variables(std::vector<Variable> grad_outputs)
    -> std::vector<Variable> {
    TENZOR_CHECK(!grad_outputs.empty(), "FusedAttentionBackward::backward_with_variables needs >=1 grad_output");
    const auto& saved = saved_tensors();
    TENZOR_CHECK(saved.size() >= 4,
                 "FusedAttentionBackward::backward_with_variables: not enough saved tensors");
    Variable dO = grad_outputs[0];
    // audit-10 NN.1: see FlashAttentionBackward — preserve Q/K/V graph.
    Variable Q = has_saved_variables() ? saved_variables()[0] : Variable(saved[0], false);
    Variable K = has_saved_variables() ? saved_variables()[1] : Variable(saved[1], false);
    Variable V = has_saved_variables() ? saved_variables()[2] : Variable(saved[2], false);
    return composed_attention_backward_variable(dO, Q, K, V, scale_, causal_);
}

// Audit B.3 — real higher-order backward for FlexAttention.
// Only the identity score-mod (score_mod_id == 0) path has a closed-form
// Variable-level pipeline; non-trivial score mods are user-supplied
// OpIds that don't compose at Variable level, so we fall back to the
// structural-zero stub for those (and for any case where a block_mask
// was applied — the block mask is non-differentiable).
auto FlexAttentionBackward::backward_with_variables(std::vector<Variable> grad_outputs)
    -> std::vector<Variable> {
    TENZOR_CHECK(!grad_outputs.empty(), "FlexAttentionBackward::backward_with_variables needs >=1 grad_output");
    const auto& saved = saved_tensors();
    TENZOR_CHECK(saved.size() >= 4,
                 "FlexAttentionBackward::backward_with_variables: not enough saved tensors");
    if (score_mod_id_ != 0 || has_block_mask_) {
        return passthrough_stub_backward(std::move(grad_outputs));
    }
    Variable dO = grad_outputs[0];
    // A-01: reads only the FIXED leading slots Q,K,V (saved[0..2]); the
    // independently-optional trailing tensors (L, block_mask) are never read
    // here, so the running-index decode used in ::backward is unnecessary — the
    // positional collision cannot occur for slots 0..2.
    // audit-10 NN.1: see FlashAttentionBackward — preserve Q/K/V graph.
    Variable Q = has_saved_variables() ? saved_variables()[0] : Variable(saved[0], false);
    Variable K = has_saved_variables() ? saved_variables()[1] : Variable(saved[1], false);
    Variable V = has_saved_variables() ? saved_variables()[2] : Variable(saved[2], false);
    return composed_attention_backward_variable(dO, Q, K, V, scale_, /*causal=*/false);
}

// ============================================================================
// User-facing apply helpers
// ============================================================================

namespace {

// Run the FlashAttention dispatch and pad the result to a [output, lse, seed, offset]
// quadruple so the Function can save a uniform layout. Backends that don't yet
// honor the full 4-tuple (audit C1; M3-M7 fix) just return a shorter vector;
// missing entries become empty Tensors.
auto run_flash_dispatch(const Tensor& Q, const Tensor& K, const Tensor& V,
                        float scale, bool causal, float dropout_p, bool is_training)
    -> std::vector<Tensor> {
    OpAttributes attrs;
    attrs.set(AttrKey::Scale, static_cast<double>(scale));
    attrs.set(AttrKey::Causal, causal);
    attrs.set(AttrKey::DropoutP, static_cast<double>(dropout_p));
    attrs.set(AttrKey::IsTraining, is_training);
    if (dropout_p > 0.0f && is_training) {
        // Dropout determinism contract: the Philox key derives from the
        // device's default generator, so manual_seed() + identical inputs
        // reproduce the mask bit-exactly. Without this the backends invented
        // their own keys (CUDA hashed the Q data pointer; the oneAPI/Vulkan
        // composed path drew from std::random_device), so a re-seeded
        // forward was NOT reproducible.
        auto seed = static_cast<int64_t>(default_generator(Q.device()).next_seed());
        if (seed == 0) seed = 1;  // 0 means "unset" to the backend wrappers
        attrs.set(AttrKey::Seed, seed);
    }
    std::vector<Tensor> inputs = {Q, K, V};
    std::vector<Tensor> outs = dispatch<OpId::FlashAttention>(inputs, attrs);
    while (outs.size() < 4) outs.emplace_back();  // empty Tensor fillers
    return outs;
}

auto run_fused_dispatch(const Tensor& Q, const Tensor& K, const Tensor& V,
                        float scale, bool causal, bool use_cudnn_sdpa)
    -> std::vector<Tensor> {
    OpAttributes attrs;
    attrs.set(AttrKey::Scale, static_cast<double>(scale));
    attrs.set(AttrKey::Causal, causal);
    attrs.set(AttrKey::UseCudnnSdpa, use_cudnn_sdpa);
    std::vector<Tensor> inputs = {Q, K, V};
    std::vector<Tensor> outs = dispatch<OpId::FusedAttention>(inputs, attrs);
    while (outs.size() < 2) outs.emplace_back();
    return outs;
}

auto run_flex_dispatch(const Tensor& Q, const Tensor& K, const Tensor& V,
                       float scale, int64_t score_mod_id, const Tensor& block_mask,
                       int64_t window_size, int64_t prefix_length,
                       const Tensor& relpos_bias)
    -> std::vector<Tensor> {
    OpAttributes attrs;
    attrs.set(AttrKey::Scale, static_cast<double>(scale));
    attrs.set(AttrKey::ScoreModId, static_cast<int64_t>(score_mod_id));
    // Built-in score-mod parameters. WindowSize feeds ids 2/6 (sliding_window),
    // PrefixLength feeds id 5 (prefix_lm); the shared helper only reads the one
    // relevant to score_mod_id, so setting a value harmlessly no-ops otherwise.
    if (window_size > 0) attrs.set(AttrKey::WindowSize, static_cast<int64_t>(window_size));
    if (prefix_length > 0) attrs.set(AttrKey::PrefixLength, static_cast<int64_t>(prefix_length));
    const bool has_block_mask = block_mask.is_valid() && block_mask.shape().size() > 0;
    const bool has_relpos_bias = relpos_bias.is_valid() && relpos_bias.shape().size() > 0;
    // AUTOGRAD-R027: surface presence explicitly so backend wrappers can
    // decide whether their score_mod_id∈{0,1} fast path (which forwards the
    // raw input span positionally) is safe to take, instead of guessing from
    // inputs.size() — see try_flex_backward_or_throw's identical rationale.
    attrs.set(AttrKey::HasBlockMask, has_block_mask);
    attrs.set(AttrKey::HasRelposBias, has_relpos_bias);
    std::vector<Tensor> inputs = {Q, K, V};
    if (has_block_mask) inputs.push_back(block_mask);
    // relpos_bias (id 3) is appended LAST so the forward helper's contract holds
    // in both cases: inputs[3] when there is no block_mask, else inputs.back().
    if (has_relpos_bias) inputs.push_back(relpos_bias);
    std::vector<Tensor> outs = dispatch<OpId::FlexAttention>(inputs, attrs);
    while (outs.size() < 2) outs.emplace_back();
    return outs;
}

} // anonymous namespace

auto flash_attention(const Variable& Q,
                     const Variable& K,
                     const Variable& V,
                     float scale,
                     bool causal,
                     float dropout_p,
                     bool is_training) -> Variable {
    bool any_grad = Q.requires_grad() || K.requires_grad() || V.requires_grad();

    // audit-2026-05-03 / audit A.11 — Float64 path.
    //
    // Most backend FlashAttention kernels still upcast Float64 → Float32
    // internally before computing (per attention-contract.md). When the
    // dispatched kernel returns Float32-precision output but the composed
    // backward computes in Float64, Float64 gradcheck breaks. Route Float64
    // through pure Variable-level ops so forward and backward (and
    // numerical-vs-analytical) all use the same double-precision math.
    //
    // EXCEPTION: CPU now has a native Float64 FlashAttention kernel (this
    // file's CPU registry calls `flash_attention_forward` which dispatches on
    // dtype to a `double` typed kernel — no Float32 round-trip). On CPU we
    // therefore go through the fused kernel + FlashAttentionBackward like
    // Float32, gaining the cache-tiled / SIMD speedup and exercising the
    // real kernel under test rather than the composed-ops scaffold.
    //
    // audit A.11 (CUDA): CUDA now also has native Float64 forward+backward
    // kernels (src/backends/cuda/kernels/flash_attention_f64.cu) for
    // head_dim ∈ {16, 32, 48, 64, 80, 96, 128}. For other head_dims the CUDA
    // backend's FlashAttentionBackward registry falls back to a composed-ops
    // backward — that composed path already runs in double on CUDA via the
    // standard op dispatch, so the Variable-level bypass is no longer needed
    // for CUDA either.
    //
    // audit A.11 (ROCm): ROCm now also has native Float64 forward+backward
    // kernels (src/backends/rocm/kernels/flash_attention_f64.hip.cpp) for the
    // same head_dim set. The ROCm registry falls back to a composed-ops
    // backward for other head_dims (still native FP64 via the standard op
    // dispatch).
    //
    // audit A.11 (OneAPI): OneAPI now also has native Float64 forward+backward
    // kernels (src/backends/oneapi/kernels/flash_attention_f64.cpp) for the
    // same head_dim set; the registry routes Float64 inputs at supported
    // head_dims to those SYCL kernels and falls back to the composed FP64 op
    // dispatch otherwise.
    //
    // audit A.11 (Vulkan): Vulkan now also has a native Float64 fused fast
    // path (src/backends/vulkan/kernels/flash_attention_f64.comp) for
    // head_dim/head_v <= 128, gated at runtime on
    // VkPhysicalDeviceFeatures::shaderFloat64. Devices without FP64 throw
    // a clear error (project rule: no CPU fallback, no Float32 upcast).
    // Outside the fast-path constraints the composed FP64 op dispatch
    // already runs in double on Vulkan, so the Variable-level bypass is
    // no longer needed for Vulkan either.
    //
    // audit A.11 (MPS): Apple Metal has no Float64 — the Metal Shading
    // Language specification defines no `double` scalar type, and there is
    // no MSL FP64 extension. A "native Float64 FlashAttention" is therefore
    // impossible on MPS. The MPS dispatch table registers FlashAttention
    // (and FlashAttentionBackward / FusedAttention) with a wrapper that
    // throws std::runtime_error on Float64 inputs (see
    // src/backends/mps/mps_kernel_registry.mm). We list MPS in
    // backend_native_f64 deliberately so the Variable-level composed-ops
    // bypass does NOT run on MPS — that bypass would silently downcast
    // through the underlying MPS op kernels (matmul/softmax/where), which
    // is the exact "Float32 upcast workaround" the project forbids. The
    // throw at dispatch is the honest answer: users must move to a backend
    // that actually supports FP64 (CPU, CUDA, ROCm, OneAPI, or a Vulkan
    // device that advertises shaderFloat64).
    const Device::Type dev_type = Q.tensor().device().type;
    const bool backend_native_f64 =
        (dev_type == Device::Type::CPU) ||
        (dev_type == Device::Type::CUDA) ||
        (dev_type == Device::Type::ROCm) ||
        (dev_type == Device::Type::OneAPI) ||
        (dev_type == Device::Type::Vulkan) ||
        (dev_type == Device::Type::MPS);
    if (Q.tensor().dtype() == DType::Float64 && dropout_p == 0.0f
        && !backend_native_f64) {
        auto Kt = transpose(K, -1, -2);
        auto S = matmul(Q, Kt);
        auto S_shape = S.shape();
        Variable scale_var(::tenzor::full({1}, static_cast<double>(scale),
            S.tensor().dtype(), S.tensor().device()), false);
        S = S * scale_var;
        if (causal) {
            int64_t S_q = S_shape[S_shape.size() - 2];
            int64_t S_k = S_shape[S_shape.size() - 1];
            // Use where(mask > 0, -inf, S) instead of S + (mask * -inf).
            // The additive form computes 0 * -inf = NaN at masked positions
            // and propagates NaN through softmax for certain dtype/backend
            // combinations. where() is the contract-compliant pattern.
            // F026: BOTTOM-RIGHT causal alignment (mask ki > qi + (S_k - S_q))
            // to match the forward; the previous top-left offset (1) diverged
            // whenever S_q != S_k (causal cross-attention / cached-KV decode),
            // yielding wrong dQ/dK/dV on backends that hit this Float64 bypass.
            auto mask_t = ::tenzor::triu(::tenzor::ones({S_q, S_k},
                S.tensor().dtype(), S.tensor().device()), 1 + (S_k - S_q));
            auto neg_inf_t = ::tenzor::full({1},
                -std::numeric_limits<double>::infinity(),
                S.tensor().dtype(), S.tensor().device());
            Variable mask_var(mask_t, false);
            Variable neg_inf_var(neg_inf_t, false);
            S = where(mask_var, neg_inf_var, S);
        }
        auto P = softmax(S, -1);
        return matmul(P, V);
    }

    if (!any_grad || !is_grad_enabled()) {
        // No autograd path needed — call dispatch raw.
        auto outs = run_flash_dispatch(Q.tensor(), K.tensor(), V.tensor(),
                                       scale, causal, dropout_p, is_training);
        return Variable(outs[0], false);
    }

    auto outs = run_flash_dispatch(Q.tensor(), K.tensor(), V.tensor(),
                                   scale, causal, dropout_p, is_training);
    const Tensor& O_t = outs[0];
    const Tensor& L_t = outs[1];
    const Tensor& seed_t = outs[2];
    const Tensor& offset_t = outs[3];

    auto grad_fn = std::make_shared<FlashAttentionBackward>(scale, causal, dropout_p, is_training);
    // Saved layout matches FlashAttentionBackward::backward expectation:
    // [Q, K, V, O, L?, seed?, offset?]. The autograd graph machinery balks at
    // uninitialized Tensor in save_for_backward, so only push what's actually
    // valid. Backward checks saved.size() to know which auxiliaries arrived.
    std::vector<Tensor> saved = {Q.tensor(), K.tensor(), V.tensor(), O_t};
    if (L_t.is_valid())      saved.push_back(L_t);
    if (seed_t.is_valid())   saved.push_back(seed_t);
    if (offset_t.is_valid()) saved.push_back(offset_t);
    grad_fn->save_for_backward(saved);
    // audit-10 NN.1: also save Q/K/V as Variables so the higher-order
    // backward composes through their grad_fn chains instead of severing
    // the graph by wrapping the raw saved tensors as non-grad Variables.
    grad_fn->save_variables_for_backward({Q, K, V});
    grad_fn->set_next_functions({Q.grad_fn(), K.grad_fn(), V.grad_fn()});
    grad_fn->set_input_variables({Q, K, V});

    Variable out_var(O_t, true);
    out_var.set_grad_fn(grad_fn);
    return out_var;
}

auto fused_attention(const Variable& Q,
                     const Variable& K,
                     const Variable& V,
                     float scale,
                     bool causal,
                     bool use_cudnn_sdpa) -> Variable {
    bool any_grad = Q.requires_grad() || K.requires_grad() || V.requires_grad();
    if (!any_grad || !is_grad_enabled()) {
        auto outs = run_fused_dispatch(Q.tensor(), K.tensor(), V.tensor(),
                                       scale, causal, use_cudnn_sdpa);
        return Variable(outs[0], false);
    }

    auto outs = run_fused_dispatch(Q.tensor(), K.tensor(), V.tensor(),
                                   scale, causal, use_cudnn_sdpa);
    const Tensor& O_t = outs[0];
    const Tensor& L_t = outs[1];

    auto grad_fn = std::make_shared<FusedAttentionBackward>(scale, causal, use_cudnn_sdpa);
    std::vector<Tensor> saved = {Q.tensor(), K.tensor(), V.tensor(), O_t};
    if (L_t.is_valid()) saved.push_back(L_t);
    grad_fn->save_for_backward(saved);
    // audit-10 NN.1: see flash_attention — preserve graph through Q/K/V.
    grad_fn->save_variables_for_backward({Q, K, V});
    grad_fn->set_next_functions({Q.grad_fn(), K.grad_fn(), V.grad_fn()});
    grad_fn->set_input_variables({Q, K, V});

    Variable out_var(O_t, true);
    out_var.set_grad_fn(grad_fn);
    return out_var;
}

auto flex_attention(const Variable& Q,
                    const Variable& K,
                    const Variable& V,
                    float scale,
                    int64_t score_mod_id,
                    const Tensor& block_mask,
                    int64_t window_size,
                    int64_t prefix_length,
                    const Tensor& relpos_bias) -> Variable {
    bool any_grad = Q.requires_grad() || K.requires_grad() || V.requires_grad();
    if (!any_grad || !is_grad_enabled()) {
        auto outs = run_flex_dispatch(Q.tensor(), K.tensor(), V.tensor(),
                                      scale, score_mod_id, block_mask,
                                      window_size, prefix_length, relpos_bias);
        return Variable(outs[0], false);
    }

    auto outs = run_flex_dispatch(Q.tensor(), K.tensor(), V.tensor(),
                                  scale, score_mod_id, block_mask,
                                  window_size, prefix_length, relpos_bias);
    const Tensor& O_t = outs[0];
    const Tensor& L_t = outs[1];

    bool has_block_mask = block_mask.is_valid() && block_mask.shape().size() > 0;
    // Carry the built-in score-mod parameters (WindowSize/PrefixLength/relpos
    // bias) into the backward node so the replayed scores match the forward.
    auto grad_fn = std::make_shared<FlexAttentionBackward>(
        scale, score_mod_id, has_block_mask, window_size, prefix_length,
        relpos_bias);
    std::vector<Tensor> saved = {Q.tensor(), K.tensor(), V.tensor(), O_t};
    if (L_t.is_valid()) saved.push_back(L_t);
    if (has_block_mask) saved.push_back(block_mask);
    grad_fn->save_for_backward(saved);
    // audit-10 NN.1: see flash_attention — preserve graph through Q/K/V.
    grad_fn->save_variables_for_backward({Q, K, V});
    grad_fn->set_next_functions({Q.grad_fn(), K.grad_fn(), V.grad_fn()});
    grad_fn->set_input_variables({Q, K, V});

    Variable out_var(O_t, true);
    out_var.set_grad_fn(grad_fn);
    return out_var;
}

} // namespace tenzor
