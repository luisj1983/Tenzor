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
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/utils/error.hpp"

#include <stdexcept>
#include <vector>

namespace tenzor {

namespace {

// Composed-ops fallback for FlashAttention backward when L (and seed/offset)
// are not available. This is mathematically equivalent but uses O(N^2) memory.
//
// Math:
//   S = Q @ K^T * scale  + (causal mask if applicable)
//   P = softmax(S, dim=-1)
//   dV = P^T @ dO
//   dP = dO @ V^T
//   dS = P * (dP - sum(dP * P, dim=-1, keepdim))
//   dQ = dS @ K * scale
//   dK = dS^T @ Q * scale
auto composed_attention_backward(const Tensor& dO,
                                 const Tensor& Q,
                                 const Tensor& K,
                                 const Tensor& V,
                                 float scale,
                                 bool causal) -> std::vector<Tensor> {
    // Q, K, V, dO all have shape [B, H, S, D] or [B*H, S, D]. We work on
    // 3D for bmm convenience by collapsing leading dims if 4D. shape() returns
    // a span; convert to vector for the reshape API.
    auto Q_shape_span = Q.shape();
    std::vector<int64_t> orig_shape(Q_shape_span.begin(), Q_shape_span.end());
    std::vector<int64_t> K_shape(K.shape().begin(), K.shape().end());
    std::vector<int64_t> V_shape(V.shape().begin(), V.shape().end());
    bool is_4d = (orig_shape.size() == 4);

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
        // Apply -INFINITY mask above diagonal. Per attention-contract.md sentinels:
        // never -1e9 (FP16 saturates to -65504, leaks gradient mass).
        int64_t S_q = S.shape()[1];
        int64_t S_k = S.shape()[2];
        Tensor mask = tenzor::triu(tenzor::ones({S_q, S_k}, S.dtype(), S.device()),
                                   1 + (S_k - S_q));
        Tensor neg_inf = tenzor::full({1}, -std::numeric_limits<float>::infinity(),
                                      S.dtype(), S.device());
        S = S + (mask * neg_inf);
    }

    // P = softmax(S, dim=-1) — inlined stable softmax (no Tensor overload exists;
    // wrapping in Variable would needlessly allocate autograd nodes for a no-grad
    // recomputation in a backward pass).
    Tensor S_max = tenzor::max(S, /*dim=*/std::optional<int64_t>{-1}, /*keepdim=*/true);
    Tensor S_centered = S - S_max;
    Tensor S_exp = tenzor::exp(S_centered);
    Tensor S_exp_sum = tenzor::sum(S_exp, /*dim=*/std::optional<int64_t>{-1}, /*keepdim=*/true);
    Tensor P = S_exp / S_exp_sum;

    // dV = P^T @ dO
    Tensor Pt = tenzor::transpose(P, -1, -2);
    Tensor dV3 = tenzor::bmm(Pt, dO3);

    // dP = dO @ V^T
    Tensor Vt = tenzor::transpose(V3, -1, -2);
    Tensor dP = tenzor::bmm(dO3, Vt);

    // dS = P * (dP - sum(dP * P, dim=-1, keepdim))
    Tensor dPP = dP * P;
    Tensor row_sum = tenzor::sum(dPP, /*dim=*/std::optional<int64_t>{-1}, /*keepdim=*/true);
    Tensor dS = P * (dP - row_sum);

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
                                   float scale,
                                   bool causal,
                                   float dropout_p,
                                   const Tensor& philox_seed,
                                   const Tensor& philox_offset) -> std::vector<Tensor> {
    if (!L.is_valid() || L.shape().size() == 0) {
        // No saved logsumexp — backward must recompute from scratch.
        return composed_attention_backward(dO, Q, K, V, scale, causal);
    }
    try {
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
    } catch (const std::exception&) {
        // Fall through — backend doesn't support this combo.
        return composed_attention_backward(dO, Q, K, V, scale, causal);
    }
}

// Same defensive dispatch path for FlexAttention backward.
auto try_flex_backward_or_throw(const Tensor& dO,
                                const Tensor& Q,
                                const Tensor& K,
                                const Tensor& V,
                                const Tensor& O,
                                const Tensor& L,
                                float scale,
                                int64_t score_mod_id,
                                const Tensor& block_mask) -> std::vector<Tensor> {
    std::vector<Tensor> bwd_inputs = {dO, Q, K, V, O};
    if (L.is_valid() && L.shape().size() > 0) bwd_inputs.push_back(L);
    if (block_mask.is_valid() && block_mask.shape().size() > 0) bwd_inputs.push_back(block_mask);
    OpAttributes bwd_attrs;
    bwd_attrs.set(AttrKey::Scale, static_cast<double>(scale));
    bwd_attrs.set(AttrKey::ScoreModId, static_cast<int64_t>(score_mod_id));
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
    Tensor L = (saved.size() > 4) ? saved[4] : Tensor{};
    Tensor seed = (saved.size() > 5) ? saved[5] : Tensor{};
    Tensor offset = (saved.size() > 6) ? saved[6] : Tensor{};

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
    return {flex_attention(inputs[0], inputs[1], inputs[2], scale_, score_mod_id_, block_mask)};
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
    Tensor L = (saved.size() > 4) ? saved[4] : Tensor{};
    Tensor block_mask = (saved.size() > 5) ? saved[5] : Tensor{};

    return try_flex_backward_or_throw(dO, Q, K, V, O, L, scale_, score_mod_id_, block_mask);
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
                       float scale, int64_t score_mod_id, const Tensor& block_mask)
    -> std::vector<Tensor> {
    OpAttributes attrs;
    attrs.set(AttrKey::Scale, static_cast<double>(scale));
    attrs.set(AttrKey::ScoreModId, static_cast<int64_t>(score_mod_id));
    std::vector<Tensor> inputs = {Q, K, V};
    if (block_mask.is_valid() && block_mask.shape().size() > 0) inputs.push_back(block_mask);
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
                    const Tensor& block_mask) -> Variable {
    bool any_grad = Q.requires_grad() || K.requires_grad() || V.requires_grad();
    if (!any_grad || !is_grad_enabled()) {
        auto outs = run_flex_dispatch(Q.tensor(), K.tensor(), V.tensor(),
                                      scale, score_mod_id, block_mask);
        return Variable(outs[0], false);
    }

    auto outs = run_flex_dispatch(Q.tensor(), K.tensor(), V.tensor(),
                                  scale, score_mod_id, block_mask);
    const Tensor& O_t = outs[0];
    const Tensor& L_t = outs[1];

    bool has_block_mask = block_mask.is_valid() && block_mask.shape().size() > 0;
    auto grad_fn = std::make_shared<FlexAttentionBackward>(scale, score_mod_id, has_block_mask);
    std::vector<Tensor> saved = {Q.tensor(), K.tensor(), V.tensor(), O_t};
    if (L_t.is_valid()) saved.push_back(L_t);
    if (has_block_mask) saved.push_back(block_mask);
    grad_fn->save_for_backward(saved);
    grad_fn->set_next_functions({Q.grad_fn(), K.grad_fn(), V.grad_fn()});
    grad_fn->set_input_variables({Q, K, V});

    Variable out_var(O_t, true);
    out_var.set_grad_fn(grad_fn);
    return out_var;
}

} // namespace tenzor
