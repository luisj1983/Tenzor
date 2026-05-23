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
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/philox_dropout.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/utils/log.hpp"

#include <mutex>
#include <stdexcept>
#include <vector>

namespace tenzor {

namespace {

// Host-portable Philox4x32-10 (matches CPU flash_attention.cpp constants /
// algorithm). Used by the composed backward to replay the forward dropout
// mask using the saved seed. Counter convention matches the CUDA/ROCm flash
// kernels: (batch_head, query_idx, kv_pos, 0) so backward + forward
// produce identical masks within those backends. CPU forward uses a
// different convention (b, h, qi, ki), so for CPU-saved seeds the composed
// backward isn't bit-exact — but the CPU backward kernel handles its own
// replay (per src/backends/cpu/kernels/flash_attention.cpp), so this code
// path only runs for GPU-saved seeds where the conventions match.
namespace {
inline void host_philox_round(uint32_t ctr[4], const uint32_t key[2]) {
    constexpr uint64_t M0 = 0xD2511F53ULL;
    constexpr uint64_t M1 = 0xCD9E8D57ULL;
    uint64_t prod0 = M0 * static_cast<uint64_t>(ctr[0]);
    uint64_t prod1 = M1 * static_cast<uint64_t>(ctr[2]);
    uint32_t hi0 = static_cast<uint32_t>(prod0 >> 32);
    uint32_t lo0 = static_cast<uint32_t>(prod0);
    uint32_t hi1 = static_cast<uint32_t>(prod1 >> 32);
    uint32_t lo1 = static_cast<uint32_t>(prod1);
    uint32_t new0 = hi1 ^ ctr[1] ^ key[0];
    uint32_t new1 = lo1;
    uint32_t new2 = hi0 ^ ctr[3] ^ key[1];
    uint32_t new3 = lo0;
    ctr[0] = new0; ctr[1] = new1; ctr[2] = new2; ctr[3] = new3;
}
inline float host_philox_uniform(uint32_t batch_head, uint32_t query_idx,
                                  uint32_t kv_pos, uint32_t rng_seed) {
    uint32_t ctr[4] = {batch_head, query_idx, kv_pos, 0};
    uint32_t k[2] = {rng_seed, rng_seed ^ 0x1BD11BDAU};
    constexpr uint32_t W0 = 0x9E3779B9U;
    constexpr uint32_t W1 = 0xBB67AE85U;
    for (int r = 0; r < 10; ++r) {
        host_philox_round(ctr, k);
        if (r < 9) { k[0] += W0; k[1] += W1; }
    }
    return (static_cast<float>(ctr[0] >> 8)) * (1.0f / 16777216.0f);
}
}  // anonymous namespace

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
                                 float scale,
                                 bool causal,
                                 float dropout_p = 0.0f,
                                 uint32_t rng_seed = 0u) -> std::vector<Tensor> {
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
        // Apply -INFINITY mask above diagonal via where(mask>0, -inf, S) —
        // replaces the previous additive form `S + (mask * -inf)` which
        // computes 0 * -inf = NaN at unmasked positions on backends that
        // don't short-circuit the multiply. See attention-contract.md
        // (sentinel must be -INFINITY; never -1e9 / -1e30 — FP16 saturates
        // and leaks gradient mass).
        int64_t S_q = S.shape()[1];
        int64_t S_k = S.shape()[2];
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

    // Replay forward Philox dropout on P (audit M4-rem follow-up, A.11):
    // Build the same dropout mask the forward applied using the shared
    // `philox_dropout_mask` op (same counter convention `(bh, qi, ki, 0)`
    // as host_philox_uniform / forward replay). The mask is generated on
    // CPU then ferried to P's device — replaces the previous host triple-
    // for-loop, which was effectively a hand-rolled re-implementation of
    // philox_dropout_mask. The mask op already returns scale=1/(1-p) for
    // kept positions and 0 for dropped, which is exactly the
    // multiplicative correction we need on P (inverted dropout).
    if (dropout_p > 0.0f && rng_seed != 0u) {
        auto P_shape = P.shape();
        std::vector<int64_t> mask_shape(P_shape.begin(), P_shape.end());
        Tensor mask_cpu = philox_dropout_mask(
            mask_shape,
            static_cast<double>(dropout_p),
            static_cast<uint64_t>(rng_seed),
            /*offset=*/0,
            DType::Float32);
        Tensor mask_dev = mask_cpu.to(P.device()).to(P.dtype());
        P = P * mask_dev;
    }

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
    // Read seed (if any) for the composed-fallback dropout replay.
    uint32_t seed_for_replay = 0u;
    if (philox_seed.is_valid() && philox_seed.numel() > 0) {
        // Move to CPU to read scalar; tiny single-int copy.
        Tensor seed_cpu = philox_seed.cpu();
        if (seed_cpu.dtype() == DType::Int64) {
            seed_for_replay = static_cast<uint32_t>(seed_cpu.data<int64_t>()[0]);
        } else if (seed_cpu.dtype() == DType::Int32) {
            seed_for_replay = static_cast<uint32_t>(seed_cpu.data<int32_t>()[0]);
        }
    }

    if (!L.is_valid() || L.shape().size() == 0) {
        // No saved logsumexp — backward must recompute from scratch.
        // If dropout was applied in forward, replay the mask using the
        // saved seed so dV/dQ/dK match the masked attention pattern.
        return composed_attention_backward(dO, Q, K, V, scale, causal,
                                            dropout_p, seed_for_replay);
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
        return composed_attention_backward(dO, Q, K, V, scale, causal,
                                            dropout_p, seed_for_replay);
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
                                          float scale,
                                          bool causal) -> std::vector<Variable> {
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

    auto Kt = tenzor::transpose(K3, -1, -2);
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
        auto mask_t = ::tenzor::triu(
            ::tenzor::ones({S_q, S_k}, S.tensor().dtype(), S.tensor().device()),
            1 + (S_k - S_q));
        auto neg_inf_t = ::tenzor::full({1}, -std::numeric_limits<double>::infinity(),
                                         S.tensor().dtype(), S.tensor().device());
        Variable mask_v(mask_t, false);
        Variable neg_inf_v(neg_inf_t, false);
        S = tenzor::where(mask_v, neg_inf_v, S);
    }

    auto P = tenzor::softmax(S, -1);

    auto Pt = tenzor::transpose(P, -1, -2);
    auto dV3 = tenzor::bmm(Pt, dO3);

    auto Vt = tenzor::transpose(V3, -1, -2);
    auto dP = tenzor::bmm(dO3, Vt);

    auto dPP = dP * P;
    auto row_sum = tenzor::sum(dPP, std::optional<int64_t>{-1}, /*keepdim=*/true);
    auto dS = P * (dP - row_sum);

    Variable scale_v(::tenzor::full({1}, static_cast<double>(scale),
                                     dS.tensor().dtype(), dS.tensor().device()),
                      false);
    auto dQ3 = tenzor::bmm(dS, K3) * scale_v;
    auto dSt = tenzor::transpose(dS, -1, -2);
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
    Variable Q(saved[0], false);
    Variable K(saved[1], false);
    Variable V(saved[2], false);
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
    Variable Q(saved[0], false);
    Variable K(saved[1], false);
    Variable V(saved[2], false);
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
    Variable Q(saved[0], false);
    Variable K(saved[1], false);
    Variable V(saved[2], false);
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
