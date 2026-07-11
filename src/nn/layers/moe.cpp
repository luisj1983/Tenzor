/**
 * @file moe.cpp
 * @brief Mixture of Experts implementation
 */

#include "tenzor/nn/layers/moe.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/nn/utils/variable_cast.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/jit/tracer.hpp"
#include <cmath>
#include <tuple>

namespace tenzor {
namespace nn {

MixtureOfExperts::MixtureOfExperts(int64_t input_dim, int64_t hidden_dim,
                                     int64_t num_experts, int64_t top_k,
                                     double capacity_factor,
                                     double aux_loss_weight,
                                     double dropout_p)
    : input_dim_(input_dim), hidden_dim_(hidden_dim),
      num_experts_(num_experts), top_k_(top_k),
      capacity_factor_(capacity_factor),
      aux_loss_weight_(aux_loss_weight) {

    // Routing selects top_k experts per token; it must lie in [1, num_experts].
    // top_k < 1 routes to nothing; top_k > num_experts overruns the topk call.
    if (top_k < 1 || top_k > num_experts) {
        throw std::invalid_argument(
            "MixtureOfExperts: top_k must satisfy 1 <= top_k <= num_experts (got top_k=" +
            std::to_string(top_k) + ", num_experts=" + std::to_string(num_experts) + ")");
    }

    // Router: maps input to expert logits
    router_ = std::make_shared<Linear>(input_dim, num_experts, /*bias=*/false);
    register_module("router", router_);

    // Expert FFNs: each expert has up + down projections
    for (int64_t i = 0; i < num_experts; ++i) {
        auto up = std::make_shared<Linear>(input_dim, hidden_dim, /*bias=*/false);
        auto down = std::make_shared<Linear>(hidden_dim, input_dim, /*bias=*/false);
        register_module("expert_up_" + std::to_string(i), up);
        register_module("expert_down_" + std::to_string(i), down);
        up_.push_back(std::move(up));
        down_.push_back(std::move(down));
    }

    if (dropout_p > 0.0) {
        dropout_ = std::make_shared<Dropout>(dropout_p);
        register_module("dropout", dropout_);
    }
}

auto MixtureOfExperts::forward_impl(const Variable& input) -> Variable {
    auto [output, _] = forward_with_loss(input);
    return output;
}

auto MixtureOfExperts::forward_with_loss(const Variable& input)
    -> std::pair<Variable, Variable> {
    // JIT-R050: MoE's per-expert routing is fundamentally data-dependent —
    // which experts fire, how many tokens each gets (M = routed.shape()[0]
    // below), and which tokens get dropped to capacity are all host-side
    // decisions made from the ACTUAL routing output of THIS call. A trace
    // captures exactly one input's routing outcome and bakes it in as
    // static graph structure; every future compiled-and-replayed input
    // would be forced through that frozen routing regardless of its own
    // actual routing decision — a silent wrong answer, not a crash. No
    // dedicated MoE OpType exists to represent dynamic per-expert dispatch
    // (that needs real JIT-aware dynamic-shape primitives — a new feature,
    // not a point fix). Refuse loudly instead of silently compiling
    // something wrong, matching the established "can't be safely captured
    // mid-trace" contract used elsewhere (e.g. autograd::spmm's
    // CSR-layout guard in src/autograd/ops.cpp).
    if (::tenzor::jit::Tracer::get_instance().is_tracing()) {
        throw std::runtime_error(
            "MixtureOfExperts::forward: cannot be traced by @tz.jit — "
            "per-expert routing (which experts fire, how many tokens each "
            "gets, capacity-based token dropping) is data-dependent and "
            "would be permanently frozen to this trace call's routing "
            "decision, silently producing wrong output for every other "
            "input. Run this module eagerly (outside a traced region).");
    }

    // input: [..., input_dim]. Flatten to [N, input_dim] for routing.
    //
    // Followup #21 fix: previous version used raw tensor ops throughout this
    // function (Variable(reshape(input.tensor()), ...), `flat_t * mask`,
    // `output_t = output_t + weighted` with Tensor accumulator, etc.) which
    // dropped the autograd graph between input and output. Backward then
    // populated nothing on input.grad. Rewrite uses Variable-level ops for
    // every value-carrying step; mask construction stays at the tensor level
    // because mask values are non-differentiable indicator tensors.
    auto orig_shape = input.shape();
    int64_t ndim = orig_shape.size();
    int64_t N = 1;
    for (int64_t i = 0; i < ndim - 1; ++i) N *= orig_shape[i];
    int64_t D = orig_shape[ndim - 1];

    // Flatten input through Variable-level reshape so the graph is preserved.
    auto flat = tenzor::reshape(input, std::vector<int64_t>{N, D});

    // Router logits: [N, num_experts]
    auto logits = router_->forward(flat);

    // Softmax over experts (Variable-level — gradient flows back to logits).
    auto probs_v = nn::softmax(logits, /*dim=*/1);
    Tensor probs_t = probs_v.tensor();

    // Top-k selection — non-differentiable, runs on the tensor probs_t.
    auto topk_result = tenzor::topk(probs_t, top_k_, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
    auto topk_vals = std::get<0>(topk_result);
    auto topk_idx = std::get<1>(topk_result);

    // Normalize top-k weights (still tensor-level — these are routing
    // coefficients selected by argmax-style logic, not learned through the
    // expert path; load balancing flows separately through aux_loss).
    auto topk_sum = tenzor::sum(topk_vals, /*dim=*/1, /*keepdim=*/true);
    auto topk_weights = topk_vals / topk_sum;

    // On-graph combine weights: the gate that multiplies expert outputs must be
    // FULLY differentiable w.r.t. the router logits — BOTH the numerator (probs)
    // AND the renormalisation denominator S = sum of the selected probs. The
    // previous code detached the denominator (route_norm = sel_mask / topk_sum
    // with topk_sum a constant), so combine_v = probs * (1/S) dropped the
    // -probs/S^2 quotient term of w = probs/S. That trained the router on a
    // fabricated signal and, for top_k == 1, injected a spurious grad*(1/probs_max)
    // where the gate is identically 1 and MUST carry zero gradient.
    //
    // Only the SELECTION (argmax-style top-k membership) is non-differentiable;
    // encode it as a constant 0/1 mask, then normalise ON-GRAPH so both numerator
    // and denominator flow gradient:
    //   masked[n,e]    = sel_mask[n,e] * probs_v[n,e]   (zero on non-selected)
    //   S[n]           = sum_e(masked[n,e])             (Variable-level)
    //   combine_v[n,e] = masked[n,e] / S[n]             (differentiable gate)
    // For top_k == 1 this is identically 1 on the selected expert with zero grad.
    Tensor sel_mask = tenzor::zeros({N, num_experts_}, probs_t.dtype(),
                                    input.tensor().device());
    Tensor ones_nk = tenzor::full({N, top_k_}, 1.0, probs_t.dtype(),
                                  input.tensor().device());
    sel_mask = tenzor::scatter(sel_mask, /*dim=*/1, topk_idx, ones_nk);  // 1 at selected
    Variable sel_mask_v(sel_mask, /*requires_grad=*/false);
    Variable masked_probs = probs_v * sel_mask_v;                          // [N, E]
    Variable S_v = tenzor::sum(masked_probs, /*dim=*/1, /*keepdim=*/true);  // [N, 1], on-graph
    Variable combine_v = masked_probs / S_v;                              // differentiable gate

    // Variable accumulator so the addition lands on the autograd graph.
    Variable output = Variable(
        tenzor::zeros({N, D}, input.tensor().dtype(), input.tensor().device()),
        input.requires_grad());

    for (int64_t e = 0; e < num_experts_; ++e) {
        // Per-row combined routing weight for expert e: sum over the top_k slots
        // of (this row routes to e) * its routing coefficient. Non-routed rows
        // are exactly 0. Routing weights are constants (load balancing flows via
        // aux_loss), so this stays at the Tensor level.
        Tensor w_e = tenzor::zeros({N}, input.tensor().dtype(),
                                   input.tensor().device());
        for (int64_t k = 0; k < top_k_; ++k) {
            // JIT-R050: raw Tensor::slice()/squeeze() METHOD calls never go
            // through dispatch() (tenzor::slice(Tensor,...)/squeeze(Tensor,...)
            // are themselves thin wrappers around the same raw methods, per
            // JIT-R031's identical gotcha — NOT dispatched either). Route
            // through the autograd Variable-level slice/squeeze, which IS
            // dispatch()-routed (and hence tracer-visible) even when wrapped
            // requires_grad=false, matching this file's existing
            // Variable(sel_mask,false)-style idiom for non-differentiable
            // index/coefficient tensors.
            auto idx_col = tenzor::squeeze(
                tenzor::slice(Variable(topk_idx, false), 1, k, k + 1), 1).tensor();        // [N] int
            auto weight_col = tenzor::squeeze(
                tenzor::slice(Variable(topk_weights, false), 1, k, k + 1), 1).tensor(); // [N]
            auto e_scalar = tenzor::full({1}, static_cast<double>(e),
                                         idx_col.dtype(), idx_col.device());
            auto mask_f = tenzor::eq(idx_col, e_scalar).to(input.tensor().dtype());
            w_e = w_e + mask_f * weight_col;
        }

        // Rows routed to expert e (and only those). nonzero materializes the
        // count, so M is known host-side without a separate reduction/sync.
        Tensor routed = tenzor::nonzero(w_e);   // [M, 1] int64
        if (routed.shape()[0] == 0) continue;   // expert unused for this batch
        // Switch-Transformer per-expert capacity: an expert processes at most
        //   expert_capacity = ceil(capacity_factor * N * top_k / num_experts)
        // tokens. Overflow tokens (highest original row index first, since
        // nonzero() returns ascending indices) are DROPPED — they receive no
        // contribution from this expert, matching fixed-buffer MoE semantics.
        // capacity_factor_ <= 0 disables the cap (process every routed token).
        if (capacity_factor_ > 0.0) {
            int64_t expert_capacity = static_cast<int64_t>(std::ceil(
                capacity_factor_ * static_cast<double>(N) *
                static_cast<double>(top_k_) / static_cast<double>(num_experts_)));
            if (expert_capacity < 1) expert_capacity = 1;
            if (routed.shape()[0] > expert_capacity) {
                // Make the dropped-token set deterministic across backends: keep
                // the LOWEST original row indices regardless of nonzero()'s
                // per-backend ordering (previously narrow() relied on nonzero()
                // returning ascending indices, so a backend with a different
                // ordering dropped a different set of tokens — both the forward
                // output and which tokens receive zero grad then diverge).
                auto [routed_sorted, routed_perm] =
                    tenzor::sort(routed, /*dim=*/0, /*descending=*/false);
                (void)routed_perm;
                routed = tenzor::narrow(routed_sorted, 0, 0, expert_capacity);
            }
        }
        const int64_t M = routed.shape()[0];
        Tensor routed_idx = tenzor::reshape(routed, std::vector<int64_t>{M});

        // Gather ONLY the routed rows and run the expert FFN on that subset —
        // O(top_k * N) work total instead of O(num_experts * N). index_select is
        // autograd-aware (grad: index_add), so gradients flow back to `flat`.
        Variable sub = tenzor::index_select(flat, 0, routed_idx);  // [M, D]
        auto hidden = up_[e]->forward(sub);
        hidden = relu(hidden);
        if (dropout_) hidden = dropout_->forward(hidden);
        auto expert_out = down_[e]->forward(hidden);               // [M, D]

        // Weight each routed row by its routing coefficient — pulled ON-GRAPH
        // from combine_v (column e) so gradients flow back to the router. Both
        // index_select ops are autograd-aware, so the gate stays differentiable.
        Tensor col_idx = tenzor::full({1}, static_cast<double>(e), DType::Int64,
                                      input.tensor().device());
        auto w_col_v = tenzor::index_select(combine_v, /*dim=*/1, col_idx);  // [N, 1]
        w_col_v = tenzor::reshape(w_col_v, std::vector<int64_t>{N});         // [N]
        auto w_sub_var = tenzor::index_select(w_col_v, 0, routed_idx);       // [M]
        w_sub_var = tenzor::reshape(w_sub_var, std::vector<int64_t>{M, 1});  // [M, 1]
        auto weighted = expert_out * w_sub_var;                             // [M, D]

        // Scatter-add the weighted outputs back to their original rows.
        // scatter_add is autograd-aware (grad: identity for `output`, gather for
        // `weighted`), keeping the graph connected end to end.
        Tensor idx_md = tenzor::reshape(routed_idx, std::vector<int64_t>{M, 1});
        idx_md = tenzor::expand(idx_md, std::vector<int64_t>{M, D}).contiguous();
        output = tenzor::scatter_add(output, 0, idx_md, weighted);
    }

    // Reshape back through Variable-level reshape.
    output = tenzor::reshape(output, std::vector<int64_t>(orig_shape.begin(), orig_shape.end()));

    // Auxiliary load balancing loss (separate path; doesn't need to flow
    // through the expert outputs to remain useful, but it does need to flow
    // through `probs_v` so the router learns).
    // JIT-R050: same raw-Tensor-method dispatch-bypass fix as the loop above.
    auto top1_idx = tenzor::squeeze(
        tenzor::slice(Variable(topk_idx, false), 1, 0, 1), 1).tensor();

    auto expert_counts = tenzor::zeros({num_experts_}, DType::Float32, input.tensor().device());
    for (int64_t e = 0; e < num_experts_; ++e) {
        auto emask = tenzor::eq(top1_idx, tenzor::full({1}, static_cast<double>(e),
                                top1_idx.dtype(), top1_idx.device()));
        auto cnt = tenzor::sum(emask.to(DType::Float32)).reshape({1});
        auto sidx = tenzor::full({1}, static_cast<double>(e), DType::Int64, cnt.device());
        expert_counts = tenzor::scatter(expert_counts, 0, sidx, cnt);
    }

    // audit-8 GG.4: keep probs_v as a Variable so the router's gradient
    // path remains connected through the aux loss.  The previous code
    // extracted probs_t via probs_v.tensor(), severing autograd; the
    // resulting aux_loss had no grad_fn back to router_->weight/bias,
    // so the load-balance signal silently never updated the router.
    auto N_scalar = tenzor::full({1}, static_cast<double>(N), DType::Float32, input.tensor().device());
    auto freq_t = expert_counts / N_scalar;                              // [num_experts], no grad
    auto scale_t = tenzor::full({1}, static_cast<double>(num_experts_) * aux_loss_weight_,
                                DType::Float32, input.tensor().device());

    // Variable-level mean over the batch dim — preserves grad_fn to logits.
    auto probs_mean_v = tenzor::mean(probs_v, /*dim=*/0, /*keepdim=*/false);
    auto probs_mean_f32 = nn::variable_cast(probs_mean_v, DType::Float32);

    // freq and scale are constants (no gradient flows through them).
    auto freq_v = Variable(freq_t, /*requires_grad=*/false);
    auto scale_v = Variable(scale_t, /*requires_grad=*/false);

    auto aux_loss = tenzor::sum(freq_v * probs_mean_f32) * scale_v;

    return std::make_pair(output, aux_loss);
}

} // namespace nn
} // namespace tenzor
