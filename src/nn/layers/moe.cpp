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

    // Variable accumulator so the addition lands on the autograd graph.
    Variable output = Variable(
        tenzor::zeros({N, D}, input.tensor().dtype(), input.tensor().device()),
        input.requires_grad());

    for (int64_t e = 0; e < num_experts_; ++e) {
        for (int64_t k = 0; k < top_k_; ++k) {
            auto idx_col = topk_idx.slice(1, k, k + 1).squeeze(1);
            auto weight_col = topk_weights.slice(1, k, k + 1).squeeze(1);

            auto expert_scalar = tenzor::full({1}, static_cast<double>(e),
                                              idx_col.dtype(), idx_col.device());
            auto mask = tenzor::eq(idx_col, expert_scalar);

            // Skip if no tokens route to this expert at this position
            Tensor mask_count = tenzor::sum(mask.to(DType::Float32)).to(Device::cpu());
            float mc_val = mask_count.template item<float>();
            if (mc_val == 0.0f) continue;

            auto mask_f = mask.to(input.tensor().dtype());

            // Variable-level masked input: flat * mask. The mask is treated
            // as a constant (no gradient flows through indicator selection),
            // wrap it in a no-grad Variable so the multiplication graph still
            // includes `flat` on the differentiable side.
            auto mask_var = Variable(mask_f.unsqueeze(1), /*requires_grad=*/false);
            auto masked_input = flat * mask_var;

            // Expert forward: up -> relu -> down. All Variable-level so the
            // chain stays connected from input → flat → masked_input → hidden
            // → expert_out.
            auto hidden = up_[e]->forward(masked_input);
            hidden = relu(hidden);
            if (dropout_) hidden = dropout_->forward(hidden);
            auto expert_out = down_[e]->forward(hidden);

            // Weight by the routing coefficients (Variable * Variable so the
            // graph picks up the expert path).
            auto wm = (weight_col * mask_f).unsqueeze(1);
            auto wm_var = Variable(wm, /*requires_grad=*/false);
            auto weighted = expert_out * wm_var;

            output = output + weighted;
        }
    }

    // Reshape back through Variable-level reshape.
    output = tenzor::reshape(output, std::vector<int64_t>(orig_shape.begin(), orig_shape.end()));

    // Auxiliary load balancing loss (separate path; doesn't need to flow
    // through the expert outputs to remain useful, but it does need to flow
    // through `probs_v` so the router learns).
    auto probs_mean = tenzor::mean(probs_t, /*dim=*/0, /*keepdim=*/false);
    auto top1_idx = topk_idx.slice(1, 0, 1).squeeze(1);

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
