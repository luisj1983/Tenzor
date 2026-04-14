/**
 * @file moe.cpp
 * @brief Mixture of Experts implementation
 */

#include "tenzor/nn/layers/moe.hpp"
#include "tenzor/nn/activations/activations.hpp"
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
    auto orig_shape = input.shape();
    int64_t ndim = orig_shape.size();
    int64_t N = 1;
    for (int64_t i = 0; i < ndim - 1; ++i) N *= orig_shape[i];
    int64_t D = orig_shape[ndim - 1];

    // Flatten input for routing
    auto flat_t = tenzor::reshape(input.tensor(), {N, D});
    auto flat = Variable(flat_t, input.requires_grad());

    // Router logits: [N, num_experts]
    auto logits = router_->forward(flat);

    // Softmax over experts
    auto probs = nn::softmax(logits, /*dim=*/1);
    Tensor probs_t = probs.tensor();

    // Top-k selection
    auto topk_result = tenzor::topk(probs_t, top_k_, /*dim=*/1, /*largest=*/true, /*sorted=*/true);
    auto topk_vals = std::get<0>(topk_result);
    auto topk_idx = std::get<1>(topk_result);

    // Normalize top-k weights
    auto topk_sum = tenzor::sum(topk_vals, /*dim=*/1, /*keepdim=*/true);
    auto topk_weights = topk_vals / topk_sum;

    // Dispatch tokens to experts
    auto output_t = tenzor::zeros({N, D}, input.tensor().dtype(), input.tensor().device());

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
            auto masked_input = flat_t * mask_f.unsqueeze(1);

            // Expert forward: up -> relu -> down
            auto expert_in = Variable(masked_input, input.requires_grad());
            auto hidden = up_[e]->forward(expert_in);
            hidden = relu(hidden);
            if (dropout_) hidden = dropout_->forward(hidden);
            auto expert_out = down_[e]->forward(hidden);

            // Weight and accumulate
            auto weighted = expert_out.tensor() * (weight_col * mask_f).unsqueeze(1);
            output_t = output_t + weighted;
        }
    }

    // Reshape output back
    auto output = Variable(
        tenzor::reshape(output_t, std::vector<int64_t>(orig_shape.begin(), orig_shape.end())),
        input.requires_grad());

    // Auxiliary load balancing loss
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

    auto N_scalar = tenzor::full({1}, static_cast<double>(N), DType::Float32, input.tensor().device());
    auto freq = expert_counts / N_scalar;
    auto scale = tenzor::full({1}, static_cast<double>(num_experts_) * aux_loss_weight_,
                              DType::Float32, input.tensor().device());
    auto aux_loss_t = tenzor::sum(freq * probs_mean.to(DType::Float32)) * scale;

    auto aux_loss = Variable(aux_loss_t, input.requires_grad());

    return std::make_pair(output, aux_loss);
}

} // namespace nn
} // namespace tenzor
