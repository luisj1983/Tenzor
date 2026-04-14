/**
 * @file moe.hpp
 * @brief Mixture of Experts (MoE) layer
 *
 * Implements sparse MoE with top-k routing, capacity factor, and
 * auxiliary load balancing loss. Used in GPT-4, Mixtral, Switch Transformer.
 */

#pragma once

#include <memory>
#include <vector>
#include "../module.hpp"
#include "linear.hpp"
#include "dropout.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Mixture of Experts layer with top-k routing.
 *
 * Routes each token to its top-k experts based on a learned router.
 * Includes auxiliary load balancing loss to encourage uniform expert utilization.
 *
 * Forward returns a pair: (output, aux_loss).
 */
class MixtureOfExperts : public Module {
public:
    /**
     * @brief Construct MoE layer.
     *
     * @param input_dim Input/output dimension
     * @param hidden_dim Expert FFN hidden dimension
     * @param num_experts Number of expert sub-networks
     * @param top_k Number of experts each token is routed to (default: 2)
     * @param capacity_factor Capacity factor for expert buffer sizing (default: 1.25)
     * @param aux_loss_weight Weight for load balancing auxiliary loss (default: 0.01)
     * @param dropout Dropout probability in expert FFNs (default: 0.0)
     */
    MixtureOfExperts(int64_t input_dim, int64_t hidden_dim,
                     int64_t num_experts, int64_t top_k = 2,
                     double capacity_factor = 1.25,
                     double aux_loss_weight = 0.01,
                     double dropout = 0.0);

    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Forward with auxiliary loss returned separately.
     *
     * @param input [batch, seq_len, input_dim] or [tokens, input_dim]
     * @return {output, aux_loss} where aux_loss is a scalar Variable
     */
    auto forward_with_loss(const Variable& input) -> std::pair<Variable, Variable>;

    auto num_experts() const -> int64_t { return num_experts_; }
    auto top_k() const -> int64_t { return top_k_; }
    auto capacity_factor() const -> double { return capacity_factor_; }

private:
    int64_t input_dim_;
    int64_t hidden_dim_;
    int64_t num_experts_;
    int64_t top_k_;
    double capacity_factor_;
    double aux_loss_weight_;

    std::shared_ptr<Linear> router_;              // [input_dim -> num_experts]
    std::vector<std::shared_ptr<Linear>> up_;     // Expert up-projections
    std::vector<std::shared_ptr<Linear>> down_;   // Expert down-projections
    std::shared_ptr<Dropout> dropout_;
};

} // namespace nn
} // namespace tenzor
