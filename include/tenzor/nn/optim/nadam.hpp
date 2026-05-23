/**
 * @file nadam.hpp
 * @brief NAdam optimizer (Nesterov-accelerated Adam)
 */

#pragma once

#include "optimizer.hpp"

namespace tenzor {
namespace optim {

/**
 * @brief NAdam optimizer — Adam with Nesterov momentum.
 *
 * Dozat (2016). Incorporates Nesterov lookahead into the Adam first-moment
 * update and uses a running "momentum schedule" \f$\mu_t\f$:
 *
 * \f[
 * \mu_t = \beta_1 \cdot (1 - \tfrac{1}{2} \cdot 0.96^{t \cdot \psi})
 * \f]
 *
 * where \f$\psi\f$ is `momentum_decay` (default 4e-3). At each step the
 * bias-corrected first moment is adjusted with a one-step lookahead using
 * \f$\mu_t\f$ and \f$\mu_{t+1}\f$, which gives NAdam its characteristic
 * faster early convergence on well-conditioned losses.
 *
 * Defaults match PyTorch's torch.optim.NAdam.
 */
class NAdam : public Optimizer {
public:
    NAdam(std::vector<std::shared_ptr<Variable>> params,
          double lr = 2e-3,
          double beta1 = 0.9,
          double beta2 = 0.999,
          double eps = 1e-8,
          double weight_decay = 0.0,
          double momentum_decay = 4e-3);

    explicit NAdam(std::vector<optim::ParamGroup> groups,
                   double default_lr = 2e-3,
                   double default_beta1 = 0.9,
                   double default_beta2 = 0.999,
                   double default_eps = 1e-8,
                   double default_weight_decay = 0.0,
                   double default_momentum_decay = 4e-3);

    auto step_impl() -> void override;

    auto set_lr(double lr) -> void;
    auto get_lr() const -> double;

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

protected:
    // Audit K.1: extend exp_avg_ / exp_avg_sq_ when add_param_group
    // appends new parameters mid-training.  mu_product_ is optimiser-wide.
    auto on_parameters_appended_(size_t old_count, size_t new_count) -> void override;

private:
    double lr_;
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;
    double momentum_decay_;

    int64_t step_count_{0};
    // Product of \mu_t over 1..t, used to debias the first-moment lookahead.
    double mu_product_{1.0};

    std::vector<Tensor> exp_avg_;
    std::vector<Tensor> exp_avg_sq_;

    auto initialize_buffers() -> void;
};

} // namespace optim
} // namespace tenzor
