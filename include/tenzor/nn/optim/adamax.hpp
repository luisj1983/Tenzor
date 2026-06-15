/**
 * @file adamax.hpp
 * @brief Adamax optimizer (Adam variant based on infinity norm)
 */

#pragma once

#include "optimizer.hpp"

namespace tenzor {
namespace optim {

/**
 * @brief Adamax optimizer — Adam with the second moment replaced by the
 * running infinity norm of the gradient.
 *
 * Kingma & Ba (2014), section 7. The L2-norm second moment \f$v_t\f$ is
 * replaced with \f$u_t\f$, the element-wise exponentially-weighted
 * running max of \f$|g_t|\f$:
 *
 * \f[
 * u_t = \max(\beta_2 \cdot u_{t-1},\; |g_t|)
 * \f]
 *
 * Parameter update is
 * \f[
 * \theta_t = \theta_{t-1} - \frac{\eta}{1 - \beta_1^t} \cdot \frac{m_t}{u_t + \epsilon}
 * \f]
 *
 * Adamax is often more numerically stable than Adam when gradients have
 * large outliers, since the infinity norm caps denominator growth. It
 * uses the same (m_t, u_t) buffer layout as Adam's (m_t, v_t).
 *
 * Defaults match PyTorch's torch.optim.Adamax.
 */
class Adamax : public Optimizer {
public:
    Adamax(std::vector<std::shared_ptr<Variable>> params,
           double lr = 2e-3,
           double beta1 = 0.9,
           double beta2 = 0.999,
           double eps = 1e-8,
           double weight_decay = 0.0);

    explicit Adamax(std::vector<optim::ParamGroup> groups,
                    double default_lr = 2e-3,
                    double default_beta1 = 0.9,
                    double default_beta2 = 0.999,
                    double default_eps = 1e-8,
                    double default_weight_decay = 0.0);

    auto step_impl() -> void override;

    auto set_lr(double lr) -> void override;
    auto get_lr() const -> double override;

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

protected:
    // Audit K.1: extend exp_avg_ / exp_inf_ when add_param_group
    // appends new parameters mid-training.
    auto on_parameters_appended_(size_t old_count, size_t new_count) -> void override;

private:
    double lr_;
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;

    int64_t step_count_{0};
    std::vector<Tensor> exp_avg_;       // m_t
    std::vector<Tensor> exp_inf_;       // u_t — running infinity norm

    auto initialize_buffers() -> void;
};

} // namespace optim
} // namespace tenzor
