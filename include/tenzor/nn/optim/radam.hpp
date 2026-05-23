/**
 * @file radam.hpp
 * @brief RAdam (Rectified Adam) optimizer
 */

#pragma once

#include "optimizer.hpp"

namespace tenzor {
namespace optim {

/**
 * @brief RAdam (Rectified Adam) optimizer
 *
 * Automatically adjusts the adaptive learning rate based on variance of
 * the second moment estimate. When variance is high (early training),
 * falls back to SGD with momentum. This removes the need for learning
 * rate warmup.
 *
 * Algorithm:
 * - rho_inf = 2/(1-beta2) - 1
 * - rho_t = rho_inf - 2*t*beta2^t/(1-beta2^t)
 * - If rho_t > 5: use variance-rectified Adam update
 * - Otherwise: use SGD with momentum (first moment only)
 *
 * **Advantages:**
 * - No learning rate warmup needed
 * - More robust to learning rate choice than Adam
 * - Same convergence as Adam in later training
 *
 * **Recommended Hyperparameters:**
 * - lr: 1e-3 (default)
 * - beta1: 0.9 (first moment decay)
 * - beta2: 0.999 (second moment decay)
 * - eps: 1e-8 (numerical stability)
 * - weight_decay: 0.0 (L2 regularization)
 *
 * @param params Parameters to optimize
 * @param lr Learning rate (default: 1e-3)
 * @param beta1 First moment decay rate (default: 0.9)
 * @param beta2 Second moment decay rate (default: 0.999)
 * @param eps Term for numerical stability (default: 1e-8)
 * @param weight_decay L2 penalty (default: 0.0)
 *
 * @par Complexity
 * - Time: O(P) per step
 * - Space: O(2P) for moment estimates
 *
 * @code
 * auto optimizer = RAdam(model.parameters(), 1e-3, 0.9, 0.999, 1e-8);
 * @endcode
 *
 * @see Adam, AdamW, SGD
 */
class RAdam : public Optimizer {
public:
    RAdam(std::vector<std::shared_ptr<Variable>> params,
          double lr = 1e-3,
          double beta1 = 0.9,
          double beta2 = 0.999,
          double eps = 1e-8,
          double weight_decay = 0.0);

    /**
     * @brief Construct with parameter groups (audit D.4).
     *
     * Hyperparameters set per-group on each `ParamGroup` (lr, weight_decay
     * directly; beta1/beta2/eps via the corresponding `std::optional`
     * members) override the optimizer-wide defaults stored on this
     * instance for parameters in that group. Parameters outside any
     * group, or groups with an unset optional, fall back to the
     * defaults passed here.
     */
    explicit RAdam(std::vector<optim::ParamGroup> groups,
                   double default_lr = 1e-3,
                   double default_beta1 = 0.9,
                   double default_beta2 = 0.999,
                   double default_eps = 1e-8,
                   double default_weight_decay = 0.0);

    /** @brief Perform single RAdam step */
    auto step_impl() -> void override;

    /** @brief Set new learning rate */
    auto set_lr(double lr) -> void override;

    /** @brief Get current learning rate */
    auto get_lr() const -> double override;

    /** @brief Get optimizer state (moment estimates) for serialization */
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;

    /** @brief Load optimizer state from dictionary */
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

protected:
    // Audit K.1: extend exp_avg_ / exp_avg_sq_ when add_param_group
    // appends new parameters mid-training.
    auto on_parameters_appended_(size_t old_count, size_t new_count) -> void override;

private:
    double lr_;
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;

    int64_t step_count_{0};
    std::vector<Tensor> exp_avg_;       // First moment estimates
    std::vector<Tensor> exp_avg_sq_;    // Second moment estimates

    auto initialize_buffers() -> void;
};

} // namespace optim
} // namespace tenzor
