/**
 * @file asgd.hpp
 * @brief Averaged Stochastic Gradient Descent optimizer
 *
 * Implements ASGD which maintains a running average of parameters that starts
 * after a configurable warmup period, often yielding better generalization.
 */

#pragma once

#include "optimizer.hpp"

namespace tenzor {
namespace optim {

/**
 * @brief Averaged Stochastic Gradient Descent (ASGD) optimizer
 *
 * Implements the ASGD algorithm from "Acceleration of stochastic approximation
 * by averaging" (Polyak & Juditsky, 1992).
 *
 * **Parameter Update:**
 * \f[
 * \eta_t = \frac{\text{lr}}{(1 + \lambda \cdot \text{lr} \cdot t)^\alpha}
 * \f]
 * \f[
 * x_t = x_{t-1} - \eta_t \cdot \nabla L(x_{t-1})
 * \f]
 *
 * **Averaging (after step t0):**
 * \f[
 * \mu_t = \max(1, t - t_0)
 * \f]
 * \f[
 * ax_t = \left(1 - \frac{1}{\mu_t}\right) ax_{t-1} + \frac{1}{\mu_t} x_t
 * \f]
 *
 * where:
 * - \f$x\f$: parameters (point iterate)
 * - \f$ax\f$: averaged parameters (used for final prediction)
 * - \f$\eta_t\f$: step-dependent learning rate
 * - \f$\lambda\f$: decay term for learning rate schedule
 * - \f$\alpha\f$: power for eta update
 * - \f$t_0\f$: step at which averaging begins
 *
 * **Benefits:**
 * - Better generalization than plain SGD
 * - Theoretical convergence guarantees for convex objectives
 * - Automatic learning rate decay via eta schedule
 *
 * **Parameter Recommendations:**
 * - lr: 0.01 (default)
 * - lambd: 1e-4 (controls lr decay rate)
 * - alpha: 0.75 (power for eta schedule)
 * - t0: 1e6 (delay averaging; set lower for faster averaging)
 * - weight_decay: 0.0 (L2 regularization)
 *
 * @param params Parameters to optimize
 * @param lr Learning rate (default: 0.01)
 * @param lambd Decay term (default: 1e-4)
 * @param alpha Power for eta update (default: 0.75)
 * @param t0 Start averaging after this many steps (default: 1e6)
 * @param weight_decay L2 regularization coefficient (default: 0.0)
 *
 * @par Complexity
 * - Time: O(P) per step, where P is number of parameters
 * - Space: O(2P) for averaged parameter buffers and eta state
 *
 * @code
 * // Standard ASGD
 * auto optimizer = ASGD(model.parameters(), 0.01);
 *
 * // ASGD with early averaging
 * auto optimizer = ASGD(model.parameters(), 0.01, 1e-4, 0.75, 100.0, 1e-4);
 * @endcode
 *
 * @see SGD for standard stochastic gradient descent
 */
class ASGD : public Optimizer {
public:
    ASGD(std::vector<std::shared_ptr<Variable>> params,
         double lr = 0.01,
         double lambd = 1e-4,
         double alpha = 0.75,
         double t0 = 1e6,
         double weight_decay = 0.0);

    /**
     * @brief Construct ASGD from parameter groups (audit D.4).
     *
     * Each group's lr and weight_decay (plus the optional alpha override)
     * are honoured per-parameter in step_impl(); other hyperparams fall
     * back to the optimizer-wide defaults supplied here.
     */
    explicit ASGD(std::vector<optim::ParamGroup> groups,
                  double default_lr = 0.01,
                  double default_lambd = 1e-4,
                  double default_alpha = 0.75,
                  double default_t0 = 1e6,
                  double default_weight_decay = 0.0);

    /** @brief Perform single ASGD step with parameter averaging */
    auto step_impl() -> void override;

    /** @brief Set new learning rate */
    auto set_lr(double lr) -> void override;

    /** @brief Get current learning rate */
    auto get_lr() const -> double override;

    /** @brief Get optimizer state (averaged buffers, step count) for serialization */
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;

    /** @brief Load optimizer state from dictionary */
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

protected:
    // Audit K.1: extend ax_buffers_ when add_param_group appends new
    // parameters mid-training.
    auto on_parameters_appended_(size_t old_count, size_t new_count) -> void override;

private:
    double lr_;
    double lambd_;
    double alpha_;
    double t0_;
    double weight_decay_;
    int64_t step_count_{0};

    std::vector<Tensor> ax_buffers_;  ///< Averaged parameters

    auto initialize_buffers() -> void;
};

} // namespace optim
} // namespace tenzor
