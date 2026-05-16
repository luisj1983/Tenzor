/**
 * @file sgd.hpp
 * @brief Stochastic Gradient Descent optimizer
 *
 * Implements SGD with support for momentum, dampening, weight decay, and Nesterov acceleration.
 */

#pragma once

#include "optimizer.hpp"

namespace tenzor {
namespace optim {

/**
 * @brief Stochastic Gradient Descent (SGD) optimizer
 *
 * Implements the SGD algorithm with optional momentum and Nesterov acceleration:
 *
 * **Without Momentum:**
 * \f[
 * \theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
 * \f]
 *
 * **With Momentum:**
 * \f[
 * v_{t+1} = \mu v_t + \nabla L(\theta_t) \\
 * \theta_{t+1} = \theta_t - \eta v_{t+1}
 * \f]
 *
 * **With Nesterov Momentum:**
 * \f[
 * v_{t+1} = \mu v_t + \nabla L(\theta_t) \\
 * \theta_{t+1} = \theta_t - \eta(\mu v_{t+1} + \nabla L(\theta_t))
 * \f]
 *
 * where:
 * - \f$\theta\f$: parameters
 * - \f$\eta\f$: learning rate
 * - \f$\mu\f$: momentum coefficient (0.9 typical)
 * - \f$v\f$: velocity (momentum buffer)
 * - \f$\nabla L\f$: gradient
 *
 * **Momentum Benefits:**
 * - Accelerates convergence in relevant directions
 * - Dampens oscillations in irrelevant directions
 * - Helps escape local minima
 *
 * **Parameter Recommendations:**
 * - lr: 0.01 to 0.1 (task-dependent, use learning rate scheduling)
 * - momentum: 0.9 (standard choice)
 * - weight_decay: 1e-4 to 1e-5 (L2 regularization)
 * - nesterov: true (often improves convergence)
 *
 * @param params Parameters to optimize
 * @param lr Learning rate (default: required)
 * @param momentum Momentum factor (default: 0.0, typical: 0.9)
 * @param dampening Dampening for momentum (default: 0.0)
 * @param weight_decay L2 regularization coefficient (default: 0.0)
 * @param nesterov Enable Nesterov momentum (default: false)
 *
 * @par Complexity
 * - Time: O(P) per step, where P is number of parameters
 * - Space: O(P) for velocity buffers when momentum > 0
 *
 * @code
 * // Standard SGD with momentum
 * auto optimizer = SGD(model.parameters(), 0.01, 0.9, 0.0, 1e-4, false);
 *
 * // Nesterov accelerated gradient
 * auto optimizer_nag = SGD(model.parameters(), 0.01, 0.9, 0.0, 0.0, true);
 * @endcode
 *
 * @see Adam for adaptive learning rate alternative
 * @see StepLR for learning rate scheduling
 */
class SGD : public Optimizer {
public:
    SGD(std::vector<std::shared_ptr<Variable>> params,
        double lr,
        double momentum = 0.0,
        double dampening = 0.0,
        double weight_decay = 0.0,
        bool nesterov = false);

    /** @brief Perform single SGD step with momentum */
    auto step_impl() -> void override;

    /** @brief Set new learning rate */
    auto set_lr(double lr) -> void override;  // M15: explicit override marker

    /** @brief Get current learning rate */
    auto get_lr() const -> double override;  // M15: explicit override marker

    /** @brief Get optimizer state (velocity buffers) for serialization */
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;

    /** @brief Load optimizer state from dictionary */
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    double lr_;
    double momentum_;
    double dampening_;
    double weight_decay_;
    bool nesterov_;

    std::vector<Tensor> velocity_buffers_;

    auto initialize_buffers() -> void;
};

} // namespace optim
} // namespace tenzor
