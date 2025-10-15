/**
 * @file rmsprop.hpp
 * @brief RMSprop optimizer with centered and momentum variants
 *
 * Implements RMSprop (Root Mean Square Propagation), an adaptive learning rate
 * optimizer that maintains per-parameter learning rates based on moving averages
 * of squared gradients.
 */

#pragma once

#include "optimizer.hpp"

namespace tenzor {
namespace optim {

/**
 * @brief RMSprop (Root Mean Square Propagation) optimizer
 *
 * Adapts learning rates for each parameter by dividing by a running average of
 * recent gradient magnitudes. Addresses Adagrad's monotonically decreasing learning rates.
 *
 * **Standard RMSprop Update:**
 * \f[
 * v_t = \alpha v_{t-1} + (1 - \alpha) g_t^2 \\
 * \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t} + \epsilon} g_t
 * \f]
 *
 * **With Momentum:**
 * \f[
 * v_t = \alpha v_{t-1} + (1 - \alpha) g_t^2 \\
 * b_t = \rho b_{t-1} + \frac{g_t}{\sqrt{v_t} + \epsilon} \\
 * \theta_t = \theta_{t-1} - \eta b_t
 * \f]
 *
 * **Centered RMSprop (Reduces Variance):**
 * \f[
 * v_t = \alpha v_{t-1} + (1 - \alpha) g_t^2 \\
 * m_t = \alpha m_{t-1} + (1 - \alpha) g_t \\
 * \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t - m_t^2} + \epsilon} g_t
 * \f]
 *
 * where:
 * - \f$v_t\f$: Moving average of squared gradients (second moment)
 * - \f$m_t\f$: Moving average of gradients (first moment, if centered)
 * - \f$b_t\f$: Momentum buffer
 * - \f$\alpha\f$: Decay rate for moving average (typical: 0.99)
 * - \f$\rho\f$: Momentum coefficient (typical: 0.9)
 * - \f$\eta\f$: Learning rate
 * - \f$\epsilon\f$: Small constant for numerical stability
 *
 * **Advantages:**
 * - Works well with non-stationary objectives
 * - Good for RNNs and online learning
 * - Less aggressive learning rate decay than Adagrad
 * - Handles noisy gradients effectively
 *
 * **Disadvantages:**
 * - Requires tuning of α (decay rate)
 * - May not converge as reliably as Adam
 * - Sensitive to learning rate initialization
 *
 * **When to Use:**
 * - Training RNNs and LSTMs
 * - Non-stationary problems (e.g., online learning)
 * - When Adam is unstable
 * - Mini-batch stochastic optimization
 *
 * **Recommended Hyperparameters:**
 * - lr: 1e-2 to 1e-3 (start with 1e-2)
 * - alpha: 0.99 (standard, controls memory of past gradients)
 * - momentum: 0.0 or 0.9 (enable for faster convergence)
 * - eps: 1e-8 (numerical stability)
 * - centered: false (enable to reduce variance, slight overhead)
 *
 * @param params Parameters to optimize
 * @param lr Learning rate (default: 1e-2)
 * @param alpha Smoothing constant (decay rate, default: 0.99)
 * @param eps Term for numerical stability (default: 1e-8)
 * @param weight_decay L2 penalty (default: 0.0)
 * @param momentum Momentum factor (default: 0.0)
 * @param centered Use centered RMSprop (default: false)
 *
 * @par Complexity
 * - Time: O(P) per step
 * - Space: O(P) for square_avg, O(P) for grad_avg if centered, O(P) for momentum
 *
 * @code
 * // Standard RMSprop
 * auto optimizer = RMSprop(model.parameters(), 1e-2, 0.99, 1e-8);
 *
 * // RMSprop with momentum for RNN training
 * auto optimizer = RMSprop(model.parameters(), 1e-3, 0.99, 1e-8, 0.0, 0.9);
 *
 * // Centered RMSprop for reduced variance
 * auto optimizer = RMSprop(model.parameters(), 1e-2, 0.99, 1e-8, 0.0, 0.0, true);
 * @endcode
 *
 * @see Adam, Adagrad, Adadelta
 */
class RMSprop : public Optimizer {
public:
    /**
     * @brief Construct RMSprop optimizer
     *
     * @param params Parameters to optimize
     * @param lr Learning rate
     * @param alpha Smoothing constant (decay rate for moving average)
     * @param eps Epsilon for numerical stability
     * @param weight_decay L2 regularization coefficient
     * @param momentum Momentum factor (0.0 disables momentum)
     * @param centered Enable centered RMSprop variant
     */
    RMSprop(std::vector<std::shared_ptr<Variable>> params,
            double lr = 1e-2,
            double alpha = 0.99,
            double eps = 1e-8,
            double weight_decay = 0.0,
            double momentum = 0.0,
            bool centered = false);

    /**
     * @brief Perform single RMSprop optimization step
     *
     * Updates parameters using adaptive learning rates based on
     * exponentially decaying average of squared gradients.
     *
     * @pre Gradients must be computed via backward()
     */
    auto step() -> void override;

    /**
     * @brief Zero all parameter gradients
     */
    auto zero_grad() -> void;

    /**
     * @brief Set new learning rate
     * @param lr New learning rate
     */
    auto set_lr(double lr) -> void;

    /**
     * @brief Get current learning rate
     * @return Current learning rate
     */
    auto get_lr() const -> double;

    /**
     * @brief Get optimizer state for serialization
     *
     * Returns square_avg, grad_avg (if centered), and momentum_buffer (if momentum > 0)
     *
     * @return Map of state names to tensors
     */
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;

    /**
     * @brief Load optimizer state from dictionary
     *
     * @param state State dictionary to restore
     */
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    double lr_;
    double alpha_;
    double eps_;
    double weight_decay_;
    double momentum_;
    bool centered_;

    std::vector<Tensor> square_avg_;      ///< E[g^2] - running average of squared gradients
    std::vector<Tensor> grad_avg_;        ///< E[g] - running average of gradients (if centered)
    std::vector<Tensor> momentum_buffer_; ///< Momentum buffer (if momentum > 0)

    /**
     * @brief Initialize state buffers to zeros matching parameter shapes
     */
    auto initialize_buffers() -> void;
};

} // namespace optim
} // namespace tenzor
