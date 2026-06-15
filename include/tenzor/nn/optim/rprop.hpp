/**
 * @file rprop.hpp
 * @brief Resilient Propagation (Rprop) optimizer
 *
 * Implements the Rprop algorithm which adapts per-parameter step sizes based
 * on gradient sign changes rather than gradient magnitudes.
 */

#pragma once

#include "optimizer.hpp"

namespace tenzor {
namespace optim {

/**
 * @brief Resilient Propagation (Rprop) optimizer
 *
 * Rprop adapts individual step sizes for each parameter based on the sign of
 * consecutive gradients. Unlike SGD or Adam, it ignores gradient magnitude and
 * only uses the sign information:
 *
 * **Update Rules:**
 * \f[
 * \Delta_i^{(t)} = \begin{cases}
 *   \min(\Delta_i^{(t-1)} \cdot \eta^+,\, \Delta_{\max}) & \text{if } g_i^{(t-1)} \cdot g_i^{(t)} > 0 \\
 *   \max(\Delta_i^{(t-1)} \cdot \eta^-,\, \Delta_{\min}) & \text{if } g_i^{(t-1)} \cdot g_i^{(t)} < 0 \\
 *   \Delta_i^{(t-1)} & \text{otherwise}
 * \end{cases}
 * \f]
 *
 * \f[
 * \theta_i^{(t+1)} = \theta_i^{(t)} - \text{sign}(g_i^{(t)}) \cdot \Delta_i^{(t)}
 * \f]
 *
 * When the gradient sign flips, the previous update is reverted and the current
 * gradient is zeroed to prevent a double penalty.
 *
 * where:
 * - \f$\Delta_i\f$: per-parameter step size
 * - \f$\eta^+\f$: multiplicative increase factor (default: 1.2)
 * - \f$\eta^-\f$: multiplicative decrease factor (default: 0.5)
 * - \f$g_i\f$: gradient of parameter \f$i\f$
 *
 * **Rprop Benefits:**
 * - Robust to gradient scale (only uses sign information)
 * - Fast convergence for full-batch training
 * - No learning rate tuning sensitivity
 *
 * **Parameter Recommendations:**
 * - lr: 0.01 (initial step size, less critical than in SGD)
 * - etas: (0.5, 1.2) (standard choice, rarely needs tuning)
 * - step_sizes: (1e-6, 50) (prevents vanishing/exploding steps)
 *
 * @note Rprop is designed for full-batch training. For mini-batch training,
 *       consider using Adam or SGD with momentum instead.
 *
 * @param params Parameters to optimize
 * @param lr Initial step size (default: 0.01)
 * @param eta_minus Multiplicative decrease factor (default: 0.5)
 * @param eta_plus Multiplicative increase factor (default: 1.2)
 * @param step_min Minimum allowed step size (default: 1e-6)
 * @param step_max Maximum allowed step size (default: 50.0)
 *
 * @par Complexity
 * - Time: O(P) per step, where P is number of parameters
 * - Space: O(2P) for step size and previous gradient buffers
 *
 * @code
 * // Standard Rprop
 * auto optimizer = Rprop(model.parameters(), 0.01);
 *
 * // Custom Rprop with aggressive step adaptation
 * auto optimizer = Rprop(model.parameters(), 0.01, 0.3, 1.5, 1e-7, 100.0);
 * @endcode
 *
 * @see SGD for mini-batch gradient descent
 * @see Adam for adaptive learning rate alternative
 */
class Rprop : public Optimizer {
public:
    Rprop(std::vector<std::shared_ptr<Variable>> params,
          double lr = 0.01,
          double eta_minus = 0.5,
          double eta_plus = 1.2,
          double step_min = 1e-6,
          double step_max = 50.0);

    /**
     * @brief Construct Rprop from explicit parameter groups (audit D.4).
     *
     * Each ParamGroup may override `lr`. Rprop-specific hyperparameters
     * (eta_minus, eta_plus, step_min, step_max) have no ParamGroup field,
     * so they fall back to the optimiser-wide defaults supplied here.
     */
    explicit Rprop(std::vector<optim::ParamGroup> groups,
                   double default_lr = 0.01,
                   double default_eta_minus = 0.5,
                   double default_eta_plus = 1.2,
                   double default_step_min = 1e-6,
                   double default_step_max = 50.0);

    /** @brief Perform single Rprop step */
    auto step_impl() -> void override;

    /** @brief Set new learning rate (initial step size) */
    auto set_lr(double lr) -> void override;

    /** @brief Get current learning rate */
    auto get_lr() const -> double override;

    /** @brief Get optimizer state (step sizes, prev gradients) for serialization */
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;

    /** @brief Load optimizer state from dictionary */
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

protected:
    // Audit K.1: extend step_sizes_ / prev_grads_ when add_param_group
    // appends new parameters mid-training.
    auto on_parameters_appended_(size_t old_count, size_t new_count) -> void override;

private:
    double lr_;
    double eta_minus_;
    double eta_plus_;
    double step_min_;
    double step_max_;

    std::vector<Tensor> step_sizes_;    ///< Per-parameter step sizes
    std::vector<Tensor> prev_grads_;    ///< Previous gradients (for sign comparison)
    bool first_step_{true};

    auto initialize_buffers() -> void;
};

} // namespace optim
} // namespace tenzor
