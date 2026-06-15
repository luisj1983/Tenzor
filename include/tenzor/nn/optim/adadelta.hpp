/**
 * @file adadelta.hpp
 * @brief Adadelta optimizer with adaptive learning rates
 *
 * Implements Adadelta, an extension of Adagrad that addresses its aggressive
 * learning rate decay by using a moving window of gradient updates rather than
 * accumulating all past gradients.
 */

#pragma once

#include "optimizer.hpp"

namespace tenzor {
namespace optim {

/**
 * @brief Adadelta optimizer (parameter-free adaptive learning)
 *
 * Extends Adagrad by restricting the window of accumulated past gradients to a
 * fixed size using exponential moving average. Unlike Adagrad, learning rates
 * don't monotonically decrease.
 *
 * **Update Rule:**
 * \f[
 * E[g^2]_t = \rho E[g^2]_{t-1} + (1 - \rho) g_t^2 \\
 * \Delta\theta_t = -\frac{\sqrt{E[\Delta\theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} g_t \\
 * E[\Delta\theta^2]_t = \rho E[\Delta\theta^2]_{t-1} + (1 - \rho) \Delta\theta_t^2 \\
 * \theta_t = \theta_{t-1} + \eta \Delta\theta_t
 * \f]
 *
 * where:
 * - \f$E[g^2]_t\f$: Exponential moving average of squared gradients
 * - \f$E[\Delta\theta^2]_t\f$: Exponential moving average of squared parameter updates
 * - \f$\rho\f$: Decay rate (typical: 0.9)
 * - \f$\epsilon\f$: Small constant for numerical stability
 * - \f$\eta\f$: Learning rate (typically 1.0, as method is parameter-free)
 *
 * **Key Innovation:**
 * - **No learning rate required**: Uses RMS of parameter updates instead
 * - **Unit correction**: Update step has same hypothetical units as parameter
 * - **Bounded accumulation**: Uses exponential moving average, not infinite sum
 *
 * **Advantages:**
 * - No manual learning rate tuning (set lr=1.0)
 * - Addresses Adagrad's aggressive decay
 * - Dimensionally correct updates
 * - Continues learning (no monotonic decrease)
 * - Robust to hyperparameters
 *
 * **Disadvantages:**
 * - Slower convergence than Adam
 * - Less popular than Adam (fewer practitioners familiar)
 * - May require more epochs
 * - Still has rho hyperparameter to tune
 *
 * **When to Use:**
 * - Don't want to tune learning rate
 * - Long training runs
 * - When Adagrad stops learning too early
 * - Sparse gradients with long training
 *
 * **Recommended Hyperparameters:**
 * - lr: 1.0 (method is designed to not need learning rate)
 * - rho: 0.9 (controls memory window size)
 * - eps: 1e-6 (numerical stability)
 * - weight_decay: 0.0 (use if needed)
 *
 * **Comparison:**
 * - **Adagrad**: Accumulates all gradients → aggressive decay
 * - **Adadelta**: Moving average of gradients → continues learning
 * - **RMSprop**: Similar but requires learning rate
 * - **Adam**: Adadelta + momentum, requires learning rate
 *
 * @param params Parameters to optimize
 * @param lr Learning rate (default: 1.0, typically not changed)
 * @param rho Coefficient for computing running averages (default: 0.9)
 * @param eps Term for numerical stability (default: 1e-6)
 * @param weight_decay L2 penalty (default: 0.0)
 *
 * @par Complexity
 * - Time: O(P) per step
 * - Space: O(2P) for two moving averages (square_avg and acc_delta)
 *
 * @code
 * // Standard Adadelta (no learning rate tuning needed)
 * auto optimizer = Adadelta(model.parameters());  // Uses lr=1.0, rho=0.9
 *
 * // Custom decay rate
 * auto optimizer = Adadelta(model.parameters(), 1.0, 0.95);
 *
 * // Training loop - no learning rate schedule needed
 * for (int epoch = 0; epoch < epochs; ++epoch) {
 *     optimizer.zero_grad();
 *     auto loss = model.forward(input);
 *     loss.backward();
 *     optimizer.step();  // Automatically adapts effective learning rate
 * }
 * @endcode
 *
 * @par References
 * Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method.
 * arXiv:1212.5701
 *
 * @see Adagrad, RMSprop, Adam
 */
class Adadelta : public Optimizer {
public:
    /**
     * @brief Construct Adadelta optimizer
     *
     * @param params Parameters to optimize
     * @param lr Learning rate (typically 1.0)
     * @param rho Decay rate for moving averages
     * @param eps Epsilon for numerical stability
     * @param weight_decay L2 regularization coefficient
     */
    Adadelta(std::vector<std::shared_ptr<Variable>> params,
             double lr = 1.0,
             double rho = 0.9,
             double eps = 1e-6,
             double weight_decay = 0.0);

    /**
     * @brief Construct from `ParamGroup`s (audit D.4).
     *
     * Each group may override `lr` / `weight_decay` via ParamGroup's
     * non-optional fields and `rho` / `eps` via the optional fields.
     * Defaults supplied here apply when a group does not override.
     */
    explicit Adadelta(std::vector<optim::ParamGroup> groups,
                      double default_lr = 1.0,
                      double default_rho = 0.9,
                      double default_eps = 1e-6,
                      double default_weight_decay = 0.0);

    /**
     * @brief Perform single Adadelta optimization step
     *
     * Updates parameters using exponential moving averages of
     * squared gradients and squared parameter updates.
     *
     * @pre Gradients must be computed via backward()
     */
    auto step_impl() -> void override;

    /**
     * @brief Zero all parameter gradients (QQ.14: see Optimizer::zero_grad).
     */
    auto zero_grad(bool set_to_none = true) -> void override;

    /**
     * @brief Set new learning rate
     * @param lr New learning rate (rarely changed from 1.0)
     */
    auto set_lr(double lr) -> void override;

    /**
     * @brief Get current learning rate
     * @return Current learning rate
     */
    auto get_lr() const -> double override;

    /**
     * @brief Get optimizer state for serialization
     *
     * Returns square_avg (E[g^2]) and acc_delta (E[Δθ^2]).
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

protected:
    // Audit K.1: extend square_avg_ / acc_delta_ when add_param_group
    // appends new parameters mid-training.
    auto on_parameters_appended_(size_t old_count, size_t new_count) -> void override;

private:
    double lr_;
    double rho_;
    double eps_;
    double weight_decay_;

    std::vector<Tensor> square_avg_;  ///< E[g^2] - running average of squared gradients
    std::vector<Tensor> acc_delta_;   ///< E[Δθ^2] - running average of squared parameter updates

    /**
     * @brief Initialize state buffers to zeros matching parameter shapes
     */
    auto initialize_buffers() -> void;
};

} // namespace optim
} // namespace tenzor
