/**
 * @file adagrad.hpp
 * @brief Adagrad optimizer with adaptive learning rates
 *
 * Implements Adagrad (Adaptive Gradient Algorithm), an optimizer that adapts
 * learning rates based on accumulated historical gradient information.
 */

#pragma once

#include "optimizer.hpp"

namespace tenzor {
namespace optim {

/**
 * @brief Adagrad (Adaptive Gradient) optimizer
 *
 * Adapts learning rates for each parameter based on accumulated sum of squared gradients.
 * Parameters with frequently occurring features get smaller updates, while parameters
 * with infrequent features get larger updates.
 *
 * **Update Rule:**
 * \f[
 * G_t = G_{t-1} + g_t^2 \\
 * \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t
 * \f]
 *
 * **With Learning Rate Decay:**
 * \f[
 * \eta_t = \frac{\eta}{1 + (t-1) \cdot \text{lr\_decay}}
 * \f]
 *
 * where:
 * - \f$G_t\f$: Accumulated sum of squared gradients
 * - \f$g_t\f$: Current gradient
 * - \f$\eta\f$: Initial learning rate
 * - \f$\epsilon\f$: Small constant for numerical stability
 *
 * **Key Characteristics:**
 * - **Automatic learning rate adaptation**: No manual tuning per parameter
 * - **Accumulation**: Sum (not average) of all past squared gradients
 * - **Monotonic decrease**: Learning rates only decrease over time
 * - **Sparse gradients**: Excellent for sparse data (NLP, recommender systems)
 *
 * **Advantages:**
 * - No manual learning rate tuning needed
 * - Works well with sparse gradients (e.g., embedding layers)
 * - Good for convex optimization
 * - Eliminates need for learning rate schedules
 *
 * **Disadvantages:**
 * - Aggressive learning rate decay (can stop learning too early)
 * - Accumulator grows indefinitely
 * - Poor performance on non-convex problems
 * - Not ideal for deep neural networks (prefer Adam/RMSprop)
 *
 * **When to Use:**
 * - Sparse data (text classification, click-through rate prediction)
 * - Embedding layers with infrequent tokens
 * - Convex optimization problems
 * - When you want automatic learning rate adaptation
 *
 * **When NOT to Use:**
 * - Deep neural networks (use Adam instead)
 * - Problems requiring long training (learning rate decays too fast)
 * - Non-stationary objectives
 *
 * **Recommended Hyperparameters:**
 * - lr: 1e-2 (default is robust)
 * - lr_decay: 0.0 (usually not needed, already has implicit decay)
 * - initial_accumulator_value: 0.0 (default)
 * - eps: 1e-10 (smaller than Adam's 1e-8)
 *
 * @param params Parameters to optimize
 * @param lr Learning rate (default: 1e-2)
 * @param lr_decay Learning rate decay (default: 0.0)
 * @param weight_decay L2 penalty (default: 0.0)
 * @param initial_accumulator_value Initial value for accumulator (default: 0.0)
 * @param eps Term for numerical stability (default: 1e-10)
 *
 * @par Complexity
 * - Time: O(P) per step
 * - Space: O(P) for accumulator
 *
 * @code
 * // Standard Adagrad for sparse embeddings
 * auto optimizer = Adagrad(model.parameters(), 1e-2);
 *
 * // With learning rate decay
 * auto optimizer = Adagrad(model.parameters(), 1e-2, 1e-6);
 *
 * // Training loop
 * for (int epoch = 0; epoch < epochs; ++epoch) {
 *     optimizer.zero_grad();
 *     auto loss = model.forward(input);
 *     loss.backward();
 *     optimizer.step();  // Automatic per-parameter learning rate adaptation
 * }
 * @endcode
 *
 * @see RMSprop (addresses monotonic decay issue), Adadelta (parameter-free variant)
 */
class Adagrad : public Optimizer {
public:
    /**
     * @brief Construct Adagrad optimizer
     *
     * @param params Parameters to optimize
     * @param lr Learning rate
     * @param lr_decay Additional learning rate decay
     * @param weight_decay L2 regularization coefficient
     * @param initial_accumulator_value Starting value for gradient accumulator
     * @param eps Epsilon for numerical stability
     */
    Adagrad(std::vector<std::shared_ptr<Variable>> params,
            double lr = 1e-2,
            double lr_decay = 0.0,
            double weight_decay = 0.0,
            double initial_accumulator_value = 0.0,
            double eps = 1e-10);

    /**
     * @brief Construct from `ParamGroup`s (audit D.4).
     */
    explicit Adagrad(std::vector<optim::ParamGroup> groups,
                     double default_lr = 1e-2,
                     double default_lr_decay = 0.0,
                     double default_weight_decay = 0.0,
                     double default_initial_accumulator_value = 0.0,
                     double default_eps = 1e-10);

    /**
     * @brief Perform single Adagrad optimization step
     *
     * Accumulates squared gradients and updates parameters with
     * adaptive learning rates.
     *
     * @pre Gradients must be computed via backward()
     */
    auto step_impl() -> void override;

    /**
     * @brief Zero all parameter gradients (QQ.14: see Optimizer::zero_grad).
     */
    auto zero_grad(bool set_to_none = true) -> void;

    /**
     * @brief Set new learning rate
     * @param lr New learning rate
     */
    auto set_lr(double lr) -> void;

    /**
     * @brief Get current learning rate
     * @return Current learning rate (after decay)
     */
    auto get_lr() const -> double;

    /**
     * @brief Get optimizer state for serialization
     *
     * Returns accumulated squared gradients and step count.
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
    // Audit K.1: extend sum_ when add_param_group appends new
    // parameters mid-training.
    auto on_parameters_appended_(size_t old_count, size_t new_count) -> void override;

private:
    double lr_;
    double lr_decay_;
    double weight_decay_;
    double initial_accumulator_value_;
    double eps_;

    int64_t step_count_{0};           ///< Number of steps taken (for lr_decay)
    std::vector<Tensor> sum_;         ///< Accumulated sum of squared gradients G_t

    /**
     * @brief Initialize accumulator buffers to initial_accumulator_value
     */
    auto initialize_buffers() -> void;

    /**
     * @brief Compute effective learning rate with decay
     * @return lr / (1 + (step_count - 1) * lr_decay)
     */
    auto effective_lr() const -> double;
};

} // namespace optim
} // namespace tenzor
