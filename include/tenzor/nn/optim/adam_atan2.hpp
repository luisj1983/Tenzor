/**
 * @file adam_atan2.hpp
 * @brief Adam-atan2 optimizer from the HRM paper
 *
 * Implements Adam with atan2-based update rule for bounded gradient steps.
 * This provides more stable training by naturally bounding the update magnitude.
 *
 * Reference: "Hierarchical Reasoning Model" (Wang et al., 2025)
 */

#pragma once

#include "optimizer.hpp"

namespace tenzor {
namespace optim {

/**
 * @brief Adam-atan2 optimizer
 *
 * A variant of Adam that uses atan2 for computing parameter updates,
 * providing naturally bounded update magnitudes. This is particularly
 * useful for training HRM models where stability is crucial.
 *
 * Update rule:
 * \f[
 * m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
 * v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
 * \hat{m}_t = \frac{m_t}{1 - \beta_1^t} \\
 * \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
 * \theta_t = \theta_{t-1} - \eta \cdot \text{atan2}(\hat{m}_t, \sqrt{\hat{v}_t} + \epsilon)
 * \f]
 *
 * The atan2 function naturally bounds the update to [-π/2, π/2], providing:
 * - Bounded updates even for large gradient/variance ratios
 * - Smooth interpolation between SGD-like and sign-based updates
 * - Better numerical stability for small batch sizes
 *
 * **When to Use:**
 * - Training HRM models
 * - Small batch/sample scenarios
 * - When training stability is critical
 *
 * **Recommended Hyperparameters (from HRM paper):**
 * - lr: 1e-3 to 3e-4
 * - beta1: 0.9
 * - beta2: 0.999
 * - eps: 1e-8
 * - weight_decay: 0.01 (decoupled, like AdamW)
 *
 * @code
 * // Training HRM with Adam-atan2
 * auto optimizer = AdamAtan2(model.parameters(), 1e-3, 0.9, 0.999, 1e-8, 0.01);
 *
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     optimizer.zero_grad();
 *     auto output = model.forward(input);
 *     auto loss = criterion(output, targets);
 *     loss.backward();
 *     optimizer.step();
 * }
 * @endcode
 *
 * @see Adam, AdamW
 */
class AdamAtan2 : public Optimizer {
public:
    /**
     * @brief Construct Adam-atan2 optimizer
     *
     * @param params Parameters to optimize
     * @param lr Learning rate (default: 1e-3)
     * @param beta1 First moment decay rate (default: 0.9)
     * @param beta2 Second moment decay rate (default: 0.999)
     * @param eps Numerical stability term (default: 1e-8)
     * @param weight_decay Decoupled weight decay (default: 0.01)
     * @param amsgrad Use AMSGrad variant (default: false)
     */
    AdamAtan2(std::vector<std::shared_ptr<Variable>> params,
              double lr = 1e-3,
              double beta1 = 0.9,
              double beta2 = 0.999,
              double eps = 1e-8,
              double weight_decay = 0.01,
              bool amsgrad = false);

    /**
     * @brief Construct Adam-atan2 with per-group hyperparameters (audit D.4)
     *
     * Each `ParamGroup` may override `lr`, `weight_decay`, `beta1`, `beta2`,
     * `eps`; otherwise the corresponding default is used. `amsgrad` remains
     * a single optimiser-wide setting (no ParamGroup field).
     */
    explicit AdamAtan2(std::vector<optim::ParamGroup> groups,
                       double default_lr = 1e-3,
                       double default_beta1 = 0.9,
                       double default_beta2 = 0.999,
                       double default_eps = 1e-8,
                       double default_weight_decay = 0.01,
                       bool amsgrad = false);

    /**
     * @brief Perform single optimization step with atan2 update
     */
    auto step_impl() -> void override;

    /**
     * @brief Set new learning rate
     * @param lr New learning rate
     */
    auto set_lr(double lr) -> void override;

    /**
     * @brief Get current learning rate
     * @return Current learning rate
     */
    auto get_lr() const -> double override;

    /**
     * @brief Get optimizer state for serialization
     * @return Map of state tensors
     */
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;

    /**
     * @brief Load optimizer state from dictionary
     * @param state State dictionary to load
     */
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

    /**
     * @brief Enable or disable per-step statistics tracking
     *
     * When disabled (default), avoids GPU synchronization overhead from
     * extracting scalar values during the optimization step.
     *
     * @param enable Whether to enable statistics tracking
     */
    auto set_track_statistics(bool enable) -> void { track_statistics_ = enable; }

    /**
     * @brief Get statistics about update magnitudes
     *
     * Only populated when track_statistics is enabled via set_track_statistics().
     */
    struct UpdateStats {
        double avg_update_magnitude;  ///< Average |update| across all params
        double max_update_magnitude;  ///< Maximum |update| encountered
        double avg_gradient_magnitude; ///< Average |gradient|
    };
    auto last_update_stats() const -> const UpdateStats& { return update_stats_; }

protected:
    // Audit K.1: extend exp_avg_ / exp_avg_sq_ (and max_exp_avg_sq_ when
    // amsgrad_) when add_param_group appends new parameters mid-training.
    auto on_parameters_appended_(size_t old_count, size_t new_count) -> void override;

private:
    double lr_;
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;
    bool amsgrad_;
    bool track_statistics_{false};

    int64_t step_count_{0};
    std::vector<Tensor> exp_avg_;        ///< First moment estimates
    std::vector<Tensor> exp_avg_sq_;     ///< Second moment estimates
    std::vector<Tensor> max_exp_avg_sq_; ///< Max second moment (AMSGrad)

    UpdateStats update_stats_;

    /**
     * @brief Initialize momentum buffers
     */
    auto initialize_buffers() -> void;
};

/**
 * @brief Linear warmup learning rate scheduler
 *
 * Linearly increases learning rate from 0 to base_lr over warmup_steps.
 * Commonly used with Adam-atan2 for HRM training.
 */
class LinearWarmup {
public:
    /**
     * @brief Construct linear warmup scheduler
     *
     * @param optimizer Optimizer to schedule
     * @param warmup_steps Number of warmup steps
     * @param base_lr Base learning rate to warm up to
     */
    LinearWarmup(Optimizer& optimizer, int64_t warmup_steps, double base_lr);

    /**
     * @brief Update learning rate based on current step
     *
     * @param step Current training step
     */
    auto update(int64_t step) -> void;

    /**
     * @brief Get current learning rate
     */
    auto get_lr() const -> double { return current_lr_; }

private:
    Optimizer& optimizer_;
    int64_t warmup_steps_;
    double base_lr_;
    double current_lr_;
};

} // namespace optim
} // namespace tenzor
