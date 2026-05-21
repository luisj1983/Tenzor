/**
 * @file lion.hpp
 * @brief Lion (EvoLved Sign Momentum) optimizer
 *
 * Lion was proposed in "Symbolic Discovery of Optimization Algorithms"
 * (Chen et al., 2023). It maintains a single momentum buffer per
 * parameter and uses only the sign of the momentum to update, which is
 * ~2x more memory-efficient than Adam while matching or exceeding Adam
 * performance on large-scale vision and language training.
 */

#pragma once

#include "optimizer.hpp"

namespace tenzor {
namespace optim {

/**
 * @brief Lion (EvoLved Sign Momentum) optimizer
 *
 * Lion tracks a single running momentum buffer per parameter and updates
 * with only the sign of a beta1-weighted combination. Because the update
 * is sign-based, it has uniform magnitude per element and is invariant to
 * the gradient scale, which changes the effective learning-rate regime:
 * Lion's learning rate is typically **3×–10× smaller** than Adam's for
 * the same model.
 *
 * Update rule (per step t):
 * \f[
 * c_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
 * \theta_t = \theta_{t-1} - \eta \bigl(\text{sign}(c_t) + \lambda \theta_{t-1}\bigr) \\
 * m_t = \beta_2 m_{t-1} + (1 - \beta_2) g_t
 * \f]
 *
 * Notice that the update direction `c_t` and the momentum state `m_t`
 * use **different** decay rates — this is what distinguishes Lion from a
 * naive sign-SGD.
 *
 * **Recommended hyperparameters:**
 * - lr: 1e-4 (transformers), 3e-5 (ViT). Typically 3–10× smaller than Adam.
 * - beta1: 0.9
 * - beta2: 0.99
 * - weight_decay: 0.01 (3–10× larger than Adam to compensate for the smaller lr)
 *
 * **Advantages vs Adam / AdamW:**
 * - Half the optimizer memory (one momentum vs two moment estimates)
 * - Competitive or better accuracy on large-scale training
 * - Implementation is much simpler — no bias correction, no eps, no sqrt
 *
 * @see Adam, AdamW
 */
class Lion : public Optimizer {
public:
    Lion(std::vector<std::shared_ptr<Variable>> params,
         double lr = 1e-4,
         double beta1 = 0.9,
         double beta2 = 0.99,
         double weight_decay = 0.0);

    /**
     * @brief Construct from a list of parameter groups (audit D.4).
     *
     * Per-group hyperparameters override the defaults supplied here when
     * resolved inside @ref step_impl.
     */
    explicit Lion(std::vector<optim::ParamGroup> groups,
                  double default_lr = 1e-4,
                  double default_beta1 = 0.9,
                  double default_beta2 = 0.99,
                  double default_weight_decay = 0.0);

    /** @brief Perform a single Lion step */
    auto step_impl() -> void override;

    /** @brief Set new learning rate */
    auto set_lr(double lr) -> void;

    /** @brief Get current learning rate */
    auto get_lr() const -> double;

    /** @brief Get optimizer state (momentum buffers + config) */
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;

    /** @brief Load optimizer state from dictionary */
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    double lr_;
    double beta1_;
    double beta2_;
    double weight_decay_;

    int64_t step_count_{0};
    std::vector<Tensor> momentum_;  // Running momentum buffer (one per parameter)

    auto initialize_buffers() -> void;
};

} // namespace optim
} // namespace tenzor
