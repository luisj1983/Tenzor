/**
 * @file adam.hpp
 * @brief Adam and AdamW optimizers
 *
 * Implements Adam (Adaptive Moment Estimation) and AdamW (Adam with decoupled weight decay).
 */

#pragma once

#include "optimizer.hpp"

namespace tenzor {
namespace optim {

/**
 * @brief Adam (Adaptive Moment Estimation) optimizer
 *
 * Computes adaptive learning rates for each parameter using estimates of first
 * and second moments of the gradients:
 *
 * \f[
 * m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
 * v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
 * \hat{m}_t = \frac{m_t}{1 - \beta_1^t} \\
 * \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
 * \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
 * \f]
 *
 * where:
 * - \f$m_t\f$: first moment (mean) estimate
 * - \f$v_t\f$: second moment (variance) estimate
 * - \f$\beta_1, \beta_2\f$: exponential decay rates
 * - \f$\eta\f$: learning rate
 * - \f$\epsilon\f$: numerical stability constant
 *
 * **Advantages:**
 * - Adaptive learning rates per parameter
 * - Works well with sparse gradients
 * - Requires little hyperparameter tuning
 * - Default choice for many deep learning tasks
 *
 * **Recommended Hyperparameters:**
 * - lr: 1e-3 (default, often works well)
 * - beta1: 0.9 (first moment decay)
 * - beta2: 0.999 (second moment decay)
 * - eps: 1e-8 (numerical stability)
 * - weight_decay: 0.0 (for AdamW, use 0.01)
 *
 * @param params Parameters to optimize
 * @param lr Learning rate (default: 1e-3)
 * @param beta1 First moment decay rate (default: 0.9)
 * @param beta2 Second moment decay rate (default: 0.999)
 * @param eps Term for numerical stability (default: 1e-8)
 * @param weight_decay L2 penalty (default: 0.0). Adam applies coupled weight decay
 *        (L2 regularization added to gradients before moment updates). For decoupled
 *        weight decay (applied directly to parameters), use AdamW instead.
 * @param amsgrad Use AMSGrad variant (default: false)
 *
 * @par Complexity
 * - Time: O(P) per step
 * - Space: O(2P) for moment estimates (or 3P with AMSGrad)
 *
 * @code
 * auto optimizer = Adam(model.parameters(), 1e-3, 0.9, 0.999, 1e-8);
 * @endcode
 *
 * @see AdamW, SGD
 */
class Adam : public Optimizer {
public:
    Adam(std::vector<std::shared_ptr<Variable>> params,
         double lr = 1e-3,
         double beta1 = 0.9,
         double beta2 = 0.999,
         double eps = 1e-8,
         double weight_decay = 0.0,
         bool amsgrad = false);

    /** @brief Perform single Adam step */
    auto step_impl() -> void override;

    /** @brief Set new learning rate */
    auto set_lr(double lr) -> void override;  // M15: explicit override marker

    /** @brief Get current learning rate */
    auto get_lr() const -> double override;  // M15: explicit override marker

    /** @brief Get optimizer state (moment estimates) for serialization */
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;

    /** @brief Load optimizer state from dictionary */
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;


    /** @brief Hyperparameters: lr, beta1, beta2, eps, weight_decay, amsgrad. */
    auto defaults() const -> std::unordered_map<std::string, double> override {
        return {
            {"lr",           lr_},
            {"beta1",        beta1_},
            {"beta2",        beta2_},
            {"eps",          eps_},
            {"weight_decay", weight_decay_},
            {"amsgrad",      amsgrad_ ? 1.0 : 0.0},
        };
    }

private:
    double lr_;
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;
    bool amsgrad_;

    int64_t step_count_{0};
    std::vector<Tensor> exp_avg_;       // First moment estimates
    std::vector<Tensor> exp_avg_sq_;    // Second moment estimates
    std::vector<Tensor> max_exp_avg_sq_; // Max of second moment (for AMSGrad)

    auto initialize_buffers() -> void;
};

/**
 * @brief AdamW optimizer (Adam with decoupled weight decay)
 *
 * Implements Adam with decoupled weight decay regularization:
 *
 * \f[
 * \theta_t = \theta_{t-1} - \eta\left(\frac{1}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t + \lambda\theta_{t-1}\right)
 * \f]
 *
 * **Key Difference from Adam:**
 * - Weight decay is applied directly to parameters (decoupled from gradient)
 * - Adam applies weight decay to gradients (coupled)
 * - AdamW provides better regularization and generalization
 *
 * **When to Use AdamW over Adam:**
 * - Training transformers and large models
 * - When weight decay/L2 regularization is needed
 * - For improved generalization
 *
 * **Recommended Hyperparameters:**
 * - lr: 1e-3 to 3e-4
 * - beta1: 0.9
 * - beta2: 0.999
 * - weight_decay: 0.01 to 0.1 (typical: 0.01)
 *
 * @param params Parameters to optimize
 * @param lr Learning rate (default: 1e-3)
 * @param beta1 First moment decay rate (default: 0.9)
 * @param beta2 Second moment decay rate (default: 0.999)
 * @param eps Term for numerical stability (default: 1e-8)
 * @param weight_decay Weight decay coefficient (default: 0.01)
 * @param amsgrad Use AMSGrad variant (default: false)
 *
 * @par Complexity
 * - Time: O(P) per step
 * - Space: O(2P) for moment estimates (or 3P with AMSGrad)
 *
 * @code
 * // AdamW for transformer training
 * auto optimizer = AdamW(model.parameters(), 3e-4, 0.9, 0.999, 1e-8, 0.01);
 * @endcode
 *
 * @see Adam, SGD
 */
class AdamW : public Optimizer {
public:
    AdamW(std::vector<std::shared_ptr<Variable>> params,
          double lr = 1e-3,
          double beta1 = 0.9,
          double beta2 = 0.999,
          double eps = 1e-8,
          double weight_decay = 0.01,
          bool amsgrad = false);

    /** @brief Perform single AdamW step with decoupled weight decay */
    auto step_impl() -> void override;

    /** @brief Set new learning rate */
    auto set_lr(double lr) -> void override;  // M15: explicit override marker

    /** @brief Get current learning rate */
    auto get_lr() const -> double override;  // M15: explicit override marker

    /** @brief Get optimizer state for serialization */
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;

    /** @brief Load optimizer state from dictionary */
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;


    /** @brief Hyperparameters: lr, beta1, beta2, eps, weight_decay, amsgrad. */
    auto defaults() const -> std::unordered_map<std::string, double> override {
        return {
            {"lr",           lr_},
            {"beta1",        beta1_},
            {"beta2",        beta2_},
            {"eps",          eps_},
            {"weight_decay", weight_decay_},
            {"amsgrad",      amsgrad_ ? 1.0 : 0.0},
        };
    }

private:
    double lr_;
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;
    bool amsgrad_;

    int64_t step_count_{0};
    std::vector<Tensor> exp_avg_;
    std::vector<Tensor> exp_avg_sq_;
    std::vector<Tensor> max_exp_avg_sq_;

    auto initialize_buffers() -> void;
};

} // namespace optim
} // namespace tenzor
