/**
 * @file scheduler.hpp
 * @brief Learning rate scheduling strategies
 *
 * Provides learning rate schedules that adjust the learning rate during training
 * to improve convergence and final performance.
 */

#pragma once

#include "optimizer.hpp"
#include <memory>
#include <cmath>
#include <numbers>

namespace tenzor {
namespace optim {

// Forward declarations
class SGD;
class Adam;
class AdamW;

/**
 * @brief Base class for learning rate schedulers
 *
 * Learning rate schedules adjust the learning rate during training:
 * - Start with higher learning rate for fast initial progress
 * - Reduce learning rate to fine-tune and converge to optimum
 *
 * **Common Patterns:**
 * - Step decay: Reduce LR by factor every N epochs
 * - Exponential decay: Smooth continuous reduction
 * - Cosine annealing: Gradual cosine-shaped reduction
 *
 * @par Usage
 * Call scheduler.step() once per epoch (after optimizer.step())
 *
 * @code
 * auto scheduler = StepLR(optimizer, 30, 0.1);
 * for (int epoch = 0; epoch < 100; ++epoch) {
 *     train_one_epoch();
 *     scheduler.step();  // Reduce LR every 30 epochs
 * }
 * @endcode
 *
 * @see StepLR, ExponentialLR, CosineAnnealingLR
 */
class LRScheduler {
public:
    virtual ~LRScheduler() = default;

    /**
     * @brief Update learning rate according to schedule
     *
     * Typically called once per epoch after training.
     * Updates the optimizer's learning rate based on the scheduling strategy.
     */
    virtual auto step() -> void = 0;

    /**
     * @brief Get the last computed learning rate
     * @return Current learning rate value
     */
    virtual auto get_last_lr() const -> double = 0;

    /**
     * @brief Get the current learning rate (alias for get_last_lr)
     * @return Current learning rate value
     */
    auto get_lr() const -> double { return get_last_lr(); }

protected:
    LRScheduler() = default;
};

/**
 * @brief Step learning rate scheduler
 *
 * Multiplies learning rate by gamma every step_size epochs:
 *
 * \f[
 * \eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}
 * \f]
 *
 * where:
 * - \f$\eta_0\f$: initial learning rate
 * - \f$\gamma\f$: multiplicative factor (default: 0.1)
 * - \f$s\f$: step size in epochs
 * - \f$t\f$: current epoch
 *
 * **Use Cases:**
 * - Most common LR schedule
 * - Simple and effective
 * - Good default choice
 *
 * **Typical Configuration:**
 * - step_size: 30-50 epochs
 * - gamma: 0.1 (reduce LR by 10x)
 *
 * @param optimizer Optimizer to schedule (SGD, Adam, or AdamW)
 * @param step_size Period of learning rate decay in epochs
 * @param gamma Multiplicative factor of learning rate decay (default: 0.1)
 *
 * @code
 * auto scheduler = StepLR(optimizer, 30, 0.1);
 * // LR = initial_lr at epochs 0-29
 * // LR = initial_lr * 0.1 at epochs 30-59
 * // LR = initial_lr * 0.01 at epochs 60-89
 * @endcode
 *
 * @see ExponentialLR, CosineAnnealingLR
 */
class StepLR : public LRScheduler {
public:
    // Constructor for SGD optimizer
    StepLR(SGD& optimizer, int step_size, double gamma = 0.1);

    // Constructor for Adam optimizer
    StepLR(Adam& optimizer, int step_size, double gamma = 0.1);

    // Constructor for AdamW optimizer
    StepLR(AdamW& optimizer, int step_size, double gamma = 0.1);

    auto step() -> void override;
    auto get_last_lr() const -> double override { return last_lr_; }

    // Get current epoch
    auto get_epoch() const -> int { return epoch_; }

private:
    enum class OptimizerType { SGD, Adam, AdamW };

    union OptimizerPtr {
        SGD* sgd;
        Adam* adam;
        AdamW* adamw;
        OptimizerPtr() : sgd(nullptr) {}
    };

    OptimizerPtr optimizer_;
    OptimizerType optimizer_type_;
    double base_lr_;
    double last_lr_;
    int step_size_;
    double gamma_;
    int epoch_;

    auto update_lr() -> void;
    auto get_current_lr() const -> double;
    auto set_optimizer_lr(double lr) -> void;
};

/**
 * @brief Exponential learning rate scheduler
 *
 * Multiplies learning rate by gamma every epoch:
 *
 * \f[
 * \eta_t = \eta_0 \cdot \gamma^t
 * \f]
 *
 * Provides smooth, continuous decay unlike step-based schedules.
 *
 * **Use Cases:**
 * - When you want gradual LR reduction
 * - Smooth optimization landscapes
 * - Alternative to step decay
 *
 * **Typical Configuration:**
 * - gamma: 0.95-0.99 for gradual decay
 * - gamma: 0.9 for faster decay
 *
 * @param optimizer Optimizer to schedule (SGD, Adam, or AdamW)
 * @param gamma Multiplicative factor (typical: 0.95-0.99)
 *
 * @code
 * auto scheduler = ExponentialLR(optimizer, 0.95);
 * // After 10 epochs: LR = initial_lr * 0.95^10 ≈ initial_lr * 0.60
 * // After 20 epochs: LR = initial_lr * 0.95^20 ≈ initial_lr * 0.36
 * @endcode
 *
 * @see StepLR, CosineAnnealingLR
 */
class ExponentialLR : public LRScheduler {
public:
    // Constructor for SGD optimizer
    ExponentialLR(SGD& optimizer, double gamma);

    // Constructor for Adam optimizer
    ExponentialLR(Adam& optimizer, double gamma);

    // Constructor for AdamW optimizer
    ExponentialLR(AdamW& optimizer, double gamma);

    auto step() -> void override;
    auto get_last_lr() const -> double override { return last_lr_; }

    // Get current epoch
    auto get_epoch() const -> int { return epoch_; }

private:
    enum class OptimizerType { SGD, Adam, AdamW };

    union OptimizerPtr {
        SGD* sgd;
        Adam* adam;
        AdamW* adamw;
        OptimizerPtr() : sgd(nullptr) {}
    };

    OptimizerPtr optimizer_;
    OptimizerType optimizer_type_;
    double base_lr_;
    double last_lr_;
    double gamma_;
    int epoch_;

    auto update_lr() -> void;
    auto get_current_lr() const -> double;
    auto set_optimizer_lr(double lr) -> void;
};

/**
 * @brief Cosine annealing learning rate scheduler
 *
 * Anneals learning rate using cosine function:
 *
 * \f[
 * \eta_t = \eta_{min} + \frac{\eta_0 - \eta_{min}}{2}\left(1 + \cos\left(\frac{\pi t}{T_{max}}\right)\right)
 * \f]
 *
 * Provides smooth decay with fast initial reduction and gradual final approach.
 *
 * **Advantages:**
 * - Smooth, continuous schedule
 * - No hyperparameters to tune (besides T_max)
 * - Often better than step/exponential decay
 * - Popular in modern deep learning
 *
 * **Use Cases:**
 * - Training transformers and large models
 * - When you know total number of epochs
 * - Alternative to step decay with better properties
 *
 * **Behavior:**
 * - Fast reduction in first half of training
 * - Gradual fine-tuning in second half
 * - Reaches eta_min at T_max epochs
 *
 * @param optimizer Optimizer to schedule (SGD, Adam, or AdamW)
 * @param T_max Maximum number of epochs (total training epochs)
 * @param eta_min Minimum learning rate (default: 0.0)
 *
 * @code
 * auto scheduler = CosineAnnealingLR(optimizer, 100, 0.0);
 * // Smooth cosine decay from initial_lr to 0 over 100 epochs
 * // At epoch 50: LR ≈ initial_lr / 2
 * // At epoch 100: LR ≈ 0
 * @endcode
 *
 * @see StepLR, ExponentialLR
 */
class CosineAnnealingLR : public LRScheduler {
public:
    // Constructor for SGD optimizer
    CosineAnnealingLR(SGD& optimizer, int T_max, double eta_min = 0.0);

    // Constructor for Adam optimizer
    CosineAnnealingLR(Adam& optimizer, int T_max, double eta_min = 0.0);

    // Constructor for AdamW optimizer
    CosineAnnealingLR(AdamW& optimizer, int T_max, double eta_min = 0.0);

    auto step() -> void override;
    auto get_last_lr() const -> double override { return last_lr_; }

    // Get current epoch
    auto get_epoch() const -> int { return epoch_; }

private:
    enum class OptimizerType { SGD, Adam, AdamW };

    union OptimizerPtr {
        SGD* sgd;
        Adam* adam;
        AdamW* adamw;
        OptimizerPtr() : sgd(nullptr) {}
    };

    OptimizerPtr optimizer_;
    OptimizerType optimizer_type_;
    double base_lr_;
    double last_lr_;
    int T_max_;
    double eta_min_;
    int epoch_;

    auto update_lr() -> void;
    auto get_current_lr() const -> double;
    auto set_optimizer_lr(double lr) -> void;
};

} // namespace optim
} // namespace tenzor
