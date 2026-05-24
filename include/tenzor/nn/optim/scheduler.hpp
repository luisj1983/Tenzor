/**
 * @file scheduler.hpp
 * @brief Learning rate scheduling strategies
 *
 * Provides learning rate schedules that adjust the learning rate during training
 * to improve convergence and final performance.
 */

#pragma once

#include "optimizer.hpp"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <numbers>

namespace tenzor {
namespace optim {

// Forward declarations
class SGD;
class Adam;
class AdamW;
class RMSprop;
class Adagrad;
class Adadelta;

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

    /**
     * @brief Get last learning rates for all parameter groups
     * @return Vector of learning rates (default: single element)
     */
    virtual auto get_last_lr_vec() const -> std::vector<double> {
        return {get_last_lr()};
    }

    /**
     * @brief Serialise scheduler state for checkpointing (audit Q.11).
     *
     * Returns scalar Tensors mirroring the optimiser state_dict convention
     * (Float64/Int64 1-element CPU tensors). The base implementation
     * captures @c last_lr — every derived class is expected to call the
     * base and overlay its own counters (epoch_, step_count_, T_cur_, …).
     *
     * @return Map of state name to scalar Tensor on CPU.
     */
    virtual auto state_dict() const -> std::unordered_map<std::string, Tensor>;

    /**
     * @brief Restore scheduler state from a state_dict (audit Q.11).
     *
     * @param state Map produced by a previous @c state_dict call.
     */
    virtual auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state) -> void;

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
    StepLR(Optimizer& optimizer, int step_size, double gamma = 0.1);

    auto step() -> void override;
    auto get_last_lr() const -> double override { return last_lr_; }

    // Get current epoch
    auto get_epoch() const -> int { return epoch_; }

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    Optimizer* optimizer_;
    double base_lr_;
    double last_lr_;
    int step_size_;
    double gamma_;
    int epoch_;

    auto update_lr() -> void;
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
    ExponentialLR(Optimizer& optimizer, double gamma);

    auto step() -> void override;
    auto get_last_lr() const -> double override { return last_lr_; }

    // Get current epoch
    auto get_epoch() const -> int { return epoch_; }

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    Optimizer* optimizer_;
    double base_lr_;
    double last_lr_;
    double gamma_;
    int epoch_;

    auto update_lr() -> void;
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
    CosineAnnealingLR(Optimizer& optimizer, int T_max, double eta_min = 0.0);

    auto step() -> void override;
    auto get_last_lr() const -> double override { return last_lr_; }

    // Get current epoch
    auto get_epoch() const -> int { return epoch_; }

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    Optimizer* optimizer_;
    double base_lr_;
    double last_lr_;
    int T_max_;
    double eta_min_;
    int epoch_;

    auto update_lr() -> void;
};

/**
 * @brief Reduce learning rate on plateau scheduler
 *
 * Reduces learning rate when a metric has stopped improving.
 * Monitors a quantity (e.g., validation loss) and reduces LR by factor
 * when no improvement is seen for a patience number of epochs.
 *
 * \f[
 * \text{if no improvement for patience epochs: } \eta_{new} = \eta \cdot \text{factor}
 * \f]
 *
 * **Key Features:**
 * - Metric-based (not epoch-based) - call step(metric) instead of step()
 * - Automatic LR reduction when training plateaus
 * - Configurable patience and threshold
 * - Cooldown period after reduction
 *
 * **Use Cases:**
 * - When you don't know optimal schedule in advance
 * - Dynamic adjustment based on validation metrics
 * - Automatic hyperparameter tuning
 *
 * **Typical Configuration:**
 * - mode: "min" for loss, "max" for accuracy
 * - factor: 0.1 (reduce by 10x)
 * - patience: 10 epochs
 * - threshold: 1e-4 (improvement threshold)
 *
 * @param optimizer Optimizer to schedule
 * @param mode "min" for loss (reduce when not decreasing), "max" for metrics (reduce when not increasing)
 * @param factor Multiplicative factor of learning rate decay (default: 0.1)
 * @param patience Number of epochs with no improvement before reducing LR (default: 10)
 * @param threshold Threshold for measuring improvement (default: 1e-4)
 * @param threshold_mode "rel" for relative threshold, "abs" for absolute (default: "rel")
 * @param cooldown Epochs to wait before resuming monitoring after LR reduction (default: 0)
 * @param min_lr Minimum learning rate (default: 0.0)
 * @param eps Minimum decay for learning rate (default: 1e-8)
 *
 * @code
 * auto scheduler = ReduceLROnPlateau(optimizer, "min", 0.1, 10);
 * for (int epoch = 0; epoch < 100; ++epoch) {
 *     train_one_epoch();
 *     double val_loss = validate();
 *     scheduler.step(val_loss);  // Pass metric, not epoch-based!
 * }
 * @endcode
 *
 * @see CyclicLR, OneCycleLR
 */
class ReduceLROnPlateau : public LRScheduler {
public:
    // Constructor for SGD optimizer
    ReduceLROnPlateau(SGD& optimizer,
                     const std::string& mode = "min",
                     double factor = 0.1,
                     int64_t patience = 10,
                     double threshold = 1e-4,
                     const std::string& threshold_mode = "rel",
                     int64_t cooldown = 0,
                     double min_lr = 0.0,
                     double eps = 1e-8);

    // Constructor for Adam optimizer
    ReduceLROnPlateau(Adam& optimizer,
                     const std::string& mode = "min",
                     double factor = 0.1,
                     int64_t patience = 10,
                     double threshold = 1e-4,
                     const std::string& threshold_mode = "rel",
                     int64_t cooldown = 0,
                     double min_lr = 0.0,
                     double eps = 1e-8);

    // Constructor for AdamW optimizer
    ReduceLROnPlateau(AdamW& optimizer,
                     const std::string& mode = "min",
                     double factor = 0.1,
                     int64_t patience = 10,
                     double threshold = 1e-4,
                     const std::string& threshold_mode = "rel",
                     int64_t cooldown = 0,
                     double min_lr = 0.0,
                     double eps = 1e-8);

    // Metric-based step (not epoch-based!)
    auto step(double metric) -> void;

    // Override base step() to throw error (must use step(metric))
    auto step() -> void override {
        throw std::runtime_error("ReduceLROnPlateau requires metric argument: use step(metric)");
    }

    auto get_last_lr() const -> double override { return last_lr_; }

    // Get number of bad epochs
    auto get_num_bad_epochs() const -> int64_t { return num_bad_epochs_; }

    // Check if in cooldown
    auto in_cooldown() const -> bool { return cooldown_counter_ > 0; }

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state) -> void override;

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
    std::string mode_;
    std::string threshold_mode_;
    double factor_;
    double threshold_;
    double min_lr_;
    double eps_;
    int64_t patience_;
    int64_t cooldown_;
    double best_metric_;
    int64_t num_bad_epochs_;
    int64_t cooldown_counter_;
    double last_lr_;

    auto is_better(double current, double best) const -> bool;
    auto reduce_lr() -> void;
    auto get_current_lr() const -> double;
    auto set_optimizer_lr(double lr) -> void;
};

/**
 * @brief Cyclic learning rate scheduler
 *
 * Cycles learning rate between base_lr and max_lr with configurable cycle shape.
 * Learning rate oscillates during training, which can help escape local minima
 * and speed up convergence.
 *
 * **Cycle Modes:**
 * - **triangular**: Linear increase then decrease (constant amplitude)
 * - **triangular2**: Same but max_lr halves each cycle
 * - **exp_range**: Exponential scaling with gamma per iteration
 *
 * **Important:** Call step() every batch, not every epoch!
 *
 * **Advantages:**
 * - Helps escape saddle points
 * - Often faster convergence
 * - Can achieve better final accuracy
 * - Self-regularizing effect
 *
 * **Use Cases:**
 * - Training CNNs
 * - When stuck in local minima
 * - Alternative to fixed LR schedule
 *
 * **Typical Configuration:**
 * - step_size_up: 2000-8000 iterations (2-8 epochs)
 * - max_lr: 3-5x base_lr
 * - mode: "triangular" or "triangular2"
 *
 * @param optimizer Optimizer to schedule
 * @param base_lr Minimum learning rate in cycle
 * @param max_lr Maximum learning rate in cycle
 * @param step_size_up Number of iterations in increasing phase (default: 2000)
 * @param step_size_down Number of iterations in decreasing phase (default: equals step_size_up)
 * @param mode Cycle shape: "triangular", "triangular2", "exp_range" (default: "triangular")
 * @param gamma Scaling factor for exp_range mode (default: 1.0)
 * @param scale_fn Custom scaling function (default: 1.0)
 * @param scale_mode "cycle" or "iterations" (default: "cycle")
 *
 * @code
 * auto scheduler = CyclicLR(optimizer, 0.001, 0.006, 2000);
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     for (auto batch : dataloader) {
 *         optimizer.step();
 *         scheduler.step();  // Call every batch!
 *     }
 * }
 * @endcode
 *
 * @see OneCycleLR, ReduceLROnPlateau
 */
class CyclicLR : public LRScheduler {
public:
    // Constructor for SGD optimizer
    CyclicLR(SGD& optimizer,
             double base_lr, double max_lr,
             int64_t step_size_up = 2000,
             int64_t step_size_down = -1,
             const std::string& mode = "triangular",
             double gamma = 1.0,
             double scale_fn = 1.0,
             const std::string& scale_mode = "cycle");

    // Constructor for Adam optimizer
    CyclicLR(Adam& optimizer,
             double base_lr, double max_lr,
             int64_t step_size_up = 2000,
             int64_t step_size_down = -1,
             const std::string& mode = "triangular",
             double gamma = 1.0,
             double scale_fn = 1.0,
             const std::string& scale_mode = "cycle");

    // Constructor for AdamW optimizer
    CyclicLR(AdamW& optimizer,
             double base_lr, double max_lr,
             int64_t step_size_up = 2000,
             int64_t step_size_down = -1,
             const std::string& mode = "triangular",
             double gamma = 1.0,
             double scale_fn = 1.0,
             const std::string& scale_mode = "cycle");

    auto step() -> void override;
    auto get_last_lr() const -> double override { return last_lr_; }

    // Get current iteration
    auto get_iteration() const -> int64_t { return step_count_; }

    // Get current cycle
    auto get_cycle() const -> int64_t { return step_count_ / cycle_size_; }

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state) -> void override;

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
    double max_lr_;
    int64_t step_size_up_;
    int64_t step_size_down_;
    int64_t cycle_size_;
    std::string mode_;
    double gamma_;
    double scale_fn_;
    std::string scale_mode_;
    int64_t step_count_;
    double last_lr_;

    auto compute_lr() -> double;
    auto get_scale_factor(int64_t cycle) const -> double;
    auto get_current_lr() const -> double;
    auto set_optimizer_lr(double lr) -> void;
};

/**
 * @brief One cycle learning rate scheduler
 *
 * Implements the 1cycle learning rate policy popularized by Leslie Smith.
 * Two-phase schedule:
 * 1. Warmup: Increase from initial_lr to max_lr
 * 2. Annealing: Decrease from max_lr to final_lr
 *
 * \f[
 * \text{Phase 1: } \eta_t \text{ increases from } \frac{\eta_{max}}{div} \text{ to } \eta_{max} \\
 * \text{Phase 2: } \eta_t \text{ decreases from } \eta_{max} \text{ to } \frac{\eta_{max}}{final\_div}
 * \f]
 *
 * **Important:** Call step() every batch, not every epoch!
 *
 * **Advantages:**
 * - Fast convergence (often 5-10x faster)
 * - Better final accuracy
 * - Built-in warmup and annealing
 * - Single hyperparameter (max_lr)
 *
 * **Use Cases:**
 * - Training modern architectures (ResNets, Transformers)
 * - When total training steps are known
 * - Fast training with good results
 *
 * **Typical Configuration:**
 * - max_lr: Find with LR range test
 * - pct_start: 0.3 (30% for warmup)
 * - anneal_strategy: "cos" or "linear"
 *
 * @param optimizer Optimizer to schedule
 * @param max_lr Maximum learning rate
 * @param total_steps Total number of training steps
 * @param epochs Alternative to total_steps
 * @param steps_per_epoch Alternative to total_steps (with epochs)
 * @param pct_start Percentage of cycle for warmup (default: 0.3)
 * @param anneal_strategy "cos" or "linear" annealing (default: "cos")
 * @param div_factor Initial LR = max_lr / div_factor (default: 25.0)
 * @param final_div_factor Final LR = max_lr / final_div_factor (default: 1e4)
 *
 * @code
 * int total_steps = num_epochs * batches_per_epoch;
 * auto scheduler = OneCycleLR(optimizer, 0.1, total_steps);
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     for (auto batch : dataloader) {
 *         optimizer.step();
 *         scheduler.step();  // Call every batch!
 *     }
 * }
 * @endcode
 *
 * @see CyclicLR, CosineAnnealingLR
 */
class OneCycleLR : public LRScheduler {
public:
    // Constructor for SGD optimizer
    OneCycleLR(SGD& optimizer,
               double max_lr,
               int64_t total_steps,
               int64_t epochs = -1,
               int64_t steps_per_epoch = -1,
               double pct_start = 0.3,
               const std::string& anneal_strategy = "cos",
               double div_factor = 25.0,
               double final_div_factor = 1e4);

    // Constructor for Adam optimizer
    OneCycleLR(Adam& optimizer,
               double max_lr,
               int64_t total_steps,
               int64_t epochs = -1,
               int64_t steps_per_epoch = -1,
               double pct_start = 0.3,
               const std::string& anneal_strategy = "cos",
               double div_factor = 25.0,
               double final_div_factor = 1e4);

    // Constructor for AdamW optimizer
    OneCycleLR(AdamW& optimizer,
               double max_lr,
               int64_t total_steps,
               int64_t epochs = -1,
               int64_t steps_per_epoch = -1,
               double pct_start = 0.3,
               const std::string& anneal_strategy = "cos",
               double div_factor = 25.0,
               double final_div_factor = 1e4);

    auto step() -> void override;
    auto get_last_lr() const -> double override { return last_lr_; }

    // Get current step
    auto get_step() const -> int64_t { return step_count_; }

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state) -> void override;

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
    double max_lr_;
    double pct_start_;
    double div_factor_;
    double final_div_factor_;
    int64_t total_steps_;
    std::string anneal_strategy_;
    int64_t step_count_;
    double last_lr_;

    auto compute_lr() -> double;
    auto anneal_func(double start, double end, double pct) -> double;
    auto get_current_lr() const -> double;
    auto set_optimizer_lr(double lr) -> void;
};

/**
 * @brief Cosine annealing with warm restarts
 *
 * Implements SGDR: Stochastic Gradient Descent with Warm Restarts.
 * Periodically resets learning rate to initial value, creating
 * multiple cosine annealing cycles with increasing periods.
 *
 * \f[
 * \eta_t = \eta_{min} + \frac{\eta_0 - \eta_{min}}{2}\left(1 + \cos\left(\frac{T_{cur}}{T_i}\pi\right)\right)
 * \f]
 *
 * After each restart, the period multiplies by T_mult.
 *
 * **Advantages:**
 * - Escapes local minima via restarts
 * - Can achieve better results than standard schedules
 * - Multiple attempts at different learning rates
 * - Built-in exploration/exploitation balance
 *
 * **Use Cases:**
 * - Long training runs
 * - When you want multiple optimization attempts
 * - Snapshot ensembling (save models at each restart)
 *
 * **Typical Configuration:**
 * - T_0: 10-50 epochs for first restart
 * - T_mult: 1 (constant period) or 2 (doubling period)
 * - eta_min: 0.0 or small positive value
 *
 * @param optimizer Optimizer to schedule
 * @param T_0 Number of iterations for the first restart
 * @param T_mult Period multiplier after each restart (default: 1)
 * @param eta_min Minimum learning rate (default: 0.0)
 *
 * @code
 * auto scheduler = CosineAnnealingWarmRestarts(optimizer, 10, 2);
 * // Restarts at epochs: 10, 30, 70, 150, ...
 * // (periods: 10, 20, 40, 80, ...)
 * for (int epoch = 0; epoch < 200; ++epoch) {
 *     train_one_epoch();
 *     scheduler.step();
 * }
 * @endcode
 *
 * @see CosineAnnealingLR, OneCycleLR
 */
class CosineAnnealingWarmRestarts : public LRScheduler {
public:
    // Constructor for SGD optimizer
    CosineAnnealingWarmRestarts(SGD& optimizer,
                               int64_t T_0,
                               int64_t T_mult = 1,
                               double eta_min = 0.0);

    // Constructor for Adam optimizer
    CosineAnnealingWarmRestarts(Adam& optimizer,
                               int64_t T_0,
                               int64_t T_mult = 1,
                               double eta_min = 0.0);

    // Constructor for AdamW optimizer
    CosineAnnealingWarmRestarts(AdamW& optimizer,
                               int64_t T_0,
                               int64_t T_mult = 1,
                               double eta_min = 0.0);

    auto step() -> void override;
    auto get_last_lr() const -> double override { return last_lr_; }

    // Get current iteration within restart period
    auto get_T_cur() const -> int64_t { return T_cur_; }

    // Get current period length
    auto get_T_i() const -> int64_t { return T_i_; }

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state) -> void override;

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
    int64_t T_0_;
    int64_t T_mult_;
    double eta_min_;
    int64_t T_i_;
    int64_t T_cur_;
    double base_lr_;
    int64_t step_count_;
    double last_lr_;

    auto update_lr() -> void;
    auto get_current_lr() const -> double;
    auto set_optimizer_lr(double lr) -> void;
};

/**
 * @brief Linear warmup scheduler wrapping any base scheduler.
 *
 * Applies a linear warmup from warmup_start_factor * base_lr to base_lr
 * over warmup_steps steps, then delegates to the wrapped base scheduler.
 *
 * During warmup (step < warmup_steps):
 * \f[
 * \eta_t = \eta_0 \cdot \left(\text{start\_factor} + \frac{1 - \text{start\_factor}}{\text{warmup\_steps}} \cdot t\right)
 * \f]
 *
 * After warmup (step >= warmup_steps):
 * Delegates to the base scheduler's step().
 *
 * **Use Cases:**
 * - Transformer training (warmup is critical for Adam)
 * - Preventing early divergence with large learning rates
 * - Combining with any existing scheduler
 *
 * @code
 * // Cosine annealing with linear warmup
 * auto base_scheduler = std::make_shared<CosineAnnealingLR>(optimizer, 100);
 * LinearWarmupScheduler scheduler(optimizer, base_scheduler, 1000, 0.01);
 * // First 1000 steps: linear ramp from 0.01*lr to lr
 * // After 1000 steps: cosine annealing from lr to 0
 * for (int step = 0; step < total_steps; ++step) {
 *     train_one_batch();
 *     scheduler.step();
 * }
 * @endcode
 *
 * @see CosineAnnealingLR, StepLR
 */
class LinearWarmupScheduler : public LRScheduler {
public:
    /**
     * @brief Construct linear warmup scheduler wrapping a base scheduler.
     *
     * @param optimizer Reference to the optimizer (used to set LR directly during warmup)
     * @param base_scheduler Base scheduler to use after warmup completes
     * @param warmup_steps Number of steps for linear warmup
     * @param warmup_start_factor Starting factor (LR starts at base_lr * factor, default: 0.0)
     *
     * @throws std::invalid_argument if warmup_steps < 0
     * @throws std::invalid_argument if warmup_start_factor not in [0, 1]
     */
    LinearWarmupScheduler(Optimizer& optimizer,
                         std::shared_ptr<LRScheduler> base_scheduler,
                         int64_t warmup_steps,
                         double warmup_start_factor = 0.0);

    /**
     * @brief Step the scheduler.
     *
     * During warmup: linearly interpolates learning rate.
     * After warmup: delegates to base scheduler.
     */
    auto step() -> void override;

    auto get_last_lr() const -> double override { return last_lr_; }

    /** @brief Get the current step count. */
    auto get_step_count() const -> int64_t { return step_count_; }

    /** @brief Check if still in warmup phase. */
    auto in_warmup() const -> bool { return step_count_ < warmup_steps_; }

    /** @brief Get the warmup steps count. */
    auto warmup_steps() const -> int64_t { return warmup_steps_; }

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    Optimizer& optimizer_;
    std::shared_ptr<LRScheduler> base_scheduler_;
    int64_t warmup_steps_;
    double warmup_start_factor_;
    double base_lr_;
    double last_lr_;
    int64_t step_count_{0};
};

/**
 * @brief Lambda learning rate scheduler
 *
 * Sets the learning rate to base_lr * lr_lambda(epoch) each step.
 * Provides maximum flexibility via a user-defined function.
 *
 * @code
 * auto scheduler = LambdaLR(optimizer, [](int epoch) {
 *     return 1.0 / (1.0 + 0.1 * epoch);  // Inverse time decay
 * });
 * @endcode
 */
class LambdaLR : public LRScheduler {
public:
    using LrLambda = std::function<double(int)>;

    /**
     * @brief Construct LambdaLR.
     *
     * Audit-4 W.12: an optional @p name tag identifies the lambda the
     * scheduler was built with. state_dict() serialises this name and
     * load_state_dict() refuses (by default) to load a checkpoint whose
     * saved name differs from the destination's — preventing a silent
     * wrong-LR trajectory when the user rebuilds the scheduler with a
     * different lambda. Pass @p name="" (default) to opt out of the
     * guard; pass @p force=true to load_state_dict() to bypass it.
     */
    LambdaLR(Optimizer& optimizer, LrLambda lr_lambda, std::string name = {});

    auto step() -> void override;
    auto get_last_lr() const -> double override { return last_lr_; }
    auto get_epoch() const -> int { return epoch_; }

    /** @brief Identifier of the lambda this scheduler was built with. */
    auto name() const -> const std::string& { return name_; }

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state) -> void override;
    /**
     * @brief Audit-4 W.12: load_state_dict with explicit force flag.
     *
     * When @p force is true the saved lambda-name guard is skipped — use
     * only when the caller has verified out-of-band that the rebuilt
     * lambda is identical to the one that produced the checkpoint.
     */
    auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state, bool force) -> void;

private:
    Optimizer& optimizer_;
    LrLambda lr_lambda_;
    double base_lr_;
    double last_lr_;
    int epoch_{0};
    std::string name_;
};

/**
 * @brief Multi-step learning rate scheduler
 *
 * Decays the learning rate by gamma at each milestone epoch:
 *
 * @code
 * auto scheduler = MultiStepLR(optimizer, {30, 60, 90}, 0.1);
 * // LR = base_lr         for epochs 0-29
 * // LR = base_lr * 0.1   for epochs 30-59
 * // LR = base_lr * 0.01  for epochs 60-89
 * // LR = base_lr * 0.001 for epochs 90+
 * @endcode
 */
class MultiStepLR : public LRScheduler {
public:
    MultiStepLR(Optimizer& optimizer, std::vector<int> milestones, double gamma = 0.1);

    auto step() -> void override;
    auto get_last_lr() const -> double override { return last_lr_; }
    auto get_epoch() const -> int { return epoch_; }

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    Optimizer& optimizer_;
    std::vector<int> milestones_;
    double gamma_;
    double base_lr_;
    double last_lr_;
    int epoch_{0};
};

/**
 * @brief Polynomial learning rate scheduler
 *
 * Decays the learning rate using a polynomial schedule:
 *
 * \f[
 * \eta_t = (\eta_0 - \eta_{end}) \cdot \left(1 - \frac{t}{T}\right)^{p} + \eta_{end}
 * \f]
 *
 * where \f$p\f$ is the polynomial power (default: 1.0 for linear decay).
 *
 * @code
 * auto scheduler = PolynomialLR(optimizer, 100, 1e-7, 2.0);
 * // Quadratic decay from base_lr to 1e-7 over 100 epochs
 * @endcode
 */
class PolynomialLR : public LRScheduler {
public:
    PolynomialLR(Optimizer& optimizer, int total_iters, double end_lr = 0.0,
                 double power = 1.0);

    auto step() -> void override;
    auto get_last_lr() const -> double override { return last_lr_; }
    auto get_epoch() const -> int { return epoch_; }

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    Optimizer& optimizer_;
    int total_iters_;
    double end_lr_;
    double power_;
    double base_lr_;
    double last_lr_;
    int epoch_{0};
};

/**
 * @brief Constant learning rate for a fixed number of epochs
 *
 * Multiplies base_lr by a constant factor for the first total_iters epochs,
 * then restores base_lr.
 */
class ConstantLR : public LRScheduler {
public:
    ConstantLR(Optimizer& optimizer, double factor = 1.0 / 3.0, int total_iters = 5);

    auto step() -> void override;
    auto get_last_lr() const -> double override { return last_lr_; }

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    Optimizer& optimizer_;
    double factor_;
    int total_iters_;
    double base_lr_;
    double last_lr_;
    int epoch_{0};
};

/**
 * @brief Linear learning rate decay
 *
 * Linearly interpolates the learning rate multiplier from start_factor to
 * end_factor over total_iters epochs.
 */
class LinearLR : public LRScheduler {
public:
    LinearLR(Optimizer& optimizer, double start_factor = 1.0 / 3.0,
             double end_factor = 1.0, int total_iters = 5);

    auto step() -> void override;
    auto get_last_lr() const -> double override { return last_lr_; }

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    Optimizer& optimizer_;
    double start_factor_;
    double end_factor_;
    int total_iters_;
    double base_lr_;
    double last_lr_;
    int epoch_{0};
};

/**
 * @brief Multiplicative learning rate scheduler
 *
 * Each epoch, the learning rate is multiplied by a user-provided function:
 * lr_epoch = lr_epoch-1 * lr_lambda(epoch)
 */
class MultiplicativeLR : public LRScheduler {
public:
    using LambdaFunc = std::function<double(int)>;

    /**
     * @brief Construct MultiplicativeLR.
     *
     * Audit-4 W.12: optional @p name tag — see LambdaLR for the full
     * round-trip contract. Pass @p force=true to load_state_dict() to
     * bypass the name guard.
     */
    MultiplicativeLR(Optimizer& optimizer, LambdaFunc lr_lambda,
                     std::string name = {});

    auto step() -> void override;
    auto get_last_lr() const -> double override { return last_lr_; }

    /** @brief Identifier of the lambda this scheduler was built with. */
    auto name() const -> const std::string& { return name_; }

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state) -> void override;
    auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state, bool force) -> void;

private:
    Optimizer& optimizer_;
    LambdaFunc lr_lambda_;
    double last_lr_;
    int epoch_{0};
    std::string name_;
};

/**
 * @brief Sequential composition of schedulers
 *
 * Applies a sequence of schedulers, each active for a span defined
 * by milestones. milestones[i] is the epoch at which scheduler i+1 starts.
 */
class SequentialLR : public LRScheduler {
public:
    SequentialLR(Optimizer& optimizer,
                 std::vector<std::shared_ptr<LRScheduler>> schedulers,
                 std::vector<int> milestones);

    auto step() -> void override;
    auto get_last_lr() const -> double override { return last_lr_; }

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    Optimizer& optimizer_;
    std::vector<std::shared_ptr<LRScheduler>> schedulers_;
    std::vector<int> milestones_;
    double last_lr_;
    int epoch_{0};
};

/**
 * @brief Chain multiple schedulers together (multiplicative)
 *
 * Applies all schedulers simultaneously. Each scheduler's step() is called
 * in sequence every epoch.
 *
 * @par State serialisation (audit-4 W.10)
 *
 * ChainedScheduler holds no counters of its own — there is no `epoch_` or
 * `step_count_` member. Each wrapped child scheduler maintains its own
 * counter and is restored independently by ChainedScheduler::state_dict()
 * via per-child prefixed entries (`"childN_.<key>"`). This delegation
 * contract is intentional and covered by the
 * `ChainedScheduler_StateDict_RoundTrip` test in
 * `tests/core/test_new_features.cpp` — do not add a parent counter without
 * also revisiting that test, since the children would then double-step.
 */
class ChainedScheduler : public LRScheduler {
public:
    explicit ChainedScheduler(std::vector<std::shared_ptr<LRScheduler>> schedulers);

    auto step() -> void override;
    auto get_last_lr() const -> double override;

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(
        const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    std::vector<std::shared_ptr<LRScheduler>> schedulers_;
};

} // namespace optim
} // namespace tenzor
