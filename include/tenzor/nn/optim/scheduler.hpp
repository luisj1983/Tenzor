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

// Base class for all learning rate schedulers
class LRScheduler {
public:
    virtual ~LRScheduler() = default;

    // Step the scheduler (typically called once per epoch)
    virtual auto step() -> void = 0;

    // Get the last computed learning rate
    virtual auto get_last_lr() const -> double = 0;

    // Get the current learning rate (alias for get_last_lr)
    auto get_lr() const -> double { return get_last_lr(); }

protected:
    LRScheduler() = default;
};

// StepLR: Decays the learning rate by gamma every step_size epochs
// Formula: lr = initial_lr * gamma^(epoch / step_size)
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

// ExponentialLR: Decays the learning rate by gamma every epoch
// Formula: lr = initial_lr * gamma^epoch
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

// CosineAnnealingLR: Cosine annealing schedule
// Formula: lr = eta_min + (initial_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
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
