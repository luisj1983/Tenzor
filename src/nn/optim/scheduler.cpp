#include "tenzor/nn/optim/scheduler.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/nn/optim/rmsprop.hpp"
#include "tenzor/nn/optim/adagrad.hpp"
#include "tenzor/nn/optim/adadelta.hpp"
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace tenzor::optim {

//==============================================================================
// StepLR Implementation
//==============================================================================

StepLR::StepLR(Optimizer& optimizer, int step_size, double gamma)
    : optimizer_(&optimizer), step_size_(step_size), gamma_(gamma), epoch_(0) {
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

auto StepLR::step() -> void {
    epoch_++;
    update_lr();
}

auto StepLR::update_lr() -> void {
    // lr = base_lr * gamma^(epoch / step_size)
    if (step_size_ == 0) {
        throw std::runtime_error("StepLR: step_size cannot be zero");
    }
    int num_decays = epoch_ / step_size_;
    double new_lr = base_lr_ * std::pow(gamma_, num_decays);
    last_lr_ = new_lr;
    optimizer_->set_lr(new_lr);
}

//==============================================================================
// ExponentialLR Implementation
//==============================================================================

ExponentialLR::ExponentialLR(Optimizer& optimizer, double gamma)
    : optimizer_(&optimizer), gamma_(gamma), epoch_(0) {
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

auto ExponentialLR::step() -> void {
    epoch_++;
    update_lr();
}

auto ExponentialLR::update_lr() -> void {
    // lr = base_lr * gamma^epoch
    double new_lr = base_lr_ * std::pow(gamma_, epoch_);
    last_lr_ = new_lr;
    optimizer_->set_lr(new_lr);
}

//==============================================================================
// CosineAnnealingLR Implementation
//==============================================================================

CosineAnnealingLR::CosineAnnealingLR(Optimizer& optimizer, int T_max, double eta_min)
    : optimizer_(&optimizer), T_max_(T_max), eta_min_(eta_min), epoch_(0) {
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

auto CosineAnnealingLR::step() -> void {
    epoch_++;
    update_lr();
}

auto CosineAnnealingLR::update_lr() -> void {
    // lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
    if (T_max_ == 0) {
        throw std::runtime_error("CosineAnnealingLR: T_max cannot be zero");
    }
    double cosine_term = std::cos(std::numbers::pi * epoch_ / T_max_);
    double new_lr = eta_min_ + (base_lr_ - eta_min_) * (1.0 + cosine_term) / 2.0;
    last_lr_ = new_lr;
    optimizer_->set_lr(new_lr);
}

//==============================================================================
// LinearWarmupScheduler Implementation
//==============================================================================

LinearWarmupScheduler::LinearWarmupScheduler(Optimizer& optimizer,
                                           std::shared_ptr<LRScheduler> base_scheduler,
                                           int64_t warmup_steps,
                                           double warmup_start_factor)
    : optimizer_(optimizer),
      base_scheduler_(std::move(base_scheduler)),
      warmup_steps_(warmup_steps),
      warmup_start_factor_(warmup_start_factor),
      step_count_(0) {

    if (warmup_steps_ < 0) {
        throw std::invalid_argument(
            "LinearWarmupScheduler: warmup_steps must be non-negative, got " +
            std::to_string(warmup_steps_));
    }

    if (warmup_start_factor_ < 0.0 || warmup_start_factor_ > 1.0) {
        throw std::invalid_argument(
            "LinearWarmupScheduler: warmup_start_factor must be in [0, 1], got " +
            std::to_string(warmup_start_factor_));
    }

    base_lr_ = optimizer_.get_lr();
    last_lr_ = base_lr_ * warmup_start_factor_;

    // Set initial LR to the warmup starting LR
    if (warmup_steps_ > 0) {
        optimizer_.set_lr(last_lr_);
    }
}

auto LinearWarmupScheduler::step() -> void {
    step_count_++;

    if (step_count_ <= warmup_steps_) {
        // Linear warmup phase:
        // lr = base_lr * (start_factor + (1 - start_factor) * step / warmup_steps)
        double progress = static_cast<double>(step_count_) / static_cast<double>(warmup_steps_);
        double factor = warmup_start_factor_ + (1.0 - warmup_start_factor_) * progress;
        last_lr_ = base_lr_ * factor;
        optimizer_.set_lr(last_lr_);
    } else {
        // After warmup: delegate to base scheduler
        base_scheduler_->step();
        last_lr_ = base_scheduler_->get_last_lr();
    }
}

//==============================================================================
// ConstantLR Implementation
//==============================================================================

ConstantLR::ConstantLR(Optimizer& optimizer, double factor, int total_iters)
    : optimizer_(optimizer), factor_(factor), total_iters_(total_iters) {
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_ * factor_;
    optimizer_.set_lr(last_lr_);
}

auto ConstantLR::step() -> void {
    epoch_++;
    if (epoch_ >= total_iters_) {
        last_lr_ = base_lr_;
    } else {
        last_lr_ = base_lr_ * factor_;
    }
    optimizer_.set_lr(last_lr_);
}

//==============================================================================
// LinearLR Implementation
//==============================================================================

LinearLR::LinearLR(Optimizer& optimizer, double start_factor, double end_factor,
                   int total_iters)
    : optimizer_(optimizer), start_factor_(start_factor), end_factor_(end_factor),
      total_iters_(total_iters) {
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_ * start_factor_;
    optimizer_.set_lr(last_lr_);
}

auto LinearLR::step() -> void {
    epoch_++;
    double factor;
    if (epoch_ >= total_iters_) {
        factor = end_factor_;
    } else {
        double t = static_cast<double>(epoch_) / static_cast<double>(total_iters_);
        factor = start_factor_ + (end_factor_ - start_factor_) * t;
    }
    last_lr_ = base_lr_ * factor;
    optimizer_.set_lr(last_lr_);
}

//==============================================================================
// MultiplicativeLR Implementation
//==============================================================================

MultiplicativeLR::MultiplicativeLR(Optimizer& optimizer, LambdaFunc lr_lambda)
    : optimizer_(optimizer), lr_lambda_(std::move(lr_lambda)) {
    last_lr_ = optimizer.get_lr();
}

auto MultiplicativeLR::step() -> void {
    epoch_++;
    last_lr_ *= lr_lambda_(epoch_);
    optimizer_.set_lr(last_lr_);
}

//==============================================================================
// SequentialLR Implementation
//==============================================================================

SequentialLR::SequentialLR(Optimizer& optimizer,
                           std::vector<std::shared_ptr<LRScheduler>> schedulers,
                           std::vector<int> milestones)
    : optimizer_(optimizer), schedulers_(std::move(schedulers)),
      milestones_(std::move(milestones)) {
    if (schedulers_.empty()) {
        throw std::invalid_argument("SequentialLR: at least one scheduler required");
    }
    if (milestones_.size() != schedulers_.size() - 1) {
        throw std::invalid_argument(
            "SequentialLR: milestones must have len(schedulers) - 1 elements");
    }
    last_lr_ = optimizer.get_lr();
}

auto SequentialLR::step() -> void {
    epoch_++;
    // Find which scheduler is active
    size_t idx = 0;
    for (size_t i = 0; i < milestones_.size(); ++i) {
        if (epoch_ >= milestones_[i]) {
            idx = i + 1;
        }
    }
    schedulers_[idx]->step();
    last_lr_ = schedulers_[idx]->get_last_lr();
}

//==============================================================================
// ChainedScheduler Implementation
//==============================================================================

ChainedScheduler::ChainedScheduler(std::vector<std::shared_ptr<LRScheduler>> schedulers)
    : schedulers_(std::move(schedulers)) {
    if (schedulers_.empty()) {
        throw std::invalid_argument("ChainedScheduler: at least one scheduler required");
    }
}

auto ChainedScheduler::step() -> void {
    for (auto& scheduler : schedulers_) {
        scheduler->step();
    }
}

auto ChainedScheduler::get_last_lr() const -> double {
    return schedulers_.back()->get_last_lr();
}

//==============================================================================
// MultiStepLR + LambdaLR are implemented in scheduler_advanced.cpp.
//==============================================================================

} // namespace tenzor::optim
