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
    : step_size_(step_size), gamma_(gamma), epoch_(0) {
    optimizer_.generic = &optimizer;
    optimizer_type_ = OptimizerType::Generic;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

StepLR::StepLR(SGD& optimizer, int step_size, double gamma)
    : step_size_(step_size), gamma_(gamma), epoch_(0) {
    optimizer_.sgd = &optimizer;
    optimizer_type_ = OptimizerType::SGD;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

StepLR::StepLR(Adam& optimizer, int step_size, double gamma)
    : step_size_(step_size), gamma_(gamma), epoch_(0) {
    optimizer_.adam = &optimizer;
    optimizer_type_ = OptimizerType::Adam;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

StepLR::StepLR(AdamW& optimizer, int step_size, double gamma)
    : step_size_(step_size), gamma_(gamma), epoch_(0) {
    optimizer_.adamw = &optimizer;
    optimizer_type_ = OptimizerType::AdamW;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

StepLR::StepLR(RMSprop& optimizer, int step_size, double gamma)
    : step_size_(step_size), gamma_(gamma), epoch_(0) {
    optimizer_.rmsprop = &optimizer;
    optimizer_type_ = OptimizerType::RMSprop;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

StepLR::StepLR(Adagrad& optimizer, int step_size, double gamma)
    : step_size_(step_size), gamma_(gamma), epoch_(0) {
    optimizer_.adagrad = &optimizer;
    optimizer_type_ = OptimizerType::Adagrad;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

StepLR::StepLR(Adadelta& optimizer, int step_size, double gamma)
    : step_size_(step_size), gamma_(gamma), epoch_(0) {
    optimizer_.adadelta = &optimizer;
    optimizer_type_ = OptimizerType::Adadelta;
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
    set_optimizer_lr(new_lr);
}

auto StepLR::get_current_lr() const -> double {
    switch (optimizer_type_) {
        case OptimizerType::Generic:
            return optimizer_.generic->get_lr();
        case OptimizerType::SGD:
            return optimizer_.sgd->get_lr();
        case OptimizerType::Adam:
            return optimizer_.adam->get_lr();
        case OptimizerType::AdamW:
            return optimizer_.adamw->get_lr();
        case OptimizerType::RMSprop:
            return optimizer_.rmsprop->get_lr();
        case OptimizerType::Adagrad:
            return optimizer_.adagrad->get_lr();
        case OptimizerType::Adadelta:
            return optimizer_.adadelta->get_lr();
    }
    return 0.0;
}

auto StepLR::set_optimizer_lr(double lr) -> void {
    switch (optimizer_type_) {
        case OptimizerType::Generic:
            optimizer_.generic->set_lr(lr);
            break;
        case OptimizerType::SGD:
            optimizer_.sgd->set_lr(lr);
            break;
        case OptimizerType::Adam:
            optimizer_.adam->set_lr(lr);
            break;
        case OptimizerType::AdamW:
            optimizer_.adamw->set_lr(lr);
            break;
        case OptimizerType::RMSprop:
            optimizer_.rmsprop->set_lr(lr);
            break;
        case OptimizerType::Adagrad:
            optimizer_.adagrad->set_lr(lr);
            break;
        case OptimizerType::Adadelta:
            optimizer_.adadelta->set_lr(lr);
            break;
    }
}

//==============================================================================
// ExponentialLR Implementation
//==============================================================================

ExponentialLR::ExponentialLR(Optimizer& optimizer, double gamma)
    : gamma_(gamma), epoch_(0) {
    optimizer_.generic = &optimizer;
    optimizer_type_ = OptimizerType::Generic;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

ExponentialLR::ExponentialLR(SGD& optimizer, double gamma)
    : gamma_(gamma), epoch_(0) {
    optimizer_.sgd = &optimizer;
    optimizer_type_ = OptimizerType::SGD;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

ExponentialLR::ExponentialLR(Adam& optimizer, double gamma)
    : gamma_(gamma), epoch_(0) {
    optimizer_.adam = &optimizer;
    optimizer_type_ = OptimizerType::Adam;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

ExponentialLR::ExponentialLR(AdamW& optimizer, double gamma)
    : gamma_(gamma), epoch_(0) {
    optimizer_.adamw = &optimizer;
    optimizer_type_ = OptimizerType::AdamW;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

ExponentialLR::ExponentialLR(RMSprop& optimizer, double gamma)
    : gamma_(gamma), epoch_(0) {
    optimizer_.rmsprop = &optimizer;
    optimizer_type_ = OptimizerType::RMSprop;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

ExponentialLR::ExponentialLR(Adagrad& optimizer, double gamma)
    : gamma_(gamma), epoch_(0) {
    optimizer_.adagrad = &optimizer;
    optimizer_type_ = OptimizerType::Adagrad;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

ExponentialLR::ExponentialLR(Adadelta& optimizer, double gamma)
    : gamma_(gamma), epoch_(0) {
    optimizer_.adadelta = &optimizer;
    optimizer_type_ = OptimizerType::Adadelta;
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
    set_optimizer_lr(new_lr);
}

auto ExponentialLR::get_current_lr() const -> double {
    switch (optimizer_type_) {
        case OptimizerType::Generic:
            return optimizer_.generic->get_lr();
        case OptimizerType::SGD:
            return optimizer_.sgd->get_lr();
        case OptimizerType::Adam:
            return optimizer_.adam->get_lr();
        case OptimizerType::AdamW:
            return optimizer_.adamw->get_lr();
        case OptimizerType::RMSprop:
            return optimizer_.rmsprop->get_lr();
        case OptimizerType::Adagrad:
            return optimizer_.adagrad->get_lr();
        case OptimizerType::Adadelta:
            return optimizer_.adadelta->get_lr();
    }
    return 0.0;
}

auto ExponentialLR::set_optimizer_lr(double lr) -> void {
    switch (optimizer_type_) {
        case OptimizerType::Generic:
            optimizer_.generic->set_lr(lr);
            break;
        case OptimizerType::SGD:
            optimizer_.sgd->set_lr(lr);
            break;
        case OptimizerType::Adam:
            optimizer_.adam->set_lr(lr);
            break;
        case OptimizerType::AdamW:
            optimizer_.adamw->set_lr(lr);
            break;
        case OptimizerType::RMSprop:
            optimizer_.rmsprop->set_lr(lr);
            break;
        case OptimizerType::Adagrad:
            optimizer_.adagrad->set_lr(lr);
            break;
        case OptimizerType::Adadelta:
            optimizer_.adadelta->set_lr(lr);
            break;
    }
}

//==============================================================================
// CosineAnnealingLR Implementation
//==============================================================================

CosineAnnealingLR::CosineAnnealingLR(Optimizer& optimizer, int T_max, double eta_min)
    : T_max_(T_max), eta_min_(eta_min), epoch_(0) {
    optimizer_.generic = &optimizer;
    optimizer_type_ = OptimizerType::Generic;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

CosineAnnealingLR::CosineAnnealingLR(SGD& optimizer, int T_max, double eta_min)
    : T_max_(T_max), eta_min_(eta_min), epoch_(0) {
    optimizer_.sgd = &optimizer;
    optimizer_type_ = OptimizerType::SGD;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

CosineAnnealingLR::CosineAnnealingLR(Adam& optimizer, int T_max, double eta_min)
    : T_max_(T_max), eta_min_(eta_min), epoch_(0) {
    optimizer_.adam = &optimizer;
    optimizer_type_ = OptimizerType::Adam;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

CosineAnnealingLR::CosineAnnealingLR(AdamW& optimizer, int T_max, double eta_min)
    : T_max_(T_max), eta_min_(eta_min), epoch_(0) {
    optimizer_.adamw = &optimizer;
    optimizer_type_ = OptimizerType::AdamW;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

CosineAnnealingLR::CosineAnnealingLR(RMSprop& optimizer, int T_max, double eta_min)
    : T_max_(T_max), eta_min_(eta_min), epoch_(0) {
    optimizer_.rmsprop = &optimizer;
    optimizer_type_ = OptimizerType::RMSprop;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

CosineAnnealingLR::CosineAnnealingLR(Adagrad& optimizer, int T_max, double eta_min)
    : T_max_(T_max), eta_min_(eta_min), epoch_(0) {
    optimizer_.adagrad = &optimizer;
    optimizer_type_ = OptimizerType::Adagrad;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;
}

CosineAnnealingLR::CosineAnnealingLR(Adadelta& optimizer, int T_max, double eta_min)
    : T_max_(T_max), eta_min_(eta_min), epoch_(0) {
    optimizer_.adadelta = &optimizer;
    optimizer_type_ = OptimizerType::Adadelta;
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
    set_optimizer_lr(new_lr);
}

auto CosineAnnealingLR::get_current_lr() const -> double {
    switch (optimizer_type_) {
        case OptimizerType::Generic:
            return optimizer_.generic->get_lr();
        case OptimizerType::SGD:
            return optimizer_.sgd->get_lr();
        case OptimizerType::Adam:
            return optimizer_.adam->get_lr();
        case OptimizerType::AdamW:
            return optimizer_.adamw->get_lr();
        case OptimizerType::RMSprop:
            return optimizer_.rmsprop->get_lr();
        case OptimizerType::Adagrad:
            return optimizer_.adagrad->get_lr();
        case OptimizerType::Adadelta:
            return optimizer_.adadelta->get_lr();
    }
    return 0.0;
}

auto CosineAnnealingLR::set_optimizer_lr(double lr) -> void {
    switch (optimizer_type_) {
        case OptimizerType::Generic:
            optimizer_.generic->set_lr(lr);
            break;
        case OptimizerType::SGD:
            optimizer_.sgd->set_lr(lr);
            break;
        case OptimizerType::Adam:
            optimizer_.adam->set_lr(lr);
            break;
        case OptimizerType::AdamW:
            optimizer_.adamw->set_lr(lr);
            break;
        case OptimizerType::RMSprop:
            optimizer_.rmsprop->set_lr(lr);
            break;
        case OptimizerType::Adagrad:
            optimizer_.adagrad->set_lr(lr);
            break;
        case OptimizerType::Adadelta:
            optimizer_.adadelta->set_lr(lr);
            break;
    }
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

} // namespace tenzor::optim
