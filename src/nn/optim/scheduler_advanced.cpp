/**
 * @file scheduler_advanced.cpp
 * @brief Implementation of advanced learning rate schedulers
 */

#include "tenzor/nn/optim/scheduler.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <numbers>

namespace tenzor::optim {

//==============================================================================
// ReduceLROnPlateau Implementation
//==============================================================================

ReduceLROnPlateau::ReduceLROnPlateau(
    SGD& optimizer,
    const std::string& mode,
    double factor,
    int64_t patience,
    double threshold,
    const std::string& threshold_mode,
    int64_t cooldown,
    double min_lr,
    double eps)
    : mode_(mode), threshold_mode_(threshold_mode),
      factor_(factor), threshold_(threshold), min_lr_(min_lr), eps_(eps),
      patience_(patience), cooldown_(cooldown),
      num_bad_epochs_(0), cooldown_counter_(0) {

    optimizer_.sgd = &optimizer;
    optimizer_type_ = OptimizerType::SGD;
    last_lr_ = optimizer.get_lr();

    // Initialize best_metric based on mode
    if (mode_ == "min") {
        best_metric_ = std::numeric_limits<double>::infinity();
    } else if (mode_ == "max") {
        best_metric_ = -std::numeric_limits<double>::infinity();
    } else {
        throw std::invalid_argument("ReduceLROnPlateau: mode must be 'min' or 'max'");
    }

    if (threshold_mode_ != "rel" && threshold_mode_ != "abs") {
        throw std::invalid_argument("ReduceLROnPlateau: threshold_mode must be 'rel' or 'abs'");
    }
}

ReduceLROnPlateau::ReduceLROnPlateau(
    Adam& optimizer,
    const std::string& mode,
    double factor,
    int64_t patience,
    double threshold,
    const std::string& threshold_mode,
    int64_t cooldown,
    double min_lr,
    double eps)
    : mode_(mode), threshold_mode_(threshold_mode),
      factor_(factor), threshold_(threshold), min_lr_(min_lr), eps_(eps),
      patience_(patience), cooldown_(cooldown),
      num_bad_epochs_(0), cooldown_counter_(0) {

    optimizer_.adam = &optimizer;
    optimizer_type_ = OptimizerType::Adam;
    last_lr_ = optimizer.get_lr();

    if (mode_ == "min") {
        best_metric_ = std::numeric_limits<double>::infinity();
    } else if (mode_ == "max") {
        best_metric_ = -std::numeric_limits<double>::infinity();
    } else {
        throw std::invalid_argument("ReduceLROnPlateau: mode must be 'min' or 'max'");
    }

    if (threshold_mode_ != "rel" && threshold_mode_ != "abs") {
        throw std::invalid_argument("ReduceLROnPlateau: threshold_mode must be 'rel' or 'abs'");
    }
}

ReduceLROnPlateau::ReduceLROnPlateau(
    AdamW& optimizer,
    const std::string& mode,
    double factor,
    int64_t patience,
    double threshold,
    const std::string& threshold_mode,
    int64_t cooldown,
    double min_lr,
    double eps)
    : mode_(mode), threshold_mode_(threshold_mode),
      factor_(factor), threshold_(threshold), min_lr_(min_lr), eps_(eps),
      patience_(patience), cooldown_(cooldown),
      num_bad_epochs_(0), cooldown_counter_(0) {

    optimizer_.adamw = &optimizer;
    optimizer_type_ = OptimizerType::AdamW;
    last_lr_ = optimizer.get_lr();

    if (mode_ == "min") {
        best_metric_ = std::numeric_limits<double>::infinity();
    } else if (mode_ == "max") {
        best_metric_ = -std::numeric_limits<double>::infinity();
    } else {
        throw std::invalid_argument("ReduceLROnPlateau: mode must be 'min' or 'max'");
    }

    if (threshold_mode_ != "rel" && threshold_mode_ != "abs") {
        throw std::invalid_argument("ReduceLROnPlateau: threshold_mode must be 'rel' or 'abs'");
    }
}

ReduceLROnPlateau::ReduceLROnPlateau(
    Optimizer& optimizer,
    const std::string& mode,
    double factor,
    int64_t patience,
    double threshold,
    const std::string& threshold_mode,
    int64_t cooldown,
    double min_lr,
    double eps)
    : mode_(mode), threshold_mode_(threshold_mode),
      factor_(factor), threshold_(threshold), min_lr_(min_lr), eps_(eps),
      patience_(patience), cooldown_(cooldown),
      num_bad_epochs_(0), cooldown_counter_(0) {

    optimizer_.generic = &optimizer;
    optimizer_type_ = OptimizerType::Generic;
    last_lr_ = optimizer.get_lr();

    if (mode_ == "min") {
        best_metric_ = std::numeric_limits<double>::infinity();
    } else if (mode_ == "max") {
        best_metric_ = -std::numeric_limits<double>::infinity();
    } else {
        throw std::invalid_argument("ReduceLROnPlateau: mode must be 'min' or 'max'");
    }

    if (threshold_mode_ != "rel" && threshold_mode_ != "abs") {
        throw std::invalid_argument("ReduceLROnPlateau: threshold_mode must be 'rel' or 'abs'");
    }
}

auto ReduceLROnPlateau::is_better(double current, double best) const -> bool {
    if (mode_ == "min") {
        if (threshold_mode_ == "rel") {
            return current < best * (1.0 - threshold_);
        } else {  // abs
            return current < best - threshold_;
        }
    } else {  // max
        if (threshold_mode_ == "rel") {
            return current > best * (1.0 + threshold_);
        } else {  // abs
            return current > best + threshold_;
        }
    }
}

auto ReduceLROnPlateau::step(double metric) -> void {
    // Check if in cooldown period - decrement AFTER using it
    bool in_cooldown = cooldown_counter_ > 0;

    if (in_cooldown) {
        cooldown_counter_--;
        // Update best metric even during cooldown
        if (is_better(metric, best_metric_)) {
            best_metric_ = metric;
        }
        return;
    }

    // Check if metric improved
    if (is_better(metric, best_metric_)) {
        best_metric_ = metric;
        num_bad_epochs_ = 0;
    } else {
        num_bad_epochs_++;
    }

    // Reduce LR after patience epochs of no improvement
    // Matches PyTorch: patience=2 means reduce when bad_epochs reaches 2
    if (num_bad_epochs_ >= patience_) {
        reduce_lr();
        // Set cooldown counter for next cooldown_ epochs
        cooldown_counter_ = cooldown_;
        num_bad_epochs_ = 0;
    }
}

auto ReduceLROnPlateau::reduce_lr() -> void {
    double current_lr = get_current_lr();
    double new_lr = std::max(current_lr * factor_, min_lr_);

    // Only update if change is significant
    if (current_lr - new_lr > eps_) {
        last_lr_ = new_lr;
        set_optimizer_lr(new_lr);
    }
}

auto ReduceLROnPlateau::get_current_lr() const -> double {
    switch (optimizer_type_) {
        case OptimizerType::SGD:
            return optimizer_.sgd->get_lr();
        case OptimizerType::Adam:
            return optimizer_.adam->get_lr();
        case OptimizerType::AdamW:
            return optimizer_.adamw->get_lr();
        case OptimizerType::Generic:
            return optimizer_.generic->get_lr();
    }
    return 0.0;
}

auto ReduceLROnPlateau::set_optimizer_lr(double lr) -> void {
    switch (optimizer_type_) {
        case OptimizerType::SGD:
            optimizer_.sgd->set_lr(lr);
            break;
        case OptimizerType::Adam:
            optimizer_.adam->set_lr(lr);
            break;
        case OptimizerType::AdamW:
            optimizer_.adamw->set_lr(lr);
            break;
        case OptimizerType::Generic:
            optimizer_.generic->set_lr(lr);
            break;
    }
}

//==============================================================================
// CyclicLR Implementation
//==============================================================================

CyclicLR::CyclicLR(
    SGD& optimizer,
    double base_lr, double max_lr,
    int64_t step_size_up,
    int64_t step_size_down,
    const std::string& mode,
    double gamma,
    double scale_fn,
    const std::string& scale_mode)
    : base_lr_(base_lr), max_lr_(max_lr),
      step_size_up_(step_size_up),
      step_size_down_(step_size_down == -1 ? step_size_up : step_size_down),
      mode_(mode), gamma_(gamma), scale_fn_(scale_fn), scale_mode_(scale_mode),
      step_count_(0) {

    optimizer_.sgd = &optimizer;
    optimizer_type_ = OptimizerType::SGD;
    cycle_size_ = step_size_up_ + step_size_down_;
    last_lr_ = base_lr_;
    optimizer.set_lr(base_lr_);

    if (step_size_up_ <= 0) {
        throw std::invalid_argument("CyclicLR: step_size_up must be positive");
    }
    if (step_size_down_ < 0) {
        throw std::invalid_argument("CyclicLR: step_size_down must be non-negative");
    }
    if (cycle_size_ <= 0) {
        throw std::invalid_argument("CyclicLR: cycle size (step_size_up + step_size_down) must be positive");
    }

    if (mode_ != "triangular" && mode_ != "triangular2" && mode_ != "exp_range") {
        throw std::invalid_argument("CyclicLR: mode must be 'triangular', 'triangular2', or 'exp_range'");
    }
}

CyclicLR::CyclicLR(
    Adam& optimizer,
    double base_lr, double max_lr,
    int64_t step_size_up,
    int64_t step_size_down,
    const std::string& mode,
    double gamma,
    double scale_fn,
    const std::string& scale_mode)
    : base_lr_(base_lr), max_lr_(max_lr),
      step_size_up_(step_size_up),
      step_size_down_(step_size_down == -1 ? step_size_up : step_size_down),
      mode_(mode), gamma_(gamma), scale_fn_(scale_fn), scale_mode_(scale_mode),
      step_count_(0) {

    optimizer_.adam = &optimizer;
    optimizer_type_ = OptimizerType::Adam;
    cycle_size_ = step_size_up_ + step_size_down_;
    last_lr_ = base_lr_;
    optimizer.set_lr(base_lr_);

    if (step_size_up_ <= 0) {
        throw std::invalid_argument("CyclicLR: step_size_up must be positive");
    }
    if (step_size_down_ < 0) {
        throw std::invalid_argument("CyclicLR: step_size_down must be non-negative");
    }
    if (cycle_size_ <= 0) {
        throw std::invalid_argument("CyclicLR: cycle size (step_size_up + step_size_down) must be positive");
    }

    if (mode_ != "triangular" && mode_ != "triangular2" && mode_ != "exp_range") {
        throw std::invalid_argument("CyclicLR: mode must be 'triangular', 'triangular2', or 'exp_range'");
    }
}

CyclicLR::CyclicLR(
    AdamW& optimizer,
    double base_lr, double max_lr,
    int64_t step_size_up,
    int64_t step_size_down,
    const std::string& mode,
    double gamma,
    double scale_fn,
    const std::string& scale_mode)
    : base_lr_(base_lr), max_lr_(max_lr),
      step_size_up_(step_size_up),
      step_size_down_(step_size_down == -1 ? step_size_up : step_size_down),
      mode_(mode), gamma_(gamma), scale_fn_(scale_fn), scale_mode_(scale_mode),
      step_count_(0) {

    optimizer_.adamw = &optimizer;
    optimizer_type_ = OptimizerType::AdamW;
    cycle_size_ = step_size_up_ + step_size_down_;
    last_lr_ = base_lr_;
    optimizer.set_lr(base_lr_);

    if (step_size_up_ <= 0) {
        throw std::invalid_argument("CyclicLR: step_size_up must be positive");
    }
    if (step_size_down_ < 0) {
        throw std::invalid_argument("CyclicLR: step_size_down must be non-negative");
    }
    if (cycle_size_ <= 0) {
        throw std::invalid_argument("CyclicLR: cycle size (step_size_up + step_size_down) must be positive");
    }

    if (mode_ != "triangular" && mode_ != "triangular2" && mode_ != "exp_range") {
        throw std::invalid_argument("CyclicLR: mode must be 'triangular', 'triangular2', or 'exp_range'");
    }
}

CyclicLR::CyclicLR(
    Optimizer& optimizer,
    double base_lr, double max_lr,
    int64_t step_size_up,
    int64_t step_size_down,
    const std::string& mode,
    double gamma,
    double scale_fn,
    const std::string& scale_mode)
    : base_lr_(base_lr), max_lr_(max_lr),
      step_size_up_(step_size_up),
      step_size_down_(step_size_down == -1 ? step_size_up : step_size_down),
      mode_(mode), gamma_(gamma), scale_fn_(scale_fn), scale_mode_(scale_mode),
      step_count_(0) {

    optimizer_.generic = &optimizer;
    optimizer_type_ = OptimizerType::Generic;
    cycle_size_ = step_size_up_ + step_size_down_;
    last_lr_ = base_lr_;
    optimizer.set_lr(base_lr_);

    if (step_size_up_ <= 0) {
        throw std::invalid_argument("CyclicLR: step_size_up must be positive");
    }
    if (step_size_down_ < 0) {
        throw std::invalid_argument("CyclicLR: step_size_down must be non-negative");
    }
    if (cycle_size_ <= 0) {
        throw std::invalid_argument("CyclicLR: cycle size (step_size_up + step_size_down) must be positive");
    }

    if (mode_ != "triangular" && mode_ != "triangular2" && mode_ != "exp_range") {
        throw std::invalid_argument("CyclicLR: mode must be 'triangular', 'triangular2', or 'exp_range'");
    }
}

auto CyclicLR::get_scale_factor(int64_t cycle) const -> double {
    if (mode_ == "triangular") {
        return 1.0;
    } else if (mode_ == "triangular2") {
        return 1.0 / std::pow(2.0, cycle);
    } else {  // exp_range
        // For exp_range mode, ALWAYS use iterations-based scaling
        // This ensures exponential decay over time regardless of scale_mode setting
        return std::pow(gamma_, step_count_);
    }
}

auto CyclicLR::compute_lr() -> double {
    int64_t cycle = step_count_ / cycle_size_;
    int64_t x = step_count_ % cycle_size_;

    double scale = get_scale_factor(cycle);
    double pct;

    if (x < step_size_up_) {
        // Increasing phase
        pct = static_cast<double>(x) / static_cast<double>(step_size_up_);
    } else {
        // Decreasing phase
        pct = 1.0 - static_cast<double>(x - step_size_up_) / static_cast<double>(step_size_down_);
    }

    // Compute base LR at current position in cycle
    double lr = base_lr_ + (max_lr_ - base_lr_) * pct;

    // Apply scaling based on mode
    if (mode_ == "triangular") {
        // No scaling (scale is always 1.0)
        return lr;
    } else if (mode_ == "triangular2" || mode_ == "exp_range") {
        // Scale the amplitude (distance from base_lr)
        return base_lr_ + (lr - base_lr_) * scale;
    }

    return lr;
}

auto CyclicLR::step() -> void {
    step_count_++;
    double new_lr = compute_lr();
    last_lr_ = new_lr;
    set_optimizer_lr(new_lr);
}

auto CyclicLR::get_current_lr() const -> double {
    switch (optimizer_type_) {
        case OptimizerType::SGD:
            return optimizer_.sgd->get_lr();
        case OptimizerType::Adam:
            return optimizer_.adam->get_lr();
        case OptimizerType::AdamW:
            return optimizer_.adamw->get_lr();
        case OptimizerType::Generic:
            return optimizer_.generic->get_lr();
    }
    return 0.0;
}

auto CyclicLR::set_optimizer_lr(double lr) -> void {
    switch (optimizer_type_) {
        case OptimizerType::SGD:
            optimizer_.sgd->set_lr(lr);
            break;
        case OptimizerType::Adam:
            optimizer_.adam->set_lr(lr);
            break;
        case OptimizerType::AdamW:
            optimizer_.adamw->set_lr(lr);
            break;
        case OptimizerType::Generic:
            optimizer_.generic->set_lr(lr);
            break;
    }
}

//==============================================================================
// OneCycleLR Implementation
//==============================================================================

OneCycleLR::OneCycleLR(
    SGD& optimizer,
    double max_lr,
    int64_t total_steps,
    int64_t epochs,
    int64_t steps_per_epoch,
    double pct_start,
    const std::string& anneal_strategy,
    double div_factor,
    double final_div_factor)
    : max_lr_(max_lr), pct_start_(pct_start),
      div_factor_(div_factor), final_div_factor_(final_div_factor),
      anneal_strategy_(anneal_strategy), step_count_(-1) {
    // LL.6: step_count_ starts at -1 so the first step() advances it to 0
    // before computing — PyTorch convention.

    optimizer_.sgd = &optimizer;
    optimizer_type_ = OptimizerType::SGD;

    // Compute total steps
    if (epochs != -1 && steps_per_epoch != -1) {
        total_steps_ = epochs * steps_per_epoch;
    } else {
        total_steps_ = total_steps;
    }

    if (total_steps_ <= 0) {
        throw std::invalid_argument("OneCycleLR: total_steps must be positive");
    }

    if (anneal_strategy_ != "cos" && anneal_strategy_ != "linear") {
        throw std::invalid_argument("OneCycleLR: anneal_strategy must be 'cos' or 'linear'");
    }

    // LL.6: Don't pre-set the optimizer LR here. The first step() call will
    // increment step_count_ to 0 and compute the same initial LR.
    last_lr_ = max_lr_ / div_factor_;
}

OneCycleLR::OneCycleLR(
    Adam& optimizer,
    double max_lr,
    int64_t total_steps,
    int64_t epochs,
    int64_t steps_per_epoch,
    double pct_start,
    const std::string& anneal_strategy,
    double div_factor,
    double final_div_factor)
    : max_lr_(max_lr), pct_start_(pct_start),
      div_factor_(div_factor), final_div_factor_(final_div_factor),
      anneal_strategy_(anneal_strategy), step_count_(-1) {
    // LL.6: step_count_ starts at -1 so the first step() advances to 0.

    optimizer_.adam = &optimizer;
    optimizer_type_ = OptimizerType::Adam;

    if (epochs != -1 && steps_per_epoch != -1) {
        total_steps_ = epochs * steps_per_epoch;
    } else {
        total_steps_ = total_steps;
    }

    if (total_steps_ <= 0) {
        throw std::invalid_argument("OneCycleLR: total_steps must be positive");
    }

    if (anneal_strategy_ != "cos" && anneal_strategy_ != "linear") {
        throw std::invalid_argument("OneCycleLR: anneal_strategy must be 'cos' or 'linear'");
    }

    // LL.6: don't pre-set optimizer LR; first step() yields same value.
    last_lr_ = max_lr_ / div_factor_;
}

OneCycleLR::OneCycleLR(
    AdamW& optimizer,
    double max_lr,
    int64_t total_steps,
    int64_t epochs,
    int64_t steps_per_epoch,
    double pct_start,
    const std::string& anneal_strategy,
    double div_factor,
    double final_div_factor)
    : max_lr_(max_lr), pct_start_(pct_start),
      div_factor_(div_factor), final_div_factor_(final_div_factor),
      anneal_strategy_(anneal_strategy), step_count_(-1) {
    // LL.6: step_count_ starts at -1 so the first step() advances to 0.

    optimizer_.adamw = &optimizer;
    optimizer_type_ = OptimizerType::AdamW;

    if (epochs != -1 && steps_per_epoch != -1) {
        total_steps_ = epochs * steps_per_epoch;
    } else {
        total_steps_ = total_steps;
    }

    if (total_steps_ <= 0) {
        throw std::invalid_argument("OneCycleLR: total_steps must be positive");
    }

    if (anneal_strategy_ != "cos" && anneal_strategy_ != "linear") {
        throw std::invalid_argument("OneCycleLR: anneal_strategy must be 'cos' or 'linear'");
    }

    // LL.6: don't pre-set optimizer LR; first step() yields same value.
    last_lr_ = max_lr_ / div_factor_;
}

OneCycleLR::OneCycleLR(
    Optimizer& optimizer,
    double max_lr,
    int64_t total_steps,
    int64_t epochs,
    int64_t steps_per_epoch,
    double pct_start,
    const std::string& anneal_strategy,
    double div_factor,
    double final_div_factor)
    : max_lr_(max_lr), pct_start_(pct_start),
      div_factor_(div_factor), final_div_factor_(final_div_factor),
      anneal_strategy_(anneal_strategy), step_count_(-1) {
    // LL.6: step_count_ starts at -1 so the first step() advances to 0.

    optimizer_.generic = &optimizer;
    optimizer_type_ = OptimizerType::Generic;

    if (epochs != -1 && steps_per_epoch != -1) {
        total_steps_ = epochs * steps_per_epoch;
    } else {
        total_steps_ = total_steps;
    }

    if (total_steps_ <= 0) {
        throw std::invalid_argument("OneCycleLR: total_steps must be positive");
    }

    if (anneal_strategy_ != "cos" && anneal_strategy_ != "linear") {
        throw std::invalid_argument("OneCycleLR: anneal_strategy must be 'cos' or 'linear'");
    }

    // LL.6: don't pre-set optimizer LR; first step() yields same value.
    last_lr_ = max_lr_ / div_factor_;
}

auto OneCycleLR::anneal_func(double start, double end, double pct) -> double {
    if (anneal_strategy_ == "cos") {
        double cos_out = std::cos(std::numbers::pi * pct) + 1.0;
        return end + (start - end) / 2.0 * cos_out;
    } else {  // linear
        return (end - start) * pct + start;
    }
}

auto OneCycleLR::compute_lr() -> double {
    if (step_count_ >= total_steps_) {
        return max_lr_ / final_div_factor_;
    }

    // Round (not truncate) the warmup length, and ensure a configured warmup
    // fraction never collapses to zero steps. Plain truncation drops the entire
    // warmup phase whenever pct_start * total_steps < 1 (small total_steps),
    // starting training by annealing from max_lr and also risking a divide by
    // zero in the warmup pct below.
    int64_t warmup_steps =
        static_cast<int64_t>(std::llround(pct_start_ * static_cast<double>(total_steps_)));
    if (pct_start_ > 0.0 && warmup_steps < 1) {
        warmup_steps = 1;
    }

    if (step_count_ < warmup_steps) {
        // Phase 1: Warmup
        double pct = static_cast<double>(step_count_) / static_cast<double>(warmup_steps);
        double start_lr = max_lr_ / div_factor_;
        return anneal_func(start_lr, max_lr_, pct);
    } else {
        // Phase 2: Annealing
        int64_t anneal_steps = total_steps_ - warmup_steps;
        double pct = static_cast<double>(step_count_ - warmup_steps) / static_cast<double>(anneal_steps);
        double end_lr = max_lr_ / final_div_factor_;
        return anneal_func(max_lr_, end_lr, pct);
    }
}

auto OneCycleLR::step() -> void {
    // LL.6: increment FIRST, then compute (PyTorch convention).
    step_count_++;
    double new_lr = compute_lr();
    last_lr_ = new_lr;
    set_optimizer_lr(new_lr);
}

auto OneCycleLR::get_current_lr() const -> double {
    switch (optimizer_type_) {
        case OptimizerType::SGD:
            return optimizer_.sgd->get_lr();
        case OptimizerType::Adam:
            return optimizer_.adam->get_lr();
        case OptimizerType::AdamW:
            return optimizer_.adamw->get_lr();
        case OptimizerType::Generic:
            return optimizer_.generic->get_lr();
    }
    return 0.0;
}

auto OneCycleLR::set_optimizer_lr(double lr) -> void {
    switch (optimizer_type_) {
        case OptimizerType::SGD:
            optimizer_.sgd->set_lr(lr);
            break;
        case OptimizerType::Adam:
            optimizer_.adam->set_lr(lr);
            break;
        case OptimizerType::AdamW:
            optimizer_.adamw->set_lr(lr);
            break;
        case OptimizerType::Generic:
            optimizer_.generic->set_lr(lr);
            break;
    }
}

//==============================================================================
// CosineAnnealingWarmRestarts Implementation
//==============================================================================

CosineAnnealingWarmRestarts::CosineAnnealingWarmRestarts(
    SGD& optimizer,
    int64_t T_0,
    int64_t T_mult,
    double eta_min)
    : T_0_(T_0), T_mult_(T_mult), eta_min_(eta_min),
      T_i_(T_0), T_cur_(0), step_count_(0) {

    optimizer_.sgd = &optimizer;
    optimizer_type_ = OptimizerType::SGD;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;

    if (T_0_ <= 0) {
        throw std::invalid_argument("CosineAnnealingWarmRestarts: T_0 must be positive");
    }
    if (T_mult_ < 1) {
        throw std::invalid_argument("CosineAnnealingWarmRestarts: T_mult must be >= 1");
    }
}

CosineAnnealingWarmRestarts::CosineAnnealingWarmRestarts(
    Adam& optimizer,
    int64_t T_0,
    int64_t T_mult,
    double eta_min)
    : T_0_(T_0), T_mult_(T_mult), eta_min_(eta_min),
      T_i_(T_0), T_cur_(0), step_count_(0) {

    optimizer_.adam = &optimizer;
    optimizer_type_ = OptimizerType::Adam;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;

    if (T_0_ <= 0) {
        throw std::invalid_argument("CosineAnnealingWarmRestarts: T_0 must be positive");
    }
    if (T_mult_ < 1) {
        throw std::invalid_argument("CosineAnnealingWarmRestarts: T_mult must be >= 1");
    }
}

CosineAnnealingWarmRestarts::CosineAnnealingWarmRestarts(
    AdamW& optimizer,
    int64_t T_0,
    int64_t T_mult,
    double eta_min)
    : T_0_(T_0), T_mult_(T_mult), eta_min_(eta_min),
      T_i_(T_0), T_cur_(0), step_count_(0) {

    optimizer_.adamw = &optimizer;
    optimizer_type_ = OptimizerType::AdamW;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;

    if (T_0_ <= 0) {
        throw std::invalid_argument("CosineAnnealingWarmRestarts: T_0 must be positive");
    }
    if (T_mult_ < 1) {
        throw std::invalid_argument("CosineAnnealingWarmRestarts: T_mult must be >= 1");
    }
}

CosineAnnealingWarmRestarts::CosineAnnealingWarmRestarts(
    Optimizer& optimizer,
    int64_t T_0,
    int64_t T_mult,
    double eta_min)
    : T_0_(T_0), T_mult_(T_mult), eta_min_(eta_min),
      T_i_(T_0), T_cur_(0), step_count_(0) {

    optimizer_.generic = &optimizer;
    optimizer_type_ = OptimizerType::Generic;
    base_lr_ = optimizer.get_lr();
    last_lr_ = base_lr_;

    if (T_0_ <= 0) {
        throw std::invalid_argument("CosineAnnealingWarmRestarts: T_0 must be positive");
    }
    if (T_mult_ < 1) {
        throw std::invalid_argument("CosineAnnealingWarmRestarts: T_mult must be >= 1");
    }
}

auto CosineAnnealingWarmRestarts::step() -> void {
    // LL.6: increment FIRST (both step_count_ and T_cur_), THEN compute LR
    // and handle restart roll-over (PyTorch convention).
    step_count_++;
    T_cur_++;

    // Check if we've completed a restart period and need to reset before
    // computing the new LR for the upcoming step.
    if (T_cur_ >= T_i_) {
        T_cur_ = 0;          // Reset to start of new period
        T_i_ *= T_mult_;     // Increase period length
    }

    // Update learning rate for the CURRENT (post-increment) T_cur position.
    update_lr();
}

auto CosineAnnealingWarmRestarts::update_lr() -> void {
    // PyTorch SGDR formula: eta_t = eta_min + (base_lr - eta_min) *
    //                                (1 + cos(pi * T_cur / T_i)) / 2.
    // The denominator is the full period T_i (NOT T_i - 1). Using T_i - 1 makes
    // eta_min reached one step early each restart and shifts every intermediate
    // LR away from the reference schedule. T_i_ is guaranteed >= 1 by the ctor.
    int64_t period = std::max(T_i_, int64_t(1));  // Avoid division by zero
    double cos_term = std::cos(std::numbers::pi * static_cast<double>(T_cur_) / static_cast<double>(period));
    double new_lr = eta_min_ + (base_lr_ - eta_min_) * (1.0 + cos_term) / 2.0;

    last_lr_ = new_lr;
    set_optimizer_lr(new_lr);
}

auto CosineAnnealingWarmRestarts::get_current_lr() const -> double {
    switch (optimizer_type_) {
        case OptimizerType::SGD:
            return optimizer_.sgd->get_lr();
        case OptimizerType::Adam:
            return optimizer_.adam->get_lr();
        case OptimizerType::AdamW:
            return optimizer_.adamw->get_lr();
        case OptimizerType::Generic:
            return optimizer_.generic->get_lr();
    }
    return 0.0;
}

auto CosineAnnealingWarmRestarts::set_optimizer_lr(double lr) -> void {
    switch (optimizer_type_) {
        case OptimizerType::SGD:
            optimizer_.sgd->set_lr(lr);
            break;
        case OptimizerType::Adam:
            optimizer_.adam->set_lr(lr);
            break;
        case OptimizerType::AdamW:
            optimizer_.adamw->set_lr(lr);
            break;
        case OptimizerType::Generic:
            optimizer_.generic->set_lr(lr);
            break;
    }
}

//==============================================================================
// LambdaLR Implementation
//==============================================================================

LambdaLR::LambdaLR(Optimizer& optimizer, LrLambda lr_lambda, std::string name)
    : optimizer_(optimizer)
    , lr_lambda_(std::move(lr_lambda))
    , epoch_(0)
    , name_(std::move(name)) {
    if (!lr_lambda_) {
        throw std::invalid_argument("LambdaLR: lr_lambda must not be null");
    }
    base_lr_ = optimizer_.get_lr();
    last_lr_ = base_lr_ * lr_lambda_(0);
    optimizer_.set_lr(last_lr_);
}

auto LambdaLR::step() -> void {
    epoch_++;
    last_lr_ = base_lr_ * lr_lambda_(epoch_);
    optimizer_.set_lr(last_lr_);
}

//==============================================================================
// MultiStepLR Implementation
//==============================================================================

MultiStepLR::MultiStepLR(Optimizer& optimizer, std::vector<int> milestones, double gamma)
    : optimizer_(optimizer)
    , milestones_(std::move(milestones))
    , gamma_(gamma)
    , epoch_(0) {
    // Sort milestones for binary search
    std::sort(milestones_.begin(), milestones_.end());
    base_lr_ = optimizer_.get_lr();
    last_lr_ = base_lr_;
}

auto MultiStepLR::step() -> void {
    epoch_++;
    // Count how many milestones have been passed
    int num_decays = 0;
    for (int milestone : milestones_) {
        if (epoch_ >= milestone) {
            num_decays++;
        } else {
            break;  // milestones are sorted
        }
    }
    last_lr_ = base_lr_ * std::pow(gamma_, num_decays);
    optimizer_.set_lr(last_lr_);
}

//==============================================================================
// PolynomialLR Implementation
//==============================================================================

PolynomialLR::PolynomialLR(Optimizer& optimizer, int total_iters, double end_lr,
                           double power)
    : optimizer_(optimizer)
    , total_iters_(total_iters)
    , end_lr_(end_lr)
    , power_(power)
    , epoch_(0) {
    if (total_iters_ <= 0) {
        throw std::invalid_argument("PolynomialLR: total_iters must be positive, got " +
                                    std::to_string(total_iters_));
    }
    base_lr_ = optimizer_.get_lr();
    last_lr_ = base_lr_;
}

auto PolynomialLR::step() -> void {
    epoch_++;
    if (epoch_ >= total_iters_) {
        last_lr_ = end_lr_;
    } else {
        double progress = 1.0 - static_cast<double>(epoch_) / static_cast<double>(total_iters_);
        last_lr_ = (base_lr_ - end_lr_) * std::pow(progress, power_) + end_lr_;
    }
    optimizer_.set_lr(last_lr_);
}

//==============================================================================
// state_dict / load_state_dict overrides for advanced schedulers (audit Q.11)
//==============================================================================

namespace {

inline Tensor make_scalar_i64(int64_t v) {
    Tensor t({1}, DType::Int64, Device::cpu());
    t.data<int64_t>()[0] = v;
    return t;
}

inline Tensor make_scalar_f64(double v) {
    Tensor t({1}, DType::Float64, Device::cpu());
    t.data<double>()[0] = v;
    return t;
}

inline int64_t read_scalar_i64(const Tensor& t) {
    Tensor host = (t.device().type != Device::Type::CPU) ? t.cpu() : t;
    return host.data<int64_t>()[0];
}

inline double read_scalar_f64(const Tensor& t) {
    Tensor host = (t.device().type != Device::Type::CPU) ? t.cpu() : t;
    return host.data<double>()[0];
}

// Audit-4 W.12: see src/nn/optim/scheduler.cpp for the matching helper
// used by MultiplicativeLR.
inline Tensor make_name_tensor(const std::string& name) {
    Tensor t({static_cast<int64_t>(name.size())}, DType::UInt8, Device::cpu());
    if (!name.empty()) {
        std::memcpy(t.data<uint8_t>(), name.data(), name.size());
    }
    return t;
}

inline std::string read_name_tensor(const Tensor& t) {
    Tensor host = (t.device().type != Device::Type::CPU) ? t.cpu() : t;
    const int64_t n = host.numel();
    if (n <= 0) return {};
    const auto* bytes = reinterpret_cast<const char*>(host.data<uint8_t>());
    return std::string(bytes, static_cast<size_t>(n));
}

}  // namespace

auto ReduceLROnPlateau::state_dict() const -> std::unordered_map<std::string, Tensor> {
    auto state = LRScheduler::state_dict();
    state["scheduler_type"]   = make_scheduler_type_tensor("ReduceLROnPlateau");
    state["best_metric"]      = make_scalar_f64(best_metric_);
    state["num_bad_epochs"]   = make_scalar_i64(num_bad_epochs_);
    state["cooldown_counter"] = make_scalar_i64(cooldown_counter_);
    return state;
}

auto ReduceLROnPlateau::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state) -> void {
    check_scheduler_type(state, "ReduceLROnPlateau", /*force=*/false);
    LRScheduler::load_state_dict(state);
    if (auto it = state.find("best_metric");      it != state.end()) best_metric_      = read_scalar_f64(it->second);
    if (auto it = state.find("num_bad_epochs");   it != state.end()) num_bad_epochs_   = read_scalar_i64(it->second);
    if (auto it = state.find("cooldown_counter"); it != state.end()) cooldown_counter_ = read_scalar_i64(it->second);
    if (auto it = state.find("last_lr");          it != state.end()) {
        last_lr_ = read_scalar_f64(it->second);
        set_optimizer_lr(last_lr_);
    }
}

auto CyclicLR::state_dict() const -> std::unordered_map<std::string, Tensor> {
    auto state = LRScheduler::state_dict();
    state["scheduler_type"] = make_scheduler_type_tensor("CyclicLR");
    state["step_count"] = make_scalar_i64(step_count_);
    state["base_lr"]    = make_scalar_f64(base_lr_);
    state["max_lr"]     = make_scalar_f64(max_lr_);
    return state;
}

auto CyclicLR::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state) -> void {
    check_scheduler_type(state, "CyclicLR", /*force=*/false);
    LRScheduler::load_state_dict(state);
    if (auto it = state.find("step_count"); it != state.end()) step_count_ = read_scalar_i64(it->second);
    if (auto it = state.find("base_lr");    it != state.end()) base_lr_    = read_scalar_f64(it->second);
    if (auto it = state.find("max_lr");     it != state.end()) max_lr_     = read_scalar_f64(it->second);
    if (auto it = state.find("last_lr");    it != state.end()) {
        last_lr_ = read_scalar_f64(it->second);
        set_optimizer_lr(last_lr_);
    }
}

auto OneCycleLR::state_dict() const -> std::unordered_map<std::string, Tensor> {
    auto state = LRScheduler::state_dict();
    state["scheduler_type"] = make_scheduler_type_tensor("OneCycleLR");
    state["step_count"]  = make_scalar_i64(step_count_);
    state["max_lr"]      = make_scalar_f64(max_lr_);
    state["total_steps"] = make_scalar_i64(total_steps_);
    return state;
}

auto OneCycleLR::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state) -> void {
    check_scheduler_type(state, "OneCycleLR", /*force=*/false);
    LRScheduler::load_state_dict(state);
    if (auto it = state.find("step_count");  it != state.end()) step_count_  = read_scalar_i64(it->second);
    if (auto it = state.find("max_lr");      it != state.end()) max_lr_      = read_scalar_f64(it->second);
    if (auto it = state.find("total_steps"); it != state.end()) total_steps_ = read_scalar_i64(it->second);
    if (auto it = state.find("last_lr");     it != state.end()) {
        last_lr_ = read_scalar_f64(it->second);
        set_optimizer_lr(last_lr_);
    }
}

auto CosineAnnealingWarmRestarts::state_dict() const -> std::unordered_map<std::string, Tensor> {
    auto state = LRScheduler::state_dict();
    state["scheduler_type"] = make_scheduler_type_tensor("CosineAnnealingWarmRestarts");
    state["step_count"] = make_scalar_i64(step_count_);
    state["T_cur"]      = make_scalar_i64(T_cur_);
    state["T_i"]        = make_scalar_i64(T_i_);
    state["base_lr"]    = make_scalar_f64(base_lr_);
    state["eta_min"]    = make_scalar_f64(eta_min_);
    return state;
}

auto CosineAnnealingWarmRestarts::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state) -> void {
    check_scheduler_type(state, "CosineAnnealingWarmRestarts", /*force=*/false);
    LRScheduler::load_state_dict(state);
    if (auto it = state.find("step_count"); it != state.end()) step_count_ = read_scalar_i64(it->second);
    if (auto it = state.find("T_cur");      it != state.end()) T_cur_      = read_scalar_i64(it->second);
    if (auto it = state.find("T_i");        it != state.end()) T_i_        = read_scalar_i64(it->second);
    if (auto it = state.find("base_lr");    it != state.end()) base_lr_    = read_scalar_f64(it->second);
    if (auto it = state.find("eta_min");    it != state.end()) eta_min_    = read_scalar_f64(it->second);
    if (auto it = state.find("last_lr");    it != state.end()) {
        last_lr_ = read_scalar_f64(it->second);
        set_optimizer_lr(last_lr_);
    }
}

auto LambdaLR::state_dict() const -> std::unordered_map<std::string, Tensor> {
    auto state = LRScheduler::state_dict();
    state["scheduler_type"] = make_scheduler_type_tensor("LambdaLR");
    state["epoch"]   = make_scalar_i64(static_cast<int64_t>(epoch_));
    state["base_lr"] = make_scalar_f64(base_lr_);
    // Audit-4 W.12: persist the lambda identifier so a mismatched
    // destination is detected at load time rather than silently
    // producing a wrong-LR trajectory.
    state["lambda_name"] = make_name_tensor(name_);
    return state;
}

auto LambdaLR::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state) -> void {
    load_state_dict(state, /*force=*/false);
}

auto LambdaLR::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state, bool force) -> void {
    check_scheduler_type(state, "LambdaLR", force);
    LRScheduler::load_state_dict(state);
    // Audit-4 W.12: verify the saved lambda identifier matches the
    // destination's before touching counters. Throws unless force=true.
    if (auto it = state.find("lambda_name"); it != state.end() && !force) {
        const std::string saved = read_name_tensor(it->second);
        if (saved != name_) {
            throw std::runtime_error(
                "LambdaLR::load_state_dict: lambda name mismatch — saved '" +
                saved + "' but destination is '" + name_ +
                "'. Pass force=true to override.");
        }
    }
    if (auto it = state.find("epoch");   it != state.end()) epoch_   = static_cast<int>(read_scalar_i64(it->second));
    if (auto it = state.find("base_lr"); it != state.end()) base_lr_ = read_scalar_f64(it->second);
    if (auto it = state.find("last_lr"); it != state.end()) {
        last_lr_ = read_scalar_f64(it->second);
        optimizer_.set_lr(last_lr_);
    }
}

auto MultiStepLR::state_dict() const -> std::unordered_map<std::string, Tensor> {
    auto state = LRScheduler::state_dict();
    state["scheduler_type"] = make_scheduler_type_tensor("MultiStepLR");
    state["epoch"]   = make_scalar_i64(static_cast<int64_t>(epoch_));
    state["base_lr"] = make_scalar_f64(base_lr_);
    return state;
}

auto MultiStepLR::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state) -> void {
    check_scheduler_type(state, "MultiStepLR", /*force=*/false);
    LRScheduler::load_state_dict(state);
    if (auto it = state.find("epoch");   it != state.end()) epoch_   = static_cast<int>(read_scalar_i64(it->second));
    if (auto it = state.find("base_lr"); it != state.end()) base_lr_ = read_scalar_f64(it->second);
    if (auto it = state.find("last_lr"); it != state.end()) {
        last_lr_ = read_scalar_f64(it->second);
        optimizer_.set_lr(last_lr_);
    }
}

auto PolynomialLR::state_dict() const -> std::unordered_map<std::string, Tensor> {
    auto state = LRScheduler::state_dict();
    state["scheduler_type"] = make_scheduler_type_tensor("PolynomialLR");
    state["epoch"]   = make_scalar_i64(static_cast<int64_t>(epoch_));
    state["base_lr"] = make_scalar_f64(base_lr_);
    state["end_lr"]  = make_scalar_f64(end_lr_);
    return state;
}

auto PolynomialLR::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state) -> void {
    check_scheduler_type(state, "PolynomialLR", /*force=*/false);
    LRScheduler::load_state_dict(state);
    if (auto it = state.find("epoch");   it != state.end()) epoch_   = static_cast<int>(read_scalar_i64(it->second));
    if (auto it = state.find("base_lr"); it != state.end()) base_lr_ = read_scalar_f64(it->second);
    if (auto it = state.find("end_lr");  it != state.end()) end_lr_  = read_scalar_f64(it->second);
    if (auto it = state.find("last_lr"); it != state.end()) {
        last_lr_ = read_scalar_f64(it->second);
        optimizer_.set_lr(last_lr_);
    }
}

} // namespace tenzor::optim
