/**
 * @file scheduler_advanced.cpp
 * @brief Implementation of advanced learning rate schedulers
 */

#include "tenzor/nn/optim/scheduler.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include <cmath>
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
      anneal_strategy_(anneal_strategy), step_count_(0) {

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

    // Set initial LR
    double initial_lr = max_lr_ / div_factor_;
    last_lr_ = initial_lr;
    optimizer.set_lr(initial_lr);
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
      anneal_strategy_(anneal_strategy), step_count_(0) {

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

    double initial_lr = max_lr_ / div_factor_;
    last_lr_ = initial_lr;
    optimizer.set_lr(initial_lr);
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
      anneal_strategy_(anneal_strategy), step_count_(0) {

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

    double initial_lr = max_lr_ / div_factor_;
    last_lr_ = initial_lr;
    optimizer.set_lr(initial_lr);
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

    int64_t warmup_steps = static_cast<int64_t>(pct_start_ * total_steps_);

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
    double new_lr = compute_lr();
    last_lr_ = new_lr;
    set_optimizer_lr(new_lr);
    step_count_++;
}

auto OneCycleLR::get_current_lr() const -> double {
    switch (optimizer_type_) {
        case OptimizerType::SGD:
            return optimizer_.sgd->get_lr();
        case OptimizerType::Adam:
            return optimizer_.adam->get_lr();
        case OptimizerType::AdamW:
            return optimizer_.adamw->get_lr();
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

auto CosineAnnealingWarmRestarts::step() -> void {
    step_count_++;

    // Update learning rate for CURRENT T_cur position
    update_lr();

    // THEN increment T_cur
    T_cur_++;

    // Check if we've completed a restart period and need to reset
    if (T_cur_ >= T_i_) {
        T_cur_ = 0;  // Reset to start of new period
        T_i_ *= T_mult_;  // Increase period length
    }
}

auto CosineAnnealingWarmRestarts::update_lr() -> void {
    // Cosine annealing formula
    // Use T_i - 1 in denominator so that at T_cur = T_i - 1 (last step of cycle),
    // cos(π) = -1 and LR reaches eta_min
    int64_t period = std::max(T_i_ - 1, int64_t(1));  // Avoid division by zero
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
    }
}

} // namespace tenzor::optim
