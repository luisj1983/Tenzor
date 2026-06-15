#include "tenzor/nn/optim/scheduler.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/nn/optim/rmsprop.hpp"
#include "tenzor/nn/optim/adagrad.hpp"
#include "tenzor/nn/optim/adadelta.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numbers>
#include <stdexcept>

namespace tenzor::optim {

//==============================================================================
// LRScheduler base state_dict / load_state_dict (audit Q.11)
//==============================================================================
//
// Helpers used by every scheduler override to build scalar Tensors that
// match the optimiser state_dict convention (1-D, length-1, CPU).
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
    // Move to CPU if a checkpoint round-tripped through GPU storage.
    Tensor host = (t.device().type != Device::Type::CPU) ? t.cpu() : t;
    return host.data<int64_t>()[0];
}

inline double read_scalar_f64(const Tensor& t) {
    Tensor host = (t.device().type != Device::Type::CPU) ? t.cpu() : t;
    return host.data<double>()[0];
}

// Audit-4 W.12: encode the lambda-identifier as a UInt8 byte tensor so
// it travels through the existing Tensor-only state_dict checkpoint
// pipeline. Empty names are represented as a length-0 UInt8 tensor.
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

// Audit-5 Z.10: scheduler_type guard helpers shared across subclasses.
auto LRScheduler::make_scheduler_type_tensor(const std::string& name) -> Tensor {
    return make_name_tensor(name);
}

auto LRScheduler::check_scheduler_type(
    const std::unordered_map<std::string, Tensor>& state,
    const std::string& expected,
    bool force) -> void {
    if (force) return;
    auto it = state.find("scheduler_type");
    if (it == state.end()) return;  // older checkpoint — allow silent load
    const std::string saved = read_name_tensor(it->second);
    if (saved != expected) {
        throw std::runtime_error(
            "LRScheduler::load_state_dict: scheduler type mismatch — "
            "saved '" + saved + "' but destination is '" + expected +
            "'. Pass force=true to override.");
    }
}

auto LRScheduler::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;
    state["last_lr"] = make_scalar_f64(get_last_lr());
    return state;
}

auto LRScheduler::load_state_dict(
    const std::unordered_map<std::string, Tensor>& /*state*/) -> void {
    // Base class has no settable state beyond what derived classes track.
    // Derived classes call this base method first, then unpack their
    // own counters.
}

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
    // PyTorch's CosineAnnealingLR implements the SGDR cosine schedule via the
    // closed form below WITHOUT clamping last_epoch: past T_max the cosine
    // keeps oscillating (period 2*T_max), rebounding toward base_lr. A clamp
    // here would incorrectly pin the LR at eta_min after T_max.
    double cosine_term =
        std::cos(std::numbers::pi * static_cast<double>(epoch_) / T_max_);
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
        // Audit II.7: when transitioning out of warmup, the base
        // scheduler's internal epoch_ (or step) counter is still at 0
        // even though `warmup_steps_` global steps have already
        // elapsed. Naive delegation feeds the base its first epoch
        // here, so e.g. a CosineAnnealingLR with T_max = total_steps
        // overshoots by warmup_steps_. Fix: on the first post-warmup
        // call, eagerly advance the base by warmup_steps_ ticks so its
        // counter aligns with the global step number BEFORE applying
        // its first lr.
        if (step_count_ == warmup_steps_ + 1) {
            for (int64_t i = 0; i < warmup_steps_; ++i) {
                base_scheduler_->step();
            }
        }
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

MultiplicativeLR::MultiplicativeLR(Optimizer& optimizer, LambdaFunc lr_lambda,
                                   std::string name)
    : optimizer_(optimizer), lr_lambda_(std::move(lr_lambda)),
      name_(std::move(name)) {
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
    // NN.17: each child scheduler captures its own `base_lr_` at construction
    // and writes optimizer.set_lr(base_lr * factor) every step.  Naively
    // looping just lets the last child's write win, throwing away every
    // earlier child's contribution — the chain is supposed to be the product
    // of every child's factor.  Fix: snapshot the optimizer LR once before
    // any child runs, then for each child compute its own factor relative
    // to that snapshot, multiply, and write the cumulative product at the end.
    //
    // We need access to the shared optimizer.  Every concrete LRScheduler
    // overrides `optimizer()` to expose its `optimizer_` pointer; we pick
    // the first child that returns non-null.  (ChainedScheduler itself
    // returns nullptr.)
    Optimizer* opt = nullptr;
    for (const auto& s : schedulers_) {
        if (s) {
            opt = s->optimizer();
            if (opt) break;
        }
    }

    if (!opt) {
        // No child exposes an optimizer (e.g. all children are nested
        // ChainedSchedulers with stale APIs).  Fall back to the original
        // last-wins behaviour rather than silently dropping the step.
        for (auto& scheduler : schedulers_) {
            if (scheduler) scheduler->step();
        }
        // Best-effort: report the last child's LR since no shared optimizer
        // LR is observable here.
        if (!schedulers_.empty() && schedulers_.back()) {
            last_lr_ = schedulers_.back()->get_last_lr();
        }
        return;
    }

    const double initial_lr = opt->get_lr();
    if (initial_lr == 0.0) {
        // Degenerate: no factor can be recovered relative to zero.  Just
        // run the children in order and let whoever wins, win.
        for (auto& scheduler : schedulers_) {
            if (scheduler) scheduler->step();
        }
        last_lr_ = opt->get_lr();
        return;
    }

    double cumulative_factor = 1.0;
    for (auto& scheduler : schedulers_) {
        if (!scheduler) continue;
        // Restore the snapshot before each child runs so every child sees
        // the SAME baseline and reports its own factor (rather than seeing
        // a previously-multiplied LR and reporting the wrong factor).
        opt->set_lr(initial_lr);
        scheduler->step();
        const double child_lr = opt->get_lr();
        const double factor   = child_lr / initial_lr;
        cumulative_factor    *= factor;
    }

    const double final_lr = initial_lr * cumulative_factor;
    opt->set_lr(final_lr);
    // Cache the LR actually written so get_last_lr() agrees with the optimizer
    // (the product of all child factors), not just the last child's factor.
    last_lr_ = final_lr;
}

auto ChainedScheduler::get_last_lr() const -> double {
    return last_lr_;
}

//==============================================================================
// MultiStepLR + LambdaLR are implemented in scheduler_advanced.cpp.
//==============================================================================

//==============================================================================
// state_dict / load_state_dict overrides (audit Q.11)
//
// Each derived class chains to LRScheduler::state_dict() (which captures
// last_lr) and overlays its own per-step counters.  load_state_dict
// restores those counters; base learning-rate hyperparameters
// (gamma_, T_max_, eta_min_, …) are taken from the constructor and are
// expected to match the saved schedule.
//==============================================================================

auto StepLR::state_dict() const -> std::unordered_map<std::string, Tensor> {
    auto state = LRScheduler::state_dict();
    state["scheduler_type"] = make_scheduler_type_tensor("StepLR");
    state["epoch"]    = make_scalar_i64(static_cast<int64_t>(epoch_));
    state["base_lr"]  = make_scalar_f64(base_lr_);
    return state;
}

auto StepLR::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state) -> void {
    check_scheduler_type(state, "StepLR", /*force=*/false);
    LRScheduler::load_state_dict(state);
    if (auto it = state.find("epoch");    it != state.end()) epoch_   = static_cast<int>(read_scalar_i64(it->second));
    if (auto it = state.find("base_lr");  it != state.end()) base_lr_ = read_scalar_f64(it->second);
    if (auto it = state.find("last_lr");  it != state.end()) {
        last_lr_ = read_scalar_f64(it->second);
        optimizer_->set_lr(last_lr_);
    }
}

auto ExponentialLR::state_dict() const -> std::unordered_map<std::string, Tensor> {
    auto state = LRScheduler::state_dict();
    state["scheduler_type"] = make_scheduler_type_tensor("ExponentialLR");
    state["epoch"]   = make_scalar_i64(static_cast<int64_t>(epoch_));
    state["base_lr"] = make_scalar_f64(base_lr_);
    return state;
}

auto ExponentialLR::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state) -> void {
    check_scheduler_type(state, "ExponentialLR", /*force=*/false);
    LRScheduler::load_state_dict(state);
    if (auto it = state.find("epoch");    it != state.end()) epoch_   = static_cast<int>(read_scalar_i64(it->second));
    if (auto it = state.find("base_lr");  it != state.end()) base_lr_ = read_scalar_f64(it->second);
    if (auto it = state.find("last_lr");  it != state.end()) {
        last_lr_ = read_scalar_f64(it->second);
        optimizer_->set_lr(last_lr_);
    }
}

auto CosineAnnealingLR::state_dict() const -> std::unordered_map<std::string, Tensor> {
    auto state = LRScheduler::state_dict();
    state["scheduler_type"] = make_scheduler_type_tensor("CosineAnnealingLR");
    state["epoch"]   = make_scalar_i64(static_cast<int64_t>(epoch_));
    state["base_lr"] = make_scalar_f64(base_lr_);
    state["eta_min"] = make_scalar_f64(eta_min_);
    return state;
}

auto CosineAnnealingLR::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state) -> void {
    check_scheduler_type(state, "CosineAnnealingLR", /*force=*/false);
    LRScheduler::load_state_dict(state);
    if (auto it = state.find("epoch");    it != state.end()) epoch_   = static_cast<int>(read_scalar_i64(it->second));
    if (auto it = state.find("base_lr");  it != state.end()) base_lr_ = read_scalar_f64(it->second);
    if (auto it = state.find("eta_min");  it != state.end()) eta_min_ = read_scalar_f64(it->second);
    if (auto it = state.find("last_lr");  it != state.end()) {
        last_lr_ = read_scalar_f64(it->second);
        optimizer_->set_lr(last_lr_);
    }
}

auto LinearWarmupScheduler::state_dict() const -> std::unordered_map<std::string, Tensor> {
    auto state = LRScheduler::state_dict();
    state["scheduler_type"] = make_scheduler_type_tensor("LinearWarmupScheduler");
    state["step_count"] = make_scalar_i64(step_count_);
    state["base_lr"]    = make_scalar_f64(base_lr_);
    // Nested base-scheduler state lives under a "base_." prefix.  Use
    // PyTorch's convention of a single namespaced map; downstream callers
    // can filter by the prefix to recover the child state.
    if (base_scheduler_) {
        auto child = base_scheduler_->state_dict();
        for (auto& [k, v] : child) {
            state["base_." + k] = std::move(v);
        }
    }
    return state;
}

auto LinearWarmupScheduler::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state) -> void {
    check_scheduler_type(state, "LinearWarmupScheduler", /*force=*/false);
    LRScheduler::load_state_dict(state);
    if (auto it = state.find("step_count"); it != state.end()) step_count_ = read_scalar_i64(it->second);
    if (auto it = state.find("base_lr");    it != state.end()) base_lr_    = read_scalar_f64(it->second);
    if (auto it = state.find("last_lr");    it != state.end()) {
        last_lr_ = read_scalar_f64(it->second);
        optimizer_.set_lr(last_lr_);
    }
    if (base_scheduler_) {
        std::unordered_map<std::string, Tensor> child;
        const std::string prefix = "base_.";
        for (const auto& [k, v] : state) {
            if (k.rfind(prefix, 0) == 0) {
                child[k.substr(prefix.size())] = v;
            }
        }
        if (!child.empty()) {
            base_scheduler_->load_state_dict(child);
        }
    }
}

auto ConstantLR::state_dict() const -> std::unordered_map<std::string, Tensor> {
    auto state = LRScheduler::state_dict();
    state["scheduler_type"] = make_scheduler_type_tensor("ConstantLR");
    state["epoch"]   = make_scalar_i64(static_cast<int64_t>(epoch_));
    state["base_lr"] = make_scalar_f64(base_lr_);
    return state;
}

auto ConstantLR::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state) -> void {
    check_scheduler_type(state, "ConstantLR", /*force=*/false);
    LRScheduler::load_state_dict(state);
    if (auto it = state.find("epoch");   it != state.end()) epoch_   = static_cast<int>(read_scalar_i64(it->second));
    if (auto it = state.find("base_lr"); it != state.end()) base_lr_ = read_scalar_f64(it->second);
    if (auto it = state.find("last_lr"); it != state.end()) {
        last_lr_ = read_scalar_f64(it->second);
        optimizer_.set_lr(last_lr_);
    }
}

auto LinearLR::state_dict() const -> std::unordered_map<std::string, Tensor> {
    auto state = LRScheduler::state_dict();
    state["scheduler_type"] = make_scheduler_type_tensor("LinearLR");
    state["epoch"]   = make_scalar_i64(static_cast<int64_t>(epoch_));
    state["base_lr"] = make_scalar_f64(base_lr_);
    return state;
}

auto LinearLR::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state) -> void {
    check_scheduler_type(state, "LinearLR", /*force=*/false);
    LRScheduler::load_state_dict(state);
    if (auto it = state.find("epoch");   it != state.end()) epoch_   = static_cast<int>(read_scalar_i64(it->second));
    if (auto it = state.find("base_lr"); it != state.end()) base_lr_ = read_scalar_f64(it->second);
    if (auto it = state.find("last_lr"); it != state.end()) {
        last_lr_ = read_scalar_f64(it->second);
        optimizer_.set_lr(last_lr_);
    }
}

auto MultiplicativeLR::state_dict() const -> std::unordered_map<std::string, Tensor> {
    auto state = LRScheduler::state_dict();
    state["scheduler_type"] = make_scheduler_type_tensor("MultiplicativeLR");
    state["epoch"] = make_scalar_i64(static_cast<int64_t>(epoch_));
    // Audit-4 W.12: persist the lambda identifier so load_state_dict can
    // detect mismatched destinations.
    state["lambda_name"] = make_name_tensor(name_);
    return state;
}

auto MultiplicativeLR::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state) -> void {
    load_state_dict(state, /*force=*/false);
}

auto MultiplicativeLR::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state, bool force) -> void {
    check_scheduler_type(state, "MultiplicativeLR", force);
    LRScheduler::load_state_dict(state);
    // Audit-4 W.12: enforce lambda-identifier match before mutating
    // counters, so a wrong-lambda destination fails loudly instead of
    // proceeding with a silent wrong-LR trajectory.
    if (auto it = state.find("lambda_name"); it != state.end() && !force) {
        const std::string saved = read_name_tensor(it->second);
        if (saved != name_) {
            throw std::runtime_error(
                "MultiplicativeLR::load_state_dict: lambda name mismatch — "
                "saved '" + saved + "' but destination is '" + name_ +
                "'. Pass force=true to override.");
        }
    }
    if (auto it = state.find("epoch");   it != state.end()) epoch_   = static_cast<int>(read_scalar_i64(it->second));
    if (auto it = state.find("last_lr"); it != state.end()) {
        last_lr_ = read_scalar_f64(it->second);
        optimizer_.set_lr(last_lr_);
    }
}

auto SequentialLR::state_dict() const -> std::unordered_map<std::string, Tensor> {
    auto state = LRScheduler::state_dict();
    state["scheduler_type"] = make_scheduler_type_tensor("SequentialLR");
    state["epoch"] = make_scalar_i64(static_cast<int64_t>(epoch_));
    for (size_t i = 0; i < schedulers_.size(); ++i) {
        if (!schedulers_[i]) continue;
        auto child = schedulers_[i]->state_dict();
        std::string prefix = "child" + std::to_string(i) + "_.";
        for (auto& [k, v] : child) {
            state[prefix + k] = std::move(v);
        }
    }
    return state;
}

auto SequentialLR::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state) -> void {
    check_scheduler_type(state, "SequentialLR", /*force=*/false);
    LRScheduler::load_state_dict(state);
    if (auto it = state.find("epoch");   it != state.end()) epoch_   = static_cast<int>(read_scalar_i64(it->second));
    if (auto it = state.find("last_lr"); it != state.end()) {
        last_lr_ = read_scalar_f64(it->second);
        optimizer_.set_lr(last_lr_);
    }
    for (size_t i = 0; i < schedulers_.size(); ++i) {
        if (!schedulers_[i]) continue;
        std::unordered_map<std::string, Tensor> child;
        std::string prefix = "child" + std::to_string(i) + "_.";
        for (const auto& [k, v] : state) {
            if (k.rfind(prefix, 0) == 0) {
                child[k.substr(prefix.size())] = v;
            }
        }
        // BB.14: the outer SequentialLR also writes a top-level
        // "scheduler_type" = "SequentialLR" entry. After stripping the
        // "child{i}_." prefix, that key has no prefix and would be
        // forwarded verbatim into the child's load_state_dict, where
        // check_scheduler_type throws because the child expects its own
        // identifier (e.g. "StepLR"). The prefix-match loop above already
        // excludes it, but children may carry their own properly-prefixed
        // "child{i}_.scheduler_type" entry — strip any stray bare key
        // defensively so the child only sees its own prefixed identifier.
        child.erase("scheduler_type");
        if (!child.empty()) {
            schedulers_[i]->load_state_dict(child);
        }
    }
}

// Audit-4 W.10: ChainedScheduler has no parent epoch_ / step_count_ of its
// own; every child scheduler maintains its own counter (StepLR::epoch_,
// CosineAnnealingLR::epoch_, CyclicLR::step_count_, …) and ChainedScheduler
// merely invokes each child's step() in order. The state_dict therefore
// intentionally delegates fully to the children: the per-child prefixed
// state captures every counter needed to restore the chain exactly, and
// adding our own epoch_/step_count_ would be redundant (and would
// double-count if both we and the children stepped). This contract is
// covered by the ChainedScheduler_StateDict_RoundTrip test.
auto ChainedScheduler::state_dict() const -> std::unordered_map<std::string, Tensor> {
    auto state = LRScheduler::state_dict();
    state["scheduler_type"] = make_scheduler_type_tensor("ChainedScheduler");
    for (size_t i = 0; i < schedulers_.size(); ++i) {
        if (!schedulers_[i]) continue;
        auto child = schedulers_[i]->state_dict();
        std::string prefix = "child" + std::to_string(i) + "_.";
        for (auto& [k, v] : child) {
            state[prefix + k] = std::move(v);
        }
    }
    return state;
}

auto ChainedScheduler::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state) -> void {
    check_scheduler_type(state, "ChainedScheduler", /*force=*/false);
    LRScheduler::load_state_dict(state);
    for (size_t i = 0; i < schedulers_.size(); ++i) {
        if (!schedulers_[i]) continue;
        std::unordered_map<std::string, Tensor> child;
        std::string prefix = "child" + std::to_string(i) + "_.";
        for (const auto& [k, v] : state) {
            if (k.rfind(prefix, 0) == 0) {
                child[k.substr(prefix.size())] = v;
            }
        }
        // BB.14: mirror SequentialLR — strip any unprefixed "scheduler_type"
        // entry so children only see their own prefixed identifier. Without
        // this, a nested ChainedScheduler-of-ChainedScheduler whose entries
        // collide on the bare key would fail check_scheduler_type.
        child.erase("scheduler_type");
        if (!child.empty()) {
            schedulers_[i]->load_state_dict(child);
        }
    }
}

} // namespace tenzor::optim
