#include "tenzor/nn/optim/asgd.hpp"
#include "tenzor/ops/creation.hpp"

#include <cmath>
#include <algorithm>

namespace tenzor::optim {

ASGD::ASGD(std::vector<std::shared_ptr<Variable>> params, double lr, double lambd,
            double alpha, double t0, double weight_decay)
    : Optimizer(std::move(params)), lr_(lr), lambd_(lambd),
      alpha_(alpha), t0_(t0), weight_decay_(weight_decay) {
    initialize_buffers();
}

ASGD::ASGD(std::vector<optim::ParamGroup> groups,
           double default_lr, double default_lambd,
           double default_alpha, double default_t0,
           double default_weight_decay)
    : Optimizer(std::move(groups)),
      lr_(default_lr),
      lambd_(default_lambd),
      alpha_(default_alpha),
      t0_(default_t0),
      weight_decay_(default_weight_decay) {
    initialize_buffers();
}

auto ASGD::step_impl() -> void {
    // Audit D.4: per-parameter hyperparameters resolve from the
    // active ParamGroup (when one was set up) or fall through to
    // the optimiser-wide defaults stored on this ASGD instance.
    // lambd and t0 have no ParamGroup-level override fields, so they
    // always use the optimiser-wide defaults.
    struct ASGDHP {
        double lr;
        double lambd;
        double alpha;
        double t0;
        double weight_decay;
    };

    auto resolve = [&](size_t i) -> ASGDHP {
        ASGDHP hp{lr_, lambd_, alpha_, t0_, weight_decay_};
        if (const auto* g = find_group_for_param(i)) {
            hp.lr           = g->lr;
            hp.weight_decay = g->weight_decay;
            hp.alpha        = ParamGroup::or_else(g->alpha, alpha_);
        }
        return hp;
    };

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        const ASGDHP hp = resolve(i);

        Tensor& param_tensor = param.tensor();
        const Tensor& grad_tensor = *param.grad();

        // Use dtype-appropriate scalar tensors to preserve Float64 precision
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, param_tensor.dtype(), param_tensor.device());
        };

        auto grad = grad_tensor.clone();

        // Weight decay (L2 regularization applied to gradient)
        if (hp.weight_decay > 0.0) {
            grad = grad + param_tensor * scalar(hp.weight_decay);
        }

        // Compute step-dependent learning rate: eta_t = lr / (1 + lambd * lr * t)^alpha
        double eta = hp.lr / std::pow(1.0 + hp.lambd * hp.lr * static_cast<double>(step_count_), hp.alpha);

        // SGD update with decayed learning rate
        param_tensor = param_tensor - grad * scalar(eta);

        // Update running average after t0 steps
        // mu_t = max(1, t - t0)
        // ax_t = (1 - 1/mu_t) * ax_{t-1} + (1/mu_t) * x_t
        double mu = std::max(1.0, static_cast<double>(step_count_) - hp.t0);
        double inv_mu = 1.0 / mu;
        ax_buffers_[i] = ax_buffers_[i] * scalar(1.0 - inv_mu) + param_tensor * scalar(inv_mu);
    }

    step_count_++;
}

auto ASGD::set_lr(double lr) -> void {
    lr_ = lr;
}

auto ASGD::get_lr() const -> double {
    return lr_;
}

auto ASGD::initialize_buffers() -> void {
    ax_buffers_.clear();
    for (auto& param : parameters_) {
        if (param) {
            // Initialize averaged parameters as a copy of the initial parameters
            ax_buffers_.push_back(param->tensor().clone());
        }
    }
}

// Audit K.1: extend ax_buffers_ for parameters appended via
// add_param_group.  Matches initialize_buffers — averaged params start
// as a clone of the freshly-added parameter value.
auto ASGD::on_parameters_appended_(size_t old_count, size_t new_count) -> void {
    ax_buffers_.reserve(new_count);
    for (size_t i = old_count; i < new_count; ++i) {
        const auto& param = parameters_[i];
        if (param) {
            ax_buffers_.push_back(param->tensor().clone());
        } else {
            ax_buffers_.push_back(Tensor{});
        }
    }
}

auto ASGD::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    // Save optimizer configuration as tensors
    state["lr"] = Tensor({1}, DType::Float64, Device::cpu());
    state["lr"].data<double>()[0] = lr_;

    state["lambd"] = Tensor({1}, DType::Float64, Device::cpu());
    state["lambd"].data<double>()[0] = lambd_;

    state["alpha"] = Tensor({1}, DType::Float64, Device::cpu());
    state["alpha"].data<double>()[0] = alpha_;

    state["t0"] = Tensor({1}, DType::Float64, Device::cpu());
    state["t0"].data<double>()[0] = t0_;

    state["weight_decay"] = Tensor({1}, DType::Float64, Device::cpu());
    state["weight_decay"].data<double>()[0] = weight_decay_;

    state["step_count"] = Tensor({1}, DType::Int64, Device::cpu());
    state["step_count"].data<int64_t>()[0] = step_count_;

    // Save averaged parameter buffers
    for (size_t i = 0; i < ax_buffers_.size(); ++i) {
        state["ax_" + std::to_string(i)] = ax_buffers_[i].clone();
    }

    return state;
}

auto ASGD::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    // Load optimizer configuration
    if (state.count("lr")) {
        lr_ = state.at("lr").data<double>()[0];
    }

    if (state.count("lambd")) {
        lambd_ = state.at("lambd").data<double>()[0];
    }

    if (state.count("alpha")) {
        alpha_ = state.at("alpha").data<double>()[0];
    }

    if (state.count("t0")) {
        t0_ = state.at("t0").data<double>()[0];
    }

    if (state.count("weight_decay")) {
        weight_decay_ = state.at("weight_decay").data<double>()[0];
    }

    if (state.count("step_count")) {
        step_count_ = state.at("step_count").data<int64_t>()[0];
    }

    // Load averaged parameter buffers
    for (size_t i = 0; i < ax_buffers_.size(); ++i) {
        std::string ax_key = "ax_" + std::to_string(i);
        if (state.count(ax_key)) {
            ax_buffers_[i] = state.at(ax_key).clone();
        }
    }
}

} // namespace tenzor::optim
