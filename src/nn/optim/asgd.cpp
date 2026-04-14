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

auto ASGD::step_impl() -> void {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        Tensor& param_tensor = param.tensor();
        const Tensor& grad_tensor = *param.grad();

        // Use dtype-appropriate scalar tensors to preserve Float64 precision
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, param_tensor.dtype(), param_tensor.device());
        };

        auto grad = grad_tensor.clone();

        // Weight decay (L2 regularization applied to gradient)
        if (weight_decay_ > 0.0) {
            grad = grad + param_tensor * scalar(weight_decay_);
        }

        // Compute step-dependent learning rate: eta_t = lr / (1 + lambd * lr * t)^alpha
        double eta = lr_ / std::pow(1.0 + lambd_ * lr_ * static_cast<double>(step_count_), alpha_);

        // SGD update with decayed learning rate
        param_tensor = param_tensor - grad * scalar(eta);

        // Update running average after t0 steps
        // mu_t = max(1, t - t0)
        // ax_t = (1 - 1/mu_t) * ax_{t-1} + (1/mu_t) * x_t
        double mu = std::max(1.0, static_cast<double>(step_count_) - t0_);
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
