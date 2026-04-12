#include "tenzor/nn/optim/adamax.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cmath>

namespace tenzor::optim {

Adamax::Adamax(std::vector<std::shared_ptr<Variable>> params, double lr, double beta1,
               double beta2, double eps, double weight_decay)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), weight_decay_(weight_decay) {
    initialize_buffers();
}

auto Adamax::step_impl() -> void {
    step_count_++;

    const double bias_correction1 = 1.0 - std::pow(beta1_, static_cast<double>(step_count_));
    const double step_size = lr_ / bias_correction1;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        const Tensor& grad = param.grad().value();

        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, param.tensor().dtype(), param.tensor().device());
        };

        auto grad_copy = grad.clone();

        if (weight_decay_ > 0.0) {
            grad_copy = grad_copy + param.tensor() * scalar(weight_decay_);
        }

        // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        exp_avg_[i] = exp_avg_[i] * scalar(beta1_) +
                      grad_copy    * scalar(1.0 - beta1_);

        // u_t = max(beta2 * u_{t-1}, |g_t|)
        // element-wise via maximum(), not a reduction.
        exp_inf_[i] = maximum(exp_inf_[i] * scalar(beta2_), abs(grad_copy));

        // denom = u_t + eps
        auto denom = exp_inf_[i] + scalar(eps_);

        // theta -= (lr / (1 - beta1^t)) * m_t / denom
        param.tensor() = param.tensor() -
                         div(exp_avg_[i], denom) * scalar(step_size);
    }
}

auto Adamax::initialize_buffers() -> void {
    exp_avg_.clear();
    exp_inf_.clear();

    for (auto& param : parameters_) {
        if (param) {
            exp_avg_.push_back(zeros_like(param->tensor()));
            exp_inf_.push_back(zeros_like(param->tensor()));
        }
    }
}

auto Adamax::set_lr(double lr) -> void { lr_ = lr; }
auto Adamax::get_lr() const -> double { return lr_; }

auto Adamax::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    state["step_count"] = Tensor({1}, DType::Int64, Device::cpu());
    state["step_count"].data<int64_t>()[0] = step_count_;

    auto put_double = [&](const char* key, double value) {
        state[key] = Tensor({1}, DType::Float64, Device::cpu());
        state[key].data<double>()[0] = value;
    };
    put_double("lr", lr_);
    put_double("beta1", beta1_);
    put_double("beta2", beta2_);
    put_double("eps", eps_);
    put_double("weight_decay", weight_decay_);

    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        state["exp_avg_" + std::to_string(i)] = exp_avg_[i].clone();
        state["exp_inf_" + std::to_string(i)] = exp_inf_[i].clone();
    }
    return state;
}

auto Adamax::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    auto load_double = [&](const char* key, double& out) {
        auto it = state.find(key);
        if (it != state.end()) out = it->second.data<double>()[0];
    };

    if (state.count("step_count")) {
        step_count_ = state.at("step_count").data<int64_t>()[0];
    }
    load_double("lr", lr_);
    load_double("beta1", beta1_);
    load_double("beta2", beta2_);
    load_double("eps", eps_);
    load_double("weight_decay", weight_decay_);

    size_t saved_count = 0;
    for (const auto& [key, _] : state) {
        if (key.starts_with("exp_avg_")) {
            ++saved_count;
        }
    }
    if (saved_count > 0 && saved_count != exp_avg_.size()) {
        throw std::runtime_error(
            "Adamax::load_state_dict: momentum buffer count mismatch - "
            "saved " + std::to_string(saved_count) + " but have " +
            std::to_string(exp_avg_.size()) + " parameters");
    }

    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        std::string exp_avg_key = "exp_avg_" + std::to_string(i);
        std::string exp_inf_key = "exp_inf_" + std::to_string(i);
        if (state.count(exp_avg_key)) exp_avg_[i] = state.at(exp_avg_key).clone();
        if (state.count(exp_inf_key)) exp_inf_[i] = state.at(exp_inf_key).clone();
    }
}

} // namespace tenzor::optim
