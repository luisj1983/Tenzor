#include "tenzor/nn/optim/lion.hpp"

#include "tenzor/core/device.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

namespace tenzor::optim {

Lion::Lion(std::vector<std::shared_ptr<Variable>> params,
           double lr, double beta1, double beta2, double weight_decay)
    : Optimizer(std::move(params)),
      lr_(lr), beta1_(beta1), beta2_(beta2), weight_decay_(weight_decay) {
    initialize_buffers();
}

auto Lion::step_impl() -> void {
    step_count_++;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        const Tensor& grad = param.grad().value();

        // Dtype-matched scalar helper (preserves Float64 precision that
        // static_cast<float> would truncate).
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, param.tensor().dtype(), param.tensor().device());
        };

        // c_t = beta1 * m_{t-1} + (1 - beta1) * g_t   — update direction
        auto c = momentum_[i] * scalar(beta1_) + grad * scalar(1.0 - beta1_);

        // theta_t = theta_{t-1} - lr * (sign(c_t) + wd * theta_{t-1})
        Tensor step = sign(c);
        if (weight_decay_ > 0.0) {
            step = step + param.tensor() * scalar(weight_decay_);
        }
        param.tensor() = param.tensor() - step * scalar(lr_);

        // m_t = beta2 * m_{t-1} + (1 - beta2) * g_t   — momentum state
        momentum_[i] = momentum_[i] * scalar(beta2_) + grad * scalar(1.0 - beta2_);
    }
}

auto Lion::initialize_buffers() -> void {
    momentum_.clear();
    for (auto& param : parameters_) {
        if (param) {
            momentum_.push_back(zeros_like(param->tensor()));
        }
    }
}

auto Lion::set_lr(double lr) -> void { lr_ = lr; }
auto Lion::get_lr() const -> double { return lr_; }

auto Lion::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    state["step_count"] = Tensor({1}, DType::Int64, Device::cpu());
    state["step_count"].data<int64_t>()[0] = step_count_;

    state["lr"] = Tensor({1}, DType::Float64, Device::cpu());
    state["lr"].data<double>()[0] = lr_;

    state["beta1"] = Tensor({1}, DType::Float64, Device::cpu());
    state["beta1"].data<double>()[0] = beta1_;

    state["beta2"] = Tensor({1}, DType::Float64, Device::cpu());
    state["beta2"].data<double>()[0] = beta2_;

    state["weight_decay"] = Tensor({1}, DType::Float64, Device::cpu());
    state["weight_decay"].data<double>()[0] = weight_decay_;

    for (size_t i = 0; i < momentum_.size(); ++i) {
        state["momentum_" + std::to_string(i)] = momentum_[i].clone();
    }
    return state;
}

auto Lion::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    if (state.count("step_count"))  step_count_   = state.at("step_count").data<int64_t>()[0];
    if (state.count("lr"))          lr_           = state.at("lr").data<double>()[0];
    if (state.count("beta1"))       beta1_        = state.at("beta1").data<double>()[0];
    if (state.count("beta2"))       beta2_        = state.at("beta2").data<double>()[0];
    if (state.count("weight_decay")) weight_decay_ = state.at("weight_decay").data<double>()[0];

    // Validate momentum buffer count matches current parameter count
    size_t saved_count = 0;
    for (const auto& [key, _] : state) {
        if (key.starts_with("momentum_")) ++saved_count;
    }
    if (saved_count > 0 && saved_count != momentum_.size()) {
        throw std::runtime_error(
            "Lion::load_state_dict: momentum buffer count mismatch - saved " +
            std::to_string(saved_count) + " but have " +
            std::to_string(momentum_.size()) + " parameters");
    }

    for (size_t i = 0; i < momentum_.size(); ++i) {
        std::string key = "momentum_" + std::to_string(i);
        if (state.count(key)) {
            momentum_[i] = state.at(key).clone();
        }
    }
}

} // namespace tenzor::optim
