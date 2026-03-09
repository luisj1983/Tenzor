#include "tenzor/nn/optim/radam.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cmath>

namespace tenzor::optim {

RAdam::RAdam(std::vector<std::shared_ptr<Variable>> params, double lr, double beta1,
             double beta2, double eps, double weight_decay)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), weight_decay_(weight_decay) {
    initialize_buffers();
}

auto RAdam::step_impl() -> void {
    step_count_++;

    double rho_inf = 2.0 / (1.0 - beta2_) - 1.0;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        const Tensor& grad = param.grad().value();

        // CPU fallback path — use dtype-appropriate scalar tensors to
        // preserve Float64 precision (static_cast<float> truncates doubles)
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, param.tensor().dtype(), param.tensor().device());
        };

        auto grad_copy = grad.clone();

        // Weight decay (L2 regularization)
        if (weight_decay_ > 0.0) {
            grad_copy = grad_copy + param.tensor() * scalar(weight_decay_);
        }

        // Update biased first moment estimate
        exp_avg_[i] = exp_avg_[i] * scalar(beta1_) +
                     grad_copy * scalar(1.0 - beta1_);

        // Update biased second raw moment estimate
        exp_avg_sq_[i] = exp_avg_sq_[i] * scalar(beta2_) +
                        grad_copy * grad_copy * scalar(1.0 - beta2_);

        // Bias corrections
        double bias_correction1 = 1.0 - std::pow(beta1_, step_count_);
        double beta2_t = std::pow(beta2_, step_count_);
        double rho_t = rho_inf - 2.0 * step_count_ * beta2_t / (1.0 - beta2_t);

        if (rho_t > 5.0) {
            // Variance is tractable — use rectified Adam update
            double bias_correction2 = 1.0 - beta2_t;
            double rect = std::sqrt(
                (rho_t - 4.0) * (rho_t - 2.0) * rho_inf /
                ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
            );
            double step_size = lr_ * rect / bias_correction1;

            auto denom = sqrt(exp_avg_sq_[i]) * scalar(1.0 / std::sqrt(bias_correction2))
                        + scalar(eps_);
            param.tensor() = param.tensor() -
                            div(exp_avg_[i], denom) * scalar(step_size);
        } else {
            // Variance is intractable — use SGD with momentum
            double step_size = lr_ / bias_correction1;
            param.tensor() = param.tensor() -
                            exp_avg_[i] * scalar(step_size);
        }
    }
}

auto RAdam::initialize_buffers() -> void {
    exp_avg_.clear();
    exp_avg_sq_.clear();

    for (auto& param : parameters_) {
        if (param) {
            exp_avg_.push_back(zeros_like(param->tensor()));
            exp_avg_sq_.push_back(zeros_like(param->tensor()));
        }
    }
}

auto RAdam::set_lr(double lr) -> void {
    lr_ = lr;
}

auto RAdam::get_lr() const -> double {
    return lr_;
}

auto RAdam::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    // Save optimizer configuration as tensors
    state["step_count"] = Tensor({1}, DType::Int64, Device::cpu());
    state["step_count"].data<int64_t>()[0] = step_count_;

    state["lr"] = Tensor({1}, DType::Float64, Device::cpu());
    state["lr"].data<double>()[0] = lr_;

    state["beta1"] = Tensor({1}, DType::Float64, Device::cpu());
    state["beta1"].data<double>()[0] = beta1_;

    state["beta2"] = Tensor({1}, DType::Float64, Device::cpu());
    state["beta2"].data<double>()[0] = beta2_;

    state["eps"] = Tensor({1}, DType::Float64, Device::cpu());
    state["eps"].data<double>()[0] = eps_;

    state["weight_decay"] = Tensor({1}, DType::Float64, Device::cpu());
    state["weight_decay"].data<double>()[0] = weight_decay_;

    // Save momentum buffers
    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        state["exp_avg_" + std::to_string(i)] = exp_avg_[i].clone();
        state["exp_avg_sq_" + std::to_string(i)] = exp_avg_sq_[i].clone();
    }

    return state;
}

auto RAdam::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    // Load optimizer configuration
    if (state.count("step_count")) {
        step_count_ = state.at("step_count").data<int64_t>()[0];
    }

    if (state.count("lr")) {
        lr_ = state.at("lr").data<double>()[0];
    }

    if (state.count("beta1")) {
        beta1_ = state.at("beta1").data<double>()[0];
    }

    if (state.count("beta2")) {
        beta2_ = state.at("beta2").data<double>()[0];
    }

    if (state.count("eps")) {
        eps_ = state.at("eps").data<double>()[0];
    }

    if (state.count("weight_decay")) {
        weight_decay_ = state.at("weight_decay").data<double>()[0];
    }

    // Validate momentum buffer counts match current parameter count
    size_t saved_count = 0;
    for (const auto& [key, _] : state) {
        if (key.starts_with("exp_avg_") && !key.starts_with("exp_avg_sq_")) {
            ++saved_count;
        }
    }
    if (saved_count > 0 && saved_count != exp_avg_.size()) {
        throw std::runtime_error(
            "RAdam::load_state_dict: momentum buffer count mismatch - "
            "saved " + std::to_string(saved_count) + " but have " +
            std::to_string(exp_avg_.size()) + " parameters");
    }

    // Load momentum buffers
    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        std::string exp_avg_key = "exp_avg_" + std::to_string(i);
        std::string exp_avg_sq_key = "exp_avg_sq_" + std::to_string(i);

        if (state.count(exp_avg_key)) {
            exp_avg_[i] = state.at(exp_avg_key).clone();
        }

        if (state.count(exp_avg_sq_key)) {
            exp_avg_sq_[i] = state.at(exp_avg_sq_key).clone();
        }
    }
}

} // namespace tenzor::optim
