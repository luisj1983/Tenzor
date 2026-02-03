#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include <cmath>

namespace tenzor::optim {

// Adam::Adam implementation
Adam::Adam(std::vector<std::shared_ptr<Variable>> params, double lr, double beta1,
          double beta2, double eps, double weight_decay, bool amsgrad)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), weight_decay_(weight_decay), amsgrad_(amsgrad) {
    initialize_buffers();
}

auto Adam::step() -> void {
    step_count_++;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        const Tensor& grad = param.grad().value();

        // Use fused CUDA kernel for CUDA tensors (single kernel vs ~15 kernels)
        if (param.tensor().device().type == Device::Type::CUDA &&
            grad.device().type == Device::Type::CUDA &&
            (param.tensor().dtype() == DType::Float32 || param.tensor().dtype() == DType::Float64)) {

            // Prepare inputs for dispatch
            std::vector<Tensor> inputs = {
                param.tensor(), grad, exp_avg_[i], exp_avg_sq_[i]
            };
            if (amsgrad_ && i < max_exp_avg_sq_.size()) {
                inputs.push_back(max_exp_avg_sq_[i]);
            }

            // Prepare attributes (use double precision for Float64 accuracy)
            OpAttributes attrs;
            attrs["lr"] = std::to_string(lr_);
            attrs["beta1"] = std::to_string(beta1_);
            attrs["beta2"] = std::to_string(beta2_);
            attrs["eps"] = std::to_string(eps_);
            attrs["weight_decay"] = std::to_string(weight_decay_);
            attrs["step"] = std::to_string(step_count_);
            attrs["decoupled"] = "false";  // L2 regularization for Adam
            attrs["amsgrad"] = amsgrad_ ? "true" : "false";

            dispatch(OpId::FusedAdamStep, inputs, attrs);
            continue;
        }

        // CPU fallback path
        auto grad_copy = grad.clone();

        // Weight decay
        if (weight_decay_ > 0.0) {
            grad_copy = grad_copy + param.tensor() * static_cast<float>(weight_decay_);
        }

        // Update biased first moment estimate
        exp_avg_[i] = exp_avg_[i] * static_cast<float>(beta1_) +
                     grad_copy * static_cast<float>(1.0 - beta1_);

        // Update biased second raw moment estimate
        exp_avg_sq_[i] = exp_avg_sq_[i] * static_cast<float>(beta2_) +
                        grad_copy * grad_copy * static_cast<float>(1.0 - beta2_);

        // Bias correction
        double bias_correction1 = 1.0 - std::pow(beta1_, step_count_);
        double bias_correction2 = 1.0 - std::pow(beta2_, step_count_);

        double step_size = lr_ / bias_correction1;

        // Compute denominator
        auto denom = sqrt(exp_avg_sq_[i]) *
                    static_cast<float>(1.0 / std::sqrt(bias_correction2)) +
                    static_cast<float>(eps_);

        // Update parameters
        param.tensor() = param.tensor() -
                        div(exp_avg_[i], denom) * static_cast<float>(step_size);
    }
}

auto Adam::initialize_buffers() -> void {
    exp_avg_.clear();
    exp_avg_sq_.clear();
    max_exp_avg_sq_.clear();

    for (auto& param : parameters_) {
        if (param) {
            exp_avg_.push_back(zeros_like(param->tensor()));
            exp_avg_sq_.push_back(zeros_like(param->tensor()));
            if (amsgrad_) {
                max_exp_avg_sq_.push_back(zeros_like(param->tensor()));
            }
        }
    }
}

auto Adam::set_lr(double lr) -> void {
    lr_ = lr;
}

auto Adam::get_lr() const -> double {
    return lr_;
}

auto Adam::state_dict() const -> std::unordered_map<std::string, Tensor> {
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

auto Adam::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
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

// AdamW implementation
AdamW::AdamW(std::vector<std::shared_ptr<Variable>> params, double lr, double beta1,
            double beta2, double eps, double weight_decay, bool amsgrad)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), weight_decay_(weight_decay), amsgrad_(amsgrad) {
    initialize_buffers();
}

auto AdamW::step() -> void {
    step_count_++;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        const Tensor& grad = param.grad().value();

        // Use fused CUDA kernel for CUDA tensors (single kernel vs ~15 kernels)
        if (param.tensor().device().type == Device::Type::CUDA &&
            grad.device().type == Device::Type::CUDA &&
            (param.tensor().dtype() == DType::Float32 || param.tensor().dtype() == DType::Float64)) {

            // Prepare inputs for dispatch
            std::vector<Tensor> inputs = {
                param.tensor(), grad, exp_avg_[i], exp_avg_sq_[i]
            };
            if (amsgrad_ && i < max_exp_avg_sq_.size()) {
                inputs.push_back(max_exp_avg_sq_[i]);
            }

            // Prepare attributes (use double precision for Float64 accuracy)
            OpAttributes attrs;
            attrs["lr"] = std::to_string(lr_);
            attrs["beta1"] = std::to_string(beta1_);
            attrs["beta2"] = std::to_string(beta2_);
            attrs["eps"] = std::to_string(eps_);
            attrs["weight_decay"] = std::to_string(weight_decay_);
            attrs["step"] = std::to_string(step_count_);
            attrs["decoupled"] = "true";  // Decoupled weight decay for AdamW
            attrs["amsgrad"] = amsgrad_ ? "true" : "false";

            dispatch(OpId::FusedAdamStep, inputs, attrs);
            continue;
        }

        // CPU fallback path
        auto grad_copy = grad.clone();

        // Update biased first moment estimate
        exp_avg_[i] = exp_avg_[i] * static_cast<float>(beta1_) +
                     grad_copy * static_cast<float>(1.0 - beta1_);

        // Update biased second raw moment estimate
        exp_avg_sq_[i] = exp_avg_sq_[i] * static_cast<float>(beta2_) +
                        grad_copy * grad_copy * static_cast<float>(1.0 - beta2_);

        // Bias correction
        double bias_correction1 = 1.0 - std::pow(beta1_, step_count_);
        double bias_correction2 = 1.0 - std::pow(beta2_, step_count_);

        double step_size = lr_ / bias_correction1;

        // Compute denominator
        auto denom = sqrt(exp_avg_sq_[i]) *
                    static_cast<float>(1.0 / std::sqrt(bias_correction2)) +
                    static_cast<float>(eps_);

        // Decoupled weight decay (AdamW)
        if (weight_decay_ > 0) {
            param.tensor() = param.tensor() * static_cast<float>(1.0 - lr_ * weight_decay_);
        }

        // Update parameters
        param.tensor() = param.tensor() -
                        div(exp_avg_[i], denom) * static_cast<float>(step_size);
    }
}

auto AdamW::set_lr(double lr) -> void {
    lr_ = lr;
}

auto AdamW::get_lr() const -> double {
    return lr_;
}

auto AdamW::initialize_buffers() -> void {
    exp_avg_.clear();
    exp_avg_sq_.clear();
    max_exp_avg_sq_.clear();

    for (auto& param : parameters_) {
        if (param) {
            exp_avg_.push_back(zeros_like(param->tensor()));
            exp_avg_sq_.push_back(zeros_like(param->tensor()));
            if (amsgrad_) {
                max_exp_avg_sq_.push_back(zeros_like(param->tensor()));
            }
        }
    }
}

auto AdamW::state_dict() const -> std::unordered_map<std::string, Tensor> {
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

auto AdamW::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
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
