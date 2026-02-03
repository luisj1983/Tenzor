/**
 * @file adam_atan2.cpp
 * @brief Adam-atan2 optimizer implementation
 */

#include "tenzor/nn/optim/adam_atan2.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include <cmath>
#include <algorithm>

namespace tenzor::optim {

// ============================================================================
// AdamAtan2 Implementation
// ============================================================================

AdamAtan2::AdamAtan2(std::vector<std::shared_ptr<Variable>> params,
                     double lr, double beta1, double beta2,
                     double eps, double weight_decay, bool amsgrad)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), weight_decay_(weight_decay), amsgrad_(amsgrad) {
    initialize_buffers();
    update_stats_ = {0.0, 0.0, 0.0};
}

auto AdamAtan2::step() -> void {
    step_count_++;

    double total_update_mag = 0.0;
    double max_update_mag = 0.0;
    double total_grad_mag = 0.0;
    int64_t total_params = 0;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        Tensor& param_tensor = param.tensor();
        const Tensor& grad_tensor = param.grad().value();

        // Use fused CUDA kernel for CUDA tensors
        if (param_tensor.device().type == Device::Type::CUDA &&
            grad_tensor.device().type == Device::Type::CUDA &&
            (param_tensor.dtype() == DType::Float32 || param_tensor.dtype() == DType::Float64)) {

            std::vector<Tensor> inputs = {
                param_tensor, grad_tensor, exp_avg_[i], exp_avg_sq_[i]
            };
            if (amsgrad_ && i < max_exp_avg_sq_.size()) {
                inputs.push_back(max_exp_avg_sq_[i]);
            }

            OpAttributes attrs;
            attrs["lr"] = std::to_string(static_cast<float>(lr_));
            attrs["beta1"] = std::to_string(static_cast<float>(beta1_));
            attrs["beta2"] = std::to_string(static_cast<float>(beta2_));
            attrs["eps"] = std::to_string(static_cast<float>(eps_));
            attrs["weight_decay"] = std::to_string(static_cast<float>(weight_decay_));
            attrs["step"] = std::to_string(step_count_);
            attrs["amsgrad"] = amsgrad_ ? "true" : "false";

            dispatch(OpId::FusedAdamAtan2Step, inputs, attrs);
            total_params++;
            continue;
        }

        auto grad = param.grad()->clone();

        // Track gradient magnitude only when statistics tracking is enabled
        // (.item() forces GPU synchronization per parameter, which is expensive)
        if (track_statistics_) {
            auto grad_sq_sum = tenzor::sum(grad * grad).item<float>();
            total_grad_mag += std::sqrt(grad_sq_sum);
        }
        total_params++;

        // Update biased first moment estimate
        exp_avg_[i] = exp_avg_[i] * static_cast<float>(beta1_) +
                     grad * static_cast<float>(1.0 - beta1_);

        // Update biased second raw moment estimate
        exp_avg_sq_[i] = exp_avg_sq_[i] * static_cast<float>(beta2_) +
                        grad * grad * static_cast<float>(1.0 - beta2_);

        // Bias correction
        double bias_correction1 = 1.0 - std::pow(beta1_, step_count_);
        double bias_correction2 = 1.0 - std::pow(beta2_, step_count_);

        // Compute bias-corrected estimates
        auto m_hat = exp_avg_[i] * static_cast<float>(1.0 / bias_correction1);
        auto v_hat = exp_avg_sq_[i] * static_cast<float>(1.0 / bias_correction2);

        // AMSGrad: use maximum of past squared gradients
        if (amsgrad_) {
            // Element-wise max using comparison
            auto mask = max_exp_avg_sq_[i] > v_hat;
            auto mask_f = mask.to(DType::Float32);
            auto inv_mask = ones_like(mask_f) - mask_f;
            max_exp_avg_sq_[i] = max_exp_avg_sq_[i] * mask_f + v_hat * inv_mask;
            v_hat = max_exp_avg_sq_[i];
        }

        // Compute sqrt(v_hat) + eps
        auto denom = sqrt(v_hat) + static_cast<float>(eps_);

        // Adam-atan2 update: use atan(m_hat / denom) to approximate atan2
        // atan2(y, x) ≈ atan(y/x) when x > 0 (which is always true for denom)
        // This provides bounded updates in [-π/2, π/2]
        auto ratio = div(m_hat, denom);
        auto update = tenzor::atan(ratio);

        // Track update magnitude only when statistics tracking is enabled
        if (track_statistics_) {
            auto update_flat = update.view({-1});
            auto update_sq = update_flat * update_flat;
            auto update_sq_sum = tenzor::sum(update_sq).template item<float>();
            double update_mag = std::sqrt(update_sq_sum);
            total_update_mag += update_mag;
            max_update_mag = std::max(max_update_mag, update_mag);
        }

        // Decoupled weight decay (like AdamW)
        if (weight_decay_ > 0) {
            param.tensor() = param.tensor() * static_cast<float>(1.0 - lr_ * weight_decay_);
        }

        // Apply update
        param.tensor() = param.tensor() - update * static_cast<float>(lr_);
    }

    // Update statistics (only meaningful when tracking is enabled)
    if (track_statistics_ && total_params > 0) {
        update_stats_.avg_update_magnitude = total_update_mag / total_params;
        update_stats_.max_update_magnitude = max_update_mag;
        update_stats_.avg_gradient_magnitude = total_grad_mag / total_params;
    }
}

auto AdamAtan2::initialize_buffers() -> void {
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

auto AdamAtan2::set_lr(double lr) -> void {
    lr_ = lr;
}

auto AdamAtan2::get_lr() const -> double {
    return lr_;
}

auto AdamAtan2::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    // Save optimizer configuration
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
        if (amsgrad_ && i < max_exp_avg_sq_.size()) {
            state["max_exp_avg_sq_" + std::to_string(i)] = max_exp_avg_sq_[i].clone();
        }
    }

    return state;
}

auto AdamAtan2::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
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
        std::string max_key = "max_exp_avg_sq_" + std::to_string(i);

        if (state.count(exp_avg_key)) {
            exp_avg_[i] = state.at(exp_avg_key).clone();
        }

        if (state.count(exp_avg_sq_key)) {
            exp_avg_sq_[i] = state.at(exp_avg_sq_key).clone();
        }

        if (amsgrad_ && state.count(max_key) && i < max_exp_avg_sq_.size()) {
            max_exp_avg_sq_[i] = state.at(max_key).clone();
        }
    }
}

// ============================================================================
// LinearWarmup Implementation
// ============================================================================

LinearWarmup::LinearWarmup(Optimizer& optimizer, int64_t warmup_steps, double base_lr)
    : optimizer_(optimizer), warmup_steps_(warmup_steps), base_lr_(base_lr), current_lr_(0.0) {
}

auto LinearWarmup::update(int64_t step) -> void {
    if (step < warmup_steps_) {
        // Linear warmup
        current_lr_ = base_lr_ * static_cast<double>(step + 1) / static_cast<double>(warmup_steps_);
    } else {
        // After warmup, use base learning rate
        current_lr_ = base_lr_;
    }

    // Update optimizer's learning rate
    // This requires the optimizer to have a set_lr method
    // We use dynamic dispatch via dynamic_cast
    if (auto* adam_atan2 = dynamic_cast<AdamAtan2*>(&optimizer_)) {
        adam_atan2->set_lr(current_lr_);
    }
    // Add other optimizer types as needed
}

} // namespace tenzor::optim
