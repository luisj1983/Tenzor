/**
 * @file adagrad.cpp
 * @brief Implementation of Adagrad optimizer
 */

#include "tenzor/nn/optim/adagrad.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include <cmath>
#include <stdexcept>

namespace tenzor {
namespace optim {

Adagrad::Adagrad(std::vector<std::shared_ptr<Variable>> params,
                 double lr, double lr_decay, double weight_decay,
                 double initial_accumulator_value, double eps)
    : Optimizer(std::move(params)),
      lr_(lr),
      lr_decay_(lr_decay),
      weight_decay_(weight_decay),
      initial_accumulator_value_(initial_accumulator_value),
      eps_(eps),
      step_count_(0) {

    if (lr < 0.0) {
        throw std::invalid_argument("Learning rate must be non-negative");
    }
    if (lr_decay < 0.0) {
        throw std::invalid_argument("Learning rate decay must be non-negative");
    }
    if (weight_decay < 0.0) {
        throw std::invalid_argument("Weight decay must be non-negative");
    }
    if (initial_accumulator_value < 0.0) {
        throw std::invalid_argument("Initial accumulator value must be non-negative");
    }
    if (eps < 0.0) {
        throw std::invalid_argument("Epsilon must be non-negative");
    }

    initialize_buffers();
}

auto Adagrad::initialize_buffers() -> void {
    sum_.clear();

    for (auto& param : parameters_) {
        if (!param) continue;
        const auto& param_data = param->tensor();
        auto original_device = param_data.device();

        // Initialize accumulator G_0
        if (initial_accumulator_value_ == 0.0) {
            sum_.push_back(zeros_like(param_data));
        } else {
            // Create on CPU for data access, then move to target device
            std::vector<int64_t> shape_vec(param_data.shape().begin(), param_data.shape().end());
            Tensor accumulator_cpu = zeros(shape_vec, param_data.dtype(), Device::cpu());
            int64_t numel = accumulator_cpu.numel();
            DType dtype = param_data.dtype();

            if (dtype == DType::Float64) {
                double* acc_ptr = accumulator_cpu.data<double>();
                for (int64_t i = 0; i < numel; ++i) {
                    acc_ptr[i] = initial_accumulator_value_;
                }
            } else {
                float* acc_ptr = accumulator_cpu.data<float>();
                for (int64_t i = 0; i < numel; ++i) {
                    acc_ptr[i] = static_cast<float>(initial_accumulator_value_);
                }
            }

            sum_.push_back(accumulator_cpu.to(original_device));
        }
    }
}

auto Adagrad::effective_lr() const -> double {
    if (lr_decay_ == 0.0 || step_count_ == 0) {
        return lr_;
    }
    return lr_ / (1.0 + step_count_ * lr_decay_);
}

auto Adagrad::step() -> void {
    step_count_++;
    double current_lr = effective_lr();

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param = parameters_[i];

        if (!param || !param->has_grad()) {
            continue;  // Skip parameters without gradients
        }

        const auto& grad_orig = param->grad().value();
        const auto& param_data_orig = param->tensor();
        auto original_device = param_data_orig.device();

        // Vulkan fast path: fused kernel avoids GPU→CPU→GPU round-trip
        if (original_device.type == Device::Type::Vulkan &&
            grad_orig.device().type == Device::Type::Vulkan) {

            std::vector<Tensor> inputs = {grad_orig, param->tensor(), sum_[i]};

            OpAttributes attrs;
            attrs["lr"] = std::to_string(static_cast<float>(current_lr));
            attrs["eps"] = std::to_string(static_cast<float>(eps_));
            attrs["weight_decay"] = std::to_string(static_cast<float>(weight_decay_));

            dispatch(OpId::FusedAdagradStep, inputs, attrs);
            continue;
        }

        // Move all tensors to CPU for data access
        Tensor grad = grad_orig.to(Device::cpu());
        Tensor param_data = param_data_orig.to(Device::cpu());
        Tensor sum_cpu = sum_[i].to(Device::cpu());

        int64_t numel = param_data.numel();
        DType dtype = param_data.dtype();

        // Handle different dtypes
        if (dtype == DType::Float64) {
            auto grad_ptr = const_cast<double*>(grad.data<double>());
            auto param_ptr = const_cast<double*>(param_data.data<double>());
            auto sum_ptr = sum_cpu.data<double>();

            // Apply weight decay if specified
            if (weight_decay_ > 0.0) {
                for (int64_t j = 0; j < numel; ++j) {
                    grad_ptr[j] += weight_decay_ * param_ptr[j];
                }
            }

            // Update accumulator: G_t = G_{t-1} + g_t^2
            // Update parameters: theta_t = theta_{t-1} - lr / (sqrt(G_t) + eps) * g_t
            for (int64_t j = 0; j < numel; ++j) {
                sum_ptr[j] += grad_ptr[j] * grad_ptr[j];
                double std_dev = std::sqrt(sum_ptr[j]) + eps_;
                param_ptr[j] -= current_lr * grad_ptr[j] / std_dev;
            }
        } else {
            // Float32 path (default)
            auto grad_ptr = const_cast<float*>(grad.data<float>());
            auto param_ptr = const_cast<float*>(param_data.data<float>());
            auto sum_ptr = sum_cpu.data<float>();

            // Apply weight decay if specified
            if (weight_decay_ > 0.0) {
                for (int64_t j = 0; j < numel; ++j) {
                    grad_ptr[j] += weight_decay_ * param_ptr[j];
                }
            }

            // Update accumulator: G_t = G_{t-1} + g_t^2
            // Update parameters: theta_t = theta_{t-1} - lr / (sqrt(G_t) + eps) * g_t
            for (int64_t j = 0; j < numel; ++j) {
                sum_ptr[j] += grad_ptr[j] * grad_ptr[j];
                float std_dev = std::sqrt(sum_ptr[j]) + static_cast<float>(eps_);
                param_ptr[j] -= static_cast<float>(current_lr) * grad_ptr[j] / std_dev;
            }
        }

        // Copy updated values back to original device
        param->tensor() = param_data.to(original_device);
        sum_[i] = sum_cpu.to(original_device);
    }
}

auto Adagrad::zero_grad() -> void {
    Optimizer::zero_grad();
}

auto Adagrad::set_lr(double lr) -> void {
    if (lr < 0.0) {
        throw std::invalid_argument("Learning rate must be non-negative");
    }
    lr_ = lr;
}

auto Adagrad::get_lr() const -> double {
    return effective_lr();
}

auto Adagrad::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        std::string prefix = "param_" + std::to_string(i);
        state[prefix + ".sum"] = sum_[i];
    }

    // Store step count as scalar tensor
    auto step_tensor = zeros({1});
    auto step_ptr = step_tensor.data<float>();
    step_ptr[0] = static_cast<float>(step_count_);
    state["step_count"] = step_tensor;

    return state;
}

auto Adagrad::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        std::string prefix = "param_" + std::to_string(i);

        auto it = state.find(prefix + ".sum");
        if (it != state.end()) {
            sum_[i] = it->second;
        }
    }

    // Load step count
    auto it = state.find("step_count");
    if (it != state.end()) {
        auto step_ptr = it->second.data<float>();
        step_count_ = static_cast<int64_t>(step_ptr[0]);
    }
}

} // namespace optim
} // namespace tenzor
