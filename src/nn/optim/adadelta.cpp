/**
 * @file adadelta.cpp
 * @brief Implementation of Adadelta optimizer
 */

#include "tenzor/nn/optim/adadelta.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include <cmath>
#include <stdexcept>

namespace tenzor {
namespace optim {

Adadelta::Adadelta(std::vector<std::shared_ptr<Variable>> params,
                   double lr, double rho, double eps, double weight_decay)
    : Optimizer(std::move(params)),
      lr_(lr),
      rho_(rho),
      eps_(eps),
      weight_decay_(weight_decay) {

    if (lr < 0.0) {
        throw std::invalid_argument("Learning rate must be non-negative");
    }
    if (rho < 0.0 || rho > 1.0) {
        throw std::invalid_argument("Rho must be in [0, 1]");
    }
    if (eps < 0.0) {
        throw std::invalid_argument("Epsilon must be non-negative");
    }
    if (weight_decay < 0.0) {
        throw std::invalid_argument("Weight decay must be non-negative");
    }

    initialize_buffers();
}

auto Adadelta::initialize_buffers() -> void {
    square_avg_.clear();
    acc_delta_.clear();

    for (auto& param : parameters_) {
        if (!param) continue;
        const auto& param_data = param->tensor();

        // Initialize E[g^2] to zeros
        square_avg_.push_back(zeros_like(param_data));

        // Initialize E[Δθ^2] to zeros
        acc_delta_.push_back(zeros_like(param_data));
    }
}

auto Adadelta::step() -> void {
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

            std::vector<Tensor> inputs = {
                grad_orig, param->tensor(), square_avg_[i], acc_delta_[i]
            };

            OpAttributes attrs;
            attrs["lr"] = std::to_string(static_cast<float>(lr_));
            attrs["rho"] = std::to_string(static_cast<float>(rho_));
            attrs["eps"] = std::to_string(static_cast<float>(eps_));
            attrs["weight_decay"] = std::to_string(static_cast<float>(weight_decay_));

            dispatch(OpId::FusedAdadeltaStep, inputs, attrs);
            continue;
        }

        // Move all tensors to CPU for data access
        Tensor grad = grad_orig.to(Device::cpu());
        Tensor param_data = param_data_orig.to(Device::cpu());
        Tensor square_avg_cpu = square_avg_[i].to(Device::cpu());
        Tensor acc_delta_cpu = acc_delta_[i].to(Device::cpu());

        int64_t numel = param_data.numel();
        DType dtype = param_data.dtype();

        // Handle different dtypes
        if (dtype == DType::Float64) {
            auto grad_ptr = const_cast<double*>(grad.data<double>());
            auto param_ptr = const_cast<double*>(param_data.data<double>());
            auto square_avg_ptr = square_avg_cpu.data<double>();
            auto acc_delta_ptr = acc_delta_cpu.data<double>();

            // Apply weight decay if specified
            if (weight_decay_ > 0.0) {
                for (int64_t j = 0; j < numel; ++j) {
                    grad_ptr[j] += weight_decay_ * param_ptr[j];
                }
            }

            // Adadelta update
            for (int64_t j = 0; j < numel; ++j) {
                // Accumulate squared gradient
                square_avg_ptr[j] = rho_ * square_avg_ptr[j] +
                                    (1.0 - rho_) * grad_ptr[j] * grad_ptr[j];

                // Compute std of gradients
                double std_grad = std::sqrt(square_avg_ptr[j] + eps_);

                // Compute std of updates (using previous accumulator)
                double std_delta = std::sqrt(acc_delta_ptr[j] + eps_);

                // Compute parameter update
                double delta = -(std_delta / std_grad) * grad_ptr[j];

                // Apply update to parameter
                param_ptr[j] += lr_ * delta;

                // Accumulate squared update (after applying delta)
                acc_delta_ptr[j] = rho_ * acc_delta_ptr[j] +
                                   (1.0 - rho_) * delta * delta;
            }
        } else {
            // Float32 path (default)
            auto grad_ptr = const_cast<float*>(grad.data<float>());
            auto param_ptr = const_cast<float*>(param_data.data<float>());
            auto square_avg_ptr = square_avg_cpu.data<float>();
            auto acc_delta_ptr = acc_delta_cpu.data<float>();

            // Apply weight decay if specified
            if (weight_decay_ > 0.0) {
                for (int64_t j = 0; j < numel; ++j) {
                    grad_ptr[j] += weight_decay_ * param_ptr[j];
                }
            }

            // Adadelta update
            for (int64_t j = 0; j < numel; ++j) {
                // Accumulate squared gradient
                square_avg_ptr[j] = rho_ * square_avg_ptr[j] +
                                    (1.0f - rho_) * grad_ptr[j] * grad_ptr[j];

                // Compute std of gradients
                float std_grad = std::sqrt(square_avg_ptr[j] + eps_);

                // Compute std of updates (using previous accumulator)
                float std_delta = std::sqrt(acc_delta_ptr[j] + eps_);

                // Compute parameter update
                float delta = -(std_delta / std_grad) * grad_ptr[j];

                // Apply update to parameter
                param_ptr[j] += lr_ * delta;

                // Accumulate squared update (after applying delta)
                acc_delta_ptr[j] = rho_ * acc_delta_ptr[j] +
                                   (1.0f - rho_) * delta * delta;
            }
        }

        // Copy updated values back to original device
        param->tensor() = param_data.to(original_device);
        square_avg_[i] = square_avg_cpu.to(original_device);
        acc_delta_[i] = acc_delta_cpu.to(original_device);
    }
}

auto Adadelta::zero_grad() -> void {
    Optimizer::zero_grad();
}

auto Adadelta::set_lr(double lr) -> void {
    if (lr < 0.0) {
        throw std::invalid_argument("Learning rate must be non-negative");
    }
    lr_ = lr;
}

auto Adadelta::get_lr() const -> double {
    return lr_;
}

auto Adadelta::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        std::string prefix = "param_" + std::to_string(i);
        state[prefix + ".square_avg"] = square_avg_[i];
        state[prefix + ".acc_delta"] = acc_delta_[i];
    }

    return state;
}

auto Adadelta::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        std::string prefix = "param_" + std::to_string(i);

        auto it = state.find(prefix + ".square_avg");
        if (it != state.end()) {
            square_avg_[i] = it->second;
        }

        it = state.find(prefix + ".acc_delta");
        if (it != state.end()) {
            acc_delta_[i] = it->second;
        }
    }
}

} // namespace optim
} // namespace tenzor
