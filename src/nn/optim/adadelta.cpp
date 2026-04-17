/**
 * @file adadelta.cpp
 * @brief Implementation of Adadelta optimizer
 */

#include "tenzor/nn/optim/adadelta.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include <cmath>
#include <cstring>
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

auto Adadelta::step_impl() -> void {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param = parameters_[i];

        if (!param || !param->has_grad()) {
            continue;  // Skip parameters without gradients
        }

        const auto& grad_orig = param->grad().value();
        const auto& param_data_orig = param->tensor();
        auto original_device = param_data_orig.device();

        // Vulkan fast path: fused kernel avoids GPU→CPU→GPU round-trip.
        // Input order matches the CUDA/CPU contract [param, grad, square_avg,
        // acc_delta]; the Vulkan dispatch remaps to shader-binding order
        // internally. Previous [grad, param, ...] order was a mismatch with
        // dispatchFusedAdadeltaStep's expected layout.
        if (original_device.type == Device::Type::Vulkan &&
            grad_orig.device().type == Device::Type::Vulkan) {

            std::vector<Tensor> inputs = {
                param->tensor(), grad_orig, square_avg_[i], acc_delta_[i]
            };

            NewOpAttributes attrs;
            attrs.set(AttrKey::Lr, static_cast<float>(lr_));
            attrs.set(AttrKey::Rho, static_cast<float>(rho_));
            attrs.set(AttrKey::Eps, static_cast<float>(eps_));
            attrs.set(AttrKey::WeightDecay, static_cast<float>(weight_decay_));

            dispatch(OpId::FusedAdadeltaStep, inputs, attrs);
            continue;
        }

        // CUDA fast path: fused kernel avoids GPU→CPU→GPU round-trip
        if (original_device.type == Device::Type::CUDA &&
            grad_orig.device().type == Device::Type::CUDA) {

            // CUDA registry expects: [param, grad, square_avg, acc_delta]
            std::vector<Tensor> inputs = {
                param->tensor(), grad_orig, square_avg_[i], acc_delta_[i]
            };

            NewOpAttributes attrs;
            attrs.set(AttrKey::Rho, static_cast<float>(rho_));
            attrs.set(AttrKey::Eps, static_cast<float>(eps_));
            attrs.set(AttrKey::Lr, static_cast<float>(lr_));
            attrs.set(AttrKey::WeightDecay, static_cast<float>(weight_decay_));

            dispatch(OpId::FusedAdadeltaStep, inputs, attrs);
            continue;
        }

        // Generic fallback using tensor-level ops (device-agnostic)
        Tensor grad = grad_orig;
        Tensor param_data = param_data_orig;
        float rho = static_cast<float>(rho_);
        float eps = static_cast<float>(eps_);
        float lr = static_cast<float>(lr_);

        // Apply weight decay: g = g + weight_decay * param
        if (weight_decay_ > 0.0) {
            grad = grad + param_data * static_cast<float>(weight_decay_);
        }

        // Accumulate squared gradient: v = rho * v + (1 - rho) * g^2
        square_avg_[i] = square_avg_[i] * rho + grad * grad * (1.0f - rho);

        // Compute RMS of gradients and updates
        auto std_grad = sqrt(square_avg_[i] + eps);
        auto std_delta = sqrt(acc_delta_[i] + eps);

        // Compute parameter update: delta = -(std_delta / std_grad) * g
        auto delta = (std_delta / std_grad) * grad * (-1.0f);

        // Apply update: param += lr * delta
        auto new_param = param_data + delta * lr;

        // Copy result into existing tensor storage (preserves pointer stability on CPU)
        if (original_device.type == Device::Type::CPU) {
            auto src = new_param.contiguous();
            std::memcpy(param->tensor().data_ptr(), src.data_ptr(),
                        src.numel() * dtype_size(src.dtype()));
        } else {
            param->tensor() = new_param;
        }

        // Accumulate squared update: acc = rho * acc + (1 - rho) * delta^2
        acc_delta_[i] = acc_delta_[i] * rho + delta * delta * (1.0f - rho);
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
