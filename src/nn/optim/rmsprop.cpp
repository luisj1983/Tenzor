/**
 * @file rmsprop.cpp
 * @brief Implementation of RMSprop optimizer
 */

#include "tenzor/nn/optim/rmsprop.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include <cmath>
#include <stdexcept>

namespace tenzor {
namespace optim {

RMSprop::RMSprop(std::vector<std::shared_ptr<Variable>> params,
                 double lr, double alpha, double eps,
                 double weight_decay, double momentum, bool centered)
    : Optimizer(std::move(params)),
      lr_(lr),
      alpha_(alpha),
      eps_(eps),
      weight_decay_(weight_decay),
      momentum_(momentum),
      centered_(centered) {

    if (lr < 0.0) {
        throw std::invalid_argument("Learning rate must be non-negative");
    }
    if (alpha < 0.0 || alpha > 1.0) {
        throw std::invalid_argument("Alpha must be in [0, 1]");
    }
    if (eps < 0.0) {
        throw std::invalid_argument("Epsilon must be non-negative");
    }
    if (weight_decay < 0.0) {
        throw std::invalid_argument("Weight decay must be non-negative");
    }
    if (momentum < 0.0) {
        throw std::invalid_argument("Momentum must be non-negative");
    }

    initialize_buffers();
}

auto RMSprop::initialize_buffers() -> void {
    square_avg_.clear();
    grad_avg_.clear();
    momentum_buffer_.clear();

    for (auto& param : parameters_) {
        if (!param) continue;
        const auto& param_data = param->tensor();

        // Initialize square_avg (E[g^2]) to zeros
        square_avg_.push_back(zeros_like(param_data));

        // Initialize grad_avg (E[g]) if centered
        if (centered_) {
            grad_avg_.push_back(zeros_like(param_data));
        }

        // Initialize momentum buffer if momentum > 0
        if (momentum_ > 0.0) {
            momentum_buffer_.push_back(zeros_like(param_data));
        }
    }
}

auto RMSprop::step() -> void {
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

            std::vector<Tensor> inputs = {grad_orig, param->tensor(), square_avg_[i]};
            if (momentum_ > 0.0 && i < momentum_buffer_.size()) {
                inputs.push_back(momentum_buffer_[i]);
            }
            if (centered_ && i < grad_avg_.size()) {
                inputs.push_back(grad_avg_[i]);
            }

            OpAttributes attrs;
            attrs["lr"] = std::to_string(static_cast<float>(lr_));
            attrs["alpha"] = std::to_string(static_cast<float>(alpha_));
            attrs["eps"] = std::to_string(static_cast<float>(eps_));
            attrs["weight_decay"] = std::to_string(static_cast<float>(weight_decay_));
            attrs["momentum"] = std::to_string(static_cast<float>(momentum_));
            attrs["centered"] = centered_ ? "1" : "0";

            dispatch(OpId::FusedRMSPropStep, inputs, attrs);
            continue;
        }

        // Move all tensors to CPU for data access
        Tensor grad = grad_orig.to(Device::cpu());
        Tensor param_data = param_data_orig.to(Device::cpu());
        Tensor square_avg_cpu = square_avg_[i].to(Device::cpu());
        Tensor grad_avg_cpu;
        if (centered_) {
            grad_avg_cpu = grad_avg_[i].to(Device::cpu());
        }
        Tensor momentum_buffer_cpu;
        if (momentum_ > 0.0) {
            momentum_buffer_cpu = momentum_buffer_[i].to(Device::cpu());
        }

        int64_t numel = param_data.numel();
        DType dtype = param_data.dtype();

        // Handle different dtypes
        if (dtype == DType::Float64) {
            auto grad_ptr = const_cast<double*>(grad.data<double>());
            auto param_ptr = const_cast<double*>(param_data.data<double>());
            auto square_avg_ptr = square_avg_cpu.data<double>();

            // Apply weight decay if specified
            if (weight_decay_ > 0.0) {
                for (int64_t j = 0; j < numel; ++j) {
                    grad_ptr[j] += weight_decay_ * param_ptr[j];
                }
            }

            // Update square_avg: v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
            for (int64_t j = 0; j < numel; ++j) {
                square_avg_ptr[j] = alpha_ * square_avg_ptr[j] +
                                    (1.0 - alpha_) * grad_ptr[j] * grad_ptr[j];
            }

            double* avg_ptr = nullptr;

            if (centered_) {
                // Update grad_avg: m_t = alpha * m_{t-1} + (1 - alpha) * g_t
                avg_ptr = grad_avg_cpu.data<double>();
                for (int64_t j = 0; j < numel; ++j) {
                    avg_ptr[j] = alpha_ * avg_ptr[j] + (1.0 - alpha_) * grad_ptr[j];
                }
            }

            if (momentum_ > 0.0) {
                // With momentum: buf_t = momentum * buf_{t-1} + g_t / (sqrt(v_t) + eps)
                auto buf_ptr = momentum_buffer_cpu.data<double>();

                for (int64_t j = 0; j < numel; ++j) {
                    double denom;
                    if (centered_) {
                        double centered_var = square_avg_ptr[j] - avg_ptr[j] * avg_ptr[j];
                        denom = std::sqrt(centered_var + eps_);
                    } else {
                        denom = std::sqrt(square_avg_ptr[j] + eps_);
                    }

                    buf_ptr[j] = momentum_ * buf_ptr[j] + grad_ptr[j] / denom;
                    param_ptr[j] -= lr_ * buf_ptr[j];
                }
            } else {
                // Without momentum: theta_t = theta_{t-1} - lr * g_t / (sqrt(v_t) + eps)
                for (int64_t j = 0; j < numel; ++j) {
                    double denom;
                    if (centered_) {
                        double centered_var = square_avg_ptr[j] - avg_ptr[j] * avg_ptr[j];
                        denom = std::sqrt(centered_var + eps_);
                    } else {
                        denom = std::sqrt(square_avg_ptr[j] + eps_);
                    }

                    param_ptr[j] -= lr_ * grad_ptr[j] / denom;
                }
            }
        } else {
            // Float32 path (default)
            auto grad_ptr = const_cast<float*>(grad.data<float>());
            auto param_ptr = const_cast<float*>(param_data.data<float>());
            auto square_avg_ptr = square_avg_cpu.data<float>();

            // Apply weight decay if specified
            if (weight_decay_ > 0.0) {
                for (int64_t j = 0; j < numel; ++j) {
                    grad_ptr[j] += weight_decay_ * param_ptr[j];
                }
            }

            // Update square_avg: v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
            for (int64_t j = 0; j < numel; ++j) {
                square_avg_ptr[j] = alpha_ * square_avg_ptr[j] +
                                    (1.0f - alpha_) * grad_ptr[j] * grad_ptr[j];
            }

            float* avg_ptr = nullptr;

            if (centered_) {
                // Update grad_avg: m_t = alpha * m_{t-1} + (1 - alpha) * g_t
                avg_ptr = grad_avg_cpu.data<float>();
                for (int64_t j = 0; j < numel; ++j) {
                    avg_ptr[j] = alpha_ * avg_ptr[j] + (1.0f - alpha_) * grad_ptr[j];
                }
            }

            if (momentum_ > 0.0) {
                // With momentum: buf_t = momentum * buf_{t-1} + g_t / (sqrt(v_t) + eps)
                auto buf_ptr = momentum_buffer_cpu.data<float>();

                for (int64_t j = 0; j < numel; ++j) {
                    float denom;
                    if (centered_) {
                        float centered_var = square_avg_ptr[j] - avg_ptr[j] * avg_ptr[j];
                        denom = std::sqrt(centered_var + eps_);
                    } else {
                        denom = std::sqrt(square_avg_ptr[j] + eps_);
                    }

                    buf_ptr[j] = momentum_ * buf_ptr[j] + grad_ptr[j] / denom;
                    param_ptr[j] -= lr_ * buf_ptr[j];
                }
            } else {
                // Without momentum: theta_t = theta_{t-1} - lr * g_t / (sqrt(v_t) + eps)
                for (int64_t j = 0; j < numel; ++j) {
                    float denom;
                    if (centered_) {
                        float centered_var = square_avg_ptr[j] - avg_ptr[j] * avg_ptr[j];
                        denom = std::sqrt(centered_var + eps_);
                    } else {
                        denom = std::sqrt(square_avg_ptr[j] + eps_);
                    }

                    param_ptr[j] -= lr_ * grad_ptr[j] / denom;
                }
            }
        }

        // Copy updated values back to original device
        // Update param data by assigning new tensor
        param->tensor() = param_data.to(original_device);

        // Update state buffers
        square_avg_[i] = square_avg_cpu.to(original_device);
        if (centered_) {
            grad_avg_[i] = grad_avg_cpu.to(original_device);
        }
        if (momentum_ > 0.0) {
            momentum_buffer_[i] = momentum_buffer_cpu.to(original_device);
        }
    }
}

auto RMSprop::zero_grad() -> void {
    Optimizer::zero_grad();
}

auto RMSprop::set_lr(double lr) -> void {
    if (lr < 0.0) {
        throw std::invalid_argument("Learning rate must be non-negative");
    }
    lr_ = lr;
}

auto RMSprop::get_lr() const -> double {
    return lr_;
}

auto RMSprop::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        std::string prefix = "param_" + std::to_string(i);
        state[prefix + ".square_avg"] = square_avg_[i];

        if (centered_ && i < grad_avg_.size()) {
            state[prefix + ".grad_avg"] = grad_avg_[i];
        }

        if (momentum_ > 0.0 && i < momentum_buffer_.size()) {
            state[prefix + ".momentum_buffer"] = momentum_buffer_[i];
        }
    }

    return state;
}

auto RMSprop::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        std::string prefix = "param_" + std::to_string(i);

        auto it = state.find(prefix + ".square_avg");
        if (it != state.end()) {
            square_avg_[i] = it->second;
        }

        if (centered_) {
            it = state.find(prefix + ".grad_avg");
            if (it != state.end() && i < grad_avg_.size()) {
                grad_avg_[i] = it->second;
            }
        }

        if (momentum_ > 0.0) {
            it = state.find(prefix + ".momentum_buffer");
            if (it != state.end() && i < momentum_buffer_.size()) {
                momentum_buffer_[i] = it->second;
            }
        }
    }
}

} // namespace optim
} // namespace tenzor
