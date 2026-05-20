/**
 * @file rmsprop.cpp
 * @brief Implementation of RMSprop optimizer
 */

#include "tenzor/nn/optim/rmsprop.hpp"
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

auto RMSprop::step_impl() -> void {
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

            NewOpAttributes attrs;
            attrs.set(AttrKey::Lr, lr_);
            attrs.set(AttrKey::Alpha, alpha_);
            attrs.set(AttrKey::Eps, eps_);
            attrs.set(AttrKey::WeightDecay, weight_decay_);
            attrs.set(AttrKey::Momentum, momentum_);
            attrs.set(AttrKey::Centered, centered_);

            dispatch(OpId::FusedRMSPropStep, inputs, attrs);
            continue;
        }

        // CUDA fast path: fused kernel avoids GPU→CPU→GPU round-trip
        if (original_device.type == Device::Type::CUDA &&
            grad_orig.device().type == Device::Type::CUDA) {

            // CUDA registry expects: [param, grad, square_avg, grad_avg?, momentum_buffer?]
            std::vector<Tensor> inputs = {param->tensor(), grad_orig, square_avg_[i]};
            if (centered_ && i < grad_avg_.size()) {
                inputs.push_back(grad_avg_[i]);
            }
            if (momentum_ > 0.0 && i < momentum_buffer_.size()) {
                inputs.push_back(momentum_buffer_[i]);
            }

            NewOpAttributes attrs;
            attrs.set(AttrKey::Lr, lr_);
            attrs.set(AttrKey::Alpha, alpha_);
            attrs.set(AttrKey::Eps, eps_);
            attrs.set(AttrKey::WeightDecay, weight_decay_);
            attrs.set(AttrKey::Momentum, momentum_);
            attrs.set(AttrKey::Centered, centered_);

            dispatch(OpId::FusedRMSPropStep, inputs, attrs);
            continue;
        }

        // Generic fallback using tensor-level ops (device-agnostic)
        Tensor grad = grad_orig;
        Tensor param_data = param_data_orig;
        float alpha = static_cast<float>(alpha_);
        float eps = static_cast<float>(eps_);
        float lr = static_cast<float>(lr_);

        // Apply weight decay: g = g + weight_decay * param
        if (weight_decay_ > 0.0) {
            grad = grad + param_data * static_cast<float>(weight_decay_);
        }

        // Update square_avg: v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
        square_avg_[i] = square_avg_[i] * alpha + grad * grad * (1.0f - alpha);

        // Compute denominator
        Tensor denom;
        if (centered_) {
            // Update grad_avg: m_t = alpha * m_{t-1} + (1 - alpha) * g_t
            grad_avg_[i] = grad_avg_[i] * alpha + grad * (1.0f - alpha);
            // denom = sqrt(v - m^2 + eps)
            denom = sqrt(square_avg_[i] - grad_avg_[i] * grad_avg_[i] + eps);
        } else {
            denom = sqrt(square_avg_[i] + eps);
        }

        Tensor new_param;
        if (momentum_ > 0.0) {
            float mom = static_cast<float>(momentum_);
            // buf = momentum * buf + g / denom
            momentum_buffer_[i] = momentum_buffer_[i] * mom + grad / denom;
            // param -= lr * buf
            new_param = param_data - momentum_buffer_[i] * lr;
        } else {
            // param -= lr * g / denom
            new_param = param_data - grad / denom * lr;
        }

        // Copy result into existing tensor storage (preserves pointer stability on CPU)
        if (original_device.type == Device::Type::CPU) {
            auto src = new_param.contiguous();
            std::memcpy(param->tensor().data_ptr(), src.data_ptr(),
                        src.numel() * dtype_size(src.dtype()));
        } else {
            param->tensor() = new_param;
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

// Audit item D.5: state_dict / load_state_dict must persist the
// hyperparams alongside the buffers so an optimiser restarted from
// checkpoint resumes with the exact same configuration.  Previously
// RMSprop dropped lr/alpha/eps/weight_decay/momentum/centered.

namespace {

inline Tensor scalar_f64(double v) {
    Tensor t({1}, DType::Float64, Device::cpu());
    t.data<double>()[0] = v;
    return t;
}

inline Tensor scalar_i64(int64_t v) {
    Tensor t({1}, DType::Int64, Device::cpu());
    t.data<int64_t>()[0] = v;
    return t;
}

inline double get_scalar_f64(const std::unordered_map<std::string, Tensor>& m,
                             const std::string& key,
                             double fallback) {
    auto it = m.find(key);
    if (it == m.end()) return fallback;
    if (it->second.dtype() == DType::Float64) {
        return it->second.data<double>()[0];
    }
    if (it->second.dtype() == DType::Float32) {
        return static_cast<double>(it->second.data<float>()[0]);
    }
    return fallback;
}

inline int64_t get_scalar_i64(const std::unordered_map<std::string, Tensor>& m,
                              const std::string& key,
                              int64_t fallback) {
    auto it = m.find(key);
    if (it == m.end()) return fallback;
    if (it->second.dtype() == DType::Int64) {
        return it->second.data<int64_t>()[0];
    }
    if (it->second.dtype() == DType::Int32) {
        return static_cast<int64_t>(it->second.data<int32_t>()[0]);
    }
    return fallback;
}

}  // namespace

auto RMSprop::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    // Per-parameter buffers
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

    // Hyperparameters — required for full round-trip (D.5).
    state["lr"]            = scalar_f64(lr_);
    state["alpha"]         = scalar_f64(alpha_);
    state["eps"]           = scalar_f64(eps_);
    state["weight_decay"]  = scalar_f64(weight_decay_);
    state["momentum"]      = scalar_f64(momentum_);
    state["centered"]      = scalar_i64(centered_ ? 1 : 0);
    state["num_params"]    = scalar_i64(static_cast<int64_t>(parameters_.size()));

    return state;
}

auto RMSprop::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    // Buffer-count validation (matches Adam's pattern).
    int64_t expected_n = get_scalar_i64(state, "num_params",
                                        static_cast<int64_t>(parameters_.size()));
    if (expected_n != static_cast<int64_t>(parameters_.size())) {
        throw std::runtime_error(
            "RMSprop::load_state_dict: parameter count mismatch (state has " +
            std::to_string(expected_n) + ", optimiser has " +
            std::to_string(parameters_.size()) + ")");
    }

    // Restore hyperparameters first so derived flags (e.g. centered_) are
    // up-to-date when we read the per-parameter buffers.
    lr_           = get_scalar_f64(state, "lr",            lr_);
    alpha_        = get_scalar_f64(state, "alpha",         alpha_);
    eps_          = get_scalar_f64(state, "eps",           eps_);
    weight_decay_ = get_scalar_f64(state, "weight_decay",  weight_decay_);
    momentum_     = get_scalar_f64(state, "momentum",      momentum_);
    centered_     = (get_scalar_i64(state, "centered", centered_ ? 1 : 0) != 0);

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
