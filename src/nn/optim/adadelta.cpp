/**
 * @file adadelta.cpp
 * @brief Implementation of Adadelta optimizer
 */

#include "tenzor/nn/optim/adadelta.hpp"
#include "tenzor/nn/optim/master_weights.hpp"
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

Adadelta::Adadelta(std::vector<optim::ParamGroup> groups,
                   double default_lr, double default_rho,
                   double default_eps, double default_weight_decay)
    : Optimizer(std::move(groups)),
      lr_(default_lr),
      rho_(default_rho),
      eps_(default_eps),
      weight_decay_(default_weight_decay) {
    initialize_buffers();
}

auto Adadelta::initialize_buffers() -> void {
    square_avg_.clear();
    acc_delta_.clear();

    for (auto& param : parameters_) {
        if (!param) continue;
        const auto& param_data = param->tensor();

        // R.16: half-precision params get Float32 state buffers.
        square_avg_.push_back(make_optim_state(param_data));
        acc_delta_.push_back(make_optim_state(param_data));
    }
}

// Audit K.1: extend square_avg_ / acc_delta_ for parameters appended via
// add_param_group.  Both buffers are indexed by parameter position, so
// every new param needs a matching entry (Tensor{} placeholder for null).
auto Adadelta::on_parameters_appended_(size_t old_count, size_t new_count) -> void {
    square_avg_.reserve(new_count);
    acc_delta_.reserve(new_count);
    for (size_t i = old_count; i < new_count; ++i) {
        const auto& param = parameters_[i];
        if (param) {
            // R.16: half-precision params get Float32 state buffers.
            square_avg_.push_back(make_optim_state(param->tensor()));
            acc_delta_.push_back(make_optim_state(param->tensor()));
        } else {
            square_avg_.push_back(Tensor{});
            acc_delta_.push_back(Tensor{});
        }
    }
}

auto Adadelta::step_impl() -> void {
    // Audit D.4: per-parameter hyperparameters resolve from the
    // active ParamGroup (when one was set up) or fall through to
    // the optimiser-wide defaults stored on this Adadelta instance.
    struct AdadeltaHP {
        double lr;
        double rho;
        double eps;
        double weight_decay;
    };

    auto resolve = [&](size_t i) -> AdadeltaHP {
        AdadeltaHP hp{lr_, rho_, eps_, weight_decay_};
        if (const auto* g = find_group_for_param(i)) {
            hp.lr           = g->lr;
            hp.weight_decay = g->weight_decay;
            hp.rho          = ParamGroup::or_else(g->rho, rho_);
            hp.eps          = ParamGroup::or_else(g->eps, eps_);
        }
        return hp;
    };

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param = parameters_[i];

        if (!param || !param->has_grad()) {
            continue;  // Skip parameters without gradients
        }

        const AdadeltaHP hp = resolve(i);

        const auto& grad_orig = param->grad().value();
        const auto& param_data_orig = param->tensor();
        auto original_device = param_data_orig.device();
        const DType param_dt = param_data_orig.dtype();
        const DType state_dt = optim_state_dtype(param_dt);
        const bool needs_upcast = (state_dt != param_dt);

        // Vulkan fast path: fused kernel avoids GPU→CPU→GPU round-trip.
        // R.16: skip the fused path for half-precision params — state lives
        // at F32 while the fused kernel expects state at param dtype.
        if (!needs_upcast &&
            original_device.type == Device::Type::Vulkan &&
            grad_orig.device().type == Device::Type::Vulkan) {

            std::vector<Tensor> inputs = {
                param->tensor(), grad_orig, square_avg_[i], acc_delta_[i]
            };

            // Audit I.14: pass Float64 hyperparams via AttrKey
            // (was static_cast<float>, losing precision in Float64
            // training).  The Vulkan kernel reads via attrs.get_float
            // which returns double, so round-trip is double-clean.
            NewOpAttributes attrs;
            attrs.set(AttrKey::Lr, hp.lr);
            attrs.set(AttrKey::Rho, hp.rho);
            attrs.set(AttrKey::Eps, hp.eps);
            attrs.set(AttrKey::WeightDecay, hp.weight_decay);

            dispatch(OpId::FusedAdadeltaStep, inputs, attrs);
            continue;
        }

        // CUDA fast path: fused kernel avoids GPU→CPU→GPU round-trip.
        // R.16: skip the fused path for half-precision params.
        if (!needs_upcast &&
            original_device.type == Device::Type::CUDA &&
            grad_orig.device().type == Device::Type::CUDA) {

            // CUDA registry expects: [param, grad, square_avg, acc_delta]
            std::vector<Tensor> inputs = {
                param->tensor(), grad_orig, square_avg_[i], acc_delta_[i]
            };

            // Audit I.14: pass Float64 hyperparams via AttrKey.
            NewOpAttributes attrs;
            attrs.set(AttrKey::Rho, hp.rho);
            attrs.set(AttrKey::Eps, hp.eps);
            attrs.set(AttrKey::Lr, hp.lr);
            attrs.set(AttrKey::WeightDecay, hp.weight_decay);

            dispatch(OpId::FusedAdadeltaStep, inputs, attrs);
            continue;
        }

        // Generic fallback using tensor-level ops (device-agnostic).
        // R.16: run in state_dt (Float32 for half-precision params).
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, state_dt, param_data_orig.device());
        };
        Tensor param_data = needs_upcast ? param_data_orig.to(state_dt) : param_data_orig;
        Tensor grad = needs_upcast ? grad_orig.to(state_dt) : grad_orig;

        // Apply weight decay: g = g + weight_decay * param
        if (hp.weight_decay > 0.0) {
            grad = grad + param_data * scalar(hp.weight_decay);
        }

        // Accumulate squared gradient: v = rho * v + (1 - rho) * g^2
        square_avg_[i] = square_avg_[i] * scalar(hp.rho)
                       + grad * grad * scalar(1.0 - hp.rho);

        // Compute RMS of gradients and updates
        auto std_grad = sqrt(square_avg_[i] + scalar(hp.eps));
        auto std_delta = sqrt(acc_delta_[i] + scalar(hp.eps));

        // Compute parameter update: delta = -(std_delta / std_grad) * g
        auto delta = (std_delta / std_grad) * grad * scalar(-1.0);

        // Apply update: param += lr * delta
        auto new_param = param_data + delta * scalar(hp.lr);

        if (needs_upcast) {
            new_param = new_param.to(param_dt);
        }

        // Copy result into existing tensor storage (preserves pointer stability on CPU)
        if (original_device.type == Device::Type::CPU) {
            auto src = new_param.contiguous();
            std::memcpy(param->tensor().data_ptr(), src.data_ptr(),
                        src.numel() * dtype_size(src.dtype()));
        } else {
            param->tensor() = new_param;
        }

        // Accumulate squared update: acc = rho * acc + (1 - rho) * delta^2
        acc_delta_[i] = acc_delta_[i] * scalar(hp.rho)
                      + delta * delta * scalar(1.0 - hp.rho);
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

// Audit item D.5: persist hyperparams (lr / rho / eps / weight_decay)
// in addition to the per-parameter buffers.

namespace {

inline Tensor adadelta_scalar_f64(double v) {
    Tensor t({1}, DType::Float64, Device::cpu());
    t.data<double>()[0] = v;
    return t;
}

inline Tensor adadelta_scalar_i64(int64_t v) {
    Tensor t({1}, DType::Int64, Device::cpu());
    t.data<int64_t>()[0] = v;
    return t;
}

inline double adadelta_get_f64(const std::unordered_map<std::string, Tensor>& m,
                               const std::string& key, double fallback) {
    auto it = m.find(key);
    if (it == m.end()) return fallback;
    if (it->second.dtype() == DType::Float64) return it->second.data<double>()[0];
    if (it->second.dtype() == DType::Float32) return static_cast<double>(it->second.data<float>()[0]);
    return fallback;
}

inline int64_t adadelta_get_i64(const std::unordered_map<std::string, Tensor>& m,
                                const std::string& key, int64_t fallback) {
    auto it = m.find(key);
    if (it == m.end()) return fallback;
    if (it->second.dtype() == DType::Int64) return it->second.data<int64_t>()[0];
    return fallback;
}

}  // namespace

auto Adadelta::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        std::string prefix = "param_" + std::to_string(i);
        state[prefix + ".square_avg"] = square_avg_[i];
        state[prefix + ".acc_delta"] = acc_delta_[i];
    }

    state["lr"]           = adadelta_scalar_f64(lr_);
    state["rho"]          = adadelta_scalar_f64(rho_);
    state["eps"]          = adadelta_scalar_f64(eps_);
    state["weight_decay"] = adadelta_scalar_f64(weight_decay_);
    state["num_params"]   = adadelta_scalar_i64(
        static_cast<int64_t>(parameters_.size()));

    return state;
}

auto Adadelta::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    int64_t expected_n = adadelta_get_i64(state, "num_params",
                                          static_cast<int64_t>(parameters_.size()));
    if (expected_n != static_cast<int64_t>(parameters_.size())) {
        throw std::runtime_error(
            "Adadelta::load_state_dict: parameter count mismatch (state has " +
            std::to_string(expected_n) + ", optimiser has " +
            std::to_string(parameters_.size()) + ")");
    }

    lr_           = adadelta_get_f64(state, "lr",           lr_);
    rho_          = adadelta_get_f64(state, "rho",          rho_);
    eps_          = adadelta_get_f64(state, "eps",          eps_);
    weight_decay_ = adadelta_get_f64(state, "weight_decay", weight_decay_);

    // V.27: cast to R.16 master-weights dtype on load.
    for (size_t i = 0; i < parameters_.size(); ++i) {
        std::string prefix = "param_" + std::to_string(i);
        const DType state_dt = parameters_[i]
            ? optim_state_dtype(parameters_[i]->tensor().dtype())
            : DType::Float32;

        auto it = state.find(prefix + ".square_avg");
        if (it != state.end()) {
            square_avg_[i] = it->second.to(state_dt);
        }

        it = state.find(prefix + ".acc_delta");
        if (it != state.end()) {
            acc_delta_[i] = it->second.to(state_dt);
        }
    }
}

} // namespace optim
} // namespace tenzor
