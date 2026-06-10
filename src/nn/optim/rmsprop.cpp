/**
 * @file rmsprop.cpp
 * @brief Implementation of RMSprop optimizer
 */

#include "tenzor/nn/optim/rmsprop.hpp"
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

RMSprop::RMSprop(std::vector<optim::ParamGroup> groups,
                 double default_lr, double default_alpha, double default_eps,
                 double default_weight_decay, double default_momentum,
                 bool default_centered)
    : Optimizer(std::move(groups)),
      lr_(default_lr),
      alpha_(default_alpha),
      eps_(default_eps),
      weight_decay_(default_weight_decay),
      momentum_(default_momentum),
      centered_(default_centered) {
    initialize_buffers();
}

auto RMSprop::initialize_buffers() -> void {
    square_avg_.clear();
    grad_avg_.clear();
    momentum_buffer_.clear();

    // KK.15: per-group resolution.  Any param whose ParamGroup requests
    // centered=true (or momentum>0) must have a buffer allocated, even
    // when the optimiser-wide flag is false; otherwise step_impl()'s
    // `i < grad_avg_.size()` / `i < momentum_buffer_.size()` guards
    // silently fall back to vanilla RMSProp.
    square_avg_.reserve(parameters_.size());
    grad_avg_.reserve(parameters_.size());
    momentum_buffer_.reserve(parameters_.size());

    for (size_t i = 0; i < parameters_.size(); ++i) {
        const auto& param = parameters_[i];
        if (!param) {
            square_avg_.push_back(Tensor{});
            grad_avg_.push_back(Tensor{});
            momentum_buffer_.push_back(Tensor{});
            continue;
        }
        const auto& param_data = param->tensor();

        // R.16: half-precision params get Float32 state buffers.
        square_avg_.push_back(make_optim_state(param_data));

        // KK.15: resolve centered/momentum per-group with optimizer-wide
        // fallback, matching step_impl()'s resolve() lambda.
        const auto* g = find_group_for_param(i);
        const bool   want_centered = g ? ParamGroup::or_else(g->centered, centered_) : centered_;
        const double want_momentum = g ? ParamGroup::or_else(g->momentum, momentum_) : momentum_;

        if (want_centered) {
            grad_avg_.push_back(make_optim_state(param_data));
        } else {
            grad_avg_.push_back(Tensor{});
        }

        if (want_momentum > 0.0) {
            momentum_buffer_.push_back(make_optim_state(param_data));
        } else {
            momentum_buffer_.push_back(Tensor{});
        }
    }
}

// Audit K.1: extend square_avg_ (and grad_avg_ / momentum_buffer_ when
// the resolved per-group centered / momentum flags request them) for
// parameters appended via add_param_group.  Mirrors initialize_buffers
// so the per-parameter indexing in step_impl() stays valid.
auto RMSprop::on_parameters_appended_(size_t old_count, size_t new_count) -> void {
    square_avg_.reserve(new_count);
    grad_avg_.reserve(new_count);
    momentum_buffer_.reserve(new_count);

    for (size_t i = old_count; i < new_count; ++i) {
        const auto& param = parameters_[i];
        if (param) {
            const auto& param_data = param->tensor();
            // R.16: see RMSprop::initialize_buffers for dtype rationale.
            square_avg_.push_back(make_optim_state(param_data));

            // KK.15: honour the per-group centered/momentum settings of
            // the freshly-appended param.  Without this, a group with
            // `centered=true` on an optimiser whose default `centered_`
            // is false would have `i >= grad_avg_.size()` and silently
            // run uncentered.
            const auto* g = find_group_for_param(i);
            const bool   want_centered = g ? ParamGroup::or_else(g->centered, centered_) : centered_;
            const double want_momentum = g ? ParamGroup::or_else(g->momentum, momentum_) : momentum_;

            if (want_centered) {
                grad_avg_.push_back(make_optim_state(param_data));
            } else {
                grad_avg_.push_back(Tensor{});
            }

            if (want_momentum > 0.0) {
                momentum_buffer_.push_back(make_optim_state(param_data));
            } else {
                momentum_buffer_.push_back(Tensor{});
            }
        } else {
            square_avg_.push_back(Tensor{});
            grad_avg_.push_back(Tensor{});
            momentum_buffer_.push_back(Tensor{});
        }
    }
}

auto RMSprop::step_impl() -> void {
    // Audit D.4: resolve hyperparams per-param from the active ParamGroup
    // with optimizer-member fallback.  Flat-param constructor →
    // find_group_for_param returns nullptr → uses optimizer defaults.
    struct HP {
        double lr, alpha, eps, weight_decay, momentum;
        bool   centered;
    };
    auto resolve = [this](size_t i) -> HP {
        HP hp{lr_, alpha_, eps_, weight_decay_, momentum_, centered_};
        const auto* g = find_group_for_param(i);
        if (g) {
            hp.lr           = g->lr;
            hp.weight_decay = g->weight_decay;
            hp.alpha        = ParamGroup::or_else(g->alpha,    alpha_);
            hp.eps          = ParamGroup::or_else(g->eps,      eps_);
            hp.momentum     = ParamGroup::or_else(g->momentum, momentum_);
            hp.centered     = ParamGroup::or_else(g->centered, centered_);
        }
        return hp;
    };

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param = parameters_[i];

        if (!param || !param->has_grad()) {
            continue;  // Skip parameters without gradients
        }

        HP hp = resolve(i);

        const auto& grad_orig = param->grad().value();
        const auto& param_data_orig = param->tensor();
        auto original_device = param_data_orig.device();
        const DType param_dt = param_data_orig.dtype();
        const DType state_dt = optim_state_dtype(param_dt);
        const bool needs_upcast = (state_dt != param_dt);

        // Vulkan fast path: fused kernel avoids GPU→CPU→GPU round-trip.
        // R.16: skip the fused path for half-precision params — state lives
        // at F32, the fused kernel expects state at param dtype.
        if (!needs_upcast &&
            original_device.type == Device::Type::Vulkan &&
            grad_orig.device().type == Device::Type::Vulkan) {

            // Input order MUST match the kernel contract [param, grad,
            // square_avg, grad_avg?(centered), momentum?(momentum)] — the same
            // order the CUDA path uses below. Previously this passed
            // {grad, param, ...}, so the fused kernel (binding 1 = param) wrote
            // the update into the GRAD buffer and left param unchanged (the
            // RMSprop step was a silent no-op on Vulkan).
            std::vector<Tensor> inputs = {param->tensor(), grad_orig, square_avg_[i]};
            // The Vulkan fused kernel reads grad_avg from the FIXED slot
            // inputs[3] and momentum_buffer from the FIXED slot inputs[4]. When
            // momentum is active but centered is not, slot 3 must still be
            // present (placeholder) so momentum lands at slot 4.
            const bool want_momentum = (hp.momentum > 0.0 && i < momentum_buffer_.size());
            const bool want_gradavg  = (hp.centered && i < grad_avg_.size());
            if (want_gradavg || want_momentum) {
                inputs.push_back(want_gradavg ? grad_avg_[i] : square_avg_[i]);  // slot 3 = grad_avg
            }
            if (want_momentum) {
                inputs.push_back(momentum_buffer_[i]);                           // slot 4 = momentum
            }

            NewOpAttributes attrs;
            attrs.set(AttrKey::Lr, hp.lr);
            attrs.set(AttrKey::Alpha, hp.alpha);
            attrs.set(AttrKey::Eps, hp.eps);
            attrs.set(AttrKey::WeightDecay, hp.weight_decay);
            attrs.set(AttrKey::Momentum, hp.momentum);
            attrs.set(AttrKey::Centered, hp.centered);

            dispatch(OpId::FusedRMSPropStep, inputs, attrs);
            continue;
        }

        // CUDA fast path: fused kernel avoids GPU→CPU→GPU round-trip.
        // R.16: skip the fused path for half-precision params (see above).
        if (!needs_upcast &&
            original_device.type == Device::Type::CUDA &&
            grad_orig.device().type == Device::Type::CUDA) {

            // CUDA registry expects: [param, grad, square_avg, grad_avg?, momentum_buffer?]
            std::vector<Tensor> inputs = {param->tensor(), grad_orig, square_avg_[i]};
            if (hp.centered && i < grad_avg_.size()) {
                inputs.push_back(grad_avg_[i]);
            }
            if (hp.momentum > 0.0 && i < momentum_buffer_.size()) {
                inputs.push_back(momentum_buffer_[i]);
            }

            NewOpAttributes attrs;
            attrs.set(AttrKey::Lr, hp.lr);
            attrs.set(AttrKey::Alpha, hp.alpha);
            attrs.set(AttrKey::Eps, hp.eps);
            attrs.set(AttrKey::WeightDecay, hp.weight_decay);
            attrs.set(AttrKey::Momentum, hp.momentum);
            attrs.set(AttrKey::Centered, hp.centered);

            dispatch(OpId::FusedRMSPropStep, inputs, attrs);
            continue;
        }

        // Generic fallback using tensor-level ops (device-agnostic).
        // R.16: arithmetic runs in state_dt (Float32 for half-precision
        // params); param/grad upcast on entry, result downcast on write.
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, state_dt, param_data_orig.device());
        };
        Tensor param_data = needs_upcast ? param_data_orig.to(state_dt) : param_data_orig;
        Tensor grad = needs_upcast ? grad_orig.to(state_dt) : grad_orig;

        // Apply weight decay: g = g + weight_decay * param
        if (hp.weight_decay > 0.0) {
            grad = grad + param_data * scalar(hp.weight_decay);
        }

        // Update square_avg: v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
        square_avg_[i] = square_avg_[i] * scalar(hp.alpha)
                       + grad * grad * scalar(1.0 - hp.alpha);

        // Compute denominator
        Tensor denom;
        if (hp.centered) {
            // Update grad_avg: m_t = alpha * m_{t-1} + (1 - alpha) * g_t
            grad_avg_[i] = grad_avg_[i] * scalar(hp.alpha)
                         + grad * scalar(1.0 - hp.alpha);
            // denom = sqrt(v - m^2 + eps)
            denom = sqrt(square_avg_[i] - grad_avg_[i] * grad_avg_[i] + scalar(hp.eps));
        } else {
            denom = sqrt(square_avg_[i] + scalar(hp.eps));
        }

        Tensor new_param;
        if (hp.momentum > 0.0) {
            // buf = momentum * buf + g / denom
            momentum_buffer_[i] = momentum_buffer_[i] * scalar(hp.momentum)
                                + grad / denom;
            // param -= lr * buf
            new_param = param_data - momentum_buffer_[i] * scalar(hp.lr);
        } else {
            // param -= lr * g / denom
            new_param = param_data - grad / denom * scalar(hp.lr);
        }

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
    }
}

auto RMSprop::zero_grad(bool set_to_none) -> void {
    Optimizer::zero_grad(set_to_none);
}

auto RMSprop::set_lr(double lr) -> void {
    if (lr < 0.0) {
        throw std::invalid_argument("Learning rate must be non-negative");
    }
    // HH.14: rescale every ParamGroup's lr by lr/old_lr.
    const double old_lr = lr_;
    lr_ = lr;
    if (old_lr == 0.0) {
        for (auto& g : param_groups_) g.lr = lr;
    } else {
        const double scale = lr / old_lr;
        for (auto& g : param_groups_) g.lr *= scale;
    }
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
        // V.27: cast to R.16 master-weights dtype on load so half-precision
        // params restore F32 state, not the checkpoint's possibly-F16 dtype.
        const DType state_dt = parameters_[i]
            ? optim_state_dtype(parameters_[i]->tensor().dtype())
            : DType::Float32;

        auto it = state.find(prefix + ".square_avg");
        if (it != state.end()) {
            square_avg_[i] = it->second.to(state_dt);
        }

        if (centered_) {
            it = state.find(prefix + ".grad_avg");
            if (it != state.end() && i < grad_avg_.size()) {
                grad_avg_[i] = it->second.to(state_dt);
            }
        }

        if (momentum_ > 0.0) {
            it = state.find(prefix + ".momentum_buffer");
            if (it != state.end() && i < momentum_buffer_.size()) {
                momentum_buffer_[i] = it->second.to(state_dt);
            }
        }
    }
}

} // namespace optim
} // namespace tenzor
