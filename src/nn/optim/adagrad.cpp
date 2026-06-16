/**
 * @file adagrad.cpp
 * @brief Implementation of Adagrad optimizer
 */

#include "tenzor/nn/optim/adagrad.hpp"
#include "tenzor/nn/optim/master_weights.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/optim/sparse_helpers.hpp"
#include "tenzor/utils/logging.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace tenzor {
namespace optim {

namespace {
// S27 wiring: same fallback pattern as Adam — see adam.cpp for the
// rationale. Adagrad has no sparse-aware path yet, so we densify any
// producer-supplied sparse grad and the caller emits a one-shot warning.
inline auto adagrad_densify_sparse_grad(tenzor::Variable& param) -> bool {
    auto sparse_opt = tenzor::optim::detail::extract_sparse_grad(param);
    if (!sparse_opt.has_value()) return false;

    tenzor::Tensor dense = sparse_opt->to_dense();
    if (param.has_grad()) {
        const tenzor::Tensor& existing = param.grad().value();
        param.set_grad(existing + dense);
    } else {
        param.set_grad(std::move(dense));
    }
    param.clear_sparse_grad();
    return true;
}
}  // namespace

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

Adagrad::Adagrad(std::vector<optim::ParamGroup> groups,
                 double default_lr, double default_lr_decay,
                 double default_weight_decay,
                 double default_initial_accumulator_value, double default_eps)
    : Optimizer(std::move(groups)),
      lr_(default_lr),
      lr_decay_(default_lr_decay),
      weight_decay_(default_weight_decay),
      initial_accumulator_value_(default_initial_accumulator_value),
      eps_(default_eps),
      step_count_(0) {
    initialize_buffers();
}

auto Adagrad::initialize_buffers() -> void {
    sum_.clear();

    for (auto& param : parameters_) {
        if (!param) {
            // Keep sum_ index-aligned with parameters_ (matches
            // on_parameters_appended_); a null param preceding a valid one
            // must not shift later accumulators.
            sum_.push_back(Tensor{});
            continue;
        }
        const auto& param_data = param->tensor();
        auto original_device = param_data.device();
        // R.16: half-precision params get Float32 state.
        const DType state_dt = optim_state_dtype(param_data.dtype());

        // Initialize accumulator G_0
        if (initial_accumulator_value_ == 0.0) {
            sum_.push_back(make_optim_state(param_data));
        } else {
            // Create on CPU for data access, then move to target device.
            // Use the state dtype so half-precision params hold F32 accumulators.
            std::vector<int64_t> shape_vec(param_data.shape().begin(), param_data.shape().end());
            Tensor accumulator_cpu = zeros(shape_vec, state_dt, Device::cpu());
            int64_t numel = accumulator_cpu.numel();

            if (state_dt == DType::Float64) {
                double* acc_ptr = accumulator_cpu.data<double>();
                for (int64_t i = 0; i < numel; ++i) {
                    acc_ptr[i] = initial_accumulator_value_;
                }
            } else {
                // Float32 (covers F16/BF16 R.16 master state too).
                float* acc_ptr = accumulator_cpu.data<float>();
                for (int64_t i = 0; i < numel; ++i) {
                    acc_ptr[i] = static_cast<float>(initial_accumulator_value_);
                }
            }

            sum_.push_back(accumulator_cpu.to(original_device));
        }
    }
}

// Audit K.1: extend sum_ for parameters appended via add_param_group.
// When initial_accumulator_value_ != 0 the new slice is filled with that
// constant on the param's own device (matches initialize_buffers).
auto Adagrad::on_parameters_appended_(size_t old_count, size_t new_count) -> void {
    sum_.reserve(new_count);
    for (size_t i = old_count; i < new_count; ++i) {
        const auto& param = parameters_[i];
        if (!param) {
            sum_.push_back(Tensor{});
            continue;
        }
        const auto& param_data = param->tensor();
        const DType state_dt = optim_state_dtype(param_data.dtype());
        if (initial_accumulator_value_ == 0.0) {
            sum_.push_back(make_optim_state(param_data));
        } else {
            // R.16: half-precision params need Float32 accumulators.
            std::vector<int64_t> shape_vec(param_data.shape().begin(), param_data.shape().end());
            sum_.push_back(full(shape_vec, initial_accumulator_value_, state_dt, param_data.device()));
        }
    }
}

auto Adagrad::effective_lr() const -> double {
    // Must match step_impl()'s decay indexing exactly: the update applied on
    // step N (after step_count_ was bumped to N at the top of step_impl) uses
    // lr / (1 + (N-1) * lr_decay). Reporting step_count_*lr_decay here was an
    // off-by-one decay step versus the LR actually applied.
    if (lr_decay_ == 0.0 || step_count_ == 0) {
        return lr_;
    }
    return lr_ / (1.0 + static_cast<double>(step_count_ - 1) * lr_decay_);
}

auto Adagrad::step_impl() -> void {
    step_count_++;

    // Audit D.4: each parameter's hyperparameters resolve from its
    // ParamGroup override first, falling back to the optimiser-wide
    // defaults stored on this Adagrad instance. Effective LR honours
    // the per-group lr_decay independently.
    struct AdagradHP {
        double lr;
        double lr_decay;
        double weight_decay;
        double eps;
    };

    auto resolve = [&](size_t i) -> AdagradHP {
        AdagradHP hp{lr_, lr_decay_, weight_decay_, eps_};
        if (const auto* g = find_group_for_param(i)) {
            hp.lr           = g->lr;
            hp.weight_decay = g->weight_decay;
            hp.lr_decay     = ParamGroup::or_else(g->lr_decay, lr_decay_);
            hp.eps          = ParamGroup::or_else(g->eps, eps_);
        }
        // Adagrad effective lr: lr / (1 + (step-1) * lr_decay)
        hp.lr = hp.lr / (1.0 + static_cast<double>(step_count_ - 1) * hp.lr_decay);
        return hp;
    };

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param = parameters_[i];

        if (!param) continue;
        // S27: detect producer-supplied sparse grads and densify; emit a
        // one-shot warning. Adagrad has no sparse-aware fast path yet.
        if (adagrad_densify_sparse_grad(*param)) {
            TENZOR_WARN_ONCE(
                "Adagrad: parameter received a sparse gradient but this "
                "optimiser does not yet implement a sparse-aware update; "
                "densifying and running the standard dense step. Use "
                "SparseAdam for memory/perf-optimal sparse updates.");
        }
        if (!param->has_grad()) {
            continue;  // Skip parameters without gradients
        }

        const AdagradHP hp = resolve(i);

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

            // Input order MUST be [param, grad, sum_sq] to match the fused
            // kernel (binding 1 = param). Previously {grad, param, ...} made the
            // kernel write the update into the GRAD buffer, leaving param
            // unchanged — Adagrad was a silent no-op on Vulkan.
            std::vector<Tensor> inputs = {param->tensor(), grad_orig, sum_[i]};

            // Audit item I.14: pass Float64 hyperparams through to dispatch
            // (was static_cast<float>, losing precision vs Adam/AdamW which
            // already preserve double via AttrKey::Lr).
            NewOpAttributes attrs;
            attrs.set(AttrKey::Lr, hp.lr);
            attrs.set(AttrKey::Eps, hp.eps);
            attrs.set(AttrKey::WeightDecay, hp.weight_decay);

            dispatch(OpId::FusedAdagradStep, inputs, attrs);
            continue;
        }

        // CUDA fast path: fused kernel avoids GPU→CPU→GPU round-trip.
        // R.16: skip the fused path for half-precision params.
        if (!needs_upcast &&
            original_device.type == Device::Type::CUDA &&
            grad_orig.device().type == Device::Type::CUDA) {

            // CUDA registry expects: [param, grad, sum_sq]
            std::vector<Tensor> inputs = {param->tensor(), grad_orig, sum_[i]};

            // CUDA kernel receives the raw base lr + lr_decay + step and
            // computes its own effective lr; pass base lr_ for this group,
            // not hp.lr (which already has decay applied).
            const ParamGroup* g = find_group_for_param(i);
            double base_lr = g ? g->lr : lr_;

            NewOpAttributes attrs;
            attrs.set(AttrKey::Lr, base_lr);
            attrs.set(AttrKey::LrDecay, hp.lr_decay);
            attrs.set(AttrKey::Eps, hp.eps);
            attrs.set(AttrKey::WeightDecay, hp.weight_decay);
            attrs.set(AttrKey::Step, step_count_);

            dispatch(OpId::FusedAdagradStep, inputs, attrs);
            continue;
        }

        // Generic fallback using tensor-level ops (device-agnostic).
        // V.25 + R.16: use dtype-appropriate scalar tensors at state_dt
        // (Float32 for half-precision params) — the prior code force-cast
        // hyperparams via static_cast<float>, losing Float64 precision.
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, state_dt, param_data_orig.device());
        };
        Tensor param_data = needs_upcast ? param_data_orig.to(state_dt) : param_data_orig;
        Tensor grad = needs_upcast ? grad_orig.to(state_dt) : grad_orig;

        // Apply weight decay: g = g + weight_decay * param
        if (hp.weight_decay > 0.0) {
            grad = grad + param_data * scalar(hp.weight_decay);
        }

        // Update accumulator: G_t = G_{t-1} + g_t^2
        sum_[i] = sum_[i] + grad * grad;

        // Update parameters: theta = theta - lr * g / (sqrt(G) + eps)
        auto std_dev = sqrt(sum_[i]) + scalar(hp.eps);
        auto new_param = param_data - grad * scalar(hp.lr) / std_dev;

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

auto Adagrad::zero_grad(bool set_to_none) -> void {
    Optimizer::zero_grad(set_to_none);
}

auto Adagrad::set_lr(double lr) -> void {
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

auto Adagrad::get_lr() const -> double {
    // Return the BASE lr_ to match set_lr's unit and all sibling optimizers
    // (SGD/RMSprop/Adadelta/ASGD/Rprop/AdamAtan2). The time-decayed value is
    // available separately via effective_lr(). Returning the decayed value here
    // corrupts ReduceLROnPlateau/cyclic scheduler reductions when lr_decay>0.
    return lr_;
}

// Audit item D.5: persist hyperparams and use Int64 for step_count to
// avoid precision loss past 2^24 steps (previous code stored as
// Float32).

namespace {

inline Tensor adagrad_scalar_f64(double v) {
    Tensor t({1}, DType::Float64, Device::cpu());
    t.data<double>()[0] = v;
    return t;
}

inline Tensor adagrad_scalar_i64(int64_t v) {
    Tensor t({1}, DType::Int64, Device::cpu());
    t.data<int64_t>()[0] = v;
    return t;
}

inline double adagrad_get_f64(const std::unordered_map<std::string, Tensor>& m,
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

inline int64_t adagrad_get_i64(const std::unordered_map<std::string, Tensor>& m,
                               const std::string& key,
                               int64_t fallback) {
    auto it = m.find(key);
    if (it == m.end()) return fallback;
    if (it->second.dtype() == DType::Int64) {
        return it->second.data<int64_t>()[0];
    }
    if (it->second.dtype() == DType::Float32) {
        return static_cast<int64_t>(it->second.data<float>()[0]);
    }
    return fallback;
}

}  // namespace

auto Adagrad::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        std::string prefix = "param_" + std::to_string(i);
        state[prefix + ".sum"] = sum_[i];
    }

    // Int64 step count avoids the Float32 precision cliff at 2^24 steps.
    state["step_count"] = adagrad_scalar_i64(step_count_);

    // Hyperparameters.
    state["lr"]                          = adagrad_scalar_f64(lr_);
    state["lr_decay"]                    = adagrad_scalar_f64(lr_decay_);
    state["weight_decay"]                = adagrad_scalar_f64(weight_decay_);
    state["initial_accumulator_value"]   = adagrad_scalar_f64(initial_accumulator_value_);
    state["eps"]                         = adagrad_scalar_f64(eps_);
    state["num_params"]                  = adagrad_scalar_i64(
        static_cast<int64_t>(parameters_.size()));

    return state;
}

auto Adagrad::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    int64_t expected_n = adagrad_get_i64(state, "num_params",
                                         static_cast<int64_t>(parameters_.size()));
    if (expected_n != static_cast<int64_t>(parameters_.size())) {
        throw std::runtime_error(
            "Adagrad::load_state_dict: parameter count mismatch (state has " +
            std::to_string(expected_n) + ", optimiser has " +
            std::to_string(parameters_.size()) + ")");
    }

    lr_                        = adagrad_get_f64(state, "lr",                        lr_);
    lr_decay_                  = adagrad_get_f64(state, "lr_decay",                  lr_decay_);
    weight_decay_              = adagrad_get_f64(state, "weight_decay",              weight_decay_);
    initial_accumulator_value_ = adagrad_get_f64(state, "initial_accumulator_value", initial_accumulator_value_);
    eps_                       = adagrad_get_f64(state, "eps",                       eps_);

    // V.27: cast to R.16 master-weights dtype on load.
    for (size_t i = 0; i < parameters_.size(); ++i) {
        std::string prefix = "param_" + std::to_string(i);
        const DType state_dt = parameters_[i]
            ? optim_state_dtype(parameters_[i]->tensor().dtype())
            : DType::Float32;
        auto it = state.find(prefix + ".sum");
        if (it != state.end()) {
            sum_[i] = it->second.to(state_dt);
        }
    }

    step_count_ = adagrad_get_i64(state, "step_count", step_count_);
}

} // namespace optim
} // namespace tenzor
