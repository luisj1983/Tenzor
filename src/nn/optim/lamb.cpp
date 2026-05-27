#include "tenzor/nn/optim/lamb.hpp"
#include "tenzor/nn/optim/master_weights.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include <algorithm>
#include <cmath>

namespace tenzor::optim {

// R.16 master-weights helpers (optim_state_dtype / make_optim_state) are
// shared across all optimisers via include/tenzor/nn/optim/master_weights.hpp.

LAMB::LAMB(std::vector<std::shared_ptr<Variable>> params, double lr, double beta1,
           double beta2, double eps, double weight_decay,
           double min_norm, double max_norm)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), weight_decay_(weight_decay),
      min_norm_(min_norm), max_norm_(max_norm) {
    initialize_buffers();
}

LAMB::LAMB(std::vector<optim::ParamGroup> groups,
           double default_lr, double default_beta1, double default_beta2,
           double default_eps, double default_weight_decay,
           double default_min_norm, double default_max_norm)
    : Optimizer(std::move(groups)),
      lr_(default_lr),
      beta1_(default_beta1),
      beta2_(default_beta2),
      eps_(default_eps),
      weight_decay_(default_weight_decay),
      min_norm_(default_min_norm),
      max_norm_(default_max_norm) {
    initialize_buffers();
}

auto LAMB::step_impl() -> void {
    step_count_++;

    // Audit D.4: per-parameter hyperparameters resolve from the
    // active ParamGroup (when one was set up) or fall through to
    // the optimiser-wide defaults stored on this LAMB instance.
    struct LAMBHP {
        double lr;
        double beta1;
        double beta2;
        double eps;
        double weight_decay;
        double trust_min;
        double trust_max;
    };

    auto resolve = [&](size_t i) -> LAMBHP {
        LAMBHP hp{lr_, beta1_, beta2_, eps_, weight_decay_, min_norm_, max_norm_};
        if (const auto* g = find_group_for_param(i)) {
            hp.lr           = g->lr;
            hp.weight_decay = g->weight_decay;
            hp.beta1        = ParamGroup::or_else(g->beta1, beta1_);
            hp.beta2        = ParamGroup::or_else(g->beta2, beta2_);
            hp.eps          = ParamGroup::or_else(g->eps,   eps_);
            hp.trust_min    = ParamGroup::or_else(g->trust_ratio_min, min_norm_);
            hp.trust_max    = ParamGroup::or_else(g->trust_ratio_max, max_norm_);
        }
        return hp;
    };

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        const LAMBHP hp = resolve(i);

        const Tensor& grad = param.grad().value();

        // R.16: half-precision params run the math in state_dt (Float32).
        const DType param_dt = param.tensor().dtype();
        const DType state_dt = optim_state_dtype(param_dt);
        const bool needs_upcast = (state_dt != param_dt);
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, state_dt, param.tensor().device());
        };

        Tensor param_hi = needs_upcast ? param.tensor().to(state_dt) : param.tensor();
        Tensor grad_hi  = needs_upcast ? grad.to(state_dt) : grad;

        // Update moment estimates (no weight decay in moment computation)
        exp_avg_[i] = exp_avg_[i] * scalar(hp.beta1) +
                     grad_hi * scalar(1.0 - hp.beta1);
        exp_avg_sq_[i] = exp_avg_sq_[i] * scalar(hp.beta2) +
                        grad_hi * grad_hi * scalar(1.0 - hp.beta2);

        // Bias correction
        double bias_correction1 = 1.0 - std::pow(hp.beta1, step_count_);
        double bias_correction2 = 1.0 - std::pow(hp.beta2, step_count_);

        auto m_hat = exp_avg_[i] * scalar(1.0 / bias_correction1);
        auto v_hat = exp_avg_sq_[i] * scalar(1.0 / bias_correction2);

        // Adam update direction
        auto update = div(m_hat, sqrt(v_hat) + scalar(hp.eps));

        // Decoupled weight decay
        if (hp.weight_decay > 0.0) {
            update = update + param_hi * scalar(hp.weight_decay);
        }

        // Compute trust ratio (LAMB scaling)
        // param_norm = ||param||_2, update_norm = ||update||_2
        auto param_norm_t = sqrt(sum(param_hi * param_hi));
        auto update_norm_t = sqrt(sum(update * update));

        double param_norm = 0.0;
        double update_norm = 0.0;
        // Read scalar values (move to CPU and upcast to Float32 so the read
        // works uniformly for Float16 / BFloat16 parameters too — the reduced
        // norms are small tensors so the conversion cost is negligible).
        {
            auto pn = param_norm_t.to(Device::cpu()).to(DType::Float32);
            auto un = update_norm_t.to(Device::cpu()).to(DType::Float32);
            param_norm = static_cast<double>(pn.data<float>()[0]);
            update_norm = static_cast<double>(un.data<float>()[0]);
        }

        // QQ.12: clamp trust_ratio to [trust_min, trust_max].  Without this,
        // a near-zero update_norm produces trust_ratio -> +Inf and the
        // parameter step explodes.  PyTorch NVlamb defaults to [0, 10].
        double trust_ratio = 1.0;
        if (param_norm > 0.0 && update_norm > 0.0) {
            trust_ratio = param_norm / update_norm;
        }
        trust_ratio = std::clamp(trust_ratio, hp.trust_min, hp.trust_max);

        // Apply update with trust ratio
        Tensor updated = param_hi - update * scalar(hp.lr * trust_ratio);
        param.tensor() = needs_upcast ? updated.to(param_dt) : updated;
    }
}

auto LAMB::initialize_buffers() -> void {
    exp_avg_.clear();
    exp_avg_sq_.clear();
    for (auto& param : parameters_) {
        if (param) {
            // R.16: half-precision params get Float32 state buffers.
            exp_avg_.push_back(make_optim_state(param->tensor()));
            exp_avg_sq_.push_back(make_optim_state(param->tensor()));
        }
    }
}

// Audit K.1: extend exp_avg_ / exp_avg_sq_ for parameters appended via
// add_param_group.
auto LAMB::on_parameters_appended_(size_t old_count, size_t new_count) -> void {
    exp_avg_.reserve(new_count);
    exp_avg_sq_.reserve(new_count);
    for (size_t i = old_count; i < new_count; ++i) {
        const auto& param = parameters_[i];
        if (param) {
            // R.16: see LAMB::initialize_buffers.
            exp_avg_.push_back(make_optim_state(param->tensor()));
            exp_avg_sq_.push_back(make_optim_state(param->tensor()));
        } else {
            exp_avg_.push_back(Tensor{});
            exp_avg_sq_.push_back(Tensor{});
        }
    }
}

auto LAMB::set_lr(double lr) -> void {
    // HH.14: rescale every ParamGroup's lr by lr/old_lr (PyTorch convention).
    const double old_lr = lr_;
    lr_ = lr;
    if (old_lr == 0.0) {
        for (auto& g : param_groups_) g.lr = lr;
    } else {
        const double scale = lr / old_lr;
        for (auto& g : param_groups_) g.lr *= scale;
    }
}
auto LAMB::get_lr() const -> double { return lr_; }

// Audit item D.5: persist beta1 / beta2 / eps / weight_decay alongside
// lr / step / per-parameter exp_avg buffers.  Previously dropped.
auto LAMB::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;
    state["step_count"] = Tensor({1}, DType::Int64, Device::cpu());
    state["step_count"].data<int64_t>()[0] = step_count_;
    auto scalar_f64 = [](double v) {
        Tensor t({1}, DType::Float64, Device::cpu());
        t.data<double>()[0] = v;
        return t;
    };
    state["lr"]           = scalar_f64(lr_);
    state["beta1"]        = scalar_f64(beta1_);
    state["beta2"]        = scalar_f64(beta2_);
    state["eps"]          = scalar_f64(eps_);
    state["weight_decay"] = scalar_f64(weight_decay_);
    // QQ.12: persist trust-ratio clamp range.
    state["min_norm"]     = scalar_f64(min_norm_);
    state["max_norm"]     = scalar_f64(max_norm_);

    Tensor n({1}, DType::Int64, Device::cpu());
    n.data<int64_t>()[0] = static_cast<int64_t>(exp_avg_.size());
    state["num_params"] = n;

    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        state["exp_avg_" + std::to_string(i)] = exp_avg_[i].clone();
        state["exp_avg_sq_" + std::to_string(i)] = exp_avg_sq_[i].clone();
    }
    return state;
}

auto LAMB::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    // Parameter-count guard (matches Adam's protection against silently
    // mis-aligned per-parameter buffers).
    if (state.count("num_params")) {
        int64_t expected = state.at("num_params").data<int64_t>()[0];
        if (expected != static_cast<int64_t>(exp_avg_.size())) {
            throw std::runtime_error(
                "LAMB::load_state_dict: parameter count mismatch (state has " +
                std::to_string(expected) + ", optimiser has " +
                std::to_string(exp_avg_.size()) + ")");
        }
    }

    auto get_f64 = [&](const std::string& key, double fallback) {
        auto it = state.find(key);
        if (it == state.end()) return fallback;
        if (it->second.dtype() == DType::Float64) return it->second.data<double>()[0];
        if (it->second.dtype() == DType::Float32) return static_cast<double>(it->second.data<float>()[0]);
        return fallback;
    };

    if (state.count("step_count"))
        step_count_ = state.at("step_count").data<int64_t>()[0];
    lr_           = get_f64("lr",           lr_);
    beta1_        = get_f64("beta1",        beta1_);
    beta2_        = get_f64("beta2",        beta2_);
    eps_          = get_f64("eps",          eps_);
    weight_decay_ = get_f64("weight_decay", weight_decay_);
    // QQ.12: trust-ratio clamps (default to current values for backwards
    // compatibility with pre-QQ.12 checkpoints).
    min_norm_     = get_f64("min_norm",     min_norm_);
    max_norm_     = get_f64("max_norm",     max_norm_);

    // V.27 / Y.17: cast to the R.16 master-weights dtype on load. Pre-R.16
    // checkpoints stored half-precision buffers; restoring them as-is would
    // silently break the master-weights invariant on the next step.
    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        std::string ea_key = "exp_avg_" + std::to_string(i);
        std::string eas_key = "exp_avg_sq_" + std::to_string(i);
        const DType state_dt = (i < parameters_.size() && parameters_[i])
            ? optim_state_dtype(parameters_[i]->tensor().dtype())
            : DType::Float32;
        if (state.count(ea_key)) exp_avg_[i] = state.at(ea_key).to(state_dt);
        if (state.count(eas_key)) exp_avg_sq_[i] = state.at(eas_key).to(state_dt);
    }
}

} // namespace tenzor::optim
