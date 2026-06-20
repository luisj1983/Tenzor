#include "tenzor/nn/optim/adamax.hpp"
#include "tenzor/nn/optim/master_weights.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cmath>

namespace tenzor::optim {

Adamax::Adamax(std::vector<std::shared_ptr<Variable>> params, double lr, double beta1,
               double beta2, double eps, double weight_decay)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), weight_decay_(weight_decay) {
    initialize_buffers();
}

Adamax::Adamax(std::vector<optim::ParamGroup> groups,
               double default_lr, double default_beta1, double default_beta2,
               double default_eps, double default_weight_decay)
    : Optimizer(std::move(groups)),
      lr_(default_lr),
      beta1_(default_beta1),
      beta2_(default_beta2),
      eps_(default_eps),
      weight_decay_(default_weight_decay) {
    initialize_buffers();
}

auto Adamax::step_impl() -> void {
    // Audit D.4: per-parameter hyperparameters resolve from the
    // active ParamGroup (when one was set up) or fall through to
    // the optimiser-wide defaults stored on this Adamax instance.
    struct AdamaxHP {
        double lr;
        double beta1;
        double beta2;
        double eps;
        double weight_decay;
    };

    auto resolve = [&](size_t i) -> AdamaxHP {
        AdamaxHP hp{lr_, beta1_, beta2_, eps_, weight_decay_};
        if (const auto* g = find_group_for_param(i)) {
            hp.lr           = g->lr;
            hp.weight_decay = g->weight_decay;
            hp.beta1        = ParamGroup::or_else(g->beta1, beta1_);
            hp.beta2        = ParamGroup::or_else(g->beta2, beta2_);
            hp.eps          = ParamGroup::or_else(g->eps,   eps_);
        }
        return hp;
    };

    step_count_++;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        const AdamaxHP hp = resolve(i);

        const double bias_correction1 =
            1.0 - std::pow(hp.beta1, static_cast<double>(step_count_));
        const double step_size = hp.lr / bias_correction1;

        const Tensor& grad = param.grad().value();

        // R.16: run math in the optimiser state dtype (Float32 for halves)
        // and cast back to the param dtype on assignment.
        const DType param_dt = param.tensor().dtype();
        const DType state_dt = optim_state_dtype(param_dt);
        const bool needs_upcast = (state_dt != param_dt);

        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, state_dt, param.tensor().device());
        };

        Tensor param_hi = needs_upcast ? param.tensor().to(state_dt) : param.tensor();
        Tensor grad_copy = needs_upcast ? grad.to(state_dt) : grad.clone();

        if (hp.weight_decay > 0.0) {
            grad_copy = grad_copy + param_hi * scalar(hp.weight_decay);
        }

        // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        exp_avg_[i] = exp_avg_[i] * scalar(hp.beta1) +
                      grad_copy    * scalar(1.0 - hp.beta1);

        // u_t = max(beta2 * u_{t-1}, |g_t|)
        // element-wise via maximum(), not a reduction.
        exp_inf_[i] = maximum(exp_inf_[i] * scalar(hp.beta2), abs(grad_copy));

        // denom = u_t + eps
        auto denom = exp_inf_[i] + scalar(hp.eps);

        // theta -= (lr / (1 - beta1^t)) * m_t / denom
        Tensor updated = param_hi -
                         div(exp_avg_[i], denom) * scalar(step_size);
        param.tensor() = needs_upcast ? updated.to(param_dt) : updated;
    }
}

auto Adamax::initialize_buffers() -> void {
    exp_avg_.clear();
    exp_inf_.clear();

    for (auto& param : parameters_) {
        if (param) {
            // R.16: half-precision params get Float32 state buffers.
            exp_avg_.push_back(make_optim_state(param->tensor()));
            exp_inf_.push_back(make_optim_state(param->tensor()));
        } else {
            // Keep state vectors index-aligned with parameters_ so that
            // step_impl()'s positional indexing never skews after a null
            // param (mirrors on_parameters_appended_).
            exp_avg_.push_back(Tensor{});
            exp_inf_.push_back(Tensor{});
        }
    }
}

// Audit K.1: extend exp_avg_ / exp_inf_ for parameters appended via
// add_param_group.  step_count_ is optimiser-wide and untouched.
auto Adamax::on_parameters_appended_(size_t old_count, size_t new_count) -> void {
    exp_avg_.reserve(new_count);
    exp_inf_.reserve(new_count);
    for (size_t i = old_count; i < new_count; ++i) {
        const auto& param = parameters_[i];
        if (param) {
            // R.16: see Adamax::initialize_buffers for dtype rationale.
            exp_avg_.push_back(make_optim_state(param->tensor()));
            exp_inf_.push_back(make_optim_state(param->tensor()));
        } else {
            exp_avg_.push_back(Tensor{});
            exp_inf_.push_back(Tensor{});
        }
    }
}

auto Adamax::set_lr(double lr) -> void {
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
auto Adamax::get_lr() const -> double { return lr_; }

auto Adamax::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    state["step_count"] = Tensor({1}, DType::Int64, Device::cpu());
    state["step_count"].data<int64_t>()[0] = step_count_;

    auto put_double = [&](const char* key, double value) {
        state[key] = Tensor({1}, DType::Float64, Device::cpu());
        state[key].data<double>()[0] = value;
    };
    put_double("lr", lr_);
    put_double("beta1", beta1_);
    put_double("beta2", beta2_);
    put_double("eps", eps_);
    put_double("weight_decay", weight_decay_);

    // Z.12: persist parameter count for fail-fast load (mirrors Adam / LAMB).
    Tensor n({1}, DType::Int64, Device::cpu());
    n.data<int64_t>()[0] = static_cast<int64_t>(exp_avg_.size());
    state["num_params"] = n;

    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        state["exp_avg_" + std::to_string(i)] = exp_avg_[i].clone();
        state["exp_inf_" + std::to_string(i)] = exp_inf_[i].clone();
    }
    return state;
}

auto Adamax::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    auto load_double = [&](const char* key, double& out) {
        auto it = state.find(key);
        if (it != state.end()) out = it->second.data<double>()[0];
    };

    // Z.12: explicit num_params guard mirroring Adam / LAMB / SparseAdam.
    if (state.count("num_params")) {
        int64_t expected = state.at("num_params").data<int64_t>()[0];
        if (expected != static_cast<int64_t>(exp_avg_.size())) {
            throw std::runtime_error(
                "Adamax::load_state_dict: parameter count mismatch (state has " +
                std::to_string(expected) + ", optimiser has " +
                std::to_string(exp_avg_.size()) + ")");
        }
    }

    if (state.count("step_count")) {
        step_count_ = state.at("step_count").data<int64_t>()[0];
    }
    load_double("lr", lr_);
    load_double("beta1", beta1_);
    load_double("beta2", beta2_);
    load_double("eps", eps_);
    load_double("weight_decay", weight_decay_);

    size_t saved_count = 0;
    for (const auto& [key, _] : state) {
        if (key.starts_with("exp_avg_")) {
            ++saved_count;
        }
    }
    if (saved_count > 0 && saved_count != exp_avg_.size()) {
        throw std::runtime_error(
            "Adamax::load_state_dict: momentum buffer count mismatch - "
            "saved " + std::to_string(saved_count) + " but have " +
            std::to_string(exp_avg_.size()) + " parameters");
    }

    // V.27: cast to R.16 master-weights dtype on load.
    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        std::string exp_avg_key = "exp_avg_" + std::to_string(i);
        std::string exp_inf_key = "exp_inf_" + std::to_string(i);
        const DType state_dt = (i < parameters_.size() && parameters_[i])
            ? optim_state_dtype(parameters_[i]->tensor().dtype())
            : DType::Float32;
        if (state.count(exp_avg_key)) exp_avg_[i] = state.at(exp_avg_key).to(state_dt);
        if (state.count(exp_inf_key)) exp_inf_[i] = state.at(exp_inf_key).to(state_dt);
    }
}

} // namespace tenzor::optim
