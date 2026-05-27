#include "tenzor/nn/optim/asgd.hpp"
#include "tenzor/nn/optim/master_weights.hpp"
#include "tenzor/ops/creation.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace tenzor::optim {

ASGD::ASGD(std::vector<std::shared_ptr<Variable>> params, double lr, double lambd,
            double alpha, double t0, double weight_decay)
    : Optimizer(std::move(params)), lr_(lr), lambd_(lambd),
      alpha_(alpha), t0_(t0), weight_decay_(weight_decay) {
    initialize_buffers();
}

ASGD::ASGD(std::vector<optim::ParamGroup> groups,
           double default_lr, double default_lambd,
           double default_alpha, double default_t0,
           double default_weight_decay)
    : Optimizer(std::move(groups)),
      lr_(default_lr),
      lambd_(default_lambd),
      alpha_(default_alpha),
      t0_(default_t0),
      weight_decay_(default_weight_decay) {
    initialize_buffers();
}

auto ASGD::step_impl() -> void {
    // Audit D.4: per-parameter hyperparameters resolve from the
    // active ParamGroup (when one was set up) or fall through to
    // the optimiser-wide defaults stored on this ASGD instance.
    // audit-11 RR.6: lambd and t0 now also resolve per-group via the
    // new ParamGroup::lambd / t0 optional overrides, mirroring NN.15's
    // Rprop pattern.
    struct ASGDHP {
        double lr;
        double lambd;
        double alpha;
        double t0;
        double weight_decay;
    };

    auto resolve = [&](size_t i) -> ASGDHP {
        ASGDHP hp{lr_, lambd_, alpha_, t0_, weight_decay_};
        if (const auto* g = find_group_for_param(i)) {
            hp.lr           = g->lr;
            hp.weight_decay = g->weight_decay;
            hp.alpha        = ParamGroup::or_else(g->alpha, alpha_);
            hp.lambd        = ParamGroup::or_else(g->lambd, lambd_);
            hp.t0           = ParamGroup::or_else(g->t0, t0_);
        }
        return hp;
    };

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        const ASGDHP hp = resolve(i);

        Tensor& param_tensor = param.tensor();
        const Tensor& grad_tensor = *param.grad();

        // R.16: half-precision params use Float32 ax buffer; run math in
        // state dtype and cast back on assignment.
        const DType param_dt = param_tensor.dtype();
        const DType state_dt = optim_state_dtype(param_dt);
        const bool needs_upcast = (state_dt != param_dt);

        // Use dtype-appropriate scalar tensors to preserve Float64 precision
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, state_dt, param_tensor.device());
        };

        Tensor param_hi = needs_upcast ? param_tensor.to(state_dt) : param_tensor;
        Tensor grad = needs_upcast ? grad_tensor.to(state_dt) : grad_tensor.clone();

        // Weight decay (L2 regularization applied to gradient)
        if (hp.weight_decay > 0.0) {
            grad = grad + param_hi * scalar(hp.weight_decay);
        }

        // Compute step-dependent learning rate: eta_t = lr / (1 + lambd * lr * t)^alpha
        double eta = hp.lr / std::pow(1.0 + hp.lambd * hp.lr * static_cast<double>(step_count_), hp.alpha);

        // SGD update with decayed learning rate
        param_hi = param_hi - grad * scalar(eta);
        param_tensor = needs_upcast ? param_hi.to(param_dt) : param_hi;

        // Update running average after t0 steps
        // mu_t = max(1, t - t0)
        // ax_t = (1 - 1/mu_t) * ax_{t-1} + (1/mu_t) * x_t
        double mu = std::max(1.0, static_cast<double>(step_count_) - hp.t0);
        double inv_mu = 1.0 / mu;
        ax_buffers_[i] = ax_buffers_[i] * scalar(1.0 - inv_mu) + param_hi * scalar(inv_mu);
    }

    step_count_++;
}

auto ASGD::set_lr(double lr) -> void {
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

auto ASGD::get_lr() const -> double {
    return lr_;
}

auto ASGD::initialize_buffers() -> void {
    ax_buffers_.clear();
    for (auto& param : parameters_) {
        if (param) {
            // R.16: half-precision params get Float32 ax buffer (upcast on
            // init so the running average doesn't erode in BF16).
            const auto& pt = param->tensor();
            const DType state_dt = optim_state_dtype(pt.dtype());
            ax_buffers_.push_back(state_dt == pt.dtype() ? pt.clone() : pt.to(state_dt));
        }
    }
}

// Audit K.1: extend ax_buffers_ for parameters appended via
// add_param_group.  Matches initialize_buffers — averaged params start
// as a clone of the freshly-added parameter value.
auto ASGD::on_parameters_appended_(size_t old_count, size_t new_count) -> void {
    ax_buffers_.reserve(new_count);
    for (size_t i = old_count; i < new_count; ++i) {
        const auto& param = parameters_[i];
        if (param) {
            // R.16: see ASGD::initialize_buffers for dtype rationale.
            const auto& pt = param->tensor();
            const DType state_dt = optim_state_dtype(pt.dtype());
            ax_buffers_.push_back(state_dt == pt.dtype() ? pt.clone() : pt.to(state_dt));
        } else {
            ax_buffers_.push_back(Tensor{});
        }
    }
}

auto ASGD::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    // Save optimizer configuration as tensors
    state["lr"] = Tensor({1}, DType::Float64, Device::cpu());
    state["lr"].data<double>()[0] = lr_;

    state["lambd"] = Tensor({1}, DType::Float64, Device::cpu());
    state["lambd"].data<double>()[0] = lambd_;

    state["alpha"] = Tensor({1}, DType::Float64, Device::cpu());
    state["alpha"].data<double>()[0] = alpha_;

    state["t0"] = Tensor({1}, DType::Float64, Device::cpu());
    state["t0"].data<double>()[0] = t0_;

    state["weight_decay"] = Tensor({1}, DType::Float64, Device::cpu());
    state["weight_decay"].data<double>()[0] = weight_decay_;

    state["step_count"] = Tensor({1}, DType::Int64, Device::cpu());
    state["step_count"].data<int64_t>()[0] = step_count_;

    // Save averaged parameter buffers
    for (size_t i = 0; i < ax_buffers_.size(); ++i) {
        state["ax_" + std::to_string(i)] = ax_buffers_[i].clone();
    }

    // S.12 / P.2: num_params guard so load_state_dict can reject mismatched
    // models before any tensor is overwritten (mirrors Adam::state_dict).
    state["num_params"] = Tensor({1}, DType::Int64, Device::cpu());
    state["num_params"].data<int64_t>()[0] = static_cast<int64_t>(parameters_.size());

    return state;
}

auto ASGD::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    // S.12 / P.2: param-count guard up front. Reject a restore against the
    // wrong model loudly instead of partially populating per-parameter
    // buffers and then succeeding silently.
    if (state.count("num_params")) {
        const int64_t expected = state.at("num_params").data<int64_t>()[0];
        if (expected != static_cast<int64_t>(parameters_.size())) {
            throw std::runtime_error(
                "ASGD::load_state_dict: parameter count mismatch - saved " +
                std::to_string(expected) + " but have " +
                std::to_string(parameters_.size()));
        }
    }

    // Load optimizer configuration
    if (state.count("lr")) {
        lr_ = state.at("lr").data<double>()[0];
    }

    if (state.count("lambd")) {
        lambd_ = state.at("lambd").data<double>()[0];
    }

    if (state.count("alpha")) {
        alpha_ = state.at("alpha").data<double>()[0];
    }

    if (state.count("t0")) {
        t0_ = state.at("t0").data<double>()[0];
    }

    if (state.count("weight_decay")) {
        weight_decay_ = state.at("weight_decay").data<double>()[0];
    }

    if (state.count("step_count")) {
        step_count_ = state.at("step_count").data<int64_t>()[0];
    }

    // V.27: cast to R.16 master-weights dtype on load.
    for (size_t i = 0; i < ax_buffers_.size(); ++i) {
        std::string ax_key = "ax_" + std::to_string(i);
        const DType state_dt = (i < parameters_.size() && parameters_[i])
            ? optim_state_dtype(parameters_[i]->tensor().dtype())
            : DType::Float32;
        if (state.count(ax_key)) {
            ax_buffers_[i] = state.at(ax_key).to(state_dt);
        }
    }
}

} // namespace tenzor::optim
