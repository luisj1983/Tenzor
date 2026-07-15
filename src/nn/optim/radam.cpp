#include "tenzor/nn/optim/radam.hpp"
#include "tenzor/nn/optim/master_weights.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cmath>

namespace tenzor::optim {

// R.16 master-weights helpers come from include/tenzor/nn/optim/master_weights.hpp.

RAdam::RAdam(std::vector<std::shared_ptr<Variable>> params, double lr, double beta1,
             double beta2, double eps, double weight_decay)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), weight_decay_(weight_decay) {
    initialize_buffers();
}

RAdam::RAdam(std::vector<optim::ParamGroup> groups,
             double default_lr, double default_beta1,
             double default_beta2, double default_eps,
             double default_weight_decay)
    : Optimizer(std::move(groups)),
      lr_(default_lr),
      beta1_(default_beta1),
      beta2_(default_beta2),
      eps_(default_eps),
      weight_decay_(default_weight_decay) {
    initialize_buffers();
}

auto RAdam::step_impl() -> void {
    // Audit D.4: per-parameter hyperparameters resolve from the
    // active ParamGroup (when one was set up) or fall through to
    // the optimiser-wide defaults stored on this RAdam instance.
    // step_count_ stays a single optimiser-wide counter (matches
    // PyTorch's per-optimizer step semantics).
    struct RAdamHP {
        double lr;
        double beta1;
        double beta2;
        double eps;
        double weight_decay;
    };

    auto resolve = [this](size_t i) -> RAdamHP {
        RAdamHP hp{lr_, beta1_, beta2_, eps_, weight_decay_};
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

        const RAdamHP hp = resolve(i);
        const double rho_inf = 2.0 / (1.0 - hp.beta2) - 1.0;

        // Dangling-reference fix: grad() returns optional<Tensor> by value; bind by value, not reference.
        const Tensor grad = param.grad().value();

        // R.16: half-precision params run the math in the state dtype.
        const DType param_dt = param.tensor().dtype();
        const DType state_dt = optim_state_dtype(param_dt);
        const bool needs_upcast = (state_dt != param_dt);

        // CPU fallback path — use dtype-appropriate scalar tensors to
        // preserve Float64 precision (static_cast<float> truncates doubles)
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, state_dt, param.tensor().device());
        };

        Tensor param_hi  = needs_upcast ? param.tensor().to(state_dt) : param.tensor();
        Tensor grad_copy = needs_upcast ? grad.to(state_dt) : grad.clone();

        // Weight decay (L2 regularization)
        if (hp.weight_decay > 0.0) {
            grad_copy = grad_copy + param_hi * scalar(hp.weight_decay);
        }

        // Update biased first moment estimate
        exp_avg_[i] = exp_avg_[i] * scalar(hp.beta1) +
                     grad_copy * scalar(1.0 - hp.beta1);

        // Update biased second raw moment estimate
        exp_avg_sq_[i] = exp_avg_sq_[i] * scalar(hp.beta2) +
                        grad_copy * grad_copy * scalar(1.0 - hp.beta2);

        // Bias corrections
        double bias_correction1 = 1.0 - std::pow(hp.beta1, step_count_);
        double beta2_t = std::pow(hp.beta2, step_count_);
        double rho_t = rho_inf - 2.0 * step_count_ * beta2_t / (1.0 - beta2_t);

        Tensor updated;
        if (rho_t > 5.0) {
            // Variance is tractable — use rectified Adam update
            double bias_correction2 = 1.0 - beta2_t;
            double rect = std::sqrt(
                (rho_t - 4.0) * (rho_t - 2.0) * rho_inf /
                ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
            );
            double step_size = hp.lr * rect / bias_correction1;

            // PyTorch adds eps to sqrt(v) BEFORE dividing by sqrt(bias_correction2):
            //   adaptive_lr = sqrt(bias_correction2) / (sqrt(v) + eps)
            // i.e. denom = (sqrt(v) + eps) / sqrt(bias_correction2). Folding eps
            // inside the sqrt keeps parity with the reference (eps effectively
            // scaled by 1/sqrt(bias_correction2)).
            auto denom = (sqrt(exp_avg_sq_[i]) + scalar(hp.eps))
                        * scalar(1.0 / std::sqrt(bias_correction2));
            updated = param_hi - div(exp_avg_[i], denom) * scalar(step_size);
        } else {
            // Variance is intractable — use SGD with momentum
            double step_size = hp.lr / bias_correction1;
            updated = param_hi - exp_avg_[i] * scalar(step_size);
        }
        param.tensor() = needs_upcast ? updated.to(param_dt) : updated;
    }
}

auto RAdam::initialize_buffers() -> void {
    exp_avg_.clear();
    exp_avg_sq_.clear();

    for (auto& param : parameters_) {
        if (param) {
            // R.16: half-precision params get Float32 state buffers.
            exp_avg_.push_back(make_optim_state(param->tensor()));
            exp_avg_sq_.push_back(make_optim_state(param->tensor()));
        } else {
            // A null param preceding a valid one must not shift later buffers:
            // keep exp_avg_/exp_avg_sq_ index-aligned with parameters_ so
            // step_impl()'s exp_avg_[i]/exp_avg_sq_[i] lookups stay correct.
            exp_avg_.push_back(Tensor{});
            exp_avg_sq_.push_back(Tensor{});
        }
    }
}

// Audit K.1: extend exp_avg_ / exp_avg_sq_ for parameters appended via
// add_param_group.  step_count_ stays optimiser-wide.
auto RAdam::on_parameters_appended_(size_t old_count, size_t new_count) -> void {
    exp_avg_.reserve(new_count);
    exp_avg_sq_.reserve(new_count);
    for (size_t i = old_count; i < new_count; ++i) {
        const auto& param = parameters_[i];
        if (param) {
            // R.16: see RAdam::initialize_buffers.
            exp_avg_.push_back(make_optim_state(param->tensor()));
            exp_avg_sq_.push_back(make_optim_state(param->tensor()));
        } else {
            exp_avg_.push_back(Tensor{});
            exp_avg_sq_.push_back(Tensor{});
        }
    }
}

auto RAdam::set_lr(double lr) -> void {
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

auto RAdam::get_lr() const -> double {
    return lr_;
}

auto RAdam::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    // Save optimizer configuration as tensors
    state["step_count"] = Tensor({1}, DType::Int64, Device::cpu());
    state["step_count"].data<int64_t>()[0] = step_count_;

    state["lr"] = Tensor({1}, DType::Float64, Device::cpu());
    state["lr"].data<double>()[0] = lr_;

    state["beta1"] = Tensor({1}, DType::Float64, Device::cpu());
    state["beta1"].data<double>()[0] = beta1_;

    state["beta2"] = Tensor({1}, DType::Float64, Device::cpu());
    state["beta2"].data<double>()[0] = beta2_;

    state["eps"] = Tensor({1}, DType::Float64, Device::cpu());
    state["eps"].data<double>()[0] = eps_;

    state["weight_decay"] = Tensor({1}, DType::Float64, Device::cpu());
    state["weight_decay"].data<double>()[0] = weight_decay_;

    // Save momentum buffers
    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        state["exp_avg_" + std::to_string(i)] = exp_avg_[i].clone();
        state["exp_avg_sq_" + std::to_string(i)] = exp_avg_sq_[i].clone();
    }

    // Audit-4 W.14: num_params guard mirrors Adam::state_dict so a restore
    // against a mismatched model count fails loudly before any per-parameter
    // buffer is overwritten.
    state["num_params"] = Tensor({1}, DType::Int64, Device::cpu());
    state["num_params"].data<int64_t>()[0] = static_cast<int64_t>(parameters_.size());

    return state;
}

auto RAdam::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    // Audit-4 W.14: param-count guard up front. Mirrors Adam::load_state_dict.
    if (state.count("num_params")) {
        const int64_t expected = state.at("num_params").data<int64_t>()[0];
        if (expected != static_cast<int64_t>(parameters_.size())) {
            throw std::runtime_error(
                "RAdam::load_state_dict: parameter count mismatch - saved " +
                std::to_string(expected) + " but have " +
                std::to_string(parameters_.size()));
        }
    }

    // Load optimizer configuration
    if (state.count("step_count")) {
        step_count_ = state.at("step_count").data<int64_t>()[0];
    }

    if (state.count("lr")) {
        lr_ = state.at("lr").data<double>()[0];
    }

    if (state.count("beta1")) {
        beta1_ = state.at("beta1").data<double>()[0];
    }

    if (state.count("beta2")) {
        beta2_ = state.at("beta2").data<double>()[0];
    }

    if (state.count("eps")) {
        eps_ = state.at("eps").data<double>()[0];
    }

    if (state.count("weight_decay")) {
        weight_decay_ = state.at("weight_decay").data<double>()[0];
    }

    // Validate momentum buffer counts match current parameter count
    size_t saved_count = 0;
    for (const auto& [key, _] : state) {
        if (key.starts_with("exp_avg_") && !key.starts_with("exp_avg_sq_")) {
            ++saved_count;
        }
    }
    if (saved_count > 0 && saved_count != exp_avg_.size()) {
        throw std::runtime_error(
            "RAdam::load_state_dict: momentum buffer count mismatch - "
            "saved " + std::to_string(saved_count) + " but have " +
            std::to_string(exp_avg_.size()) + " parameters");
    }

    // V.27: cast to R.16 master-weights dtype on load.
    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        std::string exp_avg_key = "exp_avg_" + std::to_string(i);
        std::string exp_avg_sq_key = "exp_avg_sq_" + std::to_string(i);
        const DType state_dt = (i < parameters_.size() && parameters_[i])
            ? optim_state_dtype(parameters_[i]->tensor().dtype())
            : DType::Float32;

        if (state.count(exp_avg_key)) {
            exp_avg_[i] = state.at(exp_avg_key).to(state_dt);
        }

        if (state.count(exp_avg_sq_key)) {
            exp_avg_sq_[i] = state.at(exp_avg_sq_key).to(state_dt);
        }
    }
}

} // namespace tenzor::optim
