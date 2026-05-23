#include "tenzor/nn/optim/nadam.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cmath>

namespace tenzor::optim {

NAdam::NAdam(std::vector<std::shared_ptr<Variable>> params, double lr, double beta1,
             double beta2, double eps, double weight_decay, double momentum_decay)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), weight_decay_(weight_decay), momentum_decay_(momentum_decay) {
    initialize_buffers();
}

NAdam::NAdam(std::vector<optim::ParamGroup> groups,
             double default_lr, double default_beta1, double default_beta2,
             double default_eps, double default_weight_decay, double default_momentum_decay)
    : Optimizer(std::move(groups)),
      lr_(default_lr),
      beta1_(default_beta1),
      beta2_(default_beta2),
      eps_(default_eps),
      weight_decay_(default_weight_decay),
      momentum_decay_(default_momentum_decay) {
    initialize_buffers();
}

auto NAdam::step_impl() -> void {
    step_count_++;

    // Audit D.4: per-parameter hyperparameters resolve from the active
    // ParamGroup (when one was set up) or fall through to the optimiser-wide
    // defaults stored on this NAdam instance. `momentum_decay` has no
    // dedicated ParamGroup optional so it always uses the member default.
    struct NAdamHP {
        double lr;
        double beta1;
        double beta2;
        double eps;
        double weight_decay;
        double momentum_decay;
    };

    auto resolve = [&](size_t i) -> NAdamHP {
        NAdamHP hp{lr_, beta1_, beta2_, eps_, weight_decay_, momentum_decay_};
        if (const auto* g = find_group_for_param(i)) {
            hp.lr           = g->lr;
            hp.weight_decay = g->weight_decay;
            hp.beta1        = ParamGroup::or_else(g->beta1, beta1_);
            hp.beta2        = ParamGroup::or_else(g->beta2, beta2_);
            hp.eps          = ParamGroup::or_else(g->eps,   eps_);
        }
        return hp;
    };

    // The momentum-schedule running product (mu_product_) is a global
    // optimiser-wide quantity in PyTorch's reference NAdam, so we keep it
    // on the optimiser (not per-group). It is advanced once per step using
    // the optimiser-default beta1/momentum_decay so that two groups with
    // different beta1 don't fight over its update; per-group beta1 is
    // applied at the parameter update level below.
    const double t = static_cast<double>(step_count_);
    const double mu_t_default  = beta1_ * (1.0 - 0.5 * std::pow(0.96, t       * momentum_decay_));
    mu_product_ *= mu_t_default;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        const NAdamHP hp = resolve(i);

        // Per-parameter momentum schedule using the (possibly per-group)
        // beta1 / momentum_decay.
        const double mu_t  = hp.beta1 * (1.0 - 0.5 * std::pow(0.96, t        * hp.momentum_decay));
        const double mu_t1 = hp.beta1 * (1.0 - 0.5 * std::pow(0.96, (t+1.0)  * hp.momentum_decay));
        const double mu_product_next = mu_product_ * mu_t1;

        const double bias_correction2 = 1.0 - std::pow(hp.beta2, t);

        const Tensor& grad = param.grad().value();

        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, param.tensor().dtype(), param.tensor().device());
        };

        auto grad_copy = grad.clone();

        // Coupled weight decay (L2), matching torch.optim.NAdam's default.
        if (hp.weight_decay > 0.0) {
            grad_copy = grad_copy + param.tensor() * scalar(hp.weight_decay);
        }

        // First and second moment updates (same as Adam).
        exp_avg_[i] = exp_avg_[i] * scalar(hp.beta1) +
                      grad_copy    * scalar(1.0 - hp.beta1);
        exp_avg_sq_[i] = exp_avg_sq_[i] * scalar(hp.beta2) +
                         grad_copy * grad_copy * scalar(1.0 - hp.beta2);

        // NAdam lookahead: blend the current (debiased with mu_product_next)
        // and the fresh gradient (debiased with (1 - mu_t)) first moment
        // terms. Second moment is bias-corrected via bias_correction2.
        auto denom = sqrt(exp_avg_sq_[i]) * scalar(1.0 / std::sqrt(bias_correction2))
                     + scalar(hp.eps);

        const double coeff_m    = mu_t1 / (1.0 - mu_product_next);
        const double coeff_grad = (1.0 - mu_t) / (1.0 - mu_product_);

        // numerator = coeff_m * m_t + coeff_grad * grad
        auto numerator = exp_avg_[i] * scalar(coeff_m) +
                         grad_copy   * scalar(coeff_grad);

        param.tensor() = param.tensor() -
                         div(numerator, denom) * scalar(hp.lr);
    }
}

auto NAdam::initialize_buffers() -> void {
    exp_avg_.clear();
    exp_avg_sq_.clear();

    for (auto& param : parameters_) {
        if (param) {
            exp_avg_.push_back(zeros_like(param->tensor()));
            exp_avg_sq_.push_back(zeros_like(param->tensor()));
        }
    }
}

// Audit K.1: extend exp_avg_ / exp_avg_sq_ for parameters appended via
// add_param_group.  step_count_ and mu_product_ remain optimiser-wide.
auto NAdam::on_parameters_appended_(size_t old_count, size_t new_count) -> void {
    exp_avg_.reserve(new_count);
    exp_avg_sq_.reserve(new_count);
    for (size_t i = old_count; i < new_count; ++i) {
        const auto& param = parameters_[i];
        if (param) {
            exp_avg_.push_back(zeros_like(param->tensor()));
            exp_avg_sq_.push_back(zeros_like(param->tensor()));
        } else {
            exp_avg_.push_back(Tensor{});
            exp_avg_sq_.push_back(Tensor{});
        }
    }
}

auto NAdam::set_lr(double lr) -> void { lr_ = lr; }
auto NAdam::get_lr() const -> double { return lr_; }

auto NAdam::state_dict() const -> std::unordered_map<std::string, Tensor> {
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
    put_double("momentum_decay", momentum_decay_);
    put_double("mu_product", mu_product_);

    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        state["exp_avg_"    + std::to_string(i)] = exp_avg_[i].clone();
        state["exp_avg_sq_" + std::to_string(i)] = exp_avg_sq_[i].clone();
    }
    return state;
}

auto NAdam::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    auto load_double = [&](const char* key, double& out) {
        auto it = state.find(key);
        if (it != state.end()) out = it->second.data<double>()[0];
    };

    if (state.count("step_count")) {
        step_count_ = state.at("step_count").data<int64_t>()[0];
    }
    load_double("lr", lr_);
    load_double("beta1", beta1_);
    load_double("beta2", beta2_);
    load_double("eps", eps_);
    load_double("weight_decay", weight_decay_);
    load_double("momentum_decay", momentum_decay_);
    load_double("mu_product", mu_product_);

    size_t saved_count = 0;
    for (const auto& [key, _] : state) {
        if (key.starts_with("exp_avg_") && !key.starts_with("exp_avg_sq_")) {
            ++saved_count;
        }
    }
    if (saved_count > 0 && saved_count != exp_avg_.size()) {
        throw std::runtime_error(
            "NAdam::load_state_dict: momentum buffer count mismatch - "
            "saved " + std::to_string(saved_count) + " but have " +
            std::to_string(exp_avg_.size()) + " parameters");
    }

    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        std::string exp_avg_key    = "exp_avg_"    + std::to_string(i);
        std::string exp_avg_sq_key = "exp_avg_sq_" + std::to_string(i);
        if (state.count(exp_avg_key))    exp_avg_[i]    = state.at(exp_avg_key).clone();
        if (state.count(exp_avg_sq_key)) exp_avg_sq_[i] = state.at(exp_avg_sq_key).clone();
    }
}

} // namespace tenzor::optim
