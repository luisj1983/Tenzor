/**
 * @file adam_atan2.cpp
 * @brief Adam-atan2 optimizer implementation
 */

#include "tenzor/nn/optim/adam_atan2.hpp"
#include "tenzor/nn/optim/master_weights.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include <cmath>
#include <algorithm>

namespace tenzor::optim {

// ============================================================================
// AdamAtan2 Implementation
// ============================================================================

AdamAtan2::AdamAtan2(std::vector<std::shared_ptr<Variable>> params,
                     double lr, double beta1, double beta2,
                     double eps, double weight_decay, bool amsgrad)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), weight_decay_(weight_decay), amsgrad_(amsgrad) {
    initialize_buffers();
    update_stats_ = {0.0, 0.0, 0.0};
}

AdamAtan2::AdamAtan2(std::vector<optim::ParamGroup> groups,
                     double default_lr, double default_beta1, double default_beta2,
                     double default_eps, double default_weight_decay, bool amsgrad)
    : Optimizer(std::move(groups)),
      lr_(default_lr), beta1_(default_beta1), beta2_(default_beta2),
      eps_(default_eps), weight_decay_(default_weight_decay), amsgrad_(amsgrad) {
    initialize_buffers();
    update_stats_ = {0.0, 0.0, 0.0};
}

auto AdamAtan2::step_impl() -> void {
    step_count_++;

    // Audit D.4: per-parameter hyperparameters resolve from the
    // active ParamGroup (when one was set up) or fall through to
    // the optimiser-wide defaults stored on this AdamAtan2 instance.
    // EE.16: amsgrad is now a ParamGroup field too — falls back to the
    // optimiser-wide default when the group leaves it unset.
    struct AdamAtan2HP {
        double lr;
        double beta1;
        double beta2;
        double eps;
        double weight_decay;
        bool   amsgrad;
    };

    auto resolve = [&](size_t i) -> AdamAtan2HP {
        AdamAtan2HP hp{lr_, beta1_, beta2_, eps_, weight_decay_, amsgrad_};
        if (const auto* g = find_group_for_param(i)) {
            hp.lr           = g->lr;
            hp.weight_decay = g->weight_decay;
            hp.beta1        = ParamGroup::or_else(g->beta1, beta1_);
            hp.beta2        = ParamGroup::or_else(g->beta2, beta2_);
            hp.eps          = ParamGroup::or_else(g->eps,   eps_);
            hp.amsgrad      = ParamGroup::or_else(g->amsgrad, amsgrad_);
        }
        return hp;
    };

    double total_update_mag = 0.0;
    double max_update_mag = 0.0;
    double total_grad_mag = 0.0;
    int64_t total_params = 0;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        const AdamAtan2HP hp = resolve(i);

        Tensor& param_tensor = param.tensor();
        const Tensor& grad_tensor = param.grad().value();

        // Use fused CUDA kernel for CUDA tensors
        if (param_tensor.device().type == Device::Type::CUDA &&
            grad_tensor.device().type == Device::Type::CUDA &&
            (param_tensor.dtype() == DType::Float32 || param_tensor.dtype() == DType::Float64)) {

            std::vector<Tensor> inputs = {
                param_tensor, grad_tensor, exp_avg_[i], exp_avg_sq_[i]
            };
            if (hp.amsgrad && i < max_exp_avg_sq_.size()) {
                inputs.push_back(max_exp_avg_sq_[i]);
            }

            NewOpAttributes attrs;
            // Z.9: pass native double to match the fused-Adam contract; the
            // previous static_cast<float> narrowed Float64 params' hyperparams.
            attrs.set(AttrKey::Lr, hp.lr);
            attrs.set(AttrKey::Beta1, hp.beta1);
            attrs.set(AttrKey::Beta2, hp.beta2);
            attrs.set(AttrKey::Eps, hp.eps);
            attrs.set(AttrKey::WeightDecay, hp.weight_decay);
            attrs.set(AttrKey::Step, step_count_);
            attrs.set(AttrKey::Amsgrad, hp.amsgrad);

            dispatch(OpId::FusedAdamAtan2Step, inputs, attrs);
            total_params++;
            continue;
        }

        // R.16: half-precision params use Float32 state buffers; run math
        // in state dtype, cast back to param dtype on write.
        const DType param_dt = param_tensor.dtype();
        const DType state_dt = optim_state_dtype(param_dt);
        const bool needs_upcast = (state_dt != param_dt);

        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, state_dt, param_tensor.device());
        };

        Tensor param_hi = needs_upcast ? param_tensor.to(state_dt) : param_tensor;
        Tensor grad = needs_upcast ? grad_tensor.to(state_dt) : grad_tensor.clone();

        // Track gradient magnitude only when statistics tracking is enabled
        // (.item() forces GPU synchronization per parameter, which is expensive)
        if (track_statistics_) {
            auto grad_sq_sum = tenzor::sum(grad * grad).item<float>();
            total_grad_mag += std::sqrt(grad_sq_sum);
        }
        total_params++;

        // Update biased first moment estimate
        exp_avg_[i] = exp_avg_[i] * scalar(hp.beta1) +
                     grad * scalar(1.0 - hp.beta1);

        // Update biased second raw moment estimate
        exp_avg_sq_[i] = exp_avg_sq_[i] * scalar(hp.beta2) +
                        grad * grad * scalar(1.0 - hp.beta2);

        // Bias correction
        double bias_correction1 = 1.0 - std::pow(hp.beta1, step_count_);
        double bias_correction2 = 1.0 - std::pow(hp.beta2, step_count_);

        // Compute bias-corrected estimates
        auto m_hat = exp_avg_[i] * scalar(1.0 / bias_correction1);
        auto v_hat = exp_avg_sq_[i] * scalar(1.0 / bias_correction2);

        // AMSGrad: use maximum of past squared gradients
        // EE.16: hp.amsgrad honours per-group override; resolves to amsgrad_
        // when the group leaves it unset (preserving the pre-EE.16 default).
        // initialize_buffers() / on_parameters_appended_ still allocate
        // max_exp_avg_sq_ slots whenever amsgrad_ is set on the optimiser, so
        // a group that turns amsgrad on at add_param_group time still has the
        // buffer available; a group that turns it off simply skips the slot.
        if (hp.amsgrad && i < max_exp_avg_sq_.size()) {
            // Element-wise max using comparison
            auto mask = max_exp_avg_sq_[i] > v_hat;
            auto mask_f = mask.to(state_dt);
            auto inv_mask = ones_like(mask_f) - mask_f;
            max_exp_avg_sq_[i] = max_exp_avg_sq_[i] * mask_f + v_hat * inv_mask;
            v_hat = max_exp_avg_sq_[i];
        }

        // Compute sqrt(v_hat) + eps
        auto denom = sqrt(v_hat) + scalar(hp.eps);

        // Adam-atan2 update: use atan(m_hat / denom) to approximate atan2
        // atan2(y, x) ≈ atan(y/x) when x > 0 (which is always true for denom)
        // This provides bounded updates in [-π/2, π/2]
        auto ratio = div(m_hat, denom);
        auto update = tenzor::atan(ratio);

        // Track update magnitude only when statistics tracking is enabled
        if (track_statistics_) {
            auto update_flat = update.view({-1});
            auto update_sq = update_flat * update_flat;
            auto update_sq_sum = tenzor::sum(update_sq).template item<float>();
            double update_mag = std::sqrt(update_sq_sum);
            total_update_mag += update_mag;
            max_update_mag = std::max(max_update_mag, update_mag);
        }

        // Decoupled weight decay (like AdamW)
        if (hp.weight_decay > 0) {
            param_hi = param_hi * scalar(1.0 - hp.lr * hp.weight_decay);
        }

        // Apply update
        param_hi = param_hi - update * scalar(hp.lr);
        param.tensor() = needs_upcast ? param_hi.to(param_dt) : param_hi;
    }

    // Update statistics (only meaningful when tracking is enabled)
    if (track_statistics_ && total_params > 0) {
        update_stats_.avg_update_magnitude = total_update_mag / total_params;
        update_stats_.max_update_magnitude = max_update_mag;
        update_stats_.avg_gradient_magnitude = total_grad_mag / total_params;
    }
}

auto AdamAtan2::initialize_buffers() -> void {
    exp_avg_.clear();
    exp_avg_sq_.clear();
    max_exp_avg_sq_.clear();

    for (auto& param : parameters_) {
        if (param) {
            // R.16: half-precision params get Float32 state buffers.
            exp_avg_.push_back(make_optim_state(param->tensor()));
            exp_avg_sq_.push_back(make_optim_state(param->tensor()));
            if (amsgrad_) {
                max_exp_avg_sq_.push_back(make_optim_state(param->tensor()));
            }
        }
    }
}

// Audit K.1: extend exp_avg_ / exp_avg_sq_ (and max_exp_avg_sq_ when
// amsgrad_ is enabled) for parameters appended via add_param_group.
auto AdamAtan2::on_parameters_appended_(size_t old_count, size_t new_count) -> void {
    exp_avg_.reserve(new_count);
    exp_avg_sq_.reserve(new_count);
    if (amsgrad_) max_exp_avg_sq_.reserve(new_count);
    for (size_t i = old_count; i < new_count; ++i) {
        const auto& param = parameters_[i];
        if (param) {
            // R.16: see AdamAtan2::initialize_buffers for dtype rationale.
            exp_avg_.push_back(make_optim_state(param->tensor()));
            exp_avg_sq_.push_back(make_optim_state(param->tensor()));
            if (amsgrad_) {
                max_exp_avg_sq_.push_back(make_optim_state(param->tensor()));
            }
        } else {
            exp_avg_.push_back(Tensor{});
            exp_avg_sq_.push_back(Tensor{});
            if (amsgrad_) max_exp_avg_sq_.push_back(Tensor{});
        }
    }
}

auto AdamAtan2::set_lr(double lr) -> void {
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

auto AdamAtan2::get_lr() const -> double {
    return lr_;
}

auto AdamAtan2::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    // Save optimizer configuration
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
        if (amsgrad_ && i < max_exp_avg_sq_.size()) {
            state["max_exp_avg_sq_" + std::to_string(i)] = max_exp_avg_sq_[i].clone();
        }
    }

    return state;
}

auto AdamAtan2::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
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

    // V.27: cast to R.16 master-weights dtype on load.
    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        std::string exp_avg_key = "exp_avg_" + std::to_string(i);
        std::string exp_avg_sq_key = "exp_avg_sq_" + std::to_string(i);
        std::string max_key = "max_exp_avg_sq_" + std::to_string(i);
        const DType state_dt = (i < parameters_.size() && parameters_[i])
            ? optim_state_dtype(parameters_[i]->tensor().dtype())
            : DType::Float32;

        if (state.count(exp_avg_key)) {
            exp_avg_[i] = state.at(exp_avg_key).to(state_dt);
        }

        if (state.count(exp_avg_sq_key)) {
            exp_avg_sq_[i] = state.at(exp_avg_sq_key).to(state_dt);
        }

        if (amsgrad_ && state.count(max_key) && i < max_exp_avg_sq_.size()) {
            max_exp_avg_sq_[i] = state.at(max_key).to(state_dt);
        }
    }
}

// ============================================================================
// LinearWarmup Implementation
// ============================================================================

LinearWarmup::LinearWarmup(Optimizer& optimizer, int64_t warmup_steps, double base_lr)
    : optimizer_(optimizer), warmup_steps_(warmup_steps), base_lr_(base_lr), current_lr_(0.0) {
}

auto LinearWarmup::update(int64_t step) -> void {
    if (step < warmup_steps_) {
        // Linear warmup
        current_lr_ = base_lr_ * static_cast<double>(step + 1) / static_cast<double>(warmup_steps_);
    } else {
        // After warmup, use base learning rate
        current_lr_ = base_lr_;
    }

    // Update optimizer's learning rate
    // This requires the optimizer to have a set_lr method
    // We use dynamic dispatch via dynamic_cast
    if (auto* adam_atan2 = dynamic_cast<AdamAtan2*>(&optimizer_)) {
        adam_atan2->set_lr(current_lr_);
    }
    // Add other optimizer types as needed
}

} // namespace tenzor::optim
