#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/foreach.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include <cmath>

namespace tenzor::optim {

// Adam::Adam implementation
Adam::Adam(std::vector<std::shared_ptr<Variable>> params, double lr, double beta1,
          double beta2, double eps, double weight_decay, bool amsgrad)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), weight_decay_(weight_decay), amsgrad_(amsgrad) {
    initialize_buffers();
}

auto Adam::step_impl() -> void {
    step_count_++;

    // Bias correction factors — same for every parameter at this step.
    double bias_correction1 = 1.0 - std::pow(beta1_, step_count_);
    double bias_correction2 = 1.0 - std::pow(beta2_, step_count_);
    double step_size = lr_ / bias_correction1;

    // Collect CPU params eligible for _foreach_* batch path
    // (non-CUDA, non-AMSGrad).  AMSGrad needs a per-param maximum() call
    // that doesn't have a clean foreach equivalent yet; handle those below.
    std::vector<Tensor*>  batch_params;
    std::vector<Tensor>   batch_grads;
    std::vector<Tensor*>  batch_exp_avg;
    std::vector<Tensor*>  batch_exp_avg_sq;
    std::vector<size_t>   batch_indices;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;
        const Tensor& grad = param.grad().value();

        // ── CUDA path: fused single-kernel dispatch ──────────────────────────
        if (param.tensor().device().type == Device::Type::CUDA &&
            grad.device().type == Device::Type::CUDA &&
            (param.tensor().dtype() == DType::Float32 || param.tensor().dtype() == DType::Float64)) {

            std::vector<Tensor> inputs = {
                param.tensor(), grad, exp_avg_[i], exp_avg_sq_[i]
            };
            if (amsgrad_ && i < max_exp_avg_sq_.size()) {
                inputs.push_back(max_exp_avg_sq_[i]);
            }

            NewOpAttributes attrs;
            attrs.set(AttrKey::Lr, lr_);
            attrs.set(AttrKey::Beta1, beta1_);
            attrs.set(AttrKey::Beta2, beta2_);
            attrs.set(AttrKey::Eps, eps_);
            attrs.set(AttrKey::WeightDecay, weight_decay_);
            attrs.set(AttrKey::Step, step_count_);
            attrs.set(AttrKey::Decoupled, false);
            attrs.set(AttrKey::Amsgrad, amsgrad_);
            dispatch(OpId::FusedAdamStep, inputs, attrs);
            continue;
        }

        // ── AMSGrad CPU: scalar per-param path (needs per-param maximum()) ──
        if (amsgrad_ && i < max_exp_avg_sq_.size()) {
            auto scalar = [&](double value) -> Tensor {
                return full({1}, value, param.tensor().dtype(), param.tensor().device());
            };
            auto grad_copy = grad.clone();
            if (weight_decay_ > 0.0)
                grad_copy = grad_copy + param.tensor() * scalar(weight_decay_);

            exp_avg_[i]    = exp_avg_[i]    * scalar(beta1_) + grad_copy * scalar(1.0 - beta1_);
            exp_avg_sq_[i] = exp_avg_sq_[i] * scalar(beta2_) +
                             grad_copy * grad_copy * scalar(1.0 - beta2_);

            max_exp_avg_sq_[i] = maximum(max_exp_avg_sq_[i], exp_avg_sq_[i]);
            auto denom = sqrt(max_exp_avg_sq_[i]) * scalar(1.0 / std::sqrt(bias_correction2))
                        + scalar(eps_);
            param.tensor() = param.tensor() - div(exp_avg_[i], denom) * scalar(step_size);
            continue;
        }

        // ── Standard CPU: collect for _foreach_* batch ──────────────────────
        auto grad_copy = grad.clone();
        if (weight_decay_ > 0.0) {
            // Weight decay: grad += param * weight_decay (L2 reg)
            // Use addcmul equivalent: grad_copy = grad + weight_decay * param
            grad_copy = grad_copy + param.tensor() *
                full({1}, weight_decay_, param.tensor().dtype(), param.tensor().device());
        }
        batch_params.push_back(&param.tensor());
        batch_grads.push_back(std::move(grad_copy));
        batch_exp_avg.push_back(&exp_avg_[i]);
        batch_exp_avg_sq.push_back(&exp_avg_sq_[i]);
        batch_indices.push_back(i);
    }

    if (batch_params.empty()) return;

    // Build mutable list views for _foreach_* (which take vector<Tensor>&)
    std::vector<Tensor> params_view, exp_avg_view, exp_avg_sq_view;
    params_view.reserve(batch_params.size());
    exp_avg_view.reserve(batch_params.size());
    exp_avg_sq_view.reserve(batch_params.size());
    for (size_t k = 0; k < batch_params.size(); ++k) {
        params_view.push_back(*batch_params[k]);        // shallow copies
        exp_avg_view.push_back(*batch_exp_avg[k]);
        exp_avg_sq_view.push_back(*batch_exp_avg_sq[k]);
    }

    // exp_avg  = beta1 * exp_avg  + (1-beta1) * grad
    foreach_lerp_(exp_avg_view, batch_grads, 1.0 - beta1_);

    // grad_sq[i] = grad[i] * grad[i]
    auto grad_sq = foreach_mul(batch_grads, batch_grads);

    // exp_avg_sq = beta2 * exp_avg_sq + (1-beta2) * grad^2
    foreach_lerp_(exp_avg_sq_view, grad_sq, 1.0 - beta2_);

    // denom[i] = sqrt(exp_avg_sq[i] / bias_correction2) + eps
    //          = foreach_sqrt(exp_avg_sq) * (1/sqrt(bc2)) + eps
    // Represent as: denom = foreach_sqrt(exp_avg_sq_bc2) + eps_tensor
    // where exp_avg_sq_bc2[i] = exp_avg_sq[i] / bias_correction2.
    //
    // Use addcdiv: param[i] -= step_size * exp_avg[i] / denom[i]
    // We compute denom inline without foreach (one extra alloc but still O(P*N)):
    auto sqrt_exp_avg_sq_bc2 = foreach_sqrt(exp_avg_sq_view);  // sqrt(v_t)
    // Scale by 1/sqrt(bc2) and add eps — not yet a foreach op, so use addcmul trick:
    // denom[i] = sqrt_v[i] * inv_sqrt_bc2 + eps
    // Implemented as: denom = addcmul(eps_vec, sqrt_v, ones, inv_sqrt_bc2)
    // Simplest correct approach: build denom per-tensor (still O(elements), not O(params)):
    double inv_sqrt_bc2 = 1.0 / std::sqrt(bias_correction2);
    std::vector<Tensor> denom_list;
    denom_list.reserve(sqrt_exp_avg_sq_bc2.size());
    for (size_t k = 0; k < sqrt_exp_avg_sq_bc2.size(); ++k) {
        // denom[k] = sqrt_v[k] * inv_sqrt_bc2 + eps  (scalar broadcast)
        auto& sv = sqrt_exp_avg_sq_bc2[k];
        auto dtype = sv.dtype();
        auto dev   = sv.device();
        denom_list.push_back(sv * full({1}, inv_sqrt_bc2, dtype, dev)
                              + full({1}, eps_, dtype, dev));
    }

    // param -= step_size * exp_avg / denom
    foreach_addcdiv_(params_view, exp_avg_view, denom_list, -step_size);

    // Write updated tensors back (shallow copies don't auto-propagate writes
    // when params_view[k] is reassigned — but foreach_addcdiv_ modifies in-place
    // via the Tensor's shared storage, so the originals are already updated).
    // Write back exp_avg and exp_avg_sq buffers (modified in-place via shared storage).
    for (size_t k = 0; k < batch_indices.size(); ++k) {
        size_t i = batch_indices[k];
        exp_avg_[i]    = exp_avg_view[k];
        exp_avg_sq_[i] = exp_avg_sq_view[k];
        *batch_params[k] = params_view[k];
    }
}

auto Adam::initialize_buffers() -> void {
    exp_avg_.clear();
    exp_avg_sq_.clear();
    max_exp_avg_sq_.clear();

    for (auto& param : parameters_) {
        if (param) {
            exp_avg_.push_back(zeros_like(param->tensor()));
            exp_avg_sq_.push_back(zeros_like(param->tensor()));
            if (amsgrad_) {
                max_exp_avg_sq_.push_back(zeros_like(param->tensor()));
            }
        }
    }
}

auto Adam::set_lr(double lr) -> void {
    lr_ = lr;
}

auto Adam::get_lr() const -> double {
    return lr_;
}

auto Adam::state_dict() const -> std::unordered_map<std::string, Tensor> {
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

    state["amsgrad"] = Tensor({1}, DType::Int64, Device::cpu());
    state["amsgrad"].data<int64_t>()[0] = amsgrad_ ? 1 : 0;

    // Save momentum buffers
    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        state["exp_avg_" + std::to_string(i)] = exp_avg_[i].clone();
        state["exp_avg_sq_" + std::to_string(i)] = exp_avg_sq_[i].clone();
    }

    // Save AMSGrad max second moment buffers
    for (size_t i = 0; i < max_exp_avg_sq_.size(); ++i) {
        state["max_exp_avg_sq_" + std::to_string(i)] = max_exp_avg_sq_[i].clone();
    }

    return state;
}

auto Adam::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
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

    if (state.count("amsgrad")) {
        amsgrad_ = state.at("amsgrad").data<int64_t>()[0] != 0;
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
            "Adam::load_state_dict: momentum buffer count mismatch - "
            "saved " + std::to_string(saved_count) + " but have " +
            std::to_string(exp_avg_.size()) + " parameters");
    }

    // Load momentum buffers
    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        std::string exp_avg_key = "exp_avg_" + std::to_string(i);
        std::string exp_avg_sq_key = "exp_avg_sq_" + std::to_string(i);

        if (state.count(exp_avg_key)) {
            exp_avg_[i] = state.at(exp_avg_key).clone();
        }

        if (state.count(exp_avg_sq_key)) {
            exp_avg_sq_[i] = state.at(exp_avg_sq_key).clone();
        }
    }

    // Load AMSGrad max second moment buffers
    for (size_t i = 0; i < max_exp_avg_sq_.size(); ++i) {
        std::string key = "max_exp_avg_sq_" + std::to_string(i);
        if (state.count(key)) {
            max_exp_avg_sq_[i] = state.at(key).clone();
        }
    }
}

// AdamW implementation
AdamW::AdamW(std::vector<std::shared_ptr<Variable>> params, double lr, double beta1,
            double beta2, double eps, double weight_decay, bool amsgrad)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), weight_decay_(weight_decay), amsgrad_(amsgrad) {
    initialize_buffers();
}

auto AdamW::step_impl() -> void {
    step_count_++;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        const Tensor& grad = param.grad().value();

        // Use fused CUDA kernel for CUDA tensors (single kernel vs ~15 kernels)
        if (param.tensor().device().type == Device::Type::CUDA &&
            grad.device().type == Device::Type::CUDA &&
            (param.tensor().dtype() == DType::Float32 || param.tensor().dtype() == DType::Float64)) {

            // Prepare inputs for dispatch
            std::vector<Tensor> inputs = {
                param.tensor(), grad, exp_avg_[i], exp_avg_sq_[i]
            };
            if (amsgrad_ && i < max_exp_avg_sq_.size()) {
                inputs.push_back(max_exp_avg_sq_[i]);
            }

            // Prepare attributes (use double precision for Float64 accuracy)
            NewOpAttributes attrs;
            attrs.set(AttrKey::Lr, lr_);
            attrs.set(AttrKey::Beta1, beta1_);
            attrs.set(AttrKey::Beta2, beta2_);
            attrs.set(AttrKey::Eps, eps_);
            attrs.set(AttrKey::WeightDecay, weight_decay_);
            attrs.set(AttrKey::Step, step_count_);
            attrs.set(AttrKey::Decoupled, true);   // Decoupled weight decay for AdamW
            attrs.set(AttrKey::Amsgrad, amsgrad_);

            dispatch(OpId::FusedAdamStep, inputs, attrs);
            continue;
        }

        // CPU fallback path — use dtype-appropriate scalar tensors to
        // preserve Float64 precision (static_cast<float> truncates doubles)
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, param.tensor().dtype(), param.tensor().device());
        };

        auto grad_copy = grad.clone();

        // Update biased first moment estimate
        exp_avg_[i] = exp_avg_[i] * scalar(beta1_) +
                     grad_copy * scalar(1.0 - beta1_);

        // Update biased second raw moment estimate
        exp_avg_sq_[i] = exp_avg_sq_[i] * scalar(beta2_) +
                        grad_copy * grad_copy * scalar(1.0 - beta2_);

        // Bias correction
        double bias_correction1 = 1.0 - std::pow(beta1_, step_count_);
        double bias_correction2 = 1.0 - std::pow(beta2_, step_count_);

        double step_size = lr_ / bias_correction1;

        // Compute denominator: use max of second moment if AMSGrad is enabled
        Tensor denom_base;
        if (amsgrad_ && i < max_exp_avg_sq_.size()) {
            max_exp_avg_sq_[i] = maximum(max_exp_avg_sq_[i], exp_avg_sq_[i]);
            denom_base = max_exp_avg_sq_[i];
        } else {
            denom_base = exp_avg_sq_[i];
        }

        auto denom = sqrt(denom_base) * scalar(1.0 / std::sqrt(bias_correction2))
                    + scalar(eps_);

        // Decoupled weight decay (AdamW)
        if (weight_decay_ > 0) {
            param.tensor() = param.tensor() * scalar(1.0 - lr_ * weight_decay_);
        }

        // Update parameters
        param.tensor() = param.tensor() -
                        div(exp_avg_[i], denom) * scalar(step_size);
    }
}

auto AdamW::set_lr(double lr) -> void {
    lr_ = lr;
}

auto AdamW::get_lr() const -> double {
    return lr_;
}

auto AdamW::initialize_buffers() -> void {
    exp_avg_.clear();
    exp_avg_sq_.clear();
    max_exp_avg_sq_.clear();

    for (auto& param : parameters_) {
        if (param) {
            exp_avg_.push_back(zeros_like(param->tensor()));
            exp_avg_sq_.push_back(zeros_like(param->tensor()));
            if (amsgrad_) {
                max_exp_avg_sq_.push_back(zeros_like(param->tensor()));
            }
        }
    }
}

auto AdamW::state_dict() const -> std::unordered_map<std::string, Tensor> {
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

    state["amsgrad"] = Tensor({1}, DType::Int64, Device::cpu());
    state["amsgrad"].data<int64_t>()[0] = amsgrad_ ? 1 : 0;

    // Save momentum buffers
    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        state["exp_avg_" + std::to_string(i)] = exp_avg_[i].clone();
        state["exp_avg_sq_" + std::to_string(i)] = exp_avg_sq_[i].clone();
    }

    // Save AMSGrad max second moment buffers
    for (size_t i = 0; i < max_exp_avg_sq_.size(); ++i) {
        state["max_exp_avg_sq_" + std::to_string(i)] = max_exp_avg_sq_[i].clone();
    }

    return state;
}

auto AdamW::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
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

    if (state.count("amsgrad")) {
        amsgrad_ = state.at("amsgrad").data<int64_t>()[0] != 0;
    }

    // Load momentum buffers
    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        std::string exp_avg_key = "exp_avg_" + std::to_string(i);
        std::string exp_avg_sq_key = "exp_avg_sq_" + std::to_string(i);

        if (state.count(exp_avg_key)) {
            exp_avg_[i] = state.at(exp_avg_key).clone();
        }

        if (state.count(exp_avg_sq_key)) {
            exp_avg_sq_[i] = state.at(exp_avg_sq_key).clone();
        }
    }

    // Load AMSGrad max second moment buffers
    for (size_t i = 0; i < max_exp_avg_sq_.size(); ++i) {
        std::string key = "max_exp_avg_sq_" + std::to_string(i);
        if (state.count(key)) {
            max_exp_avg_sq_[i] = state.at(key).clone();
        }
    }
}

} // namespace tenzor::optim
