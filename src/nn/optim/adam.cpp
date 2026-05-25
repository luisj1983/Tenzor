#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/nn/optim/master_weights.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/foreach.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include <cmath>
#include <map>
#include <tuple>

namespace tenzor::optim {

// R.16 master-weights helpers (optim_state_dtype / make_optim_state) are
// shared across all optimisers via include/tenzor/nn/optim/master_weights.hpp.

// Adam::Adam implementation
Adam::Adam(std::vector<std::shared_ptr<Variable>> params, double lr, double beta1,
          double beta2, double eps, double weight_decay, bool amsgrad)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), weight_decay_(weight_decay), amsgrad_(amsgrad) {
    initialize_buffers();
}

Adam::Adam(std::vector<optim::ParamGroup> groups,
           double default_lr, double default_beta1, double default_beta2,
           double default_eps, double default_weight_decay, bool default_amsgrad)
    : Optimizer(std::move(groups)),
      lr_(default_lr),
      beta1_(default_beta1),
      beta2_(default_beta2),
      eps_(default_eps),
      weight_decay_(default_weight_decay),
      amsgrad_(default_amsgrad) {
    initialize_buffers();
}

auto Adam::step_impl() -> void {
    step_count_++;

    // Audit D.4: resolve hyperparams per-param from the active
    // ParamGroup, with optimizer-member fallback.  When constructed
    // from a flat parameter list (no groups), `find_group_for_param`
    // returns nullptr and every param uses the Adam member defaults —
    // preserving pre-D.4 behaviour bit-for-bit.
    struct HP {
        double lr, beta1, beta2, eps, weight_decay;
        bool   amsgrad;
    };
    auto resolve = [this](size_t i) -> HP {
        HP hp{lr_, beta1_, beta2_, eps_, weight_decay_, amsgrad_};
        const auto* g = find_group_for_param(i);
        if (g) {
            hp.lr           = g->lr;
            hp.weight_decay = g->weight_decay;
            hp.beta1        = ParamGroup::or_else(g->beta1, beta1_);
            hp.beta2        = ParamGroup::or_else(g->beta2, beta2_);
            hp.eps          = ParamGroup::or_else(g->eps,   eps_);
            hp.amsgrad      = ParamGroup::or_else(g->amsgrad, amsgrad_);
        }
        return hp;
    };

    // Collect CPU params eligible for _foreach_* batch path
    // (non-CUDA, non-AMSGrad).  AMSGrad needs a per-param maximum() call
    // that doesn't have a clean foreach equivalent yet; handle those below.
    //
    // Audit D.4: the foreach fast path requires uniform beta1/beta2/eps
    // across the batch (the step_size and bias_correction factors are
    // shared scalars).  We therefore key the batch by the resolved
    // (beta1, beta2, eps) tuple and run one foreach pass per bucket.
    // For the common case of a single ParamGroup this is identical to
    // the old single-batch path; mixed groups simply do K passes.
    struct CpuEntry {
        Tensor* param;
        Tensor  grad;
        Tensor* exp_avg;
        Tensor* exp_avg_sq;
        size_t  idx;
        HP      hp;
    };
    std::vector<CpuEntry> cpu;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;
        const Tensor& grad = param.grad().value();
        HP hp = resolve(i);

        // ── CUDA path: fused single-kernel dispatch ──────────────────────────
        if (param.tensor().device().type == Device::Type::CUDA &&
            grad.device().type == Device::Type::CUDA &&
            (param.tensor().dtype() == DType::Float32 || param.tensor().dtype() == DType::Float64)) {

            std::vector<Tensor> inputs = {
                param.tensor(), grad, exp_avg_[i], exp_avg_sq_[i]
            };
            if (hp.amsgrad && i < max_exp_avg_sq_.size()) {
                inputs.push_back(max_exp_avg_sq_[i]);
            }

            NewOpAttributes attrs;
            attrs.set(AttrKey::Lr, hp.lr);
            attrs.set(AttrKey::Beta1, hp.beta1);
            attrs.set(AttrKey::Beta2, hp.beta2);
            attrs.set(AttrKey::Eps, hp.eps);
            attrs.set(AttrKey::WeightDecay, hp.weight_decay);
            attrs.set(AttrKey::Step, step_count_);
            attrs.set(AttrKey::Decoupled, false);
            attrs.set(AttrKey::Amsgrad, hp.amsgrad);
            dispatch(OpId::FusedAdamStep, inputs, attrs);
            continue;
        }

        // R.16: when the parameter is half precision, run the step math in
        // the state buffer's dtype (Float32). param/grad are upcast on entry
        // and downcast on write-back; exp_avg / exp_avg_sq already live at
        // the state dtype.
        const DType param_dt = param.tensor().dtype();
        const DType state_dt = optim_state_dtype(param_dt);
        const bool needs_upcast = (state_dt != param_dt);

        // ── AMSGrad CPU: scalar per-param path (needs per-param maximum()) ──
        if (hp.amsgrad && i < max_exp_avg_sq_.size()) {
            double bc1 = 1.0 - std::pow(hp.beta1, step_count_);
            double bc2 = 1.0 - std::pow(hp.beta2, step_count_);
            double step_size = hp.lr / bc1;
            auto scalar = [&](double value) -> Tensor {
                return full({1}, value, state_dt, param.tensor().device());
            };
            Tensor param_hi = needs_upcast ? param.tensor().to(state_dt) : param.tensor();
            Tensor grad_copy = needs_upcast ? grad.to(state_dt) : grad.clone();
            if (hp.weight_decay > 0.0)
                grad_copy = grad_copy + param_hi * scalar(hp.weight_decay);

            exp_avg_[i]    = exp_avg_[i]    * scalar(hp.beta1) + grad_copy * scalar(1.0 - hp.beta1);
            exp_avg_sq_[i] = exp_avg_sq_[i] * scalar(hp.beta2) +
                             grad_copy * grad_copy * scalar(1.0 - hp.beta2);

            max_exp_avg_sq_[i] = maximum(max_exp_avg_sq_[i], exp_avg_sq_[i]);
            auto denom = sqrt(max_exp_avg_sq_[i]) * scalar(1.0 / std::sqrt(bc2))
                        + scalar(hp.eps);
            Tensor updated = param_hi - div(exp_avg_[i], denom) * scalar(step_size);
            param.tensor() = needs_upcast ? updated.to(param_dt) : updated;
            continue;
        }

        // R.16: half-precision params cannot use the foreach fast path
        // because foreach_lerp_/foreach_addcdiv_ require all the lists to
        // share dtype with the param. Inline the same Adam update in the
        // state dtype and skip the bucket. Other dtypes flow through the
        // foreach batch below unchanged.
        if (needs_upcast) {
            double bc1 = 1.0 - std::pow(hp.beta1, step_count_);
            double bc2 = 1.0 - std::pow(hp.beta2, step_count_);
            double step_size = hp.lr / bc1;
            auto scalar = [&](double value) -> Tensor {
                return full({1}, value, state_dt, param.tensor().device());
            };
            Tensor param_hi  = param.tensor().to(state_dt);
            Tensor grad_copy = grad.to(state_dt);
            if (hp.weight_decay > 0.0)
                grad_copy = grad_copy + param_hi * scalar(hp.weight_decay);

            exp_avg_[i]    = exp_avg_[i]    * scalar(hp.beta1) + grad_copy * scalar(1.0 - hp.beta1);
            exp_avg_sq_[i] = exp_avg_sq_[i] * scalar(hp.beta2) +
                             grad_copy * grad_copy * scalar(1.0 - hp.beta2);

            auto denom = sqrt(exp_avg_sq_[i]) * scalar(1.0 / std::sqrt(bc2))
                        + scalar(hp.eps);
            Tensor updated = param_hi - div(exp_avg_[i], denom) * scalar(step_size);
            param.tensor() = updated.to(param_dt);
            continue;
        }

        // ── Standard CPU: collect with resolved hyperparams ─────────────────
        auto grad_copy = grad.clone();
        if (hp.weight_decay > 0.0) {
            grad_copy = grad_copy + param.tensor() *
                full({1}, hp.weight_decay, param.tensor().dtype(), param.tensor().device());
        }
        cpu.push_back({&param.tensor(), std::move(grad_copy),
                       &exp_avg_[i], &exp_avg_sq_[i], i, hp});
    }

    if (cpu.empty()) return;

    // Audit D.4: bucket the CPU batch by (beta1, beta2, eps) tuple so
    // the foreach fast path applies within each bucket (uniform betas
    // and step_size are required because foreach_lerp_ takes a single
    // scalar weight). With a single ParamGroup this is one bucket and
    // identical to the pre-D.4 path; with mixed groups it's K buckets.
    auto bucket_key = [](const CpuEntry& e) {
        return std::make_tuple(e.hp.beta1, e.hp.beta2, e.hp.eps, e.hp.lr);
    };
    std::map<std::tuple<double, double, double, double>,
             std::vector<size_t>> buckets;
    for (size_t k = 0; k < cpu.size(); ++k) {
        buckets[bucket_key(cpu[k])].push_back(k);
    }

    for (auto& [key, indices] : buckets) {
        const auto& [beta1, beta2, eps, lr] = key;
        double bias_correction1 = 1.0 - std::pow(beta1, step_count_);
        double bias_correction2 = 1.0 - std::pow(beta2, step_count_);
        double step_size = lr / bias_correction1;

        // Build per-bucket _foreach_* views.
        std::vector<Tensor> bgrad, params_view, exp_avg_view, exp_avg_sq_view;
        bgrad.reserve(indices.size());
        params_view.reserve(indices.size());
        exp_avg_view.reserve(indices.size());
        exp_avg_sq_view.reserve(indices.size());
        for (size_t k : indices) {
            bgrad.push_back(cpu[k].grad);
            params_view.push_back(*cpu[k].param);
            exp_avg_view.push_back(*cpu[k].exp_avg);
            exp_avg_sq_view.push_back(*cpu[k].exp_avg_sq);
        }

        foreach_lerp_(exp_avg_view, bgrad, 1.0 - beta1);
        auto grad_sq = foreach_mul(bgrad, bgrad);
        foreach_lerp_(exp_avg_sq_view, grad_sq, 1.0 - beta2);

        auto sqrt_v = foreach_sqrt(exp_avg_sq_view);
        double inv_sqrt_bc2 = 1.0 / std::sqrt(bias_correction2);
        std::vector<Tensor> denom_list;
        denom_list.reserve(sqrt_v.size());
        for (size_t k = 0; k < sqrt_v.size(); ++k) {
            auto& sv = sqrt_v[k];
            auto dtype = sv.dtype();
            auto dev   = sv.device();
            denom_list.push_back(sv * full({1}, inv_sqrt_bc2, dtype, dev)
                                  + full({1}, eps, dtype, dev));
        }
        foreach_addcdiv_(params_view, exp_avg_view, denom_list, -step_size);

        for (size_t b = 0; b < indices.size(); ++b) {
            size_t k = indices[b];
            *cpu[k].exp_avg    = exp_avg_view[b];
            *cpu[k].exp_avg_sq = exp_avg_sq_view[b];
            *cpu[k].param      = params_view[b];
        }
    }
}

auto Adam::initialize_buffers() -> void {
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

// Audit K.1: extend exp_avg_ / exp_avg_sq_ / max_exp_avg_sq_ when
// add_param_group appends new parameters mid-training.
auto Adam::on_parameters_appended_(size_t old_count, size_t new_count) -> void {
    exp_avg_.reserve(new_count);
    exp_avg_sq_.reserve(new_count);
    if (amsgrad_) max_exp_avg_sq_.reserve(new_count);
    for (size_t i = old_count; i < new_count; ++i) {
        const auto& param = parameters_[i];
        if (param) {
            // R.16: see Adam::initialize_buffers for the dtype rationale.
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

    // P.2: num_params guard so load_state_dict can reject mismatched models
    // before any tensor is overwritten (mirrors SGD::state_dict).
    state["num_params"] = Tensor({1}, DType::Int64, Device::cpu());
    state["num_params"].data<int64_t>()[0] = static_cast<int64_t>(parameters_.size());

    return state;
}

auto Adam::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    // P.2: param-count guard up front. Mirrors SGD::load_state_dict so a
    // restore against the wrong model fails loudly instead of partially
    // populating per-parameter buffers and then succeeding silently.
    if (state.count("num_params")) {
        const int64_t expected = state.at("num_params").data<int64_t>()[0];
        if (expected != static_cast<int64_t>(parameters_.size())) {
            throw std::runtime_error(
                "Adam::load_state_dict: parameter count mismatch - saved " +
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

    if (state.count("amsgrad")) {
        amsgrad_ = state.at("amsgrad").data<int64_t>()[0] != 0;
    }

    // Y.18: if amsgrad flipped false -> true on load, allocate the
    // max_exp_avg_sq_ buffers now (the constructor only sized them when
    // amsgrad was true). Without this, the load loop below iterates an
    // empty vector and silently discards every saved max_exp_avg_sq_i.
    if (amsgrad_ && max_exp_avg_sq_.empty()) {
        max_exp_avg_sq_.reserve(parameters_.size());
        for (auto& param : parameters_) {
            if (param) {
                max_exp_avg_sq_.push_back(make_optim_state(param->tensor()));
            } else {
                max_exp_avg_sq_.push_back(Tensor{});
            }
        }
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

    // V.27: cast to the R.16 master-weights dtype on load. Pre-R.16
    // checkpoints stored half-precision buffers; restoring them as-is
    // would silently break the master-weights invariant on the next step.
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

    // Load AMSGrad max second moment buffers
    for (size_t i = 0; i < max_exp_avg_sq_.size(); ++i) {
        std::string key = "max_exp_avg_sq_" + std::to_string(i);
        const DType state_dt = (i < parameters_.size() && parameters_[i])
            ? optim_state_dtype(parameters_[i]->tensor().dtype())
            : DType::Float32;
        if (state.count(key)) {
            max_exp_avg_sq_[i] = state.at(key).to(state_dt);
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

AdamW::AdamW(std::vector<optim::ParamGroup> groups,
             double default_lr, double default_beta1, double default_beta2,
             double default_eps, double default_weight_decay, bool default_amsgrad)
    : Optimizer(std::move(groups)),
      lr_(default_lr),
      beta1_(default_beta1),
      beta2_(default_beta2),
      eps_(default_eps),
      weight_decay_(default_weight_decay),
      amsgrad_(default_amsgrad) {
    initialize_buffers();
}

auto AdamW::step_impl() -> void {
    step_count_++;

    // Audit D.4: resolve per-param hyperparams from the active
    // ParamGroup (with optimizer-member fallback).
    struct HP {
        double lr, beta1, beta2, eps, weight_decay;
        bool   amsgrad;
    };
    auto resolve = [this](size_t i) -> HP {
        HP hp{lr_, beta1_, beta2_, eps_, weight_decay_, amsgrad_};
        const auto* g = find_group_for_param(i);
        if (g) {
            hp.lr           = g->lr;
            hp.weight_decay = g->weight_decay;
            hp.beta1        = ParamGroup::or_else(g->beta1, beta1_);
            hp.beta2        = ParamGroup::or_else(g->beta2, beta2_);
            hp.eps          = ParamGroup::or_else(g->eps,   eps_);
            hp.amsgrad      = ParamGroup::or_else(g->amsgrad, amsgrad_);
        }
        return hp;
    };

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;
        HP hp = resolve(i);

        const Tensor& grad = param.grad().value();

        // Use fused CUDA kernel for CUDA tensors (single kernel vs ~15 kernels)
        if (param.tensor().device().type == Device::Type::CUDA &&
            grad.device().type == Device::Type::CUDA &&
            (param.tensor().dtype() == DType::Float32 || param.tensor().dtype() == DType::Float64)) {

            // Prepare inputs for dispatch
            std::vector<Tensor> inputs = {
                param.tensor(), grad, exp_avg_[i], exp_avg_sq_[i]
            };
            if (hp.amsgrad && i < max_exp_avg_sq_.size()) {
                inputs.push_back(max_exp_avg_sq_[i]);
            }

            // Prepare attributes (use double precision for Float64 accuracy)
            NewOpAttributes attrs;
            attrs.set(AttrKey::Lr, hp.lr);
            attrs.set(AttrKey::Beta1, hp.beta1);
            attrs.set(AttrKey::Beta2, hp.beta2);
            attrs.set(AttrKey::Eps, hp.eps);
            attrs.set(AttrKey::WeightDecay, hp.weight_decay);
            attrs.set(AttrKey::Step, step_count_);
            attrs.set(AttrKey::Decoupled, true);   // Decoupled weight decay for AdamW
            attrs.set(AttrKey::Amsgrad, hp.amsgrad);

            dispatch(OpId::FusedAdamStep, inputs, attrs);
            continue;
        }

        // R.16: half-precision params use Float32 state buffers; cast
        // param/grad to the state dtype for the math, cast back on write.
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

        // Update biased first moment estimate
        exp_avg_[i] = exp_avg_[i] * scalar(hp.beta1) +
                     grad_copy * scalar(1.0 - hp.beta1);

        // Update biased second raw moment estimate
        exp_avg_sq_[i] = exp_avg_sq_[i] * scalar(hp.beta2) +
                        grad_copy * grad_copy * scalar(1.0 - hp.beta2);

        // Bias correction
        double bias_correction1 = 1.0 - std::pow(hp.beta1, step_count_);
        double bias_correction2 = 1.0 - std::pow(hp.beta2, step_count_);

        double step_size = hp.lr / bias_correction1;

        // Compute denominator: use max of second moment if AMSGrad is enabled
        Tensor denom_base;
        if (hp.amsgrad && i < max_exp_avg_sq_.size()) {
            max_exp_avg_sq_[i] = maximum(max_exp_avg_sq_[i], exp_avg_sq_[i]);
            denom_base = max_exp_avg_sq_[i];
        } else {
            denom_base = exp_avg_sq_[i];
        }

        auto denom = sqrt(denom_base) * scalar(1.0 / std::sqrt(bias_correction2))
                    + scalar(hp.eps);

        // Decoupled weight decay (AdamW)
        if (hp.weight_decay > 0) {
            param_hi = param_hi * scalar(1.0 - hp.lr * hp.weight_decay);
        }

        // Update parameters
        param_hi = param_hi - div(exp_avg_[i], denom) * scalar(step_size);
        param.tensor() = needs_upcast ? param_hi.to(param_dt) : param_hi;
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
            // R.16: half-precision params get Float32 state buffers.
            exp_avg_.push_back(make_optim_state(param->tensor()));
            exp_avg_sq_.push_back(make_optim_state(param->tensor()));
            if (amsgrad_) {
                max_exp_avg_sq_.push_back(make_optim_state(param->tensor()));
            }
        }
    }
}

// Audit K.1.
auto AdamW::on_parameters_appended_(size_t old_count, size_t new_count) -> void {
    exp_avg_.reserve(new_count);
    exp_avg_sq_.reserve(new_count);
    if (amsgrad_) max_exp_avg_sq_.reserve(new_count);
    for (size_t i = old_count; i < new_count; ++i) {
        const auto& param = parameters_[i];
        if (param) {
            // R.16: see AdamW::initialize_buffers.
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

    // P.2: num_params guard (see Adam::state_dict).
    state["num_params"] = Tensor({1}, DType::Int64, Device::cpu());
    state["num_params"].data<int64_t>()[0] = static_cast<int64_t>(parameters_.size());

    return state;
}

auto AdamW::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    // P.2: param-count guard up front.
    if (state.count("num_params")) {
        const int64_t expected = state.at("num_params").data<int64_t>()[0];
        if (expected != static_cast<int64_t>(parameters_.size())) {
            throw std::runtime_error(
                "AdamW::load_state_dict: parameter count mismatch - saved " +
                std::to_string(expected) + " but have " +
                std::to_string(parameters_.size()));
        }
    }
    // Audit item D.5: buffer-count guard matching Adam::load_state_dict.
    // Without this, a checkpoint with N exp_avg slots would silently
    // load into an AdamW with M ≠ N parameters and mis-align every
    // subsequent step.
    size_t saved_count = 0;
    for (const auto& [key, _] : state) {
        if (key.starts_with("exp_avg_") && !key.starts_with("exp_avg_sq_")) {
            ++saved_count;
        }
    }
    if (saved_count > 0 && saved_count != exp_avg_.size()) {
        throw std::runtime_error(
            "AdamW::load_state_dict: momentum buffer count mismatch - "
            "saved " + std::to_string(saved_count) + " but have " +
            std::to_string(exp_avg_.size()) + " parameters");
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

    if (state.count("amsgrad")) {
        amsgrad_ = state.at("amsgrad").data<int64_t>()[0] != 0;
    }

    // Y.18: if amsgrad flipped false -> true on load, allocate the
    // max_exp_avg_sq_ buffers now (the constructor only sized them when
    // amsgrad was true). Without this, the load loop below iterates an
    // empty vector and silently discards every saved max_exp_avg_sq_i.
    if (amsgrad_ && max_exp_avg_sq_.empty()) {
        max_exp_avg_sq_.reserve(parameters_.size());
        for (auto& param : parameters_) {
            if (param) {
                max_exp_avg_sq_.push_back(make_optim_state(param->tensor()));
            } else {
                max_exp_avg_sq_.push_back(Tensor{});
            }
        }
    }

    // V.27: cast to the R.16 master-weights dtype on load (see Adam::load_state_dict).
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

    // Load AMSGrad max second moment buffers
    for (size_t i = 0; i < max_exp_avg_sq_.size(); ++i) {
        std::string key = "max_exp_avg_sq_" + std::to_string(i);
        const DType state_dt = (i < parameters_.size() && parameters_[i])
            ? optim_state_dtype(parameters_[i]->tensor().dtype())
            : DType::Float32;
        if (state.count(key)) {
            max_exp_avg_sq_[i] = state.at(key).to(state_dt);
        }
    }
}

} // namespace tenzor::optim
