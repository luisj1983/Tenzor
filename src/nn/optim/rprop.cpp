#include "tenzor/nn/optim/rprop.hpp"
#include "tenzor/nn/optim/master_weights.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/indexing.hpp"

#include <stdexcept>
#include <string>

namespace tenzor::optim {

Rprop::Rprop(std::vector<std::shared_ptr<Variable>> params, double lr, double eta_minus,
             double eta_plus, double step_min, double step_max)
    : Optimizer(std::move(params)), lr_(lr), eta_minus_(eta_minus),
      eta_plus_(eta_plus), step_min_(step_min), step_max_(step_max) {
    initialize_buffers();
}

Rprop::Rprop(std::vector<optim::ParamGroup> groups,
             double default_lr, double default_eta_minus,
             double default_eta_plus, double default_step_min,
             double default_step_max)
    : Optimizer(std::move(groups)),
      lr_(default_lr),
      eta_minus_(default_eta_minus),
      eta_plus_(default_eta_plus),
      step_min_(default_step_min),
      step_max_(default_step_max) {
    initialize_buffers();
}

auto Rprop::step_impl() -> void {
    // Audit D.4: per-parameter hyperparameters resolve from the active
    // ParamGroup (when one was set up) or fall through to the optimiser-
    // wide defaults stored on this Rprop instance. Only `lr` has a
    // ParamGroup field; Rprop-specific knobs (eta_minus, eta_plus,
    // step_min, step_max) have no ParamGroup override and always fall
    // back to the member defaults.
    struct RpropHP {
        double lr;
        double eta_minus;
        double eta_plus;
        double step_min;
        double step_max;
    };

    auto resolve = [&](size_t i) -> RpropHP {
        RpropHP hp{lr_, eta_minus_, eta_plus_, step_min_, step_max_};
        if (const auto* g = find_group_for_param(i)) {
            hp.lr        = g->lr;
            // NN.15: honour per-ParamGroup Rprop overrides when present.
            hp.eta_minus = ParamGroup::or_else(g->eta_minus, eta_minus_);
            hp.eta_plus  = ParamGroup::or_else(g->eta_plus,  eta_plus_);
            hp.step_min  = ParamGroup::or_else(g->step_min,  step_min_);
            hp.step_max  = ParamGroup::or_else(g->step_max,  step_max_);
        }
        return hp;
    };

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        const RpropHP hp = resolve(i);

        Tensor& param_tensor = param.tensor();
        const Tensor& grad_tensor = *param.grad();

        // R.16: half-precision params use Float32 state buffers. Run math
        // in state_dt; cast param/grad on entry, cast result back on write.
        const DType param_dt = param_tensor.dtype();
        const DType state_dt = optim_state_dtype(param_dt);
        const bool needs_upcast = (state_dt != param_dt);

        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, state_dt, param_tensor.device());
        };

        Tensor param_hi = needs_upcast ? param_tensor.to(state_dt) : param_tensor;
        Tensor grad = needs_upcast ? grad_tensor.to(state_dt) : grad_tensor.clone();

        if (first_step_) {
            // First step: just use initial lr as step size, apply update
            param_hi = param_hi - sign(grad) * step_sizes_[i];
        } else {
            // Compute element-wise product of current and previous gradients
            auto sign_change = grad * prev_grads_[i];

            // Where sign is same (positive product): increase step size
            auto pos_mask = sign_change > scalar(0.0);
            // Where sign flipped (negative product): decrease step size
            auto neg_mask = sign_change < scalar(0.0);

            // Adapt step sizes
            auto new_steps = step_sizes_[i].clone();
            new_steps = where(pos_mask, new_steps * scalar(hp.eta_plus), new_steps);
            new_steps = where(neg_mask, new_steps * scalar(hp.eta_minus), new_steps);

            // Clamp step sizes
            // clamp() takes double bounds; pass hp.step_min/max directly so
            // Float64 step bounds aren't rounded to ~7 significant digits.
            new_steps = clamp(new_steps, hp.step_min, hp.step_max);
            step_sizes_[i] = new_steps;

            // Where sign flipped, zero out the gradient (don't use it this step)
            grad = where(neg_mask, scalar(0.0), grad);

            // Update parameters: x -= sign(grad) * step_size
            param_hi = param_hi - sign(grad) * step_sizes_[i];
        }

        // Store current gradient for next step
        prev_grads_[i] = grad.clone();

        param_tensor = needs_upcast ? param_hi.to(param_dt) : param_hi;

        // Note: hp.lr is captured for API symmetry with other optimisers but
        // Rprop's update does not consume the LR directly during a step --
        // it only seeds per-parameter `step_sizes_` at construction time.
        (void)hp.lr;
    }

    first_step_ = false;
}

auto Rprop::set_lr(double lr) -> void {
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

auto Rprop::get_lr() const -> double {
    return lr_;
}

auto Rprop::initialize_buffers() -> void {
    step_sizes_.clear();
    prev_grads_.clear();
    for (auto& param : parameters_) {
        if (param) {
            // R.16: half-precision params get Float32 state buffers.
            const auto& pt = param->tensor();
            const DType state_dt = optim_state_dtype(pt.dtype());
            std::vector<int64_t> shape(pt.shape().begin(), pt.shape().end());
            // Initialize step sizes to the initial learning rate
            step_sizes_.push_back(full(shape, lr_, state_dt, pt.device()));
            prev_grads_.push_back(make_optim_state(pt));
        }
    }
}

// Audit K.1: extend step_sizes_ / prev_grads_ for parameters appended
// via add_param_group.  Mirrors initialize_buffers — step_sizes start
// at lr_ (not zero) so the first sign-comparison step makes progress.
// Audit II.6 / HH.13 / EE.16: when a new ParamGroup overrides `lr`, use
// the group's lr (not the optimiser-wide `lr_`) so per-group learning
// rates take effect on the very first step. Mirrors Adam-family
// per-group resolution.
auto Rprop::on_parameters_appended_(size_t old_count, size_t new_count) -> void {
    step_sizes_.reserve(new_count);
    prev_grads_.reserve(new_count);
    for (size_t i = old_count; i < new_count; ++i) {
        const auto& param = parameters_[i];
        if (param) {
            // R.16: see Rprop::initialize_buffers for dtype rationale.
            const auto& pt = param->tensor();
            const DType state_dt = optim_state_dtype(pt.dtype());
            std::vector<int64_t> shape(pt.shape().begin(), pt.shape().end());
            double init_lr = lr_;
            if (const auto* g = find_group_for_param(i)) {
                init_lr = g->lr;
            }
            step_sizes_.push_back(full(shape, init_lr, state_dt, pt.device()));
            prev_grads_.push_back(make_optim_state(pt));
        } else {
            step_sizes_.push_back(Tensor{});
            prev_grads_.push_back(Tensor{});
        }
    }
}

auto Rprop::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    // Save optimizer configuration as tensors
    state["lr"] = Tensor({1}, DType::Float64, Device::cpu());
    state["lr"].data<double>()[0] = lr_;

    state["eta_minus"] = Tensor({1}, DType::Float64, Device::cpu());
    state["eta_minus"].data<double>()[0] = eta_minus_;

    state["eta_plus"] = Tensor({1}, DType::Float64, Device::cpu());
    state["eta_plus"].data<double>()[0] = eta_plus_;

    state["step_min"] = Tensor({1}, DType::Float64, Device::cpu());
    state["step_min"].data<double>()[0] = step_min_;

    state["step_max"] = Tensor({1}, DType::Float64, Device::cpu());
    state["step_max"].data<double>()[0] = step_max_;

    state["first_step"] = Tensor({1}, DType::Int64, Device::cpu());
    state["first_step"].data<int64_t>()[0] = first_step_ ? 1 : 0;

    // Save step size and previous gradient buffers
    for (size_t i = 0; i < step_sizes_.size(); ++i) {
        state["step_size_" + std::to_string(i)] = step_sizes_[i].clone();
    }
    for (size_t i = 0; i < prev_grads_.size(); ++i) {
        state["prev_grad_" + std::to_string(i)] = prev_grads_[i].clone();
    }

    // S.12 / P.2: num_params guard so load_state_dict can reject mismatched
    // models before any tensor is overwritten (mirrors Adam::state_dict).
    state["num_params"] = Tensor({1}, DType::Int64, Device::cpu());
    state["num_params"].data<int64_t>()[0] = static_cast<int64_t>(parameters_.size());

    return state;
}

auto Rprop::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    // S.12 / P.2: param-count guard up front. Reject a restore against the
    // wrong model loudly instead of partially populating per-parameter
    // buffers and then succeeding silently.
    if (state.count("num_params")) {
        const int64_t expected = state.at("num_params").data<int64_t>()[0];
        if (expected != static_cast<int64_t>(parameters_.size())) {
            throw std::runtime_error(
                "Rprop::load_state_dict: parameter count mismatch - saved " +
                std::to_string(expected) + " but have " +
                std::to_string(parameters_.size()));
        }
    }

    if (state.count("lr")) {
        lr_ = state.at("lr").data<double>()[0];
    }

    if (state.count("eta_minus")) {
        eta_minus_ = state.at("eta_minus").data<double>()[0];
    }

    if (state.count("eta_plus")) {
        eta_plus_ = state.at("eta_plus").data<double>()[0];
    }

    if (state.count("step_min")) {
        step_min_ = state.at("step_min").data<double>()[0];
    }

    if (state.count("step_max")) {
        step_max_ = state.at("step_max").data<double>()[0];
    }

    if (state.count("first_step")) {
        first_step_ = state.at("first_step").data<int64_t>()[0] != 0;
    }

    // V.27: cast to R.16 master-weights dtype on load.
    for (size_t i = 0; i < step_sizes_.size(); ++i) {
        std::string key = "step_size_" + std::to_string(i);
        const DType state_dt = (i < parameters_.size() && parameters_[i])
            ? optim_state_dtype(parameters_[i]->tensor().dtype())
            : DType::Float32;
        if (state.count(key)) {
            step_sizes_[i] = state.at(key).to(state_dt);
        }
    }

    for (size_t i = 0; i < prev_grads_.size(); ++i) {
        std::string key = "prev_grad_" + std::to_string(i);
        const DType state_dt = (i < parameters_.size() && parameters_[i])
            ? optim_state_dtype(parameters_[i]->tensor().dtype())
            : DType::Float32;
        if (state.count(key)) {
            prev_grads_[i] = state.at(key).to(state_dt);
        }
    }
}

} // namespace tenzor::optim
