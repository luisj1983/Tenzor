#include "tenzor/nn/optim/rprop.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/indexing.hpp"

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
            hp.lr = g->lr;
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

        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, param_tensor.dtype(), param_tensor.device());
        };

        auto grad = grad_tensor.clone();

        if (first_step_) {
            // First step: just use initial lr as step size, apply update
            param_tensor = param_tensor - sign(grad) * step_sizes_[i];
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
            new_steps = clamp(new_steps, static_cast<float>(hp.step_min), static_cast<float>(hp.step_max));
            step_sizes_[i] = new_steps;

            // Where sign flipped, zero out the gradient (don't use it this step)
            grad = where(neg_mask, scalar(0.0), grad);

            // Update parameters: x -= sign(grad) * step_size
            param_tensor = param_tensor - sign(grad) * step_sizes_[i];
        }

        // Store current gradient for next step
        prev_grads_[i] = grad.clone();

        // Note: hp.lr is captured for API symmetry with other optimisers but
        // Rprop's update does not consume the LR directly during a step --
        // it only seeds per-parameter `step_sizes_` at construction time.
        (void)hp.lr;
    }

    first_step_ = false;
}

auto Rprop::set_lr(double lr) -> void {
    lr_ = lr;
}

auto Rprop::get_lr() const -> double {
    return lr_;
}

auto Rprop::initialize_buffers() -> void {
    step_sizes_.clear();
    prev_grads_.clear();
    for (auto& param : parameters_) {
        if (param) {
            // Initialize step sizes to the initial learning rate
            step_sizes_.push_back(full_like(param->tensor(), lr_));
            prev_grads_.push_back(zeros_like(param->tensor()));
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

    return state;
}

auto Rprop::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
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

    // Load step size buffers
    for (size_t i = 0; i < step_sizes_.size(); ++i) {
        std::string key = "step_size_" + std::to_string(i);
        if (state.count(key)) {
            step_sizes_[i] = state.at(key).clone();
        }
    }

    // Load previous gradient buffers
    for (size_t i = 0; i < prev_grads_.size(); ++i) {
        std::string key = "prev_grad_" + std::to_string(i);
        if (state.count(key)) {
            prev_grads_[i] = state.at(key).clone();
        }
    }
}

} // namespace tenzor::optim
