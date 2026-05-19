#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/foreach.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"

namespace tenzor::optim {

SGD::SGD(std::vector<std::shared_ptr<Variable>> params, double lr, double momentum,
        double dampening, double weight_decay, bool nesterov)
    : Optimizer(std::move(params)), lr_(lr), momentum_(momentum),
      dampening_(dampening), weight_decay_(weight_decay), nesterov_(nesterov) {
    initialize_buffers();
}

auto SGD::step_impl() -> void {
    // Collect CPU parameters eligible for _foreach_* batch path.
    // CUDA params go through the fused kernel; all others are batched.
    std::vector<Tensor*> batch_params;
    std::vector<Tensor>  batch_grads;
    std::vector<size_t>  batch_indices;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        Tensor& param_tensor = param.tensor();
        const Tensor& grad_tensor = *param.grad();

        // ── CUDA path: fused single-kernel dispatch ──────────────────────────
        if (param_tensor.device().type == Device::Type::CUDA &&
            (param_tensor.dtype() == DType::Float32 || param_tensor.dtype() == DType::Float64)) {
            std::vector<Tensor> inputs = {param_tensor, grad_tensor};
            if (momentum_ > 0.0) {
                inputs.push_back(velocity_buffers_[i]);
            }
            NewOpAttributes attrs;
            attrs.set(AttrKey::Lr, lr_);
            attrs.set(AttrKey::Momentum, momentum_);
            attrs.set(AttrKey::WeightDecay, weight_decay_);
            attrs.set(AttrKey::Dampening, dampening_);
            attrs.set(AttrKey::Nesterov, nesterov_);
            dispatch(OpId::FusedSGDStep, inputs, attrs);
            continue;
        }

        // ── CPU: clone grad and apply weight decay ────────────────────────────
        auto g = grad_tensor.clone();
        if (weight_decay_ > 0.0) {
            g = g + param_tensor *
                full({1}, weight_decay_, param_tensor.dtype(), param_tensor.device());
        }
        batch_params.push_back(&param_tensor);
        batch_grads.push_back(std::move(g));
        batch_indices.push_back(i);
    }

    if (batch_params.empty()) return;

    if (momentum_ > 0.0) {
        // Momentum update: v = momentum*v + (1-dampening)*grad
        // This is NOT a lerp (lerp would be: v = (1-w)*v + w*grad).
        // Use per-param scalar tensors to preserve dtype precision.
        for (size_t k = 0; k < batch_params.size(); ++k) {
            size_t idx = batch_indices[k];
            auto& vel    = velocity_buffers_[idx];
            auto& p      = *batch_params[k];
            auto dtype   = p.dtype();
            auto dev     = p.device();

            vel = vel * full({1}, momentum_, dtype, dev) +
                  batch_grads[k] * full({1}, 1.0 - dampening_, dtype, dev);

            Tensor eff_grad = nesterov_
                ? batch_grads[k] + vel * full({1}, momentum_, dtype, dev)
                : vel;

            p = p - eff_grad * full({1}, lr_, dtype, dev);
        }
    } else {
        // No momentum: param -= lr * grad  — use _foreach_* batch path.
        std::vector<Tensor> params_view;
        params_view.reserve(batch_params.size());
        for (auto* p : batch_params) params_view.push_back(*p);

        std::vector<Tensor> lr_list;
        lr_list.reserve(batch_params.size());
        for (size_t k = 0; k < batch_params.size(); ++k) {
            auto dtype = batch_params[k]->dtype();
            auto dev   = batch_params[k]->device();
            lr_list.push_back(full({1}, lr_, dtype, dev));
        }
        // scaled[k] = lr * grad[k]
        auto scaled = foreach_mul(batch_grads, lr_list);
        // param[k] -= scaled[k]
        foreach_sub_(params_view, scaled);

        for (size_t k = 0; k < batch_params.size(); ++k)
            *batch_params[k] = params_view[k];
    }
}

auto SGD::set_lr(double lr) -> void {
    lr_ = lr;
}

auto SGD::get_lr() const -> double {
    return lr_;
}

auto SGD::initialize_buffers() -> void {
    velocity_buffers_.clear();
    for (auto& param : parameters_) {
        if (param) {
            velocity_buffers_.push_back(zeros_like(param->tensor()));
        }
    }
}

auto SGD::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    // Save optimizer configuration as tensors
    state["lr"] = Tensor({1}, DType::Float64, Device::cpu());
    state["lr"].data<double>()[0] = lr_;

    state["momentum"] = Tensor({1}, DType::Float64, Device::cpu());
    state["momentum"].data<double>()[0] = momentum_;

    state["dampening"] = Tensor({1}, DType::Float64, Device::cpu());
    state["dampening"].data<double>()[0] = dampening_;

    state["weight_decay"] = Tensor({1}, DType::Float64, Device::cpu());
    state["weight_decay"].data<double>()[0] = weight_decay_;

    state["nesterov"] = Tensor({1}, DType::Int64, Device::cpu());
    state["nesterov"].data<int64_t>()[0] = nesterov_ ? 1 : 0;

    // Save velocity buffers
    for (size_t i = 0; i < velocity_buffers_.size(); ++i) {
        state["velocity_" + std::to_string(i)] = velocity_buffers_[i].clone();
    }

    return state;
}

auto SGD::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    // Load optimizer configuration
    if (state.count("lr")) {
        lr_ = state.at("lr").data<double>()[0];
    }

    if (state.count("momentum")) {
        momentum_ = state.at("momentum").data<double>()[0];
    }

    if (state.count("dampening")) {
        dampening_ = state.at("dampening").data<double>()[0];
    }

    if (state.count("weight_decay")) {
        weight_decay_ = state.at("weight_decay").data<double>()[0];
    }

    if (state.count("nesterov")) {
        nesterov_ = state.at("nesterov").data<int64_t>()[0] != 0;
    }

    // Load velocity buffers
    for (size_t i = 0; i < velocity_buffers_.size(); ++i) {
        std::string velocity_key = "velocity_" + std::to_string(i);
        if (state.count(velocity_key)) {
            velocity_buffers_[i] = state.at(velocity_key).clone();
        }
    }
}

} // namespace tenzor::optim
