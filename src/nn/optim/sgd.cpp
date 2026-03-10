#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/ops/creation.hpp"
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
    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        Tensor& param_tensor = param.tensor();
        const Tensor& grad_tensor = *param.grad();

        // Use fused CUDA kernel for CUDA tensors (single kernel launch!)
        if (param_tensor.device().type == Device::Type::CUDA &&
            (param_tensor.dtype() == DType::Float32 || param_tensor.dtype() == DType::Float64)) {
            // Prepare inputs for dispatch
            std::vector<Tensor> inputs = {param_tensor, grad_tensor};
            if (momentum_ > 0.0) {
                inputs.push_back(velocity_buffers_[i]);
            }

            // Prepare attributes
            NewOpAttributes attrs;
            attrs.set(AttrKey::Lr, lr_);
            attrs.set(AttrKey::Momentum, momentum_);
            attrs.set(AttrKey::WeightDecay, weight_decay_);
            attrs.set(AttrKey::Dampening, dampening_);
            attrs.set(AttrKey::Nesterov, nesterov_);

            // Dispatch to fused kernel (modifies param and momentum buffer in-place)
            dispatch(OpId::FusedSGDStep, inputs, attrs);
            continue;
        }

        // CPU fallback: use tensor operations
        // Use dtype-appropriate scalar tensors to preserve Float64 precision
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, param_tensor.dtype(), param_tensor.device());
        };

        auto grad = grad_tensor.clone();

        // Weight decay
        if (weight_decay_ > 0.0) {
            grad = grad + param_tensor * scalar(weight_decay_);
        }

        // Momentum
        if (momentum_ > 0.0) {
            auto& velocity = velocity_buffers_[i];
            velocity = velocity * scalar(momentum_) +
                      grad * scalar(1.0 - dampening_);

            if (nesterov_) {
                grad = grad + velocity * scalar(momentum_);
            } else {
                grad = velocity;
            }
        }

        // Update parameters
        param_tensor = param_tensor - grad * scalar(lr_);
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
