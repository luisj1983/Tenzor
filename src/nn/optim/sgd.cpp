#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/ops/creation.hpp"

namespace tenzor::optim {

SGD::SGD(std::vector<Variable*> params, double lr, double momentum,
        double dampening, double weight_decay, bool nesterov)
    : Optimizer(std::move(params)), lr_(lr), momentum_(momentum),
      dampening_(dampening), weight_decay_(weight_decay), nesterov_(nesterov) {
    initialize_buffers();
}

auto SGD::step() -> void {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param = *parameters_[i];
        if (!param.has_grad()) continue;

        auto grad = param.grad()->clone();

        // Weight decay
        if (weight_decay_ > 0.0) {
            grad = grad + param.tensor() * static_cast<float>(weight_decay_);
        }

        // Momentum
        if (momentum_ > 0.0) {
            auto& velocity = velocity_buffers_[i];
            velocity = velocity * static_cast<float>(momentum_) +
                      grad * static_cast<float>(1.0 - dampening_);

            if (nesterov_) {
                grad = grad + velocity * static_cast<float>(momentum_);
            } else {
                grad = velocity;
            }
        }

        // Update parameters
        param.tensor() = param.tensor() - grad * static_cast<float>(lr_);
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
    for (auto* param : parameters_) {
        velocity_buffers_.push_back(zeros_like(param->tensor()));
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

    // Load velocity buffers
    for (size_t i = 0; i < velocity_buffers_.size(); ++i) {
        std::string velocity_key = "velocity_" + std::to_string(i);
        if (state.count(velocity_key)) {
            velocity_buffers_[i] = state.at(velocity_key).clone();
        }
    }
}

} // namespace tenzor::optim
