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

} // namespace tenzor::optim
