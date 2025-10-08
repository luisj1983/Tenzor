#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cmath>

namespace tenzor::optim {

// Adam::Adam implementation
Adam::Adam(std::vector<Variable*> params, double lr, double beta1,
          double beta2, double eps, double weight_decay, bool amsgrad)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), weight_decay_(weight_decay), amsgrad_(amsgrad) {
    initialize_buffers();
}

auto Adam::step() -> void {
    step_count_++;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param = *parameters_[i];
        if (!param.has_grad()) continue;

        auto grad = param.grad()->clone();

        // Weight decay
        if (weight_decay_ > 0.0) {
            grad = grad + param.tensor() * static_cast<float>(weight_decay_);
        }

        // Update biased first moment estimate
        exp_avg_[i] = exp_avg_[i] * static_cast<float>(beta1_) +
                     grad * static_cast<float>(1.0 - beta1_);

        // Update biased second raw moment estimate
        exp_avg_sq_[i] = exp_avg_sq_[i] * static_cast<float>(beta2_) +
                        grad * grad * static_cast<float>(1.0 - beta2_);

        // Bias correction
        double bias_correction1 = 1.0 - std::pow(beta1_, step_count_);
        double bias_correction2 = 1.0 - std::pow(beta2_, step_count_);

        double step_size = lr_ / bias_correction1;

        // Compute denominator
        auto denom = sqrt(exp_avg_sq_[i]) *
                    static_cast<float>(1.0 / std::sqrt(bias_correction2)) +
                    static_cast<float>(eps_);

        // Update parameters
        param.tensor() = param.tensor() -
                        div(exp_avg_[i], denom) * static_cast<float>(step_size);
    }
}

auto Adam::initialize_buffers() -> void {
    exp_avg_.clear();
    exp_avg_sq_.clear();

    for (auto* param : parameters_) {
        exp_avg_.push_back(zeros_like(param->tensor()));
        exp_avg_sq_.push_back(zeros_like(param->tensor()));
    }
}

} // namespace tenzor::optim
