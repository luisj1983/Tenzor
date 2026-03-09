#include "tenzor/nn/optim/lamb.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include <cmath>

namespace tenzor::optim {

LAMB::LAMB(std::vector<std::shared_ptr<Variable>> params, double lr, double beta1,
           double beta2, double eps, double weight_decay)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), weight_decay_(weight_decay) {
    initialize_buffers();
}

auto LAMB::step_impl() -> void {
    step_count_++;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        const Tensor& grad = param.grad().value();

        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, param.tensor().dtype(), param.tensor().device());
        };

        // Update moment estimates (no weight decay in moment computation)
        exp_avg_[i] = exp_avg_[i] * scalar(beta1_) +
                     grad * scalar(1.0 - beta1_);
        exp_avg_sq_[i] = exp_avg_sq_[i] * scalar(beta2_) +
                        grad * grad * scalar(1.0 - beta2_);

        // Bias correction
        double bias_correction1 = 1.0 - std::pow(beta1_, step_count_);
        double bias_correction2 = 1.0 - std::pow(beta2_, step_count_);

        auto m_hat = exp_avg_[i] * scalar(1.0 / bias_correction1);
        auto v_hat = exp_avg_sq_[i] * scalar(1.0 / bias_correction2);

        // Adam update direction
        auto update = div(m_hat, sqrt(v_hat) + scalar(eps_));

        // Decoupled weight decay
        if (weight_decay_ > 0.0) {
            update = update + param.tensor() * scalar(weight_decay_);
        }

        // Compute trust ratio (LAMB scaling)
        // param_norm = ||param||_2, update_norm = ||update||_2
        auto param_norm_t = sqrt(sum(param.tensor() * param.tensor()));
        auto update_norm_t = sqrt(sum(update * update));

        double param_norm = 0.0;
        double update_norm = 0.0;
        // Read scalar values (move to CPU if needed)
        {
            auto pn = param_norm_t.to(Device::cpu());
            auto un = update_norm_t.to(Device::cpu());
            if (pn.dtype() == DType::Float32) {
                param_norm = static_cast<double>(pn.data<float>()[0]);
                update_norm = static_cast<double>(un.data<float>()[0]);
            } else {
                param_norm = pn.data<double>()[0];
                update_norm = un.data<double>()[0];
            }
        }

        double trust_ratio = 1.0;
        if (param_norm > 0.0 && update_norm > 0.0) {
            trust_ratio = param_norm / update_norm;
        }

        // Apply update with trust ratio
        param.tensor() = param.tensor() - update * scalar(lr_ * trust_ratio);
    }
}

auto LAMB::initialize_buffers() -> void {
    exp_avg_.clear();
    exp_avg_sq_.clear();
    for (auto& param : parameters_) {
        if (param) {
            exp_avg_.push_back(zeros_like(param->tensor()));
            exp_avg_sq_.push_back(zeros_like(param->tensor()));
        }
    }
}

auto LAMB::set_lr(double lr) -> void { lr_ = lr; }
auto LAMB::get_lr() const -> double { return lr_; }

auto LAMB::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;
    state["step_count"] = Tensor({1}, DType::Int64, Device::cpu());
    state["step_count"].data<int64_t>()[0] = step_count_;
    state["lr"] = Tensor({1}, DType::Float64, Device::cpu());
    state["lr"].data<double>()[0] = lr_;
    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        state["exp_avg_" + std::to_string(i)] = exp_avg_[i].clone();
        state["exp_avg_sq_" + std::to_string(i)] = exp_avg_sq_[i].clone();
    }
    return state;
}

auto LAMB::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    if (state.count("step_count"))
        step_count_ = state.at("step_count").data<int64_t>()[0];
    if (state.count("lr"))
        lr_ = state.at("lr").data<double>()[0];
    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        std::string ea_key = "exp_avg_" + std::to_string(i);
        std::string eas_key = "exp_avg_sq_" + std::to_string(i);
        if (state.count(ea_key)) exp_avg_[i] = state.at(ea_key).clone();
        if (state.count(eas_key)) exp_avg_sq_[i] = state.at(eas_key).clone();
    }
}

} // namespace tenzor::optim
