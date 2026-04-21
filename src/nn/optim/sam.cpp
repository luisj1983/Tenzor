#include "tenzor/nn/optim/sam.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include <cmath>
#include <stdexcept>

namespace tenzor::optim {

SAM::SAM(std::shared_ptr<Optimizer> base_optimizer, double rho)
    : Optimizer(base_optimizer->parameters()),
      base_optimizer_(std::move(base_optimizer)),
      rho_(rho) {
    if (rho_ < 0.0) {
        throw std::invalid_argument("SAM: rho must be non-negative");
    }
}

auto SAM::first_step() -> void {
    // Compute global gradient norm across all parameters
    double grad_norm_sq = 0.0;

    for (auto& param_ptr : parameters_) {
        if (!param_ptr || !param_ptr->has_grad()) continue;
        const Tensor& grad = param_ptr->grad().value();
        // Sum of squared elements
        auto norm_sq = sum(grad * grad);
        // Move to CPU before reading host-side; data<T>() on a GPU tensor returns
        // a device pointer and dereferencing it from host code segfaults.
        auto norm_sq_cpu = norm_sq.to(DType::Float64).to(Device::cpu());
        grad_norm_sq += static_cast<double>(norm_sq_cpu.data<double>()[0]);
    }

    double grad_norm = std::sqrt(grad_norm_sq) + 1e-12;  // Avoid division by zero
    double scale = rho_ / grad_norm;

    // Compute and apply perturbation for each parameter
    epsilon_.clear();
    epsilon_.reserve(parameters_.size());

    for (auto& param_ptr : parameters_) {
        if (!param_ptr || !param_ptr->has_grad()) {
            epsilon_.emplace_back();  // Empty tensor placeholder
            continue;
        }

        const Tensor& grad = param_ptr->grad().value();

        // epsilon_i = rho * grad_i / ||grad||_2
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, param_ptr->tensor().dtype(), param_ptr->tensor().device());
        };

        Tensor eps = grad * scalar(scale);
        epsilon_.push_back(eps.clone());

        // Perturb weights: w = w + epsilon
        param_ptr->tensor() = param_ptr->tensor() + eps;
    }
}

auto SAM::second_step() -> void {
    if (epsilon_.empty()) {
        throw std::runtime_error("SAM::second_step: first_step() must be called first");
    }

    // Restore original weights: w = w - epsilon
    for (size_t i = 0; i < parameters_.size(); ++i) {
        if (!parameters_[i] || epsilon_[i].numel() == 0) continue;
        parameters_[i]->tensor() = parameters_[i]->tensor() - epsilon_[i];
    }

    // Step the base optimizer with the gradients from the perturbed point
    base_optimizer_->step();

    // Clear stored perturbations
    epsilon_.clear();
}

auto SAM::step_impl() -> void {
    throw std::runtime_error(
        "SAM::step_impl: Direct step() is not supported. "
        "Use first_step() followed by forward+backward, then second_step().");
}

auto SAM::set_lr(double lr) -> void {
    base_optimizer_->set_lr(lr);
}

auto SAM::get_lr() const -> double {
    return base_optimizer_->get_lr();
}

auto SAM::state_dict() const -> std::unordered_map<std::string, Tensor> {
    auto state = base_optimizer_->state_dict();

    // Add SAM-specific state
    state["sam_rho"] = Tensor({1}, DType::Float64, Device::cpu());
    state["sam_rho"].data<double>()[0] = rho_;

    return state;
}

auto SAM::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    if (state.count("sam_rho")) {
        rho_ = state.at("sam_rho").data<double>()[0];
    }

    // Forward the rest to the base optimizer
    base_optimizer_->load_state_dict(state);
}

} // namespace tenzor::optim
