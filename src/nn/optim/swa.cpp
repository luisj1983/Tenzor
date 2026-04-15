#include "tenzor/nn/optim/swa.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace tenzor::optim {

//==============================================================================
// AveragedModel Implementation
//==============================================================================

AveragedModel::AveragedModel(const std::vector<std::shared_ptr<Variable>>& params) {
    averaged_params_.reserve(params.size());
    for (const auto& param_ptr : params) {
        if (param_ptr) {
            averaged_params_.push_back(param_ptr->tensor().clone());
        } else {
            averaged_params_.emplace_back();
        }
    }
    n_averaged_ = 1;
}

auto AveragedModel::update_parameters(const std::vector<std::shared_ptr<Variable>>& params) -> void {
    if (params.size() != averaged_params_.size()) {
        throw std::runtime_error(
            "AveragedModel::update_parameters: parameter count mismatch - "
            "expected " + std::to_string(averaged_params_.size()) +
            " but got " + std::to_string(params.size()));
    }

    for (size_t i = 0; i < params.size(); ++i) {
        if (!params[i] || averaged_params_[i].numel() == 0) continue;

        const Tensor& param = params[i]->tensor();

        // avg = (avg * n + param) / (n + 1)
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, param.dtype(), param.device());
        };

        averaged_params_[i] = (averaged_params_[i] * scalar(static_cast<double>(n_averaged_))
                               + param) * scalar(1.0 / static_cast<double>(n_averaged_ + 1));
    }

    n_averaged_++;
}

auto AveragedModel::apply_to(std::vector<std::shared_ptr<Variable>>& params) const -> void {
    if (params.size() != averaged_params_.size()) {
        throw std::runtime_error(
            "AveragedModel::apply_to: parameter count mismatch - "
            "expected " + std::to_string(averaged_params_.size()) +
            " but got " + std::to_string(params.size()));
    }

    for (size_t i = 0; i < params.size(); ++i) {
        if (!params[i] || averaged_params_[i].numel() == 0) continue;
        params[i]->tensor() = averaged_params_[i].clone();
    }
}

//==============================================================================
// SWALR Implementation
//==============================================================================

SWALR::SWALR(Optimizer& optimizer, double swa_lr, int anneal_epochs,
             const std::string& anneal_strategy)
    : optimizer_(optimizer),
      swa_lr_(swa_lr),
      anneal_epochs_(anneal_epochs),
      anneal_strategy_(anneal_strategy) {

    if (anneal_strategy_ != "linear" && anneal_strategy_ != "cos") {
        throw std::invalid_argument(
            "SWALR: anneal_strategy must be 'linear' or 'cos', got '" +
            anneal_strategy_ + "'");
    }

    if (anneal_epochs_ < 0) {
        throw std::invalid_argument("SWALR: anneal_epochs must be non-negative");
    }

    initial_lr_ = optimizer_.get_lr();
    last_lr_ = initial_lr_;
}

auto SWALR::step() -> void {
    epoch_++;

    if (anneal_epochs_ == 0 || epoch_ >= anneal_epochs_) {
        // Past annealing phase: hold constant at swa_lr
        last_lr_ = swa_lr_;
    } else {
        // Annealing phase
        double t = static_cast<double>(epoch_) / static_cast<double>(anneal_epochs_);

        if (anneal_strategy_ == "cos") {
            // Cosine annealing: lr = swa_lr + (initial_lr - swa_lr) * (1 + cos(pi * t)) / 2
            last_lr_ = swa_lr_ + (initial_lr_ - swa_lr_) *
                       (1.0 + std::cos(std::numbers::pi * t)) / 2.0;
        } else {
            // Linear annealing: lr = initial_lr + t * (swa_lr - initial_lr)
            last_lr_ = initial_lr_ + t * (swa_lr_ - initial_lr_);
        }
    }

    optimizer_.set_lr(last_lr_);
}

} // namespace tenzor::optim
