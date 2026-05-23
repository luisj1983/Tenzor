#include "tenzor/nn/optim/swa.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace tenzor::optim {

namespace {
// S.13 / R.16: avg * n at n >= 512 overflows Float16 (max ~65504).  Hold the
// running mean in Float32 when params are F16/BF16; cast back on assignment.
inline auto swa_state_dtype(DType param_dtype) -> DType {
    if (param_dtype == DType::Float16 || param_dtype == DType::BFloat16) {
        return DType::Float32;
    }
    return param_dtype;
}
} // namespace

//==============================================================================
// AveragedModel Implementation
//==============================================================================

AveragedModel::AveragedModel(const std::vector<std::shared_ptr<Variable>>& params) {
    averaged_params_.reserve(params.size());
    for (const auto& param_ptr : params) {
        if (param_ptr) {
            // S.13 / R.16: promote F16/BF16 to Float32 master copy.
            const DType state_dt = swa_state_dtype(param_ptr->tensor().dtype());
            if (state_dt != param_ptr->tensor().dtype()) {
                averaged_params_.push_back(param_ptr->tensor().to(state_dt));
            } else {
                averaged_params_.push_back(param_ptr->tensor().clone());
            }
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
        // S.13: avg * n overflows F16/BF16 once n is large; do the math in
        // averaged_params_'s dtype (Float32 for half-precision params).
        const DType state_dt = averaged_params_[i].dtype();
        Tensor param_hi = (param.dtype() != state_dt) ? param.to(state_dt) : param;

        // avg = (avg * n + param) / (n + 1)
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, state_dt, param.device());
        };

        averaged_params_[i] = (averaged_params_[i] * scalar(static_cast<double>(n_averaged_))
                               + param_hi) * scalar(1.0 / static_cast<double>(n_averaged_ + 1));
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
        // S.13: cast back to param dtype on assignment when state is upcast.
        const DType param_dt = params[i]->tensor().dtype();
        if (averaged_params_[i].dtype() != param_dt) {
            params[i]->tensor() = averaged_params_[i].to(param_dt);
        } else {
            params[i]->tensor() = averaged_params_[i].clone();
        }
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
