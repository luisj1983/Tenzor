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
            // S.13 / R.16: promote F16/BF16 to Float32 master copy.  The
            // averaged_params_ slot is allocated up front (matching the
            // current param shape/dtype-policy) but treated as uninitialised
            // storage until the first update_parameters() call installs the
            // first sample.  We seed with a clone to keep numel() consistent;
            // V.26 keeps n_averaged_ at 0 so update_parameters' first-call
            // branch overwrites this seed rather than averaging into it.
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
    // V.26: PyTorch starts at 0 and update_parameters installs the first
    // sample as a plain copy.  Audit-3 set this to 1 to count the ctor's
    // clone as the first sample, which made every subsequent average
    // off-by-one (and broke cross-framework checkpoint round-trips).
    n_averaged_ = 0;
}

auto AveragedModel::update_parameters(const std::vector<std::shared_ptr<Variable>>& params) -> void {
    if (params.size() != averaged_params_.size()) {
        throw std::runtime_error(
            "AveragedModel::update_parameters: parameter count mismatch - "
            "expected " + std::to_string(averaged_params_.size()) +
            " but got " + std::to_string(params.size()));
    }

    // V.26: PyTorch semantics — the very first call installs the parameter
    // as-is (count goes 0 -> 1); every subsequent call does the running mean
    // update with the previous count.  Old code seeded n_averaged_=1 in the
    // ctor and folded that into avg arithmetic, biasing every average by one
    // virtual sample equal to the construction-time parameter snapshot.
    const bool first_sample = (n_averaged_ == 0);

    for (size_t i = 0; i < params.size(); ++i) {
        if (!params[i] || averaged_params_[i].numel() == 0) continue;

        const Tensor& param = params[i]->tensor();
        // S.13: avg * n overflows F16/BF16 once n is large; do the math in
        // averaged_params_'s dtype (Float32 for half-precision params).
        const DType state_dt = averaged_params_[i].dtype();
        Tensor param_hi = (param.dtype() != state_dt) ? param.to(state_dt) : param;

        if (first_sample) {
            // First call: install the parameter directly (matches PyTorch's
            // `self.module = copy(model)` on n_averaged == 0 path).
            averaged_params_[i] = param_hi.clone();
            continue;
        }

        // LL.5: PyTorch convention for numerically stable running mean —
        // avg += (param - avg) / (n + 1). The (avg * n + param) / (n+1) form
        // overflows at large n because avg*n grows linearly without bound.
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, state_dt, param.device());
        };

        averaged_params_[i] = averaged_params_[i] +
            (param_hi - averaged_params_[i]) *
            scalar(1.0 / static_cast<double>(n_averaged_ + 1));
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
        // V.21: write into the live tensor in-place to preserve TensorImpl
        // identity (mirrors FSDP2 R.18 pattern). Reassigning param->tensor()
        // would swap the underlying TensorImpl and orphan any view that has
        // already captured the parameter (e.g. an FSDP flat_param view, or
        // an optimizer state buffer keyed by raw pointer).
        // S.13: cast back to param dtype on the source side when state is
        // held at Float32 master copy for F16/BF16 params.
        const DType param_dt = params[i]->tensor().dtype();
        Tensor src_cast = (averaged_params_[i].dtype() != param_dt)
                              ? averaged_params_[i].to(param_dt)
                              : averaged_params_[i];
        auto& dst = params[i]->tensor();
        dst.zero_();
        add_(dst, src_cast);
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
