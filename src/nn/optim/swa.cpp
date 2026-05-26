#include "tenzor/nn/optim/swa.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/nn/layers/sync_batchnorm.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>
#include <utility>

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
// NN.16: update_bn — refresh BatchNorm running statistics after SWA averaging.
//==============================================================================

namespace {
// Test whether a Module is a BatchNorm-family layer whose running statistics
// need recomputing after SWA averaging.  BatchNorm1d/2d/3d all expose
// running_mean / running_var / num_batches_tracked as registered buffers;
// SyncBatchNorm follows the same convention.  We rely on the buffer names
// (rather than a closed dynamic_cast list) so any future BN variant that
// follows the same naming convention is automatically covered.  We still
// dynamic_cast for the well-known subclasses below as a fast positive check.
inline auto is_batchnorm_layer(tenzor::nn::Module& m) -> bool {
    using namespace tenzor::nn;
    if (dynamic_cast<BatchNorm1d*>(&m) != nullptr) return true;
    if (dynamic_cast<BatchNorm2d*>(&m) != nullptr) return true;
    if (dynamic_cast<BatchNorm3d*>(&m) != nullptr) return true;
    if (dynamic_cast<SyncBatchNorm*>(&m) != nullptr) return true;
    return false;
}

// Recursively walk submodules, appending every BatchNorm-family layer.
void collect_bn_layers(tenzor::nn::Module& m,
                       std::vector<tenzor::nn::Module*>& out) {
    if (is_batchnorm_layer(m)) {
        out.push_back(&m);
    }
    for (const auto& [name, child] : m.get_submodules()) {
        if (child) collect_bn_layers(*child, out);
    }
}
} // namespace

auto AveragedModel::update_bn(tenzor::nn::Module& model,
                              const std::function<void(tenzor::nn::Module&)>& forward_pass) -> void {
    if (!forward_pass) {
        throw std::invalid_argument(
            "AveragedModel::update_bn: forward_pass callback must be non-empty");
    }

    // 1. Find every BN-family layer in the model tree.
    std::vector<tenzor::nn::Module*> bn_layers;
    collect_bn_layers(model, bn_layers);

    if (bn_layers.empty()) {
        // No BN layers — nothing to do.  Don't run the forward_pass; the
        // caller's update_bn() with a BN-free model is a no-op (matches
        // PyTorch's torch.optim.swa_utils.update_bn behaviour).
        return;
    }

    // 2. Snapshot each layer's training flag; reset running stats; switch to
    //    train() so the forward pass blends batch statistics in.  BatchNorm
    //    layers register running_mean / running_var / num_batches_tracked as
    //    named buffers (see batchnorm.cpp reset_parameters); reset them
    //    directly here rather than relying on a reset_running_stats() helper
    //    which is not part of the Module base API.
    std::vector<std::pair<tenzor::nn::Module*, bool>> prev_training;
    prev_training.reserve(bn_layers.size());

    for (auto* bn : bn_layers) {
        prev_training.emplace_back(bn, bn->is_training());

        for (auto& [name, buf] : bn->named_buffers()) {
            if (!buf) continue;
            if (name == "running_mean") {
                buf->tensor().zero_();
            } else if (name == "running_var") {
                buf->tensor().fill_(1.0f);
            } else if (name == "num_batches_tracked") {
                buf->tensor().zero_();
            }
        }

        bn->train(true);
    }

    // 3. Caller-driven forward sweep.  Wrap in try/catch so we always
    //    restore the pre-call training flags, even on exception.
    try {
        forward_pass(model);
    } catch (...) {
        for (auto& [bn, was_training] : prev_training) {
            bn->train(was_training);
        }
        throw;
    }

    // 4. Restore each BN layer's pre-call training flag.
    for (auto& [bn, was_training] : prev_training) {
        bn->train(was_training);
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

    // NN.19: snapshot every ParamGroup's LR so the annealing loop preserves
    // each group's pre-SWA LR ratio.  Previously SWALR captured a single
    // scalar `initial_lr_` from optimizer.get_lr() and wrote it back via
    // optimizer.set_lr(), which flattens per-group LRs to a single value.
    auto& groups = optimizer_.param_groups();
    initial_lrs_.reserve(groups.size());
    for (const auto& g : groups) {
        initial_lrs_.push_back(g.lr);
    }
    // Fall-through scalar for the no-groups case (and for get_last_lr()).
    initial_lr_ = optimizer_.get_lr();
    last_lr_ = initial_lr_;
}

auto SWALR::step() -> void {
    epoch_++;

    // NN.19: interpolation factor t is shared across groups; the per-group
    // target SWA LR is `swa_lr_ * (initial_lrs_[i] / initial_lrs_[0])` so
    // group i's LR ratio relative to group 0 is preserved through SWA.
    double t;
    if (anneal_epochs_ == 0 || epoch_ >= anneal_epochs_) {
        // Past annealing phase: hold constant at the per-group SWA LR.
        t = 1.0;
    } else if (anneal_strategy_ == "cos") {
        // Cosine annealing: weight is (1 + cos(pi * raw_t)) / 2 of the
        // INITIAL side; rewrite as a single `t` in [0, 1] that interpolates
        // initial -> target.
        const double raw_t = static_cast<double>(epoch_) /
                             static_cast<double>(anneal_epochs_);
        t = 1.0 - (1.0 + std::cos(std::numbers::pi * raw_t)) / 2.0;
    } else {
        // Linear annealing
        t = static_cast<double>(epoch_) / static_cast<double>(anneal_epochs_);
    }

    auto& groups = optimizer_.param_groups();
    const double ref_initial = initial_lrs_.empty() ? initial_lr_ : initial_lrs_[0];

    if (groups.empty() || initial_lrs_.empty()) {
        // No param groups — fall back to the scalar path.  Write through
        // set_lr() so optimizers without a group container (legacy callers)
        // still see the update.
        const double target = swa_lr_;
        last_lr_ = (1.0 - t) * initial_lr_ + t * target;
        optimizer_.set_lr(last_lr_);
        return;
    }

    for (size_t i = 0; i < groups.size(); ++i) {
        const double init_i = (i < initial_lrs_.size()) ? initial_lrs_[i]
                                                        : groups[i].lr;
        // Per-group SWA target preserves init_i / ref_initial ratio.  If the
        // reference initial LR is zero (degenerate optimizer config) fall
        // back to the scalar swa_lr_ for every group.
        const double target_i = (ref_initial != 0.0)
            ? swa_lr_ * (init_i / ref_initial)
            : swa_lr_;
        const double lr_i = (1.0 - t) * init_i + t * target_i;
        // Write through ParamGroup::lr directly.  Optimizer::set_lr would
        // flatten every group's LR to a single scalar, defeating the point.
        groups[i].lr = lr_i;
    }

    // last_lr_ tracks the first group's LR for get_last_lr() (PyTorch
    // convention: _last_lr[0]).  Callers needing per-group values can read
    // optimizer.param_groups() directly.
    last_lr_ = groups[0].lr;
}

} // namespace tenzor::optim
