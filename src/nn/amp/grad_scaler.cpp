/**
 * @file grad_scaler.cpp
 * @brief Implementation of gradient scaler for automatic mixed precision training
 */

#include "tenzor/nn/amp/grad_scaler.hpp"
#include "tenzor/nn/utils/clip_grad.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstring>

namespace tenzor {
namespace nn {
namespace amp {

GradScaler::GradScaler(float init_scale,
                       float growth_factor,
                       float backoff_factor,
                       int growth_interval)
    : scale_(init_scale)
    , init_scale_(init_scale)
    , growth_factor_(growth_factor)
    , backoff_factor_(backoff_factor)
    , growth_interval_(growth_interval)
    , growth_tracker_(0)
    , found_inf_nan_(false)
    , unscaled_for_() {

    if (init_scale <= 0.0f) {
        throw std::invalid_argument("init_scale must be positive");
    }
    if (growth_factor <= 1.0f) {
        throw std::invalid_argument("growth_factor must be greater than 1.0");
    }
    if (backoff_factor <= 0.0f || backoff_factor >= 1.0f) {
        throw std::invalid_argument("backoff_factor must be in range (0, 1)");
    }
    if (growth_interval <= 0) {
        throw std::invalid_argument("growth_interval must be positive");
    }
}

// audit-10 OO.7: detect nested `scaler.scale(loss)` calls issued from inside
// the closure passed to `scaler.step(opt, closure)`.  The outer step() runs
// unscale_() on the optimizer before invoking the closure; that sets the
// per-optimizer flag in `unscaled_for_`.  If the closure now calls
// scale(loss) and backward()s through it, the resulting scaled grads are
// added on top of the already-unscaled ones, silently corrupting the step.
// PyTorch raises a RuntimeError in this scenario; do the same here so the
// caller gets a fail-loud diagnostic instead of a numerically wrong update.
auto GradScaler::scale(const Variable& loss) -> Variable {
    if (!unscaled_for_.empty()) {
        throw std::runtime_error(
            "GradScaler::scale called while some optimisers are mid-unscale; "
            "nested scale inside loss closures is not supported (PyTorch parity)");
    }
    // Use Variable multiplication to preserve autograd graph
    // Raw Tensor multiplication would sever the computation graph
    auto scale_tensor = full({1}, static_cast<float>(scale_),
                             loss.dtype(), loss.device());
    Variable scale_var(scale_tensor, false);
    return loss * scale_var;
}

auto GradScaler::unscale_(optim::Optimizer& optimizer) -> void {
    // audit-8 GG.6: track per-optimizer.  The previous single bool aliased
    // across optimizers sharing this scaler — scaler.unscale_(opt_a) set
    // the flag, scaler.step(opt_b) short-circuited unscale_(opt_b), and
    // opt_b stepped with still-scaled grads.
    if (unscaled_for_.count(&optimizer) > 0) {
        return;  // Already unscaled for this optimizer this step
    }

    // audit-7 FF.16: scope the F32 unscaled-grad side-table to *this*
    // unscale_() invocation.  The side-table is keyed by raw param pointer,
    // but a GradScaler instance may be shared across multiple optimizers
    // (or the same optimizer may have param_groups added/removed between
    // steps).  Relying on the trailing clear in step() leaves stale entries
    // visible to check_inf_nan_() if a caller invokes unscale_() directly
    // for a different optimizer without an intervening step().  Clear here
    // before rebuilding the table for the current optimizer's params.
    f32_unscaled_grads_.clear();

    // U.8: build inv_scale in Float32. For scale = 2^17 the F16
    // representation of 1/scale ≈ 7.6e-6 is denormal (rounds to zero),
    // so the F16/BF16 path below ALWAYS upcasts the grad to Float32,
    // multiplies by an F32 inv_scale tensor, then casts back to the grad
    // dtype. Mirrors PyTorch's _amp_foreach_non_finite_check_and_unscale_,
    // which always runs the unscale arithmetic in F32 regardless of grad
    // dtype.
    const float inv_scale_f32 = 1.0f / scale_;

    // Unscale all parameter gradients
    for (auto& param : optimizer.parameters()) {
        if (!param->has_grad()) {
            continue;
        }

        const auto& grad = param->grad();
        if (!grad.has_value()) {
            continue;
        }

        const DType grad_dt = grad->dtype();
        // Clone gradient before modifying to avoid corrupting shared gradient
        // data (e.g., when retain_graph=true and backward is called multiple
        // times).
        if (grad_dt == DType::Float16 || grad_dt == DType::BFloat16) {
            // U.8: upcast to F32 for the multiply, then cast back.
            auto inv_scale = full({1}, inv_scale_f32,
                                  DType::Float32, grad->device());
            Tensor unscaled = grad->to(DType::Float32) * inv_scale;
            // Y.32: stash the F32 unscaled tensor in a side-table keyed by
            // the param's raw pointer so ``check_inf_nan_`` can read it
            // back. Storing the cast-back F16/BF16 on the Variable means
            // the optimizer's ``step()`` sees the right dtype, but values
            // that fit in F32 yet round to inf on cast-back must NOT
            // trigger a spurious overflow event — those should be detected
            // against the pre-cast F32 representation here.
            f32_unscaled_grads_[param.get()] = unscaled;
            param->set_grad(unscaled.to(grad_dt));
        } else {
            auto inv_scale = full({1}, inv_scale_f32,
                                  grad_dt, grad->device());
            param->set_grad(grad->clone() * inv_scale);
        }
    }

    unscaled_for_.insert(&optimizer);  // audit-8 GG.6
}

auto GradScaler::check_inf_nan_(const optim::Optimizer& optimizer) const -> bool {
    // Accumulate the per-parameter inf/nan flags into a single DEVICE tensor and
    // perform exactly ONE device->host sync at the end (PyTorch's found_inf
    // pattern), instead of a D2H copy + implicit stream sync per parameter which
    // serialized the step on many tiny transfers for large models.
    std::optional<Tensor> found_inf;
    for (const auto& param : optimizer.parameters()) {
        if (!param->has_grad()) {
            continue;
        }

        const auto& grad = param->grad();
        if (!grad.has_value()) {
            continue;
        }

        // Y.32: when the original grad dtype was F16/BF16, the cast-back
        // copy stored on the Variable may have rounded values that fit in
        // F32 up to inf. Check the F32 side-table entry — the genuine
        // post-unscale value — instead. A finite F32 unscaled grad should
        // not trigger backoff just because its half-precision representation
        // saturated.
        auto it = f32_unscaled_grads_.find(param.get());
        const Tensor& probe = (it != f32_unscaled_grads_.end())
                                  ? it->second
                                  : *grad;

        // has_inf_nan stays on device; OR the flags together on-device.
        Tensor flag = has_inf_nan(probe).to(DType::Float32);
        found_inf = found_inf.has_value() ? (*found_inf + flag) : flag;
    }

    if (!found_inf.has_value()) {
        return false;
    }
    // The single host sync for the whole optimizer.
    return found_inf->cpu().data<float>()[0] > 0.5f;
}

auto GradScaler::step(optim::Optimizer& optimizer) -> bool {
    // Unscale gradients if not already done for this optimizer (GG.6).
    if (unscaled_for_.count(&optimizer) == 0) {
        unscale_(optimizer);
    }

    // KK.16: check for inf/nan in unscaled gradients.  Accumulate across
    // optimizers that share this scaler — a wholesale assignment would let
    // step(B) overwrite step(A)'s true with false, and update() would then
    // grow the scale instead of backing off.  The flag is reset only in
    // update() (or reset()), which marks the end of the iteration.
    const bool optimizer_has_inf_nan = check_inf_nan_(optimizer);
    found_inf_nan_ = found_inf_nan_ || optimizer_has_inf_nan;

    if (optimizer_has_inf_nan) {
        // Skip optimizer step due to overflow.  Clear *only this optimizer's*
        // unscale flag; siblings sharing the scaler keep their state intact.
        unscaled_for_.erase(&optimizer);
        f32_unscaled_grads_.clear();  // Y.32: drop the side-table.
        return false;
    }

    // Perform optimizer step
    optimizer.step();

    // Reset unscale flag for this optimizer; siblings unaffected (GG.6).
    unscaled_for_.erase(&optimizer);
    f32_unscaled_grads_.clear();  // Y.32: side-table only lives for one step.

    return true;
}

auto GradScaler::update() -> void {
    if (found_inf_nan_) {
        // Overflow detected: decrease scale
        scale_ *= backoff_factor_;

        // Ensure scale doesn't become too small
        scale_ = std::max(scale_, 1.0f);

        // Reset growth tracker
        growth_tracker_ = 0;

        // Reset overflow flag
        found_inf_nan_ = false;
    } else {
        // No overflow: increment growth tracker
        growth_tracker_++;

        // Increase scale if we've reached growth interval
        if (growth_tracker_ >= growth_interval_) {
            scale_ *= growth_factor_;

            // Cap growth only to prevent the NEXT multiply from overflowing to
            // +Inf. The old fixed 2^24 (~1.6e7) cap sat below common init scales
            // (PyTorch defaults to 2^16 and grows well past 2^24), silently
            // clamping the scale and creating an asymmetry with reset()'s
            // init_scale_. This bound is always >= any realistic init_scale_.
            const float max_scale =
                std::numeric_limits<float>::max() / growth_factor_;
            scale_ = std::min(scale_, max_scale);

            // Reset growth tracker
            growth_tracker_ = 0;
        }
    }
}

auto GradScaler::get_scale() const -> float {
    return scale_;
}

auto GradScaler::get_growth_tracker() const -> int {
    return growth_tracker_;
}

auto GradScaler::found_inf_nan() const -> bool {
    return found_inf_nan_;
}

auto GradScaler::reset() -> void {
    // Reset to initial state using stored init_scale_
    scale_ = init_scale_;
    growth_tracker_ = 0;
    found_inf_nan_ = false;
    unscaled_for_.clear();  // audit-8 GG.6
    f32_unscaled_grads_.clear();  // Y.32
}

auto GradScaler::state_dict() const -> std::unordered_map<std::string, float> {
    return {
        {"scale", scale_},
        {"growth_factor", growth_factor_},
        {"backoff_factor", backoff_factor_},
        {"growth_interval", static_cast<float>(growth_interval_)},
        {"growth_tracker", static_cast<float>(growth_tracker_)}
    };
}

auto GradScaler::clip_grad_norm_(optim::Optimizer& optimizer, double max_norm, double norm_type) -> double {
    // Ensure gradients are unscaled before clipping (GG.6).
    if (unscaled_for_.count(&optimizer) == 0) {
        unscale_(optimizer);
    }

    // Collect parameters and delegate to the utility function
    // Use a mutable copy of the parameter vector since clip_grad_norm_ takes by value
    auto params = optimizer.parameters();
    double result = nn::utils::clip_grad_norm_(
        std::vector<std::shared_ptr<Variable>>(params.begin(), params.end()),
        max_norm, norm_type
    );

    // BB.13: Y.32 introduced the F32 unscaled-grad side-table that
    // ``check_inf_nan_`` consults when the param's grad is F16/BF16. The
    // clip utility just mutated ``param->grad()`` in place — the F32 entry
    // now holds the pre-clip value and would either (a) report a stale
    // overflow that was actually clipped away, or (b) hide a freshly
    // introduced clipped-to-inf value. Re-sync the side-table from the
    // post-clip grad so check_inf_nan_ sees the same numbers the optimizer
    // will step against.
    for (auto& param : optimizer.parameters()) {
        if (!param || !param->has_grad()) continue;
        if (f32_unscaled_grads_.find(param.get()) == f32_unscaled_grads_.end()) {
            continue;
        }
        f32_unscaled_grads_[param.get()] = param->grad().value().to(DType::Float32);
    }

    return result;
}

auto GradScaler::clip_grad_value_(optim::Optimizer& optimizer, double clip_value) -> void {
    // Ensure gradients are unscaled before clipping (GG.6).
    if (unscaled_for_.count(&optimizer) == 0) {
        unscale_(optimizer);
    }

    // Collect parameters and delegate to the utility function
    auto params = optimizer.parameters();
    nn::utils::clip_grad_value_(
        std::vector<std::shared_ptr<Variable>>(params.begin(), params.end()),
        clip_value
    );

    // BB.13: re-sync the F32 side-table after the in-place value clip — see
    // the comment in clip_grad_norm_ above for the rationale.
    for (auto& param : optimizer.parameters()) {
        if (!param || !param->has_grad()) continue;
        if (f32_unscaled_grads_.find(param.get()) == f32_unscaled_grads_.end()) {
            continue;
        }
        f32_unscaled_grads_[param.get()] = param->grad().value().to(DType::Float32);
    }
}

auto GradScaler::load_state_dict(const std::unordered_map<std::string, float>& state) -> void {
    if (state.count("scale")) {
        scale_ = state.at("scale");
    }
    if (state.count("growth_factor")) {
        growth_factor_ = state.at("growth_factor");
    }
    if (state.count("backoff_factor")) {
        backoff_factor_ = state.at("backoff_factor");
    }
    if (state.count("growth_interval")) {
        growth_interval_ = static_cast<int>(state.at("growth_interval"));
    }
    if (state.count("growth_tracker")) {
        growth_tracker_ = static_cast<int>(state.at("growth_tracker"));
    }
}

} // namespace amp
} // namespace nn
} // namespace tenzor
