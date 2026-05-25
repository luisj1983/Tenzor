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
    , has_unscaled_(false) {

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

auto GradScaler::scale(const Variable& loss) -> Variable {
    // Use Variable multiplication to preserve autograd graph
    // Raw Tensor multiplication would sever the computation graph
    auto scale_tensor = full({1}, static_cast<float>(scale_),
                             loss.dtype(), loss.device());
    Variable scale_var(scale_tensor, false);
    return loss * scale_var;
}

auto GradScaler::unscale_(optim::Optimizer& optimizer) -> void {
    if (has_unscaled_) {
        return;  // Already unscaled
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

    has_unscaled_ = true;
}

auto GradScaler::check_inf_nan_(const optim::Optimizer& optimizer) const -> bool {
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

        // Use fused kernel — stays on device, no per-param D2H transfer
        auto result = has_inf_nan(probe);
        // Single D2H transfer of 1 bool
        if (result.cpu().data<bool>()[0]) {
            return true;
        }
    }

    return false;
}

auto GradScaler::step(optim::Optimizer& optimizer) -> bool {
    // Unscale gradients if not already done
    if (!has_unscaled_) {
        unscale_(optimizer);
    }

    // Check for inf/nan in unscaled gradients
    found_inf_nan_ = check_inf_nan_(optimizer);

    if (found_inf_nan_) {
        // Skip optimizer step due to overflow
        has_unscaled_ = false;  // Reset for next iteration
        f32_unscaled_grads_.clear();  // Y.32: drop the side-table.
        return false;
    }

    // Perform optimizer step
    optimizer.step();

    // Reset unscale flag for next iteration
    has_unscaled_ = false;
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

            // Ensure scale doesn't become too large
            scale_ = std::min(scale_, static_cast<float>(1ULL << 24));

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
    has_unscaled_ = false;
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
    // Ensure gradients are unscaled before clipping
    if (!has_unscaled_) {
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
    // Ensure gradients are unscaled before clipping
    if (!has_unscaled_) {
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
