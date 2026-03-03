/**
 * @file grad_scaler.cpp
 * @brief Implementation of gradient scaler for automatic mixed precision training
 */

#include "tenzor/nn/amp/grad_scaler.hpp"
#include "tenzor/nn/utils/clip_grad.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
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

    const float inv_scale = 1.0f / scale_;

    // Unscale all parameter gradients
    for (auto& param : optimizer.parameters()) {
        if (!param->has_grad()) {
            continue;
        }

        auto& grad = param->grad();
        if (!grad.has_value()) {
            continue;
        }

        // Clone gradient before modifying to avoid corrupting shared gradient data
        // (e.g., when retain_graph=true and backward is called multiple times)
        *grad = grad->clone() * inv_scale;
    }

    has_unscaled_ = true;
}

auto GradScaler::check_inf_nan_(const optim::Optimizer& optimizer) const -> bool {
    // TODO(perf): This implementation transfers every parameter's gradient to CPU
    // individually via .cpu(), which is extremely slow for models with many parameters
    // on GPU (e.g., hundreds of small tensors each triggering a synchronous D2H copy).
    // This should be replaced with a fused GPU kernel that:
    //   1. Launches a single kernel per parameter to check for inf/nan (isinf || isnan),
    //      writing a boolean flag to a device-side scalar.
    //   2. Uses a single cudaMemcpy to transfer the combined result back to host.
    // PyTorch uses _amp_foreach_non_finite_check_and_unscale_ which fuses the unscale
    // and inf/nan check into a single pass over all gradients.
    for (const auto& param : optimizer.parameters()) {
        if (!param->has_grad()) {
            continue;
        }

        const auto& grad = param->grad();
        if (!grad.has_value()) {
            continue;
        }

        // Copy gradient to CPU for inspection (handles CUDA tensors)
        const auto& grad_tensor = *grad;
        auto cpu_grad = grad_tensor.cpu();
        const int64_t numel = cpu_grad.numel();

        // Check each element for inf or nan based on dtype
        // BFloat16/Float16: convert to Float32 before scanning
        Tensor scan_grad = cpu_grad;
        if (scan_grad.dtype() == DType::BFloat16 || scan_grad.dtype() == DType::Float16) {
            scan_grad = scan_grad.to(DType::Float32);
        }

        if (scan_grad.dtype() == DType::Float64) {
            const double* data_ptr = scan_grad.data<double>();
            for (int64_t i = 0; i < numel; ++i) {
                const double val = data_ptr[i];
                if (std::isinf(val) || std::isnan(val)) {
                    return true;
                }
            }
        } else {
            // Default to Float32 (Float16/BFloat16 already converted above)
            const float* data_ptr = scan_grad.data<float>();
            for (int64_t i = 0; i < numel; ++i) {
                const float val = data_ptr[i];
                if (std::isinf(val) || std::isnan(val)) {
                    return true;
                }
            }
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
        return false;
    }

    // Perform optimizer step
    optimizer.step();

    // Reset unscale flag for next iteration
    has_unscaled_ = false;

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
    return nn::utils::clip_grad_norm_(
        std::vector<std::shared_ptr<Variable>>(params.begin(), params.end()),
        max_norm, norm_type
    );
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
