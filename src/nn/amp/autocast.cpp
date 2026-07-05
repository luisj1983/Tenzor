/**
 * @file autocast.cpp
 * @brief Implementation of automatic mixed precision context manager
 */

#include "tenzor/nn/amp/autocast.hpp"
#include "tenzor/nn/amp/autocast_interceptor.hpp"
#include <algorithm>

namespace tenzor {
namespace nn {
namespace amp {

// ============================================================================
// AutocastPolicyRegistry
// ============================================================================

AutocastPolicyRegistry::AutocastPolicyRegistry()
    : compute_heavy_ops_{
        "matmul", "mm", "bmm", "addmm", "baddbmm",
        "conv1d", "conv2d", "conv3d",
        "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
        "linear", "gru", "lstm", "rnn", "transformer"
    }
    , stability_critical_ops_{
        "softmax", "log_softmax", "cross_entropy", "nll_loss",
        "batch_norm", "layer_norm", "group_norm", "instance_norm",
        "normalize", "sum", "mean", "var", "std", "prod",
        "cumsum", "cumprod", "amax", "amin", "argmax", "argmin"
    } {}

auto AutocastPolicyRegistry::instance() -> AutocastPolicyRegistry& {
    static AutocastPolicyRegistry registry;
    return registry;
}

auto AutocastPolicyRegistry::register_compute_heavy(const std::string& op_name) -> void {
    std::unique_lock lock(mutex_);
    stability_critical_ops_.erase(op_name);
    compute_heavy_ops_.insert(op_name);
    override_count_.store(compute_heavy_ops_.size() + stability_critical_ops_.size(),
                          std::memory_order_release);
}

auto AutocastPolicyRegistry::register_stability_critical(const std::string& op_name) -> void {
    std::unique_lock lock(mutex_);
    compute_heavy_ops_.erase(op_name);
    stability_critical_ops_.insert(op_name);
    override_count_.store(compute_heavy_ops_.size() + stability_critical_ops_.size(),
                          std::memory_order_release);
}

auto AutocastPolicyRegistry::unregister(const std::string& op_name) -> void {
    std::unique_lock lock(mutex_);
    compute_heavy_ops_.erase(op_name);
    stability_critical_ops_.erase(op_name);
    override_count_.store(compute_heavy_ops_.size() + stability_critical_ops_.size(),
                          std::memory_order_release);
}

auto AutocastPolicyRegistry::is_compute_heavy(const std::string& op_name) const -> bool {
    std::shared_lock lock(mutex_);
    return compute_heavy_ops_.count(op_name) > 0;
}

auto AutocastPolicyRegistry::is_stability_critical(const std::string& op_name) const -> bool {
    std::shared_lock lock(mutex_);
    return stability_critical_ops_.count(op_name) > 0;
}

// ============================================================================
// Autocast
// ============================================================================

// Thread-local state initialization
thread_local bool Autocast::enabled_ = false;
thread_local std::optional<DType> Autocast::dtype_ = std::nullopt;
thread_local std::optional<Device::Type> Autocast::device_type_ = std::nullopt;

Autocast::Autocast(bool enabled, DType dtype, std::optional<Device::Type> device_type)
    : prev_enabled_(enabled_)
    , prev_dtype_(dtype_)
    , prev_device_type_(device_type_) {

    // Validate dtype
    if (enabled && dtype != DType::Float16 && dtype != DType::BFloat16) {
        throw std::invalid_argument(
            "Autocast: dtype must be Float16 or BFloat16"
        );
    }

    // Set new state
    enabled_ = enabled;
    if (enabled) {
        dtype_ = dtype;
        device_type_ = device_type;
        // Push autocast interceptor onto the dispatch stack
        DispatchInterceptorStack::push(nn::amp::make_autocast_interceptor());
        pushed_interceptor_ = true;
    } else {
        dtype_ = std::nullopt;
        device_type_ = std::nullopt;
    }
}

Autocast::~Autocast() {
    // Pop autocast interceptor if we pushed one
    if (pushed_interceptor_) {
        DispatchInterceptorStack::pop();
    }

    // Restore previous state
    enabled_ = prev_enabled_;
    dtype_ = prev_dtype_;
    device_type_ = prev_device_type_;
}

auto Autocast::is_enabled() -> bool {
    return enabled_;
}

auto Autocast::get_dtype() -> std::optional<DType> {
    return dtype_;
}

auto Autocast::get_device_type() -> std::optional<Device::Type> {
    return device_type_;
}

auto Autocast::should_autocast(const std::string& op_name, const Device& device) -> bool {
    // Autocast must be enabled
    if (!enabled_) {
        return false;
    }

    // Check if device matches
    if (device_type_.has_value() && device.type != device_type_.value()) {
        return false;
    }

    // Only skip CPU autocast if no device type is specified (backward compat)
    // With device_type unset (the default), autocast applies to the op's actual
    // device — CPU included — so a device-agnostic Autocast scope behaves the
    // same on CPU and CUDA. An explicit device_type (handled above) restricts it.

    // Stability-critical operations should not be autocast
    if (is_stability_critical_op(op_name)) {
        return false;
    }

    // Compute-heavy operations should be autocast
    if (is_compute_heavy_op(op_name)) {
        return true;
    }

    // By default, don't autocast other operations
    return false;
}

auto Autocast::get_autocast_dtype(const std::string& op_name, DType input_dtype) -> DType {
    if (!enabled_ || !dtype_.has_value()) {
        return input_dtype;
    }

    // If input is already in lower precision, keep it
    if (input_dtype == DType::Float16 || input_dtype == DType::BFloat16) {
        return input_dtype;
    }

    // Float64 is preserved - if user is using Float64, they need the higher precision
    // Only Float32 gets autocast to the target dtype (Float16/BFloat16)
    if (input_dtype == DType::Float32) {
        // Probe should_autocast with this scope's own target device, not a
        // hardcoded cuda(0). should_autocast() returns false when the queried
        // device type does not match device_type_, so passing cuda(0) for a
        // CPU/ROCm/OneAPI scope would always report "no autocast" and silently
        // leave Float32 untouched. When no device type is pinned, fall back to
        // a CPU device (should_autocast handles the backward-compat gate).
        Device target_device = device_type_.has_value()
                                   ? Device{device_type_.value(), 0}
                                   : Device::cpu();
        if (should_autocast(op_name, target_device)) {
            return dtype_.value();
        }
    }

    // For other dtypes (integers, etc.), keep original
    return input_dtype;
}

auto Autocast::is_compute_heavy_op(const std::string& op_name) -> bool {
    return AutocastPolicyRegistry::instance().is_compute_heavy(op_name);
}

auto Autocast::is_stability_critical_op(const std::string& op_name) -> bool {
    return AutocastPolicyRegistry::instance().is_stability_critical(op_name);
}

} // namespace amp
} // namespace nn
} // namespace tenzor
