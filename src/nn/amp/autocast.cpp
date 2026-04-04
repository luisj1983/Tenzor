/**
 * @file autocast.cpp
 * @brief Implementation of automatic mixed precision context manager
 */

#include "tenzor/nn/amp/autocast.hpp"
#include "tenzor/nn/amp/autocast_interceptor.hpp"
#include <unordered_set>
#include <algorithm>

namespace tenzor {
namespace nn {
namespace amp {

// Thread-local state initialization
thread_local bool Autocast::enabled_ = false;
thread_local std::optional<DType> Autocast::dtype_ = std::nullopt;
thread_local std::optional<Device::Type> Autocast::device_type_ = std::nullopt;

// Operations that benefit from lower precision (compute-heavy)
static const std::unordered_set<std::string> COMPUTE_HEAVY_OPS = {
    "matmul",
    "mm",
    "bmm",
    "addmm",
    "baddbmm",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "linear",
    "gru",
    "lstm",
    "rnn",
    "transformer"
};

// Operations that need higher precision for numerical stability
static const std::unordered_set<std::string> STABILITY_CRITICAL_OPS = {
    "softmax",
    "log_softmax",
    "cross_entropy",
    "nll_loss",
    "batch_norm",
    "layer_norm",
    "group_norm",
    "instance_norm",
    "normalize",
    "sum",
    "mean",
    "var",
    "std",
    "prod",
    "cumsum",
    "cumprod",
    "amax",
    "amin",
    "argmax",
    "argmin"
};

Autocast::Autocast(bool enabled, DType dtype, Device::Type device_type)
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
    // When user explicitly enables CPU autocast, allow it (BFloat16 benefits on
    // recent Intel CPUs with AMX/AVX-512 BF16 instructions)
    if (device.type == Device::Type::CPU && !device_type_.has_value()) {
        return false;
    }

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
        if (should_autocast(op_name, Device::cuda(0))) {
            return dtype_.value();
        }
    }

    // For other dtypes (integers, etc.), keep original
    return input_dtype;
}

auto Autocast::is_compute_heavy_op(const std::string& op_name) -> bool {
    return COMPUTE_HEAVY_OPS.find(op_name) != COMPUTE_HEAVY_OPS.end();
}

auto Autocast::is_stability_critical_op(const std::string& op_name) -> bool {
    return STABILITY_CRITICAL_OPS.find(op_name) != STABILITY_CRITICAL_OPS.end();
}

} // namespace amp
} // namespace nn
} // namespace tenzor
