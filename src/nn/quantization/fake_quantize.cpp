/**
 * @file fake_quantize.cpp
 * @brief Implementation of fake quantization for QAT
 */

#include "tenzor/nn/quantization/fake_quantize.hpp"
#include <stdexcept>
#include <cmath>

namespace tenzor {
namespace nn {
namespace quantization {

// ============================================================================
// FakeQuantize
// ============================================================================

FakeQuantize::FakeQuantize(
    QuantDType dtype,
    QuantizationScheme scheme,
    bool learnable,
    bool observer_enabled,
    int64_t axis
) : dtype_(dtype),
    scheme_(scheme),
    learnable_(learnable),
    observer_enabled_(observer_enabled),
    axis_(axis) {

    // Create observer
    bool use_histogram = false;  // Can be made configurable
    observer_ = make_observer(scheme, use_histogram, axis);

    // Initialize quantization params with dummy values
    Tensor scale({1}, DType::Float32, Device::cpu());
    Tensor zero_point({1}, DType::Int32, Device::cpu());
    scale.fill_(1.0f);
    zero_point.fill_(0);

    qparams_ = std::make_unique<QuantizationParams>(
        scale, zero_point, dtype, scheme, axis
    );

    if (learnable_) {
        // Register scale and zero_point as parameters
        Variable scale_var(qparams_->scale, true);
        Variable zp_var(qparams_->zero_point, false);  // Zero point typically not learnable
        register_parameter("scale", scale_var);
        register_parameter("zero_point", zp_var);
    }
}

auto FakeQuantize::forward_impl(const Variable& input) -> Variable {
    if (!fake_quant_enabled_) {
        return input;  // Pass through
    }

    // Update observer if enabled (even in eval mode for calibration)
    if (observer_enabled_) {
        observer_->observe(input.tensor());

        // Update qparams if observer has data and in training mode
        if (observer_->has_data() && !learnable_ && training_) {
            calculate_qparams();
        }
    }

    // Apply fake quantization
    Tensor quantized = apply_fake_quantization(input.tensor());

    // Create output variable maintaining gradient tracking
    return Variable(quantized, input.requires_grad());
}

auto FakeQuantize::apply_fake_quantization(const Tensor& input) const -> Tensor {
    // Quantize then immediately dequantize
    QuantizedTensor q_tensor = quantize_tensor(input, *qparams_);
    Tensor output = dequantize_tensor(q_tensor);
    // Preserve original dtype
    if (output.dtype() != input.dtype()) {
        output = output.to(input.dtype());
    }
    return output;
}

auto FakeQuantize::set_qparams(const QuantizationParams& params) -> void {
    qparams_ = std::make_unique<QuantizationParams>(params);

    if (learnable_) {
        // Update registered parameters
        Variable scale_var(qparams_->scale, true);
        register_parameter("scale", scale_var);
    }
}

auto FakeQuantize::calculate_qparams() -> void {
    if (!observer_->has_data()) {
        throw std::runtime_error("Observer has no data to calculate qparams");
    }

    *qparams_ = observer_->calculate_qparams(dtype_, scheme_);

    if (learnable_) {
        // Update registered parameters
        Variable scale_var(qparams_->scale, true);
        register_parameter("scale", scale_var);
    }
}

auto FakeQuantize::reset_observer() -> void {
    observer_->reset();
}

// ============================================================================
// LearnableFakeQuantize
// ============================================================================

LearnableFakeQuantize::LearnableFakeQuantize(
    QuantDType dtype,
    QuantizationScheme scheme,
    int64_t axis
) : FakeQuantize(dtype, scheme, true, true, axis) {}

auto LearnableFakeQuantize::init_from_observer() -> void {
    if (!observer()->has_data()) {
        throw std::runtime_error("Observer has no data for initialization");
    }

    calculate_qparams();
}

// ============================================================================
// Functional Interface
// ============================================================================

auto fake_quantize_activation(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    QuantDType dtype
) -> Tensor {
    QuantizationParams params(
        scale, zero_point, dtype,
        QuantizationScheme::PerTensorSymmetric, -1
    );

    QuantizedTensor q_tensor = quantize_tensor(input, params);
    Tensor output = dequantize_tensor(q_tensor);
    // Preserve original dtype
    if (output.dtype() != input.dtype()) {
        output = output.to(input.dtype());
    }
    return output;
}

auto fake_quantize_weight(
    const Tensor& weight,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    QuantDType dtype
) -> Tensor {
    QuantizationParams params(
        scale, zero_point, dtype,
        QuantizationScheme::PerChannelSymmetric, axis
    );

    QuantizedTensor q_tensor = quantize_tensor(weight, params);
    Tensor output = dequantize_tensor(q_tensor);
    // Preserve original dtype
    if (output.dtype() != weight.dtype()) {
        output = output.to(weight.dtype());
    }
    return output;
}

// ============================================================================
// QATHelper
// ============================================================================

auto QATHelper::prepare_qat(
    Module& model,
    QuantDType dtype,
    QuantizationScheme scheme,
    bool learnable
) -> void {
    // This would traverse the model and insert fake quantization modules
    // For now, this is a simplified implementation
    // In production, would need model introspection and transformation

    // Store reference to fake quant modules for later management
    // Implementation would depend on model structure traversal
}

auto QATHelper::enable_observer() -> void {
    for (auto& fq : fake_quant_modules_) {
        fq->enable_observer(true);
    }
}

auto QATHelper::disable_observer() -> void {
    for (auto& fq : fake_quant_modules_) {
        fq->disable_observer();
    }
}

auto QATHelper::freeze_bn_stats() -> void {
    // Freeze batch norm statistics and disable observers
    disable_observer();

    for (auto& fq : fake_quant_modules_) {
        fq->calculate_qparams();
    }
}

auto QATHelper::convert_to_quantized(Module& model) -> std::shared_ptr<Module> {
    // Convert fake quantization to actual quantized operations
    // This would replace FakeQuantize modules with actual quantized layers
    // and convert floating-point layers to quantized variants

    // Note: Full QAT to quantized conversion requires traversing model architecture
    return std::make_shared<Sequential>();
}

// ============================================================================
// Straight-Through Estimator
// ============================================================================

auto StraightThroughEstimator::forward(const Tensor& input, float quant_min, float quant_max)
    -> Tensor {
    // Clamp values to quantization range
    // Remember original device
    auto original_device = input.device();

    // Work in Float32 on CPU for processing
    Tensor input_f32 = input;
    if (input.dtype() != DType::Float32) {
        input_f32 = input.to(DType::Float32);
    }
    if (input_f32.device() != Device::cpu()) {
        input_f32 = input_f32.to(Device::cpu());
    }
    Tensor output = input_f32.clone();
    float* data = output.data<float>();
    int64_t n = output.numel();

    for (int64_t i = 0; i < n; ++i) {
        data[i] = std::clamp(data[i], quant_min, quant_max);
    }

    // Move back to original device
    output = output.to(original_device);

    // Convert back to original dtype if needed
    if (input.dtype() != DType::Float32) {
        output = output.to(input.dtype());
    }

    return output;
}

auto StraightThroughEstimator::backward(
    const Tensor& grad_output,
    const Tensor& input,
    float quant_min,
    float quant_max
) -> Tensor {
    // Pass gradients through for values within range, zero otherwise
    // Remember original device
    auto original_device = grad_output.device();

    // Work in Float32 on CPU for processing
    Tensor input_f32 = input;
    if (input.dtype() != DType::Float32) {
        input_f32 = input.to(DType::Float32);
    }
    if (input_f32.device() != Device::cpu()) {
        input_f32 = input_f32.to(Device::cpu());
    }
    Tensor grad_f32 = grad_output;
    if (grad_output.dtype() != DType::Float32) {
        grad_f32 = grad_output.to(DType::Float32);
    }
    if (grad_f32.device() != Device::cpu()) {
        grad_f32 = grad_f32.to(Device::cpu());
    }
    Tensor grad_input = grad_f32.clone();
    const float* input_data = input_f32.data<const float>();
    float* grad_data = grad_input.data<float>();
    int64_t n = grad_input.numel();

    for (int64_t i = 0; i < n; ++i) {
        if (input_data[i] < quant_min || input_data[i] > quant_max) {
            grad_data[i] = 0.0f;  // Zero gradient for out-of-range values
        }
    }

    // Move back to original device
    grad_input = grad_input.to(original_device);

    // Convert back to original dtype if needed
    if (grad_output.dtype() != DType::Float32) {
        grad_input = grad_input.to(grad_output.dtype());
    }

    return grad_input;
}

} // namespace quantization
} // namespace nn
} // namespace tenzor
