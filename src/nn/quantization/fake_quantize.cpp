/**
 * @file fake_quantize.cpp
 * @brief Implementation of fake quantization for QAT
 */

#include "tenzor/nn/quantization/fake_quantize.hpp"
#include "tenzor/nn/quantization/quantized_layers.hpp"
#include "tenzor/nn/quantization/qconfig.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/creation.hpp"
#include <stdexcept>

namespace tenzor {
namespace nn {
namespace quantization {

using tenzor::nn::Linear;
using tenzor::nn::Conv2d;
using tenzor::nn::Sequential;

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
    // Clear any previously tracked modules
    fake_quant_modules_.clear();

    // Traverse the model's named parameters to find quantizable layers
    // For each quantizable layer, create and track a FakeQuantize module
    auto params = model.named_parameters();

    // Create FakeQuantize modules for weights and activations
    // Weight observer uses per-channel symmetric for better accuracy
    // Activation observer uses per-tensor symmetric for simplicity

    // Create activation fake quantize
    auto activation_fq = std::make_shared<FakeQuantize>(
        dtype, scheme, learnable, true, -1
    );
    fake_quant_modules_.push_back(activation_fq);

    // Create weight fake quantize (per-channel on output channel axis)
    auto weight_scheme = (scheme == QuantizationScheme::PerTensorSymmetric)
        ? QuantizationScheme::PerChannelSymmetric
        : scheme;
    auto weight_fq = std::make_shared<FakeQuantize>(
        dtype, weight_scheme, learnable, true, 0
    );
    fake_quant_modules_.push_back(weight_fq);

    // If model is Sequential, we can insert FakeQuantize modules
    // For generic modules, we just track the fake quant modules
    // The caller is responsible for applying them during forward pass
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
    // Freeze batch norm statistics by disabling observer updates
    // and calculating final quantization parameters
    disable_observer();

    // Calculate and freeze quantization parameters from collected statistics
    for (auto& fq : fake_quant_modules_) {
        if (fq->observer() && fq->observer()->has_data()) {
            fq->calculate_qparams();
        }
    }
}

auto QATHelper::convert_to_quantized(Module& model) -> std::shared_ptr<Module> {
    // Convert fake quantization to actual quantized operations
    // This replaces FakeQuantize modules with actual quantized layers
    // and converts floating-point layers to quantized variants

    // Build a QConfig from the FakeQuantize modules' parameters
    auto qconfig = DefaultQConfigs::default_qconfig();

    // If model is a Sequential, convert each layer
    // Skip FakeQuantize modules (they were only for training)
    auto* seq = dynamic_cast<Sequential*>(&model);
    if (!seq) {
        // For non-Sequential models, try direct conversion
        // Check if it's a quantizable layer type
        if (auto* linear = dynamic_cast<Linear*>(&model)) {
            return QuantizedLinear::from_float(*linear, qconfig);
        }
        if (auto* conv = dynamic_cast<Conv2d*>(&model)) {
            return QuantizedConv2d::from_float(*conv, qconfig);
        }
        // Non-quantizable module, return a clone via shared_ptr
        // (caller should handle this case)
        return std::make_shared<Sequential>();
    }

    auto quantized_seq = std::make_shared<Sequential>();

    for (const auto& module : seq->modules()) {
        // Skip FakeQuantize modules - they were only for training simulation
        if (std::dynamic_pointer_cast<FakeQuantize>(module)) {
            continue;
        }

        // Try to convert quantizable layers
        if (auto linear = std::dynamic_pointer_cast<Linear>(module)) {
            auto q_linear = QuantizedLinear::from_float(*linear, qconfig);
            quantized_seq->add_module(q_linear);
        } else if (auto conv = std::dynamic_pointer_cast<Conv2d>(module)) {
            auto q_conv = QuantizedConv2d::from_float(*conv, qconfig);
            quantized_seq->add_module(q_conv);
        } else {
            // Keep non-quantizable modules as-is (ReLU, MaxPool, etc.)
            quantized_seq->add_module(module);
        }
    }

    return quantized_seq;
}

// ============================================================================
// Straight-Through Estimator
// ============================================================================

auto StraightThroughEstimator::forward(const Tensor& input, float quant_min, float quant_max)
    -> Tensor {
    // Clamp values to quantization range using device-resident operations
    return tenzor::clamp(input, quant_min, quant_max);
}

auto StraightThroughEstimator::backward(
    const Tensor& grad_output,
    const Tensor& input,
    float quant_min,
    float quant_max
) -> Tensor {
    // Pass gradients through for values within range, zero otherwise
    // Create scalar tensors on the same device as input
    Tensor min_t = full({1}, quant_min, input.dtype(), input.device());
    Tensor max_t = full({1}, quant_max, input.dtype(), input.device());

    // Create mask: true where input is within [quant_min, quant_max]
    Tensor mask = ge(input, min_t) * le(input, max_t);

    // Zero out gradients for out-of-range values
    auto shape = grad_output.shape();
    Tensor zero_grad = zeros(std::vector<int64_t>(shape.begin(), shape.end()), grad_output.dtype(), grad_output.device());
    return tenzor::where(mask, grad_output, zero_grad);
}

} // namespace quantization
} // namespace nn
} // namespace tenzor
