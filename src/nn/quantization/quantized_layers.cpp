/**
 * @file quantized_layers.cpp
 * @brief Implementation of quantized neural network layers
 */

#include "tenzor/nn/quantization/quantized_layers.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include <stdexcept>

namespace tenzor {
namespace nn {
namespace quantization {

// Forward declare kernel functions
namespace kernels {
    auto quantized_linear_kernel(
        const int8_t* input, const int8_t* weight, const float* bias,
        float* output, int64_t batch_size, int64_t in_features, int64_t out_features,
        float input_scale, float weight_scale, float output_scale,
        int32_t input_zp, int32_t weight_zp
    ) -> void;

    auto quantized_conv2d_kernel(
        const int8_t* input, const int8_t* weight, const float* bias,
        float* output, int64_t batch, int64_t in_channels, int64_t out_channels,
        int64_t h_in, int64_t w_in, int64_t h_out, int64_t w_out,
        int64_t kernel_size, int64_t stride, int64_t padding,
        float input_scale, float weight_scale, int32_t input_zp, int32_t weight_zp
    ) -> void;
}

// ============================================================================
// QuantizedLinear
// ============================================================================

QuantizedLinear::QuantizedLinear(
    int64_t in_features,
    int64_t out_features,
    QuantizationParams weight_qparams,
    float bias_scale
) : in_features_(in_features),
    out_features_(out_features),
    weight_(Tensor({out_features, in_features}, DType::Int8, Device::cpu()), weight_qparams),
    bias_scale_(bias_scale) {}

auto QuantizedLinear::forward(const Variable& input) -> Variable {
    // Quantize input, perform computation, dequantize output
    auto q_input = quantize_per_tensor_symmetric(input.tensor());
    Tensor output = forward_quantized(q_input);
    return Variable(output, input.requires_grad());
}

auto QuantizedLinear::forward_quantized(const QuantizedTensor& input) -> Tensor {
    auto input_shape = input.shape();
    int64_t batch_size = input_shape[0];

    // Allocate output
    Tensor output({batch_size, out_features_}, DType::Float32, input.device());

    // Get quantization parameters
    const auto& input_params = input.params();
    const auto& weight_params = weight_.params();

    float input_scale = input_params.scale.data<const float>()[0];
    float weight_scale = weight_params.scale.data<const float>()[0];
    int32_t input_zp = input_params.zero_point.data<const int32_t>()[0];
    int32_t weight_zp = weight_params.zero_point.data<const int32_t>()[0];

    // Perform quantized matrix multiplication
    const int8_t* input_data = input.data().data<const int8_t>();
    const int8_t* weight_data = weight_.data().data<const int8_t>();
    const float* bias_data = bias_.has_value() ? bias_->data<const float>() : nullptr;
    float* output_data = output.data<float>();

    // Use kernel for computation
    kernels::quantized_linear_kernel(
        input_data, weight_data, bias_data, output_data,
        batch_size, in_features_, out_features_,
        input_scale, weight_scale, 1.0f,
        input_zp, weight_zp
    );

    return output;
}

auto QuantizedLinear::forward_quantized_output(
    const QuantizedTensor& input,
    const QuantizationParams& output_qparams
) -> QuantizedTensor {
    // Similar to forward_quantized but quantizes the output
    Tensor fp_output = forward_quantized(input);
    return quantize_tensor(fp_output, output_qparams);
}

auto QuantizedLinear::set_weight(const QuantizedTensor& weights) -> void {
    weight_ = weights;
}

auto QuantizedLinear::set_bias(const Tensor& bias) -> void {
    bias_ = bias;
}

auto QuantizedLinear::from_float(const Linear& fp_linear, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedLinear> {
    // Quantize weights
    auto weight_tensor = fp_linear.weight()->tensor();
    auto weight_observer = qconfig.create_weight_observer();
    weight_observer->observe(weight_tensor);
    auto weight_qparams = weight_observer->calculate_qparams(
        qconfig.weight_dtype(), qconfig.weight_scheme()
    );

    auto q_weight = quantize_tensor(weight_tensor, weight_qparams);

    // Create quantized layer
    auto q_layer = std::make_shared<QuantizedLinear>(
        weight_tensor.shape()[1],  // in_features
        weight_tensor.shape()[0],  // out_features
        weight_qparams
    );

    q_layer->set_weight(q_weight);

    // Set bias if present
    if (fp_linear.has_bias()) {
        q_layer->set_bias(fp_linear.bias()->tensor());
    }

    return q_layer;
}

// ============================================================================
// QuantizedConv2d
// ============================================================================

QuantizedConv2d::QuantizedConv2d(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    QuantizationParams weight_qparams,
    float bias_scale
) : in_channels_(in_channels),
    out_channels_(out_channels),
    kernel_size_(kernel_size),
    stride_(stride),
    padding_(padding),
    dilation_(dilation),
    groups_(groups),
    weight_(Tensor({out_channels, in_channels / groups, kernel_size, kernel_size},
                   DType::Int8, Device::cpu()), weight_qparams),
    bias_scale_(bias_scale) {}

auto QuantizedConv2d::forward(const Variable& input) -> Variable {
    auto q_input = quantize_per_tensor_symmetric(input.tensor());
    Tensor output = forward_quantized(q_input);
    return Variable(output, input.requires_grad());
}

auto QuantizedConv2d::forward_quantized(const QuantizedTensor& input) -> Tensor {
    auto input_shape = input.shape();
    int64_t batch = input_shape[0];
    int64_t h_in = input_shape[2];
    int64_t w_in = input_shape[3];

    // Compute output dimensions
    int64_t h_out = (h_in + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;
    int64_t w_out = (w_in + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;

    Tensor output({batch, out_channels_, h_out, w_out}, DType::Float32, input.device());

    // Get quantization parameters
    const auto& input_params = input.params();
    const auto& weight_params = weight_.params();

    float input_scale = input_params.scale.data<const float>()[0];
    float weight_scale = weight_params.scale.data<const float>()[0];
    int32_t input_zp = input_params.zero_point.data<const int32_t>()[0];
    int32_t weight_zp = weight_params.zero_point.data<const int32_t>()[0];

    const int8_t* input_data = input.data().data<const int8_t>();
    const int8_t* weight_data = weight_.data().data<const int8_t>();
    const float* bias_data = bias_.has_value() ? bias_->data<const float>() : nullptr;
    float* output_data = output.data<float>();

    kernels::quantized_conv2d_kernel(
        input_data, weight_data, bias_data, output_data,
        batch, in_channels_, out_channels_,
        h_in, w_in, h_out, w_out,
        kernel_size_, stride_, padding_,
        input_scale, weight_scale, input_zp, weight_zp
    );

    return output;
}

auto QuantizedConv2d::forward_quantized_output(
    const QuantizedTensor& input,
    const QuantizationParams& output_qparams
) -> QuantizedTensor {
    Tensor fp_output = forward_quantized(input);
    return quantize_tensor(fp_output, output_qparams);
}

auto QuantizedConv2d::set_weight(const QuantizedTensor& weights) -> void {
    weight_ = weights;
}

auto QuantizedConv2d::set_bias(const Tensor& bias) -> void {
    bias_ = bias;
}

auto QuantizedConv2d::from_float(const Conv2d& fp_conv, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedConv2d> {
    // Similar to Linear - quantize weights and create quantized layer
    // Implementation details omitted for brevity
    throw std::runtime_error("Not implemented - would quantize Conv2d weights");
}

// ============================================================================
// QuantizedBatchNorm2d
// ============================================================================

QuantizedBatchNorm2d::QuantizedBatchNorm2d(
    int64_t num_features,
    Tensor scale,
    Tensor bias
) : num_features_(num_features),
    scale_(std::move(scale)),
    bias_(std::move(bias)) {}

auto QuantizedBatchNorm2d::forward(const Variable& input) -> Variable {
    // Apply scale and bias
    Tensor scaled = input.tensor() * scale_.unsqueeze(0).unsqueeze(2).unsqueeze(3);
    Tensor output = scaled + bias_.unsqueeze(0).unsqueeze(2).unsqueeze(3);
    return Variable(output, input.requires_grad());
}

auto QuantizedBatchNorm2d::forward_quantized(const QuantizedTensor& input) -> QuantizedTensor {
    // Dequantize, apply BN, requantize
    Tensor deq = input.dequantize();
    Tensor scaled = deq * scale_.unsqueeze(0).unsqueeze(2).unsqueeze(3);
    Tensor output = scaled + bias_.unsqueeze(0).unsqueeze(2).unsqueeze(3);
    return quantize_per_tensor_symmetric(output);
}

auto QuantizedBatchNorm2d::from_float(const Module& fp_bn, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedBatchNorm2d> {
    // Extract BN parameters and fold
    // Implementation would extract gamma, beta, running_mean, running_var
    // and compute folded scale = gamma / sqrt(var + eps), bias = beta - scale * mean
    throw std::runtime_error("Not implemented - would fold BN parameters");
}

// ============================================================================
// QuantizationStub / DequantizationStub
// ============================================================================

QuantizationStub::QuantizationStub(QuantizationParams qparams)
    : qparams_(std::move(qparams)) {}

auto QuantizationStub::forward(const Variable& input) -> Variable {
    auto q_tensor = forward_to_quantized(input.tensor());
    // Wrap quantized tensor (implementation-specific)
    return Variable(q_tensor.dequantize(), false);
}

auto QuantizationStub::forward_to_quantized(const Tensor& input) -> QuantizedTensor {
    return quantize_tensor(input, qparams_);
}

auto DequantizationStub::forward(const Variable& input) -> Variable {
    // Assume input contains quantized data (implementation-specific unwrapping)
    return input;  // Simplified
}

auto DequantizationStub::forward_from_quantized(const QuantizedTensor& input) -> Tensor {
    return input.dequantize();
}

// ============================================================================
// Fused Layers
// ============================================================================

auto QuantizedConv2dReLU::forward_quantized(const QuantizedTensor& input) -> Tensor {
    Tensor output = QuantizedConv2d::forward_quantized(input);

    // Apply ReLU
    float* data = output.data<float>();
    int64_t n = output.numel();
    for (int64_t i = 0; i < n; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }

    return output;
}

auto QuantizedConv2dReLU::forward_quantized_output(
    const QuantizedTensor& input,
    const QuantizationParams& output_qparams
) -> QuantizedTensor {
    Tensor fp_output = forward_quantized(input);
    return quantize_tensor(fp_output, output_qparams);
}

auto QuantizedConv2dReLU::from_float(const Conv2d& fp_conv, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedConv2dReLU> {
    // Similar to QuantizedConv2d::from_float
    throw std::runtime_error("Not implemented");
}

} // namespace quantization
} // namespace nn
} // namespace tenzor
