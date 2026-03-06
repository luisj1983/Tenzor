/**
 * @file quantized_layers.cpp
 * @brief Implementation of quantized neural network layers
 */

#include "tenzor/nn/quantization/quantized_layers.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
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
        float input_scale, float weight_scale, int32_t input_zp, int32_t weight_zp,
        int64_t dilation = 1, int64_t groups = 1
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

auto QuantizedLinear::forward_impl(const Variable& input) -> Variable {
    // Quantize input, perform computation, dequantize output
    auto q_input = quantize_per_tensor_symmetric(input.tensor());
    Tensor output = forward_quantized(q_input);
    return Variable(output, input.requires_grad());
}

auto QuantizedLinear::forward_quantized(const QuantizedTensor& input) -> Tensor {
    auto input_shape = input.shape();
    int64_t batch_size = input_shape[0];

    // Remember original device
    auto original_device = input.device();

    // Allocate output on CPU for computation
    Tensor output({batch_size, out_features_}, DType::Float32, Device::cpu());

    // Get quantization parameters - move to CPU for data access
    const auto& input_params = input.params();
    const auto& weight_params = weight_.params();

    Tensor input_scale_cpu = input_params.scale;
    Tensor weight_scale_cpu = weight_params.scale;
    Tensor input_zp_cpu = input_params.zero_point;
    Tensor weight_zp_cpu = weight_params.zero_point;

    if (input_scale_cpu.device() != Device::cpu()) {
        input_scale_cpu = input_scale_cpu.to(Device::cpu());
    }
    if (weight_scale_cpu.device() != Device::cpu()) {
        weight_scale_cpu = weight_scale_cpu.to(Device::cpu());
    }
    if (input_zp_cpu.device() != Device::cpu()) {
        input_zp_cpu = input_zp_cpu.to(Device::cpu());
    }
    if (weight_zp_cpu.device() != Device::cpu()) {
        weight_zp_cpu = weight_zp_cpu.to(Device::cpu());
    }

    float input_scale = input_scale_cpu.data<const float>()[0];
    float weight_scale = weight_scale_cpu.data<const float>()[0];
    int32_t input_zp = input_zp_cpu.data<int32_t>()[0];
    int32_t weight_zp = weight_zp_cpu.data<int32_t>()[0];

    // Move data to CPU for computation
    Tensor input_data_cpu = input.data();
    Tensor weight_data_cpu = weight_.data();
    if (input_data_cpu.device() != Device::cpu()) {
        input_data_cpu = input_data_cpu.to(Device::cpu());
    }
    if (weight_data_cpu.device() != Device::cpu()) {
        weight_data_cpu = weight_data_cpu.to(Device::cpu());
    }

    // Perform quantized matrix multiplication
    const int8_t* input_data = input_data_cpu.data<int8_t>();
    const int8_t* weight_data = weight_data_cpu.data<int8_t>();
    // Convert bias to Float32 and CPU if needed
    std::optional<Tensor> bias_f32;
    const float* bias_data = nullptr;
    if (bias_.has_value()) {
        Tensor bias_cpu = *bias_;
        if (bias_cpu.dtype() != DType::Float32) {
            bias_cpu = bias_cpu.to(DType::Float32);
        }
        if (bias_cpu.device() != Device::cpu()) {
            bias_cpu = bias_cpu.to(Device::cpu());
        }
        bias_f32 = bias_cpu;
        bias_data = bias_f32->data<const float>();
    }
    float* output_data = output.data<float>();

    // Use kernel for computation
    // Use bias_scale_ for output scaling instead of hardcoded 1.0f
    // bias_scale_ controls the dequantization scaling: combined_scale = input_scale * weight_scale / output_scale
    kernels::quantized_linear_kernel(
        input_data, weight_data, bias_data, output_data,
        batch_size, in_features_, out_features_,
        input_scale, weight_scale, bias_scale_,
        input_zp, weight_zp
    );

    // Move output back to original device
    return output.to(original_device);
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

auto QuantizedConv2d::forward_impl(const Variable& input) -> Variable {
    auto q_input = quantize_per_tensor_symmetric(input.tensor());
    Tensor output = forward_quantized(q_input);
    return Variable(output, input.requires_grad());
}

auto QuantizedConv2d::forward_quantized(const QuantizedTensor& input) -> Tensor {
    auto input_shape = input.shape();
    int64_t batch = input_shape[0];
    int64_t h_in = input_shape[2];
    int64_t w_in = input_shape[3];

    // Remember original device
    auto original_device = input.device();

    // Compute output dimensions
    int64_t h_out = (h_in + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;
    int64_t w_out = (w_in + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;

    // Allocate output on CPU
    Tensor output({batch, out_channels_, h_out, w_out}, DType::Float32, Device::cpu());

    // Get quantization parameters - move to CPU for data access
    const auto& input_params = input.params();
    const auto& weight_params = weight_.params();

    Tensor input_scale_cpu = input_params.scale;
    Tensor weight_scale_cpu = weight_params.scale;
    Tensor input_zp_cpu = input_params.zero_point;
    Tensor weight_zp_cpu = weight_params.zero_point;

    if (input_scale_cpu.device() != Device::cpu()) {
        input_scale_cpu = input_scale_cpu.to(Device::cpu());
    }
    if (weight_scale_cpu.device() != Device::cpu()) {
        weight_scale_cpu = weight_scale_cpu.to(Device::cpu());
    }
    if (input_zp_cpu.device() != Device::cpu()) {
        input_zp_cpu = input_zp_cpu.to(Device::cpu());
    }
    if (weight_zp_cpu.device() != Device::cpu()) {
        weight_zp_cpu = weight_zp_cpu.to(Device::cpu());
    }

    float input_scale = input_scale_cpu.data<const float>()[0];
    float weight_scale = weight_scale_cpu.data<const float>()[0];
    int32_t input_zp = input_zp_cpu.data<int32_t>()[0];
    int32_t weight_zp = weight_zp_cpu.data<int32_t>()[0];

    // Move data to CPU for computation
    Tensor input_data_cpu = input.data();
    Tensor weight_data_cpu = weight_.data();
    if (input_data_cpu.device() != Device::cpu()) {
        input_data_cpu = input_data_cpu.to(Device::cpu());
    }
    if (weight_data_cpu.device() != Device::cpu()) {
        weight_data_cpu = weight_data_cpu.to(Device::cpu());
    }

    const int8_t* input_data = input_data_cpu.data<int8_t>();
    const int8_t* weight_data = weight_data_cpu.data<int8_t>();
    // Convert bias to Float32 and CPU if needed
    std::optional<Tensor> bias_f32;
    const float* bias_data = nullptr;
    if (bias_.has_value()) {
        Tensor bias_cpu = *bias_;
        if (bias_cpu.dtype() != DType::Float32) {
            bias_cpu = bias_cpu.to(DType::Float32);
        }
        if (bias_cpu.device() != Device::cpu()) {
            bias_cpu = bias_cpu.to(Device::cpu());
        }
        bias_f32 = bias_cpu;
        bias_data = bias_f32->data<const float>();
    }
    float* output_data = output.data<float>();

    kernels::quantized_conv2d_kernel(
        input_data, weight_data, bias_data, output_data,
        batch, in_channels_, out_channels_,
        h_in, w_in, h_out, w_out,
        kernel_size_, stride_, padding_,
        input_scale, weight_scale, input_zp, weight_zp,
        dilation_, groups_
    );

    // Move output back to original device
    return output.to(original_device);
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
    // Extract weight and bias from float Conv2d
    auto state_dict = fp_conv.state_dict();

    Tensor fp_weight = state_dict.at("weight");
    std::optional<Tensor> fp_bias;
    if (state_dict.find("bias") != state_dict.end()) {
        fp_bias = state_dict.at("bias");
    }

    // Quantize weights using the weight observer from qconfig
    auto weight_observer = qconfig.create_weight_observer();
    weight_observer->observe(fp_weight);
    auto weight_qparams = weight_observer->calculate_qparams(
        qconfig.weight_dtype(),
        qconfig.weight_scheme()
    );

    // Extract Conv2d parameters from weight shape
    auto weight_shape = fp_weight.shape();
    int64_t out_channels = weight_shape[0];
    int64_t in_channels = weight_shape[1];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];
    if (kernel_h != kernel_w) {
        throw std::runtime_error(
            "QuantizedConv2d::from_float: non-square kernels not supported (got " +
            std::to_string(kernel_h) + "x" + std::to_string(kernel_w) + ")");
    }

    // Create quantized Conv2d with actual parameters from source Conv2d
    auto q_conv = std::make_shared<QuantizedConv2d>(
        in_channels,
        out_channels,
        kernel_h,
        fp_conv.stride_h(),
        fp_conv.padding_h(),
        fp_conv.dilation_h(),
        fp_conv.groups(),
        weight_qparams,
        1.0f       // bias_scale (default)
    );

    // Quantize and set the weights
    QuantizedTensor q_weight = quantize_tensor(fp_weight, weight_qparams);
    q_conv->set_weight(q_weight);

    // Set bias if available
    if (fp_bias.has_value()) {
        q_conv->set_bias(fp_bias.value());
    }

    return q_conv;
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

auto QuantizedBatchNorm2d::forward_impl(const Variable& input) -> Variable {
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
    // Extract BatchNorm parameters from state_dict
    auto state_dict = fp_bn.state_dict();

    // BatchNorm parameters: weight (gamma), bias (beta), running_mean, running_var
    Tensor gamma = state_dict.at("weight");
    Tensor beta = state_dict.at("bias");
    Tensor running_mean = state_dict.at("running_mean");
    Tensor running_var = state_dict.at("running_var");

    // Fold BatchNorm parameters into scale and bias
    // Formula:
    //   scale = gamma / sqrt(running_var + eps)
    //   bias = beta - scale * running_mean
    // This allows BN to be applied as: y = scale * x + bias

    // Read epsilon from the source module if possible, otherwise use default
    double eps = 1e-5;
    if (auto* bn2d = dynamic_cast<const BatchNorm2d*>(&fp_bn)) {
        eps = bn2d->eps();
    } else if (auto* bn1d = dynamic_cast<const BatchNorm1d*>(&fp_bn)) {
        eps = bn1d->eps();
    }

    // Compute scale = gamma / sqrt(var + eps)
    Tensor sqrt_var = sqrt(running_var + eps);
    Tensor scale = gamma / sqrt_var;

    // Compute bias = beta - scale * mean
    Tensor bias = beta - scale * running_mean;

    // Get number of features from gamma shape
    int64_t num_features = gamma.shape()[0];

    // Create quantized BatchNorm with folded parameters
    auto q_bn = std::make_shared<QuantizedBatchNorm2d>(
        num_features,
        scale,
        bias
    );

    return q_bn;
}

// ============================================================================
// QuantizedLayerNorm Implementation
// ============================================================================

QuantizedLayerNorm::QuantizedLayerNorm(
    std::vector<int64_t> normalized_shape,
    Tensor weight,
    Tensor bias,
    double eps
) : normalized_shape_(std::move(normalized_shape)),
    weight_(std::move(weight)),
    bias_(std::move(bias)),
    eps_(eps) {}

auto QuantizedLayerNorm::forward_impl(const Variable& input) -> Variable {
    // Layer norm: y = (x - mean) / sqrt(var + eps) * weight + bias
    // For quantized inference, we compute at float precision
    Tensor x = input.tensor();
    auto x_shape = x.shape();
    int64_t norm_size = 1;
    for (auto d : normalized_shape_) norm_size *= d;

    // Compute mean and variance over the last N dimensions
    int64_t outer_size = x.numel() / norm_size;
    Tensor x_flat = x.reshape({outer_size, norm_size});

    Tensor x_mean = ::tenzor::mean(x_flat, 1, true);    // [outer, 1]
    Tensor centered = x_flat - x_mean;
    Tensor var = ::tenzor::mean(centered * centered, 1, true);  // [outer, 1]

    // inv_std = 1 / sqrt(var + eps)
    auto eps_t = ::tenzor::full({outer_size, 1}, static_cast<float>(eps_),
                                 var.dtype(), var.device());
    Tensor std_val = ::tenzor::sqrt(var + eps_t);
    Tensor ones_t = ::tenzor::ones({outer_size, 1}, var.dtype(), var.device());
    Tensor inv_std = ::tenzor::div(ones_t, std_val);
    Tensor normalized = centered * inv_std;

    // Reshape back and apply weight + bias
    normalized = normalized.reshape(std::vector<int64_t>(x_shape.begin(), x_shape.end()));
    Tensor output = normalized * weight_ + bias_;

    return Variable(output, input.requires_grad());
}

auto QuantizedLayerNorm::forward_quantized(const QuantizedTensor& input) -> QuantizedTensor {
    // Dequantize → layer norm → requantize
    Tensor deq = input.dequantize();
    auto result = forward_impl(Variable(deq, false));
    return quantize_per_tensor_symmetric(result.tensor());
}

auto QuantizedLayerNorm::from_float(const Module& fp_ln, const QConfig& /*qconfig*/)
    -> std::shared_ptr<QuantizedLayerNorm> {
    auto state_dict = fp_ln.state_dict();

    Tensor weight = state_dict.at("weight");
    Tensor bias = state_dict.at("bias");

    // Infer normalized_shape from weight shape
    auto wshape = weight.shape();
    std::vector<int64_t> normalized_shape(wshape.begin(), wshape.end());

    // Read epsilon from the source module if possible, otherwise use default
    double eps = 1e-5;
    if (auto* ln = dynamic_cast<const LayerNorm*>(&fp_ln)) {
        eps = ln->eps();
    }

    return std::make_shared<QuantizedLayerNorm>(
        normalized_shape, weight, bias, eps);
}

// ============================================================================
// QuantStub / DeQuantStub - Full Quantization/Dequantization Implementation
// ============================================================================

QuantStub::QuantStub(QuantizationParams qparams)
    : qparams_(std::move(qparams)) {}

auto QuantStub::forward_impl(const Variable& input) -> Variable {
    // Quantize input tensor and immediately dequantize for Variable compatibility
    // This maintains the computational graph while simulating quantization
    auto q_tensor = forward_to_quantized(input.tensor());
    Tensor dequantized = q_tensor.dequantize();
    return Variable(dequantized, input.requires_grad());
}

auto QuantStub::forward_to_quantized(const Tensor& input) -> QuantizedTensor {
    // Perform full quantization:
    // 1. Validate input is floating-point
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) {
        throw std::runtime_error("QuantStub: Input must be floating-point type");
    }

    // 2. Convert to Float32 if needed
    Tensor fp_input = (input.dtype() == DType::Float32) ? input : input.to(DType::Float32);

    // 3. Apply quantization using stored parameters
    // This handles both per-tensor and per-channel quantization
    // Formula: q = clamp(round(x / scale) + zero_point, qmin, qmax)
    return quantize_tensor(fp_input, qparams_);
}

auto DeQuantStub::forward_impl(const Variable& input) -> Variable {
    // For Variable input, we assume it contains quantized data that needs dequantization
    // In a real implementation, the Variable would have metadata indicating it's quantized
    // For now, we pass through assuming the tensor is already in FP32 format
    // This is a limitation of the Variable wrapper not having quantization metadata

    // Direct passthrough since Variable doesn't have QuantizedTensor metadata
    // In production, this would extract quantization params from Variable metadata
    return Variable(input.tensor(), input.requires_grad());
}

auto DeQuantStub::forward_from_quantized(const QuantizedTensor& input) -> Tensor {
    // Perform full dequantization:
    // 1. Extract quantization parameters
    const auto& params = input.params();

    // 2. Track if this was per-channel for debugging
    last_per_channel_ = (params.axis != -1);

    // 3. Apply dequantization based on scheme
    // Formula: x = (q - zero_point) * scale
    // This handles both:
    //   - Per-tensor: Single scale and zero_point for entire tensor
    //   - Per-channel: Different scale/zero_point per channel along specified axis

    return dequantize_tensor(input);
}

// ============================================================================
// Fused Layers
// ============================================================================

auto QuantizedConv2dReLU::forward_quantized(const QuantizedTensor& input) -> Tensor {
    Tensor output = QuantizedConv2d::forward_quantized(input);

    // Remember original device
    auto original_device = output.device();

    // Move to CPU for data access
    Tensor output_cpu = output;
    if (output_cpu.device() != Device::cpu()) {
        output_cpu = output_cpu.to(Device::cpu());
    }

    // Apply ReLU
    float* data = output_cpu.data<float>();
    int64_t n = output_cpu.numel();
    for (int64_t i = 0; i < n; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }

    // Move back to original device
    return output_cpu.to(original_device);
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
    // Extract weight and bias from float Conv2d
    auto state_dict = fp_conv.state_dict();

    Tensor fp_weight = state_dict.at("weight");
    std::optional<Tensor> fp_bias;
    if (state_dict.find("bias") != state_dict.end()) {
        fp_bias = state_dict.at("bias");
    }

    // Quantize weights using the weight observer from qconfig
    auto weight_observer = qconfig.create_weight_observer();
    weight_observer->observe(fp_weight);
    auto weight_qparams = weight_observer->calculate_qparams(
        qconfig.weight_dtype(),
        qconfig.weight_scheme()
    );

    // Extract Conv2d parameters from weight shape
    auto weight_shape = fp_weight.shape();
    int64_t out_channels = weight_shape[0];
    int64_t in_channels = weight_shape[1];
    int64_t kernel_h = weight_shape[2];

    // Create QuantizedConv2dReLU with actual parameters from source Conv2d
    auto q_conv_relu = std::shared_ptr<QuantizedConv2dReLU>(
        new QuantizedConv2dReLU(
            in_channels,
            out_channels,
            kernel_h,
            fp_conv.stride_h(),
            fp_conv.padding_h(),
            fp_conv.dilation_h(),
            fp_conv.groups(),
            weight_qparams,
            1.0f       // bias_scale
        )
    );

    // Quantize and set the weights
    QuantizedTensor q_weight = quantize_tensor(fp_weight, weight_qparams);
    q_conv_relu->set_weight(q_weight);

    // Set bias if available
    if (fp_bias.has_value()) {
        q_conv_relu->set_bias(fp_bias.value());
    }

    return q_conv_relu;
}

// ============================================================================
// QuantizedConv2dBnReLU
// ============================================================================

QuantizedConv2dBnReLU::QuantizedConv2dBnReLU(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    QuantizationParams weight_qparams,
    Tensor bn_scale,
    Tensor bn_bias
) : bn_scale_(std::move(bn_scale)),
    bn_bias_(std::move(bn_bias)) {
    // Create internal quantized conv with the folded weights
    conv_ = std::make_shared<QuantizedConv2d>(
        in_channels, out_channels, kernel_size,
        stride, padding, dilation, groups,
        weight_qparams
    );
}

auto QuantizedConv2dBnReLU::forward_impl(const Variable& input) -> Variable {
    // Quantize input, run fused conv+BN+ReLU, return float output
    auto q_input = quantize_per_tensor_symmetric(input.tensor());
    Tensor output = forward_quantized(q_input);
    return Variable(output, input.requires_grad());
}

auto QuantizedConv2dBnReLU::forward_quantized(const QuantizedTensor& input) -> Tensor {
    // Run quantized convolution (with BN folded into weights)
    Tensor output = conv_->forward_quantized(input);

    auto original_device = output.device();

    // Move to CPU for element-wise operations
    Tensor output_cpu = output;
    if (output_cpu.device() != Device::cpu()) {
        output_cpu = output_cpu.to(Device::cpu());
    }

    // Apply BN scale and bias (already folded from running stats)
    // output shape: [N, C, H, W]
    // bn_scale_ and bn_bias_ shape: [C]
    auto output_shape = output_cpu.shape();
    int64_t batch = output_shape[0];
    int64_t channels = output_shape[1];
    int64_t h_out = output_shape[2];
    int64_t w_out = output_shape[3];

    Tensor bn_scale_cpu = bn_scale_;
    Tensor bn_bias_cpu = bn_bias_;
    if (bn_scale_cpu.device() != Device::cpu()) {
        bn_scale_cpu = bn_scale_cpu.to(Device::cpu());
    }
    if (bn_bias_cpu.device() != Device::cpu()) {
        bn_bias_cpu = bn_bias_cpu.to(Device::cpu());
    }

    float* out_data = output_cpu.data<float>();
    const float* scale_data = bn_scale_cpu.data<const float>();
    const float* bias_data = bn_bias_cpu.data<const float>();

    // Apply BN: y = scale * x + bias, then ReLU: y = max(0, y)
    int64_t spatial = h_out * w_out;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            float s = scale_data[c];
            float bi = bias_data[c];
            float* channel_data = out_data + (b * channels + c) * spatial;
            for (int64_t i = 0; i < spatial; ++i) {
                float val = channel_data[i] * s + bi;
                channel_data[i] = std::max(0.0f, val);  // Fused ReLU
            }
        }
    }

    return output_cpu.to(original_device);
}

auto QuantizedConv2dBnReLU::from_float(
    const Conv2d& fp_conv,
    const Module& fp_bn,
    const QConfig& qconfig
) -> std::shared_ptr<QuantizedConv2dBnReLU> {
    // Extract BatchNorm parameters from state_dict
    auto bn_state = fp_bn.state_dict();

    Tensor gamma = bn_state.at("weight");
    Tensor beta = bn_state.at("bias");
    Tensor running_mean = bn_state.at("running_mean");
    Tensor running_var = bn_state.at("running_var");

    // Read epsilon from the source module
    double eps = 1e-5;
    if (auto* bn2d = dynamic_cast<const BatchNorm2d*>(&fp_bn)) {
        eps = bn2d->eps();
    }

    // Fold BN into conv weights:
    // w_folded = gamma / sqrt(var + eps) * w_conv
    // b_folded = gamma / sqrt(var + eps) * (b_conv - mean) + beta
    Tensor sqrt_var = sqrt(running_var + eps);
    Tensor bn_scale = gamma / sqrt_var;  // [C]

    // Get conv weights and bias
    auto conv_state = fp_conv.state_dict();
    Tensor fp_weight = conv_state.at("weight");  // [out_channels, in_channels/groups, kH, kW]
    std::optional<Tensor> fp_bias;
    if (conv_state.find("bias") != conv_state.end()) {
        fp_bias = conv_state.at("bias");
    }

    // Fold BN scale into conv weights
    // weight shape: [out_channels, in_channels/groups, kH, kW]
    auto weight_shape = fp_weight.shape();
    int64_t out_channels = weight_shape[0];

    // Scale each output channel's weights by bn_scale[c]
    Tensor folded_weight = fp_weight.clone();
    Tensor folded_weight_cpu = folded_weight;
    if (folded_weight_cpu.device() != Device::cpu()) {
        folded_weight_cpu = folded_weight_cpu.to(Device::cpu());
    }
    Tensor bn_scale_cpu = bn_scale;
    if (bn_scale_cpu.device() != Device::cpu()) {
        bn_scale_cpu = bn_scale_cpu.to(Device::cpu());
    }

    float* w_data = folded_weight_cpu.data<float>();
    const float* s_data = bn_scale_cpu.data<const float>();
    int64_t channel_size = folded_weight_cpu.numel() / out_channels;
    for (int64_t c = 0; c < out_channels; ++c) {
        float s = s_data[c];
        for (int64_t i = 0; i < channel_size; ++i) {
            w_data[c * channel_size + i] *= s;
        }
    }

    // Fold BN into bias
    // b_folded = bn_scale * (b_conv - running_mean) + beta
    Tensor folded_bias({out_channels}, DType::Float32, Device::cpu());
    float* fb_data = folded_bias.data<float>();
    Tensor running_mean_cpu = running_mean;
    Tensor beta_cpu = beta;
    if (running_mean_cpu.device() != Device::cpu()) {
        running_mean_cpu = running_mean_cpu.to(Device::cpu());
    }
    if (beta_cpu.device() != Device::cpu()) {
        beta_cpu = beta_cpu.to(Device::cpu());
    }
    const float* mean_data = running_mean_cpu.data<const float>();
    const float* beta_data = beta_cpu.data<const float>();

    for (int64_t c = 0; c < out_channels; ++c) {
        float conv_bias = 0.0f;
        if (fp_bias.has_value()) {
            Tensor bias_cpu = *fp_bias;
            if (bias_cpu.device() != Device::cpu()) {
                bias_cpu = bias_cpu.to(Device::cpu());
            }
            conv_bias = bias_cpu.data<const float>()[c];
        }
        fb_data[c] = s_data[c] * (conv_bias - mean_data[c]) + beta_data[c];
    }

    // Quantize folded weights
    auto weight_observer = qconfig.create_weight_observer();
    weight_observer->observe(folded_weight_cpu);
    auto weight_qparams = weight_observer->calculate_qparams(
        qconfig.weight_dtype(),
        qconfig.weight_scheme()
    );

    // Extract Conv2d parameters
    int64_t in_channels = weight_shape[1] * fp_conv.groups();
    int64_t kernel_h = weight_shape[2];

    // Create the fused layer - BN scale/bias are identity since folded into weights
    Tensor ones_scale({out_channels}, DType::Float32, Device::cpu());
    Tensor zeros_bias({out_channels}, DType::Float32, Device::cpu());
    ones_scale.fill_(1.0f);
    zeros_bias.fill_(0.0f);

    auto fused = std::make_shared<QuantizedConv2dBnReLU>(
        in_channels, out_channels, kernel_h,
        fp_conv.stride_h(), fp_conv.padding_h(),
        fp_conv.dilation_h(), fp_conv.groups(),
        weight_qparams,
        ones_scale,   // BN already folded into weights, so scale = 1
        zeros_bias    // BN already folded into bias, so bias = 0
    );

    // Set quantized weights with folded BN
    QuantizedTensor q_weight = quantize_tensor(folded_weight_cpu, weight_qparams);
    fused->conv_->set_weight(q_weight);
    fused->conv_->set_bias(folded_bias);

    return fused;
}

} // namespace quantization
} // namespace nn
} // namespace tenzor
