/**
 * @file quantized_layers.cpp
 * @brief Implementation of quantized neural network layers
 */

#include "tenzor/nn/quantization/quantized_layers.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/nn/layers/embedding.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/ops.hpp"
#include "../../backends/cpu/kernels/fused_quantized_ops.hpp"
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

    // Per-channel quantized variants
    auto quantized_linear_per_channel_kernel(
        const int8_t* input, const int8_t* weight, const float* bias,
        float* output, int64_t batch_size, int64_t in_features, int64_t out_features,
        float input_scale, const float* weight_scales, float output_scale,
        int32_t input_zp, const int32_t* weight_zps
    ) -> void;

    auto quantized_conv2d_per_channel_kernel(
        const int8_t* input, const int8_t* weight, const float* bias,
        float* output, int64_t batch, int64_t in_channels, int64_t out_channels,
        int64_t h_in, int64_t w_in, int64_t h_out, int64_t w_out,
        int64_t kernel_size, int64_t stride, int64_t padding,
        float input_scale, const float* weight_scales, int32_t input_zp, const int32_t* weight_zps,
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
    // QuantizedLinear is an inference-only int8 layer (analogous to
    // torch.ao.nn.quantized.Linear). The int8 matmul is not differentiable;
    // gradients cannot flow through. For QAT training, compose a regular
    // Linear with a FakeQuantize module, which uses the STE backward path.
    // Returning a Variable with requires_grad=false makes this honest.
    auto q_input = quantize_per_tensor_symmetric(input.tensor());
    Tensor output = forward_quantized(q_input);
    return Variable(output, /*requires_grad=*/false);
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
    int32_t input_zp = input_zp_cpu.data<int32_t>()[0];

    bool is_per_channel = (weight_params.scheme == QuantizationScheme::PerChannelSymmetric ||
                           weight_params.scheme == QuantizationScheme::PerChannelAsymmetric);

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

    // INT4 (QInt4x2) weight path: use fused dequantizing matmul
    if (weight_data_cpu.dtype() == DType::QInt4x2) {
        const uint8_t* weight_packed = weight_data_cpu.data<uint8_t>();
        float weight_scale = weight_scale_cpu.data<const float>()[0];
        cpu::fused_qlinear_dequant(
            input_data, weight_packed, bias_data, output_data,
            batch_size, out_features_, in_features_,
            input_scale, weight_scale);
        return output.to(original_device);
    }

    const int8_t* weight_data = weight_data_cpu.data<int8_t>();

    if (is_per_channel) {
        // Per-channel: weight_scale_cpu has [out_features] scales
        const float* weight_scales = weight_scale_cpu.data<const float>();
        const int32_t* weight_zps = weight_zp_cpu.data<int32_t>();
        kernels::quantized_linear_per_channel_kernel(
            input_data, weight_data, bias_data, output_data,
            batch_size, in_features_, out_features_,
            input_scale, weight_scales, bias_scale_,
            input_zp, weight_zps
        );
    } else {
        // Per-tensor: scalar scale and zero point
        float weight_scale = weight_scale_cpu.data<const float>()[0];
        int32_t weight_zp = weight_zp_cpu.data<int32_t>()[0];
        kernels::quantized_linear_kernel(
            input_data, weight_data, bias_data, output_data,
            batch_size, in_features_, out_features_,
            input_scale, weight_scale, bias_scale_,
            input_zp, weight_zp
        );
    }

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
    // Inference-only: see QuantizedLinear::forward_impl for rationale.
    // Compose Conv2d + FakeQuantize for QAT training instead.
    auto q_input = quantize_per_tensor_symmetric(input.tensor());
    Tensor output = forward_quantized(q_input);
    return Variable(output, /*requires_grad=*/false);
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
    int32_t input_zp = input_zp_cpu.data<int32_t>()[0];

    bool is_per_channel = (weight_params.scheme == QuantizationScheme::PerChannelSymmetric ||
                           weight_params.scheme == QuantizationScheme::PerChannelAsymmetric);

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

    if (is_per_channel) {
        const float* weight_scales = weight_scale_cpu.data<const float>();
        const int32_t* weight_zps = weight_zp_cpu.data<int32_t>();
        kernels::quantized_conv2d_per_channel_kernel(
            input_data, weight_data, bias_data, output_data,
            batch, in_channels_, out_channels_,
            h_in, w_in, h_out, w_out,
            kernel_size_, stride_, padding_,
            input_scale, weight_scales, input_zp, weight_zps,
            dilation_, groups_
        );
    } else {
        float weight_scale = weight_scale_cpu.data<const float>()[0];
        int32_t weight_zp = weight_zp_cpu.data<int32_t>()[0];
        kernels::quantized_conv2d_kernel(
            input_data, weight_data, bias_data, output_data,
            batch, in_channels_, out_channels_,
            h_in, w_in, h_out, w_out,
            kernel_size_, stride_, padding_,
            input_scale, weight_scale, input_zp, weight_zp,
            dilation_, groups_
        );
    }

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

// ============================================================================
// QuantizedEmbedding
// ============================================================================

QuantizedEmbedding::QuantizedEmbedding(
    int64_t num_embeddings,
    int64_t embedding_dim,
    QuantizationParams weight_qparams,
    int64_t padding_idx
) : num_embeddings_(num_embeddings),
    embedding_dim_(embedding_dim),
    padding_idx_(padding_idx),
    weight_(Tensor({num_embeddings, embedding_dim}, DType::Int8, Device::cpu()),
            std::move(weight_qparams)) {}

auto QuantizedEmbedding::forward_impl(const Variable& input) -> Variable {
    Tensor output = forward_quantized(input.tensor());
    return Variable(output, false);  // Quantized inference, no grad
}

auto QuantizedEmbedding::forward_quantized(const Tensor& indices) -> Tensor {
    // Look up quantized rows and dequantize them on-the-fly
    int64_t num_indices = indices.numel();
    Tensor output({num_indices, embedding_dim_}, DType::Float32, Device::cpu());

    // Get quantization parameters on CPU
    Tensor scale_cpu = weight_.params().scale;
    Tensor zp_cpu = weight_.params().zero_point;
    if (scale_cpu.device() != Device::cpu()) scale_cpu = scale_cpu.to(Device::cpu());
    if (zp_cpu.device() != Device::cpu()) zp_cpu = zp_cpu.to(Device::cpu());

    float scale = scale_cpu.data<float>()[0];
    int32_t zp = zp_cpu.data<int32_t>()[0];

    Tensor weight_cpu = weight_.data();
    if (weight_cpu.device() != Device::cpu()) weight_cpu = weight_cpu.to(Device::cpu());
    const int8_t* w_data = weight_cpu.data<int8_t>();

    Tensor indices_cpu = indices;
    if (indices_cpu.device() != Device::cpu()) indices_cpu = indices_cpu.to(Device::cpu());
    const int64_t* idx_data = indices_cpu.data<int64_t>();

    float* out_data = output.data<float>();

    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = idx_data[i];
        if (idx < 0) idx += num_embeddings_;
        if (idx < 0 || idx >= num_embeddings_) {
            throw std::out_of_range("QuantizedEmbedding: index " +
                std::to_string(idx_data[i]) + " out of range [0, " +
                std::to_string(num_embeddings_) + ")");
        }

        const int8_t* row = w_data + idx * embedding_dim_;
        float* out_row = out_data + i * embedding_dim_;

        // Dequantize: fp = (q - zp) * scale
        for (int64_t j = 0; j < embedding_dim_; ++j) {
            out_row[j] = (static_cast<float>(row[j]) - zp) * scale;
        }

        // Zero out padding index
        if (idx == padding_idx_) {
            for (int64_t j = 0; j < embedding_dim_; ++j) {
                out_row[j] = 0.0f;
            }
        }
    }

    // Transfer to original device if needed
    if (indices.device() != Device::cpu()) {
        output = output.to(indices.device());
    }

    return output;
}

auto QuantizedEmbedding::set_weight(const QuantizedTensor& weights) -> void {
    weight_ = weights;
}

auto QuantizedEmbedding::from_float(Module& fp_embedding,
                                     const QConfig& qconfig)
    -> std::shared_ptr<QuantizedEmbedding> {
    // Extract weight from the embedding module
    auto params = fp_embedding.named_parameters();
    Tensor weight;
    bool found = false;
    for (auto& [name, var] : params) {
        if (name == "weight") { weight = var->tensor(); found = true; break; }
    }
    if (!found) {
        throw std::runtime_error("QuantizedEmbedding::from_float: module has no 'weight' parameter");
    }
    auto shape = weight.shape();
    int64_t num_embeddings = shape[0];
    int64_t embedding_dim = shape[1];

    // Quantize weight
    auto weight_cpu = (weight.device() == Device::cpu()) ? weight : weight.to(Device::cpu());
    auto q_weight = quantize_per_tensor_symmetric(weight_cpu);

    auto result = std::make_shared<QuantizedEmbedding>(
        num_embeddings, embedding_dim, q_weight.params());
    result->set_weight(q_weight);
    return result;
}

// ============================================================================
// QuantizedLSTM
// ============================================================================

QuantizedLSTM::QuantizedLSTM(
    int64_t input_size,
    int64_t hidden_size,
    int64_t num_layers,
    bool bias,
    bool batch_first,
    bool bidirectional,
    QuantizationParams weight_qparams
) : input_size_(input_size),
    hidden_size_(hidden_size),
    num_layers_(num_layers),
    bias_(bias),
    batch_first_(batch_first),
    bidirectional_(bidirectional) {

    int64_t num_directions = bidirectional ? 2 : 1;
    layers_.reserve(num_layers * num_directions);

    for (int64_t layer = 0; layer < num_layers_ * num_directions; ++layer) {
        int64_t in_sz = (layer == 0) ? input_size : hidden_size * num_directions;
        // 4 gates: input, forget, cell, output
        QuantizedTensor w_ih(
            Tensor({4 * hidden_size, in_sz}, DType::Int8, Device::cpu()),
            weight_qparams);
        QuantizedTensor w_hh(
            Tensor({4 * hidden_size, hidden_size}, DType::Int8, Device::cpu()),
            weight_qparams);

        LayerWeights lw{std::move(w_ih), std::move(w_hh), std::nullopt, std::nullopt};
        if (bias) {
            lw.bias_ih = zeros({4 * hidden_size}, DType::Float32, Device::cpu());
            lw.bias_hh = zeros({4 * hidden_size}, DType::Float32, Device::cpu());
        }
        layers_.push_back(std::move(lw));
    }
}

auto QuantizedLSTM::forward_impl(const Variable& input) -> Variable {
    // Default: zero initial state
    auto shape = input.shape();
    int64_t batch = batch_first_ ? shape[0] : shape[1];
    int64_t num_directions = bidirectional_ ? 2 : 1;

    auto h0 = Variable(zeros({num_layers_ * num_directions, batch, hidden_size_}), false);
    auto c0 = Variable(zeros({num_layers_ * num_directions, batch, hidden_size_}), false);

    auto [output, hn, cn] = forward_with_state(input, h0, c0);
    return output;
}

auto QuantizedLSTM::forward_with_state(const Variable& input,
                                        const Variable& h0, const Variable& c0)
    -> std::tuple<Variable, Variable, Variable> {
    // Dequantize weights and run standard LSTM computation in FP32
    // This preserves the gate nonlinearities (sigmoid/tanh) accuracy
    // while reducing model memory footprint

    auto inp = input.tensor();
    if (batch_first_) {
        inp = inp.permute({1, 0, 2});  // -> [seq_len, batch, features]
    }

    auto seq_len = inp.shape()[0];
    auto batch = inp.shape()[1];
    int64_t num_directions = bidirectional_ ? 2 : 1;

    // Process through layers with dequantized weights
    Tensor current_input = inp.to(DType::Float32);

    auto h_n = zeros({num_layers_ * num_directions, batch, hidden_size_});
    auto c_n = zeros({num_layers_ * num_directions, batch, hidden_size_});

    for (int64_t layer = 0; layer < num_layers_; ++layer) {
        for (int64_t dir = 0; dir < num_directions; ++dir) {
            int64_t idx = layer * num_directions + dir;
            auto& lw = layers_[idx];

            // Dequantize weights for this layer
            Tensor w_ih = lw.weight_ih.dequantize();
            Tensor w_hh = lw.weight_hh.dequantize();

            auto h = h0.tensor().slice(0, idx, idx + 1).squeeze(0);
            auto c = c0.tensor().slice(0, idx, idx + 1).squeeze(0);

            std::vector<Tensor> outputs;
            outputs.reserve(seq_len);

            int64_t t_start = (dir == 0) ? 0 : seq_len - 1;
            int64_t t_end = (dir == 0) ? seq_len : -1;
            int64_t t_step = (dir == 0) ? 1 : -1;

            for (int64_t t = t_start; t != t_end; t += t_step) {
                auto x_t = current_input.slice(0, t, t + 1).squeeze(0);

                // gates = x_t @ w_ih^T + h @ w_hh^T + bias
                auto gates = matmul(x_t, w_ih.permute({1, 0}));
                gates = gates + matmul(h, w_hh.permute({1, 0}));
                if (lw.bias_ih) gates = gates + *lw.bias_ih;
                if (lw.bias_hh) gates = gates + *lw.bias_hh;

                // Split gates: [batch, 4*hidden] -> 4x [batch, hidden]
                auto i_gate = sigmoid(gates.slice(1, 0, hidden_size_));
                auto f_gate = sigmoid(gates.slice(1, hidden_size_, 2 * hidden_size_));
                auto g_gate = tanh(gates.slice(1, 2 * hidden_size_, 3 * hidden_size_));
                auto o_gate = sigmoid(gates.slice(1, 3 * hidden_size_, 4 * hidden_size_));

                c = f_gate * c + i_gate * g_gate;
                h = o_gate * tanh(c);

                outputs.push_back(h.unsqueeze(0));
            }

            // Store final states
            // (h_n and c_n slice assignment would go here in a full implementation)

            if (dir == 1) {
                std::reverse(outputs.begin(), outputs.end());
            }
        }
    }

    auto output = current_input;  // Simplified — full impl concatenates layer outputs
    if (batch_first_) {
        output = output.permute({1, 0, 2});
    }

    return {Variable(output, false), Variable(h_n, false), Variable(c_n, false)};
}

auto QuantizedLSTM::from_float(Module& fp_lstm, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedLSTM> {
    // Extract parameters from fp_lstm and quantize weights
    auto params = fp_lstm.named_parameters();

    auto get_param = [&](const std::string& name) -> Tensor {
        for (auto& [pname, var] : params) {
            if (pname == name) return var->tensor();
        }
        throw std::runtime_error("QuantizedLSTM::from_float: missing parameter " + name);
    };

    // Determine sizes from weight_ih_l0
    auto w_ih_0 = get_param("weight_ih_l0");
    int64_t hidden_size = w_ih_0.shape()[0] / 4;
    int64_t input_size = w_ih_0.shape()[1];

    auto has_param = [&](const std::string& name) -> bool {
        for (auto& [pname, _] : params) { if (pname == name) return true; }
        return false;
    };

    // Count layers
    int64_t num_layers = 0;
    while (has_param("weight_ih_l" + std::to_string(num_layers))) {
        num_layers++;
    }
    bool bidirectional = has_param("weight_ih_l0_reverse");
    bool has_bias = has_param("bias_ih_l0");

    auto result = std::make_shared<QuantizedLSTM>(
        input_size, hidden_size, num_layers, has_bias, true, bidirectional);

    int64_t num_directions = bidirectional ? 2 : 1;
    for (int64_t layer = 0; layer < num_layers; ++layer) {
        for (int64_t dir = 0; dir < num_directions; ++dir) {
            int64_t idx = layer * num_directions + dir;
            std::string suffix = "l" + std::to_string(layer);
            if (dir == 1) suffix += "_reverse";

            auto w_ih = get_param("weight_ih_" + suffix);
            auto w_hh = get_param("weight_hh_" + suffix);
            auto w_ih_cpu = (w_ih.device() == Device::cpu()) ? w_ih : w_ih.to(Device::cpu());
            auto w_hh_cpu = (w_hh.device() == Device::cpu()) ? w_hh : w_hh.to(Device::cpu());

            result->layers_[idx].weight_ih = quantize_per_tensor_symmetric(w_ih_cpu);
            result->layers_[idx].weight_hh = quantize_per_tensor_symmetric(w_hh_cpu);

            if (has_bias) {
                result->layers_[idx].bias_ih = get_param("bias_ih_" + suffix);
                result->layers_[idx].bias_hh = get_param("bias_hh_" + suffix);
            }
        }
    }

    return result;
}

// ============================================================================
// QuantizedConv3d
// ============================================================================

QuantizedConv3d::QuantizedConv3d(
    int64_t in_channels,
    int64_t out_channels,
    std::array<int64_t, 3> kernel_size,
    std::array<int64_t, 3> stride,
    std::array<int64_t, 3> padding,
    std::array<int64_t, 3> dilation,
    int64_t groups,
    QuantizationParams weight_qparams
) : in_channels_(in_channels),
    out_channels_(out_channels),
    kernel_size_(kernel_size),
    stride_(stride),
    padding_(padding),
    dilation_(dilation),
    groups_(groups),
    weight_(Tensor({out_channels, in_channels / groups,
                    kernel_size[0], kernel_size[1], kernel_size[2]},
                   DType::Int8, Device::cpu()),
            std::move(weight_qparams)) {}

auto QuantizedConv3d::forward_impl(const Variable& input) -> Variable {
    auto q_input = quantize_per_tensor_symmetric(input.tensor());
    Tensor output = forward_quantized(q_input);
    return Variable(output, false);
}

auto QuantizedConv3d::forward_quantized(const QuantizedTensor& input) -> Tensor {
    // Dequantize and run FP32 conv3d as fallback
    // A fused INT8 3D convolution kernel would be more efficient
    Tensor fp_input = input.dequantize();
    Tensor fp_weight = weight_.dequantize();

    // Use the standard conv3d operation
    auto input_shape = fp_input.shape();
    int64_t batch = input_shape[0];
    int64_t D_in = input_shape[2], H_in = input_shape[3], W_in = input_shape[4];

    int64_t D_out = (D_in + 2 * padding_[0] - dilation_[0] * (kernel_size_[0] - 1) - 1) / stride_[0] + 1;
    int64_t H_out = (H_in + 2 * padding_[1] - dilation_[1] * (kernel_size_[1] - 1) - 1) / stride_[1] + 1;
    int64_t W_out = (W_in + 2 * padding_[2] - dilation_[2] * (kernel_size_[2] - 1) - 1) / stride_[2] + 1;

    Tensor output = zeros({batch, out_channels_, D_out, H_out, W_out},
                          DType::Float32, fp_input.device());

    // Naive implementation — production would use fused INT8 kernel
    // For now, delegate to standard conv3d if available or implement naively
    // This provides correct results; optimization is future work
    auto fp_input_cpu = (fp_input.device() == Device::cpu()) ? fp_input : fp_input.to(Device::cpu());
    auto fp_weight_cpu = (fp_weight.device() == Device::cpu()) ? fp_weight : fp_weight.to(Device::cpu());
    auto output_cpu = (output.device() == Device::cpu()) ? output : output.to(Device::cpu());

    const float* in_data = fp_input_cpu.data<float>();
    const float* w_data = fp_weight_cpu.data<float>();
    float* out_data = output_cpu.data<float>();

    int64_t in_c_per_group = in_channels_ / groups_;
    int64_t out_c_per_group = out_channels_ / groups_;

    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t g = 0; g < groups_; ++g) {
            for (int64_t oc = 0; oc < out_c_per_group; ++oc) {
                int64_t oc_abs = g * out_c_per_group + oc;
                for (int64_t od = 0; od < D_out; ++od) {
                    for (int64_t oh = 0; oh < H_out; ++oh) {
                        for (int64_t ow = 0; ow < W_out; ++ow) {
                            float sum = 0.0f;
                            if (bias_) sum = bias_->data<float>()[oc_abs];

                            for (int64_t ic = 0; ic < in_c_per_group; ++ic) {
                                int64_t ic_abs = g * in_c_per_group + ic;
                                for (int64_t kd = 0; kd < kernel_size_[0]; ++kd) {
                                    for (int64_t kh = 0; kh < kernel_size_[1]; ++kh) {
                                        for (int64_t kw = 0; kw < kernel_size_[2]; ++kw) {
                                            int64_t id = od * stride_[0] - padding_[0] + kd * dilation_[0];
                                            int64_t ih = oh * stride_[1] - padding_[1] + kh * dilation_[1];
                                            int64_t iw = ow * stride_[2] - padding_[2] + kw * dilation_[2];

                                            if (id >= 0 && id < D_in && ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                                int64_t in_idx = ((n * in_channels_ + ic_abs) * D_in + id) * H_in * W_in + ih * W_in + iw;
                                                int64_t w_idx = ((oc_abs * in_c_per_group + ic) * kernel_size_[0] + kd) * kernel_size_[1] * kernel_size_[2] + kh * kernel_size_[2] + kw;
                                                sum += in_data[in_idx] * w_data[w_idx];
                                            }
                                        }
                                    }
                                }
                            }

                            int64_t out_idx = ((n * out_channels_ + oc_abs) * D_out + od) * H_out * W_out + oh * W_out + ow;
                            out_data[out_idx] = sum;
                        }
                    }
                }
            }
        }
    }

    if (output.device() != Device::cpu()) {
        output = output_cpu.to(output.device());
    } else {
        output = output_cpu;
    }

    return output;
}

auto QuantizedConv3d::set_weight(const QuantizedTensor& weights) -> void {
    weight_ = weights;
}

auto QuantizedConv3d::set_bias(const Tensor& bias) -> void {
    bias_ = bias;
}

auto QuantizedConv3d::from_float(Module& fp_conv3d, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedConv3d> {
    auto params = fp_conv3d.named_parameters();

    Tensor weight;
    bool found_weight = false;
    for (auto& [name, var] : params) {
        if (name == "weight") { weight = var->tensor(); found_weight = true; break; }
    }
    if (!found_weight) {
        throw std::runtime_error("QuantizedConv3d::from_float: module has no 'weight' parameter");
    }
    auto shape = weight.shape();
    int64_t out_ch = shape[0];
    int64_t in_ch_per_group = shape[1];
    std::array<int64_t, 3> ks = {shape[2], shape[3], shape[4]};

    // Extract stride/padding/dilation/groups from the source Conv3d module
    std::array<int64_t, 3> stride = {1, 1, 1};
    std::array<int64_t, 3> padding = {0, 0, 0};
    std::array<int64_t, 3> dilation = {1, 1, 1};
    int64_t groups = 1;
    if (auto* conv3d = dynamic_cast<nn::Conv3d*>(&fp_conv3d)) {
        stride = {conv3d->stride(), conv3d->stride(), conv3d->stride()};
        padding = {conv3d->padding(), conv3d->padding(), conv3d->padding()};
        dilation = {conv3d->dilation(), conv3d->dilation(), conv3d->dilation()};
        groups = conv3d->groups();
        in_ch_per_group = in_ch_per_group * groups;  // Convert per-group to total in_channels
    }

    auto weight_cpu = (weight.device() == Device::cpu()) ? weight : weight.to(Device::cpu());
    auto q_weight = quantize_per_tensor_symmetric(weight_cpu);

    auto result = std::make_shared<QuantizedConv3d>(
        in_ch_per_group, out_ch, ks, stride,
        padding, dilation, groups,
        q_weight.params());
    result->set_weight(q_weight);

    for (auto& [name, var] : params) {
        if (name == "bias") { result->set_bias(var->tensor()); break; }
    }

    return result;
}

// ============================================================================
// QuantizedMultiheadAttention
// ============================================================================

QuantizedMultiheadAttention::QuantizedMultiheadAttention(
    int64_t embed_dim,
    int64_t num_heads,
    QuantizationParams weight_qparams,
    bool bias,
    float dropout
) : embed_dim_(embed_dim),
    num_heads_(num_heads),
    head_dim_(embed_dim / num_heads),
    dropout_(dropout) {

    if (embed_dim % num_heads != 0) {
        throw std::runtime_error("QuantizedMultiheadAttention: embed_dim must be divisible by num_heads");
    }

    q_proj_ = std::make_shared<QuantizedLinear>(embed_dim, embed_dim, weight_qparams);
    k_proj_ = std::make_shared<QuantizedLinear>(embed_dim, embed_dim, weight_qparams);
    v_proj_ = std::make_shared<QuantizedLinear>(embed_dim, embed_dim, weight_qparams);
    out_proj_ = std::make_shared<QuantizedLinear>(embed_dim, embed_dim, weight_qparams);
}

auto QuantizedMultiheadAttention::forward_impl(const Variable& input) -> Variable {
    // Self-attention: Q=K=V=input
    auto [output, weights] = forward_qkv(input, input, input);
    return output;
}

auto QuantizedMultiheadAttention::forward_qkv(
    const Variable& query, const Variable& key,
    const Variable& value, const Variable& attn_mask)
    -> std::pair<Variable, Variable> {

    auto shape = query.shape();
    int64_t seq_len = shape[0];
    int64_t batch = shape[1];

    // INT8 linear projections, dequantized to FP32
    auto q = q_proj_->forward_impl(query);
    auto k = k_proj_->forward_impl(key);
    auto v = v_proj_->forward_impl(value);

    // Reshape for multi-head: [seq, batch, embed] -> [seq, batch, heads, head_dim] -> [batch*heads, seq, head_dim]
    auto reshape_for_heads = [&](const Variable& x) -> Tensor {
        auto t = x.tensor().reshape({seq_len, batch, num_heads_, head_dim_});
        t = t.permute({1, 2, 0, 3});  // [batch, heads, seq, head_dim]
        return t.reshape({batch * num_heads_, seq_len, head_dim_});
    };

    auto q_heads = reshape_for_heads(q);
    auto k_heads = reshape_for_heads(k);
    auto v_heads = reshape_for_heads(v);

    // Scaled dot-product attention in FP32
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    auto scores = matmul(q_heads, k_heads.permute({0, 2, 1})) * scale;

    if (attn_mask.is_initialized()) {
        scores = scores + attn_mask.tensor();
    }

    // softmax via autograd (no Tensor-level softmax exists)
    auto attn_weights = tenzor::softmax(Variable(scores, false), -1).tensor();
    auto context = matmul(attn_weights, v_heads);

    // Reshape back: [batch*heads, seq, head_dim] -> [seq, batch, embed]
    context = context.reshape({batch, num_heads_, seq_len, head_dim_});
    context = context.permute({2, 0, 1, 3}).reshape({seq_len, batch, embed_dim_});

    // Output projection (INT8)
    auto output = out_proj_->forward_impl(Variable(context, false));

    return {output, Variable(attn_weights.reshape({batch, num_heads_, seq_len, seq_len}), false)};
}

auto QuantizedMultiheadAttention::from_float(Module& fp_mha,
                                              const QConfig& qconfig)
    -> std::shared_ptr<QuantizedMultiheadAttention> {
    auto params = fp_mha.named_parameters();

    auto find_param = [&](const std::string& name) -> Tensor* {
        for (auto& [pname, var] : params) {
            if (pname == name) return &var->tensor();
        }
        return nullptr;
    };

    // Determine embed_dim and num_heads from q_proj weight shape
    Tensor* q_weight_ptr = find_param("q_proj.weight");
    if (!q_weight_ptr) q_weight_ptr = find_param("in_proj_weight");
    if (!q_weight_ptr) {
        throw std::runtime_error("QuantizedMultiheadAttention::from_float: cannot find projection weights");
    }

    auto q_weight = *q_weight_ptr;
    int64_t embed_dim = q_weight.shape()[1];

    // Determine num_heads (may need to be passed as parameter in practice)
    // Default: try head_dim=64
    int64_t num_heads = embed_dim / 64;
    if (num_heads < 1) num_heads = 1;

    auto weight_cpu = (q_weight.device() == Device::cpu()) ? q_weight : q_weight.to(Device::cpu());
    auto q_params = quantize_per_tensor_symmetric(weight_cpu).params();

    auto result = std::make_shared<QuantizedMultiheadAttention>(
        embed_dim, num_heads, q_params);

    // Quantize individual projection weights
    auto set_proj = [&](const std::string& name, std::shared_ptr<QuantizedLinear>& proj) {
        if (auto* w = find_param(name + ".weight")) {
            auto w_cpu = (w->device() == Device::cpu()) ? *w : w->to(Device::cpu());
            proj->set_weight(quantize_per_tensor_symmetric(w_cpu));
        }
        if (auto* b = find_param(name + ".bias")) {
            proj->set_bias(*b);
        }
    };

    set_proj("q_proj", result->q_proj_);
    set_proj("k_proj", result->k_proj_);
    set_proj("v_proj", result->v_proj_);
    set_proj("out_proj", result->out_proj_);

    return result;
}

// ============================================================================
// QuantizedGRU
// ============================================================================

QuantizedGRU::QuantizedGRU(
    int64_t input_size,
    int64_t hidden_size,
    int64_t num_layers,
    bool bias,
    bool batch_first,
    bool bidirectional,
    QuantizationParams weight_qparams
) : input_size_(input_size),
    hidden_size_(hidden_size),
    num_layers_(num_layers),
    bias_(bias),
    batch_first_(batch_first),
    bidirectional_(bidirectional) {

    int64_t num_directions = bidirectional ? 2 : 1;
    layers_.reserve(num_layers * num_directions);

    for (int64_t layer = 0; layer < num_layers; ++layer) {
        for (int64_t dir = 0; dir < num_directions; ++dir) {
            int64_t layer_input_size = (layer == 0) ? input_size : hidden_size * num_directions;

            auto w_ih = zeros({3 * hidden_size, layer_input_size}, DType::Float32);
            auto w_hh = zeros({3 * hidden_size, hidden_size}, DType::Float32);

            LayerWeights lw{
                .weight_ih = quantize_per_tensor_symmetric(w_ih),
                .weight_hh = quantize_per_tensor_symmetric(w_hh),
                .bias_ih = bias ? std::optional{zeros({3 * hidden_size}, DType::Float32)} : std::nullopt,
                .bias_hh = bias ? std::optional{zeros({3 * hidden_size}, DType::Float32)} : std::nullopt,
            };
            layers_.push_back(std::move(lw));
        }
    }
}

auto QuantizedGRU::forward_impl(const Variable& input) -> Variable {
    auto h0 = Variable(zeros({num_layers_ * (bidirectional_ ? 2 : 1),
                               input.tensor().shape()[batch_first_ ? 0 : 1],
                               hidden_size_}, input.tensor().dtype()), false);
    auto [output, h_n] = forward_with_state(input, h0);
    return output;
}

auto QuantizedGRU::forward_with_state(const Variable& input, const Variable& h0)
    -> std::pair<Variable, Variable> {

    auto x = input.tensor();
    if (batch_first_) {
        x = x.permute({1, 0, 2}); // [batch, seq, feat] -> [seq, batch, feat]
    }

    int64_t seq_len = x.shape()[0];
    int64_t batch = x.shape()[1];
    int64_t num_directions = bidirectional_ ? 2 : 1;

    auto h_n = zeros({num_layers_ * num_directions, batch, hidden_size_}, x.dtype());
    Tensor output;

    for (int64_t layer = 0; layer < num_layers_; ++layer) {
        Tensor layer_output;

        for (int64_t dir = 0; dir < num_directions; ++dir) {
            int64_t idx = layer * num_directions + dir;
            auto& lw = layers_[idx];

            // Dequantize weights to FP32
            auto w_ih = lw.weight_ih.dequantize();
            auto w_hh = lw.weight_hh.dequantize();

            auto h = h0.tensor().slice(0, idx, idx + 1).squeeze(0);

            std::vector<Tensor> outputs;
            outputs.reserve(seq_len);

            int64_t t_start = (dir == 0) ? 0 : seq_len - 1;
            int64_t t_end = (dir == 0) ? seq_len : -1;
            int64_t t_step = (dir == 0) ? 1 : -1;

            Tensor current_input = (layer == 0) ? x : output;

            for (int64_t t = t_start; t != t_end; t += t_step) {
                auto x_t = current_input.slice(0, t, t + 1).squeeze(0);

                // gates_x = x_t @ w_ih^T + bias_ih
                auto gates_x = matmul(x_t, w_ih.permute({1, 0}));
                if (lw.bias_ih) gates_x = gates_x + *lw.bias_ih;

                // gates_h = h @ w_hh^T + bias_hh
                auto gates_h = matmul(h, w_hh.permute({1, 0}));
                if (lw.bias_hh) gates_h = gates_h + *lw.bias_hh;

                // Split: [batch, 3*hidden] -> r, z, n
                auto r_x = gates_x.slice(1, 0, hidden_size_);
                auto z_x = gates_x.slice(1, hidden_size_, 2 * hidden_size_);
                auto n_x = gates_x.slice(1, 2 * hidden_size_, 3 * hidden_size_);

                auto r_h = gates_h.slice(1, 0, hidden_size_);
                auto z_h = gates_h.slice(1, hidden_size_, 2 * hidden_size_);
                auto n_h = gates_h.slice(1, 2 * hidden_size_, 3 * hidden_size_);

                auto r_gate = sigmoid(r_x + r_h);
                auto z_gate = sigmoid(z_x + z_h);
                auto n_gate = tanh(n_x + r_gate * n_h);

                h = (ones_like(z_gate) - z_gate) * n_gate + z_gate * h;
                outputs.push_back(h.unsqueeze(0));
            }

            if (dir == 1) {
                std::reverse(outputs.begin(), outputs.end());
            }

            auto dir_output = cat(outputs, 0);
            if (dir == 0) {
                layer_output = dir_output;
            } else {
                layer_output = cat({layer_output, dir_output}, 2);
            }
        }

        output = layer_output;
    }

    if (batch_first_) {
        output = output.permute({1, 0, 2});
    }

    return {Variable(output, false), Variable(h_n, false)};
}

auto QuantizedGRU::from_float(Module& fp_gru, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedGRU> {
    auto params = fp_gru.named_parameters();

    auto get_param = [&](const std::string& name) -> Tensor {
        for (auto& [pname, var] : params) {
            if (pname == name) return var->tensor();
        }
        throw std::runtime_error("QuantizedGRU::from_float: missing parameter " + name);
    };

    auto has_param = [&](const std::string& name) -> bool {
        for (auto& [pname, _] : params) { if (pname == name) return true; }
        return false;
    };

    auto w_ih_0 = get_param("weight_ih_l0");
    int64_t hidden_size = w_ih_0.shape()[0] / 3;
    int64_t input_size = w_ih_0.shape()[1];

    int64_t num_layers = 0;
    while (has_param("weight_ih_l" + std::to_string(num_layers))) {
        num_layers++;
    }
    bool bidirectional = has_param("weight_ih_l0_reverse");
    bool has_bias = has_param("bias_ih_l0");

    auto result = std::make_shared<QuantizedGRU>(
        input_size, hidden_size, num_layers, has_bias, true, bidirectional);

    int64_t num_directions = bidirectional ? 2 : 1;
    for (int64_t layer = 0; layer < num_layers; ++layer) {
        for (int64_t dir = 0; dir < num_directions; ++dir) {
            int64_t idx = layer * num_directions + dir;
            std::string suffix = "l" + std::to_string(layer);
            if (dir == 1) suffix += "_reverse";

            auto w_ih = get_param("weight_ih_" + suffix);
            auto w_hh = get_param("weight_hh_" + suffix);
            auto w_ih_cpu = (w_ih.device() == Device::cpu()) ? w_ih : w_ih.to(Device::cpu());
            auto w_hh_cpu = (w_hh.device() == Device::cpu()) ? w_hh : w_hh.to(Device::cpu());

            result->layers_[idx].weight_ih = quantize_per_tensor_symmetric(w_ih_cpu);
            result->layers_[idx].weight_hh = quantize_per_tensor_symmetric(w_hh_cpu);

            if (has_bias) {
                result->layers_[idx].bias_ih = get_param("bias_ih_" + suffix);
                result->layers_[idx].bias_hh = get_param("bias_hh_" + suffix);
            }
        }
    }

    return result;
}

// ============================================================================
// QuantizedConv1d
// ============================================================================

QuantizedConv1d::QuantizedConv1d(
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
    weight_(Tensor({out_channels, in_channels / groups, kernel_size},
                   DType::Int8, Device::cpu()), std::move(weight_qparams)),
    bias_scale_(bias_scale) {}

auto QuantizedConv1d::forward_impl(const Variable& input) -> Variable {
    auto q_input = quantize_per_tensor_symmetric(input.tensor());
    Tensor output = forward_quantized(q_input);
    return Variable(output, input.requires_grad());
}

auto QuantizedConv1d::forward_quantized(const QuantizedTensor& input) -> Tensor {
    // Dequantize and run FP32 conv1d as fallback
    // A fused INT8 1D convolution kernel would be more efficient
    Tensor fp_input = input.dequantize();
    Tensor fp_weight = weight_.dequantize();

    auto input_shape = fp_input.shape();
    int64_t batch = input_shape[0];
    int64_t l_in = input_shape[2];

    int64_t l_out = (l_in + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;

    auto original_device = fp_input.device();
    Tensor output = zeros({batch, out_channels_, l_out}, DType::Float32, Device::cpu());

    auto fp_input_cpu = (fp_input.device() == Device::cpu()) ? fp_input : fp_input.to(Device::cpu());
    auto fp_weight_cpu = (fp_weight.device() == Device::cpu()) ? fp_weight : fp_weight.to(Device::cpu());

    const float* in_data = fp_input_cpu.data<float>();
    const float* w_data = fp_weight_cpu.data<float>();
    float* out_data = output.data<float>();

    int64_t in_c_per_group = in_channels_ / groups_;
    int64_t out_c_per_group = out_channels_ / groups_;

    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t g = 0; g < groups_; ++g) {
            for (int64_t oc = 0; oc < out_c_per_group; ++oc) {
                int64_t oc_abs = g * out_c_per_group + oc;
                for (int64_t ol = 0; ol < l_out; ++ol) {
                    float sum = 0.0f;
                    if (bias_.has_value()) {
                        Tensor bias_cpu = *bias_;
                        if (bias_cpu.device() != Device::cpu())
                            bias_cpu = bias_cpu.to(Device::cpu());
                        sum = bias_cpu.data<const float>()[oc_abs];
                    }

                    for (int64_t ic = 0; ic < in_c_per_group; ++ic) {
                        int64_t ic_abs = g * in_c_per_group + ic;
                        for (int64_t k = 0; k < kernel_size_; ++k) {
                            int64_t il = ol * stride_ - padding_ + k * dilation_;
                            if (il >= 0 && il < l_in) {
                                int64_t in_idx = (n * in_channels_ + ic_abs) * l_in + il;
                                int64_t w_idx = (oc_abs * in_c_per_group + ic) * kernel_size_ + k;
                                sum += in_data[in_idx] * w_data[w_idx];
                            }
                        }
                    }

                    int64_t out_idx = (n * out_channels_ + oc_abs) * l_out + ol;
                    out_data[out_idx] = sum;
                }
            }
        }
    }

    return output.to(original_device);
}

auto QuantizedConv1d::forward_quantized_output(
    const QuantizedTensor& input,
    const QuantizationParams& output_qparams
) -> QuantizedTensor {
    Tensor fp_output = forward_quantized(input);
    return quantize_tensor(fp_output, output_qparams);
}

auto QuantizedConv1d::set_weight(const QuantizedTensor& weights) -> void {
    weight_ = weights;
}

auto QuantizedConv1d::set_bias(const Tensor& bias) -> void {
    bias_ = bias;
}

auto QuantizedConv1d::from_float(Module& fp_conv, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedConv1d> {
    auto params = fp_conv.named_parameters();

    Tensor weight;
    bool found_weight = false;
    for (auto& [name, var] : params) {
        if (name == "weight") { weight = var->tensor(); found_weight = true; break; }
    }
    if (!found_weight) {
        throw std::runtime_error("QuantizedConv1d::from_float: module has no 'weight' parameter");
    }

    auto shape = weight.shape();  // [out_channels, in_channels/groups, kernel_size]
    int64_t out_channels = shape[0];
    int64_t in_c_per_group = shape[1];
    int64_t kernel_size = shape[2];

    // Quantize weights
    auto weight_cpu = (weight.device() == Device::cpu()) ? weight : weight.to(Device::cpu());
    auto weight_observer = qconfig.create_weight_observer();
    weight_observer->observe(weight_cpu);
    auto weight_qparams = weight_observer->calculate_qparams(
        qconfig.weight_dtype(), qconfig.weight_scheme());

    // Extract stride/padding/dilation/groups from extra_repr if possible,
    // or use defaults. The Conv1d state_dict only has weight/bias.
    // Use default values; caller can override via constructor if needed.
    auto result = std::make_shared<QuantizedConv1d>(
        in_c_per_group,  // in_channels (assumes groups=1 unless overridden)
        out_channels, kernel_size,
        1, 0, 1, 1,     // stride, padding, dilation, groups defaults
        weight_qparams);

    QuantizedTensor q_weight = quantize_tensor(weight_cpu, weight_qparams);
    result->set_weight(q_weight);

    for (auto& [name, var] : params) {
        if (name == "bias") { result->set_bias(var->tensor()); break; }
    }

    return result;
}

// ============================================================================
// QuantizedConvTranspose2d
// ============================================================================

QuantizedConvTranspose2d::QuantizedConvTranspose2d(
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups,
    QuantizationParams weight_qparams,
    float bias_scale
) : in_channels_(in_channels),
    out_channels_(out_channels),
    kernel_size_(kernel_size),
    stride_(stride),
    padding_(padding),
    output_padding_(output_padding),
    groups_(groups),
    weight_(Tensor({in_channels, out_channels / groups, kernel_size, kernel_size},
                   DType::Int8, Device::cpu()), std::move(weight_qparams)),
    bias_scale_(bias_scale) {}

auto QuantizedConvTranspose2d::forward_impl(const Variable& input) -> Variable {
    auto q_input = quantize_per_tensor_symmetric(input.tensor());
    Tensor output = forward_quantized(q_input);
    return Variable(output, input.requires_grad());
}

auto QuantizedConvTranspose2d::forward_quantized(const QuantizedTensor& input) -> Tensor {
    // Dequantize and run FP32 transposed conv2d
    Tensor fp_input = input.dequantize();
    Tensor fp_weight = weight_.dequantize();

    auto input_shape = fp_input.shape();
    int64_t batch = input_shape[0];
    int64_t h_in = input_shape[2];
    int64_t w_in = input_shape[3];

    // Transposed conv output dims:
    // H_out = (H_in - 1) * stride - 2*padding + kernel_size + output_padding
    int64_t h_out = (h_in - 1) * stride_ - 2 * padding_ + kernel_size_ + output_padding_;
    int64_t w_out = (w_in - 1) * stride_ - 2 * padding_ + kernel_size_ + output_padding_;

    auto original_device = fp_input.device();
    Tensor output = zeros({batch, out_channels_, h_out, w_out}, DType::Float32, Device::cpu());

    auto fp_input_cpu = (fp_input.device() == Device::cpu()) ? fp_input : fp_input.to(Device::cpu());
    auto fp_weight_cpu = (fp_weight.device() == Device::cpu()) ? fp_weight : fp_weight.to(Device::cpu());

    const float* in_data = fp_input_cpu.data<float>();
    const float* w_data = fp_weight_cpu.data<float>();
    float* out_data = output.data<float>();

    int64_t in_c_per_group = in_channels_ / groups_;
    int64_t out_c_per_group = out_channels_ / groups_;

    // Transposed convolution: scatter input through flipped kernel
    for (int64_t n = 0; n < batch; ++n) {
        for (int64_t g = 0; g < groups_; ++g) {
            for (int64_t ic = 0; ic < in_c_per_group; ++ic) {
                int64_t ic_abs = g * in_c_per_group + ic;
                for (int64_t ih = 0; ih < h_in; ++ih) {
                    for (int64_t iw = 0; iw < w_in; ++iw) {
                        float in_val = in_data[((n * in_channels_ + ic_abs) * h_in + ih) * w_in + iw];

                        for (int64_t oc = 0; oc < out_c_per_group; ++oc) {
                            int64_t oc_abs = g * out_c_per_group + oc;
                            for (int64_t kh = 0; kh < kernel_size_; ++kh) {
                                for (int64_t kw = 0; kw < kernel_size_; ++kw) {
                                    int64_t oh = ih * stride_ - padding_ + kh;
                                    int64_t ow = iw * stride_ - padding_ + kw;

                                    if (oh >= 0 && oh < h_out && ow >= 0 && ow < w_out) {
                                        // Weight layout: [in_channels, out_channels/groups, kH, kW]
                                        int64_t w_idx = ((ic_abs * out_c_per_group + oc) * kernel_size_ + kh) * kernel_size_ + kw;
                                        int64_t out_idx = ((n * out_channels_ + oc_abs) * h_out + oh) * w_out + ow;
                                        out_data[out_idx] += in_val * w_data[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Add bias
    if (bias_.has_value()) {
        Tensor bias_cpu = *bias_;
        if (bias_cpu.device() != Device::cpu()) bias_cpu = bias_cpu.to(Device::cpu());
        const float* b_data = bias_cpu.data<const float>();
        int64_t spatial = h_out * w_out;
        for (int64_t n = 0; n < batch; ++n) {
            for (int64_t c = 0; c < out_channels_; ++c) {
                float b = b_data[c];
                float* ch_data = out_data + (n * out_channels_ + c) * spatial;
                for (int64_t i = 0; i < spatial; ++i) {
                    ch_data[i] += b;
                }
            }
        }
    }

    return output.to(original_device);
}

auto QuantizedConvTranspose2d::set_weight(const QuantizedTensor& weights) -> void {
    weight_ = weights;
}

auto QuantizedConvTranspose2d::set_bias(const Tensor& bias) -> void {
    bias_ = bias;
}

auto QuantizedConvTranspose2d::from_float(Module& fp_conv, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedConvTranspose2d> {
    auto params = fp_conv.named_parameters();

    Tensor weight;
    bool found_weight = false;
    for (auto& [name, var] : params) {
        if (name == "weight") { weight = var->tensor(); found_weight = true; break; }
    }
    if (!found_weight) {
        throw std::runtime_error("QuantizedConvTranspose2d::from_float: module has no 'weight' parameter");
    }

    auto shape = weight.shape();  // [in_channels, out_channels/groups, kH, kW]
    int64_t in_channels = shape[0];
    int64_t out_c_per_group = shape[1];
    int64_t kernel_h = shape[2];

    // Quantize weights
    auto weight_cpu = (weight.device() == Device::cpu()) ? weight : weight.to(Device::cpu());
    auto weight_observer = qconfig.create_weight_observer();
    weight_observer->observe(weight_cpu);
    auto weight_qparams = weight_observer->calculate_qparams(
        qconfig.weight_dtype(), qconfig.weight_scheme());

    // Use defaults for stride/padding/output_padding/groups
    auto result = std::make_shared<QuantizedConvTranspose2d>(
        in_channels, out_c_per_group, kernel_h,
        1, 0, 0, 1,  // stride, padding, output_padding, groups defaults
        weight_qparams);

    QuantizedTensor q_weight = quantize_tensor(weight_cpu, weight_qparams);
    result->set_weight(q_weight);

    for (auto& [name, var] : params) {
        if (name == "bias") { result->set_bias(var->tensor()); break; }
    }

    return result;
}

// ============================================================================
// QuantizedLSTMCell
// ============================================================================

QuantizedLSTMCell::QuantizedLSTMCell(
    int64_t input_size,
    int64_t hidden_size,
    bool bias,
    QuantizationParams weight_qparams
) : input_size_(input_size),
    hidden_size_(hidden_size),
    bias_(bias),
    weight_ih_(Tensor({4 * hidden_size, input_size}, DType::Int8, Device::cpu()),
               weight_qparams),
    weight_hh_(Tensor({4 * hidden_size, hidden_size}, DType::Int8, Device::cpu()),
               weight_qparams) {

    if (bias) {
        bias_ih_ = zeros({4 * hidden_size}, DType::Float32, Device::cpu());
        bias_hh_ = zeros({4 * hidden_size}, DType::Float32, Device::cpu());
    }
}

auto QuantizedLSTMCell::forward_impl(const Variable& input) -> Variable {
    // Default: zero initial state
    auto shape = input.shape();
    int64_t batch = shape[0];

    auto hx = Variable(zeros({batch, hidden_size_}, DType::Float32, input.tensor().device()), false);
    auto cx = Variable(zeros({batch, hidden_size_}, DType::Float32, input.tensor().device()), false);

    auto [h, c] = forward_cell(input, hx, cx);
    return h;
}

auto QuantizedLSTMCell::forward_cell(const Variable& input,
                                      const Variable& hx, const Variable& cx)
    -> std::pair<Variable, Variable> {
    // Dequantize weights to FP32 for gate computation
    Tensor w_ih = weight_ih_.dequantize();
    Tensor w_hh = weight_hh_.dequantize();

    auto x = input.tensor();
    auto h = hx.tensor();
    auto c = cx.tensor();

    // gates = x @ w_ih^T + h @ w_hh^T + bias
    auto gates = matmul(x, w_ih.permute({1, 0}));
    gates = gates + matmul(h, w_hh.permute({1, 0}));
    if (bias_ih_) gates = gates + *bias_ih_;
    if (bias_hh_) gates = gates + *bias_hh_;

    // Split gates: [batch, 4*hidden] -> 4x [batch, hidden]
    auto i_gate = sigmoid(gates.slice(1, 0, hidden_size_));
    auto f_gate = sigmoid(gates.slice(1, hidden_size_, 2 * hidden_size_));
    auto g_gate = tanh(gates.slice(1, 2 * hidden_size_, 3 * hidden_size_));
    auto o_gate = sigmoid(gates.slice(1, 3 * hidden_size_, 4 * hidden_size_));

    auto c_new = f_gate * c + i_gate * g_gate;
    auto h_new = o_gate * tanh(c_new);

    return {Variable(h_new, false), Variable(c_new, false)};
}

auto QuantizedLSTMCell::from_float(Module& fp_lstm_cell, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedLSTMCell> {
    auto params = fp_lstm_cell.named_parameters();

    auto get_param = [&](const std::string& name) -> Tensor {
        for (auto& [pname, var] : params) {
            if (pname == name) return var->tensor();
        }
        throw std::runtime_error("QuantizedLSTMCell::from_float: missing parameter " + name);
    };

    auto has_param = [&](const std::string& name) -> bool {
        for (auto& [pname, _] : params) { if (pname == name) return true; }
        return false;
    };

    auto w_ih = get_param("weight_ih");
    auto w_hh = get_param("weight_hh");
    int64_t hidden_size = w_ih.shape()[0] / 4;
    int64_t input_size = w_ih.shape()[1];
    bool has_bias = has_param("bias_ih");

    auto w_ih_cpu = (w_ih.device() == Device::cpu()) ? w_ih : w_ih.to(Device::cpu());
    auto w_hh_cpu = (w_hh.device() == Device::cpu()) ? w_hh : w_hh.to(Device::cpu());

    auto result = std::make_shared<QuantizedLSTMCell>(
        input_size, hidden_size, has_bias);

    result->weight_ih_ = quantize_per_tensor_symmetric(w_ih_cpu);
    result->weight_hh_ = quantize_per_tensor_symmetric(w_hh_cpu);

    if (has_bias) {
        result->bias_ih_ = get_param("bias_ih");
        result->bias_hh_ = get_param("bias_hh");
    }

    return result;
}

} // namespace quantization
} // namespace nn
} // namespace tenzor
