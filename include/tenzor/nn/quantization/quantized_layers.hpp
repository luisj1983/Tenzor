/**
 * @file quantized_layers.hpp
 * @brief Quantized neural network layers for efficient inference
 *
 * Provides quantized variants of standard layers (Linear, Conv2d, BatchNorm2d)
 * that operate directly on quantized integer tensors for improved performance
 * and reduced memory usage.
 */

#pragma once

#include "../module.hpp"
#include "quantize.hpp"
#include "qconfig.hpp"

namespace tenzor {
namespace nn {

// Forward declarations
class Linear;
class Conv2d;

namespace quantization {

/**
 * @brief Quantized linear (fully connected) layer.
 *
 * Operates on quantized INT8 inputs and weights, performing integer matrix
 * multiplication followed by dequantization. Much faster than floating-point
 * on CPUs and uses 4x less memory.
 *
 * Forward computation:
 * 1. INT8 matrix multiplication: Y_q = X_q @ W_q^T
 * 2. Scale adjustment and bias addition
 * 3. Optional output quantization or dequantization
 *
 * @code
 * // Convert floating-point Linear layer to quantized
 * Linear fp_linear(128, 64);
 * auto q_linear = QuantizedLinear::from_float(fp_linear, qconfig);
 *
 * // Forward with quantized input
 * QuantizedTensor q_input = quantize_per_tensor_symmetric(input);
 * Tensor output = q_linear->forward_quantized(q_input);
 * @endcode
 */
class QuantizedLinear : public Module {
public:
    /**
     * @brief Construct quantized linear layer.
     *
     * @param in_features Input feature dimension
     * @param out_features Output feature dimension
     * @param weight_qparams Weight quantization parameters
     * @param bias_scale Bias scale factor (default: 1.0)
     */
    QuantizedLinear(
        int64_t in_features,
        int64_t out_features,
        QuantizationParams weight_qparams,
        float bias_scale = 1.0f
    );

    /**
     * @brief Forward pass (dequantizes output).
     *
     * @param input Floating-point input variable
     * @return Floating-point output variable
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Forward pass with quantized input.
     *
     * @param input Quantized input tensor
     * @return Floating-point output tensor
     */
    auto forward_quantized(const QuantizedTensor& input) -> Tensor;

    /**
     * @brief Forward pass returning quantized output.
     *
     * @param input Quantized input tensor
     * @param output_qparams Output quantization parameters
     * @return Quantized output tensor
     */
    auto forward_quantized_output(const QuantizedTensor& input,
                                  const QuantizationParams& output_qparams)
        -> QuantizedTensor;

    /**
     * @brief Set quantized weights.
     *
     * @param weights Quantized weight tensor
     */
    auto set_weight(const QuantizedTensor& weights) -> void;

    /**
     * @brief Set bias.
     *
     * @param bias Bias tensor (floating-point or quantized)
     */
    auto set_bias(const Tensor& bias) -> void;

    /**
     * @brief Create quantized layer from floating-point layer.
     *
     * @param fp_linear Floating-point linear layer
     * @param qconfig Quantization configuration
     * @return Quantized linear layer
     */
    static auto from_float(const Linear& fp_linear, const QConfig& qconfig)
        -> std::shared_ptr<QuantizedLinear>;

private:
    int64_t in_features_;
    int64_t out_features_;
    QuantizedTensor weight_;       ///< Quantized weights
    std::optional<Tensor> bias_;   ///< Floating-point bias
    float bias_scale_;             ///< Bias scale factor
};

/**
 * @brief Quantized 2D convolution layer.
 *
 * Performs quantized convolution using INT8 operations. Significantly faster
 * than floating-point convolution, especially on CPUs with VNNI or similar
 * integer instructions.
 *
 * @code
 * Conv2d fp_conv(3, 64, 3, 1, 1);
 * auto q_conv = QuantizedConv2d::from_float(fp_conv, qconfig);
 *
 * QuantizedTensor q_input = quantize_per_tensor_symmetric(input);
 * Tensor output = q_conv->forward_quantized(q_input);
 * @endcode
 */
class QuantizedConv2d : public Module {
public:
    /**
     * @brief Construct quantized conv2d layer.
     *
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param kernel_size Kernel size
     * @param stride Stride
     * @param padding Padding
     * @param dilation Dilation
     * @param groups Number of groups
     * @param weight_qparams Weight quantization parameters
     * @param bias_scale Bias scale factor
     */
    QuantizedConv2d(
        int64_t in_channels,
        int64_t out_channels,
        int64_t kernel_size,
        int64_t stride,
        int64_t padding,
        int64_t dilation,
        int64_t groups,
        QuantizationParams weight_qparams,
        float bias_scale = 1.0f
    );

    auto forward_impl(const Variable& input) -> Variable override;
    auto forward_quantized(const QuantizedTensor& input) -> Tensor;
    auto forward_quantized_output(const QuantizedTensor& input,
                                  const QuantizationParams& output_qparams)
        -> QuantizedTensor;

    auto set_weight(const QuantizedTensor& weights) -> void;
    auto set_bias(const Tensor& bias) -> void;

    static auto from_float(const Conv2d& fp_conv, const QConfig& qconfig)
        -> std::shared_ptr<QuantizedConv2d>;

private:
    int64_t in_channels_;
    int64_t out_channels_;
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
    int64_t groups_;
    QuantizedTensor weight_;
    std::optional<Tensor> bias_;
    float bias_scale_;
};

/**
 * @brief Quantized batch normalization layer.
 *
 * Folds batch normalization parameters into scale/shift for efficient
 * quantized inference. During inference, applies:
 * y = scale * x + bias
 *
 * where scale and bias incorporate the batch norm statistics.
 */
class QuantizedBatchNorm2d : public Module {
public:
    /**
     * @brief Construct quantized batch norm layer.
     *
     * @param num_features Number of channels
     * @param scale Scale tensor (folded from gamma/running_var)
     * @param bias Bias tensor (folded from beta/running_mean)
     */
    QuantizedBatchNorm2d(
        int64_t num_features,
        Tensor scale,
        Tensor bias
    );

    auto forward_impl(const Variable& input) -> Variable override;
    auto forward_quantized(const QuantizedTensor& input) -> QuantizedTensor;

    /**
     * @brief Create quantized batch norm from floating-point version.
     *
     * Folds running statistics into scale/bias for efficient inference.
     *
     * @param fp_bn Floating-point batch norm layer
     * @param qconfig Quantization configuration
     * @return Quantized batch norm layer
     */
    static auto from_float(const Module& fp_bn, const QConfig& qconfig)
        -> std::shared_ptr<QuantizedBatchNorm2d>;

    /**
     * @brief Fold batch norm into preceding conv/linear layer.
     *
     * Folds batch norm scale/shift into the weights and bias of the
     * previous layer for further optimization.
     *
     * @param prev_layer Previous quantized layer (Conv2d or Linear)
     * @return Fused layer with batch norm folded in
     */
    static auto fold_into_layer(const Module& prev_layer,
                                const QuantizedBatchNorm2d& bn)
        -> std::shared_ptr<Module>;

private:
    int64_t num_features_;
    Tensor scale_;  ///< Folded scale (gamma / sqrt(var + eps))
    Tensor bias_;   ///< Folded bias (beta - gamma * mean / sqrt(var + eps))
};

/**
 * @brief Quantized Layer Normalization layer.
 *
 * Applies layer normalization with per-channel scale and bias for INT8 inference.
 * Created from a floating-point LayerNorm via from_float().
 */
class QuantizedLayerNorm : public Module {
public:
    /**
     * @brief Construct quantized layer norm.
     *
     * @param normalized_shape Shape of the normalized dimensions
     * @param weight Scale parameter (gamma)
     * @param bias Bias parameter (beta)
     * @param eps Epsilon for numerical stability
     */
    QuantizedLayerNorm(
        std::vector<int64_t> normalized_shape,
        Tensor weight,
        Tensor bias,
        double eps = 1e-5
    );

    auto forward_impl(const Variable& input) -> Variable override;
    auto forward_quantized(const QuantizedTensor& input) -> QuantizedTensor;

    /**
     * @brief Create quantized layer norm from floating-point version.
     *
     * @param fp_ln Floating-point LayerNorm module
     * @param qconfig Quantization configuration
     * @return Quantized layer norm layer
     */
    static auto from_float(const Module& fp_ln, const QConfig& qconfig)
        -> std::shared_ptr<QuantizedLayerNorm>;

private:
    std::vector<int64_t> normalized_shape_;
    Tensor weight_;
    Tensor bias_;
    double eps_;
};

/**
 * @brief Fused quantized Conv2d + ReLU layer.
 *
 * Combines convolution and ReLU activation in a single quantized operation
 * for improved performance. Common pattern in CNNs.
 */
class QuantizedConv2dReLU : public QuantizedConv2d {
public:
    using QuantizedConv2d::QuantizedConv2d;

    auto forward_quantized(const QuantizedTensor& input) -> Tensor;
    auto forward_quantized_output(const QuantizedTensor& input,
                                  const QuantizationParams& output_qparams)
        -> QuantizedTensor;

    static auto from_float(const Conv2d& fp_conv, const QConfig& qconfig)
        -> std::shared_ptr<QuantizedConv2dReLU>;
};

/**
 * @brief Fused quantized Conv2d + BatchNorm2d + ReLU layer.
 *
 * Fuses convolution, batch normalization, and ReLU into a single
 * quantized operation. Maximizes performance for common CNN patterns.
 */
class QuantizedConv2dBnReLU : public Module {
public:
    /**
     * @brief Construct fused quantized conv-bn-relu layer.
     */
    QuantizedConv2dBnReLU(
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
    );

    auto forward_impl(const Variable& input) -> Variable override;
    auto forward_quantized(const QuantizedTensor& input) -> Tensor;

    /**
     * @brief Create fused layer from separate modules.
     *
     * @param fp_conv Floating-point convolution
     * @param fp_bn Floating-point batch norm
     * @param qconfig Quantization configuration
     * @return Fused quantized layer
     */
    static auto from_float(const Conv2d& fp_conv,
                          const Module& fp_bn,
                          const QConfig& qconfig)
        -> std::shared_ptr<QuantizedConv2dBnReLU>;

private:
    std::shared_ptr<QuantizedConv2d> conv_;
    Tensor bn_scale_;
    Tensor bn_bias_;
};

/**
 * @brief Quantization layer for model input.
 *
 * Quantizes floating-point input tensors to INT8/UINT8 at model entrance.
 * Converts FP32 values to quantized integers using:
 *   q = clamp(round(x / scale) + zero_point, qmin, qmax)
 *
 * Supports both symmetric and asymmetric quantization:
 * - Symmetric (zero_point = 0): Uses INT8 range [-128, 127]
 * - Asymmetric (zero_point != 0): Uses full INT8 or UINT8 range
 *
 * Should be the first layer in a quantized model to prepare inputs
 * for quantized operations downstream.
 *
 * @code
 * // Symmetric INT8 quantization
 * auto qparams = compute_quantization_params(
 *     min_val, max_val,
 *     QuantDType::INT8,
 *     QuantizationScheme::PerTensorSymmetric
 * );
 * auto quant_stub = std::make_shared<QuantStub>(qparams);
 *
 * // Quantize input
 * Tensor fp_input({1, 3, 224, 224}, DType::Float32);
 * QuantizedTensor q_input = quant_stub->forward_to_quantized(fp_input);
 * @endcode
 */
class QuantStub : public Module {
public:
    /**
     * @brief Construct quantization layer.
     *
     * @param qparams Quantization parameters (scale, zero_point, dtype, scheme)
     */
    explicit QuantStub(QuantizationParams qparams);

    /**
     * @brief Forward pass: quantize floating-point input.
     *
     * Converts FP32 input to quantized representation. The output Variable
     * contains metadata about quantization parameters for downstream layers.
     *
     * @param input Floating-point input variable
     * @return Variable containing quantized tensor data
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Quantize tensor directly.
     *
     * Performs quantization without Variable wrapper:
     * 1. Scale input: x_scaled = x / scale
     * 2. Add zero point: x_shifted = x_scaled + zero_point
     * 3. Round to nearest integer: x_rounded = round(x_shifted)
     * 4. Clamp to valid range: q = clamp(x_rounded, qmin, qmax)
     *
     * @param input Floating-point tensor
     * @return QuantizedTensor with INT8/UINT8 data and parameters
     */
    auto forward_to_quantized(const Tensor& input) -> QuantizedTensor;

    /**
     * @brief Get quantization parameters.
     */
    auto qparams() const -> const QuantizationParams& { return qparams_; }

    /**
     * @brief Update quantization parameters.
     *
     * Useful for calibration or adapting to new data distributions.
     *
     * @param qparams New quantization parameters
     */
    auto set_qparams(QuantizationParams qparams) -> void {
        qparams_ = std::move(qparams);
    }

    /**
     * @brief Check if using symmetric quantization.
     */
    auto is_symmetric() const -> bool {
        return qparams_.scheme == QuantizationScheme::PerTensorSymmetric ||
               qparams_.scheme == QuantizationScheme::PerChannelSymmetric;
    }

    /**
     * @brief Check if using per-channel quantization.
     */
    auto is_per_channel() const -> bool {
        return qparams_.scheme == QuantizationScheme::PerChannelSymmetric ||
               qparams_.scheme == QuantizationScheme::PerChannelAsymmetric;
    }

private:
    QuantizationParams qparams_;  ///< Quantization parameters (scale, zero_point)
};

/**
 * @brief Dequantization layer for model output.
 *
 * Dequantizes INT8/UINT8 tensors back to floating-point at model exit.
 * Converts quantized integers back to FP32 values using:
 *   x = (q - zero_point) * scale
 *
 * This is the inverse operation of QuantStub. Should be the last layer
 * in a quantized model to convert outputs back to floating-point for
 * compatibility with loss functions and metrics.
 *
 * The dequantization process:
 * 1. Subtract zero point: x_shifted = q - zero_point
 * 2. Scale to float range: x = x_shifted * scale
 *
 * @code
 * auto dequant_stub = std::make_shared<DeQuantStub>();
 *
 * // Dequantize output from quantized model
 * QuantizedTensor q_output = quantized_model->forward_quantized(q_input);
 * Tensor fp_output = dequant_stub->forward_from_quantized(q_output);
 * @endcode
 */
class DeQuantStub : public Module {
public:
    DeQuantStub() = default;

    /**
     * @brief Forward pass: dequantize input.
     *
     * Takes a Variable that should contain quantized tensor data and
     * dequantizes it back to floating-point. The input Variable must
     * have quantization metadata attached.
     *
     * @param input Variable containing quantized tensor
     * @return Floating-point output variable
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Dequantize QuantizedTensor directly.
     *
     * Performs dequantization without Variable wrapper:
     * 1. Cast INT8/UINT8 to float: x_float = float(q)
     * 2. Subtract zero point: x_shifted = x_float - zero_point
     * 3. Scale to original range: x = x_shifted * scale
     *
     * Handles both per-tensor and per-channel quantization:
     * - Per-tensor: Single scale and zero_point for entire tensor
     * - Per-channel: Different scale/zero_point per channel
     *
     * @param input QuantizedTensor with quantization parameters
     * @return Dequantized floating-point tensor
     */
    auto forward_from_quantized(const QuantizedTensor& input) -> Tensor;

    /**
     * @brief Check if last dequantization was per-channel.
     *
     * Useful for debugging and validation.
     */
    auto last_was_per_channel() const -> bool { return last_per_channel_; }

private:
    bool last_per_channel_{false};  ///< Track if last operation was per-channel
};

} // namespace quantization
} // namespace nn
} // namespace tenzor
