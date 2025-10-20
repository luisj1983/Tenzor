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
    auto forward(const Variable& input) -> Variable override;

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

    auto forward(const Variable& input) -> Variable override;
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

    auto forward(const Variable& input) -> Variable override;
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

    auto forward(const Variable& input) -> Variable override;
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
 * @brief Quantization stub for model input.
 *
 * Quantizes floating-point input at the model entrance.
 * Should be the first layer in a quantized model.
 */
class QuantizationStub : public Module {
public:
    /**
     * @brief Construct quantization stub.
     *
     * @param qparams Input quantization parameters
     */
    explicit QuantizationStub(QuantizationParams qparams);

    /**
     * @brief Forward pass: quantize input.
     *
     * @param input Floating-point input
     * @return Quantized output (as QuantizedTensor wrapped in Variable)
     */
    auto forward(const Variable& input) -> Variable override;

    /**
     * @brief Get quantized output directly.
     */
    auto forward_to_quantized(const Tensor& input) -> QuantizedTensor;

private:
    QuantizationParams qparams_;
};

/**
 * @brief Dequantization stub for model output.
 *
 * Dequantizes quantized tensor back to floating-point at model exit.
 * Should be the last layer in a quantized model.
 */
class DequantizationStub : public Module {
public:
    DequantizationStub() = default;

    /**
     * @brief Forward pass: dequantize input.
     *
     * @param input Quantized input (wrapped as Variable)
     * @return Floating-point output
     */
    auto forward(const Variable& input) -> Variable override;

    /**
     * @brief Dequantize directly from QuantizedTensor.
     */
    auto forward_from_quantized(const QuantizedTensor& input) -> Tensor;
};

} // namespace quantization
} // namespace nn
} // namespace tenzor
