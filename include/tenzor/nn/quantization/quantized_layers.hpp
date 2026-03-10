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
#include <array>
#include <tuple>

namespace tenzor {
namespace nn {

// Forward declarations
class Linear;
class Conv1d;
class Conv2d;
class ConvTranspose2d;
class BatchNorm2d;

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

    auto weight() const -> const QuantizedTensor& { return weight_; }
    auto has_bias() const -> bool { return bias_.has_value(); }
    auto bias() const -> const Tensor& { return *bias_; }

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

    auto stride() const -> int64_t { return stride_; }
    auto padding() const -> int64_t { return padding_; }
    auto dilation() const -> int64_t { return dilation_; }
    auto groups() const -> int64_t { return groups_; }

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

/**
 * @brief Quantized embedding table for memory-efficient lookup.
 *
 * Stores embedding weights in INT8 or INT4 format, reducing memory usage
 * 4-8x compared to FP32. Dequantizes looked-up rows on-the-fly during forward.
 *
 * Supports per-tensor and per-channel (per-row) quantization.
 */
class QuantizedEmbedding : public Module {
public:
    QuantizedEmbedding(
        int64_t num_embeddings,
        int64_t embedding_dim,
        QuantizationParams weight_qparams,
        int64_t padding_idx = -1
    );

    auto forward_impl(const Variable& input) -> Variable override;

    auto forward_quantized(const Tensor& indices) -> Tensor;

    auto set_weight(const QuantizedTensor& weights) -> void;

    auto num_embeddings() const -> int64_t { return num_embeddings_; }
    auto embedding_dim() const -> int64_t { return embedding_dim_; }

    static auto from_float(Module& fp_embedding, const QConfig& qconfig)
        -> std::shared_ptr<QuantizedEmbedding>;

private:
    int64_t num_embeddings_;
    int64_t embedding_dim_;
    int64_t padding_idx_;
    QuantizedTensor weight_;
};

/**
 * @brief Quantized LSTM cell for efficient sequence processing.
 *
 * Uses INT8 quantized weights for input-to-hidden and hidden-to-hidden
 * transforms. Gate computations (sigmoid, tanh) run in FP32 after
 * dequantization to preserve accuracy.
 */
class QuantizedLSTM : public Module {
public:
    QuantizedLSTM(
        int64_t input_size,
        int64_t hidden_size,
        int64_t num_layers = 1,
        bool bias = true,
        bool batch_first = true,
        bool bidirectional = false,
        QuantizationParams weight_qparams = QuantizationParams(
            Tensor(), Tensor(), QuantDType::INT8, QuantizationScheme::PerTensorSymmetric)
    );

    auto forward_impl(const Variable& input) -> Variable override;

    auto forward_with_state(const Variable& input,
                            const Variable& h0, const Variable& c0)
        -> std::tuple<Variable, Variable, Variable>;

    auto input_size() const -> int64_t { return input_size_; }
    auto hidden_size() const -> int64_t { return hidden_size_; }
    auto num_layers() const -> int64_t { return num_layers_; }

    static auto from_float(Module& fp_lstm, const QConfig& qconfig)
        -> std::shared_ptr<QuantizedLSTM>;

private:
    int64_t input_size_;
    int64_t hidden_size_;
    int64_t num_layers_;
    bool bias_;
    bool batch_first_;
    bool bidirectional_;

    // Per-layer quantized weights: weight_ih, weight_hh, bias_ih, bias_hh
    struct LayerWeights {
        QuantizedTensor weight_ih;
        QuantizedTensor weight_hh;
        std::optional<Tensor> bias_ih;
        std::optional<Tensor> bias_hh;
    };
    std::vector<LayerWeights> layers_;
};

/**
 * @brief Quantized 3D convolution layer.
 *
 * INT8 3D convolution for volumetric data (video, medical imaging).
 * Same interface as QuantizedConv2d but extended to 3 spatial dimensions.
 */
class QuantizedConv3d : public Module {
public:
    QuantizedConv3d(
        int64_t in_channels,
        int64_t out_channels,
        std::array<int64_t, 3> kernel_size,
        std::array<int64_t, 3> stride = {1, 1, 1},
        std::array<int64_t, 3> padding = {0, 0, 0},
        std::array<int64_t, 3> dilation = {1, 1, 1},
        int64_t groups = 1,
        QuantizationParams weight_qparams = QuantizationParams(
            Tensor(), Tensor(), QuantDType::INT8, QuantizationScheme::PerTensorSymmetric)
    );

    auto forward_impl(const Variable& input) -> Variable override;
    auto forward_quantized(const QuantizedTensor& input) -> Tensor;

    auto set_weight(const QuantizedTensor& weights) -> void;
    auto set_bias(const Tensor& bias) -> void;

    static auto from_float(Module& fp_conv3d, const QConfig& qconfig)
        -> std::shared_ptr<QuantizedConv3d>;

private:
    int64_t in_channels_;
    int64_t out_channels_;
    std::array<int64_t, 3> kernel_size_;
    std::array<int64_t, 3> stride_;
    std::array<int64_t, 3> padding_;
    std::array<int64_t, 3> dilation_;
    int64_t groups_;
    QuantizedTensor weight_;
    std::optional<Tensor> bias_;
};

/**
 * @brief Quantized multihead attention for efficient transformer inference.
 *
 * Uses INT8 for Q/K/V linear projections. Softmax and attention score
 * computation run in FP32 for numerical stability.
 */
class QuantizedMultiheadAttention : public Module {
public:
    QuantizedMultiheadAttention(
        int64_t embed_dim,
        int64_t num_heads,
        QuantizationParams weight_qparams,
        bool bias = true,
        float dropout = 0.0f
    );

    auto forward_impl(const Variable& input) -> Variable override;

    auto forward_qkv(const Variable& query, const Variable& key,
                      const Variable& value,
                      const Variable& attn_mask = Variable{})
        -> std::pair<Variable, Variable>;

    auto embed_dim() const -> int64_t { return embed_dim_; }
    auto num_heads() const -> int64_t { return num_heads_; }

    static auto from_float(Module& fp_mha, const QConfig& qconfig)
        -> std::shared_ptr<QuantizedMultiheadAttention>;

private:
    int64_t embed_dim_;
    int64_t num_heads_;
    int64_t head_dim_;
    float dropout_;

    // Quantized projection weights
    std::shared_ptr<QuantizedLinear> q_proj_;
    std::shared_ptr<QuantizedLinear> k_proj_;
    std::shared_ptr<QuantizedLinear> v_proj_;
    std::shared_ptr<QuantizedLinear> out_proj_;
};

/**
 * @brief Quantized GRU cell for efficient sequence processing.
 *
 * Uses INT8 quantized weights for input-to-hidden and hidden-to-hidden
 * transforms. Gate computations (sigmoid, tanh) run in FP32 after
 * dequantization to preserve accuracy.
 */
class QuantizedGRU : public Module {
public:
    QuantizedGRU(
        int64_t input_size,
        int64_t hidden_size,
        int64_t num_layers = 1,
        bool bias = true,
        bool batch_first = true,
        bool bidirectional = false,
        QuantizationParams weight_qparams = QuantizationParams(
            Tensor(), Tensor(), QuantDType::INT8, QuantizationScheme::PerTensorSymmetric)
    );

    auto forward_impl(const Variable& input) -> Variable override;

    auto forward_with_state(const Variable& input, const Variable& h0)
        -> std::pair<Variable, Variable>;

    auto input_size() const -> int64_t { return input_size_; }
    auto hidden_size() const -> int64_t { return hidden_size_; }
    auto num_layers() const -> int64_t { return num_layers_; }

    static auto from_float(Module& fp_gru, const QConfig& qconfig)
        -> std::shared_ptr<QuantizedGRU>;

private:
    int64_t input_size_;
    int64_t hidden_size_;
    int64_t num_layers_;
    bool bias_;
    bool batch_first_;
    bool bidirectional_;

    struct LayerWeights {
        QuantizedTensor weight_ih;
        QuantizedTensor weight_hh;
        std::optional<Tensor> bias_ih;
        std::optional<Tensor> bias_hh;
    };
    std::vector<LayerWeights> layers_;
};

/**
 * @brief Quantized 1D convolution layer.
 *
 * INT8 1D convolution for sequence data (text, audio, time series).
 * Mirrors Conv1d but with quantized int8 weights, float32 scale/zero_point.
 * Forward dequantizes, computes, and re-quantizes (or returns float).
 */
class QuantizedConv1d : public Module {
public:
    /**
     * @brief Construct quantized conv1d layer.
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
    QuantizedConv1d(
        int64_t in_channels,
        int64_t out_channels,
        int64_t kernel_size,
        int64_t stride = 1,
        int64_t padding = 0,
        int64_t dilation = 1,
        int64_t groups = 1,
        QuantizationParams weight_qparams = QuantizationParams(
            Tensor(), Tensor(), QuantDType::INT8, QuantizationScheme::PerTensorSymmetric),
        float bias_scale = 1.0f
    );

    auto forward_impl(const Variable& input) -> Variable override;
    auto forward_quantized(const QuantizedTensor& input) -> Tensor;
    auto forward_quantized_output(const QuantizedTensor& input,
                                  const QuantizationParams& output_qparams)
        -> QuantizedTensor;

    auto set_weight(const QuantizedTensor& weights) -> void;
    auto set_bias(const Tensor& bias) -> void;

    auto stride() const -> int64_t { return stride_; }
    auto padding() const -> int64_t { return padding_; }
    auto dilation() const -> int64_t { return dilation_; }
    auto groups() const -> int64_t { return groups_; }

    /**
     * @brief Create quantized layer from floating-point Conv1d.
     *
     * @param fp_conv Floating-point Conv1d layer
     * @param qconfig Quantization configuration
     * @return Quantized conv1d layer
     */
    static auto from_float(Module& fp_conv, const QConfig& qconfig)
        -> std::shared_ptr<QuantizedConv1d>;

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
 * @brief Quantized transposed 2D convolution layer.
 *
 * INT8 transposed convolution for upsampling. Mirrors ConvTranspose2d
 * but with quantized int8 weights, float32 scale/zero_point.
 */
class QuantizedConvTranspose2d : public Module {
public:
    QuantizedConvTranspose2d(
        int64_t in_channels,
        int64_t out_channels,
        int64_t kernel_size,
        int64_t stride = 1,
        int64_t padding = 0,
        int64_t output_padding = 0,
        int64_t groups = 1,
        QuantizationParams weight_qparams = QuantizationParams(
            Tensor(), Tensor(), QuantDType::INT8, QuantizationScheme::PerTensorSymmetric),
        float bias_scale = 1.0f
    );

    auto forward_impl(const Variable& input) -> Variable override;
    auto forward_quantized(const QuantizedTensor& input) -> Tensor;

    auto set_weight(const QuantizedTensor& weights) -> void;
    auto set_bias(const Tensor& bias) -> void;

    auto stride() const -> int64_t { return stride_; }
    auto padding() const -> int64_t { return padding_; }
    auto output_padding() const -> int64_t { return output_padding_; }
    auto groups() const -> int64_t { return groups_; }

    /**
     * @brief Create quantized layer from floating-point ConvTranspose2d.
     *
     * @param fp_conv Floating-point ConvTranspose2d layer
     * @param qconfig Quantization configuration
     * @return Quantized transposed conv2d layer
     */
    static auto from_float(Module& fp_conv, const QConfig& qconfig)
        -> std::shared_ptr<QuantizedConvTranspose2d>;

private:
    int64_t in_channels_;
    int64_t out_channels_;
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    int64_t output_padding_;
    int64_t groups_;
    QuantizedTensor weight_;
    std::optional<Tensor> bias_;
    float bias_scale_;
};

/**
 * @brief Quantized LSTM cell for single-step recurrence.
 *
 * Uses INT8 quantized weights for input-to-hidden and hidden-to-hidden
 * transforms of all four gates (input, forget, cell, output).
 * Gate computations (sigmoid, tanh) run in FP32 after dequantization.
 */
class QuantizedLSTMCell : public Module {
public:
    /**
     * @brief Construct quantized LSTM cell.
     *
     * @param input_size Size of input features
     * @param hidden_size Size of hidden state
     * @param bias Whether to use bias (default: true)
     * @param weight_qparams Weight quantization parameters
     */
    QuantizedLSTMCell(
        int64_t input_size,
        int64_t hidden_size,
        bool bias = true,
        QuantizationParams weight_qparams = QuantizationParams(
            Tensor(), Tensor(), QuantDType::INT8, QuantizationScheme::PerTensorSymmetric)
    );

    /**
     * @brief Forward pass: single timestep LSTM cell.
     *
     * @param input Input variable (uses self-attention style, passes input as x with zero state)
     * @return Output variable (hidden state h)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Forward pass with explicit hidden and cell states.
     *
     * @param input Input tensor [batch, input_size]
     * @param hx Hidden state [batch, hidden_size]
     * @param cx Cell state [batch, hidden_size]
     * @return Tuple of (h', c') — new hidden and cell states
     */
    auto forward_cell(const Variable& input, const Variable& hx, const Variable& cx)
        -> std::pair<Variable, Variable>;

    auto input_size() const -> int64_t { return input_size_; }
    auto hidden_size() const -> int64_t { return hidden_size_; }

    /**
     * @brief Create quantized LSTM cell from floating-point version.
     */
    static auto from_float(Module& fp_lstm_cell, const QConfig& qconfig)
        -> std::shared_ptr<QuantizedLSTMCell>;

private:
    int64_t input_size_;
    int64_t hidden_size_;
    bool bias_;

    QuantizedTensor weight_ih_;   ///< [4*hidden_size, input_size]
    QuantizedTensor weight_hh_;   ///< [4*hidden_size, hidden_size]
    std::optional<Tensor> bias_ih_;
    std::optional<Tensor> bias_hh_;
};

} // namespace quantization
} // namespace nn
} // namespace tenzor
