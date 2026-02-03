/**
 * @file fused_ops.hpp
 * @brief Fused kernel operations for reduced kernel launch overhead
 *
 * Provides optimized fused operations that combine multiple elementwise
 * operations into single kernels, reducing memory bandwidth and kernel
 * launch overhead. Targets 1.5-3x speedup over unfused operations.
 *
 * Fusion patterns implemented:
 * - Linear + ReLU: Fused fully connected layer with activation
 * - Conv2D + ReLU: Fused convolution with activation
 * - BatchNorm + ReLU: Fused normalization with activation
 * - Softmax + CrossEntropy: Fused activation and loss computation
 */

#pragma once

#include "../core/tensor.hpp"

namespace tenzor {
namespace ops {

/**
 * @defgroup fused_ops Fused Operations
 * @brief Kernel fusion optimizations for improved performance
 * @{
 */

/**
 * @brief Fused linear transformation with ReLU activation.
 *
 * Combines matrix multiplication, bias addition, and ReLU activation
 * into a single operation. Avoids intermediate memory allocations.
 *
 * Formula: relu(input @ weight.T + bias)
 *
 * @param input Input tensor (N, in_features) or (*, in_features)
 * @param weight Weight matrix (out_features, in_features)
 * @param bias Bias vector (out_features) or nullptr for no bias
 * @return Output tensor (N, out_features) or (*, out_features)
 *
 * Performance: 1.5-2x faster than unfused linear + relu
 *
 * @code
 * auto x = randn({32, 128});
 * auto weight = randn({64, 128});
 * auto bias = randn({64});
 * auto y = fused_linear_relu(x, weight, bias);  // Shape: {32, 64}
 * @endcode
 */
auto fused_linear_relu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias = nullptr
) -> Tensor;

/**
 * @brief Fused 2D convolution with ReLU activation.
 *
 * Combines convolution and ReLU activation into single operation.
 * Significantly reduces memory bandwidth requirements.
 *
 * Formula: relu(conv2d(input, weight, bias, stride, padding))
 *
 * @param input Input tensor (N, C_in, H, W)
 * @param weight Convolution kernel (C_out, C_in, KH, KW)
 * @param bias Bias vector (C_out) or nullptr
 * @param stride Stride for convolution (default: 1)
 * @param padding Padding size (default: 0)
 * @return Output tensor (N, C_out, H_out, W_out)
 *
 * Performance: 1.8-2.5x faster than unfused conv2d + relu
 *
 * @code
 * auto x = randn({8, 3, 32, 32});
 * auto weight = randn({16, 3, 3, 3});
 * auto bias = randn({16});
 * auto y = fused_conv2d_relu(x, weight, &bias, 1, 1);
 * @endcode
 */
auto fused_conv2d_relu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias = nullptr,
    int64_t stride = 1,
    int64_t padding = 0
) -> Tensor;

/**
 * @brief Fused convolution with sigmoid activation.
 *
 * Combines Conv2D + bias + sigmoid in a single cuDNN call.
 * Commonly used in attention gates and output layers for binary classification.
 *
 * @param input Input tensor (N, C_in, H, W)
 * @param weight Convolution weights (C_out, C_in/groups, KH, KW)
 * @param bias Optional bias tensor (C_out), nullptr if no bias
 * @param stride Convolution stride (default: 1)
 * @param padding Convolution padding (default: 0)
 * @return Output tensor (N, C_out, H_out, W_out)
 */
auto fused_conv2d_sigmoid(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias = nullptr,
    int64_t stride = 1,
    int64_t padding = 0
) -> Tensor;

/**
 * @brief Fused convolution with tanh activation.
 *
 * Combines Conv2D + bias + tanh in a single cuDNN call.
 * Commonly used in LSTM/GRU gates and generative models.
 *
 * @param input Input tensor (N, C_in, H, W)
 * @param weight Convolution weights (C_out, C_in/groups, KH, KW)
 * @param bias Optional bias tensor (C_out), nullptr if no bias
 * @param stride Convolution stride (default: 1)
 * @param padding Convolution padding (default: 0)
 * @return Output tensor (N, C_out, H_out, W_out)
 */
auto fused_conv2d_tanh(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias = nullptr,
    int64_t stride = 1,
    int64_t padding = 0
) -> Tensor;

/**
 * @brief Fused convolution with Swish (SiLU) activation.
 *
 * Combines Conv2D + bias + swish in a single cuDNN call.
 * Swish(x) = x * sigmoid(x). Used in EfficientNet, MobileNetV3.
 *
 * @param input Input tensor (N, C_in, H, W)
 * @param weight Convolution weights (C_out, C_in/groups, KH, KW)
 * @param bias Optional bias tensor (C_out), nullptr if no bias
 * @param stride Convolution stride (default: 1)
 * @param padding Convolution padding (default: 0)
 * @return Output tensor (N, C_out, H_out, W_out)
 */
auto fused_conv2d_swish(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias = nullptr,
    int64_t stride = 1,
    int64_t padding = 0
) -> Tensor;

/**
 * @brief Fused batch normalization with ReLU activation.
 *
 * Combines batch normalization and ReLU into single kernel.
 * Particularly effective for training with frequent normalizations.
 *
 * Formula: relu((input - mean) / sqrt(var + eps) * weight + bias)
 *
 * @param input Input tensor (N, C, H, W) or (N, C)
 * @param running_mean Running mean (C)
 * @param running_var Running variance (C)
 * @param weight Scale parameter gamma (C)
 * @param bias Shift parameter beta (C)
 * @param eps Small constant for numerical stability (default: 1e-5)
 * @return Normalized and activated output tensor
 *
 * Performance: 1.6-2.2x faster than unfused batchnorm + relu
 *
 * @code
 * auto x = randn({32, 64, 16, 16});
 * auto mean = zeros({64});
 * auto var = ones({64});
 * auto gamma = ones({64});
 * auto beta = zeros({64});
 * auto y = fused_batchnorm_relu(x, mean, var, gamma, beta);
 * @endcode
 */
auto fused_batchnorm_relu(
    const Tensor& input,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& weight,
    const Tensor& bias,
    float eps = 1e-5f
) -> Tensor;

/**
 * @brief Fused softmax with cross-entropy loss.
 *
 * Numerically stable implementation combining softmax activation
 * and cross-entropy loss computation. Avoids materializing softmax
 * probabilities, saving memory and improving numerical stability.
 *
 * Formula: -log(exp(logits[target]) / sum(exp(logits)))
 *
 * Uses log-sum-exp trick for numerical stability:
 * log_softmax(x) = x - log(sum(exp(x - max(x)))) - max(x)
 *
 * @param logits Input logits (N, num_classes)
 * @param targets Target class indices (N,) with values in [0, num_classes)
 * @param reduction Reduction mode: "mean", "sum", or "none"
 * @return Scalar loss (mean/sum) or per-sample losses (none)
 *
 * Performance: 2-3x faster than unfused softmax + cross_entropy
 * Memory: 50% less memory usage (no intermediate probabilities)
 *
 * @code
 * auto logits = randn({32, 10});
 * auto targets = randint(0, 10, {32});
 * auto loss = fused_softmax_cross_entropy(logits, targets, "mean");
 * @endcode
 *
 * @see LogSoftmax for gradient-friendly alternative
 */
auto fused_softmax_cross_entropy(
    const Tensor& logits,
    const Tensor& targets,
    const std::string& reduction = "mean"
) -> Tensor;

/**
 * @brief Fused add with ReLU activation (for residual connections).
 *
 * Combines element-wise addition and ReLU activation.
 * Common pattern in residual networks.
 *
 * Formula: relu(a + b)
 *
 * @param a First input tensor
 * @param b Second input tensor (must be broadcastable with a)
 * @return Output tensor with same shape as result of a + b
 *
 * Performance: 1.3-1.8x faster than unfused add + relu
 *
 * @code
 * auto x = randn({32, 64});
 * auto residual = randn({32, 64});
 * auto y = fused_add_relu(x, residual);
 * @endcode
 */
auto fused_add_relu(const Tensor& a, const Tensor& b) -> Tensor;

/**
 * @brief Fused GELU activation (Gaussian Error Linear Unit).
 *
 * Optimized implementation of GELU activation function used in
 * transformer models. Uses tanh approximation for efficiency.
 *
 * Formula (approximation):
 * gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * @param input Input tensor
 * @return Output tensor with GELU activation applied
 *
 * Performance: 1.5x faster than separate operations
 *
 * @code
 * auto x = randn({32, 512});
 * auto y = fused_gelu(x);
 * @endcode
 */
auto fused_gelu(const Tensor& input) -> Tensor;

/**
 * @brief Fused layer normalization (mean, variance, normalize).
 *
 * Single-pass layer normalization avoiding multiple memory passes.
 * Computes mean and variance in single pass using Welford's algorithm.
 *
 * Formula: (input - mean) / sqrt(var + eps) * weight + bias
 *
 * @param input Input tensor (*, normalized_shape)
 * @param normalized_shape Shape of features to normalize
 * @param weight Scale parameter (normalized_shape)
 * @param bias Shift parameter (normalized_shape)
 * @param eps Small constant for numerical stability (default: 1e-5)
 * @return Normalized output tensor
 *
 * Performance: 1.4-2x faster than unfused operations
 *
 * @code
 * auto x = randn({32, 512});
 * auto weight = ones({512});
 * auto bias = zeros({512});
 * auto y = fused_layer_norm(x, {512}, weight, bias);
 * @endcode
 */
auto fused_layer_norm(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    float eps = 1e-5f
) -> Tensor;

/** @} */ // end of fused_ops group

} // namespace ops
} // namespace tenzor
