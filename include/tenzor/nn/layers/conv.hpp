/**
 * @file conv.hpp
 * @brief Convolutional neural network layers (1D, 2D, transposed)
 *
 * Implements standard and transposed convolution operations for
 * spatial feature extraction in neural networks.
 */

#pragma once

#include <optional>
#include <utility>
#include "../module.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief 2D convolutional layer.
 *
 * Applies 2D convolution over an input signal composed of multiple planes.
 * Commonly used in computer vision for spatial feature extraction.
 *
 * Shape transformations:
 * - Input: (N, C_in, H_in, W_in)
 * - Output: (N, C_out, H_out, W_out)
 * - Weight: (C_out, C_in/groups, K, K)
 * - Bias: (C_out) if enabled
 *
 * Output dimensions:
 * - H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
 * - W_out = floor((W_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
 *
 * Parameters are initialized using Kaiming uniform initialization suitable
 * for convolutional layers.
 *
 * @code
 * // 3-channel input -> 64-channel output, 3x3 kernel
 * Conv2d conv(3, 64, 3, 1, 1);  // stride=1, padding=1
 *
 * // Forward pass
 * Variable x(Tensor({batch, 3, 32, 32}, DType::Float32, Device::cpu()), true);
 * Variable features = conv.forward(x);  // Shape: {batch, 64, 32, 32}
 * @endcode
 *
 * @see Module for base class interface
 * @see Conv1d for 1D convolution
 * @see ConvTranspose2d for upsampling
 */
class Conv2d : public Module {
public:
    /**
     * @brief Construct 2D convolutional layer with square kernel.
     *
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels (filters)
     * @param kernel_size Size of convolving kernel (square)
     * @param stride Stride of convolution (default: 1)
     * @param padding Zero-padding added to both sides (default: 0)
     * @param dilation Spacing between kernel elements (default: 1)
     * @param groups Number of blocked connections (default: 1)
     * @param bias If true, add learnable bias (default: true)
     *
     * @note groups=1 is standard convolution, groups=in_channels is depthwise
     *
     * @code
     * Conv2d conv1(3, 64, 3, 1, 1);  // Standard 3x3 conv
     * Conv2d conv2(64, 64, 3, 2, 1);  // Stride 2 for downsampling
     * Conv2d dw_conv(64, 64, 3, 1, 1, 1, 64);  // Depthwise
     * @endcode
     */
    Conv2d(int64_t in_channels,
           int64_t out_channels,
           int64_t kernel_size,
           int64_t stride = 1,
           int64_t padding = 0,
           int64_t dilation = 1,
           int64_t groups = 1,
           bool bias = true);

    /**
     * @brief Construct 2D convolutional layer with non-square kernel.
     *
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels (filters)
     * @param kernel_size Pair of (height, width) kernel dimensions
     * @param stride Pair of (height, width) strides (default: {1,1})
     * @param padding Pair of (height, width) padding (default: {0,0})
     * @param dilation Pair of (height, width) dilation (default: {1,1})
     * @param groups Number of blocked connections (default: 1)
     * @param bias If true, add learnable bias (default: true)
     *
     * @code
     * Conv2d conv(3, 64, {3, 5});             // 3x5 kernel
     * Conv2d conv(3, 64, {3, 5}, {1, 2});     // Non-square stride
     * Conv2d conv(3, 64, {3, 5}, {1, 1}, {1, 2}); // Non-square padding
     * @endcode
     */
    Conv2d(int64_t in_channels,
           int64_t out_channels,
           std::pair<int64_t, int64_t> kernel_size,
           std::pair<int64_t, int64_t> stride = {1, 1},
           std::pair<int64_t, int64_t> padding = {0, 0},
           std::pair<int64_t, int64_t> dilation = {1, 1},
           int64_t groups = 1,
           bool bias = true);

    /**
     * @brief Forward pass through 2D convolution.
     *
     * @param input Input variable of shape (N, C_in, H, W)
     * @return Output variable of shape (N, C_out, H_out, W_out)
     *
     * @throws std::runtime_error if input channels don't match
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t in_channels_;   ///< Number of input channels
    int64_t out_channels_;  ///< Number of output channels
    int64_t kernel_h_;      ///< Kernel height
    int64_t kernel_w_;      ///< Kernel width
    int64_t stride_h_;      ///< Stride height
    int64_t stride_w_;      ///< Stride width
    int64_t padding_h_;     ///< Padding height
    int64_t padding_w_;     ///< Padding width
    int64_t dilation_h_;    ///< Dilation height
    int64_t dilation_w_;    ///< Dilation width
    int64_t groups_;        ///< Number of groups

    /**
     * @brief Initialize parameters using Kaiming uniform.
     */
    auto reset_parameters() -> void;
};

/**
 * @brief 1D convolutional layer.
 *
 * Applies 1D convolution over an input signal.
 * Commonly used for sequence data, time series, and audio processing.
 *
 * Shape transformations:
 * - Input: (N, C_in, L_in)
 * - Output: (N, C_out, L_out)
 * - Weight: (C_out, C_in/groups, K)
 *
 * Output length:
 * - L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
 *
 * @code
 * // 128-dim embeddings -> 256 features, kernel size 3
 * Conv1d conv(128, 256, 3, 1, 1);
 *
 * Variable seq(Tensor({batch, 128, seq_len}, DType::Float32, Device::cpu()), true);
 * Variable features = conv.forward(seq);  // Shape: {batch, 256, seq_len}
 * @endcode
 *
 * @see Conv2d for 2D convolution
 */
class Conv1d : public Module {
public:
    /**
     * @brief Construct 1D convolutional layer.
     *
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param kernel_size Size of convolving kernel
     * @param stride Stride of convolution (default: 1)
     * @param padding Zero-padding added to both sides (default: 0)
     * @param dilation Spacing between kernel elements (default: 1)
     * @param groups Number of blocked connections (default: 1)
     * @param bias If true, add learnable bias (default: true)
     */
    Conv1d(int64_t in_channels,
           int64_t out_channels,
           int64_t kernel_size,
           int64_t stride = 1,
           int64_t padding = 0,
           int64_t dilation = 1,
           int64_t groups = 1,
           bool bias = true);

    /**
     * @brief Forward pass through 1D convolution.
     *
     * @param input Input variable of shape (N, C_in, L)
     * @return Output variable of shape (N, C_out, L_out)
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t in_channels_;   ///< Number of input channels
    int64_t out_channels_;  ///< Number of output channels
    int64_t kernel_size_;   ///< Kernel size
    int64_t stride_;        ///< Stride
    int64_t padding_;       ///< Padding
    int64_t dilation_;      ///< Dilation
    int64_t groups_;        ///< Number of groups

    /**
     * @brief Initialize parameters using Kaiming uniform.
     */
    auto reset_parameters() -> void;
};

/**
 * @brief Transposed 2D convolution (deconvolution) layer.
 *
 * Applies a 2D transposed convolution for upsampling spatial dimensions.
 * Often used in decoders, generators, and semantic segmentation networks.
 *
 * Unlike regular convolution which reduces spatial dimensions, transposed
 * convolution increases them, making it useful for upsampling.
 *
 * Shape transformations:
 * - Input: (N, C_in, H_in, W_in)
 * - Output: (N, C_out, H_out, W_out)
 *
 * Output dimensions:
 * - H_out = (H_in - 1) * stride - 2*padding + kernel_size + output_padding
 * - W_out = (W_in - 1) * stride - 2*padding + kernel_size + output_padding
 *
 * @code
 * // Upsample from 64 to 32 channels, 2x spatial size
 * ConvTranspose2d upconv(64, 32, 4, 2, 1);  // stride=2 doubles size
 *
 * Variable x(Tensor({batch, 64, 16, 16}, DType::Float32, Device::cpu()), true);
 * Variable upsampled = upconv.forward(x);  // Shape: {batch, 32, 32, 32}
 * @endcode
 *
 * @see Conv2d for standard convolution
 */
class ConvTranspose2d : public Module {
public:
    /**
     * @brief Construct transposed 2D convolutional layer.
     *
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param kernel_size Size of convolving kernel (square)
     * @param stride Stride of convolution (default: 1)
     * @param padding Zero-padding added to input (default: 0)
     * @param output_padding Additional size added to output (default: 0)
     * @param groups Number of blocked connections (default: 1)
     * @param bias If true, add learnable bias (default: true)
     *
     * @note output_padding is used to resolve ambiguity when multiple input
     *       sizes map to the same output size.
     *
     * @code
     * ConvTranspose2d up1(128, 64, 4, 2, 1);  // 2x upsampling
     * ConvTranspose2d up2(64, 32, 4, 2, 1);   // Another 2x upsampling
     * @endcode
     */
    ConvTranspose2d(int64_t in_channels,
                    int64_t out_channels,
                    int64_t kernel_size,
                    int64_t stride = 1,
                    int64_t padding = 0,
                    int64_t output_padding = 0,
                    int64_t groups = 1,
                    bool bias = true);

    /**
     * @brief Forward pass through transposed 2D convolution.
     *
     * @param input Input variable of shape (N, C_in, H_in, W_in)
     * @return Output variable of shape (N, C_out, H_out, W_out)
     *
     * @throws std::runtime_error if input channels don't match
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t in_channels_;       ///< Number of input channels
    int64_t out_channels_;      ///< Number of output channels
    int64_t kernel_size_;       ///< Kernel size
    int64_t stride_;            ///< Stride
    int64_t padding_;           ///< Padding
    int64_t output_padding_;    ///< Output padding
    int64_t groups_;            ///< Number of groups

    // Parameters accessed via parameters_ map — no member variable duplicates.
    // Use parameters_["weight"] and parameters_.find("bias") in forward().

    /**
     * @brief Initialize parameters using Kaiming uniform.
     */
    auto reset_parameters() -> void;
};

/**
 * @brief 3D convolutional layer.
 *
 * Applies 3D convolution over volumetric input (video, medical imaging).
 *
 * Shape transformations:
 * - Input: (N, C_in, D_in, H_in, W_in)
 * - Output: (N, C_out, D_out, H_out, W_out)
 * - Weight: (C_out, C_in/groups, K, K, K)
 *
 * @code
 * Conv3d conv(1, 32, 3, 1, 1);
 * Variable x(Tensor({batch, 1, 16, 64, 64}, DType::Float32, Device::cpu()), true);
 * Variable features = conv.forward(x);  // Shape: {batch, 32, 16, 64, 64}
 * @endcode
 */
class Conv3d : public Module {
public:
    Conv3d(int64_t in_channels,
           int64_t out_channels,
           int64_t kernel_size,
           int64_t stride = 1,
           int64_t padding = 0,
           int64_t dilation = 1,
           int64_t groups = 1,
           bool bias = true);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t in_channels_;
    int64_t out_channels_;
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
    int64_t groups_;

    auto reset_parameters() -> void;
};

} // namespace nn
} // namespace tenzor
