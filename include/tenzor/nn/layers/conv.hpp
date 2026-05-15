/**
 * @file conv.hpp
 * @brief Convolutional neural network layers (1D, 2D, transposed)
 *
 * Implements standard and transposed convolution operations for
 * spatial feature extraction in neural networks.
 */

#pragma once

#include <optional>
#include <tuple>      // Audit I5: per-axis Conv3d/ConvTranspose{2,3}d ctors
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

    // Public accessors for quantization from_float()
    auto stride_h() const -> int64_t { return stride_h_; }
    auto stride_w() const -> int64_t { return stride_w_; }
    auto padding_h() const -> int64_t { return padding_h_; }
    auto padding_w() const -> int64_t { return padding_w_; }
    auto dilation_h() const -> int64_t { return dilation_h_; }
    auto dilation_w() const -> int64_t { return dilation_w_; }
    auto groups() const -> int64_t { return groups_; }

    auto extra_repr() const -> std::string override {
        return "in_channels=" + std::to_string(in_channels_) +
               ", out_channels=" + std::to_string(out_channels_) +
               ", kernel_size=(" + std::to_string(kernel_h_) + ", " + std::to_string(kernel_w_) + ")" +
               ", stride=(" + std::to_string(stride_h_) + ", " + std::to_string(stride_w_) + ")" +
               ", padding=(" + std::to_string(padding_h_) + ", " + std::to_string(padding_w_) + ")" +
               ", bias=" + (parameters_.count("bias") ? "True" : "False");
    }

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

    auto extra_repr() const -> std::string override {
        return "in_channels=" + std::to_string(in_channels_) +
               ", out_channels=" + std::to_string(out_channels_) +
               ", kernel_size=" + std::to_string(kernel_size_) +
               ", stride=" + std::to_string(stride_) +
               ", padding=" + std::to_string(padding_) +
               ", bias=" + (parameters_.count("bias") ? "True" : "False");
    }

    // Accessors used by ONNX export (and anywhere else that needs to
    // introspect the Conv1d's shape configuration).
    auto in_channels() const -> int64_t { return in_channels_; }
    auto out_channels() const -> int64_t { return out_channels_; }
    auto kernel_size() const -> int64_t { return kernel_size_; }
    auto stride() const -> int64_t { return stride_; }
    auto padding() const -> int64_t { return padding_; }
    auto dilation() const -> int64_t { return dilation_; }
    auto groups() const -> int64_t { return groups_; }

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
     * @brief Audit I5: per-axis ConvTranspose2d ctor. Each spatial axis (H/W)
     * gets its own kernel/stride/padding/output_padding/dilation value. Use
     * this for anisotropic transposed-conv decoders.
     */
    ConvTranspose2d(int64_t in_channels,
                    int64_t out_channels,
                    std::pair<int64_t, int64_t> kernel_size,
                    std::pair<int64_t, int64_t> stride,
                    std::pair<int64_t, int64_t> padding,
                    std::pair<int64_t, int64_t> output_padding,
                    std::pair<int64_t, int64_t> dilation,
                    int64_t groups = 1,
                    bool bias = true);

    /**
     * @brief Forward pass through transposed 2D convolution.
     */
    auto forward_impl(const Variable& input) -> Variable override;

    auto extra_repr() const -> std::string override {
        return "in_channels=" + std::to_string(in_channels_) +
               ", out_channels=" + std::to_string(out_channels_) +
               ", kernel_size=(" + std::to_string(kH_) + "," + std::to_string(kW_) + ")" +
               ", stride=(" + std::to_string(sH_) + "," + std::to_string(sW_) + ")" +
               ", padding=(" + std::to_string(pH_) + "," + std::to_string(pW_) + ")" +
               ", output_padding=(" + std::to_string(opH_) + "," + std::to_string(opW_) + ")" +
               ", bias=" + (parameters_.count("bias") ? "True" : "False");
    }

    auto in_channels() const -> int64_t { return in_channels_; }
    auto out_channels() const -> int64_t { return out_channels_; }
    // Scalar accessors return the H-axis value (back-compat for isotropic callers).
    auto kernel_size() const -> int64_t { return kH_; }
    auto stride() const -> int64_t { return sH_; }
    auto padding() const -> int64_t { return pH_; }
    auto output_padding() const -> int64_t { return opH_; }
    auto dilation() const -> int64_t { return dH_; }
    auto groups() const -> int64_t { return groups_; }
    // Per-axis accessors (audit I5).
    auto kernel_size_h() const -> int64_t { return kH_; }
    auto kernel_size_w() const -> int64_t { return kW_; }
    auto stride_h() const -> int64_t { return sH_; }
    auto stride_w() const -> int64_t { return sW_; }
    auto padding_h() const -> int64_t { return pH_; }
    auto padding_w() const -> int64_t { return pW_; }
    auto output_padding_h() const -> int64_t { return opH_; }
    auto output_padding_w() const -> int64_t { return opW_; }
    auto dilation_h() const -> int64_t { return dH_; }
    auto dilation_w() const -> int64_t { return dW_; }

private:
    int64_t in_channels_;
    int64_t out_channels_;
    // Per-axis kernel / stride / padding / output_padding / dilation (audit I5).
    int64_t kH_, kW_;
    int64_t sH_, sW_;
    int64_t pH_, pW_;
    int64_t opH_, opW_;
    int64_t dH_, dW_;
    int64_t groups_;

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
    /**
     * @brief Construct 3-D convolutional layer (scalar = same value for D/H/W).
     */
    Conv3d(int64_t in_channels,
           int64_t out_channels,
           int64_t kernel_size,
           int64_t stride = 1,
           int64_t padding = 0,
           int64_t dilation = 1,
           int64_t groups = 1,
           bool bias = true);

    /**
     * @brief Audit I5: per-axis Conv3d ctor. Each spatial axis (D/H/W) gets
     * its own kernel/stride/padding/dilation value. Use this for anisotropic
     * volumetric models — temporal convolutions, 3-D segmentation networks
     * with different spatial vs depth strides, etc.
     */
    Conv3d(int64_t in_channels,
           int64_t out_channels,
           std::tuple<int64_t, int64_t, int64_t> kernel_size,
           std::tuple<int64_t, int64_t, int64_t> stride,
           std::tuple<int64_t, int64_t, int64_t> padding,
           std::tuple<int64_t, int64_t, int64_t> dilation,
           int64_t groups = 1,
           bool bias = true);

    auto forward_impl(const Variable& input) -> Variable override;

    auto extra_repr() const -> std::string override {
        return "in_channels=" + std::to_string(in_channels_) +
               ", out_channels=" + std::to_string(out_channels_) +
               ", kernel_size=(" + std::to_string(kD_) + "," + std::to_string(kH_) + "," + std::to_string(kW_) + ")" +
               ", stride=(" + std::to_string(sD_) + "," + std::to_string(sH_) + "," + std::to_string(sW_) + ")" +
               ", padding=(" + std::to_string(pD_) + "," + std::to_string(pH_) + "," + std::to_string(pW_) + ")" +
               ", bias=" + (parameters_.count("bias") ? "True" : "False");
    }

    auto in_channels() const -> int64_t { return in_channels_; }
    auto out_channels() const -> int64_t { return out_channels_; }
    // Scalar accessors return the D-axis value (back-compat for isotropic callers).
    auto kernel_size() const -> int64_t { return kD_; }
    auto stride() const -> int64_t { return sD_; }
    auto padding() const -> int64_t { return pD_; }
    auto dilation() const -> int64_t { return dD_; }
    auto groups() const -> int64_t { return groups_; }
    // Per-axis accessors (audit I5).
    auto kernel_size_d() const -> int64_t { return kD_; }
    auto kernel_size_h() const -> int64_t { return kH_; }
    auto kernel_size_w() const -> int64_t { return kW_; }
    auto stride_d() const -> int64_t { return sD_; }
    auto stride_h() const -> int64_t { return sH_; }
    auto stride_w() const -> int64_t { return sW_; }
    auto padding_d() const -> int64_t { return pD_; }
    auto padding_h() const -> int64_t { return pH_; }
    auto padding_w() const -> int64_t { return pW_; }
    auto dilation_d() const -> int64_t { return dD_; }
    auto dilation_h() const -> int64_t { return dH_; }
    auto dilation_w() const -> int64_t { return dW_; }

private:
    int64_t in_channels_;
    int64_t out_channels_;
    // Per-axis kernel / stride / padding / dilation (audit I5).
    int64_t kD_, kH_, kW_;
    int64_t sD_, sH_, sW_;
    int64_t pD_, pH_, pW_;
    int64_t dD_, dH_, dW_;
    int64_t groups_;

    auto reset_parameters() -> void;
};

/**
 * @brief 3D transposed convolutional layer.
 *
 * Applies transposed 3D convolution (deconvolution) for volumetric upsampling.
 *
 * Shape transformations:
 * - Input: (N, C_in, D_in, H_in, W_in)
 * - Output: (N, C_out, D_out, H_out, W_out)
 * - Weight: (C_in, C_out/groups, K, K, K)
 *
 * Output size formula per spatial dim:
 *   out = (in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
 */
class ConvTranspose3d : public Module {
public:
    ConvTranspose3d(int64_t in_channels,
                    int64_t out_channels,
                    int64_t kernel_size,
                    int64_t stride = 1,
                    int64_t padding = 0,
                    int64_t output_padding = 0,
                    int64_t dilation = 1,
                    int64_t groups = 1,
                    bool bias = true);

    /// Audit I5: per-axis ConvTranspose3d ctor.
    ConvTranspose3d(int64_t in_channels,
                    int64_t out_channels,
                    std::tuple<int64_t, int64_t, int64_t> kernel_size,
                    std::tuple<int64_t, int64_t, int64_t> stride,
                    std::tuple<int64_t, int64_t, int64_t> padding,
                    std::tuple<int64_t, int64_t, int64_t> output_padding,
                    std::tuple<int64_t, int64_t, int64_t> dilation,
                    int64_t groups = 1,
                    bool bias = true);

    auto forward_impl(const Variable& input) -> Variable override;

    auto extra_repr() const -> std::string override {
        return "in_channels=" + std::to_string(in_channels_) +
               ", out_channels=" + std::to_string(out_channels_) +
               ", kernel_size=(" + std::to_string(kD_) + "," + std::to_string(kH_) + "," + std::to_string(kW_) + ")" +
               ", stride=(" + std::to_string(sD_) + "," + std::to_string(sH_) + "," + std::to_string(sW_) + ")" +
               ", padding=(" + std::to_string(pD_) + "," + std::to_string(pH_) + "," + std::to_string(pW_) + ")" +
               ", output_padding=(" + std::to_string(opD_) + "," + std::to_string(opH_) + "," + std::to_string(opW_) + ")" +
               ", bias=" + (parameters_.count("bias") ? "True" : "False");
    }

    auto in_channels() const -> int64_t { return in_channels_; }
    auto out_channels() const -> int64_t { return out_channels_; }
    auto kernel_size() const -> int64_t { return kD_; }
    auto stride() const -> int64_t { return sD_; }
    auto padding() const -> int64_t { return pD_; }
    auto output_padding() const -> int64_t { return opD_; }
    auto dilation() const -> int64_t { return dD_; }
    auto groups() const -> int64_t { return groups_; }
    // Per-axis accessors (audit I5).
    auto kernel_size_d() const -> int64_t { return kD_; }
    auto kernel_size_h() const -> int64_t { return kH_; }
    auto kernel_size_w() const -> int64_t { return kW_; }
    auto stride_d() const -> int64_t { return sD_; }
    auto stride_h() const -> int64_t { return sH_; }
    auto stride_w() const -> int64_t { return sW_; }
    auto padding_d() const -> int64_t { return pD_; }
    auto padding_h() const -> int64_t { return pH_; }
    auto padding_w() const -> int64_t { return pW_; }
    auto output_padding_d() const -> int64_t { return opD_; }
    auto output_padding_h() const -> int64_t { return opH_; }
    auto output_padding_w() const -> int64_t { return opW_; }
    auto dilation_d() const -> int64_t { return dD_; }
    auto dilation_h() const -> int64_t { return dH_; }
    auto dilation_w() const -> int64_t { return dW_; }

private:
    int64_t in_channels_;
    int64_t out_channels_;
    int64_t kD_, kH_, kW_;
    int64_t sD_, sH_, sW_;
    int64_t pD_, pH_, pW_;
    int64_t opD_, opH_, opW_;
    int64_t dD_, dH_, dW_;
    int64_t groups_;

    auto reset_parameters() -> void;
};

/**
 * @brief Transposed 1D convolution (deconvolution) layer.
 *
 * Applies a 1D transposed convolution for upsampling sequences.
 * Often used in sequence generation, audio upsampling, and decoders.
 *
 * Shape transformations:
 * - Input: (N, C_in, L_in)
 * - Output: (N, C_out, L_out)
 * - Weight: (C_in, C_out/groups, K)
 *
 * Output length:
 * - L_out = (L_in - 1) * stride - 2*padding + kernel_size + output_padding
 *
 * @see Conv1d for standard 1D convolution
 * @see ConvTranspose2d for 2D transposed convolution
 */
class ConvTranspose1d : public Module {
public:
    ConvTranspose1d(int64_t in_channels,
                    int64_t out_channels,
                    int64_t kernel_size,
                    int64_t stride = 1,
                    int64_t padding = 0,
                    int64_t output_padding = 0,
                    int64_t groups = 1,
                    bool bias = true);

    auto forward_impl(const Variable& input) -> Variable override;

    auto extra_repr() const -> std::string override {
        return "in_channels=" + std::to_string(in_channels_) +
               ", out_channels=" + std::to_string(out_channels_) +
               ", kernel_size=" + std::to_string(kernel_size_) +
               ", stride=" + std::to_string(stride_) +
               ", padding=" + std::to_string(padding_) +
               ", output_padding=" + std::to_string(output_padding_) +
               ", bias=" + (parameters_.count("bias") ? "True" : "False");
    }

    auto in_channels() const -> int64_t { return in_channels_; }
    auto out_channels() const -> int64_t { return out_channels_; }
    auto kernel_size() const -> int64_t { return kernel_size_; }
    auto stride() const -> int64_t { return stride_; }
    auto padding() const -> int64_t { return padding_; }
    auto output_padding() const -> int64_t { return output_padding_; }
    auto groups() const -> int64_t { return groups_; }

private:
    int64_t in_channels_;
    int64_t out_channels_;
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    int64_t output_padding_;
    int64_t groups_;

    auto reset_parameters() -> void;
};

/**
 * @brief Deformable 2D convolutional layer (DCNv2).
 *
 * Applies deformable convolution where sampling locations are augmented
 * with learnable offsets and optional modulation masks. Unlike regular
 * convolution with a fixed receptive field, deformable conv adapts the
 * sampling grid to object geometry.
 *
 * The layer owns weight and bias parameters. Offset and mask tensors
 * are produced externally (typically by a small Conv2d predicting
 * 2*kH*kW offsets and kH*kW mask values per spatial location) and
 * passed to forward().
 *
 * Shape transformations:
 * - Input: (N, C_in, H, W)
 * - Offset: (N, offset_groups * 2 * kH * kW, H_out, W_out)
 * - Mask: (N, offset_groups * kH * kW, H_out, W_out)  [optional]
 * - Output: (N, C_out, H_out, W_out)
 *
 * Reference: Zhu et al., "Deformable ConvNets v2" (CVPR 2019)
 *
 * @code
 * DeformableConv2d dcn(64, 128, 3, 1, 1);
 * Variable input = ...;   // (N, 64, H, W)
 * Variable offset = ...;  // (N, 2*3*3, H, W) from offset predictor
 * Variable mask = ...;    // (N, 3*3, H, W) from mask predictor
 * Variable out = dcn.forward(input, offset, mask);  // (N, 128, H, W)
 * @endcode
 */
class DeformableConv2d : public Module {
public:
    /**
     * @brief Construct DeformableConv2d with square kernel.
     *
     * @param in_channels Input channels
     * @param out_channels Output channels (filters)
     * @param kernel_size Square kernel size
     * @param stride Stride (default: 1)
     * @param padding Padding (default: 0)
     * @param dilation Dilation (default: 1)
     * @param groups Channel groups (default: 1)
     * @param offset_groups Offset groups — channels_in / offset_groups gives
     *                      the number of input channels sharing each set of
     *                      offsets (default: 1)
     * @param bias If true, add learnable bias (default: true)
     */
    DeformableConv2d(int64_t in_channels,
                     int64_t out_channels,
                     int64_t kernel_size,
                     int64_t stride = 1,
                     int64_t padding = 0,
                     int64_t dilation = 1,
                     int64_t groups = 1,
                     int64_t offset_groups = 1,
                     bool bias = true);

    /**
     * @brief Forward pass through deformable 2D convolution.
     *
     * @param input Input variable of shape (N, C_in, H, W)
     * @param offset Offset variable of shape (N, offset_groups*2*kH*kW, H_out, W_out)
     * @param mask Modulation mask of shape (N, offset_groups*kH*kW, H_out, W_out).
     *             Pass a default-constructed Variable for DCNv1 (no mask).
     * @return Output variable of shape (N, C_out, H_out, W_out)
     */
    auto forward(const Variable& input, const Variable& offset,
                 const Variable& mask = Variable()) -> Variable;

    /// Module interface (single-input forward). Not supported for DeformableConv2d;
    /// use forward(input, offset, mask) instead.
    auto forward_impl(const Variable& input) -> Variable override;

    auto stride_h() const -> int64_t { return stride_; }
    auto stride_w() const -> int64_t { return stride_; }
    auto padding_h() const -> int64_t { return padding_; }
    auto padding_w() const -> int64_t { return padding_; }
    auto dilation_h() const -> int64_t { return dilation_; }
    auto dilation_w() const -> int64_t { return dilation_; }
    auto groups() const -> int64_t { return groups_; }
    auto offset_groups() const -> int64_t { return offset_groups_; }

    auto extra_repr() const -> std::string override {
        return "in_channels=" + std::to_string(in_channels_) +
               ", out_channels=" + std::to_string(out_channels_) +
               ", kernel_size=" + std::to_string(kernel_size_) +
               ", stride=" + std::to_string(stride_) +
               ", padding=" + std::to_string(padding_) +
               ", offset_groups=" + std::to_string(offset_groups_) +
               ", bias=" + (parameters_.count("bias") ? "True" : "False");
    }

private:
    int64_t in_channels_;
    int64_t out_channels_;
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
    int64_t groups_;
    int64_t offset_groups_;

    auto reset_parameters() -> void;
};

} // namespace nn
} // namespace tenzor
