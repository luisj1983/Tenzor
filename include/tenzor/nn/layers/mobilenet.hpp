/**
 * @file mobilenet.hpp
 * @brief MobileNet and EfficientNet building blocks
 *
 * Implements efficient CNN components for mobile and embedded vision:
 * - SqueezeExcitation: Channel attention module
 * - InvertedResidual: Mobile inverted bottleneck (MBConv)
 * - FusedMBConv: Fused variant for early layers
 */

#pragma once

#include "../module.hpp"
#include <string>

namespace tenzor {
namespace nn {

/**
 * @brief Squeeze-and-Excitation block for channel attention.
 *
 * Recalibrates channel-wise feature responses by modeling channel
 * interdependencies using global pooling and gating mechanism.
 * Used in EfficientNet, MobileNetV3, ResNet-SE, and many modern CNNs.
 *
 * Architecture:
 *   Input (B, C, H, W)
 *   -> GlobalAvgPool -> (B, C, 1, 1)
 *   -> FC: C -> C/reduction_ratio
 *   -> Activation (ReLU or Swish)
 *   -> FC: C/reduction_ratio -> C
 *   -> Sigmoid
 *   -> Scale input channels element-wise
 *   -> Output (B, C, H, W)
 *
 * Paper: "Squeeze-and-Excitation Networks" (2018)
 * https://arxiv.org/abs/1709.01507
 *
 * @code
 * SqueezeExcitation se(256, 16);  // 256 channels, reduction=16
 *
 * Variable x({batch, 256, 28, 28}, DType::Float32, Device::cpu(), true);
 * Variable out = se.forward(x);  // Same shape, channel-wise rescaled
 * @endcode
 */
class SqueezeExcitation : public Module {
public:
    /**
     * @brief Construct SE block.
     *
     * @param channels Number of input/output channels
     * @param reduction Reduction ratio for bottleneck (default: 16)
     * @param activation Activation function ("relu" or "swish", default: "relu")
     *
     * @code
     * SqueezeExcitation se1(256, 16, "relu");   // Standard SE
     * SqueezeExcitation se2(64, 4, "swish");    // EfficientNet-style
     * @endcode
     */
    SqueezeExcitation(int64_t channels,
                      int64_t reduction = 16,
                      std::string activation = "relu");

    /**
     * @brief Forward pass through SE block.
     *
     * @param input Input features of shape (N, C, H, W)
     * @return Channel-wise rescaled features of shape (N, C, H, W)
     */
    auto forward(const Variable& input) -> Variable override;

private:
    int64_t channels_;          ///< Number of channels
    int64_t reduction_;         ///< Reduction ratio
    std::string activation_;    ///< Activation type

    std::shared_ptr<Module> pool_;  ///< Global average pooling
    std::shared_ptr<Module> fc1_;   ///< First fully connected layer
    std::shared_ptr<Module> fc2_;   ///< Second fully connected layer
    std::shared_ptr<Module> act_;   ///< Activation function
};

/**
 * @brief Inverted residual block (Mobile Inverted Bottleneck Convolution).
 *
 * Core building block of MobileNetV2, MobileNetV3, and EfficientNet.
 * Uses inverted bottleneck structure: narrow -> wide -> narrow (opposite
 * of traditional ResNet bottleneck).
 *
 * Architecture:
 *   1. [Optional] Expansion: 1x1 conv (C -> expand_ratio*C)
 *      - Skipped if expand_ratio == 1
 *   2. Depthwise: 3x3 or 5x5 depthwise conv (groups = channels)
 *   3. [Optional] Squeeze-and-Excitation
 *   4. Projection: 1x1 conv (expand_ratio*C -> out_channels)
 *      - NO activation (linear bottleneck)
 *   5. [Optional] Skip connection if stride==1 and in_channels==out_channels
 *
 * Key features:
 * - Linear bottleneck: No activation after final projection
 * - Depthwise separable convolutions for efficiency
 * - Optional SE module for channel attention
 *
 * Papers:
 * - MobileNetV2: https://arxiv.org/abs/1801.04381
 * - EfficientNet: https://arxiv.org/abs/1905.11946
 *
 * @code
 * InvertedResidual block(32, 64, 6, 1, true, 3);
 * // in=32, out=64, expand=6, stride=1, use_se=true, kernel=3
 *
 * Variable x({batch, 32, 56, 56}, DType::Float32, Device::cpu(), true);
 * Variable out = block.forward(x);  // {batch, 64, 56, 56}
 * @endcode
 */
class InvertedResidual : public Module {
public:
    /**
     * @brief Construct inverted residual block.
     *
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param expand_ratio Channel expansion ratio (default: 6)
     * @param stride Stride for depthwise conv (default: 1)
     * @param use_se Use Squeeze-and-Excitation (default: false)
     * @param kernel_size Depthwise kernel size (default: 3)
     * @param activation Activation type ("relu", "relu6", "swish", default: "relu6")
     *
     * @code
     * InvertedResidual mb1(32, 16, 1, 1);        // MBConv1
     * InvertedResidual mb6(32, 64, 6, 2, true);  // MBConv6 with SE
     * @endcode
     */
    InvertedResidual(int64_t in_channels,
                     int64_t out_channels,
                     int64_t expand_ratio = 6,
                     int64_t stride = 1,
                     bool use_se = false,
                     int64_t kernel_size = 3,
                     std::string activation = "relu6");

    /**
     * @brief Forward pass through inverted residual block.
     *
     * @param input Input features of shape (N, in_channels, H, W)
     * @return Output features of shape (N, out_channels, H/stride, W/stride)
     */
    auto forward(const Variable& input) -> Variable override;

private:
    int64_t in_channels_;       ///< Input channels
    int64_t out_channels_;      ///< Output channels
    int64_t stride_;            ///< Stride
    bool use_residual_;         ///< Use skip connection

    std::shared_ptr<Module> conv_;  ///< Main conv path (Sequential)
    std::shared_ptr<Module> se_;    ///< Optional SE block
};

/**
 * @brief Fused Mobile Inverted Bottleneck (for EfficientNetV2).
 *
 * Replaces depthwise separable convolution (expansion + depthwise) with
 * a single regular 3x3 convolution. More efficient in early layers where
 * resolution is high and computation is memory-bound.
 *
 * Architecture:
 *   1. Fused: 3x3 regular conv (C -> expand_ratio*C)
 *   2. [Optional] Squeeze-and-Excitation
 *   3. Projection: 1x1 conv (expand_ratio*C -> out_channels)
 *   4. [Optional] Skip connection
 *
 * Paper: "EfficientNetV2: Smaller Models and Faster Training"
 * https://arxiv.org/abs/2104.00298
 *
 * @code
 * FusedMBConv block(32, 64, 4, 1, false);
 * // Fused expansion with ratio 4
 * @endcode
 */
class FusedMBConv : public Module {
public:
    /**
     * @brief Construct fused MBConv block.
     *
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param expand_ratio Channel expansion ratio (default: 4)
     * @param stride Stride for fused conv (default: 1)
     * @param use_se Use Squeeze-and-Excitation (default: false)
     * @param activation Activation type (default: "swish")
     *
     * @code
     * FusedMBConv fmb(32, 64, 4, 2);  // 2x downsampling
     * @endcode
     */
    FusedMBConv(int64_t in_channels,
                int64_t out_channels,
                int64_t expand_ratio = 4,
                int64_t stride = 1,
                bool use_se = false,
                std::string activation = "swish");

    /**
     * @brief Forward pass through fused MBConv block.
     *
     * @param input Input features of shape (N, in_channels, H, W)
     * @return Output features of shape (N, out_channels, H/stride, W/stride)
     */
    auto forward(const Variable& input) -> Variable override;

private:
    int64_t in_channels_;       ///< Input channels
    int64_t out_channels_;      ///< Output channels
    int64_t stride_;            ///< Stride
    bool use_residual_;         ///< Use skip connection

    std::shared_ptr<Module> conv_;  ///< Main conv path
    std::shared_ptr<Module> se_;    ///< Optional SE block
};

} // namespace nn
} // namespace tenzor
