/**
 * @file resnet.hpp
 * @brief ResNet family of deep residual networks
 *
 * Implements the complete ResNet family including BasicBlock and Bottleneck
 * architectures, supporting ResNet-18, ResNet-34, ResNet-50, ResNet-101,
 * ResNet-152, ResNeXt, and Wide ResNet variants.
 *
 * Reference: "Deep Residual Learning for Image Recognition" (He et al., 2015)
 * Reference: "Aggregated Residual Transformations for Deep Neural Networks" (Xie et al., 2017)
 */

#pragma once

#include <memory>
#include <vector>
#include "../nn/module.hpp"
#include "../nn/layers/conv.hpp"
#include "../nn/layers/batchnorm.hpp"
#include "../nn/layers/linear.hpp"
#include "../nn/layers/pooling.hpp"
#include "../nn/activations/activations.hpp"

namespace tenzor {
namespace models {

/**
 * @brief Basic building block for ResNet-18 and ResNet-34.
 *
 * BasicBlock consists of two 3x3 convolutions with batch normalization
 * and skip connections. This is the building block for shallower ResNets.
 *
 * Architecture:
 * ```
 * x -> [Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN] -> + -> ReLU
 *  |______________________________________________|
 *                    (skip connection)
 * ```
 *
 * If stride != 1 or in_channels != out_channels, a 1x1 conv downsamples
 * the skip connection to match dimensions.
 */
class BasicBlock : public nn::Module {
public:
    static constexpr int64_t expansion = 1;  ///< Channel expansion factor

    /**
     * @brief Construct BasicBlock.
     *
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param stride Stride for first convolution (default: 1)
     * @param groups Number of groups for grouped convolutions (default: 1)
     * @param base_width Base width for grouped convolutions (default: 64)
     * @param downsample Optional downsample module for skip connection
     */
    BasicBlock(int64_t in_channels,
               int64_t out_channels,
               int64_t stride = 1,
               int64_t groups = 1,
               int64_t base_width = 64,
               std::shared_ptr<nn::Module> downsample = nullptr);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::shared_ptr<nn::Conv2d> conv1_;
    std::shared_ptr<nn::BatchNorm2d> bn1_;
    std::shared_ptr<nn::Conv2d> conv2_;
    std::shared_ptr<nn::BatchNorm2d> bn2_;
    std::shared_ptr<nn::Module> downsample_;
    nn::ReLU relu_;
};

/**
 * @brief Bottleneck building block for ResNet-50, ResNet-101, ResNet-152.
 *
 * Bottleneck consists of 1x1 -> 3x3 -> 1x1 convolutions with batch normalization
 * and skip connections. The 1x1 convolutions reduce and restore dimensions,
 * making the 3x3 convolution more efficient.
 *
 * Architecture:
 * ```
 * x -> [Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1 -> BN] -> + -> ReLU
 *  |__________________________________________________________________|
 *                         (skip connection)
 * ```
 *
 * Output channels = out_channels * expansion (expansion = 4)
 */
class Bottleneck : public nn::Module {
public:
    static constexpr int64_t expansion = 4;  ///< Channel expansion factor

    /**
     * @brief Construct Bottleneck block.
     *
     * @param in_channels Number of input channels
     * @param out_channels Number of intermediate channels (before expansion)
     * @param stride Stride for 3x3 convolution (default: 1)
     * @param groups Number of groups for 3x3 convolution (default: 1)
     * @param base_width Base width for grouped convolutions (default: 64)
     * @param downsample Optional downsample module for skip connection
     */
    Bottleneck(int64_t in_channels,
               int64_t out_channels,
               int64_t stride = 1,
               int64_t groups = 1,
               int64_t base_width = 64,
               std::shared_ptr<nn::Module> downsample = nullptr);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::shared_ptr<nn::Conv2d> conv1_;
    std::shared_ptr<nn::BatchNorm2d> bn1_;
    std::shared_ptr<nn::Conv2d> conv2_;
    std::shared_ptr<nn::BatchNorm2d> bn2_;
    std::shared_ptr<nn::Conv2d> conv3_;
    std::shared_ptr<nn::BatchNorm2d> bn3_;
    std::shared_ptr<nn::Module> downsample_;
    nn::ReLU relu_;
};

/**
 * @brief ResNet base class for all variants.
 *
 * ResNet (Residual Network) is a deep convolutional neural network architecture
 * that uses skip connections to enable training of very deep networks (100+ layers).
 *
 * Standard architecture:
 * 1. Initial conv (7x7, stride 2) + BN + ReLU + MaxPool
 * 2. Four residual layer groups (conv2_x, conv3_x, conv4_x, conv5_x)
 * 3. Global average pooling
 * 4. Fully connected layer for classification
 *
 * Input: (N, 3, 224, 224) for ImageNet
 * Output: (N, num_classes)
 */
class ResNet : public nn::Module {
public:
    /**
     * @brief Construct ResNet with BasicBlock architecture.
     *
     * Used for ResNet-18 and ResNet-34.
     *
     * @param layers Number of blocks in each of the 4 layers [layer1, layer2, layer3, layer4]
     * @param num_classes Number of output classes (default: 1000 for ImageNet)
     * @param use_basic_block Must be true for BasicBlock architecture
     * @param groups Number of groups (must be 1 for BasicBlock)
     * @param width_per_group Base width (must be 64 for BasicBlock)
     */
    ResNet(const std::vector<int64_t>& layers,
           int64_t num_classes,
           bool use_basic_block,
           int64_t groups = 1,
           int64_t width_per_group = 64);

    /**
     * @brief Forward pass through ResNet.
     *
     * @param input Input image tensor of shape (N, 3, H, W)
     * @return Output logits of shape (N, num_classes)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Extract feature maps for detection tasks.
     *
     * Returns features from layer4 (C5) before global average pooling
     * and classification head. Used for object detection and segmentation.
     *
     * @param input Input image tensor of shape (N, 3, H, W)
     * @return Feature maps of shape (N, C, H/32, W/32)
     *         where C=512 for ResNet-18/34 or C=2048 for ResNet-50/101/152
     */
    auto forward_features(const Variable& input) -> Variable;

    /**
     * @brief Load pretrained weights.
     *
     * @param path Path to pretrained weights file
     * @throws std::runtime_error if file doesn't exist or format is invalid
     */
    auto load_pretrained(const std::string& path) -> void;

private:
    /**
     * @brief Create a residual layer with BasicBlock blocks.
     */
    auto make_layer_basic(int64_t out_channels, int64_t num_blocks, int64_t stride)
        -> std::shared_ptr<nn::Sequential>;

    /**
     * @brief Create a residual layer with Bottleneck blocks.
     */
    auto make_layer_bottleneck(int64_t out_channels, int64_t num_blocks, int64_t stride)
        -> std::shared_ptr<nn::Sequential>;

    bool use_basic_block_;         ///< True for BasicBlock, false for Bottleneck
    int64_t in_channels_{64};      ///< Current number of channels
    int64_t groups_{1};            ///< Groups for convolutions
    int64_t base_width_{64};       ///< Base width for channels

    // Network layers
    std::shared_ptr<nn::Conv2d> conv1_;
    std::shared_ptr<nn::BatchNorm2d> bn1_;
    nn::ReLU relu_;
    std::shared_ptr<nn::MaxPool2d> maxpool_;
    std::shared_ptr<nn::Sequential> layer1_;
    std::shared_ptr<nn::Sequential> layer2_;
    std::shared_ptr<nn::Sequential> layer3_;
    std::shared_ptr<nn::Sequential> layer4_;
    std::shared_ptr<nn::AdaptiveAvgPool2d> avgpool_;
    std::shared_ptr<nn::Linear> fc_;
};

// ============================================================================
// Factory Functions for Standard ResNet Variants
// ============================================================================

/**
 * @brief Create ResNet-18 model.
 *
 * Architecture: BasicBlock with [2, 2, 2, 2] blocks
 * Parameters: ~11.7M
 * Top-1 Accuracy (ImageNet): ~69.8%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to ResNet-18 model
 *
 * @code
 * auto model = models::resnet18(1000, false);
 * Variable output = model->forward(input);
 * @endcode
 */
auto resnet18(int64_t num_classes = 1000, bool pretrained = false) -> std::shared_ptr<ResNet>;

/**
 * @brief Create ResNet-34 model.
 *
 * Architecture: BasicBlock with [3, 4, 6, 3] blocks
 * Parameters: ~21.8M
 * Top-1 Accuracy (ImageNet): ~73.3%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to ResNet-34 model
 */
auto resnet34(int64_t num_classes = 1000, bool pretrained = false) -> std::shared_ptr<ResNet>;

/**
 * @brief Create ResNet-50 model.
 *
 * Architecture: Bottleneck with [3, 4, 6, 3] blocks
 * Parameters: ~25.6M
 * Top-1 Accuracy (ImageNet): ~76.1%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to ResNet-50 model
 */
auto resnet50(int64_t num_classes = 1000, bool pretrained = false) -> std::shared_ptr<ResNet>;

/**
 * @brief Create ResNet-101 model.
 *
 * Architecture: Bottleneck with [3, 4, 23, 3] blocks
 * Parameters: ~44.5M
 * Top-1 Accuracy (ImageNet): ~77.4%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to ResNet-101 model
 */
auto resnet101(int64_t num_classes = 1000, bool pretrained = false) -> std::shared_ptr<ResNet>;

/**
 * @brief Create ResNet-152 model.
 *
 * Architecture: Bottleneck with [3, 8, 36, 3] blocks
 * Parameters: ~60.2M
 * Top-1 Accuracy (ImageNet): ~78.3%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to ResNet-152 model
 */
auto resnet152(int64_t num_classes = 1000, bool pretrained = false) -> std::shared_ptr<ResNet>;

/**
 * @brief Create ResNeXt-50 (32x4d) model.
 *
 * ResNeXt uses grouped convolutions for improved representational power.
 * Architecture: Bottleneck with [3, 4, 6, 3] blocks, 32 groups, width 4
 * Parameters: ~25.0M
 * Top-1 Accuracy (ImageNet): ~77.6%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to ResNeXt-50 model
 */
auto resnext50_32x4d(int64_t num_classes = 1000, bool pretrained = false) -> std::shared_ptr<ResNet>;

/**
 * @brief Create ResNeXt-101 (32x8d) model.
 *
 * Architecture: Bottleneck with [3, 4, 23, 3] blocks, 32 groups, width 8
 * Parameters: ~88.8M
 * Top-1 Accuracy (ImageNet): ~79.3%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to ResNeXt-101 model
 */
auto resnext101_32x8d(int64_t num_classes = 1000, bool pretrained = false) -> std::shared_ptr<ResNet>;

/**
 * @brief Create Wide ResNet-50-2 model.
 *
 * Wide ResNet increases channel width for improved accuracy.
 * Architecture: Bottleneck with [3, 4, 6, 3] blocks, width 128 (2x standard)
 * Parameters: ~68.9M
 * Top-1 Accuracy (ImageNet): ~78.5%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to Wide ResNet-50-2 model
 */
auto wide_resnet50_2(int64_t num_classes = 1000, bool pretrained = false) -> std::shared_ptr<ResNet>;

/**
 * @brief Create Wide ResNet-101-2 model.
 *
 * Architecture: Bottleneck with [3, 4, 23, 3] blocks, width 128 (2x standard)
 * Parameters: ~126.9M
 * Top-1 Accuracy (ImageNet): ~78.8%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to Wide ResNet-101-2 model
 */
auto wide_resnet101_2(int64_t num_classes = 1000, bool pretrained = false) -> std::shared_ptr<ResNet>;

} // namespace models
} // namespace tenzor
