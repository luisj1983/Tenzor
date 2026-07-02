/**
 * @file convnext.hpp
 * @brief ConvNeXt - A modernized ConvNet for the 2020s
 *
 * Implements the ConvNeXt family of convolutional neural networks.
 * ConvNeXt modernizes ResNet by incorporating design choices from Vision Transformers
 * while remaining purely convolutional, achieving competitive performance with Transformers.
 *
 * Key innovations:
 * - Large 7x7 depthwise convolutions
 * - Inverted bottleneck (expand then compress)
 * - LayerNorm instead of BatchNorm
 * - GELU activation
 * - Layer Scale for training stability
 * - Stochastic depth (drop path)
 *
 * Reference: "A ConvNet for the 2020s" (Liu et al., CVPR 2022)
 * Paper URL: https://arxiv.org/abs/2201.03545
 */

#pragma once

#include <memory>
#include <vector>
#include "../nn/module.hpp"
#include "../nn/layers/conv.hpp"
#include "../nn/layers/normalization.hpp"
#include "../nn/layers/linear.hpp"
#include "../nn/layers/pooling.hpp"
#include "../nn/layers/dropout.hpp"
#include "../nn/layers/drop_path.hpp"
#include "../nn/activations/activations.hpp"

namespace tenzor {
namespace models {

/**
 * @brief Layer Scale module for training stability.
 *
 * Layer Scale applies learnable per-channel scaling factors.
 * Initialized to a small value (1e-6) to improve training stability,
 * especially for deeper models.
 *
 * Formula: output = gamma * input
 * where gamma is learnable and initialized to init_value.
 *
 * Reference: "Going deeper with Image Transformers" (Touvron et al., 2021)
 */
class LayerScale : public nn::Module {
public:
    /**
     * @brief Construct LayerScale module.
     *
     * @param dim Number of channels
     * @param init_value Initial value for scale parameters (default: 1e-6)
     */
    LayerScale(int64_t dim, double init_value = 1e-6);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    Variable gamma_;  ///< Learnable scale parameter [dim, 1, 1]
};

/**
 * @brief ConvNeXt building block.
 *
 * Architecture:
 * ```
 * Input (C channels)
 *   ↓
 * Depthwise Conv 7×7 (C → C)
 *   ↓
 * LayerNorm (channels_first)
 *   ↓
 * Pointwise Conv 1×1 (C → 4C) - Expansion
 *   ↓
 * GELU
 *   ↓
 * Pointwise Conv 1×1 (4C → C) - Projection
 *   ↓
 * Layer Scale
 *   ↓
 * [Stochastic Depth]
 *   ↓
 * Residual Connection
 *   ↓
 * Output (C channels)
 * ```
 *
 * Key differences from ResNet Bottleneck:
 * - Depthwise conv at the beginning (not middle)
 * - Larger kernel size (7×7 vs 3×3)
 * - Inverted bottleneck (expand then compress, not compress-expand-compress)
 * - LayerNorm instead of BatchNorm
 * - GELU instead of ReLU
 * - Fewer activations (only one GELU per block)
 */
class ConvNeXtBlock : public nn::Module {
public:
    /**
     * @brief Construct ConvNeXt block.
     *
     * @param dim Number of input/output channels
     * @param drop_path Drop path rate (stochastic depth) [0.0, 1.0]
     * @param layer_scale_init_value Initial value for layer scale (default: 1e-6)
     */
    ConvNeXtBlock(int64_t dim,
                  double drop_path = 0.0,
                  double layer_scale_init_value = 1e-6);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::shared_ptr<nn::Conv2d> dwconv_;        ///< 7×7 depthwise convolution
    std::shared_ptr<nn::LayerNorm> norm_;       ///< Layer normalization
    std::shared_ptr<nn::Conv2d> pwconv1_;       ///< 1×1 expansion conv
    nn::GELU gelu_;                             ///< GELU activation
    std::shared_ptr<nn::Conv2d> pwconv2_;       ///< 1×1 projection conv
    Variable gamma_;                            ///< Layer scale parameter, timm-flat shape [dim] (reshaped to [dim,1,1] at use)
    std::shared_ptr<nn::DropPath> drop_path_;   ///< Stochastic depth (per-sample, seedable, train/eval-gated)
};

/**
 * @brief ConvNeXt model.
 *
 * ConvNeXt is a purely convolutional architecture that achieves competitive
 * performance with Vision Transformers by modernizing the standard ResNet design.
 *
 * Architecture:
 * 1. Stem: Conv 4×4, stride=4 (aggressive downsampling like ViT)
 * 2. Four stages with ConvNeXt blocks
 * 3. Downsampling between stages: LayerNorm + Conv 2×2, stride=2
 * 4. Global average pooling + LayerNorm + FC
 *
 * Input: (N, 3, 224, 224)
 * Output: (N, num_classes)
 */
class ConvNeXt : public nn::Module {
public:
    /**
     * @brief Construct ConvNeXt model.
     *
     * @param in_channels Number of input channels (default: 3 for RGB)
     * @param num_classes Number of output classes (default: 1000)
     * @param depths Number of blocks in each stage [stage1, stage2, stage3, stage4]
     * @param dims Number of channels in each stage [stage1, stage2, stage3, stage4]
     * @param drop_path_rate Maximum drop path rate (linearly scaled across depth)
     * @param layer_scale_init_value Initial value for layer scale
     */
    ConvNeXt(int64_t in_channels = 3,
             int64_t num_classes = 1000,
             const std::vector<int64_t>& depths = {3, 3, 9, 3},
             const std::vector<int64_t>& dims = {96, 192, 384, 768},
             double drop_path_rate = 0.0,
             double layer_scale_init_value = 1e-6);

    /**
     * @brief Forward pass through ConvNeXt.
     *
     * @param input Input image tensor of shape (N, C, H, W)
     * @return Output logits of shape (N, num_classes)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Load pretrained weights.
     *
     * @param path Path to pretrained weights file
     */
    auto load_pretrained(const std::string& path) -> void;

private:
    /**
     * @brief Create a ConvNeXt stage with multiple blocks.
     */
    auto make_stage(int64_t dim, int64_t depth, double drop_path_start, double drop_path_end)
        -> std::shared_ptr<nn::Sequential>;

    std::vector<int64_t> depths_;
    std::vector<int64_t> dims_;
    double layer_scale_init_value_;

    // Network layers
    std::shared_ptr<nn::Conv2d> stem_conv_;
    std::shared_ptr<nn::LayerNorm> stem_norm_;
    std::shared_ptr<nn::Sequential> stage1_;
    std::shared_ptr<nn::Sequential> downsample1_conv_;
    std::shared_ptr<nn::LayerNorm> downsample1_norm_;
    std::shared_ptr<nn::Sequential> stage2_;
    std::shared_ptr<nn::Sequential> downsample2_conv_;
    std::shared_ptr<nn::LayerNorm> downsample2_norm_;
    std::shared_ptr<nn::Sequential> stage3_;
    std::shared_ptr<nn::Sequential> downsample3_conv_;
    std::shared_ptr<nn::LayerNorm> downsample3_norm_;
    std::shared_ptr<nn::Sequential> stage4_;
    std::shared_ptr<nn::LayerNorm> norm_;
    std::shared_ptr<nn::AdaptiveAvgPool2d> avgpool_;
    std::shared_ptr<nn::Linear> head_;
};

// ============================================================================
// Factory Functions for ConvNeXt Variants
// ============================================================================

/**
 * @brief Create ConvNeXt-Tiny model.
 *
 * Architecture: Blocks [3, 3, 9, 3], Channels [96, 192, 384, 768]
 * Parameters: ~28M
 * FLOPs: ~4.5G (at 224×224)
 * Top-1 Accuracy (ImageNet-1K): ~82.1%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to ConvNeXt-Tiny model
 */
auto convnext_tiny(int64_t num_classes = 1000, bool pretrained = false)
    -> std::shared_ptr<ConvNeXt>;

/**
 * @brief Create ConvNeXt-Small model.
 *
 * Architecture: Blocks [3, 3, 27, 3], Channels [96, 192, 384, 768]
 * Parameters: ~50M
 * FLOPs: ~8.7G (at 224×224)
 * Top-1 Accuracy (ImageNet-1K): ~83.1%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to ConvNeXt-Small model
 */
auto convnext_small(int64_t num_classes = 1000, bool pretrained = false)
    -> std::shared_ptr<ConvNeXt>;

/**
 * @brief Create ConvNeXt-Base model.
 *
 * Architecture: Blocks [3, 3, 27, 3], Channels [128, 256, 512, 1024]
 * Parameters: ~89M
 * FLOPs: ~15.4G (at 224×224)
 * Top-1 Accuracy (ImageNet-1K): ~83.8%
 * Top-1 Accuracy (ImageNet-22K → 1K): ~85.8%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to ConvNeXt-Base model
 */
auto convnext_base(int64_t num_classes = 1000, bool pretrained = false)
    -> std::shared_ptr<ConvNeXt>;

/**
 * @brief Create ConvNeXt-Large model.
 *
 * Architecture: Blocks [3, 3, 27, 3], Channels [192, 384, 768, 1536]
 * Parameters: ~198M
 * FLOPs: ~34.4G (at 224×224)
 * Top-1 Accuracy (ImageNet-1K): ~84.3%
 * Top-1 Accuracy (ImageNet-22K → 1K): ~86.6%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to ConvNeXt-Large model
 */
auto convnext_large(int64_t num_classes = 1000, bool pretrained = false)
    -> std::shared_ptr<ConvNeXt>;

/**
 * @brief Create ConvNeXt-XLarge model.
 *
 * Architecture: Blocks [3, 3, 27, 3], Channels [256, 512, 1024, 2048]
 * Parameters: ~350M
 * FLOPs: ~60.9G (at 224×224)
 * Top-1 Accuracy (ImageNet-22K → 1K): ~87.0%
 *
 * Note: XLarge is only available with ImageNet-22K pretraining.
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet-22K weights (default: false)
 * @return Shared pointer to ConvNeXt-XLarge model
 */
auto convnext_xlarge(int64_t num_classes = 1000, bool pretrained = false)
    -> std::shared_ptr<ConvNeXt>;

} // namespace models
} // namespace tenzor
