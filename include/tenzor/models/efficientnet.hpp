/**
 * @file efficientnet.hpp
 * @brief EfficientNet family of compound-scaled CNNs (B0-B7)
 *
 * Implements the complete EfficientNet family using compound scaling to
 * uniformly scale network width, depth, and resolution.
 *
 * Reference: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
 * Mingxing Tan, Quoc V. Le (Google Research), ICML 2019
 * Paper: https://arxiv.org/abs/1905.11946
 *
 * Key Features:
 * - Compound scaling: Balances depth, width, and resolution
 * - MBConv blocks: Inverted residual with depthwise separable convolutions
 * - Squeeze-and-Excitation (SE) modules for channel attention
 * - Swish activation throughout
 * - Stochastic depth for regularization
 */

#pragma once

#include <memory>
#include <vector>
#include "../nn/module.hpp"
#include "../nn/layers/conv.hpp"
#include "../nn/layers/batchnorm.hpp"
#include "../nn/layers/linear.hpp"
#include "../nn/layers/pooling.hpp"
#include "../nn/layers/dropout.hpp"
#include "../nn/layers/drop_path.hpp"  // Audit G14: real stochastic depth
#include "../nn/activations/activations.hpp"

namespace tenzor {
namespace models {

/**
 * @brief Configuration for a single MBConv stage
 */
struct MBConvConfig {
    int64_t expand_ratio;      ///< Expansion ratio for bottleneck (1 or 6)
    int64_t in_channels;       ///< Input channels
    int64_t out_channels;      ///< Output channels
    int64_t num_layers;        ///< Number of repeated blocks
    int64_t kernel_size;       ///< Depthwise conv kernel size (3 or 5)
    int64_t stride;            ///< Stride for first block (1 or 2)
    bool use_se;               ///< Use Squeeze-and-Excitation module
    double se_ratio;           ///< SE reduction ratio (default: 0.25)
};

/**
 * @brief EfficientNet compound scaling configuration
 */
struct EfficientNetConfig {
    double width_mult;         ///< Width multiplier (channels)
    double depth_mult;         ///< Depth multiplier (layers)
    int64_t resolution;        ///< Input resolution (e.g., 224, 240, etc.)
    double dropout_rate;       ///< Dropout rate before classifier
    double drop_connect_rate;  ///< Stochastic depth rate
    int64_t num_classes;       ///< Number of output classes

    std::vector<MBConvConfig> block_configs;  ///< MBConv block configurations

    /**
     * @brief Get baseline EfficientNet-B0 configuration
     */
    static EfficientNetConfig efficientnet_b0(int64_t num_classes = 1000);

    /**
     * @brief Apply compound scaling to baseline config
     * @param phi Compound coefficient (0 for B0, 0.5 for B1, etc.)
     */
    void apply_compound_scaling(double phi);

    /**
     * @brief Round channels to nearest multiple of divisor
     */
    static int64_t round_channels(double channels, int64_t divisor = 8);

    /**
     * @brief Round number of layers
     */
    static int64_t round_layers(double layers);
};

/**
 * @brief Squeeze-and-Excitation module for channel attention
 *
 * SE modules recalibrate channel-wise feature responses by explicitly
 * modeling interdependencies between channels.
 *
 * Architecture:
 * ```
 * Input (C channels)
 *   ↓
 * Global Average Pooling → [1×1×C]
 *   ↓
 * FC: C → C/r (reduction)
 *   ↓
 * Swish
 *   ↓
 * FC: C/r → C (expansion)
 *   ↓
 * Sigmoid
 *   ↓
 * Channel-wise Multiplication with Input
 *   ↓
 * Output
 * ```
 *
 * @param channels Number of input channels
 * @param reduction_ratio Reduction ratio (default: 0.25 for EfficientNet)
 */
class EfficientNetSqueezeExcitation : public nn::Module {
public:
    /**
     * @brief Construct SE module
     * @param channels Number of input channels
     * @param reduction_ratio Reduction ratio (channels / reduced_channels)
     */
    // `channels` is the (expanded) feature-map channel count the SE convs
    // operate on. `reduction_base_channels`, when > 0, is the block's INPUT
    // channel count that the squeezed (bottleneck) width is computed from —
    // the EfficientNet/torchvision convention reduces relative to block input,
    // not the expanded channels. Defaults to `channels` for backward compat.
    EfficientNetSqueezeExcitation(int64_t channels, double reduction_ratio = 0.25,
                                  int64_t reduction_base_channels = -1);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::shared_ptr<nn::AdaptiveAvgPool2d> pool_;
    std::shared_ptr<nn::Conv2d> fc1_;      // Use 1x1 conv instead of FC
    std::shared_ptr<nn::Conv2d> fc2_;      // More efficient for 2D inputs
    nn::Swish swish_;
    nn::Sigmoid sigmoid_;
    int64_t reduced_channels_;
};

/**
 * @brief Mobile Inverted Bottleneck Convolution (MBConv) block
 *
 * MBConv is the core building block of EfficientNet. It combines:
 * - Inverted residual structure (narrow → wide → narrow)
 * - Depthwise separable convolutions
 * - Squeeze-and-Excitation attention
 * - Skip connections with stochastic depth
 *
 * Architecture:
 * ```
 * Input
 *   ↓
 * [Optional: 1×1 Conv Expansion] (if expand_ratio != 1)
 *   ↓
 * Batch Normalization + Swish
 *   ↓
 * Depthwise Conv (3×3 or 5×5)
 *   ↓
 * Batch Normalization + Swish
 *   ↓
 * Squeeze-and-Excitation Module
 *   ↓
 * 1×1 Conv Projection
 *   ↓
 * Batch Normalization
 *   ↓
 * [Stochastic Depth + Skip Connection]
 *   ↓
 * Output
 * ```
 */
class MBConvBlock : public nn::Module {
public:
    /**
     * @brief Construct MBConv block
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param expand_ratio Expansion ratio (1 or 6)
     * @param kernel_size Depthwise conv kernel size (3 or 5)
     * @param stride Stride for depthwise conv (1 or 2)
     * @param use_se Use Squeeze-and-Excitation module
     * @param se_ratio SE reduction ratio
     * @param drop_connect_rate Stochastic depth probability
     */
    MBConvBlock(int64_t in_channels,
                int64_t out_channels,
                int64_t expand_ratio,
                int64_t kernel_size,
                int64_t stride,
                bool use_se = true,
                double se_ratio = 0.25,
                double drop_connect_rate = 0.0);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    bool has_expansion_;
    bool has_skip_;
    int64_t expanded_channels_;
    double drop_connect_rate_;

    // Expansion phase (optional)
    std::shared_ptr<nn::Conv2d> expand_conv_;
    std::shared_ptr<nn::BatchNorm2d> expand_bn_;

    // Depthwise convolution
    std::shared_ptr<nn::Conv2d> depthwise_conv_;
    std::shared_ptr<nn::BatchNorm2d> depthwise_bn_;

    // Squeeze-and-Excitation (optional)
    std::shared_ptr<EfficientNetSqueezeExcitation> se_;

    // Projection phase
    std::shared_ptr<nn::Conv2d> project_conv_;
    std::shared_ptr<nn::BatchNorm2d> project_bn_;

    // Audit G14: real stochastic depth. nullptr when the block has no skip
    // connection or drop_connect_rate <= 0 (DropPath becomes a no-op in
    // either case). When non-null, applied to the residual branch before
    // adding `input` in `forward_impl`.
    std::shared_ptr<nn::DropPath> drop_path_;

    nn::Swish swish_;
};

/**
 * @brief EfficientNet base class
 *
 * EfficientNet uses compound scaling to uniformly scale network width, depth,
 * and resolution, achieving state-of-the-art accuracy with significantly fewer
 * parameters and FLOPs compared to previous CNNs.
 *
 * Compound Scaling Formula:
 * - depth: d = α^φ
 * - width: w = β^φ
 * - resolution: r = γ^φ
 *
 * where α=1.2, β=1.1, γ=1.15, and φ is the compound coefficient.
 *
 * Architecture:
 * - Stem: Conv 3×3, stride 2
 * - 7 MBConv stages with different configurations
 * - Head: Conv 1×1 + Global AvgPool + Dropout + FC
 *
 * Shape Transformations:
 * - Input: (N, 3, 224, 224) for B0
 * - After stem: (N, 32, 112, 112)
 * - After blocks: (N, 320, 7, 7)
 * - After head conv: (N, 1280, 7, 7)
 * - After pooling: (N, 1280)
 * - Output: (N, num_classes)
 */
class EfficientNet : public nn::Module {
public:
    /**
     * @brief Construct EfficientNet from configuration
     * @param config EfficientNet configuration
     */
    explicit EfficientNet(const EfficientNetConfig& config);

    /**
     * @brief Forward pass through EfficientNet
     * @param input Input tensor of shape (N, 3, H, W)
     * @return Output logits of shape (N, num_classes)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Load pretrained weights from file
     * @param path Path to pretrained weights
     */
    auto load_pretrained(const std::string& path) -> void;

    /// Enable/disable activation (gradient) checkpointing on the MBConv stages.
    /// Recomputes block activations in backward to cut peak memory (lets the
    /// large variants like B7 train within tight GPU memory); gradients are
    /// unchanged. Off by default.
    auto set_gradient_checkpointing(bool enabled) -> void {
        if (stages_) stages_->set_gradient_checkpointing(enabled);
    }

private:
    /**
     * @brief Build stem layer
     */
    auto make_stem(int64_t stem_channels) -> void;

    /**
     * @brief Build MBConv stages from configuration
     */
    auto make_stages(const std::vector<MBConvConfig>& block_configs,
                     double drop_connect_rate) -> void;

    /**
     * @brief Build head (classifier) layers
     */
    auto make_head(int64_t final_stage_channels, int64_t head_channels,
                   int64_t num_classes, double dropout_rate) -> void;

    // Stem layers
    std::shared_ptr<nn::Conv2d> stem_conv_;
    std::shared_ptr<nn::BatchNorm2d> stem_bn_;
    nn::Swish stem_swish_;

    // MBConv stages
    std::shared_ptr<nn::Sequential> stages_;

    // Head layers
    std::shared_ptr<nn::Conv2d> head_conv_;
    std::shared_ptr<nn::BatchNorm2d> head_bn_;
    nn::Swish head_swish_;
    std::shared_ptr<nn::AdaptiveAvgPool2d> avgpool_;
    std::shared_ptr<nn::Dropout> dropout_;
    std::shared_ptr<nn::Linear> fc_;
};

// ============================================================================
// Factory Functions for EfficientNet Variants (B0-B7)
// ============================================================================

/**
 * @brief Create EfficientNet-B0 model
 *
 * Baseline model with φ=0:
 * - Resolution: 224×224
 * - Depth scale: 1.0
 * - Width scale: 1.0
 * - Dropout: 0.2
 * - Parameters: 5.3M
 * - FLOPs: 0.39B
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to EfficientNet-B0 model
 */
auto efficientnet_b0(int64_t num_classes = 1000, bool pretrained = false) -> std::shared_ptr<EfficientNet>;

/**
 * @brief Create EfficientNet-B1 model
 *
 * φ=0.5:
 * - Resolution: 240×240
 * - Depth scale: 1.1
 * - Width scale: 1.0
 * - Dropout: 0.2
 * - Parameters: 7.8M
 * - FLOPs: 0.70B
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to EfficientNet-B1 model
 */
auto efficientnet_b1(int64_t num_classes = 1000, bool pretrained = false) -> std::shared_ptr<EfficientNet>;

/**
 * @brief Create EfficientNet-B2 model
 *
 * φ=1:
 * - Resolution: 260×260
 * - Depth scale: 1.2
 * - Width scale: 1.1
 * - Dropout: 0.3
 * - Parameters: 9.2M
 * - FLOPs: 1.0B
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to EfficientNet-B2 model
 */
auto efficientnet_b2(int64_t num_classes = 1000, bool pretrained = false) -> std::shared_ptr<EfficientNet>;

/**
 * @brief Create EfficientNet-B3 model
 *
 * φ=2:
 * - Resolution: 300×300
 * - Depth scale: 1.4
 * - Width scale: 1.2
 * - Dropout: 0.3
 * - Parameters: 12M
 * - FLOPs: 1.8B
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to EfficientNet-B3 model
 */
auto efficientnet_b3(int64_t num_classes = 1000, bool pretrained = false) -> std::shared_ptr<EfficientNet>;

/**
 * @brief Create EfficientNet-B4 model
 *
 * φ=3:
 * - Resolution: 380×380
 * - Depth scale: 1.8
 * - Width scale: 1.4
 * - Dropout: 0.4
 * - Parameters: 19M
 * - FLOPs: 4.2B
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to EfficientNet-B4 model
 */
auto efficientnet_b4(int64_t num_classes = 1000, bool pretrained = false) -> std::shared_ptr<EfficientNet>;

/**
 * @brief Create EfficientNet-B5 model
 *
 * φ=4:
 * - Resolution: 456×456
 * - Depth scale: 2.2
 * - Width scale: 1.6
 * - Dropout: 0.4
 * - Parameters: 30M
 * - FLOPs: 9.9B
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to EfficientNet-B5 model
 */
auto efficientnet_b5(int64_t num_classes = 1000, bool pretrained = false) -> std::shared_ptr<EfficientNet>;

/**
 * @brief Create EfficientNet-B6 model
 *
 * φ=5:
 * - Resolution: 528×528
 * - Depth scale: 2.6
 * - Width scale: 1.8
 * - Dropout: 0.5
 * - Parameters: 43M
 * - FLOPs: 19B
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to EfficientNet-B6 model
 */
auto efficientnet_b6(int64_t num_classes = 1000, bool pretrained = false) -> std::shared_ptr<EfficientNet>;

/**
 * @brief Create EfficientNet-B7 model
 *
 * φ=6:
 * - Resolution: 600×600
 * - Depth scale: 3.1
 * - Width scale: 2.0
 * - Dropout: 0.5
 * - Parameters: 66M
 * - FLOPs: 37B
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to EfficientNet-B7 model
 */
auto efficientnet_b7(int64_t num_classes = 1000, bool pretrained = false) -> std::shared_ptr<EfficientNet>;

} // namespace models
} // namespace tenzor
