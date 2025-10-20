/**
 * @file deeplabv3plus.hpp
 * @brief DeepLab v3+ - State-of-the-art semantic segmentation
 *
 * Implements DeepLab v3+, an encoder-decoder architecture for semantic
 * segmentation using atrous (dilated) convolutions and Atrous Spatial
 * Pyramid Pooling (ASPP) for multi-scale feature extraction.
 *
 * Key Features:
 * - Encoder: ResNet/MobileNet backbone with atrous convolutions
 * - ASPP: Multi-scale context aggregation with 5 parallel branches
 * - Decoder: Lightweight decoder with skip connections from low-level features
 * - Atrous Separable Convolution: Efficient multi-scale feature extraction
 *
 * Architecture:
 * ```
 * Input (H×W×3)
 *   ↓
 * Encoder (ResNet/MobileNet with ASPP)
 *   ├─→ Low-level features (H/4 × W/4)
 *   └─→ High-level features + ASPP (H/16 × W/16)
 *         ↓ 4× Upsample
 * Decoder:
 *   Concat(Low-level, Upsampled ASPP)
 *   ↓ 3×3 Conv
 *   ↓ 4× Upsample
 * Output: (H×W×num_classes)
 * ```
 *
 * Reference: "Encoder-Decoder with Atrous Separable Convolution for Semantic
 * Image Segmentation" (Chen et al., ECCV 2018)
 */

#pragma once

#include <memory>
#include <vector>
#include <string>
#include "../nn/module.hpp"
#include "../nn/layers/conv.hpp"
#include "../nn/layers/batchnorm.hpp"
#include "../nn/layers/pooling.hpp"
#include "../nn/layers/segmentation.hpp"
#include "../nn/activations/activations.hpp"
#include "resnet.hpp"
#include "mobilenet.hpp"

namespace tenzor {
namespace models {

/**
 * @brief DeepLab v3+ Encoder.
 *
 * Extracts multi-scale features using a backbone network (ResNet or MobileNet)
 * combined with ASPP for capturing context at different scales.
 *
 * Outputs:
 * 1. Low-level features from early layers (1/4 resolution, for skip connection)
 * 2. High-level ASPP features (1/16 resolution, encoded context)
 *
 * The encoder uses atrous convolutions in the backbone's later layers to
 * maintain larger feature maps (output_stride=16 instead of 32).
 */
class DeepLabV3PlusEncoder : public nn::Module {
public:
    /**
     * @brief Construct encoder with ResNet backbone.
     *
     * @param backbone_name Backbone architecture ("resnet50", "resnet101")
     * @param output_stride Output stride of backbone (8 or 16, default: 16)
     * @param pretrained Use pretrained ImageNet weights (default: false)
     *
     * @note output_stride=16 is standard, output_stride=8 uses more computation
     *       but captures finer details
     */
    DeepLabV3PlusEncoder(const std::string& backbone_name = "resnet50",
                         int64_t output_stride = 16,
                         bool pretrained = false);

    /**
     * @brief Forward pass through encoder.
     *
     * @param input Input image of shape (N, 3, H, W)
     * @return Pair of (ASPP features, low-level features)
     *         - ASPP features: (N, 256, H/16, W/16)
     *         - Low-level features: (N, C_low, H/4, W/4)
     */
    auto forward_impl(const Variable& input)
        -> std::pair<Variable, Variable>;

    // Module interface implementation - returns ASPP features only
    auto forward(const Variable& input) -> Variable override {
        auto [aspp_feat, low_feat] = forward_impl(input);
        return aspp_feat;
    }

    /**
     * @brief Get number of low-level feature channels.
     *
     * This is needed by the decoder to match channel dimensions.
     *
     * @return Number of channels in low-level features
     */
    auto get_low_level_channels() const -> int64_t {
        return low_level_channels_;
    }

private:
    std::shared_ptr<nn::Module> backbone_;  ///< ResNet or MobileNet backbone
    std::shared_ptr<nn::ASPP> aspp_;        ///< ASPP module
    std::shared_ptr<nn::Conv2d> feature_proj_;  ///< Project backbone features to low-level channels
    std::string backbone_name_;             ///< Name of backbone
    int64_t output_stride_;                 ///< Output stride (8 or 16)
    int64_t low_level_channels_;            ///< Channels in low-level features
    int64_t high_level_channels_;           ///< Channels before ASPP

    /**
     * @brief Create ResNet backbone for segmentation.
     *
     * Modifies standard ResNet to use atrous convolutions in later layers.
     */
    auto create_resnet_backbone(const std::string& name, bool pretrained)
        -> std::shared_ptr<nn::Module>;
};

/**
 * @brief DeepLab v3+ Decoder.
 *
 * Lightweight decoder that refines the coarse ASPP features by combining
 * them with low-level features from the encoder via skip connections.
 *
 * Architecture:
 * 1. Reduce low-level feature channels (256 -> 48)
 * 2. Upsample ASPP features 4×
 * 3. Concatenate with reduced low-level features
 * 4. Refine with 3×3 convolutions
 * 5. Upsample 4× to original resolution
 * 6. Apply 1×1 conv for class prediction
 */
class DeepLabV3PlusDecoder : public nn::Module {
public:
    /**
     * @brief Construct decoder.
     *
     * @param num_classes Number of segmentation classes
     * @param low_level_channels Number of channels in low-level features (from encoder)
     * @param aspp_channels Number of ASPP output channels (typically 256)
     *
     * @code
     * DeepLabV3PlusDecoder decoder(21, 256, 256);  // PASCAL VOC 21 classes
     * @endcode
     */
    DeepLabV3PlusDecoder(int64_t num_classes,
                         int64_t low_level_channels,
                         int64_t aspp_channels = 256);

    /**
     * @brief Forward pass through decoder.
     *
     * @param aspp_features High-level ASPP features (N, 256, H/16, W/16)
     * @param low_level_features Low-level features (N, C_low, H/4, W/4)
     * @return Segmentation logits of shape (N, num_classes, H, W)
     *
     * @note Output resolution matches low_level_features * 4
     */
    auto forward(const Variable& aspp_features,
                const Variable& low_level_features) -> Variable;

    // Module interface implementation (not used - decoder requires 2 inputs)
    auto forward(const Variable& input) -> Variable override {
        // This shouldn't be called - decoder requires both aspp and low-level features
        // Return input as dummy implementation
        return input;
    }

private:
    // Reduce low-level feature channels
    std::shared_ptr<nn::Module> low_level_reduce_;

    // Refinement convolutions after concatenation
    std::shared_ptr<nn::Module> refine_;

    // Final classifier
    std::shared_ptr<nn::Conv2d> classifier_;

    int64_t num_classes_;  ///< Number of segmentation classes
};

/**
 * @brief DeepLab v3+ complete model.
 *
 * Full encoder-decoder architecture for semantic segmentation. Combines
 * a powerful encoder (ResNet/MobileNet + ASPP) with a lightweight decoder.
 *
 * Features:
 * - Multi-scale feature extraction via ASPP
 * - Atrous convolutions for large receptive field
 * - Skip connections from encoder to decoder
 * - Efficient separable convolutions
 *
 * Typical usage:
 * ```cpp
 * // Create model for PASCAL VOC (21 classes)
 * auto model = DeepLabV3Plus_ResNet50(21);
 *
 * // Forward pass
 * Variable input(Tensor({1, 3, 512, 512}, DType::Float32, Device::cpu()), true);
 * Variable output = model->forward(input);  // Shape: {1, 21, 512, 512}
 *
 * // Apply softmax and argmax for segmentation map
 * auto probs = softmax(output, 1);
 * auto seg_map = argmax(probs, 1);  // Shape: {1, 512, 512}
 * ```
 */
class DeepLabV3Plus : public nn::Module {
public:
    /**
     * @brief Construct DeepLab v3+ model.
     *
     * @param num_classes Number of segmentation classes
     * @param backbone Backbone architecture ("resnet50", "resnet101", "mobilenetv2")
     * @param output_stride Output stride (8 or 16, default: 16)
     * @param pretrained Use pretrained backbone weights (default: false)
     *
     * @code
     * DeepLabV3Plus model(21, "resnet50", 16, false);
     * @endcode
     */
    DeepLabV3Plus(int64_t num_classes,
                  const std::string& backbone = "resnet50",
                  int64_t output_stride = 16,
                  bool pretrained = false);

    /**
     * @brief Forward pass through DeepLab v3+.
     *
     * @param input Input image of shape (N, 3, H, W)
     * @return Segmentation logits of shape (N, num_classes, H, W)
     *
     * @note Apply softmax for probabilities, argmax for class predictions
     */
    auto forward(const Variable& input) -> Variable override;

    /**
     * @brief Predict segmentation map.
     *
     * Convenience method that applies softmax and argmax to get final
     * segmentation map with class indices.
     *
     * @param input Input image of shape (N, 3, H, W)
     * @return Segmentation map of shape (N, H, W) with class indices
     */
    auto predict(const Variable& input) -> Tensor;

    /**
     * @brief Load pretrained weights.
     *
     * @param path Path to pretrained weights file
     * @throws std::runtime_error if file doesn't exist or format is invalid
     */
    auto load_pretrained(const std::string& path) -> void;

private:
    std::shared_ptr<DeepLabV3PlusEncoder> encoder_;  ///< Encoder network
    std::shared_ptr<DeepLabV3PlusDecoder> decoder_;  ///< Decoder network
    int64_t num_classes_;                            ///< Number of classes
    std::string backbone_;                           ///< Backbone name
};

// ============================================================================
// Factory Functions for DeepLab v3+ Variants
// ============================================================================

/**
 * @brief Create DeepLab v3+ with ResNet-50 backbone.
 *
 * Standard configuration for semantic segmentation tasks.
 *
 * Parameters: ~39.8M
 * Expected mIoU (PASCAL VOC): ~78.5%
 * Expected mIoU (Cityscapes): ~78.8%
 *
 * @param num_classes Number of segmentation classes
 * @param output_stride Output stride (8 or 16, default: 16)
 * @param pretrained Use pretrained ResNet-50 backbone (default: false)
 * @return Shared pointer to DeepLab v3+ model
 *
 * @code
 * auto model = models::DeepLabV3Plus_ResNet50(21);  // PASCAL VOC
 * auto output = model->forward(input);
 * @endcode
 */
auto DeepLabV3Plus_ResNet50(int64_t num_classes,
                            int64_t output_stride = 16,
                            bool pretrained = false)
    -> std::shared_ptr<DeepLabV3Plus>;

/**
 * @brief Create DeepLab v3+ with ResNet-101 backbone.
 *
 * More powerful variant for higher accuracy at cost of computation.
 *
 * Parameters: ~58.8M
 * Expected mIoU (PASCAL VOC): ~79.3%
 * Expected mIoU (Cityscapes): ~80.2%
 *
 * @param num_classes Number of segmentation classes
 * @param output_stride Output stride (8 or 16, default: 16)
 * @param pretrained Use pretrained ResNet-101 backbone (default: false)
 * @return Shared pointer to DeepLab v3+ model
 */
auto DeepLabV3Plus_ResNet101(int64_t num_classes,
                             int64_t output_stride = 16,
                             bool pretrained = false)
    -> std::shared_ptr<DeepLabV3Plus>;

/**
 * @brief Create DeepLab v3+ with MobileNetV2 backbone.
 *
 * Lightweight variant for mobile and edge devices.
 *
 * Parameters: ~5.8M
 * Expected mIoU (PASCAL VOC): ~70.7%
 * Expected mIoU (Cityscapes): ~72.4%
 * Inference speed: ~3-5x faster than ResNet-50
 *
 * @param num_classes Number of segmentation classes
 * @param output_stride Output stride (8 or 16, default: 16)
 * @param pretrained Use pretrained MobileNetV2 backbone (default: false)
 * @return Shared pointer to DeepLab v3+ model
 *
 * @note Recommended for real-time applications and resource-constrained devices
 */
auto DeepLabV3Plus_MobileNetV2(int64_t num_classes,
                               int64_t output_stride = 16,
                               bool pretrained = false)
    -> std::shared_ptr<DeepLabV3Plus>;

} // namespace models
} // namespace tenzor
