/**
 * @file segmentation.hpp
 * @brief Specialized layers for semantic segmentation
 *
 * Implements segmentation-specific layers including Atrous Spatial Pyramid
 * Pooling (ASPP), Atrous Separable Convolution, and other components for
 * semantic segmentation models like DeepLab v3+.
 */

#pragma once

#include <memory>
#include <vector>
#include "../module.hpp"
#include "conv.hpp"
#include "batchnorm.hpp"
#include "pooling.hpp"
#include "dropout.hpp"
#include "../activations/activations.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Atrous (Dilated) Separable Convolution.
 *
 * Combines depthwise separable convolution with atrous/dilated convolution
 * for efficient multi-scale feature extraction. Used in DeepLab models to
 * increase receptive field without increasing parameters.
 *
 * Architecture:
 * ```
 * Input -> Depthwise Atrous Conv -> BN -> ReLU -> Pointwise Conv -> BN -> ReLU
 * ```
 *
 * Benefits:
 * - Reduced computational cost vs standard atrous convolution
 * - Larger receptive field with dilation
 * - Maintains spatial resolution
 *
 * @code
 * AtrousSeparableConv2d asconv(256, 256, 3, 6);  // dilation=6
 * Variable x(Tensor({batch, 256, 32, 32}, DType::Float32, Device::cpu()), true);
 * Variable out = asconv.forward(x);  // Same spatial size
 * @endcode
 */
class AtrousSeparableConv2d : public Module {
public:
    /**
     * @brief Construct atrous separable convolution.
     *
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param kernel_size Kernel size (typically 3)
     * @param dilation Dilation rate for atrous convolution
     * @param bias Whether to include bias (default: false, BN handles it)
     *
     * @note Padding is automatically calculated to maintain spatial dimensions
     */
    AtrousSeparableConv2d(int64_t in_channels,
                          int64_t out_channels,
                          int64_t kernel_size = 3,
                          int64_t dilation = 1,
                          bool bias = false);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::shared_ptr<Conv2d> depthwise_;  ///< Depthwise atrous convolution
    std::shared_ptr<BatchNorm2d> bn1_;   ///< Batch norm after depthwise
    std::shared_ptr<Conv2d> pointwise_;  ///< Pointwise 1x1 convolution
    std::shared_ptr<BatchNorm2d> bn2_;   ///< Batch norm after pointwise
    ReLU relu_;                          ///< ReLU activation
};

/**
 * @brief Atrous Spatial Pyramid Pooling (ASPP).
 *
 * Multi-scale feature extraction module using parallel atrous convolutions
 * with different dilation rates. Core component of DeepLab v3+.
 *
 * Architecture:
 * ```
 * Input (H×W×C)
 *   ├─→ 1×1 conv (rate=1)           -> 256 channels
 *   ├─→ 3×3 atrous conv (rate=6)    -> 256 channels
 *   ├─→ 3×3 atrous conv (rate=12)   -> 256 channels
 *   ├─→ 3×3 atrous conv (rate=18)   -> 256 channels
 *   └─→ Global Avg Pool -> 1×1 conv -> Upsample -> 256 channels
 *         ↓
 *   Concatenate (1280 channels)
 *         ↓
 *   1×1 conv + BN + ReLU + Dropout -> 256 channels
 * ```
 *
 * The five parallel branches capture features at different scales:
 * - Branch 1: Point-wise features (rate=1)
 * - Branches 2-4: Multi-scale context (rates=6,12,18)
 * - Branch 5: Global context via pooling
 *
 * @code
 * ASPP aspp(2048, 256);  // ResNet output -> 256 channels
 * Variable features(Tensor({batch, 2048, 32, 32}, DType::Float32, Device::cpu()), true);
 * Variable aspp_out = aspp.forward(features);  // Shape: {batch, 256, 32, 32}
 * @endcode
 *
 * Reference: "Encoder-Decoder with Atrous Separable Convolution for Semantic
 * Image Segmentation" (Chen et al., ECCV 2018)
 */
class ASPP : public Module {
public:
    /**
     * @brief Construct ASPP module.
     *
     * @param in_channels Number of input channels (e.g., 2048 for ResNet)
     * @param out_channels Number of output channels (typically 256)
     * @param atrous_rates Dilation rates for atrous convolutions (default: {6,12,18})
     * @param use_separable Use separable convolutions for efficiency (default: true)
     * @param dropout_rate Dropout rate after fusion (default: 0.5)
     *
     * @code
     * ASPP aspp1(2048, 256);  // Default rates {6, 12, 18}
     * ASPP aspp2(2048, 256, {12, 24, 36});  // Custom rates for output_stride=8
     * @endcode
     */
    ASPP(int64_t in_channels,
         int64_t out_channels = 256,
         std::vector<int64_t> atrous_rates = {6, 12, 18},
         bool use_separable = true,
         float dropout_rate = 0.5);

    /**
     * @brief Forward pass through ASPP.
     *
     * Applies five parallel branches and fuses them via concatenation
     * followed by 1x1 convolution.
     *
     * @param input Input features of shape (N, in_channels, H, W)
     * @return ASPP output of shape (N, out_channels, H, W)
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    // Branch 1: 1×1 convolution
    std::shared_ptr<Module> conv1x1_;

    // Branches 2-4: Atrous convolutions (3 branches)
    std::vector<std::shared_ptr<Module>> atrous_convs_;

    // Branch 5: Global pooling branch
    std::shared_ptr<AdaptiveAvgPool2d> global_pool_;
    std::shared_ptr<Module> global_conv_;

    // Fusion: Project concatenated features
    std::shared_ptr<Module> project_;

    int64_t out_channels_;  ///< Output channel dimension
};

/**
 * @brief Bilinear upsampling helper function.
 *
 * Performs bilinear interpolation to upsample feature maps to target size.
 * This is a temporary implementation until a full interpolate operation
 * is added to Tenzor core.
 *
 * @param input Input tensor of shape (N, C, H_in, W_in)
 * @param target_h Target height
 * @param target_w Target width
 * @return Upsampled tensor of shape (N, C, target_h, target_w)
 *
 * @note This is implemented using nearest neighbor for now. Full bilinear
 *       interpolation should be implemented in core ops.
 */
auto upsample_bilinear(const Variable& input, int64_t target_h, int64_t target_w)
    -> Variable;

/**
 * @brief Create a sequential block with Conv2d + BatchNorm2d + ReLU.
 *
 * Common pattern in segmentation networks. Creates a composite module.
 *
 * @param in_channels Number of input channels
 * @param out_channels Number of output channels
 * @param kernel_size Kernel size
 * @param stride Stride (default: 1)
 * @param padding Padding (default: 0)
 * @param dilation Dilation (default: 1)
 * @param groups Groups (default: 1)
 * @return Shared pointer to sequential module
 */
auto make_conv_bn_relu(int64_t in_channels,
                       int64_t out_channels,
                       int64_t kernel_size,
                       int64_t stride = 1,
                       int64_t padding = 0,
                       int64_t dilation = 1,
                       int64_t groups = 1)
    -> std::shared_ptr<Module>;

} // namespace nn
} // namespace tenzor
