/**
 * @file unet.cpp
 * @brief Implementation of U-Net architecture for semantic segmentation
 */

#include "tenzor/models/unet.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/nn/layers/segmentation.hpp"  // For nn::upsample_bilinear()
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/vision.hpp"
#include "tenzor/autograd/ops.hpp"  // For gradient-aware cat()
#include <stdexcept>

namespace tenzor {
namespace models {

// ============================================================================
// DoubleConv Implementation
// ============================================================================

DoubleConv::DoubleConv(int64_t in_channels, int64_t out_channels, int64_t mid_channels)
{
    // Use out_channels as mid_channels if not specified
    if (mid_channels == -1) {
        mid_channels = out_channels;
    }

    // First convolution: in_channels -> mid_channels
    // 3x3 kernel, stride 1, padding 1 (same padding)
    conv1_ = std::make_shared<nn::Conv2d>(in_channels, mid_channels, 3, 1, 1);
    bn1_ = std::make_shared<nn::BatchNorm2d>(mid_channels);
    relu1_ = std::make_shared<nn::ReLU>();

    // Second convolution: mid_channels -> out_channels
    conv2_ = std::make_shared<nn::Conv2d>(mid_channels, out_channels, 3, 1, 1);
    bn2_ = std::make_shared<nn::BatchNorm2d>(out_channels);
    relu2_ = std::make_shared<nn::ReLU>();

    // Register submodules for parameter management
    register_module("conv1", conv1_);
    register_module("bn1", bn1_);
    register_module("relu1", relu1_);
    register_module("conv2", conv2_);
    register_module("bn2", bn2_);
    register_module("relu2", relu2_);
}

auto DoubleConv::forward_impl(const Variable& input) -> Variable
{
    // First conv block: Conv -> BN -> ReLU
    auto x = conv1_->forward(input);
    x = bn1_->forward(x);
    x = relu1_->forward(x);

    // Second conv block: Conv -> BN -> ReLU
    x = conv2_->forward(x);
    x = bn2_->forward(x);
    x = relu2_->forward(x);

    return x;
}

// ============================================================================
// Down (Encoder) Implementation
// ============================================================================

Down::Down(int64_t in_channels, int64_t out_channels)
{
    // Max pooling for downsampling: 2x2, stride 2
    pool_ = std::make_shared<nn::MaxPool2d>(2, 2, 0);

    // Double convolution after pooling
    conv_ = std::make_shared<DoubleConv>(in_channels, out_channels);

    register_module("pool", pool_);
    register_module("conv", conv_);
}

auto Down::forward_impl(const Variable& input) -> Variable
{
    // Downsample then convolve
    auto x = pool_->forward(input);
    x = conv_->forward(x);
    return x;
}

// ============================================================================
// Up (Decoder) Implementation
// ============================================================================

Up::Up(int64_t in_channels, int64_t out_channels, bool bilinear)
    : bilinear_(bilinear)
{
    if (bilinear) {
        // Bilinear upsampling + 1x1 conv to reduce channels
        // We'll implement bilinear upsampling in forward pass using interpolate
        // For now, use a 1x1 conv to adjust channels before concatenation
        up_ = std::make_shared<nn::Conv2d>(in_channels, in_channels / 2, 1, 1, 0);
        conv_ = std::make_shared<DoubleConv>(in_channels, out_channels);
    } else {
        // Learned upsampling via transposed convolution
        // kernel=2, stride=2 doubles spatial dimensions
        up_ = std::make_shared<nn::ConvTranspose2d>(in_channels, in_channels / 2, 2, 2, 0);
        conv_ = std::make_shared<DoubleConv>(in_channels, out_channels);
    }

    register_module("up", up_);
    register_module("conv", conv_);
}

auto Up::forward(const Variable& input, const Variable& skip) -> Variable
{
    Variable x;

    if (bilinear_) {
        // Get target size from skip connection
        auto skip_shape = skip.tensor().shape();
        int64_t target_h = skip_shape[2];
        int64_t target_w = skip_shape[3];

        // Upsample using bilinear interpolation with proper autograd support
        x = nn::upsample_bilinear(input, target_h, target_w);

        // Apply 1x1 conv to reduce channels
        x = up_->forward(x);
    } else {
        // Use transposed convolution for learned upsampling
        x = up_->forward(input);
    }

    // Concatenate with skip connection along channel dimension (dim=1)
    // Input: x [N, C1, H, W], skip [N, C2, H, W]
    // Output: [N, C1+C2, H, W]
    // Use gradient-aware cat() to preserve autograd graph
    std::vector<Variable> vars = {x, skip};
    auto concat_var = tenzor::cat(vars, 1);  // Uses CatBackward for gradients

    // Apply double convolution
    auto output = conv_->forward(concat_var);

    return output;
}

// ============================================================================
// UNet Implementation
// ============================================================================

UNet::UNet(int64_t in_channels, int64_t num_classes, bool bilinear)
    : in_channels_(in_channels)
    , num_classes_(num_classes)
    , bilinear_(bilinear)
{
    // Bottleneck factor is always 2 (1024 max channels)
    // This ensures correct channel dimensions in the decoder path
    // regardless of upsampling method (bilinear or transposed conv)
    bottleneck_factor_ = 2;

    // Initial convolution (no downsampling)
    // in_channels -> 64
    inc_ = std::make_shared<DoubleConv>(in_channels, 64);

    // Encoder path (downsampling)
    down1_ = std::make_shared<Down>(64, 128);        // 64 -> 128
    down2_ = std::make_shared<Down>(128, 256);       // 128 -> 256
    down3_ = std::make_shared<Down>(256, 512);       // 256 -> 512
    down4_ = std::make_shared<Down>(512, 512 * bottleneck_factor_);  // 512 -> 1024 (or 512)

    // Decoder path (upsampling)
    // Up(in_channels_from_previous_layer, output_channels, bilinear)
    up1_ = std::make_shared<Up>(512 * bottleneck_factor_, 512, bilinear);  // 1024 -> 512 (or 512 -> 512)
    up2_ = std::make_shared<Up>(512, 256, bilinear);                         // 512 -> 256
    up3_ = std::make_shared<Up>(256, 128, bilinear);                         // 256 -> 128
    up4_ = std::make_shared<Up>(128, 64, bilinear);                          // 128 -> 64

    // Output convolution: 64 -> num_classes
    // 1x1 convolution for pixel-wise classification
    outc_ = std::make_shared<nn::Conv2d>(64, num_classes, 1, 1, 0);

    // Register all submodules
    register_module("inc", inc_);
    register_module("down1", down1_);
    register_module("down2", down2_);
    register_module("down3", down3_);
    register_module("down4", down4_);
    register_module("up1", up1_);
    register_module("up2", up2_);
    register_module("up3", up3_);
    register_module("up4", up4_);
    register_module("outc", outc_);
}

auto UNet::forward_impl(const Variable& input) -> Variable
{
    // Validate input shape
    auto input_shape = input.tensor().shape();
    if (input_shape.size() != 4) {
        throw std::runtime_error(
            "UNet expects 4D input [N, C, H, W], got " +
            std::to_string(input_shape.size()) + "D"
        );
    }
    if (input_shape[1] != in_channels_) {
        throw std::runtime_error(
            "UNet expects " + std::to_string(in_channels_) + " input channels, got " +
            std::to_string(input_shape[1])
        );
    }

    // ========================================================================
    // Encoder path with skip connections
    // ========================================================================

    // Initial convolution (no downsampling)
    auto x1 = inc_->forward(input);  // [N, 64, H, W]

    // Encoder level 1
    auto x2 = down1_->forward(x1);   // [N, 128, H/2, W/2]

    // Encoder level 2
    auto x3 = down2_->forward(x2);   // [N, 256, H/4, W/4]

    // Encoder level 3
    auto x4 = down3_->forward(x3);   // [N, 512, H/8, W/8]

    // Encoder level 4 / Bottleneck
    auto x5 = down4_->forward(x4);   // [N, 1024, H/16, W/16] or [N, 512, H/16, W/16]

    // ========================================================================
    // Decoder path with skip connections
    // ========================================================================

    // Decoder level 1: upsample and concat with x4
    auto x = up1_->forward(x5, x4);  // [N, 512, H/8, W/8] or [N, 256, H/8, W/8]

    // Decoder level 2: upsample and concat with x3
    x = up2_->forward(x, x3);        // [N, 256, H/4, W/4] or [N, 128, H/4, W/4]

    // Decoder level 3: upsample and concat with x2
    x = up3_->forward(x, x2);        // [N, 128, H/2, W/2] or [N, 64, H/2, W/2]

    // Decoder level 4: upsample and concat with x1
    x = up4_->forward(x, x1);        // [N, 64, H, W]

    // ========================================================================
    // Output head
    // ========================================================================

    // Final 1x1 convolution for pixel-wise classification
    auto output = outc_->forward(x); // [N, num_classes, H, W]

    return output;
}

} // namespace models
} // namespace tenzor
