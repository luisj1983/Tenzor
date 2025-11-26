/**
 * @file unet.hpp
 * @brief U-Net architecture for semantic segmentation
 *
 * Implements the U-Net encoder-decoder architecture with skip connections
 * for biomedical image segmentation and general semantic segmentation tasks.
 *
 * Reference: Ronneberger et al. "U-Net: Convolutional Networks for Biomedical
 * Image Segmentation" (MICCAI 2015)
 */

#pragma once

#include "../nn/module.hpp"
#include "../nn/layers/conv.hpp"
#include "../nn/layers/batchnorm.hpp"
#include "../nn/layers/pooling.hpp"
#include "../nn/activations/activations.hpp"
#include <memory>
#include <vector>

namespace tenzor {
namespace models {

/**
 * @brief Double convolution block - fundamental building block of U-Net
 *
 * Applies two consecutive 3x3 convolutions with batch normalization and ReLU:
 * Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU
 *
 * This pattern is used in both encoder and decoder paths.
 *
 * Shape:
 * - Input: (N, in_channels, H, W)
 * - Output: (N, out_channels, H, W)
 *
 * @code
 * DoubleConv block(64, 128);  // 64 -> 128 channels
 * Variable x(Tensor({batch, 64, 32, 32}, DType::Float32, Device::cpu()), true);
 * Variable out = block.forward(x);  // Shape: {batch, 128, 32, 32}
 * @endcode
 */
class DoubleConv : public nn::Module {
public:
    /**
     * @brief Construct double convolution block
     *
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param mid_channels Number of middle channels (default: out_channels)
     *
     * @code
     * DoubleConv conv1(3, 64);      // RGB -> 64 features
     * DoubleConv conv2(64, 128);    // 64 -> 128 features
     * DoubleConv conv3(128, 128, 96); // Custom mid channels
     * @endcode
     */
    DoubleConv(int64_t in_channels, int64_t out_channels, int64_t mid_channels = -1);

    /**
     * @brief Forward pass through double convolution
     *
     * @param input Input variable of shape (N, in_channels, H, W)
     * @return Output variable of shape (N, out_channels, H, W)
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::shared_ptr<nn::Conv2d> conv1_;
    std::shared_ptr<nn::BatchNorm2d> bn1_;
    std::shared_ptr<nn::Module> relu1_;
    std::shared_ptr<nn::Conv2d> conv2_;
    std::shared_ptr<nn::BatchNorm2d> bn2_;
    std::shared_ptr<nn::Module> relu2_;
};

/**
 * @brief Encoder (downsampling) block
 *
 * Applies max pooling followed by double convolution for downsampling:
 * MaxPool2d -> DoubleConv
 *
 * Reduces spatial dimensions by factor of 2.
 *
 * Shape:
 * - Input: (N, in_channels, H, W)
 * - Output: (N, out_channels, H/2, W/2)
 *
 * @code
 * Down block(64, 128);
 * Variable x(Tensor({batch, 64, 32, 32}, DType::Float32, Device::cpu()), true);
 * Variable out = block.forward(x);  // Shape: {batch, 128, 16, 16}
 * @endcode
 */
class Down : public nn::Module {
public:
    /**
     * @brief Construct encoder block
     *
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     */
    Down(int64_t in_channels, int64_t out_channels);

    /**
     * @brief Forward pass through downsampling
     *
     * @param input Input variable of shape (N, in_channels, H, W)
     * @return Output variable of shape (N, out_channels, H/2, W/2)
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::shared_ptr<nn::MaxPool2d> pool_;
    std::shared_ptr<DoubleConv> conv_;
};

/**
 * @brief Decoder (upsampling) block with skip connections
 *
 * Applies upsampling, concatenates with skip connection, then double convolution:
 * - Option 1 (bilinear=true): Bilinear upsample + Conv2d
 * - Option 2 (bilinear=false): ConvTranspose2d for learned upsampling
 * Concatenate with skip -> DoubleConv
 *
 * Increases spatial dimensions by factor of 2.
 *
 * Shape:
 * - Input: (N, in_channels, H, W)
 * - Skip: (N, skip_channels, 2H, 2W)
 * - Output: (N, out_channels, 2H, 2W)
 *
 * @note Skip connection doubles the channels before DoubleConv
 *
 * @code
 * Up block(256, 128, false);  // Transposed conv upsampling
 * Variable x(Tensor({batch, 256, 16, 16}, DType::Float32, Device::cpu()), true);
 * Variable skip(Tensor({batch, 128, 32, 32}, DType::Float32, Device::cpu()), true);
 * Variable out = block.forward(x, skip);  // Shape: {batch, 128, 32, 32}
 * @endcode
 */
class Up : public nn::Module {
public:
    /**
     * @brief Construct decoder block
     *
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param bilinear If true, use bilinear upsampling; else use ConvTranspose2d
     *
     * @code
     * Up up1(512, 256, false);  // Learned upsampling
     * Up up2(256, 128, true);   // Bilinear upsampling
     * @endcode
     */
    Up(int64_t in_channels, int64_t out_channels, bool bilinear);

    /**
     * @brief Forward pass with skip connection
     *
     * @param input Input variable to upsample
     * @param skip Skip connection from encoder path
     * @return Output variable after upsampling and fusion
     */
    auto forward(const Variable& input, const Variable& skip) -> Variable;

    // Override base forward for Module interface
    auto forward_impl(const Variable& input) -> Variable override {
        throw std::runtime_error("Up layer requires skip connection. Use forward(input, skip)");
    }

private:
    bool bilinear_;
    std::shared_ptr<nn::Module> up_;  // Either ConvTranspose2d or Conv2d for bilinear
    std::shared_ptr<DoubleConv> conv_;
};

/**
 * @brief Complete U-Net model for semantic segmentation
 *
 * Encoder-decoder architecture with skip connections:
 *
 * Architecture (default):
 * ```
 * Input: [N, in_channels, H, W]
 *     ↓
 * Enc1: DoubleConv -> [N, 64, H, W] ----→ Skip1
 *     ↓ Down (MaxPool)
 * Enc2: [N, 128, H/2, W/2] ----→ Skip2
 *     ↓ Down
 * Enc3: [N, 256, H/4, W/4] ----→ Skip3
 *     ↓ Down
 * Enc4: [N, 512, H/8, W/8] ----→ Skip4
 *     ↓ Down
 * Bottleneck: [N, 1024, H/16, W/16] (bilinear=false) or [N, 512, H/16, W/16] (bilinear=true)
 *     ↑ Up + Concat(Skip4)
 * Dec1: [N, 512, H/8, W/8]
 *     ↑ Up + Concat(Skip3)
 * Dec2: [N, 256, H/4, W/4]
 *     ↑ Up + Concat(Skip2)
 * Dec3: [N, 128, H/2, W/2]
 *     ↑ Up + Concat(Skip1)
 * Dec4: [N, 64, H, W]
 *     ↓ 1×1 Conv
 * Output: [N, num_classes, H, W]
 * ```
 *
 * **Applications:**
 * - Medical image segmentation (original use case)
 * - Semantic segmentation (Pascal VOC, Cityscapes)
 * - Binary segmentation tasks
 * - Multi-class segmentation
 *
 * **Input/Output:**
 * - Input: RGB or grayscale images [N, in_channels, H, W]
 * - Output: Logits [N, num_classes, H, W] (apply sigmoid/softmax for probabilities)
 *
 * **Loss Functions:**
 * - Binary: BCEWithLogitsLoss + DiceLoss
 * - Multi-class: CrossEntropyLoss
 *
 * @code
 * // Binary segmentation (e.g., tumor detection)
 * UNet model(1, 1, false);  // Grayscale input, 1 output class
 *
 * // Multi-class segmentation (e.g., 21 classes like Pascal VOC)
 * UNet model2(3, 21, true);  // RGB input, 21 classes, bilinear upsampling
 *
 * // Forward pass
 * Variable input(Tensor({batch, 3, 256, 256}, DType::Float32, Device::cpu()), true);
 * Variable output = model2.forward(input);  // Shape: {batch, 21, 256, 256}
 * @endcode
 */
class UNet : public nn::Module {
public:
    /**
     * @brief Construct U-Net model
     *
     * @param in_channels Number of input channels (1 for grayscale, 3 for RGB)
     * @param num_classes Number of output classes
     * @param bilinear If true, use bilinear upsampling; else use ConvTranspose2d
     *
     * @note bilinear=true uses less memory and may train faster, but
     *       bilinear=false (learned upsampling) may give better results
     *
     * @code
     * UNet unet1(1, 2, false);   // Medical imaging, 2 classes
     * UNet unet2(3, 21, true);   // Pascal VOC, 21 classes, bilinear
     * UNet unet3(3, 1, false);   // RGB to binary mask
     * @endcode
     */
    UNet(int64_t in_channels, int64_t num_classes, bool bilinear = false);

    /**
     * @brief Forward pass through U-Net
     *
     * @param input Input variable of shape (N, in_channels, H, W)
     * @return Output logits of shape (N, num_classes, H, W)
     *
     * @note Apply sigmoid (binary) or softmax (multi-class) to get probabilities
     *
     * @throws std::runtime_error if input dimensions are invalid
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Get number of input channels
     * @return Number of input channels
     */
    auto get_in_channels() const -> int64_t { return in_channels_; }

    /**
     * @brief Get number of output classes
     * @return Number of output classes
     */
    auto get_num_classes() const -> int64_t { return num_classes_; }

    /**
     * @brief Check if using bilinear upsampling
     * @return true if using bilinear upsampling
     */
    auto is_bilinear() const -> bool { return bilinear_; }

private:
    int64_t in_channels_;   ///< Number of input channels
    int64_t num_classes_;   ///< Number of output classes
    bool bilinear_;         ///< Use bilinear upsampling vs learned upsampling

    // Encoder path
    std::shared_ptr<DoubleConv> inc_;   ///< Initial convolution
    std::shared_ptr<Down> down1_;       ///< Encoder level 1
    std::shared_ptr<Down> down2_;       ///< Encoder level 2
    std::shared_ptr<Down> down3_;       ///< Encoder level 3
    std::shared_ptr<Down> down4_;       ///< Encoder level 4 / Bottleneck entry

    // Bottleneck
    int64_t bottleneck_factor_;         ///< Channel multiplier (1 for bilinear, 2 for learned)

    // Decoder path
    std::shared_ptr<Up> up1_;           ///< Decoder level 1
    std::shared_ptr<Up> up2_;           ///< Decoder level 2
    std::shared_ptr<Up> up3_;           ///< Decoder level 3
    std::shared_ptr<Up> up4_;           ///< Decoder level 4

    // Output head
    std::shared_ptr<nn::Conv2d> outc_;  ///< Final 1x1 convolution
};

} // namespace models
} // namespace tenzor
