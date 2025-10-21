/**
 * @file mobilenet.hpp
 * @brief MobileNet V2 and V3 - Efficient mobile architectures
 *
 * Implements the MobileNet family of efficient convolutional neural networks
 * designed for mobile and embedded vision applications. These models use
 * depthwise separable convolutions and inverted residual structures.
 *
 * MobileNetV2:
 * - Inverted residual blocks with linear bottlenecks
 * - Expansion factor of 6
 * - ReLU6 activation
 * - Width multiplier support
 *
 * MobileNetV3:
 * - Neural architecture search (NAS) optimized
 * - Hard-Swish activation for better efficiency
 * - Squeeze-and-Excitation (SE) modules
 * - Redesigned expensive layers
 *
 * References:
 * - MobileNetV2: "Inverted Residuals and Linear Bottlenecks" (Sandler et al., CVPR 2018)
 * - MobileNetV3: "Searching for MobileNetV3" (Howard et al., ICCV 2019)
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
#include "../nn/activations/activations.hpp"

namespace tenzor {
namespace models {

/**
 * @brief Hard-Swish activation function.
 *
 * Hard-Swish is a computationally efficient approximation of Swish:
 *
 * h-swish(x) = x * ReLU6(x + 3) / 6
 *
 * Properties:
 * - Avoids expensive sigmoid computation
 * - Quantization-friendly (piecewise linear)
 * - Nearly identical behavior to Swish
 * - Used in deeper layers of MobileNetV3
 *
 * Formula:
 * ```
 * h-swish(x) = {
 *     0,         if x ≤ -3
 *     x,         if x ≥ +3
 *     x(x+3)/6,  otherwise
 * }
 * ```
 */
class HardSwish : public nn::Module {
public:
    HardSwish() = default;
    auto forward(const Variable& input) -> Variable override;
};

/**
 * @brief Hard-Sigmoid activation function.
 *
 * Hard-Sigmoid is used in SE modules for MobileNetV3:
 *
 * h-sigmoid(x) = ReLU6(x + 3) / 6
 *
 * Formula:
 * ```
 * h-sigmoid(x) = {
 *     0,       if x ≤ -3
 *     1,       if x ≥ +3
 *     (x+3)/6, otherwise
 * }
 * ```
 */
class HardSigmoid : public nn::Module {
public:
    HardSigmoid() = default;
    auto forward(const Variable& input) -> Variable override;
};

/**
 * @brief Squeeze-and-Excitation (SE) module.
 *
 * SE module applies channel-wise attention:
 * 1. Squeeze: Global average pooling
 * 2. Excitation: Two FC layers with activation
 * 3. Scale: Multiply with input
 *
 * Architecture:
 * ```
 * Input (C channels)
 *   ↓
 * Global Average Pool → [1×1×C]
 *   ↓
 * FC: C → C/reduction
 *   ↓
 * ReLU
 *   ↓
 * FC: C/reduction → C
 *   ↓
 * Hard-Sigmoid (or Sigmoid)
 *   ↓
 * Channel-wise Multiply with Input
 *   ↓
 * Output
 * ```
 */
class MobileNetSqueezeExcitation : public nn::Module {
public:
    /**
     * @brief Construct SE module.
     *
     * @param channels Number of input channels
     * @param reduction Reduction ratio (default: 4)
     * @param use_hard_sigmoid Use Hard-Sigmoid instead of Sigmoid (default: true for MobileNetV3)
     */
    MobileNetSqueezeExcitation(int64_t channels,
                               int64_t reduction = 4,
                               bool use_hard_sigmoid = true);

    auto forward(const Variable& input) -> Variable override;

private:
    std::shared_ptr<nn::AdaptiveAvgPool2d> pool_;
    std::shared_ptr<nn::Linear> fc1_;
    nn::ReLU relu_;
    std::shared_ptr<nn::Linear> fc2_;
    bool use_hard_sigmoid_;
    HardSigmoid hard_sigmoid_;
    nn::Sigmoid sigmoid_;
};

/**
 * @brief Inverted Residual Block (MBConv).
 *
 * Used in both MobileNetV2 and MobileNetV3.
 * Unlike traditional residuals (wide → narrow → wide), this uses
 * inverted residuals (narrow → wide → narrow).
 *
 * Architecture:
 * ```
 * Input (C_in, narrow)
 *   ↓
 * [Optional: 1×1 Conv Expansion if exp_ratio != 1] → t*C_in (wide)
 *   ↓
 * BatchNorm + Activation
 *   ↓
 * Depthwise Conv k×k → t*C_in
 *   ↓
 * BatchNorm + Activation
 *   ↓
 * [Optional: SE Module]
 *   ↓
 * 1×1 Conv Projection → C_out (narrow)
 *   ↓
 * BatchNorm (NO activation - Linear Bottleneck)
 *   ↓
 * [Residual if stride==1 and C_in==C_out]
 *   ↓
 * Output (C_out, narrow)
 * ```
 *
 * Key points:
 * - Expansion ratio (t) usually 6 for MobileNetV2, varies for V3
 * - Linear bottleneck: No activation after final projection
 * - Residual connection only when stride=1 and input/output match
 */
class InvertedResidual : public nn::Module {
public:
    /**
     * @brief Construct Inverted Residual block.
     *
     * @param in_channels Number of input channels
     * @param out_channels Number of output channels
     * @param stride Stride for depthwise conv (1 or 2)
     * @param expand_ratio Channel expansion ratio
     * @param kernel_size Depthwise conv kernel size (3 or 5)
     * @param use_se Use Squeeze-and-Excitation module
     * @param use_hs Use Hard-Swish activation (otherwise ReLU6)
     */
    InvertedResidual(int64_t in_channels,
                     int64_t out_channels,
                     int64_t stride,
                     int64_t expand_ratio,
                     int64_t kernel_size = 3,
                     bool use_se = false,
                     bool use_hs = false);

    auto forward(const Variable& input) -> Variable override;

private:
    bool use_residual_;  ///< Whether to use skip connection
    std::shared_ptr<nn::Sequential> conv_;  ///< Main convolution sequence
};

/**
 * @brief MobileNetV2 model.
 *
 * MobileNetV2 uses inverted residual blocks with linear bottlenecks.
 * Efficient architecture for mobile and embedded devices.
 *
 * Parameters: ~3.4M (width=1.0)
 * FLOPs: ~300M (width=1.0, resolution=224)
 * Top-1 Accuracy (ImageNet): ~72.0%
 */
class MobileNetV2 : public nn::Module {
public:
    /**
     * @brief Construct MobileNetV2 model.
     *
     * @param num_classes Number of output classes (default: 1000)
     * @param width_mult Width multiplier (0.5, 0.75, 1.0, 1.4)
     * @param dropout Dropout rate (default: 0.2)
     */
    MobileNetV2(int64_t num_classes = 1000,
                double width_mult = 1.0,
                double dropout = 0.2);

    auto forward(const Variable& input) -> Variable override;
    auto forward_features(const Variable& input) -> Variable;
    auto load_pretrained(const std::string& path) -> void;

private:
    auto make_divisible(int64_t v, int64_t divisor = 8) -> int64_t;

    std::shared_ptr<nn::Sequential> features_;
    std::shared_ptr<nn::Sequential> classifier_;
};

/**
 * @brief MobileNetV3 model (Large and Small variants).
 *
 * MobileNetV3 uses NAS-discovered architectures with:
 * - Hard-Swish activation
 * - SE modules in selected layers
 * - Redesigned expensive layers
 *
 * MobileNetV3-Large:
 * - Parameters: ~5.4M
 * - FLOPs: ~219M
 * - Top-1 Accuracy (ImageNet): ~75.2%
 *
 * MobileNetV3-Small:
 * - Parameters: ~2.9M
 * - FLOPs: ~66M
 * - Top-1 Accuracy (ImageNet): ~67.4%
 */
class MobileNetV3 : public nn::Module {
public:
    /**
     * @brief Construct MobileNetV3 model.
     *
     * @param num_classes Number of output classes (default: 1000)
     * @param mode Model size: "large" or "small"
     * @param width_mult Width multiplier (0.75, 1.0)
     * @param dropout Dropout rate (default: 0.2)
     */
    MobileNetV3(int64_t num_classes = 1000,
                const std::string& mode = "large",
                double width_mult = 1.0,
                double dropout = 0.2);

    auto forward(const Variable& input) -> Variable override;
    auto load_pretrained(const std::string& path) -> void;

private:
    auto make_divisible(int64_t v, int64_t divisor = 8) -> int64_t;
    auto build_large_config() -> std::vector<std::vector<int64_t>>;
    auto build_small_config() -> std::vector<std::vector<int64_t>>;

    std::shared_ptr<nn::Sequential> features_;
    std::shared_ptr<nn::Sequential> classifier_;
};

// ============================================================================
// Factory Functions for MobileNet Variants
// ============================================================================

/**
 * @brief Create MobileNetV2 model with width multiplier 1.0.
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to MobileNetV2 model
 */
auto mobilenet_v2(int64_t num_classes = 1000, bool pretrained = false)
    -> std::shared_ptr<MobileNetV2>;

/**
 * @brief Create MobileNetV2 model with custom width multiplier.
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param width_mult Width multiplier (0.5, 0.75, 1.0, 1.4)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to MobileNetV2 model
 */
auto mobilenet_v2_width(int64_t num_classes, double width_mult, bool pretrained = false)
    -> std::shared_ptr<MobileNetV2>;

/**
 * @brief Create MobileNetV3-Large model.
 *
 * Architecture optimized for high accuracy.
 * Parameters: ~5.4M
 * Top-1 Accuracy: ~75.2%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to MobileNetV3-Large model
 */
auto mobilenet_v3_large(int64_t num_classes = 1000, bool pretrained = false)
    -> std::shared_ptr<MobileNetV3>;

/**
 * @brief Create MobileNetV3-Small model.
 *
 * Architecture optimized for minimal latency.
 * Parameters: ~2.9M
 * Top-1 Accuracy: ~67.4%
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained ImageNet weights (default: false)
 * @return Shared pointer to MobileNetV3-Small model
 */
auto mobilenet_v3_small(int64_t num_classes = 1000, bool pretrained = false)
    -> std::shared_ptr<MobileNetV3>;

} // namespace models
} // namespace tenzor
