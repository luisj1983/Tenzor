/**
 * @file vgg.hpp
 * @brief VGG (Visual Geometry Group) convolutional neural networks
 *
 * Implements VGG-11, VGG-13, VGG-16, and VGG-19 architectures from:
 * "Very Deep Convolutional Networks for Large-Scale Image Recognition"
 * Simonyan & Zisserman, ICLR 2015
 *
 * VGG networks are characterized by:
 * - Deep sequential convolutional layers with small 3x3 kernels
 * - Batch normalization after each conv layer
 * - Max pooling for downsampling
 * - Three fully connected layers at the end
 */

#pragma once

#include "../nn/module.hpp"
#include "../nn/layers/conv.hpp"
#include "../nn/layers/linear.hpp"
#include "../nn/layers/pooling.hpp"
#include "../nn/layers/batchnorm.hpp"
#include "../nn/layers/dropout.hpp"
#include "../nn/activations/activations.hpp"
#include <vector>
#include <string>

namespace tenzor {
namespace models {

/**
 * @brief VGG configuration for different variants
 *
 * Defines the convolutional layer structure for each VGG variant.
 * Numbers represent output channels, 'M' represents max pooling.
 */
struct VGGConfig {
    std::vector<int64_t> layers;  ///< Layer configuration (channels or -1 for maxpool)

    static VGGConfig vgg11();  ///< VGG-11: 8 conv + 3 FC
    static VGGConfig vgg13();  ///< VGG-13: 10 conv + 3 FC
    static VGGConfig vgg16();  ///< VGG-16: 13 conv + 3 FC
    static VGGConfig vgg19();  ///< VGG-19: 16 conv + 3 FC
};

/**
 * @brief VGG convolutional neural network
 *
 * VGG networks use stacks of 3x3 convolutional layers with increasing depth,
 * followed by max pooling and fully connected layers. The network is simple
 * but very deep, demonstrating that depth is a critical component for
 * achieving good performance.
 *
 * Architecture:
 * - Input: 224x224x3 RGB images
 * - Features: Sequential conv blocks with max pooling
 * - Classifier: 3 FC layers (4096, 4096, num_classes)
 * - Activation: ReLU throughout
 *
 * Shape transformations:
 * - Input: (N, 3, 224, 224)
 * - After features: (N, 512, 7, 7)
 * - After flatten: (N, 512*7*7)
 * - Output: (N, num_classes)
 *
 * @code
 * // Create VGG-16 for ImageNet
 * auto vgg16 = std::make_shared<VGG>(VGGConfig::vgg16(), 1000, true);
 *
 * // Forward pass
 * Variable input(Tensor({1, 3, 224, 224}, DType::Float32, Device::cpu()), true);
 * Variable output = vgg16->forward(input);  // Shape: {1, 1000}
 * @endcode
 */
class VGG : public nn::Module {
public:
    /**
     * @brief Construct VGG network
     *
     * @param config VGG configuration (vgg11, vgg13, vgg16, or vgg19)
     * @param num_classes Number of output classes (default: 1000 for ImageNet)
     * @param batch_norm Enable batch normalization after conv layers (default: true)
     * @param dropout Dropout probability in classifier (default: 0.5)
     * @param init_weights Initialize weights using Kaiming uniform (default: true)
     *
     * @code
     * auto vgg16 = VGG(VGGConfig::vgg16(), 1000, true);     // VGG-16 with BN
     * auto vgg19 = VGG(VGGConfig::vgg19(), 10, true, 0.3);  // VGG-19 for CIFAR-10
     * @endcode
     */
    VGG(const VGGConfig& config,
        int64_t num_classes = 1000,
        bool batch_norm = true,
        double dropout = 0.5,
        bool init_weights = true);

    /**
     * @brief Forward pass through VGG network
     *
     * @param x Input tensor of shape (N, 3, 224, 224)
     * @return Output logits of shape (N, num_classes)
     *
     * @throws std::runtime_error if input shape is invalid
     */
    auto forward_impl(const Variable& x) -> Variable override;

private:
    /**
     * @brief Build feature extraction layers
     *
     * Creates conv blocks based on configuration.
     * Each block has one or more conv layers followed by max pooling.
     *
     * @param config Layer configuration
     * @param batch_norm Enable batch normalization
     */
    auto make_features(const VGGConfig& config, bool batch_norm) -> void;

    /**
     * @brief Build classifier layers
     *
     * Creates 3 FC layers: 512*7*7 -> 4096 -> 4096 -> num_classes
     *
     * @param num_classes Number of output classes
     * @param dropout Dropout probability
     */
    auto make_classifier(int64_t num_classes, double dropout) -> void;

    /**
     * @brief Initialize weights using Kaiming uniform
     *
     * - Conv layers: Kaiming uniform initialization
     * - Linear layers: Kaiming uniform initialization
     * - Batch norm: gamma=1, beta=0
     */
    auto initialize_weights() -> void;

    std::shared_ptr<nn::Sequential> features_;     ///< Feature extraction layers
    std::shared_ptr<nn::Sequential> avgpool_;      ///< Adaptive average pooling
    std::shared_ptr<nn::Sequential> classifier_;   ///< Classifier layers
};

/**
 * @brief Create VGG-11 network
 *
 * VGG-11 has 8 convolutional layers:
 * - Conv blocks: [64] -> [128] -> [256, 256] -> [512, 512] -> [512, 512]
 * - Each block followed by max pooling
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param batch_norm Enable batch normalization (default: true)
 * @param pretrained Load pretrained weights (default: false)
 * @return Shared pointer to VGG-11 model
 */
auto vgg11(int64_t num_classes = 1000, bool batch_norm = true, bool pretrained = false) -> std::shared_ptr<VGG>;

/**
 * @brief Create VGG-13 network
 *
 * VGG-13 has 10 convolutional layers:
 * - Conv blocks: [64, 64] -> [128, 128] -> [256, 256] -> [512, 512] -> [512, 512]
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param batch_norm Enable batch normalization (default: true)
 * @param pretrained Load pretrained weights (default: false)
 * @return Shared pointer to VGG-13 model
 */
auto vgg13(int64_t num_classes = 1000, bool batch_norm = true, bool pretrained = false) -> std::shared_ptr<VGG>;

/**
 * @brief Create VGG-16 network (most commonly used variant)
 *
 * VGG-16 has 13 convolutional layers:
 * - Conv blocks: [64, 64] -> [128, 128] -> [256, 256, 256] -> [512, 512, 512] -> [512, 512, 512]
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param batch_norm Enable batch normalization (default: true)
 * @param pretrained Load pretrained weights (default: false)
 * @return Shared pointer to VGG-16 model
 */
auto vgg16(int64_t num_classes = 1000, bool batch_norm = true, bool pretrained = false) -> std::shared_ptr<VGG>;

/**
 * @brief Create VGG-19 network (deepest variant)
 *
 * VGG-19 has 16 convolutional layers:
 * - Conv blocks: [64, 64] -> [128, 128] -> [256, 256, 256, 256] -> [512, 512, 512, 512] -> [512, 512, 512, 512]
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param batch_norm Enable batch normalization (default: true)
 * @param pretrained Load pretrained weights (default: false)
 * @return Shared pointer to VGG-19 model
 */
auto vgg19(int64_t num_classes = 1000, bool batch_norm = true, bool pretrained = false) -> std::shared_ptr<VGG>;

} // namespace models
} // namespace tenzor
