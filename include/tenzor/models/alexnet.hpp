/**
 * @file alexnet.hpp
 * @brief AlexNet convolutional neural network
 *
 * Implements AlexNet architecture from:
 * "ImageNet Classification with Deep Convolutional Neural Networks"
 * Krizhevsky, Sutskever & Hinton, NeurIPS 2012
 *
 * AlexNet was the breakthrough network that won ImageNet 2012 and sparked
 * the deep learning revolution in computer vision. It introduced:
 * - ReLU activation (instead of tanh/sigmoid)
 * - Dropout regularization
 * - Data augmentation
 * - GPU training
 */

#pragma once

#include "../nn/module.hpp"
#include "../nn/layers/conv.hpp"
#include "../nn/layers/linear.hpp"
#include "../nn/layers/pooling.hpp"
#include "../nn/layers/dropout.hpp"
#include "../nn/activations/activations.hpp"
#include <memory>

namespace tenzor {
namespace models {

/**
 * @brief AlexNet convolutional neural network
 *
 * AlexNet consists of 5 convolutional layers followed by 3 fully connected layers.
 * Key innovations include ReLU activation, overlapping pooling, and dropout.
 *
 * Architecture:
 * - Input: 224x224x3 RGB images
 * - Conv1: 96 filters, 11x11, stride 4 -> ReLU -> MaxPool 3x3, stride 2
 * - Conv2: 256 filters, 5x5, pad 2 -> ReLU -> MaxPool 3x3, stride 2
 * - Conv3: 384 filters, 3x3, pad 1 -> ReLU
 * - Conv4: 384 filters, 3x3, pad 1 -> ReLU
 * - Conv5: 256 filters, 3x3, pad 1 -> ReLU -> MaxPool 3x3, stride 2
 * - FC1: 4096 -> ReLU -> Dropout
 * - FC2: 4096 -> ReLU -> Dropout
 * - FC3: num_classes
 *
 * Shape transformations:
 * - Input: (N, 3, 224, 224)
 * - Conv1 out: (N, 96, 55, 55) -> Pool: (N, 96, 27, 27)
 * - Conv2 out: (N, 256, 27, 27) -> Pool: (N, 256, 13, 13)
 * - Conv3 out: (N, 384, 13, 13)
 * - Conv4 out: (N, 384, 13, 13)
 * - Conv5 out: (N, 256, 13, 13) -> Pool: (N, 256, 6, 6)
 * - Flatten: (N, 256*6*6) = (N, 9216)
 * - FC1 out: (N, 4096)
 * - FC2 out: (N, 4096)
 * - FC3 out: (N, num_classes)
 *
 * @code
 * // Create AlexNet for ImageNet
 * auto alexnet = std::make_shared<AlexNet>(1000);
 *
 * // Forward pass
 * Variable input(Tensor({1, 3, 224, 224}, DType::Float32, Device::cpu()), true);
 * Variable output = alexnet->forward(input);  // Shape: {1, 1000}
 * @endcode
 */
class AlexNet : public nn::Module {
public:
    /**
     * @brief Construct AlexNet
     *
     * @param num_classes Number of output classes (default: 1000 for ImageNet)
     * @param dropout Dropout probability (default: 0.5)
     *
     * @code
     * auto alexnet_imagenet = AlexNet(1000);      // For ImageNet
     * auto alexnet_cifar = AlexNet(10, 0.3);      // For CIFAR-10
     * @endcode
     */
    explicit AlexNet(int64_t num_classes = 1000, double dropout = 0.5);

    /**
     * @brief Forward pass through AlexNet
     *
     * @param x Input tensor of shape (N, 3, 224, 224)
     * @return Output logits of shape (N, num_classes)
     *
     * @throws std::runtime_error if input shape is invalid
     */
    auto forward_impl(const Variable& x) -> Variable override;

private:
    /**
     * @brief Build convolutional feature extractor
     *
     * Creates 5 convolutional layers with ReLU and max pooling.
     */
    auto make_features() -> void;

    /**
     * @brief Build classifier
     *
     * Creates 3 FC layers with ReLU and dropout.
     *
     * @param num_classes Number of output classes
     * @param dropout Dropout probability
     */
    auto make_classifier(int64_t num_classes, double dropout) -> void;

    /**
     * @brief Initialize weights
     *
     * - Conv layers: Kaiming uniform (for ReLU)
     * - FC layers: Kaiming uniform
     * - Biases: Small constant (0.01)
     */
    auto initialize_weights() -> void;

    std::shared_ptr<nn::Sequential> features_;     ///< Convolutional layers
    std::shared_ptr<nn::Sequential> avgpool_;      ///< Adaptive average pooling
    std::shared_ptr<nn::Sequential> classifier_;   ///< Fully connected layers
};

/**
 * @brief Create AlexNet model
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained weights (default: false)
 * @return Shared pointer to AlexNet model
 *
 * @code
 * auto model = alexnet(1000, false);  // Random initialization
 * auto pretrained_model = alexnet(1000, true);  // Load pretrained weights
 * @endcode
 */
auto alexnet(int64_t num_classes = 1000, bool pretrained = false) -> std::shared_ptr<AlexNet>;

} // namespace models
} // namespace tenzor
