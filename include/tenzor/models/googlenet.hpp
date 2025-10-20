/**
 * @file googlenet.hpp
 * @brief GoogLeNet (Inception v1) convolutional neural network
 *
 * Implements GoogLeNet architecture from:
 * "Going Deeper with Convolutions"
 * Szegedy et al., CVPR 2015
 *
 * GoogLeNet introduced the Inception module, which applies multiple
 * convolutional filters of different sizes in parallel, allowing the
 * network to learn multi-scale features efficiently.
 *
 * Key innovations:
 * - Inception modules (multi-scale feature extraction)
 * - 1x1 convolutions for dimension reduction
 * - Global average pooling (no FC layers)
 * - Auxiliary classifiers for training deep networks
 */

#pragma once

#include "../nn/module.hpp"
#include "../nn/layers/conv.hpp"
#include "../nn/layers/linear.hpp"
#include "../nn/layers/pooling.hpp"
#include "../nn/layers/batchnorm.hpp"
#include "../nn/layers/dropout.hpp"
#include "../nn/layers/flatten.hpp"
#include "../nn/activations/activations.hpp"
#include <memory>
#include <optional>

namespace tenzor {
namespace models {

/**
 * @brief Inception module (basic building block of GoogLeNet)
 *
 * Inception module applies 4 parallel operations:
 * 1. 1x1 convolution
 * 2. 1x1 convolution -> 3x3 convolution
 * 3. 1x1 convolution -> 5x5 convolution
 * 4. 3x3 max pooling -> 1x1 convolution
 *
 * All outputs are concatenated along the channel dimension.
 *
 * The 1x1 convolutions reduce dimensionality before expensive 3x3 and 5x5
 * operations, making the network computationally efficient.
 *
 * Shape transformations:
 * - Input: (N, in_channels, H, W)
 * - Output: (N, out_1x1 + out_3x3 + out_5x5 + out_pool, H, W)
 *
 * @code
 * // Inception module as in GoogLeNet
 * auto inception3a = std::make_shared<InceptionModule>(
 *     192,        // in_channels
 *     64,         // out_1x1
 *     96, 128,    // reduce_3x3, out_3x3
 *     16, 32,     // reduce_5x5, out_5x5
 *     32          // out_pool_proj
 * );
 *
 * Variable x(Tensor({1, 192, 28, 28}, DType::Float32, Device::cpu()), true);
 * Variable out = inception3a->forward(x);  // Shape: {1, 256, 28, 28}
 * @endcode
 */
class InceptionModule : public nn::Module {
public:
    /**
     * @brief Construct Inception module
     *
     * @param in_channels Number of input channels
     * @param out_1x1 Output channels for 1x1 branch
     * @param reduce_3x3 Reduction channels before 3x3 conv
     * @param out_3x3 Output channels for 3x3 branch
     * @param reduce_5x5 Reduction channels before 5x5 conv
     * @param out_5x5 Output channels for 5x5 branch
     * @param out_pool_proj Output channels for pooling branch projection
     *
     * @code
     * InceptionModule inc(192, 64, 96, 128, 16, 32, 32);
     * // Total output channels: 64 + 128 + 32 + 32 = 256
     * @endcode
     */
    InceptionModule(int64_t in_channels,
                    int64_t out_1x1,
                    int64_t reduce_3x3, int64_t out_3x3,
                    int64_t reduce_5x5, int64_t out_5x5,
                    int64_t out_pool_proj);

    /**
     * @brief Forward pass through Inception module
     *
     * @param x Input tensor of shape (N, in_channels, H, W)
     * @return Concatenated output of all branches
     */
    auto forward(const Variable& x) -> Variable override;

private:
    // Branch 1: 1x1 convolution
    std::shared_ptr<nn::Conv2d> branch1_conv1x1_;
    std::shared_ptr<nn::BatchNorm2d> branch1_bn_;

    // Branch 2: 1x1 -> 3x3 convolution
    std::shared_ptr<nn::Conv2d> branch2_conv1x1_;
    std::shared_ptr<nn::BatchNorm2d> branch2_bn1_;
    std::shared_ptr<nn::Conv2d> branch2_conv3x3_;
    std::shared_ptr<nn::BatchNorm2d> branch2_bn2_;

    // Branch 3: 1x1 -> 5x5 convolution
    std::shared_ptr<nn::Conv2d> branch3_conv1x1_;
    std::shared_ptr<nn::BatchNorm2d> branch3_bn1_;
    std::shared_ptr<nn::Conv2d> branch3_conv5x5_;
    std::shared_ptr<nn::BatchNorm2d> branch3_bn2_;

    // Branch 4: max pool -> 1x1 convolution
    std::shared_ptr<nn::MaxPool2d> branch4_pool_;
    std::shared_ptr<nn::Conv2d> branch4_conv1x1_;
    std::shared_ptr<nn::BatchNorm2d> branch4_bn_;
};

/**
 * @brief Auxiliary classifier for GoogLeNet
 *
 * GoogLeNet uses auxiliary classifiers attached to intermediate layers
 * during training to combat vanishing gradients in very deep networks.
 * They provide additional gradient signal to lower layers.
 *
 * During training, total loss = main_loss + 0.3 * aux_loss1 + 0.3 * aux_loss2
 * During inference, auxiliary classifiers are not used.
 *
 * Architecture:
 * - 5x5 average pooling, stride 3
 * - 1x1 conv, 128 filters
 * - FC, 1024 units -> ReLU -> Dropout
 * - FC, num_classes units
 *
 * @code
 * auto aux = std::make_shared<InceptionAux>(512, 1000);
 *
 * Variable x(Tensor({1, 512, 14, 14}, DType::Float32, Device::cpu()), true);
 * Variable aux_out = aux->forward(x);  // Shape: {1, 1000}
 * @endcode
 */
class InceptionAux : public nn::Module {
public:
    /**
     * @brief Construct auxiliary classifier
     *
     * @param in_channels Number of input channels
     * @param num_classes Number of output classes
     * @param dropout Dropout probability (default: 0.7)
     */
    InceptionAux(int64_t in_channels, int64_t num_classes, double dropout = 0.7);

    /**
     * @brief Forward pass through auxiliary classifier
     *
     * @param x Input tensor from intermediate layer
     * @return Classification logits
     */
    auto forward(const Variable& x) -> Variable override;

private:
    std::shared_ptr<nn::AvgPool2d> avgpool_;
    std::shared_ptr<nn::Conv2d> conv_;
    std::shared_ptr<nn::BatchNorm2d> bn_;
    std::shared_ptr<nn::Linear> fc1_;
    std::shared_ptr<nn::Linear> fc2_;
    std::shared_ptr<nn::Dropout> dropout_;
};

/**
 * @brief GoogLeNet (Inception v1) convolutional neural network
 *
 * GoogLeNet is 22 layers deep and uses Inception modules to achieve
 * efficient multi-scale feature extraction. It won ImageNet 2014.
 *
 * Architecture:
 * - Input: 224x224x3 RGB images
 * - Conv1: 64 filters, 7x7, stride 2 -> MaxPool
 * - Conv2: 192 filters, 3x3 -> MaxPool
 * - Inception3a, Inception3b -> MaxPool
 * - Inception4a, Inception4b, Inception4c, Inception4d, Inception4e -> MaxPool
 * - Inception5a, Inception5b
 * - Global average pooling
 * - Dropout -> FC (num_classes)
 *
 * Auxiliary classifiers at Inception4a and Inception4d (training only)
 *
 * Shape transformations:
 * - Input: (N, 3, 224, 224)
 * - After conv blocks: (N, 1024, 7, 7)
 * - After global pooling: (N, 1024)
 * - Output: (N, num_classes)
 *
 * @code
 * // Create GoogLeNet for ImageNet
 * auto googlenet = std::make_shared<GoogLeNet>(1000, true);
 *
 * // Training mode (with auxiliary outputs)
 * googlenet->train();
 * auto [main_out, aux1_out, aux2_out] = googlenet->forward_with_aux(input);
 * auto loss = main_loss + 0.3 * aux1_loss + 0.3 * aux2_loss;
 *
 * // Inference mode (no auxiliary outputs)
 * googlenet->eval();
 * Variable output = googlenet->forward(input);
 * @endcode
 */
class GoogLeNet : public nn::Module {
public:
    /**
     * @brief Construct GoogLeNet
     *
     * @param num_classes Number of output classes (default: 1000)
     * @param aux_logits Enable auxiliary classifiers for training (default: true)
     * @param dropout Dropout probability (default: 0.4)
     * @param init_weights Initialize weights (default: true)
     *
     * @code
     * GoogLeNet googlenet_imagenet(1000, true);   // With aux classifiers
     * GoogLeNet googlenet_inference(1000, false); // No aux classifiers
     * @endcode
     */
    explicit GoogLeNet(int64_t num_classes = 1000,
                       bool aux_logits = true,
                       double dropout = 0.4,
                       bool init_weights = true);

    /**
     * @brief Forward pass through GoogLeNet
     *
     * In training mode with aux_logits=true, auxiliary outputs are computed
     * but not returned by this method. Use forward_with_aux() instead.
     *
     * @param x Input tensor of shape (N, 3, 224, 224)
     * @return Output logits of shape (N, num_classes)
     */
    auto forward(const Variable& x) -> Variable override;

    /**
     * @brief Forward pass with auxiliary outputs
     *
     * Returns main output and two auxiliary outputs for training.
     * Only use during training when aux_logits=true.
     *
     * @param x Input tensor
     * @return Tuple of (main_output, aux1_output, aux2_output)
     *
     * @code
     * auto [main, aux1, aux2] = model->forward_with_aux(x);
     * auto loss = criterion(main, target) +
     *             0.3 * criterion(aux1, target) +
     *             0.3 * criterion(aux2, target);
     * @endcode
     */
    auto forward_with_aux(const Variable& x) -> std::tuple<Variable, Variable, Variable>;

private:
    /**
     * @brief Build network architecture
     */
    auto make_layers(int64_t num_classes, bool aux_logits, double dropout) -> void;

    /**
     * @brief Initialize weights
     *
     * - Conv layers: Kaiming uniform
     * - Linear layers: Kaiming uniform
     * - Batch norm: gamma=1, beta=0
     */
    auto initialize_weights() -> void;

    bool aux_logits_;  ///< Whether to use auxiliary classifiers

    // Initial convolutional layers
    std::shared_ptr<nn::Conv2d> conv1_;
    std::shared_ptr<nn::BatchNorm2d> bn1_;
    std::shared_ptr<nn::MaxPool2d> maxpool1_;

    std::shared_ptr<nn::Conv2d> conv2_;
    std::shared_ptr<nn::BatchNorm2d> bn2_;
    std::shared_ptr<nn::Conv2d> conv3_;
    std::shared_ptr<nn::BatchNorm2d> bn3_;
    std::shared_ptr<nn::MaxPool2d> maxpool2_;

    // Inception modules
    std::shared_ptr<InceptionModule> inception3a_;
    std::shared_ptr<InceptionModule> inception3b_;
    std::shared_ptr<nn::MaxPool2d> maxpool3_;

    std::shared_ptr<InceptionModule> inception4a_;
    std::shared_ptr<InceptionModule> inception4b_;
    std::shared_ptr<InceptionModule> inception4c_;
    std::shared_ptr<InceptionModule> inception4d_;
    std::shared_ptr<InceptionModule> inception4e_;
    std::shared_ptr<nn::MaxPool2d> maxpool4_;

    std::shared_ptr<InceptionModule> inception5a_;
    std::shared_ptr<InceptionModule> inception5b_;

    // Auxiliary classifiers (optional)
    std::shared_ptr<InceptionAux> aux1_;
    std::shared_ptr<InceptionAux> aux2_;

    // Final layers
    std::shared_ptr<nn::AdaptiveAvgPool2d> avgpool_;
    std::shared_ptr<nn::Dropout> dropout_;
    std::shared_ptr<nn::Linear> fc_;
};

/**
 * @brief Create GoogLeNet model
 *
 * @param num_classes Number of output classes (default: 1000)
 * @param pretrained Load pretrained weights (default: false)
 * @param aux_logits Enable auxiliary classifiers (default: true)
 * @return Shared pointer to GoogLeNet model
 *
 * @code
 * auto model = googlenet(1000, false, true);  // For training
 * auto inference_model = googlenet(1000, true, false);  // For inference
 * @endcode
 */
auto googlenet(int64_t num_classes = 1000, bool pretrained = false, bool aux_logits = true) -> std::shared_ptr<GoogLeNet>;

} // namespace models
} // namespace tenzor
