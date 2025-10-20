/**
 * @file alexnet.cpp
 * @brief Implementation of AlexNet
 */

#include "../../include/tenzor/models/alexnet.hpp"
#include "../../include/tenzor/ops/transform.hpp"
#include <stdexcept>

namespace tenzor {
namespace models {

AlexNet::AlexNet(int64_t num_classes, double dropout) {
    // Build convolutional features
    make_features();

    // Adaptive average pooling to fixed size
    avgpool_ = std::make_shared<nn::Sequential>();
    avgpool_->add_module(std::make_shared<nn::AdaptiveAvgPool2d>(6, 6));
    register_module("avgpool", avgpool_);

    // Build classifier
    make_classifier(num_classes, dropout);

    // Initialize weights
    initialize_weights();
}

auto AlexNet::forward(const Variable& x) -> Variable {
    // Feature extraction
    auto features = features_->forward(x);

    // Adaptive pooling
    auto pooled = avgpool_->forward(features);

    // Flatten: (N, 256, 6, 6) -> (N, 256*6*6)
    auto& pooled_tensor = pooled.tensor();
    auto flat_tensor = flatten(pooled_tensor, 1);
    Variable flat(flat_tensor, pooled.requires_grad());

    // Classification
    return classifier_->forward(flat);
}

auto AlexNet::make_features() -> void {
    features_ = std::make_shared<nn::Sequential>();

    // Conv1: 3 -> 96, 11x11, stride 4, padding 2
    auto conv1 = std::make_shared<nn::Conv2d>(3, 96, 11, 4, 2);
    features_->add_module(conv1);
    features_->add_module(std::make_shared<nn::ReLU>());
    features_->add_module(std::make_shared<nn::MaxPool2d>(3, 2));  // 3x3, stride 2

    // Conv2: 96 -> 256, 5x5, stride 1, padding 2
    auto conv2 = std::make_shared<nn::Conv2d>(96, 256, 5, 1, 2);
    features_->add_module(conv2);
    features_->add_module(std::make_shared<nn::ReLU>());
    features_->add_module(std::make_shared<nn::MaxPool2d>(3, 2));  // 3x3, stride 2

    // Conv3: 256 -> 384, 3x3, stride 1, padding 1
    auto conv3 = std::make_shared<nn::Conv2d>(256, 384, 3, 1, 1);
    features_->add_module(conv3);
    features_->add_module(std::make_shared<nn::ReLU>());

    // Conv4: 384 -> 384, 3x3, stride 1, padding 1
    auto conv4 = std::make_shared<nn::Conv2d>(384, 384, 3, 1, 1);
    features_->add_module(conv4);
    features_->add_module(std::make_shared<nn::ReLU>());

    // Conv5: 384 -> 256, 3x3, stride 1, padding 1
    auto conv5 = std::make_shared<nn::Conv2d>(384, 256, 3, 1, 1);
    features_->add_module(conv5);
    features_->add_module(std::make_shared<nn::ReLU>());
    features_->add_module(std::make_shared<nn::MaxPool2d>(3, 2));  // 3x3, stride 2

    register_module("features", features_);
}

auto AlexNet::make_classifier(int64_t num_classes, double dropout) -> void {
    classifier_ = std::make_shared<nn::Sequential>();

    // FC1: 256*6*6 -> 4096
    auto fc1 = std::make_shared<nn::Linear>(256 * 6 * 6, 4096);
    classifier_->add_module(fc1);
    classifier_->add_module(std::make_shared<nn::ReLU>());
    classifier_->add_module(std::make_shared<nn::Dropout>(dropout));

    // FC2: 4096 -> 4096
    auto fc2 = std::make_shared<nn::Linear>(4096, 4096);
    classifier_->add_module(fc2);
    classifier_->add_module(std::make_shared<nn::ReLU>());
    classifier_->add_module(std::make_shared<nn::Dropout>(dropout));

    // FC3: 4096 -> num_classes
    auto fc3 = std::make_shared<nn::Linear>(4096, num_classes);
    classifier_->add_module(fc3);

    register_module("classifier", classifier_);
}

auto AlexNet::initialize_weights() -> void {
    // Weights are already initialized by individual layers using Kaiming uniform
    // This is appropriate for ReLU activation
}

// Factory function

auto alexnet(int64_t num_classes, bool pretrained) -> std::shared_ptr<AlexNet> {
    auto model = std::make_shared<AlexNet>(num_classes);

    if (pretrained) {
        // TODO: Load pretrained weights
        throw std::runtime_error("Pretrained weights not yet implemented");
    }

    return model;
}

} // namespace models
} // namespace tenzor
