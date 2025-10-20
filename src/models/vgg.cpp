/**
 * @file vgg.cpp
 * @brief Implementation of VGG networks
 */

#include "../../include/tenzor/models/vgg.hpp"
#include "../../include/tenzor/ops/transform.hpp"
#include <stdexcept>
#include <cmath>

namespace tenzor {
namespace models {

// VGG configurations
// -1 represents max pooling layer

auto VGGConfig::vgg11() -> VGGConfig {
    return VGGConfig{
        {64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1}
    };
}

auto VGGConfig::vgg13() -> VGGConfig {
    return VGGConfig{
        {64, 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1}
    };
}

auto VGGConfig::vgg16() -> VGGConfig {
    return VGGConfig{
        {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1}
    };
}

auto VGGConfig::vgg19() -> VGGConfig {
    return VGGConfig{
        {64, 64, -1, 128, 128, -1, 256, 256, 256, 256, -1, 512, 512, 512, 512, -1, 512, 512, 512, 512, -1}
    };
}

VGG::VGG(const VGGConfig& config,
         int64_t num_classes,
         bool batch_norm,
         double dropout,
         bool init_weights) {
    // Build feature extractor
    make_features(config, batch_norm);

    // Adaptive average pooling to fixed size
    avgpool_ = std::make_shared<nn::Sequential>();
    avgpool_->add_module(std::make_shared<nn::AdaptiveAvgPool2d>(7, 7));
    register_module("avgpool", avgpool_);

    // Build classifier
    make_classifier(num_classes, dropout);

    // Initialize weights
    if (init_weights) {
        initialize_weights();
    }
}

auto VGG::forward(const Variable& x) -> Variable {
    // Extract features
    auto features = features_->forward(x);

    // Adaptive pooling
    auto pooled = avgpool_->forward(features);

    // Flatten
    auto& pooled_tensor = pooled.tensor();
    auto batch_size = pooled_tensor.shape()[0];
    auto flat_tensor = flatten(pooled_tensor, 1);
    Variable flat(flat_tensor, pooled.requires_grad());

    // Classify
    return classifier_->forward(flat);
}

auto VGG::make_features(const VGGConfig& config, bool batch_norm) -> void {
    features_ = std::make_shared<nn::Sequential>();

    int64_t in_channels = 3;  // RGB input

    for (size_t i = 0; i < config.layers.size(); ++i) {
        int64_t v = config.layers[i];

        if (v == -1) {
            // Max pooling layer
            auto pool = std::make_shared<nn::MaxPool2d>(2, 2);  // 2x2, stride 2
            features_->add_module(pool);
        } else {
            // Convolutional layer
            auto conv = std::make_shared<nn::Conv2d>(in_channels, v, 3, 1, 1);  // 3x3, stride 1, padding 1
            features_->add_module(conv);

            if (batch_norm) {
                auto bn = std::make_shared<nn::BatchNorm2d>(v);
                features_->add_module(bn);
            }

            auto relu = std::make_shared<nn::ReLU>();
            features_->add_module(relu);

            in_channels = v;
        }
    }

    register_module("features", features_);
}

auto VGG::make_classifier(int64_t num_classes, double dropout) -> void {
    classifier_ = std::make_shared<nn::Sequential>();

    // FC1: 512*7*7 -> 4096
    auto fc1 = std::make_shared<nn::Linear>(512 * 7 * 7, 4096);
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

auto VGG::initialize_weights() -> void {
    // Weights are already initialized by individual layers using Kaiming uniform
    // This method can be extended for custom initialization or pretrained weights
}

// Factory functions

auto vgg11(int64_t num_classes, bool batch_norm, bool pretrained) -> std::shared_ptr<VGG> {
    auto model = std::make_shared<VGG>(VGGConfig::vgg11(), num_classes, batch_norm);

    if (pretrained) {
        // TODO: Load pretrained weights
        throw std::runtime_error("Pretrained weights not yet implemented");
    }

    return model;
}

auto vgg13(int64_t num_classes, bool batch_norm, bool pretrained) -> std::shared_ptr<VGG> {
    auto model = std::make_shared<VGG>(VGGConfig::vgg13(), num_classes, batch_norm);

    if (pretrained) {
        throw std::runtime_error("Pretrained weights not yet implemented");
    }

    return model;
}

auto vgg16(int64_t num_classes, bool batch_norm, bool pretrained) -> std::shared_ptr<VGG> {
    auto model = std::make_shared<VGG>(VGGConfig::vgg16(), num_classes, batch_norm);

    if (pretrained) {
        throw std::runtime_error("Pretrained weights not yet implemented");
    }

    return model;
}

auto vgg19(int64_t num_classes, bool batch_norm, bool pretrained) -> std::shared_ptr<VGG> {
    auto model = std::make_shared<VGG>(VGGConfig::vgg19(), num_classes, batch_norm);

    if (pretrained) {
        throw std::runtime_error("Pretrained weights not yet implemented");
    }

    return model;
}

} // namespace models
} // namespace tenzor
