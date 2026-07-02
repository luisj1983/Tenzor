/**
 * @file alexnet.cpp
 * @brief Implementation of AlexNet
 */

#include "../../include/tenzor/models/alexnet.hpp"
#include "../../include/tenzor/ops/transform.hpp"
#include "../../include/tenzor/models/hub.hpp"
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

auto AlexNet::forward_impl(const Variable& x) -> Variable {
    // Feature extraction
    auto features = features_->forward(x);

    // Adaptive pooling
    auto pooled = avgpool_->forward(features);

    // Flatten using autograd-aware reshape to preserve gradient flow
    auto& pooled_tensor = pooled.tensor();
    auto batch_size = pooled_tensor.shape()[0];
    // Calculate flattened size: all dimensions after batch
    int64_t flat_size = 1;
    for (size_t i = 1; i < pooled_tensor.shape().size(); ++i) {
        flat_size *= pooled_tensor.shape()[i];
    }
    auto flat = pooled.reshape({batch_size, flat_size});

    // Classification
    return classifier_->forward(flat);
}

auto AlexNet::make_features() -> void {
    features_ = std::make_shared<nn::Sequential>();

    // Channel widths match torchvision's AlexNet (64/192/384/256/256), which is
    // the checkpoint the pretrained loader downloads. The original paper used
    // 96/256/384/384/256; using those widths made the pretrained weights fail to
    // map onto the layers.

    // Conv1: 3 -> 64, 11x11, stride 4, padding 2
    auto conv1 = std::make_shared<nn::Conv2d>(3, 64, 11, 4, 2);
    features_->add_module(conv1);
    features_->add_module(std::make_shared<nn::ReLU>());
    features_->add_module(std::make_shared<nn::MaxPool2d>(3, 2));  // 3x3, stride 2

    // Conv2: 64 -> 192, 5x5, stride 1, padding 2
    auto conv2 = std::make_shared<nn::Conv2d>(64, 192, 5, 1, 2);
    features_->add_module(conv2);
    features_->add_module(std::make_shared<nn::ReLU>());
    features_->add_module(std::make_shared<nn::MaxPool2d>(3, 2));  // 3x3, stride 2

    // Conv3: 192 -> 384, 3x3, stride 1, padding 1
    auto conv3 = std::make_shared<nn::Conv2d>(192, 384, 3, 1, 1);
    features_->add_module(conv3);
    features_->add_module(std::make_shared<nn::ReLU>());

    // Conv4: 384 -> 256, 3x3, stride 1, padding 1
    auto conv4 = std::make_shared<nn::Conv2d>(384, 256, 3, 1, 1);
    features_->add_module(conv4);
    features_->add_module(std::make_shared<nn::ReLU>());

    // Conv5: 256 -> 256, 3x3, stride 1, padding 1
    auto conv5 = std::make_shared<nn::Conv2d>(256, 256, 3, 1, 1);
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
        auto path = ModelHub::download_pretrained_safetensors("alexnet");
        ModelHub::load_pretrained_weights(*model, path, /*strict=*/false);
    }

    return model;
}

} // namespace models
} // namespace tenzor
