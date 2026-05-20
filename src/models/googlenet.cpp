/**
 * @file googlenet.cpp
 * @brief Implementation of GoogLeNet (Inception v1)
 */

#include "../../include/tenzor/models/googlenet.hpp"
#include "../../include/tenzor/ops/transform.hpp"
#include "../../include/tenzor/autograd/ops.hpp"
#include "../../include/tenzor/models/hub.hpp"
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace models {

// ============================================================================
// InceptionModule Implementation
// ============================================================================

InceptionModule::InceptionModule(int64_t in_channels,
                                 int64_t out_1x1,
                                 int64_t reduce_3x3, int64_t out_3x3,
                                 int64_t reduce_5x5, int64_t out_5x5,
                                 int64_t out_pool_proj) {
    // Branch 1: 1x1 convolution
    branch1_conv1x1_ = std::make_shared<nn::Conv2d>(in_channels, out_1x1, 1);
    branch1_bn_ = std::make_shared<nn::BatchNorm2d>(out_1x1);
    register_module("branch1_conv1x1", branch1_conv1x1_);
    register_module("branch1_bn", branch1_bn_);

    // Branch 2: 1x1 -> 3x3 convolution
    branch2_conv1x1_ = std::make_shared<nn::Conv2d>(in_channels, reduce_3x3, 1);
    branch2_bn1_ = std::make_shared<nn::BatchNorm2d>(reduce_3x3);
    branch2_conv3x3_ = std::make_shared<nn::Conv2d>(reduce_3x3, out_3x3, 3, 1, 1);  // padding=1
    branch2_bn2_ = std::make_shared<nn::BatchNorm2d>(out_3x3);
    register_module("branch2_conv1x1", branch2_conv1x1_);
    register_module("branch2_bn1", branch2_bn1_);
    register_module("branch2_conv3x3", branch2_conv3x3_);
    register_module("branch2_bn2", branch2_bn2_);

    // Branch 3: 1x1 -> 5x5 convolution
    branch3_conv1x1_ = std::make_shared<nn::Conv2d>(in_channels, reduce_5x5, 1);
    branch3_bn1_ = std::make_shared<nn::BatchNorm2d>(reduce_5x5);
    branch3_conv5x5_ = std::make_shared<nn::Conv2d>(reduce_5x5, out_5x5, 5, 1, 2);  // padding=2
    branch3_bn2_ = std::make_shared<nn::BatchNorm2d>(out_5x5);
    register_module("branch3_conv1x1", branch3_conv1x1_);
    register_module("branch3_bn1", branch3_bn1_);
    register_module("branch3_conv5x5", branch3_conv5x5_);
    register_module("branch3_bn2", branch3_bn2_);

    // Branch 4: max pool -> 1x1 convolution
    branch4_pool_ = std::make_shared<nn::MaxPool2d>(3, 1, 1);  // kernel=3, stride=1, padding=1
    branch4_conv1x1_ = std::make_shared<nn::Conv2d>(in_channels, out_pool_proj, 1);
    branch4_bn_ = std::make_shared<nn::BatchNorm2d>(out_pool_proj);
    register_module("branch4_pool", branch4_pool_);
    register_module("branch4_conv1x1", branch4_conv1x1_);
    register_module("branch4_bn", branch4_bn_);
}

auto InceptionModule::forward_impl(const Variable& x) -> Variable {
    // Branch 1: 1x1
    auto b1 = branch1_conv1x1_->forward(x);
    b1 = branch1_bn_->forward(b1);
    b1 = nn::relu(b1);

    // Branch 2: 1x1 -> 3x3
    auto b2 = branch2_conv1x1_->forward(x);
    b2 = branch2_bn1_->forward(b2);
    b2 = nn::relu(b2);
    b2 = branch2_conv3x3_->forward(b2);
    b2 = branch2_bn2_->forward(b2);
    b2 = nn::relu(b2);

    // Branch 3: 1x1 -> 5x5
    auto b3 = branch3_conv1x1_->forward(x);
    b3 = branch3_bn1_->forward(b3);
    b3 = nn::relu(b3);
    b3 = branch3_conv5x5_->forward(b3);
    b3 = branch3_bn2_->forward(b3);
    b3 = nn::relu(b3);

    // Branch 4: pool -> 1x1
    auto b4 = branch4_pool_->forward(x);
    b4 = branch4_conv1x1_->forward(b4);
    b4 = branch4_bn_->forward(b4);
    b4 = nn::relu(b4);

    // Concatenate along channel dimension (dim=1) using Variable-level cat for autograd
    std::vector<Variable> outputs = {b1, b2, b3, b4};
    return tenzor::cat(outputs, 1);
}

// ============================================================================
// InceptionAux Implementation
// ============================================================================

InceptionAux::InceptionAux(int64_t in_channels, int64_t num_classes, double dropout) {
    // Average pooling: 5x5, stride 3
    avgpool_ = std::make_shared<nn::AvgPool2d>(5, 3);
    register_module("avgpool", avgpool_);

    // 1x1 convolution to 128 channels
    conv_ = std::make_shared<nn::Conv2d>(in_channels, 128, 1);
    bn_ = std::make_shared<nn::BatchNorm2d>(128);
    register_module("conv", conv_);
    register_module("bn", bn_);

    // FC layers
    fc1_ = std::make_shared<nn::Linear>(128 * 4 * 4, 1024);  // Assuming 4x4 spatial size after pooling
    fc2_ = std::make_shared<nn::Linear>(1024, num_classes);
    dropout_ = std::make_shared<nn::Dropout>(dropout);

    register_module("fc1", fc1_);
    register_module("fc2", fc2_);
    register_module("dropout", dropout_);
}

auto InceptionAux::forward_impl(const Variable& x) -> Variable {
    // Average pooling
    auto pooled = avgpool_->forward(x);

    // 1x1 convolution
    auto conv_out = conv_->forward(pooled);
    conv_out = bn_->forward(conv_out);
    conv_out = nn::relu(conv_out);

    // Flatten using autograd-aware reshape
    auto& conv_out_tensor = conv_out.tensor();
    auto batch_size = conv_out_tensor.shape()[0];
    int64_t flat_size = 1;
    for (size_t i = 1; i < conv_out_tensor.shape().size(); ++i) {
        flat_size *= conv_out_tensor.shape()[i];
    }
    auto flat = conv_out.reshape({batch_size, flat_size});

    // FC1
    auto fc1_out = fc1_->forward(flat);
    fc1_out = nn::relu(fc1_out);
    fc1_out = dropout_->forward(fc1_out);

    // FC2
    return fc2_->forward(fc1_out);
}

// ============================================================================
// GoogLeNet Implementation
// ============================================================================

GoogLeNet::GoogLeNet(int64_t num_classes, bool aux_logits, double dropout, bool init_weights)
    : aux_logits_(aux_logits) {
    make_layers(num_classes, aux_logits, dropout);

    if (init_weights) {
        initialize_weights();
    }
}

auto GoogLeNet::forward_impl(const Variable& x) -> Variable {
    // Initial convolutions
    auto out = conv1_->forward(x);
    out = bn1_->forward(out);
    out = nn::relu(out);
    out = maxpool1_->forward(out);

    out = conv2_->forward(out);
    out = bn2_->forward(out);
    out = nn::relu(out);

    out = conv3_->forward(out);
    out = bn3_->forward(out);
    out = nn::relu(out);
    out = maxpool2_->forward(out);

    // Inception 3a, 3b
    out = inception3a_->forward(out);
    out = inception3b_->forward(out);
    out = maxpool3_->forward(out);

    // Inception 4a-4e
    out = inception4a_->forward(out);

    // Auxiliary classifier 1 (not used during inference)
    // if (training_ && aux_logits_) { aux1_output = aux1_->forward(out); }

    out = inception4b_->forward(out);
    out = inception4c_->forward(out);
    out = inception4d_->forward(out);

    // Auxiliary classifier 2 (not used during inference)
    // if (training_ && aux_logits_) { aux2_output = aux2_->forward(out); }

    out = inception4e_->forward(out);
    out = maxpool4_->forward(out);

    // Inception 5a, 5b
    out = inception5a_->forward(out);
    out = inception5b_->forward(out);

    // Global average pooling
    out = avgpool_->forward(out);

    // Flatten using autograd-aware reshape
    auto& out_tensor = out.tensor();
    auto batch_size = out_tensor.shape()[0];
    int64_t flat_size = 1;
    for (size_t i = 1; i < out_tensor.shape().size(); ++i) {
        flat_size *= out_tensor.shape()[i];
    }
    auto flat = out.reshape({batch_size, flat_size});

    // Dropout and FC
    flat = dropout_->forward(flat);
    return fc_->forward(flat);
}

auto GoogLeNet::forward_with_aux(const Variable& x) -> std::tuple<Variable, Variable, Variable> {
    // Initial convolutions
    auto out = conv1_->forward(x);
    out = bn1_->forward(out);
    out = nn::relu(out);
    out = maxpool1_->forward(out);

    out = conv2_->forward(out);
    out = bn2_->forward(out);
    out = nn::relu(out);

    out = conv3_->forward(out);
    out = bn3_->forward(out);
    out = nn::relu(out);
    out = maxpool2_->forward(out);

    // Inception 3a, 3b
    out = inception3a_->forward(out);
    out = inception3b_->forward(out);
    out = maxpool3_->forward(out);

    // Inception 4a
    out = inception4a_->forward(out);

    // Auxiliary classifier 1
    Variable aux1_output;
    if (aux_logits_ && aux1_) {
        aux1_output = aux1_->forward(out);
    }

    // Inception 4b-4d
    out = inception4b_->forward(out);
    out = inception4c_->forward(out);
    out = inception4d_->forward(out);

    // Auxiliary classifier 2
    Variable aux2_output;
    if (aux_logits_ && aux2_) {
        aux2_output = aux2_->forward(out);
    }

    // Inception 4e, 5a, 5b
    out = inception4e_->forward(out);
    out = maxpool4_->forward(out);

    out = inception5a_->forward(out);
    out = inception5b_->forward(out);

    // Global average pooling
    out = avgpool_->forward(out);

    // Flatten using autograd-aware reshape
    auto& out_tensor = out.tensor();
    auto batch_size = out_tensor.shape()[0];
    int64_t flat_size = 1;
    for (size_t i = 1; i < out_tensor.shape().size(); ++i) {
        flat_size *= out_tensor.shape()[i];
    }
    auto flat = out.reshape({batch_size, flat_size});

    // Dropout and FC
    flat = dropout_->forward(flat);
    auto main_output = fc_->forward(flat);

    return {main_output, aux1_output, aux2_output};
}

auto GoogLeNet::make_layers(int64_t num_classes, bool aux_logits, double dropout) -> void {
    // Initial layers
    // Conv1: 3 -> 64, 7x7, stride 2, padding 3
    conv1_ = std::make_shared<nn::Conv2d>(3, 64, 7, 2, 3);
    bn1_ = std::make_shared<nn::BatchNorm2d>(64);
    maxpool1_ = std::make_shared<nn::MaxPool2d>(3, 2, 1);  // ceil_mode in PyTorch
    register_module("conv1", conv1_);
    register_module("bn1", bn1_);
    register_module("maxpool1", maxpool1_);

    // Conv2: 64 -> 64, 1x1
    conv2_ = std::make_shared<nn::Conv2d>(64, 64, 1);
    bn2_ = std::make_shared<nn::BatchNorm2d>(64);
    register_module("conv2", conv2_);
    register_module("bn2", bn2_);

    // Conv3: 64 -> 192, 3x3, padding 1
    conv3_ = std::make_shared<nn::Conv2d>(64, 192, 3, 1, 1);
    bn3_ = std::make_shared<nn::BatchNorm2d>(192);
    maxpool2_ = std::make_shared<nn::MaxPool2d>(3, 2, 1);
    register_module("conv3", conv3_);
    register_module("bn3", bn3_);
    register_module("maxpool2", maxpool2_);

    // Inception 3a: 192 -> 256 (64 + 128 + 32 + 32)
    inception3a_ = std::make_shared<InceptionModule>(192, 64, 96, 128, 16, 32, 32);
    register_module("inception3a", inception3a_);

    // Inception 3b: 256 -> 480 (128 + 192 + 96 + 64)
    inception3b_ = std::make_shared<InceptionModule>(256, 128, 128, 192, 32, 96, 64);
    maxpool3_ = std::make_shared<nn::MaxPool2d>(3, 2, 1);
    register_module("inception3b", inception3b_);
    register_module("maxpool3", maxpool3_);

    // Inception 4a: 480 -> 512 (192 + 208 + 48 + 64)
    inception4a_ = std::make_shared<InceptionModule>(480, 192, 96, 208, 16, 48, 64);
    register_module("inception4a", inception4a_);

    // Auxiliary classifier 1 (attached to inception4a)
    if (aux_logits) {
        aux1_ = std::make_shared<InceptionAux>(512, num_classes);
        register_module("aux1", aux1_);
    }

    // Inception 4b: 512 -> 512 (160 + 224 + 64 + 64)
    inception4b_ = std::make_shared<InceptionModule>(512, 160, 112, 224, 24, 64, 64);
    register_module("inception4b", inception4b_);

    // Inception 4c: 512 -> 512 (128 + 256 + 64 + 64)
    inception4c_ = std::make_shared<InceptionModule>(512, 128, 128, 256, 24, 64, 64);
    register_module("inception4c", inception4c_);

    // Inception 4d: 512 -> 528 (112 + 288 + 64 + 64)
    inception4d_ = std::make_shared<InceptionModule>(512, 112, 144, 288, 32, 64, 64);
    register_module("inception4d", inception4d_);

    // Auxiliary classifier 2 (attached to inception4d)
    if (aux_logits) {
        aux2_ = std::make_shared<InceptionAux>(528, num_classes);
        register_module("aux2", aux2_);
    }

    // Inception 4e: 528 -> 832 (256 + 320 + 128 + 128)
    inception4e_ = std::make_shared<InceptionModule>(528, 256, 160, 320, 32, 128, 128);
    maxpool4_ = std::make_shared<nn::MaxPool2d>(3, 2, 1);
    register_module("inception4e", inception4e_);
    register_module("maxpool4", maxpool4_);

    // Inception 5a: 832 -> 832 (256 + 320 + 128 + 128)
    inception5a_ = std::make_shared<InceptionModule>(832, 256, 160, 320, 32, 128, 128);
    register_module("inception5a", inception5a_);

    // Inception 5b: 832 -> 1024 (384 + 384 + 128 + 128)
    inception5b_ = std::make_shared<InceptionModule>(832, 384, 192, 384, 48, 128, 128);
    register_module("inception5b", inception5b_);

    // Global average pooling
    avgpool_ = std::make_shared<nn::AdaptiveAvgPool2d>(1, 1);
    register_module("avgpool", avgpool_);

    // Dropout and FC
    dropout_ = std::make_shared<nn::Dropout>(dropout);
    fc_ = std::make_shared<nn::Linear>(1024, num_classes);
    register_module("dropout", dropout_);
    register_module("fc", fc_);
}

auto GoogLeNet::initialize_weights() -> void {
    // Weights are already initialized by individual layers
}

// Factory function

auto googlenet(int64_t num_classes, bool pretrained, bool aux_logits) -> std::shared_ptr<GoogLeNet> {
    auto model = std::make_shared<GoogLeNet>(num_classes, aux_logits);

    if (pretrained) {
        auto path = ModelHub::download_pretrained_safetensors("googlenet");
        ModelHub::load_pretrained_weights(*model, path, /*strict=*/false);
    }

    return model;
}

} // namespace models
} // namespace tenzor
