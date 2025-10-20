/**
 * @file mobilenet.cpp
 * @brief Implementation of MobileNet and EfficientNet building blocks
 */

#include "tenzor/nn/layers/mobilenet.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/pooling.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <stdexcept>
#include <memory>

namespace tenzor::nn {

// ============================================================================
// SqueezeExcitation Implementation
// ============================================================================

SqueezeExcitation::SqueezeExcitation(int64_t channels,
                                     int64_t reduction,
                                     std::string activation)
    : channels_(channels)
    , reduction_(reduction)
    , activation_(activation)
{
    if (channels % reduction != 0) {
        throw std::invalid_argument(
            "channels (" + std::to_string(channels) + ") must be divisible by reduction (" +
            std::to_string(reduction) + ")");
    }

    int64_t reduced_channels = channels / reduction;

    // Global average pooling to (1, 1)
    pool_ = std::make_shared<AdaptiveAvgPool2d>(1, 1);
    register_module("pool", pool_);

    // First FC layer: C -> C/reduction
    fc1_ = std::make_shared<Linear>(channels, reduced_channels, true);
    register_module("fc1", fc1_);

    // Activation function
    if (activation == "relu") {
        act_ = std::make_shared<ReLU>();
    } else if (activation == "swish") {
        act_ = std::make_shared<Swish>();
    } else {
        throw std::invalid_argument("Unsupported activation: " + activation);
    }
    register_module("act", act_);

    // Second FC layer: C/reduction -> C
    fc2_ = std::make_shared<Linear>(reduced_channels, channels, true);
    register_module("fc2", fc2_);
}

auto SqueezeExcitation::forward(const Variable& input) -> Variable {
    auto shape = input.tensor().shape();
    if (shape.size() != 4) {
        throw std::runtime_error("SqueezeExcitation expects 4D input (N, C, H, W)");
    }

    int64_t batch = shape[0];
    int64_t channels = shape[1];

    if (channels != channels_) {
        throw std::runtime_error(
            "Input channels (" + std::to_string(channels) + ") doesn't match expected (" +
            std::to_string(channels_) + ")");
    }

    // Global average pooling: (N, C, H, W) -> (N, C, 1, 1)
    auto x = pool_->forward(input);

    // Squeeze spatial dimensions: (N, C, 1, 1) -> (N, C)
    x = x.reshape({batch, channels});

    // First FC + activation
    x = fc1_->forward(x);
    x = act_->forward(x);

    // Second FC + sigmoid
    x = fc2_->forward(x);
    x = nn::Sigmoid().forward(x);

    // Expand back to (N, C, 1, 1) for broadcasting
    x = x.reshape({batch, channels, 1, 1});

    // Channel-wise multiplication (broadcasting over H, W)
    auto output = input * x;

    return output;
}

// ============================================================================
// InvertedResidual Implementation
// ============================================================================

InvertedResidual::InvertedResidual(int64_t in_channels,
                                   int64_t out_channels,
                                   int64_t expand_ratio,
                                   int64_t stride,
                                   bool use_se,
                                   int64_t kernel_size,
                                   std::string activation)
    : in_channels_(in_channels)
    , out_channels_(out_channels)
    , stride_(stride)
    , use_residual_(stride == 1 && in_channels == out_channels)
{
    int64_t hidden_dim = in_channels * expand_ratio;
    int64_t padding = kernel_size / 2;  // Same padding for odd kernel sizes

    // Create activation module
    std::shared_ptr<Module> act;
    if (activation == "relu") {
        act = std::make_shared<ReLU>();
    } else if (activation == "relu6") {
        act = std::make_shared<ReLU6>();
    } else if (activation == "swish") {
        act = std::make_shared<Swish>();
    } else {
        throw std::invalid_argument("Unsupported activation: " + activation);
    }

    // Build main convolution path
    auto layers = std::make_shared<Sequential>();

    // 1. Expansion (if expand_ratio != 1)
    if (expand_ratio != 1) {
        // Pointwise expansion: in_channels -> hidden_dim
        layers->add_module(std::make_shared<Conv2d>(in_channels, hidden_dim, 1, 1, 0));
        layers->add_module(std::make_shared<BatchNorm2d>(hidden_dim));
        layers->add_module(act);
    }

    // 2. Depthwise convolution
    layers->add_module(std::make_shared<Conv2d>(
        hidden_dim, hidden_dim, kernel_size, stride, padding,
        1,          // dilation
        hidden_dim  // groups = hidden_dim for depthwise
    ));
    layers->add_module(std::make_shared<BatchNorm2d>(hidden_dim));
    layers->add_module(act);

    // 3. Squeeze-and-Excitation (optional)
    if (use_se) {
        // SE reduction ratio of 4 (standard for MobileNetV3/EfficientNet)
        int64_t se_reduction = std::max(int64_t(1), hidden_dim / 4);
        se_ = std::make_shared<SqueezeExcitation>(hidden_dim, se_reduction, activation);
        register_module("se", se_);
    }

    // 4. Pointwise projection: hidden_dim -> out_channels
    // NO activation (linear bottleneck)
    layers->add_module(std::make_shared<Conv2d>(hidden_dim, out_channels, 1, 1, 0));
    layers->add_module(std::make_shared<BatchNorm2d>(out_channels));

    conv_ = layers;
    register_module("conv", conv_);
}

auto InvertedResidual::forward(const Variable& input) -> Variable {
    auto x = conv_->forward(input);

    // Apply SE if present
    if (se_) {
        x = se_->forward(x);
    }

    // Add skip connection if conditions are met
    if (use_residual_) {
        x = x + input;
    }

    return x;
}

// ============================================================================
// FusedMBConv Implementation
// ============================================================================

FusedMBConv::FusedMBConv(int64_t in_channels,
                         int64_t out_channels,
                         int64_t expand_ratio,
                         int64_t stride,
                         bool use_se,
                         std::string activation)
    : in_channels_(in_channels)
    , out_channels_(out_channels)
    , stride_(stride)
    , use_residual_(stride == 1 && in_channels == out_channels)
{
    int64_t hidden_dim = in_channels * expand_ratio;

    // Create activation module
    std::shared_ptr<Module> act;
    if (activation == "swish") {
        act = std::make_shared<Swish>();
    } else if (activation == "relu") {
        act = std::make_shared<ReLU>();
    } else if (activation == "relu6") {
        act = std::make_shared<ReLU6>();
    } else {
        throw std::invalid_argument("Unsupported activation: " + activation);
    }

    // Build fused convolution path
    auto layers = std::make_shared<Sequential>();

    if (expand_ratio != 1) {
        // 1. Fused expansion: 3x3 regular conv instead of 1x1 + 3x3 depthwise
        layers->add_module(std::make_shared<Conv2d>(
            in_channels, hidden_dim,
            3,      // kernel_size
            stride, // stride
            1       // padding (same)
        ));
        layers->add_module(std::make_shared<BatchNorm2d>(hidden_dim));
        layers->add_module(act);

        // 2. Squeeze-and-Excitation (optional)
        if (use_se) {
            int64_t se_reduction = std::max(int64_t(1), hidden_dim / 4);
            se_ = std::make_shared<SqueezeExcitation>(hidden_dim, se_reduction, activation);
            register_module("se", se_);
        }

        // 3. Pointwise projection: hidden_dim -> out_channels
        // NO activation (linear bottleneck)
        layers->add_module(std::make_shared<Conv2d>(hidden_dim, out_channels, 1, 1, 0));
        layers->add_module(std::make_shared<BatchNorm2d>(out_channels));
    } else {
        // No expansion, just 3x3 conv
        layers->add_module(std::make_shared<Conv2d>(
            in_channels, out_channels,
            3,      // kernel_size
            stride, // stride
            1       // padding
        ));
        layers->add_module(std::make_shared<BatchNorm2d>(out_channels));
        layers->add_module(act);
    }

    conv_ = layers;
    register_module("conv", conv_);
}

auto FusedMBConv::forward(const Variable& input) -> Variable {
    auto x = conv_->forward(input);

    // Apply SE if present
    if (se_) {
        x = se_->forward(x);
    }

    // Add skip connection if conditions are met
    if (use_residual_) {
        x = x + input;
    }

    return x;
}

} // namespace tenzor::nn
