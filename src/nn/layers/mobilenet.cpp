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
#include <algorithm>

namespace tenzor::nn {

namespace {

// Choose a Squeeze-and-Excitation reduction ratio that (a) targets the standard
// ~hidden_dim/4 reduced-channel count and (b) exactly divides hidden_dim, which
// SqueezeExcitation's constructor requires. When hidden_dim is divisible by 4
// this returns hidden_dim/4 (identical to the historical behaviour, yielding 4
// reduced channels); otherwise it returns the largest divisor of hidden_dim not
// exceeding the desired ratio, guaranteeing channels % reduction == 0 for any
// (in_channels, expand_ratio) combination.
int64_t se_reduction_for(int64_t hidden_dim) {
    int64_t desired = std::max(int64_t(1), hidden_dim / 4);
    // Find the largest divisor of hidden_dim that is <= desired.
    for (int64_t r = desired; r >= 1; --r) {
        if (hidden_dim % r == 0) {
            return r;
        }
    }
    return 1;  // 1 always divides hidden_dim (unreachable given r=1 above).
}

} // namespace

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

auto SqueezeExcitation::forward_impl(const Variable& input) -> Variable {
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

    // Second FC + sigmoid. Use the stateless functional sigmoid instead of
    // constructing/destroying a throwaway nn::Sigmoid module every forward.
    x = fc2_->forward(x);
    x = sigmoid(x);

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

    // 3. Squeeze-and-Excitation (optional). SE recalibrates the EXPANDED feature
    // map (hidden_dim channels) and must run BEFORE the pointwise projection, so
    // when use_se the projection moves into a separate module applied after SE.
    if (use_se) {
        // SE reduction ratio of 4 (standard for MobileNetV3/EfficientNet),
        // adjusted to a divisor of hidden_dim so the SE ctor never throws.
        int64_t se_reduction = se_reduction_for(hidden_dim);
        se_ = std::make_shared<SqueezeExcitation>(hidden_dim, se_reduction, activation);
        register_module("se", se_);

        // 4a. Pointwise projection (hidden_dim -> out_channels), applied after SE.
        auto proj = std::make_shared<Sequential>();
        proj->add_module(std::make_shared<Conv2d>(hidden_dim, out_channels, 1, 1, 0));
        proj->add_module(std::make_shared<BatchNorm2d>(out_channels));
        proj_ = proj;
        register_module("proj", proj_);
    } else {
        // 4b. Pointwise projection appended to the main path (no SE in between).
        // NO activation (linear bottleneck).
        layers->add_module(std::make_shared<Conv2d>(hidden_dim, out_channels, 1, 1, 0));
        layers->add_module(std::make_shared<BatchNorm2d>(out_channels));
    }

    conv_ = layers;
    register_module("conv", conv_);
}

auto InvertedResidual::forward_impl(const Variable& input) -> Variable {
    // conv_ ends at hidden_dim when use_se (projection deferred), else out_channels.
    auto x = conv_->forward(input);

    // SE recalibrates the expanded tensor, then project down to out_channels.
    if (se_) {
        x = se_->forward(x);
        x = proj_->forward(x);
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

        // 2. Squeeze-and-Excitation (optional). SE recalibrates the expanded
        // (hidden_dim) tensor and must run BEFORE the projection, so when use_se
        // the projection moves into a separate post-SE module.
        if (use_se) {
            // SE reduction ratio of 4, adjusted to a divisor of hidden_dim so
            // the SE ctor never throws on non-multiple-of-4 hidden_dim.
            int64_t se_reduction = se_reduction_for(hidden_dim);
            se_ = std::make_shared<SqueezeExcitation>(hidden_dim, se_reduction, activation);
            register_module("se", se_);

            // 3a. Pointwise projection applied after SE (linear bottleneck).
            auto proj = std::make_shared<Sequential>();
            proj->add_module(std::make_shared<Conv2d>(hidden_dim, out_channels, 1, 1, 0));
            proj->add_module(std::make_shared<BatchNorm2d>(out_channels));
            proj_ = proj;
            register_module("proj", proj_);
        } else {
            // 3b. Pointwise projection: hidden_dim -> out_channels
            // NO activation (linear bottleneck)
            layers->add_module(std::make_shared<Conv2d>(hidden_dim, out_channels, 1, 1, 0));
            layers->add_module(std::make_shared<BatchNorm2d>(out_channels));
        }
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

auto FusedMBConv::forward_impl(const Variable& input) -> Variable {
    // conv_ ends at hidden_dim when use_se (projection deferred), else out_channels.
    auto x = conv_->forward(input);

    // SE recalibrates the expanded tensor, then project down to out_channels.
    if (se_) {
        x = se_->forward(x);
        x = proj_->forward(x);
    }

    // Add skip connection if conditions are met
    if (use_residual_) {
        x = x + input;
    }

    return x;
}

} // namespace tenzor::nn
