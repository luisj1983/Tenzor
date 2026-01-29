/**
 * @file segmentation.cpp
 * @brief Implementation of segmentation-specific layers
 */

#include "tenzor/nn/layers/segmentation.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace tenzor {
namespace nn {

// ============================================================================
// AtrousSeparableConv2d Implementation
// ============================================================================

AtrousSeparableConv2d::AtrousSeparableConv2d(int64_t in_channels,
                                             int64_t out_channels,
                                             int64_t kernel_size,
                                             int64_t dilation,
                                             bool bias)
{
    // Calculate padding to maintain spatial dimensions
    // padding = dilation * (kernel_size - 1) / 2
    int64_t padding = dilation * (kernel_size - 1) / 2;

    // Depthwise atrous convolution (groups = in_channels)
    depthwise_ = std::make_shared<Conv2d>(in_channels, in_channels, kernel_size,
                                1, padding, dilation, in_channels, bias);
    register_module("depthwise", depthwise_);

    bn1_ = std::make_shared<BatchNorm2d>(in_channels);
    register_module("bn1", bn1_);

    // Pointwise 1x1 convolution
    pointwise_ = std::make_shared<Conv2d>(in_channels, out_channels, 1,
                                1, 0, 1, 1, bias);
    register_module("pointwise", pointwise_);

    bn2_ = std::make_shared<BatchNorm2d>(out_channels);
    register_module("bn2", bn2_);
}

auto AtrousSeparableConv2d::forward_impl(const Variable& input) -> Variable {
    // Depthwise atrous convolution
    auto x = depthwise_->forward(input);
    x = bn1_->forward(x);
    x = relu_.forward(x);

    // Pointwise convolution
    x = pointwise_->forward(x);
    x = bn2_->forward(x);
    x = relu_.forward(x);

    return x;
}

// ============================================================================
// ASPP Implementation
// ============================================================================

ASPP::ASPP(int64_t in_channels,
           int64_t out_channels,
           std::vector<int64_t> atrous_rates,
           bool use_separable,
           float dropout_rate)
    : out_channels_(out_channels)
{
    if (atrous_rates.size() != 3) {
        throw std::invalid_argument("ASPP requires exactly 3 atrous rates");
    }

    // Branch 1: 1×1 convolution
    conv1x1_ = make_conv_bn_relu(in_channels, out_channels, 1);
    register_module("conv1x1", conv1x1_);

    // Branches 2-4: Atrous convolutions with different rates
    for (size_t i = 0; i < atrous_rates.size(); ++i) {
        int64_t rate = atrous_rates[i];
        std::shared_ptr<Module> conv;

        if (use_separable) {
            // Use atrous separable convolution for efficiency
            conv = std::make_shared<AtrousSeparableConv2d>(
                in_channels, out_channels, 3, rate);
        } else {
            // Use standard atrous convolution
            conv = make_conv_bn_relu(in_channels, out_channels, 3, 1,
                                    rate * (3 - 1) / 2, rate);
        }

        std::string name = "atrous_conv" + std::to_string(i);
        register_module(name, conv);
        atrous_convs_.push_back(conv);
    }

    // Branch 5: Global average pooling
    global_pool_ = std::make_shared<AdaptiveAvgPool2d>(1, 1);
    register_module("global_pool", global_pool_);

    global_conv_ = make_conv_bn_relu(in_channels, out_channels, 1);
    register_module("global_conv", global_conv_);

    // Fusion layer: Concatenate all branches (5 * out_channels) -> out_channels
    auto project_conv = std::make_shared<Conv2d>(
        out_channels * 5, out_channels, 1, 1, 0);
    auto project_bn = std::make_shared<BatchNorm2d>(out_channels);
    auto project_relu = std::make_shared<ReLU>();
    auto project_dropout = std::make_shared<Dropout>(dropout_rate);

    // Create sequential project module manually
    class ProjectModule : public Module {
    public:
        ProjectModule(std::shared_ptr<Conv2d> conv,
                     std::shared_ptr<BatchNorm2d> bn,
                     std::shared_ptr<ReLU> relu,
                     std::shared_ptr<Dropout> dropout)
            : conv_(conv), bn_(bn), relu_(relu), dropout_(dropout) {}

        auto forward_impl(const Variable& input) -> Variable override {
            auto x = conv_->forward(input);
            x = bn_->forward(x);
            x = relu_->forward(x);
            x = dropout_->forward(x);
            return x;
        }

    private:
        std::shared_ptr<Conv2d> conv_;
        std::shared_ptr<BatchNorm2d> bn_;
        std::shared_ptr<ReLU> relu_;
        std::shared_ptr<Dropout> dropout_;
    };

    project_ = std::make_shared<ProjectModule>(project_conv, project_bn,
                                       project_relu, project_dropout);
    register_module("project", project_);
}

auto ASPP::forward_impl(const Variable& input) -> Variable {
    const auto& shape = input.tensor().shape();
    if (shape.size() != 4) {
        throw std::runtime_error("ASPP expects 4D input (N, C, H, W)");
    }

    int64_t H = shape[2];
    int64_t W = shape[3];

    // Branch 1: 1×1 convolution
    auto feat1 = conv1x1_->forward(input);

    // Branches 2-4: Atrous convolutions
    std::vector<Variable> features = {feat1};
    for (auto& conv : atrous_convs_) {
        features.push_back(conv->forward(input));
    }

    // Branch 5: Global pooling
    auto feat5 = global_pool_->forward(input);  // [N, C, 1, 1]
    feat5 = global_conv_->forward(feat5);       // [N, out_channels, 1, 1]
    feat5 = upsample_bilinear(feat5, H, W);     // [N, out_channels, H, W]
    features.push_back(feat5);

    // Concatenate all branches along channel dimension using autograd-aware cat
    auto concat = tenzor::cat(features, 1);  // [N, out_channels * 5, H, W]

    // Project to output channels
    auto output = project_->forward(concat);  // [N, out_channels, H, W]

    return output;
}

// ============================================================================
// Helper Functions
// ============================================================================

auto upsample_bilinear(const Variable& input, int64_t target_h, int64_t target_w)
    -> Variable
{
    const auto& shape = input.tensor().shape();
    if (shape.size() != 4) {
        throw std::runtime_error("upsample_bilinear expects 4D input (N, C, H, W)");
    }

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H_in = shape[2];
    int64_t W_in = shape[3];

    // Use backend dispatch for bilinear interpolation (avoids GPU→CPU→GPU round-trip)
    OpAttributes interp_attrs;
    interp_attrs["mode"] = "bilinear";
    interp_attrs["size"] = std::to_string(target_h) + "," + std::to_string(target_w);
    interp_attrs["align_corners"] = "0";

    std::vector<Tensor> interp_inputs = {input.tensor()};
    auto interp_results = dispatch(OpId::Interpolate, interp_inputs, interp_attrs);
    Tensor output = interp_results[0];

    // Create Variable with gradient tracking if needed
    Variable result(output, input.requires_grad());

    // Set up autograd if gradients are needed
    if (input.requires_grad() && is_grad_enabled()) {
        // Create backward function
        auto grad_fn = std::make_shared<UpsampleBilinearBackward>(
            H_in, W_in, target_h, target_w
        );

        // Save input tensor for backward pass
        grad_fn->save_for_backward({input.tensor()});

        // Set up backward graph
        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.push_back(input.grad_fn());  // nullptr if input is leaf
        grad_fn->set_next_functions(next_funcs);

        // Track input variable for gradient accumulation
        std::vector<Variable> input_vars;
        if (input.requires_grad()) {
            input_vars.push_back(input);
        }
        grad_fn->set_input_variables(input_vars);

        // Attach gradient function to result
        result.set_grad_fn(grad_fn);
    }

    return result;
}

auto make_conv_bn_relu(int64_t in_channels,
                       int64_t out_channels,
                       int64_t kernel_size,
                       int64_t stride,
                       int64_t padding,
                       int64_t dilation,
                       int64_t groups)
    -> std::shared_ptr<Module>
{
    class ConvBnRelu : public Module {
    public:
        ConvBnRelu(int64_t in_channels,
                   int64_t out_channels,
                   int64_t kernel_size,
                   int64_t stride,
                   int64_t padding,
                   int64_t dilation,
                   int64_t groups)
        {
            conv_ = std::make_shared<Conv2d>(in_channels, out_channels, kernel_size,
                                        stride, padding, dilation, groups, false);
            register_module("conv", conv_);
            bn_ = std::make_shared<BatchNorm2d>(out_channels);
            register_module("bn", bn_);
        }

        auto forward_impl(const Variable& input) -> Variable override {
            auto x = conv_->forward(input);
            x = bn_->forward(x);
            x = relu_.forward(x);
            return x;
        }

    private:
        std::shared_ptr<Conv2d> conv_;
        std::shared_ptr<BatchNorm2d> bn_;
        ReLU relu_;
    };

    return std::make_shared<ConvBnRelu>(in_channels, out_channels, kernel_size,
                                        stride, padding, dilation, groups);
}

} // namespace nn
} // namespace tenzor
