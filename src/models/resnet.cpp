/**
 * @file resnet.cpp
 * @brief Implementation of ResNet family
 */

#include "../../include/tenzor/models/resnet.hpp"
#include "../../include/tenzor/autograd/ops.hpp"
#include <cmath>
#include <stdexcept>

namespace tenzor {
namespace models {

// ============================================================================
// BasicBlock Implementation
// ============================================================================

BasicBlock::BasicBlock(int64_t in_channels,
                       int64_t out_channels,
                       int64_t stride,
                       int64_t groups,
                       int64_t base_width,
                       std::shared_ptr<nn::Module> downsample)
    : downsample_(downsample) {

    // BasicBlock doesn't support groups or base_width != 64
    if (groups != 1 || base_width != 64) {
        throw std::invalid_argument("BasicBlock only supports groups=1 and base_width=64");
    }

    // First 3x3 convolution
    conv1_ = std::make_shared<nn::Conv2d>(
        in_channels, out_channels, 3, stride, 1, 1, 1, false);
    register_module("conv1", conv1_);

    bn1_ = std::make_shared<nn::BatchNorm2d>(out_channels);
    register_module("bn1", bn1_);

    // Second 3x3 convolution
    conv2_ = std::make_shared<nn::Conv2d>(
        out_channels, out_channels, 3, 1, 1, 1, 1, false);
    register_module("conv2", conv2_);

    bn2_ = std::make_shared<nn::BatchNorm2d>(out_channels);
    register_module("bn2", bn2_);

    if (downsample_) {
        register_module("downsample", downsample_);
    }
}

auto BasicBlock::forward_impl(const Variable& input) -> Variable {
    Variable identity = input;

    // First conv block: conv -> bn -> relu
    auto out = conv1_->forward(input);
    out = bn1_->forward(out);
    out = relu_.forward(out);

    // Second conv block: conv -> bn
    out = conv2_->forward(out);
    out = bn2_->forward(out);

    // Apply downsampling to identity if needed
    if (downsample_) {
        identity = downsample_->forward(input);
    }

    // Add skip connection
    out = out + identity;

    // Final ReLU
    out = relu_.forward(out);

    return out;
}

// ============================================================================
// Bottleneck Implementation
// ============================================================================

Bottleneck::Bottleneck(int64_t in_channels,
                       int64_t out_channels,
                       int64_t stride,
                       int64_t groups,
                       int64_t base_width,
                       std::shared_ptr<nn::Module> downsample,
                       int64_t dilation)
    : downsample_(downsample) {

    // Calculate width for grouped convolutions (ResNeXt)
    int64_t width = static_cast<int64_t>(out_channels * (base_width / 64.0)) * groups;

    // 1x1 convolution to reduce dimensions
    conv1_ = std::make_shared<nn::Conv2d>(
        in_channels, width, 1, 1, 0, 1, 1, false);
    register_module("conv1", conv1_);

    bn1_ = std::make_shared<nn::BatchNorm2d>(width);
    register_module("bn1", bn1_);

    // Audit G11: 3×3 conv with optional atrous dilation. Padding = dilation
    // is the standard convention that preserves spatial size for stride=1
    // atrous convolutions used in DeepLabV3+'s output_stride=16/8 paths.
    conv2_ = std::make_shared<nn::Conv2d>(
        width, width, 3, stride, /*padding=*/dilation, /*dilation=*/dilation,
        groups, false);
    register_module("conv2", conv2_);

    bn2_ = std::make_shared<nn::BatchNorm2d>(width);
    register_module("bn2", bn2_);

    // 1x1 convolution to restore dimensions (with expansion)
    conv3_ = std::make_shared<nn::Conv2d>(
        width, out_channels * expansion, 1, 1, 0, 1, 1, false);
    register_module("conv3", conv3_);

    bn3_ = std::make_shared<nn::BatchNorm2d>(out_channels * expansion);
    register_module("bn3", bn3_);

    if (downsample_) {
        register_module("downsample", downsample_);
    }
}

auto Bottleneck::forward_impl(const Variable& input) -> Variable {
    Variable identity = input;

    // First conv block: 1x1 conv -> bn -> relu
    auto out = conv1_->forward(input);
    out = bn1_->forward(out);
    out = relu_.forward(out);

    // Second conv block: 3x3 conv -> bn -> relu
    out = conv2_->forward(out);
    out = bn2_->forward(out);
    out = relu_.forward(out);

    // Third conv block: 1x1 conv -> bn
    out = conv3_->forward(out);
    out = bn3_->forward(out);

    // Apply downsampling to identity if needed
    if (downsample_) {
        identity = downsample_->forward(input);
    }

    // Add skip connection
    out = out + identity;

    // Final ReLU
    out = relu_.forward(out);

    return out;
}

// ============================================================================
// ResNet Implementation
// ============================================================================

ResNet::ResNet(const std::vector<int64_t>& layers,
               int64_t num_classes,
               bool use_basic_block,
               int64_t groups,
               int64_t width_per_group,
               int64_t output_stride)
    : use_basic_block_(use_basic_block), groups_(groups), base_width_(width_per_group) {

    if (layers.size() != 4) {
        throw std::invalid_argument("ResNet requires exactly 4 layer specifications");
    }

    // Initial convolution: 7x7, stride 2
    conv1_ = std::make_shared<nn::Conv2d>(3, 64, 7, 2, 3, 1, 1, false);
    register_module("conv1", conv1_);

    bn1_ = std::make_shared<nn::BatchNorm2d>(64);
    register_module("bn1", bn1_);

    // Max pooling: 3x3, stride 2
    maxpool_ = std::make_shared<nn::MaxPool2d>(3, 2, 1);
    register_module("maxpool", maxpool_);

    // Four residual layer groups.
    // Audit G11: when `output_stride` < 32 (DeepLab-style atrous), replace
    // the layer3/layer4 stride-2 downsamples with stride-1 + dilated 3×3
    // convs so the receptive field stays the same but the spatial size
    // doesn't shrink. Only Bottleneck variants support atrous; BasicBlock
    // (ResNet-18/34) is rarely used with atrous decoders so we reject
    // non-default output_stride for the BasicBlock path.
    if (use_basic_block_) {
        if (output_stride != 32) {
            throw std::invalid_argument(
                "ResNet (BasicBlock): atrous output_stride < 32 not supported. "
                "Use a Bottleneck variant (resnet50/101/152) for DeepLab-style atrous.");
        }
        layer1_ = make_layer_basic(64, layers[0], 1);
        layer2_ = make_layer_basic(128, layers[1], 2);
        layer3_ = make_layer_basic(256, layers[2], 2);
        layer4_ = make_layer_basic(512, layers[3], 2);
    } else {
        // Map output_stride → (layer3_stride, layer3_dilation, layer4_stride, layer4_dilation).
        // Stem (conv1+maxpool) contributes stride 4. layer1 keeps stride 1.
        // layer2 always strides by 2 (giving stride 8 there). Then:
        //   output_stride=32: layer3 stride=2 (→16), layer4 stride=2 (→32). dilations=1.
        //   output_stride=16: layer3 stride=2 (→16), layer4 stride=1 dilation=2 (stays 16).
        //   output_stride=8:  layer3 stride=1 dilation=2 (stays 8), layer4 stride=1 dilation=4.
        int64_t l3_stride, l3_dilation, l4_stride, l4_dilation;
        switch (output_stride) {
            case 32: l3_stride = 2; l3_dilation = 1; l4_stride = 2; l4_dilation = 1; break;
            case 16: l3_stride = 2; l3_dilation = 1; l4_stride = 1; l4_dilation = 2; break;
            case  8: l3_stride = 1; l3_dilation = 2; l4_stride = 1; l4_dilation = 4; break;
            default:
                throw std::invalid_argument(
                    "ResNet: output_stride must be 8, 16, or 32 (got " +
                    std::to_string(output_stride) + ")");
        }
        layer1_ = make_layer_bottleneck(64, layers[0], 1);
        layer2_ = make_layer_bottleneck(128, layers[1], 2);
        layer3_ = make_layer_bottleneck(256, layers[2], l3_stride, l3_dilation);
        layer4_ = make_layer_bottleneck(512, layers[3], l4_stride, l4_dilation);
    }

    register_module("layer1", layer1_);
    register_module("layer2", layer2_);
    register_module("layer3", layer3_);
    register_module("layer4", layer4_);

    // Global average pooling to 1x1
    avgpool_ = std::make_shared<nn::AdaptiveAvgPool2d>(1);
    register_module("avgpool", avgpool_);

    // Fully connected layer for classification
    // BasicBlock has expansion=1, Bottleneck has expansion=4
    int64_t final_channels = use_basic_block_ ? 512 : 512 * 4;
    fc_ = std::make_shared<nn::Linear>(final_channels, num_classes);
    register_module("fc", fc_);
}

auto ResNet::make_layer_basic(int64_t out_channels, int64_t num_blocks, int64_t stride)
    -> std::shared_ptr<nn::Sequential> {

    std::shared_ptr<nn::Module> downsample = nullptr;

    // Create downsample module if needed (BasicBlock has expansion=1)
    if (stride != 1 || in_channels_ != out_channels) {
        auto downsample_seq = std::make_shared<nn::Sequential>();

        auto conv = std::make_shared<nn::Conv2d>(
            in_channels_, out_channels, 1, stride, 0, 1, 1, false);
        auto bn = std::make_shared<nn::BatchNorm2d>(out_channels);

        downsample_seq->add_module(conv);
        downsample_seq->add_module(bn);

        downsample = downsample_seq;
    }

    // Build layer with multiple blocks
    auto layer = std::make_shared<nn::Sequential>();

    // First block (may have stride != 1 and/or downsampling)
    auto first_block = std::make_shared<BasicBlock>(
        in_channels_, out_channels, stride, groups_, base_width_, downsample);
    layer->add_module(first_block);

    in_channels_ = out_channels;  // BasicBlock expansion = 1

    // Remaining blocks (stride = 1, no downsampling)
    for (int64_t i = 1; i < num_blocks; ++i) {
        auto block = std::make_shared<BasicBlock>(
            in_channels_, out_channels, 1, groups_, base_width_, nullptr);
        layer->add_module(block);
    }

    return layer;
}

auto ResNet::make_layer_bottleneck(int64_t out_channels, int64_t num_blocks,
                                    int64_t stride, int64_t dilation)
    -> std::shared_ptr<nn::Sequential> {

    std::shared_ptr<nn::Module> downsample = nullptr;

    // Create downsample module if needed (Bottleneck has expansion=4).
    // The 1×1 downsample uses stride only — no dilation needed (it operates
    // on a single-pixel kernel).
    if (stride != 1 || in_channels_ != out_channels * 4) {
        auto downsample_seq = std::make_shared<nn::Sequential>();

        auto conv = std::make_shared<nn::Conv2d>(
            in_channels_, out_channels * 4, 1, stride, 0, 1, 1, false);
        auto bn = std::make_shared<nn::BatchNorm2d>(out_channels * 4);

        downsample_seq->add_module(conv);
        downsample_seq->add_module(bn);

        downsample = downsample_seq;
    }

    // Build layer with multiple blocks
    auto layer = std::make_shared<nn::Sequential>();

    // First block (may have stride != 1 and/or downsampling). Atrous applies
    // to every block in the layer — the receptive field is preserved by the
    // dilated 3×3 conv inside each Bottleneck.
    auto first_block = std::make_shared<Bottleneck>(
        in_channels_, out_channels, stride, groups_, base_width_, downsample,
        dilation);
    layer->add_module(first_block);

    in_channels_ = out_channels * 4;  // Bottleneck expansion = 4

    // Remaining blocks (stride = 1, no downsampling, same dilation).
    for (int64_t i = 1; i < num_blocks; ++i) {
        auto block = std::make_shared<Bottleneck>(
            in_channels_, out_channels, 1, groups_, base_width_, nullptr,
            dilation);
        layer->add_module(block);
    }

    return layer;
}

auto ResNet::forward_impl(const Variable& input) -> Variable {
    // Extract features through all residual layers
    auto x = forward_features(input);

    // Global average pooling
    x = avgpool_->forward(x);

    // Flatten to (N, C) using autograd-aware reshape
    auto shape = x.tensor().shape();
    x = tenzor::reshape(x, std::vector<int64_t>{shape[0], -1});

    // Fully connected layer
    x = fc_->forward(x);

    return x;
}

auto ResNet::forward_features(const Variable& input) -> Variable {
    // Initial conv: conv -> bn -> relu -> maxpool
    auto x = conv1_->forward(input);
    x = bn1_->forward(x);
    x = relu_.forward(x);
    x = maxpool_->forward(x);

    // Four residual layer groups
    x = layer1_->forward(x);
    x = layer2_->forward(x);
    x = layer3_->forward(x);
    x = layer4_->forward(x);

    // Return feature maps before pooling and classification
    return x;
}

auto ResNet::forward_features_multi(const Variable& input)
    -> std::tuple<Variable, Variable, Variable, Variable> {
    // Audit G6: expose intermediate stage outputs so FPN-style decoders can
    // build P2-P5 via lateral + top-down connections.
    auto x = conv1_->forward(input);
    x = bn1_->forward(x);
    x = relu_.forward(x);
    x = maxpool_->forward(x);

    auto c2 = layer1_->forward(x);   // stride 4
    auto c3 = layer2_->forward(c2);  // stride 8
    auto c4 = layer3_->forward(c3);  // stride 16
    auto c5 = layer4_->forward(c4);  // stride 32
    return std::make_tuple(c2, c3, c4, c5);
}

auto ResNet::load_pretrained(const std::string& path) -> void {
    // Load pretrained weights from file
    // This would deserialize weights and call load_state_dict
    load(path);
}

// ============================================================================
// Factory Functions
// ============================================================================

auto resnet18(int64_t num_classes, bool pretrained) -> std::shared_ptr<ResNet> {
    auto model = std::make_shared<ResNet>(
        std::vector<int64_t>{2, 2, 2, 2}, num_classes, true, 1, 64);

    if (pretrained) {
        model->load_pretrained("resnet18_imagenet.pth");
    }

    return model;
}

auto resnet34(int64_t num_classes, bool pretrained) -> std::shared_ptr<ResNet> {
    auto model = std::make_shared<ResNet>(
        std::vector<int64_t>{3, 4, 6, 3}, num_classes, true, 1, 64);

    if (pretrained) {
        model->load_pretrained("resnet34_imagenet.pth");
    }

    return model;
}

auto resnet50(int64_t num_classes, bool pretrained) -> std::shared_ptr<ResNet> {
    auto model = std::make_shared<ResNet>(
        std::vector<int64_t>{3, 4, 6, 3}, num_classes, false, 1, 64);

    if (pretrained) {
        model->load_pretrained("resnet50_imagenet.pth");
    }

    return model;
}

auto resnet101(int64_t num_classes, bool pretrained) -> std::shared_ptr<ResNet> {
    auto model = std::make_shared<ResNet>(
        std::vector<int64_t>{3, 4, 23, 3}, num_classes, false, 1, 64);

    if (pretrained) {
        model->load_pretrained("resnet101_imagenet.pth");
    }

    return model;
}

auto resnet152(int64_t num_classes, bool pretrained) -> std::shared_ptr<ResNet> {
    auto model = std::make_shared<ResNet>(
        std::vector<int64_t>{3, 8, 36, 3}, num_classes, false, 1, 64);

    if (pretrained) {
        model->load_pretrained("resnet152_imagenet.pth");
    }

    return model;
}

// ----------------------------------------------------------------------------
// Audit G11: atrous-modified ResNet factories for DeepLab-style decoders.
// Same layer counts as the base variants, but layer3/layer4 are constructed
// with the dilation pattern matching the requested output_stride.
// ----------------------------------------------------------------------------
auto resnet50_atrous(int64_t num_classes, int64_t output_stride, bool pretrained)
    -> std::shared_ptr<ResNet> {
    auto model = std::make_shared<ResNet>(
        std::vector<int64_t>{3, 4, 6, 3}, num_classes, false, 1, 64, output_stride);
    if (pretrained) model->load_pretrained("resnet50_imagenet.pth");
    return model;
}

auto resnet101_atrous(int64_t num_classes, int64_t output_stride, bool pretrained)
    -> std::shared_ptr<ResNet> {
    auto model = std::make_shared<ResNet>(
        std::vector<int64_t>{3, 4, 23, 3}, num_classes, false, 1, 64, output_stride);
    if (pretrained) model->load_pretrained("resnet101_imagenet.pth");
    return model;
}

auto resnet152_atrous(int64_t num_classes, int64_t output_stride, bool pretrained)
    -> std::shared_ptr<ResNet> {
    auto model = std::make_shared<ResNet>(
        std::vector<int64_t>{3, 8, 36, 3}, num_classes, false, 1, 64, output_stride);
    if (pretrained) model->load_pretrained("resnet152_imagenet.pth");
    return model;
}

auto resnext50_32x4d(int64_t num_classes, bool pretrained) -> std::shared_ptr<ResNet> {
    auto model = std::make_shared<ResNet>(
        std::vector<int64_t>{3, 4, 6, 3}, num_classes, false, 32, 4);

    if (pretrained) {
        model->load_pretrained("resnext50_32x4d_imagenet.pth");
    }

    return model;
}

auto resnext101_32x8d(int64_t num_classes, bool pretrained) -> std::shared_ptr<ResNet> {
    auto model = std::make_shared<ResNet>(
        std::vector<int64_t>{3, 4, 23, 3}, num_classes, false, 32, 8);

    if (pretrained) {
        model->load_pretrained("resnext101_32x8d_imagenet.pth");
    }

    return model;
}

auto wide_resnet50_2(int64_t num_classes, bool pretrained) -> std::shared_ptr<ResNet> {
    auto model = std::make_shared<ResNet>(
        std::vector<int64_t>{3, 4, 6, 3}, num_classes, false, 1, 128);

    if (pretrained) {
        model->load_pretrained("wide_resnet50_2_imagenet.pth");
    }

    return model;
}

auto wide_resnet101_2(int64_t num_classes, bool pretrained) -> std::shared_ptr<ResNet> {
    auto model = std::make_shared<ResNet>(
        std::vector<int64_t>{3, 4, 23, 3}, num_classes, false, 1, 128);

    if (pretrained) {
        model->load_pretrained("wide_resnet101_2_imagenet.pth");
    }

    return model;
}

} // namespace models
} // namespace tenzor
