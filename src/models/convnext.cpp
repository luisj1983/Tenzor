/**
 * @file convnext.cpp
 * @brief Implementation of ConvNeXt models
 */

#include "../../include/tenzor/models/convnext.hpp"
#include "../../include/tenzor/autograd/ops.hpp"
#include "../../include/tenzor/models/hub.hpp"
#include <cmath>
#include <stdexcept>

namespace tenzor {
namespace models {

// ============================================================================
// LayerScale Implementation
// ============================================================================

LayerScale::LayerScale(int64_t dim, double init_value) {
    // Initialize gamma as learnable parameter [dim, 1, 1]
    auto gamma_tensor = Tensor({dim, 1, 1}, DType::Float32, Device::cpu());
    gamma_tensor.fill_(static_cast<float>(init_value));

    gamma_ = Variable(gamma_tensor, true);  // requires_grad=true
    register_parameter("gamma", gamma_);
}

auto LayerScale::forward_impl(const Variable& input) -> Variable {
    // Multiply input by learnable scale factor
    // Convert gamma to match input dtype for multi-dtype support
    auto gamma_tensor_matched = gamma_.tensor().to(input.dtype());
    auto gamma_matched = Variable(gamma_tensor_matched, gamma_.requires_grad());
    return input * gamma_matched;
}

// ============================================================================
// ConvNeXtBlock Implementation
// ============================================================================

ConvNeXtBlock::ConvNeXtBlock(int64_t dim,
                             double drop_path,
                             double layer_scale_init_value)
    : drop_path_(drop_path) {

    // Depthwise 7×7 convolution (groups = channels for depthwise)
    dwconv_ = std::make_shared<nn::Conv2d>(
        dim, dim, 7, 1, 3, 1, dim, false);  // kernel=7, stride=1, padding=3, groups=dim
    register_module("dwconv", dwconv_);

    // LayerNorm - normalize over [C, H, W] dimensions
    // For channels_first format (NCHW), normalize over last dimension (channels)
    norm_ = std::make_shared<nn::LayerNorm>(
        std::vector<int64_t>{dim}, 1e-6, true);
    register_module("norm", norm_);

    // Pointwise 1×1 expansion: dim → 4*dim
    int64_t hidden_dim = 4 * dim;
    pwconv1_ = std::make_shared<nn::Conv2d>(
        dim, hidden_dim, 1, 1, 0, 1, 1, true);  // with bias
    register_module("pwconv1", pwconv1_);

    // Pointwise 1×1 projection: 4*dim → dim
    pwconv2_ = std::make_shared<nn::Conv2d>(
        hidden_dim, dim, 1, 1, 0, 1, 1, true);  // with bias
    register_module("pwconv2", pwconv2_);

    // Layer Scale
    gamma_ = std::make_shared<LayerScale>(dim, layer_scale_init_value);
    register_module("gamma", gamma_);
}

auto ConvNeXtBlock::forward_impl(const Variable& input) -> Variable {
    Variable shortcut = input;

    // Depthwise convolution
    auto x = dwconv_->forward(input);

    // LayerNorm
    // Need to permute from NCHW to NHWC for LayerNorm, then back
    // Permute NCHW → NHWC
    x = x.permute({0, 2, 3, 1});  // [N, H, W, C]

    // Apply LayerNorm
    x = norm_->forward(x);

    // Permute back NHWC → NCHW
    x = x.permute({0, 3, 1, 2});  // [N, C, H, W]

    // Pointwise expansion
    x = pwconv1_->forward(x);

    // GELU activation
    x = gelu_.forward(x);

    // Pointwise projection
    x = pwconv2_->forward(x);

    // Layer Scale
    x = gamma_->forward(x);

    // Stochastic depth (drop path)
    if (is_training() && drop_path_ > 0.0) {
        // Simple stochastic depth: randomly drop the entire residual branch
        auto drop_mask = Tensor({}, DType::Float32, x.device());
        // Generate random value and compare with drop_path_
        // For simplicity, we'll implement a basic version
        // In production, use proper random number generation
        if (static_cast<double>(rand()) / RAND_MAX < drop_path_) {
            // Drop the branch - return only shortcut
            return shortcut;
        } else {
            // Keep the branch, scale by survival probability
            x = x * (1.0 / (1.0 - drop_path_));
        }
    }

    // Residual connection
    return shortcut + x;
}

// ============================================================================
// ConvNeXt Implementation
// ============================================================================

ConvNeXt::ConvNeXt(int64_t in_channels,
                   int64_t num_classes,
                   const std::vector<int64_t>& depths,
                   const std::vector<int64_t>& dims,
                   double drop_path_rate,
                   double layer_scale_init_value)
    : depths_(depths), dims_(dims), layer_scale_init_value_(layer_scale_init_value) {

    if (depths.size() != 4 || dims.size() != 4) {
        throw std::invalid_argument("ConvNeXt requires exactly 4 stages");
    }

    // Stem: 4×4 conv with stride 4 (aggressive downsampling like ViT)
    // Store conv and norm separately to handle permutations in forward pass
    stem_conv_ = std::make_shared<nn::Conv2d>(
        in_channels, dims[0], 4, 4, 0, 1, 1, false);
    stem_norm_ = std::make_shared<nn::LayerNorm>(
        std::vector<int64_t>{dims[0]}, 1e-6, true);
    register_module("stem_conv", stem_conv_);
    register_module("stem_norm", stem_norm_);

    // Calculate total depth for drop path scheduling
    int64_t total_depth = 0;
    for (auto d : depths) total_depth += d;

    int64_t block_idx = 0;

    // Stage 1
    stage1_ = make_stage(dims[0], depths[0],
                         0.0,
                         drop_path_rate * (depths[0]) / total_depth);
    register_module("stage1", stage1_);
    block_idx += depths[0];

    // Downsample 1→2
    downsample1_norm_ = std::make_shared<nn::LayerNorm>(
        std::vector<int64_t>{dims[0]}, 1e-6, true);
    downsample1_conv_ = std::make_shared<nn::Sequential>();
    auto ds1_conv = std::make_shared<nn::Conv2d>(dims[0], dims[1], 2, 2, 0, 1, 1, false);
    downsample1_conv_->add_module(ds1_conv);
    register_module("downsample1_norm", downsample1_norm_);
    register_module("downsample1_conv", downsample1_conv_);

    // Stage 2
    double dp_start2 = drop_path_rate * block_idx / total_depth;
    block_idx += depths[1];
    double dp_end2 = drop_path_rate * block_idx / total_depth;
    stage2_ = make_stage(dims[1], depths[1], dp_start2, dp_end2);
    register_module("stage2", stage2_);

    // Downsample 2→3
    downsample2_norm_ = std::make_shared<nn::LayerNorm>(
        std::vector<int64_t>{dims[1]}, 1e-6, true);
    downsample2_conv_ = std::make_shared<nn::Sequential>();
    auto ds2_conv = std::make_shared<nn::Conv2d>(dims[1], dims[2], 2, 2, 0, 1, 1, false);
    downsample2_conv_->add_module(ds2_conv);
    register_module("downsample2_norm", downsample2_norm_);
    register_module("downsample2_conv", downsample2_conv_);

    // Stage 3
    double dp_start3 = drop_path_rate * block_idx / total_depth;
    block_idx += depths[2];
    double dp_end3 = drop_path_rate * block_idx / total_depth;
    stage3_ = make_stage(dims[2], depths[2], dp_start3, dp_end3);
    register_module("stage3", stage3_);

    // Downsample 3→4
    downsample3_norm_ = std::make_shared<nn::LayerNorm>(
        std::vector<int64_t>{dims[2]}, 1e-6, true);
    downsample3_conv_ = std::make_shared<nn::Sequential>();
    auto ds3_conv = std::make_shared<nn::Conv2d>(dims[2], dims[3], 2, 2, 0, 1, 1, false);
    downsample3_conv_->add_module(ds3_conv);
    register_module("downsample3_norm", downsample3_norm_);
    register_module("downsample3_conv", downsample3_conv_);

    // Stage 4
    double dp_start4 = drop_path_rate * block_idx / total_depth;
    double dp_end4 = drop_path_rate;
    stage4_ = make_stage(dims[3], depths[3], dp_start4, dp_end4);
    register_module("stage4", stage4_);

    // Global average pooling
    avgpool_ = std::make_shared<nn::AdaptiveAvgPool2d>(1);
    register_module("avgpool", avgpool_);

    // Final layer norm
    norm_ = std::make_shared<nn::LayerNorm>(
        std::vector<int64_t>{dims[3]}, 1e-6, true);
    register_module("norm", norm_);

    // Classification head
    head_ = std::make_shared<nn::Linear>(dims[3], num_classes);
    register_module("head", head_);
}

auto ConvNeXt::make_stage(int64_t dim, int64_t depth,
                          double drop_path_start, double drop_path_end)
    -> std::shared_ptr<nn::Sequential> {

    auto stage = std::make_shared<nn::Sequential>();

    for (int64_t i = 0; i < depth; ++i) {
        // Linearly scale drop path rate
        double dp_rate = drop_path_start +
                        (drop_path_end - drop_path_start) * i / std::max(depth - 1, (int64_t)1);

        auto block = std::make_shared<ConvNeXtBlock>(
            dim, dp_rate, layer_scale_init_value_);
        stage->add_module(block);
    }

    return stage;
}

auto ConvNeXt::make_downsample(int64_t in_dim, int64_t out_dim)
    -> std::shared_ptr<nn::Sequential> {

    auto downsample = std::make_shared<nn::Sequential>();

    // LayerNorm
    auto norm = std::make_shared<nn::LayerNorm>(
        std::vector<int64_t>{in_dim}, 1e-6, true);
    downsample->add_module(norm);

    // 2×2 conv with stride 2
    auto conv = std::make_shared<nn::Conv2d>(
        in_dim, out_dim, 2, 2, 0, 1, 1, false);
    downsample->add_module(conv);

    return downsample;
}

auto ConvNeXt::forward_impl(const Variable& input) -> Variable {
    // Stem: Conv + LayerNorm with proper permutations
    auto x = stem_conv_->forward(input);  // Output: NCHW
    x = x.permute({0, 2, 3, 1});           // NCHW → NHWC
    x = stem_norm_->forward(x);            // LayerNorm expects NHWC
    x = x.permute({0, 3, 1, 2});           // NHWC → NCHW

    // Stage 1
    x = stage1_->forward(x);

    // Downsample 1→2: LayerNorm + Conv with proper permutations
    x = x.permute({0, 2, 3, 1});           // NCHW → NHWC
    x = downsample1_norm_->forward(x);     // LayerNorm expects NHWC
    x = x.permute({0, 3, 1, 2});           // NHWC → NCHW
    x = downsample1_conv_->forward(x);     // Conv expects NCHW
    x = stage2_->forward(x);

    // Downsample 2→3: LayerNorm + Conv with proper permutations
    x = x.permute({0, 2, 3, 1});           // NCHW → NHWC
    x = downsample2_norm_->forward(x);     // LayerNorm expects NHWC
    x = x.permute({0, 3, 1, 2});           // NHWC → NCHW
    x = downsample2_conv_->forward(x);     // Conv expects NCHW
    x = stage3_->forward(x);

    // Downsample 3→4: LayerNorm + Conv with proper permutations
    x = x.permute({0, 2, 3, 1});           // NCHW → NHWC
    x = downsample3_norm_->forward(x);     // LayerNorm expects NHWC
    x = x.permute({0, 3, 1, 2});           // NHWC → NCHW
    x = downsample3_conv_->forward(x);     // Conv expects NCHW
    x = stage4_->forward(x);

    // Global average pooling
    x = avgpool_->forward(x);

    // Flatten and apply final LayerNorm
    auto shape = x.tensor().shape();
    x = tenzor::reshape(x, std::vector<int64_t>{shape[0], -1});
    x = norm_->forward(x);

    // Classification head
    x = head_->forward(x);

    return x;
}

auto ConvNeXt::load_pretrained(const std::string& path) -> void {
    load(path);
}

// ============================================================================
// Factory Functions
// ============================================================================

auto convnext_tiny(int64_t num_classes, bool pretrained)
    -> std::shared_ptr<ConvNeXt> {
    auto model = std::make_shared<ConvNeXt>(
        3, num_classes,
        std::vector<int64_t>{3, 3, 9, 3},
        std::vector<int64_t>{96, 192, 384, 768},
        0.1,  // drop_path_rate
        1e-6);

    if (pretrained) {
        auto _path = ModelHub::download_pretrained_safetensors("convnext_tiny");
        ModelHub::load_pretrained_weights(*model, _path, /*strict=*/false);
    }

    return model;
}

auto convnext_small(int64_t num_classes, bool pretrained)
    -> std::shared_ptr<ConvNeXt> {
    auto model = std::make_shared<ConvNeXt>(
        3, num_classes,
        std::vector<int64_t>{3, 3, 27, 3},
        std::vector<int64_t>{96, 192, 384, 768},
        0.4,  // drop_path_rate
        1e-6);

    if (pretrained) {
        auto _path = ModelHub::download_pretrained_safetensors("convnext_small");
        ModelHub::load_pretrained_weights(*model, _path, /*strict=*/false);
    }

    return model;
}

auto convnext_base(int64_t num_classes, bool pretrained)
    -> std::shared_ptr<ConvNeXt> {
    auto model = std::make_shared<ConvNeXt>(
        3, num_classes,
        std::vector<int64_t>{3, 3, 27, 3},
        std::vector<int64_t>{128, 256, 512, 1024},
        0.5,  // drop_path_rate
        1e-6);

    if (pretrained) {
        auto _path = ModelHub::download_pretrained_safetensors("convnext_base");
        ModelHub::load_pretrained_weights(*model, _path, /*strict=*/false);
    }

    return model;
}

auto convnext_large(int64_t num_classes, bool pretrained)
    -> std::shared_ptr<ConvNeXt> {
    auto model = std::make_shared<ConvNeXt>(
        3, num_classes,
        std::vector<int64_t>{3, 3, 27, 3},
        std::vector<int64_t>{192, 384, 768, 1536},
        0.5,  // drop_path_rate
        1e-6);

    if (pretrained) {
        auto _path = ModelHub::download_pretrained_safetensors("convnext_large");
        ModelHub::load_pretrained_weights(*model, _path, /*strict=*/false);
    }

    return model;
}

auto convnext_xlarge(int64_t num_classes, bool pretrained)
    -> std::shared_ptr<ConvNeXt> {
    auto model = std::make_shared<ConvNeXt>(
        3, num_classes,
        std::vector<int64_t>{3, 3, 27, 3},
        std::vector<int64_t>{256, 512, 1024, 2048},
        0.5,  // drop_path_rate
        1e-6);

    if (pretrained) {
        auto _path = ModelHub::download_pretrained_safetensors("convnext_xlarge");
        ModelHub::load_pretrained_weights(*model, _path, /*strict=*/false);
    }

    return model;
}

} // namespace models
} // namespace tenzor
