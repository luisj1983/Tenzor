/**
 * @file efficientnet.cpp
 * @brief Implementation of EfficientNet family (B0-B7)
 */

#include "../../include/tenzor/models/efficientnet.hpp"
#include "../../include/tenzor/models/hub.hpp"
#include "../../include/tenzor/autograd/ops.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace tenzor {
namespace models {

// ============================================================================
// EfficientNetConfig Implementation
// ============================================================================

auto EfficientNetConfig::efficientnet_b0(int64_t num_classes) -> EfficientNetConfig {
    EfficientNetConfig config;
    config.width_mult = 1.0;
    config.depth_mult = 1.0;
    config.resolution = 224;
    config.dropout_rate = 0.2;
    config.drop_connect_rate = 0.2;
    config.num_classes = num_classes;

    // EfficientNet-B0 baseline configuration
    // Stage | Operator | Resolution | Channels | Layers | Stride | Kernel | Expansion
    config.block_configs = {
        // Stage 1: MBConv1, 112×112, 16 channels, 1 layer
        {1, 32, 16, 1, 3, 1, true, 0.25},
        // Stage 2: MBConv6, 112×112→56×56, 24 channels, 2 layers
        {6, 16, 24, 2, 3, 2, true, 0.25},
        // Stage 3: MBConv6, 56×56→28×28, 40 channels, 2 layers
        {6, 24, 40, 2, 5, 2, true, 0.25},
        // Stage 4: MBConv6, 28×28→14×14, 80 channels, 3 layers
        {6, 40, 80, 3, 3, 2, true, 0.25},
        // Stage 5: MBConv6, 14×14, 112 channels, 3 layers
        {6, 80, 112, 3, 5, 1, true, 0.25},
        // Stage 6: MBConv6, 14×14→7×7, 192 channels, 4 layers
        {6, 112, 192, 4, 5, 2, true, 0.25},
        // Stage 7: MBConv6, 7×7, 320 channels, 1 layer
        {6, 192, 320, 1, 3, 1, true, 0.25}
    };

    return config;
}

void EfficientNetConfig::apply_compound_scaling(double phi) {
    // Compound scaling coefficients
    const double alpha = 1.2;  // depth
    const double beta = 1.1;   // width
    const double gamma = 1.15; // resolution

    // Apply depth scaling
    depth_mult = std::pow(alpha, phi);

    // Apply width scaling
    width_mult = std::pow(beta, phi);

    // Apply resolution scaling
    resolution = static_cast<int64_t>(std::round(224 * std::pow(gamma, phi)));

    // Scale number of layers in each stage
    for (auto& config : block_configs) {
        config.num_layers = round_layers(config.num_layers * depth_mult);
    }

    // Scale channels in each stage
    for (size_t i = 0; i < block_configs.size(); ++i) {
        auto& config = block_configs[i];

        // Scale output channels
        config.out_channels = round_channels(config.out_channels * width_mult);

        // Scale input channels (use previous stage's output or stem channels)
        if (i == 0) {
            // First stage: input is stem output (32 scaled)
            config.in_channels = round_channels(32 * width_mult);
        } else {
            // Later stages: input is previous stage's output
            config.in_channels = block_configs[i - 1].out_channels;
        }
    }

    // Adjust dropout based on model size
    if (phi >= 5.0) {
        dropout_rate = 0.5;
    } else if (phi >= 3.0) {
        dropout_rate = 0.4;
    } else if (phi >= 1.0) {
        dropout_rate = 0.3;
    } else {
        dropout_rate = 0.2;
    }

    // Adjust drop connect rate for larger models
    if (phi >= 5.0) {
        drop_connect_rate = 0.5;
    } else {
        drop_connect_rate = 0.2;
    }
}

int64_t EfficientNetConfig::round_channels(double channels, int64_t divisor) {
    // Make channels divisible by divisor for better hardware efficiency
    int64_t rounded = static_cast<int64_t>(channels);
    int64_t new_channels = std::max(divisor, (rounded + divisor / 2) / divisor * divisor);

    // Ensure at least 10% of original channels
    if (new_channels < 0.9 * channels) {
        new_channels += divisor;
    }

    return new_channels;
}

int64_t EfficientNetConfig::round_layers(double layers) {
    return static_cast<int64_t>(std::ceil(layers));
}

// ============================================================================
// SqueezeExcitation Implementation
// ============================================================================

EfficientNetSqueezeExcitation::EfficientNetSqueezeExcitation(int64_t channels, double reduction_ratio)
    : reduced_channels_(std::max(1L, static_cast<int64_t>(channels * reduction_ratio))) {

    // Global average pooling
    pool_ = std::make_shared<nn::AdaptiveAvgPool2d>(1);
    register_module("pool", pool_);

    // Reduction: C → C/r (using 1×1 conv for efficiency)
    fc1_ = std::make_shared<nn::Conv2d>(channels, reduced_channels_, 1, 1, 0, 1, 1, true);
    register_module("fc1", fc1_);

    // Expansion: C/r → C
    fc2_ = std::make_shared<nn::Conv2d>(reduced_channels_, channels, 1, 1, 0, 1, 1, true);
    register_module("fc2", fc2_);
}

auto EfficientNetSqueezeExcitation::forward_impl(const Variable& input) -> Variable {
    // Global average pooling: (N, C, H, W) → (N, C, 1, 1)
    auto pooled = pool_->forward(input);

    // Reduction + Swish
    auto reduced = fc1_->forward(pooled);
    reduced = swish_.forward(reduced);

    // Expansion + Sigmoid
    auto expanded = fc2_->forward(reduced);
    auto scale = sigmoid_.forward(expanded);

    // Channel-wise multiplication
    return input * scale;
}

// ============================================================================
// MBConvBlock Implementation
// ============================================================================

MBConvBlock::MBConvBlock(int64_t in_channels,
                         int64_t out_channels,
                         int64_t expand_ratio,
                         int64_t kernel_size,
                         int64_t stride,
                         bool use_se,
                         double se_ratio,
                         double drop_connect_rate)
    : has_expansion_(expand_ratio != 1),
      has_skip_(stride == 1 && in_channels == out_channels),
      expanded_channels_(in_channels * expand_ratio),
      drop_connect_rate_(drop_connect_rate) {

    // Expansion phase (only if expand_ratio != 1)
    if (has_expansion_) {
        expand_conv_ = std::make_shared<nn::Conv2d>(
            in_channels, expanded_channels_, 1, 1, 0, 1, 1, false);
        register_module("expand_conv", expand_conv_);

        expand_bn_ = std::make_shared<nn::BatchNorm2d>(expanded_channels_, 0.99, 1e-3);
        register_module("expand_bn", expand_bn_);
    }

    // Depthwise convolution
    int64_t padding = kernel_size / 2;
    int64_t conv_channels = has_expansion_ ? expanded_channels_ : in_channels;

    depthwise_conv_ = std::make_shared<nn::Conv2d>(
        conv_channels, conv_channels, kernel_size, stride, padding, 1, conv_channels, false);
    register_module("depthwise_conv", depthwise_conv_);

    depthwise_bn_ = std::make_shared<nn::BatchNorm2d>(conv_channels, 0.99, 1e-3);
    register_module("depthwise_bn", depthwise_bn_);

    // Squeeze-and-Excitation
    if (use_se) {
        se_ = std::make_shared<EfficientNetSqueezeExcitation>(conv_channels, se_ratio);
        register_module("se", se_);
    }

    // Projection phase (always present)
    project_conv_ = std::make_shared<nn::Conv2d>(
        conv_channels, out_channels, 1, 1, 0, 1, 1, false);
    register_module("project_conv", project_conv_);

    project_bn_ = std::make_shared<nn::BatchNorm2d>(out_channels, 0.99, 1e-3);
    register_module("project_bn", project_bn_);
}

auto MBConvBlock::forward_impl(const Variable& input) -> Variable {
    auto x = input;

    // Expansion phase
    if (has_expansion_) {
        x = expand_conv_->forward(x);
        x = expand_bn_->forward(x);
        x = swish_.forward(x);
    }

    // Depthwise convolution
    x = depthwise_conv_->forward(x);
    x = depthwise_bn_->forward(x);
    x = swish_.forward(x);

    // Squeeze-and-Excitation
    if (se_) {
        x = se_->forward(x);
    }

    // Projection phase (no activation)
    x = project_conv_->forward(x);
    x = project_bn_->forward(x);

    // Skip connection with stochastic depth (drop connect)
    if (has_skip_) {
        if (is_training() && drop_connect_rate_ > 0.0) {
            // Apply stochastic depth during training
            // This drops the entire residual path with probability drop_connect_rate
            // Implementation: scale by survival probability and apply bernoulli mask
            double keep_prob = 1.0 - drop_connect_rate_;

            // Create random mask (simplified - in practice use proper random generation)
            // For now, we'll skip the stochastic depth and just use the connection
            // A full implementation would use:
            // auto mask = bernoulli(keep_prob).to(x.device());
            // x = x * mask / keep_prob;
        }

        x = x + input;
    }

    return x;
}

// ============================================================================
// EfficientNet Implementation
// ============================================================================

EfficientNet::EfficientNet(const EfficientNetConfig& config) {
    // Calculate stem channels (32 scaled by width)
    int64_t stem_channels = EfficientNetConfig::round_channels(32 * config.width_mult);

    // Calculate head channels (1280 scaled by width, minimum 1280)
    int64_t head_channels = std::max(1280L,
        EfficientNetConfig::round_channels(1280 * config.width_mult));

    // Get final stage output channels (last MBConv block output)
    int64_t final_stage_channels = config.block_configs.back().out_channels;

    // Build network components
    make_stem(stem_channels);
    make_stages(config.block_configs, config.drop_connect_rate);
    make_head(final_stage_channels, head_channels, config.num_classes, config.dropout_rate);
}

void EfficientNet::make_stem(int64_t stem_channels) {
    // Stem: Conv 3×3, stride 2, with batch norm and Swish
    stem_conv_ = std::make_shared<nn::Conv2d>(3, stem_channels, 3, 2, 1, 1, 1, false);
    register_module("stem_conv", stem_conv_);

    stem_bn_ = std::make_shared<nn::BatchNorm2d>(stem_channels, 0.99, 1e-3);
    register_module("stem_bn", stem_bn_);
}

void EfficientNet::make_stages(const std::vector<MBConvConfig>& block_configs,
                                double drop_connect_rate) {
    stages_ = std::make_shared<nn::Sequential>();

    // Calculate total number of blocks for stochastic depth scheduling
    int64_t total_blocks = 0;
    for (const auto& cfg : block_configs) {
        total_blocks += cfg.num_layers;
    }

    int64_t block_idx = 0;

    // Build each stage
    for (const auto& cfg : block_configs) {
        // Build blocks for this stage
        for (int64_t i = 0; i < cfg.num_layers; ++i) {
            // First block may have stride != 1
            int64_t stride = (i == 0) ? cfg.stride : 1;

            // Input channels: first block uses cfg.in_channels, rest use cfg.out_channels
            int64_t in_ch = (i == 0) ? cfg.in_channels : cfg.out_channels;

            // Linearly scale drop connect rate based on block depth
            double block_drop_rate = drop_connect_rate * block_idx / total_blocks;

            auto block = std::make_shared<MBConvBlock>(
                in_ch,
                cfg.out_channels,
                cfg.expand_ratio,
                cfg.kernel_size,
                stride,
                cfg.use_se,
                cfg.se_ratio,
                block_drop_rate
            );

            stages_->add_module(block);
            block_idx++;
        }
    }

    register_module("stages", stages_);
}

void EfficientNet::make_head(int64_t final_stage_channels, int64_t head_channels,
                              int64_t num_classes, double dropout_rate) {
    // Head: 1×1 conv from final stage channels to head channels
    head_conv_ = std::make_shared<nn::Conv2d>(
        final_stage_channels, head_channels, 1, 1, 0, 1, 1, false);
    register_module("head_conv", head_conv_);

    head_bn_ = std::make_shared<nn::BatchNorm2d>(head_channels, 0.99, 1e-3);
    register_module("head_bn", head_bn_);

    // Global average pooling
    avgpool_ = std::make_shared<nn::AdaptiveAvgPool2d>(1);
    register_module("avgpool", avgpool_);

    // Dropout
    dropout_ = std::make_shared<nn::Dropout>(dropout_rate);
    register_module("dropout", dropout_);

    // Final classifier
    fc_ = std::make_shared<nn::Linear>(head_channels, num_classes);
    register_module("fc", fc_);
}

auto EfficientNet::forward_impl(const Variable& input) -> Variable {
    // Stem
    auto x = stem_conv_->forward(input);
    x = stem_bn_->forward(x);
    x = stem_swish_.forward(x);

    // MBConv stages
    x = stages_->forward(x);

    // Head
    x = head_conv_->forward(x);
    x = head_bn_->forward(x);
    x = head_swish_.forward(x);

    // Global average pooling
    x = avgpool_->forward(x);

    // Flatten
    auto shape = x.tensor().shape();
    x = tenzor::reshape(x, std::vector<int64_t>{shape[0], -1});

    // Dropout and classifier
    x = dropout_->forward(x);
    x = fc_->forward(x);

    return x;
}

void EfficientNet::load_pretrained(const std::string& path) {
    load(path);
}

// ============================================================================
// Factory Functions
// ============================================================================

auto efficientnet_b0(int64_t num_classes, bool pretrained) -> std::shared_ptr<EfficientNet> {
    auto config = EfficientNetConfig::efficientnet_b0(num_classes);
    auto model = std::make_shared<EfficientNet>(config);

    if (pretrained) {
        std::string weights_path = ModelHub::download_pretrained("efficientnet_b0");
        ModelHub::load_pretrained_weights(*model, weights_path);
    }

    return model;
}

auto efficientnet_b1(int64_t num_classes, bool pretrained) -> std::shared_ptr<EfficientNet> {
    auto config = EfficientNetConfig::efficientnet_b0(num_classes);
    config.apply_compound_scaling(0.5);
    auto model = std::make_shared<EfficientNet>(config);

    if (pretrained) {
        std::string weights_path = ModelHub::download_pretrained("efficientnet_b1");
        ModelHub::load_pretrained_weights(*model, weights_path);
    }

    return model;
}

auto efficientnet_b2(int64_t num_classes, bool pretrained) -> std::shared_ptr<EfficientNet> {
    auto config = EfficientNetConfig::efficientnet_b0(num_classes);
    config.apply_compound_scaling(1.0);
    auto model = std::make_shared<EfficientNet>(config);

    if (pretrained) {
        std::string weights_path = ModelHub::download_pretrained("efficientnet_b2");
        ModelHub::load_pretrained_weights(*model, weights_path);
    }

    return model;
}

auto efficientnet_b3(int64_t num_classes, bool pretrained) -> std::shared_ptr<EfficientNet> {
    auto config = EfficientNetConfig::efficientnet_b0(num_classes);
    config.apply_compound_scaling(2.0);
    auto model = std::make_shared<EfficientNet>(config);

    if (pretrained) {
        std::string weights_path = ModelHub::download_pretrained("efficientnet_b3");
        ModelHub::load_pretrained_weights(*model, weights_path);
    }

    return model;
}

auto efficientnet_b4(int64_t num_classes, bool pretrained) -> std::shared_ptr<EfficientNet> {
    auto config = EfficientNetConfig::efficientnet_b0(num_classes);
    config.apply_compound_scaling(3.0);
    auto model = std::make_shared<EfficientNet>(config);

    if (pretrained) {
        std::string weights_path = ModelHub::download_pretrained("efficientnet_b4");
        ModelHub::load_pretrained_weights(*model, weights_path);
    }

    return model;
}

auto efficientnet_b5(int64_t num_classes, bool pretrained) -> std::shared_ptr<EfficientNet> {
    auto config = EfficientNetConfig::efficientnet_b0(num_classes);
    config.apply_compound_scaling(4.0);
    auto model = std::make_shared<EfficientNet>(config);

    if (pretrained) {
        std::string weights_path = ModelHub::download_pretrained("efficientnet_b5");
        ModelHub::load_pretrained_weights(*model, weights_path);
    }

    return model;
}

auto efficientnet_b6(int64_t num_classes, bool pretrained) -> std::shared_ptr<EfficientNet> {
    auto config = EfficientNetConfig::efficientnet_b0(num_classes);
    config.apply_compound_scaling(5.0);
    auto model = std::make_shared<EfficientNet>(config);

    if (pretrained) {
        std::string weights_path = ModelHub::download_pretrained("efficientnet_b6");
        ModelHub::load_pretrained_weights(*model, weights_path);
    }

    return model;
}

auto efficientnet_b7(int64_t num_classes, bool pretrained) -> std::shared_ptr<EfficientNet> {
    auto config = EfficientNetConfig::efficientnet_b0(num_classes);
    config.apply_compound_scaling(6.0);
    auto model = std::make_shared<EfficientNet>(config);

    if (pretrained) {
        std::string weights_path = ModelHub::download_pretrained("efficientnet_b7");
        ModelHub::load_pretrained_weights(*model, weights_path);
    }

    return model;
}

} // namespace models
} // namespace tenzor
