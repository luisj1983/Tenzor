/**
 * @file mobilenet.cpp
 * @brief Implementation of MobileNet V2 and V3 models
 */

#include "../../include/tenzor/models/mobilenet.hpp"
#include "../../include/tenzor/autograd/ops.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace tenzor {
namespace models {

// ============================================================================
// Hard-Swish and Hard-Sigmoid Implementations
// ============================================================================

auto HardSwish::forward(const Variable& input) -> Variable {
    // h-swish(x) = x * ReLU6(x + 3) / 6
    // ReLU6(x) = min(max(x, 0), 6)
    auto x_plus_3 = input + 3.0;
    auto relu6_result = tenzor::clamp(x_plus_3, 0.0, 6.0);
    return input * relu6_result / 6.0;
}

auto HardSigmoid::forward(const Variable& input) -> Variable {
    // h-sigmoid(x) = ReLU6(x + 3) / 6
    auto x_plus_3 = input + 3.0;
    return tenzor::clamp(x_plus_3, 0.0, 6.0) / 6.0;
}

// ============================================================================
// Squeeze-and-Excitation Implementation
// ============================================================================

MobileNetSqueezeExcitation::MobileNetSqueezeExcitation(int64_t channels,
                                                       int64_t reduction,
                                                       bool use_hard_sigmoid)
    : use_hard_sigmoid_(use_hard_sigmoid) {

    // Global average pooling
    pool_ = std::make_shared<nn::AdaptiveAvgPool2d>(1);
    register_module("pool", pool_);

    // Squeeze: channels → channels/reduction
    int64_t squeeze_channels = std::max((int64_t)1, channels / reduction);
    fc1_ = std::make_shared<nn::Linear>(channels, squeeze_channels);
    register_module("fc1", fc1_);

    // Excitation: channels/reduction → channels
    fc2_ = std::make_shared<nn::Linear>(squeeze_channels, channels);
    register_module("fc2", fc2_);
}

auto MobileNetSqueezeExcitation::forward(const Variable& input) -> Variable {
    auto shape = input.tensor().shape();
    int64_t batch = shape[0];
    int64_t channels = shape[1];

    // Squeeze: Global average pooling
    auto x = pool_->forward(input);

    // Flatten for FC layers
    x = tenzor::reshape(x, std::vector<int64_t>{batch, channels});

    // FC1 + ReLU
    x = fc1_->forward(x);
    x = relu_.forward(x);

    // FC2 + Sigmoid/Hard-Sigmoid
    x = fc2_->forward(x);
    if (use_hard_sigmoid_) {
        x = hard_sigmoid_.forward(x);
    } else {
        x = sigmoid_.forward(x);
    }

    // Reshape for channel-wise multiplication
    x = tenzor::reshape(x, std::vector<int64_t>{batch, channels, 1, 1});

    // Excitation: scale input
    return input * x;
}

// ============================================================================
// Inverted Residual Block Implementation
// ============================================================================

InvertedResidual::InvertedResidual(int64_t in_channels,
                                   int64_t out_channels,
                                   int64_t stride,
                                   int64_t expand_ratio,
                                   int64_t kernel_size,
                                   bool use_se,
                                   bool use_hs)
    : use_residual_(stride == 1 && in_channels == out_channels) {

    int64_t hidden_dim = in_channels * expand_ratio;
    int64_t padding = (kernel_size - 1) / 2;

    conv_ = std::make_shared<nn::Sequential>();

    // Expansion phase (only if expand_ratio != 1)
    if (expand_ratio != 1) {
        // 1×1 pointwise conv
        auto pw_conv = std::make_shared<nn::Conv2d>(
            in_channels, hidden_dim, 1, 1, 0, 1, 1, false);
        auto pw_bn = std::make_shared<nn::BatchNorm2d>(hidden_dim);

        conv_->add_module(pw_conv);
        conv_->add_module(pw_bn);

        if (use_hs) {
            conv_->add_module(std::make_shared<HardSwish>());
        } else {
            // ReLU6 for MobileNetV2
            conv_->add_module(std::make_shared<nn::ReLU6>());
        }
    }

    // Depthwise convolution
    auto dw_conv = std::make_shared<nn::Conv2d>(
        hidden_dim, hidden_dim, kernel_size, stride, padding,
        1, hidden_dim, false);  // groups=hidden_dim for depthwise
    auto dw_bn = std::make_shared<nn::BatchNorm2d>(hidden_dim);

    conv_->add_module(dw_conv);
    conv_->add_module(dw_bn);

    if (use_hs) {
        conv_->add_module(std::make_shared<HardSwish>());
    } else {
        // ReLU6
        conv_->add_module(std::make_shared<nn::ReLU6>());
    }

    // Squeeze-and-Excitation
    if (use_se) {
        auto se = std::make_shared<MobileNetSqueezeExcitation>(
            hidden_dim, 4, use_hs);  // reduction=4, use hard-sigmoid if use_hs
        conv_->add_module(se);
    }

    // Projection phase (linear bottleneck - NO activation!)
    auto proj_conv = std::make_shared<nn::Conv2d>(
        hidden_dim, out_channels, 1, 1, 0, 1, 1, false);
    auto proj_bn = std::make_shared<nn::BatchNorm2d>(out_channels);

    conv_->add_module(proj_conv);
    conv_->add_module(proj_bn);
    // NOTE: No activation after projection (linear bottleneck)

    register_module("conv", conv_);
}

auto InvertedResidual::forward(const Variable& input) -> Variable {
    auto x = conv_->forward(input);

    if (use_residual_) {
        return input + x;
    }
    return x;
}

// ============================================================================
// MobileNetV2 Implementation
// ============================================================================

auto MobileNetV2::make_divisible(int64_t v, int64_t divisor) -> int64_t {
    int64_t new_v = std::max(divisor, (v + divisor / 2) / divisor * divisor);
    // Make sure that round down does not go down by more than 10%
    if (new_v < 0.9 * v) {
        new_v += divisor;
    }
    return new_v;
}

MobileNetV2::MobileNetV2(int64_t num_classes, double width_mult, double dropout) {
    // MobileNetV2 architecture configuration
    // [expansion, output_channels, num_blocks, stride]
    std::vector<std::vector<int64_t>> inverted_residual_settings = {
        {1, 16, 1, 1},    // First block with expansion=1
        {6, 24, 2, 2},
        {6, 32, 3, 2},
        {6, 64, 4, 2},
        {6, 96, 3, 1},
        {6, 160, 3, 2},
        {6, 320, 1, 1}
    };

    features_ = std::make_shared<nn::Sequential>();

    // First conv layer
    int64_t input_channels = make_divisible(static_cast<int64_t>(32 * width_mult), 8);
    auto first_conv = std::make_shared<nn::Conv2d>(
        3, input_channels, 3, 2, 1, 1, 1, false);
    auto first_bn = std::make_shared<nn::BatchNorm2d>(input_channels);
    // ReLU6 activation

    features_->add_module(first_conv);
    features_->add_module(first_bn);

    // Inverted residual blocks
    for (const auto& setting : inverted_residual_settings) {
        int64_t expand_ratio = setting[0];
        int64_t out_channels = make_divisible(
            static_cast<int64_t>(setting[1] * width_mult), 8);
        int64_t num_blocks = setting[2];
        int64_t stride = setting[3];

        for (int64_t i = 0; i < num_blocks; ++i) {
            int64_t block_stride = (i == 0) ? stride : 1;
            auto block = std::make_shared<InvertedResidual>(
                input_channels, out_channels, block_stride,
                expand_ratio, 3, false, false);  // kernel=3, no SE, no HS (use ReLU6)
            features_->add_module(block);
            input_channels = out_channels;
        }
    }

    // Final conv layer
    int64_t last_channels = make_divisible(static_cast<int64_t>(1280 * width_mult), 8);
    if (width_mult > 1.0) {
        last_channels = 1280;
    }

    auto last_conv = std::make_shared<nn::Conv2d>(
        input_channels, last_channels, 1, 1, 0, 1, 1, false);
    auto last_bn = std::make_shared<nn::BatchNorm2d>(last_channels);
    // ReLU6

    features_->add_module(last_conv);
    features_->add_module(last_bn);

    register_module("features", features_);

    // Classifier
    classifier_ = std::make_shared<nn::Sequential>();
    auto dropout_layer = std::make_shared<nn::Dropout>(dropout);
    auto fc = std::make_shared<nn::Linear>(last_channels, num_classes);

    classifier_->add_module(dropout_layer);
    classifier_->add_module(fc);

    register_module("classifier", classifier_);
}

auto MobileNetV2::forward(const Variable& input) -> Variable {
    auto x = features_->forward(input);

    // Global average pooling
    auto shape = x.tensor().shape();
    auto pool = nn::AdaptiveAvgPool2d(1);
    x = pool.forward(x);

    // Flatten
    x = tenzor::reshape(x, std::vector<int64_t>{shape[0], -1});

    // Classifier
    x = classifier_->forward(x);

    return x;
}

auto MobileNetV2::load_pretrained(const std::string& path) -> void {
    load(path);
}

// ============================================================================
// MobileNetV3 Implementation
// ============================================================================

auto MobileNetV3::make_divisible(int64_t v, int64_t divisor) -> int64_t {
    int64_t new_v = std::max(divisor, (v + divisor / 2) / divisor * divisor);
    if (new_v < 0.9 * v) {
        new_v += divisor;
    }
    return new_v;
}

auto MobileNetV3::build_large_config() -> std::vector<std::vector<int64_t>> {
    // MobileNetV3-Large configuration
    // [kernel, exp_size, out_channels, use_se, use_hs, stride]
    return {
        {3, 16, 16, 0, 0, 1},      // RE = ReLU
        {3, 64, 24, 0, 0, 2},
        {3, 72, 24, 0, 0, 1},
        {5, 72, 40, 1, 0, 2},      // SE = 1
        {5, 120, 40, 1, 0, 1},
        {5, 120, 40, 1, 0, 1},
        {3, 240, 80, 0, 1, 2},     // HS = Hard-Swish
        {3, 200, 80, 0, 1, 1},
        {3, 184, 80, 0, 1, 1},
        {3, 184, 80, 0, 1, 1},
        {3, 480, 112, 1, 1, 1},
        {3, 672, 112, 1, 1, 1},
        {5, 672, 160, 1, 1, 2},
        {5, 960, 160, 1, 1, 1},
        {5, 960, 160, 1, 1, 1}
    };
}

auto MobileNetV3::build_small_config() -> std::vector<std::vector<int64_t>> {
    // MobileNetV3-Small configuration
    // [kernel, exp_size, out_channels, use_se, use_hs, stride]
    return {
        {3, 16, 16, 1, 0, 2},      // SE + ReLU
        {3, 72, 24, 0, 0, 2},
        {3, 88, 24, 0, 0, 1},
        {5, 96, 40, 1, 1, 2},      // SE + HS
        {5, 240, 40, 1, 1, 1},
        {5, 240, 40, 1, 1, 1},
        {5, 120, 48, 1, 1, 1},
        {5, 144, 48, 1, 1, 1},
        {5, 288, 96, 1, 1, 2},
        {5, 576, 96, 1, 1, 1},
        {5, 576, 96, 1, 1, 1}
    };
}

MobileNetV3::MobileNetV3(int64_t num_classes,
                         const std::string& mode,
                         double width_mult,
                         double dropout) {
    if (mode != "large" && mode != "small") {
        throw std::invalid_argument("MobileNetV3 mode must be 'large' or 'small'");
    }

    auto config = (mode == "large") ? build_large_config() : build_small_config();

    features_ = std::make_shared<nn::Sequential>();

    // First conv
    int64_t input_channels = make_divisible(static_cast<int64_t>(16 * width_mult), 8);
    auto first_conv = std::make_shared<nn::Conv2d>(
        3, input_channels, 3, 2, 1, 1, 1, false);
    auto first_bn = std::make_shared<nn::BatchNorm2d>(input_channels);
    auto first_act = std::make_shared<HardSwish>();

    features_->add_module(first_conv);
    features_->add_module(first_bn);
    features_->add_module(first_act);

    // Inverted residual blocks
    for (const auto& layer_config : config) {
        int64_t kernel = layer_config[0];
        int64_t exp_size = make_divisible(
            static_cast<int64_t>(layer_config[1] * width_mult), 8);
        int64_t out_channels = make_divisible(
            static_cast<int64_t>(layer_config[2] * width_mult), 8);
        bool use_se = layer_config[3] != 0;
        bool use_hs = layer_config[4] != 0;
        int64_t stride = layer_config[5];

        int64_t expand_ratio = exp_size / input_channels;

        auto block = std::make_shared<InvertedResidual>(
            input_channels, out_channels, stride,
            expand_ratio, kernel, use_se, use_hs);
        features_->add_module(block);

        input_channels = out_channels;
    }

    // Final conv layers
    int64_t exp_size = (mode == "large") ? 960 : 576;
    exp_size = make_divisible(static_cast<int64_t>(exp_size * width_mult), 8);

    auto final_conv = std::make_shared<nn::Conv2d>(
        input_channels, exp_size, 1, 1, 0, 1, 1, false);
    auto final_bn = std::make_shared<nn::BatchNorm2d>(exp_size);
    auto final_act = std::make_shared<HardSwish>();

    features_->add_module(final_conv);
    features_->add_module(final_bn);
    features_->add_module(final_act);

    register_module("features", features_);

    // Classifier
    classifier_ = std::make_shared<nn::Sequential>();

    // Efficient last stage: pool first, then expand
    auto pool = std::make_shared<nn::AdaptiveAvgPool2d>(1);
    classifier_->add_module(pool);

    int64_t last_channels = (mode == "large") ? 1280 : 1024;
    last_channels = make_divisible(
        static_cast<int64_t>(last_channels * width_mult), 8);

    auto fc1 = std::make_shared<nn::Conv2d>(
        exp_size, last_channels, 1, 1, 0, 1, 1, true);
    auto fc1_act = std::make_shared<HardSwish>();

    classifier_->add_module(fc1);
    classifier_->add_module(fc1_act);

    auto dropout_layer = std::make_shared<nn::Dropout>(dropout);
    auto fc2 = std::make_shared<nn::Conv2d>(
        last_channels, num_classes, 1, 1, 0, 1, 1, true);

    classifier_->add_module(dropout_layer);
    classifier_->add_module(fc2);

    register_module("classifier", classifier_);
}

auto MobileNetV3::forward(const Variable& input) -> Variable {
    auto x = features_->forward(input);

    // Classifier (includes pooling)
    x = classifier_->forward(x);

    // Flatten
    auto shape = x.tensor().shape();
    x = tenzor::reshape(x, std::vector<int64_t>{shape[0], -1});

    return x;
}

auto MobileNetV3::load_pretrained(const std::string& path) -> void {
    load(path);
}

// ============================================================================
// Factory Functions
// ============================================================================

auto mobilenet_v2(int64_t num_classes, bool pretrained)
    -> std::shared_ptr<MobileNetV2> {
    auto model = std::make_shared<MobileNetV2>(num_classes, 1.0, 0.2);

    if (pretrained) {
        model->load_pretrained("mobilenet_v2_imagenet.pth");
    }

    return model;
}

auto mobilenet_v2_width(int64_t num_classes, double width_mult, bool pretrained)
    -> std::shared_ptr<MobileNetV2> {
    auto model = std::make_shared<MobileNetV2>(num_classes, width_mult, 0.2);

    if (pretrained) {
        std::string path = "mobilenet_v2_" + std::to_string(width_mult) + "_imagenet.pth";
        model->load_pretrained(path);
    }

    return model;
}

auto mobilenet_v3_large(int64_t num_classes, bool pretrained)
    -> std::shared_ptr<MobileNetV3> {
    auto model = std::make_shared<MobileNetV3>(num_classes, "large", 1.0, 0.2);

    if (pretrained) {
        model->load_pretrained("mobilenet_v3_large_imagenet.pth");
    }

    return model;
}

auto mobilenet_v3_small(int64_t num_classes, bool pretrained)
    -> std::shared_ptr<MobileNetV3> {
    auto model = std::make_shared<MobileNetV3>(num_classes, "small", 1.0, 0.2);

    if (pretrained) {
        model->load_pretrained("mobilenet_v3_small_imagenet.pth");
    }

    return model;
}

} // namespace models
} // namespace tenzor
