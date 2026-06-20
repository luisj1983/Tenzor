/**
 * @file deeplabv3plus.cpp
 * @brief Implementation of DeepLab v3+ semantic segmentation model
 */

#include "tenzor/models/deeplabv3plus.hpp"
#include "tenzor/models/mobilenet.hpp"
#include "tenzor/models/hub.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/checkpoint.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include <stdexcept>
#include <sstream>

namespace tenzor {
namespace models {

// ============================================================================
// DeepLabV3PlusEncoder Implementation
// ============================================================================

DeepLabV3PlusEncoder::DeepLabV3PlusEncoder(const std::string& backbone_name,
                                           int64_t output_stride,
                                           bool pretrained)
    : backbone_name_(backbone_name)
    , output_stride_(output_stride)
{
    if (output_stride != 8 && output_stride != 16) {
        throw std::invalid_argument("output_stride must be 8 or 16");
    }

    // Create backbone. Audit G11: pass output_stride through so the atrous
    // variants of ResNet are used when output_stride < 32 (replaces the
    // previous heuristic of regular ResNet + project+upsample fake).
    backbone_ = create_resnet_backbone(backbone_name, pretrained, output_stride);

    // Determine channel dimensions based on backbone
    if (backbone_name == "resnet50" || backbone_name == "resnet101" ||
        backbone_name == "resnet152") {
        // ResNet with Bottleneck blocks
        low_level_channels_ = 256;    // After layer1 (1/4 resolution)
        high_level_channels_ = 2048;  // After layer4 (1/16 or 1/8 resolution)
    } else if (backbone_name == "resnet18" || backbone_name == "resnet34") {
        // ResNet with BasicBlock
        low_level_channels_ = 64;     // After layer1
        high_level_channels_ = 512;   // After layer4
    } else if (backbone_name == "mobilenetv2") {
        // Audit G.8: real low-level (stride 4, 24ch) + high-level (1280ch)
        // features now come from MobileNetV2::forward_features_multi.
        low_level_channels_ = 24;
        high_level_channels_ = 1280;
    } else {
        throw std::invalid_argument("Unsupported backbone: " + backbone_name);
    }

    // Create ASPP module
    std::vector<int64_t> atrous_rates;
    if (output_stride == 16) {
        atrous_rates = {6, 12, 18};  // Standard rates for output_stride=16
    } else {
        atrous_rates = {12, 24, 36}; // Doubled rates for output_stride=8
    }

    aspp_ = std::make_shared<nn::ASPP>(high_level_channels_, 256, atrous_rates, true, 0.5f);
    register_module("aspp", aspp_);

    // Audit G.8 / G11: every supported backbone (ResNet, atrous MobileNetV2)
    // now produces real low-level features via forward_features_multi. The
    // legacy project+upsample heuristic (feature_proj_) is no longer used; the
    // pointer remains in the header for ABI/state-dict stability but is left
    // unconstructed here.
}

auto DeepLabV3PlusEncoder::create_resnet_backbone(const std::string& name,
                                                   bool pretrained,
                                                   int64_t output_stride)
    -> std::shared_ptr<nn::Module>
{
    // Audit G11: select atrous-modified ResNet variants when output_stride < 32.
    // ResNet18/34 (BasicBlock) don't support atrous, so we require
    // output_stride=32 for them.
    std::shared_ptr<nn::Module> backbone;

    if (name == "resnet50") {
        backbone = (output_stride == 32) ? resnet50(1000, pretrained)
                                         : resnet50_atrous(1000, output_stride, pretrained);
    } else if (name == "resnet101") {
        backbone = (output_stride == 32) ? resnet101(1000, pretrained)
                                         : resnet101_atrous(1000, output_stride, pretrained);
    } else if (name == "resnet152") {
        backbone = (output_stride == 32) ? resnet152(1000, pretrained)
                                         : resnet152_atrous(1000, output_stride, pretrained);
    } else if (name == "resnet18") {
        if (output_stride != 32) {
            throw std::invalid_argument(
                "ResNet-18 (BasicBlock) does not support atrous output_stride < 32. "
                "Use resnet50/101/152 for DeepLab-style decoders.");
        }
        backbone = resnet18(1000, pretrained);
    } else if (name == "resnet34") {
        if (output_stride != 32) {
            throw std::invalid_argument(
                "ResNet-34 (BasicBlock) does not support atrous output_stride < 32. "
                "Use resnet50/101/152 for DeepLab-style decoders.");
        }
        backbone = resnet34(1000, pretrained);
    } else if (name == "mobilenetv2") {
        // Audit G.8: MobileNetV2 now supports atrous mode. For output_stride<32
        // we route through the atrous builder so the last stages preserve
        // spatial resolution; output_stride=32 falls back to the standard model.
        backbone = (output_stride == 32)
                       ? mobilenet_v2(1000, pretrained)
                       : atrous_mobilenet_v2(1000, pretrained, output_stride);
    } else {
        throw std::invalid_argument("Unsupported backbone variant: " + name);
    }

    register_module("backbone", backbone);
    return backbone;
}

auto DeepLabV3PlusEncoder::forward_multi(const Variable& input)
    -> std::pair<Variable, Variable>
{
    // Audit G.8 / G11: real low-level feature extraction for every backbone.
    //
    // ResNet path: forward_features_multi -> (C2, C3, C4, C5). C2 is the
    // canonical low-level feature (layer1 output, stride 4, 256 ch for
    // Bottleneck / 64 ch for BasicBlock — matches `low_level_channels_`).
    // ASPP runs on C5; the standard DeepLabV3+ "multi-scale" design uses
    // multiple parallel atrous rates *within* ASPP applied to that single
    // feature map (1x1 conv + 3 parallel atrous convs at rates {6,12,18} for
    // output_stride=16 or {12,24,36} for output_stride=8 + global image
    // pooling). C3/C4 are intentionally unused — they are not consumed by the
    // DeepLabV3+ paper architecture.
    //
    // MobileNetV2 path: atrous MobileNetV2 + forward_features_multi gives
    // (low_level, high_level) directly — no project+upsample fake.
    auto resnet = std::dynamic_pointer_cast<ResNet>(backbone_);
    if (resnet) {
        auto [c2, c3, c4, c5] = resnet->forward_features_multi(input);
        (void)c3;
        (void)c4;
        Variable aspp_features      = aspp_->forward(c5);
        Variable low_level_features = c2;
        return {aspp_features, low_level_features};
    }

    auto mobilenet = std::dynamic_pointer_cast<MobileNetV2>(backbone_);
    if (!mobilenet) {
        throw std::runtime_error("Unsupported backbone type for DeepLabV3+");
    }

    auto [low_level_features, high_level_features] =
        mobilenet->forward_features_multi(input);
    Variable aspp_features = aspp_->forward(high_level_features);
    return {aspp_features, low_level_features};
}

// ============================================================================
// DeepLabV3PlusDecoder Implementation
// ============================================================================

DeepLabV3PlusDecoder::DeepLabV3PlusDecoder(int64_t num_classes,
                                           int64_t low_level_channels,
                                           int64_t aspp_channels)
    : num_classes_(num_classes)
{
    // Reduce low-level feature channels (e.g., 256 -> 48)
    low_level_reduce_ = nn::make_conv_bn_relu(low_level_channels, 48, 1);
    register_module("low_level_reduce", low_level_reduce_);

    // Refinement convolutions after concatenation
    // Input: aspp_channels + 48, Output: 256
    class RefineModule : public nn::Module {
    public:
        RefineModule(int64_t in_channels, int64_t out_channels) {
            conv1_ = nn::make_conv_bn_relu(in_channels, out_channels, 3, 1, 1);
            register_module("conv1", conv1_);
            conv2_ = nn::make_conv_bn_relu(out_channels, out_channels, 3, 1, 1);
            register_module("conv2", conv2_);
        }

        auto forward_impl(const Variable& input) -> Variable override {
            auto x = conv1_->forward(input);
            x = conv2_->forward(x);
            return x;
        }

    private:
        std::shared_ptr<nn::Module> conv1_;
        std::shared_ptr<nn::Module> conv2_;
    };

    refine_ = std::make_shared<RefineModule>(aspp_channels + 48, 256);
    register_module("refine", refine_);

    // Final classifier: 1×1 conv to num_classes
    classifier_ = std::make_shared<nn::Conv2d>(256, num_classes, 1);
    register_module("classifier", classifier_);
}

auto DeepLabV3PlusDecoder::forward(const Variable& aspp_features,
                                   const Variable& low_level_features,
                                   int64_t output_h,
                                   int64_t output_w)
    -> Variable
{
    const auto& low_shape = low_level_features.tensor().shape();
    if (low_shape.size() != 4) {
        throw std::runtime_error("Low-level features must be 4D (N, C, H, W)");
    }

    int64_t target_h = low_shape[2];
    int64_t target_w = low_shape[3];

    // Upsample ASPP features by 4× to match low-level feature size
    auto upsampled = nn::upsample_bilinear(aspp_features, target_h, target_w);

    // Reduce low-level feature channels
    auto low_reduced = low_level_reduce_->forward(low_level_features);

    // Concatenate upsampled ASPP with reduced low-level features
    // Use gradient-aware cat() to preserve autograd graph
    std::vector<Variable> vars = {upsampled, low_reduced};
    auto concat = tenzor::cat(vars, 1);  // Uses CatBackward for gradients

    // Refine features
    auto refined = refine_->forward(concat);

    // Classify
    auto logits = classifier_->forward(refined);

    // Final upsample to the original input resolution. When the caller threads
    // the true input spatial size (output_h/output_w > 0) we upsample to that
    // exact size so the segmentation map aligns per-pixel with the input. The
    // low-level feature map is at stride 4, so target*4 only reconstructs the
    // input size when H/W are divisible by 4; for non-multiple-of-4 inputs that
    // estimate is off by 1-3 pixels. Falling back to target*4 only when no
    // explicit size is supplied preserves backward compatibility.
    const int64_t final_h = (output_h > 0) ? output_h : target_h * 4;
    const int64_t final_w = (output_w > 0) ? output_w : target_w * 4;
    auto output = nn::upsample_bilinear(logits, final_h, final_w);

    return output;
}

// ============================================================================
// DeepLabV3Plus Implementation
// ============================================================================

DeepLabV3Plus::DeepLabV3Plus(int64_t num_classes,
                             const std::string& backbone,
                             int64_t output_stride,
                             bool pretrained)
    : num_classes_(num_classes)
    , backbone_(backbone)
{
    // Create encoder
    encoder_ = std::make_shared<DeepLabV3PlusEncoder>(backbone, output_stride, pretrained);
    register_module("encoder", encoder_);

    // Create decoder
    decoder_ = std::make_shared<DeepLabV3PlusDecoder>(
            num_classes,
            encoder_->get_low_level_channels(),
            256  // ASPP output channels
        );
    register_module("decoder", decoder_);
}

auto DeepLabV3Plus::forward_impl(const Variable& input) -> Variable {
    // Capture the true input spatial size so the decoder upsamples the final
    // logits back to exactly (H, W) instead of inferring stride-4*4 = H'/W',
    // which drifts by 1-3 px for inputs whose H/W are not multiples of 4.
    const auto& in_shape = input.tensor().shape();
    if (in_shape.size() != 4) {
        throw std::runtime_error(
            "DeepLabV3Plus::forward_impl: input must be 4D (N, C, H, W)");
    }
    const int64_t input_h = in_shape[2];
    const int64_t input_w = in_shape[3];

    // Encode
    auto [aspp_features, low_level_features] = encoder_->forward_multi(input);

    // Decode, upsampling logits to the original input resolution.
    auto output =
        decoder_->forward(aspp_features, low_level_features, input_h, input_w);

    return output;
}

auto DeepLabV3Plus::predict(const Variable& input) -> Tensor {
    // Forward pass to get logits
    auto logits = forward(input);

    // Softmax is strictly monotonic per spatial position, so
    // argmax(softmax(x)) == argmax(x). Take argmax over the channel dim
    // directly on the logits and skip the redundant exp/sum/div over an
    // [N, num_classes, H, W] tensor on the inference path.
    auto predictions = argmax(logits.tensor(), 1);

    return predictions;
}

auto DeepLabV3Plus::load_pretrained(const std::string& path) -> void {
    // Load checkpoint from file using the checkpoint infrastructure
    nn::ModelCheckpoint checkpoint_manager;
    try {
        auto checkpoint = checkpoint_manager.load(path);

        // Validate checkpoint has model state
        if (checkpoint.model_state.empty()) {
            throw std::runtime_error("Checkpoint file contains no model state");
        }

        // Load state dictionary into the model
        load_state_dict(checkpoint.model_state);

    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to load pretrained DeepLabV3+ weights from '" + path + "': " + e.what() +
            "\n\nTo use pretrained weights:\n"
            "  1. Download pretrained weights (e.g., from PyTorch model zoo)\n"
            "  2. Convert to Tenzor checkpoint format\n"
            "  3. Ensure architecture matches (ResNet50/ResNet101 backbone, output_stride=16)\n"
            "For training from scratch, do not call load_pretrained()"
        );
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

auto DeepLabV3Plus_ResNet50(int64_t num_classes,
                            int64_t output_stride,
                            bool pretrained)
    -> std::shared_ptr<DeepLabV3Plus>
{
    return std::make_shared<DeepLabV3Plus>(
        num_classes, "resnet50", output_stride, pretrained);
}

auto DeepLabV3Plus_ResNet101(int64_t num_classes,
                             int64_t output_stride,
                             bool pretrained)
    -> std::shared_ptr<DeepLabV3Plus>
{
    return std::make_shared<DeepLabV3Plus>(
        num_classes, "resnet101", output_stride, pretrained);
}

auto DeepLabV3Plus_MobileNetV2(int64_t num_classes,
                               int64_t output_stride,
                               bool pretrained)
    -> std::shared_ptr<DeepLabV3Plus>
{
    return std::make_shared<DeepLabV3Plus>(
        num_classes, "mobilenetv2", output_stride, pretrained);
}

// Sibling factories that load the full-model COCO weights (encoder + decoder)
// via ModelHub. The hub registry has `deeplabv3_resnet50` / `_resnet101` /
// `_mobilenet_v3_large` keys pointing at torchvision's published COCO models.
auto deeplabv3_resnet50(int64_t num_classes,
                        int64_t output_stride,
                        bool pretrained) -> std::shared_ptr<DeepLabV3Plus>
{
    auto model = std::make_shared<DeepLabV3Plus>(
        num_classes, "resnet50", output_stride, /*pretrained=*/false);
    if (pretrained) {
        auto path = ModelHub::download_pretrained_safetensors("deeplabv3_resnet50");
        ModelHub::load_pretrained_weights(*model, path, /*strict=*/false);
    }
    return model;
}

auto deeplabv3_resnet101(int64_t num_classes,
                         int64_t output_stride,
                         bool pretrained) -> std::shared_ptr<DeepLabV3Plus>
{
    auto model = std::make_shared<DeepLabV3Plus>(
        num_classes, "resnet101", output_stride, /*pretrained=*/false);
    if (pretrained) {
        auto path = ModelHub::download_pretrained_safetensors("deeplabv3_resnet101");
        ModelHub::load_pretrained_weights(*model, path, /*strict=*/false);
    }
    return model;
}

} // namespace models
} // namespace tenzor
