/**
 * @file yolo.cpp
 * @brief Implementation of YOLO object detection models
 */

#include "../../include/tenzor/models/yolo.hpp"
#include "../../include/tenzor/models/hub.hpp"
#include "../../include/tenzor/nn/checkpoint.hpp"
#include "../../include/tenzor/autograd/ops.hpp"
#include "../../include/tenzor/nn/utils/variable_cast.hpp"
#include "../../include/tenzor/ops/detection.hpp"
#include "../../include/tenzor/ops/transform.hpp"
#include "../../include/tenzor/ops/math.hpp"
#include "../../include/tenzor/ops/vision.hpp"
#include "../../include/tenzor/utils/error.hpp"  // TENZOR_CHECK
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace tenzor {
namespace models {

using namespace nn;

using namespace ops;

// ============================================================================
// YOLOv3 Components Implementation
// ============================================================================

// DarknetResidualBlock
DarknetResidualBlock::DarknetResidualBlock(int64_t channels)
    : act_(0.1) {  // LeakyReLU with slope 0.1

    // 1x1 conv to reduce channels
    conv1_ = std::make_shared<Conv2d>(channels, channels / 2, 1, 1, 0, 1, 1, false);
    register_module("conv1", conv1_);

    bn1_ = std::make_shared<BatchNorm2d>(channels / 2);
    register_module("bn1", bn1_);

    // 3x3 conv to restore channels
    conv2_ = std::make_shared<Conv2d>(channels / 2, channels, 3, 1, 1, 1, 1, false);
    register_module("conv2", conv2_);

    bn2_ = std::make_shared<BatchNorm2d>(channels);
    register_module("bn2", bn2_);
}

auto DarknetResidualBlock::forward_impl(const Variable& input) -> Variable {
    Variable identity = input;

    auto out = conv1_->forward(input);
    out = bn1_->forward(out);
    out = act_.forward(out);

    out = conv2_->forward(out);
    out = bn2_->forward(out);
    out = act_.forward(out);

    // Add skip connection
    return out + identity;
}

// Darknet53
Darknet53::Darknet53(int64_t in_channels)
    : act_(0.1) {

    // Initial convolution: 3x3, stride 1
    conv1_ = std::make_shared<Conv2d>(in_channels, 32, 3, 1, 1, 1, 1, false);
    register_module("conv1", conv1_);

    bn1_ = std::make_shared<BatchNorm2d>(32);
    register_module("bn1", bn1_);

    // Build residual layers
    layer1_ = make_layer(32, 64, 1);    // 64 channels, 1 block
    layer2_ = make_layer(64, 128, 2);   // 128 channels, 2 blocks
    layer3_ = make_layer(128, 256, 8);  // 256 channels, 8 blocks (output 1/8)
    layer4_ = make_layer(256, 512, 8);  // 512 channels, 8 blocks (output 1/16)
    layer5_ = make_layer(512, 1024, 4); // 1024 channels, 4 blocks (output 1/32)

    // Register all layers
    int idx = 0;
    for (auto& layer : layer1_) register_module("layer1_" + std::to_string(idx++), layer);
    idx = 0;
    for (auto& layer : layer2_) register_module("layer2_" + std::to_string(idx++), layer);
    idx = 0;
    for (auto& layer : layer3_) register_module("layer3_" + std::to_string(idx++), layer);
    idx = 0;
    for (auto& layer : layer4_) register_module("layer4_" + std::to_string(idx++), layer);
    idx = 0;
    for (auto& layer : layer5_) register_module("layer5_" + std::to_string(idx++), layer);
}

auto Darknet53::make_layer(int64_t in_channels, int64_t out_channels, int64_t num_blocks)
    -> std::vector<std::shared_ptr<Module>> {

    std::vector<std::shared_ptr<Module>> layers;

    // Downsampling conv: 3x3, stride 2
    auto downsample = std::make_shared<Conv2d>(in_channels, out_channels, 3, 2, 1, 1, 1, false);
    layers.push_back(downsample);

    auto bn = std::make_shared<BatchNorm2d>(out_channels);
    layers.push_back(bn);

    // Residual blocks
    for (int64_t i = 0; i < num_blocks; ++i) {
        auto block = std::make_shared<DarknetResidualBlock>(out_channels);
        layers.push_back(block);
    }

    return layers;
}

auto Darknet53::forward_impl(const Variable& input) -> Variable {
    auto features = forward_multiscale(input);
    return features.back();  // Return final feature map
}

auto Darknet53::forward_multiscale(const Variable& input) -> std::vector<Variable> {
    // Initial conv
    auto x = conv1_->forward(input);
    x = bn1_->forward(x);
    x = act_.forward(x);

    // Layer 1: 64 channels
    for (auto& layer : layer1_) {
        x = layer->forward(x);
        if (auto bn = std::dynamic_pointer_cast<BatchNorm2d>(layer)) {
            x = act_.forward(x);
        }
    }

    // Layer 2: 128 channels
    for (auto& layer : layer2_) {
        x = layer->forward(x);
        if (auto bn = std::dynamic_pointer_cast<BatchNorm2d>(layer)) {
            x = act_.forward(x);
        }
    }

    // Layer 3: 256 channels (1/8 scale - for small objects)
    for (auto& layer : layer3_) {
        x = layer->forward(x);
        if (auto bn = std::dynamic_pointer_cast<BatchNorm2d>(layer)) {
            x = act_.forward(x);
        }
    }
    Variable feat_small = x;

    // Layer 4: 512 channels (1/16 scale - for medium objects)
    for (auto& layer : layer4_) {
        x = layer->forward(x);
        if (auto bn = std::dynamic_pointer_cast<BatchNorm2d>(layer)) {
            x = act_.forward(x);
        }
    }
    Variable feat_medium = x;

    // Layer 5: 1024 channels (1/32 scale - for large objects)
    for (auto& layer : layer5_) {
        x = layer->forward(x);
        if (auto bn = std::dynamic_pointer_cast<BatchNorm2d>(layer)) {
            x = act_.forward(x);
        }
    }
    Variable feat_large = x;

    return {feat_small, feat_medium, feat_large};
}

// YOLOv3Head
YOLOv3Head::YOLOv3Head(int64_t in_channels, int64_t num_classes, int64_t num_anchors)
    : num_anchors_(num_anchors), num_classes_(num_classes), act_(0.1) {

    // Intermediate conv layers
    conv1_ = std::make_shared<Conv2d>(in_channels, in_channels / 2, 1, 1, 0, 1, 1, false);
    register_module("conv1", conv1_);

    bn1_ = std::make_shared<BatchNorm2d>(in_channels / 2);
    register_module("bn1", bn1_);

    // Detection layer: outputs (num_anchors * (5 + num_classes)) channels
    // 5 = (tx, ty, tw, th, objectness)
    int64_t output_channels = num_anchors * (5 + num_classes);
    detect_ = std::make_shared<Conv2d>(in_channels / 2, output_channels, 1, 1, 0, 1, 1, true);
    register_module("detect", detect_);
}

auto YOLOv3Head::forward_impl(const Variable& input) -> Variable {
    auto x = conv1_->forward(input);
    x = bn1_->forward(x);
    x = act_.forward(x);

    auto detect = detect_->forward(x);

    // Reshape to (N, num_anchors, grid_h, grid_w, 5 + num_classes)
    auto shape = detect.tensor().shape();
    int64_t N = shape[0];
    int64_t grid_h = shape[2];
    int64_t grid_w = shape[3];

    // Permute from (N, C, H, W) to (N, H, W, C)
    detect = tenzor::permute(detect, {0, 2, 3, 1});

    // Reshape to (N, grid_h, grid_w, num_anchors, 5 + num_classes)
    detect = tenzor::reshape(detect, {N, grid_h, grid_w, num_anchors_, 5 + num_classes_});

    // Permute to (N, num_anchors, grid_h, grid_w, 5 + num_classes)
    detect = tenzor::permute(detect, {0, 3, 1, 2, 4});

    return detect;
}

// YOLOv3
YOLOv3::YOLOv3(int64_t num_classes, bool pretrained,
               double conf_threshold, double nms_threshold)
    : num_classes_(num_classes),
      conf_threshold_(conf_threshold),
      nms_threshold_(nms_threshold) {

    // Create backbone
    backbone_ = std::make_shared<Darknet53>(3);
    register_module("backbone", backbone_);

    // Build FPN neck
    build_fpn_neck();

    // Create detection heads for 3 scales
    head_large_ = std::make_shared<YOLOv3Head>(1024, num_classes, 3);  // 13x13
    register_module("head_large", head_large_);

    head_medium_ = std::make_shared<YOLOv3Head>(512, num_classes, 3);  // 26x26
    register_module("head_medium", head_medium_);

    head_small_ = std::make_shared<YOLOv3Head>(256, num_classes, 3);   // 52x52
    register_module("head_small", head_small_);

    // COCO anchors for 416x416 input
    // Large objects (13x13 grid, stride=32)
    anchors_large_ = {{116.0f, 90.0f}, {156.0f, 198.0f}, {373.0f, 326.0f}};

    // Medium objects (26x26 grid, stride=16)
    anchors_medium_ = {{30.0f, 61.0f}, {62.0f, 45.0f}, {59.0f, 119.0f}};

    // Small objects (52x52 grid, stride=8)
    anchors_small_ = {{10.0f, 13.0f}, {16.0f, 30.0f}, {33.0f, 23.0f}};

    if (pretrained) {
        load_pretrained("yolov3");
    }
}

auto YOLOv3::build_fpn_neck() -> void {
    // FPN upsampling layers
    fpn_upsample1_ = std::make_shared<Conv2d>(1024, 512, 1, 1, 0, 1, 1, false);
    register_module("fpn_upsample1", fpn_upsample1_);

    fpn_upsample2_ = std::make_shared<Conv2d>(512, 256, 1, 1, 0, 1, 1, false);
    register_module("fpn_upsample2", fpn_upsample2_);

    // Lateral connections
    fpn_lateral1_ = std::make_shared<Conv2d>(512, 512, 1, 1, 0, 1, 1, false);
    register_module("fpn_lateral1", fpn_lateral1_);

    fpn_lateral2_ = std::make_shared<Conv2d>(256, 256, 1, 1, 0, 1, 1, false);
    register_module("fpn_lateral2", fpn_lateral2_);
}

auto YOLOv3::forward_impl(const Variable& input) -> Variable {
    // YOLOv3's 53-layer backbone produces activations that exceed Float16's
    // 65504 saturation point even with an eval-mode BN warmup — the conv
    // outputs between BN layers can't be rescued by the BN widen path
    // alone. Autocast reduced-precision forwards to Float32 and cast the
    // final output back. This is the conservative approach used by
    // PyTorch's amp.autocast for deep detection backbones; training-mode
    // warmup calls benefit too so that the running stats are populated
    // in a numerically sane range.
    const DType in_dtype = input.tensor().dtype();
    const bool autocast = (in_dtype == DType::Float16 || in_dtype == DType::BFloat16);
    if (autocast) {
        // Promote module params/buffers to Float32 for the forward, then
        // restore on the way out so external state observers (parameter
        // snapshots, state_dict checks) continue to see the user-set
        // dtype.
        this->to(DType::Float32);
        // Use autograd-aware casts so the Float32 forward stays attached
        // to the input Variable and the Float16 output stays attached to
        // the Float32 forward's grad_fn chain. A raw `Variable(t.to(...))`
        // would drop grad_fn and silently break parameter grad flow
        // (observed as YOLOMultiDTypeTest.YOLOv3GradientFlow/Float16
        // reporting 0 / N parameters with gradients).
        Variable f32_input = nn::variable_cast(input, DType::Float32);
        auto result = forward_impl(f32_input);   // recurse with Float32 input
        this->to(in_dtype);
        return nn::variable_cast(result, in_dtype);
    }

    auto predictions = forward_raw(input);

    // For training, return raw predictions
    if (is_training()) {
        // Concatenate predictions from all scales
        // This would be used with YOLO loss function
        return predictions[0];  // Return raw predictions for loss computation
    }

    // For inference, decode and post-process
    auto img_size = input.tensor().shape()[2];
    auto boxes = decode_predictions(predictions, img_size);

    // Extract scores (objectness * max class probability)
    // Predictions format: [tx, ty, tw, th, objectness, class_probs...]
    std::vector<Tensor> all_scores;

    for (size_t si = 0; si < predictions.size(); ++si) {
        auto pred = predictions[si].tensor();
        auto shape = pred.shape();
        int64_t N = shape[0];
        int64_t num_anchors = shape[1];
        int64_t grid_h = shape[2];
        int64_t grid_w = shape[3];

        int64_t num_boxes = grid_h * grid_w * num_anchors;
        // Create scores with proper shape for batched_nms: {N, num_boxes, num_classes}
        Tensor scores = Tensor({N, num_boxes, num_classes_}, DType::Float32, Device::cpu());

        // Convert prediction to Float32 on CPU for data extraction
        auto pred_cpu = pred.to(Device::cpu()).to(DType::Float32);
        const float* pred_data = pred_cpu.data<float>();
        float* scores_data = scores.data<float>();

        // Extract objectness * class_prob for each box and class
        for (int64_t b = 0; b < N; ++b) {
            int64_t box_idx = 0;
            for (int64_t gy = 0; gy < grid_h; ++gy) {
                for (int64_t gx = 0; gx < grid_w; ++gx) {
                    for (int64_t a = 0; a < num_anchors; ++a) {
                        int64_t pred_idx = ((b * num_anchors + a) * grid_h + gy) * grid_w + gx;
                        int64_t pred_offset = pred_idx * (5 + num_classes_);

                        // Get objectness
                        float objectness = 1.0f / (1.0f + std::exp(-pred_data[pred_offset + 4]));

                        // Extract score for each class (objectness * class_prob)
                        for (int64_t c = 0; c < num_classes_; ++c) {
                            float class_prob = 1.0f / (1.0f + std::exp(-pred_data[pred_offset + 5 + c]));
                            int64_t score_idx = (b * num_boxes + box_idx) * num_classes_ + c;
                            scores_data[score_idx] = objectness * class_prob;
                        }
                        box_idx++;
                    }
                }
            }
        }

        // Move scores to original device
        all_scores.push_back(scores.to(pred.device()));
    }

    auto scores = tenzor::cat(all_scores, 1);

    // Apply post-processing
    auto detections = postprocess(boxes, scores);

    // Return first batch's detections as Variable
    return Variable(std::get<0>(detections[0]));
}

auto YOLOv3::forward_raw(const Variable& input) -> std::vector<Variable> {
    // Extract multi-scale features from backbone
    auto features = backbone_->forward_multiscale(input);
    auto feat_small = features[0];   // 256 channels, 1/8 scale (52x52)
    auto feat_medium = features[1];  // 512 channels, 1/16 scale (26x26)
    auto feat_large = features[2];   // 1024 channels, 1/32 scale (13x13)

    // Top-down FPN pathway
    // P5 -> P4: Upsample large features to match medium features
    auto p5_up = fpn_upsample1_->forward(feat_large);

    // Upsample by 2x using bilinear interpolation
    auto p5_shape = feat_medium.tensor().shape();
    int64_t target_h = p5_shape[2];
    int64_t target_w = p5_shape[3];
    auto p5_up_tensor = ops::interpolate(p5_up.tensor(), {target_h, target_w}, "bilinear", false);
    p5_up = Variable(p5_up_tensor, p5_up.requires_grad());

    auto p4_lateral = fpn_lateral1_->forward(feat_medium);
    auto p4 = p5_up + p4_lateral;

    // P4 -> P3: Upsample medium features to match small features
    auto p4_up = fpn_upsample2_->forward(p4);

    auto p3_shape = feat_small.tensor().shape();
    target_h = p3_shape[2];
    target_w = p3_shape[3];
    auto p4_up_tensor = ops::interpolate(p4_up.tensor(), {target_h, target_w}, "bilinear", false);
    p4_up = Variable(p4_up_tensor, p4_up.requires_grad());

    auto p3_lateral = fpn_lateral2_->forward(feat_small);
    auto p3 = p4_up + p3_lateral;

    // Apply detection heads
    auto pred_large = head_large_->forward(feat_large);   // 13x13
    auto pred_medium = head_medium_->forward(p4);         // 26x26
    auto pred_small = head_small_->forward(p3);           // 52x52

    return {pred_large, pred_medium, pred_small};
}

auto YOLOv3::decode_predictions(const std::vector<Variable>& predictions, int64_t img_size)
    -> Tensor {

    std::vector<Tensor> all_boxes;
    std::vector<int64_t> strides = {32, 16, 8};  // For large, medium, small

    for (size_t i = 0; i < predictions.size(); ++i) {
        auto pred = predictions[i].tensor();
        auto shape = pred.shape();
        int64_t N = shape[0];
        int64_t num_anchors = shape[1];
        int64_t grid_h = shape[2];
        int64_t grid_w = shape[3];
        int64_t stride = strides[i];

        // Get anchors for this scale
        const auto& anchors = (i == 0) ? anchors_large_ :
                             (i == 1) ? anchors_medium_ : anchors_small_;

        // Create output tensor for decoded boxes: [N, grid_h * grid_w * num_anchors, 4]
        int64_t num_boxes = grid_h * grid_w * num_anchors;
        Tensor boxes = Tensor({N, num_boxes, 4}, DType::Float32, pred.device());

        // Convert prediction to Float32 on CPU for data extraction
        auto pred_cpu = pred.to(Device::cpu()).to(DType::Float32);
        auto boxes_cpu = boxes.to(Device::cpu());

        // Get raw prediction data
        const float* pred_data = pred_cpu.data<float>();
        float* boxes_data = boxes_cpu.data<float>();

        // Decode boxes for each batch
        for (int64_t b = 0; b < N; ++b) {
            int64_t box_idx = 0;

            // Iterate over grid cells
            for (int64_t gy = 0; gy < grid_h; ++gy) {
                for (int64_t gx = 0; gx < grid_w; ++gx) {
                    // Iterate over anchors at this grid cell
                    for (int64_t a = 0; a < num_anchors; ++a) {
                        // Prediction format: [tx, ty, tw, th, objectness, class_probs...]
                        // Index into prediction tensor
                        int64_t pred_idx = ((b * num_anchors + a) * grid_h + gy) * grid_w + gx;
                        int64_t pred_offset = pred_idx * (5 + num_classes_);

                        // Extract box parameters
                        float tx = pred_data[pred_offset + 0];
                        float ty = pred_data[pred_offset + 1];
                        float tw = pred_data[pred_offset + 2];
                        float th = pred_data[pred_offset + 3];

                        // Apply sigmoid to tx, ty
                        float sigmoid_tx = 1.0f / (1.0f + std::exp(-tx));
                        float sigmoid_ty = 1.0f / (1.0f + std::exp(-ty));

                        // Compute box center
                        float bx = (gx + sigmoid_tx) * stride;
                        float by = (gy + sigmoid_ty) * stride;

                        // Compute box size using anchor dimensions
                        float anchor_w = anchors[a].first;
                        float anchor_h = anchors[a].second;
                        float bw = anchor_w * std::exp(tw);
                        float bh = anchor_h * std::exp(th);

                        // Convert to (x1, y1, x2, y2) format
                        float x1 = bx - bw / 2.0f;
                        float y1 = by - bh / 2.0f;
                        float x2 = bx + bw / 2.0f;
                        float y2 = by + bh / 2.0f;

                        // Clamp to image boundaries
                        x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_size)));
                        y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_size)));
                        x2 = std::max(0.0f, std::min(x2, static_cast<float>(img_size)));
                        y2 = std::max(0.0f, std::min(y2, static_cast<float>(img_size)));

                        // Write to output
                        int64_t out_offset = (b * num_boxes + box_idx) * 4;
                        boxes_data[out_offset + 0] = x1;
                        boxes_data[out_offset + 1] = y1;
                        boxes_data[out_offset + 2] = x2;
                        boxes_data[out_offset + 3] = y2;

                        box_idx++;
                    }
                }
            }
        }

        // Move back to original device if needed
        all_boxes.push_back(boxes_cpu.to(pred.device()));
    }

    // Concatenate all boxes from different scales
    return tenzor::cat(all_boxes, 1);
}

auto YOLOv3::postprocess(const Tensor& boxes, const Tensor& scores)
    -> std::vector<std::tuple<Tensor, Tensor, Tensor>> {

    std::vector<std::tuple<Tensor, Tensor, Tensor>> results;

    // Handle batch dimension: boxes is (batch, num_boxes, 4), scores is (batch, num_boxes, num_classes)
    if (boxes.ndim() == 3) {
        int64_t batch_size = boxes.shape()[0];

        for (int64_t b = 0; b < batch_size; ++b) {
            // Extract boxes and scores for this image: shape (num_boxes, 4) and (num_boxes, num_classes)
            auto img_boxes = boxes.slice(0, b, b + 1).squeeze(0);
            auto img_scores = scores.slice(0, b, b + 1).squeeze(0);

            auto [kept_boxes, kept_scores, kept_labels] =
                ops::batched_nms(img_boxes, img_scores, nms_threshold_, conf_threshold_, 100);

            results.emplace_back(kept_boxes, kept_scores, kept_labels);
        }
    } else {
        // Assume boxes is already (N, 4) and scores is (N, num_classes)
        auto [kept_boxes, kept_scores, kept_labels] =
            ops::batched_nms(boxes, scores, nms_threshold_, conf_threshold_, 100);

        results.emplace_back(kept_boxes, kept_scores, kept_labels);
    }

    return results;
}

auto YOLOv3::load_pretrained(const std::string& path) -> void {
    // Either a path-based local checkpoint, or a hub key (e.g. "yolov3") in
    // which case we route through ModelHub. Hub-keys are the path basename
    // without extension or directory.
    if (ModelHub::is_registered(path)) {
        auto cached = ModelHub::download_pretrained_safetensors(path);
        ModelHub::load_pretrained_weights(*this, cached, /*strict=*/false);
        return;
    }
    // Otherwise treat `path` as a literal filesystem path.
    nn::ModelCheckpoint mgr;
    auto ckpt = mgr.load(path);
    if (ckpt.model_state.empty()) {
        throw std::runtime_error("YOLOv3::load_pretrained: '" + path +
                                 "' contains no model state");
    }
    load_state_dict(ckpt.model_state);
}

// ============================================================================
// YOLOv5 Components Implementation
// ============================================================================

// CSPBottleneck
CSPBottleneck::CSPBottleneck(int64_t in_channels, int64_t out_channels,
                             int64_t num_blocks, [[maybe_unused]] bool shortcut) {

    int64_t hidden_channels = out_channels / 2;

    // Split path
    conv1_ = std::make_shared<Conv2d>(in_channels, hidden_channels, 1, 1, 0, 1, 1, false);
    register_module("conv1", conv1_);

    bn1_ = std::make_shared<BatchNorm2d>(hidden_channels);
    register_module("bn1", bn1_);

    // Main path
    conv2_ = std::make_shared<Conv2d>(in_channels, hidden_channels, 1, 1, 0, 1, 1, false);
    register_module("conv2", conv2_);

    bn2_ = std::make_shared<BatchNorm2d>(hidden_channels);
    register_module("bn2", bn2_);

    // Bottleneck blocks
    for (int64_t i = 0; i < num_blocks; ++i) {
        auto conv_a = std::make_shared<Conv2d>(hidden_channels, hidden_channels, 1, 1, 0, 1, 1, false);
        auto bn_a = std::make_shared<BatchNorm2d>(hidden_channels);
        auto conv_b = std::make_shared<Conv2d>(hidden_channels, hidden_channels, 3, 1, 1, 1, 1, false);
        auto bn_b = std::make_shared<BatchNorm2d>(hidden_channels);

        bottlenecks_.push_back(conv_a);
        bottlenecks_.push_back(bn_a);
        bottlenecks_.push_back(conv_b);
        bottlenecks_.push_back(bn_b);

        register_module("bottleneck_" + std::to_string(i * 4), conv_a);
        register_module("bottleneck_" + std::to_string(i * 4 + 1), bn_a);
        register_module("bottleneck_" + std::to_string(i * 4 + 2), conv_b);
        register_module("bottleneck_" + std::to_string(i * 4 + 3), bn_b);
    }

    // Merge path
    conv_merge_ = std::make_shared<Conv2d>(hidden_channels * 2, out_channels, 1, 1, 0, 1, 1, false);
    register_module("conv_merge", conv_merge_);

    bn_merge_ = std::make_shared<BatchNorm2d>(out_channels);
    register_module("bn_merge", bn_merge_);
}

auto CSPBottleneck::forward_impl(const Variable& input) -> Variable {
    nn::Swish act;  // SiLU activation (Swish)

    // Split path
    auto x1 = conv1_->forward(input);
    x1 = bn1_->forward(x1);
    x1 = act.forward(x1);

    // Main path with bottlenecks
    auto x2 = conv2_->forward(input);
    x2 = bn2_->forward(x2);
    x2 = act.forward(x2);

    // Apply bottleneck blocks
    for (size_t i = 0; i < bottlenecks_.size(); i += 4) {
        Variable identity = x2;

        x2 = bottlenecks_[i]->forward(x2);      // conv 1x1
        x2 = bottlenecks_[i + 1]->forward(x2);  // bn
        x2 = act.forward(x2);

        x2 = bottlenecks_[i + 2]->forward(x2);  // conv 3x3
        x2 = bottlenecks_[i + 3]->forward(x2);  // bn
        x2 = act.forward(x2);

        x2 = x2 + identity;  // Skip connection
    }

    // Concatenate and merge
    std::vector<Tensor> to_concat = {x1.tensor(), x2.tensor()};
    auto concat = tenzor::cat(to_concat, 1);
    auto merged = conv_merge_->forward(Variable(concat));
    merged = bn_merge_->forward(merged);
    merged = act.forward(merged);

    return merged;
}

// CSPDarknet
CSPDarknet::CSPDarknet(double depth_multiple, double width_multiple)
    : depth_multiple_(depth_multiple), width_multiple_(width_multiple) {

    int64_t base_channels = make_divisible(static_cast<int64_t>(64 * width_multiple));

    // Stem (Focus or Conv)
    stem_ = std::make_shared<Conv2d>(3, base_channels, 6, 2, 2, 1, 1, false);
    register_module("stem", stem_);

    bn_stem_ = std::make_shared<BatchNorm2d>(base_channels);
    register_module("bn_stem", bn_stem_);

    // Build CSP stages
    auto make_stage = [this](int64_t in_ch, int64_t out_ch, int64_t num_blocks) {
        std::vector<std::shared_ptr<Module>> stage;

        auto conv = std::make_shared<Conv2d>(in_ch, out_ch, 3, 2, 1, 1, 1, false);
        auto bn = std::make_shared<BatchNorm2d>(out_ch);
        auto csp = std::make_shared<CSPBottleneck>(out_ch, out_ch, num_blocks);

        stage.push_back(conv);
        stage.push_back(bn);
        stage.push_back(csp);

        return stage;
    };

    // YOLOv5 standard channel configuration
    int64_t ch1 = make_divisible(static_cast<int64_t>(128 * width_multiple_));
    int64_t ch2 = make_divisible(static_cast<int64_t>(256 * width_multiple_));
    int64_t ch3 = make_divisible(static_cast<int64_t>(256 * width_multiple_));  // P3 output
    int64_t ch4 = make_divisible(static_cast<int64_t>(512 * width_multiple_));  // P4 output
    int64_t ch5 = make_divisible(static_cast<int64_t>(1024 * width_multiple_)); // P5 output

    int64_t n1 = std::max(static_cast<int64_t>(1 * depth_multiple_), 1L);
    int64_t n2 = std::max(static_cast<int64_t>(3 * depth_multiple_), 1L);
    int64_t n3 = std::max(static_cast<int64_t>(9 * depth_multiple_), 1L);
    int64_t n4 = std::max(static_cast<int64_t>(9 * depth_multiple_), 1L);
    int64_t n5 = std::max(static_cast<int64_t>(3 * depth_multiple_), 1L);

    stage1_ = make_stage(base_channels, ch1, n1);
    stage2_ = make_stage(ch1, ch2, n2);
    stage3_ = make_stage(ch2, ch3, n3);  // P3 (1/8 scale)
    stage4_ = make_stage(ch3, ch4, n4);  // P4 (1/16 scale)
    stage5_ = make_stage(ch4, ch5, n5);  // P5 (1/32 scale)

    // Register stages
    for (size_t i = 0; i < stage1_.size(); ++i)
        register_module("stage1_" + std::to_string(i), stage1_[i]);
    for (size_t i = 0; i < stage2_.size(); ++i)
        register_module("stage2_" + std::to_string(i), stage2_[i]);
    for (size_t i = 0; i < stage3_.size(); ++i)
        register_module("stage3_" + std::to_string(i), stage3_[i]);
    for (size_t i = 0; i < stage4_.size(); ++i)
        register_module("stage4_" + std::to_string(i), stage4_[i]);
    for (size_t i = 0; i < stage5_.size(); ++i)
        register_module("stage5_" + std::to_string(i), stage5_[i]);
}

auto CSPDarknet::make_divisible(int64_t x, int64_t divisor) -> int64_t {
    return ((x + divisor / 2) / divisor) * divisor;
}

auto CSPDarknet::forward_impl(const Variable& input) -> Variable {
    auto features = forward_multiscale(input);
    return features.back();
}

auto CSPDarknet::forward_multiscale(const Variable& input) -> std::vector<Variable> {
    nn::Swish act;  // SiLU activation (Swish)

    // Stem
    auto x = stem_->forward(input);
    x = bn_stem_->forward(x);
    x = act.forward(x);

    // Stage 1
    for (auto& layer : stage1_) {
        x = layer->forward(x);
        if (std::dynamic_pointer_cast<BatchNorm2d>(layer)) {
            x = act.forward(x);
        }
    }

    // Stage 2
    for (auto& layer : stage2_) {
        x = layer->forward(x);
        if (std::dynamic_pointer_cast<BatchNorm2d>(layer)) {
            x = act.forward(x);
        }
    }

    // Stage 3 (P3 - 1/8 scale)
    for (auto& layer : stage3_) {
        x = layer->forward(x);
        if (std::dynamic_pointer_cast<BatchNorm2d>(layer)) {
            x = act.forward(x);
        }
    }
    Variable feat_p3 = x;

    // Stage 4 (P4 - 1/16 scale)
    for (auto& layer : stage4_) {
        x = layer->forward(x);
        if (std::dynamic_pointer_cast<BatchNorm2d>(layer)) {
            x = act.forward(x);
        }
    }
    Variable feat_p4 = x;

    // Stage 5 (P5 - 1/32 scale)
    for (auto& layer : stage5_) {
        x = layer->forward(x);
        if (std::dynamic_pointer_cast<BatchNorm2d>(layer)) {
            x = act.forward(x);
        }
    }
    Variable feat_p5 = x;

    return {feat_p3, feat_p4, feat_p5};
}

// PANet
PANet::PANet(const std::vector<int64_t>& channels) {
    // Top-down convs - reduce channels to match target layer
    td_conv1_ = std::make_shared<Conv2d>(channels[2], channels[1], 1, 1, 0, 1, 1, false);
    register_module("td_conv1", td_conv1_);

    td_conv2_ = std::make_shared<Conv2d>(channels[1], channels[0], 1, 1, 0, 1, 1, false);
    register_module("td_conv2", td_conv2_);

    // Bottom-up convs - increase channels to match target layer
    bu_conv1_ = std::make_shared<Conv2d>(channels[0], channels[1], 3, 2, 1, 1, 1, false);
    register_module("bu_conv1", bu_conv1_);

    bu_conv2_ = std::make_shared<Conv2d>(channels[1], channels[2], 3, 2, 1, 1, 1, false);
    register_module("bu_conv2", bu_conv2_);

    // Batch norms
    bn1_ = std::make_shared<BatchNorm2d>(channels[1]);
    bn2_ = std::make_shared<BatchNorm2d>(channels[0]);
    bn3_ = std::make_shared<BatchNorm2d>(channels[1]);
    bn4_ = std::make_shared<BatchNorm2d>(channels[2]);

    register_module("bn1", bn1_);
    register_module("bn2", bn2_);
    register_module("bn3", bn3_);
    register_module("bn4", bn4_);
}

auto PANet::forward_impl(const Variable& input) -> Variable {
    // PANet is a feature-pyramid fusion network and operates on multi-scale
    // features (typically P3/P4/P5 from the backbone). The previous
    // single-input pass-through constructed a 1-element vector and then
    // dereferenced indices [1] and [2] inside forward_multi → OOB read.
    //
    // A single-input forward is semantically ill-defined for PANet; callers
    // must use forward_multi(features) with at least 3 pyramid levels.
    TENZOR_CHECK(false,
                 "PANet::forward_impl single-input variant is not supported; "
                 "call forward_multi(features) with at least 3 pyramid "
                 "levels (P3, P4, P5) from the backbone.");
    return input;  // unreachable — TENZOR_CHECK throws on false condition.
}

auto PANet::forward_multi(const std::vector<Variable>& features) -> std::vector<Variable> {
    nn::Swish act;  // SiLU activation (Swish)
    auto p3 = features[0];
    auto p4 = features[1];
    auto p5 = features[2];

    // Top-down pathway: P5 -> P4
    auto p5_up = td_conv1_->forward(p5);
    p5_up = bn1_->forward(p5_up);
    p5_up = act.forward(p5_up);

    // Upsample p5_up to match p4 size
    auto p4_shape = p4.tensor().shape();
    auto p5_up_tensor = ops::interpolate(p5_up.tensor(), {p4_shape[2], p4_shape[3]}, "bilinear", false);
    p5_up = Variable(p5_up_tensor, p5_up.requires_grad());

    auto p4_fused = p4 + p5_up;

    // Top-down pathway: P4 -> P3
    auto p4_up = td_conv2_->forward(p4_fused);
    p4_up = bn2_->forward(p4_up);
    p4_up = act.forward(p4_up);

    // Upsample p4_up to match p3 size
    auto p3_shape = p3.tensor().shape();
    auto p4_up_tensor = ops::interpolate(p4_up.tensor(), {p3_shape[2], p3_shape[3]}, "bilinear", false);
    p4_up = Variable(p4_up_tensor, p4_up.requires_grad());

    auto p3_fused = p3 + p4_up;

    // Bottom-up pathway: P3 -> P4
    auto p3_down = bu_conv1_->forward(p3_fused);
    p3_down = bn3_->forward(p3_down);
    p3_down = act.forward(p3_down);
    auto p4_out = p4_fused + p3_down;

    // Bottom-up pathway: P4 -> P5
    auto p4_down = bu_conv2_->forward(p4_out);
    p4_down = bn4_->forward(p4_down);
    p4_down = act.forward(p4_down);
    auto p5_out = p5 + p4_down;

    return {p3_fused, p4_out, p5_out};
}

// YOLOv5Head
YOLOv5Head::YOLOv5Head(int64_t in_channels, int64_t num_classes, int64_t num_anchors)
    : num_anchors_(num_anchors), num_classes_(num_classes) {

    conv1_ = std::make_shared<Conv2d>(in_channels, in_channels / 2, 1, 1, 0, 1, 1, false);
    register_module("conv1", conv1_);

    bn1_ = std::make_shared<BatchNorm2d>(in_channels / 2);
    register_module("bn1", bn1_);

    int64_t output_channels = num_anchors * (5 + num_classes);
    detect_ = std::make_shared<Conv2d>(in_channels / 2, output_channels, 1, 1, 0, 1, 1, true);
    register_module("detect", detect_);
}

auto YOLOv5Head::forward_impl(const Variable& input) -> Variable {
    nn::Swish act;  // SiLU activation (Swish)
    auto x = conv1_->forward(input);
    x = bn1_->forward(x);
    x = act.forward(x);

    auto detect = detect_->forward(x);

    // Reshape similar to YOLOv3
    auto shape = detect.tensor().shape();
    int64_t N = shape[0];
    int64_t grid_h = shape[2];
    int64_t grid_w = shape[3];

    detect = tenzor::permute(detect, {0, 2, 3, 1});
    detect = tenzor::reshape(detect, {N, grid_h, grid_w, num_anchors_, 5 + num_classes_});
    detect = tenzor::permute(detect, {0, 3, 1, 2, 4});

    return detect;
}

// YOLOv5
YOLOv5::YOLOv5(Size size, int64_t num_classes, bool pretrained,
               double conf_threshold, double nms_threshold)
    : size_(size),
      num_classes_(num_classes),
      conf_threshold_(conf_threshold),
      nms_threshold_(nms_threshold) {

    auto [depth_mult, width_mult] = get_size_params(size);

    // Create backbone
    backbone_ = std::make_shared<CSPDarknet>(depth_mult, width_mult);
    register_module("backbone", backbone_);

    // Helper to match CSPDarknet's make_divisible logic
    auto make_divisible = [](int64_t x, int64_t divisor = 8) -> int64_t {
        return ((x + divisor / 2) / divisor) * divisor;
    };

    // Create neck - use same channel calculation as CSPDarknet
    int64_t ch_p3 = make_divisible(static_cast<int64_t>(256 * width_mult));
    int64_t ch_p4 = make_divisible(static_cast<int64_t>(512 * width_mult));
    int64_t ch_p5 = make_divisible(static_cast<int64_t>(1024 * width_mult));
    neck_ = std::make_shared<PANet>(std::vector<int64_t>{ch_p3, ch_p4, ch_p5});
    register_module("neck", neck_);

    // Create detection heads
    head_p3_ = std::make_shared<YOLOv5Head>(ch_p3, num_classes, 3);
    head_p4_ = std::make_shared<YOLOv5Head>(ch_p4, num_classes, 3);
    head_p5_ = std::make_shared<YOLOv5Head>(ch_p5, num_classes, 3);

    register_module("head_p3", head_p3_);
    register_module("head_p4", head_p4_);
    register_module("head_p5", head_p5_);

    // COCO anchors (auto-learned in practice)
    anchors_p3_ = {{10.0f, 13.0f}, {16.0f, 30.0f}, {33.0f, 23.0f}};
    anchors_p4_ = {{30.0f, 61.0f}, {62.0f, 45.0f}, {59.0f, 119.0f}};
    anchors_p5_ = {{116.0f, 90.0f}, {156.0f, 198.0f}, {373.0f, 326.0f}};

    if (pretrained) {
        // Map the Size enum to the hub registry key.
        const char* key = "yolov5s";
        switch (size) {
            case Size::Nano:    key = "yolov5n"; break;
            case Size::Small:   key = "yolov5s"; break;
            case Size::Medium:  key = "yolov5m"; break;
            case Size::Large:   key = "yolov5l"; break;
            case Size::XLarge:  key = "yolov5x"; break;
        }
        load_pretrained(key);
    }
}

// ----------------------------------------------------------------------------
// Audit G17: dataset-aware anchor refit via k=9 k-means.
//
// Ultralytics-style: distance = 1 - IoU between (w, h) pairs treated as boxes
// centered at the origin. IoU between (w₁, h₁) and (w₂, h₂) is
//   min(w₁, w₂) * min(h₁, h₂) / (w₁·h₁ + w₂·h₂ - min(w₁, w₂)·min(h₁, h₂))
// This rewards centroids whose aspect ratio AND size both match cluster
// members — which is the actual quantity anchor matching cares about.
//
// k-means seeding: pick 9 evenly spaced quantile box sizes (rather than
// random init) so results are deterministic.
// ----------------------------------------------------------------------------
auto YOLOv5::refit_anchors_kmeans(const std::vector<std::pair<float, float>>& box_sizes,
                                   int64_t iters) -> void {
    constexpr int K = 9;
    if (static_cast<int64_t>(box_sizes.size()) < K) {
        throw std::invalid_argument(
            "YOLOv5::refit_anchors_kmeans requires at least 9 boxes; got " +
            std::to_string(box_sizes.size()));
    }

    auto iou_wh = [](float w1, float h1, float w2, float h2) -> float {
        const float inter = std::min(w1, w2) * std::min(h1, h2);
        const float uni   = w1 * h1 + w2 * h2 - inter;
        return uni > 0.0f ? inter / uni : 0.0f;
    };

    // Seed centroids: pick 9 boxes spread by area (deterministic, robust).
    std::vector<std::pair<float, float>> sorted_by_area = box_sizes;
    std::sort(sorted_by_area.begin(), sorted_by_area.end(),
              [](auto& a, auto& b) { return a.first * a.second < b.first * b.second; });
    std::vector<std::pair<float, float>> centroids(K);
    const size_t N = sorted_by_area.size();
    for (int k = 0; k < K; ++k) {
        // Spread across the size distribution: indices 0, N/8, 2N/8, …, 7N/8.
        const size_t idx = std::min(N - 1, static_cast<size_t>((static_cast<double>(k) / K) * N));
        centroids[k] = sorted_by_area[idx];
    }

    // Lloyd's algorithm with 1-IoU distance.
    std::vector<int> assign(N, 0);
    for (int64_t it = 0; it < iters; ++it) {
        bool changed = false;
        for (size_t i = 0; i < N; ++i) {
            int best = 0;
            float best_iou = -1.0f;
            for (int k = 0; k < K; ++k) {
                float ki = iou_wh(box_sizes[i].first, box_sizes[i].second,
                                  centroids[k].first, centroids[k].second);
                if (ki > best_iou) { best_iou = ki; best = k; }
            }
            if (assign[i] != best) { assign[i] = best; changed = true; }
        }
        if (!changed) break;

        // Update centroids as median of cluster members. Median is more robust
        // than mean to outlier huge/tiny boxes (Ultralytics uses mean; median
        // is a reasonable variant).
        for (int k = 0; k < K; ++k) {
            std::vector<float> ws, hs;
            for (size_t i = 0; i < N; ++i) {
                if (assign[i] == k) {
                    ws.push_back(box_sizes[i].first);
                    hs.push_back(box_sizes[i].second);
                }
            }
            if (ws.empty()) continue;  // empty cluster: keep previous centroid.
            std::sort(ws.begin(), ws.end());
            std::sort(hs.begin(), hs.end());
            centroids[k] = {ws[ws.size() / 2], hs[hs.size() / 2]};
        }
    }

    // Sort the 9 centroids by area ascending, then assign 3 to each pyramid level.
    std::sort(centroids.begin(), centroids.end(),
              [](auto& a, auto& b) { return a.first * a.second < b.first * b.second; });
    anchors_p3_.assign(centroids.begin(),     centroids.begin() + 3);  // smallest
    anchors_p4_.assign(centroids.begin() + 3, centroids.begin() + 6);  // mid
    anchors_p5_.assign(centroids.begin() + 6, centroids.begin() + 9);  // largest
}

auto YOLOv5::get_size_params(Size size) -> std::pair<double, double> {
    switch (size) {
        case Size::Nano:   return {0.33, 0.25};
        case Size::Small:  return {0.33, 0.50};
        case Size::Medium: return {0.67, 0.75};
        case Size::Large:  return {1.00, 1.00};
        case Size::XLarge: return {1.33, 1.25};
        default:           return {0.33, 0.50};
    }
}

auto YOLOv5::forward_impl(const Variable& input) -> Variable {
    auto predictions = forward_raw(input);

    if (is_training()) {
        return predictions[0];
    }

    auto img_size = input.tensor().shape()[2];
    auto boxes = decode_predictions(predictions, img_size);

    // Extract scores (same as YOLOv3)
    std::vector<Tensor> all_scores;
    for (const auto& pred_var : predictions) {
        auto pred = pred_var.tensor();
        auto shape = pred.shape();
        int64_t N = shape[0];
        int64_t num_anchors = shape[1];
        int64_t grid_h = shape[2];
        int64_t grid_w = shape[3];

        int64_t num_boxes = grid_h * grid_w * num_anchors;
        // Create scores with proper shape for batched_nms: {N, num_boxes, num_classes}
        Tensor scores = Tensor({N, num_boxes, num_classes_}, DType::Float32, Device::cpu());

        // Convert prediction to Float32 on CPU for data extraction
        auto pred_cpu = pred.to(Device::cpu()).to(DType::Float32);
        const float* pred_data = pred_cpu.data<float>();
        float* scores_data = scores.data<float>();

        for (int64_t b = 0; b < N; ++b) {
            int64_t box_idx = 0;
            for (int64_t gy = 0; gy < grid_h; ++gy) {
                for (int64_t gx = 0; gx < grid_w; ++gx) {
                    for (int64_t a = 0; a < num_anchors; ++a) {
                        int64_t pred_idx = ((b * num_anchors + a) * grid_h + gy) * grid_w + gx;
                        int64_t pred_offset = pred_idx * (5 + num_classes_);

                        float objectness = 1.0f / (1.0f + std::exp(-pred_data[pred_offset + 4]));

                        // Extract score for each class (objectness * class_prob)
                        for (int64_t c = 0; c < num_classes_; ++c) {
                            float class_prob = 1.0f / (1.0f + std::exp(-pred_data[pred_offset + 5 + c]));
                            int64_t score_idx = (b * num_boxes + box_idx) * num_classes_ + c;
                            scores_data[score_idx] = objectness * class_prob;
                        }
                        box_idx++;
                    }
                }
            }
        }

        // Move scores to original device
        all_scores.push_back(scores.to(pred.device()));
    }

    auto scores = tenzor::cat(all_scores, 1);

    auto detections = postprocess(boxes, scores);
    return Variable(std::get<0>(detections[0]));
}

auto YOLOv5::forward_raw(const Variable& input) -> std::vector<Variable> {
    auto features = backbone_->forward_multiscale(input);
    auto fused_features = neck_->forward_multi(features);

    auto pred_p3 = head_p3_->forward(fused_features[0]);
    auto pred_p4 = head_p4_->forward(fused_features[1]);
    auto pred_p5 = head_p5_->forward(fused_features[2]);

    return {pred_p3, pred_p4, pred_p5};
}

auto YOLOv5::decode_predictions(const std::vector<Variable>& predictions, int64_t img_size)
    -> Tensor {
    std::vector<Tensor> all_boxes;
    std::vector<int64_t> strides = {8, 16, 32};  // P3, P4, P5

    for (size_t i = 0; i < predictions.size(); ++i) {
        auto pred = predictions[i].tensor();
        auto shape = pred.shape();
        int64_t N = shape[0];
        int64_t num_anchors = shape[1];
        int64_t grid_h = shape[2];
        int64_t grid_w = shape[3];
        int64_t stride = strides[i];

        // Get anchors for this scale
        const auto& anchors = (i == 0) ? anchors_p3_ :
                              (i == 1) ? anchors_p4_ : anchors_p5_;

        // Create output tensor for decoded boxes: [N, grid_h * grid_w * num_anchors, 4]
        int64_t num_boxes = grid_h * grid_w * num_anchors;
        Tensor boxes = Tensor({N, num_boxes, 4}, DType::Float32, pred.device());

        // Convert prediction to Float32 on CPU for data extraction
        auto pred_cpu = pred.to(Device::cpu()).to(DType::Float32);
        auto boxes_cpu = boxes.to(Device::cpu());

        // Get raw prediction data
        const float* pred_data = pred_cpu.data<float>();
        float* boxes_data = boxes_cpu.data<float>();

        // Decode boxes for each batch
        for (int64_t b = 0; b < N; ++b) {
            int64_t box_idx = 0;

            // Iterate over grid cells
            for (int64_t gy = 0; gy < grid_h; ++gy) {
                for (int64_t gx = 0; gx < grid_w; ++gx) {
                    // Iterate over anchors at this grid cell
                    for (int64_t a = 0; a < num_anchors; ++a) {
                        // Prediction format: [tx, ty, tw, th, objectness, class_probs...]
                        int64_t pred_idx = ((b * num_anchors + a) * grid_h + gy) * grid_w + gx;
                        int64_t pred_offset = pred_idx * (5 + num_classes_);

                        // Extract box parameters
                        float tx = pred_data[pred_offset + 0];
                        float ty = pred_data[pred_offset + 1];
                        float tw = pred_data[pred_offset + 2];
                        float th = pred_data[pred_offset + 3];

                        // YOLOv5 uses sigmoid for tx, ty
                        float sigmoid_tx = 1.0f / (1.0f + std::exp(-tx));
                        float sigmoid_ty = 1.0f / (1.0f + std::exp(-ty));

                        // Compute box center
                        float bx = (gx + sigmoid_tx) * stride;
                        float by = (gy + sigmoid_ty) * stride;

                        // Compute box size using anchor dimensions
                        float anchor_w = anchors[a].first;
                        float anchor_h = anchors[a].second;
                        float bw = anchor_w * std::exp(tw);
                        float bh = anchor_h * std::exp(th);

                        // Convert to (x1, y1, x2, y2) format
                        float x1 = bx - bw / 2.0f;
                        float y1 = by - bh / 2.0f;
                        float x2 = bx + bw / 2.0f;
                        float y2 = by + bh / 2.0f;

                        // Clamp to image boundaries
                        x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_size)));
                        y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_size)));
                        x2 = std::max(0.0f, std::min(x2, static_cast<float>(img_size)));
                        y2 = std::max(0.0f, std::min(y2, static_cast<float>(img_size)));

                        // Write to output
                        int64_t out_offset = (b * num_boxes + box_idx) * 4;
                        boxes_data[out_offset + 0] = x1;
                        boxes_data[out_offset + 1] = y1;
                        boxes_data[out_offset + 2] = x2;
                        boxes_data[out_offset + 3] = y2;

                        box_idx++;
                    }
                }
            }
        }

        // Move back to original device if needed
        all_boxes.push_back(boxes_cpu.to(pred.device()));
    }

    return tenzor::cat(all_boxes, 1);
}

auto YOLOv5::postprocess(const Tensor& boxes, const Tensor& scores)
    -> std::vector<std::tuple<Tensor, Tensor, Tensor>> {

    std::vector<std::tuple<Tensor, Tensor, Tensor>> results;

    // Handle batch dimension: boxes is (batch, num_boxes, 4), scores is (batch, num_boxes, num_classes)
    if (boxes.ndim() == 3) {
        int64_t batch_size = boxes.shape()[0];

        for (int64_t b = 0; b < batch_size; ++b) {
            // Extract boxes and scores for this image: shape (num_boxes, 4) and (num_boxes, num_classes)
            auto img_boxes = boxes.slice(0, b, b + 1).squeeze(0);
            auto img_scores = scores.slice(0, b, b + 1).squeeze(0);

            auto [kept_boxes, kept_scores, kept_labels] =
                ops::batched_nms(img_boxes, img_scores, nms_threshold_, conf_threshold_, 100);

            results.emplace_back(kept_boxes, kept_scores, kept_labels);
        }
    } else {
        // Assume boxes is already (N, 4) and scores is (N, num_classes)
        auto [kept_boxes, kept_scores, kept_labels] =
            ops::batched_nms(boxes, scores, nms_threshold_, conf_threshold_, 100);

        results.emplace_back(kept_boxes, kept_scores, kept_labels);
    }

    return results;
}

auto YOLOv5::load_pretrained(const std::string& path) -> void {
    if (ModelHub::is_registered(path)) {
        auto cached = ModelHub::download_pretrained_safetensors(path);
        ModelHub::load_pretrained_weights(*this, cached, /*strict=*/false);
        return;
    }
    nn::ModelCheckpoint mgr;
    auto ckpt = mgr.load(path);
    if (ckpt.model_state.empty()) {
        throw std::runtime_error("YOLOv5::load_pretrained: '" + path +
                                 "' contains no model state");
    }
    load_state_dict(ckpt.model_state);
}

// ============================================================================
// Factory Functions
// ============================================================================

auto yolov3(int64_t num_classes, bool pretrained) -> std::shared_ptr<YOLOv3> {
    return std::make_shared<YOLOv3>(num_classes, pretrained);
}

auto yolov5n(int64_t num_classes, bool pretrained) -> std::shared_ptr<YOLOv5> {
    return std::make_shared<YOLOv5>(YOLOv5::Size::Nano, num_classes, pretrained);
}

auto yolov5s(int64_t num_classes, bool pretrained) -> std::shared_ptr<YOLOv5> {
    return std::make_shared<YOLOv5>(YOLOv5::Size::Small, num_classes, pretrained);
}

auto yolov5m(int64_t num_classes, bool pretrained) -> std::shared_ptr<YOLOv5> {
    return std::make_shared<YOLOv5>(YOLOv5::Size::Medium, num_classes, pretrained);
}

auto yolov5l(int64_t num_classes, bool pretrained) -> std::shared_ptr<YOLOv5> {
    return std::make_shared<YOLOv5>(YOLOv5::Size::Large, num_classes, pretrained);
}

auto yolov5x(int64_t num_classes, bool pretrained) -> std::shared_ptr<YOLOv5> {
    return std::make_shared<YOLOv5>(YOLOv5::Size::XLarge, num_classes, pretrained);
}

} // namespace models
} // namespace tenzor
