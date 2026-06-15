/**
 * @file faster_rcnn.cpp
 * @brief Faster R-CNN implementation
 */

#include "tenzor/models/faster_rcnn.hpp"
#include "tenzor/models/hub.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include <stdexcept>
#include <iostream>

namespace tenzor {
namespace models {

// ============================================================================
// FasterRCNN Implementation
// ============================================================================

FasterRCNN::FasterRCNN(
    std::shared_ptr<nn::Module> backbone,
    int64_t num_classes,
    std::vector<float> rpn_anchor_sizes,
    std::vector<float> rpn_aspect_ratios,
    double rpn_fg_iou_thresh,
    double rpn_bg_iou_thresh,
    int64_t rpn_batch_size_per_image,
    double rpn_positive_fraction,
    int64_t rpn_pre_nms_top_n,
    int64_t rpn_post_nms_top_n,
    double rpn_nms_thresh,
    int64_t roi_output_size,
    double roi_spatial_scale,
    int64_t roi_sampling_ratio,
    double roi_fg_iou_thresh,
    double roi_bg_iou_thresh,
    int64_t roi_batch_size_per_image,
    double roi_positive_fraction,
    double roi_score_thresh,
    double roi_nms_thresh,
    int64_t roi_detections_per_img,
    int64_t backbone_out_channels)
    : num_classes_(num_classes),
      backbone_(std::move(backbone)) {

    if (!backbone_) {
        throw std::invalid_argument("Backbone cannot be null");
    }

    register_module("backbone", backbone_);

    // Create anchor generator
    auto anchor_generator = std::make_shared<nn::detection::AnchorGenerator>(
        rpn_anchor_sizes, rpn_aspect_ratios
    );

    // Audit G8: query the backbone for its actual terminal channel count
    // instead of hard-coding 2048 for every variant. The previous code
    // silently sized the RPN for Bottleneck (2048 ch) even when the backbone
    // was ResNet-18/34 (BasicBlock, 512 ch) — which would cause a Conv2d
    // shape mismatch at the first RPN forward pass.
    //
    // Resolution order:
    //   1. An explicit caller-supplied channel count (backbone_out_channels >= 0)
    //      always wins — this is the only correct path for non-ResNet custom
    //      backbones whose channels cannot be introspected.
    //   2. Otherwise introspect a ResNet backbone via out_channels().
    //   3. Otherwise fall back to 2048 (Bottleneck default).
    int64_t resolved_out_channels = backbone_out_channels;
    if (resolved_out_channels < 0) {
        resolved_out_channels = 2048;
        if (auto resnet = std::dynamic_pointer_cast<ResNet>(backbone_)) {
            resolved_out_channels = resnet->out_channels();
        }
    }

    // Create RPN
    rpn_ = std::make_shared<nn::detection::RegionProposalNetwork>(
        resolved_out_channels,
        anchor_generator,
        rpn_fg_iou_thresh,
        rpn_bg_iou_thresh,
        rpn_batch_size_per_image,
        rpn_positive_fraction,
        rpn_pre_nms_top_n,
        rpn_post_nms_top_n,
        rpn_nms_thresh,
        0.0  // score_thresh
    );
    register_module("rpn", rpn_);

    // Create ROI Head
    roi_head_ = std::make_shared<nn::detection::RoIHead>(
        resolved_out_channels,
        num_classes,
        roi_output_size,
        roi_spatial_scale,
        roi_sampling_ratio,
        roi_fg_iou_thresh,
        roi_bg_iou_thresh,
        roi_batch_size_per_image,
        roi_positive_fraction,
        roi_score_thresh,
        roi_nms_thresh,
        roi_detections_per_img
    );
    register_module("roi_head", roi_head_);
}

auto FasterRCNN::get_image_shapes(const Variable& images)
    -> std::vector<std::pair<int64_t, int64_t>> {

    auto batch_size = images.shape()[0];
    auto height = images.shape()[2];
    auto width = images.shape()[3];

    std::vector<std::pair<int64_t, int64_t>> shapes;
    shapes.reserve(batch_size);

    for (int64_t i = 0; i < batch_size; ++i) {
        shapes.emplace_back(height, width);
    }

    return shapes;
}

auto FasterRCNN::forward_inference(
    const Variable& images,
    const std::vector<std::pair<int64_t, int64_t>>* image_shapes)
    -> std::vector<std::unordered_map<std::string, Tensor>> {

    if (is_training()) {
        throw std::runtime_error(
            "forward_inference called in training mode. "
            "Use eval() or forward_train() instead."
        );
    }

    // Get image shapes
    auto shapes = image_shapes ? *image_shapes : get_image_shapes(images);

    // Extract features from backbone
    // Try to use forward_features if backbone is ResNet, otherwise use forward
    Variable features;
    auto resnet = std::dynamic_pointer_cast<ResNet>(backbone_);
    if (resnet) {
        features = resnet->forward_features(images);
    } else {
        features = backbone_->forward(images);
    }

    // Generate proposals with RPN
    auto proposals = rpn_->forward_proposals(features, shapes);

    // Get detections from ROI head
    auto detections = roi_head_->forward_detections(
        features, proposals, shapes
    );

    return detections;
}

auto FasterRCNN::forward_train(
    const Variable& images,
    const std::vector<std::unordered_map<std::string, Tensor>>& targets,
    const std::vector<std::pair<int64_t, int64_t>>* image_shapes)
    -> std::unordered_map<std::string, Variable> {

    if (!is_training()) {
        throw std::runtime_error(
            "forward_train called in eval mode. Use train() first."
        );
    }

    // Get image shapes
    auto shapes = image_shapes ? *image_shapes : get_image_shapes(images);

    // Extract ground truth boxes and labels from targets
    std::vector<Tensor> gt_boxes;
    std::vector<Tensor> gt_labels;
    gt_boxes.reserve(targets.size());
    gt_labels.reserve(targets.size());
    for (const auto& target : targets) {
        if (target.find("boxes") == target.end()) {
            throw std::invalid_argument("Target missing 'boxes' key");
        }
        if (target.find("labels") == target.end()) {
            throw std::invalid_argument("Target missing 'labels' key");
        }
        gt_boxes.push_back(target.at("boxes"));
        gt_labels.push_back(target.at("labels"));
    }

    // Extract features from backbone
    // Try to use forward_features if backbone is ResNet, otherwise use forward
    Variable features;
    auto resnet = std::dynamic_pointer_cast<ResNet>(backbone_);
    if (resnet) {
        features = resnet->forward_features(images);
    } else {
        features = backbone_->forward(images);
    }

    // Generate proposals with RPN (computes RPN losses)
    auto proposals = rpn_->forward_proposals(features, shapes, &gt_boxes);

    // Get detections from ROI head (computes ROI losses)
    // ROI head needs both boxes and labels for proper class assignment
    roi_head_->forward_detections(features, proposals, shapes, &gt_boxes, &gt_labels);

    // Collect all losses
    std::unordered_map<std::string, Variable> all_losses;

    // RPN losses
    auto rpn_losses = rpn_->get_losses();
    all_losses["loss_objectness"] = rpn_losses["loss_objectness"];
    all_losses["loss_rpn_box_reg"] = rpn_losses["loss_rpn_box_reg"];

    // ROI losses
    auto roi_losses = roi_head_->get_losses();
    all_losses["loss_classifier"] = roi_losses["loss_classifier"];
    all_losses["loss_box_reg"] = roi_losses["loss_box_reg"];

    return all_losses;
}

auto FasterRCNN::forward_impl(const Variable& input) -> Variable {
    if (is_training()) {
        throw std::runtime_error(
            "Use forward_train() in training mode with targets"
        );
    }

    // Audit G9: replace the `Variable(zeros({1}), ...)` dummy with a real
    // packed output. Each row is `(batch_idx, x1, y1, x2, y2, score, label)`
    // — the YOLOv5 (N, 6) `[x1, y1, x2, y2, score, label]` convention prefixed
    // by an image index, since Faster R-CNN's `forward_inference` returns one
    // dict per image and we need to flatten across the batch into a single
    // Variable. The structured per-image output remains available via
    // `forward_inference` for callers that prefer keyed access.
    auto per_image = forward_inference(input);

    const DType  dtype  = input.dtype();
    const Device device = input.device();

    std::vector<Tensor> rows;
    rows.reserve(per_image.size());
    for (int64_t b = 0; b < static_cast<int64_t>(per_image.size()); ++b) {
        const auto& det = per_image[static_cast<size_t>(b)];
        auto bit = det.find("boxes");
        auto lit = det.find("labels");
        auto sit = det.find("scores");
        if (bit == det.end() || lit == det.end() || sit == det.end()) continue;
        const int64_t K = bit->second.shape()[0];
        if (K == 0) continue;

        Tensor bx     = bit->second.to(dtype).to(device).reshape({K, 4});
        Tensor sc_col = sit->second.to(dtype).to(device).reshape({K, 1});
        Tensor lb_col = lit->second.to(dtype).to(device).reshape({K, 1});
        Tensor b_col  = ops::full({K, 1}, static_cast<double>(b), dtype, device);

        rows.push_back(tenzor::cat({b_col, bx, sc_col, lb_col}, /*dim=*/1));  // (K, 7)
    }

    if (rows.empty()) {
        return Variable(Tensor({0, 7}, dtype, device), false);
    }
    return Variable(tenzor::cat(rows, /*dim=*/0), false);
}

auto FasterRCNN::load_pretrained(const std::string& path) -> void {
    // Load state dictionary from file
    load(path);
}

// ============================================================================
// Factory Functions
// ============================================================================

auto faster_rcnn_resnet50(
    int64_t num_classes,
    bool pretrained,
    const std::string& pretrained_backbone)
    -> std::shared_ptr<FasterRCNN> {

    // When loading full-model COCO weights below, the backbone is overwritten.
    // Start with random backbone in that case; only set pretrained=true here
    // if no full-model weights will be loaded.
    auto resnet = resnet50(1000, /*pretrained=*/false);

    if (!pretrained_backbone.empty()) {
        resnet->load_pretrained(pretrained_backbone);
    }

    // Create Faster R-CNN with default parameters
    auto model = std::make_shared<FasterRCNN>(
        resnet,
        num_classes,
        std::vector<float>{32.0f, 64.0f, 128.0f, 256.0f, 512.0f},  // anchor sizes
        std::vector<float>{0.5f, 1.0f, 2.0f},                       // aspect ratios
        0.7,   // rpn_fg_iou_thresh
        0.3,   // rpn_bg_iou_thresh
        256,   // rpn_batch_size_per_image
        0.5,   // rpn_positive_fraction
        2000,  // rpn_pre_nms_top_n
        1000,  // rpn_post_nms_top_n
        0.7,   // rpn_nms_thresh
        7,     // roi_output_size
        1.0 / 16.0,  // roi_spatial_scale (ResNet C4)
        2,     // roi_sampling_ratio
        0.5,   // roi_fg_iou_thresh
        0.5,   // roi_bg_iou_thresh
        512,   // roi_batch_size_per_image
        0.25,  // roi_positive_fraction
        0.05,  // roi_score_thresh
        0.5,   // roi_nms_thresh
        100    // roi_detections_per_img
    );

    if (pretrained) {
        auto path = ModelHub::download_pretrained_safetensors("faster_rcnn_resnet50_fpn");
        ModelHub::load_pretrained_weights(*model, path, /*strict=*/false);
    }

    return model;
}

auto faster_rcnn_resnet101(
    int64_t num_classes,
    bool pretrained,
    const std::string& pretrained_backbone)
    -> std::shared_ptr<FasterRCNN> {

    // No full-model COCO weights for ResNet-101 variant in the registry —
    // backbone-only ImageNet pretraining is the canonical path here.
    auto resnet = resnet101(1000, pretrained);

    if (!pretrained_backbone.empty()) {
        resnet->load_pretrained(pretrained_backbone);
    }

    // Create Faster R-CNN with default parameters
    auto model = std::make_shared<FasterRCNN>(
        resnet,
        num_classes,
        std::vector<float>{32.0f, 64.0f, 128.0f, 256.0f, 512.0f},
        std::vector<float>{0.5f, 1.0f, 2.0f},
        0.7,   // rpn_fg_iou_thresh
        0.3,   // rpn_bg_iou_thresh
        256,   // rpn_batch_size_per_image
        0.5,   // rpn_positive_fraction
        2000,  // rpn_pre_nms_top_n
        1000,  // rpn_post_nms_top_n
        0.7,   // rpn_nms_thresh
        7,     // roi_output_size
        1.0 / 16.0,  // roi_spatial_scale
        2,     // roi_sampling_ratio
        0.5,   // roi_fg_iou_thresh
        0.5,   // roi_bg_iou_thresh
        512,   // roi_batch_size_per_image
        0.25,  // roi_positive_fraction
        0.05,  // roi_score_thresh
        0.5,   // roi_nms_thresh
        100    // roi_detections_per_img
    );

    return model;
}

auto faster_rcnn_custom(
    std::shared_ptr<nn::Module> backbone,
    int64_t backbone_out_channels,
    int64_t num_classes)
    -> std::shared_ptr<FasterRCNN> {

    // Create Faster R-CNN with custom backbone. The caller-supplied channel
    // count is threaded through so the RPN/ROIHead Conv2d are sized for the
    // actual backbone output (the whole point of the custom factory) rather
    // than the hard-coded 2048 ResNet-Bottleneck default.
    auto model = std::make_shared<FasterRCNN>(
        backbone,
        num_classes,
        std::vector<float>{32.0f, 64.0f, 128.0f, 256.0f, 512.0f},
        std::vector<float>{0.5f, 1.0f, 2.0f},
        0.7, 0.3, 256, 0.5, 2000, 1000, 0.7,
        7, 1.0 / 16.0, 2,
        0.5, 0.5, 512, 0.25, 0.05, 0.5, 100,
        backbone_out_channels
    );

    return model;
}

} // namespace models
} // namespace tenzor
