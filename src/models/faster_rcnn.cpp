/**
 * @file faster_rcnn.cpp
 * @brief Faster R-CNN implementation
 */

#include "tenzor/models/faster_rcnn.hpp"
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
    int64_t roi_detections_per_img)
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

    // Determine backbone output channels
    // For ResNet-50/101, layer4 output is 512 * Bottleneck::expansion = 512 * 4 = 2048
    // For ResNet-18/34, layer4 output is 512 * BasicBlock::expansion = 512 * 1 = 512
    int64_t backbone_out_channels = 2048;  // Default for ResNet-50/101

    // Check if backbone is ResNet and adjust channels accordingly
    auto resnet = std::dynamic_pointer_cast<ResNet>(backbone_);
    if (resnet) {
        // ResNet-50/101/152 use Bottleneck (expansion=4): 512 * 4 = 2048
        // ResNet-18/34 use BasicBlock (expansion=1): 512 * 1 = 512
        // We detect this by checking the actual output when possible
        // For now, assume ResNet-50/101 (Bottleneck) with 2048 channels
        backbone_out_channels = 2048;
    }

    // Create RPN
    rpn_ = std::make_shared<nn::detection::RegionProposalNetwork>(
        backbone_out_channels,
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
        backbone_out_channels,
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

    // Call inference and return dummy tensor
    // (Actual detections should be retrieved via forward_inference)
    auto detections = forward_inference(input);

    // Return dummy output (Module interface requires Variable return)
    return Variable(
        ops::zeros({1}, input.dtype(), input.device()),
        false
    );
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

    // Create ResNet-50 backbone
    // For detection, we typically use features before the final
    // classification layer (layer4 output)
    auto resnet = resnet50(1000, pretrained);

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

    return model;
}

auto faster_rcnn_resnet101(
    int64_t num_classes,
    bool pretrained,
    const std::string& pretrained_backbone)
    -> std::shared_ptr<FasterRCNN> {

    // Create ResNet-101 backbone
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
    [[maybe_unused]] int64_t backbone_out_channels,
    int64_t num_classes)
    -> std::shared_ptr<FasterRCNN> {

    // Create Faster R-CNN with custom backbone
    auto model = std::make_shared<FasterRCNN>(
        backbone,
        num_classes,
        std::vector<float>{32.0f, 64.0f, 128.0f, 256.0f, 512.0f},
        std::vector<float>{0.5f, 1.0f, 2.0f},
        0.7, 0.3, 256, 0.5, 2000, 1000, 0.7,
        7, 1.0 / 16.0, 2,
        0.5, 0.5, 512, 0.25, 0.05, 0.5, 100
    );

    return model;
}

} // namespace models
} // namespace tenzor
