/**
 * @file mask_rcnn.cpp
 * @brief Implementation of Mask R-CNN for instance segmentation
 */

#include "tenzor/models/mask_rcnn.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/detection.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/vision.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/loss/losses.hpp"
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace tenzor {
namespace models {

// ============================================================================
// Helper Functions for Loss Computation
// ============================================================================

namespace {

/**
 * @brief Compute Smooth L1 loss for bounding box regression.
 *
 * Smooth L1 = 0.5 * (x - y)^2 / beta    if |x - y| < beta
 *           = |x - y| - 0.5 * beta      otherwise
 */
auto smooth_l1_loss(const Tensor& input, const Tensor& target, float beta = 1.0f) -> Tensor {
    auto diff = input - target;
    auto abs_diff = tenzor::abs(diff);

    // Create mask for elements where |diff| < beta
    auto beta_tensor = tenzor::ones_like(abs_diff) * beta;
    auto mask = abs_diff < beta_tensor;

    // Compute l2_loss = 0.5 * diff^2 / beta
    auto diff_sq = diff * diff;
    auto l2_loss = diff_sq * (0.5f / beta);

    // Compute l1_loss = |diff| - 0.5 * beta
    auto l1_loss = abs_diff - (beta * 0.5f);

    // Select based on mask: use l2 where |diff| < beta, else l1
    auto loss = tenzor::where(mask, l2_loss, l1_loss);

    return loss;
}

/**
 * @brief Encode boxes relative to anchors for regression targets.
 *
 * Converts (x1, y1, x2, y2) to (dx, dy, dw, dh) relative to anchors.
 */
auto encode_boxes(const Tensor& boxes, const Tensor& anchors) -> Tensor {
    // boxes: (N, 4) as (x1, y1, x2, y2)
    // anchors: (N, 4) as (x1, y1, x2, y2)

    auto N = boxes.shape()[0];
    auto device = boxes.device();

    // Extract box coordinates
    auto boxes_x1 = tenzor::select(boxes, 1, 0);
    auto boxes_y1 = tenzor::select(boxes, 1, 1);
    auto boxes_x2 = tenzor::select(boxes, 1, 2);
    auto boxes_y2 = tenzor::select(boxes, 1, 3);

    auto anchors_x1 = tenzor::select(anchors, 1, 0);
    auto anchors_y1 = tenzor::select(anchors, 1, 1);
    auto anchors_x2 = tenzor::select(anchors, 1, 2);
    auto anchors_y2 = tenzor::select(anchors, 1, 3);

    // Compute box centers and sizes
    auto boxes_w = boxes_x2 - boxes_x1;
    auto boxes_h = boxes_y2 - boxes_y1;
    auto boxes_cx = boxes_x1 + boxes_w * 0.5f;
    auto boxes_cy = boxes_y1 + boxes_h * 0.5f;

    auto anchors_w = anchors_x2 - anchors_x1;
    auto anchors_h = anchors_y2 - anchors_y1;
    auto anchors_cx = anchors_x1 + anchors_w * 0.5f;
    auto anchors_cy = anchors_y1 + anchors_h * 0.5f;

    // Avoid division by zero
    auto eps = 1e-6f;
    anchors_w = tenzor::clamp_min(anchors_w, eps);
    anchors_h = tenzor::clamp_min(anchors_h, eps);
    boxes_w = tenzor::clamp_min(boxes_w, eps);
    boxes_h = tenzor::clamp_min(boxes_h, eps);

    // Encode: dx, dy, dw, dh
    auto dx = (boxes_cx - anchors_cx) / anchors_w;
    auto dy = (boxes_cy - anchors_cy) / anchors_h;
    auto dw = tenzor::log(boxes_w / anchors_w);
    auto dh = tenzor::log(boxes_h / anchors_h);

    // Stack into (N, 4)
    std::vector<Tensor> deltas = {dx, dy, dw, dh};
    auto encoded = ops::stack(deltas, 1);

    return encoded;
}

/**
 * @brief Decode box deltas to absolute coordinates.
 */
auto decode_boxes(const Tensor& deltas, const Tensor& anchors) -> Tensor {
    // deltas: (N, 4) as (dx, dy, dw, dh)
    // anchors: (N, 4) as (x1, y1, x2, y2)

    auto N = deltas.shape()[0];

    // Extract deltas
    auto dx = tenzor::select(deltas, 1, 0);
    auto dy = tenzor::select(deltas, 1, 1);
    auto dw = tenzor::select(deltas, 1, 2);
    auto dh = tenzor::select(deltas, 1, 3);

    // Extract anchor coordinates
    auto anchors_x1 = tenzor::select(anchors, 1, 0);
    auto anchors_y1 = tenzor::select(anchors, 1, 1);
    auto anchors_x2 = tenzor::select(anchors, 1, 2);
    auto anchors_y2 = tenzor::select(anchors, 1, 3);

    // Compute anchor centers and sizes
    auto anchors_w = anchors_x2 - anchors_x1;
    auto anchors_h = anchors_y2 - anchors_y1;
    auto anchors_cx = anchors_x1 + anchors_w * 0.5f;
    auto anchors_cy = anchors_y1 + anchors_h * 0.5f;

    // Avoid division by zero
    auto eps = 1e-6f;
    anchors_w = tenzor::clamp_min(anchors_w, eps);
    anchors_h = tenzor::clamp_min(anchors_h, eps);

    // Decode to box centers and sizes
    auto boxes_cx = dx * anchors_w + anchors_cx;
    auto boxes_cy = dy * anchors_h + anchors_cy;
    auto boxes_w = tenzor::exp(dw) * anchors_w;
    auto boxes_h = tenzor::exp(dh) * anchors_h;

    // Convert to (x1, y1, x2, y2)
    auto boxes_x1 = boxes_cx - boxes_w * 0.5f;
    auto boxes_y1 = boxes_cy - boxes_h * 0.5f;
    auto boxes_x2 = boxes_cx + boxes_w * 0.5f;
    auto boxes_y2 = boxes_cy + boxes_h * 0.5f;

    // Stack into (N, 4)
    std::vector<Tensor> coords = {boxes_x1, boxes_y1, boxes_x2, boxes_y2};
    auto decoded = ops::stack(coords, 1);

    return decoded;
}

/**
 * @brief Assign anchors to ground truth boxes based on IoU.
 *
 * Returns:
 * - labels: (N,) with 1 for positive, 0 for negative, -1 for ignore
 * - matched_gt_boxes: (N, 4) ground truth boxes matched to each anchor
 */
auto assign_anchors_to_targets(
    const Tensor& anchors,
    const Tensor& gt_boxes,
    float pos_threshold = 0.7f,
    float neg_threshold = 0.3f
) -> std::tuple<Tensor, Tensor> {
    // anchors: (N, 4)
    // gt_boxes: (M, 4)

    auto N = anchors.shape()[0];
    auto M = gt_boxes.shape()[0];
    auto device = anchors.device();

    if (M == 0) {
        // No ground truth boxes - all anchors are negative
        auto labels = Tensor({N}, DType::Int64, device);
        labels.fill_(0);  // All negative
        auto matched_boxes = tenzor::zeros({N, 4}, anchors.dtype(), device);
        return std::make_tuple(labels, matched_boxes);
    }

    // Compute IoU matrix: (N, M)
    auto iou_matrix = ops::box_iou(anchors, gt_boxes);

    // For each anchor, find best matching GT box
    auto max_iou = tenzor::max(iou_matrix, 1);  // (N,)
    auto matched_idx = tenzor::argmax(iou_matrix, 1);  // (N,)

    // Create labels on CPU for data access, then move to device
    auto labels_cpu = Tensor({N}, DType::Int64, Device::cpu());
    auto* labels_data = labels_cpu.data<int64_t>();
    // Convert max_iou to Float32 on CPU for data access
    auto max_iou_f32 = max_iou.to(Device::cpu()).to(DType::Float32);
    auto* max_iou_data = max_iou_f32.data<float>();

    for (int64_t i = 0; i < N; ++i) {
        if (max_iou_data[i] >= pos_threshold) {
            labels_data[i] = 1;  // Positive
        } else if (max_iou_data[i] < neg_threshold) {
            labels_data[i] = 0;  // Negative
        } else {
            labels_data[i] = -1;  // Ignore (between thresholds)
        }
    }

    // Move labels to target device
    auto labels = labels_cpu.to(device);

    // Gather matched GT boxes
    auto matched_boxes = tenzor::index_select(gt_boxes, 0, matched_idx);

    return std::make_tuple(labels, matched_boxes);
}

} // anonymous namespace

// Forward declarations for loss computation functions
auto compute_rpn_loss(
    const Tensor& objectness_logits,
    const Tensor& box_regression,
    const std::vector<Tensor>& targets,
    const Tensor& anchors
) -> std::tuple<Variable, Variable>;

auto compute_roi_head_loss(
    const Tensor& class_logits,
    const Tensor& box_regression,
    const std::vector<Tensor>& targets
) -> std::tuple<Variable, Variable>;

auto compute_mask_loss(
    const Tensor& mask_logits,
    const std::vector<Tensor>& targets
) -> Variable;

// ============================================================================
// RPN Implementation
// ============================================================================

RPN::RPN(int64_t in_channels, int64_t num_anchors) {
    // 3×3 conv for spatial context
    conv_ = std::make_shared<nn::Conv2d>(in_channels, 512, 3, 1, 1);
    register_module("conv", conv_);

    // Classification: objectness scores (2 per anchor: bg/fg)
    cls_logits_ = std::make_shared<nn::Conv2d>(512, num_anchors * 2, 1);
    register_module("cls_logits", cls_logits_);

    // Regression: box deltas (4 per anchor: dx, dy, dw, dh)
    bbox_pred_ = std::make_shared<nn::Conv2d>(512, num_anchors * 4, 1);
    register_module("bbox_pred", bbox_pred_);
}

auto RPN::forward_multi(const Variable& features)
    -> std::tuple<Variable, Variable> {
    // features: (N, C, H, W)

    // Apply 3×3 conv with ReLU
    auto x = nn::relu(conv_->forward(features));

    // Get objectness scores and box deltas
    auto cls = cls_logits_->forward(x);  // (N, num_anchors*2, H, W)
    auto bbox = bbox_pred_->forward(x);  // (N, num_anchors*4, H, W)

    // Reshape to (N, H*W*num_anchors, 2) and (N, H*W*num_anchors, 4)
    auto N = cls.tensor().shape()[0];
    auto H = cls.tensor().shape()[2];
    auto W = cls.tensor().shape()[3];
    auto num_anchors = cls.tensor().shape()[1] / 2;

    // Permute (N, num_anchors*2, H, W) -> (N, H, W, num_anchors*2)
    cls = cls.permute({0, 2, 3, 1});
    bbox = bbox.permute({0, 2, 3, 1});

    // Reshape to (N, H*W*num_anchors, 2) and (N, H*W*num_anchors, 4)
    cls = cls.reshape(std::vector<int64_t>{N, H * W * num_anchors, 2});
    bbox = bbox.reshape(std::vector<int64_t>{N, H * W * num_anchors, 4});

    return std::make_tuple(cls, bbox);
}

// ============================================================================
// ROI Head Implementation
// ============================================================================

ROIHead::ROIHead(int64_t in_channels, int64_t num_classes, int64_t roi_size) {
    auto feature_dim = in_channels * roi_size * roi_size;

    // Two FC layers
    fc1_ = std::make_shared<nn::Linear>(feature_dim, 1024);
    register_module("fc1", fc1_);

    fc2_ = std::make_shared<nn::Linear>(1024, 1024);
    register_module("fc2", fc2_);

    // Classification head (num_classes including background)
    cls_score_ = std::make_shared<nn::Linear>(1024, num_classes + 1);
    register_module("cls_score", cls_score_);

    // Box regression head (4 coordinates per class)
    bbox_pred_ = std::make_shared<nn::Linear>(1024, (num_classes + 1) * 4);
    register_module("bbox_pred", bbox_pred_);
}

auto ROIHead::forward_multi(const Variable& roi_features)
    -> std::tuple<Variable, Variable> {
    // roi_features: (num_rois, C, roi_size, roi_size)

    // Flatten
    auto N = roi_features.tensor().shape()[0];
    auto x = roi_features.reshape(std::vector<int64_t>{N, -1});

    // FC layers with ReLU
    x = nn::relu(fc1_->forward(x));
    x = nn::relu(fc2_->forward(x));

    // Classification and box regression
    auto cls_logits = cls_score_->forward(x);  // (num_rois, num_classes+1)
    auto bbox_deltas = bbox_pred_->forward(x);  // (num_rois, (num_classes+1)*4)

    return std::make_tuple(cls_logits, bbox_deltas);
}

// ============================================================================
// Mask R-CNN Implementation
// ============================================================================

MaskRCNN::MaskRCNN(std::shared_ptr<nn::Module> backbone,
                   int64_t num_classes,
                   int64_t min_size,
                   int64_t max_size,
                   int64_t rpn_pre_nms_top_n_train,
                   int64_t rpn_pre_nms_top_n_test,
                   int64_t rpn_post_nms_top_n_train,
                   int64_t rpn_post_nms_top_n_test,
                   double rpn_nms_thresh,
                   double box_score_thresh,
                   double box_nms_thresh,
                   int64_t box_detections_per_img)
    : backbone_(backbone)
    , num_classes_(num_classes)
    , min_size_(min_size)
    , max_size_(max_size)
    , rpn_pre_nms_top_n_train_(rpn_pre_nms_top_n_train)
    , rpn_pre_nms_top_n_test_(rpn_pre_nms_top_n_test)
    , rpn_post_nms_top_n_train_(rpn_post_nms_top_n_train)
    , rpn_post_nms_top_n_test_(rpn_post_nms_top_n_test)
    , rpn_nms_thresh_(rpn_nms_thresh)
    , box_score_thresh_(box_score_thresh)
    , box_nms_thresh_(box_nms_thresh)
    , box_detections_per_img_(box_detections_per_img) {

    // Register backbone
    register_module("backbone", backbone_);

    // Create feature projection layer to convert ResNet output (2048 channels) to FPN channels (256)
    // ResNet Bottleneck layer4 outputs 512 * 4 = 2048 channels
    feature_proj_ = std::make_shared<nn::Conv2d>(2048, 256, 1, 1, 0);  // 1x1 conv, stride=1, padding=0
    register_module("feature_proj", feature_proj_);

    // Create RPN (assumes backbone output has 256 channels for FPN)
    // Anchor generator uses 3 sizes × 3 aspect ratios = 9 anchors per location
    auto num_anchors = 9;
    rpn_ = std::make_shared<RPN>(256, num_anchors);
    register_module("rpn", rpn_);

    // Create anchor generator
    // Standard Faster R-CNN anchors: 3 scales × 3 aspect ratios = 9 per location
    // For FPN, we typically use 3 anchors per level
    anchor_generator_ = std::make_shared<nn::detection::AnchorGenerator>(
        std::vector<float>{32.0f, 64.0f, 128.0f},  // sizes
        std::vector<float>{0.5f, 1.0f, 2.0f}       // aspect ratios
    );

    // ROI Align for boxes: 7×7 output, 1/16 spatial scale
    roi_align_box_ = std::make_shared<nn::detection::ROIAlign>(
        7, 7, 1.0 / 16.0, 2, true
    );

    // ROI Align for masks: 14×14 output, 1/16 spatial scale
    roi_align_mask_ = std::make_shared<nn::detection::ROIAlign>(
        14, 14, 1.0 / 16.0, 2, true
    );

    // ROI Head for box classification and regression
    roi_head_ = std::make_shared<ROIHead>(256, num_classes, 7);
    register_module("roi_head", roi_head_);

    // Mask Head for mask prediction
    mask_head_ = std::make_shared<nn::detection::MaskHead>(256, num_classes);
    register_module("mask_head", mask_head_);
}

auto MaskRCNN::forward_impl(const Variable& images) -> Variable {
    if (is_training()) {
        throw std::runtime_error(
            "MaskRCNN::forward() requires ground truth targets during training. "
            "Use forward_train(images, gt_boxes, gt_labels, gt_masks) instead."
        );
    }

    // Inference mode
    auto [boxes, labels, scores, masks] = forward_test(images);

    // For now, return a dummy variable
    // In a real implementation, we'd package this into a proper output format
    return Variable(boxes, false);
}

auto MaskRCNN::forward_train(const Variable& images,
                              const Tensor& gt_boxes,
                              const Tensor& gt_labels,
                              const Tensor& gt_masks)
    -> std::tuple<Variable, Variable, Variable, Variable, Variable> {

    // 1. Extract features from backbone
    auto features = extract_features(images);

    // 2. Generate proposals with RPN
    auto [rpn_cls_logits, rpn_bbox_deltas] = rpn_->forward_multi(features);

    // Generate anchor boxes on the same device as features
    auto H = features.tensor().shape()[2];
    auto W = features.tensor().shape()[3];
    auto anchors = anchor_generator_->generate(H, W, 16, features.tensor().device());  // stride=16

    // Compute RPN losses
    // Prepare ground truth boxes for each image in batch
    auto batch_size = images.tensor().shape()[0];
    std::vector<Tensor> rpn_targets;

    for (int64_t b = 0; b < batch_size; ++b) {
        // Extract GT boxes for this image: gt_boxes[b, :, :]
        auto gt_boxes_b = tenzor::select(gt_boxes, 0, b);  // (max_objects, 4)
        rpn_targets.push_back(gt_boxes_b);
    }

    auto [rpn_cls_loss, rpn_bbox_loss] = compute_rpn_loss(
        rpn_cls_logits.tensor(),
        rpn_bbox_deltas.tensor(),
        rpn_targets,
        anchors
    );

    // 3. Generate proposals from RPN output
    auto proposals = generate_proposals(features);

    // 4. Select training samples (positive and negative ROIs)
    auto sampled_rois = select_training_samples(proposals, gt_boxes, gt_labels);

    // 5. ROI Align for box head (7×7)
    auto roi_features_box = roi_align_box_->forward(features, sampled_rois);

    // 6. Box classification and regression
    auto [cls_logits, bbox_deltas] = roi_head_->forward_multi(roi_features_box);

    // 7. Match sampled ROIs to ground truth to get their labels and target boxes
    auto num_sampled = sampled_rois.shape()[0];
    auto original_device = sampled_rois.device();

    // Create labels and boxes on CPU for data manipulation
    auto sampled_labels_cpu = Tensor({num_sampled}, DType::Int64, Device::cpu());
    auto sampled_target_boxes_cpu = Tensor({num_sampled, 4}, DType::Float32, Device::cpu());

    // Initialize all to background
    auto* sampled_labels_data = sampled_labels_cpu.data<int64_t>();
    for (int64_t i = 0; i < num_sampled; ++i) {
        sampled_labels_data[i] = 0;
    }

    // For each image in batch, match ROIs to GT boxes
    // Simplified: assign first few ROIs to GT boxes based on IoU
    // In production, this would use proper ROI sampling with IoU matching
    auto num_gt = gt_boxes.shape()[1];

    if (num_gt > 0 && num_sampled > 0) {
        // Extract GT boxes and labels from first image (simplified for single-image batch)
        auto gt_boxes_0 = tenzor::select(gt_boxes, 0, 0);  // (num_gt, 4)
        auto gt_labels_0 = tenzor::select(gt_labels, 0, 0);  // (num_gt,)

        // Extract ROI boxes (remove batch index column if present)
        // sampled_rois format: (num_sampled, 5) as (batch_idx, x1, y1, x2, y2)
        auto roi_boxes = sampled_rois.slice(1, 1, 5);  // (num_sampled, 4)

        // Compute IoU between ROIs and GT boxes
        auto iou_matrix = ops::box_iou(roi_boxes, gt_boxes_0);  // (num_sampled, num_gt)

        // Assign each ROI to best matching GT box
        auto matched_gt_idx = tenzor::argmax(iou_matrix, 1);  // (num_sampled,)
        auto max_iou = tenzor::max(iou_matrix, 1);  // (num_sampled,)

        // Move all tensors to CPU for data access
        auto matched_idx_cpu = matched_gt_idx.to(Device::cpu());
        auto* matched_idx_data = matched_idx_cpu.data<int64_t>();
        auto max_iou_f32 = max_iou.to(Device::cpu()).to(DType::Float32);
        auto* max_iou_data = max_iou_f32.data<float>();
        auto gt_labels_cpu = gt_labels_0.to(Device::cpu());
        auto* gt_labels_data = gt_labels_cpu.data<int64_t>();

        // Convert GT boxes to Float32 on CPU for data access
        auto gt_boxes_0_f32 = gt_boxes_0.to(Device::cpu()).to(DType::Float32);
        auto* gt_boxes_data = gt_boxes_0_f32.data<float>();

        // Use the CPU target boxes buffer
        auto* target_boxes_data = sampled_target_boxes_cpu.data<float>();

        // Assign labels and target boxes based on IoU threshold
        for (int64_t i = 0; i < num_sampled; ++i) {
            if (max_iou_data[i] >= 0.5f) {  // Positive threshold
                int64_t gt_idx = matched_idx_data[i];
                sampled_labels_data[i] = gt_labels_data[gt_idx];

                // Copy target box coordinates
                for (int j = 0; j < 4; ++j) {
                    target_boxes_data[i * 4 + j] = gt_boxes_data[gt_idx * 4 + j];
                }
            }
        }
    }

    // Move CPU tensors to target device
    auto sampled_labels = sampled_labels_cpu.to(original_device);
    auto sampled_target_boxes = sampled_target_boxes_cpu.to(sampled_rois.dtype()).to(original_device);

    // Compute ROI head losses
    std::vector<Tensor> roi_targets = {sampled_labels, sampled_target_boxes};
    auto [roi_cls_loss, roi_bbox_loss] = compute_roi_head_loss(
        cls_logits.tensor(),
        bbox_deltas.tensor(),
        roi_targets
    );

    // 8. ROI Align for mask head (14×14, only for positive samples)
    auto roi_features_mask = roi_align_mask_->forward(features, sampled_rois);

    // 8. Mask prediction
    auto mask_logits = mask_head_->forward(roi_features_mask);

    // 9. Create sampled_masks with correct shape to match mask_logits output
    // mask_logits has shape [num_sampled, num_classes, mask_H, mask_W]
    // where mask_H and mask_W are typically 28x28 for mask head output
    auto mask_output_H = mask_logits.tensor().shape()[2];
    auto mask_output_W = mask_logits.tensor().shape()[3];

    // Create masks on CPU for data manipulation
    auto sampled_masks_cpu = Tensor({num_sampled, mask_output_H, mask_output_W},
                                    DType::Float32, Device::cpu());
    sampled_masks_cpu.fill_(0.0f);

    // Simplified mask sampling: for foreground ROIs, extract corresponding GT masks
    if (num_gt > 0) {
        // Extract GT masks for first image
        auto gt_masks_0 = tenzor::select(gt_masks, 0, 0);  // (num_gt, H, W)

        for (int64_t i = 0; i < num_sampled; ++i) {
            if (sampled_labels_data[i] > 0) {  // Foreground ROI (using CPU labels)
                // Find best matching GT box
                auto roi_box = sampled_rois.slice(0, i, i+1).slice(1, 1, 5);  // (1, 4)

                auto gt_boxes_0 = tenzor::select(gt_boxes, 0, 0);
                auto iou = ops::box_iou(roi_box.squeeze(0).unsqueeze(0), gt_boxes_0);
                auto best_gt_idx = tenzor::argmax(iou, 1).to(Device::cpu()).item<int64_t>();

                if (best_gt_idx < num_gt) {
                    // Extract GT mask and resize to match mask head output
                    auto gt_mask = tenzor::select(gt_masks_0, 0, best_gt_idx);  // (H, W)

                    // Resize GT mask to (mask_output_H, mask_output_W)
                    auto resized_mask = ops::interpolate(
                        gt_mask.unsqueeze(0).unsqueeze(0),  // (1, 1, H, W)
                        std::vector<int64_t>{mask_output_H, mask_output_W},
                        "bilinear",
                        false
                    );
                    resized_mask = resized_mask.squeeze(0).squeeze(0).to(DType::Float32).to(Device::cpu());

                    // Copy to sampled_masks_cpu
                    auto target_mask_cpu = tenzor::select(sampled_masks_cpu, 0, i);
                    auto* resized_data = resized_mask.data<float>();
                    auto* target_data = target_mask_cpu.data<float>();
                    std::copy(resized_data, resized_data + mask_output_H * mask_output_W, target_data);
                }
            }
        }
    }

    // Move sampled_masks to target device
    auto sampled_masks = sampled_masks_cpu.to(original_device);

    // 10. Compute mask loss
    // Convert sampled_masks to match mask_logits dtype
    auto sampled_masks_converted = sampled_masks.to(mask_logits.tensor().dtype());
    std::vector<Tensor> mask_targets = {sampled_masks_converted, sampled_labels};
    auto mask_loss_val = compute_mask_loss(
        mask_logits.tensor(),
        mask_targets
    );

    return std::make_tuple(
        rpn_cls_loss,
        rpn_bbox_loss,
        roi_cls_loss,
        roi_bbox_loss,
        mask_loss_val
    );
}

auto MaskRCNN::forward_test(const Variable& images)
    -> std::tuple<Tensor, Tensor, Tensor, Tensor> {

    // 1. Extract features
    auto features = extract_features(images);

    // 2. Generate proposals
    auto [rpn_cls_logits, rpn_bbox_deltas] = rpn_->forward_multi(features);
    auto proposals = generate_proposals(features);

    // 3. ROI Align for boxes
    auto roi_features_box = roi_align_box_->forward(features, proposals);

    // 4. Box classification and regression
    auto [cls_logits, bbox_deltas] = roi_head_->forward_multi(roi_features_box);

    // 5. Apply NMS and get final detections
    auto cls_probs = tenzor::softmax(cls_logits, 1).tensor();

    // For simplicity, take top predictions
    // In production, this would apply proper NMS per class
    auto scores = tenzor::max(cls_probs, 1);
    auto labels = tenzor::argmax(cls_probs, 1);

    // Decode boxes (simplified - would normally use decode_boxes)
    auto boxes = proposals.slice(1, 1, 5);  // Remove batch index

    // 6. ROI Align for masks (only for detected objects)
    auto roi_features_mask = roi_align_mask_->forward(features, proposals);

    // 7. Mask prediction
    auto mask_logits = mask_head_->forward(roi_features_mask);

    // 8. Post-process masks
    auto image_h = images.tensor().shape()[2];
    auto image_w = images.tensor().shape()[3];
    auto masks = nn::detection::process_masks(
        mask_logits.tensor(),
        boxes,
        labels,
        image_h,
        image_w
    );

    return std::make_tuple(boxes, labels, scores, masks);
}

auto compute_rpn_loss(
    const Tensor& objectness_logits,
    const Tensor& box_regression,
    const std::vector<Tensor>& targets,
    const Tensor& anchors
) -> std::tuple<Variable, Variable> {
    // objectness_logits: (N, num_anchors*H*W, 2)
    // box_regression: (N, num_anchors*H*W, 4)
    // targets: vector of {boxes, labels} per image
    // anchors: (num_anchors*H*W, 4)

    auto batch_size = objectness_logits.shape()[0];
    auto num_anchors = objectness_logits.shape()[1];
    auto device = objectness_logits.device();

    // Collect losses for all images in batch (keep as Variables for gradient tracking)
    std::vector<Variable> cls_losses;
    std::vector<Variable> bbox_losses;

    for (int64_t b = 0; b < batch_size; ++b) {
        // Get ground truth for this image
        if (b >= static_cast<int64_t>(targets.size())) {
            continue;  // No GT for this image
        }

        auto gt_boxes = targets[b];  // (M, 4)

        // Ensure anchors match gt_boxes dtype for multi-dtype support
        auto anchors_typed = (anchors.dtype() != gt_boxes.dtype())
            ? anchors.to(gt_boxes.dtype())
            : anchors;

        // Assign anchors to GT boxes
        auto [labels, matched_boxes] = assign_anchors_to_targets(
            anchors_typed, gt_boxes, 0.7f, 0.3f
        );

        // Get objectness and bbox predictions for this image
        auto obj_logits_b = tenzor::select(objectness_logits, 0, b);  // (num_anchors, 2)
        auto bbox_reg_b = tenzor::select(box_regression, 0, b);  // (num_anchors, 4)

        // Classification loss: Binary cross-entropy for objectness
        // Move labels to CPU for data access
        auto num_total = labels.shape()[0];
        auto labels_cpu = labels.to(Device::cpu());
        auto* labels_data = labels_cpu.data<int64_t>();

        // Count positive and negative samples
        int64_t num_pos = 0;
        int64_t num_neg = 0;
        for (int64_t i = 0; i < num_total; ++i) {
            if (labels_data[i] == 1) num_pos++;
            else if (labels_data[i] == 0) num_neg++;
        }

        if (num_pos + num_neg == 0) {
            // No valid anchors, skip
            continue;
        }

        // Sample anchors: use all positives, subsample negatives for balance
        // Standard: 256 samples with 1:1 positive:negative ratio
        int64_t max_samples = 256;
        int64_t target_pos = std::min(num_pos, max_samples / 2);
        int64_t target_neg = std::min(num_neg, max_samples - target_pos);

        // Create mask for sampled anchors
        std::vector<int64_t> sampled_indices;
        int64_t pos_count = 0;
        int64_t neg_count = 0;

        for (int64_t i = 0; i < num_total; ++i) {
            if (labels_data[i] == 1 && pos_count < target_pos) {
                sampled_indices.push_back(i);
                pos_count++;
            } else if (labels_data[i] == 0 && neg_count < target_neg) {
                sampled_indices.push_back(i);
                neg_count++;
            }
        }

        if (sampled_indices.empty()) {
            continue;
        }

        // Create index tensor on CPU and move to device
        auto indices_tensor_cpu = Tensor({static_cast<int64_t>(sampled_indices.size())}, DType::Int64, Device::cpu());
        auto* indices_data = indices_tensor_cpu.data<int64_t>();
        for (size_t i = 0; i < sampled_indices.size(); ++i) {
            indices_data[i] = sampled_indices[i];
        }
        auto indices_tensor = indices_tensor_cpu.to(device);

        // Select sampled objectness logits and labels
        auto sampled_logits = tenzor::index_select(obj_logits_b, 0, indices_tensor);
        auto sampled_labels_full = tenzor::index_select(labels, 0, indices_tensor);

        // Compute classification loss: cross-entropy
        nn::CrossEntropyLoss ce_loss;
        auto cls_loss_var = ce_loss.forward(
            Variable(sampled_logits, true),  // requires_grad=true for gradient tracking
            sampled_labels_full
        );
        cls_losses.push_back(cls_loss_var);

        // Regression loss: only for positive anchors
        if (pos_count > 0) {
            // Get positive anchor indices
            std::vector<int64_t> pos_indices;
            for (size_t i = 0; i < sampled_indices.size(); ++i) {
                if (labels_data[sampled_indices[i]] == 1) {
                    pos_indices.push_back(sampled_indices[i]);
                }
            }

            auto pos_indices_tensor_cpu = Tensor({static_cast<int64_t>(pos_indices.size())}, DType::Int64, Device::cpu());
            auto* pos_indices_data = pos_indices_tensor_cpu.data<int64_t>();
            for (size_t i = 0; i < pos_indices.size(); ++i) {
                pos_indices_data[i] = pos_indices[i];
            }
            auto pos_indices_tensor = pos_indices_tensor_cpu.to(device);

            // Get predicted deltas for positive anchors
            auto pos_bbox_pred = tenzor::index_select(bbox_reg_b, 0, pos_indices_tensor);

            // Get matched GT boxes for positive anchors
            auto pos_matched_boxes = tenzor::index_select(matched_boxes, 0, pos_indices_tensor);
            auto pos_anchors = tenzor::index_select(anchors, 0, pos_indices_tensor);

            // Encode GT boxes as regression targets
            auto bbox_targets = encode_boxes(pos_matched_boxes, pos_anchors);

            // Compute Smooth L1 loss using nn::SmoothL1Loss for gradient tracking
            nn::SmoothL1Loss smooth_l1(nn::Reduction::Mean, 1.0);
            auto bbox_loss_var = smooth_l1.forward(
                Variable(pos_bbox_pred, true),
                Variable(bbox_targets, false)
            );
            bbox_losses.push_back(bbox_loss_var);
        }
    }

    // Average losses across batch - sum Variables and divide by count for gradient tracking
    Variable cls_loss_final;
    Variable bbox_loss_final;

    if (!cls_losses.empty()) {
        cls_loss_final = cls_losses[0];
        for (size_t i = 1; i < cls_losses.size(); ++i) {
            cls_loss_final = cls_loss_final + cls_losses[i];
        }
        cls_loss_final = cls_loss_final / static_cast<float>(cls_losses.size());
    } else {
        // Even with no samples, return differentiable zero for gradient flow
        cls_loss_final = Variable(Tensor({}, objectness_logits.dtype(), device).fill_(0.0), true);
    }

    if (!bbox_losses.empty()) {
        bbox_loss_final = bbox_losses[0];
        for (size_t i = 1; i < bbox_losses.size(); ++i) {
            bbox_loss_final = bbox_loss_final + bbox_losses[i];
        }
        bbox_loss_final = bbox_loss_final / static_cast<float>(bbox_losses.size());
    } else {
        // Even with no positive samples, return differentiable zero for gradient flow
        bbox_loss_final = Variable(Tensor({}, objectness_logits.dtype(), device).fill_(0.0), true);
    }

    return std::make_tuple(cls_loss_final, bbox_loss_final);
}

auto compute_roi_head_loss(
    const Tensor& class_logits,
    const Tensor& box_regression,
    const std::vector<Tensor>& targets
) -> std::tuple<Variable, Variable> {
    // class_logits: (num_rois, num_classes+1)
    // box_regression: (num_rois, (num_classes+1)*4)
    // targets: vector containing {labels, target_boxes} for sampled ROIs

    auto num_rois = class_logits.shape()[0];
    auto num_classes = class_logits.shape()[1];
    auto device = class_logits.device();

    if (num_rois == 0 || targets.empty()) {
        // Return differentiable zero for gradient flow
        auto zero_loss = Variable(Tensor({}, class_logits.dtype(), device).fill_(0.0), true);
        return std::make_tuple(zero_loss, zero_loss);
    }

    // For simplicity, assume targets[0] is labels, targets[1] is target boxes
    // In a full implementation, this would come from ROI sampling
    auto roi_labels = targets[0];  // (num_rois,)
    auto roi_target_boxes = targets.size() > 1 ? targets[1] : Tensor({num_rois, 4}, box_regression.dtype(), device);

    // Classification loss: multi-class cross-entropy
    nn::CrossEntropyLoss ce_loss;
    auto cls_loss = ce_loss.forward(
        Variable(class_logits, true),  // requires_grad=true for gradient tracking
        roi_labels
    );

    // Box regression loss: only for non-background classes
    // Move labels to CPU for data access
    auto roi_labels_cpu = roi_labels.to(Device::cpu());
    auto* labels_data = roi_labels_cpu.data<int64_t>();
    std::vector<int64_t> fg_indices;

    for (int64_t i = 0; i < num_rois; ++i) {
        if (labels_data[i] > 0) {  // Foreground (non-background)
            fg_indices.push_back(i);
        }
    }

    Variable bbox_loss;
    if (!fg_indices.empty()) {
        // Create index tensor on CPU and move to device
        auto fg_idx_tensor_cpu = Tensor({static_cast<int64_t>(fg_indices.size())}, DType::Int64, Device::cpu());
        auto* fg_idx_data = fg_idx_tensor_cpu.data<int64_t>();
        for (size_t i = 0; i < fg_indices.size(); ++i) {
            fg_idx_data[i] = fg_indices[i];
        }
        auto fg_idx_tensor = fg_idx_tensor_cpu.to(device);

        // Get foreground box predictions and targets
        auto fg_box_regression = tenzor::index_select(box_regression, 0, fg_idx_tensor);
        auto fg_target_boxes = tenzor::index_select(roi_target_boxes, 0, fg_idx_tensor);
        auto fg_labels = tenzor::index_select(roi_labels, 0, fg_idx_tensor);
        auto fg_labels_cpu = fg_labels.to(Device::cpu());

        // Extract box deltas for the predicted class
        auto num_fg = fg_box_regression.shape()[0];
        std::vector<Tensor> selected_deltas;

        for (int64_t i = 0; i < num_fg; ++i) {
            auto fg_row = tenzor::select(fg_box_regression, 0, i);
            int64_t class_idx = fg_labels_cpu.data<int64_t>()[i];

            // Select 4 values for this class
            auto start_idx = class_idx * 4;
            auto delta_slice = fg_row.slice(0, start_idx, start_idx + 4);
            selected_deltas.push_back(delta_slice);
        }

        auto fg_selected_deltas = ops::stack(selected_deltas, 0);  // (num_fg, 4)

        // Compute Smooth L1 loss using nn::SmoothL1Loss for gradient tracking
        nn::SmoothL1Loss smooth_l1(nn::Reduction::Mean, 1.0);
        bbox_loss = smooth_l1.forward(
            Variable(fg_selected_deltas, true),
            Variable(fg_target_boxes, false)
        );
    } else {
        // Return differentiable zero for gradient flow
        bbox_loss = Variable(Tensor({}, class_logits.dtype(), device).fill_(0.0), true);
    }

    return std::make_tuple(cls_loss, bbox_loss);
}

auto compute_mask_loss(
    const Tensor& mask_logits,
    const std::vector<Tensor>& targets
) -> Variable {
    // This is a wrapper around the existing nn::detection::mask_loss
    // mask_logits: (num_rois, num_classes, H, W)
    // targets: vector containing {mask_targets, class_labels}

    auto num_rois = mask_logits.shape()[0];
    auto device = mask_logits.device();

    if (num_rois == 0 || targets.size() < 2) {
        // Return differentiable zero for gradient flow
        return Variable(Tensor({}, mask_logits.dtype(), device).fill_(0.0), true);
    }

    auto mask_targets = targets[0];  // (num_rois, H, W)
    auto class_labels = targets[1];  // (num_rois,)

    // Use the existing mask_loss function from mask_head
    auto loss = nn::detection::mask_loss(
        Variable(mask_logits, true),  // requires_grad=true for gradient tracking
        mask_targets,
        class_labels
    );

    return loss;
}

// ============================================================================
// MaskRCNN Implementation
// ============================================================================

auto MaskRCNN::extract_features(const Variable& images) -> Variable {
    // Extract features from backbone
    // For ResNet-FPN, this returns multi-scale features
    // We use the finest scale (P2) for now

    // Cast backbone to ResNet to access forward_features
    auto resnet = std::dynamic_pointer_cast<ResNet>(backbone_);
    if (!resnet) {
        throw std::runtime_error("Mask R-CNN requires ResNet backbone");
    }

    // Use forward_features to get feature maps before global pooling
    // This returns features from layer4 (C5) at 1/32 resolution with 2048 channels
    auto backbone_features = resnet->forward_features(images);  // Shape: (N, 2048, H, W)

    // Project from 2048 channels to 256 channels using 1x1 conv
    // This simulates FPN's top-down pathway
    auto projected_features = feature_proj_->forward(backbone_features);  // Shape: (N, 256, H, W)

    return projected_features;
}

auto MaskRCNN::generate_proposals(const Variable& features) -> Tensor {
    // Generate proposals from RPN output
    // This is a simplified version - production would implement full proposal generation

    auto H = features.tensor().shape()[2];
    auto W = features.tensor().shape()[3];
    auto batch_size = features.tensor().shape()[0];

    // Generate anchor boxes on the same device as features
    auto anchors = anchor_generator_->generate(H, W, 16, features.tensor().device());

    // For now, return top K anchors as proposals
    // Format: (batch_idx, x1, y1, x2, y2)
    auto num_proposals = std::min(
        static_cast<int64_t>(anchors.shape()[0]),
        is_training() ? rpn_post_nms_top_n_train_ : rpn_post_nms_top_n_test_
    );

    auto proposals = Tensor(
        std::vector<int64_t>{num_proposals, 5},
        features.tensor().dtype(),  // Use input dtype for multi-dtype support
        features.tensor().device()
    );

    // Fill with dummy proposals (in production, would decode RPN predictions)
    proposals.fill_(0.0);

    return proposals;
}

auto MaskRCNN::select_training_samples(const Tensor& proposals,
                                        const Tensor& gt_boxes,
                                        const Tensor& gt_labels) -> Tensor {
    // Select positive and negative training samples
    // This is a simplified version - production would implement proper sampling

    // For now, just return the proposals
    return proposals;
}

void MaskRCNN::load_pretrained(const std::string& path, bool strict) {
    // Load pretrained weights
    // This would integrate with Tenzor's serialization system
    throw std::runtime_error("Pretrained weight loading not yet implemented");
}

// ============================================================================
// Factory Functions
// ============================================================================

auto mask_rcnn_resnet50_fpn(int64_t num_classes, bool pretrained)
    -> std::shared_ptr<MaskRCNN> {

    // Create ResNet-50 backbone
    auto backbone = resnet50(1000, pretrained);

    // Create Mask R-CNN model
    auto model = std::make_shared<MaskRCNN>(
        backbone,
        num_classes,
        800,   // min_size
        1333,  // max_size
        2000,  // rpn_pre_nms_top_n_train
        1000,  // rpn_pre_nms_top_n_test
        2000,  // rpn_post_nms_top_n_train
        1000,  // rpn_post_nms_top_n_test
        0.7,   // rpn_nms_thresh
        0.05,  // box_score_thresh
        0.5,   // box_nms_thresh
        100    // box_detections_per_img
    );

    if (pretrained) {
        // Load pretrained COCO weights
        // model->load_pretrained("mask_rcnn_resnet50_fpn_coco.pth");
    }

    return model;
}

auto mask_rcnn_resnet101_fpn(int64_t num_classes, bool pretrained)
    -> std::shared_ptr<MaskRCNN> {

    // Create ResNet-101 backbone
    auto backbone = resnet101(1000, pretrained);

    // Create Mask R-CNN model
    auto model = std::make_shared<MaskRCNN>(
        backbone,
        num_classes,
        800,   // min_size
        1333,  // max_size
        2000,  // rpn_pre_nms_top_n_train
        1000,  // rpn_pre_nms_top_n_test
        2000,  // rpn_post_nms_top_n_train
        1000,  // rpn_post_nms_top_n_test
        0.7,   // rpn_nms_thresh
        0.05,  // box_score_thresh
        0.5,   // box_nms_thresh
        100    // box_detections_per_img
    );

    if (pretrained) {
        // Load pretrained COCO weights
        // model->load_pretrained("mask_rcnn_resnet101_fpn_coco.pth");
    }

    return model;
}

} // namespace models
} // namespace tenzor
