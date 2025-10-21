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
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/loss/losses.hpp"
#include <stdexcept>

namespace tenzor {
namespace models {

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

auto RPN::forward_impl(const Variable& features)
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

auto ROIHead::forward_impl(const Variable& roi_features)
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
    auto num_anchors = 3;  // 3 scales per location
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

auto MaskRCNN::forward(const Variable& images) -> Variable {
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
    auto [rpn_cls_logits, rpn_bbox_deltas] = rpn_->forward_impl(features);

    // Generate anchor boxes
    auto H = features.tensor().shape()[2];
    auto W = features.tensor().shape()[3];
    auto anchors = anchor_generator_->generate(H, W, 16);  // stride=16

    // TODO: Compute RPN losses
    // For now, create dummy losses
    auto rpn_cls_loss = Variable(Tensor({}, DType::Float32, images.tensor().device()).fill_(0.0), false);
    auto rpn_bbox_loss = Variable(Tensor({}, DType::Float32, images.tensor().device()).fill_(0.0), false);

    // 3. Generate proposals from RPN output
    auto proposals = generate_proposals(features);

    // 4. Select training samples (positive and negative ROIs)
    auto sampled_rois = select_training_samples(proposals, gt_boxes, gt_labels);

    // 5. ROI Align for box head (7×7)
    auto roi_features_box = roi_align_box_->forward(features, sampled_rois);

    // 6. Box classification and regression
    auto [cls_logits, bbox_deltas] = roi_head_->forward_impl(roi_features_box);

    // TODO: Compute box losses
    auto roi_cls_loss = Variable(Tensor({}, DType::Float32, images.tensor().device()).fill_(0.0), false);
    auto roi_bbox_loss = Variable(Tensor({}, DType::Float32, images.tensor().device()).fill_(0.0), false);

    // 7. Match sampled ROIs to ground truth to get their labels
    // For now, use a simple approach: assign label 0 (background) to all
    // TODO: Implement proper IoU-based matching
    auto num_sampled = sampled_rois.shape()[0];
    auto sampled_labels = Tensor({num_sampled}, DType::Int64, sampled_rois.device());

    // Initialize all to zero (background)
    auto* sampled_labels_data = sampled_labels.data<int64_t>();
    for (int64_t i = 0; i < num_sampled; ++i) {
        sampled_labels_data[i] = 0;
    }

    // For testing purposes, assign first few ROIs to GT labels
    // This is a temporary workaround - proper implementation would match by IoU
    auto num_gt = gt_labels.shape()[1];
    auto gt_labels_flat = reshape(gt_labels, {num_gt});
    auto* gt_labels_data = gt_labels_flat.data<int64_t>();
    for (int64_t i = 0; i < std::min(num_sampled, num_gt); ++i) {
        sampled_labels_data[i] = gt_labels_data[i];
    }

    // 8. ROI Align for mask head (14×14, only for positive samples)
    auto roi_features_mask = roi_align_mask_->forward(features, sampled_rois);

    // 9. Mask prediction
    auto mask_logits = mask_head_->forward(roi_features_mask);

    // 10. Create sampled_masks with correct shape to match mask_logits output
    // mask_logits has shape [num_sampled, num_classes, mask_H, mask_W]
    // where mask_H and mask_W are typically 28x28 for mask head output
    auto mask_output_H = mask_logits.tensor().shape()[2];
    auto mask_output_W = mask_logits.tensor().shape()[3];

    // Create masks at the correct resolution to match mask head output
    auto sampled_masks = Tensor({num_sampled, mask_output_H, mask_output_W},
                                DType::Float32, sampled_rois.device());

    // Initialize all masks to zero (background)
    // For proper training, these should be downsampled from GT masks
    // TODO: Implement proper mask resampling from GT masks
    auto* sampled_masks_data = static_cast<float*>(sampled_masks.data_ptr());
    std::fill(sampled_masks_data, sampled_masks_data + sampled_masks.numel(), 0.0f);

    // 11. Compute mask loss using matched masks and labels
    auto mask_loss_val = nn::detection::mask_loss(
        mask_logits,
        sampled_masks,  // Masks at correct resolution (28x28)
        sampled_labels  // Use sampled labels
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
    auto [rpn_cls_logits, rpn_bbox_deltas] = rpn_->forward_impl(features);
    auto proposals = generate_proposals(features);

    // 3. ROI Align for boxes
    auto roi_features_box = roi_align_box_->forward(features, proposals);

    // 4. Box classification and regression
    auto [cls_logits, bbox_deltas] = roi_head_->forward_impl(roi_features_box);

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

    // Generate anchor boxes
    auto anchors = anchor_generator_->generate(H, W, 16);

    // For now, return top K anchors as proposals
    // Format: (batch_idx, x1, y1, x2, y2)
    auto num_proposals = std::min(
        static_cast<int64_t>(anchors.shape()[0]),
        is_training() ? rpn_post_nms_top_n_train_ : rpn_post_nms_top_n_test_
    );

    auto proposals = Tensor(
        std::vector<int64_t>{num_proposals, 5},
        DType::Float32,
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
