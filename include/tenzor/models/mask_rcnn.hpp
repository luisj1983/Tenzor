/**
 * @file mask_rcnn.hpp
 * @brief Mask R-CNN for instance segmentation
 *
 * Implements Mask R-CNN, which extends Faster R-CNN with a mask prediction
 * branch for pixel-wise instance segmentation.
 *
 * Reference: "Mask R-CNN" (He et al., 2017)
 * https://arxiv.org/abs/1703.06870
 */

#pragma once

#include <array>
#include <memory>
#include <vector>
#include <tuple>
#include <string>
#include <cstdint>

#include "tenzor/nn/module.hpp"
#include "tenzor/nn/detection/roi_ops.hpp"
#include "tenzor/nn/detection/mask_head.hpp"
#include "tenzor/nn/detection/anchors.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/core/tensor.hpp"
#include "resnet.hpp"

namespace tenzor {
namespace models {

/**
 * @brief Region Proposal Network (RPN) for Mask R-CNN.
 *
 * Proposes object bounding boxes from feature maps using anchor boxes.
 * The RPN is shared between Faster R-CNN and Mask R-CNN.
 *
 * Architecture:
 * ```
 * Feature Map (C, H, W)
 *     ↓
 * Conv 3×3, 512 → ReLU
 *     ↓
 * ├─→ Conv 1×1, num_anchors×2 (objectness)
 * └─→ Conv 1×1, num_anchors×4 (box deltas)
 * ```
 *
 * Outputs:
 * - Objectness scores: (N, num_anchors*H*W, 2)
 * - Box deltas: (N, num_anchors*H*W, 4)
 */
class RPN : public nn::Module {
public:
    /**
     * @brief Construct RPN.
     *
     * @param in_channels Number of input channels from backbone
     * @param num_anchors Number of anchors per spatial location
     */
    RPN(int64_t in_channels, int64_t num_anchors);

    /**
     * @brief Forward pass through RPN returning both outputs.
     *
     * @param features Feature map from backbone (N, C, H, W)
     * @return Tuple of (objectness_logits, bbox_deltas)
     *         - objectness_logits: (N, num_anchors*H*W, 2)
     *         - bbox_deltas: (N, num_anchors*H*W, 4)
     */
    auto forward_multi(const Variable& features)
        -> std::tuple<Variable, Variable>;

    // Module interface implementation - returns objectness logits only
    auto forward_impl(const Variable& input) -> Variable override {
        auto [obj_logits, bbox_deltas] = forward_multi(input);
        return obj_logits;
    }

private:
    std::shared_ptr<nn::Conv2d> conv_;
    std::shared_ptr<nn::Conv2d> cls_logits_;
    std::shared_ptr<nn::Conv2d> bbox_pred_;
};

/**
 * @brief ROI Head for box classification and regression.
 *
 * Takes ROI-aligned features and predicts class labels and
 * refined bounding boxes for each proposal.
 *
 * Architecture:
 * ```
 * ROI Features (7×7×C)
 *     ↓
 * Flatten
 *     ↓
 * FC 1024 → ReLU
 *     ↓
 * FC 1024 → ReLU
 *     ↓
 * ├─→ FC num_classes (classification)
 * └─→ FC num_classes×4 (box regression)
 * ```
 */
class ROIHead : public nn::Module {
public:
    /**
     * @brief Construct ROI head.
     *
     * @param in_channels Number of input channels from ROI pooling
     * @param num_classes Number of classes (including background)
     * @param roi_size Size of ROI features (e.g., 7 for 7×7)
     */
    ROIHead(int64_t in_channels, int64_t num_classes, int64_t roi_size = 7);

    /**
     * @brief Forward pass through ROI head returning both outputs.
     *
     * @param roi_features ROI-pooled features (num_rois, C, roi_size, roi_size)
     * @return Tuple of (class_logits, bbox_deltas)
     *         - class_logits: (num_rois, num_classes)
     *         - bbox_deltas: (num_rois, num_classes*4)
     */
    auto forward_multi(const Variable& roi_features)
        -> std::tuple<Variable, Variable>;

    // Module interface implementation - returns class logits only
    auto forward_impl(const Variable& input) -> Variable override {
        auto [cls_logits, bbox_deltas] = forward_multi(input);
        return cls_logits;
    }

private:
    std::shared_ptr<nn::Linear> fc1_;
    std::shared_ptr<nn::Linear> fc2_;
    std::shared_ptr<nn::Linear> cls_score_;
    std::shared_ptr<nn::Linear> bbox_pred_;
};

/**
 * @brief Mask R-CNN model for instance segmentation.
 *
 * Mask R-CNN extends Faster R-CNN by adding a mask prediction branch
 * that runs in parallel with the box classification and regression heads.
 *
 * Architecture:
 * ```
 * Input Image
 *     ↓
 * Backbone (ResNet-50-FPN or ResNet-101-FPN)
 *     ↓
 * Feature Pyramid Network (FPN)
 *     ↓
 * Region Proposal Network (RPN)
 *     ↓
 * ROI Align (7×7 for boxes, 14×14 for masks)
 *     ↓
 * ├─→ Box Head → Classification + Box Regression
 * └─→ Mask Head → Instance Masks (28×28 per class)
 * ```
 *
 * Outputs:
 * - Training: Returns losses (rpn_cls, rpn_bbox, roi_cls, roi_bbox, mask)
 * - Inference: Returns (boxes, labels, scores, masks)
 *
 * @code
 * // Create Mask R-CNN with ResNet-50-FPN backbone
 * auto model = mask_rcnn_resnet50_fpn(80, false);  // 80 COCO classes
 *
 * // Training mode
 * model->train();
 * auto image = randn({1, 3, 800, 1200});
 * auto targets = get_ground_truth();
 * auto losses = model->forward(image, targets);
 *
 * // Inference mode
 * model->eval();
 * auto detections = model->forward(image);
 * auto [boxes, labels, scores, masks] = detections;
 * @endcode
 */
/**
 * @brief Structured output bundle for Mask R-CNN inference.
 *
 * Audit G7: returned by `MaskRCNN::detect(images)`. The generic `forward()`
 * entry point on Module returns a single Variable, so it packs boxes + scores
 * + labels into a flat (N, 6) tensor (YOLOv5 convention) and drops the masks.
 * Callers who need masks should use `detect()` instead.
 */
struct Detections {
    Tensor boxes;   ///< (N, 4) bounding boxes in (x1, y1, x2, y2) format
    Tensor labels;  ///< (N,) class indices (Int64)
    Tensor scores;  ///< (N,) confidence scores
    Tensor masks;   ///< (N, H, W) binary mask predictions
};

class MaskRCNN : public nn::Module {
public:
    /**
     * @brief Construct Mask R-CNN model.
     *
     * @param backbone Pretrained backbone network (e.g., ResNet-50)
     * @param num_classes Number of classes (excluding background)
     * @param min_size Minimum image size (default: 800)
     * @param max_size Maximum image size (default: 1333)
     * @param rpn_pre_nms_top_n_train RPN proposals before NMS during training (default: 2000)
     * @param rpn_pre_nms_top_n_test RPN proposals before NMS during testing (default: 1000)
     * @param rpn_post_nms_top_n_train RPN proposals after NMS during training (default: 2000)
     * @param rpn_post_nms_top_n_test RPN proposals after NMS during testing (default: 1000)
     * @param rpn_nms_thresh RPN NMS threshold (default: 0.7)
     * @param box_score_thresh Minimum score for detections (default: 0.05)
     * @param box_nms_thresh NMS threshold for final detections (default: 0.5)
     * @param box_detections_per_img Maximum detections per image (default: 100)
     */
    MaskRCNN(std::shared_ptr<nn::Module> backbone,
             int64_t num_classes,
             int64_t min_size = 800,
             int64_t max_size = 1333,
             int64_t rpn_pre_nms_top_n_train = 2000,
             int64_t rpn_pre_nms_top_n_test = 1000,
             int64_t rpn_post_nms_top_n_train = 2000,
             int64_t rpn_post_nms_top_n_test = 1000,
             double rpn_nms_thresh = 0.7,
             double box_score_thresh = 0.05,
             double box_nms_thresh = 0.5,
             int64_t box_detections_per_img = 100);

    /**
     * @brief Forward pass through Mask R-CNN.
     *
     * Training mode (with targets):
     * @param images Input images (N, 3, H, W)
     * @param targets Ground truth annotations (boxes, labels, masks)
     * @return Dictionary of losses
     *
     * Inference mode (without targets):
     * @param images Input images (N, 3, H, W)
     * @return Tuple of (boxes, labels, scores, masks)
     */
    auto forward_impl(const Variable& images) -> Variable override;

    /**
     * @brief Forward pass with ground truth targets (training).
     *
     * @param images Input images (N, 3, H, W)
     * @param gt_boxes Ground truth boxes (N, max_objects, 4)
     * @param gt_labels Ground truth labels (N, max_objects)
     * @param gt_masks Ground truth masks (N, max_objects, H, W)
     * @return Tuple of losses (rpn_cls, rpn_bbox, roi_cls, roi_bbox, mask_loss)
     */
    auto forward_train(const Variable& images,
                       const Tensor& gt_boxes,
                       const Tensor& gt_labels,
                       const Tensor& gt_masks)
        -> std::tuple<Variable, Variable, Variable, Variable, Variable>;

    /**
     * @brief Forward pass for inference (testing).
     *
     * @param images Input images (N, 3, H, W)
     * @return Tuple of (boxes, labels, scores, masks)
     *         - boxes: (num_detections, 4) as (x1, y1, x2, y2)
     *         - labels: (num_detections,) class indices
     *         - scores: (num_detections,) confidence scores
     *         - masks: (num_detections, H, W) binary masks
     */
    auto forward_test(const Variable& images)
        -> std::tuple<Tensor, Tensor, Tensor, Tensor>;

    /**
     * @brief Inference convenience returning a structured `Detections` bundle.
     *
     * Audit G7: callers who use the generic `Module::forward(images)` entry
     * point get a packed (N, 6) `[x1, y1, x2, y2, score, label]` Variable
     * (YOLOv5 convention) with masks dropped, because Module returns a single
     * Variable. This method gives access to all four outputs (boxes, labels,
     * scores, masks) in a named-field struct.
     *
     * @param images Input image batch (N, 3, H, W)
     * @return Detections{boxes, labels, scores, masks}
     */
    auto detect(const Variable& images) -> Detections;

    /**
     * @brief Load pretrained weights.
     *
     * Supports loading weights from:
     * - Detectron2 format (pickled PyTorch state dict)
     * - Tenzor native format
     *
     * @param path Path to pretrained weights file
     * @param strict If true, requires exact key match (default: true)
     * @throws std::runtime_error if file doesn't exist or format is invalid
     */
    auto load_pretrained(const std::string& path, bool strict = true) -> void;

    /**
     * @brief Get number of classes.
     */
    auto num_classes() const -> int64_t { return num_classes_; }

    /**
     * @brief Run the RPN and produce per-batch proposals.
     *
     * Exposed (rather than private) so regression tests can verify that the
     * decoded boxes are meaningful — see G4 in the audit-remediation plan,
     * which replaced a dummy `fill_(0.0)` proposal slab with a real
     * decode → clip → small-filter → top-K → NMS → top-K chain.
     *
     * Output shape: (num_proposals, 5) = (batch_idx, x1, y1, x2, y2).
     */
    auto generate_proposals(const Variable& features) -> Tensor;

    /**
     * @brief Sample training-time positive and negative ROIs from proposals.
     *
     * Exposed for testing (G5 audit fix). Implements torchvision-style
     * balanced sampling: positives are proposals whose max IoU with any GT
     * box in the same image is ≥ 0.5; negatives are the rest. Per image,
     * picks up to `num_samples * positive_fraction` positives, then fills
     * the remainder with negatives.
     *
     * Output shape: (sampled, 5) matching the proposal layout.
     */
    auto select_training_samples(const Tensor& proposals,
                                  const Tensor& gt_boxes,
                                  const Tensor& gt_labels) -> Tensor;

private:
    // Model components
    std::shared_ptr<nn::Module> backbone_;              ///< Feature extractor (e.g., ResNet-50)
    // Audit G6: real FPN encoder. Builds P2-P5 from C2-C5 via lateral (1×1)
    // and smoothing (3×3) convs with top-down (nearest) upsample additions.
    // extract_features returns P4 (stride 16, 256 channels) so the rest of the
    // pipeline (RPN, ROI Align, anchor generator) — all already configured for
    // stride 16 — sees a feature map at the right resolution AND benefits from
    // top-down high-level semantics injected from P5. Replaces the previous
    // single 2048→256 1×1 conv that produced a flat C5 projection.
    std::array<std::shared_ptr<nn::Conv2d>, 4> fpn_lateral_;  ///< 1×1 convs Cᵢ → 256 (i=2..5)
    std::array<std::shared_ptr<nn::Conv2d>, 4> fpn_smooth_;   ///< 3×3 anti-alias convs on each Pᵢ
    std::shared_ptr<RPN> rpn_;                          ///< Region Proposal Network
    std::shared_ptr<nn::detection::ROIAlign> roi_align_box_;   ///< ROI Align for boxes (7×7)
    std::shared_ptr<nn::detection::ROIAlign> roi_align_mask_;  ///< ROI Align for masks (14×14)
    std::shared_ptr<ROIHead> roi_head_;                 ///< Box classification and regression
    std::shared_ptr<nn::detection::MaskHead> mask_head_;       ///< Mask prediction
    std::shared_ptr<nn::detection::AnchorGenerator> anchor_generator_;  ///< Anchor generation

    // Model configuration
    int64_t num_classes_;
    int64_t min_size_;
    int64_t max_size_;
    int64_t rpn_pre_nms_top_n_train_;
    int64_t rpn_pre_nms_top_n_test_;
    int64_t rpn_post_nms_top_n_train_;
    int64_t rpn_post_nms_top_n_test_;
    double rpn_nms_thresh_;
    double box_score_thresh_;
    double box_nms_thresh_;
    int64_t box_detections_per_img_;

    // Helper functions
    auto extract_features(const Variable& images) -> Variable;
};

// ============================================================================
// Factory Functions for Mask R-CNN Variants
// ============================================================================

/**
 * @brief Create Mask R-CNN with ResNet-50-FPN backbone.
 *
 * This is the standard Mask R-CNN configuration used in the original paper.
 *
 * Architecture:
 * - Backbone: ResNet-50 with Feature Pyramid Network (FPN)
 * - RPN: Standard region proposal network
 * - ROI Head: Two FC layers (1024 hidden units)
 * - Mask Head: 4 conv layers + deconv + 1×1 conv
 *
 * Performance (COCO val2017):
 * - Box AP: ~37.9%
 * - Mask AP: ~34.6%
 *
 * Parameters: ~44M
 *
 * @param num_classes Number of classes (default: 80 for COCO)
 * @param pretrained Load pretrained COCO weights (default: false)
 * @return Shared pointer to Mask R-CNN model
 *
 * @code
 * auto model = models::mask_rcnn_resnet50_fpn(80, true);
 * Variable output = model->forward(input);
 * @endcode
 */
auto mask_rcnn_resnet50_fpn(int64_t num_classes = 80, bool pretrained = false)
    -> std::shared_ptr<MaskRCNN>;

/**
 * @brief Create Mask R-CNN with ResNet-101-FPN backbone.
 *
 * Larger variant with ResNet-101 backbone for improved accuracy.
 *
 * Architecture:
 * - Backbone: ResNet-101 with Feature Pyramid Network (FPN)
 * - Same RPN, ROI Head, and Mask Head as ResNet-50 variant
 *
 * Performance (COCO val2017):
 * - Box AP: ~40.0%
 * - Mask AP: ~36.1%
 *
 * Parameters: ~63M
 *
 * @param num_classes Number of classes (default: 80 for COCO)
 * @param pretrained Load pretrained COCO weights (default: false)
 * @return Shared pointer to Mask R-CNN model
 */
auto mask_rcnn_resnet101_fpn(int64_t num_classes = 80, bool pretrained = false)
    -> std::shared_ptr<MaskRCNN>;

} // namespace models
} // namespace tenzor
