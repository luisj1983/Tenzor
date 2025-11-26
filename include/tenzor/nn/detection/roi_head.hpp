/**
 * @file roi_head.hpp
 * @brief ROI Head for object detection (classification and box regression)
 *
 * Implements the second stage of two-stage detectors like Faster R-CNN
 * and Mask R-CNN. Takes region proposals and performs:
 * 1. ROI feature extraction (ROI Align/Pooling)
 * 2. Box classification (C classes + background)
 * 3. Box regression refinement
 */

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/detection/roi_ops.hpp"
#include "tenzor/autograd/variable.hpp"

namespace tenzor {
namespace nn {
namespace detection {

/**
 * @brief ROI Box Head for classification and regression.
 *
 * Takes ROI-aligned features and predicts:
 * - Class logits for each ROI (num_classes + 1 for background)
 * - Bounding box refinement deltas
 *
 * Architecture:
 * ```
 * ROI features (7x7xC) -> Flatten -> FC -> ReLU -> FC -> ReLU -> ┬─> FC -> class logits
 *                                                                  └─> FC -> box deltas
 * ```
 */
class RoIBoxHead : public Module {
public:
    /**
     * @brief Construct ROI box head.
     *
     * @param in_channels Number of input channels from ROI Align
     * @param roi_size Size of ROI features (e.g., 7 for 7x7)
     * @param num_classes Number of object classes (not including background)
     * @param representation_size Size of intermediate FC layers (default: 1024)
     */
    RoIBoxHead(int64_t in_channels,
               int64_t roi_size,
               int64_t num_classes,
               int64_t representation_size = 1024);

    /**
     * @brief Forward pass to get class logits and box deltas.
     *
     * @param roi_features ROI-aligned features (num_rois, C, H, W)
     * @return Tuple of (class_logits, box_deltas)
     *         - class_logits: (num_rois, num_classes+1) raw scores
     *         - box_deltas: (num_rois, num_classes*4) class-specific deltas
     */
    auto forward_features(const Variable& roi_features)
        -> std::pair<Variable, Variable>;

    /**
     * @brief Module forward (required by base class).
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t num_classes_;

    // Feature extraction layers
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;

    // Prediction heads
    std::shared_ptr<Linear> cls_score_;
    std::shared_ptr<Linear> bbox_pred_;
};

/**
 * @brief Complete ROI Head combining ROI Align and box head.
 *
 * Performs the second stage of two-stage detection:
 * 1. Extract features from proposals using ROI Align
 * 2. Classify proposals into object classes
 * 3. Refine bounding boxes
 *
 * Training:
 * - Matches proposals to ground truth boxes
 * - Samples positive and negative ROIs
 * - Computes classification and regression losses
 *
 * Inference:
 * - Classifies all proposals
 * - Applies class-specific box refinement
 * - Filters by score threshold and NMS
 */
class RoIHead : public Module {
public:
    /**
     * @brief Construct ROI head.
     *
     * @param in_channels Number of channels in feature maps
     * @param num_classes Number of object classes (not including background)
     * @param roi_output_size Output size of ROI Align (e.g., 7 for 7x7)
     * @param spatial_scale Feature map scale (e.g., 1/16)
     * @param sampling_ratio ROI Align sampling ratio (default: 2)
     * @param fg_iou_thresh IoU threshold for positive ROIs (default: 0.5)
     * @param bg_iou_thresh IoU threshold for negative ROIs (default: 0.5)
     * @param batch_size_per_image Number of ROIs per image (default: 512)
     * @param positive_fraction Fraction of positive ROIs (default: 0.25)
     * @param score_thresh Score threshold for detections (default: 0.05)
     * @param nms_thresh NMS IoU threshold (default: 0.5)
     * @param detections_per_img Maximum detections per image (default: 100)
     */
    RoIHead(int64_t in_channels,
            int64_t num_classes,
            int64_t roi_output_size = 7,
            double spatial_scale = 1.0 / 16.0,
            int64_t sampling_ratio = 2,
            double fg_iou_thresh = 0.5,
            double bg_iou_thresh = 0.5,
            int64_t batch_size_per_image = 512,
            double positive_fraction = 0.25,
            double score_thresh = 0.05,
            double nms_thresh = 0.5,
            int64_t detections_per_img = 100);

    /**
     * @brief Forward pass to get detections.
     *
     * @param features Feature maps from backbone (N, C, H, W)
     * @param proposals List of proposal boxes, one per image (K_i, 4)
     * @param image_shapes List of (height, width) for each image
     * @param gt_boxes Optional ground truth boxes for training
     * @param gt_labels Optional ground truth class labels for training
     * @return Detections: list of dicts with "boxes", "labels", "scores"
     *         If training: also computes and stores losses
     */
    auto forward_detections(
        const Variable& features,
        const std::vector<Tensor>& proposals,
        const std::vector<std::pair<int64_t, int64_t>>& image_shapes,
        const std::vector<Tensor>* gt_boxes = nullptr,
        const std::vector<Tensor>* gt_labels = nullptr
    ) -> std::vector<std::unordered_map<std::string, Tensor>>;

    /**
     * @brief Module forward (required by base class).
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Get ROI head losses.
     *
     * @return Dictionary with "loss_classifier" and "loss_box_reg"
     */
    auto get_losses() const -> std::unordered_map<std::string, Variable>;

private:
    /**
     * @brief Match proposals to ground truth boxes.
     *
     * @param proposals Proposal boxes (num_proposals, 4)
     * @param gt_boxes Ground truth boxes (num_gt, 4)
     * @param gt_labels Ground truth class labels (num_gt,)
     * @return Tuple of (labels, matched_gt_boxes)
     */
    auto match_proposals_to_gt(const Tensor& proposals,
                                const Tensor& gt_boxes,
                                const Tensor& gt_labels)
        -> std::pair<Tensor, Tensor>;

    /**
     * @brief Sample positive and negative ROIs for training.
     *
     * @param labels ROI labels (0 for background, >0 for objects)
     * @return Indices of sampled ROIs
     */
    auto sample_rois(const Tensor& labels) -> Tensor;

    /**
     * @brief Post-process predictions to get final detections.
     *
     * @param class_logits Class predictions (num_rois, num_classes+1)
     * @param box_deltas Box regression deltas (num_rois, num_classes*4)
     * @param proposals Original proposals (num_rois, 4)
     * @param image_shape Image (height, width)
     * @return Dictionary with "boxes", "labels", "scores"
     */
    auto postprocess_detections(
        const Tensor& class_logits,
        const Tensor& box_deltas,
        const Tensor& proposals,
        const std::pair<int64_t, int64_t>& image_shape
    ) -> std::unordered_map<std::string, Tensor>;

    int64_t num_classes_;

    // ROI feature extractor
    std::shared_ptr<ROIAlign> roi_align_;

    // Box head for classification and regression
    std::shared_ptr<RoIBoxHead> box_head_;

    // Training parameters
    double fg_iou_thresh_;
    double bg_iou_thresh_;
    int64_t batch_size_per_image_;
    double positive_fraction_;

    // Inference parameters
    double score_thresh_;
    double nms_thresh_;
    int64_t detections_per_img_;

    // Cached losses (set during forward)
    mutable Variable loss_classifier_;
    mutable Variable loss_box_reg_;
};

} // namespace detection
} // namespace nn
} // namespace tenzor
