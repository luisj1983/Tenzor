/**
 * @file rpn.hpp
 * @brief Region Proposal Network for Faster R-CNN
 *
 * Implements the RPN (Region Proposal Network) that generates object proposals
 * for two-stage detection models like Faster R-CNN and Mask R-CNN.
 */

#pragma once

#include <memory>
#include <vector>
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/nn/detection/anchors.hpp"

namespace tenzor {
namespace nn {
namespace detection {

/**
 * @brief RPN Head - objectness and box regression for anchors.
 *
 * Predicts whether each anchor contains an object (objectness) and
 * bounding box refinement deltas.
 *
 * Architecture:
 * ```
 * features -> Conv(3x3) -> ReLU -> ┬─> Conv(1x1) -> objectness logits
 *                                  └─> Conv(1x1) -> box regression deltas
 * ```
 *
 * Note: RPNHead does not inherit from Module because it returns a pair
 * instead of a single Variable. It's used internally by RegionProposalNetwork.
 */
class RPNHead {
public:
    /**
     * @brief Construct RPN head.
     *
     * @param in_channels Number of input feature channels
     * @param num_anchors Number of anchors per spatial location
     */
    RPNHead(int64_t in_channels, int64_t num_anchors);

    /**
     * @brief Forward pass to generate objectness scores and box deltas.
     *
     * @param features Input feature map (N, C, H, W)
     * @return Tuple of (objectness_logits, box_regression)
     *         - objectness_logits: (N, num_anchors*H*W) raw scores
     *         - box_regression: (N, num_anchors*H*W, 4) box deltas
     */
    auto forward(const Variable& features)
        -> std::pair<Variable, Variable>;

    // Note: forward(Variable) override not needed - RPNHead is used internally
    // RegionProposalNetwork provides the Module interface

    // Public members so RegionProposalNetwork can register them
    std::shared_ptr<Conv2d> conv_;       // 3x3 conv for spatial context
    std::shared_ptr<Conv2d> cls_logits_; // Objectness classification
    std::shared_ptr<Conv2d> bbox_pred_;  // Box regression

private:
    int64_t num_anchors_;
};

/**
 * @brief Region Proposal Network.
 *
 * Complete RPN that generates region proposals from feature maps.
 * Combines anchor generation, RPN head predictions, and proposal filtering.
 *
 * Training:
 * - Generates anchors at each feature map location
 * - Predicts objectness and box deltas
 * - Assigns anchors to ground truth boxes
 * - Computes classification and regression losses
 *
 * Inference:
 * - Generates proposals from anchors and predictions
 * - Applies NMS to reduce redundancy
 * - Returns top-k scoring proposals
 */
class RegionProposalNetwork : public Module {
public:
    /**
     * @brief Construct Region Proposal Network.
     *
     * @param in_channels Number of input feature channels
     * @param anchor_generator Anchor box generator
     * @param fg_iou_thresh IoU threshold for positive anchors (default: 0.7)
     * @param bg_iou_thresh IoU threshold for negative anchors (default: 0.3)
     * @param batch_size_per_image Number of anchors per image for training (default: 256)
     * @param positive_fraction Fraction of positive anchors (default: 0.5)
     * @param pre_nms_top_n Number of proposals before NMS (default: 2000)
     * @param post_nms_top_n Number of proposals after NMS (default: 1000)
     * @param nms_thresh NMS IoU threshold (default: 0.7)
     * @param score_thresh Minimum score for proposals (default: 0.0)
     */
    RegionProposalNetwork(
        int64_t in_channels,
        std::shared_ptr<AnchorGenerator> anchor_generator,
        double fg_iou_thresh = 0.7,
        double bg_iou_thresh = 0.3,
        int64_t batch_size_per_image = 256,
        double positive_fraction = 0.5,
        int64_t pre_nms_top_n = 2000,
        int64_t post_nms_top_n = 1000,
        double nms_thresh = 0.7,
        double score_thresh = 0.0
    );

    /**
     * @brief Forward pass to generate proposals.
     *
     * Training mode:
     * - Returns proposals and losses for RPN training
     * - targets must contain ground truth boxes
     *
     * Inference mode:
     * - Returns filtered proposals (after NMS)
     * - targets can be empty
     *
     * @param features Feature maps from backbone (N, C, H, W)
     * @param image_shapes List of (height, width) for each image in batch
     * @param targets Optional ground truth boxes for training
     * @return Proposals: list of (K, 4) tensors, one per image
     *         If training: also returns dict with losses
     */
    auto forward_proposals(
        const Variable& features,
        const std::vector<std::pair<int64_t, int64_t>>& image_shapes,
        const std::vector<Tensor>* targets = nullptr
    ) -> std::vector<Tensor>;

    /**
     * @brief Module forward (required by base class).
     */
    auto forward(const Variable& input) -> Variable override;

    /**
     * @brief Get RPN losses (objectness + box regression).
     *
     * @return Dictionary with "loss_objectness" and "loss_rpn_box_reg"
     */
    auto get_losses() const -> std::unordered_map<std::string, Variable>;

private:
    /**
     * @brief Assign anchors to ground truth boxes.
     *
     * @param anchors All anchors (num_anchors, 4)
     * @param gt_boxes Ground truth boxes (num_gt, 4)
     * @return Tuple of (labels, matched_gt_boxes)
     *         labels: -1 (ignore), 0 (background), 1 (object)
     */
    auto assign_anchors_to_gt(const Tensor& anchors, const Tensor& gt_boxes)
        -> std::pair<Tensor, Tensor>;

    /**
     * @brief Sample positive and negative anchors for training.
     *
     * @param labels Anchor labels (-1, 0, 1)
     * @return Indices of sampled anchors
     */
    auto sample_anchors(const Tensor& labels) -> Tensor;

    /**
     * @brief Generate proposals from anchors and predictions.
     *
     * @param anchors Anchor boxes (num_anchors, 4)
     * @param objectness Objectness scores (num_anchors,)
     * @param box_deltas Box regression deltas (num_anchors, 4)
     * @param image_shape Image (height, width)
     * @return Filtered proposals (K, 4)
     */
    auto generate_proposals(
        const Tensor& anchors,
        const Tensor& objectness,
        const Tensor& box_deltas,
        const std::pair<int64_t, int64_t>& image_shape
    ) -> Tensor;

    std::shared_ptr<RPNHead> head_;
    std::shared_ptr<AnchorGenerator> anchor_generator_;

    // Training parameters
    double fg_iou_thresh_;
    double bg_iou_thresh_;
    int64_t batch_size_per_image_;
    double positive_fraction_;

    // Inference parameters
    int64_t pre_nms_top_n_;
    int64_t post_nms_top_n_;
    double nms_thresh_;
    double score_thresh_;

    // Cached losses (set during forward)
    mutable Variable loss_objectness_;
    mutable Variable loss_rpn_box_reg_;
};

} // namespace detection
} // namespace nn
} // namespace tenzor
