/**
 * @file rpn.cpp
 * @brief Region Proposal Network implementation
 */

#include "tenzor/nn/detection/rpn.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/detection.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/core/tensor.hpp"
#include <stdexcept>
#include <algorithm>
#include <vector>

namespace tenzor {
namespace nn {
namespace detection {

// ============================================================================
// RPNHead Implementation
// ============================================================================

RPNHead::RPNHead(int64_t in_channels, int64_t num_anchors)
    : num_anchors_(num_anchors) {

    // 3x3 conv for spatial context
    conv_ = std::make_shared<Conv2d>(in_channels, in_channels, 3, 1, 1);
    // Note: RPNHead doesn't inherit from Module, so modules are registered
    // by the parent RegionProposalNetwork

    // Objectness: 1 score per anchor (foreground vs background)
    cls_logits_ = std::make_shared<Conv2d>(in_channels, num_anchors, 1);

    // Box regression: 4 deltas (dx, dy, dw, dh) per anchor
    bbox_pred_ = std::make_shared<Conv2d>(in_channels, num_anchors * 4, 1);
}

auto RPNHead::forward(const Variable& features)
    -> std::pair<Variable, Variable> {

    // Apply 3x3 conv with ReLU
    ReLU relu;
    auto x = relu.forward(conv_->forward(features));

    // Objectness classification
    auto objectness = cls_logits_->forward(x);  // (N, num_anchors, H, W)

    // Box regression
    auto bbox_reg = bbox_pred_->forward(x);  // (N, num_anchors*4, H, W)

    // Reshape for easier processing
    // objectness: (N, num_anchors, H, W) -> (N, H*W*num_anchors)
    auto N = objectness.shape()[0];
    auto num_anchors = objectness.shape()[1];
    auto H = objectness.shape()[2];
    auto W = objectness.shape()[3];

    objectness = reshape(objectness, {N, num_anchors, H * W});
    objectness = transpose(objectness, 1, 2);  // (N, H*W, num_anchors)
    objectness = reshape(objectness, {N, -1});  // (N, H*W*num_anchors)

    // bbox_reg: (N, num_anchors*4, H, W) -> (N, H*W*num_anchors, 4)
    bbox_reg = reshape(bbox_reg, {N, num_anchors, 4, H * W});
    bbox_reg = transpose(bbox_reg, 2, 3);  // (N, num_anchors, H*W, 4)
    bbox_reg = reshape(bbox_reg, {N, -1, 4});  // (N, H*W*num_anchors, 4)

    return {objectness, bbox_reg};
}

// ============================================================================
// RegionProposalNetwork Implementation
// ============================================================================

RegionProposalNetwork::RegionProposalNetwork(
    int64_t in_channels,
    std::shared_ptr<AnchorGenerator> anchor_generator,
    double fg_iou_thresh,
    double bg_iou_thresh,
    int64_t batch_size_per_image,
    double positive_fraction,
    int64_t pre_nms_top_n,
    int64_t post_nms_top_n,
    double nms_thresh,
    double score_thresh)
    : anchor_generator_(std::move(anchor_generator)),
      fg_iou_thresh_(fg_iou_thresh),
      bg_iou_thresh_(bg_iou_thresh),
      batch_size_per_image_(batch_size_per_image),
      positive_fraction_(positive_fraction),
      pre_nms_top_n_(pre_nms_top_n),
      post_nms_top_n_(post_nms_top_n),
      nms_thresh_(nms_thresh),
      score_thresh_(score_thresh) {

    head_ = std::make_shared<RPNHead>(
        in_channels,
        anchor_generator_->num_anchors_per_location()
    );
    // Register the Conv2d layers from RPNHead since it's not a Module
    register_module("head_conv", head_->conv_);
    register_module("head_cls_logits", head_->cls_logits_);
    register_module("head_bbox_pred", head_->bbox_pred_);
}

auto RegionProposalNetwork::forward_impl(const Variable& input) -> Variable {
    throw std::runtime_error(
        "RegionProposalNetwork requires image_shapes. "
        "Use forward_proposals() instead."
    );
}

auto RegionProposalNetwork::assign_anchors_to_gt(
    const Tensor& anchors,
    const Tensor& gt_boxes)
    -> std::pair<Tensor, Tensor> {

    int64_t num_anchors = anchors.shape()[0];
    int64_t num_gt = gt_boxes.shape()[0];

    // Handle edge case: no ground truth boxes
    // All anchors are background, matched_gt_boxes is zeros (won't be used)
    if (num_gt == 0) {
        auto labels = zeros({num_anchors}, DType::Int64, anchors.device());  // All background (0)
        auto matched_gt_boxes = zeros({num_anchors, 4}, anchors.dtype(), anchors.device());
        return {labels, matched_gt_boxes};
    }

    // Compute IoU between all anchors and ground truth boxes
    auto iou_matrix = ops::box_iou(anchors, gt_boxes);  // (num_anchors, num_gt)

    // For each anchor, find best matching GT box
    auto max_iou_per_anchor = ops::max(iou_matrix, 1);  // (num_anchors,)
    auto matched_gt_idx = ops::argmax(iou_matrix, 1);   // (num_anchors,)

    // Initialize labels: -1 (ignore), 0 (background), 1 (foreground)
    // Start with all labels as -1 (ignore)
    std::vector<int64_t> label_data(num_anchors, -1);

    // Get IoU values as CPU Float32 tensor for data access
    auto max_iou_cpu = max_iou_per_anchor.to(Device::cpu()).to(DType::Float32);
    const float* iou_data = max_iou_cpu.data<float>();

    // Set labels based on IoU thresholds
    for (int64_t i = 0; i < num_anchors; ++i) {
        if (iou_data[i] < bg_iou_thresh_) {
            label_data[i] = 0;  // background
        } else if (iou_data[i] >= fg_iou_thresh_) {
            label_data[i] = 1;  // foreground
        }
        // else: keep as -1 (ignore)
    }

    // Create labels tensor and copy data to avoid dangling pointer
    // (label_data vector would be destroyed, leaving tensor pointing to freed memory)
    auto labels = zeros({num_anchors}, DType::Int64, Device::cpu());
    int64_t* labels_ptr = labels.data<int64_t>();
    std::copy(label_data.begin(), label_data.end(), labels_ptr);

    if (anchors.device() != Device::cpu()) {
        labels = labels.to(anchors.device());
    }

    // Get matched ground truth boxes for each anchor

    auto matched_gt_boxes = ops::index_select(gt_boxes, 0, matched_gt_idx);

    return {labels, matched_gt_boxes};
}

auto RegionProposalNetwork::sample_anchors(const Tensor& labels) -> Tensor {

    // Sample positive and negative anchors for training
    auto ones_tensor = full(std::vector<int64_t>(labels.shape().begin(), labels.shape().end()),
                            static_cast<float>(1), labels.dtype(), labels.device());
    auto zeros_tensor = full(std::vector<int64_t>(labels.shape().begin(), labels.shape().end()),
                             static_cast<float>(0), labels.dtype(), labels.device());
    auto positive_mask = eq(labels, ones_tensor);
    auto negative_mask = eq(labels, zeros_tensor);

    // Get indices of positive and negative anchors
    // BUG FIX: Use nonzero indices directly instead of sum to avoid sum/nonzero mismatch on GPU
    auto pos_indices = ops::nonzero(positive_mask).squeeze(-1);
    auto neg_indices = ops::nonzero(negative_mask).squeeze(-1);

    // Use actual number of indices
    auto num_pos = pos_indices.numel();
    auto num_neg = neg_indices.numel();

    // Determine number of samples
    int64_t num_pos_samples = static_cast<int64_t>(
        batch_size_per_image_ * positive_fraction_
    );
    num_pos_samples = std::min(num_pos_samples, num_pos);

    int64_t num_neg_samples = batch_size_per_image_ - num_pos_samples;
    num_neg_samples = std::min(num_neg_samples, num_neg);


    // Randomly sample
    // BUG FIX: Use actual number of indices, not num_pos/num_neg
    // num_pos/num_neg count labels==1 or labels==0, but we need to sample from pos_indices/neg_indices
    // Note: randperm only supports CPU, so generate on CPU and move to target device
    auto target_device = labels.device();

    if (num_pos_samples == 0) {
        // Create empty tensor when no positive samples needed
        pos_indices = Tensor({0}, DType::Int64, target_device);
    } else if (num_pos_samples < pos_indices.numel()) {
        auto perm = ops::randperm(pos_indices.numel(), Device::cpu()).to(target_device);
        pos_indices = ops::index_select(pos_indices, 0,
                                        slice(perm, 0, 0, num_pos_samples));
    }

    if (num_neg_samples == 0) {
        // Create empty tensor when no negative samples needed
        neg_indices = Tensor({0}, DType::Int64, target_device);
    } else if (num_neg_samples < neg_indices.numel()) {
        auto perm = ops::randperm(neg_indices.numel(), Device::cpu()).to(target_device);
        neg_indices = ops::index_select(neg_indices, 0,
                                        slice(perm, 0, 0, num_neg_samples));
    }


    // Combine positive and negative samples
    auto result = ops::cat({pos_indices, neg_indices}, 0);


    return result;
}

auto RegionProposalNetwork::generate_proposals(
    const Tensor& anchors,
    const Tensor& objectness,
    const Tensor& box_deltas,
    const std::pair<int64_t, int64_t>& image_shape)
    -> Tensor {

    // Decode boxes from deltas
    auto proposals = ops::decode_boxes(box_deltas, anchors);

    // Clip to image boundaries
    proposals = ops::clip_boxes_to_image(proposals,
                                         image_shape.first,
                                         image_shape.second);

    // Remove small boxes - use very small threshold to keep more proposals
    // Original threshold of 1.0 is too aggressive for small images
    auto keep = ops::remove_small_boxes(proposals, objectness, 0.001);

    // Handle empty case: return empty proposals
    if (keep.numel() == 0) {
        return ops::zeros({0, 4}, proposals.dtype(), proposals.device());
    }

    proposals = ops::index_select(proposals, 0, keep);
    auto scores = ops::index_select(objectness, 0, keep);

    // Sort by score and keep top pre_nms_top_n
    auto sorted_indices = ops::argsort(scores, 0, true);  // descending

    if (sorted_indices.shape()[0] > pre_nms_top_n_) {
        sorted_indices = slice(sorted_indices, 0, 0, pre_nms_top_n_);
    }

    proposals = ops::index_select(proposals, 0, sorted_indices);
    scores = ops::index_select(scores, 0, sorted_indices);

    // Apply NMS
    auto keep_nms = ops::nms(proposals, scores, nms_thresh_);

    // Keep top post_nms_top_n
    if (keep_nms.shape()[0] > post_nms_top_n_) {
        keep_nms = slice(keep_nms, 0, 0, post_nms_top_n_);
    }

    return ops::index_select(proposals, 0, keep_nms);
}

auto RegionProposalNetwork::forward_proposals(
    const Variable& features,
    const std::vector<std::pair<int64_t, int64_t>>& image_shapes,
    const std::vector<Tensor>* targets)
    -> std::vector<Tensor> {

    // Get RPN predictions
    auto [objectness, box_regression] = head_->forward(features);

    // Get feature map dimensions
    auto feat_h = features.shape()[2];
    auto feat_w = features.shape()[3];
    auto batch_size = features.shape()[0];

    // Generate anchors for this feature map
    // Assuming stride of 16 (typical for ResNet-50 C4 features)
    int64_t stride = 16;
    auto anchors = anchor_generator_->generate(
        feat_h, feat_w, stride, features.device()
    );

    // Convert anchors to match features dtype
    if (anchors.dtype() != features.dtype()) {
        anchors = anchors.to(features.dtype());
    }

    std::vector<Tensor> all_proposals;

    // Process each image in batch
    for (int64_t i = 0; i < batch_size; ++i) {
        // Extract predictions for this image
        auto img_objectness = ops::select(objectness.tensor(), 0, i);
        auto img_box_reg = ops::select(box_regression.tensor(), 0, i);

        // Training mode: compute losses
        if (is_training() && targets != nullptr && static_cast<size_t>(i) < targets->size()) {
            auto gt_boxes = (*targets)[i];

            // Assign anchors to ground truth
            auto [labels, matched_gt_boxes] = assign_anchors_to_gt(
                anchors, gt_boxes
            );

            // Sample anchors
            auto sampled_indices = sample_anchors(labels);

            // Compute classification loss (binary cross entropy)
            auto sampled_objectness = ops::index_select(
                img_objectness, 0, sampled_indices
            );

            auto sampled_labels = ops::index_select(
                labels, 0, sampled_indices
            );

            BCEWithLogitsLoss bce_loss;
            // Convert labels to same dtype as objectness for BCE loss
            auto sampled_labels_float = sampled_labels.to(sampled_objectness.dtype());
            auto cls_loss = bce_loss(
                Variable(sampled_objectness, true),
                Variable(sampled_labels_float, false)
            );

            // Compute regression loss (smooth L1)
            auto ones_tensor = full(std::vector<int64_t>(sampled_labels.shape().begin(), sampled_labels.shape().end()),
                                    static_cast<float>(1), sampled_labels.dtype(), sampled_labels.device());
            auto positive_mask = eq(sampled_labels, ones_tensor);

            // Get positive indices first to avoid sum/nonzero mismatch on GPU
            auto pos_indices = ops::nonzero(positive_mask).squeeze(-1);
            auto num_positives = pos_indices.numel();

            Variable reg_loss;
            if (num_positives > 0) {
                auto pos_box_reg = ops::index_select(
                    ops::index_select(img_box_reg, 0, sampled_indices),
                    0, pos_indices
                );
                auto pos_matched_gt = ops::index_select(
                    ops::index_select(matched_gt_boxes, 0, sampled_indices),
                    0, pos_indices
                );
                auto pos_anchors = ops::index_select(
                    ops::index_select(anchors, 0, sampled_indices),
                    0, pos_indices
                );

                // Encode ground truth boxes relative to anchors
                auto target_deltas = ops::encode_boxes(
                    pos_matched_gt, pos_anchors
                );

                SmoothL1Loss smooth_l1;
                reg_loss = smooth_l1(
                    Variable(pos_box_reg, true),
                    Variable(target_deltas, false)
                );
            } else {
                // No positive samples - create zero loss
                auto zero_tensor = ops::zeros({1}, sampled_objectness.dtype(), sampled_objectness.device());
                reg_loss = Variable(zero_tensor, true);
            }

            // Accumulate losses
            if (i == 0) {
                loss_objectness_ = cls_loss;
                loss_rpn_box_reg_ = reg_loss;
            } else {
                loss_objectness_ = loss_objectness_ + cls_loss;
                loss_rpn_box_reg_ = loss_rpn_box_reg_ + reg_loss;
            }
        }

        // Generate proposals for this image
        auto img_proposals = generate_proposals(
            anchors,
            img_objectness,
            img_box_reg,
            image_shapes[i]
        );


        all_proposals.push_back(img_proposals);

    }


    // Average losses over batch
    if (is_training() && targets != nullptr) {

        loss_objectness_ = loss_objectness_ / static_cast<double>(batch_size);

        loss_rpn_box_reg_ = loss_rpn_box_reg_ / static_cast<double>(batch_size);
    }


    return all_proposals;
}

auto RegionProposalNetwork::get_losses() const
    -> std::unordered_map<std::string, Variable> {

    std::unordered_map<std::string, Variable> losses;
    losses["loss_objectness"] = loss_objectness_;
    losses["loss_rpn_box_reg"] = loss_rpn_box_reg_;
    return losses;
}

} // namespace detection
} // namespace nn
} // namespace tenzor
