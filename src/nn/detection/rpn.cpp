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
#include "tenzor/jit/tracer.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
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
    // Must use the SAME flat ordering as objectness and AnchorGenerator,
    // namely location-major / anchor-minor: flat row = (h*W+w)*num_anchors + a.
    // Start from (N, num_anchors, 4, H*W), then move H*W ahead of num_anchors
    // (dims [0, 3, 1, 2] -> (N, H*W, num_anchors, 4)) before the final reshape,
    // so the per-row (anchor, location) pairing matches the objectness scores
    // and the anchors produced by AnchorGenerator.
    bbox_reg = reshape(bbox_reg, {N, num_anchors, 4, H * W});
    bbox_reg = permute(bbox_reg, {0, 3, 1, 2});  // (N, H*W, num_anchors, 4)
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

auto RegionProposalNetwork::forward_impl([[maybe_unused]] const Variable& input) -> Variable {
    throw std::runtime_error(
        "RegionProposalNetwork requires image_shapes. "
        "Use forward_proposals() instead."
    );
}

auto RegionProposalNetwork::assign_anchors_to_gt(
    const Tensor& anchors,
    const Tensor& gt_boxes)
    -> std::pair<Tensor, Tensor> {

    // JIT-R086/R089: below this point, label assignment reads the actual IoU
    // VALUES host-side (.data<float>() over max_iou_per_anchor/iou_matrix) and
    // writes the result into a fresh zeros() tensor via a raw pointer copy —
    // the JIT-R085 "zeros() traced, real values written in after
    // registration" mechanism. A trace would silently freeze whichever
    // foreground/background assignment the trace-dummy's GT boxes happened
    // to produce, replaying it unchanged for every real image's actual GT —
    // a silent wrong-answer bug in a safety-relevant detection pipeline, not
    // a crash. No general dynamic-value-dependent-output tracing primitive
    // exists in this codebase for this shape of computation (matches JIT-
    // R050/R051's ACT/routing precedent). Refuse loudly and unconditionally
    // — NOT gated on num_gt, since a num_gt==0 trace-time warm-up would
    // otherwise silently freeze to the "all background" path per this
    // finding's own documented secondary failure mode. Run eagerly instead.
    if (::tenzor::jit::Tracer::get_instance().is_tracing()) {
        throw std::runtime_error(
            "RegionProposalNetwork::assign_anchors_to_gt cannot be traced by "
            "@tz.jit — anchor-to-ground-truth label assignment reads IoU "
            "values host-side and would be permanently frozen to this trace "
            "call's ground-truth boxes, silently producing wrong labels for "
            "every other image. Run training-mode RPN calls eagerly (outside "
            "a traced region).");
    }

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

    // Force-match each GT to its highest-IoU anchor(s) (allow_low_quality_matches),
    // mirroring torchvision Matcher.set_low_quality_matches_. Without this, a GT
    // whose best anchor IoU falls in the [bg_iou_thresh_, fg_iou_thresh_) band gets
    // no positive anchor at all. For every GT we take its max IoU over all anchors,
    // then force each anchor achieving that max to foreground (overriding any
    // background/ignore label). matched_gt_idx already holds each anchor's argmax GT
    // (== torchvision's all_matches), so it needs no adjustment here.
    {
        auto iou_cpu = iou_matrix.to(Device::cpu()).to(DType::Float32);
        const float* iou_mat = iou_cpu.data<float>();  // row-major (num_anchors, num_gt)
        std::vector<float> best_iou_per_gt(num_gt, -1.0f);
        for (int64_t a = 0; a < num_anchors; ++a) {
            for (int64_t g = 0; g < num_gt; ++g) {
                float v = iou_mat[a * num_gt + g];
                if (v > best_iou_per_gt[g]) best_iou_per_gt[g] = v;
            }
        }
        for (int64_t a = 0; a < num_anchors; ++a) {
            for (int64_t g = 0; g < num_gt; ++g) {
                // Exact-equality match against the per-GT max, as in torchvision's
                // (match_quality_matrix == highest_quality_foreach_gt) comparison.
                if (best_iou_per_gt[g] > 0.0f &&
                    iou_mat[a * num_gt + g] == best_iou_per_gt[g]) {
                    label_data[a] = 1;  // foreground
                }
            }
        }
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

    // JIT-R088: below this point, several host-side C++ `if` branches key
    // off runtime-computed, data-dependent tensor shapes/counts (keep.numel()
    // ==0, score_keep.numel()==0, sorted_indices.shape()[0]>pre_nms_top_n_,
    // keep_nms.shape()[0]>post_nms_top_n_) — a trace only ever records
    // whichever branch the trace-dummy's proposal counts happened to take,
    // and replay blindly re-executes that fixed set of recorded nodes
    // regardless of a real image's actual surviving-proposal count (matches
    // the JIT-R050/R051 MoE/HRM data-dependent-control-flow class exactly).
    // No general dynamic-control-flow tracing primitive is safely applicable
    // here without risking a silently-wrong compiled graph for the realistic
    // case (pre/post_nms_top_n_ caps are exceeded on almost every real
    // input). Refuse loudly rather than silently bake in one specific
    // trace-time branch structure.
    if (::tenzor::jit::Tracer::get_instance().is_tracing()) {
        throw std::runtime_error(
            "RegionProposalNetwork::generate_proposals cannot be traced by "
            "@tz.jit — proposal filtering (empty-box/score-threshold/top-N "
            "branches) depends on runtime-computed box counts and would be "
            "permanently frozen to this trace call's specific counts, "
            "silently misprocessing any other image with a different "
            "surviving-proposal count. Run RPN proposal generation eagerly "
            "(outside a traced region).");
    }

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

    // Apply the objectness score threshold (mirrors torchvision RPN.filter_proposals,
    // which sigmoids the objectness logits and drops boxes scoring below score_thresh
    // before NMS). `objectness` here are raw logits; since sigmoid is monotonic,
    // sigmoid(s) >= score_thresh_ is equivalent to s >= logit(score_thresh_), so we
    // threshold the logits directly and avoid an extra sigmoid pass.
    if (score_thresh_ > 0.0) {
        if (score_thresh_ >= 1.0) {
            // No score can reach a probability of 1; nothing survives.
            return ops::zeros({0, 4}, proposals.dtype(), proposals.device());
        }
        const double logit_thresh = std::log(score_thresh_ / (1.0 - score_thresh_));
        auto thresh = full(std::vector<int64_t>(scores.shape().begin(), scores.shape().end()),
                           static_cast<float>(logit_thresh), scores.dtype(), scores.device());
        auto score_keep = ops::nonzero(ge(scores, thresh)).squeeze(-1);
        if (score_keep.numel() == 0) {
            return ops::zeros({0, 4}, proposals.dtype(), proposals.device());
        }
        proposals = ops::index_select(proposals, 0, score_keep);
        scores = ops::index_select(scores, 0, score_keep);
    }

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

    // Generate anchors for this feature map.
    //
    // The feature stride is the ratio of the input image size to the feature
    // map size, not a fixed 16: an FPN level or a non-C4 backbone has a
    // different stride, and hard-coding 16 places anchor centers at the wrong
    // image coordinates (cx = (x+0.5)*stride). Derive it from the first image
    // shape (image_height / feat_h), falling back to 16 if unavailable.
    int64_t stride = 16;
    if (!image_shapes.empty() && feat_h > 0) {
        int64_t img_h = image_shapes[0].first;  // (height, width)
        if (img_h > 0) {
            stride = std::max<int64_t>(1, static_cast<int64_t>(
                std::llround(static_cast<double>(img_h) /
                             static_cast<double>(feat_h))));
        }
    }
    auto anchors = anchor_generator_->generate(
        feat_h, feat_w, stride, features.device()
    );

    // Convert anchors to match features dtype
    if (anchors.dtype() != features.dtype()) {
        anchors = anchors.to(features.dtype());
    }

    std::vector<Tensor> all_proposals;

    // Reset the mutable loss accumulators to fresh zero Variables at the start
    // of every forward. Previously they were keyed on the loop index (i == 0),
    // so if image 0 had no target the init was skipped and the final average
    // divided a stale value left over from a prior forward. Track how many
    // images actually contributed a loss so the average uses the real count.
    const bool computing_losses = is_training() && targets != nullptr;
    int64_t num_loss_images = 0;
    if (computing_losses) {
        loss_objectness_ = Variable(
            ops::zeros({1}, features.dtype(), features.device()), true);
        loss_rpn_box_reg_ = Variable(
            ops::zeros({1}, features.dtype(), features.device()), true);
    }

    // Process each image in batch
    for (int64_t i = 0; i < batch_size; ++i) {
        // Extract predictions for this image
        auto img_objectness = ops::select(objectness.tensor(), 0, i);
        auto img_box_reg = ops::select(box_regression.tensor(), 0, i);

        // Training mode: compute losses
        if (is_training() && targets != nullptr && static_cast<size_t>(i) < targets->size()) {
            auto gt_boxes = (*targets)[i];

            // Autograd-aware per-image slices: feed the losses from the Variable
            // predictions (not objectness.tensor()) so gradients reach conv_/
            // cls_logits_/bbox_pred_. The raw-tensor img_objectness/img_box_reg
            // above are only used for (non-differentiable) proposal generation.
            auto img_objectness_v = tenzor::squeeze(tenzor::slice(objectness, 0, i, i + 1), 0);
            auto img_box_reg_v    = tenzor::squeeze(tenzor::slice(box_regression, 0, i, i + 1), 0);

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
            auto sampled_objectness_v = tenzor::index_select(img_objectness_v, 0, sampled_indices);
            auto cls_loss = bce_loss(
                sampled_objectness_v,
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
                // Autograd-aware: select from the Variable prediction so the
                // regression loss backprops into bbox_pred_.
                auto pos_box_reg_v = tenzor::index_select(
                    tenzor::index_select(img_box_reg_v, 0, sampled_indices),
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
                    pos_box_reg_v,
                    Variable(target_deltas, false)
                );
            } else {
                // No positive samples - create zero loss
                auto zero_tensor = ops::zeros({1}, sampled_objectness.dtype(), sampled_objectness.device());
                reg_loss = Variable(zero_tensor, true);
            }

            // Accumulate losses unconditionally onto the zero-initialised
            // accumulators (image index is irrelevant to whether this image
            // had a target).
            loss_objectness_ = loss_objectness_ + cls_loss;
            loss_rpn_box_reg_ = loss_rpn_box_reg_ + reg_loss;
            ++num_loss_images;
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


    // Average losses over the number of images that actually had targets
    // (not the full batch_size, which would scale the loss down when
    // targets->size() < batch_size).
    if (computing_losses && num_loss_images > 0) {
        loss_objectness_ = loss_objectness_ / static_cast<double>(num_loss_images);
        loss_rpn_box_reg_ = loss_rpn_box_reg_ / static_cast<double>(num_loss_images);
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
