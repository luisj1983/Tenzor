/**
 * @file roi_head.cpp
 * @brief ROI Head implementation
 */

#include "tenzor/nn/detection/roi_head.hpp"
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
#include <cmath>
#include <vector>

namespace tenzor {
namespace nn {
namespace detection {

// ============================================================================
// RoIBoxHead Implementation
// ============================================================================

RoIBoxHead::RoIBoxHead(int64_t in_channels,
                       int64_t roi_size,
                       int64_t num_classes,
                       int64_t representation_size)
    : num_classes_(num_classes) {

    // Calculate input features size
    int64_t in_features = in_channels * roi_size * roi_size;

    // Two fully connected layers for feature extraction
    fc1_ = std::make_shared<Linear>(in_features, representation_size);
    register_module("fc1", fc1_);

    fc2_ = std::make_shared<Linear>(representation_size, representation_size);
    register_module("fc2", fc2_);

    // Classification head: num_classes + 1 (background)
    cls_score_ = std::make_shared<Linear>(representation_size, num_classes + 1);
    register_module("cls_score", cls_score_);

    // Box regression head: 4 deltas per class (class-specific)
    bbox_pred_ = std::make_shared<Linear>(representation_size,
                                           num_classes * 4);
    register_module("bbox_pred", bbox_pred_);
}

auto RoIBoxHead::forward_features(const Variable& roi_features)
    -> std::pair<Variable, Variable> {

    // Flatten ROI features
    auto num_rois = roi_features.shape()[0];

    // Handle empty ROI features
    if (num_rois == 0) {
        auto empty_logits = Variable(
            ops::zeros({0, num_classes_ + 1}, roi_features.dtype(), roi_features.device()), false);
        auto empty_deltas = Variable(
            ops::zeros({0, num_classes_ * 4}, roi_features.dtype(), roi_features.device()), false);
        return {empty_logits, empty_deltas};
    }

    auto flattened = reshape(roi_features, {num_rois, -1});

    // Two FC layers with ReLU
    ReLU relu;
    auto x = relu.forward(fc1_->forward(flattened));
    x = relu.forward(fc2_->forward(x));

    // Classification and regression heads
    auto class_logits = cls_score_->forward(x);
    auto box_deltas = bbox_pred_->forward(x);

    return {class_logits, box_deltas};
}

auto RoIBoxHead::forward_impl(const Variable& input) -> Variable {
    auto [class_logits, _] = forward_features(input);
    return class_logits;
}

// ============================================================================
// RoIHead Implementation
// ============================================================================

RoIHead::RoIHead(int64_t in_channels,
                 int64_t num_classes,
                 int64_t roi_output_size,
                 double spatial_scale,
                 int64_t sampling_ratio,
                 double fg_iou_thresh,
                 double bg_iou_thresh,
                 int64_t batch_size_per_image,
                 double positive_fraction,
                 double score_thresh,
                 double nms_thresh,
                 int64_t detections_per_img)
    : num_classes_(num_classes),
      fg_iou_thresh_(fg_iou_thresh),
      bg_iou_thresh_(bg_iou_thresh),
      batch_size_per_image_(batch_size_per_image),
      positive_fraction_(positive_fraction),
      score_thresh_(score_thresh),
      nms_thresh_(nms_thresh),
      detections_per_img_(detections_per_img) {

    // ROI Align for feature extraction
    roi_align_ = std::make_shared<ROIAlign>(
        roi_output_size, roi_output_size,
        spatial_scale, sampling_ratio
    );
    register_module("roi_align", roi_align_);

    // Box head for classification and regression
    box_head_ = std::make_shared<RoIBoxHead>(
        in_channels, roi_output_size, num_classes
    );
    register_module("box_head", box_head_);

    // Initialize loss members to a valid zero scalar Variable so get_losses()
    // returns usable (zero) Variables after an inference-mode forward (where the
    // training/gt branch never assigns them). Otherwise callers would deref a
    // default-constructed Variable wrapping no tensor.
    loss_classifier_ = Variable(ops::zeros({1}, DType::Float32, Device::cpu()), false);
    loss_box_reg_ = Variable(ops::zeros({1}, DType::Float32, Device::cpu()), false);
}

auto RoIHead::forward_impl([[maybe_unused]] const Variable& input) -> Variable {
    throw std::runtime_error(
        "RoIHead requires proposals and image_shapes. "
        "Use forward_detections() instead."
    );
}

auto RoIHead::match_proposals_to_gt(const Tensor& proposals,
                                     const Tensor& gt_boxes,
                                     const Tensor& gt_labels)
    -> std::pair<Tensor, Tensor> {

    int64_t num_proposals = proposals.shape()[0];
    int64_t num_gt = gt_boxes.shape()[0];

    // Handle edge case: no ground truth boxes
    // All proposals are background, matched_gt_boxes is zeros (won't be used)
    if (num_gt == 0) {
        auto labels = zeros({num_proposals}, DType::Int64, proposals.device());  // All background (0)
        auto matched_gt_boxes = zeros({num_proposals, 4}, proposals.dtype(), proposals.device());
        return {labels, matched_gt_boxes};
    }

    // Compute IoU between proposals and ground truth
    auto iou_matrix = ops::box_iou(proposals, gt_boxes);  // (num_proposals, num_gt)


    // For each proposal, find best matching GT box
    auto max_iou_per_proposal = ops::max(iou_matrix, 1);

    auto matched_gt_idx = ops::argmax(iou_matrix, 1);

    // Get matched GT boxes
    auto matched_gt_boxes = ops::index_select(gt_boxes, 0, matched_gt_idx);

    // Assign labels based on IoU thresholds
    // (num_proposals already declared at top of function)

    // Labels: -1 (ignore), 0 (background), >0 (foreground class index).
    // Mirrors RPN::assign_anchors_to_gt, which uses both fg_iou_thresh_ and
    // bg_iou_thresh_ with a -1 ignore band for the [bg, fg) IoU range so those
    // ambiguous proposals contribute to neither the positive nor the negative
    // sampling pools (sample_rois selects positives via gt(labels,0) and
    // negatives via eq(labels,0); a -1 label matches neither and is dropped).
    // Start with all labels as -1 (ignore).
    std::vector<int64_t> label_data(num_proposals, -1);

    // Get IoU values and matched indices as CPU tensors
    auto max_iou_cpu = max_iou_per_proposal.to(Device::cpu());
    auto matched_idx_cpu = matched_gt_idx.to(Device::cpu());
    auto gt_labels_cpu = gt_labels.to(Device::cpu());

    // Convert IoU to Float32 if needed for data access
    Tensor max_iou_f32 = max_iou_cpu;
    if (max_iou_cpu.dtype() != DType::Float32) {
        max_iou_f32 = max_iou_cpu.to(DType::Float32);
    }
    const float* iou_data = max_iou_f32.data<float>();
    const int64_t* matched_idx_data = matched_idx_cpu.data<int64_t>();
    const int64_t* gt_labels_data = gt_labels_cpu.data<int64_t>();

    // Set labels based on IoU thresholds
    // Use actual class labels from matched GT boxes
    for (int64_t i = 0; i < num_proposals; ++i) {
        if (iou_data[i] < bg_iou_thresh_) {
            label_data[i] = 0;  // background
        } else if (iou_data[i] >= fg_iou_thresh_) {
            // Assign the actual GT class label
            int64_t matched_idx = matched_idx_data[i];
            label_data[i] = gt_labels_data[matched_idx];  // Use actual class label
        }
        // else: keep as -1 (ignore) for the [bg_iou_thresh_, fg_iou_thresh_) band
    }

    // Create labels tensor and copy data to avoid dangling pointer
    // (label_data vector would be destroyed, leaving tensor pointing to freed memory)
    auto labels = zeros({num_proposals}, DType::Int64, Device::cpu());
    int64_t* labels_ptr = labels.data<int64_t>();
    std::copy(label_data.begin(), label_data.end(), labels_ptr);

    if (proposals.device() != Device::cpu()) {
        labels = labels.to(proposals.device());
    }

    return {labels, matched_gt_boxes};
}

auto RoIHead::sample_rois(const Tensor& labels) -> Tensor {

    // Sample positive and negative ROIs for training
    auto zero_tensor = full(std::vector<int64_t>(labels.shape().begin(), labels.shape().end()),
                            static_cast<float>(0), labels.dtype(), labels.device());
    auto positive_mask = gt(labels, zero_tensor);
    auto negative_mask = eq(labels, zero_tensor);

    // Get indices first to avoid sum/nonzero mismatch on GPU
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
    // Same fix as applied to RPN - must use pos_indices.numel() and neg_indices.numel()
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

    return cat({pos_indices, neg_indices}, 0);
}

auto RoIHead::postprocess_detections(
    const Tensor& class_logits,
    const Tensor& box_deltas,
    const Tensor& proposals,
    const std::pair<int64_t, int64_t>& image_shape)
    -> std::unordered_map<std::string, Tensor> {

    std::unordered_map<std::string, Tensor> result;

    auto num_rois = proposals.shape()[0];

    // Handle empty proposals
    if (num_rois == 0) {
        result["boxes"] = ops::zeros({0, 4}, proposals.dtype(), proposals.device());
        result["scores"] = ops::zeros({0}, proposals.dtype(), proposals.device());
        result["labels"] = ops::zeros({0}, DType::Int64, proposals.device());
        return result;
    }

    // Apply softmax to get class probabilities
    auto class_probs = tenzor::softmax(Variable(class_logits, false), 1).tensor();
    auto box_deltas_reshaped = tenzor::reshape(box_deltas,
                                             {num_rois, num_classes_, 4});

    std::vector<Tensor> all_boxes, all_scores, all_labels;

    // Process each class (skip background class 0)
    for (int64_t cls = 1; cls <= num_classes_; ++cls) {
        // Get scores for this class
        auto cls_scores = ops::select(class_probs, 1, cls);

        // Filter by score threshold
        auto thresh_tensor = full(std::vector<int64_t>(cls_scores.shape().begin(), cls_scores.shape().end()),
                                  static_cast<float>(score_thresh_), cls_scores.dtype(), cls_scores.device());
        auto score_mask = ge(cls_scores, thresh_tensor);
        auto keep_indices = ops::nonzero(score_mask).squeeze(-1);

        if (keep_indices.numel() == 0) {
            continue;
        }

        auto cls_scores_filtered = ops::index_select(cls_scores, 0, keep_indices);
        auto cls_proposals = ops::index_select(proposals, 0, keep_indices);

        // Get class-specific box deltas
        auto cls_box_deltas = ops::index_select(
            ops::select(box_deltas_reshaped, 1, cls - 1),  // 0-indexed for deltas
            0, keep_indices
        );

        // Decode boxes
        auto cls_boxes = ops::decode_boxes(cls_box_deltas, cls_proposals);
        cls_boxes = ops::clip_boxes_to_image(cls_boxes,
                                              image_shape.first,
                                              image_shape.second);

        // Apply NMS for this class
        auto nms_keep = ops::nms(cls_boxes, cls_scores_filtered, nms_thresh_);
        cls_boxes = ops::index_select(cls_boxes, 0, nms_keep);
        cls_scores_filtered = ops::index_select(cls_scores_filtered, 0, nms_keep);

        // Create class labels
        std::vector<int64_t> label_shape = {cls_boxes.shape()[0]};
        auto cls_labels = full(label_shape,
                               static_cast<float>(cls),
                               DType::Int64,
                               proposals.device());

        all_boxes.push_back(cls_boxes);
        all_scores.push_back(cls_scores_filtered);
        all_labels.push_back(cls_labels);
    }

    // Combine all classes
    if (all_boxes.empty()) {
        // No detections - use proposals dtype to maintain consistency
        result["boxes"] = ops::zeros({0, 4}, proposals.dtype(), proposals.device());
        result["scores"] = ops::zeros({0}, proposals.dtype(), proposals.device());
        result["labels"] = ops::zeros({0}, DType::Int64, proposals.device());
        return result;
    }

    auto final_boxes = cat(all_boxes, 0);
    auto final_scores = cat(all_scores, 0);
    auto final_labels = cat(all_labels, 0);

    // Sort by score and keep top detections_per_img
    auto sorted_indices = ops::argsort(final_scores, 0, true);  // descending
    if (sorted_indices.shape()[0] > detections_per_img_) {
        sorted_indices = slice(sorted_indices, 0, 0, detections_per_img_);
    }

    result["boxes"] = ops::index_select(final_boxes, 0, sorted_indices);
    result["scores"] = ops::index_select(final_scores, 0, sorted_indices);
    result["labels"] = ops::index_select(final_labels, 0, sorted_indices);

    return result;
}

auto RoIHead::forward_detections(
    const Variable& features,
    const std::vector<Tensor>& proposals,
    const std::vector<std::pair<int64_t, int64_t>>& image_shapes,
    const std::vector<Tensor>* gt_boxes,
    const std::vector<Tensor>* gt_labels)
    -> std::vector<std::unordered_map<std::string, Tensor>> {


    std::vector<std::unordered_map<std::string, Tensor>> all_detections;

    // Zero-initialise the loss accumulators ONCE and accumulate unconditionally
    // below (mirrors RegionProposalNetwork::forward_proposals). Keying init on
    // `i == 0` is fragile: if image 0 takes a branch that doesn't initialise the
    // accumulators, later images add to a stale/uninitialised value.
    loss_classifier_ = Variable(ops::zeros({1}, features.dtype(), features.device()), false);
    loss_box_reg_ = Variable(ops::zeros({1}, features.dtype(), features.device()), false);

    // Process each image in batch
    for (size_t i = 0; i < proposals.size(); ++i) {

        auto img_proposals = proposals[i];

        // Prepare ROIs in format: (batch_idx, x1, y1, x2, y2)
        auto num_proposals = img_proposals.shape()[0];

        auto batch_indices = ops::full({num_proposals, 1},
                                        static_cast<float>(i),
                                        img_proposals.dtype(),
                                        img_proposals.device());

        auto rois = ops::cat({batch_indices, img_proposals}, 1);

        // Extract ROI features

        auto roi_features = roi_align_->forward(features, rois);


        // Get predictions
        auto [class_logits, box_deltas] = box_head_->forward_features(roi_features);

        // Training mode: compute losses
        if (is_training() && gt_boxes != nullptr && gt_labels != nullptr &&
            i < gt_boxes->size() && i < gt_labels->size()) {

            auto img_gt_boxes = (*gt_boxes)[i];
            auto img_gt_labels = (*gt_labels)[i];


            // Match proposals to ground truth

            auto [labels, matched_gt_boxes] = match_proposals_to_gt(
                img_proposals, img_gt_boxes, img_gt_labels
            );


            // Sample ROIs
            auto sampled_indices = sample_rois(labels);


            // Handle edge case: no sampled indices (e.g., empty proposals)
            if (sampled_indices.numel() == 0) {
                auto zero_loss = Variable(ops::zeros({1}, features.dtype(), features.device()), false);
                loss_classifier_ = loss_classifier_ + zero_loss;
                loss_box_reg_ = loss_box_reg_ + zero_loss;
                // Push an empty detection map so all_detections stays aligned
                // with proposals (one entry per image). Skipping the push here
                // desynced detections[i] from image i for every later image.
                std::unordered_map<std::string, Tensor> empty_det;
                empty_det["boxes"] = ops::zeros({0, 4}, features.dtype(), features.device());
                empty_det["scores"] = ops::zeros({0}, features.dtype(), features.device());
                empty_det["labels"] = ops::zeros({0}, DType::Int64, features.device());
                all_detections.push_back(std::move(empty_det));
                continue;  // Skip to next image
            }

            // Get sampled predictions and targets. Use the autograd-aware
            // index_select on the *Variable* outputs of forward_features so the
            // grad_fn chain reaches fc1_/fc2_/cls_score_/bbox_pred_. Previously
            // these used class_logits.tensor()/box_deltas.tensor() (raw, no
            // autograd) and were re-wrapped as leaf Variables, severing the
            // graph so the ROI head never trained.
            auto sampled_logits = tenzor::index_select(
                class_logits, 0, sampled_indices
            );
            auto sampled_labels = ops::index_select(labels, 0, sampled_indices);
            auto sampled_box_deltas = tenzor::index_select(
                box_deltas, 0, sampled_indices
            );
            auto sampled_proposals = ops::index_select(
                img_proposals, 0, sampled_indices
            );
            auto sampled_matched_gt = ops::index_select(
                matched_gt_boxes, 0, sampled_indices
            );

            // Classification loss

            // Convert class indices to one-hot encoding
            // CrossEntropyLoss implementation expects one-hot encoded targets
            auto num_samples = sampled_labels.shape()[0];
            auto num_classes = sampled_logits.shape()[1];
            // Work in Float32 for one-hot encoding, convert back later
            auto one_hot = ops::zeros({num_samples, num_classes}, DType::Float32, Device::cpu());
            auto one_hot_data = one_hot.data<float>();
            auto labels_cpu = sampled_labels.to(Device::cpu());
            auto labels_data = labels_cpu.data<int64_t>();
            for (int64_t i = 0; i < num_samples; ++i) {
                auto label_idx = labels_data[i];
                if (label_idx >= 0 && label_idx < num_classes) {
                    one_hot_data[i * num_classes + label_idx] = 1.0f;
                }
            }
            // Convert one_hot to target dtype and device
            if (sampled_logits.dtype() != DType::Float32) {
                one_hot = one_hot.to(sampled_logits.dtype());
            }
            if (sampled_logits.device() != Device::cpu()) {
                one_hot = one_hot.to(sampled_logits.device());
            }

            CrossEntropyLoss ce_loss;
            auto cls_loss = ce_loss.forward(
                sampled_logits,
                one_hot
            );


            // Regression loss (only for foreground)
            auto zero_tensor = full(std::vector<int64_t>(sampled_labels.shape().begin(), sampled_labels.shape().end()),
                                    static_cast<float>(0), sampled_labels.dtype(), sampled_labels.device());
            auto fg_mask = gt(sampled_labels, zero_tensor);
            auto num_fg = ops::sum(fg_mask.to(DType::Int64)).item<int64_t>();

            // Get foreground indices
            auto fg_indices = ops::nonzero(fg_mask).squeeze(-1);

            // BUG FIX: Use actual number of indices from nonzero, not sum
            // There can be a mismatch between sum(mask) and nonzero(mask).numel() on GPU
            num_fg = fg_indices.numel();

            if (num_fg > 0) {
                // Get foreground predictions via the autograd-aware index_select
                // so the box-reg gradient flows back into bbox_pred_. The
                // previous raw ops::index_select + Variable(...,true) rewrap
                // severed the graph.
                auto fg_box_deltas = tenzor::index_select(
                    sampled_box_deltas, 0, fg_indices
                );

                // Proposals / matched GT are targets (no grad needed).
                auto fg_proposals = ops::index_select(
                    sampled_proposals, 0, fg_indices
                );
                auto fg_matched_gt = ops::index_select(
                    sampled_matched_gt, 0, fg_indices
                );

                // Encode ground truth boxes
                auto target_deltas = ops::encode_boxes(fg_matched_gt, fg_proposals);

                // For class-specific regression, select the right deltas.
                // Reshape to (num_fg, num_classes, 4) and select based on labels,
                // staying inside the autograd graph throughout.
                auto fg_labels = ops::index_select(sampled_labels, 0, fg_indices);
                auto fg_box_deltas_reshaped = tenzor::reshape(fg_box_deltas,
                                                            {num_fg, num_classes_, 4});

                // Select deltas for ground truth class (autograd-aware).
                // Read fg_labels to host ONCE (single DeviceToHost copy) and build a
                // (num_fg, 1, 4) gather index whose row j is the clamped, 0-indexed
                // class for that foreground sample, broadcast across the 4 box coords.
                // A single autograd-aware gather along dim 1 then selects the per-row
                // class deltas, eliminating the previous per-element item() device
                // sync and the per-iteration index_select allocations.
                auto fg_labels_cpu = fg_labels.to(Device::cpu()).to(DType::Int64);
                const int64_t* fg_labels_ptr = fg_labels_cpu.data<int64_t>();

                auto gather_index = ops::zeros({num_fg, 1, 4}, DType::Int64,
                                               Device::cpu());
                int64_t* gather_index_ptr = gather_index.data<int64_t>();
                for (int64_t j = 0; j < num_fg; ++j) {
                    int64_t label_scalar = fg_labels_ptr[j] - 1;  // 0-indexed
                    label_scalar = std::max(int64_t(0),
                                            std::min(label_scalar, num_classes_ - 1));
                    gather_index_ptr[j * 4 + 0] = label_scalar;
                    gather_index_ptr[j * 4 + 1] = label_scalar;
                    gather_index_ptr[j * 4 + 2] = label_scalar;
                    gather_index_ptr[j * 4 + 3] = label_scalar;
                }
                // Move the index to the deltas' device for the gather.
                auto gather_index_dev = gather_index.to(fg_box_deltas_reshaped.device());

                // gather along dim 1 -> (num_fg, 1, 4), then reshape to (num_fg, 4).
                auto fg_selected_deltas = tenzor::reshape(
                    tenzor::gather(fg_box_deltas_reshaped, 1, gather_index_dev),
                    {num_fg, 4}
                );

                SmoothL1Loss smooth_l1;
                auto reg_loss = smooth_l1(
                    fg_selected_deltas,
                    Variable(target_deltas, false)
                );

                // Accumulate losses
                loss_classifier_ = loss_classifier_ + cls_loss;
                loss_box_reg_ = loss_box_reg_ + reg_loss;
            } else {
                // No foreground samples - create zero regression loss
                auto zero_reg_loss = Variable(ops::zeros({1}, features.dtype(),
                                                         features.device()), false);
                loss_classifier_ = loss_classifier_ + cls_loss;
                loss_box_reg_ = loss_box_reg_ + zero_reg_loss;
            }
        }

        // Inference: postprocess to get final detections
        auto detections = postprocess_detections(
            class_logits.tensor(),
            box_deltas.tensor(),
            img_proposals,
            image_shapes[i]
        );

        all_detections.push_back(detections);
    }

    // Average losses over batch
    if (is_training() && gt_boxes != nullptr && !proposals.empty()) {
        loss_classifier_ = loss_classifier_ / static_cast<double>(proposals.size());
        loss_box_reg_ = loss_box_reg_ / static_cast<double>(proposals.size());
    }

    return all_detections;
}

auto RoIHead::get_losses() const
    -> std::unordered_map<std::string, Variable> {

    std::unordered_map<std::string, Variable> losses;
    losses["loss_classifier"] = loss_classifier_;
    losses["loss_box_reg"] = loss_box_reg_;
    return losses;
}

} // namespace detection
} // namespace nn
} // namespace tenzor
