/**
 * @file mask_rcnn.cpp
 * @brief Implementation of Mask R-CNN for instance segmentation
 */

#include "tenzor/models/mask_rcnn.hpp"
#include "tenzor/models/hub.hpp"
#include "tenzor/nn/checkpoint.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/init.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/detection.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/vision.hpp"
#include "tenzor/ops/advanced.hpp"  // topk for the per-image detection cap
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/nn/functional.hpp"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <random>

namespace tenzor {
namespace models {

// ============================================================================
// Helper Functions for Loss Computation
// ============================================================================

namespace {

/**
 * @brief Encode boxes relative to anchors for regression targets.
 *
 * Converts (x1, y1, x2, y2) to (dx, dy, dw, dh) relative to anchors.
 */
auto encode_boxes(const Tensor& boxes, const Tensor& anchors) -> Tensor {
    // boxes: (N, 4) as (x1, y1, x2, y2)
    // anchors: (N, 4) as (x1, y1, x2, y2)

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

    // Clamp deltas before use. dw/dh before exp (bbox_xform_clip =
    // log(1000/16), matches torchvision) to prevent exp overflow. dx/dy to
    // the same bound so the centre shift is capped at ~4 anchor extents:
    // untrained heads can emit O(100) deltas, and dx*anchor_w would exceed
    // Float16's 65504 max and decode to +-inf (finite-but-huge on Float32).
    const double bbox_xform_clip = std::log(1000.0 / 16.0);
    auto boxes_cx = tenzor::clamp(dx, -bbox_xform_clip, bbox_xform_clip) * anchors_w + anchors_cx;
    auto boxes_cy = tenzor::clamp(dy, -bbox_xform_clip, bbox_xform_clip) * anchors_h + anchors_cy;
    auto boxes_w = tenzor::exp(tenzor::clamp(dw, -bbox_xform_clip, bbox_xform_clip)) * anchors_w;
    auto boxes_h = tenzor::exp(tenzor::clamp(dh, -bbox_xform_clip, bbox_xform_clip)) * anchors_h;

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

    // Pull the full IoU matrix to CPU (Float32) so we can do both the per-anchor
    // thresholding and the per-GT allow-low-quality matching from one buffer.
    auto iou_cpu = iou_matrix.to(Device::cpu()).to(DType::Float32);
    const float* iou_data = iou_cpu.data<const float>();

    auto labels_cpu = Tensor({N}, DType::Int64, Device::cpu());
    auto* labels_data = labels_cpu.data<int64_t>();
    auto matched_idx_cpu = Tensor({N}, DType::Int64, Device::cpu());
    auto* matched_idx_data = matched_idx_cpu.data<int64_t>();

    // For each anchor, find its best matching GT box and label it by IoU.
    for (int64_t i = 0; i < N; ++i) {
        int64_t best_m = 0;
        float best = iou_data[i * M + 0];
        for (int64_t m = 1; m < M; ++m) {
            float v = iou_data[i * M + m];
            if (v > best) { best = v; best_m = m; }
        }
        matched_idx_data[i] = best_m;
        if (best >= pos_threshold) {
            labels_data[i] = 1;   // Positive
        } else if (best < neg_threshold) {
            labels_data[i] = 0;   // Negative
        } else {
            labels_data[i] = -1;  // Ignore (between thresholds)
        }
    }

    // Allow-low-quality matching: guarantee every GT has at least one positive
    // anchor. For each GT, find its highest IoU over all anchors, then force
    // every anchor that achieves that maximum to positive and match it to that
    // GT. This rescues GTs whose best anchor still falls below pos_threshold
    // (which would otherwise provide no positive training signal). Mirrors
    // torchvision's Matcher(allow_low_quality_matches=True).
    for (int64_t m = 0; m < M; ++m) {
        float gt_best = iou_data[0 * M + m];
        for (int64_t i = 1; i < N; ++i) {
            float v = iou_data[i * M + m];
            if (v > gt_best) gt_best = v;
        }
        if (gt_best <= 0.0f) continue;  // GT with no anchor overlap at all
        for (int64_t i = 0; i < N; ++i) {
            if (iou_data[i * M + m] == gt_best) {
                labels_data[i] = 1;
                matched_idx_data[i] = m;
            }
        }
    }

    // Move labels to target device
    auto labels = labels_cpu.to(device);

    // Gather matched GT boxes using the (possibly low-quality-corrected) matches.
    auto matched_idx = matched_idx_cpu.to(device);
    auto matched_boxes = tenzor::index_select(gt_boxes, 0, matched_idx);

    return std::make_tuple(labels, matched_boxes);
}

} // anonymous namespace

// Forward declarations for loss computation functions
//
// Autograd fix: the differentiable head outputs are passed as Variables (NOT
// raw Tensors) so the grad_fn chain from each loss back through the RPN / ROI /
// mask heads and the backbone stays intact. Passing `.tensor()` here, or
// re-wrapping inside the helpers with a fresh `Variable(t, true)`, severs the
// graph and backward() then produces zero gradients for those parameters.
auto compute_rpn_loss(
    const Variable& objectness_logits,
    const Variable& box_regression,
    const std::vector<Tensor>& targets,
    const Tensor& anchors
) -> std::tuple<Variable, Variable>;

auto compute_roi_head_loss(
    const Variable& class_logits,
    const Variable& box_regression,
    const std::vector<Tensor>& targets
) -> std::tuple<Variable, Variable>;

auto compute_mask_loss(
    const Variable& mask_logits,
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

    // torchvision RPNHead init: all convs normal(std=0.01), bias 0. With the
    // default (Kaiming) Conv2d init an UNtrained RPN emits O(1) box deltas, so
    // decoded proposals explode far off-image and clip to degenerate edge boxes,
    // collapsing the proposal set. Small-std init keeps untrained proposals
    // close to the anchors (in-bounds). Standard for training too.
    auto init_conv = [](const std::shared_ptr<nn::Conv2d>& m) {
        ::tenzor::nn::init::normal_(m->get_parameter("weight")->tensor(), 0.0, 0.01);
        if (auto b = m->get_parameter("bias")) ::tenzor::nn::init::zeros_(b->tensor());
    };
    init_conv(conv_);
    init_conv(cls_logits_);
    init_conv(bbox_pred_);
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

    // torchvision FastRCNNPredictor init: cls_score normal(std=0.01),
    // bbox_pred normal(std=0.001), biases 0. With the default (Kaiming) Linear
    // init an UNtrained predictor emits O(1) box deltas, so decode_boxes pushes
    // box centres tens of thousands of pixels off-image; after clipping the box
    // collapses to a degenerate edge and process_masks skips it -> all-zero
    // masks. Small-std init keeps untrained boxes ~ proposals (in-bounds).
    ::tenzor::nn::init::normal_(cls_score_->weight()->tensor(), 0.0, 0.01);
    ::tenzor::nn::init::normal_(bbox_pred_->weight()->tensor(), 0.0, 0.001);
    if (auto b = cls_score_->bias()) ::tenzor::nn::init::zeros_(b->tensor());
    if (auto b = bbox_pred_->bias()) ::tenzor::nn::init::zeros_(b->tensor());
}

auto ROIHead::forward_multi(const Variable& roi_features)
    -> std::tuple<Variable, Variable> {
    // roi_features: (num_rois, C, roi_size, roi_size)

    // Flatten. Use an explicit trailing dim (not -1): with N==0 the input has 0
    // elements and reshape({0, -1}) cannot infer the -1 ("product of known
    // dimensions is zero"). Computing C*roi_size*roi_size keeps an empty
    // proposal set from crashing here.
    auto rf_shape = roi_features.tensor().shape();
    auto N = rf_shape[0];
    int64_t flat = 1;
    for (size_t i = 1; i < rf_shape.size(); ++i) flat *= rf_shape[i];
    auto x = roi_features.reshape(std::vector<int64_t>{N, flat});

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

    // Audit G6: build a proper FPN encoder rather than a single 2048→256 1x1
    // projection. ResNet (Bottleneck) stage channels are 256, 512, 1024, 2048
    // for C2-C5 respectively. Each stage gets a 1x1 lateral to FPN's 256-channel
    // working width, then a 3x3 smoothing conv applied to each Pᵢ output.
    // extract_features returns P4 (stride 16) so downstream — which is already
    // configured for stride 16 — gets a feature map with top-down high-level
    // semantics from P5 mixed in, instead of a raw C5 projection at stride 32.
    constexpr std::array<int64_t, 4> bottleneck_stage_channels = {256, 512, 1024, 2048};
    for (size_t i = 0; i < 4; ++i) {
        // All four laterals are live: they build the full top-down path
        // (lat2..lat4 + P5 lateral) that feeds P4.
        fpn_lateral_[i] = std::make_shared<nn::Conv2d>(
            bottleneck_stage_channels[i], 256, /*kernel=*/1, /*stride=*/1, /*pad=*/0);
        register_module("fpn_lateral_c" + std::to_string(i + 2), fpn_lateral_[i]);
    }

    // Only P4 (index 2) is returned and smoothed on the forward path today, so
    // only its smoothing conv is created and registered. Registering the
    // P2/P3/P5 smooth convs while they are never executed would add dead
    // trainable parameters that bloat optimizer state and drift from loaded
    // checkpoints. The remaining fpn_smooth_ slots stay null until the
    // multi-scale RoIAlign path is wired (G6-followup); add them back there so
    // the registered parameter set always matches what is trained.
    fpn_smooth_[2] = std::make_shared<nn::Conv2d>(
        /*in=*/256, /*out=*/256, /*kernel=*/3, /*stride=*/1, /*pad=*/1);
    register_module("fpn_smooth_p4", fpn_smooth_[2]);

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

    // Audit G7: pack detection outputs into a single (N, 6) Variable matching
    // the YOLOv5 convention: each row is [x1, y1, x2, y2, score, label]. Old
    // behavior returned only the box coords wrapped as `Variable(boxes, false)`
    // and silently dropped labels, scores, AND masks — a user calling the
    // generic `forward(images)` interface had no way to see the predicted
    // classes/confidences/masks.
    //
    // Masks have per-detection spatial shape (h × w) that does not fit a
    // uniform (N, 6) row, so masks remain accessible via `forward_test` (the
    // 4-tuple entry point) or the new public `detect()` helper which returns
    // all four outputs as a struct.
    auto [boxes, labels, scores, masks] = forward_test(images);
    (void)masks;  // dropped from forward_impl output; see detect() for masks.

    const int64_t N = boxes.shape()[0];
    const DType  dtype  = boxes.dtype();
    const Device device = boxes.device();

    if (N == 0) {
        return Variable(Tensor({0, 6}, dtype, device), false);
    }

    // Cast scores / labels to the box dtype so cat is dtype-uniform; reshape
    // each to a column for column-wise concat.
    Tensor scores_col = scores.to(dtype).reshape({N, 1});
    Tensor labels_col = labels.to(dtype).reshape({N, 1});
    Tensor boxes_2d   = boxes.reshape({N, 4});

    Tensor packed = tenzor::cat({boxes_2d, scores_col, labels_col}, /*dim=*/1);  // (N, 6)
    return Variable(packed, false);
}

auto MaskRCNN::detect(const Variable& images) -> Detections {
    // Convenience accessor for callers that want the full structured output
    // (including masks). Internally this is just forward_test bundled into a
    // struct; gives users a single named-field API instead of a 4-tuple.
    auto [boxes, labels, scores, masks] = forward_test(images);
    return Detections{boxes, labels, scores, masks};
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
        rpn_cls_logits,    // Variable — keeps grad_fn back into the RPN head
        rpn_bbox_deltas,   // Variable — keeps grad_fn back into the RPN head
        rpn_targets,
        anchors
    );

    // 3. Generate proposals from RPN output
    auto proposals = generate_proposals(features);

    // 4. Select training samples (positive and negative ROIs)
    auto sampled_rois = select_training_samples(proposals, gt_boxes, gt_labels);

    // P.6: empty-batch guard. If the proposal sampler returned zero rows
    // (no positives *and* no negatives — typically a degenerate image / no
    // GT overlap), downstream RoIAlign + head forwards trip on shape[0]==0
    // tensors. Bail out early with zero head-losses (RPN losses already
    // computed above are kept) so the training loop just contributes
    // nothing for this batch instead of tearing down the whole step.
    if (sampled_rois.shape().size() == 0 || sampled_rois.shape()[0] == 0) {
        auto device = images.tensor().device();
        auto zero_loss = Variable(tenzor::zeros({}, DType::Float32, device),
                                  /*requires_grad=*/false);
        return {rpn_cls_loss, rpn_bbox_loss, zero_loss, zero_loss, zero_loss};
    }

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

    // Audit G.7: per-image IoU-matched ROI assignment across the batch.
    //
    // sampled_rois has layout (num_sampled, 5) = (batch_idx, x1, y1, x2, y2).
    // gt_boxes has shape (B, max_objects, 4); gt_labels has shape (B, max_objects).
    // Each image's ROIs must be matched against THAT image's GT boxes, never
    // against image 0's. We group sampled rows by their batch_idx column, then
    // run a per-image IoU/argmax/threshold matcher on the same device as the
    // ROI tensor, and finally scatter the results into the global label and
    // target-box buffers at their original sampled positions.
    //
    // Mirrors torchvision.models.detection.roi_heads.RoIHeads.select_training_samples'
    // per-image loop. Per-image IoU/argmax/max stay on `original_device`; only
    // the small (Nb,) result vectors hit the CPU to write into the global
    // CPU-side label/target buffers.
    const int64_t num_gt = gt_boxes.shape()[1];
    const int64_t batch_size_rois = gt_boxes.shape()[0];

    if (num_gt > 0 && num_sampled > 0) {
        // Single CPU copy of the batch_idx column to drive grouping.
        auto batch_col_cpu = sampled_rois.slice(1, 0, 1)
                                          .to(Device::cpu())
                                          .to(DType::Float32);
        const float* bc = batch_col_cpu.data<float>();

        std::vector<std::vector<int64_t>> rows_by_batch(
            static_cast<size_t>(batch_size_rois));
        for (int64_t r = 0; r < num_sampled; ++r) {
            const int64_t b = static_cast<int64_t>(bc[r]);
            if (b >= 0 && b < batch_size_rois) {
                rows_by_batch[static_cast<size_t>(b)].push_back(r);
            }
        }

        auto* target_boxes_data = sampled_target_boxes_cpu.data<float>();

        for (int64_t b = 0; b < batch_size_rois; ++b) {
            const auto& rows = rows_by_batch[static_cast<size_t>(b)];
            if (rows.empty()) continue;

            // Build the index tensor of this image's rows on `original_device`.
            Tensor rows_cpu({static_cast<int64_t>(rows.size())},
                             DType::Int64, Device::cpu());
            std::memcpy(rows_cpu.data<int64_t>(), rows.data(),
                        rows.size() * sizeof(int64_t));
            Tensor rows_idx = rows_cpu.to(original_device);

            // Per-image ROI boxes and GT (stay on device).
            Tensor rois_b = tenzor::ops::index_select(sampled_rois, 0, rows_idx);
            Tensor roi_boxes_b = rois_b.slice(1, 1, 5);          // (Nb, 4)
            Tensor gt_boxes_b  = tenzor::select(gt_boxes, 0, b); // (M, 4)
            Tensor gt_labels_b = tenzor::select(gt_labels, 0, b);// (M,)
            if (gt_boxes_b.dtype() != roi_boxes_b.dtype()) {
                gt_boxes_b = gt_boxes_b.to(roi_boxes_b.dtype());
            }

            // IoU + argmax + max on-device — no CPU fallback.
            Tensor iou_b         = ops::box_iou(roi_boxes_b, gt_boxes_b);  // (Nb, M)
            Tensor matched_gt_b  = tenzor::argmax(iou_b, 1);               // (Nb,)
            Tensor max_iou_b     = tenzor::max(iou_b, 1);                  // (Nb,)

            // Only the small per-image result vectors hit the CPU so we can
            // scatter into the global CPU-side label/target buffers.
            auto matched_idx_cpu = matched_gt_b.to(Device::cpu());
            const int64_t* matched_idx_data = matched_idx_cpu.data<int64_t>();
            auto max_iou_cpu     = max_iou_b.to(Device::cpu()).to(DType::Float32);
            const float* max_iou_data = max_iou_cpu.data<float>();
            auto gt_labels_b_cpu = gt_labels_b.to(Device::cpu()).to(DType::Int64);
            const int64_t* gt_labels_data = gt_labels_b_cpu.data<int64_t>();
            auto gt_boxes_b_cpu  = gt_boxes_b.to(Device::cpu()).to(DType::Float32);
            const float* gt_boxes_data = gt_boxes_b_cpu.data<float>();

            const int64_t Nb = static_cast<int64_t>(rows.size());
            for (int64_t i = 0; i < Nb; ++i) {
                if (max_iou_data[i] >= 0.5f) {  // Positive threshold (fg_iou_thresh)
                    const int64_t gt_idx = matched_idx_data[i];
                    const int64_t global_row = rows[static_cast<size_t>(i)];
                    sampled_labels_data[global_row] = gt_labels_data[gt_idx];
                    for (int j = 0; j < 4; ++j) {
                        target_boxes_data[global_row * 4 + j] =
                            gt_boxes_data[gt_idx * 4 + j];
                    }
                }
            }
        }
    }

    // Move CPU tensors to target device
    auto sampled_labels = sampled_labels_cpu.to(original_device);
    auto sampled_target_boxes = sampled_target_boxes_cpu.to(sampled_rois.dtype()).to(original_device);

    // The ROI box head predicts deltas (dx, dy, dw, dh) relative to each
    // proposal, so the regression targets must be the matched GT boxes ENCODED
    // against their proposals — not raw GT coordinates. Comparing predicted
    // deltas against absolute coordinates makes the Smooth-L1 loss explode
    // (~|coords|, e.g. roi_box≈224 for 800px boxes). Mirrors the RPN target
    // encoding and torchvision's BoxCoder.encode in RoIHeads. Background rows
    // (zero target box) encode to finite junk but are filtered by the
    // foreground mask inside compute_roi_head_loss, so they never reach the loss.
    Tensor roi_proposal_boxes = sampled_rois.slice(1, 1, 5).contiguous();  // (num_sampled, 4)
    Tensor sampled_target_deltas = encode_boxes(sampled_target_boxes, roi_proposal_boxes);

    // Compute ROI head losses
    std::vector<Tensor> roi_targets = {sampled_labels, sampled_target_deltas};
    auto [roi_cls_loss, roi_bbox_loss] = compute_roi_head_loss(
        cls_logits,    // Variable — keeps grad_fn back into the ROI box head
        bbox_deltas,   // Variable — keeps grad_fn back into the ROI box head
        roi_targets
    );

    // 8. Mask branch — run the mask head on POSITIVE ROIs ONLY (standard
    // Mask R-CNN). The mask loss is undefined for background ROIs, so building
    // the full num_sampled (e.g. 1024) ROI feature map + mask-head conv stack
    // when only a handful are positive wasted multiple GB of activations and
    // OOM'd at batch-2 x 800x800. Gathering the positives first keeps the mask
    // head's working set proportional to the number of foreground ROIs.
    std::vector<int64_t> pos_rows;
    pos_rows.reserve(static_cast<size_t>(num_sampled));
    for (int64_t i = 0; i < num_sampled; ++i) {
        if (sampled_labels_data[i] > 0) pos_rows.push_back(i);
    }
    const int64_t num_pos = static_cast<int64_t>(pos_rows.size());

    // Differentiable zero default (no positives -> no mask loss this batch).
    Variable mask_loss_val(tenzor::zeros({}, DType::Float32, original_device),
                           /*requires_grad=*/true);

    if (num_pos > 0) {
        Tensor pos_idx_cpu({num_pos}, DType::Int64, Device::cpu());
        std::memcpy(pos_idx_cpu.data<int64_t>(), pos_rows.data(),
                    pos_rows.size() * sizeof(int64_t));
        Tensor pos_idx = pos_idx_cpu.to(original_device);

        // ROI-align + mask head on positives only.
        Tensor pos_rois = tenzor::ops::index_select(sampled_rois, 0, pos_idx);
        auto roi_features_mask = roi_align_mask_->forward(features, pos_rois);
        auto mask_logits = mask_head_->forward(roi_features_mask);

        const int64_t mask_output_H = mask_logits.tensor().shape()[2];
        const int64_t mask_output_W = mask_logits.tensor().shape()[3];

        // Per-positive GT-mask targets (num_pos, H, W) and labels (num_pos,),
        // aligned row-for-row with pos_rois.
        Tensor pos_masks_cpu({num_pos, mask_output_H, mask_output_W},
                             DType::Float32, Device::cpu());
        pos_masks_cpu.fill_(0.0f);
        Tensor pos_labels_cpu({num_pos}, DType::Int64, Device::cpu());
        auto* pos_labels_data = pos_labels_cpu.data<int64_t>();
        for (int64_t k = 0; k < num_pos; ++k) {
            pos_labels_data[k] = sampled_labels_data[pos_rows[static_cast<size_t>(k)]];
        }

        // Audit G.7: per-image mask sampling. For each foreground ROI, locate
        // the best-matching GT box IN ITS OWN IMAGE (via the sampled_rois
        // batch_idx column), then pull the corresponding GT mask.
        if (num_gt > 0) {
            auto batch_col_cpu_masks = sampled_rois.slice(1, 0, 1)
                                                    .to(Device::cpu())
                                                    .to(DType::Float32);
            const float* bc_m = batch_col_cpu_masks.data<float>();
            const int64_t batch_size_masks = gt_boxes.shape()[0];

            for (int64_t k = 0; k < num_pos; ++k) {
                const int64_t i = pos_rows[static_cast<size_t>(k)];
                const int64_t b = static_cast<int64_t>(bc_m[i]);
                if (b < 0 || b >= batch_size_masks) continue;

                // ROI box on its native device; GT for THIS image.
                auto roi_box = sampled_rois.slice(0, i, i + 1).slice(1, 1, 5);  // (1, 4)
                auto gt_boxes_b = tenzor::select(gt_boxes, 0, b);              // (num_gt, 4)
                if (gt_boxes_b.dtype() != roi_box.dtype()) {
                    gt_boxes_b = gt_boxes_b.to(roi_box.dtype());
                }

                auto iou = ops::box_iou(roi_box, gt_boxes_b);                  // (1, num_gt)
                auto best_gt_idx = tenzor::argmax(iou, 1).to(Device::cpu()).item<int64_t>();
                if (best_gt_idx < 0 || best_gt_idx >= num_gt) continue;

                auto gt_masks_b = tenzor::select(gt_masks, 0, b);              // (num_gt, H, W)
                auto gt_mask = tenzor::select(gt_masks_b, 0, best_gt_idx);     // (H, W)

                // Resize GT mask to (mask_output_H, mask_output_W).
                auto resized_mask = ops::interpolate(
                    gt_mask.unsqueeze(0).unsqueeze(0),  // (1, 1, H, W)
                    std::vector<int64_t>{mask_output_H, mask_output_W},
                    "bilinear",
                    false
                );
                resized_mask = resized_mask.squeeze(0).squeeze(0)
                                           .to(DType::Float32).to(Device::cpu());

                auto target_mask_cpu = tenzor::select(pos_masks_cpu, 0, k);
                auto* resized_data = resized_mask.data<float>();
                auto* target_data  = target_mask_cpu.data<float>();
                std::copy(resized_data,
                          resized_data + mask_output_H * mask_output_W,
                          target_data);
            }
        }

        // Compute mask loss on positives.
        auto pos_masks = pos_masks_cpu.to(mask_logits.tensor().dtype()).to(original_device);
        auto pos_labels = pos_labels_cpu.to(original_device);
        std::vector<Tensor> mask_targets = {pos_masks, pos_labels};
        // mask_logits is a Variable carrying grad_fn back through the mask head;
        // compute_mask_loss forwards it straight into nn::detection::mask_loss
        // (already autograd-aware via gather+squeeze) without re-severing.
        mask_loss_val = compute_mask_loss(mask_logits, mask_targets);
    }

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

    // 2. Generate proposals (generate_proposals runs the RPN head internally,
    //    so there is no separate rpn_->forward_multi call here — running it
    //    twice per inference would duplicate the 3x3 + two 1x1 conv work).
    auto proposals = generate_proposals(features);

    // 3. ROI Align for boxes
    auto roi_features_box = roi_align_box_->forward(features, proposals);

    // 4. Box classification and regression
    auto [cls_logits, bbox_deltas] = roi_head_->forward_multi(roi_features_box);

    // 5. Apply NMS and get final detections
    auto cls_probs = tenzor::softmax(cls_logits, 1).tensor();

    // For simplicity, take top predictions
    // In production, this would apply proper NMS per class.
    //
    // cls_probs has shape (N, num_classes + 1) — column 0 is the
    // background class. mask_logits has shape (N, num_classes, H, W) —
    // one channel per real class, no background. process_masks/paste_masks
    // index mask_logits directly with the per-detection label (no -1
    // remap), so labels passed in must be in [0, num_classes). Argmax-ing
    // over `cls_probs[:, 1:]` skips the background column and yields a
    // 0-based index over real classes, exactly what mask_logits expects.
    // Without this slice, when the background column happens not to win
    // and class num_classes does (a normal occurrence on Vulkan Float64
    // for the RPN multi-scale test, just not on CPU due to small numeric
    // drift in softmax), `select(mask_logits, 0, num_classes)` raises
    // "Index out of range for select" because the 91-channel mask_logits
    // has no channel 91.
    auto cls_probs_no_bg = cls_probs.slice(1, 1, num_classes_ + 1);
    auto scores = tenzor::max(cls_probs_no_bg, 1);
    auto labels = tenzor::argmax(cls_probs_no_bg, 1);

    // Decode predicted box deltas relative to proposal anchors. Decode in
    // Float32 for half dtypes: even with clamped deltas, Float16 loses
    // mantissa on the dx*anchor_w products; decode_boxes itself clamps dx/dy
    // and dw/dh so the result stays inside half range before narrowing.
    auto proposal_boxes = proposals.slice(1, 1, 5);  // Remove batch index
    const DType box_dtype = bbox_deltas.tensor().dtype();
    const bool half_boxes =
        (box_dtype == DType::Float16 || box_dtype == DType::BFloat16);

    // bbox_pred_ emits class-specific deltas of shape (N, (num_classes+1)*4).
    // Select the (N, 4) deltas belonging to each detection's predicted class
    // before decoding — otherwise every box is decoded with the background
    // class deltas (columns 0..3) regardless of the predicted label, producing
    // systematically wrong inference boxes. `labels` is 0-based over the real
    // classes (background column was sliced off above), so the column block in
    // the (num_classes+1) layout is `label + 1`.
    Tensor deltas_raw = half_boxes ? bbox_deltas.tensor().to(DType::Float32)
                                   : bbox_deltas.tensor();
    const int64_t num_boxes = deltas_raw.shape()[0];
    auto deltas_per_class =
        tenzor::reshape(deltas_raw, {num_boxes, num_classes_ + 1, 4});  // (N, C+1, 4)

    // gather along the class dim with (label + 1), broadcast over the 4 coords.
    // `labels` is Int64 (from argmax); keep the index Int64 for gather.
    auto class_idx = tenzor::add(labels.to(DType::Int64), 1.0).to(DType::Int64);  // (N,)
    Tensor gather_idx = tenzor::reshape(class_idx, {num_boxes, 1, 1});
    gather_idx = tenzor::expand(gather_idx, {num_boxes, 1, 4}).contiguous();
    auto gathered_deltas = tenzor::gather(deltas_per_class, /*dim=*/1, gather_idx);
    Tensor deltas_t = tenzor::reshape(gathered_deltas, {num_boxes, 4});  // (N, 4)

    Tensor props_t = half_boxes ? proposal_boxes.to(DType::Float32)
                                : proposal_boxes;
    auto boxes = decode_boxes(deltas_t, props_t);
    if (half_boxes) boxes = boxes.to(box_dtype);

    // 6. Post-process detections: score-threshold + per-class NMS + top-k cap.
    // Previously forward_test returned every proposal's argmax box with no
    // filtering at all. torchvision applies box_score_thresh, then per-class NMS
    // at box_nms_thresh, then keeps the top box_detections_per_img detections.
    //
    // `boxes` are class-specific (decoded for each detection's argmax label) and
    // `labels`/`scores` are that label and its probability. Build a sparse
    // (N, num_classes) score matrix that holds the detection score at its
    // predicted class and 0 elsewhere, so batched_nms lets each box compete only
    // within its own class (consistent with the single decoded box geometry).
    Tensor scores_f32 = scores.to(DType::Float32);
    Tensor onehot = tenzor::one_hot(labels.to(DType::Int64), num_classes_).to(DType::Float32);
    Tensor score_matrix = onehot * tenzor::reshape(scores_f32, {num_boxes, 1});  // (N, num_classes)

    Tensor boxes_f32 = (boxes.dtype() == DType::Float32) ? boxes : boxes.to(DType::Float32);
    auto [det_boxes, det_scores, det_labels] = tenzor::ops::batched_nms(
        boxes_f32, score_matrix, box_nms_thresh_, box_score_thresh_,
        box_detections_per_img_);

    // batched_nms caps boxes per class; enforce the global per-image cap by
    // keeping only the top box_detections_per_img_ detections overall.
    if (det_scores.shape()[0] > box_detections_per_img_) {
        auto [top_scores, top_idx] =
            tenzor::topk(det_scores, box_detections_per_img_, 0, true, true);
        det_boxes = tenzor::index_select(det_boxes, 0, top_idx);
        det_labels = tenzor::index_select(det_labels, 0, top_idx);
        det_scores = top_scores;
    }

    const int64_t num_dets = det_boxes.shape()[0];

    // No detections survived score-threshold + NMS (common with untrained
    // weights). Running ROI-align / the mask head on an empty ROI set trips a
    // reshape(-1) over a zero element count; return empty results directly,
    // matching process_masks' [0, 1, H, W] empty-mask convention.
    if (num_dets == 0) {
        auto img_h0 = images.tensor().shape()[2];
        auto img_w0 = images.tensor().shape()[3];
        Tensor empty_masks(std::vector<int64_t>{0, 1, img_h0, img_w0},
                           images.tensor().dtype(), det_boxes.device());
        return std::make_tuple(det_boxes, det_labels, det_scores, empty_masks);
    }

    // 7. ROI Align for masks — align on the REFINED, post-NMS detection boxes
    //    (with a leading batch-index column), NOT the raw proposals. The mask
    //    head output is pasted onto these refined boxes, so aligning it from the
    //    raw proposals (the previous behavior) misplaced every predicted mask.
    Tensor batch_index = tenzor::zeros({num_dets, 1}, det_boxes.dtype(), det_boxes.device());
    Tensor mask_rois = tenzor::cat({batch_index, det_boxes}, 1);  // (num_dets, 5)
    if (mask_rois.dtype() != proposals.dtype()) {
        mask_rois = mask_rois.to(proposals.dtype());
    }
    auto roi_features_mask = roi_align_mask_->forward(features, mask_rois);

    // 8. Mask prediction
    auto mask_logits = mask_head_->forward(roi_features_mask);

    // 9. Post-process masks onto the refined detection boxes.
    auto image_h = images.tensor().shape()[2];
    auto image_w = images.tensor().shape()[3];
    auto masks = nn::detection::process_masks(
        mask_logits.tensor(),
        det_boxes,
        det_labels,
        image_h,
        image_w
    );

    return std::make_tuple(det_boxes, det_labels, det_scores, masks);
}

auto compute_rpn_loss(
    const Variable& objectness_logits,
    const Variable& box_regression,
    const std::vector<Tensor>& targets,
    const Tensor& anchors
) -> std::tuple<Variable, Variable> {
    // objectness_logits: (N, num_anchors*H*W, 2)  -- Variable (RPN cls head)
    // box_regression: (N, num_anchors*H*W, 4)     -- Variable (RPN box head)
    // targets: vector of {boxes, labels} per image
    // anchors: (num_anchors*H*W, 4)

    auto batch_size = objectness_logits.tensor().shape()[0];
    auto device = objectness_logits.tensor().device();

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

        // Get objectness and bbox predictions for this image.
        // Variable-level select-along-dim-0 = index_select(1 row) + squeeze, so
        // the grad_fn back into the RPN head is preserved (a raw tenzor::select
        // on .tensor() would sever it).
        Tensor batch_idx_b = Tensor({1}, DType::Int64, Device::cpu()).fill_(
            static_cast<double>(b)).to(device);
        auto obj_logits_b = tenzor::squeeze(
            tenzor::index_select(objectness_logits, 0, batch_idx_b), 0);  // (num_anchors, 2)
        auto bbox_reg_b = tenzor::squeeze(
            tenzor::index_select(box_regression, 0, batch_idx_b), 0);     // (num_anchors, 4)

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

        // Select sampled objectness logits and labels.
        // obj_logits_b is a Variable, so this resolves to the autograd
        // index_select overload and keeps the grad_fn chain. Labels are plain
        // (non-differentiable) targets and stay a Tensor.
        Variable sampled_logits = tenzor::index_select(obj_logits_b, 0, indices_tensor);
        auto sampled_labels_full = tenzor::index_select(labels, 0, indices_tensor);

        // Compute classification loss: cross-entropy. Pass the Variable
        // directly — wrapping it in a fresh Variable(..., true) would discard
        // grad_fn and zero out RPN-head gradients.
        nn::CrossEntropyLoss ce_loss;
        auto cls_loss_var = ce_loss.forward(
            sampled_logits,
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

            // Get predicted deltas for positive anchors (bbox_reg_b is a
            // Variable -> autograd index_select preserves grad_fn).
            Variable pos_bbox_pred = tenzor::index_select(bbox_reg_b, 0, pos_indices_tensor);

            // Get matched GT boxes for positive anchors (targets — plain Tensors)
            auto pos_matched_boxes = tenzor::index_select(matched_boxes, 0, pos_indices_tensor);
            // Use anchors_typed (already cast to gt_boxes.dtype()) so encode_boxes
            // runs on matching dtypes — pos_matched_boxes carry gt_boxes' dtype,
            // while the original anchors param is always Float32.
            auto pos_anchors = tenzor::index_select(anchors_typed, 0, pos_indices_tensor);

            // Encode GT boxes as regression targets
            auto bbox_targets = encode_boxes(pos_matched_boxes, pos_anchors);

            // Compute Smooth L1 loss. Pass the prediction Variable directly so
            // the grad_fn back into the RPN box head survives; only the target
            // is a fresh non-differentiable Variable.
            nn::SmoothL1Loss smooth_l1(nn::Reduction::Mean, 1.0);
            auto bbox_loss_var = smooth_l1.forward(
                pos_bbox_pred,
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
        cls_loss_final = Variable(Tensor({}, objectness_logits.tensor().dtype(), device).fill_(0.0), true);
    }

    if (!bbox_losses.empty()) {
        bbox_loss_final = bbox_losses[0];
        for (size_t i = 1; i < bbox_losses.size(); ++i) {
            bbox_loss_final = bbox_loss_final + bbox_losses[i];
        }
        bbox_loss_final = bbox_loss_final / static_cast<float>(bbox_losses.size());
    } else {
        // Even with no positive samples, return differentiable zero for gradient flow
        bbox_loss_final = Variable(Tensor({}, objectness_logits.tensor().dtype(), device).fill_(0.0), true);
    }

    return std::make_tuple(cls_loss_final, bbox_loss_final);
}

auto compute_roi_head_loss(
    const Variable& class_logits,
    const Variable& box_regression,
    const std::vector<Tensor>& targets
) -> std::tuple<Variable, Variable> {
    // class_logits: (num_rois, num_classes+1)        -- Variable (ROI cls head)
    // box_regression: (num_rois, (num_classes+1)*4)  -- Variable (ROI box head)
    // targets: vector containing {labels, target_boxes} for sampled ROIs

    auto num_rois = class_logits.tensor().shape()[0];
    auto device = class_logits.tensor().device();

    if (num_rois == 0 || targets.empty()) {
        // Return differentiable zero for gradient flow
        auto zero_loss = Variable(Tensor({}, class_logits.tensor().dtype(), device).fill_(0.0), true);
        return std::make_tuple(zero_loss, zero_loss);
    }

    // For simplicity, assume targets[0] is labels, targets[1] is target boxes
    // In a full implementation, this would come from ROI sampling
    auto roi_labels = targets[0];  // (num_rois,)
    auto roi_target_boxes = targets.size() > 1 ? targets[1] : Tensor({num_rois, 4}, box_regression.tensor().dtype(), device);

    // Classification loss: multi-class cross-entropy. Pass the class_logits
    // Variable directly so grad_fn back into the ROI cls head is preserved
    // (a fresh Variable(class_logits, true) would sever it).
    nn::CrossEntropyLoss ce_loss;
    auto cls_loss = ce_loss.forward(
        class_logits,
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

        // Get foreground box predictions and targets. box_regression is a
        // Variable -> autograd index_select keeps grad_fn into the ROI box head.
        Variable fg_box_regression = tenzor::index_select(box_regression, 0, fg_idx_tensor);
        auto fg_target_boxes = tenzor::index_select(roi_target_boxes, 0, fg_idx_tensor);
        auto fg_labels = tenzor::index_select(roi_labels, 0, fg_idx_tensor);
        auto fg_labels_cpu = fg_labels.to(Device::cpu());

        // Extract box deltas for the predicted class, staying on the autograd
        // graph: each row is sliced with the Variable-level autograd slice and
        // the per-row (1, 4) deltas are concatenated with autograd cat. The
        // earlier tenzor::select + Tensor.slice + ops::stack path operated on
        // raw tensors and severed the chain.
        auto num_fg = fg_box_regression.tensor().shape()[0];
        std::vector<Variable> selected_deltas;
        selected_deltas.reserve(static_cast<size_t>(num_fg));

        for (int64_t i = 0; i < num_fg; ++i) {
            // fg_row: (1, (num_classes+1)*4)
            Variable fg_row = tenzor::slice(fg_box_regression, 0, i, i + 1);
            int64_t class_idx = fg_labels_cpu.data<int64_t>()[i];

            // Select 4 values for this class -> (1, 4)
            auto start_idx = class_idx * 4;
            Variable delta_slice = tenzor::slice(fg_row, 1, start_idx, start_idx + 4);
            selected_deltas.push_back(delta_slice);
        }

        Variable fg_selected_deltas = tenzor::cat(selected_deltas, 0);  // (num_fg, 4)

        // Compute Smooth L1 loss. Pass the prediction Variable directly so the
        // grad_fn back into the ROI box head survives.
        nn::SmoothL1Loss smooth_l1(nn::Reduction::Mean, 1.0);
        bbox_loss = smooth_l1.forward(
            fg_selected_deltas,
            Variable(fg_target_boxes, false)
        );
    } else {
        // Return differentiable zero for gradient flow
        bbox_loss = Variable(Tensor({}, class_logits.tensor().dtype(), device).fill_(0.0), true);
    }

    return std::make_tuple(cls_loss, bbox_loss);
}

auto compute_mask_loss(
    const Variable& mask_logits,
    const std::vector<Tensor>& targets
) -> Variable {
    // This is a wrapper around the existing nn::detection::mask_loss
    // mask_logits: (num_rois, num_classes, H, W)  -- Variable (mask head)
    // targets: vector containing {mask_targets, class_labels}

    auto num_rois = mask_logits.tensor().shape()[0];
    auto device = mask_logits.tensor().device();

    if (num_rois == 0 || targets.size() < 2) {
        // Return differentiable zero for gradient flow
        return Variable(Tensor({}, mask_logits.tensor().dtype(), device).fill_(0.0), true);
    }

    auto mask_targets = targets[0];  // (num_rois, H, W)
    auto class_labels = targets[1];  // (num_rois,)

    // Use the existing mask_loss function from mask_head. It is already
    // autograd-aware (gather + squeeze), so forwarding the Variable directly —
    // rather than re-wrapping with Variable(mask_logits, true) — keeps the
    // grad_fn chain back through the mask head and backbone intact.
    auto loss = nn::detection::mask_loss(
        mask_logits,
        mask_targets,
        class_labels
    );

    return loss;
}

// ============================================================================
// MaskRCNN Implementation
// ============================================================================

auto MaskRCNN::extract_features(const Variable& images) -> Variable {
    // Audit G6: real FPN encoder.
    //   C2 (stride 4) ─┐
    //   C3 (stride 8) ─┼─→ lateral 1x1 ─→ add(upsample(P_{i+1})) ─→ smooth 3x3 ─→ Pᵢ
    //   C4 (stride 16)─┤
    //   C5 (stride 32)─┘
    // Returns P4 (stride 16, 256 ch) so downstream sees a feature map at the
    // stride the rest of the pipeline is already configured for. Top-down
    // information from P5 flows into P4 via the additive upsample step.
    auto resnet = std::dynamic_pointer_cast<ResNet>(backbone_);
    if (!resnet) {
        throw std::runtime_error("Mask R-CNN requires ResNet backbone");
    }

    auto [c2, c3, c4, c5] = resnet->forward_features_multi(images);

    // Top-down. Lateral conv first, then add the upsampled higher-level
    // pyramid. Upsample target size comes from the lateral's spatial shape,
    // since strides are factors of two but rounding may produce off-by-one.
    auto p5 = fpn_lateral_[3]->forward(c5);

    auto lat4 = fpn_lateral_[2]->forward(c4);
    auto p5_up = nn::functional::interpolate(
        p5,
        {lat4.tensor().shape()[2], lat4.tensor().shape()[3]},
        "nearest", /*align_corners=*/false);
    auto p4 = lat4 + p5_up;

    auto lat3 = fpn_lateral_[1]->forward(c3);
    auto p4_up = nn::functional::interpolate(
        p4,
        {lat3.tensor().shape()[2], lat3.tensor().shape()[3]},
        "nearest", /*align_corners=*/false);
    auto p3 = lat3 + p4_up;

    auto lat2 = fpn_lateral_[0]->forward(c2);
    auto p3_up = nn::functional::interpolate(
        p3,
        {lat2.tensor().shape()[2], lat2.tensor().shape()[3]},
        "nearest", /*align_corners=*/false);
    auto p2 = lat2 + p3_up;

    // Smoothing 3x3 — anti-aliases the upsample. Only P4 is returned today, so
    // only fpn_smooth_[2] is constructed (the other slots are null — they are
    // created together with the multi-scale path in G6-followup: per-level RPN +
    // FPN-style level assignment for ROI features). Until then p2/p3/p5 are only
    // intermediate top-down terms feeding p4.
    (void)p2; (void)p3; (void)p5;  // intentionally unused — see G6-followup
    p4 = fpn_smooth_[2]->forward(p4);

    return p4;
}

auto MaskRCNN::generate_proposals(const Variable& features) -> Tensor {
    // Audit G4: real RPN proposals via decode_boxes → clip → small-filter →
    // top-K by score → NMS → top-K. Replaces the previous `proposals.fill_(0)`
    // dummy that made inference output meaningless.
    //
    // RPN::forward_multi already produces anchor-major flat tensors:
    //   cls_logits:  (B, H*W*A, 2)   — col 1 is foreground logit
    //   bbox_deltas: (B, H*W*A, 4)
    // Flat index in dim 1 is `y * W * A + x * A + a`, matching
    // anchor_generator_->generate(H, W, stride, device) → (H*W*A, 4).
    const auto& shape = features.tensor().shape();
    const int64_t batch_size = shape[0];
    const int64_t H = shape[2];
    const int64_t W = shape[3];
    const Device device = features.tensor().device();
    const DType dtype = features.tensor().dtype();
    const int64_t stride = 16;  // Tenzor's Mask R-CNN uses C4 features.

    // 1. RPN forward — outputs already anchor-major flat.
    auto [cls_logits_var, bbox_deltas_var] = rpn_->forward_multi(features);
    Tensor cls_logits = cls_logits_var.tensor();    // (B, HWA, 2)
    Tensor bbox_deltas = bbox_deltas_var.tensor();  // (B, HWA, 4)
    const int64_t HWA = cls_logits.shape()[1];

    // 2. Anchors at this stride. Layout: (y, x, a) flattened — matches the
    //    permute+reshape RPN::forward_multi performs upstream.
    Tensor anchors = anchor_generator_->generate(H, W, stride, device);
    if (anchors.dtype() != dtype) anchors = anchors.to(dtype);

    // Image bounds for clipping (saturating, not rejecting).
    const int64_t image_h = H * stride;
    const int64_t image_w = W * stride;

    const int64_t pre_n  = is_training() ? rpn_pre_nms_top_n_train_  : rpn_pre_nms_top_n_test_;
    const int64_t post_n = is_training() ? rpn_post_nms_top_n_train_ : rpn_post_nms_top_n_test_;

    std::vector<Tensor> per_image_proposals;
    per_image_proposals.reserve(static_cast<size_t>(batch_size));

    for (int64_t b = 0; b < batch_size; ++b) {
        // 3. Per-image objectness: take fg-logit (col 1).
        Tensor cls_b = tenzor::ops::select(cls_logits, 0, b);             // (HWA, 2)
        Tensor scores = cls_b.slice(1, 1, 2).reshape({HWA}).contiguous(); // (HWA,)

        // 4. Per-image bbox deltas.
        Tensor reg_b = tenzor::ops::select(bbox_deltas, 0, b).contiguous(); // (HWA, 4)

        // 5. Decode anchor + delta → absolute boxes, clip to image.
        Tensor proposals = tenzor::ops::decode_boxes(reg_b, anchors);
        proposals = tenzor::ops::clip_boxes_to_image(proposals, image_h, image_w);

        // 6. Drop too-small boxes.
        Tensor keep = tenzor::ops::remove_small_boxes(proposals, scores, /*min_size=*/0.001);
        if (keep.numel() == 0) {
            // Every decoded box collapsed to zero area after clipping. This
            // happens when the box centres are driven far off-image — e.g. an
            // untrained deep backbone in eval mode, whose unnormalised features
            // explode over many random layers and produce enormous RPN deltas.
            // Yielding zero proposals would leave the detector with nothing to
            // score (and previously crashed the RoI head's `reshape({N,-1})`);
            // fall back to the top proposals by objectness so downstream still
            // receives candidates. A trained model never reaches this branch.
            keep = tenzor::ops::argsort(scores, 0, /*descending=*/true);
            if (keep.shape()[0] > post_n) {
                keep = keep.slice(0, 0, post_n);
            }
        }
        proposals = tenzor::ops::index_select(proposals, 0, keep);
        scores    = tenzor::ops::index_select(scores,    0, keep);

        // 7. Top pre_nms_top_n by score.
        Tensor sorted_idx = tenzor::ops::argsort(scores, 0, /*descending=*/true);
        if (sorted_idx.shape()[0] > pre_n) {
            sorted_idx = sorted_idx.slice(0, 0, pre_n);
        }
        proposals = tenzor::ops::index_select(proposals, 0, sorted_idx);
        scores    = tenzor::ops::index_select(scores,    0, sorted_idx);

        // 8. NMS → top post_nms_top_n.
        Tensor keep_nms = tenzor::ops::nms(proposals, scores, rpn_nms_thresh_);
        if (keep_nms.shape()[0] > post_n) {
            keep_nms = keep_nms.slice(0, 0, post_n);
        }
        proposals = tenzor::ops::index_select(proposals, 0, keep_nms);

        // 9. Prepend batch_idx column → shape (num_kept, 5).
        const int64_t num_kept = proposals.shape()[0];
        Tensor batch_col = tenzor::ops::full({num_kept, 1}, static_cast<double>(b), dtype, device);
        Tensor with_batch = tenzor::ops::cat({batch_col, proposals}, 1);
        per_image_proposals.push_back(with_batch);
    }

    if (per_image_proposals.empty()) {
        return Tensor({0, 5}, dtype, device);
    }
    return tenzor::ops::cat(per_image_proposals, 0);
}

auto MaskRCNN::select_training_samples(const Tensor& proposals,
                                        const Tensor& gt_boxes,
                                        [[maybe_unused]] const Tensor& gt_labels) -> Tensor {
    // Audit G5: real positive/negative IoU-based sampling.
    //
    // Replaces the previous `return proposals;` no-op. Hyperparameters mirror
    // torchvision's RoIHeads defaults (num_samples=512, positive_fraction=0.25,
    // fg_iou_thresh=0.5). A proposal is positive iff its max IoU with any GT
    // box in the same image is >= fg_iou_thresh, negative otherwise. Within
    // each image we randomly sample up to `num_positive` positives and fill
    // the remainder up to `num_samples` with negatives. When either pool is
    // smaller than its budget we take all that exist (matching torchvision).
    //
    // Per-image work: proposals carry (batch_idx, x1, y1, x2, y2) in col 0,
    // so we group rows by col 0 once and run independent sampling per image.
    constexpr int64_t num_samples       = 512;
    constexpr double  positive_fraction = 0.25;
    constexpr double  fg_iou_thresh     = 0.5;
    constexpr int64_t num_positive      = static_cast<int64_t>(num_samples * positive_fraction);

    const Device device     = proposals.device();
    const DType  dtype      = proposals.dtype();
    const int64_t batch_size = gt_boxes.shape()[0];
    const int64_t num_gt     = gt_boxes.shape()[1];

    // Standard Mask R-CNN `add_gt_proposals`: append the ground-truth boxes to
    // the proposal set so every GT object is guaranteed a positive (IoU=1 with
    // itself) ROI. Without this, a fresh/randomly-initialised RPN frequently
    // yields zero proposals reaching fg_iou_thresh — the box and mask losses
    // then collapse to a no-op (mask_loss==0). torchvision's RoIHeads does the
    // same before sampling; this is correctness, not a backend workaround.
    Tensor proposals_aug = proposals;
    if (num_gt > 0) {
        Tensor gt_cpu = gt_boxes.to(Device::cpu()).to(DType::Float32);  // (B, M, 4)
        const float* gd = gt_cpu.data<float>();
        std::vector<float> gt_rows;  // [batch_idx, x1, y1, x2, y2] per valid GT
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t m = 0; m < num_gt; ++m) {
                const float* bx = gd + (b * num_gt + m) * 4;
                if (bx[2] > bx[0] && bx[3] > bx[1]) {  // skip padding / degenerate
                    gt_rows.push_back(static_cast<float>(b));
                    gt_rows.push_back(bx[0]); gt_rows.push_back(bx[1]);
                    gt_rows.push_back(bx[2]); gt_rows.push_back(bx[3]);
                }
            }
        }
        if (!gt_rows.empty()) {
            const int64_t G = static_cast<int64_t>(gt_rows.size() / 5);
            Tensor gt_props_cpu({G, 5}, DType::Float32, Device::cpu());
            std::memcpy(gt_props_cpu.data<float>(), gt_rows.data(),
                        gt_rows.size() * sizeof(float));
            Tensor gt_props = gt_props_cpu.to(dtype).to(device);
            proposals_aug = (proposals.shape()[0] == 0)
                ? gt_props
                : tenzor::ops::cat({proposals, gt_props}, 0);
        }
    }

    const int64_t N = proposals_aug.shape()[0];
    if (N == 0) return proposals_aug;

    // Group proposal row indices by batch_idx (col 0). One CPU copy.
    Tensor batch_col_cpu = proposals_aug.slice(1, 0, 1).to(Device::cpu()).to(DType::Float32);
    const float* bc = batch_col_cpu.data<float>();
    std::vector<std::vector<int64_t>> rows_by_batch(static_cast<size_t>(batch_size));
    for (int64_t r = 0; r < N; ++r) {
        const int64_t b = static_cast<int64_t>(bc[r]);
        if (b >= 0 && b < batch_size) rows_by_batch[static_cast<size_t>(b)].push_back(r);
    }

    std::vector<int64_t> sampled_global;
    sampled_global.reserve(static_cast<size_t>(num_samples) * static_cast<size_t>(batch_size));

    // Stable per-call rng — deterministic across runs but rotates per image.
    std::mt19937 rng(0xC0FFEE);

    for (int64_t b = 0; b < batch_size; ++b) {
        const auto& rows = rows_by_batch[static_cast<size_t>(b)];
        if (rows.empty()) continue;

        // Build the index tensor of this image's proposal rows on `device`.
        Tensor rows_cpu({static_cast<int64_t>(rows.size())}, DType::Int64, Device::cpu());
        std::memcpy(rows_cpu.data<int64_t>(), rows.data(), rows.size() * sizeof(int64_t));
        Tensor rows_idx = rows_cpu.to(device);

        if (num_gt == 0) {
            // No GT for this image — all proposals are negatives. Sample up to
            // num_samples of them.
            std::vector<int64_t> shuffled(rows);
            std::shuffle(shuffled.begin(), shuffled.end(), rng);
            const int64_t take = std::min<int64_t>(num_samples, static_cast<int64_t>(shuffled.size()));
            for (int64_t i = 0; i < take; ++i) sampled_global.push_back(shuffled[static_cast<size_t>(i)]);
            continue;
        }

        Tensor props_b       = tenzor::ops::index_select(proposals_aug, 0, rows_idx); // (Nb, 5)
        Tensor props_boxes_b = props_b.slice(1, 1, 5);                                // (Nb, 4)

        Tensor gt_b = tenzor::ops::select(gt_boxes, 0, b);                            // (M, 4)
        if (gt_b.dtype() != dtype) gt_b = gt_b.to(dtype);

        Tensor iou      = tenzor::ops::box_iou(props_boxes_b, gt_b);                  // (Nb, M)
        Tensor max_iou  = tenzor::max(iou, 1);                                        // (Nb,)

        Tensor max_iou_cpu = max_iou.to(Device::cpu()).to(DType::Float32);
        const float* mp = max_iou_cpu.data<float>();

        std::vector<int64_t> pos_local, neg_local;
        pos_local.reserve(rows.size());
        neg_local.reserve(rows.size());
        for (int64_t i = 0; i < static_cast<int64_t>(rows.size()); ++i) {
            if (mp[i] >= static_cast<float>(fg_iou_thresh)) pos_local.push_back(i);
            else                                            neg_local.push_back(i);
        }

        std::shuffle(pos_local.begin(), pos_local.end(), rng);
        std::shuffle(neg_local.begin(), neg_local.end(), rng);

        const int64_t n_pos = std::min(num_positive,              static_cast<int64_t>(pos_local.size()));
        const int64_t n_neg = std::min(num_samples - n_pos,       static_cast<int64_t>(neg_local.size()));

        for (int64_t i = 0; i < n_pos; ++i) sampled_global.push_back(rows[static_cast<size_t>(pos_local[static_cast<size_t>(i)])]);
        for (int64_t i = 0; i < n_neg; ++i) sampled_global.push_back(rows[static_cast<size_t>(neg_local[static_cast<size_t>(i)])]);
    }

    if (sampled_global.empty()) {
        return Tensor({0, proposals.shape()[1]}, dtype, device);
    }

    Tensor idx_cpu({static_cast<int64_t>(sampled_global.size())}, DType::Int64, Device::cpu());
    std::memcpy(idx_cpu.data<int64_t>(), sampled_global.data(), sampled_global.size() * sizeof(int64_t));
    Tensor idx = idx_cpu.to(device);
    return tenzor::ops::index_select(proposals_aug, 0, idx);
}

void MaskRCNN::load_pretrained(const std::string& path, bool strict) {
    // Legitimate file-path loader for user-saved checkpoints.
    nn::ModelCheckpoint checkpoint_manager;
    auto checkpoint = checkpoint_manager.load(path);
    if (checkpoint.model_state.empty()) {
        throw std::runtime_error("MaskRCNN::load_pretrained: checkpoint '" + path +
                                 "' contains no model state");
    }
    load_state_dict(checkpoint.model_state, strict);
}

// ============================================================================
// Factory Functions
// ============================================================================

auto mask_rcnn_resnet50_fpn(int64_t num_classes, bool pretrained)
    -> std::shared_ptr<MaskRCNN> {

    // Backbone is initialized with randomly-initialized weights here because
    // the full Mask R-CNN COCO checkpoint below already contains the
    // ImageNet-trained backbone + RPN + ROI head + mask head weights as one
    // unified state_dict. Passing `pretrained=true` to resnet50() here would
    // download ImageNet weights only to be immediately overwritten by the
    // COCO checkpoint, so we keep the backbone unpretrained.
    auto backbone = resnet50(1000, /*pretrained=*/false);

    auto model = std::make_shared<MaskRCNN>(
        backbone, num_classes,
        800, 1333, 2000, 1000, 2000, 1000,
        0.7, 0.05, 0.5, 100);

    if (pretrained) {
        // NOTE: there is no safetensors mirror of the torchvision Mask R-CNN
        // COCO checkpoint, so this path does NOT silently return ImageNet/random
        // weights — download_pretrained_safetensors() throws a descriptive error
        // (the hub lists mask_rcnn_resnet50_fpn in removed_pretrained_reasons:
        // "torchvision detection checkpoint — no safetensors mirror"). When a
        // mirror becomes available this will load the full-model COCO weights
        // (backbone + FPN + RPN + ROI box head + mask head).
        auto path = ModelHub::download_pretrained_safetensors("mask_rcnn_resnet50_fpn");
        ModelHub::load_pretrained_weights(*model, path, /*strict=*/false);
    }

    return model;
}

auto mask_rcnn_resnet101_fpn(int64_t num_classes, bool pretrained)
    -> std::shared_ptr<MaskRCNN> {

    // Backbone is kept unpretrained: a pretrained Mask R-CNN would supply the
    // full COCO state_dict (backbone + FPN + RPN + ROI/mask heads) as one unit,
    // so downloading ImageNet backbone weights here would only be overwritten.
    auto backbone = resnet101(1000, /*pretrained=*/false);

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
        // No unified COCO checkpoint is registered for the ResNet-101 variant
        // (unlike mask_rcnn_resnet50_fpn). Returning a model with randomly
        // initialized head/RPN/mask weights while the caller believes a
        // pretrained detector was requested would be silently wrong, so fail
        // loudly instead.
        throw std::runtime_error(
            "mask_rcnn_resnet101_fpn: pretrained COCO weights are not available; "
            "construct with pretrained=false or use mask_rcnn_resnet50_fpn");
    }

    return model;
}

} // namespace models
} // namespace tenzor
