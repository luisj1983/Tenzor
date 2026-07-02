/**
 * @file faster_rcnn.hpp
 * @brief Faster R-CNN object detection model
 *
 * Implements Faster R-CNN, a two-stage object detector that achieves
 * state-of-the-art accuracy by combining:
 * 1. Region Proposal Network (RPN) for generating proposals
 * 2. ROI Head for classifying and refining proposals
 *
 * Reference: "Faster R-CNN: Towards Real-Time Object Detection with
 *            Region Proposal Networks" (Ren et al., NIPS 2015)
 *
 * Architecture:
 * ```
 * Input Image
 *     ↓
 * Backbone (ResNet-50/101) → Feature Maps
 *     ↓
 *     ├─→ RPN → Proposals
 *     └─→ ROI Head (with proposals) → Detections
 * ```
 */

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/detection/rpn.hpp"
#include "tenzor/nn/detection/roi_head.hpp"
#include "tenzor/nn/detection/anchors.hpp"
#include "tenzor/models/resnet.hpp"
#include "tenzor/autograd/variable.hpp"

namespace tenzor {
namespace models {

/**
 * @brief Faster R-CNN object detection model.
 *
 * Two-stage detector with separate networks for proposal generation
 * and object detection. Achieves high accuracy at the cost of slower
 * inference compared to single-stage detectors.
 *
 * **Training Pipeline:**
 * 1. Extract features using backbone (ResNet)
 * 2. Generate proposals with RPN
 * 3. Sample proposals and match to ground truth
 * 4. Classify and refine boxes with ROI head
 * 5. Compute combined loss (RPN + ROI)
 *
 * **Inference Pipeline:**
 * 1. Extract features using backbone
 * 2. Generate proposals with RPN
 * 3. Classify and refine with ROI head
 * 4. Apply NMS and return top detections
 *
 * @code
 * // Create Faster R-CNN with ResNet-50 backbone
 * auto model = faster_rcnn_resnet50(num_classes=80);
 *
 * // Training
 * model->train();
 * auto losses = model->forward_train(images, targets);
 * auto total_loss = losses["loss_rpn"] + losses["loss_roi"];
 * total_loss.backward();
 *
 * // Inference
 * model->eval();
 * auto detections = model->forward(images);
 * // detections[i] = {"boxes": (K,4), "labels": (K,), "scores": (K,)}
 * @endcode
 */
class FasterRCNN : public nn::Module {
public:
    /**
     * @brief Construct Faster R-CNN model.
     *
     * @param backbone Feature extraction backbone (e.g., ResNet-50)
     * @param num_classes Number of object classes (not including background)
     * @param rpn_anchor_sizes Anchor sizes for RPN (default: {32, 64, 128, 256, 512})
     * @param rpn_aspect_ratios Anchor aspect ratios (default: {0.5, 1.0, 2.0})
     * @param rpn_fg_iou_thresh RPN foreground IoU threshold (default: 0.7)
     * @param rpn_bg_iou_thresh RPN background IoU threshold (default: 0.3)
     * @param rpn_batch_size_per_image RPN batch size (default: 256)
     * @param rpn_positive_fraction RPN positive fraction (default: 0.5)
     * @param rpn_pre_nms_top_n Pre-NMS proposals (default: 2000)
     * @param rpn_post_nms_top_n Post-NMS proposals (default: 1000)
     * @param rpn_nms_thresh RPN NMS threshold (default: 0.7)
     * @param roi_output_size ROI Align output size (default: 7)
     * @param roi_spatial_scale Feature map scale (default: 1/32, matching the
     *        stride-32 ResNet C5 map returned by ResNet::forward_features)
     * @param roi_sampling_ratio ROI Align sampling ratio (default: 2)
     * @param roi_fg_iou_thresh ROI foreground IoU threshold (default: 0.5)
     * @param roi_bg_iou_thresh ROI background IoU threshold (default: 0.5)
     * @param roi_batch_size_per_image ROI batch size (default: 512)
     * @param roi_positive_fraction ROI positive fraction (default: 0.25)
     * @param roi_score_thresh Detection score threshold (default: 0.05)
     * @param roi_nms_thresh Detection NMS threshold (default: 0.5)
     * @param roi_detections_per_img Max detections per image (default: 100)
     * @param backbone_out_channels Number of feature channels produced by the
     *        backbone. Pass -1 (default) to introspect a ResNet backbone via
     *        out_channels(); required for non-ResNet custom backbones whose
     *        channel count cannot be derived automatically.
     */
    FasterRCNN(std::shared_ptr<nn::Module> backbone,
               int64_t num_classes,
               std::vector<float> rpn_anchor_sizes = {32.0f, 64.0f, 128.0f, 256.0f, 512.0f},
               std::vector<float> rpn_aspect_ratios = {0.5f, 1.0f, 2.0f},
               double rpn_fg_iou_thresh = 0.7,
               double rpn_bg_iou_thresh = 0.3,
               int64_t rpn_batch_size_per_image = 256,
               double rpn_positive_fraction = 0.5,
               int64_t rpn_pre_nms_top_n = 2000,
               int64_t rpn_post_nms_top_n = 1000,
               double rpn_nms_thresh = 0.7,
               int64_t roi_output_size = 7,
               double roi_spatial_scale = 1.0 / 32.0,
               int64_t roi_sampling_ratio = 2,
               double roi_fg_iou_thresh = 0.5,
               double roi_bg_iou_thresh = 0.5,
               int64_t roi_batch_size_per_image = 512,
               double roi_positive_fraction = 0.25,
               double roi_score_thresh = 0.05,
               double roi_nms_thresh = 0.5,
               int64_t roi_detections_per_img = 100,
               int64_t backbone_out_channels = -1);

    /**
     * @brief Forward pass for inference.
     *
     * Detects objects in input images.
     *
     * @param images Input images (N, 3, H, W)
     * @param image_shapes Optional list of (height, width) for each image
     *                     If not provided, uses images.shape()[2:4]
     * @return List of detections, one dict per image:
     *         - "boxes": (K, 4) bounding boxes in (x1, y1, x2, y2) format
     *         - "labels": (K,) class labels (1 to num_classes)
     *         - "scores": (K,) confidence scores
     */
    auto forward_inference(
        const Variable& images,
        const std::vector<std::pair<int64_t, int64_t>>* image_shapes = nullptr
    ) -> std::vector<std::unordered_map<std::string, Tensor>>;

    /**
     * @brief Forward pass for training.
     *
     * Computes detection losses for training.
     *
     * @param images Input images (N, 3, H, W)
     * @param targets List of ground truth boxes and labels per image:
     *                - "boxes": (num_gt, 4) in (x1, y1, x2, y2) format
     *                - "labels": (num_gt,) class labels
     * @param image_shapes Optional list of (height, width) for each image
     * @return Dictionary of losses:
     *         - "loss_objectness": RPN objectness loss
     *         - "loss_rpn_box_reg": RPN box regression loss
     *         - "loss_classifier": ROI classification loss
     *         - "loss_box_reg": ROI box regression loss
     */
    auto forward_train(
        const Variable& images,
        const std::vector<std::unordered_map<std::string, Tensor>>& targets,
        const std::vector<std::pair<int64_t, int64_t>>* image_shapes = nullptr
    ) -> std::unordered_map<std::string, Variable>;

    /**
     * @brief Module forward (calls forward_inference).
     *
     * @param input Input images
     * @return Dummy output (use forward_inference or forward_train instead)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Load pretrained weights from file.
     *
     * @param path Path to pretrained weights
     */
    auto load_pretrained(const std::string& path) -> void;

    /**
     * @brief Get number of classes.
     *
     * @return Number of object classes (not including background)
     */
    auto num_classes() const -> int64_t { return num_classes_; }

private:
    /**
     * @brief Extract image shapes from tensor.
     */
    auto get_image_shapes(const Variable& images)
        -> std::vector<std::pair<int64_t, int64_t>>;

    int64_t num_classes_;

    // Backbone for feature extraction
    std::shared_ptr<nn::Module> backbone_;

    // Region Proposal Network
    std::shared_ptr<nn::detection::RegionProposalNetwork> rpn_;

    // ROI Head for detection
    std::shared_ptr<nn::detection::RoIHead> roi_head_;
};

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * @brief Create Faster R-CNN with ResNet-50 backbone.
 *
 * Uses ResNet-50 pretrained on ImageNet as the backbone network.
 * The backbone is frozen up to layer3 for better training efficiency.
 *
 * @param num_classes Number of object classes (not including background)
 * @param pretrained Load pretrained ImageNet weights for backbone (default: true)
 * @param pretrained_backbone Path to custom backbone weights (optional)
 * @return Shared pointer to Faster R-CNN model
 *
 * @code
 * auto model = faster_rcnn_resnet50(80, true);  // COCO: 80 classes
 * @endcode
 */
auto faster_rcnn_resnet50(
    int64_t num_classes,
    bool pretrained = true,
    const std::string& pretrained_backbone = ""
) -> std::shared_ptr<FasterRCNN>;

/**
 * @brief Create Faster R-CNN with ResNet-101 backbone.
 *
 * Uses deeper ResNet-101 for improved accuracy at the cost of speed.
 *
 * @param num_classes Number of object classes (not including background)
 * @param pretrained Load pretrained ImageNet weights for backbone (default: true)
 * @param pretrained_backbone Path to custom backbone weights (optional)
 * @return Shared pointer to Faster R-CNN model
 */
auto faster_rcnn_resnet101(
    int64_t num_classes,
    bool pretrained = true,
    const std::string& pretrained_backbone = ""
) -> std::shared_ptr<FasterRCNN>;

/**
 * @brief Create Faster R-CNN with custom backbone.
 *
 * Allows using any custom backbone network.
 *
 * @param backbone Custom backbone module
 * @param backbone_out_channels Number of output channels from backbone
 * @param num_classes Number of object classes
 * @return Shared pointer to Faster R-CNN model
 */
auto faster_rcnn_custom(
    std::shared_ptr<nn::Module> backbone,
    int64_t backbone_out_channels,
    int64_t num_classes
) -> std::shared_ptr<FasterRCNN>;

} // namespace models
} // namespace tenzor
