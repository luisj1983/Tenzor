/**
 * @file mask_head.hpp
 * @brief Mask prediction head for Mask R-CNN
 *
 * Implements the mask prediction branch that extends Faster R-CNN with
 * pixel-wise segmentation masks for instance segmentation.
 */

#pragma once

#include <memory>
#include <cstdint>
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/autograd/variable.hpp"

namespace tenzor {
namespace nn {
namespace detection {

/**
 * @brief Mask prediction head for Mask R-CNN.
 *
 * Predicts pixel-wise binary masks for each detected object instance.
 * The head consists of:
 * - 4 convolutional layers (3×3, 256 channels)
 * - 1 deconvolution layer (2× upsampling)
 * - 1×1 convolution for per-class mask prediction
 *
 * Architecture (Detectron2 style):
 * ```
 * ROI Features (14×14×256)
 *     ↓
 * Conv 3×3, 256 → ReLU
 *     ↓
 * Conv 3×3, 256 → ReLU
 *     ↓
 * Conv 3×3, 256 → ReLU
 *     ↓
 * Conv 3×3, 256 → ReLU
 *     ↓
 * ConvTranspose 2×2, 256 (2× upsample) → ReLU
 *     ↓
 * Conv 1×1, num_classes (28×28 mask per class)
 * ```
 *
 * During training:
 * - Input: ROI features from ROI Align (num_rois, 256, 14, 14)
 * - Output: Class-specific masks (num_rois, num_classes, 28, 28)
 * - Loss: Binary cross-entropy per pixel for predicted class mask
 *
 * During inference:
 * - Select mask for predicted class
 * - Resize to original ROI size
 * - Paste into full image at ROI location
 *
 * @code
 * MaskHead mask_head(256, 80);  // 256 channels, 80 classes (COCO)
 *
 * auto roi_features = randn({100, 256, 14, 14});  // From ROI Align
 * auto masks = mask_head.forward(roi_features);   // Shape: (100, 80, 28, 28)
 *
 * // During inference, select mask for predicted class
 * auto class_idx = 5;  // Person class
 * auto person_mask = masks.select(1, class_idx);  // Shape: (100, 28, 28)
 * @endcode
 *
 * Reference: "Mask R-CNN" (He et al., 2017)
 * https://arxiv.org/abs/1703.06870
 */
class MaskHead : public Module {
public:
    /**
     * @brief Construct mask prediction head.
     *
     * @param in_channels Number of input channels from ROI Align (typically 256)
     * @param num_classes Number of classes to predict masks for
     * @param conv_dim Number of channels in intermediate conv layers (default: 256)
     * @param num_conv Number of conv layers before upsampling (default: 4)
     * @param mask_size Output mask resolution (default: 28×28)
     *
     * @note Input ROI features should be mask_size/2 × mask_size/2 (e.g., 14×14)
     *       to produce mask_size × mask_size output after 2× upsampling
     */
    MaskHead(int64_t in_channels,
             int64_t num_classes,
             int64_t conv_dim = 256,
             int64_t num_conv = 4,
             int64_t mask_size = 28);

    /**
     * @brief Forward pass through mask head.
     *
     * @param roi_features ROI-aligned features (num_rois, in_channels, H, W)
     *                     Typically (num_rois, 256, 14, 14) from ROI Align
     * @return Mask logits (num_rois, num_classes, mask_size, mask_size)
     *         Typically (num_rois, num_classes, 28, 28)
     *         Apply sigmoid for probabilities
     *
     * @note During training, use all class masks with class labels
     *       During inference, select mask for predicted class only
     */
    auto forward(const Variable& roi_features) -> Variable override;

    /**
     * @brief Get output mask resolution.
     */
    auto mask_size() const -> int64_t { return mask_size_; }

    /**
     * @brief Get number of classes.
     */
    auto num_classes() const -> int64_t { return num_classes_; }

private:
    int64_t num_classes_;
    int64_t conv_dim_;
    int64_t mask_size_;

    // Convolutional layers for feature extraction
    std::vector<std::shared_ptr<Conv2d>> conv_layers_;
    std::vector<std::shared_ptr<BatchNorm2d>> bn_layers_;

    // Deconvolution for upsampling (2×)
    std::shared_ptr<ConvTranspose2d> deconv_;
    std::shared_ptr<BatchNorm2d> deconv_bn_;

    // Final 1×1 conv for mask prediction
    std::shared_ptr<Conv2d> mask_pred_;
};

/**
 * @brief Mask loss computation for Mask R-CNN.
 *
 * Computes binary cross-entropy loss per pixel for the mask of the
 * predicted class only. This encourages the network to predict accurate
 * masks without competition between classes.
 *
 * Loss computation:
 * 1. For each ROI, select mask logits for the ground truth class
 * 2. Compute BCE loss between predicted and target masks
 * 3. Average over all positive ROIs (ignore background)
 *
 * @param mask_logits Predicted mask logits (num_rois, num_classes, H, W)
 * @param mask_targets Ground truth binary masks (num_rois, H, W) in [0, 1]
 * @param class_labels Ground truth class labels (num_rois,)
 * @return Average mask loss over positive ROIs
 *
 * @code
 * auto mask_logits = mask_head.forward(roi_features);  // (100, 80, 28, 28)
 * auto mask_targets = get_target_masks();  // (100, 28, 28)
 * auto class_labels = get_class_labels();  // (100,)
 *
 * auto loss = mask_loss(mask_logits, mask_targets, class_labels);
 * @endcode
 */
auto mask_loss(const Variable& mask_logits,
               const Tensor& mask_targets,
               const Tensor& class_labels) -> Variable;

/**
 * @brief Post-process masks for visualization and evaluation.
 *
 * Converts predicted mask logits to binary masks at the original image scale:
 * 1. Select mask for predicted class
 * 2. Apply sigmoid to get probabilities
 * 3. Resize from 28×28 to ROI size using bilinear interpolation
 * 4. Threshold at 0.5 to get binary mask
 * 5. Paste into full image at ROI location
 *
 * @param mask_logits Predicted mask logits (num_detections, num_classes, 28, 28)
 * @param boxes Predicted bounding boxes (num_detections, 4) as (x1, y1, x2, y2)
 * @param class_labels Predicted class labels (num_detections,)
 * @param image_height Original image height
 * @param image_width Original image width
 * @param threshold Probability threshold for binary mask (default: 0.5)
 * @return Binary masks (num_detections, image_height, image_width)
 *
 * @code
 * auto detections = faster_rcnn.forward(image);
 * auto [boxes, labels, scores] = detections;
 *
 * auto mask_logits = mask_head.forward(roi_features);
 * auto masks = process_masks(mask_logits, boxes, labels, 800, 1200);
 * @endcode
 */
auto process_masks(const Tensor& mask_logits,
                   const Tensor& boxes,
                   const Tensor& class_labels,
                   int64_t image_height,
                   int64_t image_width,
                   double threshold = 0.5) -> Tensor;

} // namespace detection
} // namespace nn
} // namespace tenzor
