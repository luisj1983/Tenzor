/**
 * @file detection.hpp
 * @brief Object detection operations (IoU, NMS, box encoding)
 *
 * Provides core operations for object detection models:
 * - Intersection over Union (IoU, GIoU, DIoU, CIoU)
 * - Non-Maximum Suppression (NMS)
 * - Bounding box encoding/decoding
 */

#pragma once

#include <vector>
#include <cstdint>
#include "tenzor/core/tensor.hpp"

namespace tenzor {
namespace ops {

/**
 * @brief IoU variant types for box similarity computation.
 */
enum class IoUType {
    IoU,    ///< Standard Intersection over Union
    GIoU,   ///< Generalized IoU (penalizes non-overlapping boxes)
    DIoU,   ///< Distance IoU (considers center distance)
    CIoU    ///< Complete IoU (considers overlap, distance, and aspect ratio)
};

/**
 * @brief Compute Intersection over Union between box sets.
 *
 * Computes pairwise IoU between two sets of bounding boxes.
 *
 * @param boxes1 First set of boxes (N, 4) in (x1, y1, x2, y2) format
 * @param boxes2 Second set of boxes (M, 4) in (x1, y1, x2, y2) format
 * @param iou_type Type of IoU to compute (default: standard IoU)
 * @return IoU matrix of shape (N, M)
 *
 * @code
 * auto boxes1 = randn({100, 4});
 * auto boxes2 = randn({50, 4});
 * auto iou_matrix = box_iou(boxes1, boxes2);  // Shape: (100, 50)
 * @endcode
 */
auto box_iou(const Tensor& boxes1, const Tensor& boxes2,
             IoUType iou_type = IoUType::IoU) -> Tensor;

/**
 * @brief Non-Maximum Suppression for bounding boxes.
 *
 * Filters overlapping bounding boxes by keeping only the highest-scoring
 * box in each cluster. Uses greedy algorithm:
 * 1. Sort boxes by score (descending)
 * 2. Keep highest-scoring box
 * 3. Suppress all boxes with IoU > threshold
 * 4. Repeat for remaining boxes
 *
 * @param boxes Bounding boxes (N, 4) in (x1, y1, x2, y2) format
 * @param scores Confidence scores (N,)
 * @param iou_threshold IoU threshold for suppression (default: 0.5)
 * @return Indices of kept boxes (sorted by score)
 *
 * @code
 * auto boxes = randn({1000, 4});
 * auto scores = rand({1000});
 * auto keep = nms(boxes, scores, 0.5);
 * auto filtered_boxes = boxes.index_select(0, keep);
 * @endcode
 */
auto nms(const Tensor& boxes, const Tensor& scores,
         double iou_threshold = 0.5) -> Tensor;

/**
 * @brief Batched NMS for multiple classes.
 *
 * Applies NMS separately for each class to prevent cross-class suppression.
 * Useful for multi-class object detection where same object can have
 * multiple class predictions.
 *
 * @param boxes Bounding boxes (N, 4) in (x1, y1, x2, y2) format
 * @param scores Class scores (N, num_classes)
 * @param iou_threshold IoU threshold for suppression
 * @param score_threshold Minimum score to keep a box (default: 0.05)
 * @param max_output_boxes Maximum boxes to return per class (default: 100)
 * @return Tuple of (boxes, scores, labels) for kept detections
 */
auto batched_nms(const Tensor& boxes, const Tensor& scores,
                 double iou_threshold = 0.5,
                 double score_threshold = 0.05,
                 int64_t max_output_boxes = 100)
    -> std::tuple<Tensor, Tensor, Tensor>;

/**
 * @brief Encode bounding boxes relative to anchors.
 *
 * Converts absolute box coordinates to regression targets relative to
 * anchor boxes. Uses the standard RCNN encoding:
 *
 * dx = (box_x - anchor_x) / anchor_w / weights[0]
 * dy = (box_y - anchor_y) / anchor_h / weights[1]
 * dw = log(box_w / anchor_w) / weights[2]
 * dh = log(box_h / anchor_h) / weights[3]
 *
 * @param boxes Ground truth boxes (N, 4) in (x1, y1, x2, y2) format
 * @param anchors Anchor boxes (N, 4) in (x1, y1, x2, y2) format
 * @param weights Encoding weights for (dx, dy, dw, dh) (default: [1, 1, 1, 1])
 * @return Encoded deltas (N, 4) as (dx, dy, dw, dh)
 */
auto encode_boxes(const Tensor& boxes, const Tensor& anchors,
                  const std::vector<double>& weights = {1.0, 1.0, 1.0, 1.0})
    -> Tensor;

/**
 * @brief Decode bounding boxes from deltas and anchors.
 *
 * Inverse of encode_boxes. Converts regression predictions to absolute
 * box coordinates:
 *
 * box_x = anchor_x + dx * anchor_w * weights[0]
 * box_y = anchor_y + dy * anchor_h * weights[1]
 * box_w = anchor_w * exp(dw * weights[2])
 * box_h = anchor_h * exp(dh * weights[3])
 *
 * @param deltas Predicted deltas (N, 4) as (dx, dy, dw, dh)
 * @param anchors Anchor boxes (N, 4) in (x1, y1, x2, y2) format
 * @param weights Encoding weights (must match encode_boxes)
 * @return Decoded boxes (N, 4) in (x1, y1, x2, y2) format
 */
auto decode_boxes(const Tensor& deltas, const Tensor& anchors,
                  const std::vector<double>& weights = {1.0, 1.0, 1.0, 1.0})
    -> Tensor;

/**
 * @brief Clip boxes to image boundaries.
 *
 * Ensures all box coordinates are within [0, width) x [0, height).
 *
 * @param boxes Boxes to clip (N, 4) in (x1, y1, x2, y2) format
 * @param height Image height
 * @param width Image width
 * @return Clipped boxes (N, 4)
 */
auto clip_boxes_to_image(const Tensor& boxes, int64_t height, int64_t width)
    -> Tensor;

/**
 * @brief Remove boxes that are too small.
 *
 * Filters out boxes with width or height less than min_size.
 *
 * @param boxes Boxes to filter (N, 4)
 * @param scores Box scores (N,)
 * @param min_size Minimum width and height
 * @return Indices of boxes to keep
 */
auto remove_small_boxes(const Tensor& boxes, const Tensor& scores,
                        double min_size) -> Tensor;

} // namespace ops
} // namespace tenzor
