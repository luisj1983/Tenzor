/**
 * @file detection.cpp
 * @brief Object detection operations implementation (CPU)
 */

#include "tenzor/ops/detection.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <iostream>

namespace tenzor {
namespace ops {

// Helper functions for element-wise maximum and minimum
static auto element_maximum(const Tensor& a, const Tensor& b) -> Tensor {
    // max(a, b) = (a + b + |a - b|) / 2
    return (a + b + abs(a - b)) * 0.5f;
}

static auto element_minimum(const Tensor& a, const Tensor& b) -> Tensor {
    // min(a, b) = (a + b - |a - b|) / 2
    return (a + b - abs(a - b)) * 0.5f;
}

static auto clamp_min_scalar(const Tensor& a, float min_val) -> Tensor {
    // clamp_min(a, min_val) = max(a, min_val)
    // Using element-wise formula: max(a, b) = (a + b + |a - b|) / 2
    return (a + min_val + abs(a - min_val)) * 0.5f;
}

// Helper: Compute area of boxes
static auto box_area(const Tensor& boxes) -> Tensor {
    // boxes: (N, 4) with (x1, y1, x2, y2)
    // area = (x2 - x1) * (y2 - y1)

    const int64_t N = boxes.shape()[0];

    // Manual computation to avoid slice bugs
    auto result = zeros({N}, DType::Float32, boxes.device());

    if (boxes.device() == Device::cpu()) {
        const float* boxes_data = static_cast<const float*>(boxes.data_ptr());
        float* result_data = static_cast<float*>(result.data_ptr());

        for (int64_t i = 0; i < N; i++) {
            float x1 = boxes_data[i * 4 + 0];
            float y1 = boxes_data[i * 4 + 1];
            float x2 = boxes_data[i * 4 + 2];
            float y2 = boxes_data[i * 4 + 3];

            float width = x2 - x1;
            float height = y2 - y1;
            result_data[i] = width * height;
        }
    } else {
        // For non-CPU devices, fall back to slice-based approach
        auto widths = boxes.slice(1, 2, 3) - boxes.slice(1, 0, 1);   // x2 - x1
        auto heights = boxes.slice(1, 3, 4) - boxes.slice(1, 1, 2);  // y2 - y1
        result = (widths * heights).squeeze(1);
    }

    return result;
}

auto box_iou(const Tensor& boxes1, const Tensor& boxes2, IoUType iou_type) -> Tensor {
    if (boxes1.ndim() != 2 || boxes2.ndim() != 2) {
        throw std::invalid_argument("Boxes must be 2D tensors");
    }
    if (boxes1.shape()[1] != 4 || boxes2.shape()[1] != 4) {
        throw std::invalid_argument("Boxes must have 4 coordinates");
    }

    const int64_t N = boxes1.shape()[0];
    const int64_t M = boxes2.shape()[0];

    // For CPU, use manual implementation to avoid slice bugs
    if (boxes1.device() == Device::cpu() && boxes2.device() == Device::cpu()) {
        auto result = zeros({N, M}, DType::Float32, Device::cpu());
        const float* boxes1_data = static_cast<const float*>(boxes1.data_ptr());
        const float* boxes2_data = static_cast<const float*>(boxes2.data_ptr());
        float* result_data = static_cast<float*>(result.data_ptr());

        constexpr float pi = 3.14159265358979323846f;
        constexpr float four_over_pi_sq = 4.0f / (pi * pi);

        for (int64_t i = 0; i < N; i++) {
            float x1_1 = boxes1_data[i * 4 + 0];
            float y1_1 = boxes1_data[i * 4 + 1];
            float x2_1 = boxes1_data[i * 4 + 2];
            float y2_1 = boxes1_data[i * 4 + 3];
            float w1 = x2_1 - x1_1;
            float h1 = y2_1 - y1_1;
            float area1 = w1 * h1;
            float cx1 = (x1_1 + x2_1) * 0.5f;
            float cy1 = (y1_1 + y2_1) * 0.5f;

            for (int64_t j = 0; j < M; j++) {
                float x1_2 = boxes2_data[j * 4 + 0];
                float y1_2 = boxes2_data[j * 4 + 1];
                float x2_2 = boxes2_data[j * 4 + 2];
                float y2_2 = boxes2_data[j * 4 + 3];
                float w2 = x2_2 - x1_2;
                float h2 = y2_2 - y1_2;
                float area2 = w2 * h2;
                float cx2 = (x1_2 + x2_2) * 0.5f;
                float cy2 = (y1_2 + y2_2) * 0.5f;

                // Intersection
                float inter_x1 = std::max(x1_1, x1_2);
                float inter_y1 = std::max(y1_1, y1_2);
                float inter_x2 = std::min(x2_1, x2_2);
                float inter_y2 = std::min(y2_1, y2_2);
                float inter_w = std::max(0.0f, inter_x2 - inter_x1);
                float inter_h = std::max(0.0f, inter_y2 - inter_y1);
                float inter_area = inter_w * inter_h;

                // Union
                float union_area = area1 + area2 - inter_area;

                // IoU
                float iou = inter_area / (union_area + 1e-7f);

                if (iou_type == IoUType::IoU) {
                    result_data[i * M + j] = iou;
                } else if (iou_type == IoUType::GIoU) {
                    // Enclosing box
                    float enclose_x1 = std::min(x1_1, x1_2);
                    float enclose_y1 = std::min(y1_1, y1_2);
                    float enclose_x2 = std::max(x2_1, x2_2);
                    float enclose_y2 = std::max(y2_1, y2_2);
                    float enclose_w = enclose_x2 - enclose_x1;
                    float enclose_h = enclose_y2 - enclose_y1;
                    float enclose_area = enclose_w * enclose_h;
                    result_data[i * M + j] = iou - (enclose_area - union_area) / (enclose_area + 1e-7f);
                } else if (iou_type == IoUType::DIoU || iou_type == IoUType::CIoU) {
                    // Enclosing box for diagonal distance
                    float enclose_x1 = std::min(x1_1, x1_2);
                    float enclose_y1 = std::min(y1_1, y1_2);
                    float enclose_x2 = std::max(x2_1, x2_2);
                    float enclose_y2 = std::max(y2_1, y2_2);
                    float enclose_w = enclose_x2 - enclose_x1;
                    float enclose_h = enclose_y2 - enclose_y1;

                    // Center distance squared
                    float center_dist_sq = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);

                    // Diagonal distance squared
                    float diag_dist_sq = enclose_w * enclose_w + enclose_h * enclose_h;

                    if (iou_type == IoUType::DIoU) {
                        result_data[i * M + j] = iou - center_dist_sq / (diag_dist_sq + 1e-7f);
                    } else {  // CIoU
                        // Aspect ratio consistency
                        float v = four_over_pi_sq * std::pow(std::atan(w2 / (h2 + 1e-7f)) - std::atan(w1 / (h1 + 1e-7f)), 2.0f);
                        float alpha = v / (1.0f - iou + v + 1e-7f);
                        result_data[i * M + j] = iou - center_dist_sq / (diag_dist_sq + 1e-7f) - alpha * v;
                    }
                }
            }
        }

        return result;
    }

    // Compute areas
    auto area1 = box_area(boxes1);  // (N,)
    auto area2 = box_area(boxes2);  // (M,)

    // Broadcast areas for pairwise computation (automatic with unsqueeze)
    auto area1_expanded = area1.unsqueeze(1);  // (N, 1) - broadcasts to (N, M)
    auto area2_expanded = area2.unsqueeze(0);  // (1, M) - broadcasts to (N, M)

    // Extract coordinates
    auto x1_1 = boxes1.slice(1, 0, 1);  // (N, 1)
    auto y1_1 = boxes1.slice(1, 1, 2);
    auto x2_1 = boxes1.slice(1, 2, 3);
    auto y2_1 = boxes1.slice(1, 3, 4);

    auto x1_2 = boxes2.slice(1, 0, 1);  // (M, 1)
    auto y1_2 = boxes2.slice(1, 1, 2);
    auto x2_2 = boxes2.slice(1, 2, 3);
    auto y2_2 = boxes2.slice(1, 3, 4);

    // Compute intersection coordinates
    // inter_x1 = max(x1_1, x1_2)
    auto inter_x1 = element_maximum(x1_1, x1_2.transpose(0, 1));  // (N, M)
    auto inter_y1 = element_maximum(y1_1, y1_2.transpose(0, 1));
    auto inter_x2 = element_minimum(x2_1, x2_2.transpose(0, 1));
    auto inter_y2 = element_minimum(y2_1, y2_2.transpose(0, 1));

    // Intersection area
    auto inter_w = clamp_min_scalar(inter_x2 - inter_x1, 0.0f);
    auto inter_h = clamp_min_scalar(inter_y2 - inter_y1, 0.0f);
    auto inter_area = inter_w * inter_h;  // (N, M)

    // Union area
    auto union_area = area1_expanded + area2_expanded - inter_area;

    // Compute IoU
    auto iou = inter_area / (union_area + 1e-7f);

    // DEBUG: Check IoU calculation for single box case
    if (N == 1 && M == 1) {
        const float* area1_ptr = static_cast<const float*>(area1.data_ptr());
        const float* area2_ptr = static_cast<const float*>(area2.data_ptr());
        const float* inter_ptr = static_cast<const float*>(inter_area.data_ptr());
        const float* union_ptr = static_cast<const float*>(union_area.data_ptr());
        const float* iou_ptr = static_cast<const float*>(iou.data_ptr());
        std::cerr << "[IoU DEBUG] area1=" << area1_ptr[0]
                  << " area2=" << area2_ptr[0]
                  << " inter=" << inter_ptr[0]
                  << " union=" << union_ptr[0]
                  << " IoU=" << iou_ptr[0] << std::endl;
    }

    if (iou_type == IoUType::IoU) {
        return iou;
    }

    // For GIoU, DIoU, CIoU: compute enclosing box
    auto enclose_x1 = element_minimum(x1_1, x1_2.transpose(0, 1));
    auto enclose_y1 = element_minimum(y1_1, y1_2.transpose(0, 1));
    auto enclose_x2 = element_maximum(x2_1, x2_2.transpose(0, 1));
    auto enclose_y2 = element_maximum(y2_1, y2_2.transpose(0, 1));

    auto enclose_w = enclose_x2 - enclose_x1;
    auto enclose_h = enclose_y2 - enclose_y1;
    auto enclose_area = enclose_w * enclose_h;

    if (iou_type == IoUType::GIoU) {
        // GIoU = IoU - (enclose_area - union_area) / enclose_area
        return iou - (enclose_area - union_area) / (enclose_area + 1e-7f);
    }

    // Compute center points for DIoU and CIoU
    auto cx1 = (x1_1 + x2_1) * 0.5f;
    auto cy1 = (y1_1 + y2_1) * 0.5f;
    auto cx2 = (x1_2 + x2_2) * 0.5f;
    auto cy2 = (y1_2 + y2_2) * 0.5f;

    // Center distance squared
    auto center_dist_sq = pow(cx1 - cx2.transpose(0, 1), 2.0f) +
                          pow(cy1 - cy2.transpose(0, 1), 2.0f);

    // Diagonal distance squared of enclosing box
    auto diag_dist_sq = pow(enclose_w, 2.0f) + pow(enclose_h, 2.0f);

    if (iou_type == IoUType::DIoU) {
        // DIoU = IoU - center_dist^2 / diag_dist^2
        return iou - center_dist_sq / (diag_dist_sq + 1e-7f);
    }

    // CIoU: Complete IoU with aspect ratio consistency
    // CIoU = IoU - (center_dist^2 / diag_dist^2) - alpha * v
    // where v = (4/pi^2) * (atan(w_gt/h_gt) - atan(w/h))^2
    // and alpha = v / (1 - IoU + v)

    // Compute box widths and heights
    auto w1 = (x2_1 - x1_1).squeeze(1);  // (N,)
    auto h1 = (y2_1 - y1_1).squeeze(1);
    auto w2 = (x2_2 - x1_2).squeeze(1);  // (M,)
    auto h2 = (y2_2 - y1_2).squeeze(1);

    // For now, use CPU implementation of atan for aspect ratio calculation
    // Create result tensors on CPU for computation
    constexpr float pi = 3.14159265358979323846f;
    constexpr float four_over_pi_sq = 4.0f / (pi * pi);

    // Move tensors to CPU for atan computation
    auto w1_cpu = w1.to(Device::cpu());
    auto h1_cpu = h1.to(Device::cpu());
    auto w2_cpu = w2.to(Device::cpu());
    auto h2_cpu = h2.to(Device::cpu());
    auto iou_cpu = iou.to(Device::cpu());
    auto center_dist_sq_cpu = center_dist_sq.to(Device::cpu());
    auto diag_dist_sq_cpu = diag_dist_sq.to(Device::cpu());

    const int64_t num_boxes1 = w1_cpu.shape()[0];
    const int64_t num_boxes2 = w2_cpu.shape()[0];

    // Create output tensor for v (aspect ratio penalty)
    Tensor v({num_boxes1, num_boxes2}, DType::Float32, Device::cpu());
    Tensor alpha({num_boxes1, num_boxes2}, DType::Float32, Device::cpu());

    const float* w1_data = static_cast<const float*>(w1_cpu.data_ptr());
    const float* h1_data = static_cast<const float*>(h1_cpu.data_ptr());
    const float* w2_data = static_cast<const float*>(w2_cpu.data_ptr());
    const float* h2_data = static_cast<const float*>(h2_cpu.data_ptr());
    const float* iou_data = static_cast<const float*>(iou_cpu.data_ptr());
    float* v_data = static_cast<float*>(v.data_ptr());
    float* alpha_data = static_cast<float*>(alpha.data_ptr());

    // Compute aspect ratio penalty element-wise
    for (int64_t i = 0; i < num_boxes1; ++i) {
        float ar1 = std::atan(w1_data[i] / (h1_data[i] + 1e-7f));
        for (int64_t j = 0; j < num_boxes2; ++j) {
            float ar2 = std::atan(w2_data[j] / (h2_data[j] + 1e-7f));
            float ar_diff = ar1 - ar2;
            float v_ij = four_over_pi_sq * ar_diff * ar_diff;
            v_data[i * num_boxes2 + j] = v_ij;

            // Compute alpha weighting factor
            float iou_ij = iou_data[i * num_boxes2 + j];
            alpha_data[i * num_boxes2 + j] = v_ij / (1.0f - iou_ij + v_ij + 1e-7f);
        }
    }

    // Complete IoU
    // CIoU = IoU - center_dist^2 / diag_dist^2 - alpha * v

    // DEBUG: Print shapes and first values
    if (num_boxes1 == 1 && num_boxes2 == 1) {
        const float* iou_ptr = static_cast<const float*>(iou_cpu.data_ptr());
        const float* center_ptr = static_cast<const float*>(center_dist_sq_cpu.data_ptr());
        const float* diag_ptr = static_cast<const float*>(diag_dist_sq_cpu.data_ptr());
        std::cerr << "[CIoU DEBUG] IoU=" << iou_ptr[0]
                  << " center_dist²=" << center_ptr[0]
                  << " diag_dist²=" << diag_ptr[0]
                  << " v=" << v_data[0]
                  << " alpha=" << alpha_data[0] << std::endl;
    }

    auto ciou_cpu = iou_cpu - center_dist_sq_cpu / (diag_dist_sq_cpu + 1e-7f) - alpha * v;

    // Move result back to original device
    return ciou_cpu.to(boxes1.device());
}

auto nms(const Tensor& boxes, const Tensor& scores, double iou_threshold) -> Tensor {
    if (boxes.ndim() != 2 || boxes.shape()[1] != 4) {
        throw std::invalid_argument("Boxes must be (N, 4) tensor");
    }
    if (scores.ndim() != 1) {
        throw std::invalid_argument("Scores must be 1D tensor");
    }
    if (boxes.shape()[0] != scores.shape()[0]) {
        throw std::invalid_argument("Number of boxes and scores must match");
    }

    const int64_t N = boxes.shape()[0];
    if (N == 0) {
        return tenzor::empty({0}, DType::Int64, boxes.device());
    }

    // Move to CPU for processing
    auto boxes_cpu = boxes.to(Device::cpu());
    auto scores_cpu = scores.to(Device::cpu());

    // Get data pointers
    const float* boxes_data = static_cast<const float*>(boxes_cpu.data_ptr());
    const float* scores_data = static_cast<const float*>(scores_cpu.data_ptr());

    // Sort indices by score (descending)
    std::vector<int64_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [scores_data](int64_t i, int64_t j) {
                  return scores_data[i] > scores_data[j];
              });

    // NMS algorithm
    std::vector<bool> suppressed(N, false);
    std::vector<int64_t> keep;
    keep.reserve(N);

    for (int64_t i = 0; i < N; ++i) {
        int64_t idx = indices[i];
        if (suppressed[idx]) continue;

        keep.push_back(idx);

        // Get current box
        const float x1 = boxes_data[idx * 4 + 0];
        const float y1 = boxes_data[idx * 4 + 1];
        const float x2 = boxes_data[idx * 4 + 2];
        const float y2 = boxes_data[idx * 4 + 3];
        const float area = (x2 - x1) * (y2 - y1);

        // Suppress overlapping boxes
        for (int64_t j = i + 1; j < N; ++j) {
            int64_t idx2 = indices[j];
            if (suppressed[idx2]) continue;

            // Compute IoU
            const float x1_2 = boxes_data[idx2 * 4 + 0];
            const float y1_2 = boxes_data[idx2 * 4 + 1];
            const float x2_2 = boxes_data[idx2 * 4 + 2];
            const float y2_2 = boxes_data[idx2 * 4 + 3];

            const float inter_x1 = std::max(x1, x1_2);
            const float inter_y1 = std::max(y1, y1_2);
            const float inter_x2 = std::min(x2, x2_2);
            const float inter_y2 = std::min(y2, y2_2);

            const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
            const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
            const float inter_area = inter_w * inter_h;

            const float area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
            const float union_area = area + area2 - inter_area;
            const float iou = inter_area / (union_area + 1e-7f);

            if (iou > iou_threshold) {
                suppressed[idx2] = true;
            }
        }
    }

    // Create result tensor and copy data to avoid dangling pointer
    auto result = zeros({static_cast<int64_t>(keep.size())}, DType::Int64, Device::cpu());
    int64_t* result_ptr = result.data<int64_t>();
    std::copy(keep.begin(), keep.end(), result_ptr);

    return result.to(boxes.device());
}

auto batched_nms(const Tensor& boxes, const Tensor& scores,
                 double iou_threshold, double score_threshold,
                 int64_t max_output_boxes)
    -> std::tuple<Tensor, Tensor, Tensor> {

    if (boxes.ndim() != 2 || boxes.shape()[1] != 4) {
        throw std::invalid_argument("Boxes must be (N, 4) tensor");
    }
    if (scores.ndim() != 2) {
        throw std::invalid_argument("Scores must be 2D tensor");
    }
    if (boxes.shape()[0] != scores.shape()[0]) {
        throw std::invalid_argument("Number of boxes and scores must match");
    }

    const int64_t num_boxes = boxes.shape()[0];
    const int64_t num_classes = scores.shape()[1];

    std::vector<float> all_boxes;
    std::vector<float> all_scores;
    std::vector<int64_t> all_labels;

    // Apply NMS per class
    for (int64_t cls = 0; cls < num_classes; ++cls) {
        auto class_scores = scores.slice(1, cls, cls + 1).squeeze(1);

        // Filter by score threshold - create scalar tensor for comparison
        auto score_shape = class_scores.shape();
        std::vector<int64_t> shape_vec(score_shape.begin(), score_shape.end());
        auto threshold_tensor = tenzor::full(shape_vec, static_cast<float>(score_threshold),
                                              class_scores.dtype(), class_scores.device());
        auto mask = class_scores > threshold_tensor;
        auto indices = mask.nonzero().squeeze(1);

        if (indices.shape()[0] == 0) continue;

        auto filtered_boxes = tenzor::index_select(boxes, 0, indices);
        auto filtered_scores = tenzor::index_select(class_scores, 0, indices);

        // Apply NMS
        auto keep = nms(filtered_boxes, filtered_scores, iou_threshold);

        // Limit number of boxes
        int64_t num_keep = std::min(keep.shape()[0], max_output_boxes);
        if (num_keep < keep.shape()[0]) {
            keep = keep.slice(0, 0, num_keep);
        }

        // Collect results
        auto kept_boxes = tenzor::index_select(filtered_boxes, 0, keep);
        auto kept_scores = tenzor::index_select(filtered_scores, 0, keep);

        // Append to output
        const float* box_data = static_cast<const float*>(kept_boxes.data_ptr());
        const float* score_data = static_cast<const float*>(kept_scores.data_ptr());

        for (int64_t i = 0; i < num_keep; ++i) {
            all_boxes.push_back(box_data[i * 4 + 0]);
            all_boxes.push_back(box_data[i * 4 + 1]);
            all_boxes.push_back(box_data[i * 4 + 2]);
            all_boxes.push_back(box_data[i * 4 + 3]);
            all_scores.push_back(score_data[i]);
            all_labels.push_back(cls);
        }
    }

    const int64_t total_kept = static_cast<int64_t>(all_labels.size());

    // Create result tensors and copy data to avoid dangling pointers
    auto result_boxes = zeros({total_kept, 4}, DType::Float32, Device::cpu());
    float* boxes_ptr = result_boxes.data<float>();
    std::copy(all_boxes.begin(), all_boxes.end(), boxes_ptr);
    result_boxes = result_boxes.to(boxes.device());

    auto result_scores = zeros({total_kept}, DType::Float32, Device::cpu());
    float* scores_ptr = result_scores.data<float>();
    std::copy(all_scores.begin(), all_scores.end(), scores_ptr);
    result_scores = result_scores.to(boxes.device());

    auto result_labels = zeros({total_kept}, DType::Int64, Device::cpu());
    int64_t* labels_ptr = result_labels.data<int64_t>();
    std::copy(all_labels.begin(), all_labels.end(), labels_ptr);
    result_labels = result_labels.to(boxes.device());

    return std::make_tuple(result_boxes, result_scores, result_labels);
}

auto encode_boxes(const Tensor& boxes, const Tensor& anchors,
                  const std::vector<double>& weights) -> Tensor {
    // Check shape compatibility
    auto boxes_shape = boxes.shape();
    auto anchors_shape = anchors.shape();
    if (boxes_shape.size() != anchors_shape.size()) {
        throw std::invalid_argument("Boxes and anchors must have same rank");
    }
    for (size_t i = 0; i < boxes_shape.size(); ++i) {
        if (boxes_shape[i] != anchors_shape[i]) {
            throw std::invalid_argument("Boxes and anchors must have same shape");
        }
    }
    if (weights.size() != 4) {
        throw std::invalid_argument("Weights must have 4 elements");
    }

    // Convert to (cx, cy, w, h) format
    auto boxes_x1 = boxes.slice(1, 0, 1);
    auto boxes_y1 = boxes.slice(1, 1, 2);
    auto boxes_x2 = boxes.slice(1, 2, 3);
    auto boxes_y2 = boxes.slice(1, 3, 4);

    auto anchors_x1 = anchors.slice(1, 0, 1);
    auto anchors_y1 = anchors.slice(1, 1, 2);
    auto anchors_x2 = anchors.slice(1, 2, 3);
    auto anchors_y2 = anchors.slice(1, 3, 4);

    auto boxes_w = boxes_x2 - boxes_x1;
    auto boxes_h = boxes_y2 - boxes_y1;
    auto boxes_cx = boxes_x1 + boxes_w * 0.5f;
    auto boxes_cy = boxes_y1 + boxes_h * 0.5f;

    auto anchors_w = anchors_x2 - anchors_x1;
    auto anchors_h = anchors_y2 - anchors_y1;
    auto anchors_cx = anchors_x1 + anchors_w * 0.5f;
    auto anchors_cy = anchors_y1 + anchors_h * 0.5f;

    // Encode
    auto dx = (boxes_cx - anchors_cx) / (anchors_w + 1e-7f) / weights[0];
    auto dy = (boxes_cy - anchors_cy) / (anchors_h + 1e-7f) / weights[1];
    auto dw = log(boxes_w / (anchors_w + 1e-7f)) / weights[2];
    auto dh = log(boxes_h / (anchors_h + 1e-7f)) / weights[3];

    return tenzor::cat({dx, dy, dw, dh}, 1);
}

auto decode_boxes(const Tensor& deltas, const Tensor& anchors,
                  const std::vector<double>& weights) -> Tensor {
    // Check shape compatibility
    auto deltas_shape = deltas.shape();
    auto anchors_shape = anchors.shape();
    if (deltas_shape.size() != anchors_shape.size()) {
        throw std::invalid_argument("Deltas and anchors must have same rank");
    }
    for (size_t i = 0; i < deltas_shape.size(); ++i) {
        if (deltas_shape[i] != anchors_shape[i]) {
            throw std::invalid_argument("Deltas and anchors must have same shape");
        }
    }
    if (weights.size() != 4) {
        throw std::invalid_argument("Weights must have 4 elements");
    }

    // Extract deltas
    auto dx = deltas.slice(1, 0, 1) * weights[0];
    auto dy = deltas.slice(1, 1, 2) * weights[1];
    auto dw = deltas.slice(1, 2, 3) * weights[2];
    auto dh = deltas.slice(1, 3, 4) * weights[3];

    // Extract anchors
    auto anchors_x1 = anchors.slice(1, 0, 1);
    auto anchors_y1 = anchors.slice(1, 1, 2);
    auto anchors_x2 = anchors.slice(1, 2, 3);
    auto anchors_y2 = anchors.slice(1, 3, 4);

    auto anchors_w = anchors_x2 - anchors_x1;
    auto anchors_h = anchors_y2 - anchors_y1;
    auto anchors_cx = anchors_x1 + anchors_w * 0.5f;
    auto anchors_cy = anchors_y1 + anchors_h * 0.5f;

    // Decode
    auto pred_cx = dx * anchors_w + anchors_cx;
    auto pred_cy = dy * anchors_h + anchors_cy;
    auto pred_w = exp(dw) * anchors_w;
    auto pred_h = exp(dh) * anchors_h;

    // Convert back to (x1, y1, x2, y2)
    auto pred_x1 = pred_cx - pred_w * 0.5f;
    auto pred_y1 = pred_cy - pred_h * 0.5f;
    auto pred_x2 = pred_cx + pred_w * 0.5f;
    auto pred_y2 = pred_cy + pred_h * 0.5f;

    return tenzor::cat({pred_x1, pred_y1, pred_x2, pred_y2}, 1);
}

auto clip_boxes_to_image(const Tensor& boxes, int64_t height, int64_t width) -> Tensor {
    // Clamp each coordinate separately
    auto x1 = tenzor::clamp(boxes.slice(1, 0, 1), 0.0f, static_cast<float>(width));
    auto y1 = tenzor::clamp(boxes.slice(1, 1, 2), 0.0f, static_cast<float>(height));
    auto x2 = tenzor::clamp(boxes.slice(1, 2, 3), 0.0f, static_cast<float>(width));
    auto y2 = tenzor::clamp(boxes.slice(1, 3, 4), 0.0f, static_cast<float>(height));

    return tenzor::cat({x1, y1, x2, y2}, 1);
}

auto remove_small_boxes(const Tensor& boxes, const Tensor& scores,
                        double min_size) -> Tensor {
    auto widths = boxes.slice(1, 2, 3) - boxes.slice(1, 0, 1);
    auto heights = boxes.slice(1, 3, 4) - boxes.slice(1, 1, 2);

    // Create scalar tensor for comparison
    auto width_squeezed = widths.squeeze(1);
    auto width_shape = width_squeezed.shape();
    std::vector<int64_t> shape_vec(width_shape.begin(), width_shape.end());
    auto min_size_tensor = tenzor::full(shape_vec, static_cast<float>(min_size),
                                         width_squeezed.dtype(), width_squeezed.device());

    auto valid_w = widths.squeeze(1) >= min_size_tensor;
    auto valid_h = heights.squeeze(1) >= min_size_tensor;
    auto valid = valid_w * valid_h;  // Element-wise multiplication for boolean tensors

    return valid.nonzero().squeeze(1);
}

} // namespace ops
} // namespace tenzor
