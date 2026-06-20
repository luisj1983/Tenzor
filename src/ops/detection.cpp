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
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif


// SIMD headers for optimized NMS
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace tenzor {
namespace ops {

// ============================================================================
// Optimized NMS Helper Functions
// ============================================================================

namespace {

#if defined(__AVX2__) || defined(TENZOR_HAS_AVX512)

/**
 * @brief Compute IoU between one reference box and 8 candidate boxes using AVX2
 */
inline __m256 compute_iou_avx2(
    __m256 ref_x1, __m256 ref_y1, __m256 ref_x2, __m256 ref_y2, __m256 ref_area,
    __m256 cand_x1, __m256 cand_y1, __m256 cand_x2, __m256 cand_y2
) {
    // Compute intersection coordinates
    __m256 inter_x1 = _mm256_max_ps(ref_x1, cand_x1);
    __m256 inter_y1 = _mm256_max_ps(ref_y1, cand_y1);
    __m256 inter_x2 = _mm256_min_ps(ref_x2, cand_x2);
    __m256 inter_y2 = _mm256_min_ps(ref_y2, cand_y2);

    // Compute intersection dimensions (clamped to 0)
    __m256 zero = _mm256_setzero_ps();
    __m256 inter_w = _mm256_max_ps(zero, _mm256_sub_ps(inter_x2, inter_x1));
    __m256 inter_h = _mm256_max_ps(zero, _mm256_sub_ps(inter_y2, inter_y1));

    // Compute intersection area
    __m256 inter_area = _mm256_mul_ps(inter_w, inter_h);

    // Compute candidate areas
    __m256 cand_w = _mm256_sub_ps(cand_x2, cand_x1);
    __m256 cand_h = _mm256_sub_ps(cand_y2, cand_y1);
    __m256 cand_area = _mm256_mul_ps(cand_w, cand_h);

    // Compute union area (ref_area + cand_area - inter_area)
    __m256 union_area = _mm256_add_ps(ref_area, cand_area);
    union_area = _mm256_sub_ps(union_area, inter_area);

    // Add small epsilon to avoid division by zero
    __m256 eps = _mm256_set1_ps(1e-7f);
    union_area = _mm256_add_ps(union_area, eps);

    // Compute IoU = inter_area / union_area
    return _mm256_div_ps(inter_area, union_area);
}

/**
 * @brief SIMD-accelerated batch IoU suppression check
 */
inline void suppress_batch_avx2(
    const float* boxes,
    const int64_t* indices,
    uint8_t* suppressed,
    int64_t ref_idx,
    int64_t start_j,
    int64_t end_j,
    float iou_threshold
) {
    // Load reference box (broadcast)
    const float ref_x1 = boxes[ref_idx * 4 + 0];
    const float ref_y1 = boxes[ref_idx * 4 + 1];
    const float ref_x2 = boxes[ref_idx * 4 + 2];
    const float ref_y2 = boxes[ref_idx * 4 + 3];
    const float ref_area = (ref_x2 - ref_x1) * (ref_y2 - ref_y1);

    __m256 vref_x1 = _mm256_set1_ps(ref_x1);
    __m256 vref_y1 = _mm256_set1_ps(ref_y1);
    __m256 vref_x2 = _mm256_set1_ps(ref_x2);
    __m256 vref_y2 = _mm256_set1_ps(ref_y2);
    __m256 vref_area = _mm256_set1_ps(ref_area);
    __m256 vthreshold = _mm256_set1_ps(iou_threshold);

    int64_t j = start_j;

    // Process 8 candidates at a time
    for (; j + 8 <= end_j; j += 8) {
        // Gather box coordinates for 8 candidates
        alignas(32) float cand_x1[8], cand_y1[8], cand_x2[8], cand_y2[8];

        for (int k = 0; k < 8; ++k) {
            int64_t idx = indices[j + k];
            cand_x1[k] = boxes[idx * 4 + 0];
            cand_y1[k] = boxes[idx * 4 + 1];
            cand_x2[k] = boxes[idx * 4 + 2];
            cand_y2[k] = boxes[idx * 4 + 3];
        }

        // Load candidate coordinates
        __m256 vcand_x1 = _mm256_load_ps(cand_x1);
        __m256 vcand_y1 = _mm256_load_ps(cand_y1);
        __m256 vcand_x2 = _mm256_load_ps(cand_x2);
        __m256 vcand_y2 = _mm256_load_ps(cand_y2);

        // Compute IoU for 8 candidates
        __m256 iou = compute_iou_avx2(
            vref_x1, vref_y1, vref_x2, vref_y2, vref_area,
            vcand_x1, vcand_y1, vcand_x2, vcand_y2
        );

        // Compare with threshold
        __m256 mask = _mm256_cmp_ps(iou, vthreshold, _CMP_GT_OQ);
        int suppress_mask = _mm256_movemask_ps(mask);

        // Apply suppression
        for (int k = 0; k < 8; ++k) {
            if ((suppress_mask >> k) & 1) {
                suppressed[indices[j + k]] = 1;
            }
        }
    }

    // Handle remaining candidates (scalar fallback)
    for (; j < end_j; ++j) {
        int64_t idx2 = indices[j];
        if (suppressed[idx2]) continue;

        const float x1_2 = boxes[idx2 * 4 + 0];
        const float y1_2 = boxes[idx2 * 4 + 1];
        const float x2_2 = boxes[idx2 * 4 + 2];
        const float y2_2 = boxes[idx2 * 4 + 3];

        const float inter_x1 = std::max(ref_x1, x1_2);
        const float inter_y1 = std::max(ref_y1, y1_2);
        const float inter_x2 = std::min(ref_x2, x2_2);
        const float inter_y2 = std::min(ref_y2, y2_2);

        const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
        const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
        const float inter_area = inter_w * inter_h;

        const float area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
        const float union_area = ref_area + area2 - inter_area;
        const float iou = inter_area / (union_area + 1e-7f);

        if (iou > iou_threshold) {
            suppressed[idx2] = 1;
        }
    }
}

#endif // __AVX2__

} // anonymous namespace

auto box_iou(const Tensor& boxes1, const Tensor& boxes2, IoUType iou_type) -> Tensor {
    if (boxes1.ndim() != 2 || boxes2.ndim() != 2) {
        throw std::invalid_argument("Boxes must be 2D tensors");
    }
    if (boxes1.shape()[1] != 4 || boxes2.shape()[1] != 4) {
        throw std::invalid_argument("Boxes must have 4 coordinates");
    }

    const int64_t N = boxes1.shape()[0];
    const int64_t M = boxes2.shape()[0];

    // For Float16 on any device, convert to Float32 for numerical stability
    // (Float16 has limited precision and some tensor ops may not support it well)
    if (boxes1.dtype() == DType::Float16 || boxes2.dtype() == DType::Float16) {
        auto boxes1_f32 = boxes1.to(DType::Float32);
        auto boxes2_f32 = boxes2.to(DType::Float32);
        auto result_f32 = box_iou(boxes1_f32, boxes2_f32, iou_type);
        return result_f32.to(boxes1.dtype());
    }

    // For non-CPU devices, dispatch to registered backend kernel. Force
    // contiguous: a non-contiguous column slice (e.g. proposals[:, 1:5]) would
    // otherwise be indexed assuming a packed 4-per-row layout by the kernel.
    if (boxes1.device().type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::IouType, static_cast<int>(iou_type));
        std::vector<Tensor> inputs_vec = {boxes1.contiguous(), boxes2.contiguous()};
        auto results = dispatch<OpId::BoxIoU>(inputs_vec, attrs);
        return results[0];
    }

    // CPU implementation using manual loops for reliability
    {
        // Remember original dtype for output
        DType original_dtype = boxes1.dtype();

        // Convert to Float32 for computation. .contiguous() is essential: callers
        // routinely pass a non-contiguous column slice (e.g. proposals[:, 1:5]
        // with row stride 5). Without it, .to(Float32) is a no-op view and the
        // i*4 row indexing below reads progressively shifted garbage for every
        // row past the first — silently corrupting the IoU matrix.
        auto boxes1_f32 = boxes1.to(DType::Float32).contiguous();
        auto boxes2_f32 = boxes2.to(DType::Float32).contiguous();

        auto result = zeros({N, M}, DType::Float32, Device::cpu());
        const float* boxes1_data = static_cast<const float*>(boxes1_f32.data_ptr());
        const float* boxes2_data = static_cast<const float*>(boxes2_f32.data_ptr());
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

        // Convert back to original dtype
        return result.to(original_dtype);
    }
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

    // GPU fast path: dispatch NMS to registered backend kernel (CUDA, ROCm, OneAPI, Vulkan)
    if (boxes.device().type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::IouThreshold, static_cast<float>(iou_threshold));
        auto boxes_f32 = boxes.to(DType::Float32).contiguous();
        auto scores_f32 = scores.to(DType::Float32).contiguous();
        std::vector<Tensor> nms_inputs = {boxes_f32, scores_f32};
        auto result = dispatch<OpId::NMS>(nms_inputs, attrs);
        return result[0];
    }

    // Move to CPU and convert to Float32 for processing
    auto boxes_cpu = boxes.to(Device::cpu()).to(DType::Float32);
    auto scores_cpu = scores.to(Device::cpu()).to(DType::Float32);

    // Get data pointers
    const float* boxes_data = boxes_cpu.data<float>();
    const float* scores_data = scores_cpu.data<float>();

    // Sort indices by score (descending)
    std::vector<int64_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [scores_data](int64_t i, int64_t j) {
                  return scores_data[i] > scores_data[j];
              });

    // Initialize suppression mask (using uint8_t for SIMD compatibility)
    std::vector<uint8_t> suppressed(N, 0);
    std::vector<int64_t> keep;
    keep.reserve(N);

    // Main NMS loop with SIMD acceleration
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

        // Number of remaining candidates to check
        int64_t remaining = N - i - 1;

#if defined(__AVX2__) || defined(TENZOR_HAS_AVX512)
        // Use SIMD for batch suppression when we have enough candidates
        if (remaining >= 8) {
            suppress_batch_avx2(
                boxes_data, indices.data(), suppressed.data(),
                idx, i + 1, N, static_cast<float>(iou_threshold)
            );
            continue;  // All remaining candidates have been processed
        }
#endif

        // Scalar fallback (or for small remaining sets)
        #pragma omp parallel for schedule(dynamic, 64) if(remaining > 1024)
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
                suppressed[idx2] = 1;
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

    const int64_t num_classes = scores.shape()[1];

    std::vector<Tensor> all_boxes_tensors;
    std::vector<Tensor> all_scores_tensors;
    std::vector<Tensor> all_labels_tensors;

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

        // Collect results on-device (no CPU roundtrip)
        auto kept_boxes = tenzor::index_select(filtered_boxes, 0, keep);
        auto kept_scores = tenzor::index_select(filtered_scores, 0, keep);
        auto kept_labels = tenzor::full({num_keep}, static_cast<double>(cls),
                                         DType::Int64, boxes.device());

        all_boxes_tensors.push_back(kept_boxes);
        all_scores_tensors.push_back(kept_scores);
        all_labels_tensors.push_back(kept_labels);
    }

    // Handle empty results (all classes filtered or suppressed)
    if (all_boxes_tensors.empty()) {
        return std::make_tuple(
            tenzor::zeros({0, 4}, DType::Float32, boxes.device()),
            tenzor::zeros({0}, DType::Float32, boxes.device()),
            tenzor::zeros({0}, DType::Int64, boxes.device()));
    }

    // Concatenate all results on-device (no CPU roundtrip)
    auto result_boxes = tenzor::cat(all_boxes_tensors, 0).to(DType::Float32);
    auto result_scores = tenzor::cat(all_scores_tensors, 0).to(DType::Float32);
    auto result_labels = tenzor::cat(all_labels_tensors, 0);

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
    // Clamp width/height deltas before exp to avoid overflow to +inf on large
    // (e.g. untrained/random) deltas — the standard Faster/Mask-RCNN
    // bbox_xform_clip = log(1000/16). Without it, exp(dw) can become inf and
    // poison every downstream box coordinate.
    const double bbox_xform_clip = std::log(1000.0 / 16.0);
    auto pred_w = exp(clamp(dw, -bbox_xform_clip, bbox_xform_clip)) * anchors_w;
    auto pred_h = exp(clamp(dh, -bbox_xform_clip, bbox_xform_clip)) * anchors_h;

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

auto remove_small_boxes(const Tensor& boxes, [[maybe_unused]] const Tensor& scores,
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
