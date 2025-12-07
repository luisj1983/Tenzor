#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <vector>

namespace tenzor {
namespace adaptivecpp {

// SYCL Kernel name classes
struct NMSKernelFloat32 {};
struct NMSKernelFloat64 {};
struct ROIAlignKernelFloat32 {};
struct ROIAlignKernelFloat64 {};
struct ROIAlignBackwardKernelFloat32 {};

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

// ============================================================================
// Non-Maximum Suppression (NMS)
// ============================================================================
/**
 * @brief Non-Maximum Suppression for object detection
 *
 * Filters overlapping bounding boxes based on IoU threshold.
 *
 * @param boxes Bounding boxes: (N, 4) format [x1, y1, x2, y2]
 * @param scores Confidence scores: (N,)
 * @param iou_threshold IoU threshold for suppression
 * @return Indices of kept boxes
 */
auto nms_kernel(
    const Tensor& boxes,
    const Tensor& scores,
    float iou_threshold,
    sycl::queue& queue
) -> Tensor {
    int64_t num_boxes = boxes.shape()[0];

    if (num_boxes == 0) {
        return Tensor({0}, DType::Int64, boxes.device());
    }

    // Copy to host for NMS computation (NMS is inherently sequential)
    std::vector<float> host_boxes(num_boxes * 4);
    std::vector<float> host_scores(num_boxes);

    const float* boxes_ptr = get_data_ptr<const float>(boxes);
    const float* scores_ptr = get_data_ptr<const float>(scores);

    queue.memcpy(host_boxes.data(), boxes_ptr, num_boxes * 4 * sizeof(float)).wait();
    queue.memcpy(host_scores.data(), scores_ptr, num_boxes * sizeof(float)).wait();

    // Sort boxes by score (descending)
    std::vector<int64_t> order(num_boxes);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return host_scores[a] > host_scores[b];
    });

    // Compute areas
    std::vector<float> areas(num_boxes);
    for (int64_t i = 0; i < num_boxes; ++i) {
        float x1 = host_boxes[i * 4 + 0];
        float y1 = host_boxes[i * 4 + 1];
        float x2 = host_boxes[i * 4 + 2];
        float y2 = host_boxes[i * 4 + 3];
        areas[i] = (x2 - x1) * (y2 - y1);
    }

    // NMS
    std::vector<bool> suppressed(num_boxes, false);
    std::vector<int64_t> keep;

    for (int64_t i = 0; i < num_boxes; ++i) {
        int64_t idx = order[i];
        if (suppressed[idx]) continue;

        keep.push_back(idx);

        float x1_i = host_boxes[idx * 4 + 0];
        float y1_i = host_boxes[idx * 4 + 1];
        float x2_i = host_boxes[idx * 4 + 2];
        float y2_i = host_boxes[idx * 4 + 3];

        for (int64_t j = i + 1; j < num_boxes; ++j) {
            int64_t jdx = order[j];
            if (suppressed[jdx]) continue;

            float x1_j = host_boxes[jdx * 4 + 0];
            float y1_j = host_boxes[jdx * 4 + 1];
            float x2_j = host_boxes[jdx * 4 + 2];
            float y2_j = host_boxes[jdx * 4 + 3];

            // Compute intersection
            float xx1 = std::max(x1_i, x1_j);
            float yy1 = std::max(y1_i, y1_j);
            float xx2 = std::min(x2_i, x2_j);
            float yy2 = std::min(y2_i, y2_j);

            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float intersection = w * h;

            // Compute IoU
            float iou = intersection / (areas[idx] + areas[jdx] - intersection + 1e-6f);

            if (iou > iou_threshold) {
                suppressed[jdx] = true;
            }
        }
    }

    // Create output tensor
    Tensor result({static_cast<int64_t>(keep.size())}, DType::Int64, boxes.device());
    int64_t* result_ptr = get_data_ptr<int64_t>(result);
    queue.memcpy(result_ptr, keep.data(), keep.size() * sizeof(int64_t)).wait();

    return result;
}

// ============================================================================
// ROI Align
// ============================================================================
/**
 * @brief ROI Align for object detection (bilinear interpolation)
 *
 * Extracts fixed-size feature maps from regions of interest.
 *
 * @param features Feature map: (N, C, H, W)
 * @param rois Regions of interest: (K, 5) format [batch_idx, x1, y1, x2, y2]
 * @param output_height Output height
 * @param output_width Output width
 * @param spatial_scale Scale factor from input to feature map
 * @param sampling_ratio Number of sampling points (0 for adaptive)
 * @param aligned Whether to use aligned mode
 * @return Output feature maps: (K, C, output_height, output_width)
 */
auto roi_align_kernel(
    const Tensor& features,
    const Tensor& rois,
    int64_t output_height,
    int64_t output_width,
    float spatial_scale,
    int64_t sampling_ratio,
    bool aligned,
    sycl::queue& queue
) -> Tensor {
    auto feature_shape = features.shape();
    int64_t batch_size = feature_shape[0];
    int64_t channels = feature_shape[1];
    int64_t height = feature_shape[2];
    int64_t width = feature_shape[3];

    int64_t num_rois = rois.shape()[0];

    // Create output tensor
    Tensor output({num_rois, channels, output_height, output_width},
                  features.dtype(), features.device());

    if (num_rois == 0) {
        return output;
    }

    int64_t total_elements = num_rois * channels * output_height * output_width;

    if (features.dtype() == DType::Float32) {
        const float* features_ptr = get_data_ptr<const float>(features);
        const float* rois_ptr = get_data_ptr<const float>(rois);
        float* output_ptr = get_data_ptr<float>(output);

        const float offset = aligned ? 0.5f : 0.0f;

        queue.parallel_for<ROIAlignKernelFloat32>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t pw = idx % output_width;
                int64_t ph = (idx / output_width) % output_height;
                int64_t c = (idx / output_width / output_height) % channels;
                int64_t n = idx / output_width / output_height / channels;

                // Get ROI
                const float* roi = rois_ptr + n * 5;
                int64_t batch_idx = static_cast<int64_t>(roi[0]);
                float roi_x1 = roi[1] * spatial_scale - offset;
                float roi_y1 = roi[2] * spatial_scale - offset;
                float roi_x2 = roi[3] * spatial_scale - offset;
                float roi_y2 = roi[4] * spatial_scale - offset;

                float roi_width = roi_x2 - roi_x1;
                float roi_height = roi_y2 - roi_y1;
                if (!aligned) {
                    roi_width = sycl::fmax(roi_width, 1.0f);
                    roi_height = sycl::fmax(roi_height, 1.0f);
                }

                float bin_size_h = roi_height / static_cast<float>(output_height);
                float bin_size_w = roi_width / static_cast<float>(output_width);

                // Adaptive sampling
                int64_t roi_bin_grid_h = sampling_ratio > 0 ? sampling_ratio
                    : static_cast<int64_t>(sycl::ceil(roi_height / output_height));
                int64_t roi_bin_grid_w = sampling_ratio > 0 ? sampling_ratio
                    : static_cast<int64_t>(sycl::ceil(roi_width / output_width));

                const float count = roi_bin_grid_h * roi_bin_grid_w;

                float output_val = 0.0f;

                for (int64_t iy = 0; iy < roi_bin_grid_h; ++iy) {
                    float y = roi_y1 + ph * bin_size_h + (iy + 0.5f) * bin_size_h / roi_bin_grid_h;

                    for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
                        float x = roi_x1 + pw * bin_size_w + (ix + 0.5f) * bin_size_w / roi_bin_grid_w;

                        // Bilinear interpolation
                        if (y < -1.0f || y > height || x < -1.0f || x > width) {
                            continue;
                        }

                        y = sycl::fmax(y, 0.0f);
                        x = sycl::fmax(x, 0.0f);

                        int64_t y_low = static_cast<int64_t>(y);
                        int64_t x_low = static_cast<int64_t>(x);
                        int64_t y_high = y_low + 1;
                        int64_t x_high = x_low + 1;

                        if (y_low >= height - 1) {
                            y_high = y_low = height - 1;
                            y = static_cast<float>(y_low);
                        }
                        if (x_low >= width - 1) {
                            x_high = x_low = width - 1;
                            x = static_cast<float>(x_low);
                        }

                        float ly = y - y_low;
                        float lx = x - x_low;
                        float hy = 1.0f - ly;
                        float hx = 1.0f - lx;

                        int64_t base = batch_idx * channels * height * width + c * height * width;

                        float v1 = features_ptr[base + y_low * width + x_low];
                        float v2 = features_ptr[base + y_low * width + x_high];
                        float v3 = features_ptr[base + y_high * width + x_low];
                        float v4 = features_ptr[base + y_high * width + x_high];

                        float w1 = hy * hx;
                        float w2 = hy * lx;
                        float w3 = ly * hx;
                        float w4 = ly * lx;

                        output_val += w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
                    }
                }
                output_val /= count;

                output_ptr[idx] = output_val;
            }
        ).wait();
    }
    else if (features.dtype() == DType::Float64) {
        const double* features_ptr = get_data_ptr<const double>(features);
        const double* rois_ptr = get_data_ptr<const double>(rois);
        double* output_ptr = get_data_ptr<double>(output);

        const double offset = aligned ? 0.5 : 0.0;

        queue.parallel_for<ROIAlignKernelFloat64>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t pw = idx % output_width;
                int64_t ph = (idx / output_width) % output_height;
                int64_t c = (idx / output_width / output_height) % channels;
                int64_t n = idx / output_width / output_height / channels;

                const double* roi = rois_ptr + n * 5;
                int64_t batch_idx = static_cast<int64_t>(roi[0]);
                double roi_x1 = roi[1] * spatial_scale - offset;
                double roi_y1 = roi[2] * spatial_scale - offset;
                double roi_x2 = roi[3] * spatial_scale - offset;
                double roi_y2 = roi[4] * spatial_scale - offset;

                double roi_width = roi_x2 - roi_x1;
                double roi_height = roi_y2 - roi_y1;
                if (!aligned) {
                    roi_width = sycl::fmax(roi_width, 1.0);
                    roi_height = sycl::fmax(roi_height, 1.0);
                }

                double bin_size_h = roi_height / static_cast<double>(output_height);
                double bin_size_w = roi_width / static_cast<double>(output_width);

                int64_t roi_bin_grid_h = sampling_ratio > 0 ? sampling_ratio
                    : static_cast<int64_t>(sycl::ceil(roi_height / output_height));
                int64_t roi_bin_grid_w = sampling_ratio > 0 ? sampling_ratio
                    : static_cast<int64_t>(sycl::ceil(roi_width / output_width));

                const double count = roi_bin_grid_h * roi_bin_grid_w;

                double output_val = 0.0;

                for (int64_t iy = 0; iy < roi_bin_grid_h; ++iy) {
                    double y = roi_y1 + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h;

                    for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
                        double x = roi_x1 + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w;

                        if (y < -1.0 || y > height || x < -1.0 || x > width) {
                            continue;
                        }

                        y = sycl::fmax(y, 0.0);
                        x = sycl::fmax(x, 0.0);

                        int64_t y_low = static_cast<int64_t>(y);
                        int64_t x_low = static_cast<int64_t>(x);
                        int64_t y_high = y_low + 1;
                        int64_t x_high = x_low + 1;

                        if (y_low >= height - 1) {
                            y_high = y_low = height - 1;
                            y = static_cast<double>(y_low);
                        }
                        if (x_low >= width - 1) {
                            x_high = x_low = width - 1;
                            x = static_cast<double>(x_low);
                        }

                        double ly = y - y_low;
                        double lx = x - x_low;
                        double hy = 1.0 - ly;
                        double hx = 1.0 - lx;

                        int64_t base = batch_idx * channels * height * width + c * height * width;

                        double v1 = features_ptr[base + y_low * width + x_low];
                        double v2 = features_ptr[base + y_low * width + x_high];
                        double v3 = features_ptr[base + y_high * width + x_low];
                        double v4 = features_ptr[base + y_high * width + x_high];

                        double w1 = hy * hx;
                        double w2 = hy * lx;
                        double w3 = ly * hx;
                        double w4 = ly * lx;

                        output_val += w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
                    }
                }
                output_val /= count;

                output_ptr[idx] = output_val;
            }
        ).wait();
    }
    else {
        throw std::runtime_error("roi_align: unsupported dtype");
    }

    return output;
}

} // namespace adaptivecpp
} // namespace tenzor
