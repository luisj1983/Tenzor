#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <vector>

namespace tenzor {
namespace oneapi {

// SYCL Kernel name classes
struct NMSKernelFloat32 {};
struct NMSKernelFloat64 {};
struct ROIAlignKernelFloat32 {};
struct ROIAlignKernelFloat64 {};
struct ROIAlignKernelFloat16 {};
struct ROIAlignBackwardKernelFloat32 {};
struct ROIAlignBackwardKernelFloat64 {};
struct ROIAlignBackwardKernelFloat16 {};
struct ROIAlignBackwardF16AccumKernel {};
struct ROIAlignBackwardF16ConvertKernel {};
struct GatherRelativePositionBiasKernelFloat32 {};
struct GatherRelativePositionBiasKernelFloat64 {};
struct GatherRelativePositionBiasKernelFloat16 {};
struct InterpolateNearestKernelFloat32 {};
struct InterpolateNearestKernelFloat64 {};
struct InterpolateNearestKernelFloat16 {};
struct InterpolateBilinearKernelFloat32 {};
struct InterpolateBilinearKernelFloat64 {};
struct InterpolateBilinearKernelFloat16 {};
struct InterpolateBicubicKernelFloat32 {};
struct InterpolateBicubicKernelFloat64 {};
struct InterpolateBicubicKernelFloat16 {};

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

    if (boxes.dtype() == DType::Float16) {
        // Float16: copy as half, then convert to float on host
        std::vector<sycl::half> host_boxes_h(num_boxes * 4);
        std::vector<sycl::half> host_scores_h(num_boxes);
        const sycl::half* boxes_ptr_h = get_data_ptr<const sycl::half>(boxes);
        const sycl::half* scores_ptr_h = get_data_ptr<const sycl::half>(scores);
        queue.memcpy(host_boxes_h.data(), boxes_ptr_h, num_boxes * 4 * sizeof(sycl::half)).wait();
        queue.memcpy(host_scores_h.data(), scores_ptr_h, num_boxes * sizeof(sycl::half)).wait();
        for (int64_t i = 0; i < num_boxes * 4; ++i) host_boxes[i] = static_cast<float>(host_boxes_h[i]);
        for (int64_t i = 0; i < num_boxes; ++i) host_scores[i] = static_cast<float>(host_scores_h[i]);
    } else if (boxes.dtype() == DType::Float64) {
        std::vector<double> host_boxes_d(num_boxes * 4);
        std::vector<double> host_scores_d(num_boxes);
        const double* boxes_ptr_d = get_data_ptr<const double>(boxes);
        const double* scores_ptr_d = get_data_ptr<const double>(scores);
        queue.memcpy(host_boxes_d.data(), boxes_ptr_d, num_boxes * 4 * sizeof(double)).wait();
        queue.memcpy(host_scores_d.data(), scores_ptr_d, num_boxes * sizeof(double)).wait();
        for (int64_t i = 0; i < num_boxes * 4; ++i) host_boxes[i] = static_cast<float>(host_boxes_d[i]);
        for (int64_t i = 0; i < num_boxes; ++i) host_scores[i] = static_cast<float>(host_scores_d[i]);
    } else {
        const float* boxes_ptr = get_data_ptr<const float>(boxes);
        const float* scores_ptr = get_data_ptr<const float>(scores);
        queue.memcpy(host_boxes.data(), boxes_ptr, num_boxes * 4 * sizeof(float)).wait();
        queue.memcpy(host_scores.data(), scores_ptr, num_boxes * sizeof(float)).wait();
    }

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
        );
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
        );
    }
    else if (features.dtype() == DType::Float16) {
        const sycl::half* features_ptr = get_data_ptr<const sycl::half>(features);
        const sycl::half* rois_ptr = get_data_ptr<const sycl::half>(rois);
        sycl::half* output_ptr = get_data_ptr<sycl::half>(output);

        const float offset = aligned ? 0.5f : 0.0f;

        queue.parallel_for<ROIAlignKernelFloat16>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t pw = idx % output_width;
                int64_t ph = (idx / output_width) % output_height;
                int64_t c = (idx / output_width / output_height) % channels;
                int64_t n = idx / output_width / output_height / channels;

                const sycl::half* roi = rois_ptr + n * 5;
                int64_t batch_idx = static_cast<int64_t>(float(roi[0]));
                float roi_x1 = float(roi[1]) * spatial_scale - offset;
                float roi_y1 = float(roi[2]) * spatial_scale - offset;
                float roi_x2 = float(roi[3]) * spatial_scale - offset;
                float roi_y2 = float(roi[4]) * spatial_scale - offset;

                float roi_width = roi_x2 - roi_x1;
                float roi_height = roi_y2 - roi_y1;
                if (!aligned) {
                    roi_width = sycl::fmax(roi_width, 1.0f);
                    roi_height = sycl::fmax(roi_height, 1.0f);
                }

                float bin_size_h = roi_height / static_cast<float>(output_height);
                float bin_size_w = roi_width / static_cast<float>(output_width);

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

                        float v1 = float(features_ptr[base + y_low * width + x_low]);
                        float v2 = float(features_ptr[base + y_low * width + x_high]);
                        float v3 = float(features_ptr[base + y_high * width + x_low]);
                        float v4 = float(features_ptr[base + y_high * width + x_high]);

                        float w1 = hy * hx;
                        float w2 = hy * lx;
                        float w3 = ly * hx;
                        float w4 = ly * lx;

                        output_val += w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
                    }
                }
                output_val /= count;

                output_ptr[idx] = sycl::half(output_val);
            }
        );
    }
    else {
        throw std::runtime_error("roi_align: unsupported dtype");
    }

    return output;
}

// ============================================================================
// ROI Align Backward
// ============================================================================
/**
 * @brief ROI Align backward for gradient computation
 *
 * Distributes gradients back to feature map positions using bilinear weights.
 *
 * @param grad_output Gradient from next layer: (K, C, output_height, output_width)
 * @param rois Regions of interest: (K, 5) format [batch_idx, x1, y1, x2, y2]
 * @param batch_size Number of batches in original feature map
 * @param channels Number of channels
 * @param feat_height Feature map height
 * @param feat_width Feature map width
 * @param spatial_scale Scale factor from input to feature map
 * @param sampling_ratio Number of sampling points (0 for adaptive)
 * @param aligned Whether to use aligned mode
 * @param queue SYCL queue for execution
 * @return Gradient w.r.t. features: (batch_size, channels, feat_height, feat_width)
 */
auto roi_align_backward_kernel(
    const Tensor& grad_output,
    const Tensor& rois,
    int64_t batch_size,
    int64_t channels,
    int64_t feat_height,
    int64_t feat_width,
    float spatial_scale,
    int64_t sampling_ratio,
    bool aligned,
    sycl::queue& queue
) -> Tensor {
    auto grad_shape = grad_output.shape();
    int64_t num_rois = grad_shape[0];
    int64_t output_height = grad_shape[2];
    int64_t output_width = grad_shape[3];

    Tensor grad_features({batch_size, channels, feat_height, feat_width},
                         grad_output.dtype(), grad_output.device());

    int64_t feat_size = batch_size * channels * feat_height * feat_width;
    int64_t total_elements = num_rois * channels * output_height * output_width;

    if (num_rois == 0 || total_elements == 0) {
        // Zero out and return
        if (grad_output.dtype() == DType::Float32) {
            queue.fill(get_data_ptr<float>(grad_features), 0.0f, feat_size);
        } else {
            queue.fill(get_data_ptr<double>(grad_features), 0.0, feat_size);
        }
        return grad_features;
    }

    if (grad_output.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        const float* rois_ptr = get_data_ptr<const float>(rois);
        float* grad_feat_ptr = get_data_ptr<float>(grad_features);

        queue.fill(grad_feat_ptr, 0.0f, feat_size);

        const float offset = aligned ? 0.5f : 0.0f;

        queue.parallel_for<ROIAlignBackwardKernelFloat32>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t pw = idx % output_width;
                int64_t ph = (idx / output_width) % output_height;
                int64_t c = (idx / output_width / output_height) % channels;
                int64_t n = idx / output_width / output_height / channels;

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

                int64_t roi_bin_grid_h = sampling_ratio > 0 ? sampling_ratio
                    : static_cast<int64_t>(sycl::ceil(roi_height / output_height));
                int64_t roi_bin_grid_w = sampling_ratio > 0 ? sampling_ratio
                    : static_cast<int64_t>(sycl::ceil(roi_width / output_width));

                const float count = roi_bin_grid_h * roi_bin_grid_w;
                const float grad_val = grad_out_ptr[idx] / count;

                for (int64_t iy = 0; iy < roi_bin_grid_h; ++iy) {
                    float y = roi_y1 + ph * bin_size_h + (iy + 0.5f) * bin_size_h / roi_bin_grid_h;
                    for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
                        float x = roi_x1 + pw * bin_size_w + (ix + 0.5f) * bin_size_w / roi_bin_grid_w;

                        if (y < -1.0f || y > feat_height || x < -1.0f || x > feat_width) continue;

                        y = sycl::fmax(y, 0.0f);
                        x = sycl::fmax(x, 0.0f);

                        int64_t y_low = static_cast<int64_t>(y);
                        int64_t x_low = static_cast<int64_t>(x);
                        int64_t y_high = y_low + 1;
                        int64_t x_high = x_low + 1;

                        if (y_low >= feat_height - 1) { y_high = y_low = feat_height - 1; y = static_cast<float>(y_low); }
                        if (x_low >= feat_width - 1) { x_high = x_low = feat_width - 1; x = static_cast<float>(x_low); }

                        float ly = y - y_low;
                        float lx = x - x_low;
                        float hy = 1.0f - ly;
                        float hx = 1.0f - lx;

                        int64_t base = batch_idx * channels * feat_height * feat_width + c * feat_height * feat_width;

                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                            a1(grad_feat_ptr[base + y_low * feat_width + x_low]);
                        a1 += hy * hx * grad_val;

                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                            a2(grad_feat_ptr[base + y_low * feat_width + x_high]);
                        a2 += hy * lx * grad_val;

                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                            a3(grad_feat_ptr[base + y_high * feat_width + x_low]);
                        a3 += ly * hx * grad_val;

                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                            a4(grad_feat_ptr[base + y_high * feat_width + x_high]);
                        a4 += ly * lx * grad_val;
                    }
                }
            }
        );
    }
    else if (grad_output.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* rois_ptr = get_data_ptr<const double>(rois);
        double* grad_feat_ptr = get_data_ptr<double>(grad_features);

        queue.fill(grad_feat_ptr, 0.0, feat_size);

        const double offset = aligned ? 0.5 : 0.0;

        queue.parallel_for<ROIAlignBackwardKernelFloat64>(
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
                const double grad_val = grad_out_ptr[idx] / count;

                for (int64_t iy = 0; iy < roi_bin_grid_h; ++iy) {
                    double y = roi_y1 + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h;
                    for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
                        double x = roi_x1 + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w;

                        if (y < -1.0 || y > feat_height || x < -1.0 || x > feat_width) continue;

                        y = sycl::fmax(y, 0.0);
                        x = sycl::fmax(x, 0.0);

                        int64_t y_low = static_cast<int64_t>(y);
                        int64_t x_low = static_cast<int64_t>(x);
                        int64_t y_high = y_low + 1;
                        int64_t x_high = x_low + 1;

                        if (y_low >= feat_height - 1) { y_high = y_low = feat_height - 1; y = static_cast<double>(y_low); }
                        if (x_low >= feat_width - 1) { x_high = x_low = feat_width - 1; x = static_cast<double>(x_low); }

                        double ly = y - y_low;
                        double lx = x - x_low;
                        double hy = 1.0 - ly;
                        double hx = 1.0 - lx;

                        int64_t base = batch_idx * channels * feat_height * feat_width + c * feat_height * feat_width;

                        sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device>
                            a1(grad_feat_ptr[base + y_low * feat_width + x_low]);
                        a1 += hy * hx * grad_val;

                        sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device>
                            a2(grad_feat_ptr[base + y_low * feat_width + x_high]);
                        a2 += hy * lx * grad_val;

                        sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device>
                            a3(grad_feat_ptr[base + y_high * feat_width + x_low]);
                        a3 += ly * hx * grad_val;

                        sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device>
                            a4(grad_feat_ptr[base + y_high * feat_width + x_high]);
                        a4 += ly * lx * grad_val;
                    }
                }
            }
        );
    }
    else if (grad_output.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        const sycl::half* rois_ptr = get_data_ptr<const sycl::half>(rois);
        sycl::half* grad_feat_ptr = get_data_ptr<sycl::half>(grad_features);

        // Accumulate in float32 since sycl::half doesn't support atomic operations
        float* accum_ptr = sycl::malloc_device<float>(feat_size, queue);
        queue.fill(accum_ptr, 0.0f, feat_size);

        const float offset = aligned ? 0.5f : 0.0f;

        queue.parallel_for<ROIAlignBackwardF16AccumKernel>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t pw = idx % output_width;
                int64_t ph = (idx / output_width) % output_height;
                int64_t c = (idx / output_width / output_height) % channels;
                int64_t n = idx / output_width / output_height / channels;

                const sycl::half* roi = rois_ptr + n * 5;
                int64_t batch_idx = static_cast<int64_t>(float(roi[0]));
                float roi_x1 = float(roi[1]) * spatial_scale - offset;
                float roi_y1 = float(roi[2]) * spatial_scale - offset;
                float roi_x2 = float(roi[3]) * spatial_scale - offset;
                float roi_y2 = float(roi[4]) * spatial_scale - offset;

                float roi_width = roi_x2 - roi_x1;
                float roi_height = roi_y2 - roi_y1;
                if (!aligned) {
                    roi_width = sycl::fmax(roi_width, 1.0f);
                    roi_height = sycl::fmax(roi_height, 1.0f);
                }

                float bin_size_h = roi_height / static_cast<float>(output_height);
                float bin_size_w = roi_width / static_cast<float>(output_width);

                int64_t roi_bin_grid_h = sampling_ratio > 0 ? sampling_ratio
                    : static_cast<int64_t>(sycl::ceil(roi_height / output_height));
                int64_t roi_bin_grid_w = sampling_ratio > 0 ? sampling_ratio
                    : static_cast<int64_t>(sycl::ceil(roi_width / output_width));

                const float count = roi_bin_grid_h * roi_bin_grid_w;
                const float grad_val = float(grad_out_ptr[idx]) / count;

                for (int64_t iy = 0; iy < roi_bin_grid_h; ++iy) {
                    float y = roi_y1 + ph * bin_size_h + (iy + 0.5f) * bin_size_h / roi_bin_grid_h;
                    for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
                        float x = roi_x1 + pw * bin_size_w + (ix + 0.5f) * bin_size_w / roi_bin_grid_w;

                        if (y < -1.0f || y > feat_height || x < -1.0f || x > feat_width) continue;

                        y = sycl::fmax(y, 0.0f);
                        x = sycl::fmax(x, 0.0f);

                        int64_t y_low = static_cast<int64_t>(y);
                        int64_t x_low = static_cast<int64_t>(x);
                        int64_t y_high = y_low + 1;
                        int64_t x_high = x_low + 1;

                        if (y_low >= feat_height - 1) { y_high = y_low = feat_height - 1; y = static_cast<float>(y_low); }
                        if (x_low >= feat_width - 1) { x_high = x_low = feat_width - 1; x = static_cast<float>(x_low); }

                        float ly = y - y_low;
                        float lx = x - x_low;
                        float hy = 1.0f - ly;
                        float hx = 1.0f - lx;

                        int64_t base = batch_idx * channels * feat_height * feat_width + c * feat_height * feat_width;

                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                            a1(accum_ptr[base + y_low * feat_width + x_low]);
                        a1 += hy * hx * grad_val;

                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                            a2(accum_ptr[base + y_low * feat_width + x_high]);
                        a2 += hy * lx * grad_val;

                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                            a3(accum_ptr[base + y_high * feat_width + x_low]);
                        a3 += ly * hx * grad_val;

                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device>
                            a4(accum_ptr[base + y_high * feat_width + x_high]);
                        a4 += ly * lx * grad_val;
                    }
                }
            }
        );

        // Convert float32 accumulation buffer back to half
        queue.parallel_for<ROIAlignBackwardF16ConvertKernel>(
            sycl::range<1>(feat_size),
            [=](sycl::id<1> i) {
                grad_feat_ptr[i] = sycl::half(accum_ptr[i]);
            }
        ).wait();

        sycl::free(accum_ptr, queue);
    }
    else {
        throw std::runtime_error("roi_align_backward: unsupported dtype");
    }

    return grad_features;
}

// ============================================================================
// Gather operation for relative position bias (Swin Transformer)
// ============================================================================
/**
 * @brief Gather relative position bias values from a table using indices
 *
 * Used in Swin Transformer to compute relative position bias for attention.
 * Gathers values from a 2D table based on position indices.
 *
 * @param table Position bias table: (table_size*table_size, num_heads)
 * @param indices Position indices: (num_positions, num_positions)
 * @param num_positions Number of positions (window_size * window_size)
 * @param num_heads Number of attention heads
 * @param queue SYCL queue for execution
 * @return Output tensor: (num_positions, num_positions, num_heads)
 */
auto gather_relative_position_bias_kernel(
    const Tensor& table,
    const Tensor& indices,
    int64_t num_positions,
    int64_t num_heads,
    sycl::queue& queue
) -> Tensor {
    // table: [table_size*table_size, num_heads]
    // indices: [num_positions, num_positions]
    // output: [num_positions, num_positions, num_heads]

    Tensor output({num_positions, num_positions, num_heads}, table.dtype(), table.device());

    int64_t total = num_positions * num_positions * num_heads;

    if (table.dtype() == DType::Float32) {
        const float* table_ptr = get_data_ptr<const float>(table);
        const int64_t* indices_ptr = get_data_ptr<const int64_t>(indices);
        float* output_ptr = get_data_ptr<float>(output);

        queue.parallel_for<GatherRelativePositionBiasKernelFloat32>(
            sycl::range<1>(total),
            [=](sycl::id<1> idx) {
                int64_t flat_idx = idx[0];
                int64_t h = flat_idx % num_heads;
                int64_t j = (flat_idx / num_heads) % num_positions;
                int64_t i = flat_idx / (num_heads * num_positions);

                int64_t table_idx = indices_ptr[i * num_positions + j];
                output_ptr[flat_idx] = table_ptr[table_idx * num_heads + h];
            }
        );
    }
    else if (table.dtype() == DType::Float64) {
        const double* table_ptr = get_data_ptr<const double>(table);
        const int64_t* indices_ptr = get_data_ptr<const int64_t>(indices);
        double* output_ptr = get_data_ptr<double>(output);

        queue.parallel_for<GatherRelativePositionBiasKernelFloat64>(
            sycl::range<1>(total),
            [=](sycl::id<1> idx) {
                int64_t flat_idx = idx[0];
                int64_t h = flat_idx % num_heads;
                int64_t j = (flat_idx / num_heads) % num_positions;
                int64_t i = flat_idx / (num_heads * num_positions);

                int64_t table_idx = indices_ptr[i * num_positions + j];
                output_ptr[flat_idx] = table_ptr[table_idx * num_heads + h];
            }
        );
    }
    else if (table.dtype() == DType::Float16) {
        const sycl::half* table_ptr = get_data_ptr<const sycl::half>(table);
        const int64_t* indices_ptr = get_data_ptr<const int64_t>(indices);
        sycl::half* output_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<GatherRelativePositionBiasKernelFloat16>(
            sycl::range<1>(total),
            [=](sycl::id<1> idx) {
                int64_t flat_idx = idx[0];
                int64_t h = flat_idx % num_heads;
                int64_t j = (flat_idx / num_heads) % num_positions;
                int64_t i = flat_idx / (num_heads * num_positions);

                int64_t table_idx = indices_ptr[i * num_positions + j];
                output_ptr[flat_idx] = table_ptr[table_idx * num_heads + h];
            }
        );
    }
    else {
        throw std::runtime_error("gather_relative_position_bias: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Interpolate (resize) operation
// ============================================================================

/**
 * @brief Cubic interpolation weight using Catmull-Rom spline
 */
inline float cubic_weight(float x) {
    x = std::abs(x);
    if (x < 1.0f) {
        return ((1.5f * x - 2.5f) * x) * x + 1.0f;
    } else if (x < 2.0f) {
        return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
    }
    return 0.0f;
}

/**
 * @brief Interpolate (resize) operation supporting nearest, bilinear, and bicubic modes
 *
 * Resizes 4D tensor (N, C, H, W) to target spatial dimensions.
 *
 * @param input Input tensor: (N, C, H_in, W_in)
 * @param size Target spatial dimensions {H_out, W_out}
 * @param mode Interpolation mode: "nearest", "bilinear", or "bicubic"
 * @param align_corners Whether to align corner pixels
 * @param queue SYCL queue for execution
 * @return Resized tensor: (N, C, H_out, W_out)
 */
auto interpolate_kernel(
    const Tensor& input,
    const std::vector<int64_t>& size,
    const std::string& mode,
    bool align_corners,
    sycl::queue& queue
) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("interpolate requires 4D input (N, C, H, W)");
    }
    if (size.size() != 2) {
        throw std::invalid_argument("interpolate size must have 2 elements (H_out, W_out)");
    }

    const int64_t N = input_shape[0];
    const int64_t C = input_shape[1];
    const int64_t H_in = input_shape[2];
    const int64_t W_in = input_shape[3];
    const int64_t H_out = size[0];
    const int64_t W_out = size[1];

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());
    const int64_t total = N * C * H_out * W_out;

    if (total == 0) return output;

    if (mode == "nearest") {
        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);
            float* out_ptr = get_data_ptr<float>(output);
            queue.parallel_for<InterpolateNearestKernelFloat32>(sycl::range<1>(total),
                [=](sycl::id<1> idx) {
                    int64_t temp = idx;
                    const int64_t w = temp % W_out; temp /= W_out;
                    const int64_t h = temp % H_out; temp /= H_out;
                    const int64_t c = temp % C;
                    const int64_t n = temp / C;

                    int64_t h_in = static_cast<int64_t>(static_cast<float>(h) * H_in / H_out);
                    int64_t w_in = static_cast<int64_t>(static_cast<float>(w) * W_in / W_out);
                    h_in = sycl::min(h_in, H_in - 1);
                    w_in = sycl::min(w_in, W_in - 1);

                    out_ptr[idx] = in_ptr[((n * C + c) * H_in + h_in) * W_in + w_in];
                });
        }
        else if (input.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(input);
            double* out_ptr = get_data_ptr<double>(output);
            queue.parallel_for<InterpolateNearestKernelFloat64>(sycl::range<1>(total),
                [=](sycl::id<1> idx) {
                    int64_t temp = idx;
                    const int64_t w = temp % W_out; temp /= W_out;
                    const int64_t h = temp % H_out; temp /= H_out;
                    const int64_t c = temp % C;
                    const int64_t n = temp / C;

                    int64_t h_in = static_cast<int64_t>(static_cast<double>(h) * H_in / H_out);
                    int64_t w_in = static_cast<int64_t>(static_cast<double>(w) * W_in / W_out);
                    h_in = sycl::min(h_in, H_in - 1);
                    w_in = sycl::min(w_in, W_in - 1);

                    out_ptr[idx] = in_ptr[((n * C + c) * H_in + h_in) * W_in + w_in];
                });
        }
        else if (input.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
            sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
            queue.parallel_for<InterpolateNearestKernelFloat16>(sycl::range<1>(total),
                [=](sycl::id<1> idx) {
                    int64_t temp = idx;
                    const int64_t w = temp % W_out; temp /= W_out;
                    const int64_t h = temp % H_out; temp /= H_out;
                    const int64_t c = temp % C;
                    const int64_t n = temp / C;

                    int64_t h_in = static_cast<int64_t>(static_cast<float>(h) * H_in / H_out);
                    int64_t w_in = static_cast<int64_t>(static_cast<float>(w) * W_in / W_out);
                    h_in = sycl::min(h_in, H_in - 1);
                    w_in = sycl::min(w_in, W_in - 1);

                    out_ptr[idx] = in_ptr[((n * C + c) * H_in + h_in) * W_in + w_in];
                });
        }
        else {
            throw std::runtime_error("interpolate nearest: unsupported dtype");
        }
    }
    else if (mode == "bilinear") {
        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);
            float* out_ptr = get_data_ptr<float>(output);
            queue.parallel_for<InterpolateBilinearKernelFloat32>(sycl::range<1>(total),
                [=](sycl::id<1> idx) {
                    int64_t temp = idx;
                    const int64_t w = temp % W_out; temp /= W_out;
                    const int64_t h = temp % H_out; temp /= H_out;
                    const int64_t c = temp % C;
                    const int64_t n = temp / C;

                    float h_scale = align_corners && H_out > 1
                        ? static_cast<float>(H_in - 1) / (H_out - 1)
                        : static_cast<float>(H_in) / H_out;
                    float w_scale = align_corners && W_out > 1
                        ? static_cast<float>(W_in - 1) / (W_out - 1)
                        : static_cast<float>(W_in) / W_out;

                    float h_real = align_corners ? h * h_scale : (h + 0.5f) * h_scale - 0.5f;
                    float w_real = align_corners ? w * w_scale : (w + 0.5f) * w_scale - 0.5f;

                    h_real = sycl::fmax(h_real, 0.0f);
                    w_real = sycl::fmax(w_real, 0.0f);

                    int64_t h0 = static_cast<int64_t>(h_real);
                    int64_t w0 = static_cast<int64_t>(w_real);
                    int64_t h1 = sycl::min(h0 + 1, H_in - 1);
                    int64_t w1 = sycl::min(w0 + 1, W_in - 1);
                    h0 = sycl::min(h0, H_in - 1);
                    w0 = sycl::min(w0, W_in - 1);

                    float h_lambda = h_real - h0;
                    float w_lambda = w_real - w0;

                    int64_t base = (n * C + c) * H_in;
                    float v00 = in_ptr[(base + h0) * W_in + w0];
                    float v01 = in_ptr[(base + h0) * W_in + w1];
                    float v10 = in_ptr[(base + h1) * W_in + w0];
                    float v11 = in_ptr[(base + h1) * W_in + w1];

                    float val = (1.0f - h_lambda) * ((1.0f - w_lambda) * v00 + w_lambda * v01)
                              + h_lambda * ((1.0f - w_lambda) * v10 + w_lambda * v11);
                    out_ptr[idx] = val;
                });
        }
        else if (input.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(input);
            double* out_ptr = get_data_ptr<double>(output);
            queue.parallel_for<InterpolateBilinearKernelFloat64>(sycl::range<1>(total),
                [=](sycl::id<1> idx) {
                    int64_t temp = idx;
                    const int64_t w = temp % W_out; temp /= W_out;
                    const int64_t h = temp % H_out; temp /= H_out;
                    const int64_t c = temp % C;
                    const int64_t n = temp / C;

                    double h_scale = align_corners && H_out > 1
                        ? static_cast<double>(H_in - 1) / (H_out - 1)
                        : static_cast<double>(H_in) / H_out;
                    double w_scale = align_corners && W_out > 1
                        ? static_cast<double>(W_in - 1) / (W_out - 1)
                        : static_cast<double>(W_in) / W_out;

                    double h_real = align_corners ? h * h_scale : (h + 0.5) * h_scale - 0.5;
                    double w_real = align_corners ? w * w_scale : (w + 0.5) * w_scale - 0.5;

                    h_real = sycl::fmax(h_real, 0.0);
                    w_real = sycl::fmax(w_real, 0.0);

                    int64_t h0 = static_cast<int64_t>(h_real);
                    int64_t w0 = static_cast<int64_t>(w_real);
                    int64_t h1 = sycl::min(h0 + 1, H_in - 1);
                    int64_t w1 = sycl::min(w0 + 1, W_in - 1);
                    h0 = sycl::min(h0, H_in - 1);
                    w0 = sycl::min(w0, W_in - 1);

                    double h_lambda = h_real - h0;
                    double w_lambda = w_real - w0;

                    int64_t base = (n * C + c) * H_in;
                    double v00 = in_ptr[(base + h0) * W_in + w0];
                    double v01 = in_ptr[(base + h0) * W_in + w1];
                    double v10 = in_ptr[(base + h1) * W_in + w0];
                    double v11 = in_ptr[(base + h1) * W_in + w1];

                    double val = (1.0 - h_lambda) * ((1.0 - w_lambda) * v00 + w_lambda * v01)
                               + h_lambda * ((1.0 - w_lambda) * v10 + w_lambda * v11);
                    out_ptr[idx] = val;
                });
        }
        else if (input.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
            sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
            queue.parallel_for<InterpolateBilinearKernelFloat16>(sycl::range<1>(total),
                [=](sycl::id<1> idx) {
                    int64_t temp = idx;
                    const int64_t w = temp % W_out; temp /= W_out;
                    const int64_t h = temp % H_out; temp /= H_out;
                    const int64_t c = temp % C;
                    const int64_t n = temp / C;

                    float h_scale = align_corners && H_out > 1
                        ? static_cast<float>(H_in - 1) / (H_out - 1)
                        : static_cast<float>(H_in) / H_out;
                    float w_scale = align_corners && W_out > 1
                        ? static_cast<float>(W_in - 1) / (W_out - 1)
                        : static_cast<float>(W_in) / W_out;

                    float h_real = align_corners ? h * h_scale : (h + 0.5f) * h_scale - 0.5f;
                    float w_real = align_corners ? w * w_scale : (w + 0.5f) * w_scale - 0.5f;

                    h_real = sycl::fmax(h_real, 0.0f);
                    w_real = sycl::fmax(w_real, 0.0f);

                    int64_t h0 = static_cast<int64_t>(h_real);
                    int64_t w0 = static_cast<int64_t>(w_real);
                    int64_t h1 = sycl::min(h0 + 1, H_in - 1);
                    int64_t w1 = sycl::min(w0 + 1, W_in - 1);
                    h0 = sycl::min(h0, H_in - 1);
                    w0 = sycl::min(w0, W_in - 1);

                    float h_lambda = h_real - h0;
                    float w_lambda = w_real - w0;

                    int64_t base = (n * C + c) * H_in;
                    float v00 = static_cast<float>(in_ptr[(base + h0) * W_in + w0]);
                    float v01 = static_cast<float>(in_ptr[(base + h0) * W_in + w1]);
                    float v10 = static_cast<float>(in_ptr[(base + h1) * W_in + w0]);
                    float v11 = static_cast<float>(in_ptr[(base + h1) * W_in + w1]);

                    float val = (1.0f - h_lambda) * ((1.0f - w_lambda) * v00 + w_lambda * v01)
                              + h_lambda * ((1.0f - w_lambda) * v10 + w_lambda * v11);
                    out_ptr[idx] = sycl::half(val);
                });
        }
        else {
            throw std::runtime_error("interpolate bilinear: unsupported dtype");
        }
    }
    else if (mode == "bicubic") {
        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);
            float* out_ptr = get_data_ptr<float>(output);
            queue.parallel_for<InterpolateBicubicKernelFloat32>(sycl::range<1>(total),
                [=](sycl::id<1> idx) {
                    int64_t temp = idx;
                    const int64_t w = temp % W_out; temp /= W_out;
                    const int64_t h = temp % H_out; temp /= H_out;
                    const int64_t c = temp % C;
                    const int64_t n = temp / C;

                    float h_scale = align_corners && H_out > 1
                        ? static_cast<float>(H_in - 1) / (H_out - 1)
                        : static_cast<float>(H_in) / H_out;
                    float w_scale = align_corners && W_out > 1
                        ? static_cast<float>(W_in - 1) / (W_out - 1)
                        : static_cast<float>(W_in) / W_out;

                    float h_real = align_corners ? h * h_scale : (h + 0.5f) * h_scale - 0.5f;
                    float w_real = align_corners ? w * w_scale : (w + 0.5f) * w_scale - 0.5f;

                    int64_t h_floor = static_cast<int64_t>(sycl::floor(h_real));
                    int64_t w_floor = static_cast<int64_t>(sycl::floor(w_real));

                    float val = 0.0f;
                    for (int64_t dh = -1; dh <= 2; ++dh) {
                        int64_t hi = sycl::clamp(h_floor + dh, int64_t(0), H_in - 1);
                        float h_w = 1.5f * sycl::fabs(h_real - (h_floor + dh));
                        float hw;
                        if (h_w < 1.0f) {
                            hw = ((1.5f * h_w - 2.5f) * h_w) * h_w + 1.0f;
                        } else if (h_w < 2.0f) {
                            hw = ((-0.5f * h_w + 2.5f) * h_w - 4.0f) * h_w + 2.0f;
                        } else {
                            hw = 0.0f;
                        }

                        for (int64_t dw = -1; dw <= 2; ++dw) {
                            int64_t wi = sycl::clamp(w_floor + dw, int64_t(0), W_in - 1);
                            float w_w = sycl::fabs(w_real - (w_floor + dw));
                            float ww;
                            if (w_w < 1.0f) {
                                ww = ((1.5f * w_w - 2.5f) * w_w) * w_w + 1.0f;
                            } else if (w_w < 2.0f) {
                                ww = ((-0.5f * w_w + 2.5f) * w_w - 4.0f) * w_w + 2.0f;
                            } else {
                                ww = 0.0f;
                            }

                            val += hw * ww * in_ptr[((n * C + c) * H_in + hi) * W_in + wi];
                        }
                    }
                    out_ptr[idx] = val;
                });
        }
        else if (input.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(input);
            double* out_ptr = get_data_ptr<double>(output);
            queue.parallel_for<InterpolateBicubicKernelFloat64>(sycl::range<1>(total),
                [=](sycl::id<1> idx) {
                    int64_t temp = idx;
                    const int64_t w = temp % W_out; temp /= W_out;
                    const int64_t h = temp % H_out; temp /= H_out;
                    const int64_t c = temp % C;
                    const int64_t n = temp / C;

                    double h_scale = align_corners && H_out > 1
                        ? static_cast<double>(H_in - 1) / (H_out - 1)
                        : static_cast<double>(H_in) / H_out;
                    double w_scale = align_corners && W_out > 1
                        ? static_cast<double>(W_in - 1) / (W_out - 1)
                        : static_cast<double>(W_in) / W_out;

                    double h_real = align_corners ? h * h_scale : (h + 0.5) * h_scale - 0.5;
                    double w_real = align_corners ? w * w_scale : (w + 0.5) * w_scale - 0.5;

                    int64_t h_floor = static_cast<int64_t>(sycl::floor(h_real));
                    int64_t w_floor = static_cast<int64_t>(sycl::floor(w_real));

                    double val = 0.0;
                    for (int64_t dh = -1; dh <= 2; ++dh) {
                        int64_t hi = sycl::clamp(h_floor + dh, int64_t(0), H_in - 1);
                        double h_w = sycl::fabs(h_real - (h_floor + dh));
                        double hw;
                        if (h_w < 1.0) {
                            hw = ((1.5 * h_w - 2.5) * h_w) * h_w + 1.0;
                        } else if (h_w < 2.0) {
                            hw = ((-0.5 * h_w + 2.5) * h_w - 4.0) * h_w + 2.0;
                        } else {
                            hw = 0.0;
                        }

                        for (int64_t dw = -1; dw <= 2; ++dw) {
                            int64_t wi = sycl::clamp(w_floor + dw, int64_t(0), W_in - 1);
                            double w_w = sycl::fabs(w_real - (w_floor + dw));
                            double ww;
                            if (w_w < 1.0) {
                                ww = ((1.5 * w_w - 2.5) * w_w) * w_w + 1.0;
                            } else if (w_w < 2.0) {
                                ww = ((-0.5 * w_w + 2.5) * w_w - 4.0) * w_w + 2.0;
                            } else {
                                ww = 0.0;
                            }

                            val += hw * ww * in_ptr[((n * C + c) * H_in + hi) * W_in + wi];
                        }
                    }
                    out_ptr[idx] = val;
                });
        }
        else if (input.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
            sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
            queue.parallel_for<InterpolateBicubicKernelFloat16>(sycl::range<1>(total),
                [=](sycl::id<1> idx) {
                    int64_t temp = idx;
                    const int64_t w = temp % W_out; temp /= W_out;
                    const int64_t h = temp % H_out; temp /= H_out;
                    const int64_t c = temp % C;
                    const int64_t n = temp / C;

                    float h_scale = align_corners && H_out > 1
                        ? static_cast<float>(H_in - 1) / (H_out - 1)
                        : static_cast<float>(H_in) / H_out;
                    float w_scale = align_corners && W_out > 1
                        ? static_cast<float>(W_in - 1) / (W_out - 1)
                        : static_cast<float>(W_in) / W_out;

                    float h_real = align_corners ? h * h_scale : (h + 0.5f) * h_scale - 0.5f;
                    float w_real = align_corners ? w * w_scale : (w + 0.5f) * w_scale - 0.5f;

                    int64_t h_floor = static_cast<int64_t>(sycl::floor(h_real));
                    int64_t w_floor = static_cast<int64_t>(sycl::floor(w_real));

                    float val = 0.0f;
                    for (int64_t dh = -1; dh <= 2; ++dh) {
                        int64_t hi = sycl::clamp(h_floor + dh, int64_t(0), H_in - 1);
                        float h_w = sycl::fabs(h_real - (h_floor + dh));
                        float hw;
                        if (h_w < 1.0f) {
                            hw = ((1.5f * h_w - 2.5f) * h_w) * h_w + 1.0f;
                        } else if (h_w < 2.0f) {
                            hw = ((-0.5f * h_w + 2.5f) * h_w - 4.0f) * h_w + 2.0f;
                        } else {
                            hw = 0.0f;
                        }

                        for (int64_t dw = -1; dw <= 2; ++dw) {
                            int64_t wi = sycl::clamp(w_floor + dw, int64_t(0), W_in - 1);
                            float w_w = sycl::fabs(w_real - (w_floor + dw));
                            float ww;
                            if (w_w < 1.0f) {
                                ww = ((1.5f * w_w - 2.5f) * w_w) * w_w + 1.0f;
                            } else if (w_w < 2.0f) {
                                ww = ((-0.5f * w_w + 2.5f) * w_w - 4.0f) * w_w + 2.0f;
                            } else {
                                ww = 0.0f;
                            }

                            val += hw * ww * static_cast<float>(in_ptr[((n * C + c) * H_in + hi) * W_in + wi]);
                        }
                    }
                    out_ptr[idx] = sycl::half(val);
                });
        }
        else {
            throw std::runtime_error("interpolate bicubic: unsupported dtype");
        }
    }
    else {
        throw std::runtime_error("interpolate: unsupported mode '" + mode + "' (use nearest, bilinear, or bicubic)");
    }

    return output;
}

// ============================================================================
// Box IoU Operation
// ============================================================================
// Kernel name classes
struct BoxIoUKernelFloat32 {};
struct BoxIoUKernelFloat64 {};

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_vision_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

auto box_iou_kernel(
    const Tensor& boxes1,
    const Tensor& boxes2,
    int iou_type,
    sycl::queue& queue
) -> Tensor {
    int64_t N = boxes1.shape()[0];
    int64_t M = boxes2.shape()[0];

    Tensor output({N, M}, boxes1.dtype(), boxes1.device());

    int64_t total = N * M;
    if (total == 0) return output;

    if (boxes1.dtype() == DType::Float32) {
        const float* b1_ptr = get_vision_data_ptr<const float>(boxes1);
        const float* b2_ptr = get_vision_data_ptr<const float>(boxes2);
        float* out_ptr = get_vision_data_ptr<float>(output);

        queue.parallel_for<BoxIoUKernelFloat32>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            int64_t i = idx / M;
            int64_t j = idx % M;

            // Box format: [x1, y1, x2, y2]
            float x1_1 = b1_ptr[i * 4 + 0];
            float y1_1 = b1_ptr[i * 4 + 1];
            float x2_1 = b1_ptr[i * 4 + 2];
            float y2_1 = b1_ptr[i * 4 + 3];

            float x1_2 = b2_ptr[j * 4 + 0];
            float y1_2 = b2_ptr[j * 4 + 1];
            float x2_2 = b2_ptr[j * 4 + 2];
            float y2_2 = b2_ptr[j * 4 + 3];

            // Intersection
            float inter_x1 = sycl::fmax(x1_1, x1_2);
            float inter_y1 = sycl::fmax(y1_1, y1_2);
            float inter_x2 = sycl::fmin(x2_1, x2_2);
            float inter_y2 = sycl::fmin(y2_1, y2_2);

            float inter_w = sycl::fmax(0.0f, inter_x2 - inter_x1);
            float inter_h = sycl::fmax(0.0f, inter_y2 - inter_y1);
            float inter_area = inter_w * inter_h;

            // Areas
            float area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
            float area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
            float union_area = area1 + area2 - inter_area;

            float iou = inter_area / (union_area + 1e-7f);

            if (iou_type == 1) {
                // GIoU
                float enc_x1 = sycl::fmin(x1_1, x1_2);
                float enc_y1 = sycl::fmin(y1_1, y1_2);
                float enc_x2 = sycl::fmax(x2_1, x2_2);
                float enc_y2 = sycl::fmax(y2_1, y2_2);
                float enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1);
                iou = iou - (enc_area - union_area) / (enc_area + 1e-7f);
            }

            out_ptr[i * M + j] = iou;
        });
    } else if (boxes1.dtype() == DType::Float64) {
        const double* b1_ptr = get_vision_data_ptr<const double>(boxes1);
        const double* b2_ptr = get_vision_data_ptr<const double>(boxes2);
        double* out_ptr = get_vision_data_ptr<double>(output);

        queue.parallel_for<BoxIoUKernelFloat64>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            int64_t i = idx / M;
            int64_t j = idx % M;

            double x1_1 = b1_ptr[i * 4 + 0];
            double y1_1 = b1_ptr[i * 4 + 1];
            double x2_1 = b1_ptr[i * 4 + 2];
            double y2_1 = b1_ptr[i * 4 + 3];

            double x1_2 = b2_ptr[j * 4 + 0];
            double y1_2 = b2_ptr[j * 4 + 1];
            double x2_2 = b2_ptr[j * 4 + 2];
            double y2_2 = b2_ptr[j * 4 + 3];

            double inter_x1 = sycl::fmax(x1_1, x1_2);
            double inter_y1 = sycl::fmax(y1_1, y1_2);
            double inter_x2 = sycl::fmin(x2_1, x2_2);
            double inter_y2 = sycl::fmin(y2_1, y2_2);

            double inter_w = sycl::fmax(0.0, inter_x2 - inter_x1);
            double inter_h = sycl::fmax(0.0, inter_y2 - inter_y1);
            double inter_area = inter_w * inter_h;

            double area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
            double area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
            double union_area = area1 + area2 - inter_area;

            double iou = inter_area / (union_area + 1e-7);

            if (iou_type == 1) {
                double enc_x1 = sycl::fmin(x1_1, x1_2);
                double enc_y1 = sycl::fmin(y1_1, y1_2);
                double enc_x2 = sycl::fmax(x2_1, x2_2);
                double enc_y2 = sycl::fmax(y2_1, y2_2);
                double enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1);
                iou = iou - (enc_area - union_area) / (enc_area + 1e-7);
            }

            out_ptr[i * M + j] = iou;
        });
    } else {
        throw std::runtime_error("box_iou_kernel: unsupported dtype (need Float32 or Float64)");
    }

    return output;
}

} // namespace oneapi
} // namespace tenzor
