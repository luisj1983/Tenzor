#include "tenzor/core/tensor.hpp"
#include "oneapi_kernel_utils.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>

#ifdef TENZOR_HAS_ONEDPL
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#endif

namespace tenzor {
namespace oneapi {

// SYCL Kernel name classes
struct NMSKernelFloat32 {};
struct NMSKernelFloat64 {};
struct ROIAlignKernelFloat32 {};
struct ROIAlignKernelFloat64 {};
struct ROIAlignBackwardKernelFloat32 {};
struct ROIAlignBackwardKernelFloat64 {};
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
struct GatherRelativePositionBiasKernelBFloat16 {};
struct InterpolateNearestKernelBFloat16 {};
struct InterpolateBilinearKernelBFloat16 {};
struct InterpolateBicubicKernelBFloat16 {};
struct BoxIoUKernelFloat16 {};
struct BoxIoUKernelBFloat16 {};
struct NmsIoUBitmaskKernel {};



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

    // Convert to Float32 on device if needed (avoids host roundtrip for FP16/BF16/FP64)
    // The kernels read these linearly; the else-branch aliased a possibly
    // non-contiguous input. Materialize contiguous copies.
    Tensor boxes_f32 = (boxes.dtype() != DType::Float32) ? boxes.to(DType::Float32) : boxes;
    if (!boxes_f32.is_contiguous()) boxes_f32 = boxes_f32.contiguous();
    Tensor scores_f32 = (scores.dtype() != DType::Float32) ? scores.to(DType::Float32) : scores;
    if (!scores_f32.is_contiguous()) scores_f32 = scores_f32.contiguous();

    const float* boxes_f32_ptr = get_data_ptr<const float>(boxes_f32);
    const float* scores_f32_ptr = get_data_ptr<const float>(scores_f32);

    // Sort boxes by score (descending). The sorted-by-score original-index
    // array `d_order` stays on the device; greedy suppression below consumes
    // it directly without a host roundtrip.
    int64_t* d_order = sycl::malloc_device<int64_t>(num_boxes, queue);

#ifdef TENZOR_HAS_ONEDPL
    {
        auto policy = ::oneapi::dpl::execution::make_device_policy(queue);
        float* d_scores = sycl::malloc_device<float>(num_boxes, queue);
        queue.memcpy(d_scores, scores_f32_ptr, num_boxes * sizeof(float));
        queue.parallel_for(sycl::range<1>(num_boxes), [=](sycl::id<1> i) {
            d_order[i] = static_cast<int64_t>(i[0]);
        }).wait();
        ::oneapi::dpl::sort(policy,
            ::oneapi::dpl::make_zip_iterator(d_scores, d_order),
            ::oneapi::dpl::make_zip_iterator(d_scores + num_boxes, d_order + num_boxes),
            [](const auto& a, const auto& b) {
                return std::get<0>(a) > std::get<0>(b);
            });
        sycl::free(d_scores, queue);
    }
#else
    {
        // Device-side bitonic argsort descending (no oneDPL dependency).
        int64_t padded = 1;
        while (padded < num_boxes) padded <<= 1;

        float* d_scores = sycl::malloc_device<float>(padded, queue);
        int64_t* d_order_padded = sycl::malloc_device<int64_t>(padded, queue);

        queue.memcpy(d_scores, scores_f32_ptr, num_boxes * sizeof(float));
        queue.parallel_for(sycl::range<1>(padded), [=](sycl::id<1> i) {
            int64_t idx = static_cast<int64_t>(i[0]);
            d_order_padded[idx] = idx;
            if (idx >= num_boxes) {
                d_scores[idx] = -std::numeric_limits<float>::infinity();
            }
        }).wait();

        // Bitonic sort over the power-of-two-padded score array. Padding
        // entries are -inf so they always sink to the tail, leaving the first
        // num_boxes entries in strict descending-by-score order.
        //
        // Each of the padded/2 work-items owns exactly one compare-exchange
        // pair (l, r) for the current (k, j) stage. The partner index uses the
        // canonical bitonic mapping
        //     l = 2*tid - (tid & (j-1));   r = l + j;
        // which enumerates the low element of every j-block across the FULL
        // array — including the upper half. The previous formula
        // `l = tid | (tid & ~(j-1))` collapses to `l = tid`, so for
        // j >= padded/2 the upper half was never touched and the `r <= l`
        // guard dropped half the comparisons, corrupting the order for every
        // non-trivial length. With this mapping l<r and l+j<padded always
        // hold, so no bounds guard is needed.
        //
        // Direction: in a block whose k-bit is clear we keep the LARGER value
        // first (swap on lv<rv); in its mirror we keep the smaller first. After
        // the final stage (k==padded) the whole array is globally DESCENDING.
        for (int64_t k = 2; k <= padded; k <<= 1) {
            for (int64_t j = k >> 1; j > 0; j >>= 1) {
                int64_t half = padded / 2;
                queue.parallel_for(sycl::range<1>(half),
                    [=](sycl::id<1> gid) {
                        int64_t tid = static_cast<int64_t>(gid[0]);
                        int64_t l = 2 * tid - (tid & (j - 1));
                        int64_t r = l + j;
                        bool keep_larger_first = ((l & k) == 0);
                        float lv = d_scores[l], rv = d_scores[r];
                        bool should_swap = keep_larger_first ? (lv < rv) : (lv > rv);
                        if (should_swap) {
                            d_scores[l] = rv;
                            d_scores[r] = lv;
                            int64_t tmp = d_order_padded[l];
                            d_order_padded[l] = d_order_padded[r];
                            d_order_padded[r] = tmp;
                        }
                    }).wait();
            }
        }
        queue.memcpy(d_order, d_order_padded, num_boxes * sizeof(int64_t));
        sycl::free(d_scores, queue);
        sycl::free(d_order_padded, queue);
    }
#endif

    // Compute IoU on device and produce a thresholded bitmask.
    // Instead of copying an N*N float IoU matrix to host (4*N*N bytes),
    // we threshold on-device and pack results into uint64 words,
    // transferring only N * ceil(N/64) * 8 bytes — up to 256x smaller.
    // boxes_f32 already lives on device, so point d_boxes directly at it.
    const float* d_boxes = boxes_f32_ptr;

    // Each row i has ceil(num_boxes/64) uint64 words; bit j is set if IoU(i,j) > threshold
    int64_t cols_u64 = (num_boxes + 63) / 64;
    int64_t bitmask_elems = num_boxes * cols_u64;
    uint64_t* d_bitmask = sycl::malloc_device<uint64_t>(bitmask_elems, queue);
    queue.memset(d_bitmask, 0, bitmask_elems * sizeof(uint64_t)).wait();

    // Launch one work-item per upper-triangle pair (i > j).
    // Each sets the corresponding bit in both row i and row j (symmetric).
    int64_t upper_pairs = num_boxes * (num_boxes - 1) / 2;
    float iou_thresh_val = iou_threshold;

    if (upper_pairs > 0) {
        queue.parallel_for<NmsIoUBitmaskKernel>(
            sycl::range<1>(upper_pairs), [=](sycl::id<1> gid) {
                // Map linear index to upper-triangle (i, j) where i > j
                // Using the inverse triangular formula:
                //   gid = i*(i-1)/2 + j  =>  i = floor((1+sqrt(1+8*gid))/2)
                int64_t g = static_cast<int64_t>(gid[0]);
                int64_t i = static_cast<int64_t>(
                    (1.0 + sycl::sqrt(1.0 + 8.0 * static_cast<double>(g))) * 0.5);
                // Correct for floating-point imprecision
                if (i * (i - 1) / 2 > g) --i;
                if ((i + 1) * i / 2 <= g) ++i;
                int64_t j = g - i * (i - 1) / 2;

                float x1_i = d_boxes[i * 4 + 0], y1_i = d_boxes[i * 4 + 1];
                float x2_i = d_boxes[i * 4 + 2], y2_i = d_boxes[i * 4 + 3];
                float x1_j = d_boxes[j * 4 + 0], y1_j = d_boxes[j * 4 + 1];
                float x2_j = d_boxes[j * 4 + 2], y2_j = d_boxes[j * 4 + 3];

                float area_i = (x2_i - x1_i) * (y2_i - y1_i);
                float area_j = (x2_j - x1_j) * (y2_j - y1_j);

                float xx1 = sycl::max(x1_i, x1_j);
                float yy1 = sycl::max(y1_i, y1_j);
                float xx2 = sycl::min(x2_i, x2_j);
                float yy2 = sycl::min(y2_i, y2_j);

                float w = sycl::max(0.0f, xx2 - xx1);
                float h = sycl::max(0.0f, yy2 - yy1);
                float intersection = w * h;
                float iou = intersection / (area_i + area_j - intersection + 1e-6f);

                if (iou > iou_thresh_val) {
                    // Set bit j in row i
                    int64_t word_ij = j / 64;
                    uint64_t bit_ij = uint64_t(1) << (j % 64);
                    auto addr_ij = sycl::atomic_ref<uint64_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(
                            d_bitmask[i * cols_u64 + word_ij]);
                    addr_ij.fetch_or(bit_ij);

                    // Set bit i in row j (symmetric)
                    int64_t word_ji = i / 64;
                    uint64_t bit_ji = uint64_t(1) << (i % 64);
                    auto addr_ji = sycl::atomic_ref<uint64_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(
                            d_bitmask[j * cols_u64 + word_ji]);
                    addr_ji.fetch_or(bit_ji);
                }
            }).wait();
    }

    // Greedy suppression on-device, in a single SYCL work-group. tid==0
    // drives the sequential keep/suppress decisions; the whole work-group
    // cooperates on the chunk-OR step. Mirrors the CUDA
    // `nms_greedy_suppression_kernel` design — only `num_keep` (8 bytes)
    // flows back to the host.
    int64_t* d_keep = sycl::malloc_device<int64_t>(num_boxes, queue);
    int64_t* d_num_keep = sycl::malloc_device<int64_t>(1, queue);
    queue.memset(d_num_keep, 0, sizeof(int64_t)).wait();

    constexpr int LOCAL = 256;
    int64_t cols_u64_local = cols_u64;
    int64_t num_boxes_local = num_boxes;
    int64_t num_chunks = cols_u64;

    queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint64_t, 1> s_remv(static_cast<size_t>(num_chunks), h);
        sycl::local_accessor<int64_t, 1> s_keep_count(1, h);
        sycl::local_accessor<int, 1> s_keep_flag(1, h);
        const uint64_t* mask = d_bitmask;
        const int64_t* sorted = d_order;
        int64_t* keep_out = d_keep;
        int64_t* nk_out = d_num_keep;

        h.parallel_for<class NmsGreedySuppressKernel>(
            sycl::nd_range<1>(sycl::range<1>(LOCAL), sycl::range<1>(LOCAL)),
            [=](sycl::nd_item<1> it) {
                int tid = it.get_local_linear_id();
                int wsz = it.get_local_range(0);

                for (int64_t c = tid; c < num_chunks; c += wsz) s_remv[c] = 0ULL;
                if (tid == 0) s_keep_count[0] = 0;
                sycl::group_barrier(it.get_group());

                for (int64_t i = 0; i < num_boxes_local; ++i) {
                    if (tid == 0) {
                        int64_t box_idx = sorted[i];
                        int64_t chunk_i = box_idx / 64;
                        uint64_t bit_i = uint64_t(1) << (box_idx % 64);
                        if (s_remv[chunk_i] & bit_i) {
                            s_keep_flag[0] = 0;
                        } else {
                            keep_out[s_keep_count[0]] = box_idx;
                            s_keep_count[0] = s_keep_count[0] + 1;
                            s_keep_flag[0] = 1;
                        }
                    }
                    sycl::group_barrier(it.get_group());

                    if (s_keep_flag[0]) {
                        int64_t box_idx = sorted[i];
                        const uint64_t* row = mask + box_idx * cols_u64_local;
                        for (int64_t c = tid; c < num_chunks; c += wsz) {
                            s_remv[c] |= row[c];
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }

                if (tid == 0) nk_out[0] = s_keep_count[0];
            });
    }).wait();

    int64_t num_keep = 0;
    queue.memcpy(&num_keep, d_num_keep, sizeof(int64_t)).wait();

    sycl::free(d_bitmask, queue);
    sycl::free(d_num_keep, queue);
    sycl::free(d_order, queue);

    Tensor result({num_keep}, DType::Int64, boxes.device());
    if (num_keep > 0) {
        queue.memcpy(get_data_ptr<int64_t>(result), d_keep,
                     num_keep * sizeof(int64_t)).wait();
    }
    sycl::free(d_keep, queue);

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

    // Float16/BFloat16: compute in Float32 and narrow (mirrors CPU + the
    // backward kernel). Otherwise, ensure rois matches the feature dtype: each
    // native branch below reads rois with features.dtype(), so a Float32-rois /
    // Float64-features mismatch reinterpreted the ROI bytes as the wrong type,
    // yielding garbage coordinates and wrong outputs.
    if (features.dtype() == DType::Float16 ||
        features.dtype() == DType::BFloat16) {
        Tensor feat_f32 = features.to(DType::Float32);
        Tensor rois_f32 = rois.to(DType::Float32);
        Tensor result = roi_align_kernel(
            feat_f32, rois_f32, output_height, output_width, spatial_scale,
            sampling_ratio, aligned, queue);
        return result.to(features.dtype());
    }
    if (rois.dtype() != features.dtype()) {
        Tensor rois_matched = rois.to(features.dtype());
        return roi_align_kernel(
            features, rois_matched, output_height, output_width, spatial_scale,
            sampling_ratio, aligned, queue);
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
                // batch_idx comes from untrusted upstream (e.g. RPN output) and
                // is not guaranteed in-range. Skip and zero the output element
                // for an OOB batch index, matching the CPU reference and
                // avoiding a device heap over-read.
                if (batch_idx < 0 || batch_idx >= batch_size) {
                    output_ptr[idx] = 0;
                    return;
                }
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

                // Clamp each bin grid to at least 1 so a zero-area / inverted
                // ROI (possible when aligned=true skips the roi_w/h clamp)
                // never yields count==0 and a divide-by-zero NaN/Inf, matching
                // the CPU reference and PyTorch.
                roi_bin_grid_h = sycl::max(roi_bin_grid_h, int64_t(1));
                roi_bin_grid_w = sycl::max(roi_bin_grid_w, int64_t(1));
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
                // batch_idx comes from untrusted upstream (e.g. RPN output) and
                // is not guaranteed in-range. Skip and zero the output element
                // for an OOB batch index, matching the CPU reference and
                // avoiding a device heap over-read.
                if (batch_idx < 0 || batch_idx >= batch_size) {
                    output_ptr[idx] = 0;
                    return;
                }
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

                // Clamp each bin grid to at least 1 so a zero-area / inverted
                // ROI (possible when aligned=true skips the roi_w/h clamp)
                // never yields count==0 and a divide-by-zero NaN/Inf, matching
                // the CPU reference and PyTorch.
                roi_bin_grid_h = sycl::max(roi_bin_grid_h, int64_t(1));
                roi_bin_grid_w = sycl::max(roi_bin_grid_w, int64_t(1));
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
    else {
        // Float16/BFloat16 are handled by the widen-to-Float32 recursion at the
        // top of this function, so only Float32/Float64 reach the native paths.
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

    // Float16/BFloat16: compute the whole backward in Float32 and narrow the
    // result, mirroring the CPU kernel. The dedicated reduced-precision branches
    // below read the `rois` buffer as sycl::half / uint16, but rois is supplied
    // as Float32 (it is not converted to the feature dtype) — reinterpreting its
    // bytes as half corrupted every ROI coordinate, so the bounds check rejected
    // all sample points and the gradient came back all zeros. Widening both
    // tensors here also sidesteps the lack of half/bf16 atomic adds.
    if (grad_output.dtype() == DType::Float16 ||
        grad_output.dtype() == DType::BFloat16) {
        Tensor go_f32   = grad_output.to(DType::Float32);
        Tensor rois_f32 = rois.to(DType::Float32);
        Tensor result = roi_align_backward_kernel(
            go_f32, rois_f32, batch_size, channels, feat_height, feat_width,
            spatial_scale, sampling_ratio, aligned, queue);
        return result.to(grad_output.dtype());
    }
    // rois may arrive in a different dtype than grad_output (the test, and the
    // general API, keep ROI boxes in Float32 while features may be Float64).
    // Each native branch below reads rois with grad_output's dtype, so a
    // mismatch reinterprets the ROI bytes as the wrong type → garbage
    // coordinates → all-zero gradients (this broke the Float64 path the same way
    // it broke Float16). Bring rois to the compute dtype, then recurse into the
    // matching native branch.
    if (rois.dtype() != grad_output.dtype()) {
        Tensor rois_matched = rois.to(grad_output.dtype());
        return roi_align_backward_kernel(
            grad_output, rois_matched, batch_size, channels, feat_height,
            feat_width, spatial_scale, sampling_ratio, aligned, queue);
    }

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
                // batch_idx comes from untrusted upstream (e.g. RPN output) and
                // is not guaranteed in-range. Skip the ROI for an OOB batch
                // index, matching the CPU reference and avoiding an OOB device
                // atomic write (grad_features is already zero-filled).
                if (batch_idx < 0 || batch_idx >= batch_size) {
                    return;
                }
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

                // Clamp each bin grid to at least 1 so a zero-area / inverted
                // ROI (possible when aligned=true skips the roi_w/h clamp)
                // never yields count==0 and a divide-by-zero NaN/Inf, matching
                // the CPU reference and PyTorch.
                roi_bin_grid_h = sycl::max(roi_bin_grid_h, int64_t(1));
                roi_bin_grid_w = sycl::max(roi_bin_grid_w, int64_t(1));
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
                // batch_idx comes from untrusted upstream (e.g. RPN output) and
                // is not guaranteed in-range. Skip the ROI for an OOB batch
                // index, matching the CPU reference and avoiding an OOB device
                // atomic write (grad_features is already zero-filled).
                if (batch_idx < 0 || batch_idx >= batch_size) {
                    return;
                }
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

                // Clamp each bin grid to at least 1 so a zero-area / inverted
                // ROI (possible when aligned=true skips the roi_w/h clamp)
                // never yields count==0 and a divide-by-zero NaN/Inf, matching
                // the CPU reference and PyTorch.
                roi_bin_grid_h = sycl::max(roi_bin_grid_h, int64_t(1));
                roi_bin_grid_w = sycl::max(roi_bin_grid_w, int64_t(1));
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

    // Bounds-check every relative-position index up front. Each index is used as
    // an unchecked offset into the position-bias table (table_ptr[table_idx *
    // num_heads + h]); a negative or out-of-range value (e.g. from an untrusted
    // checkpoint or mismatched shapes) would otherwise be an out-of-bounds device
    // read / information-disclosure vector. This mirrors the CPU reference
    // (src/backends/cpu/kernels/vision.cpp) which throws std::out_of_range. The
    // OneAPI table layout is [table_size, num_heads], so valid indices lie in
    // [0, table.shape()[0]).
    const int64_t table_size = table.shape()[0];
    const int64_t num_index = num_positions * num_positions;
    {
        std::vector<int64_t> host_indices(static_cast<size_t>(num_index));
        const int64_t* indices_dev = get_data_ptr<const int64_t>(indices);
        queue.memcpy(host_indices.data(), indices_dev,
                     static_cast<size_t>(num_index) * sizeof(int64_t)).wait();
        for (int64_t pos = 0; pos < num_index; ++pos) {
            int64_t table_idx = host_indices[pos];
            if (table_idx < 0 || table_idx >= table_size) {
                throw std::out_of_range(
                    "gather_relative_position_bias: rel_pos_index value " +
                    std::to_string(table_idx) + " out of range [0, " +
                    std::to_string(table_size) + ")");
            }
        }
    }

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
    else if (table.dtype() == DType::BFloat16) {
        const uint16_t* table_ptr = get_data_ptr<const uint16_t>(table);
        const int64_t* indices_ptr = get_data_ptr<const int64_t>(indices);
        uint16_t* output_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<GatherRelativePositionBiasKernelBFloat16>(
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
    const Tensor& input_in,
    const std::vector<int64_t>& size,
    const std::string& mode,
    bool align_corners,
    sycl::queue& queue
) -> Tensor {
    // Kernels read input with packed-NCHW offset arithmetic; materialize a
    // contiguous copy so non-contiguous inputs are read correctly.
    Tensor input = input_in.is_contiguous() ? input_in : input_in.contiguous();
    auto input_shape = input.shape();

    // 1D interpolate (N,C,L): reduce to the validated 4D path via a degenerate
    // height axis. Bilinear over a (1,L) image == 1D linear; nearest stays nearest.
    if (input_shape.size() == 3) {
        if (size.size() != 1) {
            throw std::invalid_argument("interpolate 1D: size must have 1 element (L_out)");
        }
        const std::string mode2d = (mode == "nearest") ? "nearest" : "bilinear";
        Tensor x4 = input.unsqueeze(2);  // (N,C,1,L)
        Tensor y4 = interpolate_kernel(x4, {1, size[0]}, mode2d, align_corners, queue);
        return y4.squeeze(2);            // (N,C,L_out)
    }

    // 3D interpolate (N,C,D,H,W): trilinear/nearest are separable, so compose two
    // validated 2D interpolations — first over (H,W) with D folded into the batch,
    // then over D with (H_out*W_out) carried as an identity-mapped width axis
    // (M->M bilinear/nearest is exact identity). Tensor-product of the two gives
    // full 3D trilinear/nearest.
    if (input_shape.size() == 5) {
        if (size.size() != 3) {
            throw std::invalid_argument("interpolate 3D: size must have 3 elements (D_out,H_out,W_out)");
        }
        const std::string mode2d = (mode == "nearest") ? "nearest" : "bilinear";
        const int64_t N5 = input_shape[0], C5 = input_shape[1];
        const int64_t D5 = input_shape[2], H5 = input_shape[3], W5 = input_shape[4];
        const int64_t Dout = size[0], Hout = size[1], Wout = size[2];
        // Step 1: interpolate H,W (D folded into the batch dimension).
        Tensor s1 = input.reshape({N5 * D5, C5, H5, W5});
        s1 = interpolate_kernel(s1, {Hout, Wout}, mode2d, align_corners, queue);
        s1 = s1.reshape({N5, C5, D5, Hout, Wout});
        // Step 2: interpolate D (H_out*W_out is the identity-mapped width axis).
        Tensor s2 = s1.reshape({N5, C5, D5, Hout * Wout});
        s2 = interpolate_kernel(s2, {Dout, Hout * Wout}, mode2d, align_corners, queue);
        return s2.reshape({N5, C5, Dout, Hout, Wout});
    }

    if (input_shape.size() != 4) {
        throw std::invalid_argument("interpolate requires 3D (1D), 4D (2D), or 5D (3D) input");
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
        else if (input.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
            queue.parallel_for<InterpolateNearestKernelBFloat16>(sycl::range<1>(total),
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
        else if (input.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
            queue.parallel_for<InterpolateBilinearKernelBFloat16>(sycl::range<1>(total),
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
                    float v00 = bf16_to_f32(in_ptr[(base + h0) * W_in + w0]);
                    float v01 = bf16_to_f32(in_ptr[(base + h0) * W_in + w1]);
                    float v10 = bf16_to_f32(in_ptr[(base + h1) * W_in + w0]);
                    float v11 = bf16_to_f32(in_ptr[(base + h1) * W_in + w1]);

                    float val = (1.0f - h_lambda) * ((1.0f - w_lambda) * v00 + w_lambda * v01)
                              + h_lambda * ((1.0f - w_lambda) * v10 + w_lambda * v11);
                    out_ptr[idx] = f32_to_bf16(val);
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
                    // Clamp continuous coord to [0,in-1] BEFORE flooring to match the
                    // CPU/CUDA/ROCm reference border convention (and the backward).
                    h_real = sycl::clamp(h_real, 0.0f, static_cast<float>(H_in - 1));
                    w_real = sycl::clamp(w_real, 0.0f, static_cast<float>(W_in - 1));

                    int64_t h_floor = static_cast<int64_t>(sycl::floor(h_real));
                    int64_t w_floor = static_cast<int64_t>(sycl::floor(w_real));

                    float val = 0.0f;
                    for (int64_t dh = -1; dh <= 2; ++dh) {
                        int64_t hi = sycl::clamp(h_floor + dh, int64_t(0), H_in - 1);
                        float h_w = sycl::fabs(h_real - (h_floor + dh));
                        float hw;
                        if (h_w < 1.0f) {
                            hw = ((1.25f * h_w - 2.25f) * h_w) * h_w + 1.0f;
                        } else if (h_w < 2.0f) {
                            hw = ((-0.75f * h_w + 3.75f) * h_w - 6.0f) * h_w + 3.0f;
                        } else {
                            hw = 0.0f;
                        }

                        for (int64_t dw = -1; dw <= 2; ++dw) {
                            int64_t wi = sycl::clamp(w_floor + dw, int64_t(0), W_in - 1);
                            float w_w = sycl::fabs(w_real - (w_floor + dw));
                            float ww;
                            if (w_w < 1.0f) {
                                ww = ((1.25f * w_w - 2.25f) * w_w) * w_w + 1.0f;
                            } else if (w_w < 2.0f) {
                                ww = ((-0.75f * w_w + 3.75f) * w_w - 6.0f) * w_w + 3.0f;
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
                    // Clamp continuous coord to [0,in-1] BEFORE flooring to match the
                    // CPU/CUDA/ROCm reference border convention (and the backward).
                    h_real = sycl::clamp(h_real, 0.0, static_cast<double>(H_in - 1));
                    w_real = sycl::clamp(w_real, 0.0, static_cast<double>(W_in - 1));

                    int64_t h_floor = static_cast<int64_t>(sycl::floor(h_real));
                    int64_t w_floor = static_cast<int64_t>(sycl::floor(w_real));

                    double val = 0.0;
                    for (int64_t dh = -1; dh <= 2; ++dh) {
                        int64_t hi = sycl::clamp(h_floor + dh, int64_t(0), H_in - 1);
                        double h_w = sycl::fabs(h_real - (h_floor + dh));
                        double hw;
                        if (h_w < 1.0) {
                            hw = ((1.25 * h_w - 2.25) * h_w) * h_w + 1.0;
                        } else if (h_w < 2.0) {
                            hw = ((-0.75 * h_w + 3.75) * h_w - 6.0) * h_w + 3.0;
                        } else {
                            hw = 0.0;
                        }

                        for (int64_t dw = -1; dw <= 2; ++dw) {
                            int64_t wi = sycl::clamp(w_floor + dw, int64_t(0), W_in - 1);
                            double w_w = sycl::fabs(w_real - (w_floor + dw));
                            double ww;
                            if (w_w < 1.0) {
                                ww = ((1.25 * w_w - 2.25) * w_w) * w_w + 1.0;
                            } else if (w_w < 2.0) {
                                ww = ((-0.75 * w_w + 3.75) * w_w - 6.0) * w_w + 3.0;
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
                    // Clamp continuous coord to [0,in-1] BEFORE flooring to match the
                    // CPU/CUDA/ROCm reference border convention (and the backward).
                    h_real = sycl::clamp(h_real, 0.0f, static_cast<float>(H_in - 1));
                    w_real = sycl::clamp(w_real, 0.0f, static_cast<float>(W_in - 1));

                    int64_t h_floor = static_cast<int64_t>(sycl::floor(h_real));
                    int64_t w_floor = static_cast<int64_t>(sycl::floor(w_real));

                    float val = 0.0f;
                    for (int64_t dh = -1; dh <= 2; ++dh) {
                        int64_t hi = sycl::clamp(h_floor + dh, int64_t(0), H_in - 1);
                        float h_w = sycl::fabs(h_real - (h_floor + dh));
                        float hw;
                        if (h_w < 1.0f) {
                            hw = ((1.25f * h_w - 2.25f) * h_w) * h_w + 1.0f;
                        } else if (h_w < 2.0f) {
                            hw = ((-0.75f * h_w + 3.75f) * h_w - 6.0f) * h_w + 3.0f;
                        } else {
                            hw = 0.0f;
                        }

                        for (int64_t dw = -1; dw <= 2; ++dw) {
                            int64_t wi = sycl::clamp(w_floor + dw, int64_t(0), W_in - 1);
                            float w_w = sycl::fabs(w_real - (w_floor + dw));
                            float ww;
                            if (w_w < 1.0f) {
                                ww = ((1.25f * w_w - 2.25f) * w_w) * w_w + 1.0f;
                            } else if (w_w < 2.0f) {
                                ww = ((-0.75f * w_w + 3.75f) * w_w - 6.0f) * w_w + 3.0f;
                            } else {
                                ww = 0.0f;
                            }

                            val += hw * ww * static_cast<float>(in_ptr[((n * C + c) * H_in + hi) * W_in + wi]);
                        }
                    }
                    out_ptr[idx] = sycl::half(val);
                });
        }
        else if (input.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
            queue.parallel_for<InterpolateBicubicKernelBFloat16>(sycl::range<1>(total),
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
                    // Clamp continuous coord to [0,in-1] BEFORE flooring to match the
                    // CPU/CUDA/ROCm reference border convention (and the backward).
                    h_real = sycl::clamp(h_real, 0.0f, static_cast<float>(H_in - 1));
                    w_real = sycl::clamp(w_real, 0.0f, static_cast<float>(W_in - 1));

                    int64_t h_floor = static_cast<int64_t>(sycl::floor(h_real));
                    int64_t w_floor = static_cast<int64_t>(sycl::floor(w_real));

                    float val = 0.0f;
                    for (int64_t dh = -1; dh <= 2; ++dh) {
                        int64_t hi = sycl::clamp(h_floor + dh, int64_t(0), H_in - 1);
                        float h_w = sycl::fabs(h_real - (h_floor + dh));
                        float hw;
                        if (h_w < 1.0f) {
                            hw = ((1.25f * h_w - 2.25f) * h_w) * h_w + 1.0f;
                        } else if (h_w < 2.0f) {
                            hw = ((-0.75f * h_w + 3.75f) * h_w - 6.0f) * h_w + 3.0f;
                        } else {
                            hw = 0.0f;
                        }

                        for (int64_t dw = -1; dw <= 2; ++dw) {
                            int64_t wi = sycl::clamp(w_floor + dw, int64_t(0), W_in - 1);
                            float w_w = sycl::fabs(w_real - (w_floor + dw));
                            float ww;
                            if (w_w < 1.0f) {
                                ww = ((1.25f * w_w - 2.25f) * w_w) * w_w + 1.0f;
                            } else if (w_w < 2.0f) {
                                ww = ((-0.75f * w_w + 3.75f) * w_w - 6.0f) * w_w + 3.0f;
                            } else {
                                ww = 0.0f;
                            }

                            val += hw * ww * bf16_to_f32(in_ptr[((n * C + c) * H_in + hi) * W_in + wi]);
                        }
                    }
                    out_ptr[idx] = f32_to_bf16(val);
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
            } else if (iou_type == 2 || iou_type == 3) {
                // DIoU (2) / CIoU (3): subtract the center-distance penalty (and,
                // for CIoU, the aspect-ratio penalty). Previously these fell
                // through and returned plain IoU. Matches the CPU reference.
                float enc_x1 = sycl::fmin(x1_1, x1_2);
                float enc_y1 = sycl::fmin(y1_1, y1_2);
                float enc_x2 = sycl::fmax(x2_1, x2_2);
                float enc_y2 = sycl::fmax(y2_1, y2_2);
                float enc_w = enc_x2 - enc_x1;
                float enc_h = enc_y2 - enc_y1;
                float cx1 = (x1_1 + x2_1) * 0.5f, cy1 = (y1_1 + y2_1) * 0.5f;
                float cx2 = (x1_2 + x2_2) * 0.5f, cy2 = (y1_2 + y2_2) * 0.5f;
                float center_dist_sq = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);
                float diag_dist_sq = enc_w * enc_w + enc_h * enc_h;
                float result = iou - center_dist_sq / (diag_dist_sq + 1e-7f);
                if (iou_type == 3) {
                    const float four_over_pi_sq =
                        4.0f / (3.14159265358979323846f * 3.14159265358979323846f);
                    float w1 = x2_1 - x1_1, h1 = y2_1 - y1_1;
                    float w2 = x2_2 - x1_2, h2 = y2_2 - y1_2;
                    float at = sycl::atan(w2 / (h2 + 1e-7f)) - sycl::atan(w1 / (h1 + 1e-7f));
                    float v = four_over_pi_sq * at * at;
                    float alpha = v / (1.0f - iou + v + 1e-7f);  // raw iou
                    result = result - alpha * v;
                }
                iou = result;
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
            } else if (iou_type == 2 || iou_type == 3) {
                // DIoU (2) / CIoU (3): center-distance (+ aspect-ratio) penalty.
                double enc_x1 = sycl::fmin(x1_1, x1_2);
                double enc_y1 = sycl::fmin(y1_1, y1_2);
                double enc_x2 = sycl::fmax(x2_1, x2_2);
                double enc_y2 = sycl::fmax(y2_1, y2_2);
                double enc_w = enc_x2 - enc_x1;
                double enc_h = enc_y2 - enc_y1;
                double cx1 = (x1_1 + x2_1) * 0.5, cy1 = (y1_1 + y2_1) * 0.5;
                double cx2 = (x1_2 + x2_2) * 0.5, cy2 = (y1_2 + y2_2) * 0.5;
                double center_dist_sq = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);
                double diag_dist_sq = enc_w * enc_w + enc_h * enc_h;
                double result = iou - center_dist_sq / (diag_dist_sq + 1e-7);
                if (iou_type == 3) {
                    const double four_over_pi_sq =
                        4.0 / (3.14159265358979323846 * 3.14159265358979323846);
                    double w1 = x2_1 - x1_1, h1 = y2_1 - y1_1;
                    double w2 = x2_2 - x1_2, h2 = y2_2 - y1_2;
                    double at = sycl::atan(w2 / (h2 + 1e-7)) - sycl::atan(w1 / (h1 + 1e-7));
                    double v = four_over_pi_sq * at * at;
                    double alpha = v / (1.0 - iou + v + 1e-7);  // raw iou
                    result = result - alpha * v;
                }
                iou = result;
            }

            out_ptr[i * M + j] = iou;
        });
    } else if (boxes1.dtype() == DType::Float16) {
        const sycl::half* b1_ptr = get_vision_data_ptr<const sycl::half>(boxes1);
        const sycl::half* b2_ptr = get_vision_data_ptr<const sycl::half>(boxes2);
        sycl::half* out_ptr = get_vision_data_ptr<sycl::half>(output);

        queue.parallel_for<BoxIoUKernelFloat16>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            int64_t i = idx / M;
            int64_t j = idx % M;

            float x1_1 = float(b1_ptr[i * 4 + 0]);
            float y1_1 = float(b1_ptr[i * 4 + 1]);
            float x2_1 = float(b1_ptr[i * 4 + 2]);
            float y2_1 = float(b1_ptr[i * 4 + 3]);

            float x1_2 = float(b2_ptr[j * 4 + 0]);
            float y1_2 = float(b2_ptr[j * 4 + 1]);
            float x2_2 = float(b2_ptr[j * 4 + 2]);
            float y2_2 = float(b2_ptr[j * 4 + 3]);

            float inter_x1 = sycl::fmax(x1_1, x1_2);
            float inter_y1 = sycl::fmax(y1_1, y1_2);
            float inter_x2 = sycl::fmin(x2_1, x2_2);
            float inter_y2 = sycl::fmin(y2_1, y2_2);

            float inter_w = sycl::fmax(0.0f, inter_x2 - inter_x1);
            float inter_h = sycl::fmax(0.0f, inter_y2 - inter_y1);
            float inter_area = inter_w * inter_h;

            float area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
            float area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
            float union_area = area1 + area2 - inter_area;

            float iou = inter_area / (union_area + 1e-7f);

            if (iou_type == 1) {
                float enc_x1 = sycl::fmin(x1_1, x1_2);
                float enc_y1 = sycl::fmin(y1_1, y1_2);
                float enc_x2 = sycl::fmax(x2_1, x2_2);
                float enc_y2 = sycl::fmax(y2_1, y2_2);
                float enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1);
                iou = iou - (enc_area - union_area) / (enc_area + 1e-7f);
            } else if (iou_type == 2 || iou_type == 3) {
                // DIoU (2) / CIoU (3): center-distance (+ aspect-ratio) penalty,
                // computed in float accumulation. Matches the Float32 path/CPU.
                float enc_x1 = sycl::fmin(x1_1, x1_2);
                float enc_y1 = sycl::fmin(y1_1, y1_2);
                float enc_x2 = sycl::fmax(x2_1, x2_2);
                float enc_y2 = sycl::fmax(y2_1, y2_2);
                float enc_w = enc_x2 - enc_x1;
                float enc_h = enc_y2 - enc_y1;
                float cx1 = (x1_1 + x2_1) * 0.5f, cy1 = (y1_1 + y2_1) * 0.5f;
                float cx2 = (x1_2 + x2_2) * 0.5f, cy2 = (y1_2 + y2_2) * 0.5f;
                float center_dist_sq = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);
                float diag_dist_sq = enc_w * enc_w + enc_h * enc_h;
                float result = iou - center_dist_sq / (diag_dist_sq + 1e-7f);
                if (iou_type == 3) {
                    const float four_over_pi_sq =
                        4.0f / (3.14159265358979323846f * 3.14159265358979323846f);
                    float w1 = x2_1 - x1_1, h1 = y2_1 - y1_1;
                    float w2 = x2_2 - x1_2, h2 = y2_2 - y1_2;
                    float at = sycl::atan(w2 / (h2 + 1e-7f)) - sycl::atan(w1 / (h1 + 1e-7f));
                    float v = four_over_pi_sq * at * at;
                    float alpha = v / (1.0f - iou + v + 1e-7f);  // raw iou
                    result = result - alpha * v;
                }
                iou = result;
            }

            out_ptr[i * M + j] = sycl::half(iou);
        });
    } else if (boxes1.dtype() == DType::BFloat16) {
        const uint16_t* b1_ptr = get_vision_data_ptr<const uint16_t>(boxes1);
        const uint16_t* b2_ptr = get_vision_data_ptr<const uint16_t>(boxes2);
        uint16_t* out_ptr = get_vision_data_ptr<uint16_t>(output);

        queue.parallel_for<BoxIoUKernelBFloat16>(sycl::range<1>(total), [=](sycl::id<1> idx) {
            int64_t i = idx / M;
            int64_t j = idx % M;

            float x1_1 = bf16_to_f32(b1_ptr[i * 4 + 0]);
            float y1_1 = bf16_to_f32(b1_ptr[i * 4 + 1]);
            float x2_1 = bf16_to_f32(b1_ptr[i * 4 + 2]);
            float y2_1 = bf16_to_f32(b1_ptr[i * 4 + 3]);

            float x1_2 = bf16_to_f32(b2_ptr[j * 4 + 0]);
            float y1_2 = bf16_to_f32(b2_ptr[j * 4 + 1]);
            float x2_2 = bf16_to_f32(b2_ptr[j * 4 + 2]);
            float y2_2 = bf16_to_f32(b2_ptr[j * 4 + 3]);

            float inter_x1 = sycl::fmax(x1_1, x1_2);
            float inter_y1 = sycl::fmax(y1_1, y1_2);
            float inter_x2 = sycl::fmin(x2_1, x2_2);
            float inter_y2 = sycl::fmin(y2_1, y2_2);

            float inter_w = sycl::fmax(0.0f, inter_x2 - inter_x1);
            float inter_h = sycl::fmax(0.0f, inter_y2 - inter_y1);
            float inter_area = inter_w * inter_h;

            float area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
            float area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
            float union_area = area1 + area2 - inter_area;

            float iou = inter_area / (union_area + 1e-7f);

            if (iou_type == 1) {
                float enc_x1 = sycl::fmin(x1_1, x1_2);
                float enc_y1 = sycl::fmin(y1_1, y1_2);
                float enc_x2 = sycl::fmax(x2_1, x2_2);
                float enc_y2 = sycl::fmax(y2_1, y2_2);
                float enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1);
                iou = iou - (enc_area - union_area) / (enc_area + 1e-7f);
            } else if (iou_type == 2 || iou_type == 3) {
                // DIoU (2) / CIoU (3): center-distance (+ aspect-ratio) penalty,
                // computed in float accumulation. Matches the Float32 path/CPU.
                float enc_x1 = sycl::fmin(x1_1, x1_2);
                float enc_y1 = sycl::fmin(y1_1, y1_2);
                float enc_x2 = sycl::fmax(x2_1, x2_2);
                float enc_y2 = sycl::fmax(y2_1, y2_2);
                float enc_w = enc_x2 - enc_x1;
                float enc_h = enc_y2 - enc_y1;
                float cx1 = (x1_1 + x2_1) * 0.5f, cy1 = (y1_1 + y2_1) * 0.5f;
                float cx2 = (x1_2 + x2_2) * 0.5f, cy2 = (y1_2 + y2_2) * 0.5f;
                float center_dist_sq = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);
                float diag_dist_sq = enc_w * enc_w + enc_h * enc_h;
                float result = iou - center_dist_sq / (diag_dist_sq + 1e-7f);
                if (iou_type == 3) {
                    const float four_over_pi_sq =
                        4.0f / (3.14159265358979323846f * 3.14159265358979323846f);
                    float w1 = x2_1 - x1_1, h1 = y2_1 - y1_1;
                    float w2 = x2_2 - x1_2, h2 = y2_2 - y1_2;
                    float at = sycl::atan(w2 / (h2 + 1e-7f)) - sycl::atan(w1 / (h1 + 1e-7f));
                    float v = four_over_pi_sq * at * at;
                    float alpha = v / (1.0f - iou + v + 1e-7f);  // raw iou
                    result = result - alpha * v;
                }
                iou = result;
            }

            out_ptr[i * M + j] = f32_to_bf16(iou);
        });
    } else {
        throw std::runtime_error(
            "box_iou_kernel: unsupported dtype " +
            std::string(dtype_name(boxes1.dtype())) +
            " (need Float32, Float64, Float16, or BFloat16)");
    }

    return output;
}

// =========================================================================
// D3-followup OneAPI: bilinear backward via sycl::atomic_ref<float/double>
// fetch_add scatter. Mirrors the CUDA `interpolate_bilinear_backward_kernel`
// 1:1; the only thing changing is the atomic primitive
// (`atomicAdd` → `sycl::atomic_ref<T, relaxed, device>::fetch_add`).
// =========================================================================
template <typename T, typename KernelName>
static void interp_bilinear_backward_dispatch(
    const T* grad_out_ptr, T* grad_in_ptr,
    int64_t N, int64_t C, int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w,
    bool align_corners, sycl::queue& queue)
{
    const int64_t total = N * C * out_h * out_w;
    queue.parallel_for<KernelName>(sycl::range<1>(total), [=](sycl::id<1> id) {
        int64_t idx = id[0];
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t c  = temp % C; temp /= C;
        int64_t b  = temp;

        T y, x;
        if (align_corners) {
            y = (out_h > 1) ? oh * static_cast<T>(in_h - 1) / static_cast<T>(out_h - 1) : T(0);
            x = (out_w > 1) ? ow * static_cast<T>(in_w - 1) / static_cast<T>(out_w - 1) : T(0);
        } else {
            T scale_h = static_cast<T>(in_h) / static_cast<T>(out_h);
            T scale_w = static_cast<T>(in_w) / static_cast<T>(out_w);
            y = (static_cast<T>(oh) + T(0.5)) * scale_h - T(0.5);
            x = (static_cast<T>(ow) + T(0.5)) * scale_w - T(0.5);
        }
        if (y < T(0)) y = T(0);
        if (x < T(0)) x = T(0);
        if (y > static_cast<T>(in_h - 1)) y = static_cast<T>(in_h - 1);
        if (x > static_cast<T>(in_w - 1)) x = static_cast<T>(in_w - 1);

        int64_t y0 = static_cast<int64_t>(y);
        int64_t x0 = static_cast<int64_t>(x);
        int64_t y1 = sycl::min(y0 + 1, in_h - 1);
        int64_t x1 = sycl::min(x0 + 1, in_w - 1);
        T fy = y - static_cast<T>(y0);
        T fx = x - static_cast<T>(x0);

        T w00 = (T(1) - fy) * (T(1) - fx);
        T w01 = (T(1) - fy) * fx;
        T w10 = fy * (T(1) - fx);
        T w11 = fy * fx;

        T g = grad_out_ptr[idx];
        int64_t base_idx = b * (C * in_h * in_w) + c * (in_h * in_w);

        // sycl::atomic_ref<T> fetch_add is supported for float and double.
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
            a00(grad_in_ptr[base_idx + y0 * in_w + x0]);
        a00.fetch_add(w00 * g);
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
            a01(grad_in_ptr[base_idx + y0 * in_w + x1]);
        a01.fetch_add(w01 * g);
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
            a10(grad_in_ptr[base_idx + y1 * in_w + x0]);
        a10.fetch_add(w10 * g);
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
            a11(grad_in_ptr[base_idx + y1 * in_w + x1]);
        a11.fetch_add(w11 * g);
    }).wait();
}

class InterpolateBilinearBackwardKernelFloat32;
class InterpolateBilinearBackwardKernelFloat64;
class InterpolateBicubicBackwardKernelFloat32;
class InterpolateBicubicBackwardKernelFloat64;
class InterpolateTrilinearBackwardKernelFloat32;
class InterpolateTrilinearBackwardKernelFloat64;

// Cubic convolution weight (a=-0.75); matches CPU cubic_interp_coeff and
// PyTorch upsample_bicubic2d.
template <typename T>
static inline T tz_cubic_w(T x) {
    T a = sycl::fabs(x);
    if (a <= T(1)) return T(1.25) * a * a * a - T(2.25) * a * a + T(1);
    if (a < T(2))  return T(-0.75) * a * a * a + T(3.75) * a * a - T(6) * a + T(3);
    return T(0);
}

// Bicubic backward (4D): scatter each output gradient to its 4x4 neighborhood (clamped).
template <typename T, typename KernelName>
static void interp_bicubic_backward_dispatch(
    const T* grad_out_ptr, T* grad_in_ptr,
    int64_t N, int64_t C, int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w, bool align_corners, sycl::queue& queue)
{
    const int64_t total = N * C * out_h * out_w;
    queue.parallel_for<KernelName>(sycl::range<1>(total), [=](sycl::id<1> id) {
        int64_t idx = id[0], temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t c  = temp % C; temp /= C; int64_t b = temp;
        T scale_h = (align_corners && out_h > 1) ? static_cast<T>(in_h - 1) / static_cast<T>(out_h - 1) : static_cast<T>(in_h) / static_cast<T>(out_h);
        T scale_w = (align_corners && out_w > 1) ? static_cast<T>(in_w - 1) / static_cast<T>(out_w - 1) : static_cast<T>(in_w) / static_cast<T>(out_w);
        T src_h = align_corners ? static_cast<T>(oh) * scale_h : (static_cast<T>(oh) + T(0.5)) * scale_h - T(0.5);
        T src_w = align_corners ? static_cast<T>(ow) * scale_w : (static_cast<T>(ow) + T(0.5)) * scale_w - T(0.5);
        // Clamp the continuous coord to [0,in-1] BEFORE flooring, exactly as the
        // forward kernel and CPU reference do, so this scatter is the precise
        // transpose of the forward at the borders (parity at edges).
        src_h = sycl::clamp(src_h, T(0), static_cast<T>(in_h - 1));
        src_w = sycl::clamp(src_w, T(0), static_cast<T>(in_w - 1));
        int64_t hi = static_cast<int64_t>(sycl::floor(src_h));
        int64_t wi = static_cast<int64_t>(sycl::floor(src_w));
        T g = grad_out_ptr[idx];
        int64_t base = b * (C * in_h * in_w) + c * (in_h * in_w);
        for (int dy = -1; dy <= 2; ++dy) {
            int64_t iy = sycl::min(sycl::max(static_cast<int64_t>(0), hi + dy), in_h - 1);
            T wy = tz_cubic_w<T>(src_h - static_cast<T>(hi + dy));
            for (int dx = -1; dx <= 2; ++dx) {
                int64_t ix = sycl::min(sycl::max(static_cast<int64_t>(0), wi + dx), in_w - 1);
                T wx = tz_cubic_w<T>(src_w - static_cast<T>(wi + dx));
                sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
                    a(grad_in_ptr[base + iy * in_w + ix]);
                a.fetch_add(wy * wx * g);
            }
        }
    }).wait();
}

// Trilinear backward (5D): scatter to 8 corners; out-of-bounds corners skipped (CPU semantics).
template <typename T, typename KernelName>
static void interp_trilinear_backward_dispatch(
    const T* grad_out_ptr, T* grad_in_ptr,
    int64_t N, int64_t C, int64_t in_d, int64_t in_h, int64_t in_w,
    int64_t out_d, int64_t out_h, int64_t out_w, bool align_corners, sycl::queue& queue)
{
    const int64_t total = N * C * out_d * out_h * out_w;
    queue.parallel_for<KernelName>(sycl::range<1>(total), [=](sycl::id<1> id) {
        int64_t idx = id[0], temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t od = temp % out_d; temp /= out_d;
        int64_t c  = temp % C; temp /= C; int64_t b = temp;
        T scd = (align_corners && out_d > 1) ? static_cast<T>(in_d - 1) / static_cast<T>(out_d - 1) : static_cast<T>(in_d) / static_cast<T>(out_d);
        T sch = (align_corners && out_h > 1) ? static_cast<T>(in_h - 1) / static_cast<T>(out_h - 1) : static_cast<T>(in_h) / static_cast<T>(out_h);
        T scw = (align_corners && out_w > 1) ? static_cast<T>(in_w - 1) / static_cast<T>(out_w - 1) : static_cast<T>(in_w) / static_cast<T>(out_w);
        T sd = align_corners ? static_cast<T>(od) * scd : (static_cast<T>(od) + T(0.5)) * scd - T(0.5);
        T sh = align_corners ? static_cast<T>(oh) * sch : (static_cast<T>(oh) + T(0.5)) * sch - T(0.5);
        T sw = align_corners ? static_cast<T>(ow) * scw : (static_cast<T>(ow) + T(0.5)) * scw - T(0.5);
        // Clamp the source coord to [0, in-1] BEFORE flooring so the backward is
        // the exact transpose of the (clamping) forward at the borders. Without
        // this a negative src (e.g. od=0 -> src=-0.25) floors to -1, its (1-f)
        // tap is dropped by the bounds check, and the entire border plane loses
        // that gradient mass — diverging from CPU. Matches interpolate_trilinear_
        // backward_impl in src/backends/cpu/kernels/vision.cpp.
        sd = sycl::clamp(sd, T(0), static_cast<T>(in_d - 1));
        sh = sycl::clamp(sh, T(0), static_cast<T>(in_h - 1));
        sw = sycl::clamp(sw, T(0), static_cast<T>(in_w - 1));
        int64_t d0 = static_cast<int64_t>(sycl::floor(sd)), h0 = static_cast<int64_t>(sycl::floor(sh)), w0 = static_cast<int64_t>(sycl::floor(sw));
        int64_t d1 = std::min(d0 + 1, in_d - 1), h1 = std::min(h0 + 1, in_h - 1), w1 = std::min(w0 + 1, in_w - 1);
        T fd = sd - static_cast<T>(d0), fh = sh - static_cast<T>(h0), fw = sw - static_cast<T>(w0);
        T g = grad_out_ptr[idx];
        int64_t base = b * (C * in_d * in_h * in_w) + c * (in_d * in_h * in_w);
        auto add = [&](int64_t dd, int64_t hh, int64_t ww, T wgt) {
            if (dd < 0 || dd >= in_d || hh < 0 || hh >= in_h || ww < 0 || ww >= in_w) return;
            sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
                a(grad_in_ptr[base + (dd) * in_h * in_w + (hh) * in_w + (ww)]);
            a.fetch_add(wgt * g);
        };
        add(d0, h0, w0, (T(1) - fd) * (T(1) - fh) * (T(1) - fw));
        add(d0, h0, w1, (T(1) - fd) * (T(1) - fh) * fw);
        add(d0, h1, w0, (T(1) - fd) * fh * (T(1) - fw));
        add(d0, h1, w1, (T(1) - fd) * fh * fw);
        add(d1, h0, w0, fd * (T(1) - fh) * (T(1) - fw));
        add(d1, h0, w1, fd * (T(1) - fh) * fw);
        add(d1, h1, w0, fd * fh * (T(1) - fw));
        add(d1, h1, w1, fd * fh * fw);
    }).wait();
}

auto interpolate_backward_kernel(const Tensor& grad_output,
                                  const std::vector<int64_t>& input_size,
                                  const std::string& mode,
                                  bool align_corners,
                                  sycl::queue& queue) -> Tensor {
    // The backward scatter uses sycl::atomic_ref, which is only available for
    // Float32/Float64 on this device. Widen Float16/BFloat16 to Float32, run,
    // and narrow back (on-device).
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        const DType orig = grad_output.dtype();
        return interpolate_backward_kernel(grad_output.to(DType::Float32),
                                           input_size, mode, align_corners, queue).to(orig);
    }
    auto shape = grad_output.shape();

    // 1D backward (grad is (N,C,L_out)): reduce to the 4D bilinear/nearest path
    // via a degenerate height axis, mirroring the 1D forward reduction. input_size
    // is the spatial input size [L_in]; the degenerate 4D spatial size is [1,L_in].
    if (shape.size() == 3) {
        if (input_size.size() != 1) {
            throw std::runtime_error("interpolate_backward 1D (OneAPI): input_size must be [L_in].");
        }
        const std::string mode2d = (mode == "nearest") ? "nearest" : "bilinear";
        Tensor g = grad_output.unsqueeze(2);  // (N,C,1,L_out)
        Tensor gi = interpolate_backward_kernel(g, {1, input_size[0]}, mode2d, align_corners, queue);
        return gi.squeeze(2);
    }

    // Trilinear backward operates on 5D (N, C, D, H, W).
    if (mode == "trilinear") {
        if (shape.size() != 5)
            throw std::runtime_error("interpolate_backward (OneAPI): trilinear requires 5D (N,C,D,H,W).");
        if (input_size.size() != 3)
            throw std::runtime_error("interpolate_backward (OneAPI): trilinear input_size must be [in_d,in_h,in_w].");
        const int64_t N = shape[0], C = shape[1], out_d = shape[2], out_h = shape[3], out_w = shape[4];
        const int64_t in_d = input_size[0], in_h = input_size[1], in_w = input_size[2];
        Tensor grad_input({N, C, in_d, in_h, in_w}, grad_output.dtype(), grad_output.device());
        queue.memset(grad_input.data_ptr(), 0,
                     static_cast<size_t>(grad_input.numel()) * dtype_size(grad_input.dtype())).wait();
        if (grad_output.dtype() == DType::Float32) {
            interp_trilinear_backward_dispatch<float, InterpolateTrilinearBackwardKernelFloat32>(
                get_data_ptr<const float>(grad_output), get_data_ptr<float>(grad_input),
                N, C, in_d, in_h, in_w, out_d, out_h, out_w, align_corners, queue);
        } else if (grad_output.dtype() == DType::Float64) {
            interp_trilinear_backward_dispatch<double, InterpolateTrilinearBackwardKernelFloat64>(
                get_data_ptr<const double>(grad_output), get_data_ptr<double>(grad_input),
                N, C, in_d, in_h, in_w, out_d, out_h, out_w, align_corners, queue);
        } else {
            throw std::runtime_error("interpolate_backward trilinear (OneAPI): only Float32/Float64 supported.");
        }
        return grad_input;
    }

    if (mode != "bilinear" && mode != "nearest" && mode != "bicubic") {
        throw std::runtime_error("interpolate_backward (OneAPI): mode '" + mode +
                                  "' not supported. Use 'bilinear'/'nearest'/'bicubic' (4D) or 'trilinear' (5D).");
    }
    if (shape.size() != 4) {
        throw std::runtime_error("interpolate_backward (OneAPI): bilinear/nearest/bicubic require 4D (N,C,H,W).");
    }
    if (input_size.size() != 2) {
        throw std::runtime_error("interpolate_backward (OneAPI): input_size must be [in_h, in_w].");
    }
    const int64_t N     = shape[0];
    const int64_t C     = shape[1];
    const int64_t out_h = shape[2];
    const int64_t out_w = shape[3];
    const int64_t in_h  = input_size[0];
    const int64_t in_w  = input_size[1];

    Tensor grad_input({N, C, in_h, in_w}, grad_output.dtype(), grad_output.device());
    queue.memset(grad_input.data_ptr(), 0,
                 static_cast<size_t>(grad_input.numel()) * dtype_size(grad_input.dtype())).wait();

    const bool is_bicubic = (mode == "bicubic");
    if (grad_output.dtype() == DType::Float32) {
        if (is_bicubic)
            interp_bicubic_backward_dispatch<float, InterpolateBicubicBackwardKernelFloat32>(
                get_data_ptr<const float>(grad_output), get_data_ptr<float>(grad_input),
                N, C, in_h, in_w, out_h, out_w, align_corners, queue);
        else
            interp_bilinear_backward_dispatch<float, InterpolateBilinearBackwardKernelFloat32>(
                get_data_ptr<const float>(grad_output), get_data_ptr<float>(grad_input),
                N, C, in_h, in_w, out_h, out_w, align_corners, queue);
    } else if (grad_output.dtype() == DType::Float64) {
        if (is_bicubic)
            interp_bicubic_backward_dispatch<double, InterpolateBicubicBackwardKernelFloat64>(
                get_data_ptr<const double>(grad_output), get_data_ptr<double>(grad_input),
                N, C, in_h, in_w, out_h, out_w, align_corners, queue);
        else
            interp_bilinear_backward_dispatch<double, InterpolateBilinearBackwardKernelFloat64>(
                get_data_ptr<const double>(grad_output), get_data_ptr<double>(grad_input),
                N, C, in_h, in_w, out_h, out_w, align_corners, queue);
    } else {
        throw std::runtime_error(
            "interpolate_backward (OneAPI): unsupported dtype " +
            std::string(dtype_name(grad_output.dtype())) +
            ". Only Float32 and Float64 are supported (sycl::atomic_ref availability).");
    }

    return grad_input;
}

} // namespace oneapi
} // namespace tenzor
