/**
 * @file roi_align.cu
 * @brief CUDA implementation of ROI Align with bilinear interpolation
 *
 * Supports Float32 and Float64 dtypes via templated kernels.
 * Float16 inputs are promoted to Float32, computed, then converted back.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace cuda {

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define ROI_CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error in roi_align: ") + cudaGetErrorString(err)); \
    } \
} while(0)

// ============================================================================
// Device Helper: Bilinear Interpolation
// ============================================================================

template<typename T>
__device__ inline T bilinear_interpolate_device(const T* data, int64_t height,
                                                 int64_t width, T y, T x) {
    if (y < T(-1.0) || y > T(height) || x < T(-1.0) || x > T(width)) {
        return T(0);
    }

    y = max(T(0), min(y, T(height - 1)));
    x = max(T(0), min(x, T(width - 1)));

    int64_t y_low = static_cast<int64_t>(floor(double(y)));
    int64_t x_low = static_cast<int64_t>(floor(double(x)));
    int64_t y_high = min(y_low + 1, height - 1);
    int64_t x_high = min(x_low + 1, width - 1);

    T ly = y - T(y_low);
    T lx = x - T(x_low);
    T hy = T(1) - ly;
    T hx = T(1) - lx;

    T v1 = data[y_low * width + x_low];
    T v2 = data[y_low * width + x_high];
    T v3 = data[y_high * width + x_low];
    T v4 = data[y_high * width + x_high];

    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

// ============================================================================
// Forward Kernel
// ============================================================================

template<typename T>
__global__ void roi_align_forward_kernel(
    const T* features,    // (N, C, H, W)
    const T* rois,        // (num_rois, 5): (batch_idx, x1, y1, x2, y2)
    T* output,            // (num_rois, C, output_h, output_w)
    int64_t num_rois, int64_t channels, int64_t feat_height, int64_t feat_width,
    int64_t output_h, int64_t output_w, T spatial_scale,
    int64_t sampling_ratio, bool aligned) {

    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_outputs = num_rois * channels * output_h * output_w;

    if (index >= total_outputs) return;

    const int64_t pw = index % output_w;
    const int64_t ph = (index / output_w) % output_h;
    const int64_t c = (index / (output_w * output_h)) % channels;
    const int64_t roi_idx = index / (output_w * output_h * channels);

    const T* roi = rois + roi_idx * 5;
    const int64_t batch_idx = static_cast<int64_t>(roi[0]);

    T roi_x1 = roi[1] * spatial_scale;
    T roi_y1 = roi[2] * spatial_scale;
    T roi_x2 = roi[3] * spatial_scale;
    T roi_y2 = roi[4] * spatial_scale;

    if (aligned) {
        roi_x1 -= T(0.5);
        roi_y1 -= T(0.5);
        roi_x2 -= T(0.5);
        roi_y2 -= T(0.5);
    }

    T roi_width = roi_x2 - roi_x1;
    T roi_height = roi_y2 - roi_y1;

    T bin_size_h = roi_height / T(output_h);
    T bin_size_w = roi_width / T(output_w);

    int64_t roi_bin_grid_h =
        (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(ceil(double(bin_size_h)));
    int64_t roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(ceil(double(bin_size_w)));

    const int64_t count = roi_bin_grid_h * roi_bin_grid_w;

    const T* channel_features =
        features + (batch_idx * channels + c) * feat_height * feat_width;

    T sum = T(0);
    for (int64_t iy = 0; iy < roi_bin_grid_h; ++iy) {
        for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
            T y = roi_y1 + T(ph) * bin_size_h +
                  (T(iy) + T(0.5)) * bin_size_h / T(roi_bin_grid_h);
            T x = roi_x1 + T(pw) * bin_size_w +
                  (T(ix) + T(0.5)) * bin_size_w / T(roi_bin_grid_w);

            sum += bilinear_interpolate_device(channel_features, feat_height,
                                                feat_width, y, x);
        }
    }

    output[index] = sum / T(count);
}

// ============================================================================
// Backward Kernel
// ============================================================================

template<typename T>
__global__ void roi_align_backward_kernel(
    const T* grad_output,    // (num_rois, C, output_h, output_w)
    const T* rois,           // (num_rois, 5)
    T* grad_features,        // (N, C, H, W)
    int64_t num_rois, int64_t channels, int64_t feat_height, int64_t feat_width,
    int64_t output_h, int64_t output_w, T spatial_scale,
    int64_t sampling_ratio, bool aligned) {

    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_grads = num_rois * channels * output_h * output_w;

    if (index >= total_grads) return;

    const int64_t pw = index % output_w;
    const int64_t ph = (index / output_w) % output_h;
    const int64_t c = (index / (output_w * output_h)) % channels;
    const int64_t roi_idx = index / (output_w * output_h * channels);

    const T* roi = rois + roi_idx * 5;
    const int64_t batch_idx = static_cast<int64_t>(roi[0]);

    T roi_x1 = roi[1] * spatial_scale;
    T roi_y1 = roi[2] * spatial_scale;
    T roi_x2 = roi[3] * spatial_scale;
    T roi_y2 = roi[4] * spatial_scale;

    if (aligned) {
        roi_x1 -= T(0.5);
        roi_y1 -= T(0.5);
        roi_x2 -= T(0.5);
        roi_y2 -= T(0.5);
    }

    T roi_width = roi_x2 - roi_x1;
    T roi_height = roi_y2 - roi_y1;

    T bin_size_h = roi_height / T(output_h);
    T bin_size_w = roi_width / T(output_w);

    int64_t roi_bin_grid_h =
        (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(ceil(double(bin_size_h)));
    int64_t roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(ceil(double(bin_size_w)));

    const int64_t count = roi_bin_grid_h * roi_bin_grid_w;

    const T grad_val = grad_output[index] / T(count);

    T* channel_grad_features =
        grad_features + (batch_idx * channels + c) * feat_height * feat_width;

    for (int64_t iy = 0; iy < roi_bin_grid_h; ++iy) {
        for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
            T y = roi_y1 + T(ph) * bin_size_h +
                  (T(iy) + T(0.5)) * bin_size_h / T(roi_bin_grid_h);
            T x = roi_x1 + T(pw) * bin_size_w +
                  (T(ix) + T(0.5)) * bin_size_w / T(roi_bin_grid_w);

            if (y < T(-1.0) || y > T(feat_height) || x < T(-1.0) || x > T(feat_width)) {
                continue;
            }

            y = max(T(0), min(y, T(feat_height - 1)));
            x = max(T(0), min(x, T(feat_width - 1)));

            int64_t y_low = static_cast<int64_t>(floor(double(y)));
            int64_t x_low = static_cast<int64_t>(floor(double(x)));
            int64_t y_high = min(y_low + 1, feat_height - 1);
            int64_t x_high = min(x_low + 1, feat_width - 1);

            T ly = y - T(y_low);
            T lx = x - T(x_low);
            T hy = T(1) - ly;
            T hx = T(1) - lx;

            atomicAdd(&channel_grad_features[y_low * feat_width + x_low],
                      grad_val * hy * hx);
            atomicAdd(&channel_grad_features[y_low * feat_width + x_high],
                      grad_val * hy * lx);
            atomicAdd(&channel_grad_features[y_high * feat_width + x_low],
                      grad_val * ly * hx);
            atomicAdd(&channel_grad_features[y_high * feat_width + x_high],
                      grad_val * ly * lx);
        }
    }
}

// ============================================================================
// Tensor-Level Host Wrappers
// ============================================================================

auto roi_align_forward(const Tensor& features, const Tensor& rois,
                       int64_t output_h, int64_t output_w,
                       float spatial_scale, int64_t sampling_ratio,
                       bool aligned) -> Tensor {
    auto shape = features.shape();
    int64_t batch_size = shape[0];
    int64_t channels = shape[1];
    int64_t feat_height = shape[2];
    int64_t feat_width = shape[3];
    int64_t num_rois = rois.shape()[0];

    // Determine working dtype: Float16 promotes to Float32
    DType original_dtype = features.dtype();
    DType compute_dtype = original_dtype;
    if (compute_dtype == DType::Float16 || compute_dtype == DType::BFloat16) {
        compute_dtype = DType::Float32;
    }

    Tensor feat_in = (compute_dtype != original_dtype) ? features.to(compute_dtype) : features;
    Tensor rois_in = rois.to(compute_dtype);

    std::vector<int64_t> output_shape = {num_rois, channels, output_h, output_w};
    Tensor output(output_shape, compute_dtype, features.device());

    int64_t total_outputs = num_rois * channels * output_h * output_w;
    if (total_outputs == 0) {
        return (compute_dtype != original_dtype) ? output.to(original_dtype) : output;
    }

    const int threads = 512;
    const int blocks = (total_outputs + threads - 1) / threads;

    if (compute_dtype == DType::Float32) {
        roi_align_forward_kernel<float><<<blocks, threads>>>(
            feat_in.data<float>(), rois_in.data<float>(), output.data<float>(),
            num_rois, channels, feat_height, feat_width,
            output_h, output_w, spatial_scale, sampling_ratio, aligned);
    } else if (compute_dtype == DType::Float64) {
        roi_align_forward_kernel<double><<<blocks, threads>>>(
            feat_in.data<double>(), rois_in.data<double>(), output.data<double>(),
            num_rois, channels, feat_height, feat_width,
            output_h, output_w, static_cast<double>(spatial_scale),
            sampling_ratio, aligned);
    } else {
        throw std::runtime_error("roi_align_forward: Unsupported dtype");
    }

    ROI_CUDA_CHECK(cudaGetLastError());
    ROI_CUDA_CHECK(cudaDeviceSynchronize());

    return (compute_dtype != original_dtype) ? output.to(original_dtype) : output;
}

auto roi_align_backward(const Tensor& grad_output, const Tensor& rois,
                        int64_t batch_size, int64_t feat_height, int64_t feat_width,
                        float spatial_scale, int64_t sampling_ratio,
                        bool aligned) -> Tensor {
    int64_t num_rois = rois.shape()[0];
    int64_t channels = grad_output.shape()[1];
    int64_t output_h = grad_output.shape()[2];
    int64_t output_w = grad_output.shape()[3];

    DType original_dtype = grad_output.dtype();
    DType compute_dtype = original_dtype;
    if (compute_dtype == DType::Float16 || compute_dtype == DType::BFloat16) {
        compute_dtype = DType::Float32;
    }

    Tensor grad_in = (compute_dtype != original_dtype) ? grad_output.to(compute_dtype) : grad_output;
    Tensor rois_in = rois.to(compute_dtype);

    std::vector<int64_t> grad_shape = {batch_size, channels, feat_height, feat_width};
    Tensor grad_features(grad_shape, compute_dtype, grad_output.device());

    // Zero out the entire gradient tensor (all batches)
    int64_t total_features = batch_size * channels * feat_height * feat_width;
    if (compute_dtype == DType::Float32) {
        ROI_CUDA_CHECK(cudaMemset(grad_features.data<float>(), 0, total_features * sizeof(float)));
    } else if (compute_dtype == DType::Float64) {
        ROI_CUDA_CHECK(cudaMemset(grad_features.data<double>(), 0, total_features * sizeof(double)));
    }

    int64_t total_grads = num_rois * channels * output_h * output_w;
    if (total_grads == 0) {
        return (compute_dtype != original_dtype) ? grad_features.to(original_dtype) : grad_features;
    }

    const int threads = 512;
    const int blocks = (total_grads + threads - 1) / threads;

    if (compute_dtype == DType::Float32) {
        roi_align_backward_kernel<float><<<blocks, threads>>>(
            grad_in.data<float>(), rois_in.data<float>(), grad_features.data<float>(),
            num_rois, channels, feat_height, feat_width,
            output_h, output_w, spatial_scale, sampling_ratio, aligned);
    } else if (compute_dtype == DType::Float64) {
        roi_align_backward_kernel<double><<<blocks, threads>>>(
            grad_in.data<double>(), rois_in.data<double>(), grad_features.data<double>(),
            num_rois, channels, feat_height, feat_width,
            output_h, output_w, static_cast<double>(spatial_scale),
            sampling_ratio, aligned);
    } else {
        throw std::runtime_error("roi_align_backward: Unsupported dtype");
    }

    ROI_CUDA_CHECK(cudaGetLastError());
    ROI_CUDA_CHECK(cudaDeviceSynchronize());

    return (compute_dtype != original_dtype) ? grad_features.to(original_dtype) : grad_features;
}

} // namespace cuda
} // namespace tenzor
