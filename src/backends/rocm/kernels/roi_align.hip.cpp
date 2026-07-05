/**
 * @file roi_align.hip.cpp
 * @brief HIP implementation of ROI Align with bilinear interpolation for AMD GPUs
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include "tenzor/core/tensor.hpp"

#define HIP_ROI_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            throw std::runtime_error( \
                std::string("HIP error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + hipGetErrorString(err) \
            ); \
        } \
    } while(0)

namespace tenzor {
namespace rocm {

// Bilinear interpolation on device
template <typename T>
__device__ inline T bilinear_interpolate_hip(const T* data, int64_t height,
                                             int64_t width, T y, T x) {
    // Empty feature map: clamping below would yield y_low=x_low=0 and read
    // data[0] from a zero-element buffer (OOB). Bail before any indexing.
    if (height <= 0 || width <= 0) {
        return T(0);
    }
    // Handle out of bounds
    if (y < T(-1) || y > static_cast<T>(height) || x < T(-1) || x > static_cast<T>(width)) {
        return T(0);
    }

    // Clamp to valid range
    y = fmax(T(0), fmin(y, static_cast<T>(height - 1)));
    x = fmax(T(0), fmin(x, static_cast<T>(width - 1)));

    // Integer coordinates
    int64_t y_low = static_cast<int64_t>(floor(y));
    int64_t x_low = static_cast<int64_t>(floor(x));
    int64_t y_high = min(y_low + 1, height - 1);
    int64_t x_high = min(x_low + 1, width - 1);

    // Interpolation weights
    T ly = y - static_cast<T>(y_low);
    T lx = x - static_cast<T>(x_low);
    T hy = T(1) - ly;
    T hx = T(1) - lx;

    // Get values at four corners
    T v1 = data[y_low * width + x_low];
    T v2 = data[y_low * width + x_high];
    T v3 = data[y_high * width + x_low];
    T v4 = data[y_high * width + x_high];

    // Bilinear interpolation
    T w1 = hy * hx;
    T w2 = hy * lx;
    T w3 = ly * hx;
    T w4 = ly * lx;

    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

// Forward kernel: each thread processes one output element
template <typename T>
__global__ void roi_align_forward_kernel(
    const T* features,      // (N, C, H, W)
    const float* rois,      // (num_rois, 5): (batch_idx, x1, y1, x2, y2)
    T* output,              // (num_rois, C, output_h, output_w)
    int64_t num_rois, int64_t channels, int64_t feat_height, int64_t feat_width,
    int64_t output_h, int64_t output_w, float spatial_scale,
    int64_t sampling_ratio, bool aligned, int64_t batch_size) {

    // Global thread index
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_outputs = num_rois * channels * output_h * output_w;

    if (index >= total_outputs) return;

    // Decode output position
    const int64_t pw = index % output_w;
    const int64_t ph = (index / output_w) % output_h;
    const int64_t c = (index / (output_w * output_h)) % channels;
    const int64_t roi_idx = index / (output_w * output_h * channels);

    // Get ROI
    const float* roi = rois + roi_idx * 5;
    const int64_t batch_idx = static_cast<int64_t>(roi[0]);

    // Guard against malformed ROIs: a batch index outside [0, batch_size) would
    // index features out of bounds. Emit 0 for such elements instead of an OOB read.
    if (batch_idx < 0 || batch_idx >= batch_size) {
        output[index] = T(0);
        return;
    }

    // Scale ROI coordinates
    const T scale = static_cast<T>(spatial_scale);
    T roi_x1 = static_cast<T>(roi[1]) * scale;
    T roi_y1 = static_cast<T>(roi[2]) * scale;
    T roi_x2 = static_cast<T>(roi[3]) * scale;
    T roi_y2 = static_cast<T>(roi[4]) * scale;

    if (aligned) {
        roi_x1 -= T(0.5);
        roi_y1 -= T(0.5);
        roi_x2 -= T(0.5);
        roi_y2 -= T(0.5);
    }

    // ROI dimensions
    T roi_width = roi_x2 - roi_x1;
    T roi_height = roi_y2 - roi_y1;
    // torchvision clamp: for non-aligned ROIs, floor the extent to 1 so a
    // degenerate/negative ROI still samples a real pixel. Matches CPU
    // vision.cpp and CUDA roi_align.cu.
    if (!aligned) {
        roi_width = fmax(roi_width, T(1));
        roi_height = fmax(roi_height, T(1));
    }

    // Bin dimensions
    T bin_size_h = roi_height / static_cast<T>(output_h);
    T bin_size_w = roi_width / static_cast<T>(output_w);

    // Sampling grid
    int64_t roi_bin_grid_h =
        (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(ceil(bin_size_h));
    int64_t roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(ceil(bin_size_w));
    // torchvision clamp: sampling grid is at least 1x1.
    roi_bin_grid_h = max(roi_bin_grid_h, int64_t(1));
    roi_bin_grid_w = max(roi_bin_grid_w, int64_t(1));

    const int64_t count = roi_bin_grid_h * roi_bin_grid_w;

    // Get feature map for this batch and channel
    const T* channel_features =
        features + (batch_idx * channels + c) * feat_height * feat_width;

    // Sample and average
    T sum = T(0);
    for (int64_t iy = 0; iy < roi_bin_grid_h; ++iy) {
        for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
            // Compute sample position
            T y = roi_y1 + static_cast<T>(ph) * bin_size_h +
                  (static_cast<T>(iy) + T(0.5)) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h);
            T x = roi_x1 + static_cast<T>(pw) * bin_size_w +
                  (static_cast<T>(ix) + T(0.5)) * bin_size_w /
                      static_cast<T>(roi_bin_grid_w);

            // Bilinear interpolation
            sum += bilinear_interpolate_hip<T>(channel_features, feat_height,
                                               feat_width, y, x);
        }
    }

    // Write output
    output[index] = (count > 0) ? (sum / static_cast<T>(count)) : T(0);
}

// Backward kernel: distribute gradients via atomic adds
template <typename T>
__global__ void roi_align_backward_kernel(
    const T* grad_output,      // (num_rois, C, output_h, output_w)
    const float* rois,         // (num_rois, 5)
    T* grad_features,          // (N, C, H, W)
    int64_t num_rois, int64_t channels, int64_t feat_height, int64_t feat_width,
    int64_t output_h, int64_t output_w, float spatial_scale,
    int64_t sampling_ratio, bool aligned, int64_t batch_size) {

    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_grads = num_rois * channels * output_h * output_w;

    if (index >= total_grads) return;

    // Decode position
    const int64_t pw = index % output_w;
    const int64_t ph = (index / output_w) % output_h;
    const int64_t c = (index / (output_w * output_h)) % channels;
    const int64_t roi_idx = index / (output_w * output_h * channels);

    // Get ROI
    const float* roi = rois + roi_idx * 5;
    const int64_t batch_idx = static_cast<int64_t>(roi[0]);

    // Guard against malformed ROIs: an out-of-range batch index would scatter
    // gradient via atomicAdd into out-of-bounds memory. Skip such elements.
    if (batch_idx < 0 || batch_idx >= batch_size) return;

    const T scale = static_cast<T>(spatial_scale);
    T roi_x1 = static_cast<T>(roi[1]) * scale;
    T roi_y1 = static_cast<T>(roi[2]) * scale;
    T roi_x2 = static_cast<T>(roi[3]) * scale;
    T roi_y2 = static_cast<T>(roi[4]) * scale;

    if (aligned) {
        roi_x1 -= T(0.5);
        roi_y1 -= T(0.5);
        roi_x2 -= T(0.5);
        roi_y2 -= T(0.5);
    }

    T roi_width = roi_x2 - roi_x1;
    T roi_height = roi_y2 - roi_y1;
    // torchvision clamp (must match forward): non-aligned ROI extent floor to 1.
    if (!aligned) {
        roi_width = fmax(roi_width, T(1));
        roi_height = fmax(roi_height, T(1));
    }

    T bin_size_h = roi_height / static_cast<T>(output_h);
    T bin_size_w = roi_width / static_cast<T>(output_w);

    int64_t roi_bin_grid_h =
        (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(ceil(bin_size_h));
    int64_t roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(ceil(bin_size_w));
    // torchvision clamp: sampling grid is at least 1x1.
    roi_bin_grid_h = max(roi_bin_grid_h, int64_t(1));
    roi_bin_grid_w = max(roi_bin_grid_w, int64_t(1));

    const int64_t count = roi_bin_grid_h * roi_bin_grid_w;
    if (count <= 0) return;

    // Get gradient value for this output position
    const T grad_val = grad_output[index] / static_cast<T>(count);

    // Get feature gradient for this batch and channel
    T* channel_grad_features =
        grad_features + (batch_idx * channels + c) * feat_height * feat_width;

    // Distribute gradient to sampled points
    for (int64_t iy = 0; iy < roi_bin_grid_h; ++iy) {
        for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
            T y = roi_y1 + static_cast<T>(ph) * bin_size_h +
                  (static_cast<T>(iy) + T(0.5)) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h);
            T x = roi_x1 + static_cast<T>(pw) * bin_size_w +
                  (static_cast<T>(ix) + T(0.5)) * bin_size_w /
                      static_cast<T>(roi_bin_grid_w);

            // Distribute gradient via bilinear weights
            if (y < T(-1) || y > static_cast<T>(feat_height) ||
                x < T(-1) || x > static_cast<T>(feat_width)) {
                continue;
            }

            y = fmax(T(0), fmin(y, static_cast<T>(feat_height - 1)));
            x = fmax(T(0), fmin(x, static_cast<T>(feat_width - 1)));

            int64_t y_low = static_cast<int64_t>(floor(y));
            int64_t x_low = static_cast<int64_t>(floor(x));
            int64_t y_high = min(y_low + 1, feat_height - 1);
            int64_t x_high = min(x_low + 1, feat_width - 1);

            T ly = y - static_cast<T>(y_low);
            T lx = x - static_cast<T>(x_low);
            T hy = T(1) - ly;
            T hx = T(1) - lx;

            // Atomic add to feature gradients
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
                       bool aligned, hipStream_t stream) -> Tensor {
    // Float16/BFloat16 have no native kernel: widen to Float32, compute, narrow
    // back. Float32 and Float64 are computed NATIVELY via the templated kernel
    // (Float64 used to be downcast to Float32, losing precision vs the CUDA
    // backend which is templated for both).
    const DType orig_dtype = features.dtype();
    if (orig_dtype != DType::Float32 && orig_dtype != DType::Float64) {
        auto features_f32 = features.to(DType::Float32);
        auto result_f32 = roi_align_forward(features_f32, rois, output_h, output_w,
                                            spatial_scale, sampling_ratio, aligned, stream);
        return result_f32.to(orig_dtype);
    }

    // Kernel addresses the feature map with dense NCHW strides derived from
    // shape, so the input must be contiguous.
    const Tensor feat = features.is_contiguous() ? features : features.contiguous();

    auto shape = feat.shape();
    int64_t batch_size = shape[0];
    int64_t channels = shape[1];
    int64_t feat_height = shape[2];
    int64_t feat_width = shape[3];
    int64_t num_rois = rois.shape()[0];

    std::vector<int64_t> output_shape = {num_rois, channels, output_h, output_w};
    Tensor output(output_shape, orig_dtype, feat.device());

    int64_t total_outputs = num_rois * channels * output_h * output_w;
    if (total_outputs == 0) return output;

    // Empty feature map (feat_height/width == 0) with a non-empty ROI grid:
    // every sample reads from a zero-element buffer. Return an all-zero output
    // (matching the bilinear-interpolate "out of bounds -> 0" contract) instead
    // of launching the kernel over OOB reads.
    if (feat_height == 0 || feat_width == 0) {
        HIP_ROI_CHECK(hipMemsetAsync(output.data_ptr(), 0,
                                     total_outputs * dtype_size(orig_dtype), stream));
        return output;
    }

    // ROI coordinates always processed as Float32
    const Tensor rois_f32 = (rois.dtype() == DType::Float32) ? rois : rois.to(DType::Float32);
    const float* rois_ptr = rois_f32.data<float>();

    const int threads = 512;
    const int blocks = (total_outputs + threads - 1) / threads;

    if (orig_dtype == DType::Float32) {
        hipLaunchKernelGGL(roi_align_forward_kernel<float>, dim3(blocks), dim3(threads), 0, stream,
                          feat.data<float>(), rois_ptr, output.data<float>(),
                          num_rois, channels, feat_height, feat_width,
                          output_h, output_w, spatial_scale, sampling_ratio, aligned, batch_size);
    } else {
        hipLaunchKernelGGL(roi_align_forward_kernel<double>, dim3(blocks), dim3(threads), 0, stream,
                          feat.data<double>(), rois_ptr, output.data<double>(),
                          num_rois, channels, feat_height, feat_width,
                          output_h, output_w, spatial_scale, sampling_ratio, aligned, batch_size);
    }

    HIP_ROI_CHECK(hipGetLastError());
    return output;
}

auto roi_align_backward(const Tensor& grad_output, const Tensor& rois,
                        int64_t batch_size, int64_t feat_height, int64_t feat_width,
                        float spatial_scale, int64_t sampling_ratio,
                        bool aligned, hipStream_t stream) -> Tensor {
    // Float16/BFloat16 widen to Float32; Float32/Float64 computed natively.
    const DType orig_dtype = grad_output.dtype();
    if (orig_dtype != DType::Float32 && orig_dtype != DType::Float64) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto result_f32 = roi_align_backward(grad_f32, rois, batch_size, feat_height,
                                             feat_width, spatial_scale, sampling_ratio, aligned, stream);
        return result_f32.to(orig_dtype);
    }

    // Kernel addresses grad_output with dense NCHW strides derived from shape,
    // so the input must be contiguous.
    const Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();

    int64_t num_rois = rois.shape()[0];
    int64_t channels = go.shape()[1];
    int64_t output_h = go.shape()[2];
    int64_t output_w = go.shape()[3];

    std::vector<int64_t> grad_shape = {batch_size, channels, feat_height, feat_width};
    int64_t total_features = batch_size * channels * feat_height * feat_width;
    int64_t total_grads = num_rois * channels * output_h * output_w;

    Tensor grad_features(grad_shape, orig_dtype, go.device());
    HIP_ROI_CHECK(hipMemsetAsync(grad_features.data_ptr(), 0,
                                 total_features * dtype_size(orig_dtype), stream));

    if (total_grads == 0) return grad_features;

    // Empty feature map (feat_height/width == 0) with a non-empty ROI grid:
    // grad_features has 0 elements but the kernel would clamp coordinates to 0
    // and atomicAdd into a zero-element buffer (OOB write). grad_features is
    // already zero-memset, so just return it without launching the kernel,
    // matching the forward path's empty-feature-map guard.
    if (feat_height == 0 || feat_width == 0) return grad_features;

    // ROI coordinates always processed as Float32
    const Tensor rois_f32 = (rois.dtype() == DType::Float32) ? rois : rois.to(DType::Float32);
    const float* rois_ptr = rois_f32.data<float>();

    const int threads = 512;
    const int blocks = (total_grads + threads - 1) / threads;

    if (orig_dtype == DType::Float32) {
        hipLaunchKernelGGL(roi_align_backward_kernel<float>, dim3(blocks), dim3(threads), 0, stream,
                          go.data<float>(), rois_ptr, grad_features.data<float>(),
                          num_rois, channels, feat_height, feat_width,
                          output_h, output_w, spatial_scale, sampling_ratio, aligned, batch_size);
    } else {
        hipLaunchKernelGGL(roi_align_backward_kernel<double>, dim3(blocks), dim3(threads), 0, stream,
                          go.data<double>(), rois_ptr, grad_features.data<double>(),
                          num_rois, channels, feat_height, feat_width,
                          output_h, output_w, spatial_scale, sampling_ratio, aligned, batch_size);
    }

    HIP_ROI_CHECK(hipGetLastError());
    return grad_features;
}

} // namespace rocm
} // namespace tenzor
