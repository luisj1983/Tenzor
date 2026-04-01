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
__device__ inline float bilinear_interpolate_hip(const float* data, int64_t height,
                                                   int64_t width, float y, float x) {
    // Handle out of bounds
    if (y < -1.0f || y > height || x < -1.0f || x > width) {
        return 0.0f;
    }

    // Clamp to valid range
    y = fmaxf(0.0f, fminf(y, static_cast<float>(height - 1)));
    x = fmaxf(0.0f, fminf(x, static_cast<float>(width - 1)));

    // Integer coordinates
    int64_t y_low = static_cast<int64_t>(floorf(y));
    int64_t x_low = static_cast<int64_t>(floorf(x));
    int64_t y_high = min(y_low + 1, height - 1);
    int64_t x_high = min(x_low + 1, width - 1);

    // Interpolation weights
    float ly = y - static_cast<float>(y_low);
    float lx = x - static_cast<float>(x_low);
    float hy = 1.0f - ly;
    float hx = 1.0f - lx;

    // Get values at four corners
    float v1 = data[y_low * width + x_low];
    float v2 = data[y_low * width + x_high];
    float v3 = data[y_high * width + x_low];
    float v4 = data[y_high * width + x_high];

    // Bilinear interpolation
    float w1 = hy * hx;
    float w2 = hy * lx;
    float w3 = ly * hx;
    float w4 = ly * lx;

    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

// Forward kernel: each thread processes one output element
__global__ void roi_align_forward_kernel(
    const float* features,  // (N, C, H, W)
    const float* rois,      // (num_rois, 5): (batch_idx, x1, y1, x2, y2)
    float* output,          // (num_rois, C, output_h, output_w)
    int64_t num_rois, int64_t channels, int64_t feat_height, int64_t feat_width,
    int64_t output_h, int64_t output_w, float spatial_scale,
    int64_t sampling_ratio, bool aligned) {

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

    // Scale ROI coordinates
    float roi_x1 = roi[1] * spatial_scale;
    float roi_y1 = roi[2] * spatial_scale;
    float roi_x2 = roi[3] * spatial_scale;
    float roi_y2 = roi[4] * spatial_scale;

    if (aligned) {
        roi_x1 -= 0.5f;
        roi_y1 -= 0.5f;
        roi_x2 -= 0.5f;
        roi_y2 -= 0.5f;
    }

    // ROI dimensions
    float roi_width = roi_x2 - roi_x1;
    float roi_height = roi_y2 - roi_y1;

    // Bin dimensions
    float bin_size_h = roi_height / static_cast<float>(output_h);
    float bin_size_w = roi_width / static_cast<float>(output_w);

    // Sampling grid
    int64_t roi_bin_grid_h =
        (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(ceilf(bin_size_h));
    int64_t roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(ceilf(bin_size_w));

    const int64_t count = roi_bin_grid_h * roi_bin_grid_w;

    // Get feature map for this batch and channel
    const float* channel_features =
        features + (batch_idx * channels + c) * feat_height * feat_width;

    // Sample and average
    float sum = 0.0f;
    for (int64_t iy = 0; iy < roi_bin_grid_h; ++iy) {
        for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
            // Compute sample position
            float y = roi_y1 + static_cast<float>(ph) * bin_size_h +
                      (static_cast<float>(iy) + 0.5f) * bin_size_h /
                          static_cast<float>(roi_bin_grid_h);
            float x = roi_x1 + static_cast<float>(pw) * bin_size_w +
                      (static_cast<float>(ix) + 0.5f) * bin_size_w /
                          static_cast<float>(roi_bin_grid_w);

            // Bilinear interpolation
            sum += bilinear_interpolate_hip(channel_features, feat_height,
                                              feat_width, y, x);
        }
    }

    // Write output
    output[index] = sum / static_cast<float>(count);
}

// Backward kernel: distribute gradients via atomic adds
__global__ void roi_align_backward_kernel(
    const float* grad_output,  // (num_rois, C, output_h, output_w)
    const float* rois,         // (num_rois, 5)
    float* grad_features,      // (N, C, H, W)
    int64_t num_rois, int64_t channels, int64_t feat_height, int64_t feat_width,
    int64_t output_h, int64_t output_w, float spatial_scale,
    int64_t sampling_ratio, bool aligned) {

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

    float roi_x1 = roi[1] * spatial_scale;
    float roi_y1 = roi[2] * spatial_scale;
    float roi_x2 = roi[3] * spatial_scale;
    float roi_y2 = roi[4] * spatial_scale;

    if (aligned) {
        roi_x1 -= 0.5f;
        roi_y1 -= 0.5f;
        roi_x2 -= 0.5f;
        roi_y2 -= 0.5f;
    }

    float roi_width = roi_x2 - roi_x1;
    float roi_height = roi_y2 - roi_y1;

    float bin_size_h = roi_height / static_cast<float>(output_h);
    float bin_size_w = roi_width / static_cast<float>(output_w);

    int64_t roi_bin_grid_h =
        (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(ceilf(bin_size_h));
    int64_t roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : static_cast<int64_t>(ceilf(bin_size_w));

    const int64_t count = roi_bin_grid_h * roi_bin_grid_w;

    // Get gradient value for this output position
    const float grad_val = grad_output[index] / static_cast<float>(count);

    // Get feature gradient for this batch and channel
    float* channel_grad_features =
        grad_features + (batch_idx * channels + c) * feat_height * feat_width;

    // Distribute gradient to sampled points
    for (int64_t iy = 0; iy < roi_bin_grid_h; ++iy) {
        for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
            float y = roi_y1 + static_cast<float>(ph) * bin_size_h +
                      (static_cast<float>(iy) + 0.5f) * bin_size_h /
                          static_cast<float>(roi_bin_grid_h);
            float x = roi_x1 + static_cast<float>(pw) * bin_size_w +
                      (static_cast<float>(ix) + 0.5f) * bin_size_w /
                          static_cast<float>(roi_bin_grid_w);

            // Distribute gradient via bilinear weights
            if (y < -1.0f || y > feat_height || x < -1.0f || x > feat_width) {
                continue;
            }

            y = fmaxf(0.0f, fminf(y, static_cast<float>(feat_height - 1)));
            x = fmaxf(0.0f, fminf(x, static_cast<float>(feat_width - 1)));

            int64_t y_low = static_cast<int64_t>(floorf(y));
            int64_t x_low = static_cast<int64_t>(floorf(x));
            int64_t y_high = min(y_low + 1, feat_height - 1);
            int64_t x_high = min(x_low + 1, feat_width - 1);

            float ly = y - static_cast<float>(y_low);
            float lx = x - static_cast<float>(x_low);
            float hy = 1.0f - ly;
            float hx = 1.0f - lx;

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
    // Non-Float32 types: compute in Float32 and convert back to original dtype
    const DType orig_dtype = features.dtype();
    if (orig_dtype != DType::Float32) {
        auto features_f32 = features.to(DType::Float32);
        auto result_f32 = roi_align_forward(features_f32, rois, output_h, output_w,
                                            spatial_scale, sampling_ratio, aligned, stream);
        return result_f32.to(orig_dtype);
    }

    auto shape = features.shape();
    int64_t batch_size = shape[0];
    int64_t channels = shape[1];
    int64_t feat_height = shape[2];
    int64_t feat_width = shape[3];
    int64_t num_rois = rois.shape()[0];

    std::vector<int64_t> output_shape = {num_rois, channels, output_h, output_w};
    Tensor output(output_shape, DType::Float32, features.device());

    int64_t total_outputs = num_rois * channels * output_h * output_w;
    if (total_outputs == 0) return output;

    // ROI coordinates always processed as Float32
    const Tensor rois_f32 = (rois.dtype() == DType::Float32) ? rois : rois.to(DType::Float32);
    const float* rois_ptr = rois_f32.data<float>();

    const int threads = 512;
    const int blocks = (total_outputs + threads - 1) / threads;

    hipLaunchKernelGGL(roi_align_forward_kernel, dim3(blocks), dim3(threads), 0, stream,
                      features.data<float>(), rois_ptr, output.data<float>(),
                      num_rois, channels, feat_height, feat_width,
                      output_h, output_w, spatial_scale, sampling_ratio, aligned);

    HIP_ROI_CHECK(hipGetLastError());
    return output;
}

auto roi_align_backward(const Tensor& grad_output, const Tensor& rois,
                        int64_t batch_size, int64_t feat_height, int64_t feat_width,
                        float spatial_scale, int64_t sampling_ratio,
                        bool aligned, hipStream_t stream) -> Tensor {
    // Non-Float32 types: compute in Float32 and convert back to original dtype
    const DType orig_dtype = grad_output.dtype();
    if (orig_dtype != DType::Float32) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto result_f32 = roi_align_backward(grad_f32, rois, batch_size, feat_height,
                                             feat_width, spatial_scale, sampling_ratio, aligned, stream);
        return result_f32.to(orig_dtype);
    }

    int64_t num_rois = rois.shape()[0];
    int64_t channels = grad_output.shape()[1];
    int64_t output_h = grad_output.shape()[2];
    int64_t output_w = grad_output.shape()[3];

    std::vector<int64_t> grad_shape = {batch_size, channels, feat_height, feat_width};
    int64_t total_features = batch_size * channels * feat_height * feat_width;
    int64_t total_grads = num_rois * channels * output_h * output_w;

    Tensor grad_features(grad_shape, DType::Float32, grad_output.device());
    HIP_ROI_CHECK(hipMemsetAsync(grad_features.data<float>(), 0, total_features * sizeof(float), stream));

    if (total_grads == 0) return grad_features;

    // ROI coordinates always processed as Float32
    const Tensor rois_f32 = (rois.dtype() == DType::Float32) ? rois : rois.to(DType::Float32);
    const float* rois_ptr = rois_f32.data<float>();

    // Ensure grad_output is Float32
    const Tensor grad_f32 = (grad_output.dtype() == DType::Float32) ? grad_output : grad_output.to(DType::Float32);

    const int threads = 512;
    const int blocks = (total_grads + threads - 1) / threads;

    hipLaunchKernelGGL(roi_align_backward_kernel, dim3(blocks), dim3(threads), 0, stream,
                      grad_f32.data<float>(), rois_ptr, grad_features.data<float>(),
                      num_rois, channels, feat_height, feat_width,
                      output_h, output_w, spatial_scale, sampling_ratio, aligned);

    HIP_ROI_CHECK(hipGetLastError());
    return grad_features;
}

} // namespace rocm
} // namespace tenzor
