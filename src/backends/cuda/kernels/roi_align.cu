/**
 * @file roi_align.cu
 * @brief CUDA implementation of ROI Align with bilinear interpolation
 *
 * Supports Float32, Float64, and Float16 feature dtypes via templated kernels.
 * ROI coordinates are always processed as Float32 regardless of feature dtype,
 * matching the convention used by torchvision and Detectron2. This avoids
 * unnecessary per-branch dtype conversions and eliminates redundant temporary
 * allocations for non-Float32 ROI tensors.
 *
 * Float16 features use a specialized kernel that reads __half, computes in
 * float, and writes __half — avoiding full-tensor FP16→FP32→FP16 conversion.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
    const T* features,       // (N, C, H, W)
    const float* rois,       // (num_rois, 5): (batch_idx, x1, y1, x2, y2) — always float
    T* output,               // (num_rois, C, output_h, output_w)
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

    const float* roi = rois + roi_idx * 5;
    const int64_t batch_idx = static_cast<int64_t>(roi[0]);

    T roi_x1 = T(roi[1]) * spatial_scale;
    T roi_y1 = T(roi[2]) * spatial_scale;
    T roi_x2 = T(roi[3]) * spatial_scale;
    T roi_y2 = T(roi[4]) * spatial_scale;

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
    const float* rois,       // (num_rois, 5) — always float
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

    const float* roi = rois + roi_idx * 5;
    const int64_t batch_idx = static_cast<int64_t>(roi[0]);

    T roi_x1 = T(roi[1]) * spatial_scale;
    T roi_y1 = T(roi[2]) * spatial_scale;
    T roi_x2 = T(roi[3]) * spatial_scale;
    T roi_y2 = T(roi[4]) * spatial_scale;

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
// FP16-Native Kernels (read __half, compute in float, write __half)
// Avoids full-tensor FP16→FP32→FP16 conversion overhead.
// ============================================================================

__device__ inline float bilinear_interpolate_fp16(const __half* data, int64_t height,
                                                   int64_t width, float y, float x) {
    if (y < -1.0f || y > static_cast<float>(height) ||
        x < -1.0f || x > static_cast<float>(width)) {
        return 0.0f;
    }

    y = fmaxf(0.0f, fminf(y, static_cast<float>(height - 1)));
    x = fmaxf(0.0f, fminf(x, static_cast<float>(width - 1)));

    int64_t y_low = static_cast<int64_t>(floorf(y));
    int64_t x_low = static_cast<int64_t>(floorf(x));
    int64_t y_high = min(y_low + 1, height - 1);
    int64_t x_high = min(x_low + 1, width - 1);

    float ly = y - static_cast<float>(y_low);
    float lx = x - static_cast<float>(x_low);
    float hy = 1.0f - ly;
    float hx = 1.0f - lx;

    float v1 = __half2float(data[y_low * width + x_low]);
    float v2 = __half2float(data[y_low * width + x_high]);
    float v3 = __half2float(data[y_high * width + x_low]);
    float v4 = __half2float(data[y_high * width + x_high]);

    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

__global__ void roi_align_forward_fp16_kernel(
    const __half* features,
    const float* rois,       // ROIs always in float — no __half→float per-element conversion
    __half* output,
    int64_t num_rois, int64_t channels, int64_t feat_height, int64_t feat_width,
    int64_t output_h, int64_t output_w, float spatial_scale,
    int64_t sampling_ratio, bool aligned) {

    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_outputs = num_rois * channels * output_h * output_w;

    if (index >= total_outputs) return;

    const int64_t pw = index % output_w;
    const int64_t ph = (index / output_w) % output_h;
    const int64_t c = (index / (output_w * output_h)) % channels;
    const int64_t roi_idx = index / (output_w * output_h * channels);

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

    const __half* channel_features =
        features + (batch_idx * channels + c) * feat_height * feat_width;

    float sum = 0.0f;
    for (int64_t iy = 0; iy < roi_bin_grid_h; ++iy) {
        for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
            float y = roi_y1 + static_cast<float>(ph) * bin_size_h +
                      (static_cast<float>(iy) + 0.5f) * bin_size_h / static_cast<float>(roi_bin_grid_h);
            float x = roi_x1 + static_cast<float>(pw) * bin_size_w +
                      (static_cast<float>(ix) + 0.5f) * bin_size_w / static_cast<float>(roi_bin_grid_w);

            sum += bilinear_interpolate_fp16(channel_features, feat_height,
                                              feat_width, y, x);
        }
    }

    output[index] = __float2half(sum / static_cast<float>(count));
}

__global__ void roi_align_backward_fp16_kernel(
    const __half* grad_output,
    const float* rois,        // ROIs always in float
    float* grad_features,     // accumulate in float to avoid atomicAdd precision loss
    int64_t num_rois, int64_t channels, int64_t feat_height, int64_t feat_width,
    int64_t output_h, int64_t output_w, float spatial_scale,
    int64_t sampling_ratio, bool aligned) {

    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_grads = num_rois * channels * output_h * output_w;

    if (index >= total_grads) return;

    const int64_t pw = index % output_w;
    const int64_t ph = (index / output_w) % output_h;
    const int64_t c = (index / (output_w * output_h)) % channels;
    const int64_t roi_idx = index / (output_w * output_h * channels);

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

    const float grad_val = __half2float(grad_output[index]) / static_cast<float>(count);

    float* channel_grad_features =
        grad_features + (batch_idx * channels + c) * feat_height * feat_width;

    for (int64_t iy = 0; iy < roi_bin_grid_h; ++iy) {
        for (int64_t ix = 0; ix < roi_bin_grid_w; ++ix) {
            float y = roi_y1 + static_cast<float>(ph) * bin_size_h +
                      (static_cast<float>(iy) + 0.5f) * bin_size_h / static_cast<float>(roi_bin_grid_h);
            float x = roi_x1 + static_cast<float>(pw) * bin_size_w +
                      (static_cast<float>(ix) + 0.5f) * bin_size_w / static_cast<float>(roi_bin_grid_w);

            if (y < -1.0f || y > static_cast<float>(feat_height) ||
                x < -1.0f || x > static_cast<float>(feat_width)) {
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

// Conversion kernel: float grad accumulation buffer → __half output
__global__ void f32_grad_to_fp16_kernel(const float* src, __half* dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
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

    DType dtype = features.dtype();

    std::vector<int64_t> output_shape = {num_rois, channels, output_h, output_w};
    Tensor output(output_shape, dtype, features.device());

    int64_t total_outputs = num_rois * channels * output_h * output_w;
    if (total_outputs == 0) {
        return output;
    }

    // ROI coordinates (batch_idx, x1, y1, x2, y2) are always processed as Float32.
    // This is a single conversion shared across all feature dtype branches, and is
    // a no-op when rois are already Float32 (the common case).
    const Tensor rois_f32 = (rois.dtype() == DType::Float32) ? rois : rois.to(DType::Float32);
    const float* rois_ptr = rois_f32.data<float>();

    int min_grid_size, block_size;

    if (dtype == DType::Float32) {
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           roi_align_forward_kernel<float>, 0, 0);
        const int blocks = (total_outputs + block_size - 1) / block_size;
        roi_align_forward_kernel<float><<<blocks, block_size>>>(
            features.data<float>(), rois_ptr, output.data<float>(),
            num_rois, channels, feat_height, feat_width,
            output_h, output_w, spatial_scale, sampling_ratio, aligned);
    } else if (dtype == DType::Float64) {
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           roi_align_forward_kernel<double>, 0, 0);
        const int blocks = (total_outputs + block_size - 1) / block_size;
        roi_align_forward_kernel<double><<<blocks, block_size>>>(
            features.data<double>(), rois_ptr, output.data<double>(),
            num_rois, channels, feat_height, feat_width,
            output_h, output_w, static_cast<double>(spatial_scale),
            sampling_ratio, aligned);
    } else if (dtype == DType::Float16) {
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           roi_align_forward_fp16_kernel, 0, 0);
        const int blocks = (total_outputs + block_size - 1) / block_size;
        roi_align_forward_fp16_kernel<<<blocks, block_size>>>(
            reinterpret_cast<const __half*>(features.data_ptr()),
            rois_ptr,
            reinterpret_cast<__half*>(output.data_ptr()),
            num_rois, channels, feat_height, feat_width,
            output_h, output_w, spatial_scale, sampling_ratio, aligned);
    } else {
        throw std::runtime_error("roi_align_forward: Unsupported dtype");
    }

    ROI_CUDA_CHECK(cudaGetLastError());

    return output;
}

auto roi_align_backward(const Tensor& grad_output, const Tensor& rois,
                        int64_t batch_size, int64_t feat_height, int64_t feat_width,
                        float spatial_scale, int64_t sampling_ratio,
                        bool aligned) -> Tensor {
    int64_t num_rois = rois.shape()[0];
    int64_t channels = grad_output.shape()[1];
    int64_t output_h = grad_output.shape()[2];
    int64_t output_w = grad_output.shape()[3];

    DType dtype = grad_output.dtype();

    std::vector<int64_t> grad_shape = {batch_size, channels, feat_height, feat_width};
    int64_t total_features = batch_size * channels * feat_height * feat_width;
    int64_t total_grads = num_rois * channels * output_h * output_w;

    // ROI coordinates always processed as Float32, shared across all branches.
    const Tensor rois_f32 = (rois.dtype() == DType::Float32) ? rois : rois.to(DType::Float32);
    const float* rois_ptr = rois_f32.data<float>();

    int min_grid_size, block_size;

    if (dtype == DType::Float32) {
        Tensor grad_features(grad_shape, DType::Float32, grad_output.device());
        ROI_CUDA_CHECK(cudaMemset(grad_features.data<float>(), 0, total_features * sizeof(float)));
        if (total_grads == 0) return grad_features;

        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           roi_align_backward_kernel<float>, 0, 0);
        const int blocks = (total_grads + block_size - 1) / block_size;
        roi_align_backward_kernel<float><<<blocks, block_size>>>(
            grad_output.data<float>(), rois_ptr, grad_features.data<float>(),
            num_rois, channels, feat_height, feat_width,
            output_h, output_w, spatial_scale, sampling_ratio, aligned);
        ROI_CUDA_CHECK(cudaGetLastError());
        return grad_features;
    } else if (dtype == DType::Float64) {
        Tensor grad_features(grad_shape, DType::Float64, grad_output.device());
        ROI_CUDA_CHECK(cudaMemset(grad_features.data<double>(), 0, total_features * sizeof(double)));
        if (total_grads == 0) return grad_features;

        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           roi_align_backward_kernel<double>, 0, 0);
        const int blocks = (total_grads + block_size - 1) / block_size;
        roi_align_backward_kernel<double><<<blocks, block_size>>>(
            grad_output.data<double>(), rois_ptr, grad_features.data<double>(),
            num_rois, channels, feat_height, feat_width,
            output_h, output_w, static_cast<double>(spatial_scale),
            sampling_ratio, aligned);
        ROI_CUDA_CHECK(cudaGetLastError());
        return grad_features;
    } else if (dtype == DType::Float16) {
        // Accumulate gradients in float for atomicAdd precision, then convert to FP16
        Tensor grad_features_f32(grad_shape, DType::Float32, grad_output.device());
        ROI_CUDA_CHECK(cudaMemset(grad_features_f32.data<float>(), 0, total_features * sizeof(float)));

        if (total_grads > 0) {
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                               roi_align_backward_fp16_kernel, 0, 0);
            const int blocks = (total_grads + block_size - 1) / block_size;
            roi_align_backward_fp16_kernel<<<blocks, block_size>>>(
                reinterpret_cast<const __half*>(grad_output.data_ptr()),
                rois_ptr,
                grad_features_f32.data<float>(),
                num_rois, channels, feat_height, feat_width,
                output_h, output_w, spatial_scale, sampling_ratio, aligned);
            ROI_CUDA_CHECK(cudaGetLastError());
        }

        // Convert accumulated float gradients to FP16 output
        Tensor grad_features(grad_shape, DType::Float16, grad_output.device());
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                           f32_grad_to_fp16_kernel, 0, 0);
        const int conv_blocks = (total_features + block_size - 1) / block_size;
        f32_grad_to_fp16_kernel<<<conv_blocks, block_size>>>(
            grad_features_f32.data<float>(),
            reinterpret_cast<__half*>(grad_features.data_ptr()),
            total_features);
        ROI_CUDA_CHECK(cudaGetLastError());
        return grad_features;
    } else {
        throw std::runtime_error("roi_align_backward: Unsupported dtype");
    }
}

} // namespace cuda
} // namespace tenzor
