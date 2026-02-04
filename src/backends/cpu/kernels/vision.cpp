/**
 * @file vision.cpp
 * @brief CPU kernel implementations for vision operations
 *
 * Includes interpolation (resize) operations with nearest, bilinear,
 * and bicubic modes.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor {
namespace cpu {

// ============================================================================
// Interpolation Helper Functions
// ============================================================================

namespace {

// Cubic interpolation coefficient function (Catmull-Rom spline)
inline float cubic_interp_coeff(float x) {
    float abs_x = std::abs(x);
    if (abs_x <= 1.0f) {
        return 1.5f * abs_x * abs_x * abs_x - 2.5f * abs_x * abs_x + 1.0f;
    } else if (abs_x < 2.0f) {
        return -0.5f * abs_x * abs_x * abs_x + 2.5f * abs_x * abs_x - 4.0f * abs_x + 2.0f;
    }
    return 0.0f;
}

// Template for nearest neighbor interpolation
template<typename T>
void interpolate_nearest_impl(
    const T* input,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_h,
    int64_t out_w
) {
    const float scale_h = static_cast<float>(in_h) / out_h;
    const float scale_w = static_cast<float>(in_w) / out_w;

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) if(batch * channels * out_h * out_w > 65536)
    #endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    // Calculate source position
                    int64_t ih = static_cast<int64_t>(oh * scale_h);
                    int64_t iw = static_cast<int64_t>(ow * scale_w);

                    // Clamp to valid range
                    ih = std::clamp(ih, int64_t(0), in_h - 1);
                    iw = std::clamp(iw, int64_t(0), in_w - 1);

                    int64_t in_idx = b * (channels * in_h * in_w) +
                                    c * (in_h * in_w) +
                                    ih * in_w + iw;

                    int64_t out_idx = b * (channels * out_h * out_w) +
                                     c * (out_h * out_w) +
                                     oh * out_w + ow;

                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

// Template for bilinear interpolation
template<typename T>
void interpolate_bilinear_impl(
    const T* input,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_h,
    int64_t out_w,
    bool align_corners
) {
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) if(batch * channels * out_h * out_w > 65536)
    #endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    // Calculate source position (floating point)
                    float y, x;
                    if (align_corners) {
                        // Align corners: map [0, out-1] to [0, in-1]
                        y = (out_h > 1) ? oh * static_cast<float>(in_h - 1) / (out_h - 1) : 0.0f;
                        x = (out_w > 1) ? ow * static_cast<float>(in_w - 1) / (out_w - 1) : 0.0f;
                    } else {
                        // Half-pixel centers: pixels are unit squares
                        float scale_h = static_cast<float>(in_h) / out_h;
                        float scale_w = static_cast<float>(in_w) / out_w;
                        y = (oh + 0.5f) * scale_h - 0.5f;
                        x = (ow + 0.5f) * scale_w - 0.5f;
                    }

                    // Clamp to valid range
                    y = std::clamp(y, 0.0f, static_cast<float>(in_h - 1));
                    x = std::clamp(x, 0.0f, static_cast<float>(in_w - 1));

                    // Get integer and fractional parts
                    int64_t y0 = static_cast<int64_t>(y);
                    int64_t x0 = static_cast<int64_t>(x);
                    int64_t y1 = std::min(y0 + 1, in_h - 1);
                    int64_t x1 = std::min(x0 + 1, in_w - 1);

                    float fy = y - y0;
                    float fx = x - x0;

                    // Bilinear interpolation weights
                    float w00 = (1.0f - fy) * (1.0f - fx);
                    float w01 = (1.0f - fy) * fx;
                    float w10 = fy * (1.0f - fx);
                    float w11 = fy * fx;

                    // Get pixel values
                    int64_t base_idx = b * (channels * in_h * in_w) + c * (in_h * in_w);
                    float v00 = static_cast<float>(input[base_idx + y0 * in_w + x0]);
                    float v01 = static_cast<float>(input[base_idx + y0 * in_w + x1]);
                    float v10 = static_cast<float>(input[base_idx + y1 * in_w + x0]);
                    float v11 = static_cast<float>(input[base_idx + y1 * in_w + x1]);

                    // Interpolate
                    float result = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;

                    int64_t out_idx = b * (channels * out_h * out_w) +
                                     c * (out_h * out_w) +
                                     oh * out_w + ow;

                    output[out_idx] = static_cast<T>(result);
                }
            }
        }
    }
}

// Template for bicubic interpolation
template<typename T>
void interpolate_bicubic_impl(
    const T* input,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_h,
    int64_t out_w,
    bool align_corners
) {
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) if(batch * channels * out_h * out_w > 65536)
    #endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    // Calculate source position (floating point)
                    float y, x;
                    if (align_corners) {
                        y = (out_h > 1) ? oh * static_cast<float>(in_h - 1) / (out_h - 1) : 0.0f;
                        x = (out_w > 1) ? ow * static_cast<float>(in_w - 1) / (out_w - 1) : 0.0f;
                    } else {
                        float scale_h = static_cast<float>(in_h) / out_h;
                        float scale_w = static_cast<float>(in_w) / out_w;
                        y = (oh + 0.5f) * scale_h - 0.5f;
                        x = (ow + 0.5f) * scale_w - 0.5f;
                    }

                    // Clamp to valid range
                    y = std::clamp(y, 0.0f, static_cast<float>(in_h - 1));
                    x = std::clamp(x, 0.0f, static_cast<float>(in_w - 1));

                    int64_t y_int = static_cast<int64_t>(y);
                    int64_t x_int = static_cast<int64_t>(x);

                    // Bicubic interpolation using 4x4 neighborhood
                    float sum = 0.0f;
                    int64_t base_idx = b * (channels * in_h * in_w) + c * (in_h * in_w);

                    for (int64_t dy = -1; dy <= 2; ++dy) {
                        for (int64_t dx = -1; dx <= 2; ++dx) {
                            int64_t iy = y_int + dy;
                            int64_t ix = x_int + dx;

                            // Clamp indices
                            iy = std::clamp(iy, int64_t(0), in_h - 1);
                            ix = std::clamp(ix, int64_t(0), in_w - 1);

                            float weight_y = cubic_interp_coeff(y - (y_int + dy));
                            float weight_x = cubic_interp_coeff(x - (x_int + dx));
                            float weight = weight_y * weight_x;

                            sum += weight * static_cast<float>(input[base_idx + iy * in_w + ix]);
                        }
                    }

                    int64_t out_idx = b * (channels * out_h * out_w) +
                                     c * (out_h * out_w) +
                                     oh * out_w + ow;

                    output[out_idx] = static_cast<T>(sum);
                }
            }
        }
    }
}

} // anonymous namespace

// ============================================================================
// Interpolate Kernel (Public Interface)
// ============================================================================

auto interpolate_kernel(const Tensor& input,
                        const std::vector<int64_t>& size,
                        const std::string& mode,
                        bool align_corners) -> Tensor {
    auto shape = input.shape();
    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t in_h = shape[2];
    int64_t in_w = shape[3];
    int64_t out_h = size[0];
    int64_t out_w = size[1];

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, channels, out_h, out_w};
    Tensor output(output_shape, input.dtype(), input.device());

    if (mode == "nearest") {
        switch (input.dtype()) {
            case DType::Float32:
                interpolate_nearest_impl<float>(
                    input.data<float>(),
                    output.data<float>(),
                    batch, channels, in_h, in_w, out_h, out_w
                );
                break;
            case DType::Float64:
                interpolate_nearest_impl<double>(
                    input.data<double>(),
                    output.data<double>(),
                    batch, channels, in_h, in_w, out_h, out_w
                );
                break;
            case DType::Float16:
                interpolate_nearest_impl<Float16>(
                    input.data<Float16>(),
                    output.data<Float16>(),
                    batch, channels, in_h, in_w, out_h, out_w
                );
                break;
            case DType::BFloat16:
                interpolate_nearest_impl<BFloat16>(
                    input.data<BFloat16>(),
                    output.data<BFloat16>(),
                    batch, channels, in_h, in_w, out_h, out_w
                );
                break;
            default:
                throw std::runtime_error("interpolate_kernel: Unsupported dtype for nearest mode");
        }
    } else if (mode == "bilinear") {
        switch (input.dtype()) {
            case DType::Float32:
                interpolate_bilinear_impl<float>(
                    input.data<float>(),
                    output.data<float>(),
                    batch, channels, in_h, in_w, out_h, out_w,
                    align_corners
                );
                break;
            case DType::Float64:
                interpolate_bilinear_impl<double>(
                    input.data<double>(),
                    output.data<double>(),
                    batch, channels, in_h, in_w, out_h, out_w,
                    align_corners
                );
                break;
            case DType::Float16:
                interpolate_bilinear_impl<Float16>(
                    input.data<Float16>(),
                    output.data<Float16>(),
                    batch, channels, in_h, in_w, out_h, out_w,
                    align_corners
                );
                break;
            case DType::BFloat16:
                interpolate_bilinear_impl<BFloat16>(
                    input.data<BFloat16>(),
                    output.data<BFloat16>(),
                    batch, channels, in_h, in_w, out_h, out_w,
                    align_corners
                );
                break;
            default:
                throw std::runtime_error("interpolate_kernel: Unsupported dtype for bilinear mode");
        }
    } else if (mode == "bicubic") {
        switch (input.dtype()) {
            case DType::Float32:
                interpolate_bicubic_impl<float>(
                    input.data<float>(),
                    output.data<float>(),
                    batch, channels, in_h, in_w, out_h, out_w,
                    align_corners
                );
                break;
            case DType::Float64:
                interpolate_bicubic_impl<double>(
                    input.data<double>(),
                    output.data<double>(),
                    batch, channels, in_h, in_w, out_h, out_w,
                    align_corners
                );
                break;
            case DType::Float16:
                interpolate_bicubic_impl<Float16>(
                    input.data<Float16>(),
                    output.data<Float16>(),
                    batch, channels, in_h, in_w, out_h, out_w,
                    align_corners
                );
                break;
            case DType::BFloat16:
                interpolate_bicubic_impl<BFloat16>(
                    input.data<BFloat16>(),
                    output.data<BFloat16>(),
                    batch, channels, in_h, in_w, out_h, out_w,
                    align_corners
                );
                break;
            default:
                throw std::runtime_error("interpolate_kernel: Unsupported dtype for bicubic mode");
        }
    } else {
        throw std::runtime_error("interpolate_kernel: Unsupported mode: " + mode);
    }

    return output;
}

// =========================================================================
// ROI Align Operations
// =========================================================================

namespace {

template<typename T>
auto bilinear_interpolate(const T* data, int64_t height, int64_t width,
                          float y, float x) -> float {
    if (y < -1.0f || y > static_cast<float>(height) ||
        x < -1.0f || x > static_cast<float>(width)) {
        return 0.0f;
    }

    y = std::max(y, 0.0f);
    x = std::max(x, 0.0f);

    int64_t y_low = static_cast<int64_t>(y);
    int64_t x_low = static_cast<int64_t>(x);
    int64_t y_high = y_low + 1;
    int64_t x_high = x_low + 1;

    if (y_low >= height - 1) { y_low = y_high = height - 1; y = static_cast<float>(y_low); }
    if (x_low >= width - 1) { x_low = x_high = width - 1; x = static_cast<float>(x_low); }

    float ly = y - static_cast<float>(y_low);
    float lx = x - static_cast<float>(x_low);
    float hy = 1.0f - ly;
    float hx = 1.0f - lx;

    float v1 = static_cast<float>(data[y_low * width + x_low]);
    float v2 = static_cast<float>(data[y_low * width + x_high]);
    float v3 = static_cast<float>(data[y_high * width + x_low]);
    float v4 = static_cast<float>(data[y_high * width + x_high]);

    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

} // anonymous namespace

auto roi_align_forward_kernel(const Tensor& features, const Tensor& rois,
                               int64_t output_h, int64_t output_w,
                               float spatial_scale, int64_t sampling_ratio,
                               bool aligned) -> Tensor {
    // features: (N, C, H, W)
    // rois: (num_rois, 5) where each row is [batch_idx, x1, y1, x2, y2]
    const auto& feat_shape = features.shape();
    int64_t channels = feat_shape[1];
    int64_t height = feat_shape[2];
    int64_t width = feat_shape[3];
    int64_t num_rois = rois.shape()[0];

    Tensor output({num_rois, channels, output_h, output_w},
                  features.dtype(), features.device());

    if (features.dtype() == DType::Float32) {
        const float* feat_data = features.data<float>();
        const float* roi_data = rois.data<float>();
        float* out_data = output.data<float>();

        float offset = aligned ? 0.5f : 0.0f;

        #pragma omp parallel for if(num_rois > 16)
        for (int64_t n = 0; n < num_rois; ++n) {
            int64_t batch_idx = static_cast<int64_t>(roi_data[n * 5 + 0]);
            float roi_x1 = roi_data[n * 5 + 1] * spatial_scale - offset;
            float roi_y1 = roi_data[n * 5 + 2] * spatial_scale - offset;
            float roi_x2 = roi_data[n * 5 + 3] * spatial_scale - offset;
            float roi_y2 = roi_data[n * 5 + 4] * spatial_scale - offset;

            float roi_w = roi_x2 - roi_x1;
            float roi_h = roi_y2 - roi_y1;
            if (!aligned) {
                roi_w = std::max(roi_w, 1.0f);
                roi_h = std::max(roi_h, 1.0f);
            }

            float bin_h = roi_h / static_cast<float>(output_h);
            float bin_w = roi_w / static_cast<float>(output_w);

            int64_t roi_bin_h = sampling_ratio > 0 ? sampling_ratio
                : static_cast<int64_t>(std::ceil(bin_h));
            int64_t roi_bin_w = sampling_ratio > 0 ? sampling_ratio
                : static_cast<int64_t>(std::ceil(bin_w));
            roi_bin_h = std::max(roi_bin_h, int64_t(1));
            roi_bin_w = std::max(roi_bin_w, int64_t(1));

            float count = static_cast<float>(roi_bin_h * roi_bin_w);

            for (int64_t c = 0; c < channels; ++c) {
                const float* channel_data = feat_data + (batch_idx * channels + c) * height * width;

                for (int64_t ph = 0; ph < output_h; ++ph) {
                    for (int64_t pw = 0; pw < output_w; ++pw) {
                        float val = 0.0f;

                        for (int64_t iy = 0; iy < roi_bin_h; ++iy) {
                            float y = roi_y1 + bin_h * (static_cast<float>(ph) +
                                (static_cast<float>(iy) + 0.5f) / static_cast<float>(roi_bin_h));
                            for (int64_t ix = 0; ix < roi_bin_w; ++ix) {
                                float x = roi_x1 + bin_w * (static_cast<float>(pw) +
                                    (static_cast<float>(ix) + 0.5f) / static_cast<float>(roi_bin_w));
                                val += bilinear_interpolate(channel_data, height, width, y, x);
                            }
                        }

                        out_data[((n * channels + c) * output_h + ph) * output_w + pw] = val / count;
                    }
                }
            }
        }
    } else if (features.dtype() == DType::Float64) {
        const double* feat_data = features.data<double>();
        const double* roi_data = rois.data<double>();
        double* out_data = output.data<double>();

        double offset = aligned ? 0.5 : 0.0;

        #pragma omp parallel for if(num_rois > 16)
        for (int64_t n = 0; n < num_rois; ++n) {
            int64_t batch_idx = static_cast<int64_t>(roi_data[n * 5 + 0]);
            float roi_x1 = static_cast<float>(roi_data[n * 5 + 1] * spatial_scale - offset);
            float roi_y1 = static_cast<float>(roi_data[n * 5 + 2] * spatial_scale - offset);
            float roi_x2 = static_cast<float>(roi_data[n * 5 + 3] * spatial_scale - offset);
            float roi_y2 = static_cast<float>(roi_data[n * 5 + 4] * spatial_scale - offset);

            float roi_w = roi_x2 - roi_x1;
            float roi_h = roi_y2 - roi_y1;
            if (!aligned) {
                roi_w = std::max(roi_w, 1.0f);
                roi_h = std::max(roi_h, 1.0f);
            }

            float bin_h = roi_h / static_cast<float>(output_h);
            float bin_w = roi_w / static_cast<float>(output_w);

            int64_t roi_bin_h = sampling_ratio > 0 ? sampling_ratio
                : static_cast<int64_t>(std::ceil(bin_h));
            int64_t roi_bin_w = sampling_ratio > 0 ? sampling_ratio
                : static_cast<int64_t>(std::ceil(bin_w));
            roi_bin_h = std::max(roi_bin_h, int64_t(1));
            roi_bin_w = std::max(roi_bin_w, int64_t(1));

            float count = static_cast<float>(roi_bin_h * roi_bin_w);

            for (int64_t c = 0; c < channels; ++c) {
                const double* channel_data = feat_data + (batch_idx * channels + c) * height * width;

                for (int64_t ph = 0; ph < output_h; ++ph) {
                    for (int64_t pw = 0; pw < output_w; ++pw) {
                        float val = 0.0f;

                        for (int64_t iy = 0; iy < roi_bin_h; ++iy) {
                            float y = roi_y1 + bin_h * (static_cast<float>(ph) +
                                (static_cast<float>(iy) + 0.5f) / static_cast<float>(roi_bin_h));
                            for (int64_t ix = 0; ix < roi_bin_w; ++ix) {
                                float x = roi_x1 + bin_w * (static_cast<float>(pw) +
                                    (static_cast<float>(ix) + 0.5f) / static_cast<float>(roi_bin_w));
                                val += bilinear_interpolate(channel_data, height, width, y, x);
                            }
                        }

                        out_data[((n * channels + c) * output_h + ph) * output_w + pw] = static_cast<double>(val / count);
                    }
                }
            }
        }
    } else if (features.dtype() == DType::Float16 || features.dtype() == DType::BFloat16) {
        // Convert to Float32, compute, convert back
        auto features_f32 = features.to(DType::Float32);
        auto rois_f32 = rois.to(DType::Float32);
        auto output_f32 = roi_align_forward_kernel(features_f32, rois_f32, output_h, output_w,
                                                     spatial_scale, sampling_ratio, aligned);
        return output_f32.to(features.dtype());
    } else {
        throw std::runtime_error("roi_align_forward: unsupported dtype");
    }

    return output;
}

auto roi_align_backward_kernel(const Tensor& grad_output, const Tensor& rois,
                                int64_t batch_size, int64_t feat_height, int64_t feat_width,
                                float spatial_scale, int64_t sampling_ratio,
                                bool aligned) -> Tensor {
    const auto& grad_shape = grad_output.shape();
    int64_t num_rois = grad_shape[0];
    int64_t channels = grad_shape[1];
    int64_t output_h = grad_shape[2];
    int64_t output_w = grad_shape[3];

    Tensor grad_input({batch_size, channels, feat_height, feat_width},
                      grad_output.dtype(), grad_output.device());
    std::memset(grad_input.data<uint8_t>(), 0,
                grad_input.numel() * dtype_size(grad_input.dtype()));

    if (grad_output.dtype() == DType::Float32) {
        float* gi_data = grad_input.data<float>();
        const float* go_data = grad_output.data<float>();
        const float* roi_data = rois.data<float>();

        float offset = aligned ? 0.5f : 0.0f;

        // Parallelize over ROIs; use atomic adds for overlapping spatial regions
        #pragma omp parallel for if(num_rois > 16)
        for (int64_t n = 0; n < num_rois; ++n) {
            int64_t batch_idx = static_cast<int64_t>(roi_data[n * 5 + 0]);
            float roi_x1 = roi_data[n * 5 + 1] * spatial_scale - offset;
            float roi_y1 = roi_data[n * 5 + 2] * spatial_scale - offset;
            float roi_x2 = roi_data[n * 5 + 3] * spatial_scale - offset;
            float roi_y2 = roi_data[n * 5 + 4] * spatial_scale - offset;

            float roi_w = roi_x2 - roi_x1;
            float roi_h = roi_y2 - roi_y1;
            if (!aligned) {
                roi_w = std::max(roi_w, 1.0f);
                roi_h = std::max(roi_h, 1.0f);
            }

            float bin_h = roi_h / static_cast<float>(output_h);
            float bin_w = roi_w / static_cast<float>(output_w);

            int64_t roi_bin_h = sampling_ratio > 0 ? sampling_ratio
                : static_cast<int64_t>(std::ceil(bin_h));
            int64_t roi_bin_w = sampling_ratio > 0 ? sampling_ratio
                : static_cast<int64_t>(std::ceil(bin_w));
            roi_bin_h = std::max(roi_bin_h, int64_t(1));
            roi_bin_w = std::max(roi_bin_w, int64_t(1));

            float count = static_cast<float>(roi_bin_h * roi_bin_w);

            for (int64_t c = 0; c < channels; ++c) {
                float* gi_channel = gi_data + (batch_idx * channels + c) * feat_height * feat_width;

                for (int64_t ph = 0; ph < output_h; ++ph) {
                    for (int64_t pw = 0; pw < output_w; ++pw) {
                        float grad_val = go_data[((n * channels + c) * output_h + ph) * output_w + pw] / count;

                        for (int64_t iy = 0; iy < roi_bin_h; ++iy) {
                            float y = roi_y1 + bin_h * (static_cast<float>(ph) +
                                (static_cast<float>(iy) + 0.5f) / static_cast<float>(roi_bin_h));
                            for (int64_t ix = 0; ix < roi_bin_w; ++ix) {
                                float x = roi_x1 + bin_w * (static_cast<float>(pw) +
                                    (static_cast<float>(ix) + 0.5f) / static_cast<float>(roi_bin_w));

                                if (y < -1.0f || y > static_cast<float>(feat_height) ||
                                    x < -1.0f || x > static_cast<float>(feat_width)) continue;

                                y = std::max(y, 0.0f);
                                x = std::max(x, 0.0f);

                                int64_t y_low = static_cast<int64_t>(y);
                                int64_t x_low = static_cast<int64_t>(x);
                                int64_t y_high = y_low + 1;
                                int64_t x_high = x_low + 1;

                                if (y_low >= feat_height - 1) { y_low = y_high = feat_height - 1; }
                                if (x_low >= feat_width - 1) { x_low = x_high = feat_width - 1; }

                                float ly = y - static_cast<float>(y_low);
                                float lx = x - static_cast<float>(x_low);

                                #pragma omp atomic
                                gi_channel[y_low * feat_width + x_low] += grad_val * (1.0f - ly) * (1.0f - lx);
                                #pragma omp atomic
                                gi_channel[y_low * feat_width + x_high] += grad_val * (1.0f - ly) * lx;
                                #pragma omp atomic
                                gi_channel[y_high * feat_width + x_low] += grad_val * ly * (1.0f - lx);
                                #pragma omp atomic
                                gi_channel[y_high * feat_width + x_high] += grad_val * ly * lx;
                            }
                        }
                    }
                }
            }
        }
    } else if (grad_output.dtype() == DType::Float64) {
        double* gi_data = grad_input.data<double>();
        const double* go_data = grad_output.data<double>();
        const double* roi_data = rois.data<double>();

        double offset = aligned ? 0.5 : 0.0;

        #pragma omp parallel for if(num_rois > 16)
        for (int64_t n = 0; n < num_rois; ++n) {
            int64_t batch_idx = static_cast<int64_t>(roi_data[n * 5 + 0]);
            float roi_x1 = static_cast<float>(roi_data[n * 5 + 1] * spatial_scale - offset);
            float roi_y1 = static_cast<float>(roi_data[n * 5 + 2] * spatial_scale - offset);
            float roi_x2 = static_cast<float>(roi_data[n * 5 + 3] * spatial_scale - offset);
            float roi_y2 = static_cast<float>(roi_data[n * 5 + 4] * spatial_scale - offset);

            float roi_w = roi_x2 - roi_x1;
            float roi_h = roi_y2 - roi_y1;
            if (!aligned) {
                roi_w = std::max(roi_w, 1.0f);
                roi_h = std::max(roi_h, 1.0f);
            }

            float bin_h = roi_h / static_cast<float>(output_h);
            float bin_w = roi_w / static_cast<float>(output_w);

            int64_t roi_bin_h = sampling_ratio > 0 ? sampling_ratio
                : static_cast<int64_t>(std::ceil(bin_h));
            int64_t roi_bin_w = sampling_ratio > 0 ? sampling_ratio
                : static_cast<int64_t>(std::ceil(bin_w));
            roi_bin_h = std::max(roi_bin_h, int64_t(1));
            roi_bin_w = std::max(roi_bin_w, int64_t(1));

            float count = static_cast<float>(roi_bin_h * roi_bin_w);

            for (int64_t c = 0; c < channels; ++c) {
                double* gi_channel = gi_data + (batch_idx * channels + c) * feat_height * feat_width;

                for (int64_t ph = 0; ph < output_h; ++ph) {
                    for (int64_t pw = 0; pw < output_w; ++pw) {
                        double grad_val = go_data[((n * channels + c) * output_h + ph) * output_w + pw] / count;

                        for (int64_t iy = 0; iy < roi_bin_h; ++iy) {
                            float y = roi_y1 + bin_h * (static_cast<float>(ph) +
                                (static_cast<float>(iy) + 0.5f) / static_cast<float>(roi_bin_h));
                            for (int64_t ix = 0; ix < roi_bin_w; ++ix) {
                                float x = roi_x1 + bin_w * (static_cast<float>(pw) +
                                    (static_cast<float>(ix) + 0.5f) / static_cast<float>(roi_bin_w));

                                if (y < -1.0f || y > static_cast<float>(feat_height) ||
                                    x < -1.0f || x > static_cast<float>(feat_width)) continue;

                                y = std::max(y, 0.0f);
                                x = std::max(x, 0.0f);

                                int64_t y_low = static_cast<int64_t>(y);
                                int64_t x_low = static_cast<int64_t>(x);
                                int64_t y_high = y_low + 1;
                                int64_t x_high = x_low + 1;

                                if (y_low >= feat_height - 1) { y_low = y_high = feat_height - 1; }
                                if (x_low >= feat_width - 1) { x_low = x_high = feat_width - 1; }

                                float ly = y - static_cast<float>(y_low);
                                float lx = x - static_cast<float>(x_low);

                                #pragma omp atomic
                                gi_channel[y_low * feat_width + x_low] += grad_val * static_cast<double>((1.0f - ly) * (1.0f - lx));
                                #pragma omp atomic
                                gi_channel[y_low * feat_width + x_high] += grad_val * static_cast<double>((1.0f - ly) * lx);
                                #pragma omp atomic
                                gi_channel[y_high * feat_width + x_low] += grad_val * static_cast<double>(ly * (1.0f - lx));
                                #pragma omp atomic
                                gi_channel[y_high * feat_width + x_high] += grad_val * static_cast<double>(ly * lx);
                            }
                        }
                    }
                }
            }
        }
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        // Convert to Float32, compute, convert back
        auto go_f32 = grad_output.to(DType::Float32);
        auto rois_f32 = rois.to(DType::Float32);
        auto result = roi_align_backward_kernel(go_f32, rois_f32, batch_size, feat_height, feat_width,
                                                  spatial_scale, sampling_ratio, aligned);
        return result.to(grad_output.dtype());
    }

    return grad_input;
}

// =========================================================================
// Box IoU
// =========================================================================

auto box_iou_kernel(const Tensor& boxes1, const Tensor& boxes2, int iou_type) -> Tensor {
    // boxes1: (N, 4), boxes2: (M, 4) in x1,y1,x2,y2 format
    // Returns (N, M) IoU matrix
    const int64_t N = boxes1.shape()[0];
    const int64_t M = boxes2.shape()[0];

    // Convert half types to Float32 for precision
    Tensor b1_f32 = boxes1;
    Tensor b2_f32 = boxes2;
    if (boxes1.dtype() == DType::Float16 || boxes1.dtype() == DType::BFloat16) {
        // Convert to Float32 by copying element-wise
        b1_f32 = Tensor({N, 4}, DType::Float32, boxes1.device());
        b2_f32 = Tensor({M, 4}, DType::Float32, boxes2.device());
        if (boxes1.dtype() == DType::Float16) {
            const Float16* src1 = boxes1.data<Float16>();
            float* dst1 = b1_f32.data<float>();
            for (int64_t i = 0; i < N * 4; ++i) dst1[i] = static_cast<float>(src1[i]);
            const Float16* src2 = boxes2.data<Float16>();
            float* dst2 = b2_f32.data<float>();
            for (int64_t i = 0; i < M * 4; ++i) dst2[i] = static_cast<float>(src2[i]);
        } else {
            const BFloat16* src1 = boxes1.data<BFloat16>();
            float* dst1 = b1_f32.data<float>();
            for (int64_t i = 0; i < N * 4; ++i) dst1[i] = static_cast<float>(src1[i]);
            const BFloat16* src2 = boxes2.data<BFloat16>();
            float* dst2 = b2_f32.data<float>();
            for (int64_t i = 0; i < M * 4; ++i) dst2[i] = static_cast<float>(src2[i]);
        }
    } else if (boxes1.dtype() == DType::Float64) {
        // Convert Float64 to Float32 for IoU computation
        b1_f32 = Tensor({N, 4}, DType::Float32, boxes1.device());
        b2_f32 = Tensor({M, 4}, DType::Float32, boxes2.device());
        const double* src1 = boxes1.data<double>();
        float* dst1 = b1_f32.data<float>();
        for (int64_t i = 0; i < N * 4; ++i) dst1[i] = static_cast<float>(src1[i]);
        const double* src2 = boxes2.data<double>();
        float* dst2 = b2_f32.data<float>();
        for (int64_t i = 0; i < M * 4; ++i) dst2[i] = static_cast<float>(src2[i]);
    }

    Tensor output({N, M}, DType::Float32, boxes1.device());
    const float* b1 = b1_f32.data<float>();
    const float* b2 = b2_f32.data<float>();
    float* out = output.data<float>();

    #pragma omp parallel for collapse(2) if(N * M > 4096)
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            float x1 = std::max(b1[i * 4 + 0], b2[j * 4 + 0]);
            float y1 = std::max(b1[i * 4 + 1], b2[j * 4 + 1]);
            float x2 = std::min(b1[i * 4 + 2], b2[j * 4 + 2]);
            float y2 = std::min(b1[i * 4 + 3], b2[j * 4 + 3]);

            float inter_w = std::max(0.0f, x2 - x1);
            float inter_h = std::max(0.0f, y2 - y1);
            float inter_area = inter_w * inter_h;

            float area1 = (b1[i * 4 + 2] - b1[i * 4 + 0]) * (b1[i * 4 + 3] - b1[i * 4 + 1]);
            float area2 = (b2[j * 4 + 2] - b2[j * 4 + 0]) * (b2[j * 4 + 3] - b2[j * 4 + 1]);
            float union_area = area1 + area2 - inter_area;

            float iou = (union_area > 0.0f) ? inter_area / union_area : 0.0f;

            if (iou_type == 1) {
                // GIoU
                float enclose_x1 = std::min(b1[i * 4 + 0], b2[j * 4 + 0]);
                float enclose_y1 = std::min(b1[i * 4 + 1], b2[j * 4 + 1]);
                float enclose_x2 = std::max(b1[i * 4 + 2], b2[j * 4 + 2]);
                float enclose_y2 = std::max(b1[i * 4 + 3], b2[j * 4 + 3]);
                float enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1);
                iou = iou - (enclose_area - union_area) / std::max(enclose_area, 1e-7f);
            }

            out[i * M + j] = iou;
        }
    }

    return output;
}

// =========================================================================
// Unfold / Fold (im2col / col2im style)
// =========================================================================

auto unfold_kernel(const Tensor& input, int64_t kernel_size,
                   int64_t stride, int64_t padding, int64_t dilation) -> Tensor {
    // input: (N, C, H, W)
    // output: (N, C * kernel_size * kernel_size, L) where L = output spatial locations
    const auto& shape = input.shape();
    int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];

    int64_t H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t L = H_out * W_out;
    int64_t cols_per_channel = kernel_size * kernel_size;

    Tensor output({N, C * cols_per_channel, L}, input.dtype(), input.device());

    auto unfold_impl = [&]<typename T>(const T* in_data, T* out_data) {
        #pragma omp parallel for if(N > 4)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                const T* channel = in_data + (n * C + c) * H * W;
                for (int64_t kh = 0; kh < kernel_size; ++kh) {
                    for (int64_t kw = 0; kw < kernel_size; ++kw) {
                        int64_t col_idx = (c * cols_per_channel + kh * kernel_size + kw);
                        T* col = out_data + (n * C * cols_per_channel + col_idx) * L;

                        for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                            for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                                int64_t h_in = h_out * stride - padding + kh * dilation;
                                int64_t w_in = w_out * stride - padding + kw * dilation;
                                int64_t l_idx = h_out * W_out + w_out;

                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                    col[l_idx] = channel[h_in * W + w_in];
                                } else {
                                    col[l_idx] = static_cast<T>(0);
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    if (input.dtype() == DType::Float32) {
        unfold_impl(input.data<float>(), output.data<float>());
    } else if (input.dtype() == DType::Float64) {
        unfold_impl(input.data<double>(), output.data<double>());
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto output_f32 = Tensor({N, C * cols_per_channel, L}, DType::Float32, input.device());
        unfold_impl(input_f32.data<float>(), output_f32.data<float>());
        return output_f32.to(input.dtype());
    } else {
        throw std::runtime_error("unfold: unsupported dtype");
    }

    return output;
}

auto fold_kernel(const Tensor& input, const std::vector<int64_t>& output_size,
                 int64_t kernel_size, int64_t stride, int64_t padding,
                 int64_t dilation) -> Tensor {
    // input: (N, C * kernel_size * kernel_size, L)
    // output: (N, C, output_size[0], output_size[1])
    const auto& shape = input.shape();
    int64_t N = shape[0];
    int64_t H_out = output_size[0], W_out = output_size[1];
    int64_t cols_per_channel = kernel_size * kernel_size;
    int64_t C = shape[1] / cols_per_channel;

    int64_t H_col = (H_out + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t W_col = (W_out + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());
    std::memset(output.data<uint8_t>(), 0, output.numel() * dtype_size(output.dtype()));

    auto fold_impl = [&]<typename T>(const T* in_data, T* out_data) {
        #pragma omp parallel for if(N > 4)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                T* channel = out_data + (n * C + c) * H_out * W_out;
                for (int64_t kh = 0; kh < kernel_size; ++kh) {
                    for (int64_t kw = 0; kw < kernel_size; ++kw) {
                        int64_t col_idx = c * cols_per_channel + kh * kernel_size + kw;
                        const T* col = in_data + (n * C * cols_per_channel + col_idx) * (H_col * W_col);

                        for (int64_t h_col = 0; h_col < H_col; ++h_col) {
                            for (int64_t w_col = 0; w_col < W_col; ++w_col) {
                                int64_t h_in = h_col * stride - padding + kh * dilation;
                                int64_t w_in = w_col * stride - padding + kw * dilation;

                                if (h_in >= 0 && h_in < H_out && w_in >= 0 && w_in < W_out) {
                                    channel[h_in * W_out + w_in] += col[h_col * W_col + w_col];
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    if (input.dtype() == DType::Float32) {
        fold_impl(input.data<float>(), output.data<float>());
    } else if (input.dtype() == DType::Float64) {
        fold_impl(input.data<double>(), output.data<double>());
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto output_f32 = Tensor({N, C, H_out, W_out}, DType::Float32, input.device());
        std::memset(output_f32.data<uint8_t>(), 0, output_f32.numel() * dtype_size(output_f32.dtype()));
        fold_impl(input_f32.data<float>(), output_f32.data<float>());
        return output_f32.to(input.dtype());
    } else {
        throw std::runtime_error("fold: unsupported dtype");
    }

    return output;
}

} // namespace cpu
} // namespace tenzor
