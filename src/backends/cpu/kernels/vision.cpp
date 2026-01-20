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

} // namespace cpu
} // namespace tenzor
