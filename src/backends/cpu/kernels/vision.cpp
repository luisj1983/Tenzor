/**
 * @file vision.cpp
 * @brief CPU kernel implementations for vision operations
 *
 * Includes interpolation (resize) operations with nearest, bilinear,
 * and bicubic modes.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/dtype_dispatch.hpp"
#include "tenzor/ops/creation.hpp"   // zeros()
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#ifdef _OPENMP
#include <omp.h>
#include "tenzor/backend/omp_thresholds.hpp"
#endif

namespace tenzor {
namespace cpu {

// ============================================================================
// Interpolation Helper Functions
// ============================================================================

namespace {

// Accumulator-precision selector for interpolation kernels: Float64 inputs keep
// `double` throughout (coordinates, weights, accumulators); every other input
// dtype (Float32, Float16, BFloat16) widens to `float` for the arithmetic and
// narrows back at store. Mirrors the pattern from
// adaptive_avgpool1d_impl in pooling.cpp.
template<typename T>
using interp_acc_t = std::conditional_t<std::is_same_v<T, double>, double, float>;

// Cubic interpolation coefficient function (Catmull-Rom spline).
// Templated on the compute type so Float64 paths keep double precision.
template<typename C>
inline C cubic_interp_coeff(C x) {
    C abs_x = std::abs(x);
    if (abs_x <= C(1)) {
        return C(1.5) * abs_x * abs_x * abs_x - C(2.5) * abs_x * abs_x + C(1);
    } else if (abs_x < C(2)) {
        return C(-0.5) * abs_x * abs_x * abs_x + C(2.5) * abs_x * abs_x - C(4) * abs_x + C(2);
    }
    return C(0);
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
    using Compute = interp_acc_t<T>;
    const Compute scale_h = static_cast<Compute>(in_h) / static_cast<Compute>(out_h);
    const Compute scale_w = static_cast<Compute>(in_w) / static_cast<Compute>(out_w);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) if(batch * channels * out_h * out_w > ::tenzor::OmpThresholds::simple())
    #endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    // Calculate source position
                    int64_t ih = static_cast<int64_t>(static_cast<Compute>(oh) * scale_h);
                    int64_t iw = static_cast<int64_t>(static_cast<Compute>(ow) * scale_w);

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
    using Compute = interp_acc_t<T>;
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) if(batch * channels * out_h * out_w > ::tenzor::OmpThresholds::simple())
    #endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    // Calculate source position (Compute-precision)
                    Compute y, x;
                    if (align_corners) {
                        // Align corners: map [0, out-1] to [0, in-1]
                        y = (out_h > 1)
                            ? static_cast<Compute>(oh) * static_cast<Compute>(in_h - 1) /
                              static_cast<Compute>(out_h - 1)
                            : Compute(0);
                        x = (out_w > 1)
                            ? static_cast<Compute>(ow) * static_cast<Compute>(in_w - 1) /
                              static_cast<Compute>(out_w - 1)
                            : Compute(0);
                    } else {
                        // Half-pixel centers: pixels are unit squares
                        Compute scale_h = static_cast<Compute>(in_h) / static_cast<Compute>(out_h);
                        Compute scale_w = static_cast<Compute>(in_w) / static_cast<Compute>(out_w);
                        y = (static_cast<Compute>(oh) + Compute(0.5)) * scale_h - Compute(0.5);
                        x = (static_cast<Compute>(ow) + Compute(0.5)) * scale_w - Compute(0.5);
                    }

                    // Clamp to valid range
                    y = std::clamp(y, Compute(0), static_cast<Compute>(in_h - 1));
                    x = std::clamp(x, Compute(0), static_cast<Compute>(in_w - 1));

                    // Get integer and fractional parts
                    int64_t y0 = static_cast<int64_t>(y);
                    int64_t x0 = static_cast<int64_t>(x);
                    int64_t y1 = std::min(y0 + 1, in_h - 1);
                    int64_t x1 = std::min(x0 + 1, in_w - 1);

                    Compute fy = y - static_cast<Compute>(y0);
                    Compute fx = x - static_cast<Compute>(x0);

                    // Bilinear interpolation weights
                    Compute w00 = (Compute(1) - fy) * (Compute(1) - fx);
                    Compute w01 = (Compute(1) - fy) * fx;
                    Compute w10 = fy * (Compute(1) - fx);
                    Compute w11 = fy * fx;

                    // Get pixel values
                    int64_t base_idx = b * (channels * in_h * in_w) + c * (in_h * in_w);
                    Compute v00 = static_cast<Compute>(input[base_idx + y0 * in_w + x0]);
                    Compute v01 = static_cast<Compute>(input[base_idx + y0 * in_w + x1]);
                    Compute v10 = static_cast<Compute>(input[base_idx + y1 * in_w + x0]);
                    Compute v11 = static_cast<Compute>(input[base_idx + y1 * in_w + x1]);

                    // Interpolate
                    Compute result = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;

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
    using Compute = interp_acc_t<T>;
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) if(batch * channels * out_h * out_w > ::tenzor::OmpThresholds::simple())
    #endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    // Calculate source position (Compute-precision)
                    Compute y, x;
                    if (align_corners) {
                        y = (out_h > 1)
                            ? static_cast<Compute>(oh) * static_cast<Compute>(in_h - 1) /
                              static_cast<Compute>(out_h - 1)
                            : Compute(0);
                        x = (out_w > 1)
                            ? static_cast<Compute>(ow) * static_cast<Compute>(in_w - 1) /
                              static_cast<Compute>(out_w - 1)
                            : Compute(0);
                    } else {
                        Compute scale_h = static_cast<Compute>(in_h) / static_cast<Compute>(out_h);
                        Compute scale_w = static_cast<Compute>(in_w) / static_cast<Compute>(out_w);
                        y = (static_cast<Compute>(oh) + Compute(0.5)) * scale_h - Compute(0.5);
                        x = (static_cast<Compute>(ow) + Compute(0.5)) * scale_w - Compute(0.5);
                    }

                    // Clamp to valid range
                    y = std::clamp(y, Compute(0), static_cast<Compute>(in_h - 1));
                    x = std::clamp(x, Compute(0), static_cast<Compute>(in_w - 1));

                    int64_t y_int = static_cast<int64_t>(y);
                    int64_t x_int = static_cast<int64_t>(x);

                    // Bicubic interpolation using 4x4 neighborhood
                    Compute sum = Compute(0);
                    int64_t base_idx = b * (channels * in_h * in_w) + c * (in_h * in_w);

                    for (int64_t dy = -1; dy <= 2; ++dy) {
                        for (int64_t dx = -1; dx <= 2; ++dx) {
                            int64_t iy = y_int + dy;
                            int64_t ix = x_int + dx;

                            // Clamp indices
                            iy = std::clamp(iy, int64_t(0), in_h - 1);
                            ix = std::clamp(ix, int64_t(0), in_w - 1);

                            Compute weight_y = cubic_interp_coeff<Compute>(
                                y - static_cast<Compute>(y_int + dy));
                            Compute weight_x = cubic_interp_coeff<Compute>(
                                x - static_cast<Compute>(x_int + dx));
                            Compute weight = weight_y * weight_x;

                            sum += weight * static_cast<Compute>(input[base_idx + iy * in_w + ix]);
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

// Template for trilinear interpolation (5D: N, C, D, H, W)
template<typename T>
void interpolate_trilinear_impl(
    const T* input,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t in_d,
    int64_t in_h,
    int64_t in_w,
    int64_t out_d,
    int64_t out_h,
    int64_t out_w,
    bool align_corners
) {
    using Compute = interp_acc_t<T>;
    int64_t total = batch * channels * out_d * out_h * out_w;

#ifdef _OPENMP
    #pragma omp parallel for if(total > ::tenzor::OmpThresholds::simple())
#endif
    for (int64_t idx = 0; idx < total; ++idx) {
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t od = temp % out_d; temp /= out_d;
        int64_t c  = temp % channels; temp /= channels;
        int64_t b  = temp;

        Compute z, y, x;
        if (align_corners) {
            z = (out_d > 1)
                ? static_cast<Compute>(od) * static_cast<Compute>(in_d - 1) /
                  static_cast<Compute>(out_d - 1)
                : Compute(0);
            y = (out_h > 1)
                ? static_cast<Compute>(oh) * static_cast<Compute>(in_h - 1) /
                  static_cast<Compute>(out_h - 1)
                : Compute(0);
            x = (out_w > 1)
                ? static_cast<Compute>(ow) * static_cast<Compute>(in_w - 1) /
                  static_cast<Compute>(out_w - 1)
                : Compute(0);
        } else {
            Compute scale_d = static_cast<Compute>(in_d) / static_cast<Compute>(out_d);
            Compute scale_h = static_cast<Compute>(in_h) / static_cast<Compute>(out_h);
            Compute scale_w = static_cast<Compute>(in_w) / static_cast<Compute>(out_w);
            z = (static_cast<Compute>(od) + Compute(0.5)) * scale_d - Compute(0.5);
            y = (static_cast<Compute>(oh) + Compute(0.5)) * scale_h - Compute(0.5);
            x = (static_cast<Compute>(ow) + Compute(0.5)) * scale_w - Compute(0.5);
        }

        z = std::max(Compute(0), std::min(z, static_cast<Compute>(in_d - 1)));
        y = std::max(Compute(0), std::min(y, static_cast<Compute>(in_h - 1)));
        x = std::max(Compute(0), std::min(x, static_cast<Compute>(in_w - 1)));

        int64_t z0 = static_cast<int64_t>(z);
        int64_t y0 = static_cast<int64_t>(y);
        int64_t x0 = static_cast<int64_t>(x);
        int64_t z1 = std::min(z0 + 1, in_d - 1);
        int64_t y1 = std::min(y0 + 1, in_h - 1);
        int64_t x1 = std::min(x0 + 1, in_w - 1);

        Compute fz = z - static_cast<Compute>(z0);
        Compute fy = y - static_cast<Compute>(y0);
        Compute fx = x - static_cast<Compute>(x0);

        // Base offset for this (b, c) slice
        int64_t base = (b * channels + c) * in_d * in_h * in_w;

        // 8-point trilinear interpolation
        Compute v000 = static_cast<Compute>(input[base + z0 * in_h * in_w + y0 * in_w + x0]);
        Compute v001 = static_cast<Compute>(input[base + z0 * in_h * in_w + y0 * in_w + x1]);
        Compute v010 = static_cast<Compute>(input[base + z0 * in_h * in_w + y1 * in_w + x0]);
        Compute v011 = static_cast<Compute>(input[base + z0 * in_h * in_w + y1 * in_w + x1]);
        Compute v100 = static_cast<Compute>(input[base + z1 * in_h * in_w + y0 * in_w + x0]);
        Compute v101 = static_cast<Compute>(input[base + z1 * in_h * in_w + y0 * in_w + x1]);
        Compute v110 = static_cast<Compute>(input[base + z1 * in_h * in_w + y1 * in_w + x0]);
        Compute v111 = static_cast<Compute>(input[base + z1 * in_h * in_w + y1 * in_w + x1]);

        Compute result =
            v000 * (Compute(1) - fz) * (Compute(1) - fy) * (Compute(1) - fx) +
            v001 * (Compute(1) - fz) * (Compute(1) - fy) * fx +
            v010 * (Compute(1) - fz) * fy * (Compute(1) - fx) +
            v011 * (Compute(1) - fz) * fy * fx +
            v100 * fz * (Compute(1) - fy) * (Compute(1) - fx) +
            v101 * fz * (Compute(1) - fy) * fx +
            v110 * fz * fy * (Compute(1) - fx) +
            v111 * fz * fy * fx;

        output[idx] = static_cast<T>(result);
    }
}

// Template for 5D nearest neighbor interpolation
template<typename T>
void interpolate_nearest_5d_impl(
    const T* input,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t in_d,
    int64_t in_h,
    int64_t in_w,
    int64_t out_d,
    int64_t out_h,
    int64_t out_w
) {
    using Compute = interp_acc_t<T>;
    Compute scale_d = static_cast<Compute>(in_d) / static_cast<Compute>(out_d);
    Compute scale_h = static_cast<Compute>(in_h) / static_cast<Compute>(out_h);
    Compute scale_w = static_cast<Compute>(in_w) / static_cast<Compute>(out_w);
    int64_t total = batch * channels * out_d * out_h * out_w;

#ifdef _OPENMP
    #pragma omp parallel for if(total > ::tenzor::OmpThresholds::simple())
#endif
    for (int64_t idx = 0; idx < total; ++idx) {
        int64_t temp = idx;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t od = temp % out_d; temp /= out_d;
        int64_t c  = temp % channels; temp /= channels;
        int64_t b  = temp;

        int64_t id = std::min(static_cast<int64_t>(static_cast<Compute>(od) * scale_d), in_d - 1);
        int64_t ih = std::min(static_cast<int64_t>(static_cast<Compute>(oh) * scale_h), in_h - 1);
        int64_t iw = std::min(static_cast<int64_t>(static_cast<Compute>(ow) * scale_w), in_w - 1);

        int64_t in_idx = ((b * channels + c) * in_d + id) * in_h * in_w + ih * in_w + iw;
        output[idx] = input[in_idx];
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

    // Handle 5D input (trilinear / nearest-5d)
    if (shape.size() == 5) {
        int64_t batch = shape[0], channels = shape[1];
        int64_t in_d = shape[2], in_h = shape[3], in_w = shape[4];
        int64_t out_d = size[0], out_h = size[1], out_w = size[2];
        Tensor output({batch, channels, out_d, out_h, out_w}, input.dtype(), input.device());

        auto dispatch_5d = [&](auto* dummy) {
            using T = std::remove_pointer_t<decltype(dummy)>;
            if (mode == "trilinear") {
                interpolate_trilinear_impl<T>(
                    input.data<T>(), output.data<T>(),
                    batch, channels, in_d, in_h, in_w, out_d, out_h, out_w, align_corners);
            } else {
                interpolate_nearest_5d_impl<T>(
                    input.data<T>(), output.data<T>(),
                    batch, channels, in_d, in_h, in_w, out_d, out_h, out_w);
            }
        };

        switch (input.dtype()) {
            case DType::Float32:  dispatch_5d(static_cast<float*>(nullptr)); break;
            case DType::Float64:  dispatch_5d(static_cast<double*>(nullptr)); break;
            case DType::Float16:  dispatch_5d(static_cast<Float16*>(nullptr)); break;
            case DType::BFloat16: dispatch_5d(static_cast<BFloat16*>(nullptr)); break;
            default:
                throw std::runtime_error("interpolate_kernel: Unsupported dtype for 5D interpolation");
        }
        return output;
    }

    // 4D path (existing)
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

// ============================================================================
// Interpolate Backward Kernel (audit D3): device-resident bilinear scatter.
// ============================================================================
//
// Computes grad_input from grad_output by distributing each output-pixel
// gradient over the four nearest input pixels weighted by fractional source
// coordinates. align_corners=false convention. Operates on CPU buffers
// directly — no `.to(cpu)` round-trip needed when caller is already on CPU.

template<typename T>
static void interpolate_bilinear_backward_impl(
    const T* grad_out, T* grad_in,
    int64_t N, int64_t C,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w,
    bool align_corners)
{
    using Compute = interp_acc_t<T>;
    // align_corners=false: scale = in_h / out_h; src = (h + 0.5) * scale - 0.5.
    // align_corners=true:  scale = (in_h - 1) / (out_h - 1); src = h * scale.
    const Compute scale_h = align_corners && out_h > 1
        ? static_cast<Compute>(in_h - 1) / static_cast<Compute>(out_h - 1)
        : static_cast<Compute>(in_h) / static_cast<Compute>(out_h);
    const Compute scale_w = align_corners && out_w > 1
        ? static_cast<Compute>(in_w - 1) / static_cast<Compute>(out_w - 1)
        : static_cast<Compute>(in_w) / static_cast<Compute>(out_w);

    // grad_in is zero-initialized by the caller.
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const T* go = grad_out + ((n * C + c) * out_h * out_w);
            T* gi = grad_in + ((n * C + c) * in_h * in_w);
            for (int64_t h = 0; h < out_h; ++h) {
                const Compute src_h = align_corners
                    ? static_cast<Compute>(h) * scale_h
                    : (static_cast<Compute>(h) + Compute(0.5)) * scale_h - Compute(0.5);
                const int64_t h0 = static_cast<int64_t>(std::floor(src_h));
                const int64_t h1 = h0 + 1;
                const Compute fh = src_h - static_cast<Compute>(h0);
                for (int64_t w = 0; w < out_w; ++w) {
                    const Compute src_w = align_corners
                        ? static_cast<Compute>(w) * scale_w
                        : (static_cast<Compute>(w) + Compute(0.5)) * scale_w - Compute(0.5);
                    const int64_t w0 = static_cast<int64_t>(std::floor(src_w));
                    const int64_t w1 = w0 + 1;
                    const Compute fw = src_w - static_cast<Compute>(w0);
                    const Compute g_val = static_cast<Compute>(go[h * out_w + w]);

                    auto add = [&](int64_t hi, int64_t wi, Compute weight) {
                        if (hi < 0 || hi >= in_h || wi < 0 || wi >= in_w) return;
                        gi[hi * in_w + wi] = static_cast<T>(
                            static_cast<Compute>(gi[hi * in_w + wi]) + g_val * weight);
                    };
                    add(h0, w0, (Compute(1) - fh) * (Compute(1) - fw));
                    add(h0, w1, (Compute(1) - fh) * fw);
                    add(h1, w0, fh * (Compute(1) - fw));
                    add(h1, w1, fh * fw);
                }
            }
        }
    }
}

// =========================================================================
// Phase 2.20: full-set interpolate backward (CPU)
// =========================================================================
//
// Adjoints of forward interpolation. Forward kernels gather pixels from input
// to output via mode-specific weights; backward scatters output gradients back
// to input using the same weights (transpose of the linear forward op).
//
// Supported modes:
//   - "nearest"        — adjoint: each output gradient lands on its source pixel
//   - "nearest-exact"  — PyTorch's UpsampleNearestExact (rounds 0.5 to even)
//   - "linear"         — 1D, two-tap scatter
//   - "bilinear"       — 2D, four-tap scatter (existing impl)
//   - "bicubic"        — 2D, 16-tap Catmull-Rom scatter
//   - "trilinear"      — 3D, eight-tap scatter
//   - "area"           — adaptive average pooling adjoint (scatters
//                        output/area uniformly to overlapping input pixels)
//
// Supported ranks: 3 (N,C,W), 4 (N,C,H,W), 5 (N,C,D,H,W).

namespace {

// Source-coord rule shared by all modes that scale a single axis. Matches
// PyTorch `aten/src/ATen/native/UpSample.h::area_pixel_compute_source_index`.
inline float src_coord(int64_t dst, int64_t in_dim, int64_t out_dim, bool align_corners,
                       bool half_pixel) {
    if (align_corners) {
        return out_dim > 1
            ? static_cast<float>(dst) * static_cast<float>(in_dim - 1) /
              static_cast<float>(out_dim - 1)
            : 0.0f;
    }
    if (half_pixel) {
        // align_corners=false (PyTorch default for {bilinear, bicubic, trilinear, linear})
        return (static_cast<float>(dst) + 0.5f) *
               static_cast<float>(in_dim) / static_cast<float>(out_dim) - 0.5f;
    }
    // Plain scale (nearest, area)
    return static_cast<float>(dst) * static_cast<float>(in_dim) /
           static_cast<float>(out_dim);
}

// Nearest source index (PyTorch UpsampleNearest1d et al.). Floor(dst*scale).
inline int64_t nearest_src(int64_t dst, int64_t in_dim, int64_t out_dim) {
    const float scale = static_cast<float>(in_dim) / static_cast<float>(out_dim);
    return std::min(static_cast<int64_t>(std::floor(static_cast<float>(dst) * scale)),
                    in_dim - 1);
}

// Nearest-exact rule per `_compute_source_index_nearest_exact`: floor((dst+0.5)*scale).
inline int64_t nearest_exact_src(int64_t dst, int64_t in_dim, int64_t out_dim) {
    const float scale = static_cast<float>(in_dim) / static_cast<float>(out_dim);
    return std::min(static_cast<int64_t>(std::floor((static_cast<float>(dst) + 0.5f) * scale)),
                    in_dim - 1);
}

// ----- 1D linear backward -----
template<typename T>
void interpolate_linear_backward_impl(
    const T* grad_out, T* grad_in,
    int64_t N, int64_t C, int64_t in_w, int64_t out_w, bool align_corners)
{
    using Compute = interp_acc_t<T>;
    const Compute scale_w = align_corners && out_w > 1
        ? static_cast<Compute>(in_w - 1) / static_cast<Compute>(out_w - 1)
        : static_cast<Compute>(in_w) / static_cast<Compute>(out_w);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const T* go = grad_out + ((n * C + c) * out_w);
            T* gi = grad_in + ((n * C + c) * in_w);
            for (int64_t w = 0; w < out_w; ++w) {
                const Compute src_w = align_corners
                    ? static_cast<Compute>(w) * scale_w
                    : (static_cast<Compute>(w) + Compute(0.5)) * scale_w - Compute(0.5);
                const int64_t w0 = static_cast<int64_t>(std::floor(src_w));
                const int64_t w1 = w0 + 1;
                const Compute fw = src_w - static_cast<Compute>(w0);
                const Compute g_val = static_cast<Compute>(go[w]);
                auto add = [&](int64_t wi, Compute weight) {
                    if (wi < 0 || wi >= in_w) return;
                    gi[wi] = static_cast<T>(static_cast<Compute>(gi[wi]) + g_val * weight);
                };
                add(w0, Compute(1) - fw);
                add(w1, fw);
            }
        }
    }
}

// ----- 2D bicubic backward (Catmull-Rom) -----
template<typename T>
void interpolate_bicubic_backward_impl(
    const T* grad_out, T* grad_in,
    int64_t N, int64_t C, int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w, bool align_corners)
{
    using Compute = interp_acc_t<T>;
    const Compute scale_h = align_corners && out_h > 1
        ? static_cast<Compute>(in_h - 1) / static_cast<Compute>(out_h - 1)
        : static_cast<Compute>(in_h) / static_cast<Compute>(out_h);
    const Compute scale_w = align_corners && out_w > 1
        ? static_cast<Compute>(in_w - 1) / static_cast<Compute>(out_w - 1)
        : static_cast<Compute>(in_w) / static_cast<Compute>(out_w);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const T* go = grad_out + ((n * C + c) * out_h * out_w);
            T* gi = grad_in + ((n * C + c) * in_h * in_w);
            for (int64_t h = 0; h < out_h; ++h) {
                const Compute src_h = align_corners
                    ? static_cast<Compute>(h) * scale_h
                    : (static_cast<Compute>(h) + Compute(0.5)) * scale_h - Compute(0.5);
                const int64_t hi = static_cast<int64_t>(std::floor(src_h));
                for (int64_t w = 0; w < out_w; ++w) {
                    const Compute src_w = align_corners
                        ? static_cast<Compute>(w) * scale_w
                        : (static_cast<Compute>(w) + Compute(0.5)) * scale_w - Compute(0.5);
                    const int64_t wi = static_cast<int64_t>(std::floor(src_w));
                    const Compute g_val = static_cast<Compute>(go[h * out_w + w]);
                    for (int64_t dy = -1; dy <= 2; ++dy) {
                        const int64_t iy = std::clamp<int64_t>(hi + dy, 0, in_h - 1);
                        const Compute wy = cubic_interp_coeff<Compute>(
                            src_h - static_cast<Compute>(hi + dy));
                        for (int64_t dx = -1; dx <= 2; ++dx) {
                            const int64_t ix = std::clamp<int64_t>(wi + dx, 0, in_w - 1);
                            const Compute wx = cubic_interp_coeff<Compute>(
                                src_w - static_cast<Compute>(wi + dx));
                            const Compute weight = wy * wx;
                            gi[iy * in_w + ix] = static_cast<T>(
                                static_cast<Compute>(gi[iy * in_w + ix]) + g_val * weight);
                        }
                    }
                }
            }
        }
    }
}

// ----- 3D trilinear backward -----
template<typename T>
void interpolate_trilinear_backward_impl(
    const T* grad_out, T* grad_in,
    int64_t N, int64_t C,
    int64_t in_d, int64_t in_h, int64_t in_w,
    int64_t out_d, int64_t out_h, int64_t out_w, bool align_corners)
{
    using Compute = interp_acc_t<T>;
    const Compute scale_d = align_corners && out_d > 1
        ? static_cast<Compute>(in_d - 1) / static_cast<Compute>(out_d - 1)
        : static_cast<Compute>(in_d) / static_cast<Compute>(out_d);
    const Compute scale_h = align_corners && out_h > 1
        ? static_cast<Compute>(in_h - 1) / static_cast<Compute>(out_h - 1)
        : static_cast<Compute>(in_h) / static_cast<Compute>(out_h);
    const Compute scale_w = align_corners && out_w > 1
        ? static_cast<Compute>(in_w - 1) / static_cast<Compute>(out_w - 1)
        : static_cast<Compute>(in_w) / static_cast<Compute>(out_w);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const T* go = grad_out + (((n * C + c) * out_d) * out_h * out_w);
            T* gi = grad_in + (((n * C + c) * in_d) * in_h * in_w);
            for (int64_t od = 0; od < out_d; ++od) {
                const Compute src_d = align_corners
                    ? static_cast<Compute>(od) * scale_d
                    : (static_cast<Compute>(od) + Compute(0.5)) * scale_d - Compute(0.5);
                const int64_t d0 = static_cast<int64_t>(std::floor(src_d));
                const int64_t d1 = d0 + 1;
                const Compute fd = src_d - static_cast<Compute>(d0);
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    const Compute src_h = align_corners
                        ? static_cast<Compute>(oh) * scale_h
                        : (static_cast<Compute>(oh) + Compute(0.5)) * scale_h - Compute(0.5);
                    const int64_t h0 = static_cast<int64_t>(std::floor(src_h));
                    const int64_t h1 = h0 + 1;
                    const Compute fh = src_h - static_cast<Compute>(h0);
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        const Compute src_w = align_corners
                            ? static_cast<Compute>(ow) * scale_w
                            : (static_cast<Compute>(ow) + Compute(0.5)) * scale_w - Compute(0.5);
                        const int64_t w0 = static_cast<int64_t>(std::floor(src_w));
                        const int64_t w1 = w0 + 1;
                        const Compute fw = src_w - static_cast<Compute>(w0);
                        const Compute g_val = static_cast<Compute>(
                            go[(od * out_h + oh) * out_w + ow]);
                        auto add = [&](int64_t di, int64_t hi, int64_t wi, Compute weight) {
                            if (di < 0 || di >= in_d ||
                                hi < 0 || hi >= in_h ||
                                wi < 0 || wi >= in_w) return;
                            const int64_t idx = (di * in_h + hi) * in_w + wi;
                            gi[idx] = static_cast<T>(static_cast<Compute>(gi[idx]) + g_val * weight);
                        };
                        const Compute one = Compute(1);
                        add(d0, h0, w0, (one-fd)*(one-fh)*(one-fw));
                        add(d0, h0, w1, (one-fd)*(one-fh)*fw);
                        add(d0, h1, w0, (one-fd)*fh*(one-fw));
                        add(d0, h1, w1, (one-fd)*fh*fw);
                        add(d1, h0, w0, fd*(one-fh)*(one-fw));
                        add(d1, h0, w1, fd*(one-fh)*fw);
                        add(d1, h1, w0, fd*fh*(one-fw));
                        add(d1, h1, w1, fd*fh*fw);
                    }
                }
            }
        }
    }
}

// ----- nearest backward (any rank): single-pixel scatter -----
// nearest_exact=true uses PyTorch's "_exact" indexing rule.
template<typename T>
void nearest_backward_axis_scatter(
    const T* grad_out, T* grad_in,
    int64_t N, int64_t C,
    const std::vector<int64_t>& in_spatial,
    const std::vector<int64_t>& out_spatial,
    bool nearest_exact)
{
    using Compute = interp_acc_t<T>;
    const int64_t spatial_dims = static_cast<int64_t>(in_spatial.size());
    int64_t out_total = 1, in_total = 1;
    for (int64_t s : out_spatial) out_total *= s;
    for (int64_t s : in_spatial) in_total *= s;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const T* go = grad_out + ((n * C + c) * out_total);
            T* gi = grad_in + ((n * C + c) * in_total);
            for (int64_t out_idx = 0; out_idx < out_total; ++out_idx) {
                // Decode out_idx into per-dim indices then map to in_idx.
                int64_t in_idx = 0;
                int64_t in_stride = 1;
                int64_t tmp = out_idx;
                std::vector<int64_t> src_indices(spatial_dims);
                // Compute src for each dim, unwind right-to-left.
                for (int64_t d = spatial_dims - 1; d >= 0; --d) {
                    const int64_t dim_idx = tmp % out_spatial[d];
                    tmp /= out_spatial[d];
                    src_indices[d] = nearest_exact
                        ? nearest_exact_src(dim_idx, in_spatial[d], out_spatial[d])
                        : nearest_src(dim_idx, in_spatial[d], out_spatial[d]);
                }
                for (int64_t d = spatial_dims - 1; d >= 0; --d) {
                    in_idx += src_indices[d] * in_stride;
                    in_stride *= in_spatial[d];
                }
                gi[in_idx] = static_cast<T>(
                    static_cast<Compute>(gi[in_idx]) + static_cast<Compute>(go[out_idx]));
            }
        }
    }
}

// ----- area backward (adaptive average pooling adjoint), any rank -----
// Forward `area` divides each output cell's weight uniformly over the input
// pixels whose centers fall inside the output cell's bin. Adjoint scatters
// `grad_out / area` uniformly back to those same input pixels.
template<typename T>
void area_backward_impl(
    const T* grad_out, T* grad_in,
    int64_t N, int64_t C,
    const std::vector<int64_t>& in_spatial,
    const std::vector<int64_t>& out_spatial)
{
    using Compute = interp_acc_t<T>;
    const int64_t spatial_dims = static_cast<int64_t>(in_spatial.size());
    int64_t out_total = 1, in_total = 1;
    for (int64_t s : out_spatial) out_total *= s;
    for (int64_t s : in_spatial) in_total *= s;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const T* go = grad_out + ((n * C + c) * out_total);
            T* gi = grad_in + ((n * C + c) * in_total);
            for (int64_t out_idx = 0; out_idx < out_total; ++out_idx) {
                // Decode out_idx into per-dim indices.
                std::vector<int64_t> dst(spatial_dims);
                int64_t tmp = out_idx;
                for (int64_t d = spatial_dims - 1; d >= 0; --d) {
                    dst[d] = tmp % out_spatial[d];
                    tmp /= out_spatial[d];
                }
                // For each spatial dim, compute the [start, end) input range
                // that maps to this output cell. PyTorch convention:
                //   start = floor(d * in/out), end = ceil((d+1) * in/out)
                std::vector<int64_t> starts(spatial_dims), ends(spatial_dims);
                int64_t area = 1;
                for (int64_t d = 0; d < spatial_dims; ++d) {
                    const Compute ratio_lo = static_cast<Compute>(dst[d]) *
                        static_cast<Compute>(in_spatial[d]) / static_cast<Compute>(out_spatial[d]);
                    const Compute ratio_hi = static_cast<Compute>(dst[d] + 1) *
                        static_cast<Compute>(in_spatial[d]) / static_cast<Compute>(out_spatial[d]);
                    starts[d] = std::max<int64_t>(0,
                        static_cast<int64_t>(std::floor(ratio_lo)));
                    ends[d] = std::min<int64_t>(in_spatial[d],
                        static_cast<int64_t>(std::ceil(ratio_hi)));
                    area *= std::max<int64_t>(1, ends[d] - starts[d]);
                }
                const Compute g_val = static_cast<Compute>(go[out_idx]) /
                                      static_cast<Compute>(area);
                // Iterate over the input region and accumulate.
                std::vector<int64_t> it = starts;
                while (true) {
                    int64_t in_idx = 0, in_stride = 1;
                    for (int64_t d = spatial_dims - 1; d >= 0; --d) {
                        in_idx += it[d] * in_stride;
                        in_stride *= in_spatial[d];
                    }
                    gi[in_idx] = static_cast<T>(static_cast<Compute>(gi[in_idx]) + g_val);
                    // Advance: rightmost iterator first.
                    int64_t d = spatial_dims - 1;
                    while (d >= 0) {
                        ++it[d];
                        if (it[d] < ends[d]) break;
                        it[d] = starts[d];
                        --d;
                    }
                    if (d < 0) break;
                }
            }
        }
    }
}

}  // anonymous namespace

auto interpolate_backward_kernel(const Tensor& grad_output,
                                  const std::vector<int64_t>& input_size,
                                  const std::string& mode,
                                  bool align_corners) -> Tensor {
    const auto& shape = grad_output.shape();
    if (shape.size() < 3 || shape.size() > 5) {
        throw std::runtime_error(
            "interpolate_backward_kernel: rank must be 3, 4, or 5 (N,C,...), got " +
            std::to_string(shape.size()) + "D");
    }
    const int64_t spatial_dims = static_cast<int64_t>(shape.size()) - 2;
    if (static_cast<int64_t>(input_size.size()) != spatial_dims) {
        throw std::runtime_error(
            "interpolate_backward_kernel: input_size has " +
            std::to_string(input_size.size()) +
            " entries but grad_output rank implies " +
            std::to_string(spatial_dims) + " spatial dims.");
    }
    const int64_t N = shape[0];
    const int64_t C = shape[1];
    std::vector<int64_t> out_spatial;
    out_spatial.reserve(spatial_dims);
    for (int64_t d = 0; d < spatial_dims; ++d) out_spatial.push_back(shape[2 + d]);

    // Validate (mode, rank) combinations and normalise.
    const bool is_nearest        = (mode == "nearest");
    const bool is_nearest_exact  = (mode == "nearest-exact" || mode == "nearest_exact");
    const bool is_linear         = (mode == "linear");        // 3D only
    const bool is_bilinear       = (mode == "bilinear");      // 4D only
    const bool is_bicubic        = (mode == "bicubic");       // 4D only
    const bool is_trilinear      = (mode == "trilinear");     // 5D only
    const bool is_area           = (mode == "area");
    if (!is_nearest && !is_nearest_exact && !is_linear && !is_bilinear &&
        !is_bicubic && !is_trilinear && !is_area) {
        throw std::runtime_error(
            "interpolate_backward_kernel: unsupported mode '" + mode +
            "'. Supported: nearest, nearest-exact, linear (3D), bilinear (4D), "
            "bicubic (4D), trilinear (5D), area.");
    }
    if (is_linear && spatial_dims != 1) {
        throw std::runtime_error("interpolate_backward_kernel: mode 'linear' requires 3D input.");
    }
    if ((is_bilinear || is_bicubic) && spatial_dims != 2) {
        throw std::runtime_error("interpolate_backward_kernel: mode '" + mode +
                                 "' requires 4D input.");
    }
    if (is_trilinear && spatial_dims != 3) {
        throw std::runtime_error("interpolate_backward_kernel: mode 'trilinear' requires 5D input.");
    }

    std::vector<int64_t> grad_in_shape = {N, C};
    grad_in_shape.insert(grad_in_shape.end(), input_size.begin(), input_size.end());
    Tensor grad_input = zeros(grad_in_shape, grad_output.dtype(), grad_output.device());

    auto run = [&](auto* dummy) {
        using T = std::remove_pointer_t<decltype(dummy)>;
        const T* go = grad_output.data<T>();
        T* gi = grad_input.data<T>();

        if (is_nearest || is_nearest_exact) {
            nearest_backward_axis_scatter<T>(
                go, gi, N, C, input_size, out_spatial, is_nearest_exact);
            return;
        }
        if (is_area) {
            area_backward_impl<T>(go, gi, N, C, input_size, out_spatial);
            return;
        }
        if (is_linear) {
            interpolate_linear_backward_impl<T>(go, gi, N, C,
                input_size[0], out_spatial[0], align_corners);
            return;
        }
        if (is_bilinear) {
            interpolate_bilinear_backward_impl<T>(go, gi, N, C,
                input_size[0], input_size[1], out_spatial[0], out_spatial[1],
                align_corners);
            return;
        }
        if (is_bicubic) {
            interpolate_bicubic_backward_impl<T>(go, gi, N, C,
                input_size[0], input_size[1], out_spatial[0], out_spatial[1],
                align_corners);
            return;
        }
        if (is_trilinear) {
            interpolate_trilinear_backward_impl<T>(go, gi, N, C,
                input_size[0], input_size[1], input_size[2],
                out_spatial[0], out_spatial[1], out_spatial[2], align_corners);
            return;
        }
    };

    switch (grad_output.dtype()) {
        case DType::Float32:  run(static_cast<float*>(nullptr)); break;
        case DType::Float64:  run(static_cast<double*>(nullptr)); break;
        case DType::Float16:  run(static_cast<Float16*>(nullptr)); break;
        case DType::BFloat16: run(static_cast<BFloat16*>(nullptr)); break;
        default:
            throw std::runtime_error("interpolate_backward_kernel: unsupported dtype");
    }
    return grad_input;
}

// =========================================================================
// ROI Align Operations
// =========================================================================

namespace {

template<typename T, typename Compute = interp_acc_t<T>>
auto bilinear_interpolate(const T* data, int64_t height, int64_t width,
                          Compute y, Compute x) -> Compute {
    if (y < Compute(-1) || y > static_cast<Compute>(height) ||
        x < Compute(-1) || x > static_cast<Compute>(width)) {
        return Compute(0);
    }

    y = std::max(y, Compute(0));
    x = std::max(x, Compute(0));

    int64_t y_low = static_cast<int64_t>(y);
    int64_t x_low = static_cast<int64_t>(x);
    int64_t y_high = y_low + 1;
    int64_t x_high = x_low + 1;

    if (y_low >= height - 1) { y_low = y_high = height - 1; y = static_cast<Compute>(y_low); }
    if (x_low >= width - 1)  { x_low = x_high = width - 1;  x = static_cast<Compute>(x_low); }

    Compute ly = y - static_cast<Compute>(y_low);
    Compute lx = x - static_cast<Compute>(x_low);
    Compute hy = Compute(1) - ly;
    Compute hx = Compute(1) - lx;

    Compute v1 = static_cast<Compute>(data[y_low * width + x_low]);
    Compute v2 = static_cast<Compute>(data[y_low * width + x_high]);
    Compute v3 = static_cast<Compute>(data[y_high * width + x_low]);
    Compute v4 = static_cast<Compute>(data[y_high * width + x_high]);

    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}


template<typename T>
void roi_align_forward_impl(
    const T* feat_data, const T* roi_data, T* out_data,
    int64_t num_rois, int64_t channels, int64_t height, int64_t width,
    int64_t output_h, int64_t output_w,
    float spatial_scale_f, int64_t sampling_ratio, bool aligned)
{
    using Compute = interp_acc_t<T>;
    const Compute spatial_scale = static_cast<Compute>(spatial_scale_f);
    const Compute offset = aligned ? Compute(0.5) : Compute(0);

    #pragma omp parallel for if(num_rois > 16)
    for (int64_t n = 0; n < num_rois; ++n) {
        int64_t batch_idx = static_cast<int64_t>(roi_data[n * 5 + 0]);
        Compute roi_x1 = static_cast<Compute>(roi_data[n * 5 + 1]) * spatial_scale - offset;
        Compute roi_y1 = static_cast<Compute>(roi_data[n * 5 + 2]) * spatial_scale - offset;
        Compute roi_x2 = static_cast<Compute>(roi_data[n * 5 + 3]) * spatial_scale - offset;
        Compute roi_y2 = static_cast<Compute>(roi_data[n * 5 + 4]) * spatial_scale - offset;

        Compute roi_w = roi_x2 - roi_x1;
        Compute roi_h = roi_y2 - roi_y1;
        if (!aligned) {
            roi_w = std::max(roi_w, Compute(1));
            roi_h = std::max(roi_h, Compute(1));
        }

        Compute bin_h = roi_h / static_cast<Compute>(output_h);
        Compute bin_w = roi_w / static_cast<Compute>(output_w);

        int64_t roi_bin_h = sampling_ratio > 0 ? sampling_ratio
            : static_cast<int64_t>(std::ceil(bin_h));
        int64_t roi_bin_w = sampling_ratio > 0 ? sampling_ratio
            : static_cast<int64_t>(std::ceil(bin_w));
        roi_bin_h = std::max(roi_bin_h, int64_t(1));
        roi_bin_w = std::max(roi_bin_w, int64_t(1));

        Compute count = static_cast<Compute>(roi_bin_h * roi_bin_w);

        for (int64_t c = 0; c < channels; ++c) {
            const T* channel_data = feat_data + (batch_idx * channels + c) * height * width;

            for (int64_t ph = 0; ph < output_h; ++ph) {
                for (int64_t pw = 0; pw < output_w; ++pw) {
                    Compute val = Compute(0);

                    for (int64_t iy = 0; iy < roi_bin_h; ++iy) {
                        Compute y = roi_y1 + bin_h * (static_cast<Compute>(ph) +
                            (static_cast<Compute>(iy) + Compute(0.5)) / static_cast<Compute>(roi_bin_h));
                        for (int64_t ix = 0; ix < roi_bin_w; ++ix) {
                            Compute x = roi_x1 + bin_w * (static_cast<Compute>(pw) +
                                (static_cast<Compute>(ix) + Compute(0.5)) / static_cast<Compute>(roi_bin_w));
                            val += bilinear_interpolate<T, Compute>(channel_data, height, width, y, x);
                        }
                    }

                    out_data[((n * channels + c) * output_h + ph) * output_w + pw] =
                        static_cast<T>(val / count);
                }
            }
        }
    }
}

template<typename T>
void roi_align_backward_impl(
    const T* go_data, const T* roi_data, T* gi_data,
    int64_t num_rois, int64_t channels, int64_t feat_height, int64_t feat_width,
    int64_t output_h, int64_t output_w,
    float spatial_scale_f, int64_t sampling_ratio, bool aligned)
{
    using Compute = interp_acc_t<T>;
    const Compute spatial_scale = static_cast<Compute>(spatial_scale_f);
    const Compute offset = aligned ? Compute(0.5) : Compute(0);

    #pragma omp parallel for if(num_rois > 16)
    for (int64_t n = 0; n < num_rois; ++n) {
        int64_t batch_idx = static_cast<int64_t>(roi_data[n * 5 + 0]);
        Compute roi_x1 = static_cast<Compute>(roi_data[n * 5 + 1]) * spatial_scale - offset;
        Compute roi_y1 = static_cast<Compute>(roi_data[n * 5 + 2]) * spatial_scale - offset;
        Compute roi_x2 = static_cast<Compute>(roi_data[n * 5 + 3]) * spatial_scale - offset;
        Compute roi_y2 = static_cast<Compute>(roi_data[n * 5 + 4]) * spatial_scale - offset;

        Compute roi_w = roi_x2 - roi_x1;
        Compute roi_h = roi_y2 - roi_y1;
        if (!aligned) {
            roi_w = std::max(roi_w, Compute(1));
            roi_h = std::max(roi_h, Compute(1));
        }

        Compute bin_h = roi_h / static_cast<Compute>(output_h);
        Compute bin_w = roi_w / static_cast<Compute>(output_w);

        int64_t roi_bin_h = sampling_ratio > 0 ? sampling_ratio
            : static_cast<int64_t>(std::ceil(bin_h));
        int64_t roi_bin_w = sampling_ratio > 0 ? sampling_ratio
            : static_cast<int64_t>(std::ceil(bin_w));
        roi_bin_h = std::max(roi_bin_h, int64_t(1));
        roi_bin_w = std::max(roi_bin_w, int64_t(1));

        Compute count = static_cast<Compute>(roi_bin_h * roi_bin_w);

        for (int64_t c = 0; c < channels; ++c) {
            T* gi_channel = gi_data + (batch_idx * channels + c) * feat_height * feat_width;

            for (int64_t ph = 0; ph < output_h; ++ph) {
                for (int64_t pw = 0; pw < output_w; ++pw) {
                    Compute grad_val =
                        static_cast<Compute>(go_data[((n * channels + c) * output_h + ph) * output_w + pw])
                        / count;

                    for (int64_t iy = 0; iy < roi_bin_h; ++iy) {
                        Compute y = roi_y1 + bin_h * (static_cast<Compute>(ph) +
                            (static_cast<Compute>(iy) + Compute(0.5)) / static_cast<Compute>(roi_bin_h));
                        for (int64_t ix = 0; ix < roi_bin_w; ++ix) {
                            Compute x = roi_x1 + bin_w * (static_cast<Compute>(pw) +
                                (static_cast<Compute>(ix) + Compute(0.5)) / static_cast<Compute>(roi_bin_w));

                            if (y < Compute(-1) || y > static_cast<Compute>(feat_height) ||
                                x < Compute(-1) || x > static_cast<Compute>(feat_width)) continue;

                            y = std::max(y, Compute(0));
                            x = std::max(x, Compute(0));

                            int64_t y_low = static_cast<int64_t>(y);
                            int64_t x_low = static_cast<int64_t>(x);
                            int64_t y_high = y_low + 1;
                            int64_t x_high = x_low + 1;

                            if (y_low >= feat_height - 1) { y_low = y_high = feat_height - 1; y = static_cast<Compute>(y_low); }
                            if (x_low >= feat_width  - 1) { x_low = x_high = feat_width  - 1; x = static_cast<Compute>(x_low); }

                            Compute ly = y - static_cast<Compute>(y_low);
                            Compute lx = x - static_cast<Compute>(x_low);
                            Compute hy = Compute(1) - ly;
                            Compute hx = Compute(1) - lx;

                            // Each atomic uses a small read-modify-write because the
                            // accumulation type is Compute and storage is T — for
                            // Float64 storage Compute is double (atomic add OK), for
                            // float storage Compute is float (atomic add OK). For
                            // Float16/BFloat16 inputs we widen at the kernel level
                            // (see the dispatch wrapper) so this branch is never hit
                            // with narrower storage.
                            #pragma omp atomic
                            gi_channel[y_low  * feat_width + x_low ] += static_cast<T>(grad_val * hy * hx);
                            #pragma omp atomic
                            gi_channel[y_low  * feat_width + x_high] += static_cast<T>(grad_val * hy * lx);
                            #pragma omp atomic
                            gi_channel[y_high * feat_width + x_low ] += static_cast<T>(grad_val * ly * hx);
                            #pragma omp atomic
                            gi_channel[y_high * feat_width + x_high] += static_cast<T>(grad_val * ly * lx);
                        }
                    }
                }
            }
        }
    }
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
    int64_t height   = feat_shape[2];
    int64_t width    = feat_shape[3];
    int64_t num_rois = rois.shape()[0];

    Tensor output({num_rois, channels, output_h, output_w},
                  features.dtype(), features.device());

    switch (features.dtype()) {
        case DType::Float32:
            roi_align_forward_impl<float>(
                features.data<float>(), rois.data<float>(), output.data<float>(),
                num_rois, channels, height, width, output_h, output_w,
                spatial_scale, sampling_ratio, aligned);
            break;
        case DType::Float64:
            roi_align_forward_impl<double>(
                features.data<double>(), rois.data<double>(), output.data<double>(),
                num_rois, channels, height, width, output_h, output_w,
                spatial_scale, sampling_ratio, aligned);
            break;
        case DType::Float16:
        case DType::BFloat16: {
            // Widen to Float32 for the inner kernel; narrow at store.
            auto features_f32 = features.to(DType::Float32);
            auto rois_f32     = rois.to(DType::Float32);
            auto output_f32   = roi_align_forward_kernel(
                features_f32, rois_f32, output_h, output_w,
                spatial_scale, sampling_ratio, aligned);
            return output_f32.to(features.dtype());
        }
        default:
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

    switch (grad_output.dtype()) {
        case DType::Float32:
            roi_align_backward_impl<float>(
                grad_output.data<float>(), rois.data<float>(), grad_input.data<float>(),
                num_rois, channels, feat_height, feat_width, output_h, output_w,
                spatial_scale, sampling_ratio, aligned);
            break;
        case DType::Float64:
            roi_align_backward_impl<double>(
                grad_output.data<double>(), rois.data<double>(), grad_input.data<double>(),
                num_rois, channels, feat_height, feat_width, output_h, output_w,
                spatial_scale, sampling_ratio, aligned);
            break;
        case DType::Float16:
        case DType::BFloat16: {
            // Widen to Float32 for the inner kernel; narrow on return.
            auto go_f32   = grad_output.to(DType::Float32);
            auto rois_f32 = rois.to(DType::Float32);
            auto result   = roi_align_backward_kernel(
                go_f32, rois_f32, batch_size, feat_height, feat_width,
                spatial_scale, sampling_ratio, aligned);
            return result.to(grad_output.dtype());
        }
        default:
            throw std::runtime_error("roi_align_backward: unsupported dtype");
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

    #pragma omp parallel for collapse(2) if(N * M > ::tenzor::OmpThresholds::complex())
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

// Per-axis unfold (im2col). Supports asymmetric kernel/stride/padding/dilation
// (kh!=kw etc.), matching the dispatcher's per-axis attrs and nn.Unfold.
auto unfold_kernel(const Tensor& input,
                   int64_t kh, int64_t kw, int64_t sh, int64_t sw,
                   int64_t ph, int64_t pw, int64_t dh, int64_t dw) -> Tensor {
    // input: (N, C, H, W); output: (N, C * kh * kw, L), L = H_out * W_out.
    const auto& shape = input.shape();
    int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];

    int64_t H_out = (H + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    int64_t W_out = (W + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
    int64_t L = H_out * W_out;
    int64_t cols_per_channel = kh * kw;

    Tensor output({N, C * cols_per_channel, L}, input.dtype(), input.device());

    auto unfold_impl = [&]<typename T>(const T* in_data, T* out_data) {
        #pragma omp parallel for if(N > 4)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                const T* channel = in_data + (n * C + c) * H * W;
                for (int64_t i = 0; i < kh; ++i) {
                    for (int64_t j = 0; j < kw; ++j) {
                        int64_t col_idx = (c * cols_per_channel + i * kw + j);
                        T* col = out_data + (n * C * cols_per_channel + col_idx) * L;

                        for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                            for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                                int64_t h_in = h_out * sh - ph + i * dh;
                                int64_t w_in = w_out * sw - pw + j * dw;
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

// Symmetric convenience wrapper (preserves the original API).
auto unfold_kernel(const Tensor& input, int64_t kernel_size,
                   int64_t stride, int64_t padding, int64_t dilation) -> Tensor {
    return unfold_kernel(input, kernel_size, kernel_size, stride, stride,
                         padding, padding, dilation, dilation);
}

// Per-axis fold (col2im). Supports asymmetric kernel/stride/padding/dilation.
auto fold_kernel(const Tensor& input, const std::vector<int64_t>& output_size,
                 int64_t kh, int64_t kw, int64_t sh, int64_t sw,
                 int64_t ph, int64_t pw, int64_t dh, int64_t dw) -> Tensor {
    // input: (N, C * kh * kw, L); output: (N, C, output_size[0], output_size[1])
    const auto& shape = input.shape();
    int64_t N = shape[0];
    int64_t H_out = output_size[0], W_out = output_size[1];
    int64_t cols_per_channel = kh * kw;
    if (cols_per_channel <= 0 || shape[1] % cols_per_channel != 0) {
        throw std::runtime_error("fold: dim 1 (C*kh*kw) must be divisible by kh*kw");
    }
    int64_t C = shape[1] / cols_per_channel;

    int64_t H_col = (H_out + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    int64_t W_col = (W_out + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());
    std::memset(output.data<uint8_t>(), 0, output.numel() * dtype_size(output.dtype()));

    auto fold_impl = [&]<typename T>(const T* in_data, T* out_data) {
        #pragma omp parallel for if(N > 4)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                T* channel = out_data + (n * C + c) * H_out * W_out;
                for (int64_t i = 0; i < kh; ++i) {
                    for (int64_t j = 0; j < kw; ++j) {
                        int64_t col_idx = c * cols_per_channel + i * kw + j;
                        const T* col = in_data + (n * C * cols_per_channel + col_idx) * (H_col * W_col);

                        for (int64_t h_col = 0; h_col < H_col; ++h_col) {
                            for (int64_t w_col = 0; w_col < W_col; ++w_col) {
                                int64_t h_in = h_col * sh - ph + i * dh;
                                int64_t w_in = w_col * sw - pw + j * dw;

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

// Symmetric convenience wrapper (preserves the original API).
auto fold_kernel(const Tensor& input, const std::vector<int64_t>& output_size,
                 int64_t kernel_size, int64_t stride, int64_t padding,
                 int64_t dilation) -> Tensor {
    return fold_kernel(input, output_size, kernel_size, kernel_size,
                       stride, stride, padding, padding, dilation, dilation);
}

auto gather_relative_position_bias_kernel(const Tensor& bias_table, const Tensor& rel_pos_index,
                                           int64_t num_positions, int64_t num_heads) -> Tensor {
    // Convert half precision to float
    if (bias_table.dtype() == DType::Float16 || bias_table.dtype() == DType::BFloat16) {
        auto bt_f32 = bias_table.to(DType::Float32);
        auto result = gather_relative_position_bias_kernel(bt_f32, rel_pos_index, num_positions, num_heads);
        return result.to(bias_table.dtype());
    }

    auto idx_shape = rel_pos_index.shape();
    int64_t seq_len = idx_shape[0];
    // Output: [num_heads, seq_len, seq_len]
    Tensor output({num_heads, seq_len, seq_len}, bias_table.dtype(), bias_table.device());

    const int64_t* idx_data = rel_pos_index.data<int64_t>();
    int64_t table_stride = bias_table.shape()[1]; // second dim of bias table

    // Bounds-check every relative-position index up front (sequentially, so we
    // can throw cleanly rather than from inside the OpenMP region below). Each
    // idx is used as an unchecked offset into bias_table; an out-of-range or
    // negative value (e.g. from an untrusted checkpoint or mismatched shapes)
    // would otherwise be an OOB read / information-disclosure vector.
    int64_t num_index = rel_pos_index.numel();
    for (int64_t pos = 0; pos < num_index; ++pos) {
        int64_t idx = idx_data[pos];
        if (idx < 0 || idx >= table_stride) {
            throw std::out_of_range(
                "gather_relative_position_bias: rel_pos_index value " + std::to_string(idx) +
                " out of range [0, " + std::to_string(table_stride) + ")");
        }
    }

    TENZOR_DISPATCH_FLOATING_TYPES(bias_table.dtype(), "gather_rel_pos_bias", [&]() {
        const scalar_t* table_data = bias_table.data<scalar_t>();
        scalar_t* out_data = output.data<scalar_t>();
        int64_t total = num_heads * seq_len * seq_len;
        _Pragma("omp parallel for if(total > 10000)")
        for (int64_t i = 0; i < total; i++) {
            int64_t h = i / (seq_len * seq_len);
            int64_t pos = i % (seq_len * seq_len);
            int64_t idx = idx_data[pos];
            out_data[i] = table_data[h * table_stride + idx];
        }
    });
    return output;
}

} // namespace cpu
} // namespace tenzor
