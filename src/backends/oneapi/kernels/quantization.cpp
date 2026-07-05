#include "tenzor/core/tensor.hpp"
#include "oneapi_kernel_utils.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace tenzor {
namespace oneapi {

// SYCL Kernel name classes
struct QuantizeKernelFloat32 {};
struct DequantizeKernelFloat32 {};
struct QuantizedLinearKernelInt8 {};
struct QuantizedConv2dKernelInt8 {};


// ============================================================================
// Quantize Operation
// ============================================================================
/**
 * @brief Quantize float tensor to int8
 *
 * Converts floating point values to int8 using scale and zero_point.
 * q = clamp(round(x / scale + zero_point), -128, 127)
 *
 * @param input Float tensor to quantize
 * @param scale Scale factor
 * @param zero_point Zero point offset
 * @return Quantized int8 tensor
 */
auto quantize_kernel(
    const Tensor& input_in,
    float scale,
    int32_t zero_point,
    sycl::queue& queue
) -> Tensor {
    // The kernel indexes input by flat position, so a non-contiguous view must be
    // materialized contiguous first.
    const Tensor input = input_in.is_contiguous() ? input_in : input_in.contiguous();
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  DType::Int8, input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        int8_t* out_ptr = get_data_ptr<int8_t>(output);

        queue.parallel_for<QuantizeKernelFloat32>(
            sycl::range<1>(numel),
            [=](sycl::id<1> idx) {
                float val = in_ptr[idx] / scale + static_cast<float>(zero_point);
                // Round half-to-even (sycl::rint) to match the authoritative
                // quantize.cpp std::nearbyint / QAT round_half_to_even. sycl::round
                // rounds half-away-from-zero and codes .5 ties one step apart.
                val = sycl::rint(val);
                val = sycl::fmax(-128.0f, sycl::fmin(127.0f, val));
                out_ptr[idx] = static_cast<int8_t>(val);
            }
        );
    }
    else {
        throw std::runtime_error("quantize: only Float32 input supported");
    }

    // Drain before the host reads the USM-shared output (and before the local
    // contiguous input copy is freed); the parallel_for is async.
    queue.wait_and_throw();
    return output;
}

// ============================================================================
// Dequantize Operation
// ============================================================================
/**
 * @brief Dequantize int8 tensor to float
 *
 * Converts int8 values back to floating point.
 * x = (q - zero_point) * scale
 *
 * @param input Int8 tensor to dequantize
 * @param scale Scale factor
 * @param zero_point Zero point offset
 * @return Dequantized float tensor
 */
auto dequantize_kernel(
    const Tensor& input_in,
    float scale,
    int32_t zero_point,
    sycl::queue& queue
) -> Tensor {
    // Indexes input by flat position; materialize a non-contiguous view.
    const Tensor input = input_in.is_contiguous() ? input_in : input_in.contiguous();
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  DType::Float32, input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Int8) {
        const int8_t* in_ptr = get_data_ptr<const int8_t>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<DequantizeKernelFloat32>(
            sycl::range<1>(numel),
            [=](sycl::id<1> idx) {
                int8_t q_val = in_ptr[idx];
                out_ptr[idx] = (static_cast<float>(q_val) - static_cast<float>(zero_point)) * scale;
            }
        );
    }
    else {
        throw std::runtime_error("dequantize: only Int8 input supported");
    }

    queue.wait_and_throw();
    return output;
}

// ============================================================================
// Quantized Linear (Fully Connected)
// ============================================================================
/**
 * @brief Quantized linear layer
 *
 * Computes quantized matrix multiplication with fused dequantization.
 * output = dequantize(quantize(input) @ quantize(weight)^T) + bias
 *
 * Uses int8 accumulation with int32 intermediate results.
 *
 * @param input Quantized input: (batch, in_features) int8
 * @param weight Quantized weight: (out_features, in_features) int8
 * @param bias Float bias: (out_features,)
 * @param input_scale Input quantization scale
 * @param input_zero_point Input zero point
 * @param weight_scale Weight quantization scale
 * @param weight_zero_point Weight zero point
 * @param output_scale Output quantization scale
 * @param output_zero_point Output zero point
 * @return Output tensor (dequantized to float)
 */
auto quantized_linear_kernel(
    const Tensor& input_in,
    const Tensor& weight_in,
    const Tensor* bias_in,
    float input_scale,
    int32_t input_zero_point,
    float weight_scale,
    int32_t weight_zero_point,
    float output_scale,
    int32_t output_zero_point,
    sycl::queue& queue,
    // Per-channel (inputs[3]/[4]): [out_features] weight scales / zero-points.
    // When weight_scales != nullptr, each output channel o uses weight_scales[o]
    // and (weight_zps ? weight_zps[o] : 0) instead of the scalar arguments —
    // mirroring the CPU quantized_linear_per_channel_kernel.
    const Tensor* weight_scales,
    const Tensor* weight_zps
) -> Tensor {
    // The kernel reads input/weight/bias with shape-derived flat offsets, so all
    // must be contiguous or it reads at wrong physical positions.
    const Tensor input = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor weight = weight_in.is_contiguous() ? weight_in : weight_in.contiguous();
    Tensor bias_cont;
    const Tensor* bias = bias_in;
    if (bias_in && !bias_in->is_contiguous()) {
        bias_cont = bias_in->contiguous();
        bias = &bias_cont;
    }
    auto input_shape = input.shape();
    int64_t batch_size = 1;
    for (size_t i = 0; i < input_shape.size() - 1; ++i) {
        batch_size *= input_shape[i];
    }
    int64_t in_features = input_shape[input_shape.size() - 1];
    int64_t out_features = weight.shape()[0];

    // Output shape
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end() - 1);
    output_shape.push_back(out_features);

    // Create float output (dequantized)
    Tensor output(output_shape, DType::Float32, input.device());

    int64_t total_elements = batch_size * out_features;

    // Per-channel weight scale/zero-point (optional). Materialize contiguous so
    // the kernel can index by output channel.
    const bool per_channel = (weight_scales != nullptr) && (weight_scales->numel() > 1);
    Tensor wscale_cont, wzp_cont;
    const float* wscale_ptr = nullptr;
    const int32_t* wzp_ptr = nullptr;
    if (per_channel) {
        wscale_cont = weight_scales->is_contiguous() ? *weight_scales : weight_scales->contiguous();
        wscale_ptr = get_data_ptr<const float>(wscale_cont);
        if (weight_zps != nullptr && weight_zps->numel() > 0) {
            wzp_cont = weight_zps->is_contiguous() ? *weight_zps : weight_zps->contiguous();
            wzp_ptr = get_data_ptr<const int32_t>(wzp_cont);
        }
    }

    if (input.dtype() == DType::Int8 && weight.dtype() == DType::Int8) {
        const int8_t* in_ptr = get_data_ptr<const int8_t>(input);
        const int8_t* weight_ptr = get_data_ptr<const int8_t>(weight);
        const float* bias_ptr = bias ? get_data_ptr<const float>(*bias) : nullptr;
        float* out_ptr = get_data_ptr<float>(output);

        const bool has_bias = (bias != nullptr);
        // Match the CPU contract (quantized_linear.cpp): the dequantized result
        // is scaled by input_scale * weight_scale / output_scale. Ignoring
        // output_scale here diverged from CPU for any output_scale != 1.0.
        const float safe_output_scale = (output_scale != 0.0f) ? output_scale : 1.0f;
        const float combined_scale = input_scale * weight_scale / safe_output_scale;
        const float input_scale_pc = input_scale;  // per-channel needs input_scale separately

        queue.parallel_for<QuantizedLinearKernelInt8>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t b = idx / out_features;
                int64_t o = idx % out_features;

                // Per-channel weight scale / zero-point (each output channel o).
                const int32_t w_zp = per_channel ? (wzp_ptr ? wzp_ptr[o] : 0)
                                                  : weight_zero_point;
                const float chan_scale = per_channel
                    ? (input_scale_pc * wscale_ptr[o] / safe_output_scale)
                    : combined_scale;

                // Accumulate in int64 to match the CUDA kernel
                // (quantized_linear.cu): each (q-zp)*(q-zp) product fits in
                // int32, but the sum over in_features can exceed INT32_MAX for
                // wide layers (in_features >~ 131072), so accumulate in int64 to
                // avoid silent wraparound.
                int64_t acc = 0;
                for (int64_t i = 0; i < in_features; ++i) {
                    int32_t input_val = static_cast<int32_t>(in_ptr[b * in_features + i]) - input_zero_point;
                    int32_t weight_val = static_cast<int32_t>(weight_ptr[o * in_features + i]) - w_zp;
                    acc += static_cast<int64_t>(input_val) * static_cast<int64_t>(weight_val);
                }

                // Dequantize to float
                float result = static_cast<float>(acc) * chan_scale;

                // Add bias if present
                if (has_bias) {
                    result += bias_ptr[o];
                }

                out_ptr[idx] = result;
            }
        );
    }
    else {
        throw std::runtime_error("quantized_linear: requires Int8 input and weight");
    }

    queue.wait_and_throw();
    return output;
}

// ============================================================================
// Quantized Conv2d
// ============================================================================
/**
 * @brief Quantized 2D convolution
 *
 * Computes quantized convolution with fused dequantization.
 *
 * @param input Quantized input: (N, C_in, H, W) int8
 * @param weight Quantized weight: (C_out, C_in, kH, kW) int8
 * @param bias Float bias: (C_out,) or nullptr
 * @param stride Convolution stride
 * @param padding Padding
 * @param dilation Dilation
 * @param groups Number of groups
 * @param input_scale Input quantization scale
 * @param input_zero_point Input zero point
 * @param weight_scale Weight quantization scale
 * @param weight_zero_point Weight zero point
 * @return Output tensor (dequantized to float)
 */
auto quantized_conv2d_kernel(
    const Tensor& input_in,
    const Tensor& weight_in,
    const Tensor* bias_in,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t groups,
    float input_scale,
    int32_t input_zero_point,
    float weight_scale,
    int32_t weight_zero_point,
    sycl::queue& queue,
    // Per-channel (inputs[3]/[4]): [out_channels] weight scales / zero-points.
    const Tensor* weight_scales,
    const Tensor* weight_zps
) -> Tensor {
    // The kernel reads input/weight/bias via shape-derived NCHW offsets, so all
    // must be contiguous.
    const Tensor input = input_in.is_contiguous() ? input_in : input_in.contiguous();
    const Tensor weight = weight_in.is_contiguous() ? weight_in : weight_in.contiguous();
    Tensor bias_cont;
    const Tensor* bias = bias_in;
    if (bias_in && !bias_in->is_contiguous()) {
        bias_cont = bias_in->contiguous();
        bias = &bias_cont;
    }
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch_size = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Calculate output dimensions (per-axis stride/padding/dilation, F044).
    int64_t out_height = (in_height + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_width = (in_width + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;

    // Create output tensor
    Tensor output({batch_size, out_channels, out_height, out_width}, DType::Float32, input.device());

    int64_t total_elements = batch_size * out_channels * out_height * out_width;
    int64_t channels_per_group = in_channels / groups;
    int64_t out_channels_per_group = out_channels / groups;

    // Per-channel weight scale/zero-point (optional). Materialize contiguous.
    const bool per_channel = (weight_scales != nullptr) && (weight_scales->numel() > 1);
    Tensor wscale_cont, wzp_cont;
    const float* wscale_ptr = nullptr;
    const int32_t* wzp_ptr = nullptr;
    if (per_channel) {
        wscale_cont = weight_scales->is_contiguous() ? *weight_scales : weight_scales->contiguous();
        wscale_ptr = get_data_ptr<const float>(wscale_cont);
        if (weight_zps != nullptr && weight_zps->numel() > 0) {
            wzp_cont = weight_zps->is_contiguous() ? *weight_zps : weight_zps->contiguous();
            wzp_ptr = get_data_ptr<const int32_t>(wzp_cont);
        }
    }

    if (input.dtype() == DType::Int8 && weight.dtype() == DType::Int8) {
        const int8_t* in_ptr = get_data_ptr<const int8_t>(input);
        const int8_t* weight_ptr = get_data_ptr<const int8_t>(weight);
        const float* bias_ptr = bias ? get_data_ptr<const float>(*bias) : nullptr;
        float* out_ptr = get_data_ptr<float>(output);

        const bool has_bias = (bias != nullptr);
        const float combined_scale = input_scale * weight_scale;
        const float input_scale_pc = input_scale;

        queue.parallel_for<QuantizedConv2dKernelInt8>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t ow = idx % out_width;
                int64_t oh = (idx / out_width) % out_height;
                int64_t oc = (idx / out_width / out_height) % out_channels;
                int64_t n = idx / out_width / out_height / out_channels;

                int64_t g = oc / out_channels_per_group;

                // Per-channel weight scale / zero-point (each output channel oc).
                const int32_t w_zp = per_channel ? (wzp_ptr ? wzp_ptr[oc] : 0)
                                                 : weight_zero_point;
                const float chan_scale = per_channel
                    ? (input_scale_pc * wscale_ptr[oc])
                    : combined_scale;

                // Accumulate in int64 for parity with the linear kernel / CUDA.
                int64_t acc = 0;

                for (int64_t ic_g = 0; ic_g < channels_per_group; ++ic_g) {
                    int64_t ic = g * channels_per_group + ic_g;

                    for (int64_t kh = 0; kh < kernel_h; ++kh) {
                        for (int64_t kw = 0; kw < kernel_w; ++kw) {
                            int64_t ih = oh * stride_h - pad_h + kh * dil_h;
                            int64_t iw = ow * stride_w - pad_w + kw * dil_w;

                            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                int64_t in_idx = n * in_channels * in_height * in_width +
                                                ic * in_height * in_width +
                                                ih * in_width + iw;
                                int64_t weight_idx = oc * channels_per_group * kernel_h * kernel_w +
                                                    ic_g * kernel_h * kernel_w +
                                                    kh * kernel_w + kw;

                                int32_t input_val = static_cast<int32_t>(in_ptr[in_idx]) - input_zero_point;
                                int32_t weight_val = static_cast<int32_t>(weight_ptr[weight_idx]) - w_zp;
                                acc += static_cast<int64_t>(input_val) * static_cast<int64_t>(weight_val);
                            }
                        }
                    }
                }

                // Dequantize
                float result = static_cast<float>(acc) * chan_scale;

                // Add bias
                if (has_bias) {
                    result += bias_ptr[oc];
                }

                out_ptr[idx] = result;
            }
        );
    }
    else {
        throw std::runtime_error("quantized_conv2d: requires Int8 input and weight");
    }

    queue.wait_and_throw();
    return output;
}

} // namespace oneapi
} // namespace tenzor
