#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace tenzor {
namespace adaptivecpp {

// SYCL Kernel name classes
struct QuantizeKernelFloat32 {};
struct DequantizeKernelFloat32 {};
struct QuantizedLinearKernelInt8 {};
struct QuantizedConv2dKernelInt8 {};

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

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
    const Tensor& input,
    float scale,
    int32_t zero_point,
    sycl::queue& queue
) -> Tensor {
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
                val = sycl::round(val);
                val = sycl::fmax(-128.0f, sycl::fmin(127.0f, val));
                out_ptr[idx] = static_cast<int8_t>(val);
            }
        ).wait();
    }
    else {
        throw std::runtime_error("quantize: only Float32 input supported");
    }

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
    const Tensor& input,
    float scale,
    int32_t zero_point,
    sycl::queue& queue
) -> Tensor {
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
        ).wait();
    }
    else {
        throw std::runtime_error("dequantize: only Int8 input supported");
    }

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
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    float input_scale,
    int32_t input_zero_point,
    float weight_scale,
    int32_t weight_zero_point,
    float output_scale,
    int32_t output_zero_point,
    sycl::queue& queue
) -> Tensor {
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

    if (input.dtype() == DType::Int8 && weight.dtype() == DType::Int8) {
        const int8_t* in_ptr = get_data_ptr<const int8_t>(input);
        const int8_t* weight_ptr = get_data_ptr<const int8_t>(weight);
        const float* bias_ptr = bias ? get_data_ptr<const float>(*bias) : nullptr;
        float* out_ptr = get_data_ptr<float>(output);

        const bool has_bias = (bias != nullptr);
        const float combined_scale = input_scale * weight_scale;

        queue.parallel_for<QuantizedLinearKernelInt8>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t b = idx / out_features;
                int64_t o = idx % out_features;

                // Accumulate in int32
                int32_t acc = 0;
                for (int64_t i = 0; i < in_features; ++i) {
                    int32_t input_val = static_cast<int32_t>(in_ptr[b * in_features + i]) - input_zero_point;
                    int32_t weight_val = static_cast<int32_t>(weight_ptr[o * in_features + i]) - weight_zero_point;
                    acc += input_val * weight_val;
                }

                // Dequantize to float
                float result = static_cast<float>(acc) * combined_scale;

                // Add bias if present
                if (has_bias) {
                    result += bias_ptr[o];
                }

                out_ptr[idx] = result;
            }
        ).wait();
    }
    else {
        throw std::runtime_error("quantized_linear: requires Int8 input and weight");
    }

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
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    float input_scale,
    int32_t input_zero_point,
    float weight_scale,
    int32_t weight_zero_point,
    sycl::queue& queue
) -> Tensor {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch_size = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Calculate output dimensions
    int64_t out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int64_t out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    // Create output tensor
    Tensor output({batch_size, out_channels, out_height, out_width}, DType::Float32, input.device());

    int64_t total_elements = batch_size * out_channels * out_height * out_width;
    int64_t channels_per_group = in_channels / groups;
    int64_t out_channels_per_group = out_channels / groups;

    if (input.dtype() == DType::Int8 && weight.dtype() == DType::Int8) {
        const int8_t* in_ptr = get_data_ptr<const int8_t>(input);
        const int8_t* weight_ptr = get_data_ptr<const int8_t>(weight);
        const float* bias_ptr = bias ? get_data_ptr<const float>(*bias) : nullptr;
        float* out_ptr = get_data_ptr<float>(output);

        const bool has_bias = (bias != nullptr);
        const float combined_scale = input_scale * weight_scale;

        queue.parallel_for<QuantizedConv2dKernelInt8>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t ow = idx % out_width;
                int64_t oh = (idx / out_width) % out_height;
                int64_t oc = (idx / out_width / out_height) % out_channels;
                int64_t n = idx / out_width / out_height / out_channels;

                int64_t g = oc / out_channels_per_group;
                int64_t oc_g = oc % out_channels_per_group;

                // Accumulate in int32
                int32_t acc = 0;

                for (int64_t ic_g = 0; ic_g < channels_per_group; ++ic_g) {
                    int64_t ic = g * channels_per_group + ic_g;

                    for (int64_t kh = 0; kh < kernel_h; ++kh) {
                        for (int64_t kw = 0; kw < kernel_w; ++kw) {
                            int64_t ih = oh * stride - padding + kh * dilation;
                            int64_t iw = ow * stride - padding + kw * dilation;

                            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                int64_t in_idx = n * in_channels * in_height * in_width +
                                                ic * in_height * in_width +
                                                ih * in_width + iw;
                                int64_t weight_idx = oc * (in_channels / groups) * kernel_h * kernel_w +
                                                    ic_g * kernel_h * kernel_w +
                                                    kh * kernel_w + kw;

                                int32_t input_val = static_cast<int32_t>(in_ptr[in_idx]) - input_zero_point;
                                int32_t weight_val = static_cast<int32_t>(weight_ptr[weight_idx]) - weight_zero_point;
                                acc += input_val * weight_val;
                            }
                        }
                    }
                }

                // Dequantize
                float result = static_cast<float>(acc) * combined_scale;

                // Add bias
                if (has_bias) {
                    result += bias_ptr[oc];
                }

                out_ptr[idx] = result;
            }
        ).wait();
    }
    else {
        throw std::runtime_error("quantized_conv2d: requires Int8 input and weight");
    }

    return output;
}

} // namespace adaptivecpp
} // namespace tenzor
