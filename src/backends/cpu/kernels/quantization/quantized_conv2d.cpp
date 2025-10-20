/**
 * @file quantized_conv2d.cpp
 * @brief CPU kernels for quantized convolution operations
 */

#include <cstdint>
#include <algorithm>

namespace tenzor {
namespace nn {
namespace quantization {
namespace kernels {

/**
 * @brief Quantized 2D convolution (CPU).
 *
 * Performs INT8 convolution with im2col + gemm approach.
 */
auto quantized_conv2d_kernel(
    const int8_t* input,
    const int8_t* weight,
    const float* bias,
    float* output,
    int64_t batch,
    int64_t in_channels,
    int64_t out_channels,
    int64_t h_in,
    int64_t w_in,
    int64_t h_out,
    int64_t w_out,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    float input_scale,
    float weight_scale,
    int32_t input_zp,
    int32_t weight_zp
) -> void {
    float combined_scale = input_scale * weight_scale;

    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t oc = 0; oc < out_channels; ++oc) {
            for (int64_t oh = 0; oh < h_out; ++oh) {
                for (int64_t ow = 0; ow < w_out; ++ow) {
                    int32_t acc = 0;

                    // Convolution over input channels and kernel
                    for (int64_t ic = 0; ic < in_channels; ++ic) {
                        for (int64_t kh = 0; kh < kernel_size; ++kh) {
                            for (int64_t kw = 0; kw < kernel_size; ++kw) {
                                int64_t ih = oh * stride + kh - padding;
                                int64_t iw = ow * stride + kw - padding;

                                // Check bounds
                                if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                                    int64_t input_idx = ((b * in_channels + ic) * h_in + ih) * w_in + iw;
                                    int64_t weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;

                                    int32_t input_val = static_cast<int32_t>(input[input_idx]);
                                    int32_t weight_val = static_cast<int32_t>(weight[weight_idx]);

                                    acc += input_val * weight_val;
                                }
                            }
                        }
                    }

                    // Zero point correction
                    int64_t kernel_elements = in_channels * kernel_size * kernel_size;
                    acc -= input_zp * weight_zp * kernel_elements;

                    // Dequantize and add bias
                    float result = static_cast<float>(acc) * combined_scale;
                    if (bias != nullptr) {
                        result += bias[oc];
                    }

                    int64_t output_idx = ((b * out_channels + oc) * h_out + oh) * w_out + ow;
                    output[output_idx] = result;
                }
            }
        }
    }
}

} // namespace kernels
} // namespace quantization
} // namespace nn
} // namespace tenzor
