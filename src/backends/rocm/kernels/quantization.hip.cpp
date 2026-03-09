#ifdef TENZOR_ROCM_AVAILABLE

#include <hip/hip_runtime.h>
#include "tenzor/core/tensor.hpp"
#include <stdexcept>
#include <cstdint>

namespace tenzor {
namespace rocm {

// Error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            throw std::runtime_error( \
                std::string("HIP error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + hipGetErrorString(error) \
            ); \
        } \
    } while(0)

// ==============================================================================
// Quantized Linear HIP Kernel (INT8 inputs, INT32 accumulation)
// ==============================================================================

__global__ void quantized_linear_kernel_hip(
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    float combined_scale,
    int32_t input_zp,
    int32_t weight_zp
) {
    // 2D grid: blockIdx.y = batch, blockIdx.x * blockDim.x + threadIdx.x = output feature
    int64_t b = blockIdx.y;
    int64_t o = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || o >= out_features) return;

    const int8_t* input_row = input + b * in_features;
    const int8_t* weight_row = weight + o * in_features;

    int32_t acc = 0;
    int32_t sum_x = 0;
    int32_t sum_w = 0;

    // Vectorized loading: process 16 int8 values at a time via int4 (16 bytes)
    constexpr int VEC_SIZE = 16;
    int64_t vec_steps = in_features / VEC_SIZE;

    for (int64_t v = 0; v < vec_steps; ++v) {
        int4 input_vec = reinterpret_cast<const int4*>(input_row)[v];
        int4 weight_vec = reinterpret_cast<const int4*>(weight_row)[v];

        const int8_t* input_bytes = reinterpret_cast<const int8_t*>(&input_vec);
        const int8_t* weight_bytes = reinterpret_cast<const int8_t*>(&weight_vec);

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            acc += static_cast<int32_t>(input_bytes[i]) * static_cast<int32_t>(weight_bytes[i]);
            sum_x += static_cast<int32_t>(input_bytes[i]);
            sum_w += static_cast<int32_t>(weight_bytes[i]);
        }
    }

    // Remainder elements
    for (int64_t i = vec_steps * VEC_SIZE; i < in_features; ++i) {
        acc += static_cast<int32_t>(input_row[i]) * static_cast<int32_t>(weight_row[i]);
        sum_x += static_cast<int32_t>(input_row[i]);
        sum_w += static_cast<int32_t>(weight_row[i]);
    }

    // Zero point correction:
    // Full expansion: sum((x_i - x_zp) * (w_j - w_zp))
    //   = sum(x_i * w_j) - x_zp * sum(w_j) - w_zp * sum(x_i) + x_zp * w_zp * K
    acc = acc - input_zp * sum_w - weight_zp * sum_x
          + input_zp * weight_zp * static_cast<int32_t>(in_features);

    // Dequantize to float and add bias
    float result = static_cast<float>(acc) * combined_scale;
    if (bias != nullptr) {
        result += bias[o];
    }

    output[b * out_features + o] = result;
}

/**
 * @brief Host wrapper for quantized linear (INT8 → INT32 accumulation → Float32 output).
 */
auto quantized_linear_hip(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    float input_scale,
    int32_t input_zero_point,
    float weight_scale,
    int32_t weight_zero_point,
    float output_scale,
    int32_t output_zero_point,
    hipStream_t stream
) -> Tensor {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    int64_t batch_size = input_shape[0];
    int64_t in_features = input_shape[1];
    int64_t out_features = weight_shape[0];

    Tensor output({batch_size, out_features}, DType::Float32, input.device());

    float combined_scale = input_scale * weight_scale / output_scale;

    const int THREADS = 256;
    dim3 blocks((out_features + THREADS - 1) / THREADS, batch_size);
    dim3 threads(THREADS);

    hipLaunchKernelGGL(quantized_linear_kernel_hip,
        blocks, threads, 0, stream,
        input.data<int8_t>(),
        weight.data<int8_t>(),
        bias ? bias->data<float>() : nullptr,
        output.data<float>(),
        batch_size, in_features, out_features,
        combined_scale, input_zero_point, weight_zero_point);

    HIP_CHECK(hipGetLastError());
    return output;
}

// ==============================================================================
// Quantized Conv2d HIP Kernel (INT8 inputs, INT32 accumulation)
// ==============================================================================

__global__ void quantized_conv2d_kernel_hip(
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
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
    int64_t dilation,
    float combined_scale,
    int32_t input_zp,
    int32_t weight_zp
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch * out_channels * h_out * w_out;
    if (idx >= total) return;

    // Decompose linear index into (n, oc, oh, ow)
    int64_t temp = idx;
    int64_t ow = temp % w_out; temp /= w_out;
    int64_t oh = temp % h_out; temp /= h_out;
    int64_t oc = temp % out_channels;
    int64_t n = temp / out_channels;

    int32_t acc = 0;
    int32_t zp_correction = 0;

    for (int64_t ic = 0; ic < in_channels; ++ic) {
        for (int64_t kh = 0; kh < kernel_size; ++kh) {
            for (int64_t kw = 0; kw < kernel_size; ++kw) {
                int64_t ih = oh * stride - padding + kh * dilation;
                int64_t iw = ow * stride - padding + kw * dilation;

                if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                    int64_t input_idx = ((n * in_channels + ic) * h_in + ih) * w_in + iw;
                    int64_t weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;

                    int32_t iv = static_cast<int32_t>(input[input_idx]);
                    int32_t wv = static_cast<int32_t>(weight[weight_idx]);
                    acc += iv * wv;

                    // Track sums for zero-point correction
                    zp_correction += iv * weight_zp + wv * input_zp;
                } else {
                    // Padding pixels: input value is 0, but zero-point correction still applies
                    // (0 - input_zp) * (w - weight_zp) contributes -input_zp * (w - weight_zp)
                    // Simplified: we just accumulate input_zp * weight_zp for padded positions
                    zp_correction += input_zp * weight_zp;
                }
            }
        }
    }

    // Apply zero-point correction
    int64_t K = in_channels * kernel_size * kernel_size;
    acc = acc - zp_correction + input_zp * weight_zp * static_cast<int32_t>(K);

    // Dequantize to float and add bias
    float result = static_cast<float>(acc) * combined_scale;
    if (bias != nullptr) {
        result += bias[oc];
    }

    output[idx] = result;
}

/**
 * @brief Host wrapper for quantized conv2d (INT8 → INT32 accumulation → Float32 output).
 */
auto quantized_conv2d_hip(
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
    float output_scale,
    int32_t output_zero_point,
    hipStream_t stream
) -> Tensor {
    auto input_shape = input.shape();
    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t h_in = input_shape[2];
    int64_t w_in = input_shape[3];

    auto weight_shape = weight.shape();
    int64_t out_channels = weight_shape[0];
    int64_t kernel_size = weight_shape[2];

    int64_t h_out = (h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t w_out = (w_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    Tensor output({batch, out_channels, h_out, w_out}, DType::Float32, input.device());

    float combined_scale = input_scale * weight_scale / output_scale;

    int64_t total = batch * out_channels * h_out * w_out;
    const int THREADS = 256;
    int grid_size = static_cast<int>((total + THREADS - 1) / THREADS);

    hipLaunchKernelGGL(quantized_conv2d_kernel_hip,
        dim3(grid_size), dim3(THREADS), 0, stream,
        input.data<int8_t>(),
        weight.data<int8_t>(),
        bias ? bias->data<float>() : nullptr,
        output.data<float>(),
        batch, in_channels, out_channels,
        h_in, w_in, h_out, w_out,
        kernel_size, stride, padding, dilation,
        combined_scale, input_zero_point, weight_zero_point);

    HIP_CHECK(hipGetLastError());
    return output;
}

} // namespace rocm
} // namespace tenzor

#endif // TENZOR_ROCM_AVAILABLE
