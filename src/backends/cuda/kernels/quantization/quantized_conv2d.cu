/**
 * @file quantized_conv2d.cu
 * @brief CUDA kernels for quantized convolution operations
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "../cuda_common.cuh"

namespace tenzor {
namespace nn {
namespace quantization {
namespace kernels {

/**
 * @brief CUDA kernel for quantized 2D convolution.
 *
 * Each thread computes one output element using direct convolution.
 */
__global__ void quantized_conv2d_cuda_kernel(
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
    float combined_scale,
    int32_t input_zp,
    int32_t weight_zp
) {
    // Compute output position
    int64_t ow = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t oh = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t oc = blockIdx.z % out_channels;
    int64_t b = blockIdx.z / out_channels;

    if (b >= batch || oc >= out_channels || oh >= h_out || ow >= w_out) return;

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

/**
 * @brief Host function to launch quantized conv2d CUDA kernel.
 */
auto quantized_conv2d_cuda(
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
    int32_t weight_zp,
    cudaStream_t stream
) -> void {
    float combined_scale = input_scale * weight_scale;

    // Launch configuration
    dim3 threads(16, 16);  // 16x16 thread block
    dim3 blocks(
        (w_out + threads.x - 1) / threads.x,
        (h_out + threads.y - 1) / threads.y,
        batch * out_channels
    );

    quantized_conv2d_cuda_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch, in_channels, out_channels,
        h_in, w_in, h_out, w_out,
        kernel_size, stride, padding,
        combined_scale, input_zp, weight_zp
    );
    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

/**
 * @brief Optimized implicit GEMM convolution using shared memory.
 *
 * Uses implicit GEMM formulation with shared memory tiling for better
 * memory bandwidth utilization.
 */
__global__ void quantized_conv2d_implicit_gemm_kernel(
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
    float combined_scale,
    int32_t input_zp,
    int32_t weight_zp
) {
    // Shared memory for tiling
    __shared__ int8_t shared_input[32][32];
    __shared__ int8_t shared_weight[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int64_t ow = blockIdx.x * blockDim.x + tx;
    int64_t oh = blockIdx.y * blockDim.y + ty;
    int64_t oc = blockIdx.z % out_channels;
    int64_t b = blockIdx.z / out_channels;

    if (b >= batch || oc >= out_channels || oh >= h_out || ow >= w_out) return;

    int32_t acc = 0;

    // Tile over input channels and kernel
    const int TILE_SIZE = 32;
    int64_t total_k = in_channels * kernel_size * kernel_size;

    for (int64_t k_base = 0; k_base < total_k; k_base += TILE_SIZE) {
        // Load input tile into shared memory
        int64_t k = k_base + tx;
        if (k < total_k) {
            int64_t ic = k / (kernel_size * kernel_size);
            int64_t k_spatial = k % (kernel_size * kernel_size);
            int64_t kh = k_spatial / kernel_size;
            int64_t kw = k_spatial % kernel_size;

            int64_t ih = oh * stride + kh - padding;
            int64_t iw = ow * stride + kw - padding;

            if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                int64_t input_idx = ((b * in_channels + ic) * h_in + ih) * w_in + iw;
                shared_input[ty][tx] = input[input_idx];
            } else {
                shared_input[ty][tx] = 0;  // Padding
            }

            // Load weight tile
            int64_t weight_idx = oc * total_k + k;
            if (weight_idx < out_channels * total_k) {
                shared_weight[ty][tx] = weight[weight_idx];
            }
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int i = 0; i < TILE_SIZE && k_base + i < total_k; ++i) {
            acc += static_cast<int32_t>(shared_input[ty][i]) *
                   static_cast<int32_t>(shared_weight[ty][i]);
        }

        __syncthreads();
    }

    // Zero point correction
    acc -= input_zp * weight_zp * total_k;

    // Dequantize and add bias
    float result = static_cast<float>(acc) * combined_scale;
    if (bias != nullptr) {
        result += bias[oc];
    }

    int64_t output_idx = ((b * out_channels + oc) * h_out + oh) * w_out + ow;
    output[output_idx] = result;
}

} // namespace kernels
} // namespace quantization
} // namespace nn
} // namespace tenzor
