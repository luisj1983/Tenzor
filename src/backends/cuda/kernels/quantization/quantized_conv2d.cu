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
    int64_t kh_size,
    int64_t kw_size,
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    int64_t dH, int64_t dW,
    int64_t groups,
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

    // int64 accumulator: each int8*int8 product fits in int32, but summed over
    // in_channels*kh*kw the running total can exceed INT32_MAX for very wide
    // layers, so accumulate in int64 to avoid silent wraparound.
    int64_t acc = 0;
    // Running sums of the quantized operands for the asymmetric zero-point
    // correction below.
    int64_t sum_i = 0;  // sum(q_i) over the receptive field (incl. padding)
    int64_t sum_w = 0;  // sum(q_w) over the weight slice

    // Grouped convolution: each output channel reads only its group's input
    // channels. Weight is laid out [out_channels, in_channels/groups, kh, kw].
    const int64_t in_ch_per_group  = in_channels / groups;
    const int64_t out_ch_per_group = out_channels / groups;
    const int64_t group            = oc / out_ch_per_group;
    const int64_t ic_start         = group * in_ch_per_group;

    // Convolution over this group's input channels and kernel.
    // Real-zero padding maps to the quantized value input_zp (since
    // x_real = scale*(q - input_zp) and x_real == 0 <=> q == input_zp); this
    // matches the CPU reference so padded taps contribute zero post-correction.
    for (int64_t icg = 0; icg < in_ch_per_group; ++icg) {
        int64_t ic = ic_start + icg;  // absolute input channel
        for (int64_t kh = 0; kh < kh_size; ++kh) {
            for (int64_t kw = 0; kw < kw_size; ++kw) {
                int64_t ih = oh * sH + kh * dH - pH;
                int64_t iw = ow * sW + kw * dW - pW;

                // Weight uses the group-local input-channel index (icg).
                int64_t weight_idx = ((oc * in_ch_per_group + icg) * kh_size + kh) * kw_size + kw;
                int32_t weight_val = static_cast<int32_t>(weight[weight_idx]);
                sum_w += weight_val;

                int32_t input_val;
                if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                    int64_t input_idx = ((b * in_channels + ic) * h_in + ih) * w_in + iw;
                    input_val = static_cast<int32_t>(input[input_idx]);
                } else {
                    input_val = input_zp;  // real-zero padding
                }

                acc += input_val * weight_val;
                sum_i += input_val;
            }
        }
    }

    // Asymmetric zero-point correction (int64 to avoid overflow):
    //   sum(q_i*q_w) - zp_w*sum(q_i) - zp_i*sum(q_w) + N*zp_i*zp_w
    int64_t kernel_elements = in_ch_per_group * kh_size * kw_size;
    int64_t corrected = static_cast<int64_t>(acc)
                      - static_cast<int64_t>(weight_zp) * sum_i
                      - static_cast<int64_t>(input_zp) * sum_w
                      + static_cast<int64_t>(input_zp) * static_cast<int64_t>(weight_zp) * kernel_elements;

    // Dequantize and add bias
    float result = static_cast<float>(corrected) * combined_scale;
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
    int64_t kh_size, int64_t kw_size,
    int64_t sH, int64_t sW,
    int64_t pH, int64_t pW,
    int64_t dH, int64_t dW,
    int64_t groups,
    float input_scale,
    float weight_scale,
    int32_t input_zp,
    int32_t weight_zp,
    cudaStream_t stream
) -> void {
    if (groups < 1) {
        throw std::runtime_error("quantized_conv2d: groups must be >= 1");
    }
    if (in_channels % groups != 0 || out_channels % groups != 0) {
        throw std::runtime_error(
            "quantized_conv2d: in_channels and out_channels must be divisible by groups");
    }

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
        kh_size, kw_size, sH, sW, pH, pW, dH, dW,
        groups, combined_scale, input_zp, weight_zp
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
