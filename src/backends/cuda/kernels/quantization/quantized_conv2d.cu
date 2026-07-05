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
    int32_t weight_zp,
    // Per-channel (F045): when weight_scales != nullptr each output channel oc
    // uses weight_scales[oc] and (weight_zps ? weight_zps[oc] : 0); the dequant
    // scale is input_scale_pc * weight_scales[oc]. nullptr => per-tensor.
    const float* __restrict__ weight_scales,
    const int32_t* __restrict__ weight_zps,
    float input_scale_pc
) {
    // Compute output position
    int64_t ow = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t oh = blockIdx.y * blockDim.y + threadIdx.y;

    if (oh >= h_out || ow >= w_out) return;

    // The (batch, out_channel) pair is folded onto the z-axis. CUDA's gridDim.z
    // hardware maximum is 65535, but batch*out_channels can far exceed that for
    // realistic models (e.g. batch=64, out_channels=2048 -> 131072). To stay
    // within the limit the host caps gridDim.z at 65535 and we grid-stride over
    // the flattened (b,oc) index here, so all (b,oc) pairs are still covered.
    const int64_t bc_total = batch * out_channels;
    const int64_t bc_stride = static_cast<int64_t>(gridDim.z);
    for (int64_t bc = blockIdx.z; bc < bc_total; bc += bc_stride) {
    int64_t oc = bc % out_channels;
    int64_t b = bc / out_channels;

    // Per-channel weight scale / zero-point (each output channel oc).
    const int32_t w_zp = weight_scales ? (weight_zps ? weight_zps[oc] : 0) : weight_zp;
    const float out_scale = weight_scales ? (input_scale_pc * weight_scales[oc])
                                          : combined_scale;

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
                      - static_cast<int64_t>(w_zp) * sum_i
                      - static_cast<int64_t>(input_zp) * sum_w
                      + static_cast<int64_t>(input_zp) * static_cast<int64_t>(w_zp) * kernel_elements;

    // Dequantize and add bias
    float result = static_cast<float>(corrected) * out_scale;
    if (bias != nullptr) {
        result += bias[oc];
    }

    int64_t output_idx = ((b * out_channels + oc) * h_out + oh) * w_out + ow;
    output[output_idx] = result;
    } // grid-stride over flattened (b, oc)
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
    cudaStream_t stream,
    // Per-channel (F045): [out_channels] device arrays; nullptr => per-tensor.
    const float* weight_scales,
    const int32_t* weight_zps
) -> void {
    if (groups < 1) {
        throw std::runtime_error("quantized_conv2d: groups must be >= 1");
    }
    if (in_channels % groups != 0 || out_channels % groups != 0) {
        throw std::runtime_error(
            "quantized_conv2d: in_channels and out_channels must be divisible by groups");
    }

    float combined_scale = input_scale * weight_scale;

    // Launch configuration.
    //
    // The (batch, out_channel) pair is folded onto the z-axis, but CUDA's
    // gridDim.z hardware maximum is 65535 on all compute capabilities. For
    // realistic quantized models batch*out_channels easily exceeds that
    // (e.g. batch=64, out_channels=2048 -> 131072), which would make the launch
    // fail with cudaErrorInvalidConfiguration. Cap gridDim.z at 65535; the
    // kernel grid-strides over the flattened (b,oc) index to cover the rest.
    constexpr int64_t kMaxGridZ = 65535;
    dim3 threads(16, 16);  // 16x16 thread block
    int64_t bc_total = batch * out_channels;
    int64_t grid_z = bc_total < kMaxGridZ ? bc_total : kMaxGridZ;
    dim3 blocks(
        static_cast<unsigned int>((w_out + threads.x - 1) / threads.x),
        static_cast<unsigned int>((h_out + threads.y - 1) / threads.y),
        static_cast<unsigned int>(grid_z)
    );

    quantized_conv2d_cuda_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch, in_channels, out_channels,
        h_in, w_in, h_out, w_out,
        kh_size, kw_size, sH, sW, pH, pW, dH, dW,
        groups, combined_scale, input_zp, weight_zp,
        weight_scales, weight_zps, input_scale
    );
    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

// NOTE: An unused quantized_conv2d_implicit_gemm_kernel was removed here. It was
// dead code (never registered — the live kernel is quantized_conv2d_cuda at
// cuda_kernel_registry.cpp) and functionally wrong: it used a 32-bit int32_t
// accumulator (overflow vs the int64 live kernel), an incomplete zero-point
// correction (only -input_zp*weight_zp*total_k, omitting the -weight_zp*sum_i
// and -input_zp*sum_w cross terms), had no groups/dilation support, and placed
// its early-return guard before the shared-memory load + __syncthreads (divergent
// __syncthreads is UB and survivors could read stale tiles). It would have
// shipped wrong results if wired up. Reintroduce only with the live kernel's
// int64 accumulator, full asymmetric zero-point correction, group/dilation
// support, non-divergent __syncthreads, and a parity test.

} // namespace kernels
} // namespace quantization
} // namespace nn
} // namespace tenzor
