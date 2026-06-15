/**
 * @file conv3d.cu
 * @brief Direct-convolution CUDA fallback for Conv3d / ConvTranspose3d.
 *
 * Used when TENZOR_HAS_CUDNN is not defined. Implements a gather-based
 * direct convolution (one thread per output element). Performance is
 * lower than cuDNN's Winograd / FFT paths, but the kernel is correct
 * for arbitrary stride / padding / dilation / groups across all four
 * supported floating-point dtypes (Float32, Float64, Float16, BFloat16).
 *
 * Half / bfloat16 inputs accumulate in float for numerical stability,
 * matching cuDNN's mixed-precision convention.
 */

#include <array>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "cuda_common.cuh"
#include "cuda_launch_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

namespace tenzor {
namespace cuda {

// ============================================================================
// Output-size helpers
// ============================================================================

__host__ __device__ inline int64_t conv3d_out_dim(
    int64_t in, int64_t k, int64_t stride, int64_t padding, int64_t dilation) {
    return (in + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
}

__host__ __device__ inline int64_t conv_transpose3d_out_dim(
    int64_t in, int64_t k, int64_t stride, int64_t padding,
    int64_t output_padding, int64_t dilation) {
    return (in - 1) * stride - 2 * padding + dilation * (k - 1) + output_padding + 1;
}

// ============================================================================
// dtype helpers — load/store with float accumulation for half/bfloat16
// ============================================================================

template<typename T>
__device__ __forceinline__ float load_as_float(const T& v) {
    if constexpr (std::is_same_v<T, __half>) {
        return __half2float(v);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(v);
    } else {
        return static_cast<float>(v);
    }
}

template<typename T>
__device__ __forceinline__ T store_from_acc(float v) {
    if constexpr (std::is_same_v<T, __half>) {
        return float2half_sat(v);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(v);
    } else {
        return static_cast<T>(v);
    }
}

template<typename T>
__device__ __forceinline__ T store_from_double(double v) {
    return static_cast<T>(v);
}

// ============================================================================
// Conv3d forward — one thread per output element
// ============================================================================

template<typename T, typename Acc>
__global__ void conv3d_forward_direct_kernel(
    const T* __restrict__ input,    // [N, Cin, D, H, W]
    const T* __restrict__ weight,   // [Cout, Cin/g, kD, kH, kW]
    const T* __restrict__ bias,     // [Cout] or null
    T* __restrict__ output,         // [N, Cout, oD, oH, oW]
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Cin_per_g,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t oD, int64_t oH, int64_t oW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t dD, int64_t dH, int64_t dW, int64_t groups,
    int64_t total)
{
    const int64_t Cout_per_g = Cout / groups;

    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % oW;
        int64_t t1 = idx / oW;
        int64_t oh = t1 % oH;
        int64_t t2 = t1 / oH;
        int64_t od = t2 % oD;
        int64_t t3 = t2 / oD;
        int64_t oc = t3 % Cout;
        int64_t n  = t3 / Cout;

        int64_t g = oc / Cout_per_g;
        int64_t in_c_start = g * Cin_per_g;

        Acc sum = static_cast<Acc>(0);

        for (int64_t kc = 0; kc < Cin_per_g; ++kc) {
            int64_t in_c = in_c_start + kc;
            for (int64_t kd = 0; kd < kD; ++kd) {
                int64_t id = od * sD - pD + kd * dD;
                if (id < 0 || id >= D) continue;
                for (int64_t kh = 0; kh < kH; ++kh) {
                    int64_t ih = oh * sH - pH + kh * dH;
                    if (ih < 0 || ih >= H) continue;
                    for (int64_t kwi = 0; kwi < kW; ++kwi) {
                        int64_t iw = ow * sW - pW + kwi * dW;
                        if (iw < 0 || iw >= W) continue;

                        int64_t in_idx =
                            ((((n * Cin + in_c) * D + id) * H + ih) * W) + iw;
                        int64_t w_idx =
                            ((((oc * Cin_per_g + kc) * kD + kd) * kH + kh) * kW) + kwi;

                        if constexpr (std::is_same_v<T, double>) {
                            sum += static_cast<double>(input[in_idx]) *
                                   static_cast<double>(weight[w_idx]);
                        } else {
                            sum += load_as_float(input[in_idx]) *
                                   load_as_float(weight[w_idx]);
                        }
                    }
                }
            }
        }

        if (bias != nullptr) {
            if constexpr (std::is_same_v<T, double>) {
                sum += static_cast<double>(bias[oc]);
            } else {
                sum += load_as_float(bias[oc]);
            }
        }

        if constexpr (std::is_same_v<T, double>) {
            output[idx] = store_from_double<T>(sum);
        } else {
            output[idx] = store_from_acc<T>(sum);
        }
    }
}

// ============================================================================
// Conv3d backward_input — gather: one thread per grad_input element
// ============================================================================

template<typename T, typename Acc>
__global__ void conv3d_backward_input_kernel(
    const T* __restrict__ grad_output,  // [N, Cout, oD, oH, oW]
    const T* __restrict__ weight,       // [Cout, Cin/g, kD, kH, kW]
    T* __restrict__ grad_input,         // [N, Cin, D, H, W]
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Cin_per_g,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t oD, int64_t oH, int64_t oW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t dD, int64_t dH, int64_t dW, int64_t groups,
    int64_t total)
{
    const int64_t Cout_per_g = Cout / groups;

    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t iw = idx % W;
        int64_t t1 = idx / W;
        int64_t ih = t1 % H;
        int64_t t2 = t1 / H;
        int64_t id = t2 % D;
        int64_t t3 = t2 / D;
        int64_t ic = t3 % Cin;
        int64_t n  = t3 / Cin;

        int64_t g = ic / Cin_per_g;
        int64_t kc = ic - g * Cin_per_g;
        int64_t oc_start = g * Cout_per_g;

        Acc sum = static_cast<Acc>(0);

        for (int64_t kd = 0; kd < kD; ++kd) {
            int64_t od_num = id + pD - kd * dD;
            if (od_num < 0 || (od_num % sD) != 0) continue;
            int64_t od = od_num / sD;
            if (od < 0 || od >= oD) continue;

            for (int64_t kh = 0; kh < kH; ++kh) {
                int64_t oh_num = ih + pH - kh * dH;
                if (oh_num < 0 || (oh_num % sH) != 0) continue;
                int64_t oh = oh_num / sH;
                if (oh < 0 || oh >= oH) continue;

                for (int64_t kwi = 0; kwi < kW; ++kwi) {
                    int64_t ow_num = iw + pW - kwi * dW;
                    if (ow_num < 0 || (ow_num % sW) != 0) continue;
                    int64_t ow = ow_num / sW;
                    if (ow < 0 || ow >= oW) continue;

                    for (int64_t oc_off = 0; oc_off < Cout_per_g; ++oc_off) {
                        int64_t oc = oc_start + oc_off;
                        int64_t go_idx =
                            ((((n * Cout + oc) * oD + od) * oH + oh) * oW) + ow;
                        int64_t w_idx =
                            ((((oc * Cin_per_g + kc) * kD + kd) * kH + kh) * kW) + kwi;

                        if constexpr (std::is_same_v<T, double>) {
                            sum += static_cast<double>(grad_output[go_idx]) *
                                   static_cast<double>(weight[w_idx]);
                        } else {
                            sum += load_as_float(grad_output[go_idx]) *
                                   load_as_float(weight[w_idx]);
                        }
                    }
                }
            }
        }

        if constexpr (std::is_same_v<T, double>) {
            grad_input[idx] = store_from_double<T>(sum);
        } else {
            grad_input[idx] = store_from_acc<T>(sum);
        }
    }
}

// ============================================================================
// Conv3d backward_weight — gather: one thread per grad_weight element
// ============================================================================

template<typename T, typename Acc>
__global__ void conv3d_backward_weight_kernel(
    const T* __restrict__ grad_output,  // [N, Cout, oD, oH, oW]
    const T* __restrict__ input,        // [N, Cin, D, H, W]
    T* __restrict__ grad_weight,        // [Cout, Cin/g, kD, kH, kW]
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Cin_per_g,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t oD, int64_t oH, int64_t oW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t dD, int64_t dH, int64_t dW, int64_t groups,
    int64_t total)
{
    const int64_t Cout_per_g = Cout / groups;

    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t kwi = idx % kW;
        int64_t t1  = idx / kW;
        int64_t kh  = t1 % kH;
        int64_t t2  = t1 / kH;
        int64_t kd  = t2 % kD;
        int64_t t3  = t2 / kD;
        int64_t kc  = t3 % Cin_per_g;
        int64_t oc  = t3 / Cin_per_g;

        int64_t g = oc / Cout_per_g;
        int64_t ic = g * Cin_per_g + kc;

        Acc sum = static_cast<Acc>(0);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t od = 0; od < oD; ++od) {
                int64_t id = od * sD - pD + kd * dD;
                if (id < 0 || id >= D) continue;
                for (int64_t oh = 0; oh < oH; ++oh) {
                    int64_t ih = oh * sH - pH + kh * dH;
                    if (ih < 0 || ih >= H) continue;
                    for (int64_t ow = 0; ow < oW; ++ow) {
                        int64_t iw = ow * sW - pW + kwi * dW;
                        if (iw < 0 || iw >= W) continue;

                        int64_t go_idx =
                            ((((n * Cout + oc) * oD + od) * oH + oh) * oW) + ow;
                        int64_t in_idx =
                            ((((n * Cin + ic) * D + id) * H + ih) * W) + iw;

                        if constexpr (std::is_same_v<T, double>) {
                            sum += static_cast<double>(grad_output[go_idx]) *
                                   static_cast<double>(input[in_idx]);
                        } else {
                            sum += load_as_float(grad_output[go_idx]) *
                                   load_as_float(input[in_idx]);
                        }
                    }
                }
            }
        }

        if constexpr (std::is_same_v<T, double>) {
            grad_weight[idx] = store_from_double<T>(sum);
        } else {
            grad_weight[idx] = store_from_acc<T>(sum);
        }
    }
}

// ============================================================================
// Conv3d backward_bias — sum grad_output over (N, oD, oH, oW) per Cout channel
// ============================================================================

template<typename T, typename Acc>
__global__ void conv3d_backward_bias_kernel(
    const T* __restrict__ grad_output,  // [N, Cout, oD, oH, oW]
    T* __restrict__ grad_bias,          // [Cout]
    int64_t N, int64_t Cout,
    int64_t oD, int64_t oH, int64_t oW)
{
    int64_t oc = blockIdx.x;
    if (oc >= Cout) return;

    int64_t spatial = N * oD * oH * oW;
    Acc sum = static_cast<Acc>(0);

    for (int64_t i = threadIdx.x; i < spatial; i += blockDim.x) {
        int64_t n = i / (oD * oH * oW);
        int64_t s = i % (oD * oH * oW);
        int64_t go_idx = ((n * Cout + oc) * oD * oH * oW) + s;

        if constexpr (std::is_same_v<T, double>) {
            sum += static_cast<double>(grad_output[go_idx]);
        } else {
            sum += load_as_float(grad_output[go_idx]);
        }
    }

    // Block reduction in shared memory
    __shared__ Acc smem[1024];
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if constexpr (std::is_same_v<T, double>) {
            grad_bias[oc] = store_from_double<T>(smem[0]);
        } else {
            grad_bias[oc] = store_from_acc<T>(smem[0]);
        }
    }
}

// ============================================================================
// ConvTranspose3d forward — gather: one thread per output element
//
// ConvTranspose forward is the same gather pattern as Conv backward_input,
// just with weight reinterpreted as [Cin, Cout/g, kD, kH, kW] and the output
// dimensions expanded accordingly.
// ============================================================================

template<typename T, typename Acc>
__global__ void conv_transpose3d_forward_kernel_impl(
    const T* __restrict__ input,    // [N, Cin, D, H, W]
    const T* __restrict__ weight,   // [Cin, Cout/g, kD, kH, kW]
    const T* __restrict__ bias,     // [Cout] or null
    T* __restrict__ output,         // [N, Cout, oD, oH, oW]
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Cout_per_g,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t oD, int64_t oH, int64_t oW,
    int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
    int64_t total)
{
    const int64_t Cin_per_g = Cin / groups;

    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % oW;
        int64_t t1 = idx / oW;
        int64_t oh = t1 % oH;
        int64_t t2 = t1 / oH;
        int64_t od = t2 % oD;
        int64_t t3 = t2 / oD;
        int64_t oc = t3 % Cout;
        int64_t n  = t3 / Cout;

        int64_t g = oc / Cout_per_g;
        int64_t oc_in_group = oc - g * Cout_per_g;
        int64_t in_c_start = g * Cin_per_g;

        Acc sum = static_cast<Acc>(0);

        for (int64_t kd = 0; kd < kD; ++kd) {
            int64_t id_num = od + padding - kd * dilation;
            if (id_num < 0 || (id_num % stride) != 0) continue;
            int64_t id = id_num / stride;
            if (id < 0 || id >= D) continue;

            for (int64_t kh = 0; kh < kH; ++kh) {
                int64_t ih_num = oh + padding - kh * dilation;
                if (ih_num < 0 || (ih_num % stride) != 0) continue;
                int64_t ih = ih_num / stride;
                if (ih < 0 || ih >= H) continue;

                for (int64_t kwi = 0; kwi < kW; ++kwi) {
                    int64_t iw_num = ow + padding - kwi * dilation;
                    if (iw_num < 0 || (iw_num % stride) != 0) continue;
                    int64_t iw = iw_num / stride;
                    if (iw < 0 || iw >= W) continue;

                    for (int64_t kc = 0; kc < Cin_per_g; ++kc) {
                        int64_t ic = in_c_start + kc;
                        int64_t in_idx =
                            ((((n * Cin + ic) * D + id) * H + ih) * W) + iw;
                        int64_t w_idx =
                            ((((ic * Cout_per_g + oc_in_group) * kD + kd) * kH + kh) * kW) + kwi;

                        if constexpr (std::is_same_v<T, double>) {
                            sum += static_cast<double>(input[in_idx]) *
                                   static_cast<double>(weight[w_idx]);
                        } else {
                            sum += load_as_float(input[in_idx]) *
                                   load_as_float(weight[w_idx]);
                        }
                    }
                }
            }
        }

        if (bias != nullptr) {
            if constexpr (std::is_same_v<T, double>) {
                sum += static_cast<double>(bias[oc]);
            } else {
                sum += load_as_float(bias[oc]);
            }
        }

        if constexpr (std::is_same_v<T, double>) {
            output[idx] = store_from_double<T>(sum);
        } else {
            output[idx] = store_from_acc<T>(sum);
        }
    }
}

// ============================================================================
// ConvTranspose3d backward_input
//
// grad_input has the same shape as input. Its gradient is the regular forward
// convolution of grad_output (shape [N, Cout, oD, oH, oW]) with the weight
// reinterpreted in transposed mode.
// ============================================================================

template<typename T, typename Acc>
__global__ void conv_transpose3d_backward_input_kernel_impl(
    const T* __restrict__ grad_output,  // [N, Cout, oD, oH, oW]
    const T* __restrict__ weight,       // [Cin, Cout/g, kD, kH, kW]
    T* __restrict__ grad_input,         // [N, Cin, D, H, W]
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Cout_per_g,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t oD, int64_t oH, int64_t oW,
    int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
    int64_t total)
{
    const int64_t Cin_per_g = Cin / groups;

    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t iw = idx % W;
        int64_t t1 = idx / W;
        int64_t ih = t1 % H;
        int64_t t2 = t1 / H;
        int64_t id = t2 % D;
        int64_t t3 = t2 / D;
        int64_t ic = t3 % Cin;
        int64_t n  = t3 / Cin;

        int64_t g = ic / Cin_per_g;
        int64_t oc_start = g * Cout_per_g;

        Acc sum = static_cast<Acc>(0);

        for (int64_t kd = 0; kd < kD; ++kd) {
            int64_t od = id * stride - padding + kd * dilation;
            if (od < 0 || od >= oD) continue;
            for (int64_t kh = 0; kh < kH; ++kh) {
                int64_t oh = ih * stride - padding + kh * dilation;
                if (oh < 0 || oh >= oH) continue;
                for (int64_t kwi = 0; kwi < kW; ++kwi) {
                    int64_t ow = iw * stride - padding + kwi * dilation;
                    if (ow < 0 || ow >= oW) continue;

                    for (int64_t oc_off = 0; oc_off < Cout_per_g; ++oc_off) {
                        int64_t oc = oc_start + oc_off;
                        int64_t go_idx =
                            ((((n * Cout + oc) * oD + od) * oH + oh) * oW) + ow;
                        int64_t w_idx =
                            ((((ic * Cout_per_g + oc_off) * kD + kd) * kH + kh) * kW) + kwi;

                        if constexpr (std::is_same_v<T, double>) {
                            sum += static_cast<double>(grad_output[go_idx]) *
                                   static_cast<double>(weight[w_idx]);
                        } else {
                            sum += load_as_float(grad_output[go_idx]) *
                                   load_as_float(weight[w_idx]);
                        }
                    }
                }
            }
        }

        // Note: this is a "backward_input" so we are inverting ConvTranspose's
        // forward; the gather direction is opposite. The condition must match
        // the forward: for ConvTranspose forward, output od = id*stride - pad + kd*dilation
        // — so for grad_input we sum over the (od, kd) pairs that produced this id.
        // The above loop already enforces od = id*stride - pad + kd*dilation by
        // iterating kd and computing od; if od is in range, the contribution
        // is valid (no divisibility check because we are projecting forward).

        if constexpr (std::is_same_v<T, double>) {
            grad_input[idx] = store_from_double<T>(sum);
        } else {
            grad_input[idx] = store_from_acc<T>(sum);
        }
    }
}

// ============================================================================
// ConvTranspose3d backward_weight
// ============================================================================

template<typename T, typename Acc>
__global__ void conv_transpose3d_backward_weight_kernel_impl(
    const T* __restrict__ grad_output,  // [N, Cout, oD, oH, oW]
    const T* __restrict__ input,        // [N, Cin, D, H, W]
    T* __restrict__ grad_weight,        // [Cin, Cout/g, kD, kH, kW]
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Cout_per_g,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t oD, int64_t oH, int64_t oW,
    int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
    int64_t total)
{
    const int64_t Cin_per_g = Cin / groups;

    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        // grad_weight layout: [Cin, Cout/g, kD, kH, kW]
        int64_t kwi      = idx % kW;
        int64_t t1       = idx / kW;
        int64_t kh       = t1 % kH;
        int64_t t2       = t1 / kH;
        int64_t kd       = t2 % kD;
        int64_t t3       = t2 / kD;
        int64_t oc_in_g  = t3 % Cout_per_g;
        int64_t ic       = t3 / Cout_per_g;

        int64_t g = ic / Cin_per_g;
        int64_t oc = g * Cout_per_g + oc_in_g;

        Acc sum = static_cast<Acc>(0);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t id = 0; id < D; ++id) {
                int64_t od = id * stride - padding + kd * dilation;
                if (od < 0 || od >= oD) continue;
                for (int64_t ih = 0; ih < H; ++ih) {
                    int64_t oh = ih * stride - padding + kh * dilation;
                    if (oh < 0 || oh >= oH) continue;
                    for (int64_t iw = 0; iw < W; ++iw) {
                        int64_t ow = iw * stride - padding + kwi * dilation;
                        if (ow < 0 || ow >= oW) continue;

                        int64_t in_idx =
                            ((((n * Cin + ic) * D + id) * H + ih) * W) + iw;
                        int64_t go_idx =
                            ((((n * Cout + oc) * oD + od) * oH + oh) * oW) + ow;

                        if constexpr (std::is_same_v<T, double>) {
                            sum += static_cast<double>(input[in_idx]) *
                                   static_cast<double>(grad_output[go_idx]);
                        } else {
                            sum += load_as_float(input[in_idx]) *
                                   load_as_float(grad_output[go_idx]);
                        }
                    }
                }
            }
        }

        if constexpr (std::is_same_v<T, double>) {
            grad_weight[idx] = store_from_double<T>(sum);
        } else {
            grad_weight[idx] = store_from_acc<T>(sum);
        }
    }
}

// ============================================================================
// Public dispatch helpers
// ============================================================================

namespace {

template<typename T, typename Acc>
void launch_conv3d_forward(
    const Tensor& input, const Tensor& weight, const Tensor* bias,
    Tensor& output,
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Cin_per_g,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t oD, int64_t oH, int64_t oW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t dD, int64_t dH, int64_t dW, int64_t groups,
    cudaStream_t stream)
{
    int64_t total = N * Cout * oD * oH * oW;
    dim3 grid, block;
    compute_launch_config_1d(total, grid, block);

    conv3d_forward_direct_kernel<T, Acc><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(input.data_ptr()),
        reinterpret_cast<const T*>(weight.data_ptr()),
        bias ? reinterpret_cast<const T*>(bias->data_ptr()) : nullptr,
        reinterpret_cast<T*>(output.data_ptr()),
        N, Cin, D, H, W, Cout, Cin_per_g, kD, kH, kW,
        oD, oH, oW, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups, total);
    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

template<typename T, typename Acc>
void launch_conv3d_backward_input(
    const Tensor& grad_output, const Tensor& weight, Tensor& grad_input,
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Cin_per_g,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t oD, int64_t oH, int64_t oW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t dD, int64_t dH, int64_t dW, int64_t groups,
    cudaStream_t stream)
{
    int64_t total = N * Cin * D * H * W;
    dim3 grid, block;
    compute_launch_config_1d(total, grid, block);

    conv3d_backward_input_kernel<T, Acc><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(grad_output.data_ptr()),
        reinterpret_cast<const T*>(weight.data_ptr()),
        reinterpret_cast<T*>(grad_input.data_ptr()),
        N, Cin, D, H, W, Cout, Cin_per_g, kD, kH, kW,
        oD, oH, oW, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups, total);
    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

template<typename T, typename Acc>
void launch_conv3d_backward_weight(
    const Tensor& grad_output, const Tensor& input, Tensor& grad_weight,
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Cin_per_g,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t oD, int64_t oH, int64_t oW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    int64_t dD, int64_t dH, int64_t dW, int64_t groups,
    cudaStream_t stream)
{
    int64_t total = Cout * Cin_per_g * kD * kH * kW;
    dim3 grid, block;
    compute_launch_config_1d(total, grid, block);

    conv3d_backward_weight_kernel<T, Acc><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(grad_output.data_ptr()),
        reinterpret_cast<const T*>(input.data_ptr()),
        reinterpret_cast<T*>(grad_weight.data_ptr()),
        N, Cin, D, H, W, Cout, Cin_per_g, kD, kH, kW,
        oD, oH, oW, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups, total);
    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

template<typename T, typename Acc>
void launch_conv3d_backward_bias(
    const Tensor& grad_output, Tensor& grad_bias,
    int64_t N, int64_t Cout, int64_t oD, int64_t oH, int64_t oW,
    cudaStream_t stream)
{
    dim3 grid(static_cast<unsigned int>(Cout), 1, 1);
    dim3 block(256, 1, 1);
    conv3d_backward_bias_kernel<T, Acc><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(grad_output.data_ptr()),
        reinterpret_cast<T*>(grad_bias.data_ptr()),
        N, Cout, oD, oH, oW);
    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

template<typename T, typename Acc>
void launch_conv_transpose3d_forward(
    const Tensor& input, const Tensor& weight, const Tensor* bias,
    Tensor& output,
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Cout_per_g,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t oD, int64_t oH, int64_t oW,
    int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
    cudaStream_t stream)
{
    int64_t total = N * Cout * oD * oH * oW;
    dim3 grid, block;
    compute_launch_config_1d(total, grid, block);

    conv_transpose3d_forward_kernel_impl<T, Acc><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(input.data_ptr()),
        reinterpret_cast<const T*>(weight.data_ptr()),
        bias ? reinterpret_cast<const T*>(bias->data_ptr()) : nullptr,
        reinterpret_cast<T*>(output.data_ptr()),
        N, Cin, D, H, W, Cout, Cout_per_g, kD, kH, kW,
        oD, oH, oW, stride, padding, dilation, groups, total);
    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

template<typename T, typename Acc>
void launch_conv_transpose3d_backward_input(
    const Tensor& grad_output, const Tensor& weight, Tensor& grad_input,
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Cout_per_g,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t oD, int64_t oH, int64_t oW,
    int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
    cudaStream_t stream)
{
    int64_t total = N * Cin * D * H * W;
    dim3 grid, block;
    compute_launch_config_1d(total, grid, block);

    conv_transpose3d_backward_input_kernel_impl<T, Acc><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(grad_output.data_ptr()),
        reinterpret_cast<const T*>(weight.data_ptr()),
        reinterpret_cast<T*>(grad_input.data_ptr()),
        N, Cin, D, H, W, Cout, Cout_per_g, kD, kH, kW,
        oD, oH, oW, stride, padding, dilation, groups, total);
    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

template<typename T, typename Acc>
void launch_conv_transpose3d_backward_weight(
    const Tensor& grad_output, const Tensor& input, Tensor& grad_weight,
    int64_t N, int64_t Cin, int64_t D, int64_t H, int64_t W,
    int64_t Cout, int64_t Cout_per_g,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t oD, int64_t oH, int64_t oW,
    int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
    cudaStream_t stream)
{
    int64_t total = Cin * Cout_per_g * kD * kH * kW;
    dim3 grid, block;
    compute_launch_config_1d(total, grid, block);

    conv_transpose3d_backward_weight_kernel_impl<T, Acc><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(grad_output.data_ptr()),
        reinterpret_cast<const T*>(input.data_ptr()),
        reinterpret_cast<T*>(grad_weight.data_ptr()),
        N, Cin, D, H, W, Cout, Cout_per_g, kD, kH, kW,
        oD, oH, oW, stride, padding, dilation, groups, total);
    TENZOR_CUDA_POST_LAUNCH_CHECK();
}

}  // anonymous namespace

// ============================================================================
// Public API: Conv3d forward
// ============================================================================

auto conv3d_forward_kernel(
    const Tensor& input, const Tensor& weight, const Tensor* bias,
    std::array<int64_t, 3> stride,
    std::array<int64_t, 3> padding,
    std::array<int64_t, 3> dilation,
    int64_t groups,
    cudaStream_t stream
) -> Tensor {
    if (stride[0] == 0 || stride[1] == 0 || stride[2] == 0)
        throw std::invalid_argument("Conv3d: stride cannot be zero");
    if (groups == 0)  throw std::invalid_argument("Conv3d: groups cannot be zero");
    // Per-axis stride/padding/dilation: the direct kernels index each spatial
    // axis (D, H, W) independently, so anisotropic params are fully supported.
    const int64_t sD = stride[0], sH = stride[1], sW = stride[2];
    const int64_t pD = padding[0], pH = padding[1], pW = padding[2];
    const int64_t dD = dilation[0], dH = dilation[1], dW = dilation[2];

    auto in_shape = input.shape();
    auto w_shape  = weight.shape();
    int64_t N    = in_shape[0];
    int64_t Cin  = in_shape[1];
    int64_t D    = in_shape[2];
    int64_t H    = in_shape[3];
    int64_t W    = in_shape[4];
    int64_t Cout = w_shape[0];
    int64_t Cin_per_g = w_shape[1];
    int64_t kD = w_shape[2];
    int64_t kH = w_shape[3];
    int64_t kW = w_shape[4];

    int64_t oD = conv3d_out_dim(D, kD, sD, pD, dD);
    int64_t oH = conv3d_out_dim(H, kH, sH, pH, dH);
    int64_t oW = conv3d_out_dim(W, kW, sW, pW, dW);

    Tensor output({N, Cout, oD, oH, oW}, input.dtype(), input.device());

    switch (input.dtype()) {
        case DType::Float32:
            // Upgrade the f32 accumulator to double, matching conv3d_backward.
            // The forward reduction sums Cin_per_g*kD*kH*kW MACs; a float
            // accumulator drifts O(sqrt(N)) ULPs and diverges from the CPU
            // reference past the cross-backend rtol=1e-3.
            launch_conv3d_forward<float, double>(
                input, weight, bias, output, N, Cin, D, H, W, Cout, Cin_per_g,
                kD, kH, kW, oD, oH, oW, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups, stream);
            break;
        case DType::Float64:
            launch_conv3d_forward<double, double>(
                input, weight, bias, output, N, Cin, D, H, W, Cout, Cin_per_g,
                kD, kH, kW, oD, oH, oW, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups, stream);
            break;
        case DType::Float16:
            launch_conv3d_forward<__half, float>(
                input, weight, bias, output, N, Cin, D, H, W, Cout, Cin_per_g,
                kD, kH, kW, oD, oH, oW, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups, stream);
            break;
        case DType::BFloat16:
            launch_conv3d_forward<__nv_bfloat16, float>(
                input, weight, bias, output, N, Cin, D, H, W, Cout, Cin_per_g,
                kD, kH, kW, oD, oH, oW, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups, stream);
            break;
        default:
            throw std::invalid_argument(
                "Conv3d CUDA fallback supports Float32/Float64/Float16/BFloat16");
    }
    return output;
}

// ============================================================================
// Public API: Conv3d backward
// ============================================================================

auto conv3d_backward_kernel(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    std::array<int64_t, 3> stride,
    std::array<int64_t, 3> padding,
    std::array<int64_t, 3> dilation,
    int64_t groups,
    bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    if (stride[0] == 0 || stride[1] == 0 || stride[2] == 0)
        throw std::invalid_argument("Conv3d: stride cannot be zero");
    if (groups == 0)  throw std::invalid_argument("Conv3d: groups cannot be zero");
    // Per-axis stride/padding/dilation; backward kernels index each spatial
    // axis independently so anisotropic params are fully supported.
    const int64_t sD = stride[0], sH = stride[1], sW = stride[2];
    const int64_t pD = padding[0], pH = padding[1], pW = padding[2];
    const int64_t dD = dilation[0], dH = dilation[1], dW = dilation[2];

    auto in_shape = input.shape();
    auto w_shape  = weight.shape();
    auto go_shape = grad_output.shape();
    int64_t N    = in_shape[0];
    int64_t Cin  = in_shape[1];
    int64_t D    = in_shape[2];
    int64_t H    = in_shape[3];
    int64_t W    = in_shape[4];
    int64_t Cout = w_shape[0];
    int64_t Cin_per_g = w_shape[1];
    int64_t kD = w_shape[2];
    int64_t kH = w_shape[3];
    int64_t kW = w_shape[4];
    int64_t oD = go_shape[2];
    int64_t oH = go_shape[3];
    int64_t oW = go_shape[4];

    // Match cudnn_conv3d_backward: always allocate all three tensors. The
    // unrequested ones are returned uninitialized (caller discards them).
    Tensor grad_input ({N, Cin, D, H, W},                  input.dtype(),  input.device());
    Tensor grad_weight({Cout, Cin_per_g, kD, kH, kW},      weight.dtype(), weight.device());
    Tensor grad_bias  ({Cout},                              weight.dtype(), weight.device());

#define TENZOR_CONV3D_BWD_DISPATCH(T_, ACC_)                                            \
    do {                                                                                 \
        if (compute_grad_input) {                                                        \
            launch_conv3d_backward_input<T_, ACC_>(                                      \
                grad_output, weight, grad_input,                                         \
                N, Cin, D, H, W, Cout, Cin_per_g, kD, kH, kW,                            \
                oD, oH, oW, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups, stream);                  \
        }                                                                                \
        if (compute_grad_weight) {                                                       \
            launch_conv3d_backward_weight<T_, ACC_>(                                     \
                grad_output, input, grad_weight,                                         \
                N, Cin, D, H, W, Cout, Cin_per_g, kD, kH, kW,                            \
                oD, oH, oW, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups, stream);                  \
        }                                                                                \
        if (compute_grad_bias) {                                                         \
            launch_conv3d_backward_bias<T_, ACC_>(                                       \
                grad_output, grad_bias, N, Cout, oD, oH, oW, stream);                    \
        }                                                                                \
    } while (0)

    switch (input.dtype()) {
        // Upgrade f32 accumulator to double. Each conv3d_backward_weight
        // thread sums thousands of grad_output*input pairs; a float
        // accumulator drifts by O(sqrt(N)) ULPs, which pushes the
        // cross-backend parity test past its rtol=1e-3 threshold. Using
        // double here costs some perf per MAC but brings CUDA in line
        // with the CPU reference.
        case DType::Float32:  TENZOR_CONV3D_BWD_DISPATCH(float,           double); break;
        case DType::Float64:  TENZOR_CONV3D_BWD_DISPATCH(double,          double); break;
        case DType::Float16:  TENZOR_CONV3D_BWD_DISPATCH(__half,          float); break;
        case DType::BFloat16: TENZOR_CONV3D_BWD_DISPATCH(__nv_bfloat16,   float); break;
        default:
            throw std::invalid_argument(
                "Conv3d CUDA fallback supports Float32/Float64/Float16/BFloat16");
    }
#undef TENZOR_CONV3D_BWD_DISPATCH

    return {grad_input, grad_weight, grad_bias};
}

// ============================================================================
// Public API: ConvTranspose3d forward
// ============================================================================

auto conv_transpose3d_forward_kernel(
    const Tensor& input, const Tensor& weight, const Tensor* bias,
    int64_t stride, int64_t padding, int64_t output_padding,
    int64_t dilation, int64_t groups,
    cudaStream_t stream
) -> Tensor {
    if (stride == 0)  throw std::invalid_argument("ConvTranspose3d: stride cannot be zero");
    if (groups == 0)  throw std::invalid_argument("ConvTranspose3d: groups cannot be zero");
    int64_t s = stride, p = padding, d = dilation;  // aliases for launch helpers

    auto in_shape = input.shape();
    auto w_shape  = weight.shape();
    int64_t N    = in_shape[0];
    int64_t Cin  = in_shape[1];
    int64_t D    = in_shape[2];
    int64_t H    = in_shape[3];
    int64_t W    = in_shape[4];
    // Weight layout: [Cin, Cout/g, kD, kH, kW]
    int64_t Cout_per_g = w_shape[1];
    int64_t Cout = Cout_per_g * groups;
    int64_t kD = w_shape[2];
    int64_t kH = w_shape[3];
    int64_t kW = w_shape[4];

    int64_t oD = conv_transpose3d_out_dim(D, kD, stride, padding, output_padding, dilation);
    int64_t oH = conv_transpose3d_out_dim(H, kH, stride, padding, output_padding, dilation);
    int64_t oW = conv_transpose3d_out_dim(W, kW, stride, padding, output_padding, dilation);

    Tensor output({N, Cout, oD, oH, oW}, input.dtype(), input.device());

    switch (input.dtype()) {
        case DType::Float32:
            // Accumulate Float32 ConvTranspose3d in double, matching the Conv3d
            // decision (a float accumulator drifts O(sqrt(N)) ULPs over the
            // Cin_per_g*kD*kH*kW reduction and pushes the cross-backend rtol=1e-3
            // parity test over threshold).
            launch_conv_transpose3d_forward<float, double>(
                input, weight, bias, output, N, Cin, D, H, W, Cout, Cout_per_g,
                kD, kH, kW, oD, oH, oW, s, p, d, groups, stream);
            break;
        case DType::Float64:
            launch_conv_transpose3d_forward<double, double>(
                input, weight, bias, output, N, Cin, D, H, W, Cout, Cout_per_g,
                kD, kH, kW, oD, oH, oW, s, p, d, groups, stream);
            break;
        case DType::Float16:
            launch_conv_transpose3d_forward<__half, float>(
                input, weight, bias, output, N, Cin, D, H, W, Cout, Cout_per_g,
                kD, kH, kW, oD, oH, oW, s, p, d, groups, stream);
            break;
        case DType::BFloat16:
            launch_conv_transpose3d_forward<__nv_bfloat16, float>(
                input, weight, bias, output, N, Cin, D, H, W, Cout, Cout_per_g,
                kD, kH, kW, oD, oH, oW, s, p, d, groups, stream);
            break;
        default:
            throw std::invalid_argument(
                "ConvTranspose3d CUDA fallback supports Float32/Float64/Float16/BFloat16");
    }
    return output;
}

// ============================================================================
// Public API: ConvTranspose3d backward (input/weight returned separately)
// ============================================================================

auto conv_transpose3d_backward_input_kernel(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    int64_t stride, int64_t padding, int64_t /*output_padding*/,
    int64_t dilation, int64_t groups,
    cudaStream_t stream
) -> Tensor {
    int64_t s = stride, p = padding, d = dilation;  // aliases for launch helpers
    auto in_shape = input.shape();
    auto w_shape  = weight.shape();
    auto go_shape = grad_output.shape();
    int64_t N    = in_shape[0];
    int64_t Cin  = in_shape[1];
    int64_t D    = in_shape[2];
    int64_t H    = in_shape[3];
    int64_t W    = in_shape[4];
    int64_t Cout_per_g = w_shape[1];
    int64_t Cout = Cout_per_g * groups;
    int64_t kD = w_shape[2];
    int64_t kH = w_shape[3];
    int64_t kW = w_shape[4];
    int64_t oD = go_shape[2];
    int64_t oH = go_shape[3];
    int64_t oW = go_shape[4];

    Tensor grad_input({N, Cin, D, H, W}, input.dtype(), input.device());

    switch (input.dtype()) {
        case DType::Float32:
            // Double accumulator for Float32 (cross-backend parity); see fwd note.
            launch_conv_transpose3d_backward_input<float, double>(
                grad_output, weight, grad_input, N, Cin, D, H, W, Cout, Cout_per_g,
                kD, kH, kW, oD, oH, oW, s, p, d, groups, stream);
            break;
        case DType::Float64:
            launch_conv_transpose3d_backward_input<double, double>(
                grad_output, weight, grad_input, N, Cin, D, H, W, Cout, Cout_per_g,
                kD, kH, kW, oD, oH, oW, s, p, d, groups, stream);
            break;
        case DType::Float16:
            launch_conv_transpose3d_backward_input<__half, float>(
                grad_output, weight, grad_input, N, Cin, D, H, W, Cout, Cout_per_g,
                kD, kH, kW, oD, oH, oW, s, p, d, groups, stream);
            break;
        case DType::BFloat16:
            launch_conv_transpose3d_backward_input<__nv_bfloat16, float>(
                grad_output, weight, grad_input, N, Cin, D, H, W, Cout, Cout_per_g,
                kD, kH, kW, oD, oH, oW, s, p, d, groups, stream);
            break;
        default:
            throw std::invalid_argument(
                "ConvTranspose3d CUDA fallback supports Float32/Float64/Float16/BFloat16");
    }
    return grad_input;
}

auto conv_transpose3d_backward_weight_kernel(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    int64_t stride, int64_t padding, int64_t /*output_padding*/,
    int64_t dilation, int64_t groups,
    cudaStream_t stream
) -> Tensor {
    int64_t s = stride, p = padding, d = dilation;  // aliases for launch helpers
    auto in_shape = input.shape();
    auto w_shape  = weight.shape();
    auto go_shape = grad_output.shape();
    int64_t N    = in_shape[0];
    int64_t Cin  = in_shape[1];
    int64_t D    = in_shape[2];
    int64_t H    = in_shape[3];
    int64_t W    = in_shape[4];
    int64_t Cout_per_g = w_shape[1];
    int64_t Cout = Cout_per_g * groups;
    int64_t kD = w_shape[2];
    int64_t kH = w_shape[3];
    int64_t kW = w_shape[4];
    int64_t oD = go_shape[2];
    int64_t oH = go_shape[3];
    int64_t oW = go_shape[4];

    Tensor grad_weight({Cin, Cout_per_g, kD, kH, kW}, input.dtype(), input.device());

    switch (input.dtype()) {
        case DType::Float32:
            // Double accumulator for Float32 (cross-backend parity); see fwd note.
            launch_conv_transpose3d_backward_weight<float, double>(
                grad_output, input, grad_weight, N, Cin, D, H, W, Cout, Cout_per_g,
                kD, kH, kW, oD, oH, oW, s, p, d, groups, stream);
            break;
        case DType::Float64:
            launch_conv_transpose3d_backward_weight<double, double>(
                grad_output, input, grad_weight, N, Cin, D, H, W, Cout, Cout_per_g,
                kD, kH, kW, oD, oH, oW, s, p, d, groups, stream);
            break;
        case DType::Float16:
            launch_conv_transpose3d_backward_weight<__half, float>(
                grad_output, input, grad_weight, N, Cin, D, H, W, Cout, Cout_per_g,
                kD, kH, kW, oD, oH, oW, s, p, d, groups, stream);
            break;
        case DType::BFloat16:
            launch_conv_transpose3d_backward_weight<__nv_bfloat16, float>(
                grad_output, input, grad_weight, N, Cin, D, H, W, Cout, Cout_per_g,
                kD, kH, kW, oD, oH, oW, s, p, d, groups, stream);
            break;
        default:
            throw std::invalid_argument(
                "ConvTranspose3d CUDA fallback supports Float32/Float64/Float16/BFloat16");
    }
    return grad_weight;
}

}  // namespace cuda
}  // namespace tenzor
