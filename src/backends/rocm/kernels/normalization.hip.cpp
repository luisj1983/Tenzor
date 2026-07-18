/**
 * @file normalization.hip.cpp
 * @brief HIP normalization kernels for AMD GPUs
 *
 * Implements LayerNorm, GroupNorm, and InstanceNorm operations with forward and backward passes.
 */

#include "rocm_nan_helpers.hip.h"  // E.2: safe_f2h / safe_h2f / safe_f2bf / safe_bf2f
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "fp16_saturate.h"
#include "../rocm_error.hpp"
#include <stdexcept>
#include <cmath>
#include <vector>

namespace tenzor {
namespace rocm {

// Grid-stride loop helper
#define HIP_KERNEL_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

constexpr int BLOCK_SIZE = 256;

// Warp-level reduction using shuffle instructions
// Uses HIP built-in warpSize to support both wave32 (RDNA 3/4) and wave64 (CDNA)
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
// Returns the total sum in ALL threads (broadcast via shared memory)
template<typename T>
__device__ T block_reduce_sum(T val, T* shared) {
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : T(0);
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    // Broadcast result from thread 0 to all threads
    if (threadIdx.x == 0) {
        shared[0] = val;
    }
    __syncthreads();

    return shared[0];
}

// ==============================================================================
// Layer Normalization Forward
// ==============================================================================

// `Acc` is the reduction accumulator type: for a Float32 input it is `double`,
// so mean and variance are accumulated/divided in double — matching the CPU
// reference and the CUDA sibling (fused_ops.cu forces Acc=double for F32). A
// float accumulator over a large normalized_size drifts.
template<typename T, typename Acc = T>
__global__ void layer_norm_forward_kernel(
    const T* input,
    const T* weight,  // gamma, can be nullptr
    const T* bias,    // beta, can be nullptr
    T* output,
    T* mean_out,      // optional, for backward
    T* rstd_out,      // optional, for backward
    int64_t batch_size,
    int64_t normalized_size,
    double eps
) {
    extern __shared__ unsigned char shared_mem[];
    Acc* shared = reinterpret_cast<Acc*>(shared_mem);

    int64_t batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const T* input_row = input + batch_idx * normalized_size;
    T* output_row = output + batch_idx * normalized_size;

    // Compute mean (in the wide accumulator type)
    Acc sum = Acc(0);
    for (int64_t i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        sum += static_cast<Acc>(input_row[i]);
    }
    sum = block_reduce_sum(sum, shared);
    __syncthreads();

    Acc mean = sum / Acc(normalized_size);
    if (threadIdx.x == 0 && mean_out) {
        mean_out[batch_idx] = static_cast<T>(mean);
    }

    // Compute variance (in the wide accumulator type)
    Acc var_sum = Acc(0);
    for (int64_t i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        Acc diff = static_cast<Acc>(input_row[i]) - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, shared);
    __syncthreads();

    Acc variance = var_sum / Acc(normalized_size);
    Acc rstd = rsqrt(variance + static_cast<Acc>(eps));
    if (threadIdx.x == 0 && rstd_out) {
        rstd_out[batch_idx] = static_cast<T>(rstd);
    }

    // Normalize and apply affine transform
    for (int64_t i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        T normalized = static_cast<T>((static_cast<Acc>(input_row[i]) - mean) * rstd);
        if (weight && bias) {
            output_row[i] = normalized * weight[i] + bias[i];
        } else if (weight) {
            output_row[i] = normalized * weight[i];
        } else if (bias) {
            output_row[i] = normalized + bias[i];
        } else {
            output_row[i] = normalized;
        }
    }
}

auto layer_norm_kernel_with_stats(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor* weight,
    const Tensor* bias,
    double eps,
    hipStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    // Float16: upcast to Float32 to prevent precision loss in mean/rstd
    // storage. Stats stay Float32 per the attention contract (CUDA cuDNN does
    // the same: saved stats Float32 for half-precision input).
    if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        const Tensor* weight_f32_ptr = nullptr;
        const Tensor* bias_f32_ptr   = nullptr;
        Tensor weight_f32, bias_f32;
        if (weight) { weight_f32 = weight->to(DType::Float32); weight_f32_ptr = &weight_f32; }
        if (bias)   { bias_f32   = bias->to(DType::Float32);   bias_f32_ptr   = &bias_f32;   }
        auto [out_f32, mean_f32, rstd_f32] = layer_norm_kernel_with_stats(
            input_f32, normalized_shape, weight_f32_ptr, bias_f32_ptr, eps, stream);
        auto out_f16 = out_f32.to(DType::Float16);
        fp16_saturate(out_f16.data_ptr(), out_f16.numel(), stream);
        return {out_f16, mean_f32, rstd_f32};
    }

    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        const Tensor* weight_f32_ptr = nullptr;
        const Tensor* bias_f32_ptr   = nullptr;
        Tensor weight_f32, bias_f32;
        if (weight) { weight_f32 = weight->to(DType::Float32); weight_f32_ptr = &weight_f32; }
        if (bias)   { bias_f32   = bias->to(DType::Float32);   bias_f32_ptr   = &bias_f32;   }
        auto [out_f32, mean_f32, rstd_f32] = layer_norm_kernel_with_stats(
            input_f32, normalized_shape, weight_f32_ptr, bias_f32_ptr, eps, stream);
        return {out_f32.to(DType::BFloat16), mean_f32, rstd_f32};
    }

    // The forward kernel indexes storage with flat per-row offsets
    // (input + batch_idx * normalized_size), which assumes a contiguous
    // layout. A transposed/sliced/permuted input would read the wrong
    // storage, so materialize a contiguous copy first (matches the CUDA
    // sibling fused_layer_norm_cuda and ROCm RMSNorm). weight/bias are
    // indexed by the normalized position and are contiguified too.
    Tensor input_contig = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_contig, bias_contig;
    const Tensor* weight_c = weight;
    const Tensor* bias_c   = bias;
    if (weight && !weight->is_contiguous()) {
        weight_contig = weight->contiguous();
        weight_c = &weight_contig;
    }
    if (bias && !bias->is_contiguous()) {
        bias_contig = bias->contiguous();
        bias_c = &bias_contig;
    }

    auto input_shape = input_contig.shape();
    int64_t ndim = input_shape.size();
    int64_t norm_ndim = normalized_shape.size();

    int64_t normalized_size = 1;
    for (auto dim : normalized_shape) {
        normalized_size *= dim;
    }
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - norm_ndim; ++i) {
        batch_size *= input_shape[i];
    }

    Tensor output(std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                  input_contig.dtype(), input_contig.device());

    // Saved stats shape is the leading (ndim - norm_ndim) batch dims.
    std::vector<int64_t> stats_shape(input_shape.begin(),
                                     input_shape.begin() + (ndim - norm_ndim));
    // Empty stats_shape (normalized_shape covers every dim, no batch dims
    // left) is a true 0-dim scalar -- do not force a size-1 dim.
    Tensor mean(stats_shape, input_contig.dtype(), input_contig.device());
    Tensor rstd(stats_shape, input_contig.dtype(), input_contig.device());

    int threads = BLOCK_SIZE;
    int shared_mem_size = (threads / 32 + 1) * sizeof(float);

    if (input_contig.dtype() == DType::Float32) {
        // Accumulate mean/variance in double (Acc=double); shared memory sized
        // for the wider accumulator (matches the F64 launch size).
        hipLaunchKernelGGL((layer_norm_forward_kernel<float, double>),
            dim3(batch_size), dim3(threads), shared_mem_size * 2, stream,
            input_contig.data<float>(),
            weight_c ? weight_c->data<float>() : nullptr,
            bias_c ? bias_c->data<float>() : nullptr,
            output.data<float>(),
            mean.data<float>(), rstd.data<float>(),
            batch_size, normalized_size, eps);
        HIP_POST_LAUNCH_CHECK();
    } else if (input_contig.dtype() == DType::Float64) {
        hipLaunchKernelGGL(layer_norm_forward_kernel<double>,
            dim3(batch_size), dim3(threads), shared_mem_size * 2, stream,
            input_contig.data<double>(),
            weight_c ? weight_c->data<double>() : nullptr,
            bias_c ? bias_c->data<double>() : nullptr,
            output.data<double>(),
            mean.data<double>(), rstd.data<double>(),
            batch_size, normalized_size, eps);
        HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("layer_norm_kernel: unsupported dtype");
    }

    return {output, mean, rstd};
}

// Output-only convenience wrapper (callers that don't need saved stats).
auto layer_norm_kernel(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor* weight,
    const Tensor* bias,
    double eps,
    hipStream_t stream
) -> Tensor {
    auto [output, mean, rstd] = layer_norm_kernel_with_stats(
        input, normalized_shape, weight, bias, eps, stream);
    (void)mean; (void)rstd;
    return output;
}

// ==============================================================================
// Layer Normalization Backward
// ==============================================================================

// `Acc` is the reduction accumulator type: for a Float32 input it is `double`,
// so the ds/db dot-products accumulate in double — matching the CUDA sibling.
template<typename T, typename Acc = T>
__global__ void layer_norm_backward_kernel(
    const T* grad_output,
    const T* input,
    const T* weight,
    const T* mean,
    const T* rstd,
    T* grad_input,
    T* grad_weight,  // atomically accumulated
    T* grad_bias,    // atomically accumulated
    int64_t batch_size,
    int64_t normalized_size
) {
    extern __shared__ unsigned char shared_mem[];
    Acc* shared = reinterpret_cast<Acc*>(shared_mem);

    int64_t batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const T* grad_out_row = grad_output + batch_idx * normalized_size;
    const T* input_row = input + batch_idx * normalized_size;
    T* grad_in_row = grad_input + batch_idx * normalized_size;

    Acc m = static_cast<Acc>(mean[batch_idx]);
    Acc rs = static_cast<Acc>(rstd[batch_idx]);

    // Compute ds and db (dot products) in the wide accumulator type
    Acc ds = Acc(0);
    Acc db = Acc(0);
    for (int64_t i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        Acc x_hat = (static_cast<Acc>(input_row[i]) - m) * rs;
        Acc w = weight ? static_cast<Acc>(weight[i]) : Acc(1);
        Acc go = static_cast<Acc>(grad_out_row[i]);
        ds += go * w * x_hat;
        db += go * w;
    }

    ds = block_reduce_sum(ds, shared);
    __syncthreads();
    db = block_reduce_sum(db, shared);
    __syncthreads();

    // Compute gradient for input
    // dx_i = rstd * (go_i*w_i - mean(go*w) - x_hat_i*mean(go*w*x_hat))
    //      = rstd * (go_i*w_i - scale*(db + x_hat_i*ds))
    // ds/db already include the `w` factor (accumulated as go*w and go*w*x_hat
    // above), so `w` must NOT also multiply the correction term -- the
    // previous `rs * w * (go - scale*(db+x_hat*ds))` incorrectly applied w to
    // the whole expression instead of just the go term, matching neither the
    // CPU nor CUDA reference (cuda/kernels/fused_ops.cu's
    // fused_layer_norm_backward_kernel: grad_out_weighted - mean_grad_out -
    // normalized*mean_grad_out_normalized, all scaled by inv_std at the end).
    Acc scale = Acc(1) / Acc(normalized_size);
    for (int64_t i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        Acc x_hat = (static_cast<Acc>(input_row[i]) - m) * rs;
        Acc w = weight ? static_cast<Acc>(weight[i]) : Acc(1);
        Acc go_w = w * static_cast<Acc>(grad_out_row[i]);
        grad_in_row[i] = static_cast<T>(rs * (go_w - scale * (db + x_hat * ds)));

        // Accumulate gradients for weight and bias
        if (grad_weight) {
            atomicAdd(&grad_weight[i], static_cast<T>(static_cast<Acc>(grad_out_row[i]) * x_hat));
        }
        if (grad_bias) {
            atomicAdd(&grad_bias[i], grad_out_row[i]);
        }
    }
}

auto layer_norm_backward_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor* weight,
    hipStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    // Float16: upcast to Float32 to prevent precision loss from rstd stored as half
    if (input.dtype() == DType::Float16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto mean_f32 = mean.to(DType::Float32);
        auto rstd_f32 = rstd.to(DType::Float32);
        const Tensor* weight_f32_ptr = nullptr;
        Tensor weight_f32;
        if (weight) { weight_f32 = weight->to(DType::Float32); weight_f32_ptr = &weight_f32; }
        auto [gi_f32, gw_f32, gb_f32] = layer_norm_backward_kernel(grad_output_f32, input_f32, mean_f32, rstd_f32, weight_f32_ptr, stream);
        auto gi = gi_f32.to(DType::Float16);
        fp16_saturate(gi.data_ptr(), gi.numel(), stream);
        auto gw = gw_f32.numel() > 0 ? gw_f32.to(DType::Float16) : gw_f32;
        if (gw.numel() > 0) fp16_saturate(gw.data_ptr(), gw.numel(), stream);
        auto gb = gb_f32.numel() > 0 ? gb_f32.to(DType::Float16) : gb_f32;
        if (gb.numel() > 0) fp16_saturate(gb.data_ptr(), gb.numel(), stream);
        return {gi, gw, gb};
    }

    auto input_shape = input.shape();
    int64_t batch_size = mean.numel();
    int64_t normalized_size = input.numel() / batch_size;

    Tensor grad_input(std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                      input.dtype(), input.device());

    // Create gradient tensors for weight and bias if weight is provided
    Tensor grad_weight, grad_bias;
    if (weight) {
        grad_weight = Tensor({normalized_size}, input.dtype(), input.device());
        grad_bias = Tensor({normalized_size}, input.dtype(), input.device());
        // Zero initialize
        HIP_CHECK(hipMemsetAsync(grad_weight.data<uint8_t>(), 0,
            grad_weight.numel() * dtype_size(input.dtype()), stream));
        HIP_CHECK(hipMemsetAsync(grad_bias.data<uint8_t>(), 0,
            grad_bias.numel() * dtype_size(input.dtype()), stream));
    }

    int threads = std::min(static_cast<int64_t>(BLOCK_SIZE), normalized_size);
    int shared_mem_size = (threads / 32 + 1) * sizeof(float);

    if (input.dtype() == DType::Float32) {
        // Accumulate ds/db in double (Acc=double); shared mem sized for double.
        hipLaunchKernelGGL((layer_norm_backward_kernel<float, double>),
            dim3(batch_size), dim3(threads), shared_mem_size * 2, stream,
            grad_output.data<float>(),
            input.data<float>(),
            weight ? weight->data<float>() : nullptr,
            mean.data<float>(),
            rstd.data<float>(),
            grad_input.data<float>(),
            weight ? grad_weight.data<float>() : nullptr,
            weight ? grad_bias.data<float>() : nullptr,
            batch_size, normalized_size);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(layer_norm_backward_kernel<double>,
            dim3(batch_size), dim3(threads), shared_mem_size * 2, stream,
            grad_output.data<double>(),
            input.data<double>(),
            weight ? weight->data<double>() : nullptr,
            mean.data<double>(),
            rstd.data<double>(),
            grad_input.data<double>(),
            weight ? grad_weight.data<double>() : nullptr,
            weight ? grad_bias.data<double>() : nullptr,
            batch_size, normalized_size);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto mean_f32 = mean.to(DType::Float32);
        auto rstd_f32 = rstd.to(DType::Float32);
        const Tensor* weight_f32_ptr = nullptr;
        Tensor weight_f32;
        if (weight) { weight_f32 = weight->to(DType::Float32); weight_f32_ptr = &weight_f32; }
        auto [gi_f32, gw_f32, gb_f32] = layer_norm_backward_kernel(grad_output_f32, input_f32, mean_f32, rstd_f32, weight_f32_ptr, stream);
        return {gi_f32.to(DType::BFloat16),
                gw_f32.numel() > 0 ? gw_f32.to(DType::BFloat16) : gw_f32,
                gb_f32.numel() > 0 ? gb_f32.to(DType::BFloat16) : gb_f32};
    } else {
        throw std::runtime_error("layer_norm_backward_kernel: unsupported dtype");
    }

    return {grad_input, grad_weight, grad_bias};
}

// ==============================================================================
// Group Normalization Forward
// ==============================================================================

// `Acc` is the reduction accumulator type: for a Float32 input it is `double`,
// so per-group mean and variance accumulate in double — matching the CUDA
// sibling. (The dedicated FP16 kernel below already uses a float accumulator.)
template<typename T, typename Acc = T>
__global__ void group_norm_forward_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    T* output,
    int64_t N,
    int64_t C,
    int64_t HW,
    int64_t num_groups,
    int64_t channels_per_group,
    double eps,
    float* saved_mean = nullptr,
    float* saved_rstd = nullptr
) {
    extern __shared__ unsigned char shared_mem[];
    Acc* shared = reinterpret_cast<Acc*>(shared_mem);

    // Each block handles one (batch, group) pair
    int64_t idx = blockIdx.x;
    int64_t n = idx / num_groups;
    int64_t g = idx % num_groups;

    if (n >= N) return;

    int64_t group_size = channels_per_group * HW;
    int64_t group_offset = n * C * HW + g * channels_per_group * HW;

    // Compute mean (in the wide accumulator type)
    Acc sum = Acc(0);
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int64_t c_local = i / HW;
        int64_t hw = i % HW;
        int64_t c = g * channels_per_group + c_local;
        sum += static_cast<Acc>(input[n * C * HW + c * HW + hw]);
    }
    sum = block_reduce_sum(sum, shared);
    __syncthreads();

    Acc mean = sum / Acc(group_size);

    // Compute variance (in the wide accumulator type)
    Acc var_sum = Acc(0);
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int64_t c_local = i / HW;
        int64_t hw = i % HW;
        int64_t c = g * channels_per_group + c_local;
        Acc diff = static_cast<Acc>(input[n * C * HW + c * HW + hw]) - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, shared);
    __syncthreads();

    Acc variance = var_sum / Acc(group_size);
    Acc rstd = rsqrt(variance + static_cast<Acc>(eps));

    // Optionally save stats for backward pass
    if (threadIdx.x == 0 && saved_mean) {
        saved_mean[idx] = static_cast<float>(mean);
        saved_rstd[idx] = static_cast<float>(rstd);
    }

    // Normalize and apply affine
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int64_t c_local = i / HW;
        int64_t hw = i % HW;
        int64_t c = g * channels_per_group + c_local;
        int64_t in_idx = n * C * HW + c * HW + hw;

        T normalized = static_cast<T>((static_cast<Acc>(input[in_idx]) - mean) * rstd);
        if (weight && bias) {
            output[in_idx] = normalized * weight[c] + bias[c];
        } else if (weight) {
            output[in_idx] = normalized * weight[c];
        } else if (bias) {
            output[in_idx] = normalized + bias[c];
        } else {
            output[in_idx] = normalized;
        }
    }
}

// Float16 Group Norm Forward - uses float computation internally
__global__ void group_norm_forward_kernel_fp16(
    const __half* input,
    const __half* weight,
    const __half* bias,
    __half* output,
    int64_t N,
    int64_t C,
    int64_t HW,
    int64_t num_groups,
    int64_t channels_per_group,
    float eps,
    float* saved_mean = nullptr,
    float* saved_rstd = nullptr
) {
    extern __shared__ unsigned char shared_mem[];
    float* shared = reinterpret_cast<float*>(shared_mem);

    int64_t idx = blockIdx.x;
    int64_t n = idx / num_groups;
    int64_t g = idx % num_groups;

    if (n >= N) return;

    int64_t group_size = channels_per_group * HW;

    // Compute mean in float
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int64_t c_local = i / HW;
        int64_t hw = i % HW;
        int64_t c = g * channels_per_group + c_local;
        sum += tenzor::rocm::safe_h2f(input[n * C * HW + c * HW + hw]);
    }
    sum = block_reduce_sum(sum, shared);
    __syncthreads();

    float mean = sum / static_cast<float>(group_size);

    // Compute variance in float
    float var_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int64_t c_local = i / HW;
        int64_t hw = i % HW;
        int64_t c = g * channels_per_group + c_local;
        float diff = tenzor::rocm::safe_h2f(input[n * C * HW + c * HW + hw]) - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, shared);
    __syncthreads();

    float variance = var_sum / static_cast<float>(group_size);
    float rstd = rsqrtf(variance + eps);

    // Optionally save stats for backward pass
    if (threadIdx.x == 0 && saved_mean) {
        saved_mean[idx] = mean;
        saved_rstd[idx] = rstd;
    }

    // Normalize and apply affine
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int64_t c_local = i / HW;
        int64_t hw = i % HW;
        int64_t c = g * channels_per_group + c_local;
        int64_t in_idx = n * C * HW + c * HW + hw;

        float normalized = (tenzor::rocm::safe_h2f(input[in_idx]) - mean) * rstd;
        if (weight && bias) {
            output[in_idx] = tenzor::rocm::safe_f2h(normalized * tenzor::rocm::safe_h2f(weight[c]) + tenzor::rocm::safe_h2f(bias[c]));
        } else if (weight) {
            output[in_idx] = tenzor::rocm::safe_f2h(normalized * tenzor::rocm::safe_h2f(weight[c]));
        } else if (bias) {
            output[in_idx] = tenzor::rocm::safe_f2h(normalized + tenzor::rocm::safe_h2f(bias[c]));
        } else {
            output[in_idx] = tenzor::rocm::safe_f2h(normalized);
        }
    }
}

auto group_norm_kernel(
    const Tensor& input,
    int64_t num_groups,
    const Tensor* weight,
    const Tensor* bias,
    double eps,
    hipStream_t stream
) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() < 2) {
        throw std::runtime_error("group_norm_kernel: input must have at least 2 dimensions");
    }

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t HW = 1;
    for (size_t i = 2; i < input_shape.size(); ++i) {
        HW *= input_shape[i];
    }

    if (C % num_groups != 0) {
        throw std::runtime_error("group_norm_kernel: num_groups must divide num_channels");
    }

    int64_t channels_per_group = C / num_groups;

    Tensor output(std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                  input.dtype(), input.device());

    int64_t group_size = channels_per_group * HW;
    int threads = std::min(static_cast<int64_t>(BLOCK_SIZE), group_size);
    int blocks = N * num_groups;
    int shared_mem_size = (threads / 32 + 1) * sizeof(float);

    if (input.dtype() == DType::Float32) {
        // Accumulate mean/variance in double (Acc=double); shared mem doubled.
        hipLaunchKernelGGL((group_norm_forward_kernel<float, double>),
            dim3(blocks), dim3(threads), shared_mem_size * 2, stream,
            input.data<float>(),
            weight ? weight->data<float>() : nullptr,
            bias ? bias->data<float>() : nullptr,
            output.data<float>(),
            N, C, HW, num_groups, channels_per_group, eps);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(group_norm_forward_kernel<double>,
            dim3(blocks), dim3(threads), shared_mem_size * 2, stream,
            input.data<double>(),
            weight ? weight->data<double>() : nullptr,
            bias ? bias->data<double>() : nullptr,
            output.data<double>(),
            N, C, HW, num_groups, channels_per_group, eps);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(group_norm_forward_kernel_fp16,
            dim3(blocks), dim3(threads), shared_mem_size, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            weight ? reinterpret_cast<const __half*>(weight->data<Float16>()) : nullptr,
            bias ? reinterpret_cast<const __half*>(bias->data<Float16>()) : nullptr,
            reinterpret_cast<__half*>(output.data<Float16>()),
            N, C, HW, num_groups, channels_per_group, static_cast<float>(eps));
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        const Tensor* weight_f32_ptr = nullptr;
        const Tensor* bias_f32_ptr = nullptr;
        Tensor weight_f32, bias_f32;
        if (weight) { weight_f32 = weight->to(DType::Float32); weight_f32_ptr = &weight_f32; }
        if (bias) { bias_f32 = bias->to(DType::Float32); bias_f32_ptr = &bias_f32; }
        auto result_f32 = group_norm_kernel(input_f32, num_groups, weight_f32_ptr, bias_f32_ptr, eps, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("group_norm_kernel: unsupported dtype");
    }

    return output;
}

// Wrapper that returns (output, saved_mean, saved_rstd) for the backward pass.
// Mean and rstd are computed on GPU alongside the forward pass (no CPU transfer).
auto group_norm_forward_with_stats(
    const Tensor& input,
    int64_t num_groups,
    const Tensor* weight,
    const Tensor* bias,
    double eps,
    hipStream_t stream
) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    if (input_shape.size() < 2) {
        throw std::runtime_error("group_norm_forward_with_stats: input must have at least 2 dimensions");
    }

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t HW = 1;
    for (size_t i = 2; i < input_shape.size(); ++i) HW *= input_shape[i];

    if (C % num_groups != 0) {
        throw std::runtime_error("group_norm_forward_with_stats: num_groups must divide num_channels");
    }

    int64_t channels_per_group = C / num_groups;
    int64_t group_size = channels_per_group * HW;

    Tensor output(std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                  input.dtype(), input.device());
    Tensor saved_mean({N, num_groups}, DType::Float32, input.device());
    Tensor saved_rstd({N, num_groups}, DType::Float32, input.device());

    int threads = std::min(static_cast<int64_t>(BLOCK_SIZE), group_size);
    int blocks = N * num_groups;
    int shared_mem_size = (threads / 32 + 1) * sizeof(float);

    if (input.dtype() == DType::Float32) {
        // Accumulate mean/variance in double (Acc=double); shared mem doubled.
        hipLaunchKernelGGL((group_norm_forward_kernel<float, double>),
            dim3(blocks), dim3(threads), shared_mem_size * 2, stream,
            input.data<float>(),
            weight ? weight->data<float>() : nullptr,
            bias ? bias->data<float>() : nullptr,
            output.data<float>(),
            N, C, HW, num_groups, channels_per_group, eps,
            saved_mean.data<float>(), saved_rstd.data<float>());
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(group_norm_forward_kernel<double>,
            dim3(blocks), dim3(threads), shared_mem_size * 2, stream,
            input.data<double>(),
            weight ? weight->data<double>() : nullptr,
            bias ? bias->data<double>() : nullptr,
            output.data<double>(),
            N, C, HW, num_groups, channels_per_group, eps,
            saved_mean.data<float>(), saved_rstd.data<float>());
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(group_norm_forward_kernel_fp16,
            dim3(blocks), dim3(threads), shared_mem_size, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            weight ? reinterpret_cast<const __half*>(weight->data<Float16>()) : nullptr,
            bias ? reinterpret_cast<const __half*>(bias->data<Float16>()) : nullptr,
            reinterpret_cast<__half*>(output.data<Float16>()),
            N, C, HW, num_groups, channels_per_group, static_cast<float>(eps),
            saved_mean.data<float>(), saved_rstd.data<float>());
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        const Tensor* weight_f32_ptr = nullptr;
        const Tensor* bias_f32_ptr = nullptr;
        Tensor weight_f32, bias_f32;
        if (weight) { weight_f32 = weight->to(DType::Float32); weight_f32_ptr = &weight_f32; }
        if (bias) { bias_f32 = bias->to(DType::Float32); bias_f32_ptr = &bias_f32; }
        auto result = group_norm_forward_with_stats(input_f32, num_groups, weight_f32_ptr, bias_f32_ptr, eps, stream);
        return {result[0].to(DType::BFloat16), result[1], result[2]};
    } else {
        throw std::runtime_error("group_norm_forward_with_stats: unsupported dtype");
    }

    return {output, saved_mean, saved_rstd};
}

// ==============================================================================
// Group Normalization Backward — HIP kernel
// ==============================================================================

// Each thread block handles one (sample, group) pair.
// Pass 1: compute sum_dy and sum_dy_xhat via block reduction.
// Pass 2: compute grad_input using the normalization backward formula.
// Accumulate grad_weight/grad_bias via atomicAdd across samples.
template<typename T>
__global__ void group_norm_backward_hip_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ mean_saved,
    const T* __restrict__ inv_std_saved,
    T* __restrict__ grad_input,
    // F-085 pattern: double-precision scratch accumulators, atomicAdd'd into
    // by every contributing (n, group) block, then narrowed to T once by
    // narrow_group_norm_grad_accum_kernel after this kernel completes. A
    // plain-T atomicAdd accumulator (previous behavior) loses precision for
    // Float32 across large N, unlike CPU's double accumulation.
    double* __restrict__ grad_weight_accum,
    double* __restrict__ grad_bias_accum,
    int64_t N, int64_t C, int64_t HW,
    int64_t num_groups, int64_t channels_per_group) {

    int64_t group_idx = blockIdx.x;
    int64_t n = group_idx / num_groups;
    int64_t g = group_idx % num_groups;

    if (n >= N || g >= num_groups) return;

    int64_t c_start = g * channels_per_group;
    int64_t group_size = channels_per_group * HW;

    T mean_val = mean_saved[group_idx];
    T inv_std = inv_std_saved[group_idx];

    // Pass 1: compute sum_dy and sum_dy_xhat.
    // F038: accumulate the Float32 reduction in double to match ROCm LayerNorm
    // backward and the CPU reference (Float64 input already runs T=double).
    double local_sum_dy = 0.0;
    double local_sum_dy_xhat = 0.0;

    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int64_t c_offset = i / HW;
        int64_t hw = i % HW;
        int64_t c = c_start + c_offset;
        int64_t idx = (n * C + c) * HW + hw;
        T dy = grad_output[idx];
        if (weight) dy = dy * weight[c];
        T xhat = (input[idx] - mean_val) * inv_std;
        local_sum_dy += static_cast<double>(dy);
        local_sum_dy_xhat += static_cast<double>(dy) * static_cast<double>(xhat);
    }

    // Block-level reduction using existing warp_reduce_sum + shared memory
    // Max warps per block: 1024/32=32 (RDNA) or 1024/64=16 (CDNA)
    constexpr int MAX_WARPS = 32;
    __shared__ double shared_dy[MAX_WARPS];
    __shared__ double shared_dy_xhat[MAX_WARPS];

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    local_sum_dy = warp_reduce_sum(local_sum_dy);
    local_sum_dy_xhat = warp_reduce_sum(local_sum_dy_xhat);

    if (lane == 0) {
        shared_dy[warp_id] = local_sum_dy;
        shared_dy_xhat[warp_id] = local_sum_dy_xhat;
    }
    __syncthreads();

    double sum_dy, sum_dy_xhat;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (warp_id == 0) {
        local_sum_dy = (lane < num_warps) ? shared_dy[lane] : 0.0;
        local_sum_dy_xhat = (lane < num_warps) ? shared_dy_xhat[lane] : 0.0;
        local_sum_dy = warp_reduce_sum(local_sum_dy);
        local_sum_dy_xhat = warp_reduce_sum(local_sum_dy_xhat);
        if (lane == 0) {
            shared_dy[0] = local_sum_dy;
            shared_dy_xhat[0] = local_sum_dy_xhat;
        }
    }
    __syncthreads();
    sum_dy = shared_dy[0];
    sum_dy_xhat = shared_dy_xhat[0];

    double inv_group_size = 1.0 / static_cast<double>(group_size);

    // Pass 2: compute grad_input
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int64_t c_offset = i / HW;
        int64_t hw = i % HW;
        int64_t c = c_start + c_offset;
        int64_t idx = (n * C + c) * HW + hw;
        T dy = grad_output[idx];
        if (weight) dy = dy * weight[c];
        T xhat = (input[idx] - mean_val) * inv_std;
        grad_input[idx] = static_cast<T>(static_cast<double>(inv_std) *
            (static_cast<double>(dy) - inv_group_size *
                (sum_dy + static_cast<double>(xhat) * sum_dy_xhat)));
    }

    // Accumulate grad_weight and grad_bias (atomic since multiple samples contribute).
    // Accumulate in double precision regardless of T (F-085 pattern).
    if (weight && grad_weight_accum && grad_bias_accum) {
        for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
            int64_t c_offset = i / HW;
            int64_t hw = i % HW;
            int64_t c = c_start + c_offset;
            int64_t idx = (n * C + c) * HW + hw;
            T xhat = (input[idx] - mean_val) * inv_std;
            atomicAdd(&grad_weight_accum[c], static_cast<double>(grad_output[idx]) * static_cast<double>(xhat));
            atomicAdd(&grad_bias_accum[c], static_cast<double>(grad_output[idx]));
        }
    }
}

// Narrows the double-precision grad_weight/grad_bias scratch accumulators
// (populated via atomicAdd across every (n, group) block in
// group_norm_backward_hip_kernel) down to the output dtype T, once per channel.
template<typename T>
__global__ void narrow_group_norm_grad_accum_kernel(
    const double* __restrict__ grad_weight_accum,
    const double* __restrict__ grad_bias_accum,
    T* __restrict__ grad_weight,
    T* __restrict__ grad_bias,
    int64_t C) {
    int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c < C) {
        grad_weight[c] = static_cast<T>(grad_weight_accum[c]);
        grad_bias[c] = static_cast<T>(grad_bias_accum[c]);
    }
}

auto group_norm_backward_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    int64_t num_groups,
    const Tensor* weight,
    hipStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t HW = 1;
    for (size_t i = 2; i < shape.size(); ++i) HW *= shape[i];
    int64_t channels_per_group = C / num_groups;

    Tensor grad_input(shape, input.dtype(), input.device());
    Tensor grad_weight({C}, input.dtype(), input.device());
    Tensor grad_bias({C}, input.dtype(), input.device());

    // F-085 pattern: double-precision scratch accumulators, atomicAdd'd into
    // by every (n, group) block in group_norm_backward_hip_kernel, then
    // narrowed to input.dtype() in a single pass
    // (narrow_group_norm_grad_accum_kernel) once the main kernel completes.
    // Matches CPU's double accumulation and avoids the float32 atomicAdd
    // rounding error compounding over large N.
    Tensor grad_weight_accum({C}, DType::Float64, input.device());
    Tensor grad_bias_accum({C}, DType::Float64, input.device());
    HIP_CHECK(hipMemsetAsync(grad_weight_accum.data<double>(), 0, C * sizeof(double), stream));
    HIP_CHECK(hipMemsetAsync(grad_bias_accum.data<double>(), 0, C * sizeof(double), stream));

    int64_t num_group_instances = N * num_groups;
    int block_size = BLOCK_SIZE;
    int narrow_blocks = static_cast<int>((C + block_size - 1) / block_size);

    // Saved stats from group_norm_forward_with_stats are always Float32.
    // Cast to input dtype for the typed backward kernel.
    Tensor mean_typed = (mean.dtype() == input.dtype()) ? mean : mean.to(input.dtype());
    Tensor rstd_typed = (rstd.dtype() == input.dtype()) ? rstd : rstd.to(input.dtype());

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(group_norm_backward_hip_kernel<float>,
            dim3(num_group_instances), dim3(block_size), 0, stream,
            grad_output.data<float>(), input.data<float>(),
            weight ? weight->data<float>() : nullptr,
            mean_typed.data<float>(), rstd_typed.data<float>(),
            grad_input.data<float>(), grad_weight_accum.data<double>(), grad_bias_accum.data<double>(),
            N, C, HW, num_groups, channels_per_group);
        HIP_POST_LAUNCH_CHECK();
        hipLaunchKernelGGL(narrow_group_norm_grad_accum_kernel<float>,
            dim3(narrow_blocks), dim3(block_size), 0, stream,
            grad_weight_accum.data<double>(), grad_bias_accum.data<double>(),
            grad_weight.data<float>(), grad_bias.data<float>(), C);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(group_norm_backward_hip_kernel<double>,
            dim3(num_group_instances), dim3(block_size), 0, stream,
            grad_output.data<double>(), input.data<double>(),
            weight ? weight->data<double>() : nullptr,
            mean_typed.data<double>(), rstd_typed.data<double>(),
            grad_input.data<double>(), grad_weight_accum.data<double>(), grad_bias_accum.data<double>(),
            N, C, HW, num_groups, channels_per_group);
        HIP_POST_LAUNCH_CHECK();
        hipLaunchKernelGGL(narrow_group_norm_grad_accum_kernel<double>,
            dim3(narrow_blocks), dim3(block_size), 0, stream,
            grad_weight_accum.data<double>(), grad_bias_accum.data<double>(),
            grad_weight.data<double>(), grad_bias.data<double>(), C);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        // Mixed precision: compute in Float32, convert back
        Tensor grad_out_f32 = grad_output.to(DType::Float32);
        Tensor input_f32 = input.to(DType::Float32);
        Tensor mean_f32 = mean.to(DType::Float32);
        Tensor rstd_f32 = rstd.to(DType::Float32);
        Tensor weight_f32;
        const Tensor* weight_f32_ptr = nullptr;
        if (weight) { weight_f32 = weight->to(DType::Float32); weight_f32_ptr = &weight_f32; }

        auto [gi, gw, gb] = group_norm_backward_kernel(
            grad_out_f32, input_f32, mean_f32, rstd_f32, num_groups, weight_f32_ptr, stream);
        return {gi.to(input.dtype()), gw.to(input.dtype()), gb.to(input.dtype())};
    } else {
        throw std::runtime_error("group_norm_backward: unsupported dtype");
    }

    return {grad_input, grad_weight, grad_bias};
}

// ==============================================================================
// Instance Normalization Backward (delegates to GroupNorm with groups=C)
// ==============================================================================

auto instance_norm_backward_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor* weight,
    hipStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    int64_t C = input.shape()[1];
    return group_norm_backward_kernel(grad_output, input, mean, rstd, C, weight, stream);
}

} // namespace rocm
} // namespace tenzor
