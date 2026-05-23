/**
 * @file pooling.cu
 * @brief CUDA pooling kernel implementations (fallback when cuDNN is not available)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "cuda_launch_utils.cuh"
#include "../cuda_error.hpp"
#include <array>
#include <stdexcept>
#include <vector>
#include <utility>

namespace tenzor {
namespace cuda {

// ============================================================================
// Device helpers for type-generic load/store
// ============================================================================

__device__ inline float dev_load(const float* p, int64_t i) { return p[i]; }
__device__ inline float dev_load(const double* p, int64_t i) { return static_cast<float>(p[i]); }
__device__ inline float dev_load(const __half* p, int64_t i) { return __half2float(p[i]); }
__device__ inline float dev_load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }

__device__ inline void dev_store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void dev_store(double* p, int64_t i, float v) { p[i] = static_cast<double>(v); }
__device__ inline void dev_store(__half* p, int64_t i, float v) { p[i] = __float2half(v); }
__device__ inline void dev_store(__nv_bfloat16* p, int64_t i, float v) { p[i] = __float2bfloat16(v); }

// audit-2026-05-03 — pool_compute_t<T> = double when T is double, else float.
// Pooling kernels that need Float64 precision (e.g. for autograd gradcheck)
// use dev_load_compute / dev_store_compute below; legacy callers can keep
// using dev_load / dev_store which always go through float.
template<typename T> struct pool_compute { using type = float; };
template<> struct pool_compute<double> { using type = double; };
template<typename T> using pool_compute_t = typename pool_compute<T>::type;

// Native-precision load: returns float for any non-double T (widens half /
// bfloat16), and double for double T.
__device__ inline float  dev_load_compute(const float* p, int64_t i) { return p[i]; }
__device__ inline double dev_load_compute(const double* p, int64_t i) { return p[i]; }
__device__ inline float  dev_load_compute(const __half* p, int64_t i) { return __half2float(p[i]); }
__device__ inline float  dev_load_compute(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }

__device__ inline void dev_store_compute(float* p, int64_t i, float v)  { p[i] = v; }
__device__ inline void dev_store_compute(double* p, int64_t i, double v) { p[i] = v; }
__device__ inline void dev_store_compute(__half* p, int64_t i, float v) { p[i] = __float2half(v); }
__device__ inline void dev_store_compute(__nv_bfloat16* p, int64_t i, float v) { p[i] = __float2bfloat16(v); }

// ============================================================================
// Launch config helpers
// ============================================================================

static inline Tensor create_zeros_cuda(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream = nullptr) {
    Tensor t(shape, dtype, device);
    size_t bytes = t.numel() * dtype_size(dtype);
    cudaMemsetAsync(t.data_ptr(), 0, bytes, stream);
    return t;
}

// ============================================================================
// MaxPool2d Forward Kernel
// ============================================================================

template<typename T>
__global__ void maxpool2d_forward_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t* __restrict__ indices,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t H_out, int64_t W_out,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation
) {
    using Compute = pool_compute_t<T>;
    const int64_t total = N * C * H_out * W_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t c  = (idx / (W_out * H_out)) % C;
        int64_t n  = idx / (W_out * H_out * C);

        int64_t h_start = oh * stride - padding;
        int64_t w_start = ow * stride - padding;

        Compute max_val = Compute(-1e38);
        int64_t max_idx = 0;

        for (int64_t kh = 0; kh < kernel_size; ++kh) {
            for (int64_t kw = 0; kw < kernel_size; ++kw) {
                int64_t h = h_start + kh * dilation;
                int64_t w = w_start + kw * dilation;

                if (h >= 0 && h < H && w >= 0 && w < W) {
                    int64_t in_idx = ((n * C + c) * H + h) * W + w;
                    Compute val = dev_load_compute(input, in_idx);
                    if (val > max_val) {
                        max_val = val;
                        max_idx = h * W + w;
                    }
                }
            }
        }

        dev_store_compute(output, idx, max_val);
        indices[idx] = max_idx;
    }
}

auto maxpool2d_forward_kernel(const Tensor& input, int64_t kernel_size,
                               int64_t stride, int64_t padding, int64_t dilation,
                               cudaStream_t stream) -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    int64_t H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());
    Tensor indices({N, C, H_out, W_out}, DType::Int64, input.device());

    int64_t total = N * C * H_out * W_out;

    if (input.dtype() == DType::Float32) {
        auto [grid, block] = optimal_launch_config(maxpool2d_forward_impl<float>, total);
        maxpool2d_forward_impl<float><<<grid, block, 0, stream>>>(
            input.data<float>(), output.data<float>(), indices.data<int64_t>(),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding, dilation);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        auto [grid, block] = optimal_launch_config(maxpool2d_forward_impl<double>, total);
        maxpool2d_forward_impl<double><<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(), indices.data<int64_t>(),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding, dilation);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        auto [grid, block] = optimal_launch_config(maxpool2d_forward_impl<__half>, total);
        maxpool2d_forward_impl<__half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            indices.data<int64_t>(),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding, dilation);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto [grid, block] = optimal_launch_config(maxpool2d_forward_impl<__nv_bfloat16>, total);
        maxpool2d_forward_impl<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data<BFloat16>()),
            indices.data<int64_t>(),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding, dilation);
            CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("maxpool2d_forward_kernel: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return {output, indices};
}

// ============================================================================
// MaxPool2d Backward Kernel (float32 accumulation)
// ============================================================================

__global__ void maxpool2d_backward_f32(
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    float* __restrict__ grad_input,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t H_out, int64_t W_out
) {
    const int64_t total = N * C * H_out * W_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t c = (idx / (W_out * H_out)) % C;
        int64_t n = idx / (W_out * H_out * C);

        int64_t max_idx = indices[idx];
        int64_t h = max_idx / W;
        int64_t w = max_idx % W;
        int64_t in_idx = ((n * C + c) * H + h) * W + w;
        float grad_val = grad_output[idx];

#if __CUDA_ARCH__ >= 700
        // Warp-level pre-reduction: find threads targeting the same in_idx
        // and sum their values before atomicAdd, reducing contention.
        // Uses low 32 bits of in_idx for matching (safe for tensors < 2B elements).
        unsigned int peers = __match_any_sync(0xFFFFFFFF, static_cast<unsigned int>(in_idx));
        int leader = __ffs(peers) - 1;
        int lane = threadIdx.x & 31;

        // All matching threads sum their peer values via shuffle
        float sum = 0.0f;
        unsigned int p = peers;
        while (p) {
            int src = __ffs(p) - 1;
            sum += __shfl_sync(peers, grad_val, src);
            p &= p - 1;
        }

        // Only the leader thread does the atomic write
        if (lane == leader) {
            atomicAdd(&grad_input[in_idx], sum);
        }
#else
        atomicAdd(&grad_input[in_idx], grad_val);
#endif
    }
}

__global__ void maxpool2d_backward_f64(
    const double* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    double* __restrict__ grad_input,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t H_out, int64_t W_out
) {
    const int64_t total = N * C * H_out * W_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t c = (idx / (W_out * H_out)) % C;
        int64_t n = idx / (W_out * H_out * C);

        int64_t max_idx = indices[idx];
        int64_t h = max_idx / W;
        int64_t w = max_idx % W;
        int64_t in_idx = ((n * C + c) * H + h) * W + w;
        double grad_val = grad_output[idx];

#if __CUDA_ARCH__ >= 700
        // Warp-level pre-reduction: find threads targeting the same in_idx
        // and sum their values before atomicAdd, reducing contention.
        // Uses low 32 bits of in_idx for matching (safe for tensors < 2B elements).
        unsigned int peers = __match_any_sync(0xFFFFFFFF, static_cast<unsigned int>(in_idx));
        int leader = __ffs(peers) - 1;
        int lane = threadIdx.x & 31;

        // All matching threads sum their peer values via shuffle
        double sum = 0.0;
        unsigned int p = peers;
        while (p) {
            int src = __ffs(p) - 1;
            sum += __shfl_sync(peers, grad_val, src);
            p &= p - 1;
        }

        // Only the leader thread does the atomic write
        if (lane == leader) {
            atomicAdd(&grad_input[in_idx], sum);
        }
#else
        atomicAdd(&grad_input[in_idx], grad_val);
#endif
    }
}

// Conversion kernel: float -> half types
template<typename T>
__global__ void convert_f32_to(const float* __restrict__ src, T* __restrict__ dst, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        dev_store(dst, idx, src[idx]);
    }
}

// Conversion kernel: half types -> float
template<typename T>
__global__ void convert_to_f32(const T* __restrict__ src, float* __restrict__ dst, int64_t n) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        dst[idx] = dev_load(src, idx);
    }
}

auto maxpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                const std::vector<int64_t>& input_shape,
                                cudaStream_t stream) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = input_shape[2];
    int64_t W = input_shape[3];

    auto grad_shape = grad_output.shape();
    int64_t H_out = grad_shape[2];
    int64_t W_out = grad_shape[3];

    int64_t total_out = N * C * H_out * W_out;
    int64_t total_in = N * C * H * W;

    if (grad_output.dtype() == DType::Float32) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        auto [grid_out, block_out] = optimal_launch_config(maxpool2d_backward_f32, total_out);
        maxpool2d_backward_f32<<<grid_out, block_out, 0, stream>>>(
            grad_output.data<float>(), indices.data<int64_t>(), grad_input.data<float>(),
            N, C, H, W, H_out, W_out);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float64) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float64, grad_output.device(), stream);
        auto [grid_out, block_out] = optimal_launch_config(maxpool2d_backward_f64, total_out);
        maxpool2d_backward_f64<<<grid_out, block_out, 0, stream>>>(
            grad_output.data<double>(), indices.data<int64_t>(), grad_input.data<double>(),
            N, C, H, W, H_out, W_out);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        // Accumulate in float32 then convert back
        Tensor grad_f32 = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        Tensor go_f32({N, C, H_out, W_out}, DType::Float32, grad_output.device());

        // Convert grad_output to float32
        if (grad_output.dtype() == DType::Float16) {
            auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__half>, total_out);
            convert_to_f32<__half><<<grid_conv, block_conv, 0, stream>>>(
                reinterpret_cast<const __half*>(grad_output.data<Float16>()), go_f32.data<float>(), total_out);
                CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__nv_bfloat16>, total_out);
            convert_to_f32<__nv_bfloat16><<<grid_conv, block_conv, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(grad_output.data<BFloat16>()), go_f32.data<float>(), total_out);
                CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaGetLastError());

        // Run backward in float32
        auto [grid_bwd, block_bwd] = optimal_launch_config(maxpool2d_backward_f32, total_out);
        maxpool2d_backward_f32<<<grid_bwd, block_bwd, 0, stream>>>(
            go_f32.data<float>(), indices.data<int64_t>(), grad_f32.data<float>(),
            N, C, H, W, H_out, W_out);
        CUDA_CHECK(cudaGetLastError());

        // Convert result back
        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        if (grad_output.dtype() == DType::Float16) {
            auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__half>, total_in);
            convert_f32_to<__half><<<grid_back, block_back, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__half*>(grad_input.data<Float16>()), total_in);
                CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__nv_bfloat16>, total_in);
            convert_f32_to<__nv_bfloat16><<<grid_back, block_back, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__nv_bfloat16*>(grad_input.data<BFloat16>()), total_in);
                CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    }

    throw std::runtime_error("maxpool2d_backward_kernel: unsupported dtype");
}

// ============================================================================
// AvgPool2d Forward Kernel
// ============================================================================

template<typename T>
__global__ void avgpool2d_forward_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t H_out, int64_t W_out,
    int64_t kernel_size, int64_t stride, int64_t padding
) {
    using Compute = pool_compute_t<T>;
    const int64_t total = N * C * H_out * W_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t c  = (idx / (W_out * H_out)) % C;
        int64_t n  = idx / (W_out * H_out * C);

        int64_t h_start = oh * stride - padding;
        int64_t w_start = ow * stride - padding;

        Compute sum = Compute(0);
        int count = 0;

        for (int64_t kh = 0; kh < kernel_size; ++kh) {
            for (int64_t kw = 0; kw < kernel_size; ++kw) {
                int64_t h = h_start + kh;
                int64_t w = w_start + kw;

                if (h >= 0 && h < H && w >= 0 && w < W) {
                    sum += dev_load_compute(input, ((n * C + c) * H + h) * W + w);
                    count++;
                }
            }
        }

        dev_store_compute(output, idx, sum / count);
    }
}

auto avgpool2d_forward_kernel(const Tensor& input, int64_t kernel_size,
                               int64_t stride, int64_t padding,
                               cudaStream_t stream) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    int64_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t W_out = (W + 2 * padding - kernel_size) / stride + 1;

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());

    int64_t total = N * C * H_out * W_out;

    if (input.dtype() == DType::Float32) {
        auto [grid, block] = optimal_launch_config(avgpool2d_forward_impl<float>, total);
        avgpool2d_forward_impl<float><<<grid, block, 0, stream>>>(
            input.data<float>(), output.data<float>(),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        auto [grid, block] = optimal_launch_config(avgpool2d_forward_impl<double>, total);
        avgpool2d_forward_impl<double><<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        auto [grid, block] = optimal_launch_config(avgpool2d_forward_impl<__half>, total);
        avgpool2d_forward_impl<__half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto [grid, block] = optimal_launch_config(avgpool2d_forward_impl<__nv_bfloat16>, total);
        avgpool2d_forward_impl<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data<BFloat16>()),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding);
            CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("avgpool2d_forward_kernel: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// AvgPool2d Backward Kernel (float32 accumulation)
// ============================================================================

__global__ void avgpool2d_backward_f32(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t H_out, int64_t W_out,
    int64_t kernel_size, int64_t stride, int64_t padding
) {
    const int64_t total = N * C * H_out * W_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t c  = (idx / (W_out * H_out)) % C;
        int64_t n  = idx / (W_out * H_out * C);

        int64_t h_start = oh * stride - padding;
        int64_t w_start = ow * stride - padding;

        int count = 0;
        for (int64_t kh = 0; kh < kernel_size; ++kh) {
            for (int64_t kw = 0; kw < kernel_size; ++kw) {
                int64_t h = h_start + kh;
                int64_t w = w_start + kw;
                if (h >= 0 && h < H && w >= 0 && w < W) count++;
            }
        }

        float grad_val = grad_output[idx] / count;

        for (int64_t kh = 0; kh < kernel_size; ++kh) {
            for (int64_t kw = 0; kw < kernel_size; ++kw) {
                int64_t h = h_start + kh;
                int64_t w = w_start + kw;
                if (h >= 0 && h < H && w >= 0 && w < W) {
                    int64_t in_idx = ((n * C + c) * H + h) * W + w;
                    atomicAdd(&grad_input[in_idx], grad_val);
                }
            }
        }
    }
}

__global__ void avgpool2d_backward_f64(
    const double* __restrict__ grad_output,
    double* __restrict__ grad_input,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t H_out, int64_t W_out,
    int64_t kernel_size, int64_t stride, int64_t padding
) {
    const int64_t total = N * C * H_out * W_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t c  = (idx / (W_out * H_out)) % C;
        int64_t n  = idx / (W_out * H_out * C);

        int64_t h_start = oh * stride - padding;
        int64_t w_start = ow * stride - padding;

        int count = 0;
        for (int64_t kh = 0; kh < kernel_size; ++kh) {
            for (int64_t kw = 0; kw < kernel_size; ++kw) {
                int64_t h = h_start + kh;
                int64_t w = w_start + kw;
                if (h >= 0 && h < H && w >= 0 && w < W) count++;
            }
        }

        double grad_val = grad_output[idx] / count;

        for (int64_t kh = 0; kh < kernel_size; ++kh) {
            for (int64_t kw = 0; kw < kernel_size; ++kw) {
                int64_t h = h_start + kh;
                int64_t w = w_start + kw;
                if (h >= 0 && h < H && w >= 0 && w < W) {
                    int64_t in_idx = ((n * C + c) * H + h) * W + w;
                    atomicAdd(&grad_input[in_idx], grad_val);
                }
            }
        }
    }
}

auto avgpool2d_backward_kernel(const Tensor& grad_output,
                                const std::vector<int64_t>& input_shape,
                                int64_t kernel_size, int64_t stride, int64_t padding,
                                cudaStream_t stream) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = input_shape[2];
    int64_t W = input_shape[3];

    auto grad_shape = grad_output.shape();
    int64_t H_out = grad_shape[2];
    int64_t W_out = grad_shape[3];

    int64_t total_out = N * C * H_out * W_out;
    int64_t total_in = N * C * H * W;

    if (grad_output.dtype() == DType::Float32) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        auto [grid_out, block_out] = optimal_launch_config(avgpool2d_backward_f32, total_out);
        avgpool2d_backward_f32<<<grid_out, block_out, 0, stream>>>(
            grad_output.data<float>(), grad_input.data<float>(),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float64) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float64, grad_output.device(), stream);
        auto [grid_out, block_out] = optimal_launch_config(avgpool2d_backward_f64, total_out);
        avgpool2d_backward_f64<<<grid_out, block_out, 0, stream>>>(
            grad_output.data<double>(), grad_input.data<double>(),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        // Accumulate in float32 then convert back
        Tensor grad_f32 = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        Tensor go_f32({N, C, H_out, W_out}, DType::Float32, grad_output.device());

        if (grad_output.dtype() == DType::Float16) {
            auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__half>, total_out);
            convert_to_f32<__half><<<grid_conv, block_conv, 0, stream>>>(
                reinterpret_cast<const __half*>(grad_output.data<Float16>()), go_f32.data<float>(), total_out);
                CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__nv_bfloat16>, total_out);
            convert_to_f32<__nv_bfloat16><<<grid_conv, block_conv, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(grad_output.data<BFloat16>()), go_f32.data<float>(), total_out);
                CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaGetLastError());

        auto [grid_bwd, block_bwd] = optimal_launch_config(avgpool2d_backward_f32, total_out);
        avgpool2d_backward_f32<<<grid_bwd, block_bwd, 0, stream>>>(
            go_f32.data<float>(), grad_f32.data<float>(),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());

        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        if (grad_output.dtype() == DType::Float16) {
            auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__half>, total_in);
            convert_f32_to<__half><<<grid_back, block_back, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__half*>(grad_input.data<Float16>()), total_in);
                CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__nv_bfloat16>, total_in);
            convert_f32_to<__nv_bfloat16><<<grid_back, block_back, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__nv_bfloat16*>(grad_input.data<BFloat16>()), total_in);
                CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    }

    throw std::runtime_error("avgpool2d_backward_kernel: unsupported dtype");
}

// ============================================================================
// MaxPool1d Forward Kernel
// ============================================================================

template<typename T>
__global__ void maxpool1d_forward_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t* __restrict__ indices,
    int64_t N, int64_t C, int64_t L,
    int64_t L_out,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation
) {
    using Compute = pool_compute_t<T>;
    const int64_t total = N * C * L_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = ol * stride - padding;

        Compute max_val = Compute(-1e38);
        int64_t max_idx = 0;

        for (int64_t k = 0; k < kernel_size; ++k) {
            int64_t l = l_start + k * dilation;
            if (l >= 0 && l < L) {
                int64_t in_idx = (n * C + c) * L + l;
                Compute val = dev_load_compute(input, in_idx);
                if (val > max_val) {
                    max_val = val;
                    max_idx = l;
                }
            }
        }

        dev_store_compute(output, idx, max_val);
        indices[idx] = max_idx;
    }
}

auto maxpool1d_forward_kernel(const Tensor& input, std::array<int64_t, 1> kernel_size_a,
                               std::array<int64_t, 1> stride_a, std::array<int64_t, 1> padding_a, std::array<int64_t, 1> dilation_a,
                               cudaStream_t stream) -> std::pair<Tensor, Tensor> {
    // Q.5: per-axis std::array<int64_t, 1> signature. 1D has a single spatial
    // axis (W), so destructure to scalars for the existing impl.
    const int64_t kernel_size = kernel_size_a[0];
    const int64_t stride      = stride_a[0];
    const int64_t padding     = padding_a[0];
    const int64_t dilation    = dilation_a[0];
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t L = shape[2];

    int64_t L_out = (L + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    Tensor output({N, C, L_out}, input.dtype(), input.device());
    Tensor indices({N, C, L_out}, DType::Int64, input.device());

    int64_t total = N * C * L_out;

    if (input.dtype() == DType::Float32) {
        auto [grid, block] = optimal_launch_config(maxpool1d_forward_impl<float>, total);
        maxpool1d_forward_impl<float><<<grid, block, 0, stream>>>(
            input.data<float>(), output.data<float>(), indices.data<int64_t>(),
            N, C, L, L_out, kernel_size, stride, padding, dilation);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        auto [grid, block] = optimal_launch_config(maxpool1d_forward_impl<double>, total);
        maxpool1d_forward_impl<double><<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(), indices.data<int64_t>(),
            N, C, L, L_out, kernel_size, stride, padding, dilation);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        auto [grid, block] = optimal_launch_config(maxpool1d_forward_impl<__half>, total);
        maxpool1d_forward_impl<__half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            indices.data<int64_t>(),
            N, C, L, L_out, kernel_size, stride, padding, dilation);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto [grid, block] = optimal_launch_config(maxpool1d_forward_impl<__nv_bfloat16>, total);
        maxpool1d_forward_impl<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data<BFloat16>()),
            indices.data<int64_t>(),
            N, C, L, L_out, kernel_size, stride, padding, dilation);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("maxpool1d_forward_kernel: unsupported dtype");
    }

    return {output, indices};
}

// ============================================================================
// MaxPool1d Backward Kernel
// ============================================================================

__global__ void maxpool1d_backward_f32(
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    float* __restrict__ grad_input,
    int64_t N, int64_t C, int64_t L,
    int64_t L_out
) {
    const int64_t total = N * C * L_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t c = (idx / L_out) % C;
        int64_t n = idx / (L_out * C);

        int64_t max_idx = indices[idx];
        int64_t in_idx = (n * C + c) * L + max_idx;
        float grad_val = grad_output[idx];

#if __CUDA_ARCH__ >= 700
        unsigned int peers = __match_any_sync(0xFFFFFFFF, static_cast<unsigned int>(in_idx));
        int leader = __ffs(peers) - 1;
        int lane = threadIdx.x & 31;

        float sum = 0.0f;
        unsigned int p = peers;
        while (p) {
            int src = __ffs(p) - 1;
            sum += __shfl_sync(peers, grad_val, src);
            p &= p - 1;
        }

        if (lane == leader) {
            atomicAdd(&grad_input[in_idx], sum);
        }
#else
        atomicAdd(&grad_input[in_idx], grad_val);
#endif
    }
}

__global__ void maxpool1d_backward_f64(
    const double* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    double* __restrict__ grad_input,
    int64_t N, int64_t C, int64_t L,
    int64_t L_out
) {
    const int64_t total = N * C * L_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t c = (idx / L_out) % C;
        int64_t n = idx / (L_out * C);

        int64_t max_idx = indices[idx];
        int64_t in_idx = (n * C + c) * L + max_idx;
        double grad_val = grad_output[idx];

#if __CUDA_ARCH__ >= 700
        unsigned int peers = __match_any_sync(0xFFFFFFFF, static_cast<unsigned int>(in_idx));
        int leader = __ffs(peers) - 1;
        int lane = threadIdx.x & 31;

        double sum = 0.0;
        unsigned int p = peers;
        while (p) {
            int src = __ffs(p) - 1;
            sum += __shfl_sync(peers, grad_val, src);
            p &= p - 1;
        }

        if (lane == leader) {
            atomicAdd(&grad_input[in_idx], sum);
        }
#else
        atomicAdd(&grad_input[in_idx], grad_val);
#endif
    }
}

auto maxpool1d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                const std::vector<int64_t>& input_shape,
                                cudaStream_t stream) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t L = input_shape[2];

    auto grad_shape = grad_output.shape();
    int64_t L_out = grad_shape[2];

    int64_t total_out = N * C * L_out;
    int64_t total_in = N * C * L;

    if (grad_output.dtype() == DType::Float32) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        auto [grid_out, block_out] = optimal_launch_config(maxpool1d_backward_f32, total_out);
        maxpool1d_backward_f32<<<grid_out, block_out, 0, stream>>>(
            grad_output.data<float>(), indices.data<int64_t>(), grad_input.data<float>(),
            N, C, L, L_out);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float64) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float64, grad_output.device(), stream);
        auto [grid_out, block_out] = optimal_launch_config(maxpool1d_backward_f64, total_out);
        maxpool1d_backward_f64<<<grid_out, block_out, 0, stream>>>(
            grad_output.data<double>(), indices.data<int64_t>(), grad_input.data<double>(),
            N, C, L, L_out);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        Tensor grad_f32 = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        Tensor go_f32({N, C, L_out}, DType::Float32, grad_output.device());

        if (grad_output.dtype() == DType::Float16) {
            auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__half>, total_out);
            convert_to_f32<__half><<<grid_conv, block_conv, 0, stream>>>(
                reinterpret_cast<const __half*>(grad_output.data<Float16>()), go_f32.data<float>(), total_out);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__nv_bfloat16>, total_out);
            convert_to_f32<__nv_bfloat16><<<grid_conv, block_conv, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(grad_output.data<BFloat16>()), go_f32.data<float>(), total_out);
            CUDA_CHECK(cudaGetLastError());
        }

        auto [grid_bwd, block_bwd] = optimal_launch_config(maxpool1d_backward_f32, total_out);
        maxpool1d_backward_f32<<<grid_bwd, block_bwd, 0, stream>>>(
            go_f32.data<float>(), indices.data<int64_t>(), grad_f32.data<float>(),
            N, C, L, L_out);
        CUDA_CHECK(cudaGetLastError());

        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        if (grad_output.dtype() == DType::Float16) {
            auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__half>, total_in);
            convert_f32_to<__half><<<grid_back, block_back, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__half*>(grad_input.data<Float16>()), total_in);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__nv_bfloat16>, total_in);
            convert_f32_to<__nv_bfloat16><<<grid_back, block_back, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__nv_bfloat16*>(grad_input.data<BFloat16>()), total_in);
            CUDA_CHECK(cudaGetLastError());
        }
        return grad_input;
    }

    throw std::runtime_error("maxpool1d_backward_kernel: unsupported dtype");
}

// ============================================================================
// AvgPool1d Forward Kernel
// ============================================================================

template<typename T>
__global__ void avgpool1d_forward_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t N, int64_t C, int64_t L,
    int64_t L_out,
    int64_t kernel_size, int64_t stride, int64_t padding
) {
    using Compute = pool_compute_t<T>;
    const int64_t total = N * C * L_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = ol * stride - padding;

        Compute sum = Compute(0);
        int count = 0;

        for (int64_t k = 0; k < kernel_size; ++k) {
            int64_t l = l_start + k;
            if (l >= 0 && l < L) {
                sum += dev_load_compute(input, (n * C + c) * L + l);
                count++;
            }
        }

        dev_store_compute(output, idx, sum / count);
    }
}

auto avgpool1d_forward_kernel(const Tensor& input, std::array<int64_t, 1> kernel_size_a,
                               std::array<int64_t, 1> stride_a, std::array<int64_t, 1> padding_a,
                               cudaStream_t stream) -> Tensor {
    // Q.5: per-axis std::array<int64_t, 1> signature.
    const int64_t kernel_size = kernel_size_a[0];
    const int64_t stride      = stride_a[0];
    const int64_t padding     = padding_a[0];
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t L = shape[2];

    int64_t L_out = (L + 2 * padding - kernel_size) / stride + 1;

    Tensor output({N, C, L_out}, input.dtype(), input.device());

    int64_t total = N * C * L_out;

    if (input.dtype() == DType::Float32) {
        auto [grid, block] = optimal_launch_config(avgpool1d_forward_impl<float>, total);
        avgpool1d_forward_impl<float><<<grid, block, 0, stream>>>(
            input.data<float>(), output.data<float>(),
            N, C, L, L_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        auto [grid, block] = optimal_launch_config(avgpool1d_forward_impl<double>, total);
        avgpool1d_forward_impl<double><<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(),
            N, C, L, L_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        auto [grid, block] = optimal_launch_config(avgpool1d_forward_impl<__half>, total);
        avgpool1d_forward_impl<__half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            N, C, L, L_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto [grid, block] = optimal_launch_config(avgpool1d_forward_impl<__nv_bfloat16>, total);
        avgpool1d_forward_impl<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data<BFloat16>()),
            N, C, L, L_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("avgpool1d_forward_kernel: unsupported dtype");
    }

    return output;
}

// ============================================================================
// AvgPool1d Backward Kernel
// ============================================================================

__global__ void avgpool1d_backward_f32(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int64_t N, int64_t C, int64_t L,
    int64_t L_out,
    int64_t kernel_size, int64_t stride, int64_t padding
) {
    const int64_t total = N * C * L_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = ol * stride - padding;

        int count = 0;
        for (int64_t k = 0; k < kernel_size; ++k) {
            int64_t l = l_start + k;
            if (l >= 0 && l < L) count++;
        }

        float grad_val = grad_output[idx] / count;

        for (int64_t k = 0; k < kernel_size; ++k) {
            int64_t l = l_start + k;
            if (l >= 0 && l < L) {
                int64_t in_idx = (n * C + c) * L + l;
                atomicAdd(&grad_input[in_idx], grad_val);
            }
        }
    }
}

__global__ void avgpool1d_backward_f64(
    const double* __restrict__ grad_output,
    double* __restrict__ grad_input,
    int64_t N, int64_t C, int64_t L,
    int64_t L_out,
    int64_t kernel_size, int64_t stride, int64_t padding
) {
    const int64_t total = N * C * L_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = ol * stride - padding;

        int count = 0;
        for (int64_t k = 0; k < kernel_size; ++k) {
            int64_t l = l_start + k;
            if (l >= 0 && l < L) count++;
        }

        double grad_val = grad_output[idx] / count;

        for (int64_t k = 0; k < kernel_size; ++k) {
            int64_t l = l_start + k;
            if (l >= 0 && l < L) {
                int64_t in_idx = (n * C + c) * L + l;
                atomicAdd(&grad_input[in_idx], grad_val);
            }
        }
    }
}

auto avgpool1d_backward_kernel(const Tensor& grad_output,
                                const std::vector<int64_t>& input_shape,
                                std::array<int64_t, 1> kernel_size_a, std::array<int64_t, 1> stride_a, std::array<int64_t, 1> padding_a,
                                cudaStream_t stream) -> Tensor {
    // Q.5: per-axis std::array<int64_t, 1> signature.
    const int64_t kernel_size = kernel_size_a[0];
    const int64_t stride      = stride_a[0];
    const int64_t padding     = padding_a[0];
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t L = input_shape[2];

    auto grad_shape = grad_output.shape();
    int64_t L_out = grad_shape[2];

    int64_t total_out = N * C * L_out;
    int64_t total_in = N * C * L;

    if (grad_output.dtype() == DType::Float32) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        auto [grid_out, block_out] = optimal_launch_config(avgpool1d_backward_f32, total_out);
        avgpool1d_backward_f32<<<grid_out, block_out, 0, stream>>>(
            grad_output.data<float>(), grad_input.data<float>(),
            N, C, L, L_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float64) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float64, grad_output.device(), stream);
        auto [grid_out, block_out] = optimal_launch_config(avgpool1d_backward_f64, total_out);
        avgpool1d_backward_f64<<<grid_out, block_out, 0, stream>>>(
            grad_output.data<double>(), grad_input.data<double>(),
            N, C, L, L_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        Tensor grad_f32 = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        Tensor go_f32({N, C, L_out}, DType::Float32, grad_output.device());

        if (grad_output.dtype() == DType::Float16) {
            auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__half>, total_out);
            convert_to_f32<__half><<<grid_conv, block_conv, 0, stream>>>(
                reinterpret_cast<const __half*>(grad_output.data<Float16>()), go_f32.data<float>(), total_out);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__nv_bfloat16>, total_out);
            convert_to_f32<__nv_bfloat16><<<grid_conv, block_conv, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(grad_output.data<BFloat16>()), go_f32.data<float>(), total_out);
            CUDA_CHECK(cudaGetLastError());
        }

        auto [grid_bwd, block_bwd] = optimal_launch_config(avgpool1d_backward_f32, total_out);
        avgpool1d_backward_f32<<<grid_bwd, block_bwd, 0, stream>>>(
            go_f32.data<float>(), grad_f32.data<float>(),
            N, C, L, L_out, kernel_size, stride, padding);
        CUDA_CHECK(cudaGetLastError());

        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        if (grad_output.dtype() == DType::Float16) {
            auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__half>, total_in);
            convert_f32_to<__half><<<grid_back, block_back, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__half*>(grad_input.data<Float16>()), total_in);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__nv_bfloat16>, total_in);
            convert_f32_to<__nv_bfloat16><<<grid_back, block_back, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__nv_bfloat16*>(grad_input.data<BFloat16>()), total_in);
            CUDA_CHECK(cudaGetLastError());
        }
        return grad_input;
    }

    throw std::runtime_error("avgpool1d_backward_kernel: unsupported dtype");
}

// ============================================================================
// Adaptive MaxPool1d Forward Kernel
// ============================================================================

template<typename T>
__global__ void adaptive_maxpool1d_forward_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t* __restrict__ indices,
    int64_t N, int64_t C, int64_t L_in, int64_t L_out
) {
    using Compute = pool_compute_t<T>;
    const int64_t total = N * C * L_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = (ol * L_in) / L_out;
        int64_t l_end   = ((ol + 1) * L_in) / L_out;

        Compute max_val = Compute(-1e38);
        int64_t max_idx = l_start;

        for (int64_t l = l_start; l < l_end; ++l) {
            int64_t in_idx = (n * C + c) * L_in + l;
            Compute val = dev_load_compute(input, in_idx);
            if (val > max_val) {
                max_val = val;
                max_idx = l;
            }
        }

        dev_store_compute(output, idx, max_val);
        indices[idx] = max_idx;
    }
}

auto adaptive_maxpool1d_forward(const Tensor& input, int64_t output_size,
                                 cudaStream_t stream) -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], L_in = shape[2];

    Tensor output({N, C, output_size}, input.dtype(), input.device());
    Tensor indices({N, C, output_size}, DType::Int64, input.device());

    int64_t total = N * C * output_size;

    if (input.dtype() == DType::Float32) {
        auto [grid, block] = optimal_launch_config(adaptive_maxpool1d_forward_impl<float>, total);
        adaptive_maxpool1d_forward_impl<float><<<grid, block, 0, stream>>>(
            input.data<float>(), output.data<float>(), indices.data<int64_t>(),
            N, C, L_in, output_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        auto [grid, block] = optimal_launch_config(adaptive_maxpool1d_forward_impl<double>, total);
        adaptive_maxpool1d_forward_impl<double><<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(), indices.data<int64_t>(),
            N, C, L_in, output_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        auto [grid, block] = optimal_launch_config(adaptive_maxpool1d_forward_impl<__half>, total);
        adaptive_maxpool1d_forward_impl<__half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            indices.data<int64_t>(), N, C, L_in, output_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto [grid, block] = optimal_launch_config(adaptive_maxpool1d_forward_impl<__nv_bfloat16>, total);
        adaptive_maxpool1d_forward_impl<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data<BFloat16>()),
            indices.data<int64_t>(), N, C, L_in, output_size);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("adaptive_maxpool1d_forward: unsupported dtype");
    }

    return {output, indices};
}

// ============================================================================
// Adaptive MaxPool1d Backward Kernel
// ============================================================================

auto adaptive_maxpool1d_backward(const Tensor& grad_output, const Tensor& indices,
                                  const std::vector<int64_t>& input_shape,
                                  cudaStream_t stream) -> Tensor {
    // Reuse the maxpool1d backward kernel since it uses the same indices-based scatter
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t L = input_shape[2];

    auto grad_shape = grad_output.shape();
    int64_t L_out = grad_shape[2];

    int64_t total_out = N * C * L_out;
    int64_t total_in = N * C * L;

    if (grad_output.dtype() == DType::Float32) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        auto [grid, block] = optimal_launch_config(maxpool1d_backward_f32, total_out);
        maxpool1d_backward_f32<<<grid, block, 0, stream>>>(
            grad_output.data<float>(), indices.data<int64_t>(), grad_input.data<float>(),
            N, C, L, L_out);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float64) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float64, grad_output.device(), stream);
        auto [grid, block] = optimal_launch_config(maxpool1d_backward_f64, total_out);
        maxpool1d_backward_f64<<<grid, block, 0, stream>>>(
            grad_output.data<double>(), indices.data<int64_t>(), grad_input.data<double>(),
            N, C, L, L_out);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        Tensor grad_f32 = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        Tensor go_f32({N, C, L_out}, DType::Float32, grad_output.device());

        if (grad_output.dtype() == DType::Float16) {
            auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__half>, total_out);
            convert_to_f32<__half><<<grid_conv, block_conv, 0, stream>>>(
                reinterpret_cast<const __half*>(grad_output.data<Float16>()), go_f32.data<float>(), total_out);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__nv_bfloat16>, total_out);
            convert_to_f32<__nv_bfloat16><<<grid_conv, block_conv, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(grad_output.data<BFloat16>()), go_f32.data<float>(), total_out);
            CUDA_CHECK(cudaGetLastError());
        }

        auto [grid_bwd, block_bwd] = optimal_launch_config(maxpool1d_backward_f32, total_out);
        maxpool1d_backward_f32<<<grid_bwd, block_bwd, 0, stream>>>(
            go_f32.data<float>(), indices.data<int64_t>(), grad_f32.data<float>(),
            N, C, L, L_out);
        CUDA_CHECK(cudaGetLastError());

        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        if (grad_output.dtype() == DType::Float16) {
            auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__half>, total_in);
            convert_f32_to<__half><<<grid_back, block_back, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__half*>(grad_input.data<Float16>()), total_in);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__nv_bfloat16>, total_in);
            convert_f32_to<__nv_bfloat16><<<grid_back, block_back, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__nv_bfloat16*>(grad_input.data<BFloat16>()), total_in);
            CUDA_CHECK(cudaGetLastError());
        }
        return grad_input;
    }

    throw std::runtime_error("adaptive_maxpool1d_backward: unsupported dtype");
}

// ============================================================================
// Adaptive AvgPool1d Forward Kernel
// ============================================================================

template<typename T>
__global__ void adaptive_avgpool1d_forward_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t N, int64_t C, int64_t L_in, int64_t L_out
) {
    using Compute = pool_compute_t<T>;
    const int64_t total = N * C * L_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = (ol * L_in) / L_out;
        int64_t l_end   = ((ol + 1) * L_in) / L_out;

        Compute sum = Compute(0);
        for (int64_t l = l_start; l < l_end; ++l) {
            sum += dev_load_compute(input, (n * C + c) * L_in + l);
        }

        dev_store_compute(output, idx, sum / (l_end - l_start));
    }
}

auto adaptive_avgpool1d_forward(const Tensor& input, int64_t output_size,
                                 cudaStream_t stream) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], L_in = shape[2];

    Tensor output({N, C, output_size}, input.dtype(), input.device());

    int64_t total = N * C * output_size;

    if (input.dtype() == DType::Float32) {
        auto [grid, block] = optimal_launch_config(adaptive_avgpool1d_forward_impl<float>, total);
        adaptive_avgpool1d_forward_impl<float><<<grid, block, 0, stream>>>(
            input.data<float>(), output.data<float>(), N, C, L_in, output_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        auto [grid, block] = optimal_launch_config(adaptive_avgpool1d_forward_impl<double>, total);
        adaptive_avgpool1d_forward_impl<double><<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(), N, C, L_in, output_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        auto [grid, block] = optimal_launch_config(adaptive_avgpool1d_forward_impl<__half>, total);
        adaptive_avgpool1d_forward_impl<__half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            N, C, L_in, output_size);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto [grid, block] = optimal_launch_config(adaptive_avgpool1d_forward_impl<__nv_bfloat16>, total);
        adaptive_avgpool1d_forward_impl<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data<BFloat16>()),
            N, C, L_in, output_size);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("adaptive_avgpool1d_forward: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Adaptive AvgPool1d Backward Kernel
// ============================================================================

template<typename T>
__global__ void adaptive_avgpool1d_backward_impl(
    const T* __restrict__ grad_output,
    T* __restrict__ grad_input,
    int64_t N, int64_t C, int64_t L_in, int64_t L_out
) {
    using Compute = pool_compute_t<T>;
    const int64_t total = N * C * L_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = (ol * L_in) / L_out;
        int64_t l_end   = ((ol + 1) * L_in) / L_out;

        float grad_val = dev_load(grad_output, idx) / static_cast<float>(l_end - l_start);

        for (int64_t l = l_start; l < l_end; ++l) {
            int64_t in_idx = (n * C + c) * L_in + l;
            atomicAdd(reinterpret_cast<float*>(grad_input) + in_idx, grad_val);
        }
    }
}

auto adaptive_avgpool1d_backward(const Tensor& grad_output,
                                  const std::vector<int64_t>& input_shape,
                                  cudaStream_t stream) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t L_in = input_shape[2];

    auto grad_shape = grad_output.shape();
    int64_t L_out = grad_shape[2];

    int64_t total_out = N * C * L_out;
    int64_t total_in = N * C * L_in;

    if (grad_output.dtype() == DType::Float32) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        auto [grid, block] = optimal_launch_config(adaptive_avgpool1d_backward_impl<float>, total_out);
        adaptive_avgpool1d_backward_impl<float><<<grid, block, 0, stream>>>(
            grad_output.data<float>(), grad_input.data<float>(),
            N, C, L_in, L_out);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float64) {
        // Float64: accumulate in float32 then convert (atomicAdd(double*) is slow)
        Tensor grad_f32 = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        Tensor go_f32({N, C, L_out}, DType::Float32, grad_output.device());

        auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<double>, total_out);
        convert_to_f32<double><<<grid_conv, block_conv, 0, stream>>>(
            grad_output.data<double>(), go_f32.data<float>(), total_out);
        CUDA_CHECK(cudaGetLastError());

        auto [grid, block] = optimal_launch_config(adaptive_avgpool1d_backward_impl<float>, total_out);
        adaptive_avgpool1d_backward_impl<float><<<grid, block, 0, stream>>>(
            go_f32.data<float>(), grad_f32.data<float>(),
            N, C, L_in, L_out);
        CUDA_CHECK(cudaGetLastError());

        Tensor grad_input(input_shape, DType::Float64, grad_output.device());
        auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<double>, total_in);
        convert_f32_to<double><<<grid_back, block_back, 0, stream>>>(
            grad_f32.data<float>(), grad_input.data<double>(), total_in);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        Tensor grad_f32 = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        Tensor go_f32({N, C, L_out}, DType::Float32, grad_output.device());

        if (grad_output.dtype() == DType::Float16) {
            auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__half>, total_out);
            convert_to_f32<__half><<<grid_conv, block_conv, 0, stream>>>(
                reinterpret_cast<const __half*>(grad_output.data<Float16>()), go_f32.data<float>(), total_out);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__nv_bfloat16>, total_out);
            convert_to_f32<__nv_bfloat16><<<grid_conv, block_conv, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(grad_output.data<BFloat16>()), go_f32.data<float>(), total_out);
            CUDA_CHECK(cudaGetLastError());
        }

        auto [grid, block] = optimal_launch_config(adaptive_avgpool1d_backward_impl<float>, total_out);
        adaptive_avgpool1d_backward_impl<float><<<grid, block, 0, stream>>>(
            go_f32.data<float>(), grad_f32.data<float>(),
            N, C, L_in, L_out);
        CUDA_CHECK(cudaGetLastError());

        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        if (grad_output.dtype() == DType::Float16) {
            auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__half>, total_in);
            convert_f32_to<__half><<<grid_back, block_back, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__half*>(grad_input.data<Float16>()), total_in);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__nv_bfloat16>, total_in);
            convert_f32_to<__nv_bfloat16><<<grid_back, block_back, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__nv_bfloat16*>(grad_input.data<BFloat16>()), total_in);
            CUDA_CHECK(cudaGetLastError());
        }
        return grad_input;
    }

    throw std::runtime_error("adaptive_avgpool1d_backward: unsupported dtype");
}

// ============================================================================
// MaxPool3d Forward Kernel
// ============================================================================

template<typename T>
__global__ void maxpool3d_forward_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t* __restrict__ indices,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW
) {
    using Compute = pool_compute_t<T>;
    const int64_t total = N * C * D_out * H_out * W_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = od * sD - pD;
        int64_t h_start = oh * sH - pH;
        int64_t w_start = ow * sW - pW;

        Compute max_val = Compute(-1e38);
        int64_t max_idx = 0;

        for (int64_t kd = 0; kd < kD; ++kd) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t d = d_start + kd;
                    int64_t h = h_start + kh;
                    int64_t w = w_start + kw;

                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                        int64_t in_idx = ((n * C + c) * D + d) * H * W + h * W + w;
                        Compute val = dev_load_compute(input, in_idx);
                        if (val > max_val) {
                            max_val = val;
                            max_idx = d * H * W + h * W + w;
                        }
                    }
                }
            }
        }

        dev_store_compute(output, idx, max_val);
        indices[idx] = max_idx;
    }
}

auto maxpool3d_forward_kernel(const Tensor& input, std::array<int64_t, 3> kernel_size,
                               std::array<int64_t, 3> stride, std::array<int64_t, 3> padding,
                               cudaStream_t stream) -> std::pair<Tensor, Tensor> {
    // Q.5: per-axis std::array<int64_t, 3>. kernel_size/stride/padding now
    // honour asymmetric D/H/W (previously collapsed to a single scalar).
    const int64_t kD = kernel_size[0], kH = kernel_size[1], kW = kernel_size[2];
    const int64_t sD = stride[0],      sH = stride[1],      sW = stride[2];
    const int64_t pD = padding[0],     pH = padding[1],     pW = padding[2];
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t D = shape[2];
    int64_t H = shape[3];
    int64_t W = shape[4];

    int64_t D_out = (D + 2 * pD - kD) / sD + 1;
    int64_t H_out = (H + 2 * pH - kH) / sH + 1;
    int64_t W_out = (W + 2 * pW - kW) / sW + 1;

    Tensor output({N, C, D_out, H_out, W_out}, input.dtype(), input.device());
    Tensor indices({N, C, D_out, H_out, W_out}, DType::Int64, input.device());

    int64_t total = N * C * D_out * H_out * W_out;

    if (input.dtype() == DType::Float32) {
        auto [grid, block] = optimal_launch_config(maxpool3d_forward_impl<float>, total);
        maxpool3d_forward_impl<float><<<grid, block, 0, stream>>>(
            input.data<float>(), output.data<float>(), indices.data<int64_t>(),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        auto [grid, block] = optimal_launch_config(maxpool3d_forward_impl<double>, total);
        maxpool3d_forward_impl<double><<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(), indices.data<int64_t>(),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        auto [grid, block] = optimal_launch_config(maxpool3d_forward_impl<__half>, total);
        maxpool3d_forward_impl<__half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            indices.data<int64_t>(),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto [grid, block] = optimal_launch_config(maxpool3d_forward_impl<__nv_bfloat16>, total);
        maxpool3d_forward_impl<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data<BFloat16>()),
            indices.data<int64_t>(),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("maxpool3d_forward_kernel: unsupported dtype");
    }

    return {output, indices};
}

// ============================================================================
// MaxPool3d Backward Kernel
// ============================================================================

__global__ void maxpool3d_backward_f32(
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    float* __restrict__ grad_input,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t D_out, int64_t H_out, int64_t W_out
) {
    const int64_t total = N * C * D_out * H_out * W_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t c = (idx / (W_out * H_out * D_out)) % C;
        int64_t n = idx / (W_out * H_out * D_out * C);

        int64_t max_idx = indices[idx];
        int64_t in_idx = ((n * C + c) * D * H * W) + max_idx;
        float grad_val = grad_output[idx];

#if __CUDA_ARCH__ >= 700
        unsigned int peers = __match_any_sync(0xFFFFFFFF, static_cast<unsigned int>(in_idx));
        int leader = __ffs(peers) - 1;
        int lane = threadIdx.x & 31;

        float sum = 0.0f;
        unsigned int p = peers;
        while (p) {
            int src = __ffs(p) - 1;
            sum += __shfl_sync(peers, grad_val, src);
            p &= p - 1;
        }

        if (lane == leader) {
            atomicAdd(&grad_input[in_idx], sum);
        }
#else
        atomicAdd(&grad_input[in_idx], grad_val);
#endif
    }
}

__global__ void maxpool3d_backward_f64(
    const double* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    double* __restrict__ grad_input,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t D_out, int64_t H_out, int64_t W_out
) {
    const int64_t total = N * C * D_out * H_out * W_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t c = (idx / (W_out * H_out * D_out)) % C;
        int64_t n = idx / (W_out * H_out * D_out * C);

        int64_t max_idx = indices[idx];
        int64_t in_idx = ((n * C + c) * D * H * W) + max_idx;
        double grad_val = grad_output[idx];

#if __CUDA_ARCH__ >= 700
        unsigned int peers = __match_any_sync(0xFFFFFFFF, static_cast<unsigned int>(in_idx));
        int leader = __ffs(peers) - 1;
        int lane = threadIdx.x & 31;

        double sum = 0.0;
        unsigned int p = peers;
        while (p) {
            int src = __ffs(p) - 1;
            sum += __shfl_sync(peers, grad_val, src);
            p &= p - 1;
        }

        if (lane == leader) {
            atomicAdd(&grad_input[in_idx], sum);
        }
#else
        atomicAdd(&grad_input[in_idx], grad_val);
#endif
    }
}

auto maxpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                const std::vector<int64_t>& input_shape,
                                cudaStream_t stream) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t D = input_shape[2];
    int64_t H = input_shape[3];
    int64_t W = input_shape[4];

    auto grad_shape = grad_output.shape();
    int64_t D_out = grad_shape[2];
    int64_t H_out = grad_shape[3];
    int64_t W_out = grad_shape[4];

    int64_t total_out = N * C * D_out * H_out * W_out;
    int64_t total_in = N * C * D * H * W;

    if (grad_output.dtype() == DType::Float32) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        auto [grid, block] = optimal_launch_config(maxpool3d_backward_f32, total_out);
        maxpool3d_backward_f32<<<grid, block, 0, stream>>>(
            grad_output.data<float>(), indices.data<int64_t>(), grad_input.data<float>(),
            N, C, D, H, W, D_out, H_out, W_out);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float64) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float64, grad_output.device(), stream);
        auto [grid, block] = optimal_launch_config(maxpool3d_backward_f64, total_out);
        maxpool3d_backward_f64<<<grid, block, 0, stream>>>(
            grad_output.data<double>(), indices.data<int64_t>(), grad_input.data<double>(),
            N, C, D, H, W, D_out, H_out, W_out);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        Tensor grad_f32 = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        Tensor go_f32({N, C, D_out, H_out, W_out}, DType::Float32, grad_output.device());

        if (grad_output.dtype() == DType::Float16) {
            auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__half>, total_out);
            convert_to_f32<__half><<<grid_conv, block_conv, 0, stream>>>(
                reinterpret_cast<const __half*>(grad_output.data<Float16>()), go_f32.data<float>(), total_out);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__nv_bfloat16>, total_out);
            convert_to_f32<__nv_bfloat16><<<grid_conv, block_conv, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(grad_output.data<BFloat16>()), go_f32.data<float>(), total_out);
            CUDA_CHECK(cudaGetLastError());
        }

        auto [grid_bwd, block_bwd] = optimal_launch_config(maxpool3d_backward_f32, total_out);
        maxpool3d_backward_f32<<<grid_bwd, block_bwd, 0, stream>>>(
            go_f32.data<float>(), indices.data<int64_t>(), grad_f32.data<float>(),
            N, C, D, H, W, D_out, H_out, W_out);
        CUDA_CHECK(cudaGetLastError());

        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        if (grad_output.dtype() == DType::Float16) {
            auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__half>, total_in);
            convert_f32_to<__half><<<grid_back, block_back, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__half*>(grad_input.data<Float16>()), total_in);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__nv_bfloat16>, total_in);
            convert_f32_to<__nv_bfloat16><<<grid_back, block_back, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__nv_bfloat16*>(grad_input.data<BFloat16>()), total_in);
            CUDA_CHECK(cudaGetLastError());
        }
        return grad_input;
    }

    throw std::runtime_error("maxpool3d_backward_kernel: unsupported dtype");
}

// ============================================================================
// AvgPool3d Forward Kernel
// ============================================================================

template<typename T>
__global__ void avgpool3d_forward_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW
) {
    using Compute = pool_compute_t<T>;
    const int64_t total = N * C * D_out * H_out * W_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = od * sD - pD;
        int64_t h_start = oh * sH - pH;
        int64_t w_start = ow * sW - pW;

        Compute sum = Compute(0);
        int count = 0;

        for (int64_t kd = 0; kd < kD; ++kd) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t d = d_start + kd;
                    int64_t h = h_start + kh;
                    int64_t w = w_start + kw;

                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                        sum += dev_load_compute(input, ((n * C + c) * D + d) * H * W + h * W + w);
                        count++;
                    }
                }
            }
        }

        dev_store_compute(output, idx, sum / static_cast<Compute>(count));
    }
}

auto avgpool3d_forward_kernel(const Tensor& input, std::array<int64_t, 3> kernel_size,
                               std::array<int64_t, 3> stride, std::array<int64_t, 3> padding,
                               cudaStream_t stream) -> Tensor {
    // Q.5: per-axis std::array<int64_t, 3>.
    const int64_t kD = kernel_size[0], kH = kernel_size[1], kW = kernel_size[2];
    const int64_t sD = stride[0],      sH = stride[1],      sW = stride[2];
    const int64_t pD = padding[0],     pH = padding[1],     pW = padding[2];
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t D = shape[2];
    int64_t H = shape[3];
    int64_t W = shape[4];

    int64_t D_out = (D + 2 * pD - kD) / sD + 1;
    int64_t H_out = (H + 2 * pH - kH) / sH + 1;
    int64_t W_out = (W + 2 * pW - kW) / sW + 1;

    Tensor output({N, C, D_out, H_out, W_out}, input.dtype(), input.device());

    int64_t total = N * C * D_out * H_out * W_out;

    if (input.dtype() == DType::Float32) {
        auto [grid, block] = optimal_launch_config(avgpool3d_forward_impl<float>, total);
        avgpool3d_forward_impl<float><<<grid, block, 0, stream>>>(
            input.data<float>(), output.data<float>(),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        auto [grid, block] = optimal_launch_config(avgpool3d_forward_impl<double>, total);
        avgpool3d_forward_impl<double><<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        auto [grid, block] = optimal_launch_config(avgpool3d_forward_impl<__half>, total);
        avgpool3d_forward_impl<__half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto [grid, block] = optimal_launch_config(avgpool3d_forward_impl<__nv_bfloat16>, total);
        avgpool3d_forward_impl<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data<BFloat16>()),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("avgpool3d_forward_kernel: unsupported dtype");
    }

    return output;
}

// ============================================================================
// AvgPool3d Backward Kernel
// ============================================================================

__global__ void avgpool3d_backward_f32(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW
) {
    const int64_t total = N * C * D_out * H_out * W_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = od * sD - pD;
        int64_t h_start = oh * sH - pH;
        int64_t w_start = ow * sW - pW;

        int count = 0;
        for (int64_t kd = 0; kd < kD; ++kd) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t d = d_start + kd;
                    int64_t h = h_start + kh;
                    int64_t w = w_start + kw;
                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) count++;
                }
            }
        }

        float grad_val = grad_output[idx] / count;

        for (int64_t kd = 0; kd < kD; ++kd) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t d = d_start + kd;
                    int64_t h = h_start + kh;
                    int64_t w = w_start + kw;
                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                        int64_t in_idx = ((n * C + c) * D + d) * H * W + h * W + w;
                        atomicAdd(&grad_input[in_idx], grad_val);
                    }
                }
            }
        }
    }
}

__global__ void avgpool3d_backward_f64(
    const double* __restrict__ grad_output,
    double* __restrict__ grad_input,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW
) {
    const int64_t total = N * C * D_out * H_out * W_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = od * sD - pD;
        int64_t h_start = oh * sH - pH;
        int64_t w_start = ow * sW - pW;

        int count = 0;
        for (int64_t kd = 0; kd < kD; ++kd) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t d = d_start + kd;
                    int64_t h = h_start + kh;
                    int64_t w = w_start + kw;
                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) count++;
                }
            }
        }

        double grad_val = grad_output[idx] / count;

        for (int64_t kd = 0; kd < kD; ++kd) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t d = d_start + kd;
                    int64_t h = h_start + kh;
                    int64_t w = w_start + kw;
                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                        int64_t in_idx = ((n * C + c) * D + d) * H * W + h * W + w;
                        atomicAdd(&grad_input[in_idx], grad_val);
                    }
                }
            }
        }
    }
}

auto avgpool3d_backward_kernel(const Tensor& grad_output,
                                const std::vector<int64_t>& input_shape,
                                std::array<int64_t, 3> kernel_size, std::array<int64_t, 3> stride, std::array<int64_t, 3> padding,
                                cudaStream_t stream) -> Tensor {
    // Q.5: per-axis std::array<int64_t, 3>.
    const int64_t kD = kernel_size[0], kH = kernel_size[1], kW = kernel_size[2];
    const int64_t sD = stride[0],      sH = stride[1],      sW = stride[2];
    const int64_t pD = padding[0],     pH = padding[1],     pW = padding[2];
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t D = input_shape[2];
    int64_t H = input_shape[3];
    int64_t W = input_shape[4];

    auto grad_shape = grad_output.shape();
    int64_t D_out = grad_shape[2];
    int64_t H_out = grad_shape[3];
    int64_t W_out = grad_shape[4];

    int64_t total_out = N * C * D_out * H_out * W_out;
    int64_t total_in = N * C * D * H * W;

    if (grad_output.dtype() == DType::Float32) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        auto [grid, block] = optimal_launch_config(avgpool3d_backward_f32, total_out);
        avgpool3d_backward_f32<<<grid, block, 0, stream>>>(
            grad_output.data<float>(), grad_input.data<float>(),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float64) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float64, grad_output.device(), stream);
        auto [grid, block] = optimal_launch_config(avgpool3d_backward_f64, total_out);
        avgpool3d_backward_f64<<<grid, block, 0, stream>>>(
            grad_output.data<double>(), grad_input.data<double>(),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        Tensor grad_f32 = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        Tensor go_f32({N, C, D_out, H_out, W_out}, DType::Float32, grad_output.device());

        if (grad_output.dtype() == DType::Float16) {
            auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__half>, total_out);
            convert_to_f32<__half><<<grid_conv, block_conv, 0, stream>>>(
                reinterpret_cast<const __half*>(grad_output.data<Float16>()), go_f32.data<float>(), total_out);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__nv_bfloat16>, total_out);
            convert_to_f32<__nv_bfloat16><<<grid_conv, block_conv, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(grad_output.data<BFloat16>()), go_f32.data<float>(), total_out);
            CUDA_CHECK(cudaGetLastError());
        }

        auto [grid_bwd, block_bwd] = optimal_launch_config(avgpool3d_backward_f32, total_out);
        avgpool3d_backward_f32<<<grid_bwd, block_bwd, 0, stream>>>(
            go_f32.data<float>(), grad_f32.data<float>(),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW);
        CUDA_CHECK(cudaGetLastError());

        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        if (grad_output.dtype() == DType::Float16) {
            auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__half>, total_in);
            convert_f32_to<__half><<<grid_back, block_back, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__half*>(grad_input.data<Float16>()), total_in);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__nv_bfloat16>, total_in);
            convert_f32_to<__nv_bfloat16><<<grid_back, block_back, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__nv_bfloat16*>(grad_input.data<BFloat16>()), total_in);
            CUDA_CHECK(cudaGetLastError());
        }
        return grad_input;
    }

    throw std::runtime_error("avgpool3d_backward_kernel: unsupported dtype");
}

// ============================================================================
// Adaptive MaxPool3d Forward Kernel
// ============================================================================

template<typename T>
__global__ void adaptive_maxpool3d_forward_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t* __restrict__ indices,
    int64_t N, int64_t C,
    int64_t D_in, int64_t H_in, int64_t W_in,
    int64_t D_out, int64_t H_out, int64_t W_out
) {
    using Compute = pool_compute_t<T>;
    const int64_t total = N * C * D_out * H_out * W_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = (od * D_in) / D_out;
        int64_t d_end   = ((od + 1) * D_in) / D_out;
        int64_t h_start = (oh * H_in) / H_out;
        int64_t h_end   = ((oh + 1) * H_in) / H_out;
        int64_t w_start = (ow * W_in) / W_out;
        int64_t w_end   = ((ow + 1) * W_in) / W_out;

        Compute max_val = Compute(-1e38);
        int64_t max_idx = d_start * H_in * W_in + h_start * W_in + w_start;

        for (int64_t d = d_start; d < d_end; ++d) {
            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    int64_t in_idx = ((n * C + c) * D_in + d) * H_in * W_in + h * W_in + w;
                    Compute val = dev_load_compute(input, in_idx);
                    if (val > max_val) {
                        max_val = val;
                        max_idx = d * H_in * W_in + h * W_in + w;
                    }
                }
            }
        }

        dev_store_compute(output, idx, max_val);
        indices[idx] = max_idx;
    }
}

auto adaptive_maxpool3d_forward(const Tensor& input,
                                 int64_t output_d, int64_t output_h, int64_t output_w,
                                 cudaStream_t stream) -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1];
    int64_t D_in = shape[2], H_in = shape[3], W_in = shape[4];

    Tensor output({N, C, output_d, output_h, output_w}, input.dtype(), input.device());
    Tensor indices({N, C, output_d, output_h, output_w}, DType::Int64, input.device());

    int64_t total = N * C * output_d * output_h * output_w;

    if (input.dtype() == DType::Float32) {
        auto [grid, block] = optimal_launch_config(adaptive_maxpool3d_forward_impl<float>, total);
        adaptive_maxpool3d_forward_impl<float><<<grid, block, 0, stream>>>(
            input.data<float>(), output.data<float>(), indices.data<int64_t>(),
            N, C, D_in, H_in, W_in, output_d, output_h, output_w);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        auto [grid, block] = optimal_launch_config(adaptive_maxpool3d_forward_impl<double>, total);
        adaptive_maxpool3d_forward_impl<double><<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(), indices.data<int64_t>(),
            N, C, D_in, H_in, W_in, output_d, output_h, output_w);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        auto [grid, block] = optimal_launch_config(adaptive_maxpool3d_forward_impl<__half>, total);
        adaptive_maxpool3d_forward_impl<__half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            indices.data<int64_t>(),
            N, C, D_in, H_in, W_in, output_d, output_h, output_w);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto [grid, block] = optimal_launch_config(adaptive_maxpool3d_forward_impl<__nv_bfloat16>, total);
        adaptive_maxpool3d_forward_impl<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data<BFloat16>()),
            indices.data<int64_t>(),
            N, C, D_in, H_in, W_in, output_d, output_h, output_w);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("adaptive_maxpool3d_forward: unsupported dtype");
    }

    return {output, indices};
}

// ============================================================================
// Adaptive MaxPool3d Backward Kernel (reuses maxpool3d backward — same index scatter)
// ============================================================================

auto adaptive_maxpool3d_backward(const Tensor& grad_output, const Tensor& indices,
                                  const std::vector<int64_t>& input_shape,
                                  cudaStream_t stream) -> Tensor {
    // Identical to maxpool3d_backward_kernel — scatter grad by stored indices
    return maxpool3d_backward_kernel(grad_output, indices, input_shape, stream);
}

// ============================================================================
// Adaptive AvgPool3d Forward Kernel
// ============================================================================

template<typename T>
__global__ void adaptive_avgpool3d_forward_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t N, int64_t C,
    int64_t D_in, int64_t H_in, int64_t W_in,
    int64_t D_out, int64_t H_out, int64_t W_out
) {
    using Compute = pool_compute_t<T>;
    const int64_t total = N * C * D_out * H_out * W_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = (od * D_in) / D_out;
        int64_t d_end   = ((od + 1) * D_in) / D_out;
        int64_t h_start = (oh * H_in) / H_out;
        int64_t h_end   = ((oh + 1) * H_in) / H_out;
        int64_t w_start = (ow * W_in) / W_out;
        int64_t w_end   = ((ow + 1) * W_in) / W_out;

        Compute sum = Compute(0);
        int count = 0;

        for (int64_t d = d_start; d < d_end; ++d) {
            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    sum += dev_load_compute(input, ((n * C + c) * D_in + d) * H_in * W_in + h * W_in + w);
                    count++;
                }
            }
        }

        dev_store_compute(output, idx, count > 0 ? sum / count : 0.0f);
    }
}

auto adaptive_avgpool3d_forward(const Tensor& input,
                                 int64_t output_d, int64_t output_h, int64_t output_w,
                                 cudaStream_t stream) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1];
    int64_t D_in = shape[2], H_in = shape[3], W_in = shape[4];

    Tensor output({N, C, output_d, output_h, output_w}, input.dtype(), input.device());

    int64_t total = N * C * output_d * output_h * output_w;

    if (input.dtype() == DType::Float32) {
        auto [grid, block] = optimal_launch_config(adaptive_avgpool3d_forward_impl<float>, total);
        adaptive_avgpool3d_forward_impl<float><<<grid, block, 0, stream>>>(
            input.data<float>(), output.data<float>(),
            N, C, D_in, H_in, W_in, output_d, output_h, output_w);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        auto [grid, block] = optimal_launch_config(adaptive_avgpool3d_forward_impl<double>, total);
        adaptive_avgpool3d_forward_impl<double><<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(),
            N, C, D_in, H_in, W_in, output_d, output_h, output_w);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        auto [grid, block] = optimal_launch_config(adaptive_avgpool3d_forward_impl<__half>, total);
        adaptive_avgpool3d_forward_impl<__half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            N, C, D_in, H_in, W_in, output_d, output_h, output_w);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto [grid, block] = optimal_launch_config(adaptive_avgpool3d_forward_impl<__nv_bfloat16>, total);
        adaptive_avgpool3d_forward_impl<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data<BFloat16>()),
            N, C, D_in, H_in, W_in, output_d, output_h, output_w);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("adaptive_avgpool3d_forward: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Adaptive AvgPool3d Backward Kernel
// ============================================================================

template<typename T>
__global__ void adaptive_avgpool3d_backward_impl(
    const T* __restrict__ grad_output,
    T* __restrict__ grad_input,
    int64_t N, int64_t C,
    int64_t D_in, int64_t H_in, int64_t W_in,
    int64_t D_out, int64_t H_out, int64_t W_out
) {
    using Compute = pool_compute_t<T>;
    const int64_t total = N * C * D_out * H_out * W_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = (od * D_in) / D_out;
        int64_t d_end   = ((od + 1) * D_in) / D_out;
        int64_t h_start = (oh * H_in) / H_out;
        int64_t h_end   = ((oh + 1) * H_in) / H_out;
        int64_t w_start = (ow * W_in) / W_out;
        int64_t w_end   = ((ow + 1) * W_in) / W_out;

        int count = static_cast<int>((d_end - d_start) * (h_end - h_start) * (w_end - w_start));
        float grad_val = dev_load(grad_output, idx) / static_cast<float>(count);

        for (int64_t d = d_start; d < d_end; ++d) {
            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    int64_t in_idx = ((n * C + c) * D_in + d) * H_in * W_in + h * W_in + w;
                    atomicAdd(reinterpret_cast<float*>(grad_input) + in_idx, grad_val);
                }
            }
        }
    }
}

auto adaptive_avgpool3d_backward(const Tensor& grad_output,
                                  const std::vector<int64_t>& input_shape,
                                  cudaStream_t stream) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t D_in = input_shape[2];
    int64_t H_in = input_shape[3];
    int64_t W_in = input_shape[4];

    auto grad_shape = grad_output.shape();
    int64_t D_out = grad_shape[2];
    int64_t H_out = grad_shape[3];
    int64_t W_out = grad_shape[4];

    int64_t total_out = N * C * D_out * H_out * W_out;
    int64_t total_in = N * C * D_in * H_in * W_in;

    if (grad_output.dtype() == DType::Float32) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        auto [grid, block] = optimal_launch_config(adaptive_avgpool3d_backward_impl<float>, total_out);
        adaptive_avgpool3d_backward_impl<float><<<grid, block, 0, stream>>>(
            grad_output.data<float>(), grad_input.data<float>(),
            N, C, D_in, H_in, W_in, D_out, H_out, W_out);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float64) {
        Tensor grad_f32 = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        Tensor go_f32({N, C, D_out, H_out, W_out}, DType::Float32, grad_output.device());

        auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<double>, total_out);
        convert_to_f32<double><<<grid_conv, block_conv, 0, stream>>>(
            grad_output.data<double>(), go_f32.data<float>(), total_out);
        CUDA_CHECK(cudaGetLastError());

        auto [grid, block] = optimal_launch_config(adaptive_avgpool3d_backward_impl<float>, total_out);
        adaptive_avgpool3d_backward_impl<float><<<grid, block, 0, stream>>>(
            go_f32.data<float>(), grad_f32.data<float>(),
            N, C, D_in, H_in, W_in, D_out, H_out, W_out);
        CUDA_CHECK(cudaGetLastError());

        Tensor grad_input(input_shape, DType::Float64, grad_output.device());
        auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<double>, total_in);
        convert_f32_to<double><<<grid_back, block_back, 0, stream>>>(
            grad_f32.data<float>(), grad_input.data<double>(), total_in);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        Tensor grad_f32 = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);
        Tensor go_f32({N, C, D_out, H_out, W_out}, DType::Float32, grad_output.device());

        if (grad_output.dtype() == DType::Float16) {
            auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__half>, total_out);
            convert_to_f32<__half><<<grid_conv, block_conv, 0, stream>>>(
                reinterpret_cast<const __half*>(grad_output.data<Float16>()), go_f32.data<float>(), total_out);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__nv_bfloat16>, total_out);
            convert_to_f32<__nv_bfloat16><<<grid_conv, block_conv, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(grad_output.data<BFloat16>()), go_f32.data<float>(), total_out);
            CUDA_CHECK(cudaGetLastError());
        }

        auto [grid, block] = optimal_launch_config(adaptive_avgpool3d_backward_impl<float>, total_out);
        adaptive_avgpool3d_backward_impl<float><<<grid, block, 0, stream>>>(
            go_f32.data<float>(), grad_f32.data<float>(),
            N, C, D_in, H_in, W_in, D_out, H_out, W_out);
        CUDA_CHECK(cudaGetLastError());

        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        if (grad_output.dtype() == DType::Float16) {
            auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__half>, total_in);
            convert_f32_to<__half><<<grid_back, block_back, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__half*>(grad_input.data<Float16>()), total_in);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__nv_bfloat16>, total_in);
            convert_f32_to<__nv_bfloat16><<<grid_back, block_back, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__nv_bfloat16*>(grad_input.data<BFloat16>()), total_in);
            CUDA_CHECK(cudaGetLastError());
        }
        return grad_input;
    }

    throw std::runtime_error("adaptive_avgpool3d_backward: unsupported dtype");
}

// ============================================================================
// Fractional Max Pool 2D Forward
// ============================================================================

__global__ void fractional_maxpool2d_forward_impl(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const float* __restrict__ samples,  // may be nullptr
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t out_h, int64_t out_w,
    int64_t total)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ow = idx % out_w;
        int64_t oh = (idx / out_w) % out_h;
        int64_t c  = (idx / (out_w * out_h)) % C;
        int64_t n  = idx / (out_w * out_h * C);

        float sample_h = samples ? samples[(n * C + c) * 2 + 0] : 0.5f;
        float sample_w = samples ? samples[(n * C + c) * 2 + 1] : 0.5f;

        float ratio_h = static_cast<float>(H) / out_h;
        float ratio_w = static_cast<float>(W) / out_w;

        int64_t h_start = static_cast<int64_t>(floorf((oh + sample_h) * ratio_h - sample_h));
        int64_t h_end   = static_cast<int64_t>(floorf((oh + 1 + sample_h) * ratio_h - sample_h));
        int64_t w_start = static_cast<int64_t>(floorf((ow + sample_w) * ratio_w - sample_w));
        int64_t w_end   = static_cast<int64_t>(floorf((ow + 1 + sample_w) * ratio_w - sample_w));

        h_start = max(h_start, int64_t{0}); h_end = min(h_end, H);
        w_start = max(w_start, int64_t{0}); w_end = min(w_end, W);
        if (h_end <= h_start) h_end = min(h_start + 1, H);
        if (w_end <= w_start) w_end = min(w_start + 1, W);

        float max_val = -1e38f;
        int64_t max_idx = h_start * W + w_start;

        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t w = w_start; w < w_end; ++w) {
                int64_t in_idx = ((n * C + c) * H + h) * W + w;
                float val = input[in_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = h * W + w;
                }
            }
        }

        output[idx] = max_val;
        indices[idx] = max_idx;
    }
}

// audit-2026-05-03 — Float64 native fractional_maxpool2d to preserve
// gradient precision through autograd gradcheck.
__global__ void fractional_maxpool2d_forward_impl_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    int64_t* __restrict__ indices,
    const float* __restrict__ samples,  // sample tensor is always Float32
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t out_h, int64_t out_w,
    int64_t total)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ow = idx % out_w;
        int64_t oh = (idx / out_w) % out_h;
        int64_t c  = (idx / (out_w * out_h)) % C;
        int64_t n  = idx / (out_w * out_h * C);

        float sample_h = samples ? samples[(n * C + c) * 2 + 0] : 0.5f;
        float sample_w = samples ? samples[(n * C + c) * 2 + 1] : 0.5f;

        float ratio_h = static_cast<float>(H) / out_h;
        float ratio_w = static_cast<float>(W) / out_w;

        int64_t h_start = static_cast<int64_t>(floorf((oh + sample_h) * ratio_h - sample_h));
        int64_t h_end   = static_cast<int64_t>(floorf((oh + 1 + sample_h) * ratio_h - sample_h));
        int64_t w_start = static_cast<int64_t>(floorf((ow + sample_w) * ratio_w - sample_w));
        int64_t w_end   = static_cast<int64_t>(floorf((ow + 1 + sample_w) * ratio_w - sample_w));

        h_start = max(h_start, int64_t{0}); h_end = min(h_end, H);
        w_start = max(w_start, int64_t{0}); w_end = min(w_end, W);
        if (h_end <= h_start) h_end = min(h_start + 1, H);
        if (w_end <= w_start) w_end = min(w_start + 1, W);

        double max_val = -1e308;
        int64_t max_idx = h_start * W + w_start;

        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t w = w_start; w < w_end; ++w) {
                int64_t in_idx = ((n * C + c) * H + h) * W + w;
                double val = input[in_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = h * W + w;
                }
            }
        }

        output[idx] = max_val;
        indices[idx] = max_idx;
    }
}

auto fractional_maxpool2d_forward_kernel(const Tensor& input,
                                          int64_t out_h, int64_t out_w,
                                          const Tensor* random_samples,
                                          cudaStream_t stream)
    -> std::pair<Tensor, Tensor>
{
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    int64_t total = N * C * out_h * out_w;

    auto output  = Tensor({N, C, out_h, out_w}, DType::Float32, input.device());
    auto idx_out = Tensor({N, C, out_h, out_w}, DType::Int64, input.device());

    const float* samples_ptr = nullptr;
    if (random_samples && random_samples->numel() > 0) {
        samples_ptr = random_samples->data<float>();
    }

    if (input.dtype() == DType::Float32) {
        auto [grid, block] = optimal_launch_config(fractional_maxpool2d_forward_impl, total);
        fractional_maxpool2d_forward_impl<<<grid, block, 0, stream>>>(
            input.data<float>(), output.data<float>(), idx_out.data<int64_t>(),
            samples_ptr, N, C, H, W, out_h, out_w, total);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        // Native Float64 path — preserves precision for autograd gradcheck.
        Tensor out_f64({N, C, out_h, out_w}, DType::Float64, input.device());
        auto [grid, block] = optimal_launch_config(fractional_maxpool2d_forward_impl_f64, total);
        fractional_maxpool2d_forward_impl_f64<<<grid, block, 0, stream>>>(
            input.data<double>(), out_f64.data<double>(), idx_out.data<int64_t>(),
            samples_ptr, N, C, H, W, out_h, out_w, total);
        CUDA_CHECK(cudaGetLastError());
        return {out_f64, idx_out};
    } else if (input.dtype() == DType::Float16) {
        Tensor in_f32 = create_zeros_cuda({N, C, H, W}, DType::Float32, input.device(), stream);
        int64_t in_total = N * C * H * W;
        auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__half>, in_total);
        convert_to_f32<__half><<<grid_conv, block_conv, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()), in_f32.data<float>(), in_total);
        CUDA_CHECK(cudaGetLastError());

        auto [grid, block] = optimal_launch_config(fractional_maxpool2d_forward_impl, total);
        fractional_maxpool2d_forward_impl<<<grid, block, 0, stream>>>(
            in_f32.data<float>(), output.data<float>(), idx_out.data<int64_t>(),
            samples_ptr, N, C, H, W, out_h, out_w, total);
        CUDA_CHECK(cudaGetLastError());

        Tensor out_f16({N, C, out_h, out_w}, DType::Float16, input.device());
        auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__half>, total);
        convert_f32_to<__half><<<grid_back, block_back, 0, stream>>>(
            output.data<float>(), reinterpret_cast<__half*>(out_f16.data<Float16>()), total);
        CUDA_CHECK(cudaGetLastError());
        return {out_f16, idx_out};
    } else if (input.dtype() == DType::BFloat16) {
        Tensor in_f32 = create_zeros_cuda({N, C, H, W}, DType::Float32, input.device(), stream);
        int64_t in_total = N * C * H * W;
        auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__nv_bfloat16>, in_total);
        convert_to_f32<__nv_bfloat16><<<grid_conv, block_conv, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()), in_f32.data<float>(), in_total);
        CUDA_CHECK(cudaGetLastError());

        auto [grid, block] = optimal_launch_config(fractional_maxpool2d_forward_impl, total);
        fractional_maxpool2d_forward_impl<<<grid, block, 0, stream>>>(
            in_f32.data<float>(), output.data<float>(), idx_out.data<int64_t>(),
            samples_ptr, N, C, H, W, out_h, out_w, total);
        CUDA_CHECK(cudaGetLastError());

        Tensor out_bf16({N, C, out_h, out_w}, DType::BFloat16, input.device());
        auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__nv_bfloat16>, total);
        convert_f32_to<__nv_bfloat16><<<grid_back, block_back, 0, stream>>>(
            output.data<float>(), reinterpret_cast<__nv_bfloat16*>(out_bf16.data<BFloat16>()), total);
        CUDA_CHECK(cudaGetLastError());
        return {out_bf16, idx_out};
    } else {
        throw std::runtime_error("fractional_maxpool2d_forward: unsupported dtype");
    }

    return {output, idx_out};
}

// ============================================================================
// Fractional Max Pool 2D Backward
// ============================================================================

__global__ void fractional_maxpool2d_backward_impl(
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    float* __restrict__ grad_input,
    int64_t N, int64_t C,
    int64_t in_spatial, int64_t out_spatial,
    int64_t total)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t nc = idx / out_spatial;

        int64_t max_idx = indices[idx];
        float grad_val = grad_output[idx];

        atomicAdd(&grad_input[nc * in_spatial + max_idx], grad_val);
    }
}

__global__ void fractional_maxpool2d_backward_impl_f64(
    const double* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    double* __restrict__ grad_input,
    int64_t N, int64_t C,
    int64_t in_spatial, int64_t out_spatial,
    int64_t total)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t nc = idx / out_spatial;

        int64_t max_idx = indices[idx];
        double grad_val = grad_output[idx];

        atomicAdd(&grad_input[nc * in_spatial + max_idx], grad_val);
    }
}

auto fractional_maxpool2d_backward_kernel(const Tensor& grad_output,
                                           const Tensor& indices,
                                           const std::vector<int64_t>& input_shape,
                                           cudaStream_t stream)
    -> Tensor
{
    auto grad_shape = grad_output.shape();
    int64_t N = input_shape[0], C = input_shape[1];
    int64_t H = input_shape[2], W = input_shape[3];
    int64_t out_h = grad_shape[2], out_w = grad_shape[3];
    int64_t in_spatial = H * W;
    int64_t out_spatial = out_h * out_w;
    int64_t total = N * C * out_spatial;

    auto grad_input = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);

    if (grad_output.dtype() == DType::Float32) {
        auto [grid, block] = optimal_launch_config(fractional_maxpool2d_backward_impl, total);
        fractional_maxpool2d_backward_impl<<<grid, block, 0, stream>>>(
            grad_output.data<float>(), indices.data<int64_t>(),
            grad_input.data<float>(), N, C, in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float64) {
        // Native Float64 backward.
        Tensor result = create_zeros_cuda(input_shape, DType::Float64, grad_output.device(), stream);
        auto [grid, block] = optimal_launch_config(fractional_maxpool2d_backward_impl_f64, total);
        fractional_maxpool2d_backward_impl_f64<<<grid, block, 0, stream>>>(
            grad_output.data<double>(), indices.data<int64_t>(),
            result.data<double>(), N, C, in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (grad_output.dtype() == DType::Float16) {
        Tensor go_f32 = create_zeros_cuda(std::vector<int64_t>(grad_shape.begin(), grad_shape.end()),
                                          DType::Float32, grad_output.device(), stream);
        auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__half>, total);
        convert_to_f32<__half><<<grid_conv, block_conv, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data<Float16>()), go_f32.data<float>(), total);
        CUDA_CHECK(cudaGetLastError());

        auto [grid, block] = optimal_launch_config(fractional_maxpool2d_backward_impl, total);
        fractional_maxpool2d_backward_impl<<<grid, block, 0, stream>>>(
            go_f32.data<float>(), indices.data<int64_t>(),
            grad_input.data<float>(), N, C, in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());

        Tensor result(input_shape, DType::Float16, grad_output.device());
        int64_t in_total = N * C * in_spatial;
        auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__half>, in_total);
        convert_f32_to<__half><<<grid_back, block_back, 0, stream>>>(
            grad_input.data<float>(), reinterpret_cast<__half*>(result.data<Float16>()), in_total);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (grad_output.dtype() == DType::BFloat16) {
        Tensor go_f32 = create_zeros_cuda(std::vector<int64_t>(grad_shape.begin(), grad_shape.end()),
                                          DType::Float32, grad_output.device(), stream);
        auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__nv_bfloat16>, total);
        convert_to_f32<__nv_bfloat16><<<grid_conv, block_conv, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data<BFloat16>()), go_f32.data<float>(), total);
        CUDA_CHECK(cudaGetLastError());

        auto [grid, block] = optimal_launch_config(fractional_maxpool2d_backward_impl, total);
        fractional_maxpool2d_backward_impl<<<grid, block, 0, stream>>>(
            go_f32.data<float>(), indices.data<int64_t>(),
            grad_input.data<float>(), N, C, in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());

        Tensor result(input_shape, DType::BFloat16, grad_output.device());
        int64_t in_total = N * C * in_spatial;
        auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__nv_bfloat16>, in_total);
        convert_f32_to<__nv_bfloat16><<<grid_back, block_back, 0, stream>>>(
            grad_input.data<float>(), reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), in_total);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

    throw std::runtime_error("fractional_maxpool2d_backward: unsupported dtype");
}

// ============================================================================
// Fractional Max Pool 3D Forward
// ============================================================================

__global__ void fractional_maxpool3d_forward_impl(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const float* __restrict__ samples,
    int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
    int64_t out_d, int64_t out_h, int64_t out_w,
    int64_t total)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ow = idx % out_w;
        int64_t oh = (idx / out_w) % out_h;
        int64_t od = (idx / (out_w * out_h)) % out_d;
        int64_t c  = (idx / (out_w * out_h * out_d)) % C;
        int64_t n  = idx / (out_w * out_h * out_d * C);

        float sample_d = samples ? samples[(n * C + c) * 3 + 0] : 0.5f;
        float sample_h = samples ? samples[(n * C + c) * 3 + 1] : 0.5f;
        float sample_w = samples ? samples[(n * C + c) * 3 + 2] : 0.5f;

        float ratio_d = static_cast<float>(D) / out_d;
        float ratio_h = static_cast<float>(H) / out_h;
        float ratio_w = static_cast<float>(W) / out_w;

        int64_t d_start = static_cast<int64_t>(floorf((od + sample_d) * ratio_d - sample_d));
        int64_t d_end   = static_cast<int64_t>(floorf((od + 1 + sample_d) * ratio_d - sample_d));
        int64_t h_start = static_cast<int64_t>(floorf((oh + sample_h) * ratio_h - sample_h));
        int64_t h_end   = static_cast<int64_t>(floorf((oh + 1 + sample_h) * ratio_h - sample_h));
        int64_t w_start = static_cast<int64_t>(floorf((ow + sample_w) * ratio_w - sample_w));
        int64_t w_end   = static_cast<int64_t>(floorf((ow + 1 + sample_w) * ratio_w - sample_w));

        d_start = max(d_start, int64_t{0}); d_end = min(d_end, D);
        h_start = max(h_start, int64_t{0}); h_end = min(h_end, H);
        w_start = max(w_start, int64_t{0}); w_end = min(w_end, W);
        if (d_end <= d_start) d_end = min(d_start + 1, D);
        if (h_end <= h_start) h_end = min(h_start + 1, H);
        if (w_end <= w_start) w_end = min(w_start + 1, W);

        float max_val = -1e38f;
        int64_t max_idx = d_start * H * W + h_start * W + w_start;

        for (int64_t d = d_start; d < d_end; ++d) {
            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    int64_t in_idx = (((n * C + c) * D + d) * H + h) * W + w;
                    float val = input[in_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = (d * H + h) * W + w;
                    }
                }
            }
        }

        output[idx] = max_val;
        indices[idx] = max_idx;
    }
}

__global__ void fractional_maxpool3d_forward_impl_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    int64_t* __restrict__ indices,
    const float* __restrict__ samples,
    int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
    int64_t out_d, int64_t out_h, int64_t out_w,
    int64_t total)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ow = idx % out_w;
        int64_t oh = (idx / out_w) % out_h;
        int64_t od = (idx / (out_w * out_h)) % out_d;
        int64_t c  = (idx / (out_w * out_h * out_d)) % C;
        int64_t n  = idx / (out_w * out_h * out_d * C);

        float sample_d = samples ? samples[(n * C + c) * 3 + 0] : 0.5f;
        float sample_h = samples ? samples[(n * C + c) * 3 + 1] : 0.5f;
        float sample_w = samples ? samples[(n * C + c) * 3 + 2] : 0.5f;

        float ratio_d = static_cast<float>(D) / out_d;
        float ratio_h = static_cast<float>(H) / out_h;
        float ratio_w = static_cast<float>(W) / out_w;

        int64_t d_start = static_cast<int64_t>(floorf((od + sample_d) * ratio_d - sample_d));
        int64_t d_end   = static_cast<int64_t>(floorf((od + 1 + sample_d) * ratio_d - sample_d));
        int64_t h_start = static_cast<int64_t>(floorf((oh + sample_h) * ratio_h - sample_h));
        int64_t h_end   = static_cast<int64_t>(floorf((oh + 1 + sample_h) * ratio_h - sample_h));
        int64_t w_start = static_cast<int64_t>(floorf((ow + sample_w) * ratio_w - sample_w));
        int64_t w_end   = static_cast<int64_t>(floorf((ow + 1 + sample_w) * ratio_w - sample_w));

        d_start = max(d_start, int64_t{0}); d_end = min(d_end, D);
        h_start = max(h_start, int64_t{0}); h_end = min(h_end, H);
        w_start = max(w_start, int64_t{0}); w_end = min(w_end, W);
        if (d_end <= d_start) d_end = min(d_start + 1, D);
        if (h_end <= h_start) h_end = min(h_start + 1, H);
        if (w_end <= w_start) w_end = min(w_start + 1, W);

        double max_val = -1e308;
        int64_t max_idx = d_start * H * W + h_start * W + w_start;

        for (int64_t d = d_start; d < d_end; ++d) {
            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    int64_t in_idx = (((n * C + c) * D + d) * H + h) * W + w;
                    double val = input[in_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = (d * H + h) * W + w;
                    }
                }
            }
        }

        output[idx] = max_val;
        indices[idx] = max_idx;
    }
}

auto fractional_maxpool3d_forward_kernel(const Tensor& input,
                                          int64_t out_d, int64_t out_h, int64_t out_w,
                                          const Tensor* random_samples,
                                          cudaStream_t stream)
    -> std::pair<Tensor, Tensor>
{
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], D = shape[2], H = shape[3], W = shape[4];
    int64_t total = N * C * out_d * out_h * out_w;

    auto output  = Tensor({N, C, out_d, out_h, out_w}, DType::Float32, input.device());
    auto idx_out = Tensor({N, C, out_d, out_h, out_w}, DType::Int64, input.device());

    const float* samples_ptr = nullptr;
    if (random_samples && random_samples->numel() > 0) {
        samples_ptr = random_samples->data<float>();
    }

    if (input.dtype() == DType::Float32) {
        auto [grid, block] = optimal_launch_config(fractional_maxpool3d_forward_impl, total);
        fractional_maxpool3d_forward_impl<<<grid, block, 0, stream>>>(
            input.data<float>(), output.data<float>(), idx_out.data<int64_t>(),
            samples_ptr, N, C, D, H, W, out_d, out_h, out_w, total);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        // Native Float64 path.
        Tensor out_f64({N, C, out_d, out_h, out_w}, DType::Float64, input.device());
        auto [grid, block] = optimal_launch_config(fractional_maxpool3d_forward_impl_f64, total);
        fractional_maxpool3d_forward_impl_f64<<<grid, block, 0, stream>>>(
            input.data<double>(), out_f64.data<double>(), idx_out.data<int64_t>(),
            samples_ptr, N, C, D, H, W, out_d, out_h, out_w, total);
        CUDA_CHECK(cudaGetLastError());
        return {out_f64, idx_out};
    } else if (input.dtype() == DType::Float16) {
        Tensor in_f32 = create_zeros_cuda({N, C, D, H, W}, DType::Float32, input.device(), stream);
        int64_t in_total = N * C * D * H * W;
        auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__half>, in_total);
        convert_to_f32<__half><<<grid_conv, block_conv, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()), in_f32.data<float>(), in_total);
        CUDA_CHECK(cudaGetLastError());

        auto [grid, block] = optimal_launch_config(fractional_maxpool3d_forward_impl, total);
        fractional_maxpool3d_forward_impl<<<grid, block, 0, stream>>>(
            in_f32.data<float>(), output.data<float>(), idx_out.data<int64_t>(),
            samples_ptr, N, C, D, H, W, out_d, out_h, out_w, total);
        CUDA_CHECK(cudaGetLastError());

        Tensor out_f16({N, C, out_d, out_h, out_w}, DType::Float16, input.device());
        auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__half>, total);
        convert_f32_to<__half><<<grid_back, block_back, 0, stream>>>(
            output.data<float>(), reinterpret_cast<__half*>(out_f16.data<Float16>()), total);
        CUDA_CHECK(cudaGetLastError());
        return {out_f16, idx_out};
    } else if (input.dtype() == DType::BFloat16) {
        Tensor in_f32 = create_zeros_cuda({N, C, D, H, W}, DType::Float32, input.device(), stream);
        int64_t in_total = N * C * D * H * W;
        auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__nv_bfloat16>, in_total);
        convert_to_f32<__nv_bfloat16><<<grid_conv, block_conv, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()), in_f32.data<float>(), in_total);
        CUDA_CHECK(cudaGetLastError());

        auto [grid, block] = optimal_launch_config(fractional_maxpool3d_forward_impl, total);
        fractional_maxpool3d_forward_impl<<<grid, block, 0, stream>>>(
            in_f32.data<float>(), output.data<float>(), idx_out.data<int64_t>(),
            samples_ptr, N, C, D, H, W, out_d, out_h, out_w, total);
        CUDA_CHECK(cudaGetLastError());

        Tensor out_bf16({N, C, out_d, out_h, out_w}, DType::BFloat16, input.device());
        auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__nv_bfloat16>, total);
        convert_f32_to<__nv_bfloat16><<<grid_back, block_back, 0, stream>>>(
            output.data<float>(), reinterpret_cast<__nv_bfloat16*>(out_bf16.data<BFloat16>()), total);
        CUDA_CHECK(cudaGetLastError());
        return {out_bf16, idx_out};
    } else {
        throw std::runtime_error("fractional_maxpool3d_forward: unsupported dtype");
    }

    return {output, idx_out};
}

// ============================================================================
// Fractional Max Pool 3D Backward
// ============================================================================

__global__ void fractional_maxpool3d_backward_impl(
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    float* __restrict__ grad_input,
    int64_t N, int64_t C,
    int64_t in_spatial, int64_t out_spatial,
    int64_t total)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t nc = idx / out_spatial;
        int64_t max_idx = indices[idx];
        float grad_val = grad_output[idx];

        atomicAdd(&grad_input[nc * in_spatial + max_idx], grad_val);
    }
}

__global__ void fractional_maxpool3d_backward_impl_f64(
    const double* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    double* __restrict__ grad_input,
    int64_t N, int64_t C,
    int64_t in_spatial, int64_t out_spatial,
    int64_t total)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t nc = idx / out_spatial;
        int64_t max_idx = indices[idx];
        double grad_val = grad_output[idx];

        atomicAdd(&grad_input[nc * in_spatial + max_idx], grad_val);
    }
}

auto fractional_maxpool3d_backward_kernel(const Tensor& grad_output,
                                           const Tensor& indices,
                                           const std::vector<int64_t>& input_shape,
                                           cudaStream_t stream)
    -> Tensor
{
    auto grad_shape = grad_output.shape();
    int64_t N = input_shape[0], C = input_shape[1];
    int64_t D = input_shape[2], H = input_shape[3], W = input_shape[4];
    int64_t out_d = grad_shape[2], out_h = grad_shape[3], out_w = grad_shape[4];
    int64_t in_spatial = D * H * W;
    int64_t out_spatial = out_d * out_h * out_w;
    int64_t total = N * C * out_spatial;

    auto grad_input = create_zeros_cuda(input_shape, DType::Float32, grad_output.device(), stream);

    if (grad_output.dtype() == DType::Float32) {
        auto [grid, block] = optimal_launch_config(fractional_maxpool3d_backward_impl, total);
        fractional_maxpool3d_backward_impl<<<grid, block, 0, stream>>>(
            grad_output.data<float>(), indices.data<int64_t>(),
            grad_input.data<float>(), N, C, in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float64) {
        // Native Float64 path.
        Tensor result = create_zeros_cuda(input_shape, DType::Float64, grad_output.device(), stream);
        auto [grid, block] = optimal_launch_config(fractional_maxpool3d_backward_impl_f64, total);
        fractional_maxpool3d_backward_impl_f64<<<grid, block, 0, stream>>>(
            grad_output.data<double>(), indices.data<int64_t>(),
            result.data<double>(), N, C, in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (grad_output.dtype() == DType::Float16) {
        Tensor go_f32 = create_zeros_cuda(std::vector<int64_t>(grad_shape.begin(), grad_shape.end()),
                                          DType::Float32, grad_output.device(), stream);
        auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__half>, total);
        convert_to_f32<__half><<<grid_conv, block_conv, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data<Float16>()), go_f32.data<float>(), total);
        CUDA_CHECK(cudaGetLastError());

        auto [grid, block] = optimal_launch_config(fractional_maxpool3d_backward_impl, total);
        fractional_maxpool3d_backward_impl<<<grid, block, 0, stream>>>(
            go_f32.data<float>(), indices.data<int64_t>(),
            grad_input.data<float>(), N, C, in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());

        Tensor result(input_shape, DType::Float16, grad_output.device());
        int64_t in_total = N * C * in_spatial;
        auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__half>, in_total);
        convert_f32_to<__half><<<grid_back, block_back, 0, stream>>>(
            grad_input.data<float>(), reinterpret_cast<__half*>(result.data<Float16>()), in_total);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (grad_output.dtype() == DType::BFloat16) {
        Tensor go_f32 = create_zeros_cuda(std::vector<int64_t>(grad_shape.begin(), grad_shape.end()),
                                          DType::Float32, grad_output.device(), stream);
        auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__nv_bfloat16>, total);
        convert_to_f32<__nv_bfloat16><<<grid_conv, block_conv, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data<BFloat16>()), go_f32.data<float>(), total);
        CUDA_CHECK(cudaGetLastError());

        auto [grid, block] = optimal_launch_config(fractional_maxpool3d_backward_impl, total);
        fractional_maxpool3d_backward_impl<<<grid, block, 0, stream>>>(
            go_f32.data<float>(), indices.data<int64_t>(),
            grad_input.data<float>(), N, C, in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());

        Tensor result(input_shape, DType::BFloat16, grad_output.device());
        int64_t in_total = N * C * in_spatial;
        auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__nv_bfloat16>, in_total);
        convert_f32_to<__nv_bfloat16><<<grid_back, block_back, 0, stream>>>(
            grad_input.data<float>(), reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), in_total);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

    throw std::runtime_error("fractional_maxpool3d_backward: unsupported dtype");
}

// ============================================================================
// Max Unpool 2D Forward
// ============================================================================

__global__ void max_unpool2d_forward_impl(
    const float* __restrict__ input,
    const int64_t* __restrict__ indices,
    float* __restrict__ output,
    int64_t in_spatial, int64_t out_spatial,
    int64_t total)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t nc = idx / in_spatial;

        int64_t out_idx = indices[idx];
        if (out_idx >= 0 && out_idx < out_spatial) {
            output[nc * out_spatial + out_idx] = input[idx];
        }
    }
}

// audit-2026-05-03 — Float64 native max_unpool2d (avoid f32 detour).
__global__ void max_unpool2d_forward_impl_f64(
    const double* __restrict__ input,
    const int64_t* __restrict__ indices,
    double* __restrict__ output,
    int64_t in_spatial, int64_t out_spatial,
    int64_t total)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t nc = idx / in_spatial;

        int64_t out_idx = indices[idx];
        if (out_idx >= 0 && out_idx < out_spatial) {
            output[nc * out_spatial + out_idx] = input[idx];
        }
    }
}

auto max_unpool2d_forward_kernel(const Tensor& input, const Tensor& indices,
                                  int64_t out_h, int64_t out_w,
                                  cudaStream_t stream) -> Tensor
{
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], in_h = shape[2], in_w = shape[3];
    int64_t in_spatial = in_h * in_w;
    int64_t out_spatial = out_h * out_w;
    int64_t total = N * C * in_spatial;

    auto output = create_zeros_cuda({N, C, out_h, out_w}, DType::Float32, input.device(), stream);

    if (input.dtype() == DType::Float32) {
        auto [grid, block] = optimal_launch_config(max_unpool2d_forward_impl, total);
        max_unpool2d_forward_impl<<<grid, block, 0, stream>>>(
            input.data<float>(), indices.data<int64_t>(),
            output.data<float>(), in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());
        return output;
    } else if (input.dtype() == DType::Float64) {
        // Native Float64 path — preserves precision through the autograd
        // gradcheck. Previous f32 detour dropped ~30 mantissa bits.
        Tensor result_f64 = create_zeros_cuda({N, C, out_h, out_w}, DType::Float64, input.device(), stream);
        auto [grid, block] = optimal_launch_config(max_unpool2d_forward_impl_f64, total);
        max_unpool2d_forward_impl_f64<<<grid, block, 0, stream>>>(
            input.data<double>(), indices.data<int64_t>(),
            result_f64.data<double>(), in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());
        return result_f64;
    } else if (input.dtype() == DType::Float16) {
        Tensor in_f32 = create_zeros_cuda({N, C, in_h, in_w}, DType::Float32, input.device(), stream);
        auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__half>, total);
        convert_to_f32<__half><<<grid_conv, block_conv, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()), in_f32.data<float>(), total);
        CUDA_CHECK(cudaGetLastError());

        auto [grid, block] = optimal_launch_config(max_unpool2d_forward_impl, total);
        max_unpool2d_forward_impl<<<grid, block, 0, stream>>>(
            in_f32.data<float>(), indices.data<int64_t>(),
            output.data<float>(), in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());

        Tensor result({N, C, out_h, out_w}, DType::Float16, input.device());
        int64_t out_total = N * C * out_spatial;
        auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__half>, out_total);
        convert_f32_to<__half><<<grid_back, block_back, 0, stream>>>(
            output.data<float>(), reinterpret_cast<__half*>(result.data<Float16>()), out_total);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (input.dtype() == DType::BFloat16) {
        Tensor in_f32 = create_zeros_cuda({N, C, in_h, in_w}, DType::Float32, input.device(), stream);
        auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__nv_bfloat16>, total);
        convert_to_f32<__nv_bfloat16><<<grid_conv, block_conv, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()), in_f32.data<float>(), total);
        CUDA_CHECK(cudaGetLastError());

        auto [grid, block] = optimal_launch_config(max_unpool2d_forward_impl, total);
        max_unpool2d_forward_impl<<<grid, block, 0, stream>>>(
            in_f32.data<float>(), indices.data<int64_t>(),
            output.data<float>(), in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());

        Tensor result({N, C, out_h, out_w}, DType::BFloat16, input.device());
        int64_t out_total = N * C * out_spatial;
        auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__nv_bfloat16>, out_total);
        convert_f32_to<__nv_bfloat16><<<grid_back, block_back, 0, stream>>>(
            output.data<float>(), reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), out_total);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

    throw std::runtime_error("max_unpool2d_forward: unsupported dtype");
}

// ============================================================================
// Max Unpool 2D Backward
// ============================================================================

__global__ void max_unpool2d_backward_impl(
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    float* __restrict__ grad_input,
    int64_t in_spatial, int64_t out_spatial,
    int64_t total)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t nc = idx / in_spatial;
        int64_t out_idx = indices[idx];

        if (out_idx >= 0 && out_idx < out_spatial) {
            grad_input[idx] = grad_output[nc * out_spatial + out_idx];
        } else {
            grad_input[idx] = 0.0f;
        }
    }
}

__global__ void max_unpool2d_backward_impl_f64(
    const double* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    double* __restrict__ grad_input,
    int64_t in_spatial, int64_t out_spatial,
    int64_t total)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t nc = idx / in_spatial;
        int64_t out_idx = indices[idx];

        if (out_idx >= 0 && out_idx < out_spatial) {
            grad_input[idx] = grad_output[nc * out_spatial + out_idx];
        } else {
            grad_input[idx] = 0.0;
        }
    }
}

auto max_unpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                   const std::vector<int64_t>& input_shape,
                                   cudaStream_t stream) -> Tensor
{
    auto grad_shape = grad_output.shape();
    int64_t N = input_shape[0], C = input_shape[1];
    int64_t in_h = input_shape[2], in_w = input_shape[3];
    int64_t out_h = grad_shape[2], out_w = grad_shape[3];
    int64_t in_spatial = in_h * in_w;
    int64_t out_spatial = out_h * out_w;
    int64_t total = N * C * in_spatial;

    if (grad_output.dtype() == DType::Float32) {
        Tensor grad_input(input_shape, DType::Float32, grad_output.device());
        auto [grid, block] = optimal_launch_config(max_unpool2d_backward_impl, total);
        max_unpool2d_backward_impl<<<grid, block, 0, stream>>>(
            grad_output.data<float>(), indices.data<int64_t>(),
            grad_input.data<float>(), in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float64) {
        // Native Float64 path — preserves precision through autograd gradcheck.
        Tensor result(input_shape, DType::Float64, grad_output.device());
        auto [grid, block] = optimal_launch_config(max_unpool2d_backward_impl_f64, total);
        max_unpool2d_backward_impl_f64<<<grid, block, 0, stream>>>(
            grad_output.data<double>(), indices.data<int64_t>(),
            result.data<double>(), in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (grad_output.dtype() == DType::Float16) {
        int64_t out_total = N * C * out_spatial;
        Tensor go_f32 = create_zeros_cuda(std::vector<int64_t>(grad_shape.begin(), grad_shape.end()),
                                          DType::Float32, grad_output.device(), stream);
        auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__half>, out_total);
        convert_to_f32<__half><<<grid_conv, block_conv, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data<Float16>()), go_f32.data<float>(), out_total);
        CUDA_CHECK(cudaGetLastError());

        Tensor gi_f32(input_shape, DType::Float32, grad_output.device());
        auto [grid, block] = optimal_launch_config(max_unpool2d_backward_impl, total);
        max_unpool2d_backward_impl<<<grid, block, 0, stream>>>(
            go_f32.data<float>(), indices.data<int64_t>(),
            gi_f32.data<float>(), in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());

        Tensor result(input_shape, DType::Float16, grad_output.device());
        auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__half>, total);
        convert_f32_to<__half><<<grid_back, block_back, 0, stream>>>(
            gi_f32.data<float>(), reinterpret_cast<__half*>(result.data<Float16>()), total);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (grad_output.dtype() == DType::BFloat16) {
        int64_t out_total = N * C * out_spatial;
        Tensor go_f32 = create_zeros_cuda(std::vector<int64_t>(grad_shape.begin(), grad_shape.end()),
                                          DType::Float32, grad_output.device(), stream);
        auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__nv_bfloat16>, out_total);
        convert_to_f32<__nv_bfloat16><<<grid_conv, block_conv, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data<BFloat16>()), go_f32.data<float>(), out_total);
        CUDA_CHECK(cudaGetLastError());

        Tensor gi_f32(input_shape, DType::Float32, grad_output.device());
        auto [grid, block] = optimal_launch_config(max_unpool2d_backward_impl, total);
        max_unpool2d_backward_impl<<<grid, block, 0, stream>>>(
            go_f32.data<float>(), indices.data<int64_t>(),
            gi_f32.data<float>(), in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());

        Tensor result(input_shape, DType::BFloat16, grad_output.device());
        auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__nv_bfloat16>, total);
        convert_f32_to<__nv_bfloat16><<<grid_back, block_back, 0, stream>>>(
            gi_f32.data<float>(), reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), total);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

    throw std::runtime_error("max_unpool2d_backward: unsupported dtype");
}

// ============================================================================
// Max Unpool 3D Forward
// ============================================================================

__global__ void max_unpool3d_forward_impl(
    const float* __restrict__ input,
    const int64_t* __restrict__ indices,
    float* __restrict__ output,
    int64_t in_spatial, int64_t out_spatial,
    int64_t total)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t nc = idx / in_spatial;
        int64_t out_idx = indices[idx];

        if (out_idx >= 0 && out_idx < out_spatial) {
            output[nc * out_spatial + out_idx] = input[idx];
        }
    }
}

__global__ void max_unpool3d_forward_impl_f64(
    const double* __restrict__ input,
    const int64_t* __restrict__ indices,
    double* __restrict__ output,
    int64_t in_spatial, int64_t out_spatial,
    int64_t total)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t nc = idx / in_spatial;
        int64_t out_idx = indices[idx];

        if (out_idx >= 0 && out_idx < out_spatial) {
            output[nc * out_spatial + out_idx] = input[idx];
        }
    }
}

auto max_unpool3d_forward_kernel(const Tensor& input, const Tensor& indices,
                                  int64_t out_d, int64_t out_h, int64_t out_w,
                                  cudaStream_t stream) -> Tensor
{
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], in_d = shape[2], in_h = shape[3], in_w = shape[4];
    int64_t in_spatial = in_d * in_h * in_w;
    int64_t out_spatial = out_d * out_h * out_w;
    int64_t total = N * C * in_spatial;

    auto output = create_zeros_cuda({N, C, out_d, out_h, out_w}, DType::Float32, input.device(), stream);

    if (input.dtype() == DType::Float32) {
        auto [grid, block] = optimal_launch_config(max_unpool3d_forward_impl, total);
        max_unpool3d_forward_impl<<<grid, block, 0, stream>>>(
            input.data<float>(), indices.data<int64_t>(),
            output.data<float>(), in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());
        return output;
    } else if (input.dtype() == DType::Float64) {
        // Native Float64 path — preserves precision for autograd gradcheck.
        Tensor result_f64 = create_zeros_cuda({N, C, out_d, out_h, out_w}, DType::Float64, input.device(), stream);
        auto [grid, block] = optimal_launch_config(max_unpool3d_forward_impl_f64, total);
        max_unpool3d_forward_impl_f64<<<grid, block, 0, stream>>>(
            input.data<double>(), indices.data<int64_t>(),
            result_f64.data<double>(), in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());
        return result_f64;
    } else if (input.dtype() == DType::Float16) {
        Tensor in_f32 = create_zeros_cuda({N, C, in_d, in_h, in_w}, DType::Float32, input.device(), stream);
        auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__half>, total);
        convert_to_f32<__half><<<grid_conv, block_conv, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()), in_f32.data<float>(), total);
        CUDA_CHECK(cudaGetLastError());

        auto [grid, block] = optimal_launch_config(max_unpool3d_forward_impl, total);
        max_unpool3d_forward_impl<<<grid, block, 0, stream>>>(
            in_f32.data<float>(), indices.data<int64_t>(),
            output.data<float>(), in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());

        Tensor result({N, C, out_d, out_h, out_w}, DType::Float16, input.device());
        int64_t out_total = N * C * out_spatial;
        auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__half>, out_total);
        convert_f32_to<__half><<<grid_back, block_back, 0, stream>>>(
            output.data<float>(), reinterpret_cast<__half*>(result.data<Float16>()), out_total);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (input.dtype() == DType::BFloat16) {
        Tensor in_f32 = create_zeros_cuda({N, C, in_d, in_h, in_w}, DType::Float32, input.device(), stream);
        auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__nv_bfloat16>, total);
        convert_to_f32<__nv_bfloat16><<<grid_conv, block_conv, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()), in_f32.data<float>(), total);
        CUDA_CHECK(cudaGetLastError());

        auto [grid, block] = optimal_launch_config(max_unpool3d_forward_impl, total);
        max_unpool3d_forward_impl<<<grid, block, 0, stream>>>(
            in_f32.data<float>(), indices.data<int64_t>(),
            output.data<float>(), in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());

        Tensor result({N, C, out_d, out_h, out_w}, DType::BFloat16, input.device());
        int64_t out_total = N * C * out_spatial;
        auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__nv_bfloat16>, out_total);
        convert_f32_to<__nv_bfloat16><<<grid_back, block_back, 0, stream>>>(
            output.data<float>(), reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), out_total);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

    throw std::runtime_error("max_unpool3d_forward: unsupported dtype");
}

// ============================================================================
// Max Unpool 3D Backward
// ============================================================================

__global__ void max_unpool3d_backward_impl(
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    float* __restrict__ grad_input,
    int64_t in_spatial, int64_t out_spatial,
    int64_t total)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t nc = idx / in_spatial;
        int64_t out_idx = indices[idx];

        if (out_idx >= 0 && out_idx < out_spatial) {
            grad_input[idx] = grad_output[nc * out_spatial + out_idx];
        } else {
            grad_input[idx] = 0.0f;
        }
    }
}

__global__ void max_unpool3d_backward_impl_f64(
    const double* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    double* __restrict__ grad_input,
    int64_t in_spatial, int64_t out_spatial,
    int64_t total)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t nc = idx / in_spatial;
        int64_t out_idx = indices[idx];

        if (out_idx >= 0 && out_idx < out_spatial) {
            grad_input[idx] = grad_output[nc * out_spatial + out_idx];
        } else {
            grad_input[idx] = 0.0;
        }
    }
}

auto max_unpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                   const std::vector<int64_t>& input_shape,
                                   cudaStream_t stream) -> Tensor
{
    auto grad_shape = grad_output.shape();
    int64_t N = input_shape[0], C = input_shape[1];
    int64_t in_d = input_shape[2], in_h = input_shape[3], in_w = input_shape[4];
    int64_t out_d = grad_shape[2], out_h = grad_shape[3], out_w = grad_shape[4];
    int64_t in_spatial = in_d * in_h * in_w;
    int64_t out_spatial = out_d * out_h * out_w;
    int64_t total = N * C * in_spatial;

    if (grad_output.dtype() == DType::Float32) {
        Tensor grad_input(input_shape, DType::Float32, grad_output.device());
        auto [grid, block] = optimal_launch_config(max_unpool3d_backward_impl, total);
        max_unpool3d_backward_impl<<<grid, block, 0, stream>>>(
            grad_output.data<float>(), indices.data<int64_t>(),
            grad_input.data<float>(), in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());
        return grad_input;
    } else if (grad_output.dtype() == DType::Float64) {
        // Native Float64 path.
        Tensor result(input_shape, DType::Float64, grad_output.device());
        auto [grid, block] = optimal_launch_config(max_unpool3d_backward_impl_f64, total);
        max_unpool3d_backward_impl_f64<<<grid, block, 0, stream>>>(
            grad_output.data<double>(), indices.data<int64_t>(),
            result.data<double>(), in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (grad_output.dtype() == DType::Float16) {
        int64_t out_total = N * C * out_spatial;
        Tensor go_f32 = create_zeros_cuda(std::vector<int64_t>(grad_shape.begin(), grad_shape.end()),
                                          DType::Float32, grad_output.device(), stream);
        auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__half>, out_total);
        convert_to_f32<__half><<<grid_conv, block_conv, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data<Float16>()), go_f32.data<float>(), out_total);
        CUDA_CHECK(cudaGetLastError());

        Tensor gi_f32(input_shape, DType::Float32, grad_output.device());
        auto [grid, block] = optimal_launch_config(max_unpool3d_backward_impl, total);
        max_unpool3d_backward_impl<<<grid, block, 0, stream>>>(
            go_f32.data<float>(), indices.data<int64_t>(),
            gi_f32.data<float>(), in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());

        Tensor result(input_shape, DType::Float16, grad_output.device());
        auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__half>, total);
        convert_f32_to<__half><<<grid_back, block_back, 0, stream>>>(
            gi_f32.data<float>(), reinterpret_cast<__half*>(result.data<Float16>()), total);
        CUDA_CHECK(cudaGetLastError());
        return result;
    } else if (grad_output.dtype() == DType::BFloat16) {
        int64_t out_total = N * C * out_spatial;
        Tensor go_f32 = create_zeros_cuda(std::vector<int64_t>(grad_shape.begin(), grad_shape.end()),
                                          DType::Float32, grad_output.device(), stream);
        auto [grid_conv, block_conv] = optimal_launch_config(convert_to_f32<__nv_bfloat16>, out_total);
        convert_to_f32<__nv_bfloat16><<<grid_conv, block_conv, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data<BFloat16>()), go_f32.data<float>(), out_total);
        CUDA_CHECK(cudaGetLastError());

        Tensor gi_f32(input_shape, DType::Float32, grad_output.device());
        auto [grid, block] = optimal_launch_config(max_unpool3d_backward_impl, total);
        max_unpool3d_backward_impl<<<grid, block, 0, stream>>>(
            go_f32.data<float>(), indices.data<int64_t>(),
            gi_f32.data<float>(), in_spatial, out_spatial, total);
        CUDA_CHECK(cudaGetLastError());

        Tensor result(input_shape, DType::BFloat16, grad_output.device());
        auto [grid_back, block_back] = optimal_launch_config(convert_f32_to<__nv_bfloat16>, total);
        convert_f32_to<__nv_bfloat16><<<grid_back, block_back, 0, stream>>>(
            gi_f32.data<float>(), reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()), total);
        CUDA_CHECK(cudaGetLastError());
        return result;
    }

    throw std::runtime_error("max_unpool3d_backward: unsupported dtype");
}

// ============================================================================
// Phase A.1 — Max Unpool 1D (CUDA). Reshape (N, C, L) → (N, C, L, 1) and
// reuse the existing 2D kernel; reshape is metadata-only on the GPU.
// ============================================================================

auto max_unpool1d_forward_kernel(const Tensor& input, const Tensor& indices,
                                 int64_t out_l, cudaStream_t stream) -> Tensor
{
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], in_l = shape[2];
    auto input_4d = input.contiguous().reshape({N, C, in_l, 1});
    auto indices_4d = indices.contiguous().reshape({N, C, in_l, 1});
    auto out_4d = max_unpool2d_forward_kernel(input_4d, indices_4d, out_l, /*out_w=*/1, stream);
    return out_4d.reshape({N, C, out_l});
}

auto max_unpool1d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                   const std::vector<int64_t>& input_shape,
                                   cudaStream_t stream) -> Tensor
{
    int64_t N = input_shape[0], C = input_shape[1], in_l = input_shape[2];
    int64_t out_l = grad_output.shape()[2];
    std::vector<int64_t> input_shape_4d = {N, C, in_l, 1};
    auto grad_4d = grad_output.contiguous().reshape({N, C, out_l, 1});
    auto indices_4d = indices.contiguous().reshape({N, C, in_l, 1});
    auto grad_in_4d = max_unpool2d_backward_kernel(grad_4d, indices_4d, input_shape_4d, stream);
    return grad_in_4d.reshape({N, C, in_l});
}

} // namespace cuda
} // namespace tenzor
