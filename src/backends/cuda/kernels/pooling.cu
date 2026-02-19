/**
 * @file pooling.cu
 * @brief CUDA pooling kernel implementations (fallback when cuDNN is not available)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
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

// ============================================================================
// Launch config helpers
// ============================================================================

static inline int clamp_grid(int64_t blocks) {
    int64_t clamped = std::min(blocks, static_cast<int64_t>(65535));
    return static_cast<int>(clamped > 0 ? clamped : 1);
}

static inline Tensor create_zeros_cuda(const std::vector<int64_t>& shape, DType dtype, Device device) {
    Tensor t(shape, dtype, device);
    size_t bytes = t.numel() * dtype_size(dtype);
    cudaMemset(t.data_ptr(), 0, bytes);
    return t;
}

constexpr int POOL_BLOCK = 256;

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

        float max_val = -1e38f;
        int64_t max_idx = 0;

        for (int64_t kh = 0; kh < kernel_size; ++kh) {
            for (int64_t kw = 0; kw < kernel_size; ++kw) {
                int64_t h = h_start + kh * dilation;
                int64_t w = w_start + kw * dilation;

                if (h >= 0 && h < H && w >= 0 && w < W) {
                    int64_t in_idx = ((n * C + c) * H + h) * W + w;
                    float val = dev_load(input, in_idx);
                    if (val > max_val) {
                        max_val = val;
                        max_idx = h * W + w;
                    }
                }
            }
        }

        dev_store(output, idx, max_val);
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
    int grid = clamp_grid((total + POOL_BLOCK - 1) / POOL_BLOCK);

    if (input.dtype() == DType::Float32) {
        maxpool2d_forward_impl<float><<<grid, POOL_BLOCK, 0, stream>>>(
            input.data<float>(), output.data<float>(), indices.data<int64_t>(),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding, dilation);
    } else if (input.dtype() == DType::Float64) {
        maxpool2d_forward_impl<double><<<grid, POOL_BLOCK, 0, stream>>>(
            input.data<double>(), output.data<double>(), indices.data<int64_t>(),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding, dilation);
    } else if (input.dtype() == DType::Float16) {
        maxpool2d_forward_impl<__half><<<grid, POOL_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            indices.data<int64_t>(),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding, dilation);
    } else if (input.dtype() == DType::BFloat16) {
        maxpool2d_forward_impl<__nv_bfloat16><<<grid, POOL_BLOCK, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data<BFloat16>()),
            indices.data<int64_t>(),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding, dilation);
    } else {
        throw std::runtime_error("maxpool2d_forward_kernel: unsupported dtype");
    }

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

        atomicAdd(&grad_input[in_idx], grad_output[idx]);
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

        atomicAdd(&grad_input[in_idx], grad_output[idx]);
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
    int grid_out = clamp_grid((total_out + POOL_BLOCK - 1) / POOL_BLOCK);
    int grid_in = clamp_grid((total_in + POOL_BLOCK - 1) / POOL_BLOCK);

    if (grad_output.dtype() == DType::Float32) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float32, grad_output.device());
        maxpool2d_backward_f32<<<grid_out, POOL_BLOCK, 0, stream>>>(
            grad_output.data<float>(), indices.data<int64_t>(), grad_input.data<float>(),
            N, C, H, W, H_out, W_out);
        return grad_input;
    } else if (grad_output.dtype() == DType::Float64) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float64, grad_output.device());
        maxpool2d_backward_f64<<<grid_out, POOL_BLOCK, 0, stream>>>(
            grad_output.data<double>(), indices.data<int64_t>(), grad_input.data<double>(),
            N, C, H, W, H_out, W_out);
        return grad_input;
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        // Accumulate in float32 then convert back
        Tensor grad_f32 = create_zeros_cuda(input_shape, DType::Float32, grad_output.device());
        Tensor go_f32({N, C, H_out, W_out}, DType::Float32, grad_output.device());

        // Convert grad_output to float32
        if (grad_output.dtype() == DType::Float16) {
            convert_to_f32<__half><<<clamp_grid((total_out + POOL_BLOCK - 1) / POOL_BLOCK), POOL_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(grad_output.data<Float16>()), go_f32.data<float>(), total_out);
        } else {
            convert_to_f32<__nv_bfloat16><<<clamp_grid((total_out + POOL_BLOCK - 1) / POOL_BLOCK), POOL_BLOCK, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(grad_output.data<BFloat16>()), go_f32.data<float>(), total_out);
        }

        // Run backward in float32
        maxpool2d_backward_f32<<<grid_out, POOL_BLOCK, 0, stream>>>(
            go_f32.data<float>(), indices.data<int64_t>(), grad_f32.data<float>(),
            N, C, H, W, H_out, W_out);

        // Convert result back
        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        if (grad_output.dtype() == DType::Float16) {
            convert_f32_to<__half><<<grid_in, POOL_BLOCK, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__half*>(grad_input.data<Float16>()), total_in);
        } else {
            convert_f32_to<__nv_bfloat16><<<grid_in, POOL_BLOCK, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__nv_bfloat16*>(grad_input.data<BFloat16>()), total_in);
        }
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

        float sum = 0.0f;
        int count = 0;

        for (int64_t kh = 0; kh < kernel_size; ++kh) {
            for (int64_t kw = 0; kw < kernel_size; ++kw) {
                int64_t h = h_start + kh;
                int64_t w = w_start + kw;

                if (h >= 0 && h < H && w >= 0 && w < W) {
                    sum += dev_load(input, ((n * C + c) * H + h) * W + w);
                    count++;
                }
            }
        }

        dev_store(output, idx, sum / count);
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
    int grid = clamp_grid((total + POOL_BLOCK - 1) / POOL_BLOCK);

    if (input.dtype() == DType::Float32) {
        avgpool2d_forward_impl<float><<<grid, POOL_BLOCK, 0, stream>>>(
            input.data<float>(), output.data<float>(),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding);
    } else if (input.dtype() == DType::Float64) {
        avgpool2d_forward_impl<double><<<grid, POOL_BLOCK, 0, stream>>>(
            input.data<double>(), output.data<double>(),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding);
    } else if (input.dtype() == DType::Float16) {
        avgpool2d_forward_impl<__half><<<grid, POOL_BLOCK, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding);
    } else if (input.dtype() == DType::BFloat16) {
        avgpool2d_forward_impl<__nv_bfloat16><<<grid, POOL_BLOCK, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data<BFloat16>()),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding);
    } else {
        throw std::runtime_error("avgpool2d_forward_kernel: unsupported dtype");
    }

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
    int grid_out = clamp_grid((total_out + POOL_BLOCK - 1) / POOL_BLOCK);
    int grid_in = clamp_grid((total_in + POOL_BLOCK - 1) / POOL_BLOCK);

    if (grad_output.dtype() == DType::Float32) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float32, grad_output.device());
        avgpool2d_backward_f32<<<grid_out, POOL_BLOCK, 0, stream>>>(
            grad_output.data<float>(), grad_input.data<float>(),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding);
        return grad_input;
    } else if (grad_output.dtype() == DType::Float64) {
        Tensor grad_input = create_zeros_cuda(input_shape, DType::Float64, grad_output.device());
        avgpool2d_backward_f64<<<grid_out, POOL_BLOCK, 0, stream>>>(
            grad_output.data<double>(), grad_input.data<double>(),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding);
        return grad_input;
    } else if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        // Accumulate in float32 then convert back
        Tensor grad_f32 = create_zeros_cuda(input_shape, DType::Float32, grad_output.device());
        Tensor go_f32({N, C, H_out, W_out}, DType::Float32, grad_output.device());

        if (grad_output.dtype() == DType::Float16) {
            convert_to_f32<__half><<<clamp_grid((total_out + POOL_BLOCK - 1) / POOL_BLOCK), POOL_BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(grad_output.data<Float16>()), go_f32.data<float>(), total_out);
        } else {
            convert_to_f32<__nv_bfloat16><<<clamp_grid((total_out + POOL_BLOCK - 1) / POOL_BLOCK), POOL_BLOCK, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(grad_output.data<BFloat16>()), go_f32.data<float>(), total_out);
        }

        avgpool2d_backward_f32<<<grid_out, POOL_BLOCK, 0, stream>>>(
            go_f32.data<float>(), grad_f32.data<float>(),
            N, C, H, W, H_out, W_out, kernel_size, stride, padding);

        Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
        if (grad_output.dtype() == DType::Float16) {
            convert_f32_to<__half><<<grid_in, POOL_BLOCK, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__half*>(grad_input.data<Float16>()), total_in);
        } else {
            convert_f32_to<__nv_bfloat16><<<grid_in, POOL_BLOCK, 0, stream>>>(
                grad_f32.data<float>(), reinterpret_cast<__nv_bfloat16*>(grad_input.data<BFloat16>()), total_in);
        }
        return grad_input;
    }

    throw std::runtime_error("avgpool2d_backward_kernel: unsupported dtype");
}

} // namespace cuda
} // namespace tenzor
