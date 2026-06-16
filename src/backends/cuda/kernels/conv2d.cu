#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/caching_allocator.hpp"
#include "cuda_common.cuh"
#include "cuda_launch_utils.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>  // For __half
#include <mma.h>        // For Tensor Cores (WMMA)
#include <stdexcept>
#include <vector>
#include <iostream>

#include "../cublas_handle_pool.hpp"

namespace tenzor {
namespace cuda {

// ============================================================================
// Kernel Launch Helpers
// ============================================================================

// Uses cudaOccupancyMaxPotentialBlockSize via a representative kernel
// (im2col_kernel<float>) to determine an architecture-optimal block size
// instead of the previous hardcoded 256.
template<typename T>
__global__ void im2col_kernel(
    const T* input, T* output,
    int64_t batch, int64_t channels, int64_t height, int64_t width,
    int64_t kernel_h, int64_t kernel_w, int64_t stride, int64_t padding,
    int64_t dilation, int64_t out_h, int64_t out_w);

// compute_launch_config_1d() is now in cuda_launch_utils.cuh



// ============================================================================
// Helper Functions
// ============================================================================

// Calculate output size for convolution
__host__ __device__ inline int64_t calculate_output_size(int64_t input_size, int64_t kernel_size,
                                                          int64_t stride, int64_t padding, int64_t dilation) {
    #ifndef __CUDA_ARCH__
    // Host-side validation (not in device code)
    if (stride == 0) {
        throw std::invalid_argument("Conv2d: stride cannot be zero");
    }
    #endif
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
}

// ============================================================================
// NHWC to NCHW Transpose Kernel
// ============================================================================

/**
 * @brief Transpose cuBLAS output from NHWC to NCHW layout
 *
 * cuBLAS produces output in (batch*out_h*out_w, channels) layout (NHWC)
 * but we need [batch, channels, out_h, out_w] layout (NCHW)
 */
template<typename T>
__global__ void nhwc_to_nchw_kernel(
    const T* nhwc_input,       // (batch * out_h * out_w, channels_per_group)
    T* nchw_output,            // [batch, out_channels, out_h, out_w]
    int64_t batch,
    int64_t out_h,
    int64_t out_w,
    int64_t out_channels,      // Total output channels
    int64_t channels_per_group,
    int64_t channel_offset     // Starting channel index (out_start)
) {
    // Each thread handles one element
    int64_t total_spatial = batch * out_h * out_w;

    TENZOR_CUDA_KERNEL_LOOP(idx, total_spatial * channels_per_group) {
        // Decode flat index to (spatial_idx, c)
        int64_t c = idx % channels_per_group;
        int64_t spatial_idx = idx / channels_per_group;

        // Decode spatial_idx to (b, h, w)
        int64_t b = spatial_idx / (out_h * out_w);
        int64_t hw = spatial_idx % (out_h * out_w);
        int64_t h = hw / out_w;
        int64_t w = hw % out_w;

        // Global channel index
        int64_t global_c = channel_offset + c;

        // NCHW index: [b][global_c][h][w]
        int64_t nchw_idx = ((b * out_channels + global_c) * out_h + h) * out_w + w;

        // Copy from NHWC to NCHW
        nchw_output[nchw_idx] = nhwc_input[idx];
    }
}

// ============================================================================
// im2col CUDA Kernel
// ============================================================================

// im2col kernel: Convert 4D input (N,C,H,W) to 2D matrix for convolution
// Input: (batch, in_channels, height, width)
// Output: (batch * out_h * out_w, kernel_h * kernel_w * in_channels)
template<typename T>
__global__ void im2col_kernel(
    const T* input,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t out_h,
    int64_t out_w
) {
    int64_t total_elements = batch * out_h * out_w * channels * kernel_h * kernel_w;

    TENZOR_CUDA_KERNEL_LOOP(idx, total_elements) {
        // Decode flat index to (b, oh, ow, c, kh, kw)
        int64_t temp = idx;
        int64_t kw = temp % kernel_w; temp /= kernel_w;
        int64_t kh = temp % kernel_h; temp /= kernel_h;
        int64_t c = temp % channels; temp /= channels;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t b = temp;

        // Calculate input position with padding and dilation
        int64_t ih = oh * stride - padding + kh * dilation;
        int64_t iw = ow * stride - padding + kw * dilation;

        // Output index in col matrix
        // Shape: (batch * out_h * out_w, channels * kernel_h * kernel_w)
        int64_t out_row = b * out_h * out_w + oh * out_w + ow;
        int64_t out_col = c * kernel_h * kernel_w + kh * kernel_w + kw;
        int64_t out_idx = out_row * (channels * kernel_h * kernel_w) + out_col;

        // Check bounds and apply padding
        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            int64_t input_idx = b * (channels * height * width) +
                               c * (height * width) +
                               ih * width + iw;
            output[out_idx] = input[input_idx];
        } else {
            output[out_idx] = T(0);  // Padding with zeros
        }
    }
}

// ============================================================================
// FP16 im2col CUDA Kernel
// ============================================================================

// FP16 im2col kernel: Convert 4D input (N,C,H,W) to 2D matrix for convolution
// Same logic as float version but uses __half for FP16 operations
__global__ void im2col_kernel_f16(
    const __half* input,
    __half* output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t out_h,
    int64_t out_w
) {
    int64_t total_elements = batch * out_h * out_w * channels * kernel_h * kernel_w;

    TENZOR_CUDA_KERNEL_LOOP(idx, total_elements) {
        // Decode flat index to (b, oh, ow, c, kh, kw)
        int64_t temp = idx;
        int64_t kw = temp % kernel_w; temp /= kernel_w;
        int64_t kh = temp % kernel_h; temp /= kernel_h;
        int64_t c = temp % channels; temp /= channels;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t b = temp;

        // Calculate input position with padding and dilation
        int64_t ih = oh * stride - padding + kh * dilation;
        int64_t iw = ow * stride - padding + kw * dilation;

        // Output index in col matrix
        int64_t out_row = b * out_h * out_w + oh * out_w + ow;
        int64_t out_col = c * kernel_h * kernel_w + kh * kernel_w + kw;
        int64_t out_idx = out_row * (channels * kernel_h * kernel_w) + out_col;

        // Check bounds and apply padding
        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            int64_t input_idx = b * (channels * height * width) +
                               c * (height * width) +
                               ih * width + iw;
            output[out_idx] = input[input_idx];
        } else {
            output[out_idx] = __float2half(0.0f);  // Padding with zeros
        }
    }
}

// ============================================================================
// col2im CUDA Kernel - Optimized Version
// ============================================================================

// col2im kernel: Reverse of im2col for gradient computation
// Input: (batch * out_h * out_w, kernel_h * kernel_w * in_channels)
// Output: (batch, in_channels, height, width)
//
// Output-centric strategy: each thread processes one output element and
// accumulates from all contributing col positions. Eliminates atomic
// contention entirely at the cost of more work per thread.
template<typename T>
__global__ void col2im_kernel_output_centric(
    const T* col,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t out_h,
    int64_t out_w
) {
    // Each thread processes one output element
    int64_t total_output = batch * channels * height * width;

    TENZOR_CUDA_KERNEL_LOOP(output_idx, total_output) {
        // Decode output index to (b, c, ih, iw)
        int64_t temp = output_idx;
        int64_t iw = temp % width; temp /= width;
        int64_t ih = temp % height; temp /= height;
        int64_t c = temp % channels; temp /= channels;
        int64_t b = temp;

        // Accumulate from all kernel positions that contribute to this output
        T sum = T(0);

        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                // Reverse the mapping: given (ih, iw) and (kh, kw), find (oh, ow)
                // ih = oh * stride - padding + kh * dilation
                // => oh = (ih + padding - kh * dilation) / stride

                int64_t ih_shifted = ih + padding - kh * dilation;
                int64_t iw_shifted = iw + padding - kw * dilation;

                // Check if this maps to a valid output position
                if (ih_shifted % stride == 0 && iw_shifted % stride == 0) {
                    int64_t oh = ih_shifted / stride;
                    int64_t ow = iw_shifted / stride;

                    if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                        // This kernel position contributes to our output
                        int64_t col_row = b * out_h * out_w + oh * out_w + ow;
                        int64_t col_col = c * kernel_h * kernel_w + kh * kernel_w + kw;
                        int64_t col_idx = col_row * (channels * kernel_h * kernel_w) + col_col;

                        sum += col[col_idx];
                    }
                }
            }
        }

        // Direct write, no atomics needed!
        output[output_idx] = sum;
    }
}

// Primary col2im dispatcher: uses output-centric approach to eliminate atomics
// This function signature is called by the conv2d backward pass
template<typename T>
__global__ void col2im_kernel(
    const T* col,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t out_h,
    int64_t out_w
) {
    // PERFORMANCE OPTIMIZATION - ELIMINATING ATOMIC BOTTLENECK:
    //
    // Original approach (col-centric with atomics):
    // - Each thread processes one element from col buffer
    // - Uses atomicAdd because multiple col elements map to same output
    // - Problem: 2-5x slowdown due to atomic serialization
    //
    // Optimized approach (output-centric, NO atomics):
    // - Each thread processes one output element
    // - Accumulates from all contributing col positions
    // - Uses direct write (no atomic needed!)
    //
    // Trade-off analysis:
    // - Extra work per thread: O(kernel_h * kernel_w) iterations
    // - For typical 3x3 kernels: 9 iterations per thread
    // - For 5x5 kernels: 25 iterations per thread
    // - Benefit: ZERO atomic contention (was causing 2-5x slowdown)
    // - Result: Net speedup despite more work per thread

    // Each thread processes one output element
    int64_t total_output = batch * channels * height * width;

    TENZOR_CUDA_KERNEL_LOOP(output_idx, total_output) {
        // Decode output index to (b, c, ih, iw)
        int64_t temp = output_idx;
        int64_t iw = temp % width; temp /= width;
        int64_t ih = temp % height; temp /= height;
        int64_t c = temp % channels; temp /= channels;
        int64_t b = temp;

        // Accumulate from all kernel positions that contribute to this output
        T sum = T(0);

        // Iterate through kernel positions
        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw_iter = 0; kw_iter < kernel_w; ++kw_iter) {
                // Reverse the im2col mapping: given output (ih, iw) and kernel (kh, kw), find col (oh, ow)
                // Forward mapping: ih = oh * stride - padding + kh * dilation
                // Reverse: oh = (ih + padding - kh * dilation) / stride

                int64_t ih_shifted = ih + padding - kh * dilation;
                int64_t iw_shifted = iw + padding - kw_iter * dilation;

                // Check if this maps to a valid col position
                // Must be divisible by stride to be a valid position
                if (ih_shifted % stride == 0 && iw_shifted % stride == 0) {
                    int64_t oh = ih_shifted / stride;
                    int64_t ow = iw_shifted / stride;

                    // Check bounds
                    if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                        // This kernel position contributes to our output
                        // Calculate col buffer index
                        int64_t col_row = b * out_h * out_w + oh * out_w + ow;
                        int64_t col_col = c * kernel_h * kernel_w + kh * kernel_w + kw_iter;
                        int64_t col_idx = col_row * (channels * kernel_h * kernel_w) + col_col;

                        // Accumulate the contribution
                        sum += col[col_idx];
                    }
                }
            }
        }

        // Direct write - NO ATOMIC NEEDED!
        // This is the key optimization that eliminates the bottleneck
        output[output_idx] = sum;
    }
}

// ============================================================================
// FP16 col2im CUDA Kernel - Output-Centric (No Atomics)
// ============================================================================

// FP16 col2im kernel: Reverse of im2col for gradient computation
// Uses output-centric approach to eliminate atomic operations
__global__ void col2im_kernel_f16(
    const __half* col,
    __half* output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t out_h,
    int64_t out_w
) {
    // Each thread processes one output element
    int64_t total_output = batch * channels * height * width;

    TENZOR_CUDA_KERNEL_LOOP(output_idx, total_output) {
        // Decode output index to (b, c, ih, iw)
        int64_t temp = output_idx;
        int64_t iw = temp % width; temp /= width;
        int64_t ih = temp % height; temp /= height;
        int64_t c = temp % channels; temp /= channels;
        int64_t b = temp;

        // Accumulate from all kernel positions that contribute to this output
        float sum = 0.0f;  // Use float for accumulation to avoid precision loss

        // Iterate through kernel positions
        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw_iter = 0; kw_iter < kernel_w; ++kw_iter) {
                // Reverse the im2col mapping
                int64_t ih_shifted = ih + padding - kh * dilation;
                int64_t iw_shifted = iw + padding - kw_iter * dilation;

                // Check if this maps to a valid col position
                if (ih_shifted % stride == 0 && iw_shifted % stride == 0) {
                    int64_t oh = ih_shifted / stride;
                    int64_t ow = iw_shifted / stride;

                    // Check bounds
                    if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                        // Calculate col buffer index
                        int64_t col_row = b * out_h * out_w + oh * out_w + ow;
                        int64_t col_col = c * kernel_h * kernel_w + kh * kernel_w + kw_iter;
                        int64_t col_idx = col_row * (channels * kernel_h * kernel_w) + col_col;

                        // Accumulate the contribution (convert to float for accuracy)
                        sum += __half2float(col[col_idx]);
                    }
                }
            }
        }

        // Direct write - NO ATOMIC NEEDED!
        output[output_idx] = float2half_sat(sum);
    }
}

// ============================================================================
// Bias Addition Kernel
// ============================================================================

// Simple kernel for bias addition
__global__ void add_bias_kernel(
    float* output,
    const float* bias,
    int64_t batch,
    int64_t channels,
    int64_t spatial_size,
    int64_t n
) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        int64_t c = (idx / spatial_size) % channels;
        output[idx] += bias[c];
    }
}

// Float64 bias addition kernel
__global__ void add_bias_kernel_f64(
    double* output,
    const double* bias,
    int64_t batch,
    int64_t channels,
    int64_t spatial_size,
    int64_t n
) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        int64_t c = (idx / spatial_size) % channels;
        output[idx] += bias[c];
    }
}

// FP16 bias addition kernel
__global__ void add_bias_kernel_f16(
    __half* output,
    const __half* bias,
    int64_t batch,
    int64_t channels,
    int64_t spatial_size,
    int64_t n
) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        int64_t c = (idx / spatial_size) % channels;
        output[idx] = __hadd(output[idx], bias[c]);
    }
}

// ============================================================================
// Bias Gradient Kernel
// ============================================================================

// Kernel to compute bias gradient by summing over spatial dimensions
__global__ void sum_bias_grad_kernel(
    const float* grad_output,
    float* grad_bias,
    int64_t batch,
    int64_t channels,
    int64_t spatial_size
) {
    int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < channels) {
        float sum = 0.0f;
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t s = 0; s < spatial_size; ++s) {
                int64_t idx = b * (channels * spatial_size) + c * spatial_size + s;
                sum += grad_output[idx];
            }
        }
        grad_bias[c] = sum;
    }
}

// FP16 bias gradient kernel
__global__ void sum_bias_grad_kernel_f16(
    const __half* grad_output,
    __half* grad_bias,
    int64_t batch,
    int64_t channels,
    int64_t spatial_size
) {
    int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < channels) {
        float sum = 0.0f;  // Use float for accumulation to avoid precision loss
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t s = 0; s < spatial_size; ++s) {
                int64_t idx = b * (channels * spatial_size) + c * spatial_size + s;
                sum += __half2float(grad_output[idx]);
            }
        }
        grad_bias[c] = float2half_sat(sum);
    }
}

// Float64 bias gradient kernel
__global__ void sum_bias_grad_kernel_f64(
    const double* grad_output,
    double* grad_bias,
    int64_t batch,
    int64_t channels,
    int64_t spatial_size
) {
    int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < channels) {
        double sum = 0.0;
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t s = 0; s < spatial_size; ++s) {
                int64_t idx = b * (channels * spatial_size) + c * spatial_size + s;
                sum += grad_output[idx];
            }
        }
        grad_bias[c] = sum;
    }
}

// ============================================================================
// FP16 Tensor Core Matmul (Forward Declaration)
// ============================================================================

// Forward declaration - implementation is in matmul.cu
void matmul_f16(
    const __half* A, const __half* B, __half* C,
    int64_t M, int64_t N, int64_t K,
    cudaStream_t stream);

// ============================================================================
// Conv2d Forward GPU Implementation
// ============================================================================
//
// NOTE: A standalone conv2d_forward_f16() helper was removed here. It was dead
// code (never declared in any header, never called — the live FP16 conv2d path
// uses cublas GemmEx followed by nhwc_to_nchw_kernel). It wrote the raw
// (out_channels_per_group, batch*out_h*out_w) GEMM result directly into the
// NCHW output WITHOUT the nhwc_to_nchw transpose, so it produced wrong layout
// for batch>1 — a latent trap if it had ever been wired up.

// Conv2d forward using im2col + cuBLAS gemm
auto conv2d_forward_kernel(
    const Tensor& input,         // (batch, in_channels, height, width)
    const Tensor& weight,        // (out_channels, in_channels, kernel_h, kernel_w)
    const Tensor* bias,          // (out_channels) or nullptr
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    cudaStream_t stream
) -> Tensor {
    // Extract dimensions
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t in_channels_per_group = weight_shape[1];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Validate parameters to prevent division by zero
    if (stride == 0) {
        throw std::invalid_argument("Conv2d: stride cannot be zero");
    }
    if (groups == 0) {
        throw std::invalid_argument("Conv2d: groups cannot be zero");
    }

    // Calculate output dimensions
    int64_t out_h = calculate_output_size(height, kernel_h, stride, padding, dilation);
    int64_t out_w = calculate_output_size(width, kernel_w, stride, padding, dilation);

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, out_channels, out_h, out_w};
    Tensor output(output_shape, input.dtype(), input.device());

    // Check dtype and dispatch to appropriate implementation
    if (input.dtype() == DType::Float16) {
        // Mixed-precision path: FP16 I/O with FP32 accumulation via cuBLAS GemmEx.
        // This eliminates the 3x memory overhead of promoting entire tensors to Float32
        // while maintaining numerical stability through FP32 accumulation (matching cuDNN).
        TENZOR_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, output.numel() * sizeof(Float16), stream));

        cublasHandle_t cublas_handle = CuBLASHandlePool::get(stream);

        int64_t out_channels_per_group = out_channels / groups;

        for (int64_t g = 0; g < groups; ++g) {
            int64_t in_start = g * in_channels_per_group;
            int64_t out_start = g * out_channels_per_group;

            int64_t col_rows = batch * out_h * out_w;
            int64_t col_cols = in_channels_per_group * kernel_h * kernel_w;
            backend::CachedMemoryGuard col_buffer_guard(col_rows * col_cols * sizeof(__half));
            auto* col_buffer = static_cast<__half*>(col_buffer_guard.get());

            // im2col in FP16
            dim3 grid, block;
            int64_t total_elements = batch * out_h * out_w * in_channels_per_group * kernel_h * kernel_w;
            compute_launch_config_1d(total_elements, grid, block);

            const __half* input_ptr = reinterpret_cast<const __half*>(
                input.data<Float16>() + in_start * height * width
            );

            im2col_kernel_f16<<<grid, block, 0, stream>>>(
                input_ptr, col_buffer, batch, in_channels_per_group,
                height, width, kernel_h, kernel_w, stride, padding, dilation, out_h, out_w
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();

            // cuBLAS GemmEx: FP16 I/O with FP32 accumulation (Tensor Core accelerated)
            int64_t M = col_rows;
            int64_t K = col_cols;
            int64_t N = out_channels_per_group;

            float alpha = 1.0f;
            float beta = 0.0f;

            const __half* weight_ptr = reinterpret_cast<const __half*>(
                weight.data<Float16>() + out_start * in_channels_per_group * kernel_h * kernel_w
            );

            backend::CachedMemoryGuard temp_output_guard(M * N * sizeof(__half));
            auto* temp_output = static_cast<__half*>(temp_output_guard.get());

            TENZOR_CUBLAS_CHECK(cublasGemmEx(
                cublas_handle,
                CUBLAS_OP_T,    // transpose weight
                CUBLAS_OP_N,    // don't transpose col_buffer
                N, M, K,
                &alpha,
                weight_ptr, CUDA_R_16F, K,
                col_buffer, CUDA_R_16F, K,
                &beta,
                temp_output, CUDA_R_16F, N,
                CUBLAS_COMPUTE_32F,  // FP32 accumulation for numerical stability
                CUBLAS_GEMM_DEFAULT
            ));

            // Transpose NHWC → NCHW
            dim3 t_grid, t_block;
            compute_launch_config_1d(M * N, t_grid, t_block);

            nhwc_to_nchw_kernel<__half><<<t_grid, t_block, 0, stream>>>(
                temp_output,
                reinterpret_cast<__half*>(output.data<Float16>()),
                batch, out_h, out_w, out_channels, N, out_start
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }

        // Add bias if present
        if (bias != nullptr) {
            int64_t spatial_size = out_h * out_w;
            const __half* bias_data = reinterpret_cast<const __half*>(bias->data<Float16>());
            __half* output_data = reinterpret_cast<__half*>(output.data<Float16>());

            dim3 grid, block;
            int64_t total = batch * out_channels * out_h * out_w;
            compute_launch_config_1d(total, grid, block);

            add_bias_kernel_f16<<<grid, block, 0, stream>>>(
                output_data, bias_data, batch, out_channels, spatial_size, total
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }

        return output;
    }

    // Float64 path: im2col + cublasDgemm (mirrors Float32 path below)
    if (input.dtype() == DType::Float64) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(output.data<double>(), 0, output.numel() * sizeof(double), stream));

        cublasHandle_t cublas_handle = CuBLASHandlePool::get(stream);
        int64_t out_channels_per_group = out_channels / groups;

        for (int64_t g = 0; g < groups; ++g) {
            int64_t in_start = g * in_channels_per_group;
            int64_t out_start = g * out_channels_per_group;

            int64_t col_rows = batch * out_h * out_w;
            int64_t col_cols = in_channels_per_group * kernel_h * kernel_w;
            backend::CachedMemoryGuard col_buffer_guard(col_rows * col_cols * sizeof(double));
            auto* col_buffer = static_cast<double*>(col_buffer_guard.get());

            dim3 grid, block;
            int64_t total_elements = batch * out_h * out_w * in_channels_per_group * kernel_h * kernel_w;
            compute_launch_config_1d(total_elements, grid, block);

            const double* input_ptr = input.data<double>() + in_start * height * width;
            im2col_kernel<double><<<grid, block, 0, stream>>>(
                input_ptr, col_buffer,
                batch, in_channels_per_group, height, width,
                kernel_h, kernel_w, stride, padding, dilation, out_h, out_w
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();

            int64_t M = col_rows;
            int64_t K = col_cols;
            int64_t N = out_channels_per_group;

            double alpha = 1.0;
            double beta = 0.0;

            const double* weight_ptr = weight.data<double>() + out_start * in_channels_per_group * kernel_h * kernel_w;

            backend::CachedMemoryGuard temp_output_guard(M * N * sizeof(double));
            auto* temp_output = static_cast<double*>(temp_output_guard.get());

            TENZOR_CUBLAS_CHECK(cublasDgemm(
                cublas_handle,
                CUBLAS_OP_T,    // transpose B (weight)
                CUBLAS_OP_N,    // don't transpose A (col_buffer)
                N, M, K,
                &alpha,
                weight_ptr, K,
                col_buffer, K,
                &beta,
                temp_output, N
            ));

            dim3 transpose_grid, transpose_block;
            compute_launch_config_1d(M * N, transpose_grid, transpose_block);

            nhwc_to_nchw_kernel<double><<<transpose_grid, transpose_block, 0, stream>>>(
                temp_output, output.data<double>(),
                batch, out_h, out_w, out_channels, N, out_start
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }

        if (bias != nullptr) {
            int64_t spatial_size = out_h * out_w;
            const double* bias_data = bias->data<double>();
            double* output_data = output.data<double>();

            dim3 grid, block;
            int64_t total = batch * out_channels * out_h * out_w;
            compute_launch_config_1d(total, grid, block);

            add_bias_kernel_f64<<<grid, block, 0, stream>>>(
                output_data, bias_data, batch, out_channels, spatial_size, total
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }

        return output;
    }

    // Initialize output to zeros (Float32 path)
    TENZOR_CUDA_CHECK(cudaMemsetAsync(output.data<float>(), 0, output.numel() * sizeof(float), stream));

    // Get cached cuBLAS handle (avoids per-call create/destroy overhead)
    cublasHandle_t cublas_handle = CuBLASHandlePool::get(stream);

    // Process each group separately
    int64_t out_channels_per_group = out_channels / groups;

    for (int64_t g = 0; g < groups; ++g) {
        // Calculate channel offsets
        int64_t in_start = g * in_channels_per_group;
        int64_t out_start = g * out_channels_per_group;

        // Allocate im2col buffer for this group
        // Shape: (batch * out_h * out_w, in_channels_per_group * kernel_h * kernel_w)
        int64_t col_rows = batch * out_h * out_w;
        int64_t col_cols = in_channels_per_group * kernel_h * kernel_w;
        backend::CachedMemoryGuard col_buffer_guard(col_rows * col_cols * sizeof(float));
        auto* col_buffer = static_cast<float*>(col_buffer_guard.get());

        // Apply im2col transformation for this group's input channels
        dim3 grid, block;
        int64_t total_elements = batch * out_h * out_w * in_channels_per_group * kernel_h * kernel_w;
        compute_launch_config_1d(total_elements, grid, block);

        // Launch im2col for this group (offset input pointer)
        const float* input_ptr = input.data<float>() + in_start * height * width;
        im2col_kernel<<<grid, block, 0, stream>>>(
            input_ptr,
            col_buffer,
            batch,
            in_channels_per_group,
            height,
            width,
            kernel_h,
            kernel_w,
            stride,
            padding,
            dilation,
            out_h,
            out_w
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();

        // Matrix multiplication using cuBLAS
        // weight_group: (out_channels_per_group, in_channels_per_group * kernel_h * kernel_w)
        // col_buffer: (batch * out_h * out_w, in_channels_per_group * kernel_h * kernel_w)
        // output: (batch * out_h * out_w, out_channels_per_group)
        //
        // We compute: output = col_buffer @ weight_group^T
        // In cuBLAS: C = alpha * op(A) * op(B) + beta * C
        // C: (M, N), A: (M, K), B: (K, N)
        // Here: M = batch * out_h * out_w, K = in_channels_per_group * kernel_h * kernel_w, N = out_channels_per_group

        int64_t M = col_rows;
        int64_t K = col_cols;
        int64_t N = out_channels_per_group;

        float alpha = 1.0f;
        float beta = 0.0f;

        const float* weight_ptr = weight.data<float>() + out_start * in_channels_per_group * kernel_h * kernel_w;

        // Allocate temporary buffer for cuBLAS output (NHWC layout)
        // cuBLAS will write (M, N) = (batch*out_h*out_w, out_channels_per_group) in row-major
        backend::CachedMemoryGuard temp_output_guard(M * N * sizeof(float));
        auto* temp_output = static_cast<float*>(temp_output_guard.get());

        // cuBLAS uses column-major ordering
        // We want: C = A @ B^T where A is row-major (M, K), B is row-major (N, K)
        // In column-major view: C^T = B @ A^T
        // So we compute: C^T = B @ A^T, which means C = (B @ A^T)^T = A @ B^T
        TENZOR_CUBLAS_CHECK(cublasSgemm(
            cublas_handle,
            CUBLAS_OP_T,    // transpose B (weight)
            CUBLAS_OP_N,    // don't transpose A (col_buffer)
            N,              // rows of B^T (out_channels_per_group)
            M,              // rows of A (batch * out_h * out_w)
            K,              // cols of A, rows of B (in_channels_per_group * kernel_h * kernel_w)
            &alpha,
            weight_ptr,     // B (N, K) in row-major = (K, N) in col-major
            K,              // leading dimension of B
            col_buffer,     // A (M, K) in row-major = (K, M) in col-major
            K,              // leading dimension of A
            &beta,
            temp_output,    // C (M, N) in row-major (NHWC layout)
            N               // leading dimension of C
        ));

        // Transpose from NHWC (temp_output) to NCHW (output tensor)
        dim3 transpose_grid, transpose_block;
        int64_t transpose_elements = M * N;
        compute_launch_config_1d(transpose_elements, transpose_grid, transpose_block);

        nhwc_to_nchw_kernel<<<transpose_grid, transpose_block, 0, stream>>>(
            temp_output,
            output.data<float>(),
            batch,
            out_h,
            out_w,
            out_channels,
            N,
            out_start
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();

    }

    // Add bias if present
    if (bias != nullptr) {
        // Broadcast bias across spatial dimensions
        // bias: (out_channels), output: (batch, out_channels, out_h, out_w)
        int64_t spatial_size = out_h * out_w;
        const float* bias_data = bias->data<float>();
        float* output_data = output.data<float>();

        dim3 grid, block;
        int64_t total = batch * out_channels * out_h * out_w;
        compute_launch_config_1d(total, grid, block);

        add_bias_kernel<<<grid, block, 0, stream>>>(
            output_data, bias_data, batch, out_channels, spatial_size, total
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    }

    return output;
}

// ============================================================================
// FP16 Conv2d Backward with Tensor Cores
// ============================================================================

// Complete FP16 Conv2d backward pass using Tensor Core matmul
auto conv2d_backward_f16(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    bool compute_grad_input,
    bool compute_grad_weight,
    bool compute_grad_bias,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    // Extract dimensions
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    auto grad_shape = grad_output.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t in_channels_per_group = weight_shape[1];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    int64_t out_h = grad_shape[2];
    int64_t out_w = grad_shape[3];

    // Validate parameters
    if (stride == 0 || groups == 0) {
        throw std::invalid_argument("Conv2d backward: stride and groups cannot be zero");
    }

    // Initialize outputs
    Tensor grad_input({batch, in_channels, height, width}, DType::Float16, input.device());
    Tensor grad_weight({out_channels, in_channels_per_group, kernel_h, kernel_w}, DType::Float16, weight.device());
    Tensor grad_bias({out_channels}, DType::Float16, weight.device());

    if (compute_grad_input) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(grad_input.data<Float16>(), 0,
                                   grad_input.numel() * sizeof(Float16), stream));
    }
    if (compute_grad_weight) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(grad_weight.data<Float16>(), 0,
                                   grad_weight.numel() * sizeof(Float16), stream));
    }
    if (compute_grad_bias) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(grad_bias.data<Float16>(), 0,
                                   grad_bias.numel() * sizeof(Float16), stream));
    }

    int64_t out_channels_per_group = out_channels / groups;
    int64_t col_rows = batch * out_h * out_w;
    int64_t col_cols = in_channels_per_group * kernel_h * kernel_w;

    for (int64_t g = 0; g < groups; ++g) {
        int64_t in_start = g * in_channels_per_group;
        int64_t out_start = g * out_channels_per_group;

        // Gradient w.r.t input
        if (compute_grad_input) {
            // Allocate col buffer
            backend::CachedMemoryGuard grad_col_guard(col_rows * col_cols * sizeof(__half));
            auto* grad_col = static_cast<__half*>(grad_col_guard.get());

            // Compute grad_col = grad_output @ weight
            // grad_output: (batch * out_h * out_w, out_channels_per_group)
            // weight: (out_channels_per_group, in_channels_per_group * kernel_h * kernel_w)
            // grad_col: (batch * out_h * out_w, in_channels_per_group * kernel_h * kernel_w)

            const __half* grad_out_ptr = reinterpret_cast<const __half*>(
                grad_output.data<Float16>() + out_start * out_h * out_w
            );
            const __half* weight_ptr = reinterpret_cast<const __half*>(
                weight.data<Float16>() + out_start * in_channels_per_group * kernel_h * kernel_w
            );

            // We need: grad_col = grad_output @ weight
            // grad_output is (col_rows, out_channels_per_group), weight is (out_channels_per_group, col_cols)
            // Result is (col_rows, col_cols)

            // Compute: grad_col^T = weight^T @ grad_output^T
            // This gives us (K, N) which is what we want transposed
            matmul_f16(weight_ptr, grad_out_ptr, grad_col, col_cols, col_rows, out_channels_per_group, stream);

            // Apply col2im to accumulate gradients
            dim3 grid, block;
            int64_t total_output = batch * in_channels_per_group * height * width;
            compute_launch_config_1d(total_output, grid, block);

            __half* grad_input_ptr = reinterpret_cast<__half*>(
                grad_input.data<Float16>() + in_start * height * width
            );

            col2im_kernel_f16<<<grid, block, 0, stream>>>(
                grad_col,
                grad_input_ptr,
                batch,
                in_channels_per_group,
                height,
                width,
                kernel_h,
                kernel_w,
                stride,
                padding,
                dilation,
                out_h,
                out_w
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }

        // Gradient w.r.t weight
        if (compute_grad_weight) {
            // Apply im2col to input
            backend::CachedMemoryGuard input_col_guard(col_rows * col_cols * sizeof(__half));
            auto* input_col = static_cast<__half*>(input_col_guard.get());

            dim3 grid, block;
            int64_t total_elements = batch * out_h * out_w * in_channels_per_group * kernel_h * kernel_w;
            compute_launch_config_1d(total_elements, grid, block);

            const __half* input_ptr = reinterpret_cast<const __half*>(
                input.data<Float16>() + in_start * height * width
            );

            im2col_kernel_f16<<<grid, block, 0, stream>>>(
                input_ptr,
                input_col,
                batch,
                in_channels_per_group,
                height,
                width,
                kernel_h,
                kernel_w,
                stride,
                padding,
                dilation,
                out_h,
                out_w
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();

            // Compute grad_weight = grad_output^T @ input_col
            // grad_output: (batch * out_h * out_w, out_channels_per_group)
            // input_col: (batch * out_h * out_w, in_channels_per_group * kernel_h * kernel_w)
            // grad_weight: (out_channels_per_group, in_channels_per_group * kernel_h * kernel_w)

            const __half* grad_out_ptr = reinterpret_cast<const __half*>(
                grad_output.data<Float16>() + out_start * out_h * out_w
            );
            __half* grad_weight_ptr = reinterpret_cast<__half*>(
                grad_weight.data<Float16>() + out_start * in_channels_per_group * kernel_h * kernel_w
            );

            int64_t M = out_channels_per_group;
            int64_t N = col_cols;
            int64_t K = col_rows;

            // Compute: grad_weight = grad_output^T @ input_col
            matmul_f16(grad_out_ptr, input_col, grad_weight_ptr, M, N, K, stream);
        }
    }

    // Gradient w.r.t bias
    if (compute_grad_bias) {
        int64_t spatial_size = out_h * out_w;
        const __half* grad_out_data = reinterpret_cast<const __half*>(grad_output.data<Float16>());
        __half* grad_bias_data = reinterpret_cast<__half*>(grad_bias.data<Float16>());

        dim3 grid, block;
        compute_launch_config_1d(out_channels, grid, block);

        sum_bias_grad_kernel_f16<<<grid, block, 0, stream>>>(
            grad_out_data, grad_bias_data, batch, out_channels, spatial_size
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    }

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// ============================================================================
// Conv2d Backward GPU Implementation
// ============================================================================

// Conv2d backward - computes gradients w.r.t input, weight, and bias
auto conv2d_backward_kernel(
    const Tensor& grad_output,   // (batch, out_channels, out_h, out_w)
    const Tensor& input,         // (batch, in_channels, height, width)
    const Tensor& weight,        // (out_channels, in_channels, kernel_h, kernel_w)
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    bool compute_grad_input,
    bool compute_grad_weight,
    bool compute_grad_bias,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    // Extract dimensions
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    auto grad_shape = grad_output.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t in_channels_per_group = weight_shape[1];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    int64_t out_h = grad_shape[2];
    int64_t out_w = grad_shape[3];

    // Validate parameters to prevent division by zero
    if (stride == 0) {
        throw std::invalid_argument("Conv2d backward: stride cannot be zero");
    }
    if (groups == 0) {
        throw std::invalid_argument("Conv2d backward: groups cannot be zero");
    }

    // Check dtype and dispatch to appropriate implementation
    if (input.dtype() == DType::Float16) {
        // FP16 path with Tensor Cores
        return conv2d_backward_f16(grad_output, input, weight, stride, padding, dilation, groups,
                                   compute_grad_input, compute_grad_weight, compute_grad_bias, stream);
    }

    // Float64 backward: mirror Float32 path with cublasDgemm + double types
    if (input.dtype() == DType::Float64) {
        Tensor grad_input({batch, in_channels, height, width}, DType::Float64, input.device());
        Tensor grad_weight({out_channels, in_channels_per_group, kernel_h, kernel_w}, DType::Float64, weight.device());
        Tensor grad_bias({out_channels}, DType::Float64, weight.device());

        if (compute_grad_input) {
            TENZOR_CUDA_CHECK(cudaMemsetAsync(grad_input.data<double>(), 0, grad_input.numel() * sizeof(double), stream));
        }
        if (compute_grad_weight) {
            TENZOR_CUDA_CHECK(cudaMemsetAsync(grad_weight.data<double>(), 0, grad_weight.numel() * sizeof(double), stream));
        }
        if (compute_grad_bias) {
            TENZOR_CUDA_CHECK(cudaMemsetAsync(grad_bias.data<double>(), 0, grad_bias.numel() * sizeof(double), stream));
        }

        cublasHandle_t cublas_handle = CuBLASHandlePool::get(stream);
        int64_t out_channels_per_group = out_channels / groups;
        int64_t col_rows = batch * out_h * out_w;
        int64_t col_cols = in_channels_per_group * kernel_h * kernel_w;

        for (int64_t g = 0; g < groups; ++g) {
            int64_t in_start = g * in_channels_per_group;
            int64_t out_start = g * out_channels_per_group;

            if (compute_grad_input) {
                backend::CachedMemoryGuard grad_col_guard(col_rows * col_cols * sizeof(double));
                auto* grad_col = static_cast<double*>(grad_col_guard.get());

                int64_t M = col_rows, K = out_channels_per_group, N = col_cols;
                double alpha = 1.0, beta = 0.0;

                const double* grad_out_ptr = grad_output.data<double>() + out_start * out_h * out_w;
                const double* weight_ptr = weight.data<double>() + out_start * in_channels_per_group * kernel_h * kernel_w;

                TENZOR_CUBLAS_CHECK(cublasDgemm(
                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, weight_ptr, N, grad_out_ptr, K, &beta, grad_col, N));

                dim3 grid, block;
                int64_t total_elements = batch * out_h * out_w * in_channels_per_group * kernel_h * kernel_w;
                compute_launch_config_1d(total_elements, grid, block);

                double* grad_input_ptr = grad_input.data<double>() + in_start * height * width;
                col2im_kernel<double><<<grid, block, 0, stream>>>(
                    grad_col, grad_input_ptr, batch, in_channels_per_group,
                    height, width, kernel_h, kernel_w, stride, padding, dilation, out_h, out_w);
                TENZOR_CUDA_POST_LAUNCH_CHECK();
            }

            if (compute_grad_weight) {
                backend::CachedMemoryGuard input_col_guard(col_rows * col_cols * sizeof(double));
                auto* input_col = static_cast<double*>(input_col_guard.get());

                dim3 grid, block;
                int64_t total_elements = batch * out_h * out_w * in_channels_per_group * kernel_h * kernel_w;
                compute_launch_config_1d(total_elements, grid, block);

                const double* input_ptr = input.data<double>() + in_start * height * width;
                im2col_kernel<double><<<grid, block, 0, stream>>>(
                    input_ptr, input_col, batch, in_channels_per_group,
                    height, width, kernel_h, kernel_w, stride, padding, dilation, out_h, out_w);
                TENZOR_CUDA_POST_LAUNCH_CHECK();

                int64_t M = out_channels_per_group, K = col_rows, N = col_cols;
                double alpha = 1.0, beta = 0.0;

                const double* grad_out_ptr = grad_output.data<double>() + out_start * out_h * out_w;
                double* grad_weight_ptr = grad_weight.data<double>() + out_start * in_channels_per_group * kernel_h * kernel_w;

                TENZOR_CUBLAS_CHECK(cublasDgemm(
                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    N, M, K, &alpha, input_col, N, grad_out_ptr, out_channels_per_group, &beta, grad_weight_ptr, N));
            }
        }

        if (compute_grad_bias) {
            int64_t spatial_size = out_h * out_w;
            const double* grad_out_data = grad_output.data<double>();
            double* grad_bias_data = grad_bias.data<double>();

            // Per-channel sum over batch and spatial dims (same pattern as Float32)
            dim3 grid, block;
            compute_launch_config_1d(out_channels, grid, block);
            sum_bias_grad_kernel_f64<<<grid, block, 0, stream>>>(
                grad_out_data, grad_bias_data, batch, out_channels, spatial_size);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }

        return std::make_tuple(grad_input, grad_weight, grad_bias);
    }

    // Initialize outputs (Float32 path)
    Tensor grad_input({batch, in_channels, height, width}, input.dtype(), input.device());
    Tensor grad_weight({out_channels, in_channels_per_group, kernel_h, kernel_w}, weight.dtype(), weight.device());
    Tensor grad_bias({out_channels}, weight.dtype(), weight.device());

    if (compute_grad_input) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(grad_input.data<float>(), 0, grad_input.numel() * sizeof(float), stream));
    }
    if (compute_grad_weight) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(grad_weight.data<float>(), 0, grad_weight.numel() * sizeof(float), stream));
    }
    if (compute_grad_bias) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(grad_bias.data<float>(), 0, grad_bias.numel() * sizeof(float), stream));
    }

    // Get cached cuBLAS handle (avoids per-call create/destroy overhead)
    cublasHandle_t cublas_handle = CuBLASHandlePool::get(stream);

    int64_t out_channels_per_group = out_channels / groups;
    int64_t col_rows = batch * out_h * out_w;
    int64_t col_cols = in_channels_per_group * kernel_h * kernel_w;

    for (int64_t g = 0; g < groups; ++g) {
        int64_t in_start = g * in_channels_per_group;
        int64_t out_start = g * out_channels_per_group;

        // Gradient w.r.t input
        if (compute_grad_input) {
            // Allocate col buffer
            backend::CachedMemoryGuard grad_col_guard(col_rows * col_cols * sizeof(float));
            auto* grad_col = static_cast<float*>(grad_col_guard.get());

            // Compute grad_col = grad_output @ weight
            // grad_output: (batch * out_h * out_w, out_channels_per_group)
            // weight: (out_channels_per_group, in_channels_per_group * kernel_h * kernel_w)
            // grad_col: (batch * out_h * out_w, in_channels_per_group * kernel_h * kernel_w)

            int64_t M = col_rows;
            int64_t K = out_channels_per_group;
            int64_t N = col_cols;

            float alpha = 1.0f;
            float beta = 0.0f;

            const float* grad_out_ptr = grad_output.data<float>() + out_start * out_h * out_w;
            const float* weight_ptr = weight.data<float>() + out_start * in_channels_per_group * kernel_h * kernel_w;

            TENZOR_CUBLAS_CHECK(cublasSgemm(
                cublas_handle,
                CUBLAS_OP_N,    // don't transpose weight
                CUBLAS_OP_N,    // don't transpose grad_output
                N,              // cols of result
                M,              // rows of result
                K,              // inner dimension
                &alpha,
                weight_ptr,     // (K, N) in col-major
                N,              // leading dim
                grad_out_ptr,   // (M, K) in row-major = (K, M) in col-major
                K,              // leading dim
                &beta,
                grad_col,       // (M, N) in row-major = (N, M) in col-major
                N               // leading dim
            ));

            // Apply col2im to accumulate gradients
            dim3 grid, block;
            int64_t total_elements = batch * out_h * out_w * in_channels_per_group * kernel_h * kernel_w;
            compute_launch_config_1d(total_elements, grid, block);

            float* grad_input_ptr = grad_input.data<float>() + in_start * height * width;
            col2im_kernel<<<grid, block, 0, stream>>>(
                grad_col,
                grad_input_ptr,
                batch,
                in_channels_per_group,
                height,
                width,
                kernel_h,
                kernel_w,
                stride,
                padding,
                dilation,
                out_h,
                out_w
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }

        // Gradient w.r.t weight
        if (compute_grad_weight) {
            // Apply im2col to input
            backend::CachedMemoryGuard input_col_guard(col_rows * col_cols * sizeof(float));
            auto* input_col = static_cast<float*>(input_col_guard.get());

            dim3 grid, block;
            int64_t total_elements = batch * out_h * out_w * in_channels_per_group * kernel_h * kernel_w;
            compute_launch_config_1d(total_elements, grid, block);

            const float* input_ptr = input.data<float>() + in_start * height * width;
            im2col_kernel<<<grid, block, 0, stream>>>(
                input_ptr,
                input_col,
                batch,
                in_channels_per_group,
                height,
                width,
                kernel_h,
                kernel_w,
                stride,
                padding,
                dilation,
                out_h,
                out_w
            );
            TENZOR_CUDA_POST_LAUNCH_CHECK();

            // Compute grad_weight = grad_output^T @ input_col
            // grad_output: (batch * out_h * out_w, out_channels_per_group)
            // input_col: (batch * out_h * out_w, in_channels_per_group * kernel_h * kernel_w)
            // grad_weight: (out_channels_per_group, in_channels_per_group * kernel_h * kernel_w)

            int64_t M = out_channels_per_group;
            int64_t K = col_rows;
            int64_t N = col_cols;

            float alpha = 1.0f;
            float beta = 0.0f;

            const float* grad_out_ptr = grad_output.data<float>() + out_start * out_h * out_w;
            float* grad_weight_ptr = grad_weight.data<float>() + out_start * in_channels_per_group * kernel_h * kernel_w;

            TENZOR_CUBLAS_CHECK(cublasSgemm(
                cublas_handle,
                CUBLAS_OP_N,    // don't transpose input_col
                CUBLAS_OP_T,    // transpose grad_output
                N,              // cols of result
                M,              // rows of result
                K,              // inner dimension
                &alpha,
                input_col,      // (K, N) in col-major
                N,              // leading dim
                grad_out_ptr,   // (K, M) in col-major (transposed)
                out_channels_per_group,  // leading dim of original (M, K) in row-major
                &beta,
                grad_weight_ptr, // (M, N) in row-major = (N, M) in col-major
                N               // leading dim
            ));

        }
    }

    // Gradient w.r.t bias
    if (compute_grad_bias) {
        // Sum over batch, height, width dimensions
        int64_t spatial_size = out_h * out_w;
        const float* grad_out_data = grad_output.data<float>();
        float* grad_bias_data = grad_bias.data<float>();

        dim3 grid, block;
        compute_launch_config_1d(out_channels, grid, block);
        sum_bias_grad_kernel<<<grid, block, 0, stream>>>(
            grad_out_data, grad_bias_data, batch, out_channels, spatial_size
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    }

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// ============================================================================
// ConvTranspose2d Forward GPU Implementation
// ============================================================================

// Calculate output size for transposed convolution
__host__ __device__ inline int64_t calculate_transpose_output_size(
    int64_t input_size, int64_t kernel_size,
    int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation
) {
    return (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
}

// ConvTranspose2d forward kernel using gather approach
// Each thread computes one output element by gathering from all contributing input positions.
// Acc is the accumulator type (defaults to T); Float32 dispatch uses Acc=double to match
// conv3d's parity fix (a float accumulator drifts O(sqrt(N)) ULPs vs the CPU reference).
template<typename T, typename Acc = T>
__global__ void conv_transpose2d_forward_kernel_impl(
    const T* input,           // (batch, in_channels, in_h, in_w)
    const T* weight,          // (in_channels, out_channels/groups, kernel_h, kernel_w)
    const T* bias,            // (out_channels) or nullptr
    T* output,                // (batch, out_channels, out_h, out_w)
    int64_t batch,
    int64_t in_channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_channels,
    int64_t out_h,
    int64_t out_w,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,      // M12 fix: per-axis stride/padding/dilation
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    int64_t in_channels_per_group,
    int64_t out_channels_per_group,
    bool has_bias
) {
    int64_t total_output = batch * out_channels * out_h * out_w;

    TENZOR_CUDA_KERNEL_LOOP(idx, total_output) {
        // Decode output position
        int64_t w = idx % out_w;
        int64_t h = (idx / out_w) % out_h;
        int64_t c = (idx / (out_w * out_h)) % out_channels;
        int64_t b = idx / (out_w * out_h * out_channels);

        // Determine group
        int64_t g = c / out_channels_per_group;
        int64_t oc = c % out_channels_per_group;  // Output channel within group
        int64_t in_start = g * in_channels_per_group;

        // Initialize accumulator using the Acc type for proper precision
        Acc sum = Acc(0);

        // Gather from all input positions that contribute to this output position
        for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    // For transposed conv: out = (in - 1) * stride - 2*padding + dilation * (kernel - 1) + out_padding + 1
                    // Inverse: which input position (ih, iw) with kernel (kh, kw) contributes to (h, w)?
                    // oh = ih * stride - padding + kh * dilation
                    // ih * stride = oh + padding - kh * dilation
                    // ih = (oh + padding - kh * dilation) / stride (must be integer and in bounds)

                    int64_t h_shifted = h + padding_h - kh * dilation_h;
                    int64_t w_shifted = w + padding_w - kw * dilation_w;

                    // Check if this maps to a valid input position
                    if (h_shifted >= 0 && h_shifted % stride_h == 0 &&
                        w_shifted >= 0 && w_shifted % stride_w == 0) {

                        int64_t ih = h_shifted / stride_h;
                        int64_t iw = w_shifted / stride_w;

                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            // Get input value
                            int64_t input_idx = b * (in_channels * in_h * in_w) +
                                               (in_start + ic) * (in_h * in_w) +
                                               ih * in_w + iw;
                            T input_val = input[input_idx];

                            // Get weight value
                            // Weight shape: (in_channels, out_channels/groups, kernel_h, kernel_w)
                            int64_t weight_idx = (in_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                                                oc * (kernel_h * kernel_w) +
                                                kh * kernel_w + kw;
                            T weight_val = weight[weight_idx];

                            sum += static_cast<Acc>(input_val) * static_cast<Acc>(weight_val);
                        }
                    }
                }
            }
        }

        // Add bias if present
        if (has_bias) {
            sum += static_cast<Acc>(bias[c]);
        }

        output[idx] = static_cast<T>(sum);
    }
}

// FP16 specialization of ConvTranspose2d kernel
__global__ void conv_transpose2d_forward_kernel_f16(
    const __half* input,
    const __half* weight,
    const __half* bias,
    __half* output,
    int64_t batch,
    int64_t in_channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_channels,
    int64_t out_h,
    int64_t out_w,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,      // M12 fix: per-axis
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    int64_t in_channels_per_group,
    int64_t out_channels_per_group,
    bool has_bias
) {
    int64_t total_output = batch * out_channels * out_h * out_w;

    TENZOR_CUDA_KERNEL_LOOP(idx, total_output) {
        // Decode output position
        int64_t w = idx % out_w;
        int64_t h = (idx / out_w) % out_h;
        int64_t c = (idx / (out_w * out_h)) % out_channels;
        int64_t b = idx / (out_w * out_h * out_channels);

        // Determine group
        int64_t g = c / out_channels_per_group;
        int64_t oc = c % out_channels_per_group;
        int64_t in_start = g * in_channels_per_group;

        // Use float for accumulation to avoid precision loss
        float sum = 0.0f;

        // Gather from all input positions
        for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t h_shifted = h + padding_h - kh * dilation_h;
                    int64_t w_shifted = w + padding_w - kw * dilation_w;

                    if (h_shifted >= 0 && h_shifted % stride_h == 0 &&
                        w_shifted >= 0 && w_shifted % stride_w == 0) {

                        int64_t ih = h_shifted / stride_h;
                        int64_t iw = w_shifted / stride_w;

                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            int64_t input_idx = b * (in_channels * in_h * in_w) +
                                               (in_start + ic) * (in_h * in_w) +
                                               ih * in_w + iw;
                            float input_val = __half2float(input[input_idx]);

                            int64_t weight_idx = (in_start + ic) * (out_channels_per_group * kernel_h * kernel_w) +
                                                oc * (kernel_h * kernel_w) +
                                                kh * kernel_w + kw;
                            float weight_val = __half2float(weight[weight_idx]);

                            sum += input_val * weight_val;
                        }
                    }
                }
            }
        }

        // Add bias if present
        if (has_bias) {
            sum += __half2float(bias[c]);
        }

        output[idx] = float2half_sat(sum);
    }
}

// ConvTranspose2d forward wrapper function
// M12 fix: per-axis stride/padding/output_padding/dilation wrapper. The
// scalar version is preserved as a thin shim below for back-compat.
auto conv_transpose2d_forward_kernel(
    const Tensor& input,          // (batch, in_channels, in_h, in_w)
    const Tensor& weight,         // (in_channels, out_channels/groups, kernel_h, kernel_w)
    const Tensor* bias,           // (out_channels) or nullptr
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t output_padding_h,
    int64_t output_padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    cudaStream_t stream
) -> Tensor {
    // Extract dimensions
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];

    int64_t in_channels_per_group = weight_shape[0] / groups;
    int64_t out_channels_per_group = weight_shape[1];
    int64_t out_channels = out_channels_per_group * groups;
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Calculate output dimensions for transposed convolution (per-axis).
    int64_t out_h = calculate_transpose_output_size(in_h, kernel_h, stride_h, padding_h, output_padding_h, dilation_h);
    int64_t out_w = calculate_transpose_output_size(in_w, kernel_w, stride_w, padding_w, output_padding_w, dilation_w);

    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument(
            "Invalid ConvTranspose2d configuration: output dimensions are non-positive (out_h=" +
            std::to_string(out_h) + ", out_w=" + std::to_string(out_w) + ")"
        );
    }

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, out_channels, out_h, out_w};
    Tensor output(output_shape, input.dtype(), input.device());

    // Compute launch configuration
    int64_t total_output = batch * out_channels * out_h * out_w;
    dim3 grid, block;
    compute_launch_config_1d(total_output, grid, block);

    bool has_bias = (bias != nullptr);

    // Dispatch based on dtype
    if (input.dtype() == DType::Float16) {
        const __half* input_ptr = reinterpret_cast<const __half*>(input.data<Float16>());
        const __half* weight_ptr = reinterpret_cast<const __half*>(weight.data<Float16>());
        const __half* bias_ptr = has_bias ? reinterpret_cast<const __half*>(bias->data<Float16>()) : nullptr;
        __half* output_ptr = reinterpret_cast<__half*>(output.data<Float16>());

        conv_transpose2d_forward_kernel_f16<<<grid, block, 0, stream>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch, in_channels, in_h, in_w,
            out_channels, out_h, out_w,
            kernel_h, kernel_w,
            stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,
            in_channels_per_group, out_channels_per_group,
            has_bias
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        // Float64 path
        const double* input_ptr = input.data<double>();
        const double* weight_ptr = weight.data<double>();
        const double* bias_ptr = has_bias ? bias->data<double>() : nullptr;
        double* output_ptr = output.data<double>();

        conv_transpose2d_forward_kernel_impl<double><<<grid, block, 0, stream>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch, in_channels, in_h, in_w,
            out_channels, out_h, out_w,
            kernel_h, kernel_w,
            stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,
            in_channels_per_group, out_channels_per_group,
            has_bias
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        // Float32 path (default)
        const float* input_ptr = input.data<float>();
        const float* weight_ptr = weight.data<float>();
        const float* bias_ptr = has_bias ? bias->data<float>() : nullptr;
        float* output_ptr = output.data<float>();

        conv_transpose2d_forward_kernel_impl<float, double><<<grid, block, 0, stream>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch, in_channels, in_h, in_w,
            out_channels, out_h, out_w,
            kernel_h, kernel_w,
            stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,
            in_channels_per_group, out_channels_per_group,
            has_bias
        );
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return output;
}

// RR.9: scalar-arg overload removed.  The dispatcher
// (cuda_kernel_registry.cpp) now always reads per-axis stride/padding/
// output_padding/dilation via TENZOR_READ_CONVT2D_ATTRS() and calls the
// per-axis overload directly.  The previous shim duplicated a single
// `output_padding` into both axes, silently squashing asymmetric values.


// ============================================================================
// Depthwise Conv2d CUDA Kernels
// ============================================================================

// Original kernel (fallback for large filter sizes > MAX_DEPTHWISE_FILTER)
template<typename T>
__global__ void depthwise_conv2d_forward_kernel_impl(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int64_t batch, int64_t channels,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w,
    bool has_bias) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch * channels * out_h * out_w;
    if (idx >= total) return;

    int64_t ow = idx % out_w;
    int64_t oh = (idx / out_w) % out_h;
    int64_t c = (idx / (out_w * out_h)) % channels;
    int64_t n = idx / (out_w * out_h * channels);

    T sum = T(0);
    for (int64_t kh = 0; kh < kernel_h; ++kh) {
        for (int64_t kw = 0; kw < kernel_w; ++kw) {
            int64_t ih = oh * stride_h - pad_h + kh * dilation_h;
            int64_t iw = ow * stride_w - pad_w + kw * dilation_w;

            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int64_t input_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                int64_t weight_idx = (c * kernel_h + kh) * kernel_w + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    if (has_bias) {
        sum += bias[c];
    }

    output[idx] = sum;
}

// ============================================================================
// Shared Memory Depthwise Conv2d - Caches filter weights in shared memory
// ============================================================================
// For typical depthwise convolutions (3x3, 5x5, 7x7), filter weights are
// small enough to fit entirely in shared memory. Each block handles one
// channel and loads the filter once, then all threads in the block reuse it.
// Grid: (ceil(out_h * out_w / blockDim.x), channels, batch)
//
// Maximum supported filter size: 11x11 = 121 elements (484 bytes for float)

constexpr int MAX_DEPTHWISE_FILTER_ELEMS = 121;  // 11x11

template<typename T>
__global__ void depthwise_conv2d_smem_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w,
    bool has_bias) {

    // Grid mapping: blockIdx.z = batch, blockIdx.y = channel
    int64_t n = blockIdx.z;
    int64_t c = blockIdx.y;
    int64_t channels = gridDim.y;
    int64_t filter_size = kernel_h * kernel_w;

    // Load filter weights into shared memory (one load per block)
    extern __shared__ char smem_raw[];
    T* smem_filter = reinterpret_cast<T*>(smem_raw);

    for (int i = threadIdx.x; i < filter_size; i += blockDim.x) {
        smem_filter[i] = weight[c * filter_size + i];
    }
    __syncthreads();

    // Each thread computes one output spatial position
    int64_t spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (spatial_idx >= out_h * out_w) return;

    int64_t oh = spatial_idx / out_w;
    int64_t ow = spatial_idx % out_w;

    T sum = T(0);
    for (int64_t kh = 0; kh < kernel_h; ++kh) {
        int64_t ih = oh * stride_h - pad_h + kh * dilation_h;
        if (ih < 0 || ih >= in_h) continue;

        for (int64_t kw = 0; kw < kernel_w; ++kw) {
            int64_t iw = ow * stride_w - pad_w + kw * dilation_w;
            if (iw < 0 || iw >= in_w) continue;

            int64_t input_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
            sum += input[input_idx] * smem_filter[kh * kernel_w + kw];
        }
    }

    if (has_bias) {
        sum += bias[c];
    }

    output[((n * channels + c) * out_h + oh) * out_w + ow] = sum;
}

// FP16 specialization using FP32 accumulation with shared memory filter cache
__global__ void depthwise_conv2d_smem_kernel_f16(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w,
    bool has_bias) {

    int64_t n = blockIdx.z;
    int64_t c = blockIdx.y;
    int64_t channels = gridDim.y;
    int64_t filter_size = kernel_h * kernel_w;

    extern __shared__ char smem_raw_f16[];
    __half* smem_filter = reinterpret_cast<__half*>(smem_raw_f16);

    for (int i = threadIdx.x; i < filter_size; i += blockDim.x) {
        smem_filter[i] = weight[c * filter_size + i];
    }
    __syncthreads();

    int64_t spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (spatial_idx >= out_h * out_w) return;

    int64_t oh = spatial_idx / out_w;
    int64_t ow = spatial_idx % out_w;

    // FP32 accumulation for numerical stability
    float sum = 0.0f;
    for (int64_t kh = 0; kh < kernel_h; ++kh) {
        int64_t ih = oh * stride_h - pad_h + kh * dilation_h;
        if (ih < 0 || ih >= in_h) continue;

        for (int64_t kw = 0; kw < kernel_w; ++kw) {
            int64_t iw = ow * stride_w - pad_w + kw * dilation_w;
            if (iw < 0 || iw >= in_w) continue;

            int64_t input_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
            sum += __half2float(input[input_idx]) * __half2float(smem_filter[kh * kernel_w + kw]);
        }
    }

    if (has_bias) {
        sum += __half2float(bias[c]);
    }

    output[((n * channels + c) * out_h + oh) * out_w + ow] = float2half_sat(sum);
}

__global__ void depthwise_conv2d_forward_kernel_f16(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int64_t batch, int64_t channels,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w,
    bool has_bias) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch * channels * out_h * out_w;
    if (idx >= total) return;

    int64_t ow = idx % out_w;
    int64_t oh = (idx / out_w) % out_h;
    int64_t c = (idx / (out_w * out_h)) % channels;
    int64_t n = idx / (out_w * out_h * channels);

    float sum = 0.0f;
    for (int64_t kh = 0; kh < kernel_h; ++kh) {
        for (int64_t kw = 0; kw < kernel_w; ++kw) {
            int64_t ih = oh * stride_h - pad_h + kh * dilation_h;
            int64_t iw = ow * stride_w - pad_w + kw * dilation_w;

            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                int64_t input_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                int64_t weight_idx = (c * kernel_h + kh) * kernel_w + kw;
                sum += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
            }
        }
    }

    if (has_bias) {
        sum += __half2float(bias[c]);
    }

    output[idx] = float2half_sat(sum);
}

auto depthwise_conv2d_forward_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    cudaStream_t stream
) -> Tensor;

auto depthwise_conv2d_forward_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    cudaStream_t stream
) -> Tensor {
    return depthwise_conv2d_forward_kernel(
        input, weight, bias,
        stride, stride,
        padding, padding,
        dilation, dilation,
        stream);
}

auto depthwise_conv2d_forward_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    cudaStream_t stream
) -> Tensor {
    // BFloat16 has no dedicated depthwise kernel (only Float16 does). Widen to
    // Float32, compute, then narrow back. Mirrors the widen path other conv2d
    // entry points use; without this BF16 hit "unsupported dtype".
    if (input.dtype() == DType::BFloat16) {
        Tensor f_in = input.to(DType::Float32);
        Tensor f_w = weight.to(DType::Float32);
        if (bias != nullptr) {
            Tensor f_bias = bias->to(DType::Float32);
            return depthwise_conv2d_forward_kernel(
                f_in, f_w, &f_bias,
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, stream
            ).to(DType::BFloat16);
        }
        return depthwise_conv2d_forward_kernel(
            f_in, f_w, nullptr,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, stream
        ).to(DType::BFloat16);
    }

    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    int64_t out_h = (in_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_w = (in_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;

    Tensor output({batch, channels, out_h, out_w}, input.dtype(), input.device());

    bool has_bias = (bias != nullptr);
    int64_t filter_elems = kernel_h * kernel_w;

    // Use shared memory kernel when filter fits (typical: 3x3, 5x5, 7x7, up to 11x11)
    bool use_smem = (filter_elems <= MAX_DEPTHWISE_FILTER_ELEMS);

    if (use_smem) {
        // Grid: (spatial_blocks, channels, batch) — each block handles one (n, c) pair's spatial tile
        int64_t spatial_total = out_h * out_w;
        int block_size = 256;
        int spatial_blocks = static_cast<int>((spatial_total + block_size - 1) / block_size);
        dim3 grid(spatial_blocks, static_cast<unsigned int>(channels), static_cast<unsigned int>(batch));

        if (input.dtype() == DType::Float32) {
            size_t smem_bytes = filter_elems * sizeof(float);
            depthwise_conv2d_smem_kernel<float><<<grid, block_size, smem_bytes, stream>>>(
                input.data<float>(), weight.data<float>(),
                has_bias ? bias->data<float>() : nullptr,
                output.data<float>(),
                in_h, in_w, out_h, out_w,
                kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                dil_h, dil_w, has_bias);
        } else if (input.dtype() == DType::Float64) {
            size_t smem_bytes = filter_elems * sizeof(double);
            depthwise_conv2d_smem_kernel<double><<<grid, block_size, smem_bytes, stream>>>(
                input.data<double>(), weight.data<double>(),
                has_bias ? bias->data<double>() : nullptr,
                output.data<double>(),
                in_h, in_w, out_h, out_w,
                kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                dil_h, dil_w, has_bias);
        } else if (input.dtype() == DType::Float16) {
            size_t smem_bytes = filter_elems * sizeof(__half);
            depthwise_conv2d_smem_kernel_f16<<<grid, block_size, smem_bytes, stream>>>(
                reinterpret_cast<const __half*>(input.data_ptr()),
                reinterpret_cast<const __half*>(weight.data_ptr()),
                has_bias ? reinterpret_cast<const __half*>(bias->data_ptr()) : nullptr,
                reinterpret_cast<__half*>(output.data_ptr()),
                in_h, in_w, out_h, out_w,
                kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                dil_h, dil_w, has_bias);
        } else {
            throw std::runtime_error("depthwise_conv2d_forward: unsupported dtype");
        }
    } else {
        // Fallback: original flat-grid kernel for very large filters
        int64_t total = batch * channels * out_h * out_w;
        int block_size = 256;
        int num_blocks = static_cast<int>((total + block_size - 1) / block_size);

        if (input.dtype() == DType::Float32) {
            depthwise_conv2d_forward_kernel_impl<float><<<num_blocks, block_size, 0, stream>>>(
                input.data<float>(), weight.data<float>(),
                has_bias ? bias->data<float>() : nullptr,
                output.data<float>(),
                batch, channels, in_h, in_w, out_h, out_w,
                kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                dil_h, dil_w, has_bias);
        } else if (input.dtype() == DType::Float64) {
            depthwise_conv2d_forward_kernel_impl<double><<<num_blocks, block_size, 0, stream>>>(
                input.data<double>(), weight.data<double>(),
                has_bias ? bias->data<double>() : nullptr,
                output.data<double>(),
                batch, channels, in_h, in_w, out_h, out_w,
                kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                dil_h, dil_w, has_bias);
        } else if (input.dtype() == DType::Float16) {
            depthwise_conv2d_forward_kernel_f16<<<num_blocks, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(input.data_ptr()),
                reinterpret_cast<const __half*>(weight.data_ptr()),
                has_bias ? reinterpret_cast<const __half*>(bias->data_ptr()) : nullptr,
                reinterpret_cast<__half*>(output.data_ptr()),
                batch, channels, in_h, in_w, out_h, out_w,
                kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                dil_h, dil_w, has_bias);
        } else {
            throw std::runtime_error("depthwise_conv2d_forward: unsupported dtype");
        }
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return output;
}

// ============================================================================
// Deformable Conv2d (DCNv2) CUDA Kernels
// ============================================================================

/**
 * @brief Bilinear interpolation sampling from a single-channel 2D feature map.
 *
 * Computes the bilinearly-interpolated value at continuous location (h, w)
 * within a feature map of spatial size (height x width).  Out-of-bounds
 * locations return zero.
 */
template<typename T>
__device__ inline T deformable_bilinear_sample(
    const T* data, int64_t height, int64_t width,
    T h, T w)
{
    if (h <= static_cast<T>(-1) || h >= static_cast<T>(height) ||
        w <= static_cast<T>(-1) || w >= static_cast<T>(width)) {
        return static_cast<T>(0);
    }

    int64_t h_low = static_cast<int64_t>(floor(static_cast<double>(h)));
    int64_t w_low = static_cast<int64_t>(floor(static_cast<double>(w)));
    int64_t h_high = h_low + 1;
    int64_t w_high = w_low + 1;

    T lh = h - static_cast<T>(h_low);
    T lw = w - static_cast<T>(w_low);
    T hh = static_cast<T>(1) - lh;
    T hw = static_cast<T>(1) - lw;

    T v1 = (h_low >= 0 && w_low >= 0)   ? data[h_low  * width + w_low]  : static_cast<T>(0);
    T v2 = (h_low >= 0 && w_high < width)  ? data[h_low  * width + w_high] : static_cast<T>(0);
    T v3 = (h_high < height && w_low >= 0) ? data[h_high * width + w_low]  : static_cast<T>(0);
    T v4 = (h_high < height && w_high < width) ? data[h_high * width + w_high] : static_cast<T>(0);

    return hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4;
}

/**
 * @brief Partial derivatives of bilinear interpolation w.r.t. spatial coords.
 *
 * Returns (dval/dh, dval/dw) at the continuous position (h, w).
 */
template<typename T>
__device__ inline void deformable_bilinear_sample_grad_hw(
    const T* data, int64_t height, int64_t width,
    T h, T w,
    T& grad_h, T& grad_w)
{
    grad_h = static_cast<T>(0);
    grad_w = static_cast<T>(0);

    if (h <= static_cast<T>(-1) || h >= static_cast<T>(height) ||
        w <= static_cast<T>(-1) || w >= static_cast<T>(width)) {
        return;
    }

    int64_t h_low = static_cast<int64_t>(floor(static_cast<double>(h)));
    int64_t w_low = static_cast<int64_t>(floor(static_cast<double>(w)));
    int64_t h_high = h_low + 1;
    int64_t w_high = w_low + 1;

    T lh = h - static_cast<T>(h_low);
    T lw = w - static_cast<T>(w_low);
    T hh = static_cast<T>(1) - lh;
    T hw = static_cast<T>(1) - lw;

    T v1 = (h_low >= 0 && w_low >= 0)          ? data[h_low  * width + w_low]  : static_cast<T>(0);
    T v2 = (h_low >= 0 && w_high < width)       ? data[h_low  * width + w_high] : static_cast<T>(0);
    T v3 = (h_high < height && w_low >= 0)      ? data[h_high * width + w_low]  : static_cast<T>(0);
    T v4 = (h_high < height && w_high < width)  ? data[h_high * width + w_high] : static_cast<T>(0);

    // d/dh of bilinear: hh = (1-lh), lh = (h - h_low)
    //   d hh/dh = -1, d lh/dh = +1
    grad_h = -hw * v1 - lw * v2 + hw * v3 + lw * v4;
    // d/dw of bilinear: hw = (1-lw), lw = (w - w_low)
    grad_w = -hh * v1 + hh * v2 - lh * v3 + lh * v4;
}

/**
 * @brief Atomically scatter bilinear interpolation gradients to grad_input.
 *
 * Given a gradient scalar `val` to distribute at continuous position (h, w),
 * scatters the four bilinear-weighted contributions via atomicAdd.
 */
template<typename T>
__device__ inline void deformable_bilinear_scatter(
    T* grad_data, int64_t height, int64_t width,
    T h, T w, T val)
{
    if (h <= static_cast<T>(-1) || h >= static_cast<T>(height) ||
        w <= static_cast<T>(-1) || w >= static_cast<T>(width)) {
        return;
    }

    int64_t h_low = static_cast<int64_t>(floor(static_cast<double>(h)));
    int64_t w_low = static_cast<int64_t>(floor(static_cast<double>(w)));
    int64_t h_high = h_low + 1;
    int64_t w_high = w_low + 1;

    T lh = h - static_cast<T>(h_low);
    T lw = w - static_cast<T>(w_low);
    T hh = static_cast<T>(1) - lh;
    T hw = static_cast<T>(1) - lw;

    if (h_low >= 0 && w_low >= 0)
        atomicAdd(&grad_data[h_low * width + w_low], hh * hw * val);
    if (h_low >= 0 && w_high < width)
        atomicAdd(&grad_data[h_low * width + w_high], hh * lw * val);
    if (h_high < height && w_low >= 0)
        atomicAdd(&grad_data[h_high * width + w_low], lh * hw * val);
    if (h_high < height && w_high < width)
        atomicAdd(&grad_data[h_high * width + w_high], lh * lw * val);
}

// Specialization for double atomicAdd (not natively supported on all archs)
__device__ inline double atomicAdd_double(double* address, double val) {
    unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

template<>
__device__ inline void deformable_bilinear_scatter<double>(
    double* grad_data, int64_t height, int64_t width,
    double h, double w, double val)
{
    if (h <= -1.0 || h >= static_cast<double>(height) ||
        w <= -1.0 || w >= static_cast<double>(width)) {
        return;
    }

    int64_t h_low = static_cast<int64_t>(floor(h));
    int64_t w_low = static_cast<int64_t>(floor(w));
    int64_t h_high = h_low + 1;
    int64_t w_high = w_low + 1;

    double lh = h - static_cast<double>(h_low);
    double lw = w - static_cast<double>(w_low);
    double hh = 1.0 - lh;
    double hw = 1.0 - lw;

    if (h_low >= 0 && w_low >= 0)
        atomicAdd_double(&grad_data[h_low * width + w_low], hh * hw * val);
    if (h_low >= 0 && w_high < width)
        atomicAdd_double(&grad_data[h_low * width + w_high], hh * lw * val);
    if (h_high < height && w_low >= 0)
        atomicAdd_double(&grad_data[h_high * width + w_low], lh * hw * val);
    if (h_high < height && w_high < width)
        atomicAdd_double(&grad_data[h_high * width + w_high], lh * lw * val);
}

// ============================================================================
// DCNv2 Forward Kernel
// ============================================================================

template<typename T>
__global__ void deformable_conv2d_forward_impl(
    const T* input,          // (N, C_in, H_in, W_in)
    const T* offset,         // (N, offset_groups * 2 * kH * kW, H_out, W_out)
    const T* weight,         // (C_out, C_in/groups, kH, kW)
    const T* bias,           // (C_out,) or nullptr
    const T* mask,           // (N, offset_groups * kH * kW, H_out, W_out) or nullptr
    T* output,               // (N, C_out, H_out, W_out)
    int64_t batch, int64_t in_channels, int64_t in_h, int64_t in_w,
    int64_t out_channels, int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    bool has_bias, bool has_mask)
{
    int64_t total = batch * out_channels * out_h * out_w;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        // Decode flat index -> (n, oc, oh, ow)
        int64_t ow = idx % out_w;
        int64_t oh = (idx / out_w) % out_h;
        int64_t oc = (idx / (out_w * out_h)) % out_channels;
        int64_t n  = idx / (out_w * out_h * out_channels);

        int64_t out_channels_per_group = out_channels / groups;
        int64_t in_channels_per_group = in_channels / groups;
        int64_t g = oc / out_channels_per_group;

        // Which offset group does this output channel belong to?
        int64_t channels_per_offset_group = in_channels / offset_groups;

        T sum = static_cast<T>(0);

        for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
            int64_t c_in = g * in_channels_per_group + ic;
            // Determine offset group for this input channel
            int64_t og = c_in / channels_per_offset_group;

            const T* input_channel = input + (n * in_channels + c_in) * in_h * in_w;

            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t offset_idx_base = og * 2 * kernel_h * kernel_w;
                    int64_t offset_h_idx = offset_idx_base + 2 * (kh * kernel_w + kw);
                    int64_t offset_w_idx = offset_h_idx + 1;

                    // Offset tensor layout: (N, offset_groups*2*kH*kW, H_out, W_out)
                    T off_h = offset[((n * offset_groups * 2 * kernel_h * kernel_w + offset_h_idx) * out_h + oh) * out_w + ow];
                    T off_w = offset[((n * offset_groups * 2 * kernel_h * kernel_w + offset_w_idx) * out_h + oh) * out_w + ow];

                    T h_in = static_cast<T>(oh * stride_h - pad_h + kh * dil_h) + off_h;
                    T w_in = static_cast<T>(ow * stride_w - pad_w + kw * dil_w) + off_w;

                    T val = deformable_bilinear_sample(input_channel, in_h, in_w, h_in, w_in);

                    // DCNv2 modulation mask
                    if (has_mask) {
                        int64_t mask_idx_base = og * kernel_h * kernel_w;
                        int64_t mask_idx = mask_idx_base + kh * kernel_w + kw;
                        T m = mask[((n * offset_groups * kernel_h * kernel_w + mask_idx) * out_h + oh) * out_w + ow];
                        val *= m;
                    }

                    // Weight layout: (C_out, C_in/groups, kH, kW)
                    int64_t w_idx = ((oc * in_channels_per_group + ic) * kernel_h + kh) * kernel_w + kw;
                    sum += val * weight[w_idx];
                }
            }
        }

        if (has_bias) {
            sum += bias[oc];
        }

        output[idx] = sum;
    }
}

// ============================================================================
// DCNv2 Backward Input/Offset/Mask Kernel
// ============================================================================

template<typename T>
__global__ void deformable_conv2d_backward_input_impl(
    const T* grad_output,    // (N, C_out, H_out, W_out)
    const T* input,          // (N, C_in, H_in, W_in)
    const T* offset,         // (N, offset_groups * 2 * kH * kW, H_out, W_out)
    const T* weight,         // (C_out, C_in/groups, kH, kW)
    const T* mask,           // (N, offset_groups * kH * kW, H_out, W_out) or nullptr
    T* grad_input,           // (N, C_in, H_in, W_in)
    T* grad_offset,          // (N, offset_groups * 2 * kH * kW, H_out, W_out)
    T* grad_mask,            // (N, offset_groups * kH * kW, H_out, W_out) or nullptr
    int64_t batch, int64_t in_channels, int64_t in_h, int64_t in_w,
    int64_t out_channels, int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    bool has_mask)
{
    // Iterate over (n, og, kh, kw, oh, ow) for offset/mask gradients
    // and scatter grad_input via bilinear
    int64_t total = batch * out_channels * out_h * out_w;
    int64_t out_channels_per_group = out_channels / groups;
    int64_t in_channels_per_group = in_channels / groups;
    int64_t channels_per_offset_group = in_channels / offset_groups;

    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % out_w;
        int64_t oh = (idx / out_w) % out_h;
        int64_t oc = (idx / (out_w * out_h)) % out_channels;
        int64_t n  = idx / (out_w * out_h * out_channels);

        int64_t g = oc / out_channels_per_group;
        T grad_out_val = grad_output[idx];

        for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
            int64_t c_in = g * in_channels_per_group + ic;
            int64_t og = c_in / channels_per_offset_group;

            const T* input_channel = input + (n * in_channels + c_in) * in_h * in_w;
            T* grad_input_channel = grad_input + (n * in_channels + c_in) * in_h * in_w;

            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t offset_idx_base = og * 2 * kernel_h * kernel_w;
                    int64_t offset_h_idx = offset_idx_base + 2 * (kh * kernel_w + kw);
                    int64_t offset_w_idx = offset_h_idx + 1;

                    T off_h = offset[((n * offset_groups * 2 * kernel_h * kernel_w + offset_h_idx) * out_h + oh) * out_w + ow];
                    T off_w = offset[((n * offset_groups * 2 * kernel_h * kernel_w + offset_w_idx) * out_h + oh) * out_w + ow];

                    T h_in = static_cast<T>(oh * stride_h - pad_h + kh * dil_h) + off_h;
                    T w_in = static_cast<T>(ow * stride_w - pad_w + kw * dil_w) + off_w;

                    int64_t w_weight_idx = ((oc * in_channels_per_group + ic) * kernel_h + kh) * kernel_w + kw;
                    T w_val = weight[w_weight_idx];

                    T sampled_val = deformable_bilinear_sample(input_channel, in_h, in_w, h_in, w_in);

                    T m_val = static_cast<T>(1);
                    if (has_mask) {
                        int64_t mask_idx_base = og * kernel_h * kernel_w;
                        int64_t mask_idx = mask_idx_base + kh * kernel_w + kw;
                        m_val = mask[((n * offset_groups * kernel_h * kernel_w + mask_idx) * out_h + oh) * out_w + ow];
                    }

                    // grad_input: scatter bilinear
                    T grad_to_scatter = grad_out_val * w_val * m_val;
                    deformable_bilinear_scatter(grad_input_channel, in_h, in_w, h_in, w_in, grad_to_scatter);

                    // grad_offset: d/dh and d/dw of bilinear * weight * mask * grad_output
                    T dval_dh, dval_dw;
                    deformable_bilinear_sample_grad_hw(input_channel, in_h, in_w, h_in, w_in, dval_dh, dval_dw);

                    T grad_offset_h = grad_out_val * w_val * m_val * dval_dh;
                    T grad_offset_w = grad_out_val * w_val * m_val * dval_dw;

                    int64_t off_h_flat = ((n * offset_groups * 2 * kernel_h * kernel_w + offset_h_idx) * out_h + oh) * out_w + ow;
                    int64_t off_w_flat = ((n * offset_groups * 2 * kernel_h * kernel_w + offset_w_idx) * out_h + oh) * out_w + ow;

                    atomicAdd(&grad_offset[off_h_flat], grad_offset_h);
                    atomicAdd(&grad_offset[off_w_flat], grad_offset_w);

                    // grad_mask: sampled_val * weight * grad_output
                    if (has_mask) {
                        int64_t mask_idx_base = og * kernel_h * kernel_w;
                        int64_t mask_idx = mask_idx_base + kh * kernel_w + kw;
                        int64_t mask_flat = ((n * offset_groups * kernel_h * kernel_w + mask_idx) * out_h + oh) * out_w + ow;
                        T grad_mask_val = grad_out_val * w_val * sampled_val;
                        atomicAdd(&grad_mask[mask_flat], grad_mask_val);
                    }
                }
            }
        }
    }
}

// Double specialization for atomicAdd in backward input kernel
template<>
__global__ void deformable_conv2d_backward_input_impl<double>(
    const double* grad_output,
    const double* input,
    const double* offset,
    const double* weight,
    const double* mask,
    double* grad_input,
    double* grad_offset,
    double* grad_mask,
    int64_t batch, int64_t in_channels, int64_t in_h, int64_t in_w,
    int64_t out_channels, int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    bool has_mask)
{
    int64_t total = batch * out_channels * out_h * out_w;
    int64_t out_channels_per_group = out_channels / groups;
    int64_t in_channels_per_group = in_channels / groups;
    int64_t channels_per_offset_group = in_channels / offset_groups;

    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % out_w;
        int64_t oh = (idx / out_w) % out_h;
        int64_t oc = (idx / (out_w * out_h)) % out_channels;
        int64_t n  = idx / (out_w * out_h * out_channels);

        int64_t g = oc / out_channels_per_group;
        double grad_out_val = grad_output[idx];

        for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
            int64_t c_in = g * in_channels_per_group + ic;
            int64_t og = c_in / channels_per_offset_group;

            const double* input_channel = input + (n * in_channels + c_in) * in_h * in_w;
            double* grad_input_channel = grad_input + (n * in_channels + c_in) * in_h * in_w;

            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t offset_idx_base = og * 2 * kernel_h * kernel_w;
                    int64_t offset_h_idx = offset_idx_base + 2 * (kh * kernel_w + kw);
                    int64_t offset_w_idx = offset_h_idx + 1;

                    double off_h = offset[((n * offset_groups * 2 * kernel_h * kernel_w + offset_h_idx) * out_h + oh) * out_w + ow];
                    double off_w = offset[((n * offset_groups * 2 * kernel_h * kernel_w + offset_w_idx) * out_h + oh) * out_w + ow];

                    double h_in = static_cast<double>(oh * stride_h - pad_h + kh * dil_h) + off_h;
                    double w_in = static_cast<double>(ow * stride_w - pad_w + kw * dil_w) + off_w;

                    int64_t w_weight_idx = ((oc * in_channels_per_group + ic) * kernel_h + kh) * kernel_w + kw;
                    double w_val = weight[w_weight_idx];

                    double sampled_val = deformable_bilinear_sample(input_channel, in_h, in_w, h_in, w_in);

                    double m_val = 1.0;
                    if (has_mask) {
                        int64_t mask_idx_base = og * kernel_h * kernel_w;
                        int64_t mask_idx = mask_idx_base + kh * kernel_w + kw;
                        m_val = mask[((n * offset_groups * kernel_h * kernel_w + mask_idx) * out_h + oh) * out_w + ow];
                    }

                    // grad_input
                    double grad_to_scatter = grad_out_val * w_val * m_val;
                    deformable_bilinear_scatter(grad_input_channel, in_h, in_w, h_in, w_in, grad_to_scatter);

                    // grad_offset
                    double dval_dh, dval_dw;
                    deformable_bilinear_sample_grad_hw(input_channel, in_h, in_w, h_in, w_in, dval_dh, dval_dw);

                    int64_t off_h_flat = ((n * offset_groups * 2 * kernel_h * kernel_w + offset_h_idx) * out_h + oh) * out_w + ow;
                    int64_t off_w_flat = ((n * offset_groups * 2 * kernel_h * kernel_w + offset_w_idx) * out_h + oh) * out_w + ow;

                    atomicAdd_double(&grad_offset[off_h_flat], grad_out_val * w_val * m_val * dval_dh);
                    atomicAdd_double(&grad_offset[off_w_flat], grad_out_val * w_val * m_val * dval_dw);

                    // grad_mask
                    if (has_mask) {
                        int64_t mask_idx_base = og * kernel_h * kernel_w;
                        int64_t mask_idx = mask_idx_base + kh * kernel_w + kw;
                        int64_t mask_flat = ((n * offset_groups * kernel_h * kernel_w + mask_idx) * out_h + oh) * out_w + ow;
                        atomicAdd_double(&grad_mask[mask_flat], grad_out_val * w_val * sampled_val);
                    }
                }
            }
        }
    }
}

// ============================================================================
// DCNv2 Backward Weight Kernel
// ============================================================================

template<typename T>
__global__ void deformable_conv2d_backward_weight_impl(
    const T* grad_output,    // (N, C_out, H_out, W_out)
    const T* input,          // (N, C_in, H_in, W_in)
    const T* offset,         // (N, offset_groups * 2 * kH * kW, H_out, W_out)
    const T* mask,           // (N, offset_groups * kH * kW, H_out, W_out) or nullptr
    T* grad_weight,          // (C_out, C_in/groups, kH, kW)
    int64_t batch, int64_t in_channels, int64_t in_h, int64_t in_w,
    int64_t out_channels, int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    bool has_mask)
{
    // Each thread accumulates one element of grad_weight
    int64_t in_channels_per_group = in_channels / groups;
    int64_t out_channels_per_group = out_channels / groups;
    int64_t channels_per_offset_group = in_channels / offset_groups;
    int64_t total_weight = out_channels * in_channels_per_group * kernel_h * kernel_w;

    TENZOR_CUDA_KERNEL_LOOP(w_idx, total_weight) {
        int64_t kw = w_idx % kernel_w;
        int64_t kh = (w_idx / kernel_w) % kernel_h;
        int64_t ic = (w_idx / (kernel_w * kernel_h)) % in_channels_per_group;
        int64_t oc = w_idx / (kernel_w * kernel_h * in_channels_per_group);

        int64_t g = oc / out_channels_per_group;
        int64_t c_in = g * in_channels_per_group + ic;
        int64_t og = c_in / channels_per_offset_group;

        T grad_w_acc = static_cast<T>(0);

        for (int64_t n = 0; n < batch; ++n) {
            const T* input_channel = input + (n * in_channels + c_in) * in_h * in_w;

            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    int64_t offset_idx_base = og * 2 * kernel_h * kernel_w;
                    int64_t offset_h_idx = offset_idx_base + 2 * (kh * kernel_w + kw);
                    int64_t offset_w_idx = offset_h_idx + 1;

                    T off_h = offset[((n * offset_groups * 2 * kernel_h * kernel_w + offset_h_idx) * out_h + oh) * out_w + ow];
                    T off_w = offset[((n * offset_groups * 2 * kernel_h * kernel_w + offset_w_idx) * out_h + oh) * out_w + ow];

                    T h_in = static_cast<T>(oh * stride_h - pad_h + kh * dil_h) + off_h;
                    T w_in = static_cast<T>(ow * stride_w - pad_w + kw * dil_w) + off_w;

                    T val = deformable_bilinear_sample(input_channel, in_h, in_w, h_in, w_in);

                    if (has_mask) {
                        int64_t mask_idx_base = og * kernel_h * kernel_w;
                        int64_t mask_idx = mask_idx_base + kh * kernel_w + kw;
                        T m = mask[((n * offset_groups * kernel_h * kernel_w + mask_idx) * out_h + oh) * out_w + ow];
                        val *= m;
                    }

                    int64_t go_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
                    grad_w_acc += grad_output[go_idx] * val;
                }
            }
        }

        atomicAdd(&grad_weight[w_idx], grad_w_acc);
    }
}

// Double specialization for backward weight
template<>
__global__ void deformable_conv2d_backward_weight_impl<double>(
    const double* grad_output,
    const double* input,
    const double* offset,
    const double* mask,
    double* grad_weight,
    int64_t batch, int64_t in_channels, int64_t in_h, int64_t in_w,
    int64_t out_channels, int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    bool has_mask)
{
    int64_t in_channels_per_group = in_channels / groups;
    int64_t out_channels_per_group = out_channels / groups;
    int64_t channels_per_offset_group = in_channels / offset_groups;
    int64_t total_weight = out_channels * in_channels_per_group * kernel_h * kernel_w;

    TENZOR_CUDA_KERNEL_LOOP(w_idx, total_weight) {
        int64_t kw = w_idx % kernel_w;
        int64_t kh = (w_idx / kernel_w) % kernel_h;
        int64_t ic = (w_idx / (kernel_w * kernel_h)) % in_channels_per_group;
        int64_t oc = w_idx / (kernel_w * kernel_h * in_channels_per_group);

        int64_t g = oc / out_channels_per_group;
        int64_t c_in = g * in_channels_per_group + ic;
        int64_t og = c_in / channels_per_offset_group;

        double grad_w_acc = 0.0;

        for (int64_t n = 0; n < batch; ++n) {
            const double* input_channel = input + (n * in_channels + c_in) * in_h * in_w;

            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    int64_t offset_idx_base = og * 2 * kernel_h * kernel_w;
                    int64_t offset_h_idx = offset_idx_base + 2 * (kh * kernel_w + kw);
                    int64_t offset_w_idx = offset_h_idx + 1;

                    double off_h = offset[((n * offset_groups * 2 * kernel_h * kernel_w + offset_h_idx) * out_h + oh) * out_w + ow];
                    double off_w = offset[((n * offset_groups * 2 * kernel_h * kernel_w + offset_w_idx) * out_h + oh) * out_w + ow];

                    double h_in = static_cast<double>(oh * stride_h - pad_h + kh * dil_h) + off_h;
                    double w_in = static_cast<double>(ow * stride_w - pad_w + kw * dil_w) + off_w;

                    double val = deformable_bilinear_sample(input_channel, in_h, in_w, h_in, w_in);

                    if (has_mask) {
                        int64_t mask_idx_base = og * kernel_h * kernel_w;
                        int64_t mask_idx = mask_idx_base + kh * kernel_w + kw;
                        double m = mask[((n * offset_groups * kernel_h * kernel_w + mask_idx) * out_h + oh) * out_w + ow];
                        val *= m;
                    }

                    int64_t go_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
                    grad_w_acc += grad_output[go_idx] * val;
                }
            }
        }

        atomicAdd_double(&grad_weight[w_idx], grad_w_acc);
    }
}

// ============================================================================
// DCNv2 Host Launcher Functions
// ============================================================================

auto deformable_conv2d_forward_kernel(
    const Tensor& input,       // (N, C_in, H_in, W_in)
    const Tensor& offset,      // (N, offset_groups*2*kH*kW, H_out, W_out)
    const Tensor& weight,      // (C_out, C_in/groups, kH, kW)
    const Tensor& bias,        // (C_out,)
    const Tensor& mask,        // (N, offset_groups*kH*kW, H_out, W_out)
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    cudaStream_t stream) -> Tensor
{
    // Float16 / BFloat16 don't have dedicated CUDA template specializations.
    // Promote to Float32, compute, narrow back. Matches the CPU kernel's
    // strategy and the flex_attention pattern.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto out = deformable_conv2d_forward_kernel(
            input.to(DType::Float32),
            offset.to(DType::Float32),
            weight.to(DType::Float32),
            bias.numel() > 0 ? bias.to(DType::Float32) : bias,
            mask.numel() > 0 ? mask.to(DType::Float32) : mask,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, stream);
        return out.to(orig);
    }

    auto in_shape = input.shape();
    auto w_shape = weight.shape();

    int64_t batch = in_shape[0];
    int64_t in_channels = in_shape[1];
    int64_t in_h = in_shape[2];
    int64_t in_w = in_shape[3];

    int64_t out_channels = w_shape[0];
    int64_t kernel_h = w_shape[2];
    int64_t kernel_w = w_shape[3];

    int64_t out_h = (in_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_w = (in_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;

    Tensor output({batch, out_channels, out_h, out_w}, input.dtype(), input.device());

    bool has_bias = (bias.numel() > 0);
    bool has_mask = (mask.numel() > 0);

    int64_t total = batch * out_channels * out_h * out_w;

    if (input.dtype() == DType::Float32) {
        auto [num_blocks, block_size] = optimal_launch_config(
            deformable_conv2d_forward_impl<float>, total);
        deformable_conv2d_forward_impl<float><<<num_blocks, block_size, 0, stream>>>(
            input.data<float>(), offset.data<float>(), weight.data<float>(),
            has_bias ? bias.data<float>() : nullptr,
            has_mask ? mask.data<float>() : nullptr,
            output.data<float>(),
            batch, in_channels, in_h, in_w,
            out_channels, out_h, out_w,
            kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, has_bias, has_mask);
    } else if (input.dtype() == DType::Float64) {
        auto [num_blocks, block_size] = optimal_launch_config(
            deformable_conv2d_forward_impl<double>, total);
        deformable_conv2d_forward_impl<double><<<num_blocks, block_size, 0, stream>>>(
            input.data<double>(), offset.data<double>(), weight.data<double>(),
            has_bias ? bias.data<double>() : nullptr,
            has_mask ? mask.data<double>() : nullptr,
            output.data<double>(),
            batch, in_channels, in_h, in_w,
            out_channels, out_h, out_w,
            kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, has_bias, has_mask);
    } else {
        throw std::runtime_error("deformable_conv2d_forward: unsupported dtype (requires Float32 or Float64)");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return output;
}

auto deformable_conv2d_backward_input_kernel(
    const Tensor& grad_output,   // (N, C_out, H_out, W_out)
    const Tensor& input,         // (N, C_in, H_in, W_in)
    const Tensor& offset,        // (N, offset_groups*2*kH*kW, H_out, W_out)
    const Tensor& weight,        // (C_out, C_in/groups, kH, kW)
    const Tensor& mask,          // (N, offset_groups*kH*kW, H_out, W_out)
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    cudaStream_t stream) -> std::vector<Tensor>
{
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto outs = deformable_conv2d_backward_input_kernel(
            grad_output.to(DType::Float32),
            input.to(DType::Float32),
            offset.to(DType::Float32),
            weight.to(DType::Float32),
            mask.numel() > 0 ? mask.to(DType::Float32) : mask,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, offset_groups,
            stream);
        std::vector<Tensor> narrowed;
        narrowed.reserve(outs.size());
        for (auto& t : outs) narrowed.push_back(t.numel() > 0 ? t.to(orig) : t);
        return narrowed;
    }

    auto in_shape = input.shape();
    auto off_shape = offset.shape();
    auto w_shape = weight.shape();
    auto go_shape = grad_output.shape();

    int64_t batch = in_shape[0];
    int64_t in_channels = in_shape[1];
    int64_t in_h = in_shape[2];
    int64_t in_w = in_shape[3];
    int64_t out_channels = w_shape[0];
    int64_t out_h = go_shape[2];
    int64_t out_w = go_shape[3];
    int64_t kernel_h = w_shape[2];
    int64_t kernel_w = w_shape[3];

    bool has_mask = (mask.numel() > 0);

    Tensor grad_input(std::vector<int64_t>(in_shape.begin(), in_shape.end()), input.dtype(), input.device());
    Tensor grad_offset(std::vector<int64_t>(off_shape.begin(), off_shape.end()), offset.dtype(), offset.device());
    Tensor grad_mask_tensor;
    if (has_mask) {
        auto ms = mask.shape();
        grad_mask_tensor = Tensor(std::vector<int64_t>(ms.begin(), ms.end()), mask.dtype(), mask.device());
    }

    int64_t total = batch * out_channels * out_h * out_w;

    if (input.dtype() == DType::Float32) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(grad_input.data<float>(), 0, grad_input.numel() * sizeof(float), stream));
        TENZOR_CUDA_CHECK(cudaMemsetAsync(grad_offset.data<float>(), 0, grad_offset.numel() * sizeof(float), stream));
        if (has_mask) {
            TENZOR_CUDA_CHECK(cudaMemsetAsync(grad_mask_tensor.data<float>(), 0, grad_mask_tensor.numel() * sizeof(float), stream));
        }

        auto [num_blocks, block_size] = optimal_launch_config(
            deformable_conv2d_backward_input_impl<float>, total);
        deformable_conv2d_backward_input_impl<float><<<num_blocks, block_size, 0, stream>>>(
            grad_output.data<float>(), input.data<float>(), offset.data<float>(),
            weight.data<float>(),
            has_mask ? mask.data<float>() : nullptr,
            grad_input.data<float>(), grad_offset.data<float>(),
            has_mask ? grad_mask_tensor.data<float>() : nullptr,
            batch, in_channels, in_h, in_w,
            out_channels, out_h, out_w,
            kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, has_mask);
    } else if (input.dtype() == DType::Float64) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(grad_input.data<double>(), 0, grad_input.numel() * sizeof(double), stream));
        TENZOR_CUDA_CHECK(cudaMemsetAsync(grad_offset.data<double>(), 0, grad_offset.numel() * sizeof(double), stream));
        if (has_mask) {
            TENZOR_CUDA_CHECK(cudaMemsetAsync(grad_mask_tensor.data<double>(), 0, grad_mask_tensor.numel() * sizeof(double), stream));
        }

        auto [num_blocks, block_size] = optimal_launch_config(
            deformable_conv2d_backward_input_impl<double>, total);
        deformable_conv2d_backward_input_impl<double><<<num_blocks, block_size, 0, stream>>>(
            grad_output.data<double>(), input.data<double>(), offset.data<double>(),
            weight.data<double>(),
            has_mask ? mask.data<double>() : nullptr,
            grad_input.data<double>(), grad_offset.data<double>(),
            has_mask ? grad_mask_tensor.data<double>() : nullptr,
            batch, in_channels, in_h, in_w,
            out_channels, out_h, out_w,
            kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, has_mask);
    } else {
        throw std::runtime_error("deformable_conv2d_backward_input: unsupported dtype (requires Float32 or Float64)");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();

    if (has_mask) {
        return {grad_input, grad_offset, grad_mask_tensor};
    }
    return {grad_input, grad_offset};
}

auto deformable_conv2d_backward_weight_kernel(
    const Tensor& grad_output,   // (N, C_out, H_out, W_out)
    const Tensor& input,         // (N, C_in, H_in, W_in)
    const Tensor& offset,        // (N, offset_groups*2*kH*kW, H_out, W_out)
    const Tensor& mask,          // (N, offset_groups*kH*kW, H_out, W_out)
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    const std::vector<int64_t>& weight_shape,
    cudaStream_t stream) -> Tensor
{
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto out = deformable_conv2d_backward_weight_kernel(
            grad_output.to(DType::Float32),
            input.to(DType::Float32),
            offset.to(DType::Float32),
            mask.numel() > 0 ? mask.to(DType::Float32) : mask,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, offset_groups,
            weight_shape, stream);
        return out.to(orig);
    }

    auto in_shape = input.shape();
    auto go_shape = grad_output.shape();

    int64_t batch = in_shape[0];
    int64_t in_channels = in_shape[1];
    int64_t in_h = in_shape[2];
    int64_t in_w = in_shape[3];
    int64_t out_channels = weight_shape[0];
    int64_t in_channels_per_group = weight_shape[1];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];
    int64_t out_h = go_shape[2];
    int64_t out_w = go_shape[3];

    bool has_mask = (mask.numel() > 0);

    Tensor grad_weight(weight_shape, input.dtype(), input.device());

    int64_t total_weight = out_channels * in_channels_per_group * kernel_h * kernel_w;

    if (input.dtype() == DType::Float32) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(grad_weight.data<float>(), 0, grad_weight.numel() * sizeof(float), stream));

        auto [num_blocks, block_size] = optimal_launch_config(
            deformable_conv2d_backward_weight_impl<float>, total_weight);
        deformable_conv2d_backward_weight_impl<float><<<num_blocks, block_size, 0, stream>>>(
            grad_output.data<float>(), input.data<float>(), offset.data<float>(),
            has_mask ? mask.data<float>() : nullptr,
            grad_weight.data<float>(),
            batch, in_channels, in_h, in_w,
            out_channels, out_h, out_w,
            kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, has_mask);
    } else if (input.dtype() == DType::Float64) {
        TENZOR_CUDA_CHECK(cudaMemsetAsync(grad_weight.data<double>(), 0, grad_weight.numel() * sizeof(double), stream));

        auto [num_blocks, block_size] = optimal_launch_config(
            deformable_conv2d_backward_weight_impl<double>, total_weight);
        deformable_conv2d_backward_weight_impl<double><<<num_blocks, block_size, 0, stream>>>(
            grad_output.data<double>(), input.data<double>(), offset.data<double>(),
            has_mask ? mask.data<double>() : nullptr,
            grad_weight.data<double>(),
            batch, in_channels, in_h, in_w,
            out_channels, out_h, out_w,
            kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, has_mask);
    } else {
        throw std::runtime_error("deformable_conv2d_backward_weight: unsupported dtype (requires Float32 or Float64)");
    }

    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return grad_weight;
}

// ============================================================================
// Depthwise Conv1d / Conv3d (forward, groups == channels).
//   Conv1d contract: input [N,C,1,L], weight [C,1,1,kL], output [N,C,1,L_out].
//   Conv3d contract: input [N,C,D,H,W], weight [C,1,kD,kH,kW], output [N,C,Do,Ho,Wo].
// Float32/Float64 native; Float16/BFloat16 widen to Float32. Backward is
// autograd-composed in the NN layer (no dedicated *Backward OpId).
// ============================================================================
template <typename T>
__global__ void depthwise_conv1d_fwd_kernel(
    const T* __restrict__ in, const T* __restrict__ w, const T* __restrict__ bias,
    T* __restrict__ out, int N, int C, int L, int kL, int Lo,
    int stride, int pad, int dil) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Lo;
    if (idx >= total) return;
    int ol = idx % Lo;
    int c  = (idx / Lo) % C;
    int n  = idx / (C * Lo);
    const T* in_nc = in + (n * C + c) * L;
    const T* w_c   = w + c * kL;
    T acc = bias ? bias[c] : T(0);
    for (int k = 0; k < kL; ++k) {
        int il = ol * stride - pad + k * dil;
        if (il >= 0 && il < L) acc += in_nc[il] * w_c[k];
    }
    out[(n * C + c) * Lo + ol] = acc;
}

template <typename T>
__global__ void depthwise_conv3d_fwd_kernel(
    const T* __restrict__ in, const T* __restrict__ w, const T* __restrict__ bias,
    T* __restrict__ out, int N, int C, int Di, int Hi, int Wi,
    int kD, int kH, int kW, int Do, int Ho, int Wo,
    int sD, int sH, int sW, int pD, int pH, int pW, int dD, int dH, int dW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Do * Ho * Wo;
    if (idx >= total) return;
    int ow = idx % Wo;
    int oh = (idx / Wo) % Ho;
    int od = (idx / (Wo * Ho)) % Do;
    int c  = (idx / (Wo * Ho * Do)) % C;
    int n  = idx / (C * Do * Ho * Wo);
    const T* in_nc = in + (n * C + c) * Di * Hi * Wi;
    const T* w_c   = w + c * kD * kH * kW;
    T acc = bias ? bias[c] : T(0);
    for (int kd = 0; kd < kD; ++kd) {
        int id = od * sD - pD + kd * dD;
        if (id < 0 || id >= Di) continue;
        for (int kh = 0; kh < kH; ++kh) {
            int ih = oh * sH - pH + kh * dH;
            if (ih < 0 || ih >= Hi) continue;
            for (int kw = 0; kw < kW; ++kw) {
                int iw = ow * sW - pW + kw * dW;
                if (iw < 0 || iw >= Wi) continue;
                acc += in_nc[(id * Hi + ih) * Wi + iw] * w_c[(kd * kH + kh) * kW + kw];
            }
        }
    }
    out[(((n * C + c) * Do + od) * Ho + oh) * Wo + ow] = acc;
}

auto depthwise_conv1d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                             int64_t stride, int64_t padding, int64_t dilation,
                             cudaStream_t stream) -> Tensor {
    auto is = input.shape();   // [N,C,1,L]
    auto ws = weight.shape();  // [C,1,1,kL]
    int N = (int)is[0], C = (int)is[1], L = (int)is[3], kL = (int)ws[3];
    int Lo = (int)((L + 2 * padding - dilation * (kL - 1) - 1) / stride + 1);
    if (Lo <= 0) throw std::runtime_error("depthwise_conv1d (CUDA): non-positive output length");

    auto run = [&](DType dt) {
        Tensor in = input.dtype() == dt ? input.contiguous() : input.to(dt);
        Tensor w  = weight.dtype() == dt ? weight.contiguous() : weight.to(dt);
        Tensor b; const void* bptr = nullptr;
        if (bias) { b = bias->dtype() == dt ? bias->contiguous() : bias->to(dt); bptr = b.data_ptr(); }
        Tensor out({(int64_t)N, (int64_t)C, 1, (int64_t)Lo}, dt, input.device());
        int total = N * C * Lo;
        int blocks = (total + 255) / 256;
        if (dt == DType::Float64) {
            depthwise_conv1d_fwd_kernel<double><<<blocks, 256, 0, stream>>>(
                in.data<double>(), w.data<double>(), (const double*)bptr, out.data<double>(),
                N, C, L, kL, Lo, (int)stride, (int)padding, (int)dilation);
        } else {
            depthwise_conv1d_fwd_kernel<float><<<blocks, 256, 0, stream>>>(
                in.data<float>(), w.data<float>(), (const float*)bptr, out.data<float>(),
                N, C, L, kL, Lo, (int)stride, (int)padding, (int)dilation);
        }
        TENZOR_CUDA_POST_LAUNCH_CHECK();
        return out;
    };

    DType in_dt = input.dtype();
    if (in_dt == DType::Float64) return run(DType::Float64);
    if (in_dt == DType::Float32) return run(DType::Float32);
    if (in_dt == DType::Float16 || in_dt == DType::BFloat16) return run(DType::Float32).to(in_dt);
    throw std::runtime_error("depthwise_conv1d (CUDA): unsupported dtype");
}

auto depthwise_conv3d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                             int64_t sD, int64_t sH, int64_t sW,
                             int64_t pD, int64_t pH, int64_t pW,
                             int64_t dD, int64_t dH, int64_t dW,
                             cudaStream_t stream) -> Tensor {
    auto is = input.shape();   // [N,C,D,H,W]
    auto ws = weight.shape();  // [C,1,kD,kH,kW]
    int N = (int)is[0], C = (int)is[1], Di = (int)is[2], Hi = (int)is[3], Wi = (int)is[4];
    int kD = (int)ws[2], kH = (int)ws[3], kW = (int)ws[4];
    int Do = (int)((Di + 2 * pD - dD * (kD - 1) - 1) / sD + 1);
    int Ho = (int)((Hi + 2 * pH - dH * (kH - 1) - 1) / sH + 1);
    int Wo = (int)((Wi + 2 * pW - dW * (kW - 1) - 1) / sW + 1);
    if (Do <= 0 || Ho <= 0 || Wo <= 0) throw std::runtime_error("depthwise_conv3d (CUDA): non-positive output size");

    auto run = [&](DType dt) {
        Tensor in = input.dtype() == dt ? input.contiguous() : input.to(dt);
        Tensor w  = weight.dtype() == dt ? weight.contiguous() : weight.to(dt);
        Tensor b; const void* bptr = nullptr;
        if (bias) { b = bias->dtype() == dt ? bias->contiguous() : bias->to(dt); bptr = b.data_ptr(); }
        Tensor out({(int64_t)N, (int64_t)C, (int64_t)Do, (int64_t)Ho, (int64_t)Wo}, dt, input.device());
        int total = N * C * Do * Ho * Wo;
        int blocks = (total + 255) / 256;
        if (dt == DType::Float64) {
            depthwise_conv3d_fwd_kernel<double><<<blocks, 256, 0, stream>>>(
                in.data<double>(), w.data<double>(), (const double*)bptr, out.data<double>(),
                N, C, Di, Hi, Wi, kD, kH, kW, Do, Ho, Wo,
                (int)sD,(int)sH,(int)sW,(int)pD,(int)pH,(int)pW,(int)dD,(int)dH,(int)dW);
        } else {
            depthwise_conv3d_fwd_kernel<float><<<blocks, 256, 0, stream>>>(
                in.data<float>(), w.data<float>(), (const float*)bptr, out.data<float>(),
                N, C, Di, Hi, Wi, kD, kH, kW, Do, Ho, Wo,
                (int)sD,(int)sH,(int)sW,(int)pD,(int)pH,(int)pW,(int)dD,(int)dH,(int)dW);
        }
        TENZOR_CUDA_POST_LAUNCH_CHECK();
        return out;
    };

    DType in_dt = input.dtype();
    if (in_dt == DType::Float64) return run(DType::Float64);
    if (in_dt == DType::Float32) return run(DType::Float32);
    if (in_dt == DType::Float16 || in_dt == DType::BFloat16) return run(DType::Float32).to(in_dt);
    throw std::runtime_error("depthwise_conv3d (CUDA): unsupported dtype");
}

} // namespace cuda
} // namespace tenzor
