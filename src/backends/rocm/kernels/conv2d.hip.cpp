#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/loader_fwd.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#ifdef USE_MIOPEN
#include <miopen/miopen.h>
#endif
#include <stdexcept>
#include <vector>
#include <iostream>
#include <type_traits>
#include "fp16_saturate.h"
#include "../rocm_error.hpp"
#include "../miopen_guards.hpp"
#include "../hip_buffer.hpp"
#include "../rocm_arch_detect.hpp"

namespace tenzor {
namespace rocm {

#ifdef USE_MIOPEN
#define MIOPEN_CHECK(call) do { \
    miopenStatus_t status = call; \
    if (status != miopenStatusSuccess) { \
        throw std::runtime_error(std::string("MIOpen error: ") + std::to_string(status)); \
    } \
} while(0)
#endif

// Thread-local cached rocBLAS handle for the conv2d forward/backward paths.
// rocblas_create_handle is expensive (device workspace alloc + property
// queries) and is meant to be reused. Creating/destroying one per conv call —
// i.e. every conv layer, every training step — was pure overhead, so we keep a
// single handle alive per thread and only rebind the stream per invocation.
namespace {
struct CachedConvHandle {
    rocblas_handle handle = nullptr;
    ~CachedConvHandle() {
        // Guard teardown: destroying after the backend library has unloaded
        // calls into freed code (mirrors the rocSPARSE/rocSOLVER/quant pools).
        if (handle && tenzor::is_backend_registry_alive()) {
            rocblas_destroy_handle(handle);
            handle = nullptr;
        }
    }
};
inline rocblas_handle get_cached_conv_handle(hipStream_t stream) {
    thread_local CachedConvHandle cached;
    if (cached.handle == nullptr) {
        ROCBLAS_CHECK(rocblas_create_handle(&cached.handle));
    }
    ROCBLAS_CHECK(rocblas_set_stream(cached.handle, stream));
    return cached.handle;
}
}  // namespace

// ============================================================================
// Kernel Launch Helpers - Optimized for AMD GPUs
// ============================================================================

// Use 4 wavefronts per block (optimal for most kernels).
// Wavefront size is queried from the device (64 for CDNA/GCN, 32 for RDNA3+).
inline void compute_launch_config_1d(int64_t n, dim3& grid, dim3& block) {
    const int block_size = rocm::get_wavefront_size() * 4;
    block = dim3(block_size, 1, 1);
    grid = dim3((n + block_size - 1) / block_size, 1, 1);
}

inline void compute_launch_config_2d(int64_t rows, int64_t cols, dim3& grid, dim3& block) {
    // Use 16x16 = 256 threads per block (optimal for AMD)
    const int block_x = 16;
    const int block_y = 16;
    block = dim3(block_x, block_y, 1);
    grid = dim3((cols + block_x - 1) / block_x, (rows + block_y - 1) / block_y, 1);
}

#define HIP_KERNEL_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

// ============================================================================
// Helper Functions
// ============================================================================

// Calculate output size for convolution
__host__ __device__ inline int64_t calculate_output_size(int64_t input_size, int64_t kernel_size,
                                                          int64_t stride, int64_t padding, int64_t dilation) {
    #ifndef __HIP_DEVICE_COMPILE__
    // Host-side validation (not in device code)
    if (stride == 0) {
        throw std::invalid_argument("Conv2d: stride cannot be zero");
    }
    #endif
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
}

// ============================================================================
// Data Layout Support
// ============================================================================

enum class DataLayout {
    NCHW,  // Batch, Channels, Height, Width (default)
    NHWC   // Batch, Height, Width, Channels (TensorFlow style)
};

// ============================================================================
// NHWC → NCHW transpose for conv2d output.
//
// rocBLAS (column-major) given a row-major (M×K) col_buffer with ld=K and
// a row-major (N×K) weight_ptr with ld=K, where we ask op(A)=trans(weight)
// and op(B)=col, computes op(A) · op(B) = (N,K) · (K,M) = (N×M). Written
// back with ld=N, the resulting memory layout is (M,N) row-major — i.e.
// output[b*OH*OW + oh*OW + ow, oc] — which flattens as
// [b, oh, ow, oc] (NHWC), not [b, oc, oh, ow] (NCHW). Conv2d's public
// contract is NCHW, so we GEMM into a temp buffer and permute with
// this kernel.
// ============================================================================
template<typename T>
__global__ void conv_nhwc_to_nchw_kernel(
    const T* __restrict__ nhwc,   // shape: (batch * out_h * out_w, channels_per_group)
    T* __restrict__ nchw_out,     // shape: (batch, out_channels, out_h, out_w)
    int64_t batch,
    int64_t out_h,
    int64_t out_w,
    int64_t out_channels,         // total output channels (across all groups)
    int64_t channels_per_group,
    int64_t channel_offset        // group's first channel in the global output
) {
    int64_t total_spatial = batch * out_h * out_w;
    HIP_KERNEL_LOOP(idx, total_spatial * channels_per_group) {
        int64_t c = idx % channels_per_group;
        int64_t spatial_idx = idx / channels_per_group;
        int64_t b = spatial_idx / (out_h * out_w);
        int64_t hw = spatial_idx % (out_h * out_w);
        int64_t h = hw / out_w;
        int64_t w = hw % out_w;
        int64_t global_c = channel_offset + c;
        int64_t nchw_idx = ((b * out_channels + global_c) * out_h + h) * out_w + w;
        nchw_out[nchw_idx] = nhwc[idx];
    }
}

// ============================================================================
// NCHW → NHWC permute for conv2d backward.
//
// The exact inverse of conv_nhwc_to_nchw_kernel. The backward GEMMs feed
// grad_output to rocBLAS as op(B) read column-major [K×M] with ld=K, i.e.
// rocBLAS sees element (k,m) at grad_out[m*K + k] = row-major [col_row][oc]
// = [b][oh][ow][oc] (NHWC). But grad_output's real memory is NCHW
// [b][oc][oh][ow]. This kernel gathers the per-group NCHW grad_output into
// a contiguous NHWC temp buffer (batch * out_h * out_w, channels_per_group)
// so the backward consumes exactly the layout the forward produced.
// ============================================================================
template<typename T>
__global__ void conv_nchw_to_nhwc_kernel(
    const T* __restrict__ nchw,   // shape: (batch, out_channels, out_h, out_w)
    T* __restrict__ nhwc_out,     // shape: (batch * out_h * out_w, channels_per_group)
    int64_t batch,
    int64_t out_h,
    int64_t out_w,
    int64_t out_channels,         // total output channels (across all groups)
    int64_t channels_per_group,
    int64_t channel_offset        // group's first channel in the global output
) {
    int64_t total_spatial = batch * out_h * out_w;
    HIP_KERNEL_LOOP(idx, total_spatial * channels_per_group) {
        int64_t c = idx % channels_per_group;
        int64_t spatial_idx = idx / channels_per_group;
        int64_t b = spatial_idx / (out_h * out_w);
        int64_t hw = spatial_idx % (out_h * out_w);
        int64_t h = hw / out_w;
        int64_t w = hw % out_w;
        int64_t global_c = channel_offset + c;
        int64_t nchw_idx = ((b * out_channels + global_c) * out_h + h) * out_w + w;
        nhwc_out[idx] = nchw[nchw_idx];
    }
}

// ============================================================================
// im2col HIP Kernel - Optimized for AMD GPUs
// ============================================================================

// im2col kernel: Convert 4D input (N,C,H,W) to 2D matrix for convolution
// Input: (batch, in_channels, height, width) - NCHW layout
// Output: (batch * out_h * out_w, kernel_h * kernel_w * in_channels)
//
// AMD GPU OPTIMIZATIONS:
// - Uses grid-stride loop for better work distribution
// - Memory access patterns optimized for global memory coalescing
// - Wavefront-aware indexing to minimize bank conflicts
template<typename T>
__global__ void im2col_kernel_nchw(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t out_h,
    int64_t out_w
) {
    int64_t total_elements = batch * out_h * out_w * channels * kernel_h * kernel_w;

    HIP_KERNEL_LOOP(idx, total_elements) {
        // Decode flat index to (b, oh, ow, c, kh, kw)
        int64_t temp = idx;
        int64_t kw = temp % kernel_w; temp /= kernel_w;
        int64_t kh = temp % kernel_h; temp /= kernel_h;
        int64_t c = temp % channels; temp /= channels;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t b = temp;

        // Calculate input position with per-axis padding and dilation
        int64_t ih = oh * stride_h - pad_h + kh * dil_h;
        int64_t iw = ow * stride_w - pad_w + kw * dil_w;

        // Output index in col matrix
        // Shape: (batch * out_h * out_w, channels * kernel_h * kernel_w)
        int64_t out_row = b * out_h * out_w + oh * out_w + ow;
        int64_t out_col = c * kernel_h * kernel_w + kh * kernel_w + kw;
        int64_t out_idx = out_row * (channels * kernel_h * kernel_w) + out_col;

        // Check bounds and apply padding (use ternary for better instruction scheduling)
        T value = (ih >= 0 && ih < height && iw >= 0 && iw < width)
                  ? input[b * (channels * height * width) + c * (height * width) + ih * width + iw]
                  : T(0);
        output[out_idx] = value;
    }
}

// im2col kernel for NHWC layout (optimized for TensorFlow-style tensors)
template<typename T>
__global__ void im2col_kernel_nhwc(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t batch,
    int64_t height,
    int64_t width,
    int64_t channels,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t out_h,
    int64_t out_w
) {
    int64_t total_elements = batch * out_h * out_w * channels * kernel_h * kernel_w;

    HIP_KERNEL_LOOP(idx, total_elements) {
        // Decode flat index to (b, oh, ow, c, kh, kw)
        int64_t temp = idx;
        int64_t kw = temp % kernel_w; temp /= kernel_w;
        int64_t kh = temp % kernel_h; temp /= kernel_h;
        int64_t c = temp % channels; temp /= channels;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t b = temp;

        // Calculate input position with per-axis padding and dilation
        int64_t ih = oh * stride_h - pad_h + kh * dil_h;
        int64_t iw = ow * stride_w - pad_w + kw * dil_w;

        // Output index in col matrix (same as NCHW)
        int64_t out_row = b * out_h * out_w + oh * out_w + ow;
        int64_t out_col = c * kernel_h * kernel_w + kh * kernel_w + kw;
        int64_t out_idx = out_row * (channels * kernel_h * kernel_w) + out_col;

        // NHWC input layout: (batch, height, width, channels)
        T value = (ih >= 0 && ih < height && iw >= 0 && iw < width)
                  ? input[b * (height * width * channels) + ih * (width * channels) + iw * channels + c]
                  : T(0);
        output[out_idx] = value;
    }
}

// ============================================================================
// col2im HIP Kernel - Output-Centric Approach (No Atomics!)
// ============================================================================

// col2im kernel: Reverse of im2col for gradient computation
// Input: (batch * out_h * out_w, kernel_h * kernel_w * in_channels)
// Output: (batch, in_channels, height, width) - NCHW layout
//
// CRITICAL OPTIMIZATION for AMD GPUs:
// Uses output-centric approach to completely eliminate atomic operations
// This is especially important on AMD GPUs where atomics can be slower than NVIDIA
//
// Performance considerations:
// - AMD GPUs have excellent global memory bandwidth (up to 2TB/s on MI250X)
// - Wavefront size of 64 means good parallelism for small kernels
// - No atomic contention = predictable performance
// - Extra work per thread (kernel_h * kernel_w iterations) is negligible
template<typename T>
__global__ void col2im_kernel_nchw(
    const T* __restrict__ col,
    T* __restrict__ output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t out_h,
    int64_t out_w
) {
    // Each thread processes one output element
    int64_t total_output = batch * channels * height * width;

    HIP_KERNEL_LOOP(output_idx, total_output) {
        // Decode output index to (b, c, ih, iw)
        int64_t temp = output_idx;
        int64_t iw = temp % width; temp /= width;
        int64_t ih = temp % height; temp /= height;
        int64_t c = temp % channels; temp /= channels;
        int64_t b = temp;

        // Accumulate from all kernel positions that contribute to this output
        T sum = T(0);

        // Accumulate from all kernel positions (per-axis stride/padding/dilation)
        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw_iter = 0; kw_iter < kernel_w; ++kw_iter) {
                // Reverse the im2col mapping: given output (ih, iw) and kernel (kh, kw), find col (oh, ow)
                int64_t ih_shifted = ih + pad_h - kh * dil_h;
                int64_t iw_shifted = iw + pad_w - kw_iter * dil_w;

                // Check if this maps to a valid col position
                if (ih_shifted % stride_h == 0 && iw_shifted % stride_w == 0) {
                    int64_t oh = ih_shifted / stride_h;
                    int64_t ow = iw_shifted / stride_w;

                    // Check bounds
                    if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                        // Calculate col buffer index
                        int64_t col_row = b * out_h * out_w + oh * out_w + ow;
                        int64_t col_col = c * kernel_h * kernel_w + kh * kernel_w + kw_iter;
                        int64_t col_idx = col_row * (channels * kernel_h * kernel_w) + col_col;

                        // Accumulate contribution
                        sum += col[col_idx];
                    }
                }
            }
        }

        // Direct write - NO ATOMIC NEEDED!
        output[output_idx] = sum;
    }
}

// col2im kernel for NHWC layout
template<typename T>
__global__ void col2im_kernel_nhwc(
    const T* __restrict__ col,
    T* __restrict__ output,
    int64_t batch,
    int64_t height,
    int64_t width,
    int64_t channels,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t out_h,
    int64_t out_w
) {
    int64_t total_output = batch * height * width * channels;

    HIP_KERNEL_LOOP(output_idx, total_output) {
        // Decode output index to (b, ih, iw, c) for NHWC
        int64_t temp = output_idx;
        int64_t c = temp % channels; temp /= channels;
        int64_t iw = temp % width; temp /= width;
        int64_t ih = temp % height; temp /= height;
        int64_t b = temp;

        T sum = T(0);

        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw_iter = 0; kw_iter < kernel_w; ++kw_iter) {
                int64_t ih_shifted = ih + pad_h - kh * dil_h;
                int64_t iw_shifted = iw + pad_w - kw_iter * dil_w;

                if (ih_shifted % stride_h == 0 && iw_shifted % stride_w == 0) {
                    int64_t oh = ih_shifted / stride_h;
                    int64_t ow = iw_shifted / stride_w;

                    if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                        int64_t col_row = b * out_h * out_w + oh * out_w + ow;
                        int64_t col_col = c * kernel_h * kernel_w + kh * kernel_w + kw_iter;
                        int64_t col_idx = col_row * (channels * kernel_h * kernel_w) + col_col;

                        sum += col[col_idx];
                    }
                }
            }
        }

        output[output_idx] = sum;
    }
}

// ============================================================================
// Bias Addition Kernel
// ============================================================================

__global__ void add_bias_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int64_t batch,
    int64_t channels,
    int64_t spatial_size,
    int64_t n
) {
    HIP_KERNEL_LOOP(idx, n) {
        int64_t c = (idx / spatial_size) % channels;
        output[idx] += bias[c];
    }
}

// Bias addition for NHWC layout
__global__ void add_bias_kernel_nhwc(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int64_t batch,
    int64_t height,
    int64_t width,
    int64_t channels,
    int64_t n
) {
    HIP_KERNEL_LOOP(idx, n) {
        int64_t c = idx % channels;
        output[idx] += bias[c];
    }
}

// ============================================================================
// Bias Gradient Kernel - Wavefront-Optimized
// ============================================================================


// Optimized version using wave-level reduction
template<typename T>
__global__ void sum_bias_grad_kernel_wave_reduce(
    const T* __restrict__ grad_output,
    T* __restrict__ grad_bias,
    int64_t batch,
    int64_t channels,
    int64_t spatial_size
) {
    // Accumulate in an accumulator type wide enough for T: double for Float64,
    // float otherwise. Hardcoding float would truncate every Float64 grad_output
    // value to single precision, degrading Float64 bias gradients to ~7 digits.
    using Acc = std::conditional_t<std::is_same_v<T, double>, double, float>;
    __shared__ Acc shared_data[256];

    int64_t c = blockIdx.x;
    if (c < channels) {
        int64_t tid = threadIdx.x;
        int64_t block_size = blockDim.x;

        // Each thread accumulates multiple values (accumulate in Acc for precision)
        Acc local_sum = Acc(0);
        for (int64_t idx = tid; idx < batch * spatial_size; idx += block_size) {
            int64_t b = idx / spatial_size;
            int64_t s = idx % spatial_size;
            int64_t grad_idx = b * (channels * spatial_size) + c * spatial_size + s;
            local_sum += static_cast<Acc>(grad_output[grad_idx]);
        }

        // Store in shared memory
        shared_data[tid] = local_sum;
        __syncthreads();

        // Reduce within block
        for (int64_t stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_data[tid] += shared_data[tid + stride];
            }
            __syncthreads();
        }

        // Write result (convert back to T from float accumulator)
        if (tid == 0) {
            grad_bias[c] = static_cast<T>(shared_data[0]);
        }
    }
}

// ============================================================================
// Conv2d Forward HIP Implementation
// ============================================================================

#ifdef USE_MIOPEN
// MIOpen-accelerated path for standard convolutions
auto conv2d_forward_miopen(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    hipStream_t stream
) -> Tensor {
    // Extract dimensions (assume NCHW layout)
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Calculate output dimensions
    int64_t out_h = calculate_output_size(height, kernel_h, stride, padding, dilation);
    int64_t out_w = calculate_output_size(width, kernel_w, stride, padding, dilation);

    // Create output tensor
    Tensor output({batch, out_channels, out_h, out_w}, input.dtype(), input.device());

    // Create MIOpen handle (RAII — automatically destroyed on scope exit)
    tenzor::rocm::MiopenHandleGuard miopen_guard;
    MIOPEN_CHECK(miopenSetStream(miopen_guard.handle, stream));

    // Create tensor descriptors (RAII)
    tenzor::rocm::MiopenTensorDescGuard input_desc_guard;
    tenzor::rocm::MiopenTensorDescGuard output_desc_guard;

    // Set input tensor descriptor (NCHW format)
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(
        input_desc_guard.desc,
        miopenFloat,  // data type
        batch, in_channels, height, width
    ));

    // Set output tensor descriptor
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(
        output_desc_guard.desc,
        miopenFloat,
        batch, out_channels, out_h, out_w
    ));

    // Create convolution descriptor (RAII)
    tenzor::rocm::MiopenConvDescGuard conv_desc_guard;

    // Initialize convolution descriptor with parameters
    MIOPEN_CHECK(miopenInitConvolutionDescriptor(
        conv_desc_guard.desc,
        miopenConvolution,  // mode
        padding, padding,   // pad_h, pad_w
        stride, stride,     // stride_h, stride_w
        dilation, dilation  // dilation_h, dilation_w
    ));

    // Set group count for grouped convolutions
    if (groups > 1) {
        MIOPEN_CHECK(miopenSetConvolutionGroupCount(conv_desc_guard.desc, groups));
    }

    // Create filter descriptor (RAII)
    tenzor::rocm::MiopenFilterDescGuard filter_desc_guard;
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(
        filter_desc_guard.desc,
        miopenFloat,
        out_channels, in_channels / groups, kernel_h, kernel_w
    ));

    // Find best algorithm for this convolution
    const int algo_request_count = 5;
    int algo_count = 0;
    miopenConvAlgoPerf_t perf_results[algo_request_count];

    // Query workspace size needed for algorithm search
    size_t workspace_size = 0;
    MIOPEN_CHECK(miopenConvolutionForwardGetWorkSpaceSize(
        miopen_guard.handle,
        filter_desc_guard.desc,
        input_desc_guard.desc,
        conv_desc_guard.desc,
        output_desc_guard.desc,
        &workspace_size
    ));

    // Allocate workspace (RAII)
    tenzor::rocm::HipBuffer workspace(workspace_size);

    // Find the best algorithm
    MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
        miopen_guard.handle,
        input_desc_guard.desc,
        input.data<float>(),
        filter_desc_guard.desc,
        weight.data<float>(),
        conv_desc_guard.desc,
        output_desc_guard.desc,
        output.data<float>(),
        algo_request_count,
        &algo_count,
        perf_results,
        workspace.ptr,
        workspace_size,
        false  // exhaustive search (false for heuristics, true for benchmarking)
    ));

    if (algo_count == 0) {
        throw std::runtime_error("MIOpen: No suitable convolution algorithm found");
    }

    // Use the fastest algorithm
    miopenConvFwdAlgorithm_t algo = perf_results[0].fwd_algo;

    // Execute forward convolution
    float alpha = 1.0f;
    float beta = 0.0f;

    MIOPEN_CHECK(miopenConvolutionForward(
        miopen_guard.handle,
        &alpha,
        input_desc_guard.desc,
        input.data<float>(),
        filter_desc_guard.desc,
        weight.data<float>(),
        conv_desc_guard.desc,
        algo,
        &beta,
        output_desc_guard.desc,
        output.data<float>(),
        workspace.ptr,
        workspace_size
    ));

    // Add bias if present
    if (bias != nullptr) {
        tenzor::rocm::MiopenTensorDescGuard bias_desc_guard;
        MIOPEN_CHECK(miopenSet4dTensorDescriptor(
            bias_desc_guard.desc,
            miopenFloat,
            1, out_channels, 1, 1  // Bias shape: (1, C, 1, 1) for broadcasting
        ));

        float alpha_bias = 1.0f;
        float beta_bias = 1.0f;  // Add to existing output

        MIOPEN_CHECK(miopenConvolutionForwardBias(
            miopen_guard.handle,
            &alpha_bias,
            bias_desc_guard.desc,
            bias->data<float>(),
            &beta_bias,
            output_desc_guard.desc,
            output.data<float>()
        ));
    }

    // audit-9 JJ.6: sync `stream` before `workspace` (HipBuffer RAII at L633)
    // goes out of scope.  ~HipBuffer calls bare `hipFree(ptr)`; MIOpen
    // forward + ForwardBias are async on `stream` and still read the
    // workspace — without the sync, hipFree can complete (and the page be
    // reused) before MIOpen finishes, producing nondeterministic conv
    // output under stream-parallel inference.
    HIP_CHECK(hipStreamSynchronize(stream));
    return output;
}
#endif

// Bias addition kernel for double precision
__global__ void add_bias_kernel_double(
    double* __restrict__ output,
    const double* __restrict__ bias,
    int64_t batch,
    int64_t channels,
    int64_t spatial_size,
    int64_t n
) {
    HIP_KERNEL_LOOP(idx, n) {
        int64_t c = (idx / spatial_size) % channels;
        output[idx] += bias[c];
    }
}

// Bias addition for NHWC layout (double precision)
__global__ void add_bias_kernel_nhwc_double(
    double* __restrict__ output,
    const double* __restrict__ bias,
    int64_t batch,
    int64_t height,
    int64_t width,
    int64_t channels,
    int64_t n
) {
    HIP_KERNEL_LOOP(idx, n) {
        int64_t c = idx % channels;
        output[idx] += bias[c];
    }
}

// Bias addition kernel for half precision (Float16)
__global__ void add_bias_kernel_half(
    __half* __restrict__ output,
    const __half* __restrict__ bias,
    int64_t batch,
    int64_t channels,
    int64_t spatial_size,
    int64_t n
) {
    HIP_KERNEL_LOOP(idx, n) {
        int64_t c = (idx / spatial_size) % channels;
        output[idx] = __hadd(output[idx], bias[c]);
    }
}

// Bias addition for NHWC layout (half precision)
__global__ void add_bias_kernel_nhwc_half(
    __half* __restrict__ output,
    const __half* __restrict__ bias,
    int64_t batch,
    int64_t height,
    int64_t width,
    int64_t channels,
    int64_t n
) {
    HIP_KERNEL_LOOP(idx, n) {
        int64_t c = idx % channels;
        output[idx] = __hadd(output[idx], bias[c]);
    }
}

// rocBLAS-based implementation using im2col + GEMM (per-axis stride/pad/dilation).
auto conv2d_forward_kernel(
    const Tensor& input,         // (batch, in_channels, height, width)
    const Tensor& weight,        // (out_channels, in_channels, kernel_h, kernel_w)
    const Tensor* bias,          // (out_channels) or nullptr
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t groups,
    hipStream_t stream,
    DataLayout layout = DataLayout::NCHW
) -> Tensor {
    // Scalar aliases for legacy code paths that still reference stride/padding/dilation
    // (validation, calculate_output_size in symmetric case, etc.). Asymmetric ops use
    // stride_h/stride_w/pad_h/pad_w/dil_h/dil_w directly below.
    int64_t stride = stride_h;     // for legacy validation
    int64_t padding = pad_h;        // unused in asymmetric paths
    int64_t dilation = dil_h;       // unused in asymmetric paths
    (void)padding; (void)dilation;  // silence unused warnings when asymmetric paths take over

    // Float16: upcast to Float32, compute, convert back
    if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        Tensor bias_f32;
        const Tensor* bias_f32_ptr = nullptr;
        if (bias) {
            bias_f32 = bias->to(DType::Float32);
            bias_f32_ptr = &bias_f32;
        }
        auto result = conv2d_forward_kernel(input_f32, weight_f32, bias_f32_ptr,
                                            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, stream, layout);
        auto result_f16 = result.to(DType::Float16);
        fp16_saturate(result_f16.data_ptr(), result_f16.numel(), stream);
        return result_f16;
    }

    // BFloat16: upcast to Float32, compute, convert back
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        Tensor bias_f32;
        const Tensor* bias_f32_ptr = nullptr;
        if (bias) {
            bias_f32 = bias->to(DType::Float32);
            bias_f32_ptr = &bias_f32;
        }
        auto result = conv2d_forward_kernel(input_f32, weight_f32, bias_f32_ptr,
                                            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, stream, layout);
        return result.to(DType::BFloat16);
    }

    // Extract dimensions
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    const DType dtype = input.dtype();

    int64_t batch, in_channels, height, width;

    if (layout == DataLayout::NCHW) {
        batch = input_shape[0];
        in_channels = input_shape[1];
        height = input_shape[2];
        width = input_shape[3];
    } else {  // NHWC
        batch = input_shape[0];
        height = input_shape[1];
        width = input_shape[2];
        in_channels = input_shape[3];
    }

    int64_t out_channels = weight_shape[0];
    int64_t in_channels_per_group = weight_shape[1];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Validate parameters
    if (stride_h == 0 || stride_w == 0) {
        throw std::invalid_argument("Conv2d: stride cannot be zero");
    }
    if (groups == 0) {
        throw std::invalid_argument("Conv2d: groups cannot be zero");
    }

    // Calculate output dimensions (per-axis)
    int64_t out_h = calculate_output_size(height, kernel_h, stride_h, pad_h, dil_h);
    int64_t out_w = calculate_output_size(width,  kernel_w, stride_w, pad_w, dil_w);

    // Create output tensor
    std::vector<int64_t> output_shape;
    if (layout == DataLayout::NCHW) {
        output_shape = {batch, out_channels, out_h, out_w};
    } else {
        output_shape = {batch, out_h, out_w, out_channels};
    }
    Tensor output(output_shape, dtype, input.device());

    // Initialize output to zeros
    size_t elem_size = (dtype == DType::Float64) ? sizeof(double) :
                       (dtype == DType::Float16) ? sizeof(__half) : sizeof(float);
    HIP_CHECK(hipMemsetAsync(output.data_ptr(), 0, output.numel() * elem_size, stream));

    // Cached per-thread rocBLAS handle; stream rebound per call (no per-conv
    // create/destroy).
    rocblas_handle rocblas_handle = get_cached_conv_handle(stream);

    // Process each group separately
    int64_t out_channels_per_group = out_channels / groups;

    for (int64_t g = 0; g < groups; ++g) {
        // Calculate channel offsets
        int64_t in_start = g * in_channels_per_group;
        int64_t out_start = g * out_channels_per_group;

        // Allocate im2col buffer for this group
        int64_t col_rows = batch * out_h * out_w;
        int64_t col_cols = in_channels_per_group * kernel_h * kernel_w;

        // Apply im2col transformation
        dim3 grid, block;
        int64_t total_elements = batch * out_h * out_w * in_channels_per_group * kernel_h * kernel_w;
        compute_launch_config_1d(total_elements, grid, block);

        // Matrix multiplication using rocBLAS
        // weight_group: (out_channels_per_group, in_channels_per_group * kernel_h * kernel_w)
        // col_buffer: (batch * out_h * out_w, in_channels_per_group * kernel_h * kernel_w)
        // output: (batch * out_h * out_w, out_channels_per_group)
        //
        // We compute: output = col_buffer @ weight_group^T

        int64_t M = col_rows;
        int64_t K = col_cols;
        int64_t N = out_channels_per_group;

        if (dtype == DType::Float32) {
            HipBuffer col_buf(col_rows * col_cols * sizeof(float));
            float* col_buffer = col_buf.as<float>();

            if (layout == DataLayout::NCHW) {
                const float* input_ptr = input.data<float>() + in_start * height * width;
                im2col_kernel_nchw<<<grid, block, 0, stream>>>(
                    input_ptr, col_buffer, batch, in_channels_per_group,
                    height, width, kernel_h, kernel_w,
                    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h, out_w
                );
            } else {  // NHWC
                const float* input_ptr = input.data<float>() + in_start;
                im2col_kernel_nhwc<<<grid, block, 0, stream>>>(
                    input_ptr, col_buffer, batch, height, width, in_channels_per_group,
                    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h, out_w
                );
            }
            HIP_CHECK(hipGetLastError());

            float alpha = 1.0f;
            float beta = 0.0f;

            const float* weight_ptr = weight.data<float>() + out_start * in_channels_per_group * kernel_h * kernel_w;

            // GEMM into a temp buffer in NHWC order, then transpose to
            // NCHW. Writing rocblas_sgemm directly into the NCHW output
            // pointer stored (b*oh*ow, oc) row-major = NHWC, not NCHW.
            HipBuffer gemm_out_buf(M * N * sizeof(float));
            float* gemm_out = gemm_out_buf.as<float>();

            // rocBLAS uses column-major ordering (same as cuBLAS)
            ROCBLAS_CHECK(rocblas_sgemm(
                rocblas_handle,
                rocblas_operation_transpose,  // transpose weight
                rocblas_operation_none,       // don't transpose col_buffer
                N,                           // rows of weight^T
                M,                           // rows of col_buffer
                K,                           // cols of col_buffer, rows of weight
                &alpha,
                weight_ptr,                  // weight matrix
                K,                           // leading dimension
                col_buffer,                  // col matrix
                K,                           // leading dimension
                &beta,
                gemm_out,                    // NHWC-order temp buffer
                N                            // leading dimension
            ));

            if (layout == DataLayout::NCHW) {
                dim3 t_grid, t_block;
                compute_launch_config_1d(M * N, t_grid, t_block);
                conv_nhwc_to_nchw_kernel<float><<<t_grid, t_block, 0, stream>>>(
                    gemm_out, output.data<float>(),
                    batch, out_h, out_w, out_channels, N, out_start
                );
                HIP_CHECK(hipGetLastError());
            } else {  // NHWC — the GEMM output layout matches the tensor layout
                float* output_ptr = output.data<float>() + out_start;
                HIP_CHECK(hipMemcpyAsync(output_ptr, gemm_out,
                    M * N * sizeof(float), hipMemcpyDeviceToDevice, stream));
            }

        } else if (dtype == DType::Float64) {
            HipBuffer col_buf(col_rows * col_cols * sizeof(double));
            double* col_buffer = col_buf.as<double>();

            if (layout == DataLayout::NCHW) {
                const double* input_ptr = input.data<double>() + in_start * height * width;
                im2col_kernel_nchw<<<grid, block, 0, stream>>>(
                    input_ptr, col_buffer, batch, in_channels_per_group,
                    height, width, kernel_h, kernel_w,
                    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h, out_w
                );
            } else {  // NHWC
                const double* input_ptr = input.data<double>() + in_start;
                im2col_kernel_nhwc<<<grid, block, 0, stream>>>(
                    input_ptr, col_buffer, batch, height, width, in_channels_per_group,
                    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h, out_w
                );
            }
            HIP_CHECK(hipGetLastError());

            double alpha = 1.0;
            double beta = 0.0;

            const double* weight_ptr = weight.data<double>() + out_start * in_channels_per_group * kernel_h * kernel_w;

            HipBuffer gemm_out_buf(M * N * sizeof(double));
            double* gemm_out = gemm_out_buf.as<double>();

            // rocBLAS dgemm for double precision
            ROCBLAS_CHECK(rocblas_dgemm(
                rocblas_handle,
                rocblas_operation_transpose,  // transpose weight
                rocblas_operation_none,       // don't transpose col_buffer
                N,                           // rows of weight^T
                M,                           // rows of col_buffer
                K,                           // cols of col_buffer, rows of weight
                &alpha,
                weight_ptr,                  // weight matrix
                K,                           // leading dimension
                col_buffer,                  // col matrix
                K,                           // leading dimension
                &beta,
                gemm_out,                    // NHWC-order temp buffer
                N                            // leading dimension
            ));

            if (layout == DataLayout::NCHW) {
                dim3 t_grid, t_block;
                compute_launch_config_1d(M * N, t_grid, t_block);
                conv_nhwc_to_nchw_kernel<double><<<t_grid, t_block, 0, stream>>>(
                    gemm_out, output.data<double>(),
                    batch, out_h, out_w, out_channels, N, out_start
                );
                HIP_CHECK(hipGetLastError());
            } else {  // NHWC — direct copy
                double* output_ptr = output.data<double>() + out_start;
                HIP_CHECK(hipMemcpyAsync(output_ptr, gemm_out,
                    M * N * sizeof(double), hipMemcpyDeviceToDevice, stream));
            }

        } else if (dtype == DType::Float16) {
            HipBuffer col_buf(col_rows * col_cols * sizeof(__half));
            __half* col_buffer = col_buf.as<__half>();

            if (layout == DataLayout::NCHW) {
                const __half* input_ptr = reinterpret_cast<const __half*>(input.data<Float16>()) + in_start * height * width;
                im2col_kernel_nchw<<<grid, block, 0, stream>>>(
                    input_ptr, col_buffer, batch, in_channels_per_group,
                    height, width, kernel_h, kernel_w,
                    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h, out_w
                );
            } else {  // NHWC
                const __half* input_ptr = reinterpret_cast<const __half*>(input.data<Float16>()) + in_start;
                im2col_kernel_nhwc<<<grid, block, 0, stream>>>(
                    input_ptr, col_buffer, batch, height, width, in_channels_per_group,
                    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h, out_w
                );
            }
            HIP_CHECK(hipGetLastError());

            // rocblas_half is compatible with __half
            // IEEE 754 half-precision constants
            static constexpr uint16_t kFP16One  = 0x3C00;  // 1.0
            static constexpr uint16_t kFP16Zero = 0x0000;  // 0.0
            rocblas_half alpha_h = rocblas_half{kFP16One};
            rocblas_half beta_h  = rocblas_half{kFP16Zero};

            const rocblas_half* weight_ptr = reinterpret_cast<const rocblas_half*>(weight.data<Float16>()) + out_start * in_channels_per_group * kernel_h * kernel_w;

            HipBuffer gemm_out_buf(M * N * sizeof(__half));
            __half* gemm_out = gemm_out_buf.as<__half>();

            // rocBLAS hgemm for half precision
            ROCBLAS_CHECK(rocblas_hgemm(
                rocblas_handle,
                rocblas_operation_transpose,  // transpose weight
                rocblas_operation_none,       // don't transpose col_buffer
                N,                           // rows of weight^T
                M,                           // rows of col_buffer
                K,                           // cols of col_buffer, rows of weight
                &alpha_h,
                weight_ptr,                  // weight matrix
                K,                           // leading dimension
                reinterpret_cast<const rocblas_half*>(col_buffer),  // col matrix
                K,                           // leading dimension
                &beta_h,
                reinterpret_cast<rocblas_half*>(gemm_out),  // NHWC-order temp
                N                            // leading dimension
            ));

            if (layout == DataLayout::NCHW) {
                dim3 t_grid, t_block;
                compute_launch_config_1d(M * N, t_grid, t_block);
                conv_nhwc_to_nchw_kernel<__half><<<t_grid, t_block, 0, stream>>>(
                    gemm_out,
                    reinterpret_cast<__half*>(output.data<Float16>()),
                    batch, out_h, out_w, out_channels, N, out_start
                );
                HIP_CHECK(hipGetLastError());
            } else {  // NHWC — direct copy
                __half* output_ptr =
                    reinterpret_cast<__half*>(output.data<Float16>()) + out_start;
                HIP_CHECK(hipMemcpyAsync(output_ptr, gemm_out,
                    M * N * sizeof(__half), hipMemcpyDeviceToDevice, stream));
            }

        } else {
            throw std::runtime_error("Conv2d: unsupported dtype (only Float32, Float64, and Float16 supported)");
        }
    }

    // Add bias if present
    if (bias != nullptr) {
        int64_t spatial_size = out_h * out_w;

        dim3 grid, block;
        int64_t total = batch * out_channels * out_h * out_w;
        compute_launch_config_1d(total, grid, block);

        if (dtype == DType::Float32) {
            const float* bias_data = bias->data<float>();
            float* output_data = output.data<float>();

            if (layout == DataLayout::NCHW) {
                add_bias_kernel<<<grid, block, 0, stream>>>(
                    output_data, bias_data, batch, out_channels, spatial_size, total
                );
            } else {  // NHWC
                add_bias_kernel_nhwc<<<grid, block, 0, stream>>>(
                    output_data, bias_data, batch, out_h, out_w, out_channels, total
                );
            }
        } else if (dtype == DType::Float64) {
            const double* bias_data = bias->data<double>();
            double* output_data = output.data<double>();

            if (layout == DataLayout::NCHW) {
                add_bias_kernel_double<<<grid, block, 0, stream>>>(
                    output_data, bias_data, batch, out_channels, spatial_size, total
                );
            } else {  // NHWC
                add_bias_kernel_nhwc_double<<<grid, block, 0, stream>>>(
                    output_data, bias_data, batch, out_h, out_w, out_channels, total
                );
            }
        } else if (dtype == DType::Float16) {
            const __half* bias_data = reinterpret_cast<const __half*>(bias->data<Float16>());
            __half* output_data = reinterpret_cast<__half*>(output.data<Float16>());

            if (layout == DataLayout::NCHW) {
                add_bias_kernel_half<<<grid, block, 0, stream>>>(
                    output_data, bias_data, batch, out_channels, spatial_size, total
                );
            } else {  // NHWC
                add_bias_kernel_nhwc_half<<<grid, block, 0, stream>>>(
                    output_data, bias_data, batch, out_h, out_w, out_channels, total
                );
            }
        }
        HIP_CHECK(hipGetLastError());
    }

    return output;
}

// Wave B3: scalar overload forwards to per-axis with duplicated values.
// Preserves all existing scalar callers (e.g. rocm_kernel_registry.cpp's
// symmetric dispatch lambda).
auto conv2d_forward_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    hipStream_t stream,
    DataLayout layout
) -> Tensor {
    return conv2d_forward_kernel(input, weight, bias,
                                  stride, stride, padding, padding, dilation, dilation,
                                  groups, stream, layout);
}

// ============================================================================
// Conv2d Backward HIP Implementation
// ============================================================================

auto conv2d_backward_kernel(
    const Tensor& grad_output,   // (batch, out_channels, out_h, out_w)
    const Tensor& input,         // (batch, in_channels, height, width)
    const Tensor& weight,        // (out_channels, in_channels, kernel_h, kernel_w)
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t groups,
    bool compute_grad_input,
    bool compute_grad_weight,
    bool compute_grad_bias,
    hipStream_t stream,
    DataLayout layout = DataLayout::NCHW
) -> std::tuple<Tensor, Tensor, Tensor> {
    // Float16: upcast to Float32, compute, convert back
    if (input.dtype() == DType::Float16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        auto [gi, gw, gb] = conv2d_backward_kernel(grad_output_f32, input_f32, weight_f32,
                                                     stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups,
                                                     compute_grad_input, compute_grad_weight,
                                                     compute_grad_bias, stream, layout);
        if (compute_grad_input) { gi = gi.to(DType::Float16); fp16_saturate(gi.data_ptr(), gi.numel(), stream); }
        if (compute_grad_weight) { gw = gw.to(DType::Float16); fp16_saturate(gw.data_ptr(), gw.numel(), stream); }
        if (compute_grad_bias) { gb = gb.to(DType::Float16); fp16_saturate(gb.data_ptr(), gb.numel(), stream); }
        return {gi, gw, gb};
    }

    // BFloat16: upcast to Float32, compute, convert back
    if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        auto [gi, gw, gb] = conv2d_backward_kernel(grad_output_f32, input_f32, weight_f32,
                                                     stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups,
                                                     compute_grad_input, compute_grad_weight,
                                                     compute_grad_bias, stream, layout);
        if (compute_grad_input) gi = gi.to(DType::BFloat16);
        if (compute_grad_weight) gw = gw.to(DType::BFloat16);
        if (compute_grad_bias) gb = gb.to(DType::BFloat16);
        return {gi, gw, gb};
    }

    // Extract dimensions
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    auto grad_shape = grad_output.shape();

    int64_t batch, in_channels, height, width;

    if (layout == DataLayout::NCHW) {
        batch = input_shape[0];
        in_channels = input_shape[1];
        height = input_shape[2];
        width = input_shape[3];
    } else {  // NHWC
        batch = input_shape[0];
        height = input_shape[1];
        width = input_shape[2];
        in_channels = input_shape[3];
    }

    int64_t out_channels = weight_shape[0];
    int64_t in_channels_per_group = weight_shape[1];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    int64_t out_h, out_w;
    if (layout == DataLayout::NCHW) {
        out_h = grad_shape[2];
        out_w = grad_shape[3];
    } else {
        out_h = grad_shape[1];
        out_w = grad_shape[2];
    }

    // Validate parameters
    if (stride_h == 0 || stride_w == 0) {
        throw std::invalid_argument("Conv2d backward: stride cannot be zero");
    }
    if (groups == 0) {
        throw std::invalid_argument("Conv2d backward: groups cannot be zero");
    }

    // Initialize outputs
    std::vector<int64_t> grad_input_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(grad_input_shape, input.dtype(), input.device());
    Tensor grad_weight({out_channels, in_channels_per_group, kernel_h, kernel_w}, weight.dtype(), weight.device());
    Tensor grad_bias({out_channels}, weight.dtype(), weight.device());

    DType dtype = input.dtype();
    size_t elem_size = (dtype == DType::Float64) ? sizeof(double) :
                       (dtype == DType::Float16) ? sizeof(__half) : sizeof(float);

    if (compute_grad_input) {
        HIP_CHECK(hipMemsetAsync(grad_input.data_ptr(), 0, grad_input.numel() * elem_size, stream));
    }
    if (compute_grad_weight) {
        HIP_CHECK(hipMemsetAsync(grad_weight.data_ptr(), 0, grad_weight.numel() * elem_size, stream));
    }
    if (compute_grad_bias) {
        HIP_CHECK(hipMemsetAsync(grad_bias.data_ptr(), 0, grad_bias.numel() * elem_size, stream));
    }

    // Cached per-thread rocBLAS handle; stream rebound per call.
    rocblas_handle rocblas_handle = get_cached_conv_handle(stream);

    int64_t out_channels_per_group = out_channels / groups;
    int64_t col_rows = batch * out_h * out_w;
    int64_t col_cols = in_channels_per_group * kernel_h * kernel_w;

    for (int64_t g = 0; g < groups; ++g) {
        int64_t in_start = g * in_channels_per_group;
        int64_t out_start = g * out_channels_per_group;

        // Gradient w.r.t input
        if (compute_grad_input) {
            // Compute grad_col = grad_output @ weight
            int64_t M = col_rows;
            int64_t K = out_channels_per_group;
            int64_t N = col_cols;

            if (dtype == DType::Float16) {
                // Float16 path using rocblas_hgemm
                HipBuffer grad_col_buf(col_rows * col_cols * sizeof(rocblas_half));
                rocblas_half* grad_col = grad_col_buf.as<rocblas_half>();

                rocblas_half alpha_h{static_cast<uint16_t>(0x3C00)};  // 1.0 in FP16
                rocblas_half beta_h{static_cast<uint16_t>(0x0000)};   // 0.0 in FP16

                // rocBLAS reads grad_out as op(B) column-major [K×M] with ld=K,
                // i.e. row-major [col_row][oc] = NHWC. For an NCHW tensor we
                // must permute the grad_output into a contiguous NHWC temp
                // first; an NHWC tensor is already [col_row][oc] row-major.
                HipBuffer grad_out_nhwc_buf(0);
                const rocblas_half* grad_out_ptr;
                if (layout == DataLayout::NCHW) {
                    grad_out_nhwc_buf = HipBuffer(col_rows * out_channels_per_group * sizeof(rocblas_half));
                    rocblas_half* grad_out_nhwc = grad_out_nhwc_buf.as<rocblas_half>();
                    dim3 p_grid, p_block;
                    compute_launch_config_1d(col_rows * out_channels_per_group, p_grid, p_block);
                    conv_nchw_to_nhwc_kernel<__half><<<p_grid, p_block, 0, stream>>>(
                        reinterpret_cast<const __half*>(grad_output.data<Float16>()),
                        reinterpret_cast<__half*>(grad_out_nhwc),
                        batch, out_h, out_w, out_channels, out_channels_per_group, out_start
                    );
                    HIP_CHECK(hipGetLastError());
                    grad_out_ptr = grad_out_nhwc;
                } else {
                    grad_out_ptr = reinterpret_cast<const rocblas_half*>(grad_output.data<Float16>()) + out_start;
                }
                const rocblas_half* weight_ptr = reinterpret_cast<const rocblas_half*>(weight.data<Float16>()) + out_start * in_channels_per_group * kernel_h * kernel_w;

                ROCBLAS_CHECK(rocblas_hgemm(
                    rocblas_handle,
                    rocblas_operation_none,
                    rocblas_operation_none,
                    N, M, K,
                    &alpha_h,
                    weight_ptr, N,
                    grad_out_ptr, K,
                    &beta_h,
                    grad_col, N
                ));

                // Apply col2im
                dim3 grid, block;
                int64_t total_output = batch * in_channels_per_group * height * width;
                compute_launch_config_1d(total_output, grid, block);

                __half* grad_input_ptr;
                if (layout == DataLayout::NCHW) {
                    grad_input_ptr = reinterpret_cast<__half*>(grad_input.data<Float16>()) + in_start * height * width;
                    col2im_kernel_nchw<<<grid, block, 0, stream>>>(
                        reinterpret_cast<__half*>(grad_col), grad_input_ptr, batch, in_channels_per_group,
                        height, width, kernel_h, kernel_w,
                        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h, out_w
                    );
                } else {
                    grad_input_ptr = reinterpret_cast<__half*>(grad_input.data<Float16>()) + in_start;
                    col2im_kernel_nhwc<<<grid, block, 0, stream>>>(
                        reinterpret_cast<__half*>(grad_col), grad_input_ptr, batch, height, width, in_channels_per_group,
                        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h, out_w
                    );
                }
                HIP_CHECK(hipGetLastError());
            } else if (dtype == DType::Float64) {
                // Float64 path using rocblas_dgemm
                HipBuffer grad_col_buf(col_rows * col_cols * sizeof(double));
                double* grad_col = grad_col_buf.as<double>();

                double alpha = 1.0;
                double beta = 0.0;

                // rocBLAS reads grad_out as op(B) column-major [K×M] with ld=K,
                // i.e. row-major [col_row][oc] = NHWC. Permute NCHW grad_output
                // into a contiguous NHWC temp; NHWC is already that layout.
                HipBuffer grad_out_nhwc_buf(0);
                const double* grad_out_ptr;
                if (layout == DataLayout::NCHW) {
                    grad_out_nhwc_buf = HipBuffer(col_rows * out_channels_per_group * sizeof(double));
                    double* grad_out_nhwc = grad_out_nhwc_buf.as<double>();
                    dim3 p_grid, p_block;
                    compute_launch_config_1d(col_rows * out_channels_per_group, p_grid, p_block);
                    conv_nchw_to_nhwc_kernel<double><<<p_grid, p_block, 0, stream>>>(
                        grad_output.data<double>(), grad_out_nhwc,
                        batch, out_h, out_w, out_channels, out_channels_per_group, out_start
                    );
                    HIP_CHECK(hipGetLastError());
                    grad_out_ptr = grad_out_nhwc;
                } else {
                    grad_out_ptr = grad_output.data<double>() + out_start;
                }
                const double* weight_ptr = weight.data<double>() + out_start * in_channels_per_group * kernel_h * kernel_w;

                ROCBLAS_CHECK(rocblas_dgemm(
                    rocblas_handle,
                    rocblas_operation_none,
                    rocblas_operation_none,
                    N, M, K,
                    &alpha,
                    weight_ptr, N,
                    grad_out_ptr, K,
                    &beta,
                    grad_col, N
                ));

                // Apply col2im
                dim3 grid, block;
                int64_t total_output = batch * in_channels_per_group * height * width;
                compute_launch_config_1d(total_output, grid, block);

                double* grad_input_ptr;
                if (layout == DataLayout::NCHW) {
                    grad_input_ptr = grad_input.data<double>() + in_start * height * width;
                    col2im_kernel_nchw<<<grid, block, 0, stream>>>(
                        grad_col, grad_input_ptr, batch, in_channels_per_group,
                        height, width, kernel_h, kernel_w,
                        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h, out_w
                    );
                } else {
                    grad_input_ptr = grad_input.data<double>() + in_start;
                    col2im_kernel_nhwc<<<grid, block, 0, stream>>>(
                        grad_col, grad_input_ptr, batch, height, width, in_channels_per_group,
                        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h, out_w
                    );
                }
                HIP_CHECK(hipGetLastError());
            } else {
                // Float32 path (original code)
                HipBuffer grad_col_buf(col_rows * col_cols * sizeof(float));
                float* grad_col = grad_col_buf.as<float>();

                float alpha = 1.0f;
                float beta = 0.0f;

                // rocBLAS reads grad_out as op(B) column-major [K×M] with ld=K,
                // i.e. row-major [col_row][oc] = NHWC. Permute NCHW grad_output
                // into a contiguous NHWC temp; NHWC is already that layout.
                HipBuffer grad_out_nhwc_buf(0);
                const float* grad_out_ptr;
                if (layout == DataLayout::NCHW) {
                    grad_out_nhwc_buf = HipBuffer(col_rows * out_channels_per_group * sizeof(float));
                    float* grad_out_nhwc = grad_out_nhwc_buf.as<float>();
                    dim3 p_grid, p_block;
                    compute_launch_config_1d(col_rows * out_channels_per_group, p_grid, p_block);
                    conv_nchw_to_nhwc_kernel<float><<<p_grid, p_block, 0, stream>>>(
                        grad_output.data<float>(), grad_out_nhwc,
                        batch, out_h, out_w, out_channels, out_channels_per_group, out_start
                    );
                    HIP_CHECK(hipGetLastError());
                    grad_out_ptr = grad_out_nhwc;
                } else {
                    grad_out_ptr = grad_output.data<float>() + out_start;
                }
                const float* weight_ptr = weight.data<float>() + out_start * in_channels_per_group * kernel_h * kernel_w;

                ROCBLAS_CHECK(rocblas_sgemm(
                    rocblas_handle,
                    rocblas_operation_none,      // don't transpose weight
                    rocblas_operation_none,      // don't transpose grad_output
                    N,                          // cols of result
                    M,                          // rows of result
                    K,                          // inner dimension
                    &alpha,
                    weight_ptr,                 // weight matrix
                    N,                          // leading dim
                    grad_out_ptr,               // grad_output matrix
                    K,                          // leading dim
                    &beta,
                    grad_col,                   // output matrix
                    N                           // leading dim
                ));

                // Apply col2im
                dim3 grid, block;
                int64_t total_output = batch * in_channels_per_group * height * width;
                compute_launch_config_1d(total_output, grid, block);

                float* grad_input_ptr;
                if (layout == DataLayout::NCHW) {
                    grad_input_ptr = grad_input.data<float>() + in_start * height * width;
                    col2im_kernel_nchw<<<grid, block, 0, stream>>>(
                        grad_col, grad_input_ptr, batch, in_channels_per_group,
                        height, width, kernel_h, kernel_w,
                        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h, out_w
                    );
                } else {  // NHWC
                    grad_input_ptr = grad_input.data<float>() + in_start;
                    col2im_kernel_nhwc<<<grid, block, 0, stream>>>(
                        grad_col, grad_input_ptr, batch, height, width, in_channels_per_group,
                        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h, out_w
                    );
                }
                HIP_CHECK(hipGetLastError());
            }
        }

        // Gradient w.r.t weight
        if (compute_grad_weight) {
            if (dtype == DType::Float16) {
                // Float16 path
                HipBuffer input_col_buf(col_rows * col_cols * sizeof(rocblas_half));
                rocblas_half* input_col = input_col_buf.as<rocblas_half>();

                dim3 grid, block;
                int64_t total_elements = batch * out_h * out_w * in_channels_per_group * kernel_h * kernel_w;
                compute_launch_config_1d(total_elements, grid, block);

                if (layout == DataLayout::NCHW) {
                    const __half* input_ptr = reinterpret_cast<const __half*>(input.data<Float16>()) + in_start * height * width;
                    im2col_kernel_nchw<<<grid, block, 0, stream>>>(
                        input_ptr, reinterpret_cast<__half*>(input_col), batch, in_channels_per_group,
                        height, width, kernel_h, kernel_w,
                        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h, out_w
                    );
                } else {
                    const __half* input_ptr = reinterpret_cast<const __half*>(input.data<Float16>()) + in_start;
                    im2col_kernel_nhwc<<<grid, block, 0, stream>>>(
                        input_ptr, reinterpret_cast<__half*>(input_col), batch, height, width, in_channels_per_group,
                        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h, out_w
                    );
                }
                HIP_CHECK(hipGetLastError());

                int64_t M = out_channels_per_group;
                int64_t K = col_rows;
                int64_t N = col_cols;

                rocblas_half alpha_h{static_cast<uint16_t>(0x3C00)};
                rocblas_half beta_h{static_cast<uint16_t>(0x0000)};

                // With op(B)=transpose, rocBLAS reads grad_out as stored [M×K]
                // col-major ld=M => row-major [col_row][oc] = NHWC. Permute the
                // NCHW grad_output into a contiguous NHWC temp first.
                HipBuffer grad_out_nhwc_buf(0);
                const rocblas_half* grad_out_ptr;
                if (layout == DataLayout::NCHW) {
                    grad_out_nhwc_buf = HipBuffer(col_rows * out_channels_per_group * sizeof(rocblas_half));
                    rocblas_half* grad_out_nhwc = grad_out_nhwc_buf.as<rocblas_half>();
                    dim3 p_grid, p_block;
                    compute_launch_config_1d(col_rows * out_channels_per_group, p_grid, p_block);
                    conv_nchw_to_nhwc_kernel<__half><<<p_grid, p_block, 0, stream>>>(
                        reinterpret_cast<const __half*>(grad_output.data<Float16>()),
                        reinterpret_cast<__half*>(grad_out_nhwc),
                        batch, out_h, out_w, out_channels, out_channels_per_group, out_start
                    );
                    HIP_CHECK(hipGetLastError());
                    grad_out_ptr = grad_out_nhwc;
                } else {
                    grad_out_ptr = reinterpret_cast<const rocblas_half*>(grad_output.data<Float16>()) + out_start;
                }
                rocblas_half* grad_weight_ptr = reinterpret_cast<rocblas_half*>(grad_weight.data<Float16>()) + out_start * in_channels_per_group * kernel_h * kernel_w;

                ROCBLAS_CHECK(rocblas_hgemm(
                    rocblas_handle,
                    rocblas_operation_none,
                    rocblas_operation_transpose,
                    N, M, K,
                    &alpha_h,
                    input_col, N,
                    grad_out_ptr, out_channels_per_group,
                    &beta_h,
                    grad_weight_ptr, N
                ));

            } else if (dtype == DType::Float64) {
                // Float64 path using rocblas_dgemm
                HipBuffer input_col_buf(col_rows * col_cols * sizeof(double));
                double* input_col = input_col_buf.as<double>();

                dim3 grid, block;
                int64_t total_elements = batch * out_h * out_w * in_channels_per_group * kernel_h * kernel_w;
                compute_launch_config_1d(total_elements, grid, block);

                if (layout == DataLayout::NCHW) {
                    const double* input_ptr = input.data<double>() + in_start * height * width;
                    im2col_kernel_nchw<<<grid, block, 0, stream>>>(
                        input_ptr, input_col, batch, in_channels_per_group,
                        height, width, kernel_h, kernel_w,
                        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h, out_w
                    );
                } else {
                    const double* input_ptr = input.data<double>() + in_start;
                    im2col_kernel_nhwc<<<grid, block, 0, stream>>>(
                        input_ptr, input_col, batch, height, width, in_channels_per_group,
                        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h, out_w
                    );
                }
                HIP_CHECK(hipGetLastError());

                int64_t M = out_channels_per_group;
                int64_t K = col_rows;
                int64_t N = col_cols;

                double alpha = 1.0;
                double beta = 0.0;

                // With op(B)=transpose, rocBLAS reads grad_out as stored [M×K]
                // col-major ld=M => row-major [col_row][oc] = NHWC. Permute the
                // NCHW grad_output into a contiguous NHWC temp first.
                HipBuffer grad_out_nhwc_buf(0);
                const double* grad_out_ptr;
                if (layout == DataLayout::NCHW) {
                    grad_out_nhwc_buf = HipBuffer(col_rows * out_channels_per_group * sizeof(double));
                    double* grad_out_nhwc = grad_out_nhwc_buf.as<double>();
                    dim3 p_grid, p_block;
                    compute_launch_config_1d(col_rows * out_channels_per_group, p_grid, p_block);
                    conv_nchw_to_nhwc_kernel<double><<<p_grid, p_block, 0, stream>>>(
                        grad_output.data<double>(), grad_out_nhwc,
                        batch, out_h, out_w, out_channels, out_channels_per_group, out_start
                    );
                    HIP_CHECK(hipGetLastError());
                    grad_out_ptr = grad_out_nhwc;
                } else {
                    grad_out_ptr = grad_output.data<double>() + out_start;
                }
                double* grad_weight_ptr = grad_weight.data<double>() + out_start * in_channels_per_group * kernel_h * kernel_w;

                ROCBLAS_CHECK(rocblas_dgemm(
                    rocblas_handle,
                    rocblas_operation_none,
                    rocblas_operation_transpose,
                    N, M, K,
                    &alpha,
                    input_col, N,
                    grad_out_ptr, out_channels_per_group,
                    &beta,
                    grad_weight_ptr, N
                ));

            } else {
                // Float32 path
                HipBuffer input_col_buf(col_rows * col_cols * sizeof(float));
                float* input_col = input_col_buf.as<float>();

                dim3 grid, block;
                int64_t total_elements = batch * out_h * out_w * in_channels_per_group * kernel_h * kernel_w;
                compute_launch_config_1d(total_elements, grid, block);

                if (layout == DataLayout::NCHW) {
                    const float* input_ptr = input.data<float>() + in_start * height * width;
                    im2col_kernel_nchw<<<grid, block, 0, stream>>>(
                        input_ptr, input_col, batch, in_channels_per_group,
                        height, width, kernel_h, kernel_w,
                        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h, out_w
                    );
                } else {
                    const float* input_ptr = input.data<float>() + in_start;
                    im2col_kernel_nhwc<<<grid, block, 0, stream>>>(
                        input_ptr, input_col, batch, height, width, in_channels_per_group,
                        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, out_h, out_w
                    );
                }
                HIP_CHECK(hipGetLastError());

                int64_t M = out_channels_per_group;
                int64_t K = col_rows;
                int64_t N = col_cols;

                float alpha = 1.0f;
                float beta = 0.0f;

                // With op(B)=transpose, rocBLAS reads grad_out as stored [M×K]
                // col-major ld=M => row-major [col_row][oc] = NHWC. Permute the
                // NCHW grad_output into a contiguous NHWC temp first.
                HipBuffer grad_out_nhwc_buf(0);
                const float* grad_out_ptr;
                if (layout == DataLayout::NCHW) {
                    grad_out_nhwc_buf = HipBuffer(col_rows * out_channels_per_group * sizeof(float));
                    float* grad_out_nhwc = grad_out_nhwc_buf.as<float>();
                    dim3 p_grid, p_block;
                    compute_launch_config_1d(col_rows * out_channels_per_group, p_grid, p_block);
                    conv_nchw_to_nhwc_kernel<float><<<p_grid, p_block, 0, stream>>>(
                        grad_output.data<float>(), grad_out_nhwc,
                        batch, out_h, out_w, out_channels, out_channels_per_group, out_start
                    );
                    HIP_CHECK(hipGetLastError());
                    grad_out_ptr = grad_out_nhwc;
                } else {
                    grad_out_ptr = grad_output.data<float>() + out_start;
                }
                float* grad_weight_ptr = grad_weight.data<float>() + out_start * in_channels_per_group * kernel_h * kernel_w;

                ROCBLAS_CHECK(rocblas_sgemm(
                    rocblas_handle,
                    rocblas_operation_none,
                    rocblas_operation_transpose,
                    N, M, K,
                    &alpha,
                    input_col, N,
                    grad_out_ptr, out_channels_per_group,
                    &beta,
                    grad_weight_ptr, N
                ));

            }
        }
    }

    // Gradient w.r.t bias
    if (compute_grad_bias) {
        int64_t spatial_size = out_h * out_w;

        dim3 grid, block;
        compute_launch_config_1d(out_channels, grid, block);

        if (dtype == DType::Float16) {
            const __half* grad_out_data = reinterpret_cast<const __half*>(grad_output.data<Float16>());
            __half* grad_bias_data = reinterpret_cast<__half*>(grad_bias.data<Float16>());

            sum_bias_grad_kernel_wave_reduce<<<out_channels, 256, 0, stream>>>(
                grad_out_data, grad_bias_data, batch, out_channels, spatial_size
            );
        } else if (dtype == DType::Float64) {
            const double* grad_out_data = grad_output.data<double>();
            double* grad_bias_data = grad_bias.data<double>();

            sum_bias_grad_kernel_wave_reduce<<<out_channels, 256, 0, stream>>>(
                grad_out_data, grad_bias_data, batch, out_channels, spatial_size
            );
        } else {
            const float* grad_out_data = grad_output.data<float>();
            float* grad_bias_data = grad_bias.data<float>();

            sum_bias_grad_kernel_wave_reduce<<<out_channels, 256, 0, stream>>>(
                grad_out_data, grad_bias_data, batch, out_channels, spatial_size
            );
        }
        HIP_CHECK(hipGetLastError());
    }

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// ============================================================================
// Standalone Gradient Functions for Separate Calls
// ============================================================================

// Wave B3: scalar overload of conv2d_backward_kernel forwards to per-axis.
auto conv2d_backward_kernel(
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
    hipStream_t stream,
    DataLayout layout
) -> std::tuple<Tensor, Tensor, Tensor> {
    return conv2d_backward_kernel(grad_output, input, weight,
                                   stride, stride, padding, padding, dilation, dilation,
                                   groups, compute_grad_input, compute_grad_weight,
                                   compute_grad_bias, stream, layout);
}

auto conv2d_backward_input(
    const Tensor& grad_output,
    const Tensor& weight,
    const std::vector<int64_t>& input_shape,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups,
    hipStream_t stream
) -> Tensor {
    Tensor dummy_input(input_shape, grad_output.dtype(), grad_output.device());
    auto [grad_input, _, __] = conv2d_backward_kernel(
        grad_output, dummy_input, weight,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups,
        true, false, false, stream);
    return grad_input;
}

// Scalar overload of conv2d_backward_input forwards to per-axis.
auto conv2d_backward_input(
    const Tensor& grad_output,
    const Tensor& weight,
    const std::vector<int64_t>& input_shape,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    hipStream_t stream
) -> Tensor {
    return conv2d_backward_input(grad_output, weight, input_shape,
                                  stride, stride, padding, padding, dilation, dilation,
                                  groups, stream);
}

auto conv2d_backward_weight(
    const Tensor& grad_output,
    const Tensor& input,
    const std::vector<int64_t>& weight_shape,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups,
    hipStream_t stream
) -> Tensor {
    Tensor dummy_weight(weight_shape, input.dtype(), input.device());
    auto [_, grad_weight, __] = conv2d_backward_kernel(
        grad_output, input, dummy_weight,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups,
        false, true, false, stream);
    return grad_weight;
}

// Scalar overload of conv2d_backward_weight forwards to per-axis.
auto conv2d_backward_weight(
    const Tensor& grad_output,
    const Tensor& input,
    const std::vector<int64_t>& weight_shape,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    hipStream_t stream
) -> Tensor {
    return conv2d_backward_weight(grad_output, input, weight_shape,
                                   stride, stride, padding, padding, dilation, dilation,
                                   groups, stream);
}

auto conv2d_backward_bias(
    const Tensor& grad_output,
    hipStream_t stream
) -> Tensor {
    // BFloat16: upcast to Float32, compute, convert back
    if (grad_output.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto result = conv2d_backward_bias(grad_output_f32, stream);
        return result.to(DType::BFloat16);
    }

    auto grad_shape = grad_output.shape();
    int64_t out_channels = grad_shape[1];  // Assumes NCHW
    DType dtype = grad_output.dtype();

    Tensor grad_bias({out_channels}, dtype, grad_output.device());
    size_t elem_size = (dtype == DType::Float64) ? sizeof(double) :
                       (dtype == DType::Float16) ? sizeof(__half) : sizeof(float);
    HIP_CHECK(hipMemsetAsync(grad_bias.data_ptr(), 0, out_channels * elem_size, stream));

    int64_t batch = grad_shape[0];
    int64_t out_h = grad_shape[2];
    int64_t out_w = grad_shape[3];
    int64_t spatial_size = out_h * out_w;

    dim3 grid, block;
    compute_launch_config_1d(out_channels, grid, block);

    if (dtype == DType::Float16) {
        sum_bias_grad_kernel_wave_reduce<<<out_channels, 256, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            reinterpret_cast<__half*>(grad_bias.data<Float16>()),
            batch, out_channels, spatial_size
        );
    } else if (dtype == DType::Float64) {
        sum_bias_grad_kernel_wave_reduce<<<out_channels, 256, 0, stream>>>(
            grad_output.data<double>(), grad_bias.data<double>(),
            batch, out_channels, spatial_size
        );
    } else {
        sum_bias_grad_kernel_wave_reduce<<<out_channels, 256, 0, stream>>>(
            grad_output.data<float>(), grad_bias.data<float>(),
            batch, out_channels, spatial_size
        );
    }
    HIP_CHECK(hipGetLastError());

    return grad_bias;
}

// ==============================================================================
// Transpose Convolution (Deconvolution) Forward
// ==============================================================================

template<typename T>
__global__ void conv_transpose2d_forward_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    T* output,
    int64_t batch,
    int64_t in_channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_channels,
    int64_t out_h,
    int64_t out_w,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t output_padding_h,
    int64_t output_padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups
) {
    int64_t total_elements = batch * out_channels * out_h * out_w;
    // Grouped conv-transpose: weight is [in_channels, out_channels/groups, kH, kW].
    // Output channel oc belongs to group g and sees only that group's inputs.
    // groups==1 reduces to the dense case (out_cpg==out_channels, in_cpg==in_channels).
    const int64_t out_cpg = out_channels / groups;
    const int64_t in_cpg  = in_channels / groups;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements;
         idx += blockDim.x * gridDim.x) {

        int64_t ow = idx % out_w;
        int64_t oh = (idx / out_w) % out_h;
        int64_t oc = (idx / (out_w * out_h)) % out_channels;
        int64_t n = idx / (out_w * out_h * out_channels);

        int64_t g = oc / out_cpg;          // group of this output channel
        int64_t oc_in_g = oc % out_cpg;    // its index within the group

        T sum = bias ? bias[oc] : T(0);

        // Q.8: honour per-axis dilation. The output position oh/ow maps back to
        // input via (oh + padding_h - kh * dilation_h) / stride_h (and same for
        // W). Setting dilation_h/w=1 reproduces the previous behaviour.
        for (int64_t ic_local = 0; ic_local < in_cpg; ++ic_local) {
            int64_t ic = g * in_cpg + ic_local;
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    // Find corresponding input position
                    int64_t h_offset = oh + padding_h - kh * dilation_h;
                    int64_t w_offset = ow + padding_w - kw * dilation_w;

                    if (h_offset % stride_h != 0 || w_offset % stride_w != 0) continue;

                    int64_t ih = h_offset / stride_h;
                    int64_t iw = w_offset / stride_w;

                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        int64_t input_idx = n * (in_channels * in_h * in_w) +
                                           ic * (in_h * in_w) + ih * in_w + iw;
                        // Weight: [in_channels, out_channels/groups, kh, kw]
                        int64_t weight_idx = ic * (out_cpg * kernel_h * kernel_w) +
                                            oc_in_g * (kernel_h * kernel_w) + kh * kernel_w + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        output[idx] = sum;
    }
}

auto conv_transpose2d_forward_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t output_padding_h,
    int64_t output_padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    hipStream_t stream
) -> Tensor {
    // Q.8: per-axis dilation_h/w added. PyTorch ConvTranspose2d supports
    // dilation > 1; previous ROCm kernel silently forced 1.
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];

    // ConvTranspose weight layout is [in_channels, out_channels/groups, kH, kW],
    // so the true output-channel count is weight_shape[1] * groups. The previous
    // code used weight_shape[1] directly, producing out_channels/groups channels
    // (e.g. 8 instead of 32 for groups=4) and ignoring grouping entirely.
    int64_t out_channels = weight_shape[1] * groups;
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Output size for transpose conv (dilated kernel effective size).
    int64_t out_h = (in_h - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1;
    int64_t out_w = (in_w - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1;

    Tensor output({batch, out_channels, out_h, out_w}, input.dtype(), input.device());

    int64_t total_elements = output.numel();
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total_elements + threads - 1) / threads;
    // Empty output: a zero-grid launch is rejected by HIP; return as-is.
    if (blocks == 0) return output;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(conv_transpose2d_forward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(), weight.data<float>(),
            bias ? bias->data<float>() : nullptr,
            output.data<float>(),
            batch, in_channels, in_h, in_w, out_channels, out_h, out_w,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, output_padding_h, output_padding_w,
            dilation_h, dilation_w, groups);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(conv_transpose2d_forward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(), weight.data<double>(),
            bias ? bias->data<double>() : nullptr,
            output.data<double>(),
            batch, in_channels, in_h, in_w, out_channels, out_h, out_w,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, output_padding_h, output_padding_w,
            dilation_h, dilation_w, groups);
    } else if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        Tensor bias_f32;
        const Tensor* bias_f32_ptr = nullptr;
        if (bias) {
            bias_f32 = bias->to(DType::Float32);
            bias_f32_ptr = &bias_f32;
        }
        auto result = conv_transpose2d_forward_kernel(input_f32, weight_f32, bias_f32_ptr,
                                                       stride_h, stride_w, padding_h, padding_w,
                                                       output_padding_h, output_padding_w,
                                                       dilation_h, dilation_w, groups, stream);
        auto result_f16 = result.to(DType::Float16);
        fp16_saturate(result_f16.data_ptr(), result_f16.numel(), stream);
        return result_f16;
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        Tensor bias_f32;
        const Tensor* bias_f32_ptr = nullptr;
        if (bias) {
            bias_f32 = bias->to(DType::Float32);
            bias_f32_ptr = &bias_f32;
        }
        auto result = conv_transpose2d_forward_kernel(input_f32, weight_f32, bias_f32_ptr,
                                                       stride_h, stride_w, padding_h, padding_w,
                                                       output_padding_h, output_padding_w,
                                                       dilation_h, dilation_w, groups, stream);
        return result.to(DType::BFloat16);
    } else {
        throw std::runtime_error("conv_transpose2d_forward: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

// ==============================================================================
// Depthwise Convolution Forward
// ==============================================================================

template<typename T>
__global__ void depthwise_conv2d_forward_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_h,
    int64_t out_w,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w
) {
    int64_t total_elements = batch * channels * out_h * out_w;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements;
         idx += blockDim.x * gridDim.x) {

        int64_t ow = idx % out_w;
        int64_t oh = (idx / out_w) % out_h;
        int64_t c = (idx / (out_w * out_h)) % channels;
        int64_t n = idx / (out_w * out_h * channels);

        T sum = bias ? bias[c] : T(0);

        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                int64_t ih = oh * stride_h - padding_h + kh * dilation_h;
                int64_t iw = ow * stride_w - padding_w + kw * dilation_w;

                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int64_t input_idx = n * (channels * in_h * in_w) +
                                       c * (in_h * in_w) + ih * in_w + iw;
                    // Weight: [channels, 1, kh, kw]
                    int64_t weight_idx = c * (kernel_h * kernel_w) + kh * kernel_w + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }

        output[idx] = sum;
    }
}

auto depthwise_conv2d_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    hipStream_t stream
) -> Tensor {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];

    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Compute output dimensions
    int64_t out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    Tensor output({batch, channels, out_h, out_w}, input.dtype(), input.device());

    int64_t total_elements = output.numel();
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total_elements + threads - 1) / threads;
    // Empty output: a zero-grid launch is rejected by HIP; return as-is.
    if (blocks == 0) return output;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(depthwise_conv2d_forward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(), weight.data<float>(),
            bias ? bias->data<float>() : nullptr,
            output.data<float>(),
            batch, channels, in_h, in_w, out_h, out_w,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(depthwise_conv2d_forward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(), weight.data<double>(),
            bias ? bias->data<double>() : nullptr,
            output.data<double>(),
            batch, channels, in_h, in_w, out_h, out_w,
            kernel_h, kernel_w, stride_h, stride_w,
            padding_h, padding_w, dilation_h, dilation_w);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        Tensor bias_f32;
        const Tensor* bias_f32_ptr = nullptr;
        if (bias) {
            bias_f32 = bias->to(DType::Float32);
            bias_f32_ptr = &bias_f32;
        }
        auto result = depthwise_conv2d_kernel(input_f32, weight_f32, bias_f32_ptr,
                                               stride_h, stride_w, padding_h, padding_w,
                                               dilation_h, dilation_w, stream);
        return result.to(DType::BFloat16);
    } else {
        throw std::runtime_error("depthwise_conv2d: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

// ==============================================================================
// Deformable Convolution v2 (DCNv2) - HIP Kernels
// ==============================================================================

// DCN accumulator type: double for Float64 inputs, float otherwise (incl __half).
// Keeps the Float64 forward/backward path at full double precision instead of
// silently truncating to single precision.
template <typename T>
using DcnAcc = std::conditional_t<std::is_same_v<T, double>, double, float>;

// Device-side bilinear interpolation for deformable convolution.
// Returns 0 for out-of-bounds positions. Computes in accumulator type Acc.
template <typename Acc, typename T>
__device__ inline Acc dcn_bilinear_interpolate(
    const T* data, int64_t H, int64_t W, Acc h, Acc w) {
    if (h <= Acc(-1) || h >= static_cast<Acc>(H) ||
        w <= Acc(-1) || w >= static_cast<Acc>(W))
        return Acc(0);

    int64_t h_low = static_cast<int64_t>(floor(h));
    int64_t w_low = static_cast<int64_t>(floor(w));
    int64_t h_high = h_low + 1;
    int64_t w_high = w_low + 1;

    Acc lh = h - static_cast<Acc>(h_low);
    Acc lw = w - static_cast<Acc>(w_low);
    Acc hh = Acc(1) - lh;
    Acc hw = Acc(1) - lw;

    Acc v00 = (h_low >= 0 && h_low < H && w_low >= 0 && w_low < W)
                    ? static_cast<Acc>(data[h_low * W + w_low]) : Acc(0);
    Acc v01 = (h_low >= 0 && h_low < H && w_high >= 0 && w_high < W)
                    ? static_cast<Acc>(data[h_low * W + w_high]) : Acc(0);
    Acc v10 = (h_high >= 0 && h_high < H && w_low >= 0 && w_low < W)
                    ? static_cast<Acc>(data[h_high * W + w_low]) : Acc(0);
    Acc v11 = (h_high >= 0 && h_high < H && w_high >= 0 && w_high < W)
                    ? static_cast<Acc>(data[h_high * W + w_high]) : Acc(0);

    return hh * hw * v00 + hh * lw * v01 + lh * hw * v10 + lh * lw * v11;
}

// Device-side bilinear interpolation offset gradient (d/dh, d/dw) in Acc.
template <typename Acc, typename T>
__device__ inline void dcn_bilinear_offset_gradient(
    const T* data, int64_t H, int64_t W, Acc h, Acc w,
    Acc& grad_h, Acc& grad_w) {
    grad_h = Acc(0);
    grad_w = Acc(0);
    if (h <= Acc(-1) || h >= static_cast<Acc>(H) ||
        w <= Acc(-1) || w >= static_cast<Acc>(W))
        return;

    int64_t h_low = static_cast<int64_t>(floor(h));
    int64_t w_low = static_cast<int64_t>(floor(w));
    int64_t h_high = h_low + 1;
    int64_t w_high = w_low + 1;

    Acc lh = h - static_cast<Acc>(h_low);
    Acc lw = w - static_cast<Acc>(w_low);
    Acc hh = Acc(1) - lh;
    Acc hw = Acc(1) - lw;

    Acc v00 = (h_low >= 0 && h_low < H && w_low >= 0 && w_low < W)
                    ? static_cast<Acc>(data[h_low * W + w_low]) : Acc(0);
    Acc v01 = (h_low >= 0 && h_low < H && w_high >= 0 && w_high < W)
                    ? static_cast<Acc>(data[h_low * W + w_high]) : Acc(0);
    Acc v10 = (h_high >= 0 && h_high < H && w_low >= 0 && w_low < W)
                    ? static_cast<Acc>(data[h_high * W + w_low]) : Acc(0);
    Acc v11 = (h_high >= 0 && h_high < H && w_high >= 0 && w_high < W)
                    ? static_cast<Acc>(data[h_high * W + w_high]) : Acc(0);

    // d(bilinear)/dh
    grad_h = -hw * v00 - lw * v01 + hw * v10 + lw * v11;
    // d(bilinear)/dw
    grad_w = -hh * v00 + hh * v01 - lh * v10 + lh * v11;
}

// ---------------------------------------------------------------------------
// Forward kernel: one thread per output element (N * C_out * H_out * W_out)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void dcn_forward_kernel(
    const T* __restrict__ input,
    const T* __restrict__ offset,
    const T* __restrict__ weight,
    const T* __restrict__ bias,      // may be nullptr
    const T* __restrict__ mask,      // may be nullptr
    T* __restrict__ output,
    int64_t N, int64_t C_in, int64_t H, int64_t W,
    int64_t C_out, int64_t kH, int64_t kW,
    int64_t H_out, int64_t W_out,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    bool use_mask)
{
    using Acc = DcnAcc<T>;
    int64_t total = N * C_out * H_out * W_out;
    int64_t channels_per_group = C_in / groups;
    int64_t out_channels_per_group = C_out / groups;
    int64_t channels_per_offset_group = C_in / offset_groups;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t oc = (idx / (W_out * H_out)) % C_out;
        int64_t n  = idx / (W_out * H_out * C_out);

        int64_t g = oc / out_channels_per_group;

        Acc sum = Acc(0);

        for (int64_t ic_local = 0; ic_local < channels_per_group; ++ic_local) {
            int64_t ic = g * channels_per_group + ic_local;
            int64_t og = ic / channels_per_offset_group;

            const T* input_plane = input + (n * C_in + ic) * H * W;
            const T* weight_plane = weight + (oc * channels_per_group + ic_local) * kH * kW;

            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t k_linear = kh * kW + kw;
                    int64_t offset_base = og * 2 * kH * kW;

                    Acc h_base = static_cast<Acc>(oh * stride_h - pad_h + kh * dil_h);
                    Acc w_base = static_cast<Acc>(ow * stride_w - pad_w + kw * dil_w);

                    int64_t off_h_chan = offset_base + 2 * k_linear;
                    int64_t off_w_chan = offset_base + 2 * k_linear + 1;

                    Acc h_off = static_cast<Acc>(
                        offset[(n * offset_groups * 2 * kH * kW + off_h_chan) * H_out * W_out +
                               oh * W_out + ow]);
                    Acc w_off = static_cast<Acc>(
                        offset[(n * offset_groups * 2 * kH * kW + off_w_chan) * H_out * W_out +
                               oh * W_out + ow]);

                    Acc h_loc = h_base + h_off;
                    Acc w_loc = w_base + w_off;

                    Acc val = dcn_bilinear_interpolate<Acc>(input_plane, H, W, h_loc, w_loc);

                    if (use_mask) {
                        int64_t mask_chan = og * kH * kW + k_linear;
                        Acc m = static_cast<Acc>(
                            mask[(n * offset_groups * kH * kW + mask_chan) * H_out * W_out +
                                 oh * W_out + ow]);
                        val *= m;
                    }

                    sum += val * static_cast<Acc>(weight_plane[k_linear]);
                }
            }
        }

        if (bias) {
            sum += static_cast<Acc>(bias[oc]);
        }

        output[idx] = static_cast<T>(sum);
    }
}

// ---------------------------------------------------------------------------
// Backward-input kernel: one thread per (n, oc, oh, ow).
// Accumulates grad_input, grad_offset, grad_mask via atomicAdd.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void dcn_backward_input_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ input,
    const T* __restrict__ offset,
    const T* __restrict__ weight,
    const T* __restrict__ mask,       // may be nullptr
    T* __restrict__ grad_input,
    T* __restrict__ grad_offset,
    T* __restrict__ grad_mask,        // may be nullptr
    int64_t N, int64_t C_in, int64_t H, int64_t W,
    int64_t C_out, int64_t kH, int64_t kW,
    int64_t H_out, int64_t W_out,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    bool use_mask)
{
    using Acc = DcnAcc<T>;
    int64_t total = N * C_out * H_out * W_out;
    int64_t channels_per_group = C_in / groups;
    int64_t out_channels_per_group = C_out / groups;
    int64_t channels_per_offset_group = C_in / offset_groups;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t oc = (idx / (W_out * H_out)) % C_out;
        int64_t n  = idx / (W_out * H_out * C_out);

        int64_t g = oc / out_channels_per_group;

        Acc grad_out_val = static_cast<Acc>(
            grad_output[(n * C_out + oc) * H_out * W_out + oh * W_out + ow]);

        for (int64_t ic_local = 0; ic_local < channels_per_group; ++ic_local) {
            int64_t ic = g * channels_per_group + ic_local;
            int64_t og = ic / channels_per_offset_group;

            const T* input_plane = input + (n * C_in + ic) * H * W;
            const T* weight_plane = weight + (oc * channels_per_group + ic_local) * kH * kW;

            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t k_linear = kh * kW + kw;
                    int64_t offset_base = og * 2 * kH * kW;

                    Acc h_base = static_cast<Acc>(oh * stride_h - pad_h + kh * dil_h);
                    Acc w_base = static_cast<Acc>(ow * stride_w - pad_w + kw * dil_w);

                    int64_t off_h_chan = offset_base + 2 * k_linear;
                    int64_t off_w_chan = offset_base + 2 * k_linear + 1;

                    int64_t off_h_idx = (n * offset_groups * 2 * kH * kW + off_h_chan) * H_out * W_out +
                                        oh * W_out + ow;
                    int64_t off_w_idx = (n * offset_groups * 2 * kH * kW + off_w_chan) * H_out * W_out +
                                        oh * W_out + ow;

                    Acc h_off = static_cast<Acc>(offset[off_h_idx]);
                    Acc w_off = static_cast<Acc>(offset[off_w_idx]);

                    Acc h_loc = h_base + h_off;
                    Acc w_loc = w_base + w_off;

                    Acc w_val = static_cast<Acc>(weight_plane[k_linear]);
                    Acc m_val = Acc(1);
                    if (use_mask) {
                        int64_t mask_chan = og * kH * kW + k_linear;
                        int64_t mask_idx = (n * offset_groups * kH * kW + mask_chan) * H_out * W_out +
                                           oh * W_out + ow;
                        m_val = static_cast<Acc>(mask[mask_idx]);
                    }

                    Acc top_grad = grad_out_val * w_val * m_val;

                    // --- grad_input: scatter through bilinear interpolation ---
                    if (h_loc > Acc(-1) && h_loc < static_cast<Acc>(H) &&
                        w_loc > Acc(-1) && w_loc < static_cast<Acc>(W)) {

                        int64_t h_low = static_cast<int64_t>(floor(h_loc));
                        int64_t w_low = static_cast<int64_t>(floor(w_loc));
                        int64_t h_high = h_low + 1;
                        int64_t w_high = w_low + 1;

                        Acc lh = h_loc - static_cast<Acc>(h_low);
                        Acc lw = w_loc - static_cast<Acc>(w_low);
                        Acc hh_val = Acc(1) - lh;
                        Acc hw_val = Acc(1) - lw;

                        T* gi_plane = grad_input + (n * C_in + ic) * H * W;
                        if (h_low >= 0 && h_low < H && w_low >= 0 && w_low < W)
                            atomicAdd(&gi_plane[h_low * W + w_low], static_cast<T>(hh_val * hw_val * top_grad));
                        if (h_low >= 0 && h_low < H && w_high >= 0 && w_high < W)
                            atomicAdd(&gi_plane[h_low * W + w_high], static_cast<T>(hh_val * lw * top_grad));
                        if (h_high >= 0 && h_high < H && w_low >= 0 && w_low < W)
                            atomicAdd(&gi_plane[h_high * W + w_low], static_cast<T>(lh * hw_val * top_grad));
                        if (h_high >= 0 && h_high < H && w_high >= 0 && w_high < W)
                            atomicAdd(&gi_plane[h_high * W + w_high], static_cast<T>(lh * lw * top_grad));
                    }

                    // --- grad_offset ---
                    Acc gh, gw;
                    dcn_bilinear_offset_gradient<Acc>(input_plane, H, W, h_loc, w_loc, gh, gw);
                    Acc off_grad_h = grad_out_val * w_val * m_val * gh;
                    Acc off_grad_w = grad_out_val * w_val * m_val * gw;
                    atomicAdd(&grad_offset[off_h_idx], static_cast<T>(off_grad_h));
                    atomicAdd(&grad_offset[off_w_idx], static_cast<T>(off_grad_w));

                    // --- grad_mask ---
                    if (use_mask && grad_mask) {
                        Acc interp_val = dcn_bilinear_interpolate<Acc>(input_plane, H, W, h_loc, w_loc);
                        Acc mask_grad = grad_out_val * w_val * interp_val;
                        int64_t mask_chan = og * kH * kW + k_linear;
                        int64_t mi = (n * offset_groups * kH * kW + mask_chan) * H_out * W_out +
                                     oh * W_out + ow;
                        atomicAdd(&grad_mask[mi], static_cast<T>(mask_grad));
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Backward-weight kernel: one thread per (oc, ic_local, kh, kw).
// Each thread loops over (n, oh, ow) to accumulate grad_weight.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void dcn_backward_weight_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ input,
    const T* __restrict__ offset,
    const T* __restrict__ mask,       // may be nullptr
    T* __restrict__ grad_weight,
    int64_t N, int64_t C_in, int64_t H, int64_t W,
    int64_t C_out, int64_t kH, int64_t kW,
    int64_t H_out, int64_t W_out,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    bool use_mask)
{
    using Acc = DcnAcc<T>;
    int64_t channels_per_group = C_in / groups;
    int64_t out_channels_per_group = C_out / groups;
    int64_t channels_per_offset_group = C_in / offset_groups;
    int64_t total = C_out * channels_per_group * kH * kW;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t kw_idx = idx % kW;
        int64_t kh_idx = (idx / kW) % kH;
        int64_t ic_local = (idx / (kW * kH)) % channels_per_group;
        int64_t oc = idx / (kW * kH * channels_per_group);

        int64_t g = oc / out_channels_per_group;
        int64_t ic = g * channels_per_group + ic_local;
        int64_t og = ic / channels_per_offset_group;
        int64_t k_linear = kh_idx * kW + kw_idx;
        int64_t offset_base = og * 2 * kH * kW;
        int64_t mask_base = og * kH * kW;

        Acc sum = Acc(0);

        for (int64_t n = 0; n < N; ++n) {
            const T* input_plane = input + (n * C_in + ic) * H * W;

            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    Acc h_base = static_cast<Acc>(oh * stride_h - pad_h + kh_idx * dil_h);
                    Acc w_base = static_cast<Acc>(ow * stride_w - pad_w + kw_idx * dil_w);

                    int64_t off_h_chan = offset_base + 2 * k_linear;
                    int64_t off_w_chan = offset_base + 2 * k_linear + 1;

                    Acc h_off = static_cast<Acc>(
                        offset[(n * offset_groups * 2 * kH * kW + off_h_chan) * H_out * W_out +
                               oh * W_out + ow]);
                    Acc w_off = static_cast<Acc>(
                        offset[(n * offset_groups * 2 * kH * kW + off_w_chan) * H_out * W_out +
                               oh * W_out + ow]);

                    Acc h_loc = h_base + h_off;
                    Acc w_loc = w_base + w_off;

                    Acc val = dcn_bilinear_interpolate<Acc>(input_plane, H, W, h_loc, w_loc);

                    if (use_mask) {
                        int64_t mask_chan = mask_base + k_linear;
                        Acc m = static_cast<Acc>(
                            mask[(n * offset_groups * kH * kW + mask_chan) * H_out * W_out +
                                 oh * W_out + ow]);
                        val *= m;
                    }

                    Acc go = static_cast<Acc>(
                        grad_output[(n * C_out + oc) * H_out * W_out + oh * W_out + ow]);
                    sum += go * val;
                }
            }
        }

        grad_weight[(oc * channels_per_group + ic_local) * kH * kW + k_linear] = static_cast<T>(sum);
    }
}

// ==============================================================================
// Host wrappers
// ==============================================================================

auto deformable_conv2d_forward_kernel(
    const Tensor& input, const Tensor& offset, const Tensor& weight,
    const Tensor& bias, const Tensor& mask,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    hipStream_t stream) -> Tensor {

    auto ishape = input.shape();
    auto wshape = weight.shape();
    int64_t N = ishape[0], C_in = ishape[1], H = ishape[2], W = ishape[3];
    int64_t C_out = wshape[0], kH = wshape[2], kW = wshape[3];
    int64_t H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    int64_t W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;

    bool use_mask = mask.numel() > 0;
    Tensor output({N, C_out, H_out, W_out}, input.dtype(), input.device());

    int64_t total = N * C_out * H_out * W_out;
    // Empty output: a zero-grid launch is rejected by HIP; return as-is.
    if (total == 0) return output;
    dim3 grid, block;
    compute_launch_config_1d(total, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(dcn_forward_kernel<float>,
            grid, block, 0, stream,
            input.data<float>(), offset.data<float>(), weight.data<float>(),
            bias.numel() > 0 ? bias.data<float>() : nullptr,
            use_mask ? mask.data<float>() : nullptr,
            output.data<float>(),
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, use_mask);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(dcn_forward_kernel<double>,
            grid, block, 0, stream,
            input.data<double>(), offset.data<double>(), weight.data<double>(),
            bias.numel() > 0 ? bias.data<double>() : nullptr,
            use_mask ? mask.data<double>() : nullptr,
            output.data<double>(),
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, use_mask);
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        // No native F16/BF16 kernel; run in Float32 and cast back.
        auto input_f32 = input.to(DType::Float32);
        auto offset_f32 = offset.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        auto bias_f32 = bias.numel() > 0 ? bias.to(DType::Float32) : bias;
        auto mask_f32 = mask.numel() > 0 ? mask.to(DType::Float32) : mask;
        auto out_f32 = deformable_conv2d_forward_kernel(
            input_f32, offset_f32, weight_f32, bias_f32, mask_f32,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, stream);
        return out_f32.to(input.dtype());
    } else {
        throw std::runtime_error("deformable_conv2d_forward: unsupported dtype (requires Float32 or Float64)");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

auto deformable_conv2d_backward_input_kernel(
    const Tensor& grad_output, const Tensor& input, const Tensor& offset,
    const Tensor& weight, const Tensor& mask,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    hipStream_t stream) -> std::vector<Tensor> {

    // F16/BF16 widen-narrow: compute in Float32, cast back.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto results = deformable_conv2d_backward_input_kernel(
            grad_output.to(DType::Float32), input.to(DType::Float32),
            offset.to(DType::Float32), weight.to(DType::Float32),
            mask.numel() > 0 ? mask.to(DType::Float32) : mask,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, stream);
        std::vector<Tensor> narrowed;
        narrowed.reserve(results.size());
        for (auto& r : results) narrowed.push_back(r.to(orig));
        return narrowed;
    }

    auto ishape = input.shape();
    auto wshape = weight.shape();
    auto oshape = offset.shape();
    int64_t N = ishape[0], C_in = ishape[1], H = ishape[2], W = ishape[3];
    int64_t C_out = wshape[0], kH = wshape[2], kW = wshape[3];
    int64_t H_out = grad_output.shape()[2], W_out = grad_output.shape()[3];

    bool use_mask = mask.numel() > 0;

    // Allocate gradient tensors
    Tensor grad_input(std::vector<int64_t>(ishape.begin(), ishape.end()), input.dtype(), input.device());
    Tensor grad_offset(std::vector<int64_t>(oshape.begin(), oshape.end()), input.dtype(), input.device());
    Tensor grad_mask;
    if (use_mask) {
        auto ms = mask.shape();
        grad_mask = Tensor(std::vector<int64_t>(ms.begin(), ms.end()), input.dtype(), input.device());
    }

    // Zero-initialize on device
    size_t elem_size = (input.dtype() == DType::Float64) ? sizeof(double) : sizeof(float);
    HIP_CHECK(hipMemsetAsync(grad_input.data_ptr(), 0, grad_input.numel() * elem_size, stream));
    HIP_CHECK(hipMemsetAsync(grad_offset.data_ptr(), 0, grad_offset.numel() * elem_size, stream));
    if (use_mask) {
        HIP_CHECK(hipMemsetAsync(grad_mask.data_ptr(), 0, grad_mask.numel() * elem_size, stream));
    }

    int64_t total = N * C_out * H_out * W_out;
    dim3 grid, block;
    compute_launch_config_1d(total, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(dcn_backward_input_kernel<float>,
            grid, block, 0, stream,
            grad_output.data<float>(), input.data<float>(), offset.data<float>(),
            weight.data<float>(),
            use_mask ? mask.data<float>() : nullptr,
            grad_input.data<float>(), grad_offset.data<float>(),
            use_mask ? grad_mask.data<float>() : nullptr,
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, use_mask);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(dcn_backward_input_kernel<double>,
            grid, block, 0, stream,
            grad_output.data<double>(), input.data<double>(), offset.data<double>(),
            weight.data<double>(),
            use_mask ? mask.data<double>() : nullptr,
            grad_input.data<double>(), grad_offset.data<double>(),
            use_mask ? grad_mask.data<double>() : nullptr,
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, use_mask);
    } else {
        throw std::runtime_error("deformable_conv2d_backward_input: unsupported dtype (requires Float32 or Float64)");
    }

    HIP_CHECK(hipGetLastError());

    if (use_mask) {
        return {grad_input, grad_offset, grad_mask};
    }
    return {grad_input, grad_offset};
}

auto deformable_conv2d_backward_weight_kernel(
    const Tensor& grad_output, const Tensor& input, const Tensor& offset,
    const Tensor& mask,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups, int64_t offset_groups,
    const std::vector<int64_t>& weight_shape,
    hipStream_t stream) -> Tensor {

    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto r = deformable_conv2d_backward_weight_kernel(
            grad_output.to(DType::Float32), input.to(DType::Float32),
            offset.to(DType::Float32),
            mask.numel() > 0 ? mask.to(DType::Float32) : mask,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, weight_shape, stream);
        return r.to(orig);
    }

    auto ishape = input.shape();
    int64_t N = ishape[0], C_in = ishape[1], H = ishape[2], W = ishape[3];
    int64_t C_out = weight_shape[0], kH = weight_shape[2], kW = weight_shape[3];
    int64_t H_out = grad_output.shape()[2], W_out = grad_output.shape()[3];

    bool use_mask = mask.numel() > 0;
    int64_t channels_per_group = C_in / groups;

    Tensor grad_weight(weight_shape, input.dtype(), input.device());

    // grad_weight is written directly (one thread per weight element), no need
    // for zero-init since each thread writes its full accumulated sum.

    int64_t total = C_out * channels_per_group * kH * kW;
    dim3 grid, block;
    compute_launch_config_1d(total, grid, block);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(dcn_backward_weight_kernel<float>,
            grid, block, 0, stream,
            grad_output.data<float>(), input.data<float>(), offset.data<float>(),
            use_mask ? mask.data<float>() : nullptr,
            grad_weight.data<float>(),
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, use_mask);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(dcn_backward_weight_kernel<double>,
            grid, block, 0, stream,
            grad_output.data<double>(), input.data<double>(), offset.data<double>(),
            use_mask ? mask.data<double>() : nullptr,
            grad_weight.data<double>(),
            N, C_in, H, W, C_out, kH, kW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, use_mask);
    } else {
        throw std::runtime_error("deformable_conv2d_backward_weight: unsupported dtype (requires Float32 or Float64)");
    }

    HIP_CHECK(hipGetLastError());
    return grad_weight;
}

// ============================================================================
// Depthwise Conv1d / Conv3d (forward, groups == channels).
//   Conv1d: input [N,C,1,L], weight [C,1,1,kL], output [N,C,1,L_out].
//   Conv3d: input [N,C,D,H,W], weight [C,1,kD,kH,kW], output [N,C,Do,Ho,Wo].
// Float32/Float64 native; Float16/BFloat16 widen to Float32. Backward is
// autograd-composed in the NN layer.
// ============================================================================
template <typename T>
__global__ void depthwise_conv1d_fwd_kernel(
    const T* __restrict__ in, const T* __restrict__ w, const T* __restrict__ bias,
    T* __restrict__ out, int64_t N, int64_t C, int64_t L, int64_t kL, int64_t Lo,
    int64_t stride, int64_t pad, int64_t dil) {
    // int64_t indices/counts/offsets: element counts (N*C*Lo) and derived input
    // offsets can exceed 2^31 for large tensors; grid-stride loop covers any size.
    int64_t total = N * C * Lo;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t ol = idx % Lo;
        int64_t c  = (idx / Lo) % C;
        int64_t n  = idx / (C * Lo);
        const T* in_nc = in + (n * C + c) * L;
        const T* w_c   = w + c * kL;
        T acc = bias ? bias[c] : T(0);
        for (int64_t k = 0; k < kL; ++k) {
            int64_t il = ol * stride - pad + k * dil;
            if (il >= 0 && il < L) acc += in_nc[il] * w_c[k];
        }
        out[(n * C + c) * Lo + ol] = acc;
    }
}

template <typename T>
__global__ void depthwise_conv3d_fwd_kernel(
    const T* __restrict__ in, const T* __restrict__ w, const T* __restrict__ bias,
    T* __restrict__ out, int64_t N, int64_t C, int64_t Di, int64_t Hi, int64_t Wi,
    int64_t kD, int64_t kH, int64_t kW, int64_t Do, int64_t Ho, int64_t Wo,
    int64_t sD, int64_t sH, int64_t sW, int64_t pD, int64_t pH, int64_t pW,
    int64_t dD, int64_t dH, int64_t dW) {
    int64_t total = N * C * Do * Ho * Wo;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int64_t ow = idx % Wo;
        int64_t oh = (idx / Wo) % Ho;
        int64_t od = (idx / (Wo * Ho)) % Do;
        int64_t c  = (idx / (Wo * Ho * Do)) % C;
        int64_t n  = idx / (C * Do * Ho * Wo);
        const T* in_nc = in + (n * C + c) * Di * Hi * Wi;
        const T* w_c   = w + c * kD * kH * kW;
        T acc = bias ? bias[c] : T(0);
        for (int64_t kd = 0; kd < kD; ++kd) {
            int64_t id = od * sD - pD + kd * dD;
            if (id < 0 || id >= Di) continue;
            for (int64_t kh = 0; kh < kH; ++kh) {
                int64_t ih = oh * sH - pH + kh * dH;
                if (ih < 0 || ih >= Hi) continue;
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t iw = ow * sW - pW + kw * dW;
                    if (iw < 0 || iw >= Wi) continue;
                    acc += in_nc[(id * Hi + ih) * Wi + iw] * w_c[(kd * kH + kh) * kW + kw];
                }
            }
        }
        out[(((n * C + c) * Do + od) * Ho + oh) * Wo + ow] = acc;
    }
}

auto depthwise_conv1d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                             int64_t stride, int64_t padding, int64_t dilation,
                             hipStream_t stream) -> Tensor {
    auto is = input.shape();
    auto ws = weight.shape();
    int64_t N = is[0], C = is[1], L = is[3], kL = ws[3];
    int64_t Lo = (L + 2 * padding - dilation * (kL - 1) - 1) / stride + 1;
    if (Lo <= 0) throw std::runtime_error("depthwise_conv1d (ROCm): non-positive output length");

    auto run = [&](DType dt) {
        Tensor in = input.dtype() == dt ? input.contiguous() : input.to(dt);
        Tensor w  = weight.dtype() == dt ? weight.contiguous() : weight.to(dt);
        Tensor b; const void* bptr = nullptr;
        if (bias) { b = bias->dtype() == dt ? bias->contiguous() : bias->to(dt); bptr = b.data_ptr(); }
        Tensor out({N, C, 1, Lo}, dt, input.device());
        int64_t total = N * C * Lo;
        int64_t want_blocks = (total + 255) / 256;
        int blocks = static_cast<int>(want_blocks < 65535 ? want_blocks : 65535);
        if (dt == DType::Float64) {
            hipLaunchKernelGGL(depthwise_conv1d_fwd_kernel<double>, dim3(blocks), dim3(256), 0, stream,
                in.data<double>(), w.data<double>(), (const double*)bptr, out.data<double>(),
                N, C, L, kL, Lo, stride, padding, dilation);
        } else {
            hipLaunchKernelGGL(depthwise_conv1d_fwd_kernel<float>, dim3(blocks), dim3(256), 0, stream,
                in.data<float>(), w.data<float>(), (const float*)bptr, out.data<float>(),
                N, C, L, kL, Lo, stride, padding, dilation);
        }
        HIP_CHECK(hipGetLastError());
        return out;
    };

    DType in_dt = input.dtype();
    if (in_dt == DType::Float64) return run(DType::Float64);
    if (in_dt == DType::Float32) return run(DType::Float32);
    if (in_dt == DType::Float16 || in_dt == DType::BFloat16) return run(DType::Float32).to(in_dt);
    throw std::runtime_error("depthwise_conv1d (ROCm): unsupported dtype");
}

auto depthwise_conv3d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                             int64_t sD, int64_t sH, int64_t sW,
                             int64_t pD, int64_t pH, int64_t pW,
                             int64_t dD, int64_t dH, int64_t dW,
                             hipStream_t stream) -> Tensor {
    auto is = input.shape();
    auto ws = weight.shape();
    int64_t N = is[0], C = is[1], Di = is[2], Hi = is[3], Wi = is[4];
    int64_t kD = ws[2], kH = ws[3], kW = ws[4];
    int64_t Do = (Di + 2 * pD - dD * (kD - 1) - 1) / sD + 1;
    int64_t Ho = (Hi + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    int64_t Wo = (Wi + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    if (Do <= 0 || Ho <= 0 || Wo <= 0) throw std::runtime_error("depthwise_conv3d (ROCm): non-positive output size");

    auto run = [&](DType dt) {
        Tensor in = input.dtype() == dt ? input.contiguous() : input.to(dt);
        Tensor w  = weight.dtype() == dt ? weight.contiguous() : weight.to(dt);
        Tensor b; const void* bptr = nullptr;
        if (bias) { b = bias->dtype() == dt ? bias->contiguous() : bias->to(dt); bptr = b.data_ptr(); }
        Tensor out({N, C, Do, Ho, Wo}, dt, input.device());
        int64_t total = N * C * Do * Ho * Wo;
        int64_t want_blocks = (total + 255) / 256;
        int blocks = static_cast<int>(want_blocks < 65535 ? want_blocks : 65535);
        if (dt == DType::Float64) {
            hipLaunchKernelGGL(depthwise_conv3d_fwd_kernel<double>, dim3(blocks), dim3(256), 0, stream,
                in.data<double>(), w.data<double>(), (const double*)bptr, out.data<double>(),
                N, C, Di, Hi, Wi, kD, kH, kW, Do, Ho, Wo,
                sD, sH, sW, pD, pH, pW, dD, dH, dW);
        } else {
            hipLaunchKernelGGL(depthwise_conv3d_fwd_kernel<float>, dim3(blocks), dim3(256), 0, stream,
                in.data<float>(), w.data<float>(), (const float*)bptr, out.data<float>(),
                N, C, Di, Hi, Wi, kD, kH, kW, Do, Ho, Wo,
                sD, sH, sW, pD, pH, pW, dD, dH, dW);
        }
        HIP_CHECK(hipGetLastError());
        return out;
    };

    DType in_dt = input.dtype();
    if (in_dt == DType::Float64) return run(DType::Float64);
    if (in_dt == DType::Float32) return run(DType::Float32);
    if (in_dt == DType::Float16 || in_dt == DType::BFloat16) return run(DType::Float32).to(in_dt);
    throw std::runtime_error("depthwise_conv3d (ROCm): unsupported dtype");
}

} // namespace rocm
} // namespace tenzor
