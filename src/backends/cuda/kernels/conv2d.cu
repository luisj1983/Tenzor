#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace tenzor {
namespace cuda {

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuBLAS error: ") + std::to_string(status)); \
    } \
} while(0)

// ============================================================================
// Kernel Launch Helpers
// ============================================================================

inline void compute_launch_config_1d(int64_t n, dim3& grid, dim3& block) {
    const int block_size = 256;
    block = dim3(block_size, 1, 1);
    grid = dim3((n + block_size - 1) / block_size, 1, 1);
}

inline void compute_launch_config_2d(int64_t rows, int64_t cols, dim3& grid, dim3& block) {
    const int block_x = 16;
    const int block_y = 16;
    block = dim3(block_x, block_y, 1);
    grid = dim3((cols + block_x - 1) / block_x, (rows + block_y - 1) / block_y, 1);
}

#define CUDA_KERNEL_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

// ============================================================================
// Helper Functions
// ============================================================================

// Calculate output size for convolution
__host__ __device__ inline int64_t calculate_output_size(int64_t input_size, int64_t kernel_size,
                                                          int64_t stride, int64_t padding, int64_t dilation) {
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
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

    CUDA_KERNEL_LOOP(idx, total_elements) {
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
// col2im CUDA Kernel
// ============================================================================

// col2im kernel: Reverse of im2col for gradient computation
// Input: (batch * out_h * out_w, kernel_h * kernel_w * in_channels)
// Output: (batch, in_channels, height, width)
// Note: This accumulates gradients for overlapping regions
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
    int64_t total_elements = batch * out_h * out_w * channels * kernel_h * kernel_w;

    CUDA_KERNEL_LOOP(idx, total_elements) {
        // Decode flat index to (b, oh, ow, c, kh, kw)
        int64_t temp = idx;
        int64_t kw = temp % kernel_w; temp /= kernel_w;
        int64_t kh = temp % kernel_h; temp /= kernel_h;
        int64_t c = temp % channels; temp /= channels;
        int64_t ow = temp % out_w; temp /= out_w;
        int64_t oh = temp % out_h; temp /= out_h;
        int64_t b = temp;

        // Calculate input position
        int64_t ih = oh * stride - padding + kh * dilation;
        int64_t iw = ow * stride - padding + kw * dilation;

        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            // Col index
            int64_t col_row = b * out_h * out_w + oh * out_w + ow;
            int64_t col_col = c * kernel_h * kernel_w + kh * kernel_w + kw;
            int64_t col_idx = col_row * (channels * kernel_h * kernel_w) + col_col;

            // Output index
            int64_t output_idx = b * (channels * height * width) +
                                c * (height * width) +
                                ih * width + iw;

            // Atomic add for gradient accumulation (multiple kernel positions map to same output)
            atomicAdd(&output[output_idx], col[col_idx]);
        }
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
    CUDA_KERNEL_LOOP(idx, n) {
        int64_t c = (idx / spatial_size) % channels;
        output[idx] += bias[c];
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

// ============================================================================
// Conv2d Forward GPU Implementation
// ============================================================================

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

    // Calculate output dimensions
    int64_t out_h = calculate_output_size(height, kernel_h, stride, padding, dilation);
    int64_t out_w = calculate_output_size(width, kernel_w, stride, padding, dilation);

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, out_channels, out_h, out_w};
    Tensor output(output_shape, input.dtype(), input.device());

    // Initialize output to zeros
    CUDA_CHECK(cudaMemsetAsync(output.data<float>(), 0, output.numel() * sizeof(float), stream));

    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

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
        float* col_buffer;
        CUDA_CHECK(cudaMalloc(&col_buffer, col_rows * col_cols * sizeof(float)));

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
        CUDA_CHECK(cudaGetLastError());

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
        float* output_ptr = output.data<float>() + out_start * out_h * out_w;

        // cuBLAS uses column-major ordering
        // We want: C = A @ B^T where A is row-major (M, K), B is row-major (N, K)
        // In column-major view: C^T = B @ A^T
        // So we compute: C^T = B @ A^T, which means C = (B @ A^T)^T = A @ B^T
        CUBLAS_CHECK(cublasSgemm(
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
            output_ptr,     // C (M, N) in row-major = (N, M) in col-major
            N               // leading dimension of C
        ));

        // Free col buffer
        CUDA_CHECK(cudaFree(col_buffer));
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
        CUDA_CHECK(cudaGetLastError());
    }

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(cublas_handle));

    return output;
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

    // Initialize outputs
    Tensor grad_input({batch, in_channels, height, width}, input.dtype(), input.device());
    Tensor grad_weight({out_channels, in_channels_per_group, kernel_h, kernel_w}, weight.dtype(), weight.device());
    Tensor grad_bias({out_channels}, weight.dtype(), weight.device());

    if (compute_grad_input) {
        CUDA_CHECK(cudaMemsetAsync(grad_input.data<float>(), 0, grad_input.numel() * sizeof(float), stream));
    }
    if (compute_grad_weight) {
        CUDA_CHECK(cudaMemsetAsync(grad_weight.data<float>(), 0, grad_weight.numel() * sizeof(float), stream));
    }
    if (compute_grad_bias) {
        CUDA_CHECK(cudaMemsetAsync(grad_bias.data<float>(), 0, grad_bias.numel() * sizeof(float), stream));
    }

    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    int64_t out_channels_per_group = out_channels / groups;
    int64_t col_rows = batch * out_h * out_w;
    int64_t col_cols = in_channels_per_group * kernel_h * kernel_w;

    for (int64_t g = 0; g < groups; ++g) {
        int64_t in_start = g * in_channels_per_group;
        int64_t out_start = g * out_channels_per_group;

        // Gradient w.r.t input
        if (compute_grad_input) {
            // Allocate col buffer
            float* grad_col;
            CUDA_CHECK(cudaMalloc(&grad_col, col_rows * col_cols * sizeof(float)));

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

            CUBLAS_CHECK(cublasSgemm(
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
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaFree(grad_col));
        }

        // Gradient w.r.t weight
        if (compute_grad_weight) {
            // Apply im2col to input
            float* input_col;
            CUDA_CHECK(cudaMalloc(&input_col, col_rows * col_cols * sizeof(float)));

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
            CUDA_CHECK(cudaGetLastError());

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

            CUBLAS_CHECK(cublasSgemm(
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

            CUDA_CHECK(cudaFree(input_col));
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
        CUDA_CHECK(cudaGetLastError());
    }

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(cublas_handle));

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

} // namespace cuda
} // namespace tenzor
