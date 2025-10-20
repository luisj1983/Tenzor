#ifdef TENZOR_CUDA_AVAILABLE

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include "tenzor/core/tensor.hpp"
#include <stdexcept>
#include <cmath>

namespace tenzor {
namespace cuda {

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + cudaGetErrorString(error) \
            ); \
        } \
    } while(0)

// ==============================================================================
// Fused Linear + ReLU CUDA Kernel
// ==============================================================================

/**
 * @brief CUDA kernel for fused linear + ReLU
 *
 * Computes: out = max(0, input @ weight.T + bias)
 * Uses grid-stride loop for large tensors.
 */
template<typename T>
__global__ void fused_linear_relu_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    T* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    bool has_bias
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * out_features;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t idx = tid; idx < total_elements; idx += stride) {
        int64_t b = idx / out_features;
        int64_t o = idx % out_features;

        T sum = 0;
        for (int64_t i = 0; i < in_features; ++i) {
            sum += input[b * in_features + i] * weight[o * in_features + i];
        }

        if (has_bias) {
            sum += bias[o];
        }

        // ReLU
        output[idx] = (sum > T(0)) ? sum : T(0);
    }
}

/**
 * @brief Fused linear + ReLU host function
 */
auto fused_linear_relu_cuda(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias
) -> Tensor {
    // Flatten input to 2D
    auto input_shape = input.shape();
    int64_t batch_size = 1;
    for (size_t i = 0; i < input_shape.size() - 1; ++i) {
        batch_size *= input_shape[i];
    }
    int64_t in_features = input_shape[input_shape.size() - 1];
    int64_t out_features = weight.shape()[0];

    // Create output tensor
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end() - 1);
    output_shape.push_back(out_features);
    Tensor output = zeros(output_shape, input.dtype(), input.device());

    // Launch kernel
    int64_t total_elements = batch_size * out_features;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = std::min(blocks, static_cast<int64_t>(65535));

    if (input.dtype() == DType::Float32) {
        const float* bias_ptr = bias ? bias->data<float>() : nullptr;
        fused_linear_relu_kernel<<<blocks, threads>>>(
            input.data<float>(),
            weight.data<float>(),
            bias_ptr,
            output.data<float>(),
            batch_size,
            in_features,
            out_features,
            bias != nullptr
        );
    } else {
        throw std::runtime_error("fused_linear_relu_cuda: Only Float32 supported");
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

// ==============================================================================
// Fused BatchNorm + ReLU CUDA Kernel
// ==============================================================================

template<typename T>
__global__ void fused_batchnorm_relu_kernel(
    const T* input,
    const T* mean,
    const T* var,
    const T* gamma,
    const T* beta,
    T* output,
    int64_t batch_size,
    int64_t num_features,
    int64_t spatial_size,
    T eps
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * num_features * spatial_size;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t idx = tid; idx < total_elements; idx += stride) {
        int64_t s = idx % spatial_size;
        int64_t c = (idx / spatial_size) % num_features;
        int64_t n = idx / (spatial_size * num_features);

        T normalized = (input[idx] - mean[c]) * rsqrtf(var[c] + eps);
        T scaled = normalized * gamma[c] + beta[c];

        // ReLU
        output[idx] = (scaled > T(0)) ? scaled : T(0);
    }
}

auto fused_batchnorm_relu_cuda(
    const Tensor& input,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& weight,
    const Tensor& bias,
    float eps
) -> Tensor {
    int64_t batch_size = input.shape()[0];
    int64_t num_features = input.shape()[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < input.shape().size(); ++i) {
        spatial_size *= input.shape()[i];
    }

    Tensor output = zeros(input.shape(), input.dtype(), input.device());

    int64_t total_elements = input.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = std::min(blocks, static_cast<int64_t>(65535));

    if (input.dtype() == DType::Float32) {
        fused_batchnorm_relu_kernel<<<blocks, threads>>>(
            input.data<float>(),
            running_mean.data<float>(),
            running_var.data<float>(),
            weight.data<float>(),
            bias.data<float>(),
            output.data<float>(),
            batch_size,
            num_features,
            spatial_size,
            eps
        );
    } else {
        throw std::runtime_error("fused_batchnorm_relu_cuda: Only Float32 supported");
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

// ==============================================================================
// Fused Softmax + CrossEntropy CUDA Kernel
// ==============================================================================

template<typename T, int BLOCK_SIZE>
__global__ void fused_softmax_cross_entropy_kernel(
    const T* logits,
    const int64_t* targets,
    T* losses,
    int64_t batch_size,
    int64_t num_classes
) {
    int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const T* row = logits + b * num_classes;
    int64_t target = targets[b];

    // Shared memory for reduction
    __shared__ T shared_data[BLOCK_SIZE];

    // Find max (for numerical stability)
    T max_val = -INFINITY;
    for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
        max_val = fmaxf(max_val, row[i]);
    }

    // Block-wide max reduction
    shared_data[threadIdx.x] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] = fmaxf(shared_data[threadIdx.x], shared_data[threadIdx.x + s]);
        }
        __syncthreads();
    }

    T global_max = shared_data[0];
    __syncthreads();

    // Compute sum(exp(x - max))
    T sum_exp = 0;
    for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
        sum_exp += expf(row[i] - global_max);
    }

    shared_data[threadIdx.x] = sum_exp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Compute loss
    if (threadIdx.x == 0) {
        T log_sum_exp = logf(shared_data[0]) + global_max;
        losses[b] = log_sum_exp - row[target];
    }
}

auto fused_softmax_cross_entropy_cuda(
    const Tensor& logits,
    const Tensor& targets,
    const std::string& reduction
) -> Tensor {
    int64_t batch_size = logits.shape()[0];
    int64_t num_classes = logits.shape()[1];

    Tensor losses = zeros({batch_size}, logits.dtype(), logits.device());

    constexpr int BLOCK_SIZE = 256;
    int blocks = batch_size;

    if (logits.dtype() == DType::Float32) {
        fused_softmax_cross_entropy_kernel<float, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
            logits.data<float>(),
            targets.data<int64_t>(),
            losses.data<float>(),
            batch_size,
            num_classes
        );
    } else {
        throw std::runtime_error("fused_softmax_cross_entropy_cuda: Only Float32 supported");
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Apply reduction
    if (reduction == "mean") {
        return mean(losses);
    } else if (reduction == "sum") {
        return sum(losses);
    } else {
        return losses;
    }
}

// ==============================================================================
// Fused Add + ReLU CUDA Kernel
// ==============================================================================

template<typename T>
__global__ void fused_add_relu_kernel(
    const T* a,
    const T* b,
    T* output,
    int64_t n
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        T sum = a[i] + b[i];
        output[i] = (sum > T(0)) ? sum : T(0);
    }
}

auto fused_add_relu_cuda(const Tensor& a, const Tensor& b) -> Tensor {
    Tensor result = zeros(a.shape(), a.dtype(), a.device());

    int64_t n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = std::min(blocks, static_cast<int64_t>(65535));

    if (a.dtype() == DType::Float32) {
        fused_add_relu_kernel<<<blocks, threads>>>(
            a.data<float>(),
            b.data<float>(),
            result.data<float>(),
            n
        );
    } else {
        throw std::runtime_error("fused_add_relu_cuda: Only Float32 supported");
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return result;
}

// ==============================================================================
// Fused GELU CUDA Kernel
// ==============================================================================

template<typename T>
__global__ void fused_gelu_kernel(
    const T* input,
    T* output,
    int64_t n
) {
    constexpr T sqrt_2_over_pi = 0.7978845608f;
    constexpr T coeff = 0.044715f;

    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        T x = input[i];
        T x_cubed = x * x * x;
        T inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        T tanh_val = tanhf(inner);
        output[i] = T(0.5) * x * (T(1.0) + tanh_val);
    }
}

auto fused_gelu_cuda(const Tensor& input) -> Tensor {
    Tensor output = zeros(input.shape(), input.dtype(), input.device());

    int64_t n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = std::min(blocks, static_cast<int64_t>(65535));

    if (input.dtype() == DType::Float32) {
        fused_gelu_kernel<<<blocks, threads>>>(
            input.data<float>(),
            output.data<float>(),
            n
        );
    } else {
        throw std::runtime_error("fused_gelu_cuda: Only Float32 supported");
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

// ==============================================================================
// Fused Layer Norm CUDA Kernel
// ==============================================================================

template<typename T, int BLOCK_SIZE>
__global__ void fused_layer_norm_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    T* output,
    int64_t batch_size,
    int64_t norm_size,
    T eps
) {
    int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const T* batch_in = input + b * norm_size;
    T* batch_out = output + b * norm_size;

    __shared__ T shared_data[BLOCK_SIZE];

    // Compute mean
    T sum = 0;
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        sum += batch_in[i];
    }

    shared_data[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    T mean = shared_data[0] / norm_size;
    __syncthreads();

    // Compute variance
    T var_sum = 0;
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        T diff = batch_in[i] - mean;
        var_sum += diff * diff;
    }

    shared_data[threadIdx.x] = var_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    T variance = shared_data[0] / norm_size;
    T inv_std = rsqrtf(variance + eps);

    // Normalize and scale
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        T normalized = (batch_in[i] - mean) * inv_std;
        batch_out[i] = normalized * weight[i] + bias[i];
    }
}

auto fused_layer_norm_cuda(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    float eps
) -> Tensor {
    int64_t norm_size = 1;
    for (auto dim : normalized_shape) {
        norm_size *= dim;
    }

    int64_t batch_size = input.numel() / norm_size;

    Tensor output = zeros(input.shape(), input.dtype(), input.device());

    constexpr int BLOCK_SIZE = 256;
    int blocks = batch_size;

    if (input.dtype() == DType::Float32) {
        fused_layer_norm_kernel<float, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
            input.data<float>(),
            weight.data<float>(),
            bias.data<float>(),
            output.data<float>(),
            batch_size,
            norm_size,
            eps
        );
    } else {
        throw std::runtime_error("fused_layer_norm_cuda: Only Float32 supported");
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

// ==============================================================================
// Fused Conv2D + BatchNorm + ReLU CUDA Kernel
// ==============================================================================

/**
 * @brief Fused Conv2D + BatchNorm + ReLU kernel
 *
 * Combines convolution, batch normalization, and ReLU into single kernel.
 * Highly optimized for training and inference pipelines.
 */
template<typename T>
__global__ void fused_conv2d_bn_relu_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    const T* bn_mean,
    const T* bn_var,
    const T* bn_gamma,
    const T* bn_beta,
    T* output,
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_h,
    int64_t out_w,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t padding,
    T eps,
    bool has_bias
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * out_channels * out_h * out_w;
    int64_t stride_loop = blockDim.x * gridDim.x;

    for (int64_t idx = tid; idx < total_elements; idx += stride_loop) {
        // Decode output position
        int64_t w_out = idx % out_w;
        int64_t h_out = (idx / out_w) % out_h;
        int64_t c_out = (idx / (out_w * out_h)) % out_channels;
        int64_t n = idx / (out_w * out_h * out_channels);

        // Compute convolution
        T conv_sum = 0;
        for (int64_t c_in = 0; c_in < in_channels; ++c_in) {
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t h_in = h_out * stride - padding + kh;
                    int64_t w_in = w_out * stride - padding + kw;

                    if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                        int64_t input_idx = ((n * in_channels + c_in) * in_h + h_in) * in_w + w_in;
                        int64_t weight_idx = ((c_out * in_channels + c_in) * kernel_h + kh) * kernel_w + kw;
                        conv_sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        // Add bias if present
        if (has_bias) {
            conv_sum += bias[c_out];
        }

        // Apply batch normalization
        T normalized = (conv_sum - bn_mean[c_out]) * rsqrtf(bn_var[c_out] + eps);
        T bn_out = normalized * bn_gamma[c_out] + bn_beta[c_out];

        // Apply ReLU
        output[idx] = (bn_out > T(0)) ? bn_out : T(0);
    }
}

auto fused_conv2d_bn_relu_cuda(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    const Tensor& bn_mean,
    const Tensor& bn_var,
    const Tensor& bn_gamma,
    const Tensor& bn_beta,
    int64_t stride,
    int64_t padding,
    float eps
) -> Tensor {
    int64_t batch_size = input.shape()[0];
    int64_t in_channels = input.shape()[1];
    int64_t in_h = input.shape()[2];
    int64_t in_w = input.shape()[3];

    int64_t out_channels = weight.shape()[0];
    int64_t kernel_h = weight.shape()[2];
    int64_t kernel_w = weight.shape()[3];

    int64_t out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    int64_t out_w = (in_w + 2 * padding - kernel_w) / stride + 1;

    Tensor output = zeros({batch_size, out_channels, out_h, out_w}, input.dtype(), input.device());

    int64_t total_elements = batch_size * out_channels * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = std::min(blocks, static_cast<int64_t>(65535));

    if (input.dtype() == DType::Float32) {
        const float* bias_ptr = bias ? bias->data<float>() : nullptr;
        fused_conv2d_bn_relu_kernel<<<blocks, threads>>>(
            input.data<float>(),
            weight.data<float>(),
            bias_ptr,
            bn_mean.data<float>(),
            bn_var.data<float>(),
            bn_gamma.data<float>(),
            bn_beta.data<float>(),
            output.data<float>(),
            batch_size,
            in_channels,
            out_channels,
            in_h,
            in_w,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
            eps,
            bias != nullptr
        );
    } else {
        throw std::runtime_error("fused_conv2d_bn_relu_cuda: Only Float32 supported");
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

// ==============================================================================
// Fused MatMul + Add (Bias) CUDA Kernel
// ==============================================================================

/**
 * @brief Fused matrix multiplication with bias addition
 *
 * Computes: C = A @ B + bias
 * Optimized for batch processing with tiling.
 */
template<typename T, int TILE_SIZE = 16>
__global__ void fused_matmul_add_kernel(
    const T* A,
    const T* B,
    const T* bias,
    T* C,
    int64_t M,
    int64_t N,
    int64_t K,
    bool has_bias
) {
    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    T sum = 0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute partial products
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result with bias
    if (row < M && col < N) {
        if (has_bias) {
            C[row * N + col] = sum + bias[col];
        } else {
            C[row * N + col] = sum;
        }
    }
}

auto fused_matmul_add_cuda(
    const Tensor& A,
    const Tensor& B,
    const Tensor* bias
) -> Tensor {
    // Assume A: (M, K), B: (K, N), bias: (N,)
    int64_t M = A.shape()[0];
    int64_t K = A.shape()[1];
    int64_t N = B.shape()[1];

    Tensor C = zeros({M, N}, A.dtype(), A.device());

    constexpr int TILE_SIZE = 16;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    if (A.dtype() == DType::Float32) {
        const float* bias_ptr = bias ? bias->data<float>() : nullptr;
        fused_matmul_add_kernel<float, TILE_SIZE><<<blocks, threads>>>(
            A.data<float>(),
            B.data<float>(),
            bias_ptr,
            C.data<float>(),
            M,
            N,
            K,
            bias != nullptr
        );
    } else {
        throw std::runtime_error("fused_matmul_add_cuda: Only Float32 supported");
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return C;
}

// ==============================================================================
// Fused Element-wise Chain CUDA Kernel
// ==============================================================================

/**
 * @brief Fused element-wise operations: add + mul + relu
 *
 * Computes: relu((a + b) * c)
 * Can be extended for arbitrary element-wise chains.
 */
template<typename T>
__global__ void fused_elementwise_chain_kernel(
    const T* a,
    const T* b,
    const T* c,
    T* output,
    int64_t n,
    int op_type  // 0: add+mul+relu, 1: mul+add+relu, etc.
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        T result;
        switch (op_type) {
            case 0:  // (a + b) * c + relu
                result = (a[i] + b[i]) * c[i];
                result = (result > T(0)) ? result : T(0);
                break;
            case 1:  // (a * b) + c + relu
                result = a[i] * b[i] + c[i];
                result = (result > T(0)) ? result : T(0);
                break;
            default:
                result = a[i];
        }
        output[i] = result;
    }
}

auto fused_elementwise_chain_cuda(
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    int op_type
) -> Tensor {
    Tensor output = zeros(a.shape(), a.dtype(), a.device());

    int64_t n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = std::min(blocks, static_cast<int64_t>(65535));

    if (a.dtype() == DType::Float32) {
        fused_elementwise_chain_kernel<<<blocks, threads>>>(
            a.data<float>(),
            b.data<float>(),
            c.data<float>(),
            output.data<float>(),
            n,
            op_type
        );
    } else {
        throw std::runtime_error("fused_elementwise_chain_cuda: Only Float32 supported");
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

// ==============================================================================
// Fused Multi-Head Attention CUDA Kernel (Simplified)
// ==============================================================================

/**
 * @brief Fused attention: Q @ K.T -> Softmax -> @ V
 *
 * Simplified single-head attention for demonstration.
 * Full multi-head would require more complex implementation.
 */
template<typename T, int BLOCK_SIZE>
__global__ void fused_attention_kernel(
    const T* Q,     // (batch, seq_len, d_k)
    const T* K,     // (batch, seq_len, d_k)
    const T* V,     // (batch, seq_len, d_v)
    T* output,      // (batch, seq_len, d_v)
    int64_t batch_size,
    int64_t seq_len,
    int64_t d_k,
    int64_t d_v,
    T scale
) {
    int64_t batch = blockIdx.z;
    int64_t row = blockIdx.y;

    if (batch >= batch_size || row >= seq_len) return;

    __shared__ T shared_scores[BLOCK_SIZE];
    __shared__ T shared_sum[BLOCK_SIZE];

    const T* q_row = Q + (batch * seq_len + row) * d_k;

    // Compute attention scores: Q @ K.T
    T max_score = -INFINITY;
    for (int64_t col = threadIdx.x; col < seq_len; col += blockDim.x) {
        const T* k_row = K + (batch * seq_len + col) * d_k;
        T score = 0;
        for (int64_t i = 0; i < d_k; ++i) {
            score += q_row[i] * k_row[i];
        }
        score *= scale;
        max_score = fmaxf(max_score, score);
        shared_scores[threadIdx.x] = score;
    }

    // Find global max for numerical stability
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && threadIdx.x + s < seq_len) {
            max_score = fmaxf(max_score, shared_scores[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Compute softmax
    T sum_exp = 0;
    for (int64_t col = threadIdx.x; col < seq_len; col += blockDim.x) {
        const T* k_row = K + (batch * seq_len + col) * d_k;
        T score = 0;
        for (int64_t i = 0; i < d_k; ++i) {
            score += q_row[i] * k_row[i];
        }
        score = expf(score * scale - max_score);
        shared_scores[threadIdx.x] = score;
        sum_exp += score;
    }

    shared_sum[threadIdx.x] = sum_exp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    T sum_total = shared_sum[0];

    // Compute attention @ V
    for (int64_t d = threadIdx.x; d < d_v; d += blockDim.x) {
        T result = 0;
        for (int64_t col = 0; col < seq_len; ++col) {
            const T* k_row = K + (batch * seq_len + col) * d_k;
            T score = 0;
            for (int64_t i = 0; i < d_k; ++i) {
                score += q_row[i] * k_row[i];
            }
            T attention_weight = expf(score * scale - max_score) / sum_total;

            const T* v_row = V + (batch * seq_len + col) * d_v;
            result += attention_weight * v_row[d];
        }
        output[(batch * seq_len + row) * d_v + d] = result;
    }
}

auto fused_attention_cuda(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    float scale
) -> Tensor {
    int64_t batch_size = Q.shape()[0];
    int64_t seq_len = Q.shape()[1];
    int64_t d_k = Q.shape()[2];
    int64_t d_v = V.shape()[2];

    Tensor output = zeros({batch_size, seq_len, d_v}, Q.dtype(), Q.device());

    constexpr int BLOCK_SIZE = 256;
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(1, seq_len, batch_size);

    if (Q.dtype() == DType::Float32) {
        fused_attention_kernel<float, BLOCK_SIZE><<<blocks, threads>>>(
            Q.data<float>(),
            K.data<float>(),
            V.data<float>(),
            output.data<float>(),
            batch_size,
            seq_len,
            d_k,
            d_v,
            scale
        );
    } else {
        throw std::runtime_error("fused_attention_cuda: Only Float32 supported");
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_CUDA_AVAILABLE
