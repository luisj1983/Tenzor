#ifdef TENZOR_CUDA_AVAILABLE

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include "tenzor/core/tensor.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <vector>

namespace tenzor {
namespace cuda {

// Helper to clamp blocks to max grid size
inline int clamp_blocks(int64_t blocks) {
    return static_cast<int>(std::min(blocks, static_cast<int64_t>(65535)));
}

// Helper to create a zero-initialized CUDA tensor
inline Tensor create_cuda_zeros(const std::vector<int64_t>& shape, DType dtype, Device device) {
    Tensor t(shape, dtype, device);
    size_t bytes = t.numel() * dtype_size(dtype);
    // Use data_ptr() which returns void* for any dtype
    cudaMemset(t.data_ptr(), 0, bytes);
    return t;
}

// Helper to convert span to vector
inline std::vector<int64_t> to_vector(const std::span<const int64_t>& s) {
    return std::vector<int64_t>(s.begin(), s.end());
}

// Helper for computing sum of a 1D tensor on GPU (for reduction)
inline float cuda_sum_float(const Tensor& t) {
    int64_t n = t.numel();
    const float* data = t.data<float>();

    // Allocate temp storage for CUB reduction
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));

    // Determine temp storage requirements
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, data, d_out, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sum
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, data, d_out, n);

    // Copy result back
    float result;
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_temp_storage);
    cudaFree(d_out);

    return result;
}

// Helper to create scalar tensor on device
inline Tensor create_scalar_tensor(float value, DType dtype, Device device) {
    Tensor t({1}, dtype, device);
    cudaMemcpy(t.data_ptr(), &value, sizeof(float), cudaMemcpyHostToDevice);
    return t;
}

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
    Tensor output = create_cuda_zeros(output_shape, input.dtype(), input.device());

    // Launch kernel
    int64_t total_elements = batch_size * out_features;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = clamp_blocks(blocks);

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

    Tensor output = create_cuda_zeros(to_vector(input.shape()), input.dtype(), input.device());

    int64_t total_elements = input.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = clamp_blocks(blocks);

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

    Tensor losses = create_cuda_zeros({batch_size}, logits.dtype(), logits.device());

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

    // Apply reduction
    if (reduction == "mean") {
        float sum_val = cuda_sum_float(losses);
        return create_scalar_tensor(sum_val / batch_size, logits.dtype(), logits.device());
    } else if (reduction == "sum") {
        float sum_val = cuda_sum_float(losses);
        return create_scalar_tensor(sum_val, logits.dtype(), logits.device());
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
    Tensor result = create_cuda_zeros(to_vector(a.shape()), a.dtype(), a.device());

    int64_t n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = clamp_blocks(blocks);

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
    Tensor output = create_cuda_zeros(to_vector(input.shape()), input.dtype(), input.device());

    int64_t n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = clamp_blocks(blocks);

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
    T* mean_out,        // Output: saved mean for backward
    T* inv_std_out,     // Output: saved inv_std for backward
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

    // Save mean and inv_std for backward pass
    if (threadIdx.x == 0) {
        mean_out[b] = mean;
        inv_std_out[b] = inv_std;
    }

    // Normalize and scale
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        T normalized = (batch_in[i] - mean) * inv_std;
        batch_out[i] = normalized * weight[i] + bias[i];
    }
}

/**
 * @brief Host wrapper for LayerNorm forward CUDA kernel
 *
 * @param input Input tensor
 * @param normalized_shape Shape of normalized dimensions
 * @param weight Weight (gamma) parameter
 * @param bias Bias (beta) parameter
 * @param eps Epsilon for numerical stability
 * @return Tuple of (output, mean, inv_std) where mean and inv_std are saved for backward
 */
auto fused_layer_norm_cuda(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    float eps
) -> std::tuple<Tensor, Tensor, Tensor> {
    int64_t norm_size = 1;
    for (auto dim : normalized_shape) {
        norm_size *= dim;
    }

    int64_t batch_size = input.numel() / norm_size;

    Tensor output = create_cuda_zeros(to_vector(input.shape()), input.dtype(), input.device());
    Tensor mean = create_cuda_zeros({batch_size}, input.dtype(), input.device());
    Tensor inv_std = create_cuda_zeros({batch_size}, input.dtype(), input.device());

    constexpr int BLOCK_SIZE = 256;
    int blocks = batch_size;

    if (input.dtype() == DType::Float32) {
        fused_layer_norm_kernel<float, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
            input.data<float>(),
            weight.data<float>(),
            bias.data<float>(),
            output.data<float>(),
            mean.data<float>(),
            inv_std.data<float>(),
            batch_size,
            norm_size,
            eps
        );
    } else {
        throw std::runtime_error("fused_layer_norm_cuda: Only Float32 supported");
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaDeviceSynchronize() - async execution is critical for performance

    return std::make_tuple(output, mean, inv_std);
}

// ==============================================================================
// Fused Layer Norm Backward CUDA Kernel
// ==============================================================================

/**
 * @brief CUDA kernel for LayerNorm backward pass
 *
 * Computes gradients for input, weight, and bias given output gradients.
 * Uses efficient parallel reduction for batch-wise operations.
 */
template<typename T, int BLOCK_SIZE>
__global__ void fused_layer_norm_backward_kernel(
    const T* grad_output,    // Gradient from next layer
    const T* input,          // Original input from forward pass
    const T* weight,         // Weight (gamma) from forward pass
    const T* mean,           // Saved mean from forward pass
    const T* inv_std,        // Saved 1/sqrt(var + eps) from forward pass
    T* grad_input,           // Output: gradient w.r.t. input
    T* grad_weight,          // Output: gradient w.r.t. weight (accumulated)
    T* grad_bias,            // Output: gradient w.r.t. bias (accumulated)
    int64_t batch_size,
    int64_t norm_size
) {
    int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const T* batch_grad_out = grad_output + b * norm_size;
    const T* batch_in = input + b * norm_size;
    T* batch_grad_in = grad_input + b * norm_size;

    T batch_mean = mean[b];
    T batch_inv_std = inv_std[b];

    __shared__ T shared_sum1[BLOCK_SIZE];  // For sum(grad_out * weight)
    __shared__ T shared_sum2[BLOCK_SIZE];  // For sum(grad_out * weight * normalized)

    // Compute sums needed for input gradient
    T sum_grad_out = 0;
    T sum_grad_out_normalized = 0;

    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        T normalized = (batch_in[i] - batch_mean) * batch_inv_std;
        T grad_out_weighted = batch_grad_out[i] * weight[i];

        sum_grad_out += grad_out_weighted;
        sum_grad_out_normalized += grad_out_weighted * normalized;

        // Accumulate weight and bias gradients atomically
        atomicAdd(&grad_weight[i], batch_grad_out[i] * normalized);
        atomicAdd(&grad_bias[i], batch_grad_out[i]);
    }

    shared_sum1[threadIdx.x] = sum_grad_out;
    shared_sum2[threadIdx.x] = sum_grad_out_normalized;
    __syncthreads();

    // Parallel reduction for sums
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum1[threadIdx.x] += shared_sum1[threadIdx.x + s];
            shared_sum2[threadIdx.x] += shared_sum2[threadIdx.x + s];
        }
        __syncthreads();
    }

    T mean_grad_out = shared_sum1[0] / norm_size;
    T mean_grad_out_normalized = shared_sum2[0] / norm_size;

    // Compute input gradients
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        T normalized = (batch_in[i] - batch_mean) * batch_inv_std;
        T grad_out_weighted = batch_grad_out[i] * weight[i];

        batch_grad_in[i] = (grad_out_weighted - mean_grad_out -
                           normalized * mean_grad_out_normalized) * batch_inv_std;
    }
}

/**
 * @brief Host wrapper for LayerNorm backward CUDA kernel
 *
 * @param grad_output Gradient from next layer
 * @param input Original input from forward pass
 * @param weight Weight (gamma) parameter
 * @param mean Saved mean from forward pass
 * @param inv_std Saved inverse standard deviation from forward pass
 * @param normalized_shape Shape of normalized dimensions
 * @return Tuple of (grad_input, grad_weight, grad_bias)
 */
auto fused_layer_norm_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& mean,
    const Tensor& inv_std,
    const std::vector<int64_t>& normalized_shape
) -> std::tuple<Tensor, Tensor, Tensor> {
    int64_t norm_size = 1;
    for (auto dim : normalized_shape) {
        norm_size *= dim;
    }

    int64_t batch_size = input.numel() / norm_size;

    // Allocate output tensors
    Tensor grad_input = create_cuda_zeros(to_vector(input.shape()), input.dtype(), input.device());
    Tensor grad_weight = create_cuda_zeros({norm_size}, input.dtype(), input.device());
    Tensor grad_bias = create_cuda_zeros({norm_size}, input.dtype(), input.device());

    constexpr int BLOCK_SIZE = 256;
    int blocks = batch_size;

    if (input.dtype() == DType::Float32) {
        fused_layer_norm_backward_kernel<float, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
            grad_output.data<float>(),
            input.data<float>(),
            weight.data<float>(),
            mean.data<float>(),
            inv_std.data<float>(),
            grad_input.data<float>(),
            grad_weight.data<float>(),
            grad_bias.data<float>(),
            batch_size,
            norm_size
        );
    } else {
        throw std::runtime_error("fused_layer_norm_backward_cuda: Only Float32 supported");
    }

    CUDA_CHECK(cudaGetLastError());
    // NOTE: Removed cudaDeviceSynchronize() - async execution is critical for performance

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// ==============================================================================
// Fused RMSNorm CUDA Kernel
// ==============================================================================

/**
 * @brief Optimized RMSNorm forward kernel using warp-level primitives
 *
 * RMSNorm: output = x * weight / sqrt(mean(x^2) + eps)
 * Uses warp shuffle for fast reduction and vectorized memory access.
 */
template<typename T, int BLOCK_SIZE>
__global__ void fused_rms_norm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    T* __restrict__ rrms_out,
    int64_t batch_size,
    int64_t norm_size,
    T eps
) {
    int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const T* batch_in = input + b * norm_size;
    T* batch_out = output + b * norm_size;

    // Use vectorized loads when possible (float4 = 4 floats at once)
    const int vec_size = 4;
    int64_t vec_norm_size = norm_size / vec_size;
    int64_t remainder = norm_size % vec_size;

    // Compute sum of squares with vectorized loads
    T sum_sq = 0;

    // Vectorized portion
    const float4* batch_in_vec = reinterpret_cast<const float4*>(batch_in);
    for (int64_t i = threadIdx.x; i < vec_norm_size; i += blockDim.x) {
        float4 v = batch_in_vec[i];
        sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Handle remainder
    for (int64_t i = vec_norm_size * vec_size + threadIdx.x; i < norm_size; i += blockDim.x) {
        T val = batch_in[i];
        sum_sq += val * val;
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Inter-warp reduction via shared memory
    __shared__ T warp_sums[32];  // Max 32 warps per block
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if (lane == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0) {
        sum_sq = (lane < (BLOCK_SIZE / 32)) ? warp_sums[lane] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    // Broadcast rrms to all threads
    __shared__ T shared_rrms;
    if (threadIdx.x == 0) {
        T mean_sq = sum_sq / norm_size;
        shared_rrms = rsqrtf(mean_sq + eps);
        rrms_out[b] = shared_rrms;
    }
    __syncthreads();
    T rrms = shared_rrms;

    // Apply normalization with vectorized stores
    float4* batch_out_vec = reinterpret_cast<float4*>(batch_out);
    const float4* weight_vec = reinterpret_cast<const float4*>(weight);

    for (int64_t i = threadIdx.x; i < vec_norm_size; i += blockDim.x) {
        float4 v = batch_in_vec[i];
        float4 w = weight_vec[i];
        float4 out;
        out.x = v.x * rrms * w.x;
        out.y = v.y * rrms * w.y;
        out.z = v.z * rrms * w.z;
        out.w = v.w * rrms * w.w;
        batch_out_vec[i] = out;
    }

    // Handle remainder
    for (int64_t i = vec_norm_size * vec_size + threadIdx.x; i < norm_size; i += blockDim.x) {
        batch_out[i] = batch_in[i] * rrms * weight[i];
    }
}

/**
 * @brief Fused RMSNorm forward pass
 *
 * @param input Input tensor [batch_size, ..., norm_size]
 * @param weight Weight tensor [norm_size]
 * @param eps Epsilon for numerical stability
 * @return Tuple of (output, rrms) where rrms is saved for backward
 */
auto fused_rms_norm_cuda(
    const Tensor& input,
    const Tensor& weight,
    float eps
) -> std::tuple<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t norm_size = shape.back();

    // Calculate batch size (all dimensions except last)
    int64_t batch_size = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        batch_size *= shape[i];
    }

    Tensor output = create_cuda_zeros(to_vector(input.shape()), input.dtype(), input.device());
    Tensor rrms = create_cuda_zeros({batch_size}, input.dtype(), input.device());

    constexpr int BLOCK_SIZE = 256;
    int blocks = batch_size;

    if (input.dtype() == DType::Float32) {
        fused_rms_norm_kernel<float, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
            input.data<float>(),
            weight.data<float>(),
            output.data<float>(),
            rrms.data<float>(),
            batch_size,
            norm_size,
            eps
        );
    } else {
        throw std::runtime_error("fused_rms_norm_cuda: Only Float32 supported");
    }

    CUDA_CHECK(cudaGetLastError());

    return std::make_tuple(output, rrms);
}

/**
 * @brief Fused RMSNorm backward kernel
 *
 * Computes gradients for input and weight.
 * grad_input = weight * rrms * (grad_out - x * rrms^2 * mean(grad_out * x * weight))
 */
template<typename T, int BLOCK_SIZE>
__global__ void fused_rms_norm_backward_kernel(
    const T* grad_output,    // Gradient from next layer
    const T* input,          // Original input from forward pass
    const T* weight,         // Weight from forward pass
    const T* rrms,           // Saved reciprocal RMS from forward pass
    T* grad_input,           // Output: gradient w.r.t. input
    T* grad_weight,          // Output: gradient w.r.t. weight (accumulated)
    int64_t batch_size,
    int64_t norm_size
) {
    int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const T* batch_grad_out = grad_output + b * norm_size;
    const T* batch_in = input + b * norm_size;
    T* batch_grad_in = grad_input + b * norm_size;

    T batch_rrms = rrms[b];

    __shared__ T shared_sum[BLOCK_SIZE];

    // Compute sum(grad_out * x * weight) / norm_size for input gradient
    T sum_grad_x_w = 0;
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        sum_grad_x_w += batch_grad_out[i] * batch_in[i] * weight[i];
    }

    shared_sum[threadIdx.x] = sum_grad_x_w;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    T mean_grad_x_w = shared_sum[0] / norm_size;

    // Compute input gradient and accumulate weight gradient
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        T x_i = batch_in[i];
        T w_i = weight[i];
        T grad_out_i = batch_grad_out[i];

        // grad_input = rrms * (grad_out * weight - x * rrms^2 * mean_grad_x_w)
        batch_grad_in[i] = batch_rrms * (grad_out_i * w_i - x_i * batch_rrms * batch_rrms * mean_grad_x_w);

        // grad_weight accumulation (atomic for thread safety across batches)
        atomicAdd(&grad_weight[i], grad_out_i * x_i * batch_rrms);
    }
}

/**
 * @brief Fused RMSNorm backward pass
 */
auto fused_rms_norm_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& rrms
) -> std::tuple<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t norm_size = shape.back();

    int64_t batch_size = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        batch_size *= shape[i];
    }

    Tensor grad_input = create_cuda_zeros(to_vector(input.shape()), input.dtype(), input.device());
    Tensor grad_weight = create_cuda_zeros({norm_size}, input.dtype(), input.device());

    constexpr int BLOCK_SIZE = 256;
    int blocks = batch_size;

    if (input.dtype() == DType::Float32) {
        fused_rms_norm_backward_kernel<float, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
            grad_output.data<float>(),
            input.data<float>(),
            weight.data<float>(),
            rrms.data<float>(),
            grad_input.data<float>(),
            grad_weight.data<float>(),
            batch_size,
            norm_size
        );
    } else {
        throw std::runtime_error("fused_rms_norm_backward_cuda: Only Float32 supported");
    }

    CUDA_CHECK(cudaGetLastError());

    return std::make_tuple(grad_input, grad_weight);
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

    Tensor output = create_cuda_zeros({batch_size, out_channels, out_h, out_w}, input.dtype(), input.device());

    int64_t total_elements = batch_size * out_channels * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = clamp_blocks(blocks);

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

    Tensor C = create_cuda_zeros({M, N}, A.dtype(), A.device());

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
    Tensor output = create_cuda_zeros(to_vector(a.shape()), a.dtype(), a.device());

    int64_t n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = clamp_blocks(blocks);

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

    return output;
}

// ==============================================================================
// Flash Attention CUDA Kernel - Optimized with Warp-Level Parallelism
// ==============================================================================

/**
 * @brief Warp-level max reduction using shuffle
 */
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

/**
 * @brief Warp-level sum reduction using shuffle
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * @brief Optimized Flash Attention V2 kernel
 *
 * This kernel implements the Flash Attention algorithm with online softmax,
 * achieving O(N) memory complexity instead of O(N^2).
 *
 * Key optimizations:
 * 1. Each block processes one query row with 256 threads
 * 2. KV tiles loaded cooperatively into shared memory
 * 3. Online softmax with running max/sum for numerical stability
 * 4. Warp shuffle for fast reductions
 * 5. All threads participate in output accumulation
 *
 * Supports head_dim: 32, 64, 80, 96, 128
 */
template<int HEAD_DIM, int BLOCK_SIZE = 256>
__global__ void flash_attention_v2_kernel(
    const float* __restrict__ Q,     // [batch_heads, seq_len_q, head_dim]
    const float* __restrict__ K,     // [batch_heads, seq_len_k, head_dim]
    const float* __restrict__ V,     // [batch_heads, seq_len_k, head_dim]
    float* __restrict__ O,           // [batch_heads, seq_len_q, head_dim]
    const int seq_len_q,
    const int seq_len_k,
    const float scale
) {
    // Configuration
    constexpr int Bc = 32;  // Keys per tile - small for register pressure

    const int batch_head = blockIdx.x;
    const int query_idx = blockIdx.y;

    if (query_idx >= seq_len_q) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = BLOCK_SIZE / 32;

    // Base pointers
    const float* Q_row = Q + batch_head * seq_len_q * HEAD_DIM + query_idx * HEAD_DIM;
    const float* K_base = K + batch_head * seq_len_k * HEAD_DIM;
    const float* V_base = V + batch_head * seq_len_k * HEAD_DIM;
    float* O_row = O + batch_head * seq_len_q * HEAD_DIM + query_idx * HEAD_DIM;

    // Shared memory layout
    constexpr int K_STRIDE = HEAD_DIM + 4;  // Padding for bank conflicts
    extern __shared__ float smem[];
    float* K_tile = smem;                          // [Bc][K_STRIDE]
    float* V_tile = smem + Bc * K_STRIDE;          // [Bc][K_STRIDE]
    float* Q_shared = smem + 2 * Bc * K_STRIDE;    // [HEAD_DIM]
    float* scores_shared = Q_shared + HEAD_DIM;    // [Bc]
    float* reduce_buf = scores_shared + Bc;        // [num_warps]

    // Load Q row into shared memory
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        Q_shared[d] = Q_row[d] * scale;
    }
    __syncthreads();

    // Output accumulator - each thread handles a subset of output dimensions
    float o_local[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Up to 4 elements per thread

    // Online softmax state (shared across block via reduction)
    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    const int num_kv_blocks = (seq_len_k + Bc - 1) / Bc;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int k_start = kv_block * Bc;
        const int actual_Bc = min(Bc, seq_len_k - k_start);

        // Load K/V tile cooperatively
        for (int idx = tid; idx < actual_Bc * HEAD_DIM; idx += BLOCK_SIZE) {
            int row = idx / HEAD_DIM;
            int col = idx % HEAD_DIM;
            K_tile[row * K_STRIDE + col] = K_base[(k_start + row) * HEAD_DIM + col];
            V_tile[row * K_STRIDE + col] = V_base[(k_start + row) * HEAD_DIM + col];
        }
        __syncthreads();

        // Step 1: Compute all Q·K scores and find max
        // Each thread computes one or more scores
        float local_max = -INFINITY;

        for (int j = tid; j < actual_Bc; j += BLOCK_SIZE) {
            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += Q_shared[d] * K_tile[j * K_STRIDE + d];
            }
            scores_shared[j] = score;
            local_max = fmaxf(local_max, score);
        }

        // Reduce max across block using warp shuffle + shared memory
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
        }
        if (lane_id == 0) {
            reduce_buf[warp_id] = local_max;
        }
        __syncthreads();

        if (warp_id == 0) {
            float val = (lane_id < num_warps) ? reduce_buf[lane_id] : -INFINITY;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
            }
            if (lane_id == 0) {
                reduce_buf[0] = val;
            }
        }
        __syncthreads();
        float block_max = reduce_buf[0];

        // Step 2: Compute exp(score - max) and sum
        float local_sum = 0.0f;

        for (int j = tid; j < actual_Bc; j += BLOCK_SIZE) {
            float exp_score = expf(scores_shared[j] - block_max);
            scores_shared[j] = exp_score;  // Store exp for P @ V
            local_sum += exp_score;
        }
        __syncthreads();

        // Reduce sum
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
        }
        if (lane_id == 0) {
            reduce_buf[warp_id] = local_sum;
        }
        __syncthreads();

        if (warp_id == 0) {
            float val = (lane_id < num_warps) ? reduce_buf[lane_id] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                val += __shfl_xor_sync(0xffffffff, val, offset);
            }
            if (lane_id == 0) {
                reduce_buf[0] = val;
            }
        }
        __syncthreads();
        float block_sum = reduce_buf[0];

        // Step 3: Online softmax rescaling
        float m_new = fmaxf(m_prev, block_max);
        float exp_prev = expf(m_prev - m_new);
        float exp_curr = expf(block_max - m_new);
        float l_new = exp_prev * l_prev + exp_curr * block_sum;

        // Step 4: Rescale previous output and accumulate P @ V
        // Each thread handles HEAD_DIM / BLOCK_SIZE output elements
        for (int i = 0; i < 4; ++i) {
            int d = tid + i * BLOCK_SIZE;
            if (d < HEAD_DIM) {
                // Rescale previous accumulator
                o_local[i] *= exp_prev;

                // Add new contribution: sum over j of P[j] * V[j, d]
                float pv_sum = 0.0f;
                for (int j = 0; j < actual_Bc; ++j) {
                    pv_sum += scores_shared[j] * V_tile[j * K_STRIDE + d];
                }
                o_local[i] += exp_curr * pv_sum;
            }
        }

        m_prev = m_new;
        l_prev = l_new;

        __syncthreads();
    }

    // Final normalization and write output
    float l_inv = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;

    for (int i = 0; i < 4; ++i) {
        int d = tid + i * BLOCK_SIZE;
        if (d < HEAD_DIM) {
            O_row[d] = o_local[i] * l_inv;
        }
    }
}

/**
 * @brief Legacy Flash Attention kernel (fallback for non-64 head_dim)
 */
template<int Br, int Bc, int HEAD_DIM, int BLOCK_SIZE>
__global__ void flash_attention_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int64_t batch_heads,
    int64_t seq_len_q,
    int64_t seq_len_k,
    int64_t head_dim,
    float scale
) {
    const int64_t batch_head = blockIdx.x;
    const int64_t q_block_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int64_t q_start = q_block_idx * Br;
    if (q_start >= seq_len_q) return;
    const int64_t q_end = min(q_start + (int64_t)Br, seq_len_q);
    const int actual_Br = (int)(q_end - q_start);

    extern __shared__ float smem[];
    float* Q_tile = smem;
    float* K_tile = Q_tile + Br * HEAD_DIM;
    float* V_tile = K_tile + Bc * HEAD_DIM;
    float* S_tile = V_tile + Bc * HEAD_DIM;
    float* O_acc = S_tile + Br * Bc;
    float* m_i = O_acc + Br * HEAD_DIM;
    float* l_i = m_i + Br;

    const float* Q_base = Q + batch_head * seq_len_q * head_dim;
    const float* K_base = K + batch_head * seq_len_k * head_dim;
    const float* V_base = V + batch_head * seq_len_k * head_dim;
    float* O_base = O + batch_head * seq_len_q * head_dim;

    for (int i = tid; i < Br; i += BLOCK_SIZE) {
        m_i[i] = -INFINITY;
        l_i[i] = 0.0f;
    }
    for (int i = tid; i < Br * HEAD_DIM; i += BLOCK_SIZE) {
        O_acc[i] = 0.0f;
    }
    for (int i = tid; i < actual_Br * HEAD_DIM; i += BLOCK_SIZE) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        Q_tile[row * HEAD_DIM + col] = Q_base[(q_start + row) * head_dim + col];
    }
    __syncthreads();

    const int64_t num_kv_blocks = (seq_len_k + Bc - 1) / Bc;

    for (int64_t kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int64_t k_start = kv_block * Bc;
        const int64_t k_end = min(k_start + (int64_t)Bc, seq_len_k);
        const int actual_Bc = (int)(k_end - k_start);

        for (int i = tid; i < actual_Bc * HEAD_DIM; i += BLOCK_SIZE) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            K_tile[row * HEAD_DIM + col] = K_base[(k_start + row) * head_dim + col];
            V_tile[row * HEAD_DIM + col] = V_base[(k_start + row) * head_dim + col];
        }
        for (int i = tid + actual_Bc * HEAD_DIM; i < Bc * HEAD_DIM; i += BLOCK_SIZE) {
            K_tile[i] = 0.0f;
            V_tile[i] = 0.0f;
        }
        __syncthreads();

        for (int idx = tid; idx < actual_Br * actual_Bc; idx += BLOCK_SIZE) {
            int i = idx / actual_Bc;
            int j = idx % actual_Bc;
            float sum = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; ++d) {
                sum += Q_tile[i * HEAD_DIM + d] * K_tile[j * HEAD_DIM + d];
            }
            S_tile[i * Bc + j] = sum * scale;
        }
        for (int idx = tid; idx < Br * Bc; idx += BLOCK_SIZE) {
            int i = idx / Bc;
            int j = idx % Bc;
            if (i >= actual_Br || j >= actual_Bc) {
                S_tile[idx] = -INFINITY;
            }
        }
        __syncthreads();

        for (int row = tid; row < actual_Br; row += BLOCK_SIZE) {
            float row_max = -INFINITY;
            for (int j = 0; j < actual_Bc; ++j) {
                row_max = fmaxf(row_max, S_tile[row * Bc + j]);
            }
            float row_sum = 0.0f;
            for (int j = 0; j < actual_Bc; ++j) {
                float val = expf(S_tile[row * Bc + j] - row_max);
                S_tile[row * Bc + j] = val;
                row_sum += val;
            }
            float m_old = m_i[row];
            float m_new = fmaxf(m_old, row_max);
            float l_old = l_i[row];
            float scale_old = expf(m_old - m_new);
            float scale_cur = expf(row_max - m_new);
            float l_new = scale_old * l_old + scale_cur * row_sum;
            for (int d = 0; d < HEAD_DIM; ++d) {
                O_acc[row * HEAD_DIM + d] *= scale_old;
            }
            for (int j = 0; j < actual_Bc; ++j) {
                float p_ij = S_tile[row * Bc + j] * scale_cur;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    O_acc[row * HEAD_DIM + d] += p_ij * V_tile[j * HEAD_DIM + d];
                }
            }
            m_i[row] = m_new;
            l_i[row] = l_new;
        }
        __syncthreads();
    }

    for (int i = tid; i < actual_Br * HEAD_DIM; i += BLOCK_SIZE) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        float l_inv = 1.0f / l_i[row];
        O_base[(q_start + row) * head_dim + col] = O_acc[row * HEAD_DIM + col] * l_inv;
    }
}

// ==============================================================================
// Naive Fused Attention CUDA Kernel (Fallback for non-standard head_dim)
// ==============================================================================

/**
 * @brief Naive fused attention (fallback when Flash Attention constraints not met)
 */
template<typename T, int BLOCK_SIZE, int MAX_SEQ_LEN = 1024>
__global__ void fused_attention_kernel_naive(
    const T* __restrict__ Q,     // (batch_heads, seq_len_q, head_dim)
    const T* __restrict__ K,     // (batch_heads, seq_len_k, head_dim)
    const T* __restrict__ V,     // (batch_heads, seq_len_k, head_dim)
    T* __restrict__ output,      // (batch_heads, seq_len_q, head_dim)
    int64_t batch_heads,
    int64_t seq_len_q,
    int64_t seq_len_k,
    int64_t head_dim,
    T scale
) {
    // Each block handles one (batch_head, query_row) pair
    int64_t batch_head = blockIdx.x;
    int64_t query_row = blockIdx.y;

    if (batch_head >= batch_heads || query_row >= seq_len_q) return;

    // Shared memory for scores and partial results
    extern __shared__ char shared_mem[];
    T* scores = reinterpret_cast<T*>(shared_mem);  // [seq_len_k] for attention scores
    T* partial_max = scores + seq_len_k;           // [BLOCK_SIZE] for reduction
    T* partial_sum = partial_max + BLOCK_SIZE;     // [BLOCK_SIZE] for reduction

    const T* q_row = Q + (batch_head * seq_len_q + query_row) * head_dim;
    const T* k_base = K + batch_head * seq_len_k * head_dim;
    const T* v_base = V + batch_head * seq_len_k * head_dim;
    T* out_row = output + (batch_head * seq_len_q + query_row) * head_dim;

    // =========================================================================
    // Step 1: Compute all attention scores Q @ K.T (store in shared memory)
    // =========================================================================
    T thread_max = -INFINITY;
    for (int64_t k_idx = threadIdx.x; k_idx < seq_len_k; k_idx += blockDim.x) {
        const T* k_row = k_base + k_idx * head_dim;
        T score = 0;
        #pragma unroll 4
        for (int64_t d = 0; d < head_dim; ++d) {
            score += q_row[d] * k_row[d];
        }
        score *= scale;
        scores[k_idx] = score;
        thread_max = fmaxf(thread_max, score);
    }

    // =========================================================================
    // Step 2: Find global max (parallel reduction)
    // =========================================================================
    partial_max[threadIdx.x] = thread_max;
    __syncthreads();

    // Warp-level reduction first
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, thread_max, offset);
        thread_max = fmaxf(thread_max, other);
    }

    // Inter-warp reduction
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) {
        partial_max[warp_id] = thread_max;
    }
    __syncthreads();

    T global_max;
    if (threadIdx.x == 0) {
        global_max = partial_max[0];
        int num_warps = (blockDim.x + 31) / 32;
        for (int i = 1; i < num_warps; ++i) {
            global_max = fmaxf(global_max, partial_max[i]);
        }
        partial_max[0] = global_max;
    }
    __syncthreads();
    global_max = partial_max[0];

    // =========================================================================
    // Step 3: Compute exp(score - max) and sum (softmax denominator)
    // =========================================================================
    T thread_sum = 0;
    for (int64_t k_idx = threadIdx.x; k_idx < seq_len_k; k_idx += blockDim.x) {
        T exp_score = expf(scores[k_idx] - global_max);
        scores[k_idx] = exp_score;  // Store normalized exp for later use
        thread_sum += exp_score;
    }

    // Reduce sum
    partial_sum[threadIdx.x] = thread_sum;
    __syncthreads();

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, thread_sum, offset);
        thread_sum += other;
    }

    if (lane == 0) {
        partial_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    T global_sum;
    if (threadIdx.x == 0) {
        global_sum = partial_sum[0];
        int num_warps = (blockDim.x + 31) / 32;
        for (int i = 1; i < num_warps; ++i) {
            global_sum += partial_sum[i];
        }
        partial_sum[0] = global_sum;
    }
    __syncthreads();
    global_sum = partial_sum[0];

    T inv_sum = 1.0f / global_sum;

    // =========================================================================
    // Step 4: Normalize scores (complete softmax)
    // =========================================================================
    for (int64_t k_idx = threadIdx.x; k_idx < seq_len_k; k_idx += blockDim.x) {
        scores[k_idx] *= inv_sum;
    }
    __syncthreads();

    // =========================================================================
    // Step 5: Compute output = attention_weights @ V
    // =========================================================================
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        T result = 0;
        #pragma unroll 4
        for (int64_t k_idx = 0; k_idx < seq_len_k; ++k_idx) {
            result += scores[k_idx] * v_base[k_idx * head_dim + d];
        }
        out_row[d] = result;
    }
}

auto fused_attention_cuda(
    const Tensor& Q,     // (batch_heads, seq_len_q, head_dim)
    const Tensor& K,     // (batch_heads, seq_len_k, head_dim)
    const Tensor& V,     // (batch_heads, seq_len_k, head_dim)
    float scale
) -> Tensor {
    int64_t batch_heads = Q.shape()[0];
    int64_t seq_len_q = Q.shape()[1];
    int64_t head_dim = Q.shape()[2];
    int64_t seq_len_k = K.shape()[1];

    if (Q.dtype() != DType::Float32) {
        throw std::runtime_error("fused_attention_cuda: Only Float32 supported");
    }

    Tensor output = create_cuda_zeros({batch_heads, seq_len_q, head_dim}, Q.dtype(), Q.device());

    // Optimized Flash Attention V2 - supports multiple head dimensions
    constexpr int BLOCK_SIZE = 256;
    constexpr int Bc = 32;  // Keys per tile

    // Grid: one block per (batch_head, query_row) pair
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(batch_heads, seq_len_q);

    // Compute shared memory size based on head_dim
    // Layout: K_tile[Bc][HEAD_DIM+4] + V_tile[Bc][HEAD_DIM+4] + Q_shared[HEAD_DIM] + scores[Bc] + reduce[8]
    auto compute_smem_size = [](int hd) {
        int k_stride = hd + 4;
        return (2 * Bc * k_stride + hd + Bc + 8) * sizeof(float);
    };

    // Dispatch based on head_dim for optimal unrolling
    if (head_dim == 64) {
        size_t smem_size = compute_smem_size(64);
        flash_attention_v2_kernel<64, BLOCK_SIZE><<<blocks, threads, smem_size>>>(
            Q.data<float>(), K.data<float>(), V.data<float>(), output.data<float>(),
            static_cast<int>(seq_len_q), static_cast<int>(seq_len_k), scale);
    } else if (head_dim == 128) {
        size_t smem_size = compute_smem_size(128);
        flash_attention_v2_kernel<128, BLOCK_SIZE><<<blocks, threads, smem_size>>>(
            Q.data<float>(), K.data<float>(), V.data<float>(), output.data<float>(),
            static_cast<int>(seq_len_q), static_cast<int>(seq_len_k), scale);
    } else if (head_dim == 32) {
        size_t smem_size = compute_smem_size(32);
        flash_attention_v2_kernel<32, BLOCK_SIZE><<<blocks, threads, smem_size>>>(
            Q.data<float>(), K.data<float>(), V.data<float>(), output.data<float>(),
            static_cast<int>(seq_len_q), static_cast<int>(seq_len_k), scale);
    } else if (head_dim == 80) {
        size_t smem_size = compute_smem_size(80);
        flash_attention_v2_kernel<80, BLOCK_SIZE><<<blocks, threads, smem_size>>>(
            Q.data<float>(), K.data<float>(), V.data<float>(), output.data<float>(),
            static_cast<int>(seq_len_q), static_cast<int>(seq_len_k), scale);
    } else if (head_dim == 96) {
        size_t smem_size = compute_smem_size(96);
        flash_attention_v2_kernel<96, BLOCK_SIZE><<<blocks, threads, smem_size>>>(
            Q.data<float>(), K.data<float>(), V.data<float>(), output.data<float>(),
            static_cast<int>(seq_len_q), static_cast<int>(seq_len_k), scale);
    } else {
        // Fallback to naive kernel for non-standard head_dim
        constexpr int NAIVE_BLOCK_SIZE = 256;
        size_t shared_mem_size = (seq_len_k + 2 * NAIVE_BLOCK_SIZE) * sizeof(float);

        fused_attention_kernel_naive<float, NAIVE_BLOCK_SIZE><<<blocks, threads, shared_mem_size>>>(
            Q.data<float>(),
            K.data<float>(),
            V.data<float>(),
            output.data<float>(),
            batch_heads,
            seq_len_q,
            seq_len_k,
            head_dim,
            scale
        );
    }

    CUDA_CHECK(cudaGetLastError());

    return output;
}

// ==============================================================================
// Fused Adam Optimizer CUDA Kernel
// ==============================================================================

/**
 * @brief Fused Adam/AdamW optimizer kernel
 *
 * Performs the complete Adam update in a single kernel:
 * - First moment update: m = beta1 * m + (1-beta1) * grad
 * - Second moment update: v = beta2 * v + (1-beta2) * grad^2
 * - Bias correction applied through step_size
 * - Parameter update: param = param - step_size * m / (sqrt(v_hat) + eps)
 *
 * For AdamW (decoupled weight decay): param = param * (1 - lr * weight_decay) first
 *
 * This eliminates ~15-20 kernel launches per parameter in the naive implementation.
 */
__global__ void fused_adam_kernel(
    float* __restrict__ param,         // Parameter tensor (modified in-place)
    const float* __restrict__ grad,    // Gradient tensor
    float* __restrict__ exp_avg,       // First moment (m) - modified in-place
    float* __restrict__ exp_avg_sq,    // Second moment (v) - modified in-place
    int64_t numel,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,    // 1 - beta1^step
    float bias_correction2,    // 1 - beta2^step
    bool amsgrad,              // Not implemented yet
    bool decoupled_weight_decay  // True for AdamW, false for L2 regularization
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float g = grad[idx];
    float p = param[idx];
    float m = exp_avg[idx];
    float v = exp_avg_sq[idx];

    // Apply L2 regularization to gradient (Adam style) if not decoupled
    if (weight_decay > 0.0f && !decoupled_weight_decay) {
        g = g + weight_decay * p;
    }

    // Update biased first moment estimate
    m = beta1 * m + (1.0f - beta1) * g;

    // Update biased second raw moment estimate
    v = beta2 * v + (1.0f - beta2) * g * g;

    // Compute step size with bias correction
    float step_size = lr / bias_correction1;

    // Compute bias-corrected second moment and update parameter
    // denom = sqrt(v / bias_correction2) + eps
    float denom = sqrtf(v / bias_correction2) + eps;

    // Apply decoupled weight decay (AdamW style) before update
    if (weight_decay > 0.0f && decoupled_weight_decay) {
        p = p * (1.0f - lr * weight_decay);
    }

    // Update parameter
    p = p - step_size * m / denom;

    // Store updated values
    param[idx] = p;
    exp_avg[idx] = m;
    exp_avg_sq[idx] = v;
}

/**
 * @brief Vectorized fused Adam kernel using float4
 *
 * Processes 4 elements per thread for better memory bandwidth utilization.
 */
__global__ void fused_adam_kernel_vec4(
    float4* __restrict__ param,
    const float4* __restrict__ grad,
    float4* __restrict__ exp_avg,
    float4* __restrict__ exp_avg_sq,
    int64_t numel_vec4,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    bool decoupled_weight_decay
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel_vec4) return;

    float4 g = grad[idx];
    float4 p = param[idx];
    float4 m = exp_avg[idx];
    float4 v = exp_avg_sq[idx];

    float step_size = lr / bias_correction1;
    float bc2_inv = 1.0f / bias_correction2;

    // Process all 4 elements
    #define ADAM_UPDATE(comp) \
        if (weight_decay > 0.0f && !decoupled_weight_decay) { \
            g.comp = g.comp + weight_decay * p.comp; \
        } \
        m.comp = beta1 * m.comp + (1.0f - beta1) * g.comp; \
        v.comp = beta2 * v.comp + (1.0f - beta2) * g.comp * g.comp; \
        if (weight_decay > 0.0f && decoupled_weight_decay) { \
            p.comp = p.comp * (1.0f - lr * weight_decay); \
        } \
        p.comp = p.comp - step_size * m.comp / (sqrtf(v.comp * bc2_inv) + eps);

    ADAM_UPDATE(x)
    ADAM_UPDATE(y)
    ADAM_UPDATE(z)
    ADAM_UPDATE(w)

    #undef ADAM_UPDATE

    param[idx] = p;
    exp_avg[idx] = m;
    exp_avg_sq[idx] = v;
}

/**
 * @brief Host function for fused Adam optimizer step
 *
 * @param param Parameter tensor (modified in-place)
 * @param grad Gradient tensor
 * @param exp_avg First moment buffer (modified in-place)
 * @param exp_avg_sq Second moment buffer (modified in-place)
 * @param lr Learning rate
 * @param beta1 First moment decay rate
 * @param beta2 Second moment decay rate
 * @param eps Epsilon for numerical stability
 * @param weight_decay Weight decay coefficient
 * @param step Current step count (for bias correction)
 * @param decoupled_weight_decay If true, use AdamW style decoupled weight decay
 */
auto fused_adam_step_cuda(
    Tensor& param,
    const Tensor& grad,
    Tensor& exp_avg,
    Tensor& exp_avg_sq,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int64_t step,
    bool decoupled_weight_decay,
    cudaStream_t stream
) -> void {
    if (param.dtype() != DType::Float32) {
        throw std::runtime_error("fused_adam_step_cuda: Only Float32 supported");
    }

    int64_t numel = param.numel();

    // Compute bias corrections
    float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(step));

    // Check if we can use vectorized kernel
    bool can_vectorize = (numel % 4 == 0) &&
                         (reinterpret_cast<uintptr_t>(param.data<float>()) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(grad.data<float>()) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(exp_avg.data<float>()) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(exp_avg_sq.data<float>()) % 16 == 0);

    constexpr int BLOCK_SIZE = 256;

    if (can_vectorize && numel >= 1024) {
        int64_t numel_vec4 = numel / 4;
        int blocks = (numel_vec4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        blocks = clamp_blocks(blocks);

        fused_adam_kernel_vec4<<<blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<float4*>(param.data<float>()),
            reinterpret_cast<const float4*>(grad.data<float>()),
            reinterpret_cast<float4*>(exp_avg.data<float>()),
            reinterpret_cast<float4*>(exp_avg_sq.data<float>()),
            numel_vec4,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
            decoupled_weight_decay
        );
    } else {
        int blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
        blocks = clamp_blocks(blocks);

        fused_adam_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            param.data<float>(),
            grad.data<float>(),
            exp_avg.data<float>(),
            exp_avg_sq.data<float>(),
            numel,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
            false,  // amsgrad not implemented
            decoupled_weight_decay
        );
    }

    CUDA_CHECK(cudaGetLastError());
}

// ==============================================================================
// Fused SGD with Momentum CUDA Kernel
// ==============================================================================

/**
 * @brief Fused SGD with momentum and weight decay
 *
 * Nesterov momentum: v = momentum * v + grad; param = param - lr * (grad + momentum * v)
 * Standard momentum: v = momentum * v + grad; param = param - lr * v
 */
__global__ void fused_sgd_kernel(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ momentum_buffer,
    int64_t numel,
    float lr,
    float momentum,
    float weight_decay,
    float dampening,
    bool nesterov,
    bool has_momentum_buffer
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float g = grad[idx];
    float p = param[idx];

    // Apply weight decay
    if (weight_decay > 0.0f) {
        g = g + weight_decay * p;
    }

    if (has_momentum_buffer && momentum > 0.0f) {
        float v = momentum_buffer[idx];

        // Update momentum buffer
        v = momentum * v + (1.0f - dampening) * g;
        momentum_buffer[idx] = v;

        if (nesterov) {
            g = g + momentum * v;
        } else {
            g = v;
        }
    }

    // Update parameter
    param[idx] = p - lr * g;
}

auto fused_sgd_step_cuda(
    Tensor& param,
    const Tensor& grad,
    Tensor* momentum_buffer,
    float lr,
    float momentum,
    float weight_decay,
    float dampening,
    bool nesterov,
    cudaStream_t stream
) -> void {
    if (param.dtype() != DType::Float32) {
        throw std::runtime_error("fused_sgd_step_cuda: Only Float32 supported");
    }

    int64_t numel = param.numel();
    constexpr int BLOCK_SIZE = 256;
    int blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = clamp_blocks(blocks);

    float* momentum_ptr = momentum_buffer ? momentum_buffer->data<float>() : nullptr;

    fused_sgd_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        param.data<float>(),
        grad.data<float>(),
        momentum_ptr,
        numel,
        lr,
        momentum,
        weight_decay,
        dampening,
        nesterov,
        momentum_buffer != nullptr
    );

    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_CUDA_AVAILABLE
