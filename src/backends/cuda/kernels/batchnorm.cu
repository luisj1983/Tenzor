#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace cuda {

// ============================================================================
// CUDA Helper Functions
// ============================================================================

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Grid-stride loop helper
#define CUDA_GRID_STRIDE_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

// FP16 saturating conversion
__device__ __forceinline__ __half __float2half_sat(float x) {
    constexpr float kHalfMax = 65504.0f;
    x = fminf(fmaxf(x, -kHalfMax), kHalfMax);
    return __float2half(x);
}

// Optimal block size
constexpr int BLOCK_SIZE = 256;
constexpr int BATCHNORM_BLOCK_SIZE = 512;

// Calculate grid size
inline int get_num_blocks(int64_t n, int block_size = BLOCK_SIZE) {
    return (n + block_size - 1) / block_size;
}

// ============================================================================
// Warp and Block Reduction Primitives
// ============================================================================

// Warp-level reduction using shuffle instructions
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
template<typename T>
__device__ T block_reduce_sum(T val, T* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : T(0);
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

// ============================================================================
// BatchNorm2d Mean/Variance Computation (Welford's Algorithm)
// ============================================================================

// Compute per-channel mean and variance using Welford's online algorithm
// Input: [N, C, H, W] - NCHW format
// Output: mean[C], variance[C]
template<typename T>
__global__ void batchnorm_mean_var_kernel(const T* input,
                                          T* mean,
                                          T* variance,
                                          int64_t N,
                                          int64_t C,
                                          int64_t H,
                                          int64_t W) {
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* shared = reinterpret_cast<T*>(shared_mem);

    int64_t spatial_size = H * W;
    int64_t total_elements = N * spatial_size;

    // Each block handles one channel
    int64_t c = blockIdx.x;
    if (c >= C) return;

    // Welford's algorithm for numerically stable mean/variance computation
    T mean_acc = T(0);
    T m2_acc = T(0);  // Sum of squared differences from mean
    int64_t count = 0;

    // Each thread processes multiple elements
    for (int64_t n = 0; n < N; n++) {
        for (int64_t idx = threadIdx.x; idx < spatial_size; idx += blockDim.x) {
            int64_t h = idx / W;
            int64_t w = idx % W;
            int64_t tensor_idx = ((n * C + c) * H + h) * W + w;

            T value = input[tensor_idx];
            count++;

            // Welford's update
            T delta = value - mean_acc;
            mean_acc += delta / T(count);
            T delta2 = value - mean_acc;
            m2_acc += delta * delta2;
        }
    }

    // Reduce across threads in block
    mean_acc = block_reduce_sum(mean_acc, shared);
    __syncthreads();
    m2_acc = block_reduce_sum(m2_acc, shared);
    __syncthreads();

    // Thread 0 writes final result
    if (threadIdx.x == 0) {
        mean[c] = mean_acc / T(blockDim.x);
        variance[c] = m2_acc / T(total_elements);
    }
}

// Optimized version using two-pass algorithm (more parallel but requires two passes)
template<typename T>
__global__ void batchnorm_mean_kernel(const T* input,
                                      T* mean,
                                      int64_t N,
                                      int64_t C,
                                      int64_t H,
                                      int64_t W) {
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* shared = reinterpret_cast<T*>(shared_mem);

    int64_t spatial_size = H * W;
    int64_t total_elements = N * spatial_size;

    // Each block handles one channel
    int64_t c = blockIdx.x;
    if (c >= C) return;

    // Compute sum
    T sum = T(0);
    for (int64_t n = 0; n < N; n++) {
        for (int64_t idx = threadIdx.x; idx < spatial_size; idx += blockDim.x) {
            int64_t h = idx / W;
            int64_t w = idx % W;
            int64_t tensor_idx = ((n * C + c) * H + h) * W + w;
            sum += input[tensor_idx];
        }
    }

    // Reduce sum across threads
    sum = block_reduce_sum(sum, shared);

    // Thread 0 writes mean
    if (threadIdx.x == 0) {
        mean[c] = sum / T(total_elements);
    }
}

template<typename T>
__global__ void batchnorm_variance_kernel(const T* input,
                                          const T* mean,
                                          T* variance,
                                          int64_t N,
                                          int64_t C,
                                          int64_t H,
                                          int64_t W) {
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* shared = reinterpret_cast<T*>(shared_mem);

    int64_t spatial_size = H * W;
    int64_t total_elements = N * spatial_size;

    // Each block handles one channel
    int64_t c = blockIdx.x;
    if (c >= C) return;

    T channel_mean = mean[c];

    // Compute sum of squared differences
    T sum_sq_diff = T(0);
    for (int64_t n = 0; n < N; n++) {
        for (int64_t idx = threadIdx.x; idx < spatial_size; idx += blockDim.x) {
            int64_t h = idx / W;
            int64_t w = idx % W;
            int64_t tensor_idx = ((n * C + c) * H + h) * W + w;
            T diff = input[tensor_idx] - channel_mean;
            sum_sq_diff += diff * diff;
        }
    }

    // Reduce sum across threads
    sum_sq_diff = block_reduce_sum(sum_sq_diff, shared);

    // Thread 0 writes variance
    if (threadIdx.x == 0) {
        variance[c] = sum_sq_diff / T(total_elements);
    }
}

// ============================================================================
// BatchNorm2d Normalization Kernel
// ============================================================================

// Normalize: (x - mean) / sqrt(variance + epsilon)
template<typename T>
__global__ void batchnorm_normalize_kernel(const T* input,
                                           T* output,
                                           const T* mean,
                                           const T* variance,
                                           T epsilon,
                                           int64_t N,
                                           int64_t C,
                                           int64_t H,
                                           int64_t W) {
    int64_t spatial_size = H * W;
    int64_t total_size = N * C * spatial_size;

    CUDA_GRID_STRIDE_LOOP(idx, total_size) {
        // Decode NCHW index
        int64_t w = idx % W;
        int64_t h = (idx / W) % H;
        int64_t c = (idx / (W * H)) % C;
        int64_t n = idx / (C * W * H);

        T channel_mean = mean[c];
        T channel_var = variance[c];
        T invstd = rsqrt(channel_var + epsilon);  // 1/sqrt(var + eps)

        output[idx] = (input[idx] - channel_mean) * invstd;
    }
}

// Float16 specialization: compute normalization in Float32 to prevent overflow
__global__ void batchnorm_normalize_fp16_kernel(const __half* input,
                                                 __half* output,
                                                 const __half* mean,
                                                 const __half* variance,
                                                 float epsilon,
                                                 int64_t N,
                                                 int64_t C,
                                                 int64_t H,
                                                 int64_t W) {
    int64_t spatial_size = H * W;
    int64_t total_size = N * C * spatial_size;

    CUDA_GRID_STRIDE_LOOP(idx, total_size) {
        int64_t c = (idx / (W * H)) % C;

        float channel_mean = __half2float(mean[c]);
        float channel_var = __half2float(variance[c]);
        float invstd = rsqrtf(channel_var + epsilon);

        float result = (__half2float(input[idx]) - channel_mean) * invstd;
        output[idx] = __float2half_sat(result);
    }
}

// ============================================================================
// BatchNorm2d Affine Transform Kernel
// ============================================================================

// Apply affine transform: y = gamma * normalized + beta
template<typename T>
__global__ void batchnorm_affine_kernel(const T* normalized,
                                        T* output,
                                        const T* gamma,
                                        const T* beta,
                                        int64_t N,
                                        int64_t C,
                                        int64_t H,
                                        int64_t W) {
    int64_t spatial_size = H * W;
    int64_t total_size = N * C * spatial_size;

    CUDA_GRID_STRIDE_LOOP(idx, total_size) {
        // Decode NCHW index
        int64_t c = (idx / (H * W)) % C;

        output[idx] = gamma[c] * normalized[idx] + beta[c];
    }
}

// Combined normalization + affine (more efficient)
template<typename T>
__global__ void batchnorm_forward_affine_kernel(const T* input,
                                                T* output,
                                                const T* mean,
                                                const T* variance,
                                                const T* gamma,
                                                const T* beta,
                                                T epsilon,
                                                int64_t N,
                                                int64_t C,
                                                int64_t H,
                                                int64_t W) {
    int64_t spatial_size = H * W;
    int64_t total_size = N * C * spatial_size;

    CUDA_GRID_STRIDE_LOOP(idx, total_size) {
        // Decode NCHW index
        int64_t c = (idx / (H * W)) % C;

        T channel_mean = mean[c];
        T channel_var = variance[c];
        T invstd = rsqrt(channel_var + epsilon);

        T normalized = (input[idx] - channel_mean) * invstd;
        output[idx] = gamma[c] * normalized + beta[c];
    }
}

// Float16 specialization: compute in Float32 internally to prevent overflow
__global__ void batchnorm_forward_affine_fp16_kernel(const __half* input,
                                                      __half* output,
                                                      const __half* mean,
                                                      const __half* variance,
                                                      const __half* gamma,
                                                      const __half* beta,
                                                      float epsilon,
                                                      int64_t N,
                                                      int64_t C,
                                                      int64_t H,
                                                      int64_t W) {
    int64_t spatial_size = H * W;
    int64_t total_size = N * C * spatial_size;

    CUDA_GRID_STRIDE_LOOP(idx, total_size) {
        int64_t c = (idx / (H * W)) % C;

        float channel_mean = __half2float(mean[c]);
        float channel_var = __half2float(variance[c]);
        float invstd = rsqrtf(channel_var + epsilon);

        float normalized = (__half2float(input[idx]) - channel_mean) * invstd;
        float result = __half2float(gamma[c]) * normalized + __half2float(beta[c]);
        output[idx] = __float2half_sat(result);
    }
}

// ============================================================================
// Optimized BatchNorm Inference Kernel (Single-pass with shared memory)
// ============================================================================

// Ultra-optimized single kernel for BatchNorm inference
// Uses shared memory to cache per-channel scale/bias, avoids multiple kernel launches
// Block processes one channel at a time for optimal memory access
template<int BLOCK_SIZE_OPT = 256>
__global__ void batchnorm_forward_affine_fast_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ mean,
    const float* __restrict__ variance,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float epsilon,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W) {

    // Shared memory for this channel's parameters (computed once per block)
    __shared__ float s_scale;
    __shared__ float s_bias;

    int64_t spatial_size = H * W;
    int64_t elements_per_channel = N * spatial_size;

    // Each block handles one channel
    int64_t c = blockIdx.x;
    if (c >= C) return;

    // Thread 0 computes and stores the scale/bias for this channel
    if (threadIdx.x == 0) {
        float invstd = rsqrtf(variance[c] + epsilon);
        s_scale = gamma[c] * invstd;
        s_bias = beta[c] - mean[c] * s_scale;
    }
    __syncthreads();

    // Load scale/bias into registers for fast access
    float scale = s_scale;
    float bias = s_bias;

    // Process all elements for this channel with grid-stride loop
    for (int64_t i = threadIdx.x; i < elements_per_channel; i += BLOCK_SIZE_OPT) {
        // Compute actual tensor index: for NCHW, we need (n, c, h, w)
        // where n = i / spatial_size, and hw_idx = i % spatial_size
        int64_t n = i / spatial_size;
        int64_t hw_idx = i % spatial_size;
        int64_t idx = ((n * C + c) * H * W) + hw_idx;

        output[idx] = __fmaf_rn(input[idx], scale, bias);  // fused multiply-add
    }
}

// Alternative: process all elements in parallel (better for large tensors)
__global__ void batchnorm_forward_affine_parallel_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ mean,
    const float* __restrict__ variance,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float epsilon,
    int64_t total_size,
    int64_t C,
    int64_t HW) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    // Decode channel from NCHW layout
    int64_t c = (idx / HW) % C;

    // Compute scale and bias inline (register pressure is fine, avoids shared memory sync)
    float invstd = rsqrtf(variance[c] + epsilon);
    float scale = gamma[c] * invstd;
    float val = (input[idx] - mean[c]) * scale + beta[c];
    output[idx] = val;
}

// Vectorized version: processes 4 elements per thread
__global__ void batchnorm_forward_affine_vec4_inline_kernel(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const float* __restrict__ mean,
    const float* __restrict__ variance,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float epsilon,
    int64_t total_vec4,
    int64_t C,
    int64_t HW) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_vec4) return;

    // Decode channel from NCHW layout (for vec4, multiply by 4)
    int64_t scalar_idx = idx * 4;
    int64_t c = (scalar_idx / HW) % C;

    // Load parameters for this channel
    float m = mean[c];
    float invstd = rsqrtf(variance[c] + epsilon);
    float g = gamma[c];
    float b = beta[c];

    // Load 4 elements
    float4 in = input[idx];

    // Apply transformation: y = gamma * (x - mean) * invstd + beta
    float4 out;
    out.x = g * (in.x - m) * invstd + b;
    out.y = g * (in.y - m) * invstd + b;
    out.z = g * (in.z - m) * invstd + b;
    out.w = g * (in.w - m) * invstd + b;

    output[idx] = out;
}

// ============================================================================
// BatchNorm2d Running Statistics Update Kernel
// ============================================================================

// Update running statistics: running = (1 - momentum) * running + momentum * batch
template<typename T>
__global__ void batchnorm_update_running_stats_kernel(T* running_mean,
                                                      T* running_var,
                                                      const T* batch_mean,
                                                      const T* batch_var,
                                                      T momentum,
                                                      int64_t C) {
    CUDA_GRID_STRIDE_LOOP(c, C) {
        running_mean[c] = (T(1) - momentum) * running_mean[c] + momentum * batch_mean[c];
        running_var[c] = (T(1) - momentum) * running_var[c] + momentum * batch_var[c];
    }
}

// ============================================================================
// BatchNorm2d Backward Kernels
// ============================================================================

// Compute gradients w.r.t gamma and beta
// grad_gamma = sum(grad_output * normalized)
// grad_beta = sum(grad_output)
template<typename T>
__global__ void batchnorm_backward_gamma_beta_kernel(const T* grad_output,
                                                     const T* normalized,
                                                     T* grad_gamma,
                                                     T* grad_beta,
                                                     int64_t N,
                                                     int64_t C,
                                                     int64_t H,
                                                     int64_t W) {
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* shared = reinterpret_cast<T*>(shared_mem);

    int64_t spatial_size = H * W;

    // Each block handles one channel
    int64_t c = blockIdx.x;
    if (c >= C) return;

    T sum_grad_gamma = T(0);
    T sum_grad_beta = T(0);

    // Each thread accumulates partial sums
    for (int64_t n = 0; n < N; n++) {
        for (int64_t idx = threadIdx.x; idx < spatial_size; idx += blockDim.x) {
            int64_t h = idx / W;
            int64_t w = idx % W;
            int64_t tensor_idx = ((n * C + c) * H + h) * W + w;

            T grad_out = grad_output[tensor_idx];
            T norm = normalized[tensor_idx];

            sum_grad_gamma += grad_out * norm;
            sum_grad_beta += grad_out;
        }
    }

    // Reduce across threads
    sum_grad_gamma = block_reduce_sum(sum_grad_gamma, shared);
    __syncthreads();
    sum_grad_beta = block_reduce_sum(sum_grad_beta, shared);

    // Thread 0 writes results
    if (threadIdx.x == 0) {
        grad_gamma[c] = sum_grad_gamma;
        grad_beta[c] = sum_grad_beta;
    }
}

// Compute gradient w.r.t input
// Efficient formulation: grad_input = gamma * invstd * (grad_output - mean(grad_output) - normalized * mean(grad_output * normalized))
template<typename T>
__global__ void batchnorm_backward_input_kernel(const T* grad_output,
                                                const T* input,
                                                T* grad_input,
                                                const T* mean,
                                                const T* variance,
                                                const T* gamma,
                                                T epsilon,
                                                int64_t N,
                                                int64_t C,
                                                int64_t H,
                                                int64_t W) {
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* shared = reinterpret_cast<T*>(shared_mem);

    int64_t spatial_size = H * W;
    int64_t total_elements = N * spatial_size;

    // Each block handles one channel
    int64_t c = blockIdx.x;
    if (c >= C) return;

    T channel_mean = mean[c];
    T channel_var = variance[c];
    T invstd = rsqrt(channel_var + epsilon);
    T channel_gamma = gamma[c];

    // Compute auxiliary statistics
    // sum_grad = sum(grad_output)
    // sum_grad_normalized = sum(grad_output * normalized)
    T sum_grad = T(0);
    T sum_grad_norm = T(0);

    for (int64_t n = 0; n < N; n++) {
        for (int64_t idx = threadIdx.x; idx < spatial_size; idx += blockDim.x) {
            int64_t h = idx / W;
            int64_t w = idx % W;
            int64_t tensor_idx = ((n * C + c) * H + h) * W + w;

            T grad_out = grad_output[tensor_idx];
            T normalized = (input[tensor_idx] - channel_mean) * invstd;

            sum_grad += grad_out;
            sum_grad_norm += grad_out * normalized;
        }
    }

    // Reduce across threads
    sum_grad = block_reduce_sum(sum_grad, shared);
    __syncthreads();
    sum_grad_norm = block_reduce_sum(sum_grad_norm, shared);
    __syncthreads();

    // Broadcast to all threads
    if (threadIdx.x == 0) {
        shared[0] = sum_grad / T(total_elements);
        shared[1] = sum_grad_norm / T(total_elements);
    }
    __syncthreads();
    T mean_grad = shared[0];
    T mean_grad_norm = shared[1];

    // Compute gradient w.r.t input
    for (int64_t n = 0; n < N; n++) {
        for (int64_t idx = threadIdx.x; idx < spatial_size; idx += blockDim.x) {
            int64_t h = idx / W;
            int64_t w = idx % W;
            int64_t tensor_idx = ((n * C + c) * H + h) * W + w;

            T grad_out = grad_output[tensor_idx];
            T normalized = (input[tensor_idx] - channel_mean) * invstd;

            // Efficient backward formulation
            T grad_normalized = grad_out - mean_grad - normalized * mean_grad_norm;
            grad_input[tensor_idx] = channel_gamma * invstd * grad_normalized;
        }
    }
}

// ============================================================================
// Host Functions (C++ API)
// ============================================================================

// Compute mean and variance for a batch
auto batchnorm2d_mean_var(const Tensor& input,
                          Tensor& mean,
                          Tensor& variance,
                          cudaStream_t stream) -> void {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    // Check for division by zero
    int64_t total_elements = N * H * W;
    if (total_elements == 0) {
        throw std::runtime_error("BatchNorm2d CUDA: Cannot compute mean/variance for empty tensor (N*H*W = 0)");
    }

    if (input.dtype() == DType::Float32) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / 32) * sizeof(float);
        batchnorm_mean_kernel<float><<<C, BATCHNORM_BLOCK_SIZE, shared_mem_size, stream>>>(
            input.data<float>(), mean.data<float>(), N, C, H, W);
        CUDA_CHECK(cudaGetLastError());

        batchnorm_variance_kernel<float><<<C, BATCHNORM_BLOCK_SIZE, shared_mem_size, stream>>>(
            input.data<float>(), mean.data<float>(), variance.data<float>(), N, C, H, W);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / 32) * sizeof(double);
        batchnorm_mean_kernel<double><<<C, BATCHNORM_BLOCK_SIZE, shared_mem_size, stream>>>(
            input.data<double>(), mean.data<double>(), N, C, H, W);
        CUDA_CHECK(cudaGetLastError());

        batchnorm_variance_kernel<double><<<C, BATCHNORM_BLOCK_SIZE, shared_mem_size, stream>>>(
            input.data<double>(), mean.data<double>(), variance.data<double>(), N, C, H, W);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / 32) * sizeof(__half);
        batchnorm_mean_kernel<__half><<<C, BATCHNORM_BLOCK_SIZE, shared_mem_size, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(mean.data<Float16>()), N, C, H, W);
        CUDA_CHECK(cudaGetLastError());

        batchnorm_variance_kernel<__half><<<C, BATCHNORM_BLOCK_SIZE, shared_mem_size, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<const __half*>(mean.data<Float16>()),
            reinterpret_cast<__half*>(variance.data<Float16>()), N, C, H, W);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("BatchNorm2D only supports Float32, Float64, and Float16 dtypes");
    }
}

// Forward pass (normalization only, no affine)
auto batchnorm2d_forward(const Tensor& input,
                         const Tensor& mean,
                         const Tensor& variance,
                         float epsilon,
                         cudaStream_t stream) -> Tensor {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];
    int64_t total_size = N * C * H * W;

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(total_size);
        batchnorm_normalize_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), output.data<float>(),
            mean.data<float>(), variance.data<float>(),
            epsilon, N, C, H, W);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(total_size);
        batchnorm_normalize_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), output.data<double>(),
            mean.data<double>(), variance.data<double>(),
            epsilon, N, C, H, W);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(total_size);
        batchnorm_normalize_fp16_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            reinterpret_cast<const __half*>(mean.data<Float16>()),
            reinterpret_cast<const __half*>(variance.data<Float16>()),
            epsilon, N, C, H, W);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("BatchNorm2D only supports Float32, Float64, and Float16 dtypes");
    }

    return output;
}

// Forward pass with affine transform
auto batchnorm2d_forward_affine(const Tensor& input,
                                const Tensor& mean,
                                const Tensor& variance,
                                const Tensor& gamma,
                                const Tensor& beta,
                                float epsilon,
                                cudaStream_t stream) -> Tensor {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];
    int64_t total_size = N * C * H * W;

    if (input.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(total_size);
        batchnorm_forward_affine_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), output.data<float>(),
            mean.data<float>(), variance.data<float>(),
            gamma.data<float>(), beta.data<float>(),
            epsilon, N, C, H, W);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(total_size);
        batchnorm_forward_affine_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), output.data<double>(),
            mean.data<double>(), variance.data<double>(),
            gamma.data<double>(), beta.data<double>(),
            epsilon, N, C, H, W);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(total_size);
        batchnorm_forward_affine_fp16_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            reinterpret_cast<const __half*>(mean.data<Float16>()),
            reinterpret_cast<const __half*>(variance.data<Float16>()),
            reinterpret_cast<const __half*>(gamma.data<Float16>()),
            reinterpret_cast<const __half*>(beta.data<Float16>()),
            epsilon, N, C, H, W);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("BatchNorm2D only supports Float32, Float64, and Float16 dtypes");
    }

    return output;
}

// Optimized forward pass with affine transform (single kernel, vectorized for large tensors)
// Uses inline computation to avoid kernel launch overhead
auto batchnorm2d_forward_affine_optimized(const Tensor& input,
                                          const Tensor& mean,
                                          const Tensor& variance,
                                          const Tensor& gamma,
                                          const Tensor& beta,
                                          float epsilon,
                                          cudaStream_t stream) -> Tensor {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];
    int64_t HW = H * W;
    int64_t total_size = N * C * HW;

    if (input.dtype() == DType::Float32) {
        // Check if we can use vectorized path (need alignment and multiple of 4)
        bool can_vectorize = (total_size % 4 == 0) &&
                             (reinterpret_cast<uintptr_t>(input.data<float>()) % 16 == 0) &&
                             (reinterpret_cast<uintptr_t>(output.data<float>()) % 16 == 0);

        if (can_vectorize && total_size >= 1024) {
            // Use vectorized kernel for large tensors
            int64_t total_vec4 = total_size / 4;
            int vec_blocks = (total_vec4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
            batchnorm_forward_affine_vec4_inline_kernel<<<vec_blocks, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const float4*>(input.data<float>()),
                reinterpret_cast<float4*>(output.data<float>()),
                mean.data<float>(), variance.data<float>(),
                gamma.data<float>(), beta.data<float>(),
                epsilon, total_vec4, C, HW);
            CUDA_CHECK(cudaGetLastError());
        } else {
            // Use parallel kernel (single kernel launch, processes all elements)
            int num_blocks = get_num_blocks(total_size);
            batchnorm_forward_affine_parallel_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                input.data<float>(), output.data<float>(),
                mean.data<float>(), variance.data<float>(),
                gamma.data<float>(), beta.data<float>(),
                epsilon, total_size, C, HW);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (input.dtype() == DType::Float64) {
        // For Float64, use the standard kernel
        int num_blocks = get_num_blocks(total_size);
        batchnorm_forward_affine_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), output.data<double>(),
            mean.data<double>(), variance.data<double>(),
            gamma.data<double>(), beta.data<double>(),
            epsilon, N, C, H, W);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        // For Float16, use FP16 kernel that computes in Float32 for numerical stability
        int num_blocks = get_num_blocks(total_size);
        batchnorm_forward_affine_fp16_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            reinterpret_cast<const __half*>(mean.data<Float16>()),
            reinterpret_cast<const __half*>(variance.data<Float16>()),
            reinterpret_cast<const __half*>(gamma.data<Float16>()),
            reinterpret_cast<const __half*>(beta.data<Float16>()),
            epsilon, N, C, H, W);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("BatchNorm2D only supports Float32, Float64, and Float16 dtypes");
    }

    return output;
}

// Update running statistics
auto batchnorm2d_update_running_stats(Tensor& running_mean,
                                      Tensor& running_var,
                                      const Tensor& batch_mean,
                                      const Tensor& batch_var,
                                      float momentum,
                                      cudaStream_t stream) -> void {
    int64_t C = batch_mean.shape()[0];

    if (running_mean.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(C);
        batchnorm_update_running_stats_kernel<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            running_mean.data<float>(), running_var.data<float>(),
            batch_mean.data<float>(), batch_var.data<float>(),
            momentum, C);
        CUDA_CHECK(cudaGetLastError());
    } else if (running_mean.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(C);
        batchnorm_update_running_stats_kernel<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            running_mean.data<double>(), running_var.data<double>(),
            batch_mean.data<double>(), batch_var.data<double>(),
            momentum, C);
        CUDA_CHECK(cudaGetLastError());
    } else if (running_mean.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(C);
        batchnorm_update_running_stats_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<__half*>(running_mean.data<Float16>()),
            reinterpret_cast<__half*>(running_var.data<Float16>()),
            reinterpret_cast<const __half*>(batch_mean.data<Float16>()),
            reinterpret_cast<const __half*>(batch_var.data<Float16>()),
            __float2half(momentum), C);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("BatchNorm2D only supports Float32, Float64, and Float16 dtypes");
    }
}

// Backward pass - compute gradients
auto batchnorm2d_backward(const Tensor& grad_output,
                         const Tensor& input,
                         const Tensor& mean,
                         const Tensor& variance,
                         const Tensor& gamma,
                         float epsilon,
                         cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    // Check for division by zero
    int64_t total_elements = N * H * W;
    if (total_elements == 0) {
        throw std::runtime_error("BatchNorm2d CUDA backward: Cannot compute gradients for empty tensor (N*H*W = 0)");
    }

    // Allocate output gradients
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor grad_input(shape_vec, input.dtype(), input.device());
    Tensor grad_gamma({C}, input.dtype(), input.device());
    Tensor grad_beta({C}, input.dtype(), input.device());

    // Compute normalized input for gradient computation
    Tensor normalized = batchnorm2d_forward(input, mean, variance, epsilon, stream);

    if (input.dtype() == DType::Float32) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / 32) * sizeof(float);

        // Compute grad_gamma and grad_beta
        batchnorm_backward_gamma_beta_kernel<float><<<C, BATCHNORM_BLOCK_SIZE, shared_mem_size, stream>>>(
            grad_output.data<float>(), normalized.data<float>(),
            grad_gamma.data<float>(), grad_beta.data<float>(),
            N, C, H, W);
        CUDA_CHECK(cudaGetLastError());

        // Compute grad_input
        batchnorm_backward_input_kernel<float><<<C, BATCHNORM_BLOCK_SIZE, shared_mem_size, stream>>>(
            grad_output.data<float>(), input.data<float>(), grad_input.data<float>(),
            mean.data<float>(), variance.data<float>(), gamma.data<float>(),
            epsilon, N, C, H, W);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / 32) * sizeof(double);

        // Compute grad_gamma and grad_beta
        batchnorm_backward_gamma_beta_kernel<double><<<C, BATCHNORM_BLOCK_SIZE, shared_mem_size, stream>>>(
            grad_output.data<double>(), normalized.data<double>(),
            grad_gamma.data<double>(), grad_beta.data<double>(),
            N, C, H, W);
        CUDA_CHECK(cudaGetLastError());

        // Compute grad_input
        batchnorm_backward_input_kernel<double><<<C, BATCHNORM_BLOCK_SIZE, shared_mem_size, stream>>>(
            grad_output.data<double>(), input.data<double>(), grad_input.data<double>(),
            mean.data<double>(), variance.data<double>(), gamma.data<double>(),
            epsilon, N, C, H, W);
        CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / 32) * sizeof(__half);

        // Compute grad_gamma and grad_beta
        batchnorm_backward_gamma_beta_kernel<__half><<<C, BATCHNORM_BLOCK_SIZE, shared_mem_size, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            reinterpret_cast<const __half*>(normalized.data<Float16>()),
            reinterpret_cast<__half*>(grad_gamma.data<Float16>()),
            reinterpret_cast<__half*>(grad_beta.data<Float16>()),
            N, C, H, W);
        CUDA_CHECK(cudaGetLastError());

        // Compute grad_input
        batchnorm_backward_input_kernel<__half><<<C, BATCHNORM_BLOCK_SIZE, shared_mem_size, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(grad_input.data<Float16>()),
            reinterpret_cast<const __half*>(mean.data<Float16>()),
            reinterpret_cast<const __half*>(variance.data<Float16>()),
            reinterpret_cast<const __half*>(gamma.data<Float16>()),
            __float2half(epsilon), N, C, H, W);
        CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("BatchNorm2D only supports Float32, Float64, and Float16 dtypes");
    }

    return std::make_tuple(grad_input, grad_gamma, grad_beta);
}


// ============================================================================
// GroupNorm CUDA Kernels
// ============================================================================

template<typename T>
__global__ void group_norm_forward_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    T* __restrict__ mean_out,
    T* __restrict__ inv_std_out,
    int64_t N, int64_t C, int64_t HW,
    int64_t num_groups, int64_t channels_per_group,
    float eps) {

    // Each block handles one (sample, group) pair
    int64_t group_idx = blockIdx.x;
    int64_t n = group_idx / num_groups;
    int64_t g = group_idx % num_groups;

    if (n >= N || g >= num_groups) return;

    int64_t c_start = g * channels_per_group;
    int64_t group_size = channels_per_group * HW;

    // Compute mean using parallel reduction
    T local_sum = T(0);
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int64_t c_offset = i / HW;
        int64_t hw = i % HW;
        int64_t c = c_start + c_offset;
        int64_t idx = (n * C + c) * HW + hw;
        local_sum += input[idx];
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    // Shared memory for inter-warp reduction
    __shared__ T shared_sum[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    // Final reduction in first warp
    T mean = T(0);
    if (warp_id == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_sum = (lane < num_warps) ? shared_sum[lane] : T(0);
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        if (lane == 0) {
            mean = local_sum / T(group_size);
            shared_sum[0] = mean;
        }
    }
    __syncthreads();
    mean = shared_sum[0];

    // Compute variance
    T local_var = T(0);
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int64_t c_offset = i / HW;
        int64_t hw = i % HW;
        int64_t c = c_start + c_offset;
        int64_t idx = (n * C + c) * HW + hw;
        T diff = input[idx] - mean;
        local_var += diff * diff;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_var += __shfl_down_sync(0xFFFFFFFF, local_var, offset);
    }

    if (lane == 0) shared_sum[warp_id] = local_var;
    __syncthreads();

    T inv_std = T(0);
    if (warp_id == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_var = (lane < num_warps) ? shared_sum[lane] : T(0);
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_var += __shfl_down_sync(0xFFFFFFFF, local_var, offset);
        }
        if (lane == 0) {
            T variance = local_var / T(group_size);
            inv_std = T(1) / sqrt(variance + T(eps));
            shared_sum[0] = inv_std;
            if (mean_out) mean_out[group_idx] = mean;
            if (inv_std_out) inv_std_out[group_idx] = inv_std;
        }
    }
    __syncthreads();
    inv_std = shared_sum[0];

    // Normalize and apply affine
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int64_t c_offset = i / HW;
        int64_t hw = i % HW;
        int64_t c = c_start + c_offset;
        int64_t idx = (n * C + c) * HW + hw;
        T normalized = (input[idx] - mean) * inv_std;
        if (weight && bias) {
            output[idx] = normalized * weight[c] + bias[c];
        } else {
            output[idx] = normalized;
        }
    }
}

template<typename T>
__global__ void group_norm_backward_kernel(
    const T* __restrict__ grad_output,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ mean_saved,
    const T* __restrict__ inv_std_saved,
    T* __restrict__ grad_input,
    T* __restrict__ grad_weight,
    T* __restrict__ grad_bias,
    int64_t N, int64_t C, int64_t HW,
    int64_t num_groups, int64_t channels_per_group) {

    int64_t group_idx = blockIdx.x;
    int64_t n = group_idx / num_groups;
    int64_t g = group_idx % num_groups;

    if (n >= N || g >= num_groups) return;

    int64_t c_start = g * channels_per_group;
    int64_t group_size = channels_per_group * HW;

    T mean = mean_saved[group_idx];
    T inv_std = inv_std_saved[group_idx];

    // Compute sum of grad_output * normalized and sum of grad_output
    T local_sum_dy = T(0);
    T local_sum_dy_xhat = T(0);

    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int64_t c_offset = i / HW;
        int64_t hw = i % HW;
        int64_t c = c_start + c_offset;
        int64_t idx = (n * C + c) * HW + hw;
        T dy = grad_output[idx];
        if (weight) dy = dy * weight[c];
        T xhat = (input[idx] - mean) * inv_std;
        local_sum_dy += dy;
        local_sum_dy_xhat += dy * xhat;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum_dy += __shfl_down_sync(0xFFFFFFFF, local_sum_dy, offset);
        local_sum_dy_xhat += __shfl_down_sync(0xFFFFFFFF, local_sum_dy_xhat, offset);
    }

    __shared__ T shared_dy[32];
    __shared__ T shared_dy_xhat[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        shared_dy[warp_id] = local_sum_dy;
        shared_dy_xhat[warp_id] = local_sum_dy_xhat;
    }
    __syncthreads();

    T sum_dy, sum_dy_xhat;
    if (warp_id == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_sum_dy = (lane < num_warps) ? shared_dy[lane] : T(0);
        local_sum_dy_xhat = (lane < num_warps) ? shared_dy_xhat[lane] : T(0);
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_sum_dy += __shfl_down_sync(0xFFFFFFFF, local_sum_dy, offset);
            local_sum_dy_xhat += __shfl_down_sync(0xFFFFFFFF, local_sum_dy_xhat, offset);
        }
        if (lane == 0) {
            shared_dy[0] = local_sum_dy;
            shared_dy_xhat[0] = local_sum_dy_xhat;
        }
    }
    __syncthreads();
    sum_dy = shared_dy[0];
    sum_dy_xhat = shared_dy_xhat[0];

    T inv_group_size = T(1) / T(group_size);

    // Compute grad_input
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int64_t c_offset = i / HW;
        int64_t hw = i % HW;
        int64_t c = c_start + c_offset;
        int64_t idx = (n * C + c) * HW + hw;
        T dy = grad_output[idx];
        if (weight) dy = dy * weight[c];
        T xhat = (input[idx] - mean) * inv_std;
        grad_input[idx] = inv_std * (dy - inv_group_size * (sum_dy + xhat * sum_dy_xhat));
    }

    // Accumulate grad_weight and grad_bias (atomic since multiple samples contribute)
    if (weight && grad_weight && grad_bias) {
        for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
            int64_t c_offset = i / HW;
            int64_t hw = i % HW;
            int64_t c = c_start + c_offset;
            int64_t idx = (n * C + c) * HW + hw;
            T xhat = (input[idx] - mean) * inv_std;
            atomicAdd(&grad_weight[c], grad_output[idx] * xhat);
            atomicAdd(&grad_bias[c], grad_output[idx]);
        }
    }
}

// GroupNorm forward launcher
auto group_norm_forward_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t num_groups,
    float eps,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t HW = 1;
    for (size_t i = 2; i < shape.size(); ++i) HW *= shape[i];
    int64_t channels_per_group = C / num_groups;

    Tensor output(shape, input.dtype(), input.device());
    Tensor mean_out({N * num_groups}, input.dtype(), input.device());
    Tensor inv_std_out({N * num_groups}, input.dtype(), input.device());

    int64_t num_group_instances = N * num_groups;
    int block_size = 256;

    if (input.dtype() == DType::Float32) {
        group_norm_forward_kernel<float><<<num_group_instances, block_size, 0, stream>>>(
            input.data<float>(), weight.data<float>(), bias.data<float>(),
            output.data<float>(), mean_out.data<float>(), inv_std_out.data<float>(),
            N, C, HW, num_groups, channels_per_group, eps);
    } else if (input.dtype() == DType::Float64) {
        group_norm_forward_kernel<double><<<num_group_instances, block_size, 0, stream>>>(
            input.data<double>(), weight.data<double>(), bias.data<double>(),
            output.data<double>(), mean_out.data<double>(), inv_std_out.data<double>(),
            N, C, HW, num_groups, channels_per_group, eps);
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        // Mixed precision: compute in Float32, convert back
        // Create Float32 temporaries
        Tensor input_f32 = input.to(DType::Float32);
        Tensor weight_f32 = weight.to(DType::Float32);
        Tensor bias_f32 = bias.to(DType::Float32);
        Tensor output_f32(shape, DType::Float32, input.device());
        Tensor mean_f32({N * num_groups}, DType::Float32, input.device());
        Tensor inv_std_f32({N * num_groups}, DType::Float32, input.device());

        group_norm_forward_kernel<float><<<num_group_instances, block_size, 0, stream>>>(
            input_f32.data<float>(), weight_f32.data<float>(), bias_f32.data<float>(),
            output_f32.data<float>(), mean_f32.data<float>(), inv_std_f32.data<float>(),
            N, C, HW, num_groups, channels_per_group, eps);

        CUDA_CHECK(cudaGetLastError());

        // Convert outputs back to original dtype
        output = output_f32.to(input.dtype());
        mean_out = mean_f32.to(input.dtype());
        inv_std_out = inv_std_f32.to(input.dtype());
        return {output, mean_out, inv_std_out};
    } else {
        throw std::runtime_error("group_norm_forward: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return {output, mean_out, inv_std_out};
}

// GroupNorm backward launcher
auto group_norm_backward_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& mean_saved,
    const Tensor& inv_std_saved,
    int64_t num_groups,
    cudaStream_t stream
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

    // Zero-initialize grad_weight and grad_bias (atomicAdd accumulation)
    cudaMemsetAsync(grad_weight.data_ptr(), 0, C * dtype_size(input.dtype()), stream);
    cudaMemsetAsync(grad_bias.data_ptr(), 0, C * dtype_size(input.dtype()), stream);

    int64_t num_group_instances = N * num_groups;
    int block_size = 256;

    if (input.dtype() == DType::Float32) {
        group_norm_backward_kernel<float><<<num_group_instances, block_size, 0, stream>>>(
            grad_output.data<float>(), input.data<float>(), weight.data<float>(),
            mean_saved.data<float>(), inv_std_saved.data<float>(),
            grad_input.data<float>(), grad_weight.data<float>(), grad_bias.data<float>(),
            N, C, HW, num_groups, channels_per_group);
    } else if (input.dtype() == DType::Float64) {
        group_norm_backward_kernel<double><<<num_group_instances, block_size, 0, stream>>>(
            grad_output.data<double>(), input.data<double>(), weight.data<double>(),
            mean_saved.data<double>(), inv_std_saved.data<double>(),
            grad_input.data<double>(), grad_weight.data<double>(), grad_bias.data<double>(),
            N, C, HW, num_groups, channels_per_group);
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        // Mixed precision: compute in Float32, convert back
        Tensor grad_out_f32 = grad_output.to(DType::Float32);
        Tensor input_f32 = input.to(DType::Float32);
        Tensor weight_f32 = weight.to(DType::Float32);
        Tensor mean_f32 = mean_saved.to(DType::Float32);
        Tensor inv_std_f32 = inv_std_saved.to(DType::Float32);

        Tensor grad_input_f32(shape, DType::Float32, input.device());
        Tensor grad_weight_f32({C}, DType::Float32, input.device());
        Tensor grad_bias_f32({C}, DType::Float32, input.device());

        cudaMemsetAsync(grad_weight_f32.data_ptr(), 0, C * sizeof(float), stream);
        cudaMemsetAsync(grad_bias_f32.data_ptr(), 0, C * sizeof(float), stream);

        group_norm_backward_kernel<float><<<num_group_instances, block_size, 0, stream>>>(
            grad_out_f32.data<float>(), input_f32.data<float>(), weight_f32.data<float>(),
            mean_f32.data<float>(), inv_std_f32.data<float>(),
            grad_input_f32.data<float>(), grad_weight_f32.data<float>(), grad_bias_f32.data<float>(),
            N, C, HW, num_groups, channels_per_group);

        CUDA_CHECK(cudaGetLastError());

        // Convert outputs back to original dtype
        grad_input = grad_input_f32.to(input.dtype());
        grad_weight = grad_weight_f32.to(input.dtype());
        grad_bias = grad_bias_f32.to(input.dtype());
        return {grad_input, grad_weight, grad_bias};
    } else {
        throw std::runtime_error("group_norm_backward: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return {grad_input, grad_weight, grad_bias};
}

// ============================================================================
// InstanceNorm CUDA Kernels (delegates to GroupNorm with groups=C)
// ============================================================================

auto instance_norm_forward_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    float eps,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    int64_t C = input.shape()[1];
    return group_norm_forward_kernel(input, weight, bias, C, eps, stream);
}

auto instance_norm_backward_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& mean_saved,
    const Tensor& inv_std_saved,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    int64_t C = input.shape()[1];
    return group_norm_backward_kernel(grad_output, input, weight, mean_saved, inv_std_saved, C, stream);
}
} // namespace cuda
} // namespace tenzor
