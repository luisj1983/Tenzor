#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
    } else {
        throw std::runtime_error("BatchNorm2D only supports Float32 and Float64 dtypes");
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
    } else {
        throw std::runtime_error("BatchNorm2D only supports Float32 and Float64 dtypes");
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
    } else {
        throw std::runtime_error("BatchNorm2D only supports Float32 and Float64 dtypes");
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
    } else {
        throw std::runtime_error("BatchNorm2D only supports Float32 and Float64 dtypes");
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
    } else {
        throw std::runtime_error("BatchNorm2D only supports Float32 and Float64 dtypes");
    }

    return std::make_tuple(grad_input, grad_gamma, grad_beta);
}

} // namespace cuda
} // namespace tenzor
