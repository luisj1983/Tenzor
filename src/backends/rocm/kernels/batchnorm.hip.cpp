#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor {
namespace rocm {

// ============================================================================
// HIP Helper Functions
// ============================================================================

// Error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            throw std::runtime_error(std::string("HIP error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + ": " + hipGetErrorString(err)); \
        } \
    } while(0)

// Grid-stride loop helper
#define HIP_GRID_STRIDE_LOOP(i, n) \
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

// Minimum wavefront size across AMD GPUs (RDNA 3/4 = 32, CDNA = 64).
// Used for host-side shared memory allocation — must be the SMALLEST possible
// to ensure enough shared memory for the most warps per block.
// Device code uses the HIP built-in `warpSize` instead.
constexpr int MIN_WAVEFRONT_SIZE = 32;

// Warp-level reduction using shuffle instructions
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
template<typename T>
__device__ T block_reduce_sum(T val, T* shared) {
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : T(0);
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

// ============================================================================
// BatchNorm2d Mean/Variance Computation (Two-Pass Algorithm)
// ============================================================================

// Compute per-channel mean and variance using two-pass algorithm
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
    HIP_DYNAMIC_SHARED(unsigned char, shared_mem);
    T* shared = reinterpret_cast<T*>(shared_mem);

    int64_t spatial_size = H * W;
    int64_t total_elements = N * spatial_size;

    // Each block handles one channel
    int64_t c = blockIdx.x;
    if (c >= C) return;

    // Pass 1: Compute sum for mean
    T sum = T(0);
    for (int64_t n = 0; n < N; n++) {
        for (int64_t idx = threadIdx.x; idx < spatial_size; idx += blockDim.x) {
            int64_t h = idx / W;
            int64_t w = idx % W;
            int64_t tensor_idx = ((n * C + c) * H + h) * W + w;
            sum += input[tensor_idx];
        }
    }

    // Reduce sum across threads in block
    sum = block_reduce_sum(sum, shared);
    __syncthreads();

    // Thread 0 computes and stores mean
    if (threadIdx.x == 0) {
        shared[0] = sum / T(total_elements);
    }
    __syncthreads();

    // All threads read the mean
    T channel_mean = shared[0];

    // Pass 2: Compute sum of squared differences for variance
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

    // Reduce sum of squared differences across threads
    sum_sq_diff = block_reduce_sum(sum_sq_diff, shared);
    __syncthreads();

    // Thread 0 writes final results
    if (threadIdx.x == 0) {
        mean[c] = channel_mean;
        variance[c] = sum_sq_diff / T(total_elements);
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
    HIP_DYNAMIC_SHARED(unsigned char, shared_mem);
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
    HIP_DYNAMIC_SHARED(unsigned char, shared_mem);
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

    HIP_GRID_STRIDE_LOOP(idx, total_size) {
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

    HIP_GRID_STRIDE_LOOP(idx, total_size) {
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

    HIP_GRID_STRIDE_LOOP(idx, total_size) {
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
    HIP_GRID_STRIDE_LOOP(c, C) {
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
    HIP_DYNAMIC_SHARED(unsigned char, shared_mem);
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
    HIP_DYNAMIC_SHARED(unsigned char, shared_mem);
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
// LayerNorm Forward/Backward Kernels
// ============================================================================

// LayerNorm: normalize over last dimension
// Input: [..., D] where D is the normalized dimension
template<typename T>
__global__ void layernorm_forward_kernel(const T* input,
                                        T* output,
                                        const T* gamma,
                                        const T* beta,
                                        T epsilon,
                                        int64_t outer_size,
                                        int64_t normalized_size) {
    HIP_DYNAMIC_SHARED(unsigned char, shared_mem);
    T* shared = reinterpret_cast<T*>(shared_mem);

    // Each block handles one instance
    int64_t instance = blockIdx.x;
    if (instance >= outer_size) return;

    int64_t offset = instance * normalized_size;

    // Compute mean
    T sum = T(0);
    for (int64_t i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        sum += input[offset + i];
    }
    sum = block_reduce_sum(sum, shared);
    __syncthreads();

    T mean = sum / T(normalized_size);

    // Broadcast mean
    if (threadIdx.x == 0) {
        shared[0] = mean;
    }
    __syncthreads();
    mean = shared[0];

    // Compute variance
    T var_sum = T(0);
    for (int64_t i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        T diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, shared);
    __syncthreads();

    T variance = var_sum / T(normalized_size);
    T invstd = rsqrt(variance + epsilon);

    // Broadcast invstd
    if (threadIdx.x == 0) {
        shared[0] = invstd;
    }
    __syncthreads();
    invstd = shared[0];

    // Normalize and apply affine
    for (int64_t i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        T normalized = (input[offset + i] - mean) * invstd;
        output[offset + i] = gamma[i] * normalized + beta[i];
    }
}

template<typename T>
__global__ void layernorm_backward_kernel(const T* grad_output,
                                         const T* input,
                                         const T* gamma,
                                         T* grad_input,
                                         T* grad_gamma,
                                         T* grad_beta,
                                         T epsilon,
                                         int64_t outer_size,
                                         int64_t normalized_size) {
    HIP_DYNAMIC_SHARED(unsigned char, shared_mem);
    T* shared = reinterpret_cast<T*>(shared_mem);

    // Each block handles one instance for input grad, or accumulates for gamma/beta grad
    int64_t instance = blockIdx.x;
    if (instance >= outer_size) return;

    int64_t offset = instance * normalized_size;

    // Compute mean
    T sum = T(0);
    for (int64_t i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        sum += input[offset + i];
    }
    sum = block_reduce_sum(sum, shared);
    __syncthreads();
    T mean = sum / T(normalized_size);

    // Compute variance
    T var_sum = T(0);
    for (int64_t i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        T diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, shared);
    __syncthreads();
    T variance = var_sum / T(normalized_size);
    T invstd = rsqrt(variance + epsilon);

    // Compute gradient statistics
    T grad_output_sum = T(0);
    T grad_output_norm_sum = T(0);
    for (int64_t i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        T normalized = (input[offset + i] - mean) * invstd;
        T grad_out = grad_output[offset + i];
        grad_output_sum += grad_out;
        grad_output_norm_sum += grad_out * normalized;
    }
    grad_output_sum = block_reduce_sum(grad_output_sum, shared);
    __syncthreads();
    grad_output_norm_sum = block_reduce_sum(grad_output_norm_sum, shared);
    __syncthreads();

    T mean_grad = grad_output_sum / T(normalized_size);
    T mean_grad_norm = grad_output_norm_sum / T(normalized_size);

    // Compute input gradient
    for (int64_t i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        T normalized = (input[offset + i] - mean) * invstd;
        T grad_out = grad_output[offset + i];
        T grad_normalized = grad_out - mean_grad - normalized * mean_grad_norm;
        grad_input[offset + i] = gamma[i] * invstd * grad_normalized;

        // Accumulate gradients for gamma and beta (requires atomic adds across blocks)
        atomicAdd(&grad_gamma[i], grad_out * normalized);
        atomicAdd(&grad_beta[i], grad_out);
    }
}

// ============================================================================
// InstanceNorm Forward/Backward Kernels
// ============================================================================

// InstanceNorm: normalize over spatial dimensions per channel per instance
// Input: [N, C, H, W]
template<typename T>
__global__ void instancenorm_forward_kernel(const T* input,
                                           T* output,
                                           const T* gamma,
                                           const T* beta,
                                           T epsilon,
                                           int64_t N,
                                           int64_t C,
                                           int64_t H,
                                           int64_t W) {
    HIP_DYNAMIC_SHARED(unsigned char, shared_mem);
    T* shared = reinterpret_cast<T*>(shared_mem);

    int64_t spatial_size = H * W;

    // Each block handles one (N, C) pair
    int64_t nc = blockIdx.x;
    if (nc >= N * C) return;

    int64_t n = nc / C;
    int64_t c = nc % C;
    int64_t offset = (n * C + c) * spatial_size;

    // Compute mean
    T sum = T(0);
    for (int64_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        sum += input[offset + i];
    }
    sum = block_reduce_sum(sum, shared);
    __syncthreads();
    T mean = sum / T(spatial_size);

    // Compute variance
    T var_sum = T(0);
    for (int64_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        T diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, shared);
    __syncthreads();
    T variance = var_sum / T(spatial_size);
    T invstd = rsqrt(variance + epsilon);

    // Normalize and apply affine
    for (int64_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        T normalized = (input[offset + i] - mean) * invstd;
        output[offset + i] = gamma[c] * normalized + beta[c];
    }
}

template<typename T>
__global__ void instancenorm_backward_kernel(const T* grad_output,
                                            const T* input,
                                            const T* gamma,
                                            T* grad_input,
                                            T* grad_gamma,
                                            T* grad_beta,
                                            T epsilon,
                                            int64_t N,
                                            int64_t C,
                                            int64_t H,
                                            int64_t W) {
    HIP_DYNAMIC_SHARED(unsigned char, shared_mem);
    T* shared = reinterpret_cast<T*>(shared_mem);

    int64_t spatial_size = H * W;

    // Each block handles one (N, C) pair
    int64_t nc = blockIdx.x;
    if (nc >= N * C) return;

    int64_t n = nc / C;
    int64_t c = nc % C;
    int64_t offset = (n * C + c) * spatial_size;

    // Compute mean and variance
    T sum = T(0);
    for (int64_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        sum += input[offset + i];
    }
    sum = block_reduce_sum(sum, shared);
    __syncthreads();
    T mean = sum / T(spatial_size);

    T var_sum = T(0);
    for (int64_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        T diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, shared);
    __syncthreads();
    T variance = var_sum / T(spatial_size);
    T invstd = rsqrt(variance + epsilon);

    // Compute gradient statistics
    T grad_output_sum = T(0);
    T grad_output_norm_sum = T(0);
    for (int64_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        T normalized = (input[offset + i] - mean) * invstd;
        T grad_out = grad_output[offset + i];
        grad_output_sum += grad_out;
        grad_output_norm_sum += grad_out * normalized;
    }
    grad_output_sum = block_reduce_sum(grad_output_sum, shared);
    __syncthreads();
    grad_output_norm_sum = block_reduce_sum(grad_output_norm_sum, shared);
    __syncthreads();

    T mean_grad = grad_output_sum / T(spatial_size);
    T mean_grad_norm = grad_output_norm_sum / T(spatial_size);

    // Compute gradients
    for (int64_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        T normalized = (input[offset + i] - mean) * invstd;
        T grad_out = grad_output[offset + i];
        T grad_normalized = grad_out - mean_grad - normalized * mean_grad_norm;
        grad_input[offset + i] = gamma[c] * invstd * grad_normalized;

        // Accumulate gradients for gamma and beta
        atomicAdd(&grad_gamma[c], grad_out * normalized);
        atomicAdd(&grad_beta[c], grad_out);
    }
}

// ============================================================================
// GroupNorm Forward/Backward Kernels
// ============================================================================

// GroupNorm: normalize over channels and spatial dimensions per group
// Input: [N, C, H, W], groups: number of groups
template<typename T>
__global__ void groupnorm_forward_kernel(const T* input,
                                        T* output,
                                        const T* gamma,
                                        const T* beta,
                                        T epsilon,
                                        int64_t N,
                                        int64_t C,
                                        int64_t H,
                                        int64_t W,
                                        int64_t groups) {
    HIP_DYNAMIC_SHARED(unsigned char, shared_mem);
    T* shared = reinterpret_cast<T*>(shared_mem);

    int64_t channels_per_group = C / groups;
    int64_t spatial_size = H * W;
    int64_t group_size = channels_per_group * spatial_size;

    // Each block handles one (N, group) pair
    int64_t ng = blockIdx.x;
    if (ng >= N * groups) return;

    int64_t n = ng / groups;
    int64_t g = ng % groups;
    int64_t offset = (n * C + g * channels_per_group) * spatial_size;

    // Compute mean over group
    T sum = T(0);
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        sum += input[offset + i];
    }
    sum = block_reduce_sum(sum, shared);
    __syncthreads();
    T mean = sum / T(group_size);

    // Compute variance over group
    T var_sum = T(0);
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        T diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, shared);
    __syncthreads();
    T variance = var_sum / T(group_size);
    T invstd = rsqrt(variance + epsilon);

    // Normalize and apply affine (channel-wise gamma/beta)
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int64_t c_local = i / spatial_size;
        int64_t c_global = g * channels_per_group + c_local;
        T normalized = (input[offset + i] - mean) * invstd;
        output[offset + i] = gamma[c_global] * normalized + beta[c_global];
    }
}

// Float16 specialized layernorm backward with float accumulation
__global__ void layernorm_backward_kernel_fp16(const __half* grad_output,
                                         const __half* input,
                                         const __half* gamma,
                                         __half* grad_input,
                                         float* grad_gamma_f32,
                                         float* grad_beta_f32,
                                         float epsilon,
                                         int64_t outer_size,
                                         int64_t normalized_size) {
    HIP_DYNAMIC_SHARED(unsigned char, shared_mem);
    float* shared = reinterpret_cast<float*>(shared_mem);

    int64_t instance = blockIdx.x;
    if (instance >= outer_size) return;

    int64_t offset = instance * normalized_size;

    // Compute mean in float
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        sum += __half2float(input[offset + i]);
    }
    sum = block_reduce_sum(sum, shared);
    __syncthreads();
    float mean = sum / static_cast<float>(normalized_size);

    // Compute variance in float
    float var_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        float diff = __half2float(input[offset + i]) - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, shared);
    __syncthreads();
    float variance = var_sum / static_cast<float>(normalized_size);
    float invstd = rsqrtf(variance + epsilon);

    // Compute gradient statistics
    float grad_output_sum = 0.0f;
    float grad_output_norm_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        float normalized = (__half2float(input[offset + i]) - mean) * invstd;
        float grad_out = __half2float(grad_output[offset + i]);
        grad_output_sum += grad_out;
        grad_output_norm_sum += grad_out * normalized;
    }
    grad_output_sum = block_reduce_sum(grad_output_sum, shared);
    __syncthreads();
    grad_output_norm_sum = block_reduce_sum(grad_output_norm_sum, shared);
    __syncthreads();

    float mean_grad = grad_output_sum / static_cast<float>(normalized_size);
    float mean_grad_norm = grad_output_norm_sum / static_cast<float>(normalized_size);

    // Compute input gradient
    for (int64_t i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        float normalized = (__half2float(input[offset + i]) - mean) * invstd;
        float grad_out = __half2float(grad_output[offset + i]);
        float g = __half2float(gamma[i]);
        float grad_normalized = grad_out - mean_grad - normalized * mean_grad_norm;
        grad_input[offset + i] = __float2half(g * invstd * grad_normalized);

        // Accumulate gradients in float
        atomicAdd(&grad_gamma_f32[i], grad_out * normalized);
        atomicAdd(&grad_beta_f32[i], grad_out);
    }
}

// Float16 specialized instancenorm backward with float accumulation
__global__ void instancenorm_backward_kernel_fp16(const __half* grad_output,
                                            const __half* input,
                                            const __half* gamma,
                                            __half* grad_input,
                                            float* grad_gamma_f32,
                                            float* grad_beta_f32,
                                            float epsilon,
                                            int64_t N,
                                            int64_t C,
                                            int64_t H,
                                            int64_t W) {
    HIP_DYNAMIC_SHARED(unsigned char, shared_mem);
    float* shared = reinterpret_cast<float*>(shared_mem);

    int64_t spatial_size = H * W;

    int64_t nc = blockIdx.x;
    if (nc >= N * C) return;

    int64_t n = nc / C;
    int64_t c = nc % C;
    int64_t offset = (n * C + c) * spatial_size;

    // Compute mean and variance in float
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        sum += __half2float(input[offset + i]);
    }
    sum = block_reduce_sum(sum, shared);
    __syncthreads();
    float mean = sum / static_cast<float>(spatial_size);

    float var_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float diff = __half2float(input[offset + i]) - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, shared);
    __syncthreads();
    float variance = var_sum / static_cast<float>(spatial_size);
    float invstd = rsqrtf(variance + epsilon);

    // Compute gradient statistics
    float grad_output_sum = 0.0f;
    float grad_output_norm_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float normalized = (__half2float(input[offset + i]) - mean) * invstd;
        float grad_out = __half2float(grad_output[offset + i]);
        grad_output_sum += grad_out;
        grad_output_norm_sum += grad_out * normalized;
    }
    grad_output_sum = block_reduce_sum(grad_output_sum, shared);
    __syncthreads();
    grad_output_norm_sum = block_reduce_sum(grad_output_norm_sum, shared);
    __syncthreads();

    float mean_grad = grad_output_sum / static_cast<float>(spatial_size);
    float mean_grad_norm = grad_output_norm_sum / static_cast<float>(spatial_size);

    float g = __half2float(gamma[c]);

    // Compute gradients
    for (int64_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float normalized = (__half2float(input[offset + i]) - mean) * invstd;
        float grad_out = __half2float(grad_output[offset + i]);
        float grad_normalized = grad_out - mean_grad - normalized * mean_grad_norm;
        grad_input[offset + i] = __float2half(g * invstd * grad_normalized);

        // Accumulate gradients in float
        atomicAdd(&grad_gamma_f32[c], grad_out * normalized);
        atomicAdd(&grad_beta_f32[c], grad_out);
    }
}

// Float16 specialized groupnorm backward with float accumulation
__global__ void groupnorm_backward_kernel_fp16(const __half* grad_output,
                                         const __half* input,
                                         const __half* gamma,
                                         __half* grad_input,
                                         float* grad_gamma_f32,
                                         float* grad_beta_f32,
                                         float epsilon,
                                         int64_t N,
                                         int64_t C,
                                         int64_t H,
                                         int64_t W,
                                         int64_t groups) {
    HIP_DYNAMIC_SHARED(unsigned char, shared_mem);
    float* shared = reinterpret_cast<float*>(shared_mem);

    int64_t channels_per_group = C / groups;
    int64_t spatial_size = H * W;
    int64_t group_size = channels_per_group * spatial_size;

    int64_t ng = blockIdx.x;
    if (ng >= N * groups) return;

    int64_t n = ng / groups;
    int64_t g_idx = ng % groups;
    int64_t offset = (n * C + g_idx * channels_per_group) * spatial_size;

    // Compute mean and variance in float
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        sum += __half2float(input[offset + i]);
    }
    sum = block_reduce_sum(sum, shared);
    __syncthreads();
    float mean = sum / static_cast<float>(group_size);

    float var_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        float diff = __half2float(input[offset + i]) - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, shared);
    __syncthreads();
    float variance = var_sum / static_cast<float>(group_size);
    float invstd = rsqrtf(variance + epsilon);

    // Compute gradient statistics
    float grad_output_sum = 0.0f;
    float grad_output_norm_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        float normalized = (__half2float(input[offset + i]) - mean) * invstd;
        float grad_out = __half2float(grad_output[offset + i]);
        grad_output_sum += grad_out;
        grad_output_norm_sum += grad_out * normalized;
    }
    grad_output_sum = block_reduce_sum(grad_output_sum, shared);
    __syncthreads();
    grad_output_norm_sum = block_reduce_sum(grad_output_norm_sum, shared);
    __syncthreads();

    float mean_grad = grad_output_sum / static_cast<float>(group_size);
    float mean_grad_norm = grad_output_norm_sum / static_cast<float>(group_size);

    // Compute gradients
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        float normalized = (__half2float(input[offset + i]) - mean) * invstd;
        int64_t c_local = i / spatial_size;
        int64_t c_global = g_idx * channels_per_group + c_local;
        float grad_out = __half2float(grad_output[offset + i]);
        float g = __half2float(gamma[c_global]);
        float grad_normalized = grad_out - mean_grad - normalized * mean_grad_norm;
        grad_input[offset + i] = __float2half(g * invstd * grad_normalized);

        // Accumulate gradients in float
        atomicAdd(&grad_gamma_f32[c_global], grad_out * normalized);
        atomicAdd(&grad_beta_f32[c_global], grad_out);
    }
}

// Kernel to convert float gradients to half
__global__ void batchnorm_convert_f32_to_f16_kernel(const float* src, __half* dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

template<typename T>
__global__ void groupnorm_backward_kernel(const T* grad_output,
                                         const T* input,
                                         const T* gamma,
                                         T* grad_input,
                                         T* grad_gamma,
                                         T* grad_beta,
                                         T epsilon,
                                         int64_t N,
                                         int64_t C,
                                         int64_t H,
                                         int64_t W,
                                         int64_t groups) {
    HIP_DYNAMIC_SHARED(unsigned char, shared_mem);
    T* shared = reinterpret_cast<T*>(shared_mem);

    int64_t channels_per_group = C / groups;
    int64_t spatial_size = H * W;
    int64_t group_size = channels_per_group * spatial_size;

    // Each block handles one (N, group) pair
    int64_t ng = blockIdx.x;
    if (ng >= N * groups) return;

    int64_t n = ng / groups;
    int64_t g = ng % groups;
    int64_t offset = (n * C + g * channels_per_group) * spatial_size;

    // Compute mean and variance
    T sum = T(0);
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        sum += input[offset + i];
    }
    sum = block_reduce_sum(sum, shared);
    __syncthreads();
    T mean = sum / T(group_size);

    T var_sum = T(0);
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        T diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, shared);
    __syncthreads();
    T variance = var_sum / T(group_size);
    T invstd = rsqrt(variance + epsilon);

    // Compute gradient statistics
    T grad_output_sum = T(0);
    T grad_output_norm_sum = T(0);
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        T normalized = (input[offset + i] - mean) * invstd;
        int64_t c_local = i / spatial_size;
        int64_t c_global = g * channels_per_group + c_local;
        T grad_out = grad_output[offset + i];
        grad_output_sum += grad_out;
        grad_output_norm_sum += grad_out * normalized;
    }
    grad_output_sum = block_reduce_sum(grad_output_sum, shared);
    __syncthreads();
    grad_output_norm_sum = block_reduce_sum(grad_output_norm_sum, shared);
    __syncthreads();

    T mean_grad = grad_output_sum / T(group_size);
    T mean_grad_norm = grad_output_norm_sum / T(group_size);

    // Compute gradients
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        T normalized = (input[offset + i] - mean) * invstd;
        int64_t c_local = i / spatial_size;
        int64_t c_global = g * channels_per_group + c_local;
        T grad_out = grad_output[offset + i];
        T grad_normalized = grad_out - mean_grad - normalized * mean_grad_norm;
        grad_input[offset + i] = gamma[c_global] * invstd * grad_normalized;

        // Accumulate gradients for gamma and beta
        atomicAdd(&grad_gamma[c_global], grad_out * normalized);
        atomicAdd(&grad_beta[c_global], grad_out);
    }
}

// ============================================================================
// Host Functions (C++ API)
// ============================================================================

// Compute mean and variance for a batch
auto batchnorm2d_mean_var(const Tensor& input,
                          Tensor& mean,
                          Tensor& variance,
                          hipStream_t stream) -> void {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    // Check for division by zero
    int64_t total_elements = N * H * W;
    if (total_elements == 0) {
        throw std::runtime_error("BatchNorm2d HIP: Cannot compute mean/variance for empty tensor (N*H*W = 0)");
    }

    if (input.dtype() == DType::Float32) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(float);
        hipLaunchKernelGGL(batchnorm_mean_kernel<float>, dim3(C), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          input.data<float>(), mean.data<float>(), N, C, H, W);
        HIP_CHECK(hipGetLastError());

        hipLaunchKernelGGL(batchnorm_variance_kernel<float>, dim3(C), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          input.data<float>(), mean.data<float>(), variance.data<float>(), N, C, H, W);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(double);
        hipLaunchKernelGGL(batchnorm_mean_kernel<double>, dim3(C), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          input.data<double>(), mean.data<double>(), N, C, H, W);
        HIP_CHECK(hipGetLastError());

        hipLaunchKernelGGL(batchnorm_variance_kernel<double>, dim3(C), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          input.data<double>(), mean.data<double>(), variance.data<double>(), N, C, H, W);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float16) {
        // For Float16, use __half type
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(__half);
        hipLaunchKernelGGL(batchnorm_mean_kernel<__half>, dim3(C), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<__half*>(mean.data<Float16>()), N, C, H, W);
        HIP_CHECK(hipGetLastError());

        hipLaunchKernelGGL(batchnorm_variance_kernel<__half>, dim3(C), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<__half*>(mean.data<Float16>()),
                          reinterpret_cast<__half*>(variance.data<Float16>()), N, C, H, W);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        Tensor mean_f32({mean.shape()[0]}, DType::Float32, mean.device());
        Tensor var_f32({variance.shape()[0]}, DType::Float32, variance.device());
        batchnorm2d_mean_var(input_f32, mean_f32, var_f32, stream);
        // Copy results back to BFloat16
        auto mean_bf16 = mean_f32.to(DType::BFloat16);
        auto var_bf16 = var_f32.to(DType::BFloat16);
        HIP_CHECK(hipMemcpyAsync(mean.data_ptr(), mean_bf16.data_ptr(),
            mean.numel() * dtype_size(DType::BFloat16), hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(variance.data_ptr(), var_bf16.data_ptr(),
            variance.numel() * dtype_size(DType::BFloat16), hipMemcpyDeviceToDevice, stream));
    } else {
        throw std::runtime_error("BatchNorm2D only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }
}

// Forward pass (normalization only, no affine)
auto batchnorm2d_forward(const Tensor& input,
                         const Tensor& mean,
                         const Tensor& variance,
                         float epsilon,
                         hipStream_t stream) -> Tensor {
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
        hipLaunchKernelGGL(batchnorm_normalize_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE),
                          0, stream,
                          input.data<float>(), output.data<float>(),
                          mean.data<float>(), variance.data<float>(),
                          epsilon, N, C, H, W);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(total_size);
        hipLaunchKernelGGL(batchnorm_normalize_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE),
                          0, stream,
                          input.data<double>(), output.data<double>(),
                          mean.data<double>(), variance.data<double>(),
                          epsilon, N, C, H, W);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(total_size);
        hipLaunchKernelGGL(batchnorm_normalize_kernel<__half>, dim3(num_blocks), dim3(BLOCK_SIZE),
                          0, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<__half*>(output.data<Float16>()),
                          reinterpret_cast<const __half*>(mean.data<Float16>()),
                          reinterpret_cast<const __half*>(variance.data<Float16>()),
                          epsilon, N, C, H, W);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto mean_f32 = mean.to(DType::Float32);
        auto variance_f32 = variance.to(DType::Float32);
        auto result_f32 = batchnorm2d_forward(input_f32, mean_f32, variance_f32, epsilon, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("BatchNorm2D only supports Float32, Float64, Float16, and BFloat16 dtypes");
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
                                hipStream_t stream) -> Tensor {
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
        hipLaunchKernelGGL(batchnorm_forward_affine_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE),
                          0, stream,
                          input.data<float>(), output.data<float>(),
                          mean.data<float>(), variance.data<float>(),
                          gamma.data<float>(), beta.data<float>(),
                          epsilon, N, C, H, W);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(total_size);
        hipLaunchKernelGGL(batchnorm_forward_affine_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE),
                          0, stream,
                          input.data<double>(), output.data<double>(),
                          mean.data<double>(), variance.data<double>(),
                          gamma.data<double>(), beta.data<double>(),
                          epsilon, N, C, H, W);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(total_size);
        hipLaunchKernelGGL(batchnorm_forward_affine_kernel<__half>, dim3(num_blocks), dim3(BLOCK_SIZE),
                          0, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<__half*>(output.data<Float16>()),
                          reinterpret_cast<const __half*>(mean.data<Float16>()),
                          reinterpret_cast<const __half*>(variance.data<Float16>()),
                          reinterpret_cast<const __half*>(gamma.data<Float16>()),
                          reinterpret_cast<const __half*>(beta.data<Float16>()),
                          epsilon, N, C, H, W);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto mean_f32 = mean.to(DType::Float32);
        auto variance_f32 = variance.to(DType::Float32);
        auto gamma_f32 = gamma.to(DType::Float32);
        auto beta_f32 = beta.to(DType::Float32);
        auto result_f32 = batchnorm2d_forward_affine(input_f32, mean_f32, variance_f32, gamma_f32, beta_f32, epsilon, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("BatchNorm2D only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    return output;
}

// Update running statistics
auto batchnorm2d_update_running_stats(Tensor& running_mean,
                                      Tensor& running_var,
                                      const Tensor& batch_mean,
                                      const Tensor& batch_var,
                                      float momentum,
                                      hipStream_t stream) -> void {
    int64_t C = batch_mean.shape()[0];

    if (running_mean.dtype() == DType::Float32) {
        int num_blocks = get_num_blocks(C);
        hipLaunchKernelGGL(batchnorm_update_running_stats_kernel<float>, dim3(num_blocks), dim3(BLOCK_SIZE),
                          0, stream,
                          running_mean.data<float>(), running_var.data<float>(),
                          batch_mean.data<float>(), batch_var.data<float>(),
                          momentum, C);
        HIP_CHECK(hipGetLastError());
    } else if (running_mean.dtype() == DType::Float64) {
        int num_blocks = get_num_blocks(C);
        hipLaunchKernelGGL(batchnorm_update_running_stats_kernel<double>, dim3(num_blocks), dim3(BLOCK_SIZE),
                          0, stream,
                          running_mean.data<double>(), running_var.data<double>(),
                          batch_mean.data<double>(), batch_var.data<double>(),
                          momentum, C);
        HIP_CHECK(hipGetLastError());
    } else if (running_mean.dtype() == DType::Float16) {
        int num_blocks = get_num_blocks(C);
        hipLaunchKernelGGL(batchnorm_update_running_stats_kernel<__half>, dim3(num_blocks), dim3(BLOCK_SIZE),
                          0, stream,
                          reinterpret_cast<__half*>(running_mean.data<Float16>()),
                          reinterpret_cast<__half*>(running_var.data<Float16>()),
                          reinterpret_cast<const __half*>(batch_mean.data<Float16>()),
                          reinterpret_cast<const __half*>(batch_var.data<Float16>()),
                          momentum, C);
        HIP_CHECK(hipGetLastError());
    } else if (running_mean.dtype() == DType::BFloat16) {
        auto rm_f32 = running_mean.to(DType::Float32);
        auto rv_f32 = running_var.to(DType::Float32);
        auto bm_f32 = batch_mean.to(DType::Float32);
        auto bv_f32 = batch_var.to(DType::Float32);
        batchnorm2d_update_running_stats(rm_f32, rv_f32, bm_f32, bv_f32, momentum, stream);
        auto rm_bf16 = rm_f32.to(DType::BFloat16);
        auto rv_bf16 = rv_f32.to(DType::BFloat16);
        HIP_CHECK(hipMemcpyAsync(running_mean.data_ptr(), rm_bf16.data_ptr(),
            running_mean.numel() * dtype_size(DType::BFloat16), hipMemcpyDeviceToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(running_var.data_ptr(), rv_bf16.data_ptr(),
            running_var.numel() * dtype_size(DType::BFloat16), hipMemcpyDeviceToDevice, stream));
    } else {
        throw std::runtime_error("BatchNorm2D only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }
}

// Backward pass - compute gradients
auto batchnorm2d_backward(const Tensor& grad_output,
                         const Tensor& input,
                         const Tensor& mean,
                         const Tensor& variance,
                         const Tensor& gamma,
                         float epsilon,
                         hipStream_t stream) -> std::tuple<Tensor, Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    // Check for division by zero
    int64_t total_elements = N * H * W;
    if (total_elements == 0) {
        throw std::runtime_error("BatchNorm2d HIP backward: Cannot compute gradients for empty tensor (N*H*W = 0)");
    }

    // Allocate output gradients
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor grad_input(shape_vec, input.dtype(), input.device());
    Tensor grad_gamma({C}, input.dtype(), input.device());
    Tensor grad_beta({C}, input.dtype(), input.device());

    // Compute normalized input for gradient computation
    Tensor normalized = batchnorm2d_forward(input, mean, variance, epsilon, stream);

    if (input.dtype() == DType::Float32) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(float);

        // Compute grad_gamma and grad_beta
        hipLaunchKernelGGL(batchnorm_backward_gamma_beta_kernel<float>, dim3(C), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          grad_output.data<float>(), normalized.data<float>(),
                          grad_gamma.data<float>(), grad_beta.data<float>(),
                          N, C, H, W);
        HIP_CHECK(hipGetLastError());

        // Compute grad_input
        hipLaunchKernelGGL(batchnorm_backward_input_kernel<float>, dim3(C), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          grad_output.data<float>(), input.data<float>(), grad_input.data<float>(),
                          mean.data<float>(), variance.data<float>(), gamma.data<float>(),
                          epsilon, N, C, H, W);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(double);

        // Compute grad_gamma and grad_beta
        hipLaunchKernelGGL(batchnorm_backward_gamma_beta_kernel<double>, dim3(C), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          grad_output.data<double>(), normalized.data<double>(),
                          grad_gamma.data<double>(), grad_beta.data<double>(),
                          N, C, H, W);
        HIP_CHECK(hipGetLastError());

        // Compute grad_input
        hipLaunchKernelGGL(batchnorm_backward_input_kernel<double>, dim3(C), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          grad_output.data<double>(), input.data<double>(), grad_input.data<double>(),
                          mean.data<double>(), variance.data<double>(), gamma.data<double>(),
                          epsilon, N, C, H, W);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(__half);

        // Compute grad_gamma and grad_beta
        hipLaunchKernelGGL(batchnorm_backward_gamma_beta_kernel<__half>, dim3(C), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          reinterpret_cast<const __half*>(grad_output.data<Float16>()),
                          reinterpret_cast<const __half*>(normalized.data<Float16>()),
                          reinterpret_cast<__half*>(grad_gamma.data<Float16>()),
                          reinterpret_cast<__half*>(grad_beta.data<Float16>()),
                          N, C, H, W);
        HIP_CHECK(hipGetLastError());

        // Compute grad_input
        hipLaunchKernelGGL(batchnorm_backward_input_kernel<__half>, dim3(C), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          reinterpret_cast<const __half*>(grad_output.data<Float16>()),
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<__half*>(grad_input.data<Float16>()),
                          reinterpret_cast<const __half*>(mean.data<Float16>()),
                          reinterpret_cast<const __half*>(variance.data<Float16>()),
                          reinterpret_cast<const __half*>(gamma.data<Float16>()),
                          epsilon, N, C, H, W);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto mean_f32 = mean.to(DType::Float32);
        auto variance_f32 = variance.to(DType::Float32);
        auto gamma_f32 = gamma.to(DType::Float32);
        auto [gi_f32, gg_f32, gb_f32] = batchnorm2d_backward(grad_output_f32, input_f32, mean_f32, variance_f32, gamma_f32, epsilon, stream);
        return std::make_tuple(gi_f32.to(DType::BFloat16), gg_f32.to(DType::BFloat16), gb_f32.to(DType::BFloat16));
    } else {
        throw std::runtime_error("BatchNorm2D only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    return std::make_tuple(grad_input, grad_gamma, grad_beta);
}

// LayerNorm host functions
auto layernorm_forward(const Tensor& input,
                      const Tensor& gamma,
                      const Tensor& beta,
                      float epsilon,
                      int64_t normalized_size,
                      hipStream_t stream) -> Tensor {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());

    int64_t total_size = 1;
    for (auto s : shape) total_size *= s;
    int64_t outer_size = total_size / normalized_size;

    if (input.dtype() == DType::Float32) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(float);
        hipLaunchKernelGGL(layernorm_forward_kernel<float>, dim3(outer_size), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          input.data<float>(), output.data<float>(),
                          gamma.data<float>(), beta.data<float>(),
                          epsilon, outer_size, normalized_size);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(double);
        hipLaunchKernelGGL(layernorm_forward_kernel<double>, dim3(outer_size), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          input.data<double>(), output.data<double>(),
                          gamma.data<double>(), beta.data<double>(),
                          epsilon, outer_size, normalized_size);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(__half);
        hipLaunchKernelGGL(layernorm_forward_kernel<__half>, dim3(outer_size), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<__half*>(output.data<Float16>()),
                          reinterpret_cast<const __half*>(gamma.data<Float16>()),
                          reinterpret_cast<const __half*>(beta.data<Float16>()),
                          epsilon, outer_size, normalized_size);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto gamma_f32 = gamma.to(DType::Float32);
        auto beta_f32 = beta.to(DType::Float32);
        auto result_f32 = layernorm_forward(input_f32, gamma_f32, beta_f32, epsilon, normalized_size, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("LayerNorm only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    return output;
}

auto layernorm_backward(const Tensor& grad_output,
                       const Tensor& input,
                       const Tensor& gamma,
                       float epsilon,
                       int64_t normalized_size,
                       hipStream_t stream) -> std::tuple<Tensor, Tensor, Tensor> {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor grad_input(shape_vec, input.dtype(), input.device());
    Tensor grad_gamma({normalized_size}, input.dtype(), input.device());
    Tensor grad_beta({normalized_size}, input.dtype(), input.device());

    int64_t total_size = 1;
    for (auto s : shape) total_size *= s;
    int64_t outer_size = total_size / normalized_size;

    // Zero initialize grad_gamma and grad_beta
    HIP_CHECK(hipMemsetAsync(grad_gamma.data_ptr(), 0, normalized_size * dtype_size(input.dtype()), stream));
    HIP_CHECK(hipMemsetAsync(grad_beta.data_ptr(), 0, normalized_size * dtype_size(input.dtype()), stream));

    if (input.dtype() == DType::Float32) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(float);
        hipLaunchKernelGGL(layernorm_backward_kernel<float>, dim3(outer_size), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          grad_output.data<float>(), input.data<float>(), gamma.data<float>(),
                          grad_input.data<float>(), grad_gamma.data<float>(), grad_beta.data<float>(),
                          epsilon, outer_size, normalized_size);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(double);
        hipLaunchKernelGGL(layernorm_backward_kernel<double>, dim3(outer_size), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          grad_output.data<double>(), input.data<double>(), gamma.data<double>(),
                          grad_input.data<double>(), grad_gamma.data<double>(), grad_beta.data<double>(),
                          epsilon, outer_size, normalized_size);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float16) {
        // Allocate float accumulators for gradients
        Tensor grad_gamma_f32({normalized_size}, DType::Float32, input.device());
        Tensor grad_beta_f32({normalized_size}, DType::Float32, input.device());
        HIP_CHECK(hipMemsetAsync(grad_gamma_f32.data<float>(), 0, normalized_size * sizeof(float), stream));
        HIP_CHECK(hipMemsetAsync(grad_beta_f32.data<float>(), 0, normalized_size * sizeof(float), stream));

        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(float);
        hipLaunchKernelGGL(layernorm_backward_kernel_fp16, dim3(outer_size), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          reinterpret_cast<const __half*>(grad_output.data<Float16>()),
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<const __half*>(gamma.data<Float16>()),
                          reinterpret_cast<__half*>(grad_input.data<Float16>()),
                          grad_gamma_f32.data<float>(),
                          grad_beta_f32.data<float>(),
                          epsilon, outer_size, normalized_size);
        HIP_CHECK(hipGetLastError());

        // Convert float gradients to half
        int convert_blocks = (normalized_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        hipLaunchKernelGGL(batchnorm_convert_f32_to_f16_kernel, dim3(convert_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_gamma_f32.data<float>(),
                          reinterpret_cast<__half*>(grad_gamma.data<Float16>()),
                          normalized_size);
        hipLaunchKernelGGL(batchnorm_convert_f32_to_f16_kernel, dim3(convert_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_beta_f32.data<float>(),
                          reinterpret_cast<__half*>(grad_beta.data<Float16>()),
                          normalized_size);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto gamma_f32 = gamma.to(DType::Float32);
        auto [gi_f32, gg_f32, gb_f32] = layernorm_backward(grad_output_f32, input_f32, gamma_f32, epsilon, normalized_size, stream);
        return std::make_tuple(gi_f32.to(DType::BFloat16), gg_f32.to(DType::BFloat16), gb_f32.to(DType::BFloat16));
    } else {
        throw std::runtime_error("LayerNorm only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    return std::make_tuple(grad_input, grad_gamma, grad_beta);
}

// InstanceNorm host functions
auto instancenorm_forward(const Tensor& input,
                         const Tensor& gamma,
                         const Tensor& beta,
                         float epsilon,
                         hipStream_t stream) -> Tensor {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    if (input.dtype() == DType::Float32) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(float);
        hipLaunchKernelGGL(instancenorm_forward_kernel<float>, dim3(N * C), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          input.data<float>(), output.data<float>(),
                          gamma.data<float>(), beta.data<float>(),
                          epsilon, N, C, H, W);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(double);
        hipLaunchKernelGGL(instancenorm_forward_kernel<double>, dim3(N * C), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          input.data<double>(), output.data<double>(),
                          gamma.data<double>(), beta.data<double>(),
                          epsilon, N, C, H, W);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(__half);
        hipLaunchKernelGGL(instancenorm_forward_kernel<__half>, dim3(N * C), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<__half*>(output.data<Float16>()),
                          reinterpret_cast<const __half*>(gamma.data<Float16>()),
                          reinterpret_cast<const __half*>(beta.data<Float16>()),
                          epsilon, N, C, H, W);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto gamma_f32 = gamma.to(DType::Float32);
        auto beta_f32 = beta.to(DType::Float32);
        auto result_f32 = instancenorm_forward(input_f32, gamma_f32, beta_f32, epsilon, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("InstanceNorm only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    return output;
}

auto instancenorm_backward(const Tensor& grad_output,
                          const Tensor& input,
                          const Tensor& gamma,
                          float epsilon,
                          hipStream_t stream) -> std::tuple<Tensor, Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor grad_input(shape_vec, input.dtype(), input.device());
    Tensor grad_gamma({C}, input.dtype(), input.device());
    Tensor grad_beta({C}, input.dtype(), input.device());

    // Zero initialize grad_gamma and grad_beta
    HIP_CHECK(hipMemsetAsync(grad_gamma.data_ptr(), 0, C * dtype_size(input.dtype()), stream));
    HIP_CHECK(hipMemsetAsync(grad_beta.data_ptr(), 0, C * dtype_size(input.dtype()), stream));

    if (input.dtype() == DType::Float32) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(float);
        hipLaunchKernelGGL(instancenorm_backward_kernel<float>, dim3(N * C), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          grad_output.data<float>(), input.data<float>(), gamma.data<float>(),
                          grad_input.data<float>(), grad_gamma.data<float>(), grad_beta.data<float>(),
                          epsilon, N, C, H, W);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(double);
        hipLaunchKernelGGL(instancenorm_backward_kernel<double>, dim3(N * C), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          grad_output.data<double>(), input.data<double>(), gamma.data<double>(),
                          grad_input.data<double>(), grad_gamma.data<double>(), grad_beta.data<double>(),
                          epsilon, N, C, H, W);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float16) {
        // Allocate float accumulators for gradients
        Tensor grad_gamma_f32({C}, DType::Float32, input.device());
        Tensor grad_beta_f32({C}, DType::Float32, input.device());
        HIP_CHECK(hipMemsetAsync(grad_gamma_f32.data<float>(), 0, C * sizeof(float), stream));
        HIP_CHECK(hipMemsetAsync(grad_beta_f32.data<float>(), 0, C * sizeof(float), stream));

        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(float);
        hipLaunchKernelGGL(instancenorm_backward_kernel_fp16, dim3(N * C), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          reinterpret_cast<const __half*>(grad_output.data<Float16>()),
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<const __half*>(gamma.data<Float16>()),
                          reinterpret_cast<__half*>(grad_input.data<Float16>()),
                          grad_gamma_f32.data<float>(),
                          grad_beta_f32.data<float>(),
                          epsilon, N, C, H, W);
        HIP_CHECK(hipGetLastError());

        // Convert float gradients to half
        int convert_blocks = (C + BLOCK_SIZE - 1) / BLOCK_SIZE;
        hipLaunchKernelGGL(batchnorm_convert_f32_to_f16_kernel, dim3(convert_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_gamma_f32.data<float>(),
                          reinterpret_cast<__half*>(grad_gamma.data<Float16>()),
                          C);
        hipLaunchKernelGGL(batchnorm_convert_f32_to_f16_kernel, dim3(convert_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_beta_f32.data<float>(),
                          reinterpret_cast<__half*>(grad_beta.data<Float16>()),
                          C);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto gamma_f32 = gamma.to(DType::Float32);
        auto [gi_f32, gg_f32, gb_f32] = instancenorm_backward(grad_output_f32, input_f32, gamma_f32, epsilon, stream);
        return std::make_tuple(gi_f32.to(DType::BFloat16), gg_f32.to(DType::BFloat16), gb_f32.to(DType::BFloat16));
    } else {
        throw std::runtime_error("InstanceNorm only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    return std::make_tuple(grad_input, grad_gamma, grad_beta);
}

// GroupNorm host functions
auto groupnorm_forward(const Tensor& input,
                      const Tensor& gamma,
                      const Tensor& beta,
                      int64_t groups,
                      float epsilon,
                      hipStream_t stream) -> Tensor {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    if (C % groups != 0) {
        throw std::runtime_error("GroupNorm: Number of channels must be divisible by number of groups");
    }

    if (input.dtype() == DType::Float32) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(float);
        hipLaunchKernelGGL(groupnorm_forward_kernel<float>, dim3(N * groups), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          input.data<float>(), output.data<float>(),
                          gamma.data<float>(), beta.data<float>(),
                          epsilon, N, C, H, W, groups);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(double);
        hipLaunchKernelGGL(groupnorm_forward_kernel<double>, dim3(N * groups), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          input.data<double>(), output.data<double>(),
                          gamma.data<double>(), beta.data<double>(),
                          epsilon, N, C, H, W, groups);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float16) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(__half);
        hipLaunchKernelGGL(groupnorm_forward_kernel<__half>, dim3(N * groups), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<__half*>(output.data<Float16>()),
                          reinterpret_cast<const __half*>(gamma.data<Float16>()),
                          reinterpret_cast<const __half*>(beta.data<Float16>()),
                          epsilon, N, C, H, W, groups);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto gamma_f32 = gamma.to(DType::Float32);
        auto beta_f32 = beta.to(DType::Float32);
        auto result_f32 = groupnorm_forward(input_f32, gamma_f32, beta_f32, groups, epsilon, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("GroupNorm only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    return output;
}

auto groupnorm_backward(const Tensor& grad_output,
                       const Tensor& input,
                       const Tensor& gamma,
                       int64_t groups,
                       float epsilon,
                       hipStream_t stream) -> std::tuple<Tensor, Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor grad_input(shape_vec, input.dtype(), input.device());
    Tensor grad_gamma({C}, input.dtype(), input.device());
    Tensor grad_beta({C}, input.dtype(), input.device());

    // Zero initialize grad_gamma and grad_beta
    HIP_CHECK(hipMemsetAsync(grad_gamma.data_ptr(), 0, C * dtype_size(input.dtype()), stream));
    HIP_CHECK(hipMemsetAsync(grad_beta.data_ptr(), 0, C * dtype_size(input.dtype()), stream));

    if (input.dtype() == DType::Float32) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(float);
        hipLaunchKernelGGL(groupnorm_backward_kernel<float>, dim3(N * groups), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          grad_output.data<float>(), input.data<float>(), gamma.data<float>(),
                          grad_input.data<float>(), grad_gamma.data<float>(), grad_beta.data<float>(),
                          epsilon, N, C, H, W, groups);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float64) {
        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(double);
        hipLaunchKernelGGL(groupnorm_backward_kernel<double>, dim3(N * groups), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          grad_output.data<double>(), input.data<double>(), gamma.data<double>(),
                          grad_input.data<double>(), grad_gamma.data<double>(), grad_beta.data<double>(),
                          epsilon, N, C, H, W, groups);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::Float16) {
        // Allocate float accumulators for gradients
        Tensor grad_gamma_f32({C}, DType::Float32, input.device());
        Tensor grad_beta_f32({C}, DType::Float32, input.device());
        HIP_CHECK(hipMemsetAsync(grad_gamma_f32.data<float>(), 0, C * sizeof(float), stream));
        HIP_CHECK(hipMemsetAsync(grad_beta_f32.data<float>(), 0, C * sizeof(float), stream));

        int shared_mem_size = (BATCHNORM_BLOCK_SIZE / MIN_WAVEFRONT_SIZE) * sizeof(float);
        hipLaunchKernelGGL(groupnorm_backward_kernel_fp16, dim3(N * groups), dim3(BATCHNORM_BLOCK_SIZE),
                          shared_mem_size, stream,
                          reinterpret_cast<const __half*>(grad_output.data<Float16>()),
                          reinterpret_cast<const __half*>(input.data<Float16>()),
                          reinterpret_cast<const __half*>(gamma.data<Float16>()),
                          reinterpret_cast<__half*>(grad_input.data<Float16>()),
                          grad_gamma_f32.data<float>(),
                          grad_beta_f32.data<float>(),
                          epsilon, N, C, H, W, groups);
        HIP_CHECK(hipGetLastError());

        // Convert float gradients to half
        int convert_blocks = (C + BLOCK_SIZE - 1) / BLOCK_SIZE;
        hipLaunchKernelGGL(batchnorm_convert_f32_to_f16_kernel, dim3(convert_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_gamma_f32.data<float>(),
                          reinterpret_cast<__half*>(grad_gamma.data<Float16>()),
                          C);
        hipLaunchKernelGGL(batchnorm_convert_f32_to_f16_kernel, dim3(convert_blocks), dim3(BLOCK_SIZE), 0, stream,
                          grad_beta_f32.data<float>(),
                          reinterpret_cast<__half*>(grad_beta.data<Float16>()),
                          C);
        HIP_CHECK(hipGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto gamma_f32 = gamma.to(DType::Float32);
        auto [gi_f32, gg_f32, gb_f32] = groupnorm_backward(grad_output_f32, input_f32, gamma_f32, groups, epsilon, stream);
        return std::make_tuple(gi_f32.to(DType::BFloat16), gg_f32.to(DType::BFloat16), gb_f32.to(DType::BFloat16));
    } else {
        throw std::runtime_error("GroupNorm only supports Float32, Float64, Float16, and BFloat16 dtypes");
    }

    return std::make_tuple(grad_input, grad_gamma, grad_beta);
}

} // namespace rocm
} // namespace tenzor
