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
#include "../cuda_error.hpp"
#include "cuda_launch_utils.cuh"
#include "cuda_common.cuh"

namespace tenzor {
namespace cuda {

// Optimal block size
constexpr int BLOCK_SIZE = 256;
constexpr int BATCHNORM_BLOCK_SIZE = 512;

// Calculate grid size
inline int get_num_blocks(int64_t n, int block_size = BLOCK_SIZE) {
    // Clamp to the CUDA grid x-dimension limit (2^31-1) and compute the count in
    // 64-bit so a >2^31-element launch does not silently truncate to a wrong int.
    int64_t blocks = (n + block_size - 1) / block_size;
    if (blocks < 1) blocks = 1;
    if (blocks > 2147483647LL) blocks = 2147483647LL;
    return static_cast<int>(blocks);
}

// Warp/block reduction primitives are in cuda_common.cuh

// ============================================================================
// BatchNorm2d Mean/Variance Computation (Welford's Algorithm)
// ============================================================================

// Two-pass algorithm for per-channel mean and variance computation.
// `Acc` is the accumulator type: for a Float32 input it is `double`, so the
// per-channel sum and sum-of-squared-deviations are accumulated in double —
// matching the CPU reference (Welford in double). A `float` accumulator over
// N*H*W elements drifts and the population variance diverges from CPU past the
// cross-backend tolerance for large spatial sizes.
template<typename T, typename Acc>
__global__ void batchnorm_mean_kernel(const T* input,
                                      T* mean,
                                      int64_t N,
                                      int64_t C,
                                      int64_t H,
                                      int64_t W) {
    extern __shared__ __align__(sizeof(Acc)) unsigned char shared_mem[];
    Acc* shared = reinterpret_cast<Acc*>(shared_mem);

    int64_t spatial_size = H * W;
    int64_t total_elements = N * spatial_size;

    // Each block handles one channel
    int64_t c = blockIdx.x;
    if (c >= C) return;

    // Compute sum in the wide accumulator type
    Acc sum = Acc(0);
    for (int64_t n = 0; n < N; n++) {
        for (int64_t idx = threadIdx.x; idx < spatial_size; idx += blockDim.x) {
            int64_t h = idx / W;
            int64_t w = idx % W;
            int64_t tensor_idx = ((n * C + c) * H + h) * W + w;
            sum += static_cast<Acc>(input[tensor_idx]);
        }
    }

    // Reduce sum across threads (in Acc)
    sum = block_reduce_sum(sum, shared);

    // Thread 0 writes mean
    if (threadIdx.x == 0) {
        mean[c] = static_cast<T>(sum / Acc(total_elements));
    }
}

template<typename T, typename Acc>
__global__ void batchnorm_variance_kernel(const T* input,
                                          const T* mean,
                                          T* variance,
                                          int64_t N,
                                          int64_t C,
                                          int64_t H,
                                          int64_t W) {
    extern __shared__ __align__(sizeof(Acc)) unsigned char shared_mem[];
    Acc* shared = reinterpret_cast<Acc*>(shared_mem);

    int64_t spatial_size = H * W;
    int64_t total_elements = N * spatial_size;

    // Each block handles one channel
    int64_t c = blockIdx.x;
    if (c >= C) return;

    Acc channel_mean = static_cast<Acc>(mean[c]);

    // Compute sum of squared differences in the wide accumulator type
    Acc sum_sq_diff = Acc(0);
    for (int64_t n = 0; n < N; n++) {
        for (int64_t idx = threadIdx.x; idx < spatial_size; idx += blockDim.x) {
            int64_t h = idx / W;
            int64_t w = idx % W;
            int64_t tensor_idx = ((n * C + c) * H + h) * W + w;
            Acc diff = static_cast<Acc>(input[tensor_idx]) - channel_mean;
            sum_sq_diff += diff * diff;
        }
    }

    // Reduce sum across threads (in Acc)
    sum_sq_diff = block_reduce_sum(sum_sq_diff, shared);

    // Thread 0 writes variance
    if (threadIdx.x == 0) {
        variance[c] = static_cast<T>(sum_sq_diff / Acc(total_elements));
    }
}

// ============================================================================
// Chunked Mean/Variance Kernels (2D grid for better SM utilization when C < 64)
// ============================================================================

// Chunked mean kernel: 2D grid (C, spatial_chunks) for better GPU utilization
// when C is small. `Acc` is the wide accumulator type (double for Float32) so
// the cross-block partial sums combine in double, matching the CPU reference.
template<typename T, typename Acc>
__global__ void batchnorm_mean_chunked_kernel(const T* input,
                                               Acc* partial_sums,
                                               int64_t N, int64_t C,
                                               int64_t H, int64_t W,
                                               int64_t spatial_chunk_size) {
    extern __shared__ __align__(sizeof(Acc)) unsigned char shared_mem[];
    Acc* shared = reinterpret_cast<Acc*>(shared_mem);

    int64_t c = blockIdx.x;
    int64_t chunk_id = blockIdx.y;
    if (c >= C) return;

    int64_t spatial_size = H * W;
    int64_t total_elements = N * spatial_size;
    int64_t chunk_start = chunk_id * spatial_chunk_size;
    int64_t chunk_end = min(chunk_start + spatial_chunk_size, total_elements);

    Acc sum = Acc(0);
    for (int64_t idx = chunk_start + threadIdx.x; idx < chunk_end; idx += blockDim.x) {
        int64_t n = idx / spatial_size;
        int64_t spatial_idx = idx % spatial_size;
        int64_t h = spatial_idx / W;
        int64_t w = spatial_idx % W;
        int64_t tensor_idx = ((n * C + c) * H + h) * W + w;
        sum += static_cast<Acc>(input[tensor_idx]);
    }

    sum = block_reduce_sum(sum, shared);

    if (threadIdx.x == 0) {
        atomicAdd(&partial_sums[c], sum);
    }
}

// Chunked variance kernel: 2D grid for better utilization when C is small.
template<typename T, typename Acc>
__global__ void batchnorm_variance_chunked_kernel(const T* input,
                                                    const Acc* mean,
                                                    Acc* partial_sums,
                                                    int64_t N, int64_t C,
                                                    int64_t H, int64_t W,
                                                    int64_t spatial_chunk_size) {
    extern __shared__ __align__(sizeof(Acc)) unsigned char shared_mem[];
    Acc* shared = reinterpret_cast<Acc*>(shared_mem);

    int64_t c = blockIdx.x;
    int64_t chunk_id = blockIdx.y;
    if (c >= C) return;

    int64_t spatial_size = H * W;
    int64_t total_elements = N * spatial_size;
    int64_t chunk_start = chunk_id * spatial_chunk_size;
    int64_t chunk_end = min(chunk_start + spatial_chunk_size, total_elements);

    Acc channel_mean = mean[c];
    Acc sum_sq_diff = Acc(0);
    for (int64_t idx = chunk_start + threadIdx.x; idx < chunk_end; idx += blockDim.x) {
        int64_t n = idx / spatial_size;
        int64_t spatial_idx = idx % spatial_size;
        int64_t h = spatial_idx / W;
        int64_t w = spatial_idx % W;
        int64_t tensor_idx = ((n * C + c) * H + h) * W + w;
        Acc diff = static_cast<Acc>(input[tensor_idx]) - channel_mean;
        sum_sq_diff += diff * diff;
    }

    sum_sq_diff = block_reduce_sum(sum_sq_diff, shared);

    if (threadIdx.x == 0) {
        atomicAdd(&partial_sums[c], sum_sq_diff);
    }
}

// Finalize: divide the Acc partial sums by total element count and write the
// result narrowed to the output dtype T.
template<typename T, typename Acc>
__global__ void batchnorm_finalize_mean_kernel(const Acc* partial, T* out,
                                                int64_t C,
                                                int64_t total_elements) {
    int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) {
        out[c] = static_cast<T>(partial[c] / Acc(total_elements));
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

    TENZOR_CUDA_KERNEL_LOOP(idx, total_size) {
        // Decode NCHW index - only channel index needed for normalization
        int64_t c = (idx / (W * H)) % C;

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

    TENZOR_CUDA_KERNEL_LOOP(idx, total_size) {
        int64_t c = (idx / (W * H)) % C;

        float channel_mean = __half2float(mean[c]);
        float channel_var = __half2float(variance[c]);
        float invstd = rsqrtf(channel_var + epsilon);

        float result = (__half2float(input[idx]) - channel_mean) * invstd;
        output[idx] = float2half_sat(result);
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

    TENZOR_CUDA_KERNEL_LOOP(idx, total_size) {
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

    TENZOR_CUDA_KERNEL_LOOP(idx, total_size) {
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

    TENZOR_CUDA_KERNEL_LOOP(idx, total_size) {
        int64_t c = (idx / (H * W)) % C;

        float channel_mean = __half2float(mean[c]);
        float channel_var = __half2float(variance[c]);
        float invstd = rsqrtf(channel_var + epsilon);

        float normalized = (__half2float(input[idx]) - channel_mean) * invstd;
        float result = __half2float(gamma[c]) * normalized + __half2float(beta[c]);
        output[idx] = float2half_sat(result);
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

    const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_size; idx += grid_stride) {
        // Decode channel from NCHW layout
        int64_t c = (idx / HW) % C;

        // Compute scale and bias inline (avoids shared memory sync)
        float invstd = rsqrtf(variance[c] + epsilon);
        float scale = gamma[c] * invstd;
        float val = (input[idx] - mean[c]) * scale + beta[c];
        output[idx] = val;
    }
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

    const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_vec4; idx += grid_stride) {
        // Decode channel from NCHW layout (for vec4, multiply by 4)
        int64_t scalar_idx = idx * 4;
        int64_t c0 = (scalar_idx / HW) % C;

        // Load 4 elements
        float4 in = input[idx];

        // A float4 group can straddle a channel boundary when HW is not a multiple
        // of 4 (or when the group simply spans the end of one channel and the start
        // of the next). Compute the channel per-lane so each lane is normalized with
        // its own channel's statistics, matching the scalar/parallel/CPU path.
        int64_t c1 = ((scalar_idx + 1) / HW) % C;
        int64_t c2 = ((scalar_idx + 2) / HW) % C;
        int64_t c3 = ((scalar_idx + 3) / HW) % C;

        float4 out;
        {
            float m = mean[c0];
            float invstd = rsqrtf(variance[c0] + epsilon);
            out.x = gamma[c0] * (in.x - m) * invstd + beta[c0];
        }
        {
            float m = mean[c1];
            float invstd = rsqrtf(variance[c1] + epsilon);
            out.y = gamma[c1] * (in.y - m) * invstd + beta[c1];
        }
        {
            float m = mean[c2];
            float invstd = rsqrtf(variance[c2] + epsilon);
            out.z = gamma[c2] * (in.z - m) * invstd + beta[c2];
        }
        {
            float m = mean[c3];
            float invstd = rsqrtf(variance[c3] + epsilon);
            out.w = gamma[c3] * (in.w - m) * invstd + beta[c3];
        }

        output[idx] = out;
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
    TENZOR_CUDA_KERNEL_LOOP(c, C) {
        running_mean[c] = (T(1) - momentum) * running_mean[c] + momentum * batch_mean[c];
        running_var[c] = (T(1) - momentum) * running_var[c] + momentum * batch_var[c];
    }
}

// Float16 specialization: accumulate the EMA blend in Float32 to keep the
// running statistics from drifting versus the CPU reference (which keeps
// running stats in float). momentum is kept as a float scalar rather than
// being narrowed to __half, and each blend loads/stores via widen-narrow.
__global__ void batchnorm_update_running_stats_fp16_kernel(__half* running_mean,
                                                           __half* running_var,
                                                           const __half* batch_mean,
                                                           const __half* batch_var,
                                                           float momentum,
                                                           int64_t C) {
    TENZOR_CUDA_KERNEL_LOOP(c, C) {
        float r_mean = __half2float(running_mean[c]);
        float b_mean = __half2float(batch_mean[c]);
        running_mean[c] = __float2half((1.0f - momentum) * r_mean + momentum * b_mean);

        float r_var = __half2float(running_var[c]);
        float b_var = __half2float(batch_var[c]);
        running_var[c] = __float2half((1.0f - momentum) * r_var + momentum * b_var);
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

    // Use 2D chunked kernels when C < 64 to improve SM utilization
    constexpr int64_t CHUNKED_THRESHOLD = 64;
    bool use_chunked = (C < CHUNKED_THRESHOLD);

    if (input.dtype() == DType::Float32) {
        // Float32 statistics accumulate in DOUBLE (Acc=double) to match the CPU
        // reference (Welford in double). A float accumulator over N*H*W elements
        // diverges from CPU past the cross-backend tolerance for large spatial.
        if (use_chunked) {
            auto [grid_cm, bs_cm] = optimal_launch_config(
                batchnorm_mean_chunked_kernel<float, double>, N * H * W, (256 / 32) * sizeof(double));
            auto [grid_cv, bs_cv] = optimal_launch_config(
                batchnorm_variance_chunked_kernel<float, double>, N * H * W, (256 / 32) * sizeof(double));
            int bn_bs = std::max(32, std::min(std::min(bs_cm, bs_cv), 1024));
            bn_bs = (bn_bs / 32) * 32;
            int shared_mem_size = (bn_bs / 32) * sizeof(double);

            int64_t spatial_chunk_size = static_cast<int64_t>(bn_bs) * 4;
            int64_t spatial_chunks = (total_elements + spatial_chunk_size - 1) / spatial_chunk_size;
            dim3 grid(C, spatial_chunks);

            // Double scratch for the cross-block partial sums.
            Tensor mean_acc({C}, DType::Float64, input.device());
            Tensor var_acc({C}, DType::Float64, input.device());
            CUDA_CHECK(cudaMemsetAsync(mean_acc.data<double>(), 0, C * sizeof(double), stream));
            batchnorm_mean_chunked_kernel<float, double><<<grid, bn_bs, shared_mem_size, stream>>>(
                input.data<float>(), mean_acc.data<double>(), N, C, H, W, spatial_chunk_size);
            CUDA_CHECK(cudaGetLastError());
            int finalize_blocks = (C + 255) / 256;
            // mean_acc currently holds SUM; finalize into a double mean buffer
            // (reused: divide in place via dedicated kernel writing double).
            batchnorm_finalize_mean_kernel<double, double><<<finalize_blocks, 256, 0, stream>>>(
                mean_acc.data<double>(), mean_acc.data<double>(), C, total_elements);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemsetAsync(var_acc.data<double>(), 0, C * sizeof(double), stream));
            batchnorm_variance_chunked_kernel<float, double><<<grid, bn_bs, shared_mem_size, stream>>>(
                input.data<float>(), mean_acc.data<double>(), var_acc.data<double>(),
                N, C, H, W, spatial_chunk_size);
            CUDA_CHECK(cudaGetLastError());
            // Narrow mean/variance down to the float output tensors.
            batchnorm_finalize_mean_kernel<float, double><<<finalize_blocks, 256, 0, stream>>>(
                mean_acc.data<double>(), mean.data<float>(), C, 1);
            batchnorm_finalize_mean_kernel<float, double><<<finalize_blocks, 256, 0, stream>>>(
                var_acc.data<double>(), variance.data<float>(), C, total_elements);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_m, bs_m] = optimal_launch_config(
                batchnorm_mean_kernel<float, double>, N * H * W, (256 / 32) * sizeof(double));
            auto [grid_v, bs_v] = optimal_launch_config(
                batchnorm_variance_kernel<float, double>, N * H * W, (256 / 32) * sizeof(double));
            int bn_bs = std::max(32, std::min(std::min(bs_m, bs_v), 1024));
            bn_bs = (bn_bs / 32) * 32;
            int shared_mem_size = (bn_bs / 32) * sizeof(double);

            batchnorm_mean_kernel<float, double><<<C, bn_bs, shared_mem_size, stream>>>(
                input.data<float>(), mean.data<float>(), N, C, H, W);
            CUDA_CHECK(cudaGetLastError());

            batchnorm_variance_kernel<float, double><<<C, bn_bs, shared_mem_size, stream>>>(
                input.data<float>(), mean.data<float>(), variance.data<float>(), N, C, H, W);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (input.dtype() == DType::Float64) {
        if (use_chunked) {
            auto [grid_cm, bs_cm] = optimal_launch_config(
                batchnorm_mean_chunked_kernel<double, double>, N * H * W, (256 / 32) * sizeof(double));
            auto [grid_cv, bs_cv] = optimal_launch_config(
                batchnorm_variance_chunked_kernel<double, double>, N * H * W, (256 / 32) * sizeof(double));
            int bn_bs = std::max(32, std::min(std::min(bs_cm, bs_cv), 1024));
            bn_bs = (bn_bs / 32) * 32;
            int shared_mem_size = (bn_bs / 32) * sizeof(double);

            int64_t spatial_chunk_size = static_cast<int64_t>(bn_bs) * 4;
            int64_t spatial_chunks = (total_elements + spatial_chunk_size - 1) / spatial_chunk_size;
            dim3 grid(C, spatial_chunks);

            // mean tensor is double; accumulate partials directly into it.
            CUDA_CHECK(cudaMemsetAsync(mean.data<double>(), 0, C * sizeof(double), stream));
            batchnorm_mean_chunked_kernel<double, double><<<grid, bn_bs, shared_mem_size, stream>>>(
                input.data<double>(), mean.data<double>(), N, C, H, W, spatial_chunk_size);
            CUDA_CHECK(cudaGetLastError());
            int finalize_blocks = (C + 255) / 256;
            batchnorm_finalize_mean_kernel<double, double><<<finalize_blocks, 256, 0, stream>>>(
                mean.data<double>(), mean.data<double>(), C, total_elements);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemsetAsync(variance.data<double>(), 0, C * sizeof(double), stream));
            batchnorm_variance_chunked_kernel<double, double><<<grid, bn_bs, shared_mem_size, stream>>>(
                input.data<double>(), mean.data<double>(), variance.data<double>(),
                N, C, H, W, spatial_chunk_size);
            CUDA_CHECK(cudaGetLastError());
            batchnorm_finalize_mean_kernel<double, double><<<finalize_blocks, 256, 0, stream>>>(
                variance.data<double>(), variance.data<double>(), C, total_elements);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto [grid_m, bs_m] = optimal_launch_config(
                batchnorm_mean_kernel<double, double>, N * H * W, (256 / 32) * sizeof(double));
            auto [grid_v, bs_v] = optimal_launch_config(
                batchnorm_variance_kernel<double, double>, N * H * W, (256 / 32) * sizeof(double));
            int bn_bs = std::max(32, std::min(std::min(bs_m, bs_v), 1024));
            bn_bs = (bn_bs / 32) * 32;
            int shared_mem_size = (bn_bs / 32) * sizeof(double);

            batchnorm_mean_kernel<double, double><<<C, bn_bs, shared_mem_size, stream>>>(
                input.data<double>(), mean.data<double>(), N, C, H, W);
            CUDA_CHECK(cudaGetLastError());

            batchnorm_variance_kernel<double, double><<<C, bn_bs, shared_mem_size, stream>>>(
                input.data<double>(), mean.data<double>(), variance.data<double>(), N, C, H, W);
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (input.dtype() == DType::Float16) {
        // Compute statistics in Float32 to avoid a half-precision accumulator,
        // which loses precision / overflows and diverges from the CPU/PyTorch
        // reference. This runs on the GPU (widen the input, reuse the Float32
        // reduction path, narrow the resulting stats back to Float16) — it is NOT
        // a CPU fallback. Mirrors the ROCm BatchNorm widen-narrow pattern.
        Tensor input_f32 = input.to(DType::Float32);
        Tensor mean_f32 = mean.to(DType::Float32);
        Tensor variance_f32 = variance.to(DType::Float32);
        batchnorm2d_mean_var(input_f32, mean_f32, variance_f32, stream);
        mean = mean_f32.to(DType::Float16);
        variance = variance_f32.to(DType::Float16);
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
            int vec_blocks = get_num_blocks(total_vec4);
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
        batchnorm_update_running_stats_fp16_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<__half*>(running_mean.data<Float16>()),
            reinterpret_cast<__half*>(running_var.data<Float16>()),
            reinterpret_cast<const __half*>(batch_mean.data<Float16>()),
            reinterpret_cast<const __half*>(batch_var.data<Float16>()),
            momentum, C);
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
        // Widen to Float32, recurse (float accumulators / block_reduce_sum<float>),
        // then narrow back. The native __half path reduced gradients in half
        // precision (narrow accumulator), diverging from CPU and from the
        // forward pass. Mirrors the ROCm widen-narrow pattern.
        auto [gi, gg, gb] = batchnorm2d_backward(
            grad_output.to(DType::Float32), input.to(DType::Float32),
            mean.to(DType::Float32), variance.to(DType::Float32),
            gamma.to(DType::Float32), epsilon, stream);
        return std::make_tuple(gi.to(DType::Float16),
                               gg.to(DType::Float16),
                               gb.to(DType::Float16));
    } else {
        throw std::runtime_error("BatchNorm2D only supports Float32, Float64, and Float16 dtypes");
    }

    return std::make_tuple(grad_input, grad_gamma, grad_beta);
}


// ============================================================================
// GroupNorm CUDA Kernels
// ============================================================================

// Storage-type-aware load/store helpers used by the mixed-precision kernels.
// Allow FP16/BF16 storage with float32 internal accumulation in a single kernel,
// avoiding the previous "alloc f32 tensor + cast + run float kernel + cast back"
// round-trip pattern.
__device__ inline float gn_load(const float* p, int64_t i) { return p[i]; }
__device__ inline float gn_load(const __half* p, int64_t i) { return __half2float(p[i]); }
__device__ inline float gn_load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }

__device__ inline void gn_store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void gn_store(__half* p, int64_t i, float v) { p[i] = __float2half(v); }
__device__ inline void gn_store(__nv_bfloat16* p, int64_t i, float v) { p[i] = __float2bfloat16(v); }

// Per-channel grad_weight/grad_bias accumulation is done in Float32 only (the
// FP16/BF16 backward path widens to Float32 first), so no half-precision atomic
// add helpers are needed here — accumulating across N*HW contributions in half
// precision lost accuracy and could saturate.

// Mixed-precision GroupNorm forward: storage type StorageT (Float16/BFloat16),
// float32 internal accumulation, fused load/compute/store in a single kernel.
template<typename StorageT>
__global__ void group_norm_forward_kernel_mixed(
    const StorageT* __restrict__ input,
    const StorageT* __restrict__ weight,
    const StorageT* __restrict__ bias,
    StorageT* __restrict__ output,
    float* __restrict__ mean_out,
    float* __restrict__ inv_std_out,
    int64_t N, int64_t C, int64_t HW,
    int64_t num_groups, int64_t channels_per_group,
    float eps) {

    int64_t group_idx = blockIdx.x;
    int64_t n = group_idx / num_groups;
    int64_t g = group_idx % num_groups;
    if (n >= N || g >= num_groups) return;

    int64_t c_start = g * channels_per_group;
    int64_t group_size = channels_per_group * HW;

    // Mean reduction (float32 accumulator)
    float local_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int64_t c_offset = i / HW;
        int64_t hw = i % HW;
        int64_t c = c_start + c_offset;
        int64_t idx = (n * C + c) * HW + hw;
        local_sum += gn_load(input, idx);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    __shared__ float shared_sum[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    float mean = 0.0f;
    if (warp_id == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_sum = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        if (lane == 0) {
            mean = local_sum / static_cast<float>(group_size);
            shared_sum[0] = mean;
        }
    }
    __syncthreads();
    mean = shared_sum[0];

    // Variance reduction
    float local_var = 0.0f;
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int64_t c_offset = i / HW;
        int64_t hw = i % HW;
        int64_t c = c_start + c_offset;
        int64_t idx = (n * C + c) * HW + hw;
        float diff = gn_load(input, idx) - mean;
        local_var += diff * diff;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_var += __shfl_down_sync(0xFFFFFFFF, local_var, offset);
    }
    if (lane == 0) shared_sum[warp_id] = local_var;
    __syncthreads();

    float inv_std = 0.0f;
    if (warp_id == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        local_var = (lane < num_warps) ? shared_sum[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            local_var += __shfl_down_sync(0xFFFFFFFF, local_var, offset);
        }
        if (lane == 0) {
            float variance = local_var / static_cast<float>(group_size);
            inv_std = rsqrtf(variance + eps);
            shared_sum[0] = inv_std;
            // Save per-group statistics in float32 (matches the CPU reference,
            // which stores mean/inv_std as Float32 for the FP16/BF16 path). The
            // backward pass reads these back at full precision, avoiding the
            // ~3-significant-digit loss of FP16-saved stats.
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
        float normalized = (gn_load(input, idx) - mean) * inv_std;
        if (weight && bias) {
            float result = normalized * gn_load(weight, c) + gn_load(bias, c);
            gn_store(output, idx, result);
        } else {
            gn_store(output, idx, normalized);
        }
    }
}

// Note: the mixed-precision GroupNorm backward is no longer a dedicated kernel.
// The FP16/BF16 backward path widens its inputs to Float32, runs the Float32
// kernel (which accumulates grad_weight/grad_bias via atomicAdd<float>), then
// narrows the results — see group_norm_backward_kernel launcher. This avoids
// accumulating per-channel gradients in half precision (a CAS read-modify-write
// in FP16/BF16 lost precision and saturated for large N*HW).

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
    // Saved statistics precision: full precision for Float64, otherwise Float32.
    // For FP16/BF16 inputs the stats are stored in Float32 (matching the CPU
    // reference group_norm_kernel_with_stats), so the backward pass recovers
    // mean/inv_std at full precision instead of ~3 decimal digits.
    DType stats_dtype = (input.dtype() == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor mean_out({N * num_groups}, stats_dtype, input.device());
    Tensor inv_std_out({N * num_groups}, stats_dtype, input.device());

    int64_t num_group_instances = N * num_groups;
    int block_size = 256;

    // The kernels support no-affine GroupNorm/InstanceNorm via an internal
    // `if (weight && bias)` guard, so pass null when affine is absent instead of
    // dereferencing a zero-size storage (which would yield a non-null garbage
    // pointer and defeat the guard).
    const bool has_affine = weight.numel() > 0 && bias.numel() > 0;

    if (input.dtype() == DType::Float32) {
        group_norm_forward_kernel<float><<<num_group_instances, block_size, 0, stream>>>(
            input.data<float>(),
            has_affine ? weight.data<float>() : nullptr,
            has_affine ? bias.data<float>() : nullptr,
            output.data<float>(), mean_out.data<float>(), inv_std_out.data<float>(),
            N, C, HW, num_groups, channels_per_group, eps);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        group_norm_forward_kernel<double><<<num_group_instances, block_size, 0, stream>>>(
            input.data<double>(),
            has_affine ? weight.data<double>() : nullptr,
            has_affine ? bias.data<double>() : nullptr,
            output.data<double>(), mean_out.data<double>(), inv_std_out.data<double>(),
            N, C, HW, num_groups, channels_per_group, eps);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        // Mixed precision: native FP16 storage, float32 internal accumulation,
        // single fused kernel — no extra tensor allocations or cast launches.
        group_norm_forward_kernel_mixed<__half><<<num_group_instances, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data<Float16>()),
            has_affine ? reinterpret_cast<const __half*>(weight.data<Float16>()) : nullptr,
            has_affine ? reinterpret_cast<const __half*>(bias.data<Float16>()) : nullptr,
            reinterpret_cast<__half*>(output.data<Float16>()),
            mean_out.data<float>(),
            inv_std_out.data<float>(),
            N, C, HW, num_groups, channels_per_group, eps);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        group_norm_forward_kernel_mixed<__nv_bfloat16><<<num_group_instances, block_size, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data<BFloat16>()),
            has_affine ? reinterpret_cast<const __nv_bfloat16*>(weight.data<BFloat16>()) : nullptr,
            has_affine ? reinterpret_cast<const __nv_bfloat16*>(bias.data<BFloat16>()) : nullptr,
            reinterpret_cast<__nv_bfloat16*>(output.data<BFloat16>()),
            mean_out.data<float>(),
            inv_std_out.data<float>(),
            N, C, HW, num_groups, channels_per_group, eps);
            CUDA_CHECK(cudaGetLastError());
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

    // FP16/BF16: widen all data tensors to Float32, run the float32 backward
    // (float accumulators / atomicAdd<float> for grad_weight/grad_bias), then
    // narrow the results back. The saved mean/inv_std are already Float32
    // (group_norm_forward_kernel stores stats in Float32 for the half path), so
    // they pass straight through. This mirrors the BatchNorm2D widen-narrow
    // path above and the ROCm LayerNorm reference, and avoids accumulating the
    // N*HW per-channel grad_weight/grad_bias contributions in half precision.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        auto [gi, gw, gb] = group_norm_backward_kernel(
            grad_output.to(DType::Float32), input.to(DType::Float32),
            weight.to(DType::Float32), mean_saved, inv_std_saved,
            num_groups, stream);
        return std::make_tuple(gi.to(orig), gw.to(orig), gb.to(orig));
    }

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
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        group_norm_backward_kernel<double><<<num_group_instances, block_size, 0, stream>>>(
            grad_output.data<double>(), input.data<double>(), weight.data<double>(),
            mean_saved.data<double>(), inv_std_saved.data<double>(),
            grad_input.data<double>(), grad_weight.data<double>(), grad_bias.data<double>(),
            N, C, HW, num_groups, channels_per_group);
            CUDA_CHECK(cudaGetLastError());
    } else {
        // FP16/BF16 handled above via widen-recurse-narrow.
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
