#include "tenzor/core/tensor.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cub/cub.cuh>

namespace tenzor {
namespace cuda {

// Constants
constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_SIZE = 1024;
constexpr int REDUCTION_BLOCK_SIZE = 256;

// ============================================================================
// Type helpers for __half support
// ============================================================================

// Device-side helpers
template<typename T>
__device__ __forceinline__ T cuda_zero() { return T(0); }

template<>
__device__ __forceinline__ __half cuda_zero<__half>() { return __float2half(0.0f); }

template<typename T>
__device__ __forceinline__ T cuda_add(T a, T b) { return a + b; }

template<>
__device__ __forceinline__ __half cuda_add<__half>(__half a, __half b) { return __hadd(a, b); }

// Host-side helpers
template<typename T>
inline T host_zero() { return T(0); }

template<>
inline __half host_zero<__half>() { return __float2half(0.0f); }


// ============================================================================
// Helper kernels for GPU-side scalar operations (avoids D2H transfers)
// ============================================================================

template<typename T>
__global__ void divide_scalar_kernel(T* val, T divisor) {
    *val = *val / divisor;
}

template<typename T>
__global__ void pow_scalar_kernel(T* val, T exponent) {
    *val = pow(*val, exponent);
}

// Convert __half array to float array (for sorting/reduction on types CUB doesn't support)
__global__ void half_to_float_kernel(const __half* src, float* dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __half2float(src[idx]);
}

// Phase-2 reduction kernels: given block-level argmax/argmin indices,
// compare original values at those indices to find the global winner.
template<typename T>
__global__ void argmax_final_kernel(const T* input, const int64_t* block_indices, int64_t* output, int num_blocks) {
    __shared__ T shared_vals[REDUCTION_BLOCK_SIZE];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    T best_val;
    int64_t best_idx;

    if (tid < num_blocks) {
        best_idx = block_indices[tid];
        best_val = input[best_idx];
    } else {
        best_val = -FLT_MAX;
        best_idx = 0;
    }

    // Handle case where num_blocks > blockDim.x (unlikely but safe)
    for (int i = tid + blockDim.x; i < num_blocks; i += blockDim.x) {
        int64_t idx = block_indices[i];
        T val = input[idx];
        if (val > best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    shared_vals[tid] = best_val;
    shared_idxs[tid] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_vals[tid + stride] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = shared_idxs[0];
    }
}

template<typename T>
__global__ void argmin_final_kernel(const T* input, const int64_t* block_indices, int64_t* output, int num_blocks) {
    __shared__ T shared_vals[REDUCTION_BLOCK_SIZE];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    T best_val;
    int64_t best_idx;

    if (tid < num_blocks) {
        best_idx = block_indices[tid];
        best_val = input[best_idx];
    } else {
        best_val = FLT_MAX;
        best_idx = 0;
    }

    for (int i = tid + blockDim.x; i < num_blocks; i += blockDim.x) {
        int64_t idx = block_indices[i];
        T val = input[idx];
        if (val < best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    shared_vals[tid] = best_val;
    shared_idxs[tid] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_vals[tid + stride] < shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = shared_idxs[0];
    }
}

// Half-precision variants of final reduction kernels
__global__ void argmax_final_kernel_half(const __half* input, const int64_t* block_indices, int64_t* output, int num_blocks) {
    __shared__ float shared_vals[REDUCTION_BLOCK_SIZE];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    float best_val;
    int64_t best_idx;

    if (tid < num_blocks) {
        best_idx = block_indices[tid];
        best_val = __half2float(input[best_idx]);
    } else {
        best_val = -FLT_MAX;
        best_idx = 0;
    }

    for (int i = tid + blockDim.x; i < num_blocks; i += blockDim.x) {
        int64_t idx = block_indices[i];
        float val = __half2float(input[idx]);
        if (val > best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    shared_vals[tid] = best_val;
    shared_idxs[tid] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_vals[tid + stride] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = shared_idxs[0];
    }
}

__global__ void argmin_final_kernel_half(const __half* input, const int64_t* block_indices, int64_t* output, int num_blocks) {
    __shared__ float shared_vals[REDUCTION_BLOCK_SIZE];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    float best_val;
    int64_t best_idx;

    if (tid < num_blocks) {
        best_idx = block_indices[tid];
        best_val = __half2float(input[best_idx]);
    } else {
        best_val = FLT_MAX;
        best_idx = 0;
    }

    for (int i = tid + blockDim.x; i < num_blocks; i += blockDim.x) {
        int64_t idx = block_indices[i];
        float val = __half2float(input[idx]);
        if (val < best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    shared_vals[tid] = best_val;
    shared_idxs[tid] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_vals[tid + stride] < shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = shared_idxs[0];
    }
}
// ============================================================================
// Warp-level reduction primitives
// ============================================================================

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Specialization for __half
template<>
__device__ __forceinline__ __half warp_reduce_sum(__half val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = __hadd(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Helper for __half comparison (using __hgt and __hlt)
__device__ __forceinline__ __half cuda_max_val(__half a, __half b) {
    return __hgt(a, b) ? a : b;
}

__device__ __forceinline__ __half cuda_min_val(__half a, __half b) {
    return __hlt(a, b) ? a : b;
}

// Get negative infinity for __half
__device__ __forceinline__ __half half_neg_inf() {
    return __ushort_as_half(0xFC00);  // -inf in half precision
}

// Get positive infinity for __half
__device__ __forceinline__ __half half_pos_inf() {
    return __ushort_as_half(0x7C00);  // +inf in half precision
}

template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = (val > other) ? val : other;
    }
    return val;
}

// Specialization for __half
template<>
__device__ __forceinline__ __half warp_reduce_max(__half val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        __half other = __shfl_down_sync(0xffffffff, val, offset);
        val = cuda_max_val(val, other);
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T warp_reduce_min(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = (val < other) ? val : other;
    }
    return val;
}

// Specialization for __half
template<>
__device__ __forceinline__ __half warp_reduce_min(__half val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        __half other = __shfl_down_sync(0xffffffff, val, offset);
        val = cuda_min_val(val, other);
    }
    return val;
}

// ============================================================================
// Block-level reduction kernels (full reduction)
// ============================================================================

template<typename T>
__global__ void sum_reduce_kernel(const T* input, T* output, int64_t n) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Grid-stride loop for better occupancy
    T thread_sum = cuda_zero<T>();
    for (int64_t i = idx; i < n; i += grid_size) {
        thread_sum = cuda_add(thread_sum, input[i]);
    }

    shared[tid] = thread_sum;
    __syncthreads();

    // Block-level reduction in shared memory
    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = cuda_add(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    // Warp-level reduction
    if (tid < WARP_SIZE) {
        T val = shared[tid];
        val = warp_reduce_sum(val);

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

template<typename T>
__global__ void max_reduce_kernel(const T* input, T* output, int64_t n) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or negative infinity
    T thread_max = (idx < n) ? input[idx] : -FLT_MAX;

    // Grid-stride loop
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        T val = input[i];
        thread_max = (val > thread_max) ? val : thread_max;
    }

    shared[tid] = thread_max;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            T other = shared[tid + stride];
            shared[tid] = (shared[tid] > other) ? shared[tid] : other;
        }
        __syncthreads();
    }

    // Warp-level reduction
    if (tid < WARP_SIZE) {
        T val = shared[tid];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            if (tid + offset < blockDim.x) {
                T other = shared[tid + offset];
                val = (val > other) ? val : other;
            }
        }
        val = warp_reduce_max(val);

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

template<typename T>
__global__ void min_reduce_kernel(const T* input, T* output, int64_t n) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or positive infinity
    T thread_min = (idx < n) ? input[idx] : FLT_MAX;

    // Grid-stride loop
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        T val = input[i];
        thread_min = (val < thread_min) ? val : thread_min;
    }

    shared[tid] = thread_min;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            T other = shared[tid + stride];
            shared[tid] = (shared[tid] < other) ? shared[tid] : other;
        }
        __syncthreads();
    }

    // Warp-level reduction
    if (tid < WARP_SIZE) {
        T val = shared[tid];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            if (tid + offset < blockDim.x) {
                T other = shared[tid + offset];
                val = (val < other) ? val : other;
            }
        }
        val = warp_reduce_min(val);

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// ============================================================================
// Half-precision specializations for max/min reduction kernels
// ============================================================================

// Specialized max_reduce_kernel for __half
__global__ void max_reduce_kernel_half(const __half* input, __half* output, int64_t n) {
    __shared__ __half shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or negative infinity
    __half thread_max = (idx < n) ? input[idx] : half_neg_inf();

    // Grid-stride loop
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        __half val = input[i];
        thread_max = cuda_max_val(val, thread_max);
    }

    shared[tid] = thread_max;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            __half other = shared[tid + stride];
            shared[tid] = cuda_max_val(shared[tid], other);
        }
        __syncthreads();
    }

    // Warp-level reduction
    if (tid < WARP_SIZE) {
        __half val = shared[tid];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            if (tid + offset < blockDim.x) {
                __half other = shared[tid + offset];
                val = cuda_max_val(val, other);
            }
        }
        val = warp_reduce_max(val);

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// Specialized min_reduce_kernel for __half
__global__ void min_reduce_kernel_half(const __half* input, __half* output, int64_t n) {
    __shared__ __half shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or positive infinity
    __half thread_min = (idx < n) ? input[idx] : half_pos_inf();

    // Grid-stride loop
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        __half val = input[i];
        thread_min = cuda_min_val(val, thread_min);
    }

    shared[tid] = thread_min;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            __half other = shared[tid + stride];
            shared[tid] = cuda_min_val(shared[tid], other);
        }
        __syncthreads();
    }

    // Warp-level reduction
    if (tid < WARP_SIZE) {
        __half val = shared[tid];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            if (tid + offset < blockDim.x) {
                __half other = shared[tid + offset];
                val = cuda_min_val(val, other);
            }
        }
        val = warp_reduce_min(val);

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// ============================================================================
// Dimensional reduction kernels
// ============================================================================

// Sum reduction along a specific dimension
template<typename T>
__global__ void sum_along_dim_kernel(
    const T* input,
    T* output,
    const int64_t* input_shape,
    const int64_t* input_strides,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= output_size) return;

    // Compute multi-dimensional indices for output position
    int64_t indices[8];  // Support up to 8D tensors
    int64_t tmp = out_idx;

    for (int64_t d = 0; d < ndim; d++) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % input_shape[d];
        tmp /= input_shape[d];
    }

    // Sum along the reduction dimension
    T sum = cuda_zero<T>();
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;

        // Compute flat index
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }

        sum = cuda_add(sum, input[in_idx]);
    }

    output[out_idx] = sum;
}

// Max reduction along a specific dimension
template<typename T>
__global__ void max_along_dim_kernel(
    const T* input,
    T* output,
    const int64_t* input_shape,
    const int64_t* input_strides,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= output_size) return;

    // Compute multi-dimensional indices
    int64_t indices[8];
    int64_t tmp = out_idx;

    for (int64_t d = 0; d < ndim; d++) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % input_shape[d];
        tmp /= input_shape[d];
    }

    // Find max along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * input_strides[d];
    }
    T max_val = input[in_idx];

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }
        T val = input[in_idx];
        max_val = (val > max_val) ? val : max_val;
    }

    output[out_idx] = max_val;
}

// Min reduction along a specific dimension
template<typename T>
__global__ void min_along_dim_kernel(
    const T* input,
    T* output,
    const int64_t* input_shape,
    const int64_t* input_strides,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= output_size) return;

    // Compute multi-dimensional indices
    int64_t indices[8];
    int64_t tmp = out_idx;

    for (int64_t d = 0; d < ndim; d++) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % input_shape[d];
        tmp /= input_shape[d];
    }

    // Find min along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * input_strides[d];
    }
    T min_val = input[in_idx];

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }
        T val = input[in_idx];
        min_val = (val < min_val) ? val : min_val;
    }

    output[out_idx] = min_val;
}

// Specialized max_along_dim_kernel for __half
__global__ void max_along_dim_kernel_half(
    const __half* input,
    __half* output,
    const int64_t* input_shape,
    const int64_t* input_strides,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= output_size) return;

    // Compute multi-dimensional indices
    int64_t indices[8];
    int64_t tmp = out_idx;

    for (int64_t d = 0; d < ndim; d++) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % input_shape[d];
        tmp /= input_shape[d];
    }

    // Find max along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * input_strides[d];
    }
    __half max_val = input[in_idx];

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }
        __half val = input[in_idx];
        max_val = cuda_max_val(val, max_val);
    }

    output[out_idx] = max_val;
}

// Specialized min_along_dim_kernel for __half
__global__ void min_along_dim_kernel_half(
    const __half* input,
    __half* output,
    const int64_t* input_shape,
    const int64_t* input_strides,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= output_size) return;

    // Compute multi-dimensional indices
    int64_t indices[8];
    int64_t tmp = out_idx;

    for (int64_t d = 0; d < ndim; d++) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % input_shape[d];
        tmp /= input_shape[d];
    }

    // Find min along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * input_strides[d];
    }
    __half min_val = input[in_idx];

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }
        __half val = input[in_idx];
        min_val = cuda_min_val(val, min_val);
    }

    output[out_idx] = min_val;
}

// ============================================================================
// Helper functions
// ============================================================================

static auto compute_reduction_shape(
    const std::vector<int64_t>& input_shape,
    int64_t dim,
    bool keepdim
) -> std::vector<int64_t> {
    if (dim == INT64_MIN) {
        // Full reduction
        if (keepdim) {
            return std::vector<int64_t>(input_shape.size(), 1);
        }
        return {};  // Scalar (0D tensor)
    }

    // Normalize negative dimension
    int64_t normalized_dim = dim;
    if (dim < 0) {
        normalized_dim = static_cast<int64_t>(input_shape.size()) + dim;
        if (normalized_dim < 0 || normalized_dim >= static_cast<int64_t>(input_shape.size())) {
            throw std::invalid_argument("Dimension out of range in compute_reduction_shape");
        }
    }

    std::vector<int64_t> output_shape = input_shape;
    if (keepdim) {
        output_shape[normalized_dim] = 1;
    } else {
        output_shape.erase(output_shape.begin() + normalized_dim);
        // Keep empty shape for scalar result
    }
    return output_shape;
}

template<typename T>
static void launch_full_reduction_sum(const T* d_input, T* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        T zero = host_zero<T>();
        cudaMemcpyAsync(d_output, &zero, sizeof(T), cudaMemcpyHostToDevice, stream);
        return;
    }

    if (n == 1) {
        cudaMemcpyAsync(d_output, d_input, sizeof(T), cudaMemcpyDeviceToDevice, stream);
        return;
    }

    // Two-phase reduction for large arrays
    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        sum_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA kernel launch failed in sum_reduce_kernel: ") + cudaGetErrorString(err));
        }
    } else {
        // Phase 1: Reduce to num_blocks intermediate results
        T* d_temp;
        cudaMalloc(&d_temp, num_blocks * sizeof(T));
        sum_reduce_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA kernel launch failed in sum_reduce_kernel phase 1: ") + cudaGetErrorString(err));
        }

        // Phase 2: Final reduction
        sum_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA kernel launch failed in sum_reduce_kernel phase 2: ") + cudaGetErrorString(err));
        }

#if CUDART_VERSION >= 11020
        cudaFreeAsync(d_temp, stream);
#else
        cudaStreamSynchronize(stream);
        cudaFree(d_temp);
#endif
    }
    // Check for any errors during synchronization
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in sum reduction: ") + cudaGetErrorString(err));
    }
}

template<typename T>
static void launch_full_reduction_max(const T* d_input, T* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("max: input tensor is empty");
    }

    if (n == 1) {
        cudaMemcpyAsync(d_output, d_input, sizeof(T), cudaMemcpyDeviceToDevice, stream);
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        max_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
    } else {
        T* d_temp;
        cudaMalloc(&d_temp, num_blocks * sizeof(T));
        max_reduce_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        max_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);

#if CUDART_VERSION >= 11020
        cudaFreeAsync(d_temp, stream);
#else
        cudaStreamSynchronize(stream);
        cudaFree(d_temp);
#endif
    }
}

template<typename T>
static void launch_full_reduction_min(const T* d_input, T* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("min: input tensor is empty");
    }

    if (n == 1) {
        cudaMemcpyAsync(d_output, d_input, sizeof(T), cudaMemcpyDeviceToDevice, stream);
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        min_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
    } else {
        T* d_temp;
        cudaMalloc(&d_temp, num_blocks * sizeof(T));
        min_reduce_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        min_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);

#if CUDART_VERSION >= 11020
        cudaFreeAsync(d_temp, stream);
#else
        cudaStreamSynchronize(stream);
        cudaFree(d_temp);
#endif
    }
}

template<typename T>
static void launch_dim_reduction_sum(
    const T* d_input,
    T* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    // Compute output size
    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    // Copy shape and strides to device
    int64_t* d_shape;
    int64_t* d_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_shape, input_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, input_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Launch kernel
    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    sum_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size
    );

#if CUDART_VERSION >= 11020
    cudaFreeAsync(d_shape, nullptr);
    cudaFreeAsync(d_strides, nullptr);
#else
    cudaDeviceSynchronize();
    cudaFree(d_shape);
    cudaFree(d_strides);
#endif
}

template<typename T>
static void launch_dim_reduction_max(
    const T* d_input,
    T* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    int64_t* d_shape;
    int64_t* d_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_shape, input_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, input_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    max_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size
    );

#if CUDART_VERSION >= 11020
    cudaFreeAsync(d_shape, nullptr);
    cudaFreeAsync(d_strides, nullptr);
#else
    cudaDeviceSynchronize();
    cudaFree(d_shape);
    cudaFree(d_strides);
#endif
}

template<typename T>
static void launch_dim_reduction_min(
    const T* d_input,
    T* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    int64_t* d_shape;
    int64_t* d_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_shape, input_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, input_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    min_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size
    );

#if CUDART_VERSION >= 11020
    cudaFreeAsync(d_shape, nullptr);
    cudaFreeAsync(d_strides, nullptr);
#else
    cudaDeviceSynchronize();
    cudaFree(d_shape);
    cudaFree(d_strides);
#endif
}

// ============================================================================
// Half-precision specialized launch functions
// ============================================================================

static void launch_full_reduction_max_half(const __half* d_input, __half* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("max: input tensor is empty");
    }

    if (n == 1) {
        cudaMemcpyAsync(d_output, d_input, sizeof(__half), cudaMemcpyDeviceToDevice, stream);
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        max_reduce_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
    } else {
        __half* d_temp;
        cudaMalloc(&d_temp, num_blocks * sizeof(__half));
        max_reduce_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        max_reduce_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);

#if CUDART_VERSION >= 11020
        cudaFreeAsync(d_temp, stream);
#else
        cudaStreamSynchronize(stream);
        cudaFree(d_temp);
#endif
    }
}

static void launch_full_reduction_min_half(const __half* d_input, __half* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("min: input tensor is empty");
    }

    if (n == 1) {
        cudaMemcpyAsync(d_output, d_input, sizeof(__half), cudaMemcpyDeviceToDevice, stream);
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        min_reduce_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
    } else {
        __half* d_temp;
        cudaMalloc(&d_temp, num_blocks * sizeof(__half));
        min_reduce_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        min_reduce_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);

#if CUDART_VERSION >= 11020
        cudaFreeAsync(d_temp, stream);
#else
        cudaStreamSynchronize(stream);
        cudaFree(d_temp);
#endif
    }
}

static void launch_dim_reduction_max_half(
    const __half* d_input,
    __half* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    int64_t* d_shape;
    int64_t* d_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_shape, input_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, input_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    max_along_dim_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size
    );

#if CUDART_VERSION >= 11020
    cudaFreeAsync(d_shape, nullptr);
    cudaFreeAsync(d_strides, nullptr);
#else
    cudaDeviceSynchronize();
    cudaFree(d_shape);
    cudaFree(d_strides);
#endif
}

static void launch_dim_reduction_min_half(
    const __half* d_input,
    __half* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    int64_t* d_shape;
    int64_t* d_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_shape, input_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, input_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    min_along_dim_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size
    );

#if CUDART_VERSION >= 11020
    cudaFreeAsync(d_shape, nullptr);
    cudaFreeAsync(d_strides, nullptr);
#else
    cudaDeviceSynchronize();
    cudaFree(d_shape);
    cudaFree(d_strides);
#endif
}

// ============================================================================
// Public API
// ============================================================================

auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize negative dimension
    int64_t normalized_dim = dim;
    if (dim != INT64_MIN) {  // INT64_MIN means reduce all dimensions
        if (dim < 0) {
            normalized_dim = ndim + dim;
        }
        // Validate dimension is within bounds
        if (normalized_dim < 0 || normalized_dim >= ndim) {
            throw std::runtime_error("Dimension " + std::to_string(dim) +
                " out of range for tensor with " + std::to_string(ndim) + " dimensions");
        }
    }

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        normalized_dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            if (dim == INT64_MIN) {  // Full reduction
                launch_full_reduction_sum(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_sum(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (dim == INT64_MIN) {  // Full reduction
                launch_full_reduction_sum(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_sum(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int32_t>();

            if (dim == INT64_MIN) {  // Full reduction
                launch_full_reduction_sum(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_sum(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {  // Full reduction
                launch_full_reduction_sum(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_sum(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        case DType::Float16: {
            auto* input_data = reinterpret_cast<const __half*>(input.data_ptr());
            auto* output_data = reinterpret_cast<__half*>(output.data_ptr());

            if (dim == INT64_MIN) {  // Full reduction
                launch_full_reduction_sum(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_sum(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        default:
            throw std::runtime_error("sum: unsupported dtype");
    }

    cudaStreamSynchronize(stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in sum_kernel: ") + cudaGetErrorString(err));
    }

    return output;
}

// Scaling kernel for mean computation
template<typename T>
__global__ void scale_kernel(T* data, int64_t n, T scale) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

// Specialization for __half (no *= operator)
template<>
__global__ void scale_kernel<__half>(__half* data, int64_t n, __half scale) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = __float2half(__half2float(data[idx]) * __half2float(scale));
    }
}

auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    const auto dtype = input.dtype();

    if (dtype != DType::Float32 && dtype != DType::Float64 && dtype != DType::Float16) {
        throw std::runtime_error("mean: only Float32, Float64, and Float16 are supported");
    }

    // Compute sum first
    auto sum_result = sum_kernel(input, dim, keepdim, stream);

    // Compute count
    int64_t count;
    if (dim < 0) {
        count = input.numel();
    } else {
        count = input.shape()[dim];
    }

    if (count == 0) {
        // Return 0 for mean of empty tensor - this is practical for loss computation
        // in object detection where no samples may be selected
        return sum_result;  // sum_result is already 0 for empty tensor
    }

    // Divide by count (in-place) using proper CUDA kernel
    const int64_t n = sum_result.numel();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    if (dtype == DType::Float32) {
        auto* data = sum_result.data<float>();
        const float scale = 1.0f / static_cast<float>(count);
        scale_kernel<<<grid_size, block_size, 0, stream>>>(data, n, scale);
    } else if (dtype == DType::Float64) {
        auto* data = sum_result.data<double>();
        const double scale = 1.0 / static_cast<double>(count);
        scale_kernel<<<grid_size, block_size, 0, stream>>>(data, n, scale);
    } else {  // Float16
        auto* data = reinterpret_cast<__half*>(sum_result.data_ptr());
        const __half scale = __float2half(1.0f / static_cast<float>(count));
        scale_kernel<<<grid_size, block_size, 0, stream>>>(data, n, scale);
    }

    cudaStreamSynchronize(stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in mean_kernel: ") + cudaGetErrorString(err));
    }

    return sum_result;
}

auto max_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            if (dim < 0) {
                launch_full_reduction_max(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_max(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim
                );
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (dim < 0) {
                launch_full_reduction_max(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_max(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim
                );
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int32_t>();

            if (dim < 0) {
                launch_full_reduction_max(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_max(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim
                );
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (dim < 0) {
                launch_full_reduction_max(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_max(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim
                );
            }
            break;
        }
        case DType::Float16: {
            auto* input_data = reinterpret_cast<const __half*>(input.data_ptr());
            auto* output_data = reinterpret_cast<__half*>(output.data_ptr());

            if (dim < 0) {
                launch_full_reduction_max_half(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_max_half(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim
                );
            }
            break;
        }
        default:
            throw std::runtime_error("max: unsupported dtype");
    }

    cudaStreamSynchronize(stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in max_kernel: ") + cudaGetErrorString(err));
    }

    return output;
}

auto min_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            if (dim < 0) {
                launch_full_reduction_min(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_min(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim
                );
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (dim < 0) {
                launch_full_reduction_min(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_min(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim
                );
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int32_t>();

            if (dim < 0) {
                launch_full_reduction_min(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_min(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim
                );
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (dim < 0) {
                launch_full_reduction_min(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_min(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim
                );
            }
            break;
        }
        case DType::Float16: {
            auto* input_data = reinterpret_cast<const __half*>(input.data_ptr());
            auto* output_data = reinterpret_cast<__half*>(output.data_ptr());

            if (dim < 0) {
                launch_full_reduction_min_half(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_min_half(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim
                );
            }
            break;
        }
        default:
            throw std::runtime_error("min: unsupported dtype");
    }

    cudaStreamSynchronize(stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in min_kernel: ") + cudaGetErrorString(err));
    }

    return output;
}

// ============================================================================
// Argmax/Argmin operations (return indices, not values)
// ============================================================================

// Argmax kernel for full reduction
template<typename T>
__global__ void argmax_full_kernel(const T* input, int64_t* output, int64_t n) {
    __shared__ T shared_vals[REDUCTION_BLOCK_SIZE];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or negative infinity
    T thread_max = (idx < n) ? input[idx] : -FLT_MAX;
    int64_t thread_idx = (idx < n) ? idx : 0;

    // Grid-stride loop to find local maximum
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        T val = input[i];
        if (val > thread_max) {
            thread_max = val;
            thread_idx = i;
        }
    }

    shared_vals[tid] = thread_max;
    shared_idxs[tid] = thread_idx;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            if (shared_vals[tid + stride] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    // Warp-level reduction (last 32 threads)
    if (tid < 32) {
        T val = shared_vals[tid];
        int64_t val_idx = shared_idxs[tid];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            T other_val = __shfl_down_sync(0xffffffff, val, offset);
            int64_t other_idx = __shfl_down_sync(0xffffffff, val_idx, offset);
            if (other_val > val) {
                val = other_val;
                val_idx = other_idx;
            }
        }

        if (tid == 0) {
            output[blockIdx.x] = val_idx;
        }
    }
}

// Argmin kernel for full reduction
template<typename T>
__global__ void argmin_full_kernel(const T* input, int64_t* output, int64_t n) {
    __shared__ T shared_vals[REDUCTION_BLOCK_SIZE];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or positive infinity
    T thread_min = (idx < n) ? input[idx] : FLT_MAX;
    int64_t thread_idx = (idx < n) ? idx : 0;

    // Grid-stride loop to find local minimum
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        T val = input[i];
        if (val < thread_min) {
            thread_min = val;
            thread_idx = i;
        }
    }

    shared_vals[tid] = thread_min;
    shared_idxs[tid] = thread_idx;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            if (shared_vals[tid + stride] < shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    // Warp-level reduction (last 32 threads)
    if (tid < 32) {
        T val = shared_vals[tid];
        int64_t val_idx = shared_idxs[tid];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            T other_val = __shfl_down_sync(0xffffffff, val, offset);
            int64_t other_idx = __shfl_down_sync(0xffffffff, val_idx, offset);
            if (other_val < val) {
                val = other_val;
                val_idx = other_idx;
            }
        }

        if (tid == 0) {
            output[blockIdx.x] = val_idx;
        }
    }
}

// Argmax along a specific dimension
template<typename T>
__global__ void argmax_along_dim_kernel(
    const T* input,
    int64_t* output,
    const int64_t* input_shape,
    const int64_t* input_strides,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= output_size) return;

    // Compute multi-dimensional indices
    int64_t indices[8];
    int64_t tmp = out_idx;

    for (int64_t d = 0; d < ndim; d++) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % input_shape[d];
        tmp /= input_shape[d];
    }

    // Find argmax along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * input_strides[d];
    }
    T max_val = input[in_idx];
    int64_t max_idx = 0;

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }
        T val = input[in_idx];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    output[out_idx] = max_idx;
}

// Argmin along a specific dimension
template<typename T>
__global__ void argmin_along_dim_kernel(
    const T* input,
    int64_t* output,
    const int64_t* input_shape,
    const int64_t* input_strides,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= output_size) return;

    // Compute multi-dimensional indices
    int64_t indices[8];
    int64_t tmp = out_idx;

    for (int64_t d = 0; d < ndim; d++) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % input_shape[d];
        tmp /= input_shape[d];
    }

    // Find argmin along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * input_strides[d];
    }
    T min_val = input[in_idx];
    int64_t min_idx = 0;

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }
        T val = input[in_idx];
        if (val < min_val) {
            min_val = val;
            min_idx = i;
        }
    }

    output[out_idx] = min_idx;
}

// ============================================================================
// Half-precision specializations for argmax/argmin kernels
// ============================================================================

// Specialized argmax_full_kernel for __half
__global__ void argmax_full_kernel_half(const __half* input, int64_t* output, int64_t n) {
    __shared__ __half shared_vals[REDUCTION_BLOCK_SIZE];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or negative infinity
    __half thread_max = (idx < n) ? input[idx] : half_neg_inf();
    int64_t thread_idx = (idx < n) ? idx : 0;

    // Grid-stride loop to find local maximum
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        __half val = input[i];
        if (__hgt(val, thread_max)) {
            thread_max = val;
            thread_idx = i;
        }
    }

    shared_vals[tid] = thread_max;
    shared_idxs[tid] = thread_idx;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            if (__hgt(shared_vals[tid + stride], shared_vals[tid])) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    // Warp-level reduction (last 32 threads)
    if (tid < 32) {
        __half val = shared_vals[tid];
        int64_t val_idx = shared_idxs[tid];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            __half other_val = __shfl_down_sync(0xffffffff, val, offset);
            int64_t other_idx = __shfl_down_sync(0xffffffff, val_idx, offset);
            if (__hgt(other_val, val)) {
                val = other_val;
                val_idx = other_idx;
            }
        }

        if (tid == 0) {
            output[blockIdx.x] = val_idx;
        }
    }
}

// Specialized argmin_full_kernel for __half
__global__ void argmin_full_kernel_half(const __half* input, int64_t* output, int64_t n) {
    __shared__ __half shared_vals[REDUCTION_BLOCK_SIZE];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or positive infinity
    __half thread_min = (idx < n) ? input[idx] : half_pos_inf();
    int64_t thread_idx = (idx < n) ? idx : 0;

    // Grid-stride loop to find local minimum
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        __half val = input[i];
        if (__hlt(val, thread_min)) {
            thread_min = val;
            thread_idx = i;
        }
    }

    shared_vals[tid] = thread_min;
    shared_idxs[tid] = thread_idx;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            if (__hlt(shared_vals[tid + stride], shared_vals[tid])) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    // Warp-level reduction (last 32 threads)
    if (tid < 32) {
        __half val = shared_vals[tid];
        int64_t val_idx = shared_idxs[tid];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            __half other_val = __shfl_down_sync(0xffffffff, val, offset);
            int64_t other_idx = __shfl_down_sync(0xffffffff, val_idx, offset);
            if (__hlt(other_val, val)) {
                val = other_val;
                val_idx = other_idx;
            }
        }

        if (tid == 0) {
            output[blockIdx.x] = val_idx;
        }
    }
}

// Specialized argmax_along_dim_kernel for __half
__global__ void argmax_along_dim_kernel_half(
    const __half* input,
    int64_t* output,
    const int64_t* input_shape,
    const int64_t* input_strides,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= output_size) return;

    // Compute multi-dimensional indices
    int64_t indices[8];
    int64_t tmp = out_idx;

    for (int64_t d = 0; d < ndim; d++) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % input_shape[d];
        tmp /= input_shape[d];
    }

    // Find argmax along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * input_strides[d];
    }
    __half max_val = input[in_idx];
    int64_t max_idx = 0;

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }
        __half val = input[in_idx];
        if (__hgt(val, max_val)) {
            max_val = val;
            max_idx = i;
        }
    }

    output[out_idx] = max_idx;
}

// Specialized argmin_along_dim_kernel for __half
__global__ void argmin_along_dim_kernel_half(
    const __half* input,
    int64_t* output,
    const int64_t* input_shape,
    const int64_t* input_strides,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= output_size) return;

    // Compute multi-dimensional indices
    int64_t indices[8];
    int64_t tmp = out_idx;

    for (int64_t d = 0; d < ndim; d++) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % input_shape[d];
        tmp /= input_shape[d];
    }

    // Find argmin along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * input_strides[d];
    }
    __half min_val = input[in_idx];
    int64_t min_idx = 0;

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }
        __half val = input[in_idx];
        if (__hlt(val, min_val)) {
            min_val = val;
            min_idx = i;
        }
    }

    output[out_idx] = min_idx;
}

// Helper function to launch argmax reduction
template<typename T>
static void launch_full_argmax(const T* d_input, int64_t* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("argmax: input tensor is empty");
    }

    if (n == 1) {
        int64_t zero = 0;
        cudaMemcpyAsync(d_output, &zero, sizeof(int64_t), cudaMemcpyHostToDevice, stream);
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        argmax_full_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
    } else {
        // Two-phase GPU-only reduction
        int64_t* d_temp;
        cudaMalloc(&d_temp, num_blocks * sizeof(int64_t));
        argmax_full_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);

        // Phase 2: compare block results entirely on GPU
        argmax_final_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, d_output, num_blocks);
        cudaFree(d_temp);
    }
}
// Helper function to launch argmin reduction
template<typename T>
static void launch_full_argmin(const T* d_input, int64_t* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("argmin: input tensor is empty");
    }

    if (n == 1) {
        int64_t zero = 0;
        cudaMemcpyAsync(d_output, &zero, sizeof(int64_t), cudaMemcpyHostToDevice, stream);
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        argmin_full_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
    } else {
        // Two-phase GPU-only reduction
        int64_t* d_temp;
        cudaMalloc(&d_temp, num_blocks * sizeof(int64_t));
        argmin_full_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);

        // Phase 2: compare block results entirely on GPU
        argmin_final_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, d_output, num_blocks);
        cudaFree(d_temp);
    }
}


// Helper function for dimensional argmax
template<typename T>
static void launch_dim_argmax(
    const T* d_input,
    int64_t* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    // Copy shape and strides to device
    int64_t* d_shape;
    int64_t* d_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_shape, input_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, input_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    argmax_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size
    );

#if CUDART_VERSION >= 11020
    cudaFreeAsync(d_shape, nullptr);
    cudaFreeAsync(d_strides, nullptr);
#else
    cudaDeviceSynchronize();
    cudaFree(d_shape);
    cudaFree(d_strides);
#endif
}

// Helper function for dimensional argmin
template<typename T>
static void launch_dim_argmin(
    const T* d_input,
    int64_t* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    // Copy shape and strides to device
    int64_t* d_shape;
    int64_t* d_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_shape, input_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, input_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    argmin_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size
    );

#if CUDART_VERSION >= 11020
    cudaFreeAsync(d_shape, nullptr);
    cudaFreeAsync(d_strides, nullptr);
#else
    cudaDeviceSynchronize();
    cudaFree(d_shape);
    cudaFree(d_strides);
#endif
}

// ============================================================================
// Half-precision specialized launch functions for argmax/argmin
// ============================================================================

static void launch_full_argmax_half(const __half* d_input, int64_t* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("argmax: input tensor is empty");
    }

    if (n == 1) {
        int64_t zero = 0;
        cudaMemcpyAsync(d_output, &zero, sizeof(int64_t), cudaMemcpyHostToDevice, stream);
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        argmax_full_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
    } else {
        // Two-phase GPU-only reduction
        int64_t* d_temp;
        cudaMalloc(&d_temp, num_blocks * sizeof(int64_t));
        argmax_full_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);

        // Phase 2: compare block results entirely on GPU
        argmax_final_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, d_output, num_blocks);
        cudaFree(d_temp);
    }
}

static void launch_full_argmin_half(const __half* d_input, int64_t* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("argmin: input tensor is empty");
    }

    if (n == 1) {
        int64_t zero = 0;
        cudaMemcpyAsync(d_output, &zero, sizeof(int64_t), cudaMemcpyHostToDevice, stream);
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        argmin_full_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
    } else {
        // Two-phase GPU-only reduction
        int64_t* d_temp;
        cudaMalloc(&d_temp, num_blocks * sizeof(int64_t));
        argmin_full_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);

        // Phase 2: compare block results entirely on GPU
        argmin_final_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, d_output, num_blocks);
        cudaFree(d_temp);
    }
}


static void launch_dim_argmax_half(
    const __half* d_input,
    int64_t* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    // Copy shape and strides to device
    int64_t* d_shape;
    int64_t* d_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_shape, input_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, input_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    argmax_along_dim_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size
    );

#if CUDART_VERSION >= 11020
    cudaFreeAsync(d_shape, nullptr);
    cudaFreeAsync(d_strides, nullptr);
#else
    cudaDeviceSynchronize();
    cudaFree(d_shape);
    cudaFree(d_strides);
#endif
}

static void launch_dim_argmin_half(
    const __half* d_input,
    int64_t* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    // Copy shape and strides to device
    int64_t* d_shape;
    int64_t* d_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_shape, input_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, input_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    argmin_along_dim_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size
    );

#if CUDART_VERSION >= 11020
    cudaFreeAsync(d_shape, nullptr);
    cudaFreeAsync(d_strides, nullptr);
#else
    cudaDeviceSynchronize();
    cudaFree(d_shape);
    cudaFree(d_strides);
#endif
}

// Public API for argmax
auto argmax_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();

    // Normalize negative dimension
    int64_t normalized_dim = dim;
    if (dim < 0 && dim != INT64_MIN) {  // INT64_MIN means reduce all dimensions
        normalized_dim = static_cast<int64_t>(input_shape.size()) + dim;
        if (normalized_dim < 0 || normalized_dim >= static_cast<int64_t>(input_shape.size())) {
            throw std::invalid_argument("Dimension out of range in argmax_kernel");
        }
    }

    // Compute output shape (always Int64 for indices)
    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, DType::Int64, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmax(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmax(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmax(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmax(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmax(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmax(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmax(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmax(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        case DType::Float16: {
            auto* input_data = reinterpret_cast<const __half*>(input.data_ptr());
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmax_half(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmax_half(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        default:
            throw std::runtime_error("argmax: only Float32, Float64, Float16, Int32, and Int64 are supported");
    }

    cudaStreamSynchronize(stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in argmax_kernel: ") + cudaGetErrorString(err));
    }

    return output;
}

// Public API for argmin
auto argmin_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();

    // Normalize negative dimension
    int64_t normalized_dim = dim;
    if (dim < 0 && dim != INT64_MIN) {  // INT64_MIN means reduce all dimensions
        normalized_dim = static_cast<int64_t>(input_shape.size()) + dim;
        if (normalized_dim < 0 || normalized_dim >= static_cast<int64_t>(input_shape.size())) {
            throw std::invalid_argument("Dimension out of range in argmin_kernel");
        }
    }

    // Compute output shape (always Int64 for indices)
    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, DType::Int64, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmin(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmin(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmin(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmin(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmin(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmin(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmin(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmin(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        case DType::Float16: {
            auto* input_data = reinterpret_cast<const __half*>(input.data_ptr());
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmin_half(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmin_half(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        default:
            throw std::runtime_error("argmin: only Float32, Float64, Float16, Int32, and Int64 are supported");
    }

    cudaStreamSynchronize(stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in argmin_kernel: ") + cudaGetErrorString(err));
    }

    return output;
}

// ============================================================================
// Product reduction operation
// ============================================================================

// Product reduction kernel (full reduction only)
template<typename T>
__global__ void prod_reduce_kernel(const T* input, T* output, int64_t n) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with 1 (identity for multiplication)
    T thread_prod = 1;

    // Grid-stride loop
    for (int64_t i = idx; i < n; i += grid_size) {
        thread_prod *= input[i];
    }

    shared[tid] = thread_prod;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            shared[tid] *= shared[tid + stride];
        }
        __syncthreads();
    }

    // Warp-level reduction
    if (tid < WARP_SIZE) {
        T val = shared[tid];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val *= __shfl_down_sync(0xffffffff, val, offset);
        }

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// Product reduction along a specific dimension
template<typename T>
__global__ void prod_along_dim_kernel(
    const T* input,
    T* output,
    const int64_t* input_shape,
    const int64_t* input_strides,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= output_size) return;

    // Compute multi-dimensional indices
    int64_t indices[8];
    int64_t tmp = out_idx;

    for (int64_t d = 0; d < ndim; d++) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % input_shape[d];
        tmp /= input_shape[d];
    }

    // Product along the reduction dimension
    T prod_val = 1;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;

        // Compute flat index
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }

        prod_val *= input[in_idx];
    }

    output[out_idx] = prod_val;
}

// Helper function to launch product reduction
template<typename T>
static void launch_full_reduction_prod(const T* d_input, T* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        T one = 1;
        cudaMemcpyAsync(d_output, &one, sizeof(T), cudaMemcpyHostToDevice, stream);
        return;
    }

    if (n == 1) {
        cudaMemcpyAsync(d_output, d_input, sizeof(T), cudaMemcpyDeviceToDevice, stream);
        return;
    }

    // Two-phase reduction for large arrays
    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        prod_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
    } else {
        // Phase 1: Reduce to num_blocks intermediate results
        T* d_temp;
        cudaMalloc(&d_temp, num_blocks * sizeof(T));
        prod_reduce_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);

        // Phase 2: Final reduction
        prod_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);

#if CUDART_VERSION >= 11020
        cudaFreeAsync(d_temp, stream);
#else
        cudaStreamSynchronize(stream);
        cudaFree(d_temp);
#endif
    }
}

// Helper function for dimensional product reduction
template<typename T>
static void launch_dim_reduction_prod(
    const T* d_input,
    T* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    // Copy shape and strides to device
    int64_t* d_shape;
    int64_t* d_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_shape, input_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, input_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    prod_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size
    );

#if CUDART_VERSION >= 11020
    cudaFreeAsync(d_shape, nullptr);
    cudaFreeAsync(d_strides, nullptr);
#else
    cudaDeviceSynchronize();
    cudaFree(d_shape);
    cudaFree(d_strides);
#endif
}

// Public API for prod (product reduction)
auto prod_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();

    // Normalize negative dimension
    int64_t normalized_dim = dim;
    if (dim < 0 && dim != INT64_MIN) {  // INT64_MIN means reduce all dimensions
        normalized_dim = static_cast<int64_t>(input_shape.size()) + dim;
        if (normalized_dim < 0 || normalized_dim >= static_cast<int64_t>(input_shape.size())) {
            throw std::invalid_argument("Dimension out of range in prod_kernel");
        }
    }

    // Compute output shape
    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            if (dim == INT64_MIN) {
                launch_full_reduction_prod(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_prod(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (dim == INT64_MIN) {
                launch_full_reduction_prod(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_prod(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int32_t>();

            if (dim == INT64_MIN) {
                launch_full_reduction_prod(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_prod(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_reduction_prod(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_prod(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        default:
            throw std::runtime_error("prod: only Float32, Float64, Int32, and Int64 are supported");
    }

    cudaStreamSynchronize(stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in prod_kernel: ") + cudaGetErrorString(err));
    }

    return output;
}

// ============================================================================
// Variance and Standard Deviation operations (Welford single-pass)
// ============================================================================

// Device helper: combine two Welford accumulators
template<typename T>
__device__ __forceinline__ void welford_combine(
    T& mean_a, T& m2_a, int64_t& count_a,
    T mean_b, T m2_b, int64_t count_b
) {
    if (count_b == 0) return;
    int64_t count_ab = count_a + count_b;
    T delta = mean_b - mean_a;
    T new_mean = mean_a + delta * static_cast<T>(count_b) / static_cast<T>(count_ab);
    m2_a += m2_b + delta * delta * static_cast<T>(count_a) * static_cast<T>(count_b) / static_cast<T>(count_ab);
    mean_a = new_mean;
    count_a = count_ab;
}

// Welford single-pass variance kernel — per-block (mean, M2, count) output
template<typename T>
__global__ void welford_variance_kernel(
    const T* input,
    T* block_means,
    T* block_m2s,
    int64_t* block_counts,
    int64_t n
) {
    __shared__ T s_mean[REDUCTION_BLOCK_SIZE];
    __shared__ T s_m2[REDUCTION_BLOCK_SIZE];
    __shared__ int64_t s_count[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Per-thread Welford accumulation
    T t_mean = 0;
    T t_m2 = 0;
    int64_t t_count = 0;

    for (int64_t i = idx; i < n; i += grid_size) {
        t_count++;
        T delta = input[i] - t_mean;
        t_mean += delta / static_cast<T>(t_count);
        T delta2 = input[i] - t_mean;
        t_m2 += delta * delta2;
    }

    s_mean[tid] = t_mean;
    s_m2[tid] = t_m2;
    s_count[tid] = t_count;
    __syncthreads();

    // Block-level tree reduction using Welford combine
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            welford_combine(s_mean[tid], s_m2[tid], s_count[tid],
                            s_mean[tid + stride], s_m2[tid + stride], s_count[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_means[blockIdx.x] = s_mean[0];
        block_m2s[blockIdx.x] = s_m2[0];
        block_counts[blockIdx.x] = s_count[0];
    }
}

// Final merge kernel: combines per-block Welford results into a single variance
template<typename T>
__global__ void welford_finalize_kernel(
    const T* block_means,
    const T* block_m2s,
    const int64_t* block_counts,
    T* output,
    int num_blocks,
    int64_t correction
) {
    T mean = block_means[0];
    T m2 = block_m2s[0];
    int64_t count = block_counts[0];

    for (int i = 1; i < num_blocks; ++i) {
        welford_combine(mean, m2, count, block_means[i], block_m2s[i], block_counts[i]);
    }

    int64_t divisor = count - correction;
    if (divisor <= 0) divisor = 1;
    output[0] = m2 / static_cast<T>(divisor);
}

// Helper function to compute variance — single-pass Welford, zero D2H transfers
template<typename T>
static void launch_variance_computation(const T* d_input, T* d_output, int64_t n, int64_t correction, cudaStream_t stream = nullptr) {
    if (n == 0) {
        T zero = 0;
        cudaMemcpyAsync(d_output, &zero, sizeof(T), cudaMemcpyHostToDevice, stream);
        return;
    }

    if (n == 1) {
        T zero = 0;
        cudaMemcpyAsync(d_output, &zero, sizeof(T), cudaMemcpyHostToDevice, stream);
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    // Allocate per-block Welford accumulators
    T* d_block_means;
    T* d_block_m2s;
    int64_t* d_block_counts;
    cudaMalloc(&d_block_means, num_blocks * sizeof(T));
    cudaMalloc(&d_block_m2s, num_blocks * sizeof(T));
    cudaMalloc(&d_block_counts, num_blocks * sizeof(int64_t));

    // Phase 1: Per-block Welford reduction
    welford_variance_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_block_means, d_block_m2s, d_block_counts, n);

    // Phase 2: Merge block results and compute final variance (single thread)
    welford_finalize_kernel<<<1, 1, 0, stream>>>(
        d_block_means, d_block_m2s, d_block_counts, d_output, num_blocks, correction);

#if CUDART_VERSION >= 11020
    cudaFreeAsync(d_block_means, stream);
    cudaFreeAsync(d_block_m2s, stream);
    cudaFreeAsync(d_block_counts, stream);
#else
    cudaStreamSynchronize(stream);
    cudaFree(d_block_means);
    cudaFree(d_block_m2s);
    cudaFree(d_block_counts);
#endif
}

// Public API for variance
auto var_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction, cudaStream_t stream) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();

    // For now, only support full reduction (dim=INT64_MIN or dim=-1)
    // INT64_MIN is the sentinel for "reduce all dimensions"
    if (dim != INT64_MIN && dim != -1) {
        throw std::runtime_error("var: only full reduction is currently supported for CUDA");
    }

    // Compute output shape
    std::vector<int64_t> output_shape;
    if (keepdim) {
        output_shape.resize(input_shape.size(), 1);
    } else {
        output_shape = {};  // Scalar
    }

    Tensor output(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();
            launch_variance_computation(input_data, output_data, input.numel(), correction, stream);
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();
            launch_variance_computation(input_data, output_data, input.numel(), correction, stream);
            break;
        }
        default:
            throw std::runtime_error("var: only Float32 and Float64 are supported");
    }

    cudaStreamSynchronize(stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in var_kernel: ") + cudaGetErrorString(err));
    }

    return output;
}

// Simple sqrt kernel
template<typename T>
__global__ void sqrt_elementwise_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sqrt(input[idx]);
    }
}

// Public API for standard deviation
auto std_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction, cudaStream_t stream) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();

    // For now, only support full reduction (dim=INT64_MIN or dim=-1)
    // INT64_MIN is the sentinel for "reduce all dimensions"
    if (dim != INT64_MIN && dim != -1) {
        throw std::runtime_error("std: only full reduction is currently supported for CUDA");
    }

    // Compute output shape
    std::vector<int64_t> output_shape;
    if (keepdim) {
        output_shape.resize(input_shape.size(), 1);
    } else {
        output_shape = {};  // Scalar
    }

    // First compute variance
    Tensor var_result(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* var_data = var_result.data<float>();
            launch_variance_computation(input_data, var_data, input.numel(), correction, stream);

            // Now compute sqrt of variance
            Tensor output(output_shape, dtype, device);
            auto* output_data = output.data<float>();

            int64_t n = var_result.numel();
            int block_size = 256;
            int grid_size = (n + block_size - 1) / block_size;
            sqrt_elementwise_kernel<<<grid_size, block_size, 0, stream>>>(var_data, output_data, n);

            cudaStreamSynchronize(stream);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA error in std_kernel: ") + cudaGetErrorString(err));
            }

            return output;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* var_data = var_result.data<double>();
            launch_variance_computation(input_data, var_data, input.numel(), correction, stream);

            // Now compute sqrt of variance
            Tensor output(output_shape, dtype, device);
            auto* output_data = output.data<double>();

            int64_t n = var_result.numel();
            int block_size = 256;
            int grid_size = (n + block_size - 1) / block_size;
            sqrt_elementwise_kernel<<<grid_size, block_size, 0, stream>>>(var_data, output_data, n);

            cudaStreamSynchronize(stream);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA error in std_kernel: ") + cudaGetErrorString(err));
            }

            return output;
        }
        default:
            throw std::runtime_error("std: only Float32 and Float64 are supported");
    }
}

// L1 norm kernel - sum of absolute values
template<typename T>
__global__ void l1_norm_kernel(const T* input, T* output, int64_t n) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    T thread_sum = 0;
    for (int64_t i = idx; i < n; i += grid_size) {
        thread_sum += abs(input[i]);
    }

    shared[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid < WARP_SIZE) {
        T val = shared[tid];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// L2 norm kernel - sqrt of sum of squares
template<typename T>
__global__ void l2_norm_squared_kernel(const T* input, T* output, int64_t n) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    T thread_sum = 0;
    for (int64_t i = idx; i < n; i += grid_size) {
        T val = input[i];
        thread_sum += val * val;
    }

    shared[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid < WARP_SIZE) {
        T val = shared[tid];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// Lp norm kernel - (sum(|x|^p))^(1/p)
template<typename T>
__global__ void lp_norm_kernel(const T* input, T* output, int64_t n, float p) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    T thread_sum = 0;
    for (int64_t i = idx; i < n; i += grid_size) {
        thread_sum += pow(abs(input[i]), static_cast<T>(p));
    }

    shared[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid < WARP_SIZE) {
        T val = shared[tid];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// Norm kernel implementation
auto norm_kernel(const Tensor& input, float p, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    // INT64_MIN is the sentinel for "reduce all dimensions"
    if (dim != INT64_MIN && dim != -1) {
        throw std::runtime_error("norm: only full reduction is currently supported for CUDA");
    }

    auto shape = input.shape();
    auto dtype = input.dtype();
    auto device = input.device();

    std::vector<int64_t> output_shape;
    if (keepdim) {
        output_shape.resize(shape.size(), 1);
    } else {
        output_shape = {1};
    }

    Tensor output(output_shape, dtype, device);

    int64_t n = input.numel();
    if (n == 0) {
        throw std::runtime_error("norm: input tensor is empty");
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            float result;

            if (p == 1.0f) {
                // L1 norm
                if (num_blocks == 1) {
                    l1_norm_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, output_data, n);
                } else {
                    float* d_temp;
                    cudaMalloc(&d_temp, num_blocks * sizeof(float));
                    l1_norm_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n);
                    sum_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, num_blocks);
                    cudaFree(d_temp);
                }
            } else if (p == 2.0f) {
                // L2 norm
                float* d_temp;
                cudaMalloc(&d_temp, sizeof(float));

                if (num_blocks == 1) {
                    l2_norm_squared_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n);
                } else {
                    float* d_temp2;
                    cudaMalloc(&d_temp2, num_blocks * sizeof(float));
                    l2_norm_squared_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp2, n);
                    sum_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp2, d_temp, num_blocks);
                    cudaFree(d_temp2);
                }

                sqrt_elementwise_kernel<<<1, 1, 0, stream>>>(d_temp, output_data, 1);
                cudaFree(d_temp);
            } else {
                // General Lp norm
                float* d_temp;
                cudaMalloc(&d_temp, sizeof(float));

                if (num_blocks == 1) {
                    lp_norm_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n, p);
                } else {
                    float* d_temp2;
                    cudaMalloc(&d_temp2, num_blocks * sizeof(float));
                    lp_norm_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp2, n, p);
                    sum_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp2, d_temp, num_blocks);
                    cudaFree(d_temp2);
                }

                // Take p-th root on GPU
                pow_scalar_kernel<<<1, 1, 0, stream>>>(d_temp, 1.0f / p);
                cudaMemcpyAsync(output_data, d_temp, sizeof(float), cudaMemcpyDeviceToDevice, stream);
                cudaFree(d_temp);
            }

            cudaStreamSynchronize(stream);
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            double result;

            if (p == 1.0f) {
                // L1 norm
                if (num_blocks == 1) {
                    l1_norm_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, output_data, n);
                } else {
                    double* d_temp;
                    cudaMalloc(&d_temp, num_blocks * sizeof(double));
                    l1_norm_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n);
                    sum_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, num_blocks);
                    cudaFree(d_temp);
                }
            } else if (p == 2.0f) {
                // L2 norm
                double* d_temp;
                cudaMalloc(&d_temp, sizeof(double));

                if (num_blocks == 1) {
                    l2_norm_squared_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n);
                } else {
                    double* d_temp2;
                    cudaMalloc(&d_temp2, num_blocks * sizeof(double));
                    l2_norm_squared_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp2, n);
                    sum_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp2, d_temp, num_blocks);
                    cudaFree(d_temp2);
                }

                sqrt_elementwise_kernel<<<1, 1, 0, stream>>>(d_temp, output_data, 1);
                cudaFree(d_temp);
            } else {
                // General Lp norm
                double* d_temp;
                cudaMalloc(&d_temp, sizeof(double));

                if (num_blocks == 1) {
                    lp_norm_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n, p);
                } else {
                    double* d_temp2;
                    cudaMalloc(&d_temp2, num_blocks * sizeof(double));
                    lp_norm_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp2, n, p);
                    sum_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp2, d_temp, num_blocks);
                    cudaFree(d_temp2);
                }

                // Take p-th root on GPU
                pow_scalar_kernel<<<1, 1, 0, stream>>>(d_temp, 1.0 / p);
                cudaMemcpyAsync(output_data, d_temp, sizeof(double), cudaMemcpyDeviceToDevice, stream);
                cudaFree(d_temp);
            }

            cudaStreamSynchronize(stream);
            break;
        }
        default:
            throw std::runtime_error("norm: only Float32 and Float64 are supported");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in norm_kernel: ") + cudaGetErrorString(err));
    }

    return output;
}


// ============================================================================
// Argsort kernel
// ============================================================================

template<typename T>
__global__ void argsort_kernel(const T* input, int64_t* output, int64_t n, bool descending) {
    // Bitonic sort for moderate-sized arrays on GPU
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize output indices
    if (tid < n) {
        output[tid] = tid;
    }
    __syncthreads();

    // Bitonic sort passes
    for (int64_t size = 2; size <= n; size *= 2) {
        for (int64_t stride = size / 2; stride > 0; stride /= 2) {
            if (tid < n) {
                int64_t partner = tid ^ stride;
                if (partner > tid && partner < n) {
                    bool swap;
                    T val_tid = input[output[tid]];
                    T val_partner = input[output[partner]];

                    // Determine sort direction for this sub-sequence
                    bool ascending_dir = ((tid & size) == 0);
                    if (descending) ascending_dir = !ascending_dir;

                    if (ascending_dir) {
                        swap = (val_tid > val_partner);
                    } else {
                        swap = (val_tid < val_partner);
                    }

                    if (swap) {
                        int64_t temp = output[tid];
                        output[tid] = output[partner];
                        output[partner] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Index initialization kernel for argsort
__global__ void iota_kernel(int64_t* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = idx;
    }
}

// CUB-based argsort for larger arrays
template<typename T>
static void launch_argsort(const T* d_input, int64_t* d_output, int64_t n, bool descending, cudaStream_t stream = nullptr) {
    if (n == 0) return;

    if (n == 1) {
        int64_t zero = 0;
        cudaMemcpyAsync(d_output, &zero, sizeof(int64_t), cudaMemcpyHostToDevice, stream);
        return;
    }

    // For small arrays, use the bitonic sort kernel
    if (n <= MAX_BLOCK_SIZE) {
        int block_size = 1;
        while (block_size < n) block_size *= 2;
        if (block_size > MAX_BLOCK_SIZE) block_size = MAX_BLOCK_SIZE;
        argsort_kernel<<<1, block_size, 0, stream>>>(d_input, d_output, n, descending);
        return;
    }

    // For larger arrays, use CUB DeviceRadixSort with key-value pairs
    int64_t* d_indices_in;
    cudaMalloc(&d_indices_in, n * sizeof(int64_t));

    // Initialize indices 0..n-1
    int init_blocks = (n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    iota_kernel<<<init_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_indices_in, n);

    // Allocate output keys buffer
    T* d_keys_out;
    cudaMalloc(&d_keys_out, n * sizeof(T));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    if (descending) {
        cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
            d_input, d_keys_out, d_indices_in, d_output, n, 0, sizeof(T) * 8, stream);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
            d_input, d_keys_out, d_indices_in, d_output, n, 0, sizeof(T) * 8, stream);
    } else {
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_input, d_keys_out, d_indices_in, d_output, n, 0, sizeof(T) * 8, stream);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_input, d_keys_out, d_indices_in, d_output, n, 0, sizeof(T) * 8, stream);
    }

    cudaFree(d_temp_storage);
    cudaFree(d_keys_out);
    cudaFree(d_indices_in);
}


auto argsort_kernel(const Tensor& input, int64_t dim, bool descending, cudaStream_t stream) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const int64_t n = input.numel();

    // Currently only supports sorting the flattened tensor (dim=-1 or single dimension)
    // For multi-dim support, would need to sort along a specific axis
    Tensor output(std::vector<int64_t>(input_shape.begin(), input_shape.end()), DType::Int64, device);

    switch (dtype) {
        case DType::Float32:
            launch_argsort(input.data<float>(), output.data<int64_t>(), n, descending, stream);
            break;
        case DType::Float64:
            launch_argsort(input.data<double>(), output.data<int64_t>(), n, descending, stream);
            break;
        case DType::Float16: {
            // CUB RadixSort doesn't support __half, so convert to float32 and sort that
            float* d_float_buf;
            cudaMalloc(&d_float_buf, n * sizeof(float));
            const __half* d_half = reinterpret_cast<const __half*>(input.data_ptr());
            int blocks = (n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
            half_to_float_kernel<<<blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_half, d_float_buf, n);
            launch_argsort(d_float_buf, output.data<int64_t>(), n, descending, stream);
            cudaFree(d_float_buf);
            break;
        }
        default:
            throw std::runtime_error("argsort_kernel: unsupported dtype");
    }

    return output;
}

} // namespace cuda
} // namespace tenzor
