#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/caching_allocator.hpp"
#include "tenzor/backend/dtype_dispatch.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cub/cub.cuh>
#include "../cuda_error.hpp"
#include "cuda_launch_utils.cuh"

namespace tenzor {
namespace cuda {

// Constants
constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_SIZE = 1024;
// Reduction block size is used for static __shared__ memory in kernels.
// 256 is a good default for occupancy on most architectures (Volta+).
// For kernels without shared memory constraints, use optimal_launch_config().
constexpr int REDUCTION_BLOCK_SIZE = 256;

// Type-safe sentinel values for max/min reductions.
// Using -FLT_MAX for double produces wrong results since FLT_MAX << DBL_MAX.
template<typename T> __device__ __host__ inline T sentinel_lowest();
template<typename T> __device__ __host__ inline T sentinel_max();

template<> __device__ __host__ inline float sentinel_lowest<float>() { return -FLT_MAX; }
template<> __device__ __host__ inline float sentinel_max<float>() { return FLT_MAX; }
template<> __device__ __host__ inline double sentinel_lowest<double>() { return -DBL_MAX; }
template<> __device__ __host__ inline double sentinel_max<double>() { return DBL_MAX; }
template<> __device__ __host__ inline int32_t sentinel_lowest<int32_t>() { return INT32_MIN; }
template<> __device__ __host__ inline int32_t sentinel_max<int32_t>() { return INT32_MAX; }
template<> __device__ __host__ inline int64_t sentinel_lowest<int64_t>() { return INT64_MIN; }
template<> __device__ __host__ inline int64_t sentinel_max<int64_t>() { return INT64_MAX; }

// Metadata struct passed by value to kernels (avoids cudaMalloc for shape/stride arrays)
struct DimMeta {
    int64_t shape[8];
    int64_t strides[8];
};

static DimMeta make_dim_meta(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides) {
    DimMeta meta{};
    for (size_t i = 0; i < shape.size() && i < 8; ++i) {
        meta.shape[i] = shape[i];
        meta.strides[i] = strides[i];
    }
    return meta;
}

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

// BFloat16 type helpers
template<>
__device__ __forceinline__ __nv_bfloat16 cuda_zero<__nv_bfloat16>() { return __float2bfloat16(0.0f); }

template<>
__device__ __forceinline__ __nv_bfloat16 cuda_add<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) { return __hadd(a, b); }

// Get negative infinity for __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 bfloat16_neg_inf() {
    return __ushort_as_bfloat16(0xFF80);  // -inf in bfloat16
}

// Get positive infinity for __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 bfloat16_pos_inf() {
    return __ushort_as_bfloat16(0x7F80);  // +inf in bfloat16
}

// BFloat16 comparison helpers
__device__ __forceinline__ __nv_bfloat16 cuda_max_val(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hgt(a, b) ? a : b;
}

__device__ __forceinline__ __nv_bfloat16 cuda_min_val(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hlt(a, b) ? a : b;
}

// GPU-side scalar fill (replaces host_zero + cudaMemcpyAsync H2D pattern to avoid UB)
template<typename T>
__global__ void fill_scalar_kernel(T* dst, T value) { dst[0] = value; }

// ============================================================================
// Helper kernels for GPU-side scalar operations (avoids D2H transfers)
// ============================================================================

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

// Specialization for __nv_bfloat16
template<>
__device__ __forceinline__ __nv_bfloat16 warp_reduce_sum(__nv_bfloat16 val) {
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

// Specialization for __nv_bfloat16
template<>
__device__ __forceinline__ __nv_bfloat16 warp_reduce_max(__nv_bfloat16 val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        __nv_bfloat16 other = __shfl_down_sync(0xffffffff, val, offset);
        val = cuda_max_val(val, other);
    }
    return val;
}

// Specialization for __nv_bfloat16
template<>
__device__ __forceinline__ __nv_bfloat16 warp_reduce_min(__nv_bfloat16 val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        __nv_bfloat16 other = __shfl_down_sync(0xffffffff, val, offset);
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

    // Initialize with first element or type-appropriate negative infinity
    T thread_max = (idx < n) ? input[idx] : sentinel_lowest<T>();

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

    // Initialize with first element or type-appropriate positive infinity
    T thread_min = (idx < n) ? input[idx] : sentinel_max<T>();

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
// BFloat16 specializations for max/min reduction kernels
// ============================================================================

// Specialized max_reduce_kernel for __nv_bfloat16
__global__ void max_reduce_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    __shared__ __nv_bfloat16 shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or negative infinity
    __nv_bfloat16 thread_max = (idx < n) ? input[idx] : bfloat16_neg_inf();

    // Grid-stride loop
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        __nv_bfloat16 val = input[i];
        thread_max = cuda_max_val(val, thread_max);
    }

    shared[tid] = thread_max;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            __nv_bfloat16 other = shared[tid + stride];
            shared[tid] = cuda_max_val(shared[tid], other);
        }
        __syncthreads();
    }

    // Warp-level reduction
    if (tid < WARP_SIZE) {
        __nv_bfloat16 val = shared[tid];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            if (tid + offset < blockDim.x) {
                __nv_bfloat16 other = shared[tid + offset];
                val = cuda_max_val(val, other);
            }
        }
        val = warp_reduce_max(val);

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// Specialized min_reduce_kernel for __nv_bfloat16
__global__ void min_reduce_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    __shared__ __nv_bfloat16 shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or positive infinity
    __nv_bfloat16 thread_min = (idx < n) ? input[idx] : bfloat16_pos_inf();

    // Grid-stride loop
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        __nv_bfloat16 val = input[i];
        thread_min = cuda_min_val(val, thread_min);
    }

    shared[tid] = thread_min;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            __nv_bfloat16 other = shared[tid + stride];
            shared[tid] = cuda_min_val(shared[tid], other);
        }
        __syncthreads();
    }

    // Warp-level reduction
    if (tid < WARP_SIZE) {
        __nv_bfloat16 val = shared[tid];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            if (tid + offset < blockDim.x) {
                __nv_bfloat16 other = shared[tid + offset];
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
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= output_size) return;

    // Compute multi-dimensional indices for output position.
    // Decompose in reverse order (innermost dim first) to match row-major layout.
    int64_t indices[8];  // Support up to 8D tensors
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Sum along the reduction dimension
    T sum = cuda_zero<T>();
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;

        // Compute flat index
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
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
    DimMeta meta,
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
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find max along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    T max_val = input[in_idx];

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
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
    DimMeta meta,
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
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find min along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    T min_val = input[in_idx];

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
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
    DimMeta meta,
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
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find max along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    __half max_val = input[in_idx];

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
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
    DimMeta meta,
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
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find min along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    __half min_val = input[in_idx];

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        __half val = input[in_idx];
        min_val = cuda_min_val(val, min_val);
    }

    output[out_idx] = min_val;
}

// Specialized max_along_dim_kernel for __nv_bfloat16
__global__ void max_along_dim_kernel_bf16(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    DimMeta meta,
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
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find max along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    __nv_bfloat16 max_val = input[in_idx];

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        __nv_bfloat16 val = input[in_idx];
        max_val = cuda_max_val(val, max_val);
    }

    output[out_idx] = max_val;
}

// Specialized min_along_dim_kernel for __nv_bfloat16
__global__ void min_along_dim_kernel_bf16(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    DimMeta meta,
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
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find min along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    __nv_bfloat16 min_val = input[in_idx];

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        __nv_bfloat16 val = input[in_idx];
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
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, T(0));
        CUDA_CHECK(cudaGetLastError());
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
        CUDA_CHECK(cudaGetLastError());
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA kernel launch failed in sum_reduce_kernel: ") + cudaGetErrorString(err));
        }
    } else {
        // Phase 1: Reduce to num_blocks intermediate results
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(T));
        auto* d_temp = static_cast<T*>(d_temp_guard.get());
        sum_reduce_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());
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
        CUDA_CHECK(cudaGetLastError());
    } else {
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(T));
        auto* d_temp = static_cast<T*>(d_temp_guard.get());
        max_reduce_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());
        max_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
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
        CUDA_CHECK(cudaGetLastError());
    } else {
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(T));
        auto* d_temp = static_cast<T*>(d_temp_guard.get());
        min_reduce_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());
        min_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
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

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    // Launch kernel
    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    sum_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
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

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    max_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
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

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    min_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
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
        CUDA_CHECK(cudaGetLastError());
    } else {
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(__half));
        auto* d_temp = static_cast<__half*>(d_temp_guard.get());
        max_reduce_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());
        max_reduce_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
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
        CUDA_CHECK(cudaGetLastError());
    } else {
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(__half));
        auto* d_temp = static_cast<__half*>(d_temp_guard.get());
        min_reduce_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());
        min_reduce_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
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

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    max_along_dim_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
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

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    min_along_dim_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// BFloat16 specialized launch functions
// ============================================================================

static void launch_full_reduction_max_bf16(const __nv_bfloat16* d_input, __nv_bfloat16* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("max: input tensor is empty");
    }

    if (n == 1) {
        cudaMemcpyAsync(d_output, d_input, sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        max_reduce_kernel_bf16<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(__nv_bfloat16));
        auto* d_temp = static_cast<__nv_bfloat16*>(d_temp_guard.get());
        max_reduce_kernel_bf16<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());
        max_reduce_kernel_bf16<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void launch_full_reduction_min_bf16(const __nv_bfloat16* d_input, __nv_bfloat16* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("min: input tensor is empty");
    }

    if (n == 1) {
        cudaMemcpyAsync(d_output, d_input, sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        min_reduce_kernel_bf16<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(__nv_bfloat16));
        auto* d_temp = static_cast<__nv_bfloat16*>(d_temp_guard.get());
        min_reduce_kernel_bf16<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());
        min_reduce_kernel_bf16<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void launch_dim_reduction_max_bf16(
    const __nv_bfloat16* d_input,
    __nv_bfloat16* d_output,
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

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    max_along_dim_kernel_bf16<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

static void launch_dim_reduction_min_bf16(
    const __nv_bfloat16* d_input,
    __nv_bfloat16* d_output,
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

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    min_along_dim_kernel_bf16<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
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
        case DType::BFloat16: {
            auto* input_data = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
            auto* output_data = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

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

#ifndef NDEBUG
    cudaStreamSynchronize(stream);
#endif
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

// Specialization for __nv_bfloat16 (no *= operator)
template<>
__global__ void scale_kernel<__nv_bfloat16>(__nv_bfloat16* data, int64_t n, __nv_bfloat16 scale) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = __float2bfloat16(__bfloat162float(data[idx]) * __bfloat162float(scale));
    }
}

auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    const auto dtype = input.dtype();

    if (dtype != DType::Float32 && dtype != DType::Float64 && dtype != DType::Float16 && dtype != DType::BFloat16) {
        throw std::runtime_error("mean: only Float32, Float64, Float16, and BFloat16 are supported");
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
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float64) {
        auto* data = sum_result.data<double>();
        const double scale = 1.0 / static_cast<double>(count);
        scale_kernel<<<grid_size, block_size, 0, stream>>>(data, n, scale);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float16) {
        auto* data = reinterpret_cast<__half*>(sum_result.data_ptr());
        const __half scale = __float2half(1.0f / static_cast<float>(count));
        scale_kernel<<<grid_size, block_size, 0, stream>>>(data, n, scale);
        CUDA_CHECK(cudaGetLastError());
    } else {  // BFloat16
        auto* data = reinterpret_cast<__nv_bfloat16*>(sum_result.data_ptr());
        const __nv_bfloat16 scale = __float2bfloat16(1.0f / static_cast<float>(count));
        scale_kernel<<<grid_size, block_size, 0, stream>>>(data, n, scale);
        CUDA_CHECK(cudaGetLastError());
    }

#ifndef NDEBUG
    cudaStreamSynchronize(stream);
#endif
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
        case DType::BFloat16: {
            auto* input_data = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
            auto* output_data = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

            if (dim < 0) {
                launch_full_reduction_max_bf16(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_max_bf16(
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

#ifndef NDEBUG
    cudaStreamSynchronize(stream);
#endif
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
        case DType::BFloat16: {
            auto* input_data = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
            auto* output_data = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

            if (dim < 0) {
                launch_full_reduction_min_bf16(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_min_bf16(
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

#ifndef NDEBUG
    cudaStreamSynchronize(stream);
#endif
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

    // Initialize with first element or type-appropriate negative infinity
    T thread_max = (idx < n) ? input[idx] : sentinel_lowest<T>();
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

    // Initialize with first element or type-appropriate positive infinity
    T thread_min = (idx < n) ? input[idx] : sentinel_max<T>();
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
    DimMeta meta,
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
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find argmax along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    T max_val = input[in_idx];
    int64_t max_idx = 0;

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
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
    DimMeta meta,
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
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find argmin along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    T min_val = input[in_idx];
    int64_t min_idx = 0;

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
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
    DimMeta meta,
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
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find argmax along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    __half max_val = input[in_idx];
    int64_t max_idx = 0;

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
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
    DimMeta meta,
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
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find argmin along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    __half min_val = input[in_idx];
    int64_t min_idx = 0;

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        __half val = input[in_idx];
        if (__hlt(val, min_val)) {
            min_val = val;
            min_idx = i;
        }
    }

    output[out_idx] = min_idx;
}

// ============================================================================
// BFloat16 specializations for argmax/argmin kernels
// ============================================================================

// BFloat16 final kernel for argmax
__global__ void argmax_final_kernel_bf16(const __nv_bfloat16* input, const int64_t* block_indices, int64_t* output, int num_blocks) {
    __shared__ float shared_vals[REDUCTION_BLOCK_SIZE];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    float best_val;
    int64_t best_idx;

    if (tid < num_blocks) {
        best_idx = block_indices[tid];
        best_val = __bfloat162float(input[best_idx]);
    } else {
        best_val = -FLT_MAX;
        best_idx = 0;
    }

    for (int i = tid + blockDim.x; i < num_blocks; i += blockDim.x) {
        int64_t idx = block_indices[i];
        float val = __bfloat162float(input[idx]);
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

// BFloat16 final kernel for argmin
__global__ void argmin_final_kernel_bf16(const __nv_bfloat16* input, const int64_t* block_indices, int64_t* output, int num_blocks) {
    __shared__ float shared_vals[REDUCTION_BLOCK_SIZE];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    float best_val;
    int64_t best_idx;

    if (tid < num_blocks) {
        best_idx = block_indices[tid];
        best_val = __bfloat162float(input[best_idx]);
    } else {
        best_val = FLT_MAX;
        best_idx = 0;
    }

    for (int i = tid + blockDim.x; i < num_blocks; i += blockDim.x) {
        int64_t idx = block_indices[i];
        float val = __bfloat162float(input[idx]);
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

// Specialized argmax_full_kernel for __nv_bfloat16
__global__ void argmax_full_kernel_bf16(const __nv_bfloat16* input, int64_t* output, int64_t n) {
    __shared__ __nv_bfloat16 shared_vals[REDUCTION_BLOCK_SIZE];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or negative infinity
    __nv_bfloat16 thread_max = (idx < n) ? input[idx] : bfloat16_neg_inf();
    int64_t thread_idx = (idx < n) ? idx : 0;

    // Grid-stride loop to find local maximum
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        __nv_bfloat16 val = input[i];
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
        __nv_bfloat16 val = shared_vals[tid];
        int64_t val_idx = shared_idxs[tid];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            __nv_bfloat16 other_val = __shfl_down_sync(0xffffffff, val, offset);
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

// Specialized argmin_full_kernel for __nv_bfloat16
__global__ void argmin_full_kernel_bf16(const __nv_bfloat16* input, int64_t* output, int64_t n) {
    __shared__ __nv_bfloat16 shared_vals[REDUCTION_BLOCK_SIZE];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or positive infinity
    __nv_bfloat16 thread_min = (idx < n) ? input[idx] : bfloat16_pos_inf();
    int64_t thread_idx = (idx < n) ? idx : 0;

    // Grid-stride loop to find local minimum
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        __nv_bfloat16 val = input[i];
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
        __nv_bfloat16 val = shared_vals[tid];
        int64_t val_idx = shared_idxs[tid];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            __nv_bfloat16 other_val = __shfl_down_sync(0xffffffff, val, offset);
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

// Specialized argmax_along_dim_kernel for __nv_bfloat16
__global__ void argmax_along_dim_kernel_bf16(
    const __nv_bfloat16* input,
    int64_t* output,
    DimMeta meta,
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
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find argmax along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    __nv_bfloat16 max_val = input[in_idx];
    int64_t max_idx = 0;

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        __nv_bfloat16 val = input[in_idx];
        if (__hgt(val, max_val)) {
            max_val = val;
            max_idx = i;
        }
    }

    output[out_idx] = max_idx;
}

// Specialized argmin_along_dim_kernel for __nv_bfloat16
__global__ void argmin_along_dim_kernel_bf16(
    const __nv_bfloat16* input,
    int64_t* output,
    DimMeta meta,
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
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find argmin along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    __nv_bfloat16 min_val = input[in_idx];
    int64_t min_idx = 0;

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        __nv_bfloat16 val = input[in_idx];
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
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, int64_t(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        argmax_full_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Two-phase GPU-only reduction
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(int64_t));
        auto* d_temp = static_cast<int64_t*>(d_temp_guard.get());
        argmax_full_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());

        // Phase 2: compare block results entirely on GPU
        argmax_final_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}
// Helper function to launch argmin reduction
template<typename T>
static void launch_full_argmin(const T* d_input, int64_t* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("argmin: input tensor is empty");
    }

    if (n == 1) {
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, int64_t(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        argmin_full_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Two-phase GPU-only reduction
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(int64_t));
        auto* d_temp = static_cast<int64_t*>(d_temp_guard.get());
        argmin_full_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());

        // Phase 2: compare block results entirely on GPU
        argmin_final_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
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

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    argmax_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
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

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    argmin_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Half-precision specialized launch functions for argmax/argmin
// ============================================================================

static void launch_full_argmax_half(const __half* d_input, int64_t* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("argmax: input tensor is empty");
    }

    if (n == 1) {
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, int64_t(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        argmax_full_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Two-phase GPU-only reduction
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(int64_t));
        auto* d_temp = static_cast<int64_t*>(d_temp_guard.get());
        argmax_full_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());

        // Phase 2: compare block results entirely on GPU
        argmax_final_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void launch_full_argmin_half(const __half* d_input, int64_t* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("argmin: input tensor is empty");
    }

    if (n == 1) {
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, int64_t(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        argmin_full_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Two-phase GPU-only reduction
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(int64_t));
        auto* d_temp = static_cast<int64_t*>(d_temp_guard.get());
        argmin_full_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());

        // Phase 2: compare block results entirely on GPU
        argmin_final_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
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

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    argmax_along_dim_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
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

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    argmin_along_dim_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// BFloat16 specialized launch functions for argmax/argmin
// ============================================================================

static void launch_full_argmax_bf16(const __nv_bfloat16* d_input, int64_t* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("argmax: input tensor is empty");
    }

    if (n == 1) {
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, int64_t(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        argmax_full_kernel_bf16<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Two-phase GPU-only reduction
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(int64_t));
        auto* d_temp = static_cast<int64_t*>(d_temp_guard.get());
        argmax_full_kernel_bf16<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());

        // Phase 2: compare block results entirely on GPU
        argmax_final_kernel_bf16<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void launch_full_argmin_bf16(const __nv_bfloat16* d_input, int64_t* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("argmin: input tensor is empty");
    }

    if (n == 1) {
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, int64_t(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        argmin_full_kernel_bf16<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Two-phase GPU-only reduction
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(int64_t));
        auto* d_temp = static_cast<int64_t*>(d_temp_guard.get());
        argmin_full_kernel_bf16<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());

        // Phase 2: compare block results entirely on GPU
        argmin_final_kernel_bf16<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void launch_dim_argmax_bf16(
    const __nv_bfloat16* d_input,
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

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    argmax_along_dim_kernel_bf16<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

static void launch_dim_argmin_bf16(
    const __nv_bfloat16* d_input,
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

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    argmin_along_dim_kernel_bf16<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
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
        case DType::BFloat16: {
            auto* input_data = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmax_bf16(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmax_bf16(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        default:
            throw std::runtime_error("argmax: only Float32, Float64, Float16, BFloat16, Int32, and Int64 are supported");
    }

#ifndef NDEBUG
    cudaStreamSynchronize(stream);
#endif
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
        case DType::BFloat16: {
            auto* input_data = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmin_bf16(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmin_bf16(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim
                );
            }
            break;
        }
        default:
            throw std::runtime_error("argmin: only Float32, Float64, Float16, BFloat16, Int32, and Int64 are supported");
    }

#ifndef NDEBUG
    cudaStreamSynchronize(stream);
#endif
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
    DimMeta meta,
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
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Product along the reduction dimension
    T prod_val = 1;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;

        // Compute flat index
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }

        prod_val *= input[in_idx];
    }

    output[out_idx] = prod_val;
}

// Helper function to launch product reduction
template<typename T>
static void launch_full_reduction_prod(const T* d_input, T* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, T(1));
        CUDA_CHECK(cudaGetLastError());
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
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Phase 1: Reduce to num_blocks intermediate results
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(T));
        auto* d_temp = static_cast<T*>(d_temp_guard.get());
        prod_reduce_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());

        // Phase 2: Final reduction
        prod_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
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

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    prod_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
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

#ifndef NDEBUG
    cudaStreamSynchronize(stream);
#endif
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
// Uses parallel shared-memory tree reduction with 256 threads
template<typename T>
__global__ void welford_finalize_kernel(
    const T* block_means,
    const T* block_m2s,
    const int64_t* block_counts,
    T* output,
    int num_blocks,
    int64_t correction
) {
    // Shared memory for tree reduction: mean, m2, count per thread
    extern __shared__ char smem_raw[];
    T* s_mean = reinterpret_cast<T*>(smem_raw);
    T* s_m2 = s_mean + blockDim.x;
    int64_t* s_count = reinterpret_cast<int64_t*>(s_m2 + blockDim.x);

    int tid = threadIdx.x;

    // Each thread loads one or more block states via grid-stride
    T t_mean = T(0);
    T t_m2 = T(0);
    int64_t t_count = 0;

    if (tid < num_blocks) {
        t_mean = block_means[tid];
        t_m2 = block_m2s[tid];
        t_count = block_counts[tid];
    }
    // Grid-stride for blocks > blockDim.x
    for (int i = tid + blockDim.x; i < num_blocks; i += blockDim.x) {
        welford_combine(t_mean, t_m2, t_count,
                        block_means[i], block_m2s[i], block_counts[i]);
    }

    s_mean[tid] = t_mean;
    s_m2[tid] = t_m2;
    s_count[tid] = t_count;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (tid + stride) < num_blocks) {
            welford_combine(s_mean[tid], s_m2[tid], s_count[tid],
                            s_mean[tid + stride], s_m2[tid + stride], s_count[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        int64_t divisor = s_count[0] - correction;
        if (divisor <= 0) divisor = 1;
        output[0] = s_m2[0] / static_cast<T>(divisor);
    }
}

// Helper function to compute variance — single-pass Welford, zero D2H transfers
template<typename T>
static void launch_variance_computation(const T* d_input, T* d_output, int64_t n, int64_t correction, cudaStream_t stream = nullptr) {
    if (n == 0) {
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, T(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (n == 1) {
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, T(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    // Allocate per-block Welford accumulators
    backend::CachedMemoryGuard d_block_means_guard(num_blocks * sizeof(T));
    backend::CachedMemoryGuard d_block_m2s_guard(num_blocks * sizeof(T));
    backend::CachedMemoryGuard d_block_counts_guard(num_blocks * sizeof(int64_t));
    auto* d_block_means = static_cast<T*>(d_block_means_guard.get());
    auto* d_block_m2s = static_cast<T*>(d_block_m2s_guard.get());
    auto* d_block_counts = static_cast<int64_t*>(d_block_counts_guard.get());

    // Phase 1: Per-block Welford reduction
    welford_variance_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_block_means, d_block_m2s, d_block_counts, n);
    CUDA_CHECK(cudaGetLastError());

    // Phase 2: Merge block results — parallel tree reduction with 256 threads
    constexpr int FINALIZE_THREADS = 256;
    size_t welford_smem = FINALIZE_THREADS * (2 * sizeof(T) + sizeof(int64_t));
    welford_finalize_kernel<<<1, FINALIZE_THREADS, welford_smem, stream>>>(
        d_block_means, d_block_m2s, d_block_counts, d_output, num_blocks, correction);
    CUDA_CHECK(cudaGetLastError());
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

#ifndef NDEBUG
    cudaStreamSynchronize(stream);
#endif
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in var_kernel: ") + cudaGetErrorString(err));
    }

    return output;
}

// Fused sum-reduce with sqrt finalization for L2 norm
// Reduces input[0..n-1] and writes sqrt(sum) to output[0]
template<typename T>
__global__ void sum_reduce_sqrt_kernel(const T* input, T* output, int64_t n) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];
    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    T thread_sum = cuda_zero<T>();
    for (int64_t i = idx; i < n; i += grid_size) {
        thread_sum = cuda_add(thread_sum, input[i]);
    }
    shared[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) shared[tid] = cuda_add(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    if (tid < WARP_SIZE) {
        T val = shared[tid];
        val = warp_reduce_sum(val);
        if (tid == 0) output[blockIdx.x] = sqrt(val);
    }
}

// Fused sum-reduce with pow finalization for Lp norm
// Reduces input[0..n-1] and writes pow(sum, exponent) to output[0]
template<typename T>
__global__ void sum_reduce_pow_kernel(const T* input, T* output, int64_t n, T exponent) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];
    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    T thread_sum = cuda_zero<T>();
    for (int64_t i = idx; i < n; i += grid_size) {
        thread_sum = cuda_add(thread_sum, input[i]);
    }
    shared[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) shared[tid] = cuda_add(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    if (tid < WARP_SIZE) {
        T val = shared[tid];
        val = warp_reduce_sum(val);
        if (tid == 0) output[blockIdx.x] = pow(val, exponent);
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

    // First compute variance, then sqrt in-place for std deviation
    Tensor var_result(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* var_data = var_result.data<float>();
            launch_variance_computation(input_data, var_data, input.numel(), correction, stream);

            // Compute sqrt of variance — fused into single-element reduce+sqrt
            Tensor output(output_shape, dtype, device);
            auto* output_data = output.data<float>();
            sum_reduce_sqrt_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(var_data, output_data, 1);
            CUDA_CHECK(cudaGetLastError());

#ifndef NDEBUG
            cudaStreamSynchronize(stream);
#endif
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

            // Compute sqrt of variance — fused into single-element reduce+sqrt
            Tensor output(output_shape, dtype, device);
            auto* output_data = output.data<double>();
            sum_reduce_sqrt_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(var_data, output_data, 1);
            CUDA_CHECK(cudaGetLastError());

#ifndef NDEBUG
            cudaStreamSynchronize(stream);
#endif
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

// Per-dimension norm reduction kernel (follows sum_along_dim_kernel pattern)
template<typename T>
__global__ void norm_along_dim_kernel(
    const T* input,
    T* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size,
    float p
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= output_size) return;

    // Compute multi-dimensional indices for output position
    int64_t indices[8];
    int64_t tmp = out_idx;

    for (int64_t d = 0; d < ndim; d++) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Accumulate along the reduction dimension
    T acc = cuda_zero<T>();

    if (p == 1.0f) {
        // L1 norm: sum of |x|
        for (int64_t i = 0; i < dim_size; i++) {
            indices[dim] = i;
            int64_t in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * meta.strides[d];
            }
            T val = input[in_idx];
            acc = acc + (val < cuda_zero<T>() ? -val : val);
        }
        output[out_idx] = acc;
    } else if (p == 2.0f) {
        // L2 norm: sqrt(sum of x^2)
        for (int64_t i = 0; i < dim_size; i++) {
            indices[dim] = i;
            int64_t in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * meta.strides[d];
            }
            T val = input[in_idx];
            acc = acc + val * val;
        }
        output[out_idx] = sqrt(acc);
    } else {
        // General Lp norm: (sum of |x|^p)^(1/p)
        for (int64_t i = 0; i < dim_size; i++) {
            indices[dim] = i;
            int64_t in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * meta.strides[d];
            }
            T val = input[in_idx];
            acc = acc + pow(val < cuda_zero<T>() ? -val : val, T(p));
        }
        output[out_idx] = pow(acc, T(1.0f / p));
    }
}

// Norm kernel implementation
auto norm_kernel(const Tensor& input, float p, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    // INT64_MIN is the sentinel for "reduce all dimensions"
    if (dim != INT64_MIN && dim != -1) {
        // Per-dimension norm reduction
        auto shape_span = input.shape();
        auto strides_span = input.strides();
        int64_t ndim = static_cast<int64_t>(shape_span.size());

        // Normalize dim
        int64_t actual_dim = dim;
        if (actual_dim < 0) actual_dim += ndim;
        if (actual_dim < 0 || actual_dim >= ndim) {
            throw std::runtime_error("norm: dim out of range");
        }

        int64_t dim_size = shape_span[actual_dim];

        // Compute output shape
        std::vector<int64_t> output_shape;
        for (int64_t d = 0; d < ndim; d++) {
            if (d == actual_dim) {
                if (keepdim) output_shape.push_back(1);
            } else {
                output_shape.push_back(shape_span[d]);
            }
        }
        if (output_shape.empty()) output_shape.push_back(1);

        int64_t output_size = 1;
        for (auto s : output_shape) output_size *= s;

        Tensor output(output_shape, input.dtype(), input.device());

        // Build DimMeta from input shape/strides
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        std::vector<int64_t> strides_vec(strides_span.begin(), strides_span.end());
        DimMeta meta = make_dim_meta(shape_vec, strides_vec);

        constexpr int BLOCK = 256;
        int blocks = (output_size + BLOCK - 1) / BLOCK;

        TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "norm", [&]() {
            norm_along_dim_kernel<scalar_t><<<blocks, BLOCK, 0, stream>>>(
                input.data<scalar_t>(), output.data<scalar_t>(), meta,
                ndim, actual_dim, output_size, dim_size, p);
            CUDA_CHECK(cudaGetLastError());
        });

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error in norm_along_dim_kernel: ") + cudaGetErrorString(err));
        }

        return output;
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

            if (p == 1.0f) {
                // L1 norm
                if (num_blocks == 1) {
                    l1_norm_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, output_data, n);
                    CUDA_CHECK(cudaGetLastError());
                } else {
                    backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(float));
                    auto* d_temp = static_cast<float*>(d_temp_guard.get());
                    l1_norm_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, num_blocks);
                    CUDA_CHECK(cudaGetLastError());
                }
            } else if (p == 2.0f) {
                // L2 norm — fused sqrt into final reduction
                if (num_blocks == 1) {
                    // Single block: reduce and sqrt in one kernel
                    backend::CachedMemoryGuard d_temp_guard(sizeof(float));
                    auto* d_temp = static_cast<float*>(d_temp_guard.get());
                    l2_norm_squared_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_sqrt_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, 1);
                    CUDA_CHECK(cudaGetLastError());
                } else {
                    backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(float));
                    auto* d_temp = static_cast<float*>(d_temp_guard.get());
                    l2_norm_squared_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_sqrt_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, num_blocks);
                    CUDA_CHECK(cudaGetLastError());
                }
            } else {
                // General Lp norm — fused pow into final reduction
                if (num_blocks == 1) {
                    backend::CachedMemoryGuard d_temp_guard(sizeof(float));
                    auto* d_temp = static_cast<float*>(d_temp_guard.get());
                    lp_norm_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n, p);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_pow_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, 1, 1.0f / p);
                    CUDA_CHECK(cudaGetLastError());
                } else {
                    backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(float));
                    auto* d_temp = static_cast<float*>(d_temp_guard.get());
                    lp_norm_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n, p);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_pow_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, num_blocks, 1.0f / p);
                    CUDA_CHECK(cudaGetLastError());
                }
            }

#ifndef NDEBUG
            cudaStreamSynchronize(stream);
#endif
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (p == 1.0f) {
                // L1 norm
                if (num_blocks == 1) {
                    l1_norm_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, output_data, n);
                    CUDA_CHECK(cudaGetLastError());
                } else {
                    backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(double));
                    auto* d_temp = static_cast<double*>(d_temp_guard.get());
                    l1_norm_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, num_blocks);
                    CUDA_CHECK(cudaGetLastError());
                }
            } else if (p == 2.0f) {
                // L2 norm — fused sqrt into final reduction
                if (num_blocks == 1) {
                    backend::CachedMemoryGuard d_temp_guard(sizeof(double));
                    auto* d_temp = static_cast<double*>(d_temp_guard.get());
                    l2_norm_squared_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_sqrt_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, 1);
                    CUDA_CHECK(cudaGetLastError());
                } else {
                    backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(double));
                    auto* d_temp = static_cast<double*>(d_temp_guard.get());
                    l2_norm_squared_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_sqrt_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, num_blocks);
                    CUDA_CHECK(cudaGetLastError());
                }
            } else {
                // General Lp norm — fused pow into final reduction
                if (num_blocks == 1) {
                    backend::CachedMemoryGuard d_temp_guard(sizeof(double));
                    auto* d_temp = static_cast<double*>(d_temp_guard.get());
                    lp_norm_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n, p);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_pow_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, 1, static_cast<double>(1.0 / p));
                    CUDA_CHECK(cudaGetLastError());
                } else {
                    backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(double));
                    auto* d_temp = static_cast<double*>(d_temp_guard.get());
                    lp_norm_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n, p);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_pow_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, num_blocks, static_cast<double>(1.0 / p));
                    CUDA_CHECK(cudaGetLastError());
                }
            }

#ifndef NDEBUG
            cudaStreamSynchronize(stream);
#endif
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
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, int64_t(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // For small arrays, use the bitonic sort kernel
    if (n <= MAX_BLOCK_SIZE) {
        int block_size = 1;
        while (block_size < n) block_size *= 2;
        if (block_size > MAX_BLOCK_SIZE) block_size = MAX_BLOCK_SIZE;
        argsort_kernel<<<1, block_size, 0, stream>>>(d_input, d_output, n, descending);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // For larger arrays, use CUB DeviceRadixSort with key-value pairs
    backend::CachedMemoryGuard d_indices_in_guard(n * sizeof(int64_t));
    auto* d_indices_in = static_cast<int64_t*>(d_indices_in_guard.get());

    // Initialize indices 0..n-1
    int init_blocks = (n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    iota_kernel<<<init_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_indices_in, n);
    CUDA_CHECK(cudaGetLastError());

    // Allocate output keys buffer
    backend::CachedMemoryGuard d_keys_out_guard(n * sizeof(T));
    auto* d_keys_out = static_cast<T*>(d_keys_out_guard.get());

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    if (descending) {
        cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
            d_input, d_keys_out, d_indices_in, d_output, n, 0, sizeof(T) * 8, stream);
        backend::CachedMemoryGuard d_temp_storage_guard(temp_storage_bytes);
        d_temp_storage = d_temp_storage_guard.get();
        cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
            d_input, d_keys_out, d_indices_in, d_output, n, 0, sizeof(T) * 8, stream);
    } else {
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_input, d_keys_out, d_indices_in, d_output, n, 0, sizeof(T) * 8, stream);
        backend::CachedMemoryGuard d_temp_storage_guard(temp_storage_bytes);
        d_temp_storage = d_temp_storage_guard.get();
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_input, d_keys_out, d_indices_in, d_output, n, 0, sizeof(T) * 8, stream);
    }
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
            backend::CachedMemoryGuard d_float_buf_guard(n * sizeof(float));
            auto* d_float_buf = static_cast<float*>(d_float_buf_guard.get());
            const __half* d_half = reinterpret_cast<const __half*>(input.data_ptr());
            int blocks = (n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
            half_to_float_kernel<<<blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_half, d_float_buf, n);
            CUDA_CHECK(cudaGetLastError());
            launch_argsort(d_float_buf, output.data<int64_t>(), n, descending, stream);
            break;
        }
        default:
            throw std::runtime_error("argsort_kernel: unsupported dtype");
    }

    return output;
}

} // namespace cuda
} // namespace tenzor
