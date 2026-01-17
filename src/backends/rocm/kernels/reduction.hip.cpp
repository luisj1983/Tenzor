/**
 * @file reduction.hip.cpp
 * @brief HIP reduction kernels for AMD GPUs
 *
 * Implements parallel reduction operations (sum, mean, max, min) using HIP.
 * Supports both full tensor reduction and dimension-specific reduction.
 * Optimized for AMD GPU architectures with wavefront-aware patterns.
 */

#include "tenzor/core/tensor.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <climits>
#include <algorithm>
#include <stdexcept>

namespace tenzor {
namespace rocm {

// ============================================================================
// Constants
// ============================================================================

constexpr int WAVEFRONT_SIZE = 32;  // AMD GPU wavefront size (32 for RDNA/wave32, compatible with wave64)
constexpr int MAX_BLOCK_SIZE = 1024;
constexpr int REDUCTION_BLOCK_SIZE = 256;

// ============================================================================
// HIP Error Checking
// ============================================================================

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
    } \
} while(0)

// ============================================================================
// Wavefront-level reduction primitives (AMD GPU specific)
// ============================================================================

/**
 * @brief Elementwise sqrt kernel for std calculation
 */
template<typename T>
__global__ void elementwise_sqrt_kernel(const T* __restrict__ input, T* __restrict__ output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sqrt(input[idx]);
    }
}

/**
 * @brief Wavefront-level sum reduction using AMD GPU warp shuffle
 * @tparam T Data type
 * @param val Value to reduce
 * @return Reduced sum (valid only in lane 0)
 *
 * Uses __shfl_down for efficient intra-wavefront communication
 */
template<typename T>
__device__ __forceinline__ T wavefront_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset, WAVEFRONT_SIZE);
    }
    return val;
}

/**
 * @brief Wavefront-level max reduction
 * @tparam T Data type
 * @param val Value to reduce
 * @return Reduced max (valid only in lane 0)
 */
template<typename T>
__device__ __forceinline__ T wavefront_reduce_max(T val) {
    #pragma unroll
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
        T other = __shfl_down(val, offset, WAVEFRONT_SIZE);
        val = (val > other) ? val : other;
    }
    return val;
}

/**
 * @brief Wavefront-level min reduction
 * @tparam T Data type
 * @param val Value to reduce
 * @return Reduced min (valid only in lane 0)
 */
template<typename T>
__device__ __forceinline__ T wavefront_reduce_min(T val) {
    #pragma unroll
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
        T other = __shfl_down(val, offset, WAVEFRONT_SIZE);
        val = (val < other) ? val : other;
    }
    return val;
}

// ============================================================================
// Block-level reduction kernels (full reduction)
// ============================================================================

/**
 * @brief Sum reduction kernel using LDS (Local Data Share)
 * @tparam T Data type
 * @param input Input tensor data
 * @param output Output buffer (one value per block)
 * @param n Total number of elements
 *
 * Two-level reduction:
 * 1. Grid-stride loop accumulates values per thread
 * 2. Block-level reduction in LDS (AMD's shared memory)
 * 3. Wavefront-level reduction for final sum
 */
template<typename T>
__global__ void sum_reduce_kernel(const T* input, T* output, int64_t n) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Grid-stride loop for better occupancy on AMD GPUs
    T thread_sum = 0;
    for (int64_t i = idx; i < n; i += grid_size) {
        thread_sum += input[i];
    }

    shared[tid] = thread_sum;
    __syncthreads();

    // Block-level reduction in LDS
    for (int stride = blockDim.x / 2; stride >= WAVEFRONT_SIZE; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // Wavefront-level reduction for final stages
    if (tid < WAVEFRONT_SIZE) {
        T val = shared[tid];
        val = wavefront_reduce_sum(val);

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

/**
 * @brief Max reduction kernel using LDS
 * @tparam T Data type
 * @param input Input tensor data
 * @param output Output buffer (one value per block)
 * @param n Total number of elements
 */
template<typename T>
__global__ void max_reduce_kernel(const T* input, T* output, int64_t n) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or negative infinity
    T thread_max;
    if constexpr (std::is_same_v<T, float>) {
        thread_max = (idx < n) ? input[idx] : -FLT_MAX;
    } else if constexpr (std::is_same_v<T, double>) {
        thread_max = (idx < n) ? input[idx] : -DBL_MAX;
    } else {
        thread_max = (idx < n) ? input[idx] : std::numeric_limits<T>::lowest();
    }

    // Grid-stride loop
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        T val = input[i];
        thread_max = (val > thread_max) ? val : thread_max;
    }

    shared[tid] = thread_max;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= WAVEFRONT_SIZE; stride >>= 1) {
        if (tid < stride) {
            T other = shared[tid + stride];
            shared[tid] = (shared[tid] > other) ? shared[tid] : other;
        }
        __syncthreads();
    }

    // Wavefront-level reduction
    if (tid < WAVEFRONT_SIZE) {
        T val = shared[tid];
        val = wavefront_reduce_max(val);

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

/**
 * @brief Min reduction kernel using LDS
 * @tparam T Data type
 * @param input Input tensor data
 * @param output Output buffer (one value per block)
 * @param n Total number of elements
 */
template<typename T>
__global__ void min_reduce_kernel(const T* input, T* output, int64_t n) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or positive infinity
    T thread_min;
    if constexpr (std::is_same_v<T, float>) {
        thread_min = (idx < n) ? input[idx] : FLT_MAX;
    } else if constexpr (std::is_same_v<T, double>) {
        thread_min = (idx < n) ? input[idx] : DBL_MAX;
    } else {
        thread_min = (idx < n) ? input[idx] : std::numeric_limits<T>::max();
    }

    // Grid-stride loop
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        T val = input[i];
        thread_min = (val < thread_min) ? val : thread_min;
    }

    shared[tid] = thread_min;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= WAVEFRONT_SIZE; stride >>= 1) {
        if (tid < stride) {
            T other = shared[tid + stride];
            shared[tid] = (shared[tid] < other) ? shared[tid] : other;
        }
        __syncthreads();
    }

    // Wavefront-level reduction
    if (tid < WAVEFRONT_SIZE) {
        T val = shared[tid];
        val = wavefront_reduce_min(val);

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// ============================================================================
// Dimensional reduction kernels
// ============================================================================

/**
 * @brief Sum reduction along a specific dimension
 * @tparam T Data type
 * @param input Input tensor data
 * @param output Output tensor data
 * @param input_shape Shape of input tensor
 * @param input_strides Strides of input tensor
 * @param ndim Number of dimensions
 * @param dim Dimension to reduce along
 * @param output_size Total size of output tensor
 * @param dim_size Size of the reduction dimension
 *
 * Each thread computes one output element by summing along the reduction dimension
 */
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

    // Convert flat output index to multi-dimensional coordinates
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        int64_t dim_size_d = input_shape[d];
        indices[d] = tmp % dim_size_d;
        tmp /= dim_size_d;
    }

    // Sum along the reduction dimension
    T sum = 0;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;

        // Compute flat index from multi-dimensional indices
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }

        sum += input[in_idx];
    }

    output[out_idx] = sum;
}

/**
 * @brief Max reduction along a specific dimension
 * @tparam T Data type
 */
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

    for (int64_t d = ndim - 1; d >= 0; --d) {
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

/**
 * @brief Min reduction along a specific dimension
 * @tparam T Data type
 */
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

    for (int64_t d = ndim - 1; d >= 0; --d) {
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

/**
 * @brief Argmax reduction along a specific dimension
 * @tparam T Data type
 */
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

/**
 * @brief Argmin reduction along a specific dimension
 * @tparam T Data type
 */
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

/**
 * @brief Product reduction along a specific dimension
 * @tparam T Data type
 */
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
    T prod_val = T(1);
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

// ============================================================================
// Scaling kernel for mean computation
// ============================================================================

/**
 * @brief Element-wise scaling kernel for mean computation
 * @tparam T Data type
 * @param data Input/output data (in-place operation)
 * @param scale Scaling factor (1.0 / count)
 * @param n Number of elements
 */
template<typename T>
__global__ void scale_kernel(T* data, T scale, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

// Specialized scale kernel for Float16 - compute in float for precision
__global__ void scale_kernel_fp16(__half* data, float scale, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(data[idx]);
        data[idx] = __float2half(val * scale);
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/**
 * @brief Compute output shape for reduction operation
 * @param input_shape Input tensor shape
 * @param dim Dimension to reduce (-1 for full reduction)
 * @param keepdim Whether to keep the reduced dimension as size 1
 * @return Output tensor shape
 */
static auto compute_reduction_shape(
    const std::vector<int64_t>& input_shape,
    int64_t dim,
    bool keepdim
) -> std::vector<int64_t> {
    if (dim < 0) {
        // Full reduction
        if (keepdim) {
            return std::vector<int64_t>(input_shape.size(), 1);
        }
        return {};  // Scalar (0D tensor)
    }

    std::vector<int64_t> output_shape = input_shape;
    if (keepdim) {
        output_shape[dim] = 1;
    } else {
        output_shape.erase(output_shape.begin() + dim);
    }
    return output_shape;
}

/**
 * @brief Launch full reduction sum (two-phase for large arrays)
 * @tparam T Data type
 * @param d_input Device input pointer
 * @param d_output Device output pointer
 * @param n Number of elements
 * @param stream HIP stream
 */
template<typename T>
static void launch_full_reduction_sum(const T* d_input, T* d_output, int64_t n, hipStream_t stream) {
    if (n == 0) {
        T zero = 0;
        HIP_CHECK(hipMemcpyAsync(d_output, &zero, sizeof(T), hipMemcpyHostToDevice, stream));
        return;
    }

    if (n == 1) {
        HIP_CHECK(hipMemcpyAsync(d_output, d_input, sizeof(T), hipMemcpyDeviceToDevice, stream));
        return;
    }

    // Two-phase reduction for large arrays
    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        hipLaunchKernelGGL(sum_reduce_kernel<T>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_input, d_output, n);
    } else {
        // Phase 1: Reduce to num_blocks intermediate results
        T* d_temp;
        HIP_CHECK(hipMalloc(&d_temp, num_blocks * sizeof(T)));
        hipLaunchKernelGGL(sum_reduce_kernel<T>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_input, d_temp, n);

        // Phase 2: Final reduction
        hipLaunchKernelGGL(sum_reduce_kernel<T>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_temp, d_output, num_blocks);

        // Synchronize before freeing temp buffer
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipFree(d_temp));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
}

/**
 * @brief Launch full reduction max
 * @tparam T Data type
 */
template<typename T>
static void launch_full_reduction_max(const T* d_input, T* d_output, int64_t n, hipStream_t stream) {
    if (n == 0) {
        throw std::runtime_error("max: input tensor is empty");
    }

    if (n == 1) {
        HIP_CHECK(hipMemcpyAsync(d_output, d_input, sizeof(T), hipMemcpyDeviceToDevice, stream));
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        hipLaunchKernelGGL(max_reduce_kernel<T>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_input, d_output, n);
    } else {
        T* d_temp;
        HIP_CHECK(hipMalloc(&d_temp, num_blocks * sizeof(T)));
        hipLaunchKernelGGL(max_reduce_kernel<T>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_input, d_temp, n);
        hipLaunchKernelGGL(max_reduce_kernel<T>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_temp, d_output, num_blocks);

        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipFree(d_temp));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
}

/**
 * @brief Launch full reduction min
 * @tparam T Data type
 */
template<typename T>
static void launch_full_reduction_min(const T* d_input, T* d_output, int64_t n, hipStream_t stream) {
    if (n == 0) {
        throw std::runtime_error("min: input tensor is empty");
    }

    if (n == 1) {
        HIP_CHECK(hipMemcpyAsync(d_output, d_input, sizeof(T), hipMemcpyDeviceToDevice, stream));
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        hipLaunchKernelGGL(min_reduce_kernel<T>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_input, d_output, n);
    } else {
        T* d_temp;
        HIP_CHECK(hipMalloc(&d_temp, num_blocks * sizeof(T)));
        hipLaunchKernelGGL(min_reduce_kernel<T>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_input, d_temp, n);
        hipLaunchKernelGGL(min_reduce_kernel<T>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_temp, d_output, num_blocks);

        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipFree(d_temp));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
}

/**
 * @brief Launch dimensional reduction sum
 * @tparam T Data type
 */
template<typename T>
static void launch_dim_reduction_sum(
    const T* d_input,
    T* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    hipStream_t stream
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
    HIP_CHECK(hipMalloc(&d_shape, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_strides, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(d_shape, input_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_strides, input_strides.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));

    // Launch kernel
    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    hipLaunchKernelGGL(sum_along_dim_kernel<T>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size);

    // Must wait for kernel to complete before freeing device memory it uses
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_shape));
    HIP_CHECK(hipFree(d_strides));
}

/**
 * @brief Launch dimensional reduction max
 * @tparam T Data type
 */
template<typename T>
static void launch_dim_reduction_max(
    const T* d_input,
    T* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    hipStream_t stream
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
    HIP_CHECK(hipMalloc(&d_shape, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_strides, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(d_shape, input_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_strides, input_strides.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    hipLaunchKernelGGL(max_along_dim_kernel<T>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size);

    // Must wait for kernel to complete before freeing device memory it uses
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_shape));
    HIP_CHECK(hipFree(d_strides));
}

/**
 * @brief Launch dimensional reduction min
 * @tparam T Data type
 */
template<typename T>
static void launch_dim_reduction_min(
    const T* d_input,
    T* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    hipStream_t stream
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
    HIP_CHECK(hipMalloc(&d_shape, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_strides, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(d_shape, input_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_strides, input_strides.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    hipLaunchKernelGGL(min_along_dim_kernel<T>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size);

    // Must wait for kernel to complete before freeing device memory it uses
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_shape));
    HIP_CHECK(hipFree(d_strides));
}

/**
 * @brief Launch dimensional argmax
 * @tparam T Data type
 */
template<typename T>
static void launch_dim_argmax(
    const T* d_input,
    int64_t* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    hipStream_t stream
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
    HIP_CHECK(hipMalloc(&d_shape, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_strides, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(d_shape, input_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_strides, input_strides.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    hipLaunchKernelGGL(argmax_along_dim_kernel<T>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size);

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_shape));
    HIP_CHECK(hipFree(d_strides));
}

/**
 * @brief Launch dimensional argmin
 * @tparam T Data type
 */
template<typename T>
static void launch_dim_argmin(
    const T* d_input,
    int64_t* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    hipStream_t stream
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
    HIP_CHECK(hipMalloc(&d_shape, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_strides, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(d_shape, input_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_strides, input_strides.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    hipLaunchKernelGGL(argmin_along_dim_kernel<T>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size);

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_shape));
    HIP_CHECK(hipFree(d_strides));
}

/**
 * @brief Launch dimensional product
 * @tparam T Data type
 */
template<typename T>
static void launch_dim_prod(
    const T* d_input,
    T* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    hipStream_t stream
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
    HIP_CHECK(hipMalloc(&d_shape, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_strides, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(d_shape, input_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_strides, input_strides.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    hipLaunchKernelGGL(prod_along_dim_kernel<T>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size);

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_shape));
    HIP_CHECK(hipFree(d_strides));
}

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Sum reduction kernel
 * @param input Input tensor
 * @param dim Dimension to reduce along (-1 for full reduction)
 * @param keepdim Whether to keep the reduced dimension
 * @param stream HIP stream
 * @return Output tensor with sum
 */
auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize negative dimension (but preserve special value INT64_MIN for full reduction)
    bool full_reduction = (dim == INT64_MIN);
    if (dim < 0 && dim != INT64_MIN) {
        dim += ndim;
    }

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        full_reduction ? -1 : dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            if (full_reduction) {
                launch_full_reduction_sum(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_sum(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (full_reduction) {
                launch_full_reduction_sum(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_sum(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int32_t>();

            if (full_reduction) {
                launch_full_reduction_sum(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_sum(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (full_reduction) {
                launch_full_reduction_sum(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_sum(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
            }
            break;
        }
        case DType::Float16: {
            auto* input_data = reinterpret_cast<const __half*>(input.data<Float16>());
            auto* output_data = reinterpret_cast<__half*>(output.data<Float16>());

            if (full_reduction) {
                launch_full_reduction_sum(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_sum(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
            }
            break;
        }
        default:
            throw std::runtime_error("sum: unsupported dtype");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in sum_kernel: ") + hipGetErrorString(err));
    }

    return output;
}

/**
 * @brief Mean reduction kernel
 * @param input Input tensor
 * @param dim Dimension to reduce along (-1 for full reduction)
 * @param keepdim Whether to keep the reduced dimension
 * @param stream HIP stream
 * @return Output tensor with mean
 */
auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor {
    const auto dtype = input.dtype();
    const int64_t ndim = input.ndim();

    if (dtype != DType::Float32 && dtype != DType::Float64 && dtype != DType::Float16) {
        throw std::runtime_error("mean: only Float32, Float64, and Float16 are supported");
    }

    // Normalize negative dimension (but preserve special value INT64_MIN for full reduction)
    bool full_reduction = (dim == INT64_MIN);
    if (dim < 0 && dim != INT64_MIN) {
        dim += ndim;
    }

    // Compute sum first
    auto sum_result = sum_kernel(input, full_reduction ? INT64_MIN : dim, keepdim, stream);

    // Compute count
    int64_t count;
    if (full_reduction) {
        count = input.numel();
    } else {
        count = input.shape()[dim];
    }

    if (count == 0) {
        throw std::runtime_error("mean: cannot compute mean of empty tensor");
    }

    // Divide by count using scaling kernel
    const int64_t n = sum_result.numel();

    if (dtype == DType::Float32) {
        auto* data = sum_result.data<float>();
        const float scale = 1.0f / static_cast<float>(count);

        int num_blocks = (n + 255) / 256;
        hipLaunchKernelGGL(scale_kernel<float>, dim3(num_blocks), dim3(256), 0, stream,
            data, scale, n);
    } else if (dtype == DType::Float64) {
        auto* data = sum_result.data<double>();
        const double scale = 1.0 / static_cast<double>(count);

        int num_blocks = (n + 255) / 256;
        hipLaunchKernelGGL(scale_kernel<double>, dim3(num_blocks), dim3(256), 0, stream,
            data, scale, n);
    } else {  // Float16
        auto* data = reinterpret_cast<__half*>(sum_result.data<Float16>());
        const float scale = 1.0f / static_cast<float>(count);

        int num_blocks = (n + 255) / 256;
        hipLaunchKernelGGL(scale_kernel_fp16, dim3(num_blocks), dim3(256), 0, stream,
            data, scale, n);
    }

    HIP_CHECK(hipGetLastError());
    return sum_result;
}

/**
 * @brief Max reduction kernel
 * @param input Input tensor
 * @param dim Dimension to reduce along (-1 for full reduction)
 * @param keepdim Whether to keep the reduced dimension
 * @param stream HIP stream
 * @return Output tensor with max values
 */
auto max_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize negative dimension (but preserve special value INT64_MIN for full reduction)
    bool full_reduction = (dim == INT64_MIN);
    if (dim < 0 && dim != INT64_MIN) {
        dim += ndim;
    }

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        full_reduction ? -1 : dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            if (full_reduction) {
                launch_full_reduction_max(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_max(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (full_reduction) {
                launch_full_reduction_max(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_max(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int32_t>();

            if (full_reduction) {
                launch_full_reduction_max(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_max(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (full_reduction) {
                launch_full_reduction_max(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_max(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
            }
            break;
        }
        default:
            throw std::runtime_error("max: unsupported dtype");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in max_kernel: ") + hipGetErrorString(err));
    }

    return output;
}

/**
 * @brief Min reduction kernel
 * @param input Input tensor
 * @param dim Dimension to reduce along (-1 for full reduction)
 * @param keepdim Whether to keep the reduced dimension
 * @param stream HIP stream
 * @return Output tensor with min values
 */
auto min_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor {
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
                    dim, stream
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
                    dim, stream
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
                    dim, stream
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
                    dim, stream
                );
            }
            break;
        }
        default:
            throw std::runtime_error("min: unsupported dtype");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in min_kernel: ") + hipGetErrorString(err));
    }

    return output;
}

// ============================================================================
// ArgMax/ArgMin Kernels
// ============================================================================

template<typename T>
__global__ void argmax_kernel(const T* input, int64_t* output, int64_t n) {
    __shared__ T sdata[REDUCTION_BLOCK_SIZE];
    __shared__ int64_t sidx[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    T max_val = (idx < n) ? input[idx] : std::numeric_limits<T>::lowest();
    int64_t max_idx = (idx < n) ? idx : 0;

    // Grid-stride loop to handle large inputs
    for (idx += blockDim.x * gridDim.x; idx < n; idx += blockDim.x * gridDim.x) {
        if (input[idx] > max_val) {
            max_val = input[idx];
            max_idx = idx;
        }
    }

    sdata[tid] = max_val;
    sidx[tid] = max_idx;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sidx[0];
        // Store max value in a temporary location for multi-block reduction
    }
}

template<typename T>
__global__ void argmin_kernel(const T* input, int64_t* output, int64_t n) {
    __shared__ T sdata[REDUCTION_BLOCK_SIZE];
    __shared__ int64_t sidx[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    T min_val = (idx < n) ? input[idx] : std::numeric_limits<T>::max();
    int64_t min_idx = (idx < n) ? idx : 0;

    for (idx += blockDim.x * gridDim.x; idx < n; idx += blockDim.x * gridDim.x) {
        if (input[idx] < min_val) {
            min_val = input[idx];
            min_idx = idx;
        }
    }

    sdata[tid] = min_val;
    sidx[tid] = min_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) {
                sdata[tid] = sdata[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sidx[0];
    }
}

template<typename T>
__global__ void prod_kernel(const T* input, T* output, int64_t n) {
    __shared__ T sdata[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    T prod = T(1);
    while (idx < n) {
        prod *= input[idx];
        idx += blockDim.x * gridDim.x;
    }

    sdata[tid] = prod;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] *= sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

template<typename T>
__global__ void var_kernel(const T* input, T mean, T* output, int64_t n) {
    __shared__ T sdata[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = T(0);
    while (idx < n) {
        T diff = input[idx] - mean;
        sum += diff * diff;
        idx += blockDim.x * gridDim.x;
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

template<typename T>
__global__ void norm_kernel(const T* input, T* output, int64_t n, T p) {
    __shared__ T sdata[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = T(0);
    while (idx < n) {
        T val = input[idx];
        if (val < T(0)) val = -val;  // abs
        sum += pow(val, p);
        idx += blockDim.x * gridDim.x;
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief ArgMax reduction kernel
 */
auto argmax_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    int64_t n = input.numel();

    if (n == 0) {
        throw std::runtime_error("argmax: cannot compute argmax of empty tensor");
    }

    // For full reduction (dim < 0), output is a scalar index
    std::vector<int64_t> output_shape;
    if (dim >= 0) {
        output_shape = compute_reduction_shape(
            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
            dim, keepdim);
    }

    Tensor output(output_shape, DType::Int64, device);

    if (dim < 0) {
        // Full tensor argmax
        int num_blocks = std::min((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, static_cast<int64_t>(1024));

        if (dtype == DType::Float32) {
            int64_t* d_partial;
            float* d_vals;
            HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(int64_t)));
            HIP_CHECK(hipMalloc(&d_vals, num_blocks * sizeof(float)));

            hipLaunchKernelGGL(argmax_kernel<float>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<float>(), d_partial, n);

            // Single block for final reduction if needed
            if (num_blocks > 1) {
                // For simplicity, use the first result
                std::vector<int64_t> h_partial(num_blocks);
                HIP_CHECK(hipMemcpy(h_partial.data(), d_partial, num_blocks * sizeof(int64_t), hipMemcpyDeviceToHost));

                // Find the actual max among partial results
                float max_val = std::numeric_limits<float>::lowest();
                int64_t max_idx = 0;
                for (int i = 0; i < num_blocks; ++i) {
                    float val;
                    HIP_CHECK(hipMemcpy(&val, input.data<float>() + h_partial[i], sizeof(float), hipMemcpyDeviceToHost));
                    if (val > max_val) {
                        max_val = val;
                        max_idx = h_partial[i];
                    }
                }
                HIP_CHECK(hipMemcpy(output.data<int64_t>(), &max_idx, sizeof(int64_t), hipMemcpyHostToDevice));
            } else {
                HIP_CHECK(hipMemcpy(output.data<int64_t>(), d_partial, sizeof(int64_t), hipMemcpyDeviceToDevice));
            }

            HIP_CHECK(hipFree(d_partial));
            HIP_CHECK(hipFree(d_vals));
        } else if (dtype == DType::Float64) {
            int64_t* d_partial;
            HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(int64_t)));

            hipLaunchKernelGGL(argmax_kernel<double>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<double>(), d_partial, n);

            if (num_blocks > 1) {
                std::vector<int64_t> h_partial(num_blocks);
                HIP_CHECK(hipMemcpy(h_partial.data(), d_partial, num_blocks * sizeof(int64_t), hipMemcpyDeviceToHost));

                double max_val = std::numeric_limits<double>::lowest();
                int64_t max_idx = 0;
                for (int i = 0; i < num_blocks; ++i) {
                    double val;
                    HIP_CHECK(hipMemcpy(&val, input.data<double>() + h_partial[i], sizeof(double), hipMemcpyDeviceToHost));
                    if (val > max_val) {
                        max_val = val;
                        max_idx = h_partial[i];
                    }
                }
                HIP_CHECK(hipMemcpy(output.data<int64_t>(), &max_idx, sizeof(int64_t), hipMemcpyHostToDevice));
            } else {
                HIP_CHECK(hipMemcpy(output.data<int64_t>(), d_partial, sizeof(int64_t), hipMemcpyDeviceToDevice));
            }

            HIP_CHECK(hipFree(d_partial));
        } else {
            throw std::runtime_error("argmax: unsupported dtype");
        }
    } else {
        // Dimensional argmax
        const auto& input_strides = input.strides();

        switch (dtype) {
            case DType::Float32:
                launch_dim_argmax(
                    input.data<float>(), output.data<int64_t>(),
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
                break;
            case DType::Float64:
                launch_dim_argmax(
                    input.data<double>(), output.data<int64_t>(),
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
                break;
            case DType::Int32:
                launch_dim_argmax(
                    input.data<int32_t>(), output.data<int64_t>(),
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
                break;
            case DType::Int64:
                launch_dim_argmax(
                    input.data<int64_t>(), output.data<int64_t>(),
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
                break;
            default:
                throw std::runtime_error("argmax: unsupported dtype for dimensional reduction");
        }
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

/**
 * @brief ArgMin reduction kernel
 */
auto argmin_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    int64_t n = input.numel();

    if (n == 0) {
        throw std::runtime_error("argmin: cannot compute argmin of empty tensor");
    }

    std::vector<int64_t> output_shape;
    if (dim >= 0) {
        output_shape = compute_reduction_shape(
            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
            dim, keepdim);
    }

    Tensor output(output_shape, DType::Int64, device);

    if (dim < 0) {
        int num_blocks = std::min((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, static_cast<int64_t>(1024));

        if (dtype == DType::Float32) {
            int64_t* d_partial;
            HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(int64_t)));

            hipLaunchKernelGGL(argmin_kernel<float>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<float>(), d_partial, n);

            if (num_blocks > 1) {
                std::vector<int64_t> h_partial(num_blocks);
                HIP_CHECK(hipMemcpy(h_partial.data(), d_partial, num_blocks * sizeof(int64_t), hipMemcpyDeviceToHost));

                float min_val = std::numeric_limits<float>::max();
                int64_t min_idx = 0;
                for (int i = 0; i < num_blocks; ++i) {
                    float val;
                    HIP_CHECK(hipMemcpy(&val, input.data<float>() + h_partial[i], sizeof(float), hipMemcpyDeviceToHost));
                    if (val < min_val) {
                        min_val = val;
                        min_idx = h_partial[i];
                    }
                }
                HIP_CHECK(hipMemcpy(output.data<int64_t>(), &min_idx, sizeof(int64_t), hipMemcpyHostToDevice));
            } else {
                HIP_CHECK(hipMemcpy(output.data<int64_t>(), d_partial, sizeof(int64_t), hipMemcpyDeviceToDevice));
            }

            HIP_CHECK(hipFree(d_partial));
        } else if (dtype == DType::Float64) {
            int64_t* d_partial;
            HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(int64_t)));

            hipLaunchKernelGGL(argmin_kernel<double>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<double>(), d_partial, n);

            if (num_blocks > 1) {
                std::vector<int64_t> h_partial(num_blocks);
                HIP_CHECK(hipMemcpy(h_partial.data(), d_partial, num_blocks * sizeof(int64_t), hipMemcpyDeviceToHost));

                double min_val = std::numeric_limits<double>::max();
                int64_t min_idx = 0;
                for (int i = 0; i < num_blocks; ++i) {
                    double val;
                    HIP_CHECK(hipMemcpy(&val, input.data<double>() + h_partial[i], sizeof(double), hipMemcpyDeviceToHost));
                    if (val < min_val) {
                        min_val = val;
                        min_idx = h_partial[i];
                    }
                }
                HIP_CHECK(hipMemcpy(output.data<int64_t>(), &min_idx, sizeof(int64_t), hipMemcpyHostToDevice));
            } else {
                HIP_CHECK(hipMemcpy(output.data<int64_t>(), d_partial, sizeof(int64_t), hipMemcpyDeviceToDevice));
            }

            HIP_CHECK(hipFree(d_partial));
        } else {
            throw std::runtime_error("argmin: unsupported dtype");
        }
    } else {
        // Dimensional argmin
        const auto& input_strides = input.strides();

        switch (dtype) {
            case DType::Float32:
                launch_dim_argmin(
                    input.data<float>(), output.data<int64_t>(),
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
                break;
            case DType::Float64:
                launch_dim_argmin(
                    input.data<double>(), output.data<int64_t>(),
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
                break;
            case DType::Int32:
                launch_dim_argmin(
                    input.data<int32_t>(), output.data<int64_t>(),
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
                break;
            case DType::Int64:
                launch_dim_argmin(
                    input.data<int64_t>(), output.data<int64_t>(),
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
                break;
            default:
                throw std::runtime_error("argmin: unsupported dtype for dimensional reduction");
        }
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

/**
 * @brief Product reduction kernel
 */
auto prod_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    int64_t n = input.numel();

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim);

    Tensor output(output_shape, dtype, device);

    if (dim < 0) {
        // Full product
        int num_blocks = std::min((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, static_cast<int64_t>(1024));

        if (dtype == DType::Float32) {
            float* d_partial;
            HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(float)));

            hipLaunchKernelGGL(prod_kernel<float>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<float>(), d_partial, n);

            if (num_blocks > 1) {
                std::vector<float> h_partial(num_blocks);
                HIP_CHECK(hipMemcpy(h_partial.data(), d_partial, num_blocks * sizeof(float), hipMemcpyDeviceToHost));

                float prod = 1.0f;
                for (int i = 0; i < num_blocks; ++i) {
                    prod *= h_partial[i];
                }
                HIP_CHECK(hipMemcpy(output.data<float>(), &prod, sizeof(float), hipMemcpyHostToDevice));
            } else {
                HIP_CHECK(hipMemcpy(output.data<float>(), d_partial, sizeof(float), hipMemcpyDeviceToDevice));
            }

            HIP_CHECK(hipFree(d_partial));
        } else if (dtype == DType::Float64) {
            double* d_partial;
            HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(double)));

            hipLaunchKernelGGL(prod_kernel<double>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<double>(), d_partial, n);

            if (num_blocks > 1) {
                std::vector<double> h_partial(num_blocks);
                HIP_CHECK(hipMemcpy(h_partial.data(), d_partial, num_blocks * sizeof(double), hipMemcpyDeviceToHost));

                double prod = 1.0;
                for (int i = 0; i < num_blocks; ++i) {
                    prod *= h_partial[i];
                }
                HIP_CHECK(hipMemcpy(output.data<double>(), &prod, sizeof(double), hipMemcpyHostToDevice));
            } else {
                HIP_CHECK(hipMemcpy(output.data<double>(), d_partial, sizeof(double), hipMemcpyDeviceToDevice));
            }

            HIP_CHECK(hipFree(d_partial));
        } else if (dtype == DType::Int32) {
            int32_t* d_partial;
            HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(int32_t)));

            hipLaunchKernelGGL(prod_kernel<int32_t>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<int32_t>(), d_partial, n);

            if (num_blocks > 1) {
                std::vector<int32_t> h_partial(num_blocks);
                HIP_CHECK(hipMemcpy(h_partial.data(), d_partial, num_blocks * sizeof(int32_t), hipMemcpyDeviceToHost));

                int32_t prod = 1;
                for (int i = 0; i < num_blocks; ++i) {
                    prod *= h_partial[i];
                }
                HIP_CHECK(hipMemcpy(output.data<int32_t>(), &prod, sizeof(int32_t), hipMemcpyHostToDevice));
            } else {
                HIP_CHECK(hipMemcpy(output.data<int32_t>(), d_partial, sizeof(int32_t), hipMemcpyDeviceToDevice));
            }

            HIP_CHECK(hipFree(d_partial));
        } else if (dtype == DType::Int64) {
            int64_t* d_partial;
            HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(int64_t)));

            hipLaunchKernelGGL(prod_kernel<int64_t>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<int64_t>(), d_partial, n);

            if (num_blocks > 1) {
                std::vector<int64_t> h_partial(num_blocks);
                HIP_CHECK(hipMemcpy(h_partial.data(), d_partial, num_blocks * sizeof(int64_t), hipMemcpyDeviceToHost));

                int64_t prod = 1;
                for (int i = 0; i < num_blocks; ++i) {
                    prod *= h_partial[i];
                }
                HIP_CHECK(hipMemcpy(output.data<int64_t>(), &prod, sizeof(int64_t), hipMemcpyHostToDevice));
            } else {
                HIP_CHECK(hipMemcpy(output.data<int64_t>(), d_partial, sizeof(int64_t), hipMemcpyDeviceToDevice));
            }

            HIP_CHECK(hipFree(d_partial));
        } else {
            throw std::runtime_error("prod: unsupported dtype");
        }
    } else {
        // Dimensional product
        const auto& input_strides = input.strides();

        switch (dtype) {
            case DType::Float32:
                launch_dim_prod(
                    input.data<float>(), output.data<float>(),
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
                break;
            case DType::Float64:
                launch_dim_prod(
                    input.data<double>(), output.data<double>(),
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
                break;
            case DType::Int32:
                launch_dim_prod(
                    input.data<int32_t>(), output.data<int32_t>(),
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
                break;
            case DType::Int64:
                launch_dim_prod(
                    input.data<int64_t>(), output.data<int64_t>(),
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream
                );
                break;
            default:
                throw std::runtime_error("prod: unsupported dtype for dimensional reduction");
        }
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

/**
 * @brief Variance reduction kernel
 */
auto var_kernel(const Tensor& input, int64_t dim, bool keepdim, bool unbiased, hipStream_t stream) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    int64_t n = input.numel();

    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("var: only Float32 and Float64 are supported");
    }

    // First compute mean
    auto mean_tensor = mean_kernel(input, dim, true, stream);

    // Then compute variance
    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input.shape().begin(), input.shape().end()),
        dim, keepdim);

    Tensor output(output_shape, dtype, device);

    if (dim < 0) {
        int num_blocks = std::min((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, static_cast<int64_t>(1024));

        if (dtype == DType::Float32) {
            float mean;
            HIP_CHECK(hipMemcpy(&mean, mean_tensor.data<float>(), sizeof(float), hipMemcpyDeviceToHost));

            float* d_partial;
            HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(float)));

            hipLaunchKernelGGL(var_kernel<float>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<float>(), mean, d_partial, n);

            std::vector<float> h_partial(num_blocks);
            HIP_CHECK(hipMemcpy(h_partial.data(), d_partial, num_blocks * sizeof(float), hipMemcpyDeviceToHost));

            float sum = 0.0f;
            for (int i = 0; i < num_blocks; ++i) {
                sum += h_partial[i];
            }

            float var = sum / (unbiased ? (n - 1) : n);
            HIP_CHECK(hipMemcpy(output.data<float>(), &var, sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipFree(d_partial));
        } else {
            double mean;
            HIP_CHECK(hipMemcpy(&mean, mean_tensor.data<double>(), sizeof(double), hipMemcpyDeviceToHost));

            double* d_partial;
            HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(double)));

            hipLaunchKernelGGL(var_kernel<double>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<double>(), mean, d_partial, n);

            std::vector<double> h_partial(num_blocks);
            HIP_CHECK(hipMemcpy(h_partial.data(), d_partial, num_blocks * sizeof(double), hipMemcpyDeviceToHost));

            double sum = 0.0;
            for (int i = 0; i < num_blocks; ++i) {
                sum += h_partial[i];
            }

            double var = sum / (unbiased ? (n - 1) : n);
            HIP_CHECK(hipMemcpy(output.data<double>(), &var, sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipFree(d_partial));
        }
    } else {
        throw std::runtime_error("var along specific dimension not yet implemented");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

/**
 * @brief Standard deviation reduction kernel
 */
auto std_kernel(const Tensor& input, int64_t dim, bool keepdim, bool unbiased, hipStream_t stream) -> Tensor {
    auto var_result = var_kernel(input, dim, keepdim, unbiased, stream);

    // Compute sqrt of variance
    int64_t n = var_result.numel();
    Tensor output(std::vector<int64_t>(var_result.shape().begin(), var_result.shape().end()),
                  var_result.dtype(), var_result.device());

    if (n == 0) return output;

    dim3 grid, block;
    block = dim3(256, 1, 1);
    grid = dim3((n + 255) / 256, 1, 1);

    if (var_result.dtype() == DType::Float32) {
        hipLaunchKernelGGL(elementwise_sqrt_kernel<float>, grid, block, 0, stream,
                           var_result.data<float>(), output.data<float>(), n);
    } else {
        hipLaunchKernelGGL(elementwise_sqrt_kernel<double>, grid, block, 0, stream,
                           var_result.data<double>(), output.data<double>(), n);
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

/**
 * @brief Norm reduction kernel (p-norm)
 */
auto norm_kernel(const Tensor& input, float p, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    int64_t n = input.numel();

    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("norm: only Float32 and Float64 are supported");
    }

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input.shape().begin(), input.shape().end()),
        dim, keepdim);

    Tensor output(output_shape, dtype, device);

    if (dim < 0) {
        int num_blocks = std::min((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, static_cast<int64_t>(1024));

        if (dtype == DType::Float32) {
            float* d_partial;
            HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(float)));

            hipLaunchKernelGGL(norm_kernel<float>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<float>(), d_partial, n, p);

            std::vector<float> h_partial(num_blocks);
            HIP_CHECK(hipMemcpy(h_partial.data(), d_partial, num_blocks * sizeof(float), hipMemcpyDeviceToHost));

            float sum = 0.0f;
            for (int i = 0; i < num_blocks; ++i) {
                sum += h_partial[i];
            }

            float norm = powf(sum, 1.0f / p);
            HIP_CHECK(hipMemcpy(output.data<float>(), &norm, sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipFree(d_partial));
        } else {
            double* d_partial;
            HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(double)));

            hipLaunchKernelGGL(norm_kernel<double>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<double>(), d_partial, n, static_cast<double>(p));

            std::vector<double> h_partial(num_blocks);
            HIP_CHECK(hipMemcpy(h_partial.data(), d_partial, num_blocks * sizeof(double), hipMemcpyDeviceToHost));

            double sum = 0.0;
            for (int i = 0; i < num_blocks; ++i) {
                sum += h_partial[i];
            }

            double norm = pow(sum, 1.0 / p);
            HIP_CHECK(hipMemcpy(output.data<double>(), &norm, sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipFree(d_partial));
        }
    } else {
        throw std::runtime_error("norm along specific dimension not yet implemented");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

} // namespace rocm
} // namespace tenzor
