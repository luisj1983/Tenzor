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
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace tenzor {
namespace rocm {

// ============================================================================
// Constants
// ============================================================================

constexpr int WAVEFRONT_SIZE = 64;  // AMD GPU wavefront size
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

            if (dim < 0) {
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

            if (dim < 0) {
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

            if (dim < 0) {
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

    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("mean: only Float32 and Float64 are supported");
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
    } else {  // Float64
        auto* data = sum_result.data<double>();
        const double scale = 1.0 / static_cast<double>(count);

        int num_blocks = (n + 255) / 256;
        hipLaunchKernelGGL(scale_kernel<double>, dim3(num_blocks), dim3(256), 0, stream,
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
                    dim, stream
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
                    dim, stream
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
                    dim, stream
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

} // namespace rocm
} // namespace tenzor
