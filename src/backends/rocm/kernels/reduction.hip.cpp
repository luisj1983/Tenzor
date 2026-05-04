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
#include <limits>
#include <cstdint>
#include <climits>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include "fp16_saturate.h"
#include "reduction_utils.hip.h"
#include "../rocm_arch_detect.hpp"

namespace tenzor {
// Forward declarations for tensor entry points used by the Complex sum path.
// Pulling in math.hpp / creation.hpp directly would make tenzor::sqrt etc.
// visible to device code and shadow the unqualified ::sqrt used below.
auto real(const Tensor& input) -> Tensor;
auto imag(const Tensor& input) -> Tensor;
auto complex(const Tensor& real_t, const Tensor& imag_t) -> Tensor;
namespace rocm {

// ============================================================================
// Constants
// ============================================================================

// NOTE: Do NOT hardcode wavefront size — RDNA 3/4 GPUs use wave32, CDNA uses wave64.
// Use the HIP built-in `warpSize` in device code instead.
constexpr int MAX_BLOCK_SIZE = 1024;
// REDUCTION_BLOCK_SIZE defined in reduction_utils.hip.h

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

// wavefront_reduce_sum<T> is provided by reduction_utils.hip.h

/**
 * @brief Wavefront-level max reduction
 * @tparam T Data type
 * @param val Value to reduce
 * @return Reduced max (valid only in lane 0)
 */
template<typename T>
__device__ __forceinline__ T wavefront_reduce_max(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        T other = __shfl_down(val, offset);
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
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        T other = __shfl_down(val, offset);
        val = (val < other) ? val : other;
    }
    return val;
}

// ============================================================================
// Block-level reduction kernels (full reduction)
// ============================================================================

// sum_reduce_kernel<T> is provided by reduction_utils.hip.h

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
    for (int stride = blockDim.x / 2; stride >= warpSize; stride >>= 1) {
        if (tid < stride) {
            T other = shared[tid + stride];
            shared[tid] = (shared[tid] > other) ? shared[tid] : other;
        }
        __syncthreads();
    }

    // Wavefront-level reduction
    if (tid < warpSize) {
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
    for (int stride = blockDim.x / 2; stride >= warpSize; stride >>= 1) {
        if (tid < stride) {
            T other = shared[tid + stride];
            shared[tid] = (shared[tid] < other) ? shared[tid] : other;
        }
        __syncthreads();
    }

    // Wavefront-level reduction
    if (tid < warpSize) {
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
 * @brief Variance reduction along a specific dimension using Welford's algorithm
 * @tparam T Data type
 */
template<typename T>
__global__ void var_along_dim_kernel(
    const T* input,
    T* output,
    const int64_t* input_shape,
    const int64_t* input_strides,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size,
    int64_t correction
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= output_size) return;

    // Compute multi-dimensional indices for output position
    int64_t indices[8];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) { indices[d] = 0; continue; }
        indices[d] = tmp % input_shape[d];
        tmp /= input_shape[d];
    }

    // Welford single-pass algorithm for numerical stability
    T mean = T(0), m2 = T(0);
    int64_t count = 0;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) in_idx += indices[d] * input_strides[d];
        T val = input[in_idx];
        count++;
        T delta = val - mean;
        mean = mean + delta / T(count);
        T delta2 = val - mean;
        m2 = m2 + delta * delta2;
    }

    int64_t denom = count - correction;
    output[out_idx] = denom > 0 ? m2 / T(denom) : T(0);
}

/**
 * @brief Norm reduction along a specific dimension (p-norm)
 * @tparam T Data type
 */
template<typename T>
__global__ void norm_along_dim_kernel(
    const T* input,
    T* output,
    const int64_t* input_shape,
    const int64_t* input_strides,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size,
    T p
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= output_size) return;

    // Compute multi-dimensional indices for output position
    int64_t indices[8];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) { indices[d] = 0; continue; }
        indices[d] = tmp % input_shape[d];
        tmp /= input_shape[d];
    }

    T acc = T(0);
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) in_idx += indices[d] * input_strides[d];
        T val = input[in_idx];
        T abs_val = val < T(0) ? -val : val;
        if (p == T(1)) acc += abs_val;
        else if (p == T(2)) acc += abs_val * abs_val;
        else acc += pow(abs_val, p);
    }

    if (p == T(1)) output[out_idx] = acc;
    else if (p == T(2)) output[out_idx] = sqrt(acc);
    else output[out_idx] = pow(acc, T(1) / p);
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
// launch_full_reduction_sum<T> is provided by reduction_utils.hip.h

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
 * @brief Launch dimensional reduction variance (Welford)
 * @tparam T Data type
 */
template<typename T>
static void launch_dim_reduction_var(
    const T* d_input,
    T* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    int64_t correction,
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
    hipLaunchKernelGGL(var_along_dim_kernel<T>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size, correction);

    // Must wait for kernel to complete before freeing device memory it uses
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_shape));
    HIP_CHECK(hipFree(d_strides));
}

/**
 * @brief Launch dimensional reduction norm (p-norm)
 * @tparam T Data type
 */
template<typename T>
static void launch_dim_reduction_norm(
    const T* d_input,
    T* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    T p,
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
    hipLaunchKernelGGL(norm_along_dim_kernel<T>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size, p);

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
auto sum_kernel(const Tensor& input_raw, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor {
    // audit-2026-05-03 bug #2: ensure contiguous input (mirror of CPU/CUDA fix).
    auto input = input_raw.contiguous();
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
        case DType::BFloat16: {
            auto input_f32 = input.to(DType::Float32);
            auto result_f32 = sum_kernel(input_f32, full_reduction ? INT64_MIN : dim, keepdim, stream);
            return result_f32.to(DType::BFloat16);
        }
        case DType::Complex64:
        case DType::Complex128: {
            // Complex sum: reduce real and imaginary parts independently using
            // the real-valued sum kernels, then recombine. Matches the CPU
            // behaviour where Σ c_i = (Σ Re(c_i)) + i·(Σ Im(c_i)).
            auto real_part = tenzor::real(input);
            auto imag_part = tenzor::imag(input);
            auto real_sum = sum_kernel(real_part, full_reduction ? INT64_MIN : dim, keepdim, stream);
            auto imag_sum = sum_kernel(imag_part, full_reduction ? INT64_MIN : dim, keepdim, stream);
            return tenzor::complex(real_sum, imag_sum);
        }
        default:
            throw std::runtime_error("sum: unsupported dtype");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in sum_kernel: ") + hipGetErrorString(err));
    }

    if (dtype == DType::Float16) {
        fp16_saturate(output.data_ptr(), output.numel(), stream);
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

    if (dtype != DType::Float32 && dtype != DType::Float64 && dtype != DType::Float16 && dtype != DType::BFloat16) {
        throw std::runtime_error("mean: only Float32, Float64, Float16, and BFloat16 are supported");
    }

    // Float16: upcast to Float32, compute mean, convert back
    if (dtype == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = mean_kernel(input_f32, dim, keepdim, stream);
        auto result_f16 = result_f32.to(DType::Float16);
        fp16_saturate(result_f16.data_ptr(), result_f16.numel(), stream);
        return result_f16;
    }

    // BFloat16: upcast to Float32, compute mean, convert back
    if (dtype == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = mean_kernel(input_f32, dim, keepdim, stream);
        return result_f32.to(DType::BFloat16);
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
        // Return 0 for mean of empty tensor - this is practical for loss computation
        // in object detection where no samples may be selected
        return sum_result;  // sum_result is already 0 for empty tensor
    }

    // Divide by count using scaling kernel
    const int64_t n = sum_result.numel();

    const int block_size = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block

    if (dtype == DType::Float32) {
        auto* data = sum_result.data<float>();
        const float scale = 1.0f / static_cast<float>(count);

        int num_blocks = (n + block_size - 1) / block_size;
        hipLaunchKernelGGL(scale_kernel<float>, dim3(num_blocks), dim3(block_size), 0, stream,
            data, scale, n);
    } else if (dtype == DType::Float64) {
        auto* data = sum_result.data<double>();
        const double scale = 1.0 / static_cast<double>(count);

        int num_blocks = (n + block_size - 1) / block_size;
        hipLaunchKernelGGL(scale_kernel<double>, dim3(num_blocks), dim3(block_size), 0, stream,
            data, scale, n);
    } else {  // Float16
        auto* data = reinterpret_cast<__half*>(sum_result.data<Float16>());
        const float scale = 1.0f / static_cast<float>(count);

        int num_blocks = (n + block_size - 1) / block_size;
        hipLaunchKernelGGL(scale_kernel_fp16, dim3(num_blocks), dim3(block_size), 0, stream,
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
        case DType::Float16: {
            auto input_f32 = input.to(DType::Float32);
            auto result_f32 = max_kernel(input_f32, dim, keepdim, stream);
            return result_f32.to(DType::Float16);
        }
        case DType::BFloat16: {
            auto input_f32 = input.to(DType::Float32);
            auto result_f32 = max_kernel(input_f32, dim, keepdim, stream);
            return result_f32.to(DType::BFloat16);
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

    // Normalize user-specified negative dims (e.g. -1 = last) to positive
    // while leaving INT64_MIN alone — that's the project-wide "reduce all
    // dims" sentinel which the rest of this kernel detects via dim < 0.
    if (dim != INT64_MIN && dim < 0) {
        dim += static_cast<int64_t>(input_shape.size());
    }

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
        case DType::Float16: {
            auto input_f32 = input.to(DType::Float32);
            auto result_f32 = min_kernel(input_f32, dim, keepdim, stream);
            return result_f32.to(DType::Float16);
        }
        case DType::BFloat16: {
            auto input_f32 = input.to(DType::Float32);
            auto result_f32 = min_kernel(input_f32, dim, keepdim, stream);
            return result_f32.to(DType::BFloat16);
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
__global__ void argmax_kernel(const T* input, int64_t* output_idx, T* output_val, int64_t n) {
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
        output_idx[blockIdx.x] = sidx[0];
        output_val[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief Second-pass argmax kernel: reduces partial (value, index) pairs to a single result
 */
template<typename T>
__global__ void argmax_final_kernel(const T* partial_vals, const int64_t* partial_idx,
                                     int64_t* output, int num_partials) {
    __shared__ T sdata[REDUCTION_BLOCK_SIZE];
    __shared__ int64_t sidx[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    T max_val = std::numeric_limits<T>::lowest();
    int64_t max_idx = 0;

    // Grid-stride load (single block, so stride = blockDim.x)
    for (int i = tid; i < num_partials; i += blockDim.x) {
        T v = partial_vals[i];
        if (v > max_val) {
            max_val = v;
            max_idx = partial_idx[i];
        }
    }

    sdata[tid] = max_val;
    sidx[tid] = max_idx;
    __syncthreads();

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
        output[0] = sidx[0];
    }
}

template<typename T>
__global__ void argmin_kernel(const T* input, int64_t* output_idx, T* output_val, int64_t n) {
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
        output_idx[blockIdx.x] = sidx[0];
        output_val[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief Second-pass argmin kernel: reduces partial (value, index) pairs to a single result
 */
template<typename T>
__global__ void argmin_final_kernel(const T* partial_vals, const int64_t* partial_idx,
                                     int64_t* output, int num_partials) {
    __shared__ T sdata[REDUCTION_BLOCK_SIZE];
    __shared__ int64_t sidx[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    T min_val = std::numeric_limits<T>::max();
    int64_t min_idx = 0;

    for (int i = tid; i < num_partials; i += blockDim.x) {
        T v = partial_vals[i];
        if (v < min_val) {
            min_val = v;
            min_idx = partial_idx[i];
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
        output[0] = sidx[0];
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
__global__ void var_kernel(const T* input, const T* mean_ptr, T* output, int64_t n) {
    __shared__ T sdata[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    T mean = *mean_ptr;

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
 * @brief Second-pass prod kernel: reduces partial products via multiplication
 */
template<typename T>
__global__ void prod_final_kernel(const T* partials, T* output, int num_partials) {
    __shared__ T sdata[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    T val = T(1);
    for (int i = tid; i < num_partials; i += blockDim.x) {
        val *= partials[i];
    }

    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] *= sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = sdata[0];
    }
}

/**
 * @brief Second-pass var kernel: sums partials and divides by count
 * @param partials Partial sum-of-squared-differences from first pass
 * @param output Final variance result
 * @param num_partials Number of partial results
 * @param count Divisor: n for population variance, (n-1) for sample variance
 */
template<typename T>
__global__ void var_final_kernel(const T* partials, T* output, int num_partials, T count) {
    __shared__ T sdata[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    T val = T(0);
    for (int i = tid; i < num_partials; i += blockDim.x) {
        val += partials[i];
    }

    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = sdata[0] / count;
    }
}

/**
 * @brief Second-pass norm kernel: sums partials and applies pow(sum, 1/p)
 */
template<typename T>
__global__ void norm_final_kernel(const T* partials, T* output, int num_partials, T p) {
    __shared__ T sdata[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    T val = T(0);
    for (int i = tid; i < num_partials; i += blockDim.x) {
        val += partials[i];
    }

    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = pow(sdata[0], T(1) / p);
    }
}

/**
 * @brief Second-pass any kernel: OR reduction over uint8_t partial results
 */
__global__ void any_final_kernel(const uint8_t* partials, uint8_t* output, int num_partials) {
    __shared__ uint8_t sdata[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    uint8_t val = 0;
    for (int i = tid; i < num_partials; i += blockDim.x) {
        val = val | partials[i];
    }

    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = sdata[tid] | sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = sdata[0];
    }
}

/**
 * @brief Second-pass all kernel: AND reduction over uint8_t partial results
 */
__global__ void all_final_kernel(const uint8_t* partials, uint8_t* output, int num_partials) {
    __shared__ uint8_t sdata[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    uint8_t val = (tid < num_partials) ? partials[tid] : 1;
    for (int i = tid + blockDim.x; i < num_partials; i += blockDim.x) {
        val = val & partials[i];
    }

    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = sdata[tid] & sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = sdata[0];
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
            int64_t* d_partial_idx;
            float* d_partial_vals;
            HIP_CHECK(hipMalloc(&d_partial_idx, num_blocks * sizeof(int64_t)));
            HIP_CHECK(hipMalloc(&d_partial_vals, num_blocks * sizeof(float)));

            hipLaunchKernelGGL(argmax_kernel<float>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<float>(), d_partial_idx, d_partial_vals, n);

            if (num_blocks > 1) {
                // Second-pass: single block reduces partial (value, index) pairs on device
                hipLaunchKernelGGL(argmax_final_kernel<float>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    d_partial_vals, d_partial_idx, output.data<int64_t>(), num_blocks);
            } else {
                HIP_CHECK(hipMemcpy(output.data<int64_t>(), d_partial_idx, sizeof(int64_t), hipMemcpyDeviceToDevice));
            }

            HIP_CHECK(hipFree(d_partial_idx));
            HIP_CHECK(hipFree(d_partial_vals));
        } else if (dtype == DType::Float64) {
            int64_t* d_partial_idx;
            double* d_partial_vals;
            HIP_CHECK(hipMalloc(&d_partial_idx, num_blocks * sizeof(int64_t)));
            HIP_CHECK(hipMalloc(&d_partial_vals, num_blocks * sizeof(double)));

            hipLaunchKernelGGL(argmax_kernel<double>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<double>(), d_partial_idx, d_partial_vals, n);

            if (num_blocks > 1) {
                hipLaunchKernelGGL(argmax_final_kernel<double>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    d_partial_vals, d_partial_idx, output.data<int64_t>(), num_blocks);
            } else {
                HIP_CHECK(hipMemcpy(output.data<int64_t>(), d_partial_idx, sizeof(int64_t), hipMemcpyDeviceToDevice));
            }

            HIP_CHECK(hipFree(d_partial_idx));
            HIP_CHECK(hipFree(d_partial_vals));
        } else if (dtype == DType::Float16 || dtype == DType::BFloat16) {
            auto input_f32 = input.to(DType::Float32);
            return argmax_kernel(input_f32, dim, keepdim, stream);
        } else {
            throw std::runtime_error("argmax: unsupported dtype");
        }
    } else {
        // Dimensional argmax
        if (dtype == DType::Float16 || dtype == DType::BFloat16) {
            auto input_f32 = input.to(DType::Float32);
            return argmax_kernel(input_f32, dim, keepdim, stream);
        }

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
            int64_t* d_partial_idx;
            float* d_partial_vals;
            HIP_CHECK(hipMalloc(&d_partial_idx, num_blocks * sizeof(int64_t)));
            HIP_CHECK(hipMalloc(&d_partial_vals, num_blocks * sizeof(float)));

            hipLaunchKernelGGL(argmin_kernel<float>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<float>(), d_partial_idx, d_partial_vals, n);

            if (num_blocks > 1) {
                hipLaunchKernelGGL(argmin_final_kernel<float>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    d_partial_vals, d_partial_idx, output.data<int64_t>(), num_blocks);
            } else {
                HIP_CHECK(hipMemcpy(output.data<int64_t>(), d_partial_idx, sizeof(int64_t), hipMemcpyDeviceToDevice));
            }

            HIP_CHECK(hipFree(d_partial_idx));
            HIP_CHECK(hipFree(d_partial_vals));
        } else if (dtype == DType::Float64) {
            int64_t* d_partial_idx;
            double* d_partial_vals;
            HIP_CHECK(hipMalloc(&d_partial_idx, num_blocks * sizeof(int64_t)));
            HIP_CHECK(hipMalloc(&d_partial_vals, num_blocks * sizeof(double)));

            hipLaunchKernelGGL(argmin_kernel<double>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<double>(), d_partial_idx, d_partial_vals, n);

            if (num_blocks > 1) {
                hipLaunchKernelGGL(argmin_final_kernel<double>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    d_partial_vals, d_partial_idx, output.data<int64_t>(), num_blocks);
            } else {
                HIP_CHECK(hipMemcpy(output.data<int64_t>(), d_partial_idx, sizeof(int64_t), hipMemcpyDeviceToDevice));
            }

            HIP_CHECK(hipFree(d_partial_idx));
            HIP_CHECK(hipFree(d_partial_vals));
        } else if (dtype == DType::Float16 || dtype == DType::BFloat16) {
            auto input_f32 = input.to(DType::Float32);
            return argmin_kernel(input_f32, dim, keepdim, stream);
        } else {
            throw std::runtime_error("argmin: unsupported dtype");
        }
    } else {
        // Dimensional argmin
        if (dtype == DType::Float16 || dtype == DType::BFloat16) {
            auto input_f32 = input.to(DType::Float32);
            return argmin_kernel(input_f32, dim, keepdim, stream);
        }

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

    // Normalize user-specified negative dims (e.g. -1 = last) to positive
    // while leaving INT64_MIN alone — that's the project-wide "reduce all
    // dims" sentinel which the rest of this kernel detects via dim < 0.
    if (dim != INT64_MIN && dim < 0) {
        dim += static_cast<int64_t>(input_shape.size());
    }

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
                hipLaunchKernelGGL(prod_final_kernel<float>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    d_partial, output.data<float>(), num_blocks);
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
                hipLaunchKernelGGL(prod_final_kernel<double>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    d_partial, output.data<double>(), num_blocks);
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
                hipLaunchKernelGGL(prod_final_kernel<int32_t>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    d_partial, output.data<int32_t>(), num_blocks);
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
                hipLaunchKernelGGL(prod_final_kernel<int64_t>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    d_partial, output.data<int64_t>(), num_blocks);
            } else {
                HIP_CHECK(hipMemcpy(output.data<int64_t>(), d_partial, sizeof(int64_t), hipMemcpyDeviceToDevice));
            }

            HIP_CHECK(hipFree(d_partial));
        } else if (dtype == DType::Float16) {
            auto input_f32 = input.to(DType::Float32);
            auto result_f32 = prod_kernel(input_f32, dim, keepdim, stream);
            return result_f32.to(DType::Float16);
        } else if (dtype == DType::BFloat16) {
            auto input_f32 = input.to(DType::Float32);
            auto result_f32 = prod_kernel(input_f32, dim, keepdim, stream);
            return result_f32.to(DType::BFloat16);
        } else {
            throw std::runtime_error("prod: unsupported dtype");
        }
    } else {
        // Dimensional product
        if (dtype == DType::Float16) {
            auto input_f32 = input.to(DType::Float32);
            auto result_f32 = prod_kernel(input_f32, dim, keepdim, stream);
            return result_f32.to(DType::Float16);
        }
        if (dtype == DType::BFloat16) {
            auto input_f32 = input.to(DType::Float32);
            auto result_f32 = prod_kernel(input_f32, dim, keepdim, stream);
            return result_f32.to(DType::BFloat16);
        }

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

    // Normalize user-specified negative dims (e.g. -1 = last) to positive
    // while leaving INT64_MIN alone — that's the project-wide "reduce all
    // dims" sentinel which the rest of this kernel detects via dim < 0.
    if (dim != INT64_MIN && dim < 0) {
        dim += static_cast<int64_t>(input.shape().size());
    }

    if (dtype != DType::Float32 && dtype != DType::Float64 && dtype != DType::Float16 && dtype != DType::BFloat16) {
        throw std::runtime_error("var: only Float32, Float64, Float16, and BFloat16 are supported");
    }

    // Float16: upcast to Float32, compute variance, convert back
    if (dtype == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = var_kernel(input_f32, dim, keepdim, unbiased, stream);
        return result_f32.to(DType::Float16);
    }

    // BFloat16: upcast to Float32, compute variance, convert back
    if (dtype == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = var_kernel(input_f32, dim, keepdim, unbiased, stream);
        return result_f32.to(DType::BFloat16);
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
            float* d_partial;
            HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(float)));

            hipLaunchKernelGGL(var_kernel<float>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<float>(), mean_tensor.data<float>(), d_partial, n);

            float divisor = static_cast<float>(unbiased ? (n - 1) : n);
            hipLaunchKernelGGL(var_final_kernel<float>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                d_partial, output.data<float>(), num_blocks, divisor);

            HIP_CHECK(hipFree(d_partial));
        } else {
            double* d_partial;
            HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(double)));

            hipLaunchKernelGGL(var_kernel<double>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<double>(), mean_tensor.data<double>(), d_partial, n);

            double divisor = static_cast<double>(unbiased ? (n - 1) : n);
            hipLaunchKernelGGL(var_final_kernel<double>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                d_partial, output.data<double>(), num_blocks, divisor);

            HIP_CHECK(hipFree(d_partial));
        }
    } else {
        // Per-dimension variance using Welford's algorithm
        const auto& input_shape = input.shape();
        const auto& input_strides = input.strides();
        int64_t ndim = static_cast<int64_t>(input_shape.size());
        int64_t actual_dim = dim;
        if (actual_dim < 0) actual_dim += ndim;
        if (actual_dim < 0 || actual_dim >= ndim) {
            throw std::runtime_error("var: dimension out of range");
        }
        int64_t correction = unbiased ? 1 : 0;

        if (dtype == DType::Float32) {
            launch_dim_reduction_var(
                input.data<float>(), output.data<float>(),
                std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                actual_dim, correction, stream);
        } else {
            launch_dim_reduction_var(
                input.data<double>(), output.data<double>(),
                std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                actual_dim, correction, stream);
        }
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

/**
 * @brief Standard deviation reduction kernel
 */
auto std_kernel(const Tensor& input, int64_t dim, bool keepdim, bool unbiased, hipStream_t stream) -> Tensor {
    // Float16: upcast to Float32, compute std, convert back
    if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = std_kernel(input_f32, dim, keepdim, unbiased, stream);
        return result_f32.to(DType::Float16);
    }

    // BFloat16: upcast to Float32, compute std, convert back
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = std_kernel(input_f32, dim, keepdim, unbiased, stream);
        return result_f32.to(DType::BFloat16);
    }

    auto var_result = var_kernel(input, dim, keepdim, unbiased, stream);

    // Compute sqrt of variance
    int64_t n = var_result.numel();
    Tensor output(std::vector<int64_t>(var_result.shape().begin(), var_result.shape().end()),
                  var_result.dtype(), var_result.device());

    if (n == 0) return output;

    dim3 grid, block;
    const int blk_size = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    block = dim3(blk_size, 1, 1);
    grid = dim3((n + blk_size - 1) / blk_size, 1, 1);

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

    if (dtype != DType::Float32 && dtype != DType::Float64 && dtype != DType::Float16 && dtype != DType::BFloat16) {
        throw std::runtime_error("norm: only Float32, Float64, Float16, and BFloat16 are supported");
    }

    // Float16: upcast to Float32, compute norm, convert back
    if (dtype == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = norm_kernel(input_f32, p, dim, keepdim, stream);
        return result_f32.to(DType::Float16);
    }

    // BFloat16: upcast to Float32, compute norm, convert back
    if (dtype == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = norm_kernel(input_f32, p, dim, keepdim, stream);
        return result_f32.to(DType::BFloat16);
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

            if (num_blocks > 1) {
                hipLaunchKernelGGL(norm_final_kernel<float>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    d_partial, output.data<float>(), num_blocks, p);
            } else {
                // Single block: still need to apply pow(sum, 1/p)
                hipLaunchKernelGGL(norm_final_kernel<float>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    d_partial, output.data<float>(), 1, p);
            }

            HIP_CHECK(hipFree(d_partial));
        } else {
            double* d_partial;
            HIP_CHECK(hipMalloc(&d_partial, num_blocks * sizeof(double)));

            hipLaunchKernelGGL(norm_kernel<double>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                input.data<double>(), d_partial, n, static_cast<double>(p));

            double p_double = static_cast<double>(p);
            if (num_blocks > 1) {
                hipLaunchKernelGGL(norm_final_kernel<double>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    d_partial, output.data<double>(), num_blocks, p_double);
            } else {
                hipLaunchKernelGGL(norm_final_kernel<double>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    d_partial, output.data<double>(), 1, p_double);
            }

            HIP_CHECK(hipFree(d_partial));
        }
    } else {
        // Per-dimension p-norm reduction
        const auto& input_shape = input.shape();
        const auto& input_strides = input.strides();
        int64_t ndim = static_cast<int64_t>(input_shape.size());
        int64_t actual_dim = dim;
        if (actual_dim < 0) actual_dim += ndim;
        if (actual_dim < 0 || actual_dim >= ndim) {
            throw std::runtime_error("norm: dimension out of range");
        }

        if (dtype == DType::Float32) {
            launch_dim_reduction_norm(
                input.data<float>(), output.data<float>(),
                std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                actual_dim, p, stream);
        } else {
            launch_dim_reduction_norm(
                input.data<double>(), output.data<double>(),
                std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                actual_dim, static_cast<double>(p), stream);
        }
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

// ============================================================================
// Any/All Reduction Kernels
// ============================================================================

/**
 * @brief Full any reduction kernel - OR reduction over entire tensor
 * @tparam T Data type
 * @param input Input tensor data
 * @param output Output buffer (one value per block)
 * @param n Total number of elements
 *
 * Each thread checks its elements for non-zero, block reduces with OR.
 */
template<typename T>
__global__ void any_reduce_kernel(const T* input, uint8_t* output, int64_t n) {
    __shared__ int shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Grid-stride loop: check if any element is non-zero
    int thread_any = 0;
    for (int64_t i = idx; i < n; i += grid_size) {
        if (input[i] != T(0)) {
            thread_any = 1;
        }
    }

    shared[tid] = thread_any;
    __syncthreads();

    // Block-level OR reduction in LDS
    for (int stride = blockDim.x / 2; stride >= warpSize; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = shared[tid] | shared[tid + stride];
        }
        __syncthreads();
    }

    // Wavefront-level reduction
    if (tid < warpSize) {
        int val = shared[tid];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val |= __shfl_down(val, offset);
        }
        if (tid == 0) {
            output[blockIdx.x] = val ? 1 : 0;
        }
    }
}

/**
 * @brief Full all reduction kernel - AND reduction over entire tensor
 * @tparam T Data type
 * @param input Input tensor data
 * @param output Output buffer (one value per block)
 * @param n Total number of elements
 */
template<typename T>
__global__ void all_reduce_kernel(const T* input, uint8_t* output, int64_t n) {
    __shared__ int shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Grid-stride loop: check if all elements are non-zero
    int thread_all = 1;
    for (int64_t i = idx; i < n; i += grid_size) {
        if (input[i] == T(0)) {
            thread_all = 0;
        }
    }

    shared[tid] = thread_all;
    __syncthreads();

    // Block-level AND reduction in LDS
    for (int stride = blockDim.x / 2; stride >= warpSize; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = shared[tid] & shared[tid + stride];
        }
        __syncthreads();
    }

    // Wavefront-level reduction
    if (tid < warpSize) {
        int val = shared[tid];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val &= __shfl_down(val, offset);
        }
        if (tid == 0) {
            output[blockIdx.x] = val ? 1 : 0;
        }
    }
}

/**
 * @brief Any reduction along a specific dimension
 * @tparam T Data type
 */
template<typename T>
__global__ void any_along_dim_kernel(
    const T* input,
    uint8_t* output,
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
    int64_t indices[8];
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        int64_t dim_size_d = input_shape[d];
        indices[d] = tmp % dim_size_d;
        tmp /= dim_size_d;
    }

    // Check if any element along the reduction dimension is non-zero
    uint8_t result = 0;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;

        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }

        if (input[in_idx] != T(0)) {
            result = 1;
            break;  // Short-circuit: found a non-zero
        }
    }

    output[out_idx] = result;
}

/**
 * @brief All reduction along a specific dimension
 * @tparam T Data type
 */
template<typename T>
__global__ void all_along_dim_kernel(
    const T* input,
    uint8_t* output,
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
    int64_t indices[8];
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        int64_t dim_size_d = input_shape[d];
        indices[d] = tmp % dim_size_d;
        tmp /= dim_size_d;
    }

    // Check if all elements along the reduction dimension are non-zero
    uint8_t result = 1;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;

        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }

        if (input[in_idx] == T(0)) {
            result = 0;
            break;  // Short-circuit: found a zero
        }
    }

    output[out_idx] = result;
}

/**
 * @brief Launch full any reduction
 * @tparam T Data type
 */
template<typename T>
static void launch_full_any(const T* d_input, uint8_t* d_output, int64_t n, hipStream_t stream) {
    if (n == 0) {
        // any of empty tensor = false
        uint8_t zero = 0;
        HIP_CHECK(hipMemcpyAsync(d_output, &zero, sizeof(uint8_t), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        hipLaunchKernelGGL(any_reduce_kernel<T>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_input, d_output, n);
    } else {
        // Phase 1: Reduce to num_blocks intermediate uint8_t results
        uint8_t* d_temp;
        HIP_CHECK(hipMalloc(&d_temp, num_blocks * sizeof(uint8_t)));
        hipLaunchKernelGGL(any_reduce_kernel<T>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_input, d_temp, n);

        // Phase 2: Final OR reduction over intermediates on device
        hipLaunchKernelGGL(any_final_kernel, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_temp, d_output, num_blocks);

        HIP_CHECK(hipFree(d_temp));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
}

/**
 * @brief Launch full all reduction
 * @tparam T Data type
 */
template<typename T>
static void launch_full_all(const T* d_input, uint8_t* d_output, int64_t n, hipStream_t stream) {
    if (n == 0) {
        // all of empty tensor = true (vacuous truth)
        uint8_t one = 1;
        HIP_CHECK(hipMemcpyAsync(d_output, &one, sizeof(uint8_t), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        hipLaunchKernelGGL(all_reduce_kernel<T>, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_input, d_output, n);
    } else {
        uint8_t* d_temp;
        HIP_CHECK(hipMalloc(&d_temp, num_blocks * sizeof(uint8_t)));
        hipLaunchKernelGGL(all_reduce_kernel<T>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_input, d_temp, n);

        // Phase 2: Final AND reduction over intermediates on device
        hipLaunchKernelGGL(all_final_kernel, dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
            d_temp, d_output, num_blocks);

        HIP_CHECK(hipFree(d_temp));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
}

/**
 * @brief Launch dimensional any reduction
 * @tparam T Data type
 */
template<typename T>
static void launch_dim_any(
    const T* d_input,
    uint8_t* d_output,
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
    hipLaunchKernelGGL(any_along_dim_kernel<T>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size);

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_shape));
    HIP_CHECK(hipFree(d_strides));
}

/**
 * @brief Launch dimensional all reduction
 * @tparam T Data type
 */
template<typename T>
static void launch_dim_all(
    const T* d_input,
    uint8_t* d_output,
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
    hipLaunchKernelGGL(all_along_dim_kernel<T>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size);

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_shape));
    HIP_CHECK(hipFree(d_strides));
}

/**
 * @brief Any reduction kernel - returns Bool tensor
 * @param input Input tensor
 * @param dim Dimension to reduce along (INT64_MIN for full reduction)
 * @param keepdim Whether to keep the reduced dimension
 * @param stream HIP stream
 * @return Bool tensor with any-reduction result
 */
auto any_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor {
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

    // Output is always Bool (uint8_t)
    Tensor output(output_shape, DType::Bool, device);
    auto* output_data = output.data<uint8_t>();

    // Dispatch based on input dtype - cast to appropriate type for comparison
    auto launch = [&](auto* input_data) {
        if (full_reduction) {
            launch_full_any(input_data, output_data, input.numel(), stream);
        } else {
            launch_dim_any(
                input_data, output_data,
                std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                dim, stream
            );
        }
    };

    switch (dtype) {
        case DType::Float32:
            launch(input.data<float>());
            break;
        case DType::Float64:
            launch(input.data<double>());
            break;
        case DType::Float16:
            launch(reinterpret_cast<const __half*>(input.data<Float16>()));
            break;
        case DType::Int32:
            launch(input.data<int32_t>());
            break;
        case DType::Int64:
            launch(input.data<int64_t>());
            break;
        case DType::Bool:
        case DType::UInt8:
            launch(input.data<uint8_t>());
            break;
        case DType::Int8:
            launch(input.data<int8_t>());
            break;
        case DType::BFloat16: {
            auto input_f32 = input.to(DType::Float32);
            return any_kernel(input_f32, dim, keepdim, stream);
        }
        default:
            throw std::runtime_error("any: unsupported dtype");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in any_kernel: ") + hipGetErrorString(err));
    }

    return output;
}

/**
 * @brief All reduction kernel - returns Bool tensor
 * @param input Input tensor
 * @param dim Dimension to reduce along (INT64_MIN for full reduction)
 * @param keepdim Whether to keep the reduced dimension
 * @param stream HIP stream
 * @return Bool tensor with all-reduction result
 */
auto all_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor {
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

    // Output is always Bool (uint8_t)
    Tensor output(output_shape, DType::Bool, device);
    auto* output_data = output.data<uint8_t>();

    auto launch = [&](auto* input_data) {
        if (full_reduction) {
            launch_full_all(input_data, output_data, input.numel(), stream);
        } else {
            launch_dim_all(
                input_data, output_data,
                std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                dim, stream
            );
        }
    };

    switch (dtype) {
        case DType::Float32:
            launch(input.data<float>());
            break;
        case DType::Float64:
            launch(input.data<double>());
            break;
        case DType::Float16:
            launch(reinterpret_cast<const __half*>(input.data<Float16>()));
            break;
        case DType::Int32:
            launch(input.data<int32_t>());
            break;
        case DType::Int64:
            launch(input.data<int64_t>());
            break;
        case DType::Bool:
        case DType::UInt8:
            launch(input.data<uint8_t>());
            break;
        case DType::Int8:
            launch(input.data<int8_t>());
            break;
        case DType::BFloat16: {
            auto input_f32 = input.to(DType::Float32);
            return all_kernel(input_f32, dim, keepdim, stream);
        }
        default:
            throw std::runtime_error("all: unsupported dtype");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in all_kernel: ") + hipGetErrorString(err));
    }

    return output;
}

// ============================================================================
// LogSumExp - Numerically stable log(sum(exp(x)))
// ============================================================================

/**
 * @brief Find max along a specific dimension for logsumexp
 * @tparam T Input data type
 * @tparam Acc Accumulation type (float for __half, T otherwise)
 */
template<typename T, typename Acc = T>
__global__ void logsumexp_max_along_dim_kernel(
    const T* __restrict__ input,
    Acc* __restrict__ max_out,
    const int64_t* __restrict__ input_shape,
    const int64_t* __restrict__ input_strides,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= output_size) return;

    int64_t indices[8];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) { indices[d] = 0; continue; }
        indices[d] = tmp % input_shape[d];
        tmp /= input_shape[d];
    }

    Acc m;
    if constexpr (std::is_same_v<Acc, float>) {
        m = -FLT_MAX;
    } else if constexpr (std::is_same_v<Acc, double>) {
        m = -DBL_MAX;
    } else {
        m = Acc(-1e38);
    }

    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }
        Acc val = Acc(input[in_idx]);
        if (val > m) m = val;
    }
    max_out[out_idx] = m;
}

/**
 * @brief Compute sum(exp(x - max)) and result = max + log(sum) along a dimension
 * @tparam T Input/output data type
 * @tparam Acc Accumulation type
 */
template<typename T, typename Acc = T>
__global__ void logsumexp_sum_exp_kernel(
    const T* __restrict__ input,
    const Acc* __restrict__ max_vals,
    T* __restrict__ output,
    const int64_t* __restrict__ input_shape,
    const int64_t* __restrict__ input_strides,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= output_size) return;

    int64_t indices[8];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) { indices[d] = 0; continue; }
        indices[d] = tmp % input_shape[d];
        tmp /= input_shape[d];
    }

    Acc m = max_vals[out_idx];
    // Handle inf: if max is +/-inf, result should be the same inf
    {
        float fm = float(m);
        uint32_t mbits; memcpy(&mbits, &fm, sizeof(mbits));
        if ((mbits & 0x7FFFFFFFu) == 0x7F800000u) {
            output[out_idx] = T(m);
            return;
        }
    }

    Acc sum = Acc(0);
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }
        sum += exp(Acc(input[in_idx]) - m);
    }
    output[out_idx] = T(log(sum) + m);
}

/**
 * @brief Full tensor logsumexp reduction using shared memory
 * @tparam T Input/output data type
 * @tparam Acc Accumulation type
 */
template<typename T, typename Acc = T>
__global__ void logsumexp_full_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t n
) {
    // Step 1: Find max using shared memory reduction
    __shared__ Acc smax[REDUCTION_BLOCK_SIZE];
    int tid = threadIdx.x;
    Acc local_max;
    if constexpr (std::is_same_v<Acc, float>) {
        local_max = -FLT_MAX;
    } else if constexpr (std::is_same_v<Acc, double>) {
        local_max = -DBL_MAX;
    } else {
        local_max = Acc(-1e38);
    }

    for (int64_t i = tid; i < n; i += blockDim.x) {
        Acc val = Acc(input[i]);
        if (val > local_max) local_max = val;
    }
    smax[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (smax[tid + stride] > smax[tid])
                smax[tid] = smax[tid + stride];
        }
        __syncthreads();
    }
    Acc m = smax[0];
    __syncthreads();

    // Handle inf
    {
        float fm = float(m);
        uint32_t mbits; memcpy(&mbits, &fm, sizeof(mbits));
        if ((mbits & 0x7FFFFFFFu) == 0x7F800000u) {
            if (tid == 0) output[0] = T(m);
            return;
        }
    }

    // Step 2: Sum exp(x - max) using shared memory reduction
    __shared__ Acc ssum[REDUCTION_BLOCK_SIZE];
    Acc local_sum = Acc(0);
    for (int64_t i = tid; i < n; i += blockDim.x) {
        local_sum += exp(Acc(input[i]) - m);
    }
    ssum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = T(log(ssum[0]) + m);
    }
}

/**
 * @brief LogSumExp reduction kernel - numerically stable log(sum(exp(x)))
 * @param input Input tensor
 * @param dim Dimension to reduce along (INT64_MIN for full reduction)
 * @param keepdim Whether to keep the reduced dimension
 * @param stream HIP stream
 * @return Output tensor with logsumexp
 */
auto logsumexp_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    if (dtype != DType::Float32 && dtype != DType::Float64 &&
        dtype != DType::Float16 && dtype != DType::BFloat16) {
        throw std::runtime_error("logsumexp: only Float32, Float64, Float16, and BFloat16 are supported");
    }

    // BFloat16: upcast to Float32, compute, convert back
    if (dtype == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = logsumexp_kernel(input_f32, dim, keepdim, stream);
        return result_f32.to(DType::BFloat16);
    }

    // Normalize negative dimension (but preserve special value INT64_MIN for full reduction)
    bool full_reduction = (dim == INT64_MIN);
    int64_t normalized_dim = dim;
    if (!full_reduction) {
        if (dim < 0) normalized_dim = ndim + dim;
        if (normalized_dim < 0 || normalized_dim >= ndim) {
            throw std::runtime_error("Dimension " + std::to_string(dim) +
                " out of range for tensor with " + std::to_string(ndim) + " dimensions");
        }
    }

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        full_reduction ? -1 : normalized_dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    if (full_reduction) {
        // Full reduction
        int64_t n = input.numel();
        if (n == 0) {
            // logsumexp of empty set is -inf
            switch (dtype) {
                case DType::Float32: {
                    float neg_inf = -FLT_MAX;
                    HIP_CHECK(hipMemcpyAsync(output.data<float>(), &neg_inf, sizeof(float),
                              hipMemcpyHostToDevice, stream));
                    break;
                }
                case DType::Float64: {
                    double neg_inf = -DBL_MAX;
                    HIP_CHECK(hipMemcpyAsync(output.data<double>(), &neg_inf, sizeof(double),
                              hipMemcpyHostToDevice, stream));
                    break;
                }
                default: break;
            }
            HIP_CHECK(hipStreamSynchronize(stream));
            return output;
        }

        switch (dtype) {
            case DType::Float32:
                hipLaunchKernelGGL((logsumexp_full_kernel<float, float>),
                    dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    input.data<float>(), output.data<float>(), n);
                break;
            case DType::Float64:
                hipLaunchKernelGGL((logsumexp_full_kernel<double, double>),
                    dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    input.data<double>(), output.data<double>(), n);
                break;
            case DType::Float16:
                hipLaunchKernelGGL((logsumexp_full_kernel<__half, float>),
                    dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    reinterpret_cast<const __half*>(input.data<Float16>()),
                    reinterpret_cast<__half*>(output.data<Float16>()), n);
                break;
            default: break;
        }
    } else {
        // Dim reduction
        int64_t dim_size = input_shape[normalized_dim];
        int64_t output_size = 1;
        for (int64_t i = 0; i < ndim; i++) {
            if (i != normalized_dim) output_size *= input_shape[i];
        }

        if (output_size == 0 || dim_size == 0) {
            return output;
        }

        auto shape_vec = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        auto strides_vec = std::vector<int64_t>(input_strides.begin(), input_strides.end());

        // Copy shape and strides to device
        int64_t* d_shape;
        int64_t* d_strides;
        HIP_CHECK(hipMalloc(&d_shape, ndim * sizeof(int64_t)));
        HIP_CHECK(hipMalloc(&d_strides, ndim * sizeof(int64_t)));
        HIP_CHECK(hipMemcpy(d_shape, shape_vec.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_strides, strides_vec.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));

        int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;

        switch (dtype) {
            case DType::Float32: {
                Tensor max_buf({output_size}, DType::Float32, device);
                hipLaunchKernelGGL((logsumexp_max_along_dim_kernel<float, float>),
                    dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    input.data<float>(), max_buf.data<float>(),
                    d_shape, d_strides, ndim, normalized_dim, output_size, dim_size);

                hipLaunchKernelGGL((logsumexp_sum_exp_kernel<float, float>),
                    dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    input.data<float>(), max_buf.data<float>(), output.data<float>(),
                    d_shape, d_strides, ndim, normalized_dim, output_size, dim_size);
                break;
            }
            case DType::Float64: {
                Tensor max_buf({output_size}, DType::Float64, device);
                hipLaunchKernelGGL((logsumexp_max_along_dim_kernel<double, double>),
                    dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    input.data<double>(), max_buf.data<double>(),
                    d_shape, d_strides, ndim, normalized_dim, output_size, dim_size);

                hipLaunchKernelGGL((logsumexp_sum_exp_kernel<double, double>),
                    dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    input.data<double>(), max_buf.data<double>(), output.data<double>(),
                    d_shape, d_strides, ndim, normalized_dim, output_size, dim_size);
                break;
            }
            case DType::Float16: {
                // Use float accumulator for max values
                Tensor max_buf({output_size}, DType::Float32, device);
                hipLaunchKernelGGL((logsumexp_max_along_dim_kernel<__half, float>),
                    dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    reinterpret_cast<const __half*>(input.data<Float16>()),
                    max_buf.data<float>(),
                    d_shape, d_strides, ndim, normalized_dim, output_size, dim_size);

                hipLaunchKernelGGL((logsumexp_sum_exp_kernel<__half, float>),
                    dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                    reinterpret_cast<const __half*>(input.data<Float16>()),
                    max_buf.data<float>(),
                    reinterpret_cast<__half*>(output.data<Float16>()),
                    d_shape, d_strides, ndim, normalized_dim, output_size, dim_size);
                break;
            }
            default: break;
        }

        // Must wait for kernels to complete before freeing device memory they use
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipFree(d_shape));
        HIP_CHECK(hipFree(d_strides));
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in logsumexp_kernel: ") + hipGetErrorString(err));
    }

    return output;
}

// ============================================================================
// Median kernel — sort-based, one block per slice
// ============================================================================

template <typename T>
__global__ void median_per_slice_kernel(
    const T* __restrict__ input,
    T* __restrict__ out_values,
    int64_t* __restrict__ out_indices,
    const int64_t* __restrict__ shape,
    const int64_t* __restrict__ strides,
    int64_t ndim, int64_t dim, int64_t dim_size, int64_t outer_size)
{
    int64_t slice_idx = blockIdx.x;
    if (slice_idx >= outer_size) return;

    // Compute multi-dim index for this slice (skip the reduction dim)
    int64_t tmp = slice_idx;
    int64_t base_offset = 0;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) continue;
        int64_t coord = tmp % shape[d];
        tmp /= shape[d];
        base_offset += coord * strides[d];
    }

    // Each thread in the block cooperates; for simplicity use thread 0
    // (dim_size is typically small enough for a single-thread sort)
    if (threadIdx.x == 0) {
        // Allocate local arrays via dynamic shared memory would be complex;
        // use register-based approach for small dims, else naive loop
        // For GPU: simple insertion sort in registers (dim_size usually < 10K)
        // We'll use a selection approach: partial sort to find median

        int64_t mid = (dim_size - 1) / 2;

        // Find the (mid+1)-th smallest element using repeated scans
        // For better perf we do a simple selection algorithm
        T pivot_val{};
        int64_t pivot_orig_idx = 0;

        // Count-based selection: for each element, count how many are smaller
        // This is O(dim_size^2) but fully parallel-friendly
        for (int64_t i = 0; i < dim_size; ++i) {
            int64_t offset_i = base_offset + i * strides[dim];
            T val_i = input[offset_i];
            int64_t rank = 0;
            for (int64_t j = 0; j < dim_size; ++j) {
                T val_j = input[base_offset + j * strides[dim]];
                if (val_j < val_i || (val_j == val_i && j < i)) {
                    rank++;
                }
            }
            if (rank == mid) {
                pivot_val = val_i;
                pivot_orig_idx = i;
                break;
            }
        }

        out_values[slice_idx] = pivot_val;
        out_indices[slice_idx] = pivot_orig_idx;
    }
}

auto median_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream)
    -> std::vector<Tensor>
{
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    if (input.numel() == 0) {
        throw std::runtime_error("median: cannot compute median of empty tensor");
    }

    // Normalize dim: INT64_MIN means full reduction
    int64_t normalized_dim = dim;
    if (normalized_dim == INT64_MIN) {
        // Full reduction: flatten and reduce along dim 0
        Tensor flat = input.contiguous().reshape({input.numel()});
        auto result = median_kernel(flat, 0, false, stream);
        if (keepdim) {
            std::vector<int64_t> kshape(ndim, 1);
            result[0] = result[0].reshape(kshape);
            result[1] = result[1].reshape(kshape);
        }
        return result;
    }

    if (normalized_dim < 0) normalized_dim += ndim;

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        normalized_dim, keepdim);

    int64_t dim_size = input_shape[normalized_dim];
    int64_t outer_size = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != normalized_dim) outer_size *= input_shape[d];
    }

    Tensor values(output_shape, dtype, device);
    Tensor indices(output_shape, DType::Int64, device);

    // Copy shape and strides to device
    int64_t* d_shape;
    int64_t* d_strides;
    HIP_CHECK(hipMalloc(&d_shape, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_strides, ndim * sizeof(int64_t)));

    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
    std::vector<int64_t> strides_vec(input.strides().begin(), input.strides().end());
    HIP_CHECK(hipMemcpy(d_shape, shape_vec.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_strides, strides_vec.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));

    int block_size = 1; // One thread per slice (selection algorithm)
    int num_blocks = static_cast<int>(outer_size);

    auto launch = [&]<typename T>(T*) {
        hipLaunchKernelGGL((median_per_slice_kernel<T>),
            dim3(num_blocks), dim3(block_size), 0, stream,
            input.data<T>(), values.data<T>(), indices.data<int64_t>(),
            d_shape, d_strides, ndim, normalized_dim, dim_size, outer_size);
    };

    switch (dtype) {
        case DType::Float32: launch(static_cast<float*>(nullptr)); break;
        case DType::Float64: launch(static_cast<double*>(nullptr)); break;
        case DType::Int32:   launch(static_cast<int32_t*>(nullptr)); break;
        case DType::Int64:   launch(static_cast<int64_t*>(nullptr)); break;
        case DType::Float16:
        case DType::BFloat16: {
            auto input_f32 = input.to(DType::Float32);
            auto result = median_kernel(input_f32, dim, keepdim, stream);
            result[0] = result[0].to(dtype);
            HIP_CHECK(hipStreamSynchronize(stream));
            HIP_CHECK(hipFree(d_shape));
            HIP_CHECK(hipFree(d_strides));
            return result;
        }
        default: throw std::runtime_error("median_kernel: unsupported dtype");
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_shape));
    HIP_CHECK(hipFree(d_strides));

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in median_kernel: ") + hipGetErrorString(err));
    }

    return {values, indices};
}

// ============================================================================
// Mode kernel — sort-based, one block per slice
// ============================================================================

template <typename T>
__global__ void mode_per_slice_kernel(
    const T* __restrict__ input,
    T* __restrict__ out_values,
    int64_t* __restrict__ out_indices,
    const int64_t* __restrict__ shape,
    const int64_t* __restrict__ strides,
    int64_t ndim, int64_t dim, int64_t dim_size, int64_t outer_size)
{
    int64_t slice_idx = blockIdx.x;
    if (slice_idx >= outer_size) return;

    int64_t tmp = slice_idx;
    int64_t base_offset = 0;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) continue;
        int64_t coord = tmp % shape[d];
        tmp /= shape[d];
        base_offset += coord * strides[d];
    }

    if (threadIdx.x == 0) {
        // Collect (value, original_index) pairs, sort by value, find longest run
        // Use simple bubble sort for correctness (dim_size typically manageable)
        // For large dims, a more sophisticated approach would be needed

        // First pass: find mode by counting equal neighbors after sorting indices by value
        // We sort an index array by value using selection sort
        // (avoids dynamic allocation on device)

        T best_val = input[base_offset];
        int64_t best_idx = 0;
        int64_t best_count = 0;

        // For each unique value, count occurrences
        for (int64_t i = 0; i < dim_size; ++i) {
            T val_i = input[base_offset + i * strides[dim]];
            int64_t count = 0;
            for (int64_t j = 0; j < dim_size; ++j) {
                if (input[base_offset + j * strides[dim]] == val_i) {
                    count++;
                }
            }
            // Pick this value if count is higher, or same count but smaller index
            if (count > best_count || (count == best_count && i < best_idx)) {
                best_count = count;
                best_val = val_i;
                best_idx = i;
            }
        }

        out_values[slice_idx] = best_val;
        out_indices[slice_idx] = best_idx;
    }
}

auto mode_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream)
    -> std::vector<Tensor>
{
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    if (input.numel() == 0) {
        throw std::runtime_error("mode: cannot compute mode of empty tensor");
    }

    int64_t normalized_dim = dim;
    if (normalized_dim == INT64_MIN) {
        Tensor flat = input.contiguous().reshape({input.numel()});
        auto result = mode_kernel(flat, 0, false, stream);
        if (keepdim) {
            std::vector<int64_t> kshape(ndim, 1);
            result[0] = result[0].reshape(kshape);
            result[1] = result[1].reshape(kshape);
        }
        return result;
    }

    if (normalized_dim < 0) normalized_dim += ndim;

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        normalized_dim, keepdim);

    int64_t dim_size = input_shape[normalized_dim];
    int64_t outer_size = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != normalized_dim) outer_size *= input_shape[d];
    }

    Tensor values(output_shape, dtype, device);
    Tensor indices(output_shape, DType::Int64, device);

    int64_t* d_shape;
    int64_t* d_strides;
    HIP_CHECK(hipMalloc(&d_shape, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_strides, ndim * sizeof(int64_t)));

    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
    std::vector<int64_t> strides_vec(input.strides().begin(), input.strides().end());
    HIP_CHECK(hipMemcpy(d_shape, shape_vec.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_strides, strides_vec.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));

    int block_size = 1;
    int num_blocks = static_cast<int>(outer_size);

    auto launch = [&]<typename T>(T*) {
        hipLaunchKernelGGL((mode_per_slice_kernel<T>),
            dim3(num_blocks), dim3(block_size), 0, stream,
            input.data<T>(), values.data<T>(), indices.data<int64_t>(),
            d_shape, d_strides, ndim, normalized_dim, dim_size, outer_size);
    };

    switch (dtype) {
        case DType::Float32: launch(static_cast<float*>(nullptr)); break;
        case DType::Float64: launch(static_cast<double*>(nullptr)); break;
        case DType::Int32:   launch(static_cast<int32_t*>(nullptr)); break;
        case DType::Int64:   launch(static_cast<int64_t*>(nullptr)); break;
        case DType::Float16:
        case DType::BFloat16: {
            auto input_f32 = input.to(DType::Float32);
            auto result = mode_kernel(input_f32, dim, keepdim, stream);
            result[0] = result[0].to(dtype);
            HIP_CHECK(hipStreamSynchronize(stream));
            HIP_CHECK(hipFree(d_shape));
            HIP_CHECK(hipFree(d_strides));
            return result;
        }
        default: throw std::runtime_error("mode_kernel: unsupported dtype");
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_shape));
    HIP_CHECK(hipFree(d_strides));

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in mode_kernel: ") + hipGetErrorString(err));
    }

    return {values, indices};
}

// ============================================================================
// CountNonzero — full reduction kernel
// ============================================================================

__global__ void count_nonzero_all_f32(const float* __restrict__ input,
                                      int64_t* __restrict__ output, int64_t n) {
    __shared__ int64_t scount;
    if (threadIdx.x == 0) scount = 0;
    __syncthreads();

    int64_t local_count = 0;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += grid_size) {
        if (input[i] != 0.0f) local_count++;
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(&scount),
              static_cast<unsigned long long>(local_count));
    __syncthreads();
    if (threadIdx.x == 0) output[0] = scount;
}

auto count_nonzero_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    Tensor in = input;
    if (in.dtype() != DType::Float32) {
        in = in.to(DType::Float32);
    }
    int64_t n = in.numel();
    Tensor result({1}, DType::Int64, in.device());
    // Single block — shared memory reduction requires all threads in same block
    hipLaunchKernelGGL(count_nonzero_all_f32,
                       dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                       in.data<float>(), result.data<int64_t>(), n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    return result;
}

// ============================================================================
// CountNonzero — dim-specific reduction kernel (native HIP, no CPU fallback)
// ============================================================================

template<typename T>
__global__ void count_nonzero_along_dim_kernel(
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

    // Decompose output index into multi-dimensional indices
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

    // Count non-zero elements along the reduction dimension
    int64_t count = 0;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }
        if (static_cast<float>(input[in_idx]) != 0.0f) {
            count++;
        }
    }
    output[out_idx] = count;
}

template<typename T>
static void launch_dim_count_nonzero(
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
        if (i != dim) output_size *= input_shape[i];
    }
    if (output_size == 0 || dim_size == 0) return;

    // Copy shape and strides to device
    int64_t* d_shape;
    int64_t* d_strides;
    HIP_CHECK(hipMalloc(&d_shape, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_strides, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(d_shape, input_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_strides, input_strides.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));

    int num_blocks = (output_size + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    hipLaunchKernelGGL(count_nonzero_along_dim_kernel<T>, dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
        d_input, d_output, d_shape, d_strides, ndim, dim, output_size, dim_size);

    // Must wait for kernel to complete before freeing device memory it uses
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_shape));
    HIP_CHECK(hipFree(d_strides));
}

auto count_nonzero_dim_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor {
    Tensor in = input;
    const auto& input_shape = in.shape();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());
    int64_t normalized_dim = dim;
    if (dim < 0) normalized_dim = ndim + dim;

    // Compute output shape (remove reduced dim)
    std::vector<int64_t> output_shape;
    for (int64_t d = 0; d < ndim; d++) {
        if (d != normalized_dim) output_shape.push_back(input_shape[d]);
    }
    if (output_shape.empty()) output_shape.push_back(1);

    Tensor result(output_shape, DType::Int64, in.device());

    // Upcast to Float32 for comparison if needed
    if (in.dtype() != DType::Float32 && in.dtype() != DType::Float64 &&
        in.dtype() != DType::Int32 && in.dtype() != DType::Int64) {
        in = in.to(DType::Float32);
    }

    auto shape_vec = std::vector<int64_t>(in.shape().begin(), in.shape().end());
    auto strides_vec = std::vector<int64_t>(in.strides().begin(), in.strides().end());

    if (in.dtype() == DType::Float32) {
        launch_dim_count_nonzero(in.data<float>(), result.data<int64_t>(), shape_vec, strides_vec, normalized_dim, stream);
    } else if (in.dtype() == DType::Float64) {
        launch_dim_count_nonzero(in.data<double>(), result.data<int64_t>(), shape_vec, strides_vec, normalized_dim, stream);
    } else if (in.dtype() == DType::Int32) {
        launch_dim_count_nonzero(in.data<int32_t>(), result.data<int64_t>(), shape_vec, strides_vec, normalized_dim, stream);
    } else if (in.dtype() == DType::Int64) {
        launch_dim_count_nonzero(in.data<int64_t>(), result.data<int64_t>(), shape_vec, strides_vec, normalized_dim, stream);
    }
    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Nansum — full reduction kernel (sum skipping NaN)
// ============================================================================

__global__ void nansum_all_f32(const float* __restrict__ input,
                               float* __restrict__ output, int64_t n) {
    __shared__ float ssum;
    if (threadIdx.x == 0) ssum = 0.0f;
    __syncthreads();

    float local_sum = 0.0f;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += grid_size) {
        float v = input[i];
        if (!isnan(v)) local_sum += v;
    }
    atomicAdd(&ssum, local_sum);
    __syncthreads();
    if (threadIdx.x == 0) output[0] = ssum;
}

__global__ void nansum_all_f64(const double* __restrict__ input,
                               double* __restrict__ output, int64_t n) {
    __shared__ double ssum;
    if (threadIdx.x == 0) ssum = 0.0;
    __syncthreads();

    double local_sum = 0.0;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += grid_size) {
        double v = input[i];
        if (!isnan(v)) local_sum += v;
    }
    // HIP supports atomicAdd for double
    atomicAdd(&ssum, local_sum);
    __syncthreads();
    if (threadIdx.x == 0) output[0] = ssum;
}

auto nansum_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    DType orig_dtype = input.dtype();
    int64_t n = input.numel();

    if (orig_dtype == DType::Float64) {
        Tensor result({1}, DType::Float64, input.device());
        // Single block — shared memory reduction requires all threads in same block
        hipLaunchKernelGGL(nansum_all_f64,
                           dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                           input.data<double>(), result.data<double>(), n);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));
        return result;
    }

    // All other dtypes: upcast to Float32
    Tensor in = input;
    if (orig_dtype != DType::Float32) {
        in = in.to(DType::Float32);
    }
    Tensor result({1}, DType::Float32, in.device());
    // Single block — shared memory reduction requires all threads in same block
    hipLaunchKernelGGL(nansum_all_f32,
                       dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                       in.data<float>(), result.data<float>(), n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    return (orig_dtype != DType::Float32 && orig_dtype != DType::Float64)
               ? result.to(orig_dtype) : result;
}

// ============================================================================
// Nanmean — nansum / count_non_nan
// ============================================================================

__global__ void count_non_nan_all_f32(const float* __restrict__ input,
                                      int64_t* __restrict__ output, int64_t n) {
    __shared__ int64_t scount;
    if (threadIdx.x == 0) scount = 0;
    __syncthreads();

    int64_t local_count = 0;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += grid_size) {
        if (!isnan(input[i])) local_count++;
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(&scount),
              static_cast<unsigned long long>(local_count));
    __syncthreads();
    if (threadIdx.x == 0) output[0] = scount;
}

__global__ void nanmean_div_f32(const float* __restrict__ sum,
                                const int64_t* __restrict__ count,
                                float* __restrict__ output) {
    if (threadIdx.x == 0) {
        int64_t c = count[0];
        output[0] = (c > 0) ? sum[0] / static_cast<float>(c) : 0.0f;
    }
}

auto nanmean_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    DType orig_dtype = input.dtype();
    Tensor in = input;
    if (in.dtype() != DType::Float32) {
        in = in.to(DType::Float32);
    }
    int64_t n = in.numel();

    // Compute nansum on GPU (single block)
    Tensor sum_result({1}, DType::Float32, in.device());
    hipLaunchKernelGGL(nansum_all_f32,
                       dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                       in.data<float>(), sum_result.data<float>(), n);

    // Count non-NaN elements on GPU (single block)
    Tensor count_result({1}, DType::Int64, in.device());
    hipLaunchKernelGGL(count_non_nan_all_f32,
                       dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                       in.data<float>(), count_result.data<int64_t>(), n);

    // Divide sum by count on GPU
    Tensor result({1}, DType::Float32, in.device());
    hipLaunchKernelGGL(nanmean_div_f32,
                       dim3(1), dim3(1), 0, stream,
                       sum_result.data<float>(), count_result.data<int64_t>(),
                       result.data<float>());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));

    return (orig_dtype != DType::Float32) ? result.to(orig_dtype) : result;
}

// ============================================================================
// Aminmax — compute min and max in a single pass
// ============================================================================

__global__ void aminmax_all_f32(const float* __restrict__ input,
                                float* __restrict__ out_min,
                                float* __restrict__ out_max,
                                int64_t n) {
    __shared__ float smin[REDUCTION_BLOCK_SIZE];
    __shared__ float smax[REDUCTION_BLOCK_SIZE];

    float local_min = __int_as_float(0x7f800000);   // +Inf
    float local_max = __int_as_float(0xff800000);   // -Inf

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += grid_size) {
        float v = input[i];
        if (v < local_min) local_min = v;
        if (v > local_max) local_max = v;
    }

    smin[threadIdx.x] = local_min;
    smax[threadIdx.x] = local_max;
    __syncthreads();

    // Block-level tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            if (smin[threadIdx.x + stride] < smin[threadIdx.x])
                smin[threadIdx.x] = smin[threadIdx.x + stride];
            if (smax[threadIdx.x + stride] > smax[threadIdx.x])
                smax[threadIdx.x] = smax[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_min[0] = smin[0];
        out_max[0] = smax[0];
    }
}

__global__ void aminmax_all_f64(const double* __restrict__ input,
                                double* __restrict__ out_min,
                                double* __restrict__ out_max,
                                int64_t n) {
    __shared__ double smin[REDUCTION_BLOCK_SIZE];
    __shared__ double smax[REDUCTION_BLOCK_SIZE];

    double local_min = __longlong_as_double(0x7ff0000000000000LL);   // +Inf
    double local_max = __longlong_as_double(0xfff0000000000000LL);   // -Inf

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += grid_size) {
        double v = input[i];
        if (v < local_min) local_min = v;
        if (v > local_max) local_max = v;
    }

    smin[threadIdx.x] = local_min;
    smax[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            if (smin[threadIdx.x + stride] < smin[threadIdx.x])
                smin[threadIdx.x] = smin[threadIdx.x + stride];
            if (smax[threadIdx.x + stride] > smax[threadIdx.x])
                smax[threadIdx.x] = smax[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_min[0] = smin[0];
        out_max[0] = smax[0];
    }
}

auto aminmax_kernel(const Tensor& input, hipStream_t stream)
    -> std::pair<Tensor, Tensor> {
    DType orig_dtype = input.dtype();
    int64_t n = input.numel();
    int num_blocks = 1; // Single block — shared-memory reduction

    if (orig_dtype == DType::Float64) {
        Tensor min_result({1}, DType::Float64, input.device());
        Tensor max_result({1}, DType::Float64, input.device());
        hipLaunchKernelGGL(aminmax_all_f64,
                           dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                           input.data<double>(), min_result.data<double>(),
                           max_result.data<double>(), n);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));
        return {min_result, max_result};
    }

    // All other dtypes: upcast to Float32
    Tensor in = input;
    if (orig_dtype != DType::Float32) {
        in = in.to(DType::Float32);
    }
    Tensor min_result({1}, DType::Float32, in.device());
    Tensor max_result({1}, DType::Float32, in.device());
    hipLaunchKernelGGL(aminmax_all_f32,
                       dim3(num_blocks), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                       in.data<float>(), min_result.data<float>(),
                       max_result.data<float>(), n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));

    if (orig_dtype != DType::Float32 && orig_dtype != DType::Float64) {
        min_result = min_result.to(orig_dtype);
        max_result = max_result.to(orig_dtype);
    }
    return {min_result, max_result};
}

// ============================================================================
// NanVar — NaN-ignoring variance (full reduction)
// ============================================================================

__global__ void nanvar_all_f32(const float* __restrict__ input,
                               const float* __restrict__ mean,
                               float* __restrict__ output,
                               int64_t* __restrict__ count_out,
                               int64_t n) {
    __shared__ float ssum;
    __shared__ int64_t scount;
    if (threadIdx.x == 0) { ssum = 0.0f; scount = 0; }
    __syncthreads();

    float local_sum = 0.0f;
    int64_t local_count = 0;
    float m = mean[0];
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += grid_size) {
        float v = input[i];
        if (!isnan(v)) {
            float diff = v - m;
            local_sum += diff * diff;
            local_count++;
        }
    }
    atomicAdd(&ssum, local_sum);
    atomicAdd(reinterpret_cast<unsigned long long*>(&scount),
              static_cast<unsigned long long>(local_count));
    __syncthreads();
    if (threadIdx.x == 0) {
        output[0] = ssum;
        count_out[0] = scount;
    }
}

__global__ void nanvar_finalize_f32(const float* __restrict__ sum_sq,
                                    const int64_t* __restrict__ count,
                                    float* __restrict__ output,
                                    int64_t correction) {
    if (threadIdx.x == 0) {
        int64_t c = count[0];
        int64_t denom = c - correction;
        output[0] = (denom > 0) ? sum_sq[0] / static_cast<float>(denom) : 0.0f;
    }
}

auto nanvar_kernel(const Tensor& input, bool unbiased, hipStream_t stream) -> Tensor {
    DType orig_dtype = input.dtype();
    Tensor in = input;
    if (in.dtype() != DType::Float32) {
        in = in.to(DType::Float32);
    }
    int64_t n = in.numel();

    // First compute nanmean
    Tensor sum_result({1}, DType::Float32, in.device());
    hipLaunchKernelGGL(nansum_all_f32,
                       dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                       in.data<float>(), sum_result.data<float>(), n);

    Tensor count_result({1}, DType::Int64, in.device());
    hipLaunchKernelGGL(count_non_nan_all_f32,
                       dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                       in.data<float>(), count_result.data<int64_t>(), n);

    Tensor mean_result({1}, DType::Float32, in.device());
    hipLaunchKernelGGL(nanmean_div_f32,
                       dim3(1), dim3(1), 0, stream,
                       sum_result.data<float>(), count_result.data<int64_t>(),
                       mean_result.data<float>());

    // Now compute sum of squared deviations from mean, ignoring NaN
    Tensor sum_sq({1}, DType::Float32, in.device());
    Tensor count2({1}, DType::Int64, in.device());
    hipLaunchKernelGGL(nanvar_all_f32,
                       dim3(1), dim3(REDUCTION_BLOCK_SIZE), 0, stream,
                       in.data<float>(), mean_result.data<float>(),
                       sum_sq.data<float>(), count2.data<int64_t>(), n);

    // Finalize: sum_sq / (count - correction)
    int64_t correction = unbiased ? 1 : 0;
    Tensor result({1}, DType::Float32, in.device());
    hipLaunchKernelGGL(nanvar_finalize_f32,
                       dim3(1), dim3(1), 0, stream,
                       sum_sq.data<float>(), count2.data<int64_t>(),
                       result.data<float>(), correction);

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    return (orig_dtype != DType::Float32) ? result.to(orig_dtype) : result;
}

// ============================================================================
// NanStd — sqrt(NanVar)
// ============================================================================

__global__ void sqrt_scalar_f32(const float* input, float* output) {
    if (threadIdx.x == 0) {
        output[0] = sqrtf(input[0]);
    }
}

auto nanstd_kernel(const Tensor& input, bool unbiased, hipStream_t stream) -> Tensor {
    DType orig_dtype = input.dtype();
    Tensor var_result = nanvar_kernel(input, unbiased, stream);
    // var_result is Float32

    Tensor result({1}, DType::Float32, var_result.device());
    hipLaunchKernelGGL(sqrt_scalar_f32,
                       dim3(1), dim3(1), 0, stream,
                       var_result.data<float>(), result.data<float>());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    return (orig_dtype != DType::Float32) ? result.to(orig_dtype) : result;
}

} // namespace rocm
} // namespace tenzor
