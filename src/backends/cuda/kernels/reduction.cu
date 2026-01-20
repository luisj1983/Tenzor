#include "tenzor/core/tensor.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <stdexcept>

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

template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = (val > other) ? val : other;
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
        // Synchronize to ensure kernel completes
        cudaStreamSynchronize(stream);
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

        // Synchronize before freeing temp buffer
        cudaStreamSynchronize(stream);
        cudaFree(d_temp);
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
        // Synchronize to ensure kernel completes
        cudaStreamSynchronize(stream);
    } else {
        T* d_temp;
        cudaMalloc(&d_temp, num_blocks * sizeof(T));
        max_reduce_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        max_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);

        cudaStreamSynchronize(stream);
        cudaFree(d_temp);
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
        cudaStreamSynchronize(stream);
    } else {
        T* d_temp;
        cudaMalloc(&d_temp, num_blocks * sizeof(T));
        min_reduce_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        min_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);

        cudaStreamSynchronize(stream);
        cudaFree(d_temp);
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

    // Wait for kernel to complete before freeing memory
    cudaDeviceSynchronize();

    cudaFree(d_shape);
    cudaFree(d_strides);
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

    // Wait for kernel to complete before freeing memory
    cudaDeviceSynchronize();

    cudaFree(d_shape);
    cudaFree(d_strides);
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

    // Wait for kernel to complete before freeing memory
    cudaDeviceSynchronize();

    cudaFree(d_shape);
    cudaFree(d_strides);
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
        throw std::runtime_error("mean: cannot compute mean of empty tensor");
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

    // Additional device-wide synchronization to ensure all operations complete
    cudaDeviceSynchronize();

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
        cudaStreamSynchronize(stream);
    } else {
        // Two-phase reduction
        int64_t* d_temp;
        cudaMalloc(&d_temp, num_blocks * sizeof(int64_t));
        argmax_full_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);

        // Second pass: find argmax of argmaxes
        // Copy intermediate values and indices to host for final reduction
        std::vector<int64_t> temp_idxs(num_blocks);
        cudaMemcpyAsync(temp_idxs.data(), d_temp, num_blocks * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Find the true maximum among the block results
        T max_val = d_input[temp_idxs[0]];
        int64_t max_idx = temp_idxs[0];

        std::vector<T> temp_vals(num_blocks);
        for (int i = 0; i < num_blocks; i++) {
            cudaMemcpyAsync(&temp_vals[i], &d_input[temp_idxs[i]], sizeof(T), cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);

        for (int i = 1; i < num_blocks; i++) {
            if (temp_vals[i] > max_val) {
                max_val = temp_vals[i];
                max_idx = temp_idxs[i];
            }
        }

        cudaMemcpyAsync(d_output, &max_idx, sizeof(int64_t), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
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
        cudaStreamSynchronize(stream);
    } else {
        // Two-phase reduction
        int64_t* d_temp;
        cudaMalloc(&d_temp, num_blocks * sizeof(int64_t));
        argmin_full_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);

        // Second pass: find argmin of argmins
        std::vector<int64_t> temp_idxs(num_blocks);
        cudaMemcpyAsync(temp_idxs.data(), d_temp, num_blocks * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Find the true minimum among the block results
        T min_val = d_input[temp_idxs[0]];
        int64_t min_idx = temp_idxs[0];

        std::vector<T> temp_vals(num_blocks);
        for (int i = 0; i < num_blocks; i++) {
            cudaMemcpyAsync(&temp_vals[i], &d_input[temp_idxs[i]], sizeof(T), cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);

        for (int i = 1; i < num_blocks; i++) {
            if (temp_vals[i] < min_val) {
                min_val = temp_vals[i];
                min_idx = temp_idxs[i];
            }
        }

        cudaMemcpyAsync(d_output, &min_idx, sizeof(int64_t), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
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

    cudaDeviceSynchronize();

    cudaFree(d_shape);
    cudaFree(d_strides);
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

    cudaDeviceSynchronize();

    cudaFree(d_shape);
    cudaFree(d_strides);
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
        default:
            throw std::runtime_error("argmax: only Float32, Float64, Int32, and Int64 are supported");
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
        default:
            throw std::runtime_error("argmin: only Float32, Float64, Int32, and Int64 are supported");
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
        cudaStreamSynchronize(stream);
    } else {
        // Phase 1: Reduce to num_blocks intermediate results
        T* d_temp;
        cudaMalloc(&d_temp, num_blocks * sizeof(T));
        prod_reduce_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);

        // Phase 2: Final reduction
        prod_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);

        cudaStreamSynchronize(stream);
        cudaFree(d_temp);
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

    cudaDeviceSynchronize();

    cudaFree(d_shape);
    cudaFree(d_strides);
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
// Variance and Standard Deviation operations
// ============================================================================

// Variance kernel (two-pass algorithm for numerical stability)
template<typename T>
__global__ void variance_kernel(const T* input, T mean, T* output, int64_t n, int64_t correction) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Compute sum of squared differences
    T thread_sum = 0;
    for (int64_t i = idx; i < n; i += grid_size) {
        T diff = input[i] - mean;
        thread_sum += diff * diff;
    }

    shared[tid] = thread_sum;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // Warp-level reduction
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

// Helper function to compute variance
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

    // Step 1: Compute mean
    T* d_mean_temp;
    cudaMalloc(&d_mean_temp, sizeof(T));
    launch_full_reduction_sum(d_input, d_mean_temp, n, stream);

    T mean;
    cudaMemcpyAsync(&mean, d_mean_temp, sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    mean /= static_cast<T>(n);

    // Step 2: Compute sum of squared differences
    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        variance_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, mean, d_mean_temp, n, correction);
        cudaStreamSynchronize(stream);

        // Divide by (n - correction)
        T var_sum;
        cudaMemcpyAsync(&var_sum, d_mean_temp, sizeof(T), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        int64_t divisor = n - correction;
        if (divisor <= 0) divisor = 1;
        T variance = var_sum / static_cast<T>(divisor);

        cudaMemcpyAsync(d_output, &variance, sizeof(T), cudaMemcpyHostToDevice, stream);
    } else {
        // Two-phase reduction
        T* d_temp;
        cudaMalloc(&d_temp, num_blocks * sizeof(T));

        variance_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, mean, d_temp, n, correction);

        // Second phase: sum the partial results
        sum_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_mean_temp, num_blocks);
        cudaStreamSynchronize(stream);

        // Divide by (n - correction)
        T var_sum;
        cudaMemcpyAsync(&var_sum, d_mean_temp, sizeof(T), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        int64_t divisor = n - correction;
        if (divisor <= 0) divisor = 1;
        T variance = var_sum / static_cast<T>(divisor);

        cudaMemcpyAsync(d_output, &variance, sizeof(T), cudaMemcpyHostToDevice, stream);
        cudaFree(d_temp);
    }

    cudaFree(d_mean_temp);
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

                // Take p-th root on host
                cudaMemcpyAsync(&result, d_temp, sizeof(float), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                result = std::pow(result, 1.0f / p);
                cudaMemcpyAsync(output_data, &result, sizeof(float), cudaMemcpyHostToDevice, stream);
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

                // Take p-th root on host
                cudaMemcpyAsync(&result, d_temp, sizeof(double), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                result = std::pow(result, 1.0 / p);
                cudaMemcpyAsync(output_data, &result, sizeof(double), cudaMemcpyHostToDevice, stream);
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

} // namespace cuda
} // namespace tenzor
