#include "tenzor/core/tensor.hpp"
#include <cuda_runtime.h>
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
    T thread_sum = 0;
    for (int64_t i = idx; i < n; i += grid_size) {
        thread_sum += input[i];
    }

    shared[tid] = thread_sum;
    __syncthreads();

    // Block-level reduction in shared memory
    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
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
    T sum = 0;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;

        // Compute flat index
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }

        sum += input[in_idx];
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
        // Keep empty shape for scalar result
    }
    return output_shape;
}

template<typename T>
static void launch_full_reduction_sum(const T* d_input, T* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        T zero = 0;
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
    } else {
        // Phase 1: Reduce to num_blocks intermediate results
        T* d_temp;
        cudaMalloc(&d_temp, num_blocks * sizeof(T));
        sum_reduce_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);

        // Phase 2: Final reduction
        sum_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);

        // Synchronize before freeing temp buffer
        cudaStreamSynchronize(stream);
        cudaFree(d_temp);
    }
    cudaStreamSynchronize(stream);
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

        cudaStreamSynchronize(stream);
        cudaFree(d_temp);
    }
    cudaStreamSynchronize(stream);
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

        cudaStreamSynchronize(stream);
        cudaFree(d_temp);
    }
    cudaStreamSynchronize(stream);
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
                    dim
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
                    dim
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
                    dim
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
                    dim
                );
            }
            break;
        }
        default:
            throw std::runtime_error("sum: unsupported dtype");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in sum_kernel: ") + cudaGetErrorString(err));
    }

    return output;
}

auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
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

    // Divide by count (in-place)
    const int64_t n = sum_result.numel();

    if (dtype == DType::Float32) {
        auto* data = sum_result.data<float>();
        const float scale = 1.0f / static_cast<float>(count);

        // Simple scaling kernel
        int num_blocks = (n + 255) / 256;
        auto scale_kernel = [scale] __device__ (float* d, int64_t n) {
            int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                d[idx] *= scale;
            }
        };

        // Launch inline lambda kernel (requires C++17)
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;

        // Simple element-wise scaling
        auto scale_fn = [] __device__ (float val, float scale) { return val * scale; };

        // For simplicity, use a host-side loop with async copies for small tensors
        // For production, implement a proper scaling kernel
        std::vector<float> host_data(n);
        cudaMemcpy(host_data.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost);
        for (int64_t i = 0; i < n; i++) {
            host_data[i] *= scale;
        }
        cudaMemcpy(data, host_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    } else {  // Float64
        auto* data = sum_result.data<double>();
        const double scale = 1.0 / static_cast<double>(count);

        std::vector<double> host_data(n);
        cudaMemcpy(host_data.data(), data, n * sizeof(double), cudaMemcpyDeviceToHost);
        for (int64_t i = 0; i < n; i++) {
            host_data[i] *= scale;
        }
        cudaMemcpy(data, host_data.data(), n * sizeof(double), cudaMemcpyHostToDevice);
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
        default:
            throw std::runtime_error("max: unsupported dtype");
    }

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

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in min_kernel: ") + cudaGetErrorString(err));
    }

    return output;
}

} // namespace cuda
} // namespace tenzor
