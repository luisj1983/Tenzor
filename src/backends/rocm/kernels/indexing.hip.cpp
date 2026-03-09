#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <hipcub/hipcub.hpp>
#include <thrust/counting_iterator.h>
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace rocm {

// HIP Error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            throw std::runtime_error( \
                std::string("HIP error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + hipGetErrorString(err) \
            ); \
        } \
    } while(0)

// Grid-stride loop for HIP kernels
#define HIP_KERNEL_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

// Helper for atomic operations - template to use built-in atomicAdd where available
template<typename T>
__device__ __forceinline__ T atomicAddHelper(T* address, T val) {
    return atomicAdd(address, val);  // Use HIP built-in for float, double, int32, etc.
}

// Specialization for int64_t (not natively supported in HIP)
template<>
__device__ __forceinline__ int64_t atomicAddHelper<int64_t>(int64_t* address, int64_t val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, assumed + val);
    } while (assumed != old);
    return old;
}

// Specialization for __half (half precision float)
template<>
__device__ __forceinline__ __half atomicAddHelper<__half>(__half* address, __half val) {
    // Use CAS-based atomic add for half precision
    unsigned short int* address_as_ushort = (unsigned short int*)address;
    unsigned short int old = *address_as_ushort, assumed;
    do {
        assumed = old;
        __half assumed_half = *reinterpret_cast<__half*>(&assumed);
        __half new_val = __hadd(assumed_half, val);
        unsigned short int new_bits = *reinterpret_cast<unsigned short int*>(&new_val);
        old = atomicCAS(address_as_ushort, assumed, new_bits);
    } while (assumed != old);
    return *reinterpret_cast<__half*>(&old);
}

// ==============================================================================
// Gather Operation
// ==============================================================================

template<typename T>
__global__ void gather_kernel(
    const T* input,
    const int64_t* indices,
    T* output,
    int64_t input_size,
    int64_t indices_size,
    int64_t inner_size,
    int64_t dim_size
) {
    int64_t total_elements = indices_size * inner_size;

    HIP_KERNEL_LOOP(idx, total_elements) {
        int64_t inner_idx = idx % inner_size;
        int64_t indices_idx = idx / inner_size;

        int64_t index = indices[indices_idx];

        // Handle negative indices
        if (index < 0) {
            index += dim_size;
        }

        // Bounds checking
        if (index >= 0 && index < dim_size) {
            int64_t input_idx = indices_idx / indices_size * dim_size * inner_size +
                               index * inner_size + inner_idx;
            output[idx] = input[input_idx];
        }
    }
}

auto gather_hip(
    const Tensor& input,
    int64_t dim,
    const Tensor& indices
) -> Tensor {

    auto input_shape = input.shape();
    auto indices_shape = indices.shape();

    // Compute output shape
    std::vector<int64_t> output_shape;
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (static_cast<int64_t>(i) == dim) {
            output_shape.push_back(indices_shape[i]);
        } else {
            output_shape.push_back(input_shape[i]);
        }
    }

    Tensor output = Tensor(output_shape, input.dtype(), input.device());

    // Calculate dimensions
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input_shape[i];
    }

    int64_t dim_size = input_shape[dim];

    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < input_shape.size(); ++i) {
        inner_size *= input_shape[i];
    }

    int64_t indices_size = indices.numel();
    int64_t total_elements = indices_size * inner_size;

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(gather_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<float>(),
            indices.data<int64_t>(),
            output.data<float>(),
            input.numel(),
            indices_size,
            inner_size,
            dim_size
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(gather_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<double>(),
            indices.data<int64_t>(),
            output.data<double>(),
            input.numel(),
            indices_size,
            inner_size,
            dim_size
        );
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(gather_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<int32_t>(),
            indices.data<int64_t>(),
            output.data<int32_t>(),
            input.numel(),
            indices_size,
            inner_size,
            dim_size
        );
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(gather_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<int64_t>(),
            indices.data<int64_t>(),
            output.data<int64_t>(),
            input.numel(),
            indices_size,
            inner_size,
            dim_size
        );
    } else {
        throw std::runtime_error("gather_hip: Unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    return output;
}

// ==============================================================================
// Scatter Operation
// ==============================================================================

template<typename T>
__global__ void scatter_kernel(
    T* output,
    const int64_t* indices,
    const T* src,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size,
    int64_t index_dim_size,
    int64_t total_scatter,
    bool reduce_add
) {
    HIP_KERNEL_LOOP(idx, total_scatter) {
        int64_t inner_idx = idx % inner_size;
        int64_t temp = idx / inner_size;
        int64_t index_pos = temp % index_dim_size;
        int64_t outer_idx = temp / index_dim_size;

        // Get the index value at this position
        int64_t index_offset = outer_idx * index_dim_size * inner_size +
                               index_pos * inner_size + inner_idx;
        int64_t scatter_idx = indices[index_offset];

        // Handle negative indices
        if (scatter_idx < 0) {
            scatter_idx += dim_size;
        }

        // Bounds checking
        if (scatter_idx >= 0 && scatter_idx < dim_size) {
            // Compute output offset
            int64_t output_offset = outer_idx * dim_size * inner_size +
                                    scatter_idx * inner_size +
                                    inner_idx;

            if (reduce_add) {
                atomicAddHelper(&output[output_offset], src[idx]);
            } else {
                output[output_offset] = src[idx];
            }
        }
    }
}

auto scatter_hip(
    Tensor& output,
    int64_t dim,
    const Tensor& indices,
    const Tensor& src,
    const std::string& reduce
) -> Tensor {

    auto output_shape = output.shape();
    auto indices_shape = indices.shape();

    // Normalize dimension
    int64_t ndim = output.ndim();
    if (dim < 0) dim += ndim;

    // Calculate dimensions
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= output_shape[i];
    }

    int64_t dim_size = output_shape[dim];
    int64_t index_dim_size = indices_shape[dim];

    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < output_shape.size(); ++i) {
        inner_size *= output_shape[i];
    }

    int64_t total_scatter = indices.numel();

    bool reduce_add = (reduce == "add");

    int threads = 256;
    int blocks = (total_scatter + threads - 1) / threads;

    if (output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(scatter_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<float>(),
            indices.data<int64_t>(),
            src.data<float>(),
            outer_size,
            dim_size,
            inner_size,
            index_dim_size,
            total_scatter,
            reduce_add
        );
    } else if (output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(scatter_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<double>(),
            indices.data<int64_t>(),
            src.data<double>(),
            outer_size,
            dim_size,
            inner_size,
            index_dim_size,
            total_scatter,
            reduce_add
        );
    } else if (output.dtype() == DType::Float16) {
        hipLaunchKernelGGL(scatter_kernel<__half>,
            dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<__half*>(output.data<Float16>()),
            indices.data<int64_t>(),
            reinterpret_cast<const __half*>(src.data<Float16>()),
            outer_size,
            dim_size,
            inner_size,
            index_dim_size,
            total_scatter,
            reduce_add
        );
    } else if (output.dtype() == DType::Int32) {
        hipLaunchKernelGGL(scatter_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<int32_t>(),
            indices.data<int64_t>(),
            src.data<int32_t>(),
            outer_size,
            dim_size,
            inner_size,
            index_dim_size,
            total_scatter,
            reduce_add
        );
    } else if (output.dtype() == DType::Int64) {
        hipLaunchKernelGGL(scatter_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<int64_t>(),
            indices.data<int64_t>(),
            src.data<int64_t>(),
            outer_size,
            dim_size,
            inner_size,
            index_dim_size,
            total_scatter,
            reduce_add
        );
    } else {
        throw std::runtime_error("scatter_hip: Unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    return output;
}

// ==============================================================================
// Index Select Operation
// ==============================================================================

template<typename T>
__global__ void index_select_kernel(
    const T* input,
    const int64_t* indices,
    T* output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size,
    int64_t num_indices
) {
    int64_t total_elements = outer_size * num_indices * inner_size;

    HIP_KERNEL_LOOP(idx, total_elements) {
        int64_t inner_idx = idx % inner_size;
        int64_t index_idx = (idx / inner_size) % num_indices;
        int64_t outer_idx = idx / (inner_size * num_indices);

        int64_t selected_idx = indices[index_idx];

        // Handle negative indices
        if (selected_idx < 0) {
            selected_idx += dim_size;
        }

        // Bounds checking
        if (selected_idx >= 0 && selected_idx < dim_size) {
            int64_t input_idx = (outer_idx * dim_size + selected_idx) * inner_size + inner_idx;
            output[idx] = input[input_idx];
        }
    }
}

auto index_select_hip(
    const Tensor& input,
    int64_t dim,
    const Tensor& indices
) -> Tensor {

    auto input_shape = input.shape();

    // Compute output shape
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    output_shape[dim] = indices.numel();

    Tensor output = Tensor(output_shape, input.dtype(), input.device());

    // Calculate dimensions
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input_shape[i];
    }

    int64_t dim_size = input_shape[dim];

    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < input_shape.size(); ++i) {
        inner_size *= input_shape[i];
    }

    int64_t num_indices = indices.numel();
    int64_t total_elements = outer_size * num_indices * inner_size;

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(index_select_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<float>(),
            indices.data<int64_t>(),
            output.data<float>(),
            outer_size,
            dim_size,
            inner_size,
            num_indices
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(index_select_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<double>(),
            indices.data<int64_t>(),
            output.data<double>(),
            outer_size,
            dim_size,
            inner_size,
            num_indices
        );
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(index_select_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<int32_t>(),
            indices.data<int64_t>(),
            output.data<int32_t>(),
            outer_size,
            dim_size,
            inner_size,
            num_indices
        );
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(index_select_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<int64_t>(),
            indices.data<int64_t>(),
            output.data<int64_t>(),
            outer_size,
            dim_size,
            inner_size,
            num_indices
        );
    } else {
        throw std::runtime_error("index_select_hip: Unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    return output;
}

// ==============================================================================
// Masked Fill Operation
// ==============================================================================

template<typename T>
__global__ void masked_fill_kernel(
    T* output,
    const bool* mask,
    T value,
    int64_t total_elements
) {
    HIP_KERNEL_LOOP(idx, total_elements) {
        if (mask[idx]) {
            output[idx] = value;
        }
    }
}

auto masked_fill_hip(
    Tensor& input,
    const Tensor& mask,
    float value
) -> Tensor {

    Tensor output = input.clone();

    int64_t total_elements = output.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(masked_fill_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<float>(),
            mask.data<bool>(),
            static_cast<float>(value),
            total_elements
        );
    } else if (output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(masked_fill_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<double>(),
            mask.data<bool>(),
            static_cast<double>(value),
            total_elements
        );
    } else if (output.dtype() == DType::Int32) {
        hipLaunchKernelGGL(masked_fill_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<int32_t>(),
            mask.data<bool>(),
            static_cast<int32_t>(value),
            total_elements
        );
    } else if (output.dtype() == DType::Int64) {
        hipLaunchKernelGGL(masked_fill_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<int64_t>(),
            mask.data<bool>(),
            static_cast<int64_t>(value),
            total_elements
        );
    } else {
        throw std::runtime_error("masked_fill_hip: Unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    return output;
}

// ==============================================================================
// Masked Select Operation
// ==============================================================================

template<typename T>
__global__ void masked_select_count_kernel(
    const bool* mask,
    int64_t* count,
    int64_t total_elements
) {
    __shared__ int64_t shared_count[256];

    int tid = threadIdx.x;
    shared_count[tid] = 0;
    __syncthreads();

    HIP_KERNEL_LOOP(idx, total_elements) {
        if (mask[idx]) {
            atomicAddHelper(&shared_count[tid], static_cast<int64_t>(1));
        }
    }
    __syncthreads();

    // Reduce within block
    if (tid == 0) {
        int64_t block_count = 0;
        for (int i = 0; i < 256; ++i) {
            block_count += shared_count[i];
        }
        atomicAddHelper(count, block_count);
    }
}

template<typename T>
__global__ void masked_select_kernel(
    const T* input,
    const bool* mask,
    T* output,
    int64_t* output_idx,
    int64_t total_elements
) {
    HIP_KERNEL_LOOP(idx, total_elements) {
        if (mask[idx]) {
            int64_t out_idx = atomicAddHelper(output_idx, static_cast<int64_t>(1));
            output[out_idx] = input[idx];
        }
    }
}

auto masked_select_hip(
    const Tensor& input,
    const Tensor& mask
) -> Tensor {

    int64_t total_elements = input.numel();

    // First pass: count how many elements match the mask
    int64_t* d_count;
    HIP_CHECK(hipMalloc(&d_count, sizeof(int64_t)));
    HIP_CHECK(hipMemset(d_count, 0, sizeof(int64_t)));

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(masked_select_count_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            mask.data<bool>(),
            d_count,
            total_elements
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(masked_select_count_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            mask.data<bool>(),
            d_count,
            total_elements
        );
    } else {
        HIP_CHECK(hipFree(d_count));
        throw std::runtime_error("masked_select_hip: Only Float32 and Float64 supported");
    }

    // Get count from device
    int64_t h_count;
    HIP_CHECK(hipMemcpy(&h_count, d_count, sizeof(int64_t), hipMemcpyDeviceToHost));

    // Create output tensor
    Tensor output = Tensor({h_count}, input.dtype(), input.device());

    // Second pass: copy selected elements
    int64_t* d_output_idx;
    HIP_CHECK(hipMalloc(&d_output_idx, sizeof(int64_t)));
    HIP_CHECK(hipMemset(d_output_idx, 0, sizeof(int64_t)));

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(masked_select_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<float>(),
            mask.data<bool>(),
            output.data<float>(),
            d_output_idx,
            total_elements
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(masked_select_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<double>(),
            mask.data<bool>(),
            output.data<double>(),
            d_output_idx,
            total_elements
        );
    }

    HIP_CHECK(hipFree(d_count));
    HIP_CHECK(hipFree(d_output_idx));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    return output;
}

// ==============================================================================
// Take Operation (1D indexing)
// ==============================================================================

template<typename T>
__global__ void take_kernel(
    const T* input,
    const int64_t* indices,
    T* output,
    int64_t input_size,
    int64_t indices_size
) {
    HIP_KERNEL_LOOP(idx, indices_size) {
        int64_t index = indices[idx];

        // Handle negative indices
        if (index < 0) {
            index += input_size;
        }

        // Bounds checking
        if (index >= 0 && index < input_size) {
            output[idx] = input[index];
        }
    }
}

auto take_hip(
    const Tensor& input,
    const Tensor& indices
) -> Tensor {

    int64_t input_size = input.numel();
    int64_t indices_size = indices.numel();

    Tensor output = Tensor({indices_size}, input.dtype(), input.device());

    int threads = 256;
    int blocks = (indices_size + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(take_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<float>(),
            indices.data<int64_t>(),
            output.data<float>(),
            input_size,
            indices_size
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(take_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<double>(),
            indices.data<int64_t>(),
            output.data<double>(),
            input_size,
            indices_size
        );
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(take_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<int32_t>(),
            indices.data<int64_t>(),
            output.data<int32_t>(),
            input_size,
            indices_size
        );
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(take_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<int64_t>(),
            indices.data<int64_t>(),
            output.data<int64_t>(),
            input_size,
            indices_size
        );
    } else {
        throw std::runtime_error("take_hip: Unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    return output;
}

// ==============================================================================
// Where Operation
// ==============================================================================

template<typename T>
__global__ void where_kernel(
    const bool* condition,
    const T* x,
    const T* y,
    T* output,
    int64_t total_elements
) {
    HIP_KERNEL_LOOP(idx, total_elements) {
        output[idx] = condition[idx] ? x[idx] : y[idx];
    }
}

auto where_hip(
    const Tensor& condition,
    const Tensor& x,
    const Tensor& y,
    hipStream_t stream
) -> Tensor {
    if (!std::equal(x.shape().begin(), x.shape().end(), y.shape().begin(), y.shape().end())) {
        throw std::runtime_error("where_hip: x and y must have the same shape");
    }
    if (x.dtype() != y.dtype()) {
        throw std::runtime_error("where_hip: x and y must have the same dtype");
    }

    Tensor output = Tensor(std::vector<int64_t>(x.shape().begin(), x.shape().end()), x.dtype(), x.device());

    int64_t total_elements = x.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (x.dtype() == DType::Float32) {
        hipLaunchKernelGGL(where_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            condition.data<bool>(),
            x.data<float>(),
            y.data<float>(),
            output.data<float>(),
            total_elements
        );
    } else if (x.dtype() == DType::Float64) {
        hipLaunchKernelGGL(where_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            condition.data<bool>(),
            x.data<double>(),
            y.data<double>(),
            output.data<double>(),
            total_elements
        );
    } else if (x.dtype() == DType::Int32) {
        hipLaunchKernelGGL(where_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, stream,
            condition.data<bool>(),
            x.data<int32_t>(),
            y.data<int32_t>(),
            output.data<int32_t>(),
            total_elements
        );
    } else if (x.dtype() == DType::Int64) {
        hipLaunchKernelGGL(where_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            condition.data<bool>(),
            x.data<int64_t>(),
            y.data<int64_t>(),
            output.data<int64_t>(),
            total_elements
        );
    } else {
        throw std::runtime_error("where_hip: Unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

// ==============================================================================
// Slice Operation
// ==============================================================================

template<typename T>
__global__ void slice_kernel(
    const T* input,
    T* output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size,
    int64_t start,
    int64_t end,
    int64_t step,
    int64_t output_dim_size
) {
    int64_t total_elements = outer_size * output_dim_size * inner_size;

    HIP_KERNEL_LOOP(idx, total_elements) {
        int64_t inner_idx = idx % inner_size;
        int64_t out_dim_idx = (idx / inner_size) % output_dim_size;
        int64_t outer_idx = idx / (inner_size * output_dim_size);

        int64_t in_dim_idx = start + out_dim_idx * step;

        int64_t input_idx = (outer_idx * dim_size + in_dim_idx) * inner_size + inner_idx;
        output[idx] = input[input_idx];
    }
}

auto slice_hip(
    const Tensor& input,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step,
    hipStream_t stream
) -> Tensor {
    auto input_shape = input.shape();
    int64_t ndim = input_shape.size();

    // Handle negative indices
    if (dim < 0) dim += ndim;
    if (start < 0) start += input_shape[dim];
    if (end < 0) end += input_shape[dim];

    // Clamp indices
    start = std::max(int64_t(0), std::min(start, input_shape[dim]));
    end = std::max(int64_t(0), std::min(end, input_shape[dim]));

    // Compute output dim size
    int64_t output_dim_size = (end - start + step - 1) / step;
    if (output_dim_size < 0) output_dim_size = 0;

    // Compute output shape
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    output_shape[dim] = output_dim_size;

    Tensor output = Tensor(output_shape, input.dtype(), input.device());

    if (output.numel() == 0) return output;

    // Calculate dimensions
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input_shape[i];
    }

    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < input_shape.size(); ++i) {
        inner_size *= input_shape[i];
    }

    int64_t total_elements = outer_size * output_dim_size * inner_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(slice_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(),
            output.data<float>(),
            outer_size,
            input_shape[dim],
            inner_size,
            start,
            end,
            step,
            output_dim_size
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(slice_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(),
            output.data<double>(),
            outer_size,
            input_shape[dim],
            inner_size,
            start,
            end,
            step,
            output_dim_size
        );
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(slice_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<int32_t>(),
            output.data<int32_t>(),
            outer_size,
            input_shape[dim],
            inner_size,
            start,
            end,
            step,
            output_dim_size
        );
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(slice_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<int64_t>(),
            output.data<int64_t>(),
            outer_size,
            input_shape[dim],
            inner_size,
            start,
            end,
            step,
            output_dim_size
        );
    } else {
        throw std::runtime_error("slice_hip: Unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

// ==============================================================================
// Cat (Concatenate) Operation
// ==============================================================================

template<typename T>
__global__ void cat_kernel(
    const T* const* inputs,
    const int64_t* input_offsets,
    T* output,
    int64_t num_inputs,
    int64_t outer_size,
    int64_t total_dim_size,
    int64_t inner_size,
    const int64_t* dim_sizes
) {
    int64_t total_elements = outer_size * total_dim_size * inner_size;

    HIP_KERNEL_LOOP(idx, total_elements) {
        int64_t inner_idx = idx % inner_size;
        int64_t dim_idx = (idx / inner_size) % total_dim_size;
        int64_t outer_idx = idx / (inner_size * total_dim_size);

        // Find which input tensor this element belongs to
        int64_t input_idx = 0;
        int64_t local_dim_idx = dim_idx;
        for (int64_t i = 0; i < num_inputs; ++i) {
            if (local_dim_idx < dim_sizes[i]) {
                input_idx = i;
                break;
            }
            local_dim_idx -= dim_sizes[i];
        }

        int64_t src_idx = (outer_idx * dim_sizes[input_idx] + local_dim_idx) * inner_size + inner_idx;
        output[idx] = inputs[input_idx][src_idx];
    }
}

auto cat_hip(
    const std::vector<Tensor>& tensors,
    int64_t dim,
    hipStream_t stream
) -> Tensor {
    if (tensors.empty()) {
        throw std::runtime_error("cat_hip: tensors list cannot be empty");
    }

    // Ensure all tensors are contiguous - the kernel assumes contiguous memory layout
    // This is critical for sliced tensors (e.g., from roll operation) which may have
    // non-unit strides or non-zero offsets
    std::vector<Tensor> cont_tensors;
    cont_tensors.reserve(tensors.size());
    for (const auto& t : tensors) {
        cont_tensors.push_back(t.is_contiguous() ? t : t.contiguous());
    }

    auto& first = cont_tensors[0];
    auto first_shape = first.shape();
    int64_t ndim = first_shape.size();

    // Handle negative dim
    if (dim < 0) dim += ndim;

    // Validate shapes and compute total dim size
    int64_t total_dim_size = 0;
    std::vector<int64_t> dim_sizes;

    for (size_t i = 0; i < cont_tensors.size(); ++i) {
        auto shape = cont_tensors[i].shape();
        if (shape.size() != first_shape.size()) {
            throw std::runtime_error("cat_hip: all tensors must have the same number of dimensions");
        }
        for (size_t d = 0; d < shape.size(); ++d) {
            if (d != static_cast<size_t>(dim) && shape[d] != first_shape[d]) {
                throw std::runtime_error("cat_hip: all tensors must have the same shape except in the cat dimension");
            }
        }
        dim_sizes.push_back(shape[dim]);
        total_dim_size += shape[dim];
    }

    // Compute output shape
    std::vector<int64_t> output_shape(first_shape.begin(), first_shape.end());
    output_shape[dim] = total_dim_size;

    Tensor output = Tensor(output_shape, first.dtype(), first.device());

    if (output.numel() == 0) return output;

    // Calculate dimensions
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= first_shape[i];
    }

    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < first_shape.size(); ++i) {
        inner_size *= first_shape[i];
    }

    // Copy input pointers and dim sizes to device
    std::vector<const void*> h_input_ptrs(cont_tensors.size());
    for (size_t i = 0; i < cont_tensors.size(); ++i) {
        h_input_ptrs[i] = cont_tensors[i].data_ptr();
    }

    void** d_input_ptrs;
    int64_t* d_dim_sizes;
    int64_t* d_input_offsets;

    HIP_CHECK(hipMalloc(&d_input_ptrs, cont_tensors.size() * sizeof(void*)));
    HIP_CHECK(hipMalloc(&d_dim_sizes, cont_tensors.size() * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_input_offsets, cont_tensors.size() * sizeof(int64_t)));

    HIP_CHECK(hipMemcpy(d_input_ptrs, h_input_ptrs.data(), cont_tensors.size() * sizeof(void*), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dim_sizes, dim_sizes.data(), cont_tensors.size() * sizeof(int64_t), hipMemcpyHostToDevice));

    int64_t total_elements = outer_size * total_dim_size * inner_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (first.dtype() == DType::Float32) {
        hipLaunchKernelGGL(cat_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            (const float* const*)d_input_ptrs,
            d_input_offsets,
            output.data<float>(),
            static_cast<int64_t>(cont_tensors.size()),
            outer_size,
            total_dim_size,
            inner_size,
            d_dim_sizes
        );
    } else if (first.dtype() == DType::Float64) {
        hipLaunchKernelGGL(cat_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            (const double* const*)d_input_ptrs,
            d_input_offsets,
            output.data<double>(),
            static_cast<int64_t>(cont_tensors.size()),
            outer_size,
            total_dim_size,
            inner_size,
            d_dim_sizes
        );
    } else if (first.dtype() == DType::Int32) {
        hipLaunchKernelGGL(cat_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, stream,
            (const int32_t* const*)d_input_ptrs,
            d_input_offsets,
            output.data<int32_t>(),
            static_cast<int64_t>(cont_tensors.size()),
            outer_size,
            total_dim_size,
            inner_size,
            d_dim_sizes
        );
    } else if (first.dtype() == DType::Int64) {
        hipLaunchKernelGGL(cat_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            (const int64_t* const*)d_input_ptrs,
            d_input_offsets,
            output.data<int64_t>(),
            static_cast<int64_t>(cont_tensors.size()),
            outer_size,
            total_dim_size,
            inner_size,
            d_dim_sizes
        );
    } else if (first.dtype() == DType::Float16) {
        hipLaunchKernelGGL(cat_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            (const __half* const*)d_input_ptrs,
            d_input_offsets,
            reinterpret_cast<__half*>(output.data<Float16>()),
            static_cast<int64_t>(cont_tensors.size()),
            outer_size,
            total_dim_size,
            inner_size,
            d_dim_sizes
        );
    } else {
        HIP_CHECK(hipFree(d_input_ptrs));
        HIP_CHECK(hipFree(d_dim_sizes));
        HIP_CHECK(hipFree(d_input_offsets));
        throw std::runtime_error("cat_hip: Unsupported dtype");
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_input_ptrs));
    HIP_CHECK(hipFree(d_dim_sizes));
    HIP_CHECK(hipFree(d_input_offsets));
    HIP_CHECK(hipGetLastError());

    return output;
}

// ==============================================================================
// Put Operation
// ==============================================================================

template<typename T>
__global__ void put_kernel(
    T* output,
    const int64_t* indices,
    const T* source,
    int64_t num_indices,
    int64_t total_size,
    bool accumulate
) {
    HIP_KERNEL_LOOP(idx, num_indices) {
        int64_t target_idx = indices[idx];

        // Handle negative indices
        if (target_idx < 0) {
            target_idx += total_size;
        }

        // Bounds checking
        if (target_idx >= 0 && target_idx < total_size) {
            if (accumulate) {
                atomicAddHelper(&output[target_idx], source[idx]);
            } else {
                output[target_idx] = source[idx];
            }
        }
    }
}

auto put_hip(
    Tensor& input,
    const Tensor& indices,
    const Tensor& source,
    bool accumulate,
    hipStream_t stream
) -> Tensor {
    Tensor output = input.clone();

    int64_t num_indices = indices.numel();
    int64_t total_size = input.numel();

    int threads = 256;
    int blocks = (num_indices + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(put_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<float>(),
            indices.data<int64_t>(),
            source.data<float>(),
            num_indices,
            total_size,
            accumulate
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(put_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<double>(),
            indices.data<int64_t>(),
            source.data<double>(),
            num_indices,
            total_size,
            accumulate
        );
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(put_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<int32_t>(),
            indices.data<int64_t>(),
            source.data<int32_t>(),
            num_indices,
            total_size,
            accumulate
        );
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(put_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<int64_t>(),
            indices.data<int64_t>(),
            source.data<int64_t>(),
            num_indices,
            total_size,
            accumulate
        );
    } else {
        throw std::runtime_error("put_hip: Unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

// ==============================================================================
// Stream-aware wrapper functions for existing operations
// ==============================================================================

auto gather_hip(
    const Tensor& input,
    int64_t dim,
    const Tensor& indices,
    hipStream_t stream
) -> Tensor {
    auto input_shape = input.shape();
    auto indices_shape = indices.shape();

    std::vector<int64_t> output_shape;
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (static_cast<int64_t>(i) == dim) {
            output_shape.push_back(indices_shape[i]);
        } else {
            output_shape.push_back(input_shape[i]);
        }
    }

    Tensor output = Tensor(output_shape, input.dtype(), input.device());

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input_shape[i];
    }

    int64_t dim_size = input_shape[dim];

    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < input_shape.size(); ++i) {
        inner_size *= input_shape[i];
    }

    int64_t indices_size = indices.numel();
    int64_t total_elements = indices_size * inner_size;

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(gather_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(),
            indices.data<int64_t>(),
            output.data<float>(),
            input.numel(),
            indices_size,
            inner_size,
            dim_size
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(gather_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(),
            indices.data<int64_t>(),
            output.data<double>(),
            input.numel(),
            indices_size,
            inner_size,
            dim_size
        );
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(gather_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<int32_t>(),
            indices.data<int64_t>(),
            output.data<int32_t>(),
            input.numel(),
            indices_size,
            inner_size,
            dim_size
        );
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(gather_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<int64_t>(),
            indices.data<int64_t>(),
            output.data<int64_t>(),
            input.numel(),
            indices_size,
            inner_size,
            dim_size
        );
    } else {
        throw std::runtime_error("gather_hip: Unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

auto scatter_hip(
    const Tensor& input,
    int64_t dim,
    const Tensor& indices,
    const Tensor& src,
    hipStream_t stream
) -> Tensor {
    Tensor output = input.clone();
    auto output_shape = output.shape();
    auto indices_shape = indices.shape();

    // Normalize dimension
    int64_t ndim = output.ndim();
    if (dim < 0) dim += ndim;

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= output_shape[i];
    }

    int64_t dim_size = output_shape[dim];
    int64_t index_dim_size = indices_shape[dim];

    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < output_shape.size(); ++i) {
        inner_size *= output_shape[i];
    }

    int64_t total_scatter = indices.numel();

    int threads = 256;
    int blocks = (total_scatter + threads - 1) / threads;

    if (output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(scatter_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<float>(),
            indices.data<int64_t>(),
            src.data<float>(),
            outer_size,
            dim_size,
            inner_size,
            index_dim_size,
            total_scatter,
            false
        );
    } else if (output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(scatter_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<double>(),
            indices.data<int64_t>(),
            src.data<double>(),
            outer_size,
            dim_size,
            inner_size,
            index_dim_size,
            total_scatter,
            false
        );
    } else if (output.dtype() == DType::Float16) {
        hipLaunchKernelGGL(scatter_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<__half*>(output.data<Float16>()),
            indices.data<int64_t>(),
            reinterpret_cast<const __half*>(src.data<Float16>()),
            outer_size,
            dim_size,
            inner_size,
            index_dim_size,
            total_scatter,
            false
        );
    } else if (output.dtype() == DType::Int32) {
        hipLaunchKernelGGL(scatter_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<int32_t>(),
            indices.data<int64_t>(),
            src.data<int32_t>(),
            outer_size,
            dim_size,
            inner_size,
            index_dim_size,
            total_scatter,
            false
        );
    } else if (output.dtype() == DType::Int64) {
        hipLaunchKernelGGL(scatter_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<int64_t>(),
            indices.data<int64_t>(),
            src.data<int64_t>(),
            outer_size,
            dim_size,
            inner_size,
            index_dim_size,
            total_scatter,
            false
        );
    } else {
        throw std::runtime_error("scatter_hip: Unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

auto index_select_hip(
    const Tensor& input,
    int64_t dim,
    const Tensor& indices,
    hipStream_t stream
) -> Tensor {
    auto input_shape = input.shape();

    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    output_shape[dim] = indices.numel();

    Tensor output = Tensor(output_shape, input.dtype(), input.device());

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input_shape[i];
    }

    int64_t dim_size = input_shape[dim];

    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < input_shape.size(); ++i) {
        inner_size *= input_shape[i];
    }

    int64_t num_indices = indices.numel();
    int64_t total_elements = outer_size * num_indices * inner_size;

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(index_select_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(),
            indices.data<int64_t>(),
            output.data<float>(),
            outer_size,
            dim_size,
            inner_size,
            num_indices
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(index_select_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(),
            indices.data<int64_t>(),
            output.data<double>(),
            outer_size,
            dim_size,
            inner_size,
            num_indices
        );
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(index_select_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<int32_t>(),
            indices.data<int64_t>(),
            output.data<int32_t>(),
            outer_size,
            dim_size,
            inner_size,
            num_indices
        );
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(index_select_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<int64_t>(),
            indices.data<int64_t>(),
            output.data<int64_t>(),
            outer_size,
            dim_size,
            inner_size,
            num_indices
        );
    } else {
        throw std::runtime_error("index_select_hip: Unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

auto masked_fill_hip(
    const Tensor& input,
    const Tensor& mask,
    double value,
    hipStream_t stream
) -> Tensor {
    Tensor output = input.clone();

    int64_t total_elements = output.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(masked_fill_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<float>(),
            mask.data<bool>(),
            static_cast<float>(value),
            total_elements
        );
    } else if (output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(masked_fill_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<double>(),
            mask.data<bool>(),
            static_cast<double>(value),
            total_elements
        );
    } else if (output.dtype() == DType::Int32) {
        hipLaunchKernelGGL(masked_fill_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<int32_t>(),
            mask.data<bool>(),
            static_cast<int32_t>(value),
            total_elements
        );
    } else if (output.dtype() == DType::Int64) {
        hipLaunchKernelGGL(masked_fill_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<int64_t>(),
            mask.data<bool>(),
            static_cast<int64_t>(value),
            total_elements
        );
    } else {
        throw std::runtime_error("masked_fill_hip: Unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

auto masked_select_hip(
    const Tensor& input,
    const Tensor& mask,
    hipStream_t stream
) -> Tensor {
    int64_t total_elements = input.numel();

    int64_t* d_count;
    HIP_CHECK(hipMalloc(&d_count, sizeof(int64_t)));
    HIP_CHECK(hipMemset(d_count, 0, sizeof(int64_t)));

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(masked_select_count_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            mask.data<bool>(),
            d_count,
            total_elements
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(masked_select_count_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            mask.data<bool>(),
            d_count,
            total_elements
        );
    } else {
        HIP_CHECK(hipFree(d_count));
        throw std::runtime_error("masked_select_hip: Only Float32 and Float64 supported");
    }

    HIP_CHECK(hipStreamSynchronize(stream));

    int64_t h_count;
    HIP_CHECK(hipMemcpy(&h_count, d_count, sizeof(int64_t), hipMemcpyDeviceToHost));

    Tensor output = Tensor({h_count}, input.dtype(), input.device());

    int64_t* d_output_idx;
    HIP_CHECK(hipMalloc(&d_output_idx, sizeof(int64_t)));
    HIP_CHECK(hipMemset(d_output_idx, 0, sizeof(int64_t)));

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(masked_select_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(),
            mask.data<bool>(),
            output.data<float>(),
            d_output_idx,
            total_elements
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(masked_select_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(),
            mask.data<bool>(),
            output.data<double>(),
            d_output_idx,
            total_elements
        );
    }

    HIP_CHECK(hipFree(d_count));
    HIP_CHECK(hipFree(d_output_idx));
    HIP_CHECK(hipGetLastError());

    return output;
}

auto take_hip(
    const Tensor& input,
    const Tensor& indices,
    hipStream_t stream
) -> Tensor {
    int64_t input_size = input.numel();
    int64_t indices_size = indices.numel();

    Tensor output = Tensor({indices_size}, input.dtype(), input.device());

    int threads = 256;
    int blocks = (indices_size + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(take_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(),
            indices.data<int64_t>(),
            output.data<float>(),
            input_size,
            indices_size
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(take_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(),
            indices.data<int64_t>(),
            output.data<double>(),
            input_size,
            indices_size
        );
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(take_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<int32_t>(),
            indices.data<int64_t>(),
            output.data<int32_t>(),
            input_size,
            indices_size
        );
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(take_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<int64_t>(),
            indices.data<int64_t>(),
            output.data<int64_t>(),
            input_size,
            indices_size
        );
    } else {
        throw std::runtime_error("take_hip: Unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

// ==============================================================================
// Gather Relative Position Bias (for Swin Transformer)
// ==============================================================================

template<typename T>
__global__ void gather_2d_kernel(
    const T* table, const int64_t* indices, T* output,
    int64_t num_positions, int64_t num_heads, int64_t table_stride) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_positions * num_positions * num_heads;

    if (idx >= total) return;

    int64_t h = idx % num_heads;
    int64_t j = (idx / num_heads) % num_positions;
    int64_t i = idx / (num_heads * num_positions);

    int64_t table_idx = indices[i * num_positions + j];
    output[idx] = table[table_idx * num_heads + h];
}

auto gather_relative_position_bias_kernel(const Tensor& table, const Tensor& indices,
                                          int64_t num_positions, int64_t num_heads,
                                          hipStream_t stream) -> Tensor {
    // table: [table_size*table_size, num_heads]
    // indices: [num_positions, num_positions]
    // output: [num_positions, num_positions, num_heads]

    Tensor output({num_positions, num_positions, num_heads}, table.dtype(), table.device());

    int64_t total = num_positions * num_positions * num_heads;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    // Ensure indices are on the same device
    Tensor indices_device = indices.device() == table.device() ? indices : indices.to(table.device());

    if (table.dtype() == DType::Float32) {
        hipLaunchKernelGGL(gather_2d_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            table.data<float>(), indices_device.data<int64_t>(), output.data<float>(),
            num_positions, num_heads, num_heads);
    } else if (table.dtype() == DType::Float64) {
        hipLaunchKernelGGL(gather_2d_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            table.data<double>(), indices_device.data<int64_t>(), output.data<double>(),
            num_positions, num_heads, num_heads);
    } else if (table.dtype() == DType::Float16) {
        hipLaunchKernelGGL(gather_2d_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(table.data<Float16>()),
            indices_device.data<int64_t>(),
            reinterpret_cast<__half*>(output.data<Float16>()),
            num_positions, num_heads, num_heads);
    } else {
        throw std::runtime_error("gather_relative_position_bias: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

// ==============================================================================
// Scatter Add Operation — uses atomicAdd for overlapping indices
// ==============================================================================

template<typename T, typename IndexT>
__global__ void scatter_add_kernel_impl(
    const IndexT* indices,
    const T* src,
    T* output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size,
    int64_t index_dim_size,
    int64_t total_scatter,
    int* error_flag) {

    HIP_KERNEL_LOOP(idx, total_scatter) {
        int64_t inner_idx = idx % inner_size;
        int64_t temp = idx / inner_size;
        int64_t index_pos = temp % index_dim_size;
        int64_t outer_idx = temp / index_dim_size;

        int64_t index_offset = outer_idx * index_dim_size * inner_size +
                               index_pos * inner_size + inner_idx;
        int64_t scatter_idx = static_cast<int64_t>(indices[index_offset]);
        if (scatter_idx < 0) scatter_idx += dim_size;
        if (scatter_idx < 0 || scatter_idx >= dim_size) {
            atomicExch(error_flag, 1);
            return;
        }

        int64_t output_offset = outer_idx * dim_size * inner_size +
                                scatter_idx * inner_size +
                                inner_idx;

        atomicAddHelper(&output[output_offset], src[idx]);
    }
}

auto scatter_add_kernel(const Tensor& input, int64_t dim, const Tensor& index,
                        const Tensor& src, hipStream_t stream) -> Tensor {
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("scatter_add: dimension out of range");
    }

    std::vector<int64_t> output_shape(input.shape().begin(), input.shape().end());
    Tensor output(output_shape, input.dtype(), input.device());

    int64_t total_input = input.numel();
    int64_t total_scatter = index.numel();

    if (total_input == 0) return output;

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= input.shape()[i];
    int64_t dim_size = input.shape()[dim];
    int64_t index_dim_size = index.shape()[dim];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= input.shape()[i];

    bool idx_is_int32 = (index.dtype() == DType::Int32);

    // Step 1: Copy input to output
    HIP_CHECK(hipMemcpyAsync(output.data_ptr(), input.data_ptr(),
                             total_input * dtype_size(input.dtype()),
                             hipMemcpyDeviceToDevice, stream));

    // Step 2: Scatter-add with atomicAdd
    if (total_scatter == 0) return output;

    int threads = 256;
    int blocks = (total_scatter + threads - 1) / threads;

    // Device-side OOB error flag
    int* d_error_flag = nullptr;
    HIP_CHECK(hipMalloc(&d_error_flag, sizeof(int)));
    HIP_CHECK(hipMemsetAsync(d_error_flag, 0, sizeof(int), stream));

    #define LAUNCH_SCATTER_ADD_HIP(T) \
        if (idx_is_int32) \
            hipLaunchKernelGGL((scatter_add_kernel_impl<T, int32_t>), \
                dim3(blocks), dim3(threads), 0, stream, \
                index.data<int32_t>(), src.data<T>(), output.data<T>(), \
                outer_size, dim_size, inner_size, index_dim_size, total_scatter, \
                d_error_flag); \
        else \
            hipLaunchKernelGGL((scatter_add_kernel_impl<T, int64_t>), \
                dim3(blocks), dim3(threads), 0, stream, \
                index.data<int64_t>(), src.data<T>(), output.data<T>(), \
                outer_size, dim_size, inner_size, index_dim_size, total_scatter, \
                d_error_flag); \
        HIP_CHECK(hipGetLastError())

    switch (input.dtype()) {
        case DType::Float32: LAUNCH_SCATTER_ADD_HIP(float); break;
        case DType::Float64: LAUNCH_SCATTER_ADD_HIP(double); break;
        case DType::Int32:   LAUNCH_SCATTER_ADD_HIP(int32_t); break;
        case DType::Int64:   LAUNCH_SCATTER_ADD_HIP(int64_t); break;
        case DType::Float16:
        case DType::BFloat16: {
            // Upcast to Float32 for atomicAdd, then cast back
            Tensor input_f32 = input.to(DType::Float32);
            Tensor src_f32 = src.to(DType::Float32);
            Tensor output_f32(output_shape, DType::Float32, input.device());
            HIP_CHECK(hipMemcpyAsync(output_f32.data_ptr(), input_f32.data_ptr(),
                                     total_input * sizeof(float),
                                     hipMemcpyDeviceToDevice, stream));
            if (total_scatter > 0) {
                int blocks_f32 = (total_scatter + threads - 1) / threads;
                if (idx_is_int32)
                    hipLaunchKernelGGL((scatter_add_kernel_impl<float, int32_t>),
                        dim3(blocks_f32), dim3(threads), 0, stream,
                        index.data<int32_t>(), src_f32.data<float>(), output_f32.data<float>(),
                        outer_size, dim_size, inner_size, index_dim_size, total_scatter,
                        d_error_flag);
                else
                    hipLaunchKernelGGL((scatter_add_kernel_impl<float, int64_t>),
                        dim3(blocks_f32), dim3(threads), 0, stream,
                        index.data<int64_t>(), src_f32.data<float>(), output_f32.data<float>(),
                        outer_size, dim_size, inner_size, index_dim_size, total_scatter,
                        d_error_flag);
                HIP_CHECK(hipGetLastError());
            }
            output = output_f32.to(input.dtype());
            break;
        }
        default: throw std::runtime_error("scatter_add: unsupported dtype " +
                     std::string(dtype_name(input.dtype())));
    }

    #undef LAUNCH_SCATTER_ADD_HIP

    // Check for out-of-bounds index errors
    int host_error = 0;
    HIP_CHECK(hipMemcpyAsync(&host_error, d_error_flag, sizeof(int),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_error_flag));

    if (host_error) {
        throw std::out_of_range(
            "scatter_add: index out of range for dimension of size " +
            std::to_string(dim_size));
    }

    return output;
}

// ==============================================================================
// EmbeddingBag Forward — sum/mean/max aggregation of embedding bags
// ==============================================================================

template<typename T>
__global__ void embedding_bag_sum_kernel_hip(
    const T* embeddings,
    const int64_t* offsets,
    T* output,
    int64_t num_bags,
    int64_t total_elements,
    int64_t embedding_dim,
    int64_t offsets_size,
    bool divide_by_count)
{
    int64_t bag = blockIdx.x;
    if (bag >= num_bags) return;

    int64_t start = offsets[bag];
    int64_t end = (bag + 1 < offsets_size) ? offsets[bag + 1] : total_elements;
    int64_t bag_size = end - start;

    for (int64_t j = threadIdx.x; j < embedding_dim; j += blockDim.x) {
        T acc = T(0);
        for (int64_t i = start; i < end; ++i) {
            acc += embeddings[i * embedding_dim + j];
        }
        if (divide_by_count && bag_size > 0) {
            acc = acc / T(bag_size);
        }
        output[bag * embedding_dim + j] = acc;
    }
}

template<typename T>
__global__ void embedding_bag_max_kernel_hip(
    const T* embeddings,
    const int64_t* offsets,
    T* output,
    int64_t num_bags,
    int64_t total_elements,
    int64_t embedding_dim,
    int64_t offsets_size)
{
    int64_t bag = blockIdx.x;
    if (bag >= num_bags) return;

    int64_t start = offsets[bag];
    int64_t end = (bag + 1 < offsets_size) ? offsets[bag + 1] : total_elements;

    if (start >= end) return;

    for (int64_t j = threadIdx.x; j < embedding_dim; j += blockDim.x) {
        T max_val = embeddings[start * embedding_dim + j];
        for (int64_t i = start + 1; i < end; ++i) {
            T val = embeddings[i * embedding_dim + j];
            if (val > max_val) max_val = val;
        }
        output[bag * embedding_dim + j] = max_val;
    }
}

auto embedding_bag_forward_kernel(const Tensor& embeddings, const Tensor& offsets,
                                   const std::string& mode, int64_t embedding_dim,
                                   bool include_last_offset, hipStream_t stream) -> Tensor {
    int64_t total_elements = embeddings.shape()[0];
    int64_t offsets_size = offsets.numel();
    int64_t num_bags = include_last_offset ? (offsets_size - 1) : offsets_size;

    if (num_bags <= 0) {
        return Tensor({0, embedding_dim}, embeddings.dtype(), embeddings.device());
    }

    // Create zero-initialized output
    Tensor output({num_bags, embedding_dim}, embeddings.dtype(), embeddings.device());
    HIP_CHECK(hipMemsetAsync(output.data_ptr(), 0,
                             num_bags * embedding_dim * dtype_size(embeddings.dtype()), stream));

    int threads = std::min(static_cast<int>(embedding_dim), 256);
    int blocks = static_cast<int>(num_bags);

    bool is_mean = (mode == "mean");
    bool is_max = (mode == "max");

    switch (embeddings.dtype()) {
        case DType::Float32:
            if (is_max) {
                hipLaunchKernelGGL(embedding_bag_max_kernel_hip<float>,
                    dim3(blocks), dim3(threads), 0, stream,
                    embeddings.data<float>(), offsets.data<int64_t>(),
                    output.data<float>(), num_bags, total_elements,
                    embedding_dim, offsets_size);
            } else {
                hipLaunchKernelGGL(embedding_bag_sum_kernel_hip<float>,
                    dim3(blocks), dim3(threads), 0, stream,
                    embeddings.data<float>(), offsets.data<int64_t>(),
                    output.data<float>(), num_bags, total_elements,
                    embedding_dim, offsets_size, is_mean);
            }
            break;
        case DType::Float64:
            if (is_max) {
                hipLaunchKernelGGL(embedding_bag_max_kernel_hip<double>,
                    dim3(blocks), dim3(threads), 0, stream,
                    embeddings.data<double>(), offsets.data<int64_t>(),
                    output.data<double>(), num_bags, total_elements,
                    embedding_dim, offsets_size);
            } else {
                hipLaunchKernelGGL(embedding_bag_sum_kernel_hip<double>,
                    dim3(blocks), dim3(threads), 0, stream,
                    embeddings.data<double>(), offsets.data<int64_t>(),
                    output.data<double>(), num_bags, total_elements,
                    embedding_dim, offsets_size, is_mean);
            }
            break;
        case DType::Float16:
            if (is_max) {
                hipLaunchKernelGGL(embedding_bag_max_kernel_hip<__half>,
                    dim3(blocks), dim3(threads), 0, stream,
                    reinterpret_cast<const __half*>(embeddings.data_ptr()),
                    offsets.data<int64_t>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    num_bags, total_elements, embedding_dim, offsets_size);
            } else {
                hipLaunchKernelGGL(embedding_bag_sum_kernel_hip<__half>,
                    dim3(blocks), dim3(threads), 0, stream,
                    reinterpret_cast<const __half*>(embeddings.data_ptr()),
                    offsets.data<int64_t>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    num_bags, total_elements, embedding_dim, offsets_size, is_mean);
            }
            break;
        default:
            throw std::runtime_error("embedding_bag_forward: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

// ==============================================================================
// OneHot Operation
// ==============================================================================

template<typename IndexT>
__global__ void one_hot_kernel_impl(
    const IndexT* indices,
    float* output,
    int64_t batch_size,
    int64_t num_classes) {

    int64_t total = batch_size * num_classes;
    HIP_KERNEL_LOOP(idx, total) {
        int64_t batch = idx / num_classes;
        int64_t cls = idx % num_classes;
        output[idx] = (static_cast<int64_t>(indices[batch]) == cls) ? 1.0f : 0.0f;
    }
}

auto one_hot_kernel(const Tensor& indices, int64_t num_classes,
                    hipStream_t stream) -> Tensor {
    int64_t batch_size = indices.numel();

    Tensor output({batch_size, num_classes}, DType::Float32, indices.device());

    if (batch_size == 0) return output;

    int64_t total = batch_size * num_classes;
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (indices.dtype()) {
        case DType::Int32:
            hipLaunchKernelGGL(one_hot_kernel_impl<int32_t>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                indices.data<int32_t>(), output.data<float>(), batch_size, num_classes);
            break;
        case DType::Int64:
            hipLaunchKernelGGL(one_hot_kernel_impl<int64_t>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                indices.data<int64_t>(), output.data<float>(), batch_size, num_classes);
            break;
        default:
            throw std::runtime_error("one_hot: unsupported index dtype (expected Int32 or Int64)");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

// ==============================================================================
// Nonzero Operation
// ==============================================================================

template<typename T>
__global__ void nonzero_flag_kernel_hip(
    const T* input,
    int64_t* flags,
    int64_t n) {

    HIP_KERNEL_LOOP(i, n) {
        flags[i] = (input[i] != static_cast<T>(0)) ? 1 : 0;
    }
}

// Specialization for __half
template<>
__global__ void nonzero_flag_kernel_hip<__half>(
    const __half* input,
    int64_t* flags,
    int64_t n) {

    HIP_KERNEL_LOOP(i, n) {
        flags[i] = (__hne(input[i], __float2half(0.0f))) ? 1 : 0;
    }
}

// Decompose compacted flat indices into multi-dimensional indices
__global__ void decompose_flat_indices_kernel_hip(
    const int64_t* flat_indices,
    int64_t* output,
    const int64_t* shape,
    int64_t num_indices,
    int64_t ndim) {

    HIP_KERNEL_LOOP(i, num_indices) {
        int64_t flat = flat_indices[i];
        for (int64_t d = ndim - 1; d >= 0; --d) {
            output[i * ndim + d] = flat % shape[d];
            flat /= shape[d];
        }
    }
}

auto nonzero_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    int64_t n = input.numel();
    int64_t ndim = input.ndim();

    if (n == 0) {
        return Tensor({0, ndim}, DType::Int64, input.device());
    }

    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate flags array
    int64_t* d_flags = nullptr;
    HIP_CHECK(hipMalloc(&d_flags, n * sizeof(int64_t)));

    // Launch flag kernel based on dtype
    #define LAUNCH_NONZERO_FLAG_HIP(T) \
        hipLaunchKernelGGL(nonzero_flag_kernel_hip<T>, \
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream, \
            input.data<T>(), d_flags, n); \
        HIP_CHECK(hipGetLastError())

    switch (input.dtype()) {
        case DType::Float32: LAUNCH_NONZERO_FLAG_HIP(float); break;
        case DType::Float64: LAUNCH_NONZERO_FLAG_HIP(double); break;
        case DType::Int32:   LAUNCH_NONZERO_FLAG_HIP(int32_t); break;
        case DType::Int64:   LAUNCH_NONZERO_FLAG_HIP(int64_t); break;
        case DType::Int8:    LAUNCH_NONZERO_FLAG_HIP(int8_t); break;
        case DType::UInt8:   LAUNCH_NONZERO_FLAG_HIP(uint8_t); break;
        case DType::Bool:    LAUNCH_NONZERO_FLAG_HIP(bool); break;
        case DType::Float16:
            hipLaunchKernelGGL(nonzero_flag_kernel_hip<__half>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                reinterpret_cast<const __half*>(input.data_ptr()), d_flags, n);
            HIP_CHECK(hipGetLastError());
            break;
        default:
            HIP_CHECK(hipFree(d_flags));
            throw std::runtime_error("nonzero: unsupported dtype");
    }

    #undef LAUNCH_NONZERO_FLAG_HIP

    // Use hipcub DeviceSelect::Flagged with CountingInputIterator to compact
    // nonzero flat indices in a single pass
    thrust::counting_iterator<int64_t> iota(0);

    int64_t* d_flat_indices = nullptr;
    HIP_CHECK(hipMalloc(&d_flat_indices, n * sizeof(int64_t)));

    int* d_num_selected = nullptr;
    HIP_CHECK(hipMalloc(&d_num_selected, sizeof(int)));

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    hipcub::DeviceSelect::Flagged(d_temp, temp_bytes,
        iota, d_flags, d_flat_indices, d_num_selected,
        static_cast<int>(n), stream);
    HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
    hipcub::DeviceSelect::Flagged(d_temp, temp_bytes,
        iota, d_flags, d_flat_indices, d_num_selected,
        static_cast<int>(n), stream);

    // D2H sync to get count
    int total_nonzero;
    HIP_CHECK(hipMemcpyAsync(&total_nonzero, d_num_selected, sizeof(int), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipFree(d_flags));
    HIP_CHECK(hipFree(d_temp));
    HIP_CHECK(hipFree(d_num_selected));

    if (total_nonzero == 0) {
        HIP_CHECK(hipFree(d_flat_indices));
        return Tensor({0, ndim}, DType::Int64, input.device());
    }

    // Allocate output tensor and decompose flat indices to multi-dim
    Tensor output({static_cast<int64_t>(total_nonzero), ndim}, DType::Int64, input.device());

    int64_t* d_shape = nullptr;
    HIP_CHECK(hipMalloc(&d_shape, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMemcpyAsync(d_shape, input.shape().data(),
                              ndim * sizeof(int64_t), hipMemcpyHostToDevice, stream));

    int decompose_blocks = (total_nonzero + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(decompose_flat_indices_kernel_hip,
        dim3(decompose_blocks), dim3(BLOCK_SIZE), 0, stream,
        d_flat_indices, output.data<int64_t>(), d_shape, total_nonzero, ndim);

    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipFree(d_flat_indices));
    HIP_CHECK(hipFree(d_shape));

    return output;
}

} // namespace rocm
} // namespace tenzor
