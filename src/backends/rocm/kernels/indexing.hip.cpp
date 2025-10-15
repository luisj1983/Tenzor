#include <hip/hip_runtime.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
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
    int64_t indices_size,
    int64_t inner_size,
    int64_t dim_size,
    bool reduce_add
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
            int64_t output_idx = indices_idx / indices_size * dim_size * inner_size +
                                index * inner_size + inner_idx;

            if (reduce_add) {
                atomicAddHelper(&output[output_idx], src[idx]);
            } else {
                output[output_idx] = src[idx];
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

    // Calculate dimensions
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= output_shape[i];
    }

    int64_t dim_size = output_shape[dim];

    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < output_shape.size(); ++i) {
        inner_size *= output_shape[i];
    }

    int64_t indices_size = indices.numel();
    int64_t total_elements = indices_size * inner_size;

    bool reduce_add = (reduce == "add");

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(scatter_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<float>(),
            indices.data<int64_t>(),
            src.data<float>(),
            indices_size,
            inner_size,
            dim_size,
            reduce_add
        );
    } else if (output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(scatter_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<double>(),
            indices.data<int64_t>(),
            src.data<double>(),
            indices_size,
            inner_size,
            dim_size,
            reduce_add
        );
    } else if (output.dtype() == DType::Int32) {
        hipLaunchKernelGGL(scatter_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<int32_t>(),
            indices.data<int64_t>(),
            src.data<int32_t>(),
            indices_size,
            inner_size,
            dim_size,
            reduce_add
        );
    } else if (output.dtype() == DType::Int64) {
        hipLaunchKernelGGL(scatter_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<int64_t>(),
            indices.data<int64_t>(),
            src.data<int64_t>(),
            indices_size,
            inner_size,
            dim_size,
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

} // namespace rocm
} // namespace tenzor
