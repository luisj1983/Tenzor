#include "rocm_nan_helpers.hip.h"  // E.2: safe_f2h / safe_h2f / safe_f2bf / safe_bf2f
#include "bfloat16_helpers.hpp"   // S.10 / R.11: f32_to_bf16_rne
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/backend.hpp"
#include <hipcub/hipcub.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <stdexcept>
#include <vector>
#include <cstring>
#include "../rocm_error.hpp"
#include "tenzor/ops/creation.hpp"

namespace tenzor {
namespace rocm {

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

// Specialization for hip_bfloat16
// R.11 / S.10: the previous `hip_bfloat16 new_bf16(new_f)` truncating ctor
// dropped the low 16 bits of the float32 sum on every CAS retry. Under
// contention (multiple gradients into the same embedding row) the truncation
// bias compounds — scatter/index_add BF16 grads diverged systematically from
// the Float32 reference. We now route through the shared
// f32_to_bf16_rne(...) helper so accumulation uses round-to-nearest-even.
template<>
__device__ __forceinline__ hip_bfloat16 atomicAddHelper<hip_bfloat16>(hip_bfloat16* address, hip_bfloat16 val) {
    unsigned short int* address_as_ushort = (unsigned short int*)address;
    unsigned short int old = *address_as_ushort, assumed;
    do {
        assumed = old;
        float assumed_f = static_cast<float>(*reinterpret_cast<hip_bfloat16*>(&assumed));
        float new_f = assumed_f + static_cast<float>(val);
        hip_bfloat16 new_bf16 = tenzor::rocm::f32_to_bf16_rne(new_f);
        unsigned short int new_bits = *reinterpret_cast<unsigned short int*>(&new_bf16);
        old = atomicCAS(address_as_ushort, assumed, new_bits);
    } while (assumed != old);
    return *reinterpret_cast<hip_bfloat16*>(&old);
}

// ==============================================================================
// Gather Operation
// ==============================================================================

// Gather with PyTorch semantics: output shape == index shape.
// For dim d: out[i0..i_{d-1}, p, i_{d+1}..] = input[i0..i_{d-1}, index[i0..p..], i_{d+1}..]
template<typename T>
__global__ void gather_kernel(
    const T* input,
    const int64_t* indices,
    T* output,
    int64_t /*input_size*/,
    int64_t /*indices_numel_unused*/,
    int64_t inner_size,
    int64_t dim_size,
    int64_t index_dim_size,
    int64_t total_output
) {
    HIP_KERNEL_LOOP(idx, total_output) {
        int64_t inner_idx = idx % inner_size;
        int64_t temp = idx / inner_size;
        int64_t index_pos = temp % index_dim_size;
        int64_t outer_idx = temp / index_dim_size;

        int64_t index_offset = outer_idx * index_dim_size * inner_size +
                               index_pos * inner_size + inner_idx;
        int64_t gather_idx = indices[index_offset];
        if (gather_idx < 0) gather_idx += dim_size;
        if (gather_idx < 0 || gather_idx >= dim_size) continue;

        int64_t input_offset = outer_idx * dim_size * inner_size +
                               gather_idx * inner_size + inner_idx;
        output[idx] = input[input_offset];
    }
}

auto gather_hip(
    const Tensor& input,
    int64_t dim,
    const Tensor& indices
) -> Tensor {

    auto input_shape = input.shape();
    auto indices_shape = indices.shape();

    // Normalize negative dim (PyTorch semantics: dim=-1 → last)
    if (dim < 0) dim += static_cast<int64_t>(input_shape.size());
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::out_of_range("gather_hip: dim out of range");
    }

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
    int64_t index_dim_size = indices_shape[dim];
    int64_t total_output = output.numel();

    int threads = 256;
    int blocks = static_cast<int>((total_output + threads - 1) / threads);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(gather_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<float>(),
            indices.data<int64_t>(),
            output.data<float>(),
            input.numel(), indices_size, inner_size, dim_size,
            index_dim_size, total_output);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(gather_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<double>(),
            indices.data<int64_t>(),
            output.data<double>(),
            input.numel(), indices_size, inner_size, dim_size,
            index_dim_size, total_output);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(gather_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<int32_t>(),
            indices.data<int64_t>(),
            output.data<int32_t>(),
            input.numel(), indices_size, inner_size, dim_size,
            index_dim_size, total_output);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(gather_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<int64_t>(),
            indices.data<int64_t>(),
            output.data<int64_t>(),
            input.numel(), indices_size, inner_size, dim_size,
            index_dim_size, total_output);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(gather_kernel<__half>,
            dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            indices.data<int64_t>(),
            reinterpret_cast<__half*>(output.data<Float16>()),
            input.numel(), indices_size, inner_size, dim_size,
            index_dim_size, total_output);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(gather_kernel<hip_bfloat16>,
            dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            indices.data<int64_t>(),
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            input.numel(), indices_size, inner_size, dim_size,
            index_dim_size, total_output);
        HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("gather_hip: Unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();

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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(scatter_kernel<hip_bfloat16>,
            dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            indices.data<int64_t>(),
            reinterpret_cast<const hip_bfloat16*>(src.data<BFloat16>()),
            outer_size,
            dim_size,
            inner_size,
            index_dim_size,
            total_scatter,
            reduce_add
        );
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("scatter_hip: Unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();

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

    // Normalize negative dim (PyTorch semantics: dim=-1 → last)
    if (dim < 0) dim += static_cast<int64_t>(input_shape.size());
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::out_of_range("index_select_hip: dim out of range");
    }

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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(index_select_kernel<__half>,
            dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            indices.data<int64_t>(),
            reinterpret_cast<__half*>(output.data<Float16>()),
            outer_size,
            dim_size,
            inner_size,
            num_indices
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(index_select_kernel<hip_bfloat16>,
            dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            indices.data<int64_t>(),
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            outer_size,
            dim_size,
            inner_size,
            num_indices
        );
        HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("index_select_hip: Unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();

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
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(masked_fill_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<double>(),
            mask.data<bool>(),
            static_cast<double>(value),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Int32) {
        hipLaunchKernelGGL(masked_fill_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<int32_t>(),
            mask.data<bool>(),
            static_cast<int32_t>(value),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Int64) {
        hipLaunchKernelGGL(masked_fill_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<int64_t>(),
            mask.data<bool>(),
            static_cast<int64_t>(value),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Float16) {
        hipLaunchKernelGGL(masked_fill_kernel<__half>,
            dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<__half*>(output.data<Float16>()),
            mask.data<bool>(),
            tenzor::rocm::safe_f2h(static_cast<float>(value)),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(masked_fill_kernel<hip_bfloat16>,
            dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            mask.data<bool>(),
            // S.10: RNE-round so a Float64 user-supplied fill value doesn't
            // truncate its low mantissa silently.
            tenzor::rocm::f32_to_bf16_rne(static_cast<float>(value)),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("masked_fill_hip: Unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();

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
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(masked_select_count_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            mask.data<bool>(),
            d_count,
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(masked_select_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<double>(),
            mask.data<bool>(),
            output.data<double>(),
            d_output_idx,
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    }

    HIP_CHECK(hipFree(d_count));
    HIP_CHECK(hipFree(d_output_idx));
    HIP_POST_LAUNCH_CHECK();

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
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(take_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<double>(),
            indices.data<int64_t>(),
            output.data<double>(),
            input_size,
            indices_size
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(take_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<int32_t>(),
            indices.data<int64_t>(),
            output.data<int32_t>(),
            input_size,
            indices_size
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(take_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<int64_t>(),
            indices.data<int64_t>(),
            output.data<int64_t>(),
            input_size,
            indices_size
        );
        HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("take_hip: Unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();

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
        HIP_POST_LAUNCH_CHECK();
    } else if (x.dtype() == DType::Float64) {
        hipLaunchKernelGGL(where_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            condition.data<bool>(),
            x.data<double>(),
            y.data<double>(),
            output.data<double>(),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (x.dtype() == DType::Int32) {
        hipLaunchKernelGGL(where_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, stream,
            condition.data<bool>(),
            x.data<int32_t>(),
            y.data<int32_t>(),
            output.data<int32_t>(),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (x.dtype() == DType::Int64) {
        hipLaunchKernelGGL(where_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            condition.data<bool>(),
            x.data<int64_t>(),
            y.data<int64_t>(),
            output.data<int64_t>(),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (x.dtype() == DType::Float16) {
        hipLaunchKernelGGL(where_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            condition.data<bool>(),
            reinterpret_cast<const __half*>(x.data<Float16>()),
            reinterpret_cast<const __half*>(y.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (x.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(where_kernel<hip_bfloat16>,
            dim3(blocks), dim3(threads), 0, stream,
            condition.data<bool>(),
            reinterpret_cast<const hip_bfloat16*>(x.data<BFloat16>()),
            reinterpret_cast<const hip_bfloat16*>(y.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("where_hip: Unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(slice_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            outer_size,
            input_shape[dim],
            inner_size,
            start,
            end,
            step,
            output_dim_size
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(slice_kernel<hip_bfloat16>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            outer_size,
            input_shape[dim],
            inner_size,
            start,
            end,
            step,
            output_dim_size
        );
        HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("slice_hip: Unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
    } else if (first.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(cat_kernel<hip_bfloat16>,
            dim3(blocks), dim3(threads), 0, stream,
            (const hip_bfloat16* const*)d_input_ptrs,
            d_input_offsets,
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            static_cast<int64_t>(cont_tensors.size()),
            outer_size,
            total_dim_size,
            inner_size,
            d_dim_sizes
        );
        HIP_POST_LAUNCH_CHECK();
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
    HIP_POST_LAUNCH_CHECK();

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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("put_hip: Unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
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

    // Normalize negative dim (PyTorch semantics: dim=-1 → last)
    if (dim < 0) dim += static_cast<int64_t>(input_shape.size());
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::out_of_range("gather_hip: dim out of range");
    }

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
    int64_t index_dim_size = indices_shape[dim];
    int64_t total_output = output.numel();

    int threads = 256;
    int blocks = static_cast<int>((total_output + threads - 1) / threads);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(gather_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(),
            indices.data<int64_t>(),
            output.data<float>(),
            input.numel(), indices_size, inner_size, dim_size,
            index_dim_size, total_output);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(gather_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(),
            indices.data<int64_t>(),
            output.data<double>(),
            input.numel(), indices_size, inner_size, dim_size,
            index_dim_size, total_output);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(gather_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<int32_t>(),
            indices.data<int64_t>(),
            output.data<int32_t>(),
            input.numel(), indices_size, inner_size, dim_size,
            index_dim_size, total_output);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(gather_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<int64_t>(),
            indices.data<int64_t>(),
            output.data<int64_t>(),
            input.numel(), indices_size, inner_size, dim_size,
            index_dim_size, total_output);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(gather_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            indices.data<int64_t>(),
            reinterpret_cast<__half*>(output.data<Float16>()),
            input.numel(), indices_size, inner_size, dim_size,
            index_dim_size, total_output);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(gather_kernel<hip_bfloat16>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            indices.data<int64_t>(),
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            input.numel(), indices_size, inner_size, dim_size,
            index_dim_size, total_output);
        HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("gather_hip: Unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(scatter_kernel<hip_bfloat16>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            indices.data<int64_t>(),
            reinterpret_cast<const hip_bfloat16*>(src.data<BFloat16>()),
            outer_size,
            dim_size,
            inner_size,
            index_dim_size,
            total_scatter,
            false
        );
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("scatter_hip: Unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return output;
}

auto index_select_hip(
    const Tensor& input,
    int64_t dim,
    const Tensor& indices,
    hipStream_t stream
) -> Tensor {
    auto input_shape = input.shape();

    // Normalize negative dim (PyTorch semantics: dim=-1 → last)
    if (dim < 0) dim += static_cast<int64_t>(input_shape.size());
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::out_of_range("index_select_hip: dim out of range");
    }

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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(index_select_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            indices.data<int64_t>(),
            reinterpret_cast<__half*>(output.data<Float16>()),
            outer_size,
            dim_size,
            inner_size,
            num_indices
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(index_select_kernel<hip_bfloat16>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            indices.data<int64_t>(),
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            outer_size,
            dim_size,
            inner_size,
            num_indices
        );
        HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("index_select_hip: Unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(masked_fill_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<double>(),
            mask.data<bool>(),
            static_cast<double>(value),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Int32) {
        hipLaunchKernelGGL(masked_fill_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<int32_t>(),
            mask.data<bool>(),
            static_cast<int32_t>(value),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Int64) {
        hipLaunchKernelGGL(masked_fill_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<int64_t>(),
            mask.data<bool>(),
            static_cast<int64_t>(value),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Float16) {
        hipLaunchKernelGGL(masked_fill_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<__half*>(output.data<Float16>()),
            mask.data<bool>(),
            tenzor::rocm::safe_f2h(static_cast<float>(value)),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(masked_fill_kernel<hip_bfloat16>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            mask.data<bool>(),
            // S.10: RNE round, see paired site above.
            tenzor::rocm::f32_to_bf16_rne(static_cast<float>(value)),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("masked_fill_hip: Unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
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
    HIP_CHECK(hipMemsetAsync(d_count, 0, sizeof(int64_t), stream));

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(masked_select_count_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            mask.data<bool>(),
            d_count,
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(masked_select_count_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            mask.data<bool>(),
            d_count,
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
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
    HIP_CHECK(hipMemsetAsync(d_output_idx, 0, sizeof(int64_t), stream));

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(masked_select_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(),
            mask.data<bool>(),
            output.data<float>(),
            d_output_idx,
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(masked_select_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(),
            mask.data<bool>(),
            output.data<double>(),
            d_output_idx,
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    }

    HIP_CHECK(hipFree(d_count));
    HIP_CHECK(hipFree(d_output_idx));
    HIP_POST_LAUNCH_CHECK();

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
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(take_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(),
            indices.data<int64_t>(),
            output.data<double>(),
            input_size,
            indices_size
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(take_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<int32_t>(),
            indices.data<int64_t>(),
            output.data<int32_t>(),
            input_size,
            indices_size
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(take_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<int64_t>(),
            indices.data<int64_t>(),
            output.data<int64_t>(),
            input_size,
            indices_size
        );
        HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("take_hip: Unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
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
            HIP_POST_LAUNCH_CHECK();
    } else if (table.dtype() == DType::Float64) {
        hipLaunchKernelGGL(gather_2d_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            table.data<double>(), indices_device.data<int64_t>(), output.data<double>(),
            num_positions, num_heads, num_heads);
            HIP_POST_LAUNCH_CHECK();
    } else if (table.dtype() == DType::Float16) {
        hipLaunchKernelGGL(gather_2d_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(table.data<Float16>()),
            indices_device.data<int64_t>(),
            reinterpret_cast<__half*>(output.data<Float16>()),
            num_positions, num_heads, num_heads);
            HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("gather_relative_position_bias: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
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
        HIP_POST_LAUNCH_CHECK();

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
                if (idx_is_int32) {
                    hipLaunchKernelGGL((scatter_add_kernel_impl<float, int32_t>),
                        dim3(blocks_f32), dim3(threads), 0, stream,
                        index.data<int32_t>(), src_f32.data<float>(), output_f32.data<float>(),
                        outer_size, dim_size, inner_size, index_dim_size, total_scatter,
                        d_error_flag);
                    HIP_POST_LAUNCH_CHECK();
                } else {
                    hipLaunchKernelGGL((scatter_add_kernel_impl<float, int64_t>),
                        dim3(blocks_f32), dim3(threads), 0, stream,
                        index.data<int64_t>(), src_f32.data<float>(), output_f32.data<float>(),
                        outer_size, dim_size, inner_size, index_dim_size, total_scatter,
                        d_error_flag);
                    HIP_POST_LAUNCH_CHECK();
                }
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

// BFloat16 needs an F32-accumulator specialisation because hip_bfloat16's
// operator overloads aren't comprehensive enough for the templated kernel
// above (see the F32 round-trip in atomicAddHelper<hip_bfloat16> earlier
// in this file). Load as BF16, compute in F32, store as BF16.
__global__ void embedding_bag_sum_kernel_hip_bf16(
    const hip_bfloat16* embeddings,
    const int64_t* offsets,
    hip_bfloat16* output,
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
        float acc = 0.0f;
        for (int64_t i = start; i < end; ++i) {
            acc += static_cast<float>(embeddings[i * embedding_dim + j]);
        }
        if (divide_by_count && bag_size > 0) {
            acc /= static_cast<float>(bag_size);
        }
        // S.10: RNE-round the accumulator instead of the truncating ctor —
        // mean over a bag drifts predictably if we drop the low mantissa.
        output[bag * embedding_dim + j] = tenzor::rocm::f32_to_bf16_rne(acc);
    }
}

__global__ void embedding_bag_max_kernel_hip_bf16(
    const hip_bfloat16* embeddings,
    const int64_t* offsets,
    hip_bfloat16* output,
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
        float max_val = static_cast<float>(embeddings[start * embedding_dim + j]);
        for (int64_t i = start + 1; i < end; ++i) {
            float val = static_cast<float>(embeddings[i * embedding_dim + j]);
            if (val > max_val) max_val = val;
        }
        // S.10: max over a bag is exact in float32 but the back-cast still
        // needs RNE rounding to avoid the truncating-ctor bias documented
        // in R.11.
        output[bag * embedding_dim + j] = tenzor::rocm::f32_to_bf16_rne(max_val);
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
                HIP_POST_LAUNCH_CHECK();
            } else {
                hipLaunchKernelGGL(embedding_bag_sum_kernel_hip<float>,
                    dim3(blocks), dim3(threads), 0, stream,
                    embeddings.data<float>(), offsets.data<int64_t>(),
                    output.data<float>(), num_bags, total_elements,
                    embedding_dim, offsets_size, is_mean);
                HIP_POST_LAUNCH_CHECK();
            }
            break;
        case DType::Float64:
            if (is_max) {
                hipLaunchKernelGGL(embedding_bag_max_kernel_hip<double>,
                    dim3(blocks), dim3(threads), 0, stream,
                    embeddings.data<double>(), offsets.data<int64_t>(),
                    output.data<double>(), num_bags, total_elements,
                    embedding_dim, offsets_size);
                HIP_POST_LAUNCH_CHECK();
            } else {
                hipLaunchKernelGGL(embedding_bag_sum_kernel_hip<double>,
                    dim3(blocks), dim3(threads), 0, stream,
                    embeddings.data<double>(), offsets.data<int64_t>(),
                    output.data<double>(), num_bags, total_elements,
                    embedding_dim, offsets_size, is_mean);
                HIP_POST_LAUNCH_CHECK();
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
                HIP_POST_LAUNCH_CHECK();
            } else {
                hipLaunchKernelGGL(embedding_bag_sum_kernel_hip<__half>,
                    dim3(blocks), dim3(threads), 0, stream,
                    reinterpret_cast<const __half*>(embeddings.data_ptr()),
                    offsets.data<int64_t>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    num_bags, total_elements, embedding_dim, offsets_size, is_mean);
                HIP_POST_LAUNCH_CHECK();
            }
            break;
        case DType::BFloat16:
            if (is_max) {
                hipLaunchKernelGGL(embedding_bag_max_kernel_hip_bf16,
                    dim3(blocks), dim3(threads), 0, stream,
                    reinterpret_cast<const hip_bfloat16*>(embeddings.data_ptr()),
                    offsets.data<int64_t>(),
                    reinterpret_cast<hip_bfloat16*>(output.data_ptr()),
                    num_bags, total_elements, embedding_dim, offsets_size);
                HIP_POST_LAUNCH_CHECK();
            } else {
                hipLaunchKernelGGL(embedding_bag_sum_kernel_hip_bf16,
                    dim3(blocks), dim3(threads), 0, stream,
                    reinterpret_cast<const hip_bfloat16*>(embeddings.data_ptr()),
                    offsets.data<int64_t>(),
                    reinterpret_cast<hip_bfloat16*>(output.data_ptr()),
                    num_bags, total_elements, embedding_dim, offsets_size, is_mean);
                HIP_POST_LAUNCH_CHECK();
            }
            break;
        default:
            throw std::runtime_error("embedding_bag_forward: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return output;
}

// ==============================================================================
// EmbeddingBagBackward Operation
// ==============================================================================

// Scatter-add into rows selected by `indices` (the original vocabulary ids
// the EmbeddingBag forward looked up). Each bag spans indices[start..end];
// the upstream gradient grad_output[bag] is distributed to every row in that
// bag (divided by bag_size for mean reduction).
template<typename T>
__global__ void embedding_bag_backward_kernel_hip(
    const T* grad_output,
    const int64_t* indices,
    const int64_t* offsets,
    T* grad_weight,
    int64_t num_bags,
    int64_t total_elements,
    int64_t embedding_dim,
    int64_t num_embeddings,
    int64_t offsets_size,
    bool is_mean)
{
    int64_t bag = blockIdx.x;
    if (bag >= num_bags) return;

    int64_t start = offsets[bag];
    int64_t end = (bag + 1 < offsets_size) ? offsets[bag + 1] : total_elements;
    int64_t bag_size = end - start;
    if (bag_size <= 0) return;

    for (int64_t j = threadIdx.x; j < embedding_dim; j += blockDim.x) {
        T grad_val = grad_output[bag * embedding_dim + j];
        if (is_mean) {
            grad_val = grad_val / static_cast<T>(bag_size);
        }
        for (int64_t i = start; i < end; ++i) {
            int64_t row = indices[i];
            if (row < 0 || row >= num_embeddings) continue;
            atomicAdd(&grad_weight[row * embedding_dim + j], grad_val);
        }
    }
}

auto embedding_bag_backward_kernel(const Tensor& grad_output,
                                   const Tensor& indices,
                                   const Tensor& offsets,
                                   const OpAttributes& attrs,
                                   hipStream_t stream) -> Tensor {
    int64_t num_embeddings = attrs.get_int(AttrKey::NumEmbeddings, 0);
    int64_t embedding_dim = attrs.get_int(AttrKey::EmbeddingDim, 0);
    std::string mode{attrs.get_string(AttrKey::Mode, "sum")};
    bool include_last_offset = attrs.get_bool(AttrKey::IncludeLastOffset, false);

    if (indices.dtype() != DType::Int64) {
        throw std::runtime_error("embedding_bag_backward: indices must be Int64");
    }

    int64_t total_elements = indices.numel();
    int64_t offsets_size = offsets.numel();
    int64_t num_bags = include_last_offset ? (offsets_size - 1) : offsets_size;

    if (num_bags <= 0) {
        return Tensor({num_embeddings, embedding_dim}, grad_output.dtype(), grad_output.device());
    }

    // FP16/BF16: upcast to Float32 (indices stays Int64)
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        auto go_f32 = grad_output.to(DType::Float32);
        auto result = embedding_bag_backward_kernel(go_f32, indices, offsets, attrs, stream);
        return result.to(grad_output.dtype());
    }

    Tensor grad_weight({num_embeddings, embedding_dim}, grad_output.dtype(), grad_output.device());
    HIP_CHECK(hipMemsetAsync(grad_weight.data_ptr(), 0,
                             num_embeddings * embedding_dim * dtype_size(grad_output.dtype()), stream));

    int threads = std::min(static_cast<int>(embedding_dim), 256);
    int blocks = static_cast<int>(num_bags);
    bool is_mean = (mode == "mean");

    switch (grad_output.dtype()) {
        case DType::Float32:
            hipLaunchKernelGGL(embedding_bag_backward_kernel_hip<float>,
                dim3(blocks), dim3(threads), 0, stream,
                grad_output.data<float>(), indices.data<int64_t>(), offsets.data<int64_t>(),
                grad_weight.data<float>(), num_bags, total_elements,
                embedding_dim, num_embeddings, offsets_size, is_mean);
            HIP_POST_LAUNCH_CHECK();
            break;
        case DType::Float64:
            hipLaunchKernelGGL(embedding_bag_backward_kernel_hip<double>,
                dim3(blocks), dim3(threads), 0, stream,
                grad_output.data<double>(), indices.data<int64_t>(), offsets.data<int64_t>(),
                grad_weight.data<double>(), num_bags, total_elements,
                embedding_dim, num_embeddings, offsets_size, is_mean);
            HIP_POST_LAUNCH_CHECK();
            break;
        default:
            throw std::runtime_error("embedding_bag_backward: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return grad_weight;
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
            HIP_POST_LAUNCH_CHECK();
            break;
        case DType::Int64:
            hipLaunchKernelGGL(one_hot_kernel_impl<int64_t>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                indices.data<int64_t>(), output.data<float>(), batch_size, num_classes);
            HIP_POST_LAUNCH_CHECK();
            break;
        default:
            throw std::runtime_error("one_hot: unsupported index dtype (expected Int32 or Int64)");
    }

    HIP_POST_LAUNCH_CHECK();
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
        flags[i] = (__hne(input[i], tenzor::rocm::safe_f2h(0.0f))) ? 1 : 0;
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
        HIP_POST_LAUNCH_CHECK();

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
            HIP_POST_LAUNCH_CHECK();
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
    HIP_CHECK(hipcub::DeviceSelect::Flagged(d_temp, temp_bytes,
        iota, d_flags, d_flat_indices, d_num_selected,
        static_cast<int>(n), stream));
    HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
    HIP_CHECK(hipcub::DeviceSelect::Flagged(d_temp, temp_bytes,
        iota, d_flags, d_flat_indices, d_num_selected,
        static_cast<int>(n), stream));

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

    HIP_POST_LAUNCH_CHECK();

    HIP_CHECK(hipFree(d_flat_indices));
    HIP_CHECK(hipFree(d_shape));

    return output;
}

// ============================================================================
// SearchSorted: binary search per element in sorted 1-D sequence
// ============================================================================

template<typename T>
__global__ void searchsorted_kernel_hip(
    const T* sorted_sequence,
    const T* values,
    int64_t* output,
    int64_t seq_len,
    int64_t num_values,
    bool right) {

    HIP_KERNEL_LOOP(i, num_values) {
        T v = values[i];
        int64_t lo = 0, hi = seq_len;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            bool go_right = right ? (sorted_sequence[mid] <= v) : (sorted_sequence[mid] < v);
            if (go_right) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        output[i] = lo;
    }
}

auto searchsorted_hip(const Tensor& sorted_sequence, const Tensor& values,
                       bool right, hipStream_t stream) -> Tensor {
    if (sorted_sequence.ndim() != 1) {
        throw std::runtime_error("searchsorted: sorted_sequence must be 1-D");
    }

    Tensor seq_cont = sorted_sequence.contiguous();
    Tensor val_cont = values.contiguous();
    int64_t seq_len = seq_cont.shape()[0];
    int64_t num_values = val_cont.numel();

    Tensor result(std::vector<int64_t>(values.shape().begin(), values.shape().end()),
                  DType::Int64, values.device());

    if (num_values == 0) return result;

    int64_t* out_ptr = result.data<int64_t>();
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (num_values + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (sorted_sequence.dtype()) {
        case DType::Float32:
            hipLaunchKernelGGL(searchsorted_kernel_hip<float>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                seq_cont.data<float>(), val_cont.data<float>(), out_ptr,
                seq_len, num_values, right);
            HIP_POST_LAUNCH_CHECK();
            break;
        case DType::Float64:
            hipLaunchKernelGGL(searchsorted_kernel_hip<double>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                seq_cont.data<double>(), val_cont.data<double>(), out_ptr,
                seq_len, num_values, right);
            HIP_POST_LAUNCH_CHECK();
            break;
        case DType::Int32:
            hipLaunchKernelGGL(searchsorted_kernel_hip<int32_t>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                seq_cont.data<int32_t>(), val_cont.data<int32_t>(), out_ptr,
                seq_len, num_values, right);
            HIP_POST_LAUNCH_CHECK();
            break;
        case DType::Int64:
            hipLaunchKernelGGL(searchsorted_kernel_hip<int64_t>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                seq_cont.data<int64_t>(), val_cont.data<int64_t>(), out_ptr,
                seq_len, num_values, right);
            HIP_POST_LAUNCH_CHECK();
            break;
        case DType::Float16:
        case DType::BFloat16: {
            auto seq_f32 = sorted_sequence.to(DType::Float32).contiguous();
            auto val_f32 = values.to(DType::Float32).contiguous();
            hipLaunchKernelGGL(searchsorted_kernel_hip<float>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                seq_f32.data<float>(), val_f32.data<float>(), out_ptr,
                seq_len, num_values, right);
            HIP_POST_LAUNCH_CHECK();
            break;
        }
        default:
            throw std::runtime_error("searchsorted: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return result;
}

// ============================================================================
// AdvancedIndex / AdvancedIndexPut — native ROCm/HIP implementations
// ============================================================================
//
// NumPy-style fancy indexing: gather elements from `src` using up to N index
// tensors that broadcast to a common shape, with optional passthrough dims.
// One thread per output element; each thread reads index values from the
// indexed dims, computes a source offset, and copies its element.
//
// All indices arrive as Int64 (cast at the dispatch layer); empty (numel=0)
// indices mark "full slice on this dim".
//
// MAX_ADV_INDEX_DIMS bounds the indexed-dim count and stride/shape buffers
// stored in kernel argument arrays.

namespace {
constexpr int MAX_ADV_INDEX_DIMS = 16;
}

struct AdvancedIndexMeta {
    int num_indices;
    int src_ndim;
    int num_pass_dims;
    int64_t bc_numel;
    int64_t pass_numel;
    int64_t src_shape[MAX_ADV_INDEX_DIMS];
    int64_t src_strides[MAX_ADV_INDEX_DIMS];
    int pass_dims[MAX_ADV_INDEX_DIMS];
    int is_indexed[MAX_ADV_INDEX_DIMS];
};

template<typename T>
__global__ void advanced_index_gather_kernel_hip(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const int64_t* __restrict__ const* __restrict__ idx_ptrs,
    AdvancedIndexMeta meta,
    int64_t total_out
) {
    int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out_idx >= total_out) return;

    int64_t p = (meta.pass_numel > 0) ? (out_idx % meta.pass_numel) : 0;
    int64_t bc = (meta.pass_numel > 0) ? (out_idx / meta.pass_numel) : out_idx;

    int64_t src_offset = 0;
    for (int i = 0; i < meta.num_indices; ++i) {
        if (meta.is_indexed[i]) {
            int64_t idx_val = idx_ptrs[i][bc];
            if (idx_val < 0) idx_val += meta.src_shape[i];
            src_offset += idx_val * meta.src_strides[i];
        }
    }

    if (meta.num_pass_dims > 0) {
        int64_t remaining = p;
        for (int k = meta.num_pass_dims - 1; k >= 0; --k) {
            int d = meta.pass_dims[k];
            int64_t coord = remaining % meta.src_shape[d];
            remaining /= meta.src_shape[d];
            src_offset += coord * meta.src_strides[d];
        }
    }

    dst[out_idx] = src[src_offset];
}

template<typename T>
__global__ void advanced_index_put_kernel_hip(
    T* __restrict__ dst,
    const T* __restrict__ values,
    const int64_t* __restrict__ const* __restrict__ idx_ptrs,
    AdvancedIndexMeta meta,
    int64_t total_out
) {
    int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out_idx >= total_out) return;

    int64_t p = (meta.pass_numel > 0) ? (out_idx % meta.pass_numel) : 0;
    int64_t bc = (meta.pass_numel > 0) ? (out_idx / meta.pass_numel) : out_idx;

    int64_t dst_offset = 0;
    for (int i = 0; i < meta.num_indices; ++i) {
        if (meta.is_indexed[i]) {
            int64_t idx_val = idx_ptrs[i][bc];
            if (idx_val < 0) idx_val += meta.src_shape[i];
            dst_offset += idx_val * meta.src_strides[i];
        }
    }

    if (meta.num_pass_dims > 0) {
        int64_t remaining = p;
        for (int k = meta.num_pass_dims - 1; k >= 0; --k) {
            int d = meta.pass_dims[k];
            int64_t coord = remaining % meta.src_shape[d];
            remaining /= meta.src_shape[d];
            dst_offset += coord * meta.src_strides[d];
        }
    }

    dst[dst_offset] = values[out_idx];
}

namespace {

struct PreparedAdvancedIndex {
    AdvancedIndexMeta meta;
    std::vector<int64_t> output_shape;
    int64_t total;
};

inline PreparedAdvancedIndex prepare_advanced_index_hip(
    const Tensor& src,
    const Tensor* const* index_tensors,
    int64_t num_indices
) {
    PreparedAdvancedIndex out{};
    auto src_shape_span = src.shape();
    int64_t src_ndim = static_cast<int64_t>(src_shape_span.size());
    if (src_ndim > MAX_ADV_INDEX_DIMS) {
        throw std::runtime_error("AdvancedIndex: source ndim exceeds MAX_ADV_INDEX_DIMS");
    }
    if (num_indices > MAX_ADV_INDEX_DIMS) {
        throw std::runtime_error("AdvancedIndex: num_indices exceeds MAX_ADV_INDEX_DIMS");
    }

    out.meta.num_indices = static_cast<int>(num_indices);
    out.meta.src_ndim = static_cast<int>(src_ndim);

    for (int64_t i = 0; i < src_ndim; ++i) {
        out.meta.src_shape[i] = src_shape_span[i];
    }
    out.meta.src_strides[src_ndim - 1] = 1;
    for (int64_t d = src_ndim - 2; d >= 0; --d) {
        out.meta.src_strides[d] = out.meta.src_strides[d + 1] * src_shape_span[d + 1];
    }

    std::vector<int64_t> broadcast_shape;
    for (int i = 0; i < num_indices; ++i) {
        if (index_tensors[i] != nullptr && index_tensors[i]->numel() > 0) {
            out.meta.is_indexed[i] = 1;
            if (broadcast_shape.empty()) {
                auto s = index_tensors[i]->shape();
                broadcast_shape.assign(s.begin(), s.end());
            }
        } else {
            out.meta.is_indexed[i] = 0;
        }
    }
    if (broadcast_shape.empty()) {
        throw std::runtime_error("AdvancedIndex: at least one index tensor required");
    }

    out.output_shape = broadcast_shape;
    int pass_count = 0;
    for (int i = 0; i < num_indices; ++i) {
        if (!out.meta.is_indexed[i]) {
            out.output_shape.push_back(src_shape_span[i]);
            out.meta.pass_dims[pass_count++] = i;
        }
    }
    for (int64_t i = num_indices; i < src_ndim; ++i) {
        out.output_shape.push_back(src_shape_span[i]);
        out.meta.pass_dims[pass_count++] = static_cast<int>(i);
    }
    out.meta.num_pass_dims = pass_count;

    out.meta.bc_numel = 1;
    for (auto d : broadcast_shape) out.meta.bc_numel *= d;
    out.meta.pass_numel = 1;
    for (int k = 0; k < pass_count; ++k) {
        out.meta.pass_numel *= src_shape_span[out.meta.pass_dims[k]];
    }
    out.total = out.meta.bc_numel * out.meta.pass_numel;
    return out;
}

template<typename T>
auto launch_advanced_index_gather_hip(
    const Tensor& src,
    const Tensor* const* index_tensors,
    int64_t num_indices,
    hipStream_t stream
) -> Tensor {
    auto prep = prepare_advanced_index_hip(src, index_tensors, num_indices);
    Tensor src_contig = src.contiguous();
    Tensor result(prep.output_shape, src.dtype(), src.device());
    if (prep.total == 0) return result;

    std::vector<const int64_t*> host_ptrs(num_indices, nullptr);
    std::vector<Tensor> idx_contig(num_indices);
    for (int i = 0; i < num_indices; ++i) {
        if (prep.meta.is_indexed[i]) {
            idx_contig[i] = index_tensors[i]->contiguous();
            host_ptrs[i] = idx_contig[i].data<int64_t>();
        }
    }

    const int64_t** d_idx_ptrs = nullptr;
    HIP_CHECK(hipMalloc(&d_idx_ptrs, num_indices * sizeof(const int64_t*)));
    HIP_CHECK(hipMemcpyAsync(d_idx_ptrs, host_ptrs.data(),
                             num_indices * sizeof(const int64_t*),
                             hipMemcpyHostToDevice, stream));

    int threads = 256;
    int blocks = static_cast<int>((prep.total + threads - 1) / threads);
    // Use data_ptr() + reinterpret_cast for HIP-native types (__half, hip_bfloat16)
    // that don't have Tensor::data<T>() instantiations in the core library.
    hipLaunchKernelGGL(advanced_index_gather_kernel_hip<T>,
        dim3(blocks), dim3(threads), 0, stream,
        reinterpret_cast<const T*>(src_contig.data_ptr()),
        reinterpret_cast<T*>(result.data_ptr()),
        d_idx_ptrs,
        prep.meta,
        prep.total);
    HIP_POST_LAUNCH_CHECK();

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_idx_ptrs));
    return result;
}

template<typename T>
auto launch_advanced_index_put_hip(
    const Tensor& src,
    const Tensor& values,
    const Tensor* const* index_tensors,
    int64_t num_indices,
    hipStream_t stream
) -> Tensor {
    auto prep = prepare_advanced_index_hip(src, index_tensors, num_indices);
    Tensor result = src.clone();
    Tensor result_contig = result.contiguous();
    Tensor values_contig = values.contiguous();
    if (prep.total == 0) return result_contig;

    std::vector<const int64_t*> host_ptrs(num_indices, nullptr);
    std::vector<Tensor> idx_contig(num_indices);
    for (int i = 0; i < num_indices; ++i) {
        if (prep.meta.is_indexed[i]) {
            idx_contig[i] = index_tensors[i]->contiguous();
            host_ptrs[i] = idx_contig[i].data<int64_t>();
        }
    }

    const int64_t** d_idx_ptrs = nullptr;
    HIP_CHECK(hipMalloc(&d_idx_ptrs, num_indices * sizeof(const int64_t*)));
    HIP_CHECK(hipMemcpyAsync(d_idx_ptrs, host_ptrs.data(),
                             num_indices * sizeof(const int64_t*),
                             hipMemcpyHostToDevice, stream));

    int threads = 256;
    int blocks = static_cast<int>((prep.total + threads - 1) / threads);
    // Use data_ptr() + reinterpret_cast for HIP-native types (__half, hip_bfloat16)
    // that don't have Tensor::data<T>() instantiations in the core library.
    hipLaunchKernelGGL(advanced_index_put_kernel_hip<T>,
        dim3(blocks), dim3(threads), 0, stream,
        reinterpret_cast<T*>(result_contig.data_ptr()),
        reinterpret_cast<const T*>(values_contig.data_ptr()),
        d_idx_ptrs,
        prep.meta,
        prep.total);
    HIP_POST_LAUNCH_CHECK();

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_idx_ptrs));
    return result_contig;
}

}  // namespace

auto advanced_index_rocm_kernel(
    const Tensor& src, const std::vector<Tensor>& indices,
    int64_t num_indices, hipStream_t stream) -> Tensor {
    std::vector<const Tensor*> idx_ptrs(num_indices);
    for (int64_t i = 0; i < num_indices; ++i) idx_ptrs[i] = &indices[i];

    if (src.dtype() == DType::Float32) {
        return launch_advanced_index_gather_hip<float>(src, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Float64) {
        return launch_advanced_index_gather_hip<double>(src, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Int32) {
        return launch_advanced_index_gather_hip<int32_t>(src, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Int64) {
        return launch_advanced_index_gather_hip<int64_t>(src, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Float16) {
        return launch_advanced_index_gather_hip<__half>(src, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::BFloat16) {
        return launch_advanced_index_gather_hip<hip_bfloat16>(src, idx_ptrs.data(), num_indices, stream);
    }
    throw std::runtime_error("AdvancedIndex ROCm: unsupported dtype");
}

auto advanced_index_put_rocm_kernel(
    const Tensor& src, const std::vector<Tensor>& indices,
    const Tensor& values, int64_t num_indices, hipStream_t stream) -> Tensor {
    std::vector<const Tensor*> idx_ptrs(num_indices);
    for (int64_t i = 0; i < num_indices; ++i) idx_ptrs[i] = &indices[i];

    if (src.dtype() == DType::Float32) {
        return launch_advanced_index_put_hip<float>(src, values, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Float64) {
        return launch_advanced_index_put_hip<double>(src, values, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Int32) {
        return launch_advanced_index_put_hip<int32_t>(src, values, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Int64) {
        return launch_advanced_index_put_hip<int64_t>(src, values, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Float16) {
        return launch_advanced_index_put_hip<__half>(src, values, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::BFloat16) {
        return launch_advanced_index_put_hip<hip_bfloat16>(src, values, idx_ptrs.data(), num_indices, stream);
    }
    throw std::runtime_error("AdvancedIndexPut ROCm: unsupported dtype");
}

// ============================================================================
// take_along_dim kernel
// ============================================================================

template<typename T>
__global__ void take_along_dim_hip_kernel(
    const T* __restrict__ input, const int64_t* __restrict__ indices, T* __restrict__ output,
    int64_t numel, int64_t in_dim_size, int64_t idx_dim_size, int64_t inner_size)
{
    HIP_KERNEL_LOOP(i, numel) {
        int64_t outer = i / (idx_dim_size * inner_size);
        int64_t rem = i % (idx_dim_size * inner_size);
        int64_t inner = rem % inner_size;

        int64_t src_idx = indices[i];
        if (src_idx < 0) src_idx += in_dim_size;

        int64_t in_offset = outer * (in_dim_size * inner_size) + src_idx * inner_size + inner;
        output[i] = input[in_offset];
    }
}

auto take_along_dim_hip(const Tensor& input, const Tensor& indices, int64_t dim,
                        hipStream_t stream) -> Tensor {
    auto in_shape = input.shape();
    auto idx_shape = indices.shape();
    int64_t ndim = in_shape.size();
    if (dim < 0) dim += ndim;

    Tensor output(std::vector<int64_t>(idx_shape.begin(), idx_shape.end()),
                  input.dtype(), input.device());
    int64_t numel = indices.numel();
    if (numel == 0) return output;

    int64_t idx_dim_size = idx_shape[dim];
    int64_t in_dim_size = in_shape[dim];
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) inner_size *= idx_shape[d];

    constexpr int BLOCK = 256;
    int blocks = (numel + BLOCK - 1) / BLOCK;
    const int64_t* idx_ptr = indices.data<int64_t>();

    switch (input.dtype()) {
        case DType::Float32:
            take_along_dim_hip_kernel<float><<<blocks, BLOCK, 0, stream>>>(
                input.data<float>(), idx_ptr, output.data<float>(),
                numel, in_dim_size, idx_dim_size, inner_size);
            break;
        case DType::Float64:
            take_along_dim_hip_kernel<double><<<blocks, BLOCK, 0, stream>>>(
                input.data<double>(), idx_ptr, output.data<double>(),
                numel, in_dim_size, idx_dim_size, inner_size);
            break;
        case DType::Int32:
            take_along_dim_hip_kernel<int32_t><<<blocks, BLOCK, 0, stream>>>(
                input.data<int32_t>(), idx_ptr, output.data<int32_t>(),
                numel, in_dim_size, idx_dim_size, inner_size);
            break;
        case DType::Int64:
            take_along_dim_hip_kernel<int64_t><<<blocks, BLOCK, 0, stream>>>(
                input.data<int64_t>(), idx_ptr, output.data<int64_t>(),
                numel, in_dim_size, idx_dim_size, inner_size);
            break;
        case DType::Float16:
            take_along_dim_hip_kernel<__half><<<blocks, BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(input.data_ptr()), idx_ptr,
                reinterpret_cast<__half*>(output.data_ptr()),
                numel, in_dim_size, idx_dim_size, inner_size);
            break;
        case DType::BFloat16:
            take_along_dim_hip_kernel<hip_bfloat16><<<blocks, BLOCK, 0, stream>>>(
                reinterpret_cast<const hip_bfloat16*>(input.data_ptr()), idx_ptr,
                reinterpret_cast<hip_bfloat16*>(output.data_ptr()),
                numel, in_dim_size, idx_dim_size, inner_size);
            break;
        default:
            throw std::runtime_error("take_along_dim ROCm: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());
    return output;
}

// ============================================================================
// masked_scatter kernel — prefix sum on mask, then parallel scatter
// ============================================================================

template<typename T>
__global__ void masked_scatter_write_hip_kernel(
    const T* __restrict__ input, const bool* __restrict__ mask,
    const T* __restrict__ source, const int64_t* __restrict__ prefix_sum,
    T* __restrict__ output, int64_t numel)
{
    HIP_KERNEL_LOOP(i, numel) {
        output[i] = mask[i] ? source[prefix_sum[i]] : input[i];
    }
}

__global__ void mask_to_int64_hip_kernel(const bool* __restrict__ mask, int64_t* __restrict__ out, int64_t n)
{
    HIP_KERNEL_LOOP(i, n) {
        out[i] = mask[i] ? 1 : 0;
    }
}

auto masked_scatter_hip(const Tensor& input, const Tensor& mask,
                        const Tensor& source, hipStream_t stream) -> Tensor {
    int64_t numel = input.numel();
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    if (numel == 0) return output;

    constexpr int BLOCK = 256;
    int blocks = (numel + BLOCK - 1) / BLOCK;

    // Build exclusive prefix sum of mask
    int64_t* d_int_mask = nullptr;
    int64_t* d_prefix = nullptr;
    HIP_CHECK(hipMalloc(&d_int_mask, numel * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_prefix, numel * sizeof(int64_t)));

    mask_to_int64_hip_kernel<<<blocks, BLOCK, 0, stream>>>(mask.data<bool>(), d_int_mask, numel);

    // Use hipcub for exclusive scan
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    hipcub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_int_mask, d_prefix, numel, stream);
    HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
    hipcub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_int_mask, d_prefix, numel, stream);
    HIP_CHECK(hipFree(d_temp));
    HIP_CHECK(hipFree(d_int_mask));

    switch (input.dtype()) {
        case DType::Float32:
            masked_scatter_write_hip_kernel<float><<<blocks, BLOCK, 0, stream>>>(
                input.data<float>(), mask.data<bool>(), source.data<float>(),
                d_prefix, output.data<float>(), numel);
            break;
        case DType::Float64:
            masked_scatter_write_hip_kernel<double><<<blocks, BLOCK, 0, stream>>>(
                input.data<double>(), mask.data<bool>(), source.data<double>(),
                d_prefix, output.data<double>(), numel);
            break;
        case DType::Int32:
            masked_scatter_write_hip_kernel<int32_t><<<blocks, BLOCK, 0, stream>>>(
                input.data<int32_t>(), mask.data<bool>(), source.data<int32_t>(),
                d_prefix, output.data<int32_t>(), numel);
            break;
        case DType::Int64:
            masked_scatter_write_hip_kernel<int64_t><<<blocks, BLOCK, 0, stream>>>(
                input.data<int64_t>(), mask.data<bool>(), source.data<int64_t>(),
                d_prefix, output.data<int64_t>(), numel);
            break;
        default:
            HIP_CHECK(hipFree(d_prefix));
            throw std::runtime_error("masked_scatter ROCm: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipFree(d_prefix));
    return output;
}

// ============================================================================
// tril_indices / triu_indices — native HIP GPU kernels
// ============================================================================

// Cumulative count of lower-triangular indices through row r (inclusive).
// Rows before first_nonempty contribute 0. Each row r contributes
// min(col, r + offset + 1) elements.
__device__ inline int64_t tril_cumcount(int64_t r, int64_t col, int64_t offset,
                                        int64_t first_nonempty) {
    if (r < first_nonempty) return 0;
    int64_t num_rows = r - first_nonempty + 1;
    // Width of first contributing row
    int64_t w_first = first_nonempty + offset + 1;
    // Width of row r
    int64_t w_last = r + offset + 1;
    // Row where width first reaches col (becomes full)
    int64_t full_start = col - offset - 1; // row index where r+offset+1 == col
    if (w_last <= col) {
        // All rows in [first_nonempty, r] are partial
        return num_rows * (w_first + w_last) / 2;
    } else if (w_first >= col) {
        // All rows in [first_nonempty, r] are full
        return num_rows * col;
    } else {
        // Some partial, some full
        int64_t num_partial = full_start - first_nonempty; // rows with width < col
        int64_t partial_sum = num_partial * (w_first + (full_start - 1 + offset + 1)) / 2;
        // full_start is the first full row, but only if full_start + offset + 1 >= col
        // Rows [full_start, r] each contribute col
        int64_t num_full = r - full_start + 1;
        return partial_sum + num_full * col;
    }
}

__global__ void tril_indices_kernel(int64_t* __restrict__ row_out,
                                    int64_t* __restrict__ col_out,
                                    int64_t n, int64_t row, int64_t col,
                                    int64_t offset, int64_t first_nonempty) {
    HIP_KERNEL_LOOP(idx, n) {
        // Binary search: find smallest r such that cumcount(r) > idx
        int64_t lo = first_nonempty, hi = row - 1;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if (tril_cumcount(mid, col, offset, first_nonempty) <= idx) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        int64_t r = lo;
        int64_t prev_count = (r > first_nonempty) ?
            tril_cumcount(r - 1, col, offset, first_nonempty) : 0;
        int64_t c = idx - prev_count;
        row_out[idx] = r;
        col_out[idx] = c;
    }
}

// Cumulative count of upper-triangular indices through row r (inclusive).
// Each row r contributes max(0, col - max(0, r + offset)) elements.
__device__ inline int64_t triu_cumcount(int64_t r, int64_t col, int64_t offset) {
    // Sum over rows [0, r] of max(0, col - max(0, i + offset))
    // Find the last row that contributes: col - max(0, i + offset) > 0
    // => max(0, i + offset) < col => if offset >= 0: i < col - offset
    //                                 if offset < 0: i < col - offset (still)
    // So last contributing row is min(r, col - offset - 1) if offset >= 0,
    // or just r if all rows contribute.

    if (r < 0) return 0;

    // First row start column: max(0, 0 + offset) = max(0, offset)
    // Width of row i: col - max(0, i + offset)
    // For rows where i + offset <= 0, width = col
    // For rows where 0 < i + offset < col, width = col - (i + offset)
    // For rows where i + offset >= col, width = 0

    int64_t total = 0;

    // Phase 1: rows where i + offset <= 0, i.e. i <= -offset. Width = col each.
    int64_t phase1_end = min(r, max(-1LL, -offset));  // last row in phase 1
    if (phase1_end >= 0) {
        total += (phase1_end + 1) * col;
    }

    // Phase 2: rows where 0 < i + offset < col, i.e. max(0, -offset+1) <= i <= min(r, col-offset-1)
    int64_t phase2_start = max(0LL, -offset + 1);
    int64_t phase2_end = min(r, col - offset - 1);
    if (phase2_start <= phase2_end) {
        // Width of row i = col - (i + offset)
        int64_t w_first = col - (phase2_start + offset);
        int64_t w_last = col - (phase2_end + offset);
        int64_t num = phase2_end - phase2_start + 1;
        total += num * (w_first + w_last) / 2;
    }

    // Phase 3: rows where i + offset >= col contribute 0.
    return total;
}

__global__ void triu_indices_kernel(int64_t* __restrict__ row_out,
                                    int64_t* __restrict__ col_out,
                                    int64_t n, int64_t row, int64_t col,
                                    int64_t offset) {
    HIP_KERNEL_LOOP(idx, n) {
        // Binary search: find smallest r such that cumcount(r) > idx
        int64_t lo = 0, hi = row - 1;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if (triu_cumcount(mid, col, offset) <= idx) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        int64_t r = lo;
        int64_t prev_count = (r > 0) ? triu_cumcount(r - 1, col, offset) : 0;
        int64_t local_idx = idx - prev_count;
        int64_t start_col = max(0LL, r + offset);
        int64_t c = start_col + local_idx;
        row_out[idx] = r;
        col_out[idx] = c;
    }
}

auto tril_indices_hip(int64_t row, int64_t col, int64_t offset, hipStream_t stream) -> Tensor {
    // Closed-form total count on host
    int64_t first_nonempty = max(0LL, -offset);
    if (first_nonempty >= row || col <= 0) {
        return tenzor::empty({2, 0}, DType::Int64, Device::rocm(0));
    }

    int64_t n = 0;
    // Partial rows: rows where r + offset + 1 < col
    int64_t full_start = col - offset - 1;  // first row with full width
    if (full_start <= first_nonempty) {
        // All contributing rows are full
        n = (row - first_nonempty) * col;
    } else if (full_start >= row) {
        // All contributing rows are partial: sum of (r + offset + 1) for r in [first_nonempty, row-1]
        int64_t num = row - first_nonempty;
        int64_t w_first = first_nonempty + offset + 1;
        int64_t w_last = (row - 1) + offset + 1;
        n = num * (w_first + w_last) / 2;
    } else {
        // Mixed: partial rows [first_nonempty, full_start-1], full rows [full_start, row-1]
        int64_t num_partial = full_start - first_nonempty;
        int64_t w_first = first_nonempty + offset + 1;
        int64_t w_last_partial = (full_start - 1) + offset + 1;
        int64_t partial_sum = num_partial * (w_first + w_last_partial) / 2;
        int64_t num_full = row - full_start;
        n = partial_sum + num_full * col;
    }

    if (n == 0) return tenzor::empty({2, 0}, DType::Int64, Device::rocm(0));

    Tensor output({2, n}, DType::Int64, Device::rocm(0));
    int64_t* row_ptr = output.data<int64_t>();
    int64_t* col_ptr = row_ptr + n;

    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    hipLaunchKernelGGL(tril_indices_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        row_ptr, col_ptr, n, row, col, offset, first_nonempty);
    HIP_POST_LAUNCH_CHECK();

    return output;
}

auto triu_indices_hip(int64_t row, int64_t col, int64_t offset, hipStream_t stream) -> Tensor {
    // Closed-form total count on host
    // Each row r contributes max(0, col - max(0, r + offset))
    if (row <= 0 || col <= 0) {
        return tenzor::empty({2, 0}, DType::Int64, Device::rocm(0));
    }

    int64_t n = 0;
    // Phase 1: rows where i + offset <= 0 => full width col
    int64_t phase1_end = min(row - 1, max(-1LL, -offset));
    if (phase1_end >= 0) {
        n += (phase1_end + 1) * col;
    }
    // Phase 2: rows where 0 < i + offset < col => width = col - (i + offset)
    int64_t phase2_start = max(0LL, -offset + 1);
    int64_t phase2_end = min(row - 1, col - offset - 1);
    if (phase2_start <= phase2_end) {
        int64_t w_first = col - (phase2_start + offset);
        int64_t w_last = col - (phase2_end + offset);
        int64_t num = phase2_end - phase2_start + 1;
        n += num * (w_first + w_last) / 2;
    }
    // Phase 3: rows where i + offset >= col contribute 0.

    if (n == 0) return tenzor::empty({2, 0}, DType::Int64, Device::rocm(0));

    Tensor output({2, n}, DType::Int64, Device::rocm(0));
    int64_t* row_ptr = output.data<int64_t>();
    int64_t* col_ptr = row_ptr + n;

    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    hipLaunchKernelGGL(triu_indices_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        row_ptr, col_ptr, n, row, col, offset);
    HIP_POST_LAUNCH_CHECK();

    return output;
}

} // namespace rocm
} // namespace tenzor
