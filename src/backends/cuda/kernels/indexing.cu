/**
 * @file indexing.cu
 * @brief CUDA kernels for indexing operations
 *
 * Implements native CUDA kernels for:
 * - index_select: Select elements along a dimension using indices
 * - gather: Gather values along an axis according to indices
 * - scatter: Scatter values into a tensor at specified indices
 * - masked_select: Select elements based on a boolean mask
 * - masked_fill: Fill elements based on a boolean mask
 * - where: Conditional selection between two tensors
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/caching_allocator.hpp"
#include "tenzor/backend/backend.hpp"
#include "cuda_launch_utils.cuh"
#include <stdexcept>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <cub/cub.cuh>
#include "tenzor/utils/config.hpp"
#include <thrust/iterator/counting_iterator.h>
#include "cuda_common.cuh"
#include "cuda_nan_helpers.cuh"

#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"  // broadcast_to (F2)
#include "tenzor/core/shape.hpp"     // broadcast_shapes (F2)

namespace tenzor {
namespace cuda {

// Centralized error checking
#include "../cuda_error.hpp"

constexpr int BLOCK_SIZE = 256;

inline int get_num_blocks(int64_t n, int block_size = BLOCK_SIZE) {
    return compute_grid_size(n, block_size);
}

// Occupancy-based kernel launch: replaces hardcoded BLOCK_SIZE with per-kernel optimal config
#define LAUNCH_KERNEL(kernel, n, stream, ...) \
    do { \
        auto [grid_, block_] = optimal_launch_config(kernel, n); \
        kernel<<<grid_, block_, 0, stream>>>(__VA_ARGS__); \
        CUDA_CHECK(cudaGetLastError()); \
    } while(0)

// ============================================================================
// index_select kernel
// ============================================================================

template<typename T, typename IndexT>
__global__ void index_select_kernel_impl(
    const T* input,
    const IndexT* indices,
    T* output,
    int64_t num_indices,
    int64_t dim_size,
    int64_t outer_size,
    int64_t inner_size,
    int64_t total_output,
    int* error_flag) {

    TENZOR_CUDA_KERNEL_LOOP(idx, total_output) {
        // Decompose output index into (outer, index_pos, inner)
        int64_t inner_idx = idx % inner_size;
        int64_t temp = idx / inner_size;
        int64_t index_pos = temp % num_indices;
        int64_t outer_idx = temp / num_indices;

        // Get the actual index value with bounds checking
        int64_t selected_idx = static_cast<int64_t>(indices[index_pos]);
        if (selected_idx < 0) selected_idx += dim_size;
        if (selected_idx < 0 || selected_idx >= dim_size) {
            atomicExch(error_flag, 1);
            return;
        }

        // Compute input offset
        int64_t input_offset = outer_idx * dim_size * inner_size +
                               selected_idx * inner_size +
                               inner_idx;

        output[idx] = input[input_offset];
    }
}

auto index_select_kernel(const Tensor& input, int64_t dim, const Tensor& index,
                         cudaStream_t stream) -> Tensor {
    // Normalize dimension
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;

    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("index_select: dimension out of range");
    }

    // Compute sizes
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input.shape()[i];
    }

    int64_t dim_size = input.shape()[dim];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= input.shape()[i];
    }

    int64_t num_indices = index.numel();

    // Create output shape
    std::vector<int64_t> output_shape(input.shape().begin(), input.shape().end());
    output_shape[dim] = num_indices;

    Tensor output(output_shape, input.dtype(), input.device());
    int64_t total_output = output.numel();

    if (total_output == 0) return output;

    int num_blocks = get_num_blocks(total_output);

    bool idx_is_int32 = (index.dtype() == DType::Int32);
    bool idx_is_int64 = (index.dtype() == DType::Int64);
    if (!idx_is_int32 && !idx_is_int64) {
        throw std::invalid_argument("index_select: index must be Int32 or Int64");
    }

    // Device-side OOB error flag
    CudaBuffer error_buf(sizeof(int));
    CUDA_CHECK(cudaMemsetAsync(error_buf.as<int>(), 0, sizeof(int), stream));

    #define LAUNCH_INDEX_SELECT(T) \
        if (idx_is_int32) \
            index_select_kernel_impl<T, int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>( \
                input.data<T>(), index.data<int32_t>(), output.data<T>(), \
                num_indices, dim_size, outer_size, inner_size, total_output, \
                error_buf.as<int>()); \
        else \
            index_select_kernel_impl<T, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>( \
                input.data<T>(), index.data<int64_t>(), output.data<T>(), \
                num_indices, dim_size, outer_size, inner_size, total_output, \
                error_buf.as<int>()); \
        CUDA_CHECK(cudaGetLastError())

    switch (input.dtype()) {
        case DType::Float32: LAUNCH_INDEX_SELECT(float); break;
        case DType::Float64: LAUNCH_INDEX_SELECT(double); break;
        case DType::Int32:   LAUNCH_INDEX_SELECT(int32_t); break;
        case DType::Int64:   LAUNCH_INDEX_SELECT(int64_t); break;
        case DType::Int8:    LAUNCH_INDEX_SELECT(int8_t); break;
        case DType::UInt8:   LAUNCH_INDEX_SELECT(uint8_t); break;
        case DType::Int16:   LAUNCH_INDEX_SELECT(int16_t); break;
        case DType::UInt16:  LAUNCH_INDEX_SELECT(uint16_t); break;
        case DType::UInt32:  LAUNCH_INDEX_SELECT(uint32_t); break;
        case DType::UInt64:  LAUNCH_INDEX_SELECT(uint64_t); break;
        case DType::Float16:
            if (idx_is_int32)
                index_select_kernel_impl<__half, int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __half*>(input.data_ptr()),
                    index.data<int32_t>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    num_indices, dim_size, outer_size, inner_size, total_output,
                    error_buf.as<int>());
            else
                index_select_kernel_impl<__half, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __half*>(input.data_ptr()),
                    index.data<int64_t>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    num_indices, dim_size, outer_size, inner_size, total_output,
                    error_buf.as<int>());
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::BFloat16:
            if (idx_is_int32)
                index_select_kernel_impl<__nv_bfloat16, int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                    index.data<int32_t>(),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    num_indices, dim_size, outer_size, inner_size, total_output,
                    error_buf.as<int>());
            else
                index_select_kernel_impl<__nv_bfloat16, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                    index.data<int64_t>(),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    num_indices, dim_size, outer_size, inner_size, total_output,
                    error_buf.as<int>());
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::Complex64:
            // index_select is pure data movement; treat complex as float2.
            if (idx_is_int32)
                index_select_kernel_impl<float2, int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const float2*>(input.data_ptr()),
                    index.data<int32_t>(),
                    reinterpret_cast<float2*>(output.data_ptr()),
                    num_indices, dim_size, outer_size, inner_size, total_output,
                    error_buf.as<int>());
            else
                index_select_kernel_impl<float2, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const float2*>(input.data_ptr()),
                    index.data<int64_t>(),
                    reinterpret_cast<float2*>(output.data_ptr()),
                    num_indices, dim_size, outer_size, inner_size, total_output,
                    error_buf.as<int>());
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::Complex128:
            if (idx_is_int32)
                index_select_kernel_impl<double2, int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const double2*>(input.data_ptr()),
                    index.data<int32_t>(),
                    reinterpret_cast<double2*>(output.data_ptr()),
                    num_indices, dim_size, outer_size, inner_size, total_output,
                    error_buf.as<int>());
            else
                index_select_kernel_impl<double2, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const double2*>(input.data_ptr()),
                    index.data<int64_t>(),
                    reinterpret_cast<double2*>(output.data_ptr()),
                    num_indices, dim_size, outer_size, inner_size, total_output,
                    error_buf.as<int>());
            CUDA_CHECK(cudaGetLastError());
            break;
        default:
            throw std::runtime_error("index_select: unsupported dtype");
    }

    #undef LAUNCH_INDEX_SELECT

    // Check for out-of-bounds index errors
    int host_error = 0;
    CUDA_CHECK(cudaMemcpyAsync(&host_error, error_buf.as<int>(), sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (host_error) {
        throw std::out_of_range(
            "index_select: index out of range for dimension of size " +
            std::to_string(dim_size));
    }

    return output;
}

// ============================================================================
// gather kernel
// ============================================================================

template<typename T, typename IndexT>
__global__ void gather_kernel_impl(
    const T* input,
    const IndexT* indices,
    T* output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size,
    int64_t index_dim_size,
    int64_t total_output,
    int* error_flag) {

    TENZOR_CUDA_KERNEL_LOOP(idx, total_output) {
        // Decompose output index into (outer, index_pos, inner)
        int64_t inner_idx = idx % inner_size;
        int64_t temp = idx / inner_size;
        int64_t index_pos = temp % index_dim_size;
        int64_t outer_idx = temp / index_dim_size;

        // Get the index value at this position
        int64_t index_offset = outer_idx * index_dim_size * inner_size +
                               index_pos * inner_size + inner_idx;
        int64_t gather_idx = static_cast<int64_t>(indices[index_offset]);
        if (gather_idx < 0) gather_idx += dim_size;
        if (gather_idx < 0 || gather_idx >= dim_size) {
            atomicExch(error_flag, 1);
            continue;
        }

        // Compute input offset
        int64_t input_offset = outer_idx * dim_size * inner_size +
                               gather_idx * inner_size +
                               inner_idx;

        output[idx] = input[input_offset];
    }
}

auto gather_kernel(const Tensor& input, int64_t dim, const Tensor& index,
                   cudaStream_t stream) -> Tensor {
    // Normalize dimension
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;

    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("gather: dimension out of range");
    }

    // Output has the same shape as index
    std::vector<int64_t> output_shape(index.shape().begin(), index.shape().end());
    Tensor output(output_shape, input.dtype(), input.device());

    int64_t total_output = output.numel();
    if (total_output == 0) return output;

    CudaBuffer error_buf(sizeof(int));
    CUDA_CHECK(cudaMemsetAsync(error_buf.as<int>(), 0, sizeof(int), stream));

    // Compute sizes
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input.shape()[i];
    }

    int64_t dim_size = input.shape()[dim];
    int64_t index_dim_size = index.shape()[dim];

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= input.shape()[i];
    }

    int num_blocks = get_num_blocks(total_output);

    bool idx_is_int32 = (index.dtype() == DType::Int32);
    bool idx_is_int64 = (index.dtype() == DType::Int64);
    if (!idx_is_int32 && !idx_is_int64) {
        throw std::invalid_argument("gather: index must be Int32 or Int64");
    }

    #define LAUNCH_GATHER(T) \
        if (idx_is_int32) \
            gather_kernel_impl<T, int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>( \
                input.data<T>(), index.data<int32_t>(), output.data<T>(), \
                outer_size, dim_size, inner_size, index_dim_size, total_output, \
                error_buf.as<int>()); \
        else \
            gather_kernel_impl<T, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>( \
                input.data<T>(), index.data<int64_t>(), output.data<T>(), \
                outer_size, dim_size, inner_size, index_dim_size, total_output, \
                error_buf.as<int>()); \
        CUDA_CHECK(cudaGetLastError())

    switch (input.dtype()) {
        case DType::Float32: LAUNCH_GATHER(float); break;
        case DType::Float64: LAUNCH_GATHER(double); break;
        case DType::Int32:   LAUNCH_GATHER(int32_t); break;
        case DType::Int64:   LAUNCH_GATHER(int64_t); break;
        case DType::Int8:    LAUNCH_GATHER(int8_t); break;
        case DType::UInt8:   LAUNCH_GATHER(uint8_t); break;
        case DType::Int16:   LAUNCH_GATHER(int16_t); break;
        case DType::UInt16:  LAUNCH_GATHER(uint16_t); break;
        case DType::UInt32:  LAUNCH_GATHER(uint32_t); break;
        case DType::UInt64:  LAUNCH_GATHER(uint64_t); break;
        case DType::Float16:
            if (idx_is_int32)
                gather_kernel_impl<__half, int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __half*>(input.data_ptr()),
                    index.data<int32_t>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_output,
                    error_buf.as<int>());
            else
                gather_kernel_impl<__half, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __half*>(input.data_ptr()),
                    index.data<int64_t>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_output,
                    error_buf.as<int>());
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::BFloat16:
            if (idx_is_int32)
                gather_kernel_impl<__nv_bfloat16, int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                    index.data<int32_t>(),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_output,
                    error_buf.as<int>());
            else
                gather_kernel_impl<__nv_bfloat16, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                    index.data<int64_t>(),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_output,
                    error_buf.as<int>());
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::Complex64:
            // gather is pure data movement; treat complex as float2.
            if (idx_is_int32)
                gather_kernel_impl<float2, int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const float2*>(input.data_ptr()), index.data<int32_t>(),
                    reinterpret_cast<float2*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_output, error_buf.as<int>());
            else
                gather_kernel_impl<float2, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const float2*>(input.data_ptr()), index.data<int64_t>(),
                    reinterpret_cast<float2*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_output, error_buf.as<int>());
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::Complex128:
            if (idx_is_int32)
                gather_kernel_impl<double2, int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const double2*>(input.data_ptr()), index.data<int32_t>(),
                    reinterpret_cast<double2*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_output, error_buf.as<int>());
            else
                gather_kernel_impl<double2, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const double2*>(input.data_ptr()), index.data<int64_t>(),
                    reinterpret_cast<double2*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_output, error_buf.as<int>());
            CUDA_CHECK(cudaGetLastError());
            break;
        default:
            throw std::runtime_error("gather: unsupported dtype");
    }

    #undef LAUNCH_GATHER

    CUDA_CHECK(cudaGetLastError());

    int host_error = 0;
    CUDA_CHECK(cudaMemcpyAsync(&host_error, error_buf.as<int>(), sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (host_error) {
        throw std::out_of_range("gather: index out of range");
    }

    return output;
}

// ============================================================================
// scatter kernel
// ============================================================================
// Scatter uses two separate kernel launches (copy + scatter_values) instead of
// a single combined kernel. A combined kernel with __syncthreads() between
// grid-stride loops only synchronizes within a block, not across blocks —
// causing undefined behavior for tensors larger than a single block.
// Two separate launches provide a full device-wide synchronization barrier.

// Kernel 1: Copy input to output
template<typename T>
__global__ void copy_kernel_impl(const T* input, T* output, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = input[idx];
    }
}

// Kernel 2: Scatter src values into output at indexed positions
template<typename T, typename IndexT>
__global__ void scatter_values_kernel_impl(
    const IndexT* indices,
    const T* src,
    T* output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size,
    int64_t index_dim_size,
    int64_t total_scatter,
    int* error_flag) {

    TENZOR_CUDA_KERNEL_LOOP(idx, total_scatter) {
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
            continue;
        }

        int64_t output_offset = outer_idx * dim_size * inner_size +
                                scatter_idx * inner_size +
                                inner_idx;

        output[output_offset] = src[idx];
    }
}

auto scatter_kernel(const Tensor& input, int64_t dim, const Tensor& index,
                    const Tensor& src, cudaStream_t stream) -> Tensor {
    // Normalize dimension
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;

    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("scatter: dimension out of range");
    }

    // Output has the same shape as input
    std::vector<int64_t> output_shape(input.shape().begin(), input.shape().end());
    Tensor output(output_shape, input.dtype(), input.device());

    int64_t total_input = input.numel();
    int64_t total_scatter = index.numel();

    if (total_input == 0) return output;

    // Compute sizes
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input.shape()[i];
    }

    int64_t dim_size = input.shape()[dim];
    int64_t index_dim_size = index.shape()[dim];

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= input.shape()[i];
    }

    bool idx_is_int32 = (index.dtype() == DType::Int32);
    bool idx_is_int64 = (index.dtype() == DType::Int64);
    if (!idx_is_int32 && !idx_is_int64) {
        throw std::invalid_argument("scatter: index must be Int32 or Int64");
    }

    int num_blocks_copy = get_num_blocks(total_input);
    int num_blocks_scatter = get_num_blocks(total_scatter);

    // Device-side OOB error flag
    CudaBuffer error_buf(sizeof(int));
    CUDA_CHECK(cudaMemsetAsync(error_buf.as<int>(), 0, sizeof(int), stream));

    #define LAUNCH_SCATTER(T) \
        copy_kernel_impl<T><<<num_blocks_copy, BLOCK_SIZE, 0, stream>>>( \
            input.data<T>(), output.data<T>(), total_input); \
        CUDA_CHECK(cudaGetLastError()); \
        if (idx_is_int32) \
            scatter_values_kernel_impl<T, int32_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>( \
                index.data<int32_t>(), src.data<T>(), output.data<T>(), \
                outer_size, dim_size, inner_size, index_dim_size, total_scatter, \
                error_buf.as<int>()); \
        else \
            scatter_values_kernel_impl<T, int64_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>( \
                index.data<int64_t>(), src.data<T>(), output.data<T>(), \
                outer_size, dim_size, inner_size, index_dim_size, total_scatter, \
                error_buf.as<int>()); \
        CUDA_CHECK(cudaGetLastError())

    switch (input.dtype()) {
        case DType::Float32: LAUNCH_SCATTER(float); break;
        case DType::Float64: LAUNCH_SCATTER(double); break;
        case DType::Int32:   LAUNCH_SCATTER(int32_t); break;
        case DType::Int64:   LAUNCH_SCATTER(int64_t); break;
        case DType::Int8:    LAUNCH_SCATTER(int8_t); break;
        case DType::UInt8:   LAUNCH_SCATTER(uint8_t); break;
        case DType::Int16:   LAUNCH_SCATTER(int16_t); break;
        case DType::UInt16:  LAUNCH_SCATTER(uint16_t); break;
        case DType::UInt32:  LAUNCH_SCATTER(uint32_t); break;
        case DType::UInt64:  LAUNCH_SCATTER(uint64_t); break;
        case DType::Float16:
            copy_kernel_impl<__half><<<num_blocks_copy, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const __half*>(input.data_ptr()),
                reinterpret_cast<__half*>(output.data_ptr()), total_input);
            CUDA_CHECK(cudaGetLastError());
            if (idx_is_int32)
                scatter_values_kernel_impl<__half, int32_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>(
                    index.data<int32_t>(),
                    reinterpret_cast<const __half*>(src.data_ptr()),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_scatter,
                    error_buf.as<int>());
            else
                scatter_values_kernel_impl<__half, int64_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>(
                    index.data<int64_t>(),
                    reinterpret_cast<const __half*>(src.data_ptr()),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_scatter,
                    error_buf.as<int>());
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::BFloat16:
            copy_kernel_impl<__nv_bfloat16><<<num_blocks_copy, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), total_input);
            CUDA_CHECK(cudaGetLastError());
            if (idx_is_int32)
                scatter_values_kernel_impl<__nv_bfloat16, int32_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>(
                    index.data<int32_t>(),
                    reinterpret_cast<const __nv_bfloat16*>(src.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_scatter,
                    error_buf.as<int>());
            else
                scatter_values_kernel_impl<__nv_bfloat16, int64_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>(
                    index.data<int64_t>(),
                    reinterpret_cast<const __nv_bfloat16*>(src.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_scatter,
                    error_buf.as<int>());
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::Complex64:
            copy_kernel_impl<float2><<<num_blocks_copy, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const float2*>(input.data_ptr()),
                reinterpret_cast<float2*>(output.data_ptr()), total_input);
            CUDA_CHECK(cudaGetLastError());
            if (idx_is_int32)
                scatter_values_kernel_impl<float2, int32_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>(
                    index.data<int32_t>(), reinterpret_cast<const float2*>(src.data_ptr()),
                    reinterpret_cast<float2*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_scatter, error_buf.as<int>());
            else
                scatter_values_kernel_impl<float2, int64_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>(
                    index.data<int64_t>(), reinterpret_cast<const float2*>(src.data_ptr()),
                    reinterpret_cast<float2*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_scatter, error_buf.as<int>());
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::Complex128:
            copy_kernel_impl<double2><<<num_blocks_copy, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const double2*>(input.data_ptr()),
                reinterpret_cast<double2*>(output.data_ptr()), total_input);
            CUDA_CHECK(cudaGetLastError());
            if (idx_is_int32)
                scatter_values_kernel_impl<double2, int32_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>(
                    index.data<int32_t>(), reinterpret_cast<const double2*>(src.data_ptr()),
                    reinterpret_cast<double2*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_scatter, error_buf.as<int>());
            else
                scatter_values_kernel_impl<double2, int64_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>(
                    index.data<int64_t>(), reinterpret_cast<const double2*>(src.data_ptr()),
                    reinterpret_cast<double2*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_scatter, error_buf.as<int>());
            CUDA_CHECK(cudaGetLastError());
            break;
        default:
            throw std::runtime_error("scatter: unsupported dtype");
    }

    #undef LAUNCH_SCATTER

    // Check for out-of-bounds index errors
    int host_error = 0;
    CUDA_CHECK(cudaMemcpyAsync(&host_error, error_buf.as<int>(), sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (host_error) {
        throw std::out_of_range(
            "scatter: index out of range for dimension of size " +
            std::to_string(dim_size));
    }

    return output;
}

// ============================================================================
// scatter_add kernel — uses atomicAdd for overlapping indices
// ============================================================================

// Warp-level atomic helper: reduces values within a warp that target the same
// output offset, then one thread per unique target does a single atomicAdd.
// This reduces atomic contention by up to 32x under high-conflict patterns.
//
// CUDA correctness notes (#42):
// 1. The caller is invoked from inside a TENZOR_CUDA_KERNEL_LOOP grid-stride
//    loop. Threads with idx >= n simply skip the body and never call this
//    function, so we cannot use FULL_MASK in shuffles — only __activemask()
//    is safe (the set of lanes that are at this instruction).
// 2. The earlier version used `break` to exit the for-loop once a lane's
//    offset was processed. Other lanes that hadn't yet matched continued the
//    loop, calling more shuffles, but the broken-out lanes couldn't
//    participate — deadlock. Use a per-lane `done` flag so processed lanes
//    stay in the loop and continue participating.
// 3. The position-based advance `peer_offset = 32 - __clz(match_mask)` skips
//    over unprocessed lanes that sit between matching groups (e.g.,
//    [A, A, B, A] would skip past lane 2's B). Use ballot+ffs to find the
//    next unprocessed lane.
template<typename T>
__device__ void warp_reduce_atomic_add(T* output, int64_t output_offset, T value) {
    unsigned active_mask = __activemask();
    unsigned lane = threadIdx.x & 31;
    bool done = false;

    // Find lanes in this warp targeting the same output_offset
    // Use ballot to group matching lanes
    for (unsigned peer_offset = 0; peer_offset < 32; ) {
        // Skip if peer_offset isn't an active lane
        if (((active_mask >> peer_offset) & 1u) == 0) {
            ++peer_offset;
            continue;
        }
        // Broadcast the target offset from the leader lane
        int64_t leader_offset = __shfl_sync(active_mask, output_offset, peer_offset);
        // Only consider active+unprocessed lanes that match the leader.
        unsigned match_mask = __ballot_sync(active_mask,
                                            !done && output_offset == leader_offset);

        if (!done && output_offset == leader_offset) {
            // Warp-level reduction via shuffle
            T reduced = value;
            for (int delta = 16; delta > 0; delta >>= 1) {
                T shuffled = __shfl_down_sync(match_mask, reduced, delta);
                if ((lane & (delta * 2 - 1)) < static_cast<unsigned>(delta)) {
                    reduced += shuffled;
                }
            }
            // Lowest matching lane does the atomic
            unsigned leader_lane = __ffs(match_mask) - 1;
            if (lane == leader_lane) {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    atomicAdd(&output[output_offset], reduced);
                } else if constexpr (std::is_same_v<T, int32_t>) {
                    atomicAdd(reinterpret_cast<int*>(&output[output_offset]),
                              static_cast<int>(reduced));
                } else if constexpr (std::is_same_v<T, int64_t>) {
                    unsigned long long* addr = reinterpret_cast<unsigned long long*>(&output[output_offset]);
                    // Use atomicCAS for initial read to avoid data race (non-atomic *addr is UB)
                    unsigned long long old_val = atomicCAS(addr, 0ULL, 0ULL);
                    unsigned long long assumed;
                    do {
                        assumed = old_val;
                        unsigned long long desired = static_cast<unsigned long long>(
                            static_cast<int64_t>(assumed) + reduced);
                        old_val = atomicCAS(addr, assumed, desired);
                    } while (assumed != old_val);
                } else if constexpr (std::is_same_v<T, __half>) {
#if __CUDA_ARCH__ >= 700
                    atomicAdd(&output[output_offset], reduced);
#else
                    // CAS-based fallback for SM < 70
                    float val = __half2float(reduced);
                    unsigned int* addr = reinterpret_cast<unsigned int*>(
                        &output[output_offset & ~int64_t(1)]);
                    unsigned int old_val, new_val;
                    int lane = output_offset & 1;
                    do {
                        old_val = *addr;
                        __half* h = reinterpret_cast<__half*>(&old_val);
                        // W.7: NaN-preserving conversion.
                        __half result = ::tenzor::cuda::safe_f2half(::tenzor::cuda::safe_half2f(h[lane]) + val);
                        new_val = old_val;
                        reinterpret_cast<__half*>(&new_val)[lane] = result;
                    } while (atomicCAS(addr, old_val, new_val) != old_val);
#endif
                } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
#if __CUDA_ARCH__ >= 800
                    atomicAdd(&output[output_offset], reduced);
#else
                    float val = __bfloat162float(reduced);
                    unsigned int* addr = reinterpret_cast<unsigned int*>(
                        &output[output_offset & ~int64_t(1)]);
                    unsigned int old_val, new_val;
                    int lane = output_offset & 1;
                    do {
                        old_val = *addr;
                        __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(&old_val);
                        // W.7: NaN-preserving conversion.
                        __nv_bfloat16 result = ::tenzor::cuda::safe_f2bf16(::tenzor::cuda::safe_bf162f(h[lane]) + val);
                        new_val = old_val;
                        reinterpret_cast<__nv_bfloat16*>(&new_val)[lane] = result;
                    } while (atomicCAS(addr, old_val, new_val) != old_val);
#endif
                } else {
                    atomicAdd(&output[output_offset], reduced);
                }
            }
            done = true;
        }
        // Find the next active+unprocessed lane to use as leader.
        unsigned undone_mask = __ballot_sync(active_mask, !done);
        if (undone_mask == 0) break;
        peer_offset = __ffs(undone_mask) - 1;
    }
}

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

    TENZOR_CUDA_KERNEL_LOOP(idx, total_scatter) {
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
            continue;
        }

        int64_t output_offset = outer_idx * dim_size * inner_size +
                                scatter_idx * inner_size +
                                inner_idx;

        // Use plain atomicAdd. The previous warp_reduce_atomic_add helper
        // had multiple subtle bugs (FULL_MASK shuffles inside a grid-stride
        // loop where some lanes are inactive, position-based advance that
        // skipped non-contiguous unprocessed lanes) that caused either
        // hangs or off-by-N missed accumulations on CUDA. Plain atomicAdd
        // is correct in all cases — the warp-reduce optimization can be
        // re-introduced later with proper __activemask handling. (#42)
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            atomicAdd(&output[output_offset], src[idx]);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            atomicAdd(reinterpret_cast<int*>(&output[output_offset]),
                      static_cast<int>(src[idx]));
        } else if constexpr (std::is_same_v<T, int64_t>) {
            unsigned long long* addr = reinterpret_cast<unsigned long long*>(&output[output_offset]);
            unsigned long long old_val = atomicCAS(addr, 0ULL, 0ULL);
            unsigned long long assumed;
            T add_val = src[idx];
            do {
                assumed = old_val;
                unsigned long long desired = static_cast<unsigned long long>(
                    static_cast<int64_t>(assumed) + add_val);
                old_val = atomicCAS(addr, assumed, desired);
            } while (assumed != old_val);
        } else {
            atomicAdd(&output[output_offset], src[idx]);
        }
    }
}

// Helper kernel: compute flat output indices for each scatter element
template<typename IndexT>
__global__ void compute_flat_scatter_indices_kernel(
    const IndexT* indices,
    int64_t* flat_indices,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size,
    int64_t index_dim_size,
    int64_t total_scatter,
    int* error_flag) {
    TENZOR_CUDA_KERNEL_LOOP(idx, total_scatter) {
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
            flat_indices[idx] = -1;  // sentinel for invalid
            continue;
        }

        flat_indices[idx] = outer_idx * dim_size * inner_size +
                            scatter_idx * inner_size +
                            inner_idx;
    }
}

// Helper kernel: deterministic segment accumulate after sorting by output index
template<typename T>
__global__ void deterministic_segment_accumulate_kernel(
    const int64_t* sorted_flat_indices,
    const T* sorted_values,
    T* output,
    int64_t total_scatter) {
    // Each thread processes one segment boundary
    TENZOR_CUDA_KERNEL_LOOP(idx, total_scatter) {
        int64_t flat_idx = sorted_flat_indices[idx];
        if (flat_idx < 0) continue;  // skip invalid indices

        // Check if this is the start of a new segment
        bool is_segment_start = (idx == 0) || (sorted_flat_indices[idx - 1] != flat_idx);
        if (!is_segment_start) continue;

        // Accumulate all values in this segment sequentially
        T sum = sorted_values[idx];
        for (int64_t j = idx + 1; j < total_scatter && sorted_flat_indices[j] == flat_idx; ++j) {
            sum += sorted_values[j];
        }
        output[flat_idx] += sum;
    }
}

auto scatter_add_kernel(const Tensor& input, int64_t dim, const Tensor& index,
                        const Tensor& src, cudaStream_t stream) -> Tensor {
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

    // Step 1: Copy input to output (raw byte copy — works for all dtypes)
    CUDA_CHECK(cudaMemcpyAsync(output.data_ptr(), input.data_ptr(),
                               total_input * dtype_size(input.dtype()),
                               cudaMemcpyDeviceToDevice, stream));

    if (tenzor::is_deterministic()) {
        // Deterministic scatter_add: sort by output index, then segment-reduce
        if (total_scatter == 0) return output;

        constexpr int BLOCK_SIZE_DET = 256;
        int num_blocks_det = static_cast<int>((total_scatter + BLOCK_SIZE_DET - 1) / BLOCK_SIZE_DET);

        // Step 1: Compute flat output indices
        CudaBuffer flat_indices_buf(total_scatter * sizeof(int64_t));
        CudaBuffer error_buf_det(sizeof(int));
        CUDA_CHECK(cudaMemsetAsync(error_buf_det.as<int>(), 0, sizeof(int), stream));

        if (idx_is_int32) {
            compute_flat_scatter_indices_kernel<int32_t><<<num_blocks_det, BLOCK_SIZE_DET, 0, stream>>>(
                index.data<int32_t>(), flat_indices_buf.as<int64_t>(),
                outer_size, dim_size, inner_size, index_dim_size, total_scatter,
                error_buf_det.as<int>());
        } else {
            compute_flat_scatter_indices_kernel<int64_t><<<num_blocks_det, BLOCK_SIZE_DET, 0, stream>>>(
                index.data<int64_t>(), flat_indices_buf.as<int64_t>(),
                outer_size, dim_size, inner_size, index_dim_size, total_scatter,
                error_buf_det.as<int>());
        }
        CUDA_CHECK(cudaGetLastError());

        // Check for OOB errors
        int host_error_det = 0;
        CUDA_CHECK(cudaMemcpyAsync(&host_error_det, error_buf_det.as<int>(), sizeof(int),
                                    cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (host_error_det) {
            throw std::out_of_range(
                "scatter_add: index out of range for dimension of size " +
                std::to_string(dim_size));
        }

        // Step 2: Sort (flat_index, value) pairs by flat_index using CUB
        CudaBuffer sorted_indices_buf(total_scatter * sizeof(int64_t));

        // Determine temp storage for CUB sort
        size_t temp_storage_bytes = 0;

        auto do_sort = [&](auto* src_vals, auto* out_vals) {
            using ValT = std::remove_const_t<std::remove_pointer_t<decltype(src_vals)>>;
            CudaBuffer sorted_vals_buf(total_scatter * sizeof(ValT));

            {
                void* d_temp = nullptr;
                cub::DeviceRadixSort::SortPairs(
                    d_temp, temp_storage_bytes,
                    flat_indices_buf.as<int64_t>(), sorted_indices_buf.as<int64_t>(),
                    static_cast<const ValT*>(src_vals), sorted_vals_buf.as<ValT>(),
                    static_cast<int>(total_scatter), 0, static_cast<int>(sizeof(int64_t) * 8), stream);
            }

            CudaBuffer temp_storage(temp_storage_bytes);

            cub::DeviceRadixSort::SortPairs(
                temp_storage.as<void>(), temp_storage_bytes,
                flat_indices_buf.as<int64_t>(), sorted_indices_buf.as<int64_t>(),
                src_vals, sorted_vals_buf.as<ValT>(),
                static_cast<int>(total_scatter), 0, static_cast<int>(sizeof(int64_t) * 8), stream);
            CUDA_CHECK(cudaGetLastError());

            // Step 3: Deterministic segment accumulate
            deterministic_segment_accumulate_kernel<ValT><<<num_blocks_det, BLOCK_SIZE_DET, 0, stream>>>(
                sorted_indices_buf.as<int64_t>(),
                sorted_vals_buf.as<ValT>(),
                out_vals,  // output tensor data
                total_scatter);
            CUDA_CHECK(cudaGetLastError());
        };

        if (input.dtype() == DType::Float32) {
            do_sort(src.data<float>(), output.data<float>());
        } else if (input.dtype() == DType::Float64) {
            do_sort(src.data<double>(), output.data<double>());
        } else if (input.dtype() == DType::Int32) {
            do_sort(src.data<int32_t>(), output.data<int32_t>());
        } else if (input.dtype() == DType::Int64) {
            do_sort(src.data<int64_t>(), output.data<int64_t>());
        } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
            // Upcast to float32 for deterministic path
            Tensor src_f32 = src.to(DType::Float32);
            Tensor output_f32 = output.to(DType::Float32);
            do_sort(src_f32.data<float>(), output_f32.data<float>());
            // Copy back
            Tensor result = output_f32.to(input.dtype());
            CUDA_CHECK(cudaMemcpyAsync(output.data_ptr(), result.data_ptr(),
                                        output.numel() * dtype_size(input.dtype()),
                                        cudaMemcpyDeviceToDevice, stream));
        } else if (input.dtype() == DType::Int8 || input.dtype() == DType::Int16 ||
                   input.dtype() == DType::UInt8 || input.dtype() == DType::UInt16 ||
                   input.dtype() == DType::UInt32 || input.dtype() == DType::UInt64) {
            // No native deterministic accumulate for 8/16-bit/unsigned ints; upcast to Int64.
            Tensor src_i64 = src.to(DType::Int64);
            Tensor output_i64 = output.to(DType::Int64);
            do_sort(src_i64.data<int64_t>(), output_i64.data<int64_t>());
            Tensor result = output_i64.to(input.dtype());
            CUDA_CHECK(cudaMemcpyAsync(output.data_ptr(), result.data_ptr(),
                                        output.numel() * dtype_size(input.dtype()),
                                        cudaMemcpyDeviceToDevice, stream));
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        return output;
    }

    // Step 2: Scatter-add with atomicAdd (non-deterministic path)
    if (total_scatter == 0) return output;
    int num_blocks_scatter = get_num_blocks(total_scatter);

    // Device-side OOB error flag
    CudaBuffer error_buf(sizeof(int));
    CUDA_CHECK(cudaMemsetAsync(error_buf.as<int>(), 0, sizeof(int), stream));

    #define LAUNCH_SCATTER_ADD(T) \
        if (idx_is_int32) \
            scatter_add_kernel_impl<T, int32_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>( \
                index.data<int32_t>(), src.data<T>(), output.data<T>(), \
                outer_size, dim_size, inner_size, index_dim_size, total_scatter, \
                error_buf.as<int>()); \
        else \
            scatter_add_kernel_impl<T, int64_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>( \
                index.data<int64_t>(), src.data<T>(), output.data<T>(), \
                outer_size, dim_size, inner_size, index_dim_size, total_scatter, \
                error_buf.as<int>()); \
        CUDA_CHECK(cudaGetLastError())

    switch (input.dtype()) {
        case DType::Float32: LAUNCH_SCATTER_ADD(float); break;
        case DType::Float64: LAUNCH_SCATTER_ADD(double); break;
        case DType::Int32:   LAUNCH_SCATTER_ADD(int32_t); break;
        case DType::Int64:   LAUNCH_SCATTER_ADD(int64_t); break;
        case DType::Float16:
        case DType::BFloat16: {
            // Float16/BFloat16: upcast to Float32 for atomicAdd, then cast back.
            // This is correct because atomicAdd is not natively supported for half types
            // on all architectures, and upcasting avoids precision issues in accumulation.
            Tensor input_f32 = input.to(DType::Float32);
            Tensor src_f32 = src.to(DType::Float32);
            Tensor output_f32(output_shape, DType::Float32, input.device());
            CUDA_CHECK(cudaMemcpyAsync(output_f32.data_ptr(), input_f32.data_ptr(),
                                       total_input * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
            if (total_scatter > 0) {
                int num_blocks_scatter_f32 = get_num_blocks(total_scatter);
                if (idx_is_int32)
                    scatter_add_kernel_impl<float, int32_t><<<num_blocks_scatter_f32, BLOCK_SIZE, 0, stream>>>(
                        index.data<int32_t>(), src_f32.data<float>(), output_f32.data<float>(),
                        outer_size, dim_size, inner_size, index_dim_size, total_scatter,
                        error_buf.as<int>());
                else
                    scatter_add_kernel_impl<float, int64_t><<<num_blocks_scatter_f32, BLOCK_SIZE, 0, stream>>>(
                        index.data<int64_t>(), src_f32.data<float>(), output_f32.data<float>(),
                        outer_size, dim_size, inner_size, index_dim_size, total_scatter,
                        error_buf.as<int>());
                CUDA_CHECK(cudaGetLastError());
            }
            output = output_f32.to(input.dtype());
            break;
        }
        case DType::Int8:
        case DType::Int16:
        case DType::UInt8:
        case DType::UInt16:
        case DType::UInt32:
        case DType::UInt64: {
            // 8/16-bit and unsigned ints have no native atomicAdd; upcast to
            // Int64 for accumulation, then narrow back to the original dtype.
            Tensor input_i64 = input.to(DType::Int64);
            Tensor src_i64 = src.to(DType::Int64);
            Tensor output_i64(output_shape, DType::Int64, input.device());
            CUDA_CHECK(cudaMemcpyAsync(output_i64.data_ptr(), input_i64.data_ptr(),
                                       total_input * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream));
            if (total_scatter > 0) {
                int nb = get_num_blocks(total_scatter);
                if (idx_is_int32)
                    scatter_add_kernel_impl<int64_t, int32_t><<<nb, BLOCK_SIZE, 0, stream>>>(
                        index.data<int32_t>(), src_i64.data<int64_t>(), output_i64.data<int64_t>(),
                        outer_size, dim_size, inner_size, index_dim_size, total_scatter, error_buf.as<int>());
                else
                    scatter_add_kernel_impl<int64_t, int64_t><<<nb, BLOCK_SIZE, 0, stream>>>(
                        index.data<int64_t>(), src_i64.data<int64_t>(), output_i64.data<int64_t>(),
                        outer_size, dim_size, inner_size, index_dim_size, total_scatter, error_buf.as<int>());
                CUDA_CHECK(cudaGetLastError());
            }
            output = output_i64.to(input.dtype());
            break;
        }
        default: throw std::runtime_error("scatter_add: unsupported dtype " +
                     std::string(dtype_name(input.dtype())));
    }

    #undef LAUNCH_SCATTER_ADD

    // Check for out-of-bounds index errors
    int host_error = 0;
    CUDA_CHECK(cudaMemcpyAsync(&host_error, error_buf.as<int>(), sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (host_error) {
        throw std::out_of_range(
            "scatter_add: index out of range for dimension of size " +
            std::to_string(dim_size));
    }

    return output;
}

// ============================================================================
// masked_select kernel — uses CUB DeviceSelect::Flagged for single-pass
// compaction instead of count + prefix_sum + gather (3 kernels → 1 CUB call)
// ============================================================================

auto masked_select_kernel(const Tensor& input, const Tensor& mask,
                          cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();

    if (n == 0) {
        return Tensor({0}, input.dtype(), input.device());
    }

    // Convert mask to bool if needed
    Tensor bool_mask = mask;
    if (mask.dtype() != DType::Bool) {
        bool_mask = mask;
    }

    const bool* d_flags = reinterpret_cast<const bool*>(bool_mask.data_ptr());
    size_t elem_size = dtype_size(input.dtype());

    // Allocate max-size temp output buffer and device counter for num_selected
    backend::CachedMemoryGuard d_out_guard(n * elem_size);
    backend::CachedMemoryGuard d_num_selected_guard(sizeof(int));
    auto* d_num_selected = static_cast<int*>(d_num_selected_guard.get());

    // Use CUB DeviceSelect::Flagged — single optimized pass replaces
    // count_true_kernel + compute_positions_kernel + masked_select_kernel_impl
    #define RUN_FLAGGED_SELECT(T) do { \
        const T* d_in = reinterpret_cast<const T*>(input.data_ptr()); \
        T* d_output = static_cast<T*>(d_out_guard.get()); \
        void* d_temp = nullptr; \
        size_t temp_bytes = 0; \
        cub::DeviceSelect::Flagged(d_temp, temp_bytes, \
            d_in, d_flags, d_output, d_num_selected, \
            static_cast<int>(n), stream); \
        backend::CachedMemoryGuard d_temp_guard(temp_bytes); \
        d_temp = d_temp_guard.get(); \
        cub::DeviceSelect::Flagged(d_temp, temp_bytes, \
            d_in, d_flags, d_output, d_num_selected, \
            static_cast<int>(n), stream); \
    } while(0)

    switch (input.dtype()) {
        case DType::Float32: RUN_FLAGGED_SELECT(float); break;
        case DType::Float64: RUN_FLAGGED_SELECT(double); break;
        case DType::Int32:   RUN_FLAGGED_SELECT(int32_t); break;
        case DType::Int64:   RUN_FLAGGED_SELECT(int64_t); break;
        case DType::Int8:    RUN_FLAGGED_SELECT(int8_t); break;
        case DType::UInt8:   RUN_FLAGGED_SELECT(uint8_t); break;
        case DType::Float16: RUN_FLAGGED_SELECT(__half); break;
        case DType::BFloat16: RUN_FLAGGED_SELECT(__nv_bfloat16); break;
        case DType::Int16:   RUN_FLAGGED_SELECT(int16_t); break;
        case DType::UInt16:  RUN_FLAGGED_SELECT(uint16_t); break;
        case DType::UInt32:  RUN_FLAGGED_SELECT(uint32_t); break;
        case DType::UInt64:  RUN_FLAGGED_SELECT(uint64_t); break;
        case DType::Complex64:  RUN_FLAGGED_SELECT(float2); break;
        case DType::Complex128: RUN_FLAGGED_SELECT(double2); break;
        default:
            throw std::runtime_error("masked_select: unsupported dtype");
    }

    #undef RUN_FLAGGED_SELECT

    // D2H sync — unavoidable for dynamic output size, but now happens after
    // a single optimized CUB operation instead of 3 separate kernel launches
    int h_count;
    CUDA_CHECK(cudaMemcpyAsync(&h_count, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (h_count == 0) {
        return Tensor({0}, input.dtype(), input.device());
    }

    // Create properly-sized output and D2D copy from temp buffer
    Tensor output({static_cast<int64_t>(h_count)}, input.dtype(), input.device());
    CUDA_CHECK(cudaMemcpyAsync(output.data_ptr(), d_out_guard.get(),
                                h_count * elem_size, cudaMemcpyDeviceToDevice, stream));

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// masked_fill kernel
// ============================================================================

template<typename T>
__global__ void masked_fill_kernel_impl(
    const T* input,
    const bool* mask,
    T fill_value,
    T* output,
    int64_t n) {

    TENZOR_CUDA_KERNEL_LOOP(i, n) {
        output[i] = mask[i] ? fill_value : input[i];
    }
}

auto masked_fill_kernel(const Tensor& input, const Tensor& mask, double value,
                        cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();

    std::vector<int64_t> output_shape(input.shape().begin(), input.shape().end());
    Tensor output(output_shape, input.dtype(), input.device());

    if (n == 0) return output;

    // F2: broadcast mask to input shape before kernel launch. Previously the
    // kernel read `mask.data<bool>()` as a flat sequence of length `n` —
    // when the mask had fewer dims or smaller dims (the common case is a
    // (B, 1, S, S) attention mask broadcast across heads to (B, H, S, S)),
    // the kernel either read past the end of the mask buffer (UB) or
    // silently miscompiled the per-element gate. broadcasting via the
    // existing `broadcast_to` op materialises a full mask once; the kernel
    // then sees a 1:1 mapping with input.
    Tensor mask_broadcast = mask;
    std::vector<int64_t> input_shape_vec(input.shape().begin(), input.shape().end());
    std::vector<int64_t> mask_shape_vec(mask.shape().begin(), mask.shape().end());
    if (mask_shape_vec != input_shape_vec) {
        // Verify the broadcast is legal (will throw with a clear msg otherwise).
        auto broadcast_shape = tenzor::broadcast_shapes(mask.shape(), input.shape());
        if (broadcast_shape != input_shape_vec) {
            throw std::invalid_argument(
                "masked_fill: mask shape is not broadcast-compatible with input shape");
        }
        mask_broadcast = tenzor::broadcast_to(mask, input_shape_vec).contiguous();
    }
    const bool* mask_ptr = reinterpret_cast<const bool*>(mask_broadcast.data_ptr());

    #define LAUNCH_MASKED_FILL(T, cast_val) do { \
        auto [grid_size, block_size] = optimal_launch_config( \
            masked_fill_kernel_impl<T>, n); \
        masked_fill_kernel_impl<T><<<grid_size, block_size, 0, stream>>>( \
            input.data<T>(), mask_ptr, static_cast<T>(cast_val), output.data<T>(), n); \
        CUDA_CHECK(cudaGetLastError()); \
    } while(0)

    switch (input.dtype()) {
        case DType::Float32: LAUNCH_MASKED_FILL(float, value); break;
        case DType::Float64: LAUNCH_MASKED_FILL(double, value); break;
        case DType::Int32:   LAUNCH_MASKED_FILL(int32_t, value); break;
        case DType::Int64:   LAUNCH_MASKED_FILL(int64_t, value); break;
        case DType::Int8:    LAUNCH_MASKED_FILL(int8_t, value); break;
        case DType::UInt8:   LAUNCH_MASKED_FILL(uint8_t, value); break;
        case DType::Float16: {
            __half fill_val = __float2half(static_cast<float>(value));
            auto [grid_size, block_size] = optimal_launch_config(
                masked_fill_kernel_impl<__half>, n);
            masked_fill_kernel_impl<__half><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(input.data_ptr()),
                mask_ptr, fill_val,
                reinterpret_cast<__half*>(output.data_ptr()), n);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case DType::BFloat16: {
            __nv_bfloat16 fill_val = __float2bfloat16(static_cast<float>(value));
            auto [grid_size, block_size] = optimal_launch_config(
                masked_fill_kernel_impl<__nv_bfloat16>, n);
            masked_fill_kernel_impl<__nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                mask_ptr, fill_val,
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), n);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case DType::Complex64: {
            float2 fill_val = make_float2(static_cast<float>(value), 0.0f);
            auto [grid_size, block_size] = optimal_launch_config(masked_fill_kernel_impl<float2>, n);
            masked_fill_kernel_impl<float2><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const float2*>(input.data_ptr()), mask_ptr, fill_val,
                reinterpret_cast<float2*>(output.data_ptr()), n);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case DType::Complex128: {
            double2 fill_val = make_double2(value, 0.0);
            auto [grid_size, block_size] = optimal_launch_config(masked_fill_kernel_impl<double2>, n);
            masked_fill_kernel_impl<double2><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const double2*>(input.data_ptr()), mask_ptr, fill_val,
                reinterpret_cast<double2*>(output.data_ptr()), n);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        default:
            throw std::runtime_error("masked_fill: unsupported dtype");
    }

    #undef LAUNCH_MASKED_FILL

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// where kernel
// ============================================================================

template<typename T>
__global__ void where_kernel_impl(
    const bool* condition,
    const T* x,
    const T* y,
    T* output,
    int64_t n) {

    TENZOR_CUDA_KERNEL_LOOP(i, n) {
        output[i] = condition[i] ? x[i] : y[i];
    }
}

auto where_kernel(const Tensor& condition, const Tensor& x, const Tensor& y,
                  cudaStream_t stream) -> Tensor {
    int64_t n = condition.numel();

    // Output shape matches condition (assume all same shape)
    std::vector<int64_t> output_shape(condition.shape().begin(), condition.shape().end());
    Tensor output(output_shape, x.dtype(), x.device());

    if (n == 0) return output;

    const bool* cond_ptr = reinterpret_cast<const bool*>(condition.data_ptr());

    #define LAUNCH_WHERE(T) do { \
        auto [grid_size, block_size] = optimal_launch_config( \
            where_kernel_impl<T>, n); \
        where_kernel_impl<T><<<grid_size, block_size, 0, stream>>>( \
            cond_ptr, x.data<T>(), y.data<T>(), output.data<T>(), n); \
        CUDA_CHECK(cudaGetLastError()); \
    } while(0)

    switch (x.dtype()) {
        case DType::Float32: LAUNCH_WHERE(float); break;
        case DType::Float64: LAUNCH_WHERE(double); break;
        case DType::Int32:   LAUNCH_WHERE(int32_t); break;
        case DType::Int64:   LAUNCH_WHERE(int64_t); break;
        case DType::Int8:    LAUNCH_WHERE(int8_t); break;
        case DType::UInt8:   LAUNCH_WHERE(uint8_t); break;
        case DType::Float16: {
            auto [grid_size, block_size] = optimal_launch_config(
                where_kernel_impl<__half>, n);
            where_kernel_impl<__half><<<grid_size, block_size, 0, stream>>>(
                cond_ptr,
                reinterpret_cast<const __half*>(x.data_ptr()),
                reinterpret_cast<const __half*>(y.data_ptr()),
                reinterpret_cast<__half*>(output.data_ptr()), n);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case DType::BFloat16: {
            auto [grid_size, block_size] = optimal_launch_config(
                where_kernel_impl<__nv_bfloat16>, n);
            where_kernel_impl<__nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
                cond_ptr,
                reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), n);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case DType::Int16:   LAUNCH_WHERE(int16_t); break;
        case DType::UInt16:  LAUNCH_WHERE(uint16_t); break;
        case DType::UInt32:  LAUNCH_WHERE(uint32_t); break;
        case DType::UInt64:  LAUNCH_WHERE(uint64_t); break;
        case DType::Complex64: {
            auto [grid_size, block_size] = optimal_launch_config(where_kernel_impl<float2>, n);
            where_kernel_impl<float2><<<grid_size, block_size, 0, stream>>>(
                cond_ptr,
                reinterpret_cast<const float2*>(x.data_ptr()),
                reinterpret_cast<const float2*>(y.data_ptr()),
                reinterpret_cast<float2*>(output.data_ptr()), n);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case DType::Complex128: {
            auto [grid_size, block_size] = optimal_launch_config(where_kernel_impl<double2>, n);
            where_kernel_impl<double2><<<grid_size, block_size, 0, stream>>>(
                cond_ptr,
                reinterpret_cast<const double2*>(x.data_ptr()),
                reinterpret_cast<const double2*>(y.data_ptr()),
                reinterpret_cast<double2*>(output.data_ptr()), n);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        default:
            throw std::runtime_error("where: unsupported dtype");
    }

    #undef LAUNCH_WHERE

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// Deferred index-out-of-range error flag (per device).
//
// Bounds-checked index kernels (embedding) write a device flag on OOB and the
// kernel itself stays memory-safe (writes 0 for the bad row, never reads OOB).
// Instead of a blocking cudaStreamSynchronize per call to surface the throw
// — which dominated runtime for these tiny kernels and which PyTorch doesn't
// pay (it does no bounds check at all) — we copy the flag to pinned host memory
// asynchronously and let the error surface at the next device synchronization
// (`cuda_drain_index_errors`, called from the backend's synchronize()). This
// matches CUDA/PyTorch async-error semantics: the kernel is safe; the catchable
// std::out_of_range is raised at the next sync rather than inline.
// ============================================================================
namespace {
struct IndexErrorFlag {
    int* d_flag = nullptr;   // device-side, set by the kernel on OOB
    int* h_flag = nullptr;   // pinned host mirror, updated async after each call
};
std::mutex g_index_err_mutex;
std::unordered_map<int, IndexErrorFlag> g_index_err_flags;

IndexErrorFlag& get_index_error_flag(int device) {
    // caller holds g_index_err_mutex
    auto& f = g_index_err_flags[device];
    if (!f.d_flag) {
        int prev = 0; cudaGetDevice(&prev);
        cudaSetDevice(device);
        cudaMalloc(&f.d_flag, sizeof(int));
        cudaMemset(f.d_flag, 0, sizeof(int));
        cudaMallocHost(&f.h_flag, sizeof(int));
        *f.h_flag = 0;
        cudaSetDevice(prev);
    }
    return f;
}
}  // namespace

// Enqueue (no sync) the async copy of the device OOB flag to its pinned host
// mirror on `stream`, after a bounds-checked kernel launch on the same stream.
void record_index_error_async(int device, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(g_index_err_mutex);
    auto& f = get_index_error_flag(device);
    cudaMemcpyAsync(f.h_flag, f.d_flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
}

// Surface any pending out-of-range error as a catchable exception. Safe to call
// only at a point where the relevant stream work has completed (the backend
// calls it right after cudaDeviceSynchronize), so the pinned host mirror is
// valid. Resets the flags before throwing so the next op starts clean.
void cuda_drain_index_errors() {
    int bad_device = -1;
    {
        std::lock_guard<std::mutex> lk(g_index_err_mutex);
        for (auto& [dev, f] : g_index_err_flags) {
            if (f.h_flag && *f.h_flag != 0) {
                *f.h_flag = 0;
                int prev = 0; cudaGetDevice(&prev);
                cudaSetDevice(dev);
                cudaMemset(f.d_flag, 0, sizeof(int));
                cudaSetDevice(prev);
                bad_device = dev;
                break;
            }
        }
    }
    if (bad_device >= 0) {
        throw std::out_of_range("Embedding index out of range (detected at synchronization)");
    }
}

// ============================================================================
// embedding kernel (lookup table for token IDs)
// ============================================================================

template<typename T, typename IndexT>
__global__ void embedding_kernel_impl(
    const T* weight,           // [num_embeddings, embedding_dim]
    const IndexT* indices,     // [*] (any shape of indices)
    T* output,                 // [*, embedding_dim]
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t num_embeddings,
    int* error_flag) {

    int64_t total_elements = num_indices * embedding_dim;

    TENZOR_CUDA_KERNEL_LOOP(idx, total_elements) {
        int64_t i = idx / embedding_dim;  // which index
        int64_t j = idx % embedding_dim;  // which embedding dimension

        int64_t token_idx = static_cast<int64_t>(indices[i]);
        // Device-side bounds check: flag out-of-range ids and skip the OOB read
        // (which would corrupt memory or fault). The host reads the flag once
        // after launch and throws — replacing the previous host-side D2H of all
        // indices + serial validation loop that serialized the GPU path.
        if (token_idx < 0 || token_idx >= num_embeddings) {
            atomicExch(error_flag, 1);
            output[idx] = T{};
            continue;
        }
        output[idx] = weight[token_idx * embedding_dim + j];
    }
}

auto embedding_kernel(const Tensor& weight, const Tensor& indices,
                      cudaStream_t stream) -> Tensor {
    // weight: [num_embeddings, embedding_dim]
    // indices: [*] (any shape of int64 indices)
    // output: [*, embedding_dim]

    auto w_shape = weight.shape();
    auto idx_shape = indices.shape();

    int64_t embedding_dim = w_shape[1];
    int64_t num_embeddings = w_shape[0];
    int64_t num_indices = indices.numel();

    // Build output shape: indices shape + embedding_dim
    std::vector<int64_t> output_shape(idx_shape.begin(), idx_shape.end());
    output_shape.push_back(embedding_dim);

    Tensor output(output_shape, weight.dtype(), weight.device());

    if (num_indices == 0) return output;

    int64_t total_elements = num_indices * embedding_dim;

    bool idx_is_int32 = (indices.dtype() == DType::Int32);
    bool idx_is_int64 = (indices.dtype() == DType::Int64);
    if (!idx_is_int32 && !idx_is_int64) {
        throw std::invalid_argument("embedding: indices must be Int32 or Int64");
    }

    // Persistent per-device device-side OOB flag. The kernel sets it on an
    // out-of-range id (and writes a zero row — no illegal access). We do NOT
    // synchronize here: the flag is async-copied to pinned host below and the
    // catchable std::out_of_range is raised at the next device sync via
    // cuda_drain_index_errors(). This removes the per-call cudaMalloc/cudaFree
    // (the old CudaBuffer) AND the blocking cudaStreamSynchronize, matching
    // PyTorch's no-per-call-sync forward.
    const int err_dev = weight.device().index;
    int* err_flag = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_index_err_mutex);
        err_flag = get_index_error_flag(err_dev).d_flag;
    }

    #define LAUNCH_EMBEDDING(T) \
        if (idx_is_int32) { \
            auto [grid_size, block_size] = optimal_launch_config( \
                embedding_kernel_impl<T, int32_t>, total_elements); \
            embedding_kernel_impl<T, int32_t><<<grid_size, block_size, 0, stream>>>( \
                weight.data<T>(), indices.data<int32_t>(), output.data<T>(), \
                num_indices, embedding_dim, num_embeddings, err_flag); \
            CUDA_CHECK(cudaGetLastError()); \
        } else { \
            auto [grid_size, block_size] = optimal_launch_config( \
                embedding_kernel_impl<T, int64_t>, total_elements); \
            embedding_kernel_impl<T, int64_t><<<grid_size, block_size, 0, stream>>>( \
                weight.data<T>(), indices.data<int64_t>(), output.data<T>(), \
                num_indices, embedding_dim, num_embeddings, err_flag); \
            CUDA_CHECK(cudaGetLastError()); \
        }

    switch (weight.dtype()) {
        case DType::Float32: LAUNCH_EMBEDDING(float); break;
        case DType::Float64: LAUNCH_EMBEDDING(double); break;
        case DType::Float16:
            if (idx_is_int32) {
                auto [grid_size, block_size] = optimal_launch_config(
                    embedding_kernel_impl<__half, int32_t>, total_elements);
                embedding_kernel_impl<__half, int32_t><<<grid_size, block_size, 0, stream>>>(
                    reinterpret_cast<const __half*>(weight.data_ptr()),
                    indices.data<int32_t>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    num_indices, embedding_dim, num_embeddings, err_flag);
                CUDA_CHECK(cudaGetLastError());
            } else {
                auto [grid_size, block_size] = optimal_launch_config(
                    embedding_kernel_impl<__half, int64_t>, total_elements);
                embedding_kernel_impl<__half, int64_t><<<grid_size, block_size, 0, stream>>>(
                    reinterpret_cast<const __half*>(weight.data_ptr()),
                    indices.data<int64_t>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    num_indices, embedding_dim, num_embeddings, err_flag);
                CUDA_CHECK(cudaGetLastError());
            }
            break;
        case DType::BFloat16:
            if (idx_is_int32) {
                auto [grid_size, block_size] = optimal_launch_config(
                    embedding_kernel_impl<__nv_bfloat16, int32_t>, total_elements);
                embedding_kernel_impl<__nv_bfloat16, int32_t><<<grid_size, block_size, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
                    indices.data<int32_t>(),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    num_indices, embedding_dim, num_embeddings, err_flag);
                CUDA_CHECK(cudaGetLastError());
            } else {
                auto [grid_size, block_size] = optimal_launch_config(
                    embedding_kernel_impl<__nv_bfloat16, int64_t>, total_elements);
                embedding_kernel_impl<__nv_bfloat16, int64_t><<<grid_size, block_size, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
                    indices.data<int64_t>(),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    num_indices, embedding_dim, num_embeddings, err_flag);
                CUDA_CHECK(cudaGetLastError());
            }
            break;
        default:
            throw std::runtime_error("embedding: unsupported dtype");
    }

    #undef LAUNCH_EMBEDDING

    CUDA_CHECK(cudaGetLastError());

    // Async-copy the OOB flag to its pinned host mirror on the same stream and
    // return WITHOUT synchronizing. A pending out-of-range id is raised as a
    // catchable std::out_of_range at the next device sync via
    // cuda_drain_index_errors() (CUDA async-error semantics).
    record_index_error_async(err_dev, stream);

    return output;
}

// ============================================================================
// embedding_backward kernel (gradient accumulation)
// ============================================================================

// Row-oriented backward: one block per index row, threads stride over
// embedding_dim. This eliminates the per-element int64 div/mod of the old
// element-parallel kernel (which dominated runtime), keeps grad_output reads
// and grad_weight atomics fully coalesced, and computes each row's base
// offsets once. atomicAdd handles overlapping (repeated) indices.
template<typename T, typename IndexT>
__global__ void embedding_backward_kernel_impl(
    const T* grad_output,      // [*, embedding_dim]
    const IndexT* indices,     // [*]
    T* grad_weight,            // [num_embeddings, embedding_dim]
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t num_embeddings) {

    for (int64_t i = blockIdx.x; i < num_indices; i += gridDim.x) {
        int64_t token_idx = static_cast<int64_t>(indices[i]);
        if (token_idx < 0 || token_idx >= num_embeddings) continue;
        const T* go = grad_output + i * embedding_dim;
        T* gw = grad_weight + token_idx * embedding_dim;
        for (int64_t j = threadIdx.x; j < embedding_dim; j += blockDim.x) {
            atomicAdd(&gw[j], go[j]);
        }
    }
}

// Separate FP16 backward kernel (cannot partially specialize function templates)
template<typename IndexT>
__global__ void embedding_backward_fp16_kernel_impl(
    const __half* grad_output,
    const IndexT* indices,
    __half* grad_weight,
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t num_embeddings) {

    for (int64_t i = blockIdx.x; i < num_indices; i += gridDim.x) {
        int64_t token_idx = static_cast<int64_t>(indices[i]);
        if (token_idx < 0 || token_idx >= num_embeddings) continue;
        const __half* go = grad_output + i * embedding_dim;
        for (int64_t j = threadIdx.x; j < embedding_dim; j += blockDim.x) {
            int64_t flat = token_idx * embedding_dim + j;
#if __CUDA_ARCH__ >= 700
            atomicAdd(&grad_weight[flat], go[j]);
#else
            // Fallback for older architectures: compare-and-swap based atomic add
            float val = __half2float(go[j]);
            unsigned int* addr = reinterpret_cast<unsigned int*>(&grad_weight[flat & ~1]);
            unsigned int old_val, new_val;
            do {
                old_val = atomicCAS(addr, 0u, 0u);  // Atomic initial read
                __half* h = reinterpret_cast<__half*>(&old_val);
                // W.7: NaN-preserving conversion.
                __half result = ::tenzor::cuda::safe_f2half(::tenzor::cuda::safe_half2f(h[flat & 1]) + val);
                new_val = old_val;
                reinterpret_cast<__half*>(&new_val)[flat & 1] = result;
            } while (atomicCAS(addr, old_val, new_val) != old_val);
#endif
        }
    }
}

// Separate BF16 backward kernel — atomicAdd(__nv_bfloat16) requires SM >= 80
template<typename IndexT>
__global__ void embedding_backward_bf16_kernel_impl(
    const __nv_bfloat16* grad_output,
    const IndexT* indices,
    __nv_bfloat16* grad_weight,
    int64_t num_indices,
    int64_t embedding_dim,
    int64_t num_embeddings) {

    for (int64_t i = blockIdx.x; i < num_indices; i += gridDim.x) {
        int64_t token_idx = static_cast<int64_t>(indices[i]);
        if (token_idx < 0 || token_idx >= num_embeddings) continue;
        const __nv_bfloat16* go = grad_output + i * embedding_dim;
        for (int64_t j = threadIdx.x; j < embedding_dim; j += blockDim.x) {
            int64_t flat = token_idx * embedding_dim + j;
#if __CUDA_ARCH__ >= 800
            atomicAdd(&grad_weight[flat], go[j]);
#else
            // Fallback for SM < 80: CAS-based atomic add via float conversion
            float val = __bfloat162float(go[j]);
            unsigned int* addr = reinterpret_cast<unsigned int*>(&grad_weight[flat & ~1]);
            unsigned int old_val, new_val;
            do {
                old_val = atomicCAS(addr, 0u, 0u);  // Atomic initial read (avoids data race)
                __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(&old_val);
                // W.7: NaN-preserving conversion.
                __nv_bfloat16 result = ::tenzor::cuda::safe_f2bf16(::tenzor::cuda::safe_bf162f(h[flat & 1]) + val);
                new_val = old_val;
                reinterpret_cast<__nv_bfloat16*>(&new_val)[flat & 1] = result;
            } while (atomicCAS(addr, old_val, new_val) != old_val);
#endif
        }
    }
}

auto embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                               int64_t num_embeddings, cudaStream_t stream) -> Tensor {
    // grad_output: [*, embedding_dim]
    // indices: [*]
    // output (grad_weight): [num_embeddings, embedding_dim]

    auto grad_shape = grad_output.shape();
    int64_t embedding_dim = grad_shape[grad_shape.size() - 1];
    int64_t num_indices = indices.numel();

    // Create zero-initialized gradient weight tensor
    std::vector<int64_t> grad_weight_shape = {num_embeddings, embedding_dim};
    Tensor grad_weight(grad_weight_shape, grad_output.dtype(), grad_output.device());

    // Zero initialize
    cudaMemsetAsync(grad_weight.data_ptr(), 0, grad_weight.numel() * dtype_size(grad_output.dtype()), stream);

    if (num_indices == 0) return grad_weight;

    // Row-per-block launch: one block per index row, block sized to the
    // embedding dim (warp-rounded, capped at 256) so small dims don't waste a
    // full 256-thread block, and the grid spans the index rows (grid-stride
    // covers any overflow).
    int block_size = static_cast<int>(
        std::min<int64_t>(256, std::max<int64_t>(32, ((embedding_dim + 31) / 32) * 32)));
    int num_blocks = static_cast<int>(std::min<int64_t>(num_indices, 2147483647LL));
    if (num_blocks < 1) num_blocks = 1;

    bool idx_is_int32 = (indices.dtype() == DType::Int32);
    bool idx_is_int64 = (indices.dtype() == DType::Int64);
    if (!idx_is_int32 && !idx_is_int64) {
        throw std::invalid_argument("embedding_backward: indices must be Int32 or Int64");
    }

    #define LAUNCH_EMBEDDING_BWD(T) \
        if (idx_is_int32) \
            embedding_backward_kernel_impl<T, int32_t><<<num_blocks, block_size, 0, stream>>>( \
                grad_output.data<T>(), indices.data<int32_t>(), grad_weight.data<T>(), \
                num_indices, embedding_dim, num_embeddings); \
        else \
            embedding_backward_kernel_impl<T, int64_t><<<num_blocks, block_size, 0, stream>>>( \
                grad_output.data<T>(), indices.data<int64_t>(), grad_weight.data<T>(), \
                num_indices, embedding_dim, num_embeddings); \
        CUDA_CHECK(cudaGetLastError())

    switch (grad_output.dtype()) {
        case DType::Float32: LAUNCH_EMBEDDING_BWD(float); break;
        case DType::Float64: LAUNCH_EMBEDDING_BWD(double); break;
        case DType::Float16:
            if (idx_is_int32)
                embedding_backward_fp16_kernel_impl<int32_t><<<num_blocks, block_size, 0, stream>>>(
                    reinterpret_cast<const __half*>(grad_output.data_ptr()),
                    indices.data<int32_t>(),
                    reinterpret_cast<__half*>(grad_weight.data_ptr()),
                    num_indices, embedding_dim, num_embeddings);
            else
                embedding_backward_fp16_kernel_impl<int64_t><<<num_blocks, block_size, 0, stream>>>(
                    reinterpret_cast<const __half*>(grad_output.data_ptr()),
                    indices.data<int64_t>(),
                    reinterpret_cast<__half*>(grad_weight.data_ptr()),
                    num_indices, embedding_dim, num_embeddings);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::BFloat16:
            if (idx_is_int32)
                embedding_backward_bf16_kernel_impl<int32_t><<<num_blocks, block_size, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
                    indices.data<int32_t>(),
                    reinterpret_cast<__nv_bfloat16*>(grad_weight.data_ptr()),
                    num_indices, embedding_dim, num_embeddings);
            else
                embedding_backward_bf16_kernel_impl<int64_t><<<num_blocks, block_size, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
                    indices.data<int64_t>(),
                    reinterpret_cast<__nv_bfloat16*>(grad_weight.data_ptr()),
                    num_indices, embedding_dim, num_embeddings);
            CUDA_CHECK(cudaGetLastError());
            break;
        default:
            throw std::runtime_error("embedding_backward: unsupported dtype");
    }

    #undef LAUNCH_EMBEDDING_BWD

    CUDA_CHECK(cudaGetLastError());
    return grad_weight;
}

// ============================================================================
// one_hot kernel
// ============================================================================

template<typename IndexT>
__global__ void one_hot_kernel_impl(
    const IndexT* indices,
    float* output,
    int64_t batch_size,
    int64_t num_classes) {

    int64_t total = batch_size * num_classes;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t batch = idx / num_classes;
        int64_t cls = idx % num_classes;
        output[idx] = (static_cast<int64_t>(indices[batch]) == cls) ? 1.0f : 0.0f;
    }
}

auto one_hot_kernel(const Tensor& indices, int64_t num_classes,
                    cudaStream_t stream) -> Tensor {
    int64_t batch_size = indices.numel();

    Tensor output({batch_size, num_classes}, DType::Float32, indices.device());

    if (batch_size == 0) return output;

    int64_t total = batch_size * num_classes;
    int num_blocks = get_num_blocks(total);

    switch (indices.dtype()) {
        case DType::Int32:
            one_hot_kernel_impl<int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                indices.data<int32_t>(), output.data<float>(), batch_size, num_classes);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::Int64:
            one_hot_kernel_impl<int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                indices.data<int64_t>(), output.data<float>(), batch_size, num_classes);
            CUDA_CHECK(cudaGetLastError());
            break;
        default:
            throw std::runtime_error("one_hot: unsupported index dtype (expected Int32 or Int64)");
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// nonzero kernel
// ============================================================================

template<typename T>
__global__ void nonzero_flag_kernel(
    const T* input,
    int64_t* flags,
    int64_t n) {

    TENZOR_CUDA_KERNEL_LOOP(i, n) {
        flags[i] = (input[i] != static_cast<T>(0)) ? 1 : 0;
    }
}

// Specialization for __half
template<>
__global__ void nonzero_flag_kernel<__half>(
    const __half* input,
    int64_t* flags,
    int64_t n) {

    TENZOR_CUDA_KERNEL_LOOP(i, n) {
        flags[i] = (__hne(input[i], __float2half(0.0f))) ? 1 : 0;
    }
}

// Specialization for __nv_bfloat16
template<>
__global__ void nonzero_flag_kernel<__nv_bfloat16>(
    const __nv_bfloat16* input,
    int64_t* flags,
    int64_t n) {

    TENZOR_CUDA_KERNEL_LOOP(i, n) {
#if __CUDA_ARCH__ >= 800
        flags[i] = (__hne(input[i], __float2bfloat16(0.0f))) ? 1 : 0;
#else
        flags[i] = (__bfloat162float(input[i]) != 0.0f) ? 1 : 0;
#endif
    }
}

// Decompose compacted flat indices into multi-dimensional indices
__global__ void decompose_flat_indices_kernel(
    const int64_t* flat_indices,
    int64_t* output,
    const int64_t* shape,
    int64_t num_indices,
    int64_t ndim) {

    TENZOR_CUDA_KERNEL_LOOP(i, num_indices) {
        int64_t flat = flat_indices[i];
        for (int64_t d = ndim - 1; d >= 0; --d) {
            output[i * ndim + d] = flat % shape[d];
            flat /= shape[d];
        }
    }
}

auto nonzero_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();
    int64_t ndim = input.ndim();

    if (n == 0) {
        return Tensor({0, ndim}, DType::Int64, input.device());
    }

    // Allocate flags array
    backend::CachedMemoryGuard d_flags_guard(n * sizeof(int64_t));
    auto* d_flags = static_cast<int64_t*>(d_flags_guard.get());

    int num_blocks = get_num_blocks(n);

    // Launch flag kernel based on dtype
    #define LAUNCH_NONZERO_FLAG(T) \
        nonzero_flag_kernel<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>( \
            input.data<T>(), d_flags, n); \
        CUDA_CHECK(cudaGetLastError())

    switch (input.dtype()) {
        case DType::Float32: LAUNCH_NONZERO_FLAG(float); break;
        case DType::Float64: LAUNCH_NONZERO_FLAG(double); break;
        case DType::Int32:   LAUNCH_NONZERO_FLAG(int32_t); break;
        case DType::Int64:   LAUNCH_NONZERO_FLAG(int64_t); break;
        case DType::Int8:    LAUNCH_NONZERO_FLAG(int8_t); break;
        case DType::UInt8:   LAUNCH_NONZERO_FLAG(uint8_t); break;
        case DType::Bool:    LAUNCH_NONZERO_FLAG(bool); break;
        case DType::Float16:
            nonzero_flag_kernel<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const __half*>(input.data_ptr()), d_flags, n);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::BFloat16:
            nonzero_flag_kernel<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()), d_flags, n);
            CUDA_CHECK(cudaGetLastError());
            break;
        default:
            throw std::runtime_error("nonzero: unsupported dtype");
    }

    #undef LAUNCH_NONZERO_FLAG

    // Use CUB DeviceSelect::Flagged with CountingInputIterator to compact
    // nonzero flat indices in a single pass — replaces InclusiveSum + gather
    thrust::counting_iterator<int64_t> iota(0);

    backend::CachedMemoryGuard d_flat_indices_guard(n * sizeof(int64_t));
    auto* d_flat_indices = static_cast<int64_t*>(d_flat_indices_guard.get());

    backend::CachedMemoryGuard d_num_selected_guard(sizeof(int));
    auto* d_num_selected = static_cast<int*>(d_num_selected_guard.get());

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceSelect::Flagged(d_temp, temp_bytes,
        iota, d_flags, d_flat_indices, d_num_selected,
        static_cast<int>(n), stream);
    backend::CachedMemoryGuard d_temp_guard(temp_bytes);
    d_temp = d_temp_guard.get();
    cub::DeviceSelect::Flagged(d_temp, temp_bytes,
        iota, d_flags, d_flat_indices, d_num_selected,
        static_cast<int>(n), stream);

    // Single D2H sync to get count (replaces two syncs in the old code)
    int total_nonzero;
    CUDA_CHECK(cudaMemcpyAsync(&total_nonzero, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (total_nonzero == 0) {
        return Tensor({0, ndim}, DType::Int64, input.device());
    }

    // Allocate output tensor and decompose flat indices to multi-dim
    Tensor output({static_cast<int64_t>(total_nonzero), ndim}, DType::Int64, input.device());

    backend::CachedMemoryGuard d_shape_guard(ndim * sizeof(int64_t));
    auto* d_shape = static_cast<int64_t*>(d_shape_guard.get());
    CUDA_CHECK(cudaMemcpyAsync(d_shape, input.shape().data(),
                                ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    int decompose_blocks = get_num_blocks(total_nonzero);
    decompose_flat_indices_kernel<<<decompose_blocks, BLOCK_SIZE, 0, stream>>>(
        d_flat_indices, output.data<int64_t>(), d_shape, total_nonzero, ndim);

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// Take kernel — gather from flattened input
// ============================================================================

template<typename T>
__global__ void take_kernel_impl(
    const T* __restrict__ input,
    const int64_t* __restrict__ indices,
    T* __restrict__ output,
    int64_t input_size,
    int64_t indices_size
) {
    TENZOR_CUDA_KERNEL_LOOP(idx, indices_size) {
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

auto take_kernel(const Tensor& input, const Tensor& indices, cudaStream_t stream) -> Tensor {
    int64_t input_size = input.numel();
    int64_t indices_size = indices.numel();

    Tensor output({indices_size}, input.dtype(), input.device());

    if (indices_size == 0) return output;

    #define LAUNCH_TAKE(T) do { \
        auto [grid_size, block_size] = optimal_launch_config( \
            take_kernel_impl<T>, indices_size); \
        take_kernel_impl<T><<<grid_size, block_size, 0, stream>>>( \
            input.data<T>(), indices.data<int64_t>(), output.data<T>(), \
            input_size, indices_size); \
        CUDA_CHECK(cudaGetLastError()); \
    } while(0)

    switch (input.dtype()) {
        case DType::Float32: LAUNCH_TAKE(float); break;
        case DType::Float64: LAUNCH_TAKE(double); break;
        case DType::Int32:   LAUNCH_TAKE(int32_t); break;
        case DType::Int64:   LAUNCH_TAKE(int64_t); break;
        case DType::Int8:    LAUNCH_TAKE(int8_t); break;
        case DType::UInt8:   LAUNCH_TAKE(uint8_t); break;
        case DType::Float16: {
            auto [grid_size, block_size] = optimal_launch_config(
                take_kernel_impl<__half>, indices_size);
            take_kernel_impl<__half><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const __half*>(input.data_ptr()),
                indices.data<int64_t>(),
                reinterpret_cast<__half*>(output.data_ptr()),
                input_size, indices_size);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        case DType::BFloat16: {
            auto [grid_size, block_size] = optimal_launch_config(
                take_kernel_impl<__nv_bfloat16>, indices_size);
            take_kernel_impl<__nv_bfloat16><<<grid_size, block_size, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                indices.data<int64_t>(),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                input_size, indices_size);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        default:
            throw std::runtime_error("take_kernel: unsupported dtype");
    }

    #undef LAUNCH_TAKE

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// Put kernel — scatter into flattened output
// ============================================================================

template<typename T>
__global__ void put_kernel_impl(
    T* __restrict__ output,
    const int64_t* __restrict__ indices,
    const T* __restrict__ source,
    int64_t num_indices,
    int64_t total_size,
    bool accumulate
) {
    TENZOR_CUDA_KERNEL_LOOP(idx, num_indices) {
        int64_t target_idx = indices[idx];
        // Handle negative indices
        if (target_idx < 0) {
            target_idx += total_size;
        }
        // Bounds checking
        if (target_idx >= 0 && target_idx < total_size) {
            if (accumulate) {
                atomicAdd(&output[target_idx], source[idx]);
            } else {
                output[target_idx] = source[idx];
            }
        }
    }
}

// Specialization for int8_t — atomicAdd not available, use CAS on containing int
template<>
__global__ void put_kernel_impl<int8_t>(
    int8_t* __restrict__ output,
    const int64_t* __restrict__ indices,
    const int8_t* __restrict__ source,
    int64_t num_indices,
    int64_t total_size,
    bool accumulate
) {
    TENZOR_CUDA_KERNEL_LOOP(idx, num_indices) {
        int64_t target_idx = indices[idx];
        if (target_idx < 0) {
            target_idx += total_size;
        }
        if (target_idx >= 0 && target_idx < total_size) {
            if (accumulate) {
                // CAS loop on the aligned 32-bit word containing this byte
                unsigned int byte_offset = static_cast<unsigned int>(target_idx) & 3u;
                unsigned int* addr = reinterpret_cast<unsigned int*>(
                    reinterpret_cast<char*>(output) + (target_idx - byte_offset));
                unsigned int old_val, new_val;
                do {
                    old_val = atomicCAS(addr, 0u, 0u);  // Atomic initial read
                    int8_t cur = static_cast<int8_t>((old_val >> (byte_offset * 8)) & 0xFF);
                    int8_t sum = static_cast<int8_t>(cur + source[idx]);
                    new_val = (old_val & ~(0xFFu << (byte_offset * 8))) |
                              (static_cast<unsigned int>(static_cast<uint8_t>(sum)) << (byte_offset * 8));
                } while (atomicCAS(addr, old_val, new_val) != old_val);
            } else {
                output[target_idx] = source[idx];
            }
        }
    }
}

// Specialization for uint8_t — same CAS approach
template<>
__global__ void put_kernel_impl<uint8_t>(
    uint8_t* __restrict__ output,
    const int64_t* __restrict__ indices,
    const uint8_t* __restrict__ source,
    int64_t num_indices,
    int64_t total_size,
    bool accumulate
) {
    TENZOR_CUDA_KERNEL_LOOP(idx, num_indices) {
        int64_t target_idx = indices[idx];
        if (target_idx < 0) {
            target_idx += total_size;
        }
        if (target_idx >= 0 && target_idx < total_size) {
            if (accumulate) {
                unsigned int byte_offset = static_cast<unsigned int>(target_idx) & 3u;
                unsigned int* addr = reinterpret_cast<unsigned int*>(
                    reinterpret_cast<char*>(output) + (target_idx - byte_offset));
                unsigned int old_val, new_val;
                do {
                    old_val = atomicCAS(addr, 0u, 0u);  // Atomic initial read
                    uint8_t cur = static_cast<uint8_t>((old_val >> (byte_offset * 8)) & 0xFF);
                    uint8_t sum = static_cast<uint8_t>(cur + source[idx]);
                    new_val = (old_val & ~(0xFFu << (byte_offset * 8))) |
                              (static_cast<unsigned int>(sum) << (byte_offset * 8));
                } while (atomicCAS(addr, old_val, new_val) != old_val);
            } else {
                output[target_idx] = source[idx];
            }
        }
    }
}

// Specialization for int32_t atomicAdd (not natively supported on all archs)
template<>
__global__ void put_kernel_impl<int32_t>(
    int32_t* __restrict__ output,
    const int64_t* __restrict__ indices,
    const int32_t* __restrict__ source,
    int64_t num_indices,
    int64_t total_size,
    bool accumulate
) {
    TENZOR_CUDA_KERNEL_LOOP(idx, num_indices) {
        int64_t target_idx = indices[idx];
        if (target_idx < 0) {
            target_idx += total_size;
        }
        if (target_idx >= 0 && target_idx < total_size) {
            if (accumulate) {
                atomicAdd(reinterpret_cast<int*>(&output[target_idx]), static_cast<int>(source[idx]));
            } else {
                output[target_idx] = source[idx];
            }
        }
    }
}

// Specialization for int64_t atomicAdd via CAS
__device__ inline int64_t atomicAdd_int64(int64_t* address, int64_t val) {
    unsigned long long* addr_ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *addr_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_ull, assumed,
                        static_cast<unsigned long long>(static_cast<int64_t>(assumed) + val));
    } while (assumed != old);
    return static_cast<int64_t>(old);
}

template<>
__global__ void put_kernel_impl<int64_t>(
    int64_t* __restrict__ output,
    const int64_t* __restrict__ indices,
    const int64_t* __restrict__ source,
    int64_t num_indices,
    int64_t total_size,
    bool accumulate
) {
    TENZOR_CUDA_KERNEL_LOOP(idx, num_indices) {
        int64_t target_idx = indices[idx];
        if (target_idx < 0) {
            target_idx += total_size;
        }
        if (target_idx >= 0 && target_idx < total_size) {
            if (accumulate) {
                atomicAdd_int64(&output[target_idx], source[idx]);
            } else {
                output[target_idx] = source[idx];
            }
        }
    }
}

// Specialization for __half atomicAdd in put
template<>
__global__ void put_kernel_impl<__half>(
    __half* __restrict__ output,
    const int64_t* __restrict__ indices,
    const __half* __restrict__ source,
    int64_t num_indices,
    int64_t total_size,
    bool accumulate
) {
    TENZOR_CUDA_KERNEL_LOOP(idx, num_indices) {
        int64_t target_idx = indices[idx];
        if (target_idx < 0) {
            target_idx += total_size;
        }
        if (target_idx >= 0 && target_idx < total_size) {
            if (accumulate) {
#if __CUDA_ARCH__ >= 700
                atomicAdd(&output[target_idx], source[idx]);
#else
                float val = __half2float(source[idx]);
                unsigned int* addr = reinterpret_cast<unsigned int*>(&output[target_idx & ~1]);
                unsigned int old_val, new_val;
                do {
                    old_val = atomicCAS(addr, 0u, 0u);  // Atomic initial read
                    __half* h = reinterpret_cast<__half*>(&old_val);
                    // W.7: NaN-preserving conversion.
                    __half result = ::tenzor::cuda::safe_f2half(::tenzor::cuda::safe_half2f(h[target_idx & 1]) + val);
                    new_val = old_val;
                    reinterpret_cast<__half*>(&new_val)[target_idx & 1] = result;
                } while (atomicCAS(addr, old_val, new_val) != old_val);
#endif
            } else {
                output[target_idx] = source[idx];
            }
        }
    }
}

// Specialization for __nv_bfloat16 atomicAdd in put
template<>
__global__ void put_kernel_impl<__nv_bfloat16>(
    __nv_bfloat16* __restrict__ output,
    const int64_t* __restrict__ indices,
    const __nv_bfloat16* __restrict__ source,
    int64_t num_indices,
    int64_t total_size,
    bool accumulate
) {
    TENZOR_CUDA_KERNEL_LOOP(idx, num_indices) {
        int64_t target_idx = indices[idx];
        if (target_idx < 0) {
            target_idx += total_size;
        }
        if (target_idx >= 0 && target_idx < total_size) {
            if (accumulate) {
#if __CUDA_ARCH__ >= 800
                atomicAdd(&output[target_idx], source[idx]);
#else
                float val = __bfloat162float(source[idx]);
                unsigned int* addr = reinterpret_cast<unsigned int*>(&output[target_idx & ~1]);
                unsigned int old_val, new_val;
                do {
                    old_val = atomicCAS(addr, 0u, 0u);  // Atomic initial read
                    __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(&old_val);
                    // W.7: NaN-preserving conversion.
                    __nv_bfloat16 result = ::tenzor::cuda::safe_f2bf16(::tenzor::cuda::safe_bf162f(h[target_idx & 1]) + val);
                    new_val = old_val;
                    reinterpret_cast<__nv_bfloat16*>(&new_val)[target_idx & 1] = result;
                } while (atomicCAS(addr, old_val, new_val) != old_val);
#endif
            } else {
                output[target_idx] = source[idx];
            }
        }
    }
}

auto put_kernel(Tensor& input, const Tensor& indices, const Tensor& source,
                bool accumulate, cudaStream_t stream) -> Tensor {
    Tensor output = input.clone();
    int64_t num_indices = indices.numel();
    int64_t total_size = input.numel();

    if (num_indices == 0) return output;

    int blocks = get_num_blocks(num_indices);

    switch (input.dtype()) {
        case DType::Float32:
            put_kernel_impl<float><<<blocks, BLOCK_SIZE, 0, stream>>>(
                output.data<float>(), indices.data<int64_t>(), source.data<float>(),
                num_indices, total_size, accumulate);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::Float64:
            put_kernel_impl<double><<<blocks, BLOCK_SIZE, 0, stream>>>(
                output.data<double>(), indices.data<int64_t>(), source.data<double>(),
                num_indices, total_size, accumulate);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::Int32:
            put_kernel_impl<int32_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
                output.data<int32_t>(), indices.data<int64_t>(), source.data<int32_t>(),
                num_indices, total_size, accumulate);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::Int64:
            put_kernel_impl<int64_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
                output.data<int64_t>(), indices.data<int64_t>(), source.data<int64_t>(),
                num_indices, total_size, accumulate);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::Int8:
            put_kernel_impl<int8_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
                output.data<int8_t>(), indices.data<int64_t>(), source.data<int8_t>(),
                num_indices, total_size, accumulate);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::UInt8:
            put_kernel_impl<uint8_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
                output.data<uint8_t>(), indices.data<int64_t>(), source.data<uint8_t>(),
                num_indices, total_size, accumulate);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::Float16:
            put_kernel_impl<__half><<<blocks, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<__half*>(output.data_ptr()),
                indices.data<int64_t>(),
                reinterpret_cast<const __half*>(source.data_ptr()),
                num_indices, total_size, accumulate);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::BFloat16:
            put_kernel_impl<__nv_bfloat16><<<blocks, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                indices.data<int64_t>(),
                reinterpret_cast<const __nv_bfloat16*>(source.data_ptr()),
                num_indices, total_size, accumulate);
            CUDA_CHECK(cudaGetLastError());
            break;
        default:
            throw std::runtime_error("put_kernel: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// EmbeddingBag forward aggregation kernel
// ============================================================================

// Each block handles one bag. Threads within a block cooperate on reduction.
template<typename T>
__global__ void embedding_bag_sum_kernel(
    const T* embeddings,       // [total_elements, embedding_dim]
    const int64_t* offsets,    // [num_bags] or [num_bags+1]
    T* output,                 // [num_bags, embedding_dim]
    int64_t num_bags,
    int64_t total_elements,
    int64_t embedding_dim,
    int64_t offsets_size,      // actual size of offsets tensor
    bool divide_by_count)      // true for mean mode
{
    int64_t bag = blockIdx.x;
    if (bag >= num_bags) return;

    int64_t start = offsets[bag];
    int64_t end = (bag + 1 < offsets_size) ? offsets[bag + 1] : total_elements;
    int64_t bag_size = end - start;

    // Each thread handles a subset of embedding dimensions
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
__global__ void embedding_bag_max_kernel(
    const T* embeddings,
    const int64_t* offsets,
    T* output,
    int64_t* max_indices,   // [num_bags, embedding_dim] global argmax element index
    int64_t num_bags,
    int64_t total_elements,
    int64_t embedding_dim,
    int64_t offsets_size)
{
    int64_t bag = blockIdx.x;
    if (bag >= num_bags) return;

    int64_t start = offsets[bag];
    int64_t end = (bag + 1 < offsets_size) ? offsets[bag + 1] : total_elements;

    if (start >= end) return;  // empty bag: max_indices stays at its -1 prefill

    for (int64_t j = threadIdx.x; j < embedding_dim; j += blockDim.x) {
        T max_val = embeddings[start * embedding_dim + j];
        int64_t arg = start;
        for (int64_t i = start + 1; i < end; ++i) {
            T val = embeddings[i * embedding_dim + j];
            if (val > max_val) { max_val = val; arg = i; }  // strict '>': first wins
        }
        output[bag * embedding_dim + j] = max_val;
        if (max_indices != nullptr) max_indices[bag * embedding_dim + j] = arg;
    }
}

auto embedding_bag_forward_kernel(const Tensor& embeddings, const Tensor& offsets,
                                   const std::string& mode, int64_t embedding_dim,
                                   bool include_last_offset, cudaStream_t stream) -> std::vector<Tensor> {
    int64_t total_elements = embeddings.shape()[0];
    int64_t offsets_size = offsets.numel();
    int64_t num_bags = include_last_offset ? (offsets_size - 1) : offsets_size;

    bool is_mean = (mode == "mean");
    bool is_max = (mode == "max");

    if (num_bags <= 0) {
        return {tenzor::zeros({0, embedding_dim}, embeddings.dtype(), embeddings.device()),
                tenzor::zeros({0}, DType::Int64, embeddings.device())};
    }

    auto output = tenzor::zeros({num_bags, embedding_dim}, embeddings.dtype(), embeddings.device());

    // For max mode also emit the per-(bag,feature) global argmax element index
    // (-1 for empty bags), so the autograd node can route the gradient exactly
    // on-device. Empty / unused otherwise.
    Tensor max_indices = is_max
        ? tenzor::full({num_bags, embedding_dim}, static_cast<double>(-1),
                       DType::Int64, embeddings.device())
        : tenzor::zeros({0}, DType::Int64, embeddings.device());
    int64_t* argmax_ptr = is_max ? max_indices.data<int64_t>() : nullptr;

    int threads = std::min(static_cast<int>(embedding_dim), 256);
    int blocks = static_cast<int>(num_bags);

    switch (embeddings.dtype()) {
        case DType::Float32:
            if (is_max) {
                embedding_bag_max_kernel<float><<<blocks, threads, 0, stream>>>(
                    embeddings.data<float>(), offsets.data<int64_t>(),
                    output.data<float>(), argmax_ptr, num_bags, total_elements,
                    embedding_dim, offsets_size);
            } else {
                embedding_bag_sum_kernel<float><<<blocks, threads, 0, stream>>>(
                    embeddings.data<float>(), offsets.data<int64_t>(),
                    output.data<float>(), num_bags, total_elements,
                    embedding_dim, offsets_size, is_mean);
            }
            break;
        case DType::Float64:
            if (is_max) {
                embedding_bag_max_kernel<double><<<blocks, threads, 0, stream>>>(
                    embeddings.data<double>(), offsets.data<int64_t>(),
                    output.data<double>(), argmax_ptr, num_bags, total_elements,
                    embedding_dim, offsets_size);
            } else {
                embedding_bag_sum_kernel<double><<<blocks, threads, 0, stream>>>(
                    embeddings.data<double>(), offsets.data<int64_t>(),
                    output.data<double>(), num_bags, total_elements,
                    embedding_dim, offsets_size, is_mean);
            }
            break;
        case DType::Float16:
            if (is_max) {
                embedding_bag_max_kernel<__half><<<blocks, threads, 0, stream>>>(
                    reinterpret_cast<const __half*>(embeddings.data_ptr()),
                    offsets.data<int64_t>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    argmax_ptr, num_bags, total_elements, embedding_dim, offsets_size);
            } else {
                embedding_bag_sum_kernel<__half><<<blocks, threads, 0, stream>>>(
                    reinterpret_cast<const __half*>(embeddings.data_ptr()),
                    offsets.data<int64_t>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    num_bags, total_elements, embedding_dim, offsets_size, is_mean);
            }
            break;
        case DType::BFloat16:
            if (is_max) {
                embedding_bag_max_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(embeddings.data_ptr()),
                    offsets.data<int64_t>(),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    argmax_ptr, num_bags, total_elements, embedding_dim, offsets_size);
            } else {
                embedding_bag_sum_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(embeddings.data_ptr()),
                    offsets.data<int64_t>(),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    num_bags, total_elements, embedding_dim, offsets_size, is_mean);
            }
            break;
        default:
            throw std::runtime_error("embedding_bag_forward: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return {output, max_indices};
}

// ============================================================================
// SearchSorted kernel: binary search per element in sorted 1-D sequence
// ============================================================================

template<typename T>
__global__ void searchsorted_kernel_impl(
    const T* sorted_sequence,
    const T* values,
    int64_t* output,
    int64_t seq_len,
    int64_t num_values,
    bool right) {

    TENZOR_CUDA_KERNEL_LOOP(i, num_values) {
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

auto searchsorted_kernel(const Tensor& sorted_sequence, const Tensor& values,
                          bool right, cudaStream_t stream) -> Tensor {
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

    switch (sorted_sequence.dtype()) {
        case DType::Float32:
            LAUNCH_KERNEL(searchsorted_kernel_impl<float>, num_values, stream,
                seq_cont.data<float>(), val_cont.data<float>(), out_ptr,
                seq_len, num_values, right);
            break;
        case DType::Float64:
            LAUNCH_KERNEL(searchsorted_kernel_impl<double>, num_values, stream,
                seq_cont.data<double>(), val_cont.data<double>(), out_ptr,
                seq_len, num_values, right);
            break;
        case DType::Int32:
            LAUNCH_KERNEL(searchsorted_kernel_impl<int32_t>, num_values, stream,
                seq_cont.data<int32_t>(), val_cont.data<int32_t>(), out_ptr,
                seq_len, num_values, right);
            break;
        case DType::Int64:
            LAUNCH_KERNEL(searchsorted_kernel_impl<int64_t>, num_values, stream,
                seq_cont.data<int64_t>(), val_cont.data<int64_t>(), out_ptr,
                seq_len, num_values, right);
            break;
        case DType::Float16: {
            // Convert to Float32 on device
            auto seq_f32 = sorted_sequence.to(DType::Float32).contiguous();
            auto val_f32 = values.to(DType::Float32).contiguous();
            LAUNCH_KERNEL(searchsorted_kernel_impl<float>, num_values, stream,
                seq_f32.data<float>(), val_f32.data<float>(), out_ptr,
                seq_len, num_values, right);
            break;
        }
        case DType::BFloat16: {
            auto seq_f32 = sorted_sequence.to(DType::Float32).contiguous();
            auto val_f32 = values.to(DType::Float32).contiguous();
            LAUNCH_KERNEL(searchsorted_kernel_impl<float>, num_values, stream,
                seq_f32.data<float>(), val_f32.data<float>(), out_ptr,
                seq_len, num_values, right);
            break;
        }
        default:
            throw std::runtime_error("searchsorted: unsupported dtype");
    }

    return result;
}

// ============================================================================
// EmbeddingBag Backward
// ============================================================================

// Scatter-add into rows selected by `indices` (the original vocabulary ids
// the EmbeddingBag forward looked up). Each bag spans indices[start..end];
// the upstream gradient grad_output[bag] is distributed to every row in that
// bag (divided by bag_size for mean reduction).
template<typename T>
__global__ void embedding_bag_backward_kernel_cuda(
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
            // Skip out-of-range indices defensively. Host-side validation
            // (CPU kernel) throws; GPU drops them to keep the kernel safe.
            if (row < 0 || row >= num_embeddings) continue;
            atomicAdd(&grad_weight[row * embedding_dim + j], grad_val);
        }
    }
}

auto embedding_bag_backward_kernel(const Tensor& grad_output,
                                   const Tensor& indices,
                                   const Tensor& offsets,
                                   const OpAttributes& attrs,
                                   cudaStream_t stream) -> Tensor {
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
    CUDA_CHECK(cudaMemsetAsync(grad_weight.data_ptr(), 0,
                               num_embeddings * embedding_dim * dtype_size(grad_output.dtype()), stream));

    int threads = std::min(static_cast<int>(embedding_dim), 256);
    int blocks = static_cast<int>(num_bags);
    bool is_mean = (mode == "mean");

    switch (grad_output.dtype()) {
        case DType::Float32:
            embedding_bag_backward_kernel_cuda<float><<<blocks, threads, 0, stream>>>(
                grad_output.data<float>(), indices.data<int64_t>(), offsets.data<int64_t>(),
                grad_weight.data<float>(), num_bags, total_elements,
                embedding_dim, num_embeddings, offsets_size, is_mean);
            break;
        case DType::Float64:
            embedding_bag_backward_kernel_cuda<double><<<blocks, threads, 0, stream>>>(
                grad_output.data<double>(), indices.data<int64_t>(), offsets.data<int64_t>(),
                grad_weight.data<double>(), num_bags, total_elements,
                embedding_dim, num_embeddings, offsets_size, is_mean);
            break;
        default:
            throw std::runtime_error("embedding_bag_backward: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return grad_weight;
}

// ============================================================================
// take_along_dim kernel
// ============================================================================

template<typename T>
__global__ void take_along_dim_cuda_kernel(
    const T* __restrict__ input, const int64_t* __restrict__ indices, T* __restrict__ output,
    int64_t numel, int64_t in_dim_size, int64_t idx_dim_size, int64_t inner_size)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;

    int64_t outer = i / (idx_dim_size * inner_size);
    int64_t rem = i % (idx_dim_size * inner_size);
    int64_t inner = rem % inner_size;

    int64_t src_idx = indices[i];
    if (src_idx < 0) src_idx += in_dim_size;

    int64_t in_offset = outer * (in_dim_size * inner_size) + src_idx * inner_size + inner;
    output[i] = input[in_offset];
}

auto take_along_dim_kernel(const Tensor& input, const Tensor& indices, int64_t dim,
                           cudaStream_t stream) -> Tensor {
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

    int blocks = get_num_blocks(numel);
    const int64_t* idx_ptr = indices.data<int64_t>();

    switch (input.dtype()) {
        case DType::Float32:
            take_along_dim_cuda_kernel<float><<<blocks, BLOCK_SIZE, 0, stream>>>(
                input.data<float>(), idx_ptr, output.data<float>(),
                numel, in_dim_size, idx_dim_size, inner_size);
            break;
        case DType::Float64:
            take_along_dim_cuda_kernel<double><<<blocks, BLOCK_SIZE, 0, stream>>>(
                input.data<double>(), idx_ptr, output.data<double>(),
                numel, in_dim_size, idx_dim_size, inner_size);
            break;
        case DType::Int32:
            take_along_dim_cuda_kernel<int32_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
                input.data<int32_t>(), idx_ptr, output.data<int32_t>(),
                numel, in_dim_size, idx_dim_size, inner_size);
            break;
        case DType::Int64:
            take_along_dim_cuda_kernel<int64_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
                input.data<int64_t>(), idx_ptr, output.data<int64_t>(),
                numel, in_dim_size, idx_dim_size, inner_size);
            break;
        case DType::Float16:
            take_along_dim_cuda_kernel<__half><<<blocks, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const __half*>(input.data_ptr()), idx_ptr,
                reinterpret_cast<__half*>(output.data_ptr()),
                numel, in_dim_size, idx_dim_size, inner_size);
            break;
        case DType::BFloat16:
            take_along_dim_cuda_kernel<__nv_bfloat16><<<blocks, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()), idx_ptr,
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                numel, in_dim_size, idx_dim_size, inner_size);
            break;
        default:
            throw std::runtime_error("take_along_dim CUDA: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// masked_scatter kernel — prefix sum on mask, then parallel scatter
// ============================================================================

__global__ void masked_scatter_write_kernel_f32(
    const float* __restrict__ input, const bool* __restrict__ mask,
    const float* __restrict__ source, const int64_t* __restrict__ prefix_sum,
    float* __restrict__ output, int64_t numel)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    output[i] = mask[i] ? source[prefix_sum[i]] : input[i];
}

__global__ void masked_scatter_write_kernel_f64(
    const double* __restrict__ input, const bool* __restrict__ mask,
    const double* __restrict__ source, const int64_t* __restrict__ prefix_sum,
    double* __restrict__ output, int64_t numel)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    output[i] = mask[i] ? source[prefix_sum[i]] : input[i];
}

__global__ void masked_scatter_write_kernel_i32(
    const int32_t* __restrict__ input, const bool* __restrict__ mask,
    const int32_t* __restrict__ source, const int64_t* __restrict__ prefix_sum,
    int32_t* __restrict__ output, int64_t numel)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    output[i] = mask[i] ? source[prefix_sum[i]] : input[i];
}

__global__ void masked_scatter_write_kernel_i64(
    const int64_t* __restrict__ input, const bool* __restrict__ mask,
    const int64_t* __restrict__ source, const int64_t* __restrict__ prefix_sum,
    int64_t* __restrict__ output, int64_t numel)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    output[i] = mask[i] ? source[prefix_sum[i]] : input[i];
}

// Compute exclusive prefix sum of mask (bool -> int64)
__global__ void mask_to_int64_kernel(const bool* __restrict__ mask, int64_t* __restrict__ out, int64_t n)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = mask[i] ? 1 : 0;
}

auto masked_scatter_kernel(const Tensor& input, const Tensor& mask,
                           const Tensor& source, cudaStream_t stream) -> Tensor {
    int64_t numel = input.numel();
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    if (numel == 0) return output;

    // Build exclusive prefix sum of mask to get write positions
    int64_t* d_int_mask = nullptr;
    int64_t* d_prefix = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_int_mask, numel * sizeof(int64_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_prefix, numel * sizeof(int64_t), stream));

    int blocks = get_num_blocks(numel);
    mask_to_int64_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(mask.data<bool>(), d_int_mask, numel);

    // Simple sequential prefix sum via thrust-style scan
    // Use CUB for exclusive scan
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_int_mask, d_prefix, numel, stream);
    CUDA_CHECK(cudaMallocAsync(&d_temp, temp_bytes, stream));
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_int_mask, d_prefix, numel, stream);
    CUDA_CHECK(cudaFreeAsync(d_temp, stream));
    CUDA_CHECK(cudaFreeAsync(d_int_mask, stream));

    switch (input.dtype()) {
        case DType::Float32:
            masked_scatter_write_kernel_f32<<<blocks, BLOCK_SIZE, 0, stream>>>(
                input.data<float>(), mask.data<bool>(), source.data<float>(),
                d_prefix, output.data<float>(), numel);
            break;
        case DType::Float64:
            masked_scatter_write_kernel_f64<<<blocks, BLOCK_SIZE, 0, stream>>>(
                input.data<double>(), mask.data<bool>(), source.data<double>(),
                d_prefix, output.data<double>(), numel);
            break;
        case DType::Int32:
            masked_scatter_write_kernel_i32<<<blocks, BLOCK_SIZE, 0, stream>>>(
                input.data<int32_t>(), mask.data<bool>(), source.data<int32_t>(),
                d_prefix, output.data<int32_t>(), numel);
            break;
        case DType::Int64:
            masked_scatter_write_kernel_i64<<<blocks, BLOCK_SIZE, 0, stream>>>(
                input.data<int64_t>(), mask.data<bool>(), source.data<int64_t>(),
                d_prefix, output.data<int64_t>(), numel);
            break;
        default:
            CUDA_CHECK(cudaFreeAsync(d_prefix, stream));
            throw std::runtime_error("masked_scatter CUDA: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(d_prefix, stream));
    return output;
}

// ============================================================================
// tril_indices / triu_indices — native CUDA kernels
// ============================================================================

// Cumulative element count for tril through row r (inclusive).
// Each row i contributes min(col, max(0, i + offset + 1)) elements.
// Rows before first_nonempty contribute 0. Partial rows form an arithmetic
// series; full rows contribute col each.
__device__ inline int64_t tril_cumcount(int64_t r, int64_t col, int64_t offset) {
    // first row that has any element
    int64_t first = max(static_cast<int64_t>(0), -offset);
    if (r < first) return 0;

    // row index where the contribution first reaches col (becomes "full")
    int64_t full_from = col - offset - 1; // row index threshold

    if (full_from <= first) {
        // all non-empty rows are full rows
        return (r - first + 1) * col;
    }

    if (r < full_from) {
        // all rows in [first, r] are partial
        // contribution of row i = i + offset + 1
        // sum from i=first to r: sum of (i + offset + 1)
        int64_t count = r - first + 1;
        int64_t first_val = first + offset + 1;
        int64_t last_val  = r + offset + 1;
        return count * (first_val + last_val) / 2;
    }

    // mixed: partial rows [first, full_from-1] + full rows [full_from, r]
    int64_t partial_count = full_from - first;
    int64_t first_val = first + offset + 1;
    int64_t last_val  = full_from - 1 + offset + 1;
    int64_t partial_sum = partial_count * (first_val + last_val) / 2;
    int64_t full_sum = (r - full_from + 1) * col;
    return partial_sum + full_sum;
}

// Cumulative element count for triu through row r (inclusive).
// Each row i contributes max(0, col - max(0, i + offset)) elements.
__device__ inline int64_t triu_cumcount(int64_t r, int64_t col, int64_t offset) {
    int64_t total = 0;
    // Rows where i + offset <= 0 contribute col elements each.
    // Rows where i + offset >= col contribute 0.
    // In between: col - (i + offset).

    // last row that contributes col elements: i + offset <= 0 => i <= -offset
    int64_t last_full = min(r, -offset);
    if (last_full >= 0) {
        total += (last_full + 1) * col;
    }

    // first partial row
    int64_t first_partial = max(static_cast<int64_t>(0), last_full + 1);
    // last row that contributes anything: i + offset < col => i < col - offset
    int64_t last_nonempty = min(r, col - offset - 1);

    if (first_partial <= last_nonempty) {
        // contribution of row i = col - (i + offset)
        // sum from i=first_partial to last_nonempty
        int64_t count = last_nonempty - first_partial + 1;
        int64_t first_val = col - (first_partial + offset);
        int64_t last_val  = col - (last_nonempty + offset);
        total += count * (first_val + last_val) / 2;
    }

    return total;
}

__global__ void tril_indices_kernel_impl(
    int64_t* row_out,
    int64_t* col_out,
    int64_t n,
    int64_t row,
    int64_t col,
    int64_t offset) {

    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        // Binary search for the row: find smallest r such that tril_cumcount(r) > idx
        int64_t first = max(static_cast<int64_t>(0), -offset);
        int64_t lo = first, hi = row - 1;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if (tril_cumcount(mid, col, offset) <= idx) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        int64_t r = lo;
        int64_t prev = (r > first) ? tril_cumcount(r - 1, col, offset) : 0;
        int64_t c = idx - prev;
        row_out[idx] = r;
        col_out[idx] = c;
    }
}

__global__ void triu_indices_kernel_impl(
    int64_t* row_out,
    int64_t* col_out,
    int64_t n,
    int64_t row,
    int64_t col,
    int64_t offset) {

    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        // Binary search for the row: find smallest r such that triu_cumcount(r) > idx
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
        int64_t prev = (r > 0) ? triu_cumcount(r - 1, col, offset) : 0;
        int64_t start_c = max(static_cast<int64_t>(0), r + offset);
        int64_t c = start_c + (idx - prev);
        row_out[idx] = r;
        col_out[idx] = c;
    }
}

auto tril_indices_kernel(int64_t row, int64_t col, int64_t offset,
                         cudaStream_t stream) -> Tensor {
    // Compute total element count on host (O(1) closed-form)
    int64_t first = std::max(static_cast<int64_t>(0), -offset);
    int64_t n = 0;
    if (first < row) {
        int64_t full_from = col - offset - 1;
        if (full_from <= first) {
            n = (row - first) * col;
        } else if (full_from >= row) {
            int64_t count = row - first;
            int64_t first_val = first + offset + 1;
            int64_t last_val  = row - 1 + offset + 1;
            n = count * (first_val + last_val) / 2;
        } else {
            int64_t partial_count = full_from - first;
            int64_t first_val = first + offset + 1;
            int64_t last_val  = full_from - 1 + offset + 1;
            int64_t partial_sum = partial_count * (first_val + last_val) / 2;
            int64_t full_sum = (row - full_from) * col;
            n = partial_sum + full_sum;
        }
    }

    if (n == 0) return tenzor::empty({2, 0}, DType::Int64, Device::cuda(0));

    Tensor output({2, n}, DType::Int64, Device::cuda(0));
    int64_t* ptr = output.data<int64_t>();
    int blocks = get_num_blocks(n);
    tril_indices_kernel_impl<<<blocks, BLOCK_SIZE, 0, stream>>>(
        ptr, ptr + n, n, row, col, offset);
    CUDA_CHECK(cudaGetLastError());
    return output;
}

auto triu_indices_kernel(int64_t row, int64_t col, int64_t offset,
                         cudaStream_t stream) -> Tensor {
    // Compute total element count on host (O(1) closed-form)
    int64_t n = 0;
    int64_t last_full = std::min(row - 1, -offset);
    if (last_full >= 0) {
        n += (last_full + 1) * col;
    }
    int64_t first_partial = std::max(static_cast<int64_t>(0), last_full + 1);
    int64_t last_nonempty = std::min(row - 1, col - offset - 1);
    if (first_partial <= last_nonempty) {
        int64_t count = last_nonempty - first_partial + 1;
        int64_t first_val = col - (first_partial + offset);
        int64_t last_val  = col - (last_nonempty + offset);
        n += count * (first_val + last_val) / 2;
    }

    if (n == 0) return tenzor::empty({2, 0}, DType::Int64, Device::cuda(0));

    Tensor output({2, n}, DType::Int64, Device::cuda(0));
    int64_t* ptr = output.data<int64_t>();
    int blocks = get_num_blocks(n);
    triu_indices_kernel_impl<<<blocks, BLOCK_SIZE, 0, stream>>>(
        ptr, ptr + n, n, row, col, offset);
    CUDA_CHECK(cudaGetLastError());
    return output;
}

} // namespace cuda
} // namespace tenzor
