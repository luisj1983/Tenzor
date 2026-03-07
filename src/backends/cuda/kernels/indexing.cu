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
#include "cuda_launch_utils.cuh"
#include <stdexcept>
#include <vector>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>

namespace tenzor {

// Forward-declare: nvcc can't include creation.hpp (uses std::expected)
auto zeros(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor;

namespace cuda {

// Centralized error checking
#include "../cuda_error.hpp"

#define CUDA_GRID_STRIDE_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

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
    int64_t total_output) {

    CUDA_GRID_STRIDE_LOOP(idx, total_output) {
        // Decompose output index into (outer, index_pos, inner)
        int64_t inner_idx = idx % inner_size;
        int64_t temp = idx / inner_size;
        int64_t index_pos = temp % num_indices;
        int64_t outer_idx = temp / num_indices;

        // Get the actual index value with bounds handling
        int64_t selected_idx = static_cast<int64_t>(indices[index_pos]);
        if (selected_idx < 0) selected_idx += dim_size;
        selected_idx = max(int64_t(0), min(selected_idx, dim_size - 1));

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

    #define LAUNCH_INDEX_SELECT(T) \
        if (idx_is_int32) \
            index_select_kernel_impl<T, int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>( \
                input.data<T>(), index.data<int32_t>(), output.data<T>(), \
                num_indices, dim_size, outer_size, inner_size, total_output); \
        else \
            index_select_kernel_impl<T, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>( \
                input.data<T>(), index.data<int64_t>(), output.data<T>(), \
                num_indices, dim_size, outer_size, inner_size, total_output); \
        CUDA_CHECK(cudaGetLastError())

    switch (input.dtype()) {
        case DType::Float32: LAUNCH_INDEX_SELECT(float); break;
        case DType::Float64: LAUNCH_INDEX_SELECT(double); break;
        case DType::Int32:   LAUNCH_INDEX_SELECT(int32_t); break;
        case DType::Int64:   LAUNCH_INDEX_SELECT(int64_t); break;
        case DType::Int8:    LAUNCH_INDEX_SELECT(int8_t); break;
        case DType::UInt8:   LAUNCH_INDEX_SELECT(uint8_t); break;
        case DType::Float16:
            if (idx_is_int32)
                index_select_kernel_impl<__half, int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __half*>(input.data_ptr()),
                    index.data<int32_t>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    num_indices, dim_size, outer_size, inner_size, total_output);
            else
                index_select_kernel_impl<__half, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __half*>(input.data_ptr()),
                    index.data<int64_t>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    num_indices, dim_size, outer_size, inner_size, total_output);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::BFloat16:
            if (idx_is_int32)
                index_select_kernel_impl<__nv_bfloat16, int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                    index.data<int32_t>(),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    num_indices, dim_size, outer_size, inner_size, total_output);
            else
                index_select_kernel_impl<__nv_bfloat16, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                    index.data<int64_t>(),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    num_indices, dim_size, outer_size, inner_size, total_output);
            CUDA_CHECK(cudaGetLastError());
            break;
        default:
            throw std::runtime_error("index_select: unsupported dtype");
    }

    #undef LAUNCH_INDEX_SELECT

    CUDA_CHECK(cudaGetLastError());
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
    int64_t total_output) {

    CUDA_GRID_STRIDE_LOOP(idx, total_output) {
        // Decompose output index into (outer, index_pos, inner)
        int64_t inner_idx = idx % inner_size;
        int64_t temp = idx / inner_size;
        int64_t index_pos = temp % index_dim_size;
        int64_t outer_idx = temp / index_dim_size;

        // Get the index value at this position
        int64_t index_offset = outer_idx * index_dim_size * inner_size +
                               index_pos * inner_size + inner_idx;
        int64_t gather_idx = static_cast<int64_t>(indices[index_offset]);

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
                outer_size, dim_size, inner_size, index_dim_size, total_output); \
        else \
            gather_kernel_impl<T, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>( \
                input.data<T>(), index.data<int64_t>(), output.data<T>(), \
                outer_size, dim_size, inner_size, index_dim_size, total_output); \
        CUDA_CHECK(cudaGetLastError())

    switch (input.dtype()) {
        case DType::Float32: LAUNCH_GATHER(float); break;
        case DType::Float64: LAUNCH_GATHER(double); break;
        case DType::Int32:   LAUNCH_GATHER(int32_t); break;
        case DType::Int64:   LAUNCH_GATHER(int64_t); break;
        case DType::Int8:    LAUNCH_GATHER(int8_t); break;
        case DType::UInt8:   LAUNCH_GATHER(uint8_t); break;
        case DType::Float16:
            if (idx_is_int32)
                gather_kernel_impl<__half, int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __half*>(input.data_ptr()),
                    index.data<int32_t>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_output);
            else
                gather_kernel_impl<__half, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __half*>(input.data_ptr()),
                    index.data<int64_t>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_output);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::BFloat16:
            if (idx_is_int32)
                gather_kernel_impl<__nv_bfloat16, int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                    index.data<int32_t>(),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_output);
            else
                gather_kernel_impl<__nv_bfloat16, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                    index.data<int64_t>(),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_output);
            CUDA_CHECK(cudaGetLastError());
            break;
        default:
            throw std::runtime_error("gather: unsupported dtype");
    }

    #undef LAUNCH_GATHER

    CUDA_CHECK(cudaGetLastError());
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
    CUDA_GRID_STRIDE_LOOP(idx, n) {
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
    int64_t total_scatter) {

    CUDA_GRID_STRIDE_LOOP(idx, total_scatter) {
        int64_t inner_idx = idx % inner_size;
        int64_t temp = idx / inner_size;
        int64_t index_pos = temp % index_dim_size;
        int64_t outer_idx = temp / index_dim_size;

        int64_t index_offset = outer_idx * index_dim_size * inner_size +
                               index_pos * inner_size + inner_idx;
        int64_t scatter_idx = static_cast<int64_t>(indices[index_offset]);
        if (scatter_idx < 0) scatter_idx += dim_size;
        scatter_idx = max(int64_t(0), min(scatter_idx, dim_size - 1));

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

    #define LAUNCH_SCATTER(T) \
        copy_kernel_impl<T><<<num_blocks_copy, BLOCK_SIZE, 0, stream>>>( \
            input.data<T>(), output.data<T>(), total_input); \
        CUDA_CHECK(cudaGetLastError()); \
        if (idx_is_int32) \
            scatter_values_kernel_impl<T, int32_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>( \
                index.data<int32_t>(), src.data<T>(), output.data<T>(), \
                outer_size, dim_size, inner_size, index_dim_size, total_scatter); \
        else \
            scatter_values_kernel_impl<T, int64_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>( \
                index.data<int64_t>(), src.data<T>(), output.data<T>(), \
                outer_size, dim_size, inner_size, index_dim_size, total_scatter); \
        CUDA_CHECK(cudaGetLastError())

    switch (input.dtype()) {
        case DType::Float32: LAUNCH_SCATTER(float); break;
        case DType::Float64: LAUNCH_SCATTER(double); break;
        case DType::Int32:   LAUNCH_SCATTER(int32_t); break;
        case DType::Int64:   LAUNCH_SCATTER(int64_t); break;
        case DType::Int8:    LAUNCH_SCATTER(int8_t); break;
        case DType::UInt8:   LAUNCH_SCATTER(uint8_t); break;
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
                    outer_size, dim_size, inner_size, index_dim_size, total_scatter);
            else
                scatter_values_kernel_impl<__half, int64_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>(
                    index.data<int64_t>(),
                    reinterpret_cast<const __half*>(src.data_ptr()),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_scatter);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::BFloat16:
            copy_kernel_impl<__nv_bfloat16><<<num_blocks_copy, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), total_input);
            CUDA_CHECK(cudaGetLastError());
            // No sync needed: CUDA stream ordering guarantees sequential execution
            if (idx_is_int32)
                scatter_values_kernel_impl<__nv_bfloat16, int32_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>(
                    index.data<int32_t>(),
                    reinterpret_cast<const __nv_bfloat16*>(src.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_scatter);
            else
                scatter_values_kernel_impl<__nv_bfloat16, int64_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>(
                    index.data<int64_t>(),
                    reinterpret_cast<const __nv_bfloat16*>(src.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    outer_size, dim_size, inner_size, index_dim_size, total_scatter);
            CUDA_CHECK(cudaGetLastError());
            break;
        default:
            throw std::runtime_error("scatter: unsupported dtype");
    }

    #undef LAUNCH_SCATTER

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// scatter_add kernel — uses atomicAdd for overlapping indices
// ============================================================================

// Warp-level atomic helper: reduces values within a warp that target the same
// output offset, then one thread per unique target does a single atomicAdd.
// This reduces atomic contention by up to 32x under high-conflict patterns.
template<typename T>
__device__ void warp_reduce_atomic_add(T* output, int64_t output_offset, T value) {
    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;
    unsigned lane = threadIdx.x & 31;

    // Find lanes in this warp targeting the same output_offset
    // Use ballot to group matching lanes
    for (unsigned peer_offset = 0; peer_offset < 32; ) {
        // Broadcast the target offset from the lowest active lane
        int64_t leader_offset = __shfl_sync(FULL_MASK, output_offset, peer_offset);
        unsigned match_mask = __ballot_sync(FULL_MASK, output_offset == leader_offset);

        if (output_offset == leader_offset) {
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
                    unsigned long long old_val = *addr;
                    unsigned long long assumed;
                    do {
                        assumed = old_val;
                        unsigned long long desired = static_cast<unsigned long long>(
                            static_cast<int64_t>(assumed) + reduced);
                        old_val = atomicCAS(addr, assumed, desired);
                    } while (assumed != old_val);
                } else {
                    atomicAdd(&output[output_offset], reduced);
                }
            }
            return;
        }
        // Advance past all lanes matching this leader
        peer_offset = 32 - __clz(match_mask);
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
    int64_t total_scatter) {

    CUDA_GRID_STRIDE_LOOP(idx, total_scatter) {
        int64_t inner_idx = idx % inner_size;
        int64_t temp = idx / inner_size;
        int64_t index_pos = temp % index_dim_size;
        int64_t outer_idx = temp / index_dim_size;

        int64_t index_offset = outer_idx * index_dim_size * inner_size +
                               index_pos * inner_size + inner_idx;
        int64_t scatter_idx = static_cast<int64_t>(indices[index_offset]);
        if (scatter_idx < 0) scatter_idx += dim_size;
        scatter_idx = max(int64_t(0), min(scatter_idx, dim_size - 1));

        int64_t output_offset = outer_idx * dim_size * inner_size +
                                scatter_idx * inner_size +
                                inner_idx;

        warp_reduce_atomic_add(output, output_offset, src[idx]);
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

    // Step 1: Copy input to output
    int num_blocks_copy = get_num_blocks(total_input);
    copy_kernel_impl<float><<<num_blocks_copy, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const float*>(input.data_ptr()),
        reinterpret_cast<float*>(output.data_ptr()),
        (total_input * dtype_size(input.dtype()) + 3) / 4);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Scatter-add with atomicAdd
    if (total_scatter == 0) return output;
    int num_blocks_scatter = get_num_blocks(total_scatter);

    #define LAUNCH_SCATTER_ADD(T) \
        if (idx_is_int32) \
            scatter_add_kernel_impl<T, int32_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>( \
                index.data<int32_t>(), src.data<T>(), output.data<T>(), \
                outer_size, dim_size, inner_size, index_dim_size, total_scatter); \
        else \
            scatter_add_kernel_impl<T, int64_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>( \
                index.data<int64_t>(), src.data<T>(), output.data<T>(), \
                outer_size, dim_size, inner_size, index_dim_size, total_scatter); \
        CUDA_CHECK(cudaGetLastError())

    switch (input.dtype()) {
        case DType::Float32: LAUNCH_SCATTER_ADD(float); break;
        case DType::Float64: LAUNCH_SCATTER_ADD(double); break;
        case DType::Int32:   LAUNCH_SCATTER_ADD(int32_t); break;
        case DType::Int64:   LAUNCH_SCATTER_ADD(int64_t); break;
        default: throw std::runtime_error("scatter_add: unsupported dtype (atomicAdd requires float/double/int)");
    }

    #undef LAUNCH_SCATTER_ADD

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

    CUDA_GRID_STRIDE_LOOP(i, n) {
        output[i] = mask[i] ? fill_value : input[i];
    }
}

auto masked_fill_kernel(const Tensor& input, const Tensor& mask, double value,
                        cudaStream_t stream) -> Tensor {
    int64_t n = input.numel();

    std::vector<int64_t> output_shape(input.shape().begin(), input.shape().end());
    Tensor output(output_shape, input.dtype(), input.device());

    if (n == 0) return output;

    // Handle mask broadcasting if needed (assume same shape for now)
    const bool* mask_ptr = reinterpret_cast<const bool*>(mask.data_ptr());

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

    CUDA_GRID_STRIDE_LOOP(i, n) {
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
        default:
            throw std::runtime_error("where: unsupported dtype");
    }

    #undef LAUNCH_WHERE

    CUDA_CHECK(cudaGetLastError());
    return output;
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
    int64_t embedding_dim) {

    int64_t total_elements = num_indices * embedding_dim;

    CUDA_GRID_STRIDE_LOOP(idx, total_elements) {
        int64_t i = idx / embedding_dim;  // which index
        int64_t j = idx % embedding_dim;  // which embedding dimension

        int64_t token_idx = static_cast<int64_t>(indices[i]);
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

    #define LAUNCH_EMBEDDING(T) \
        if (idx_is_int32) { \
            auto [grid_size, block_size] = optimal_launch_config( \
                embedding_kernel_impl<T, int32_t>, total_elements); \
            embedding_kernel_impl<T, int32_t><<<grid_size, block_size, 0, stream>>>( \
                weight.data<T>(), indices.data<int32_t>(), output.data<T>(), \
                num_indices, embedding_dim); \
            CUDA_CHECK(cudaGetLastError()); \
        } else { \
            auto [grid_size, block_size] = optimal_launch_config( \
                embedding_kernel_impl<T, int64_t>, total_elements); \
            embedding_kernel_impl<T, int64_t><<<grid_size, block_size, 0, stream>>>( \
                weight.data<T>(), indices.data<int64_t>(), output.data<T>(), \
                num_indices, embedding_dim); \
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
                    num_indices, embedding_dim);
                CUDA_CHECK(cudaGetLastError());
            } else {
                auto [grid_size, block_size] = optimal_launch_config(
                    embedding_kernel_impl<__half, int64_t>, total_elements);
                embedding_kernel_impl<__half, int64_t><<<grid_size, block_size, 0, stream>>>(
                    reinterpret_cast<const __half*>(weight.data_ptr()),
                    indices.data<int64_t>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    num_indices, embedding_dim);
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
                    num_indices, embedding_dim);
                CUDA_CHECK(cudaGetLastError());
            } else {
                auto [grid_size, block_size] = optimal_launch_config(
                    embedding_kernel_impl<__nv_bfloat16, int64_t>, total_elements);
                embedding_kernel_impl<__nv_bfloat16, int64_t><<<grid_size, block_size, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
                    indices.data<int64_t>(),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    num_indices, embedding_dim);
                CUDA_CHECK(cudaGetLastError());
            }
            break;
        default:
            throw std::runtime_error("embedding: unsupported dtype");
    }

    #undef LAUNCH_EMBEDDING

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// embedding_backward kernel (gradient accumulation)
// ============================================================================

template<typename T, typename IndexT>
__global__ void embedding_backward_kernel_impl(
    const T* grad_output,      // [*, embedding_dim]
    const IndexT* indices,     // [*]
    T* grad_weight,            // [num_embeddings, embedding_dim]
    int64_t num_indices,
    int64_t embedding_dim) {

    int64_t total_elements = num_indices * embedding_dim;

    CUDA_GRID_STRIDE_LOOP(idx, total_elements) {
        int64_t i = idx / embedding_dim;  // which index
        int64_t j = idx % embedding_dim;  // which embedding dimension

        int64_t token_idx = static_cast<int64_t>(indices[i]);
        // Use atomicAdd for thread-safe accumulation
        atomicAdd(&grad_weight[token_idx * embedding_dim + j], grad_output[idx]);
    }
}

// Separate FP16 backward kernel (cannot partially specialize function templates)
template<typename IndexT>
__global__ void embedding_backward_fp16_kernel_impl(
    const __half* grad_output,
    const IndexT* indices,
    __half* grad_weight,
    int64_t num_indices,
    int64_t embedding_dim) {

    int64_t total_elements = num_indices * embedding_dim;

    CUDA_GRID_STRIDE_LOOP(idx, total_elements) {
        int64_t i = idx / embedding_dim;
        int64_t j = idx % embedding_dim;

        int64_t token_idx = static_cast<int64_t>(indices[i]);
#if __CUDA_ARCH__ >= 700
        atomicAdd(&grad_weight[token_idx * embedding_dim + j], grad_output[idx]);
#else
        // Fallback for older architectures: compare-and-swap based atomic add
        float val = __half2float(grad_output[idx]);
        unsigned int* addr = reinterpret_cast<unsigned int*>(&grad_weight[(token_idx * embedding_dim + j) & ~1]);
        unsigned int old_val, new_val;
        do {
            old_val = *addr;
            __half* h = reinterpret_cast<__half*>(&old_val);
            __half result = __float2half(__half2float(h[(token_idx * embedding_dim + j) & 1]) + val);
            new_val = old_val;
            reinterpret_cast<__half*>(&new_val)[(token_idx * embedding_dim + j) & 1] = result;
        } while (atomicCAS(addr, old_val, new_val) != old_val);
#endif
    }
}

// Separate BF16 backward kernel — atomicAdd(__nv_bfloat16) requires SM >= 80
template<typename IndexT>
__global__ void embedding_backward_bf16_kernel_impl(
    const __nv_bfloat16* grad_output,
    const IndexT* indices,
    __nv_bfloat16* grad_weight,
    int64_t num_indices,
    int64_t embedding_dim) {

    int64_t total_elements = num_indices * embedding_dim;

    CUDA_GRID_STRIDE_LOOP(idx, total_elements) {
        int64_t i = idx / embedding_dim;
        int64_t j = idx % embedding_dim;

        int64_t token_idx = static_cast<int64_t>(indices[i]);
#if __CUDA_ARCH__ >= 800
        atomicAdd(&grad_weight[token_idx * embedding_dim + j], grad_output[idx]);
#else
        // Fallback for SM < 80: CAS-based atomic add via float conversion
        float val = __bfloat162float(grad_output[idx]);
        unsigned int* addr = reinterpret_cast<unsigned int*>(&grad_weight[(token_idx * embedding_dim + j) & ~1]);
        unsigned int old_val, new_val;
        do {
            old_val = *addr;
            __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(&old_val);
            __nv_bfloat16 result = __float2bfloat16(__bfloat162float(h[(token_idx * embedding_dim + j) & 1]) + val);
            new_val = old_val;
            reinterpret_cast<__nv_bfloat16*>(&new_val)[(token_idx * embedding_dim + j) & 1] = result;
        } while (atomicCAS(addr, old_val, new_val) != old_val);
#endif
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

    int64_t total_elements = num_indices * embedding_dim;
    int num_blocks = get_num_blocks(total_elements);

    bool idx_is_int32 = (indices.dtype() == DType::Int32);
    bool idx_is_int64 = (indices.dtype() == DType::Int64);
    if (!idx_is_int32 && !idx_is_int64) {
        throw std::invalid_argument("embedding_backward: indices must be Int32 or Int64");
    }

    #define LAUNCH_EMBEDDING_BWD(T) \
        if (idx_is_int32) \
            embedding_backward_kernel_impl<T, int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>( \
                grad_output.data<T>(), indices.data<int32_t>(), grad_weight.data<T>(), \
                num_indices, embedding_dim); \
        else \
            embedding_backward_kernel_impl<T, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>( \
                grad_output.data<T>(), indices.data<int64_t>(), grad_weight.data<T>(), \
                num_indices, embedding_dim); \
        CUDA_CHECK(cudaGetLastError())

    switch (grad_output.dtype()) {
        case DType::Float32: LAUNCH_EMBEDDING_BWD(float); break;
        case DType::Float64: LAUNCH_EMBEDDING_BWD(double); break;
        case DType::Float16:
            if (idx_is_int32)
                embedding_backward_fp16_kernel_impl<int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __half*>(grad_output.data_ptr()),
                    indices.data<int32_t>(),
                    reinterpret_cast<__half*>(grad_weight.data_ptr()),
                    num_indices, embedding_dim);
            else
                embedding_backward_fp16_kernel_impl<int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __half*>(grad_output.data_ptr()),
                    indices.data<int64_t>(),
                    reinterpret_cast<__half*>(grad_weight.data_ptr()),
                    num_indices, embedding_dim);
            CUDA_CHECK(cudaGetLastError());
            break;
        case DType::BFloat16:
            if (idx_is_int32)
                embedding_backward_bf16_kernel_impl<int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
                    indices.data<int32_t>(),
                    reinterpret_cast<__nv_bfloat16*>(grad_weight.data_ptr()),
                    num_indices, embedding_dim);
            else
                embedding_backward_bf16_kernel_impl<int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
                    indices.data<int64_t>(),
                    reinterpret_cast<__nv_bfloat16*>(grad_weight.data_ptr()),
                    num_indices, embedding_dim);
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
    CUDA_GRID_STRIDE_LOOP(idx, total) {
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

    CUDA_GRID_STRIDE_LOOP(i, n) {
        flags[i] = (input[i] != static_cast<T>(0)) ? 1 : 0;
    }
}

// Specialization for __half
template<>
__global__ void nonzero_flag_kernel<__half>(
    const __half* input,
    int64_t* flags,
    int64_t n) {

    CUDA_GRID_STRIDE_LOOP(i, n) {
        flags[i] = (__hne(input[i], __float2half(0.0f))) ? 1 : 0;
    }
}

// Specialization for __nv_bfloat16
template<>
__global__ void nonzero_flag_kernel<__nv_bfloat16>(
    const __nv_bfloat16* input,
    int64_t* flags,
    int64_t n) {

    CUDA_GRID_STRIDE_LOOP(i, n) {
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

    CUDA_GRID_STRIDE_LOOP(i, num_indices) {
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
    CUDA_GRID_STRIDE_LOOP(idx, indices_size) {
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
    CUDA_GRID_STRIDE_LOOP(idx, num_indices) {
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
    CUDA_GRID_STRIDE_LOOP(idx, num_indices) {
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
                    old_val = *addr;
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
    CUDA_GRID_STRIDE_LOOP(idx, num_indices) {
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
                    old_val = *addr;
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
    CUDA_GRID_STRIDE_LOOP(idx, num_indices) {
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
    CUDA_GRID_STRIDE_LOOP(idx, num_indices) {
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
    CUDA_GRID_STRIDE_LOOP(idx, num_indices) {
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
                    old_val = *addr;
                    __half* h = reinterpret_cast<__half*>(&old_val);
                    __half result = __float2half(__half2float(h[target_idx & 1]) + val);
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
    CUDA_GRID_STRIDE_LOOP(idx, num_indices) {
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
                    old_val = *addr;
                    __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(&old_val);
                    __nv_bfloat16 result = __float2bfloat16(__bfloat162float(h[target_idx & 1]) + val);
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
                                   bool include_last_offset, cudaStream_t stream) -> Tensor {
    int64_t total_elements = embeddings.shape()[0];
    int64_t offsets_size = offsets.numel();
    int64_t num_bags = include_last_offset ? (offsets_size - 1) : offsets_size;

    if (num_bags <= 0) {
        return tenzor::zeros({0, embedding_dim}, embeddings.dtype(), embeddings.device());
    }

    auto output = tenzor::zeros({num_bags, embedding_dim}, embeddings.dtype(), embeddings.device());

    int threads = std::min(static_cast<int>(embedding_dim), 256);
    int blocks = static_cast<int>(num_bags);

    bool is_mean = (mode == "mean");
    bool is_max = (mode == "max");

    switch (embeddings.dtype()) {
        case DType::Float32:
            if (is_max) {
                embedding_bag_max_kernel<float><<<blocks, threads, 0, stream>>>(
                    embeddings.data<float>(), offsets.data<int64_t>(),
                    output.data<float>(), num_bags, total_elements,
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
                    output.data<double>(), num_bags, total_elements,
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
                    num_bags, total_elements, embedding_dim, offsets_size);
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
                    num_bags, total_elements, embedding_dim, offsets_size);
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
    return output;
}

} // namespace cuda
} // namespace tenzor
