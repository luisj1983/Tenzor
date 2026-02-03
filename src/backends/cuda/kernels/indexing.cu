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
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/caching_allocator.hpp"
#include <stdexcept>
#include <vector>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>

namespace tenzor {
namespace cuda {

// CUDA Helper macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUDA_GRID_STRIDE_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

constexpr int BLOCK_SIZE = 256;

inline int get_num_blocks(int64_t n, int block_size = BLOCK_SIZE) {
    int num_blocks = (n + block_size - 1) / block_size;
    // Ensure at least 1 block to avoid CUDA invalid argument error
    // Grid-stride loop will naturally handle n=0 by not executing any iterations
    return std::max(1, std::min(num_blocks, 65535));
}

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

        // Get the actual index value
        int64_t selected_idx = static_cast<int64_t>(indices[index_pos]);

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
                num_indices, dim_size, outer_size, inner_size, total_output)

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
                outer_size, dim_size, inner_size, index_dim_size, total_output)

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

template<typename T, typename IndexT>
__global__ void scatter_kernel_impl(
    const T* input,
    const IndexT* indices,
    const T* src,
    T* output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size,
    int64_t index_dim_size,
    int64_t total_input,
    int64_t total_scatter) {

    // First, copy input to output (if not in-place)
    CUDA_GRID_STRIDE_LOOP(idx, total_input) {
        output[idx] = input[idx];
    }

    __syncthreads();

    // Then scatter src values
    CUDA_GRID_STRIDE_LOOP(idx, total_scatter) {
        int64_t inner_idx = idx % inner_size;
        int64_t temp = idx / inner_size;
        int64_t index_pos = temp % index_dim_size;
        int64_t outer_idx = temp / index_dim_size;

        // Get the index value at this position
        int64_t index_offset = outer_idx * index_dim_size * inner_size +
                               index_pos * inner_size + inner_idx;
        int64_t scatter_idx = static_cast<int64_t>(indices[index_offset]);

        // Compute output offset
        int64_t output_offset = outer_idx * dim_size * inner_size +
                                scatter_idx * inner_size +
                                inner_idx;

        output[output_offset] = src[idx];
    }
}

// Separate kernel for copy phase
template<typename T>
__global__ void copy_kernel_impl(const T* input, T* output, int64_t n) {
    CUDA_GRID_STRIDE_LOOP(idx, n) {
        output[idx] = input[idx];
    }
}

// Separate kernel for scatter phase
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
        if (idx_is_int32) \
            scatter_values_kernel_impl<T, int32_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>( \
                index.data<int32_t>(), src.data<T>(), output.data<T>(), \
                outer_size, dim_size, inner_size, index_dim_size, total_scatter); \
        else \
            scatter_values_kernel_impl<T, int64_t><<<num_blocks_scatter, BLOCK_SIZE, 0, stream>>>( \
                index.data<int64_t>(), src.data<T>(), output.data<T>(), \
                outer_size, dim_size, inner_size, index_dim_size, total_scatter)

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
            break;
        default:
            throw std::runtime_error("scatter: unsupported dtype");
    }

    #undef LAUNCH_SCATTER

    CUDA_CHECK(cudaGetLastError());
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

    int num_blocks = get_num_blocks(n);

    // Handle mask broadcasting if needed (assume same shape for now)
    const bool* mask_ptr = reinterpret_cast<const bool*>(mask.data_ptr());

    #define LAUNCH_MASKED_FILL(T, cast_val) \
        masked_fill_kernel_impl<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>( \
            input.data<T>(), mask_ptr, static_cast<T>(cast_val), output.data<T>(), n)

    switch (input.dtype()) {
        case DType::Float32: LAUNCH_MASKED_FILL(float, value); break;
        case DType::Float64: LAUNCH_MASKED_FILL(double, value); break;
        case DType::Int32:   LAUNCH_MASKED_FILL(int32_t, value); break;
        case DType::Int64:   LAUNCH_MASKED_FILL(int64_t, value); break;
        case DType::Int8:    LAUNCH_MASKED_FILL(int8_t, value); break;
        case DType::UInt8:   LAUNCH_MASKED_FILL(uint8_t, value); break;
        case DType::Float16: {
            __half fill_val = __float2half(static_cast<float>(value));
            masked_fill_kernel_impl<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<const __half*>(input.data_ptr()),
                mask_ptr, fill_val,
                reinterpret_cast<__half*>(output.data_ptr()), n);
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

    int num_blocks = get_num_blocks(n);

    const bool* cond_ptr = reinterpret_cast<const bool*>(condition.data_ptr());

    #define LAUNCH_WHERE(T) \
        where_kernel_impl<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>( \
            cond_ptr, x.data<T>(), y.data<T>(), output.data<T>(), n)

    switch (x.dtype()) {
        case DType::Float32: LAUNCH_WHERE(float); break;
        case DType::Float64: LAUNCH_WHERE(double); break;
        case DType::Int32:   LAUNCH_WHERE(int32_t); break;
        case DType::Int64:   LAUNCH_WHERE(int64_t); break;
        case DType::Int8:    LAUNCH_WHERE(int8_t); break;
        case DType::UInt8:   LAUNCH_WHERE(uint8_t); break;
        case DType::Float16:
            where_kernel_impl<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                cond_ptr,
                reinterpret_cast<const __half*>(x.data_ptr()),
                reinterpret_cast<const __half*>(y.data_ptr()),
                reinterpret_cast<__half*>(output.data_ptr()), n);
            break;
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
    int num_blocks = get_num_blocks(total_elements);

    bool idx_is_int32 = (indices.dtype() == DType::Int32);
    bool idx_is_int64 = (indices.dtype() == DType::Int64);
    if (!idx_is_int32 && !idx_is_int64) {
        throw std::invalid_argument("embedding: indices must be Int32 or Int64");
    }

    #define LAUNCH_EMBEDDING(T) \
        if (idx_is_int32) \
            embedding_kernel_impl<T, int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>( \
                weight.data<T>(), indices.data<int32_t>(), output.data<T>(), \
                num_indices, embedding_dim); \
        else \
            embedding_kernel_impl<T, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>( \
                weight.data<T>(), indices.data<int64_t>(), output.data<T>(), \
                num_indices, embedding_dim)

    switch (weight.dtype()) {
        case DType::Float32: LAUNCH_EMBEDDING(float); break;
        case DType::Float64: LAUNCH_EMBEDDING(double); break;
        case DType::Float16:
            if (idx_is_int32)
                embedding_kernel_impl<__half, int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __half*>(weight.data_ptr()),
                    indices.data<int32_t>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    num_indices, embedding_dim);
            else
                embedding_kernel_impl<__half, int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __half*>(weight.data_ptr()),
                    indices.data<int64_t>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    num_indices, embedding_dim);
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
                num_indices, embedding_dim)

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
            break;
        case DType::Int64:
            one_hot_kernel_impl<int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                indices.data<int64_t>(), output.data<float>(), batch_size, num_classes);
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
            input.data<T>(), d_flags, n)

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

    int blocks = get_num_blocks(indices_size);

    switch (input.dtype()) {
        case DType::Float32:
            take_kernel_impl<float><<<blocks, BLOCK_SIZE, 0, stream>>>(
                input.data<float>(), indices.data<int64_t>(), output.data<float>(),
                input_size, indices_size);
            break;
        case DType::Float64:
            take_kernel_impl<double><<<blocks, BLOCK_SIZE, 0, stream>>>(
                input.data<double>(), indices.data<int64_t>(), output.data<double>(),
                input_size, indices_size);
            break;
        case DType::Int32:
            take_kernel_impl<int32_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
                input.data<int32_t>(), indices.data<int64_t>(), output.data<int32_t>(),
                input_size, indices_size);
            break;
        case DType::Int64:
            take_kernel_impl<int64_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
                input.data<int64_t>(), indices.data<int64_t>(), output.data<int64_t>(),
                input_size, indices_size);
            break;
        default:
            throw std::runtime_error("take_kernel: unsupported dtype");
    }

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
            break;
        case DType::Float64:
            put_kernel_impl<double><<<blocks, BLOCK_SIZE, 0, stream>>>(
                output.data<double>(), indices.data<int64_t>(), source.data<double>(),
                num_indices, total_size, accumulate);
            break;
        case DType::Int32:
            put_kernel_impl<int32_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
                output.data<int32_t>(), indices.data<int64_t>(), source.data<int32_t>(),
                num_indices, total_size, accumulate);
            break;
        case DType::Int64:
            put_kernel_impl<int64_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
                output.data<int64_t>(), indices.data<int64_t>(), source.data<int64_t>(),
                num_indices, total_size, accumulate);
            break;
        default:
            throw std::runtime_error("put_kernel: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

} // namespace cuda
} // namespace tenzor
