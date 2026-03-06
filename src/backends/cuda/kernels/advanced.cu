/**
 * @file advanced.cu
 * @brief CUDA kernels for advanced tensor operations: topk, sort, cumsum, cumprod, unique
 *
 * Uses CUB for radix sort and prefix scan, custom kernels for topk and unique.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/caching_allocator.hpp"
#include "cuda_common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <cfloat>
#include <stdexcept>

namespace tenzor {
namespace cuda {

// Custom multiply functor for CUB InclusiveScan (cumprod)
struct MultOp {
    template<typename T>
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a * b; }
};

// ============================================================================
// TopK kernel using parallel block-wide selection
// ============================================================================

// Each block handles one (outer, inner) slice. All threads participate:
// 1. Each thread scans a stripe of dim_size elements, maintaining a local
//    thread-best candidate.
// 2. Block-level reduction finds the global best among all threads.
// 3. The winning thread writes the best to shared memory and marks its
//    element as "consumed".
// 4. Repeat k times to collect the top-k elements.
// 5. Sort the k results with a parallel odd-even sort.

template<typename T>
__global__ void topk_slice_kernel(
    const T* __restrict__ input, T* __restrict__ values, int64_t* __restrict__ indices,
    int64_t dim_size, int64_t k, int64_t inner_size, int64_t outer_stride,
    int64_t k_stride, bool largest)
{
    int64_t slice_idx = blockIdx.x;
    int64_t outer = slice_idx / inner_size;
    int64_t inner = slice_idx % inner_size;

    int64_t in_base = outer * outer_stride + inner;
    int64_t out_base = outer * k_stride + inner;

    // Shared memory layout: k values, k indices, blockDim.x candidate values,
    // blockDim.x candidate indices, blockDim.x candidate positions
    extern __shared__ char smem[];
    T* s_topk_vals = reinterpret_cast<T*>(smem);
    size_t vals_bytes = k * sizeof(T);
    size_t aligned_vals = (vals_bytes + 7) & ~size_t(7);
    int64_t* s_topk_idx = reinterpret_cast<int64_t*>(smem + aligned_vals);
    size_t idx_bytes = k * sizeof(int64_t);
    size_t aligned_idx = (idx_bytes + 7) & ~size_t(7);

    // Candidate arrays for block-wide reduction
    char* cand_base = smem + aligned_vals + aligned_idx;
    T* s_cand_vals = reinterpret_cast<T*>(cand_base);
    size_t cand_vals_bytes = blockDim.x * sizeof(T);
    size_t aligned_cand_vals = (cand_vals_bytes + 7) & ~size_t(7);
    int64_t* s_cand_pos = reinterpret_cast<int64_t*>(cand_base + aligned_cand_vals);

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Each thread finds its best among its stripe elements.
    // We'll mark consumed elements with a sentinel.
    // To avoid modifying input, each thread tracks consumed positions locally
    // by iterating k rounds.

    // Phase 1: Load all elements into shared/register consideration
    // For memory efficiency, we use an iterative approach:
    // each round, every thread finds its best unconsumed element
    for (int64_t round = 0; round < k; ++round) {
        // Each thread scans its stripe to find the best unconsumed element
        T best_val;
        int64_t best_pos = -1;
        bool has_candidate = false;

        for (int64_t i = tid; i < dim_size; i += nthreads) {
            T val = input[in_base + i * inner_size];

            // Check if this position was already selected in a previous round
            bool consumed = false;
            for (int64_t r = 0; r < round; ++r) {
                if (s_topk_idx[r] == i) {
                    consumed = true;
                    break;
                }
            }
            if (consumed) continue;

            if (!has_candidate ||
                (largest ? (val > best_val) : (val < best_val)) ||
                (val == best_val && i < best_pos)) {
                best_val = val;
                best_pos = i;
                has_candidate = true;
            }
        }

        // Store each thread's candidate in shared memory
        // Use extreme sentinel for threads without candidates
        s_cand_vals[tid] = best_val;  // value doesn't matter if best_pos < 0
        s_cand_pos[tid] = best_pos;
        __syncthreads();

        // Block-wide reduction to find the global best
        for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                bool right_wins;
                if (s_cand_pos[tid] < 0 && s_cand_pos[tid + stride] >= 0) {
                    right_wins = true;
                } else if (s_cand_pos[tid] >= 0 && s_cand_pos[tid + stride] < 0) {
                    right_wins = false;
                } else if (s_cand_pos[tid] < 0 && s_cand_pos[tid + stride] < 0) {
                    right_wins = false;
                } else {
                    right_wins = largest ?
                        (s_cand_vals[tid + stride] > s_cand_vals[tid] ||
                         (s_cand_vals[tid + stride] == s_cand_vals[tid] &&
                          s_cand_pos[tid + stride] < s_cand_pos[tid])) :
                        (s_cand_vals[tid + stride] < s_cand_vals[tid] ||
                         (s_cand_vals[tid + stride] == s_cand_vals[tid] &&
                          s_cand_pos[tid + stride] < s_cand_pos[tid]));
                }
                if (right_wins) {
                    s_cand_vals[tid] = s_cand_vals[tid + stride];
                    s_cand_pos[tid] = s_cand_pos[tid + stride];
                }
            }
            __syncthreads();
        }

        // Thread 0 writes the winner to topk arrays
        if (tid == 0) {
            s_topk_vals[round] = s_cand_vals[0];
            s_topk_idx[round] = s_cand_pos[0];
        }
        __syncthreads();
    }

    // Phase 2: Sort the k results using parallel odd-even transposition sort
    for (int64_t phase = 0; phase < k; ++phase) {
        int64_t i = 2 * tid + (phase & 1);
        if (i + 1 < k) {
            bool should_swap = largest ?
                (s_topk_vals[i] < s_topk_vals[i + 1]) :
                (s_topk_vals[i] > s_topk_vals[i + 1]);
            if (should_swap) {
                T tmp_v = s_topk_vals[i];
                s_topk_vals[i] = s_topk_vals[i + 1];
                s_topk_vals[i + 1] = tmp_v;
                int64_t tmp_i = s_topk_idx[i];
                s_topk_idx[i] = s_topk_idx[i + 1];
                s_topk_idx[i + 1] = tmp_i;
            }
        }
        __syncthreads();
    }

    // Write results
    for (int64_t i = tid; i < k; i += nthreads) {
        values[out_base + i * inner_size] = s_topk_vals[i];
        indices[out_base + i * inner_size] = s_topk_idx[i];
    }
}

auto topk_kernel(const Tensor& input, int64_t k, int64_t dim, bool largest,
                 bool sorted, cudaStream_t stream) -> std::pair<Tensor, Tensor>
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = input.ndim();
    const int64_t dim_size = shape[dim];
    const auto dtype = input.dtype();
    const auto device = input.device();

    // Compute output shape
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    output_shape[dim] = k;

    Tensor values(output_shape, dtype, device);
    Tensor indices(output_shape, DType::Int64, device);

    // Compute outer/inner sizes
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    int64_t num_slices = outer_size * inner_size;
    int64_t outer_stride = dim_size * inner_size;
    int64_t k_stride = k * inner_size;

    auto launch = [&]<typename T>() {
        int block_size = 256;
        // Shared memory: topk values + topk indices + candidate values + candidate positions
        size_t topk_vals_bytes = k * sizeof(T);
        size_t aligned_topk_vals = (topk_vals_bytes + 7) & ~size_t(7);
        size_t topk_idx_bytes = k * sizeof(int64_t);
        size_t aligned_topk_idx = (topk_idx_bytes + 7) & ~size_t(7);
        size_t cand_vals_bytes = block_size * sizeof(T);
        size_t aligned_cand_vals = (cand_vals_bytes + 7) & ~size_t(7);
        size_t cand_pos_bytes = block_size * sizeof(int64_t);
        size_t smem_size = aligned_topk_vals + aligned_topk_idx +
                           aligned_cand_vals + cand_pos_bytes;
        topk_slice_kernel<T><<<num_slices, block_size, smem_size, stream>>>(
            input_cont.data<T>(), values.data<T>(), indices.data<int64_t>(),
            dim_size, k, inner_size, outer_stride, k_stride, largest);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    };

    switch (dtype) {
        case DType::Float32: launch.template operator()<float>(); break;
        case DType::Float64: launch.template operator()<double>(); break;
        case DType::Int32:   launch.template operator()<int32_t>(); break;
        case DType::Int64:   launch.template operator()<int64_t>(); break;
        default: throw std::runtime_error("topk CUDA: unsupported dtype");
    }

    return {values, indices};
}

// ============================================================================
// Sort kernel using Thrust
// ============================================================================

template<typename T>
static void sort_1d_thrust(const T* input, T* values, int64_t* indices_out,
                           int64_t n, bool descending, cudaStream_t stream)
{
    auto policy = thrust::cuda::par.on(stream);

    // Copy input to values
    cudaMemcpyAsync(values, input, n * sizeof(T), cudaMemcpyDeviceToDevice, stream);

    // Initialize indices
    thrust::sequence(policy, thrust::device_pointer_cast(indices_out),
                     thrust::device_pointer_cast(indices_out + n), int64_t(0));

    // Sort by key (values are keys, indices are values)
    if (descending) {
        thrust::sort_by_key(policy,
            thrust::device_pointer_cast(values),
            thrust::device_pointer_cast(values + n),
            thrust::device_pointer_cast(indices_out),
            thrust::greater<T>());
    } else {
        thrust::sort_by_key(policy,
            thrust::device_pointer_cast(values),
            thrust::device_pointer_cast(values + n),
            thrust::device_pointer_cast(indices_out));
    }
}

template<typename T>
__global__ void extract_slice_kernel(const T* __restrict__ input, T* __restrict__ slice,
                                     int64_t dim_size, int64_t inner_size,
                                     int64_t outer, int64_t inner)
{
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dim_size;
         i += gridDim.x * blockDim.x) {
        slice[i] = input[outer * dim_size * inner_size + i * inner_size + inner];
    }
}

template<typename T>
__global__ void scatter_slice_kernel(const T* __restrict__ sorted_vals,
                                     const int64_t* __restrict__ sorted_idx,
                                     T* __restrict__ out_vals, int64_t* __restrict__ out_idx,
                                     int64_t dim_size, int64_t inner_size,
                                     int64_t outer, int64_t inner)
{
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dim_size;
         i += gridDim.x * blockDim.x) {
        int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
        out_vals[offset] = sorted_vals[i];
        out_idx[offset] = sorted_idx[i];
    }
}

auto sort_kernel(const Tensor& input, int64_t dim, bool descending,
                 cudaStream_t stream) -> std::pair<Tensor, Tensor>
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = input.ndim();
    const int64_t dim_size = shape[dim];
    const auto dtype = input.dtype();
    const auto device = input.device();

    Tensor values(std::vector<int64_t>(shape.begin(), shape.end()), dtype, device);
    Tensor indices(std::vector<int64_t>(shape.begin(), shape.end()), DType::Int64, device);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    auto launch = [&]<typename T>() {
        // Temp buffers for one slice
        backend::CachedMemoryGuard slice_guard(dim_size * sizeof(T));
        T* d_slice = static_cast<T*>(slice_guard.get());
        backend::CachedMemoryGuard idx_guard(dim_size * sizeof(int64_t));
        int64_t* d_idx = static_cast<int64_t*>(idx_guard.get());

        int block = 256;
        int grid = std::min(int((dim_size + block - 1) / block), 1024);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                // Extract slice
                extract_slice_kernel<T><<<grid, block, 0, stream>>>(
                    input_cont.data<T>(), d_slice, dim_size, inner_size, outer, inner);
                TENZOR_CUDA_POST_LAUNCH_CHECK();

                // Sort slice
                sort_1d_thrust<T>(d_slice, d_slice, d_idx, dim_size, descending, stream);

                // Scatter back
                scatter_slice_kernel<T><<<grid, block, 0, stream>>>(
                    d_slice, d_idx, values.data<T>(), indices.data<int64_t>(),
                    dim_size, inner_size, outer, inner);
                TENZOR_CUDA_POST_LAUNCH_CHECK();
            }
        }
    };

    switch (dtype) {
        case DType::Float32: launch.template operator()<float>(); break;
        case DType::Float64: launch.template operator()<double>(); break;
        case DType::Int32:   launch.template operator()<int32_t>(); break;
        case DType::Int64:   launch.template operator()<int64_t>(); break;
        default: throw std::runtime_error("sort CUDA: unsupported dtype");
    }

    return {values, indices};
}

// ============================================================================
// CumSum kernel using CUB InclusiveScan
// ============================================================================

template<typename T>
__global__ void extract_strided_slice(const T* __restrict__ input, T* __restrict__ output,
                                      int64_t dim_size, int64_t inner_size,
                                      int64_t outer, int64_t inner)
{
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dim_size;
         i += gridDim.x * blockDim.x) {
        output[i] = input[outer * dim_size * inner_size + i * inner_size + inner];
    }
}

template<typename T>
__global__ void scatter_strided_slice(const T* __restrict__ input, T* __restrict__ output,
                                      int64_t dim_size, int64_t inner_size,
                                      int64_t outer, int64_t inner)
{
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dim_size;
         i += gridDim.x * blockDim.x) {
        output[outer * dim_size * inner_size + i * inner_size + inner] = input[i];
    }
}

template<typename T>
static void cumsum_slice_cub(const T* d_in, T* d_out, int64_t n, cudaStream_t stream)
{
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in, d_out,
                                  static_cast<int>(n), stream);
    backend::CachedMemoryGuard temp_guard(temp_bytes);
    d_temp = temp_guard.get();
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in, d_out,
                                  static_cast<int>(n), stream);
}

auto cumsum_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = input.ndim();
    const int64_t dim_size = shape[dim];
    const auto dtype = input.dtype();
    const auto device = input.device();

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), dtype, device);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    // Fast path: contiguous along dim (inner_size == 1 and dim is last)
    // In that case we can run CUB directly on contiguous slices
    auto launch = [&]<typename T>() {
        if (inner_size == 1) {
            // Contiguous slices — run CUB directly
            for (int64_t outer = 0; outer < outer_size; ++outer) {
                const T* d_in = input_cont.data<T>() + outer * dim_size;
                T* d_out = output.data<T>() + outer * dim_size;
                cumsum_slice_cub<T>(d_in, d_out, dim_size, stream);
            }
        } else {
            // Non-contiguous: extract, scan, scatter
            backend::CachedMemoryGuard in_guard(dim_size * sizeof(T));
            backend::CachedMemoryGuard out_guard(dim_size * sizeof(T));
            T* d_slice_in = static_cast<T*>(in_guard.get());
            T* d_slice_out = static_cast<T*>(out_guard.get());

            int block = 256;
            int grid = std::min(int((dim_size + block - 1) / block), 1024);

            for (int64_t outer = 0; outer < outer_size; ++outer) {
                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    extract_strided_slice<T><<<grid, block, 0, stream>>>(
                        input_cont.data<T>(), d_slice_in, dim_size, inner_size, outer, inner);
                    TENZOR_CUDA_POST_LAUNCH_CHECK();
                    cumsum_slice_cub<T>(d_slice_in, d_slice_out, dim_size, stream);
                    scatter_strided_slice<T><<<grid, block, 0, stream>>>(
                        d_slice_out, output.data<T>(), dim_size, inner_size, outer, inner);
                    TENZOR_CUDA_POST_LAUNCH_CHECK();
                }
            }
        }
    };

    switch (dtype) {
        case DType::Float32: launch.template operator()<float>(); break;
        case DType::Float64: launch.template operator()<double>(); break;
        case DType::Int32:   launch.template operator()<int32_t>(); break;
        case DType::Int64:   launch.template operator()<int64_t>(); break;
        default: throw std::runtime_error("cumsum CUDA: unsupported dtype");
    }

    return output;
}

// ============================================================================
// CumProd kernel using CUB InclusiveScan with multiply operator
// ============================================================================

template<typename T>
static void cumprod_slice_cub(const T* d_in, T* d_out, int64_t n, cudaStream_t stream)
{
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveScan(d_temp, temp_bytes, d_in, d_out,
                                   MultOp(), static_cast<int>(n), stream);
    backend::CachedMemoryGuard temp_guard(temp_bytes);
    d_temp = temp_guard.get();
    cub::DeviceScan::InclusiveScan(d_temp, temp_bytes, d_in, d_out,
                                   MultOp(), static_cast<int>(n), stream);
}

auto cumprod_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = input.ndim();
    const int64_t dim_size = shape[dim];
    const auto dtype = input.dtype();
    const auto device = input.device();

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), dtype, device);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    auto launch = [&]<typename T>() {
        if (inner_size == 1) {
            for (int64_t outer = 0; outer < outer_size; ++outer) {
                const T* d_in = input_cont.data<T>() + outer * dim_size;
                T* d_out = output.data<T>() + outer * dim_size;
                cumprod_slice_cub<T>(d_in, d_out, dim_size, stream);
            }
        } else {
            backend::CachedMemoryGuard in_guard(dim_size * sizeof(T));
            backend::CachedMemoryGuard out_guard(dim_size * sizeof(T));
            T* d_slice_in = static_cast<T*>(in_guard.get());
            T* d_slice_out = static_cast<T*>(out_guard.get());

            int block = 256;
            int grid = std::min(int((dim_size + block - 1) / block), 1024);

            for (int64_t outer = 0; outer < outer_size; ++outer) {
                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    extract_strided_slice<T><<<grid, block, 0, stream>>>(
                        input_cont.data<T>(), d_slice_in, dim_size, inner_size, outer, inner);
                    TENZOR_CUDA_POST_LAUNCH_CHECK();
                    cumprod_slice_cub<T>(d_slice_in, d_slice_out, dim_size, stream);
                    scatter_strided_slice<T><<<grid, block, 0, stream>>>(
                        d_slice_out, output.data<T>(), dim_size, inner_size, outer, inner);
                    TENZOR_CUDA_POST_LAUNCH_CHECK();
                }
            }
        }
    };

    switch (dtype) {
        case DType::Float32: launch.template operator()<float>(); break;
        case DType::Float64: launch.template operator()<double>(); break;
        case DType::Int32:   launch.template operator()<int32_t>(); break;
        case DType::Int64:   launch.template operator()<int64_t>(); break;
        default: throw std::runtime_error("cumprod CUDA: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Unique kernel using Thrust
// ============================================================================

// Kernel to fill inverse indices: each thread handles one unique group
__global__ void fill_inverse_kernel(const int64_t* __restrict__ orig_idx,
                                    const int64_t* __restrict__ offsets,
                                    int64_t* __restrict__ inverse,
                                    int64_t num_unique, int64_t numel) {
    int64_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= num_unique) return;
    int64_t start = offsets[g];
    int64_t end = (g + 1 < num_unique) ? offsets[g + 1] : numel;
    for (int64_t i = start; i < end; ++i) {
        inverse[orig_idx[i]] = g;
    }
}

template<typename T>
static auto unique_thrust(const Tensor& input, bool sorted_output,
                          bool return_inverse, bool return_counts,
                          cudaStream_t stream)
    -> std::tuple<Tensor, Tensor, Tensor>
{
    const int64_t numel = input.numel();
    const auto device = input.device();
    auto policy = thrust::cuda::par.on(stream);

    // Copy and flatten input
    backend::CachedMemoryGuard sorted_guard(numel * sizeof(T));
    T* d_sorted = static_cast<T*>(sorted_guard.get());
    cudaMemcpyAsync(d_sorted, input.data<T>(), numel * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);

    // Create index mapping for inverse
    backend::CachedMemoryGuard orig_idx_guard(numel * sizeof(int64_t));
    int64_t* d_orig_idx = static_cast<int64_t*>(orig_idx_guard.get());
    thrust::sequence(policy, thrust::device_pointer_cast(d_orig_idx),
                     thrust::device_pointer_cast(d_orig_idx + numel), int64_t(0));

    // Sort input (always sort to find unique, even if sorted_output is false)
    thrust::sort_by_key(policy,
        thrust::device_pointer_cast(d_sorted),
        thrust::device_pointer_cast(d_sorted + numel),
        thrust::device_pointer_cast(d_orig_idx));

    // Find unique elements
    backend::CachedMemoryGuard unique_guard(numel * sizeof(T));
    T* d_unique = static_cast<T*>(unique_guard.get());
    backend::CachedMemoryGuard counts_guard(numel * sizeof(int64_t));
    int64_t* d_counts = static_cast<int64_t*>(counts_guard.get());

    // Use run-length encoding to get unique values and counts
    backend::CachedMemoryGuard num_runs_guard(sizeof(int64_t));
    int64_t* d_num_runs = static_cast<int64_t*>(num_runs_guard.get());

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(
        d_temp, temp_bytes, d_sorted, d_unique, d_counts, d_num_runs,
        static_cast<int>(numel), stream);
    backend::CachedMemoryGuard temp_guard(temp_bytes);
    d_temp = temp_guard.get();
    cub::DeviceRunLengthEncode::Encode(
        d_temp, temp_bytes, d_sorted, d_unique, d_counts, d_num_runs,
        static_cast<int>(numel), stream);

    // Get num_unique on host
    int64_t num_unique = 0;
    cudaMemcpyAsync(&num_unique, d_num_runs, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Create unique values tensor
    Tensor unique_vals({num_unique}, input.dtype(), device);
    cudaMemcpyAsync(unique_vals.data<T>(), d_unique, num_unique * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);

    // Create counts tensor if requested
    Tensor counts_tensor;
    if (return_counts) {
        counts_tensor = Tensor({num_unique}, DType::Int64, device);
        cudaMemcpyAsync(counts_tensor.data<int64_t>(), d_counts, num_unique * sizeof(int64_t),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // Create inverse indices if requested
    Tensor inverse_tensor;
    if (return_inverse) {
        inverse_tensor = Tensor({numel}, DType::Int64, device);
        // For each original element, find its index in the unique array
        // Since sorted values are run-length encoded, compute exclusive prefix sum of counts
        // to get the start index of each unique run, then scatter
        backend::CachedMemoryGuard offsets_guard((num_unique + 1) * sizeof(int64_t));
        int64_t* d_offsets = static_cast<int64_t*>(offsets_guard.get());

        void* d_scan_temp = nullptr;
        size_t scan_temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes, d_counts, d_offsets,
                                      static_cast<int>(num_unique), stream);
        backend::CachedMemoryGuard scan_temp_guard(scan_temp_bytes);
        d_scan_temp = scan_temp_guard.get();
        cub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes, d_counts, d_offsets,
                                      static_cast<int>(num_unique), stream);

        // For each position in the sorted array, find which unique group it belongs to
        // using binary search (or simpler: iterate groups and fill)
        // Simple approach: use the sorted order and orig_idx to map back
        // d_orig_idx[i] = original index of sorted position i
        // sorted position i belongs to unique group g where offsets[g] <= i < offsets[g+1]
        // So inverse[d_orig_idx[i]] = g

        int block = 256;
        int grid = (num_unique + block - 1) / block;
        fill_inverse_kernel<<<grid, block, 0, stream>>>(
            d_orig_idx, d_offsets, inverse_tensor.data<int64_t>(),
            num_unique, numel);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    }

    cudaStreamSynchronize(stream);
    return {unique_vals, inverse_tensor, counts_tensor};
}

auto unique_kernel(const Tensor& input, bool sorted_output, bool return_inverse,
                   bool return_counts, cudaStream_t stream)
    -> std::tuple<Tensor, Tensor, Tensor>
{
    Tensor flat = input.flatten().contiguous();

    switch (input.dtype()) {
        case DType::Float32:
            return unique_thrust<float>(flat, sorted_output, return_inverse, return_counts, stream);
        case DType::Float64:
            return unique_thrust<double>(flat, sorted_output, return_inverse, return_counts, stream);
        case DType::Int32:
            return unique_thrust<int32_t>(flat, sorted_output, return_inverse, return_counts, stream);
        case DType::Int64:
            return unique_thrust<int64_t>(flat, sorted_output, return_inverse, return_counts, stream);
        default:
            throw std::runtime_error("unique CUDA: unsupported dtype");
    }
}

} // namespace cuda
} // namespace tenzor
