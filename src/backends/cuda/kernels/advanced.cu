/**
 * @file advanced.cu
 * @brief CUDA kernels for advanced tensor operations: topk, sort, cumsum, cumprod, unique
 *
 * Uses CUB for radix sort and prefix scan, custom kernels for topk and unique.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/caching_allocator.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
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
#include <chrono>
#include <stdexcept>
#include <optional>

// Forward declarations for CPU fallback functions (avoid C++23 headers in CUDA)
namespace tenzor {
    auto zeros(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor;
    auto index(const Tensor& input,
               const std::vector<std::optional<Tensor>>& indices) -> Tensor;
    auto index_put(Tensor& input,
                   const std::vector<std::optional<Tensor>>& indices,
                   const Tensor& values) -> void;
namespace fft {
    auto stft(const Tensor& input, int64_t n_fft, int64_t hop_length,
              int64_t win_length, const Tensor& window, bool center,
              bool normalized, bool onesided) -> Tensor;
    auto istft(const Tensor& input, int64_t n_fft, int64_t hop_length,
               int64_t win_length, const Tensor& window, bool center,
               bool normalized, bool onesided,
               std::optional<int64_t> length) -> Tensor;
} // namespace fft
} // namespace tenzor

namespace tenzor {
namespace cuda {

// Custom multiply functor for CUB InclusiveScan (cumprod)
struct MultOp {
    template<typename T>
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a * b; }
};

// Safe comparison helpers for half/bfloat16 types (C++ operators may not exist on all archs)
template<typename T>
__device__ __forceinline__ bool cuda_gt(const T& a, const T& b) { return a > b; }
template<typename T>
__device__ __forceinline__ bool cuda_lt(const T& a, const T& b) { return a < b; }
template<typename T>
__device__ __forceinline__ bool cuda_eq(const T& a, const T& b) { return a == b; }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
template<> __device__ __forceinline__ bool cuda_gt(const __half& a, const __half& b) { return __hgt(a, b); }
template<> __device__ __forceinline__ bool cuda_lt(const __half& a, const __half& b) { return __hlt(a, b); }
template<> __device__ __forceinline__ bool cuda_eq(const __half& a, const __half& b) { return __heq(a, b); }
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template<> __device__ __forceinline__ bool cuda_gt(const __nv_bfloat16& a, const __nv_bfloat16& b) { return __hgt(a, b); }
template<> __device__ __forceinline__ bool cuda_lt(const __nv_bfloat16& a, const __nv_bfloat16& b) { return __hlt(a, b); }
template<> __device__ __forceinline__ bool cuda_eq(const __nv_bfloat16& a, const __nv_bfloat16& b) { return __heq(a, b); }
#endif

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
                (largest ? cuda_gt(val, best_val) : cuda_lt(val, best_val)) ||
                (cuda_eq(val, best_val) && i < best_pos)) {
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
                        (cuda_gt(s_cand_vals[tid + stride], s_cand_vals[tid]) ||
                         (cuda_eq(s_cand_vals[tid + stride], s_cand_vals[tid]) &&
                          s_cand_pos[tid + stride] < s_cand_pos[tid])) :
                        (cuda_lt(s_cand_vals[tid + stride], s_cand_vals[tid]) ||
                         (cuda_eq(s_cand_vals[tid + stride], s_cand_vals[tid]) &&
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
                cuda_lt(s_topk_vals[i], s_topk_vals[i + 1]) :
                cuda_gt(s_topk_vals[i], s_topk_vals[i + 1]);
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
    if (dim < 0) dim += ndim;
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
        // Use data_ptr() + reinterpret_cast for CUDA-native types (__half, __nv_bfloat16)
        // that don't have Tensor::data<T>() instantiations in the core library
        auto* input_ptr = reinterpret_cast<const T*>(input_cont.data_ptr());
        auto* values_ptr = reinterpret_cast<T*>(values.data_ptr());
        auto* indices_ptr = reinterpret_cast<int64_t*>(indices.data_ptr());

        topk_slice_kernel<T><<<num_slices, block_size, smem_size, stream>>>(
            input_ptr, values_ptr, indices_ptr,
            dim_size, k, inner_size, outer_stride, k_stride, largest);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    };

    switch (dtype) {
        case DType::Float32:  launch.template operator()<float>(); break;
        case DType::Float64:  launch.template operator()<double>(); break;
        case DType::Float16:  launch.template operator()<__half>(); break;
        case DType::BFloat16: launch.template operator()<__nv_bfloat16>(); break;
        case DType::Int32:    launch.template operator()<int32_t>(); break;
        case DType::Int64:    launch.template operator()<int64_t>(); break;
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
    TENZOR_CUDA_CHECK(cudaMemcpyAsync(values, input, n * sizeof(T), cudaMemcpyDeviceToDevice, stream));

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

// Conversion kernels for half-type sort/cumsum/cumprod upcast
template<typename HalfT>
__global__ void half_to_float_kernel(const HalfT* __restrict__ in, float* __restrict__ out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = static_cast<float>(in[idx]);
}

template<typename HalfT>
__global__ void float_to_half_kernel(const float* __restrict__ in, HalfT* __restrict__ out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = static_cast<HalfT>(in[idx]);
}

// Gather original half values by sorted float indices
template<typename HalfT>
__global__ void gather_by_indices_kernel(const HalfT* __restrict__ src, const int64_t* __restrict__ indices,
                                          HalfT* __restrict__ dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[indices[idx]];
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

    // Half-type sort: upcast to Float32, sort, gather original values by indices
    auto launch_half = [&]<typename HalfT>() {
        int64_t numel = input_cont.numel();
        int cvt_block = 256;
        int cvt_grid = (numel + cvt_block - 1) / cvt_block;

        // Allocate Float32 buffer
        Tensor f32_input(std::vector<int64_t>(shape.begin(), shape.end()), DType::Float32, device);
        half_to_float_kernel<HalfT><<<cvt_grid, cvt_block, 0, stream>>>(
            reinterpret_cast<const HalfT*>(input_cont.data_ptr()),
            f32_input.data<float>(), numel);
        TENZOR_CUDA_POST_LAUNCH_CHECK();

        Tensor f32_values(std::vector<int64_t>(shape.begin(), shape.end()), DType::Float32, device);

        // Sort Float32 copy
        backend::CachedMemoryGuard slice_guard(dim_size * sizeof(float));
        float* d_slice = static_cast<float*>(slice_guard.get());
        backend::CachedMemoryGuard idx_guard(dim_size * sizeof(int64_t));
        int64_t* d_idx = static_cast<int64_t*>(idx_guard.get());

        int block = 256;
        int grid = std::min(int((dim_size + block - 1) / block), 1024);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                extract_slice_kernel<float><<<grid, block, 0, stream>>>(
                    f32_input.data<float>(), d_slice, dim_size, inner_size, outer, inner);
                TENZOR_CUDA_POST_LAUNCH_CHECK();
                sort_1d_thrust<float>(d_slice, d_slice, d_idx, dim_size, descending, stream);
                scatter_slice_kernel<float><<<grid, block, 0, stream>>>(
                    d_slice, d_idx, f32_values.data<float>(), indices.data<int64_t>(),
                    dim_size, inner_size, outer, inner);
                TENZOR_CUDA_POST_LAUNCH_CHECK();
            }
        }

        // Convert sorted Float32 values back to half
        float_to_half_kernel<HalfT><<<cvt_grid, cvt_block, 0, stream>>>(
            f32_values.data<float>(), reinterpret_cast<HalfT*>(values.data_ptr()), numel);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    };

    switch (dtype) {
        case DType::Float32: launch.template operator()<float>(); break;
        case DType::Float64: launch.template operator()<double>(); break;
        case DType::Int32:   launch.template operator()<int32_t>(); break;
        case DType::Int64:   launch.template operator()<int64_t>(); break;
        case DType::Float16: launch_half.template operator()<__half>(); break;
        case DType::BFloat16: launch_half.template operator()<__nv_bfloat16>(); break;
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

    // Half-type cumsum: upcast to Float32, cumsum, convert back
    auto launch_half_cumsum = [&]<typename HalfT>() {
        int64_t numel = input_cont.numel();
        int cvt_block = 256;
        int cvt_grid = (numel + cvt_block - 1) / cvt_block;

        Tensor f32_input(std::vector<int64_t>(shape.begin(), shape.end()), DType::Float32, device);
        half_to_float_kernel<HalfT><<<cvt_grid, cvt_block, 0, stream>>>(
            reinterpret_cast<const HalfT*>(input_cont.data_ptr()),
            f32_input.data<float>(), numel);
        TENZOR_CUDA_POST_LAUNCH_CHECK();

        Tensor f32_output(std::vector<int64_t>(shape.begin(), shape.end()), DType::Float32, device);

        if (inner_size == 1) {
            for (int64_t outer = 0; outer < outer_size; ++outer) {
                const float* d_in = f32_input.data<float>() + outer * dim_size;
                float* d_out = f32_output.data<float>() + outer * dim_size;
                cumsum_slice_cub<float>(d_in, d_out, dim_size, stream);
            }
        } else {
            backend::CachedMemoryGuard in_guard(dim_size * sizeof(float));
            backend::CachedMemoryGuard out_guard(dim_size * sizeof(float));
            float* d_slice_in = static_cast<float*>(in_guard.get());
            float* d_slice_out = static_cast<float*>(out_guard.get());
            int block = 256;
            int grid = std::min(int((dim_size + block - 1) / block), 1024);
            for (int64_t outer = 0; outer < outer_size; ++outer) {
                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    extract_strided_slice<float><<<grid, block, 0, stream>>>(
                        f32_input.data<float>(), d_slice_in, dim_size, inner_size, outer, inner);
                    TENZOR_CUDA_POST_LAUNCH_CHECK();
                    cumsum_slice_cub<float>(d_slice_in, d_slice_out, dim_size, stream);
                    scatter_strided_slice<float><<<grid, block, 0, stream>>>(
                        d_slice_out, f32_output.data<float>(), dim_size, inner_size, outer, inner);
                    TENZOR_CUDA_POST_LAUNCH_CHECK();
                }
            }
        }

        float_to_half_kernel<HalfT><<<cvt_grid, cvt_block, 0, stream>>>(
            f32_output.data<float>(), reinterpret_cast<HalfT*>(output.data_ptr()), numel);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    };

    switch (dtype) {
        case DType::Float32: launch.template operator()<float>(); break;
        case DType::Float64: launch.template operator()<double>(); break;
        case DType::Int32:   launch.template operator()<int32_t>(); break;
        case DType::Int64:   launch.template operator()<int64_t>(); break;
        case DType::Float16: launch_half_cumsum.template operator()<__half>(); break;
        case DType::BFloat16: launch_half_cumsum.template operator()<__nv_bfloat16>(); break;
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

    // Half-type cumprod: upcast to Float32, cumprod, convert back
    auto launch_half_cumprod = [&]<typename HalfT>() {
        int64_t numel = input_cont.numel();
        int cvt_block = 256;
        int cvt_grid = (numel + cvt_block - 1) / cvt_block;

        Tensor f32_input(std::vector<int64_t>(shape.begin(), shape.end()), DType::Float32, device);
        half_to_float_kernel<HalfT><<<cvt_grid, cvt_block, 0, stream>>>(
            reinterpret_cast<const HalfT*>(input_cont.data_ptr()),
            f32_input.data<float>(), numel);
        TENZOR_CUDA_POST_LAUNCH_CHECK();

        Tensor f32_output(std::vector<int64_t>(shape.begin(), shape.end()), DType::Float32, device);

        if (inner_size == 1) {
            for (int64_t outer = 0; outer < outer_size; ++outer) {
                const float* d_in = f32_input.data<float>() + outer * dim_size;
                float* d_out = f32_output.data<float>() + outer * dim_size;
                cumprod_slice_cub<float>(d_in, d_out, dim_size, stream);
            }
        } else {
            backend::CachedMemoryGuard in_guard(dim_size * sizeof(float));
            backend::CachedMemoryGuard out_guard(dim_size * sizeof(float));
            float* d_slice_in = static_cast<float*>(in_guard.get());
            float* d_slice_out = static_cast<float*>(out_guard.get());
            int block = 256;
            int grid = std::min(int((dim_size + block - 1) / block), 1024);
            for (int64_t outer = 0; outer < outer_size; ++outer) {
                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    extract_strided_slice<float><<<grid, block, 0, stream>>>(
                        f32_input.data<float>(), d_slice_in, dim_size, inner_size, outer, inner);
                    TENZOR_CUDA_POST_LAUNCH_CHECK();
                    cumprod_slice_cub<float>(d_slice_in, d_slice_out, dim_size, stream);
                    scatter_strided_slice<float><<<grid, block, 0, stream>>>(
                        d_slice_out, f32_output.data<float>(), dim_size, inner_size, outer, inner);
                    TENZOR_CUDA_POST_LAUNCH_CHECK();
                }
            }
        }

        float_to_half_kernel<HalfT><<<cvt_grid, cvt_block, 0, stream>>>(
            f32_output.data<float>(), reinterpret_cast<HalfT*>(output.data_ptr()), numel);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    };

    switch (dtype) {
        case DType::Float32: launch.template operator()<float>(); break;
        case DType::Float64: launch.template operator()<double>(); break;
        case DType::Int32:   launch.template operator()<int32_t>(); break;
        case DType::Int64:   launch.template operator()<int64_t>(); break;
        case DType::Float16: launch_half_cumprod.template operator()<__half>(); break;
        case DType::BFloat16: launch_half_cumprod.template operator()<__nv_bfloat16>(); break;
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
    TENZOR_CUDA_CHECK(cudaMemcpyAsync(d_sorted, input.data<T>(), numel * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream));

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
    TENZOR_CUDA_CHECK(cudaMemcpyAsync(&num_unique, d_num_runs, sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Create unique values tensor
    Tensor unique_vals({num_unique}, input.dtype(), device);
    TENZOR_CUDA_CHECK(cudaMemcpyAsync(unique_vals.data<T>(), d_unique, num_unique * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream));

    // Create counts tensor if requested
    Tensor counts_tensor;
    if (return_counts) {
        counts_tensor = Tensor({num_unique}, DType::Int64, device);
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(counts_tensor.data<int64_t>(), d_counts, num_unique * sizeof(int64_t),
                        cudaMemcpyDeviceToDevice, stream));
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
        case DType::Bool: {
            // Bool -> Int32 for sort/unique, then cast back
            Tensor flat_i32 = flat.to(DType::Int32);
            auto [uniq, inv, cnt] = unique_thrust<int32_t>(flat_i32, sorted_output, return_inverse, return_counts, stream);
            Tensor uniq_bool = uniq.to(DType::Bool);
            return {uniq_bool, inv, cnt};
        }
        default:
            throw std::runtime_error("unique CUDA: unsupported dtype");
    }
}

// ============================================================================
// Median kernel — sort each slice, pick middle element
// ============================================================================

template<typename T>
__global__ void extract_median_kernel(const T* __restrict__ sorted_vals,
                                      const int64_t* __restrict__ sorted_idx,
                                      T* __restrict__ out_vals,
                                      int64_t* __restrict__ out_idx,
                                      int64_t dim_size, int64_t inner_size,
                                      int64_t mid, int64_t outer, int64_t inner)
{
    // Single-thread kernel — only element 0 does the work
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int64_t out_offset = outer * inner_size + inner;
        out_vals[out_offset] = sorted_vals[mid];
        out_idx[out_offset] = sorted_idx[mid];
    }
}

auto median_kernel(const Tensor& input, int64_t dim, bool keepdim,
                   cudaStream_t stream) -> std::vector<Tensor>
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = static_cast<int64_t>(shape.size());
    const auto dtype = input.dtype();
    const auto device = input.device();

    // Normalize dim
    int64_t normalized_dim = dim;
    if (dim == INT64_MIN) {
        // Full reduction — flatten and reduce along dim 0
        Tensor flat = input_cont.reshape({input.numel()});
        auto result = median_kernel(flat, 0, false, stream);
        if (keepdim) {
            std::vector<int64_t> kshape(ndim, 1);
            result[0] = result[0].reshape(kshape);
            result[1] = result[1].reshape(kshape);
        }
        return result;
    }
    if (normalized_dim < 0) normalized_dim += ndim;

    const int64_t dim_size = shape[normalized_dim];
    const int64_t mid = (dim_size - 1) / 2;  // lower median for even sizes

    // Compute output shape
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    if (keepdim) {
        output_shape[normalized_dim] = 1;
    } else {
        output_shape.erase(output_shape.begin() + normalized_dim);
    }

    Tensor values(output_shape, dtype, device);
    Tensor indices(output_shape, DType::Int64, device);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < normalized_dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = normalized_dim + 1; i < ndim; ++i) inner_size *= shape[i];

    auto launch = [&]<typename U>() {
        backend::CachedMemoryGuard slice_guard(dim_size * sizeof(U));
        U* d_slice = static_cast<U*>(slice_guard.get());
        backend::CachedMemoryGuard idx_guard(dim_size * sizeof(int64_t));
        int64_t* d_idx = static_cast<int64_t*>(idx_guard.get());

        int block = 256;
        int grid = std::min(int((dim_size + block - 1) / block), 1024);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                // Extract slice along dim
                extract_slice_kernel<U><<<grid, block, 0, stream>>>(
                    input_cont.data<U>(), d_slice, dim_size, inner_size, outer, inner);
                TENZOR_CUDA_POST_LAUNCH_CHECK();

                // Sort the slice
                sort_1d_thrust<U>(d_slice, d_slice, d_idx, dim_size, false, stream);

                // Extract the median element
                extract_median_kernel<U><<<1, 1, 0, stream>>>(
                    d_slice, d_idx,
                    values.data<U>(), indices.data<int64_t>(),
                    dim_size, inner_size, mid, outer, inner);
                TENZOR_CUDA_POST_LAUNCH_CHECK();
            }
        }
    };

    // Half types: upcast to float, sort, then gather original values
    auto launch_half = [&]<typename HalfT>() {
        int64_t slice_numel = dim_size;
        int cvt_block = 256;
        int cvt_grid = (slice_numel + cvt_block - 1) / cvt_block;

        backend::CachedMemoryGuard half_slice_guard(dim_size * sizeof(HalfT));
        HalfT* d_half_slice = static_cast<HalfT*>(half_slice_guard.get());
        backend::CachedMemoryGuard f32_slice_guard(dim_size * sizeof(float));
        float* d_f32_slice = static_cast<float*>(f32_slice_guard.get());
        backend::CachedMemoryGuard idx_guard(dim_size * sizeof(int64_t));
        int64_t* d_idx = static_cast<int64_t*>(idx_guard.get());

        int block = 256;
        int grid = std::min(int((dim_size + block - 1) / block), 1024);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                // Extract half slice
                extract_slice_kernel<HalfT><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const HalfT*>(input_cont.data_ptr()),
                    d_half_slice, dim_size, inner_size, outer, inner);
                TENZOR_CUDA_POST_LAUNCH_CHECK();

                // Convert to float
                half_to_float_kernel<HalfT><<<cvt_grid, cvt_block, 0, stream>>>(
                    d_half_slice, d_f32_slice, dim_size);
                TENZOR_CUDA_POST_LAUNCH_CHECK();

                // Sort float copy
                sort_1d_thrust<float>(d_f32_slice, d_f32_slice, d_idx, dim_size, false, stream);

                // Extract median index from sorted indices
                // Copy the median index back to host to gather the original half value
                int64_t out_offset = outer * inner_size + inner;
                indices.data<int64_t>()[out_offset] = 0;  // placeholder

                // Use a device kernel to write the result
                // d_f32_slice[mid] has the sorted float value, d_idx[mid] has original index
                // We need to write the original half value, so gather from half slice
                // But half slice was extracted before sort — d_idx[mid] is the original position
                // in the slice, so we need the original half values
                // Actually d_half_slice was not sorted, but we overwrote d_f32_slice.
                // Re-extract the half slice to get unsorted original values
                extract_slice_kernel<HalfT><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const HalfT*>(input_cont.data_ptr()),
                    d_half_slice, dim_size, inner_size, outer, inner);
                TENZOR_CUDA_POST_LAUNCH_CHECK();

                // Now use a simple kernel to write value and index
                // We need d_idx[mid] and d_half_slice[d_idx[mid]]
                // Use extract_median_kernel with the float sorted indices
                extract_median_kernel<int64_t><<<1, 1, 0, stream>>>(
                    d_idx, d_idx,  // dummy — we just need d_idx[mid]
                    indices.data<int64_t>(), indices.data<int64_t>(),
                    dim_size, inner_size, mid, outer, inner);
                TENZOR_CUDA_POST_LAUNCH_CHECK();

                // For the value, we need to gather from the original half data
                // Use a gather kernel: out[offset] = half_slice[sorted_idx[mid]]
                // Simpler: just write f32 median and convert back
                float_to_half_kernel<HalfT><<<1, 1, 0, stream>>>(
                    d_f32_slice + mid,
                    reinterpret_cast<HalfT*>(values.data_ptr()) + out_offset,
                    1);
                TENZOR_CUDA_POST_LAUNCH_CHECK();
            }
        }
    };

    switch (dtype) {
        case DType::Float32:  launch.template operator()<float>(); break;
        case DType::Float64:  launch.template operator()<double>(); break;
        case DType::Int32:    launch.template operator()<int32_t>(); break;
        case DType::Int64:    launch.template operator()<int64_t>(); break;
        case DType::Float16:  launch_half.template operator()<__half>(); break;
        case DType::BFloat16: launch_half.template operator()<__nv_bfloat16>(); break;
        default: throw std::runtime_error("median CUDA: unsupported dtype");
    }

    return {values, indices};
}

// ============================================================================
// Mode kernel — sort each slice, find longest run of consecutive equal values
// ============================================================================

template<typename T>
__global__ void find_mode_kernel(const T* __restrict__ sorted_vals,
                                 const int64_t* __restrict__ sorted_idx,
                                 T* __restrict__ out_vals,
                                 int64_t* __restrict__ out_idx,
                                 int64_t dim_size, int64_t inner_size,
                                 int64_t outer, int64_t inner)
{
    // Single-thread kernel for simplicity — one slice at a time
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        T best_val = sorted_vals[0];
        int64_t best_idx = sorted_idx[0];
        int64_t best_count = 1;
        int64_t cur_count = 1;

        for (int64_t i = 1; i < dim_size; ++i) {
            if (sorted_vals[i] == sorted_vals[i - 1]) {
                cur_count++;
            } else {
                cur_count = 1;
            }
            if (cur_count > best_count) {
                best_count = cur_count;
                best_val = sorted_vals[i];
                best_idx = sorted_idx[i];
            }
        }

        int64_t out_offset = outer * inner_size + inner;
        out_vals[out_offset] = best_val;
        out_idx[out_offset] = best_idx;
    }
}

auto mode_kernel(const Tensor& input, int64_t dim, bool keepdim,
                 cudaStream_t stream) -> std::vector<Tensor>
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = static_cast<int64_t>(shape.size());
    const auto dtype = input.dtype();
    const auto device = input.device();

    // Normalize dim
    int64_t normalized_dim = dim;
    if (dim == INT64_MIN) {
        Tensor flat = input_cont.reshape({input.numel()});
        auto result = mode_kernel(flat, 0, false, stream);
        if (keepdim) {
            std::vector<int64_t> kshape(ndim, 1);
            result[0] = result[0].reshape(kshape);
            result[1] = result[1].reshape(kshape);
        }
        return result;
    }
    if (normalized_dim < 0) normalized_dim += ndim;

    const int64_t dim_size = shape[normalized_dim];

    // Compute output shape
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    if (keepdim) {
        output_shape[normalized_dim] = 1;
    } else {
        output_shape.erase(output_shape.begin() + normalized_dim);
    }

    Tensor values(output_shape, dtype, device);
    Tensor indices(output_shape, DType::Int64, device);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < normalized_dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = normalized_dim + 1; i < ndim; ++i) inner_size *= shape[i];

    auto launch = [&]<typename U>() {
        backend::CachedMemoryGuard slice_guard(dim_size * sizeof(U));
        U* d_slice = static_cast<U*>(slice_guard.get());
        backend::CachedMemoryGuard idx_guard(dim_size * sizeof(int64_t));
        int64_t* d_idx = static_cast<int64_t*>(idx_guard.get());

        int block = 256;
        int grid = std::min(int((dim_size + block - 1) / block), 1024);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                extract_slice_kernel<U><<<grid, block, 0, stream>>>(
                    input_cont.data<U>(), d_slice, dim_size, inner_size, outer, inner);
                TENZOR_CUDA_POST_LAUNCH_CHECK();

                sort_1d_thrust<U>(d_slice, d_slice, d_idx, dim_size, false, stream);

                find_mode_kernel<U><<<1, 1, 0, stream>>>(
                    d_slice, d_idx,
                    values.data<U>(), indices.data<int64_t>(),
                    dim_size, inner_size, outer, inner);
                TENZOR_CUDA_POST_LAUNCH_CHECK();
            }
        }
    };

    auto launch_half = [&]<typename HalfT>() {
        int cvt_block = 256;
        int cvt_grid = (dim_size + cvt_block - 1) / cvt_block;

        backend::CachedMemoryGuard half_slice_guard(dim_size * sizeof(HalfT));
        HalfT* d_half_slice = static_cast<HalfT*>(half_slice_guard.get());
        backend::CachedMemoryGuard f32_slice_guard(dim_size * sizeof(float));
        float* d_f32_slice = static_cast<float*>(f32_slice_guard.get());
        backend::CachedMemoryGuard idx_guard(dim_size * sizeof(int64_t));
        int64_t* d_idx = static_cast<int64_t*>(idx_guard.get());

        int block = 256;
        int grid = std::min(int((dim_size + block - 1) / block), 1024);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                // Extract half slice
                extract_slice_kernel<HalfT><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const HalfT*>(input_cont.data_ptr()),
                    d_half_slice, dim_size, inner_size, outer, inner);
                TENZOR_CUDA_POST_LAUNCH_CHECK();

                // Convert to float for sorting
                half_to_float_kernel<HalfT><<<cvt_grid, cvt_block, 0, stream>>>(
                    d_half_slice, d_f32_slice, dim_size);
                TENZOR_CUDA_POST_LAUNCH_CHECK();

                // Sort float copy (indices track original positions)
                sort_1d_thrust<float>(d_f32_slice, d_f32_slice, d_idx, dim_size, false, stream);

                // Find mode from sorted float values
                // We need to write the original half value, not the float approximation
                // find_mode_kernel gives us the index into the original slice
                // So we use float for comparison but write original half value via index

                // First find mode index using float sorted data
                // We'll store the mode index, then gather from original half slice
                // Use a temporary single-element buffer for the mode's original index
                backend::CachedMemoryGuard mode_val_guard(sizeof(float));
                float* d_mode_val = static_cast<float*>(mode_val_guard.get());
                backend::CachedMemoryGuard mode_idx_guard(sizeof(int64_t));
                int64_t* d_mode_idx = static_cast<int64_t*>(mode_idx_guard.get());

                // find_mode writes to out_vals/out_idx at [outer*inner_size+inner]
                // We can write directly to the output tensors
                // But output tensors are half type and find_mode_kernel uses T=float...
                // Instead, use a float temp and convert

                // Use find_mode on float slice, writing to temp buffers
                find_mode_kernel<float><<<1, 1, 0, stream>>>(
                    d_f32_slice, d_idx, d_mode_val, d_mode_idx,
                    dim_size, 1, 0, 0);
                TENZOR_CUDA_POST_LAUNCH_CHECK();

                // Write mode index to output indices
                int64_t out_offset = outer * inner_size + inner;
                TENZOR_CUDA_CHECK(cudaMemcpyAsync(
                    indices.data<int64_t>() + out_offset,
                    d_mode_idx, sizeof(int64_t),
                    cudaMemcpyDeviceToDevice, stream));

                // Gather the original half value using the mode index
                // d_half_slice[d_mode_idx[0]] -> out_vals[out_offset]
                gather_by_indices_kernel<HalfT><<<1, 1, 0, stream>>>(
                    d_half_slice, d_mode_idx,
                    reinterpret_cast<HalfT*>(values.data_ptr()) + out_offset,
                    1);
                TENZOR_CUDA_POST_LAUNCH_CHECK();
            }
        }
    };

    switch (dtype) {
        case DType::Float32:  launch.template operator()<float>(); break;
        case DType::Float64:  launch.template operator()<double>(); break;
        case DType::Int32:    launch.template operator()<int32_t>(); break;
        case DType::Int64:    launch.template operator()<int64_t>(); break;
        case DType::Float16:  launch_half.template operator()<__half>(); break;
        case DType::BFloat16: launch_half.template operator()<__nv_bfloat16>(); break;
        default: throw std::runtime_error("mode CUDA: unsupported dtype");
    }

    return {values, indices};
}

// ============================================================================
// Bernoulli sampling kernel
// ============================================================================

__global__ void bernoulli_kernel_impl(const float* probs, float* output,
                                       int64_t n, uint64_t seed) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Simple LCG-based PRNG per thread (sufficient for sampling)
    uint64_t state = seed + tid * 6364136223846793005ULL + 1442695040888963407ULL;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float u = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);

    output[tid] = (u < probs[tid]) ? 1.0f : 0.0f;
}

auto bernoulli_kernel(const Tensor& probs, cudaStream_t stream) -> Tensor {
    auto input = probs.contiguous();
    if (input.dtype() != DType::Float32) {
        input = input.to(DType::Float32);
    }
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto result = Tensor(shape_vec, DType::Float32, input.device());
    int64_t n = input.numel();
    if (n == 0) return result;

    int threads = 256;
    int blocks_n = (n + threads - 1) / threads;

    // Use a time-based seed
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    bernoulli_kernel_impl<<<blocks_n, threads, 0, stream>>>(
        input.data<float>(), result.data<float>(), n, seed);
    return result;
}

// ============================================================================
// Poisson sampling kernel (Knuth algorithm)
// ============================================================================

__global__ void poisson_kernel_impl(const float* rates, int64_t* output,
                                     int64_t n, uint64_t seed) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float lambda = rates[tid];

    // Knuth algorithm for Poisson sampling
    float L = expf(-lambda);
    int64_t k = 0;
    float p = 1.0f;

    // Simple LCG-based PRNG per thread (same pattern as bernoulli)
    uint64_t state = seed + tid * 6364136223846793005ULL + 1442695040888963407ULL;

    do {
        k++;
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float u = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);
        p *= u;
    } while (p > L);

    output[tid] = k - 1;
}

auto poisson_sample_kernel(const Tensor& rates, cudaStream_t stream) -> Tensor {
    auto input = rates.contiguous();
    if (input.dtype() != DType::Float32) {
        input = input.to(DType::Float32);
    }
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto result = Tensor(shape_vec, DType::Int64, input.device());
    int64_t n = input.numel();
    if (n == 0) return result;

    int threads = 256;
    int blocks_n = (n + threads - 1) / threads;

    // Use a time-based seed
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    poisson_kernel_impl<<<blocks_n, threads, 0, stream>>>(
        input.data<float>(), result.data<int64_t>(), n, seed);
    return result;
}

// ============================================================================
// Normal sampling kernel (Box-Muller transform)
// ============================================================================

__global__ void normal_sample_kernel_impl(const float* mean, const float* stddev,
                                           float* output, int64_t n, uint64_t seed) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // LCG-based PRNG per thread (same pattern as bernoulli/poisson)
    uint64_t state = seed + tid * 6364136223846793005ULL + 1442695040888963407ULL;

    // Generate two uniform random numbers for Box-Muller
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float u1 = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);
    // Clamp u1 away from zero to avoid log(0)
    u1 = fmaxf(u1, 1.0e-7f);

    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float u2 = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);

    // Box-Muller transform
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);

    output[tid] = mean[tid] + stddev[tid] * z;
}

auto normal_sample_kernel(const Tensor& mean, const Tensor& stddev, cudaStream_t stream) -> Tensor {
    auto m = mean.contiguous();
    auto s = stddev.contiguous();
    if (m.dtype() != DType::Float32) m = m.to(DType::Float32);
    if (s.dtype() != DType::Float32) s = s.to(DType::Float32);

    std::vector<int64_t> shape_vec(m.shape().begin(), m.shape().end());
    auto result = Tensor(shape_vec, DType::Float32, m.device());
    int64_t n = m.numel();
    if (n == 0) return result;

    int threads = 256;
    int blocks_n = (n + threads - 1) / threads;
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    normal_sample_kernel_impl<<<blocks_n, threads, 0, stream>>>(
        m.data<float>(), s.data<float>(), result.data<float>(), n, seed);
    return result;
}

// ============================================================================
// Exponential sampling kernel (inverse CDF method)
// ============================================================================

__global__ void exponential_sample_kernel_impl(const float* rate, float* output,
                                                int64_t n, uint64_t seed) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // LCG-based PRNG per thread
    uint64_t state = seed + tid * 6364136223846793005ULL + 1442695040888963407ULL;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float u = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31);
    // Clamp away from 1.0 to avoid log(0)
    u = fminf(u, 1.0f - 1.0e-7f);

    // Inverse CDF: -log(1 - u) / rate
    output[tid] = -logf(1.0f - u) / rate[tid];
}

auto exponential_sample_kernel(const Tensor& rate, cudaStream_t stream) -> Tensor {
    auto input = rate.contiguous();
    if (input.dtype() != DType::Float32) input = input.to(DType::Float32);

    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto result = Tensor(shape_vec, DType::Float32, input.device());
    int64_t n = input.numel();
    if (n == 0) return result;

    int threads = 256;
    int blocks_n = (n + threads - 1) / threads;
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    exponential_sample_kernel_impl<<<blocks_n, threads, 0, stream>>>(
        input.data<float>(), result.data<float>(), n, seed);
    return result;
}

// ============================================================================
// Multinomial sampling kernel
// ============================================================================

__global__ void multinomial_cdf_kernel(const float* probs, float* cdf,
                                        int64_t num_categories) {
    // Simple single-block CDF computation for one distribution
    // For batch, call per distribution
    extern __shared__ float shared[];
    int tid = threadIdx.x;

    // Load and compute prefix sum
    float val = (tid < num_categories) ? probs[tid] : 0.0f;
    shared[tid] = val;
    __syncthreads();

    // Inclusive scan (Blelloch-style, simplified)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float tmp = (tid >= stride) ? shared[tid - stride] : 0.0f;
        __syncthreads();
        shared[tid] += tmp;
        __syncthreads();
    }

    if (tid < num_categories) {
        cdf[tid] = shared[tid];
    }
}

__global__ void multinomial_sample_kernel(const float* cdf, int64_t* output,
                                           int64_t num_categories,
                                           int64_t num_samples, float total,
                                           uint64_t seed) {
    int64_t sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_samples) return;

    // Generate random number in [0, total)
    uint64_t state = seed + sid * 6364136223846793005ULL + 1442695040888963407ULL;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float u = static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31) * total;

    // Binary search in CDF
    int64_t lo = 0, hi = num_categories - 1;
    while (lo < hi) {
        int64_t mid = (lo + hi) / 2;
        if (cdf[mid] <= u) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    output[sid] = lo;
}

auto multinomial_kernel(const Tensor& probs, int64_t num_samples,
                        bool replacement, cudaStream_t stream) -> Tensor {
    auto input = probs.contiguous();
    if (input.dtype() != DType::Float32) {
        input = input.to(DType::Float32);
    }

    // Handle 1D and 2D
    bool was_1d = (input.dim() == 1);
    if (was_1d) {
        input = input.reshape({1, input.numel()});
    }

    int64_t batch_size = input.shape()[0];
    int64_t num_categories = input.shape()[1];
    auto result = Tensor({batch_size, num_samples}, DType::Int64, input.device());
    auto cdf_buf = Tensor({batch_size, num_categories}, DType::Float32, input.device());

    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* prob_ptr = input.data<float>() + b * num_categories;
        float* cdf_ptr = cdf_buf.data<float>() + b * num_categories;
        int64_t* out_ptr = result.data<int64_t>() + b * num_samples;

        // Compute CDF
        int block_size = 1;
        while (block_size < num_categories) block_size *= 2;
        if (block_size > 1024) block_size = 1024;
        multinomial_cdf_kernel<<<1, block_size, block_size * sizeof(float), stream>>>(
            prob_ptr, cdf_ptr, num_categories);

        // Get total (last element of CDF) - copy to host
        float total = 0.0f;
        cudaMemcpyAsync(&total, cdf_ptr + num_categories - 1, sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (total <= 0.0f) total = 1.0f;

        // Sample
        int threads = 256;
        int blocks = (num_samples + threads - 1) / threads;
        multinomial_sample_kernel<<<blocks, threads, 0, stream>>>(
            cdf_ptr, out_ptr, num_categories, num_samples, total,
            seed + b * 1000003);
    }

    if (was_1d) {
        result = result.reshape({num_samples});
    }
    return result;
}

// ============================================================================
// Bucketize kernel
// ============================================================================

__global__ void bucketize_kernel_impl(const float* input, const float* boundaries,
                                       int64_t* output, int64_t n,
                                       int64_t num_boundaries, bool right) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float val = input[tid];
    // Binary search
    int64_t lo = 0, hi = num_boundaries;
    while (lo < hi) {
        int64_t mid = (lo + hi) / 2;
        bool cond = right ? (boundaries[mid] <= val) : (boundaries[mid] < val);
        if (cond) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    output[tid] = lo;
}

auto bucketize_kernel(const Tensor& input, const Tensor& boundaries,
                      bool right, cudaStream_t stream) -> Tensor {
    auto in_contig = input.contiguous();
    auto bound_contig = boundaries.contiguous();
    if (in_contig.dtype() != DType::Float32) {
        in_contig = in_contig.to(DType::Float32);
    }
    if (bound_contig.dtype() != DType::Float32) {
        bound_contig = bound_contig.to(DType::Float32);
    }

    std::vector<int64_t> in_shape(in_contig.shape().begin(), in_contig.shape().end());
    auto result = Tensor(in_shape, DType::Int64, in_contig.device());
    int64_t n = in_contig.numel();
    if (n == 0) return result;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    bucketize_kernel_impl<<<blocks, threads, 0, stream>>>(
        in_contig.data<float>(), bound_contig.data<float>(),
        result.data<int64_t>(), n, bound_contig.numel(), right);
    return result;
}

// ============================================================================
// Histogram kernel
// ============================================================================

__global__ void histogram_kernel_impl(const float* input, int64_t* counts,
                                       int64_t n, int64_t num_bins,
                                       float min_val, float bin_width) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float val = input[tid];
    int64_t bin = static_cast<int64_t>((val - min_val) / bin_width);
    if (bin < 0) bin = 0;
    if (bin >= num_bins) bin = num_bins - 1;
    atomicAdd(reinterpret_cast<unsigned long long*>(&counts[bin]),
              static_cast<unsigned long long>(1));
}

// Fill bin-edge tensor on device: edges[i] = min + i * bin_width
__global__ void fill_bin_edges_kernel(float* edges, float min_val, float bin_width, int64_t num_edges) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_edges) return;
    edges[i] = min_val + static_cast<float>(i) * bin_width;
}

auto histogram_kernel(const Tensor& input, int64_t bins,
                      double min_val, double max_val,
                      cudaStream_t stream) -> std::pair<Tensor, Tensor> {
    auto in_contig = input.contiguous();
    if (in_contig.dtype() != DType::Float32) {
        in_contig = in_contig.to(DType::Float32);
    }

    int64_t n = in_contig.numel();

    // Auto-detect range if not specified — compute min/max on device via CUB,
    // then read 2 scalars back to host (necessary metadata, not a CPU fallback).
    if (min_val == 0.0 && max_val == 0.0 && n > 0) {
        backend::CachedMemoryGuard scratch_guard(2 * sizeof(float));
        float* d_min_max = static_cast<float*>(scratch_guard.get());

        // CUB Min reduction
        size_t temp_min = 0;
        cub::DeviceReduce::Min(nullptr, temp_min, in_contig.data<float>(), d_min_max,
                               static_cast<int>(n), stream);
        backend::CachedMemoryGuard min_temp_guard(temp_min);
        cub::DeviceReduce::Min(min_temp_guard.get(), temp_min, in_contig.data<float>(), d_min_max,
                               static_cast<int>(n), stream);

        // CUB Max reduction
        size_t temp_max = 0;
        cub::DeviceReduce::Max(nullptr, temp_max, in_contig.data<float>(), d_min_max + 1,
                               static_cast<int>(n), stream);
        backend::CachedMemoryGuard max_temp_guard(temp_max);
        cub::DeviceReduce::Max(max_temp_guard.get(), temp_max, in_contig.data<float>(), d_min_max + 1,
                               static_cast<int>(n), stream);

        // Scalar readback (2 floats) — unavoidable to compute bin_width on host
        float h_min_max[2];
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(h_min_max, d_min_max, 2 * sizeof(float),
                                          cudaMemcpyDeviceToHost, stream));
        TENZOR_CUDA_CHECK(cudaStreamSynchronize(stream));
        min_val = h_min_max[0];
        max_val = h_min_max[1];
    }
    if (max_val <= min_val) max_val = min_val + 1.0;

    float bin_width = static_cast<float>((max_val - min_val) / bins);

    auto counts = tenzor::zeros({bins}, DType::Int64, in_contig.device());

    if (n > 0) {
        int threads = 256;
        int blocks_n = (n + threads - 1) / threads;
        histogram_kernel_impl<<<blocks_n, threads, 0, stream>>>(
            in_contig.data<float>(), counts.data<int64_t>(),
            n, bins, static_cast<float>(min_val), bin_width);
    }

    // Compute bin edges on-device
    auto edges = Tensor({bins + 1}, DType::Float32, in_contig.device());
    {
        int64_t num_edges = bins + 1;
        int threads_e = 128;
        int blocks_e = static_cast<int>((num_edges + threads_e - 1) / threads_e);
        fill_bin_edges_kernel<<<blocks_e, threads_e, 0, stream>>>(
            edges.data<float>(), static_cast<float>(min_val), bin_width, num_edges);
    }

    return {counts, edges};
}

// ============================================================================
// CDist (pairwise distance) kernel
// ============================================================================

__global__ void cdist_l2_kernel_impl(const float* x1, const float* x2,
                                      float* output,
                                      int64_t B, int64_t P, int64_t R, int64_t M) {
    // x1: (B, P, M), x2: (B, R, M), output: (B, P, R)
    // Block: (R, P) tiles per batch; gridDim.z = B
    int64_t b = blockIdx.z;
    int64_t p = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || p >= P || r >= R) return;

    const float* a_b = x1 + b * P * M;
    const float* b_b = x2 + b * R * M;
    float sum = 0.0f;
    for (int64_t m = 0; m < M; ++m) {
        float diff = a_b[p * M + m] - b_b[r * M + m];
        sum += diff * diff;
    }
    output[(b * P + p) * R + r] = sqrtf(sum);
}

// Generic Lp distance kernel: sum_m |a[m] - b[m]|^p, take p-th root.
// Specializations for p=1.0 (Manhattan) and p=inf would be faster but the
// generic path is correct for all finite p >= 1.
__global__ void cdist_lp_kernel_impl(const float* x1, const float* x2,
                                      float* output, float p,
                                      int64_t B, int64_t P, int64_t R, int64_t M) {
    int64_t b = blockIdx.z;
    int64_t p_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || p_idx >= P || r >= R) return;

    const float* a_b = x1 + b * P * M;
    const float* b_b = x2 + b * R * M;
    float sum = 0.0f;
    for (int64_t m = 0; m < M; ++m) {
        float diff = fabsf(a_b[p_idx * M + m] - b_b[r * M + m]);
        sum += powf(diff, p);
    }
    output[(b * P + p_idx) * R + r] = powf(sum, 1.0f / p);
}

// L1 specialization — skips the pow() calls for the common Manhattan case.
__global__ void cdist_l1_kernel_impl(const float* x1, const float* x2,
                                      float* output,
                                      int64_t B, int64_t P, int64_t R, int64_t M) {
    int64_t b = blockIdx.z;
    int64_t p_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || p_idx >= P || r >= R) return;

    const float* a_b = x1 + b * P * M;
    const float* b_b = x2 + b * R * M;
    float sum = 0.0f;
    for (int64_t m = 0; m < M; ++m) {
        sum += fabsf(a_b[p_idx * M + m] - b_b[r * M + m]);
    }
    output[(b * P + p_idx) * R + r] = sum;
}

auto cdist_kernel(const Tensor& x1, const Tensor& x2, double p,
                  cudaStream_t stream) -> Tensor {
    auto a = x1.contiguous();
    auto b = x2.contiguous();
    if (a.dtype() != DType::Float32) a = a.to(DType::Float32);
    if (b.dtype() != DType::Float32) b = b.to(DType::Float32);

    // Accept either 2D (P, M) or 3D (B, P, M). The 2D case is treated as B=1.
    int64_t B, P, M, R;
    if (a.ndim() == 2 && b.ndim() == 2) {
        B = 1;
        P = a.shape()[0];
        M = a.shape()[1];
        R = b.shape()[0];
    } else if (a.ndim() == 3 && b.ndim() == 3) {
        B = a.shape()[0];
        P = a.shape()[1];
        M = a.shape()[2];
        R = b.shape()[1];
        if (b.shape()[0] != B) {
            throw std::runtime_error("cdist: batch dimensions must match");
        }
    } else {
        throw std::runtime_error("cdist: inputs must be 2D or 3D");
    }

    std::vector<int64_t> result_shape;
    if (a.ndim() == 2) result_shape = {P, R};
    else               result_shape = {B, P, R};
    auto result = Tensor(result_shape, DType::Float32, a.device());

    if (B == 0 || P == 0 || R == 0) return result;

    dim3 threads(16, 16, 1);
    dim3 blocks((R + 15) / 16, (P + 15) / 16, B);
    // Specialize p=2.0 (L2, most common) and p=1.0 (Manhattan, second most
    // common) to avoid the per-element pow() in the generic kernel.
    if (p == 2.0) {
        cdist_l2_kernel_impl<<<blocks, threads, 0, stream>>>(
            a.data<float>(), b.data<float>(), result.data<float>(), B, P, R, M);
    } else if (p == 1.0) {
        cdist_l1_kernel_impl<<<blocks, threads, 0, stream>>>(
            a.data<float>(), b.data<float>(), result.data<float>(), B, P, R, M);
    } else {
        cdist_lp_kernel_impl<<<blocks, threads, 0, stream>>>(
            a.data<float>(), b.data<float>(), result.data<float>(),
            static_cast<float>(p), B, P, R, M);
    }

    return result;
}

// ============================================================================
// AdvancedIndex / AdvancedIndexPut — native CUDA implementations
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
// MAX_INDEX_DIMS bounds the indexed-dim count and stride/shape buffers stored
// in __constant__-style kernel argument arrays.

namespace {
constexpr int MAX_INDEX_DIMS = 16;
}

struct AdvancedIndexMeta {
    int num_indices;        // Number of dims that have an index tensor (indexed or null)
    int src_ndim;
    int num_pass_dims;
    int64_t bc_numel;
    int64_t pass_numel;
    int64_t src_shape[MAX_INDEX_DIMS];
    int64_t src_strides[MAX_INDEX_DIMS];
    int pass_dims[MAX_INDEX_DIMS];     // Source dim indices that are passthrough
    int is_indexed[MAX_INDEX_DIMS];    // 1 if dim is indexed, 0 if full-slice
};

template<typename T>
__global__ void advanced_index_gather_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const int64_t* __restrict__ const* __restrict__ idx_ptrs,  // [num_indices] pointers
    AdvancedIndexMeta meta,
    int64_t total_out
) {
    int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out_idx >= total_out) return;

    // Decode output position into (bc, p) where bc is broadcast index, p is pass index
    int64_t p = (meta.pass_numel > 0) ? (out_idx % meta.pass_numel) : 0;
    int64_t bc = (meta.pass_numel > 0) ? (out_idx / meta.pass_numel) : out_idx;

    // Compute src offset from indexed dims
    int64_t src_offset = 0;
    for (int i = 0; i < meta.num_indices; ++i) {
        if (meta.is_indexed[i]) {
            int64_t idx_val = idx_ptrs[i][bc];
            if (idx_val < 0) idx_val += meta.src_shape[i];
            // Bounds check elided for hot path
            src_offset += idx_val * meta.src_strides[i];
        }
    }

    // Compute pass offset from passthrough dims
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
__global__ void advanced_index_put_kernel(
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

// Build meta from src + indices and compute output shape.
// Returns {meta, output_shape, total_output_numel}.
struct PreparedAdvancedIndex {
    AdvancedIndexMeta meta;
    std::vector<int64_t> output_shape;
    int64_t total;
};

inline PreparedAdvancedIndex prepare_advanced_index(
    const Tensor& src,
    const Tensor* const* index_tensors,
    int64_t num_indices
) {
    PreparedAdvancedIndex out{};
    auto src_shape_span = src.shape();
    int64_t src_ndim = static_cast<int64_t>(src_shape_span.size());
    if (src_ndim > MAX_INDEX_DIMS) {
        throw std::runtime_error("AdvancedIndex: source ndim exceeds MAX_INDEX_DIMS");
    }
    if (num_indices > MAX_INDEX_DIMS) {
        throw std::runtime_error("AdvancedIndex: num_indices exceeds MAX_INDEX_DIMS");
    }

    out.meta.num_indices = static_cast<int>(num_indices);
    out.meta.src_ndim = static_cast<int>(src_ndim);

    for (int64_t i = 0; i < src_ndim; ++i) {
        out.meta.src_shape[i] = src_shape_span[i];
    }
    // Row-major strides
    out.meta.src_strides[src_ndim - 1] = 1;
    for (int64_t d = src_ndim - 2; d >= 0; --d) {
        out.meta.src_strides[d] = out.meta.src_strides[d + 1] * src_shape_span[d + 1];
    }

    // Identify is_indexed and broadcast shape
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

    // Output shape = broadcast_shape + passthrough dims
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
auto launch_advanced_index_gather(
    const Tensor& src,
    const Tensor* const* index_tensors,
    int64_t num_indices,
    cudaStream_t stream
) -> Tensor {
    auto prep = prepare_advanced_index(src, index_tensors, num_indices);
    Tensor src_contig = src.contiguous();
    Tensor result(prep.output_shape, src.dtype(), src.device());
    if (prep.total == 0) return result;

    // Pack idx pointers into a small device array
    std::vector<const int64_t*> host_ptrs(num_indices, nullptr);
    std::vector<Tensor> idx_contig(num_indices);
    for (int i = 0; i < num_indices; ++i) {
        if (prep.meta.is_indexed[i]) {
            idx_contig[i] = index_tensors[i]->contiguous();
            host_ptrs[i] = idx_contig[i].data<int64_t>();
        }
    }
    backend::CachedMemoryGuard ptr_buf_guard(num_indices * sizeof(const int64_t*));
    auto* d_idx_ptrs = static_cast<const int64_t**>(ptr_buf_guard.get());
    cudaMemcpyAsync(d_idx_ptrs, host_ptrs.data(),
                    num_indices * sizeof(const int64_t*),
                    cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = static_cast<int>((prep.total + threads - 1) / threads);
    // Use data_ptr() + reinterpret_cast for CUDA-native types (__half, __nv_bfloat16)
    // that don't have Tensor::data<T>() instantiations in the core library.
    advanced_index_gather_kernel<T><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const T*>(src_contig.data_ptr()),
        reinterpret_cast<T*>(result.data_ptr()),
        d_idx_ptrs,
        prep.meta,
        prep.total);
    TENZOR_CUDA_CHECK(cudaGetLastError());
    return result;
}

template<typename T>
auto launch_advanced_index_put(
    const Tensor& src,
    const Tensor& values,
    const Tensor* const* index_tensors,
    int64_t num_indices,
    cudaStream_t stream
) -> Tensor {
    auto prep = prepare_advanced_index(src, index_tensors, num_indices);
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
    backend::CachedMemoryGuard ptr_buf_guard(num_indices * sizeof(const int64_t*));
    auto* d_idx_ptrs = static_cast<const int64_t**>(ptr_buf_guard.get());
    cudaMemcpyAsync(d_idx_ptrs, host_ptrs.data(),
                    num_indices * sizeof(const int64_t*),
                    cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = static_cast<int>((prep.total + threads - 1) / threads);
    // Use data_ptr() + reinterpret_cast for CUDA-native types (__half, __nv_bfloat16)
    // that don't have Tensor::data<T>() instantiations in the core library.
    advanced_index_put_kernel<T><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T*>(result_contig.data_ptr()),
        reinterpret_cast<const T*>(values_contig.data_ptr()),
        d_idx_ptrs,
        prep.meta,
        prep.total);
    TENZOR_CUDA_CHECK(cudaGetLastError());
    return result_contig;
}

template<typename FnFloat, typename FnDouble, typename FnInt32, typename FnInt64>
auto dispatch_advanced_index_dtype(DType dt,
                                   FnFloat ff, FnDouble fd, FnInt32 fi32, FnInt64 fi64,
                                   const std::string& op) {
    switch (dt) {
        case DType::Float32: return ff();
        case DType::Float64: return fd();
        case DType::Int32:   return fi32();
        case DType::Int64:   return fi64();
        default:
            throw std::runtime_error(op + ": unsupported dtype (only Float32, Float64, Int32, Int64)");
    }
}

}  // namespace

auto advanced_index_cuda_kernel(
    const Tensor& src, const std::vector<Tensor>& indices,
    int64_t num_indices, cudaStream_t stream) -> Tensor {
    std::vector<const Tensor*> idx_ptrs(num_indices);
    for (int64_t i = 0; i < num_indices; ++i) idx_ptrs[i] = &indices[i];

    if (src.dtype() == DType::Float32) {
        return launch_advanced_index_gather<float>(src, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Float64) {
        return launch_advanced_index_gather<double>(src, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Int32) {
        return launch_advanced_index_gather<int32_t>(src, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Int64) {
        return launch_advanced_index_gather<int64_t>(src, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Float16) {
        return launch_advanced_index_gather<__half>(src, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::BFloat16) {
        return launch_advanced_index_gather<__nv_bfloat16>(src, idx_ptrs.data(), num_indices, stream);
    }
    throw std::runtime_error("AdvancedIndex CUDA: unsupported dtype");
}

auto advanced_index_put_cuda_kernel(
    const Tensor& src, const std::vector<Tensor>& indices,
    const Tensor& values, int64_t num_indices, cudaStream_t stream) -> Tensor {
    std::vector<const Tensor*> idx_ptrs(num_indices);
    for (int64_t i = 0; i < num_indices; ++i) idx_ptrs[i] = &indices[i];

    if (src.dtype() == DType::Float32) {
        return launch_advanced_index_put<float>(src, values, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Float64) {
        return launch_advanced_index_put<double>(src, values, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Int32) {
        return launch_advanced_index_put<int32_t>(src, values, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Int64) {
        return launch_advanced_index_put<int64_t>(src, values, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Float16) {
        return launch_advanced_index_put<__half>(src, values, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::BFloat16) {
        return launch_advanced_index_put<__nv_bfloat16>(src, values, idx_ptrs.data(), num_indices, stream);
    }
    throw std::runtime_error("AdvancedIndexPut CUDA: unsupported dtype");
}

// ============================================================================
// STFT / ISTFT — native CUDA implementations
// ============================================================================
//
// Strategy: build a (B, num_frames, n_fft) framed+windowed tensor on-device
// (with optional reflection padding embedded in the indexing math), then call
// the existing cuda_rfft_kernel / cuda_fft_kernel which already wrap cuFFT
// with batching. Inverse path uses cuda_irfft_kernel + an overlap-add kernel.

// Forward decls for the FFT kernels (defined in fft.cu)
auto cuda_rfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                      const std::string& norm, cudaStream_t stream) -> Tensor;
auto cuda_fft_kernel(const Tensor& input, int64_t dim, int64_t n,
                     const std::string& norm, cudaStream_t stream) -> Tensor;
auto cuda_irfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                       const std::string& norm, cudaStream_t stream) -> Tensor;
auto cuda_ifft_kernel(const Tensor& input, int64_t dim, int64_t n,
                      const std::string& norm, cudaStream_t stream) -> Tensor;

__global__ void stft_frame_window_kernel(
    const float* __restrict__ signal,
    const float* __restrict__ window,
    float* __restrict__ framed,
    int64_t batch_size,
    int64_t signal_length,
    int64_t num_frames,
    int64_t n_fft,
    int64_t hop_length,
    int64_t pad,           // n_fft/2 if center else 0
    int64_t total          // batch_size * num_frames * n_fft
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t i = idx % n_fft;
    int64_t f = (idx / n_fft) % num_frames;
    int64_t b = idx / (n_fft * num_frames);

    // Effective signal index (with center-padding offset)
    int64_t sig_idx = f * hop_length + i - pad;

    float val;
    if (signal_length <= 1) {
        val = (signal_length == 1) ? signal[b * signal_length] : 0.0f;
    } else if (sig_idx < 0) {
        // Reflect-pad on the left
        int64_t r = -sig_idx;
        if (r >= signal_length) r = signal_length - 1;
        val = signal[b * signal_length + r];
    } else if (sig_idx >= signal_length) {
        // Reflect-pad on the right
        int64_t r = 2 * signal_length - 2 - sig_idx;
        if (r < 0) r = 0;
        val = signal[b * signal_length + r];
    } else {
        val = signal[b * signal_length + sig_idx];
    }

    framed[idx] = val * window[i];
}

auto stft_cuda_kernel(const Tensor& input, int64_t n_fft,
                      int64_t hop_length, int64_t win_length,
                      const Tensor& window, bool center,
                      bool normalized, bool onesided,
                      cudaStream_t stream) -> Tensor {
    if (n_fft <= 0) throw std::runtime_error("stft: n_fft must be > 0");
    if (hop_length <= 0) hop_length = n_fft / 4;
    if (win_length <= 0) win_length = n_fft;
    if (win_length > n_fft) throw std::runtime_error("stft: win_length must be <= n_fft");

    auto input_f32 = (input.dtype() != DType::Float32) ? input.to(DType::Float32) : input.contiguous();

    auto in_shape = input_f32.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    if (ndim < 1) throw std::runtime_error("stft: input must have at least 1 dim");

    int64_t signal_length = in_shape[ndim - 1];
    int64_t batch_size = 1;
    for (int64_t d = 0; d < ndim - 1; ++d) batch_size *= in_shape[d];

    int64_t pad = center ? (n_fft / 2) : 0;
    int64_t padded_length = signal_length + 2 * pad;
    int64_t num_frames = (padded_length - n_fft) / hop_length + 1;
    if (num_frames <= 0) throw std::runtime_error("stft: signal too short");

    int64_t freq_bins = onesided ? (n_fft / 2 + 1) : n_fft;

    // Build window data: pad win_length to n_fft (rectangular default)
    Tensor window_dev({n_fft}, DType::Float32, input.device());
    cudaMemsetAsync(window_dev.data_ptr(), 0, n_fft * sizeof(float), stream);
    int64_t win_offset = (n_fft - win_length) / 2;
    if (window.is_valid() && window.numel() > 0) {
        Tensor win_f32 = (window.dtype() != DType::Float32) ? window.to(DType::Float32)
                                                              : window.contiguous();
        cudaMemcpyAsync(static_cast<float*>(window_dev.data_ptr()) + win_offset,
                        win_f32.data<float>(),
                        win_length * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        // Fill with ones at [win_offset, win_offset+win_length)
        std::vector<float> host_ones(win_length, 1.0f);
        cudaMemcpyAsync(static_cast<float*>(window_dev.data_ptr()) + win_offset,
                        host_ones.data(),
                        win_length * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }

    // Frame+window kernel: produce (batch_size, num_frames, n_fft)
    Tensor framed({batch_size, num_frames, n_fft}, DType::Float32, input.device());
    int64_t total = batch_size * num_frames * n_fft;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    stft_frame_window_kernel<<<blocks, threads, 0, stream>>>(
        input_f32.data<float>(),
        window_dev.data<float>(),
        framed.data<float>(),
        batch_size, signal_length, num_frames, n_fft, hop_length, pad, total);
    TENZOR_CUDA_CHECK(cudaGetLastError());

    // Run batched FFT along the last dim (n_fft)
    Tensor fft_out;
    if (onesided) {
        fft_out = cuda_rfft_kernel(framed, /*dim=*/2, n_fft,
                                   normalized ? "ortho" : "backward", stream);
    } else {
        fft_out = cuda_fft_kernel(framed, /*dim=*/2, n_fft,
                                  normalized ? "ortho" : "backward", stream);
    }
    // fft_out shape: (batch_size, num_frames, freq_bins) Complex64

    // Output expects shape (..., freq_bins, num_frames). We need to transpose
    // the last two dims of fft_out and re-add the original leading batch dims.
    // tenzor::transpose handles this on-device.
    Tensor transposed = tenzor::transpose(fft_out, -1, -2);
    // transposed shape: (batch_size, freq_bins, num_frames)

    // Reshape to original leading dims + (freq_bins, num_frames)
    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim - 1; ++d) out_shape.push_back(in_shape[d]);
    out_shape.push_back(freq_bins);
    out_shape.push_back(num_frames);
    return tenzor::reshape(transposed, out_shape);
}

__global__ void istft_overlap_add_kernel(
    const float* __restrict__ time_frames,    // (B, num_frames, n_fft) — output of IFFT
    const float* __restrict__ window,         // (n_fft,)
    float* __restrict__ output,               // (B, expected_length)
    float* __restrict__ window_sum,           // (B, expected_length) — accumulator
    int64_t batch_size,
    int64_t num_frames,
    int64_t n_fft,
    int64_t hop_length,
    int64_t expected_length
) {
    // Each thread handles one element of one frame: (b, f, i)
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch_size * num_frames * n_fft;
    if (idx >= total) return;

    int64_t i = idx % n_fft;
    int64_t f = (idx / n_fft) % num_frames;
    int64_t b = idx / (n_fft * num_frames);

    int64_t out_pos = f * hop_length + i;
    if (out_pos < 0 || out_pos >= expected_length) return;

    float w = window[i];
    float val = time_frames[((b * num_frames) + f) * n_fft + i] * w;
    float w2 = w * w;

    // Atomic accumulation since multiple frames overlap
    atomicAdd(&output[b * expected_length + out_pos], val);
    atomicAdd(&window_sum[b * expected_length + out_pos], w2);
}

__global__ void istft_normalize_kernel(
    float* __restrict__ output,
    const float* __restrict__ window_sum,
    int64_t total
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float ws = window_sum[idx];
    if (ws > 1e-11f) output[idx] /= ws;
}

auto istft_cuda_kernel(const Tensor& input, int64_t n_fft,
                       int64_t hop_length, int64_t win_length,
                       const Tensor& window, bool center,
                       bool /*normalized*/, bool onesided,
                       int64_t length, cudaStream_t stream) -> Tensor {
    if (n_fft <= 0) throw std::runtime_error("istft: n_fft must be > 0");
    if (hop_length <= 0) hop_length = n_fft / 4;
    if (win_length <= 0) win_length = n_fft;

    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    if (ndim < 2) throw std::runtime_error("istft: input must have ≥ 2 dims");

    int64_t freq_bins = in_shape[ndim - 2];
    int64_t num_frames = in_shape[ndim - 1];
    int64_t batch_size = 1;
    for (int64_t d = 0; d < ndim - 2; ++d) batch_size *= in_shape[d];

    int64_t expected_length = n_fft + (num_frames - 1) * hop_length;

    // Reshape/transpose input from (..., freq_bins, num_frames) → (B, num_frames, freq_bins)
    Tensor input_c64 = (input.dtype() != DType::Complex64) ? input.to(DType::Complex64)
                                                            : input.contiguous();
    Tensor reshaped = tenzor::reshape(input_c64, {batch_size, freq_bins, num_frames});
    Tensor transposed = tenzor::transpose(reshaped, -1, -2);
    // transposed: (B, num_frames, freq_bins)
    Tensor transposed_contig = transposed.contiguous();

    // Inverse FFT along last dim → (B, num_frames, n_fft) real
    Tensor time_frames;
    if (onesided) {
        time_frames = cuda_irfft_kernel(transposed_contig, /*dim=*/2, n_fft,
                                        "backward", stream);
    } else {
        time_frames = cuda_ifft_kernel(transposed_contig, /*dim=*/2, n_fft,
                                       "backward", stream);
        // ifft returns complex; take real part
        // Use existing real() dispatch to extract real component
        time_frames = tenzor::real(time_frames);
    }

    // Build window
    Tensor window_dev({n_fft}, DType::Float32, input.device());
    cudaMemsetAsync(window_dev.data_ptr(), 0, n_fft * sizeof(float), stream);
    int64_t win_offset = (n_fft - win_length) / 2;
    if (window.is_valid() && window.numel() > 0) {
        Tensor win_f32 = (window.dtype() != DType::Float32) ? window.to(DType::Float32)
                                                              : window.contiguous();
        cudaMemcpyAsync(static_cast<float*>(window_dev.data_ptr()) + win_offset,
                        win_f32.data<float>(),
                        win_length * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        std::vector<float> host_ones(win_length, 1.0f);
        cudaMemcpyAsync(static_cast<float*>(window_dev.data_ptr()) + win_offset,
                        host_ones.data(),
                        win_length * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }

    // Allocate output and window_sum accumulators (zeroed)
    Tensor output_buf({batch_size, expected_length}, DType::Float32, input.device());
    Tensor wsum_buf({batch_size, expected_length}, DType::Float32, input.device());
    cudaMemsetAsync(output_buf.data_ptr(), 0, batch_size * expected_length * sizeof(float), stream);
    cudaMemsetAsync(wsum_buf.data_ptr(),  0, batch_size * expected_length * sizeof(float), stream);

    // Overlap-add
    {
        int64_t total = batch_size * num_frames * n_fft;
        int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        istft_overlap_add_kernel<<<blocks, threads, 0, stream>>>(
            time_frames.data<float>(),
            window_dev.data<float>(),
            output_buf.data<float>(),
            wsum_buf.data<float>(),
            batch_size, num_frames, n_fft, hop_length, expected_length);
        TENZOR_CUDA_CHECK(cudaGetLastError());
    }

    // Normalize by accumulated window²
    {
        int64_t total = batch_size * expected_length;
        int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        istft_normalize_kernel<<<blocks, threads, 0, stream>>>(
            output_buf.data<float>(),
            wsum_buf.data<float>(),
            total);
        TENZOR_CUDA_CHECK(cudaGetLastError());
    }

    // Trim center padding if needed
    int64_t pad = center ? (n_fft / 2) : 0;
    int64_t out_length = expected_length - 2 * pad;
    if (length >= 0) out_length = length;

    Tensor result;
    if (pad == 0 && length < 0) {
        result = output_buf;
    } else {
        // Slice [:, pad : pad + out_length]
        Tensor sliced = tenzor::slice(output_buf, 1, pad, pad + out_length);
        result = sliced.contiguous();
    }

    // Reshape to original batch dims + (out_length)
    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim - 2; ++d) out_shape.push_back(in_shape[d]);
    out_shape.push_back(out_length);
    return tenzor::reshape(result, out_shape);
}

// ============================================================================
// logcumsumexp — CUDA kernel
// ============================================================================

// Each thread handles one "slice" (one combination of outer/inner indices).
// Sequential scan along the dim dimension within each thread.
__global__ void logcumsumexp_kernel_f32(
    const float* __restrict__ input, float* __restrict__ output,
    int64_t dim_size, int64_t inner_size, int64_t total_slices)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_slices) return;

    int64_t outer = idx / inner_size;
    int64_t inner = idx % inner_size;

    const float NEG_INF_F = __int_as_float(0xff800000);
    float running_max = NEG_INF_F;
    float running_lse = NEG_INF_F;

    for (int64_t i = 0; i < dim_size; ++i) {
        int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
        float x = input[offset];
        float new_max = fmaxf(running_max, x);

        if (::isinf(new_max) && new_max < 0.0f) {
            running_lse = NEG_INF_F;
        } else {
            running_lse = new_max + logf(expf(running_lse - new_max) + expf(x - new_max));
        }
        running_max = new_max;
        output[offset] = running_lse;
    }
}

__global__ void logcumsumexp_kernel_f64(
    const double* __restrict__ input, double* __restrict__ output,
    int64_t dim_size, int64_t inner_size, int64_t total_slices)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_slices) return;

    int64_t outer = idx / inner_size;
    int64_t inner = idx % inner_size;

    const double NEG_INF_D = -1.0 / 0.0;
    double running_max = NEG_INF_D;
    double running_lse = NEG_INF_D;

    for (int64_t i = 0; i < dim_size; ++i) {
        int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
        double x = input[offset];
        double new_max = ::fmax(running_max, x);

        if (::isinf(new_max) && new_max < 0.0) {
            running_lse = NEG_INF_D;
        } else {
            running_lse = new_max + ::log(::exp(running_lse - new_max) + ::exp(x - new_max));
        }
        running_max = new_max;
        output[offset] = running_lse;
    }
}

auto logcumsumexp_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = input.ndim();
    const auto dtype = input.dtype();
    const auto device = input.device();

    if (dim < 0) dim += ndim;

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), dtype, device);

    int64_t dim_size = shape[dim];
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    int64_t total_slices = outer_size * inner_size;
    if (total_slices == 0 || dim_size == 0) return output;

    int block = 256;
    int grid = static_cast<int>((total_slices + block - 1) / block);

    // Half types: upcast to Float32
    if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        auto f32_input = input_cont.to(DType::Float32);
        auto f32_result = logcumsumexp_kernel(f32_input, dim, stream);
        return f32_result.to(dtype);
    }

    if (dtype == DType::Float32) {
        logcumsumexp_kernel_f32<<<grid, block, 0, stream>>>(
            input_cont.data<float>(), output.data<float>(),
            dim_size, inner_size, total_slices);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else if (dtype == DType::Float64) {
        logcumsumexp_kernel_f64<<<grid, block, 0, stream>>>(
            input_cont.data<double>(), output.data<double>(),
            dim_size, inner_size, total_slices);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("logcumsumexp_kernel: unsupported dtype");
    }

    return output;
}

// ============================================================================
// bincount — CUDA kernel
// ============================================================================

__global__ void bincount_no_weights_kernel(
    const int64_t* __restrict__ input, int64_t* __restrict__ output, int64_t n)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    atomicAdd(reinterpret_cast<unsigned long long*>(&output[input[idx]]), 1ULL);
}

__global__ void bincount_weights_f32_kernel(
    const int64_t* __restrict__ input, const float* __restrict__ weights,
    double* __restrict__ output, int64_t n)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    atomicAdd(&output[input[idx]], static_cast<double>(weights[idx]));
}

__global__ void bincount_weights_f64_kernel(
    const int64_t* __restrict__ input, const double* __restrict__ weights,
    double* __restrict__ output, int64_t n)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    atomicAdd(&output[input[idx]], weights[idx]);
}

__global__ void bincount_find_max_kernel(
    const int64_t* __restrict__ input, int64_t* __restrict__ max_val, int64_t n)
{
    __shared__ int64_t smax;
    if (threadIdx.x == 0) smax = -1;
    __syncthreads();

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicMax(reinterpret_cast<long long*>(&smax),
                  static_cast<long long>(input[idx]));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<long long*>(max_val),
                  static_cast<long long>(smax));
    }
}

auto bincount_kernel(const Tensor& input, const Tensor* weights,
                     int64_t minlength, cudaStream_t stream) -> Tensor
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    Tensor input_i64 = (input_cont.dtype() == DType::Int64)
        ? input_cont : input_cont.to(DType::Int64);

    int64_t n = input_i64.numel();
    auto device = input.device();

    // Find max value on GPU
    Tensor max_tensor({1}, DType::Int64, device);
    cudaMemsetAsync(max_tensor.data<int64_t>(), 0xFF, sizeof(int64_t), stream); // set to -1
    // Actually set to -1 properly
    int64_t neg_one = -1;
    cudaMemcpyAsync(max_tensor.data<int64_t>(), &neg_one, sizeof(int64_t),
                    cudaMemcpyHostToDevice, stream);

    if (n > 0) {
        int block = 256;
        int grid = static_cast<int>((n + block - 1) / block);
        bincount_find_max_kernel<<<grid, block, 0, stream>>>(
            input_i64.data<int64_t>(), max_tensor.data<int64_t>(), n);
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    }

    // Copy max back to host
    int64_t max_val = -1;
    cudaMemcpyAsync(&max_val, max_tensor.data<int64_t>(), sizeof(int64_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int64_t output_size = std::max(max_val + 1, minlength);

    bool has_weights = (weights != nullptr);

    if (has_weights) {
        Tensor output({output_size}, DType::Float64, device);
        cudaMemsetAsync(output.data<double>(), 0,
                        static_cast<size_t>(output_size) * sizeof(double), stream);

        if (n > 0) {
            int block = 256;
            int grid = static_cast<int>((n + block - 1) / block);
            Tensor w_cont = weights->is_contiguous() ? *weights : weights->contiguous();

            if (w_cont.dtype() == DType::Float64) {
                bincount_weights_f64_kernel<<<grid, block, 0, stream>>>(
                    input_i64.data<int64_t>(), w_cont.data<double>(),
                    output.data<double>(), n);
            } else {
                Tensor w_f32 = (w_cont.dtype() == DType::Float32) ? w_cont : w_cont.to(DType::Float32);
                bincount_weights_f32_kernel<<<grid, block, 0, stream>>>(
                    input_i64.data<int64_t>(), w_f32.data<float>(),
                    output.data<double>(), n);
            }
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }
        return output;
    } else {
        Tensor output({output_size}, DType::Int64, device);
        cudaMemsetAsync(output.data<int64_t>(), 0,
                        static_cast<size_t>(output_size) * sizeof(int64_t), stream);

        if (n > 0) {
            int block = 256;
            int grid = static_cast<int>((n + block - 1) / block);
            bincount_no_weights_kernel<<<grid, block, 0, stream>>>(
                input_i64.data<int64_t>(), output.data<int64_t>(), n);
            TENZOR_CUDA_POST_LAUNCH_CHECK();
        }
        return output;
    }
}

// ============================================================================
// CumMax kernel — cumulative maximum along a dimension (returns values + indices)
// ============================================================================

template<typename T>
__global__ void cummax_kernel_impl(
    const T* __restrict__ input, T* __restrict__ values, int64_t* __restrict__ indices,
    int64_t dim_size, int64_t inner_size, int64_t total_slices)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_slices) return;

    int64_t outer = idx / inner_size;
    int64_t inner = idx % inner_size;

    T running_max = input[outer * dim_size * inner_size + inner];
    int64_t running_idx = 0;
    values[outer * dim_size * inner_size + inner] = running_max;
    indices[outer * dim_size * inner_size + inner] = 0;

    for (int64_t i = 1; i < dim_size; ++i) {
        int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
        T val = input[offset];
        if (val > running_max) {
            running_max = val;
            running_idx = i;
        }
        values[offset] = running_max;
        indices[offset] = running_idx;
    }
}

auto cummax_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> std::pair<Tensor, Tensor>
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = input_cont.ndim();
    const int64_t dim_size = shape[dim];
    const auto dtype = input_cont.dtype();
    const auto device = input_cont.device();

    Tensor values(std::vector<int64_t>(shape.begin(), shape.end()), dtype, device);
    Tensor indices_out(std::vector<int64_t>(shape.begin(), shape.end()), DType::Int64, device);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    int64_t total_slices = outer_size * inner_size;
    int block = 256;
    int grid = static_cast<int>((total_slices + block - 1) / block);

    switch (dtype) {
        case DType::Float32:
            cummax_kernel_impl<float><<<grid, block, 0, stream>>>(
                input_cont.data<float>(), values.data<float>(), indices_out.data<int64_t>(),
                dim_size, inner_size, total_slices);
            break;
        case DType::Float64:
            cummax_kernel_impl<double><<<grid, block, 0, stream>>>(
                input_cont.data<double>(), values.data<double>(), indices_out.data<int64_t>(),
                dim_size, inner_size, total_slices);
            break;
        case DType::Int32:
            cummax_kernel_impl<int32_t><<<grid, block, 0, stream>>>(
                input_cont.data<int32_t>(), values.data<int32_t>(), indices_out.data<int64_t>(),
                dim_size, inner_size, total_slices);
            break;
        case DType::Int64:
            cummax_kernel_impl<int64_t><<<grid, block, 0, stream>>>(
                input_cont.data<int64_t>(), values.data<int64_t>(), indices_out.data<int64_t>(),
                dim_size, inner_size, total_slices);
            break;
        default:
            throw std::runtime_error("cummax CUDA: unsupported dtype");
    }
    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return {values, indices_out};
}

// ============================================================================
// CumMin kernel — cumulative minimum along a dimension (returns values + indices)
// ============================================================================

template<typename T>
__global__ void cummin_kernel_impl(
    const T* __restrict__ input, T* __restrict__ values, int64_t* __restrict__ indices,
    int64_t dim_size, int64_t inner_size, int64_t total_slices)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_slices) return;

    int64_t outer = idx / inner_size;
    int64_t inner = idx % inner_size;

    T running_min = input[outer * dim_size * inner_size + inner];
    int64_t running_idx = 0;
    values[outer * dim_size * inner_size + inner] = running_min;
    indices[outer * dim_size * inner_size + inner] = 0;

    for (int64_t i = 1; i < dim_size; ++i) {
        int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
        T val = input[offset];
        if (val < running_min) {
            running_min = val;
            running_idx = i;
        }
        values[offset] = running_min;
        indices[offset] = running_idx;
    }
}

auto cummin_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> std::pair<Tensor, Tensor>
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = input_cont.ndim();
    const int64_t dim_size = shape[dim];
    const auto dtype = input_cont.dtype();
    const auto device = input_cont.device();

    Tensor values(std::vector<int64_t>(shape.begin(), shape.end()), dtype, device);
    Tensor indices_out(std::vector<int64_t>(shape.begin(), shape.end()), DType::Int64, device);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    int64_t total_slices = outer_size * inner_size;
    int block = 256;
    int grid = static_cast<int>((total_slices + block - 1) / block);

    switch (dtype) {
        case DType::Float32:
            cummin_kernel_impl<float><<<grid, block, 0, stream>>>(
                input_cont.data<float>(), values.data<float>(), indices_out.data<int64_t>(),
                dim_size, inner_size, total_slices);
            break;
        case DType::Float64:
            cummin_kernel_impl<double><<<grid, block, 0, stream>>>(
                input_cont.data<double>(), values.data<double>(), indices_out.data<int64_t>(),
                dim_size, inner_size, total_slices);
            break;
        case DType::Int32:
            cummin_kernel_impl<int32_t><<<grid, block, 0, stream>>>(
                input_cont.data<int32_t>(), values.data<int32_t>(), indices_out.data<int64_t>(),
                dim_size, inner_size, total_slices);
            break;
        case DType::Int64:
            cummin_kernel_impl<int64_t><<<grid, block, 0, stream>>>(
                input_cont.data<int64_t>(), values.data<int64_t>(), indices_out.data<int64_t>(),
                dim_size, inner_size, total_slices);
            break;
        default:
            throw std::runtime_error("cummin CUDA: unsupported dtype");
    }
    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return {values, indices_out};
}

// ============================================================================
// Fmax kernel — NaN-aware element-wise maximum (IEEE 754-2008: if one is NaN, return the other)
// ============================================================================

template<typename T>
__global__ void fmax_kernel_impl(const T* __restrict__ a, const T* __restrict__ b,
                                  T* __restrict__ out, int64_t n)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += blockDim.x * gridDim.x) {
        out[idx] = ::fmax(static_cast<double>(a[idx]), static_cast<double>(b[idx]));
    }
}

template<>
__global__ void fmax_kernel_impl<float>(const float* __restrict__ a, const float* __restrict__ b,
                                         float* __restrict__ out, int64_t n)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += blockDim.x * gridDim.x) {
        out[idx] = ::fmaxf(a[idx], b[idx]);
    }
}

template<>
__global__ void fmax_kernel_impl<double>(const double* __restrict__ a, const double* __restrict__ b,
                                          double* __restrict__ out, int64_t n)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += blockDim.x * gridDim.x) {
        out[idx] = ::fmax(a[idx], b[idx]);
    }
}

auto fmax_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor
{
    Tensor a_cont = a.is_contiguous() ? a : a.contiguous();
    Tensor b_cont = b.is_contiguous() ? b : b.contiguous();
    int64_t n = a_cont.numel();
    Tensor output(std::vector<int64_t>(a_cont.shape().begin(), a_cont.shape().end()),
                  a_cont.dtype(), a_cont.device());

    int block = 256;
    int grid = std::min(static_cast<int>((n + block - 1) / block), 65535);

    switch (a_cont.dtype()) {
        case DType::Float32:
            fmax_kernel_impl<float><<<grid, block, 0, stream>>>(
                a_cont.data<float>(), b_cont.data<float>(), output.data<float>(), n);
            break;
        case DType::Float64:
            fmax_kernel_impl<double><<<grid, block, 0, stream>>>(
                a_cont.data<double>(), b_cont.data<double>(), output.data<double>(), n);
            break;
        case DType::Int32:
            fmax_kernel_impl<int32_t><<<grid, block, 0, stream>>>(
                a_cont.data<int32_t>(), b_cont.data<int32_t>(), output.data<int32_t>(), n);
            break;
        case DType::Int64:
            fmax_kernel_impl<int64_t><<<grid, block, 0, stream>>>(
                a_cont.data<int64_t>(), b_cont.data<int64_t>(), output.data<int64_t>(), n);
            break;
        default:
            throw std::runtime_error("fmax CUDA: unsupported dtype");
    }
    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return output;
}

// ============================================================================
// Fmin kernel — NaN-aware element-wise minimum (IEEE 754-2008: if one is NaN, return the other)
// ============================================================================

template<typename T>
__global__ void fmin_kernel_impl(const T* __restrict__ a, const T* __restrict__ b,
                                  T* __restrict__ out, int64_t n)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += blockDim.x * gridDim.x) {
        out[idx] = ::fmin(static_cast<double>(a[idx]), static_cast<double>(b[idx]));
    }
}

template<>
__global__ void fmin_kernel_impl<float>(const float* __restrict__ a, const float* __restrict__ b,
                                         float* __restrict__ out, int64_t n)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += blockDim.x * gridDim.x) {
        out[idx] = ::fminf(a[idx], b[idx]);
    }
}

template<>
__global__ void fmin_kernel_impl<double>(const double* __restrict__ a, const double* __restrict__ b,
                                          double* __restrict__ out, int64_t n)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += blockDim.x * gridDim.x) {
        out[idx] = ::fmin(a[idx], b[idx]);
    }
}

auto fmin_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor
{
    Tensor a_cont = a.is_contiguous() ? a : a.contiguous();
    Tensor b_cont = b.is_contiguous() ? b : b.contiguous();
    int64_t n = a_cont.numel();
    Tensor output(std::vector<int64_t>(a_cont.shape().begin(), a_cont.shape().end()),
                  a_cont.dtype(), a_cont.device());

    int block = 256;
    int grid = std::min(static_cast<int>((n + block - 1) / block), 65535);

    switch (a_cont.dtype()) {
        case DType::Float32:
            fmin_kernel_impl<float><<<grid, block, 0, stream>>>(
                a_cont.data<float>(), b_cont.data<float>(), output.data<float>(), n);
            break;
        case DType::Float64:
            fmin_kernel_impl<double><<<grid, block, 0, stream>>>(
                a_cont.data<double>(), b_cont.data<double>(), output.data<double>(), n);
            break;
        case DType::Int32:
            fmin_kernel_impl<int32_t><<<grid, block, 0, stream>>>(
                a_cont.data<int32_t>(), b_cont.data<int32_t>(), output.data<int32_t>(), n);
            break;
        case DType::Int64:
            fmin_kernel_impl<int64_t><<<grid, block, 0, stream>>>(
                a_cont.data<int64_t>(), b_cont.data<int64_t>(), output.data<int64_t>(), n);
            break;
        default:
            throw std::runtime_error("fmin CUDA: unsupported dtype");
    }
    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return output;
}

// ============================================================================
// Isin kernel — set membership test using sorted array + binary search
// ============================================================================

template<typename T>
__global__ void isin_kernel_impl(const T* __restrict__ elements, int64_t num_elements,
                                  const T* __restrict__ test_sorted, int64_t num_test,
                                  bool* __restrict__ output)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements;
         idx += blockDim.x * gridDim.x) {
        T val = elements[idx];
        // Binary search in sorted test_elements
        int64_t lo = 0, hi = num_test - 1;
        bool found = false;
        while (lo <= hi) {
            int64_t mid = lo + (hi - lo) / 2;
            T mid_val = test_sorted[mid];
            if (mid_val == val) {
                found = true;
                break;
            } else if (mid_val < val) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        output[idx] = found;
    }
}

auto isin_kernel(const Tensor& elements, const Tensor& test_elements, cudaStream_t stream) -> Tensor
{
    Tensor elem_cont = elements.is_contiguous() ? elements : elements.contiguous();
    Tensor test_cont = test_elements.is_contiguous() ? test_elements : test_elements.contiguous();

    // Sort test_elements on GPU using thrust
    Tensor test_sorted(std::vector<int64_t>(test_cont.shape().begin(), test_cont.shape().end()),
                       test_cont.dtype(), test_cont.device());
    cudaMemcpyAsync(test_sorted.data_ptr(), test_cont.data_ptr(),
                    test_cont.numel() * test_cont.element_size(),
                    cudaMemcpyDeviceToDevice, stream);

    int64_t num_test = test_sorted.numel();
    int64_t num_elements = elem_cont.numel();

    Tensor output(std::vector<int64_t>(elem_cont.shape().begin(), elem_cont.shape().end()),
                  DType::Bool, elem_cont.device());

    int block = 256;
    int grid_sz = std::min(static_cast<int>((num_elements + block - 1) / block), 65535);

    auto exec_policy = thrust::cuda::par.on(stream);

    switch (elem_cont.dtype()) {
        case DType::Float32: {
            thrust::sort(exec_policy, thrust::device_pointer_cast(test_sorted.data<float>()),
                         thrust::device_pointer_cast(test_sorted.data<float>() + num_test));
            isin_kernel_impl<float><<<grid_sz, block, 0, stream>>>(
                elem_cont.data<float>(), num_elements, test_sorted.data<float>(), num_test,
                reinterpret_cast<bool*>(output.data_ptr()));
            break;
        }
        case DType::Float64: {
            thrust::sort(exec_policy, thrust::device_pointer_cast(test_sorted.data<double>()),
                         thrust::device_pointer_cast(test_sorted.data<double>() + num_test));
            isin_kernel_impl<double><<<grid_sz, block, 0, stream>>>(
                elem_cont.data<double>(), num_elements, test_sorted.data<double>(), num_test,
                reinterpret_cast<bool*>(output.data_ptr()));
            break;
        }
        case DType::Int32: {
            thrust::sort(exec_policy, thrust::device_pointer_cast(test_sorted.data<int32_t>()),
                         thrust::device_pointer_cast(test_sorted.data<int32_t>() + num_test));
            isin_kernel_impl<int32_t><<<grid_sz, block, 0, stream>>>(
                elem_cont.data<int32_t>(), num_elements, test_sorted.data<int32_t>(), num_test,
                reinterpret_cast<bool*>(output.data_ptr()));
            break;
        }
        case DType::Int64: {
            thrust::sort(exec_policy, thrust::device_pointer_cast(test_sorted.data<int64_t>()),
                         thrust::device_pointer_cast(test_sorted.data<int64_t>() + num_test));
            isin_kernel_impl<int64_t><<<grid_sz, block, 0, stream>>>(
                elem_cont.data<int64_t>(), num_elements, test_sorted.data<int64_t>(), num_test,
                reinterpret_cast<bool*>(output.data_ptr()));
            break;
        }
        default:
            throw std::runtime_error("isin CUDA: unsupported dtype");
    }
    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return output;
}

// ============================================================================
// Kthvalue kernel — k-th smallest value along a dimension using partial sort
// ============================================================================

template<typename T>
__global__ void kthvalue_kernel_impl(
    const T* __restrict__ input, T* __restrict__ values, int64_t* __restrict__ indices,
    int64_t dim_size, int64_t inner_size, int64_t k, int64_t total_slices,
    T* __restrict__ workspace)
{
    int64_t slice_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slice_idx >= total_slices) return;

    int64_t outer = slice_idx / inner_size;
    int64_t inner = slice_idx % inner_size;

    // Copy slice to workspace
    T* ws = workspace + slice_idx * dim_size;
    for (int64_t i = 0; i < dim_size; ++i) {
        ws[i] = input[outer * dim_size * inner_size + i * inner_size + inner];
    }

    // Partial insertion sort to find k-th smallest
    // For small dim_size this is efficient; for large dims a selection algorithm would be better
    for (int64_t i = 0; i < k; ++i) {
        int64_t min_idx = i;
        T min_val = ws[i];
        for (int64_t j = i + 1; j < dim_size; ++j) {
            if (ws[j] < min_val) {
                min_val = ws[j];
                min_idx = j;
            }
        }
        if (min_idx != i) {
            ws[min_idx] = ws[i];
            ws[i] = min_val;
        }
    }

    T kth_val = ws[k - 1];
    values[slice_idx] = kth_val;

    // Find original index of k-th value
    for (int64_t i = 0; i < dim_size; ++i) {
        T orig = input[outer * dim_size * inner_size + i * inner_size + inner];
        if (orig == kth_val) {
            indices[slice_idx] = i;
            break;
        }
    }
}

auto kthvalue_kernel(const Tensor& input, int64_t k, int64_t dim, bool keepdim,
                     cudaStream_t stream) -> std::pair<Tensor, Tensor>
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = input_cont.ndim();
    const int64_t dim_size = shape[dim];
    const auto dtype = input_cont.dtype();
    const auto device = input_cont.device();

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    int64_t total_slices = outer_size * inner_size;

    // Output shape: replace dim with size 1 (or remove it)
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor values(out_shape, dtype, device);
    Tensor indices_out(out_shape, DType::Int64, device);

    int block = 256;
    int grid = std::min(static_cast<int>((total_slices + block - 1) / block), 65535);

    // Allocate workspace for partial sort
    size_t ws_bytes = total_slices * dim_size;

    switch (dtype) {
        case DType::Float32: {
            backend::CachedMemoryGuard ws_guard(ws_bytes * sizeof(float));
            kthvalue_kernel_impl<float><<<grid, block, 0, stream>>>(
                input_cont.data<float>(), values.data<float>(), indices_out.data<int64_t>(),
                dim_size, inner_size, k, total_slices, static_cast<float*>(ws_guard.get()));
            break;
        }
        case DType::Float64: {
            backend::CachedMemoryGuard ws_guard(ws_bytes * sizeof(double));
            kthvalue_kernel_impl<double><<<grid, block, 0, stream>>>(
                input_cont.data<double>(), values.data<double>(), indices_out.data<int64_t>(),
                dim_size, inner_size, k, total_slices, static_cast<double*>(ws_guard.get()));
            break;
        }
        case DType::Int32: {
            backend::CachedMemoryGuard ws_guard(ws_bytes * sizeof(int32_t));
            kthvalue_kernel_impl<int32_t><<<grid, block, 0, stream>>>(
                input_cont.data<int32_t>(), values.data<int32_t>(), indices_out.data<int64_t>(),
                dim_size, inner_size, k, total_slices, static_cast<int32_t*>(ws_guard.get()));
            break;
        }
        case DType::Int64: {
            backend::CachedMemoryGuard ws_guard(ws_bytes * sizeof(int64_t));
            kthvalue_kernel_impl<int64_t><<<grid, block, 0, stream>>>(
                input_cont.data<int64_t>(), values.data<int64_t>(), indices_out.data<int64_t>(),
                dim_size, inner_size, k, total_slices, static_cast<int64_t*>(ws_guard.get()));
            break;
        }
        default:
            throw std::runtime_error("kthvalue CUDA: unsupported dtype");
    }
    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return {values, indices_out};
}

// ============================================================================
// Quantile kernel — interpolated quantile along dim (sort + linear interp)
// ============================================================================

template<typename T>
__global__ void quantile_kernel_impl(
    const T* __restrict__ sorted_input, T* __restrict__ output,
    double q, int64_t dim_size, int64_t inner_size, int64_t total_slices)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_slices) return;

    // Linear interpolation index
    double pos = q * (dim_size - 1);
    int64_t lo = static_cast<int64_t>(pos);
    int64_t hi = lo + 1;
    if (hi >= dim_size) hi = dim_size - 1;
    double frac = pos - lo;

    T lo_val = sorted_input[idx * dim_size + lo];
    T hi_val = sorted_input[idx * dim_size + hi];
    output[idx] = static_cast<T>(static_cast<double>(lo_val) * (1.0 - frac) +
                                  static_cast<double>(hi_val) * frac);
}

auto quantile_kernel(const Tensor& input, double q, int64_t dim, bool keepdim,
                     cudaStream_t stream) -> Tensor
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = input_cont.ndim();
    const int64_t dim_size = shape[dim];
    const auto dtype = input_cont.dtype();
    const auto device = input_cont.device();

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];
    int64_t total_slices = outer_size * inner_size;

    // Transpose input so dim is last, reshape to (total_slices, dim_size), sort each row
    // For simplicity, copy slices to contiguous workspace, sort, then interpolate
    Tensor workspace({total_slices, dim_size}, dtype, device);

    int block = 256;
    int grid_extract = std::min(static_cast<int>((dim_size + block - 1) / block), 1024);

    auto exec_policy = thrust::cuda::par.on(stream);

    auto launch = [&]<typename T>() {
        // Extract slices into workspace
        for (int64_t s = 0; s < total_slices; ++s) {
            int64_t outer = s / inner_size;
            int64_t inner = s % inner_size;
            extract_strided_slice<T><<<grid_extract, block, 0, stream>>>(
                input_cont.data<T>(), workspace.data<T>() + s * dim_size,
                dim_size, inner_size, outer, inner);
        }
        // Sort each slice
        for (int64_t s = 0; s < total_slices; ++s) {
            T* slice = workspace.data<T>() + s * dim_size;
            thrust::sort(exec_policy, thrust::device_pointer_cast(slice),
                         thrust::device_pointer_cast(slice + dim_size));
        }
    };

    switch (dtype) {
        case DType::Float32: launch.template operator()<float>(); break;
        case DType::Float64: launch.template operator()<double>(); break;
        default: throw std::runtime_error("quantile CUDA: unsupported dtype (need float)");
    }

    // Output shape
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor output(out_shape, dtype, device);
    int grid_q = std::min(static_cast<int>((total_slices + block - 1) / block), 65535);

    switch (dtype) {
        case DType::Float32:
            quantile_kernel_impl<float><<<grid_q, block, 0, stream>>>(
                workspace.data<float>(), output.data<float>(), q, dim_size, inner_size, total_slices);
            break;
        case DType::Float64:
            quantile_kernel_impl<double><<<grid_q, block, 0, stream>>>(
                workspace.data<double>(), output.data<double>(), q, dim_size, inner_size, total_slices);
            break;
        default: break;
    }
    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return output;
}

// ============================================================================
// Nanquantile kernel — NaN-ignoring quantile (filter NaN, sort, interpolate)
// ============================================================================

template<typename T>
__global__ void nanquantile_kernel_impl(
    const T* __restrict__ input, T* __restrict__ output,
    double q, int64_t dim_size, int64_t inner_size, int64_t total_slices,
    T* __restrict__ workspace, int64_t* __restrict__ valid_counts)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_slices) return;

    int64_t outer = idx / inner_size;
    int64_t inner = idx % inner_size;

    // Collect non-NaN values
    T* ws = workspace + idx * dim_size;
    int64_t count = 0;
    for (int64_t i = 0; i < dim_size; ++i) {
        T val = input[outer * dim_size * inner_size + i * inner_size + inner];
        if (!::isnan(float(val))) {
            ws[count++] = val;
        }
    }
    valid_counts[idx] = count;

    if (count == 0) {
        output[idx] = static_cast<T>(NAN);
        return;
    }

    // Simple insertion sort for the non-NaN values
    for (int64_t i = 1; i < count; ++i) {
        T key = ws[i];
        int64_t j = i - 1;
        while (j >= 0 && ws[j] > key) {
            ws[j + 1] = ws[j];
            --j;
        }
        ws[j + 1] = key;
    }

    double pos = q * (count - 1);
    int64_t lo = static_cast<int64_t>(pos);
    int64_t hi = lo + 1;
    if (hi >= count) hi = count - 1;
    double frac = pos - lo;
    output[idx] = static_cast<T>(static_cast<double>(ws[lo]) * (1.0 - frac) +
                                  static_cast<double>(ws[hi]) * frac);
}

auto nanquantile_kernel(const Tensor& input, double q, int64_t dim, bool keepdim,
                        cudaStream_t stream) -> Tensor
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = input_cont.ndim();
    const int64_t dim_size = shape[dim];
    const auto dtype = input_cont.dtype();
    const auto device = input_cont.device();

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];
    int64_t total_slices = outer_size * inner_size;

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor output(out_shape, dtype, device);
    int block = 256;
    int grid = std::min(static_cast<int>((total_slices + block - 1) / block), 65535);

    switch (dtype) {
        case DType::Float32: {
            backend::CachedMemoryGuard ws_guard(total_slices * dim_size * sizeof(float));
            backend::CachedMemoryGuard vc_guard(total_slices * sizeof(int64_t));
            nanquantile_kernel_impl<float><<<grid, block, 0, stream>>>(
                input_cont.data<float>(), output.data<float>(), q, dim_size, inner_size,
                total_slices, static_cast<float*>(ws_guard.get()),
                static_cast<int64_t*>(vc_guard.get()));
            break;
        }
        case DType::Float64: {
            backend::CachedMemoryGuard ws_guard(total_slices * dim_size * sizeof(double));
            backend::CachedMemoryGuard vc_guard(total_slices * sizeof(int64_t));
            nanquantile_kernel_impl<double><<<grid, block, 0, stream>>>(
                input_cont.data<double>(), output.data<double>(), q, dim_size, inner_size,
                total_slices, static_cast<double*>(ws_guard.get()),
                static_cast<int64_t*>(vc_guard.get()));
            break;
        }
        default:
            throw std::runtime_error("nanquantile CUDA: unsupported dtype");
    }
    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return output;
}

// ============================================================================
// Nanmedian kernel — NaN-ignoring median
// ============================================================================

auto nanmedian_kernel(const Tensor& input, int64_t dim, bool keepdim,
                      cudaStream_t stream) -> Tensor
{
    // nanmedian is nanquantile with q=0.5
    return nanquantile_kernel(input, 0.5, dim, keepdim, stream);
}

// ============================================================================
// Histc kernel — fixed-bin histogram using atomicAdd
// ============================================================================

__global__ void histc_kernel_f32(const float* __restrict__ input, float* __restrict__ output,
                                  int64_t n, int64_t bins, float min_val, float max_val)
{
    float bin_width = (max_val - min_val) / static_cast<float>(bins);
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += blockDim.x * gridDim.x) {
        float val = input[idx];
        if (val >= min_val && val <= max_val) {
            int64_t bin = static_cast<int64_t>((val - min_val) / bin_width);
            if (bin >= bins) bin = bins - 1;
            atomicAdd(&output[bin], 1.0f);
        }
    }
}

auto histc_kernel(const Tensor& input, int64_t bins, double min_val, double max_val,
                  cudaStream_t stream) -> Tensor
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    int64_t n = input_cont.numel();
    const auto dtype = input_cont.dtype();
    const auto device = input_cont.device();

    // Auto-detect range if min_val >= max_val
    if (min_val >= max_val) {
        // Use thrust to find min/max
        if (dtype == DType::Float32) {
            auto begin = thrust::device_pointer_cast(input_cont.data<float>());
            auto end = begin + n;
            auto minmax = thrust::minmax_element(thrust::cuda::par.on(stream), begin, end);
            float h_min, h_max;
            cudaMemcpyAsync(&h_min, minmax.first.get(), sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(&h_max, minmax.second.get(), sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            min_val = h_min;
            max_val = h_max;
        } else if (dtype == DType::Float64) {
            auto begin = thrust::device_pointer_cast(input_cont.data<double>());
            auto end = begin + n;
            auto minmax = thrust::minmax_element(thrust::cuda::par.on(stream), begin, end);
            double h_min, h_max;
            cudaMemcpyAsync(&h_min, minmax.first.get(), sizeof(double), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(&h_max, minmax.second.get(), sizeof(double), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            min_val = h_min;
            max_val = h_max;
        }
    }

    Tensor output({bins}, dtype, device);
    cudaMemsetAsync(output.data_ptr(), 0, bins * output.element_size(), stream);

    int block = 256;
    int grid = std::min(static_cast<int>((n + block - 1) / block), 65535);

    switch (dtype) {
        case DType::Float32:
            histc_kernel_f32<<<grid, block, 0, stream>>>(
                input_cont.data<float>(), output.data<float>(), n, bins,
                static_cast<float>(min_val), static_cast<float>(max_val));
            break;
        case DType::Float64:
            {
                // Convert double input to float, run float32 histogram, then convert output
                Tensor f32_input = input_cont.to(DType::Float32);
                Tensor f32_output({bins}, DType::Float32, device);
                cudaMemsetAsync(f32_output.data_ptr(), 0, bins * sizeof(float), stream);
                histc_kernel_f32<<<grid, block, 0, stream>>>(
                    f32_input.data<float>(), f32_output.data<float>(), n, bins,
                    static_cast<float>(min_val), static_cast<float>(max_val));
                output = f32_output.to(DType::Float64);
            }
            break;
        default:
            throw std::runtime_error("histc CUDA: unsupported dtype (only float32/float64)");
    }
    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return output;
}

// ============================================================================
// UniqueConsecutive kernel — deduplicate consecutive equal elements
// ============================================================================

template<typename T>
__global__ void unique_consecutive_mask_kernel(const T* __restrict__ input,
                                                int32_t* __restrict__ mask,
                                                int64_t n)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += blockDim.x * gridDim.x) {
        if (idx == 0) {
            mask[idx] = 1;  // First element is always unique
        } else {
            mask[idx] = (input[idx] != input[idx - 1]) ? 1 : 0;
        }
    }
}

template<typename T>
__global__ void unique_consecutive_gather_kernel(const T* __restrict__ input,
                                                  const int32_t* __restrict__ prefix_sum,
                                                  const int32_t* __restrict__ mask,
                                                  T* __restrict__ output,
                                                  int64_t* __restrict__ inverse,
                                                  int64_t n)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += blockDim.x * gridDim.x) {
        int32_t out_idx = prefix_sum[idx] - 1;  // prefix_sum is 1-based
        if (mask[idx]) {
            output[out_idx] = input[idx];
        }
        inverse[idx] = out_idx;
    }
}

template<typename T>
__global__ void unique_consecutive_counts_kernel(const int64_t* __restrict__ inverse,
                                                  int64_t* __restrict__ counts,
                                                  int64_t n, int64_t num_unique)
{
    // Zero counts
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_unique;
         idx += blockDim.x * gridDim.x) {
        counts[idx] = 0;
    }
    __syncthreads();
    // Count occurrences
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += blockDim.x * gridDim.x) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&counts[inverse[idx]]),
                  static_cast<unsigned long long>(1));
    }
}

auto unique_consecutive_kernel(const Tensor& input, bool return_inverse,
                                cudaStream_t stream)
    -> std::tuple<Tensor, Tensor, Tensor>
{
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    int64_t n = input_cont.numel();
    const auto dtype = input_cont.dtype();
    const auto device = input_cont.device();

    if (n == 0) {
        return {Tensor({0}, dtype, device), Tensor({0}, DType::Int64, device),
                Tensor({0}, DType::Int64, device)};
    }

    // Step 1: compute mask (1 where value differs from predecessor)
    Tensor mask({n}, DType::Int32, device);
    Tensor prefix({n}, DType::Int32, device);
    Tensor inverse_out({n}, DType::Int64, device);

    int block = 256;
    int grid = std::min(static_cast<int>((n + block - 1) / block), 65535);

    auto exec_policy = thrust::cuda::par.on(stream);

    auto launch = [&]<typename T>() {
        unique_consecutive_mask_kernel<T><<<grid, block, 0, stream>>>(
            input_cont.data<T>(), mask.data<int32_t>(), n);
        TENZOR_CUDA_POST_LAUNCH_CHECK();

        // Step 2: inclusive prefix sum on mask
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, mask.data<int32_t>(),
                                       prefix.data<int32_t>(), static_cast<int>(n), stream);
        backend::CachedMemoryGuard temp_guard(temp_bytes);
        d_temp = temp_guard.get();
        cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, mask.data<int32_t>(),
                                       prefix.data<int32_t>(), static_cast<int>(n), stream);

        // Get total unique count from last element of prefix sum
        int32_t num_unique_h;
        cudaMemcpyAsync(&num_unique_h, prefix.data<int32_t>() + n - 1,
                        sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        int64_t num_unique = num_unique_h;

        // Step 3: gather unique elements and compute inverse indices
        Tensor unique_out({num_unique}, dtype, device);
        unique_consecutive_gather_kernel<T><<<grid, block, 0, stream>>>(
            input_cont.data<T>(), prefix.data<int32_t>(), mask.data<int32_t>(),
            unique_out.data<T>(), inverse_out.data<int64_t>(), n);
        TENZOR_CUDA_POST_LAUNCH_CHECK();

        // Step 4: compute counts
        Tensor counts({num_unique}, DType::Int64, device);
        cudaMemsetAsync(counts.data<int64_t>(), 0, num_unique * sizeof(int64_t), stream);
        unique_consecutive_counts_kernel<T><<<grid, block, 0, stream>>>(
            inverse_out.data<int64_t>(), counts.data<int64_t>(), n, num_unique);
        TENZOR_CUDA_POST_LAUNCH_CHECK();

        return std::make_tuple(unique_out, inverse_out, counts);
    };

    switch (dtype) {
        case DType::Float32: return launch.template operator()<float>();
        case DType::Float64: return launch.template operator()<double>();
        case DType::Int32:   return launch.template operator()<int32_t>();
        case DType::Int64:   return launch.template operator()<int64_t>();
        default: throw std::runtime_error("unique_consecutive CUDA: unsupported dtype");
    }
}

// ============================================================================
// SegmentReduce — reduce over segments defined by offsets
// Modes: 0=sum, 1=mean, 2=max, 3=min, 4=prod
// One warp per segment, using warp-level shuffle reductions.
// ============================================================================

template<typename T>
__global__ void segment_reduce_kernel_cuda(
    const T* __restrict__ data,
    const int64_t* __restrict__ offsets,
    T* __restrict__ output,
    int64_t num_segments,
    int64_t outer_size,
    int64_t axis_size,
    int64_t inner_size,
    int mode)
{
    // Each warp handles one (outer, segment, inner) triple
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    int64_t total_work = outer_size * num_segments * inner_size;
    if (warp_id >= total_work) return;

    int64_t inner = warp_id % inner_size;
    int64_t seg = (warp_id / inner_size) % num_segments;
    int64_t outer = warp_id / (inner_size * num_segments);

    int64_t seg_start = offsets[seg];
    int64_t seg_end = offsets[seg + 1];
    int64_t seg_len = seg_end - seg_start;

    // Identity value
    T identity;
    if (mode == 0 || mode == 1) identity = T(0);       // sum/mean
    else if (mode == 4) identity = T(1);                 // prod
    else if (mode == 2) identity = T(-1e38);             // max
    else identity = T(1e38);                             // min

    // Each lane processes elements with stride 32
    T acc = identity;
    for (int64_t i = lane; i < seg_len; i += 32) {
        int64_t d = seg_start + i;
        int64_t in_idx = (outer * axis_size + d) * inner_size + inner;
        T val = data[in_idx];
        if (mode == 0 || mode == 1) acc += val;
        else if (mode == 4) acc *= val;
        else if (mode == 2) acc = acc > val ? acc : val;
        else acc = acc < val ? acc : val;
    }

    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        T other = __shfl_down_sync(0xffffffff, acc, offset);
        if (mode == 0 || mode == 1) acc += other;
        else if (mode == 4) acc *= other;
        else if (mode == 2) acc = acc > other ? acc : other;
        else acc = acc < other ? acc : other;
    }

    if (lane == 0) {
        if (mode == 1 && seg_len > 0) {
            acc /= static_cast<T>(seg_len);
        }
        // Handle empty segments
        if (seg_len == 0) {
            acc = (mode == 0 || mode == 1) ? T(0) : identity;
        }
        int64_t out_idx = (outer * num_segments + seg) * inner_size + inner;
        output[out_idx] = acc;
    }
}

auto segment_reduce_kernel(const Tensor& data, const Tensor& offsets,
                           const std::string& reduce, int64_t axis,
                           cudaStream_t stream) -> Tensor {
    Tensor cont = data.is_contiguous() ? data : data.contiguous();
    Tensor offs = offsets.is_contiguous() ? offsets : offsets.contiguous();

    int64_t ndim = cont.ndim();
    if (axis < 0) axis += ndim;

    const auto& shape = cont.shape();
    int64_t axis_size = shape[axis];
    int64_t num_segments = offs.numel() - 1;

    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = axis + 1; i < ndim; ++i) inner_size *= shape[i];

    // Build output shape
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        out_shape.push_back(i == axis ? num_segments : shape[i]);
    }

    int mode = 0;
    if (reduce == "sum") mode = 0;
    else if (reduce == "mean") mode = 1;
    else if (reduce == "max") mode = 2;
    else if (reduce == "min") mode = 3;
    else if (reduce == "prod") mode = 4;

    auto dtype = cont.dtype();
    auto device = cont.device();
    Tensor output(out_shape, dtype, device);

    int64_t total_warps = outer_size * num_segments * inner_size;
    int64_t total_threads = total_warps * 32;
    int block = 256;
    int grid = static_cast<int>((total_threads + block - 1) / block);
    grid = std::min(grid, 65535);

    const int64_t* offsets_ptr = offs.data<int64_t>();

    switch (dtype) {
        case DType::Float32:
            segment_reduce_kernel_cuda<float><<<grid, block, 0, stream>>>(
                cont.data<float>(), offsets_ptr, output.data<float>(),
                num_segments, outer_size, axis_size, inner_size, mode);
            break;
        case DType::Float64:
            segment_reduce_kernel_cuda<double><<<grid, block, 0, stream>>>(
                cont.data<double>(), offsets_ptr, output.data<double>(),
                num_segments, outer_size, axis_size, inner_size, mode);
            break;
        case DType::Int32:
            segment_reduce_kernel_cuda<int32_t><<<grid, block, 0, stream>>>(
                cont.data<int32_t>(), offsets_ptr, output.data<int32_t>(),
                num_segments, outer_size, axis_size, inner_size, mode);
            break;
        case DType::Int64:
            segment_reduce_kernel_cuda<int64_t><<<grid, block, 0, stream>>>(
                cont.data<int64_t>(), offsets_ptr, output.data<int64_t>(),
                num_segments, outer_size, axis_size, inner_size, mode);
            break;
        default: {
            // Float16/BFloat16: upcast to Float32
            DType orig = dtype;
            Tensor cont_f32 = cont.to(DType::Float32);
            Tensor output_f32(out_shape, DType::Float32, device);
            segment_reduce_kernel_cuda<float><<<grid, block, 0, stream>>>(
                cont_f32.data<float>(), offsets_ptr, output_f32.data<float>(),
                num_segments, outer_size, axis_size, inner_size, mode);
            output = output_f32.to(orig);
        }
    }
    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return output;
}

// ============================================================================
// Trapezoid integration kernel
// ============================================================================

__global__ void trapezoid_kernel_impl(
    const float* __restrict__ y, const float* __restrict__ x,
    float* __restrict__ output, uint32_t outer, uint32_t inner,
    uint32_t n, float dx, bool has_x) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;

    uint32_t o = idx / inner;
    uint32_t i_inner = idx % inner;

    float sum = 0.0f;
    for (uint32_t k = 0; k < n - 1; k++) {
        uint32_t idx_k  = (o * n + k) * inner + i_inner;
        uint32_t idx_k1 = (o * n + k + 1) * inner + i_inner;
        float y_k = y[idx_k], y_k1 = y[idx_k1];
        float h = has_x ? (x[idx_k1] - x[idx_k]) : dx;
        sum += 0.5f * (y_k + y_k1) * h;
    }
    output[idx] = sum;
}

auto trapezoid_kernel(const Tensor& y, int64_t dim, double dx,
                       const Tensor* x_ptr, cudaStream_t stream) -> Tensor {
    Tensor yf = (y.dtype() == DType::Float32) ? y.contiguous() : y.contiguous().to(DType::Float32);
    auto shape = yf.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    int64_t n = shape[dim];

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim; d++) {
        if (d != dim) out_shape.push_back(shape[d]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor result(out_shape, DType::Float32, y.device());
    int64_t total = outer * inner;
    if (n < 2 || total == 0) return result;

    const float* x_data = nullptr;
    Tensor xf;
    if (x_ptr) {
        xf = (x_ptr->dtype() == DType::Float32) ? x_ptr->contiguous() : x_ptr->contiguous().to(DType::Float32);
        x_data = xf.data<float>();
    }

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    trapezoid_kernel_impl<<<blocks, threads, 0, stream>>>(
        yf.data<float>(), x_data, result.data<float>(),
        static_cast<uint32_t>(outer), static_cast<uint32_t>(inner),
        static_cast<uint32_t>(n), static_cast<float>(dx), x_ptr != nullptr);
    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return result;
}

// ============================================================================
// Cumulative trapezoid integration kernel
// ============================================================================

__global__ void cumulative_trapezoid_kernel_impl(
    const float* __restrict__ y, const float* __restrict__ x,
    float* __restrict__ output, uint32_t outer, uint32_t inner,
    uint32_t n, float dx, bool has_x) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;

    uint32_t o = idx / inner;
    uint32_t i_inner = idx % inner;

    float cumsum = 0.0f;
    for (uint32_t k = 0; k < n - 1; k++) {
        uint32_t idx_k  = (o * n + k) * inner + i_inner;
        uint32_t idx_k1 = (o * n + k + 1) * inner + i_inner;
        float y_k = y[idx_k], y_k1 = y[idx_k1];
        float h = has_x ? (x[idx_k1] - x[idx_k]) : dx;
        cumsum += 0.5f * (y_k + y_k1) * h;
        output[(o * (n - 1) + k) * inner + i_inner] = cumsum;
    }
}

auto cumulative_trapezoid_kernel(const Tensor& y, int64_t dim, double dx,
                                  const Tensor* x_ptr, cudaStream_t stream) -> Tensor {
    Tensor yf = (y.dtype() == DType::Float32) ? y.contiguous() : y.contiguous().to(DType::Float32);
    auto shape = yf.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    int64_t n = shape[dim];

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = (n < 2) ? 0 : n - 1;
    Tensor result(out_shape, DType::Float32, y.device());

    int64_t total = outer * inner;
    if (n < 2 || total == 0) return result;

    const float* x_data = nullptr;
    Tensor xf;
    if (x_ptr) {
        xf = (x_ptr->dtype() == DType::Float32) ? x_ptr->contiguous() : x_ptr->contiguous().to(DType::Float32);
        x_data = xf.data<float>();
    }

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cumulative_trapezoid_kernel_impl<<<blocks, threads, 0, stream>>>(
        yf.data<float>(), x_data, result.data<float>(),
        static_cast<uint32_t>(outer), static_cast<uint32_t>(inner),
        static_cast<uint32_t>(n), static_cast<float>(dx), x_ptr != nullptr);
    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return result;
}

// ============================================================================
// Numerical gradient kernel (central differences)
// ============================================================================

__global__ void gradient_kernel_impl(
    const float* __restrict__ input, float* __restrict__ output,
    uint32_t outer, uint32_t inner, uint32_t n, float spacing) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer * inner) return;

    uint32_t o = idx / inner;
    uint32_t i_inner = idx % inner;

    auto at = [&](uint32_t k) -> float {
        return input[(o * n + k) * inner + i_inner];
    };
    auto out_at = [&](uint32_t k) -> float& {
        return output[(o * n + k) * inner + i_inner];
    };

    // Forward difference at left boundary
    out_at(0) = (at(1) - at(0)) / spacing;

    // Central differences for interior
    for (uint32_t k = 1; k < n - 1; k++) {
        out_at(k) = (at(k + 1) - at(k - 1)) / (2.0f * spacing);
    }

    // Backward difference at right boundary
    out_at(n - 1) = (at(n - 1) - at(n - 2)) / spacing;
}

auto gradient_kernel(const Tensor& input, int64_t dim, double spacing,
                      cudaStream_t stream) -> Tensor {
    Tensor inf = (input.dtype() == DType::Float32) ? input.contiguous() : input.contiguous().to(DType::Float32);
    auto shape = inf.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    int64_t n = shape[dim];

    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= shape[d];
    for (int64_t d = dim + 1; d < ndim; d++) inner *= shape[d];

    Tensor result(std::vector<int64_t>(shape.begin(), shape.end()), DType::Float32, input.device());
    int64_t total = outer * inner;
    if (n < 2 || total == 0) return result;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    gradient_kernel_impl<<<blocks, threads, 0, stream>>>(
        inf.data<float>(), result.data<float>(),
        static_cast<uint32_t>(outer), static_cast<uint32_t>(inner),
        static_cast<uint32_t>(n), static_cast<float>(spacing));
    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return result;
}

// ============================================================================
// Pairwise distance kernel
// ============================================================================

__global__ void pairwise_distance_kernel_impl(
    const float* __restrict__ x1, const float* __restrict__ x2,
    float* __restrict__ output, uint32_t N, uint32_t D, float p) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sum = 0.0f;
    if (p == 2.0f) {
        for (uint32_t j = 0; j < D; j++) {
            float diff = x1[i * D + j] - x2[i * D + j];
            sum += diff * diff;
        }
        output[i] = sqrtf(sum);
    } else if (p == 1.0f) {
        for (uint32_t j = 0; j < D; j++) {
            sum += fabsf(x1[i * D + j] - x2[i * D + j]);
        }
        output[i] = sum;
    } else {
        for (uint32_t j = 0; j < D; j++) {
            sum += powf(fabsf(x1[i * D + j] - x2[i * D + j]), p);
        }
        output[i] = powf(sum, 1.0f / p);
    }
}

auto pairwise_distance_kernel(const Tensor& x1, const Tensor& x2, double p,
                               cudaStream_t stream) -> Tensor {
    Tensor a = (x1.dtype() == DType::Float32) ? x1.contiguous() : x1.contiguous().to(DType::Float32);
    Tensor b = (x2.dtype() == DType::Float32) ? x2.contiguous() : x2.contiguous().to(DType::Float32);

    int64_t N = a.shape()[0], D = a.shape()[1];
    Tensor result({N}, DType::Float32, x1.device());
    if (N == 0) return result;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    pairwise_distance_kernel_impl<<<blocks, threads, 0, stream>>>(
        a.data<float>(), b.data<float>(), result.data<float>(),
        static_cast<uint32_t>(N), static_cast<uint32_t>(D), static_cast<float>(p));
    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return result;
}

// ============================================================================
// Pdist kernel (all-pairs pairwise distances)
// ============================================================================

__global__ void pdist_kernel_impl(
    const float* __restrict__ data, float* __restrict__ output,
    uint32_t N, uint32_t D, uint32_t num_pairs, float p) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    // Map flat index to (i, j) pair where i < j
    uint32_t i = 0, offset = 0;
    while (offset + (N - 1 - i) <= idx) {
        offset += (N - 1 - i);
        i++;
    }
    uint32_t j = idx - offset + i + 1;

    float sum = 0.0f;
    if (p == 2.0f) {
        for (uint32_t d = 0; d < D; d++) {
            float diff = data[i * D + d] - data[j * D + d];
            sum += diff * diff;
        }
        output[idx] = sqrtf(sum);
    } else if (p == 1.0f) {
        for (uint32_t d = 0; d < D; d++) {
            sum += fabsf(data[i * D + d] - data[j * D + d]);
        }
        output[idx] = sum;
    } else {
        for (uint32_t d = 0; d < D; d++) {
            sum += powf(fabsf(data[i * D + d] - data[j * D + d]), p);
        }
        output[idx] = powf(sum, 1.0f / p);
    }
}

auto pdist_kernel(const Tensor& input, double p, cudaStream_t stream) -> Tensor {
    Tensor inf = (input.dtype() == DType::Float32) ? input.contiguous() : input.contiguous().to(DType::Float32);
    int64_t N = inf.shape()[0], D = inf.shape()[1];
    int64_t num_pairs = N * (N - 1) / 2;

    Tensor result({num_pairs}, DType::Float32, input.device());
    if (num_pairs == 0) return result;

    int threads = 256;
    int blocks = (num_pairs + threads - 1) / threads;
    pdist_kernel_impl<<<blocks, threads, 0, stream>>>(
        inf.data<float>(), result.data<float>(),
        static_cast<uint32_t>(N), static_cast<uint32_t>(D),
        static_cast<uint32_t>(num_pairs), static_cast<float>(p));
    TENZOR_CUDA_POST_LAUNCH_CHECK();
    return result;
}

// ============================================================================
// Histogramdd (multi-dimensional histogram) kernel
// ============================================================================

// Device kernel: each thread processes one sample, computing a flat bin index
// and atomically incrementing the counts tensor.
// params_buf layout per dimension d: [min_d, step_d, bins_d, stride_d] (4 doubles each)
template <typename T>
__global__ void histogramdd_kernel_impl(const T* input, int64_t* counts,
                                         const double* params_buf,
                                         int64_t N, int64_t D) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    const T* sample = input + tid * D;
    int64_t flat = 0;
    for (int64_t d = 0; d < D; ++d) {
        int64_t base = d * 4;
        double fmin   = params_buf[base + 0];
        double step   = params_buf[base + 1];
        int64_t nbins = static_cast<int64_t>(params_buf[base + 2]);
        int64_t str   = static_cast<int64_t>(params_buf[base + 3]);

        double v = static_cast<double>(sample[d]);
        double upper = fmin + step * static_cast<double>(nbins);
        if (v < fmin || v > upper) return; // out of range — skip sample

        int64_t b = static_cast<int64_t>((v - fmin) / step);
        if (b >= nbins) b = nbins - 1;
        if (b < 0) b = 0;
        flat += b * str;
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(&counts[flat]),
              static_cast<unsigned long long>(1));
}

// Device kernel: fill edge tensor for one dimension
template <typename T>
__global__ void histogramdd_fill_edges_kernel(T* edges, T fmin, T step, int64_t num_edges) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_edges) return;
    edges[i] = fmin + static_cast<T>(i) * step;
}

// Device kernel: density normalisation — convert int64 counts to float
template <typename T>
__global__ void histogramdd_density_kernel(const int64_t* counts, T* output,
                                            int64_t total_bins, double norm) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_bins) return;
    output[i] = static_cast<T>(static_cast<double>(counts[i]) / norm);
}

// Column min/max reduction: one block per dimension, walks all N rows.
template <typename T>
__global__ void histogramdd_col_minmax_kernel(const T* input, double* min_max_out,
                                               int64_t N, int64_t D) {
    extern __shared__ char smem_raw[];
    double* smin = reinterpret_cast<double*>(smem_raw);
    double* smax = smin + blockDim.x;

    int64_t d = blockIdx.x; // one block per dimension
    int tid = threadIdx.x;

    double lmin = 1e308;
    double lmax = -1e308;
    for (int64_t i = tid; i < N; i += blockDim.x) {
        double v = static_cast<double>(input[i * D + d]);
        if (v < lmin) lmin = v;
        if (v > lmax) lmax = v;
    }
    smin[tid] = lmin;
    smax[tid] = lmax;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smin[tid + s] < smin[tid]) smin[tid] = smin[tid + s];
            if (smax[tid + s] > smax[tid]) smax[tid] = smax[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        min_max_out[d * 2 + 0] = smin[0];
        min_max_out[d * 2 + 1] = smax[0];
    }
}

auto histogramdd_kernel(const Tensor& input,
                        std::vector<int64_t> bins,
                        std::vector<std::pair<double,double>> ranges,
                        bool density,
                        cudaStream_t stream)
    -> std::pair<Tensor, std::vector<Tensor>> {

    if (input.dim() != 2) {
        throw std::runtime_error("histogramdd_kernel: input must be 2-D (N, D)");
    }

    const int64_t N = input.shape()[0];
    const int64_t D = input.shape()[1];

    if (static_cast<int64_t>(bins.size()) != D) {
        throw std::runtime_error("histogramdd_kernel: bins length must equal D");
    }

    const auto orig_dtype = input.dtype();
    const bool use_f64 = (orig_dtype == DType::Float64);
    const auto compute_dtype = use_f64 ? DType::Float64 : DType::Float32;
    auto inp = (input.dtype() != compute_dtype) ? input.to(compute_dtype) : input;
    inp = inp.contiguous();
    const auto& device = input.device();

    bool auto_range = ranges.empty();
    if (auto_range) {
        ranges.resize(static_cast<size_t>(D));
    }

    // Auto-detect ranges on device
    if (auto_range && N > 0) {
        // Allocate device buffer for D*(min,max) pairs
        backend::CachedMemoryGuard mm_guard(static_cast<size_t>(D) * 2 * sizeof(double));
        double* d_minmax = static_cast<double*>(mm_guard.get());

        int block_size = 256;
        size_t smem_bytes = 2 * block_size * sizeof(double);
        if (use_f64) {
            histogramdd_col_minmax_kernel<double><<<static_cast<int>(D), block_size, smem_bytes, stream>>>(
                inp.data<double>(), d_minmax, N, D);
        } else {
            histogramdd_col_minmax_kernel<float><<<static_cast<int>(D), block_size, smem_bytes, stream>>>(
                inp.data<float>(), d_minmax, N, D);
        }

        // Readback — unavoidable to compute bin parameters on host
        std::vector<double> h_minmax(static_cast<size_t>(D) * 2);
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(h_minmax.data(), d_minmax,
                                          h_minmax.size() * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream));
        TENZOR_CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int64_t d = 0; d < D; ++d) {
            double vmin = h_minmax[static_cast<size_t>(d) * 2 + 0];
            double vmax = h_minmax[static_cast<size_t>(d) * 2 + 1];
            if (vmin == vmax) { vmin -= 0.5; vmax += 0.5; }
            ranges[static_cast<size_t>(d)] = {vmin, vmax};
        }
    } else if (auto_range && N == 0) {
        for (int64_t d = 0; d < D; ++d) {
            ranges[static_cast<size_t>(d)] = {0.0, 1.0};
        }
    }

    // Build per-dimension parameters and edge tensors
    std::vector<double> dim_min(static_cast<size_t>(D));
    std::vector<double> dim_step(static_cast<size_t>(D));
    std::vector<Tensor> edges_vec;
    edges_vec.reserve(static_cast<size_t>(D));

    for (int64_t d = 0; d < D; ++d) {
        auto sd = static_cast<size_t>(d);
        int64_t nb = bins[sd];
        double fmin = ranges[sd].first;
        double fmax = ranges[sd].second;
        double step = (fmax - fmin) / static_cast<double>(nb);
        dim_min[sd] = fmin;
        dim_step[sd] = step;

        Tensor edge({nb + 1}, compute_dtype, device);
        int64_t num_edges = nb + 1;
        int threads_e = 128;
        int blocks_e = static_cast<int>((num_edges + threads_e - 1) / threads_e);
        if (use_f64) {
            histogramdd_fill_edges_kernel<double><<<blocks_e, threads_e, 0, stream>>>(
                edge.data<double>(), fmin, step, num_edges);
        } else {
            histogramdd_fill_edges_kernel<float><<<blocks_e, threads_e, 0, stream>>>(
                edge.data<float>(), static_cast<float>(fmin), static_cast<float>(step), num_edges);
        }
        edges_vec.push_back(std::move(edge));
    }

    // Compute output shape and strides (row-major)
    std::vector<int64_t> out_shape(bins.begin(), bins.end());
    std::vector<int64_t> out_strides(static_cast<size_t>(D));
    int64_t stride = 1;
    for (int64_t d = D - 1; d >= 0; --d) {
        out_strides[static_cast<size_t>(d)] = stride;
        stride *= bins[static_cast<size_t>(d)];
    }
    int64_t total_bins = stride;

    // Allocate counts (zero-initialised)
    Tensor counts(out_shape, DType::Int64, device);
    TENZOR_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr(), 0,
                                      static_cast<size_t>(total_bins) * sizeof(int64_t), stream));

    // Upload per-dimension params buffer to device: [min, step, bins, stride] * D
    std::vector<double> h_params(static_cast<size_t>(D) * 4);
    for (int64_t d = 0; d < D; ++d) {
        auto sd = static_cast<size_t>(d);
        h_params[sd * 4 + 0] = dim_min[sd];
        h_params[sd * 4 + 1] = dim_step[sd];
        h_params[sd * 4 + 2] = static_cast<double>(bins[sd]);
        h_params[sd * 4 + 3] = static_cast<double>(out_strides[sd]);
    }
    backend::CachedMemoryGuard params_guard(h_params.size() * sizeof(double));
    double* d_params = static_cast<double*>(params_guard.get());
    TENZOR_CUDA_CHECK(cudaMemcpyAsync(d_params, h_params.data(),
                                      h_params.size() * sizeof(double),
                                      cudaMemcpyHostToDevice, stream));

    // Launch histogram kernel
    if (N > 0) {
        int threads = 256;
        int blocks_n = static_cast<int>((N + threads - 1) / threads);
        if (use_f64) {
            histogramdd_kernel_impl<double><<<blocks_n, threads, 0, stream>>>(
                inp.data<double>(), counts.data<int64_t>(), d_params, N, D);
        } else {
            histogramdd_kernel_impl<float><<<blocks_n, threads, 0, stream>>>(
                inp.data<float>(), counts.data<int64_t>(), d_params, N, D);
        }
        TENZOR_CUDA_POST_LAUNCH_CHECK();
    }

    // Density normalisation
    Tensor result = counts;
    if (density && N > 0) {
        double bin_volume = 1.0;
        for (int64_t d = 0; d < D; ++d) {
            bin_volume *= dim_step[static_cast<size_t>(d)];
        }
        double norm = static_cast<double>(N) * bin_volume;

        Tensor density_out(out_shape, compute_dtype, device);
        int threads_d = 256;
        int blocks_d = static_cast<int>((total_bins + threads_d - 1) / threads_d);
        if (use_f64) {
            histogramdd_density_kernel<double><<<blocks_d, threads_d, 0, stream>>>(
                counts.data<int64_t>(), density_out.data<double>(), total_bins, norm);
        } else {
            histogramdd_density_kernel<float><<<blocks_d, threads_d, 0, stream>>>(
                counts.data<int64_t>(), density_out.data<float>(), total_bins, norm);
        }
        TENZOR_CUDA_POST_LAUNCH_CHECK();
        result = density_out;
    }

    return {std::move(result), std::move(edges_vec)};
}

} // namespace cuda
} // namespace tenzor
