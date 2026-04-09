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

auto cdist_kernel(const Tensor& x1, const Tensor& x2, double /*p*/,
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

    // Only L2 distance for now
    dim3 threads(16, 16, 1);
    dim3 blocks((R + 15) / 16, (P + 15) / 16, B);
    cdist_l2_kernel_impl<<<blocks, threads, 0, stream>>>(
        a.data<float>(), b.data<float>(), result.data<float>(), B, P, R, M);

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

} // namespace cuda
} // namespace tenzor
