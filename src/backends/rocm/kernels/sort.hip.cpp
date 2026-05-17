/**
 * @file sort.hip.cpp
 * @brief ROCm/HIP kernels for sort, topk, argsort, and unique operations.
 *
 * Ported from CUDA kernels using hipcub (HIP equivalent of CUB) and
 * thrust (HIP-compatible) for radix sort, prefix scan, and run-length encoding.
 */

#include "rocm_nan_helpers.hip.h"  // E.2: safe_f2h / safe_h2f / safe_f2bf / safe_bf2f
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"

#include <hipcub/hipcub.hpp>
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
#include <algorithm>

namespace tenzor {
namespace rocm {

// ============================================================================
// HIP Error Checking
// ============================================================================

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP error at ") + __FILE__ + ":" + \
            std::to_string(__LINE__) + " - " + hipGetErrorString(err)); \
    } \
} while(0)

#define HIP_KERNEL_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

constexpr int BLOCK_SIZE = 256;

inline int get_num_blocks(int64_t n, int block_size = BLOCK_SIZE) {
    return static_cast<int>((n + block_size - 1) / block_size);
}

// Custom multiply functor for hipcub InclusiveScan (used in cumprod — shared code)
struct MultOp {
    template<typename T>
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a * b; }
};

// Safe comparison helpers for half types
template<typename T>
__device__ __forceinline__ bool hip_gt(const T& a, const T& b) { return a > b; }
template<typename T>
__device__ __forceinline__ bool hip_lt(const T& a, const T& b) { return a < b; }
template<typename T>
__device__ __forceinline__ bool hip_eq(const T& a, const T& b) { return a == b; }

template<> __device__ __forceinline__ bool hip_gt(const __half& a, const __half& b) { return __hgt(a, b); }
template<> __device__ __forceinline__ bool hip_lt(const __half& a, const __half& b) { return __hlt(a, b); }
template<> __device__ __forceinline__ bool hip_eq(const __half& a, const __half& b) { return __heq(a, b); }

// ============================================================================
// TopK kernel using parallel block-wide selection
// ============================================================================

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

    extern __shared__ char smem[];
    T* s_topk_vals = reinterpret_cast<T*>(smem);
    size_t vals_bytes = k * sizeof(T);
    size_t aligned_vals = (vals_bytes + 7) & ~size_t(7);
    int64_t* s_topk_idx = reinterpret_cast<int64_t*>(smem + aligned_vals);
    size_t idx_bytes = k * sizeof(int64_t);
    size_t aligned_idx = (idx_bytes + 7) & ~size_t(7);

    char* cand_base = smem + aligned_vals + aligned_idx;
    T* s_cand_vals = reinterpret_cast<T*>(cand_base);
    size_t cand_vals_bytes = blockDim.x * sizeof(T);
    size_t aligned_cand_vals = (cand_vals_bytes + 7) & ~size_t(7);
    int64_t* s_cand_pos = reinterpret_cast<int64_t*>(cand_base + aligned_cand_vals);

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    for (int64_t round = 0; round < k; ++round) {
        T best_val;
        int64_t best_pos = -1;
        bool has_candidate = false;

        for (int64_t i = tid; i < dim_size; i += nthreads) {
            T val = input[in_base + i * inner_size];

            bool consumed = false;
            for (int64_t r = 0; r < round; ++r) {
                if (s_topk_idx[r] == i) {
                    consumed = true;
                    break;
                }
            }
            if (consumed) continue;

            if (!has_candidate ||
                (largest ? hip_gt(val, best_val) : hip_lt(val, best_val)) ||
                (hip_eq(val, best_val) && i < best_pos)) {
                best_val = val;
                best_pos = i;
                has_candidate = true;
            }
        }

        s_cand_vals[tid] = best_val;
        s_cand_pos[tid] = best_pos;
        __syncthreads();

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
                        (hip_gt(s_cand_vals[tid + stride], s_cand_vals[tid]) ||
                         (hip_eq(s_cand_vals[tid + stride], s_cand_vals[tid]) &&
                          s_cand_pos[tid + stride] < s_cand_pos[tid])) :
                        (hip_lt(s_cand_vals[tid + stride], s_cand_vals[tid]) ||
                         (hip_eq(s_cand_vals[tid + stride], s_cand_vals[tid]) &&
                          s_cand_pos[tid + stride] < s_cand_pos[tid]));
                }
                if (right_wins) {
                    s_cand_vals[tid] = s_cand_vals[tid + stride];
                    s_cand_pos[tid] = s_cand_pos[tid + stride];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            s_topk_vals[round] = s_cand_vals[0];
            s_topk_idx[round] = s_cand_pos[0];
        }
        __syncthreads();
    }

    // Sort the k results using parallel odd-even transposition sort
    for (int64_t phase = 0; phase < k; ++phase) {
        int64_t i = 2 * tid + (phase & 1);
        if (i + 1 < k) {
            bool should_swap = largest ?
                hip_lt(s_topk_vals[i], s_topk_vals[i + 1]) :
                hip_gt(s_topk_vals[i], s_topk_vals[i + 1]);
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
                 bool sorted, hipStream_t stream) -> std::pair<Tensor, Tensor>
{
    // BFloat16 upcast: convert to Float32, compute, convert values back
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto [values, indices] = topk_kernel(input_f32, k, dim, largest, sorted, stream);
        return {values.to(DType::BFloat16), indices};
    }

    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t ndim = input.ndim();
    const int64_t dim_size = shape[dim];
    const auto dtype = input.dtype();
    const auto device = input.device();

    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    output_shape[dim] = k;

    Tensor values(output_shape, dtype, device);
    Tensor indices(output_shape, DType::Int64, device);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    int64_t num_slices = outer_size * inner_size;
    int64_t outer_stride = dim_size * inner_size;
    int64_t k_stride = k * inner_size;

    auto launch = [&]<typename T>() {
        int block_size = 256;
        size_t topk_vals_bytes = k * sizeof(T);
        size_t aligned_topk_vals = (topk_vals_bytes + 7) & ~size_t(7);
        size_t topk_idx_bytes = k * sizeof(int64_t);
        size_t aligned_topk_idx = (topk_idx_bytes + 7) & ~size_t(7);
        size_t cand_vals_bytes = block_size * sizeof(T);
        size_t aligned_cand_vals = (cand_vals_bytes + 7) & ~size_t(7);
        size_t cand_pos_bytes = block_size * sizeof(int64_t);
        size_t smem_size = aligned_topk_vals + aligned_topk_idx +
                           aligned_cand_vals + cand_pos_bytes;
        auto* input_ptr = reinterpret_cast<const T*>(input_cont.data_ptr());
        auto* values_ptr = reinterpret_cast<T*>(values.data_ptr());
        auto* indices_ptr = reinterpret_cast<int64_t*>(indices.data_ptr());
        hipLaunchKernelGGL(topk_slice_kernel<T>,
            dim3(static_cast<unsigned>(num_slices)), dim3(block_size), smem_size, stream,
            input_ptr, values_ptr, indices_ptr,
            dim_size, k, inner_size, outer_stride, k_stride, largest);
        HIP_CHECK(hipGetLastError());
    };

    switch (dtype) {
        case DType::Float32:  launch.template operator()<float>(); break;
        case DType::Float64:  launch.template operator()<double>(); break;
        case DType::Float16:  launch.template operator()<__half>(); break;
        case DType::Int32:    launch.template operator()<int32_t>(); break;
        case DType::Int64:    launch.template operator()<int64_t>(); break;
        default: throw std::runtime_error("topk ROCm: unsupported dtype");
    }

    return {values, indices};
}

// ============================================================================
// Sort kernel using Thrust
// ============================================================================

template<typename T>
static void sort_1d_thrust(const T* input, T* values, int64_t* indices_out,
                           int64_t n, bool descending, hipStream_t stream)
{
    auto policy = thrust::hip::par.on(stream);

    HIP_CHECK(hipMemcpyAsync(values, input, n * sizeof(T), hipMemcpyDeviceToDevice, stream));

    thrust::sequence(policy, thrust::device_pointer_cast(indices_out),
                     thrust::device_pointer_cast(indices_out + n), int64_t(0));

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
                 hipStream_t stream) -> std::pair<Tensor, Tensor>
{
    // Float16 upcast: convert to Float32, compute, convert values back
    if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto [values, indices] = sort_kernel(input_f32, dim, descending, stream);
        return {values.to(DType::Float16), indices};
    }

    // BFloat16 upcast: convert to Float32, compute, convert values back
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto [values, indices] = sort_kernel(input_f32, dim, descending, stream);
        return {values.to(DType::BFloat16), indices};
    }

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
        T* d_slice = nullptr;
        int64_t* d_idx = nullptr;
        HIP_CHECK(hipMalloc(&d_slice, dim_size * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_idx, dim_size * sizeof(int64_t)));

        int block = 256;
        int grid = std::min(int((dim_size + block - 1) / block), 1024);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                hipLaunchKernelGGL(extract_slice_kernel<T>,
                    dim3(grid), dim3(block), 0, stream,
                    input_cont.data<T>(), d_slice, dim_size, inner_size, outer, inner);
                HIP_CHECK(hipGetLastError());

                sort_1d_thrust<T>(d_slice, d_slice, d_idx, dim_size, descending, stream);

                hipLaunchKernelGGL(scatter_slice_kernel<T>,
                    dim3(grid), dim3(block), 0, stream,
                    d_slice, d_idx, values.data<T>(), indices.data<int64_t>(),
                    dim_size, inner_size, outer, inner);
                HIP_CHECK(hipGetLastError());
            }
        }

        HIP_CHECK(hipFree(d_slice));
        HIP_CHECK(hipFree(d_idx));
    };

    switch (dtype) {
        case DType::Float32: launch.template operator()<float>(); break;
        case DType::Float64: launch.template operator()<double>(); break;
        case DType::Int32:   launch.template operator()<int32_t>(); break;
        case DType::Int64:   launch.template operator()<int64_t>(); break;
        default: throw std::runtime_error("sort ROCm: unsupported dtype");
    }

    return {values, indices};
}

// ============================================================================
// ArgSort kernel
// ============================================================================

template<typename T>
__global__ void argsort_bitonic_kernel(const T* input, int64_t* output, int64_t n, bool descending) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        output[tid] = tid;
    }
    __syncthreads();

    for (int64_t size = 2; size <= n; size *= 2) {
        for (int64_t stride = size / 2; stride > 0; stride /= 2) {
            if (tid < n) {
                int64_t partner = tid ^ stride;
                if (partner > tid && partner < n) {
                    T val_tid = input[output[tid]];
                    T val_partner = input[output[partner]];

                    bool ascending_dir = ((tid & size) == 0);
                    if (descending) ascending_dir = !ascending_dir;

                    bool swap = ascending_dir ? (val_tid > val_partner) : (val_tid < val_partner);

                    if (swap) {
                        int64_t temp = output[tid];
                        output[tid] = output[partner];
                        output[partner] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }
}

__global__ void iota_kernel(int64_t* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = idx;
    }
}

__global__ void half_to_float_kernel(const __half* input, float* output, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        output[idx] = tenzor::rocm::safe_h2f(input[idx]);
    }
}

template<typename T>
static void launch_argsort(const T* d_input, int64_t* d_output, int64_t n,
                           bool descending, hipStream_t stream)
{
    if (n == 0) return;

    if (n == 1) {
        HIP_CHECK(hipMemsetAsync(d_output, 0, sizeof(int64_t), stream));
        return;
    }

    constexpr int MAX_BLOCK_SIZE = 1024;

    if (n <= MAX_BLOCK_SIZE) {
        int block_size = 1;
        while (block_size < n) block_size *= 2;
        if (block_size > MAX_BLOCK_SIZE) block_size = MAX_BLOCK_SIZE;
        hipLaunchKernelGGL(argsort_bitonic_kernel<T>,
            dim3(1), dim3(block_size), 0, stream,
            d_input, d_output, n, descending);
        HIP_CHECK(hipGetLastError());
        return;
    }

    // For larger arrays, use hipcub DeviceRadixSort with key-value pairs
    int64_t* d_indices_in = nullptr;
    HIP_CHECK(hipMalloc(&d_indices_in, n * sizeof(int64_t)));

    int init_blocks = get_num_blocks(n);
    hipLaunchKernelGGL(iota_kernel, dim3(init_blocks), dim3(BLOCK_SIZE), 0, stream, d_indices_in, n);
    HIP_CHECK(hipGetLastError());

    T* d_keys_out = nullptr;
    HIP_CHECK(hipMalloc(&d_keys_out, n * sizeof(T)));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    if (descending) {
        HIP_CHECK(hipcub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
            d_input, d_keys_out, d_indices_in, d_output, static_cast<int>(n), 0, sizeof(T) * 8, stream));
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipcub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
            d_input, d_keys_out, d_indices_in, d_output, static_cast<int>(n), 0, sizeof(T) * 8, stream));
    } else {
        HIP_CHECK(hipcub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_input, d_keys_out, d_indices_in, d_output, static_cast<int>(n), 0, sizeof(T) * 8, stream));
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));
        HIP_CHECK(hipcub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_input, d_keys_out, d_indices_in, d_output, static_cast<int>(n), 0, sizeof(T) * 8, stream));
    }

    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipFree(d_keys_out));
    HIP_CHECK(hipFree(d_indices_in));
}

auto argsort_kernel(const Tensor& input, int64_t dim, bool descending,
                    hipStream_t stream) -> Tensor
{
    // BFloat16 upcast: convert to Float32, compute (indices are Int64, no convert back needed)
    if (input.dtype() == DType::BFloat16) {
        return argsort_kernel(input.to(DType::Float32), dim, descending, stream);
    }

    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const int64_t n = input.numel();

    Tensor output(std::vector<int64_t>(input_shape.begin(), input_shape.end()), DType::Int64, device);

    switch (dtype) {
        case DType::Float32:
            launch_argsort(input.data<float>(), output.data<int64_t>(), n, descending, stream);
            break;
        case DType::Float64:
            launch_argsort(input.data<double>(), output.data<int64_t>(), n, descending, stream);
            break;
        case DType::Float16: {
            float* d_float_buf = nullptr;
            HIP_CHECK(hipMalloc(&d_float_buf, n * sizeof(float)));
            const __half* d_half = reinterpret_cast<const __half*>(input.data_ptr());
            int blocks = get_num_blocks(n);
            hipLaunchKernelGGL(half_to_float_kernel,
                dim3(blocks), dim3(BLOCK_SIZE), 0, stream,
                d_half, d_float_buf, n);
            HIP_CHECK(hipGetLastError());
            launch_argsort(d_float_buf, output.data<int64_t>(), n, descending, stream);
            HIP_CHECK(hipFree(d_float_buf));
            break;
        }
        case DType::Int32:
            launch_argsort(input.data<int32_t>(), output.data<int64_t>(), n, descending, stream);
            break;
        case DType::Int64:
            launch_argsort(input.data<int64_t>(), output.data<int64_t>(), n, descending, stream);
            break;
        default:
            throw std::runtime_error("argsort ROCm: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Unique kernel using Thrust
// ============================================================================

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
                          hipStream_t stream)
    -> std::tuple<Tensor, Tensor, Tensor>
{
    const int64_t numel = input.numel();
    const auto device = input.device();
    auto policy = thrust::hip::par.on(stream);

    // Copy and flatten input
    T* d_sorted = nullptr;
    HIP_CHECK(hipMalloc(&d_sorted, numel * sizeof(T)));
    HIP_CHECK(hipMemcpyAsync(d_sorted, input.data<T>(), numel * sizeof(T),
                             hipMemcpyDeviceToDevice, stream));

    // Create index mapping for inverse
    int64_t* d_orig_idx = nullptr;
    HIP_CHECK(hipMalloc(&d_orig_idx, numel * sizeof(int64_t)));
    thrust::sequence(policy, thrust::device_pointer_cast(d_orig_idx),
                     thrust::device_pointer_cast(d_orig_idx + numel), int64_t(0));

    // Sort input
    thrust::sort_by_key(policy,
        thrust::device_pointer_cast(d_sorted),
        thrust::device_pointer_cast(d_sorted + numel),
        thrust::device_pointer_cast(d_orig_idx));

    // Find unique elements via run-length encoding
    T* d_unique = nullptr;
    int64_t* d_counts = nullptr;
    int64_t* d_num_runs = nullptr;
    HIP_CHECK(hipMalloc(&d_unique, numel * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_counts, numel * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_num_runs, sizeof(int64_t)));

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    HIP_CHECK(hipcub::DeviceRunLengthEncode::Encode(
        d_temp, temp_bytes, d_sorted, d_unique, d_counts, d_num_runs,
        static_cast<int>(numel), stream));
    HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
    HIP_CHECK(hipcub::DeviceRunLengthEncode::Encode(
        d_temp, temp_bytes, d_sorted, d_unique, d_counts, d_num_runs,
        static_cast<int>(numel), stream));
    HIP_CHECK(hipFree(d_temp));

    // Get num_unique on host
    int64_t num_unique = 0;
    HIP_CHECK(hipMemcpyAsync(&num_unique, d_num_runs, sizeof(int64_t), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Create unique values tensor
    Tensor unique_vals({num_unique}, input.dtype(), device);
    HIP_CHECK(hipMemcpyAsync(unique_vals.data<T>(), d_unique, num_unique * sizeof(T),
                             hipMemcpyDeviceToDevice, stream));

    // Create counts tensor if requested
    Tensor counts_tensor;
    if (return_counts) {
        counts_tensor = Tensor({num_unique}, DType::Int64, device);
        HIP_CHECK(hipMemcpyAsync(counts_tensor.data<int64_t>(), d_counts, num_unique * sizeof(int64_t),
                                 hipMemcpyDeviceToDevice, stream));
    }

    // Create inverse indices if requested
    Tensor inverse_tensor;
    if (return_inverse) {
        inverse_tensor = Tensor({numel}, DType::Int64, device);

        int64_t* d_offsets = nullptr;
        HIP_CHECK(hipMalloc(&d_offsets, (num_unique + 1) * sizeof(int64_t)));

        void* d_scan_temp = nullptr;
        size_t scan_temp_bytes = 0;
        HIP_CHECK(hipcub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes, d_counts, d_offsets,
                                         static_cast<int>(num_unique), stream));
        HIP_CHECK(hipMalloc(&d_scan_temp, scan_temp_bytes));
        HIP_CHECK(hipcub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes, d_counts, d_offsets,
                                         static_cast<int>(num_unique), stream));
        HIP_CHECK(hipFree(d_scan_temp));

        int block = 256;
        int grid = (num_unique + block - 1) / block;
        hipLaunchKernelGGL(fill_inverse_kernel,
            dim3(grid), dim3(block), 0, stream,
            d_orig_idx, d_offsets, inverse_tensor.data<int64_t>(),
            num_unique, numel);
        HIP_CHECK(hipGetLastError());

        HIP_CHECK(hipFree(d_offsets));
    }

    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipFree(d_sorted));
    HIP_CHECK(hipFree(d_orig_idx));
    HIP_CHECK(hipFree(d_unique));
    HIP_CHECK(hipFree(d_counts));
    HIP_CHECK(hipFree(d_num_runs));

    return {unique_vals, inverse_tensor, counts_tensor};
}

auto unique_kernel(const Tensor& input, bool sorted_output, bool return_inverse,
                   bool return_counts, hipStream_t stream)
    -> std::tuple<Tensor, Tensor, Tensor>
{
    // Float16 upcast: convert to Float32, compute, convert unique values back
    if (input.dtype() == DType::Float16) {
        auto [unique_vals, inverse, counts] = unique_kernel(input.to(DType::Float32),
            sorted_output, return_inverse, return_counts, stream);
        return {unique_vals.to(DType::Float16), inverse, counts};
    }

    // BFloat16 upcast: convert to Float32, compute, convert unique values back
    if (input.dtype() == DType::BFloat16) {
        auto [unique_vals, inverse, counts] = unique_kernel(input.to(DType::Float32),
            sorted_output, return_inverse, return_counts, stream);
        return {unique_vals.to(DType::BFloat16), inverse, counts};
    }

    // Bool widen: Thrust's hash-on-value path doesn't instantiate cleanly for
    // bool, so promote to Int32, run the pipeline, and narrow the resulting
    // unique values back to Bool.
    if (input.dtype() == DType::Bool) {
        auto [unique_vals, inverse, counts] = unique_kernel(input.to(DType::Int32),
            sorted_output, return_inverse, return_counts, stream);
        return {unique_vals.to(DType::Bool), inverse, counts};
    }

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
            throw std::runtime_error("unique ROCm: unsupported dtype");
    }
}

} // namespace rocm
} // namespace tenzor
