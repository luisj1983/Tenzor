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

// RAII wrapper for a typed HIP device allocation. Frees on scope exit so that an
// exception thrown by HIP_CHECK (kernel launch errors, thrust failures, etc.)
// while a buffer is live unwinds without leaking device memory. The destructor
// is noexcept and swallows the hipFree status — there is nothing actionable to
// do during stack unwinding, and the process state is already being torn down.
template<typename T>
struct ScopedDevicePtr {
    T* ptr = nullptr;

    explicit ScopedDevicePtr(int64_t count) {
        HIP_CHECK(hipMalloc(&ptr, static_cast<size_t>(count) * sizeof(T)));
    }

    ~ScopedDevicePtr() noexcept {
        if (ptr) (void)hipFree(ptr);
    }

    ScopedDevicePtr(const ScopedDevicePtr&) = delete;
    ScopedDevicePtr& operator=(const ScopedDevicePtr&) = delete;
    ScopedDevicePtr(ScopedDevicePtr&&) = delete;
    ScopedDevicePtr& operator=(ScopedDevicePtr&&) = delete;

    T* get() const { return ptr; }
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

// F139: NaN detection via IEEE-754 bit patterns. ROCm's __half/float NaN
// intrinsics (isnan / __hisnan) canonicalise unreliably (see reference note),
// so test the bit pattern directly. Integer types are never NaN.
template<typename T> __device__ __forceinline__ bool topk_isnan(const T&) { return false; }
__device__ __forceinline__ bool topk_isnan(const float& x) {
    unsigned b = __float_as_uint(x); return (b & 0x7FFFFFFFu) > 0x7F800000u;
}
__device__ __forceinline__ bool topk_isnan(const double& x) {
    unsigned long long b = __double_as_longlong(x);
    return (b & 0x7FFFFFFFFFFFFFFFULL) > 0x7FF0000000000000ULL;
}
__device__ __forceinline__ bool topk_isnan(const __half& x) {
    unsigned short b = __half_as_ushort(x); return (b & 0x7FFFu) > 0x7C00u;
}

// NaN-aware comparisons for TopK: NaN is treated as the LARGEST value (matches
// CPU/CUDA/OneAPI/Vulkan sort/topk — NaN goes to the top for largest, and is
// never selected as a smallest). Plain a>b/a<b return false for NaN, so NaN was
// silently never selected on ROCm — diverging topk on inputs containing NaN.
template<typename T> __device__ __forceinline__ bool nan_gt(const T& a, const T& b) {
    bool an = topk_isnan(a), bn = topk_isnan(b);
    if (an) return !bn;      // NaN beats any non-NaN; NaN vs NaN -> not greater
    if (bn) return false;    // b is NaN (largest), a is not
    return hip_gt(a, b);
}
template<typename T> __device__ __forceinline__ bool nan_lt(const T& a, const T& b) {
    bool an = topk_isnan(a), bn = topk_isnan(b);
    if (an) return false;    // NaN is largest, never "less"
    if (bn) return true;     // b is NaN (largest), a (non-NaN) is less
    return hip_lt(a, b);
}
// F143: thrust comparator functors so the full sort orders NaN as the LARGEST
// value regardless of sign bit. thrust::less / thrust::greater on arithmetic keys
// are bit-order (radix): a positive NaN sorts after +inf (last, OK) but a
// NEGATIVE-sign NaN sorts below -inf (first), diverging from CPU/CUDA/Vulkan
// which put every NaN last (ascending) via a sign-agnostic bit-pattern check.
template<typename T> struct NanLessFunctor {
    __device__ bool operator()(const T& a, const T& b) const { return nan_lt(a, b); }
};
template<typename T> struct NanGreaterFunctor {
    __device__ bool operator()(const T& a, const T& b) const { return nan_gt(a, b); }
};

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
                (largest ? nan_gt(val, best_val) : nan_lt(val, best_val)) ||
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
                        (nan_gt(s_cand_vals[tid + stride], s_cand_vals[tid]) ||
                         (hip_eq(s_cand_vals[tid + stride], s_cand_vals[tid]) &&
                          s_cand_pos[tid + stride] < s_cand_pos[tid])) :
                        (nan_lt(s_cand_vals[tid + stride], s_cand_vals[tid]) ||
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

    // Sort the k results using parallel odd-even transposition sort. Each thread
    // strides over ALL the pairs assigned to it so the sort covers every position
    // even when k > 2*blockDim — previously a thread only handled pair
    // i = 2*tid + (phase&1), so positions >= 2*blockDim (k>512 with 256 threads)
    // were never compared and the tail came out unsorted. Pairs within a phase
    // are non-overlapping (disjoint i, i+1), so a thread can do its swaps without
    // intra-phase synchronization; the __syncthreads() separates phases.
    for (int64_t phase = 0; phase < k; ++phase) {
        for (int64_t j = tid; 2 * j + (phase & 1) + 1 < k; j += nthreads) {
            int64_t i = 2 * j + (phase & 1);
            bool should_swap = largest ?
                nan_lt(s_topk_vals[i], s_topk_vals[i + 1]) :
                nan_gt(s_topk_vals[i], s_topk_vals[i + 1]);
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

    const int64_t ndim = input.ndim();

    // Normalize negative dim (the dispatcher passes dim through unmodified,
    // defaulting to -1, so the kernel owns normalization here).
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("topk ROCm: dimension out of range");
    }

    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
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

    // F143: NaN-aware comparators (NaN as largest, sign-agnostic) — see functors.
    // stable_sort_by_key (not sort_by_key) so duplicate-value tie order is
    // deterministic (lowest original index first), matching CPU/CUDA/Vulkan.
    if (descending) {
        thrust::stable_sort_by_key(policy,
            thrust::device_pointer_cast(values),
            thrust::device_pointer_cast(values + n),
            thrust::device_pointer_cast(indices_out),
            NanGreaterFunctor<T>());
    } else {
        thrust::stable_sort_by_key(policy,
            thrust::device_pointer_cast(values),
            thrust::device_pointer_cast(values + n),
            thrust::device_pointer_cast(indices_out),
            NanLessFunctor<T>());
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

    const int64_t ndim = input.ndim();

    // Normalize negative dim (the dispatcher passes dim through unmodified,
    // defaulting to -1, so the kernel owns normalization here).
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("sort ROCm: dimension out of range");
    }

    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
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
        // RAII-owned device scratch: freed on scope exit even if a HIP_CHECK or
        // thrust call inside the loop throws, so no device buffer is leaked.
        ScopedDevicePtr<T> d_slice(dim_size);
        ScopedDevicePtr<int64_t> d_idx(dim_size);

        int block = 256;
        int grid = std::min(int((dim_size + block - 1) / block), 1024);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                hipLaunchKernelGGL(extract_slice_kernel<T>,
                    dim3(grid), dim3(block), 0, stream,
                    input_cont.data<T>(), d_slice.get(), dim_size, inner_size, outer, inner);
                HIP_CHECK(hipGetLastError());

                sort_1d_thrust<T>(d_slice.get(), d_slice.get(), d_idx.get(), dim_size, descending, stream);

                hipLaunchKernelGGL(scatter_slice_kernel<T>,
                    dim3(grid), dim3(block), 0, stream,
                    d_slice.get(), d_idx.get(), values.data<T>(), indices.data<int64_t>(),
                    dim_size, inner_size, outer, inner);
                HIP_CHECK(hipGetLastError());
            }
        }
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

    // F143-parity fix: hipcub::DeviceRadixSort (used previously) sorts raw
    // bit patterns with no custom comparator, reproducing the exact
    // sign-dependent NaN misplacement F143 fixed for plain sort_kernel above
    // (a negative-sign NaN would sort FIRST instead of last). DeviceRadixSort
    // doesn't accept a custom comparator, so route through the same
    // thrust::stable_sort_by_key + NaN-aware comparator technique
    // sort_1d_thrust uses, so NaN consistently sorts LAST (any sign),
    // matching CPU/CUDA/OneAPI/Vulkan. stable_sort_by_key (not sort_by_key)
    // preserves the deterministic lowest-original-index tie order that
    // DeviceRadixSort's stable contract previously provided.
    // RAII scratch: a throwing HIP_CHECK (iota launch error, sort failure)
    // between the allocation and the free would otherwise leak this.
    ScopedDevicePtr<T> d_keys_buf(n);
    T* d_keys = d_keys_buf.get();
    HIP_CHECK(hipMemcpyAsync(d_keys, d_input, n * sizeof(T), hipMemcpyDeviceToDevice, stream));

    int init_blocks = get_num_blocks(n);
    hipLaunchKernelGGL(iota_kernel, dim3(init_blocks), dim3(BLOCK_SIZE), 0, stream, d_output, n);
    HIP_CHECK(hipGetLastError());

    auto policy = thrust::hip::par.on(stream);
    if (descending) {
        thrust::stable_sort_by_key(policy,
            thrust::device_pointer_cast(d_keys),
            thrust::device_pointer_cast(d_keys + n),
            thrust::device_pointer_cast(d_output),
            NanGreaterFunctor<T>());
    } else {
        thrust::stable_sort_by_key(policy,
            thrust::device_pointer_cast(d_keys),
            thrust::device_pointer_cast(d_keys + n),
            thrust::device_pointer_cast(d_output),
            NanLessFunctor<T>());
    }

    // audit-9 JJ.5: the sort is async on `stream`; sync before the RAII
    // buffer frees at scope exit so the in-flight sort doesn't read freed
    // pages.
    HIP_CHECK(hipStreamSynchronize(stream));
}

// Scatter per-slice local argsort indices (0..dim_size-1) back into the
// strided output positions for one (outer, inner) lane.
__global__ void scatter_argsort_slice_kernel(const int64_t* __restrict__ sorted_idx,
                                             int64_t* __restrict__ out_idx,
                                             int64_t dim_size, int64_t inner_size,
                                             int64_t outer, int64_t inner)
{
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dim_size;
         i += gridDim.x * blockDim.x) {
        int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
        out_idx[offset] = sorted_idx[i];
    }
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
    const int64_t ndim = input.ndim();

    // Normalize negative dim (the dispatcher passes dim through unmodified,
    // defaulting to -1, so the kernel owns normalization here).
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("argsort ROCm: dimension out of range");
    }

    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    const auto& shape = input_cont.shape();
    const int64_t dim_size = shape[dim];

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), DType::Int64, device);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    // Sort each [outer, inner] lane independently along `dim`. We extract the
    // (strided) slice into a contiguous buffer, argsort it to get per-slice
    // local indices, then scatter those indices back to the strided positions.
    auto run = [&]<typename T>(const T* base_ptr) {
        // RAII-owned device scratch: freed on scope exit even if a HIP_CHECK or
        // argsort call inside the loop throws, so no device buffer is leaked.
        ScopedDevicePtr<T> d_slice(dim_size);
        ScopedDevicePtr<int64_t> d_idx(dim_size);

        int block = 256;
        int grid = std::min(int((dim_size + block - 1) / block), 1024);

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                hipLaunchKernelGGL(extract_slice_kernel<T>,
                    dim3(grid), dim3(block), 0, stream,
                    base_ptr, d_slice.get(), dim_size, inner_size, outer, inner);
                HIP_CHECK(hipGetLastError());

                launch_argsort(d_slice.get(), d_idx.get(), dim_size, descending, stream);

                hipLaunchKernelGGL(scatter_argsort_slice_kernel,
                    dim3(grid), dim3(block), 0, stream,
                    d_idx.get(), output.data<int64_t>(),
                    dim_size, inner_size, outer, inner);
                HIP_CHECK(hipGetLastError());
            }
        }
    };

    switch (dtype) {
        case DType::Float32:
            run.template operator()<float>(input_cont.data<float>());
            break;
        case DType::Float64:
            run.template operator()<double>(input_cont.data<double>());
            break;
        case DType::Float16: {
            // Widen the whole tensor to Float32 once, then slice/sort lanes.
            Tensor input_f32 = input_cont.to(DType::Float32);
            run.template operator()<float>(input_f32.data<float>());
            break;
        }
        case DType::Int32:
            run.template operator()<int32_t>(input_cont.data<int32_t>());
            break;
        case DType::Int64:
            run.template operator()<int64_t>(input_cont.data<int64_t>());
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

    // RAII scratch (ScopedDevicePtr): a throwing HIP_CHECK / thrust error between
    // these allocations and the frees would otherwise leak device memory.
    // Copy and flatten input
    ScopedDevicePtr<T> d_sorted_buf(numel);
    T* d_sorted = d_sorted_buf.get();
    HIP_CHECK(hipMemcpyAsync(d_sorted, input.data<T>(), numel * sizeof(T),
                             hipMemcpyDeviceToDevice, stream));

    // Create index mapping for inverse
    ScopedDevicePtr<int64_t> d_orig_idx_buf(numel);
    int64_t* d_orig_idx = d_orig_idx_buf.get();
    thrust::sequence(policy, thrust::device_pointer_cast(d_orig_idx),
                     thrust::device_pointer_cast(d_orig_idx + numel), int64_t(0));

    // Sort input
    thrust::sort_by_key(policy,
        thrust::device_pointer_cast(d_sorted),
        thrust::device_pointer_cast(d_sorted + numel),
        thrust::device_pointer_cast(d_orig_idx));

    // Find unique elements via run-length encoding
    ScopedDevicePtr<T> d_unique_buf(numel);
    ScopedDevicePtr<int64_t> d_counts_buf(numel);
    ScopedDevicePtr<int64_t> d_num_runs_buf(1);
    T* d_unique = d_unique_buf.get();
    int64_t* d_counts = d_counts_buf.get();
    int64_t* d_num_runs = d_num_runs_buf.get();

    size_t temp_bytes = 0;
    HIP_CHECK(hipcub::DeviceRunLengthEncode::Encode(
        nullptr, temp_bytes, d_sorted, d_unique, d_counts, d_num_runs,
        static_cast<int>(numel), stream));
    ScopedDevicePtr<char> d_temp_buf(static_cast<int64_t>(temp_bytes));
    void* d_temp = d_temp_buf.get();
    HIP_CHECK(hipcub::DeviceRunLengthEncode::Encode(
        d_temp, temp_bytes, d_sorted, d_unique, d_counts, d_num_runs,
        static_cast<int>(numel), stream));

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

        ScopedDevicePtr<int64_t> d_offsets_buf(num_unique + 1);
        int64_t* d_offsets = d_offsets_buf.get();

        size_t scan_temp_bytes = 0;
        HIP_CHECK(hipcub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes, d_counts, d_offsets,
                                         static_cast<int>(num_unique), stream));
        ScopedDevicePtr<char> d_scan_temp_buf(static_cast<int64_t>(scan_temp_bytes));
        void* d_scan_temp = d_scan_temp_buf.get();
        HIP_CHECK(hipcub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_bytes, d_counts, d_offsets,
                                         static_cast<int>(num_unique), stream));

        int block = 256;
        int grid = (num_unique + block - 1) / block;
        hipLaunchKernelGGL(fill_inverse_kernel,
            dim3(grid), dim3(block), 0, stream,
            d_orig_idx, d_offsets, inverse_tensor.data<int64_t>(),
            num_unique, numel);
        HIP_CHECK(hipGetLastError());
    }

    // Sync before the RAII scratch buffers free at scope exit so in-flight async
    // work does not read freed pages.
    HIP_CHECK(hipStreamSynchronize(stream));

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
