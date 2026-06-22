#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/caching_allocator.hpp"
#include "tenzor/backend/dtype_dispatch.hpp"
#include "tenzor/backend/backend.hpp"  // For OpAttributes (dispatch wrappers)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <climits>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cub/cub.cuh>
#include "../cuda_error.hpp"
#include "cuda_launch_utils.cuh"
#include "cuda_common.cuh"
#include "../cuda_stream_pool.hpp"

namespace tenzor {
namespace cuda {

// Resolve a CUDA stream: if `stream` is non-null (caller-provided), use it
// directly; otherwise acquire one from the per-device stream pool.
// Returns the stream to use and an RAII guard that keeps the pool stream alive.
static std::pair<cudaStream_t, StreamGuard> resolve_stream(
    cudaStream_t stream, const Tensor& ref_tensor) {
    if (stream != nullptr) {
        return {stream, StreamGuard{}};
    }
    int device_id = ref_tensor.device().index;
    auto guard = CUDAStreamPool::instance().acquire_guard(device_id);
    return {guard.get(), std::move(guard)};
}

// Constants
constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_SIZE = 1024;
// Reduction block size is used for static __shared__ memory in kernels.
// 256 is a good default for occupancy on most architectures (Volta+).
// For kernels without shared memory constraints, use optimal_launch_config().
constexpr int REDUCTION_BLOCK_SIZE = 256;

// Type-safe sentinel values for max/min reductions.
// Using -FLT_MAX for double produces wrong results since FLT_MAX << DBL_MAX.
template<typename T> __device__ __host__ inline T sentinel_lowest();
template<typename T> __device__ __host__ inline T sentinel_max();

template<> __device__ __host__ inline float sentinel_lowest<float>() { return -FLT_MAX; }
template<> __device__ __host__ inline float sentinel_max<float>() { return FLT_MAX; }
template<> __device__ __host__ inline double sentinel_lowest<double>() { return -DBL_MAX; }
template<> __device__ __host__ inline double sentinel_max<double>() { return DBL_MAX; }
template<> __device__ __host__ inline int32_t sentinel_lowest<int32_t>() { return INT32_MIN; }
template<> __device__ __host__ inline int32_t sentinel_max<int32_t>() { return INT32_MAX; }
template<> __device__ __host__ inline int64_t sentinel_lowest<int64_t>() { return INT64_MIN; }
template<> __device__ __host__ inline int64_t sentinel_max<int64_t>() { return INT64_MAX; }
template<> __device__ __host__ inline int8_t sentinel_lowest<int8_t>() { return INT8_MIN; }
template<> __device__ __host__ inline int8_t sentinel_max<int8_t>() { return INT8_MAX; }
template<> __device__ __host__ inline int16_t sentinel_lowest<int16_t>() { return INT16_MIN; }
template<> __device__ __host__ inline int16_t sentinel_max<int16_t>() { return INT16_MAX; }
template<> __device__ __host__ inline uint8_t sentinel_lowest<uint8_t>() { return 0; }
template<> __device__ __host__ inline uint8_t sentinel_max<uint8_t>() { return UINT8_MAX; }
template<> __device__ __host__ inline uint16_t sentinel_lowest<uint16_t>() { return 0; }
template<> __device__ __host__ inline uint16_t sentinel_max<uint16_t>() { return UINT16_MAX; }
template<> __device__ __host__ inline uint32_t sentinel_lowest<uint32_t>() { return 0; }
template<> __device__ __host__ inline uint32_t sentinel_max<uint32_t>() { return UINT32_MAX; }
template<> __device__ __host__ inline uint64_t sentinel_lowest<uint64_t>() { return 0; }
template<> __device__ __host__ inline uint64_t sentinel_max<uint64_t>() { return UINT64_MAX; }
template<> __device__ __host__ inline bool sentinel_lowest<bool>() { return false; }
template<> __device__ __host__ inline bool sentinel_max<bool>() { return true; }

// Metadata struct passed by value to kernels (avoids cudaMalloc for shape/stride arrays).
//
// Audit F3: rank cap lifted from 8 to 16 — matches the activations.cu copy
// and the maximum supported in CPU/ROCm. The old 8-dim limit silently
// truncated higher-rank tensors (e.g. block-sparse attention with rank 9+).
constexpr int DIM_META_MAX_RANK = 16;
struct DimMeta {
    int64_t shape[DIM_META_MAX_RANK];
    int64_t strides[DIM_META_MAX_RANK];
};

static DimMeta make_dim_meta(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides) {
    if (shape.size() > DIM_META_MAX_RANK) {
        throw std::runtime_error(
            "CUDA DimMeta (reduction): tensor rank " + std::to_string(shape.size()) +
            " exceeds maximum " + std::to_string(DIM_META_MAX_RANK));
    }
    DimMeta meta{};
    for (size_t i = 0; i < shape.size(); ++i) {
        meta.shape[i] = shape[i];
        meta.strides[i] = strides[i];
    }
    return meta;
}

// ============================================================================
// Type helpers for __half support
// ============================================================================

// Accumulation type: use float for half/bfloat16 to prevent overflow
template<typename T> struct AccumType { using type = T; };
template<> struct AccumType<__half> { using type = float; };
template<> struct AccumType<__nv_bfloat16> { using type = float; };

// Device-side helpers
template<typename T>
__device__ __forceinline__ T cuda_zero() { return T(0); }

template<>
__device__ __forceinline__ __half cuda_zero<__half>() { return __float2half(0.0f); }

template<typename T>
__device__ __forceinline__ T cuda_add(T a, T b) { return a + b; }

template<>
__device__ __forceinline__ __half cuda_add<__half>(__half a, __half b) { return __hadd(a, b); }

// BFloat16 type helpers
template<>
__device__ __forceinline__ __nv_bfloat16 cuda_zero<__nv_bfloat16>() { return __float2bfloat16(0.0f); }

template<>
__device__ __forceinline__ __nv_bfloat16 cuda_add<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) { return __hadd(a, b); }

// Get negative infinity for __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 bfloat16_neg_inf() {
    return __ushort_as_bfloat16(0xFF80);  // -inf in bfloat16
}

// Get positive infinity for __nv_bfloat16
__device__ __forceinline__ __nv_bfloat16 bfloat16_pos_inf() {
    return __ushort_as_bfloat16(0x7F80);  // +inf in bfloat16
}

// BFloat16 comparison helpers (NaN-propagating: max/min of a set containing NaN is NaN)
__device__ __forceinline__ __nv_bfloat16 cuda_max_val(__nv_bfloat16 a, __nv_bfloat16 b) {
    return (__hisnan(a) || __hisnan(b)) ? __hadd(a, b) : (__hgt(a, b) ? a : b);
}

__device__ __forceinline__ __nv_bfloat16 cuda_min_val(__nv_bfloat16 a, __nv_bfloat16 b) {
    return (__hisnan(a) || __hisnan(b)) ? __hadd(a, b) : (__hlt(a, b) ? a : b);
}

// Generic NaN-propagating max/min for real scalar types (float/double/integers).
// For floats, if either operand is NaN the result is NaN (IEEE reduction semantics);
// integers have no NaN so this is a plain comparison.
template<typename T> __device__ __forceinline__ T cuda_max_val(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) return (isnan(a) || isnan(b)) ? (a + b) : ((a > b) ? a : b);
    else return (a > b) ? a : b;
}
template<typename T> __device__ __forceinline__ T cuda_min_val(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) return (isnan(a) || isnan(b)) ? (a + b) : ((a < b) ? a : b);
    else return (a < b) ? a : b;
}

// ----------------------------------------------------------------------------
// NaN-aware argmax/argmin candidate predicates.
//
// PyTorch / CPU reference semantics: argmax/argmin propagate NaN — if any input
// is NaN the returned index is that of the FIRST (lowest-index) NaN, because NaN
// compares as the maximum (and the minimum) of the set. Plain `cand > best`
// comparisons silently skip NaN (every comparison against NaN is false), so we
// need explicit predicates.
//
// `arg_max_takes(cand, best)` returns true iff a `cand` value should replace the
// current `best`:
//   * a NaN cand beats any non-NaN best;
//   * NaN vs NaN never replaces (keep the earlier index — first-NaN wins);
//   * otherwise replace only on a strict numeric improvement.
// Equivalent to: !isnan(best) && (isnan(cand) || cand > best).
//
// Callers MUST only replace on a true result (strict win), so that on ties the
// pre-existing (lower-index) entry is retained. In the tree/warp combine steps
// the left operand (shared[tid] / current lane) always carries the lower
// original index, so "replace only on strict win" preserves lowest-index
// semantics for NaN ties as well.
template<typename T> __device__ __forceinline__ bool arg_max_takes(T cand, T best) {
    if constexpr (std::is_floating_point_v<T>) {
        return !isnan(best) && (isnan(cand) || cand > best);
    } else {
        return cand > best;
    }
}
template<typename T> __device__ __forceinline__ bool arg_min_takes(T cand, T best) {
    if constexpr (std::is_floating_point_v<T>) {
        return !isnan(best) && (isnan(cand) || cand < best);
    } else {
        return cand < best;
    }
}
// __half / __nv_bfloat16 overloads (use __hisnan; ordered compares via __hgt/__hlt).
__device__ __forceinline__ bool arg_max_takes(__half cand, __half best) {
    return !__hisnan(best) && (__hisnan(cand) || __hgt(cand, best));
}
__device__ __forceinline__ bool arg_min_takes(__half cand, __half best) {
    return !__hisnan(best) && (__hisnan(cand) || __hlt(cand, best));
}
__device__ __forceinline__ bool arg_max_takes(__nv_bfloat16 cand, __nv_bfloat16 best) {
    return !__hisnan(best) && (__hisnan(cand) || __hgt(cand, best));
}
__device__ __forceinline__ bool arg_min_takes(__nv_bfloat16 cand, __nv_bfloat16 best) {
    return !__hisnan(best) && (__hisnan(cand) || __hlt(cand, best));
}

// NaN detection wrappers usable across real and half types.
template<typename T> __device__ __forceinline__ bool arg_is_nan(T v) {
    if constexpr (std::is_floating_point_v<T>) return isnan(v);
    else return false;
}
__device__ __forceinline__ bool arg_is_nan(__half v) { return __hisnan(v); }
__device__ __forceinline__ bool arg_is_nan(__nv_bfloat16 v) { return __hisnan(v); }

// Exact-equality wrappers usable across real and half types (half compares via
// the __heq intrinsic to match the explicit __hgt/__hlt convention used here).
template<typename T> __device__ __forceinline__ bool arg_is_eq(T a, T b) { return a == b; }
__device__ __forceinline__ bool arg_is_eq(__half a, __half b) { return __heq(a, b); }
__device__ __forceinline__ bool arg_is_eq(__nv_bfloat16 a, __nv_bfloat16 b) { return __heq(a, b); }

// Index-aware variants for combine steps where the candidate may carry a LOWER
// original index than the current best (e.g. combining unsorted block winners).
// On a NaN-vs-NaN tie (or any exact tie), the lower original index wins so the
// returned index matches the CPU "first NaN / first occurrence" reference.
template<typename T>
__device__ __forceinline__ bool arg_max_takes_idx(T cand, int64_t cand_idx, T best, int64_t best_idx) {
    if (arg_max_takes(cand, best)) return true;
    // Tie-break: on an EXACT tie (equal NaNs, or equal finite values) keep the
    // lower original index, matching the CPU reference whose ascending sequential
    // scan retains the first occurrence on value ties. Without this, the unsorted
    // tree/warp combine could keep a later index on a finite tie -> cross-backend
    // parity break for tensors with duplicated max/min values.
    if (arg_is_nan(cand) && arg_is_nan(best)) return cand_idx < best_idx;
    if (!arg_is_nan(cand) && !arg_is_nan(best) && arg_is_eq(cand, best)) return cand_idx < best_idx;
    return false;
}
template<typename T>
__device__ __forceinline__ bool arg_min_takes_idx(T cand, int64_t cand_idx, T best, int64_t best_idx) {
    if (arg_min_takes(cand, best)) return true;
    if (arg_is_nan(cand) && arg_is_nan(best)) return cand_idx < best_idx;
    if (!arg_is_nan(cand) && !arg_is_nan(best) && arg_is_eq(cand, best)) return cand_idx < best_idx;
    return false;
}

// GPU-side scalar fill (replaces host_zero + cudaMemcpyAsync H2D pattern to avoid UB)
template<typename T>
__global__ void fill_scalar_kernel(T* dst, T value) { dst[0] = value; }

// ============================================================================
// Helper kernels for GPU-side scalar operations (avoids D2H transfers)
// ============================================================================

// Convert __half array to float array (for sorting/reduction on types CUB doesn't support)
__global__ void half_to_float_kernel(const __half* src, float* dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __half2float(src[idx]);
}

// Phase-2 reduction kernels: given block-level argmax/argmin indices,
// compare original values at those indices to find the global winner.
template<typename T>
__global__ void argmax_final_kernel(const T* input, const int64_t* block_indices, int64_t* output, int num_blocks) {
    __shared__ T shared_vals[REDUCTION_BLOCK_SIZE + 1];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE + 1];

    int tid = threadIdx.x;
    T best_val;
    int64_t best_idx;

    if (tid < num_blocks) {
        best_idx = block_indices[tid];
        best_val = input[best_idx];
    } else {
        best_val = sentinel_lowest<T>();
        best_idx = 0;
    }

    // Handle case where num_blocks > blockDim.x (unlikely but safe)
    for (int i = tid + blockDim.x; i < num_blocks; i += blockDim.x) {
        int64_t idx = block_indices[i];
        T val = input[idx];
        if (arg_max_takes_idx(val, idx, best_val, best_idx)) {
            best_val = val;
            best_idx = idx;
        }
    }

    shared_vals[tid] = best_val;
    shared_idxs[tid] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (arg_max_takes_idx(shared_vals[tid + stride], shared_idxs[tid + stride],
                                  shared_vals[tid], shared_idxs[tid])) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = shared_idxs[0];
    }
}

template<typename T>
__global__ void argmin_final_kernel(const T* input, const int64_t* block_indices, int64_t* output, int num_blocks) {
    __shared__ T shared_vals[REDUCTION_BLOCK_SIZE + 1];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE + 1];

    int tid = threadIdx.x;
    T best_val;
    int64_t best_idx;

    if (tid < num_blocks) {
        best_idx = block_indices[tid];
        best_val = input[best_idx];
    } else {
        best_val = sentinel_max<T>();
        best_idx = 0;
    }

    for (int i = tid + blockDim.x; i < num_blocks; i += blockDim.x) {
        int64_t idx = block_indices[i];
        T val = input[idx];
        if (arg_min_takes_idx(val, idx, best_val, best_idx)) {
            best_val = val;
            best_idx = idx;
        }
    }

    shared_vals[tid] = best_val;
    shared_idxs[tid] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (arg_min_takes_idx(shared_vals[tid + stride], shared_idxs[tid + stride],
                                  shared_vals[tid], shared_idxs[tid])) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = shared_idxs[0];
    }
}

// Half-precision variants of final reduction kernels
__global__ void argmax_final_kernel_half(const __half* input, const int64_t* block_indices, int64_t* output, int num_blocks) {
    // +1 padding avoids shared memory bank conflicts between vals and idxs arrays
    __shared__ float shared_vals[REDUCTION_BLOCK_SIZE + 1];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE + 1];

    int tid = threadIdx.x;
    float best_val;
    int64_t best_idx;

    if (tid < num_blocks) {
        best_idx = block_indices[tid];
        best_val = __half2float(input[best_idx]);
    } else {
        best_val = -FLT_MAX;
        best_idx = 0;
    }

    for (int i = tid + blockDim.x; i < num_blocks; i += blockDim.x) {
        int64_t idx = block_indices[i];
        float val = __half2float(input[idx]);
        if (arg_max_takes_idx(val, idx, best_val, best_idx)) {
            best_val = val;
            best_idx = idx;
        }
    }

    shared_vals[tid] = best_val;
    shared_idxs[tid] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (arg_max_takes_idx(shared_vals[tid + stride], shared_idxs[tid + stride],
                                  shared_vals[tid], shared_idxs[tid])) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = shared_idxs[0];
    }
}

__global__ void argmin_final_kernel_half(const __half* input, const int64_t* block_indices, int64_t* output, int num_blocks) {
    __shared__ float shared_vals[REDUCTION_BLOCK_SIZE + 1];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE + 1];

    int tid = threadIdx.x;
    float best_val;
    int64_t best_idx;

    if (tid < num_blocks) {
        best_idx = block_indices[tid];
        best_val = __half2float(input[best_idx]);
    } else {
        best_val = FLT_MAX;
        best_idx = 0;
    }

    for (int i = tid + blockDim.x; i < num_blocks; i += blockDim.x) {
        int64_t idx = block_indices[i];
        float val = __half2float(input[idx]);
        if (arg_min_takes_idx(val, idx, best_val, best_idx)) {
            best_val = val;
            best_idx = idx;
        }
    }

    shared_vals[tid] = best_val;
    shared_idxs[tid] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (arg_min_takes_idx(shared_vals[tid + stride], shared_idxs[tid + stride],
                                  shared_vals[tid], shared_idxs[tid])) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = shared_idxs[0];
    }
}
// ============================================================================
// Warp-level reduction primitives
// ============================================================================

// warp_reduce_sum (including __half and __nv_bfloat16 specializations) is in cuda_common.cuh

// Helper for __half comparison (using __hgt and __hlt)
__device__ __forceinline__ __half cuda_max_val(__half a, __half b) {
    return (__hisnan(a) || __hisnan(b)) ? __hadd(a, b) : (__hgt(a, b) ? a : b);
}

__device__ __forceinline__ __half cuda_min_val(__half a, __half b) {
    return (__hisnan(a) || __hisnan(b)) ? __hadd(a, b) : (__hlt(a, b) ? a : b);
}

// Get negative infinity for __half
__device__ __forceinline__ __half half_neg_inf() {
    return __ushort_as_half(0xFC00);  // -inf in half precision
}

// Get positive infinity for __half
__device__ __forceinline__ __half half_pos_inf() {
    return __ushort_as_half(0x7C00);  // +inf in half precision
}

// warp_reduce_max, warp_reduce_min are in cuda_common.cuh

// ============================================================================
// Block-level reduction kernels (full reduction)
// ============================================================================

// Block-level sum reduction.
//
// Accumulation always happens in AccumType<InT>::type (float for __half /
// __nv_bfloat16). The output element type OutT is parameterized separately so
// the two-phase full-reduction path can keep per-block partials in the float
// accumulator type between phases instead of narrowing to fp16/bf16 (which
// would discard the float-accumulation benefit). Only the FINAL scalar is
// narrowed to the storage type.
//
//   - Single-block path:    sum_reduce_kernel<T, T>      (read T,   write T)
//   - Two-phase phase 1:    sum_reduce_kernel<T, Acc>    (read T,   write float)
//   - Two-phase phase 2:    sum_reduce_kernel<Acc, T>    (read float, write T)
//
// For non-half T, Acc == T so all three instantiations behave identically to
// the original single-type kernel.
template<typename InT, typename OutT>
__global__ void sum_reduce_kernel(const InT* input, OutT* output, int64_t n) {
    using Acc = typename AccumType<InT>::type;
    __shared__ Acc shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Grid-stride loop with higher-precision accumulation
    Acc thread_sum = Acc(0);
    for (int64_t i = idx; i < n; i += grid_size) {
        thread_sum = thread_sum + Acc(input[i]);
    }

    shared[tid] = thread_sum;
    __syncthreads();

    // Block-level reduction in shared memory
    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = shared[tid] + shared[tid + stride];
        }
        __syncthreads();
    }
    __syncthreads();

    // Warp-level reduction
    if (tid < WARP_SIZE) {
        Acc val = shared[tid];
        val = warp_reduce_sum(val);

        if (tid == 0) {
            // Narrow to the storage type only at the final store.
            output[blockIdx.x] = OutT(val);
        }
    }
}

template<typename T>
__global__ void max_reduce_kernel(const T* input, T* output, int64_t n) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or type-appropriate negative infinity
    T thread_max = (idx < n) ? input[idx] : sentinel_lowest<T>();

    // Grid-stride loop
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        T val = input[i];
        thread_max = cuda_max_val(val, thread_max);
    }

    shared[tid] = thread_max;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            T other = shared[tid + stride];
            shared[tid] = cuda_max_val(shared[tid], other);
        }
        __syncthreads();
    }
    __syncthreads();

    // Warp-level reduction
    if (tid < WARP_SIZE) {
        T val = shared[tid];
        val = warp_reduce_max(val);

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

template<typename T>
__global__ void min_reduce_kernel(const T* input, T* output, int64_t n) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or type-appropriate positive infinity
    T thread_min = (idx < n) ? input[idx] : sentinel_max<T>();

    // Grid-stride loop
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        T val = input[i];
        thread_min = cuda_min_val(val, thread_min);
    }

    shared[tid] = thread_min;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            T other = shared[tid + stride];
            shared[tid] = cuda_min_val(shared[tid], other);
        }
        __syncthreads();
    }
    __syncthreads();

    // Warp-level reduction
    if (tid < WARP_SIZE) {
        T val = shared[tid];
        val = warp_reduce_min(val);

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// ============================================================================
// Half-precision specializations for max/min reduction kernels
// ============================================================================

// Specialized max_reduce_kernel for __half
__global__ void max_reduce_kernel_half(const __half* input, __half* output, int64_t n) {
    __shared__ __half shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or negative infinity
    __half thread_max = (idx < n) ? input[idx] : half_neg_inf();

    // Grid-stride loop
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        __half val = input[i];
        thread_max = cuda_max_val(val, thread_max);
    }

    shared[tid] = thread_max;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            __half other = shared[tid + stride];
            shared[tid] = cuda_max_val(shared[tid], other);
        }
        __syncthreads();
    }
    __syncthreads();

    // Warp-level reduction
    if (tid < WARP_SIZE) {
        __half val = shared[tid];
        val = warp_reduce_max(val);

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// Specialized min_reduce_kernel for __half
__global__ void min_reduce_kernel_half(const __half* input, __half* output, int64_t n) {
    __shared__ __half shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or positive infinity
    __half thread_min = (idx < n) ? input[idx] : half_pos_inf();

    // Grid-stride loop
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        __half val = input[i];
        thread_min = cuda_min_val(val, thread_min);
    }

    shared[tid] = thread_min;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            __half other = shared[tid + stride];
            shared[tid] = cuda_min_val(shared[tid], other);
        }
        __syncthreads();
    }
    __syncthreads();

    // Warp-level reduction
    if (tid < WARP_SIZE) {
        __half val = shared[tid];
        val = warp_reduce_min(val);

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// ============================================================================
// BFloat16 specializations for max/min reduction kernels
// ============================================================================

// Specialized max_reduce_kernel for __nv_bfloat16
__global__ void max_reduce_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    __shared__ __nv_bfloat16 shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or negative infinity
    __nv_bfloat16 thread_max = (idx < n) ? input[idx] : bfloat16_neg_inf();

    // Grid-stride loop
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        __nv_bfloat16 val = input[i];
        thread_max = cuda_max_val(val, thread_max);
    }

    shared[tid] = thread_max;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            __nv_bfloat16 other = shared[tid + stride];
            shared[tid] = cuda_max_val(shared[tid], other);
        }
        __syncthreads();
    }
    __syncthreads();

    // Warp-level reduction
    if (tid < WARP_SIZE) {
        __nv_bfloat16 val = shared[tid];
        val = warp_reduce_max(val);

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// Specialized min_reduce_kernel for __nv_bfloat16
__global__ void min_reduce_kernel_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, int64_t n) {
    __shared__ __nv_bfloat16 shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or positive infinity
    __nv_bfloat16 thread_min = (idx < n) ? input[idx] : bfloat16_pos_inf();

    // Grid-stride loop
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        __nv_bfloat16 val = input[i];
        thread_min = cuda_min_val(val, thread_min);
    }

    shared[tid] = thread_min;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            __nv_bfloat16 other = shared[tid + stride];
            shared[tid] = cuda_min_val(shared[tid], other);
        }
        __syncthreads();
    }
    __syncthreads();

    // Warp-level reduction
    if (tid < WARP_SIZE) {
        __nv_bfloat16 val = shared[tid];
        val = warp_reduce_min(val);

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// ============================================================================
// Dimensional reduction kernels
// ============================================================================

// Sum reduction along a specific dimension
template<typename T>
__global__ void sum_along_dim_kernel(
    const T* input,
    T* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    // Grid-stride over output positions so a grid clamped to maxGridDim still
    // covers every output slice (int64-safe for output_size > INT_MAX*block).
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    // Compute multi-dimensional indices for output position.
    // Decompose in reverse order (innermost dim first) to match row-major layout.
    int64_t indices[DIM_META_MAX_RANK];  // Support up to DIM_META_MAX_RANK dims (audit F.12)
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Sum along the reduction dimension with higher-precision accumulation
    using Acc = typename AccumType<T>::type;
    Acc sum = Acc(0);
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;

        // Compute flat index
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }

        sum = sum + Acc(input[in_idx]);
    }

    output[out_idx] = T(sum);
    }
}

// Max reduction along a specific dimension
template<typename T>
__global__ void max_along_dim_kernel(
    const T* input,
    T* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    // Compute multi-dimensional indices
    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find max along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    T max_val = input[in_idx];

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        T val = input[in_idx];
        max_val = cuda_max_val(val, max_val);
    }

    output[out_idx] = max_val;
    }
}

// Min reduction along a specific dimension
template<typename T>
__global__ void min_along_dim_kernel(
    const T* input,
    T* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    // Compute multi-dimensional indices
    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find min along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    T min_val = input[in_idx];

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        T val = input[in_idx];
        min_val = cuda_min_val(val, min_val);
    }

    output[out_idx] = min_val;
    }
}

// Specialized max_along_dim_kernel for __half
__global__ void max_along_dim_kernel_half(
    const __half* input,
    __half* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    // Compute multi-dimensional indices
    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find max along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    __half max_val = input[in_idx];

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        __half val = input[in_idx];
        max_val = cuda_max_val(val, max_val);
    }

    output[out_idx] = max_val;
    }
}

// Specialized min_along_dim_kernel for __half
__global__ void min_along_dim_kernel_half(
    const __half* input,
    __half* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    // Compute multi-dimensional indices
    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find min along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    __half min_val = input[in_idx];

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        __half val = input[in_idx];
        min_val = cuda_min_val(val, min_val);
    }

    output[out_idx] = min_val;
    }
}

// Specialized max_along_dim_kernel for __nv_bfloat16
__global__ void max_along_dim_kernel_bf16(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    // Compute multi-dimensional indices
    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find max along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    __nv_bfloat16 max_val = input[in_idx];

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        __nv_bfloat16 val = input[in_idx];
        max_val = cuda_max_val(val, max_val);
    }

    output[out_idx] = max_val;
    }
}

// Specialized min_along_dim_kernel for __nv_bfloat16
__global__ void min_along_dim_kernel_bf16(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    // Compute multi-dimensional indices
    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find min along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    __nv_bfloat16 min_val = input[in_idx];

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        __nv_bfloat16 val = input[in_idx];
        min_val = cuda_min_val(val, min_val);
    }

    output[out_idx] = min_val;
    }
}

// ============================================================================
// Helper functions
// ============================================================================

static auto compute_reduction_shape(
    const std::vector<int64_t>& input_shape,
    int64_t dim,
    bool keepdim
) -> std::vector<int64_t> {
    if (dim == INT64_MIN) {
        // Full reduction
        if (keepdim) {
            return std::vector<int64_t>(input_shape.size(), 1);
        }
        return {};  // Scalar (0D tensor)
    }

    // Normalize negative dimension
    int64_t normalized_dim = dim;
    if (dim < 0) {
        normalized_dim = static_cast<int64_t>(input_shape.size()) + dim;
        if (normalized_dim < 0 || normalized_dim >= static_cast<int64_t>(input_shape.size())) {
            throw std::invalid_argument("Dimension out of range in compute_reduction_shape");
        }
    }

    std::vector<int64_t> output_shape = input_shape;
    if (keepdim) {
        output_shape[normalized_dim] = 1;
    } else {
        output_shape.erase(output_shape.begin() + normalized_dim);
        // Keep empty shape for scalar result
    }
    return output_shape;
}

template<typename T>
static void launch_full_reduction_sum(const T* d_input, T* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, T(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (n == 1) {
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(d_output, d_input, sizeof(T), cudaMemcpyDeviceToDevice, stream)); // audit V.17
        return;
    }

    // Two-phase reduction for large arrays
    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        sum_reduce_kernel<T, T><<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Phase 1: Reduce to num_blocks intermediate results, kept in the
        // accumulator type (float for __half / __nv_bfloat16) so the per-block
        // partials are NOT narrowed to low precision between phases. CPU
        // accumulates the whole reduction in float; this mirrors that.
        using Acc = typename AccumType<T>::type;
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(Acc));
        auto* d_temp = static_cast<Acc*>(d_temp_guard.get());
        sum_reduce_kernel<T, Acc><<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());

        // Phase 2: Final reduction reads the float partials, accumulates in
        // float, and narrows only the final scalar to the storage type T.
        sum_reduce_kernel<Acc, T><<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
    // Check for any errors during synchronization
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in sum reduction: ") + cudaGetErrorString(err));
    }
}

template<typename T>
static void launch_full_reduction_max(const T* d_input, T* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("max: input tensor is empty");
    }

    if (n == 1) {
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(d_output, d_input, sizeof(T), cudaMemcpyDeviceToDevice, stream)); // audit V.17
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        max_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(T));
        auto* d_temp = static_cast<T*>(d_temp_guard.get());
        max_reduce_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());
        max_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

template<typename T>
static void launch_full_reduction_min(const T* d_input, T* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("min: input tensor is empty");
    }

    if (n == 1) {
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(d_output, d_input, sizeof(T), cudaMemcpyDeviceToDevice, stream)); // audit V.17
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        min_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(T));
        auto* d_temp = static_cast<T*>(d_temp_guard.get());
        min_reduce_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());
        min_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

template<typename T>
static void launch_dim_reduction_sum(
    const T* d_input,
    T* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    cudaStream_t stream = nullptr
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    // Compute output size
    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    // Launch kernel
    int num_blocks = compute_grid_size(output_size, REDUCTION_BLOCK_SIZE);
    sum_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
static void launch_dim_reduction_max(
    const T* d_input,
    T* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    cudaStream_t stream = nullptr
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = compute_grid_size(output_size, REDUCTION_BLOCK_SIZE);
    max_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
static void launch_dim_reduction_min(
    const T* d_input,
    T* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    cudaStream_t stream = nullptr
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = compute_grid_size(output_size, REDUCTION_BLOCK_SIZE);
    min_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Half-precision specialized launch functions
// ============================================================================

static void launch_full_reduction_max_half(const __half* d_input, __half* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("max: input tensor is empty");
    }

    if (n == 1) {
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(d_output, d_input, sizeof(__half), cudaMemcpyDeviceToDevice, stream)); // audit V.17
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        max_reduce_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(__half));
        auto* d_temp = static_cast<__half*>(d_temp_guard.get());
        max_reduce_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());
        max_reduce_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void launch_full_reduction_min_half(const __half* d_input, __half* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("min: input tensor is empty");
    }

    if (n == 1) {
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(d_output, d_input, sizeof(__half), cudaMemcpyDeviceToDevice, stream)); // audit V.17
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        min_reduce_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(__half));
        auto* d_temp = static_cast<__half*>(d_temp_guard.get());
        min_reduce_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());
        min_reduce_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void launch_dim_reduction_max_half(
    const __half* d_input,
    __half* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    cudaStream_t stream = nullptr
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = compute_grid_size(output_size, REDUCTION_BLOCK_SIZE);
    max_along_dim_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

static void launch_dim_reduction_min_half(
    const __half* d_input,
    __half* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    cudaStream_t stream = nullptr
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = compute_grid_size(output_size, REDUCTION_BLOCK_SIZE);
    min_along_dim_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// BFloat16 specialized launch functions
// ============================================================================

static void launch_full_reduction_max_bf16(const __nv_bfloat16* d_input, __nv_bfloat16* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("max: input tensor is empty");
    }

    if (n == 1) {
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(d_output, d_input, sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream)); // audit V.17
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        max_reduce_kernel_bf16<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(__nv_bfloat16));
        auto* d_temp = static_cast<__nv_bfloat16*>(d_temp_guard.get());
        max_reduce_kernel_bf16<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());
        max_reduce_kernel_bf16<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void launch_full_reduction_min_bf16(const __nv_bfloat16* d_input, __nv_bfloat16* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("min: input tensor is empty");
    }

    if (n == 1) {
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(d_output, d_input, sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream)); // audit V.17
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        min_reduce_kernel_bf16<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(__nv_bfloat16));
        auto* d_temp = static_cast<__nv_bfloat16*>(d_temp_guard.get());
        min_reduce_kernel_bf16<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());
        min_reduce_kernel_bf16<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void launch_dim_reduction_max_bf16(
    const __nv_bfloat16* d_input,
    __nv_bfloat16* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    cudaStream_t stream = nullptr
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = compute_grid_size(output_size, REDUCTION_BLOCK_SIZE);
    max_along_dim_kernel_bf16<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

static void launch_dim_reduction_min_bf16(
    const __nv_bfloat16* d_input,
    __nv_bfloat16* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    cudaStream_t stream = nullptr
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = compute_grid_size(output_size, REDUCTION_BLOCK_SIZE);
    min_along_dim_kernel_bf16<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Public API
// ============================================================================

auto sum_kernel(const Tensor& input_raw, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    // audit-2026-05-03 bug #2: ensure contiguous input — same root cause as
    // the CPU sum_kernel fix. Non-contiguous slice/expand views were being
    // read via a flat pointer, skipping logical elements.
    auto input = input_raw.contiguous();
    auto [resolved_stream, stream_guard] = resolve_stream(stream, input);
    stream = resolved_stream;

    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize negative dimension
    int64_t normalized_dim = dim;
    if (dim != INT64_MIN) {  // INT64_MIN means reduce all dimensions
        if (dim < 0) {
            normalized_dim = ndim + dim;
        }
        // Validate dimension is within bounds
        if (normalized_dim < 0 || normalized_dim >= ndim) {
            throw std::runtime_error("Dimension " + std::to_string(dim) +
                " out of range for tensor with " + std::to_string(ndim) + " dimensions");
        }
    }

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        normalized_dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            if (dim == INT64_MIN) {  // Full reduction
                launch_full_reduction_sum(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_sum(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (dim == INT64_MIN) {  // Full reduction
                launch_full_reduction_sum(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_sum(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int32_t>();

            if (dim == INT64_MIN) {  // Full reduction
                launch_full_reduction_sum(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_sum(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {  // Full reduction
                launch_full_reduction_sum(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_sum(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::Float16: {
            auto* input_data = reinterpret_cast<const __half*>(input.data_ptr());
            auto* output_data = reinterpret_cast<__half*>(output.data_ptr());

            if (dim == INT64_MIN) {  // Full reduction
                launch_full_reduction_sum(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_sum(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::BFloat16: {
            auto* input_data = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
            auto* output_data = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

            if (dim == INT64_MIN) {  // Full reduction
                launch_full_reduction_sum(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_sum(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::Int8:   { auto* in = input.data<int8_t>();   auto* out = output.data<int8_t>();   if (dim == INT64_MIN) launch_full_reduction_sum(in, out, input.numel(), stream); else launch_dim_reduction_sum(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        case DType::Int16:  { auto* in = input.data<int16_t>();  auto* out = output.data<int16_t>();  if (dim == INT64_MIN) launch_full_reduction_sum(in, out, input.numel(), stream); else launch_dim_reduction_sum(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        case DType::UInt8:  { auto* in = input.data<uint8_t>();  auto* out = output.data<uint8_t>();  if (dim == INT64_MIN) launch_full_reduction_sum(in, out, input.numel(), stream); else launch_dim_reduction_sum(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        case DType::UInt16: { auto* in = input.data<uint16_t>(); auto* out = output.data<uint16_t>(); if (dim == INT64_MIN) launch_full_reduction_sum(in, out, input.numel(), stream); else launch_dim_reduction_sum(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        case DType::UInt32: { auto* in = input.data<uint32_t>(); auto* out = output.data<uint32_t>(); if (dim == INT64_MIN) launch_full_reduction_sum(in, out, input.numel(), stream); else launch_dim_reduction_sum(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        case DType::UInt64: { auto* in = input.data<uint64_t>(); auto* out = output.data<uint64_t>(); if (dim == INT64_MIN) launch_full_reduction_sum(in, out, input.numel(), stream); else launch_dim_reduction_sum(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        case DType::Complex64:
        case DType::Complex128: {
            // Complex sum: reduce real and imaginary parts independently. This
            // avoids having to define AccumType/operator+ for cuFloatComplex /
            // cuDoubleComplex in every reduction template. Interleaved layout
            // (re0, im0, re1, im1, …) lets us treat each component as a
            // strided real scalar.
            const bool is64 = (dtype == DType::Complex128);
            const int64_t n_complex = input.numel();
            auto run = [&](auto real_tag) {
                using R = decltype(real_tag);
                const R* in_re_im = reinterpret_cast<const R*>(input.data_ptr());
                R* out_re_im     = reinterpret_cast<R*>(output.data_ptr());

                if (dim == INT64_MIN) {
                    // Full reduction: view input as (n_complex, 2), reduce dim 0.
                    std::vector<int64_t> view_shape  = {n_complex, int64_t(2)};
                    std::vector<int64_t> view_strides = {int64_t(2), int64_t(1)};
                    launch_dim_reduction_sum(
                        in_re_im, out_re_im,
                        view_shape, view_strides, /*dim=*/0, stream);
                } else {
                    // Dim reduction: add a trailing "2" dim of stride 1, and
                    // double every other stride so the existing real-valued
                    // dim-reduction kernel walks real and imaginary lanes
                    // independently.
                    std::vector<int64_t> view_shape(input_shape.begin(), input_shape.end());
                    std::vector<int64_t> view_strides(input_strides.begin(), input_strides.end());
                    for (auto& s : view_strides) s *= 2;
                    view_shape.push_back(2);
                    view_strides.push_back(1);
                    launch_dim_reduction_sum(
                        in_re_im, out_re_im,
                        view_shape, view_strides, normalized_dim, stream);
                }
            };
            if (is64) run(double{});
            else      run(float{});
            break;
        }
        default:
            throw std::runtime_error("sum: unsupported dtype");
    }

    CUDA_PEEK_AND_THROW(stream, "sum_kernel");

    return output;
}

// Scaling kernel for mean computation
template<typename T>
__global__ void scale_kernel(T* data, int64_t n, T scale) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

// Specialization for __half (no *= operator)
template<>
__global__ void scale_kernel<__half>(__half* data, int64_t n, __half scale) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = __float2half(__half2float(data[idx]) * __half2float(scale));
    }
}

// Specialization for __nv_bfloat16 (no *= operator)
template<>
__global__ void scale_kernel<__nv_bfloat16>(__nv_bfloat16* data, int64_t n, __nv_bfloat16 scale) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = __float2bfloat16(__bfloat162float(data[idx]) * __bfloat162float(scale));
    }
}

auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    auto [resolved_stream, stream_guard] = resolve_stream(stream, input);
    stream = resolved_stream;

    const auto dtype = input.dtype();

    if (dtype != DType::Float32 && dtype != DType::Float64 && dtype != DType::Float16 && dtype != DType::BFloat16 &&
        dtype != DType::Complex64 && dtype != DType::Complex128) {
        throw std::runtime_error("mean: only Float32, Float64, Float16, BFloat16, Complex64, and Complex128 are supported");
    }

    // Normalize negative dim (INT64_MIN sentinel means full reduction).
    // audit-2026-05-03 bug #5 (CUDA-side): previously treated all negative
    // dims as "reduce all", scaling by numel() instead of shape[ndim+dim] —
    // gave wrong gradients on dim=-1 / -2 calls. Mirrors the max_kernel
    // pattern below.
    int64_t dim_norm = dim;
    if (dim_norm < 0 && dim_norm != INT64_MIN) {
        dim_norm += input.ndim();
    }

    // Compute sum first (sum_kernel already normalizes negative dim correctly).
    auto sum_result = sum_kernel(input, dim, keepdim, stream);

    // Compute count
    int64_t count;
    if (dim_norm == INT64_MIN || dim == INT64_MIN) {
        count = input.numel();
    } else {
        count = input.shape()[dim_norm];
    }

    if (count == 0) {
        // Return 0 for mean of empty tensor - this is practical for loss computation
        // in object detection where no samples may be selected
        return sum_result;  // sum_result is already 0 for empty tensor
    }

    // Divide by count (in-place) using proper CUDA kernel
    const int64_t n = sum_result.numel();
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    if (dtype == DType::Float32) {
        auto* data = sum_result.data<float>();
        const float scale = 1.0f / static_cast<float>(count);
        scale_kernel<<<grid_size, block_size, 0, stream>>>(data, n, scale);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float64) {
        auto* data = sum_result.data<double>();
        const double scale = 1.0 / static_cast<double>(count);
        scale_kernel<<<grid_size, block_size, 0, stream>>>(data, n, scale);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Float16) {
        auto* data = reinterpret_cast<__half*>(sum_result.data_ptr());
        const __half scale = __float2half(1.0f / static_cast<float>(count));
        scale_kernel<<<grid_size, block_size, 0, stream>>>(data, n, scale);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::BFloat16) {
        auto* data = reinterpret_cast<__nv_bfloat16*>(sum_result.data_ptr());
        const __nv_bfloat16 scale = __float2bfloat16(1.0f / static_cast<float>(count));
        scale_kernel<<<grid_size, block_size, 0, stream>>>(data, n, scale);
        CUDA_CHECK(cudaGetLastError());
    } else if (dtype == DType::Complex64) {
        // Scale real & imag independently as 2*n contiguous floats.
        auto* data = reinterpret_cast<float*>(sum_result.data_ptr());
        const float scale = 1.0f / static_cast<float>(count);
        const int64_t n2 = 2 * n;
        const int grid2 = (n2 + block_size - 1) / block_size;
        scale_kernel<<<grid2, block_size, 0, stream>>>(data, n2, scale);
        CUDA_CHECK(cudaGetLastError());
    } else {  // Complex128
        auto* data = reinterpret_cast<double*>(sum_result.data_ptr());
        const double scale = 1.0 / static_cast<double>(count);
        const int64_t n2 = 2 * n;
        const int grid2 = (n2 + block_size - 1) / block_size;
        scale_kernel<<<grid2, block_size, 0, stream>>>(data, n2, scale);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_PEEK_AND_THROW(stream, "mean_kernel");

    return sum_result;
}

auto max_kernel(const Tensor& input_raw, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    // Ensure contiguous input — same root cause as the sum_kernel fix
    // (audit-2026-05-03 bug #2). The full-reduction path reads the buffer via a
    // flat pointer, so non-contiguous slice/expand/transpose views would skip
    // logical elements and produce wrong results.
    auto input = input_raw.contiguous();
    auto [resolved_stream, stream_guard] = resolve_stream(stream, input);
    stream = resolved_stream;

    // Normalize negative dim (but INT64_MIN means full reduction)
    if (dim < 0 && dim != INT64_MIN) {
        dim += input.ndim();
    }

    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            if (dim < 0) {
                launch_full_reduction_max(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_max(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream  // pass the resolved stream; default was stream 0
                );
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (dim < 0) {
                launch_full_reduction_max(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_max(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream  // pass the resolved stream; default was stream 0
                );
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int32_t>();

            if (dim < 0) {
                launch_full_reduction_max(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_max(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream  // pass the resolved stream; default was stream 0
                );
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (dim < 0) {
                launch_full_reduction_max(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_max(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream  // pass the resolved stream; default was stream 0
                );
            }
            break;
        }
        case DType::Float16: {
            auto* input_data = reinterpret_cast<const __half*>(input.data_ptr());
            auto* output_data = reinterpret_cast<__half*>(output.data_ptr());

            if (dim < 0) {
                launch_full_reduction_max_half(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_max_half(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream  // pass the resolved stream; default was stream 0
                );
            }
            break;
        }
        case DType::BFloat16: {
            auto* input_data = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
            auto* output_data = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

            if (dim < 0) {
                launch_full_reduction_max_bf16(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_max_bf16(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream  // pass the resolved stream; default was stream 0
                );
            }
            break;
        }
        default:
            throw std::runtime_error("max: unsupported dtype");
    }

    CUDA_PEEK_AND_THROW(stream, "max_kernel");

    return output;
}

auto min_kernel(const Tensor& input_raw, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    // Ensure contiguous input — same root cause as the sum_kernel fix
    // (audit-2026-05-03 bug #2). The full-reduction path reads the buffer via a
    // flat pointer, so non-contiguous views would skip logical elements.
    auto input = input_raw.contiguous();
    auto [resolved_stream, stream_guard] = resolve_stream(stream, input);
    stream = resolved_stream;

    // Normalize negative dim (but INT64_MIN means full reduction)
    if (dim < 0 && dim != INT64_MIN) {
        dim += input.ndim();
    }

    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            if (dim < 0) {
                launch_full_reduction_min(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_min(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream  // pass the resolved stream; default was stream 0
                );
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (dim < 0) {
                launch_full_reduction_min(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_min(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream  // pass the resolved stream; default was stream 0
                );
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int32_t>();

            if (dim < 0) {
                launch_full_reduction_min(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_min(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream  // pass the resolved stream; default was stream 0
                );
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (dim < 0) {
                launch_full_reduction_min(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_min(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream  // pass the resolved stream; default was stream 0
                );
            }
            break;
        }
        case DType::Float16: {
            auto* input_data = reinterpret_cast<const __half*>(input.data_ptr());
            auto* output_data = reinterpret_cast<__half*>(output.data_ptr());

            if (dim < 0) {
                launch_full_reduction_min_half(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_min_half(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream  // pass the resolved stream; default was stream 0
                );
            }
            break;
        }
        case DType::BFloat16: {
            auto* input_data = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
            auto* output_data = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

            if (dim < 0) {
                launch_full_reduction_min_bf16(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_min_bf16(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    dim, stream  // pass the resolved stream; default was stream 0
                );
            }
            break;
        }
        default:
            throw std::runtime_error("min: unsupported dtype");
    }

    CUDA_PEEK_AND_THROW(stream, "min_kernel");

    return output;
}

// ============================================================================
// Argmax/Argmin operations (return indices, not values)
// ============================================================================

// Argmax kernel for full reduction
template<typename T>
__global__ void argmax_full_kernel(const T* input, int64_t* output, int64_t n) {
    __shared__ T shared_vals[REDUCTION_BLOCK_SIZE + 1];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE + 1];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or type-appropriate negative infinity
    T thread_max = (idx < n) ? input[idx] : sentinel_lowest<T>();
    int64_t thread_idx = (idx < n) ? idx : 0;

    // Grid-stride loop to find local maximum (NaN-aware, first-NaN/lowest-index)
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        T val = input[i];
        if (arg_max_takes_idx(val, i, thread_max, thread_idx)) {
            thread_max = val;
            thread_idx = i;
        }
    }

    shared_vals[tid] = thread_max;
    shared_idxs[tid] = thread_idx;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            if (arg_max_takes_idx(shared_vals[tid + stride], shared_idxs[tid + stride],
                                  shared_vals[tid], shared_idxs[tid])) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }
    __syncthreads();

    // Warp-level reduction (last 32 threads)
    if (tid < 32) {
        T val = shared_vals[tid];
        int64_t val_idx = shared_idxs[tid];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            T other_val = __shfl_down_sync(0xffffffff, val, offset);
            int64_t other_idx = __shfl_down_sync(0xffffffff, val_idx, offset);
            if (arg_max_takes_idx(other_val, other_idx, val, val_idx)) {
                val = other_val;
                val_idx = other_idx;
            }
        }

        if (tid == 0) {
            output[blockIdx.x] = val_idx;
        }
    }
}

// Argmin kernel for full reduction
template<typename T>
__global__ void argmin_full_kernel(const T* input, int64_t* output, int64_t n) {
    __shared__ T shared_vals[REDUCTION_BLOCK_SIZE + 1];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE + 1];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or type-appropriate positive infinity
    T thread_min = (idx < n) ? input[idx] : sentinel_max<T>();
    int64_t thread_idx = (idx < n) ? idx : 0;

    // Grid-stride loop to find local minimum (NaN-aware, first-NaN/lowest-index)
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        T val = input[i];
        if (arg_min_takes_idx(val, i, thread_min, thread_idx)) {
            thread_min = val;
            thread_idx = i;
        }
    }

    shared_vals[tid] = thread_min;
    shared_idxs[tid] = thread_idx;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            if (arg_min_takes_idx(shared_vals[tid + stride], shared_idxs[tid + stride],
                                  shared_vals[tid], shared_idxs[tid])) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }
    __syncthreads();

    // Warp-level reduction (last 32 threads)
    if (tid < 32) {
        T val = shared_vals[tid];
        int64_t val_idx = shared_idxs[tid];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            T other_val = __shfl_down_sync(0xffffffff, val, offset);
            int64_t other_idx = __shfl_down_sync(0xffffffff, val_idx, offset);
            if (arg_min_takes_idx(other_val, other_idx, val, val_idx)) {
                val = other_val;
                val_idx = other_idx;
            }
        }

        if (tid == 0) {
            output[blockIdx.x] = val_idx;
        }
    }
}

// Argmax along a specific dimension
template<typename T>
__global__ void argmax_along_dim_kernel(
    const T* input,
    int64_t* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    // Compute multi-dimensional indices
    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find argmax along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    T max_val = input[in_idx];
    int64_t max_idx = 0;

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        T val = input[in_idx];
        if (arg_max_takes(val, max_val)) {
            max_val = val;
            max_idx = i;
        }
    }

    output[out_idx] = max_idx;
    }
}

// Argmin along a specific dimension
template<typename T>
__global__ void argmin_along_dim_kernel(
    const T* input,
    int64_t* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    // Compute multi-dimensional indices
    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find argmin along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    T min_val = input[in_idx];
    int64_t min_idx = 0;

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        T val = input[in_idx];
        if (arg_min_takes(val, min_val)) {
            min_val = val;
            min_idx = i;
        }
    }

    output[out_idx] = min_idx;
    }
}

// ============================================================================
// Half-precision specializations for argmax/argmin kernels
// ============================================================================

// Specialized argmax_full_kernel for __half
__global__ void argmax_full_kernel_half(const __half* input, int64_t* output, int64_t n) {
    __shared__ __half shared_vals[REDUCTION_BLOCK_SIZE + 1];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE + 1];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or negative infinity
    __half thread_max = (idx < n) ? input[idx] : half_neg_inf();
    int64_t thread_idx = (idx < n) ? idx : 0;

    // Grid-stride loop to find local maximum (NaN-aware, first-NaN/lowest-index)
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        __half val = input[i];
        if (arg_max_takes_idx(val, i, thread_max, thread_idx)) {
            thread_max = val;
            thread_idx = i;
        }
    }

    shared_vals[tid] = thread_max;
    shared_idxs[tid] = thread_idx;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            if (arg_max_takes_idx(shared_vals[tid + stride], shared_idxs[tid + stride],
                                  shared_vals[tid], shared_idxs[tid])) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }
    __syncthreads();

    // Warp-level reduction (last 32 threads)
    if (tid < 32) {
        __half val = shared_vals[tid];
        int64_t val_idx = shared_idxs[tid];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            __half other_val = __shfl_down_sync(0xffffffff, val, offset);
            int64_t other_idx = __shfl_down_sync(0xffffffff, val_idx, offset);
            if (arg_max_takes_idx(other_val, other_idx, val, val_idx)) {
                val = other_val;
                val_idx = other_idx;
            }
        }

        if (tid == 0) {
            output[blockIdx.x] = val_idx;
        }
    }
}

// Specialized argmin_full_kernel for __half
__global__ void argmin_full_kernel_half(const __half* input, int64_t* output, int64_t n) {
    __shared__ __half shared_vals[REDUCTION_BLOCK_SIZE + 1];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE + 1];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or positive infinity
    __half thread_min = (idx < n) ? input[idx] : half_pos_inf();
    int64_t thread_idx = (idx < n) ? idx : 0;

    // Grid-stride loop to find local minimum (NaN-aware, first-NaN/lowest-index)
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        __half val = input[i];
        if (arg_min_takes_idx(val, i, thread_min, thread_idx)) {
            thread_min = val;
            thread_idx = i;
        }
    }

    shared_vals[tid] = thread_min;
    shared_idxs[tid] = thread_idx;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            if (arg_min_takes_idx(shared_vals[tid + stride], shared_idxs[tid + stride],
                                  shared_vals[tid], shared_idxs[tid])) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }
    __syncthreads();

    // Warp-level reduction (last 32 threads)
    if (tid < 32) {
        __half val = shared_vals[tid];
        int64_t val_idx = shared_idxs[tid];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            __half other_val = __shfl_down_sync(0xffffffff, val, offset);
            int64_t other_idx = __shfl_down_sync(0xffffffff, val_idx, offset);
            if (arg_min_takes_idx(other_val, other_idx, val, val_idx)) {
                val = other_val;
                val_idx = other_idx;
            }
        }

        if (tid == 0) {
            output[blockIdx.x] = val_idx;
        }
    }
}

// Specialized argmax_along_dim_kernel for __half
__global__ void argmax_along_dim_kernel_half(
    const __half* input,
    int64_t* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    // Compute multi-dimensional indices
    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find argmax along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    __half max_val = input[in_idx];
    int64_t max_idx = 0;

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        __half val = input[in_idx];
        if (arg_max_takes(val, max_val)) {
            max_val = val;
            max_idx = i;
        }
    }

    output[out_idx] = max_idx;
    }
}

// Specialized argmin_along_dim_kernel for __half
__global__ void argmin_along_dim_kernel_half(
    const __half* input,
    int64_t* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    // Compute multi-dimensional indices
    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find argmin along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    __half min_val = input[in_idx];
    int64_t min_idx = 0;

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        __half val = input[in_idx];
        if (arg_min_takes(val, min_val)) {
            min_val = val;
            min_idx = i;
        }
    }

    output[out_idx] = min_idx;
    }
}

// ============================================================================
// BFloat16 specializations for argmax/argmin kernels
// ============================================================================

// BFloat16 final kernel for argmax
__global__ void argmax_final_kernel_bf16(const __nv_bfloat16* input, const int64_t* block_indices, int64_t* output, int num_blocks) {
    __shared__ float shared_vals[REDUCTION_BLOCK_SIZE + 1];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE + 1];

    int tid = threadIdx.x;
    float best_val;
    int64_t best_idx;

    if (tid < num_blocks) {
        best_idx = block_indices[tid];
        best_val = __bfloat162float(input[best_idx]);
    } else {
        best_val = -FLT_MAX;
        best_idx = 0;
    }

    for (int i = tid + blockDim.x; i < num_blocks; i += blockDim.x) {
        int64_t idx = block_indices[i];
        float val = __bfloat162float(input[idx]);
        if (arg_max_takes_idx(val, idx, best_val, best_idx)) {
            best_val = val;
            best_idx = idx;
        }
    }

    shared_vals[tid] = best_val;
    shared_idxs[tid] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (arg_max_takes_idx(shared_vals[tid + stride], shared_idxs[tid + stride],
                                  shared_vals[tid], shared_idxs[tid])) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = shared_idxs[0];
    }
}

// BFloat16 final kernel for argmin
__global__ void argmin_final_kernel_bf16(const __nv_bfloat16* input, const int64_t* block_indices, int64_t* output, int num_blocks) {
    __shared__ float shared_vals[REDUCTION_BLOCK_SIZE + 1];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE + 1];

    int tid = threadIdx.x;
    float best_val;
    int64_t best_idx;

    if (tid < num_blocks) {
        best_idx = block_indices[tid];
        best_val = __bfloat162float(input[best_idx]);
    } else {
        best_val = FLT_MAX;
        best_idx = 0;
    }

    for (int i = tid + blockDim.x; i < num_blocks; i += blockDim.x) {
        int64_t idx = block_indices[i];
        float val = __bfloat162float(input[idx]);
        if (arg_min_takes_idx(val, idx, best_val, best_idx)) {
            best_val = val;
            best_idx = idx;
        }
    }

    shared_vals[tid] = best_val;
    shared_idxs[tid] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (arg_min_takes_idx(shared_vals[tid + stride], shared_idxs[tid + stride],
                                  shared_vals[tid], shared_idxs[tid])) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = shared_idxs[0];
    }
}

// Specialized argmax_full_kernel for __nv_bfloat16
__global__ void argmax_full_kernel_bf16(const __nv_bfloat16* input, int64_t* output, int64_t n) {
    __shared__ __nv_bfloat16 shared_vals[REDUCTION_BLOCK_SIZE + 1];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE + 1];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or negative infinity
    __nv_bfloat16 thread_max = (idx < n) ? input[idx] : bfloat16_neg_inf();
    int64_t thread_idx = (idx < n) ? idx : 0;

    // Grid-stride loop to find local maximum (NaN-aware, first-NaN/lowest-index)
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        __nv_bfloat16 val = input[i];
        if (arg_max_takes_idx(val, i, thread_max, thread_idx)) {
            thread_max = val;
            thread_idx = i;
        }
    }

    shared_vals[tid] = thread_max;
    shared_idxs[tid] = thread_idx;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            if (arg_max_takes_idx(shared_vals[tid + stride], shared_idxs[tid + stride],
                                  shared_vals[tid], shared_idxs[tid])) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }
    __syncthreads();

    // Warp-level reduction (last 32 threads)
    if (tid < 32) {
        __nv_bfloat16 val = shared_vals[tid];
        int64_t val_idx = shared_idxs[tid];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            __nv_bfloat16 other_val = __shfl_down_sync(0xffffffff, val, offset);
            int64_t other_idx = __shfl_down_sync(0xffffffff, val_idx, offset);
            if (arg_max_takes_idx(other_val, other_idx, val, val_idx)) {
                val = other_val;
                val_idx = other_idx;
            }
        }

        if (tid == 0) {
            output[blockIdx.x] = val_idx;
        }
    }
}

// Specialized argmin_full_kernel for __nv_bfloat16
__global__ void argmin_full_kernel_bf16(const __nv_bfloat16* input, int64_t* output, int64_t n) {
    __shared__ __nv_bfloat16 shared_vals[REDUCTION_BLOCK_SIZE + 1];
    __shared__ int64_t shared_idxs[REDUCTION_BLOCK_SIZE + 1];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with first element or positive infinity
    __nv_bfloat16 thread_min = (idx < n) ? input[idx] : bfloat16_pos_inf();
    int64_t thread_idx = (idx < n) ? idx : 0;

    // Grid-stride loop to find local minimum (NaN-aware, first-NaN/lowest-index)
    for (int64_t i = idx + grid_size; i < n; i += grid_size) {
        __nv_bfloat16 val = input[i];
        if (arg_min_takes_idx(val, i, thread_min, thread_idx)) {
            thread_min = val;
            thread_idx = i;
        }
    }

    shared_vals[tid] = thread_min;
    shared_idxs[tid] = thread_idx;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            if (arg_min_takes_idx(shared_vals[tid + stride], shared_idxs[tid + stride],
                                  shared_vals[tid], shared_idxs[tid])) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        __syncthreads();
    }
    __syncthreads();

    // Warp-level reduction (last 32 threads)
    if (tid < 32) {
        __nv_bfloat16 val = shared_vals[tid];
        int64_t val_idx = shared_idxs[tid];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            __nv_bfloat16 other_val = __shfl_down_sync(0xffffffff, val, offset);
            int64_t other_idx = __shfl_down_sync(0xffffffff, val_idx, offset);
            if (arg_min_takes_idx(other_val, other_idx, val, val_idx)) {
                val = other_val;
                val_idx = other_idx;
            }
        }

        if (tid == 0) {
            output[blockIdx.x] = val_idx;
        }
    }
}

// Specialized argmax_along_dim_kernel for __nv_bfloat16
__global__ void argmax_along_dim_kernel_bf16(
    const __nv_bfloat16* input,
    int64_t* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    // Compute multi-dimensional indices
    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find argmax along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    __nv_bfloat16 max_val = input[in_idx];
    int64_t max_idx = 0;

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        __nv_bfloat16 val = input[in_idx];
        if (arg_max_takes(val, max_val)) {
            max_val = val;
            max_idx = i;
        }
    }

    output[out_idx] = max_idx;
    }
}

// Specialized argmin_along_dim_kernel for __nv_bfloat16
__global__ void argmin_along_dim_kernel_bf16(
    const __nv_bfloat16* input,
    int64_t* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    // Compute multi-dimensional indices
    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Find argmin along the reduction dimension
    indices[dim] = 0;
    int64_t in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        in_idx += indices[d] * meta.strides[d];
    }
    __nv_bfloat16 min_val = input[in_idx];
    int64_t min_idx = 0;

    for (int64_t i = 1; i < dim_size; i++) {
        indices[dim] = i;
        in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        __nv_bfloat16 val = input[in_idx];
        if (arg_min_takes(val, min_val)) {
            min_val = val;
            min_idx = i;
        }
    }

    output[out_idx] = min_idx;
    }
}

// Helper function to launch argmax reduction
template<typename T>
static void launch_full_argmax(const T* d_input, int64_t* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("argmax: input tensor is empty");
    }

    if (n == 1) {
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, int64_t(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        argmax_full_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Two-phase GPU-only reduction
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(int64_t));
        auto* d_temp = static_cast<int64_t*>(d_temp_guard.get());
        argmax_full_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());

        // Phase 2: compare block results entirely on GPU
        argmax_final_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}
// Helper function to launch argmin reduction
template<typename T>
static void launch_full_argmin(const T* d_input, int64_t* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("argmin: input tensor is empty");
    }

    if (n == 1) {
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, int64_t(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        argmin_full_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Two-phase GPU-only reduction
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(int64_t));
        auto* d_temp = static_cast<int64_t*>(d_temp_guard.get());
        argmin_full_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());

        // Phase 2: compare block results entirely on GPU
        argmin_final_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}


// Helper function for dimensional argmax
template<typename T>
static void launch_dim_argmax(
    const T* d_input,
    int64_t* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    cudaStream_t stream = nullptr
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = compute_grid_size(output_size, REDUCTION_BLOCK_SIZE);
    argmax_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

// Helper function for dimensional argmin
template<typename T>
static void launch_dim_argmin(
    const T* d_input,
    int64_t* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    cudaStream_t stream = nullptr
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = compute_grid_size(output_size, REDUCTION_BLOCK_SIZE);
    argmin_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Half-precision specialized launch functions for argmax/argmin
// ============================================================================

static void launch_full_argmax_half(const __half* d_input, int64_t* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("argmax: input tensor is empty");
    }

    if (n == 1) {
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, int64_t(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        argmax_full_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Two-phase GPU-only reduction
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(int64_t));
        auto* d_temp = static_cast<int64_t*>(d_temp_guard.get());
        argmax_full_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());

        // Phase 2: compare block results entirely on GPU
        argmax_final_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void launch_full_argmin_half(const __half* d_input, int64_t* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("argmin: input tensor is empty");
    }

    if (n == 1) {
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, int64_t(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        argmin_full_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Two-phase GPU-only reduction
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(int64_t));
        auto* d_temp = static_cast<int64_t*>(d_temp_guard.get());
        argmin_full_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());

        // Phase 2: compare block results entirely on GPU
        argmin_final_kernel_half<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}


static void launch_dim_argmax_half(
    const __half* d_input,
    int64_t* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    cudaStream_t stream = nullptr
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = compute_grid_size(output_size, REDUCTION_BLOCK_SIZE);
    argmax_along_dim_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

static void launch_dim_argmin_half(
    const __half* d_input,
    int64_t* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    cudaStream_t stream = nullptr
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = compute_grid_size(output_size, REDUCTION_BLOCK_SIZE);
    argmin_along_dim_kernel_half<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// BFloat16 specialized launch functions for argmax/argmin
// ============================================================================

static void launch_full_argmax_bf16(const __nv_bfloat16* d_input, int64_t* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("argmax: input tensor is empty");
    }

    if (n == 1) {
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, int64_t(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        argmax_full_kernel_bf16<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Two-phase GPU-only reduction
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(int64_t));
        auto* d_temp = static_cast<int64_t*>(d_temp_guard.get());
        argmax_full_kernel_bf16<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());

        // Phase 2: compare block results entirely on GPU
        argmax_final_kernel_bf16<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void launch_full_argmin_bf16(const __nv_bfloat16* d_input, int64_t* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        throw std::runtime_error("argmin: input tensor is empty");
    }

    if (n == 1) {
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, int64_t(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        argmin_full_kernel_bf16<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Two-phase GPU-only reduction
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(int64_t));
        auto* d_temp = static_cast<int64_t*>(d_temp_guard.get());
        argmin_full_kernel_bf16<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());

        // Phase 2: compare block results entirely on GPU
        argmin_final_kernel_bf16<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void launch_dim_argmax_bf16(
    const __nv_bfloat16* d_input,
    int64_t* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    cudaStream_t stream = nullptr
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = compute_grid_size(output_size, REDUCTION_BLOCK_SIZE);
    argmax_along_dim_kernel_bf16<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

static void launch_dim_argmin_bf16(
    const __nv_bfloat16* d_input,
    int64_t* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    cudaStream_t stream = nullptr
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = compute_grid_size(output_size, REDUCTION_BLOCK_SIZE);
    argmin_along_dim_kernel_bf16<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

// Public API for argmax
auto argmax_kernel(const Tensor& input_raw, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    // Ensure contiguous input — same root cause as the sum_kernel fix
    // (audit-2026-05-03 bug #2). The full-reduction path reads the buffer via a
    // flat pointer; for non-contiguous views the value AND the returned flat
    // index would be meaningless.
    auto input = input_raw.contiguous();
    auto [resolved_stream, stream_guard] = resolve_stream(stream, input);
    stream = resolved_stream;

    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();

    // Normalize negative dimension
    int64_t normalized_dim = dim;
    if (dim < 0 && dim != INT64_MIN) {  // INT64_MIN means reduce all dimensions
        normalized_dim = static_cast<int64_t>(input_shape.size()) + dim;
        if (normalized_dim < 0 || normalized_dim >= static_cast<int64_t>(input_shape.size())) {
            throw std::invalid_argument("Dimension out of range in argmax_kernel");
        }
    }

    // Compute output shape (always Int64 for indices)
    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, DType::Int64, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmax(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmax(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmax(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmax(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmax(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmax(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmax(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmax(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::Float16: {
            auto* input_data = reinterpret_cast<const __half*>(input.data_ptr());
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmax_half(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmax_half(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::BFloat16: {
            auto* input_data = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmax_bf16(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmax_bf16(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::Int8:   { auto* in = input.data<int8_t>();   auto* out = output.data<int64_t>(); if (dim == INT64_MIN) launch_full_argmax(in, out, input.numel(), stream); else launch_dim_argmax(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        case DType::Int16:  { auto* in = input.data<int16_t>();  auto* out = output.data<int64_t>(); if (dim == INT64_MIN) launch_full_argmax(in, out, input.numel(), stream); else launch_dim_argmax(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        case DType::UInt8:  { auto* in = input.data<uint8_t>();  auto* out = output.data<int64_t>(); if (dim == INT64_MIN) launch_full_argmax(in, out, input.numel(), stream); else launch_dim_argmax(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        case DType::UInt16: { auto* in = input.data<uint16_t>(); auto* out = output.data<int64_t>(); if (dim == INT64_MIN) launch_full_argmax(in, out, input.numel(), stream); else launch_dim_argmax(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        case DType::UInt32: { auto* in = input.data<uint32_t>(); auto* out = output.data<int64_t>(); if (dim == INT64_MIN) launch_full_argmax(in, out, input.numel(), stream); else launch_dim_argmax(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        case DType::UInt64: { auto* in = input.data<uint64_t>(); auto* out = output.data<int64_t>(); if (dim == INT64_MIN) launch_full_argmax(in, out, input.numel(), stream); else launch_dim_argmax(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        case DType::Bool:   { auto* in = input.data<bool>();     auto* out = output.data<int64_t>(); if (dim == INT64_MIN) launch_full_argmax(in, out, input.numel(), stream); else launch_dim_argmax(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        default:
            throw std::runtime_error("argmax: unsupported dtype");
    }

    CUDA_PEEK_AND_THROW(stream, "argmax_kernel");

    return output;
}

// Public API for argmin
auto argmin_kernel(const Tensor& input_raw, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    // Ensure contiguous input — same root cause as the sum_kernel fix
    // (audit-2026-05-03 bug #2). The full-reduction path reads the buffer via a
    // flat pointer; for non-contiguous views the value AND the returned flat
    // index would be meaningless.
    auto input = input_raw.contiguous();
    auto [resolved_stream, stream_guard] = resolve_stream(stream, input);
    stream = resolved_stream;

    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();

    // Normalize negative dimension
    int64_t normalized_dim = dim;
    if (dim < 0 && dim != INT64_MIN) {  // INT64_MIN means reduce all dimensions
        normalized_dim = static_cast<int64_t>(input_shape.size()) + dim;
        if (normalized_dim < 0 || normalized_dim >= static_cast<int64_t>(input_shape.size())) {
            throw std::invalid_argument("Dimension out of range in argmin_kernel");
        }
    }

    // Compute output shape (always Int64 for indices)
    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, DType::Int64, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmin(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmin(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmin(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmin(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmin(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmin(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmin(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmin(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::Float16: {
            auto* input_data = reinterpret_cast<const __half*>(input.data_ptr());
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmin_half(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmin_half(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::BFloat16: {
            auto* input_data = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_argmin_bf16(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_argmin_bf16(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::Int8:   { auto* in = input.data<int8_t>();   auto* out = output.data<int64_t>(); if (dim == INT64_MIN) launch_full_argmin(in, out, input.numel(), stream); else launch_dim_argmin(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        case DType::Int16:  { auto* in = input.data<int16_t>();  auto* out = output.data<int64_t>(); if (dim == INT64_MIN) launch_full_argmin(in, out, input.numel(), stream); else launch_dim_argmin(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        case DType::UInt8:  { auto* in = input.data<uint8_t>();  auto* out = output.data<int64_t>(); if (dim == INT64_MIN) launch_full_argmin(in, out, input.numel(), stream); else launch_dim_argmin(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        case DType::UInt16: { auto* in = input.data<uint16_t>(); auto* out = output.data<int64_t>(); if (dim == INT64_MIN) launch_full_argmin(in, out, input.numel(), stream); else launch_dim_argmin(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        case DType::UInt32: { auto* in = input.data<uint32_t>(); auto* out = output.data<int64_t>(); if (dim == INT64_MIN) launch_full_argmin(in, out, input.numel(), stream); else launch_dim_argmin(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        case DType::UInt64: { auto* in = input.data<uint64_t>(); auto* out = output.data<int64_t>(); if (dim == INT64_MIN) launch_full_argmin(in, out, input.numel(), stream); else launch_dim_argmin(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        case DType::Bool:   { auto* in = input.data<bool>();     auto* out = output.data<int64_t>(); if (dim == INT64_MIN) launch_full_argmin(in, out, input.numel(), stream); else launch_dim_argmin(in, out, std::vector<int64_t>(input_shape.begin(), input_shape.end()), std::vector<int64_t>(input_strides.begin(), input_strides.end()), normalized_dim, stream); break; }
        default:
            throw std::runtime_error("argmin: unsupported dtype");
    }

    CUDA_PEEK_AND_THROW(stream, "argmin_kernel");

    return output;
}

// ============================================================================
// Product reduction operation
// ============================================================================

// Product reduction kernel (full reduction only)
template<typename T>
__global__ void prod_reduce_kernel(const T* input, T* output, int64_t n) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Initialize with 1 (identity for multiplication)
    T thread_prod = 1;

    // Grid-stride loop
    for (int64_t i = idx; i < n; i += grid_size) {
        thread_prod *= input[i];
    }

    shared[tid] = thread_prod;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            shared[tid] *= shared[tid + stride];
        }
        __syncthreads();
    }
    __syncthreads();

    // Warp-level reduction
    if (tid < WARP_SIZE) {
        T val = shared[tid];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val *= __shfl_down_sync(0xffffffff, val, offset);
        }

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// Product reduction along a specific dimension
template<typename T>
__global__ void prod_along_dim_kernel(
    const T* input,
    T* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    // Compute multi-dimensional indices
    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Product along the reduction dimension
    T prod_val = 1;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;

        // Compute flat index
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }

        prod_val *= input[in_idx];
    }

    output[out_idx] = prod_val;
    }
}

// Helper function to launch product reduction
template<typename T>
static void launch_full_reduction_prod(const T* d_input, T* d_output, int64_t n, cudaStream_t stream = nullptr) {
    if (n == 0) {
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, T(1));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (n == 1) {
        TENZOR_CUDA_CHECK(cudaMemcpyAsync(d_output, d_input, sizeof(T), cudaMemcpyDeviceToDevice, stream)); // audit V.17
        return;
    }

    // Two-phase reduction for large arrays
    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        prod_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Phase 1: Reduce to num_blocks intermediate results
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(T));
        auto* d_temp = static_cast<T*>(d_temp_guard.get());
        prod_reduce_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());

        // Phase 2: Final reduction
        prod_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Helper function for dimensional product reduction
template<typename T>
static void launch_dim_reduction_prod(
    const T* d_input,
    T* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    cudaStream_t stream = nullptr
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    if (output_size == 0 || dim_size == 0) {
        return;
    }

    DimMeta meta = make_dim_meta(input_shape, input_strides);

    int num_blocks = compute_grid_size(output_size, REDUCTION_BLOCK_SIZE);
    prod_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

// Public API for prod (product reduction)
auto prod_kernel(const Tensor& input_raw, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    // Ensure contiguous input — same root cause as the sum_kernel fix
    // (audit-2026-05-03 bug #2). The full-reduction path reads the buffer via a
    // flat pointer, so non-contiguous views would skip logical elements.
    auto input = input_raw.contiguous();
    auto [resolved_stream, stream_guard] = resolve_stream(stream, input);
    stream = resolved_stream;

    // Float16/BFloat16: widen to Float32 on device, compute, narrow back to the
    // input dtype (matches the CPU prod contract, audit E.2). The cast runs on GPU.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig = input.dtype();
        return prod_kernel(input.to(DType::Float32), dim, keepdim, stream).to(orig);
    }

    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();

    // Normalize negative dimension
    int64_t normalized_dim = dim;
    if (dim < 0 && dim != INT64_MIN) {  // INT64_MIN means reduce all dimensions
        normalized_dim = static_cast<int64_t>(input_shape.size()) + dim;
        if (normalized_dim < 0 || normalized_dim >= static_cast<int64_t>(input_shape.size())) {
            throw std::invalid_argument("Dimension out of range in prod_kernel");
        }
    }

    // Compute output shape
    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            if (dim == INT64_MIN) {
                launch_full_reduction_prod(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_prod(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (dim == INT64_MIN) {
                launch_full_reduction_prod(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_prod(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int32_t>();

            if (dim == INT64_MIN) {
                launch_full_reduction_prod(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_prod(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (dim == INT64_MIN) {
                launch_full_reduction_prod(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_prod(
                    input_data, output_data,
                    std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                    std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                    normalized_dim, stream
                );
            }
            break;
        }
        default:
            throw std::runtime_error("prod: unsupported dtype");
    }

    CUDA_PEEK_AND_THROW(stream, "prod_kernel");

    return output;
}

// ============================================================================
// Variance and Standard Deviation operations (Welford single-pass)
// ============================================================================

// Device helper: combine two Welford accumulators
template<typename T>
__device__ __forceinline__ void welford_combine(
    T& mean_a, T& m2_a, int64_t& count_a,
    T mean_b, T m2_b, int64_t count_b
) {
    if (count_b == 0) return;
    int64_t count_ab = count_a + count_b;
    T delta = mean_b - mean_a;
    T new_mean = mean_a + delta * static_cast<T>(count_b) / static_cast<T>(count_ab);
    m2_a += m2_b + delta * delta * static_cast<T>(count_a) * static_cast<T>(count_b) / static_cast<T>(count_ab);
    mean_a = new_mean;
    count_a = count_ab;
}

// Welford single-pass variance kernel — per-block (mean, M2, count) output
template<typename T>
__global__ void welford_variance_kernel(
    const T* input,
    T* block_means,
    T* block_m2s,
    int64_t* block_counts,
    int64_t n
) {
    __shared__ T s_mean[REDUCTION_BLOCK_SIZE];
    __shared__ T s_m2[REDUCTION_BLOCK_SIZE];
    __shared__ int64_t s_count[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Per-thread Welford accumulation
    T t_mean = 0;
    T t_m2 = 0;
    int64_t t_count = 0;

    for (int64_t i = idx; i < n; i += grid_size) {
        t_count++;
        T delta = input[i] - t_mean;
        t_mean += delta / static_cast<T>(t_count);
        T delta2 = input[i] - t_mean;
        t_m2 += delta * delta2;
    }

    s_mean[tid] = t_mean;
    s_m2[tid] = t_m2;
    s_count[tid] = t_count;
    __syncthreads();

    // Block-level tree reduction using Welford combine
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            welford_combine(s_mean[tid], s_m2[tid], s_count[tid],
                            s_mean[tid + stride], s_m2[tid + stride], s_count[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_means[blockIdx.x] = s_mean[0];
        block_m2s[blockIdx.x] = s_m2[0];
        block_counts[blockIdx.x] = s_count[0];
    }
}

// Final merge kernel: combines per-block Welford results into a single variance
// Uses parallel shared-memory tree reduction with 256 threads
template<typename T>
__global__ void welford_finalize_kernel(
    const T* block_means,
    const T* block_m2s,
    const int64_t* block_counts,
    T* output,
    int num_blocks,
    int64_t correction
) {
    // Shared memory for tree reduction: mean, m2, count per thread
    extern __shared__ char smem_raw[];
    T* s_mean = reinterpret_cast<T*>(smem_raw);
    T* s_m2 = s_mean + blockDim.x;
    int64_t* s_count = reinterpret_cast<int64_t*>(s_m2 + blockDim.x);

    int tid = threadIdx.x;

    // Each thread loads one or more block states via grid-stride
    T t_mean = T(0);
    T t_m2 = T(0);
    int64_t t_count = 0;

    if (tid < num_blocks) {
        t_mean = block_means[tid];
        t_m2 = block_m2s[tid];
        t_count = block_counts[tid];
    }
    // Grid-stride for blocks > blockDim.x
    for (int i = tid + blockDim.x; i < num_blocks; i += blockDim.x) {
        welford_combine(t_mean, t_m2, t_count,
                        block_means[i], block_m2s[i], block_counts[i]);
    }

    s_mean[tid] = t_mean;
    s_m2[tid] = t_m2;
    s_count[tid] = t_count;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (tid + stride) < num_blocks) {
            welford_combine(s_mean[tid], s_m2[tid], s_count[tid],
                            s_mean[tid + stride], s_m2[tid + stride], s_count[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        int64_t divisor = s_count[0] - correction;
        // Variance is undefined (NaN) when N <= correction — match CPU/PyTorch
        // instead of clamping the divisor to 1 and returning a finite m2.
        output[0] = (divisor > 0) ? (s_m2[0] / static_cast<T>(divisor))
                                  : static_cast<T>(NAN);
    }
}

// Helper function to compute variance — single-pass Welford, zero D2H transfers
template<typename T>
static void launch_variance_computation(const T* d_input, T* d_output, int64_t n, int64_t correction, cudaStream_t stream = nullptr) {
    if (n == 0) {
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, T(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (n == 1) {
        // var/std of a single element is 0 only when correction == 0; for
        // correction >= 1 (the default unbiased=true) it is undefined (NaN),
        // matching the CPU backend and PyTorch.
        T fill = (correction <= 0) ? T(0) : static_cast<T>(NAN);
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, fill);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    // Allocate per-block Welford accumulators
    backend::CachedMemoryGuard d_block_means_guard(num_blocks * sizeof(T));
    backend::CachedMemoryGuard d_block_m2s_guard(num_blocks * sizeof(T));
    backend::CachedMemoryGuard d_block_counts_guard(num_blocks * sizeof(int64_t));
    auto* d_block_means = static_cast<T*>(d_block_means_guard.get());
    auto* d_block_m2s = static_cast<T*>(d_block_m2s_guard.get());
    auto* d_block_counts = static_cast<int64_t*>(d_block_counts_guard.get());

    // Phase 1: Per-block Welford reduction
    welford_variance_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_block_means, d_block_m2s, d_block_counts, n);
    CUDA_CHECK(cudaGetLastError());

    // Phase 2: Merge block results — parallel tree reduction with 256 threads
    constexpr int FINALIZE_THREADS = 256;
    size_t welford_smem = FINALIZE_THREADS * (2 * sizeof(T) + sizeof(int64_t));
    welford_finalize_kernel<<<1, FINALIZE_THREADS, welford_smem, stream>>>(
        d_block_means, d_block_m2s, d_block_counts, d_output, num_blocks, correction);
    CUDA_CHECK(cudaGetLastError());
}

// Per-dimension variance reduction kernel (single-pass Welford)
template<typename T>
__global__ void var_along_dim_kernel(
    const T* input,
    T* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size,
    int64_t correction
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    // Compute multi-dimensional indices for output position
    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Welford single-pass algorithm along the reduction dimension
    using Acc = typename AccumType<T>::type;
    Acc mean = Acc(0);
    Acc m2 = Acc(0);
    int64_t count = 0;

    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        Acc val = Acc(input[in_idx]);
        count++;
        Acc delta = val - mean;
        mean = mean + delta / Acc(count);
        Acc delta2 = val - mean;
        m2 = m2 + delta * delta2;
    }

    // variance = m2 / (count - correction); NaN when count <= correction
    // (undefined), matching CPU/PyTorch rather than returning 0.
    int64_t denom = count - correction;
    output[out_idx] = denom > 0 ? T(m2 / Acc(denom)) : static_cast<T>(NAN);
    }
}

// Public API for variance
auto var_kernel(const Tensor& input_raw, int64_t dim, bool keepdim, int64_t correction, cudaStream_t stream) -> Tensor {
    // Ensure contiguous input — same root cause as the sum_kernel fix
    // (audit-2026-05-03 bug #2). The full-reduction path feeds input.numel() to
    // launch_variance_computation, which reads the buffer via a flat pointer, so
    // non-contiguous views would skip logical elements.
    auto input = input_raw.contiguous();
    auto [resolved_stream, stream_guard] = resolve_stream(stream, input);
    stream = resolved_stream;

    // Float16/BFloat16: widen to Float32 on device and compute. The result stays
    // Float32 to match the CPU var contract (output_dtype promoted). On-GPU cast.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        return var_kernel(input.to(DType::Float32), dim, keepdim, correction, stream);
    }

    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();

    // INT64_MIN is the sentinel for "reduce all dimensions". Negative dims
    // (incl. -1) refer to the last dim and must hit the per-dim path —
    // previously `-1` was misinterpreted as the all-dims sentinel, which
    // produced a scalar where the autograd backward expected a per-dim
    // result and broke `var(v, dim=-1)` gradcheck on CUDA.
    if (dim != INT64_MIN) {
        // Per-dimension variance reduction
        auto strides_span = input.strides();
        int64_t ndim = static_cast<int64_t>(input_shape.size());

        // Normalize dim
        int64_t actual_dim = dim;
        if (actual_dim < 0) actual_dim += ndim;
        if (actual_dim < 0 || actual_dim >= ndim) {
            throw std::runtime_error("var: dim out of range");
        }

        int64_t dim_size = input_shape[actual_dim];

        // Compute output shape
        std::vector<int64_t> output_shape;
        for (int64_t d = 0; d < ndim; d++) {
            if (d == actual_dim) {
                if (keepdim) output_shape.push_back(1);
            } else {
                output_shape.push_back(input_shape[d]);
            }
        }
        // Rank-collapse-to-scalar (reducing the only dim with keepdim=false)
        // must yield a scalar shape {} — matching var full-reduction and
        // sum/mean/compute_reduction_shape and the CPU convention — NOT {1}.
        // A scalar tensor still has numel 1 (empty-product), so output_size
        // below is correctly 1 for the single-output kernel launch.

        int64_t output_size = 1;
        for (auto s : output_shape) output_size *= s;

        Tensor output(output_shape, dtype, device);

        // Build DimMeta from input shape/strides
        std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
        std::vector<int64_t> strides_vec(strides_span.begin(), strides_span.end());
        DimMeta meta = make_dim_meta(shape_vec, strides_vec);

        constexpr int BLOCK = 256;
        int blocks = compute_grid_size(output_size, BLOCK);

        TENZOR_DISPATCH_FLOATING_TYPES(dtype, "var", [&]() {
            var_along_dim_kernel<scalar_t><<<blocks, BLOCK, 0, stream>>>(
                input.data<scalar_t>(), output.data<scalar_t>(), meta,
                ndim, actual_dim, output_size, dim_size, correction);
            CUDA_CHECK(cudaGetLastError());
        });

        CUDA_PEEK_AND_THROW(stream, "var_along_dim_kernel");

        return output;
    }

    // Full reduction — compute output shape
    std::vector<int64_t> output_shape;
    if (keepdim) {
        output_shape.resize(input_shape.size(), 1);
    } else {
        output_shape = {};  // Scalar
    }

    Tensor output(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();
            launch_variance_computation(input_data, output_data, input.numel(), correction, stream);
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();
            launch_variance_computation(input_data, output_data, input.numel(), correction, stream);
            break;
        }
        default:
            throw std::runtime_error("var: only Float32 and Float64 are supported");
    }

    CUDA_PEEK_AND_THROW(stream, "var_kernel");

    return output;
}

// Fused sum-reduce with sqrt finalization for L2 norm
// Reduces input[0..n-1] and writes sqrt(sum) to output[0]
template<typename T>
__global__ void sum_reduce_sqrt_kernel(const T* input, T* output, int64_t n) {
    using Acc = typename AccumType<T>::type;
    __shared__ Acc shared[REDUCTION_BLOCK_SIZE];
    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    Acc thread_sum = Acc(0);
    for (int64_t i = idx; i < n; i += grid_size) {
        thread_sum = thread_sum + Acc(input[i]);
    }
    shared[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) shared[tid] = shared[tid] + shared[tid + stride];
        __syncthreads();
    }
    __syncthreads();
    if (tid < WARP_SIZE) {
        Acc val = shared[tid];
        val = warp_reduce_sum(val);
        if (tid == 0) output[blockIdx.x] = T(sqrt(val));
    }
}

// Fused sum-reduce with pow finalization for Lp norm
// Reduces input[0..n-1] and writes pow(sum, exponent) to output[0]
template<typename T>
__global__ void sum_reduce_pow_kernel(const T* input, T* output, int64_t n, T exponent) {
    using Acc = typename AccumType<T>::type;
    __shared__ Acc shared[REDUCTION_BLOCK_SIZE];
    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    Acc thread_sum = Acc(0);
    for (int64_t i = idx; i < n; i += grid_size) {
        thread_sum = thread_sum + Acc(input[i]);
    }
    shared[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) shared[tid] = shared[tid] + shared[tid + stride];
        __syncthreads();
    }
    __syncthreads();
    if (tid < WARP_SIZE) {
        Acc val = shared[tid];
        val = warp_reduce_sum(val);
        if (tid == 0) output[blockIdx.x] = T(pow(val, Acc(exponent)));
    }
}

// Elementwise sqrt — used by std per-dim path (var_kernel + sqrt).
template<typename T>
__global__ void elementwise_sqrt_kernel(const T* input, T* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = T(sqrt(double(input[idx])));
}

// Public API for standard deviation
auto std_kernel(const Tensor& input_raw, int64_t dim, bool keepdim, int64_t correction, cudaStream_t stream) -> Tensor {
    // Ensure contiguous input — same root cause as the sum_kernel fix
    // (audit-2026-05-03 bug #2). The full-reduction path feeds input.numel() to
    // launch_variance_computation, which reads the buffer via a flat pointer, so
    // non-contiguous views would skip logical elements.
    auto input = input_raw.contiguous();
    auto [resolved_stream, stream_guard] = resolve_stream(stream, input);
    stream = resolved_stream;

    // Float16/BFloat16: widen to Float32 on device and compute. The result stays
    // Float32 to match the CPU std contract (output_dtype promoted). On-GPU cast.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        return std_kernel(input.to(DType::Float32), dim, keepdim, correction, stream);
    }

    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();

    // Per-dim reduction: reuse var_kernel (which supports it) and take sqrt.
    // CPU/ROCm/OneAPI/Vulkan all implement std this way; CUDA was missing the
    // path which stopped StdBackward from flowing gradients. Negative dims
    // (incl. -1) refer to the last dim and must hit the per-dim path.
    if (dim != INT64_MIN) {
        auto var_result = var_kernel(input, dim, keepdim, correction, stream);
        // sqrt element-wise into a new tensor
        auto out_shape_vec = std::vector<int64_t>(var_result.shape().begin(), var_result.shape().end());
        Tensor out(out_shape_vec, var_result.dtype(), var_result.device());
        int64_t n = var_result.numel();
        constexpr int BLOCK = 256;
        int blocks = (n + BLOCK - 1) / BLOCK;
        TENZOR_DISPATCH_FLOATING_TYPES(dtype, "std_sqrt", [&]() {
            auto* vd = var_result.data<scalar_t>();
            auto* od = out.data<scalar_t>();
            elementwise_sqrt_kernel<scalar_t><<<blocks, BLOCK, 0, stream>>>(vd, od, n);
            CUDA_CHECK(cudaGetLastError());
        });
        CUDA_PEEK_AND_THROW(stream, "std_kernel_perdim");
        return out;
    }

    // Compute output shape
    std::vector<int64_t> output_shape;
    if (keepdim) {
        output_shape.resize(input_shape.size(), 1);
    } else {
        output_shape = {};  // Scalar
    }

    // First compute variance, then sqrt in-place for std deviation
    Tensor var_result(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* var_data = var_result.data<float>();
            launch_variance_computation(input_data, var_data, input.numel(), correction, stream);

            // Compute sqrt of variance — fused into single-element reduce+sqrt
            Tensor output(output_shape, dtype, device);
            auto* output_data = output.data<float>();
            sum_reduce_sqrt_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(var_data, output_data, 1);
            CUDA_CHECK(cudaGetLastError());

            CUDA_PEEK_AND_THROW(stream, "std_kernel");

            return output;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* var_data = var_result.data<double>();
            launch_variance_computation(input_data, var_data, input.numel(), correction, stream);

            // Compute sqrt of variance — fused into single-element reduce+sqrt
            Tensor output(output_shape, dtype, device);
            auto* output_data = output.data<double>();
            sum_reduce_sqrt_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(var_data, output_data, 1);
            CUDA_CHECK(cudaGetLastError());

            CUDA_PEEK_AND_THROW(stream, "std_kernel");

            return output;
        }
        default:
            throw std::runtime_error("std: only Float32 and Float64 are supported");
    }
}

// L1 norm kernel - sum of absolute values
template<typename T>
__global__ void l1_norm_kernel(const T* input, T* output, int64_t n) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    T thread_sum = 0;
    for (int64_t i = idx; i < n; i += grid_size) {
        thread_sum += abs(input[i]);
    }

    shared[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        T val = shared[tid];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// L2 norm kernel - sqrt of sum of squares
template<typename T>
__global__ void l2_norm_squared_kernel(const T* input, T* output, int64_t n) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    T thread_sum = 0;
    for (int64_t i = idx; i < n; i += grid_size) {
        T val = input[i];
        thread_sum += val * val;
    }

    shared[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        T val = shared[tid];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// Lp norm kernel - (sum(|x|^p))^(1/p)
template<typename T>
__global__ void lp_norm_kernel(const T* input, T* output, int64_t n, float p) {
    __shared__ T shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    T thread_sum = 0;
    for (int64_t i = idx; i < n; i += grid_size) {
        thread_sum += pow(abs(input[i]), static_cast<T>(p));
    }

    shared[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= WARP_SIZE; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        T val = shared[tid];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// Per-dimension norm reduction kernel (follows sum_along_dim_kernel pattern)
template<typename T>
__global__ void norm_along_dim_kernel(
    const T* input,
    T* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size,
    float p
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    // Compute multi-dimensional indices for output position
    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;

    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Accumulate along the reduction dimension
    T acc = cuda_zero<T>();

    if (p == 1.0f) {
        // L1 norm: sum of |x|
        for (int64_t i = 0; i < dim_size; i++) {
            indices[dim] = i;
            int64_t in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * meta.strides[d];
            }
            T val = input[in_idx];
            acc = acc + (val < cuda_zero<T>() ? -val : val);
        }
        output[out_idx] = acc;
    } else if (p == 2.0f) {
        // L2 norm: sqrt(sum of x^2)
        for (int64_t i = 0; i < dim_size; i++) {
            indices[dim] = i;
            int64_t in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * meta.strides[d];
            }
            T val = input[in_idx];
            acc = acc + val * val;
        }
        output[out_idx] = sqrt(acc);
    } else {
        // General Lp norm: (sum of |x|^p)^(1/p)
        for (int64_t i = 0; i < dim_size; i++) {
            indices[dim] = i;
            int64_t in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * meta.strides[d];
            }
            T val = input[in_idx];
            acc = acc + pow(val < cuda_zero<T>() ? -val : val, T(p));
        }
        output[out_idx] = pow(acc, T(1.0f / p));
    }
    }
}

// Norm kernel implementation
auto norm_kernel(const Tensor& input_raw, float p, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    // Ensure contiguous input — same root cause as the sum_kernel fix
    // (audit-2026-05-03 bug #2). The full-reduction path feeds input.numel() to
    // l1_norm_kernel / lp_norm_kernel, which read the buffer via a flat pointer,
    // so non-contiguous views would skip logical elements.
    auto input = input_raw.contiguous();
    // Float16/BFloat16: upcast to Float32, compute norm, downcast result
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor f32_input = input.to(DType::Float32);
        Tensor f32_result = norm_kernel(f32_input, p, dim, keepdim, stream);
        return f32_result.to(orig);
    }

    auto [resolved_stream, stream_guard] = resolve_stream(stream, input);
    stream = resolved_stream;

    // INT64_MIN is the sentinel for "reduce all dimensions". Negative
    // dims (incl. -1) refer to the last dim and must hit the per-dim path.
    if (dim != INT64_MIN) {
        // Per-dimension norm reduction
        auto shape_span = input.shape();
        auto strides_span = input.strides();
        int64_t ndim = static_cast<int64_t>(shape_span.size());

        // Normalize dim
        int64_t actual_dim = dim;
        if (actual_dim < 0) actual_dim += ndim;
        if (actual_dim < 0 || actual_dim >= ndim) {
            throw std::runtime_error("norm: dim out of range");
        }

        int64_t dim_size = shape_span[actual_dim];

        // Compute output shape
        std::vector<int64_t> output_shape;
        for (int64_t d = 0; d < ndim; d++) {
            if (d == actual_dim) {
                if (keepdim) output_shape.push_back(1);
            } else {
                output_shape.push_back(shape_span[d]);
            }
        }
        // Rank-collapse-to-scalar (reducing the only dim, keepdim=false) yields
        // a scalar shape {} to match sum/mean/var and the CPU convention. A
        // scalar tensor has numel 1 (empty-product), so output_size is 1.

        int64_t output_size = 1;
        for (auto s : output_shape) output_size *= s;

        Tensor output(output_shape, input.dtype(), input.device());

        // Build DimMeta from input shape/strides
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        std::vector<int64_t> strides_vec(strides_span.begin(), strides_span.end());
        DimMeta meta = make_dim_meta(shape_vec, strides_vec);

        constexpr int BLOCK = 256;
        int blocks = compute_grid_size(output_size, BLOCK);

        TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "norm", [&]() {
            norm_along_dim_kernel<scalar_t><<<blocks, BLOCK, 0, stream>>>(
                input.data<scalar_t>(), output.data<scalar_t>(), meta,
                ndim, actual_dim, output_size, dim_size, p);
            CUDA_CHECK(cudaGetLastError());
        });

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error in norm_along_dim_kernel: ") + cudaGetErrorString(err));
        }

        return output;
    }

    auto shape = input.shape();
    auto dtype = input.dtype();
    auto device = input.device();

    std::vector<int64_t> output_shape;
    if (keepdim) {
        output_shape.resize(shape.size(), 1);
    } else {
        output_shape = {};  // Scalar, matching sum/mean/var full-reduction
    }

    Tensor output(output_shape, dtype, device);

    int64_t n = input.numel();
    if (n == 0) {
        throw std::runtime_error("norm: input tensor is empty");
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            if (p == 1.0f) {
                // L1 norm
                if (num_blocks == 1) {
                    l1_norm_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, output_data, n);
                    CUDA_CHECK(cudaGetLastError());
                } else {
                    backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(float));
                    auto* d_temp = static_cast<float*>(d_temp_guard.get());
                    l1_norm_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, num_blocks);
                    CUDA_CHECK(cudaGetLastError());
                }
            } else if (p == 2.0f) {
                // L2 norm — fused sqrt into final reduction
                if (num_blocks == 1) {
                    // Single block: reduce and sqrt in one kernel
                    backend::CachedMemoryGuard d_temp_guard(sizeof(float));
                    auto* d_temp = static_cast<float*>(d_temp_guard.get());
                    l2_norm_squared_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_sqrt_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, 1);
                    CUDA_CHECK(cudaGetLastError());
                } else {
                    backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(float));
                    auto* d_temp = static_cast<float*>(d_temp_guard.get());
                    l2_norm_squared_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_sqrt_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, num_blocks);
                    CUDA_CHECK(cudaGetLastError());
                }
            } else {
                // General Lp norm — fused pow into final reduction
                if (num_blocks == 1) {
                    backend::CachedMemoryGuard d_temp_guard(sizeof(float));
                    auto* d_temp = static_cast<float*>(d_temp_guard.get());
                    lp_norm_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n, p);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_pow_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, 1, 1.0f / p);
                    CUDA_CHECK(cudaGetLastError());
                } else {
                    backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(float));
                    auto* d_temp = static_cast<float*>(d_temp_guard.get());
                    lp_norm_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n, p);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_pow_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, num_blocks, 1.0f / p);
                    CUDA_CHECK(cudaGetLastError());
                }
            }

#ifndef NDEBUG
            cudaStreamSynchronize(stream);
#endif
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (p == 1.0f) {
                // L1 norm
                if (num_blocks == 1) {
                    l1_norm_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, output_data, n);
                    CUDA_CHECK(cudaGetLastError());
                } else {
                    backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(double));
                    auto* d_temp = static_cast<double*>(d_temp_guard.get());
                    l1_norm_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, num_blocks);
                    CUDA_CHECK(cudaGetLastError());
                }
            } else if (p == 2.0f) {
                // L2 norm — fused sqrt into final reduction
                if (num_blocks == 1) {
                    backend::CachedMemoryGuard d_temp_guard(sizeof(double));
                    auto* d_temp = static_cast<double*>(d_temp_guard.get());
                    l2_norm_squared_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_sqrt_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, 1);
                    CUDA_CHECK(cudaGetLastError());
                } else {
                    backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(double));
                    auto* d_temp = static_cast<double*>(d_temp_guard.get());
                    l2_norm_squared_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_sqrt_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, num_blocks);
                    CUDA_CHECK(cudaGetLastError());
                }
            } else {
                // General Lp norm — fused pow into final reduction
                if (num_blocks == 1) {
                    backend::CachedMemoryGuard d_temp_guard(sizeof(double));
                    auto* d_temp = static_cast<double*>(d_temp_guard.get());
                    lp_norm_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n, p);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_pow_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, 1, static_cast<double>(1.0 / p));
                    CUDA_CHECK(cudaGetLastError());
                } else {
                    backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(double));
                    auto* d_temp = static_cast<double*>(d_temp_guard.get());
                    lp_norm_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(input_data, d_temp, n, p);
                    CUDA_CHECK(cudaGetLastError());
                    sum_reduce_pow_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, output_data, num_blocks, static_cast<double>(1.0 / p));
                    CUDA_CHECK(cudaGetLastError());
                }
            }

#ifndef NDEBUG
            cudaStreamSynchronize(stream);
#endif
            break;
        }
        default:
            throw std::runtime_error("norm: only Float32 and Float64 are supported");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in norm_kernel: ") + cudaGetErrorString(err));
    }

    return output;
}


// ============================================================================
// Argsort kernel
// ============================================================================

template<typename T>
__global__ void argsort_kernel(const T* input, int64_t* output, int64_t n, bool descending) {
    // Bitonic sort for moderate-sized arrays on GPU
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize output indices
    if (tid < n) {
        output[tid] = tid;
    }
    __syncthreads();

    // Bitonic sort passes
    for (int64_t size = 2; size <= n; size *= 2) {
        for (int64_t stride = size / 2; stride > 0; stride /= 2) {
            if (tid < n) {
                int64_t partner = tid ^ stride;
                if (partner > tid && partner < n) {
                    bool swap;
                    T val_tid = input[output[tid]];
                    T val_partner = input[output[partner]];

                    // Determine sort direction for this sub-sequence
                    bool ascending_dir = ((tid & size) == 0);
                    if (descending) ascending_dir = !ascending_dir;

                    if (ascending_dir) {
                        swap = (val_tid > val_partner);
                    } else {
                        swap = (val_tid < val_partner);
                    }

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

// Index initialization kernel for argsort
__global__ void iota_kernel(int64_t* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = idx;
    }
}

// CUB-based argsort for larger arrays
template<typename T>
static void launch_argsort(const T* d_input, int64_t* d_output, int64_t n, bool descending, cudaStream_t stream = nullptr) {
    if (n == 0) return;

    if (n == 1) {
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, int64_t(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // NOTE: the single-block bitonic argsort_kernel only sorts correctly when n is
    // a power of two (the bitonic network is incomplete otherwise), so it is not
    // used here. CUB DeviceRadixSort below is correct for any n and is used for all
    // sizes (n==0 and n==1 are handled above).

    // Use CUB DeviceRadixSort with key-value pairs (correct for arbitrary n)
    backend::CachedMemoryGuard d_indices_in_guard(n * sizeof(int64_t));
    auto* d_indices_in = static_cast<int64_t*>(d_indices_in_guard.get());

    // Initialize indices 0..n-1
    int init_blocks = (n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
    iota_kernel<<<init_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_indices_in, n);
    CUDA_CHECK(cudaGetLastError());

    // Allocate output keys buffer
    backend::CachedMemoryGuard d_keys_out_guard(n * sizeof(T));
    auto* d_keys_out = static_cast<T*>(d_keys_out_guard.get());

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    if (descending) {
        cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
            d_input, d_keys_out, d_indices_in, d_output, n, 0, sizeof(T) * 8, stream);
        backend::CachedMemoryGuard d_temp_storage_guard(temp_storage_bytes);
        d_temp_storage = d_temp_storage_guard.get();
        cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
            d_input, d_keys_out, d_indices_in, d_output, n, 0, sizeof(T) * 8, stream);
    } else {
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_input, d_keys_out, d_indices_in, d_output, n, 0, sizeof(T) * 8, stream);
        backend::CachedMemoryGuard d_temp_storage_guard(temp_storage_bytes);
        d_temp_storage = d_temp_storage_guard.get();
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            d_input, d_keys_out, d_indices_in, d_output, n, 0, sizeof(T) * 8, stream);
    }
}


// Gather the dim slices of a (possibly strided-logical but contiguous-input)
// tensor into a segment-major buffer [num_slices * dim_size], where slice
// s = outer*inner_size + inner holds the dim_size elements
// input[outer*dim_size*inner_size + i*inner_size + inner], and seed per-slice
// local indices 0..dim_size-1. This lets cub::DeviceSegmentedRadixSort sort
// each slice independently and produce *local* (per-axis) indices, matching the
// CPU argsort_along_dim contract.
template<typename T>
__global__ void argsort_gather_kernel(const T* __restrict__ input,
                                      T* __restrict__ gathered,
                                      int64_t* __restrict__ local_idx,
                                      int64_t outer_size, int64_t dim_size,
                                      int64_t inner_size) {
    const int64_t total = outer_size * dim_size * inner_size;
    for (int64_t g = blockIdx.x * blockDim.x + threadIdx.x; g < total;
         g += blockDim.x * gridDim.x) {
        // g indexes the segment-major buffer: segment = g / dim_size, i = g % dim_size
        int64_t seg = g / dim_size;
        int64_t i = g % dim_size;
        int64_t outer = seg / inner_size;
        int64_t inner = seg % inner_size;
        int64_t in_off = outer * dim_size * inner_size + i * inner_size + inner;
        gathered[g] = input[in_off];
        local_idx[g] = i;
    }
}

// Scatter the per-segment sorted local indices back to the strided output
// positions (same layout as the input slices).
__global__ void argsort_scatter_kernel(const int64_t* __restrict__ sorted_local,
                                       int64_t* __restrict__ output,
                                       int64_t outer_size, int64_t dim_size,
                                       int64_t inner_size) {
    const int64_t total = outer_size * dim_size * inner_size;
    for (int64_t g = blockIdx.x * blockDim.x + threadIdx.x; g < total;
         g += blockDim.x * gridDim.x) {
        int64_t seg = g / dim_size;
        int64_t i = g % dim_size;
        int64_t outer = seg / inner_size;
        int64_t inner = seg % inner_size;
        int64_t out_off = outer * dim_size * inner_size + i * inner_size + inner;
        output[out_off] = sorted_local[g];
    }
}

__global__ void argsort_segment_offsets_kernel(int* __restrict__ offsets,
                                               int64_t num_segments,
                                               int64_t dim_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= num_segments) {
        offsets[idx] = static_cast<int>(idx * dim_size);
    }
}

// Per-dim argsort: sort each length-dim_size slice independently, emitting
// per-slice local indices in [0, dim_size).
template<typename T>
static void launch_argsort_along_dim(const T* d_input, int64_t* d_output,
                                     int64_t outer_size, int64_t dim_size,
                                     int64_t inner_size, bool descending,
                                     cudaStream_t stream) {
    const int64_t num_segments = outer_size * inner_size;
    const int64_t total = num_segments * dim_size;
    if (total == 0) return;
    // cub::DeviceSegmentedRadixSort takes `int num_items` / `int num_segments`
    // and int begin/end offsets (offsets[idx] = idx * dim_size). For total or
    // num_segments > INT_MAX those silently overflow/garble the segmented sort.
    // Reject up front with a clear message rather than producing wrong indices.
    if (total > static_cast<int64_t>(INT_MAX) ||
        num_segments > static_cast<int64_t>(INT_MAX)) {
        throw std::runtime_error(
            "argsort: per-dim segmented sort exceeds CUB's int32 element/segment "
            "limit (total elements or segment count > 2^31-1); reduce the sorted "
            "extent or batch the operation");
    }
    if (dim_size == 1) {
        // Each slice trivially sorts to local index 0.
        CUDA_CHECK(cudaMemsetAsync(d_output, 0, total * sizeof(int64_t), stream));
        return;
    }

    backend::CachedMemoryGuard gathered_guard(total * sizeof(T));
    auto* d_gathered = static_cast<T*>(gathered_guard.get());
    backend::CachedMemoryGuard local_idx_guard(total * sizeof(int64_t));
    auto* d_local_idx = static_cast<int64_t*>(local_idx_guard.get());
    backend::CachedMemoryGuard keys_out_guard(total * sizeof(T));
    auto* d_keys_out = static_cast<T*>(keys_out_guard.get());
    backend::CachedMemoryGuard sorted_local_guard(total * sizeof(int64_t));
    auto* d_sorted_local = static_cast<int64_t*>(sorted_local_guard.get());
    backend::CachedMemoryGuard offsets_guard((num_segments + 1) * sizeof(int));
    auto* d_offsets = static_cast<int*>(offsets_guard.get());

    int gather_blocks = compute_grid_size(total, REDUCTION_BLOCK_SIZE);
    argsort_gather_kernel<T><<<gather_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_gathered, d_local_idx, outer_size, dim_size, inner_size);
    CUDA_CHECK(cudaGetLastError());

    int off_blocks = compute_grid_size(num_segments + 1, REDUCTION_BLOCK_SIZE);
    argsort_segment_offsets_kernel<<<off_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_offsets, num_segments, dim_size);
    CUDA_CHECK(cudaGetLastError());

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    if (descending) {
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
            d_temp, temp_bytes, d_gathered, d_keys_out, d_local_idx, d_sorted_local,
            static_cast<int>(total), static_cast<int>(num_segments),
            d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream);
        backend::CachedMemoryGuard temp_guard(temp_bytes);
        d_temp = temp_guard.get();
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
            d_temp, temp_bytes, d_gathered, d_keys_out, d_local_idx, d_sorted_local,
            static_cast<int>(total), static_cast<int>(num_segments),
            d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream);
    } else {
        cub::DeviceSegmentedRadixSort::SortPairs(
            d_temp, temp_bytes, d_gathered, d_keys_out, d_local_idx, d_sorted_local,
            static_cast<int>(total), static_cast<int>(num_segments),
            d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream);
        backend::CachedMemoryGuard temp_guard(temp_bytes);
        d_temp = temp_guard.get();
        cub::DeviceSegmentedRadixSort::SortPairs(
            d_temp, temp_bytes, d_gathered, d_keys_out, d_local_idx, d_sorted_local,
            static_cast<int>(total), static_cast<int>(num_segments),
            d_offsets, d_offsets + 1, 0, sizeof(T) * 8, stream);
    }

    argsort_scatter_kernel<<<gather_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_sorted_local, d_output, outer_size, dim_size, inner_size);
    CUDA_CHECK(cudaGetLastError());
}

auto argsort_kernel(const Tensor& input, int64_t dim, bool descending, cudaStream_t stream) -> Tensor {
    auto [resolved_stream, stream_guard] = resolve_stream(stream, input);
    stream = resolved_stream;

    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());
    const int64_t n = input.numel();

    // Normalize and validate dim, mirroring the CPU argsort contract.
    if (dim < 0) dim += ndim;
    if (ndim == 0 || dim < 0 || dim >= ndim) {
        throw std::out_of_range("argsort: dimension out of range");
    }

    // CUB radix sort consumes a contiguous buffer; the gather/scatter offsets
    // below assume packed input strides.
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();

    Tensor output(std::vector<int64_t>(input_shape.begin(), input_shape.end()), DType::Int64, device);

    // Decompose into outer/dim/inner. For the common case where the sort axis is
    // the only non-trivial dimension (1-D, or all other dims size 1), the
    // segment-major gather is the identity and flat indices already equal the
    // per-axis indices, so we keep the simpler whole-buffer radix sort path.
    const int64_t dim_size = input_shape[dim];
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) inner_size *= input_shape[d];
    const int64_t outer_size = (dim_size == 0) ? 0 : (n / (dim_size * (inner_size == 0 ? 1 : inner_size)));
    const bool single_slice = (n == dim_size);  // exactly one slice spans the whole tensor

    switch (dtype) {
        case DType::Float32:
            if (single_slice) {
                launch_argsort(input_cont.data<float>(), output.data<int64_t>(), n, descending, stream);
            } else {
                launch_argsort_along_dim(input_cont.data<float>(), output.data<int64_t>(),
                                         outer_size, dim_size, inner_size, descending, stream);
            }
            break;
        case DType::Float64:
            if (single_slice) {
                launch_argsort(input_cont.data<double>(), output.data<int64_t>(), n, descending, stream);
            } else {
                launch_argsort_along_dim(input_cont.data<double>(), output.data<int64_t>(),
                                         outer_size, dim_size, inner_size, descending, stream);
            }
            break;
        case DType::Float16: {
            // CUB RadixSort doesn't support __half, so convert to float32 and sort that
            backend::CachedMemoryGuard d_float_buf_guard(n * sizeof(float));
            auto* d_float_buf = static_cast<float*>(d_float_buf_guard.get());
            const __half* d_half = reinterpret_cast<const __half*>(input_cont.data_ptr());
            int blocks = (n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;
            half_to_float_kernel<<<blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_half, d_float_buf, n);
            CUDA_CHECK(cudaGetLastError());
            if (single_slice) {
                launch_argsort(d_float_buf, output.data<int64_t>(), n, descending, stream);
            } else {
                launch_argsort_along_dim(d_float_buf, output.data<int64_t>(),
                                         outer_size, dim_size, inner_size, descending, stream);
            }
            break;
        }
        default:
            throw std::runtime_error("argsort_kernel: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Any/All reductions
// ============================================================================

// Per-dimension any reduction kernel: output[out_idx] = OR of (input != 0) along dim
template<typename T>
__global__ void any_along_dim_kernel(
    const T* input,
    uint8_t* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    // Compute multi-dimensional indices for output position
    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // OR reduction along dim
    uint8_t result = 0;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        if (input[in_idx] != T(0)) {
            result = 1;
            break;  // Short-circuit: found a non-zero element
        }
    }
    output[out_idx] = result;
    }
}

// Specialization for __half (no != operator with T(0))
template<>
__global__ void any_along_dim_kernel<__half>(
    const __half* input,
    uint8_t* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) { indices[d] = 0; continue; }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    uint8_t result = 0;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) in_idx += indices[d] * meta.strides[d];
        if (__half2float(input[in_idx]) != 0.0f) { result = 1; break; }
    }
    output[out_idx] = result;
    }
}

// Specialization for __nv_bfloat16
template<>
__global__ void any_along_dim_kernel<__nv_bfloat16>(
    const __nv_bfloat16* input,
    uint8_t* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) { indices[d] = 0; continue; }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    uint8_t result = 0;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) in_idx += indices[d] * meta.strides[d];
        if (__bfloat162float(input[in_idx]) != 0.0f) { result = 1; break; }
    }
    output[out_idx] = result;
    }
}

// Per-dimension all reduction kernel: output[out_idx] = AND of (input != 0) along dim
template<typename T>
__global__ void all_along_dim_kernel(
    const T* input,
    uint8_t* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) { indices[d] = 0; continue; }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // AND reduction along dim
    uint8_t result = 1;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        if (input[in_idx] == T(0)) {
            result = 0;
            break;  // Short-circuit: found a zero element
        }
    }
    output[out_idx] = result;
    }
}

// Specialization for __half
template<>
__global__ void all_along_dim_kernel<__half>(
    const __half* input,
    uint8_t* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) { indices[d] = 0; continue; }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    uint8_t result = 1;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) in_idx += indices[d] * meta.strides[d];
        if (__half2float(input[in_idx]) == 0.0f) { result = 0; break; }
    }
    output[out_idx] = result;
    }
}

// Specialization for __nv_bfloat16
template<>
__global__ void all_along_dim_kernel<__nv_bfloat16>(
    const __nv_bfloat16* input,
    uint8_t* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) { indices[d] = 0; continue; }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    uint8_t result = 1;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) in_idx += indices[d] * meta.strides[d];
        if (__bfloat162float(input[in_idx]) == 0.0f) { result = 0; break; }
    }
    output[out_idx] = result;
    }
}

// Full reduction any kernel: check if any element in entire tensor is non-zero
template<typename T>
__global__ void any_full_reduce_kernel(const T* input, uint8_t* output, int64_t n) {
    __shared__ uint8_t shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    uint8_t thread_any = 0;
    for (int64_t i = idx; i < n; i += grid_size) {
        if (input[i] != T(0)) { thread_any = 1; break; }
    }

    shared[tid] = thread_any;
    __syncthreads();

    // OR reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = shared[tid] | shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

// Specialization for __half
template<>
__global__ void any_full_reduce_kernel<__half>(const __half* input, uint8_t* output, int64_t n) {
    __shared__ uint8_t shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    uint8_t thread_any = 0;
    for (int64_t i = idx; i < n; i += grid_size) {
        if (__half2float(input[i]) != 0.0f) { thread_any = 1; break; }
    }

    shared[tid] = thread_any;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = shared[tid] | shared[tid + stride];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = shared[0];
}

// Specialization for __nv_bfloat16
template<>
__global__ void any_full_reduce_kernel<__nv_bfloat16>(const __nv_bfloat16* input, uint8_t* output, int64_t n) {
    __shared__ uint8_t shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    uint8_t thread_any = 0;
    for (int64_t i = idx; i < n; i += grid_size) {
        if (__bfloat162float(input[i]) != 0.0f) { thread_any = 1; break; }
    }

    shared[tid] = thread_any;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = shared[tid] | shared[tid + stride];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = shared[0];
}

// Full reduction all kernel: check if all elements in entire tensor are non-zero
template<typename T>
__global__ void all_full_reduce_kernel(const T* input, uint8_t* output, int64_t n) {
    __shared__ uint8_t shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    uint8_t thread_all = 1;
    for (int64_t i = idx; i < n; i += grid_size) {
        if (input[i] == T(0)) { thread_all = 0; break; }
    }

    shared[tid] = thread_all;
    __syncthreads();

    // AND reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = shared[tid] & shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

// Specialization for __half
template<>
__global__ void all_full_reduce_kernel<__half>(const __half* input, uint8_t* output, int64_t n) {
    __shared__ uint8_t shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    uint8_t thread_all = 1;
    for (int64_t i = idx; i < n; i += grid_size) {
        if (__half2float(input[i]) == 0.0f) { thread_all = 0; break; }
    }

    shared[tid] = thread_all;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = shared[tid] & shared[tid + stride];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = shared[0];
}

// Specialization for __nv_bfloat16
template<>
__global__ void all_full_reduce_kernel<__nv_bfloat16>(const __nv_bfloat16* input, uint8_t* output, int64_t n) {
    __shared__ uint8_t shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    uint8_t thread_all = 1;
    for (int64_t i = idx; i < n; i += grid_size) {
        if (__bfloat162float(input[i]) == 0.0f) { thread_all = 0; break; }
    }

    shared[tid] = thread_all;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = shared[tid] & shared[tid + stride];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = shared[0];
}

// Final reduction for any: OR across block results
__global__ void any_final_reduce_kernel(const uint8_t* block_results, uint8_t* output, int num_blocks) {
    __shared__ uint8_t shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    uint8_t val = 0;
    if (tid < num_blocks) val = block_results[tid];
    for (int i = tid + blockDim.x; i < num_blocks; i += blockDim.x) {
        val = val | block_results[i];
    }

    shared[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = shared[tid] | shared[tid + stride];
        __syncthreads();
    }

    if (tid == 0) output[0] = shared[0];
}

// Final reduction for all: AND across block results
__global__ void all_final_reduce_kernel(const uint8_t* block_results, uint8_t* output, int num_blocks) {
    __shared__ uint8_t shared[REDUCTION_BLOCK_SIZE];

    int tid = threadIdx.x;
    uint8_t val = 1;
    if (tid < num_blocks) val = block_results[tid];
    // Threads beyond num_blocks must start with identity element for AND
    else val = 1;
    for (int i = tid + blockDim.x; i < num_blocks; i += blockDim.x) {
        val = val & block_results[i];
    }

    shared[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = shared[tid] & shared[tid + stride];
        __syncthreads();
    }

    if (tid == 0) output[0] = shared[0];
}

// Helper: launch full any reduction
template<typename T>
static void launch_full_reduction_any(const T* d_input, uint8_t* d_output, int64_t n, cudaStream_t stream) {
    if (n == 0) {
        // any() on empty tensor is false
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, uint8_t(0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        any_full_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(uint8_t));
        auto* d_temp = static_cast<uint8_t*>(d_temp_guard.get());
        any_full_reduce_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());
        any_final_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Helper: launch full all reduction
template<typename T>
static void launch_full_reduction_all(const T* d_input, uint8_t* d_output, int64_t n, cudaStream_t stream) {
    if (n == 0) {
        // all() on empty tensor is true (vacuous truth)
        fill_scalar_kernel<<<1, 1, 0, stream>>>(d_output, uint8_t(1));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int num_blocks = std::min<int>((n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE, 1024);

    if (num_blocks == 1) {
        all_full_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(uint8_t));
        auto* d_temp = static_cast<uint8_t*>(d_temp_guard.get());
        all_full_reduce_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_input, d_temp, n);
        CUDA_CHECK(cudaGetLastError());
        all_final_reduce_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(d_temp, d_output, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Helper: launch dim any reduction
template<typename T>
static void launch_dim_reduction_any(
    const T* d_input,
    uint8_t* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    cudaStream_t stream = nullptr
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) output_size *= input_shape[i];
    }

    if (output_size == 0 || dim_size == 0) return;

    DimMeta meta = make_dim_meta(input_shape, input_strides);
    int num_blocks = compute_grid_size(output_size, REDUCTION_BLOCK_SIZE);
    any_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

// Helper: launch dim all reduction
template<typename T>
static void launch_dim_reduction_all(
    const T* d_input,
    uint8_t* d_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& input_strides,
    int64_t dim,
    cudaStream_t stream = nullptr
) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) output_size *= input_shape[i];
    }

    if (output_size == 0 || dim_size == 0) return;

    DimMeta meta = make_dim_meta(input_shape, input_strides);
    int num_blocks = compute_grid_size(output_size, REDUCTION_BLOCK_SIZE);
    all_along_dim_kernel<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
        d_input, d_output, meta, ndim, dim, output_size, dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

auto any_kernel(const Tensor& input_raw, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    // Ensure contiguous input — same root cause as the sum_kernel fix
    // (audit-2026-05-03 bug #2). The full-reduction path feeds input.numel() to
    // any_full_reduce_kernel, which reads the buffer via a flat pointer, so
    // non-contiguous views would skip logical elements.
    auto input = input_raw.contiguous();
    auto [resolved_stream, stream_guard] = resolve_stream(stream, input);
    stream = resolved_stream;

    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize dimension
    int64_t normalized_dim = dim;
    if (dim != INT64_MIN) {
        if (dim < 0) normalized_dim = ndim + dim;
        if (normalized_dim < 0 || normalized_dim >= ndim) {
            throw std::runtime_error("Dimension " + std::to_string(dim) +
                " out of range for tensor with " + std::to_string(ndim) + " dimensions");
        }
    }

    // Output shape: same as input but with reduction dim removed/kept
    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        normalized_dim, keepdim
    );

    // Output is always Bool (uint8_t)
    Tensor output(output_shape, DType::Bool, device);

    auto* output_data = output.data<uint8_t>();
    auto shape_vec = std::vector<int64_t>(input_shape.begin(), input_shape.end());
    auto strides_vec = std::vector<int64_t>(input_strides.begin(), input_strides.end());

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            if (dim == INT64_MIN) {
                launch_full_reduction_any(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_any(input_data, output_data, shape_vec, strides_vec, normalized_dim, stream);
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            if (dim == INT64_MIN) {
                launch_full_reduction_any(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_any(input_data, output_data, shape_vec, strides_vec, normalized_dim, stream);
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            if (dim == INT64_MIN) {
                launch_full_reduction_any(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_any(input_data, output_data, shape_vec, strides_vec, normalized_dim, stream);
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            if (dim == INT64_MIN) {
                launch_full_reduction_any(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_any(input_data, output_data, shape_vec, strides_vec, normalized_dim, stream);
            }
            break;
        }
        case DType::Float16: {
            auto* input_data = reinterpret_cast<const __half*>(input.data_ptr());
            if (dim == INT64_MIN) {
                launch_full_reduction_any(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_any(input_data, output_data, shape_vec, strides_vec, normalized_dim, stream);
            }
            break;
        }
        case DType::BFloat16: {
            auto* input_data = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
            if (dim == INT64_MIN) {
                launch_full_reduction_any(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_any(input_data, output_data, shape_vec, strides_vec, normalized_dim, stream);
            }
            break;
        }
        case DType::Bool: {
            auto* input_data = input.data<uint8_t>();
            if (dim == INT64_MIN) {
                launch_full_reduction_any(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_any(input_data, output_data, shape_vec, strides_vec, normalized_dim, stream);
            }
            break;
        }
        default:
            throw std::runtime_error("any: unsupported dtype");
    }

    CUDA_PEEK_AND_THROW(stream, "any_kernel");
    return output;
}

auto all_kernel(const Tensor& input_raw, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    // Ensure contiguous input — same root cause as the sum_kernel fix
    // (audit-2026-05-03 bug #2). The full-reduction path feeds input.numel() to
    // all_full_reduce_kernel, which reads the buffer via a flat pointer, so
    // non-contiguous views would skip logical elements.
    auto input = input_raw.contiguous();
    auto [resolved_stream, stream_guard] = resolve_stream(stream, input);
    stream = resolved_stream;

    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Normalize dimension
    int64_t normalized_dim = dim;
    if (dim != INT64_MIN) {
        if (dim < 0) normalized_dim = ndim + dim;
        if (normalized_dim < 0 || normalized_dim >= ndim) {
            throw std::runtime_error("Dimension " + std::to_string(dim) +
                " out of range for tensor with " + std::to_string(ndim) + " dimensions");
        }
    }

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        normalized_dim, keepdim
    );

    // Output is always Bool (uint8_t)
    Tensor output(output_shape, DType::Bool, device);

    auto* output_data = output.data<uint8_t>();
    auto shape_vec = std::vector<int64_t>(input_shape.begin(), input_shape.end());
    auto strides_vec = std::vector<int64_t>(input_strides.begin(), input_strides.end());

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            if (dim == INT64_MIN) {
                launch_full_reduction_all(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_all(input_data, output_data, shape_vec, strides_vec, normalized_dim, stream);
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            if (dim == INT64_MIN) {
                launch_full_reduction_all(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_all(input_data, output_data, shape_vec, strides_vec, normalized_dim, stream);
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            if (dim == INT64_MIN) {
                launch_full_reduction_all(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_all(input_data, output_data, shape_vec, strides_vec, normalized_dim, stream);
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            if (dim == INT64_MIN) {
                launch_full_reduction_all(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_all(input_data, output_data, shape_vec, strides_vec, normalized_dim, stream);
            }
            break;
        }
        case DType::Float16: {
            auto* input_data = reinterpret_cast<const __half*>(input.data_ptr());
            if (dim == INT64_MIN) {
                launch_full_reduction_all(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_all(input_data, output_data, shape_vec, strides_vec, normalized_dim, stream);
            }
            break;
        }
        case DType::BFloat16: {
            auto* input_data = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
            if (dim == INT64_MIN) {
                launch_full_reduction_all(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_all(input_data, output_data, shape_vec, strides_vec, normalized_dim, stream);
            }
            break;
        }
        case DType::Bool: {
            auto* input_data = input.data<uint8_t>();
            if (dim == INT64_MIN) {
                launch_full_reduction_all(input_data, output_data, input.numel(), stream);
            } else {
                launch_dim_reduction_all(input_data, output_data, shape_vec, strides_vec, normalized_dim, stream);
            }
            break;
        }
        default:
            throw std::runtime_error("all: unsupported dtype");
    }

    CUDA_PEEK_AND_THROW(stream, "all_kernel");
    return output;
}

// ============================================================================
// LogSumExp - Numerically stable log(sum(exp(x)))
// ============================================================================

// Step 1: Find max along dim
template<typename T>
__global__ void logsumexp_max_along_dim_kernel(
    const T* input,
    typename AccumType<T>::type* max_out,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    using Acc = typename AccumType<T>::type;
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) { indices[d] = 0; continue; }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    Acc m = Acc(-1e38);
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        Acc val = Acc(input[in_idx]);
        if (val > m) m = val;
    }
    max_out[out_idx] = m;
    }
}

// Specialization for float: use -FLT_MAX
template<>
__global__ void logsumexp_max_along_dim_kernel<float>(
    const float* input,
    float* max_out,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) { indices[d] = 0; continue; }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    float m = -FLT_MAX;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        float val = input[in_idx];
        if (val > m) m = val;
    }
    max_out[out_idx] = m;
    }
}

// Specialization for double: use -DBL_MAX
template<>
__global__ void logsumexp_max_along_dim_kernel<double>(
    const double* input,
    double* max_out,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) { indices[d] = 0; continue; }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    double m = -DBL_MAX;
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        double val = input[in_idx];
        if (val > m) m = val;
    }
    max_out[out_idx] = m;
    }
}

// Step 2: Compute sum(exp(x - max)) and result = max + log(sum)
template<typename T>
__global__ void logsumexp_sum_exp_kernel(
    const T* input,
    const typename AccumType<T>::type* max_vals,
    T* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size
) {
    using Acc = typename AccumType<T>::type;
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == dim) { indices[d] = 0; continue; }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    Acc m = max_vals[out_idx];
    // Handle inf: if max is +/-inf, result should be the same inf.
    // `continue` (not `return`) so the grid-stride loop still processes this
    // thread's remaining output slices.
    if (isinf(float(m))) {
        output[out_idx] = T(m);
        continue;
    }

    Acc sum = Acc(0);
    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        sum += exp(Acc(input[in_idx]) - m);
    }
    output[out_idx] = T(log(sum) + m);
    }
}

// Full reduction version
template<typename T>
__global__ void logsumexp_full_kernel(
    const T* input,
    T* output,
    int64_t n
) {
    using Acc = typename AccumType<T>::type;

    // Step 1: Find max using shared memory reduction
    __shared__ Acc smax[REDUCTION_BLOCK_SIZE];
    int tid = threadIdx.x;
    Acc local_max = Acc(-1e38);

    for (int64_t i = tid; i < n; i += blockDim.x) {
        Acc val = Acc(input[i]);
        if (val > local_max) local_max = val;
    }
    smax[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (smax[tid + stride] > smax[tid])
                smax[tid] = smax[tid + stride];
        }
        __syncthreads();
    }
    Acc m = smax[0];
    __syncthreads();

    // Handle inf
    if (isinf(float(m))) {
        if (tid == 0) output[0] = T(m);
        return;
    }

    // Step 2: Sum exp(x - max) using shared memory reduction
    __shared__ Acc ssum[REDUCTION_BLOCK_SIZE];
    Acc local_sum = Acc(0);
    for (int64_t i = tid; i < n; i += blockDim.x) {
        local_sum += exp(Acc(input[i]) - m);
    }
    ssum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = T(log(ssum[0]) + m);
    }
}

auto logsumexp_kernel(const Tensor& input_raw, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor {
    // Ensure contiguous input — same root cause as the sum_kernel fix
    // (audit-2026-05-03 bug #2). The full-reduction path feeds input.numel() to
    // logsumexp_full_kernel, which reads the buffer via a flat pointer, so
    // non-contiguous views would skip logical elements.
    auto input = input_raw.contiguous();
    auto [resolved_stream, stream_guard] = resolve_stream(stream, input);
    stream = resolved_stream;

    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    if (dtype != DType::Float32 && dtype != DType::Float64 &&
        dtype != DType::Float16 && dtype != DType::BFloat16) {
        throw std::runtime_error("logsumexp: only Float32, Float64, Float16, and BFloat16 are supported");
    }

    // Normalize negative dimension
    int64_t normalized_dim = dim;
    if (dim != INT64_MIN) {
        if (dim < 0) normalized_dim = ndim + dim;
        if (normalized_dim < 0 || normalized_dim >= ndim) {
            throw std::runtime_error("Dimension " + std::to_string(dim) +
                " out of range for tensor with " + std::to_string(ndim) + " dimensions");
        }
    }

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        normalized_dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    auto shape_vec = std::vector<int64_t>(input_shape.begin(), input_shape.end());
    auto strides_vec = std::vector<int64_t>(input_strides.begin(), input_strides.end());

    if (dim == INT64_MIN) {
        // Full reduction
        int64_t n = input.numel();
        if (n == 0) {
            // logsumexp of empty set is -inf
            switch (dtype) {
                case DType::Float32:
                    fill_scalar_kernel<<<1, 1, 0, stream>>>(output.data<float>(), -FLT_MAX);
                    break;
                case DType::Float64:
                    fill_scalar_kernel<<<1, 1, 0, stream>>>(output.data<double>(), -DBL_MAX);
                    break;
                default: break;
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_PEEK_AND_THROW(stream, "logsumexp_kernel");
            return output;
        }

        switch (dtype) {
            case DType::Float32:
                logsumexp_full_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(
                    input.data<float>(), output.data<float>(), n);
                break;
            case DType::Float64:
                logsumexp_full_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(
                    input.data<double>(), output.data<double>(), n);
                break;
            case DType::Float16:
                logsumexp_full_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __half*>(input.data_ptr()),
                    reinterpret_cast<__half*>(output.data_ptr()), n);
                break;
            case DType::BFloat16:
                logsumexp_full_kernel<<<1, REDUCTION_BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), n);
                break;
            default: break;
        }
    } else {
        // Dim reduction
        int64_t dim_size = input_shape[normalized_dim];
        int64_t output_size = 1;
        for (int64_t i = 0; i < ndim; i++) {
            if (i != normalized_dim) output_size *= input_shape[i];
        }

        if (output_size == 0 || dim_size == 0) {
            CUDA_PEEK_AND_THROW(stream, "logsumexp_kernel");
            return output;
        }

        DimMeta meta = make_dim_meta(shape_vec, strides_vec);
        int num_blocks = compute_grid_size(output_size, REDUCTION_BLOCK_SIZE);

        switch (dtype) {
            case DType::Float32: {
                // Allocate temp buffer for max values
                Tensor max_buf({output_size}, DType::Float32, device);
                logsumexp_max_along_dim_kernel<float><<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
                    input.data<float>(), max_buf.data<float>(),
                    meta, ndim, normalized_dim, output_size, dim_size);
                CUDA_CHECK(cudaGetLastError());

                logsumexp_sum_exp_kernel<float><<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
                    input.data<float>(), max_buf.data<float>(), output.data<float>(),
                    meta, ndim, normalized_dim, output_size, dim_size);
                CUDA_CHECK(cudaGetLastError());
                break;
            }
            case DType::Float64: {
                Tensor max_buf({output_size}, DType::Float64, device);
                logsumexp_max_along_dim_kernel<double><<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
                    input.data<double>(), max_buf.data<double>(),
                    meta, ndim, normalized_dim, output_size, dim_size);
                CUDA_CHECK(cudaGetLastError());

                logsumexp_sum_exp_kernel<double><<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
                    input.data<double>(), max_buf.data<double>(), output.data<double>(),
                    meta, ndim, normalized_dim, output_size, dim_size);
                CUDA_CHECK(cudaGetLastError());
                break;
            }
            case DType::Float16: {
                // Use float accumulator for max
                Tensor max_buf({output_size}, DType::Float32, device);
                logsumexp_max_along_dim_kernel<__half><<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __half*>(input.data_ptr()),
                    max_buf.data<float>(),
                    meta, ndim, normalized_dim, output_size, dim_size);
                CUDA_CHECK(cudaGetLastError());

                logsumexp_sum_exp_kernel<__half><<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __half*>(input.data_ptr()),
                    max_buf.data<float>(),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    meta, ndim, normalized_dim, output_size, dim_size);
                CUDA_CHECK(cudaGetLastError());
                break;
            }
            case DType::BFloat16: {
                Tensor max_buf({output_size}, DType::Float32, device);
                logsumexp_max_along_dim_kernel<__nv_bfloat16><<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                    max_buf.data<float>(),
                    meta, ndim, normalized_dim, output_size, dim_size);
                CUDA_CHECK(cudaGetLastError());

                logsumexp_sum_exp_kernel<__nv_bfloat16><<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                    max_buf.data<float>(),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    meta, ndim, normalized_dim, output_size, dim_size);
                CUDA_CHECK(cudaGetLastError());
                break;
            }
            default: break;
        }
    }

    CUDA_PEEK_AND_THROW(stream, "logsumexp_kernel");
    return output;
}

// ============================================================================
// CosineSimilarity: sum(a*b, dim) / (norm(a, dim) * norm(b, dim) + eps)
// ============================================================================

template<typename T>
__global__ void cosine_similarity_along_dim_kernel(
    const T* a,
    const T* b,
    T* output,
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t output_size,
    int64_t dim_size,
    float eps
) {
    const int64_t stride_out = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < output_size; out_idx += stride_out) {

    // Compute multi-dimensional indices for the row-major output position.
    // The earlier loop peeled dims left-to-right which corresponds to
    // column-major storage; output is row-major so the lowest digit is the
    // LAST dim. Walking d from ndim-1 down to 0 with `tmp /= shape[d]`
    // gives the right (i_0, i_1, ...) tuple. Caught by
    // tests/backend_parity/test_stable_math_parity (CosineSimilarity_LastDim).
    int64_t indices[DIM_META_MAX_RANK];
    int64_t tmp = out_idx;
    for (int64_t d = ndim - 1; d >= 0; d--) {
        if (d == dim) {
            indices[d] = 0;
            continue;
        }
        indices[d] = tmp % meta.shape[d];
        tmp /= meta.shape[d];
    }

    // Accumulate dot product and norms along the reduction dimension
    using AccT = typename AccumType<T>::type;
    AccT dot_acc = AccT(0);
    AccT norm_a_acc = AccT(0);
    AccT norm_b_acc = AccT(0);

    for (int64_t i = 0; i < dim_size; i++) {
        indices[dim] = i;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * meta.strides[d];
        }
        AccT av = static_cast<AccT>(a[in_idx]);
        AccT bv = static_cast<AccT>(b[in_idx]);
        dot_acc += av * bv;
        norm_a_acc += av * av;
        norm_b_acc += bv * bv;
    }

    AccT denom = sqrt(norm_a_acc) * sqrt(norm_b_acc) + static_cast<AccT>(eps);
    output[out_idx] = static_cast<T>(dot_acc / denom);
    }
}

auto cosine_similarity_kernel(const Tensor& a, const Tensor& b, int64_t dim, double eps, cudaStream_t stream) -> Tensor {
    // Upcast FP16/BF16 to Float32
    if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
        DType orig = a.dtype();
        Tensor result = cosine_similarity_kernel(a.to(DType::Float32), b.to(DType::Float32), dim, eps, stream);
        return result.to(orig);
    }

    if (a.dtype() != b.dtype()) throw std::runtime_error("cosine_similarity: tensors must have the same dtype");

    auto [resolved_stream, stream_guard] = resolve_stream(stream, a);
    stream = resolved_stream;

    auto shape_span = a.shape();
    int64_t ndim = static_cast<int64_t>(shape_span.size());

    // Normalize dim
    int64_t actual_dim = dim;
    if (actual_dim < 0) actual_dim += ndim;
    if (actual_dim < 0 || actual_dim >= ndim) {
        throw std::runtime_error("cosine_similarity: dim out of range");
    }

    int64_t dim_size = shape_span[actual_dim];

    // Compute output shape (reduction dimension removed)
    std::vector<int64_t> output_shape;
    for (int64_t d = 0; d < ndim; d++) {
        if (d != actual_dim) {
            output_shape.push_back(shape_span[d]);
        }
    }
    // Fully-reduced result collapses to a scalar shape {} to match the CPU
    // convention (compute_reduction_shape and the var/norm paths do the same);
    // do NOT force {1}. output_size below is 1 for an empty shape (empty
    // product), so the single-element kernel launch is unaffected.

    int64_t output_size = 1;
    for (auto s : output_shape) output_size *= s;

    Tensor output(output_shape, a.dtype(), a.device());

    std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
    auto strides_span = a.strides();
    std::vector<int64_t> strides_vec(strides_span.begin(), strides_span.end());
    DimMeta meta = make_dim_meta(shape_vec, strides_vec);

    constexpr int BLOCK = 256;
    int blocks = compute_grid_size(output_size, BLOCK);

    TENZOR_DISPATCH_FLOATING_TYPES(a.dtype(), "cosine_similarity", [&]() {
        cosine_similarity_along_dim_kernel<scalar_t><<<blocks, BLOCK, 0, stream>>>(
            a.data<scalar_t>(), b.data<scalar_t>(), output.data<scalar_t>(),
            meta, ndim, actual_dim, output_size, dim_size, static_cast<float>(eps));
        CUDA_CHECK(cudaGetLastError());
    });

    return output;
}

Tensor cosine_similarity_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = resolve_stream(nullptr, inputs[0]);
    if (!attrs.empty() && attrs.has(AttrKey::Stream)) {
        stream = reinterpret_cast<cudaStream_t>(
            static_cast<uint64_t>(attrs.get_int(AttrKey::Stream, 0)));
        guard = StreamGuard{};
    }
    int64_t dim = attrs.get_int(AttrKey::Dim, 1);
    double eps = attrs.get_float(AttrKey::Eps, 1e-8);
    return cosine_similarity_kernel(inputs[0], inputs[1], dim, eps, stream);
}

// ============================================================================
// Renorm: scale slices along dim so that p-norm <= maxnorm
// ============================================================================

template<typename T>
__global__ void renorm_compute_norm_kernel(
    const T* input,
    T* norms,         // one norm per slice — slice_idx ranges over shape[dim]
    DimMeta meta,
    int64_t ndim,
    int64_t dim,
    int64_t num_slices,       // == shape[dim]
    int64_t slice_size,       // == product of shape[d] for d != dim
    float p
) {
    int64_t slice_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (slice_idx >= num_slices) return;

    // Sum |x|^p over all elements with index[dim] == slice_idx, iterating
    // across the complementary axes. This matches torch.renorm semantics:
    // each sub-tensor taken at a fixed value of dim has its p-norm clamped
    // to maxnorm.
    int64_t indices[DIM_META_MAX_RANK] = {};
    indices[dim] = slice_idx;

    T acc = T(0);
    for (int64_t flat = 0; flat < slice_size; ++flat) {
        // Unpack flat index into multi-dim coords for dims != dim.
        int64_t tmp = flat;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            if (d == dim) continue;
            indices[d] = tmp % meta.shape[d];
            tmp /= meta.shape[d];
        }
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; ++d) {
            in_idx += indices[d] * meta.strides[d];
        }
        T val = input[in_idx];
        T absval = val < T(0) ? -val : val;
        if (p == 1.0f) {
            acc += absval;
        } else if (p == 2.0f) {
            acc += val * val;
        } else {
            acc += pow(absval, T(p));
        }
    }

    if (p == 2.0f) {
        norms[slice_idx] = sqrt(acc);
    } else if (p != 1.0f) {
        norms[slice_idx] = pow(acc, T(1.0f / p));
    } else {
        norms[slice_idx] = acc;
    }
}

template<typename T>
__global__ void renorm_scale_kernel(
    const T* input,
    T* output,
    const T* norms,
    DimMeta meta,            // shape + INPUT strides (for the load)
    DimMeta out_meta,        // contiguous OUTPUT strides (for the store)
    int64_t ndim,
    int64_t dim,
    int64_t num_slices,       // == shape[dim]
    int64_t slice_size,       // == product of shape[d] for d != dim
    float maxnorm
) {
    int64_t total = num_slices * slice_size;
    TENZOR_CUDA_KERNEL_LOOP(flat_idx, total) {
        int64_t slice_idx = flat_idx / slice_size;
        int64_t within_slice = flat_idx % slice_size;

        int64_t indices[DIM_META_MAX_RANK] = {};
        indices[dim] = slice_idx;
        int64_t tmp = within_slice;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            if (d == dim) continue;
            indices[d] = tmp % meta.shape[d];
            tmp /= meta.shape[d];
        }

        // Load from the (possibly non-contiguous) input using its strides, but
        // store to the freshly allocated, contiguous output using the output's
        // own strides. Using input strides for the write scatters values to the
        // wrong offsets (and can write out of bounds) for non-contiguous input.
        int64_t in_idx = 0;
        int64_t out_idx = 0;
        for (int64_t d = 0; d < ndim; ++d) {
            in_idx += indices[d] * meta.strides[d];
            out_idx += indices[d] * out_meta.strides[d];
        }

        T norm_val = norms[slice_idx];
        T mn = static_cast<T>(maxnorm);
        T scale = (norm_val > mn) ? (mn / norm_val) : T(1);
        output[out_idx] = input[in_idx] * scale;
    }
}

auto renorm_kernel(const Tensor& input, double p, int64_t dim, double maxnorm, cudaStream_t stream) -> Tensor {
    // Upcast FP16/BF16 to Float32
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor result = renorm_kernel(input.to(DType::Float32), p, dim, maxnorm, stream);
        return result.to(orig);
    }

    auto [resolved_stream, stream_guard] = resolve_stream(stream, input);
    stream = resolved_stream;

    auto shape_span = input.shape();
    int64_t ndim = static_cast<int64_t>(shape_span.size());

    // Normalize dim
    int64_t actual_dim = dim;
    if (actual_dim < 0) actual_dim += ndim;
    if (actual_dim < 0 || actual_dim >= ndim) {
        throw std::runtime_error("renorm: dim out of range");
    }

    // torch.renorm semantics: there are shape[dim] slices (one per index along
    // dim), and each slice consists of the product of the other dimensions'
    // extents. Previously these were swapped (norm computed *along* dim) —
    // see renorm_compute_norm_kernel above.
    int64_t num_slices = shape_span[actual_dim];
    int64_t slice_size = 1;
    for (int64_t d = 0; d < ndim; d++) {
        if (d != actual_dim) slice_size *= shape_span[d];
    }

    std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
    auto strides_span = input.strides();
    std::vector<int64_t> strides_vec(strides_span.begin(), strides_span.end());
    DimMeta meta = make_dim_meta(shape_vec, strides_vec);

    // Output has same shape as input and is freshly allocated -> contiguous
    // (row-major) layout. The scale kernel reads input via its (possibly
    // non-contiguous) strides but must store using the output's contiguous
    // strides; otherwise non-contiguous input scatters to wrong offsets.
    std::vector<int64_t> out_strides_vec(ndim, 1);
    for (int64_t d = ndim - 2; d >= 0; --d) {
        out_strides_vec[d] = out_strides_vec[d + 1] * shape_vec[d + 1];
    }
    DimMeta out_meta = make_dim_meta(shape_vec, out_strides_vec);

    // Output has same shape as input
    Tensor output(shape_vec, input.dtype(), input.device());

    // Temporary buffer for per-slice norms
    Tensor norms({num_slices}, input.dtype(), input.device());

    constexpr int BLOCK = 256;

    TENZOR_DISPATCH_FLOATING_TYPES(input.dtype(), "renorm", [&]() {
        // Step 1: compute norms per slice
        int norm_blocks = (num_slices + BLOCK - 1) / BLOCK;
        renorm_compute_norm_kernel<scalar_t><<<norm_blocks, BLOCK, 0, stream>>>(
            input.data<scalar_t>(), norms.data<scalar_t>(), meta,
            ndim, actual_dim, num_slices, slice_size, static_cast<float>(p));
        CUDA_CHECK(cudaGetLastError());

        // Step 2: scale elements where norm > maxnorm
        int64_t total = num_slices * slice_size;
        dim3 grid, block;
        compute_launch_config_1d(total, grid, block);
        renorm_scale_kernel<scalar_t><<<grid, block, 0, stream>>>(
            input.data<scalar_t>(), output.data<scalar_t>(), norms.data<scalar_t>(),
            meta, out_meta, ndim, actual_dim, num_slices, slice_size, static_cast<float>(maxnorm));
        CUDA_CHECK(cudaGetLastError());
    });

    return output;
}

Tensor renorm_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    auto [stream, guard] = resolve_stream(nullptr, inputs[0]);
    if (!attrs.empty() && attrs.has(AttrKey::Stream)) {
        stream = reinterpret_cast<cudaStream_t>(
            static_cast<uint64_t>(attrs.get_int(AttrKey::Stream, 0)));
        guard = StreamGuard{};
    }
    double p_val = attrs.get_float(AttrKey::P, 2.0);
    int64_t dim = attrs.get_int(AttrKey::Dim, 0);
    double maxnorm = attrs.get_float(AttrKey::MaxNorm, 1.0);
    return renorm_kernel(inputs[0], p_val, dim, maxnorm, stream);
}

} // namespace cuda
} // namespace tenzor
