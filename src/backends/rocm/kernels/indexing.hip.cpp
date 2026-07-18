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
#include <limits>
#include "../rocm_error.hpp"
#include "tenzor/ops/creation.hpp"

// Plain-old-data copy types for dtype-agnostic indexing/gather/scatter of
// element widths that have no dedicated arithmetic kernel. Movement ops only
// copy whole elements, so the value semantics are irrelevant — only the byte
// width matters. Complex64 == 8 bytes (uint64_t), Complex128 == 16 bytes.
namespace {
struct alignas(16) Bytes16 {
    uint64_t lo; uint64_t hi;
    // Only present so reduction-capable kernels (scatter) instantiate; movement
    // ops use plain assignment, so this lane-wise add is never exercised for
    // Complex128 (which never takes the reduce path).
    __host__ __device__ Bytes16& operator+=(const Bytes16& o) { lo += o.lo; hi += o.hi; return *this; }
};
}

namespace tenzor {
namespace rocm {

// Grid-stride loop for HIP kernels
#define HIP_KERNEL_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

// Host-side index bounds validation for gather/index_select/scatter.
// The on-device kernels silently skip out-of-range indices, which diverges
// from the CPU reference that throws std::out_of_range. Indices live on-device,
// so copy them to host and validate against [-dim_size, dim_size) (negative
// indices are normalized exactly like the kernels do), throwing on the first
// out-of-range value to match CPU/PyTorch semantics. `op_name` is used to build
// a message that mirrors the corresponding CPU error string.
inline void validate_index_bounds(const Tensor& indices, int64_t dim_size,
                                   const char* op_name) {
    // Indices are read as int64_t both here and on-device. Reinterpreting a
    // narrower (e.g. Int32) buffer as int64 would read half as many elements
    // with garbage values, so reject non-Int64 indices cleanly (matches the CPU
    // reference, which throws for any non-Int64 gather/scatter/select index).
    if (indices.dtype() != DType::Int64) {
        throw std::invalid_argument(
            std::string(op_name) + ": index tensor must have dtype Int64");
    }
    int64_t n = indices.numel();
    if (n == 0) return;
    std::vector<int64_t> host(static_cast<size_t>(n));
    HIP_CHECK(hipMemcpy(host.data(), indices.data<int64_t>(),
                        static_cast<size_t>(n) * sizeof(int64_t),
                        hipMemcpyDeviceToHost));
    for (int64_t i = 0; i < n; ++i) {
        int64_t v = host[static_cast<size_t>(i)];
        if (v < 0) v += dim_size;  // mirror kernel negative-index normalization
        if (v < 0 || v >= dim_size) {
            throw std::out_of_range(
                std::string(op_name) + ": index " + std::to_string(host[static_cast<size_t>(i)]) +
                " out of range for dimension of size " + std::to_string(dim_size));
        }
    }
}

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
// Gather / Scatter shared per-dimension metadata
// ==============================================================================

// Maximum rank supported by the per-dimension gather/scatter decode below.
namespace {
constexpr int MAX_GATHER_SCATTER_DIMS = 16;
}

// Per-dimension metadata for gather/scatter. The flat index ranges over the
// INDEX tensor (== output for gather), so decode each coordinate against the
// index's own contiguous strides, then re-linearise against the input/output's
// contiguous strides. The collapsed outer/inner formula is only correct when
// index and input share extents on every non-dim axis; PyTorch/CPU allow the
// index extent to be smaller than the input on a non-dim axis, which requires
// this per-dim decode (mirrors gather_impl / scatter_impl in CPU indexing.cpp).
struct GatherScatterMeta {
    int64_t idx_strides[MAX_GATHER_SCATTER_DIMS];   // contiguous strides of index shape
    int64_t self_strides[MAX_GATHER_SCATTER_DIMS];  // contiguous strides of input/output shape
};

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
    int ndim,
    int dim,
    int64_t dim_size,
    int64_t total_output,
    GatherScatterMeta meta
) {
    HIP_KERNEL_LOOP(idx, total_output) {
        // Decode the flat output position (== index position) per-dimension from
        // the index strides, then re-linearise against the input's contiguous
        // strides, substituting the gathered coordinate on the `dim` axis.
        int64_t rem = idx;
        int64_t input_offset = 0;
        int64_t gather_idx = indices[idx];
        if (gather_idx < 0) gather_idx += dim_size;
        if (gather_idx < 0 || gather_idx >= dim_size) continue;
        for (int d = 0; d < ndim; ++d) {
            int64_t coord = rem / meta.idx_strides[d];
            rem %= meta.idx_strides[d];
            if (d == dim) {
                input_offset += gather_idx * meta.self_strides[d];
            } else {
                input_offset += coord * meta.self_strides[d];
            }
        }
        output[idx] = input[input_offset];
    }
}

auto gather_hip(
    const Tensor& input_arg,
    int64_t dim,
    const Tensor& indices_arg
) -> Tensor {

    // The kernel addresses the input/index with contiguous strides, so a
    // non-contiguous (transposed/sliced/permuted) view would read the wrong
    // storage elements. Materialise contiguous copies (matches the CPU
    // reference, which does the same in gather_kernel).
    Tensor input = input_arg.is_contiguous() ? input_arg : input_arg.contiguous();
    Tensor indices = indices_arg.is_contiguous() ? indices_arg : indices_arg.contiguous();

    auto input_shape = input.shape();
    auto indices_shape = indices.shape();

    // Indices must be Int64 — the kernel reads them as int64_t. The CPU
    // reference throws cleanly on any other dtype; do the same here rather than
    // reinterpreting a narrower buffer as int64 (silent garbage/OOB).
    if (indices.dtype() != DType::Int64) {
        throw std::invalid_argument("gather: index tensor must have dtype Int64");
    }

    // Normalize negative dim (PyTorch semantics: dim=-1 → last)
    if (dim < 0) dim += static_cast<int64_t>(input_shape.size());
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::out_of_range("gather_hip: dim out of range");
    }

    // PyTorch/CPU gather: index must have the same rank as input, and along every
    // non-dim axis its extent must not exceed the input's (it may select a
    // sub-block there). The output shape is exactly the index shape.
    if (indices_shape.size() != input_shape.size()) {
        throw std::invalid_argument("gather: index must have same rank as input");
    }
    int ndim = static_cast<int>(input_shape.size());
    if (ndim > MAX_GATHER_SCATTER_DIMS) {
        throw std::runtime_error("gather_hip: ndim exceeds MAX_GATHER_SCATTER_DIMS");
    }
    for (int d = 0; d < ndim; ++d) {
        if (d == dim) continue;
        if (indices_shape[d] > input_shape[d]) {
            throw std::out_of_range(
                "gather: index.size(" + std::to_string(d) + ") exceeds self.size(" +
                std::to_string(d) + ") (index extent must be <= input extent on "
                "non-gather dims)");
        }
    }

    // Output shape is exactly the index shape (PyTorch semantics).
    std::vector<int64_t> output_shape(indices_shape.begin(), indices_shape.end());

    Tensor output = Tensor(output_shape, input.dtype(), input.device());

    int64_t dim_size = input_shape[dim];
    int64_t total_output = output.numel();

    // Per-dim metadata: index strides (to decode each output coordinate) and the
    // INPUT's contiguous strides (to re-linearise every non-dim coordinate and
    // the substituted gathered index on the `dim` axis).
    GatherScatterMeta meta{};
    {
        int64_t is = 1, ss = 1;
        for (int d = ndim - 1; d >= 0; --d) {
            meta.idx_strides[d] = is;
            meta.self_strides[d] = ss;
            is *= indices_shape[d];
            ss *= input_shape[d];
        }
    }

    // Match CPU semantics: throw on any out-of-range index instead of silently
    // skipping (kernel uses `continue`, leaving output uninitialized).
    validate_index_bounds(indices, dim_size, "gather");

    if (total_output == 0) {
        return output;
    }

    int threads = 256;
    int blocks = static_cast<int>((total_output + threads - 1) / threads);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(gather_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<float>(),
            indices.data<int64_t>(),
            output.data<float>(),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(gather_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<double>(),
            indices.data<int64_t>(),
            output.data<double>(),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(gather_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<int32_t>(),
            indices.data<int64_t>(),
            output.data<int32_t>(),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(gather_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<int64_t>(),
            indices.data<int64_t>(),
            output.data<int64_t>(),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(gather_kernel<__half>,
            dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            indices.data<int64_t>(),
            reinterpret_cast<__half*>(output.data<Float16>()),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(gather_kernel<hip_bfloat16>,
            dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            indices.data<int64_t>(),
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Int16 || input.dtype() == DType::UInt16) {
        hipLaunchKernelGGL(gather_kernel<uint16_t>, dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const uint16_t*>(input.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<uint16_t*>(output.data_ptr()),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::UInt32) {
        hipLaunchKernelGGL(gather_kernel<uint32_t>, dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const uint32_t*>(input.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<uint32_t*>(output.data_ptr()),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::UInt64 || input.dtype() == DType::Complex64) {
        hipLaunchKernelGGL(gather_kernel<uint64_t>, dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const uint64_t*>(input.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<uint64_t*>(output.data_ptr()),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Complex128) {
        hipLaunchKernelGGL(gather_kernel<Bytes16>, dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const Bytes16*>(input.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<Bytes16*>(output.data_ptr()),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
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

// Types for which atomicAddHelper (and thus scatter-with-reduction) is valid.
// Movement-only element types (Bytes16 for Complex128, uint16_t, etc.) are not
// listed; scatter_kernel gates the atomic branch on this via `if constexpr`.
template<typename T> struct scatter_atomic_capable : std::false_type {};
template<> struct scatter_atomic_capable<float>        : std::true_type {};
template<> struct scatter_atomic_capable<double>       : std::true_type {};
template<> struct scatter_atomic_capable<int32_t>      : std::true_type {};
template<> struct scatter_atomic_capable<int64_t>      : std::true_type {};
template<> struct scatter_atomic_capable<uint32_t>     : std::true_type {};
template<> struct scatter_atomic_capable<uint64_t>     : std::true_type {};
template<> struct scatter_atomic_capable<__half>       : std::true_type {};
template<> struct scatter_atomic_capable<hip_bfloat16> : std::true_type {};

template<typename T>
__global__ void scatter_kernel(
    T* output,
    const int64_t* indices,
    const T* src,
    int ndim,
    int dim,
    int64_t dim_size,
    int64_t total_scatter,
    GatherScatterMeta meta,
    bool reduce_add
) {
    HIP_KERNEL_LOOP(idx, total_scatter) {
        // The flat index ranges over the INDEX/SRC tensor. Decode each coordinate
        // against the index strides; the index value at this flat position is
        // simply indices[idx] (index is contiguous). Re-linearise against the
        // output's contiguous strides, substituting the scatter coordinate on the
        // `dim` axis. Mirrors CPU scatter_impl.
        int64_t scatter_idx = indices[idx];
        if (scatter_idx < 0) {
            scatter_idx += dim_size;
        }

        // Bounds checking
        if (scatter_idx >= 0 && scatter_idx < dim_size) {
            int64_t rem = idx;
            int64_t output_offset = 0;
            for (int d = 0; d < ndim; ++d) {
                int64_t coord = rem / meta.idx_strides[d];
                rem %= meta.idx_strides[d];
                if (d == dim) {
                    output_offset += scatter_idx * meta.self_strides[d];
                } else {
                    output_offset += coord * meta.self_strides[d];
                }
            }

            if (reduce_add) {
                // atomicAdd only exists for the arithmetic types below. For
                // movement-only element types (e.g. Complex128 packed as a
                // 16-byte POD, where a 128-bit atomic-add doesn't exist), gate
                // the call out at compile time so the kernel still instantiates
                // — a pure scatter (overwrite) never takes this branch anyway.
                if constexpr (scatter_atomic_capable<T>::value) {
                    atomicAddHelper(&output[output_offset], src[idx]);
                } else {
                    output[output_offset] = src[idx];
                }
            } else {
                output[output_offset] = src[idx];
            }
        }
    }
}

// Complex scatter-with-reduce="add". A complex element is two contiguous reals
// (RealT == float for Complex64, double for Complex128). A 64/128-bit integer
// atomicAdd over the packed bits is mathematically wrong (carry crosses the
// real/imag boundary), so accumulate the real and imaginary scalars with two
// independent real atomicAdds. `output`/`src` point at the underlying real
// scalar arrays (2 reals per logical element).
template<typename RealT>
__global__ void scatter_complex_add_kernel(
    RealT* output,
    const int64_t* indices,
    const RealT* src,
    int ndim,
    int dim,
    int64_t dim_size,
    int64_t total_scatter,
    GatherScatterMeta meta
) {
    HIP_KERNEL_LOOP(idx, total_scatter) {
        int64_t scatter_idx = indices[idx];
        if (scatter_idx < 0) {
            scatter_idx += dim_size;
        }
        if (scatter_idx >= 0 && scatter_idx < dim_size) {
            int64_t rem = idx;
            int64_t output_offset = 0;
            for (int d = 0; d < ndim; ++d) {
                int64_t coord = rem / meta.idx_strides[d];
                rem %= meta.idx_strides[d];
                if (d == dim) {
                    output_offset += scatter_idx * meta.self_strides[d];
                } else {
                    output_offset += coord * meta.self_strides[d];
                }
            }
            atomicAddHelper(&output[2 * output_offset + 0], src[2 * idx + 0]);
            atomicAddHelper(&output[2 * output_offset + 1], src[2 * idx + 1]);
        }
    }
}

auto scatter_hip(
    Tensor& output,
    int64_t dim,
    const Tensor& indices_arg,
    const Tensor& src_arg,
    const std::string& reduce
) -> Tensor {

    auto output_shape = output.shape();

    // The kernel decodes index/src positions linearly, so both must be
    // contiguous. Output is the in-place destination (caller-owned, contiguous).
    Tensor indices = indices_arg.is_contiguous() ? indices_arg : indices_arg.contiguous();
    Tensor src = src_arg.is_contiguous() ? src_arg : src_arg.contiguous();
    auto indices_shape = indices.shape();

    // Indices must be Int64 — the kernel reads them as int64_t (matches CPU).
    if (indices.dtype() != DType::Int64) {
        throw std::invalid_argument("scatter: index tensor must have dtype Int64");
    }

    // Normalize dimension
    int64_t ndim = output.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("scatter_hip: dim out of range");
    }

    // PyTorch/CPU scatter: index must have the same rank as output, and along
    // every non-scatter axis its extent must not exceed the output's.
    if (static_cast<int64_t>(indices_shape.size()) != ndim) {
        throw std::invalid_argument("scatter: index must have same rank as self");
    }
    int ndim_i = static_cast<int>(ndim);
    if (ndim_i > MAX_GATHER_SCATTER_DIMS) {
        throw std::runtime_error("scatter_hip: ndim exceeds MAX_GATHER_SCATTER_DIMS");
    }
    for (int d = 0; d < ndim_i; ++d) {
        if (d == dim) continue;
        if (indices_shape[d] > output_shape[d]) {
            throw std::out_of_range(
                "scatter: index.size(" + std::to_string(d) + ") exceeds self.size(" +
                std::to_string(d) + ") (index extent must be <= self extent on "
                "non-scatter dims)");
        }
    }

    int64_t dim_size = output_shape[dim];
    int64_t total_scatter = indices.numel();

    // Per-dim metadata: index strides (to decode each flat index/src position)
    // and the OUTPUT's contiguous strides (to re-linearise each non-dim
    // coordinate and the substituted scatter index on the `dim` axis).
    GatherScatterMeta meta{};
    {
        int64_t is = 1, ss = 1;
        for (int d = ndim_i - 1; d >= 0; --d) {
            meta.idx_strides[d] = is;
            meta.self_strides[d] = ss;
            is *= indices_shape[d];
            ss *= output_shape[d];
        }
    }

    bool reduce_add = (reduce == "add");

    // Match CPU semantics: throw on any out-of-range index instead of silently
    // dropping the write.
    validate_index_bounds(indices, dim_size, "scatter");

    if (total_scatter == 0) {
        return output;
    }

    int threads = 256;
    int blocks = (total_scatter + threads - 1) / threads;

    if (output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(scatter_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<float>(),
            indices.data<int64_t>(),
            src.data<float>(),
            ndim_i,
            static_cast<int>(dim),
            dim_size,
            total_scatter,
            meta,
            reduce_add
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(scatter_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<double>(),
            indices.data<int64_t>(),
            src.data<double>(),
            ndim_i,
            static_cast<int>(dim),
            dim_size,
            total_scatter,
            meta,
            reduce_add
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Float16) {
        hipLaunchKernelGGL(scatter_kernel<__half>,
            dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<__half*>(output.data<Float16>()),
            indices.data<int64_t>(),
            reinterpret_cast<const __half*>(src.data<Float16>()),
            ndim_i,
            static_cast<int>(dim),
            dim_size,
            total_scatter,
            meta,
            reduce_add
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(scatter_kernel<hip_bfloat16>,
            dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            indices.data<int64_t>(),
            reinterpret_cast<const hip_bfloat16*>(src.data<BFloat16>()),
            ndim_i,
            static_cast<int>(dim),
            dim_size,
            total_scatter,
            meta,
            reduce_add
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Int32) {
        hipLaunchKernelGGL(scatter_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<int32_t>(),
            indices.data<int64_t>(),
            src.data<int32_t>(),
            ndim_i,
            static_cast<int>(dim),
            dim_size,
            total_scatter,
            meta,
            reduce_add
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Int64) {
        hipLaunchKernelGGL(scatter_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, 0,
            output.data<int64_t>(),
            indices.data<int64_t>(),
            src.data<int64_t>(),
            ndim_i,
            static_cast<int>(dim),
            dim_size,
            total_scatter,
            meta,
            reduce_add
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::UInt32) {
        hipLaunchKernelGGL(scatter_kernel<uint32_t>, dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<uint32_t*>(output.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<const uint32_t*>(src.data_ptr()),
            ndim_i, static_cast<int>(dim), dim_size, total_scatter, meta, reduce_add);
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Complex64) {
        if (reduce_add) {
            // Componentwise real/imag atomicAdd; a packed 64-bit integer add
            // would carry across the real/imag boundary (wrong result).
            hipLaunchKernelGGL(scatter_complex_add_kernel<float>, dim3(blocks), dim3(threads), 0, 0,
                reinterpret_cast<float*>(output.data_ptr()), indices.data<int64_t>(),
                reinterpret_cast<const float*>(src.data_ptr()),
                ndim_i, static_cast<int>(dim), dim_size, total_scatter, meta);
        } else {
            hipLaunchKernelGGL(scatter_kernel<uint64_t>, dim3(blocks), dim3(threads), 0, 0,
                reinterpret_cast<uint64_t*>(output.data_ptr()), indices.data<int64_t>(),
                reinterpret_cast<const uint64_t*>(src.data_ptr()),
                ndim_i, static_cast<int>(dim), dim_size, total_scatter, meta, reduce_add);
        }
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::UInt64) {
        hipLaunchKernelGGL(scatter_kernel<uint64_t>, dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<uint64_t*>(output.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<const uint64_t*>(src.data_ptr()),
            ndim_i, static_cast<int>(dim), dim_size, total_scatter, meta, reduce_add);
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Complex128) {
        if (reduce_add) {
            hipLaunchKernelGGL(scatter_complex_add_kernel<double>, dim3(blocks), dim3(threads), 0, 0,
                reinterpret_cast<double*>(output.data_ptr()), indices.data<int64_t>(),
                reinterpret_cast<const double*>(src.data_ptr()),
                ndim_i, static_cast<int>(dim), dim_size, total_scatter, meta);
        } else {
            hipLaunchKernelGGL(scatter_kernel<Bytes16>, dim3(blocks), dim3(threads), 0, 0,
                reinterpret_cast<Bytes16*>(output.data_ptr()), indices.data<int64_t>(),
                reinterpret_cast<const Bytes16*>(src.data_ptr()),
                ndim_i, static_cast<int>(dim), dim_size, total_scatter, meta, reduce_add);
        }
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
    const Tensor& input_orig,
    int64_t dim,
    const Tensor& indices_orig
) -> Tensor {
    // The kernel reads input and indices with flat/contiguous addressing, so a
    // non-contiguous view (transposed input, or a strided index like
    // arange(6)[::2]) would read the wrong storage. Shadow with contiguous copies.
    const Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
    const Tensor indices = indices_orig.is_contiguous() ? indices_orig : indices_orig.contiguous();

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

    // Match CPU semantics: throw on any out-of-range index instead of silently
    // skipping (kernel writes nothing for OOB, leaving output uninitialized).
    validate_index_bounds(indices, dim_size, "index_select");

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
    } else if (input.dtype() == DType::Int16 || input.dtype() == DType::UInt16) {
        hipLaunchKernelGGL(index_select_kernel<uint16_t>, dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const uint16_t*>(input.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<uint16_t*>(output.data_ptr()), outer_size, dim_size, inner_size, num_indices);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::UInt32) {
        hipLaunchKernelGGL(index_select_kernel<uint32_t>, dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const uint32_t*>(input.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<uint32_t*>(output.data_ptr()), outer_size, dim_size, inner_size, num_indices);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::UInt64 || input.dtype() == DType::Complex64) {
        hipLaunchKernelGGL(index_select_kernel<uint64_t>, dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const uint64_t*>(input.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<uint64_t*>(output.data_ptr()), outer_size, dim_size, inner_size, num_indices);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Complex128) {
        hipLaunchKernelGGL(index_select_kernel<Bytes16>, dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const Bytes16*>(input.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<Bytes16*>(output.data_ptr()), outer_size, dim_size, inner_size, num_indices);
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

// NOTE: the dead float-value masked_fill_hip overload was removed (it lost
// Float64/Complex mantissa via static_cast<double>(float)). The live overload
// below takes `double value` + stream and is the one declared/dispatched in
// rocm_kernel_registry.cpp.

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

// Note: the order-preserving, all-dtype masked_select_hip(input, mask, stream)
// lives further down (using hipcub DeviceSelect::Flagged). The previous
// no-stream masked_select_hip + masked_select_kernel here were unreferenced
// dead duplicates that launched on the default stream; removed to avoid
// divergent edits. masked_select_count_kernel above is still used by the live
// stream variant.

// ==============================================================================
// Take Operation (1D indexing)
// ==============================================================================

template<typename T>
__global__ void take_kernel(
    const T* input,
    const int64_t* indices,
    T* output,
    int64_t input_size,
    int64_t indices_size,
    int* error_flag
) {
    HIP_KERNEL_LOOP(idx, indices_size) {
        int64_t index = indices[idx];

        // Handle negative indices
        if (index < 0) {
            index += input_size;
        }

        // Bounds checking. F085: flag an out-of-range index so the host can throw
        // (matches CPU/CUDA), and write a defined value rather than leaking stale
        // device memory into the result for hostile/buggy indices.
        if (index >= 0 && index < input_size) {
            output[idx] = input[index];
        } else {
            atomicExch(error_flag, 1);
            output[idx] = T(0);
        }
    }
}

// Note: the live take_hip(input, indices, stream) overload lives further down
// and reuses take_kernel above. The previous no-stream take_hip duplicate here
// launched on the default stream and was unreferenced; removed.

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

// Normalise a numeric condition tensor to bool (nonzero -> true). The where()
// contract accepts a Bool OR numeric condition (see the CPU kernel), e.g. the
// attention causal mask is built as a Float32 triu-of-ones.
template<typename T>
__global__ void cond_to_bool_kernel(const T* in, bool* out, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        out[idx] = (in[idx] != T(0));
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
    // Empty tensor: a zero-block grid makes HIP reject the launch
    // ("invalid configuration argument"). Return the empty result.
    if (total_elements == 0) return output;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    // Accept a non-Bool (numeric) condition by converting it to a bool buffer
    // first (nonzero -> true), matching the CPU/CUDA where() contract. We sync
    // before returning in this branch so the temporary bool buffer outlives the
    // async where launch.
    Tensor cond_bool;
    bool used_temp_cond = false;
    const bool* cond_ptr;
    // Outer-scope temporary for a widened Float16/BFloat16 condition buffer —
    // must outlive the async cond_to_bool_kernel launch below, which only
    // completes at the hipStreamSynchronize further down this function.
    Tensor cond_f32_widened;
    if (condition.dtype() == DType::Bool) {
        cond_ptr = condition.data<bool>();
    } else {
        cond_bool = Tensor(std::vector<int64_t>(condition.shape().begin(), condition.shape().end()),
                           DType::Bool, condition.device());
        used_temp_cond = true;
        int cblocks = (condition.numel() + threads - 1) / threads;
        if (condition.dtype() == DType::Float32) {
            hipLaunchKernelGGL(cond_to_bool_kernel<float>, dim3(cblocks), dim3(threads), 0, stream,
                condition.data<float>(), cond_bool.data<bool>(), condition.numel());
        } else if (condition.dtype() == DType::Float64) {
            hipLaunchKernelGGL(cond_to_bool_kernel<double>, dim3(cblocks), dim3(threads), 0, stream,
                condition.data<double>(), cond_bool.data<bool>(), condition.numel());
        } else if (condition.dtype() == DType::Float16 || condition.dtype() == DType::BFloat16) {
            // Match CPU/CUDA: widen to Float32 first (lossless) instead of
            // adding half/bf16 comparison-operator instantiations. CPU/CUDA
            // both accept Float16/BFloat16 conditions; ROCm previously threw
            // for them (no branch existed at all).
            cond_f32_widened = condition.to(DType::Float32);
            hipLaunchKernelGGL(cond_to_bool_kernel<float>, dim3(cblocks), dim3(threads), 0, stream,
                cond_f32_widened.data<float>(), cond_bool.data<bool>(), condition.numel());
        } else {
            // Match CPU (indexing.cpp:1084-1089) and CUDA (validate_mask_dtype):
            // a where() condition must be Bool or a floating dtype. ROCm
            // previously accepted Int32/Int64/UInt8 here too, silently
            // succeeding on a call CPU/CUDA both reject.
            throw std::invalid_argument("where_hip: condition must be Bool or a floating dtype");
        }
        HIP_POST_LAUNCH_CHECK();
        cond_ptr = cond_bool.data<bool>();
    }

    if (x.dtype() == DType::Float32) {
        hipLaunchKernelGGL(where_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            cond_ptr,
            x.data<float>(),
            y.data<float>(),
            output.data<float>(),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (x.dtype() == DType::Float64) {
        hipLaunchKernelGGL(where_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            cond_ptr,
            x.data<double>(),
            y.data<double>(),
            output.data<double>(),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (x.dtype() == DType::Int32) {
        hipLaunchKernelGGL(where_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, stream,
            cond_ptr,
            x.data<int32_t>(),
            y.data<int32_t>(),
            output.data<int32_t>(),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (x.dtype() == DType::Int64) {
        hipLaunchKernelGGL(where_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            cond_ptr,
            x.data<int64_t>(),
            y.data<int64_t>(),
            output.data<int64_t>(),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (x.dtype() == DType::Float16) {
        hipLaunchKernelGGL(where_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            cond_ptr,
            reinterpret_cast<const __half*>(x.data<Float16>()),
            reinterpret_cast<const __half*>(y.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (x.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(where_kernel<hip_bfloat16>,
            dim3(blocks), dim3(threads), 0, stream,
            cond_ptr,
            reinterpret_cast<const hip_bfloat16*>(x.data<BFloat16>()),
            reinterpret_cast<const hip_bfloat16*>(y.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            total_elements
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (x.dtype() == DType::Complex64) {
        // Complex64 == 8-byte element; where is pure selection/copy.
        hipLaunchKernelGGL(where_kernel<uint64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            cond_ptr,
            reinterpret_cast<const uint64_t*>(x.data_ptr()),
            reinterpret_cast<const uint64_t*>(y.data_ptr()),
            reinterpret_cast<uint64_t*>(output.data_ptr()),
            total_elements);
        HIP_POST_LAUNCH_CHECK();
    } else if (x.dtype() == DType::Complex128) {
        // Complex128 == 16-byte element; where is pure selection/copy, so a
        // 16-byte bit-copy suffices (mirrors masked_fill/gather in this file).
        hipLaunchKernelGGL(where_kernel<Bytes16>,
            dim3(blocks), dim3(threads), 0, stream,
            cond_ptr,
            reinterpret_cast<const Bytes16*>(x.data_ptr()),
            reinterpret_cast<const Bytes16*>(y.data_ptr()),
            reinterpret_cast<Bytes16*>(output.data_ptr()),
            total_elements);
        HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("where_hip: Unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    // The converted-condition buffer is a function-local temporary; ensure the
    // where launch (and its conversion launch) complete before it is freed.
    if (used_temp_cond) {
        HIP_CHECK(hipStreamSynchronize(stream));
    }
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

    HIP_CHECK(hipMalloc(&d_input_ptrs, cont_tensors.size() * sizeof(void*)));
    HIP_CHECK(hipMalloc(&d_dim_sizes, cont_tensors.size() * sizeof(int64_t)));

    // RAII so both scratch buffers are freed even if a dtype-dispatched
    // HIP_POST_LAUNCH_CHECK() throws (previously the frees only ran on the
    // normal path after the switch, leaking on any launch failure).
    struct CatScratchGuard {
        void* a; void* b;
        ~CatScratchGuard() noexcept { if (a) (void)hipFree(a); if (b) (void)hipFree(b); }
    } cat_scratch{d_input_ptrs, d_dim_sizes};

    HIP_CHECK(hipMemcpy(d_input_ptrs, h_input_ptrs.data(), cont_tensors.size() * sizeof(void*), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dim_sizes, dim_sizes.data(), cont_tensors.size() * sizeof(int64_t), hipMemcpyHostToDevice));

    int64_t total_elements = outer_size * total_dim_size * inner_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (first.dtype() == DType::Float32) {
        hipLaunchKernelGGL(cat_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            (const float* const*)d_input_ptrs,
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
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            static_cast<int64_t>(cont_tensors.size()),
            outer_size,
            total_dim_size,
            inner_size,
            d_dim_sizes
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (first.dtype() == DType::Complex64) {
        // Complex64 == 8-byte element; cat is pure byte-wise concatenation.
        hipLaunchKernelGGL(cat_kernel<uint64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            (const uint64_t* const*)d_input_ptrs,
            reinterpret_cast<uint64_t*>(output.data_ptr()),
            static_cast<int64_t>(cont_tensors.size()),
            outer_size, total_dim_size, inner_size, d_dim_sizes);
        HIP_POST_LAUNCH_CHECK();
    } else if (first.dtype() == DType::Complex128) {
        hipLaunchKernelGGL(cat_kernel<Bytes16>,
            dim3(blocks), dim3(threads), 0, stream,
            (const Bytes16* const*)d_input_ptrs,
            reinterpret_cast<Bytes16*>(output.data_ptr()),
            static_cast<int64_t>(cont_tensors.size()),
            outer_size, total_dim_size, inner_size, d_dim_sizes);
        HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("cat_hip: Unsupported dtype");
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_POST_LAUNCH_CHECK();
    // cat_scratch frees d_input_ptrs / d_dim_sizes on scope exit.

    return output;
}

// ==============================================================================
// Put Operation
// ==============================================================================

// Types for which atomicAddHelper (and thus put-with-accumulate) is valid.
// Movement-only element types (Bytes16 for Complex128, uint16_t, etc.) are not
// listed; put_kernel gates the atomic branch on this via `if constexpr`.
template<typename T> struct put_atomic_capable : std::false_type {};
template<> struct put_atomic_capable<float>        : std::true_type {};
template<> struct put_atomic_capable<double>       : std::true_type {};
template<> struct put_atomic_capable<int32_t>      : std::true_type {};
template<> struct put_atomic_capable<int64_t>      : std::true_type {};
template<> struct put_atomic_capable<uint32_t>     : std::true_type {};
template<> struct put_atomic_capable<uint64_t>     : std::true_type {};
template<> struct put_atomic_capable<__half>       : std::true_type {};
template<> struct put_atomic_capable<hip_bfloat16> : std::true_type {};

template<typename T>
__global__ void put_kernel(
    T* output,
    const int64_t* indices,
    const T* source,
    int64_t num_indices,
    int64_t total_size,
    bool accumulate,
    int* error_flag
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
                // atomicAdd only exists for the arithmetic types above. For
                // movement-only element types the accumulate path is never taken
                // (the host routes complex-accumulate through the dedicated
                // componentwise kernel and rejects accumulate for byte-movement
                // types), so gate the call out at compile time to instantiate.
                if constexpr (put_atomic_capable<T>::value) {
                    atomicAddHelper(&output[target_idx], source[idx]);
                } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
                    // No 8-bit atomicAdd on HIP. CAS-loop on the aligned 32-bit
                    // word containing this byte (matches CUDA's put_kernel_impl
                    // <int8_t>/<uint8_t> specializations), so CPU/CUDA/ROCm all
                    // support put(accumulate=true) for Int8/UInt8. The word base
                    // is shifted down so the 4-byte access never runs past the
                    // buffer end when total_size % 4 != 0 and target_idx falls in
                    // the last 1-3 bytes.
                    int64_t word_base = target_idx & ~static_cast<int64_t>(3);
                    if (word_base + 4 > total_size) {
                        word_base = total_size - 4;
                    }
                    if (word_base < 0) word_base = 0;  // buffer smaller than 4 bytes
                    unsigned int byte_offset = static_cast<unsigned int>(target_idx - word_base);
                    unsigned int* addr = reinterpret_cast<unsigned int*>(
                        reinterpret_cast<char*>(output) + word_base);
                    unsigned int old_val, new_val;
                    do {
                        old_val = atomicCAS(addr, 0u, 0u);  // Atomic initial read
                        T cur = static_cast<T>((old_val >> (byte_offset * 8)) & 0xFF);
                        T sum = static_cast<T>(cur + source[idx]);
                        new_val = (old_val & ~(0xFFu << (byte_offset * 8))) |
                                  (static_cast<unsigned int>(static_cast<uint8_t>(sum)) << (byte_offset * 8));
                    } while (atomicCAS(addr, old_val, new_val) != old_val);
                } else {
                    output[target_idx] = source[idx];
                }
            } else {
                output[target_idx] = source[idx];
            }
        } else {
            // Out-of-range: record the error; the host wrapper raises a
            // catchable std::out_of_range after synchronizing (matches CPU/
            // CUDA/OneAPI). The write stays gated, so the kernel remains
            // memory-safe -- previously this silently no-op'd instead.
            atomicExch(error_flag, 1);
        }
    }
}

// Complex put-with-accumulate. Mirrors scatter_complex_add_kernel: a complex
// element is two contiguous reals, so a packed integer atomicAdd would carry
// across the real/imag boundary. Accumulate the real and imaginary scalars with
// two independent real atomicAdds. `output`/`source` point at the underlying
// real scalar arrays (2 reals per logical element).
template<typename RealT>
__global__ void put_complex_add_kernel(
    RealT* output,
    const int64_t* indices,
    const RealT* source,
    int64_t num_indices,
    int64_t total_size,
    int* error_flag
) {
    HIP_KERNEL_LOOP(idx, num_indices) {
        int64_t target_idx = indices[idx];
        if (target_idx < 0) {
            target_idx += total_size;
        }
        if (target_idx >= 0 && target_idx < total_size) {
            atomicAddHelper(&output[2 * target_idx + 0], source[2 * idx + 0]);
            atomicAddHelper(&output[2 * target_idx + 1], source[2 * idx + 1]);
        } else {
            atomicExch(error_flag, 1);
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

    // Device-side OOB error flag (matches take_hip above and CUDA's put_kernel_impl):
    // the kernel sets it when an index (after negative adjustment) is still out of
    // range instead of silently no-op'ing, and the host raises a catchable
    // std::out_of_range after synchronizing, matching CPU/CUDA/OneAPI.
    int* d_error_flag = nullptr;
    HIP_CHECK(hipMalloc(&d_error_flag, sizeof(int)));
    HIP_CHECK(hipMemsetAsync(d_error_flag, 0, sizeof(int), stream));

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(put_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<float>(),
            indices.data<int64_t>(),
            source.data<float>(),
            num_indices,
            total_size,
            accumulate,
            d_error_flag
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
            accumulate,
            d_error_flag
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
            accumulate,
            d_error_flag
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
            accumulate,
            d_error_flag
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        // atomicAddHelper<__half> exists, so both accumulate and overwrite work.
        hipLaunchKernelGGL(put_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<__half*>(output.data<Float16>()),
            indices.data<int64_t>(),
            reinterpret_cast<const __half*>(source.data<Float16>()),
            num_indices, total_size, accumulate, d_error_flag);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        // atomicAddHelper<hip_bfloat16> uses round-to-nearest-even accumulation.
        hipLaunchKernelGGL(put_kernel<hip_bfloat16>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            indices.data<int64_t>(),
            reinterpret_cast<const hip_bfloat16*>(source.data<BFloat16>()),
            num_indices, total_size, accumulate, d_error_flag);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::UInt32) {
        hipLaunchKernelGGL(put_kernel<uint32_t>, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<uint32_t*>(output.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<const uint32_t*>(source.data_ptr()),
            num_indices, total_size, accumulate, d_error_flag);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::UInt64) {
        hipLaunchKernelGGL(put_kernel<uint64_t>, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<uint64_t*>(output.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<const uint64_t*>(source.data_ptr()),
            num_indices, total_size, accumulate, d_error_flag);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Int16 || input.dtype() == DType::UInt16) {
        // No 16-bit atomicAdd; only the overwrite path is meaningful for these.
        if (accumulate) {
            HIP_CHECK(hipFree(d_error_flag));
            throw std::runtime_error("put_hip: accumulate not supported for 16-bit integer dtype");
        }
        hipLaunchKernelGGL(put_kernel<uint16_t>, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<uint16_t*>(output.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<const uint16_t*>(source.data_ptr()),
            num_indices, total_size, accumulate, d_error_flag);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Int8 || input.dtype() == DType::UInt8) {
        // put_kernel<uint8_t> now handles accumulate=true via a boundary-safe
        // CAS-loop on the containing 32-bit word (matches CPU's serial add and
        // CUDA's own int8_t/uint8_t CAS specializations).
        hipLaunchKernelGGL(put_kernel<uint8_t>, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<uint8_t*>(output.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<const uint8_t*>(source.data_ptr()),
            num_indices, total_size, accumulate, d_error_flag);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Bool) {
        // Bool accumulate has no defined semantics and CPU explicitly throws
        // for it too (put_kernel accumulate dtype allowlist excludes Bool);
        // only the overwrite path is meaningful here.
        if (accumulate) {
            HIP_CHECK(hipFree(d_error_flag));
            throw std::runtime_error("put_hip: accumulate not supported for Bool dtype");
        }
        hipLaunchKernelGGL(put_kernel<uint8_t>, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<uint8_t*>(output.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<const uint8_t*>(source.data_ptr()),
            num_indices, total_size, accumulate, d_error_flag);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Complex64) {
        if (accumulate) {
            // Componentwise real/imag atomicAdd (packed add would carry across
            // the real/imag boundary).
            hipLaunchKernelGGL(put_complex_add_kernel<float>, dim3(blocks), dim3(threads), 0, stream,
                reinterpret_cast<float*>(output.data_ptr()), indices.data<int64_t>(),
                reinterpret_cast<const float*>(source.data_ptr()),
                num_indices, total_size, d_error_flag);
        } else {
            hipLaunchKernelGGL(put_kernel<uint64_t>, dim3(blocks), dim3(threads), 0, stream,
                reinterpret_cast<uint64_t*>(output.data_ptr()), indices.data<int64_t>(),
                reinterpret_cast<const uint64_t*>(source.data_ptr()),
                num_indices, total_size, accumulate, d_error_flag);
        }
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Complex128) {
        if (accumulate) {
            hipLaunchKernelGGL(put_complex_add_kernel<double>, dim3(blocks), dim3(threads), 0, stream,
                reinterpret_cast<double*>(output.data_ptr()), indices.data<int64_t>(),
                reinterpret_cast<const double*>(source.data_ptr()),
                num_indices, total_size, d_error_flag);
        } else {
            hipLaunchKernelGGL(put_kernel<Bytes16>, dim3(blocks), dim3(threads), 0, stream,
                reinterpret_cast<Bytes16*>(output.data_ptr()), indices.data<int64_t>(),
                reinterpret_cast<const Bytes16*>(source.data_ptr()),
                num_indices, total_size, accumulate, d_error_flag);
        }
        HIP_POST_LAUNCH_CHECK();
    } else {
        HIP_CHECK(hipFree(d_error_flag));
        throw std::runtime_error("put_hip: Unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();

    // Check for out-of-bounds index errors (matches CPU/CUDA/OneAPI which throw).
    int host_error = 0;
    HIP_CHECK(hipMemcpyAsync(&host_error, d_error_flag, sizeof(int),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_error_flag));
    if (host_error != 0) {
        throw std::out_of_range("put: index out of range");
    }

    return output;
}

// ==============================================================================
// Stream-aware wrapper functions for existing operations
// ==============================================================================

auto gather_hip(
    const Tensor& input_arg,
    int64_t dim,
    const Tensor& indices_arg,
    hipStream_t stream
) -> Tensor {
    // Materialise contiguous copies — the kernel addresses input/index with
    // contiguous strides (matches the CPU reference).
    Tensor input = input_arg.is_contiguous() ? input_arg : input_arg.contiguous();
    Tensor indices = indices_arg.is_contiguous() ? indices_arg : indices_arg.contiguous();

    auto input_shape = input.shape();
    auto indices_shape = indices.shape();

    // Indices must be Int64 — the kernel reads them as int64_t (matches CPU).
    if (indices.dtype() != DType::Int64) {
        throw std::invalid_argument("gather: index tensor must have dtype Int64");
    }

    // Normalize negative dim (PyTorch semantics: dim=-1 → last)
    if (dim < 0) dim += static_cast<int64_t>(input_shape.size());
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::out_of_range("gather_hip: dim out of range");
    }

    if (indices_shape.size() != input_shape.size()) {
        throw std::invalid_argument("gather: index must have same rank as input");
    }
    int ndim = static_cast<int>(input_shape.size());
    if (ndim > MAX_GATHER_SCATTER_DIMS) {
        throw std::runtime_error("gather_hip: ndim exceeds MAX_GATHER_SCATTER_DIMS");
    }
    for (int d = 0; d < ndim; ++d) {
        if (d == dim) continue;
        if (indices_shape[d] > input_shape[d]) {
            throw std::out_of_range(
                "gather: index.size(" + std::to_string(d) + ") exceeds self.size(" +
                std::to_string(d) + ") (index extent must be <= input extent on "
                "non-gather dims)");
        }
    }

    // Output shape is exactly the index shape (PyTorch semantics).
    std::vector<int64_t> output_shape(indices_shape.begin(), indices_shape.end());

    Tensor output = Tensor(output_shape, input.dtype(), input.device());

    int64_t dim_size = input_shape[dim];
    int64_t total_output = output.numel();

    GatherScatterMeta meta{};
    {
        int64_t is = 1, ss = 1;
        for (int d = ndim - 1; d >= 0; --d) {
            meta.idx_strides[d] = is;
            meta.self_strides[d] = ss;
            is *= indices_shape[d];
            ss *= input_shape[d];
        }
    }

    // Match CPU semantics: throw on any out-of-range index instead of silently
    // skipping (kernel uses `continue`, leaving output uninitialized).
    validate_index_bounds(indices, dim_size, "gather");

    if (total_output == 0) {
        return output;
    }

    int threads = 256;
    int blocks = static_cast<int>((total_output + threads - 1) / threads);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(gather_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(),
            indices.data<int64_t>(),
            output.data<float>(),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(gather_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(),
            indices.data<int64_t>(),
            output.data<double>(),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(gather_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<int32_t>(),
            indices.data<int64_t>(),
            output.data<int32_t>(),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(gather_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<int64_t>(),
            indices.data<int64_t>(),
            output.data<int64_t>(),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(gather_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            indices.data<int64_t>(),
            reinterpret_cast<__half*>(output.data<Float16>()),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(gather_kernel<hip_bfloat16>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            indices.data<int64_t>(),
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Int16 || input.dtype() == DType::UInt16) {
        hipLaunchKernelGGL(gather_kernel<uint16_t>, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const uint16_t*>(input.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<uint16_t*>(output.data_ptr()),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::UInt32) {
        hipLaunchKernelGGL(gather_kernel<uint32_t>, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const uint32_t*>(input.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<uint32_t*>(output.data_ptr()),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::UInt64 || input.dtype() == DType::Complex64) {
        hipLaunchKernelGGL(gather_kernel<uint64_t>, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const uint64_t*>(input.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<uint64_t*>(output.data_ptr()),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Complex128) {
        hipLaunchKernelGGL(gather_kernel<Bytes16>, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const Bytes16*>(input.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<Bytes16*>(output.data_ptr()),
            ndim, static_cast<int>(dim), dim_size, total_output, meta);
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
    // 16-bit integer types have no HIP atomicAdd (required by the shared scatter
    // kernel template). Scatter is pure overwrite, so widen to Int32, scatter,
    // narrow back — the values round-trip exactly.
    if (input.dtype() == DType::Int16 || input.dtype() == DType::UInt16) {
        auto r = scatter_hip(input.to(DType::Int32), dim, indices, src.to(DType::Int32), stream);
        return r.to(input.dtype());
    }
    // Shape contract (PyTorch): index.size(d) <= src.size(d) for every axis, and
    // index.size(d) <= self.size(d) for d != dim. Reject an out-of-bounds index
    // rather than reading past the input / src on the device.
    {
        const int64_t nd = input.ndim();
        const int64_t d = dim < 0 ? dim + nd : dim;
        if (indices.ndim() != nd) {
            throw std::invalid_argument("scatter: index rank must match self rank");
        }
        for (int64_t k = 0; k < nd; ++k) {
            if (k < src.ndim() && indices.shape()[k] > src.shape()[k]) {
                throw std::invalid_argument("scatter: index size exceeds src on axis " + std::to_string(k));
            }
            if (k != d && indices.shape()[k] > input.shape()[k]) {
                throw std::invalid_argument("scatter: index size exceeds input on non-scatter axis " + std::to_string(k));
            }
        }
    }
    // input.clone() yields a contiguous destination. The kernel decodes
    // index/src positions linearly, so both must be contiguous too.
    Tensor output = input.clone();
    auto output_shape = output.shape();
    Tensor indices_c = indices.is_contiguous() ? indices : indices.contiguous();
    Tensor src_c = src.is_contiguous() ? src : src.contiguous();
    auto indices_shape = indices_c.shape();

    // Indices must be Int64 — the kernel reads them as int64_t (matches CPU).
    if (indices_c.dtype() != DType::Int64) {
        throw std::invalid_argument("scatter: index tensor must have dtype Int64");
    }

    // Normalize dimension
    int64_t ndim = output.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("scatter_hip: dim out of range");
    }

    if (static_cast<int64_t>(indices_shape.size()) != ndim) {
        throw std::invalid_argument("scatter: index must have same rank as self");
    }
    int ndim_i = static_cast<int>(ndim);
    if (ndim_i > MAX_GATHER_SCATTER_DIMS) {
        throw std::runtime_error("scatter_hip: ndim exceeds MAX_GATHER_SCATTER_DIMS");
    }
    for (int d = 0; d < ndim_i; ++d) {
        if (d == dim) continue;
        if (indices_shape[d] > output_shape[d]) {
            throw std::out_of_range(
                "scatter: index.size(" + std::to_string(d) + ") exceeds self.size(" +
                std::to_string(d) + ") (index extent must be <= self extent on "
                "non-scatter dims)");
        }
    }

    int64_t dim_size = output_shape[dim];
    int64_t total_scatter = indices_c.numel();

    GatherScatterMeta meta{};
    {
        int64_t is = 1, ss = 1;
        for (int d = ndim_i - 1; d >= 0; --d) {
            meta.idx_strides[d] = is;
            meta.self_strides[d] = ss;
            is *= indices_shape[d];
            ss *= output_shape[d];
        }
    }

    // Match CPU semantics: throw on any out-of-range index instead of silently
    // dropping the write.
    validate_index_bounds(indices_c, dim_size, "scatter");

    if (total_scatter == 0) {
        return output;
    }

    int threads = 256;
    int blocks = (total_scatter + threads - 1) / threads;

    if (output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(scatter_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<float>(),
            indices_c.data<int64_t>(),
            src_c.data<float>(),
            ndim_i,
            static_cast<int>(dim),
            dim_size,
            total_scatter,
            meta,
            false
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(scatter_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<double>(),
            indices_c.data<int64_t>(),
            src_c.data<double>(),
            ndim_i,
            static_cast<int>(dim),
            dim_size,
            total_scatter,
            meta,
            false
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Float16) {
        hipLaunchKernelGGL(scatter_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<__half*>(output.data<Float16>()),
            indices_c.data<int64_t>(),
            reinterpret_cast<const __half*>(src_c.data<Float16>()),
            ndim_i,
            static_cast<int>(dim),
            dim_size,
            total_scatter,
            meta,
            false
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(scatter_kernel<hip_bfloat16>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            indices_c.data<int64_t>(),
            reinterpret_cast<const hip_bfloat16*>(src_c.data<BFloat16>()),
            ndim_i,
            static_cast<int>(dim),
            dim_size,
            total_scatter,
            meta,
            false
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Int32) {
        hipLaunchKernelGGL(scatter_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<int32_t>(),
            indices_c.data<int64_t>(),
            src_c.data<int32_t>(),
            ndim_i,
            static_cast<int>(dim),
            dim_size,
            total_scatter,
            meta,
            false
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Int64) {
        hipLaunchKernelGGL(scatter_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            output.data<int64_t>(),
            indices_c.data<int64_t>(),
            src_c.data<int64_t>(),
            ndim_i,
            static_cast<int>(dim),
            dim_size,
            total_scatter,
            meta,
            false
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::UInt32) {
        hipLaunchKernelGGL(scatter_kernel<uint32_t>, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<uint32_t*>(output.data_ptr()), indices_c.data<int64_t>(),
            reinterpret_cast<const uint32_t*>(src_c.data_ptr()),
            ndim_i, static_cast<int>(dim), dim_size, total_scatter, meta, false);
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::UInt64 || output.dtype() == DType::Complex64) {
        hipLaunchKernelGGL(scatter_kernel<uint64_t>, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<uint64_t*>(output.data_ptr()), indices_c.data<int64_t>(),
            reinterpret_cast<const uint64_t*>(src_c.data_ptr()),
            ndim_i, static_cast<int>(dim), dim_size, total_scatter, meta, false);
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Complex128) {
        hipLaunchKernelGGL(scatter_kernel<Bytes16>, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<Bytes16*>(output.data_ptr()), indices_c.data<int64_t>(),
            reinterpret_cast<const Bytes16*>(src_c.data_ptr()),
            ndim_i, static_cast<int>(dim), dim_size, total_scatter, meta, false);
        HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("scatter_hip: Unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return output;
}

auto index_select_hip(
    const Tensor& input_orig,
    int64_t dim,
    const Tensor& indices_orig,
    hipStream_t stream
) -> Tensor {
    // Contiguify: the kernel reads input/indices with flat addressing, so a
    // non-contiguous input or a strided index view would read the wrong storage.
    const Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
    const Tensor indices = indices_orig.is_contiguous() ? indices_orig : indices_orig.contiguous();
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

    // Match CPU semantics: throw on any out-of-range index instead of silently
    // skipping (kernel writes nothing for OOB, leaving output uninitialized).
    validate_index_bounds(indices, dim_size, "index_select");

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
    } else if (input.dtype() == DType::Int16 || input.dtype() == DType::UInt16) {
        hipLaunchKernelGGL(index_select_kernel<uint16_t>, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const uint16_t*>(input.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<uint16_t*>(output.data_ptr()), outer_size, dim_size, inner_size, num_indices);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::UInt32) {
        hipLaunchKernelGGL(index_select_kernel<uint32_t>, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const uint32_t*>(input.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<uint32_t*>(output.data_ptr()), outer_size, dim_size, inner_size, num_indices);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::UInt64 || input.dtype() == DType::Complex64) {
        hipLaunchKernelGGL(index_select_kernel<uint64_t>, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const uint64_t*>(input.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<uint64_t*>(output.data_ptr()), outer_size, dim_size, inner_size, num_indices);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Complex128) {
        hipLaunchKernelGGL(index_select_kernel<Bytes16>, dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const Bytes16*>(input.data_ptr()), indices.data<int64_t>(),
            reinterpret_cast<Bytes16*>(output.data_ptr()), outer_size, dim_size, inner_size, num_indices);
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
    // Empty tensor: a zero-block grid makes HIP reject the launch
    // ("invalid configuration argument"), unlike CPU/CUDA/OneAPI/Vulkan which
    // all correctly return an empty tensor (matches where_hip's own guard).
    if (total_elements == 0) return output;
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
    } else if (output.dtype() == DType::Complex64) {
        // Fill with (value, 0): pack real float bits low, imag 0 high (8 bytes).
        float fr = static_cast<float>(value);
        uint32_t rbits; std::memcpy(&rbits, &fr, sizeof(float));
        uint64_t packed = static_cast<uint64_t>(rbits);
        hipLaunchKernelGGL(masked_fill_kernel<uint64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<uint64_t*>(output.data_ptr()),
            mask.data<bool>(), packed, total_elements);
        HIP_POST_LAUNCH_CHECK();
    } else if (output.dtype() == DType::Complex128) {
        // Fill with (value, 0): pack real double bits low, imag 0.0 high (16 bytes),
        // mirroring the Complex64 packing above.
        double dr = static_cast<double>(value);
        uint64_t rbits; std::memcpy(&rbits, &dr, sizeof(double));
        Bytes16 packed{rbits, 0ull};
        hipLaunchKernelGGL(masked_fill_kernel<Bytes16>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<Bytes16*>(output.data_ptr()),
            mask.data<bool>(), packed, total_elements);
        HIP_POST_LAUNCH_CHECK();
    } else {
        throw std::runtime_error("masked_fill_hip: Unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return output;
}

// Order-preserving stream compaction via hipcub DeviceSelect::Flagged. Unlike
// the atomic-counter compaction this replaced, Flagged writes selected elements
// in their original flattened (row-major) order, matching CPU and CUDA so
// positional consumers see identical results across backends.
template<typename T>
static void masked_select_flagged(const T* d_in, const bool* d_flags, T* d_out,
                                  int64_t* d_num_selected, int64_t total_elements,
                                  hipStream_t stream) {
    // hipcub::DeviceSelect::Flagged (rocprim backend, __HIP_PLATFORM_AMD__) takes
    // num_items as int64_t natively — unlike CUB's overload-set-by-NumItemsT
    // design, there is no 32-bit-limited overload to fall into here, so no cast
    // or size guard is needed; pass total_elements straight through (mirrors
    // CUDA's indexing.cu masked_select_kernel, which selects CUB's 64-bit
    // NumItemsT overload the same way).
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    HIP_CHECK(hipcub::DeviceSelect::Flagged(
        d_temp, temp_bytes, d_in, d_flags, d_out, d_num_selected,
        total_elements, stream));
    HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
    HIP_CHECK(hipcub::DeviceSelect::Flagged(
        d_temp, temp_bytes, d_in, d_flags, d_out, d_num_selected,
        total_elements, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_temp));
}

auto masked_select_hip(
    const Tensor& input,
    const Tensor& mask,
    hipStream_t stream
) -> Tensor {
    int64_t total_elements = input.numel();
    // Empty tensor: a zero-block grid makes HIP reject the launch
    // ("invalid configuration argument"), unlike CPU/CUDA/OneAPI/Vulkan which
    // all correctly return an empty tensor (matches where_hip's own guard).
    if (total_elements == 0) return Tensor({0}, input.dtype(), input.device());

    int64_t* d_count;
    HIP_CHECK(hipMalloc(&d_count, sizeof(int64_t)));
    HIP_CHECK(hipMemsetAsync(d_count, 0, sizeof(int64_t), stream));

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    // The count pass only inspects the boolean mask, so the element type is
    // irrelevant — a single instantiation covers every dtype.
    hipLaunchKernelGGL(masked_select_count_kernel<unsigned char>,
        dim3(blocks), dim3(threads), 0, stream,
        mask.data<bool>(), d_count, total_elements);
    HIP_POST_LAUNCH_CHECK();

    HIP_CHECK(hipStreamSynchronize(stream));

    int64_t h_count;
    HIP_CHECK(hipMemcpy(&h_count, d_count, sizeof(int64_t), hipMemcpyDeviceToHost));

    Tensor output = Tensor({h_count}, input.dtype(), input.device());

    // Order-preserving compaction. DeviceSelect::Flagged also reports the number
    // selected; we already know it (h_count) but the API requires the output, so
    // reuse d_count as the d_num_selected scratch.
    // The copy reproduces raw element bytes, so dispatch by element width: any
    // same-width integer payload covers float, integer and complex dtypes
    // (Complex64 = 8 bytes, Complex128 = 16 bytes).
    const void* in_ptr = input.data_ptr();
    void* out_ptr = const_cast<void*>(output.data_ptr());
    const bool* mask_ptr = mask.data<bool>();
    size_t width = dtype_size(input.dtype());
    if (width == 1) {
        masked_select_flagged(reinterpret_cast<const uint8_t*>(in_ptr), mask_ptr,
            reinterpret_cast<uint8_t*>(out_ptr), d_count, total_elements, stream);
    } else if (width == 2) {
        masked_select_flagged(reinterpret_cast<const uint16_t*>(in_ptr), mask_ptr,
            reinterpret_cast<uint16_t*>(out_ptr), d_count, total_elements, stream);
    } else if (width == 4) {
        masked_select_flagged(reinterpret_cast<const uint32_t*>(in_ptr), mask_ptr,
            reinterpret_cast<uint32_t*>(out_ptr), d_count, total_elements, stream);
    } else if (width == 8) {
        masked_select_flagged(reinterpret_cast<const uint64_t*>(in_ptr), mask_ptr,
            reinterpret_cast<uint64_t*>(out_ptr), d_count, total_elements, stream);
    } else if (width == 16) {
        masked_select_flagged(reinterpret_cast<const Bytes16*>(in_ptr), mask_ptr,
            reinterpret_cast<Bytes16*>(out_ptr), d_count, total_elements, stream);
    } else {
        HIP_CHECK(hipFree(d_count));
        throw std::runtime_error("masked_select_hip: unsupported element width");
    }

    HIP_CHECK(hipFree(d_count));
    HIP_POST_LAUNCH_CHECK();

    return output;
}

auto take_hip(
    const Tensor& input_arg,
    const Tensor& indices_arg,
    hipStream_t stream
) -> Tensor {
    // F085: take reads input by flat storage index, so a non-contiguous
    // (transposed/sliced) view must be materialised first (matches CPU/CUDA);
    // likewise the index tensor is read linearly.
    Tensor input   = input_arg.is_contiguous()   ? input_arg   : input_arg.contiguous();
    Tensor indices = indices_arg.is_contiguous() ? indices_arg : indices_arg.contiguous();

    // F085: Float16/BFloat16 — widen to Float32, take, narrow back (take is a
    // pure value copy; CPU/CUDA support half, ROCm previously else-threw).
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor input_f32 = input.to(DType::Float32);
        Tensor out_f32 = take_hip(input_f32, indices, stream);
        return out_f32.to(orig);
    }

    int64_t input_size = input.numel();
    int64_t indices_size = indices.numel();

    // take() output must preserve the index tensor's shape (matches CPU/CUDA/
    // OneAPI/Vulkan / PyTorch semantics), not flatten to 1D. A 2D index of
    // shape [2,3] must produce a [2,3] output, not [6].
    std::vector<int64_t> out_shape(indices_arg.shape().begin(), indices_arg.shape().end());
    Tensor output = Tensor(out_shape, input.dtype(), input.device());

    int threads = 256;
    int blocks = (indices_size + threads - 1) / threads;

    // F085: device-side OOB error flag so an out-of-range index throws (matches
    // CPU/CUDA) instead of silently returning 0.
    int* d_error_flag = nullptr;
    HIP_CHECK(hipMalloc(&d_error_flag, sizeof(int)));
    HIP_CHECK(hipMemsetAsync(d_error_flag, 0, sizeof(int), stream));

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(take_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(),
            indices.data<int64_t>(),
            output.data<float>(),
            input_size,
            indices_size,
            d_error_flag
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(take_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(),
            indices.data<int64_t>(),
            output.data<double>(),
            input_size,
            indices_size,
            d_error_flag
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(take_kernel<int32_t>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<int32_t>(),
            indices.data<int64_t>(),
            output.data<int32_t>(),
            input_size,
            indices_size,
            d_error_flag
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(take_kernel<int64_t>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<int64_t>(),
            indices.data<int64_t>(),
            output.data<int64_t>(),
            input_size,
            indices_size,
            d_error_flag
        );
        HIP_POST_LAUNCH_CHECK();
    } else {
        HIP_CHECK(hipFree(d_error_flag));
        throw std::runtime_error("take_hip: Unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();

    // Check for out-of-bounds index errors (matches CPU/CUDA which throw).
    int host_error = 0;
    HIP_CHECK(hipMemcpyAsync(&host_error, d_error_flag, sizeof(int),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_error_flag));
    if (host_error != 0) {
        throw std::out_of_range("take: index out of range");
    }

    return output;
}

// ==============================================================================
// Gather Relative Position Bias (for Swin Transformer)
// ==============================================================================

template<typename T>
__global__ void gather_2d_kernel(
    const T* table, const int64_t* indices, T* output,
    int64_t num_positions, int64_t num_heads, int64_t table_stride,
    int64_t num_table_rows) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_positions * num_positions * num_heads;

    if (idx >= total) return;

    int64_t h = idx % num_heads;
    int64_t j = (idx / num_heads) % num_positions;
    int64_t i = idx / (num_heads * num_positions);

    int64_t table_idx = indices[i * num_positions + j];
    // Bounds-check the table row index from a (possibly untrusted) index tensor
    // to avoid an out-of-bounds device read; out-of-range entries gather 0.
    if (table_idx < 0 || table_idx >= num_table_rows) {
        output[idx] = T(0);
        return;
    }
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

    // Number of rows in the table for bounds-checking the gathered index.
    int64_t num_table_rows = table.shape().empty() ? 0 : table.shape()[0];

    if (table.dtype() == DType::Float32) {
        hipLaunchKernelGGL(gather_2d_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            table.data<float>(), indices_device.data<int64_t>(), output.data<float>(),
            num_positions, num_heads, num_heads, num_table_rows);
            HIP_POST_LAUNCH_CHECK();
    } else if (table.dtype() == DType::Float64) {
        hipLaunchKernelGGL(gather_2d_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            table.data<double>(), indices_device.data<int64_t>(), output.data<double>(),
            num_positions, num_heads, num_heads, num_table_rows);
            HIP_POST_LAUNCH_CHECK();
    } else if (table.dtype() == DType::Float16) {
        hipLaunchKernelGGL(gather_2d_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(table.data<Float16>()),
            indices_device.data<int64_t>(),
            reinterpret_cast<__half*>(output.data<Float16>()),
            num_positions, num_heads, num_heads, num_table_rows);
            HIP_POST_LAUNCH_CHECK();
    } else if (table.dtype() == DType::BFloat16) {
        // BFloat16: widen the table to Float32, gather, and narrow back —
        // mirrors the CPU widen (vision.cpp). gather is a pure value copy so the
        // widen only exists to give this dtype a supported path (CUDA/ROCm
        // previously else-threw here).
        Tensor table_f32 = table.to(DType::Float32);
        Tensor out_f32({num_positions, num_positions, num_heads},
                       DType::Float32, table.device());
        hipLaunchKernelGGL(gather_2d_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            table_f32.data<float>(), indices_device.data<int64_t>(), out_f32.data<float>(),
            num_positions, num_heads, num_heads, num_table_rows);
            HIP_POST_LAUNCH_CHECK();
        output = out_f32.to(DType::BFloat16);
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
    int ndim,
    int dim,
    int64_t dim_size,
    int64_t total_scatter,
    GatherScatterMeta meta,
    int* error_flag) {

    HIP_KERNEL_LOOP(idx, total_scatter) {
        // F076: decode the flat index/src position per-dimension from the index
        // strides, then re-linearise against the output's contiguous strides,
        // substituting the scattered coordinate on the `dim` axis. The previous
        // collapsed outer/inner formula mis-addressed whenever index.size(d) <
        // self.size(d) on a non-dim axis (the gather-backward case in 3D+).
        int64_t scatter_idx = static_cast<int64_t>(indices[idx]);
        if (scatter_idx < 0) scatter_idx += dim_size;
        if (scatter_idx < 0 || scatter_idx >= dim_size) {
            atomicExch(error_flag, 1);
            return;
        }

        int64_t rem = idx;
        int64_t output_offset = 0;
        for (int d = 0; d < ndim; ++d) {
            int64_t coord = rem / meta.idx_strides[d];
            rem %= meta.idx_strides[d];
            if (d == dim) {
                output_offset += scatter_idx * meta.self_strides[d];
            } else {
                output_offset += coord * meta.self_strides[d];
            }
        }

        atomicAddHelper(&output[output_offset], src[idx]);
    }
}

auto scatter_add_kernel(const Tensor& input_arg, int64_t dim, const Tensor& index_arg,
                        const Tensor& src_arg, hipStream_t stream) -> Tensor {
    // F079: the kernel addresses input/index/src with contiguous strides, so a
    // transposed/sliced view must be materialised first (matches CPU/CUDA).
    Tensor input = input_arg.is_contiguous() ? input_arg : input_arg.contiguous();
    Tensor index = index_arg.is_contiguous() ? index_arg : index_arg.contiguous();
    Tensor src   = src_arg.is_contiguous()   ? src_arg   : src_arg.contiguous();

    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("scatter_add: dimension out of range");
    }

    // Shape contract (PyTorch): index.size(d) <= src.size(d) for every axis, and
    // index.size(d) <= self.size(d) for d != dim. Reject an out-of-bounds index
    // rather than reading past the input / src on the device.
    if (index.ndim() != ndim) {
        throw std::invalid_argument("scatter_add: index rank must match self rank");
    }
    if (ndim > MAX_GATHER_SCATTER_DIMS) {
        throw std::runtime_error("scatter_add: ndim exceeds MAX_GATHER_SCATTER_DIMS");
    }
    for (int64_t k = 0; k < ndim; ++k) {
        if (k < src.ndim() && index.shape()[k] > src.shape()[k]) {
            throw std::invalid_argument("scatter_add: index size exceeds src on axis " + std::to_string(k));
        }
        if (k != dim && index.shape()[k] > input.shape()[k]) {
            throw std::invalid_argument("scatter_add: index size exceeds input on non-scatter axis " + std::to_string(k));
        }
    }

    std::vector<int64_t> output_shape(input.shape().begin(), input.shape().end());
    Tensor output(output_shape, input.dtype(), input.device());

    int64_t total_input = input.numel();
    int64_t total_scatter = index.numel();

    if (total_input == 0) return output;

    int64_t dim_size = input.shape()[dim];
    int ndim_i = static_cast<int>(ndim);
    int dim_i = static_cast<int>(dim);

    // F076: per-dim metadata — index strides (to decode each flat scatter
    // position) and the OUTPUT's contiguous strides (to re-linearise every
    // non-dim coordinate and the substituted scatter index on the `dim` axis).
    GatherScatterMeta meta{};
    {
        int64_t is = 1, ss = 1;
        for (int d = ndim_i - 1; d >= 0; --d) {
            meta.idx_strides[d] = is;
            meta.self_strides[d] = ss;
            is *= index.shape()[d];
            ss *= input.shape()[d];
        }
    }

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
                ndim_i, dim_i, dim_size, total_scatter, meta, \
                d_error_flag); \
        else \
            hipLaunchKernelGGL((scatter_add_kernel_impl<T, int64_t>), \
                dim3(blocks), dim3(threads), 0, stream, \
                index.data<int64_t>(), src.data<T>(), output.data<T>(), \
                ndim_i, dim_i, dim_size, total_scatter, meta, \
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
                        ndim_i, dim_i, dim_size, total_scatter, meta,
                        d_error_flag);
                    HIP_POST_LAUNCH_CHECK();
                } else {
                    hipLaunchKernelGGL((scatter_add_kernel_impl<float, int64_t>),
                        dim3(blocks_f32), dim3(threads), 0, stream,
                        index.data<int64_t>(), src_f32.data<float>(), output_f32.data<float>(),
                        ndim_i, dim_i, dim_size, total_scatter, meta,
                        d_error_flag);
                    HIP_POST_LAUNCH_CHECK();
                }
            }
            output = output_f32.to(input.dtype());
            break;
        }
        case DType::UInt32: LAUNCH_SCATTER_ADD_HIP(uint32_t); break;
        case DType::UInt64: LAUNCH_SCATTER_ADD_HIP(uint64_t); break;
        case DType::Int16:
        case DType::UInt16:
        case DType::Int8:
        case DType::UInt8: {
            // No 16/8-bit atomicAdd: accumulate in Int32, then narrow back.
            HIP_CHECK(hipFree(d_error_flag));
            auto r = scatter_add_kernel(input.to(DType::Int32), dim, index,
                                        src.to(DType::Int32), stream);
            return r.to(input.dtype());
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

// Float16 needs the same F32-accumulator treatment as BFloat16: accumulating
// an entire bag sum in fp16 (T acc = T(0); acc += ...) loses precision and
// drifts for large bags / many summands. Load as __half, accumulate/divide in
// float, store via safe_f2h (NaN-preserving, RNE narrowing).
__global__ void embedding_bag_sum_kernel_hip_f16(
    const __half* embeddings,
    const int64_t* offsets,
    __half* output,
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
            acc += tenzor::rocm::safe_h2f(embeddings[i * embedding_dim + j]);
        }
        if (divide_by_count && bag_size > 0) {
            acc /= static_cast<float>(bag_size);
        }
        output[bag * embedding_dim + j] = tenzor::rocm::safe_f2h(acc);
    }
}

__global__ void embedding_bag_max_kernel_hip_bf16(
    const hip_bfloat16* embeddings,
    const int64_t* offsets,
    hip_bfloat16* output,
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
        float max_val = static_cast<float>(embeddings[start * embedding_dim + j]);
        int64_t arg = start;
        for (int64_t i = start + 1; i < end; ++i) {
            float val = static_cast<float>(embeddings[i * embedding_dim + j]);
            if (val > max_val) { max_val = val; arg = i; }  // strict '>': first wins
        }
        // S.10: max over a bag is exact in float32 but the back-cast still
        // needs RNE rounding to avoid the truncating-ctor bias documented
        // in R.11.
        output[bag * embedding_dim + j] = tenzor::rocm::f32_to_bf16_rne(max_val);
        if (max_indices != nullptr) max_indices[bag * embedding_dim + j] = arg;
    }
}

auto embedding_bag_forward_kernel(const Tensor& embeddings, const Tensor& offsets,
                                   const std::string& mode, int64_t embedding_dim,
                                   bool include_last_offset, hipStream_t stream) -> std::vector<Tensor> {
    int64_t total_elements = embeddings.shape()[0];
    int64_t offsets_size = offsets.numel();
    int64_t num_bags = include_last_offset ? (offsets_size - 1) : offsets_size;

    // Pre-validate offsets host-side before any kernel dispatch. Per-bag
    // [start, end) bounds are derived from offsets and indexed directly into
    // the embeddings buffer with no on-device bounds checking; a malformed
    // offset (negative, non-monotonic, or exceeding the embedding table
    // length) would cause an out-of-bounds device read (and an OOB write of
    // the argmax buffer in max mode). Mirrors the CPU reference
    // (nn_kernels.cpp) and OneAPI's embedding_bag_forward_kernel.
    if (num_bags > 0) {
        Tensor offsets_host = offsets.to(Device::cpu());
        const int64_t* host_offsets = offsets_host.data<int64_t>();
        int64_t prev = 0;
        for (int64_t bag = 0; bag < num_bags; ++bag) {
            int64_t start = host_offsets[bag];
            int64_t end = (bag + 1 < offsets_size) ? host_offsets[bag + 1] : total_elements;
            if (start < prev || start > total_elements ||
                end < start || end > total_elements) {
                throw std::out_of_range(
                    "embedding_bag_forward: offset out of range or non-monotonic "
                    "at bag " + std::to_string(bag) + " ([" +
                    std::to_string(start) + ", " + std::to_string(end) +
                    ") not within [0, " + std::to_string(total_elements) + "])");
            }
            prev = start;
        }
    }

    bool is_mean = (mode == "mean");
    bool is_max = (mode == "max");

    if (num_bags <= 0) {
        return {Tensor({0, embedding_dim}, embeddings.dtype(), embeddings.device()),
                tenzor::zeros({0}, DType::Int64, embeddings.device())};
    }

    // Create zero-initialized output
    Tensor output({num_bags, embedding_dim}, embeddings.dtype(), embeddings.device());
    HIP_CHECK(hipMemsetAsync(output.data_ptr(), 0,
                             num_bags * embedding_dim * dtype_size(embeddings.dtype()), stream));

    // For max mode, also emit the per-(bag,feature) GLOBAL argmax element index
    // (-1 for empty bags), so the autograd node routes the gradient exactly
    // on-device. Empty/unused otherwise.
    Tensor max_indices = is_max
        ? tenzor::full({num_bags, embedding_dim}, static_cast<double>(-1),
                       DType::Int64, embeddings.device())
        : tenzor::zeros({0}, DType::Int64, embeddings.device());
    int64_t* argmax_ptr = is_max ? max_indices.data<int64_t>() : nullptr;

    // Clamp to >= 1: a dim3 with a zero extent (embedding_dim == 0 or
    // num_bags == 0) is an invalid kernel launch. With >= 1 the kernel's bounds
    // checks make it a no-op and the pre-initialized outputs are returned.
    int threads = std::max(1, std::min(static_cast<int>(embedding_dim), 256));
    int blocks = std::max(1, static_cast<int>(num_bags));

    switch (embeddings.dtype()) {
        case DType::Float32:
            if (is_max) {
                hipLaunchKernelGGL(embedding_bag_max_kernel_hip<float>,
                    dim3(blocks), dim3(threads), 0, stream,
                    embeddings.data<float>(), offsets.data<int64_t>(),
                    output.data<float>(), argmax_ptr, num_bags, total_elements,
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
                    output.data<double>(), argmax_ptr, num_bags, total_elements,
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
                    argmax_ptr, num_bags, total_elements, embedding_dim, offsets_size);
                HIP_POST_LAUNCH_CHECK();
            } else {
                // F32-accumulator kernel: fp16 bag-sum accumulation drifts.
                hipLaunchKernelGGL(embedding_bag_sum_kernel_hip_f16,
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
                    argmax_ptr, num_bags, total_elements, embedding_dim, offsets_size);
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
    return {output, max_indices};
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

// F071 Max-mode backward: the forward emits, per (bag, feature), the GLOBAL
// element index into `indices` that achieved the maximum (or -1 for an empty
// bag). The gradient flows only to that winning row, mirroring CPU/CUDA. The
// previous code had no max branch, so max mode fell through to the sum kernel
// and scatter-added the full gradient into EVERY row of each bag — wrong grad.
template<typename T>
__global__ void embedding_bag_backward_max_kernel_hip(
    const T* grad_output,
    const int64_t* max_indices,   // [num_bags, embedding_dim]
    const int64_t* indices,       // [total_elements]
    T* grad_weight,
    int64_t num_bags,
    int64_t embedding_dim,
    int64_t total_elements,
    int64_t num_embeddings)
{
    int64_t bag = blockIdx.x;
    if (bag >= num_bags) return;
    for (int64_t j = threadIdx.x; j < embedding_dim; j += blockDim.x) {
        int64_t elem = max_indices[bag * embedding_dim + j];
        if (elem < 0 || elem >= total_elements) continue;  // empty bag / OOB
        int64_t row = indices[elem];
        if (row < 0 || row >= num_embeddings) continue;
        atomicAdd(&grad_weight[row * embedding_dim + j],
                  grad_output[bag * embedding_dim + j]);
    }
}

auto embedding_bag_backward_kernel(const Tensor& grad_output,
                                   const Tensor& indices,
                                   const Tensor& offsets,
                                   const Tensor& max_indices,
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
        auto result = embedding_bag_backward_kernel(go_f32, indices, offsets, max_indices, attrs, stream);
        return result.to(grad_output.dtype());
    }

    Tensor grad_weight({num_embeddings, embedding_dim}, grad_output.dtype(), grad_output.device());
    HIP_CHECK(hipMemsetAsync(grad_weight.data_ptr(), 0,
                             num_embeddings * embedding_dim * dtype_size(grad_output.dtype()), stream));

    // Clamp to >= 1: a dim3 with a zero extent (embedding_dim == 0 or
    // num_bags == 0) is an invalid kernel launch. With >= 1 the kernel's bounds
    // checks make it a no-op and the pre-initialized outputs are returned.
    int threads = std::max(1, std::min(static_cast<int>(embedding_dim), 256));
    int blocks = std::max(1, static_cast<int>(num_bags));

    // F071 Max mode: route the gradient only to the per-feature argmax row using
    // the max_indices saved by the forward (matches CPU/CUDA).
    if (mode == "max") {
        if (!max_indices.is_valid() || max_indices.numel() == 0) {
            throw std::runtime_error(
                "embedding_bag_backward: mode=\"max\" requires the per-feature "
                "argmax indices (4th input) produced by EmbeddingBagForward.");
        }
        if (max_indices.dtype() != DType::Int64) {
            throw std::runtime_error("embedding_bag_backward: max_indices must be Int64");
        }
        switch (grad_output.dtype()) {
            case DType::Float32:
                hipLaunchKernelGGL(embedding_bag_backward_max_kernel_hip<float>,
                    dim3(blocks), dim3(threads), 0, stream,
                    grad_output.data<float>(), max_indices.data<int64_t>(), indices.data<int64_t>(),
                    grad_weight.data<float>(), num_bags, embedding_dim, total_elements, num_embeddings);
                HIP_POST_LAUNCH_CHECK();
                break;
            case DType::Float64:
                hipLaunchKernelGGL(embedding_bag_backward_max_kernel_hip<double>,
                    dim3(blocks), dim3(threads), 0, stream,
                    grad_output.data<double>(), max_indices.data<int64_t>(), indices.data<int64_t>(),
                    grad_weight.data<double>(), num_bags, embedding_dim, total_elements, num_embeddings);
                HIP_POST_LAUNCH_CHECK();
                break;
            default:
                throw std::runtime_error("embedding_bag_backward: unsupported dtype");
        }
        HIP_POST_LAUNCH_CHECK();
        return grad_weight;
    }

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
    int64_t num_classes,
    int* error_flag) {

    int64_t total = batch_size * num_classes;
    HIP_KERNEL_LOOP(idx, total) {
        int64_t batch = idx / num_classes;
        int64_t cls = idx % num_classes;
        int64_t label = static_cast<int64_t>(indices[batch]);
        // PyTorch raises on out-of-range / negative labels rather than silently
        // emitting an all-zero (non-one-hot) row. Flag it for the host to throw.
        if (label < 0 || label >= num_classes) {
            atomicExch(error_flag, 1);
        }
        output[idx] = (label == cls) ? 1.0f : 0.0f;
    }
}

auto one_hot_kernel(const Tensor& indices, int64_t num_classes,
                    hipStream_t stream) -> Tensor {
    int64_t batch_size = indices.numel();

    // If num_classes is not specified, infer it from the max index value + 1
    // (mirrors CPU's one_hot_kernel in src/backends/cpu/kernels/indexing.cpp).
    // Not reachable via the public API today (num_classes is always resolved
    // before dispatch), but kept for defense-in-depth cross-backend parity.
    if (num_classes <= 0) {
        int64_t inferred = 0;
        if (indices.dtype() == DType::Int64) {
            std::vector<int64_t> host(static_cast<size_t>(batch_size));
            if (batch_size > 0) {
                HIP_CHECK(hipMemcpy(host.data(), indices.data<int64_t>(),
                                    static_cast<size_t>(batch_size) * sizeof(int64_t),
                                    hipMemcpyDeviceToHost));
            }
            for (int64_t v : host) {
                if (v < 0) throw std::out_of_range("one_hot: indices must be non-negative");
                if (v + 1 > inferred) inferred = v + 1;
            }
        } else if (indices.dtype() == DType::Int32) {
            std::vector<int32_t> host(static_cast<size_t>(batch_size));
            if (batch_size > 0) {
                HIP_CHECK(hipMemcpy(host.data(), indices.data<int32_t>(),
                                    static_cast<size_t>(batch_size) * sizeof(int32_t),
                                    hipMemcpyDeviceToHost));
            }
            for (int32_t v : host) {
                if (v < 0) throw std::out_of_range("one_hot: indices must be non-negative");
                int64_t v64 = static_cast<int64_t>(v) + 1;
                if (v64 > inferred) inferred = v64;
            }
        } else {
            throw std::runtime_error("one_hot: unsupported index dtype (expected Int32 or Int64)");
        }
        // An empty index tensor or all-zero indices can leave inferred == 0;
        // a one-hot dimension must be positive, so reject before allocating.
        if (inferred <= 0) {
            throw std::invalid_argument(
                "one_hot: could not infer a positive num_classes from indices; "
                "pass num_classes explicitly");
        }
        num_classes = inferred;
    }

    // one_hot appends the class axis while preserving the index shape:
    // [d0, ..., dk] -> [d0, ..., dk, num_classes] (matching PyTorch and the
    // CPU/CUDA/oneAPI kernels). The flat kernel fill below is unchanged since
    // the buffer layout [flat_index, class] is identical.
    std::vector<int64_t> out_shape(indices.shape().begin(), indices.shape().end());
    out_shape.push_back(num_classes);
    Tensor output(out_shape, DType::Float32, indices.device());

    if (batch_size == 0) return output;

    int64_t total = batch_size * num_classes;
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int* d_error_flag = nullptr;
    HIP_CHECK(hipMalloc(&d_error_flag, sizeof(int)));
    HIP_CHECK(hipMemsetAsync(d_error_flag, 0, sizeof(int), stream));

    switch (indices.dtype()) {
        case DType::Int32:
            hipLaunchKernelGGL(one_hot_kernel_impl<int32_t>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                indices.data<int32_t>(), output.data<float>(), batch_size, num_classes,
                d_error_flag);
            HIP_POST_LAUNCH_CHECK();
            break;
        case DType::Int64:
            hipLaunchKernelGGL(one_hot_kernel_impl<int64_t>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                indices.data<int64_t>(), output.data<float>(), batch_size, num_classes,
                d_error_flag);
            HIP_POST_LAUNCH_CHECK();
            break;
        default:
            HIP_CHECK(hipFree(d_error_flag));
            throw std::runtime_error("one_hot: unsupported index dtype (expected Int32 or Int64)");
    }

    HIP_POST_LAUNCH_CHECK();

    // Surface out-of-range / negative labels as an error (PyTorch parity).
    int host_error = 0;
    HIP_CHECK(hipMemcpyAsync(&host_error, d_error_flag, sizeof(int),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_error_flag));

    if (host_error) {
        throw std::out_of_range(
            "one_hot: index out of range [0, " + std::to_string(num_classes) + ")");
    }

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

    // hipcub::DeviceSelect::Flagged takes a 32-bit num_items; silently casting an
    // int64 element count > INT_MAX would truncate/overflow and compact the wrong
    // number of indices. Reject it cleanly instead of returning garbage.
    if (n > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "nonzero: element count " + std::to_string(n) +
            " exceeds hipcub DeviceSelect 32-bit limit");
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

    // D2H sync to get count. hipcub writes a 32-bit count; widen to int64 for the
    // subsequent output sizing (which is int64). The count cannot exceed n, and n
    // is guaranteed <= INT_MAX by the guard above, so no truncation occurs.
    int total_nonzero_i32 = 0;
    HIP_CHECK(hipMemcpyAsync(&total_nonzero_i32, d_num_selected, sizeof(int), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    int64_t total_nonzero = static_cast<int64_t>(total_nonzero_i32);

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
    int64_t total_out,
    int* __restrict__ error_flag
) {
    int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out_idx >= total_out) return;

    int64_t p = (meta.pass_numel > 0) ? (out_idx % meta.pass_numel) : 0;
    int64_t bc = (meta.pass_numel > 0) ? (out_idx / meta.pass_numel) : out_idx;

    int64_t src_offset = 0;
    bool oob = false;
    for (int i = 0; i < meta.num_indices; ++i) {
        if (meta.is_indexed[i]) {
            int64_t idx_val = idx_ptrs[i][bc];
            if (idx_val < 0) idx_val += meta.src_shape[i];
            if (idx_val < 0 || idx_val >= meta.src_shape[i]) {
                oob = true;
                break;
            }
            src_offset += idx_val * meta.src_strides[i];
        }
    }

    if (oob) {
        atomicExch(error_flag, 1);
        // T{} rather than T(0): AdvIdxBytesN/Bytes16 are plain byte-array
        // aggregates with no converting constructor from int.
        dst[out_idx] = T{};
        return;
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
    int64_t total_out,
    int* __restrict__ error_flag
) {
    int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out_idx >= total_out) return;

    int64_t p = (meta.pass_numel > 0) ? (out_idx % meta.pass_numel) : 0;
    int64_t bc = (meta.pass_numel > 0) ? (out_idx / meta.pass_numel) : out_idx;

    int64_t dst_offset = 0;
    bool oob = false;
    for (int i = 0; i < meta.num_indices; ++i) {
        if (meta.is_indexed[i]) {
            int64_t idx_val = idx_ptrs[i][bc];
            if (idx_val < 0) idx_val += meta.src_shape[i];
            if (idx_val < 0 || idx_val >= meta.src_shape[i]) {
                oob = true;
                break;
            }
            dst_offset += idx_val * meta.src_strides[i];
        }
    }

    if (oob) {
        atomicExch(error_flag, 1);
        return;
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

    // Device-side OOB error flag: an out-of-range advanced index previously had
    // no bounds check at all here (unlike gather_kernel/take_kernel elsewhere in
    // this file), causing an unchecked out-of-bounds device memory read. Match
    // CPU's AdvancedIndex, which throws std::out_of_range.
    int* d_error_flag = nullptr;
    HIP_CHECK(hipMalloc(&d_error_flag, sizeof(int)));
    HIP_CHECK(hipMemsetAsync(d_error_flag, 0, sizeof(int), stream));

    // Use data_ptr() + reinterpret_cast for HIP-native types (__half, hip_bfloat16)
    // that don't have Tensor::data<T>() instantiations in the core library.
    hipLaunchKernelGGL(advanced_index_gather_kernel_hip<T>,
        dim3(blocks), dim3(threads), 0, stream,
        reinterpret_cast<const T*>(src_contig.data_ptr()),
        reinterpret_cast<T*>(result.data_ptr()),
        d_idx_ptrs,
        prep.meta,
        prep.total,
        d_error_flag);
    HIP_POST_LAUNCH_CHECK();

    int host_error = 0;
    HIP_CHECK(hipMemcpyAsync(&host_error, d_error_flag, sizeof(int),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_error_flag));
    HIP_CHECK(hipFree(d_idx_ptrs));
    if (host_error != 0) {
        throw std::out_of_range("AdvancedIndex: index out of range");
    }
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

    // Device-side OOB error flag: an out-of-range advanced index previously had
    // no bounds check at all here, causing an unchecked out-of-bounds device
    // memory write. Match CPU's AdvancedIndexPut, which throws std::out_of_range.
    int* d_error_flag = nullptr;
    HIP_CHECK(hipMalloc(&d_error_flag, sizeof(int)));
    HIP_CHECK(hipMemsetAsync(d_error_flag, 0, sizeof(int), stream));

    // Use data_ptr() + reinterpret_cast for HIP-native types (__half, hip_bfloat16)
    // that don't have Tensor::data<T>() instantiations in the core library.
    hipLaunchKernelGGL(advanced_index_put_kernel_hip<T>,
        dim3(blocks), dim3(threads), 0, stream,
        reinterpret_cast<T*>(result_contig.data_ptr()),
        reinterpret_cast<const T*>(values_contig.data_ptr()),
        d_idx_ptrs,
        prep.meta,
        prep.total,
        d_error_flag);
    HIP_POST_LAUNCH_CHECK();

    int host_error = 0;
    HIP_CHECK(hipMemcpyAsync(&host_error, d_error_flag, sizeof(int),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_error_flag));
    HIP_CHECK(hipFree(d_idx_ptrs));
    if (host_error != 0) {
        throw std::out_of_range("AdvancedIndexPut: index out of range");
    }
    return result_contig;
}

}  // namespace

namespace {
// Raw same-size PODs for dtype-agnostic gather/put: the gather/put kernels
// only ever do a whole-element `dst[i] = src[i]` assignment (no arithmetic),
// so any trivially-copyable type of the right byte width reproduces CPU's
// dtype-agnostic memcpy semantics exactly -- this is how Bool/Int8/UInt8/
// Int16/UInt16/UInt32/UInt64/Complex64/Complex128 get gather/put support
// without a bespoke kernel instantiation per dtype.
struct AdvIdxBytes1 { uint8_t b[1]; };
struct AdvIdxBytes2 { uint8_t b[2]; };
struct AdvIdxBytes8 { uint8_t b[8]; };
struct AdvIdxBytes16 { uint8_t b[16]; };
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
    } else if (src.dtype() == DType::Bool || src.dtype() == DType::Int8 || src.dtype() == DType::UInt8) {
        return launch_advanced_index_gather_hip<AdvIdxBytes1>(src, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Int16 || src.dtype() == DType::UInt16) {
        return launch_advanced_index_gather_hip<AdvIdxBytes2>(src, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::UInt32) {
        return launch_advanced_index_gather_hip<int32_t>(src, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::UInt64 || src.dtype() == DType::Complex64) {
        return launch_advanced_index_gather_hip<AdvIdxBytes8>(src, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Complex128) {
        return launch_advanced_index_gather_hip<AdvIdxBytes16>(src, idx_ptrs.data(), num_indices, stream);
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
    } else if (src.dtype() == DType::Bool || src.dtype() == DType::Int8 || src.dtype() == DType::UInt8) {
        return launch_advanced_index_put_hip<AdvIdxBytes1>(src, values, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Int16 || src.dtype() == DType::UInt16) {
        return launch_advanced_index_put_hip<AdvIdxBytes2>(src, values, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::UInt32) {
        return launch_advanced_index_put_hip<int32_t>(src, values, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::UInt64 || src.dtype() == DType::Complex64) {
        return launch_advanced_index_put_hip<AdvIdxBytes8>(src, values, idx_ptrs.data(), num_indices, stream);
    } else if (src.dtype() == DType::Complex128) {
        return launch_advanced_index_put_hip<AdvIdxBytes16>(src, values, idx_ptrs.data(), num_indices, stream);
    }
    throw std::runtime_error("AdvancedIndexPut ROCm: unsupported dtype");
}

// ============================================================================
// take_along_dim kernel
// ============================================================================

namespace {
constexpr int MAX_TAKE_ALONG_DIM_DIMS = 16;
}

// Per-dimension metadata for take_along_dim: the index shape (used to decode the
// flat output coordinate per axis) and the INPUT's contiguous strides (used to
// re-linearise each decoded coordinate). These differ from the index strides
// whenever the index extent is smaller than the input on a non-dim axis.
struct TakeAlongDimMeta {
    int64_t idx_shape[MAX_TAKE_ALONG_DIM_DIMS];
    int64_t in_strides[MAX_TAKE_ALONG_DIM_DIMS];
};

template<typename T>
__global__ void take_along_dim_hip_kernel(
    const T* __restrict__ input, const int64_t* __restrict__ indices, T* __restrict__ output,
    int64_t numel, int64_t in_dim_size, int ndim, int dim,
    TakeAlongDimMeta meta, int* error_flag)
{
    HIP_KERNEL_LOOP(i, numel) {
        int64_t src_idx = indices[i];
        if (src_idx < 0) src_idx += in_dim_size;

        // Bounds-check after negative normalization (matches gather/index_select):
        // an out-of-range index would otherwise be an out-of-bounds device read.
        if (src_idx < 0 || src_idx >= in_dim_size) {
            // Flag the out-of-range index; the host throws after synchronizing
            // (matches the CPU reference). Write 0 to stay memory-safe.
            error_flag[0] = 1;
            output[i] = T(0);
            continue;
        }

        // Decode the flat output position i per-dimension from idx_shape, then
        // re-linearise against the INPUT's contiguous strides. The collapsed
        // outer/inner formula is only correct when input and index share extents
        // on every non-dim axis; PyTorch/CPU allow the index extent to be smaller
        // on a non-dim axis (selecting a sub-block), which requires this per-dim
        // decode (mirrors the CPU in_offset_of lambda in indexing.cpp).
        int64_t in_offset = 0;
        int64_t rem = i;
        for (int d = ndim - 1; d >= 0; --d) {
            int64_t c = rem % meta.idx_shape[d];
            rem /= meta.idx_shape[d];
            if (d == dim) {
                in_offset += src_idx * meta.in_strides[d];
            } else {
                in_offset += c * meta.in_strides[d];
            }
        }
        output[i] = input[in_offset];
    }
}

auto take_along_dim_hip(const Tensor& input, const Tensor& indices, int64_t dim,
                        hipStream_t stream) -> Tensor {
    // The kernel addresses the input with contiguous strides, so a non-contiguous
    // (transposed/sliced/permuted) input view would otherwise read the wrong
    // storage elements. Materialize contiguous copies (matches the CPU reference).
    Tensor in_c = input.is_contiguous() ? input : input.contiguous();
    Tensor idx_c = indices.is_contiguous() ? indices : indices.contiguous();

    auto in_shape = in_c.shape();
    auto idx_shape = idx_c.shape();
    int64_t ndim = in_shape.size();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("take_along_dim ROCm: dim out of range");
    }
    if (idx_c.dtype() != DType::Int64) {
        throw std::invalid_argument("take_along_dim ROCm: index tensor must have dtype Int64");
    }

    // Index must have the same rank as input; along every non-dim axis its extent
    // must not exceed the input's (it selects a sub-block there). Mirrors CPU.
    if (static_cast<int64_t>(idx_shape.size()) != ndim) {
        throw std::invalid_argument("take_along_dim ROCm: indices must have same rank as input");
    }
    if (ndim > MAX_TAKE_ALONG_DIM_DIMS) {
        throw std::runtime_error("take_along_dim ROCm: ndim exceeds MAX_TAKE_ALONG_DIM_DIMS");
    }
    for (int64_t d = 0; d < ndim; ++d) {
        if (d == dim) continue;
        if (idx_shape[d] > in_shape[d]) {
            throw std::out_of_range(
                "take_along_dim ROCm: index shape exceeds input shape on a non-dim axis");
        }
    }

    Tensor output(std::vector<int64_t>(idx_shape.begin(), idx_shape.end()),
                  in_c.dtype(), in_c.device());
    int64_t numel = idx_c.numel();
    if (numel == 0) return output;

    int64_t in_dim_size = in_shape[dim];

    // Per-dim metadata: index shape (to decode each output coordinate) and the
    // INPUT's contiguous strides (to re-linearise every non-dim coordinate and
    // the substituted index on the `dim` axis). The collapsed outer/inner formula
    // cannot reproduce this when index and input differ on a non-dim extent.
    TakeAlongDimMeta meta{};
    {
        int64_t s = 1;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            meta.idx_shape[d] = idx_shape[d];
            meta.in_strides[d] = s;
            s *= in_shape[d];
        }
    }

    constexpr int BLOCK = 256;
    int blocks = (numel + BLOCK - 1) / BLOCK;
    const int64_t* idx_ptr = idx_c.data<int64_t>();
    // Device flag for out-of-range indices; checked on the host after sync so we
    // can throw (matching the CPU reference) instead of silently returning 0.
    Tensor err_tensor(std::vector<int64_t>{1}, DType::Int32, in_c.device());
    HIP_CHECK(hipMemsetAsync(err_tensor.data_ptr(), 0, sizeof(int32_t), stream));
    int* err_ptr = err_tensor.data<int32_t>();

    int ndim_i = static_cast<int>(ndim);
    int dim_i = static_cast<int>(dim);

    switch (in_c.dtype()) {
        case DType::Float32:
            take_along_dim_hip_kernel<float><<<blocks, BLOCK, 0, stream>>>(
                in_c.data<float>(), idx_ptr, output.data<float>(),
                numel, in_dim_size, ndim_i, dim_i, meta, err_ptr);
            break;
        case DType::Float64:
            take_along_dim_hip_kernel<double><<<blocks, BLOCK, 0, stream>>>(
                in_c.data<double>(), idx_ptr, output.data<double>(),
                numel, in_dim_size, ndim_i, dim_i, meta, err_ptr);
            break;
        case DType::Int32:
            take_along_dim_hip_kernel<int32_t><<<blocks, BLOCK, 0, stream>>>(
                in_c.data<int32_t>(), idx_ptr, output.data<int32_t>(),
                numel, in_dim_size, ndim_i, dim_i, meta, err_ptr);
            break;
        case DType::Int64:
            take_along_dim_hip_kernel<int64_t><<<blocks, BLOCK, 0, stream>>>(
                in_c.data<int64_t>(), idx_ptr, output.data<int64_t>(),
                numel, in_dim_size, ndim_i, dim_i, meta, err_ptr);
            break;
        case DType::Float16:
            take_along_dim_hip_kernel<__half><<<blocks, BLOCK, 0, stream>>>(
                reinterpret_cast<const __half*>(in_c.data_ptr()), idx_ptr,
                reinterpret_cast<__half*>(output.data_ptr()),
                numel, in_dim_size, ndim_i, dim_i, meta, err_ptr);
            break;
        case DType::BFloat16:
            take_along_dim_hip_kernel<hip_bfloat16><<<blocks, BLOCK, 0, stream>>>(
                reinterpret_cast<const hip_bfloat16*>(in_c.data_ptr()), idx_ptr,
                reinterpret_cast<hip_bfloat16*>(output.data_ptr()),
                numel, in_dim_size, ndim_i, dim_i, meta, err_ptr);
            break;
        default:
            throw std::runtime_error("take_along_dim ROCm: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    if (err_tensor.to(Device::cpu()).data<int32_t>()[0] != 0) {
        throw std::out_of_range("take_along_dim ROCm: index out of range");
    }
    return output;
}

// ============================================================================
// masked_scatter kernel — prefix sum on mask, then parallel scatter
// ============================================================================

template<typename T>
__global__ void masked_scatter_write_hip_kernel(
    const T* __restrict__ input, const bool* __restrict__ mask,
    const T* __restrict__ source, const int64_t* __restrict__ prefix_sum,
    T* __restrict__ output, int64_t numel, int64_t source_numel,
    int* __restrict__ error_flag)
{
    HIP_KERNEL_LOOP(i, numel) {
        if (mask[i]) {
            // PyTorch raises when the number of mask-true positions exceeds
            // source.numel(). Flag the error (host throws after sync) and read
            // safely in-bounds to avoid an OOB device read meanwhile.
            int64_t s = prefix_sum[i];
            if (s >= 0 && s < source_numel) {
                output[i] = source[s];
            } else {
                atomicOr(error_flag, 1);
                output[i] = input[i];
            }
        } else {
            output[i] = input[i];
        }
    }
}

__global__ void mask_to_int64_hip_kernel(const bool* __restrict__ mask, int64_t* __restrict__ out, int64_t n)
{
    HIP_KERNEL_LOOP(i, n) {
        out[i] = mask[i] ? 1 : 0;
    }
}

auto masked_scatter_hip(const Tensor& input, const Tensor& mask_arg,
                        const Tensor& source, hipStream_t stream) -> Tensor {
    // F086: accept a non-Bool (floating/int) mask by normalizing to Bool, as the
    // CPU reference does; the kernel reads mask as bool.
    Tensor mask = (mask_arg.dtype() == DType::Bool) ? mask_arg : mask_arg.to(DType::Bool);

    // F086: Float16/BFloat16 — widen input+source to Float32, scatter, narrow
    // back (pure value copy; CPU supports half, ROCm previously else-threw).
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor out_f32 = masked_scatter_hip(input.to(DType::Float32), mask,
                                            source.to(DType::Float32), stream);
        return out_f32.to(orig);
    }

    int64_t numel = input.numel();
    int64_t source_numel = source.numel();
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
    HIP_CHECK(hipcub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_int_mask, d_prefix, numel, stream));
    HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
    HIP_CHECK(hipcub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_int_mask, d_prefix, numel, stream));
    HIP_CHECK(hipFree(d_temp));
    HIP_CHECK(hipFree(d_int_mask));

    // Device error flag: set when the mask-true count exceeds source.numel()
    // (PyTorch raises in that case). Checked host-side after a sync.
    int* d_error_flag = nullptr;
    HIP_CHECK(hipMalloc(&d_error_flag, sizeof(int)));
    HIP_CHECK(hipMemsetAsync(d_error_flag, 0, sizeof(int), stream));

    switch (input.dtype()) {
        case DType::Float32:
            masked_scatter_write_hip_kernel<float><<<blocks, BLOCK, 0, stream>>>(
                input.data<float>(), mask.data<bool>(), source.data<float>(),
                d_prefix, output.data<float>(), numel, source_numel, d_error_flag);
            break;
        case DType::Float64:
            masked_scatter_write_hip_kernel<double><<<blocks, BLOCK, 0, stream>>>(
                input.data<double>(), mask.data<bool>(), source.data<double>(),
                d_prefix, output.data<double>(), numel, source_numel, d_error_flag);
            break;
        case DType::Int32:
            masked_scatter_write_hip_kernel<int32_t><<<blocks, BLOCK, 0, stream>>>(
                input.data<int32_t>(), mask.data<bool>(), source.data<int32_t>(),
                d_prefix, output.data<int32_t>(), numel, source_numel, d_error_flag);
            break;
        case DType::Int64:
            masked_scatter_write_hip_kernel<int64_t><<<blocks, BLOCK, 0, stream>>>(
                input.data<int64_t>(), mask.data<bool>(), source.data<int64_t>(),
                d_prefix, output.data<int64_t>(), numel, source_numel, d_error_flag);
            break;
        // masked_scatter is pure data movement (output[i] = source[s]), so any
        // trivially-copyable type of the right byte width reproduces CPU/CUDA's
        // dtype-agnostic copy exactly. CPU/CUDA both additionally support Bool/
        // Int8/UInt8/Int16/Complex64/Complex128 here; ROCm previously threw for
        // all of them (only Float32/64/Int32/64/Float16/BFloat16 were handled).
        case DType::Bool:
        case DType::Int8:
        case DType::UInt8:
            masked_scatter_write_hip_kernel<AdvIdxBytes1><<<blocks, BLOCK, 0, stream>>>(
                reinterpret_cast<const AdvIdxBytes1*>(input.data_ptr()), mask.data<bool>(),
                reinterpret_cast<const AdvIdxBytes1*>(source.data_ptr()),
                d_prefix, reinterpret_cast<AdvIdxBytes1*>(output.data_ptr()),
                numel, source_numel, d_error_flag);
            break;
        case DType::Int16:
            masked_scatter_write_hip_kernel<AdvIdxBytes2><<<blocks, BLOCK, 0, stream>>>(
                reinterpret_cast<const AdvIdxBytes2*>(input.data_ptr()), mask.data<bool>(),
                reinterpret_cast<const AdvIdxBytes2*>(source.data_ptr()),
                d_prefix, reinterpret_cast<AdvIdxBytes2*>(output.data_ptr()),
                numel, source_numel, d_error_flag);
            break;
        case DType::Complex64:
            masked_scatter_write_hip_kernel<AdvIdxBytes8><<<blocks, BLOCK, 0, stream>>>(
                reinterpret_cast<const AdvIdxBytes8*>(input.data_ptr()), mask.data<bool>(),
                reinterpret_cast<const AdvIdxBytes8*>(source.data_ptr()),
                d_prefix, reinterpret_cast<AdvIdxBytes8*>(output.data_ptr()),
                numel, source_numel, d_error_flag);
            break;
        case DType::Complex128:
            masked_scatter_write_hip_kernel<AdvIdxBytes16><<<blocks, BLOCK, 0, stream>>>(
                reinterpret_cast<const AdvIdxBytes16*>(input.data_ptr()), mask.data<bool>(),
                reinterpret_cast<const AdvIdxBytes16*>(source.data_ptr()),
                d_prefix, reinterpret_cast<AdvIdxBytes16*>(output.data_ptr()),
                numel, source_numel, d_error_flag);
            break;
        default:
            HIP_CHECK(hipFree(d_prefix));
            HIP_CHECK(hipFree(d_error_flag));
            throw std::runtime_error("masked_scatter ROCm: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipFree(d_prefix));

    int host_error = 0;
    HIP_CHECK(hipMemcpyAsync(&host_error, d_error_flag, sizeof(int),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_error_flag));
    if (host_error != 0) {
        throw std::runtime_error(
            "masked_scatter ROCm: number of mask-true positions exceeds "
            "source.numel()");
    }
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
