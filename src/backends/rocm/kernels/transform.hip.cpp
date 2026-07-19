#include "rocm_nan_helpers.hip.h"  // E.2: safe_f2h / safe_h2f / safe_f2bf / safe_bf2f
#include "bfloat16_helpers.hpp"   // S.10 / R.11: f32_to_bf16_rne
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_complex.h>
#include <hipcub/hipcub.hpp>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/shape.hpp"
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace tenzor {
namespace rocm {

// Sentinel meaning "squeeze every size-1 axis". A distinct value (NOT -1) is
// required so a legitimate negative axis like squeeze(-1)/squeeze(-2) is not
// silently misinterpreted as squeeze-all. Matches the CPU/OneAPI reference.
static constexpr int64_t SQUEEZE_ALL = std::numeric_limits<int64_t>::min();

// Helper class to access Tensor private members from HIP kernels.
// Routes through TensorAccessor which is a friend of Tensor.
class HIPKernelAccess {
public:
    static auto get_impl(const Tensor& t) -> const intrusive_ptr<TensorImpl>& {
        return TensorAccessor::get_impl(t);
    }
    static auto get_impl_mutable(Tensor& t) -> intrusive_ptr<TensorImpl>& {
        return TensorAccessor::get_impl_mutable(t);
    }
};

// HIP Error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            throw std::runtime_error(std::string("HIP error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + ": " + hipGetErrorString(err)); \
        } \
    } while(0)

#define HIP_GRID_STRIDE_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

// RAII guard for HIP device memory allocations
struct HipMemGuard {
    void* ptr = nullptr;
    ~HipMemGuard() { if (ptr) (void)hipFree(ptr); }
    HipMemGuard() = default;
    HipMemGuard(const HipMemGuard&) = delete;
    HipMemGuard& operator=(const HipMemGuard&) = delete;
};

constexpr int BLOCK_SIZE = 256;

inline int get_num_blocks(int64_t n, int block_size = BLOCK_SIZE) {
    return (n + block_size - 1) / block_size;
}

// ==============================================================================
// Contiguous Kernel - copies non-contiguous data to contiguous layout
// ==============================================================================

template<typename T>
__global__ void contiguous_kernel_impl(const T* input, T* output,
                                       const int64_t* strides,
                                       const int64_t* shape,
                                       int64_t ndim, int64_t total_elements) {
    HIP_GRID_STRIDE_LOOP(idx, total_elements) {
        // Convert linear index to multi-dimensional indices
        int64_t temp_idx = idx;
        int64_t src_offset = 0;

        for (int64_t dim = ndim - 1; dim >= 0; --dim) {
            int64_t coord = temp_idx % shape[dim];
            src_offset += coord * strides[dim];
            temp_idx /= shape[dim];
        }

        output[idx] = input[src_offset];
    }
}

auto contiguous_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    // If already contiguous, return as-is
    if (input.is_contiguous()) {
        return input;
    }

    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    const int64_t ndim = input.ndim();
    const int64_t total_elements = input.numel();

    if (total_elements == 0) {
        return result;  // Empty tensor
    }

    // Copy strides and shape to device memory
    std::vector<int64_t> strides_vec(input.strides().begin(), input.strides().end());

    HipMemGuard strides_guard, shape_guard;
    HIP_CHECK(hipMalloc(&strides_guard.ptr, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&shape_guard.ptr, ndim * sizeof(int64_t)));
    int64_t* d_strides = static_cast<int64_t*>(strides_guard.ptr);
    int64_t* d_shape = static_cast<int64_t*>(shape_guard.ptr);
    HIP_CHECK(hipMemcpy(d_strides, strides_vec.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_shape, shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));

    // Launch kernel
    int num_blocks = get_num_blocks(total_elements);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(contiguous_kernel_impl<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<float>(), result.data<float>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(contiguous_kernel_impl<double>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<double>(), result.data<double>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(contiguous_kernel_impl<int32_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<int32_t>(), result.data<int32_t>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(contiguous_kernel_impl<int64_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<int64_t>(), result.data<int64_t>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(contiguous_kernel_impl<__half>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(contiguous_kernel_impl<hip_bfloat16>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::UInt8) {
        hipLaunchKernelGGL(contiguous_kernel_impl<uint8_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<uint8_t>(), result.data<uint8_t>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Int8) {
        hipLaunchKernelGGL(contiguous_kernel_impl<int8_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<int8_t>(), result.data<int8_t>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Bool) {
        hipLaunchKernelGGL(contiguous_kernel_impl<bool>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<bool>(), result.data<bool>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Complex64) {
        // Complex64 = 2x float; treat as float2 (8 bytes/element)
        hipLaunchKernelGGL(contiguous_kernel_impl<float2>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const float2*>(input.data_ptr()),
            reinterpret_cast<float2*>(result.data_ptr()),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Complex128) {
        hipLaunchKernelGGL(contiguous_kernel_impl<double2>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const double2*>(input.data_ptr()),
            reinterpret_cast<double2*>(result.data_ptr()),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Int16) {
        hipLaunchKernelGGL(contiguous_kernel_impl<int16_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<int16_t>(), result.data<int16_t>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::UInt16) {
        hipLaunchKernelGGL(contiguous_kernel_impl<uint16_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<uint16_t>(), result.data<uint16_t>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::UInt32) {
        hipLaunchKernelGGL(contiguous_kernel_impl<uint32_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<uint32_t>(), result.data<uint32_t>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::UInt64) {
        hipLaunchKernelGGL(contiguous_kernel_impl<uint64_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<uint64_t>(), result.data<uint64_t>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::FP8_E4M3 || input.dtype() == DType::FP8_E5M2 ||
               input.dtype() == DType::FP8_E4M3FNUZ || input.dtype() == DType::FP8_E5M2FNUZ ||
               input.dtype() == DType::QInt8 || input.dtype() == DType::QUInt8 ||
               input.dtype() == DType::QInt4x2) {
        // 1-byte storage types: a raw byte-wise strided copy preserves the value
        // (and, for QInt4x2, both packed nibbles since strides are byte-granular).
        hipLaunchKernelGGL(contiguous_kernel_impl<uint8_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const uint8_t*>(input.data_ptr()),
            reinterpret_cast<uint8_t*>(result.data_ptr()),
            d_strides, d_shape, ndim, total_elements);
    } else {
        throw std::runtime_error("Contiguous: unsupported dtype");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP error in contiguous_kernel: ") + hipGetErrorString(err));
    }

    // Synchronize before the HipMemGuard destructors free d_strides / d_shape.
    // The kernel launch above is asynchronous on `stream`; without this sync
    // the freed device memory would be read by the still-running kernel,
    // causing a use-after-free crash.
    HIP_CHECK(hipStreamSynchronize(stream));

    return result;
}

// ==============================================================================
// Clone Kernel - device-to-device copy
// ==============================================================================

auto clone_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    // Make contiguous first if needed
    Tensor cont = input.is_contiguous() ? input : contiguous_kernel(input, stream);

    // Create new tensor
    std::vector<int64_t> shape(cont.shape().begin(), cont.shape().end());
    Tensor result(shape, cont.dtype(), cont.device());

    // Copy data using hipMemcpy
    const size_t size_bytes = cont.numel() * dtype_size(cont.dtype());
    HIP_CHECK(hipMemcpyAsync(result.data<uint8_t>(), cont.data<uint8_t>(),
                              size_bytes, hipMemcpyDeviceToDevice, stream));

    return result;
}

// ==============================================================================
// Reshape Kernel - metadata manipulation (create view with new shape)
// ==============================================================================

auto reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape, hipStream_t stream) -> Tensor {
    // Reshape just manipulates metadata - create view
    // If not contiguous, need to make contiguous first
    if (!input.is_contiguous()) {
        return reshape_kernel(contiguous_kernel(input, stream), new_shape, stream);
    }

    // Create new tensor sharing storage (view)
    Tensor result;
    HIPKernelAccess::get_impl_mutable(result) = make_intrusive<TensorImpl>(*HIPKernelAccess::get_impl(input));
    result.mutable_shape() = new_shape;
    result.mutable_strides() = compute_strides(new_shape);

    return result;
}

// ==============================================================================
// Transpose Kernel - metadata manipulation (swap dimensions)
// ==============================================================================

auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1, hipStream_t stream) -> Tensor {
    // Transpose just swaps dimensions in metadata
    const int64_t ndim = input.ndim();
    // Normalize negative dims (the dispatcher passes dim0/dim1 through
    // unmodified, so the kernel owns normalization here, matching the sibling
    // flip/roll/split/chunk kernels in this file).
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        throw std::out_of_range("transpose ROCm: dimension out of range");
    }

    Tensor result;
    HIPKernelAccess::get_impl_mutable(result) = make_intrusive<TensorImpl>(*HIPKernelAccess::get_impl(input));
    auto& r_shape = result.mutable_shape();
    auto& r_strides = result.mutable_strides();
    std::swap(r_shape[dim0], r_shape[dim1]);
    std::swap(r_strides[dim0], r_strides[dim1]);
    return result;
}

// ==============================================================================
// Permute Kernel - metadata manipulation (permute dimensions)
// ==============================================================================

auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims, hipStream_t stream) -> Tensor {
    const int64_t ndim = input.ndim();

    Tensor result;
    HIPKernelAccess::get_impl_mutable(result) = make_intrusive<TensorImpl>(*HIPKernelAccess::get_impl(input));

    std::vector<int64_t> new_shape(ndim);
    std::vector<int64_t> new_strides(ndim);

    for (int64_t i = 0; i < ndim; ++i) {
        // Normalize negative axis: permute([-1, ...]) is the common idiom and
        // the dispatcher passes dims through unmodified.
        int64_t d = dims[i];
        if (d < 0) d += ndim;
        if (d < 0 || d >= ndim) {
            throw std::out_of_range("permute ROCm: dimension out of range");
        }
        new_shape[i] = input.shape()[d];
        new_strides[i] = input.strides()[d];
    }

    result.mutable_shape() = std::move(new_shape);
    result.mutable_strides() = std::move(new_strides);

    return result;
}

// ==============================================================================
// Squeeze Kernel - metadata manipulation (remove dimension)
// ==============================================================================

auto squeeze_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor {
    Tensor result;
    HIPKernelAccess::get_impl_mutable(result) = make_intrusive<TensorImpl>(*HIPKernelAccess::get_impl(input));

    if (dim != SQUEEZE_ALL) {
        // Squeeze a specific dimension. Normalize negatives and validate range
        // and that the axis is actually size 1 (mirrors the CPU reference and
        // Tensor::squeeze). A real squeeze(-1) is distinct from squeeze-all.
        const int64_t ndim = input.ndim();
        if (dim < 0) dim += ndim;
        if (dim < 0 || dim >= ndim) {
            throw std::out_of_range("squeeze ROCm: dimension out of range");
        }
        // PyTorch leaves a non-size-1 axis untouched.
        if (input.shape()[dim] != 1) {
            return result;
        }
        auto& r_shape = result.mutable_shape();
        auto& r_strides = result.mutable_strides();
        r_shape.erase(r_shape.begin() + dim);
        r_strides.erase(r_strides.begin() + dim);
    } else {
        // Squeeze all dimensions with size 1
        std::vector<int64_t> new_shape;
        std::vector<int64_t> new_strides;

        for (int64_t i = 0; i < input.ndim(); ++i) {
            if (input.shape()[i] != 1) {
                new_shape.push_back(input.shape()[i]);
                new_strides.push_back(input.strides()[i]);
            }
        }

        if (new_shape.empty()) {
            new_shape.push_back(1);
            new_strides.push_back(1);
        }

        result.mutable_shape() = std::move(new_shape);
        result.mutable_strides() = std::move(new_strides);
    }

    return result;
}

// ==============================================================================
// Unsqueeze Kernel - metadata manipulation (add dimension)
// ==============================================================================

auto unsqueeze_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor {
    // JIT-R107: negative dim was never normalized here, so unsqueeze(-1) (as
    // used by SyncBatchNormBackward::backward_with_variables) indexed
    // strides()[-1] via std::span::operator[], hitting an out-of-bounds
    // assertion (SIGABRT) instead of a catchable exception. Mirrors the
    // canonical Tensor::unsqueeze (src/core/tensor.cpp) and this file's own
    // squeeze_kernel convention: valid range is [-(ndim+1), ndim] since the
    // result has one more dimension than the input.
    const int64_t ndims = input.ndim();
    const int64_t new_ndims = ndims + 1;
    if (dim < 0) dim += new_ndims;
    if (dim < 0 || dim >= new_ndims) {
        throw std::out_of_range("unsqueeze ROCm: dimension out of range");
    }

    Tensor result;
    HIPKernelAccess::get_impl_mutable(result) = make_intrusive<TensorImpl>(*HIPKernelAccess::get_impl(input));

    auto& r_shape = result.mutable_shape();
    auto& r_strides = result.mutable_strides();
    r_shape.insert(r_shape.begin() + dim, 1);

    // Stride for the inserted size-1 dim (matches Tensor::unsqueeze): for a
    // contiguous parent, strides[dim]*shape[dim] keeps the result reporting
    // contiguous; strides[dim] alone (the old formula, unconditionally) is
    // only correct for a non-contiguous parent or the trailing insertion.
    int64_t new_stride;
    if (input.is_contiguous()) {
        new_stride = (dim < ndims) ? (input.strides()[dim] * input.shape()[dim]) : 1;
    } else {
        new_stride = (dim < ndims) ? input.strides()[dim] : 1;
    }
    r_strides.insert(r_strides.begin() + dim, new_stride);

    return result;
}

// ==============================================================================
// Cat (Concatenate) Kernel
// ==============================================================================

template<typename T>
__global__ void cat_kernel_impl(
    T** inputs,
    T* output,
    int64_t* input_sizes,
    int64_t* output_offsets,
    int64_t num_tensors,
    int64_t concat_dim_size,
    int64_t inner_size,
    int64_t outer_size
) {
    int64_t total_elements = outer_size * concat_dim_size * inner_size;

    HIP_GRID_STRIDE_LOOP(idx, total_elements) {
        int64_t inner_idx = idx % inner_size;
        int64_t concat_idx = (idx / inner_size) % concat_dim_size;
        int64_t outer_idx = idx / (inner_size * concat_dim_size);

        // Find which input tensor this element comes from
        int64_t current_offset = 0;
        for (int64_t i = 0; i < num_tensors; ++i) {
            if (concat_idx < current_offset + input_sizes[i]) {
                int64_t local_concat_idx = concat_idx - current_offset;
                int64_t input_idx = (outer_idx * input_sizes[i] + local_concat_idx) * inner_size + inner_idx;
                output[idx] = inputs[i][input_idx];
                break;
            }
            current_offset += input_sizes[i];
        }
    }
}

// ==============================================================================
// Split Kernel
// ==============================================================================

template<typename T>
__global__ void split_kernel_impl(
    const T* input,
    T** outputs,
    int64_t* split_sizes,
    int64_t* split_offsets,
    int64_t num_splits,
    int64_t split_dim_size,
    int64_t inner_size,
    int64_t outer_size
) {
    int64_t total_elements = outer_size * split_dim_size * inner_size;

    HIP_GRID_STRIDE_LOOP(idx, total_elements) {
        int64_t inner_idx = idx % inner_size;
        int64_t split_idx = (idx / inner_size) % split_dim_size;
        int64_t outer_idx = idx / (inner_size * split_dim_size);

        // Find which output tensor this element goes to
        int64_t current_offset = 0;
        for (int64_t i = 0; i < num_splits; ++i) {
            if (split_idx < current_offset + split_sizes[i]) {
                int64_t local_split_idx = split_idx - current_offset;
                int64_t output_idx = (outer_idx * split_sizes[i] + local_split_idx) * inner_size + inner_idx;
                outputs[i][output_idx] = input[idx];
                break;
            }
            current_offset += split_sizes[i];
        }
    }
}

// ==============================================================================
// Chunk Kernel - split into equal-sized chunks
// ==============================================================================

// Forward declaration: chunk delegates to split_kernel (defined later in this
// translation unit) so it reuses split's real device-to-device copy path.
auto split_kernel(const Tensor& input, int64_t split_size, int64_t dim, hipStream_t stream) -> std::vector<Tensor>;

auto chunk_kernel(const Tensor& input, int64_t chunks, int64_t dim, hipStream_t stream) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    // audit V.14: normalise negative dim (PyTorch convention) and range-check
    // before indexing into input_shape.
    const int64_t ndim = static_cast<int64_t>(input_shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("chunk: dim out of range");
    }
    if (chunks <= 0) {
        throw std::invalid_argument("chunk: chunks must be positive");
    }
    int64_t dim_size = input_shape[dim];

    // PyTorch chunk semantics: split into at most `chunks` pieces, each of size
    // ceil(dim_size / chunks) (the final piece may be smaller). This is exactly
    // split() with split_size = chunk_size, so delegate to split_kernel, which
    // performs the real device-to-device copy. (Previously this allocated empty
    // output tensors and returned uninitialised device memory.)
    int64_t chunk_size = (dim_size + chunks - 1) / chunks;
    if (chunk_size <= 0) {
        // dim_size == 0: split_kernel rejects a zero split_size with a throw.
        // PyTorch's chunk() on an empty dim returns `chunks` empty tensors
        // (each carrying a 0 in the chunk dim), not a single one — matches
        // CPU's chunk_kernel (src/backends/cpu/kernels/transform.cpp).
        std::vector<int64_t> empty_shape(input_shape.begin(), input_shape.end());
        empty_shape[dim] = 0;
        std::vector<Tensor> result;
        result.reserve(static_cast<size_t>(chunks));
        for (int64_t c = 0; c < chunks; ++c) {
            result.emplace_back(empty_shape, input.dtype(), input.device());
        }
        return result;
    }
    return split_kernel(input, chunk_size, dim, stream);
}

// ==============================================================================
// Flip Kernel - reverse elements along dimension
// ==============================================================================

template<typename T>
__global__ void flip_kernel_impl(
    const T* input,
    T* output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size
) {
    int64_t total_elements = outer_size * dim_size * inner_size;

    HIP_GRID_STRIDE_LOOP(idx, total_elements) {
        int64_t inner_idx = idx % inner_size;
        int64_t dim_idx = (idx / inner_size) % dim_size;
        int64_t outer_idx = idx / (inner_size * dim_size);

        // Flip the dimension index
        int64_t flipped_dim_idx = dim_size - 1 - dim_idx;
        int64_t input_idx = (outer_idx * dim_size + flipped_dim_idx) * inner_size + inner_idx;

        output[idx] = input[input_idx];
    }
}

auto flip_kernel(const Tensor& input_orig, int64_t dim, hipStream_t stream) -> Tensor {
    // The kernel below uses stride-from-shape addressing (outer*dim_size +
    // flipped)*inner_size, i.e. it assumes contiguous row-major layout. If
    // the input is a non-contiguous view (slice, transpose, permute), the
    // raw data_ptr would not match that layout and the flip would read the
    // wrong elements. Materialize a contiguous copy first.
    Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
    auto input_shape = input.shape();

    // Normalize negative dim (PyTorch semantics) before indexing shape[dim].
    int64_t ndim = static_cast<int64_t>(input_shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("flip: dim out of range");
    }

    Tensor result(std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                  input.dtype(), input.device());

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

    int64_t total_elements = input.numel();
    int num_blocks = get_num_blocks(total_elements);
    // Empty output: a zero-grid launch is rejected by HIP; return as-is.
    if (num_blocks == 0) return result;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(flip_kernel_impl<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<float>(), result.data<float>(),
            outer_size, dim_size, inner_size);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(flip_kernel_impl<double>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<double>(), result.data<double>(),
            outer_size, dim_size, inner_size);
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(flip_kernel_impl<int32_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<int32_t>(), result.data<int32_t>(),
            outer_size, dim_size, inner_size);
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(flip_kernel_impl<int64_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<int64_t>(), result.data<int64_t>(),
            outer_size, dim_size, inner_size);
    } else if (input.dtype() == DType::Float16) {
        // Reinterpret as uint16 — flip is a pure data-movement op, no
        // arithmetic, so dtype is irrelevant beyond element size.
        hipLaunchKernelGGL(flip_kernel_impl<uint16_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const uint16_t*>(input.data_ptr()),
            reinterpret_cast<uint16_t*>(result.data_ptr()),
            outer_size, dim_size, inner_size);
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(flip_kernel_impl<uint16_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const uint16_t*>(input.data_ptr()),
            reinterpret_cast<uint16_t*>(result.data_ptr()),
            outer_size, dim_size, inner_size);
    } else {
        throw std::runtime_error("Flip only supports Float16/32/64, BFloat16, and Int32/64 dtypes");
    }

    HIP_CHECK(hipGetLastError());

    return result;
}

// ==============================================================================
// Flatten Kernel - flatten dimensions start_dim to end_dim
// ==============================================================================

auto flatten_kernel(const Tensor& input, int64_t start_dim, int64_t end_dim, hipStream_t stream) -> Tensor {
    auto input_shape = input.shape();
    int64_t ndim = input_shape.size();

    // Handle negative dimensions
    if (start_dim < 0) start_dim += ndim;
    if (end_dim < 0) end_dim += ndim;

    // Compute flattened size
    int64_t flattened_size = 1;
    for (int64_t i = start_dim; i <= end_dim; ++i) {
        flattened_size *= input_shape[i];
    }

    // Build new shape
    std::vector<int64_t> new_shape;
    for (int64_t i = 0; i < start_dim; ++i) {
        new_shape.push_back(input_shape[i]);
    }
    new_shape.push_back(flattened_size);
    for (int64_t i = end_dim + 1; i < ndim; ++i) {
        new_shape.push_back(input_shape[i]);
    }

    // Flatten is just a reshape
    return reshape_kernel(input, new_shape, stream);
}

// ==============================================================================
// Repeat Kernel - repeat tensor along dimensions
// ==============================================================================

template<typename T>
__global__ void repeat_kernel_impl(
    const T* input,
    T* output,
    const int64_t* input_shape,   // effective input shape (with leading size-1 dims), length ndim
    const int64_t* output_shape,
    int64_t ndim,
    int64_t total_elements
) {
    HIP_GRID_STRIDE_LOOP(idx, total_elements) {
        int64_t temp = idx;
        int64_t input_offset = 0;
        int64_t input_stride = 1;

        for (int64_t d = ndim - 1; d >= 0; --d) {
            int64_t coord = temp % output_shape[d];
            temp /= output_shape[d];

            // Map output coord to input coord via modulus by the (effective)
            // input extent. This is "tile" semantics matching torch.repeat and
            // the CPU ground truth (src/backends/cpu/kernels/transform.cpp:711),
            // e.g. [a,b] repeated 2x -> [a,b,a,b]. Leading repeat factors beyond
            // input.ndim() act on effective size-1 dims (input_coord always 0),
            // which correctly prepends new outer dimensions.
            int64_t input_coord = coord % input_shape[d];

            input_offset += input_coord * input_stride;
            input_stride *= input_shape[d];
        }

        output[idx] = input[input_offset];
    }
}

auto repeat_kernel(const Tensor& input_in, const std::vector<int64_t>& repeats, hipStream_t stream) -> Tensor {
    // Kernel below derives contiguous strides from input_shape; non-contiguous
    // input views would be read at wrong offsets. Materialize first.
    Tensor input = input_in.is_contiguous() ? input_in : input_in.contiguous();
    auto in_shape = input.shape();
    // torch.repeat semantics: repeats may have more entries than input.ndim().
    // Extra leading repeats prepend new outer dims acting on effective size-1
    // input dims (mirrors CPU: src/backends/cpu/kernels/transform.cpp:682-701).
    int64_t ndim = static_cast<int64_t>(repeats.size());
    int64_t dim_diff = ndim - static_cast<int64_t>(input.ndim());

    // Effective input shape (with leading size-1 dims) and output shape.
    std::vector<int64_t> effective_in_shape(ndim);
    std::vector<int64_t> output_shape(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t in_idx = i - dim_diff;
        int64_t in_dim = (in_idx >= 0) ? in_shape[in_idx] : 1;
        effective_in_shape[i] = in_dim;
        output_shape[i] = in_dim * repeats[i];
    }

    Tensor output(output_shape, input.dtype(), input.device());
    int64_t total_elements = output.numel();

    if (total_elements == 0) return output;

    // Copy shapes to device
    int64_t* d_input_shape;
    int64_t* d_output_shape;
    HIP_CHECK(hipMalloc(&d_input_shape, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_output_shape, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(d_input_shape, effective_in_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_output_shape, output_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));

    int num_blocks = get_num_blocks(total_elements);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(repeat_kernel_impl<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<float>(), output.data<float>(),
            d_input_shape, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(repeat_kernel_impl<double>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<double>(), output.data<double>(),
            d_input_shape, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(repeat_kernel_impl<int32_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<int32_t>(), output.data<int32_t>(),
            d_input_shape, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(repeat_kernel_impl<int64_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<int64_t>(), output.data<int64_t>(),
            d_input_shape, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(repeat_kernel_impl<__half>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            d_input_shape, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(repeat_kernel_impl<hip_bfloat16>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            d_input_shape, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Int16 || input.dtype() == DType::UInt16) {
        hipLaunchKernelGGL(repeat_kernel_impl<uint16_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const uint16_t*>(input.data_ptr()),
            reinterpret_cast<uint16_t*>(output.data_ptr()),
            d_input_shape, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::UInt32) {
        hipLaunchKernelGGL(repeat_kernel_impl<uint32_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const uint32_t*>(input.data_ptr()),
            reinterpret_cast<uint32_t*>(output.data_ptr()),
            d_input_shape, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Int8 || input.dtype() == DType::UInt8 ||
               input.dtype() == DType::Bool) {
        hipLaunchKernelGGL(repeat_kernel_impl<uint8_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const uint8_t*>(input.data_ptr()),
            reinterpret_cast<uint8_t*>(output.data_ptr()),
            d_input_shape, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::UInt64 || input.dtype() == DType::Complex64) {
        hipLaunchKernelGGL(repeat_kernel_impl<uint64_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const uint64_t*>(input.data_ptr()),
            reinterpret_cast<uint64_t*>(output.data_ptr()),
            d_input_shape, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Complex128) {
        hipLaunchKernelGGL(repeat_kernel_impl<double2>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const double2*>(input.data_ptr()),
            reinterpret_cast<double2*>(output.data_ptr()),
            d_input_shape, d_output_shape, ndim, total_elements);
    } else {
        HIP_CHECK(hipFree(d_input_shape));
        HIP_CHECK(hipFree(d_output_shape));
        throw std::runtime_error("repeat_kernel: unsupported dtype");
    }

    // audit-9 JJ.5: sync stream before freeing async-kernel input buffers.
    // hipFree is device-sync only on the default stream; on a user stream
    // the kernel may still be reading d_*_shape when the page gets reused.
    // Mirrors stack_kernel pattern at L833.
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_input_shape));
    HIP_CHECK(hipFree(d_output_shape));
    HIP_CHECK(hipGetLastError());

    return output;
}

// ==============================================================================
// Tile Kernel - tile tensor (like repeat but prepends dimensions if needed)
// ==============================================================================

// 16-byte POD for tiling Complex128 / 16-byte elements (pure data movement).
namespace { struct alignas(16) TileBytes16 { uint64_t lo; uint64_t hi; }; }

// Proper tile (numpy.tile / torch.tile): block-repeat the whole tensor along
// each dim via per-dim modulo gather — NOT element-wise repeat. Each output
// element maps back to in[ (coord_d % in_shape_d) ... ]. Mirrors the CUDA path;
// the previous implementation delegated to repeat_kernel (element-wise), which
// produced [1,1,2,2,3,3] instead of the tiled [1,2,3,1,2,3].
template<typename T>
__global__ void tile_kernel_device(const T* __restrict__ in, T* __restrict__ out,
        int64_t out_numel, int ndim,
        const int64_t* __restrict__ out_shape,
        const int64_t* __restrict__ in_shape,
        const int64_t* __restrict__ in_strides) {
    HIP_GRID_STRIDE_LOOP(idx, out_numel) {
        int64_t rem = idx;
        int64_t in_off = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            int64_t coord = rem % out_shape[d];
            rem /= out_shape[d];
            in_off += (coord % in_shape[d]) * in_strides[d];
        }
        out[idx] = in[in_off];
    }
}

auto tile_kernel(const Tensor& input, const std::vector<int64_t>& reps, hipStream_t stream) -> Tensor {
    Tensor in = input.is_contiguous() ? input : input.contiguous();
    auto ishape = in.shape();
    int64_t in_ndim = static_cast<int64_t>(ishape.size());
    int64_t out_ndim = std::max(in_ndim, static_cast<int64_t>(reps.size()));

    // Right-align input shape and reps, padding leading dims with 1.
    std::vector<int64_t> pshape(out_ndim, 1), preps(out_ndim, 1);
    for (int64_t i = 0; i < in_ndim; ++i) pshape[out_ndim - in_ndim + i] = ishape[i];
    for (int64_t i = 0; i < static_cast<int64_t>(reps.size()); ++i)
        preps[out_ndim - static_cast<int64_t>(reps.size()) + i] = reps[i];

    std::vector<int64_t> out_shape(out_ndim), in_strides(out_ndim);
    int64_t s = 1;
    for (int64_t d = out_ndim - 1; d >= 0; --d) {
        in_strides[d] = s; s *= pshape[d];
        out_shape[d] = pshape[d] * preps[d];
    }
    int64_t out_numel = 1;
    for (auto v : out_shape) out_numel *= v;

    Tensor output(out_shape, in.dtype(), in.device());
    if (out_numel == 0) return output;

    std::vector<int64_t> meta;
    meta.reserve(out_ndim * 3);
    meta.insert(meta.end(), out_shape.begin(), out_shape.end());
    meta.insert(meta.end(), pshape.begin(), pshape.end());
    meta.insert(meta.end(), in_strides.begin(), in_strides.end());
    int64_t* d_meta = nullptr;
    HIP_CHECK(hipMalloc(&d_meta, meta.size() * sizeof(int64_t)));
    HIP_CHECK(hipMemcpyAsync(d_meta, meta.data(), meta.size() * sizeof(int64_t),
                             hipMemcpyHostToDevice, stream));
    const int64_t* d_out_shape = d_meta;
    const int64_t* d_in_shape = d_meta + out_ndim;
    const int64_t* d_in_strides = d_meta + 2 * out_ndim;

    int num_blocks = get_num_blocks(out_numel);
    auto launch = [&](auto* tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        hipLaunchKernelGGL(tile_kernel_device<T>, dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const T*>(in.data_ptr()), reinterpret_cast<T*>(output.data_ptr()),
            out_numel, static_cast<int>(out_ndim), d_out_shape, d_in_shape, d_in_strides);
        HIP_CHECK(hipGetLastError());
    };

    // tile is pure data movement; dispatch by element size.
    switch (dtype_size(in.dtype())) {
        case 1:  launch(static_cast<uint8_t*>(nullptr));  break;
        case 2:  launch(static_cast<uint16_t*>(nullptr)); break;
        case 4:  launch(static_cast<uint32_t*>(nullptr)); break;
        case 8:  launch(static_cast<uint64_t*>(nullptr)); break;
        case 16: launch(static_cast<TileBytes16*>(nullptr)); break;
        default:
            HIP_CHECK(hipFree(d_meta));
            throw std::runtime_error("tile: unsupported element size");
    }

    // Sync before freeing d_meta — the async kernel still reads it.
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_meta));
    return output;
}

// ==============================================================================
// Stack Kernel - stack tensors along new dimension
// ==============================================================================

// Stack along arbitrary dim: decompose output flat idx into
// (outer, tensor, inner) using outer_size/inner_size of the input shape.
// The previous `tensor_idx = idx / tensor_size` mapping only held for
// dim=0; for dim>=1 it selected the wrong input tensor for each output
// position, producing ~5.5 absolute error vs CPU on {32,32} StackDim1.
template<typename T>
__global__ void stack_kernel_impl(
    const T* const* inputs,
    T* output,
    int64_t num_tensors,
    int64_t tensor_size,
    int64_t outer_size,
    int64_t inner_size
) {
    int64_t total_elements = num_tensors * tensor_size;

    HIP_GRID_STRIDE_LOOP(idx, total_elements) {
        int64_t inner = idx % inner_size;
        int64_t tmp = idx / inner_size;
        int64_t tensor_idx = tmp % num_tensors;
        int64_t outer = tmp / num_tensors;
        int64_t elem_idx = outer * inner_size + inner;
        output[idx] = inputs[tensor_idx][elem_idx];
    }
}

auto stack_kernel(const std::vector<Tensor>& tensors, int64_t dim, hipStream_t stream) -> std::vector<Tensor> {
    if (tensors.empty()) {
        throw std::runtime_error("stack_kernel: tensors list cannot be empty");
    }

    auto& first = tensors[0];
    auto first_shape = first.shape();
    int64_t ndim = first_shape.size();

    // Handle negative dim
    if (dim < 0) dim += ndim + 1;

    // Build output shape (insert new dimension)
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < dim; ++i) {
        output_shape.push_back(first_shape[i]);
    }
    output_shape.push_back(static_cast<int64_t>(tensors.size()));
    for (int64_t i = dim; i < ndim; ++i) {
        output_shape.push_back(first_shape[i]);
    }

    Tensor output(output_shape, first.dtype(), first.device());

    // Copy input pointers to device
    std::vector<const void*> h_input_ptrs(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        h_input_ptrs[i] = tensors[i].data_ptr();
    }

    void** d_input_ptrs;
    HIP_CHECK(hipMalloc(&d_input_ptrs, tensors.size() * sizeof(void*)));
    HIP_CHECK(hipMemcpy(d_input_ptrs, h_input_ptrs.data(), tensors.size() * sizeof(void*), hipMemcpyHostToDevice));

    int64_t tensor_size = first.numel();
    int64_t total_elements = tensors.size() * tensor_size;
    int num_blocks = get_num_blocks(total_elements);

    // Pre-compute outer/inner sizes for stack-along-arbitrary-dim.
    int64_t outer_size = 1;
    for (int64_t d = 0; d < dim; ++d) outer_size *= first_shape[d];
    int64_t inner_size = tensor_size / outer_size;

    int64_t n_tensors_i = static_cast<int64_t>(tensors.size());

    if (first.dtype() == DType::Float32) {
        hipLaunchKernelGGL(stack_kernel_impl<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            (const float* const*)d_input_ptrs, output.data<float>(),
            n_tensors_i, tensor_size, outer_size, inner_size);
    } else if (first.dtype() == DType::Float64) {
        hipLaunchKernelGGL(stack_kernel_impl<double>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            (const double* const*)d_input_ptrs, output.data<double>(),
            n_tensors_i, tensor_size, outer_size, inner_size);
    } else if (first.dtype() == DType::Int32) {
        hipLaunchKernelGGL(stack_kernel_impl<int32_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            (const int32_t* const*)d_input_ptrs, output.data<int32_t>(),
            n_tensors_i, tensor_size, outer_size, inner_size);
    } else if (first.dtype() == DType::Int64) {
        hipLaunchKernelGGL(stack_kernel_impl<int64_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            (const int64_t* const*)d_input_ptrs, output.data<int64_t>(),
            n_tensors_i, tensor_size, outer_size, inner_size);
    } else if (first.dtype() == DType::Float16) {
        hipLaunchKernelGGL(stack_kernel_impl<__half>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            (const __half* const*)d_input_ptrs,
            reinterpret_cast<__half*>(output.data<Float16>()),
            n_tensors_i, tensor_size, outer_size, inner_size);
    } else if (first.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(stack_kernel_impl<hip_bfloat16>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            (const hip_bfloat16* const*)d_input_ptrs,
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            n_tensors_i, tensor_size, outer_size, inner_size);
    } else {
        HIP_CHECK(hipFree(d_input_ptrs));
        throw std::runtime_error("stack_kernel: unsupported dtype");
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_input_ptrs));
    HIP_CHECK(hipGetLastError());

    return {output};
}

// ==============================================================================
// Split Kernel - split tensor into chunks
// ==============================================================================

auto split_kernel(const Tensor& input, int64_t split_size, int64_t dim, hipStream_t stream) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    // Y.13: normalise negative dim (V.14 fixed chunk_kernel but not split_kernel).
    if (dim < 0) {
        dim += static_cast<int64_t>(input_shape.size());
    }
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::out_of_range(
            "split_kernel (ROCm): dim " + std::to_string(dim) +
            " out of range for tensor of rank " + std::to_string(input_shape.size()));
    }
    int64_t dim_size = input_shape[dim];

    std::vector<Tensor> results;
    int64_t current_offset = 0;

    while (current_offset < dim_size) {
        int64_t current_size = std::min(split_size, dim_size - current_offset);

        // Create output tensor for this split
        std::vector<int64_t> split_shape(input_shape.begin(), input_shape.end());
        split_shape[dim] = current_size;

        Tensor split_tensor(split_shape, input.dtype(), input.device());

        // Calculate dimensions for copy
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= input_shape[i];
        }

        int64_t inner_size = 1;
        for (size_t i = dim + 1; i < input_shape.size(); ++i) {
            inner_size *= input_shape[i];
        }

        // Copy data
        size_t elem_size = dtype_size(input.dtype());
        for (int64_t o = 0; o < outer_size; ++o) {
            const uint8_t* src = input.data<uint8_t>() +
                (o * dim_size + current_offset) * inner_size * elem_size;
            uint8_t* dst = split_tensor.data<uint8_t>() +
                o * current_size * inner_size * elem_size;
            HIP_CHECK(hipMemcpyAsync(dst, src,
                current_size * inner_size * elem_size,
                hipMemcpyDeviceToDevice, stream));
        }

        results.push_back(split_tensor);
        current_offset += current_size;
    }

    // Y.13: dropped trailing hipStreamSynchronize(stream) — caller controls
    // synchronisation, this defeated async pipelining.
    return results;
}

// ==============================================================================
// Expand Kernel - expand tensor to new shape (broadcast)
// ==============================================================================

template<typename T>
__global__ void expand_kernel_impl(
    const T* input,
    T* output,
    const int64_t* input_shape,
    const int64_t* input_strides,
    const int64_t* output_shape,
    int64_t ndim,
    int64_t total_elements
) {
    HIP_GRID_STRIDE_LOOP(idx, total_elements) {
        // Convert output index to input index using broadcast rules
        int64_t temp = idx;
        int64_t input_offset = 0;

        for (int64_t d = ndim - 1; d >= 0; --d) {
            int64_t coord = temp % output_shape[d];
            temp /= output_shape[d];

            // If input has size 1 in this dim, don't advance (broadcast)
            if (input_shape[d] != 1) {
                input_offset += coord * input_strides[d];
            }
        }

        output[idx] = input[input_offset];
    }
}

auto expand_kernel(const Tensor& input, const std::vector<int64_t>& new_shape, void* stream_ptr) -> Tensor {
    hipStream_t stream = static_cast<hipStream_t>(stream_ptr);
    auto input_shape = input.shape();
    auto input_strides = input.strides();
    int64_t ndim = new_shape.size();

    int64_t input_ndim = input_shape.size();
    if (input_ndim > ndim) {
        throw std::runtime_error(
            "expand: target shape must have at least as many dimensions as input");
    }

    // Pad input shape/strides if needed
    std::vector<int64_t> padded_input_shape(ndim, 1);
    std::vector<int64_t> padded_input_strides(ndim, 0);

    int64_t pad_size = ndim - input_ndim;

    for (int64_t i = 0; i < input_ndim; ++i) {
        padded_input_shape[pad_size + i] = input_shape[i];
        padded_input_strides[pad_size + i] = input_strides[i];
    }

    // Validate: each dim must either broadcast (input_dim == 1) or match new_shape.
    // A "-1" entry in new_shape carries the input dim unchanged.
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t in_d = padded_input_shape[i];
        int64_t out_d = new_shape[i];
        if (out_d == -1) continue;
        if (in_d != 1 && in_d != out_d) {
            throw std::runtime_error(
                "expand: size mismatch at dim " + std::to_string(i) +
                " — cannot expand dim of size " + std::to_string(in_d) +
                " to " + std::to_string(out_d));
        }
    }

    Tensor output(new_shape, input.dtype(), input.device());
    int64_t total_elements = output.numel();

    if (total_elements == 0) return output;

    // Copy to device
    int64_t* d_input_shape;
    int64_t* d_input_strides;
    int64_t* d_output_shape;
    HIP_CHECK(hipMalloc(&d_input_shape, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_input_strides, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_output_shape, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(d_input_shape, padded_input_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_input_strides, padded_input_strides.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_output_shape, new_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));

    int num_blocks = get_num_blocks(total_elements);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(expand_kernel_impl<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<float>(), output.data<float>(),
            d_input_shape, d_input_strides, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(expand_kernel_impl<double>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<double>(), output.data<double>(),
            d_input_shape, d_input_strides, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(expand_kernel_impl<int32_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<int32_t>(), output.data<int32_t>(),
            d_input_shape, d_input_strides, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(expand_kernel_impl<int64_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<int64_t>(), output.data<int64_t>(),
            d_input_shape, d_input_strides, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(expand_kernel_impl<__half>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            d_input_shape, d_input_strides, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(expand_kernel_impl<hip_bfloat16>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
            reinterpret_cast<hip_bfloat16*>(output.data<BFloat16>()),
            d_input_shape, d_input_strides, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Bool) {
        hipLaunchKernelGGL(expand_kernel_impl<bool>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<bool>(), output.data<bool>(),
            d_input_shape, d_input_strides, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Int8) {
        hipLaunchKernelGGL(expand_kernel_impl<int8_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<int8_t>(), output.data<int8_t>(),
            d_input_shape, d_input_strides, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::UInt8) {
        hipLaunchKernelGGL(expand_kernel_impl<uint8_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<uint8_t>(), output.data<uint8_t>(),
            d_input_shape, d_input_strides, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Int16) {
        hipLaunchKernelGGL(expand_kernel_impl<int16_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<int16_t>(), output.data<int16_t>(),
            d_input_shape, d_input_strides, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Complex64) {
        // Complex64 storage is two Float32s (8 bytes total). Since expand
        // only copies elements, treating each Complex64 as a 64-bit opaque
        // word is safe and avoids needing a dedicated complex kernel.
        hipLaunchKernelGGL(expand_kernel_impl<int64_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const int64_t*>(input.data_ptr()),
            reinterpret_cast<int64_t*>(const_cast<void*>(output.data_ptr())),
            d_input_shape, d_input_strides, d_output_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Complex128) {
        // Complex128 storage is two Float64s (16 bytes total). Expand as
        // pairs of 8-byte words by treating the trailing two floats as
        // independent real-valued dimensions.
        struct __attribute__((aligned(16))) pair64 { int64_t a; int64_t b; };
        hipLaunchKernelGGL(expand_kernel_impl<pair64>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const pair64*>(input.data_ptr()),
            reinterpret_cast<pair64*>(const_cast<void*>(output.data_ptr())),
            d_input_shape, d_input_strides, d_output_shape, ndim, total_elements);
    } else {
        HIP_CHECK(hipFree(d_input_shape));
        HIP_CHECK(hipFree(d_input_strides));
        HIP_CHECK(hipFree(d_output_shape));
        throw std::runtime_error("expand_kernel: unsupported dtype");
    }

    // audit-9 JJ.5: see repeat_kernel above — stream-sync before hipFree.
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_input_shape));
    HIP_CHECK(hipFree(d_input_strides));
    HIP_CHECK(hipFree(d_output_shape));
    HIP_CHECK(hipGetLastError());

    return output;
}

// ============================================================================
// Roll kernel — shift elements along a dimension with wrap-around
// ============================================================================

template<typename T>
__global__ void roll_kernel_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t total_elements,
    int64_t dim_size,
    int64_t shift,
    int64_t inner_size
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elements) return;

    int64_t inner_idx = i % inner_size;
    int64_t dim_idx = (i / inner_size) % dim_size;
    int64_t outer_idx = i / (inner_size * dim_size);

    int64_t src_dim_idx = (dim_idx - shift + dim_size) % dim_size;
    int64_t src_idx = (outer_idx * dim_size + src_dim_idx) * inner_size + inner_idx;

    output[i] = input[src_idx];
}

auto roll_kernel(const Tensor& input, int64_t shift, int64_t dim, hipStream_t stream) -> Tensor {
    auto shape = input.shape();
    auto cont = input.is_contiguous() ? input : input.contiguous();

    auto output = Tensor::empty_uninitialized(
        std::vector<int64_t>(shape.begin(), shape.end()),
        input.dtype(), input.device());

    int64_t total = input.numel();
    if (total == 0) return output;

    // Normalize negative dim (PyTorch semantics) before indexing shape[dim].
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("roll: dim out of range");
    }

    int64_t dim_size = shape[dim];
    // Normalize shift into [0, dim_size): the kernel computes
    // src = (dim_idx - shift + dim_size) % dim_size, which only stays in bounds
    // for shift in [0, dim_size). A negative or large shift would otherwise read
    // out of bounds / wrap incorrectly.
    if (dim_size > 0) {
        shift = ((shift % dim_size) + dim_size) % dim_size;
    }
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < static_cast<int64_t>(shape.size()); ++d) {
        inner_size *= shape[d];
    }

    constexpr int BLOCK = 256;
    int64_t num_blocks = (total + BLOCK - 1) / BLOCK;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(roll_kernel_impl<float>,
            dim3(num_blocks), dim3(BLOCK), 0, stream,
            cont.data<float>(), output.data<float>(),
            total, dim_size, shift, inner_size);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(roll_kernel_impl<double>,
            dim3(num_blocks), dim3(BLOCK), 0, stream,
            cont.data<double>(), output.data<double>(),
            total, dim_size, shift, inner_size);
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(roll_kernel_impl<int32_t>,
            dim3(num_blocks), dim3(BLOCK), 0, stream,
            cont.data<int32_t>(), output.data<int32_t>(),
            total, dim_size, shift, inner_size);
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(roll_kernel_impl<int64_t>,
            dim3(num_blocks), dim3(BLOCK), 0, stream,
            cont.data<int64_t>(), output.data<int64_t>(),
            total, dim_size, shift, inner_size);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(roll_kernel_impl<__half>,
            dim3(num_blocks), dim3(BLOCK), 0, stream,
            reinterpret_cast<const __half*>(cont.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            total, dim_size, shift, inner_size);
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(roll_kernel_impl<uint16_t>,
            dim3(num_blocks), dim3(BLOCK), 0, stream,
            reinterpret_cast<const uint16_t*>(cont.data<BFloat16>()),
            reinterpret_cast<uint16_t*>(output.data<BFloat16>()),
            total, dim_size, shift, inner_size);
    } else {
        throw std::runtime_error("roll_kernel: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

// ============================================================================
// FP8 device helpers — bit-level conversion between FP8 and Float32
// ============================================================================

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t bits) {
    uint32_t sign = (bits >> 7) & 0x1;
    uint32_t exp = (bits >> 3) & 0xF;
    uint32_t mantissa = bits & 0x7;
    uint32_t f_exp, f_mantissa;

    if (exp == 0) {
        if (mantissa == 0) { f_exp = 0; f_mantissa = 0; }
        else {
            int e = -1; uint32_t m = mantissa;
            do { e++; m <<= 1; } while ((m & 0x8) == 0);
            f_exp = 127 - 7 - e; f_mantissa = (m & 0x7) << 20;
        }
    } else if (exp == 0xF && mantissa == 0x7) {
        // Only exp=0xF/mantissa=0x7 is NaN (matches NVIDIA's native E4M3 /
        // src/core/dtype.cpp's FP8_E4M3 reference). Previously matched ANY
        // nonzero mantissa at exp=0xF, incorrectly decoding the valid
        // finite values 256/288/320/352/384/416/448 (mantissa 0x0-0x6) as NaN.
        f_exp = 0xFF; f_mantissa = 0x700000;  // NaN
    } else {
        f_exp = exp - 7 + 127; f_mantissa = mantissa << 20;  // normal
    }

    uint32_t f_bits = (sign << 31) | (f_exp << 23) | f_mantissa;
    float result;
    __builtin_memcpy(&result, &f_bits, sizeof(float));
    return result;
}

__device__ __forceinline__ float fp8_e5m2_to_float(uint8_t bits) {
    uint32_t sign = (bits >> 7) & 0x1;
    uint32_t exp = (bits >> 2) & 0x1F;
    uint32_t mantissa = bits & 0x3;
    uint32_t f_exp, f_mantissa;

    if (exp == 0) {
        if (mantissa == 0) { f_exp = 0; f_mantissa = 0; }
        else {
            int e = -1; uint32_t m = mantissa;
            do { e++; m <<= 1; } while ((m & 0x4) == 0);
            f_exp = 127 - 15 - e; f_mantissa = (m & 0x3) << 21;
        }
    } else if (exp == 0x1F) {
        f_exp = 0xFF; f_mantissa = mantissa << 21;  // NaN / Inf
    } else {
        f_exp = exp - 15 + 127; f_mantissa = mantissa << 21;  // normal
    }

    uint32_t f_bits = (sign << 31) | (f_exp << 23) | f_mantissa;
    float result;
    __builtin_memcpy(&result, &f_bits, sizeof(float));
    return result;
}

__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float f) {
    uint32_t f_bits;
    __builtin_memcpy(&f_bits, &f, sizeof(uint32_t));
    uint32_t sign = (f_bits >> 31) & 0x1;
    uint32_t exp = (f_bits >> 23) & 0xFF;
    uint32_t mantissa = f_bits & 0x7FFFFF;
    uint8_t h_sign = static_cast<uint8_t>(sign);
    uint8_t h_exp, h_mantissa;

    if (exp == 0xFF) { h_exp = 0xF; h_mantissa = 0x7; }          // NaN/Inf -> NaN
    else if (exp == 0) { h_exp = 0; h_mantissa = 0; }            // zero/denormal -> zero
    else {
        int32_t new_exp = static_cast<int32_t>(exp) - 127 + 7;
        // Overflow: clamp to E4M3's true max finite (exp=0xF, mantissa=0x6
        // = 448), not the stale exp=0xE/mantissa=0x7 (=240) bit pattern --
        // quantize_to_fp8 (fp8_scaling.cpp) deliberately scales its input
        // so the tensor's max element lands exactly at fp8_max_value()
        // (448) before casting, so 448.0 itself hits this branch.
        if (new_exp >= 0xF) { h_exp = 0xF; h_mantissa = 0x6; }  // overflow -> max finite
        else if (new_exp <= 0) {
            if (new_exp >= -3) {
                uint32_t m = (mantissa | 0x800000) >> (1 - new_exp);
                h_mantissa = static_cast<uint8_t>((m >> 20) & 0x7); h_exp = 0;
            } else { h_exp = 0; h_mantissa = 0; }                // underflow -> zero
        } else {
            h_exp = static_cast<uint8_t>(new_exp);
            h_mantissa = static_cast<uint8_t>((mantissa >> 20) & 0x7);
        }
    }
    return (h_sign << 7) | (h_exp << 3) | h_mantissa;
}

__device__ __forceinline__ uint8_t float_to_fp8_e5m2(float f) {
    uint32_t f_bits;
    __builtin_memcpy(&f_bits, &f, sizeof(uint32_t));
    uint32_t sign = (f_bits >> 31) & 0x1;
    uint32_t exp = (f_bits >> 23) & 0xFF;
    uint32_t mantissa = f_bits & 0x7FFFFF;
    uint8_t h_sign = static_cast<uint8_t>(sign);
    uint8_t h_exp, h_mantissa;

    if (exp == 0xFF) { h_exp = 0x1F; h_mantissa = mantissa ? 0x3 : 0; }  // NaN/Inf
    else if (exp == 0) { h_exp = 0; h_mantissa = 0; }
    else {
        int32_t new_exp = static_cast<int32_t>(exp) - 127 + 15;
        if (new_exp >= 0x1F) { h_exp = 0x1F; h_mantissa = 0; }  // overflow -> Inf
        else if (new_exp <= 0) {
            if (new_exp >= -2) {
                uint32_t m = (mantissa | 0x800000) >> (1 - new_exp);
                h_mantissa = static_cast<uint8_t>((m >> 21) & 0x3); h_exp = 0;
            } else { h_exp = 0; h_mantissa = 0; }
        } else {
            h_exp = static_cast<uint8_t>(new_exp);
            h_mantissa = static_cast<uint8_t>((mantissa >> 21) & 0x3);
        }
    }
    return (h_sign << 7) | (h_exp << 2) | h_mantissa;
}

// ----------------------------------------------------------------------------
// FNUZ FP8 device helpers (AMD-native format). FNUZ differs from IEEE/OCP FP8:
//   * exponent bias is +1 (E4M3FNUZ bias 8, E5M2FNUZ bias 16)
//   * NO infinities; both Inf and NaN collapse to the single NaN encoding 0x80
//   * negative zero (0x80) is reserved for NaN, so -0 flushes to +0
//   * exp=0xF (E4M3) / exp=0x1F (E5M2) are NORMAL finite values, not NaN/Inf
// These are bit-exact ports of the host FP8_E*FNUZ conversions in
// src/core/dtype.cpp (RNE rounding on the narrowing path), so the device cast
// matches the host/registry-default conversion exactly. Pure integer math —
// no reliance on HIP NaN intrinsics.
// ----------------------------------------------------------------------------

__device__ __forceinline__ float fp8_e4m3fnuz_to_float(uint8_t bits) {
    if (bits == 0x80) {  // single NaN encoding
        uint32_t nan_bits = 0x7FC00000;
        float r; __builtin_memcpy(&r, &nan_bits, sizeof(float)); return r;
    }
    uint32_t sign = (bits >> 7) & 0x1;
    uint32_t exp = (bits >> 3) & 0xF;
    uint32_t mantissa = bits & 0x7;
    uint32_t f_exp, f_mantissa;

    if (exp == 0) {
        if (mantissa == 0) { f_exp = 0; f_mantissa = 0; }
        else {
            int e = -1; uint32_t m = mantissa;
            do { e++; m <<= 1; } while ((m & 0x8) == 0);
            f_exp = 127 - 8 - e; f_mantissa = (m & 0x7) << 20;  // bias 8
        }
    } else {
        f_exp = exp - 8 + 127; f_mantissa = mantissa << 20;  // bias 8, all finite
    }

    uint32_t f_bits = (sign << 31) | (f_exp << 23) | f_mantissa;
    float result;
    __builtin_memcpy(&result, &f_bits, sizeof(float));
    return result;
}

__device__ __forceinline__ float fp8_e5m2fnuz_to_float(uint8_t bits) {
    if (bits == 0x80) {  // single NaN encoding
        uint32_t nan_bits = 0x7FC00000;
        float r; __builtin_memcpy(&r, &nan_bits, sizeof(float)); return r;
    }
    uint32_t sign = (bits >> 7) & 0x1;
    uint32_t exp = (bits >> 2) & 0x1F;
    uint32_t mantissa = bits & 0x3;
    uint32_t f_exp, f_mantissa;

    if (exp == 0) {
        if (mantissa == 0) { f_exp = 0; f_mantissa = 0; }
        else {
            int e = -1; uint32_t m = mantissa;
            do { e++; m <<= 1; } while ((m & 0x4) == 0);
            f_exp = 127 - 16 - e; f_mantissa = (m & 0x3) << 21;  // bias 16
        }
    } else {
        f_exp = exp - 16 + 127; f_mantissa = mantissa << 21;  // bias 16, all finite
    }

    uint32_t f_bits = (sign << 31) | (f_exp << 23) | f_mantissa;
    float result;
    __builtin_memcpy(&result, &f_bits, sizeof(float));
    return result;
}

__device__ __forceinline__ uint8_t float_to_fp8_e4m3fnuz(float f) {
    uint32_t f_bits;
    __builtin_memcpy(&f_bits, &f, sizeof(uint32_t));
    uint32_t sign = (f_bits >> 31) & 0x1;
    uint32_t exp = (f_bits >> 23) & 0xFF;
    uint32_t mantissa = f_bits & 0x7FFFFF;

    if (exp == 0xFF) { return 0x80; }  // Inf/NaN -> single NaN

    uint8_t h_sign = static_cast<uint8_t>(sign);
    uint8_t h_exp, h_mantissa;

    if (exp == 0) { h_exp = 0; h_mantissa = 0; }  // zero / f32 subnormal -> zero
    else {
        int32_t new_exp = static_cast<int32_t>(exp) - 127 + 8;  // bias 8
        if (new_exp >= 0x10) { h_exp = 0xF; h_mantissa = 0x7; }  // saturate (finite)
        else if (new_exp <= 0) {
            if (new_exp >= -3) {
                uint32_t m = (mantissa | 0x800000) >> (1 - new_exp);
                uint32_t kept = (m >> 20) & 0x7;
                uint32_t remainder = m & 0xFFFFF;
                uint32_t halfway = 0x80000;
                if (remainder > halfway || (remainder == halfway && (kept & 1))) kept++;
                h_exp = static_cast<uint8_t>(kept >> 3);
                h_mantissa = static_cast<uint8_t>(kept & 0x7);
            } else { h_exp = 0; h_mantissa = 0; }
        } else {
            uint32_t kept = (mantissa >> 20) & 0x7;
            uint32_t remainder = mantissa & 0xFFFFF;
            uint32_t halfway = 0x80000;
            uint32_t packed = (static_cast<uint32_t>(new_exp) << 3) | kept;
            if (remainder > halfway || (remainder == halfway && (kept & 1))) packed++;
            uint32_t r_exp = packed >> 3;
            uint32_t r_mant = packed & 0x7;
            if (r_exp >= 0x10) { r_exp = 0xF; r_mant = 0x7; }  // carry past max -> saturate
            h_exp = static_cast<uint8_t>(r_exp);
            h_mantissa = static_cast<uint8_t>(r_mant);
        }
    }

    uint8_t out = (h_sign << 7) | (h_exp << 3) | h_mantissa;
    if (out == 0x80) out = 0x00;  // -0 is NaN in FNUZ -> +0
    return out;
}

__device__ __forceinline__ uint8_t float_to_fp8_e5m2fnuz(float f) {
    uint32_t f_bits;
    __builtin_memcpy(&f_bits, &f, sizeof(uint32_t));
    uint32_t sign = (f_bits >> 31) & 0x1;
    uint32_t exp = (f_bits >> 23) & 0xFF;
    uint32_t mantissa = f_bits & 0x7FFFFF;

    if (exp == 0xFF) { return 0x80; }  // Inf/NaN -> single NaN

    uint8_t h_sign = static_cast<uint8_t>(sign);
    uint8_t h_exp, h_mantissa;

    if (exp == 0) { h_exp = 0; h_mantissa = 0; }
    else {
        int32_t new_exp = static_cast<int32_t>(exp) - 127 + 16;  // bias 16
        if (new_exp >= 0x20) { h_exp = 0x1F; h_mantissa = 0x3; }  // saturate (finite)
        else if (new_exp <= 0) {
            if (new_exp >= -2) {
                uint32_t m = (mantissa | 0x800000) >> (1 - new_exp);
                uint32_t kept = (m >> 21) & 0x3;
                uint32_t remainder = m & 0x1FFFFF;
                uint32_t halfway = 0x100000;
                if (remainder > halfway || (remainder == halfway && (kept & 1))) kept++;
                h_exp = static_cast<uint8_t>(kept >> 2);
                h_mantissa = static_cast<uint8_t>(kept & 0x3);
            } else { h_exp = 0; h_mantissa = 0; }
        } else {
            uint32_t kept = (mantissa >> 21) & 0x3;
            uint32_t remainder = mantissa & 0x1FFFFF;
            uint32_t halfway = 0x100000;
            uint32_t packed = (static_cast<uint32_t>(new_exp) << 2) | kept;
            if (remainder > halfway || (remainder == halfway && (kept & 1))) packed++;
            uint32_t r_exp = packed >> 2;
            uint32_t r_mant = packed & 0x3;
            if (r_exp >= 0x20) { r_exp = 0x1F; r_mant = 0x3; }  // carry past max -> saturate
            h_exp = static_cast<uint8_t>(r_exp);
            h_mantissa = static_cast<uint8_t>(r_mant);
        }
    }

    uint8_t out = (h_sign << 7) | (h_exp << 2) | h_mantissa;
    if (out == 0x80) out = 0x00;  // -0 is NaN in FNUZ -> +0
    return out;
}

// ============================================================================
// FP8 cast kernels
// ============================================================================

// Float32 <-> FP8
__global__ void cast_f32_to_fp8_e4m3_kernel(const float* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e4m3(input[idx]); }
}
__global__ void cast_f32_to_fp8_e5m2_kernel(const float* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e5m2(input[idx]); }
}
__global__ void cast_fp8_e4m3_to_f32_kernel(const uint8_t* input, float* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = fp8_e4m3_to_float(input[idx]); }
}
__global__ void cast_fp8_e5m2_to_f32_kernel(const uint8_t* input, float* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = fp8_e5m2_to_float(input[idx]); }
}

// Float64 <-> FP8
__global__ void cast_f64_to_fp8_e4m3_kernel(const double* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e4m3(static_cast<float>(input[idx])); }
}
__global__ void cast_f64_to_fp8_e5m2_kernel(const double* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e5m2(static_cast<float>(input[idx])); }
}
__global__ void cast_fp8_e4m3_to_f64_kernel(const uint8_t* input, double* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = static_cast<double>(fp8_e4m3_to_float(input[idx])); }
}
__global__ void cast_fp8_e5m2_to_f64_kernel(const uint8_t* input, double* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = static_cast<double>(fp8_e5m2_to_float(input[idx])); }
}

// Float16 <-> FP8
__global__ void cast_f16_to_fp8_e4m3_kernel(const __half* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e4m3(tenzor::rocm::safe_h2f(input[idx])); }
}
__global__ void cast_f16_to_fp8_e5m2_kernel(const __half* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e5m2(tenzor::rocm::safe_h2f(input[idx])); }
}
__global__ void cast_fp8_e4m3_to_f16_kernel(const uint8_t* input, __half* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = tenzor::rocm::safe_f2h(fp8_e4m3_to_float(input[idx])); }
}
__global__ void cast_fp8_e5m2_to_f16_kernel(const uint8_t* input, __half* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = tenzor::rocm::safe_f2h(fp8_e5m2_to_float(input[idx])); }
}

// BFloat16 <-> FP8
__global__ void cast_bf16_to_fp8_e4m3_kernel(const hip_bfloat16* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e4m3(static_cast<float>(input[idx])); }
}
__global__ void cast_bf16_to_fp8_e5m2_kernel(const hip_bfloat16* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e5m2(static_cast<float>(input[idx])); }
}
__global__ void cast_fp8_e4m3_to_bf16_kernel(const uint8_t* input, hip_bfloat16* output, int64_t n) {
    // S.10: RNE-round on float32 → bf16 (FP8 dequant can produce values
    // requiring more than 7 mantissa bits to represent without bias).
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = tenzor::rocm::f32_to_bf16_rne(fp8_e4m3_to_float(input[idx])); }
}
__global__ void cast_fp8_e5m2_to_bf16_kernel(const uint8_t* input, hip_bfloat16* output, int64_t n) {
    // S.10: same RNE round as the FP8_E4M3 path above.
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = tenzor::rocm::f32_to_bf16_rne(fp8_e5m2_to_float(input[idx])); }
}

// FP8 <-> FP8 cross-format
__global__ void cast_fp8_e4m3_to_e5m2_kernel(const uint8_t* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e5m2(fp8_e4m3_to_float(input[idx])); }
}
__global__ void cast_fp8_e5m2_to_e4m3_kernel(const uint8_t* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e4m3(fp8_e5m2_to_float(input[idx])); }
}

// Generic FP8 -> any integer/bool type
template<typename To>
__global__ void cast_fp8_e4m3_to_kernel(const uint8_t* input, To* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = static_cast<To>(fp8_e4m3_to_float(input[idx])); }
}
template<typename To>
__global__ void cast_fp8_e5m2_to_kernel(const uint8_t* input, To* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = static_cast<To>(fp8_e5m2_to_float(input[idx])); }
}

// Generic any type -> FP8 (via float)
template<typename From>
__global__ void cast_to_fp8_e4m3_kernel(const From* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e4m3(static_cast<float>(input[idx])); }
}
template<typename From>
__global__ void cast_to_fp8_e5m2_kernel(const From* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e5m2(static_cast<float>(input[idx])); }
}

// ----------------------------------------------------------------------------
// FNUZ FP8 cast kernels (mirror the IEEE FP8 kernels above)
// ----------------------------------------------------------------------------

// Float32 <-> FP8 FNUZ
__global__ void cast_f32_to_fp8_e4m3fnuz_kernel(const float* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e4m3fnuz(input[idx]); }
}
__global__ void cast_f32_to_fp8_e5m2fnuz_kernel(const float* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e5m2fnuz(input[idx]); }
}
__global__ void cast_fp8_e4m3fnuz_to_f32_kernel(const uint8_t* input, float* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = fp8_e4m3fnuz_to_float(input[idx]); }
}
__global__ void cast_fp8_e5m2fnuz_to_f32_kernel(const uint8_t* input, float* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = fp8_e5m2fnuz_to_float(input[idx]); }
}

// Float64 <-> FP8 FNUZ
__global__ void cast_f64_to_fp8_e4m3fnuz_kernel(const double* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e4m3fnuz(static_cast<float>(input[idx])); }
}
__global__ void cast_f64_to_fp8_e5m2fnuz_kernel(const double* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e5m2fnuz(static_cast<float>(input[idx])); }
}
__global__ void cast_fp8_e4m3fnuz_to_f64_kernel(const uint8_t* input, double* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = static_cast<double>(fp8_e4m3fnuz_to_float(input[idx])); }
}
__global__ void cast_fp8_e5m2fnuz_to_f64_kernel(const uint8_t* input, double* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = static_cast<double>(fp8_e5m2fnuz_to_float(input[idx])); }
}

// Float16 <-> FP8 FNUZ
__global__ void cast_f16_to_fp8_e4m3fnuz_kernel(const __half* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e4m3fnuz(tenzor::rocm::safe_h2f(input[idx])); }
}
__global__ void cast_f16_to_fp8_e5m2fnuz_kernel(const __half* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e5m2fnuz(tenzor::rocm::safe_h2f(input[idx])); }
}
__global__ void cast_fp8_e4m3fnuz_to_f16_kernel(const uint8_t* input, __half* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = tenzor::rocm::safe_f2h(fp8_e4m3fnuz_to_float(input[idx])); }
}
__global__ void cast_fp8_e5m2fnuz_to_f16_kernel(const uint8_t* input, __half* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = tenzor::rocm::safe_f2h(fp8_e5m2fnuz_to_float(input[idx])); }
}

// BFloat16 <-> FP8 FNUZ
__global__ void cast_bf16_to_fp8_e4m3fnuz_kernel(const hip_bfloat16* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e4m3fnuz(static_cast<float>(input[idx])); }
}
__global__ void cast_bf16_to_fp8_e5m2fnuz_kernel(const hip_bfloat16* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e5m2fnuz(static_cast<float>(input[idx])); }
}
__global__ void cast_fp8_e4m3fnuz_to_bf16_kernel(const uint8_t* input, hip_bfloat16* output, int64_t n) {
    // S.10: RNE-round on float32 → bf16 (FNUZ dequant can produce values
    // requiring more than 7 mantissa bits to represent without bias).
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = tenzor::rocm::f32_to_bf16_rne(fp8_e4m3fnuz_to_float(input[idx])); }
}
__global__ void cast_fp8_e5m2fnuz_to_bf16_kernel(const uint8_t* input, hip_bfloat16* output, int64_t n) {
    // S.10: same RNE round as the FP8_E4M3FNUZ path above.
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = tenzor::rocm::f32_to_bf16_rne(fp8_e5m2fnuz_to_float(input[idx])); }
}

// FP8 FNUZ <-> FP8 cross-format (all four directions, via float32)
__global__ void cast_fp8_e4m3fnuz_to_e5m2fnuz_kernel(const uint8_t* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e5m2fnuz(fp8_e4m3fnuz_to_float(input[idx])); }
}
__global__ void cast_fp8_e5m2fnuz_to_e4m3fnuz_kernel(const uint8_t* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e4m3fnuz(fp8_e5m2fnuz_to_float(input[idx])); }
}
__global__ void cast_fp8_e4m3fnuz_to_e4m3_kernel(const uint8_t* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e4m3(fp8_e4m3fnuz_to_float(input[idx])); }
}
__global__ void cast_fp8_e4m3fnuz_to_e5m2_kernel(const uint8_t* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e5m2(fp8_e4m3fnuz_to_float(input[idx])); }
}
__global__ void cast_fp8_e5m2fnuz_to_e4m3_kernel(const uint8_t* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e4m3(fp8_e5m2fnuz_to_float(input[idx])); }
}
__global__ void cast_fp8_e5m2fnuz_to_e5m2_kernel(const uint8_t* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e5m2(fp8_e5m2fnuz_to_float(input[idx])); }
}
__global__ void cast_fp8_e4m3_to_e4m3fnuz_kernel(const uint8_t* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e4m3fnuz(fp8_e4m3_to_float(input[idx])); }
}
__global__ void cast_fp8_e4m3_to_e5m2fnuz_kernel(const uint8_t* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e5m2fnuz(fp8_e4m3_to_float(input[idx])); }
}
__global__ void cast_fp8_e5m2_to_e4m3fnuz_kernel(const uint8_t* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e4m3fnuz(fp8_e5m2_to_float(input[idx])); }
}
__global__ void cast_fp8_e5m2_to_e5m2fnuz_kernel(const uint8_t* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e5m2fnuz(fp8_e5m2_to_float(input[idx])); }
}

// Generic FP8 FNUZ -> any integer/bool type
template<typename To>
__global__ void cast_fp8_e4m3fnuz_to_kernel(const uint8_t* input, To* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = static_cast<To>(fp8_e4m3fnuz_to_float(input[idx])); }
}
template<typename To>
__global__ void cast_fp8_e5m2fnuz_to_kernel(const uint8_t* input, To* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = static_cast<To>(fp8_e5m2fnuz_to_float(input[idx])); }
}

// Generic any type -> FP8 FNUZ (via float)
template<typename From>
__global__ void cast_to_fp8_e4m3fnuz_kernel(const From* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e4m3fnuz(static_cast<float>(input[idx])); }
}
template<typename From>
__global__ void cast_to_fp8_e5m2fnuz_kernel(const From* input, uint8_t* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) { output[idx] = float_to_fp8_e5m2fnuz(static_cast<float>(input[idx])); }
}

// ============================================================================
// Cast kernel — dtype conversion
// ============================================================================

template<typename SrcT, typename DstT>
__device__ __forceinline__ DstT cast_saturating(SrcT val) {
    if constexpr (std::is_floating_point_v<SrcT> && std::is_integral_v<DstT> &&
                  !std::is_same_v<DstT, bool>) {
        // Saturating conversion (NaN->0, +Inf/overflow->max, -Inf/-overflow->
        // min), matching CPU/CUDA/OneAPI/Vulkan's convention instead of a
        // bare static_cast (UB for out-of-range/NaN input; previously
        // produced 32-bit-wraparound garbage for f32->i64 on this backend
        // and an imprecise upper bound for f64->i64).
        if (isnan(val)) return DstT{0};
        if (val >= static_cast<SrcT>(std::numeric_limits<DstT>::max())) {
            return std::numeric_limits<DstT>::max();
        }
        if (val <= static_cast<SrcT>(std::numeric_limits<DstT>::min())) {
            return std::numeric_limits<DstT>::min();
        }
        return static_cast<DstT>(val);
    } else {
        return static_cast<DstT>(val);
    }
}

template<typename SrcT, typename DstT>
__global__ void cast_kernel_impl(const SrcT* input, DstT* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        output[idx] = cast_saturating<SrcT, DstT>(input[idx]);
    }
}

// Dedicated any-source -> BFloat16 cast using round-to-nearest-even, instead of
// the value-truncating hip_bfloat16(float) constructor. Matches the FP8->bf16
// cast kernels in this file and the CPU/CUDA reference, which round.
template<typename SrcT>
__global__ void cast_to_bf16_kernel(const SrcT* input, hip_bfloat16* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        output[idx] = tenzor::rocm::f32_to_bf16_rne(static_cast<float>(input[idx]));
    }
}

// Specialization for __half source
// For floating-point targets we route NaN/±Inf through explicit Float32 bit
// patterns so the IEEE special values survive round-trips regardless of how
// HIP's __half2float decides to canonicalise them on a given rocm build.
template<typename DstT>
__global__ void cast_from_f16_kernel(const __half* input, DstT* output, int64_t n) {
    HIP_GRID_STRIDE_LOOP(idx, n) {
        unsigned short bits = *reinterpret_cast<const unsigned short*>(&input[idx]);
        unsigned short exp = (bits >> 10) & 0x1Fu;
        unsigned short mant = bits & 0x3FFu;
        if constexpr (std::is_floating_point_v<DstT>) {
            if (exp == 0x1Fu) {
                if (mant != 0u) {
                    output[idx] = static_cast<DstT>(nanf(""));
                } else {
                    float inf_f32 = (bits & 0x8000u) ? -INFINITY : INFINITY;
                    output[idx] = static_cast<DstT>(inf_f32);
                }
                continue;
            }
        }
        output[idx] = static_cast<DstT>(tenzor::rocm::safe_h2f(input[idx]));
    }
}

// Specialization for __half target
// Saturating conversion: clamps finite values to ±65504 to prevent overflow
// producing Inf during later arithmetic, while preserving input NaN and Inf
// so round-trips through F16 don't silently lose them.
//
// NaN / ±Inf are emitted as explicit bit patterns because HIP's __float2half
// on some ROCm builds does not forward a Float32 NaN through to a Float16
// NaN — the payload can be lost and the result falls back to a finite value.
// We also classify via the raw F32 bit pattern rather than isnan/isinf so
// fast-math compile settings can't optimise the check away.
template<typename SrcT>
__global__ void cast_to_f16_kernel(const SrcT* input, __half* output, int64_t n) {
    constexpr float kHalfMax = 65504.0f;
    HIP_GRID_STRIDE_LOOP(idx, n) {
        float val = static_cast<float>(input[idx]);
        unsigned int vb = __float_as_uint(val);
        unsigned int exp = (vb >> 23) & 0xFFu;
        unsigned int mant = vb & 0x7FFFFFu;
        unsigned int sign16 = (vb >> 16) & 0x8000u;
        unsigned short bits16;
        if (exp == 0xFFu) {
            // NaN or Inf in Float32
            if (mant != 0u) {
                // Quiet NaN in Float16.
                bits16 = 0x7E00u;
            } else {
                // ±Inf.
                bits16 = static_cast<unsigned short>(sign16 | 0x7C00u);
            }
            __half h;
            *reinterpret_cast<unsigned short*>(&h) = bits16;
            output[idx] = h;
        } else {
            val = fminf(fmaxf(val, -kHalfMax), kHalfMax);
            output[idx] = tenzor::rocm::safe_f2h(val);
        }
    }
}

template<typename SrcT>
static Tensor cast_from_standard(const Tensor& input, DType target_dtype, int64_t n,
                                  int num_blocks, int block_size, hipStream_t stream) {
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, target_dtype, input.device());
    const SrcT* src = input.data<SrcT>();

    #define CAST_CASE(DTYPE, CppType) \
        case DTYPE: \
            hipLaunchKernelGGL((cast_kernel_impl<SrcT, CppType>), \
                dim3(num_blocks), dim3(block_size), 0, stream, \
                src, result.data<CppType>(), n); \
            break

    switch (target_dtype) {
        CAST_CASE(DType::Float32, float);
        CAST_CASE(DType::Float64, double);
        CAST_CASE(DType::Int8, int8_t);
        CAST_CASE(DType::Int16, int16_t);
        CAST_CASE(DType::Int32, int32_t);
        CAST_CASE(DType::Int64, int64_t);
        CAST_CASE(DType::UInt8, uint8_t);
        CAST_CASE(DType::UInt16, uint16_t);
        CAST_CASE(DType::UInt32, uint32_t);
        CAST_CASE(DType::UInt64, uint64_t);
        CAST_CASE(DType::Bool, bool);
        case DType::Float16:
            hipLaunchKernelGGL((cast_to_f16_kernel<SrcT>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                src, reinterpret_cast<__half*>(result.data<Float16>()), n);
            break;
        case DType::BFloat16:
            hipLaunchKernelGGL((cast_to_bf16_kernel<SrcT>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                src, reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
            break;
        case DType::FP8_E4M3:
            hipLaunchKernelGGL((cast_to_fp8_e4m3_kernel<SrcT>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                src, result.data<uint8_t>(), n);
            break;
        case DType::FP8_E5M2:
            hipLaunchKernelGGL((cast_to_fp8_e5m2_kernel<SrcT>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                src, result.data<uint8_t>(), n);
            break;
        case DType::FP8_E4M3FNUZ:
            hipLaunchKernelGGL((cast_to_fp8_e4m3fnuz_kernel<SrcT>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                src, result.data<uint8_t>(), n);
            break;
        case DType::FP8_E5M2FNUZ:
            hipLaunchKernelGGL((cast_to_fp8_e5m2fnuz_kernel<SrcT>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                src, result.data<uint8_t>(), n);
            break;
        default:
            throw std::runtime_error("cast: unsupported target dtype");
    }
    #undef CAST_CASE

    HIP_CHECK(hipGetLastError());
    return result;
}

// Complex64/Complex128 conversions: the storage is interleaved
// (re, im) pairs of float/double, so the generic cast_kernel_impl
// doesn't apply. Handle the common Float↔Complex pairs directly.
__global__ void cast_f32_to_c64_kernel_rocm(const float* in, float* out, int64_t n) {
    HIP_GRID_STRIDE_LOOP(i, n) {
        out[2 * i]     = in[i];
        out[2 * i + 1] = 0.0f;
    }
}
__global__ void cast_f64_to_c128_kernel_rocm(const double* in, double* out, int64_t n) {
    HIP_GRID_STRIDE_LOOP(i, n) {
        out[2 * i]     = in[i];
        out[2 * i + 1] = 0.0;
    }
}
__global__ void cast_c64_to_f32_kernel_rocm(const float* in, float* out, int64_t n) {
    HIP_GRID_STRIDE_LOOP(i, n) { out[i] = in[2 * i]; }
}
__global__ void cast_c128_to_f64_kernel_rocm(const double* in, double* out, int64_t n) {
    HIP_GRID_STRIDE_LOOP(i, n) { out[i] = in[2 * i]; }
}
__global__ void cast_c64_to_c128_kernel_rocm(const float* in, double* out, int64_t n) {
    HIP_GRID_STRIDE_LOOP(i, n) {
        out[2 * i]     = static_cast<double>(in[2 * i]);
        out[2 * i + 1] = static_cast<double>(in[2 * i + 1]);
    }
}
__global__ void cast_c128_to_c64_kernel_rocm(const double* in, float* out, int64_t n) {
    HIP_GRID_STRIDE_LOOP(i, n) {
        out[2 * i]     = static_cast<float>(in[2 * i]);
        out[2 * i + 1] = static_cast<float>(in[2 * i + 1]);
    }
}

auto cast_kernel(const Tensor& input, DType target_dtype, hipStream_t stream) -> Tensor {
    if (input.dtype() == target_dtype) {
        return input;
    }

    int64_t n = input.numel();
    // Empty-tensor fast path: HIP rejects zero-grid launches with
    // "invalid configuration argument". Just allocate an empty result of
    // the target dtype and return.
    if (n == 0) {
        std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
        return Tensor(shape, target_dtype, input.device());
    }
    int num_blocks = get_num_blocks(n);
    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());

    DType src_dtype = input.dtype();

    // Generic real<->complex cross product. The direct-pair block below only
    // special-cases Float32<->Complex64, Float64<->Complex128, and
    // Complex64<->Complex128 -- every other real dtype (Int8/16/32/64,
    // UInt8/16/32/64, Bool, Float16) into/out of a complex dtype previously
    // fell through to the generic per-type dispatch and threw, unlike
    // CPU/CUDA/OneAPI/Vulkan which all support the full cross product (drop
    // imaginary part on complex->real, zero-fill imaginary on real->complex).
    // Route through Float32/Float64 as a pivot -- matches the BFloat16
    // two-hop pattern already used elsewhere in this function -- so the
    // recursive call lands on one of the direct pairs below.
    {
        bool src_is_complex = (src_dtype == DType::Complex64 || src_dtype == DType::Complex128);
        bool tgt_is_complex = (target_dtype == DType::Complex64 || target_dtype == DType::Complex128);
        if (!src_is_complex && tgt_is_complex) {
            DType via = (target_dtype == DType::Complex64) ? DType::Float32 : DType::Float64;
            if (src_dtype != via) {
                return cast_kernel(cast_kernel(input, via, stream), target_dtype, stream);
            }
        } else if (src_is_complex && !tgt_is_complex) {
            DType via = (src_dtype == DType::Complex64) ? DType::Float32 : DType::Float64;
            if (target_dtype != via) {
                return cast_kernel(cast_kernel(input, via, stream), target_dtype, stream);
            }
        }
    }

    // Complex conversions: short-circuit before the generic per-type
    // dispatch. Complex64/128 storage is interleaved pairs of the
    // underlying real type, so we can't just treat it as a single
    // scalar and cast_kernel_impl would corrupt the layout. Same set
    // of paths as the CUDA and OneAPI cast kernels.
    if (src_dtype == DType::Float32 && target_dtype == DType::Complex64) {
        Tensor result(shape, target_dtype, input.device());
        hipLaunchKernelGGL(cast_f32_to_c64_kernel_rocm,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<float>(),
            reinterpret_cast<float*>(const_cast<void*>(result.data_ptr())), n);
        HIP_CHECK(hipGetLastError());
        return result;
    }
    if (src_dtype == DType::Float64 && target_dtype == DType::Complex128) {
        Tensor result(shape, target_dtype, input.device());
        hipLaunchKernelGGL(cast_f64_to_c128_kernel_rocm,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<double>(),
            reinterpret_cast<double*>(const_cast<void*>(result.data_ptr())), n);
        HIP_CHECK(hipGetLastError());
        return result;
    }
    if (src_dtype == DType::Complex64 && target_dtype == DType::Float32) {
        Tensor result(shape, target_dtype, input.device());
        hipLaunchKernelGGL(cast_c64_to_f32_kernel_rocm,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const float*>(input.data_ptr()),
            result.data<float>(), n);
        HIP_CHECK(hipGetLastError());
        return result;
    }
    if (src_dtype == DType::Complex128 && target_dtype == DType::Float64) {
        Tensor result(shape, target_dtype, input.device());
        hipLaunchKernelGGL(cast_c128_to_f64_kernel_rocm,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const double*>(input.data_ptr()),
            result.data<double>(), n);
        HIP_CHECK(hipGetLastError());
        return result;
    }
    if (src_dtype == DType::Complex64 && target_dtype == DType::Complex128) {
        Tensor result(shape, target_dtype, input.device());
        hipLaunchKernelGGL(cast_c64_to_c128_kernel_rocm,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const float*>(input.data_ptr()),
            reinterpret_cast<double*>(const_cast<void*>(result.data_ptr())), n);
        HIP_CHECK(hipGetLastError());
        return result;
    }
    if (src_dtype == DType::Complex128 && target_dtype == DType::Complex64) {
        Tensor result(shape, target_dtype, input.device());
        hipLaunchKernelGGL(cast_c128_to_c64_kernel_rocm,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const double*>(input.data_ptr()),
            reinterpret_cast<float*>(const_cast<void*>(result.data_ptr())), n);
        HIP_CHECK(hipGetLastError());
        return result;
    }

    // Float16 source
    if (src_dtype == DType::Float16) {
        Tensor result(shape, target_dtype, input.device());
        const __half* src = reinterpret_cast<const __half*>(input.data<Float16>());

        switch (target_dtype) {
            case DType::Float32:
                hipLaunchKernelGGL((cast_from_f16_kernel<float>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<float>(), n);
                break;
            case DType::Float64:
                hipLaunchKernelGGL((cast_from_f16_kernel<double>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<double>(), n);
                break;
            case DType::Int32:
                hipLaunchKernelGGL((cast_from_f16_kernel<int32_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<int32_t>(), n);
                break;
            case DType::Int64:
                hipLaunchKernelGGL((cast_from_f16_kernel<int64_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<int64_t>(), n);
                break;
            case DType::Int8:
                hipLaunchKernelGGL((cast_from_f16_kernel<int8_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<int8_t>(), n);
                break;
            case DType::UInt8:
                hipLaunchKernelGGL((cast_from_f16_kernel<uint8_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::Bool:
                hipLaunchKernelGGL((cast_from_f16_kernel<bool>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<bool>(), n);
                break;
            case DType::FP8_E4M3:
                hipLaunchKernelGGL(cast_f16_to_fp8_e4m3_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::FP8_E5M2:
                hipLaunchKernelGGL(cast_f16_to_fp8_e5m2_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::FP8_E4M3FNUZ:
                hipLaunchKernelGGL(cast_f16_to_fp8_e4m3fnuz_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::FP8_E5M2FNUZ:
                hipLaunchKernelGGL(cast_f16_to_fp8_e5m2fnuz_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::BFloat16:
                // F16 and BF16 have disjoint precision/range; route through F32
                // using the cast_from_f16_kernel<float> → cast_kernel_impl<float, bf16>
                // pipeline rather than duplicating a direct kernel.
                return cast_kernel(cast_kernel(input, DType::Float32, stream),
                                   DType::BFloat16, stream);
            default:
                throw std::runtime_error("cast: unsupported target dtype for Float16 source");
        }
        HIP_CHECK(hipGetLastError());
        return result;
    }

    // BFloat16 source: upcast to Float32 first, then cast to target
    if (src_dtype == DType::BFloat16) {
        if (target_dtype == DType::Float32) {
            // Direct BF16 → F32
            Tensor result(shape, DType::Float32, input.device());
            hipLaunchKernelGGL((cast_kernel_impl<hip_bfloat16, float>),
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
                result.data<float>(), n);
            HIP_CHECK(hipGetLastError());
            return result;
        }
        if (target_dtype == DType::FP8_E4M3) {
            Tensor result(shape, target_dtype, input.device());
            hipLaunchKernelGGL(cast_bf16_to_fp8_e4m3_kernel,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
                result.data<uint8_t>(), n);
            HIP_CHECK(hipGetLastError());
            return result;
        }
        if (target_dtype == DType::FP8_E5M2) {
            Tensor result(shape, target_dtype, input.device());
            hipLaunchKernelGGL(cast_bf16_to_fp8_e5m2_kernel,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
                result.data<uint8_t>(), n);
            HIP_CHECK(hipGetLastError());
            return result;
        }
        if (target_dtype == DType::FP8_E4M3FNUZ) {
            Tensor result(shape, target_dtype, input.device());
            hipLaunchKernelGGL(cast_bf16_to_fp8_e4m3fnuz_kernel,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
                result.data<uint8_t>(), n);
            HIP_CHECK(hipGetLastError());
            return result;
        }
        if (target_dtype == DType::FP8_E5M2FNUZ) {
            Tensor result(shape, target_dtype, input.device());
            hipLaunchKernelGGL(cast_bf16_to_fp8_e5m2fnuz_kernel,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                reinterpret_cast<const hip_bfloat16*>(input.data<BFloat16>()),
                result.data<uint8_t>(), n);
            HIP_CHECK(hipGetLastError());
            return result;
        }
        // For other targets, go through Float32
        auto f32 = cast_kernel(input, DType::Float32, stream);
        return cast_kernel(f32, target_dtype, stream);
    }

    // ---- FP8_E4M3 source ----
    if (src_dtype == DType::FP8_E4M3) {
        Tensor result(shape, target_dtype, input.device());
        const uint8_t* src = input.data<uint8_t>();
        switch (target_dtype) {
            case DType::Float32:
                hipLaunchKernelGGL(cast_fp8_e4m3_to_f32_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<float>(), n);
                break;
            case DType::Float64:
                hipLaunchKernelGGL(cast_fp8_e4m3_to_f64_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<double>(), n);
                break;
            case DType::Float16:
                hipLaunchKernelGGL(cast_fp8_e4m3_to_f16_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, reinterpret_cast<__half*>(result.data<Float16>()), n);
                break;
            case DType::BFloat16:
                hipLaunchKernelGGL(cast_fp8_e4m3_to_bf16_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
                break;
            case DType::FP8_E4M3:
                break;  // same type, already handled above
            case DType::FP8_E5M2:
                hipLaunchKernelGGL(cast_fp8_e4m3_to_e5m2_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::FP8_E4M3FNUZ:
                hipLaunchKernelGGL(cast_fp8_e4m3_to_e4m3fnuz_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::FP8_E5M2FNUZ:
                hipLaunchKernelGGL(cast_fp8_e4m3_to_e5m2fnuz_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::Int8:
                hipLaunchKernelGGL((cast_fp8_e4m3_to_kernel<int8_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<int8_t>(), n);
                break;
            case DType::Int32:
                hipLaunchKernelGGL((cast_fp8_e4m3_to_kernel<int32_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<int32_t>(), n);
                break;
            case DType::Int64:
                hipLaunchKernelGGL((cast_fp8_e4m3_to_kernel<int64_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<int64_t>(), n);
                break;
            case DType::UInt8:
                hipLaunchKernelGGL((cast_fp8_e4m3_to_kernel<uint8_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::Bool:
                hipLaunchKernelGGL((cast_fp8_e4m3_to_kernel<bool>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<bool>(), n);
                break;
            default:
                throw std::runtime_error("cast: unsupported target dtype for FP8_E4M3 source");
        }
        HIP_CHECK(hipGetLastError());
        return result;
    }

    // ---- FP8_E5M2 source ----
    if (src_dtype == DType::FP8_E5M2) {
        Tensor result(shape, target_dtype, input.device());
        const uint8_t* src = input.data<uint8_t>();
        switch (target_dtype) {
            case DType::Float32:
                hipLaunchKernelGGL(cast_fp8_e5m2_to_f32_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<float>(), n);
                break;
            case DType::Float64:
                hipLaunchKernelGGL(cast_fp8_e5m2_to_f64_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<double>(), n);
                break;
            case DType::Float16:
                hipLaunchKernelGGL(cast_fp8_e5m2_to_f16_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, reinterpret_cast<__half*>(result.data<Float16>()), n);
                break;
            case DType::BFloat16:
                hipLaunchKernelGGL(cast_fp8_e5m2_to_bf16_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
                break;
            case DType::FP8_E4M3:
                hipLaunchKernelGGL(cast_fp8_e5m2_to_e4m3_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::FP8_E5M2:
                break;  // same type, already handled above
            case DType::FP8_E4M3FNUZ:
                hipLaunchKernelGGL(cast_fp8_e5m2_to_e4m3fnuz_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::FP8_E5M2FNUZ:
                hipLaunchKernelGGL(cast_fp8_e5m2_to_e5m2fnuz_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::Int8:
                hipLaunchKernelGGL((cast_fp8_e5m2_to_kernel<int8_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<int8_t>(), n);
                break;
            case DType::Int32:
                hipLaunchKernelGGL((cast_fp8_e5m2_to_kernel<int32_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<int32_t>(), n);
                break;
            case DType::Int64:
                hipLaunchKernelGGL((cast_fp8_e5m2_to_kernel<int64_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<int64_t>(), n);
                break;
            case DType::UInt8:
                hipLaunchKernelGGL((cast_fp8_e5m2_to_kernel<uint8_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::Bool:
                hipLaunchKernelGGL((cast_fp8_e5m2_to_kernel<bool>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<bool>(), n);
                break;
            default:
                throw std::runtime_error("cast: unsupported target dtype for FP8_E5M2 source");
        }
        HIP_CHECK(hipGetLastError());
        return result;
    }

    // ---- FP8_E4M3FNUZ source ----
    if (src_dtype == DType::FP8_E4M3FNUZ) {
        Tensor result(shape, target_dtype, input.device());
        const uint8_t* src = input.data<uint8_t>();
        switch (target_dtype) {
            case DType::Float32:
                hipLaunchKernelGGL(cast_fp8_e4m3fnuz_to_f32_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<float>(), n);
                break;
            case DType::Float64:
                hipLaunchKernelGGL(cast_fp8_e4m3fnuz_to_f64_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<double>(), n);
                break;
            case DType::Float16:
                hipLaunchKernelGGL(cast_fp8_e4m3fnuz_to_f16_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, reinterpret_cast<__half*>(result.data<Float16>()), n);
                break;
            case DType::BFloat16:
                hipLaunchKernelGGL(cast_fp8_e4m3fnuz_to_bf16_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
                break;
            case DType::FP8_E4M3FNUZ:
                break;  // same type, already handled above
            case DType::FP8_E5M2FNUZ:
                hipLaunchKernelGGL(cast_fp8_e4m3fnuz_to_e5m2fnuz_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::FP8_E4M3:
                hipLaunchKernelGGL(cast_fp8_e4m3fnuz_to_e4m3_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::FP8_E5M2:
                hipLaunchKernelGGL(cast_fp8_e4m3fnuz_to_e5m2_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::Int8:
                hipLaunchKernelGGL((cast_fp8_e4m3fnuz_to_kernel<int8_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<int8_t>(), n);
                break;
            case DType::Int32:
                hipLaunchKernelGGL((cast_fp8_e4m3fnuz_to_kernel<int32_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<int32_t>(), n);
                break;
            case DType::Int64:
                hipLaunchKernelGGL((cast_fp8_e4m3fnuz_to_kernel<int64_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<int64_t>(), n);
                break;
            case DType::UInt8:
                hipLaunchKernelGGL((cast_fp8_e4m3fnuz_to_kernel<uint8_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::Bool:
                hipLaunchKernelGGL((cast_fp8_e4m3fnuz_to_kernel<bool>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<bool>(), n);
                break;
            default:
                throw std::runtime_error("cast: unsupported target dtype for FP8_E4M3FNUZ source");
        }
        HIP_CHECK(hipGetLastError());
        return result;
    }

    // ---- FP8_E5M2FNUZ source ----
    if (src_dtype == DType::FP8_E5M2FNUZ) {
        Tensor result(shape, target_dtype, input.device());
        const uint8_t* src = input.data<uint8_t>();
        switch (target_dtype) {
            case DType::Float32:
                hipLaunchKernelGGL(cast_fp8_e5m2fnuz_to_f32_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<float>(), n);
                break;
            case DType::Float64:
                hipLaunchKernelGGL(cast_fp8_e5m2fnuz_to_f64_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<double>(), n);
                break;
            case DType::Float16:
                hipLaunchKernelGGL(cast_fp8_e5m2fnuz_to_f16_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, reinterpret_cast<__half*>(result.data<Float16>()), n);
                break;
            case DType::BFloat16:
                hipLaunchKernelGGL(cast_fp8_e5m2fnuz_to_bf16_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, reinterpret_cast<hip_bfloat16*>(result.data<BFloat16>()), n);
                break;
            case DType::FP8_E4M3FNUZ:
                hipLaunchKernelGGL(cast_fp8_e5m2fnuz_to_e4m3fnuz_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::FP8_E5M2FNUZ:
                break;  // same type, already handled above
            case DType::FP8_E4M3:
                hipLaunchKernelGGL(cast_fp8_e5m2fnuz_to_e4m3_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::FP8_E5M2:
                hipLaunchKernelGGL(cast_fp8_e5m2fnuz_to_e5m2_kernel,
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::Int8:
                hipLaunchKernelGGL((cast_fp8_e5m2fnuz_to_kernel<int8_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<int8_t>(), n);
                break;
            case DType::Int32:
                hipLaunchKernelGGL((cast_fp8_e5m2fnuz_to_kernel<int32_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<int32_t>(), n);
                break;
            case DType::Int64:
                hipLaunchKernelGGL((cast_fp8_e5m2fnuz_to_kernel<int64_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<int64_t>(), n);
                break;
            case DType::UInt8:
                hipLaunchKernelGGL((cast_fp8_e5m2fnuz_to_kernel<uint8_t>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<uint8_t>(), n);
                break;
            case DType::Bool:
                hipLaunchKernelGGL((cast_fp8_e5m2fnuz_to_kernel<bool>),
                    dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                    src, result.data<bool>(), n);
                break;
            default:
                throw std::runtime_error("cast: unsupported target dtype for FP8_E5M2FNUZ source");
        }
        HIP_CHECK(hipGetLastError());
        return result;
    }

    // Standard source types
    Tensor result;
    switch (src_dtype) {
        case DType::Float32:
            result = cast_from_standard<float>(input, target_dtype, n, num_blocks, BLOCK_SIZE, stream); break;
        case DType::Float64:
            result = cast_from_standard<double>(input, target_dtype, n, num_blocks, BLOCK_SIZE, stream); break;
        case DType::Int8:
            result = cast_from_standard<int8_t>(input, target_dtype, n, num_blocks, BLOCK_SIZE, stream); break;
        case DType::Int16:
            result = cast_from_standard<int16_t>(input, target_dtype, n, num_blocks, BLOCK_SIZE, stream); break;
        case DType::Int32:
            result = cast_from_standard<int32_t>(input, target_dtype, n, num_blocks, BLOCK_SIZE, stream); break;
        case DType::Int64:
            result = cast_from_standard<int64_t>(input, target_dtype, n, num_blocks, BLOCK_SIZE, stream); break;
        case DType::UInt8:
            result = cast_from_standard<uint8_t>(input, target_dtype, n, num_blocks, BLOCK_SIZE, stream); break;
        case DType::UInt16:
            result = cast_from_standard<uint16_t>(input, target_dtype, n, num_blocks, BLOCK_SIZE, stream); break;
        case DType::UInt32:
            result = cast_from_standard<uint32_t>(input, target_dtype, n, num_blocks, BLOCK_SIZE, stream); break;
        case DType::UInt64:
            result = cast_from_standard<uint64_t>(input, target_dtype, n, num_blocks, BLOCK_SIZE, stream); break;
        case DType::Bool:
            result = cast_from_standard<bool>(input, target_dtype, n, num_blocks, BLOCK_SIZE, stream); break;
        default:
            throw std::runtime_error("cast: unsupported source dtype");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// StridedFill kernel — fill non-contiguous tensor with a value
// ============================================================================

template<typename T>
__global__ void strided_fill_kernel_device(
    T* base, T value, int64_t n,
    const int64_t* shape, const int64_t* strides, int32_t ndims) {
    HIP_GRID_STRIDE_LOOP(flat_idx, n) {
        int64_t remaining = flat_idx;
        int64_t offset = 0;
        for (int32_t d = ndims - 1; d >= 0; --d) {
            int64_t coord = remaining % shape[d];
            remaining /= shape[d];
            offset += coord * strides[d];
        }
        base[offset] = value;
    }
}

auto strided_fill_kernel(Tensor& self, double value, hipStream_t stream) -> void {
    int64_t n = self.numel();
    if (n == 0) return;

    auto ndims = self.ndim();
    auto shp = self.shape();
    auto str = self.strides();

    // Copy shape and strides to device
    std::vector<int64_t> meta(ndims * 2);
    for (int64_t d = 0; d < ndims; ++d) {
        meta[d] = shp[d];
        meta[ndims + d] = str[d];
    }
    int64_t* d_meta = nullptr;
    HIP_CHECK(hipMalloc(&d_meta, meta.size() * sizeof(int64_t)));
    HIP_CHECK(hipMemcpyAsync(d_meta, meta.data(), meta.size() * sizeof(int64_t),
                             hipMemcpyHostToDevice, stream));

    int num_blocks = get_num_blocks(n);

    auto launch = [&](auto* ptr, auto typed_value) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        hipLaunchKernelGGL(strided_fill_kernel_device<T>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            ptr, typed_value, n, d_meta, d_meta + ndims, static_cast<int32_t>(ndims));
        HIP_CHECK(hipGetLastError());
    };

    if (self.dtype() == DType::Float32) {
        launch(self.data<float>(), static_cast<float>(value));
    } else if (self.dtype() == DType::Float64) {
        launch(self.data<double>(), value);
    } else if (self.dtype() == DType::Int32) {
        launch(self.data<int32_t>(), static_cast<int32_t>(value));
    } else if (self.dtype() == DType::Int64) {
        launch(self.data<int64_t>(), static_cast<int64_t>(value));
    } else if (self.dtype() == DType::Float16) {
        __half h_value = tenzor::rocm::safe_f2h(static_cast<float>(value));
        launch(reinterpret_cast<__half*>(self.data<Float16>()), h_value);
    } else if (self.dtype() == DType::BFloat16) {
        hip_bfloat16 bf_value = tenzor::rocm::f32_to_bf16_rne(static_cast<float>(value));
        launch(reinterpret_cast<hip_bfloat16*>(self.data<BFloat16>()), bf_value);
    } else if (self.dtype() == DType::Int8) {
        launch(self.data<int8_t>(), static_cast<int8_t>(value));
    } else if (self.dtype() == DType::UInt8) {
        launch(self.data<uint8_t>(), static_cast<uint8_t>(value));
    } else if (self.dtype() == DType::Bool) {
        launch(self.data<bool>(), value != 0.0);
    } else if (self.dtype() == DType::Int16) {
        launch(self.data<int16_t>(), static_cast<int16_t>(value));
    } else if (self.dtype() == DType::UInt16) {
        launch(self.data<uint16_t>(), static_cast<uint16_t>(value));
    } else if (self.dtype() == DType::UInt32) {
        launch(self.data<uint32_t>(), static_cast<uint32_t>(value));
    } else if (self.dtype() == DType::UInt64) {
        launch(self.data<uint64_t>(), static_cast<uint64_t>(value));
    } else if (self.dtype() == DType::Complex64) {
        launch(reinterpret_cast<hipFloatComplex*>(self.data_ptr()),
               make_hipFloatComplex(static_cast<float>(value), 0.0f));
    } else if (self.dtype() == DType::Complex128) {
        launch(reinterpret_cast<hipDoubleComplex*>(self.data_ptr()),
               make_hipDoubleComplex(static_cast<double>(value), 0.0));
    } else if (self.dtype() == DType::FP8_E4M3) {
        launch(reinterpret_cast<uint8_t*>(self.data_ptr()),
               FP8_E4M3(static_cast<float>(value)).bits);
    } else if (self.dtype() == DType::FP8_E5M2) {
        launch(reinterpret_cast<uint8_t*>(self.data_ptr()),
               FP8_E5M2(static_cast<float>(value)).bits);
    } else if (self.dtype() == DType::FP8_E4M3FNUZ) {
        launch(reinterpret_cast<uint8_t*>(self.data_ptr()),
               FP8_E4M3FNUZ(static_cast<float>(value)).bits);
    } else if (self.dtype() == DType::FP8_E5M2FNUZ) {
        launch(reinterpret_cast<uint8_t*>(self.data_ptr()),
               FP8_E5M2FNUZ(static_cast<float>(value)).bits);
    } else if (self.dtype() == DType::QInt8) {
        if (self.q_scale() == 0.0) {
            HIP_CHECK(hipFree(d_meta));
            throw std::runtime_error(
                "fill_ on quantized tensor requires quantization params: "
                "call set_quantization_params(scale, zero_point) first");
        }
        const int64_t qval = static_cast<int64_t>(std::llround(value / self.q_scale())) + self.q_zero_point();
        launch(self.data<int8_t>(), static_cast<int8_t>(std::clamp<int64_t>(qval, -128, 127)));
    } else if (self.dtype() == DType::QUInt8) {
        if (self.q_scale() == 0.0) {
            HIP_CHECK(hipFree(d_meta));
            throw std::runtime_error(
                "fill_ on quantized tensor requires quantization params: "
                "call set_quantization_params(scale, zero_point) first");
        }
        const int64_t qval = static_cast<int64_t>(std::llround(value / self.q_scale())) + self.q_zero_point();
        launch(reinterpret_cast<uint8_t*>(self.data_ptr()),
               static_cast<uint8_t>(std::clamp<int64_t>(qval, 0, 255)));
    } else if (self.dtype() == DType::QInt4x2) {
        if (self.q_scale() == 0.0) {
            HIP_CHECK(hipFree(d_meta));
            throw std::runtime_error(
                "fill_ on quantized tensor requires quantization params: "
                "call set_quantization_params(scale, zero_point) first");
        }
        const int64_t qval = static_cast<int64_t>(std::llround(value / self.q_scale())) + self.q_zero_point();
        const int64_t clamped = std::clamp<int64_t>(qval, -8, 7);
        const uint8_t packed = static_cast<uint8_t>((clamped & 0xF) | ((clamped & 0xF) << 4));
        launch(reinterpret_cast<uint8_t*>(self.data_ptr()), packed);
    } else {
        HIP_CHECK(hipFree(d_meta));
        throw std::runtime_error("strided_fill: unsupported dtype");
    }

    // audit-9 JJ.5: see repeat_kernel — sync stream before freeing d_meta
    // that the async strided_fill kernel still references.
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_meta));
}

// ============================================================================
// ToMemoryFormat kernel — NCHW <-> NHWC conversion
// ============================================================================

template<typename T>
__global__ void nchw_to_nhwc_transform(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t batch, int64_t channels, int64_t height, int64_t width
) {
    const int64_t hw = height * width;
    const int64_t chw = channels * hw;
    const int64_t hwc = hw * channels;
    const int64_t total = batch * chw;

    HIP_GRID_STRIDE_LOOP(idx, total) {
        int64_t n = idx / chw;
        int64_t rem = idx % chw;
        int64_t c = rem / hw;
        rem = rem % hw;
        int64_t h = rem / width;
        int64_t w = rem % width;

        int64_t out_idx = n * hwc + h * width * channels + w * channels + c;
        output[out_idx] = input[idx];
    }
}

template<typename T>
__global__ void nhwc_to_nchw_transform(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t batch, int64_t channels, int64_t height, int64_t width
) {
    const int64_t hw = height * width;
    const int64_t chw = channels * hw;
    const int64_t hwc = hw * channels;
    const int64_t total = batch * hwc;

    HIP_GRID_STRIDE_LOOP(idx, total) {
        int64_t n = idx / hwc;
        int64_t rem = idx % hwc;
        int64_t h = rem / (width * channels);
        rem = rem % (width * channels);
        int64_t w = rem / channels;
        int64_t c = rem % channels;

        int64_t out_idx = n * chw + c * hw + h * width + w;
        output[out_idx] = input[idx];
    }
}

auto to_memory_format_kernel(const Tensor& input, MemoryFormat format, void* stream_ptr) -> Tensor {
    hipStream_t stream = static_cast<hipStream_t>(stream_ptr);

    auto shape = input.shape();

    if (shape.size() != 4) {
        if (format == MemoryFormat::ChannelsLast) {
            return input;
        }
        return input.contiguous();
    }

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    Tensor output = Tensor::empty_uninitialized(
        std::vector<int64_t>{N, C, H, W},
        input.dtype(),
        input.device()
    );

    std::vector<int64_t> target_strides;
    if (format == MemoryFormat::ChannelsLast) {
        target_strides = {H * W * C, 1, W * C, C};
    } else {
        target_strides = {C * H * W, H * W, W, 1};
    }

    output.mutable_strides() = target_strides;

    const int64_t total = N * C * H * W;
    int num_blocks = get_num_blocks(total);

    auto launch_nchw_nhwc = [&](auto* in_ptr, auto* out_ptr) {
        using T = std::remove_pointer_t<decltype(out_ptr)>;
        if (format == MemoryFormat::ChannelsLast) {
            hipLaunchKernelGGL(nchw_to_nhwc_transform<T>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                in_ptr, out_ptr, N, C, H, W);
        } else {
            hipLaunchKernelGGL(nhwc_to_nchw_transform<T>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                in_ptr, out_ptr, N, C, H, W);
        }
        HIP_CHECK(hipGetLastError());
    };

    if (input.dtype() == DType::Float32) {
        launch_nchw_nhwc(input.data<float>(), output.data<float>());
    } else if (input.dtype() == DType::Float64) {
        launch_nchw_nhwc(input.data<double>(), output.data<double>());
    } else if (input.dtype() == DType::Float16) {
        launch_nchw_nhwc(input.data<Float16>(), output.data<Float16>());
    } else if (input.dtype() == DType::Int32) {
        launch_nchw_nhwc(input.data<int32_t>(), output.data<int32_t>());
    } else {
        throw std::runtime_error("to_memory_format_kernel: unsupported dtype");
    }

    return output;
}

// ==============================================================================
// Triu Kernel - upper triangular matrix
// ==============================================================================

template<typename T>
__global__ void triu_kernel_impl(
    const T* input,
    T* output,
    int64_t batch_size,
    int64_t rows,
    int64_t cols,
    int64_t diagonal
) {
    int64_t total = batch_size * rows * cols;

    HIP_GRID_STRIDE_LOOP(idx, total) {
        int64_t col = idx % cols;
        int64_t row = (idx / cols) % rows;

        // Upper triangular: keep elements where col >= row + diagonal
        if (col >= row + diagonal) {
            output[idx] = input[idx];
        } else {
            output[idx] = T(0);
        }
    }
}

auto triu_kernel(const Tensor& input, int64_t diagonal, hipStream_t stream) -> Tensor {
    auto input_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(input_shape.size());

    if (ndim < 2) {
        throw std::runtime_error("triu: input must be at least 2D");
    }

    // The kernel indexes by flat contiguous offset, so materialize a contiguous
    // input for non-contiguous (e.g. transposed/strided) views to match CPU.
    auto cont = input.is_contiguous() ? input : input.contiguous();

    Tensor result(std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                  input.dtype(), input.device());

    int64_t rows = input_shape[ndim - 2];
    int64_t cols = input_shape[ndim - 1];

    // Compute batch size (product of all dims except last 2)
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) {
        batch_size *= input_shape[i];
    }

    int64_t total_elements = input.numel();
    int num_blocks = get_num_blocks(total_elements);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(triu_kernel_impl<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            cont.data<float>(), result.data<float>(),
            batch_size, rows, cols, diagonal);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(triu_kernel_impl<double>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            cont.data<double>(), result.data<double>(),
            batch_size, rows, cols, diagonal);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(triu_kernel_impl<__half>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const __half*>(cont.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            batch_size, rows, cols, diagonal);
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(triu_kernel_impl<int32_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            cont.data<int32_t>(), result.data<int32_t>(),
            batch_size, rows, cols, diagonal);
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(triu_kernel_impl<int64_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            cont.data<int64_t>(), result.data<int64_t>(),
            batch_size, rows, cols, diagonal);
    } else {
        throw std::runtime_error("triu: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// ==============================================================================
// Tril Kernel - lower triangular matrix
// ==============================================================================

template<typename T>
__global__ void tril_kernel_impl(
    const T* input,
    T* output,
    int64_t batch_size,
    int64_t rows,
    int64_t cols,
    int64_t diagonal
) {
    int64_t total = batch_size * rows * cols;

    HIP_GRID_STRIDE_LOOP(idx, total) {
        int64_t col = idx % cols;
        int64_t row = (idx / cols) % rows;

        // Lower triangular: keep elements where col <= row + diagonal
        if (col <= row + diagonal) {
            output[idx] = input[idx];
        } else {
            output[idx] = T(0);
        }
    }
}

auto tril_kernel(const Tensor& input, int64_t diagonal, hipStream_t stream) -> Tensor {
    auto input_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(input_shape.size());

    if (ndim < 2) {
        throw std::runtime_error("tril: input must be at least 2D");
    }

    // The kernel indexes by flat contiguous offset, so materialize a contiguous
    // input for non-contiguous (e.g. transposed/strided) views to match CPU.
    auto cont = input.is_contiguous() ? input : input.contiguous();

    Tensor result(std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                  input.dtype(), input.device());

    int64_t rows = input_shape[ndim - 2];
    int64_t cols = input_shape[ndim - 1];

    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 2; ++i) {
        batch_size *= input_shape[i];
    }

    int64_t total_elements = input.numel();
    int num_blocks = get_num_blocks(total_elements);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(tril_kernel_impl<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            cont.data<float>(), result.data<float>(),
            batch_size, rows, cols, diagonal);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(tril_kernel_impl<double>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            cont.data<double>(), result.data<double>(),
            batch_size, rows, cols, diagonal);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(tril_kernel_impl<__half>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const __half*>(cont.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            batch_size, rows, cols, diagonal);
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(tril_kernel_impl<int32_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            cont.data<int32_t>(), result.data<int32_t>(),
            batch_size, rows, cols, diagonal);
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(tril_kernel_impl<int64_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            cont.data<int64_t>(), result.data<int64_t>(),
            batch_size, rows, cols, diagonal);
    } else {
        throw std::runtime_error("tril: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// ==============================================================================
// Diag Kernel - extract diagonal (2D->1D) or construct diagonal matrix (1D->2D)
// ==============================================================================

template<typename T>
__global__ void diag_extract_kernel_impl(
    const T* input,
    T* output,
    int64_t diag_size,
    int64_t rows,
    int64_t cols,
    int64_t diagonal
) {
    HIP_GRID_STRIDE_LOOP(idx, diag_size) {
        int64_t row, col;
        if (diagonal >= 0) {
            row = idx;
            col = idx + diagonal;
        } else {
            row = idx - diagonal;
            col = idx;
        }
        output[idx] = input[row * cols + col];
    }
}

template<typename T>
__global__ void diag_construct_simple_kernel(
    const T* input,
    T* output,
    int64_t n,
    int64_t diag_size,
    int64_t diagonal
) {
    int64_t total = n * n;

    HIP_GRID_STRIDE_LOOP(idx, total) {
        int64_t row = idx / n;
        int64_t col = idx % n;

        bool on_diag = false;
        int64_t d_idx = -1;
        if (diagonal >= 0) {
            if (col - row == diagonal) {
                on_diag = true;
                d_idx = row;
            }
        } else {
            if (row - col == -diagonal) {
                on_diag = true;
                d_idx = col;
            }
        }

        if (on_diag && d_idx >= 0 && d_idx < diag_size) {
            output[idx] = input[d_idx];
        } else {
            output[idx] = T(0);
        }
    }
}

auto diag_kernel(const Tensor& input, int64_t diagonal, hipStream_t stream) -> Tensor {
    auto input_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(input_shape.size());

    // Both the 2D-extract and 1D-construct paths index by flat contiguous
    // offset, so materialize a contiguous input for strided/transposed views
    // to match CPU (src/ops/transform.cpp:797,818).
    auto cont = input.is_contiguous() ? input : input.contiguous();

    if (ndim == 2) {
        // Extract diagonal from 2D matrix -> 1D vector
        int64_t rows = input_shape[0];
        int64_t cols = input_shape[1];

        int64_t diag_size;
        if (diagonal >= 0) {
            diag_size = std::min(rows, cols - diagonal);
        } else {
            diag_size = std::min(rows + diagonal, cols);
        }

        if (diag_size <= 0) {
            return Tensor({0}, input.dtype(), input.device());
        }

        Tensor result({diag_size}, input.dtype(), input.device());
        int num_blocks = get_num_blocks(diag_size);

        if (input.dtype() == DType::Float32) {
            hipLaunchKernelGGL(diag_extract_kernel_impl<float>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                cont.data<float>(), result.data<float>(),
                diag_size, rows, cols, diagonal);
        } else if (input.dtype() == DType::Float64) {
            hipLaunchKernelGGL(diag_extract_kernel_impl<double>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                cont.data<double>(), result.data<double>(),
                diag_size, rows, cols, diagonal);
        } else if (input.dtype() == DType::Float16) {
            hipLaunchKernelGGL(diag_extract_kernel_impl<__half>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                reinterpret_cast<const __half*>(cont.data<Float16>()),
                reinterpret_cast<__half*>(result.data<Float16>()),
                diag_size, rows, cols, diagonal);
        } else if (input.dtype() == DType::Int32) {
            hipLaunchKernelGGL(diag_extract_kernel_impl<int32_t>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                cont.data<int32_t>(), result.data<int32_t>(),
                diag_size, rows, cols, diagonal);
        } else if (input.dtype() == DType::Int64) {
            hipLaunchKernelGGL(diag_extract_kernel_impl<int64_t>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                cont.data<int64_t>(), result.data<int64_t>(),
                diag_size, rows, cols, diagonal);
        } else {
            throw std::runtime_error("diag: unsupported dtype");
        }

        HIP_CHECK(hipGetLastError());
        return result;

    } else if (ndim == 1) {
        // Construct diagonal matrix from 1D vector -> 2D matrix
        int64_t diag_size = input_shape[0];
        int64_t n = diag_size + std::abs(diagonal);

        Tensor result({n, n}, input.dtype(), input.device());
        int64_t total = n * n;
        int num_blocks = get_num_blocks(total);

        if (input.dtype() == DType::Float32) {
            hipLaunchKernelGGL(diag_construct_simple_kernel<float>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                cont.data<float>(), result.data<float>(),
                n, diag_size, diagonal);
        } else if (input.dtype() == DType::Float64) {
            hipLaunchKernelGGL(diag_construct_simple_kernel<double>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                cont.data<double>(), result.data<double>(),
                n, diag_size, diagonal);
        } else if (input.dtype() == DType::Float16) {
            hipLaunchKernelGGL(diag_construct_simple_kernel<__half>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                reinterpret_cast<const __half*>(cont.data<Float16>()),
                reinterpret_cast<__half*>(result.data<Float16>()),
                n, diag_size, diagonal);
        } else if (input.dtype() == DType::Int32) {
            hipLaunchKernelGGL(diag_construct_simple_kernel<int32_t>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                cont.data<int32_t>(), result.data<int32_t>(),
                n, diag_size, diagonal);
        } else if (input.dtype() == DType::Int64) {
            hipLaunchKernelGGL(diag_construct_simple_kernel<int64_t>,
                dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
                cont.data<int64_t>(), result.data<int64_t>(),
                n, diag_size, diagonal);
        } else {
            throw std::runtime_error("diag: unsupported dtype");
        }

        HIP_CHECK(hipGetLastError());
        return result;

    } else {
        throw std::runtime_error("diag: input must be 1D or 2D");
    }
}

// ==============================================================================
// Trace Kernel - sum of diagonal elements
// ==============================================================================

template<typename T>
__global__ void trace_kernel_impl(
    const T* input,
    T* output,
    int64_t diag_size,
    int64_t cols
) {
    __shared__ T shared[256];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    // Grid-stride loop to sum diagonal elements
    T thread_sum = T(0);
    for (int64_t i = idx; i < diag_size; i += grid_size) {
        thread_sum += input[i * cols + i];
    }

    shared[tid] = thread_sum;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

// CAS-based atomicAdd for int64_t (not natively available in HIP)
__device__ inline long long atomicAddInt64(long long* addr, long long val) {
    unsigned long long* uaddr = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = atomicCAS(uaddr, 0ULL, 0ULL);  // atomic initial read
    unsigned long long assumed;
    do {
        assumed = old;
        old = atomicCAS(uaddr, assumed, assumed + static_cast<unsigned long long>(val));
    } while (assumed != old);
    return static_cast<long long>(old);
}

// Specialization for int64_t trace using CAS atomicAdd
__global__ void trace_kernel_impl_i64(
    const int64_t* input,
    int64_t* output,
    int64_t diag_size,
    int64_t cols
) {
    __shared__ int64_t shared[256];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    int64_t thread_sum = 0;
    for (int64_t i = idx; i < diag_size; i += grid_size) {
        thread_sum += input[i * cols + i];
    }

    shared[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAddInt64(reinterpret_cast<long long*>(output), static_cast<long long>(shared[0]));
    }
}

// Specialization for double atomicAdd (not natively available on all HIP devices)
__global__ void trace_kernel_impl_f64(
    const double* input,
    double* output,
    int64_t diag_size,
    int64_t cols
) {
    __shared__ double shared[256];

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    double thread_sum = 0.0;
    for (int64_t i = idx; i < diag_size; i += grid_size) {
        thread_sum += input[i * cols + i];
    }

    shared[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // CAS-based atomicAdd for double
        unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(output);
        unsigned long long int old_val = *address_as_ull;
        unsigned long long int assumed;
        do {
            assumed = old_val;
            old_val = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(__longlong_as_double(assumed) + shared[0]));
        } while (assumed != old_val);
    }
}

auto trace_kernel(const Tensor& input, hipStream_t stream) -> Tensor {
    auto input_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(input_shape.size());

    if (ndim != 2) {
        throw std::runtime_error("trace: input must be 2D");
    }

    int64_t rows = input_shape[0];
    int64_t cols = input_shape[1];
    int64_t diag_size = std::min(rows, cols);

    // The kernel sums input[i*cols+i] using contiguous addressing, so
    // materialize a contiguous input for strided/sliced square views to
    // match CPU (trace()->diag()->contiguous).
    auto cont = input.is_contiguous() ? input : input.contiguous();

    // Output is a scalar
    Tensor result({}, input.dtype(), input.device());

    if (diag_size == 0) {
        // Zero-initialize for empty diagonal
        HIP_CHECK(hipMemsetAsync(result.data_ptr(), 0, result.numel() * dtype_size(input.dtype()), stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        return result;
    }

    // Zero-initialize the output for atomicAdd
    HIP_CHECK(hipMemsetAsync(result.data_ptr(), 0, dtype_size(input.dtype()), stream));

    int num_blocks = std::min<int>(get_num_blocks(diag_size), 256);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(trace_kernel_impl<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            cont.data<float>(), result.data<float>(),
            diag_size, cols);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(trace_kernel_impl_f64,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            cont.data<double>(), result.data<double>(),
            diag_size, cols);
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(trace_kernel_impl<int32_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            cont.data<int32_t>(), result.data<int32_t>(),
            diag_size, cols);
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(trace_kernel_impl_i64,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            cont.data<int64_t>(), result.data<int64_t>(),
            diag_size, cols);
    } else {
        throw std::runtime_error("trace: unsupported dtype (supports Float32, Float64, Int32, Int64)");
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    return result;
}

// ============================================================================
// repeat_interleave — repeat each element along a dimension
// ============================================================================

template<typename T>
__global__ void repeat_interleave_scalar_kernel_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t total_elements,
    int64_t in_dim_size,
    int64_t out_dim_size,
    int64_t repeats,
    int64_t inner_size
) {
    HIP_GRID_STRIDE_LOOP(i, total_elements) {
        int64_t inner_idx = i % inner_size;
        int64_t out_dim_idx = (i / inner_size) % out_dim_size;
        int64_t outer_idx = i / (inner_size * out_dim_size);

        int64_t src_dim_idx = out_dim_idx / repeats;
        int64_t src_idx = (outer_idx * in_dim_size + src_dim_idx) * inner_size + inner_idx;

        output[i] = input[src_idx];
    }
}

template<typename T>
__global__ void repeat_interleave_tensor_kernel_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int64_t* __restrict__ prefix,
    int64_t total_elements,
    int64_t in_dim_size,
    int64_t out_dim_size,
    int64_t inner_size
) {
    HIP_GRID_STRIDE_LOOP(i, total_elements) {
        int64_t inner_idx = i % inner_size;
        int64_t out_dim_idx = (i / inner_size) % out_dim_size;
        int64_t outer_idx = i / (inner_size * out_dim_size);

        // Binary search in prefix array
        int64_t lo = 0, hi = in_dim_size;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if (prefix[mid + 1] <= out_dim_idx) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        int64_t src_dim_idx = lo;
        int64_t src_idx = (outer_idx * in_dim_size + src_dim_idx) * inner_size + inner_idx;

        output[i] = input[src_idx];
    }
}

auto repeat_interleave_scalar_kernel(const Tensor& input, int64_t repeats, int64_t dim,
                                     hipStream_t stream) -> Tensor {
    auto shape = input.shape();
    auto cont = input.is_contiguous() ? input : input.contiguous();

    int64_t ndim = shape.size();
    // Normalize negative dim (PyTorch semantics) before indexing shape[dim].
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("repeat_interleave: dim out of range");
    }
    int64_t in_dim_size = shape[dim];
    int64_t out_dim_size = in_dim_size * repeats;

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = out_dim_size;

    auto output = Tensor::empty_uninitialized(out_shape, input.dtype(), input.device());

    int64_t total = 1;
    for (auto s : out_shape) total *= s;
    if (total == 0) return output;

    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) inner_size *= shape[d];

    constexpr int BLOCK = 256;
    int64_t num_blocks = (total + BLOCK - 1) / BLOCK;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(repeat_interleave_scalar_kernel_impl<float>,
            dim3(num_blocks), dim3(BLOCK), 0, stream,
            cont.data<float>(), output.data<float>(),
            total, in_dim_size, out_dim_size, repeats, inner_size);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(repeat_interleave_scalar_kernel_impl<double>,
            dim3(num_blocks), dim3(BLOCK), 0, stream,
            cont.data<double>(), output.data<double>(),
            total, in_dim_size, out_dim_size, repeats, inner_size);
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(repeat_interleave_scalar_kernel_impl<int32_t>,
            dim3(num_blocks), dim3(BLOCK), 0, stream,
            cont.data<int32_t>(), output.data<int32_t>(),
            total, in_dim_size, out_dim_size, repeats, inner_size);
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(repeat_interleave_scalar_kernel_impl<int64_t>,
            dim3(num_blocks), dim3(BLOCK), 0, stream,
            cont.data<int64_t>(), output.data<int64_t>(),
            total, in_dim_size, out_dim_size, repeats, inner_size);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(repeat_interleave_scalar_kernel_impl<__half>,
            dim3(num_blocks), dim3(BLOCK), 0, stream,
            reinterpret_cast<const __half*>(cont.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            total, in_dim_size, out_dim_size, repeats, inner_size);
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(repeat_interleave_scalar_kernel_impl<uint16_t>,
            dim3(num_blocks), dim3(BLOCK), 0, stream,
            reinterpret_cast<const uint16_t*>(cont.data<BFloat16>()),
            reinterpret_cast<uint16_t*>(output.data<BFloat16>()),
            total, in_dim_size, out_dim_size, repeats, inner_size);
    } else {
        throw std::runtime_error("repeat_interleave_scalar_kernel: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

// Cast repeats tensor to int64 on device
template <typename SrcT>
__global__ void ri_cast_to_int64_kernel(const SrcT* __restrict__ src,
                                        int64_t* __restrict__ dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = static_cast<int64_t>(src[idx]);
}

auto repeat_interleave_tensor_kernel(const Tensor& input, const Tensor& repeats_tensor,
                                     int64_t dim, hipStream_t stream) -> Tensor {
    auto shape = input.shape();
    auto cont = input.is_contiguous() ? input : input.contiguous();

    int64_t ndim = shape.size();
    // Normalize negative dim (PyTorch semantics) before indexing shape[dim].
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("repeat_interleave: dim out of range");
    }
    int64_t in_dim_size = shape[dim];

    // Convert repeats to int64 on device (no CPU roundtrip)
    int64_t* d_repeats_i64 = nullptr;
    HIP_CHECK(hipMalloc(&d_repeats_i64, in_dim_size * sizeof(int64_t)));

    auto repeats_cont = repeats_tensor.is_contiguous() ? repeats_tensor : repeats_tensor.contiguous();
    constexpr int CAST_BLOCK = 256;
    int64_t cast_blocks = (in_dim_size + CAST_BLOCK - 1) / CAST_BLOCK;

    if (repeats_cont.dtype() == DType::Int64) {
        HIP_CHECK(hipMemcpyAsync(d_repeats_i64, repeats_cont.data<int64_t>(),
                                 in_dim_size * sizeof(int64_t),
                                 hipMemcpyDeviceToDevice, stream));
    } else if (repeats_cont.dtype() == DType::Int32) {
        hipLaunchKernelGGL(ri_cast_to_int64_kernel<int32_t>,
            dim3(cast_blocks), dim3(CAST_BLOCK), 0, stream,
            repeats_cont.data<int32_t>(), d_repeats_i64, in_dim_size);
    } else if (repeats_cont.dtype() == DType::Float32) {
        hipLaunchKernelGGL(ri_cast_to_int64_kernel<float>,
            dim3(cast_blocks), dim3(CAST_BLOCK), 0, stream,
            repeats_cont.data<float>(), d_repeats_i64, in_dim_size);
    } else if (repeats_cont.dtype() == DType::Float64) {
        hipLaunchKernelGGL(ri_cast_to_int64_kernel<double>,
            dim3(cast_blocks), dim3(CAST_BLOCK), 0, stream,
            repeats_cont.data<double>(), d_repeats_i64, in_dim_size);
    } else {
        (void)hipFree(d_repeats_i64);
        throw std::runtime_error("repeat_interleave: unsupported repeats dtype");
    }

    // Compute exclusive prefix sum on device using hipcub
    int64_t* d_prefix = nullptr;
    HIP_CHECK(hipMalloc(&d_prefix, (in_dim_size + 1) * sizeof(int64_t)));

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    HIP_CHECK(hipcub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_repeats_i64, d_prefix,
                                     static_cast<int>(in_dim_size), stream));
    HIP_CHECK(hipMalloc(&d_temp, temp_bytes));
    HIP_CHECK(hipcub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_repeats_i64, d_prefix,
                                     static_cast<int>(in_dim_size), stream));
    // audit-9 JJ.5: ExclusiveSum is async on `stream`; sync before freeing
    // the temp workspace.
    HIP_CHECK(hipStreamSynchronize(stream));
    (void)hipFree(d_temp);

    // Read only the total from device (2 scalars: last prefix + last repeat)
    int64_t last_prefix = 0, last_repeat = 0;
    HIP_CHECK(hipMemcpyAsync(&last_prefix, d_prefix + in_dim_size - 1,
                             sizeof(int64_t), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(&last_repeat, d_repeats_i64 + in_dim_size - 1,
                             sizeof(int64_t), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    int64_t out_dim_size = last_prefix + last_repeat;
    // Write total to d_prefix[in_dim_size] for the kernel
    HIP_CHECK(hipMemcpyAsync(d_prefix + in_dim_size, &out_dim_size,
                             sizeof(int64_t), hipMemcpyHostToDevice, stream));

    (void)hipFree(d_repeats_i64);

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = out_dim_size;

    auto output = Tensor::empty_uninitialized(out_shape, input.dtype(), input.device());

    int64_t total = 1;
    for (auto s : out_shape) total *= s;
    if (total == 0) {
        (void)hipFree(d_prefix);
        return output;
    }

    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) inner_size *= shape[d];

    constexpr int BLOCK = 256;
    int64_t num_blocks = (total + BLOCK - 1) / BLOCK;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(repeat_interleave_tensor_kernel_impl<float>,
            dim3(num_blocks), dim3(BLOCK), 0, stream,
            cont.data<float>(), output.data<float>(), d_prefix,
            total, in_dim_size, out_dim_size, inner_size);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(repeat_interleave_tensor_kernel_impl<double>,
            dim3(num_blocks), dim3(BLOCK), 0, stream,
            cont.data<double>(), output.data<double>(), d_prefix,
            total, in_dim_size, out_dim_size, inner_size);
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(repeat_interleave_tensor_kernel_impl<int32_t>,
            dim3(num_blocks), dim3(BLOCK), 0, stream,
            cont.data<int32_t>(), output.data<int32_t>(), d_prefix,
            total, in_dim_size, out_dim_size, inner_size);
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(repeat_interleave_tensor_kernel_impl<int64_t>,
            dim3(num_blocks), dim3(BLOCK), 0, stream,
            cont.data<int64_t>(), output.data<int64_t>(), d_prefix,
            total, in_dim_size, out_dim_size, inner_size);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(repeat_interleave_tensor_kernel_impl<__half>,
            dim3(num_blocks), dim3(BLOCK), 0, stream,
            reinterpret_cast<const __half*>(cont.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()), d_prefix,
            total, in_dim_size, out_dim_size, inner_size);
    } else if (input.dtype() == DType::BFloat16) {
        hipLaunchKernelGGL(repeat_interleave_tensor_kernel_impl<uint16_t>,
            dim3(num_blocks), dim3(BLOCK), 0, stream,
            reinterpret_cast<const uint16_t*>(cont.data<BFloat16>()),
            reinterpret_cast<uint16_t*>(output.data<BFloat16>()), d_prefix,
            total, in_dim_size, out_dim_size, inner_size);
    } else {
        (void)hipFree(d_prefix);
        throw std::runtime_error("repeat_interleave_tensor_kernel: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    (void)hipFree(d_prefix);
    return output;
}

// ============================================================================
// DiagEmbed — embed a vector as a diagonal of a matrix
//   Input: (..., N) -> Output: (..., M, M) where M = N + |offset|
// ============================================================================

template <typename T>
__global__ void diag_embed_kernel_impl(const T* __restrict__ input,
                                       T* __restrict__ output,
                                       int64_t batch_size, int64_t diag_size,
                                       int64_t mat_size, int64_t offset,
                                       int64_t dim1, int64_t dim2) {
    int64_t total = batch_size * mat_size * mat_size;
    HIP_GRID_STRIDE_LOOP(idx, total) {
        int64_t b = idx / (mat_size * mat_size);
        int64_t rem = idx % (mat_size * mat_size);
        int64_t r = rem / mat_size;
        int64_t c = rem % mat_size;

        int64_t diag_idx = -1;
        if (offset >= 0) {
            if (c - r == offset && r < diag_size) diag_idx = r;
        } else {
            if (r - c == -offset && c < diag_size) diag_idx = c;
        }

        if (diag_idx >= 0 && diag_idx < diag_size) {
            output[idx] = input[b * diag_size + diag_idx];
        } else {
            output[idx] = T(0);
        }
    }
}

auto diag_embed_kernel(const Tensor& input, int64_t offset, int64_t dim1, int64_t dim2,
                       hipStream_t stream) -> Tensor {
    auto input_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(input_shape.size());
    if (ndim < 1) {
        throw std::runtime_error("diag_embed: input must be at least 1D");
    }

    int64_t diag_size = input_shape[ndim - 1];
    int64_t mat_size = diag_size + std::abs(offset);

    // The kernel reads input[b*diag_size+diag_idx] assuming a contiguous
    // (batch, diag_size) buffer, so materialize a contiguous input for
    // strided views to match CPU/CUDA.
    auto cont = input.is_contiguous() ? input : input.contiguous();

    // Compute batch size (product of all dims except last)
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim - 1; ++i) batch_size *= input_shape[i];

    // Output shape: (..., mat_size, mat_size)
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim - 1; ++i) out_shape.push_back(input_shape[i]);
    out_shape.push_back(mat_size);
    out_shape.push_back(mat_size);

    Tensor result(out_shape, input.dtype(), input.device());
    int64_t total = batch_size * mat_size * mat_size;
    if (total == 0) return result;
    int num_blocks = get_num_blocks(total);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(diag_embed_kernel_impl<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            cont.data<float>(), result.data<float>(),
            batch_size, diag_size, mat_size, offset, dim1, dim2);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(diag_embed_kernel_impl<double>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            cont.data<double>(), result.data<double>(),
            batch_size, diag_size, mat_size, offset, dim1, dim2);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(diag_embed_kernel_impl<__half>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const __half*>(cont.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            batch_size, diag_size, mat_size, offset, dim1, dim2);
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(diag_embed_kernel_impl<int32_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            cont.data<int32_t>(), result.data<int32_t>(),
            batch_size, diag_size, mat_size, offset, dim1, dim2);
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(diag_embed_kernel_impl<int64_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            cont.data<int64_t>(), result.data<int64_t>(),
            batch_size, diag_size, mat_size, offset, dim1, dim2);
    } else {
        throw std::runtime_error("diag_embed: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return result;
}

// ============================================================================
// Diagflat — flatten input and create diagonal matrix
// ============================================================================

auto diagflat_kernel(const Tensor& input, int64_t offset, hipStream_t stream) -> Tensor {
    // Flatten the input to 1D first
    int64_t n = input.numel();
    std::vector<int64_t> flat_shape = {n};

    // Reuse diag_embed on the flattened tensor
    // diag_embed expects (..., N) and produces (..., M, M)
    // For diagflat, we flatten to (N,) and produce (M, M)
    Tensor flat_input = input;
    if (input.shape().size() != 1 || input.shape()[0] != n) {
        // Need a contiguous 1D view. Materialise the (possibly strided) input
        // contiguously, then copy the exact byte count for this dtype. The
        // previous hand-rolled element-size table mis-sized Complex64/Complex128
        // (copied only 4B/elem), Int16/UInt16 (4B instead of 2B -> OOB), and
        // the unsigned 32/64-bit types, corrupting the raw byte copy.
        Tensor src = input.is_contiguous() ? input : input.contiguous();
        flat_input = Tensor(flat_shape, input.dtype(), input.device());
        int64_t bytes = n * static_cast<int64_t>(dtype_size(input.dtype()));
        HIP_CHECK(hipMemcpyAsync(flat_input.data_ptr(), src.data_ptr(),
                                  bytes, hipMemcpyDeviceToDevice, stream));
    }

    return diag_embed_kernel(flat_input, offset, -2, -1, stream);
}

} // namespace rocm
} // namespace tenzor
