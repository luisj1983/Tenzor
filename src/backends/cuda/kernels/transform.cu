#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>
#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/caching_allocator.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/shape.hpp"
#include "cuda_common.cuh"
#include "cuda_launch_utils.cuh"
#include <stdexcept>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <type_traits>

namespace tenzor {
namespace cuda {

// Helper class to access Tensor private members from CUDA kernels.
// Routes through TensorAccessor which is a friend of Tensor.
class CUDAKernelAccess {
public:
    static auto get_impl(const Tensor& t) -> const intrusive_ptr<TensorImpl>& {
        return TensorAccessor::get_impl(t);
    }
    static auto get_impl_mutable(Tensor& t) -> intrusive_ptr<TensorImpl>& {
        return TensorAccessor::get_impl_mutable(t);
    }
};

// Use centralized CUDA error handling
#include "../cuda_error.hpp"



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

// Metadata struct passed by value to kernels (avoids cudaMalloc for shape/stride arrays).
//
// Audit F.12: rank cap lifted from 8 to 16 with a runtime throw on overflow.
// The previous code silently truncated any tensor with rank > 8 — the make
// function looped `i < 8` and the kernel iterated only those slots, dropping
// higher dimensions. 16 matches the cap used by reduction.cu / activations.cu
// (DIM_META_MAX_RANK) and the maximum supported in CPU/ROCm.
constexpr int TRANSFORM_DIM_META_MAX_RANK = 16;
struct TransformMeta {
    int64_t shape[TRANSFORM_DIM_META_MAX_RANK];
    int64_t strides[TRANSFORM_DIM_META_MAX_RANK];
};

static TransformMeta make_transform_meta(const std::vector<int64_t>& shape,
                                          const std::vector<int64_t>& strides) {
    if (shape.size() > TRANSFORM_DIM_META_MAX_RANK) {
        throw std::runtime_error(
            "CUDA TransformMeta: tensor rank " + std::to_string(shape.size()) +
            " exceeds maximum " + std::to_string(TRANSFORM_DIM_META_MAX_RANK) +
            " (raise TRANSFORM_DIM_META_MAX_RANK if needed).");
    }
    TransformMeta meta{};
    for (size_t i = 0; i < shape.size(); ++i) {
        meta.shape[i] = shape[i];
        meta.strides[i] = strides[i];
    }
    return meta;
}

// Contiguous kernel - copies non-contiguous data to contiguous layout
template<typename T>
__global__ void contiguous_kernel_impl(const T* input, T* output,
                                       TransformMeta meta,
                                       int64_t ndim, int64_t total_elements) {
    TENZOR_CUDA_KERNEL_LOOP(idx, total_elements) {
        // Convert linear index to multi-dimensional indices
        int64_t temp_idx = idx;
        int64_t src_offset = 0;

        for (int64_t dim = ndim - 1; dim >= 0; --dim) {
            int64_t coord = temp_idx % meta.shape[dim];
            src_offset += coord * meta.strides[dim];
            temp_idx /= meta.shape[dim];
        }

        output[idx] = input[src_offset];
    }
}

// Contiguous wrapper function
auto contiguous_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
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

    // Pass strides and shape by value as kernel argument
    std::vector<int64_t> strides_vec(input.strides().begin(), input.strides().end());
    TransformMeta meta = make_transform_meta(shape, strides_vec);

    // Launch kernel
    int num_blocks = get_num_blocks(total_elements);

    if (input.dtype() == DType::Float32) {
        contiguous_kernel_impl<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), result.data<float>(),
            meta, ndim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        contiguous_kernel_impl<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(),
            meta, ndim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        contiguous_kernel_impl<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()),
            meta, ndim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        contiguous_kernel_impl<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(result.data_ptr()),
            meta, ndim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int32) {
        contiguous_kernel_impl<int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<int32_t>(), result.data<int32_t>(),
            meta, ndim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int64) {
        contiguous_kernel_impl<int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<int64_t>(), result.data<int64_t>(),
            meta, ndim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int8) {
        contiguous_kernel_impl<int8_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<int8_t>(), result.data<int8_t>(),
            meta, ndim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::UInt8) {
        contiguous_kernel_impl<uint8_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<uint8_t>(), result.data<uint8_t>(),
            meta, ndim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Bool) {
        contiguous_kernel_impl<bool><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const bool*>(input.data_ptr()),
            reinterpret_cast<bool*>(result.data_ptr()),
            meta, ndim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Complex64) {
        // Complex64 = 2x float; treat as float2 (8 bytes/element)
        contiguous_kernel_impl<float2><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const float2*>(input.data_ptr()),
            reinterpret_cast<float2*>(result.data_ptr()),
            meta, ndim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Complex128) {
        contiguous_kernel_impl<double2><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const double2*>(input.data_ptr()),
            reinterpret_cast<double2*>(result.data_ptr()),
            meta, ndim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int16) {
        contiguous_kernel_impl<int16_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<int16_t>(), result.data<int16_t>(),
            meta, ndim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::UInt16) {
        contiguous_kernel_impl<uint16_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<uint16_t>(), result.data<uint16_t>(),
            meta, ndim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::UInt32) {
        contiguous_kernel_impl<uint32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<uint32_t>(), result.data<uint32_t>(),
            meta, ndim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::UInt64) {
        contiguous_kernel_impl<uint64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<uint64_t>(), result.data<uint64_t>(),
            meta, ndim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::FP8_E4M3 || input.dtype() == DType::FP8_E5M2 ||
               input.dtype() == DType::FP8_E4M3FNUZ || input.dtype() == DType::FP8_E5M2FNUZ ||
               input.dtype() == DType::QInt8 || input.dtype() == DType::QUInt8 ||
               input.dtype() == DType::QInt4x2) {
        // All 1-byte storage types: contiguous copy is a pure byte move.
        contiguous_kernel_impl<uint8_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(input.data_ptr()),
            reinterpret_cast<uint8_t*>(result.data_ptr()),
            meta, ndim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Contiguous: unsupported dtype");
    }

    return result;
}

// Clone kernel - device-to-device copy
auto clone_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    // Make contiguous first if needed
    Tensor cont = input.is_contiguous() ? input : contiguous_kernel(input, stream);

    // Create new tensor
    std::vector<int64_t> shape(cont.shape().begin(), cont.shape().end());
    Tensor result(shape, cont.dtype(), cont.device());

    // Copy data using cudaMemcpy
    const size_t size_bytes = cont.numel() * dtype_size(cont.dtype());
    CUDA_CHECK(cudaMemcpyAsync(result.data<uint8_t>(), cont.data<uint8_t>(),
                                size_bytes, cudaMemcpyDeviceToDevice, stream));

    return result;
}

// Reshape kernel - metadata manipulation (create view with new shape)
auto reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape, cudaStream_t stream) -> Tensor {
    // Reshape just manipulates metadata - create view
    // If not contiguous, need to make contiguous first
    if (!input.is_contiguous()) {
        return reshape_kernel(contiguous_kernel(input, stream), new_shape, stream);
    }

    // Create new tensor sharing storage (view)
    Tensor result;
    CUDAKernelAccess::get_impl_mutable(result) = make_intrusive<TensorImpl>(*CUDAKernelAccess::get_impl(input));
    result.mutable_shape() = new_shape;
    result.mutable_strides() = compute_strides(new_shape);

    return result;
}

// Transpose kernel - metadata manipulation (swap dimensions)
auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1, cudaStream_t stream) -> Tensor {
    // Normalize negative dims and range-check before indexing the shape/stride
    // vectors. The public Tensor::transpose normalizes, but the OpId::Transpose
    // dispatch path forwards AttrKey::Dim0/Dim1 verbatim, so a negative dim from
    // a lazy/JIT/jvp graph would index out of bounds (UB) without this guard.
    const int64_t ndim = input.ndim();
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        throw std::out_of_range("transpose: dimension out of range");
    }
    // Transpose just swaps dimensions in metadata
    Tensor result;
    CUDAKernelAccess::get_impl_mutable(result) = make_intrusive<TensorImpl>(*CUDAKernelAccess::get_impl(input));
    auto& r_shape = result.mutable_shape();
    auto& r_strides = result.mutable_strides();
    std::swap(r_shape[dim0], r_shape[dim1]);
    std::swap(r_strides[dim0], r_strides[dim1]);
    return result;
}

// Permute kernel - metadata manipulation (permute dimensions)
auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims, cudaStream_t stream) -> Tensor {
    const int64_t ndim = input.ndim();

    Tensor result;
    CUDAKernelAccess::get_impl_mutable(result) = make_intrusive<TensorImpl>(*CUDAKernelAccess::get_impl(input));

    std::vector<int64_t> new_shape(ndim);
    std::vector<int64_t> new_strides(ndim);

    // Normalize negative dims and range-check before indexing shape/strides.
    // The public Tensor::permute normalizes, but the OpId::Permute dispatch path
    // forwards AttrKey::Dims verbatim, so a negative/out-of-range dim from a
    // lazy/JIT/jvp graph would index out of bounds (UB) without this guard.
    if (static_cast<int64_t>(dims.size()) != ndim) {
        throw std::invalid_argument("permute: dims size must equal tensor rank");
    }
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t d = dims[i];
        if (d < 0) d += ndim;
        if (d < 0 || d >= ndim) {
            throw std::out_of_range("permute: dimension out of range");
        }
        new_shape[i] = input.shape()[d];
        new_strides[i] = input.strides()[d];
    }

    result.mutable_shape() = std::move(new_shape);
    result.mutable_strides() = std::move(new_strides);

    return result;
}

// Squeeze kernel - metadata manipulation (remove dimension)
auto squeeze_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor {
    Tensor result;
    CUDAKernelAccess::get_impl_mutable(result) = make_intrusive<TensorImpl>(*CUDAKernelAccess::get_impl(input));

    if (dim >= 0) {
        // Squeeze specific dimension. Validate the range: an out-of-range dim
        // (e.g. a stale/garbage AttrKey::Dim) would erase past the end of the
        // shape/stride vectors (UB). PyTorch also treats squeezing a dim whose
        // size != 1 as a no-op rather than removing it.
        if (dim >= input.ndim()) {
            throw std::out_of_range("squeeze: dimension out of range");
        }
        if (input.shape()[dim] == 1) {
            auto& r_shape = result.mutable_shape();
            auto& r_strides = result.mutable_strides();
            r_shape.erase(r_shape.begin() + dim);
            r_strides.erase(r_strides.begin() + dim);
        }
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

// Unsqueeze kernel - metadata manipulation (add dimension)
auto unsqueeze_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor {
    Tensor result;
    CUDAKernelAccess::get_impl_mutable(result) = make_intrusive<TensorImpl>(*CUDAKernelAccess::get_impl(input));

    // Normalize/validate dim against the OUTPUT rank (ndim+1), mirroring the
    // CPU kernel. A negative or out-of-range dim would otherwise produce an
    // out-of-range vector::insert offset (UB).
    const int64_t in_ndim = input.ndim();
    const int64_t out_ndim = in_ndim + 1;
    if (dim < 0) dim += out_ndim;
    if (dim < 0 || dim > in_ndim) {
        throw std::runtime_error("unsqueeze: dimension out of range");
    }

    auto& r_shape = result.mutable_shape();
    auto& r_strides = result.mutable_strides();
    r_shape.insert(r_shape.begin() + dim, 1);

    // Compute stride for new dimension (use the normalized dim)
    int64_t new_stride = (dim < in_ndim) ? input.strides()[dim] : 1;
    r_strides.insert(r_strides.begin() + dim, new_stride);

    return result;
}

// Concatenation kernel - concatenate multiple tensors along a dimension
template<typename T>
__global__ void cat_kernel_impl(T** input_ptrs, T* output,
                                 const int64_t* input_shapes,
                                 const int64_t* output_shape,
                                 const int64_t* output_strides,
                                 const int64_t* offsets_at_dim,
                                 int64_t num_tensors, int64_t ndim,
                                 int64_t concat_dim, int64_t total_elements) {
    TENZOR_CUDA_KERNEL_LOOP(idx, total_elements) {
        // Convert linear output index to multi-dimensional coordinates
        int64_t temp_idx = idx;
        int64_t coords[TRANSFORM_DIM_META_MAX_RANK];  // audit F.12: lifted from 8 → 16

        for (int64_t d = ndim - 1; d >= 0; --d) {
            coords[d] = temp_idx % output_shape[d];
            temp_idx /= output_shape[d];
        }

        // Determine which input tensor this element comes from
        int64_t coord_at_concat_dim = coords[concat_dim];
        int64_t tensor_idx = 0;
        int64_t local_coord = coord_at_concat_dim;

        for (int64_t t = 0; t < num_tensors; ++t) {
            int64_t tensor_size_at_dim = input_shapes[t * ndim + concat_dim];
            if (local_coord < tensor_size_at_dim) {
                tensor_idx = t;
                break;
            }
            local_coord -= tensor_size_at_dim;
        }

        // Calculate source index in the selected input tensor
        coords[concat_dim] = local_coord;
        int64_t src_idx = 0;

        for (int64_t d = 0; d < ndim; ++d) {
            int64_t stride = 1;
            for (int64_t i = d + 1; i < ndim; ++i) {
                stride *= input_shapes[tensor_idx * ndim + i];
            }
            src_idx += coords[d] * stride;
        }

        output[idx] = input_ptrs[tensor_idx][src_idx];
    }
}

auto cat_kernel(std::span<const Tensor> tensors, int64_t dim, cudaStream_t stream) -> Tensor {
    if (tensors.empty()) {
        throw std::invalid_argument("Cannot concatenate empty tensor list");
    }

    if (tensors.size() == 1) {
        return tensors[0];
    }

    // Validate inputs and normalize dimension
    const int64_t ndim = tensors[0].ndim();
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Dimension out of range for concatenation");
    }
    if (ndim > TRANSFORM_DIM_META_MAX_RANK) {
        throw std::runtime_error(
            "CUDA cat_kernel: tensor rank " + std::to_string(ndim) +
            " exceeds maximum " + std::to_string(TRANSFORM_DIM_META_MAX_RANK) +
            " (audit F.12; raise TRANSFORM_DIM_META_MAX_RANK if needed).");
    }

    // Make all tensors contiguous
    std::vector<Tensor> contiguous_tensors;
    contiguous_tensors.reserve(tensors.size());
    for (const auto& t : tensors) {
        if (t.ndim() != ndim) {
            throw std::invalid_argument("All tensors must have the same number of dimensions");
        }
        contiguous_tensors.push_back(t.is_contiguous() ? t : contiguous_kernel(t, stream));
    }

    // Validate shapes match except at concat dimension
    auto first_shape = contiguous_tensors[0].shape();
    int64_t total_size_at_dim = 0;

    for (const auto& t : contiguous_tensors) {
        auto shape = t.shape();
        for (int64_t i = 0; i < ndim; ++i) {
            if (i != dim && shape[i] != first_shape[i]) {
                throw std::invalid_argument(
                    "All tensors must have the same shape except in concatenation dimension");
            }
        }
        total_size_at_dim += shape[dim];
    }

    // Create output shape
    std::vector<int64_t> output_shape(first_shape.begin(), first_shape.end());
    output_shape[dim] = total_size_at_dim;

    // Create output tensor
    Tensor output(output_shape, tensors[0].dtype(), tensors[0].device());

    const int64_t total_elements = output.numel();
    if (total_elements == 0) {
        return output;  // Empty output
    }

    // Allocate device memory for metadata — packed into single contiguous buffer
    const int64_t num_tensors = contiguous_tensors.size();

    // Compute sizes for each metadata section
    const size_t ptrs_bytes    = num_tensors * sizeof(void*);
    const size_t shapes_bytes  = num_tensors * ndim * sizeof(int64_t);
    const size_t oshape_bytes  = ndim * sizeof(int64_t);
    const size_t ostride_bytes = ndim * sizeof(int64_t);
    const size_t offsets_bytes = num_tensors * sizeof(int64_t);
    const size_t total_meta_bytes = ptrs_bytes + shapes_bytes + oshape_bytes + ostride_bytes + offsets_bytes;

    // Build packed host buffer
    std::vector<char> host_meta(total_meta_bytes);
    size_t meta_offset = 0;

    // Section 1: input pointers
    {
        void** dst = reinterpret_cast<void**>(host_meta.data() + meta_offset);
        for (size_t i = 0; i < num_tensors; ++i) {
            dst[i] = contiguous_tensors[i].data_ptr();
        }
        meta_offset += ptrs_bytes;
    }

    // Section 2: input shapes
    {
        int64_t* dst = reinterpret_cast<int64_t*>(host_meta.data() + meta_offset);
        for (size_t t = 0; t < num_tensors; ++t) {
            auto shape = contiguous_tensors[t].shape();
            for (int64_t d = 0; d < ndim; ++d) {
                dst[t * ndim + d] = shape[d];
            }
        }
        meta_offset += shapes_bytes;
    }

    // Section 3: output shape
    std::memcpy(host_meta.data() + meta_offset, output_shape.data(), oshape_bytes);
    meta_offset += oshape_bytes;

    // Section 4: output strides
    std::vector<int64_t> output_strides = compute_strides(output_shape);
    std::memcpy(host_meta.data() + meta_offset, output_strides.data(), ostride_bytes);
    meta_offset += ostride_bytes;

    // Section 5: offsets
    {
        int64_t* dst = reinterpret_cast<int64_t*>(host_meta.data() + meta_offset);
        int64_t off = 0;
        for (size_t t = 0; t < num_tensors; ++t) {
            dst[t] = off;
            off += contiguous_tensors[t].shape()[dim];
        }
    }

    // Single device allocation and transfer (RAII guard handles cleanup)
    backend::CachedMemoryGuard meta_guard(total_meta_bytes);
    auto* d_meta = static_cast<char*>(meta_guard.get());
    CUDA_CHECK(cudaMemcpyAsync(d_meta, host_meta.data(), total_meta_bytes, cudaMemcpyHostToDevice, stream));

    // Compute sub-pointers as offsets into the packed buffer
    void** d_input_ptrs       = reinterpret_cast<void**>(d_meta);
    int64_t* d_input_shapes   = reinterpret_cast<int64_t*>(d_meta + ptrs_bytes);
    int64_t* d_output_shape   = reinterpret_cast<int64_t*>(d_meta + ptrs_bytes + shapes_bytes);
    int64_t* d_output_strides = reinterpret_cast<int64_t*>(d_meta + ptrs_bytes + shapes_bytes + oshape_bytes);
    int64_t* d_offsets        = reinterpret_cast<int64_t*>(d_meta + ptrs_bytes + shapes_bytes + oshape_bytes + ostride_bytes);

    // Handle empty output case - don't launch kernel with 0 blocks
    if (total_elements == 0) {
        return output;
    }

    // Launch kernel
    const int num_blocks = get_num_blocks(total_elements);

    if (tensors[0].dtype() == DType::Float32) {
        cat_kernel_impl<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<float**>(d_input_ptrs), output.data<float>(),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (tensors[0].dtype() == DType::Float64) {
        cat_kernel_impl<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<double**>(d_input_ptrs), output.data<double>(),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (tensors[0].dtype() == DType::Int32) {
        cat_kernel_impl<int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<int32_t**>(d_input_ptrs), output.data<int32_t>(),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (tensors[0].dtype() == DType::Int64) {
        cat_kernel_impl<int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<int64_t**>(d_input_ptrs), output.data<int64_t>(),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (tensors[0].dtype() == DType::Float16) {
        cat_kernel_impl<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<__half**>(d_input_ptrs), reinterpret_cast<__half*>(output.data_ptr()),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (tensors[0].dtype() == DType::BFloat16) {
        cat_kernel_impl<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16**>(d_input_ptrs), reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (tensors[0].dtype() == DType::Int8) {
        cat_kernel_impl<int8_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<int8_t**>(d_input_ptrs), output.data<int8_t>(),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (tensors[0].dtype() == DType::UInt8) {
        cat_kernel_impl<uint8_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<uint8_t**>(d_input_ptrs), output.data<uint8_t>(),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (tensors[0].dtype() == DType::Bool) {
        cat_kernel_impl<bool><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<bool**>(d_input_ptrs), reinterpret_cast<bool*>(output.data_ptr()),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (tensors[0].dtype() == DType::Complex64) {
        // Concatenation is pure data movement; treat each complex element as a
        // float2 (real,imag) so it copies correctly (matches contiguous_kernel).
        cat_kernel_impl<float2><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<float2**>(d_input_ptrs), reinterpret_cast<float2*>(output.data_ptr()),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else if (tensors[0].dtype() == DType::Complex128) {
        cat_kernel_impl<double2><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<double2**>(d_input_ptrs), reinterpret_cast<double2*>(output.data_ptr()),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
            CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("Concatenation: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());

    return output;
}

// Repeat kernel - repeat tensor elements along dimensions
template<typename T>
__global__ void repeat_kernel_device(
    const T* input, T* output,
    const int64_t* input_shape, const int64_t* input_strides,
    const int64_t* repeats, int64_t ndim, int64_t n) {

    TENZOR_CUDA_KERNEL_LOOP(out_idx, n) {
        // Calculate output coordinates
        int64_t temp = out_idx;
        int64_t in_idx = 0;

        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t out_shape_i = input_shape[i] * repeats[i];
            int64_t out_coord = temp % out_shape_i;
            temp /= out_shape_i;

            // Map output coordinate to input coordinate using TILE semantics
            // (torch.Tensor.repeat / torch.tile): [a,b] repeated 2x -> [a,b,a,b].
            // This matches the CPU reference (in_coord = out_coord % input_shape).
            int64_t in_coord = out_coord % input_shape[i];
            in_idx += in_coord * input_strides[i];
        }

        output[out_idx] = input[in_idx];
    }
}

auto repeat_kernel(const Tensor& input_in, const std::vector<int64_t>& repeats, cudaStream_t stream) -> Tensor {
    // The kernel below computes contiguous strides from input shape; any
    // non-contiguous input (e.g., permute view) would be read with the wrong
    // offsets and produce garbage. Materialize to contiguous first.
    Tensor input = input_in.is_contiguous() ? input_in : input_in.contiguous();
    auto input_shape_span = input.shape();
    std::vector<int64_t> input_shape(input_shape_span.begin(), input_shape_span.end());
    int64_t ndim = input_shape.size();

    // Calculate output shape
    std::vector<int64_t> output_shape(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        output_shape[i] = input_shape[i] * repeats[i];
    }

    // Create output tensor
    Tensor output(output_shape, input.dtype(), input.device());
    int64_t n = output.numel();

    if (n == 0) {
        return output;
    }

    // Calculate input strides
    std::vector<int64_t> input_strides(ndim);
    int64_t stride = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
        input_strides[i] = stride;
        stride *= input_shape[i];
    }

    // Pack 3 metadata arrays into single contiguous buffer
    const size_t array_bytes = ndim * sizeof(int64_t);
    const size_t total_meta_bytes = 3 * array_bytes;

    std::vector<char> host_meta(total_meta_bytes);
    std::memcpy(host_meta.data(), input_shape.data(), array_bytes);
    std::memcpy(host_meta.data() + array_bytes, input_strides.data(), array_bytes);
    std::memcpy(host_meta.data() + 2 * array_bytes, repeats.data(), array_bytes);

    backend::CachedMemoryGuard meta_guard(total_meta_bytes);
    auto* d_meta = static_cast<char*>(meta_guard.get());
    CUDA_CHECK(cudaMemcpyAsync(d_meta, host_meta.data(), total_meta_bytes, cudaMemcpyHostToDevice, stream));

    int64_t* d_input_shape   = reinterpret_cast<int64_t*>(d_meta);
    int64_t* d_input_strides = reinterpret_cast<int64_t*>(d_meta + array_bytes);
    int64_t* d_repeats       = reinterpret_cast<int64_t*>(d_meta + 2 * array_bytes);

    // Launch kernel
    if (input.dtype() == DType::Float32) {
        auto [grid, block] = optimal_launch_config(repeat_kernel_device<float>, n);
        repeat_kernel_device<<<grid, block, 0, stream>>>(
            input.data<float>(), output.data<float>(),
            d_input_shape, d_input_strides, d_repeats, ndim, n);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float64) {
        auto [grid, block] = optimal_launch_config(repeat_kernel_device<double>, n);
        repeat_kernel_device<<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(),
            d_input_shape, d_input_strides, d_repeats, ndim, n);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Float16) {
        auto [grid, block] = optimal_launch_config(repeat_kernel_device<Float16>, n);
        repeat_kernel_device<<<grid, block, 0, stream>>>(
            input.data<Float16>(), output.data<Float16>(),
            d_input_shape, d_input_strides, d_repeats, ndim, n);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::BFloat16) {
        auto [grid, block] = optimal_launch_config(repeat_kernel_device<__nv_bfloat16>, n);
        repeat_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            d_input_shape, d_input_strides, d_repeats, ndim, n);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int32) {
        auto [grid, block] = optimal_launch_config(repeat_kernel_device<int32_t>, n);
        repeat_kernel_device<<<grid, block, 0, stream>>>(
            input.data<int32_t>(), output.data<int32_t>(),
            d_input_shape, d_input_strides, d_repeats, ndim, n);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int64) {
        auto [grid, block] = optimal_launch_config(repeat_kernel_device<int64_t>, n);
        repeat_kernel_device<<<grid, block, 0, stream>>>(
            input.data<int64_t>(), output.data<int64_t>(),
            d_input_shape, d_input_strides, d_repeats, ndim, n);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int16) {
        auto [grid, block] = optimal_launch_config(repeat_kernel_device<int16_t>, n);
        repeat_kernel_device<<<grid, block, 0, stream>>>(
            input.data<int16_t>(), output.data<int16_t>(),
            d_input_shape, d_input_strides, d_repeats, ndim, n);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Int8) {
        auto [grid, block] = optimal_launch_config(repeat_kernel_device<int8_t>, n);
        repeat_kernel_device<<<grid, block, 0, stream>>>(
            input.data<int8_t>(), output.data<int8_t>(),
            d_input_shape, d_input_strides, d_repeats, ndim, n);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::UInt8) {
        auto [grid, block] = optimal_launch_config(repeat_kernel_device<uint8_t>, n);
        repeat_kernel_device<<<grid, block, 0, stream>>>(
            input.data<uint8_t>(), output.data<uint8_t>(),
            d_input_shape, d_input_strides, d_repeats, ndim, n);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::UInt16) {
        auto [grid, block] = optimal_launch_config(repeat_kernel_device<uint16_t>, n);
        repeat_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(input.data_ptr()),
            reinterpret_cast<uint16_t*>(output.data_ptr()),
            d_input_shape, d_input_strides, d_repeats, ndim, n);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::UInt32) {
        auto [grid, block] = optimal_launch_config(repeat_kernel_device<uint32_t>, n);
        repeat_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint32_t*>(input.data_ptr()),
            reinterpret_cast<uint32_t*>(output.data_ptr()),
            d_input_shape, d_input_strides, d_repeats, ndim, n);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::UInt64) {
        auto [grid, block] = optimal_launch_config(repeat_kernel_device<uint64_t>, n);
        repeat_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint64_t*>(input.data_ptr()),
            reinterpret_cast<uint64_t*>(output.data_ptr()),
            d_input_shape, d_input_strides, d_repeats, ndim, n);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Bool) {
        auto [grid, block] = optimal_launch_config(repeat_kernel_device<bool>, n);
        repeat_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<const bool*>(input.data_ptr()),
            reinterpret_cast<bool*>(output.data_ptr()),
            d_input_shape, d_input_strides, d_repeats, ndim, n);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Complex64) {
        auto [grid, block] = optimal_launch_config(repeat_kernel_device<float2>, n);
        repeat_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float2*>(input.data_ptr()),
            reinterpret_cast<float2*>(output.data_ptr()),
            d_input_shape, d_input_strides, d_repeats, ndim, n);
            CUDA_CHECK(cudaGetLastError());
    } else if (input.dtype() == DType::Complex128) {
        auto [grid, block] = optimal_launch_config(repeat_kernel_device<double2>, n);
        repeat_kernel_device<<<grid, block, 0, stream>>>(
            reinterpret_cast<const double2*>(input.data_ptr()),
            reinterpret_cast<double2*>(output.data_ptr()),
            d_input_shape, d_input_strides, d_repeats, ndim, n);
            CUDA_CHECK(cudaGetLastError());
    } else {
        throw std::runtime_error("repeat operation: unsupported dtype");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in repeat_kernel: ") + cudaGetErrorString(err));
    }

    return output;
}


// ============================================================================
// Flatten (reshape wrapper)
// ============================================================================

auto flatten_kernel(const Tensor& input, int64_t start_dim, int64_t end_dim, cudaStream_t stream) -> Tensor {
    auto shape = input.shape();
    int64_t ndim = shape.size();

    // Normalize negative dimensions
    if (start_dim < 0) start_dim += ndim;
    if (end_dim < 0) end_dim += ndim;

    // Compute new shape
    std::vector<int64_t> new_shape;
    for (int64_t i = 0; i < start_dim; ++i) {
        new_shape.push_back(shape[i]);
    }
    int64_t flat_size = 1;
    for (int64_t i = start_dim; i <= end_dim; ++i) {
        flat_size *= shape[i];
    }
    new_shape.push_back(flat_size);
    for (int64_t i = end_dim + 1; i < ndim; ++i) {
        new_shape.push_back(shape[i]);
    }

    return reshape_kernel(input, new_shape, stream);
}

// ============================================================================
// Slice
// ============================================================================

// Metadata struct passed by value to slice kernel (avoids 5x cudaMalloc for shape/stride arrays).
//
// Audit F.12: rank cap lifted from 8 to TRANSFORM_DIM_META_MAX_RANK (16).
// Previously the slice setup loop guarded `d < 8` and silently truncated
// higher-rank tensors. The host dispatcher (slice_kernel) now throws when
// ndim exceeds the cap.
struct SliceMeta {
    int64_t input_strides[TRANSFORM_DIM_META_MAX_RANK];
    int64_t output_shape[TRANSFORM_DIM_META_MAX_RANK];
    int64_t output_strides[TRANSFORM_DIM_META_MAX_RANK];
    int64_t starts[TRANSFORM_DIM_META_MAX_RANK];
    int64_t steps[TRANSFORM_DIM_META_MAX_RANK];
};

template<typename T>
__global__ void slice_kernel_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    SliceMeta meta,
    int64_t ndim,
    int64_t total_elements) {

    // Grid-stride loop so an output with > INT_MAX*BLOCK elements (a clamped
    // grid that cannot cover all elements in one pass) is fully written rather
    // than leaving the tail uninitialized.
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        // Convert linear output index to multi-dimensional index
        int64_t input_offset = 0;
        int64_t remaining = idx;
        for (int64_t d = 0; d < ndim; ++d) {
            int64_t dim_idx = remaining / meta.output_strides[d];
            remaining %= meta.output_strides[d];
            // Map output index to input index: input_idx = start + output_idx * step
            int64_t input_dim_idx = meta.starts[d] + dim_idx * meta.steps[d];
            input_offset += input_dim_idx * meta.input_strides[d];
        }

        output[idx] = input[input_offset];
    }
}

auto slice_kernel(
    const Tensor& input_orig,
    const std::vector<int64_t>& starts,
    const std::vector<int64_t>& ends,
    const std::vector<int64_t>& steps,
    cudaStream_t stream
) -> Tensor {
    // slice synthesizes input strides from shape below, which is only valid for
    // contiguous input; materialize a contiguous copy so a non-contiguous view
    // is sliced correctly.
    Tensor input_contig;
    const Tensor& input = input_orig.is_contiguous()
        ? input_orig
        : (input_contig = input_orig.contiguous());
    auto input_shape = input.shape();
    int64_t ndim = input_shape.size();

    // Compute output shape
    std::vector<int64_t> output_shape(ndim);
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t s = (d < static_cast<int64_t>(starts.size())) ? starts[d] : 0;
        int64_t e = (d < static_cast<int64_t>(ends.size())) ? ends[d] : input_shape[d];
        int64_t st = (d < static_cast<int64_t>(steps.size())) ? steps[d] : 1;

        // Normalize negative indices
        if (s < 0) s += input_shape[d];
        if (e < 0) e += input_shape[d];
        s = std::max(int64_t(0), std::min(s, input_shape[d]));
        e = std::max(int64_t(0), std::min(e, input_shape[d]));

        output_shape[d] = (e - s + st - 1) / st;
        if (output_shape[d] < 0) output_shape[d] = 0;
    }

    Tensor output(output_shape, input.dtype(), input.device());
    int64_t total = output.numel();
    if (total == 0) return output;

    // Compute strides
    std::vector<int64_t> input_strides(ndim);
    std::vector<int64_t> output_strides(ndim);
    input_strides[ndim - 1] = 1;
    output_strides[ndim - 1] = 1;
    for (int64_t d = ndim - 2; d >= 0; --d) {
        input_strides[d] = input_strides[d + 1] * input_shape[d + 1];
        output_strides[d] = output_strides[d + 1] * output_shape[d + 1];
    }

    // Pad starts and steps to ndim
    std::vector<int64_t> padded_starts(ndim, 0);
    std::vector<int64_t> padded_steps(ndim, 1);
    for (int64_t d = 0; d < ndim; ++d) {
        if (d < static_cast<int64_t>(starts.size())) {
            padded_starts[d] = starts[d];
            if (padded_starts[d] < 0) padded_starts[d] += input_shape[d];
            padded_starts[d] = std::max(int64_t(0), std::min(padded_starts[d], input_shape[d]));
        }
        if (d < static_cast<int64_t>(steps.size())) {
            padded_steps[d] = steps[d];
        }
    }

    // Build metadata struct passed by value (avoids 5x cudaMalloc).
    // Audit F.12: enforce the rank cap explicitly instead of silently truncating.
    if (ndim > TRANSFORM_DIM_META_MAX_RANK) {
        throw std::runtime_error(
            "CUDA slice_kernel: tensor rank " + std::to_string(ndim) +
            " exceeds maximum " + std::to_string(TRANSFORM_DIM_META_MAX_RANK) +
            " (raise TRANSFORM_DIM_META_MAX_RANK if needed).");
    }
    SliceMeta meta{};
    for (int64_t d = 0; d < ndim; ++d) {
        meta.input_strides[d] = input_strides[d];
        meta.output_shape[d] = output_shape[d];
        meta.output_strides[d] = output_strides[d];
        meta.starts[d] = padded_starts[d];
        meta.steps[d] = padded_steps[d];
    }

    int block_size = 256;
    // get_num_blocks clamps the grid to the device max; combined with the
    // kernel's grid-stride loop this covers an arbitrarily large int64 total
    // without narrowing to a negative/garbage int block count.
    int num_blocks = get_num_blocks(total, block_size);

    // slice is pure data movement (index-based element copy), so dispatch by
    // element size like tile/contiguous/cat rather than by semantic dtype.
    // This makes slice — and split/chunk along non-zero dims and strided
    // slices that fall back to it — work for every dtype (BFloat16, Int8,
    // UInt8, Bool, Int16, UInt16/32/64, Complex64/128) just like the CPU
    // reference, instead of throwing for the unlisted dtypes.
    auto launch_slice = [&](auto* tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        slice_kernel_impl<T><<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<const T*>(input.data_ptr()),
            reinterpret_cast<T*>(output.data_ptr()), meta, ndim, total);
        CUDA_CHECK(cudaGetLastError());
    };
    switch (dtype_size(input.dtype())) {
        case 1:  launch_slice(static_cast<uint8_t*>(nullptr));  break;   // Int8, UInt8, Bool
        case 2:  launch_slice(static_cast<uint16_t*>(nullptr)); break;   // Float16, BFloat16, Int16, UInt16
        case 4:  launch_slice(static_cast<uint32_t*>(nullptr)); break;   // Float32, Int32, UInt32
        case 8:  launch_slice(static_cast<uint64_t*>(nullptr)); break;   // Float64, Int64, UInt64, Complex64
        case 16: launch_slice(static_cast<double2*>(nullptr));  break;   // Complex128
        default:
            throw std::runtime_error("slice: unsupported element size");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in slice_kernel: ") + cudaGetErrorString(err));
    }

    return output;
}

// ============================================================================
// Stack (unsqueeze + cat)
// ============================================================================

auto stack_kernel(std::span<const Tensor> tensors, int64_t dim, cudaStream_t stream) -> Tensor {
    if (tensors.empty()) {
        throw std::runtime_error("stack: expected non-empty tensor list");
    }

    // Unsqueeze each tensor at dim, then cat
    std::vector<Tensor> unsqueezed;
    unsqueezed.reserve(tensors.size());
    for (const auto& t : tensors) {
        unsqueezed.push_back(unsqueeze_kernel(t, dim, stream));
    }

    return cat_kernel(std::span<const Tensor>(unsqueezed.data(), unsqueezed.size()), dim, stream);
}

// ============================================================================
// Split / Chunk (slice-based)
// ============================================================================

auto split_kernel(const Tensor& input, int64_t split_size, int64_t dim, cudaStream_t stream) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    int64_t dim_size = shape[dim];
    int64_t num_splits = (dim_size + split_size - 1) / split_size;

    std::vector<Tensor> results;
    results.reserve(num_splits);

    // Fast path: when splitting along the outermost dimension of a contiguous tensor,
    // use cudaMemcpyAsync instead of launching N separate slice kernels
    if (dim == 0 && input.is_contiguous()) {
        size_t elem_size = dtype_size(input.dtype());
        int64_t inner_elements = 1;
        for (int64_t d = 1; d < ndim; ++d) {
            inner_elements *= shape[d];
        }
        size_t chunk_stride = inner_elements * elem_size;

        for (int64_t i = 0; i < num_splits; ++i) {
            int64_t start = i * split_size;
            int64_t actual_size = std::min(split_size, dim_size - start);

            std::vector<int64_t> out_shape(shape.begin(), shape.end());
            out_shape[0] = actual_size;

            Tensor chunk(out_shape, input.dtype(), input.device());
            // audit V.20: per-iteration cudaMemcpyAsync errors were dropped;
            // surface them via TENZOR_CUDA_CHECK so a mid-loop failure aborts.
            TENZOR_CUDA_CHECK(cudaMemcpyAsync(
                chunk.data_ptr(),
                static_cast<const char*>(input.data_ptr()) + start * chunk_stride,
                actual_size * chunk_stride,
                cudaMemcpyDeviceToDevice,
                stream));
            results.push_back(std::move(chunk));
        }
        return results;
    }

    // General path: use slice_kernel for each split. Materialize a single
    // contiguous copy up front so a non-contiguous input is not re-contiguated
    // inside slice_kernel on every iteration (O(num_splits*numel) extra D2D
    // traffic); slice_kernel then takes the already-contiguous tensor directly.
    Tensor cont = input.is_contiguous() ? input : contiguous_kernel(input, stream);
    for (int64_t i = 0; i < num_splits; ++i) {
        int64_t start = i * split_size;
        int64_t end = std::min(start + split_size, dim_size);

        std::vector<int64_t> starts(ndim, 0);
        std::vector<int64_t> ends(shape.begin(), shape.end());
        std::vector<int64_t> steps(ndim, 1);
        starts[dim] = start;
        ends[dim] = end;

        results.push_back(slice_kernel(cont, starts, ends, steps, stream));
    }

    return results;
}

auto chunk_kernel(const Tensor& input, int64_t chunks, int64_t dim, cudaStream_t stream) -> std::vector<Tensor> {
    auto shape = input.shape();
    if (dim < 0) dim += static_cast<int64_t>(shape.size());
    int64_t dim_size = shape[dim];
    int64_t split_size = (dim_size + chunks - 1) / chunks;
    return split_kernel(input, split_size, dim, stream);
}

// ============================================================================
// Tile (repeat wrapper)
// ============================================================================

// True numpy-style tile: the WHOLE tensor is tiled `reps` times along each dim
// (source index = out_coord % in_dim), distinct from `repeat` which duplicates
// each element. Previously tile delegated to repeat_kernel — wrong values.
template<typename T>
__global__ void tile_kernel_device(const T* __restrict__ in, T* __restrict__ out,
        int64_t out_numel, int ndim,
        const int64_t* __restrict__ out_shape,
        const int64_t* __restrict__ in_shape,
        const int64_t* __restrict__ in_strides) {
    TENZOR_CUDA_KERNEL_LOOP(idx, out_numel) {
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

auto tile_kernel(const Tensor& input, const std::vector<int64_t>& reps, cudaStream_t stream) -> Tensor {
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
    CUDA_CHECK(cudaMallocAsync(&d_meta, meta.size() * sizeof(int64_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_meta, meta.data(), meta.size() * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream));
    const int64_t* d_out_shape = d_meta;
    const int64_t* d_in_shape = d_meta + out_ndim;
    const int64_t* d_in_strides = d_meta + 2 * out_ndim;

    int num_blocks = get_num_blocks(out_numel);
    auto launch = [&](auto* tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        tile_kernel_device<T><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const T*>(in.data_ptr()), reinterpret_cast<T*>(output.data_ptr()),
            out_numel, static_cast<int>(out_ndim), d_out_shape, d_in_shape, d_in_strides);
        CUDA_CHECK(cudaGetLastError());
    };
    // tile is pure data movement; dispatch by element size.
    switch (dtype_size(in.dtype())) {
        case 1:  launch(static_cast<uint8_t*>(nullptr));  break;
        case 2:  launch(static_cast<uint16_t*>(nullptr)); break;
        case 4:  launch(static_cast<uint32_t*>(nullptr)); break;
        case 8:  launch(static_cast<uint64_t*>(nullptr)); break;
        case 16: launch(static_cast<double2*>(nullptr));  break;
        default:
            CUDA_CHECK(cudaFreeAsync(d_meta, stream));
            throw std::runtime_error("tile: unsupported element size");
    }
    CUDA_CHECK(cudaFreeAsync(d_meta, stream));
    return output;
}

// ============================================================================
// Triu kernel — upper triangular: zero out elements below diagonal+k
// ============================================================================

template<typename T>
__global__ void triu_kernel_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t rows, int64_t cols,
    int64_t batch_size,
    int64_t diagonal
) {
    int64_t total = batch_size * rows * cols;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t rem = idx % (rows * cols);
        int64_t i = rem / cols;
        int64_t j = rem % cols;

        if (j >= i + diagonal) {
            output[idx] = input[idx];
        } else {
            output[idx] = T(0);
        }
    }
}

// Specialization for __half (no T(0))
template<>
__global__ void triu_kernel_impl<__half>(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int64_t rows, int64_t cols,
    int64_t batch_size,
    int64_t diagonal
) {
    int64_t total = batch_size * rows * cols;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t rem = idx % (rows * cols);
        int64_t i = rem / cols;
        int64_t j = rem % cols;
        output[idx] = (j >= i + diagonal) ? input[idx] : __float2half(0.0f);
    }
}

// Specialization for __nv_bfloat16
template<>
__global__ void triu_kernel_impl<__nv_bfloat16>(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int64_t rows, int64_t cols,
    int64_t batch_size,
    int64_t diagonal
) {
    int64_t total = batch_size * rows * cols;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t rem = idx % (rows * cols);
        int64_t i = rem / cols;
        int64_t j = rem % cols;
        output[idx] = (j >= i + diagonal) ? input[idx] : __float2bfloat16(0.0f);
    }
}

auto triu_kernel(const Tensor& input, int64_t diagonal, cudaStream_t stream) -> Tensor {
    if (input.ndim() < 2) {
        throw std::runtime_error("triu: input must be at least 2D");
    }

    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor cont = input.is_contiguous() ? input : contiguous_kernel(input, stream);
    Tensor output(shape, input.dtype(), input.device());

    int64_t rows = shape[shape.size() - 2];
    int64_t cols = shape[shape.size() - 1];
    int64_t batch_size = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        batch_size *= shape[i];
    }

    int64_t total = batch_size * rows * cols;
    if (total == 0) return output;

    switch (input.dtype()) {
        case DType::Float32: {
            auto [grid, block] = optimal_launch_config(triu_kernel_impl<float>, total);
            triu_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<float>(), output.data<float>(), rows, cols, batch_size, diagonal);
            break;
        }
        case DType::Float64: {
            auto [grid, block] = optimal_launch_config(triu_kernel_impl<double>, total);
            triu_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<double>(), output.data<double>(), rows, cols, batch_size, diagonal);
            break;
        }
        case DType::Float16: {
            auto [grid, block] = optimal_launch_config(triu_kernel_impl<__half>, total);
            triu_kernel_impl<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(cont.data_ptr()),
                reinterpret_cast<__half*>(output.data_ptr()),
                rows, cols, batch_size, diagonal);
            break;
        }
        case DType::BFloat16: {
            auto [grid, block] = optimal_launch_config(triu_kernel_impl<__nv_bfloat16>, total);
            triu_kernel_impl<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(cont.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                rows, cols, batch_size, diagonal);
            break;
        }
        case DType::Int32: {
            auto [grid, block] = optimal_launch_config(triu_kernel_impl<int32_t>, total);
            triu_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<int32_t>(), output.data<int32_t>(), rows, cols, batch_size, diagonal);
            break;
        }
        case DType::Int64: {
            auto [grid, block] = optimal_launch_config(triu_kernel_impl<int64_t>, total);
            triu_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<int64_t>(), output.data<int64_t>(), rows, cols, batch_size, diagonal);
            break;
        }
        default:
            throw std::runtime_error("triu_kernel: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// Tril kernel — lower triangular: zero out elements above diagonal+k
// ============================================================================

template<typename T>
__global__ void tril_kernel_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t rows, int64_t cols,
    int64_t batch_size,
    int64_t diagonal
) {
    int64_t total = batch_size * rows * cols;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t rem = idx % (rows * cols);
        int64_t i = rem / cols;
        int64_t j = rem % cols;

        if (j <= i + diagonal) {
            output[idx] = input[idx];
        } else {
            output[idx] = T(0);
        }
    }
}

template<>
__global__ void tril_kernel_impl<__half>(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int64_t rows, int64_t cols,
    int64_t batch_size,
    int64_t diagonal
) {
    int64_t total = batch_size * rows * cols;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t rem = idx % (rows * cols);
        int64_t i = rem / cols;
        int64_t j = rem % cols;
        output[idx] = (j <= i + diagonal) ? input[idx] : __float2half(0.0f);
    }
}

template<>
__global__ void tril_kernel_impl<__nv_bfloat16>(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int64_t rows, int64_t cols,
    int64_t batch_size,
    int64_t diagonal
) {
    int64_t total = batch_size * rows * cols;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t rem = idx % (rows * cols);
        int64_t i = rem / cols;
        int64_t j = rem % cols;
        output[idx] = (j <= i + diagonal) ? input[idx] : __float2bfloat16(0.0f);
    }
}

auto tril_kernel(const Tensor& input, int64_t diagonal, cudaStream_t stream) -> Tensor {
    if (input.ndim() < 2) {
        throw std::runtime_error("tril: input must be at least 2D");
    }

    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor cont = input.is_contiguous() ? input : contiguous_kernel(input, stream);
    Tensor output(shape, input.dtype(), input.device());

    int64_t rows = shape[shape.size() - 2];
    int64_t cols = shape[shape.size() - 1];
    int64_t batch_size = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        batch_size *= shape[i];
    }

    int64_t total = batch_size * rows * cols;
    if (total == 0) return output;

    switch (input.dtype()) {
        case DType::Float32: {
            auto [grid, block] = optimal_launch_config(tril_kernel_impl<float>, total);
            tril_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<float>(), output.data<float>(), rows, cols, batch_size, diagonal);
            break;
        }
        case DType::Float64: {
            auto [grid, block] = optimal_launch_config(tril_kernel_impl<double>, total);
            tril_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<double>(), output.data<double>(), rows, cols, batch_size, diagonal);
            break;
        }
        case DType::Float16: {
            auto [grid, block] = optimal_launch_config(tril_kernel_impl<__half>, total);
            tril_kernel_impl<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(cont.data_ptr()),
                reinterpret_cast<__half*>(output.data_ptr()),
                rows, cols, batch_size, diagonal);
            break;
        }
        case DType::BFloat16: {
            auto [grid, block] = optimal_launch_config(tril_kernel_impl<__nv_bfloat16>, total);
            tril_kernel_impl<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(cont.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                rows, cols, batch_size, diagonal);
            break;
        }
        case DType::Int32: {
            auto [grid, block] = optimal_launch_config(tril_kernel_impl<int32_t>, total);
            tril_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<int32_t>(), output.data<int32_t>(), rows, cols, batch_size, diagonal);
            break;
        }
        case DType::Int64: {
            auto [grid, block] = optimal_launch_config(tril_kernel_impl<int64_t>, total);
            tril_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<int64_t>(), output.data<int64_t>(), rows, cols, batch_size, diagonal);
            break;
        }
        default:
            throw std::runtime_error("tril_kernel: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// Diag kernel — extract diagonal from 2D or construct diagonal matrix from 1D
// ============================================================================

// Extract diagonal from 2D matrix
template<typename T>
__global__ void diag_extract_kernel_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t diag_size,
    int64_t rows, int64_t cols,
    int64_t diagonal
) {
    TENZOR_CUDA_KERNEL_LOOP(idx, diag_size) {
        int64_t i, j;
        if (diagonal >= 0) {
            i = idx;
            j = idx + diagonal;
        } else {
            i = idx - diagonal;
            j = idx;
        }
        output[idx] = input[i * cols + j];
    }
}

// Construct diagonal matrix from 1D vector
template<typename T>
__global__ void diag_construct_kernel_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t n,         // size of output matrix (n x n)
    int64_t diag_size, // number of diagonal elements
    int64_t diagonal
) {
    int64_t total = n * n;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t i = idx / n;
        int64_t j = idx % n;

        int64_t diag_idx;
        if (diagonal >= 0) {
            diag_idx = j - diagonal;  // element on diagonal if i == diag_idx
        } else {
            diag_idx = i + diagonal;  // element on diagonal if j == diag_idx
        }

        if (diagonal >= 0 && i == j - diagonal && diag_idx >= 0 && diag_idx < diag_size) {
            output[idx] = input[diag_idx];
        } else if (diagonal < 0 && j == i + diagonal && diag_idx >= 0 && diag_idx < diag_size) {
            output[idx] = input[diag_idx];
        } else {
            output[idx] = T(0);
        }
    }
}

// __half specialization for construct
template<>
__global__ void diag_construct_kernel_impl<__half>(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int64_t n,
    int64_t diag_size,
    int64_t diagonal
) {
    int64_t total = n * n;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t i = idx / n;
        int64_t j = idx % n;

        int64_t diag_idx;
        if (diagonal >= 0) {
            diag_idx = j - diagonal;
        } else {
            diag_idx = i + diagonal;
        }

        if (diagonal >= 0 && i == j - diagonal && diag_idx >= 0 && diag_idx < diag_size) {
            output[idx] = input[diag_idx];
        } else if (diagonal < 0 && j == i + diagonal && diag_idx >= 0 && diag_idx < diag_size) {
            output[idx] = input[diag_idx];
        } else {
            output[idx] = __float2half(0.0f);
        }
    }
}

// __nv_bfloat16 specialization for construct
template<>
__global__ void diag_construct_kernel_impl<__nv_bfloat16>(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int64_t n,
    int64_t diag_size,
    int64_t diagonal
) {
    int64_t total = n * n;
    TENZOR_CUDA_KERNEL_LOOP(idx, total) {
        int64_t i = idx / n;
        int64_t j = idx % n;

        int64_t diag_idx;
        if (diagonal >= 0) {
            diag_idx = j - diagonal;
        } else {
            diag_idx = i + diagonal;
        }

        if (diagonal >= 0 && i == j - diagonal && diag_idx >= 0 && diag_idx < diag_size) {
            output[idx] = input[diag_idx];
        } else if (diagonal < 0 && j == i + diagonal && diag_idx >= 0 && diag_idx < diag_size) {
            output[idx] = input[diag_idx];
        } else {
            output[idx] = __float2bfloat16(0.0f);
        }
    }
}

auto diag_kernel(const Tensor& input, int64_t diagonal, cudaStream_t stream) -> Tensor {
    Tensor cont = input.is_contiguous() ? input : contiguous_kernel(input, stream);
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (input.ndim() == 2) {
        // Extract diagonal from 2D matrix
        int64_t rows = shape[0];
        int64_t cols = shape[1];
        int64_t diag_size;
        if (diagonal >= 0) {
            diag_size = std::max<int64_t>(0, std::min(rows, cols - diagonal));
        } else {
            diag_size = std::max<int64_t>(0, std::min(rows + diagonal, cols));
        }

        Tensor output({diag_size}, input.dtype(), input.device());
        if (diag_size == 0) return output;

        switch (input.dtype()) {
            case DType::Float32: {
                auto [grid, block] = optimal_launch_config(diag_extract_kernel_impl<float>, diag_size);
                diag_extract_kernel_impl<<<grid, block, 0, stream>>>(
                    cont.data<float>(), output.data<float>(), diag_size, rows, cols, diagonal);
                break;
            }
            case DType::Float64: {
                auto [grid, block] = optimal_launch_config(diag_extract_kernel_impl<double>, diag_size);
                diag_extract_kernel_impl<<<grid, block, 0, stream>>>(
                    cont.data<double>(), output.data<double>(), diag_size, rows, cols, diagonal);
                break;
            }
            case DType::Float16: {
                auto [grid, block] = optimal_launch_config(diag_extract_kernel_impl<__half>, diag_size);
                diag_extract_kernel_impl<<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __half*>(cont.data_ptr()),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    diag_size, rows, cols, diagonal);
                break;
            }
            case DType::BFloat16: {
                auto [grid, block] = optimal_launch_config(diag_extract_kernel_impl<__nv_bfloat16>, diag_size);
                diag_extract_kernel_impl<<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(cont.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    diag_size, rows, cols, diagonal);
                break;
            }
            case DType::Int32: {
                auto [grid, block] = optimal_launch_config(diag_extract_kernel_impl<int32_t>, diag_size);
                diag_extract_kernel_impl<<<grid, block, 0, stream>>>(
                    cont.data<int32_t>(), output.data<int32_t>(), diag_size, rows, cols, diagonal);
                break;
            }
            case DType::Int64: {
                auto [grid, block] = optimal_launch_config(diag_extract_kernel_impl<int64_t>, diag_size);
                diag_extract_kernel_impl<<<grid, block, 0, stream>>>(
                    cont.data<int64_t>(), output.data<int64_t>(), diag_size, rows, cols, diagonal);
                break;
            }
            default:
                throw std::runtime_error("diag_kernel: unsupported dtype");
        }

        CUDA_CHECK(cudaGetLastError());
        return output;

    } else if (input.ndim() == 1) {
        // Construct diagonal matrix from 1D vector
        int64_t diag_size = shape[0];
        int64_t n = diag_size + std::abs(diagonal);
        int64_t total = n * n;

        Tensor output({n, n}, input.dtype(), input.device());
        if (total == 0) return output;

        switch (input.dtype()) {
            case DType::Float32: {
                auto [grid, block] = optimal_launch_config(diag_construct_kernel_impl<float>, total);
                diag_construct_kernel_impl<<<grid, block, 0, stream>>>(
                    cont.data<float>(), output.data<float>(), n, diag_size, diagonal);
                break;
            }
            case DType::Float64: {
                auto [grid, block] = optimal_launch_config(diag_construct_kernel_impl<double>, total);
                diag_construct_kernel_impl<<<grid, block, 0, stream>>>(
                    cont.data<double>(), output.data<double>(), n, diag_size, diagonal);
                break;
            }
            case DType::Float16: {
                auto [grid, block] = optimal_launch_config(diag_construct_kernel_impl<__half>, total);
                diag_construct_kernel_impl<<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __half*>(cont.data_ptr()),
                    reinterpret_cast<__half*>(output.data_ptr()),
                    n, diag_size, diagonal);
                break;
            }
            case DType::BFloat16: {
                auto [grid, block] = optimal_launch_config(diag_construct_kernel_impl<__nv_bfloat16>, total);
                diag_construct_kernel_impl<<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(cont.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                    n, diag_size, diagonal);
                break;
            }
            case DType::Int32: {
                auto [grid, block] = optimal_launch_config(diag_construct_kernel_impl<int32_t>, total);
                diag_construct_kernel_impl<<<grid, block, 0, stream>>>(
                    cont.data<int32_t>(), output.data<int32_t>(), n, diag_size, diagonal);
                break;
            }
            case DType::Int64: {
                auto [grid, block] = optimal_launch_config(diag_construct_kernel_impl<int64_t>, total);
                diag_construct_kernel_impl<<<grid, block, 0, stream>>>(
                    cont.data<int64_t>(), output.data<int64_t>(), n, diag_size, diagonal);
                break;
            }
            default:
                throw std::runtime_error("diag_kernel: unsupported dtype");
        }

        CUDA_CHECK(cudaGetLastError());
        return output;

    } else {
        throw std::runtime_error("diag_kernel: input must be 1D or 2D");
    }
}

// ============================================================================
// Trace kernel — sum of diagonal elements of a 2D matrix
// ============================================================================

// Accumulation type helper for trace: float for half types, T otherwise
template<typename T> struct TraceAccumType { using type = T; };
template<> struct TraceAccumType<__half> { using type = float; };
template<> struct TraceAccumType<__nv_bfloat16> { using type = float; };

// Trace kernel: sum diagonal elements from a 2D matrix (rows x cols)
// input layout is row-major: diagonal element i is at input[i * cols + i]
//
// When multiple blocks participate, each block's partial sum is written in the
// accumulator type (Acc) — not narrowed back to T — so that the final pass
// re-accumulates in full precision. For half/bf16 (Acc=float) this avoids
// reintroducing per-block rounding error that the float accumulator exists to
// prevent. For all other dtypes Acc == T, so the partial buffer layout is
// unchanged.
template<typename T>
__global__ void trace_diag_sum_kernel(
    const T* __restrict__ input,
    typename TraceAccumType<T>::type* output,
    int64_t diag_size,
    int64_t cols
) {
    using Acc = typename TraceAccumType<T>::type;
    extern __shared__ char trace_shared_raw[];
    Acc* shared = reinterpret_cast<Acc*>(trace_shared_raw);

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    Acc thread_sum = Acc(0);
    for (int64_t i = idx; i < diag_size; i += grid_size) {
        thread_sum = thread_sum + Acc(input[i * cols + i]);
    }

    shared[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = shared[tid] + shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Keep the partial in the accumulator type for full-precision re-sum.
        output[blockIdx.x] = shared[0];
    }
}

// Final pass of multi-block trace reduction. Reads the per-block partials in
// the accumulator type (Acc) and narrows to T only on the single scalar write.
template<typename T>
__global__ void trace_final_sum_kernel(
    const typename TraceAccumType<T>::type* __restrict__ input,
    T* output,
    int64_t n
) {
    using Acc = typename TraceAccumType<T>::type;
    extern __shared__ char trace_final_shared_raw[];
    Acc* shared = reinterpret_cast<Acc*>(trace_final_shared_raw);

    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    Acc thread_sum = Acc(0);
    for (int64_t i = idx; i < n; i += grid_size) {
        thread_sum = thread_sum + input[i];
    }

    shared[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = shared[tid] + shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = T(shared[0]);
    }
}

auto trace_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    if (input.ndim() != 2) {
        throw std::runtime_error("trace: input must be 2D");
    }

    Tensor cont = input.is_contiguous() ? input : contiguous_kernel(input, stream);
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t rows = shape[0];
    int64_t cols = shape[1];
    int64_t diag_size = std::min(rows, cols);

    // Output is a scalar tensor
    Tensor output({}, input.dtype(), input.device());

    if (diag_size == 0) {
        // Zero-fill output for empty matrix
        cudaMemsetAsync(output.data_ptr(), 0, output.numel() * dtype_size(input.dtype()), stream);
        return output;
    }

    const int block_size = 256;
    int num_blocks = std::min<int>((diag_size + block_size - 1) / block_size, 1024);

    // Helper macro to reduce boilerplate for trace dispatch.
    //
    // diag_sum now emits per-block partials in the accumulator type (Acc), so
    // inter-block partials are never narrowed back to T. For Acc == T (fp32,
    // fp64, int) the single-block case can still write straight to the output.
    // For Acc != T (half/bf16, Acc=float) we always run the two-pass form with
    // an Acc-typed temp and narrow to T only in the final scalar write, keeping
    // the reduction in full precision end to end.
    #define TRACE_DISPATCH(T, in_ptr, out_ptr, accum_size) \
        do { \
            using Acc = typename TraceAccumType<T>::type; \
            size_t smem = block_size * (accum_size); \
            if (num_blocks == 1 && std::is_same<Acc, T>::value) { \
                trace_diag_sum_kernel<T><<<1, block_size, smem, stream>>>( \
                    in_ptr, reinterpret_cast<Acc*>(out_ptr), diag_size, cols); \
            } else { \
                backend::CachedMemoryGuard d_temp_guard(num_blocks * sizeof(Acc)); \
                auto* d_temp = static_cast<Acc*>(d_temp_guard.get()); \
                trace_diag_sum_kernel<T><<<num_blocks, block_size, smem, stream>>>( \
                    in_ptr, d_temp, diag_size, cols); \
                CUDA_CHECK(cudaGetLastError()); \
                trace_final_sum_kernel<T><<<1, block_size, smem, stream>>>( \
                    d_temp, out_ptr, num_blocks); \
            } \
        } while(0)

    switch (input.dtype()) {
        case DType::Float32:
            TRACE_DISPATCH(float, cont.data<float>(), output.data<float>(), sizeof(float));
            break;
        case DType::Float64:
            TRACE_DISPATCH(double, cont.data<double>(), output.data<double>(), sizeof(double));
            break;
        case DType::Float16:
            TRACE_DISPATCH(__half,
                reinterpret_cast<const __half*>(cont.data_ptr()),
                reinterpret_cast<__half*>(output.data_ptr()),
                sizeof(float));  // AccumType is float
            break;
        case DType::BFloat16:
            TRACE_DISPATCH(__nv_bfloat16,
                reinterpret_cast<const __nv_bfloat16*>(cont.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                sizeof(float));  // AccumType is float
            break;
        case DType::Int32:
            TRACE_DISPATCH(int32_t, cont.data<int32_t>(), output.data<int32_t>(), sizeof(int32_t));
            break;
        case DType::Int64:
            TRACE_DISPATCH(int64_t, cont.data<int64_t>(), output.data<int64_t>(), sizeof(int64_t));
            break;
        default:
            throw std::runtime_error("trace_kernel: unsupported dtype");
    }

    #undef TRACE_DISPATCH

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// Flip kernel — reverse elements along a dimension
// ============================================================================

template<typename T>
__global__ void flip_kernel_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t total_elements,
    int64_t dim_size,
    int64_t inner_size  // product of sizes of dims after the flip dim
) {
    TENZOR_CUDA_KERNEL_LOOP(i, total_elements) {
        int64_t inner_idx = i % inner_size;
        int64_t dim_idx = (i / inner_size) % dim_size;
        int64_t outer_idx = i / (inner_size * dim_size);

        // Reverse the dim index
        int64_t reversed_dim_idx = dim_size - 1 - dim_idx;
        int64_t src_idx = (outer_idx * dim_size + reversed_dim_idx) * inner_size + inner_idx;

        output[i] = input[src_idx];
    }
}

auto flip_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor {
    auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    int64_t ndim = static_cast<int64_t>(shape.size());

    // Normalize dimension
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("flip: dimension " + std::to_string(dim) +
            " out of range for tensor with " + std::to_string(ndim) + " dimensions");
    }

    Tensor cont = input.is_contiguous() ? input : contiguous_kernel(input, stream);
    Tensor output(shape, input.dtype(), input.device());

    int64_t total = input.numel();
    if (total == 0) return output;

    int64_t dim_size = shape[dim];
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
        inner_size *= shape[d];
    }

    switch (input.dtype()) {
        case DType::Float32: {
            auto [grid, block] = optimal_launch_config(flip_kernel_impl<float>, total);
            flip_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<float>(), output.data<float>(), total, dim_size, inner_size);
            break;
        }
        case DType::Float64: {
            auto [grid, block] = optimal_launch_config(flip_kernel_impl<double>, total);
            flip_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<double>(), output.data<double>(), total, dim_size, inner_size);
            break;
        }
        case DType::Float16: {
            auto [grid, block] = optimal_launch_config(flip_kernel_impl<Float16>, total);
            flip_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<Float16>(), output.data<Float16>(), total, dim_size, inner_size);
            break;
        }
        case DType::BFloat16: {
            auto [grid, block] = optimal_launch_config(flip_kernel_impl<BFloat16>, total);
            flip_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<BFloat16>(), output.data<BFloat16>(), total, dim_size, inner_size);
            break;
        }
        case DType::Int32: {
            auto [grid, block] = optimal_launch_config(flip_kernel_impl<int32_t>, total);
            flip_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<int32_t>(), output.data<int32_t>(), total, dim_size, inner_size);
            break;
        }
        case DType::Int64: {
            auto [grid, block] = optimal_launch_config(flip_kernel_impl<int64_t>, total);
            flip_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<int64_t>(), output.data<int64_t>(), total, dim_size, inner_size);
            break;
        }
        default:
            throw std::runtime_error("flip_kernel: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// Memory Format Conversion (fallback when cuDNN is not available)
// ============================================================================
#ifndef TENZOR_HAS_CUDNN

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

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
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

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
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

auto to_memory_format_kernel(const Tensor& input_in, MemoryFormat format, void* stream_ptr) -> Tensor {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    // The transform kernels index input.data<T>() with flat NCHW offsets, so a
    // non-contiguous input would be read in the wrong order. Materialize first.
    Tensor input = input_in.is_contiguous() ? input_in : input_in.contiguous();
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
    const int block_size = 256;
    const int grid_size = std::min(static_cast<int>((total + block_size - 1) / block_size), 2147483647);

    if (format == MemoryFormat::ChannelsLast) {
        if (input.dtype() == DType::Float32) {
            nchw_to_nhwc_transform<float><<<grid_size, block_size, 0, stream>>>(
                input.data<float>(), output.data<float>(), N, C, H, W);
                CUDA_CHECK(cudaGetLastError());
        } else if (input.dtype() == DType::Float16) {
            nchw_to_nhwc_transform<Float16><<<grid_size, block_size, 0, stream>>>(
                input.data<Float16>(), output.data<Float16>(), N, C, H, W);
                CUDA_CHECK(cudaGetLastError());
        } else if (input.dtype() == DType::Float64) {
            nchw_to_nhwc_transform<double><<<grid_size, block_size, 0, stream>>>(
                input.data<double>(), output.data<double>(), N, C, H, W);
                CUDA_CHECK(cudaGetLastError());
        } else if (input.dtype() == DType::BFloat16) {
            nchw_to_nhwc_transform<BFloat16><<<grid_size, block_size, 0, stream>>>(
                input.data<BFloat16>(), output.data<BFloat16>(), N, C, H, W);
                CUDA_CHECK(cudaGetLastError());
        } else {
            throw std::runtime_error("to_memory_format_kernel: unsupported dtype for ChannelsLast");
        }
    } else {
        if (input.dtype() == DType::Float32) {
            nhwc_to_nchw_transform<float><<<grid_size, block_size, 0, stream>>>(
                input.data<float>(), output.data<float>(), N, C, H, W);
                CUDA_CHECK(cudaGetLastError());
        } else if (input.dtype() == DType::Float16) {
            nhwc_to_nchw_transform<Float16><<<grid_size, block_size, 0, stream>>>(
                input.data<Float16>(), output.data<Float16>(), N, C, H, W);
                CUDA_CHECK(cudaGetLastError());
        } else if (input.dtype() == DType::Float64) {
            nhwc_to_nchw_transform<double><<<grid_size, block_size, 0, stream>>>(
                input.data<double>(), output.data<double>(), N, C, H, W);
                CUDA_CHECK(cudaGetLastError());
        } else if (input.dtype() == DType::BFloat16) {
            nhwc_to_nchw_transform<BFloat16><<<grid_size, block_size, 0, stream>>>(
                input.data<BFloat16>(), output.data<BFloat16>(), N, C, H, W);
                CUDA_CHECK(cudaGetLastError());
        } else {
            throw std::runtime_error("to_memory_format_kernel: unsupported dtype for Contiguous");
        }
    }

    CUDA_CHECK(cudaGetLastError());

    return output;
}

#endif // !TENZOR_HAS_CUDNN

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
    int64_t inner_size  // product of sizes of dims after the roll dim
) {
    TENZOR_CUDA_KERNEL_LOOP(i, total_elements) {
        // Decompose flat index into (outer, dim_idx, inner)
        int64_t inner_idx = i % inner_size;
        int64_t dim_idx = (i / inner_size) % dim_size;
        int64_t outer_idx = i / (inner_size * dim_size);

        // Compute source index with wrap-around
        int64_t src_dim_idx = (dim_idx - shift + dim_size) % dim_size;
        int64_t src_idx = (outer_idx * dim_size + src_dim_idx) * inner_size + inner_idx;

        output[i] = input[src_idx];
    }
}

auto roll_kernel(const Tensor& input, int64_t shift, int64_t dim, cudaStream_t stream) -> Tensor {
    auto shape = input.shape();
    const int64_t ndim = static_cast<int64_t>(shape.size());

    // Normalize the dim and shift before they index the shape / drive the
    // kernel's modulo. The public roll() op normalizes both, but the OpId::Roll
    // dispatch path forwards AttrKey::Shift/Dim verbatim, so a negative dim
    // would index shape[] out of bounds (UB) and a shift outside [0, dim_size)
    // makes the kernel's (dim_idx - shift + dim_size) % dim_size go negative,
    // producing a negative src_idx and an out-of-bounds device read.
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("roll: dimension out of range");
    }

    auto cont = input.is_contiguous() ? input : contiguous_kernel(input, stream);

    auto output = Tensor::empty_uninitialized(
        std::vector<int64_t>(shape.begin(), shape.end()),
        input.dtype(), input.device());

    int64_t total = input.numel();
    if (total == 0) return output;

    int64_t dim_size = shape[dim];
    // Reduce shift into [0, dim_size) so the kernel's wrap-around modulo never
    // produces a negative source index.
    if (dim_size > 0) {
        shift = ((shift % dim_size) + dim_size) % dim_size;
    } else {
        shift = 0;
    }
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < static_cast<int64_t>(shape.size()); ++d) {
        inner_size *= shape[d];
    }

    switch (input.dtype()) {
        case DType::Float32: {
            auto [grid, block] = optimal_launch_config(roll_kernel_impl<float>, total);
            roll_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<float>(), output.data<float>(),
                total, dim_size, shift, inner_size);
            break;
        }
        case DType::Float64: {
            auto [grid, block] = optimal_launch_config(roll_kernel_impl<double>, total);
            roll_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<double>(), output.data<double>(),
                total, dim_size, shift, inner_size);
            break;
        }
        case DType::Int32: {
            auto [grid, block] = optimal_launch_config(roll_kernel_impl<int32_t>, total);
            roll_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<int32_t>(), output.data<int32_t>(),
                total, dim_size, shift, inner_size);
            break;
        }
        case DType::Int64: {
            auto [grid, block] = optimal_launch_config(roll_kernel_impl<int64_t>, total);
            roll_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<int64_t>(), output.data<int64_t>(),
                total, dim_size, shift, inner_size);
            break;
        }
        case DType::Float16: {
            auto [grid, block] = optimal_launch_config(roll_kernel_impl<Float16>, total);
            roll_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<Float16>(), output.data<Float16>(),
                total, dim_size, shift, inner_size);
            break;
        }
        case DType::BFloat16: {
            auto [grid, block] = optimal_launch_config(roll_kernel_impl<BFloat16>, total);
            roll_kernel_impl<<<grid, block, 0, stream>>>(
                cont.data<BFloat16>(), output.data<BFloat16>(),
                total, dim_size, shift, inner_size);
            break;
        }
        default:
            throw std::runtime_error("roll_kernel: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// ============================================================================
// repeat_interleave — repeat each element along a dimension
// ============================================================================

// Scalar repeats: output[idx] = input[idx with dim_idx / repeats]
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
    TENZOR_CUDA_KERNEL_LOOP(i, total_elements) {
        int64_t inner_idx = i % inner_size;
        int64_t out_dim_idx = (i / inner_size) % out_dim_size;
        int64_t outer_idx = i / (inner_size * out_dim_size);

        int64_t src_dim_idx = out_dim_idx / repeats;
        int64_t src_idx = (outer_idx * in_dim_size + src_dim_idx) * inner_size + inner_idx;

        output[i] = input[src_idx];
    }
}

// Tensor repeats: uses prefix sum array to map output dim index -> input dim index
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
    TENZOR_CUDA_KERNEL_LOOP(i, total_elements) {
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
                                     cudaStream_t stream) -> Tensor {
    auto shape = input.shape();
    auto cont = input.is_contiguous() ? input : contiguous_kernel(input, stream);

    int64_t ndim = shape.size();
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

    #define LAUNCH_SCALAR_RI(T) { \
        auto [grid, block] = optimal_launch_config(repeat_interleave_scalar_kernel_impl<T>, total); \
        repeat_interleave_scalar_kernel_impl<<<grid, block, 0, stream>>>( \
            cont.data<T>(), output.data<T>(), \
            total, in_dim_size, out_dim_size, repeats, inner_size); \
    }

    switch (input.dtype()) {
        case DType::Float32: LAUNCH_SCALAR_RI(float); break;
        case DType::Float64: LAUNCH_SCALAR_RI(double); break;
        case DType::Int32:   LAUNCH_SCALAR_RI(int32_t); break;
        case DType::Int64:   LAUNCH_SCALAR_RI(int64_t); break;
        case DType::Float16: LAUNCH_SCALAR_RI(Float16); break;
        case DType::BFloat16: LAUNCH_SCALAR_RI(BFloat16); break;
        default:
            throw std::runtime_error("repeat_interleave_scalar_kernel: unsupported dtype");
    }
    #undef LAUNCH_SCALAR_RI

    CUDA_CHECK(cudaGetLastError());
    return output;
}

// Cast repeats tensor to int64 on device
template <typename SrcT>
__global__ void cast_to_int64_kernel(const SrcT* __restrict__ src,
                                     int64_t* __restrict__ dst, int64_t n) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        dst[idx] = static_cast<int64_t>(src[idx]);
    }
}

// Single-thread kernel: computes the inclusive total from an exclusive-prefix
// array and writes both `prefix[N]` (sentinel consumed by the binary-search
// kernel) and a 1-element device scalar `d_total`. Replaces a pair of D2H
// copies + host-side arithmetic + an H2D write-back with one device-side
// computation.
__global__ void repeat_interleave_finalize_total_kernel(
    int64_t* __restrict__ prefix,
    const int64_t* __restrict__ repeats,
    int64_t* __restrict__ d_total,
    int64_t N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int64_t total = prefix[N - 1] + repeats[N - 1];
        prefix[N] = total;
        *d_total = total;
    }
}

auto repeat_interleave_tensor_kernel(const Tensor& input, const Tensor& repeats_tensor,
                                     int64_t dim, cudaStream_t stream) -> Tensor {
    auto shape = input.shape();
    auto cont = input.is_contiguous() ? input : contiguous_kernel(input, stream);

    int64_t ndim = shape.size();
    int64_t in_dim_size = shape[dim];

    // Empty input along `dim`: the output is empty too. Returning here avoids the
    // zero-byte device allocations + a finalize kernel that would read
    // prefix[N-1]/repeats[N-1] (i.e. index -1) out of bounds when N == 0.
    if (in_dim_size == 0) {
        std::vector<int64_t> out_shape(shape.begin(), shape.end());
        out_shape[dim] = 0;
        return Tensor::empty_uninitialized(out_shape, input.dtype(), input.device());
    }

    // Convert repeats to int64 on device (no CPU roundtrip)
    int64_t* d_repeats_i64 = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_repeats_i64, in_dim_size * sizeof(int64_t), stream));

    auto repeats_cont = repeats_tensor.is_contiguous() ? repeats_tensor : repeats_tensor.contiguous();
    if (repeats_cont.dtype() == DType::Int64) {
        CUDA_CHECK(cudaMemcpyAsync(d_repeats_i64, repeats_cont.data<int64_t>(),
                                   in_dim_size * sizeof(int64_t),
                                   cudaMemcpyDeviceToDevice, stream));
    } else {
        dim3 grid, block;
        compute_launch_config_1d(in_dim_size, grid, block);
        if (repeats_cont.dtype() == DType::Int32) {
            cast_to_int64_kernel<<<grid, block, 0, stream>>>(
                repeats_cont.data<int32_t>(), d_repeats_i64, in_dim_size);
        } else if (repeats_cont.dtype() == DType::Float32) {
            cast_to_int64_kernel<<<grid, block, 0, stream>>>(
                repeats_cont.data<float>(), d_repeats_i64, in_dim_size);
        } else if (repeats_cont.dtype() == DType::Float64) {
            cast_to_int64_kernel<<<grid, block, 0, stream>>>(
                repeats_cont.data<double>(), d_repeats_i64, in_dim_size);
        } else {
            CUDA_CHECK(cudaFreeAsync(d_repeats_i64, stream));
            throw std::runtime_error("repeat_interleave: unsupported repeats dtype");
        }
    }

    // Compute exclusive prefix sum on device using CUB
    int64_t* d_prefix = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_prefix, (in_dim_size + 1) * sizeof(int64_t), stream));

    // CUB ExclusiveSum: d_prefix[0]=0, d_prefix[i]=sum(d_repeats[0..i-1])
    // Pass the item count as int64_t (the NumItemsT template parameter) so a
    // dimension larger than INT_MAX is not silently truncated, which would scan
    // the wrong element count and corrupt the prefix array.
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_repeats_i64, d_prefix,
                                  in_dim_size, stream);
    CUDA_CHECK(cudaMallocAsync(&d_temp, temp_bytes, stream));
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_repeats_i64, d_prefix,
                                  in_dim_size, stream);
    CUDA_CHECK(cudaFreeAsync(d_temp, stream));

    // Compute total = d_prefix[N-1] + d_repeats[N-1] entirely on device,
    // writing both d_prefix[N] (sentinel for the binary-search consumer kernel)
    // and a 1-element device scalar `d_total`. Then issue a single async D2H
    // copy of the scalar into pinned host memory. This replaces 2x D2H copies
    // + host-side add + 1x H2D write-back (each of which serialised on
    // `stream`) with 1x tiny device kernel + 1x async D2H of an int64.
    int64_t* d_total = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_total, sizeof(int64_t), stream));
    repeat_interleave_finalize_total_kernel<<<1, 1, 0, stream>>>(
        d_prefix, d_repeats_i64, d_total, in_dim_size);

    // Pinned host scalar so the cudaMemcpyAsync is genuinely asynchronous
    // w.r.t. the host (pageable D2H would behave synchronously).
    int64_t* h_total_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_total_pinned, sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpyAsync(h_total_pinned, d_total, sizeof(int64_t),
                               cudaMemcpyDeviceToHost, stream));

    // The on-stream free of these temps is enqueued behind the copy, so the
    // stream-ordered allocator will recycle their memory after the copy
    // completes — no host wait required for these.
    CUDA_CHECK(cudaFreeAsync(d_repeats_i64, stream));
    CUDA_CHECK(cudaFreeAsync(d_total, stream));

    // We MUST have out_dim_size on the host before calling
    // Tensor::empty_uninitialized (the shape lives on the host). Defer the
    // single stream sync to the latest possible point so the finalize kernel
    // + D2H copy can overlap with any unrelated work already on `stream`.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int64_t out_dim_size = *h_total_pinned;
    CUDA_CHECK(cudaFreeHost(h_total_pinned));

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = out_dim_size;

    auto output = Tensor::empty_uninitialized(out_shape, input.dtype(), input.device());

    int64_t total = 1;
    for (auto s : out_shape) total *= s;
    if (total == 0) {
        CUDA_CHECK(cudaFreeAsync(d_prefix, stream));
        return output;
    }

    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) inner_size *= shape[d];

    #define LAUNCH_TENSOR_RI(T) { \
        auto [grid, block] = optimal_launch_config(repeat_interleave_tensor_kernel_impl<T>, total); \
        repeat_interleave_tensor_kernel_impl<<<grid, block, 0, stream>>>( \
            cont.data<T>(), output.data<T>(), d_prefix, \
            total, in_dim_size, out_dim_size, inner_size); \
    }

    switch (input.dtype()) {
        case DType::Float32: LAUNCH_TENSOR_RI(float); break;
        case DType::Float64: LAUNCH_TENSOR_RI(double); break;
        case DType::Int32:   LAUNCH_TENSOR_RI(int32_t); break;
        case DType::Int64:   LAUNCH_TENSOR_RI(int64_t); break;
        case DType::Float16: LAUNCH_TENSOR_RI(Float16); break;
        case DType::BFloat16: LAUNCH_TENSOR_RI(BFloat16); break;
        default:
            CUDA_CHECK(cudaFreeAsync(d_prefix, stream));
            throw std::runtime_error("repeat_interleave_tensor_kernel: unsupported dtype");
    }
    #undef LAUNCH_TENSOR_RI

    CUDA_CHECK(cudaFreeAsync(d_prefix, stream));
    CUDA_CHECK(cudaGetLastError());
    return output;
}

} // namespace cuda
} // namespace tenzor
