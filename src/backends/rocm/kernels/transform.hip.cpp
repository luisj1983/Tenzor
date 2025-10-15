#include <hip/hip_runtime.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/shape.hpp"
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace rocm {

// Helper class to access Tensor private members from HIP kernels
class HIPKernelAccess {
public:
    static auto get_impl(const Tensor& t) -> const std::shared_ptr<TensorImpl>& {
        return t.impl_;
    }
    static auto get_impl_mutable(Tensor& t) -> std::shared_ptr<TensorImpl>& {
        return t.impl_;
    }
};

// HIP Error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define HIP_GRID_STRIDE_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

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

    int64_t* d_strides;
    int64_t* d_shape;
    HIP_CHECK(hipMalloc(&d_strides, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_shape, ndim * sizeof(int64_t)));
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
    } else {
        hipFree(d_strides);
        hipFree(d_shape);
        throw std::runtime_error("Contiguous only supports Float32, Float64, Int32, and Int64 dtypes");
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        hipFree(d_strides);
        hipFree(d_shape);
        throw std::runtime_error(std::string("HIP error in contiguous_kernel: ") + hipGetErrorString(err));
    }

    // Free device memory
    hipFree(d_strides);
    hipFree(d_shape);

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
    HIPKernelAccess::get_impl_mutable(result) = std::make_shared<TensorImpl>(*HIPKernelAccess::get_impl(input));
    HIPKernelAccess::get_impl_mutable(result)->shape = new_shape;
    HIPKernelAccess::get_impl_mutable(result)->strides = compute_strides(new_shape);

    return result;
}

// ==============================================================================
// Transpose Kernel - metadata manipulation (swap dimensions)
// ==============================================================================

auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1, hipStream_t stream) -> Tensor {
    // Transpose just swaps dimensions in metadata
    Tensor result;
    HIPKernelAccess::get_impl_mutable(result) = std::make_shared<TensorImpl>(*HIPKernelAccess::get_impl(input));
    std::swap(HIPKernelAccess::get_impl_mutable(result)->shape[dim0],
              HIPKernelAccess::get_impl_mutable(result)->shape[dim1]);
    std::swap(HIPKernelAccess::get_impl_mutable(result)->strides[dim0],
              HIPKernelAccess::get_impl_mutable(result)->strides[dim1]);
    return result;
}

// ==============================================================================
// Permute Kernel - metadata manipulation (permute dimensions)
// ==============================================================================

auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims, hipStream_t stream) -> Tensor {
    const int64_t ndim = input.ndim();

    Tensor result;
    HIPKernelAccess::get_impl_mutable(result) = std::make_shared<TensorImpl>(*HIPKernelAccess::get_impl(input));

    std::vector<int64_t> new_shape(ndim);
    std::vector<int64_t> new_strides(ndim);

    for (int64_t i = 0; i < ndim; ++i) {
        new_shape[i] = HIPKernelAccess::get_impl(input)->shape[dims[i]];
        new_strides[i] = HIPKernelAccess::get_impl(input)->strides[dims[i]];
    }

    HIPKernelAccess::get_impl_mutable(result)->shape = std::move(new_shape);
    HIPKernelAccess::get_impl_mutable(result)->strides = std::move(new_strides);

    return result;
}

// ==============================================================================
// Squeeze Kernel - metadata manipulation (remove dimension)
// ==============================================================================

auto squeeze_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor {
    Tensor result;
    HIPKernelAccess::get_impl_mutable(result) = std::make_shared<TensorImpl>(*HIPKernelAccess::get_impl(input));

    if (dim >= 0) {
        // Squeeze specific dimension
        HIPKernelAccess::get_impl_mutable(result)->shape.erase(
            HIPKernelAccess::get_impl_mutable(result)->shape.begin() + dim);
        HIPKernelAccess::get_impl_mutable(result)->strides.erase(
            HIPKernelAccess::get_impl_mutable(result)->strides.begin() + dim);
    } else {
        // Squeeze all dimensions with size 1
        std::vector<int64_t> new_shape;
        std::vector<int64_t> new_strides;

        for (int64_t i = 0; i < input.ndim(); ++i) {
            if (HIPKernelAccess::get_impl(input)->shape[i] != 1) {
                new_shape.push_back(HIPKernelAccess::get_impl(input)->shape[i]);
                new_strides.push_back(HIPKernelAccess::get_impl(input)->strides[i]);
            }
        }

        if (new_shape.empty()) {
            new_shape.push_back(1);
            new_strides.push_back(1);
        }

        HIPKernelAccess::get_impl_mutable(result)->shape = std::move(new_shape);
        HIPKernelAccess::get_impl_mutable(result)->strides = std::move(new_strides);
    }

    return result;
}

// ==============================================================================
// Unsqueeze Kernel - metadata manipulation (add dimension)
// ==============================================================================

auto unsqueeze_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor {
    Tensor result;
    HIPKernelAccess::get_impl_mutable(result) = std::make_shared<TensorImpl>(*HIPKernelAccess::get_impl(input));

    HIPKernelAccess::get_impl_mutable(result)->shape.insert(
        HIPKernelAccess::get_impl_mutable(result)->shape.begin() + dim, 1);

    // Compute stride for new dimension
    int64_t new_stride = (dim < input.ndim()) ? HIPKernelAccess::get_impl(input)->strides[dim] : 1;
    HIPKernelAccess::get_impl_mutable(result)->strides.insert(
        HIPKernelAccess::get_impl_mutable(result)->strides.begin() + dim, new_stride);

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

auto chunk_kernel(const Tensor& input, int64_t chunks, int64_t dim, hipStream_t stream) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    int64_t dim_size = input_shape[dim];

    // Calculate chunk size
    int64_t chunk_size = (dim_size + chunks - 1) / chunks;

    std::vector<Tensor> results;
    for (int64_t i = 0; i < chunks; ++i) {
        int64_t start = i * chunk_size;
        int64_t end = std::min(start + chunk_size, dim_size);

        if (start >= dim_size) break;

        // Create slice for this chunk
        std::vector<int64_t> chunk_shape(input_shape.begin(), input_shape.end());
        chunk_shape[dim] = end - start;

        Tensor chunk(chunk_shape, input.dtype(), input.device());

        // Copy data (simplified - would need proper slicing kernel)
        results.push_back(chunk);
    }

    return results;
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

auto flip_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor {
    auto input_shape = input.shape();
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
    } else {
        throw std::runtime_error("Flip only supports Float32, Float64, Int32, and Int64 dtypes");
    }

    HIP_CHECK(hipGetLastError());

    return result;
}

} // namespace rocm
} // namespace tenzor
