#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/shape.hpp"
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace cuda {

// Helper class to access Tensor private members from CUDA kernels
class CUDAKernelAccess {
public:
    static auto get_impl(const Tensor& t) -> const std::shared_ptr<TensorImpl>& {
        return t.impl_;
    }
    static auto get_impl_mutable(Tensor& t) -> std::shared_ptr<TensorImpl>& {
        return t.impl_;
    }
};

// CUDA Helper macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Compute optimal grid/block dimensions for 1D kernels
inline void compute_launch_config_1d(int64_t n, dim3& grid, dim3& block) {
    constexpr int threads_per_block = 256;
    block = dim3(threads_per_block);
    int64_t num_blocks = (n + threads_per_block - 1) / threads_per_block;
    num_blocks = std::min(num_blocks, static_cast<int64_t>(65535));
    // Ensure at least 1 block to avoid CUDA invalid argument error
    // Grid-stride loop will naturally handle n=0 by not executing any iterations
    grid = dim3(num_blocks > 0 ? static_cast<unsigned int>(num_blocks) : 1);
}

#define CUDA_GRID_STRIDE_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

constexpr int BLOCK_SIZE = 256;

inline int get_num_blocks(int64_t n, int block_size = BLOCK_SIZE) {
    return (n + block_size - 1) / block_size;
}

// Contiguous kernel - copies non-contiguous data to contiguous layout
template<typename T>
__global__ void contiguous_kernel_impl(const T* input, T* output,
                                       const int64_t* strides,
                                       const int64_t* shape,
                                       int64_t ndim, int64_t total_elements) {
    CUDA_GRID_STRIDE_LOOP(idx, total_elements) {
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

    // Copy strides and shape to device memory
    std::vector<int64_t> strides_vec(input.strides().begin(), input.strides().end());

    int64_t* d_strides;
    int64_t* d_shape;
    CUDA_CHECK(cudaMalloc(&d_strides, ndim * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_shape, ndim * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_strides, strides_vec.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_shape, shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice));

    // Launch kernel
    int num_blocks = get_num_blocks(total_elements);

    if (input.dtype() == DType::Float32) {
        contiguous_kernel_impl<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), result.data<float>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Float64) {
        contiguous_kernel_impl<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Float16) {
        contiguous_kernel_impl<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(result.data_ptr()),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Int32) {
        contiguous_kernel_impl<int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<int32_t>(), result.data<int32_t>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Int64) {
        contiguous_kernel_impl<int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<int64_t>(), result.data<int64_t>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Int8) {
        contiguous_kernel_impl<int8_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<int8_t>(), result.data<int8_t>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::UInt8) {
        contiguous_kernel_impl<uint8_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<uint8_t>(), result.data<uint8_t>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Bool) {
        contiguous_kernel_impl<bool><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const bool*>(input.data_ptr()),
            reinterpret_cast<bool*>(result.data_ptr()),
            d_strides, d_shape, ndim, total_elements);
    } else {
        cudaFree(d_strides);
        cudaFree(d_shape);
        throw std::runtime_error("Contiguous: unsupported dtype");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_strides);
        cudaFree(d_shape);
        throw std::runtime_error(std::string("CUDA error in contiguous_kernel: ") + cudaGetErrorString(err));
    }

    // Free device memory
    cudaFree(d_strides);
    cudaFree(d_shape);

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
    CUDAKernelAccess::get_impl_mutable(result) = std::make_shared<TensorImpl>(*CUDAKernelAccess::get_impl(input));
    CUDAKernelAccess::get_impl_mutable(result)->shape = new_shape;
    CUDAKernelAccess::get_impl_mutable(result)->strides = compute_strides(new_shape);

    return result;
}

// Transpose kernel - metadata manipulation (swap dimensions)
auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1, cudaStream_t stream) -> Tensor {
    // Transpose just swaps dimensions in metadata
    Tensor result;
    CUDAKernelAccess::get_impl_mutable(result) = std::make_shared<TensorImpl>(*CUDAKernelAccess::get_impl(input));
    std::swap(CUDAKernelAccess::get_impl_mutable(result)->shape[dim0], CUDAKernelAccess::get_impl_mutable(result)->shape[dim1]);
    std::swap(CUDAKernelAccess::get_impl_mutable(result)->strides[dim0], CUDAKernelAccess::get_impl_mutable(result)->strides[dim1]);
    return result;
}

// Permute kernel - metadata manipulation (permute dimensions)
auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims, cudaStream_t stream) -> Tensor {
    const int64_t ndim = input.ndim();

    Tensor result;
    CUDAKernelAccess::get_impl_mutable(result) = std::make_shared<TensorImpl>(*CUDAKernelAccess::get_impl(input));

    std::vector<int64_t> new_shape(ndim);
    std::vector<int64_t> new_strides(ndim);

    for (int64_t i = 0; i < ndim; ++i) {
        new_shape[i] = CUDAKernelAccess::get_impl(input)->shape[dims[i]];
        new_strides[i] = CUDAKernelAccess::get_impl(input)->strides[dims[i]];
    }

    CUDAKernelAccess::get_impl_mutable(result)->shape = std::move(new_shape);
    CUDAKernelAccess::get_impl_mutable(result)->strides = std::move(new_strides);

    return result;
}

// Squeeze kernel - metadata manipulation (remove dimension)
auto squeeze_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor {
    Tensor result;
    CUDAKernelAccess::get_impl_mutable(result) = std::make_shared<TensorImpl>(*CUDAKernelAccess::get_impl(input));

    if (dim >= 0) {
        // Squeeze specific dimension
        CUDAKernelAccess::get_impl_mutable(result)->shape.erase(CUDAKernelAccess::get_impl_mutable(result)->shape.begin() + dim);
        CUDAKernelAccess::get_impl_mutable(result)->strides.erase(CUDAKernelAccess::get_impl_mutable(result)->strides.begin() + dim);
    } else {
        // Squeeze all dimensions with size 1
        std::vector<int64_t> new_shape;
        std::vector<int64_t> new_strides;

        for (int64_t i = 0; i < input.ndim(); ++i) {
            if (CUDAKernelAccess::get_impl(input)->shape[i] != 1) {
                new_shape.push_back(CUDAKernelAccess::get_impl(input)->shape[i]);
                new_strides.push_back(CUDAKernelAccess::get_impl(input)->strides[i]);
            }
        }

        if (new_shape.empty()) {
            new_shape.push_back(1);
            new_strides.push_back(1);
        }

        CUDAKernelAccess::get_impl_mutable(result)->shape = std::move(new_shape);
        CUDAKernelAccess::get_impl_mutable(result)->strides = std::move(new_strides);
    }

    return result;
}

// Unsqueeze kernel - metadata manipulation (add dimension)
auto unsqueeze_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor {
    Tensor result;
    CUDAKernelAccess::get_impl_mutable(result) = std::make_shared<TensorImpl>(*CUDAKernelAccess::get_impl(input));

    CUDAKernelAccess::get_impl_mutable(result)->shape.insert(CUDAKernelAccess::get_impl_mutable(result)->shape.begin() + dim, 1);

    // Compute stride for new dimension
    int64_t new_stride = (dim < input.ndim()) ? CUDAKernelAccess::get_impl(input)->strides[dim] : 1;
    CUDAKernelAccess::get_impl_mutable(result)->strides.insert(CUDAKernelAccess::get_impl_mutable(result)->strides.begin() + dim, new_stride);

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
    CUDA_GRID_STRIDE_LOOP(idx, total_elements) {
        // Convert linear output index to multi-dimensional coordinates
        int64_t temp_idx = idx;
        int64_t coords[8];  // Support up to 8 dimensions

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

    // Allocate device memory for metadata
    const int64_t num_tensors = contiguous_tensors.size();

    // Prepare input pointers
    std::vector<void*> host_input_ptrs(num_tensors);
    for (size_t i = 0; i < num_tensors; ++i) {
        host_input_ptrs[i] = contiguous_tensors[i].data_ptr();
    }

    void** d_input_ptrs;
    CUDA_CHECK(cudaMalloc(&d_input_ptrs, num_tensors * sizeof(void*)));
    CUDA_CHECK(cudaMemcpy(d_input_ptrs, host_input_ptrs.data(),
                          num_tensors * sizeof(void*), cudaMemcpyHostToDevice));

    // Prepare shapes
    std::vector<int64_t> host_input_shapes(num_tensors * ndim);
    for (size_t t = 0; t < num_tensors; ++t) {
        auto shape = contiguous_tensors[t].shape();
        for (int64_t d = 0; d < ndim; ++d) {
            host_input_shapes[t * ndim + d] = shape[d];
        }
    }

    int64_t* d_input_shapes;
    CUDA_CHECK(cudaMalloc(&d_input_shapes, num_tensors * ndim * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_input_shapes, host_input_shapes.data(),
                          num_tensors * ndim * sizeof(int64_t), cudaMemcpyHostToDevice));

    // Prepare output shape
    int64_t* d_output_shape;
    CUDA_CHECK(cudaMalloc(&d_output_shape, ndim * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_output_shape, output_shape.data(),
                          ndim * sizeof(int64_t), cudaMemcpyHostToDevice));

    // Prepare output strides
    std::vector<int64_t> output_strides = compute_strides(output_shape);
    int64_t* d_output_strides;
    CUDA_CHECK(cudaMalloc(&d_output_strides, ndim * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_output_strides, output_strides.data(),
                          ndim * sizeof(int64_t), cudaMemcpyHostToDevice));

    // Prepare offsets at concat dimension (for optimization, not used in current impl)
    std::vector<int64_t> host_offsets(num_tensors);
    int64_t offset = 0;
    for (size_t t = 0; t < num_tensors; ++t) {
        host_offsets[t] = offset;
        offset += contiguous_tensors[t].shape()[dim];
    }

    int64_t* d_offsets;
    CUDA_CHECK(cudaMalloc(&d_offsets, num_tensors * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_offsets, host_offsets.data(),
                          num_tensors * sizeof(int64_t), cudaMemcpyHostToDevice));

    // Handle empty output case - don't launch kernel with 0 blocks
    if (total_elements == 0) {
        cudaFree(d_input_ptrs);
        cudaFree(d_input_shapes);
        cudaFree(d_output_shape);
        cudaFree(d_output_strides);
        cudaFree(d_offsets);
        return output;
    }

    // Launch kernel
    const int num_blocks = get_num_blocks(total_elements);

    if (tensors[0].dtype() == DType::Float32) {
        cat_kernel_impl<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<float**>(d_input_ptrs), output.data<float>(),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
    } else if (tensors[0].dtype() == DType::Float64) {
        cat_kernel_impl<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<double**>(d_input_ptrs), output.data<double>(),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
    } else if (tensors[0].dtype() == DType::Int32) {
        cat_kernel_impl<int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<int32_t**>(d_input_ptrs), output.data<int32_t>(),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
    } else if (tensors[0].dtype() == DType::Int64) {
        cat_kernel_impl<int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<int64_t**>(d_input_ptrs), output.data<int64_t>(),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
    } else if (tensors[0].dtype() == DType::Float16) {
        cat_kernel_impl<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<__half**>(d_input_ptrs), reinterpret_cast<__half*>(output.data_ptr()),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
    } else if (tensors[0].dtype() == DType::Int8) {
        cat_kernel_impl<int8_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<int8_t**>(d_input_ptrs), output.data<int8_t>(),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
    } else if (tensors[0].dtype() == DType::UInt8) {
        cat_kernel_impl<uint8_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<uint8_t**>(d_input_ptrs), output.data<uint8_t>(),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
    } else if (tensors[0].dtype() == DType::Bool) {
        cat_kernel_impl<bool><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<bool**>(d_input_ptrs), reinterpret_cast<bool*>(output.data_ptr()),
            d_input_shapes, d_output_shape, d_output_strides, d_offsets,
            num_tensors, ndim, dim, total_elements);
    } else {
        cudaFree(d_input_ptrs);
        cudaFree(d_input_shapes);
        cudaFree(d_output_shape);
        cudaFree(d_output_strides);
        cudaFree(d_offsets);
        throw std::runtime_error("Concatenation: unsupported dtype");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input_ptrs);
        cudaFree(d_input_shapes);
        cudaFree(d_output_shape);
        cudaFree(d_output_strides);
        cudaFree(d_offsets);
        throw std::runtime_error(std::string("CUDA error in cat_kernel: ") + cudaGetErrorString(err));
    }

    // Free device memory
    cudaFree(d_input_ptrs);
    cudaFree(d_input_shapes);
    cudaFree(d_output_shape);
    cudaFree(d_output_strides);
    cudaFree(d_offsets);

    return output;
}

// Repeat kernel - repeat tensor elements along dimensions
template<typename T>
__global__ void repeat_kernel_device(
    const T* input, T* output,
    const int64_t* input_shape, const int64_t* input_strides,
    const int64_t* repeats, int64_t ndim, int64_t n) {

    CUDA_KERNEL_LOOP(out_idx, n) {
        // Calculate output coordinates
        int64_t temp = out_idx;
        int64_t in_idx = 0;

        for (int64_t i = ndim - 1; i >= 0; --i) {
            int64_t out_shape_i = input_shape[i] * repeats[i];
            int64_t out_coord = temp % out_shape_i;
            temp /= out_shape_i;

            // Map output coordinate to input coordinate
            int64_t in_coord = out_coord / repeats[i];
            in_idx += in_coord * input_strides[i];
        }

        output[out_idx] = input[in_idx];
    }
}

auto repeat_kernel(const Tensor& input, const std::vector<int64_t>& repeats, cudaStream_t stream) -> Tensor {
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

    // Copy data to device
    int64_t* d_input_shape;
    int64_t* d_input_strides;
    int64_t* d_repeats;

    cudaMalloc(&d_input_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_input_strides, ndim * sizeof(int64_t));
    cudaMalloc(&d_repeats, ndim * sizeof(int64_t));

    cudaMemcpy(d_input_shape, input_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_strides, input_strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_repeats, repeats.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 grid, block;
    compute_launch_config_1d(n, grid, block);

    if (input.dtype() == DType::Float32) {
        repeat_kernel_device<<<grid, block, 0, stream>>>(
            input.data<float>(), output.data<float>(),
            d_input_shape, d_input_strides, d_repeats, ndim, n);
    } else if (input.dtype() == DType::Float64) {
        repeat_kernel_device<<<grid, block, 0, stream>>>(
            input.data<double>(), output.data<double>(),
            d_input_shape, d_input_strides, d_repeats, ndim, n);
    } else if (input.dtype() == DType::Float16) {
        repeat_kernel_device<<<grid, block, 0, stream>>>(
            input.data<Float16>(), output.data<Float16>(),
            d_input_shape, d_input_strides, d_repeats, ndim, n);
    } else {
        cudaFree(d_input_shape);
        cudaFree(d_input_strides);
        cudaFree(d_repeats);
        throw std::runtime_error("repeat operation only supports Float32, Float64, and Float16 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input_shape);
        cudaFree(d_input_strides);
        cudaFree(d_repeats);
        throw std::runtime_error(std::string("CUDA error in repeat_kernel: ") + cudaGetErrorString(err));
    }

    cudaStreamSynchronize(stream);

    // Free device memory
    cudaFree(d_input_shape);
    cudaFree(d_input_strides);
    cudaFree(d_repeats);

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

template<typename T>
__global__ void slice_kernel_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int64_t* __restrict__ input_strides,
    const int64_t* __restrict__ output_shape,
    const int64_t* __restrict__ output_strides,
    const int64_t* __restrict__ starts,
    const int64_t* __restrict__ steps,
    int64_t ndim,
    int64_t total_elements) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Convert linear output index to multi-dimensional index
    int64_t input_offset = 0;
    int64_t remaining = idx;
    for (int64_t d = 0; d < ndim; ++d) {
        int64_t dim_idx = remaining / output_strides[d];
        remaining %= output_strides[d];
        // Map output index to input index: input_idx = start + output_idx * step
        int64_t input_dim_idx = starts[d] + dim_idx * steps[d];
        input_offset += input_dim_idx * input_strides[d];
    }

    output[idx] = input[input_offset];
}

auto slice_kernel(
    const Tensor& input,
    const std::vector<int64_t>& starts,
    const std::vector<int64_t>& ends,
    const std::vector<int64_t>& steps,
    cudaStream_t stream
) -> Tensor {
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

    // Copy metadata to device
    int64_t* d_input_strides;
    int64_t* d_output_shape;
    int64_t* d_output_strides;
    int64_t* d_starts;
    int64_t* d_steps;
    size_t meta_bytes = ndim * sizeof(int64_t);
    cudaMalloc(&d_input_strides, meta_bytes);
    cudaMalloc(&d_output_shape, meta_bytes);
    cudaMalloc(&d_output_strides, meta_bytes);
    cudaMalloc(&d_starts, meta_bytes);
    cudaMalloc(&d_steps, meta_bytes);
    cudaMemcpyAsync(d_input_strides, input_strides.data(), meta_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_output_shape, output_shape.data(), meta_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_output_strides, output_strides.data(), meta_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_starts, padded_starts.data(), meta_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_steps, padded_steps.data(), meta_bytes, cudaMemcpyHostToDevice, stream);

    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    if (input.dtype() == DType::Float32) {
        slice_kernel_impl<float><<<num_blocks, block_size, 0, stream>>>(
            input.data<float>(), output.data<float>(),
            d_input_strides, d_output_shape, d_output_strides, d_starts, d_steps, ndim, total);
    } else if (input.dtype() == DType::Float64) {
        slice_kernel_impl<double><<<num_blocks, block_size, 0, stream>>>(
            input.data<double>(), output.data<double>(),
            d_input_strides, d_output_shape, d_output_strides, d_starts, d_steps, ndim, total);
    } else if (input.dtype() == DType::Float16) {
        slice_kernel_impl<__half><<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()),
            d_input_strides, d_output_shape, d_output_strides, d_starts, d_steps, ndim, total);
    } else if (input.dtype() == DType::Int32) {
        slice_kernel_impl<int32_t><<<num_blocks, block_size, 0, stream>>>(
            input.data<int32_t>(), output.data<int32_t>(),
            d_input_strides, d_output_shape, d_output_strides, d_starts, d_steps, ndim, total);
    } else if (input.dtype() == DType::Int64) {
        slice_kernel_impl<int64_t><<<num_blocks, block_size, 0, stream>>>(
            input.data<int64_t>(), output.data<int64_t>(),
            d_input_strides, d_output_shape, d_output_strides, d_starts, d_steps, ndim, total);
    } else {
        cudaFree(d_input_strides); cudaFree(d_output_shape); cudaFree(d_output_strides);
        cudaFree(d_starts); cudaFree(d_steps);
        throw std::runtime_error("slice: unsupported dtype");
    }

    cudaError_t err = cudaGetLastError();
    cudaFree(d_input_strides); cudaFree(d_output_shape); cudaFree(d_output_strides);
    cudaFree(d_starts); cudaFree(d_steps);

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

    for (int64_t i = 0; i < num_splits; ++i) {
        int64_t start = i * split_size;
        int64_t end = std::min(start + split_size, dim_size);

        std::vector<int64_t> starts(ndim, 0);
        std::vector<int64_t> ends(shape.begin(), shape.end());
        std::vector<int64_t> steps(ndim, 1);
        starts[dim] = start;
        ends[dim] = end;

        results.push_back(slice_kernel(input, starts, ends, steps, stream));
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

auto tile_kernel(const Tensor& input, const std::vector<int64_t>& reps, cudaStream_t stream) -> Tensor {
    return repeat_kernel(input, reps, stream);
}
} // namespace cuda
} // namespace tenzor
