#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/shape.hpp"
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace rocm {

// Helper class to access Tensor private members from HIP kernels.
// Routes through TensorAccessor which is a friend of Tensor.
class HIPKernelAccess {
public:
    static auto get_impl(const Tensor& t) -> const std::shared_ptr<TensorImpl>& {
        return TensorAccessor::get_impl(t);
    }
    static auto get_impl_mutable(Tensor& t) -> std::shared_ptr<TensorImpl>& {
        return TensorAccessor::get_impl_mutable(t);
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
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(contiguous_kernel_impl<__half>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            d_strides, d_shape, ndim, total_elements);
    } else {
        hipFree(d_strides);
        hipFree(d_shape);
        throw std::runtime_error("Contiguous only supports Float32, Float64, Float16, Int32, and Int64 dtypes");
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
    result.mutable_shape() = new_shape;
    result.mutable_strides() = compute_strides(new_shape);

    return result;
}

// ==============================================================================
// Transpose Kernel - metadata manipulation (swap dimensions)
// ==============================================================================

auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1, hipStream_t stream) -> Tensor {
    // Transpose just swaps dimensions in metadata
    Tensor result;
    HIPKernelAccess::get_impl_mutable(result) = std::make_shared<TensorImpl>(*HIPKernelAccess::get_impl(input));
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
    HIPKernelAccess::get_impl_mutable(result) = std::make_shared<TensorImpl>(*HIPKernelAccess::get_impl(input));

    std::vector<int64_t> new_shape(ndim);
    std::vector<int64_t> new_strides(ndim);

    for (int64_t i = 0; i < ndim; ++i) {
        new_shape[i] = input.shape()[dims[i]];
        new_strides[i] = input.strides()[dims[i]];
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
    HIPKernelAccess::get_impl_mutable(result) = std::make_shared<TensorImpl>(*HIPKernelAccess::get_impl(input));

    if (dim >= 0) {
        // Squeeze specific dimension
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
    Tensor result;
    HIPKernelAccess::get_impl_mutable(result) = std::make_shared<TensorImpl>(*HIPKernelAccess::get_impl(input));

    auto& r_shape = result.mutable_shape();
    auto& r_strides = result.mutable_strides();
    r_shape.insert(r_shape.begin() + dim, 1);

    // Compute stride for new dimension
    int64_t new_stride = (dim < input.ndim()) ? input.strides()[dim] : 1;
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
    const int64_t* input_shape,
    const int64_t* output_shape,
    const int64_t* repeats,
    int64_t ndim,
    int64_t total_elements
) {
    HIP_GRID_STRIDE_LOOP(idx, total_elements) {
        // Convert output index to coordinates
        int64_t temp = idx;
        int64_t input_offset = 0;
        int64_t input_stride = 1;

        for (int64_t d = ndim - 1; d >= 0; --d) {
            int64_t coord = temp % output_shape[d];
            temp /= output_shape[d];

            // Map to input coordinate (wrap around)
            int64_t input_coord = coord % input_shape[d];

            // Compute input offset
            input_offset += input_coord * input_stride;
            input_stride *= input_shape[d];
        }

        output[idx] = input[input_offset];
    }
}

auto repeat_kernel(const Tensor& input, const std::vector<int64_t>& repeats, hipStream_t stream) -> Tensor {
    auto input_shape = input.shape();
    int64_t ndim = input_shape.size();

    // Compute output shape
    std::vector<int64_t> output_shape(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        output_shape[i] = input_shape[i] * repeats[i];
    }

    Tensor output(output_shape, input.dtype(), input.device());
    int64_t total_elements = output.numel();

    if (total_elements == 0) return output;

    // Copy shapes to device
    int64_t* d_input_shape;
    int64_t* d_output_shape;
    int64_t* d_repeats;
    HIP_CHECK(hipMalloc(&d_input_shape, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_output_shape, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMalloc(&d_repeats, ndim * sizeof(int64_t)));
    HIP_CHECK(hipMemcpy(d_input_shape, input_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_output_shape, output_shape.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_repeats, repeats.data(), ndim * sizeof(int64_t), hipMemcpyHostToDevice));

    int num_blocks = get_num_blocks(total_elements);

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(repeat_kernel_impl<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<float>(), output.data<float>(),
            d_input_shape, d_output_shape, d_repeats, ndim, total_elements);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(repeat_kernel_impl<double>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<double>(), output.data<double>(),
            d_input_shape, d_output_shape, d_repeats, ndim, total_elements);
    } else if (input.dtype() == DType::Int32) {
        hipLaunchKernelGGL(repeat_kernel_impl<int32_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<int32_t>(), output.data<int32_t>(),
            d_input_shape, d_output_shape, d_repeats, ndim, total_elements);
    } else if (input.dtype() == DType::Int64) {
        hipLaunchKernelGGL(repeat_kernel_impl<int64_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            input.data<int64_t>(), output.data<int64_t>(),
            d_input_shape, d_output_shape, d_repeats, ndim, total_elements);
    } else {
        HIP_CHECK(hipFree(d_input_shape));
        HIP_CHECK(hipFree(d_output_shape));
        HIP_CHECK(hipFree(d_repeats));
        throw std::runtime_error("repeat_kernel: unsupported dtype");
    }

    HIP_CHECK(hipFree(d_input_shape));
    HIP_CHECK(hipFree(d_output_shape));
    HIP_CHECK(hipFree(d_repeats));
    HIP_CHECK(hipGetLastError());

    return output;
}

// ==============================================================================
// Tile Kernel - tile tensor (like repeat but prepends dimensions if needed)
// ==============================================================================

auto tile_kernel(const Tensor& input, const std::vector<int64_t>& reps, hipStream_t stream) -> Tensor {
    auto input_shape = input.shape();
    int64_t ndim = input_shape.size();
    int64_t reps_size = reps.size();

    // Pad input shape or reps to match
    std::vector<int64_t> new_input_shape;
    std::vector<int64_t> new_reps;

    if (reps_size > ndim) {
        // Prepend 1s to input shape
        for (int64_t i = 0; i < reps_size - ndim; ++i) {
            new_input_shape.push_back(1);
        }
        for (int64_t i = 0; i < ndim; ++i) {
            new_input_shape.push_back(input_shape[i]);
        }
        new_reps = reps;
    } else {
        new_input_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        // Prepend 1s to reps
        for (int64_t i = 0; i < ndim - reps_size; ++i) {
            new_reps.push_back(1);
        }
        for (int64_t i = 0; i < reps_size; ++i) {
            new_reps.push_back(reps[i]);
        }
    }

    // Use repeat kernel with reshaped input
    Tensor reshaped = reshape_kernel(input, new_input_shape, stream);
    return repeat_kernel(reshaped, new_reps, stream);
}

// ==============================================================================
// Stack Kernel - stack tensors along new dimension
// ==============================================================================

template<typename T>
__global__ void stack_kernel_impl(
    const T* const* inputs,
    T* output,
    int64_t num_tensors,
    int64_t tensor_size
) {
    int64_t total_elements = num_tensors * tensor_size;

    HIP_GRID_STRIDE_LOOP(idx, total_elements) {
        int64_t tensor_idx = idx / tensor_size;
        int64_t elem_idx = idx % tensor_size;
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

    if (first.dtype() == DType::Float32) {
        hipLaunchKernelGGL(stack_kernel_impl<float>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            (const float* const*)d_input_ptrs, output.data<float>(),
            static_cast<int64_t>(tensors.size()), tensor_size);
    } else if (first.dtype() == DType::Float64) {
        hipLaunchKernelGGL(stack_kernel_impl<double>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            (const double* const*)d_input_ptrs, output.data<double>(),
            static_cast<int64_t>(tensors.size()), tensor_size);
    } else if (first.dtype() == DType::Int32) {
        hipLaunchKernelGGL(stack_kernel_impl<int32_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            (const int32_t* const*)d_input_ptrs, output.data<int32_t>(),
            static_cast<int64_t>(tensors.size()), tensor_size);
    } else if (first.dtype() == DType::Int64) {
        hipLaunchKernelGGL(stack_kernel_impl<int64_t>,
            dim3(num_blocks), dim3(BLOCK_SIZE), 0, stream,
            (const int64_t* const*)d_input_ptrs, output.data<int64_t>(),
            static_cast<int64_t>(tensors.size()), tensor_size);
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

    HIP_CHECK(hipStreamSynchronize(stream));
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

    // Pad input shape/strides if needed
    std::vector<int64_t> padded_input_shape(ndim, 1);
    std::vector<int64_t> padded_input_strides(ndim, 0);

    int64_t input_ndim = input_shape.size();
    int64_t pad_size = ndim - input_ndim;

    for (int64_t i = 0; i < input_ndim; ++i) {
        padded_input_shape[pad_size + i] = input_shape[i];
        padded_input_strides[pad_size + i] = input_strides[i];
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
    } else {
        HIP_CHECK(hipFree(d_input_shape));
        HIP_CHECK(hipFree(d_input_strides));
        HIP_CHECK(hipFree(d_output_shape));
        throw std::runtime_error("expand_kernel: unsupported dtype");
    }

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

    int64_t dim_size = shape[dim];
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

} // namespace rocm
} // namespace tenzor
