#include "tenzor/core/tensor.hpp"
#include <CL/sycl.hpp>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace tenzor {
namespace oneapi {

// SYCL Kernel name classes
class TransposeKernelFloat32;
class TransposeKernelFloat64;
class PermuteKernelFloat32;
class PermuteKernelFloat64;

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

// Helper to calculate strides from shape
inline auto calculate_strides(const std::vector<int64_t>& shape) -> std::vector<int64_t> {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int64_t i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

// Helper to compute flat index from multi-dimensional indices
inline auto compute_flat_index(const std::vector<int64_t>& indices,
                                const std::vector<int64_t>& strides) -> int64_t {
    int64_t flat_idx = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        flat_idx += indices[i] * strides[i];
    }
    return flat_idx;
}

// Reshape kernel - just validates and creates view (no data copy)
auto reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape, sycl::queue& queue) -> Tensor {
    // Calculate total elements
    int64_t input_numel = input.numel();
    int64_t output_numel = 1;
    for (auto dim : new_shape) {
        output_numel *= dim;
    }

    if (input_numel != output_numel) {
        throw std::invalid_argument("Reshape: total number of elements must remain constant");
    }

    // For contiguous tensors, reshape is just a view change
    // For non-contiguous, we need to copy
    Tensor output(new_shape, input.dtype(), input.device());

    // Simple memory copy since data layout is preserved
    const size_t bytes = input_numel * input.dtype_size();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.memcpy(out_ptr, in_ptr, bytes).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.memcpy(out_ptr, in_ptr, bytes).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for reshape");
    }

    return output;
}

// Transpose kernel - swap two dimensions
auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1, sycl::queue& queue) -> Tensor {
    auto shape_span = input.shape();
    const size_t ndim = shape_span.size();

    // Handle negative dimensions
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;

    if (dim0 < 0 || dim0 >= static_cast<int64_t>(ndim) ||
        dim1 < 0 || dim1 >= static_cast<int64_t>(ndim)) {
        throw std::invalid_argument("Transpose: invalid dimensions");
    }

    // Convert span to vector
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

    // Create output shape by swapping dimensions
    std::vector<int64_t> out_shape = shape;
    std::swap(out_shape[dim0], out_shape[dim1]);

    Tensor output(out_shape, input.dtype(), input.device());

    // Calculate strides
    auto in_strides = calculate_strides(shape);
    auto out_strides = calculate_strides(out_shape);

    const int64_t numel = input.numel();

    // Convert strides to device-copyable arrays
    int64_t in_strides_arr[8];
    int64_t out_strides_arr[8];
    for (size_t i = 0; i < ndim && i < 8; ++i) {
        in_strides_arr[i] = in_strides[i];
        out_strides_arr[i] = out_strides[i];
    }

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<TransposeKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            // Compute multi-dimensional index in input
            int64_t temp = idx;
            int64_t in_idx = 0;
            int64_t out_idx = 0;

            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = temp / in_strides_arr[d];
                temp %= in_strides_arr[d];
                in_idx += coord * in_strides_arr[d];

                // Map to output dimension (swap dim0 and dim1)
                size_t out_d = (d == static_cast<size_t>(dim0)) ? dim1 :
                              (d == static_cast<size_t>(dim1)) ? dim0 : d;
                out_idx += coord * out_strides_arr[out_d];
            }

            out_ptr[out_idx] = in_ptr[in_idx];
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<TransposeKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            int64_t temp = idx;
            int64_t in_idx = 0;
            int64_t out_idx = 0;

            for (size_t d = 0; d < ndim; ++d) {
                int64_t coord = temp / in_strides_arr[d];
                temp %= in_strides_arr[d];
                in_idx += coord * in_strides_arr[d];

                size_t out_d = (d == static_cast<size_t>(dim0)) ? dim1 :
                              (d == static_cast<size_t>(dim1)) ? dim0 : d;
                out_idx += coord * out_strides_arr[out_d];
            }

            out_ptr[out_idx] = in_ptr[in_idx];
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for transpose");
    }

    return output;
}

// Permute kernel - reorder dimensions
auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims, sycl::queue& queue) -> Tensor {
    auto shape_span = input.shape();
    const size_t ndim = shape_span.size();

    if (dims.size() != ndim) {
        throw std::invalid_argument("Permute: number of dimensions must match");
    }

    // Convert span to vector
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

    // Validate and handle negative dimensions
    std::vector<int64_t> perm_dims = dims;
    for (auto& d : perm_dims) {
        if (d < 0) d += ndim;
        if (d < 0 || d >= static_cast<int64_t>(ndim)) {
            throw std::invalid_argument("Permute: invalid dimension");
        }
    }

    // Create output shape
    std::vector<int64_t> out_shape(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        out_shape[i] = shape[perm_dims[i]];
    }

    Tensor output(out_shape, input.dtype(), input.device());

    // Calculate strides
    auto in_strides = calculate_strides(shape);
    auto out_strides = calculate_strides(out_shape);

    const int64_t numel = input.numel();

    // Convert vectors to device-copyable arrays
    int64_t in_strides_arr[8];
    int64_t out_strides_arr[8];
    int64_t perm_dims_arr[8];
    for (size_t i = 0; i < ndim && i < 8; ++i) {
        in_strides_arr[i] = in_strides[i];
        out_strides_arr[i] = out_strides[i];
        perm_dims_arr[i] = perm_dims[i];
    }

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<PermuteKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            // Compute multi-dimensional coordinates from flat index
            int64_t coords[8];
            int64_t temp = flat_idx;
            for (size_t d = 0; d < ndim; ++d) {
                coords[d] = temp / in_strides_arr[d];
                temp %= in_strides_arr[d];
            }

            // Compute input index
            int64_t in_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                in_idx += coords[d] * in_strides_arr[d];
            }

            // Compute output index with permuted dimensions
            int64_t out_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                out_idx += coords[perm_dims_arr[d]] * out_strides_arr[d];
            }

            out_ptr[out_idx] = in_ptr[in_idx];
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<PermuteKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t coords[8];
            int64_t temp = flat_idx;
            for (size_t d = 0; d < ndim; ++d) {
                coords[d] = temp / in_strides_arr[d];
                temp %= in_strides_arr[d];
            }

            int64_t in_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                in_idx += coords[d] * in_strides_arr[d];
            }

            int64_t out_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                out_idx += coords[perm_dims_arr[d]] * out_strides_arr[d];
            }

            out_ptr[out_idx] = in_ptr[in_idx];
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for permute");
    }

    return output;
}

// Squeeze kernel - remove dimensions of size 1
auto squeeze_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();

    std::vector<int64_t> out_shape;

    if (dim == -1) {
        // Squeeze all dimensions of size 1
        for (auto s : shape) {
            if (s != 1) {
                out_shape.push_back(s);
            }
        }
    } else {
        // Squeeze specific dimension
        if (dim < 0) dim += shape.size();
        if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
            throw std::invalid_argument("Squeeze: invalid dimension");
        }

        if (shape[dim] != 1) {
            throw std::invalid_argument("Squeeze: dimension must be size 1");
        }

        for (size_t i = 0; i < shape.size(); ++i) {
            if (static_cast<int64_t>(i) != dim) {
                out_shape.push_back(shape[i]);
            }
        }
    }

    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    // Squeeze is just a view change, copy data
    Tensor output(out_shape, input.dtype(), input.device());
    const size_t bytes = input.numel() * input.dtype_size();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.memcpy(out_ptr, in_ptr, bytes).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.memcpy(out_ptr, in_ptr, bytes).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for squeeze");
    }

    return output;
}

// Unsqueeze kernel - add dimension of size 1
auto unsqueeze_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    const size_t ndim = shape.size();

    // Handle negative dimension
    if (dim < 0) dim += ndim + 1;

    if (dim < 0 || dim > static_cast<int64_t>(ndim)) {
        throw std::invalid_argument("Unsqueeze: invalid dimension");
    }

    // Create output shape with new dimension
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < ndim; ++i) {
        if (static_cast<int64_t>(i) == dim) {
            out_shape.push_back(1);
        }
        out_shape.push_back(shape[i]);
    }

    if (dim == static_cast<int64_t>(ndim)) {
        out_shape.push_back(1);
    }

    // Unsqueeze is just a view change, copy data
    Tensor output(out_shape, input.dtype(), input.device());
    const size_t bytes = input.numel() * input.dtype_size();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.memcpy(out_ptr, in_ptr, bytes).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.memcpy(out_ptr, in_ptr, bytes).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for unsqueeze");
    }

    return output;
}

// Contiguous kernel - ensure tensor data is laid out contiguously
auto contiguous_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    // Always create a new contiguous tensor
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const size_t bytes = input.numel() * input.dtype_size();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.memcpy(out_ptr, in_ptr, bytes).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.memcpy(out_ptr, in_ptr, bytes).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for contiguous");
    }

    return output;
}

// Clone kernel - create a copy of the tensor
auto clone_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const size_t bytes = input.numel() * input.dtype_size();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.memcpy(out_ptr, in_ptr, bytes).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.memcpy(out_ptr, in_ptr, bytes).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for clone");
    }

    return output;
}

// Fill operations

// Zeros kernel
auto zeros_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, sycl::queue& queue) -> Tensor {
    Tensor output(shape, dtype, device);
    const size_t bytes = output.numel() * output.dtype_size();

    void* ptr = const_cast<void*>(output.data_ptr());
    queue.memset(ptr, 0, bytes).wait();

    return output;
}

// Ones kernel - use simpler memcpy approach for compatibility
auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, sycl::queue& queue) -> Tensor {
    Tensor output(shape, dtype, device);
    const int64_t numel = output.numel();

    if (dtype == DType::Float32) {
        // Create host buffer with ones
        std::vector<float> host_data(numel, 1.0f);
        float* device_ptr = get_data_ptr<float>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(float)).wait();
    }
    else if (dtype == DType::Float64) {
        // Create host buffer with ones
        std::vector<double> host_data(numel, 1.0);
        double* device_ptr = get_data_ptr<double>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(double)).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for ones");
    }

    return output;
}

// Full kernel - fill with specific value using memcpy
auto full_kernel(const std::vector<int64_t>& shape, float value, DType dtype, Device device, sycl::queue& queue) -> Tensor {
    Tensor output(shape, dtype, device);
    const int64_t numel = output.numel();

    if (dtype == DType::Float32) {
        std::vector<float> host_data(numel, value);
        float* device_ptr = get_data_ptr<float>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(float)).wait();
    }
    else if (dtype == DType::Float64) {
        const double value_d = static_cast<double>(value);
        std::vector<double> host_data(numel, value_d);
        double* device_ptr = get_data_ptr<double>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(double)).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for full");
    }

    return output;
}

// Fill kernel - fill existing tensor with value using memcpy
auto fill_kernel(const Tensor& tensor, float value, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(tensor.shape().begin(), tensor.shape().end()),
                  tensor.dtype(), tensor.device());

    const int64_t numel = tensor.numel();

    if (tensor.dtype() == DType::Float32) {
        std::vector<float> host_data(numel, value);
        float* device_ptr = get_data_ptr<float>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(float)).wait();
    }
    else if (tensor.dtype() == DType::Float64) {
        const double value_d = static_cast<double>(value);
        std::vector<double> host_data(numel, value_d);
        double* device_ptr = get_data_ptr<double>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(double)).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for fill");
    }

    return output;
}

} // namespace oneapi
} // namespace tenzor
