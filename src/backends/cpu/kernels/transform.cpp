#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/shape.hpp"
#include <cstring>
#include <iostream>

namespace tenzor {
namespace cpu {

// Forward declarations
auto contiguous_kernel(const Tensor& input) -> Tensor;

// CPU transform kernels

auto fill_kernel(const Tensor& input, float value) -> Tensor {
    // Create result tensor (clone)
    Tensor result(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t total_elements = result.numel();

    // Fill based on dtype
    if (input.dtype() == DType::Float32) {
        auto* data = result.data<float>();
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = value;
        }
    } else if (input.dtype() == DType::Float64) {
        auto* data = result.data<double>();
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = static_cast<double>(value);
        }
    } else if (input.dtype() == DType::Int32) {
        auto* data = result.data<int32_t>();
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = static_cast<int32_t>(value);
        }
    } else if (input.dtype() == DType::Int64) {
        auto* data = result.data<int64_t>();
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = static_cast<int64_t>(value);
        }
    }

    return result;
}

auto clone_kernel(const Tensor& input) -> Tensor {
    // Make contiguous first if needed
    Tensor cont = input.is_contiguous() ? input : contiguous_kernel(input);

    // Create new tensor
    Tensor result(std::vector<int64_t>(cont.shape().begin(), cont.shape().end()),
                  cont.dtype(), cont.device());

    // Copy data
    const size_t size_bytes = cont.numel() * dtype_size(cont.dtype());
    std::memcpy(result.data<uint8_t>(), cont.data<uint8_t>(), size_bytes);

    return result;
}

auto reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape) -> Tensor {
    // Reshape just manipulates metadata - create view
    // If not contiguous, need to make contiguous first
    if (!input.is_contiguous()) {
        return reshape_kernel(contiguous_kernel(input), new_shape);
    }

    // Create new tensor sharing storage (view)
    Tensor result;
    result.impl_ = std::make_shared<TensorImpl>(*input.impl_);
    result.impl_->shape = new_shape;
    result.impl_->strides = compute_strides(new_shape);

    return result;
}

auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor {
    // Transpose just swaps dimensions in metadata
    Tensor result;
    result.impl_ = std::make_shared<TensorImpl>(*input.impl_);
    std::swap(result.impl_->shape[dim0], result.impl_->shape[dim1]);
    std::swap(result.impl_->strides[dim0], result.impl_->strides[dim1]);
    return result;
}

auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims) -> Tensor {
    const int64_t ndim = input.ndim();

    Tensor result;
    result.impl_ = std::make_shared<TensorImpl>(*input.impl_);

    std::vector<int64_t> new_shape(ndim);
    std::vector<int64_t> new_strides(ndim);

    for (int64_t i = 0; i < ndim; ++i) {
        new_shape[i] = input.impl_->shape[dims[i]];
        new_strides[i] = input.impl_->strides[dims[i]];
    }

    result.impl_->shape = std::move(new_shape);
    result.impl_->strides = std::move(new_strides);

    return result;
}

auto squeeze_kernel(const Tensor& input, int64_t dim) -> Tensor {
    Tensor result;
    result.impl_ = std::make_shared<TensorImpl>(*input.impl_);

    if (dim >= 0) {
        // Squeeze specific dimension
        result.impl_->shape.erase(result.impl_->shape.begin() + dim);
        result.impl_->strides.erase(result.impl_->strides.begin() + dim);
    } else {
        // Squeeze all dimensions with size 1
        std::vector<int64_t> new_shape;
        std::vector<int64_t> new_strides;

        for (int64_t i = 0; i < input.ndim(); ++i) {
            if (input.impl_->shape[i] != 1) {
                new_shape.push_back(input.impl_->shape[i]);
                new_strides.push_back(input.impl_->strides[i]);
            }
        }

        if (new_shape.empty()) {
            new_shape.push_back(1);
            new_strides.push_back(1);
        }

        result.impl_->shape = std::move(new_shape);
        result.impl_->strides = std::move(new_strides);
    }

    return result;
}

auto unsqueeze_kernel(const Tensor& input, int64_t dim) -> Tensor {
    Tensor result;
    result.impl_ = std::make_shared<TensorImpl>(*input.impl_);

    result.impl_->shape.insert(result.impl_->shape.begin() + dim, 1);

    // Compute stride for new dimension
    int64_t new_stride = (dim < input.ndim()) ? input.impl_->strides[dim] : 1;
    result.impl_->strides.insert(result.impl_->strides.begin() + dim, new_stride);

    return result;
}

auto contiguous_kernel(const Tensor& input) -> Tensor {
    // If already contiguous, return as-is
    if (input.is_contiguous()) {
        return input;
    }

    // Create new contiguous tensor with same shape, dtype, device
    Tensor result(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    // Copy data using strides to access elements in correct order
    const int64_t total_elements = input.numel();
    const size_t element_size = dtype_size(input.dtype());

    // Get raw storage pointers WITHOUT offset
    // input.impl_->storage->data() returns the base pointer
    // We'll apply offset manually in the loop using element-based indexing
    auto* src = static_cast<uint8_t*>(const_cast<void*>(input.impl_->storage->data()));
    auto* dst = static_cast<uint8_t*>(static_cast<void*>(result.impl_->storage->data()));

    const int64_t ndims = input.ndim();

    if (ndims == 0) {
        // Scalar tensor - direct copy (account for offset)
        std::memcpy(dst, src + input.impl_->offset * element_size, element_size);
        return result;
    }

    // Multi-dimensional copy using stride calculations
    std::vector<int64_t> indices(ndims, 0);
    int64_t dst_offset = 0;

    auto strides = input.strides();
    auto shape = input.shape();
    const int64_t input_offset = input.impl_->offset;

    for (int64_t i = 0; i < total_elements; ++i) {
        // Calculate source offset using strides (including base offset)
        int64_t src_offset = input_offset;
        for (int64_t dim = 0; dim < ndims; ++dim) {
            src_offset += indices[dim] * strides[dim];
        }

        // Copy single element
        std::memcpy(dst + dst_offset * element_size,
                    src + src_offset * element_size,
                    element_size);

        // Increment destination offset (contiguous layout)
        ++dst_offset;

        // Increment indices (row-major order)
        for (int64_t dim = ndims - 1; dim >= 0; --dim) {
            if (++indices[dim] < shape[dim]) {
                break;
            }
            indices[dim] = 0;
        }
    }

    return result;
}

auto cat_kernel(const std::vector<Tensor>& tensors, int64_t dim) -> Tensor {
    if (tensors.empty()) {
        throw std::runtime_error("cat requires at least one tensor");
    }

    // Calculate output shape
    auto out_shape = std::vector<int64_t>(tensors[0].shape().begin(), tensors[0].shape().end());
    int64_t cat_dim_size = 0;
    for (const auto& t : tensors) {
        cat_dim_size += t.shape()[dim];
    }
    out_shape[dim] = cat_dim_size;

    auto output = Tensor::empty_uninitialized(out_shape, tensors[0].dtype(), tensors[0].device());

    // Copy data from each tensor
    int64_t offset = 0;
    size_t elem_size = dtype_size(tensors[0].dtype());

    for (const auto& t : tensors) {
        auto t_cont = t.is_contiguous() ? t : contiguous_kernel(t);
        auto t_shape = t_cont.shape();
        int64_t t_dim_size = t_shape[dim];

        // Calculate sizes before and after the cat dimension
        int64_t outer_size = 1;
        for (int64_t d = 0; d < dim; ++d) {
            outer_size *= t_shape[d];
        }
        int64_t inner_size = 1;
        for (size_t d = dim + 1; d < t_shape.size(); ++d) {
            inner_size *= t_shape[d];
        }

        const uint8_t* src = static_cast<const uint8_t*>(t_cont.impl_->storage->data());
        uint8_t* dst = static_cast<uint8_t*>(output.impl_->storage->data());

        for (int64_t o = 0; o < outer_size; ++o) {
            int64_t src_idx = o * t_dim_size * inner_size;
            int64_t dst_idx = o * cat_dim_size * inner_size + offset * inner_size;

            std::memcpy(dst + dst_idx * elem_size,
                        src + src_idx * elem_size,
                        t_dim_size * inner_size * elem_size);
        }

        offset += t_dim_size;
    }

    return output;
}

auto flatten_kernel(const Tensor& input, int64_t start_dim, int64_t end_dim) -> Tensor {
    auto shape = input.shape();
    int64_t ndim = shape.size();

    // Handle negative dims
    if (start_dim < 0) start_dim += ndim;
    if (end_dim < 0) end_dim += ndim;

    // Calculate flattened dimension size
    int64_t flat_size = 1;
    for (int64_t d = start_dim; d <= end_dim; ++d) {
        flat_size *= shape[d];
    }

    // Build new shape
    std::vector<int64_t> new_shape;
    for (int64_t d = 0; d < start_dim; ++d) {
        new_shape.push_back(shape[d]);
    }
    new_shape.push_back(flat_size);
    for (int64_t d = end_dim + 1; d < ndim; ++d) {
        new_shape.push_back(shape[d]);
    }

    return reshape_kernel(input, new_shape);
}

auto slice_kernel(const Tensor& input, int64_t dim, int64_t start, int64_t end, int64_t step) -> Tensor {
    auto shape = input.shape();
    auto strides = input.strides();
    int64_t ndim = shape.size();

    // Handle negative indices
    if (start < 0) start += shape[dim];
    if (end < 0) end += shape[dim];

    // Clamp to valid range
    start = std::max(int64_t(0), std::min(start, shape[dim]));
    end = std::max(int64_t(0), std::min(end, shape[dim]));

    // Calculate output size along the sliced dimension
    int64_t slice_size = (end - start + step - 1) / step;
    if (slice_size < 0) slice_size = 0;

    // Create new shape
    std::vector<int64_t> new_shape(shape.begin(), shape.end());
    new_shape[dim] = slice_size;

    // If step is 1, we can create a view
    if (step == 1) {
        Tensor result;
        result.impl_ = std::make_shared<TensorImpl>(*input.impl_);
        result.impl_->shape = new_shape;
        result.impl_->offset += start * strides[dim];
        return result;
    }

    // Otherwise, need to copy with stride
    auto output = Tensor::empty_uninitialized(new_shape, input.dtype(), input.device());

    size_t elem_size = dtype_size(input.dtype());
    int64_t outer_size = 1;
    for (int64_t d = 0; d < dim; ++d) {
        outer_size *= shape[d];
    }
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
        inner_size *= shape[d];
    }

    const uint8_t* src = static_cast<const uint8_t*>(input.impl_->storage->data());
    uint8_t* dst = static_cast<uint8_t*>(output.impl_->storage->data());

    int64_t dst_idx = 0;
    for (int64_t o = 0; o < outer_size; ++o) {
        for (int64_t s = start; s < end; s += step) {
            int64_t src_idx = (o * shape[dim] + s) * inner_size;
            std::memcpy(dst + dst_idx * elem_size,
                        src + src_idx * elem_size,
                        inner_size * elem_size);
            dst_idx += inner_size;
        }
    }

    return output;
}

} // namespace cpu
} // namespace tenzor
