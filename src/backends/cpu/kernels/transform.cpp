#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/shape.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <immintrin.h>

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

    const int64_t total_elements = input.numel();
    const size_t element_size = dtype_size(input.dtype());

    auto* src = static_cast<uint8_t*>(const_cast<void*>(input.impl_->storage->data()));
    auto* dst = static_cast<uint8_t*>(static_cast<void*>(result.impl_->storage->data()));

    const int64_t ndims = input.ndim();

    if (ndims == 0) {
        std::memcpy(dst, src + input.impl_->offset * element_size, element_size);
        return result;
    }

    auto strides = input.strides();
    auto shape = input.shape();
    const int64_t input_offset = input.impl_->offset;

    // Find the largest contiguous block size from the innermost dimension
    // This allows us to copy multiple elements at once when inner strides are contiguous
    int64_t inner_block_size = 1;
    int64_t expected_stride = 1;
    int64_t block_dim = ndims;  // First non-contiguous dimension from the end

    for (int64_t dim = ndims - 1; dim >= 0; --dim) {
        if (strides[dim] == expected_stride) {
            inner_block_size *= shape[dim];
            expected_stride *= shape[dim];
            block_dim = dim;
        } else {
            break;
        }
    }

    const size_t block_bytes = inner_block_size * element_size;
    const int64_t num_blocks = total_elements / inner_block_size;

    // Special optimized path for 4D tensors (common in attention: batch, heads, seq, head_dim)
    if (ndims == 4 && inner_block_size > 1) {
        const int64_t d0 = shape[0], d1 = shape[1], d2 = shape[2], d3 = shape[3];
        const int64_t s0 = strides[0], s1 = strides[1], s2 = strides[2], s3 = strides[3];

        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i0 = 0; i0 < d0; ++i0) {
            for (int64_t i1 = 0; i1 < d1; ++i1) {
                for (int64_t i2 = 0; i2 < d2; ++i2) {
                    int64_t src_base = input_offset + i0 * s0 + i1 * s1 + i2 * s2;
                    int64_t dst_base = ((i0 * d1 + i1) * d2 + i2) * d3;

                    if (s3 == 1) {
                        // Innermost dimension is contiguous - use memcpy
                        std::memcpy(dst + dst_base * element_size,
                                    src + src_base * element_size,
                                    d3 * element_size);
                    } else {
                        // Non-contiguous innermost - copy element by element
                        for (int64_t i3 = 0; i3 < d3; ++i3) {
                            std::memcpy(dst + (dst_base + i3) * element_size,
                                        src + (src_base + i3 * s3) * element_size,
                                        element_size);
                        }
                    }
                }
            }
        }
        return result;
    }

    // Special optimized path for 3D tensors (common in batched operations)
    if (ndims == 3 && inner_block_size > 1) {
        const int64_t d0 = shape[0], d1 = shape[1], d2 = shape[2];
        const int64_t s0 = strides[0], s1 = strides[1], s2 = strides[2];

        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t i0 = 0; i0 < d0; ++i0) {
            for (int64_t i1 = 0; i1 < d1; ++i1) {
                int64_t src_base = input_offset + i0 * s0 + i1 * s1;
                int64_t dst_base = (i0 * d1 + i1) * d2;

                if (s2 == 1) {
                    std::memcpy(dst + dst_base * element_size,
                                src + src_base * element_size,
                                d2 * element_size);
                } else {
                    for (int64_t i2 = 0; i2 < d2; ++i2) {
                        std::memcpy(dst + (dst_base + i2) * element_size,
                                    src + (src_base + i2 * s2) * element_size,
                                    element_size);
                    }
                }
            }
        }
        return result;
    }

    // General case with block copying - parallelize outer blocks
    if (block_dim > 0 && inner_block_size > 1) {
        // Calculate outer dimensions for parallelization
        int64_t outer_size = 1;
        for (int64_t dim = 0; dim < block_dim; ++dim) {
            outer_size *= shape[dim];
        }

        #pragma omp parallel for schedule(static)
        for (int64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            // Convert block index to multi-dimensional indices for outer dims
            int64_t src_offset_calc = input_offset;
            int64_t temp = block_idx;

            for (int64_t dim = block_dim - 1; dim >= 0; --dim) {
                int64_t idx = temp % shape[dim];
                temp /= shape[dim];
                src_offset_calc += idx * strides[dim];
            }

            int64_t dst_offset_calc = block_idx * inner_block_size;

            std::memcpy(dst + dst_offset_calc * element_size,
                        src + src_offset_calc * element_size,
                        block_bytes);
        }
        return result;
    }

    // Fallback: element-by-element copy with OpenMP parallelization
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < total_elements; ++i) {
        // Calculate multi-dimensional index from linear index
        int64_t src_offset_calc = input_offset;
        int64_t temp = i;

        for (int64_t dim = ndims - 1; dim >= 0; --dim) {
            int64_t idx = temp % shape[dim];
            temp /= shape[dim];
            src_offset_calc += idx * strides[dim];
        }

        std::memcpy(dst + i * element_size,
                    src + src_offset_calc * element_size,
                    element_size);
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

auto slice_multi_kernel(const Tensor& input,
                        const std::vector<int64_t>& starts,
                        const std::vector<int64_t>& ends,
                        const std::vector<int64_t>& steps) -> Tensor {
    Tensor result = input;
    for (size_t d = 0; d < starts.size(); ++d) {
        result = slice_kernel(result, static_cast<int64_t>(d), starts[d], ends[d], steps[d]);
    }
    return result;
}

auto expand_kernel(const Tensor& input, const std::vector<int64_t>& target_shape) -> Tensor {
    // Expand creates a view with stride=0 for broadcast dimensions
    const auto& in_shape = input.shape();
    const auto& in_strides = input.strides();
    int64_t ndim_out = static_cast<int64_t>(target_shape.size());
    int64_t ndim_in = input.ndim();
    int64_t dim_diff = ndim_out - ndim_in;

    std::vector<int64_t> new_strides(ndim_out, 0);
    for (int64_t i = ndim_out - 1; i >= 0; --i) {
        int64_t in_idx = i - dim_diff;
        if (in_idx >= 0) {
            if (in_shape[in_idx] == target_shape[i]) {
                new_strides[i] = in_strides[in_idx];
            } else if (in_shape[in_idx] == 1) {
                new_strides[i] = 0;  // Broadcast
            } else {
                throw std::runtime_error("expand: incompatible shapes");
            }
        }
        // else: new leading dimension, stride stays 0
    }

    Tensor result;
    result.impl_ = std::make_shared<TensorImpl>(*input.impl_);
    result.impl_->shape = target_shape;
    result.impl_->strides = new_strides;
    return result;
}

auto repeat_kernel(const Tensor& input, const std::vector<int64_t>& repeats) -> Tensor {
    const auto& in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(repeats.size());

    // Compute output shape
    std::vector<int64_t> out_shape(ndim);
    int64_t dim_diff = ndim - input.ndim();
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t in_idx = i - dim_diff;
        int64_t in_dim = (in_idx >= 0) ? in_shape[in_idx] : 1;
        out_shape[i] = in_dim * repeats[i];
    }

    Tensor output(out_shape, input.dtype(), input.device());
    const size_t elem_size = dtype_size(input.dtype());
    const auto* src = input.data<uint8_t>();
    auto* dst = output.data<uint8_t>();

    // Simple implementation: iterate over output indices, map back to input
    int64_t total = output.numel();
    auto out_strides = compute_strides(out_shape);
    std::vector<int64_t> effective_in_shape(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t in_idx = i - dim_diff;
        effective_in_shape[i] = (in_idx >= 0) ? in_shape[in_idx] : 1;
    }
    auto in_strides_full = compute_strides(effective_in_shape);

    #pragma omp parallel for if(total > 65536)
    for (int64_t idx = 0; idx < total; ++idx) {
        int64_t src_linear = 0;
        int64_t remaining = idx;
        for (int64_t d = 0; d < ndim; ++d) {
            int64_t coord = remaining / out_strides[d];
            remaining %= out_strides[d];
            int64_t in_coord = coord % effective_in_shape[d];
            src_linear += in_coord * in_strides_full[d];
        }
        std::memcpy(dst + idx * elem_size, src + src_linear * elem_size, elem_size);
    }

    return output;
}

auto stack_kernel(const std::vector<Tensor>& tensors, int64_t dim) -> Tensor {
    if (tensors.empty()) {
        throw std::runtime_error("stack: expected non-empty list of tensors");
    }

    const auto& first_shape = tensors[0].shape();
    int64_t ndim = tensors[0].ndim();

    if (dim < 0) dim += ndim + 1;
    if (dim < 0 || dim > ndim) {
        throw std::out_of_range("stack: dimension out of range");
    }

    // Validate all tensors have same shape
    for (size_t i = 1; i < tensors.size(); ++i) {
        const auto& s = tensors[i].shape();
        if (s.size() != first_shape.size() || !std::equal(s.begin(), s.end(), first_shape.begin())) {
            throw std::runtime_error("stack: all tensors must have same shape");
        }
    }

    // Output shape: insert new dim of size=num_tensors at position dim
    std::vector<int64_t> out_shape;
    out_shape.reserve(ndim + 1);
    for (int64_t d = 0; d < dim; ++d) out_shape.push_back(first_shape[d]);
    out_shape.push_back(static_cast<int64_t>(tensors.size()));
    for (int64_t d = dim; d < ndim; ++d) out_shape.push_back(first_shape[d]);

    Tensor output(out_shape, tensors[0].dtype(), tensors[0].device());
    const size_t elem_size = dtype_size(tensors[0].dtype());
    auto* dst = output.data<uint8_t>();

    // Each tensor contributes a slice along the new dim
    int64_t outer_size = 1;
    for (int64_t d = 0; d < dim; ++d) outer_size *= first_shape[d];
    int64_t inner_size = 1;
    for (int64_t d = dim; d < ndim; ++d) inner_size *= first_shape[d];

    int64_t num_tensors = static_cast<int64_t>(tensors.size());

    for (int64_t t = 0; t < num_tensors; ++t) {
        Tensor cont = tensors[t].is_contiguous() ? tensors[t] : contiguous_kernel(tensors[t]);
        const auto* src = cont.data<uint8_t>();
        for (int64_t o = 0; o < outer_size; ++o) {
            int64_t dst_offset = (o * num_tensors + t) * inner_size * elem_size;
            int64_t src_offset = o * inner_size * elem_size;
            std::memcpy(dst + dst_offset, src + src_offset, inner_size * elem_size);
        }
    }

    return output;
}

auto split_kernel(const Tensor& input, int64_t split_size, int64_t dim) -> std::vector<Tensor> {
    const auto& shape = input.shape();
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;

    int64_t dim_size = shape[dim];
    std::vector<Tensor> result;

    const size_t elem_size = dtype_size(input.dtype());
    Tensor cont = input.is_contiguous() ? input : contiguous_kernel(input);
    const auto* src = cont.data<uint8_t>();

    int64_t outer_size = 1;
    for (int64_t d = 0; d < dim; ++d) outer_size *= shape[d];
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) inner_size *= shape[d];

    for (int64_t offset = 0; offset < dim_size; offset += split_size) {
        int64_t chunk_size = std::min(split_size, dim_size - offset);
        std::vector<int64_t> out_shape(shape.begin(), shape.end());
        out_shape[dim] = chunk_size;

        Tensor chunk(out_shape, input.dtype(), input.device());
        auto* dst = chunk.data<uint8_t>();

        for (int64_t o = 0; o < outer_size; ++o) {
            int64_t src_off = (o * dim_size + offset) * inner_size * elem_size;
            int64_t dst_off = o * chunk_size * inner_size * elem_size;
            std::memcpy(dst + dst_off, src + src_off, chunk_size * inner_size * elem_size);
        }

        result.push_back(std::move(chunk));
    }

    return result;
}

auto chunk_kernel(const Tensor& input, int64_t chunks, int64_t dim) -> std::vector<Tensor> {
    const auto& shape = input.shape();
    if (dim < 0) dim += input.ndim();
    int64_t dim_size = shape[dim];
    int64_t split_size = (dim_size + chunks - 1) / chunks;
    return split_kernel(input, split_size, dim);
}

auto tile_kernel(const Tensor& input, const std::vector<int64_t>& reps) -> Tensor {
    return repeat_kernel(input, reps);
}

auto to_memory_format_kernel(const Tensor& input, MemoryFormat format) -> Tensor {
    if (format == MemoryFormat::Preserve || format == MemoryFormat::Contiguous) {
        return contiguous_kernel(input);
    }

    const auto& shape = input.shape();
    const int64_t ndim = input.ndim();

    if (format == MemoryFormat::ChannelsLast && ndim == 4) {
        // NCHW -> NHWC: permute dims to (0, 2, 3, 1) then store contiguously
        // with strides set for NHWC layout
        int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
        std::vector<int64_t> out_shape = {N, C, H, W};
        // NHWC strides: stride order N>H>W>C
        std::vector<int64_t> nhwc_strides = {C * H * W, 1, W * C, C};

        Tensor output(out_shape, input.dtype(), input.device());
        const size_t elem_size = dtype_size(input.dtype());

        Tensor cont = input.is_contiguous() ? input : contiguous_kernel(input);
        const auto* src = static_cast<const uint8_t*>(cont.impl_->storage->data());
        auto* dst = static_cast<uint8_t*>(output.impl_->storage->data());

        // Reorder data from NCHW to NHWC
        #pragma omp parallel for collapse(2) if(N * C * H * W > 65536)
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        int64_t src_idx = ((n * C + c) * H + h) * W + w;
                        int64_t dst_idx = ((n * H + h) * W + w) * C + c;
                        std::memcpy(dst + dst_idx * elem_size,
                                    src + src_idx * elem_size, elem_size);
                    }
                }
            }
        }

        output.impl_->strides = nhwc_strides;
        return output;
    }

    // Fallback: just make contiguous
    return contiguous_kernel(input);
}

} // namespace cpu
} // namespace tenzor
