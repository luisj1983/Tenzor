#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/shape.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <omp.h>
#include <immintrin.h>
#include "tenzor/backend/omp_thresholds.hpp"

namespace tenzor {
namespace cpu {

// Forward declarations
auto contiguous_kernel(const Tensor& input) -> Tensor;

// CPU transform kernels

auto fill_kernel(const Tensor& input, double value) -> Tensor {
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
    } else if (input.dtype() == DType::Float16) {
        auto* data = result.data<Float16>();
        Float16 val(static_cast<float>(value));
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = val;
        }
    } else if (input.dtype() == DType::BFloat16) {
        auto* data = result.data<BFloat16>();
        BFloat16 val(static_cast<float>(value));
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = val;
        }
    } else if (input.dtype() == DType::Int8) {
        auto* data = result.data<int8_t>();
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = static_cast<int8_t>(value);
        }
    } else if (input.dtype() == DType::Int16) {
        auto* data = result.data<int16_t>();
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = static_cast<int16_t>(value);
        }
    } else if (input.dtype() == DType::UInt8) {
        auto* data = result.data<uint8_t>();
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = static_cast<uint8_t>(value);
        }
    } else if (input.dtype() == DType::UInt16) {
        auto* data = result.data<uint16_t>();
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = static_cast<uint16_t>(value);
        }
    } else if (input.dtype() == DType::UInt32) {
        auto* data = result.data<uint32_t>();
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = static_cast<uint32_t>(value);
        }
    } else if (input.dtype() == DType::UInt64) {
        auto* data = result.data<uint64_t>();
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = static_cast<uint64_t>(value);
        }
    } else if (input.dtype() == DType::Bool) {
        auto* data = result.data<bool>();
        bool val = (value != 0.0f);
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = val;
        }
    } else if (input.dtype() == DType::Complex64) {
        auto* data = result.data<std::complex<float>>();
        std::complex<float> val(value, 0.0f);
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = val;
        }
    } else if (input.dtype() == DType::Complex128) {
        auto* data = result.data<std::complex<double>>();
        std::complex<double> val(static_cast<double>(value), 0.0);
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = val;
        }
    } else if (input.dtype() == DType::FP8_E4M3) {
        auto* data = result.data<FP8_E4M3>();
        FP8_E4M3 val(static_cast<float>(value));
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = val;
        }
    } else if (input.dtype() == DType::FP8_E5M2) {
        auto* data = result.data<FP8_E5M2>();
        FP8_E5M2 val(static_cast<float>(value));
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = val;
        }
    } else if (input.dtype() == DType::FP8_E4M3FNUZ) {
        auto* data = result.data<FP8_E4M3FNUZ>();
        FP8_E4M3FNUZ val(static_cast<float>(value));
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = val;
        }
    } else if (input.dtype() == DType::FP8_E5M2FNUZ) {
        auto* data = result.data<FP8_E5M2FNUZ>();
        FP8_E5M2FNUZ val(static_cast<float>(value));
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = val;
        }
    } else if (input.dtype() == DType::QInt8) {
        if (input.q_scale() == 0.0) {
            throw std::runtime_error(
                "fill_kernel: fill on quantized tensor requires quantization params: "
                "call set_quantization_params(scale, zero_point) first");
        }
        auto* data = result.data<int8_t>();
        const int64_t qval = static_cast<int64_t>(std::round(static_cast<double>(value) / input.q_scale()))
                             + input.q_zero_point();
        const int8_t qbyte = static_cast<int8_t>(
            std::clamp(qval, static_cast<int64_t>(-128), static_cast<int64_t>(127)));
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = qbyte;
        }
    } else if (input.dtype() == DType::QUInt8) {
        if (input.q_scale() == 0.0) {
            throw std::runtime_error(
                "fill_kernel: fill on quantized tensor requires quantization params: "
                "call set_quantization_params(scale, zero_point) first");
        }
        auto* data = result.data<uint8_t>();
        const int64_t qval = static_cast<int64_t>(std::round(static_cast<double>(value) / input.q_scale()))
                             + input.q_zero_point();
        const uint8_t qbyte = static_cast<uint8_t>(
            std::clamp(qval, static_cast<int64_t>(0), static_cast<int64_t>(255)));
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = qbyte;
        }
    } else if (input.dtype() == DType::QInt4x2) {
        if (input.q_scale() == 0.0) {
            throw std::runtime_error(
                "fill_kernel: fill on quantized tensor requires quantization params: "
                "call set_quantization_params(scale, zero_point) first");
        }
        // Two 4-bit signed values per byte; clamp to [-8, 7] then pack nibbles.
        // For a uniform fill both nibbles equal qval, so:
        //   byte = (qval & 0xF) | ((qval & 0xF) << 4)
        auto* data = reinterpret_cast<uint8_t*>(result.data<int8_t>());
        const int64_t qval = static_cast<int64_t>(std::round(static_cast<double>(value) / input.q_scale()))
                             + input.q_zero_point();
        const int64_t clamped = std::clamp(qval, static_cast<int64_t>(-8), static_cast<int64_t>(7));
        const uint8_t qbyte = static_cast<uint8_t>((clamped & 0xF) | ((clamped & 0xF) << 4));
        for (int64_t i = 0; i < total_elements; ++i) {
            data[i] = qbyte;
        }
    } else {
        throw std::runtime_error(std::string("fill_kernel: unsupported dtype ") +
                                 std::string(dtype_name(input.dtype())));
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
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
    result.mutable_shape() = new_shape;
    result.mutable_strides() = compute_strides(new_shape);

    return result;
}

auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor {
    const int64_t ndim = input.ndim();
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        throw std::out_of_range("transpose: dimension out of range (got " +
            std::to_string(dim0) + " and " + std::to_string(dim1) +
            " for tensor with " + std::to_string(ndim) + " dimensions)");
    }
    // Transpose just swaps dimensions in metadata
    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
    auto& r_shape = result.mutable_shape();
    auto& r_strides = result.mutable_strides();
    std::swap(r_shape[dim0], r_shape[dim1]);
    std::swap(r_strides[dim0], r_strides[dim1]);
    return result;
}

auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims) -> Tensor {
    const int64_t ndim = input.ndim();

    if (static_cast<int64_t>(dims.size()) != ndim) {
        throw std::out_of_range("permute: number of dims must match tensor rank");
    }

    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));

    std::vector<int64_t> new_shape(ndim);
    std::vector<int64_t> new_strides(ndim);

    for (int64_t i = 0; i < ndim; ++i) {
        int64_t d = dims[i];
        if (d < 0) d += ndim;
        if (d < 0 || d >= ndim) {
            throw std::out_of_range("permute: dim out of range");
        }
        new_shape[i] = input.shape()[d];
        new_strides[i] = input.strides()[d];
    }

    result.mutable_shape() = std::move(new_shape);
    result.mutable_strides() = std::move(new_strides);

    return result;
}

// Sentinel meaning "squeeze every size-1 axis". A distinct value (not -1) is
// required so that a legitimate negative axis like squeeze(-1)/squeeze(-2) is
// not silently misinterpreted as squeeze-all.
constexpr int64_t SQUEEZE_ALL = std::numeric_limits<int64_t>::min();

auto squeeze_kernel(const Tensor& input, int64_t dim) -> Tensor {
    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));

    const int64_t ndim = input.ndim();

    if (dim != SQUEEZE_ALL) {
        // Squeeze a specific dimension. Normalize negatives and validate range
        // and that the axis is actually size 1 (mirrors Tensor::squeeze and the
        // oneAPI squeeze_kernel).
        if (dim < 0) dim += ndim;
        if (dim < 0 || dim >= ndim) {
            throw std::out_of_range("squeeze: dim out of range");
        }
        if (input.shape()[dim] != 1) {
            // PyTorch leaves a non-size-1 axis untouched; return an unchanged view.
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

        for (int64_t i = 0; i < ndim; ++i) {
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

auto unsqueeze_kernel(const Tensor& input, int64_t dim) -> Tensor {
    // Normalize/validate dim: a size-1 axis can be inserted at any position in
    // [0, ndim], so the valid range after normalization is [0, ndim].
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim + 1;
    if (dim < 0 || dim > ndim) throw std::out_of_range("unsqueeze: dim out of range");

    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));

    auto& r_shape = result.mutable_shape();
    auto& r_strides = result.mutable_strides();
    r_shape.insert(r_shape.begin() + dim, 1);

    // Compute stride for new dimension
    int64_t new_stride = (dim < input.ndim()) ? input.strides()[dim] : 1;
    r_strides.insert(r_strides.begin() + dim, new_stride);

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

    auto* src = static_cast<uint8_t*>(const_cast<void*>(input.storage()->data()));
    auto* dst = static_cast<uint8_t*>(static_cast<void*>(result.storage()->data()));

    const int64_t ndims = input.ndim();

    if (ndims == 0) {
        std::memcpy(dst, src + input.offset() * element_size, element_size);
        return result;
    }

    auto strides = input.strides();
    auto shape = input.shape();
    const int64_t input_offset = input.offset();

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

    // Defensive: normalize negative dim. The public op in src/ops/transform.cpp
    // also normalizes, but the kernel is reachable directly via
    // dispatch<OpId::Cat>(...) and a future caller must not cause OOB reads.
    int64_t ndim = static_cast<int64_t>(tensors[0].ndim());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("cat_kernel: dim out of range");
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

        const uint8_t* src = static_cast<const uint8_t*>(t_cont.data<uint8_t>());
        uint8_t* dst = static_cast<uint8_t*>(output.storage()->data());

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

    if (start_dim < 0 || start_dim >= ndim || end_dim < 0 || end_dim >= ndim) {
        throw std::out_of_range("flatten: start_dim/end_dim out of range (start_dim=" +
                                std::to_string(start_dim) + ", end_dim=" + std::to_string(end_dim) +
                                ", ndim=" + std::to_string(ndim) + ")");
    }
    if (start_dim > end_dim) {
        throw std::invalid_argument("flatten: start_dim (" + std::to_string(start_dim) +
                                    ") must be <= end_dim (" + std::to_string(end_dim) + ")");
    }

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

    // Defensive: normalize negative dim. The public op in src/ops/transform.cpp
    // also normalizes, but the kernel is reachable directly via
    // dispatch<OpId::Slice>(...) and a future caller must not cause OOB reads.
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("slice_kernel: dim out of range");
    }

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
        TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
        result.mutable_shape() = new_shape;
        result.set_offset(result.offset() + start * strides[dim]);
        return result;
    }

    // Otherwise, need to copy with stride. The linear src_idx below assumes a
    // packed row-major layout, so operate on a contiguous copy and honour the
    // tensor's offset — the previous code read storage()->data() (raw base, no
    // offset) and ignored input strides, corrupting the result for a
    // non-contiguous or offset input (e.g. a strided slice fed a second strided
    // slice via slice_multi_kernel).
    Tensor in_c = input.is_contiguous() ? input : input.contiguous();
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

    const uint8_t* src = static_cast<const uint8_t*>(in_c.storage()->data())
                         + static_cast<size_t>(in_c.offset()) * elem_size;
    uint8_t* dst = static_cast<uint8_t*>(output.storage()->data());

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
        // `ends`/`steps` may be shorter than `starts` (e.g. a Slice dispatched
        // with Starts but a defaulted/empty Steps). Default a missing end to the
        // current dim's full size and a missing step to 1 — matching the CUDA
        // slice kernel's defaulting (transform.cu) so the two backends agree
        // instead of the CPU path reading out of bounds while CUDA succeeds.
        const int64_t end = (d < ends.size())
                                ? ends[d]
                                : result.shape()[static_cast<int64_t>(d)];
        const int64_t step = (d < steps.size()) ? steps[d] : 1;
        result = slice_kernel(result, static_cast<int64_t>(d), starts[d], end, step);
    }
    return result;
}

auto expand_kernel(const Tensor& input, const std::vector<int64_t>& target_shape) -> Tensor {
    // Build the stride-0 broadcast view (dim of size 1 -> stride 0), then
    // MATERIALIZE it into a packed contiguous buffer. This matches the CUDA
    // expand kernel (expand_kernel_device writes a real replicated buffer) and
    // the historical CPU public-op path, so expand() is identical across
    // backends. Materializing also prevents a stride-0 view from flowing into
    // downstream kernels that read data<T>() without honouring strides (the
    // same hazard repeat_kernel/roll_kernel guard against).
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

    Tensor view;
    TensorAccessor::get_impl_mutable(view) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
    view.mutable_shape() = target_shape;
    view.mutable_strides() = new_strides;

    // contiguous_kernel walks the (possibly stride-0) view and copies each
    // logical element, replicating broadcast dimensions into a dense output.
    return contiguous_kernel(view);
}

auto repeat_kernel(const Tensor& input, const std::vector<int64_t>& repeats) -> Tensor {
    // Contiguify: data<T>() returns storage+offset and does NOT apply strides, so a
    // non-contiguous view (transposed/permuted/sliced/broadcast) would otherwise be
    // read as if contiguous and produce wrong data. Mirror roll_kernel's guard.
    auto cont = input.is_contiguous() ? input : contiguous_kernel(input);
    const auto& in_shape = cont.shape();
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
    const auto* src = cont.data<uint8_t>();
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

    #pragma omp parallel for if(total > ::tenzor::OmpThresholds::simple())
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
    if (split_size <= 0) {
        throw std::runtime_error("split: split_size must be positive");
    }
    const auto& shape = input.shape();
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range(
            "split: dim " + std::to_string(dim) + " is out of range for tensor with " +
            std::to_string(ndim) + " dimensions");
    }

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
    if (chunks <= 0) {
        throw std::runtime_error("chunk: chunks must be positive");
    }
    const auto& shape = input.shape();
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range(
            "chunk: dim " + std::to_string(dim) + " is out of range for tensor with " +
            std::to_string(ndim) + " dimensions");
    }
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
        const auto* src = static_cast<const uint8_t*>(cont.data<uint8_t>());
        auto* dst = static_cast<uint8_t*>(output.storage()->data());

        // Reorder data from NCHW to NHWC
        #pragma omp parallel for collapse(2) if(N * C * H * W > ::tenzor::OmpThresholds::simple())
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

        output.mutable_strides() = nhwc_strides;
        return output;
    }

    if (format == MemoryFormat::ChannelsLast3d && ndim == 5) {
        // NCDHW -> NDHWC: reorder data and set channels-last-3d strides so the
        // result actually carries the requested layout (previously this fell
        // through to a plain row-major contiguous tensor — a silent wrong layout).
        int64_t N = shape[0], C = shape[1], D = shape[2], H = shape[3], W = shape[4];
        std::vector<int64_t> out_shape = {N, C, D, H, W};
        // NDHWC strides: stride order N>D>H>W>C.
        std::vector<int64_t> ndhwc_strides = {C * D * H * W, 1, H * W * C, W * C, C};

        Tensor output(out_shape, input.dtype(), input.device());
        const size_t elem_size = dtype_size(input.dtype());

        Tensor cont = input.is_contiguous() ? input : contiguous_kernel(input);
        const auto* src = static_cast<const uint8_t*>(cont.data<uint8_t>());
        auto* dst = static_cast<uint8_t*>(output.storage()->data());

        // Reorder data from NCDHW to NDHWC
        #pragma omp parallel for collapse(2) if(N * C * D * H * W > ::tenzor::OmpThresholds::simple())
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t d = 0; d < D; ++d) {
                    for (int64_t h = 0; h < H; ++h) {
                        for (int64_t w = 0; w < W; ++w) {
                            int64_t src_idx = (((n * C + c) * D + d) * H + h) * W + w;
                            int64_t dst_idx = (((n * D + d) * H + h) * W + w) * C + c;
                            std::memcpy(dst + dst_idx * elem_size,
                                        src + src_idx * elem_size, elem_size);
                        }
                    }
                }
            }
        }

        output.mutable_strides() = ndhwc_strides;
        return output;
    }

    // Fallback: just make contiguous
    return contiguous_kernel(input);
}

auto roll_kernel(const Tensor& input, int64_t shift, int64_t dim) -> Tensor {
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("roll: dimension out of range (got " +
                                std::to_string(dim) + ", ndim=" + std::to_string(ndim) + ")");
    }
    auto cont = input.is_contiguous() ? input : contiguous_kernel(input);

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()),
                  input.dtype(), input.device());

    int64_t total = input.numel();
    if (total == 0) return output;

    int64_t dim_size = shape[dim];
    // Normalize shift to [0, dim_size)
    shift = ((shift % dim_size) + dim_size) % dim_size;
    if (shift == 0) {
        std::memcpy(output.storage()->data(), cont.data<uint8_t>(),
                    total * dtype_size(input.dtype()));
        return output;
    }

    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < static_cast<int64_t>(shape.size()); ++d) {
        inner_size *= shape[d];
    }

    const size_t elem_size = dtype_size(input.dtype());
    const auto* src = static_cast<const uint8_t*>(cont.data<uint8_t>());
    auto* dst = static_cast<uint8_t*>(output.storage()->data());

    #pragma omp parallel for if(total > ::tenzor::OmpThresholds::simple())
    for (int64_t i = 0; i < total; ++i) {
        int64_t inner_idx = i % inner_size;
        int64_t dim_idx = (i / inner_size) % dim_size;
        int64_t outer_idx = i / (inner_size * dim_size);

        int64_t src_dim_idx = (dim_idx - shift + dim_size) % dim_size;
        int64_t src_idx = (outer_idx * dim_size + src_dim_idx) * inner_size + inner_idx;

        std::memcpy(dst + i * elem_size, src + src_idx * elem_size, elem_size);
    }

    return output;
}

// ============================================================================
// repeat_interleave — repeat each element along a dimension
// ============================================================================

auto repeat_interleave_scalar_kernel(const Tensor& input, int64_t repeats, int64_t dim) -> Tensor {
    auto shape = input.shape();
    auto cont = input.is_contiguous() ? input : contiguous_kernel(input);

    int64_t ndim = shape.size();

    // Defensive: normalize negative dim and range-check. The public op in
    // src/ops/transform.cpp also normalizes, but the kernel is reachable
    // directly via dispatch<OpId::RepeatInterleave>(...) and a direct caller
    // must not cause OOB reads on shape (mirrors slice_kernel).
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("repeat_interleave_scalar_kernel: dim out of range");
    }

    int64_t dim_size = shape[dim];

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = dim_size * repeats;

    Tensor output(out_shape, input.dtype(), input.device());

    int64_t total_out = 1;
    for (auto s : out_shape) total_out *= s;
    if (total_out == 0) return output;

    // inner_size = product of dims after 'dim'
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
        inner_size *= shape[d];
    }

    const size_t elem_size = dtype_size(input.dtype());
    const auto* src = static_cast<const uint8_t*>(cont.data<uint8_t>());
    auto* dst = static_cast<uint8_t*>(output.storage()->data());

    // For each output element: map back to input
    // output[outer, d_out, inner] = input[outer, d_out / repeats, inner]
    #pragma omp parallel for if(total_out > ::tenzor::OmpThresholds::simple())
    for (int64_t i = 0; i < total_out; ++i) {
        int64_t inner_idx = i % inner_size;
        int64_t out_dim_idx = (i / inner_size) % out_shape[dim];
        int64_t outer_idx = i / (inner_size * out_shape[dim]);

        int64_t src_dim_idx = out_dim_idx / repeats;
        int64_t src_idx = (outer_idx * dim_size + src_dim_idx) * inner_size + inner_idx;

        std::memcpy(dst + i * elem_size, src + src_idx * elem_size, elem_size);
    }

    return output;
}

auto repeat_interleave_tensor_kernel(const Tensor& input, const Tensor& repeats_tensor, int64_t dim) -> Tensor {
    auto shape = input.shape();
    auto cont = input.is_contiguous() ? input : contiguous_kernel(input);

    int64_t ndim = shape.size();

    // Defensive: normalize negative dim and range-check. The public op in
    // src/ops/transform.cpp also normalizes, but the kernel is reachable
    // directly via dispatch<OpId::RepeatInterleave>(...) and a direct caller
    // must not cause OOB reads on shape (mirrors slice_kernel).
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("repeat_interleave_tensor_kernel: dim out of range");
    }

    int64_t dim_size = shape[dim];

    // Compute total output size along dim from repeats tensor
    // repeats must be on CPU and Int64 or Int32
    auto repeats_cont = repeats_tensor.is_contiguous() ? repeats_tensor : repeats_tensor.contiguous();

    // Defensive: the loops below read repeats_cont entries for i in [0, dim_size).
    // The public op validates repeats.shape()[0] == dim_size, but a direct
    // dispatch with a shorter repeats tensor would cause an OOB heap read.
    if (repeats_cont.numel() < dim_size) {
        throw std::runtime_error(
            "repeat_interleave_tensor_kernel: repeats tensor length (" +
            std::to_string(repeats_cont.numel()) +
            ") is smaller than input dimension size (" +
            std::to_string(dim_size) + ")");
    }

    // Read repeats into a vector and compute prefix sums
    std::vector<int64_t> reps(dim_size);
    int64_t total_repeats = 0;
    if (repeats_cont.dtype() == DType::Int64) {
        const int64_t* rp = repeats_cont.data<int64_t>();
        for (int64_t i = 0; i < dim_size; ++i) {
            if (rp[i] < 0) throw std::invalid_argument("repeat_interleave: negative repeat count");
            reps[i] = rp[i];
            total_repeats += rp[i];
        }
    } else if (repeats_cont.dtype() == DType::Int32) {
        const int32_t* rp = repeats_cont.data<int32_t>();
        for (int64_t i = 0; i < dim_size; ++i) {
            if (rp[i] < 0) throw std::invalid_argument("repeat_interleave: negative repeat count");
            reps[i] = rp[i];
            total_repeats += rp[i];
        }
    } else if (repeats_cont.dtype() == DType::Float32) {
        const float* rp = repeats_cont.data<float>();
        for (int64_t i = 0; i < dim_size; ++i) {
            int64_t r = static_cast<int64_t>(rp[i]);
            if (r < 0) throw std::invalid_argument("repeat_interleave: negative repeat count");
            reps[i] = r;
            total_repeats += r;
        }
    } else if (repeats_cont.dtype() == DType::Float64) {
        const double* rp = repeats_cont.data<double>();
        for (int64_t i = 0; i < dim_size; ++i) {
            int64_t r = static_cast<int64_t>(rp[i]);
            if (r < 0) throw std::invalid_argument("repeat_interleave: negative repeat count");
            reps[i] = r;
            total_repeats += r;
        }
    } else {
        throw std::runtime_error("repeat_interleave: unsupported repeats dtype");
    }

    // Build exclusive prefix sum for mapping output index -> input index
    std::vector<int64_t> prefix(dim_size + 1);
    prefix[0] = 0;
    for (int64_t i = 0; i < dim_size; ++i) {
        prefix[i + 1] = prefix[i] + reps[i];
    }

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[dim] = total_repeats;

    Tensor output(out_shape, input.dtype(), input.device());

    int64_t total_out = 1;
    for (auto s : out_shape) total_out *= s;
    if (total_out == 0) return output;

    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
        inner_size *= shape[d];
    }

    const size_t elem_size = dtype_size(input.dtype());
    const auto* src = static_cast<const uint8_t*>(cont.data<uint8_t>());
    auto* dst = static_cast<uint8_t*>(output.storage()->data());

    // Binary search to find which input element owns a given output dim index
    #pragma omp parallel for if(total_out > ::tenzor::OmpThresholds::simple())
    for (int64_t i = 0; i < total_out; ++i) {
        int64_t inner_idx = i % inner_size;
        int64_t out_dim_idx = (i / inner_size) % total_repeats;
        int64_t outer_idx = i / (inner_size * total_repeats);

        // Binary search in prefix array: find largest k such that prefix[k] <= out_dim_idx
        int64_t lo = 0, hi = dim_size;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if (prefix[mid + 1] <= out_dim_idx) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        int64_t src_dim_idx = lo;

        int64_t src_idx = (outer_idx * dim_size + src_dim_idx) * inner_size + inner_idx;

        std::memcpy(dst + i * elem_size, src + src_idx * elem_size, elem_size);
    }

    return output;
}

// Reverse-stride flip: one element-wise copy pass into a preallocated buffer.
// This is the concrete CPU implementation behind OpId::Flip; it must NOT call
// the public tenzor::flip (which dispatches OpId::Flip) or it would recurse.
auto flip_kernel(const Tensor& input, std::vector<int64_t> dims) -> Tensor {
    auto ndim = input.ndim();
    for (auto& d : dims) {
        if (d < 0) d += ndim;
        if (d < 0 || d >= ndim) {
            throw std::runtime_error("flip: dimension " + std::to_string(d) +
                                     " out of range for " + std::to_string(ndim) +
                                     "D tensor");
        }
    }

    auto shape = input.shape();
    auto input_cont = input.is_contiguous() ? input : contiguous_kernel(input);
    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()),
                  input.dtype(), input.device());

    int64_t total = input_cont.numel();
    if (total == 0) {
        return output;
    }

    // Contiguous strides for the (now contiguous) input / output.
    std::vector<int64_t> strides(ndim);
    int64_t s = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
        strides[i] = s;
        s *= shape[i];
    }

    // Mark which dimensions are flipped (idempotent: a boolean mask is correct).
    std::vector<char> flipped(ndim, 0);
    for (auto dim : dims) {
        flipped[dim] = 1;
    }

    const size_t esz = dtype_size(input.dtype());
    const char* in_base = static_cast<const char*>(input_cont.data_ptr());
    char* out_base = static_cast<char*>(output.data_ptr());

#ifdef _OPENMP
    #pragma omp parallel if (total > 65536)
    {
        std::vector<int64_t> coords(ndim);
        #pragma omp for
        for (int64_t out_idx = 0; out_idx < total; ++out_idx) {
            int64_t temp = out_idx;
            for (int64_t i = ndim - 1; i >= 0; --i) {
                coords[i] = temp % shape[i];
                temp /= shape[i];
            }
            int64_t in_idx = 0;
            for (int64_t i = 0; i < ndim; ++i) {
                int64_t c = flipped[i] ? (shape[i] - 1 - coords[i]) : coords[i];
                in_idx += c * strides[i];
            }
            std::memcpy(out_base + static_cast<size_t>(out_idx) * esz,
                        in_base + static_cast<size_t>(in_idx) * esz, esz);
        }
    }
#else
    std::vector<int64_t> coords(ndim);
    for (int64_t out_idx = 0; out_idx < total; ++out_idx) {
        int64_t temp = out_idx;
        for (int64_t i = ndim - 1; i >= 0; --i) {
            coords[i] = temp % shape[i];
            temp /= shape[i];
        }
        int64_t in_idx = 0;
        for (int64_t i = 0; i < ndim; ++i) {
            int64_t c = flipped[i] ? (shape[i] - 1 - coords[i]) : coords[i];
            in_idx += c * strides[i];
        }
        std::memcpy(out_base + static_cast<size_t>(out_idx) * esz,
                    in_base + static_cast<size_t>(in_idx) * esz, esz);
    }
#endif

    return output;
}

} // namespace cpu
} // namespace tenzor
