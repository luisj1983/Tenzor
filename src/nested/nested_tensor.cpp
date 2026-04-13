/**
 * @file nested_tensor.cpp
 * @brief NestedTensor implementation — jagged layout with contiguous buffer + offsets
 */

#include "tenzor/nested/nested_tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/reduction.hpp"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace tenzor {

// =========================================================================
// Private Constructor
// =========================================================================

NestedTensor::NestedTensor(Tensor values, Tensor offsets, int64_t batch_size,
                           int64_t ragged_dim, std::vector<int64_t> regular_shape)
    : values_(std::move(values))
    , offsets_(std::move(offsets))
    , batch_size_(batch_size)
    , ragged_dim_(ragged_dim)
    , regular_shape_(std::move(regular_shape)) {}

// =========================================================================
// from_tensor_list
// =========================================================================

auto NestedTensor::from_tensor_list(std::span<const Tensor> tensors) -> NestedTensor {
    if (tensors.empty()) {
        throw std::runtime_error("NestedTensor::from_tensor_list: empty tensor list");
    }

    auto ref_dtype = tensors[0].dtype();
    auto ref_device = tensors[0].device();
    auto ref_ndim = tensors[0].ndim();

    if (ref_ndim < 1) {
        throw std::runtime_error("NestedTensor::from_tensor_list: tensors must be at least 1D");
    }

    // Extract regular (trailing) shape from first tensor — all dims except dim 0
    std::vector<int64_t> regular_shape;
    for (int64_t d = 1; d < ref_ndim; ++d) {
        regular_shape.push_back(tensors[0].shape()[d]);
    }

    // Validate consistency and build offsets
    int64_t B = static_cast<int64_t>(tensors.size());
    std::vector<int64_t> offsets_data(B + 1, 0);

    for (int64_t i = 0; i < B; ++i) {
        const auto& t = tensors[i];
        if (t.dtype() != ref_dtype) {
            throw std::runtime_error(
                "NestedTensor::from_tensor_list: dtype mismatch at index " +
                std::to_string(i));
        }
        if (t.device() != ref_device) {
            throw std::runtime_error(
                "NestedTensor::from_tensor_list: device mismatch at index " +
                std::to_string(i));
        }
        if (t.ndim() != ref_ndim) {
            throw std::runtime_error(
                "NestedTensor::from_tensor_list: ndim mismatch at index " +
                std::to_string(i));
        }
        // Check trailing dims match
        for (int64_t d = 1; d < ref_ndim; ++d) {
            if (t.shape()[d] != regular_shape[d - 1]) {
                throw std::runtime_error(
                    "NestedTensor::from_tensor_list: shape mismatch at dim " +
                    std::to_string(d) + " for tensor " + std::to_string(i) +
                    " (expected " + std::to_string(regular_shape[d - 1]) +
                    ", got " + std::to_string(t.shape()[d]) + ")");
            }
        }
        offsets_data[i + 1] = offsets_data[i] + t.shape()[0];
    }

    // Cat all tensors along dim 0 to create contiguous values buffer
    auto values = tenzor::cat(tensors, 0);

    // Build offsets tensor (always Int64, same device as values)
    auto offsets = tenzor::zeros({B + 1}, DType::Int64, ref_device);
    auto* off_ptr = offsets.data<int64_t>();
    for (int64_t i = 0; i <= B; ++i) {
        off_ptr[i] = offsets_data[i];
    }

    return NestedTensor(std::move(values), std::move(offsets), B,
                        /*ragged_dim=*/1, std::move(regular_shape));
}

// =========================================================================
// from_padded
// =========================================================================

auto NestedTensor::from_padded(const Tensor& padded,
                               const Tensor& lengths) -> NestedTensor {
    if (padded.ndim() < 2) {
        throw std::runtime_error(
            "NestedTensor::from_padded: padded tensor must be at least 2D");
    }
    if (lengths.dtype() != DType::Int64) {
        throw std::runtime_error(
            "NestedTensor::from_padded: lengths must be Int64");
    }

    int64_t B = padded.shape()[0];
    if (lengths.numel() != B) {
        throw std::runtime_error(
            "NestedTensor::from_padded: lengths.numel() must equal batch size");
    }

    // Move lengths to CPU for reading
    auto lengths_cpu = (lengths.device().type != Device::Type::CPU)
        ? lengths.to(Device::cpu()) : lengths;
    const auto* len_ptr = lengths_cpu.data<int64_t>();

    // Extract regular trailing dims (dims 2+)
    std::vector<int64_t> regular_shape;
    for (int64_t d = 2; d < padded.ndim(); ++d) {
        regular_shape.push_back(padded.shape()[d]);
    }

    // Compute total length and offsets
    std::vector<int64_t> offsets_data(B + 1, 0);
    for (int64_t i = 0; i < B; ++i) {
        if (len_ptr[i] < 0 || len_ptr[i] > padded.shape()[1]) {
            throw std::runtime_error(
                "NestedTensor::from_padded: invalid length " +
                std::to_string(len_ptr[i]) + " at index " + std::to_string(i));
        }
        offsets_data[i + 1] = offsets_data[i] + len_ptr[i];
    }

    // Slice each batch element up to its length and cat.
    // padded[i] gives shape [max_len, ...], then slice dim 0 for the length.
    std::vector<Tensor> segments;
    segments.reserve(B);
    for (int64_t i = 0; i < B; ++i) {
        if (len_ptr[i] > 0) {
            // select(0, i) removes batch dim -> [max_len, *regular]
            // slice(0, 0, len) -> [len, *regular]
            auto elem = padded.select(0, i).slice(0, 0, len_ptr[i]).contiguous();
            segments.push_back(elem);
        }
    }

    Tensor values;
    if (segments.empty()) {
        // All lengths are 0 — create empty values
        std::vector<int64_t> val_shape = {0};
        val_shape.insert(val_shape.end(), regular_shape.begin(),
                         regular_shape.end());
        values = tenzor::zeros(val_shape, padded.dtype(), padded.device());
    } else {
        values = tenzor::cat(segments, 0);
    }

    // Build offsets tensor
    auto offsets = tenzor::zeros({B + 1}, DType::Int64, padded.device());
    auto* off_ptr = offsets.data<int64_t>();
    for (int64_t i = 0; i <= B; ++i) {
        off_ptr[i] = offsets_data[i];
    }

    return NestedTensor(std::move(values), std::move(offsets), B,
                        /*ragged_dim=*/1, std::move(regular_shape));
}

// =========================================================================
// from_jagged
// =========================================================================

auto NestedTensor::from_jagged(Tensor values, Tensor offsets,
                               int64_t ragged_dim) -> NestedTensor {
    if (offsets.dtype() != DType::Int64) {
        throw std::runtime_error(
            "NestedTensor::from_jagged: offsets must be Int64");
    }
    if (offsets.ndim() != 1 || offsets.numel() < 2) {
        throw std::runtime_error(
            "NestedTensor::from_jagged: offsets must be 1D with at least 2 elements");
    }

    int64_t B = offsets.numel() - 1;

    // Extract regular shape (trailing dims of values beyond dim 0)
    std::vector<int64_t> regular_shape;
    for (int64_t d = 1; d < values.ndim(); ++d) {
        regular_shape.push_back(values.shape()[d]);
    }

    return NestedTensor(std::move(values), std::move(offsets), B,
                        ragged_dim, std::move(regular_shape));
}

// =========================================================================
// Properties
// =========================================================================

auto NestedTensor::ndim() const -> int64_t {
    // batch dim + ragged dim + regular dims
    return 1 + 1 + static_cast<int64_t>(regular_shape_.size());
}

auto NestedTensor::lengths() const -> Tensor {
    // Compute per-element lengths from offsets
    auto offsets_cpu = (offsets_.device().type != Device::Type::CPU)
        ? offsets_.to(Device::cpu()) : offsets_;
    const auto* off_ptr = offsets_cpu.data<int64_t>();

    auto result = tenzor::zeros({batch_size_}, DType::Int64, offsets_.device());
    auto* res_ptr = result.data<int64_t>();
    for (int64_t i = 0; i < batch_size_; ++i) {
        res_ptr[i] = off_ptr[i + 1] - off_ptr[i];
    }
    return result;
}

auto NestedTensor::nested_sizes() const -> std::vector<std::vector<int64_t>> {
    auto offsets_cpu = (offsets_.device().type != Device::Type::CPU)
        ? offsets_.to(Device::cpu()) : offsets_;
    const auto* off_ptr = offsets_cpu.data<int64_t>();

    std::vector<std::vector<int64_t>> sizes;
    sizes.reserve(batch_size_);
    for (int64_t i = 0; i < batch_size_; ++i) {
        std::vector<int64_t> shape;
        shape.push_back(off_ptr[i + 1] - off_ptr[i]);
        shape.insert(shape.end(), regular_shape_.begin(), regular_shape_.end());
        sizes.push_back(std::move(shape));
    }
    return sizes;
}

auto NestedTensor::max_length() const -> int64_t {
    auto offsets_cpu = (offsets_.device().type != Device::Type::CPU)
        ? offsets_.to(Device::cpu()) : offsets_;
    const auto* off_ptr = offsets_cpu.data<int64_t>();

    int64_t max_len = 0;
    for (int64_t i = 0; i < batch_size_; ++i) {
        max_len = std::max(max_len, off_ptr[i + 1] - off_ptr[i]);
    }
    return max_len;
}

// =========================================================================
// Conversion
// =========================================================================

auto NestedTensor::to_padded_tensor(double padding_value) const -> Tensor {
    int64_t max_len = max_length();

    // Output shape: [B, max_len, *regular_shape]
    std::vector<int64_t> out_shape;
    out_shape.push_back(batch_size_);
    out_shape.push_back(max_len);
    out_shape.insert(out_shape.end(), regular_shape_.begin(),
                     regular_shape_.end());

    auto output = tenzor::full(out_shape, padding_value, values_.dtype(),
                               values_.device());

    auto offsets_cpu = (offsets_.device().type != Device::Type::CPU)
        ? offsets_.to(Device::cpu()) : offsets_;
    const auto* off_ptr = offsets_cpu.data<int64_t>();

    // Compute element size and inner stride
    size_t elem_size = dtype_size(values_.dtype());
    int64_t inner_size = 1;
    for (auto d : regular_shape_) {
        inner_size *= d;
    }

    // Copy each segment into the padded output via memcpy
    for (int64_t i = 0; i < batch_size_; ++i) {
        int64_t start = off_ptr[i];
        int64_t end = off_ptr[i + 1];
        int64_t len = end - start;
        if (len > 0) {
            // src: values[start : end] is contiguous [len, *regular_shape]
            // dst: output[i, 0:len] at offset i * max_len * inner_size
            auto* src_ptr = static_cast<const char*>(values_.data_ptr()) +
                            static_cast<size_t>(start * inner_size) * elem_size;
            auto* dst_ptr = static_cast<char*>(output.data_ptr()) +
                            static_cast<size_t>(i * max_len * inner_size) * elem_size;
            std::memcpy(dst_ptr, src_ptr,
                        static_cast<size_t>(len * inner_size) * elem_size);
        }
    }

    return output;
}

auto NestedTensor::unbind() const -> std::vector<Tensor> {
    auto offsets_cpu = (offsets_.device().type != Device::Type::CPU)
        ? offsets_.to(Device::cpu()) : offsets_;
    const auto* off_ptr = offsets_cpu.data<int64_t>();

    std::vector<Tensor> result;
    result.reserve(batch_size_);
    for (int64_t i = 0; i < batch_size_; ++i) {
        result.push_back(values_.slice(0, off_ptr[i], off_ptr[i + 1]));
    }
    return result;
}

auto NestedTensor::select(int64_t index) const -> Tensor {
    // Support negative indexing
    if (index < 0) {
        index += batch_size_;
    }
    if (index < 0 || index >= batch_size_) {
        throw std::runtime_error(
            "NestedTensor::select: index " + std::to_string(index) +
            " out of range for batch_size " + std::to_string(batch_size_));
    }

    auto offsets_cpu = (offsets_.device().type != Device::Type::CPU)
        ? offsets_.to(Device::cpu()) : offsets_;
    const auto* off_ptr = offsets_cpu.data<int64_t>();

    return values_.slice(0, off_ptr[index], off_ptr[index + 1]);
}

// =========================================================================
// Device / DType Transfer
// =========================================================================

auto NestedTensor::to(Device device) const -> NestedTensor {
    if (values_.device() == device) {
        return *this;
    }
    return NestedTensor(values_.to(device), offsets_.to(device),
                        batch_size_, ragged_dim_, regular_shape_);
}

auto NestedTensor::to(DType dtype) const -> NestedTensor {
    if (values_.dtype() == dtype) {
        return *this;
    }
    // Keep offsets as Int64 always
    return NestedTensor(values_.to(dtype), offsets_,
                        batch_size_, ragged_dim_, regular_shape_);
}

auto NestedTensor::contiguous() const -> NestedTensor {
    return NestedTensor(values_.contiguous(), offsets_.contiguous(),
                        batch_size_, ragged_dim_, regular_shape_);
}

auto NestedTensor::clone() const -> NestedTensor {
    return NestedTensor(values_.clone(), offsets_.clone(),
                        batch_size_, ragged_dim_, regular_shape_);
}

// =========================================================================
// Gradient Tracking
// =========================================================================

auto NestedTensor::requires_grad() const -> bool {
    return values_.requires_grad();
}

auto NestedTensor::requires_grad_(bool requires_grad) -> NestedTensor& {
    values_.set_requires_grad(requires_grad);
    return *this;
}

// =========================================================================
// Free Factory Function
// =========================================================================

auto nested_tensor(std::vector<Tensor> tensors) -> NestedTensor {
    return NestedTensor::from_tensor_list(tensors);
}

} // namespace tenzor
