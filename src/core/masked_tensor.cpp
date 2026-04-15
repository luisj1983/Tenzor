/**
 * @file masked_tensor.cpp
 * @brief MaskedTensor implementation — tensor with boolean validity mask.
 */

#include "tenzor/core/masked_tensor.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/creation.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace tenzor {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MaskedTensor::MaskedTensor(Tensor data, Tensor mask)
    : data_(std::move(data)), mask_(std::move(mask)) {
    // Validate shapes match.
    auto ds = data_.shape();
    auto ms = mask_.shape();
    if (!std::ranges::equal(ds, ms)) {
        throw std::invalid_argument(
            "MaskedTensor: data shape and mask shape must match");
    }

    // Validate mask dtype (accept Bool or Int8 treated as bool).
    if (mask_.dtype() != DType::Bool && mask_.dtype() != DType::Int8) {
        throw std::invalid_argument(
            "MaskedTensor: mask must have dtype Bool or Int8, got " +
            std::string(dtype_name(mask_.dtype())));
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

auto MaskedTensor::count_valid() const -> int64_t {
    // Sum the mask (converts bool -> int, sums).
    auto count = tenzor::sum(mask_.to(DType::Int64));
    return count.item<int64_t>();
}

auto MaskedTensor::fill_masked(float value) const -> Tensor {
    auto fill = full_like(data_, static_cast<double>(value));
    return tenzor::where(mask_, data_, fill);
}

// ---------------------------------------------------------------------------
// Full reductions
// ---------------------------------------------------------------------------

auto MaskedTensor::sum() const -> Tensor {
    // Zero out masked elements, then sum.
    auto masked_data = data_ * mask_.to(data_.dtype());
    return tenzor::sum(masked_data);
}

auto MaskedTensor::mean() const -> Tensor {
    auto s = sum();
    auto n = count_valid();
    if (n == 0) {
        throw std::runtime_error(
            "MaskedTensor::mean: no valid elements (all masked)");
    }
    return s / static_cast<double>(n);
}

auto MaskedTensor::max() const -> Tensor {
    // Fill masked positions with -inf, then take global max.
    auto filled = fill_masked(-std::numeric_limits<float>::infinity());
    return tenzor::max(filled);
}

auto MaskedTensor::min() const -> Tensor {
    // Fill masked positions with +inf, then take global min.
    auto filled = fill_masked(std::numeric_limits<float>::infinity());
    return tenzor::min(filled);
}

// ---------------------------------------------------------------------------
// Dim reductions
// ---------------------------------------------------------------------------

auto MaskedTensor::sum(int64_t dim, bool keepdim) const -> MaskedTensor {
    auto masked_data = data_ * mask_.to(data_.dtype());
    auto result_data = tenzor::sum(masked_data, dim, keepdim);

    // Reduce the mask along dim: any valid element means the output slot is valid.
    // sum(mask, dim) > 0 gives us that.
    auto mask_sum = tenzor::sum(mask_.to(DType::Int64), dim, keepdim);
    // mask_sum > 0 → valid.  We compare by checking != 0 via to(Bool).
    // A simple approach: mask_sum > 0 is equivalent to mask_sum.to(Bool).
    auto result_mask = mask_sum.to(DType::Bool);

    return MaskedTensor(std::move(result_data), std::move(result_mask));
}

auto MaskedTensor::mean(int64_t dim, bool keepdim) const -> MaskedTensor {
    auto masked_data = data_ * mask_.to(data_.dtype());
    auto sum_data = tenzor::sum(masked_data, dim, keepdim);
    auto count = tenzor::sum(mask_.to(data_.dtype()), dim, keepdim);

    // Avoid division by zero: where count is 0, output 0.
    auto safe_count = tenzor::where(
        count.to(DType::Bool), count,
        full_like(count, 1.0));
    auto result_data = sum_data / safe_count;

    auto result_mask = count.to(DType::Bool);
    return MaskedTensor(std::move(result_data), std::move(result_mask));
}

// ---------------------------------------------------------------------------
// Element-wise binary ops
// ---------------------------------------------------------------------------

auto MaskedTensor::operator+(const MaskedTensor& other) const -> MaskedTensor {
    auto combined_mask = tenzor::logical_and(mask_, other.mask_);
    auto result_data = data_ + other.data_;
    return MaskedTensor(std::move(result_data), std::move(combined_mask));
}

auto MaskedTensor::operator-(const MaskedTensor& other) const -> MaskedTensor {
    auto combined_mask = tenzor::logical_and(mask_, other.mask_);
    auto result_data = data_ - other.data_;
    return MaskedTensor(std::move(result_data), std::move(combined_mask));
}

auto MaskedTensor::operator*(const MaskedTensor& other) const -> MaskedTensor {
    auto combined_mask = tenzor::logical_and(mask_, other.mask_);
    auto result_data = data_ * other.data_;
    return MaskedTensor(std::move(result_data), std::move(combined_mask));
}

auto MaskedTensor::operator/(const MaskedTensor& other) const -> MaskedTensor {
    auto combined_mask = tenzor::logical_and(mask_, other.mask_);
    auto result_data = data_ / other.data_;
    return MaskedTensor(std::move(result_data), std::move(combined_mask));
}

// ---------------------------------------------------------------------------
// Scalar ops
// ---------------------------------------------------------------------------

auto MaskedTensor::operator+(float scalar) const -> MaskedTensor {
    auto result_data = data_ + static_cast<double>(scalar);
    return MaskedTensor(std::move(result_data), mask_);
}

auto MaskedTensor::operator*(float scalar) const -> MaskedTensor {
    auto result_data = data_ * static_cast<double>(scalar);
    return MaskedTensor(std::move(result_data), mask_);
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

auto MaskedTensor::to_tensor(float fill_value) const -> Tensor {
    return fill_masked(fill_value);
}

} // namespace tenzor
