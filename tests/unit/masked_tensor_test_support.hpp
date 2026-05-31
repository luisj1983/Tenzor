#pragma once

/**
 * @file masked_tensor_test_support.hpp
 * @brief Test-only MaskedTensor — tensor paired with a boolean validity mask.
 *
 * This type previously lived in the production tree (include/tenzor/core/
 * masked_tensor.hpp + src/core/masked_tensor.cpp) but its only consumer was
 * test_masked_tensor.cpp, so it was relocated here to shrink the public
 * surface. The class declaration and its (inline) implementation are kept
 * together so the test translation unit is self-contained.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/creation.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace tenzor {

/**
 * @brief A tensor paired with a boolean validity mask.
 *
 * The mask has the same shape as the data tensor. True means the
 * corresponding element is valid; false means masked / invalid.
 *
 * Operations respect the mask:
 * - Element-wise ops compute only where the mask is true.
 * - Reductions aggregate only over valid elements.
 * - The mask propagates through operations (logical AND of input masks).
 */
class MaskedTensor {
public:
    MaskedTensor() = default;

    /// Create a masked tensor.
    /// @throws std::invalid_argument if shapes mismatch or mask dtype is wrong.
    MaskedTensor(Tensor data, Tensor mask);

    // -- Accessors ----------------------------------------------------------

    auto data() const -> const Tensor& { return data_; }
    auto data() -> Tensor& { return data_; }
    auto mask() const -> const Tensor& { return mask_; }

    /// Number of valid (unmasked) elements.
    [[nodiscard]] auto count_valid() const -> int64_t;

    /// Return a plain tensor with masked positions replaced by @p value.
    [[nodiscard]] auto fill_masked(float value) const -> Tensor;

    // -- Masked reductions (full) -------------------------------------------

    [[nodiscard]] auto sum() const -> Tensor;
    [[nodiscard]] auto mean() const -> Tensor;
    [[nodiscard]] auto max() const -> Tensor;
    [[nodiscard]] auto min() const -> Tensor;

    // -- Masked reductions (along dim) --------------------------------------

    [[nodiscard]] auto sum(int64_t dim,
                           bool keepdim = false) const -> MaskedTensor;
    [[nodiscard]] auto mean(int64_t dim,
                            bool keepdim = false) const -> MaskedTensor;

    // -- Element-wise ops (MaskedTensor x MaskedTensor) ---------------------

    [[nodiscard]] auto operator+(const MaskedTensor& other) const -> MaskedTensor;
    [[nodiscard]] auto operator-(const MaskedTensor& other) const -> MaskedTensor;
    [[nodiscard]] auto operator*(const MaskedTensor& other) const -> MaskedTensor;
    [[nodiscard]] auto operator/(const MaskedTensor& other) const -> MaskedTensor;

    // -- Scalar ops ---------------------------------------------------------

    [[nodiscard]] auto operator+(float scalar) const -> MaskedTensor;
    [[nodiscard]] auto operator*(float scalar) const -> MaskedTensor;

    /// Convert to a plain tensor, filling masked positions with @p fill_value.
    [[nodiscard]] auto to_tensor(float fill_value = 0.0f) const -> Tensor;

    // -- Shape delegation ---------------------------------------------------

    auto shape() const { return data_.shape(); }
    auto dtype() const { return data_.dtype(); }
    auto device() const { return data_.device(); }

private:
    Tensor data_;
    Tensor mask_;  ///< Bool tensor, same shape as data_.
};

// ===========================================================================
// Inline implementation
// ===========================================================================

namespace masked_tensor_detail {

/// Sentinel used by MaskedTensor::max() to fill masked positions before
/// taking a global maximum.  Integer dtypes use the minimum representable
/// value; floating-point dtypes use -inf; complex types use -inf for the
/// real part (audit item A.7).
inline auto dtype_min_sentinel(DType dt) -> double {
    switch (dt) {
        case DType::Float16:
        case DType::BFloat16:
        case DType::Float32:
        case DType::Float64:
        case DType::Complex64:
        case DType::Complex128:
            return -std::numeric_limits<double>::infinity();
        case DType::Int8:
            return static_cast<double>(std::numeric_limits<int8_t>::min());
        case DType::Int16:
            return static_cast<double>(std::numeric_limits<int16_t>::min());
        case DType::Int32:
            return static_cast<double>(std::numeric_limits<int32_t>::min());
        case DType::Int64:
            return static_cast<double>(std::numeric_limits<int64_t>::min());
        case DType::UInt8:
        case DType::Bool:
            return 0.0;
        default:
            return -std::numeric_limits<double>::infinity();
    }
}

/// Sentinel used by MaskedTensor::min() — maximum representable value.
inline auto dtype_max_sentinel(DType dt) -> double {
    switch (dt) {
        case DType::Float16:
        case DType::BFloat16:
        case DType::Float32:
        case DType::Float64:
        case DType::Complex64:
        case DType::Complex128:
            return std::numeric_limits<double>::infinity();
        case DType::Int8:
            return static_cast<double>(std::numeric_limits<int8_t>::max());
        case DType::Int16:
            return static_cast<double>(std::numeric_limits<int16_t>::max());
        case DType::Int32:
            return static_cast<double>(std::numeric_limits<int32_t>::max());
        case DType::Int64:
            return static_cast<double>(std::numeric_limits<int64_t>::max());
        case DType::UInt8:
            return static_cast<double>(std::numeric_limits<uint8_t>::max());
        case DType::Bool:
            return 1.0;
        default:
            return std::numeric_limits<double>::infinity();
    }
}

/// Zero out masked positions of `data` using `mask` so the result's data
/// buffer never carries NaN/Inf leakage from operations that may have produced
/// them at masked positions (audit item A.7).
inline auto zero_masked(const Tensor& data, const Tensor& mask) -> Tensor {
    auto zero = zeros_like(data);
    return tenzor::where(mask, data, zero);
}

}  // namespace masked_tensor_detail

inline MaskedTensor::MaskedTensor(Tensor data, Tensor mask)
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

inline auto MaskedTensor::count_valid() const -> int64_t {
    // Sum the mask (converts bool -> int, sums).
    auto count = tenzor::sum(mask_.to(DType::Int64));
    return count.item<int64_t>();
}

inline auto MaskedTensor::fill_masked(float value) const -> Tensor {
    auto fill = full_like(data_, static_cast<double>(value));
    return tenzor::where(mask_, data_, fill);
}

inline auto MaskedTensor::sum() const -> Tensor {
    // Zero out masked elements, then sum.
    auto masked_data = data_ * mask_.to(data_.dtype());
    return tenzor::sum(masked_data);
}

inline auto MaskedTensor::mean() const -> Tensor {
    auto s = sum();
    auto n = count_valid();
    if (n == 0) {
        throw std::runtime_error(
            "MaskedTensor::mean: no valid elements (all masked)");
    }
    return s / static_cast<double>(n);
}

inline auto MaskedTensor::max() const -> Tensor {
    // Fill masked positions with the dtype-appropriate minimum so they lose
    // every max comparison.  Using float ±inf for integer dtypes overflows
    // / wraps on cast (audit item A.7).
    auto fill = full_like(data_,
                          masked_tensor_detail::dtype_min_sentinel(data_.dtype()));
    auto filled = tenzor::where(mask_, data_, fill);
    return tenzor::max(filled);
}

inline auto MaskedTensor::min() const -> Tensor {
    // Symmetric to max() — use the dtype-appropriate maximum.
    auto fill = full_like(data_,
                          masked_tensor_detail::dtype_max_sentinel(data_.dtype()));
    auto filled = tenzor::where(mask_, data_, fill);
    return tenzor::min(filled);
}

inline auto MaskedTensor::sum(int64_t dim, bool keepdim) const -> MaskedTensor {
    auto masked_data = data_ * mask_.to(data_.dtype());
    auto result_data = tenzor::sum(masked_data, dim, keepdim);

    // Reduce the mask along dim: any valid element means the output slot is valid.
    // sum(mask, dim) > 0 gives us that.
    auto mask_sum = tenzor::sum(mask_.to(DType::Int64), dim, keepdim);
    // mask_sum > 0 → valid.  We compare by checking != 0 via to(Bool).
    auto result_mask = mask_sum.to(DType::Bool);

    return MaskedTensor(std::move(result_data), std::move(result_mask));
}

inline auto MaskedTensor::mean(int64_t dim, bool keepdim) const -> MaskedTensor {
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

inline auto MaskedTensor::operator+(const MaskedTensor& other) const -> MaskedTensor {
    auto combined_mask = tenzor::logical_and(mask_, other.mask_);
    auto result_data =
        masked_tensor_detail::zero_masked(data_ + other.data_, combined_mask);
    return MaskedTensor(std::move(result_data), std::move(combined_mask));
}

inline auto MaskedTensor::operator-(const MaskedTensor& other) const -> MaskedTensor {
    auto combined_mask = tenzor::logical_and(mask_, other.mask_);
    auto result_data =
        masked_tensor_detail::zero_masked(data_ - other.data_, combined_mask);
    return MaskedTensor(std::move(result_data), std::move(combined_mask));
}

inline auto MaskedTensor::operator*(const MaskedTensor& other) const -> MaskedTensor {
    auto combined_mask = tenzor::logical_and(mask_, other.mask_);
    auto result_data =
        masked_tensor_detail::zero_masked(data_ * other.data_, combined_mask);
    return MaskedTensor(std::move(result_data), std::move(combined_mask));
}

inline auto MaskedTensor::operator/(const MaskedTensor& other) const -> MaskedTensor {
    auto combined_mask = tenzor::logical_and(mask_, other.mask_);
    auto result_data =
        masked_tensor_detail::zero_masked(data_ / other.data_, combined_mask);
    return MaskedTensor(std::move(result_data), std::move(combined_mask));
}

inline auto MaskedTensor::operator+(float scalar) const -> MaskedTensor {
    auto result_data = data_ + static_cast<double>(scalar);
    return MaskedTensor(std::move(result_data), mask_);
}

inline auto MaskedTensor::operator*(float scalar) const -> MaskedTensor {
    auto result_data = data_ * static_cast<double>(scalar);
    return MaskedTensor(std::move(result_data), mask_);
}

inline auto MaskedTensor::to_tensor(float fill_value) const -> Tensor {
    return fill_masked(fill_value);
}

} // namespace tenzor
