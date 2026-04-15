#pragma once

#include "tensor.hpp"

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

} // namespace tenzor
