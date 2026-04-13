/**
 * @file nested_tensor.hpp
 * @brief Nested (jagged) tensor class for variable-length sequences
 *
 * NestedTensor stores batches of tensors with variable sizes along one
 * "ragged" dimension using a contiguous values buffer and cumulative
 * offsets, avoiding padding overhead. This is the jagged layout
 * representation.
 *
 * Design mirrors SparseTensor: a separate class that composes regular
 * Tensors rather than subclassing them.
 */

#pragma once

#include "../core/tensor.hpp"
#include <cstdint>
#include <span>
#include <vector>

namespace tenzor {

/**
 * @brief Nested tensor with jagged (variable-length) layout.
 *
 * Stores a batch of tensors with a ragged dimension using:
 * - values_: contiguous buffer [total_ragged_len, *regular_dims]
 * - offsets_: cumulative offsets [batch_size + 1], Int64
 *
 * Element i spans values_[offsets_[i] : offsets_[i+1]].
 *
 * @code
 * auto t0 = tenzor::ones({3, 4}, DType::Float32);
 * auto t1 = tenzor::ones({5, 4}, DType::Float32);
 * auto nt = NestedTensor::from_tensor_list({t0, t1});
 * // nt.batch_size() == 2, nt.numel() == 32
 * @endcode
 */
class NestedTensor {
public:
    // =========================================================================
    // Factory Methods
    // =========================================================================

    /**
     * @brief Create from a list of tensors.
     *
     * All tensors must have same dtype, device, and matching shapes
     * except along dimension 0 (which becomes the ragged dimension).
     *
     * @param tensors Input tensors (must not be empty)
     * @return NestedTensor with jagged layout
     */
    static auto from_tensor_list(std::span<const Tensor> tensors) -> NestedTensor;

    /// @overload Convenience overload for initializer lists.
    static inline auto from_tensor_list(std::initializer_list<Tensor> tensors) -> NestedTensor {
        std::vector<Tensor> v(tensors);
        return from_tensor_list(std::span<const Tensor>(v));
    }

    /// @overload Convenience overload for vectors.
    static inline auto from_tensor_list(const std::vector<Tensor>& tensors) -> NestedTensor {
        return from_tensor_list(std::span<const Tensor>(tensors));
    }

    /**
     * @brief Create from a padded tensor and per-element lengths.
     *
     * @param padded Padded tensor of shape [B, max_len, ...]
     * @param lengths Per-element lengths of shape [B], Int64
     * @return NestedTensor with padding stripped
     */
    static auto from_padded(const Tensor& padded, const Tensor& lengths) -> NestedTensor;

    /**
     * @brief Create directly from pre-existing jagged components.
     *
     * @param values Contiguous values buffer [total_len, *regular_dims]
     * @param offsets Cumulative offsets [B+1], Int64
     * @param ragged_dim Ragged dimension index (default: 1)
     * @return NestedTensor
     */
    static auto from_jagged(Tensor values, Tensor offsets,
                            int64_t ragged_dim = 1) -> NestedTensor;

    // =========================================================================
    // Properties
    // =========================================================================

    auto is_nested() const -> bool { return true; }
    auto batch_size() const -> int64_t { return batch_size_; }
    auto ragged_dim() const -> int64_t { return ragged_dim_; }
    auto values() const -> const Tensor& { return values_; }
    auto offsets() const -> const Tensor& { return offsets_; }
    auto dtype() const -> DType { return values_.dtype(); }
    auto device() const -> Device { return values_.device(); }
    auto numel() const -> int64_t { return values_.numel(); }

    /**
     * @brief Number of dimensions (batch + ragged + regular dims).
     */
    auto ndim() const -> int64_t;

    /**
     * @brief Length of each element along the ragged dimension.
     * @return Int64 tensor of shape [batch_size]
     */
    auto lengths() const -> Tensor;

    /**
     * @brief Per-element shapes as a vector of vectors.
     */
    auto nested_sizes() const -> std::vector<std::vector<int64_t>>;

    /**
     * @brief Trailing regular (non-ragged) dimensions.
     */
    auto regular_shape() const -> const std::vector<int64_t>& { return regular_shape_; }

    /**
     * @brief Maximum length along the ragged dimension.
     */
    auto max_length() const -> int64_t;

    // =========================================================================
    // Conversion
    // =========================================================================

    /**
     * @brief Convert to padded dense tensor.
     *
     * @param padding_value Value for padded positions (default: 0.0)
     * @return Dense tensor of shape [B, max_len, *regular_shape]
     */
    auto to_padded_tensor(double padding_value = 0.0) const -> Tensor;

    /**
     * @brief Unbind into individual tensors.
     * @return Vector of B tensors, one per batch element
     */
    auto unbind() const -> std::vector<Tensor>;

    /**
     * @brief Select a single element from the batch.
     *
     * @param index Batch index (supports negative indexing)
     * @return Tensor for element at given index
     */
    auto select(int64_t index) const -> Tensor;

    // =========================================================================
    // Device / DType Transfer
    // =========================================================================

    auto to(Device device) const -> NestedTensor;
    auto to(DType dtype) const -> NestedTensor;
    auto contiguous() const -> NestedTensor;
    auto clone() const -> NestedTensor;

    // =========================================================================
    // Gradient Tracking
    // =========================================================================

    auto requires_grad() const -> bool;
    auto requires_grad_(bool requires_grad = true) -> NestedTensor&;

private:
    NestedTensor(Tensor values, Tensor offsets, int64_t batch_size,
                 int64_t ragged_dim, std::vector<int64_t> regular_shape);

    Tensor values_;                      ///< [total_ragged_len, *regular_dims]
    Tensor offsets_;                      ///< [batch_size + 1], Int64, cumulative
    int64_t batch_size_;
    int64_t ragged_dim_;                  ///< typically 1
    std::vector<int64_t> regular_shape_;  ///< trailing regular dims
};

// =========================================================================
// Free Factory Function
// =========================================================================

/**
 * @brief Convenience factory: create a NestedTensor from a vector of tensors.
 */
auto nested_tensor(std::vector<Tensor> tensors) -> NestedTensor;

} // namespace tenzor
