/**
 * @file normalization.hpp
 * @brief Layer and group normalization layers
 *
 * Implements alternative normalization techniques to batch normalization
 * that are more stable for small batches or variable sequence lengths.
 */

#pragma once

#include "../module.hpp"
#include <vector>

namespace tenzor {
namespace nn {

/**
 * @brief Layer Normalization.
 *
 * Normalizes over specified feature dimensions instead of batch dimension.
 * Unlike batch normalization, layer normalization normalizes across features
 * for each sample independently, making it suitable for RNNs and small batches.
 *
 * Normalization formula:
 * - y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
 *
 * Where E[x] and Var[x] are computed over the normalized_shape dimensions.
 *
 * Shape:
 * - Input: (*, normalized_shape) where * is any number of dimensions
 * - Output: Same as input
 * - Weight (gamma): normalized_shape if elementwise_affine=true
 * - Bias (beta): normalized_shape if elementwise_affine=true
 *
 * @code
 * // Normalize over last 2 dimensions
 * LayerNorm ln({512, 768});
 *
 * Variable x(Tensor({batch, 512, 768}, DType::Float32, Device::cpu()), true);
 * Variable normalized = ln.forward(x);  // Normalize over 512x768
 * @endcode
 *
 * @see BatchNorm2d for batch normalization
 * @see GroupNorm for group normalization
 */
class LayerNorm : public Module {
public:
    /**
     * @brief Construct layer normalization.
     *
     * @param normalized_shape Dimensions to normalize over (from the end)
     * @param eps Small constant for numerical stability (default: 1e-5)
     * @param elementwise_affine If true, learn affine parameters (default: true)
     *
     * @code
     * LayerNorm ln1({768});           // Normalize last dimension
     * LayerNorm ln2({64, 64});        // Normalize last 2 dimensions
     * LayerNorm ln3({3, 32, 32});     // Normalize last 3 dimensions
     * @endcode
     */
    LayerNorm(std::vector<int64_t> normalized_shape,
              double eps = 1e-5,
              bool elementwise_affine = true);

    /**
     * @brief Forward pass through layer normalization.
     *
     * @param input Input variable with shape ending in normalized_shape
     * @return Normalized output (same shape as input)
     *
     * @throws std::runtime_error if input shape doesn't match normalized_shape
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /// @brief Get the epsilon value used for numerical stability.
    [[nodiscard]] auto eps() const -> double { return eps_; }

    auto extra_repr() const -> std::string override {
        std::string shape_str = "(";
        for (size_t i = 0; i < normalized_shape_.size(); ++i) {
            if (i > 0) shape_str += ", ";
            shape_str += std::to_string(normalized_shape_[i]);
        }
        shape_str += ")";
        return "normalized_shape=" + shape_str +
               ", eps=" + std::to_string(eps_) +
               ", elementwise_affine=" + std::string(elementwise_affine_ ? "True" : "False");
    }

private:
    std::vector<int64_t> normalized_shape_;  ///< Dimensions to normalize over
    double eps_;                             ///< Numerical stability constant
    bool elementwise_affine_;                ///< Whether to learn affine parameters
    int64_t num_features_;                   ///< Product of normalized_shape

    Variable weight_;  ///< Scale parameter gamma (moved-from after register_parameter)
    Variable bias_;    ///< Shift parameter beta (moved-from after register_parameter)

    // Cached pointers to parameters_ entries (avoids ~2-3ms hash map lookup overhead)
    std::shared_ptr<Variable> cached_weight_;
    std::shared_ptr<Variable> cached_bias_;

    /**
     * @brief Initialize affine parameters to ones and zeros.
     */
    auto reset_parameters() -> void;
};

/**
 * @brief Group Normalization.
 *
 * Divides channels into groups and normalizes each group independently.
 * More stable than batch normalization for small batches, and doesn't
 * depend on batch size like batch normalization.
 *
 * Normalization formula:
 * - y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
 *
 * Where E[x] and Var[x] are computed within each group.
 *
 * Shape:
 * - Input: (N, C, *) where * is spatial dimensions
 * - Output: Same as input
 * - Weight (gamma): (C) if affine=true
 * - Bias (beta): (C) if affine=true
 *
 * @code
 * // 32 channels divided into 8 groups (4 channels per group)
 * GroupNorm gn(8, 32);
 *
 * Variable x(Tensor({batch, 32, 64, 64}, DType::Float32, Device::cpu()), true);
 * Variable normalized = gn.forward(x);  // Same shape
 * @endcode
 *
 * @note num_channels must be divisible by num_groups
 *
 * @see LayerNorm for layer normalization
 * @see BatchNorm2d for batch normalization
 */
class GroupNorm : public Module {
public:
    /**
     * @brief Construct group normalization.
     *
     * @param num_groups Number of groups to divide channels into
     * @param num_channels Number of channels (C dimension)
     * @param eps Small constant for numerical stability (default: 1e-5)
     * @param affine If true, learn affine parameters (default: true)
     *
     * @throws std::invalid_argument if num_channels not divisible by num_groups
     *
     * @code
     * GroupNorm gn1(8, 32);    // 32 channels, 8 groups
     * GroupNorm gn2(16, 64);   // 64 channels, 16 groups
     * GroupNorm gn3(1, 32);    // Layer norm equivalent
     * GroupNorm gn4(32, 32);   // Instance norm equivalent
     * @endcode
     */
    GroupNorm(int64_t num_groups,
              int64_t num_channels,
              double eps = 1e-5,
              bool affine = true);

    /**
     * @brief Forward pass through group normalization.
     *
     * @param input Input variable of shape (N, C, *)
     * @return Normalized output (same shape as input)
     *
     * @throws std::runtime_error if input channels don't match num_channels
     */
    auto forward_impl(const Variable& input) -> Variable override;

    auto extra_repr() const -> std::string override {
        return "num_groups=" + std::to_string(num_groups_) +
               ", num_channels=" + std::to_string(num_channels_) +
               ", eps=" + std::to_string(eps_) +
               ", affine=" + std::string(affine_ ? "True" : "False");
    }

    [[nodiscard]] auto num_groups() const -> int64_t { return num_groups_; }
    [[nodiscard]] auto num_channels() const -> int64_t { return num_channels_; }
    [[nodiscard]] auto eps() const -> double { return eps_; }
    [[nodiscard]] auto affine() const -> bool { return affine_; }

private:
    int64_t num_groups_;    ///< Number of groups
    int64_t num_channels_;  ///< Number of channels
    double eps_;            ///< Numerical stability constant
    bool affine_;           ///< Whether to learn affine parameters

    Variable weight_;  ///< Scale parameter gamma [num_channels]
    Variable bias_;    ///< Shift parameter beta [num_channels]

    /**
     * @brief Initialize affine parameters to ones and zeros.
     */
    auto reset_parameters() -> void;
};

/**
 * @brief Instance Normalization for 2D inputs (4D tensors).
 *
 * Normalizes each channel independently across spatial dimensions for each
 * sample in the batch. Equivalent to GroupNorm with num_groups = num_channels.
 *
 * Shape:
 * - Input: (N, C, H, W)
 * - Output: Same as input
 * - Weight (gamma): (C) if affine=true
 * - Bias (beta): (C) if affine=true
 *
 * @code
 * InstanceNorm2d in(64);  // 64 channels
 * Variable x(Tensor({batch, 64, 32, 32}, DType::Float32, Device::cpu()), true);
 * Variable normalized = in.forward(x);
 * @endcode
 *
 * @see GroupNorm for group normalization
 * @see BatchNorm2d for batch normalization
 */
class InstanceNorm2d : public Module {
public:
    /**
     * @brief Construct 2D instance normalization.
     *
     * @param num_features Number of channels (C dimension)
     * @param eps Small constant for numerical stability (default: 1e-5)
     * @param affine If true, learn affine parameters (default: true)
     */
    InstanceNorm2d(int64_t num_features,
                   double eps = 1e-5,
                   bool affine = true);

    auto forward_impl(const Variable& input) -> Variable override;

    /// @brief Get the epsilon value used for numerical stability.
    [[nodiscard]] auto eps() const -> double { return eps_; }

private:
    int64_t num_features_;
    double eps_;
    bool affine_;

    Variable weight_;
    Variable bias_;

    auto reset_parameters() -> void;
};

/**
 * @brief Instance Normalization for 1D inputs (3D tensors).
 *
 * Normalizes each channel independently for each sample in the batch.
 * For 1D temporal data of shape (N, C, L).
 *
 * Shape:
 * - Input: (N, C, L)
 * - Output: Same as input
 * - Weight (gamma): (C) if affine=true
 * - Bias (beta): (C) if affine=true
 *
 * @code
 * InstanceNorm1d in(64);  // 64 channels
 * Variable x(Tensor({batch, 64, 100}, DType::Float32, Device::cpu()), true);
 * Variable normalized = in.forward(x);
 * @endcode
 */
class InstanceNorm1d : public Module {
public:
    /**
     * @brief Construct 1D instance normalization.
     *
     * @param num_features Number of channels (C dimension)
     * @param eps Small constant for numerical stability (default: 1e-5)
     * @param affine If true, learn affine parameters (default: true)
     */
    InstanceNorm1d(int64_t num_features,
                   double eps = 1e-5,
                   bool affine = true);

    auto forward_impl(const Variable& input) -> Variable override;

    /// @brief Get the epsilon value used for numerical stability.
    [[nodiscard]] auto eps() const -> double { return eps_; }

private:
    int64_t num_features_;
    double eps_;
    bool affine_;

    Variable weight_;
    Variable bias_;

    auto reset_parameters() -> void;
};

/**
 * @brief Instance Normalization for 3D inputs (5D tensors).
 *
 * Normalizes each channel independently across spatial dimensions (D, H, W)
 * for each sample in the batch. Internally reshapes to 4D and delegates
 * to InstanceNorm2d.
 *
 * Shape:
 * - Input: (N, C, D, H, W)
 * - Output: Same as input
 *
 * @see InstanceNorm2d for 2D instance normalization
 */
class InstanceNorm3d : public Module {
public:
    InstanceNorm3d(int64_t num_features,
                   double eps = 1e-5,
                   bool affine = true);

    auto forward_impl(const Variable& input) -> Variable override;

    /// @brief Get the epsilon value used for numerical stability (forwarded from the
    /// internal InstanceNorm2d delegate).
    [[nodiscard]] auto eps() const -> double { return in2d_.eps(); }

private:
    int64_t num_features_;
    InstanceNorm2d in2d_; ///< Delegate to InstanceNorm2d after reshaping
};

/**
 * @brief Root Mean Square Layer Normalization.
 *
 * Simplified variant of LayerNorm that only uses root mean square for
 * normalization, without centering (mean subtraction). This is more
 * computationally efficient and works well for transformer models.
 *
 * Normalization formula:
 * - y = x / sqrt(mean(x^2) + eps) * weight
 *
 * Unlike LayerNorm, RMSNorm:
 * - Does not subtract the mean
 * - Has no bias term (only scale)
 * - Is more efficient to compute
 *
 * Shape:
 * - Input: (*, normalized_shape) where * is any number of dimensions
 * - Output: Same as input
 * - Weight: normalized_shape
 *
 * @code
 * // Normalize over last dimension (hidden size)
 * RMSNorm rms(768);
 *
 * Variable x(Tensor({batch, seq_len, 768}, DType::Float32, Device::cpu()), true);
 * Variable normalized = rms.forward(x);  // Normalize over 768
 * @endcode
 *
 * @see LayerNorm for standard layer normalization
 */
class RMSNorm : public Module {
public:
    /**
     * @brief Construct RMS normalization.
     *
     * @param normalized_shape Size of the last dimension to normalize
     * @param eps Small constant for numerical stability (default: 1e-6)
     *
     * @code
     * RMSNorm rms1(768);    // For BERT-style transformers
     * RMSNorm rms2(4096);   // For large language models
     * @endcode
     */
    RMSNorm(int64_t normalized_shape, double eps = 1e-6);

    /**
     * @brief Forward pass through RMS normalization.
     *
     * @param input Input variable with last dimension matching normalized_shape
     * @return Normalized output (same shape as input)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /// @brief Numerical-stability epsilon (exposed for serializers).
    [[nodiscard]] auto eps() const -> double { return eps_; }

    /// @brief Size of the last dim to normalize (exposed for serializers).
    [[nodiscard]] auto normalized_shape() const -> int64_t { return normalized_shape_; }

private:
    int64_t normalized_shape_;  ///< Size of dimension to normalize
    double eps_;                ///< Numerical stability constant

    Variable weight_;  ///< Scale parameter

    // Cached pointer to parameters_ entry (avoids ~2-3ms hash map lookup overhead)
    std::shared_ptr<Variable> cached_weight_;

    /**
     * @brief Initialize weight to ones.
     */
    auto reset_parameters() -> void;
};

/**
 * @brief Local Response Normalization.
 *
 * Applies LRN across channels at each spatial position:
 * y_i = x_i / (k + alpha/size * sum(x_j^2, local_window))^beta
 *
 * where the local window spans `size` channels centered on channel i.
 *
 * Originally used in AlexNet. The normalization window covers
 * channels [max(0, i - floor(size/2)), min(C-1, i + floor(size/2))].
 *
 * Shape:
 * - Input: (N, C, *) where * is any number of spatial dimensions
 * - Output: Same as input
 *
 * @code
 * LocalResponseNorm lrn(5);  // Window size 5, default alpha/beta/k
 * Variable x(Tensor({batch, 64, 32, 32}, DType::Float32, Device::cpu()), true);
 * Variable normalized = lrn.forward(x);
 * @endcode
 *
 * @see LayerNorm for layer normalization
 * @see BatchNorm2d for batch normalization
 */
class LocalResponseNorm : public Module {
public:
    /**
     * @brief Construct Local Response Normalization.
     *
     * @param size Number of channels in the normalization window
     * @param alpha Multiplicative factor (default: 1e-4)
     * @param beta Exponent (default: 0.75)
     * @param k Additive constant (default: 1.0)
     *
     * @code
     * LocalResponseNorm lrn1(5);                    // AlexNet defaults
     * LocalResponseNorm lrn2(5, 1e-4, 0.75, 2.0);  // Custom k
     * @endcode
     */
    LocalResponseNorm(int64_t size,
                      double alpha = 1e-4,
                      double beta = 0.75,
                      double k = 1.0);

    /**
     * @brief Forward pass through local response normalization.
     *
     * @param input Input variable of shape (N, C, *)
     * @return Normalized output (same shape as input)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    auto extra_repr() const -> std::string override {
        return "size=" + std::to_string(size_) +
               ", alpha=" + std::to_string(alpha_) +
               ", beta=" + std::to_string(beta_) +
               ", k=" + std::to_string(k_);
    }

private:
    int64_t size_;   ///< Number of channels in normalization window
    double alpha_;   ///< Multiplicative factor
    double beta_;    ///< Exponent
    double k_;       ///< Additive constant
};

// Factories for the *Backward grad_fns (defined in normalization.cpp).
// Used by F::layer_norm / F::group_norm / F::instance_norm in functional.cpp
// to wire up backward properly — the functional path otherwise returns a
// Variable without a grad_fn and silently drops gradients (raw-tensor-op
// breaks autograd graph pattern).
namespace internal {
auto make_layer_norm_backward(bool elementwise_affine, double eps,
                              int64_t normalized_size,
                              std::vector<::tenzor::Tensor> tensors_to_save)
    -> std::shared_ptr<::tenzor::Function>;

// A.4 multi-output JVP walker: overload that captures the full
// normalized_shape so the LayerNormBackward function's saved_attributes()
// can expose AttrKey::NormalizedShape for the registered multi-output
// JVP rule (which reduces over the last K=normalized_shape.size() axes).
auto make_layer_norm_backward(bool elementwise_affine, double eps,
                              int64_t normalized_size,
                              std::vector<int64_t> normalized_shape,
                              std::vector<::tenzor::Tensor> tensors_to_save)
    -> std::shared_ptr<::tenzor::Function>;

auto make_group_norm_backward(bool affine, double eps,
                              int64_t num_groups, int64_t num_channels,
                              int64_t group_size,
                              std::vector<::tenzor::Tensor> tensors_to_save)
    -> std::shared_ptr<::tenzor::Function>;

auto make_instance_norm_backward(bool affine, double eps,
                                 int64_t num_features,
                                 std::vector<::tenzor::Tensor> tensors_to_save)
    -> std::shared_ptr<::tenzor::Function>;
}  // namespace internal

} // namespace nn
} // namespace tenzor
