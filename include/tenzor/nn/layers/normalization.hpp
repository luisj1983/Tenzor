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

private:
    int64_t num_features_;
    double eps_;
    bool affine_;

    Variable weight_;
    Variable bias_;

    auto reset_parameters() -> void;
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

} // namespace nn
} // namespace tenzor
