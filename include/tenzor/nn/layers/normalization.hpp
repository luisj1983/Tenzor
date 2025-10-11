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
    auto forward(const Variable& input) -> Variable override;

private:
    std::vector<int64_t> normalized_shape_;  ///< Dimensions to normalize over
    double eps_;                             ///< Numerical stability constant
    bool elementwise_affine_;                ///< Whether to learn affine parameters
    int64_t num_features_;                   ///< Product of normalized_shape

    Variable weight_;  ///< Scale parameter gamma
    Variable bias_;    ///< Shift parameter beta

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
    auto forward(const Variable& input) -> Variable override;

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

} // namespace nn
} // namespace tenzor
