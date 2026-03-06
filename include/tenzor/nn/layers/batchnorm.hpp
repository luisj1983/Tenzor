/**
 * @file batchnorm.hpp
 * @brief Batch normalization layers for neural networks
 *
 * Implements batch normalization for 1D and 2D inputs to reduce
 * internal covariate shift and stabilize training.
 */

#pragma once

#include "../module.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Batch Normalization for 2D inputs (images).
 *
 * Normalizes activations across the batch dimension for each feature channel.
 * Commonly used in convolutional networks after convolution layers.
 *
 * Normalization formula:
 * - y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
 *
 * Where E[x] and Var[x] are computed per-channel across batch and spatial dimensions.
 *
 * Shape:
 * - Input: (N, C, H, W)
 * - Output: (N, C, H, W) (same as input)
 * - Weight (gamma): (C) if affine=true
 * - Bias (beta): (C) if affine=true
 * - Running mean: (C)
 * - Running variance: (C)
 *
 * @code
 * BatchNorm2d bn(64);  // 64 feature channels
 *
 * Variable x(Tensor({batch, 64, 32, 32}, DType::Float32, Device::cpu()), true);
 * Variable normalized = bn.forward(x);  // Same shape
 * @endcode
 *
 * @see BatchNorm1d for 1D batch normalization
 */
class BatchNorm2d : public Module {
public:
    /**
     * @brief Construct batch normalization layer for 2D data.
     *
     * @param num_features Number of feature channels (C dimension)
     * @param eps Small constant for numerical stability (default: 1e-5)
     * @param momentum Momentum for running statistics update (default: 0.1)
     * @param affine If true, learn scale and shift parameters (default: true)
     * @param track_running_stats If true, track running mean/var (default: true)
     *
     * @note Momentum convention: running_stat = (1-momentum)*running_stat + momentum*batch_stat.
     *       This matches PyTorch's convention where momentum=0.1 means 10% of the new batch
     *       statistics are blended in. A momentum of 0.0 means running stats are never updated.
     */
    BatchNorm2d(int64_t num_features,
                double eps = 1e-5,
                double momentum = 0.1,
                bool affine = true,
                bool track_running_stats = true);

    /**
     * @brief Forward pass through batch normalization.
     *
     * In training mode, normalizes using batch statistics and updates running statistics.
     * In evaluation mode, normalizes using running statistics.
     *
     * @param input Input variable of shape (N, C, H, W)
     * @return Normalized output of shape (N, C, H, W)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /// @brief Get the epsilon value used for numerical stability.
    [[nodiscard]] auto eps() const -> double { return eps_; }

    auto extra_repr() const -> std::string override {
        return "num_features=" + std::to_string(num_features_) +
               ", eps=" + std::to_string(eps_) +
               ", momentum=" + std::to_string(momentum_) +
               ", affine=" + std::string(affine_ ? "True" : "False") +
               ", track_running_stats=" + std::string(track_running_stats_ ? "True" : "False");
    }

private:
    int64_t num_features_;          ///< Number of feature channels
    double eps_;                    ///< Numerical stability constant
    double momentum_;               ///< Running statistics momentum
    bool affine_;                   ///< Whether to learn affine parameters
    bool track_running_stats_;      ///< Whether to track running statistics

    Variable weight_;               ///< Scale parameter gamma [num_features]
    Variable bias_;                 ///< Shift parameter beta [num_features]
    Variable running_mean_;         ///< Running mean [num_features]
    Variable running_var_;          ///< Running variance [num_features]
    Variable num_batches_tracked_;  ///< Count of batches processed (scalar Int64 buffer)

    // Cached pointers to parameters_ entries (avoids ~2-3ms hash map lookup overhead)
    std::shared_ptr<Variable> cached_weight_;
    std::shared_ptr<Variable> cached_bias_;

    /**
     * @brief Initialize affine parameters.
     */
    auto reset_parameters() -> void;
};

/**
 * @brief Batch Normalization for 1D inputs (sequences).
 *
 * Normalizes activations across the batch dimension for each feature channel.
 * Commonly used for fully connected layers or sequential data.
 *
 * Shape:
 * - Input: (N, C) or (N, C, L)
 * - Output: Same as input
 *
 * @code
 * BatchNorm1d bn(128);  // 128 features
 *
 * Variable x(Tensor({batch, 128}, DType::Float32, Device::cpu()), true);
 * Variable normalized = bn.forward(x);
 * @endcode
 *
 * @see BatchNorm2d for 2D batch normalization
 */
class BatchNorm1d : public Module {
public:
    /**
     * @brief Construct batch normalization layer for 1D data.
     *
     * @param num_features Number of feature channels
     * @param eps Small constant for numerical stability (default: 1e-5)
     * @param momentum Momentum for running statistics update (default: 0.1)
     * @param affine If true, learn scale and shift parameters (default: true)
     * @param track_running_stats If true, track running mean/var (default: true)
     */
    BatchNorm1d(int64_t num_features,
                double eps = 1e-5,
                double momentum = 0.1,
                bool affine = true,
                bool track_running_stats = true);

    /**
     * @brief Forward pass through batch normalization.
     *
     * @param input Input variable of shape (N, C) or (N, C, L)
     * @return Normalized output (same shape as input)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /// @brief Get the epsilon value used for numerical stability.
    [[nodiscard]] auto eps() const -> double { return eps_; }

    auto extra_repr() const -> std::string override {
        return "num_features=" + std::to_string(num_features_) +
               ", eps=" + std::to_string(eps_) +
               ", momentum=" + std::to_string(momentum_) +
               ", affine=" + std::string(affine_ ? "True" : "False") +
               ", track_running_stats=" + std::string(track_running_stats_ ? "True" : "False");
    }

private:
    int64_t num_features_;          ///< Number of feature channels
    double eps_;                    ///< Numerical stability constant
    double momentum_;               ///< Running statistics momentum
    bool affine_;                   ///< Whether to learn affine parameters
    bool track_running_stats_;      ///< Whether to track running statistics

    Variable weight_;               ///< Scale parameter gamma
    Variable bias_;                 ///< Shift parameter beta
    Variable running_mean_;         ///< Running mean
    Variable running_var_;          ///< Running variance
    Variable num_batches_tracked_;  ///< Count of batches processed (scalar Int64 buffer)

    // Cached pointers to parameters_ entries (avoids ~2-3ms hash map lookup overhead)
    std::shared_ptr<Variable> cached_weight_;
    std::shared_ptr<Variable> cached_bias_;

    /**
     * @brief Initialize affine parameters.
     */
    auto reset_parameters() -> void;
};

/**
 * @brief Batch Normalization for 5D inputs (N, C, D, H, W).
 *
 * Normalizes over the (N, D, H, W) dimensions for each channel.
 * Internally reshapes to 4D, delegates to BatchNorm2d, then reshapes back.
 */
class BatchNorm3d : public Module {
public:
    BatchNorm3d(int64_t num_features,
                double eps = 1e-5,
                double momentum = 0.1,
                bool affine = true,
                bool track_running_stats = true);

    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t num_features_;
    BatchNorm2d bn2d_; ///< Delegate to BatchNorm2d after reshaping
};

} // namespace nn
} // namespace tenzor
