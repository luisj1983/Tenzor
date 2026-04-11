/**
 * @file dropout.hpp
 * @brief Dropout regularization layers
 *
 * Implements dropout and its variants for regularizing neural networks
 * by randomly zeroing elements during training.
 */

#pragma once

#include "../module.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Standard dropout layer.
 *
 * During training, randomly zeros elements with probability p using samples
 * from a Bernoulli distribution. Remaining elements are scaled by 1/(1-p).
 * During evaluation, performs identity operation (no dropout).
 *
 * Dropout is a regularization technique that prevents co-adaptation of neurons
 * and reduces overfitting.
 *
 * Shape:
 * - Input: Any shape
 * - Output: Same as input
 *
 * @code
 * Dropout dropout(0.5);  // Drop 50% of elements
 *
 * Variable x(Tensor({batch, 512}, DType::Float32, Device::cpu()), true);
 * Variable dropped = dropout.forward(x);  // Training mode
 *
 * dropout.eval();  // Switch to evaluation mode
 * Variable kept = dropout.forward(x);  // No dropout applied
 * @endcode
 *
 * @see Dropout2d for spatial dropout
 * @see AlphaDropout for SELU networks
 */
class Dropout : public Module {
public:
    /**
     * @brief Construct dropout layer.
     *
     * @param p Probability of an element being zeroed (default: 0.5)
     *
     * @throws std::invalid_argument if p not in [0, 1]
     *
     * @code
     * Dropout dropout1(0.5);   // 50% dropout rate
     * Dropout dropout2(0.2);   // 20% dropout rate
     * Dropout dropout3(0.0);   // No dropout
     * @endcode
     */
    explicit Dropout(double p = 0.5);

    /**
     * @brief Forward pass through dropout.
     *
     * In training mode: randomly zeros elements and scales remainder.
     * In evaluation mode: returns input unchanged.
     *
     * @param input Input variable of any shape
     * @return Output variable (same shape)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    auto extra_repr() const -> std::string override {
        return "p=" + std::to_string(p_);
    }

private:
    double p_;  ///< Dropout probability
};

/**
 * @brief 2D spatial dropout layer.
 *
 * Randomly zeros entire feature channels instead of individual elements.
 * For inputs with shape (N, C, H, W), drops entire C x H x W feature maps.
 *
 * This is particularly useful for convolutional layers as it encourages
 * independence between feature maps rather than individual pixels.
 *
 * Shape:
 * - Input: (N, C, H, W)
 * - Output: (N, C, H, W)
 *
 * @code
 * Dropout2d dropout(0.2);  // Drop 20% of feature maps
 *
 * Variable x(Tensor({batch, 64, 32, 32}, DType::Float32, Device::cpu()), true);
 * Variable dropped = dropout.forward(x);  // Some entire channels zeroed
 * @endcode
 *
 * @see Dropout for element-wise dropout
 */
class Dropout2d : public Module {
public:
    /**
     * @brief Construct 2D spatial dropout layer.
     *
     * @param p Probability of a channel being zeroed (default: 0.5)
     *
     * @throws std::invalid_argument if p not in [0, 1]
     */
    explicit Dropout2d(double p = 0.5);

    /**
     * @brief Forward pass through spatial dropout.
     *
     * @param input Input variable of shape (N, C, H, W)
     * @return Output variable with some channels zeroed
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    double p_;  ///< Channel dropout probability
};

/**
 * @brief 3D spatial dropout layer.
 *
 * Randomly zeros entire feature channels across all spatial dimensions.
 * For inputs with shape (N, C, D, H, W), drops entire C x D x H x W maps.
 * The 3D analogue of Dropout2d; use for volumetric / 3D-conv networks.
 *
 * Shape:
 * - Input: (N, C, D, H, W) or (C, D, H, W)
 * - Output: same shape
 */
class Dropout3d : public Module {
public:
    explicit Dropout3d(double p = 0.5);
    auto forward_impl(const Variable& input) -> Variable override;
private:
    double p_;
};

/**
 * @brief Alpha dropout for SELU networks.
 *
 * Maintains the self-normalizing property of SELU networks by using
 * a dropout variant that preserves mean and variance.
 *
 * Unlike standard dropout, alpha dropout multiplies dropped elements
 * by a negative saturation value rather than zero, which helps maintain
 * the mean near zero and variance near one.
 *
 * Should be used exclusively with SELU activation function.
 *
 * @code
 * AlphaDropout dropout(0.1);  // For SELU networks
 *
 * Variable x(Tensor({batch, 512}, DType::Float32, Device::cpu()), true);
 * Variable activated = selu(x);
 * Variable dropped = dropout.forward(activated);  // Self-normalizing preserved
 * @endcode
 *
 * @see Dropout for standard dropout
 * @note Only use with SELU activation
 */
class AlphaDropout : public Module {
public:
    /**
     * @brief Construct alpha dropout layer.
     *
     * @param p Probability of an element being dropped (default: 0.5)
     *
     * @throws std::invalid_argument if p not in [0, 1]
     */
    explicit AlphaDropout(double p = 0.5);

    /**
     * @brief Forward pass through alpha dropout.
     *
     * @param input Input variable (typically after SELU activation)
     * @return Output variable with self-normalizing property preserved
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    double p_;  ///< Dropout probability
};

/**
 * @brief Variational dropout for RNNs (Gal & Ghahramani 2016).
 *
 * Maintains the same dropout mask across time steps within a sequence.
 * Call reset_mask() at the start of each new sequence/batch to generate
 * a new mask. The mask shape is determined by the first forward call
 * after reset.
 *
 * For 3D input (T, B, F), the mask has shape (1, B, F) and broadcasts
 * across the time dimension so that the same neurons are dropped at
 * every time step.
 *
 * Shape:
 * - Input: (seq_len, batch, features) or (batch, features)
 * - Output: Same as input
 *
 * @code
 * VariationalDropout vdrop(0.3);
 * vdrop.reset_mask();  // New sequence
 * for (int t = 0; t < seq_len; ++t) {
 *     auto h = vdrop.forward(hidden[t]);  // Same mask at each step
 * }
 * @endcode
 *
 * @see Dropout for standard element-wise dropout
 */
class VariationalDropout : public Module {
public:
    /**
     * @brief Construct variational dropout layer.
     *
     * @param p Probability of an element being zeroed (default: 0.5)
     *
     * @throws std::invalid_argument if p not in [0, 1]
     */
    explicit VariationalDropout(double p = 0.5);

    /**
     * @brief Forward pass through variational dropout.
     *
     * In training mode: applies cached mask (generates on first call after reset).
     * In evaluation mode: returns input unchanged.
     *
     * @param input Input variable of shape (T, B, F) or (B, F)
     * @return Output variable with same dropout pattern across time steps
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Reset the dropout mask for a new sequence/batch.
     *
     * Must be called before processing a new sequence to generate a fresh mask.
     */
    auto reset_mask() -> void;

    auto extra_repr() const -> std::string override {
        return "p=" + std::to_string(p_);
    }

private:
    double p_;              ///< Dropout probability
    Tensor mask_;           ///< Cached mask (reused across time steps)
    bool mask_valid_{false}; ///< Whether mask_ has been generated for current sequence
};

} // namespace nn
} // namespace tenzor
