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
    auto forward(const Variable& input) -> Variable override;

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
    auto forward(const Variable& input) -> Variable override;

private:
    double p_;  ///< Channel dropout probability
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
    auto forward(const Variable& input) -> Variable override;

private:
    double p_;  ///< Dropout probability
};

} // namespace nn
} // namespace tenzor
