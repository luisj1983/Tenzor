/**
 * @file drop_path.hpp
 * @brief DropPath (Stochastic Depth) regularization layer
 *
 * Drops entire samples (batch-level) rather than individual elements,
 * used in residual architectures like Swin Transformer and DeiT.
 */

#pragma once

#include "../module.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief DropPath (Stochastic Depth) regularization.
 *
 * During training, randomly drops entire samples from the batch with
 * probability p. Each sample is independently kept or dropped, and
 * surviving samples are scaled by 1/(1-p) to maintain expected values.
 *
 * Unlike standard Dropout which zeros individual elements, DropPath
 * creates a Bernoulli mask of shape (batch, 1, 1, ..., 1) so entire
 * samples are dropped together. This is the correct regularization for
 * residual connections in vision transformers.
 *
 * During evaluation, performs identity operation.
 *
 * Shape:
 * - Input: (N, ...) any shape with batch dimension first
 * - Output: Same as input
 *
 * @code
 * DropPath drop_path(0.1);  // 10% drop rate
 *
 * // In residual connection:
 * auto out = x + drop_path.forward(block(x));
 * @endcode
 *
 * @see Dropout for element-wise dropout
 * @see Dropout2d for channel-wise dropout
 */
class DropPath : public Module {
public:
    /**
     * @brief Construct DropPath layer.
     *
     * @param p Probability of dropping a sample (default: 0.0, i.e. no drop)
     *
     * @throws std::invalid_argument if p not in [0, 1]
     */
    explicit DropPath(double p = 0.0);

    /**
     * @brief Forward pass through DropPath.
     *
     * In training mode: randomly drops entire samples and scales survivors.
     * In evaluation mode: returns input unchanged.
     *
     * @param input Input variable with batch dimension first
     * @return Output variable (same shape)
     */
    auto forward_impl(const Variable& input) -> Variable override;

private:
    double p_;  ///< Drop probability
};

} // namespace nn
} // namespace tenzor
