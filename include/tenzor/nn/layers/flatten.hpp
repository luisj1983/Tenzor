/**
 * @file flatten.hpp
 * @brief Flatten layer for reshaping tensors
 *
 * Implements tensor flattening for transitioning between convolutional
 * and fully connected layers.
 */

#pragma once

#include "../module.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Flatten layer for reshaping multi-dimensional inputs.
 *
 * Flattens a contiguous range of dimensions into a single dimension.
 * Commonly used to transition from convolutional layers (4D tensors)
 * to fully connected layers (2D tensors).
 *
 * Shape transformation:
 * - Input: (dim_0, dim_1, ..., dim_n)
 * - Output: (dim_0, ..., dim_start-1, flattened_dims, dim_end+1, ..., dim_n)
 *
 * Where flattened_dims is the product of dimensions from start_dim to end_dim.
 *
 * @code
 * Flatten flatten;  // Default: flatten from dim 1 onwards
 *
 * // Flatten spatial dimensions after conv layers
 * Variable x(Tensor({batch, 64, 7, 7}, DType::Float32, Device::cpu()), true);
 * Variable flat = flatten.forward(x);  // Shape: {batch, 3136}
 *
 * // Can now pass to fully connected layer
 * Linear fc(3136, 10);
 * Variable logits = fc.forward(flat);
 * @endcode
 *
 * @note By default preserves batch dimension (start_dim=1)
 *
 * @see Linear for fully connected layers
 */
class Flatten : public Module {
public:
    /**
     * @brief Construct flatten layer.
     *
     * @param start_dim First dimension to flatten (default: 1, preserve batch)
     * @param end_dim Last dimension to flatten (default: -1, all remaining)
     *
     * Negative indices count from the end: -1 is last dimension, -2 is second-to-last, etc.
     *
     * @code
     * Flatten flatten1;          // start_dim=1, end_dim=-1 (preserve batch)
     * Flatten flatten2(0);       // Flatten all dimensions to 1D
     * Flatten flatten3(2, 3);    // Flatten only dims 2 and 3
     * Flatten flatten4(1, -2);   // Flatten from dim 1 to second-to-last
     * @endcode
     */
    explicit Flatten(int64_t start_dim = 1, int64_t end_dim = -1);

    /**
     * @brief Forward pass through flatten layer.
     *
     * Reshapes input by flattening specified dimension range into single dimension.
     * No data copying occurs - this is a view operation.
     *
     * @param input Input variable to flatten
     * @return Flattened output variable
     *
     * @code
     * // Common usage after convolutions
     * Variable conv_out(Tensor({32, 128, 8, 8}, DType::Float32, Device::cpu()), true);
     * Flatten flatten;
     * Variable flat = flatten.forward(conv_out);  // Shape: {32, 8192}
     * @endcode
     */
    auto forward(const Variable& input) -> Variable override;

private:
    int64_t start_dim_;  ///< First dimension to include in flattening
    int64_t end_dim_;    ///< Last dimension to include in flattening
};

} // namespace nn
} // namespace tenzor
