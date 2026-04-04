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
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t start_dim_;  ///< First dimension to include in flattening
    int64_t end_dim_;    ///< Last dimension to include in flattening
};

/**
 * @brief Unflatten layer for reshaping a dimension into multiple dimensions.
 *
 * The inverse of Flatten. Expands a single dimension into a specified shape.
 *
 * @code
 * Unflatten unflatten(1, {2, 3});  // Split dim 1 into [2, 3]
 * // Input shape: {batch, 6} -> Output shape: {batch, 2, 3}
 * @endcode
 */
class Unflatten : public Module {
public:
    explicit Unflatten(int64_t dim, std::vector<int64_t> sizes);
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t dim_;
    std::vector<int64_t> sizes_;
};

/**
 * @brief PixelShuffle layer for sub-pixel convolution upscaling.
 *
 * Rearranges elements in a tensor of shape (*, C*r^2, H, W) to (*, C, H*r, W*r)
 * where r is the upscale_factor. Used in super-resolution networks.
 *
 * @code
 * PixelShuffle ps(2);  // upscale_factor=2
 * // Input shape: {1, 8, 4, 4} -> Output shape: {1, 2, 8, 8}
 * @endcode
 */
class PixelShuffle : public Module {
public:
    explicit PixelShuffle(int64_t upscale_factor);
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t upscale_factor_;
};

/**
 * @brief PixelUnshuffle layer — the inverse of PixelShuffle.
 *
 * Rearranges elements in a tensor of shape (*, C, H*r, W*r) to (*, C*r^2, H, W).
 *
 * @code
 * PixelUnshuffle pus(2);  // downscale_factor=2
 * // Input shape: {1, 2, 8, 8} -> Output shape: {1, 8, 4, 4}
 * @endcode
 */
class PixelUnshuffle : public Module {
public:
    explicit PixelUnshuffle(int64_t downscale_factor);
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t downscale_factor_;
};

/**
 * @brief ChannelShuffle layer for cross-group information exchange.
 *
 * Rearranges channels by splitting into groups, transposing, and flattening
 * back. Used in ShuffleNet architectures to enable information flow between
 * grouped convolution channels.
 *
 * Shape transformation (4D input):
 * - (B, C, H, W) -> (B, G, C/G, H, W) -> permute(0,2,1,3,4) -> (B, C, H, W)
 *
 * @code
 * ChannelShuffle cs(2);  // 2 groups
 * // Input shape: {1, 4, 8, 8}
 * // Channels [0,1] and [2,3] get interleaved: [0,2,1,3]
 * @endcode
 */
class ChannelShuffle : public Module {
public:
    explicit ChannelShuffle(int64_t groups);
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t groups_;
};

} // namespace nn
} // namespace tenzor
