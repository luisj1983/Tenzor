/**
 * @file lazy_conv.hpp
 * @brief Lazy convolutional neural network layers (1D, 2D, 3D)
 *
 * Implements lazy convolutional layers that defer weight initialization until
 * the first forward pass, when the input channel dimension becomes known.
 * This is useful when building networks where intermediate channel counts
 * are not known at construction time.
 */

#pragma once

#include <memory>
#include "../module.hpp"
#include "conv.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Lazy 1D convolutional layer.
 *
 * A variant of Conv1d that infers in_channels from the first input tensor.
 * The internal Conv1d is not created until the first forward() call,
 * at which point input.shape()[1] is used as in_channels.
 *
 * After materialization, this layer behaves identically to nn::Conv1d.
 *
 * Shape transformations (after materialization):
 * - Input: (N, C_in, L_in)
 * - Output: (N, C_out, L_out)
 *
 * @code
 * // Create lazy layer - in_channels determined at first forward()
 * LazyConv1d lazy_conv(256, 3, 1, 1);
 *
 * // Parameters not yet allocated
 * assert(lazy_conv.parameters().empty());
 * assert(!lazy_conv.is_materialized());
 *
 * // First forward call materializes parameters
 * Variable x(Tensor({batch, 128, seq_len}, DType::Float32, Device::cpu()), true);
 * Variable output = lazy_conv.forward(x);  // in_channels inferred as 128
 *
 * // Now parameters exist
 * assert(lazy_conv.is_materialized());
 * @endcode
 *
 * @see Conv1d for the eager initialization variant
 * @see Module for base class interface
 */
class LazyConv1d : public Module {
public:
    /**
     * @brief Construct lazy 1D convolutional layer.
     *
     * Only out_channels is specified; in_channels is inferred from
     * the first input tensor's channel dimension (dim 1).
     *
     * @param out_channels Number of output channels (filters)
     * @param kernel_size Size of convolving kernel
     * @param stride Stride of convolution (default: 1)
     * @param padding Zero-padding added to both sides (default: 0)
     * @param dilation Spacing between kernel elements (default: 1)
     * @param groups Number of blocked connections (default: 1)
     * @param bias If true, add learnable bias after materialization (default: true)
     */
    LazyConv1d(int64_t out_channels, int64_t kernel_size,
               int64_t stride = 1, int64_t padding = 0,
               int64_t dilation = 1, int64_t groups = 1, bool bias = true);

    /**
     * @brief Forward pass through lazy 1D convolution.
     *
     * On the first call, inspects input.shape()[1] to determine in_channels,
     * creates an internal Conv1d, then delegates forward to it.
     * Subsequent calls delegate directly to the internal Conv1d.
     *
     * @param input Input variable of shape (N, C_in, L)
     * @return Output variable of shape (N, C_out, L_out)
     *
     * @throws std::runtime_error if input has fewer than 3 dimensions
     * @throws std::runtime_error if input's channel dimension is <= 0
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Check if parameters have been materialized.
     *
     * @return true if internal Conv1d has been created
     */
    auto is_materialized() const -> bool { return conv_ != nullptr; }

    /**
     * @brief Get all parameters.
     *
     * Returns empty vector if not yet materialized, otherwise delegates
     * to base Module which recurses into the internal Conv1d submodule.
     *
     * @return Vector of shared pointers to parameters
     */
    auto parameters() -> std::vector<std::shared_ptr<Variable>> override;

    /**
     * @brief Get only this module's direct parameters (not submodules').
     *
     * Returns empty vector if not yet materialized.
     *
     * @return Vector of shared pointers to this module's own parameters only
     */
    auto own_parameters() -> std::vector<std::shared_ptr<Variable>> override;

    /**
     * @brief Get all parameters with names.
     *
     * Returns empty vector if not yet materialized.
     *
     * @return Vector of (name, shared_ptr to parameter) pairs
     */
    auto named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> override;

private:
    int64_t out_channels_;   ///< Number of output channels
    int64_t kernel_size_;    ///< Kernel size
    int64_t stride_;         ///< Stride
    int64_t padding_;        ///< Padding
    int64_t dilation_;       ///< Dilation
    int64_t groups_;         ///< Number of groups
    bool has_bias_;          ///< Whether this layer has bias
    int64_t in_channels_{0}; ///< Inferred input channels (set on first forward)

    std::shared_ptr<Conv1d> conv_;  ///< Internal Conv1d, created on first forward

    /**
     * @brief Create the internal Conv1d with inferred in_channels.
     *
     * @param in_channels Inferred input channel count
     * @param device Device to create parameters on
     */
    auto materialize(int64_t in_channels, Device device) -> void;
};

/**
 * @brief Lazy 2D convolutional layer.
 *
 * A variant of Conv2d that infers in_channels from the first input tensor.
 * The internal Conv2d is not created until the first forward() call,
 * at which point input.shape()[1] is used as in_channels.
 *
 * After materialization, this layer behaves identically to nn::Conv2d.
 *
 * Shape transformations (after materialization):
 * - Input: (N, C_in, H_in, W_in)
 * - Output: (N, C_out, H_out, W_out)
 *
 * @code
 * // Create lazy layer - in_channels determined at first forward()
 * LazyConv2d lazy_conv(64, 3, 1, 1);
 *
 * // First forward call materializes parameters
 * Variable x(Tensor({batch, 3, 32, 32}, DType::Float32, Device::cpu()), true);
 * Variable output = lazy_conv.forward(x);  // in_channels inferred as 3
 * @endcode
 *
 * @see Conv2d for the eager initialization variant
 * @see Module for base class interface
 */
class LazyConv2d : public Module {
public:
    /**
     * @brief Construct lazy 2D convolutional layer with square kernel.
     *
     * Only out_channels is specified; in_channels is inferred from
     * the first input tensor's channel dimension (dim 1).
     *
     * @param out_channels Number of output channels (filters)
     * @param kernel_size Size of convolving kernel (square)
     * @param stride Stride of convolution (default: 1)
     * @param padding Zero-padding added to both sides (default: 0)
     * @param dilation Spacing between kernel elements (default: 1)
     * @param groups Number of blocked connections (default: 1)
     * @param bias If true, add learnable bias after materialization (default: true)
     */
    LazyConv2d(int64_t out_channels, int64_t kernel_size,
               int64_t stride = 1, int64_t padding = 0,
               int64_t dilation = 1, int64_t groups = 1, bool bias = true);

    /**
     * @brief Forward pass through lazy 2D convolution.
     *
     * On the first call, inspects input.shape()[1] to determine in_channels,
     * creates an internal Conv2d, then delegates forward to it.
     * Subsequent calls delegate directly to the internal Conv2d.
     *
     * @param input Input variable of shape (N, C_in, H, W)
     * @return Output variable of shape (N, C_out, H_out, W_out)
     *
     * @throws std::runtime_error if input has fewer than 4 dimensions
     * @throws std::runtime_error if input's channel dimension is <= 0
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Check if parameters have been materialized.
     *
     * @return true if internal Conv2d has been created
     */
    auto is_materialized() const -> bool { return conv_ != nullptr; }

    /**
     * @brief Get all parameters.
     *
     * Returns empty vector if not yet materialized, otherwise delegates
     * to base Module which recurses into the internal Conv2d submodule.
     *
     * @return Vector of shared pointers to parameters
     */
    auto parameters() -> std::vector<std::shared_ptr<Variable>> override;

    /**
     * @brief Get only this module's direct parameters (not submodules').
     *
     * Returns empty vector if not yet materialized.
     *
     * @return Vector of shared pointers to this module's own parameters only
     */
    auto own_parameters() -> std::vector<std::shared_ptr<Variable>> override;

    /**
     * @brief Get all parameters with names.
     *
     * Returns empty vector if not yet materialized.
     *
     * @return Vector of (name, shared_ptr to parameter) pairs
     */
    auto named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> override;

private:
    int64_t out_channels_;   ///< Number of output channels
    int64_t kernel_size_;    ///< Kernel size
    int64_t stride_;         ///< Stride
    int64_t padding_;        ///< Padding
    int64_t dilation_;       ///< Dilation
    int64_t groups_;         ///< Number of groups
    bool has_bias_;          ///< Whether this layer has bias
    int64_t in_channels_{0}; ///< Inferred input channels (set on first forward)

    std::shared_ptr<Conv2d> conv_;  ///< Internal Conv2d, created on first forward

    /**
     * @brief Create the internal Conv2d with inferred in_channels.
     *
     * @param in_channels Inferred input channel count
     * @param device Device to create parameters on
     */
    auto materialize(int64_t in_channels, Device device) -> void;
};

/**
 * @brief Lazy 3D convolutional layer.
 *
 * A variant of Conv3d that infers in_channels from the first input tensor.
 * The internal Conv3d is not created until the first forward() call,
 * at which point input.shape()[1] is used as in_channels.
 *
 * After materialization, this layer behaves identically to nn::Conv3d.
 *
 * Shape transformations (after materialization):
 * - Input: (N, C_in, D_in, H_in, W_in)
 * - Output: (N, C_out, D_out, H_out, W_out)
 *
 * @code
 * // Create lazy layer - in_channels determined at first forward()
 * LazyConv3d lazy_conv(32, 3, 1, 1);
 *
 * // First forward call materializes parameters
 * Variable x(Tensor({batch, 1, 16, 64, 64}, DType::Float32, Device::cpu()), true);
 * Variable output = lazy_conv.forward(x);  // in_channels inferred as 1
 * @endcode
 *
 * @see Conv3d for the eager initialization variant
 * @see Module for base class interface
 */
class LazyConv3d : public Module {
public:
    /**
     * @brief Construct lazy 3D convolutional layer.
     *
     * Only out_channels is specified; in_channels is inferred from
     * the first input tensor's channel dimension (dim 1).
     *
     * @param out_channels Number of output channels (filters)
     * @param kernel_size Size of convolving kernel
     * @param stride Stride of convolution (default: 1)
     * @param padding Zero-padding added to all sides (default: 0)
     * @param dilation Spacing between kernel elements (default: 1)
     * @param groups Number of blocked connections (default: 1)
     * @param bias If true, add learnable bias after materialization (default: true)
     */
    LazyConv3d(int64_t out_channels, int64_t kernel_size,
               int64_t stride = 1, int64_t padding = 0,
               int64_t dilation = 1, int64_t groups = 1, bool bias = true);

    /**
     * @brief Forward pass through lazy 3D convolution.
     *
     * On the first call, inspects input.shape()[1] to determine in_channels,
     * creates an internal Conv3d, then delegates forward to it.
     * Subsequent calls delegate directly to the internal Conv3d.
     *
     * @param input Input variable of shape (N, C_in, D, H, W)
     * @return Output variable of shape (N, C_out, D_out, H_out, W_out)
     *
     * @throws std::runtime_error if input has fewer than 5 dimensions
     * @throws std::runtime_error if input's channel dimension is <= 0
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Check if parameters have been materialized.
     *
     * @return true if internal Conv3d has been created
     */
    auto is_materialized() const -> bool { return conv_ != nullptr; }

    /**
     * @brief Get all parameters.
     *
     * Returns empty vector if not yet materialized, otherwise delegates
     * to base Module which recurses into the internal Conv3d submodule.
     *
     * @return Vector of shared pointers to parameters
     */
    auto parameters() -> std::vector<std::shared_ptr<Variable>> override;

    /**
     * @brief Get only this module's direct parameters (not submodules').
     *
     * Returns empty vector if not yet materialized.
     *
     * @return Vector of shared pointers to this module's own parameters only
     */
    auto own_parameters() -> std::vector<std::shared_ptr<Variable>> override;

    /**
     * @brief Get all parameters with names.
     *
     * Returns empty vector if not yet materialized.
     *
     * @return Vector of (name, shared_ptr to parameter) pairs
     */
    auto named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> override;

private:
    int64_t out_channels_;   ///< Number of output channels
    int64_t kernel_size_;    ///< Kernel size
    int64_t stride_;         ///< Stride
    int64_t padding_;        ///< Padding
    int64_t dilation_;       ///< Dilation
    int64_t groups_;         ///< Number of groups
    bool has_bias_;          ///< Whether this layer has bias
    int64_t in_channels_{0}; ///< Inferred input channels (set on first forward)

    std::shared_ptr<Conv3d> conv_;  ///< Internal Conv3d, created on first forward

    /**
     * @brief Create the internal Conv3d with inferred in_channels.
     *
     * @param in_channels Inferred input channel count
     * @param device Device to create parameters on
     */
    auto materialize(int64_t in_channels, Device device) -> void;
};

} // namespace nn
} // namespace tenzor
