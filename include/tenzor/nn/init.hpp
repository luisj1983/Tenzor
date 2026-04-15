/**
 * @file init.hpp
 * @brief Weight initialization utilities for neural network parameters
 *
 * Provides common weight initialization strategies used in deep learning.
 * These functions modify tensors in-place (following PyTorch convention
 * with trailing underscore names).
 *
 * @code
 * #include <tenzor/nn/init.hpp>
 *
 * Tensor weight({256, 512}, DType::Float32, Device::cpu());
 * tenzor::nn::init::kaiming_uniform_(weight);
 *
 * Tensor bias({256}, DType::Float32, Device::cpu());
 * tenzor::nn::init::zeros_(bias);
 * @endcode
 */

#pragma once

#include <cstdint>
#include <utility>
#include <string>
#include "../core/tensor.hpp"

namespace tenzor {
namespace nn {
namespace init {

/**
 * @brief Nonlinearity types for fan calculation in Kaiming initialization.
 */
enum class FanMode {
    FanIn,   ///< Use fan_in for variance computation (default for forward pass)
    FanOut   ///< Use fan_out for variance computation (for backward pass)
};

/**
 * @brief Calculate fan_in and fan_out for a weight tensor.
 *
 * Handles 1D (linear), 2D (linear), 3D (conv1d), 4D (conv2d), 5D (conv3d).
 * - fan_in = num_input_features * receptive_field_size
 * - fan_out = num_output_features * receptive_field_size
 *
 * @param tensor Weight tensor
 * @return Pair of (fan_in, fan_out)
 * @throws std::invalid_argument if tensor has fewer than 1 dimension
 */
auto calculate_fan_in_and_fan_out(const Tensor& tensor) -> std::pair<int64_t, int64_t>;

/**
 * @brief Calculate the recommended gain for a given nonlinearity.
 *
 * Returns the gain value to use with Xavier/Kaiming initialization
 * for a specific activation function.
 *
 * @param nonlinearity Activation function name ("linear", "relu", "leaky_relu",
 *                     "tanh", "sigmoid", "selu")
 * @param param Optional parameter (negative_slope for leaky_relu, default 0.01)
 * @return Recommended gain value
 */
auto calculate_gain(const std::string& nonlinearity, double param = 0.01) -> double;

// ============================================================================
// Xavier (Glorot) Initialization
// ============================================================================

/**
 * @brief Xavier uniform initialization (Glorot uniform).
 *
 * Fills tensor with values from U(-a, a) where:
 * a = gain * sqrt(6 / (fan_in + fan_out))
 *
 * Designed to maintain variance of activations and gradients
 * through linear layers with sigmoid/tanh activations.
 *
 * @param tensor Tensor to initialize (modified in-place)
 * @param gain Scaling factor (default: 1.0)
 * @return Reference to the modified tensor
 *
 * @see calculate_gain for recommended gain values
 */
auto xavier_uniform_(Tensor& tensor, double gain = 1.0) -> Tensor&;

/**
 * @brief Xavier normal initialization (Glorot normal).
 *
 * Fills tensor with values from N(0, std^2) where:
 * std = gain * sqrt(2 / (fan_in + fan_out))
 *
 * @param tensor Tensor to initialize (modified in-place)
 * @param gain Scaling factor (default: 1.0)
 * @return Reference to the modified tensor
 */
auto xavier_normal_(Tensor& tensor, double gain = 1.0) -> Tensor&;

// ============================================================================
// Kaiming (He) Initialization
// ============================================================================

/**
 * @brief Kaiming uniform initialization (He uniform).
 *
 * Fills tensor with values from U(-bound, bound) where:
 * bound = gain * sqrt(3 / fan)
 *
 * Designed for layers followed by ReLU or Leaky ReLU activations.
 *
 * @param tensor Tensor to initialize (modified in-place)
 * @param a Negative slope for leaky_relu (default: 0, i.e. ReLU)
 * @param mode FanIn or FanOut (default: FanIn)
 * @param nonlinearity Activation function name (default: "leaky_relu")
 * @return Reference to the modified tensor
 */
auto kaiming_uniform_(Tensor& tensor,
                      double a = 0.0,
                      FanMode mode = FanMode::FanIn,
                      const std::string& nonlinearity = "leaky_relu") -> Tensor&;

/**
 * @brief Kaiming normal initialization (He normal).
 *
 * Fills tensor with values from N(0, std^2) where:
 * std = gain / sqrt(fan)
 *
 * @param tensor Tensor to initialize (modified in-place)
 * @param a Negative slope for leaky_relu (default: 0, i.e. ReLU)
 * @param mode FanIn or FanOut (default: FanIn)
 * @param nonlinearity Activation function name (default: "leaky_relu")
 * @return Reference to the modified tensor
 */
auto kaiming_normal_(Tensor& tensor,
                     double a = 0.0,
                     FanMode mode = FanMode::FanIn,
                     const std::string& nonlinearity = "leaky_relu") -> Tensor&;

// ============================================================================
// LeCun Initialization
// ============================================================================

/**
 * @brief LeCun uniform initialization.
 *
 * Fills tensor with values from U(-limit, limit) where:
 * limit = sqrt(3 / fan_in)
 *
 * Recommended for use with SELU activations.
 *
 * @param tensor Tensor to initialize (modified in-place)
 * @return Reference to the modified tensor
 */
auto lecun_uniform_(Tensor& tensor) -> Tensor&;

/**
 * @brief LeCun normal initialization.
 *
 * Fills tensor with values from N(0, 1/fan_in).
 *
 * Recommended for use with SELU activations.
 *
 * @param tensor Tensor to initialize (modified in-place)
 * @return Reference to the modified tensor
 */
auto lecun_normal_(Tensor& tensor) -> Tensor&;

// ============================================================================
// Orthogonal Initialization
// ============================================================================

/**
 * @brief Orthogonal initialization.
 *
 * Fills the tensor with a (semi) orthogonal matrix using SVD decomposition.
 * For 2D tensors: directly computes orthogonal matrix.
 * For higher-dimensional tensors: flattens to 2D, computes orthogonal matrix,
 * then reshapes back.
 *
 * @param tensor Tensor to initialize (modified in-place, must be >= 2D)
 * @param gain Scaling factor (default: 1.0)
 * @return Reference to the modified tensor
 * @throws std::invalid_argument if tensor has fewer than 2 dimensions
 */
auto orthogonal_(Tensor& tensor, double gain = 1.0) -> Tensor&;

// ============================================================================
// Simple Initialization
// ============================================================================

/**
 * @brief Fill tensor with values from uniform distribution U(low, high).
 *
 * @param tensor Tensor to initialize (modified in-place)
 * @param low Lower bound (default: 0.0)
 * @param high Upper bound (default: 1.0)
 * @return Reference to the modified tensor
 */
auto uniform_(Tensor& tensor, double low = 0.0, double high = 1.0) -> Tensor&;

/**
 * @brief Fill tensor with values from normal distribution N(mean, std^2).
 *
 * @param tensor Tensor to initialize (modified in-place)
 * @param mean Mean of the distribution (default: 0.0)
 * @param std Standard deviation (default: 1.0)
 * @return Reference to the modified tensor
 */
auto normal_(Tensor& tensor, double mean = 0.0, double std = 1.0) -> Tensor&;

/**
 * @brief Fill tensor with a constant value.
 *
 * @param tensor Tensor to initialize (modified in-place)
 * @param value Fill value
 * @return Reference to the modified tensor
 */
auto constant_(Tensor& tensor, double value) -> Tensor&;

/**
 * @brief Fill tensor with zeros.
 *
 * @param tensor Tensor to initialize (modified in-place)
 * @return Reference to the modified tensor
 */
auto zeros_(Tensor& tensor) -> Tensor&;

/**
 * @brief Fill tensor with ones.
 *
 * @param tensor Tensor to initialize (modified in-place)
 * @return Reference to the modified tensor
 */
auto ones_(Tensor& tensor) -> Tensor&;

// ============================================================================
// Truncated Normal Initialization
// ============================================================================

/**
 * @brief Truncated normal initialization.
 *
 * Fills tensor with values from a truncated normal distribution.
 * Values outside [a, b] are redrawn. Uses CDF-based inversion for
 * efficiency: samples uniform in [Phi(a), Phi(b)] and applies
 * inverse normal CDF.
 *
 * Widely used in Vision Transformers (ViT, DeiT, Swin).
 *
 * @param tensor Tensor to initialize (modified in-place)
 * @param mean Mean of the normal distribution (default: 0.0)
 * @param std Standard deviation (default: 1.0)
 * @param a Lower bound for truncation (default: -2.0)
 * @param b Upper bound for truncation (default: 2.0)
 * @return Reference to the modified tensor
 */
auto trunc_normal_(Tensor& tensor,
                   double mean = 0.0,
                   double std = 1.0,
                   double a = -2.0,
                   double b = 2.0) -> Tensor&;

// ============================================================================
// Dirac Initialization
// ============================================================================

/**
 * @brief Dirac delta initialization for convolutional layers.
 *
 * Fills the tensor so that the convolution acts as an identity mapping
 * for matching input/output channels. Sets the center element of each
 * channel-to-channel kernel to 1, all others to 0.
 *
 * Requires tensor to be 3D (Conv1d), 4D (Conv2d), or 5D (Conv3d).
 * When groups > 1, sets weight[i, i/groups, center...] = 1.
 *
 * @param tensor Tensor to initialize (modified in-place, must be 3-5D)
 * @param groups Number of groups for grouped convolution (default: 1)
 * @return Reference to the modified tensor
 * @throws std::invalid_argument if tensor is not 3-5D or dimensions are incompatible
 */
auto dirac_(Tensor& tensor, int64_t groups = 1) -> Tensor&;

// ============================================================================
// Sparse Initialization
// ============================================================================

/**
 * @brief Sparse initialization.
 *
 * Fills tensor with a normal distribution N(0, std^2), then zeros out
 * a fraction of elements in each column. The sparsity parameter controls
 * the fraction of elements set to zero.
 *
 * Requires tensor to be 2D.
 *
 * @param tensor Tensor to initialize (modified in-place, must be 2D)
 * @param sparsity Fraction of elements per column to be zero (0 to 1)
 * @param std Standard deviation of the normal distribution (default: 0.01)
 * @return Reference to the modified tensor
 * @throws std::invalid_argument if tensor is not 2D or sparsity is out of [0, 1)
 */
auto sparse_(Tensor& tensor, double sparsity, double std = 0.01) -> Tensor&;

} // namespace init
} // namespace nn
} // namespace tenzor
