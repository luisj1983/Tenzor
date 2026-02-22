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

} // namespace init
} // namespace nn
} // namespace tenzor
