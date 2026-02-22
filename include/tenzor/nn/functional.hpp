/**
 * @file functional.hpp
 * @brief Functional interface for neural network operations
 *
 * Provides stateless functional versions of all NN operations, similar to
 * PyTorch's torch.nn.functional (commonly aliased as F).
 *
 * These are thin wrappers around existing autograd-aware operations,
 * providing a consistent namespace for functional-style usage.
 *
 * @code
 * namespace F = tenzor::nn::functional;
 * auto out = F::relu(input);
 * auto attn = F::softmax(scores, -1);
 * @endcode
 */

#pragma once

#include "../autograd/variable.hpp"
#include "../autograd/ops.hpp"
#include "activations/activations.hpp"
#include "loss/losses.hpp"
#include "../ops/creation.hpp"
#include "../core/tensor.hpp"

namespace tenzor::nn::functional {

// ============================================================================
// Activation Functions
// ============================================================================

/** @brief Functional ReLU: max(0, x) */
inline auto relu(const Variable& input) -> Variable {
    return nn::relu(input);
}

/** @brief Functional Leaky ReLU */
inline auto leaky_relu(const Variable& input, double negative_slope = 0.01) -> Variable {
    return nn::leaky_relu(input, negative_slope);
}

/** @brief Functional sigmoid */
inline auto sigmoid(const Variable& input) -> Variable {
    return nn::sigmoid(input);
}

/** @brief Functional tanh */
inline auto tanh(const Variable& input) -> Variable {
    return nn::tanh(input);
}

/** @brief Functional GELU
 *  @param approximate "none" for exact, "tanh" for fast approximation
 */
inline auto gelu(const Variable& input, const std::string& approximate = "none") -> Variable {
    return nn::gelu(input, approximate);
}

/** @brief Functional ELU */
inline auto elu(const Variable& input, double alpha = 1.0) -> Variable {
    return nn::elu(input, alpha);
}

/** @brief Functional SELU */
inline auto selu(const Variable& input) -> Variable {
    return nn::selu(input);
}

/** @brief Functional Swish/SiLU */
inline auto swish(const Variable& input) -> Variable {
    return nn::swish(input);
}

/** @brief Functional Mish */
inline auto mish(const Variable& input) -> Variable {
    return nn::mish(input);
}

/** @brief Functional softmax along specified dimension */
inline auto softmax(const Variable& input, int64_t dim = -1) -> Variable {
    return nn::softmax(input, dim);
}

/** @brief Functional log-softmax (numerically stable) */
inline auto log_softmax(const Variable& input, int64_t dim = -1) -> Variable {
    return nn::log_softmax(input, dim);
}

// ============================================================================
// Linear Algebra
// ============================================================================

/** @brief Functional linear layer: y = x @ W^T + b */
inline auto linear(const Variable& input, const Variable& weight,
                    const Variable& bias) -> Variable {
    return tenzor::linear(input, weight, bias);
}

/** @brief Functional matrix multiplication with autograd */
inline auto matmul(const Variable& a, const Variable& b) -> Variable {
    return tenzor::matmul(a, b);
}

// ============================================================================
// Loss Functions (functional versions)
// ============================================================================

/** @brief Functional cross-entropy loss */
inline auto cross_entropy(const Variable& input, const Tensor& target,
                           Reduction reduction = Reduction::Mean,
                           float label_smoothing = 0.0f) -> Variable {
    CrossEntropyLoss loss(reduction, label_smoothing);
    return loss(input, target);
}

/** @brief Functional MSE loss */
inline auto mse_loss(const Variable& input, const Variable& target,
                      Reduction reduction = Reduction::Mean) -> Variable {
    MSELoss loss(reduction);
    return loss(input, target);
}

} // namespace tenzor::nn::functional
