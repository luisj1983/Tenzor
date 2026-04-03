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
 * auto conv_out = F::conv2d(input, weight, bias, {1,1}, {1,1});
 * @endcode
 */

#pragma once

#include "../autograd/variable.hpp"
#include "../autograd/ops.hpp"
#include "activations/activations.hpp"
#include "loss/losses.hpp"
#include "../ops/creation.hpp"
#include "../core/tensor.hpp"

#include <optional>
#include <utility>

// Forward declarations to avoid heavy includes — implementations in functional.cpp
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
// Convolution (implemented in functional.cpp)
// ============================================================================

/**
 * @brief Functional 2D convolution
 *
 * Applies a 2D convolution over an input signal [N, C_in, H, W].
 *
 * @param input Input tensor [N, C_in, H, W]
 * @param weight Convolution filters [C_out, C_in/groups, kH, kW]
 * @param bias Optional bias [C_out]
 * @param stride Stride of the convolution (H, W)
 * @param padding Zero-padding added to both sides (H, W)
 * @param dilation Spacing between kernel elements (H, W)
 * @param groups Number of blocked connections
 */
auto conv2d(const Variable& input, const Variable& weight,
            const std::optional<Variable>& bias = std::nullopt,
            std::pair<int64_t, int64_t> stride = {1, 1},
            std::pair<int64_t, int64_t> padding = {0, 0},
            std::pair<int64_t, int64_t> dilation = {1, 1},
            int64_t groups = 1) -> Variable;

// ============================================================================
// Pooling (implemented in functional.cpp)
// ============================================================================

/**
 * @brief Functional 2D max pooling
 *
 * @param input Input tensor [N, C, H, W]
 * @param kernel_size Size of the pooling window
 * @param stride Stride of the pooling window (default: kernel_size)
 * @param padding Zero-padding added to both sides
 */
auto max_pool2d(const Variable& input,
                std::pair<int64_t, int64_t> kernel_size,
                std::pair<int64_t, int64_t> stride = {-1, -1},
                std::pair<int64_t, int64_t> padding = {0, 0}) -> Variable;

/**
 * @brief Functional 2D average pooling
 *
 * @param input Input tensor [N, C, H, W]
 * @param kernel_size Size of the pooling window
 * @param stride Stride of the pooling window (default: kernel_size)
 * @param padding Zero-padding added to both sides
 */
auto avg_pool2d(const Variable& input,
                std::pair<int64_t, int64_t> kernel_size,
                std::pair<int64_t, int64_t> stride = {-1, -1},
                std::pair<int64_t, int64_t> padding = {0, 0}) -> Variable;

/**
 * @brief Functional adaptive 2D average pooling
 *
 * Automatically selects kernel/stride to produce the target output size.
 *
 * @param input Input tensor [N, C, H, W]
 * @param output_size Target output spatial dimensions (H, W)
 */
auto adaptive_avg_pool2d(const Variable& input,
                         std::pair<int64_t, int64_t> output_size) -> Variable;

// ============================================================================
// Normalization (implemented in functional.cpp)
// ============================================================================

/**
 * @brief Functional batch normalization
 *
 * @param input Input tensor [N, C, ...] (2D, 3D, or 4D)
 * @param running_mean Running mean [C] (used in eval mode)
 * @param running_var Running variance [C] (used in eval mode)
 * @param weight Optional affine weight (gamma) [C]
 * @param bias Optional affine bias (beta) [C]
 * @param training If true, use batch statistics; if false, use running stats
 * @param momentum Momentum for running stats update
 * @param eps Small constant for numerical stability
 */
auto batch_norm(const Variable& input,
                const Tensor& running_mean,
                const Tensor& running_var,
                const std::optional<Variable>& weight = std::nullopt,
                const std::optional<Variable>& bias = std::nullopt,
                bool training = false,
                double momentum = 0.1,
                double eps = 1e-5) -> Variable;

/**
 * @brief Functional layer normalization
 *
 * @param input Input tensor
 * @param normalized_shape Shape of the last N dimensions to normalize over
 * @param weight Optional affine weight [normalized_shape]
 * @param bias Optional affine bias [normalized_shape]
 * @param eps Small constant for numerical stability
 */
auto layer_norm(const Variable& input,
                std::vector<int64_t> normalized_shape,
                const std::optional<Variable>& weight = std::nullopt,
                const std::optional<Variable>& bias = std::nullopt,
                double eps = 1e-5) -> Variable;

// ============================================================================
// Dropout (implemented in functional.cpp)
// ============================================================================

/**
 * @brief Functional dropout
 *
 * Randomly zeroes elements with probability p during training.
 * Uses inverted dropout (scales by 1/(1-p)) so no scaling is needed at eval.
 *
 * @param input Input variable
 * @param p Probability of an element to be zeroed (default: 0.5)
 * @param training Apply dropout if true, identity if false
 */
auto dropout(const Variable& input, double p = 0.5, bool training = true) -> Variable;

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

/** @brief Functional L1 loss */
inline auto l1_loss(const Variable& input, const Variable& target,
                     Reduction reduction = Reduction::Mean) -> Variable {
    auto diff = tenzor::abs(input - target);
    if (reduction == Reduction::Mean) return tenzor::mean(diff);
    if (reduction == Reduction::Sum) return tenzor::sum(diff);
    return diff;
}

/** @brief Functional binary cross entropy with logits */
inline auto binary_cross_entropy_with_logits(
        const Variable& input, const Variable& target,
        Reduction reduction = Reduction::Mean) -> Variable {
    // Numerically stable: max(x, 0) - x*y + log(1 + exp(-|x|))
    auto max_val = nn::relu(input);
    auto neg_abs = tenzor::neg(tenzor::abs(input));
    auto loss = max_val - input * target + tenzor::log(tenzor::exp(neg_abs) + 1.0f);
    if (reduction == Reduction::Mean) return tenzor::mean(loss);
    if (reduction == Reduction::Sum) return tenzor::sum(loss);
    return loss;
}

// ============================================================================
// Additional Normalization (implemented in functional.cpp)
// ============================================================================

/**
 * @brief Functional group normalization
 *
 * @param input Input tensor [N, C, ...]
 * @param num_groups Number of groups to divide channels into
 * @param weight Optional affine weight [C]
 * @param bias Optional affine bias [C]
 * @param eps Small constant for numerical stability
 */
auto group_norm(const Variable& input, int64_t num_groups,
                const std::optional<Variable>& weight = std::nullopt,
                const std::optional<Variable>& bias = std::nullopt,
                double eps = 1e-5) -> Variable;

/**
 * @brief Functional instance normalization
 *
 * @param input Input tensor [N, C, H, W]
 * @param running_mean Running mean [C] (optional, for eval mode)
 * @param running_var Running variance [C] (optional, for eval mode)
 * @param weight Optional affine weight [C]
 * @param bias Optional affine bias [C]
 * @param training Use batch statistics if true
 * @param momentum Momentum for running stats
 * @param eps Small constant for numerical stability
 */
auto instance_norm(const Variable& input,
                   const std::optional<Tensor>& running_mean = std::nullopt,
                   const std::optional<Tensor>& running_var = std::nullopt,
                   const std::optional<Variable>& weight = std::nullopt,
                   const std::optional<Variable>& bias = std::nullopt,
                   bool training = false,
                   double momentum = 0.1,
                   double eps = 1e-5) -> Variable;

// ============================================================================
// Embedding (implemented in functional.cpp)
// ============================================================================

/**
 * @brief Functional embedding lookup
 *
 * @param input Index tensor (LongTensor)
 * @param weight Embedding weight matrix [num_embeddings, embedding_dim]
 */
auto embedding(const Tensor& input, const Variable& weight) -> Variable;

// ============================================================================
// Spatial (implemented in functional.cpp)
// ============================================================================

/**
 * @brief Functional interpolation (resize)
 *
 * @param input Input tensor [N, C, H, W]
 * @param size Target output size (H, W)
 * @param mode Interpolation mode: "nearest", "bilinear", "bicubic"
 * @param align_corners Align corners for bilinear/bicubic
 */
auto interpolate(const Variable& input,
                 std::pair<int64_t, int64_t> size,
                 const std::string& mode = "nearest",
                 bool align_corners = false) -> Variable;

// ============================================================================
// Additional Loss Functions (functional versions)
// ============================================================================

/**
 * @brief Functional negative log likelihood loss
 *
 * @param input Log-probabilities [N, C]
 * @param target Class indices [N] (LongTensor)
 * @param reduction Reduction mode
 */
auto nll_loss(const Variable& input, const Tensor& target,
              Reduction reduction = Reduction::Mean) -> Variable;

/**
 * @brief Functional smooth L1 loss (Huber loss)
 *
 * @param input Predicted values
 * @param target Target values
 * @param reduction Reduction mode
 * @param beta Threshold for L1 vs L2 (default: 1.0)
 */
auto smooth_l1_loss(const Variable& input, const Variable& target,
                    Reduction reduction = Reduction::Mean,
                    double beta = 1.0) -> Variable;

/**
 * @brief Functional cosine similarity
 *
 * @param x1 First input
 * @param x2 Second input
 * @param dim Dimension to compute similarity along
 * @param eps Small value to avoid division by zero
 */
auto cosine_similarity(const Variable& x1, const Variable& x2,
                       int64_t dim = 1, double eps = 1e-8) -> Variable;

} // namespace tenzor::nn::functional
