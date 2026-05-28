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

#include <array>
#include <optional>
#include <tuple>
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

/** @brief Functional randomized ReLU */
inline auto rrelu(const Variable& input, double lower = 1.0 / 8.0, double upper = 1.0 / 3.0,
                  bool training = false) -> Variable {
    return nn::rrelu(input, lower, upper, training);
}

/** @brief Functional log-sigmoid: log(sigmoid(x)) */
inline auto log_sigmoid(const Variable& input) -> Variable {
    return nn::log_sigmoid(input);
}

/**
 * @brief Functional PReLU: ``max(0, x) + weight * min(0, x)``.
 *
 * Audit-4 U.13: the Python wrapper used to bind through a throwaway
 * ``nn::PReLU`` module and copy the user's weight into the layer's
 * internal Parameter, which severed autograd back to the caller's
 * ``weight`` Variable.  This free function mirrors the canonical
 * ``functional_conv1d`` pattern -- both inputs are honoured as
 * Variables and the result's grad_fn chain feeds back into them.
 *
 * @param input Input Variable, shape ``[..., C, ...]`` (PReLU broadcasts
 *              ``weight`` along all non-channel dims).
 * @param weight Per-channel slope Variable; numel must be 1 (shared
 *               slope) or equal to the input's channel dimension.
 */
auto prelu(const Variable& input, const Variable& weight) -> Variable;

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

/**
 * @brief Functional 1D convolution
 *
 * Applies a 1D convolution over an input signal [N, C_in, L].
 *
 * @param input Input tensor [N, C_in, L]
 * @param weight Convolution filters [C_out, C_in/groups, kL]
 * @param bias Optional bias [C_out]
 * @param stride Stride of the convolution
 * @param padding Zero-padding added to both sides
 * @param dilation Spacing between kernel elements
 * @param groups Number of blocked connections
 */
auto conv1d(const Variable& input, const Variable& weight,
            const std::optional<Variable>& bias = std::nullopt,
            int64_t stride = 1,
            int64_t padding = 0,
            int64_t dilation = 1,
            int64_t groups = 1) -> Variable;

/**
 * @brief Functional 3D convolution
 *
 * Applies a 3D convolution over an input signal [N, C_in, D, H, W].
 *
 * @param input Input tensor [N, C_in, D, H, W]
 * @param weight Convolution filters [C_out, C_in/groups, kD, kH, kW]
 * @param bias Optional bias [C_out]
 * @param stride Stride of the convolution (D, H, W)
 * @param padding Zero-padding added to both sides (D, H, W)
 * @param dilation Spacing between kernel elements (D, H, W)
 * @param groups Number of blocked connections
 */
auto conv3d(const Variable& input, const Variable& weight,
            const std::optional<Variable>& bias = std::nullopt,
            std::tuple<int64_t, int64_t, int64_t> stride = {1, 1, 1},
            std::tuple<int64_t, int64_t, int64_t> padding = {0, 0, 0},
            std::tuple<int64_t, int64_t, int64_t> dilation = {1, 1, 1},
            int64_t groups = 1) -> Variable;

/**
 * @brief Functional transposed 1D convolution
 *
 * @param input Input tensor [N, C_in, L]
 * @param weight Convolution filters [C_in, C_out/groups, kL]
 * @param bias Optional bias [C_out]
 * @param stride Stride of the convolution
 * @param padding Zero-padding added to both sides
 * @param output_padding Additional size added to output
 * @param groups Number of blocked connections
 * @param dilation Spacing between kernel elements
 */
auto conv_transpose1d(const Variable& input, const Variable& weight,
                      const std::optional<Variable>& bias = std::nullopt,
                      int64_t stride = 1,
                      int64_t padding = 0,
                      int64_t output_padding = 0,
                      int64_t groups = 1,
                      int64_t dilation = 1) -> Variable;

/**
 * @brief Functional transposed 2D convolution
 *
 * @param input Input tensor [N, C_in, H, W]
 * @param weight Convolution filters [C_in, C_out/groups, kH, kW]
 * @param bias Optional bias [C_out]
 * @param stride Stride of the convolution (H, W)
 * @param padding Zero-padding added to both sides (H, W)
 * @param output_padding Additional size added to output (H, W)
 * @param groups Number of blocked connections
 * @param dilation Spacing between kernel elements (H, W)
 */
auto conv_transpose2d(const Variable& input, const Variable& weight,
                      const std::optional<Variable>& bias = std::nullopt,
                      std::pair<int64_t, int64_t> stride = {1, 1},
                      std::pair<int64_t, int64_t> padding = {0, 0},
                      std::pair<int64_t, int64_t> output_padding = {0, 0},
                      int64_t groups = 1,
                      std::pair<int64_t, int64_t> dilation = {1, 1}) -> Variable;

/**
 * @brief Functional transposed 3D convolution
 *
 * @param input Input tensor [N, C_in, D, H, W]
 * @param weight Convolution filters [C_in, C_out/groups, kD, kH, kW]
 * @param bias Optional bias [C_out]
 * @param stride Stride of the convolution (D, H, W)
 * @param padding Zero-padding added to both sides (D, H, W)
 * @param output_padding Additional size added to output (D, H, W)
 * @param groups Number of blocked connections
 * @param dilation Spacing between kernel elements (D, H, W)
 */
auto conv_transpose3d(const Variable& input, const Variable& weight,
                      const std::optional<Variable>& bias = std::nullopt,
                      std::tuple<int64_t, int64_t, int64_t> stride = {1, 1, 1},
                      std::tuple<int64_t, int64_t, int64_t> padding = {0, 0, 0},
                      std::tuple<int64_t, int64_t, int64_t> output_padding = {0, 0, 0},
                      int64_t groups = 1,
                      std::tuple<int64_t, int64_t, int64_t> dilation = {1, 1, 1}) -> Variable;

// ============================================================================
// Attention (implemented in functional.cpp)
// ============================================================================

/**
 * @brief Options for scaled dot-product attention.
 */
struct SDPAOptions {
    std::optional<Variable> attn_mask;  ///< Optional attention mask
    double dropout_p{0.0};              ///< Dropout probability
    bool is_causal{false};              ///< Apply causal (lower-triangular) mask
};

/**
 * @brief Scaled dot-product attention.
 *
 * Computes: softmax(Q @ K^T / sqrt(d_k) + mask) @ V
 *
 * @param query  Query tensor [B, H, L, E]
 * @param key    Key tensor [B, H, S, E]
 * @param value  Value tensor [B, H, S, Ev]
 * @param opts   Attention options (mask, dropout, causal)
 * @return Output tensor [B, H, L, Ev]
 */
auto scaled_dot_product_attention(
    const Variable& query,
    const Variable& key,
    const Variable& value,
    const SDPAOptions& opts = {}) -> Variable;

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
                std::pair<int64_t, int64_t> padding = {0, 0},
                bool count_include_pad = true) -> Variable;

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

// ----------------------------------------------------------------------------
// 1D / 3D pooling free functions (audit-5 Z.21).
//
// Mirror the 2D signatures so Python wrappers can call directly into C++
// instead of constructing a fresh `nn.MaxPool1d` / `nn.AvgPool3d` / etc.
// module per invocation. Each function delegates to the matching layer
// module's forward_impl, which already wires the correct autograd backward
// Function (MaxPoolNdBackward / AvgPoolNdBackward).
// ----------------------------------------------------------------------------

/** @brief Functional 1D max pooling. Mirrors nn.MaxPool1d.forward_impl. */
auto max_pool1d(const Variable& input,
                int64_t kernel_size,
                int64_t stride = -1,
                int64_t padding = 0) -> Variable;

/** @brief Functional 1D average pooling. Mirrors nn.AvgPool1d.forward_impl. */
auto avg_pool1d(const Variable& input,
                int64_t kernel_size,
                int64_t stride = -1,
                int64_t padding = 0,
                bool count_include_pad = true) -> Variable;

/** @brief Functional 3D max pooling. Mirrors nn.MaxPool3d.forward_impl. */
auto max_pool3d(const Variable& input,
                std::array<int64_t, 3> kernel_size,
                std::array<int64_t, 3> stride = {-1, -1, -1},
                std::array<int64_t, 3> padding = {0, 0, 0}) -> Variable;

/** @brief Functional 3D average pooling. Mirrors nn.AvgPool3d.forward_impl. */
auto avg_pool3d(const Variable& input,
                std::array<int64_t, 3> kernel_size,
                std::array<int64_t, 3> stride = {-1, -1, -1},
                std::array<int64_t, 3> padding = {0, 0, 0},
                bool count_include_pad = true) -> Variable;

/** @brief Functional 1D adaptive max pooling. */
auto adaptive_max_pool1d(const Variable& input, int64_t output_size) -> Variable;

/** @brief Functional 1D adaptive average pooling. */
auto adaptive_avg_pool1d(const Variable& input, int64_t output_size) -> Variable;

/** @brief Functional 3D adaptive max pooling. */
auto adaptive_max_pool3d(const Variable& input,
                         std::array<int64_t, 3> output_size) -> Variable;

/** @brief Functional 3D adaptive average pooling. */
auto adaptive_avg_pool3d(const Variable& input,
                         std::array<int64_t, 3> output_size) -> Variable;

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
                           float label_smoothing = 0.0f,
                           int64_t ignore_index = -100) -> Variable {
    CrossEntropyLoss loss(reduction, label_smoothing, ignore_index);
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
 * @param ignore_index Class index to skip (PyTorch default: -100). Samples
 *        whose target equals @p ignore_index contribute zero loss and are
 *        excluded from the mean denominator.
 */
auto nll_loss(const Variable& input, const Tensor& target,
              Reduction reduction = Reduction::Mean,
              int64_t ignore_index = -100) -> Variable;

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

// ============================================================================
// Normalization utilities
// ============================================================================

/**
 * @brief L_p normalization along a dimension.
 *
 * Divides the input by its L_p norm along the specified dimension:
 *   output = input / max(norm(input, p, dim), eps)
 *
 * @param input Input tensor
 * @param p     Exponent value for the norm (default: 2.0)
 * @param dim   Dimension to reduce (default: 1)
 * @param eps   Small value to avoid division by zero (default: 1e-12)
 */
auto normalize(const Variable& input, double p = 2.0, int64_t dim = 1,
               double eps = 1e-12) -> Variable;

// ============================================================================
// Padding
// ============================================================================

/**
 * @brief Pad a tensor.
 *
 * @param input Input tensor
 * @param pad   Padding sizes in reverse-dimension order: (left, right) for 1D,
 *              (left, right, top, bottom) for 2D, etc.
 * @param mode  Padding mode: "constant", "reflect", "replicate"
 * @param value Fill value for constant padding (default: 0)
 */
auto pad(const Variable& input, const std::vector<int64_t>& pad,
         const std::string& mode = "constant", double value = 0.0) -> Variable;

// ============================================================================
// Phase 9: Lp Pooling (compositions — no new backend kernels)
// ============================================================================

/**
 * @brief 1D Lp-norm pooling
 *
 * Computes (sum(|x|^p, kernel) / kernel_size)^(1/p) via composition of
 * existing ops (abs, pow, avg_pool1d).
 *
 * @param input Input tensor [N, C, L]
 * @param norm_type Exponent p for the Lp norm
 * @param kernel_size Size of the pooling window
 * @param stride Stride of the pooling window (default: kernel_size)
 * @param ceil_mode Use ceil instead of floor to compute output size
 */
auto lp_pool1d(const Variable& input, double norm_type, int64_t kernel_size,
               int64_t stride = 0, bool ceil_mode = false) -> Variable;

/**
 * @brief 2D Lp-norm pooling
 *
 * Computes (sum(|x|^p, kernel) / kernel_size)^(1/p) via composition of
 * existing ops (abs, pow, avg_pool2d).
 *
 * @param input Input tensor [N, C, H, W]
 * @param norm_type Exponent p for the Lp norm
 * @param kernel_size Size of the pooling window (H, W)
 * @param stride Stride of the pooling window (default: kernel_size)
 * @param ceil_mode Use ceil instead of floor to compute output size
 */
auto lp_pool2d(const Variable& input, double norm_type,
               std::pair<int64_t, int64_t> kernel_size,
               std::pair<int64_t, int64_t> stride = {0, 0},
               bool ceil_mode = false) -> Variable;

// ============================================================================
// Phase 9: Local Response Normalization (composition)
// ============================================================================

/**
 * @brief Local response normalization (cross-channel)
 *
 * Normalizes across nearby channels:
 *   output = input / (k + alpha/size * sum(input^2, neighborhood))^beta
 *
 * @param input Input tensor [N, C, ...] (at least 3D)
 * @param size Number of neighboring channels to normalize across
 * @param alpha Scaling constant
 * @param beta Exponent
 * @param k Additive constant
 */
auto local_response_norm(const Variable& input, int64_t size,
                         double alpha = 1e-4, double beta = 0.75,
                         double k = 1.0) -> Variable;

// ============================================================================
// Phase 9: Fractional Max Pooling (dispatch to backend kernels)
// ============================================================================

/**
 * @brief 2D fractional max pooling
 *
 * Stochastic pooling where output size is specified and pool regions are
 * randomly determined. Returns (output, indices).
 *
 * @param input Input tensor [N, C, H, W]
 * @param kernel_size Pooling kernel size (H, W)
 * @param output_size Target output spatial dimensions (H, W)
 * @param random_samples Optional pre-generated random samples [N, C, 2]
 */
auto fractional_max_pool2d(const Variable& input,
                           std::pair<int64_t, int64_t> kernel_size,
                           std::pair<int64_t, int64_t> output_size,
                           const std::optional<Tensor>& random_samples = std::nullopt)
    -> std::pair<Variable, Tensor>;

/**
 * @brief 2D fractional max pooling with per-axis output_ratio (PyTorch parity).
 *
 * Equivalent to PyTorch's `nn.FractionalMaxPool2d(output_ratio=(rh, rw))`.
 * Output size is computed deterministically as
 * `out_h = floor(H * rh), out_w = floor(W * rw)` from the actual input
 * spatial extent at call time.
 */
auto fractional_max_pool2d(const Variable& input,
                           std::pair<int64_t, int64_t> kernel_size,
                           std::pair<double, double> output_ratio,
                           const std::optional<Tensor>& random_samples = std::nullopt)
    -> std::pair<Variable, Tensor>;

/**
 * @brief 3D fractional max pooling
 *
 * @param input Input tensor [N, C, D, H, W]
 * @param kernel_size Pooling kernel size (D, H, W)
 * @param output_size Target output spatial dimensions (D, H, W)
 * @param random_samples Optional pre-generated random samples [N, C, 3]
 */
auto fractional_max_pool3d(const Variable& input,
                           std::tuple<int64_t, int64_t, int64_t> kernel_size,
                           std::tuple<int64_t, int64_t, int64_t> output_size,
                           const std::optional<Tensor>& random_samples = std::nullopt)
    -> std::pair<Variable, Tensor>;

/**
 * @brief 3D fractional max pooling with per-axis output_ratio (PyTorch parity).
 */
auto fractional_max_pool3d(const Variable& input,
                           std::tuple<int64_t, int64_t, int64_t> kernel_size,
                           std::tuple<double, double, double> output_ratio,
                           const std::optional<Tensor>& random_samples = std::nullopt)
    -> std::pair<Variable, Tensor>;

// ============================================================================
// Phase 9: Max Unpooling (dispatch to backend kernels)
// ============================================================================

/**
 * @brief 2D max unpooling (inverse of max_pool2d)
 *
 * Places input values at positions indicated by indices from a prior
 * max_pool2d. Other positions are filled with zero.
 *
 * @param input Pooled tensor [N, C, H_pool, W_pool]
 * @param indices Max indices from max_pool2d [N, C, H_pool, W_pool]
 * @param kernel_size Original pooling kernel size (H, W)
 * @param stride Original pooling stride (default: kernel_size)
 * @param padding Original pooling padding
 * @param output_size Optional explicit output size [H, W]
 */
auto max_unpool2d(const Variable& input, const Tensor& indices,
                  std::pair<int64_t, int64_t> kernel_size,
                  std::pair<int64_t, int64_t> stride = {-1, -1},
                  std::pair<int64_t, int64_t> padding = {0, 0},
                  std::optional<std::pair<int64_t, int64_t>> output_size = std::nullopt)
    -> Variable;

/**
 * @brief Functional max-unpool 1d (Phase A.1).
 *
 * Inverse of MaxPool1d: scatter input values into a larger tensor at the
 * indices saved by the matching MaxPool1d forward.
 *
 * @param input    [N, C, L_pool]   — the pooled values to scatter.
 * @param indices  [N, C, L_pool]   — Int64 spatial indices (linear within L).
 * @param kernel_size  size of the original pooling window.
 * @param stride       pooling stride (default = kernel_size).
 * @param padding      pooling padding (default = 0).
 * @param output_size  optional explicit output L (computes via formula otherwise).
 */
auto max_unpool1d(const Variable& input, const Tensor& indices,
                  int64_t kernel_size,
                  int64_t stride = -1,
                  int64_t padding = 0,
                  std::optional<int64_t> output_size = std::nullopt)
    -> Variable;

/**
 * @brief 3D max unpooling (inverse of max_pool3d)
 *
 * @param input Pooled tensor [N, C, D_pool, H_pool, W_pool]
 * @param indices Max indices from max_pool3d [N, C, D_pool, H_pool, W_pool]
 * @param kernel_size Original pooling kernel size (D, H, W)
 * @param stride Original pooling stride (default: kernel_size)
 * @param padding Original pooling padding
 * @param output_size Optional explicit output size [D, H, W]
 */
auto max_unpool3d(const Variable& input, const Tensor& indices,
                  std::tuple<int64_t, int64_t, int64_t> kernel_size,
                  std::tuple<int64_t, int64_t, int64_t> stride = {-1, -1, -1},
                  std::tuple<int64_t, int64_t, int64_t> padding = {0, 0, 0},
                  std::optional<std::tuple<int64_t, int64_t, int64_t>> output_size = std::nullopt)
    -> Variable;

// ============================================================================
// Multi-Head Attention (implemented in functional.cpp)
// ============================================================================

/**
 * @brief Functional multi-head attention forward pass.
 *
 * Composes existing operations to implement the full multi-head attention
 * computation: input projection, split heads, scaled dot-product attention,
 * merge heads, and output projection.
 *
 * @param query  Query tensor [batch, seq_q, embed_dim]
 * @param key    Key tensor [batch, seq_k, embed_dim]  (or kdim if using separate projections)
 * @param value  Value tensor [batch, seq_k, embed_dim] (or vdim if using separate projections)
 * @param num_heads Number of attention heads
 * @param in_proj_weight Combined QKV projection weight [3 * embed_dim, embed_dim]
 * @param in_proj_bias Combined QKV projection bias [3 * embed_dim]
 * @param out_proj_weight Output projection weight [embed_dim, embed_dim]
 * @param out_proj_bias Output projection bias [embed_dim]
 * @param attn_mask Optional attention mask [seq_q, seq_k] or [batch, num_heads, seq_q, seq_k]
 * @param dropout_p Dropout probability applied to attention weights
 * @param training If true, apply dropout
 * @param need_weights If true, return attention weights; otherwise second element is empty
 * @return (output [batch, seq_q, embed_dim], attn_weights [batch, num_heads, seq_q, seq_k] or empty)
 */
auto multi_head_attention_forward(
    const Tensor& query, const Tensor& key, const Tensor& value,
    int64_t num_heads,
    const Tensor& in_proj_weight, const Tensor& in_proj_bias,
    const Tensor& out_proj_weight, const Tensor& out_proj_bias,
    std::optional<Tensor> attn_mask = std::nullopt,
    double dropout_p = 0.0,
    bool training = false,
    bool need_weights = false
) -> std::pair<Tensor, Tensor>;

} // namespace tenzor::nn::functional
