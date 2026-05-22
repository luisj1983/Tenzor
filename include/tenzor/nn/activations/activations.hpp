/**
 * @file activations.hpp
 * @brief Neural network activation functions and layers
 *
 * This file provides a comprehensive set of activation functions commonly used
 * in deep learning, including both module-based (stateful) and functional (stateless)
 * implementations. Activation functions introduce non-linearity into neural networks,
 * enabling them to learn complex patterns.
 */

#pragma once

#include "../module.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Rectified Linear Unit (ReLU) activation
 *
 * Applies the rectified linear unit function element-wise:
 *
 * \f[
 * \text{ReLU}(x) = \max(0, x) = \begin{cases}
 *   x & \text{if } x > 0 \\
 *   0 & \text{otherwise}
 * \end{cases}
 * \f]
 *
 * ReLU is the most widely used activation function in deep learning due to:
 * - Computational efficiency: O(n) complexity with simple max operation
 * - Gradient flow: Eliminates vanishing gradient problem for positive values
 * - Sparse activation: Typically ~50% of neurons are zero, promoting sparse representations
 * - Empirically superior convergence compared to sigmoid/tanh
 *
 * **Use Cases:**
 * - Default choice for hidden layers in most architectures
 * - Convolutional neural networks (CNNs)
 * - Deep feedforward networks
 * - Not recommended for recurrent networks (use Tanh instead)
 *
 * **Limitations:**
 * - Dying ReLU problem: Neurons can get stuck outputting zero
 * - Not zero-centered: Can cause gradient issues
 * - Unbounded output: Can lead to exploding activations
 *
 * @par Complexity
 * - Time: O(n) where n is the number of elements
 * - Space: O(1) in-place operation
 *
 * @par Thread Safety
 * Thread-safe for forward pass on different inputs
 *
 * @code
 * auto relu = ReLU();
 * auto input = Variable::create(tensor({-1.0, 0.0, 1.0, 2.0}));
 * auto output = relu.forward(input);  // Result: [0, 0, 1, 2]
 * @endcode
 *
 * @see LeakyReLU for variant that doesn't zero negative values
 * @see GELU for smooth approximation with better gradient properties
 */
class ReLU : public Module {
public:
    ReLU() = default;
    auto forward_impl(const Variable& input) -> Variable override;
};

/**
 * @brief ReLU6 activation function
 *
 * Applies ReLU with maximum value of 6:
 * ReLU6(x) = min(max(0, x), 6)
 *
 * Used in MobileNets for better numerical stability on mobile devices.
 */
class ReLU6 : public Module {
public:
    ReLU6() = default;
    auto forward_impl(const Variable& input) -> Variable override;
};

/**
 * @brief Leaky Rectified Linear Unit (Leaky ReLU) activation
 *
 * Applies the leaky ReLU function element-wise:
 *
 * \f[
 * \text{LeakyReLU}(x) = \begin{cases}
 *   x & \text{if } x > 0 \\
 *   \alpha x & \text{otherwise}
 * \end{cases}
 * \f]
 *
 * where \f$\alpha\f$ is the negative_slope parameter (default: 0.01).
 *
 * LeakyReLU addresses the "dying ReLU" problem by allowing small negative values,
 * ensuring neurons can recover if they get stuck in negative regions.
 *
 * **Advantages over ReLU:**
 * - Prevents dying neurons by maintaining gradient for negative inputs
 * - Better gradient flow throughout the network
 * - Often improves convergence in very deep networks
 *
 * **Typical Values:**
 * - negative_slope = 0.01 (default, widely used)
 * - negative_slope = 0.2 (more aggressive, used in some GANs)
 * - negative_slope = 0.3 (Parametric ReLU default)
 *
 * @param negative_slope Controls the slope for negative inputs (default: 0.01)
 *
 * @par Complexity
 * - Time: O(n)
 * - Space: O(1)
 *
 * @code
 * auto leaky_relu = LeakyReLU(0.01);
 * auto input = Variable::create(tensor({-1.0, 0.0, 1.0}));
 * auto output = leaky_relu.forward(input);  // Result: [-0.01, 0, 1]
 * @endcode
 *
 * @see ReLU for standard variant
 * @see ELU for exponential variant
 */
class LeakyReLU : public Module {
public:
    explicit LeakyReLU(double negative_slope = 0.01);
    auto forward_impl(const Variable& input) -> Variable override;
    // H2 fix: accessor for serializers (e.g. Lite exporter).
    auto negative_slope() const -> double { return negative_slope_; }

private:
    double negative_slope_;
};

/**
 * @brief Sigmoid activation function
 *
 * Applies the sigmoid (logistic) function element-wise:
 *
 * \f[
 * \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + e^{-x}}
 * \f]
 *
 * Sigmoid squashes input values to range (0, 1), making it suitable for:
 * - Binary classification (output layer)
 * - Probability estimation
 * - Gate mechanisms in LSTM/GRU
 *
 * **Limitations:**
 * - Vanishing gradients: Gradients approach zero for large |x|
 * - Not zero-centered: Can slow down convergence
 * - Expensive computation: Requires exponential operation
 *
 * **Note:** For hidden layers, prefer ReLU or its variants for better gradient flow.
 *
 * @par Complexity
 * - Time: O(n)
 * - Space: O(1)
 *
 * @code
 * auto sigmoid = Sigmoid();
 * auto output = sigmoid.forward(input);  // Range: (0, 1)
 * @endcode
 *
 * @see Tanh for zero-centered alternative
 */
class Sigmoid : public Module {
public:
    Sigmoid() = default;
    auto forward_impl(const Variable& input) -> Variable override;
};

/**
 * @brief Hyperbolic tangent (Tanh) activation
 *
 * Applies the hyperbolic tangent function element-wise:
 *
 * \f[
 * \text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
 * \f]
 *
 * Tanh squashes input to range (-1, 1). Advantages over sigmoid:
 * - Zero-centered: Better gradient flow and faster convergence
 * - Stronger gradients: Steeper slope around zero
 *
 * **Use Cases:**
 * - Recurrent neural networks (RNNs, LSTMs)
 * - Hidden layers when zero-centered outputs are needed
 * - Between layers that should output both positive and negative values
 *
 * @par Complexity
 * - Time: O(n)
 * - Space: O(1)
 *
 * @code
 * auto tanh = Tanh();
 * auto output = tanh.forward(input);  // Range: (-1, 1)
 * @endcode
 */
class Tanh : public Module {
public:
    Tanh() = default;
    auto forward_impl(const Variable& input) -> Variable override;
};

/**
 * @brief Gaussian Error Linear Unit (GELU) activation
 *
 * Applies the GELU function element-wise:
 *
 * \f[
 * \text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
 * \f]
 *
 * where \f$\Phi(x)\f$ is the cumulative distribution function of the standard Gaussian.
 *
 * GELU provides smooth, probabilistic gating of inputs. Key advantages:
 * - Smooth approximation of ReLU with better gradient properties
 * - No "dying neuron" problem
 * - Used in BERT, GPT-2, GPT-3, and many Transformer models
 *
 * **When to Use:**
 * - Transformer architectures
 * - Natural language processing models
 * - When you need smooth, differentiable activation
 *
 * @par Complexity
 * - Time: O(n)
 * - Space: O(1)
 *
 * @code
 * auto gelu = GELU();
 * auto output = gelu.forward(input);  // Smooth non-linearity
 * @endcode
 *
 * @see Swish for similar smooth activation
 */
class GELU : public Module {
public:
    /**
     * @brief Construct GELU activation.
     *
     * @param approximate Approximation method: "none" for exact erf, "tanh" for fast tanh approximation
     */
    explicit GELU(const std::string& approximate = "none");
    auto forward_impl(const Variable& input) -> Variable override;

private:
    std::string approximate_;  ///< "none" or "tanh"
};

/**
 * @brief Softmax activation function
 *
 * Applies the softmax function along specified dimension:
 *
 * \f[
 * \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
 * \f]
 *
 * Softmax converts logits to probability distribution (values sum to 1).
 *
 * **Use Cases:**
 * - Multi-class classification (output layer)
 * - Attention mechanisms in Transformers
 * - Any task requiring probability distributions
 *
 * **Implementation Note:**
 * Uses numerically stable computation by subtracting max(x) before exponential.
 *
 * @param dim Dimension along which to apply softmax (default: -1, last dimension)
 *
 * @par Complexity
 * - Time: O(n)
 * - Space: O(1)
 *
 * @code
 * auto softmax = Softmax(-1);  // Apply along last dimension
 * auto probs = softmax.forward(logits);  // Sum to 1.0
 * @endcode
 *
 * @see LogSoftmax for numerically stable log-space version
 */
class Softmax : public Module {
public:
    explicit Softmax(int64_t dim = -1);
    auto forward_impl(const Variable& input) -> Variable override;

    /// @brief Reduction dim (exposed for serializers).
    [[nodiscard]] auto dim() const -> int64_t { return dim_; }

private:
    int64_t dim_;
};

/**
 * @brief Log-Softmax activation function
 *
 * Applies log(softmax(x)) in numerically stable way:
 *
 * \f[
 * \text{LogSoftmax}(x_i) = \log\left(\frac{e^{x_i}}{\sum_j e^{x_j}}\right) = x_i - \log\sum_j e^{x_j}
 * \f]
 *
 * Preferred over Softmax + Log for numerical stability and efficiency.
 * Commonly used with NLLLoss for classification.
 *
 * @param dim Dimension along which to apply log-softmax (default: -1)
 *
 * @par Complexity
 * - Time: O(n)
 * - Space: O(1)
 *
 * @code
 * auto log_softmax = LogSoftmax(-1);
 * auto log_probs = log_softmax.forward(logits);
 * auto loss = nll_loss(log_probs, targets);  // Efficient combination
 * @endcode
 *
 * @see Softmax, NLLLoss
 */
class LogSoftmax : public Module {
public:
    explicit LogSoftmax(int64_t dim = -1);
    auto forward_impl(const Variable& input) -> Variable override;

    /// @brief Reduction dim (exposed for serializers).
    [[nodiscard]] auto dim() const -> int64_t { return dim_; }

private:
    int64_t dim_;
};

/**
 * @brief Exponential Linear Unit (ELU) activation
 *
 * Applies the ELU function element-wise:
 *
 * \f[
 * \text{ELU}(x) = \begin{cases}
 *   x & \text{if } x > 0 \\
 *   \alpha(e^x - 1) & \text{otherwise}
 * \end{cases}
 * \f]
 *
 * ELU has negative values, pushing mean activations closer to zero.
 * Advantages:
 * - Reduces bias shift effect
 * - Smooth, differentiable everywhere
 * - Can produce negative outputs (unlike ReLU)
 *
 * @param alpha Scale for negative values (default: 1.0)
 *
 * @par Complexity
 * - Time: O(n)
 * - Space: O(1)
 *
 * @code
 * auto elu = ELU(1.0);
 * auto output = elu.forward(input);
 * @endcode
 *
 * @see SELU for self-normalizing variant
 */
class ELU : public Module {
public:
    explicit ELU(double alpha = 1.0);
    auto forward_impl(const Variable& input) -> Variable override;
    // H2 fix: accessor for serializers (e.g. Lite exporter).
    auto alpha() const -> double { return alpha_; }

private:
    double alpha_;
};

/**
 * @brief Continuously differentiable Exponential Linear Unit (CELU).
 *
 * Applies the CELU function element-wise:
 *
 * \f[
 * \text{CELU}(x, \alpha) = \max(0, x) + \min(0, \alpha(e^{x/\alpha} - 1))
 * \f]
 *
 * Unlike ELU, CELU's derivative is continuous at x = 0 for any α. It satisfies
 * the identity CELU(x, α) = α · ELU(x/α, 1), which this implementation uses to
 * reuse the existing ELU autograd path.
 *
 * @see ELU
 */
class CELU : public Module {
public:
    explicit CELU(double alpha = 1.0);
    auto forward_impl(const Variable& input) -> Variable override;

private:
    double alpha_;
};

/**
 * @brief Scaled Exponential Linear Unit (SELU) activation
 *
 * Applies the SELU function element-wise:
 *
 * \f[
 * \text{SELU}(x) = \lambda \begin{cases}
 *   x & \text{if } x > 0 \\
 *   \alpha(e^x - 1) & \text{otherwise}
 * \end{cases}
 * \f]
 *
 * where \f$\lambda \approx 1.0507\f$ and \f$\alpha \approx 1.6733\f$.
 *
 * SELU enables self-normalizing neural networks:
 * - Activations automatically normalize to zero mean and unit variance
 * - Allows training very deep networks without batch normalization
 * - Requires proper weight initialization (LeCun normal)
 *
 * **Requirements:**
 * - Use with dropout variant "Alpha Dropout"
 * - Initialize weights with LeCun normal initialization
 * - Stack many layers for self-normalizing effect
 *
 * @par Complexity
 * - Time: O(n)
 * - Space: O(1)
 *
 * @code
 * auto selu = SELU();
 * auto output = selu.forward(input);  // Self-normalizing
 * @endcode
 */
class SELU : public Module {
public:
    SELU() = default;
    auto forward_impl(const Variable& input) -> Variable override;
};

/**
 * @brief Swish (SiLU) activation function
 *
 * Applies the Swish/SiLU function element-wise:
 *
 * \f[
 * \text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
 * \f]
 *
 * Swish (also called SiLU in PyTorch) is a smooth, non-monotonic activation.
 * Properties:
 * - Smooth and differentiable everywhere
 * - Self-gated: Output depends on input magnitude
 * - Performs better than ReLU in deep networks (40+ layers)
 *
 * **Use Cases:**
 * - Very deep networks
 * - Mobile architectures (EfficientNet, MobileNetV3)
 * - When smooth activation is preferred
 *
 * @par Complexity
 * - Time: O(n)
 * - Space: O(1)
 *
 * @code
 * auto swish = Swish();
 * auto output = swish.forward(input);
 * @endcode
 *
 * @see GELU for alternative smooth activation
 * @see Mish for similar but more complex variant
 */
class Swish : public Module {
public:
    Swish() = default;
    auto forward_impl(const Variable& input) -> Variable override;
};

/**
 * @brief Mish activation function
 *
 * Applies the Mish function element-wise:
 *
 * \f[
 * \text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))
 * \f]
 *
 * Mish is a smooth, non-monotonic, self-gated activation function.
 * Advantages:
 * - Smooth and continuously differentiable
 * - Non-monotonic: Allows better expressiveness
 * - Self-regularizing: Prevents gradient vanishing
 * - Empirically outperforms Swish and ReLU in some tasks
 *
 * **Note:** Slightly more expensive to compute than Swish.
 *
 * @par Complexity
 * - Time: O(n)
 * - Space: O(1)
 *
 * @code
 * auto mish = Mish();
 * auto output = mish.forward(input);
 * @endcode
 *
 * @see Swish for similar activation
 */
class Mish : public Module {
public:
    Mish() = default;
    auto forward_impl(const Variable& input) -> Variable override;
};

/**
 * @brief PReLU: Parametric ReLU with learnable negative slope.
 *
 * Applies the function:
 * \f[ \text{PReLU}(x) = \max(0, x) + a \cdot \min(0, x) \f]
 *
 * where \f$a\f$ is a learnable parameter. Can be a single value (shared
 * across all channels) or per-channel.
 *
 * @par Example
 * @code
 * auto prelu = PReLU(64);  // Per-channel for 64 channels
 * auto output = prelu.forward(input);
 * @endcode
 */
class PReLU : public Module {
public:
    explicit PReLU(int64_t num_parameters = 1, double init = 0.25);
    auto forward_impl(const Variable& input) -> Variable override;

private:
    int64_t num_parameters_;
};

/**
 * @brief Hardswish activation function.
 *
 * Applies the function:
 * \f[ \text{Hardswish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6} \f]
 *
 * A computationally efficient approximation to Swish, commonly used in
 * MobileNetV3 and other mobile architectures.
 */
class Hardswish : public Module {
public:
    Hardswish() = default;
    auto forward_impl(const Variable& input) -> Variable override;
};

/**
 * @brief Hardsigmoid activation function.
 *
 * Applies the function:
 * \f[ \text{Hardsigmoid}(x) = \frac{\text{ReLU6}(x + 3)}{6} \f]
 *
 * A computationally efficient approximation to sigmoid.
 */
class Hardsigmoid : public Module {
public:
    Hardsigmoid() = default;
    auto forward_impl(const Variable& input) -> Variable override;
};

/**
 * @brief Gated Linear Unit (GLU): splits input in half and applies gating
 *
 * GLU(a, b) = a * sigmoid(b) where input is split into a and b along dim.
 * Commonly used in language models and as an alternative to ReLU.
 */
class GLU : public Module {
public:
    explicit GLU(int64_t dim = -1) : dim_(dim) {}
    auto forward_impl(const Variable& input) -> Variable override;
private:
    int64_t dim_;
};

// ----------------------------------------------------------------------------
// Additional activation modules — thin wrappers over the existing primitives
// for the PyTorch-compat surface.
// ----------------------------------------------------------------------------

/**
 * @brief Softmin — softmax(-x, dim). Opposite polarity of softmax.
 */
class Softmin : public Module {
public:
    explicit Softmin(int64_t dim = -1) : dim_(dim) {}
    auto forward_impl(const Variable& input) -> Variable override;
private:
    int64_t dim_;
};

/**
 * @brief Tanhshrink — x - tanh(x). Saturates to x-1 / x+1 at ±∞.
 */
class Tanhshrink : public Module {
public:
    Tanhshrink() = default;
    auto forward_impl(const Variable& input) -> Variable override;
};

/**
 * @brief Softshrink — sign(x) * max(|x| - lambda, 0). Soft thresholding.
 */
class Softshrink : public Module {
public:
    explicit Softshrink(double lambda = 0.5) : lambda_(lambda) {}
    auto forward_impl(const Variable& input) -> Variable override;
private:
    double lambda_;
};

/**
 * @brief Hardshrink — x if |x| > lambda else 0. Hard thresholding.
 */
class Hardshrink : public Module {
public:
    explicit Hardshrink(double lambda = 0.5) : lambda_(lambda) {}
    auto forward_impl(const Variable& input) -> Variable override;
private:
    double lambda_;
};

/**
 * @brief Softsign — x / (1 + |x|). Smooth saturation without exp.
 */
class Softsign : public Module {
public:
    Softsign() = default;
    auto forward_impl(const Variable& input) -> Variable override;
};

/**
 * @brief Threshold — x if x > threshold else value.
 */
class Threshold : public Module {
public:
    Threshold(double threshold, double value) : threshold_(threshold), value_(value) {}
    auto forward_impl(const Variable& input) -> Variable override;
private:
    double threshold_;
    double value_;
};

/**
 * @brief Hardtanh — clamp(x, min_val, max_val). Linear in range, flat outside.
 */
class Hardtanh : public Module {
public:
    explicit Hardtanh(double min_val = -1.0, double max_val = 1.0)
        : min_val_(min_val), max_val_(max_val) {}
    auto forward_impl(const Variable& input) -> Variable override;
private:
    double min_val_;
    double max_val_;
};

/**
 * @defgroup functional_activations Functional Activation Functions
 * @brief Stateless activation functions for flexible use
 *
 * These functional versions don't require creating module instances.
 * Useful for:
 * - Quick prototyping
 * - Inline activation in custom modules
 * - When you don't need to save activation state
 *
 * @{
 */

/** @brief Functional ReLU: max(0, x) */
auto relu(const Variable& input) -> Variable;

/** @brief Functional Leaky ReLU with configurable slope */
auto leaky_relu(const Variable& input, double negative_slope = 0.01) -> Variable;

/** @brief Functional sigmoid: 1 / (1 + exp(-x)) */
auto sigmoid(const Variable& input) -> Variable;

/** @brief Functional tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x)) */
auto tanh(const Variable& input) -> Variable;

/** @brief Functional GELU: Gaussian Error Linear Unit
 *  @param approximate "none" for exact, "tanh" for fast approximation
 */
auto gelu(const Variable& input, const std::string& approximate = "none") -> Variable;

/** @brief Functional softmax along specified dimension */
auto softmax(const Variable& input, int64_t dim = -1) -> Variable;

/** @brief Functional log-softmax (numerically stable) */
auto log_softmax(const Variable& input, int64_t dim = -1) -> Variable;

/** @brief Functional ELU with configurable alpha */
auto elu(const Variable& input, double alpha = 1.0) -> Variable;

/** @brief Functional CELU: Continuously differentiable ELU. Alpha controls the saturation. */
auto celu(const Variable& input, double alpha = 1.0) -> Variable;

/** @brief Functional SELU: Scaled ELU for self-normalizing networks */
auto selu(const Variable& input) -> Variable;

/** @brief Functional Swish/SiLU: x * sigmoid(x) */
auto swish(const Variable& input) -> Variable;

/** @brief Functional Mish: x * tanh(softplus(x)) */
auto mish(const Variable& input) -> Variable;

/** @brief Functional Hardswish: x * ReLU6(x + 3) / 6 */
auto hardswish(const Variable& input) -> Variable;

/** @brief Functional Hardsigmoid: ReLU6(x + 3) / 6 */
auto hardsigmoid(const Variable& input) -> Variable;

/** @brief Functional GLU: a * sigmoid(b) where input is split along dim */
auto glu(const Variable& input, int64_t dim = -1) -> Variable;

/** @brief Functional softmin: softmax(-x, dim) */
auto softmin(const Variable& input, int64_t dim = -1) -> Variable;

/** @brief Functional tanhshrink: x - tanh(x) */
auto tanhshrink(const Variable& input) -> Variable;

/** @brief Functional softshrink: sign(x) * max(|x| - lambda, 0) */
auto softshrink(const Variable& input, double lambda = 0.5) -> Variable;

/** @brief Functional hardshrink: x if |x| > lambda else 0 */
auto hardshrink(const Variable& input, double lambda = 0.5) -> Variable;

/** @brief Functional softsign: x / (1 + |x|) */
auto softsign(const Variable& input) -> Variable;

/** @brief Functional threshold: x if x > threshold else value */
auto threshold(const Variable& input, double threshold, double value) -> Variable;

/** @brief Functional hardtanh: clamp(x, min_val, max_val) */
auto hardtanh(const Variable& input, double min_val = -1.0, double max_val = 1.0) -> Variable;

/** @brief Functional randomized ReLU
 *  @param lower Lower bound of uniform distribution for negative slope (default: 1/8)
 *  @param upper Upper bound of uniform distribution for negative slope (default: 1/3)
 *  @param training If true, sample slope uniformly; if false, use midpoint
 */
auto rrelu(const Variable& input, double lower = 1.0 / 8.0, double upper = 1.0 / 3.0,
           bool training = false) -> Variable;

/** @brief Functional log-sigmoid: log(sigmoid(x)) = -softplus(-x)
 *
 *  Numerically stable implementation that avoids computing sigmoid explicitly.
 */
auto log_sigmoid(const Variable& input) -> Variable;

/** @} */ // end of functional_activations group

/**
 * @defgroup inplace_activations In-Place Activation Functions
 * @brief Memory-efficient in-place activation functions
 *
 * These in-place versions modify tensors without allocating new memory,
 * which is more efficient for large tensors and reduces memory fragmentation.
 * Use when the original tensor is no longer needed.
 *
 * @{
 */

/**
 * @brief In-place ReLU: modifies input to max(0, x).
 * @param input Tensor to modify in-place
 * @return Reference to modified input tensor
 * @note More memory-efficient than relu() - no new allocation
 * @warning Original values are lost - use only when input is no longer needed
 *
 * @code
 * Tensor x({1000, 1000}, DType::Float32, Device::cpu());
 * relu_(x);  // Modifies x in-place, saves ~4MB allocation
 * @endcode
 */
auto relu_(Tensor& input) -> Tensor&;

/**
 * @brief In-place sigmoid: modifies input to 1 / (1 + exp(-x)).
 * @param input Tensor to modify in-place
 * @return Reference to modified input tensor
 * @note More memory-efficient than sigmoid() - no new allocation
 * @warning Original values are lost - use only when input is no longer needed
 */
auto sigmoid_(Tensor& input) -> Tensor&;

/**
 * @brief In-place tanh: modifies input to tanh(x).
 * @param input Tensor to modify in-place
 * @return Reference to modified input tensor
 * @note More memory-efficient than tanh() - no new allocation
 * @warning Original values are lost - use only when input is no longer needed
 */
auto tanh_(Tensor& input) -> Tensor&;

/**
 * @brief In-place leaky ReLU with configurable slope.
 * @param input Tensor to modify in-place
 * @param negative_slope Slope for negative values (default: 0.01)
 * @return Reference to modified input tensor
 */
auto leaky_relu_(Tensor& input, double negative_slope = 0.01) -> Tensor&;

/**
 * @brief In-place GELU activation.
 * @param input Tensor to modify in-place
 * @return Reference to modified input tensor
 */
auto gelu_(Tensor& input) -> Tensor&;

/** @} */ // end of inplace_activations group

} // namespace nn
} // namespace tenzor
