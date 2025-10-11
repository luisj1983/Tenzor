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
    auto forward(const Variable& input) -> Variable override;
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
    auto forward(const Variable& input) -> Variable override;

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
    auto forward(const Variable& input) -> Variable override;
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
    auto forward(const Variable& input) -> Variable override;
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
    GELU() = default;
    auto forward(const Variable& input) -> Variable override;
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
    auto forward(const Variable& input) -> Variable override;

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
    auto forward(const Variable& input) -> Variable override;

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
    auto forward(const Variable& input) -> Variable override;

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
    auto forward(const Variable& input) -> Variable override;
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
    auto forward(const Variable& input) -> Variable override;
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
    auto forward(const Variable& input) -> Variable override;
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

/** @brief Functional GELU: Gaussian Error Linear Unit */
auto gelu(const Variable& input) -> Variable;

/** @brief Functional softmax along specified dimension */
auto softmax(const Variable& input, int64_t dim = -1) -> Variable;

/** @brief Functional log-softmax (numerically stable) */
auto log_softmax(const Variable& input, int64_t dim = -1) -> Variable;

/** @brief Functional ELU with configurable alpha */
auto elu(const Variable& input, double alpha = 1.0) -> Variable;

/** @brief Functional SELU: Scaled ELU for self-normalizing networks */
auto selu(const Variable& input) -> Variable;

/** @brief Functional Swish/SiLU: x * sigmoid(x) */
auto swish(const Variable& input) -> Variable;

/** @brief Functional Mish: x * tanh(softplus(x)) */
auto mish(const Variable& input) -> Variable;

/** @} */ // end of functional_activations group

} // namespace nn
} // namespace tenzor
