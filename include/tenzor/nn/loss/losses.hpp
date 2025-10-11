/**
 * @file losses.hpp
 * @brief Loss functions for neural network training
 *
 * This file provides common loss (cost/objective) functions used in supervised learning.
 * Loss functions measure the difference between predictions and targets, providing
 * the signal for backpropagation and optimization.
 */

#pragma once

#include "../module.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Reduction mode for loss functions
 *
 * Specifies how to aggregate individual element losses into a single scalar value.
 *
 * - **None**: Return loss for each element (no reduction)
 * - **Mean**: Average loss across all elements (default for most losses)
 * - **Sum**: Sum all individual losses
 *
 * **Choosing Reduction:**
 * - Mean: Use when batch size varies (normalizes by count)
 * - Sum: Use when you want gradient scale independent of batch size
 * - None: Use for custom per-element processing
 */
enum class Reduction {
    None,   ///< No reduction, return per-element loss
    Mean,   ///< Average reduction (divide by number of elements)
    Sum     ///< Sum reduction (sum all losses)
};

/**
 * @brief Mean Squared Error (MSE) Loss
 *
 * Computes the mean squared error between input and target:
 *
 * \f[
 * \text{MSE}(x, y) = \frac{1}{n}\sum_{i=1}^{n}(x_i - y_i)^2
 * \f]
 *
 * Also known as L2 loss. Commonly used for:
 * - Regression tasks
 * - Autoencoders
 * - Predicting continuous values
 *
 * **Properties:**
 * - Sensitive to outliers (squared error amplifies large differences)
 * - Smooth and differentiable everywhere
 * - Convex optimization landscape
 * - Gradient magnitude increases with error size
 *
 * **When to Use:**
 * - Target values are continuous and unbounded
 * - Outliers should be heavily penalized
 * - Need smooth, well-behaved gradients
 *
 * **Alternatives:**
 * - Use L1Loss for robustness to outliers
 * - Use SmoothL1Loss (Huber) for balanced approach
 *
 * @param reduction How to reduce the loss (default: Mean)
 *
 * @par Complexity
 * - Time: O(n) where n is number of elements
 * - Space: O(1) or O(n) depending on reduction mode
 *
 * @code
 * auto criterion = MSELoss(Reduction::Mean);
 * auto predictions = model.forward(inputs);
 * auto loss = criterion(predictions, targets);
 * loss.backward();  // Compute gradients
 * @endcode
 *
 * @see L1Loss, SmoothL1Loss
 */
class MSELoss {
public:
    explicit MSELoss(Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input, const Variable& target) -> Variable;
    auto operator()(const Variable& input, const Variable& target) -> Variable {
        return forward(input, target);
    }

private:
    Reduction reduction_;
};

/**
 * @brief Cross Entropy Loss
 *
 * Computes the cross entropy loss between input logits and target class indices:
 *
 * \f[
 * \text{CrossEntropy}(x, y) = -\log\left(\frac{e^{x_y}}{\sum_j e^{x_j}}\right) = -x_y + \log\sum_j e^{x_j}
 * \f]
 *
 * Combines LogSoftmax and NLLLoss in a single, numerically stable operation.
 * This is the standard loss for multi-class classification.
 *
 * **Use Cases:**
 * - Multi-class classification (mutually exclusive classes)
 * - Image classification
 * - Text classification
 * - Any task with discrete class labels
 *
 * **Input Requirements:**
 * - input: Raw logits (unnormalized scores), shape (N, C) or (N, C, ...)
 * - target: Class indices (integers in [0, C-1]), shape (N) or (N, ...)
 *
 * **Note:** Do NOT apply softmax to inputs; this loss expects raw logits.
 *
 * @param reduction How to reduce the loss (default: Mean)
 *
 * @par Complexity
 * - Time: O(N * C) where N is batch size, C is number of classes
 * - Space: O(1) or O(N) depending on reduction mode
 *
 * @code
 * auto criterion = CrossEntropyLoss();
 * auto logits = model.forward(images);  // Shape: (batch, num_classes)
 * auto targets = tensor({0, 2, 1});     // Class indices
 * auto loss = criterion(logits, targets);
 * @endcode
 *
 * @see NLLLoss, LogSoftmax, BCELoss
 */
class CrossEntropyLoss {
public:
    explicit CrossEntropyLoss(Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input, const Tensor& target) -> Variable;
    auto operator()(const Variable& input, const Tensor& target) -> Variable {
        return forward(input, target);
    }

private:
    Reduction reduction_;
};

/**
 * @brief Binary Cross Entropy (BCE) Loss
 *
 * Computes the binary cross entropy between input probabilities and binary targets:
 *
 * \f[
 * \text{BCE}(x, y) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log(x_i) + (1-y_i)\log(1-x_i)\right]
 * \f]
 *
 * Used for binary classification where inputs are probabilities in (0, 1).
 *
 * **Use Cases:**
 * - Binary classification
 * - Multi-label classification (independent binary decisions)
 * - Pixel-wise segmentation
 *
 * **Input Requirements:**
 * - input: Probabilities in (0, 1), typically after sigmoid activation
 * - target: Binary labels (0 or 1), same shape as input
 *
 * **Important:** Apply sigmoid to logits before passing to this loss.
 * For numerical stability with raw logits, use BCEWithLogitsLoss instead.
 *
 * @param reduction How to reduce the loss (default: Mean)
 *
 * @par Complexity
 * - Time: O(n)
 * - Space: O(1) or O(n) depending on reduction mode
 *
 * @code
 * auto criterion = BCELoss();
 * auto logits = model.forward(input);
 * auto probs = sigmoid(logits);  // Convert to probabilities
 * auto loss = criterion(probs, binary_targets);
 * @endcode
 *
 * @see BCEWithLogitsLoss for more stable version
 */
class BCELoss {
public:
    explicit BCELoss(Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input, const Variable& target) -> Variable;
    auto operator()(const Variable& input, const Variable& target) -> Variable {
        return forward(input, target);
    }

private:
    Reduction reduction_;
};

/**
 * @brief Binary Cross Entropy with Logits Loss
 *
 * Combines sigmoid activation and BCE loss in a single, numerically stable operation:
 *
 * \f[
 * \text{BCEWithLogits}(x, y) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log(\sigma(x_i)) + (1-y_i)\log(1-\sigma(x_i))\right]
 * \f]
 *
 * where \f$\sigma(x) = \frac{1}{1+e^{-x}}\f$ is the sigmoid function.
 *
 * **Advantages over BCELoss:**
 * - More numerically stable (uses log-sum-exp trick)
 * - Avoids computing sigmoid explicitly
 * - Better gradient behavior for extreme values
 *
 * **Preferred Usage:**
 * Always use this instead of Sigmoid + BCELoss for better numerical stability.
 *
 * **Input Requirements:**
 * - input: Raw logits (before sigmoid)
 * - target: Binary labels (0 or 1)
 *
 * @param reduction How to reduce the loss (default: Mean)
 *
 * @par Complexity
 * - Time: O(n)
 * - Space: O(1) or O(n) depending on reduction mode
 *
 * @code
 * auto criterion = BCEWithLogitsLoss();
 * auto logits = model.forward(input);  // No sigmoid needed
 * auto loss = criterion(logits, binary_targets);
 * @endcode
 *
 * @see BCELoss
 */
class BCEWithLogitsLoss {
public:
    explicit BCEWithLogitsLoss(Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input, const Variable& target) -> Variable;
    auto operator()(const Variable& input, const Variable& target) -> Variable {
        return forward(input, target);
    }

private:
    Reduction reduction_;
};

/**
 * @brief Negative Log Likelihood (NLL) Loss
 *
 * Computes the negative log likelihood loss:
 *
 * \f[
 * \text{NLL}(x, y) = -\frac{1}{n}\sum_{i=1}^{n} x_{i,y_i}
 * \f]
 *
 * Expects log-probabilities as input (typically from LogSoftmax).
 * This is the second half of CrossEntropyLoss.
 *
 * **Use Cases:**
 * - When you already have log-probabilities
 * - Custom architectures where LogSoftmax is explicit
 * - Building blocks for more complex losses
 *
 * **Input Requirements:**
 * - input: Log-probabilities (from LogSoftmax), shape (N, C)
 * - target: Class indices, shape (N)
 *
 * **Note:** CrossEntropyLoss = LogSoftmax + NLLLoss
 *
 * @param reduction How to reduce the loss (default: Mean)
 *
 * @par Complexity
 * - Time: O(N)
 * - Space: O(1) or O(N) depending on reduction mode
 *
 * @code
 * auto log_softmax_layer = LogSoftmax(-1);
 * auto criterion = NLLLoss();
 * auto log_probs = log_softmax_layer.forward(logits);
 * auto loss = criterion(log_probs, targets);
 * @endcode
 *
 * @see CrossEntropyLoss, LogSoftmax
 */
class NLLLoss {
public:
    explicit NLLLoss(Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input, const Tensor& target) -> Variable;
    auto operator()(const Variable& input, const Tensor& target) -> Variable {
        return forward(input, target);
    }

private:
    Reduction reduction_;
};

/**
 * @brief L1 Loss (Mean Absolute Error)
 *
 * Computes the mean absolute error between input and target:
 *
 * \f[
 * \text{L1}(x, y) = \frac{1}{n}\sum_{i=1}^{n}|x_i - y_i|
 * \f]
 *
 * Also known as MAE (Mean Absolute Error).
 *
 * **Advantages over MSE:**
 * - Robust to outliers (linear penalty vs quadratic)
 * - More balanced gradient magnitude (doesn't amplify large errors)
 * - Better for data with outliers or noise
 *
 * **Disadvantages:**
 * - Not differentiable at zero (can cause optimization issues)
 * - Constant gradient magnitude (slower convergence near optimum)
 *
 * **Use Cases:**
 * - Regression with outliers
 * - When all errors should be penalized equally
 * - Robust estimation tasks
 *
 * @param reduction How to reduce the loss (default: Mean)
 *
 * @par Complexity
 * - Time: O(n)
 * - Space: O(1) or O(n) depending on reduction mode
 *
 * @code
 * auto criterion = L1Loss();
 * auto predictions = model.forward(input);
 * auto loss = criterion(predictions, targets);
 * @endcode
 *
 * @see MSELoss, SmoothL1Loss
 */
class L1Loss {
public:
    explicit L1Loss(Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input, const Variable& target) -> Variable;
    auto operator()(const Variable& input, const Variable& target) -> Variable {
        return forward(input, target);
    }

private:
    Reduction reduction_;
};

/**
 * @brief Smooth L1 Loss (Huber Loss)
 *
 * Computes a smooth combination of L1 and L2 loss:
 *
 * \f[
 * \text{SmoothL1}(x, y) = \frac{1}{n}\sum_{i=1}^{n} z_i, \quad \text{where} \quad
 * z_i = \begin{cases}
 *   0.5(x_i - y_i)^2 / \beta & \text{if } |x_i - y_i| < \beta \\
 *   |x_i - y_i| - 0.5\beta & \text{otherwise}
 * \end{cases}
 * \f]
 *
 * Also known as Huber loss. Combines advantages of MSE and L1:
 * - Quadratic for small errors (smooth gradients near optimum)
 * - Linear for large errors (robust to outliers)
 *
 * **Advantages:**
 * - Less sensitive to outliers than MSE
 * - Smoother than L1 (better convergence)
 * - Configurable transition point (beta parameter)
 *
 * **Use Cases:**
 * - Object detection (bounding box regression)
 * - Robust regression
 * - When you want MSE smoothness with L1 robustness
 *
 * **Parameter Tuning:**
 * - beta = 1.0: Standard Huber loss (default)
 * - Small beta: More like L1 (more robust)
 * - Large beta: More like L2 (smoother)
 *
 * @param reduction How to reduce the loss (default: Mean)
 * @param beta Threshold for switching between L1 and L2 (default: 1.0)
 *
 * @par Complexity
 * - Time: O(n)
 * - Space: O(1) or O(n) depending on reduction mode
 *
 * @code
 * auto criterion = SmoothL1Loss(Reduction::Mean, 1.0);
 * auto bbox_pred = model.forward(input);
 * auto loss = criterion(bbox_pred, bbox_targets);
 * @endcode
 *
 * @see MSELoss, L1Loss
 */
class SmoothL1Loss {
public:
    explicit SmoothL1Loss(Reduction reduction = Reduction::Mean, double beta = 1.0);

    auto forward(const Variable& input, const Variable& target) -> Variable;
    auto operator()(const Variable& input, const Variable& target) -> Variable {
        return forward(input, target);
    }

private:
    Reduction reduction_;
    double beta_;
};

/**
 * @defgroup functional_losses Functional Loss Functions
 * @brief Stateless loss functions for flexible use
 *
 * These functional versions provide direct computation without instantiating loss objects.
 * Useful for:
 * - Quick prototyping and experimentation
 * - Custom loss combinations
 * - When you don't need to reuse the same loss configuration
 *
 * @{
 */

/** @brief Functional MSE loss computation */
auto mse_loss(const Variable& input, const Variable& target,
             Reduction reduction = Reduction::Mean) -> Variable;

/** @brief Functional cross entropy loss computation */
auto cross_entropy(const Variable& input, const Tensor& target,
                  Reduction reduction = Reduction::Mean) -> Variable;

/** @brief Functional BCE loss computation */
auto bce_loss(const Variable& input, const Variable& target,
             Reduction reduction = Reduction::Mean) -> Variable;

/** @brief Functional NLL loss computation */
auto nll_loss(const Variable& input, const Tensor& target,
             Reduction reduction = Reduction::Mean) -> Variable;

/** @brief Functional L1 loss computation */
auto l1_loss(const Variable& input, const Variable& target,
            Reduction reduction = Reduction::Mean) -> Variable;

/** @} */ // end of functional_losses group

} // namespace nn
} // namespace tenzor
