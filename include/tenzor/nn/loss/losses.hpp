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
#include <string>
#include <stdexcept>

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
    None,       ///< No reduction, return per-element loss
    Mean,       ///< Average reduction (divide by number of elements)
    Sum,        ///< Sum reduction (sum all losses)
    BatchMean   ///< Sum over elements, divide by batch size (standard for KL divergence)
};

/**
 * @brief Parse a reduction mode string to enum.
 *
 * Provides backward compatibility for loss classes that accept string arguments.
 *
 * @param s Reduction mode string ("none", "mean", "sum", "batchmean")
 * @return Corresponding Reduction enum value
 * @throws std::invalid_argument if string is not recognized
 */
inline auto parse_reduction(const std::string& s) -> Reduction {
    if (s == "none") return Reduction::None;
    if (s == "mean") return Reduction::Mean;
    if (s == "sum") return Reduction::Sum;
    if (s == "batchmean") return Reduction::BatchMean;
    throw std::invalid_argument("Unknown reduction mode: " + s);
}

/**
 * @brief Convert a Reduction enum to its string representation.
 */
inline auto reduction_to_string(Reduction r) -> std::string {
    switch (r) {
        case Reduction::None: return "none";
        case Reduction::Mean: return "mean";
        case Reduction::Sum: return "sum";
        case Reduction::BatchMean: return "batchmean";
    }
    return "mean";
}

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
    /**
     * @brief Construct CrossEntropyLoss.
     *
     * @param reduction Reduction mode (None, Mean, Sum)
     * @param label_smoothing Smoothing factor in [0, 1). When > 0, targets become
     *        (1 - label_smoothing) * one_hot(target) + label_smoothing / num_classes.
     *        Default 0.0 (no smoothing, matches original behavior).
     */
    explicit CrossEntropyLoss(Reduction reduction = Reduction::Mean,
                              float label_smoothing = 0.0f);

    auto forward(const Variable& input, const Tensor& target) -> Variable;
    auto operator()(const Variable& input, const Tensor& target) -> Variable {
        return forward(input, target);
    }

private:
    Reduction reduction_;
    float label_smoothing_;
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

/**
 * @brief Kullback-Leibler Divergence Loss
 *
 * Computes the KL divergence between input and target distributions:
 *
 * \f[
 * \text{KL}(P||Q) = \sum_i P_i \cdot \log\left(\frac{P_i}{Q_i}\right) = \sum_i P_i \cdot (\log P_i - \log Q_i)
 * \f]
 *
 * Measures how one probability distribution diverges from a second, expected distribution.
 *
 * **Input Format:**
 * - input: Log-probabilities (log Q), typically from LogSoftmax
 * - target: Probabilities (P) or log-probabilities if log_target=true
 *
 * **Use Cases:**
 * - Distillation (matching teacher distribution)
 * - Variational inference (VAE loss)
 * - Policy gradient methods (RL)
 * - Measuring distribution similarity
 *
 * **Reduction Modes:**
 * - "mean": Average over all elements
 * - "sum": Sum all elements
 * - "batchmean": Sum over elements, divide by batch size (standard for KL)
 * - "none": Return per-element loss
 *
 * **Important:** Unlike CrossEntropy, KL divergence is asymmetric: KL(P||Q) ≠ KL(Q||P)
 *
 * @param reduction How to reduce the loss (default: Mean)
 * @param log_target If true, target is log-probabilities (default: false)
 *
 * @par Complexity
 * - Time: O(n)
 * - Space: O(1) or O(n) depending on reduction mode
 *
 * @code
 * auto criterion = KLDivLoss("batchmean");
 * auto student_log_probs = log_softmax(student_logits);  // Log Q
 * auto teacher_probs = softmax(teacher_logits);           // P
 * auto loss = criterion(student_log_probs, teacher_probs);
 * @endcode
 *
 * @see CrossEntropyLoss, NLLLoss
 */
class KLDivLoss {
public:
    explicit KLDivLoss(const std::string& reduction = "mean", bool log_target = false);
    explicit KLDivLoss(Reduction reduction, bool log_target = false)
        : KLDivLoss(reduction_to_string(reduction), log_target) {}

    auto forward(const Variable& input, const Variable& target) -> Variable;
    auto operator()(const Variable& input, const Variable& target) -> Variable {
        return forward(input, target);
    }

private:
    std::string reduction_;
    bool log_target_;
};

/**
 * @brief Focal Loss
 *
 * Addresses class imbalance by down-weighting easy examples:
 *
 * \f[
 * \text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
 * \f]
 *
 * where \f$p_t\f$ is the model's estimated probability for the target class.
 *
 * **Key Features:**
 * - Down-weights easy examples (high confidence correct predictions)
 * - Focuses training on hard examples
 * - Addresses extreme class imbalance
 * - α (alpha): Class weighting factor
 * - γ (gamma): Focusing parameter (default: 2.0)
 *
 * **Effect of Gamma:**
 * - γ = 0: Equivalent to CrossEntropyLoss
 * - γ = 1: Moderate focusing on hard examples
 * - γ = 2: Standard focal loss (recommended)
 * - γ = 5: Strong focusing on hardest examples
 *
 * **Use Cases:**
 * - Object detection (RetinaNet)
 * - Extreme class imbalance (1:1000 ratio)
 * - Medical diagnosis (rare diseases)
 * - Fraud detection
 *
 * **Typical Configuration:**
 * - alpha: 0.25 for foreground, 0.75 for background
 * - gamma: 2.0 (standard)
 *
 * @param alpha Weighting factor for classes (default: 1.0)
 * @param gamma Focusing parameter (default: 2.0)
 * @param reduction How to reduce the loss (default: Mean)
 *
 * @par Complexity
 * - Time: O(n)
 * - Space: O(1) or O(n) depending on reduction mode
 *
 * @code
 * auto criterion = FocalLoss(0.25, 2.0);
 * auto logits = model.forward(input);
 * auto loss = criterion(logits, targets);
 * @endcode
 *
 * @see CrossEntropyLoss, BCEWithLogitsLoss
 */
class FocalLoss {
public:
    explicit FocalLoss(double alpha = 1.0, double gamma = 2.0,
                      const std::string& reduction = "mean");
    FocalLoss(double alpha, double gamma, Reduction reduction)
        : FocalLoss(alpha, gamma, reduction_to_string(reduction)) {}

    auto forward(const Variable& input, const Variable& target) -> Variable;
    auto operator()(const Variable& input, const Variable& target) -> Variable {
        return forward(input, target);
    }

private:
    double alpha_;
    double gamma_;
    std::string reduction_;
};

/**
 * @brief Dice Loss
 *
 * Computes the Dice coefficient loss for segmentation tasks:
 *
 * \f[
 * \text{Dice} = 1 - \frac{2|X \cap Y| + \text{smooth}}{|X| + |Y| + \text{smooth}}
 * \f]
 *
 * Based on the Sørensen–Dice coefficient, measuring overlap between two sets.
 *
 * **Advantages:**
 * - Handles class imbalance naturally (no per-class weighting needed)
 * - Differentiable approximation of IoU
 * - Works well for small objects
 * - Range: [0, 1] where 0 is perfect overlap
 *
 * **Use Cases:**
 * - Medical image segmentation
 * - Semantic segmentation
 * - Instance segmentation
 * - Any task with large class imbalance in pixels/voxels
 *
 * **Smooth Parameter:**
 * - Prevents division by zero
 * - Typical values: 1.0 (Laplace smoothing) or 1e-5
 *
 * **Input Requirements:**
 * - input: Probabilities (after sigmoid/softmax), shape (N, C, H, W)
 * - target: Binary masks (0 or 1), same shape as input
 *
 * @param smooth Smoothing factor to avoid division by zero (default: 1.0)
 * @param reduction How to reduce the loss (default: Mean)
 *
 * @par Complexity
 * - Time: O(n)
 * - Space: O(1)
 *
 * @code
 * auto criterion = DiceLoss(1.0);
 * auto probs = sigmoid(logits);  // Convert to probabilities
 * auto loss = criterion(probs, masks);
 * @endcode
 *
 * @see BCEWithLogitsLoss, FocalLoss
 */
class DiceLoss {
public:
    explicit DiceLoss(double smooth = 1.0, const std::string& reduction = "mean");
    DiceLoss(double smooth, Reduction reduction)
        : DiceLoss(smooth, reduction_to_string(reduction)) {}

    auto forward(const Variable& input, const Variable& target) -> Variable;
    auto operator()(const Variable& input, const Variable& target) -> Variable {
        return forward(input, target);
    }

private:
    double smooth_;
    std::string reduction_;
};

/**
 * @brief Huber Loss (Smooth L1 Loss variant)
 *
 * Computes a robust loss that is quadratic for small errors and linear for large errors:
 *
 * \f[
 * L_\delta(x, y) = \begin{cases}
 *   \frac{1}{2}(x - y)^2 & \text{if } |x - y| < \delta \\
 *   \delta \cdot (|x - y| - \frac{\delta}{2}) & \text{otherwise}
 * \end{cases}
 * \f]
 *
 * **Advantages:**
 * - Robust to outliers (like L1)
 * - Smooth gradients near zero (like L2)
 * - Configurable transition point (delta)
 *
 * **Comparison:**
 * - L2 (MSE): Sensitive to outliers, smooth everywhere
 * - L1 (MAE): Robust but not smooth at zero
 * - Huber: Best of both worlds
 *
 * **Delta Parameter:**
 * - Controls transition between quadratic and linear regions
 * - Small delta: More like L1 (robust, less smooth)
 * - Large delta: More like L2 (smooth, less robust)
 * - Typical: 1.0
 *
 * **Use Cases:**
 * - Regression with outliers
 * - Reinforcement learning (value function estimation)
 * - Robust estimation
 * - Any task requiring balance between L1 and L2
 *
 * @param delta Threshold for switching between L1 and L2 (default: 1.0)
 * @param reduction How to reduce the loss (default: Mean)
 *
 * @par Complexity
 * - Time: O(n)
 * - Space: O(1) or O(n) depending on reduction mode
 *
 * @code
 * auto criterion = HuberLoss(1.0);
 * auto predictions = model.forward(input);
 * auto loss = criterion(predictions, targets);
 * @endcode
 *
 * @see MSELoss, L1Loss, SmoothL1Loss
 */
class HuberLoss {
public:
    explicit HuberLoss(double delta = 1.0, const std::string& reduction = "mean");
    HuberLoss(double delta, Reduction reduction)
        : HuberLoss(delta, reduction_to_string(reduction)) {}

    auto forward(const Variable& input, const Variable& target) -> Variable;
    auto operator()(const Variable& input, const Variable& target) -> Variable {
        return forward(input, target);
    }

private:
    double delta_;
    std::string reduction_;
};

/**
 * @defgroup functional_advanced_losses Functional Advanced Loss Functions
 * @brief Stateless advanced loss functions for flexible use
 * @{
 */

/** @brief Functional KL divergence loss computation */
auto kl_div_loss(const Variable& input, const Variable& target,
                const std::string& reduction = "mean",
                bool log_target = false) -> Variable;

/** @brief Functional focal loss computation */
auto focal_loss(const Variable& input, const Variable& target,
               double alpha = 1.0, double gamma = 2.0,
               const std::string& reduction = "mean") -> Variable;

/** @brief Functional Dice loss computation */
auto dice_loss(const Variable& input, const Variable& target,
              double smooth = 1.0,
              const std::string& reduction = "mean") -> Variable;

/** @brief Functional Huber loss computation */
auto huber_loss(const Variable& input, const Variable& target,
               double delta = 1.0,
               const std::string& reduction = "mean") -> Variable;

/** @brief Functional Huber loss computation */

/**
 * @brief Connectionist Temporal Classification (CTC) Loss.
 *
 * Used for sequence-to-sequence problems where the alignment between
 * input and target is unknown (e.g., speech recognition, OCR).
 *
 * The algorithm uses dynamic programming in log-space for numerical stability.
 *
 * Input:
 * - log_probs: (T, N, C) - log probabilities of each class at each timestep
 * - targets: (N, S) - target sequences (class indices, 0 = blank by default)
 * - input_lengths: (N,) - length of each input sequence
 * - target_lengths: (N,) - length of each target sequence
 *
 * @code
 * CTCLoss ctc("mean", 0);
 * auto loss = ctc(log_probs_var, targets, input_lengths, target_lengths);
 * @endcode
 */
class CTCLoss {
public:
    /**
     * @brief Construct CTC loss.
     *
     * @param reduction Reduction mode: "mean", "sum", or "none"
     * @param blank Index of blank label (default: 0)
     * @param zero_infinity If true, set infinite losses to zero (default: false)
     */
    explicit CTCLoss(const std::string& reduction = "mean",
                     int64_t blank = 0,
                     bool zero_infinity = false);
    CTCLoss(Reduction reduction, int64_t blank = 0, bool zero_infinity = false)
        : CTCLoss(reduction_to_string(reduction), blank, zero_infinity) {}

    auto forward(const Variable& log_probs, const Tensor& targets,
                 const Tensor& input_lengths, const Tensor& target_lengths) -> Variable;

    auto operator()(const Variable& log_probs, const Tensor& targets,
                   const Tensor& input_lengths, const Tensor& target_lengths) -> Variable {
        return forward(log_probs, targets, input_lengths, target_lengths);
    }

private:
    std::string reduction_;
    int64_t blank_;
    bool zero_infinity_;
};

/**
 * @brief Margin Ranking Loss
 *
 * Computes loss = max(0, -y * (x1 - x2) + margin)
 * where y is +1 or -1 indicating which input should rank higher.
 *
 * Used for ranking tasks and metric learning.
 *
 * @param margin Minimum desired margin (default: 0.0)
 * @param reduction Reduction mode (default: Mean)
 */
class MarginRankingLoss {
public:
    explicit MarginRankingLoss(double margin = 0.0,
                               Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input1, const Variable& input2,
                 const Variable& target) -> Variable;

    auto operator()(const Variable& input1, const Variable& input2,
                   const Variable& target) -> Variable {
        return forward(input1, input2, target);
    }

private:
    double margin_;
    Reduction reduction_;
};

// Functional form
auto margin_ranking_loss(const Variable& input1, const Variable& input2,
                        const Variable& target,
                        double margin = 0.0,
                        Reduction reduction = Reduction::Mean) -> Variable;

/**
 * @brief Soft Margin Loss
 *
 * Computes the two-class soft margin loss:
 *
 * \f[
 * \text{loss}(x, y) = \frac{1}{n}\sum_i \log(1 + \exp(-y_i \cdot x_i))
 * \f]
 *
 * where y is +1 or -1.
 *
 * @param reduction How to reduce the loss (default: Mean)
 */
class SoftMarginLoss {
public:
    explicit SoftMarginLoss(Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input, const Variable& target) -> Variable;
    auto operator()(const Variable& input, const Variable& target) -> Variable {
        return forward(input, target);
    }

private:
    Reduction reduction_;
};

auto soft_margin_loss(const Variable& input, const Variable& target,
                     Reduction reduction = Reduction::Mean) -> Variable;

/**
 * @brief Hinge Embedding Loss
 *
 * Measures whether two inputs are similar or dissimilar:
 *
 * \f[
 * \text{loss}(x, y) = \begin{cases}
 *   x_i & \text{if } y_i = 1 \\
 *   \max(0, \text{margin} - x_i) & \text{if } y_i = -1
 * \end{cases}
 * \f]
 *
 * @param margin Margin threshold (default: 1.0)
 * @param reduction How to reduce the loss (default: Mean)
 */
class HingeEmbeddingLoss {
public:
    explicit HingeEmbeddingLoss(double margin = 1.0,
                                 Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input, const Variable& target) -> Variable;
    auto operator()(const Variable& input, const Variable& target) -> Variable {
        return forward(input, target);
    }

private:
    double margin_;
    Reduction reduction_;
};

auto hinge_embedding_loss(const Variable& input, const Variable& target,
                          double margin = 1.0,
                          Reduction reduction = Reduction::Mean) -> Variable;

/**
 * @brief Poisson Negative Log-Likelihood Loss
 *
 * Computes the loss for a Poisson-distributed target:
 *
 * \f[
 * \text{loss}(x, y) = \exp(x) - y \cdot x
 * \f]
 *
 * when log_input=true (default), or:
 *
 * \f[
 * \text{loss}(x, y) = x - y \cdot \log(x + \epsilon)
 * \f]
 *
 * @param log_input If true, input is in log-space (default: true)
 * @param full Include Stirling approximation term (default: false)
 * @param eps Small value for numerical stability (default: 1e-8)
 * @param reduction How to reduce the loss (default: Mean)
 */
class PoissonNLLLoss {
public:
    explicit PoissonNLLLoss(bool log_input = true, bool full = false,
                            double eps = 1e-8,
                            Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input, const Variable& target) -> Variable;
    auto operator()(const Variable& input, const Variable& target) -> Variable {
        return forward(input, target);
    }

private:
    bool log_input_;
    bool full_;
    double eps_;
    Reduction reduction_;
};

auto poisson_nll_loss(const Variable& input, const Variable& target,
                      bool log_input = true, bool full = false,
                      double eps = 1e-8,
                      Reduction reduction = Reduction::Mean) -> Variable;

/**
 * @brief Cosine Embedding Loss
 *
 * Measures whether two inputs are similar (y=1) or dissimilar (y=-1)
 * using cosine similarity:
 *
 * \f[
 * \text{loss}(x_1, x_2, y) = \begin{cases}
 *   1 - \cos(x_1, x_2) & \text{if } y = 1 \\
 *   \max(0, \cos(x_1, x_2) - \text{margin}) & \text{if } y = -1
 * \end{cases}
 * \f]
 *
 * @param margin Margin for dissimilar pairs (default: 0.0)
 * @param reduction How to reduce the loss (default: Mean)
 */
class CosineEmbeddingLoss {
public:
    explicit CosineEmbeddingLoss(double margin = 0.0,
                                  Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input1, const Variable& input2,
                 const Variable& target) -> Variable;

    auto operator()(const Variable& input1, const Variable& input2,
                   const Variable& target) -> Variable {
        return forward(input1, input2, target);
    }

private:
    double margin_;
    Reduction reduction_;
};

auto cosine_embedding_loss(const Variable& input1, const Variable& input2,
                           const Variable& target,
                           double margin = 0.0,
                           Reduction reduction = Reduction::Mean) -> Variable;

/**
 * @brief Triplet Margin Loss
 *
 * Alias for TripletLoss (see contrastive.hpp). Measures relative distance
 * between anchor-positive and anchor-negative pairs.
 *
 * \f[
 * L(a, p, n) = \max(0, d(a, p) - d(a, n) + \text{margin})
 * \f]
 *
 * @param margin Margin between positive/negative distances (default: 1.0)
 * @param p Norm degree for distance (default: 2.0)
 * @param swap Use distance swap heuristic (default: false)
 * @param reduction How to reduce the loss (default: Mean)
 */
class TripletMarginLoss {
public:
    explicit TripletMarginLoss(double margin = 1.0, double p = 2.0,
                                bool swap = false,
                                Reduction reduction = Reduction::Mean);

    auto forward(const Variable& anchor, const Variable& positive,
                 const Variable& negative) -> Variable;

    auto operator()(const Variable& anchor, const Variable& positive,
                   const Variable& negative) -> Variable {
        return forward(anchor, positive, negative);
    }

private:
    double margin_;
    double p_;
    bool swap_;
    Reduction reduction_;
};

auto triplet_margin_loss(const Variable& anchor, const Variable& positive,
                         const Variable& negative,
                         double margin = 1.0, double p = 2.0,
                         bool swap = false,
                         Reduction reduction = Reduction::Mean) -> Variable;

/**
 * @brief Multi-Label Soft Margin Loss
 *
 * Multi-label one-versus-all loss based on max-entropy:
 *
 * \f[
 * \text{loss}(x, y) = -\frac{1}{C}\sum_c \left[ y_c \log(\sigma(x_c))
 *                      + (1-y_c) \log(1-\sigma(x_c)) \right]
 * \f]
 *
 * @param reduction How to reduce the loss (default: Mean)
 */
class MultiLabelSoftMarginLoss {
public:
    explicit MultiLabelSoftMarginLoss(Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input, const Variable& target) -> Variable;
    auto operator()(const Variable& input, const Variable& target) -> Variable {
        return forward(input, target);
    }

private:
    Reduction reduction_;
};

auto multi_label_soft_margin_loss(const Variable& input, const Variable& target,
                                   Reduction reduction = Reduction::Mean) -> Variable;

/**
 * @brief Multi-Class Margin Loss (Hinge Loss)
 *
 * Creates a criterion that optimizes a multi-class classification hinge loss:
 *
 * \f[
 * \text{loss}(x, y) = \frac{1}{C}\sum_{j \neq y} \max(0, \text{margin} - x_y + x_j)^p
 * \f]
 *
 * @param p Exponent (1 or 2, default: 1)
 * @param margin Margin threshold (default: 1.0)
 * @param reduction How to reduce the loss (default: Mean)
 */
class MultiMarginLoss {
public:
    explicit MultiMarginLoss(int p = 1, double margin = 1.0,
                              Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input, const Tensor& target) -> Variable;
    auto operator()(const Variable& input, const Tensor& target) -> Variable {
        return forward(input, target);
    }

private:
    int p_;
    double margin_;
    Reduction reduction_;
};

auto multi_margin_loss(const Variable& input, const Tensor& target,
                       int p = 1, double margin = 1.0,
                       Reduction reduction = Reduction::Mean) -> Variable;

/**
 * @brief Gaussian Negative Log-Likelihood Loss
 *
 * Computes the negative log-likelihood of a Gaussian distribution:
 *
 * \f[
 * \text{loss}(x, y, \sigma^2) = \frac{1}{2}\left[\log(\sigma^2) +
 *    \frac{(x - y)^2}{\sigma^2}\right]
 * \f]
 *
 * Optionally includes the constant \f$\frac{1}{2}\log(2\pi)\f$ when full=true.
 *
 * @param full Include constant term (default: false)
 * @param eps Minimum variance for numerical stability (default: 1e-6)
 * @param reduction How to reduce the loss (default: Mean)
 */
class GaussianNLLLoss {
public:
    explicit GaussianNLLLoss(bool full = false, double eps = 1e-6,
                              Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input, const Variable& target,
                 const Variable& var) -> Variable;

    auto operator()(const Variable& input, const Variable& target,
                   const Variable& var) -> Variable {
        return forward(input, target, var);
    }

private:
    bool full_;
    double eps_;
    Reduction reduction_;
};

auto gaussian_nll_loss(const Variable& input, const Variable& target,
                       const Variable& var,
                       bool full = false, double eps = 1e-6,
                       Reduction reduction = Reduction::Mean) -> Variable;

/** @} */ // end of functional_advanced_losses group

} // namespace nn
} // namespace tenzor
