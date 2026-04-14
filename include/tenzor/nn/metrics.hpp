/**
 * @file metrics.hpp
 * @brief Training metrics for evaluating model performance
 *
 * Provides stateful metric classes that accumulate state across batches
 * via update(), compute the final value via compute(), and can be reset
 * for a new epoch via reset().
 *
 * Inspired by TorchMetrics / Keras metrics.
 */

#pragma once

#include <cstdint>
#include <vector>
#include "tenzor/core/tensor.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Averaging mode for multi-class classification metrics.
 *
 * - **Micro**: Aggregate TP/FP/FN globally, then compute the metric.
 * - **Macro**: Compute per-class metric, then take the unweighted mean.
 * - **Weighted**: Compute per-class metric, weight by class support, then average.
 */
enum class AverageMode { Micro, Macro, Weighted };

/**
 * @brief Abstract base class for all training metrics.
 *
 * Subclasses implement the accumulate-compute-reset protocol:
 * @code
 * Accuracy acc;
 * for (auto& [preds, targets] : loader) {
 *     acc.update(preds, targets);
 * }
 * Tensor value = acc.compute();
 * acc.reset();
 * @endcode
 */
class Metric {
public:
    virtual ~Metric() = default;

    /** @brief Accumulate a batch of predictions and targets into internal state. */
    virtual auto update(const Tensor& preds, const Tensor& targets) -> void = 0;

    /** @brief Compute the metric from accumulated state. */
    virtual auto compute() -> Tensor = 0;

    /** @brief Reset internal state for a new epoch. */
    virtual auto reset() -> void = 0;

    /** @brief Human-readable name for logging (e.g. "accuracy", "f1_score"). */
    virtual auto name() const -> std::string = 0;
};

// ============================================================================
// Classification Metrics
// ============================================================================

/**
 * @brief Classification accuracy metric.
 *
 * For binary classification: (TP + TN) / total.
 * For multiclass: correct / total (argmax-based).
 *
 * @param num_classes Number of classes (1 or 2 = binary, >2 = multiclass)
 */
class Accuracy : public Metric {
public:
    explicit Accuracy(int64_t num_classes = 2);

    auto update(const Tensor& preds, const Tensor& targets) -> void override;
    auto compute() -> Tensor override;
    auto reset() -> void override;
    auto name() const -> std::string override { return "accuracy"; }

private:
    int64_t num_classes_;
    int64_t correct_ = 0;
    int64_t total_ = 0;
};

/**
 * @brief Precision metric: TP / (TP + FP).
 *
 * Supports micro, macro, and weighted averaging for multiclass.
 *
 * @param num_classes Number of classes
 * @param average Averaging mode (default: Macro)
 */
class Precision : public Metric {
public:
    explicit Precision(int64_t num_classes = 2,
                       AverageMode average = AverageMode::Macro);

    auto update(const Tensor& preds, const Tensor& targets) -> void override;
    auto compute() -> Tensor override;
    auto reset() -> void override;
    auto name() const -> std::string override { return "precision"; }

private:
    int64_t num_classes_;
    AverageMode average_;
    Tensor tp_;   // per-class true positive counts
    Tensor fp_;   // per-class false positive counts
    Tensor fn_;   // per-class false negative counts
};

/**
 * @brief Recall metric: TP / (TP + FN).
 *
 * Supports micro, macro, and weighted averaging for multiclass.
 *
 * @param num_classes Number of classes
 * @param average Averaging mode (default: Macro)
 */
class Recall : public Metric {
public:
    explicit Recall(int64_t num_classes = 2,
                    AverageMode average = AverageMode::Macro);

    auto update(const Tensor& preds, const Tensor& targets) -> void override;
    auto compute() -> Tensor override;
    auto reset() -> void override;
    auto name() const -> std::string override { return "recall"; }

private:
    int64_t num_classes_;
    AverageMode average_;
    Tensor tp_;
    Tensor fp_;
    Tensor fn_;
};

/**
 * @brief F1 Score metric: 2 * precision * recall / (precision + recall).
 *
 * Harmonic mean of precision and recall.
 *
 * @param num_classes Number of classes
 * @param average Averaging mode (default: Macro)
 */
class F1Score : public Metric {
public:
    explicit F1Score(int64_t num_classes = 2,
                     AverageMode average = AverageMode::Macro);

    auto update(const Tensor& preds, const Tensor& targets) -> void override;
    auto compute() -> Tensor override;
    auto reset() -> void override;
    auto name() const -> std::string override { return "f1_score"; }

private:
    int64_t num_classes_;
    AverageMode average_;
    Tensor tp_;
    Tensor fp_;
    Tensor fn_;
};

/**
 * @brief Area Under the ROC Curve (AUROC) metric.
 *
 * Binary classification only. Accumulates all predictions and targets,
 * then performs a threshold sweep at compute() time using the trapezoidal rule.
 */
class AUROC : public Metric {
public:
    AUROC() = default;

    auto update(const Tensor& preds, const Tensor& targets) -> void override;
    auto compute() -> Tensor override;
    auto reset() -> void override;
    auto name() const -> std::string override { return "auroc"; }

private:
    std::vector<Tensor> all_preds_;
    std::vector<Tensor> all_targets_;
};

/**
 * @brief Confusion matrix for N-class classification.
 *
 * Maintains an NxN count tensor where entry (i, j) is the number of
 * samples with true label i predicted as label j.
 *
 * @param num_classes Number of classes
 */
class ConfusionMatrix : public Metric {
public:
    explicit ConfusionMatrix(int64_t num_classes);

    auto update(const Tensor& preds, const Tensor& targets) -> void override;
    auto compute() -> Tensor override;
    auto reset() -> void override;
    auto name() const -> std::string override { return "confusion_matrix"; }

private:
    int64_t num_classes_;
    Tensor matrix_;
};

// ============================================================================
// Regression Metrics
// ============================================================================

/**
 * @brief Mean Absolute Error metric (running average).
 *
 * Accumulates the sum of absolute errors and sample count across batches.
 */
class MeanAbsoluteError : public Metric {
public:
    MeanAbsoluteError() = default;

    auto update(const Tensor& preds, const Tensor& targets) -> void override;
    auto compute() -> Tensor override;
    auto reset() -> void override;
    auto name() const -> std::string override { return "mae"; }

private:
    double sum_abs_error_ = 0.0;
    int64_t total_ = 0;
};

/**
 * @brief Mean Squared Error metric (running average).
 *
 * Accumulates the sum of squared errors and sample count across batches.
 */
class MeanSquaredError : public Metric {
public:
    MeanSquaredError() = default;

    auto update(const Tensor& preds, const Tensor& targets) -> void override;
    auto compute() -> Tensor override;
    auto reset() -> void override;
    auto name() const -> std::string override { return "mse"; }

private:
    double sum_sq_error_ = 0.0;
    int64_t total_ = 0;
};

}  // namespace nn
}  // namespace tenzor
