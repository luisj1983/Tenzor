/**
 * @file metrics.cpp
 * @brief Training metrics implementation
 */

#include "tenzor/nn/metrics.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/advanced.hpp"

#include <stdexcept>

namespace tenzor {
namespace nn {

// ============================================================================
// Helper: convert predictions to class indices
// ============================================================================

/// If preds has >1 class dimension, argmax along dim 1; otherwise threshold at 0.5.
static auto to_class_indices(const Tensor& preds, [[maybe_unused]] int64_t num_classes) -> Tensor {
    auto shape = preds.shape();
    if (shape.size() >= 2 && shape[1] > 1) {
        // Multiclass logits/probabilities: shape (N, C, ...)
        return argmax(preds, /*dim=*/1);
    }
    // Binary: threshold at 0.5 -> cast to int64
    // preds >= 0.5 gives boolean-like tensor
    auto threshold = full(std::vector<int64_t>(shape.begin(), shape.end()),
                          0.5f, preds.dtype(), preds.device());
    return ge(preds, threshold).to(DType::Int64);
}

/// Update per-class TP, FP, FN counts given predicted and true class indices.
static auto update_confusion_counts(
    Tensor& tp, Tensor& fp, Tensor& fn,
    const Tensor& pred_classes, const Tensor& target_classes,
    [[maybe_unused]] int64_t num_classes) -> void
{
    // audit-2026-05-03 N1: bring inputs to CPU before raw-pointer access.
    // Otherwise CUDA / Vulkan / OneAPI inputs would yield device pointers
    // that the host loop below dereferences directly (hang / segv).
    auto p = pred_classes.reshape({-1}).to(DType::Int64)
                 .to(Device::cpu()).contiguous();
    auto t = target_classes.reshape({-1}).to(DType::Int64)
                 .to(Device::cpu()).contiguous();
    auto n = p.numel();

    // Access raw data for counting
    auto* p_data = p.data<int64_t>();
    auto* t_data = t.data<int64_t>();
    auto* tp_data = tp.data<float>();
    auto* fp_data = fp.data<float>();
    auto* fn_data = fn.data<float>();

    for (int64_t i = 0; i < n; ++i) {
        auto pred = p_data[i];
        auto true_cls = t_data[i];
        if (pred == true_cls) {
            tp_data[pred] += 1.0f;
        } else {
            fp_data[pred] += 1.0f;
            fn_data[true_cls] += 1.0f;
        }
    }
}

// ============================================================================
// Accuracy
// ============================================================================

Accuracy::Accuracy(int64_t num_classes)
    : num_classes_(num_classes) {}

auto Accuracy::update(const Tensor& preds, const Tensor& targets) -> void {
    auto pred_classes = to_class_indices(preds, num_classes_);
    // audit-2026-05-03 N1: move to CPU before raw-pointer iteration.
    auto target_flat = targets.reshape({-1}).to(DType::Int64)
                          .to(Device::cpu()).contiguous();
    auto pred_flat = pred_classes.reshape({-1}).to(DType::Int64)
                          .to(Device::cpu()).contiguous();

    auto n = pred_flat.numel();
    auto* p_data = pred_flat.data<int64_t>();
    auto* t_data = target_flat.data<int64_t>();

    int64_t batch_correct = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (p_data[i] == t_data[i]) {
            ++batch_correct;
        }
    }
    correct_ += batch_correct;
    total_ += n;
}

auto Accuracy::compute() -> Tensor {
    if (total_ == 0) {
        return zeros({1});
    }
    return full({1}, static_cast<float>(correct_) / static_cast<float>(total_));
}

auto Accuracy::reset() -> void {
    correct_ = 0;
    total_ = 0;
}

// ============================================================================
// Precision
// ============================================================================

Precision::Precision(int64_t num_classes, AverageMode average)
    : num_classes_(num_classes)
    , average_(average)
    , tp_(zeros({num_classes}))
    , fp_(zeros({num_classes}))
    , fn_(zeros({num_classes})) {}

auto Precision::update(const Tensor& preds, const Tensor& targets) -> void {
    auto pred_classes = to_class_indices(preds, num_classes_);
    update_confusion_counts(tp_, fp_, fn_, pred_classes, targets, num_classes_);
}

auto Precision::compute() -> Tensor {
    // precision_c = tp_c / (tp_c + fp_c)
    auto denom = tp_ + fp_;
    // Avoid division by zero: where denom == 0, set precision to 0
    auto zero_mask = eq(denom, zeros({num_classes_}));
    auto safe_denom = denom + zero_mask.to(DType::Float32);  // add 1 where zero
    auto per_class = tp_ / safe_denom;

    switch (average_) {
        case AverageMode::Micro: {
            auto tp_sum = sum(tp_);
            auto denom_sum = sum(tp_) + sum(fp_);
            auto d = denom_sum.item<float>();
            if (d == 0.0f) return zeros({1});
            return full({1}, tp_sum.item<float>() / d);
        }
        case AverageMode::Macro:
            return mean(per_class);
        case AverageMode::Weighted: {
            auto support = tp_ + fn_;  // per-class support
            auto total_support = sum(support);
            auto ts = total_support.item<float>();
            if (ts == 0.0f) return zeros({1});
            auto weights = support / total_support;
            return sum(per_class * weights);
        }
    }
    return mean(per_class);  // fallback
}

auto Precision::reset() -> void {
    tp_ = zeros({num_classes_});
    fp_ = zeros({num_classes_});
    fn_ = zeros({num_classes_});
}

// ============================================================================
// Recall
// ============================================================================

Recall::Recall(int64_t num_classes, AverageMode average)
    : num_classes_(num_classes)
    , average_(average)
    , tp_(zeros({num_classes}))
    , fp_(zeros({num_classes}))
    , fn_(zeros({num_classes})) {}

auto Recall::update(const Tensor& preds, const Tensor& targets) -> void {
    auto pred_classes = to_class_indices(preds, num_classes_);
    update_confusion_counts(tp_, fp_, fn_, pred_classes, targets, num_classes_);
}

auto Recall::compute() -> Tensor {
    // recall_c = tp_c / (tp_c + fn_c)
    auto denom = tp_ + fn_;
    auto zero_mask = eq(denom, zeros({num_classes_}));
    auto safe_denom = denom + zero_mask.to(DType::Float32);
    auto per_class = tp_ / safe_denom;

    switch (average_) {
        case AverageMode::Micro: {
            auto tp_sum = sum(tp_);
            auto denom_sum = sum(tp_) + sum(fn_);
            auto d = denom_sum.item<float>();
            if (d == 0.0f) return zeros({1});
            return full({1}, tp_sum.item<float>() / d);
        }
        case AverageMode::Macro:
            return mean(per_class);
        case AverageMode::Weighted: {
            auto support = tp_ + fn_;
            auto total_support = sum(support);
            auto ts = total_support.item<float>();
            if (ts == 0.0f) return zeros({1});
            auto weights = support / total_support;
            return sum(per_class * weights);
        }
    }
    return mean(per_class);
}

auto Recall::reset() -> void {
    tp_ = zeros({num_classes_});
    fp_ = zeros({num_classes_});
    fn_ = zeros({num_classes_});
}

// ============================================================================
// F1Score
// ============================================================================

F1Score::F1Score(int64_t num_classes, AverageMode average)
    : num_classes_(num_classes)
    , average_(average)
    , tp_(zeros({num_classes}))
    , fp_(zeros({num_classes}))
    , fn_(zeros({num_classes})) {}

auto F1Score::update(const Tensor& preds, const Tensor& targets) -> void {
    auto pred_classes = to_class_indices(preds, num_classes_);
    update_confusion_counts(tp_, fp_, fn_, pred_classes, targets, num_classes_);
}

auto F1Score::compute() -> Tensor {
    // F1 = 2*TP / (2*TP + FP + FN)
    auto numerator = tp_ * 2.0;
    auto denom = tp_ * 2.0 + fp_ + fn_;
    auto zero_mask = eq(denom, zeros({num_classes_}));
    auto safe_denom = denom + zero_mask.to(DType::Float32);
    auto per_class = numerator / safe_denom;

    switch (average_) {
        case AverageMode::Micro: {
            auto num_sum = sum(tp_).item<float>() * 2.0f;
            auto den_sum = sum(tp_).item<float>() * 2.0f
                         + sum(fp_).item<float>()
                         + sum(fn_).item<float>();
            if (den_sum == 0.0f) return zeros({1});
            return full({1}, num_sum / den_sum);
        }
        case AverageMode::Macro:
            return mean(per_class);
        case AverageMode::Weighted: {
            auto support = tp_ + fn_;
            auto total_support = sum(support);
            auto ts = total_support.item<float>();
            if (ts == 0.0f) return zeros({1});
            auto weights = support / total_support;
            return sum(per_class * weights);
        }
    }
    return mean(per_class);
}

auto F1Score::reset() -> void {
    tp_ = zeros({num_classes_});
    fp_ = zeros({num_classes_});
    fn_ = zeros({num_classes_});
}

// ============================================================================
// AUROC
// ============================================================================

auto AUROC::update(const Tensor& preds, const Tensor& targets) -> void {
    // Store flattened copies on CPU so compute() can iterate via raw
    // pointers regardless of input device (audit-2026-05-03 N1).
    all_preds_.push_back(preds.reshape({-1}).to(DType::Float32)
                              .to(Device::cpu()).contiguous());
    all_targets_.push_back(targets.reshape({-1}).to(DType::Float32)
                                .to(Device::cpu()).contiguous());
}

auto AUROC::compute() -> Tensor {
    if (all_preds_.empty()) {
        return zeros({1});
    }

    // Concatenate all stored predictions and targets
    auto all_p = cat(all_preds_, /*dim=*/0);
    auto all_t = cat(all_targets_, /*dim=*/0);

    auto n = all_p.numel();
    if (n == 0) return zeros({1});

    // Sort by prediction scores in descending order
    auto [sorted_scores, sort_indices] = sort(all_p, /*dim=*/0, /*descending=*/true);

    // sort() may produce results on the source device; force CPU layout
    // so the subsequent host-side trapezoidal walk sees valid pointers.
    sort_indices = sort_indices.to(Device::cpu()).contiguous();
    sorted_scores = sorted_scores.to(Device::cpu()).contiguous();
    all_t = all_t.to(Device::cpu()).contiguous();

    // Gather targets in sorted order
    auto* idx_data = sort_indices.data<int64_t>();
    auto* t_data = all_t.data<float>();

    // Count total positives and negatives
    int64_t total_pos = 0;
    int64_t total_neg = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (t_data[i] >= 0.5f) {
            ++total_pos;
        } else {
            ++total_neg;
        }
    }

    if (total_pos == 0 || total_neg == 0) {
        return zeros({1});  // Undefined AUROC
    }

    // Compute AUROC via trapezoidal rule on ROC curve
    // Walk thresholds from highest to lowest score
    double auc = 0.0;
    double prev_fpr = 0.0;
    double prev_tpr = 0.0;
    int64_t tp = 0;
    int64_t fp = 0;

    auto* s_data = sorted_scores.data<float>();

    for (int64_t i = 0; i < n; ++i) {
        auto idx = idx_data[i];
        if (t_data[idx] >= 0.5f) {
            ++tp;
        } else {
            ++fp;
        }

        // Only update ROC point when score changes or at the end
        if (i == n - 1 || s_data[i] != s_data[i + 1]) {
            double tpr = static_cast<double>(tp) / static_cast<double>(total_pos);
            double fpr = static_cast<double>(fp) / static_cast<double>(total_neg);

            // Trapezoidal area
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5;

            prev_fpr = fpr;
            prev_tpr = tpr;
        }
    }

    return full({1}, static_cast<float>(auc));
}

auto AUROC::reset() -> void {
    all_preds_.clear();
    all_targets_.clear();
}

// ============================================================================
// ConfusionMatrix
// ============================================================================

ConfusionMatrix::ConfusionMatrix(int64_t num_classes)
    : num_classes_(num_classes)
    , matrix_(zeros({num_classes, num_classes})) {}

auto ConfusionMatrix::update(const Tensor& preds, const Tensor& targets) -> void {
    // audit-2026-05-03 N1: bring inputs to CPU before raw-pointer access.
    auto pred_classes = to_class_indices(preds, num_classes_).reshape({-1})
                            .to(DType::Int64).to(Device::cpu()).contiguous();
    auto target_flat = targets.reshape({-1}).to(DType::Int64)
                          .to(Device::cpu()).contiguous();

    auto n = pred_classes.numel();
    auto* p_data = pred_classes.data<int64_t>();
    auto* t_data = target_flat.data<int64_t>();
    auto* m_data = matrix_.data<float>();

    for (int64_t i = 0; i < n; ++i) {
        auto true_cls = t_data[i];
        auto pred_cls = p_data[i];
        m_data[true_cls * num_classes_ + pred_cls] += 1.0f;
    }
}

auto ConfusionMatrix::compute() -> Tensor {
    return matrix_;
}

auto ConfusionMatrix::reset() -> void {
    matrix_ = zeros({num_classes_, num_classes_});
}

// ============================================================================
// MeanAbsoluteError
// ============================================================================

auto MeanAbsoluteError::update(const Tensor& preds, const Tensor& targets) -> void {
    auto diff = preds - targets;
    auto abs_diff = abs(diff);
    auto batch_sum = sum(abs_diff);
    // audit-2026-05-03 N1: item<float>() requires CPU + Float32 dtype.
    // Cast first so GPU / Float64 / half-precision inputs work.
    sum_abs_error_ += batch_sum.to(Device::cpu()).to(DType::Float32).item<float>();
    total_ += preds.numel();
}

auto MeanAbsoluteError::compute() -> Tensor {
    if (total_ == 0) return zeros({1});
    return full({1}, static_cast<float>(sum_abs_error_ / static_cast<double>(total_)));
}

auto MeanAbsoluteError::reset() -> void {
    sum_abs_error_ = 0.0;
    total_ = 0;
}

// ============================================================================
// MeanSquaredError
// ============================================================================

auto MeanSquaredError::update(const Tensor& preds, const Tensor& targets) -> void {
    auto diff = preds - targets;
    auto sq_diff = diff * diff;
    auto batch_sum = sum(sq_diff);
    // audit-2026-05-03 N1: see MAE comment.
    sum_sq_error_ += batch_sum.to(Device::cpu()).to(DType::Float32).item<float>();
    total_ += preds.numel();
}

auto MeanSquaredError::compute() -> Tensor {
    if (total_ == 0) return zeros({1});
    return full({1}, static_cast<float>(sum_sq_error_ / static_cast<double>(total_)));
}

auto MeanSquaredError::reset() -> void {
    sum_sq_error_ = 0.0;
    total_ = 0;
}

}  // namespace nn
}  // namespace tenzor
