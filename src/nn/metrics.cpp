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
#include "tenzor/ops/indexing.hpp"

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
    // Binary: threshold at 0.5 -> cast to int64.
    // The threshold (and the comparison) MUST run in a floating dtype: for
    // integer preds a 0.5 threshold built in preds.dtype() truncates to 0, so
    // ge(preds, 0) marks every non-negative element as class 1 (F059). Use the
    // preds dtype when it is already floating, otherwise compare in Float32.
    const DType cmp_dtype = is_floating_type(preds.dtype()) ? preds.dtype()
                                                            : DType::Float32;
    Tensor preds_f = (preds.dtype() == cmp_dtype) ? preds : preds.to(cmp_dtype);
    auto threshold = full(std::vector<int64_t>(shape.begin(), shape.end()),
                          0.5, cmp_dtype, preds.device());
    return ge(preds_f, threshold).to(DType::Int64);
}

/// Update per-class TP, FP, FN counts given predicted and true class indices.
/// Vectorised on-device via bincount-with-weights so the per-element loop
/// no longer drags the full pred/target tensors back to the host. Only the
/// num_classes-sized partial counters cross the device boundary at the end.
static auto update_confusion_counts(
    Tensor& tp, Tensor& fp, Tensor& fn,
    const Tensor& pred_classes, const Tensor& target_classes,
    int64_t num_classes) -> void
{
    auto p = pred_classes.reshape({-1});
    auto t = target_classes.reshape({-1});
    if (p.dtype() != DType::Int64) p = p.to(DType::Int64);
    if (t.dtype() != DType::Int64) t = t.to(DType::Int64);
    if (p.device() != t.device()) t = t.to(p.device());
    p = p.contiguous();
    t = t.contiguous();

    if (p.numel() == 0) return;

    // correct_mask[i] = 1.0f if pred[i] == target[i], else 0.0f.
    Tensor correct_mask = ::tenzor::eq(p, t).to(DType::Float32);
    // wrong_mask = 1 - correct_mask, on the same device.
    Tensor one_t = ::tenzor::full({correct_mask.numel()}, 1.0,
                                  DType::Float32, correct_mask.device());
    Tensor wrong_mask = ::tenzor::sub(one_t, correct_mask);

    // bincount(indices, weights, num_classes) → (num_classes,) Float32.
    Tensor tp_part = ::tenzor::bincount(p, correct_mask, num_classes);
    Tensor fp_part = ::tenzor::bincount(p, wrong_mask,   num_classes);
    Tensor fn_part = ::tenzor::bincount(t, wrong_mask,   num_classes);

    if (tp_part.device() != tp.device()) tp_part = tp_part.to(tp.device());
    if (fp_part.device() != fp.device()) fp_part = fp_part.to(fp.device());
    if (fn_part.device() != fn.device()) fn_part = fn_part.to(fn.device());
    if (tp_part.dtype()  != tp.dtype())  tp_part = tp_part.to(tp.dtype());
    if (fp_part.dtype()  != fp.dtype())  fp_part = fp_part.to(fp.dtype());
    if (fn_part.dtype()  != fn.dtype())  fn_part = fn_part.to(fn.dtype());

    tp = ::tenzor::add(tp, tp_part);
    fp = ::tenzor::add(fp, fp_part);
    fn = ::tenzor::add(fn, fn_part);
}

// ============================================================================
// Accuracy
// ============================================================================

Accuracy::Accuracy(int64_t num_classes)
    : num_classes_(num_classes) {}

auto Accuracy::update(const Tensor& preds, const Tensor& targets) -> void {
    auto pred_classes = to_class_indices(preds, num_classes_);
    // Compute (pred == target).sum() on-device — only a single int64 scalar
    // is read back per call, replacing the previous per-element host loop
    // over class indices that triggered a full-tensor CPU roundtrip.
    auto pred_flat = pred_classes.reshape({-1}).to(DType::Int64);
    auto target_flat = targets.reshape({-1}).to(DType::Int64);
    if (pred_flat.device() != target_flat.device()) {
        target_flat = target_flat.to(pred_flat.device());
    }
    Tensor eq_t = ::tenzor::eq(pred_flat, target_flat).to(DType::Int64);
    Tensor sum_t = ::tenzor::sum(eq_t).to(DType::Int64).to(Device::cpu());
    int64_t batch_correct = sum_t.data<int64_t>()[0];
    correct_ += batch_correct;
    total_ += pred_flat.numel();
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
    // Keep tensors on their native device — compute() now does the
    // ROC-curve construction and trapezoidal sum on-device.
    all_preds_.push_back(preds.reshape({-1}).to(DType::Float32).contiguous());
    all_targets_.push_back(targets.reshape({-1}).to(DType::Float32).contiguous());
}

auto AUROC::compute() -> Tensor {
    if (all_preds_.empty()) {
        return zeros({1});
    }

    auto all_p = cat(all_preds_, /*dim=*/0);
    auto all_t = cat(all_targets_, /*dim=*/0);
    auto n = all_p.numel();
    if (n == 0) return zeros({1});

    Device dev = all_p.device();
    if (all_t.device() != dev) all_t = all_t.to(dev);

    // Sort scores descending; gather targets by the same permutation.
    auto [sorted_scores, sort_indices] = sort(all_p, /*dim=*/0, /*descending=*/true);
    Tensor sorted_t = ::tenzor::index_select(all_t, /*dim=*/0, sort_indices);

    // Binarise targets (>= 0.5) → positive indicator. cumsum gives running
    // TP count; the running FP count is `(i+1) - tp_cum`.
    Tensor half = ::tenzor::full({n}, 0.5, DType::Float32, dev);
    Tensor pos_mask = ::tenzor::ge(sorted_t, half).to(DType::Float32);
    Tensor tp_cum = ::tenzor::cumsum(pos_mask, /*dim=*/0);
    // Read total_pos / total_neg via single scalar read (last entry).
    Tensor total_pos_t = ::tenzor::slice(tp_cum, 0, n - 1, n).to(Device::cpu());
    float total_pos = total_pos_t.data<float>()[0];
    float total_neg = static_cast<float>(n) - total_pos;
    if (total_pos == 0.0f || total_neg == 0.0f) {
        return zeros({1});
    }

    // fp_cum[i] = (i+1) - tp_cum[i]
    Tensor idx_arange = ::tenzor::arange(1.0, static_cast<double>(n + 1), 1.0,
                                          DType::Float32, dev);
    Tensor fp_cum = ::tenzor::sub(idx_arange, tp_cum);

    // Tie grouping: select only entries that are the LAST of each tie group
    // (matches the reference CPU behaviour). is_last[i] is true iff i==n-1
    // or sorted_scores[i] != sorted_scores[i+1].
    Tensor is_last;
    if (n == 1) {
        is_last = ::tenzor::full({1}, 1.0, DType::Float32, dev);
    } else {
        Tensor s_curr = ::tenzor::slice(sorted_scores, 0, 0, n - 1);
        Tensor s_next = ::tenzor::slice(sorted_scores, 0, 1, n);
        Tensor diff = ::tenzor::ne(s_curr, s_next).to(DType::Float32);
        Tensor one_tail = ::tenzor::full({1}, 1.0, DType::Float32, dev);
        is_last = ::tenzor::cat({diff, one_tail}, /*dim=*/0);
    }
    Tensor mask_bool = ::tenzor::ne(is_last,
                                    ::tenzor::full({n}, 0.0, DType::Float32, dev));
    Tensor group_idx_2d = ::tenzor::nonzero(mask_bool);
    int64_t group_count = group_idx_2d.shape()[0];
    if (group_count == 0) {
        return zeros({1});
    }
    Tensor group_idx = ::tenzor::reshape(group_idx_2d, {group_count}).contiguous();

    // tpr / fpr at each group boundary, then prepend 0.
    Tensor tp_at = ::tenzor::index_select(tp_cum, 0, group_idx);
    Tensor fp_at = ::tenzor::index_select(fp_cum, 0, group_idx);
    Tensor tpr = ::tenzor::div(tp_at, total_pos);
    Tensor fpr = ::tenzor::div(fp_at, total_neg);

    Tensor zero_lead = ::tenzor::full({1}, 0.0, DType::Float32, dev);
    Tensor tpr_full = ::tenzor::cat({zero_lead, tpr}, 0);
    Tensor fpr_full = ::tenzor::cat({zero_lead, fpr}, 0);

    int64_t k = tpr_full.numel();
    if (k <= 1) {
        return zeros({1});
    }

    // Trapezoidal: 0.5 * sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1])).
    Tensor fpr_a = ::tenzor::slice(fpr_full, 0, 0, k - 1);
    Tensor fpr_b = ::tenzor::slice(fpr_full, 0, 1, k);
    Tensor tpr_a = ::tenzor::slice(tpr_full, 0, 0, k - 1);
    Tensor tpr_b = ::tenzor::slice(tpr_full, 0, 1, k);
    Tensor dfpr = ::tenzor::sub(fpr_b, fpr_a);
    Tensor stpr = ::tenzor::add(tpr_a, tpr_b);
    Tensor area_pieces = ::tenzor::mul(dfpr, stpr);
    Tensor auc_t = ::tenzor::sum(area_pieces);
    auc_t = ::tenzor::mul(auc_t, 0.5);

    return ::tenzor::reshape(auc_t.to(Device::cpu()), {int64_t(1)});
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
    // Vectorised on-device via bincount on flat indices = true*C + pred.
    // Replaces the previous per-element host loop that pulled the entire
    // (preds, targets) pair to CPU each call.
    auto p = to_class_indices(preds, num_classes_).reshape({-1});
    auto t = targets.reshape({-1});
    if (p.dtype() != DType::Int64) p = p.to(DType::Int64);
    if (t.dtype() != DType::Int64) t = t.to(DType::Int64);
    if (p.device() != t.device()) t = t.to(p.device());
    if (p.numel() == 0) return;
    p = p.contiguous();
    t = t.contiguous();
    Device dev = p.device();

    int64_t n = p.numel();
    Tensor C_t = ::tenzor::full({n}, static_cast<double>(num_classes_),
                                 DType::Int64, dev);
    Tensor flat_idx = ::tenzor::add(::tenzor::mul(t, C_t), p);

    int64_t total_bins = num_classes_ * num_classes_;
    Tensor flat_counts = ::tenzor::bincount(flat_idx, std::nullopt, total_bins);
    if (flat_counts.dtype() != matrix_.dtype()) {
        flat_counts = flat_counts.to(matrix_.dtype());
    }
    Tensor flat_counts_2d = ::tenzor::reshape(flat_counts, {num_classes_, num_classes_});
    if (flat_counts_2d.device() != matrix_.device()) {
        flat_counts_2d = flat_counts_2d.to(matrix_.device());
    }
    matrix_ = ::tenzor::add(matrix_, flat_counts_2d);
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
