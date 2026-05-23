#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include <optional>

namespace tenzor::nn {

// Helper functions using Variable's built-in scalar operators
namespace {
    // scalar - var = -(var - scalar)
    auto scalar_sub(float scalar, const Variable& var) -> Variable {
        return (var - scalar) * -1.0f;
    }
}

// MSELoss implementation
MSELoss::MSELoss(Reduction reduction) : reduction_(reduction) {}

auto MSELoss::forward(const Variable& input, const Variable& target) -> Variable {
    auto diff = input - target;
    auto squared = diff * diff;

    switch (reduction_) {
        case Reduction::None:
            return squared;
        case Reduction::Mean:
            return mean(squared);
        case Reduction::Sum:
            return sum(squared);
        case Reduction::BatchMean:
            return sum(squared) / static_cast<float>(squared.tensor().shape()[0]);
    }
    return squared;
}

// BCELoss implementation
BCELoss::BCELoss(Reduction reduction) : reduction_(reduction) {}

auto BCELoss::forward(const Variable& input, const Variable& target) -> Variable {
    // Numerically stable BCE using the same log-sum-exp decomposition as BCEWithLogitsLoss,
    // but for inputs already passed through sigmoid (i.e., in [0, 1]).
    //
    // Direct log(p) and log(1-p) are numerically unstable near 0 and 1.
    // Instead, compute via the logit (inverse sigmoid) and use the stable formula:
    //   logit = log(p / (1-p))  (clamp p to avoid log(0))
    //   BCE = max(logit, 0) - logit * target + log(1 + exp(-|logit|))
    auto p = clamp(input, 1e-7f, 1.0f - 1e-7f);
    // Compute logit = log(p / (1 - p))  where (1 - p) = -(p - 1)
    auto one_minus_p = scalar_sub(1.0f, p);
    auto logit = log(p / one_minus_p);
    // Stable BCE: max(logit, 0) - logit * target + log(1 + exp(-|logit|))
    auto abs_logit = abs(logit);
    auto neg_abs = neg(abs_logit);
    auto max_val = (logit + abs_logit) / 2.0f;  // max(logit, 0) = (logit + |logit|) / 2
    auto log_term = log(exp(neg_abs) + 1.0f);
    auto loss = max_val - logit * target + log_term;

    switch (reduction_) {
        case Reduction::None:
            return loss;
        case Reduction::Mean:
            return mean(loss);
        case Reduction::Sum:
            return sum(loss);
        case Reduction::BatchMean:
            return sum(loss) / static_cast<float>(loss.tensor().shape()[0]);
    }
    return loss;
}

// BCEWithLogitsLoss implementation
BCEWithLogitsLoss::BCEWithLogitsLoss(Reduction reduction) : reduction_(reduction) {}

auto BCEWithLogitsLoss::forward(const Variable& input, const Variable& target) -> Variable {
    // Use log-sum-exp trick for numerical stability
    // BCE = max(x, 0) - x * z + log(1 + exp(-abs(x)))
    // where x = input, z = target
    // Simplify using: max(x, 0) = (x + abs(x)) / 2

    auto abs_input = abs(input);
    auto neg_abs = neg(abs_input);

    // Element-wise max(x, 0) = (x + abs(x)) / 2
    auto max_val = (input + abs_input) / 2.0f;

    auto xz = input * target;  // x * z
    auto log_term = log(exp(neg_abs) + 1.0f);

    auto loss_unreduced = max_val - xz + log_term;

    switch (reduction_) {
        case Reduction::None:
            return loss_unreduced;
        case Reduction::Mean:
            return mean(loss_unreduced);
        case Reduction::Sum:
            return sum(loss_unreduced);
        case Reduction::BatchMean:
            return sum(loss_unreduced) / static_cast<float>(loss_unreduced.tensor().shape()[0]);
    }
    return loss_unreduced;
}

// CrossEntropyLoss implementation
CrossEntropyLoss::CrossEntropyLoss(Reduction reduction, float label_smoothing,
                                   int64_t ignore_index)
    : reduction_(reduction),
      label_smoothing_(label_smoothing),
      ignore_index_(ignore_index) {}

auto CrossEntropyLoss::forward(const Variable& input, const Tensor& target) -> Variable {
    // Cross entropy with logits: -log_softmax(input)[target_class]
    // Use the log_softmax function from activations
    auto log_probs = nn::log_softmax(input, 1);  // Compute log_softmax along dim=1

    auto num_classes = input.tensor().shape()[1];

    // Handle both one-hot encoded targets (Float32/Float64, shape [N, C]) and class indices (Int64, shape [N])
    Variable one_hot_var;

    // Check if target is a floating point type (one-hot encoded)
    bool is_float_target = (target.dtype() == DType::Float32 || target.dtype() == DType::Float64 || target.dtype() == DType::Float16) && target.ndim() == 2;

    // J.5: track which samples to ignore (class-index path only — one-hot
    // / soft targets carry weights directly so ignore_index is N/A there).
    std::optional<Tensor> keep_mask;

    if (is_float_target) {
        // Target is already one-hot encoded (Float32 or Float64 with shape [N, C])
        // Convert to input dtype if needed for consistency
        if (target.dtype() != input.tensor().dtype()) {
            one_hot_var = Variable(target.to(input.tensor().dtype()), false);
        } else {
            one_hot_var = Variable(target, false);  // Don't need gradients w.r.t. targets
        }
    } else {
        // Target contains class indices (Int64), need to create one-hot encoding

        // Ensure target is on same device as input for dispatch / mask ops.
        Tensor target_dev = (target.device() == input.tensor().device())
            ? target : target.to(input.tensor().device());

        // Build keep mask + clamp ignored entries to a safe class index so
        // OneHot doesn't index out-of-bounds. The mask is later applied to
        // the per-sample loss so ignored samples contribute zero.
        Tensor ignore_t   = tenzor::full_like(target_dev,
                                              static_cast<double>(ignore_index_));
        Tensor is_ignored = tenzor::eq(target_dev, ignore_t);
        Tensor keep       = tenzor::ne(target_dev, ignore_t);
        Tensor zero_t     = tenzor::full_like(target_dev, 0.0);
        Tensor safe_target = tenzor::where(is_ignored, zero_t, target_dev);
        keep_mask = keep.to(input.tensor().dtype());

        // Use backend dispatch for one-hot encoding (avoids GPU→CPU→GPU round-trip)
        NewOpAttributes oh_attrs;
        oh_attrs.set(AttrKey::NumClasses, num_classes);

        std::vector<Tensor> oh_inputs = {safe_target};
        auto oh_results = dispatch(OpId::OneHot, oh_inputs, oh_attrs);
        Tensor one_hot = oh_results[0];

        // Convert to input dtype if needed (one_hot shader produces Float32)
        if (one_hot.dtype() != input.tensor().dtype()) {
            one_hot = one_hot.to(input.tensor().dtype());
        }

        one_hot_var = Variable(one_hot, false);
    }

    // Apply label smoothing if enabled:
    // smoothed_target = (1 - label_smoothing) * one_hot + label_smoothing / num_classes
    if (label_smoothing_ > 0.0f) {
        auto smooth_target = one_hot_var * (1.0f - label_smoothing_);
        one_hot_var = smooth_target + (label_smoothing_ / static_cast<float>(num_classes));
    }

    // Compute element-wise product: log_probs * one_hot
    // This selects the log probability for the correct class
    auto selected_log_probs = log_probs * one_hot_var;

    // Sum across classes (dim=1) to get the selected log prob for each sample
    auto selected = sum(selected_log_probs, 1, false);  // Shape: [batch_size]

    // Negative for NLL
    auto neg_selected = neg(selected);

    // Apply ignore_index mask (class-index path only).
    if (keep_mask.has_value()) {
        Variable mask_var(*keep_mask, /*requires_grad=*/false);
        neg_selected = neg_selected * mask_var;
    }

    switch (reduction_) {
        case Reduction::None:
            return neg_selected;
        case Reduction::Mean: {
            if (keep_mask.has_value()) {
                // Divide by number of non-ignored samples (clamped to >=1).
                // Sum on Float32 because clamp_min lacks BF16 dispatch and
                // the count is a tiny scalar — no precision concern.
                const DType out_dtype = neg_selected.tensor().dtype();
                Tensor mask_f32 = keep_mask->dtype() == DType::Float32
                                     ? *keep_mask
                                     : keep_mask->to(DType::Float32);
                Tensor denom_f32 = tenzor::clamp_min(tenzor::sum(mask_f32), 1.0);
                Tensor denom = (denom_f32.dtype() == out_dtype)
                                   ? denom_f32
                                   : denom_f32.to(out_dtype);
                Variable denom_var(denom, /*requires_grad=*/false);
                return sum(neg_selected) / denom_var;
            }
            return mean(neg_selected);
        }
        case Reduction::Sum:
            return sum(neg_selected);
        case Reduction::BatchMean:
            return sum(neg_selected) / static_cast<float>(neg_selected.tensor().shape()[0]);
    }
    return neg_selected;
}

// NLLLoss implementation
NLLLoss::NLLLoss(Reduction reduction, int64_t ignore_index)
    : reduction_(reduction), ignore_index_(ignore_index) {}

auto NLLLoss::forward(const Variable& input, const Tensor& target) -> Variable {
    // Negative log likelihood from log-probabilities.
    //
    // Accept both forms of target, matching CrossEntropyLoss::forward:
    //   - one-hot encoded floating-point, shape [N, C]
    //   - class indices Int64/Int32, shape [N]
    //
    // Previously this silently multiplied a non-broadcastable (N,)
    // class-index tensor against the (N, C) log-probabilities, which raised
    // "mul: shapes [N] and [N,C] are not broadcast-compatible" whenever
    // PyTorch-style targets were passed.
    //
    // J.5: honour ignore_index on the class-index path. Sentinel targets are
    // clamped to 0 (so OneHot stays in-bounds) and the resulting per-sample
    // loss is zeroed via a (target != ignore_index) mask, with the mean
    // denominator counting only kept samples.
    auto num_classes = input.tensor().shape()[1];

    const bool is_float_target =
        (target.dtype() == DType::Float32 || target.dtype() == DType::Float64 ||
         target.dtype() == DType::Float16) && target.ndim() == 2;

    Variable one_hot_var;
    std::optional<Tensor> keep_mask;  // populated on class-index path only
    if (is_float_target) {
        // Already one-hot (or soft-labelled) with shape [N, C].
        if (target.dtype() != input.tensor().dtype()) {
            one_hot_var = Variable(target.to(input.tensor().dtype()), false);
        } else {
            one_hot_var = Variable(target, false);
        }
    } else {
        // Class-index path: dispatch OneHot on the input's device so we
        // avoid a GPU→CPU→GPU round-trip.
        Tensor target_dev = (target.device() == input.tensor().device())
            ? target
            : target.to(input.tensor().device());

        // Build keep mask + clamp ignored entries to a valid class index.
        Tensor ignore_t   = tenzor::full_like(target_dev,
                                              static_cast<double>(ignore_index_));
        Tensor is_ignored = tenzor::eq(target_dev, ignore_t);
        Tensor keep       = tenzor::ne(target_dev, ignore_t);
        Tensor zero_t     = tenzor::full_like(target_dev, 0.0);
        Tensor safe_target = tenzor::where(is_ignored, zero_t, target_dev);
        keep_mask = keep.to(input.tensor().dtype());

        NewOpAttributes oh_attrs;
        oh_attrs.set(AttrKey::NumClasses, num_classes);

        std::vector<Tensor> oh_inputs = {safe_target};
        auto oh_results = dispatch(OpId::OneHot, oh_inputs, oh_attrs);
        Tensor one_hot = oh_results[0];

        if (one_hot.dtype() != input.tensor().dtype()) {
            one_hot = one_hot.to(input.tensor().dtype());
        }
        one_hot_var = Variable(one_hot, false);
    }

    auto weighted = input * one_hot_var;                 // [N, C]
    auto loss_per_sample = sum(weighted, 1, false);      // [N]
    auto neg_loss = neg(loss_per_sample);

    // Apply ignore_index mask (class-index path only).
    if (keep_mask.has_value()) {
        Variable mask_var(*keep_mask, /*requires_grad=*/false);
        neg_loss = neg_loss * mask_var;
    }

    switch (reduction_) {
        case Reduction::None:
            return neg_loss;
        case Reduction::Mean: {
            if (keep_mask.has_value()) {
                // Same widen-narrow for the count denominator as in
                // NLLLoss above (clamp_min has no BF16 dispatch).
                const DType out_dtype = neg_loss.tensor().dtype();
                Tensor mask_f32 = keep_mask->dtype() == DType::Float32
                                     ? *keep_mask
                                     : keep_mask->to(DType::Float32);
                Tensor denom_f32 = tenzor::clamp_min(tenzor::sum(mask_f32), 1.0);
                Tensor denom = (denom_f32.dtype() == out_dtype)
                                   ? denom_f32
                                   : denom_f32.to(out_dtype);
                Variable denom_var(denom, /*requires_grad=*/false);
                return sum(neg_loss) / denom_var;
            }
            return mean(neg_loss);
        }
        case Reduction::Sum:
            return sum(neg_loss);
        case Reduction::BatchMean:
            return sum(neg_loss) / static_cast<float>(neg_loss.tensor().shape()[0]);
    }
    return neg_loss;
}

// L1Loss implementation
L1Loss::L1Loss(Reduction reduction) : reduction_(reduction) {}

auto L1Loss::forward(const Variable& input, const Variable& target) -> Variable {
    auto diff = input - target;
    auto abs_diff = abs(diff);

    switch (reduction_) {
        case Reduction::None:
            return abs_diff;
        case Reduction::Mean:
            return mean(abs_diff);
        case Reduction::Sum:
            return sum(abs_diff);
        case Reduction::BatchMean:
            return sum(abs_diff) / static_cast<float>(abs_diff.tensor().shape()[0]);
    }
    return abs_diff;
}

// SmoothL1Loss implementation
SmoothL1Loss::SmoothL1Loss(Reduction reduction, double beta)
    : reduction_(reduction), beta_(beta) {}

auto SmoothL1Loss::forward(const Variable& input, const Variable& target) -> Variable {
    auto diff = input - target;
    auto abs_diff = abs(diff);

    // Smooth L1 Loss (Huber Loss):
    // loss = 0.5 * (diff^2) / beta,           if |diff| < beta
    // loss = |diff| - 0.5 * beta,             otherwise

    // Exact SmoothL1Loss (Huber loss) without branching:
    //   loss = 0.5 * min(|diff|, beta)^2 / beta + max(|diff| - beta, 0)
    // When |diff| < beta:  0.5 * |diff|^2 / beta + 0
    // When |diff| >= beta: 0.5 * beta + |diff| - beta = |diff| - 0.5 * beta
    auto clamped_abs = clamp(abs_diff, 0.0f, static_cast<float>(beta_));
    auto excess = abs_diff - clamped_abs;  // max(|diff| - beta, 0)
    auto loss_unreduced = (clamped_abs * clamped_abs * 0.5f) / static_cast<float>(beta_) + excess;

    switch (reduction_) {
        case Reduction::None:
            return loss_unreduced;
        case Reduction::Mean:
            return mean(loss_unreduced);
        case Reduction::Sum:
            return sum(loss_unreduced);
        case Reduction::BatchMean:
            return sum(loss_unreduced) / static_cast<float>(loss_unreduced.tensor().shape()[0]);
    }
    return loss_unreduced;
}

// Functional implementations
auto mse_loss(const Variable& input, const Variable& target,
             Reduction reduction) -> Variable {
    MSELoss loss(reduction);
    return loss.forward(input, target);
}

auto cross_entropy(const Variable& input, const Tensor& target,
                  Reduction reduction, int64_t ignore_index) -> Variable {
    CrossEntropyLoss loss(reduction, /*label_smoothing=*/0.0f, ignore_index);
    return loss.forward(input, target);
}

auto bce_loss(const Variable& input, const Variable& target,
             Reduction reduction) -> Variable {
    BCELoss loss(reduction);
    return loss.forward(input, target);
}

auto nll_loss(const Variable& input, const Tensor& target,
             Reduction reduction, int64_t ignore_index) -> Variable {
    NLLLoss loss(reduction, ignore_index);
    return loss.forward(input, target);
}

auto l1_loss(const Variable& input, const Variable& target,
            Reduction reduction) -> Variable {
    L1Loss loss(reduction);
    return loss.forward(input, target);
}

// MarginRankingLoss implementation
MarginRankingLoss::MarginRankingLoss(double margin, Reduction reduction)
    : margin_(margin), reduction_(reduction) {}

auto MarginRankingLoss::forward(const Variable& input1, const Variable& input2,
                                const Variable& target) -> Variable {
    // loss = max(0, -y * (x1 - x2) + margin)
    auto diff = input1 - input2;
    auto neg_target_diff = target * diff * -1.0f;
    auto loss = relu(neg_target_diff + static_cast<float>(margin_));

    switch (reduction_) {
        case Reduction::None:
            return loss;
        case Reduction::Mean:
            return mean(loss);
        case Reduction::Sum:
            return sum(loss);
        case Reduction::BatchMean:
            return sum(loss) / static_cast<float>(loss.tensor().shape()[0]);
    }
    return loss;
}

auto margin_ranking_loss(const Variable& input1, const Variable& input2,
                        const Variable& target,
                        double margin, Reduction reduction) -> Variable {
    MarginRankingLoss loss_fn(margin, reduction);
    return loss_fn.forward(input1, input2, target);
}

} // namespace tenzor::nn
