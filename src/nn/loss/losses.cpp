#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/nn/utils/variable_cast.hpp"
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
        case Reduction::BatchMean: {
            const auto& shp = squared.tensor().shape();
            int64_t bs = (!shp.empty()) ? shp[0] : 0;
            if (bs > 0) {
                return sum(squared) / static_cast<float>(bs);
            }
            return mean(squared);
        }
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
    //
    // AA.4 / EE.9: clamp(p, 1e-7, 1 - 1e-7) on Float16 collapses — 1e-7f
    // is below F16's smallest normal (~6.1e-5), so the lower bound rounds
    // to 0 and log(0) blows up. Widen the entire computation to Float32
    // and cast the loss back to the input dtype at the end. `variable_cast`
    // wires a TypeCastBackward node so gradients flow through the cast.
    const DType orig_dtype = input.tensor().dtype();
    const bool needs_upcast = (orig_dtype == DType::Float16 ||
                               orig_dtype == DType::BFloat16);
    Variable input_f32 = needs_upcast
        ? tenzor::nn::variable_cast(input, DType::Float32)
        : input;
    Variable target_f32 = needs_upcast
        ? tenzor::nn::variable_cast(target, DType::Float32)
        : target;

    auto p = clamp(input_f32, 1e-7f, 1.0f - 1e-7f);
    // Compute logit = log(p / (1 - p))  where (1 - p) = -(p - 1)
    auto one_minus_p = scalar_sub(1.0f, p);
    auto logit = log(p / one_minus_p);
    // Stable BCE: max(logit, 0) - logit * target + log(1 + exp(-|logit|))
    auto abs_logit = abs(logit);
    auto neg_abs = neg(abs_logit);
    auto max_val = (logit + abs_logit) / 2.0f;  // max(logit, 0) = (logit + |logit|) / 2
    auto log_term = log(exp(neg_abs) + 1.0f);
    auto loss = max_val - logit * target_f32 + log_term;

    Variable reduced;
    switch (reduction_) {
        case Reduction::None:
            reduced = loss;
            break;
        case Reduction::Mean:
            reduced = mean(loss);
            break;
        case Reduction::Sum:
            reduced = sum(loss);
            break;
        case Reduction::BatchMean: {
            const auto& shp = loss.tensor().shape();
            int64_t bs = (!shp.empty()) ? shp[0] : 0;
            reduced = (bs > 0) ? (sum(loss) / static_cast<float>(bs)) : mean(loss);
            break;
        }
    }
    if (needs_upcast) {
        reduced = tenzor::nn::variable_cast(reduced, orig_dtype);
    }
    return reduced;
}

// BCEWithLogitsLoss implementation
BCEWithLogitsLoss::BCEWithLogitsLoss(Reduction reduction) : reduction_(reduction) {}

auto BCEWithLogitsLoss::forward(const Variable& input, const Variable& target) -> Variable {
    // Use log-sum-exp trick for numerical stability
    // BCE = max(x, 0) - x * z + log(1 + exp(-abs(x)))
    // where x = input, z = target
    // Simplify using: max(x, 0) = (x + abs(x)) / 2
    //
    // NN.14 / AA.4 / KK.18: For Float16/BFloat16, exp(-|x|) flushes to 0 when
    // |x| ≥ ~7 in F16 (smallest normal ~6.1e-5).  Widen the computation to
    // Float32 and cast the reduced loss back via variable_cast (mirrors BCELoss
    // above).
    const DType orig_dtype = input.tensor().dtype();
    const bool needs_upcast = (orig_dtype == DType::Float16 ||
                               orig_dtype == DType::BFloat16);
    Variable input_f32 = needs_upcast
        ? tenzor::nn::variable_cast(input, DType::Float32)
        : input;
    Variable target_f32 = needs_upcast
        ? tenzor::nn::variable_cast(target, DType::Float32)
        : target;

    auto abs_input = abs(input_f32);
    auto neg_abs = neg(abs_input);

    // Element-wise max(x, 0) = (x + abs(x)) / 2
    auto max_val = (input_f32 + abs_input) / 2.0f;

    auto xz = input_f32 * target_f32;  // x * z
    auto log_term = log(exp(neg_abs) + 1.0f);

    auto loss_unreduced = max_val - xz + log_term;

    Variable reduced;
    switch (reduction_) {
        case Reduction::None:
            reduced = loss_unreduced;
            break;
        case Reduction::Mean:
            reduced = mean(loss_unreduced);
            break;
        case Reduction::Sum:
            reduced = sum(loss_unreduced);
            break;
        case Reduction::BatchMean: {
            const auto& shp = loss_unreduced.tensor().shape();
            int64_t bs = (!shp.empty()) ? shp[0] : 0;
            reduced = (bs > 0)
                ? (sum(loss_unreduced) / static_cast<float>(bs))
                : mean(loss_unreduced);
            break;
        }
    }
    if (needs_upcast) {
        reduced = tenzor::nn::variable_cast(reduced, orig_dtype);
    }
    return reduced;
}

// CrossEntropyLoss implementation
CrossEntropyLoss::CrossEntropyLoss(Reduction reduction, float label_smoothing,
                                   int64_t ignore_index)
    : reduction_(reduction),
      label_smoothing_(label_smoothing),
      ignore_index_(ignore_index) {}

auto CrossEntropyLoss::forward(const Variable& input, const Tensor& target) -> Variable {
    // KK.19: log_softmax in F16/BF16 + label_smoothing/num_classes collapses
    // to zero for large vocab (label_smoothing=0.1 / 32k ≈ 3.05e-6, below
    // F16's smallest normal ~6.1e-5).  Widen the entire forward to Float32
    // and narrow the loss back before the final reduction.  Subsumes the
    // earlier BF16 clamp_min workaround on the ignore-index denominator.
    const DType orig_dtype = input.tensor().dtype();
    const bool needs_upcast = (orig_dtype == DType::Float16 ||
                               orig_dtype == DType::BFloat16);
    Variable input_c = needs_upcast
        ? tenzor::nn::variable_cast(input, DType::Float32)
        : input;
    // Upcast a half-precision target to Float32 independently of the input
    // dtype: a Float32 input combined with a 2-D BFloat16/Float16 one-hot/soft
    // target must still be recognised as a float (one-hot) target. Gating the
    // upcast on the input dtype left a BF16 soft target as BF16, which the
    // dtype probe below then misclassified as integer class indices.
    Tensor target_c = (target.dtype() == DType::Float16 ||
                       target.dtype() == DType::BFloat16)
        ? target.to(DType::Float32)
        : target;

    // Cross entropy with logits: -log_softmax(input)[target_class]
    // Use the log_softmax function from activations
    auto log_probs = nn::log_softmax(input_c, 1);  // Compute log_softmax along dim=1

    auto num_classes = input_c.tensor().shape()[1];

    // Handle both one-hot encoded targets (Float32/Float64, shape [N, C]) and class indices (Int64, shape [N])
    Variable one_hot_var;

    // Check if target is a floating point type (one-hot encoded).  KK.19:
    // use the upcast target_c (F16/BF16 -> F32) for the dtype probe and the
    // downstream one-hot construction so the smoothed-target arithmetic is
    // entirely Float32.
    bool is_float_target = (target_c.dtype() == DType::Float32 || target_c.dtype() == DType::Float64 || target_c.dtype() == DType::Float16 || target_c.dtype() == DType::BFloat16) && target_c.ndim() == 2;

    // J.5: track which samples to ignore (class-index path only — one-hot
    // / soft targets carry weights directly so ignore_index is N/A there).
    std::optional<Tensor> keep_mask;

    if (is_float_target) {
        // Target is already one-hot encoded (Float32 or Float64 with shape [N, C])
        // Convert to input dtype if needed for consistency
        if (target_c.dtype() != input_c.tensor().dtype()) {
            one_hot_var = Variable(target_c.to(input_c.tensor().dtype()), false);
        } else {
            one_hot_var = Variable(target_c, false);  // Don't need gradients w.r.t. targets
        }
    } else {
        // Target contains class indices (Int64), need to create one-hot encoding

        // Ensure target is on same device as input for dispatch / mask ops.
        Tensor target_dev = (target_c.device() == input_c.tensor().device())
            ? target_c : target_c.to(input_c.tensor().device());

        // Build keep mask + clamp ignored entries to a safe class index so
        // OneHot doesn't index out-of-bounds. The mask is later applied to
        // the per-sample loss so ignored samples contribute zero.
        Tensor ignore_t   = tenzor::full_like(target_dev,
                                              static_cast<double>(ignore_index_));
        Tensor is_ignored = tenzor::eq(target_dev, ignore_t);
        Tensor keep       = tenzor::ne(target_dev, ignore_t);
        Tensor zero_t     = tenzor::full_like(target_dev, 0.0);
        Tensor safe_target = tenzor::where(is_ignored, zero_t, target_dev);
        keep_mask = keep.to(input_c.tensor().dtype());

        // Use backend dispatch for one-hot encoding (avoids GPU→CPU→GPU round-trip)
        NewOpAttributes oh_attrs;
        oh_attrs.set(AttrKey::NumClasses, num_classes);

        std::vector<Tensor> oh_inputs = {safe_target};
        auto oh_results = dispatch(OpId::OneHot, oh_inputs, oh_attrs);
        Tensor one_hot = oh_results[0];

        // Convert to input dtype if needed (one_hot shader produces Float32)
        if (one_hot.dtype() != input_c.tensor().dtype()) {
            one_hot = one_hot.to(input_c.tensor().dtype());
        }

        one_hot_var = Variable(one_hot, false);
    }

    // Apply label smoothing if enabled:
    // smoothed_target = (1 - label_smoothing) * one_hot + label_smoothing / num_classes
    // KK.19: this runs in input_c.dtype() (Float32 when the original input was
    // F16/BF16), so label_smoothing/num_classes no longer underflows.
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

    // KK.19: narrow back to the original dtype only after the reduction.
    // Keeping the per-sample tensor in F32 preserves precision for the
    // mean/sum reductions; the final scalar (or per-sample None reduction)
    // can then be cast back to F16/BF16 cheaply.
    auto finalize = [&](Variable v) -> Variable {
        if (needs_upcast) {
            return tenzor::nn::variable_cast(v, orig_dtype);
        }
        return v;
    };

    switch (reduction_) {
        case Reduction::None:
            return finalize(neg_selected);
        case Reduction::Mean: {
            if (keep_mask.has_value()) {
                // Divide by number of non-ignored samples (clamped to >=1).
                // KK.19: with the F32 widen above, the denom and neg_selected
                // share dtype (Float32 on upcast paths), so the clamp_min /
                // dtype-bridge dance from before becomes a no-op.  The path
                // still works for non-upcast (F32/F64) inputs as it always did.
                const DType out_dtype = neg_selected.tensor().dtype();
                Tensor mask_f32 = keep_mask->dtype() == DType::Float32
                                     ? *keep_mask
                                     : keep_mask->to(DType::Float32);
                Tensor denom_f32 = tenzor::clamp_min(tenzor::sum(mask_f32), 1.0);
                Tensor denom = (denom_f32.dtype() == out_dtype)
                                   ? denom_f32
                                   : denom_f32.to(out_dtype);
                Variable denom_var(denom, /*requires_grad=*/false);
                return finalize(sum(neg_selected) / denom_var);
            }
            return finalize(mean(neg_selected));
        }
        case Reduction::Sum:
            return finalize(sum(neg_selected));
        case Reduction::BatchMean: {
            const auto& shp = neg_selected.tensor().shape();
            int64_t bs = (!shp.empty()) ? shp[0] : 0;
            if (bs > 0) {
                return finalize(sum(neg_selected) / static_cast<float>(bs));
            }
            return finalize(mean(neg_selected));
        }
    }
    return finalize(neg_selected);
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
    //
    // KK.19: widen F16/BF16 inputs/targets to Float32 for the forward to
    // avoid label-smoothing-like underflow paths and to keep parity with
    // CrossEntropyLoss above; narrow the loss back at the very end.
    const DType orig_dtype = input.tensor().dtype();
    const bool needs_upcast = (orig_dtype == DType::Float16 ||
                               orig_dtype == DType::BFloat16);
    Variable input_c = needs_upcast
        ? tenzor::nn::variable_cast(input, DType::Float32)
        : input;
    // Upcast a half-precision target to Float32 independently of the input
    // dtype: a Float32 input combined with a 2-D BFloat16/Float16 one-hot/soft
    // target must still be recognised as a float (one-hot) target. Gating the
    // upcast on the input dtype left a BF16 soft target as BF16, which the
    // dtype probe below then misclassified as integer class indices.
    Tensor target_c = (target.dtype() == DType::Float16 ||
                       target.dtype() == DType::BFloat16)
        ? target.to(DType::Float32)
        : target;

    auto num_classes = input_c.tensor().shape()[1];

    const bool is_float_target =
        (target_c.dtype() == DType::Float32 || target_c.dtype() == DType::Float64 ||
         target_c.dtype() == DType::Float16 || target_c.dtype() == DType::BFloat16) &&
        target_c.ndim() == 2;

    Variable one_hot_var;
    std::optional<Tensor> keep_mask;  // populated on class-index path only
    if (is_float_target) {
        // Already one-hot (or soft-labelled) with shape [N, C].
        if (target_c.dtype() != input_c.tensor().dtype()) {
            one_hot_var = Variable(target_c.to(input_c.tensor().dtype()), false);
        } else {
            one_hot_var = Variable(target_c, false);
        }
    } else {
        // Class-index path: dispatch OneHot on the input's device so we
        // avoid a GPU→CPU→GPU round-trip.
        Tensor target_dev = (target_c.device() == input_c.tensor().device())
            ? target_c
            : target_c.to(input_c.tensor().device());

        // Build keep mask + clamp ignored entries to a valid class index.
        Tensor ignore_t   = tenzor::full_like(target_dev,
                                              static_cast<double>(ignore_index_));
        Tensor is_ignored = tenzor::eq(target_dev, ignore_t);
        Tensor keep       = tenzor::ne(target_dev, ignore_t);
        Tensor zero_t     = tenzor::full_like(target_dev, 0.0);
        Tensor safe_target = tenzor::where(is_ignored, zero_t, target_dev);
        keep_mask = keep.to(input_c.tensor().dtype());

        NewOpAttributes oh_attrs;
        oh_attrs.set(AttrKey::NumClasses, num_classes);

        std::vector<Tensor> oh_inputs = {safe_target};
        auto oh_results = dispatch(OpId::OneHot, oh_inputs, oh_attrs);
        Tensor one_hot = oh_results[0];

        if (one_hot.dtype() != input_c.tensor().dtype()) {
            one_hot = one_hot.to(input_c.tensor().dtype());
        }
        one_hot_var = Variable(one_hot, false);
    }

    auto weighted = input_c * one_hot_var;               // [N, C]
    auto loss_per_sample = sum(weighted, 1, false);      // [N]
    auto neg_loss = neg(loss_per_sample);

    // Apply ignore_index mask (class-index path only).
    if (keep_mask.has_value()) {
        Variable mask_var(*keep_mask, /*requires_grad=*/false);
        neg_loss = neg_loss * mask_var;
    }

    auto finalize = [&](Variable v) -> Variable {
        if (needs_upcast) {
            return tenzor::nn::variable_cast(v, orig_dtype);
        }
        return v;
    };

    switch (reduction_) {
        case Reduction::None:
            return finalize(neg_loss);
        case Reduction::Mean: {
            if (keep_mask.has_value()) {
                // Same widen-narrow for the count denominator as in
                // CrossEntropyLoss above (clamp_min has no BF16 dispatch).
                const DType out_dtype = neg_loss.tensor().dtype();
                Tensor mask_f32 = keep_mask->dtype() == DType::Float32
                                     ? *keep_mask
                                     : keep_mask->to(DType::Float32);
                Tensor denom_f32 = tenzor::clamp_min(tenzor::sum(mask_f32), 1.0);
                Tensor denom = (denom_f32.dtype() == out_dtype)
                                   ? denom_f32
                                   : denom_f32.to(out_dtype);
                Variable denom_var(denom, /*requires_grad=*/false);
                return finalize(sum(neg_loss) / denom_var);
            }
            return finalize(mean(neg_loss));
        }
        case Reduction::Sum:
            return finalize(sum(neg_loss));
        case Reduction::BatchMean: {
            // Guard against a scalar/empty batch: shape()[0] would be OOB on a
            // 0-d tensor and div-by-zero on an empty batch. Mirror the sibling
            // losses' guarded BatchMean pattern (e.g. line ~99 above).
            const auto& shp = neg_loss.tensor().shape();
            int64_t bs = (!shp.empty()) ? shp[0] : 0;
            return finalize((bs > 0)
                                ? (sum(neg_loss) / static_cast<float>(bs))
                                : mean(neg_loss));
        }
    }
    return finalize(neg_loss);
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
        case Reduction::BatchMean: {
            const auto& shp = abs_diff.tensor().shape();
            int64_t bs = (!shp.empty()) ? shp[0] : 0;
            if (bs > 0) {
                return sum(abs_diff) / static_cast<float>(bs);
            }
            return mean(abs_diff);
        }
    }
    return abs_diff;
}

// SmoothL1Loss implementation
SmoothL1Loss::SmoothL1Loss(Reduction reduction, double beta)
    : reduction_(reduction), beta_(beta) {
    // A negative beta is meaningless (the transition threshold cannot be < 0)
    // and would produce NaN/garbage. beta == 0 is permitted and degenerates to
    // L1Loss (matching PyTorch semantics); it is handled in forward().
    if (beta_ < 0.0) {
        throw std::invalid_argument(
            "SmoothL1Loss: beta must be >= 0 (got " + std::to_string(beta_) + ")");
    }
}

auto SmoothL1Loss::forward(const Variable& input, const Variable& target) -> Variable {
    auto diff = input - target;
    auto abs_diff = abs(diff);

    // Smooth L1 Loss (Huber Loss):
    // loss = 0.5 * (diff^2) / beta,           if |diff| < beta
    // loss = |diff| - 0.5 * beta,             otherwise

    // beta == 0 degenerates to L1Loss (PyTorch semantics). The Huber formula
    // below divides by beta, so the beta == 0 case would otherwise yield NaN
    // (0/0). Fall back to |diff| directly.
    Variable loss_unreduced = [&]() -> Variable {
        if (beta_ == 0.0) {
            return abs_diff;
        }
        // Exact SmoothL1Loss (Huber loss) without branching:
        //   loss = 0.5 * min(|diff|, beta)^2 / beta + max(|diff| - beta, 0)
        // When |diff| < beta:  0.5 * |diff|^2 / beta + 0
        // When |diff| >= beta: 0.5 * beta + |diff| - beta = |diff| - 0.5 * beta
        auto clamped_abs = clamp(abs_diff, 0.0f, static_cast<float>(beta_));
        auto excess = abs_diff - clamped_abs;  // max(|diff| - beta, 0)
        return (clamped_abs * clamped_abs * 0.5f) / static_cast<float>(beta_) + excess;
    }();

    switch (reduction_) {
        case Reduction::None:
            return loss_unreduced;
        case Reduction::Mean:
            return mean(loss_unreduced);
        case Reduction::Sum:
            return sum(loss_unreduced);
        case Reduction::BatchMean: {
            const auto& shp = loss_unreduced.tensor().shape();
            int64_t bs = (!shp.empty()) ? shp[0] : 0;
            if (bs > 0) {
                return sum(loss_unreduced) / static_cast<float>(bs);
            }
            return mean(loss_unreduced);
        }
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
        case Reduction::BatchMean: {
            const auto& shp = loss.tensor().shape();
            int64_t bs = (!shp.empty()) ? shp[0] : 0;
            if (bs > 0) {
                return sum(loss) / static_cast<float>(bs);
            }
            return mean(loss);
        }
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
