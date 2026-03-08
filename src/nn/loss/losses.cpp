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
    }
    return loss_unreduced;
}

// CrossEntropyLoss implementation
CrossEntropyLoss::CrossEntropyLoss(Reduction reduction, float label_smoothing)
    : reduction_(reduction), label_smoothing_(label_smoothing) {}

auto CrossEntropyLoss::forward(const Variable& input, const Tensor& target) -> Variable {
    // Cross entropy with logits: -log_softmax(input)[target_class]
    // Use the log_softmax function from activations
    auto log_probs = nn::log_softmax(input, 1);  // Compute log_softmax along dim=1

    auto num_classes = input.tensor().shape()[1];

    // Handle both one-hot encoded targets (Float32/Float64, shape [N, C]) and class indices (Int64, shape [N])
    Variable one_hot_var;

    // Check if target is a floating point type (one-hot encoded)
    bool is_float_target = (target.dtype() == DType::Float32 || target.dtype() == DType::Float64 || target.dtype() == DType::Float16) && target.ndim() == 2;

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
        auto batch_size = input.tensor().shape()[0];

        // Use backend dispatch for one-hot encoding (avoids GPU→CPU→GPU round-trip)
        NewOpAttributes oh_attrs;
        oh_attrs.set(AttrKey::NumClasses, num_classes);

        // Ensure target is on same device as input for dispatch
        Tensor target_dev = (target.device() == input.tensor().device())
            ? target : target.to(input.tensor().device());
        std::vector<Tensor> oh_inputs = {target_dev};
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

    switch (reduction_) {
        case Reduction::None:
            return neg_selected;
        case Reduction::Mean:
            return mean(neg_selected);
        case Reduction::Sum:
            return sum(neg_selected);
    }
    return neg_selected;
}

// NLLLoss implementation
NLLLoss::NLLLoss(Reduction reduction) : reduction_(reduction) {}

auto NLLLoss::forward(const Variable& input, const Tensor& target) -> Variable {
    // Negative log likelihood
    // Assumes input is already log probabilities
    // Convert target to match input dtype
    auto target_float = target.to(input.tensor().dtype());
    auto target_var = Variable(target_float, false);
    auto weighted = target_var * input;
    auto loss_per_sample = sum(weighted, 1, false);
    auto neg_loss = neg(loss_per_sample);

    switch (reduction_) {
        case Reduction::None:
            return neg_loss;
        case Reduction::Mean:
            return mean(neg_loss);
        case Reduction::Sum:
            return sum(neg_loss);
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
                  Reduction reduction) -> Variable {
    CrossEntropyLoss loss(reduction);
    return loss.forward(input, target);
}

auto bce_loss(const Variable& input, const Variable& target,
             Reduction reduction) -> Variable {
    BCELoss loss(reduction);
    return loss.forward(input, target);
}

auto nll_loss(const Variable& input, const Tensor& target,
             Reduction reduction) -> Variable {
    NLLLoss loss(reduction);
    return loss.forward(input, target);
}

auto l1_loss(const Variable& input, const Variable& target,
            Reduction reduction) -> Variable {
    L1Loss loss(reduction);
    return loss.forward(input, target);
}

} // namespace tenzor::nn
