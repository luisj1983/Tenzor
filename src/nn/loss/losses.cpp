#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/ops.hpp"

namespace tenzor::nn {

// Helper functions for operations that don't have autograd yet
namespace {
    auto scalar_sub(float scalar, const Variable& var) -> Variable {
        auto shape = var.shape();
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        auto scalar_tensor = full(shape_vec, scalar, var.dtype(), var.device());
        return Variable(scalar_tensor, false) - var;
    }

    auto variable_abs(const Variable& var) -> Variable {
        // TODO: Implement AbsBackward for proper autograd
        return Variable(tenzor::abs(var.tensor()), var.requires_grad());
    }

    auto variable_clamp(const Variable& var, float min, float max) -> Variable {
        // TODO: Implement ClampBackward for proper autograd
        return Variable(tenzor::clamp(var.tensor(), min, max), var.requires_grad());
    }

    auto variable_max(const Variable& var, int64_t dim, bool keepdim) -> Variable {
        // TODO: Implement MaxBackward for proper autograd
        return Variable(tenzor::max(var.tensor(), dim, keepdim), var.requires_grad());
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
    // Clamp predictions to avoid log(0)
    auto predictions_clamped = variable_clamp(input, 1e-7f, 1.0f - 1e-7f);

    // loss = -[target * log(pred) + (1 - target) * log(1 - pred)]
    auto term1 = target * log(predictions_clamped);
    auto term2 = scalar_sub(1.0f, target) * log(scalar_sub(1.0f, predictions_clamped));
    auto loss_unreduced = term1 + term2;
    auto neg_loss = neg(loss_unreduced);

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

// BCEWithLogitsLoss implementation
BCEWithLogitsLoss::BCEWithLogitsLoss(Reduction reduction) : reduction_(reduction) {}

auto BCEWithLogitsLoss::forward(const Variable& input, const Variable& target) -> Variable {
    // Use log-sum-exp trick for numerical stability
    // BCE = max(x, 0) - x * z + log(1 + exp(-abs(x)))
    // where x = input, z = target

    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto zeros_tensor = zeros(shape_vec, input.dtype(), input.device());
    auto zeros_var = Variable(zeros_tensor, false);

    auto abs_input = variable_abs(input);
    auto neg_abs = neg(abs_input);

    auto max_val = variable_max(input, 0, false);  // max(x, 0)
    auto xz = input * target;  // x * z
    auto log_term = log(scalar_sub(1.0f, zeros_var) + exp(neg_abs));

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
CrossEntropyLoss::CrossEntropyLoss(Reduction reduction) : reduction_(reduction) {}

auto CrossEntropyLoss::forward(const Variable& input, const Tensor& target) -> Variable {
    // Cross entropy with logits: -sum(target * log_softmax(input))
    // Use the log_softmax function from activations
    auto log_probs = nn::log_softmax(input, 1);  // Compute log_softmax along dim=1

    // Multiply with targets and sum along class dimension
    auto target_var = Variable(target, false);
    auto weighted = target_var * log_probs;
    auto loss_per_sample = sum(weighted, 1, false);

    // Negate to get positive loss
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

// NLLLoss implementation
NLLLoss::NLLLoss(Reduction reduction) : reduction_(reduction) {}

auto NLLLoss::forward(const Variable& input, const Tensor& target) -> Variable {
    // Negative log likelihood
    // Assumes input is already log probabilities
    auto target_var = Variable(target, false);
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
    auto abs_diff = variable_abs(diff);

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
    auto abs_diff = variable_abs(diff);

    // For now, implement a simple version
    // TODO: Properly implement smooth L1 with beta parameter
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
