#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
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
    auto predictions_clamped = clamp(input, 1e-7f, 1.0f - 1e-7f);

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
    // Simplify using: max(x, 0) = (x + abs(x)) / 2

    auto abs_input = abs(input);
    auto neg_abs = neg(abs_input);

    // Element-wise max(x, 0) = (x + abs(x)) / 2
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto two_tensor = full(shape_vec, 2.0f, input.dtype(), input.device());
    auto two_var = Variable(two_tensor, false);
    auto max_val = (input + abs_input) / two_var;

    auto xz = input * target;  // x * z
    auto ones_tensor = ones(shape_vec, input.dtype(), input.device());
    auto ones_var = Variable(ones_tensor, false);
    auto log_term = log(ones_var + exp(neg_abs));

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
    // Cross entropy with logits: -log_softmax(input)[target_class]
    // Use the log_softmax function from activations
    auto log_probs = nn::log_softmax(input, 1);  // Compute log_softmax along dim=1

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
        auto num_classes = input.tensor().shape()[1];

        // Move target to CPU for one-hot encoding (since we need to access it directly)
        auto target_cpu = target.device().type == Device::Type::CPU ? target : target.cpu();
        const int64_t* target_data = target_cpu.data<int64_t>();

        // Create one-hot tensor on CPU with proper dtype handling
        auto one_hot_cpu = tenzor::zeros({batch_size, num_classes}, input.tensor().dtype(), Device::cpu());

        if (input.tensor().dtype() == DType::Float32) {
            auto* one_hot_data = one_hot_cpu.data<float>();
            for (int64_t i = 0; i < batch_size; ++i) {
                int64_t class_idx = target_data[i];
                one_hot_data[i * num_classes + class_idx] = 1.0f;
            }
        } else if (input.tensor().dtype() == DType::Float64) {
            auto* one_hot_data = one_hot_cpu.data<double>();
            for (int64_t i = 0; i < batch_size; ++i) {
                int64_t class_idx = target_data[i];
                one_hot_data[i * num_classes + class_idx] = 1.0;
            }
        } else if (input.tensor().dtype() == DType::Float16) {
            // For Float16, work with float32 internally then convert
            auto one_hot_f32 = tenzor::zeros({batch_size, num_classes}, DType::Float32, Device::cpu());
            auto* one_hot_data = one_hot_f32.data<float>();
            for (int64_t i = 0; i < batch_size; ++i) {
                int64_t class_idx = target_data[i];
                one_hot_data[i * num_classes + class_idx] = 1.0f;
            }
            one_hot_cpu = one_hot_f32.to(DType::Float16);
        }

        // Move one_hot tensor to the same device as input
        auto one_hot = one_hot_cpu.device() == input.tensor().device() ? one_hot_cpu : one_hot_cpu.to(input.tensor().device());

        // Convert to Variable for autograd
        one_hot_var = Variable(one_hot, false);
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

    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto beta_tensor = full(shape_vec, beta_, input.dtype(), input.device());
    auto beta_var = Variable(beta_tensor, false);

    auto half_tensor = full(shape_vec, 0.5, input.dtype(), input.device());
    auto half_var = Variable(half_tensor, false);

    // Compute squared term: 0.5 * (diff^2) / beta
    auto squared_term = (half_var * diff * diff) / beta_var;

    // Compute linear term: |diff| - 0.5 * beta
    auto linear_term = abs_diff - (half_var * beta_var);

    // Select based on condition: |diff| < beta
    // mask = (abs_diff < beta) ? squared_term : linear_term
    // Approximate using: smooth transition with clamping
    auto beta_clamped_diff = clamp(abs_diff, 0.0f, static_cast<float>(beta_));
    auto is_quadratic = Variable(full(shape_vec, 1.0f, input.dtype(), input.device()), false) -
                        (abs_diff - beta_clamped_diff) / beta_var;

    // Weighted sum: quadratic region * squared_term + linear region * linear_term
    auto loss_unreduced = is_quadratic * squared_term + (Variable(full(shape_vec, 1.0f, input.dtype(), input.device()), false) - is_quadratic) * linear_term;

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
