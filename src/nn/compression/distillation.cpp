/**
 * @file distillation.cpp
 * @brief Implementation of knowledge distillation algorithms
 */

#include "tenzor/nn/compression/distillation.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/nn/utils/variable_cast.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/autograd/ops.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tenzor {
namespace nn {
namespace compression {

// =============================================================================
// Temperature-Scaled Softmax
// =============================================================================

auto temperature_softmax(
    const Variable& logits,
    float temperature,
    int64_t dim
) -> Variable {
    if (temperature <= 0.0f) {
        throw std::runtime_error("Temperature must be positive");
    }

    // Only widen Float16 / BFloat16 to Float32 for numerical stability —
    // Float64 is already more precise than Float32 and widening to Float32
    // would drop precision below test tolerances.
    //
    // Previously the raw `Variable(...tensor().to(Float32), rg)` pattern
    // severed backward to `logits`; `variable_cast` wires a TypeCastBackward
    // node so gradients flow back through the cast.
    const DType lo_dtype = logits.tensor().dtype();
    const bool widen = (lo_dtype == DType::Float16 || lo_dtype == DType::BFloat16);
    Variable logits_processed = widen
        ? nn::variable_cast(logits, DType::Float32)
        : logits;

    // Scale logits by temperature: logits / T
    // Lower temperature → sharper distribution
    // Higher temperature → softer distribution
    Variable scaled_logits = logits_processed / temperature;

    // Apply numerically stable softmax
    // The softmax function already implements:
    // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    // This prevents overflow/underflow with extreme logits or temperatures
    Variable result = nn::softmax(scaled_logits, dim);

    // Cast back to the original dtype when we widened for stability.
    if (widen) {
        result = nn::variable_cast(result, lo_dtype);
    }
    return result;
}

auto temperature_log_softmax(
    const Variable& logits,
    float temperature,
    int64_t dim
) -> Variable {
    if (temperature <= 0.0f) {
        throw std::runtime_error("Temperature must be positive");
    }

    // Only widen Float16 / BFloat16 to Float32 for numerical stability —
    // Float64 is already more precise than Float32. See temperature_softmax
    // for the autograd-sever history this fix addresses.
    const DType lo_dtype = logits.tensor().dtype();
    const bool widen = (lo_dtype == DType::Float16 || lo_dtype == DType::BFloat16);
    Variable logits_processed = widen
        ? nn::variable_cast(logits, DType::Float32)
        : logits;

    // Scale logits by temperature: logits / T
    Variable scaled_logits = logits_processed / temperature;

    // Apply numerically stable log-softmax
    // The log_softmax function already implements:
    // log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    // This is more numerically stable than log(softmax(x))
    Variable result = nn::log_softmax(scaled_logits, dim);

    // Cast back to original dtype when we widened for stability.
    if (widen) {
        result = nn::variable_cast(result, lo_dtype);
    }
    return result;
}

// =============================================================================
// Distillation Loss
// =============================================================================

auto distillation_loss(
    const Variable& student_logits,
    const Variable& teacher_logits,
    const std::optional<Tensor>& targets,
    const DistillationConfig& config
) -> Variable {
    float T = config.temperature;
    float alpha = config.alpha;

    // Only cast Float16 to Float32 for numerical stability.
    // Student side: autograd-aware cast so backward reaches the student params.
    // Teacher side: no-grad wrap (teacher is a frozen reference).
    Variable student_logits_cast = nn::variable_cast(student_logits, DType::Float32);

    auto teacher_dtype = teacher_logits.tensor().dtype();
    Variable teacher_logits_cast = (teacher_dtype == DType::Float16)
        ? Variable(teacher_logits.tensor().to(DType::Float32), false)
        : Variable(teacher_logits.tensor(), false);  // Teacher never needs gradients

    // Compute soft target loss (KL divergence)
    Variable soft_loss;
    {
        // Teacher soft targets (no gradients) - already Float32
        Variable teacher_soft = temperature_softmax(teacher_logits_cast, T, -1);

        // Student log probabilities - already Float32
        Variable student_log_soft = temperature_log_softmax(student_logits_cast, T, -1);

        // KL divergence: sum(P * log(P/Q)) = sum(P * (log P - log Q))
        // Here we use: -sum(P * log Q) since we ignore teacher's entropy
        soft_loss = kl_divergence(student_log_soft, teacher_soft, "batchmean");

        // Scale by T^2 to account for gradient magnitude
        if (config.normalize_temperature) {
            soft_loss = soft_loss * (T * T);
        }
    }

    // Compute hard target loss if enabled
    if (config.use_hard_targets && targets.has_value()) {
        // Standard cross-entropy with true labels (uses Float32 logits)
        // CrossEntropyLoss expects one-hot encoded targets, so convert class indices to one-hot
        Tensor targets_onehot;
        if (targets.value().dtype() == DType::Int64) {
            // Convert class indices to one-hot encoding.
            //
            // This path only supports the classic classification layout:
            // rank-1 class-index targets (batch,) and rank-2 logits
            // (batch, num_classes). Dense/segmentation distillation (targets
            // (N,H,W), logits (N,C,H,W)) would read num_classes from the wrong
            // dimension and produce a wrong-shaped one-hot, so reject it
            // explicitly rather than silently mis-encoding.
            auto target_shape = targets.value().shape();
            auto logits_shape = student_logits_cast.tensor().shape();
            if (target_shape.size() != 1 || logits_shape.size() != 2) {
                throw std::invalid_argument(
                    "distillation_loss: hard-target one-hot conversion requires "
                    "rank-1 Int64 class-index targets (batch,) and rank-2 logits "
                    "(batch, num_classes); got targets rank " +
                    std::to_string(target_shape.size()) + " and logits rank " +
                    std::to_string(logits_shape.size()) +
                    ". For dense/segmentation distillation provide float one-hot "
                    "targets instead.");
            }
            int64_t batch_size = target_shape[0];
            int64_t num_classes = logits_shape[1];

            // Move targets to CPU for indexing, create one-hot on CPU, then transfer
            Tensor targets_cpu = targets.value();
            if (targets.value().device() != Device::cpu()) {
                targets_cpu = targets.value().to(Device::cpu());
            }

            // Create one-hot tensor on CPU first
            std::vector<int64_t> onehot_shape = {batch_size, num_classes};
            Tensor onehot_cpu(onehot_shape, DType::Float32, Device::cpu());
            onehot_cpu.fill_(0.0f);

            // Fill one-hot encoding on CPU
            auto* onehot_data = onehot_cpu.data<float>();
            auto* indices = targets_cpu.data<int64_t>();
            for (int64_t i = 0; i < batch_size; ++i) {
                int64_t class_idx = indices[i];
                if (class_idx < 0 || class_idx >= num_classes) {
                    throw std::out_of_range(
                        "distillation_loss: class index " +
                        std::to_string(class_idx) + " out of range [0, " +
                        std::to_string(num_classes) + ")");
                }
                onehot_data[i * num_classes + class_idx] = 1.0f;
            }

            // Transfer to original device if needed
            targets_onehot = (targets.value().device() == Device::cpu())
                ? onehot_cpu
                : onehot_cpu.to(targets.value().device());
        } else {
            // Already in the correct format
            targets_onehot = targets.value();
        }

        auto ce_loss = CrossEntropyLoss(Reduction::Mean);
        Variable hard_loss = ce_loss(student_logits_cast, targets_onehot);

        // Combine losses: alpha * soft + (1 - alpha) * hard
        return soft_loss * alpha + hard_loss * (1.0f - alpha);
    }

    // Only soft loss (pure distillation)
    return soft_loss;
}

// =============================================================================
// KnowledgeDistillation Class
// =============================================================================

KnowledgeDistillation::KnowledgeDistillation(
    std::shared_ptr<Module> teacher,
    std::shared_ptr<Module> student,
    DistillationConfig config
) : teacher_(teacher),
    student_(student),
    config_(config)
{
    // Set teacher to evaluation mode and disable gradients
    teacher_->eval();

    // In practice, we'd also want to freeze teacher parameters
    // This would require setting requires_grad=false on all teacher params
}

auto KnowledgeDistillation::compute_loss(
    const Variable& input,
    const std::optional<Tensor>& targets
) -> Variable {
    // Teacher forward pass (no gradients)
    // Create a non-gradient version of input for teacher
    Variable teacher_input(input.tensor(), false);  // requires_grad=false
    Variable teacher_output = teacher_->forward(teacher_input);

    // Student forward pass (with gradients)
    Variable student_output = student_->forward(input);

    // Compute distillation loss
    return distillation_loss(student_output, teacher_output, targets, config_);
}

auto KnowledgeDistillation::forward(const Variable& input)
    -> std::pair<Variable, Variable> {
    // Teacher forward pass (no gradients)
    Variable teacher_input(input.tensor(), false);  // requires_grad=false
    Variable teacher_output = teacher_->forward(teacher_input);

    // Student forward pass (with gradients)
    Variable student_output = student_->forward(input);

    return {student_output, teacher_output};
}

// =============================================================================
// Feature Distillation
// =============================================================================

auto feature_distillation_loss(
    const Variable& student_features,
    const Variable& teacher_features,
    const std::string& loss_type
) -> Variable {
    // Cast to Float32 for consistent dtype. Student needs autograd-aware cast
    // so backward reaches the student params; teacher is a frozen reference.
    Variable student_fp32 = nn::variable_cast(student_features, DType::Float32);
    Variable teacher_fp32(teacher_features.tensor().to(DType::Float32), false);

    if (loss_type == "mse") {
        // Mean squared error between features, reduced to a scalar via the
        // autograd-aware mean() so the gradient flows back to the student.
        // (Previously this read squared.tensor() and returned it unreduced as
        // a fresh leaf Variable, which both skipped the reduction and severed
        // the autograd graph.)
        auto diff = student_fp32 - teacher_fp32;
        auto squared = diff * diff;
        return mean(squared);

    } else if (loss_type == "cosine") {
        return cosine_similarity_loss(student_fp32, teacher_fp32);

    } else if (loss_type == "attention") {
        return attention_transfer_loss(student_fp32, teacher_fp32);

    } else {
        throw std::runtime_error("Unknown feature distillation loss type: " + loss_type);
    }
}

auto attention_transfer_loss(
    const Variable& student_features,
    const Variable& teacher_features
) -> Variable {
    // Attention Transfer (Zagoruyko & Komodakis, ICLR 2017,
    // "Paying More Attention to Attention"). Given activations of shape
    // (N, C, H, W):
    //   1. Spatial attention map  F(A)_{n,h,w} = sum_c A_{n,c,h,w}^2
    //      (sum over the channel dimension -> (N, H, W)).
    //   2. Flatten each sample to a vector of length H*W and L2-normalise it:
    //      Q = vec(F) / ||vec(F)||_2.
    //   3. Loss = mean over the batch of the squared L2 distance between the
    //      normalised student and teacher attention vectors.
    //
    // Everything is expressed with autograd-aware Variable ops so the gradient
    // flows back to the student parameters (the teacher branch is detached via
    // the requires_grad=false cast).
    //
    // Cast to Float32 for consistent dtype. Student needs an autograd-aware
    // cast so backward reaches the student params; teacher is a frozen ref.
    Variable student_fp32 = nn::variable_cast(student_features, DType::Float32);
    Variable teacher_fp32(teacher_features.tensor().to(DType::Float32), false);

    const auto& in_shape = student_fp32.tensor().shape();
    const int64_t rank = static_cast<int64_t>(in_shape.size());
    if (rank < 2) {
        throw std::runtime_error(
            "attention_transfer_loss: expected features of rank >= 2 "
            "(N, C, ...), got rank " + std::to_string(rank));
    }
    const int64_t batch = in_shape[0];
    int64_t spatial = 1;
    for (int64_t d = 2; d < rank; ++d) spatial *= in_shape[d];

    // Channel-reduced spatial attention map: sum over dim=1 (the channel axis).
    // autograd::sum reduces a single dim at a time; channel is dim 1.
    auto student_attn = sum(student_fp32 * student_fp32, /*dim=*/1, /*keepdim=*/false);
    auto teacher_attn = sum(teacher_fp32 * teacher_fp32, /*dim=*/1, /*keepdim=*/false);

    // Collapse the (N, <spatial dims>) attention map to (N, spatial).
    auto student_flat = reshape(student_attn, std::vector<int64_t>{batch, spatial});
    auto teacher_flat = reshape(teacher_attn, std::vector<int64_t>{batch, spatial});

    // Per-sample L2 normalisation: Q = F / (||F||_2 + eps). eps guards the
    // all-zero-activation case so the division stays finite and the gradient
    // does not blow up.
    constexpr double EPS = 1e-12;
    auto student_l2 = sqrt(sum(student_flat * student_flat, /*dim=*/1, /*keepdim=*/true));
    auto teacher_l2 = sqrt(sum(teacher_flat * teacher_flat, /*dim=*/1, /*keepdim=*/true));
    auto student_norm = student_flat / (student_l2 + EPS);
    auto teacher_norm = teacher_flat / (teacher_l2 + EPS);

    // Mean over the batch of the squared L2 distance between normalised maps.
    auto diff = student_norm - teacher_norm;
    auto squared_diff = diff * diff;
    auto per_sample = sum(squared_diff, /*dim=*/1, /*keepdim=*/false);
    return mean(per_sample);
}



// =============================================================================
// Self-Distillation
// =============================================================================



// =============================================================================
// Advanced Techniques
// =============================================================================

auto multi_teacher_distillation(
    const Variable& student_logits,
    const std::vector<Variable>& teacher_logits_list,
    const std::optional<Tensor>& targets,
    const DistillationConfig& config,
    const std::optional<std::vector<float>>& teacher_weights
) -> Variable {
    if (teacher_logits_list.empty()) {
        throw std::runtime_error("At least one teacher required");
    }

    // Compute weights (uniform if not provided)
    std::vector<float> weights;
    if (teacher_weights.has_value()) {
        weights = teacher_weights.value();
        // The loop below indexes weights[0..teacher_logits_list.size()-1]; a
        // shorter weights vector would read out of bounds.
        if (weights.size() != teacher_logits_list.size()) {
            throw std::invalid_argument(
                "multi_teacher_distillation: teacher_weights size (" +
                std::to_string(weights.size()) +
                ") must match the number of teachers (" +
                std::to_string(teacher_logits_list.size()) + ")");
        }
    } else {
        weights.resize(teacher_logits_list.size(), 1.0f / teacher_logits_list.size());
    }

    // Cast first teacher to Float32 and average teacher predictions
    Variable avg_teacher(teacher_logits_list[0].tensor().to(DType::Float32), false);
    avg_teacher = avg_teacher * weights[0];

    for (size_t i = 1; i < teacher_logits_list.size(); ++i) {
        // Cast each teacher to Float32 before adding
        Variable teacher_fp32(teacher_logits_list[i].tensor().to(DType::Float32), false);
        avg_teacher = avg_teacher + (teacher_fp32 * weights[i]);
    }

    // Standard distillation from averaged teacher (will cast student internally)
    return distillation_loss(student_logits, avg_teacher, targets, config);
}



// =============================================================================
// Utilities
// =============================================================================

auto kl_divergence(
    const Variable& log_predictions,
    const Variable& targets,
    const std::string& reduction
) -> Variable {
    // KL(P||Q) element-wise = P * (log P − log Q).
    //
    // Previous implementation computed the entire KL sum host-side in a
    // manual loop over raw tensor data, then wrapped the result in
    // `Variable(kl_tensor, log_predictions_cast.requires_grad())` — a fresh
    // leaf Variable with NO grad_fn. Backward flowed nothing back to
    // log_predictions, silently zeroing student gradients through the
    // distillation soft-target loss. Rewritten to use Variable-level ops
    // throughout so autograd reaches the student params.
    //
    // p = 0 safety: clamp p to [eps, 1] before log so log(p) never diverges.
    // The `p * (log(p) - log_q)` term then becomes `0 * (log(eps) - log_q) = 0`
    // exactly where p = 0, matching the original semantics. For very small
    // p ∈ (0, eps], the error is bounded by p * |log(eps) - log(p)| which is
    // < 1e-10 * 23 ≈ 2e-9 — well below float precision.

    Variable log_q = nn::variable_cast(log_predictions, DType::Float32);

    auto target_dtype = targets.tensor().dtype();
    Variable p = (target_dtype == DType::Float16)
        ? Variable(targets.tensor().to(DType::Float32), false)
        : Variable(targets.tensor(), false);

    constexpr float EPSILON = 1e-10f;
    auto p_safe = ::tenzor::clamp(p, EPSILON, 1.0f);   // autograd-aware (p has no grad)
    auto log_p = ::tenzor::log(p_safe);                // autograd-aware
    auto per_element = p * (log_p - log_q);            // autograd flows through log_q
    // Clamp element-wise negatives (fp-precision noise) to 0. relu is
    // autograd-aware and piecewise-linear; grad passes through positive
    // elements and zero through the clamped ones.
    auto kl = nn::relu(per_element);

    // Apply reduction
    if (reduction == "none") {
        return kl;
    } else if (reduction == "sum") {
        // Sum all elements
        return sum(kl);
    } else if (reduction == "mean") {
        // Mean over all elements
        return mean(kl);
    } else if (reduction == "batchmean") {
        // Sum over elements, divide by batch size
        auto total = sum(kl);
        int64_t batch_size = p.tensor().shape()[0];
        return total / static_cast<float>(batch_size);
    } else {
        throw std::runtime_error("Unknown reduction type: " + reduction);
    }
}

auto cosine_similarity_loss(
    const Variable& student_features,
    const Variable& teacher_features
) -> Variable {
    // Per-sample cosine-similarity distillation loss:
    //   cos_n = <s_n, t_n> / (||s_n||_2 ||t_n||_2)
    //   loss  = mean_n (1 - cos_n)
    // The features are flattened to (N, D) so the similarity is taken over the
    // full per-sample feature vector. All ops are autograd-aware so the
    // gradient reaches the student params (teacher is a frozen reference).
    //
    // Cast to Float32 for consistent dtype. Student needs an autograd-aware
    // cast so backward reaches the student params; teacher is a frozen ref.
    Variable student_fp32 = nn::variable_cast(student_features, DType::Float32);
    Variable teacher_fp32(teacher_features.tensor().to(DType::Float32), false);

    const auto& in_shape = student_fp32.tensor().shape();
    const int64_t rank = static_cast<int64_t>(in_shape.size());
    if (rank < 1) {
        throw std::runtime_error(
            "cosine_similarity_loss: expected features of rank >= 1");
    }
    const int64_t batch = in_shape[0];
    int64_t feat = 1;
    for (int64_t d = 1; d < rank; ++d) feat *= in_shape[d];

    auto student_flat = reshape(student_fp32, std::vector<int64_t>{batch, feat});
    auto teacher_flat = reshape(teacher_fp32, std::vector<int64_t>{batch, feat});

    // Per-sample dot product and L2 norms (reduce over the feature dim=1).
    constexpr double EPS = 1e-12;
    auto dot = sum(student_flat * teacher_flat, /*dim=*/1, /*keepdim=*/false);
    auto student_l2 = sqrt(sum(student_flat * student_flat, /*dim=*/1, /*keepdim=*/false));
    auto teacher_l2 = sqrt(sum(teacher_flat * teacher_flat, /*dim=*/1, /*keepdim=*/false));

    auto cosine_sim = dot / ((student_l2 * teacher_l2) + EPS);
    // loss = mean over batch of (1 - cosine_sim).
    auto per_sample = (cosine_sim * (-1.0)) + 1.0;
    return mean(per_sample);
}

auto temperature_schedule(
    float initial_temp,
    float final_temp,
    int current_epoch,
    int total_epochs,
    const std::string& schedule_type
) -> float {
    if (total_epochs <= 0) {
        throw std::invalid_argument(
            "temperature_schedule: total_epochs must be positive");
    }
    float progress = static_cast<float>(current_epoch) / static_cast<float>(total_epochs);
    // Clamp to [0, 1] so out-of-range epochs cannot extrapolate the schedule.
    progress = std::max(0.0f, std::min(1.0f, progress));

    if (schedule_type == "linear") {
        return initial_temp + (final_temp - initial_temp) * progress;

    } else if (schedule_type == "exponential") {
        float decay = std::pow(final_temp / initial_temp, progress);
        return initial_temp * decay;

    } else if (schedule_type == "cosine") {
        float cos_val = std::cos(progress * M_PI);
        return final_temp + 0.5f * (initial_temp - final_temp) * (1.0f + cos_val);

    } else {
        throw std::runtime_error("Unknown schedule type: " + schedule_type);
    }
}



auto compute_distillation_compression_ratio(
    const std::shared_ptr<Module>& teacher,
    const std::shared_ptr<Module>& student
) -> float {
    auto teacher_params = teacher->parameters();
    auto student_params = student->parameters();

    int64_t teacher_count = 0;
    int64_t student_count = 0;

    for (auto& param : teacher_params) {
        teacher_count += param->tensor().numel();
    }

    for (auto& param : student_params) {
        student_count += param->tensor().numel();
    }

    if (student_count == 0) return INFINITY;
    return static_cast<float>(teacher_count) / static_cast<float>(student_count);
}

// =============================================================================
// Pre-configured Strategies
// =============================================================================

auto make_classification_distillation_config(
    float temperature,
    float alpha
) -> DistillationConfig {
    DistillationConfig config;
    config.temperature = temperature;
    config.alpha = alpha;
    config.use_hard_targets = true;
    config.normalize_temperature = true;
    return config;
}

auto make_detection_distillation_config(
    float temperature,
    float alpha
) -> DistillationConfig {
    DistillationConfig config;
    config.temperature = temperature;
    config.alpha = alpha;
    config.use_hard_targets = true;
    config.normalize_temperature = true;
    return config;
}

auto make_segmentation_distillation_config(
    float temperature,
    float alpha
) -> DistillationConfig {
    DistillationConfig config;
    config.temperature = temperature;
    config.alpha = alpha;
    config.use_hard_targets = true;
    config.normalize_temperature = false;  // Different for dense predictions
    return config;
}

} // namespace compression
} // namespace nn
} // namespace tenzor
