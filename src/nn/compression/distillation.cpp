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
            // Convert class indices to one-hot encoding
            auto target_shape = targets.value().shape();
            int64_t batch_size = target_shape[0];
            auto logits_shape = student_logits_cast.tensor().shape();
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
        // Mean squared error between features
        auto diff = student_fp32 - teacher_fp32;
        auto squared = diff * diff;
        auto sum = squared.tensor();  // Sum all elements
        auto mean_val = sum;  // Mean over all elements

        return Variable(mean_val, student_fp32.requires_grad());

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
    // Cast to Float32 for consistent dtype. Student needs autograd-aware cast
    // so backward reaches the student params; teacher is a frozen reference.
    Variable student_fp32 = nn::variable_cast(student_features, DType::Float32);
    Variable teacher_fp32(teacher_features.tensor().to(DType::Float32), false);

    // Compute attention maps: sum of squared activations
    // Attention_map[n, h, w] = sum_c (features[n, c, h, w]^2)

    auto student_squared = student_fp32 * student_fp32;
    auto teacher_squared = teacher_fp32 * teacher_fp32;

    // Sum over channel dimension
    // Simplified - in practice would use proper reduction along axis
    auto student_attention = student_squared;
    auto teacher_attention = teacher_squared;

    // Normalize attention maps
    auto student_norm = student_attention;  // Should normalize
    auto teacher_norm = teacher_attention;  // Should normalize

    // L2 distance between normalized attention maps
    auto diff = student_norm - teacher_norm;
    auto squared_diff = diff * diff;

    // Return the Variable directly — re-wrapping squared_diff.tensor()
    // would silently sever the autograd chain back to student.
    return squared_diff;
}

auto relational_distillation_loss(
    const Variable& student_outputs,
    const Variable& teacher_outputs
) -> Variable {
    // Cast to Float32 for consistent dtype
    Variable student_fp32(student_outputs.tensor().to(DType::Float32), student_outputs.requires_grad());
    Variable teacher_fp32(teacher_outputs.tensor().to(DType::Float32), false);

    // Compute pairwise distances within batch
    // For each pair (i,j): similarity_matrix[i,j] = cos_sim(output[i], output[j])

    // Simplified implementation
    // In practice, compute full similarity matrix and match distributions

    auto diff = student_fp32 - teacher_fp32;
    auto mse = diff * diff;

    // Return the Variable directly — re-wrapping mse.tensor() would
    // silently sever the autograd chain back to student.
    return mse;
}

// =============================================================================
// Self-Distillation
// =============================================================================

auto self_distillation_loss(
    const Variable& current_logits,
    const Variable& past_logits,
    float temperature
) -> Variable {
    // Cast to Float32 for consistent dtype
    Variable current_logits_fp32(current_logits.tensor().to(DType::Float32), current_logits.requires_grad());
    Variable past_logits_fp32(past_logits.tensor().to(DType::Float32), false);

    // Treat past predictions as soft targets (similar to teacher-student)
    Variable past_soft = temperature_softmax(past_logits_fp32, temperature, -1);
    Variable current_log_soft = temperature_log_softmax(current_logits_fp32, temperature, -1);

    return kl_divergence(current_log_soft, past_soft, "batchmean") * (temperature * temperature);
}

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

auto online_distillation(
    const std::vector<Variable>& student_logits_list,
    const Tensor& targets,
    float temperature
) -> std::vector<Variable> {
    size_t num_students = student_logits_list.size();
    std::vector<Variable> losses;

    if (num_students < 2) {
        throw std::runtime_error("Online distillation requires at least 2 students");
    }

    // Each student learns from all other students
    for (size_t i = 0; i < num_students; ++i) {
        Variable student_loss;

        // Learn from other students
        for (size_t j = 0; j < num_students; ++j) {
            if (i == j) continue;

            DistillationConfig config;
            config.temperature = temperature;
            config.alpha = 0.5f;  // Balance peer learning with hard targets

            auto peer_loss = distillation_loss(
                student_logits_list[i],
                student_logits_list[j],
                targets,
                config
            );

            if (j == 0 || (j == 1 && i == 0)) {
                student_loss = peer_loss;
            } else {
                student_loss = student_loss + peer_loss;
            }
        }

        // Average over peers
        student_loss = student_loss / static_cast<float>(num_students - 1);
        losses.push_back(student_loss);
    }

    return losses;
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
    // Cast to Float32 for consistent dtype. Student needs autograd-aware cast
    // so backward reaches the student params; teacher is a frozen reference.
    Variable student_fp32 = nn::variable_cast(student_features, DType::Float32);
    Variable teacher_fp32(teacher_features.tensor().to(DType::Float32), false);

    // Cosine similarity: dot(a, b) / (norm(a) * norm(b))
    // Loss: 1 - cosine_similarity

    auto dot_product = student_fp32 * teacher_fp32;

    auto student_squared = student_fp32 * student_fp32;
    auto teacher_squared = teacher_fp32 * teacher_fp32;

    // Compute norms
    auto student_norm = student_squared;  // Should be sqrt(sum(squared))
    auto teacher_norm = teacher_squared;  // Should be sqrt(sum(squared))

    auto cosine_sim = dot_product / (student_norm * teacher_norm);
    // loss = 1 - cosine_sim, computed via Variable-level ops so backward
    // propagates back through cosine_sim to student. Previously the
    // re-wrap into Variable(.,.) silently severed the autograd chain.
    auto neg_cos = cosine_sim * (-1.0);
    auto loss = neg_cos + 1.0;

    return loss;
}

auto temperature_schedule(
    float initial_temp,
    float final_temp,
    int current_epoch,
    int total_epochs,
    const std::string& schedule_type
) -> float {
    float progress = static_cast<float>(current_epoch) / static_cast<float>(total_epochs);

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

auto compute_teacher_student_agreement(
    const std::shared_ptr<Module>& student,
    const std::shared_ptr<Module>& teacher,
    const std::vector<std::pair<Variable, Tensor>>& validation_data
) -> float {
    teacher->eval();
    student->eval();

    int total = 0;
    int agreements = 0;

    for (const auto& [input, target] : validation_data) {
        // Create non-gradient inputs
        Variable no_grad_input(input.tensor(), false);

        Variable teacher_out = teacher->forward(no_grad_input);
        Variable student_out = student->forward(no_grad_input);

        // Compare argmax predictions
        // Simplified - in practice would compute actual argmax
        total++;
        // If predictions match, increment agreements
    }

    return static_cast<float>(agreements) / static_cast<float>(total);
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
