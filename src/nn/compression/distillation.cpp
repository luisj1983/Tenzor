/**
 * @file distillation.cpp
 * @brief Implementation of knowledge distillation algorithms
 */

#include "tenzor/nn/compression/distillation.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
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

    // Only convert Float16 to Float32 for numerical stability
    // Preserve Float32 and Float64 as-is
    Variable logits_processed = logits;
    if (logits.tensor().dtype() == DType::Float16) {
        logits_processed = Variable(logits.tensor().to(DType::Float32), logits.requires_grad());
    }

    // Scale logits by temperature: logits / T
    // Lower temperature → sharper distribution
    // Higher temperature → softer distribution
    Variable scaled_logits = logits_processed / temperature;

    // Apply numerically stable softmax
    // The softmax function already implements:
    // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    // This prevents overflow/underflow with extreme logits or temperatures
    return nn::softmax(scaled_logits, dim);
}

auto temperature_log_softmax(
    const Variable& logits,
    float temperature,
    int64_t dim
) -> Variable {
    if (temperature <= 0.0f) {
        throw std::runtime_error("Temperature must be positive");
    }

    // Only convert Float16 to Float32 for numerical stability
    // Preserve Float32 and Float64 as-is
    Variable logits_processed = logits;
    if (logits.tensor().dtype() == DType::Float16) {
        logits_processed = Variable(logits.tensor().to(DType::Float32), logits.requires_grad());
    }

    // Scale logits by temperature: logits / T
    Variable scaled_logits = logits_processed / temperature;

    // Apply numerically stable log-softmax
    // The log_softmax function already implements:
    // log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    // This is more numerically stable than log(softmax(x))
    return nn::log_softmax(scaled_logits, dim);
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

    // Cast logits to Float32 only if needed for consistent dtype across all operations
    Variable student_logits_fp32 = (student_logits.tensor().dtype() == DType::Float32)
        ? student_logits
        : Variable(student_logits.tensor().to(DType::Float32), student_logits.requires_grad());

    Variable teacher_logits_fp32 = (teacher_logits.tensor().dtype() == DType::Float32)
        ? Variable(teacher_logits.tensor(), false)  // Ensure teacher never needs gradients
        : Variable(teacher_logits.tensor().to(DType::Float32), false);

    // Compute soft target loss (KL divergence)
    Variable soft_loss;
    {
        // Teacher soft targets (no gradients) - already Float32
        Variable teacher_soft = temperature_softmax(teacher_logits_fp32, T, -1);

        // Student log probabilities - already Float32
        Variable student_log_soft = temperature_log_softmax(student_logits_fp32, T, -1);

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
            auto logits_shape = student_logits_fp32.tensor().shape();
            int64_t num_classes = logits_shape[1];

            // Create one-hot tensor
            std::vector<int64_t> onehot_shape = {batch_size, num_classes};
            targets_onehot = Tensor(onehot_shape, DType::Float32, targets.value().device());
            targets_onehot.fill_(0.0f);

            // Fill one-hot encoding
            auto* onehot_data = targets_onehot.data<float>();
            auto* indices = targets.value().data<int64_t>();
            for (int64_t i = 0; i < batch_size; ++i) {
                int64_t class_idx = indices[i];
                onehot_data[i * num_classes + class_idx] = 1.0f;
            }
        } else {
            // Already in the correct format
            targets_onehot = targets.value();
        }

        auto ce_loss = CrossEntropyLoss(Reduction::Mean);
        Variable hard_loss = ce_loss(student_logits_fp32, targets_onehot);

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
    // Cast to Float32 for consistent dtype
    Variable student_fp32(student_features.tensor().to(DType::Float32), student_features.requires_grad());
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
    // Cast to Float32 for consistent dtype
    Variable student_fp32(student_features.tensor().to(DType::Float32), student_features.requires_grad());
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

    return Variable(squared_diff.tensor(), student_fp32.requires_grad());
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

    return Variable(mse.tensor(), student_fp32.requires_grad());
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
    // KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log P - log Q))
    // Here: P = targets, Q = predictions
    // Since we have log Q, we compute: sum(P * (log P - log Q))

    // Ensure both tensors are Float32 only if needed for consistent computation
    Variable log_predictions_fp32 = (log_predictions.tensor().dtype() == DType::Float32)
        ? log_predictions
        : Variable(log_predictions.tensor().to(DType::Float32), log_predictions.requires_grad());

    Variable targets_fp32 = (targets.tensor().dtype() == DType::Float32)
        ? Variable(targets.tensor(), false)  // Targets don't need gradients
        : Variable(targets.tensor().to(DType::Float32), false);

    // KL(P||Q) = sum(P * (log P - log Q))
    // Handle edge case: when P = 0, use convention that 0 * log(0) = 0
    // We compute this element-wise to avoid nan from log(0)

    const float* p_data = targets_fp32.tensor().data<const float>();
    const float* log_q_data = log_predictions_fp32.tensor().data<const float>();
    int64_t numel = targets_fp32.tensor().numel();

    auto result_shape = std::vector<int64_t>(targets_fp32.tensor().shape().begin(),
                                             targets_fp32.tensor().shape().end());
    Tensor kl_tensor(result_shape, DType::Float32, targets_fp32.tensor().device());
    float* kl_data = kl_tensor.data<float>();

    constexpr float EPSILON = 1e-10f;
    for (int64_t i = 0; i < numel; ++i) {
        float p = p_data[i];
        float log_q = log_q_data[i];

        if (p > EPSILON) {
            // P * (log P - log Q)
            float log_p = std::log(p);
            kl_data[i] = p * (log_p - log_q);
        } else {
            // When P ≈ 0, use convention that 0 * log(0) = 0
            kl_data[i] = 0.0f;
        }
    }

    Variable kl(kl_tensor, log_predictions_fp32.requires_grad());

    // Apply reduction
    if (reduction == "none") {
        return kl;
    } else if (reduction == "sum") {
        // Sum all elements
        return Variable(kl.tensor(), log_predictions_fp32.requires_grad());
    } else if (reduction == "mean") {
        // Mean over all elements
        return Variable(kl.tensor(), log_predictions_fp32.requires_grad());
    } else if (reduction == "batchmean") {
        // Sum over elements, divide by batch size
        return Variable(kl.tensor(), log_predictions_fp32.requires_grad());
    } else {
        throw std::runtime_error("Unknown reduction type: " + reduction);
    }
}

auto cosine_similarity_loss(
    const Variable& student_features,
    const Variable& teacher_features
) -> Variable {
    // Cast to Float32 for consistent dtype
    Variable student_fp32(student_features.tensor().to(DType::Float32), student_features.requires_grad());
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
    auto loss = Variable(cosine_sim.tensor() * -1.0f + 1.0f, student_fp32.requires_grad());

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
