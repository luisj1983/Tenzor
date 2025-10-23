/**
 * @file distillation.hpp
 * @brief Knowledge distillation for model compression and transfer learning
 *
 * Implements knowledge distillation techniques where a smaller "student" model
 * learns to mimic a larger "teacher" model. Uses soft targets and temperature
 * scaling to transfer knowledge more effectively than training on hard labels alone.
 */

#pragma once

#include <memory>
#include <optional>
#include <functional>
#include "../module.hpp"
#include "../loss/losses.hpp"
#include "../../autograd/variable.hpp"

namespace tenzor {
namespace nn {
namespace compression {

/**
 * @brief Knowledge distillation configuration
 *
 * Controls how knowledge is transferred from teacher to student:
 * - **Temperature (T):** Softens probability distributions
 *   - T=1: Standard softmax (hard targets)
 *   - T>1: Softer distributions (reveals relative confidences)
 *   - Typical: 3-10 for classification
 *
 * - **Alpha (α):** Balances soft and hard losses
 *   - α=1: Only soft targets (pure distillation)
 *   - α=0: Only hard targets (standard training)
 *   - Typical: 0.5-0.9 for best results
 */
struct DistillationConfig {
    float temperature{3.0f};        ///< Temperature for softmax scaling
    float alpha{0.7f};              ///< Weight for soft target loss [0,1]
    bool use_hard_targets{true};    ///< Include hard target loss
    bool normalize_temperature{true}; ///< Normalize loss by T^2
};

/**
 * @brief Temperature-scaled softmax
 *
 * Applies softmax with temperature scaling:
 * \f[
 * \text{softmax}_T(x_i) = \frac{e^{x_i/T}}{\sum_j e^{x_j/T}}
 * \f]
 *
 * **Effect of Temperature:**
 * - T → 0: Approaches one-hot (argmax)
 * - T = 1: Standard softmax
 * - T → ∞: Approaches uniform distribution
 *
 * Higher temperature reveals relative confidences between classes,
 * providing richer learning signal for student.
 *
 * @param logits Raw model outputs (before softmax)
 * @param temperature Temperature parameter T > 0
 * @param dim Dimension to apply softmax (default: -1)
 * @return Temperature-scaled probabilities
 *
 * @code
 * Variable logits = model->forward(input);
 * Variable soft_probs = temperature_softmax(logits, 5.0f);
 * @endcode
 */
auto temperature_softmax(
    const Variable& logits,
    float temperature,
    int64_t dim = -1
) -> Variable;

/**
 * @brief Temperature-scaled log-softmax
 *
 * Numerically stable log(softmax(x/T)) computation.
 * Used internally for KL divergence calculation.
 *
 * @param logits Raw model outputs
 * @param temperature Temperature parameter
 * @param dim Softmax dimension
 * @return Log of temperature-scaled probabilities
 */
auto temperature_log_softmax(
    const Variable& logits,
    float temperature,
    int64_t dim = -1
) -> Variable;

/**
 * @brief Distillation loss combining soft and hard targets
 *
 * Computes weighted combination of:
 * 1. **Soft target loss:** KL divergence between student and teacher
 * 2. **Hard target loss:** Cross-entropy with true labels
 *
 * **Mathematical Formulation:**
 * \f[
 * L = \alpha \cdot T^2 \cdot KL(q_T || p_T) + (1-\alpha) \cdot CE(p, y)
 * \f]
 *
 * where:
 * - \f$q_T\f$ = teacher outputs at temperature T
 * - \f$p_T\f$ = student outputs at temperature T
 * - y = true labels
 * - T^2 factor normalizes gradient magnitudes
 *
 * **Intuition:**
 * - Soft targets teach relative confidences (dark knowledge)
 * - Hard targets ensure correct class selection
 * - Alpha balances exploration vs exploitation
 *
 * @param student_logits Student model raw outputs
 * @param teacher_logits Teacher model raw outputs
 * @param targets True class labels (optional if alpha=1.0)
 * @param config Distillation configuration
 * @return Combined distillation loss
 *
 * @code
 * Variable student_out = student->forward(input);
 * Variable teacher_out = teacher->forward(input);
 * Variable loss = distillation_loss(student_out, teacher_out, targets, config);
 * @endcode
 */
auto distillation_loss(
    const Variable& student_logits,
    const Variable& teacher_logits,
    const std::optional<Tensor>& targets,
    const DistillationConfig& config
) -> Variable;

/**
 * @brief Knowledge Distillation trainer
 *
 * Manages the distillation training process with teacher-student models.
 * Handles forward passes, loss computation, and metric tracking.
 *
 * **Typical Workflow:**
 * 1. Load pre-trained teacher model
 * 2. Initialize smaller student model
 * 3. Create KnowledgeDistillation instance
 * 4. Train student using distillation_loss
 * 5. Fine-tune student on hard targets (optional)
 *
 * **Example Use Cases:**
 * - ResNet-50 → MobileNetV2 (efficient mobile deployment)
 * - BERT-Large → DistilBERT (6x faster inference)
 * - Teacher ensemble → Single student (model compression)
 *
 * @code
 * // Setup
 * auto teacher = load_pretrained_model("resnet50.pth");
 * teacher->eval();  // Freeze teacher
 *
 * auto student = std::make_shared<MobileNetV2>();
 * auto distiller = KnowledgeDistillation(teacher, student);
 *
 * // Training
 * for (auto& batch : train_loader) {
 *     optimizer.zero_grad();
 *     auto loss = distiller.compute_loss(batch.input, batch.target);
 *     loss.backward();
 *     optimizer.step();
 * }
 * @endcode
 */
class KnowledgeDistillation {
public:
    /**
     * @brief Construct distillation trainer
     *
     * @param teacher Pre-trained teacher model (will be set to eval mode)
     * @param student Student model to train
     * @param config Distillation hyperparameters
     *
     * @note Teacher gradients are automatically disabled
     */
    KnowledgeDistillation(
        std::shared_ptr<Module> teacher,
        std::shared_ptr<Module> student,
        DistillationConfig config = DistillationConfig{}
    );

    /**
     * @brief Compute distillation loss for a batch
     *
     * Performs forward pass through both teacher and student,
     * then computes combined distillation loss.
     *
     * @param input Batch of input data
     * @param targets True class labels (optional if alpha=1.0)
     * @return Distillation loss (ready for backward)
     *
     * @par Thread Safety
     * Not thread-safe. Use separate instances for parallel training.
     */
    auto compute_loss(
        const Variable& input,
        const std::optional<Tensor>& targets = std::nullopt
    ) -> Variable;

    /**
     * @brief Forward pass through both models
     *
     * @param input Input batch
     * @return Pair of (student_output, teacher_output)
     */
    auto forward(const Variable& input) -> std::pair<Variable, Variable>;

    /**
     * @brief Get teacher model
     * @return Shared pointer to teacher
     */
    auto teacher() const -> std::shared_ptr<Module> { return teacher_; }

    /**
     * @brief Get student model
     * @return Shared pointer to student
     */
    auto student() const -> std::shared_ptr<Module> { return student_; }

    /**
     * @brief Get configuration
     * @return Distillation config
     */
    auto config() const -> const DistillationConfig& { return config_; }

    /**
     * @brief Update configuration
     * @param config New distillation config
     */
    auto set_config(const DistillationConfig& config) -> void { config_ = config; }

    /**
     * @brief Set temperature
     * @param temperature New temperature value
     */
    auto set_temperature(float temperature) -> void { config_.temperature = temperature; }

    /**
     * @brief Set alpha (soft/hard loss balance)
     * @param alpha New alpha value [0, 1]
     */
    auto set_alpha(float alpha) -> void { config_.alpha = alpha; }

private:
    std::shared_ptr<Module> teacher_;  ///< Teacher model (frozen)
    std::shared_ptr<Module> student_;  ///< Student model (trainable)
    DistillationConfig config_;        ///< Distillation parameters
};

// =============================================================================
// Feature Distillation
// =============================================================================

/**
 * @brief Feature-based distillation loss
 *
 * Matches intermediate feature representations between teacher and student.
 * Often more effective than output distillation for vision tasks.
 *
 * **Algorithm:**
 * 1. Extract features from teacher layer
 * 2. Extract features from student layer (may need projection)
 * 3. Compute similarity loss (MSE, cosine, attention transfer)
 *
 * **Common Approaches:**
 * - FitNets: MSE between feature maps
 * - Attention Transfer: Matching attention maps
 * - FSP: Flow of Solution Procedure
 *
 * @param student_features Student intermediate features
 * @param teacher_features Teacher intermediate features
 * @param loss_type Type of feature matching ("mse", "cosine", "attention")
 * @return Feature distillation loss
 *
 * @code
 * // Extract features from specific layers
 * Variable student_feat = student->get_layer("conv4")->output;
 * Variable teacher_feat = teacher->get_layer("conv4")->output;
 * Variable feat_loss = feature_distillation_loss(student_feat, teacher_feat, "mse");
 * @endcode
 */
auto feature_distillation_loss(
    const Variable& student_features,
    const Variable& teacher_features,
    const std::string& loss_type = "mse"
) -> Variable;

/**
 * @brief Attention transfer loss
 *
 * Matches attention maps (spatial importance) between teacher and student.
 * Attention map = sum of squared activations per spatial location.
 *
 * **Reference:** "Paying More Attention to Attention" (Zagoruyko & Komodakis, 2017)
 *
 * @param student_features Student feature maps (N, C, H, W)
 * @param teacher_features Teacher feature maps (N, C', H', W')
 * @return Attention transfer loss
 *
 * @code
 * Variable loss = attention_transfer_loss(student_conv_out, teacher_conv_out);
 * @endcode
 */
auto attention_transfer_loss(
    const Variable& student_features,
    const Variable& teacher_features
) -> Variable;

/**
 * @brief Relational knowledge distillation
 *
 * Transfers knowledge about relationships between samples in a batch.
 * Computes pairwise similarities and matches them between teacher and student.
 *
 * **Reference:** "Relational Knowledge Distillation" (Park et al., 2019)
 *
 * @param student_outputs Student outputs for batch (N, D)
 * @param teacher_outputs Teacher outputs for batch (N, D)
 * @return Relational distillation loss
 */
auto relational_distillation_loss(
    const Variable& student_outputs,
    const Variable& teacher_outputs
) -> Variable;

// =============================================================================
// Self-Distillation
// =============================================================================

/**
 * @brief Self-distillation loss (student learns from itself)
 *
 * Uses model's own predictions as soft targets. Surprisingly effective
 * for regularization and ensemble-like behavior.
 *
 * **Approaches:**
 * - Temporal ensemble: Average predictions over time
 * - Deep supervision: Early layers learn from final layer
 * - Label smoothing: Soften one-hot labels
 *
 * @param current_logits Current model outputs
 * @param past_logits Previous outputs (from EMA model or earlier epoch)
 * @param temperature Temperature for softmax
 * @return Self-distillation loss
 *
 * @code
 * // Maintain EMA of model for temporal ensemble
 * Variable current_out = model->forward(input);
 * Variable ema_out = ema_model->forward(input);
 * Variable loss = self_distillation_loss(current_out, ema_out, 2.0f);
 * @endcode
 */
auto self_distillation_loss(
    const Variable& current_logits,
    const Variable& past_logits,
    float temperature
) -> Variable;

// =============================================================================
// Advanced Distillation Techniques
// =============================================================================

/**
 * @brief Multi-teacher distillation
 *
 * Distill knowledge from multiple teacher models (ensemble distillation).
 * Student learns from averaged or weighted teacher predictions.
 *
 * @param student_logits Student outputs
 * @param teacher_logits_list List of teacher outputs
 * @param targets True labels
 * @param config Distillation config
 * @param teacher_weights Weights for each teacher (optional, uniform if nullopt)
 * @return Combined distillation loss from all teachers
 *
 * @code
 * std::vector<Variable> teacher_outputs = {
 *     teacher1->forward(input),
 *     teacher2->forward(input),
 *     teacher3->forward(input)
 * };
 * Variable loss = multi_teacher_distillation(
 *     student->forward(input),
 *     teacher_outputs,
 *     targets,
 *     config
 * );
 * @endcode
 */
auto multi_teacher_distillation(
    const Variable& student_logits,
    const std::vector<Variable>& teacher_logits_list,
    const std::optional<Tensor>& targets,
    const DistillationConfig& config,
    const std::optional<std::vector<float>>& teacher_weights = std::nullopt
) -> Variable;

/**
 * @brief Online distillation (mutual learning)
 *
 * Multiple student models learn from each other simultaneously.
 * No pre-trained teacher required - students co-evolve.
 *
 * **Reference:** "Deep Mutual Learning" (Zhang et al., 2018)
 *
 * @param student_logits_list Outputs from all student models
 * @param targets True labels
 * @param temperature Temperature for soft targets
 * @return Vector of losses (one per student)
 *
 * @code
 * std::vector<Variable> outputs = {
 *     student1->forward(input),
 *     student2->forward(input),
 *     student3->forward(input)
 * };
 * auto losses = online_distillation(outputs, targets, 3.0f);
 * auto total_loss = losses[0] + losses[1] + losses[2];
 * @endcode
 */
auto online_distillation(
    const std::vector<Variable>& student_logits_list,
    const Tensor& targets,
    float temperature
) -> std::vector<Variable>;

// =============================================================================
// Utilities
// =============================================================================

/**
 * @brief Compute KL divergence between distributions
 *
 * KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log P - log Q))
 *
 * @param log_predictions Log-probabilities from student (log Q)
 * @param targets Probabilities from teacher (P)
 * @param reduction Reduction mode ("mean", "sum", "batchmean", "none")
 * @return KL divergence loss
 */
auto kl_divergence(
    const Variable& log_predictions,
    const Variable& targets,
    const std::string& reduction = "batchmean"
) -> Variable;

/**
 * @brief Compute cosine similarity between feature maps
 *
 * Used for feature-based distillation.
 *
 * @param student_features Student feature tensors
 * @param teacher_features Teacher feature tensors
 * @return Cosine similarity loss (1 - cosine_sim)
 */
auto cosine_similarity_loss(
    const Variable& student_features,
    const Variable& teacher_features
) -> Variable;

/**
 * @brief Apply temperature annealing schedule
 *
 * Gradually decrease temperature during training for smoother convergence.
 *
 * @param initial_temp Starting temperature
 * @param final_temp Ending temperature
 * @param current_epoch Current training epoch
 * @param total_epochs Total number of epochs
 * @param schedule_type Annealing schedule ("linear", "exponential", "cosine")
 * @return Temperature for current epoch
 *
 * @code
 * // Start with soft targets (T=10), end with harder targets (T=3)
 * float temp = temperature_schedule(10.0f, 3.0f, epoch, 100, "cosine");
 * distiller.set_temperature(temp);
 * @endcode
 */
auto temperature_schedule(
    float initial_temp,
    float final_temp,
    int current_epoch,
    int total_epochs,
    const std::string& schedule_type = "linear"
) -> float;

/**
 * @brief Compute model similarity score
 *
 * Measures how similar student and teacher predictions are.
 * Useful for monitoring distillation progress.
 *
 * @param student Student model
 * @param teacher Teacher model
 * @param validation_data Validation dataset
 * @return Agreement percentage [0, 1]
 *
 * @code
 * float agreement = compute_teacher_student_agreement(student, teacher, val_data);
 * std::cout << "Student agrees with teacher " << (agreement * 100) << "% of time\n";
 * @endcode
 */
auto compute_teacher_student_agreement(
    const std::shared_ptr<Module>& student,
    const std::shared_ptr<Module>& teacher,
    const std::vector<std::pair<Variable, Tensor>>& validation_data
) -> float;

/**
 * @brief Estimate compression ratio from teacher to student
 *
 * @param teacher Teacher module
 * @param student Student module
 * @return Compression ratio (teacher_params / student_params)
 */
auto compute_distillation_compression_ratio(
    const std::shared_ptr<Module>& teacher,
    const std::shared_ptr<Module>& student
) -> float;

// =============================================================================
// Pre-configured Distillation Strategies
// =============================================================================

/**
 * @brief Create configuration for classification distillation
 *
 * @param temperature Temperature for softmax (default: 3.0)
 * @param alpha Soft loss weight (default: 0.7)
 * @return Optimized config for classification
 */
auto make_classification_distillation_config(
    float temperature = 3.0f,
    float alpha = 0.7f
) -> DistillationConfig;

/**
 * @brief Create configuration for detection distillation
 *
 * @param temperature Temperature (default: 2.0)
 * @param alpha Soft loss weight (default: 0.5)
 * @return Optimized config for object detection
 */
auto make_detection_distillation_config(
    float temperature = 2.0f,
    float alpha = 0.5f
) -> DistillationConfig;

/**
 * @brief Create configuration for semantic segmentation distillation
 *
 * @param temperature Temperature (default: 1.5)
 * @param alpha Soft loss weight (default: 0.8)
 * @return Optimized config for segmentation
 */
auto make_segmentation_distillation_config(
    float temperature = 1.5f,
    float alpha = 0.8f
) -> DistillationConfig;

} // namespace compression
} // namespace nn
} // namespace tenzor
