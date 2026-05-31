/**
 * @file pruning.hpp
 * @brief Model pruning techniques for neural network compression
 *
 * Provides structured and unstructured pruning algorithms to reduce model size
 * and computational cost while maintaining accuracy. Supports magnitude-based
 * importance scoring, iterative pruning, and fine-tuning workflows.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include "../module.hpp"
#include "../layers/conv.hpp"
#include "../layers/linear.hpp"
#include "../optim/optimizer.hpp"
#include <cstdint>

namespace tenzor {
namespace nn {
namespace compression {

/**
 * @brief Importance criterion for weight pruning
 *
 * Determines which weights are least important and can be removed:
 * - L1: Sum of absolute values (magnitude-based)
 * - L2: L2 norm (Euclidean distance)
 * - L1Norm: L1 normalized by number of parameters
 * - L2Norm: L2 normalized by number of parameters
 */
enum class ImportanceCriterion {
    L1,        ///< L1 norm (sum of absolute values)
    L2,        ///< L2 norm (Euclidean distance)
    L1Norm,    ///< L1 norm divided by parameter count
    L2Norm     ///< L2 norm divided by parameter count
};

/**
 * @brief Pruning schedule for iterative pruning
 *
 * Controls how pruning progresses during training:
 * - OneShot: Prune once at specified sparsity
 * - Iterative: Gradually increase sparsity over epochs
 * - Polynomial: Polynomial decay schedule
 */
enum class PruningSchedule {
    OneShot,     ///< Single pruning step
    Iterative,   ///< Linear sparsity increase
    Polynomial   ///< Polynomial sparsity schedule
};

/**
 * @brief Binary mask for weight pruning
 *
 * Stores a binary mask indicating which weights are active (1) or pruned (0).
 * Applied during forward pass to enforce sparsity.
 */
struct PruningMask {
    Tensor mask;                    ///< Binary mask (1=keep, 0=prune)
    std::string layer_name;         ///< Name of layer this mask applies to
    float current_sparsity{0.0f};   ///< Current sparsity level [0, 1]

    /**
     * @brief Apply mask to weights
     * @param weights Weight tensor to mask
     * @return Masked weights (weights * mask)
     */
    auto apply(const Tensor& weights) const -> Tensor;

    /**
     * @brief Compute actual sparsity of mask
     * @return Fraction of zero elements [0, 1]
     */
    auto compute_sparsity() const -> float;
};

/**
 * @brief Pruning configuration and state
 *
 * Stores pruning parameters and maintains masks for all pruned layers.
 */
struct PruningConfig {
    float target_sparsity{0.5f};                        ///< Target sparsity level [0, 1]
    float current_sparsity{0.0f};                       ///< Current actual sparsity level [0, 1]
    ImportanceCriterion criterion{ImportanceCriterion::L1}; ///< Importance metric
    PruningSchedule schedule{PruningSchedule::OneShot};     ///< Pruning schedule
    int num_iterations{1};                              ///< Number of pruning iterations
    int current_iteration{0};                           ///< Current iteration
    std::unordered_map<std::string, PruningMask> masks; ///< Per-layer pruning masks

    /**
     * @brief Get current sparsity based on schedule
     * @return Current target sparsity for this iteration
     */
    auto get_current_sparsity() const -> float;
};

/**
 * @brief Compute importance scores for tensor weights
 *
 * Evaluates importance of weights using specified criterion.
 * Used to determine which weights to prune.
 *
 * @param weights Weight tensor to score
 * @param criterion Importance metric to use
 * @return Tensor of importance scores (same shape as weights)
 *
 * @code
 * Tensor scores = compute_importance(layer_weights, ImportanceCriterion::L1);
 * @endcode
 */
auto compute_importance(const Tensor& weights, ImportanceCriterion criterion) -> Tensor;

/**
 * @brief Create binary mask from importance scores
 *
 * Generates pruning mask by thresholding importance scores to achieve
 * target sparsity. Lowest-importance weights are set to 0.
 *
 * @param importance Importance scores for each weight
 * @param sparsity Target sparsity level [0, 1]
 * @return Binary mask (1=keep, 0=prune)
 *
 * @par Algorithm
 * 1. Sort importance scores
 * 2. Compute threshold at sparsity percentile
 * 3. Set mask[i] = (importance[i] > threshold)
 */
auto create_mask_from_importance(const Tensor& importance, float sparsity) -> Tensor;

// =============================================================================
// Unstructured Pruning
// =============================================================================

/**
 * @brief Apply unstructured (fine-grained) pruning to module
 *
 * Prunes individual weights based on magnitude, regardless of structure.
 * This creates irregular sparsity patterns that may not accelerate on
 * standard hardware without specialized kernels.
 *
 * **Algorithm:**
 * 1. Compute importance scores for all weights
 * 2. Sort weights globally or per-layer
 * 3. Zero out lowest-importance weights to reach target sparsity
 * 4. Create binary masks for pruned layers
 *
 * **Advantages:**
 * - Maximum flexibility in choosing which weights to remove
 * - Can achieve higher sparsity with less accuracy loss
 * - Works with any layer type
 *
 * **Disadvantages:**
 * - Irregular sparsity patterns
 * - Limited speedup on standard hardware
 * - Requires sparse matrix libraries for acceleration
 *
 * @param module Neural network module to prune
 * @param sparsity Target sparsity level [0, 1] (0.5 = 50% zeros)
 * @param criterion Importance metric for weight selection
 * @param global_pruning If true, compute threshold globally across all layers
 * @return PruningConfig with masks for all pruned layers
 *
 * @par Typical Usage
 * @code
 * // Prune 50% of weights using L1 magnitude
 * auto config = prune_unstructured(model, 0.5, ImportanceCriterion::L1);
 *
 * // Fine-tune model with masks applied
 * for (int epoch = 0; epoch < 10; ++epoch) {
 *     apply_pruning_masks(model, config);
 *     train_epoch(model, dataloader, optimizer);
 * }
 * @endcode
 *
 * @see prune_structured for hardware-friendly structured pruning
 */
auto prune_unstructured(
    std::shared_ptr<Module> module,
    float sparsity,
    ImportanceCriterion criterion = ImportanceCriterion::L1,
    bool global_pruning = false
) -> PruningConfig;

/**
 * @brief Iterative magnitude pruning with fine-tuning
 *
 * Gradually increases sparsity over multiple iterations with fine-tuning
 * between pruning steps. This approach often maintains accuracy better
 * than one-shot pruning.
 *
 * **Algorithm (based on "Lottery Ticket Hypothesis"):**
 * 1. Start with target_sparsity / num_iterations
 * 2. For each iteration:
 *    a. Prune to current sparsity level
 *    b. Fine-tune for fine_tune_epochs
 *    c. Increase sparsity
 * 3. Return final pruned model
 *
 * @param module Module to prune
 * @param target_sparsity Final target sparsity [0, 1]
 * @param num_iterations Number of pruning steps
 * @param schedule Sparsity increase schedule
 * @param criterion Importance metric
 * @return Final pruning configuration
 *
 * @par Example
 * @code
 * // Iteratively prune to 90% sparsity over 10 steps
 * auto config = prune_iterative(
 *     model,
 *     0.9,  // 90% sparsity
 *     10,   // 10 pruning iterations
 *     PruningSchedule::Polynomial,
 *     ImportanceCriterion::L2
 * );
 * @endcode
 *
 * @note Caller responsible for fine-tuning between iterations
 */
auto prune_iterative(
    std::shared_ptr<Module> module,
    float target_sparsity,
    int num_iterations,
    PruningSchedule schedule = PruningSchedule::Iterative,
    ImportanceCriterion criterion = ImportanceCriterion::L1
) -> PruningConfig;

// =============================================================================
// Structured Pruning
// =============================================================================

/**
 * @brief Apply structured channel pruning to convolutional layers
 *
 * Removes entire output channels from Conv2d layers based on importance.
 * This creates regular sparsity that directly reduces computation and
 * memory without requiring specialized hardware.
 *
 * **Channel Importance:**
 * Importance of channel c is typically computed as:
 * - L1: sum(abs(weights[:, c, :, :]))
 * - L2: sqrt(sum(weights[:, c, :, :]^2))
 *
 * **Propagation:**
 * When pruning output channel c of layer i:
 * - Layer i: Remove filters[c, :, :, :]
 * - Layer i+1: Remove weights[:, c, :, :] (input channel)
 * - BatchNorm i: Remove running_mean[c], running_var[c], weight[c], bias[c]
 *
 * **Advantages:**
 * - Direct speedup on standard hardware
 * - Regular sparsity pattern
 * - Reduces FLOPs and memory proportionally
 * - Compatible with standard convolution kernels
 *
 * @param module Module containing Conv2d layers
 * @param sparsity Fraction of channels to prune [0, 1]
 * @param criterion Channel importance metric
 * @return New module with channels physically removed
 *
 * @par Example
 * @code
 * // Prune 30% of channels from all Conv2d layers
 * auto pruned_model = prune_channels(model, 0.3, ImportanceCriterion::L1);
 *
 * // Model now has 30% fewer channels and runs 30% faster
 * @endcode
 *
 * @see prune_unstructured for fine-grained pruning
 */
auto prune_channels(
    std::shared_ptr<Module> module,
    float sparsity,
    ImportanceCriterion criterion = ImportanceCriterion::L1
) -> std::shared_ptr<Module>;

/**
 * @brief Prune entire filters from Conv2d layers
 *
 * Removes complete 3D filters (all channels and spatial dimensions).
 * Similar to channel pruning but operates on input channels.
 *
 * @param module Module to prune
 * @param sparsity Fraction of filters to remove
 * @param criterion Filter importance metric
 * @return Pruned module with filters removed
 */
auto prune_filters(
    std::shared_ptr<Module> module,
    float sparsity,
    ImportanceCriterion criterion = ImportanceCriterion::L1
) -> std::shared_ptr<Module>;

/**
 * @brief Remove entire layers from sequential models
 *
 * Prunes complete layers (and their skip connections if present).
 * Most aggressive form of structured pruning.
 *
 * **Layer Importance:**
 * Measured by:
 * - Average weight magnitude
 * - Activation statistics
 * - Gradient magnitude during training
 *
 * @param module Sequential module
 * @param num_layers Number of layers to remove
 * @param criterion Layer importance metric
 * @return Module with layers removed
 *
 * @note Experimental - may significantly impact accuracy
 */
auto prune_layers(
    std::shared_ptr<Module> module,
    int num_layers,
    ImportanceCriterion criterion = ImportanceCriterion::L1
) -> std::shared_ptr<Module>;

// =============================================================================
// Mask Management
// =============================================================================

/**
 * @brief Apply pruning masks to module parameters
 *
 * Multiplies each layer's weights by its pruning mask to enforce sparsity.
 * Call during training after optimizer step to maintain pruned structure.
 *
 * @param module Module with trainable parameters
 * @param config Pruning configuration containing masks
 *
 * @par Training Loop Integration
 * @code
 * for (int epoch = 0; epoch < epochs; ++epoch) {
 *     for (auto& batch : dataloader) {
 *         optimizer.zero_grad();
 *         auto output = model->forward(batch.input);
 *         auto loss = criterion(output, batch.target);
 *         loss.backward();
 *         optimizer.step();
 *
 *         // Re-apply masks after parameter update
 *         apply_pruning_masks(model, pruning_config);
 *     }
 * }
 * @endcode
 */
auto apply_pruning_masks(
    std::shared_ptr<Module> module,
    const PruningConfig& config
) -> void;

/**
 * @brief Register an optimizer post-step hook that re-applies pruning
 *        masks after every parameter update (audit G.10).
 *
 * Without this hook, the optimizer's gradient step silently restores
 * the zeroed-out positions to non-zero values (each step adds a small
 * delta to the masked positions).  PyTorch's `prune.global_unstructured`
 * + `prune.PruningContainer` machinery solves this by registering a
 * pre-forward hook on the parameter; the equivalent here is a post-
 * step hook on the optimizer that calls `apply_pruning_masks` after
 * each step.
 *
 * The returned handle can be passed to `optim::Optimizer::
 * remove_post_step_hook()` to deregister later (e.g. in
 * `finalize_pruning()` after the masks have been baked into the
 * weights).
 *
 * @param optimizer Optimizer instance to attach the hook to.
 * @param module Module whose state dict gets masked after each step.
 * @param config Pruning configuration containing the masks to apply.
 * @return Hook id (stable across other registrations / removals).
 */
auto register_pruning_auto_reapply(
    optim::Optimizer& optimizer,
    std::shared_ptr<Module> module,
    PruningConfig config
) -> uint64_t;

/**
 * @brief Make pruning permanent by removing zero weights
 *
 * Converts masked (zeroed) weights to actual sparse representation.
 * After this operation, masks are no longer needed.
 *
 * @param module Module with applied masks
 * @param config Pruning configuration
 * @return New module with physically removed weights
 *
 * @note This is irreversible - weights are permanently deleted
 */
auto finalize_pruning(
    std::shared_ptr<Module> module,
    const PruningConfig& config
) -> std::shared_ptr<Module>;

/**
 * @brief Remove all pruning masks and restore dense weights
 *
 * Useful for reverting pruning during experimentation.
 *
 * @param module Module with masks
 * @param config Pruning configuration to clear
 */
auto remove_pruning(
    std::shared_ptr<Module> module,
    PruningConfig& config
) -> void;

// =============================================================================
// Analysis and Utilities
// =============================================================================

/**
 * @brief Compute actual sparsity of module parameters
 *
 * Measures fraction of weights that are exactly zero.
 *
 * @param module Module to analyze
 * @return Sparsity level [0, 1]
 *
 * @code
 * float sparsity = compute_sparsity(model);
 * std::cout << "Model is " << (sparsity * 100) << "% sparse\n";
 * @endcode
 */
auto compute_sparsity(const std::shared_ptr<Module>& module) -> float;

/**
 * @brief Analyze per-layer sparsity
 *
 * @param module Module to analyze
 * @return Map of layer name to sparsity level
 *
 * @code
 * auto layer_sparsity = analyze_layer_sparsity(model);
 * for (auto& [name, sparsity] : layer_sparsity) {
 *     std::cout << name << ": " << (sparsity * 100) << "% sparse\n";
 * }
 * @endcode
 */
auto analyze_layer_sparsity(
    const std::shared_ptr<Module>& module
) -> std::unordered_map<std::string, float>;

/**
 * @brief Estimate compression ratio achieved by pruning
 *
 * @param original_module Original unpruned module
 * @param pruned_module Pruned module
 * @return Compression ratio (original_size / pruned_size)
 *
 * @code
 * float ratio = compute_compression_ratio(original_model, pruned_model);
 * std::cout << "Model compressed " << ratio << "x\n";
 * @endcode
 */
auto compute_compression_ratio(
    const std::shared_ptr<Module>& original_module,
    const std::shared_ptr<Module>& pruned_module
) -> float;

/**
 * @brief Estimate FLOPs reduction from structured pruning
 *
 * Computes approximate reduction in multiply-accumulate operations.
 * Only accurate for structured pruning (channels/filters/layers).
 *
 * @param module Pruned module
 * @param input_shape Example input shape for FLOP calculation
 * @return FLOPs reduction ratio [0, 1]
 */
auto estimate_flops_reduction(
    const std::shared_ptr<Module>& module,
    const std::vector<int64_t>& input_shape
) -> float;

/**
 * @brief Sensitivity analysis for layer-wise pruning
 *
 * Measures accuracy drop when pruning each layer individually.
 * Helps identify which layers are most sensitive to pruning.
 *
 * @param module Module to analyze
 * @param validation_fn Function that evaluates model accuracy
 * @param sparsity_levels Sparsity levels to test
 * @return Map of layer name to accuracy drops at each sparsity level
 */
auto sensitivity_analysis(
    std::shared_ptr<Module> module,
    std::function<float(std::shared_ptr<Module>)> validation_fn,
    const std::vector<float>& sparsity_levels = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f}
) -> std::unordered_map<std::string, std::vector<float>>;

// =============================================================================
// Lottery Ticket Hypothesis
// =============================================================================

/**
 * @brief Find winning lottery ticket (iterative magnitude pruning)
 *
 * Implements the Lottery Ticket Hypothesis algorithm:
 * 1. Train network to convergence
 * 2. Prune smallest magnitude weights
 * 3. Reset remaining weights to initialization values
 * 4. Repeat training
 *
 * **Reference:** "The Lottery Ticket Hypothesis" (Frankle & Carbin, 2019)
 *
 * @param module Module to train (will be reset to init weights)
 * @param initial_weights Initial weight values to reset to
 * @param target_sparsity Final sparsity to achieve
 * @param num_rounds Number of prune-retrain cycles
 * @return Final winning ticket mask
 *
 * @note Requires storing initial weights before training
 */
auto find_lottery_ticket(
    std::shared_ptr<Module> module,
    const std::unordered_map<std::string, Tensor>& initial_weights,
    float target_sparsity,
    int num_rounds
) -> PruningConfig;

} // namespace compression
} // namespace nn
} // namespace tenzor
