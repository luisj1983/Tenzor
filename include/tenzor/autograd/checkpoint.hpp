/**
 * @file checkpoint.hpp
 * @brief Gradient checkpointing for memory-efficient training
 *
 * Gradient checkpointing trades compute for memory by not saving intermediate
 * activations during forward pass, and recomputing them during backward pass.
 * This enables training of much larger models with limited GPU memory.
 *
 * Memory savings: 50-80% for deep models
 * Computational overhead: 20-33% (one extra forward pass)
 */

#pragma once

#include <functional>
#include <memory>
#include <vector>
#include "../core/tensor.hpp"
#include "variable.hpp"
#include "function.hpp"

namespace tenzor {
namespace autograd {

/**
 * @brief Checkpoint statistics for memory profiling
 */
struct CheckpointStats {
    size_t num_checkpoints{0};           ///< Total number of checkpoints created
    size_t num_recomputations{0};        ///< Total number of recomputations performed
    size_t saved_memory_bytes{0};        ///< Estimated memory saved (bytes)
    size_t peak_memory_bytes{0};         ///< Peak memory usage during execution
    double total_recompute_time_ms{0.0}; ///< Total time spent recomputing (milliseconds)
};

/**
 * @brief Get global checkpoint statistics
 *
 * @return Reference to global checkpoint statistics
 */
auto get_checkpoint_stats() -> CheckpointStats&;

/**
 * @brief Reset checkpoint statistics
 */
auto reset_checkpoint_stats() -> void;

/**
 * @brief Checkpointed function wrapper
 *
 * Wraps a function segment to support gradient checkpointing.
 * During forward pass, only input and output are saved.
 * During backward pass, forward is recomputed to get intermediate activations.
 *
 * Algorithm:
 * 1. Forward: Save only inputs, discard intermediate activations
 * 2. Backward: Recompute forward to regenerate intermediates
 * 3. Continue backward pass normally with recomputed activations
 *
 * @par Memory vs Compute Trade-off
 * - Memory: O(1) per checkpoint segment vs O(n) for standard autograd
 * - Compute: 1 extra forward pass per checkpoint segment
 *
 * @par Thread Safety
 * Not thread-safe. Each thread should use separate checkpoint contexts.
 *
 * @code
 * // Example: Checkpoint a transformer layer
 * auto layer = [&](const Variable& x) -> Variable {
 *     auto attn_out = attention_layer->forward(x);
 *     auto ffn_out = ffn_layer->forward(attn_out);
 *     return ffn_out;
 * };
 *
 * Variable input(tensor, true);
 * Variable output = checkpoint(layer, input);  // Only input/output saved
 * output.backward();  // Recomputes layer forward during backward
 * @endcode
 */
class CheckpointFunction : public Function {
public:
    /**
     * @brief Construct checkpoint function
     *
     * @param forward_fn Function to checkpoint (must be deterministic)
     * @param allow_caching Enable caching of recomputed activations (default: true)
     */
    explicit CheckpointFunction(
        std::function<std::vector<Variable>(const std::vector<Variable>&)> forward_fn,
        bool allow_caching = true
    );

    /**
     * @brief Forward pass (saves only inputs and outputs)
     *
     * Executes the wrapped function and saves minimal state.
     * Intermediate activations are discarded to save memory.
     *
     * @param inputs Input variables to checkpoint segment
     * @return Output variables from checkpoint segment
     */
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;

    /**
     * @brief Backward pass (recomputes forward to get intermediates)
     *
     * Recomputes forward pass to regenerate intermediate activations,
     * then performs backward pass through the recomputed graph.
     *
     * @param grad_outputs Gradients with respect to outputs
     * @return Gradients with respect to inputs
     */
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;

    /**
     * @brief Get memory saved by checkpointing (bytes)
     *
     * Estimates memory savings compared to standard backprop.
     *
     * @return Estimated memory savings in bytes
     */
    auto get_memory_savings() const -> size_t;

    /**
     * @brief Get number of recomputations performed
     *
     * @return Recomputation count for this checkpoint
     */
    auto get_recompute_count() const -> size_t { return recompute_count_; }

    /**
     * @brief Store heap-allocated copies of input Variables (internal)
     *
     * Used by checkpoint() free function to keep input Variables alive
     * so input_variables_ pointers remain valid.
     *
     * @param vars Input Variables to copy and store
     */
    auto store_input_copies(const std::vector<Variable>& vars) -> void {
        input_variable_copies_.clear();
        input_variable_copies_.reserve(vars.size());
        for (const auto& var : vars) {
            input_variable_copies_.push_back(std::make_unique<Variable>(var));
        }
    }

    /**
     * @brief Get pointers to stored input Variable copies (internal)
     *
     * @return Vector of pointers to heap-allocated input Variables
     */
    auto get_input_copy_pointers() -> std::vector<Variable*> {
        std::vector<Variable*> pointers;
        pointers.reserve(input_variable_copies_.size());
        for (auto& copy : input_variable_copies_) {
            pointers.push_back(copy.get());
        }
        return pointers;
    }

    /**
     * @brief Store copies of original leaf input Variables
     *
     * For leaf Variables, we need to accumulate gradients directly to the
     * original Variables passed to checkpoint(), not to copies.
     * Stores Variable copies (which share the underlying VariableImpl via
     * shared_ptr) instead of raw Variable* to prevent dangling pointers
     * if the caller's scope exits before backward.
     *
     * @param originals Pointers to original input Variables
     */
    auto store_original_inputs(const std::vector<Variable*>& originals) -> void {
        original_input_copies_.clear();
        original_input_copies_.reserve(originals.size());
        for (const auto* var : originals) {
            original_input_copies_.push_back(*var);
        }
    }

    /**
     * @brief Get stored original leaf inputs
     *
     * Returns references to the stored Variable copies, which share
     * the same VariableImpl as the originals.
     *
     * @return Vector of Variables sharing impls with the originals
     */
    auto get_original_inputs() const -> const std::vector<Variable>& {
        return original_input_copies_;
    }

private:
    std::function<std::vector<Variable>(const std::vector<Variable>&)> forward_fn_;
    bool allow_caching_;
    size_t recompute_count_{0};
    size_t estimated_activation_memory_{0};

    // Cached recomputed outputs (optional optimization)
    std::vector<Variable> cached_recompute_outputs_;
    bool has_cached_outputs_{false};

    // Heap-allocated copies of input Variables to keep them alive
    // so input_variables_ pointers remain valid
    std::vector<std::unique_ptr<Variable>> input_variable_copies_;

    // Copies of original leaf input Variables for gradient accumulation.
    // Variable copies share the underlying VariableImpl via shared_ptr,
    // preventing dangling pointers if the caller's Variables go out of scope.
    std::vector<Variable> original_input_copies_;

    // Recomputed input Variables - must stay alive during backward pass
    // to prevent dangling pointers in the recomputed graph
    std::vector<Variable> cached_recompute_inputs_;

    // Store copies of all Variables found during graph traversal
    // This prevents dangling pointers when accessing intermediate Variables
    // created during recomputation (e.g., temporaries in multi-input checkpoints)
    // Using Variable copies instead of shared_ptr to avoid no-op deleter issues
    std::vector<Variable> recomputed_intermediates_;

    /**
     * @brief Recompute forward pass with gradient tracking
     *
     * @param inputs Input variables
     * @return Recomputed output variables with gradient functions
     */
    auto recompute_forward(const std::vector<Variable>& inputs) -> std::vector<Variable>;

    /**
     * @brief Estimate activation memory for statistics
     *
     * @param vars Variables whose memory to estimate
     * @return Estimated memory usage in bytes
     */
    auto estimate_memory(const std::vector<Variable>& vars) const -> size_t;
};

/**
 * @brief Checkpoint a function segment
 *
 * Creates a checkpoint boundary for gradient computation.
 * Use this to wrap memory-intensive layer computations.
 *
 * @param fn Function to checkpoint (must be pure/deterministic)
 * @param inputs Input variables to the function
 * @return Output variables from the function
 *
 * @par Requirements
 * - fn must be deterministic (same inputs -> same outputs)
 * - fn should not have side effects
 * - fn should be computationally significant to justify overhead
 *
 * @par Best Practices
 * - Checkpoint large layer blocks (transformer layers, ResNet blocks)
 * - Don't checkpoint tiny operations (overhead > savings)
 * - Place checkpoints at natural boundaries (layer outputs)
 *
 * @code
 * // Example: Checkpoint transformer encoder layers
 * Variable x = embedding(input_ids);
 * for (int i = 0; i < num_layers; ++i) {
 *     // Checkpoint each layer to save memory
 *     x = checkpoint([&](const std::vector<Variable>& in) {
 *         auto attn = encoder_layers[i]->self_attention(in[0]);
 *         auto ffn = encoder_layers[i]->feed_forward(attn);
 *         return std::vector<Variable>{ffn};
 *     }, {x});
 * }
 * @endcode
 */
auto checkpoint(
    std::function<std::vector<Variable>(const std::vector<Variable>&)> fn,
    const std::vector<Variable>& inputs  // Changed to const reference to avoid unnecessary copy
) -> std::vector<Variable>;

/**
 * @brief Checkpoint with explicit original input pointers (for leaf variable support)
 *
 * Advanced version that allows explicit control over gradient accumulation targets.
 * Use this when you need gradients to accumulate to specific leaf Variables.
 *
 * @param fn Function to checkpoint
 * @param inputs Input variables (copies)
 * @param original_inputs Pointers to original Variables for gradient accumulation
 * @return Output variables
 */
auto checkpoint_with_originals(
    std::function<std::vector<Variable>(const std::vector<Variable>&)> fn,
    const std::vector<Variable>& inputs,  // Changed to const reference to avoid unnecessary copy
    const std::vector<Variable*>& original_inputs  // Also const reference for consistency
) -> std::vector<Variable>;

/**
 * @brief Checkpoint a single-input, single-output function (convenience)
 *
 * Simplified interface for common case of one input and one output.
 * IMPORTANT: For leaf variables, use the TENZOR_CHECKPOINT macro instead to ensure
 * gradients accumulate correctly.
 *
 * @param fn Function mapping Variable to Variable
 * @param input Input variable
 * @return Output variable
 *
 * @code
 * // Checkpoint a single layer
 * Variable hidden = checkpoint(
 *     [&](const Variable& x) { return layer->forward(x); },
 *     input
 * );
 * @endcode
 */
auto checkpoint(
    std::function<Variable(const Variable&)> fn,
    const Variable& input
) -> Variable;

/**
 * @brief Macro to checkpoint with automatic leaf variable handling
 *
 * Use this macro when checkpointing with leaf variables to ensure gradients
 * accumulate to the correct locations.
 *
 * @code
 * Variable x(tensor, true);  // Leaf variable
 * Variable y = TENZOR_CHECKPOINT(
 *     [](const Variable& in) { return in * 2; },
 *     x
 * );
 * @endcode
 */
#define TENZOR_CHECKPOINT(fn, input) \
    ::tenzor::autograd::checkpoint_with_original((fn), (input), &(input))

/**
 * @brief Checkpoint single-input function with explicit original pointer
 *
 * Advanced version for explicit control over gradient accumulation target.
 *
 * @param fn Function to checkpoint
 * @param input Input variable (copy)
 * @param original_input Pointer to original Variable for gradient accumulation
 * @return Output variable
 */
auto checkpoint_with_original(
    std::function<Variable(const Variable&)> fn,
    const Variable& input,
    Variable* original_input
) -> Variable;

/**
 * @brief Checkpoint context manager
 *
 * RAII wrapper for checkpoint regions. Automatically manages
 * checkpoint state and statistics collection.
 *
 * @code
 * {
 *     CheckpointContext ctx;
 *     for (int i = 0; i < num_layers; ++i) {
 *         x = checkpoint(layers[i], x);
 *     }
 *     // Statistics available via ctx.get_stats()
 * }
 * @endcode
 */
class CheckpointContext {
public:
    /**
     * @brief Construct checkpoint context
     *
     * @param enabled Enable checkpointing (default: true)
     */
    explicit CheckpointContext(bool enabled = true);

    /**
     * @brief Destructor (logs statistics if enabled)
     */
    ~CheckpointContext();

    // Non-copyable, non-movable
    CheckpointContext(const CheckpointContext&) = delete;
    CheckpointContext& operator=(const CheckpointContext&) = delete;

    /**
     * @brief Check if checkpointing is enabled in this context
     *
     * @return true if checkpointing is active
     */
    auto is_enabled() const -> bool { return enabled_; }

    /**
     * @brief Get statistics for this context
     *
     * @return Checkpoint statistics accumulated in this context
     */
    auto get_stats() const -> CheckpointStats;

private:
    bool enabled_;
    bool prev_enabled_;
    CheckpointStats initial_stats_;
};

/**
 * @brief Memory tracker for checkpoint optimization
 *
 * Tracks memory usage during checkpointed execution to help
 * optimize checkpoint placement.
 */
class MemoryTracker {
public:
    /**
     * @brief Start memory tracking
     */
    static auto start_tracking() -> void;

    /**
     * @brief Stop memory tracking
     */
    static auto stop_tracking() -> void;

    /**
     * @brief Get current memory usage
     *
     * @return Current memory usage in bytes
     */
    static auto current_memory() -> size_t;

    /**
     * @brief Get peak memory usage since start
     *
     * @return Peak memory usage in bytes
     */
    static auto peak_memory() -> size_t;

    /**
     * @brief Reset memory tracking
     */
    static auto reset() -> void;

private:
    static thread_local bool tracking_enabled_;
    static thread_local size_t current_memory_;
    static thread_local size_t peak_memory_;
};

/**
 * @brief Checkpoint segment for nested checkpointing
 *
 * Supports hierarchical checkpointing with multiple nesting levels.
 * Useful for very deep models where even checkpointed segments need
 * further checkpointing.
 *
 * @code
 * // Two-level checkpointing for very deep models
 * auto outer_fn = [&](const Variable& x) {
 *     Variable y = x;
 *     for (int i = 0; i < 10; ++i) {
 *         // Inner checkpoint for each layer
 *         y = checkpoint([&](const Variable& in) {
 *             return layer[i]->forward(in);
 *         }, y);
 *     }
 *     return y;
 * };
 *
 * // Outer checkpoint for whole block
 * Variable output = checkpoint(outer_fn, input);
 * @endcode
 */
class CheckpointSegment {
public:
    /**
     * @brief Create checkpoint segment
     *
     * @param name Segment name for debugging
     * @param nesting_level Nesting depth (0 = top level)
     */
    CheckpointSegment(std::string name, int nesting_level = 0);

    /**
     * @brief Execute function within checkpoint segment
     *
     * @param fn Function to execute
     * @param inputs Input variables
     * @return Output variables
     */
    auto execute(
        std::function<std::vector<Variable>(const std::vector<Variable>&)> fn,
        std::vector<Variable> inputs
    ) -> std::vector<Variable>;

    /**
     * @brief Get segment name
     *
     * @return Segment identifier
     */
    auto name() const -> const std::string& { return name_; }

    /**
     * @brief Get nesting level
     *
     * @return Nesting depth
     */
    auto nesting_level() const -> int { return nesting_level_; }

private:
    std::string name_;
    int nesting_level_;
};

/**
 * @brief Check if gradient checkpointing is globally enabled
 *
 * @return true if checkpointing is active
 */
auto is_checkpoint_enabled() -> bool;

/**
 * @brief Set global checkpoint enabled state
 *
 * @param enabled Whether to enable checkpointing
 */
auto set_checkpoint_enabled(bool enabled) -> void;

} // namespace autograd
} // namespace tenzor
