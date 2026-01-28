/**
 * @file checkpoint.cpp
 * @brief Complete implementation of gradient checkpointing
 */

#include "../../include/tenzor/autograd/checkpoint.hpp"
#include "../../include/tenzor/ops/creation.hpp"
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include <functional>

namespace tenzor {
namespace autograd {

// ============================================================================
// Global Statistics
// ============================================================================

// Thread-local storage for checkpoint statistics
thread_local CheckpointStats global_checkpoint_stats;
thread_local bool checkpoint_enabled = true;

auto get_checkpoint_stats() -> CheckpointStats& {
    return global_checkpoint_stats;
}

auto reset_checkpoint_stats() -> void {
    global_checkpoint_stats = CheckpointStats{};
}

auto is_checkpoint_enabled() -> bool {
    return checkpoint_enabled;
}

auto set_checkpoint_enabled(bool enabled) -> void {
    checkpoint_enabled = enabled;
}

// ============================================================================
// CheckpointFunction Implementation
// ============================================================================

CheckpointFunction::CheckpointFunction(
    std::function<std::vector<Variable>(const std::vector<Variable>&)> forward_fn,
    bool allow_caching
) : forward_fn_(std::move(forward_fn)),
    allow_caching_(allow_caching),
    recompute_count_(0),
    estimated_activation_memory_(0),
    has_cached_outputs_(false) {}

auto CheckpointFunction::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    // NOTE: This function is NOT used by the checkpoint() free function.
    // It's here for completeness but should not be called directly because
    // `inputs` are passed by value, making input_variables_ pointers invalid.
    // The checkpoint() free function sets up the graph manually instead.
    throw std::runtime_error("CheckpointFunction::forward should not be called directly. Use checkpoint() free function.");
}

auto CheckpointFunction::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    if (!is_checkpoint_enabled()) {
        // Return zero gradients if checkpointing is disabled
        std::vector<Tensor> zero_grads;
        zero_grads.reserve(saved_tensors().size());
        for (const auto& tensor : saved_tensors()) {
            zero_grads.push_back(Tensor::zeros_like(tensor));
        }
        return zero_grads;
    }

    // Time the recomputation
    auto start_time = std::chrono::high_resolution_clock::now();

    // Create fresh Variables with gradient tracking enabled for recomputation
    // These form an independent autograd graph that can handle nested checkpoints
    cached_recompute_inputs_.clear();
    cached_recompute_inputs_.reserve(saved_tensors().size());
    for (const auto& tensor : saved_tensors()) {
        cached_recompute_inputs_.emplace_back(tensor, true);
    }

    // Recompute forward pass - nested checkpoints work naturally in this fresh graph
    auto recomputed_outputs = recompute_forward(cached_recompute_inputs_);

    // Validate output count
    if (recomputed_outputs.size() != grad_outputs.size()) {
        throw std::runtime_error("Checkpoint backward: output count mismatch");
    }

    // Use standard autograd backward - handles all complexity including nested checkpoints
    for (size_t i = 0; i < recomputed_outputs.size(); ++i) {
        if (recomputed_outputs[i].requires_grad() && recomputed_outputs[i].grad_fn()) {
            // retain_graph=true allows multiple outputs and nested checkpoints
            recomputed_outputs[i].backward(grad_outputs[i], /*retain_graph=*/true);
        }
    }

    // Extract gradients from the recomputed inputs
    // Return gradients directly for the engine to accumulate through shared impl_
    std::vector<Tensor> input_grads;
    input_grads.reserve(cached_recompute_inputs_.size());

    for (size_t i = 0; i < cached_recompute_inputs_.size(); ++i) {
        if (cached_recompute_inputs_[i].has_grad()) {
            input_grads.push_back(cached_recompute_inputs_[i].grad().value());
        } else {
            // No gradient computed - return zero
            input_grads.push_back(Tensor::zeros_like(cached_recompute_inputs_[i].tensor()));
        }
    }

    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    auto& stats = get_checkpoint_stats();
    stats.num_recomputations++;
    stats.total_recompute_time_ms += duration.count() / 1000.0;
    recompute_count_++;

    return input_grads;
}

auto CheckpointFunction::get_memory_savings() const -> size_t {
    return estimated_activation_memory_;
}

auto CheckpointFunction::recompute_forward(const std::vector<Variable>& inputs) -> std::vector<Variable> {
    // Check if we have cached outputs
    if (allow_caching_ && has_cached_outputs_) {
        return cached_recompute_outputs_;
    }

    // Recompute forward function with gradient tracking enabled
    auto outputs = forward_fn_(inputs);

    // Cache outputs if allowed
    if (allow_caching_) {
        cached_recompute_outputs_ = outputs;
        has_cached_outputs_ = true;
    }

    return outputs;
}

auto CheckpointFunction::estimate_memory(const std::vector<Variable>& vars) const -> size_t {
    size_t total = 0;
    for (const auto& var : vars) {
        const auto& tensor = var.tensor();
        total += tensor.numel() * tensor.dtype_size();
    }
    return total;
}

// ============================================================================
// Checkpoint Free Functions
// ============================================================================

// Internal shared_ptr-based implementation
// This is the core implementation that eliminates all Variable copying
static auto checkpoint_impl_shared(
    std::function<std::vector<Variable>(const std::vector<Variable>&)> fn,
    std::vector<std::shared_ptr<Variable>> input_ptrs,
    const std::vector<Variable*>& original_inputs
) -> std::vector<Variable> {
    // Check if any input requires gradients
    bool requires_grad = false;
    for (const auto& ptr : input_ptrs) {
        if (ptr->requires_grad()) {
            requires_grad = true;
            break;
        }
    }

    if (!is_checkpoint_enabled() || !requires_grad || !is_grad_enabled()) {
        // If checkpointing is disabled or no gradients needed, execute function normally
        std::vector<Variable> inputs_for_call;
        inputs_for_call.reserve(input_ptrs.size());
        for (const auto& ptr : input_ptrs) {
            inputs_for_call.push_back(*ptr);
        }
        return fn(inputs_for_call);
    }

    // Create checkpoint function with the user's function directly (no wrapping!)
    // The shared_ptrs keep Variables alive long enough to extract Tensors
    auto checkpoint_fn = std::make_shared<CheckpointFunction>(fn, true);

    // Save input tensors for recomputation
    std::vector<Tensor> input_tensors;
    input_tensors.reserve(input_ptrs.size());
    for (const auto& ptr : input_ptrs) {
        input_tensors.push_back(ptr->tensor());
    }
    checkpoint_fn->save_for_backward(std::move(input_tensors));

    // Set up backward graph connections to original inputs
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.reserve(input_ptrs.size());
    for (const auto& ptr : input_ptrs) {
        next_funcs.push_back(ptr->grad_fn());  // nullptr if input is leaf
    }
    checkpoint_fn->set_next_functions(next_funcs);

    // Store Variables on heap to keep them alive
    std::vector<Variable> inputs_for_storage;
    inputs_for_storage.reserve(input_ptrs.size());
    for (const auto& ptr : input_ptrs) {
        inputs_for_storage.push_back(*ptr);
    }
    checkpoint_fn->store_input_copies(inputs_for_storage);

    // Convert raw pointers to Variables for set_input_variables
    auto input_copy_ptrs = checkpoint_fn->get_input_copy_pointers();
    std::vector<Variable> input_vars;
    input_vars.reserve(input_copy_ptrs.size());
    for (auto* ptr : input_copy_ptrs) {
        input_vars.push_back(*ptr);
    }
    checkpoint_fn->set_input_variables(input_vars);

    // Store pointers to original leaf Variables for gradient accumulation
    if (!original_inputs.empty()) {
        checkpoint_fn->store_original_inputs(original_inputs);
    } else {
        // No originals provided - use the heap-allocated copies
        checkpoint_fn->store_original_inputs(checkpoint_fn->get_input_copy_pointers());
    }

    // Execute forward function using the user's function directly
    std::vector<Variable> inputs_for_execution;
    inputs_for_execution.reserve(input_ptrs.size());
    for (const auto& ptr : input_ptrs) {
        inputs_for_execution.push_back(*ptr);
    }
    auto outputs = fn(inputs_for_execution);

    // Attach gradient function to outputs
    for (auto& output : outputs) {
        if (output.requires_grad()) {
            output.set_grad_fn(checkpoint_fn);
        }
    }

    // Update statistics
    auto& stats = get_checkpoint_stats();
    stats.num_checkpoints++;
    size_t estimated_memory = 0;
    for (const auto& ptr : input_ptrs) {
        const auto& tensor = ptr->tensor();
        estimated_memory += tensor.numel() * tensor.dtype_size();
    }
    stats.saved_memory_bytes += estimated_memory;

    return outputs;
}

// Public API wrappers that convert to shared_ptr-based implementation
auto checkpoint(
    std::function<std::vector<Variable>(const std::vector<Variable>&)> fn,
    const std::vector<Variable>& inputs  // Changed to const reference to avoid unnecessary copy
) -> std::vector<Variable> {
    // Convert inputs to shared_ptrs immediately
    std::vector<std::shared_ptr<Variable>> input_ptrs;
    input_ptrs.reserve(inputs.size());
    for (const auto& input : inputs) {
        input_ptrs.push_back(std::make_shared<Variable>(input));
    }

    // Use empty original_inputs
    static const std::vector<Variable*> empty_originals;
    return checkpoint_impl_shared(fn, input_ptrs, empty_originals);
}

auto checkpoint_with_originals(
    std::function<std::vector<Variable>(const std::vector<Variable>&)> fn,
    const std::vector<Variable>& inputs,  // Changed to const reference to avoid unnecessary copy
    const std::vector<Variable*>& original_inputs  // Also const reference for consistency
) -> std::vector<Variable> {
    // Convert inputs to shared_ptrs immediately
    std::vector<std::shared_ptr<Variable>> input_ptrs;
    input_ptrs.reserve(inputs.size());
    for (const auto& input : inputs) {
        input_ptrs.push_back(std::make_shared<Variable>(input));
    }

    return checkpoint_impl_shared(fn, input_ptrs, original_inputs);
}

auto checkpoint(
    std::function<Variable(const Variable&)> fn,
    const Variable& input
) -> Variable {
    // Wrap single-input function as multi-input
    auto multi_fn = [fn](const std::vector<Variable>& inputs) -> std::vector<Variable> {
        if (inputs.size() != 1) {
            throw std::runtime_error("Checkpoint wrapper: expected single input");
        }
        return {fn(inputs[0])};
    };

    // CRITICAL: Create vector with input to avoid extra copy
    std::vector<Variable> inputs_vec;
    inputs_vec.push_back(input);
    auto outputs = checkpoint(multi_fn, inputs_vec);

    if (outputs.size() != 1) {
        throw std::runtime_error("Checkpoint wrapper: expected single output");
    }

    return outputs[0];
}

auto checkpoint_with_original(
    std::function<Variable(const Variable&)> fn,
    const Variable& input,
    Variable* original_input
) -> Variable {
    // Wrap single-input function as multi-input
    auto multi_fn = [fn](const std::vector<Variable>& inputs) -> std::vector<Variable> {
        if (inputs.size() != 1) {
            throw std::runtime_error("Checkpoint wrapper: expected single input");
        }
        return {fn(inputs[0])};
    };

    // Create vectors - this does copy Variable but necessary for API compatibility
    std::vector<Variable> inputs_vec;
    inputs_vec.push_back(input);
    std::vector<Variable*> originals_vec;
    originals_vec.push_back(original_input);
    auto outputs = checkpoint_with_originals(multi_fn, inputs_vec, originals_vec);

    if (outputs.size() != 1) {
        throw std::runtime_error("Checkpoint wrapper: expected single output");
    }

    return outputs[0];
}

// ============================================================================
// CheckpointContext Implementation
// ============================================================================

thread_local bool CheckpointContext::enabled_tls_ = true;

CheckpointContext::CheckpointContext(bool enabled)
    : enabled_(enabled),
      prev_enabled_(enabled_tls_),
      initial_stats_(get_checkpoint_stats()) {
    enabled_tls_ = enabled;
}

CheckpointContext::~CheckpointContext() {
    // Restore previous state
    enabled_tls_ = prev_enabled_;
}

auto CheckpointContext::get_stats() const -> CheckpointStats {
    // Return difference from initial stats
    auto current_stats = get_checkpoint_stats();
    CheckpointStats diff;

    diff.num_checkpoints = current_stats.num_checkpoints - initial_stats_.num_checkpoints;
    diff.num_recomputations = current_stats.num_recomputations - initial_stats_.num_recomputations;
    diff.saved_memory_bytes = current_stats.saved_memory_bytes - initial_stats_.saved_memory_bytes;
    diff.peak_memory_bytes = current_stats.peak_memory_bytes;
    diff.total_recompute_time_ms = current_stats.total_recompute_time_ms - initial_stats_.total_recompute_time_ms;

    return diff;
}

// ============================================================================
// MemoryTracker Implementation
// ============================================================================

thread_local bool MemoryTracker::tracking_enabled_ = false;
thread_local size_t MemoryTracker::current_memory_ = 0;
thread_local size_t MemoryTracker::peak_memory_ = 0;

auto MemoryTracker::start_tracking() -> void {
    tracking_enabled_ = true;
    current_memory_ = 0;
    peak_memory_ = 0;
}

auto MemoryTracker::stop_tracking() -> void {
    tracking_enabled_ = false;
}

auto MemoryTracker::current_memory() -> size_t {
    return current_memory_;
}

auto MemoryTracker::peak_memory() -> size_t {
    return peak_memory_;
}

auto MemoryTracker::reset() -> void {
    current_memory_ = 0;
    peak_memory_ = 0;
}

// ============================================================================
// CheckpointSegment Implementation
// ============================================================================

CheckpointSegment::CheckpointSegment(std::string name, int level)
    : name_(std::move(name)),
      nesting_level_(level) {}

auto CheckpointSegment::execute(
    std::function<std::vector<Variable>(const std::vector<Variable>&)> fn,
    std::vector<Variable> inputs
) -> std::vector<Variable> {
    // Execute function with checkpointing
    return checkpoint(fn, inputs);
}

} // namespace autograd
} // namespace tenzor
