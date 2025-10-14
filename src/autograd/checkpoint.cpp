/**
 * @file checkpoint.cpp
 * @brief Complete implementation of gradient checkpointing
 */

#include "../../include/tenzor/autograd/checkpoint.hpp"
#include "../../include/tenzor/ops/creation.hpp"
#include <chrono>
#include <iostream>
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
    std::function<std::vector<Variable>(std::vector<Variable>)> forward_fn,
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

    // Recompute forward pass with gradient tracking enabled
    // CRITICAL: Keep these Variables alive throughout the entire backward pass
    // Store them as a member so they outlive this function
    cached_recompute_inputs_.clear();
    cached_recompute_inputs_.reserve(saved_tensors().size());
    for (const auto& tensor : saved_tensors()) {
        cached_recompute_inputs_.emplace_back(tensor, true); // Enable gradient tracking
    }

    auto recomputed_outputs = recompute_forward(cached_recompute_inputs_);

    // Validate output count
    if (recomputed_outputs.size() != grad_outputs.size()) {
        throw std::runtime_error("Checkpoint backward: output count mismatch");
    }

    // Manual backward pass implementation:
    // We need to traverse the computation graph from the recomputed outputs
    // down to the inputs WITHOUT using Variable::backward() to avoid dangling pointers.
    //
    // Approach: Topologically sort all functions in the recomputed graph,
    // then execute backward in reverse order, accumulating gradients manually.

    // Collect all grad_fns that need to execute backward
    std::vector<std::shared_ptr<Function>> functions_to_backward;
    std::unordered_set<Function*> visited;

    // Simple DFS to collect all functions reachable from outputs
    std::function<void(std::shared_ptr<Function>)> collect_functions;
    collect_functions = [&](std::shared_ptr<Function> fn) {
        if (!fn || visited.count(fn.get())) {
            return;
        }
        visited.insert(fn.get());

        // Visit dependencies first (next_functions)
        for (const auto& next_fn : fn->next_functions()) {
            collect_functions(next_fn);
        }

        // Add this function after dependencies (reverse topological order)
        functions_to_backward.push_back(fn);
    };

    // Start DFS from each output's grad_fn
    for (const auto& output : recomputed_outputs) {
        if (output.grad_fn()) {
            collect_functions(output.grad_fn());
        }
    }

    // Now execute backward in reverse topological order
    // Map to accumulate gradients for each function
    std::unordered_map<Function*, std::vector<Tensor>> grad_map;

    // Initialize with output gradients
    for (size_t i = 0; i < recomputed_outputs.size(); ++i) {
        if (recomputed_outputs[i].grad_fn()) {
            grad_map[recomputed_outputs[i].grad_fn().get()].push_back(grad_outputs[i]);
        }
    }

    // Process functions in reverse topological order
    for (auto it = functions_to_backward.rbegin(); it != functions_to_backward.rend(); ++it) {
        auto& fn = *it;

        // Get accumulated gradient for this function's output
        auto grad_it = grad_map.find(fn.get());
        if (grad_it == grad_map.end() || grad_it->second.empty()) {
            continue; // No gradient for this function
        }

        // Sum all gradients if multiple
        Tensor output_grad = grad_it->second[0];
        for (size_t i = 1; i < grad_it->second.size(); ++i) {
            output_grad = output_grad + grad_it->second[i];
        }

        // Call backward to compute input gradients
        auto input_grads = fn->backward({output_grad});

        // Distribute input gradients to input variables and next functions
        // CRITICAL FIX FOR REPEATED INPUTS (e.g., x*x):
        // When the same Variable is used multiple times, the function's input_variables()
        // will have duplicate pointers (e.g., [&x, &x]). Both gradients must be summed
        // and accumulated to the SAME cached_recompute_inputs_ element.
        //
        // Example: y = x * x
        //   - MulBackward returns [grad_a, grad_b] where grad_a = x, grad_b = x
        //   - Both input_vars[0] and input_vars[1] point to the same recomputed Variable
        //   - We detect this by comparing pointers: if input_vars[i] == input_vars[j] for j < i,
        //     then both gradients should accumulate to cached_recompute_inputs_[j]
        //   - Final gradient: x.grad = grad_a + grad_b = x + x = 2x (CORRECT!)
        const auto& input_vars = fn->input_variables();
        const auto& next_fns = fn->next_functions();

        for (size_t i = 0; i < input_grads.size(); ++i) {
            // Check if this is a leaf: either beyond next_fns size OR next_fns[i] is nullptr
            bool is_leaf_input = (i >= next_fns.size()) || !next_fns[i];

            // If this is a leaf node, find which cached input it corresponds to
            if (is_leaf_input) {
                // This is a leaf input - match input_vars[i] to one of cached_recompute_inputs_
                // CRITICAL: input_vars points to Variables in the recomputed graph.
                //
                // For operations like x*x, both input_vars[0] and input_vars[1] point to the SAME Variable.
                // For operations like x*const, input_vars[0] is our input, input_vars[1] is a constant.
                //
                // Strategy: Only accumulate gradients for indices < cached_recompute_inputs_.size()
                // Check for repeated inputs by comparing pointers.
                size_t target_index = i;  // Default to index-based matching

                // Check if this input_vars[i] appeared earlier in input_vars
                // If so, it's a repeated input and should go to the same target
                if (i < input_vars.size() && input_vars[i] != nullptr) {
                    for (size_t j = 0; j < i; ++j) {
                        if (j < input_vars.size() && input_vars[j] == input_vars[i]) {
                            // Found a previous occurrence - use the same target index
                            target_index = j;
                            break;
                        }
                    }
                }

                // CRITICAL FIX: Only accumulate if target_index is within our actual inputs.
                // Gradients for Variables created inside the checkpoint (like constants) should be ignored.
                if (target_index < cached_recompute_inputs_.size()) {
                    if (cached_recompute_inputs_[target_index].has_grad()) {
                        cached_recompute_inputs_[target_index].grad() =
                            cached_recompute_inputs_[target_index].grad().value() + input_grads[i];
                    } else {
                        cached_recompute_inputs_[target_index].grad() = input_grads[i];
                    }
                }
                // else: This gradient is for a Variable created inside the checkpoint (index out of range), ignore it
            } else if (i < next_fns.size() && next_fns[i]) {
                // Propagate to next function
                grad_map[next_fns[i].get()].push_back(input_grads[i]);
            }
            // else: i >= next_fns.size() would mean missing next_fn entry, treated as leaf above
        }
    }

    // CRITICAL FIX: Accumulate gradients directly to original leaf inputs
    // For leaf Variables, we must accumulate to the ORIGINAL Variables passed to checkpoint(),
    // not to the recomputed copies or heap-allocated copies.
    const auto& original_inputs = get_original_inputs();
    const auto& next_fns = next_functions();

    for (size_t i = 0; i < cached_recompute_inputs_.size(); ++i) {
        // Check if this input is a leaf (no grad_fn)
        bool is_leaf = (i >= next_fns.size()) || !next_fns[i];

        if (is_leaf && i < original_inputs.size() && original_inputs[i] != nullptr) {
            // This is a leaf input - accumulate gradient directly to the original Variable
            if (cached_recompute_inputs_[i].has_grad()) {
                auto grad_tensor = *cached_recompute_inputs_[i].grad();

                // Accumulate to original Variable's grad buffer
                if (original_inputs[i]->has_grad()) {
                    original_inputs[i]->grad() = original_inputs[i]->grad().value() + grad_tensor;
                } else {
                    original_inputs[i]->grad() = grad_tensor;
                }
            }
        }
    }

    // Extract gradients from the recomputed input variables
    // These gradients will be returned and the BackwardEngine will route them
    // to the correct original input Variables using next_functions_
    // For leaf Variables that are TRULY leaves in the user's graph, we've already
    // accumulated above, so return zeros to avoid double-counting.
    std::vector<Tensor> input_grads;
    input_grads.reserve(cached_recompute_inputs_.size());
    for (size_t i = 0; i < cached_recompute_inputs_.size(); ++i) {
        bool is_leaf = (i >= next_fns.size()) || !next_fns[i];

        // CRITICAL FIX: Only use "accumulate and return zero" pattern if:
        // 1. This input is a leaf in the recomputed graph (no next_fn), AND
        // 2. The original pointer is NOT one of our heap-allocated copies (true user original)
        // For nested checkpoints, the inner checkpoint's "originals" are heap copies,
        // so we need to return gradients for propagation, not accumulate to heap copies.
        bool original_is_heap_copy = false;
        if (is_leaf && i < original_inputs.size() && original_inputs[i] != nullptr) {
            // Check if this original pointer is one of our heap-allocated copies
            for (const auto& copy : input_variable_copies_) {
                if (original_inputs[i] == copy.get()) {
                    original_is_heap_copy = true;
                    break;
                }
            }
        }

        if (is_leaf && i < original_inputs.size() && original_inputs[i] != nullptr && !original_is_heap_copy) {
            // This is a TRUE original (not a heap copy) - we already accumulated, return zero
            input_grads.push_back(Tensor::zeros_like(cached_recompute_inputs_[i].tensor()));
        } else if (cached_recompute_inputs_[i].has_grad()) {
            // Non-leaf, no original, or original is heap copy - return gradient for propagation
            input_grads.push_back(*cached_recompute_inputs_[i].grad());
        } else {
            // If no gradient was computed, return zero gradient
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

// Internal implementation used by both checkpoint variants
static auto checkpoint_impl(
    std::function<std::vector<Variable>(std::vector<Variable>)> fn,
    std::vector<Variable> inputs,
    std::vector<Variable*> original_inputs
) -> std::vector<Variable> {
    // Check if any input requires gradients
    bool requires_grad = false;
    for (const auto& input : inputs) {
        if (input.requires_grad()) {
            requires_grad = true;
            break;
        }
    }

    if (!is_checkpoint_enabled() || !requires_grad || !is_grad_enabled()) {
        // If checkpointing is disabled or no gradients needed, execute function normally
        return fn(inputs);
    }

    // Create checkpoint function
    auto checkpoint_fn = std::make_shared<CheckpointFunction>(fn, true);

    // Save input tensors for recomputation
    std::vector<Tensor> input_tensors;
    input_tensors.reserve(inputs.size());
    for (const auto& input : inputs) {
        input_tensors.push_back(input.tensor());
    }
    checkpoint_fn->save_for_backward(std::move(input_tensors));

    // Set up backward graph connections to original inputs
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.reserve(inputs.size());
    for (const auto& input : inputs) {
        next_funcs.push_back(input.grad_fn());  // nullptr if input is leaf
    }
    checkpoint_fn->set_next_functions(next_funcs);

    // Store heap-allocated copies to keep input_variables_ pointers valid
    checkpoint_fn->store_input_copies(inputs);
    checkpoint_fn->set_input_variables(checkpoint_fn->get_input_copy_pointers());

    // CRITICAL FIX: For leaf Variables, we need to store pointers to the ORIGINAL
    // Variables passed by the caller, not copies. Since `inputs` are passed by value,
    // we receive copies. Use original_inputs if provided, otherwise use copies.
    if (!original_inputs.empty()) {
        checkpoint_fn->store_original_inputs(original_inputs);
    } else {
        // No originals provided - use the heap-allocated copies
        // This won't work perfectly for leaf variables, but it's the best we can do
        checkpoint_fn->store_original_inputs(checkpoint_fn->get_input_copy_pointers());
    }

    // Execute forward function
    auto outputs = fn(inputs);

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
    for (const auto& var : inputs) {
        const auto& tensor = var.tensor();
        estimated_memory += tensor.numel() * tensor.dtype_size();
    }
    stats.saved_memory_bytes += estimated_memory;

    return outputs;
}

auto checkpoint(
    std::function<std::vector<Variable>(std::vector<Variable>)> fn,
    std::vector<Variable> inputs
) -> std::vector<Variable> {
    // Use empty original_inputs - gradients will go to heap copies (not ideal for leaf variables)
    return checkpoint_impl(fn, std::move(inputs), {});
}

auto checkpoint_with_originals(
    std::function<std::vector<Variable>(std::vector<Variable>)> fn,
    std::vector<Variable> inputs,
    std::vector<Variable*> original_inputs
) -> std::vector<Variable> {
    // Use provided original_inputs - gradients will go to originals (works for leaf variables)
    return checkpoint_impl(fn, std::move(inputs), std::move(original_inputs));
}

auto checkpoint(
    std::function<Variable(const Variable&)> fn,
    const Variable& input
) -> Variable {
    // Wrap single-input function as multi-input
    auto multi_fn = [fn](std::vector<Variable> inputs) -> std::vector<Variable> {
        if (inputs.size() != 1) {
            throw std::runtime_error("Checkpoint wrapper: expected single input");
        }
        return {fn(inputs[0])};
    };

    auto outputs = checkpoint(multi_fn, {input});

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
    auto multi_fn = [fn](std::vector<Variable> inputs) -> std::vector<Variable> {
        if (inputs.size() != 1) {
            throw std::runtime_error("Checkpoint wrapper: expected single input");
        }
        return {fn(inputs[0])};
    };

    auto outputs = checkpoint_with_originals(multi_fn, {input}, {original_input});

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
    std::function<std::vector<Variable>(std::vector<Variable>)> fn,
    std::vector<Variable> inputs
) -> std::vector<Variable> {
    // Execute function with checkpointing
    return checkpoint(fn, inputs);
}

} // namespace autograd
} // namespace tenzor
