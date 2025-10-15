#include "tenzor/autograd/engine.hpp"
#include "tenzor/ops/creation.hpp"
#include <unordered_set>
#include <functional>
#include <stdexcept>
#include <iostream>

namespace tenzor {

auto BackwardEngine::execute(Variable& root, std::optional<Tensor> gradient, bool retain_graph) -> void {
    if (!root.requires_grad()) {
        return;
    }

    // Initialize gradient
    if (!gradient.has_value()) {
        gradient = ones_like(root.tensor());
    }

    root.grad() = *gradient;

    // If no grad_fn, this is a leaf variable, nothing to backprop
    if (!root.grad_fn()) {
        return;
    }

    // Topological sort from root
    auto sorted = topological_sort(root.grad_fn());

    // Execute backward in reverse topological order
    std::cout << "Starting backward execution with " << sorted.size() << " functions" << std::endl;

    // First, validate all pointers are non-null
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (!sorted[i]) {
            std::cout << "ERROR: Function at index " << i << " is nullptr!" << std::endl;
        }
    }
    std::cout << "Pointer validation complete" << std::endl;

    size_t counter = 1;
    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it, ++counter) {
        std::cout << "Processing function " << counter << "/" << sorted.size();

        // Check if shared_ptr is valid before dereferencing
        if (!*it) {
            std::cout << " - NULLPTR DETECTED! Skipping..." << std::endl;
            continue;
        }

        auto& function = *it;
        std::cout << " at address: " << function.get()
                  << " (type: " << typeid(*function).name() << ")" << std::endl;

        // Get the gradient for this function's output
        std::vector<Tensor> grad_outputs;

        // The gradient comes from the accumulated gradients
        auto accum_grads = get_accumulated_grads(function.get());
        if (accum_grads.empty()) {
            // This is the root function, use the root gradient
            grad_outputs.push_back(root.grad().value());
        } else {
            // Sum all accumulated gradients
            Tensor total_grad = accum_grads[0];
            for (size_t i = 1; i < accum_grads.size(); ++i) {
                total_grad = total_grad + accum_grads[i];
            }
            grad_outputs.push_back(total_grad);
        }

        // Compute gradients for inputs
        std::cout << "  Calling backward()..." << std::flush;
        auto input_grads = function->backward(grad_outputs);
        std::cout << " OK, returned " << input_grads.size() << " gradients" << std::endl;

        // Accumulate gradients to input variables
        std::cout << "  Getting input_variables()..." << std::flush;
        const auto& input_vars = function->input_variables();
        std::cout << " got " << input_vars.size() << " input vars" << std::endl;

        std::cout << "  Accumulating to input vars..." << std::flush;
        for (size_t i = 0; i < input_vars.size() && i < input_grads.size(); ++i) {
            // Get reference to Variable (stored by value)
            Variable& var = const_cast<Variable&>(input_vars[i]);

            // Skip placeholder Variables (default-constructed with requires_grad=false)
            if (!var.requires_grad()) {
                continue;
            }

            std::cout << " [" << i << "]" << std::flush;

            Tensor grad_to_apply = input_grads[i];

            // Apply hooks (access through impl_ for handle pattern)
            if (var.impl_) {
                for (auto& hook : var.impl_->hooks_) {
                    grad_to_apply = hook(grad_to_apply);
                }
            }

            // Accumulate gradient to leaf variables
            if (var.is_leaf() || var.retains_grad()) {
                if (var.has_grad()) {
                    var.grad() = var.grad().value() + grad_to_apply;
                } else {
                    var.grad() = grad_to_apply;
                }
            }
        }
        std::cout << " done" << std::endl;

        // Also accumulate to next functions for non-leaf variables
        std::cout << "  Getting next_functions()..." << std::flush;
        const auto& next_funcs = function->next_functions();
        std::cout << " got " << next_funcs.size() << " next funcs" << std::endl;

        std::cout << "  Accumulating to next funcs..." << std::flush;
        for (size_t i = 0; i < next_funcs.size() && i < input_grads.size(); ++i) {
            std::cout << " [" << i << "]" << std::flush;
            if (next_funcs[i]) {
                accumulate_grad(next_funcs[i].get(), input_grads[i]);
            }
        }
        std::cout << " done" << std::endl;
    }

    std::cout << "Backward execution complete" << std::endl;
    clear_gradients();

    // Clear computation graph if not retaining
    if (!retain_graph) {
        // Note: In a full implementation, we would clear grad_fn references here
        // For now, we just clear the gradient accumulator
    }
}

auto BackwardEngine::topological_sort(std::shared_ptr<Function> root)
    -> std::vector<std::shared_ptr<Function>> {
    std::vector<std::shared_ptr<Function>> sorted;
    std::unordered_set<Function*> visited;
    std::unordered_set<Function*> recursion_stack;

    // DFS-based topological sort
    std::function<void(std::shared_ptr<Function>)> dfs;
    dfs = [&](std::shared_ptr<Function> node) {
        if (!node) return;

        // Check for cycles
        if (recursion_stack.count(node.get())) {
            throw std::runtime_error("Cycle detected in computation graph");
        }

        // Already visited
        if (visited.count(node.get())) {
            return;
        }

        visited.insert(node.get());
        recursion_stack.insert(node.get());

        // Visit all dependencies (next functions)
        for (const auto& next_func : node->next_functions()) {
            if (next_func) {
                dfs(next_func);
            }
        }

        recursion_stack.erase(node.get());
        sorted.push_back(node);
    };

    dfs(root);
    return sorted;
}

auto BackwardEngine::clear_gradients() -> void {
    grad_accumulators_.clear();
}

auto BackwardEngine::accumulate_grad(Function* func, Tensor grad) -> void {
    grad_accumulators_[func].push_back(std::move(grad));
}

auto BackwardEngine::get_accumulated_grads(Function* func) -> std::vector<Tensor> {
    auto it = grad_accumulators_.find(func);
    if (it == grad_accumulators_.end()) {
        return {};
    }
    return it->second;
}

// Global engine
auto backward_engine() -> BackwardEngine& {
    static BackwardEngine engine;
    return engine;
}

} // namespace tenzor
