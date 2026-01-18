#include "tenzor/autograd/engine.hpp"
#include "tenzor/ops/creation.hpp"
#include <unordered_set>
#include <functional>
#include <stdexcept>

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
    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
        // Check if shared_ptr is valid before dereferencing
        if (!*it) {
            continue;
        }

        auto& function = *it;

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
        auto input_grads = function->backward(grad_outputs);

        // Accumulate gradients to input variables
        const auto& input_vars = function->input_variables();

        for (size_t i = 0; i < input_vars.size() && i < input_grads.size(); ++i) {
            // Get reference to Variable (stored by value)
            Variable& var = const_cast<Variable&>(input_vars[i]);

            // Skip Variables without gradients (default-constructed with requires_grad=false)
            if (!var.requires_grad()) {
                continue;
            }

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
                    // Ensure incoming gradient matches the dtype of existing gradient
                    auto existing_grad = var.grad().value();
                    if (grad_to_apply.dtype() != existing_grad.dtype()) {
                        grad_to_apply = grad_to_apply.to(existing_grad.dtype());
                    }
                    var.grad() = existing_grad + grad_to_apply;
                } else {
                    var.grad() = grad_to_apply;
                }
            }
        }

        // Also accumulate to next functions for non-leaf variables
        const auto& next_funcs = function->next_functions();

        for (size_t i = 0; i < next_funcs.size() && i < input_grads.size(); ++i) {
            if (next_funcs[i]) {
                accumulate_grad(next_funcs[i].get(), input_grads[i]);
            }
        }
    }

    clear_gradients();

    // Clear computation graph if not retaining
    if (!retain_graph) {
        // Clear grad_fn references to prevent circular shared_ptr references
        // This is critical to avoid memory leaks and infinite loops in subsequent backward passes
        for (auto& func : sorted) {
            if (func) {
                // Clear input variables which hold circular references
                func->set_input_variables({});
                // Clear next functions to break the graph
                func->set_next_functions({});
            }
        }
        // Also clear the root's grad_fn if it's not a leaf
        if (root.grad_fn() && !root.is_leaf()) {
            root.set_grad_fn(nullptr);
        }
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
