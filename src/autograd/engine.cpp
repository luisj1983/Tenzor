#include "tenzor/autograd/engine.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/utils/error.hpp"
#include <unordered_set>
#include <functional>
#include <stdexcept>

namespace tenzor {

// Floating-point precision hierarchy for gradient accumulation.
// Higher precedence = higher precision. This avoids using dtype_size()
// which incorrectly equates Float16 and BFloat16 (both 2 bytes).
static auto dtype_precedence(DType dt) -> int {
    switch (dt) {
        case DType::Float64:    return 6;
        case DType::Complex128: return 6;
        case DType::Float32:    return 5;
        case DType::Complex64:  return 5;
        case DType::Float16:    return 4;
        case DType::BFloat16:   return 3;
        case DType::Int64:      return 2;
        case DType::Int32:      return 1;
        case DType::Int16:      return 1;
        case DType::Int8:       return 0;
        case DType::UInt8:      return 0;
        default:                return 0;
    }
}

auto BackwardEngine::execute(Variable& root, std::optional<Tensor> gradient, bool retain_graph) -> void {
    if (!root.requires_grad()) {
        return;
    }

    // Initialize gradient
    if (!gradient.has_value()) {
        if (root.tensor().numel() != 1) {
            throw AutogradException(
                "backward() requires an explicit gradient argument for non-scalar outputs "
                "(tensor has " + std::to_string(root.tensor().numel()) + " elements)");
        }
        gradient = ones_like(root.tensor());
    }

    root.grad() = *gradient;

    // If no grad_fn, this is a leaf variable, nothing to backprop
    if (!root.grad_fn()) {
        return;
    }

    // Topological sort from root
    auto sorted = topological_sort(root.grad_fn());

    // Helper to clean up computation graph (breaks circular references)
    auto cleanup_graph = [&]() {
        if (!retain_graph) {
            for (auto& func : sorted) {
                if (func) {
                    func->set_input_variables({});
                    func->set_next_functions({});
                }
            }
            if (root.grad_fn() && !root.is_leaf()) {
                root.set_grad_fn(nullptr);
            }
        }
    };

    try {
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
                // Sum all accumulated gradients with dtype promotion
                Tensor total_grad = accum_grads[0];
                for (size_t i = 1; i < accum_grads.size(); ++i) {
                    if (accum_grads[i].dtype() != total_grad.dtype()) {
                        DType target = (dtype_precedence(accum_grads[i].dtype()) >= dtype_precedence(total_grad.dtype()))
                            ? accum_grads[i].dtype() : total_grad.dtype();
                        total_grad = total_grad.to(target);
                        accum_grads[i] = accum_grads[i].to(target);
                    }
                    total_grad = total_grad + accum_grads[i];
                }
                grad_outputs.push_back(total_grad);
            }

            // Reload offloaded saved tensors back to GPU before backward
            function->reload_saved_tensors();

            // Compute gradients for inputs
            auto input_grads = function->backward(grad_outputs);

            // Release saved tensors immediately to free GPU memory for subsequent layers.
            // This is safe because saved tensors are only needed during this backward() call.
            if (!retain_graph) {
                function->release_saved_tensors();
            }

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

                // Validate gradient shape matches variable shape
                auto grad_shape = grad_to_apply.shape();
                auto var_shape = var.tensor().shape();
                if (grad_shape.size() != var_shape.size() ||
                    !std::equal(grad_shape.begin(), grad_shape.end(), var_shape.begin())) {
                    std::string expected, got;
                    for (size_t s = 0; s < var_shape.size(); ++s) {
                        if (s > 0) expected += ",";
                        expected += std::to_string(var_shape[s]);
                    }
                    for (size_t s = 0; s < grad_shape.size(); ++s) {
                        if (s > 0) got += ",";
                        got += std::to_string(grad_shape[s]);
                    }
                    throw AutogradException(
                        "Gradient shape mismatch: expected [" + expected + "] got [" + got + "]");
                }

                // Apply hooks (access through impl_ for handle pattern)
                if (var.impl_) {
                    for (auto& hook : var.impl_->hooks_) {
                        grad_to_apply = hook(grad_to_apply);
                    }
                }

                // Accumulate gradient to leaf variables
                if (var.is_leaf() || var.retains_grad()) {
                    if (var.has_grad()) {
                        // Promote to higher precision rather than silently demoting
                        auto existing_grad = var.grad().value();
                        if (grad_to_apply.dtype() != existing_grad.dtype()) {
                            DType target = (dtype_precedence(grad_to_apply.dtype()) >= dtype_precedence(existing_grad.dtype()))
                                ? grad_to_apply.dtype() : existing_grad.dtype();
                            if (target != existing_grad.dtype()) {
                                var.grad() = existing_grad.to(target);
                            }
                            grad_to_apply = grad_to_apply.to(target);
                            existing_grad = var.grad().value();
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
    } catch (...) {
        clear_gradients();
        cleanup_graph();
        throw;
    }

    clear_gradients();
    cleanup_graph();
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
            throw AutogradException("Cycle detected in computation graph");
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

// Thread-local engine — each thread gets its own instance so concurrent
// backward passes don't corrupt the shared grad_accumulators_ map.
auto backward_engine() -> BackwardEngine& {
    static thread_local BackwardEngine engine;
    return engine;
}

} // namespace tenzor
