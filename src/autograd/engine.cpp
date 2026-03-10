#include "tenzor/autograd/engine.hpp"
#include "tenzor/autograd/anomaly_mode.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/type_promotion.hpp"
#include "tenzor/utils/error.hpp"
#include <unordered_set>
#include <functional>
#include <stdexcept>
#include <typeinfo>

#ifdef __GNUC__
#include <cxxabi.h>
#endif

namespace tenzor {

// RAII scope guard for exception-safe cleanup
namespace {
struct ScopeGuard {
    std::function<void()> fn;
    bool dismissed = false;
    ~ScopeGuard() { if (!dismissed && fn) fn(); }
    void dismiss() { dismissed = true; }
};
} // anonymous namespace

// Check computed gradients for NaN/Inf when anomaly detection is enabled.
// Zero overhead when disabled (thread-local bool early-return).
static void check_for_anomaly(const std::vector<Tensor>& grads,
                              const Function* func) {
    if (!is_anomaly_detection_enabled()) return;

    for (size_t i = 0; i < grads.size(); ++i) {
        const auto& grad = grads[i];
        if (grad.numel() == 0) continue;

        // Only check floating-point gradients (integer grads can't be NaN/Inf)
        if (grad.dtype() != DType::Float32 && grad.dtype() != DType::Float64 &&
            grad.dtype() != DType::Float16 && grad.dtype() != DType::BFloat16) {
            continue;
        }

        auto nan_mask = isnan(grad);
        auto inf_mask = isinf(grad);

        // Sum bool masks as float to check if any anomalies exist
        auto nan_count = sum(nan_mask.to(DType::Float32));
        auto inf_count = sum(inf_mask.to(DType::Float32));

        bool has_nan = nan_count.item<float>() > 0.0f;
        bool has_inf = inf_count.item<float>() > 0.0f;

        if (has_nan || has_inf) {
            // Get demangled function name for readability
            std::string func_name = typeid(*func).name();
#ifdef __GNUC__
            int status = 0;
            char* demangled = abi::__cxa_demangle(func_name.c_str(), nullptr, nullptr, &status);
            if (status == 0 && demangled) {
                func_name = demangled;
                free(demangled);
            }
#endif
            std::string anomaly;
            if (has_nan && has_inf) anomaly = "NaN and Inf";
            else if (has_nan) anomaly = "NaN";
            else anomaly = "Inf";

            throw AutogradException(
                "Anomaly detected: gradient output " + std::to_string(i) +
                " contains " + anomaly + " values in backward of '" +
                func_name + "'");
        }
    }
}

// Use the shared promote_types() for gradient dtype promotion.
// This replaces the old local promote_types() that duplicated type_promotion.cpp logic.

auto BackwardEngine::execute(Variable& root, std::optional<Tensor> gradient,
                             bool retain_graph, bool create_graph) -> void {
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
    } else {
        // Validate user-supplied gradient shape matches root tensor shape
        auto grad_shape = gradient->shape();
        auto root_shape = root.tensor().shape();
        if (grad_shape.size() != root_shape.size() ||
            !std::equal(grad_shape.begin(), grad_shape.end(), root_shape.begin())) {
            auto fmt = [](std::span<const int64_t> s) {
                std::string r = "[";
                for (size_t i = 0; i < s.size(); ++i) {
                    if (i > 0) r += ", ";
                    r += std::to_string(s[i]);
                }
                return r + "]";
            };
            throw AutogradException(
                "User-supplied gradient shape mismatch: expected " +
                fmt(root_shape) + " got " + fmt(grad_shape));
        }
    }

    root.set_grad(*gradient);

    // If no grad_fn, this is a leaf variable, nothing to backprop
    if (!root.grad_fn()) {
        return;
    }

    // Save and clear grad_accumulators_ for re-entrancy safety.
    // Nested backward calls (from checkpointing) must use independent
    // accumulator maps to avoid corrupting the outer call's state.
    auto saved_accumulators = std::move(grad_accumulators_);
    grad_accumulators_.clear();

    // Topological sort from root
    // Use a local variable (not the instance cache) to be re-entrant safe.
    // Nested backward calls (e.g. from gradient checkpointing) invoke
    // execute() on the same thread-local engine; a shared cache would be
    // overwritten, invalidating the outer call's iterators.
    auto sorted = topological_sort(root.grad_fn());

    // Set the create_graph flag so backward functions know to use Variable ops
    // Use RAII guard to ensure flag is restored even on exception
    std::optional<CreateGraphGuard> graph_guard;
    if (create_graph) {
        graph_guard.emplace();
    }

    // Helper to clean up computation graph (breaks circular references)
    auto cleanup_graph = [&]() {
        if (!retain_graph) {
            for (auto& func : sorted) {
                if (func) {
                    func->set_input_variables({});
                    func->set_next_functions({});
                }
            }
            sorted.clear();
            if (root.grad_fn() && !root.is_leaf()) {
                root.set_grad_fn(nullptr);
            }
        }
    };

    // RAII guards for exception-safe cleanup
    ScopeGuard accum_guard{[&]{ grad_accumulators_ = std::move(saved_accumulators); }};
    ScopeGuard cleanup_guard{[&]{ clear_gradients(); cleanup_graph(); }};

    // Seed root gradient into accumulator so the root function is handled
    // uniformly — no fragile "empty accumulators = root" assumption.
    accumulate_grad(root.grad_fn().get(), root.grad().value());

    {
        // Execute backward in reverse topological order
        for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
            // Check if shared_ptr is valid before dereferencing
            if (!*it) {
                continue;
            }

            auto& function = *it;

            // Get the gradient for this function's output
            std::vector<Tensor> grad_outputs;

            const auto& accum_grads = get_accumulated_grads(function.get());
            if (accum_grads.empty()) {
                continue;  // No gradient flows to this function
            }

            // Sum all accumulated gradients with dtype promotion.
            // Pre-compute target dtype to avoid repeated conversions.
            DType target_dtype = accum_grads[0].dtype();
            for (size_t i = 1; i < accum_grads.size(); ++i) {
                target_dtype = promote_types(target_dtype, accum_grads[i].dtype());
            }
            Tensor total_grad = (accum_grads[0].dtype() == target_dtype)
                ? accum_grads[0]
                : accum_grads[0].to(target_dtype);
            for (size_t i = 1; i < accum_grads.size(); ++i) {
                const Tensor& gi = accum_grads[i];
                total_grad = total_grad + (gi.dtype() == target_dtype ? gi : gi.to(target_dtype));
            }
            grad_outputs.push_back(total_grad);

            // Reload offloaded saved tensors back to GPU before backward
            function->reload_saved_tensors();

            // Validate saved tensors haven't been modified in-place since forward
            function->validate_saved_tensors();

            // Compute gradients for inputs
            std::vector<Tensor> input_grads;

            if (create_graph) {
                // Higher-order gradient path: use backward_with_variables
                // which operates on Variables so the backward computation itself
                // is tracked by autograd (enabling gradients of gradients).
                std::vector<Variable> var_grad_outputs;
                var_grad_outputs.reserve(grad_outputs.size());
                for (auto& g : grad_outputs) {
                    // Wrap the gradient tensor as a Variable with requires_grad=true
                    // so that operations on it during backward are tracked
                    var_grad_outputs.emplace_back(g, true);
                }

                auto var_input_grads = function->backward_with_variables(var_grad_outputs);

                // Extract the underlying tensors for accumulation, but the Variables
                // in var_input_grads have grad_fn set from the Variable operations
                // used in backward_with_variables, enabling higher-order differentiation
                input_grads.reserve(var_input_grads.size());
                for (auto& vg : var_input_grads) {
                    input_grads.push_back(vg.tensor());
                }
            } else {
                // Standard backward path: raw Tensor operations
                input_grads = function->backward(grad_outputs);
            }

            // Check for NaN/Inf in computed gradients when anomaly detection is on
            check_for_anomaly(input_grads, function.get());

            // Validate gradient dtypes are floating-point or complex
            for (size_t i = 0; i < input_grads.size(); ++i) {
                const auto& g = input_grads[i];
                if (g.is_valid() && g.numel() > 0 &&
                    !g.is_floating_point() && !g.is_complex()) {
                    throw std::runtime_error(
                        std::string("Backward function ") + function->name() +
                        " returned non-floating-point gradient (dtype=" +
                        std::string(dtype_name(g.dtype())) + " at index " +
                        std::to_string(i) +
                        "). Gradients must be floating-point or complex.");
                }
            }

            // Release saved tensors immediately to free GPU memory for subsequent layers.
            // This is safe because saved tensors are only needed during this backward() call.
            if (!retain_graph) {
                function->release_saved_tensors();
            }

            // Accumulate gradients to input variables
            auto& input_vars = function->input_variables();

            if (input_grads.size() < input_vars.size()) {
                // Backward returned fewer gradients than inputs — warn and pad with empty
                // (this could be intentional for ops that don't need all input gradients)
            }

            for (size_t i = 0; i < input_vars.size() && i < input_grads.size(); ++i) {
                Variable& var = input_vars[i];

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
                        "In " + function->name() + ".backward(): gradient for input " +
                        std::to_string(i) + " has shape [" + got + "], expected [" + expected + "]");
                }

                // Apply hooks (access through impl_ for handle pattern)
                // Take shared_lock for thread-safe iteration, then copy hooks
                // to local before iterating — a hook may register/unregister
                if (var.impl_) {
                    std::map<size_t, std::function<Tensor(const Tensor&)>> hooks_copy;
                    {
                        std::shared_lock lock(var.impl_->hooks_mutex_);
                        if (!var.impl_->hooks_.empty()) {
                            hooks_copy = var.impl_->hooks_;
                        }
                    }
                    for (auto& [id, hook] : hooks_copy) {
                        grad_to_apply = hook(grad_to_apply);
                    }
                }

                // Accumulate gradient to leaf variables.
                // When thread_safe_ is enabled, lock to prevent concurrent
                // accumulation from corrupting the gradient tensor.
                if (var.is_leaf() || var.retains_grad()) {
                    auto accumulate = [&]() {
                        if (var.has_grad()) {
                            auto existing_grad = var.grad().value();
                            if (grad_to_apply.dtype() != existing_grad.dtype()) {
                                DType target = promote_types(grad_to_apply.dtype(), existing_grad.dtype());
                                existing_grad = existing_grad.to(target);
                                grad_to_apply = grad_to_apply.to(target);
                            }
                            var.set_grad(existing_grad + grad_to_apply);
                        } else {
                            var.set_grad(grad_to_apply);
                        }
                    };

                    if (var.impl_) {
                        // Always acquire mutex to prevent TOCTOU race:
                        // thread_safe_ could be set between check and lock acquisition
                        std::lock_guard lock(var.impl_->grad_mutex_);
                        accumulate();
                    } else {
                        accumulate();
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
    }
    // ScopeGuards handle cleanup in both normal and exception paths
}

auto BackwardEngine::topological_sort(std::shared_ptr<Function> root)
    -> std::vector<std::shared_ptr<Function>> {
    std::vector<std::shared_ptr<Function>> sorted;
    std::unordered_set<Function*> visited;
    std::unordered_set<Function*> recursion_stack;

    // Iterative DFS-based topological sort (avoids stack overflow on deep graphs)
    struct Frame {
        std::shared_ptr<Function> node;
        size_t child_idx;  // which child to visit next
    };
    std::vector<Frame> stack;
    stack.push_back({root, 0});
    visited.insert(root.get());
    recursion_stack.insert(root.get());

    while (!stack.empty()) {
        auto& frame = stack.back();
        auto& node = frame.node;

        if (!node) {
            stack.pop_back();
            continue;
        }

        const auto& children = node->next_functions();

        if (frame.child_idx < children.size()) {
            // Process next child
            const auto& child = children[frame.child_idx++];
            if (!child) continue;

            if (recursion_stack.count(child.get())) {
                throw AutogradException("Cycle detected in computation graph");
            }
            if (visited.count(child.get())) {
                continue;
            }

            visited.insert(child.get());
            recursion_stack.insert(child.get());
            stack.push_back({child, 0});
        } else {
            // All children visited — post-order: add to sorted
            recursion_stack.erase(node.get());
            sorted.push_back(std::move(node));
            stack.pop_back();
        }
    }

    return sorted;
}

auto BackwardEngine::clear_gradients() -> void {
    grad_accumulators_.clear();
}

auto BackwardEngine::accumulate_grad(Function* func, Tensor grad) -> void {
    grad_accumulators_[func].push_back(std::move(grad));
}

auto BackwardEngine::get_accumulated_grads(Function* func) -> const std::vector<Tensor>& {
    static const std::vector<Tensor> empty;
    auto it = grad_accumulators_.find(func);
    if (it == grad_accumulators_.end()) {
        return empty;
    }
    return it->second;
}

auto BackwardEngine::execute_multi(std::vector<Variable*> roots,
                                   std::vector<Tensor> gradients,
                                   bool retain_graph) -> void {
    if (roots.empty()) {
        return;
    }

    if (roots.size() != gradients.size()) {
        throw AutogradException(
            "execute_multi: number of roots (" + std::to_string(roots.size()) +
            ") must match number of gradients (" + std::to_string(gradients.size()) + ")");
    }

    // Save and clear grad_accumulators_ for re-entrancy safety.
    // Nested backward calls (from checkpointing) must use independent
    // accumulator maps to avoid corrupting the outer call's state.
    auto saved_accumulators = std::move(grad_accumulators_);
    grad_accumulators_.clear();

    // Initialize gradients for each root
    for (size_t i = 0; i < roots.size(); ++i) {
        if (!roots[i] || !roots[i]->requires_grad()) {
            continue;
        }
        roots[i]->set_grad(gradients[i]);
    }

    // Build combined topological sort from all roots using iterative DFS
    // (iterative to avoid stack overflow on deep computation graphs)
    std::vector<std::shared_ptr<Function>> sorted;
    std::unordered_set<Function*> visited;
    std::unordered_set<Function*> on_stack;

    struct DFSFrame {
        std::shared_ptr<Function> node;
        size_t child_idx;
    };
    std::vector<DFSFrame> stack;

    for (auto* root : roots) {
        if (!root || !root->grad_fn()) continue;
        auto root_fn = root->grad_fn();
        if (visited.count(root_fn.get())) continue;

        stack.push_back({root_fn, 0});
        visited.insert(root_fn.get());
        on_stack.insert(root_fn.get());

        while (!stack.empty()) {
            auto& frame = stack.back();
            const auto& children = frame.node->next_functions();

            if (frame.child_idx < children.size()) {
                auto& child = children[frame.child_idx];
                frame.child_idx++;

                if (!child) continue;
                if (on_stack.count(child.get())) {
                    throw AutogradException("Cycle detected in computation graph");
                }
                if (visited.count(child.get())) continue;

                visited.insert(child.get());
                on_stack.insert(child.get());
                stack.push_back({child, 0});
            } else {
                // All children processed — post-order visit
                on_stack.erase(frame.node.get());
                sorted.push_back(frame.node);
                stack.pop_back();
            }
        }
    }

    // Seed gradient accumulators for root grad_fns
    for (size_t i = 0; i < roots.size(); ++i) {
        if (roots[i] && roots[i]->grad_fn()) {
            accumulate_grad(roots[i]->grad_fn().get(), gradients[i]);
        }
    }

    auto cleanup_graph = [&]() {
        if (!retain_graph) {
            for (auto& func : sorted) {
                if (func) {
                    func->set_input_variables({});
                    func->set_next_functions({});
                }
            }
            for (auto* root : roots) {
                if (root && root->grad_fn() && !root->is_leaf()) {
                    root->set_grad_fn(nullptr);
                }
            }
        }
    };

    // RAII guards for exception-safe cleanup
    ScopeGuard accum_guard_multi{[&]{ grad_accumulators_ = std::move(saved_accumulators); }};
    ScopeGuard cleanup_guard_multi{[&]{ clear_gradients(); cleanup_graph(); }};

    {
        // Execute backward in reverse topological order
        for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
            if (!*it) continue;
            auto& function = *it;

            std::vector<Tensor> grad_outputs;
            const auto& accum_grads = get_accumulated_grads(function.get());
            if (accum_grads.empty()) {
                continue;  // No gradient flows to this function
            }

            // Sum all accumulated gradients with dtype promotion.
            // Pre-compute target dtype to avoid repeated conversions.
            DType target_dtype = accum_grads[0].dtype();
            for (size_t i = 1; i < accum_grads.size(); ++i) {
                target_dtype = promote_types(target_dtype, accum_grads[i].dtype());
            }
            Tensor total_grad = (accum_grads[0].dtype() == target_dtype)
                ? accum_grads[0]
                : accum_grads[0].to(target_dtype);
            for (size_t i = 1; i < accum_grads.size(); ++i) {
                const Tensor& gi = accum_grads[i];
                total_grad = total_grad + (gi.dtype() == target_dtype ? gi : gi.to(target_dtype));
            }
            grad_outputs.push_back(total_grad);

            function->reload_saved_tensors();
            function->validate_saved_tensors();
            auto input_grads = function->backward(grad_outputs);

            // Check for NaN/Inf in computed gradients when anomaly detection is on
            check_for_anomaly(input_grads, function.get());

            // Validate gradient dtypes are floating-point or complex
            for (size_t i = 0; i < input_grads.size(); ++i) {
                const auto& g = input_grads[i];
                if (g.is_valid() && g.numel() > 0 &&
                    !g.is_floating_point() && !g.is_complex()) {
                    throw std::runtime_error(
                        std::string("Backward function ") + function->name() +
                        " returned non-floating-point gradient (dtype=" +
                        std::string(dtype_name(g.dtype())) + " at index " +
                        std::to_string(i) +
                        "). Gradients must be floating-point or complex.");
                }
            }

            // Release saved tensors to free memory — only if we're not retaining the graph
            if (!retain_graph) {
                function->release_saved_tensors();
            }

            // Accumulate gradients to input variables
            auto& input_vars = function->input_variables();
            for (size_t i = 0; i < input_vars.size() && i < input_grads.size(); ++i) {
                Variable& var = input_vars[i];
                if (!var.requires_grad()) continue;

                Tensor grad_to_apply = input_grads[i];

                if (var.impl_) {
                    std::map<size_t, std::function<Tensor(const Tensor&)>> hooks_copy;
                    {
                        std::shared_lock lock(var.impl_->hooks_mutex_);
                        if (!var.impl_->hooks_.empty()) {
                            hooks_copy = var.impl_->hooks_;
                        }
                    }
                    for (auto& [id, hook] : hooks_copy) {
                        grad_to_apply = hook(grad_to_apply);
                    }
                }

                if (var.is_leaf() || var.retains_grad()) {
                    auto accumulate = [&]() {
                        if (var.has_grad()) {
                            auto existing_grad = var.grad().value();
                            if (grad_to_apply.dtype() != existing_grad.dtype()) {
                                DType target = promote_types(grad_to_apply.dtype(), existing_grad.dtype());
                                existing_grad = existing_grad.to(target);
                                grad_to_apply = grad_to_apply.to(target);
                            }
                            var.set_grad(existing_grad + grad_to_apply);
                        } else {
                            var.set_grad(grad_to_apply);
                        }
                    };

                    if (var.impl_) {
                        // Always acquire mutex to prevent TOCTOU race:
                        // thread_safe_ could be set between check and lock acquisition
                        std::lock_guard lock(var.impl_->grad_mutex_);
                        accumulate();
                    } else {
                        accumulate();
                    }
                }
            }

            const auto& next_funcs = function->next_functions();
            for (size_t i = 0; i < next_funcs.size() && i < input_grads.size(); ++i) {
                if (next_funcs[i]) {
                    accumulate_grad(next_funcs[i].get(), input_grads[i]);
                }
            }
        }
    }
    // ScopeGuards handle cleanup in both normal and exception paths
}

// Thread-local engine -- each thread gets its own instance so concurrent
// backward passes don't corrupt the shared grad_accumulators_ map.
auto backward_engine() -> BackwardEngine& {
    static thread_local BackwardEngine engine;
    return engine;
}

} // namespace tenzor
