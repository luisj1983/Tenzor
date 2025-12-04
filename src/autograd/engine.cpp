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

    // Debug: Check initial gradient for Float16
    if (root.tensor().dtype() == DType::Float16) {
        std::cerr << "[ENGINE_F16] Initial gradient created" << std::endl;
        std::cerr << "[ENGINE_F16] Root dtype: " << static_cast<int>(root.tensor().dtype()) << std::endl;
        std::cerr << "[ENGINE_F16] Gradient dtype: " << static_cast<int>(gradient->dtype()) << std::endl;
        std::cerr << "[ENGINE_F16] Gradient shape: [";
        for (size_t i = 0; i < gradient->shape().size(); ++i) {
            if (i > 0) std::cerr << ", ";
            std::cerr << gradient->shape()[i];
        }
        std::cerr << "]" << std::endl;

        // Check first few gradient values (copy to CPU first if on GPU)
        if (gradient->dtype() == DType::Float16) {
            std::cerr << "[ENGINE_F16] First 5 gradient values: ";
            // Copy gradient to CPU for safe access
            Tensor cpu_grad = gradient->to(Device::cpu());
            auto* data = cpu_grad.data<Float16>();
            for (int i = 0; i < std::min(5, static_cast<int>(cpu_grad.numel())); ++i) {
                std::cerr << static_cast<float>(data[i]) << " (bits=0x" << std::hex << data[i].bits << std::dec << ") ";
            }
            std::cerr << std::endl;
        }
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

        // Debug: Check if gradients are computed for Float16 or Float64
        if (!grad_outputs.empty() && (grad_outputs[0].dtype() == DType::Float16 || grad_outputs[0].dtype() == DType::Float64)) {
            // Check for NaN in input_grads
            bool has_nan = false;
            for (size_t i = 0; i < input_grads.size(); ++i) {
                if (input_grads[i].numel() > 0) {
                    auto grad_cpu = input_grads[i].to(Device::cpu()).to(DType::Float32);
                    auto* data = grad_cpu.data<float>();
                    for (int j = 0; j < std::min(10, static_cast<int>(grad_cpu.numel())); ++j) {
                        if (std::isnan(data[j])) {
                            has_nan = true;
                            break;
                        }
                    }
                }
                if (has_nan) break;
            }

            // Only print if NaN detected or it's the first few backward calls
            static int backward_count = 0;
            static bool first_nan_in_output = false;
            backward_count++;

            // Check if grad_output has NaN (the input to this backward)
            bool go_has_nan = false;
            {
                auto go_cpu = grad_outputs[0].to(Device::cpu()).to(DType::Float32);
                auto* go_data = go_cpu.data<float>();
                for (int j = 0; j < std::min(100, static_cast<int>(go_cpu.numel())); ++j) {
                    if (std::isnan(go_data[j])) {
                        go_has_nan = true;
                        break;
                    }
                }
            }

            // First backward that produces NaN when grad_output was clean
            if (has_nan && !go_has_nan && !first_nan_in_output) {
                first_nan_in_output = true;
                std::cerr << "[ENGINE_BACKWARD] FIRST NaN CREATED at backward #" << backward_count
                          << " (dtype=" << static_cast<int>(grad_outputs[0].dtype())
                          << "), input_grads.size=" << input_grads.size()
                          << ", func_type=" << typeid(*function).name() << "\n";
                std::cerr << "  grad_output: shape=[";
                for (size_t i = 0; i < grad_outputs[0].shape().size(); ++i) {
                    if (i > 0) std::cerr << ",";
                    std::cerr << grad_outputs[0].shape()[i];
                }
                std::cerr << "] numel=" << grad_outputs[0].numel() << "\n";
                // Print input_grads shapes
                for (size_t i = 0; i < input_grads.size(); ++i) {
                    std::cerr << "  input_grads[" << i << "]: shape=[";
                    for (size_t j = 0; j < input_grads[i].shape().size(); ++j) {
                        if (j > 0) std::cerr << ",";
                        std::cerr << input_grads[i].shape()[j];
                    }
                    std::cerr << "] numel=" << input_grads[i].numel() << "\n";
                }
            }
            // Check for inf in grad_outputs
            bool go_has_inf = false;
            {
                auto go_cpu = grad_outputs[0].to(Device::cpu()).to(DType::Float32);
                auto* go_data = go_cpu.data<float>();
                for (int j = 0; j < std::min(100, static_cast<int>(go_cpu.numel())); ++j) {
                    if (std::isinf(go_data[j])) {
                        go_has_inf = true;
                        break;
                    }
                }
            }

            // Check for inf in output
            bool out_has_inf = false;
            for (size_t i = 0; i < input_grads.size(); ++i) {
                if (input_grads[i].numel() > 0) {
                    auto grad_cpu = input_grads[i].to(Device::cpu()).to(DType::Float32);
                    auto* data = grad_cpu.data<float>();
                    for (int j = 0; j < std::min(100, static_cast<int>(grad_cpu.numel())); ++j) {
                        if (std::isinf(data[j])) {
                            out_has_inf = true;
                            break;
                        }
                    }
                }
                if (out_has_inf) break;
            }

            // Print on first inf creation
            static bool first_inf_created = false;
            if (!go_has_inf && out_has_inf && !first_inf_created) {
                first_inf_created = true;
                std::cerr << "[ENGINE_BACKWARD] FIRST INF CREATED at backward #" << backward_count
                          << " (dtype=" << static_cast<int>(grad_outputs[0].dtype())
                          << "), func=" << typeid(*function).name()
                          << ", go_shape=[";
                for (size_t i = 0; i < grad_outputs[0].shape().size(); ++i) {
                    if (i > 0) std::cerr << ",";
                    std::cerr << grad_outputs[0].shape()[i];
                }
                std::cerr << "]" << std::endl;
            }

            if (backward_count <= 15 || (has_nan && !go_has_nan)) {
                std::cerr << "[ENGINE_BACKWARD] backward #" << backward_count
                          << " (dtype=" << static_cast<int>(grad_outputs[0].dtype())
                          << "), go_has_nan=" << go_has_nan
                          << ", output_has_nan=" << has_nan
                          << ", go_has_inf=" << go_has_inf
                          << ", out_has_inf=" << out_has_inf
                          << ", input_grads.size=" << input_grads.size()
                          << ", func=" << typeid(*function).name();
                // Print grad_output shape
                std::cerr << ", go_shape=[";
                for (size_t i = 0; i < grad_outputs[0].shape().size(); ++i) {
                    if (i > 0) std::cerr << ",";
                    std::cerr << grad_outputs[0].shape()[i];
                }
                std::cerr << "]" << std::endl;
            }
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

            // Debug: Check gradient before accumulation
            if ((var.tensor().dtype() == DType::Float16 || var.tensor().dtype() == DType::Float64) && var.is_leaf()) {
                std::cerr << "[ENGINE_LEAF] Accumulating to leaf variable, dtype=" << static_cast<int>(var.tensor().dtype())
                          << ", is_leaf=" << var.is_leaf()
                          << ", has_grad=" << var.has_grad() << std::endl;

                // Debug: Check input_grad values
                auto grad_cpu = grad_to_apply.to(Device::cpu()).to(DType::Float32);
                auto* data = grad_cpu.data<float>();
                float sum = 0.0f;
                int count = std::min(10, static_cast<int>(grad_cpu.numel()));
                for (int j = 0; j < count; ++j) {
                    sum += std::abs(data[j]);
                }
                std::cerr << "[ENGINE_LEAF] Input grad avg_abs_first10=" << (sum / count) << std::endl;
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
                    // Ensure incoming gradient matches the dtype of existing gradient
                    auto existing_grad = var.grad().value();
                    if (grad_to_apply.dtype() != existing_grad.dtype()) {
                        grad_to_apply = grad_to_apply.to(existing_grad.dtype());
                    }
                    var.grad() = existing_grad + grad_to_apply;
                } else {
                    var.grad() = grad_to_apply;
                }

                // Debug: Check accumulated gradient
                if ((var.tensor().dtype() == DType::Float16 || var.tensor().dtype() == DType::Float64) && var.is_leaf() && var.has_grad()) {
                    auto grad_cpu = var.grad()->to(Device::cpu()).to(DType::Float32);
                    auto* data = grad_cpu.data<float>();
                    float sum = 0.0f;
                    int count = std::min(10, static_cast<int>(grad_cpu.numel()));
                    for (int j = 0; j < count; ++j) {
                        sum += std::abs(data[j]);
                    }
                    std::cerr << "[ENGINE_LEAF] After accumulation, avg_abs_first10=" << (sum / count)
                              << ", dtype=" << static_cast<int>(var.tensor().dtype()) << std::endl;
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
