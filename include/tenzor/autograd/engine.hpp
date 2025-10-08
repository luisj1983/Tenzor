#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include "../core/tensor.hpp"
#include "variable.hpp"
#include "function.hpp"
#include "graph.hpp"

namespace tenzor {

// Backward execution engine
class BackwardEngine {
public:
    BackwardEngine() = default;

    // Execute backward pass through computation graph
    auto execute(Variable& root, std::optional<Tensor> gradient) -> void;

    // Execute backward pass for multiple roots
    auto execute_multi(std::vector<Variable*> roots,
                      std::vector<Tensor> gradients) -> void;

    // Clear gradient accumulation
    auto clear_gradients() -> void;

private:
    // Topological sort of computation graph
    auto topological_sort(std::shared_ptr<Function> root)
        -> std::vector<std::shared_ptr<Function>>;

    // Gradient accumulation for multi-path graphs
    std::unordered_map<Function*, std::vector<Tensor>> grad_accumulators_;

    // Accumulate gradient
    auto accumulate_grad(Function* func, Tensor grad) -> void;

    // Get accumulated gradients
    auto get_accumulated_grads(Function* func) -> std::vector<Tensor>;
};

// Global backward engine
auto backward_engine() -> BackwardEngine&;

} // namespace tenzor
