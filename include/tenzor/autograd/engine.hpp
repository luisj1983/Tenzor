/**
 * @file engine.hpp
 * @brief Backward pass execution engine for automatic differentiation
 *
 * Implements the engine that executes backpropagation through the
 * computation graph, computing gradients for all variables.
 */

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include "../core/tensor.hpp"
#include "variable.hpp"
#include "function.hpp"
#include "graph.hpp"

namespace tenzor {

/**
 * @brief Execution engine for backward pass gradient computation.
 *
 * BackwardEngine orchestrates the backpropagation process through a
 * computation graph. It performs topological sorting to determine the
 * correct order of gradient computation and handles gradient accumulation
 * for variables with multiple paths in the graph.
 *
 * Key features:
 * - Automatic topological sorting of computation graph
 * - Gradient accumulation for multi-path graphs
 * - Support for single and multi-root backward passes
 * - Efficient gradient clearing and memory management
 *
 * The engine is typically accessed through Variable::backward() rather
 * than directly.
 *
 * @code
 * // Indirect usage through Variable
 * Variable x(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * Variable y = x * 2.0f + 1.0f;
 * Variable loss = y.sum();
 * loss.backward();  // Internally uses BackwardEngine
 *
 * // Direct usage (advanced)
 * auto& engine = backward_engine();
 * engine.execute(loss, std::nullopt);
 * @endcode
 *
 * @note Higher-order gradients (e.g., second-order derivatives) are supported
 * via the create_graph parameter. When create_graph=true, backward operations
 * use Variable ops instead of raw Tensor ops, building a new computation graph
 * that enables computing gradients of gradients (useful for WGAN-GP, MAML,
 * Hessian computation).
 *
 * @see Variable::backward() for user-facing API
 * @see Function for gradient function interface
 */
class BackwardEngine {
public:
    /**
     * @brief Default constructor.
     */
    BackwardEngine() = default;

    /**
     * @brief Execute backward pass through computation graph.
     *
     * Computes gradients for all variables in the computation graph
     * leading to the root variable. Performs topological sort to
     * determine execution order and accumulates gradients for variables
     * with multiple incoming paths.
     *
     * @param root Root variable to compute gradients from
     * @param gradient Optional gradient tensor (required for non-scalar root)
     * @param retain_graph If true, keep computation graph for multiple backward passes
     *
     * @throws std::runtime_error if gradient is required but not provided
     *
     * @code
     * Variable x(Tensor({3}, DType::Float32, Device::cpu()), true);
     * Variable y = x * 2.0f;
     * Variable loss = y.sum();  // Scalar
     *
     * backward_engine().execute(loss, std::nullopt, false);
     * // x.grad() now contains computed gradients
     * @endcode
     */
    auto execute(Variable& root, std::optional<Tensor> gradient,
                 bool retain_graph = false, bool create_graph = false) -> void;

    /**
     * @brief Execute backward pass for multiple roots.
     *
     * Computes gradients when there are multiple output variables.
     * Each root must have a corresponding gradient provided.
     *
     * @param roots Vector of root variables
     * @param gradients Vector of gradient tensors (one per root)
     *
     * @throws std::runtime_error if roots and gradients size mismatch
     *
     * @code
     * Variable loss1 = compute_loss1(x);
     * Variable loss2 = compute_loss2(x);
     *
     * backward_engine().execute_multi(
     *     {&loss1, &loss2},
     *     {ones_like(loss1.tensor()), ones_like(loss2.tensor())}
     * );
     * @endcode
     */
    auto execute_multi(std::vector<Variable*> roots,
                      std::vector<Tensor> gradients) -> void;

    /**
     * @brief Clear gradient accumulation buffers.
     *
     * Clears all accumulated gradients stored internally by the engine.
     * Call this between backward passes to free memory and reset state.
     *
     * @note This does not clear gradients in Variables themselves.
     * Use Variable::zero_grad() for that.
     */
    auto clear_gradients() -> void;

private:
    /**
     * @brief Perform topological sort of computation graph.
     *
     * Traverses the computation graph from root to leaves, producing
     * a topologically sorted order for gradient computation. Ensures
     * that each function is executed only after all its dependencies.
     *
     * @param root Root function node to start traversal from
     * @return Vector of functions in reverse topological order
     */
    auto topological_sort(std::shared_ptr<Function> root)
        -> std::vector<std::shared_ptr<Function>>;

    /**
     * @brief Gradient accumulation buffers for multi-path graphs.
     *
     * Maps each function to accumulated gradients from all paths.
     * Used when a variable is used multiple times in the computation.
     */
    std::unordered_map<Function*, std::vector<Tensor>> grad_accumulators_;

    /**
     * @brief Accumulate gradient for a function.
     *
     * Adds a gradient to the accumulator for the given function.
     * Used when multiple paths contribute gradients to the same variable.
     *
     * @param func Function to accumulate gradient for
     * @param grad Gradient tensor to accumulate
     */
    auto accumulate_grad(Function* func, Tensor grad) -> void;

    /**
     * @brief Get all accumulated gradients for a function.
     *
     * Retrieves all gradient contributions that have been accumulated
     * for a function from different paths in the computation graph.
     *
     * @param func Function to get gradients for
     * @return Vector of accumulated gradient tensors
     */
    auto get_accumulated_grads(Function* func) -> std::vector<Tensor>;
};

/**
 * @brief Get global backward engine instance.
 *
 * Returns a reference to the singleton backward engine instance.
 * The engine persists across calls and maintains state.
 *
 * @return Reference to global backward engine
 *
 * @code
 * auto& engine = backward_engine();
 * engine.clear_gradients();  // Clear state between uses
 * @endcode
 */
auto backward_engine() -> BackwardEngine&;

} // namespace tenzor
