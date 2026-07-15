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
#include <unordered_set>
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
 * @threadsafety The engine is a thread-local singleton (one instance per
 * thread), so concurrent backward() calls from different threads do not
 * interfere with each other. A single backward() call per thread is
 * supported at a time. Re-entrant calls within the same thread (e.g.,
 * from gradient checkpointing) are safe because execute() saves and
 * restores grad_accumulators_ around each invocation.
 *
 * ## Lock ordering and shared parameters
 *
 * Gradient accumulation to leaf Variables uses per-variable grad_mutex_
 * (see VariableImpl::grad_mutex_). The engine acquires at most ONE
 * grad_mutex_ at any point during backward traversal -- it locks, writes,
 * and releases before moving to the next variable. This single-lock-at-a-
 * time design makes deadlock structurally impossible regardless of graph
 * topology or parameter sharing patterns.
 *
 * When multiple threads perform backward() concurrently and share
 * parameters (e.g., data-parallel training), the per-variable grad_mutex_
 * serializes their gradient accumulations. Because each thread holds only
 * one mutex at a time, no lock ordering is required.
 *
 * The hooks_mutex_ on each Variable is a shared_mutex used only for
 * reading the hook map (shared_lock). It is never held simultaneously
 * with another Variable's hooks_mutex_ or grad_mutex_, so it does not
 * participate in any lock ordering.
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
                      std::vector<Tensor> gradients,
                      bool retain_graph = false,
                      bool create_graph = false) -> void;

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
     * @brief Classify which Functions in @p sorted are safe to fully
     * release (saved tensors, saved Variables, op-specific state,
     * input_variables()/next_functions()) once this backward() call
     * finishes with them.
     *
     * CR3: a Function reachable from one root's walk may also be reachable
     * from a DIFFERENT, independently-alive Variable's graph -- e.g. two
     * sibling Variables built from a shared non-leaf intermediate
     * (`a = f(y); b = g(y);` both have `y`'s Function in their
     * next_functions()). Unconditionally releasing every Function in
     * `sorted` (the pre-fix behavior, applied both mid-loop right after
     * each Function's own backward() runs and again in cleanup_graph)
     * corrupts that other graph's state out from under it.
     *
     * Function::parent_count() is a global, atomic count of how many
     * Functions currently list a given Function in their next_functions().
     * This computes each node's LOCAL in-degree using only edges between
     * functions that are themselves in `sorted`. If a node's local
     * in-degree equals its global parent_count(), every parent of that
     * node is also part of this same call, so no external graph can be
     * relying on it and it is safe to release. If parent_count() is
     * larger, an external parent exists outside `sorted` and the node's
     * state must be left untouched so that graph's later backward() still
     * works.
     *
     * Computed once, immediately after topological_sort() builds `sorted`
     * and before any release call runs anywhere in this backward() call --
     * both to snapshot parent_count() before cleanup_graph's
     * set_next_functions({}) calls start decrementing it live, and so the
     * SAME classification gates every release site consistently (a
     * Function must not be released mid-loop by one site and preserved by
     * another).
     *
     * @param sorted Topologically-sorted functions reachable from one root
     *               (as produced by topological_sort()); may contain
     *               multiple roots' reachable sets when called from
     *               execute_multi().
     * @return Set of raw Function pointers (into `sorted`'s shared_ptrs,
     *         valid for the lifetime of this backward() call) that have no
     *         parent outside `sorted` and are therefore safe to release.
     */
    static auto compute_unshared_functions(
        const std::vector<std::shared_ptr<Function>>& sorted)
        -> std::unordered_set<Function*>;

    /**
     * @brief Gradient accumulation buffers for multi-path graphs.
     *
     * Maps each function's unique ID to accumulated gradients from all paths.
     * Uses uint64_t IDs instead of raw Function* pointers to avoid potential
     * issues with address reuse after deallocation.
     */
    std::unordered_map<uint64_t, std::vector<Tensor>> grad_accumulators_;

    /**
     * @brief Graph-carrying gradient accumulation buffers (create_graph only).
     *
     * Parallel to grad_accumulators_ but stores the Variable form of each
     * gradient contribution so that, under create_graph=true, a downstream
     * Function's backward_with_variables receives grad_outputs whose grad_fn
     * still chains back to the input. Without this the engine extracts raw
     * tensors from var_input_grads and re-wraps them detached (Variable(g,
     * true)) before handing them to the next Function — severing the second-
     * order graph at every Function-to-Function gradient hand-off. That
     * severing made dependence-through-the-gradient (as opposed to dependence
     * through a saved tensor) invisible to higher-order AD, e.g. d²/dx²(x³)
     * came out 4x instead of 6x. Only populated when create_graph is active.
     */
    std::unordered_map<uint64_t, std::vector<Variable>> grad_accumulators_var_;

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
     * @brief Accumulate a graph-carrying gradient Variable for a function.
     *
     * Companion to accumulate_grad used only under create_graph=true so the
     * second-order graph stays connected across Function hand-offs.
     */
    auto accumulate_grad_var(Function* func, Variable grad) -> void;

    /**
     * @brief Build the graph-carrying grad_outputs for a Function under
     *        create_graph=true, summing the Variable accumulator entries.
     */
    auto build_var_grad_outputs(Function* func,
                                const std::vector<Tensor>& raw_grad_outputs)
        -> std::vector<Variable>;

    /**
     * @brief Get all accumulated gradients for a function.
     *
     * Retrieves all gradient contributions that have been accumulated
     * for a function from different paths in the computation graph.
     *
     * @param func Function to get gradients for
     * @return Vector of accumulated gradient tensors
     */
    auto get_accumulated_grads(Function* func) -> const std::vector<Tensor>&;

    /**
     * @brief Validate or synthesize the root gradient seed for a backward pass.
     *
     * Shared by `execute()` and `execute_multi()` (audit-6 BB.2). If the user
     * did not provide a gradient and the root is scalar (numel == 1), returns
     * `ones_like(root.tensor())`. If the user did not provide a gradient and
     * the root is non-scalar, throws AutogradException. Otherwise validates
     * that the supplied gradient's shape matches the root's shape.
     *
     * @param root Root variable being backproped.
     * @param user_grad Caller-supplied gradient, if any.
     * @return The validated or synthesized gradient tensor.
     */
    auto synthesize_or_validate_root_grad(const Variable& root,
                                          std::optional<Tensor> user_grad) -> Tensor;
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
