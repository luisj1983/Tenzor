/**
 * @file control_flow.hpp
 * @brief Control flow primitives for JIT-compatible code
 *
 * Provides cond() and while_loop() that work both in eager mode
 * and during JIT tracing. When tracing is active, they delegate
 * to the Tracer's trace_if/trace_loop to record control flow
 * in the IR graph. In eager mode, they execute directly.
 *
 * Usage:
 * @code
 * // Works in both eager and traced execution:
 * auto result = tenzor::jit::cond(
 *     pred,  // boolean scalar tensor
 *     [](const std::vector<Variable>& args) { return relu(args[0]); },
 *     [](const std::vector<Variable>& args) { return sigmoid(args[0]); },
 *     {x}
 * );
 *
 * auto final = tenzor::jit::while_loop(
 *     100,  // max iterations
 *     [](const std::vector<Variable>& state) {
 *         return state[1].tensor().item<float>() > 0.01f;  // condition
 *     },
 *     [](const std::vector<Variable>& state) {
 *         auto x = state[0];
 *         auto loss = state[1];
 *         return std::vector<Variable>{x * 0.9f, loss * 0.95f};
 *     },
 *     {x0, loss0}  // initial carried state
 * );
 * @endcode
 */

#pragma once

#include <functional>
#include <vector>
#include "../autograd/variable.hpp"
#include "../core/tensor.hpp"
#include "tracer.hpp"

namespace tenzor {
namespace jit {

/**
 * @brief Conditional execution compatible with JIT tracing.
 *
 * When tracing is active, records both branches as subgraphs
 * in the IR (via Tracer::trace_if). At graph execution time,
 * the condition determines which branch runs.
 *
 * When not tracing, evaluates the condition eagerly and calls
 * the appropriate branch function.
 *
 * @param condition Boolean scalar tensor (true/false)
 * @param then_fn Function executed when condition is true
 * @param else_fn Function executed when condition is false
 * @param args Input variables available to both branches
 * @return Outputs from the selected branch
 */
auto cond(const Tensor& condition,
          std::function<std::vector<Variable>(const std::vector<Variable>&)> then_fn,
          std::function<std::vector<Variable>(const std::vector<Variable>&)> else_fn,
          const std::vector<Variable>& args) -> std::vector<Variable>;

/**
 * @brief Single-output convenience overload.
 */
auto cond(const Tensor& condition,
          std::function<Variable(const Variable&)> then_fn,
          std::function<Variable(const Variable&)> else_fn,
          const Variable& input) -> Variable;

/**
 * @brief While loop compatible with JIT tracing.
 *
 * When tracing is active, records the loop body as a subgraph
 * in the IR (via Tracer::trace_loop). At graph execution time,
 * the loop runs up to max_iter times.
 *
 * When not tracing, executes the loop eagerly.
 *
 * @param max_iter Maximum number of iterations
 * @param cond_fn Returns a boolean tensor; loop continues while true
 * @param body_fn Computes one iteration: takes carried state, returns updated state
 * @param carried Initial carried state variables
 * @return Final carried state after loop completes
 */
auto while_loop(int64_t max_iter,
                std::function<Tensor(const std::vector<Variable>&)> cond_fn,
                std::function<std::vector<Variable>(const std::vector<Variable>&)> body_fn,
                const std::vector<Variable>& carried) -> std::vector<Variable>;

} // namespace jit
} // namespace tenzor
