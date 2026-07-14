/**
 * @file vmap.hpp
 * @brief Vectorized map (vmap) transform with batching rules
 *
 * Provides vmap, which applies a function independently to each element
 * along a batch dimension. When batching rules are available, operations
 * are executed as single batched calls (e.g., matmul -> bmm) instead of
 * the naive loop-and-stack fallback.
 */

#pragma once

#include "variable.hpp"
#include "../ops/op_id.hpp"
#include "../backend/op_attributes.hpp"
#include <functional>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace tenzor {

/**
 * @brief Type for a batching rule function.
 *
 * A batching rule transforms a per-element function into an equivalent
 * batched operation. Given the original function, batched input, and
 * batch dimension, it returns the batched result directly without
 * slicing and stacking.
 */
using BatchingRule = std::function<Variable(
    const std::function<Variable(const Variable&)>&,
    const Variable&,
    int64_t)>;

/**
 * @brief Register a batching rule for an operation.
 *
 * @param op_name Name of the operation (matched against grad_fn name)
 * @param rule Batching rule function
 */
void register_batching_rule(const std::string& op_name, BatchingRule rule);

/**
 * @brief Check if a batching rule exists for an operation.
 */
auto has_batching_rule(const std::string& op_name) -> bool;

/**
 * @brief OpId-keyed registration / lookup.
 *
 * Audit A.3: vmap rules can now be keyed by the canonical forward
 * `OpId` rather than the `Function::name()` string. The OpId-keyed
 * registry is consulted first; the legacy name-based registry remains
 * the fallback for un-opted-in Function subclasses (audit A.2 covers
 * ~128 / 158 subclasses, which is enough to make the OpId path the
 * common case while the long tail stays on the string path).
 */
void register_batching_rule(OpId op_id, BatchingRule rule);
auto has_batching_rule(OpId op_id) -> bool;

/**
 * @brief Factory: build a batching rule for ops that reduce/operate along a
 *        single `dim` attribute (Sum/Mean/Prod/Var/Std/Max/Min/TopK/Sort/…).
 *
 * Audit J.1. The legacy passthrough rule called `func(batched_input)`
 * unchanged, which is incorrect for any op carrying a `dim` argument: when
 * vmap prepends a batch axis, the user's `dim` refers to a position in the
 * *unbatched* view, not in the batched tensor we hand to the kernel.
 *
 * AUTOGRAD-R037: an earlier version of this rule fixed that by probing the
 * user function on a single slice to recover the forward `OpId`/attributes,
 * shifting the saved `dim` past the batch axis, and redispatching directly
 * on the full batched input for an O(1) call. That fast path was REMOVED
 * (see the "graph-severing fix" audit note in vmap.cpp) because rewrapping
 * the raw dispatch result as `Variable(output, requires_grad)` carried no
 * `grad_fn`, silently severing autograd for every dim-carrying op routed
 * through this rule. The returned rule now unconditionally delegates to
 * `vmap_loop_and_stack`, which calls the user's `func` once per batch slice
 * (so `dim_attr_key` never needs to be read or shifted — each slice already
 * sees the correct unbatched axis) and stitches the per-slice results back
 * together with Variable-level `unsqueeze`/`cat`, preserving the grad_fn
 * chain at the cost of O(B) sequential dispatches instead of O(1).
 *
 * @param dim_attr_key Unused by the current implementation; retained in the
 *                     signature for source compatibility with existing
 *                     registration call sites (all of which pass
 *                     `AttrKey::Dim`).
 */
auto dim_shifted_passthrough(AttrKey dim_attr_key) -> BatchingRule;

/**
 * @brief Initialize built-in batching rules.
 *
 * Registers rules for common operations: element-wise ops (passthrough),
 * matmul (via bmm), linear, conv2d, softmax, etc.
 * Called automatically on first vmap invocation.
 */
void init_builtin_batching_rules();

/**
 * @brief Vectorized map: apply func independently to each element along batch dim.
 *
 * First attempts to detect the operation and apply a batching rule for
 * efficient execution. Falls back to loop-and-stack if no rule matches.
 *
 * @param func Function to apply to each batch element
 * @param batched_input Input variable with a batch dimension
 * @param batch_dim Dimension to vectorize over (default: 0)
 * @return Variable with results stacked along batch_dim
 */
auto vmap(std::function<Variable(const Variable&)> func,
          const Variable& batched_input,
          int64_t batch_dim = 0) -> Variable;

} // namespace tenzor
