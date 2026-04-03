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
