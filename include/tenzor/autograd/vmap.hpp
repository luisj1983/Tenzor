/**
 * @file vmap.hpp
 * @brief Vectorized map (vmap) transform
 *
 * Provides vmap, which applies a function independently to each element
 * along a batch dimension, equivalent to torch.vmap.
 */

#pragma once

#include "variable.hpp"
#include <functional>
#include <cstdint>

namespace tenzor {

/**
 * @brief Vectorized map: apply func independently to each element along batch dim.
 *
 * Equivalent to looping over the batch dimension, applying func to each slice,
 * and stacking the results. This is the "loop-and-stack" fallback implementation.
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
