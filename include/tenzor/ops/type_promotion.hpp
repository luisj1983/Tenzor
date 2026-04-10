/**
 * @file type_promotion.hpp
 * @brief NumPy/PyTorch-compatible automatic type promotion for binary operations
 */

#pragma once

#include "../core/dtype.hpp"
#include "../core/tensor.hpp"

namespace tenzor {

// promote_types() moved to include/tenzor/core/dtype.hpp as a constexpr
// function so any layer can use it without pulling in ops/. It is re-exported
// here via the core header include above; existing code continues to work
// unchanged.

/**
 * @brief Determine the result dtype for a binary operation on two tensors.
 */
auto result_type(const Tensor& a, const Tensor& b) -> DType;

/**
 * @brief Promote two tensors to a common dtype, converting if necessary.
 * @return Pair of tensors with matching dtype (may be the originals if already matching)
 */
auto promote_inputs(const Tensor& a, const Tensor& b) -> std::pair<Tensor, Tensor>;

} // namespace tenzor
