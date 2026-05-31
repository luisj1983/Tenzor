/**
 * @file autograd_wrap.hpp
 * @brief Helpers to swap a Variable's underlying tensor without severing
 *        its autograd graph.
 *
 * The recurring footgun this header eliminates is:
 *
 * @code
 *     // BUG: silently drops grad_fn — backward dead-ends here.
 *     v = Variable(v.tensor().to(other_device), v.requires_grad());
 * @endcode
 *
 * `Variable(Tensor, bool)` is the leaf constructor: it produces a fresh
 * Variable with `grad_fn == nullptr` regardless of what the source Variable
 * carried. Used inside a layer's forward path (cross-device parameter move,
 * dtype promotion, contiguity normalisation), the pattern severs the
 * gradient chain and parent leaves silently receive zero gradients.
 *
 * The replacement is `Variable::set_data_view`, which mutates `data_` in
 * place while keeping `grad_fn_`, `grad_`, `requires_grad_`, hooks, and the
 * rest of the autograd state intact. These helpers wrap that intent so
 * call sites read clearly and so a future change to the underlying impl
 * has a single hook point.
 */

#pragma once

#include <utility>

#include "tenzor/autograd/variable.hpp"
#include "tenzor/core/tensor.hpp"

namespace tenzor::utils {

/**
 * @brief Replace @p v's underlying tensor with @p new_data while preserving
 *        its grad_fn, requires_grad, hooks, and the rest of its autograd
 *        state.
 *
 * Use this whenever a layer's forward path needs to change a parameter or
 * intermediate Variable's tensor — device transfer, dtype promotion,
 * contiguity normalisation — without breaking the gradient graph that
 * built it.
 *
 * @warning The new tensor must remain compatible with downstream autograd
 *          machinery (same dtype/shape unless the layer compensates).
 */
inline void wrap_preserving_grad(Variable& v, Tensor new_data) {
    v.set_data_view(std::move(new_data));
}


} // namespace tenzor::utils
