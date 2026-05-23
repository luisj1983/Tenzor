/**
 * @file functional.hpp
 * @brief Composable functional transforms for automatic differentiation
 *
 * Provides higher-order functions for computing Jacobian-vector products (JVP),
 * full Jacobians, and Hessians using forward-mode and reverse-mode AD.
 */

#pragma once

#include "variable.hpp"
#include "dual.hpp"
#include "../core/tensor.hpp"
#include <functional>
#include <optional>
#include <utility>

namespace tenzor {

/**
 * @brief Strategy selector for jvp().
 *
 *  - `Walker`: build the autograd graph by calling `func(input)`, then walk
 *    the resulting `grad_fn` chain in reverse topological order invoking
 *    `dispatch_jvp` per node. This is the existing default and works for any
 *    composed `func` whose nodes have registered JVP rules.
 *  - `Dual`: honour the `is_dual_mode()` TLS flag while invoking `func`. The
 *    flag is *set* for the duration of the call so future per-op interceptors
 *    can short-circuit the build-then-walk pipeline (see the design note in
 *    `jvp_dispatch.hpp`). For ops whose Variable wrapper has not yet been
 *    interceptor-converted, behaviour is identical to `Walker`. No tangent
 *    information is lost: the function still receives a normal Variable
 *    primal and the walker remains the source of truth.
 */
enum class JvpMode {
    Walker = 0,
    Dual   = 1,
};

/**
 * @brief Compute Jacobian-Vector Product (forward-mode AD).
 *
 * Given a function f, input x, and tangent vector v, computes:
 *   (f(x), J_f(x) * v)
 * where J_f is the Jacobian of f at x.
 *
 * @param func Differentiable function from Variable to Variable
 * @param input Point at which to evaluate
 * @param tangent Direction vector for the JVP
 * @param mode    Strategy (default: walker)
 * @return Pair of (output, tangent_output)
 */
auto jvp(std::function<Variable(const Variable&)> func,
         const Variable& input,
         const Tensor& tangent,
         JvpMode mode = JvpMode::Walker) -> std::pair<Variable, Tensor>;

/**
 * @brief Compute full Jacobian matrix of func at input.
 *
 * For f: R^n -> R^m, returns the m x n Jacobian matrix.
 * Uses forward-mode AD when n <= m (narrow outputs),
 * reverse-mode AD when m < n (wide outputs).
 *
 * @param func Differentiable function from Variable to Variable
 * @param input Point at which to evaluate the Jacobian
 * @return Jacobian matrix tensor of shape (output_size, input_size)
 */
auto jacobian(std::function<Variable(const Variable&)> func,
              const Variable& input) -> Tensor;

/**
 * @brief Compute Hessian matrix (second derivatives) of a scalar function.
 *
 * For f: R^n -> R, returns the n x n Hessian matrix.
 * Uses forward-over-reverse: computes the Jacobian of the gradient.
 *
 * @param func Scalar-valued differentiable function
 * @param input Point at which to evaluate the Hessian
 * @return Hessian matrix tensor of shape (input_size, input_size)
 */
auto hessian(std::function<Variable(const Variable&)> func,
             const Variable& input) -> Tensor;

/**
 * @brief Compute Hessian-Vector Product using forward-over-reverse mode.
 *
 * For f: R^n -> R with Hessian H, computes H @ v without materializing H.
 * Uses forward-over-reverse: JVP of the gradient function with tangent v.
 *
 * @param func Scalar-valued differentiable function
 * @param input Point at which to evaluate
 * @param v Vector to multiply (same shape as input)
 * @return Pair of (func_output, H @ v)
 */
auto hvp(std::function<Variable(const Variable&)> func,
         const Variable& input,
         const Tensor& v) -> std::pair<Variable, Tensor>;

/**
 * @brief Compute Vector-Hessian Product using reverse-over-reverse mode.
 *
 * For f: R^n -> R with Hessian H, computes v^T @ H without materializing H.
 * Uses reverse-over-reverse: backward through the gradient computation.
 *
 * @param func Scalar-valued differentiable function
 * @param input Point at which to evaluate
 * @param v Vector to multiply (same shape as input)
 * @return Pair of (func_output, v^T @ H)
 */
auto vhp(std::function<Variable(const Variable&)> func,
         const Variable& input,
         const Tensor& v) -> std::pair<Variable, Tensor>;

/**
 * @brief Compute Vector-Jacobian Product (reverse-mode AD).
 *
 * Given a function f, input x, and cotangent vector v, computes:
 *   (f(x), v^T @ J_f(x))
 * where J_f is the Jacobian of f at x.
 *
 * This is the dual of JVP and corresponds to a single backward pass
 * with the cotangent as the upstream gradient.
 *
 * @param func Differentiable function from Variable to Variable
 * @param input Point at which to evaluate
 * @param cotangent Cotangent (upstream gradient) vector
 * @return Pair of (output, v^T @ J_f(x))
 */
auto vjp(std::function<Variable(const Variable&)> func,
         const Variable& input,
         const Tensor& cotangent) -> std::pair<Variable, Tensor>;

} // namespace tenzor
