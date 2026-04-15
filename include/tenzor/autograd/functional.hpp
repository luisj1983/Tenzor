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
 * @brief Compute Jacobian-Vector Product (forward-mode AD).
 *
 * Given a function f, input x, and tangent vector v, computes:
 *   (f(x), J_f(x) * v)
 * where J_f is the Jacobian of f at x.
 *
 * @param func Differentiable function from Variable to Variable
 * @param input Point at which to evaluate
 * @param tangent Direction vector for the JVP
 * @return Pair of (output, tangent_output)
 */
auto jvp(std::function<Variable(const Variable&)> func,
         const Variable& input,
         const Tensor& tangent) -> std::pair<Variable, Tensor>;

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
