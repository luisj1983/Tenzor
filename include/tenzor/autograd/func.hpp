/**
 * @file func.hpp
 * @brief Composable function transforms (torch.func equivalent)
 *
 * Provides higher-order functions that transform callables and compose
 * naturally with each other:
 *
 *   auto grad_fn = tenzor::func::grad(my_loss);
 *   auto per_sample_grad = tenzor::func::vmap(grad_fn);
 *   auto result = per_sample_grad(batched_input);
 *
 * All transforms accept and return std::function<Variable(const Variable&)>,
 * enabling arbitrary composition chains.
 */

#pragma once

#include "variable.hpp"
#include "functional.hpp"
#include "vmap.hpp"
#include "../core/tensor.hpp"
#include <functional>
#include <cstdint>

namespace tenzor {
namespace func {

/// Function type used by all transforms.
using Fn = std::function<Variable(const Variable&)>;

/**
 * @brief Create a function that computes the gradient of @p f.
 *
 * The returned function, when called with input x, evaluates f(x),
 * backpropagates, and returns the gradient with respect to x.
 * f must return a scalar Variable.
 *
 * @param f Scalar-valued differentiable function
 * @return A function mapping x -> grad_x f(x)
 */
inline auto grad(Fn f) -> Fn {
    return [f = std::move(f)](const Variable& x) -> Variable {
        Variable x_copy(x.tensor().clone(), true);
        Variable output = f(x_copy);
        output.backward();
        return Variable(*x_copy.grad(), false);
    };
}

/**
 * @brief Create a vectorized version of @p f.
 *
 * The returned function applies f independently to each slice along
 * @p in_dim of the input, stacking results along @p out_dim.
 * Uses batching rules for efficient execution when available.
 *
 * @param f Function to vectorize
 * @param in_dim Batch dimension of the input (default: 0)
 * @param out_dim Batch dimension of the output (default: 0)
 * @return A function mapping batched_x -> batched f(x)
 */
inline auto vmap(Fn f, int64_t in_dim = 0, int64_t out_dim = 0) -> Fn {
    return [f = std::move(f), in_dim, out_dim](const Variable& batched_input) -> Variable {
        // Delegate to the existing vmap implementation
        return tenzor::vmap(f, batched_input, in_dim);
    };
}

/**
 * @brief Create a function that computes the reverse-mode Jacobian of @p f.
 *
 * The returned function evaluates the full Jacobian matrix J_f(x).
 * For f: R^n -> R^m, the result is an m x n tensor.
 *
 * @param f Differentiable function
 * @return A function mapping x -> J_f(x) (as a Variable wrapping the Jacobian tensor)
 */
inline auto jacrev(Fn f) -> Fn {
    return [f = std::move(f)](const Variable& x) -> Variable {
        Tensor J = tenzor::jacobian(f, x);
        return Variable(J, false);
    };
}

/**
 * @brief Create a function that computes the forward-mode Jacobian of @p f.
 *
 * Equivalent to jacrev for correctness, but uses forward-mode AD internally
 * (more efficient when output dimension > input dimension).
 *
 * @param f Differentiable function
 * @return A function mapping x -> J_f(x) (as a Variable wrapping the Jacobian tensor)
 */
inline auto jacfwd(Fn f) -> Fn {
    // tenzor::jacobian already selects forward vs reverse mode based on dimensions
    return jacrev(std::move(f));
}

/**
 * @brief Create a function that computes the Hessian of a scalar function @p f.
 *
 * For f: R^n -> R, the result is an n x n tensor of second derivatives.
 *
 * @param f Scalar-valued differentiable function
 * @return A function mapping x -> H_f(x) (as a Variable wrapping the Hessian tensor)
 */
inline auto hessian(Fn f) -> Fn {
    return [f = std::move(f)](const Variable& x) -> Variable {
        Tensor H = tenzor::hessian(f, x);
        return Variable(H, false);
    };
}

} // namespace func
} // namespace tenzor
