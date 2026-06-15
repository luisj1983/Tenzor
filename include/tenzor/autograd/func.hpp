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
#include "ops.hpp"
#include "../core/tensor.hpp"
#include <functional>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

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
        // create_graph=true so the returned gradient itself carries a grad_fn,
        // enabling composition of grad with other transforms (grad(grad(f)),
        // jacrev(grad(f)), ...). Without it the result is detached and yields
        // zero/non-differentiable second derivatives.
        output.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);
        // Prefer the graph-preserving gradient Variable; fall back to the plain
        // gradient tensor only if create_graph did not populate it.
        const auto& gv = x_copy.grad_variable();
        if (gv) {
            return *gv;
        }
        // Guard against dereferencing an empty optional: when the function
        // ignores its argument (or routes through a detached path), the
        // output does not depend on x_copy and backward() leaves grad()
        // unpopulated. Dereferencing *x_copy.grad() in that case is UB.
        const auto& g = x_copy.grad();
        if (!g) {
            throw std::runtime_error(
                "func::grad: function output does not depend on its input "
                "(gradient is undefined). The transformed function must use "
                "its argument through differentiable ops.");
        }
        return Variable(*g, false);
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
        // tenzor::vmap slices the input along in_dim and stacks the per-slice
        // results back along that same axis, so its output carries the batch
        // axis at position in_dim. Honour out_dim by moving that batch axis
        // from in_dim to out_dim. Previously out_dim was captured but never
        // applied, silently producing a wrong-shape result for out_dim !=
        // in_dim.
        Variable result = tenzor::vmap(f, batched_input, in_dim);

        const int64_t ndim = static_cast<int64_t>(result.tensor().shape().size());
        // Normalise the source/destination positions against the output rank.
        int64_t src = in_dim < 0 ? in_dim + ndim : in_dim;
        int64_t dst = out_dim < 0 ? out_dim + ndim : out_dim;
        if (src == dst || ndim == 0) {
            return result;
        }
        if (src < 0 || src >= ndim || dst < 0 || dst >= ndim) {
            throw std::runtime_error(
                "func::vmap: out_dim is out of range for the vmapped output");
        }
        // Build the permutation that moves axis `src` to position `dst`,
        // keeping the relative order of the remaining axes (movedim
        // semantics). Use the Variable-level autograd::permute so the
        // grad_fn chain is preserved through the axis move.
        std::vector<int64_t> order;
        order.reserve(ndim);
        for (int64_t d = 0; d < ndim; ++d) {
            if (d != src) {
                order.push_back(d);
            }
        }
        order.insert(order.begin() + dst, src);
        return tenzor::permute(result, order);
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
