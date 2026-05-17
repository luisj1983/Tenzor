/**
 * @file gradcheck_complex.hpp
 * @brief Gradient checking for functions f: R^n -> C^m via Wirtinger derivatives.
 *
 * The library-level `gradcheck()` only verifies the scalar contraction
 *   L = sum(Re(y) + Im(y))
 * which collapses the full complex Jacobian into a single real number per
 * input. That catches sign-flip / shape bugs but is insensitive to whether
 * the real and imaginary parts of the Jacobian are individually correct —
 * an op that swaps Re/Im in its backward would still pass.
 *
 * `gradcheck_complex` exercises the full complex Jacobian by running TWO
 * backward passes per check (one seeded with `(1+0i)`, one with `(0+1i)`)
 * and comparing each component against finite-difference perturbations.
 *
 * For a real input x and complex output y = f(x):
 *   - numerical: (f(x+h) - f(x-h)) / (2h)  is a complex vector — split into
 *     Re and Im components.
 *   - analytical (via backward with seed `(1+0i)*ones`):
 *       dL_re/dx_i = sum_j Re(dy_j/dx_i)
 *     which matches the numerical sum-of-Re.
 *   - analytical (via backward with seed `(0+1i)*ones`):
 *       dL_im/dx_i = sum_j Im(dy_j/dx_i)
 *     which matches the numerical sum-of-Im.
 *
 * Both checks must pass.
 */
#pragma once

#include <tenzor/autograd/variable.hpp>
#include <functional>

namespace tenzor_test {

/**
 * @brief Run gradcheck for a function returning a complex Variable.
 *
 * @param func    Function under test: takes a real Variable, returns complex.
 * @param input   Real-valued input Variable with `requires_grad=true`.
 * @param eps     Finite-difference epsilon (auto-bumped to 5e-4 for Float32).
 * @param atol    Absolute tolerance per element.
 * @param rtol    Relative tolerance per element.
 * @return        true iff both Re-seed and Im-seed checks pass.
 */
auto gradcheck_complex(
    std::function<tenzor::Variable(const tenzor::Variable&)> func,
    const tenzor::Variable& input,
    double eps = 1e-4,
    double atol = 1e-3,
    double rtol = 1e-2
) -> bool;

}  // namespace tenzor_test
