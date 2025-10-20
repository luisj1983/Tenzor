/**
 * @file gradcheck.hpp
 * @brief Numerical gradient checking for automatic differentiation verification
 *
 * Provides utilities to verify autograd correctness by comparing analytical
 * gradients (from backward()) with numerical gradients computed via finite
 * differences. Essential for debugging custom autograd functions.
 */

#pragma once

#include <functional>
#include <string>
#include <vector>
#include "../core/tensor.hpp"
#include "variable.hpp"

namespace tenzor {

/**
 * @brief Result of gradient checking operation.
 *
 * Contains detailed information about gradient comparison,
 * including per-element errors and statistics.
 */
struct GradCheckResult {
    bool passed;                     ///< True if gradients match within tolerance
    double max_abs_error;            ///< Maximum absolute error across all elements
    double max_rel_error;            ///< Maximum relative error across all elements
    int64_t total_elements;          ///< Total number of elements checked
    int64_t failing_elements;        ///< Number of elements exceeding tolerance
    std::vector<int64_t> fail_indices;  ///< Indices of failing elements (limited to first 10)
    std::vector<double> numerical_grad;  ///< Sample numerical gradient values (first 10)
    std::vector<double> analytical_grad; ///< Sample analytical gradient values (first 10)
    std::string error_message;       ///< Detailed error message if failed

    /**
     * @brief Construct successful result.
     */
    GradCheckResult() : passed(true), max_abs_error(0.0), max_rel_error(0.0),
                       total_elements(0), failing_elements(0) {}
};

/**
 * @brief Check gradients using numerical differentiation.
 *
 * Verifies autograd implementation by comparing analytical gradients
 * (computed via backward()) with numerical gradients (finite differences).
 * This is the primary tool for debugging custom autograd operations.
 *
 * Algorithm:
 * 1. For each element i in input tensor:
 *    - Perturb: x[i] += eps
 *    - Compute: f(x + eps*e_i)
 *    - Perturb: x[i] -= 2*eps
 *    - Compute: f(x - eps*e_i)
 *    - Numerical gradient[i] = (f(x+eps) - f(x-eps)) / (2*eps)
 * 2. Compute analytical gradient via backward()
 * 3. Compare: |numerical - analytical| <= atol + rtol * |analytical|
 *
 * @param func Function to check (must return scalar or single-output Variable)
 * @param input Input variable to compute gradients for
 * @param eps Epsilon for finite differences (default: 1e-6)
 * @param atol Absolute tolerance for comparison (default: 1e-5)
 * @param rtol Relative tolerance for comparison (default: 1e-3)
 * @param raise_exception If true, throw on failure; if false, return false (default: false)
 *
 * @return true if gradients match within tolerances
 *
 * @throws std::invalid_argument if input doesn't require gradients
 * @throws std::runtime_error if function doesn't return scalar/summable output
 * @throws GradCheckError if raise_exception=true and check fails
 *
 * @code
 * // Check a simple function
 * auto f = [](const Variable& x) {
 *     return (x * x).sum();
 * };
 *
 * Variable x(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * x.tensor().fill_(2.0f);
 *
 * bool ok = gradcheck(f, x);  // Should pass
 * if (!ok) {
 *     std::cerr << "Gradient check failed!" << std::endl;
 * }
 *
 * // With custom tolerances
 * bool ok2 = gradcheck(f, x, 1e-4, 1e-3, 1e-2);
 *
 * // Raise exception on failure
 * gradcheck(f, x, 1e-6, 1e-5, 1e-3, true);  // Throws if fails
 * @endcode
 *
 * @note Best practices:
 * - Use float64/double for better numerical accuracy
 * - Adjust eps based on function sensitivity
 * - Check on small tensors first (gradcheck is O(n) in input size)
 * - For non-differentiable points, gradcheck may fail spuriously
 * - Disable dropout, randomness during checking
 */
auto gradcheck(
    std::function<Variable(const Variable&)> func,
    const Variable& input,
    double eps = 1e-6,
    double atol = 1e-5,
    double rtol = 1e-3,
    bool raise_exception = false
) -> bool;

/**
 * @brief Detailed gradient checking with full result information.
 *
 * Like gradcheck() but returns detailed information about the comparison,
 * including error statistics and sample failing elements. Useful for
 * debugging and understanding where gradients diverge.
 *
 * @param func Function to check
 * @param input Input variable
 * @param eps Epsilon for finite differences
 * @param atol Absolute tolerance
 * @param rtol Relative tolerance
 *
 * @return GradCheckResult with detailed comparison information
 *
 * @code
 * auto result = gradcheck_detailed(func, x);
 * if (!result.passed) {
 *     std::cout << "Max abs error: " << result.max_abs_error << std::endl;
 *     std::cout << "Max rel error: " << result.max_rel_error << std::endl;
 *     std::cout << "Failing elements: " << result.failing_elements
 *               << " / " << result.total_elements << std::endl;
 *     std::cout << result.error_message << std::endl;
 * }
 * @endcode
 */
auto gradcheck_detailed(
    std::function<Variable(const Variable&)> func,
    const Variable& input,
    double eps = 1e-6,
    double atol = 1e-5,
    double rtol = 1e-3
) -> GradCheckResult;

/**
 * @brief Custom exception for gradient checking failures.
 *
 * Thrown when gradcheck() is called with raise_exception=true
 * and the gradient check fails.
 */
class GradCheckError : public std::runtime_error {
public:
    GradCheckResult result;  ///< Detailed check result

    explicit GradCheckError(const GradCheckResult& res)
        : std::runtime_error(res.error_message), result(res) {}
};

/**
 * @brief Compute numerical gradient using finite differences.
 *
 * Helper function that computes the numerical gradient of a function
 * at a given input point using central differences. This is the core
 * numerical differentiation routine used by gradcheck.
 *
 * @param func Function to differentiate
 * @param input Input point
 * @param eps Epsilon for finite differences
 *
 * @return Tensor with numerical gradients
 *
 * @code
 * auto f = [](const Variable& x) { return (x * x).sum(); };
 * Variable x(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
 * Tensor num_grad = numerical_gradient(f, x, 1e-6);
 * @endcode
 */
auto numerical_gradient(
    std::function<Variable(const Variable&)> func,
    const Variable& input,
    double eps
) -> Tensor;

/**
 * @brief Compare two gradient tensors with tolerances.
 *
 * Compares numerical and analytical gradients element-wise using
 * the formula: |numerical - analytical| <= atol + rtol * |analytical|
 *
 * @param numerical Numerical gradient tensor
 * @param analytical Analytical gradient tensor
 * @param atol Absolute tolerance
 * @param rtol Relative tolerance
 *
 * @return GradCheckResult with comparison details
 */
auto compare_gradients(
    const Tensor& numerical,
    const Tensor& analytical,
    double atol,
    double rtol
) -> GradCheckResult;

/**
 * @brief Verbose gradient checking with printed output.
 *
 * Like gradcheck() but prints detailed progress and results to stdout.
 * Useful for interactive debugging and understanding gradient behavior.
 *
 * @param func Function to check
 * @param input Input variable
 * @param eps Epsilon for finite differences
 * @param atol Absolute tolerance
 * @param rtol Relative tolerance
 *
 * @return true if gradients match
 *
 * @code
 * // Prints detailed information about gradient checking
 * bool ok = gradcheck_verbose(func, x);
 * // Output:
 * // Computing numerical gradients... 12 elements
 * // Computing analytical gradients...
 * // Comparing gradients...
 * // Max absolute error: 1.23e-7
 * // Max relative error: 2.45e-6
 * // ✓ Gradient check passed!
 * @endcode
 */
auto gradcheck_verbose(
    std::function<Variable(const Variable&)> func,
    const Variable& input,
    double eps = 1e-6,
    double atol = 1e-5,
    double rtol = 1e-3
) -> bool;

} // namespace tenzor
