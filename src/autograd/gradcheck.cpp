/**
 * @file gradcheck.cpp
 * @brief Implementation of numerical gradient checking
 */

#include "tenzor/autograd/gradcheck.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/core/tensor.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <chrono>

namespace tenzor {

namespace {

/**
 * @brief Create a zero tensor with same properties as input.
 */
auto zeros_like_tensor(const Tensor& tensor) -> Tensor {
    Tensor result(std::vector<int64_t>(tensor.shape().begin(), tensor.shape().end()),
                  tensor.dtype(),
                  tensor.device());
    result.zero_();
    return result;
}

/**
 * @brief Extract scalar value from a tensor (supports 0-d and 1-element tensors).
 */
auto extract_scalar(const Tensor& tensor) -> double {
    // Handle 0-dimensional tensors
    if (tensor.ndim() == 0 || tensor.numel() == 1) {
        if (tensor.dtype() == DType::Float32) {
            return static_cast<double>(tensor.item<float>());
        } else if (tensor.dtype() == DType::Float64) {
            return tensor.item<double>();
        } else if (tensor.dtype() == DType::Int32) {
            return static_cast<double>(tensor.item<int32_t>());
        } else if (tensor.dtype() == DType::Int64) {
            return static_cast<double>(tensor.item<int64_t>());
        }
    }
    throw std::runtime_error("Cannot extract scalar from non-scalar tensor");
}

} // anonymous namespace

auto numerical_gradient(
    std::function<Variable(const Variable&)> func,
    const Variable& input,
    double eps
) -> Tensor {
    // Adjust epsilon based on dtype to avoid precision loss
    // Float32 needs larger epsilon due to ~7 digits of precision
    // Float64 can use smaller epsilon with ~15 digits of precision
    if (input.dtype() == DType::Float32 && eps < 1e-4) {
        eps = 1e-4;  // Minimum safe epsilon for Float32
    }

    // Create gradient tensor with same shape as input
    auto num_grad = zeros_like_tensor(input.tensor());

    // Get input data pointer and properties
    int64_t total_elements = input.tensor().numel();

    // We need to perturb each element individually
    // Create a copy of input for perturbation
    Tensor input_copy = input.tensor().clone();

    // For each element in the input
    for (int64_t i = 0; i < total_elements; ++i) {
        // Create variables for perturbed inputs
        Variable x_plus(input_copy.clone(), false);
        Variable x_minus(input_copy.clone(), false);

        // Access data based on dtype
        if (input.dtype() == DType::Float32) {
            float* data_plus = x_plus.tensor().data<float>();
            float* data_minus = x_minus.tensor().data<float>();

            // Perturb element i
            data_plus[i] += static_cast<float>(eps);
            data_minus[i] -= static_cast<float>(eps);

        } else if (input.dtype() == DType::Float64) {
            double* data_plus = x_plus.tensor().data<double>();
            double* data_minus = x_minus.tensor().data<double>();

            data_plus[i] += eps;
            data_minus[i] -= eps;

        } else {
            throw std::runtime_error("gradcheck only supports Float32 and Float64 dtypes");
        }

        // Compute function outputs
        Variable f_plus = func(x_plus);
        Variable f_minus = func(x_minus);

        // Extract scalar values (function should return scalar or we sum it)
        double val_plus, val_minus;

        if (f_plus.tensor().numel() == 1) {
            val_plus = extract_scalar(f_plus.tensor());
            val_minus = extract_scalar(f_minus.tensor());
        } else {
            // If output is not scalar, sum it to get scalar
            // This allows checking functions with multiple outputs
            Variable sum_plus = f_plus;
            Variable sum_minus = f_minus;

            // Sum all elements
            Tensor sum_plus_tensor = f_plus.tensor();
            Tensor sum_minus_tensor = f_minus.tensor();

            // Manual sum since we might not have sum() implemented yet
            double total_plus = 0.0, total_minus = 0.0;
            int64_t n = sum_plus_tensor.numel();

            if (f_plus.dtype() == DType::Float32) {
                const float* data_p = sum_plus_tensor.data<float>();
                const float* data_m = sum_minus_tensor.data<float>();
                for (int64_t j = 0; j < n; ++j) {
                    total_plus += data_p[j];
                    total_minus += data_m[j];
                }
            } else if (f_plus.dtype() == DType::Float64) {
                const double* data_p = sum_plus_tensor.data<double>();
                const double* data_m = sum_minus_tensor.data<double>();
                for (int64_t j = 0; j < n; ++j) {
                    total_plus += data_p[j];
                    total_minus += data_m[j];
                }
            }

            val_plus = total_plus;
            val_minus = total_minus;
        }

        // Compute numerical gradient using central differences
        double numerical_grad_i = (val_plus - val_minus) / (2.0 * eps);

        // Store in gradient tensor
        if (num_grad.dtype() == DType::Float32) {
            num_grad.data<float>()[i] = static_cast<float>(numerical_grad_i);
        } else if (num_grad.dtype() == DType::Float64) {
            num_grad.data<double>()[i] = numerical_grad_i;
        }
    }

    return num_grad;
}

auto compare_gradients(
    const Tensor& numerical,
    const Tensor& analytical,
    double atol,
    double rtol
) -> GradCheckResult {
    GradCheckResult result;

    // Check shapes match
    auto num_shape = numerical.shape();
    auto ana_shape = analytical.shape();

    if (num_shape.size() != ana_shape.size()) {
        result.passed = false;
        result.error_message = "Gradient shapes don't match";
        return result;
    }

    for (size_t i = 0; i < num_shape.size(); ++i) {
        if (num_shape[i] != ana_shape[i]) {
            result.passed = false;
            result.error_message = "Gradient shapes don't match";
            return result;
        }
    }

    // Compare element-wise
    int64_t total = numerical.numel();
    result.total_elements = total;
    result.max_abs_error = 0.0;
    result.max_rel_error = 0.0;
    result.failing_elements = 0;

    const bool is_float32 = numerical.dtype() == DType::Float32;

    for (int64_t i = 0; i < total; ++i) {
        double num_val, ana_val;

        if (is_float32) {
            num_val = static_cast<double>(numerical.data<float>()[i]);
            ana_val = static_cast<double>(analytical.data<float>()[i]);
        } else {
            num_val = numerical.data<double>()[i];
            ana_val = analytical.data<double>()[i];
        }

        // Compute errors
        double abs_error = std::abs(num_val - ana_val);
        double rel_error = 0.0;

        if (std::abs(ana_val) > 1e-12) {
            rel_error = abs_error / std::abs(ana_val);
        }

        // Update max errors
        result.max_abs_error = std::max(result.max_abs_error, abs_error);
        result.max_rel_error = std::max(result.max_rel_error, rel_error);

        // Check tolerance
        double threshold = atol + rtol * std::abs(ana_val);
        bool element_passed = abs_error <= threshold;

        if (!element_passed) {
            result.failing_elements++;

            // Store first few failing indices
            if (result.fail_indices.size() < 10) {
                result.fail_indices.push_back(i);
            }
        }

        // Store first few values for debugging
        if (i < 10) {
            result.numerical_grad.push_back(num_val);
            result.analytical_grad.push_back(ana_val);
        }
    }

    // Overall pass/fail
    result.passed = (result.failing_elements == 0);

    // Generate error message if failed
    if (!result.passed) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(4);
        oss << "Gradient check failed!\n";
        oss << "  Max absolute error: " << result.max_abs_error << "\n";
        oss << "  Max relative error: " << result.max_rel_error << "\n";
        oss << "  Failing elements: " << result.failing_elements
            << " / " << result.total_elements << "\n";
        oss << "  Tolerances: atol=" << atol << ", rtol=" << rtol << "\n";

        if (!result.fail_indices.empty()) {
            oss << "  First failing indices: ";
            for (size_t i = 0; i < std::min(size_t(5), result.fail_indices.size()); ++i) {
                if (i > 0) oss << ", ";
                oss << result.fail_indices[i];
            }
            oss << "\n";
        }

        result.error_message = oss.str();
    } else {
        result.error_message = "Gradient check passed";
    }

    return result;
}

auto gradcheck_detailed(
    std::function<Variable(const Variable&)> func,
    const Variable& input,
    double eps,
    double atol,
    double rtol
) -> GradCheckResult {
    // Validate input
    if (!input.requires_grad()) {
        GradCheckResult result;
        result.passed = false;
        result.error_message = "Input variable must have requires_grad=true";
        return result;
    }

    // Adjust epsilon and tolerances based on dtype
    if (input.dtype() == DType::Float32) {
        if (eps < 1e-4) eps = 1e-4;
        if (atol < 1e-4) atol = 1e-4;
        if (rtol < 1e-2) rtol = 1e-2;
    }

    // Save original requires_grad state
    bool original_requires_grad = input.requires_grad();

    try {
        // Compute numerical gradient
        Tensor num_grad = numerical_gradient(func, input, eps);

        // Compute analytical gradient
        // Create a new variable with requires_grad for backward
        Variable input_copy(input.tensor().clone(), true);

        // Forward pass
        Variable output = func(input_copy);

        // Ensure we have a scalar or sum to scalar using AUTOGRAD sum
        Variable scalar_output = output;
        if (output.tensor().numel() != 1) {
            // Use autograd sum operation to maintain computation graph!
            // This is critical - manual sum breaks the gradient flow
            scalar_output = tenzor::sum(output);
        }

        // Backward pass
        scalar_output.backward();

        // Get analytical gradient
        if (!input_copy.has_grad()) {
            GradCheckResult result;
            result.passed = false;
            result.error_message = "No gradient computed for input (check if function uses input)";
            return result;
        }

        Tensor ana_grad = *input_copy.grad();

        // Compare gradients
        return compare_gradients(num_grad, ana_grad, atol, rtol);

    } catch (const std::exception& e) {
        GradCheckResult result;
        result.passed = false;
        result.error_message = std::string("Exception during gradient check: ") + e.what();
        return result;
    }
}

auto gradcheck(
    std::function<Variable(const Variable&)> func,
    const Variable& input,
    double eps,
    double atol,
    double rtol,
    bool raise_exception
) -> bool {
    auto result = gradcheck_detailed(func, input, eps, atol, rtol);

    if (!result.passed && raise_exception) {
        throw GradCheckError(result);
    }

    return result.passed;
}

auto gradcheck_verbose(
    std::function<Variable(const Variable&)> func,
    const Variable& input,
    double eps,
    double atol,
    double rtol
) -> bool {
    std::cout << "=== Gradient Check ===" << std::endl;
    std::cout << "Input shape: [";
    auto shape = input.shape();
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << shape[i];
    }
    std::cout << "]" << std::endl;
    std::cout << "Total elements: " << input.tensor().numel() << std::endl;
    std::cout << "Epsilon: " << std::scientific << eps << std::endl;
    std::cout << "Tolerances: atol=" << atol << ", rtol=" << rtol << std::endl;
    std::cout << std::endl;

    std::cout << "Computing numerical gradients..." << std::flush;
    auto start_time = std::chrono::high_resolution_clock::now();

    auto result = gradcheck_detailed(func, input, eps, atol, rtol);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << " done (" << duration.count() << " ms)" << std::endl;
    std::cout << std::endl;

    // Print statistics
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "Results:" << std::endl;
    std::cout << "  Max absolute error: " << result.max_abs_error << std::endl;
    std::cout << "  Max relative error: " << result.max_rel_error << std::endl;
    std::cout << "  Total elements checked: " << result.total_elements << std::endl;

    if (result.passed) {
        std::cout << std::endl;
        std::cout << "✓ Gradient check PASSED!" << std::endl;
    } else {
        std::cout << "  Failing elements: " << result.failing_elements << std::endl;
        std::cout << std::endl;
        std::cout << "✗ Gradient check FAILED!" << std::endl;
        std::cout << std::endl;

        // Show sample values
        if (!result.numerical_grad.empty()) {
            std::cout << "Sample gradients (first few elements):" << std::endl;
            size_t n_samples = std::min(size_t(5), result.numerical_grad.size());
            for (size_t i = 0; i < n_samples; ++i) {
                std::cout << "  [" << i << "] numerical: " << result.numerical_grad[i]
                         << ", analytical: " << result.analytical_grad[i]
                         << ", diff: " << std::abs(result.numerical_grad[i] - result.analytical_grad[i])
                         << std::endl;
            }
        }

        if (!result.fail_indices.empty()) {
            std::cout << std::endl;
            std::cout << "First failing indices: ";
            for (size_t i = 0; i < std::min(size_t(10), result.fail_indices.size()); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << result.fail_indices[i];
            }
            std::cout << std::endl;
        }
    }

    std::cout << "======================" << std::endl;

    return result.passed;
}

} // namespace tenzor
