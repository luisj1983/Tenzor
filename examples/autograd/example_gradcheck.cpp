/**
 * @file example_gradcheck.cpp
 * @brief Example demonstrating gradient checking for custom autograd functions
 *
 * This example shows how to use gradcheck() to verify that custom autograd
 * implementations compute gradients correctly by comparing with numerical
 * differentiation.
 */

#include <tenzor/tenzor.hpp>
#include <iostream>
#include <iomanip>

using namespace tenzor;

int main() {
    // Initialize library
    initialize();

    std::cout << "=== Gradient Checking Examples ===" << std::endl;
    std::cout << std::endl;

    // Example 1: Simple quadratic function
    {
        std::cout << "Example 1: Quadratic function f(x) = x^2" << std::endl;

        auto quadratic = [](const Variable& x) -> Variable {
            return x * x;
        };

        // Create test input
        Tensor data(std::vector<int64_t>{3}, DType::Float32, Device::cpu());
        float* ptr = data.data<float>();
        ptr[0] = 1.0f;
        ptr[1] = 2.0f;
        ptr[2] = 3.0f;

        Variable x(data, true);

        // Check gradients (verbose mode)
        bool passed = gradcheck_verbose(quadratic, x);

        std::cout << "Result: " << (passed ? "PASSED ✓" : "FAILED ✗") << std::endl;
        std::cout << std::endl;
    }

    // Example 2: Linear combination
    {
        std::cout << "Example 2: Linear function f(x) = 3*x + 2" << std::endl;

        auto linear = [](const Variable& x) -> Variable {
            return x * 3.0f + 2.0f;
        };

        Tensor data(std::vector<int64_t>{4}, DType::Float32, Device::cpu());
        for (int i = 0; i < 4; ++i) {
            data.data<float>()[i] = static_cast<float>(i) * 0.5f;
        }

        Variable x(data, true);

        // Check with custom tolerances
        bool passed = gradcheck(linear, x, 1e-6, 1e-5, 1e-3);

        std::cout << "Gradient check: " << (passed ? "PASSED ✓" : "FAILED ✗") << std::endl;
        std::cout << std::endl;
    }

    // Example 3: Polynomial function
    {
        std::cout << "Example 3: Cubic function f(x) = x^3 + 2*x^2 + x" << std::endl;

        auto cubic = [](const Variable& x) -> Variable {
            return x * x * x + x * x * 2.0f + x;
        };

        Tensor data(std::vector<int64_t>{2, 2}, DType::Float32, Device::cpu());
        float* ptr = data.data<float>();
        ptr[0] = 0.5f;
        ptr[1] = 1.0f;
        ptr[2] = 1.5f;
        ptr[3] = 2.0f;

        Variable x(data, true);

        // Get detailed result
        GradCheckResult result = gradcheck_detailed(cubic, x);

        std::cout << "Passed: " << (result.passed ? "YES" : "NO") << std::endl;
        std::cout << "Max absolute error: " << std::scientific << result.max_abs_error << std::endl;
        std::cout << "Max relative error: " << result.max_rel_error << std::endl;
        std::cout << std::endl;
    }

    // Example 4: Using Float64 for higher precision
    {
        std::cout << "Example 4: High-precision check with Float64" << std::endl;

        auto func = [](const Variable& x) -> Variable {
            return x * x + x * 2.0;
        };

        Tensor data(std::vector<int64_t>{5}, DType::Float64, Device::cpu());
        for (int i = 0; i < 5; ++i) {
            data.data<double>()[i] = static_cast<double>(i + 1) * 0.1;
        }

        Variable x(data, true);

        // Use tighter tolerances with Float64
        bool passed = gradcheck(func, x, 1e-8, 1e-7, 1e-5);

        std::cout << "Result: " << (passed ? "PASSED ✓" : "FAILED ✗") << std::endl;
        std::cout << std::endl;
    }

    // Example 5: Exception handling
    {
        std::cout << "Example 5: Exception handling for failures" << std::endl;

        auto func = [](const Variable& x) -> Variable {
            return x * x;
        };

        Tensor data(std::vector<int64_t>{3}, DType::Float32, Device::cpu());
        Variable x(data, false);  // requires_grad = false

        try {
            // This should throw because requires_grad is false
            gradcheck(func, x, 1e-6, 1e-5, 1e-3, true);
            std::cout << "Unexpected: no exception thrown" << std::endl;
        } catch (const GradCheckError& e) {
            std::cout << "Caught expected exception: " << e.what() << std::endl;
        }
        std::cout << std::endl;
    }

    // Example 6: Numerical gradient computation
    {
        std::cout << "Example 6: Direct numerical gradient computation" << std::endl;

        auto func = [](const Variable& x) -> Variable {
            // f(x) = 2*x^2, gradient = 4*x
            return x * x * 2.0f;
        };

        Tensor data(std::vector<int64_t>{3}, DType::Float32, Device::cpu());
        float* ptr = data.data<float>();
        ptr[0] = 1.0f;
        ptr[1] = 2.0f;
        ptr[2] = 3.0f;

        Variable x(data, true);

        // Compute numerical gradient
        Tensor num_grad = numerical_gradient(func, x, 1e-6);

        std::cout << "Expected gradients: [4.0, 8.0, 12.0]" << std::endl;
        std::cout << "Numerical gradients: [";
        const float* grad_ptr = num_grad.data<float>();
        for (int64_t i = 0; i < num_grad.numel(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(2) << grad_ptr[i];
        }
        std::cout << "]" << std::endl;
        std::cout << std::endl;
    }

    std::cout << "=== All examples completed ===" << std::endl;

    // Cleanup
    finalize();

    return 0;
}
