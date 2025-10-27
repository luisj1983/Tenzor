/**
 * @file test_gradcheck.cpp
 * @brief Comprehensive tests for gradient checking functionality - Parameterized across all backends
 */

#include <gtest/gtest.h>
#include "tenzor/autograd/gradcheck.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/tenzor.hpp"
#include "../backend_test_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class GradCheckBackendTest : public BackendTest {
protected:
    static void SetUpTestSuite() {
        // Initialize Tenzor library (loads and registers backends)
        tenzor::initialize();
    }

    void SetUp() override {
        BackendTest::SetUp();
        // Set gradient computation enabled
        set_grad_enabled(true);
    }
};

// Test basic quadratic function: f(x) = x^2
TEST_P(GradCheckBackendTest, QuadraticFunction) {
    auto f = [](const Variable& x) -> Variable {
        return x * x;
    };

    // Create small test input
    Tensor data(std::vector<int64_t>{3}, DType::Float32, device);
    float* ptr = data.data<float>();
    ptr[0] = 1.0f;
    ptr[1] = 2.0f;
    ptr[2] = 3.0f;

    Variable x(data, true);

    // Check gradients - should pass since autograd handles x^2 correctly
    bool passed = gradcheck(f, x, 1e-6, 1e-5, 1e-3);
    EXPECT_TRUE(passed);
}

// Test linear function: f(x) = 2*x + 3
TEST_P(GradCheckBackendTest, LinearFunction) {
    auto f = [](const Variable& x) -> Variable {
        return x * 2.0f + 3.0f;
    };

    Tensor data(std::vector<int64_t>{4}, DType::Float32, device);
    float* ptr = data.data<float>();
    for (int i = 0; i < 4; ++i) {
        ptr[i] = static_cast<float>(i + 1);
    }

    Variable x(data, true);

    bool passed = gradcheck(f, x, 1e-6, 1e-5, 1e-3);
    EXPECT_TRUE(passed);
}

// Test sum reduction: f(x) = sum(x)
TEST_P(GradCheckBackendTest, SumReduction) {
    auto f = [](const Variable& x) -> Variable {
        // Use autograd sum to maintain computation graph
        return tenzor::sum(x);
    };

    Tensor data(std::vector<int64_t>{5}, DType::Float32, device);
    float* ptr = data.data<float>();
    for (int i = 0; i < 5; ++i) {
        ptr[i] = static_cast<float>(i + 1) * 0.5f;
    }

    Variable x(data, true);

    bool passed = gradcheck(f, x, 1e-6, 1e-5, 1e-3);
    EXPECT_TRUE(passed);
}

// Test element-wise operations: f(x) = x * x + x
TEST_P(GradCheckBackendTest, ElementWiseOps) {
    auto f = [](const Variable& x) -> Variable {
        return x * x + x;
    };

    Tensor data(std::vector<int64_t>{3}, DType::Float32, device);
    float* ptr = data.data<float>();
    ptr[0] = 0.5f;
    ptr[1] = 1.0f;
    ptr[2] = 1.5f;

    Variable x(data, true);

    bool passed = gradcheck(f, x, 1e-6, 1e-5, 1e-3);
    EXPECT_TRUE(passed);
}

// Test multi-dimensional input: f(x) = sum(x^2)
TEST_P(GradCheckBackendTest, MultiDimensional) {
    auto f = [](const Variable& x) -> Variable {
        Variable x_squared = x * x;
        // Use autograd sum to maintain computation graph
        return tenzor::sum(x_squared);
    };

    Tensor data(std::vector<int64_t>{2, 3}, DType::Float32, device);
    float* ptr = data.data<float>();
    for (int i = 0; i < 6; ++i) {
        ptr[i] = static_cast<float>(i + 1) * 0.5f;
    }

    Variable x(data, true);

    bool passed = gradcheck(f, x, 1e-6, 1e-5, 1e-3);
    EXPECT_TRUE(passed);
}

// Test detailed gradient check result
TEST_P(GradCheckBackendTest, DetailedResult) {
    auto f = [](const Variable& x) -> Variable {
        return x * x;
    };

    Tensor data(std::vector<int64_t>{3}, DType::Float32, device);
    float* ptr = data.data<float>();
    ptr[0] = 1.0f;
    ptr[1] = 2.0f;
    ptr[2] = 3.0f;

    Variable x(data, true);

    GradCheckResult result = gradcheck_detailed(f, x, 1e-6, 1e-5, 1e-3);

    EXPECT_TRUE(result.passed);
    EXPECT_EQ(result.total_elements, 3);
    EXPECT_EQ(result.failing_elements, 0);
    // Adjusted tolerances for Float32 precision
    EXPECT_LT(result.max_abs_error, 1e-2);  // Float32 needs looser tolerance
    EXPECT_LT(result.max_rel_error, 5e-2);  // Float32 needs looser tolerance
}

// Test with Float64 for higher precision (CPU only due to backend support)
TEST_P(GradCheckBackendTest, Float64Precision) {
    // Skip if not CPU - Float64 support varies by backend
    if (device.type != Device::Type::CPU) {
        GTEST_SKIP() << "Float64 test only on CPU backend";
    }

    auto f = [](const Variable& x) -> Variable {
        return x * x + x * 2.0;
    };

    Tensor data(std::vector<int64_t>{4}, DType::Float64, device);
    double* ptr = data.data<double>();
    for (int i = 0; i < 4; ++i) {
        ptr[i] = static_cast<double>(i + 1) * 0.1;
    }

    Variable x(data, true);

    // Float64 can use tighter tolerances than Float32
    auto result = gradcheck_detailed(f, x, 1e-6, 1e-5, 1e-3);

    EXPECT_TRUE(result.passed);
    EXPECT_LT(result.max_abs_error, 1e-6);  // Float64 should have very tight tolerance
    EXPECT_LT(result.max_rel_error, 1e-5);  // Float64 should have very tight tolerance
}

// Test exception on requires_grad = false
TEST_P(GradCheckBackendTest, RequiresGradCheck) {
    auto f = [](const Variable& x) -> Variable {
        return x * x;
    };

    Tensor data(std::vector<int64_t>{3}, DType::Float32, device);
    Variable x(data, false);  // requires_grad = false

    // Should not throw when raise_exception = false
    bool passed = gradcheck(f, x, 1e-6, 1e-5, 1e-3, false);
    EXPECT_FALSE(passed);

    // Should throw when raise_exception = true
    EXPECT_THROW({
        gradcheck(f, x, 1e-6, 1e-5, 1e-3, true);
    }, GradCheckError);
}

// Test numerical gradient computation
TEST_P(GradCheckBackendTest, NumericalGradientComputation) {
    auto f = [](const Variable& x) -> Variable {
        // f(x) = 3*x^2, gradient = 6*x
        return x * x * 3.0f;
    };

    Tensor data(std::vector<int64_t>{2}, DType::Float32, device);
    float* ptr = data.data<float>();
    ptr[0] = 1.0f;
    ptr[1] = 2.0f;

    Variable x(data, true);

    Tensor num_grad = numerical_gradient(f, x, 1e-6);

    // Expected gradients: [6.0, 12.0]
    EXPECT_EQ(num_grad.numel(), 2);

    auto num_grad_cpu = num_grad.to(Device::cpu());
    const float* grad_ptr = num_grad_cpu.data<float>();
    // Float32 precision requires larger tolerance
    EXPECT_NEAR(grad_ptr[0], 6.0f, 1e-2);   // ~0.01 tolerance for Float32
    EXPECT_NEAR(grad_ptr[1], 12.0f, 1e-2);  // ~0.01 tolerance for Float32
}

// Test compare_gradients function
TEST_P(GradCheckBackendTest, CompareGradients) {
    Tensor num_grad(std::vector<int64_t>{3}, DType::Float32, device);
    Tensor ana_grad(std::vector<int64_t>{3}, DType::Float32, device);

    float* num_ptr = num_grad.data<float>();
    float* ana_ptr = ana_grad.data<float>();

    // Matching gradients
    num_ptr[0] = 1.0f;
    num_ptr[1] = 2.0f;
    num_ptr[2] = 3.0f;

    ana_ptr[0] = 1.0f + 1e-6f;  // Tiny difference
    ana_ptr[1] = 2.0f + 1e-6f;
    ana_ptr[2] = 3.0f + 1e-6f;

    GradCheckResult result = compare_gradients(num_grad, ana_grad, 1e-5, 1e-3);
    EXPECT_TRUE(result.passed);

    // Non-matching gradients
    ana_ptr[2] = 10.0f;  // Large difference

    result = compare_gradients(num_grad, ana_grad, 1e-5, 1e-3);
    EXPECT_FALSE(result.passed);
    EXPECT_GT(result.failing_elements, 0);
}

// Test with different epsilon values
TEST_P(GradCheckBackendTest, EpsilonSensitivity) {
    auto f = [](const Variable& x) -> Variable {
        return x * x * x;  // f(x) = x^3, gradient = 3*x^2
    };

    Tensor data(std::vector<int64_t>{2}, DType::Float32, device);
    float* ptr = data.data<float>();
    ptr[0] = 1.0f;
    ptr[1] = 2.0f;

    Variable x(data, true);

    // Try different epsilon values
    bool passed_small = gradcheck(f, x, 1e-7, 1e-5, 1e-3);
    bool passed_medium = gradcheck(f, x, 1e-6, 1e-5, 1e-3);
    bool passed_large = gradcheck(f, x, 1e-5, 1e-5, 1e-3);

    // At least one should pass
    EXPECT_TRUE(passed_small || passed_medium || passed_large);
}

// Test scalar output (single element)
TEST_P(GradCheckBackendTest, ScalarOutput) {
    auto f = [](const Variable& x) -> Variable {
        // Use autograd operations to maintain computation graph
        return tenzor::sum(x * x);  // sum(x^2)
    };

    Tensor data(std::vector<int64_t>{3}, DType::Float32, device);
    float* ptr = data.data<float>();
    ptr[0] = 1.0f;
    ptr[1] = 2.0f;
    ptr[2] = 3.0f;

    Variable x(data, true);

    bool passed = gradcheck(f, x, 1e-6, 1e-5, 1e-3);
    EXPECT_TRUE(passed);
}

// Performance test - measure gradcheck time
TEST_P(GradCheckBackendTest, PerformanceBenchmark) {
    auto f = [](const Variable& x) -> Variable {
        return x * x + x;
    };

    // Small tensor
    Tensor data(std::vector<int64_t>{10}, DType::Float32, device);
    for (int64_t i = 0; i < 10; ++i) {
        data.data<float>()[i] = static_cast<float>(i) * 0.1f;
    }

    Variable x(data, true);

    auto start = std::chrono::high_resolution_clock::now();
    bool passed = gradcheck(f, x, 1e-6, 1e-5, 1e-3);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_TRUE(passed);
    // Should complete reasonably fast (< 2 second for 10 elements, allowing for slower backends)
    EXPECT_LT(duration.count(), 2000);
}

INSTANTIATE_BACKEND_TESTS(GradCheckBackendTest);
