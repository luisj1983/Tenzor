/**
 * @file test_gradcheck.cpp
 * @brief Comprehensive tests for gradient checking functionality
 */

#include <gtest/gtest.h>
#include "tenzor/autograd/gradcheck.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/tenzor.hpp"
#include <cmath>

using namespace tenzor;

class GradCheckTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Initialize Tenzor library (loads and registers backends)
        tenzor::initialize();
    }

    void SetUp() override {
        // Set gradient computation enabled
        set_grad_enabled(true);
    }
};

// Test basic quadratic function: f(x) = x^2
TEST_F(GradCheckTest, QuadraticFunction) {
    auto f = [](const Variable& x) -> Variable {
        return x * x;
    };

    // Create small test input
    Tensor data(std::vector<int64_t>{3}, DType::Float32, Device::cpu());
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
TEST_F(GradCheckTest, LinearFunction) {
    auto f = [](const Variable& x) -> Variable {
        return x * 2.0f + 3.0f;
    };

    Tensor data(std::vector<int64_t>{4}, DType::Float32, Device::cpu());
    float* ptr = data.data<float>();
    for (int i = 0; i < 4; ++i) {
        ptr[i] = static_cast<float>(i + 1);
    }

    Variable x(data, true);

    bool passed = gradcheck(f, x, 1e-6, 1e-5, 1e-3);
    EXPECT_TRUE(passed);
}

// Test sum reduction: f(x) = sum(x)
TEST_F(GradCheckTest, SumReduction) {
    auto f = [](const Variable& x) -> Variable {
        // Use autograd sum to maintain computation graph
        return tenzor::sum(x);
    };

    Tensor data(std::vector<int64_t>{5}, DType::Float32, Device::cpu());
    float* ptr = data.data<float>();
    for (int i = 0; i < 5; ++i) {
        ptr[i] = static_cast<float>(i + 1) * 0.5f;
    }

    Variable x(data, true);

    bool passed = gradcheck(f, x, 1e-6, 1e-5, 1e-3);
    EXPECT_TRUE(passed);
}

// Test element-wise operations: f(x) = x * x + x
TEST_F(GradCheckTest, ElementWiseOps) {
    auto f = [](const Variable& x) -> Variable {
        return x * x + x;
    };

    Tensor data(std::vector<int64_t>{3}, DType::Float32, Device::cpu());
    float* ptr = data.data<float>();
    ptr[0] = 0.5f;
    ptr[1] = 1.0f;
    ptr[2] = 1.5f;

    Variable x(data, true);

    bool passed = gradcheck(f, x, 1e-6, 1e-5, 1e-3);
    EXPECT_TRUE(passed);
}

// Test multi-dimensional input: f(x) = sum(x^2)
TEST_F(GradCheckTest, MultiDimensional) {
    auto f = [](const Variable& x) -> Variable {
        Variable x_squared = x * x;
        // Use autograd sum to maintain computation graph
        return tenzor::sum(x_squared);
    };

    Tensor data(std::vector<int64_t>{2, 3}, DType::Float32, Device::cpu());
    float* ptr = data.data<float>();
    for (int i = 0; i < 6; ++i) {
        ptr[i] = static_cast<float>(i + 1) * 0.5f;
    }

    Variable x(data, true);

    bool passed = gradcheck(f, x, 1e-6, 1e-5, 1e-3);
    EXPECT_TRUE(passed);
}

// Test detailed gradient check result
TEST_F(GradCheckTest, DetailedResult) {
    auto f = [](const Variable& x) -> Variable {
        return x * x;
    };

    Tensor data(std::vector<int64_t>{3}, DType::Float32, Device::cpu());
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

// Test with Float64 for higher precision
TEST_F(GradCheckTest, Float64Precision) {
    auto f = [](const Variable& x) -> Variable {
        return x * x + x * 2.0;
    };

    Tensor data(std::vector<int64_t>{4}, DType::Float64, Device::cpu());
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
TEST_F(GradCheckTest, RequiresGradCheck) {
    auto f = [](const Variable& x) -> Variable {
        return x * x;
    };

    Tensor data(std::vector<int64_t>{3}, DType::Float32, Device::cpu());
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
TEST_F(GradCheckTest, NumericalGradientComputation) {
    auto f = [](const Variable& x) -> Variable {
        // f(x) = 3*x^2, gradient = 6*x
        return x * x * 3.0f;
    };

    Tensor data(std::vector<int64_t>{2}, DType::Float32, Device::cpu());
    float* ptr = data.data<float>();
    ptr[0] = 1.0f;
    ptr[1] = 2.0f;

    Variable x(data, true);

    Tensor num_grad = numerical_gradient(f, x, 1e-6);

    // Expected gradients: [6.0, 12.0]
    EXPECT_EQ(num_grad.numel(), 2);

    const float* grad_ptr = num_grad.data<float>();
    // Float32 precision requires larger tolerance
    EXPECT_NEAR(grad_ptr[0], 6.0f, 1e-2);   // ~0.01 tolerance for Float32
    EXPECT_NEAR(grad_ptr[1], 12.0f, 1e-2);  // ~0.01 tolerance for Float32
}

// Test compare_gradients function
TEST_F(GradCheckTest, CompareGradients) {
    Tensor num_grad(std::vector<int64_t>{3}, DType::Float32, Device::cpu());
    Tensor ana_grad(std::vector<int64_t>{3}, DType::Float32, Device::cpu());

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
TEST_F(GradCheckTest, EpsilonSensitivity) {
    auto f = [](const Variable& x) -> Variable {
        return x * x * x;  // f(x) = x^3, gradient = 3*x^2
    };

    Tensor data(std::vector<int64_t>{2}, DType::Float32, Device::cpu());
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
TEST_F(GradCheckTest, ScalarOutput) {
    auto f = [](const Variable& x) -> Variable {
        // Use autograd operations to maintain computation graph
        return tenzor::sum(x * x);  // sum(x^2)
    };

    Tensor data(std::vector<int64_t>{3}, DType::Float32, Device::cpu());
    float* ptr = data.data<float>();
    ptr[0] = 1.0f;
    ptr[1] = 2.0f;
    ptr[2] = 3.0f;

    Variable x(data, true);

    bool passed = gradcheck(f, x, 1e-6, 1e-5, 1e-3);
    EXPECT_TRUE(passed);
}

// Performance test - measure gradcheck time
TEST_F(GradCheckTest, PerformanceBenchmark) {
    auto f = [](const Variable& x) -> Variable {
        return x * x + x;
    };

    // Small tensor
    Tensor data(std::vector<int64_t>{10}, DType::Float32, Device::cpu());
    for (int64_t i = 0; i < 10; ++i) {
        data.data<float>()[i] = static_cast<float>(i) * 0.1f;
    }

    Variable x(data, true);

    auto start = std::chrono::high_resolution_clock::now();
    bool passed = gradcheck(f, x, 1e-6, 1e-5, 1e-3);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_TRUE(passed);
    // Should complete reasonably fast (< 1 second for 10 elements)
    EXPECT_LT(duration.count(), 1000);

    std::cout << "Gradcheck for 10 elements: " << duration.count() << " ms" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
