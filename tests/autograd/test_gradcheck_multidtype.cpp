/**
 * @file test_gradcheck_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for gradient checking functionality
 *
 * Converted from test_gradcheck.cpp. Gradcheck tests skip Float16 due to
 * insufficient precision for finite-difference numerical gradient computation.
 */

#include <gtest/gtest.h>
#include "tenzor/autograd/gradcheck.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/tenzor.hpp"
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class GradCheckMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        set_grad_enabled(true);
    }
};

// Test basic quadratic function: f(x) = x^2
TEST_P(GradCheckMultiDTypeTest, QuadraticFunction) {
    if (dtype() == DType::Float16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16 gradcheck precision");

    auto f = [](const Variable& x) -> Variable {
        return x * x;
    };

    Tensor data = zeros({3}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    float* ptr = data_cpu.data<float>();
    ptr[0] = 1.0f;
    ptr[1] = 2.0f;
    ptr[2] = 3.0f;
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    double gc_atol = (dtype() == DType::Float64) ? 1e-5 : 1e-3;
    double gc_rtol = (dtype() == DType::Float64) ? 1e-5 : 1e-3;
    bool passed = gradcheck(f, x, 1e-6, gc_atol, gc_rtol);
    EXPECT_TRUE(passed);
}

// Test linear function: f(x) = 2*x + 3
TEST_P(GradCheckMultiDTypeTest, LinearFunction) {
    if (dtype() == DType::Float16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16 gradcheck precision");

    auto f = [](const Variable& x) -> Variable {
        return x * 2.0f + 3.0f;
    };

    Tensor data = zeros({4}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    float* ptr = data_cpu.data<float>();
    for (int i = 0; i < 4; ++i) {
        ptr[i] = static_cast<float>(i + 1);
    }
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    double gc_atol = (dtype() == DType::Float64) ? 1e-5 : 1e-3;
    double gc_rtol = (dtype() == DType::Float64) ? 1e-5 : 1e-3;
    bool passed = gradcheck(f, x, 1e-6, gc_atol, gc_rtol);
    EXPECT_TRUE(passed);
}

// Test sum reduction: f(x) = sum(x)
TEST_P(GradCheckMultiDTypeTest, SumReduction) {
    if (dtype() == DType::Float16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16 gradcheck precision");

    auto f = [](const Variable& x) -> Variable {
        return tenzor::sum(x);
    };

    Tensor data = zeros({5}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    float* ptr = data_cpu.data<float>();
    for (int i = 0; i < 5; ++i) {
        ptr[i] = static_cast<float>(i + 1) * 0.5f;
    }
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    double gc_atol = (dtype() == DType::Float64) ? 1e-5 : 1e-3;
    double gc_rtol = (dtype() == DType::Float64) ? 1e-5 : 1e-3;
    bool passed = gradcheck(f, x, 1e-6, gc_atol, gc_rtol);
    EXPECT_TRUE(passed);
}

// Test element-wise operations: f(x) = x * x + x
TEST_P(GradCheckMultiDTypeTest, ElementWiseOps) {
    if (dtype() == DType::Float16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16 gradcheck precision");

    auto f = [](const Variable& x) -> Variable {
        return x * x + x;
    };

    Tensor data = zeros({3}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    float* ptr = data_cpu.data<float>();
    ptr[0] = 0.5f;
    ptr[1] = 1.0f;
    ptr[2] = 1.5f;
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    double gc_atol = (dtype() == DType::Float64) ? 1e-5 : 1e-3;
    double gc_rtol = (dtype() == DType::Float64) ? 1e-5 : 1e-3;
    bool passed = gradcheck(f, x, 1e-6, gc_atol, gc_rtol);
    EXPECT_TRUE(passed);
}

// Test multi-dimensional input: f(x) = sum(x^2)
TEST_P(GradCheckMultiDTypeTest, MultiDimensional) {
    if (dtype() == DType::Float16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16 gradcheck precision");

    auto f = [](const Variable& x) -> Variable {
        Variable x_squared = x * x;
        return tenzor::sum(x_squared);
    };

    Tensor data = zeros({2, 3}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    float* ptr = data_cpu.data<float>();
    for (int i = 0; i < 6; ++i) {
        ptr[i] = static_cast<float>(i + 1) * 0.5f;
    }
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    double gc_atol = (dtype() == DType::Float64) ? 1e-5 : 1e-3;
    double gc_rtol = (dtype() == DType::Float64) ? 1e-5 : 1e-3;
    bool passed = gradcheck(f, x, 1e-6, gc_atol, gc_rtol);
    EXPECT_TRUE(passed);
}

// Test detailed gradient check result
TEST_P(GradCheckMultiDTypeTest, DetailedResult) {
    if (dtype() == DType::Float16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16 gradcheck precision");

    auto f = [](const Variable& x) -> Variable {
        return x * x;
    };

    Tensor data = zeros({3}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    float* ptr = data_cpu.data<float>();
    ptr[0] = 1.0f;
    ptr[1] = 2.0f;
    ptr[2] = 3.0f;
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    double gc_atol = (dtype() == DType::Float64) ? 1e-5 : 1e-3;
    double gc_rtol = (dtype() == DType::Float64) ? 1e-5 : 1e-3;
    GradCheckResult result = gradcheck_detailed(f, x, 1e-6, gc_atol, gc_rtol);

    EXPECT_TRUE(result.passed);
    EXPECT_EQ(result.total_elements, 3);
    EXPECT_EQ(result.failing_elements, 0);
    double abs_tol = (dtype() == DType::Float64) ? 1e-6 : 1e-2;
    double rel_tol = (dtype() == DType::Float64) ? 1e-5 : 5e-2;
    EXPECT_LT(result.max_abs_error, abs_tol);
    EXPECT_LT(result.max_rel_error, rel_tol);
}

// Test exception on requires_grad = false
TEST_P(GradCheckMultiDTypeTest, RequiresGradCheck) {
    if (dtype() == DType::Float16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16 gradcheck precision");

    auto f = [](const Variable& x) -> Variable {
        return x * x;
    };

    Tensor data(std::vector<int64_t>{3}, dtype(), device());
    Variable x(data, false);  // requires_grad = false

    bool passed = gradcheck(f, x, 1e-6, 1e-5, 1e-3, false);
    EXPECT_FALSE(passed);

    EXPECT_THROW({
        gradcheck(f, x, 1e-6, 1e-5, 1e-3, true);
    }, GradCheckError);
}

// Test numerical gradient computation
TEST_P(GradCheckMultiDTypeTest, NumericalGradientComputation) {
    if (dtype() == DType::Float16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16 gradcheck precision");

    auto f = [](const Variable& x) -> Variable {
        return x * x * 3.0f;
    };

    Tensor data = zeros({2}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    float* ptr = data_cpu.data<float>();
    ptr[0] = 1.0f;
    ptr[1] = 2.0f;
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    Tensor num_grad = numerical_gradient(f, x, 1e-6);
    EXPECT_EQ(num_grad.numel(), 2);

    auto num_grad_cpu = num_grad.to(Device::cpu()).to(DType::Float32);
    const float* grad_ptr = num_grad_cpu.data<float>();
    float tol = (dtype() == DType::Float64) ? 1e-4f : 1e-2f;
    EXPECT_NEAR(grad_ptr[0], 6.0f, tol);
    EXPECT_NEAR(grad_ptr[1], 12.0f, tol);
}

// Test compare_gradients function
TEST_P(GradCheckMultiDTypeTest, CompareGradients) {
    if (dtype() == DType::Float16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16 gradcheck precision");

    Tensor num_grad = zeros({3}, dtype(), device());
    Tensor ana_grad = zeros({3}, dtype(), device());

    auto num_grad_cpu = num_grad.to(Device::cpu()).to(DType::Float32);
    auto ana_grad_cpu = ana_grad.to(Device::cpu()).to(DType::Float32);

    float* num_ptr = num_grad_cpu.data<float>();
    float* ana_ptr = ana_grad_cpu.data<float>();

    num_ptr[0] = 1.0f;
    num_ptr[1] = 2.0f;
    num_ptr[2] = 3.0f;

    ana_ptr[0] = 1.0f + 1e-6f;
    ana_ptr[1] = 2.0f + 1e-6f;
    ana_ptr[2] = 3.0f + 1e-6f;

    num_grad = num_grad_cpu.to(dtype()).to(device());
    ana_grad = ana_grad_cpu.to(dtype()).to(device());

    GradCheckResult result = compare_gradients(num_grad, ana_grad, 1e-5, 1e-3);
    EXPECT_TRUE(result.passed);

    // Non-matching gradients
    ana_grad_cpu = ana_grad.to(Device::cpu()).to(DType::Float32);
    ana_ptr = ana_grad_cpu.data<float>();
    ana_ptr[2] = 10.0f;
    ana_grad = ana_grad_cpu.to(dtype()).to(device());

    result = compare_gradients(num_grad, ana_grad, 1e-5, 1e-3);
    EXPECT_FALSE(result.passed);
    EXPECT_GT(result.failing_elements, 0);
}

// Test with different epsilon values
TEST_P(GradCheckMultiDTypeTest, EpsilonSensitivity) {
    if (dtype() == DType::Float16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16 gradcheck precision");

    auto f = [](const Variable& x) -> Variable {
        return x * x * x;
    };

    Tensor data = zeros({2}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    float* ptr = data_cpu.data<float>();
    ptr[0] = 1.0f;
    ptr[1] = 2.0f;
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    bool passed_small = gradcheck(f, x, 1e-7, 1e-5, 1e-3);
    bool passed_medium = gradcheck(f, x, 1e-6, 1e-5, 1e-3);
    bool passed_large = gradcheck(f, x, 1e-5, 1e-5, 1e-3);

    EXPECT_TRUE(passed_small || passed_medium || passed_large);
}

// Test scalar output (single element)
TEST_P(GradCheckMultiDTypeTest, ScalarOutput) {
    if (dtype() == DType::Float16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16 gradcheck precision");

    auto f = [](const Variable& x) -> Variable {
        return tenzor::sum(x * x);
    };

    Tensor data = zeros({3}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    float* ptr = data_cpu.data<float>();
    ptr[0] = 1.0f;
    ptr[1] = 2.0f;
    ptr[2] = 3.0f;
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    double gc_atol = (dtype() == DType::Float64) ? 1e-5 : 1e-3;
    double gc_rtol = (dtype() == DType::Float64) ? 1e-5 : 1e-3;
    bool passed = gradcheck(f, x, 1e-6, gc_atol, gc_rtol);
    EXPECT_TRUE(passed);
}

// Performance test - measure gradcheck time
TEST_P(GradCheckMultiDTypeTest, PerformanceBenchmark) {
    if (dtype() == DType::Float16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16 gradcheck precision");

    auto f = [](const Variable& x) -> Variable {
        return x * x + x;
    };

    Tensor data = zeros({10}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    for (int64_t i = 0; i < 10; ++i) {
        data_cpu.data<float>()[i] = static_cast<float>(i) * 0.1f;
    }
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    auto start = std::chrono::high_resolution_clock::now();
    bool passed = gradcheck(f, x, 1e-6, 1e-5, 1e-3);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_TRUE(passed);
    EXPECT_LT(duration.count(), 30000);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(GradCheckMultiDTypeTest);
