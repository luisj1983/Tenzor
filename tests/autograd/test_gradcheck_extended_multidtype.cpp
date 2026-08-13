/**
 * @file test_gradcheck_extended_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for extended gradient checking
 *
 * Converted from test_gradcheck_extended.cpp. Tests exercise gradcheck
 * infrastructure (GradCheckResult, GradCheckError, verbose mode, edge cases)
 * across multiple backends and dtypes. Gradcheck tests skip Float16 and BFloat16.
 */

#include <gtest/gtest.h>
#include "tenzor/autograd/gradcheck.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/tenzor.hpp"
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>
#include <sstream>
#include <iostream>

using namespace tenzor;
using namespace tenzor::testing;

class GradCheckExtendedMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        set_grad_enabled(true);
    }
};

// ============================================================================
// SECTION 1: Explicit Constructor Testing
// ============================================================================

TEST_P(GradCheckExtendedMultiDTypeTest, ExplicitGradCheckResultConstruction) {
    GradCheckResult result;

    EXPECT_TRUE(result.passed);
    EXPECT_DOUBLE_EQ(result.max_abs_error, 0.0);
    EXPECT_DOUBLE_EQ(result.max_rel_error, 0.0);
    EXPECT_EQ(result.total_elements, 0);
    EXPECT_EQ(result.failing_elements, 0);
    EXPECT_TRUE(result.fail_indices.empty());
    EXPECT_TRUE(result.numerical_grad.empty());
    EXPECT_TRUE(result.analytical_grad.empty());
    EXPECT_TRUE(result.error_message.empty());
}

TEST_P(GradCheckExtendedMultiDTypeTest, GradCheckResultReturnValue) {
    auto create_result = []() -> GradCheckResult {
        GradCheckResult r;
        r.passed = false;
        r.max_abs_error = 1.23;
        r.total_elements = 10;
        return r;
    };

    GradCheckResult result = create_result();
    EXPECT_FALSE(result.passed);
    EXPECT_DOUBLE_EQ(result.max_abs_error, 1.23);
    EXPECT_EQ(result.total_elements, 10);
}

// ============================================================================
// SECTION 2: Exception Path Testing
// ============================================================================

TEST_P(GradCheckExtendedMultiDTypeTest, GradCheckErrorExceptionPath) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16/BFloat16 gradcheck precision");

    auto failing_func = [](const Variable& x) -> Variable {
        Variable result(x.tensor() * 2.0f, false);
        return result;
    };

    Tensor data = zeros({3}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    float* ptr = data_cpu.data<float>();
    ptr[0] = 1.0f;
    ptr[1] = 2.0f;
    ptr[2] = 3.0f;
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    try {
        bool result = gradcheck(failing_func, x, 1e-6, 1e-5, 1e-3, true);
        FAIL() << "Expected GradCheckError to be thrown";
    } catch (const GradCheckError& e) {
        EXPECT_FALSE(e.result.passed);
        EXPECT_GT(e.result.max_abs_error, 0.0);
        EXPECT_FALSE(e.result.error_message.empty());
        EXPECT_STREQ(e.what(), e.result.error_message.c_str());
    } catch (...) {
        FAIL() << "Expected GradCheckError but got different exception";
    }
}

TEST_P(GradCheckExtendedMultiDTypeTest, GradCheckErrorRequiresGradFalse) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16/BFloat16 gradcheck precision");

    auto f = [](const Variable& x) -> Variable {
        return x * x;
    };

    Tensor data(std::vector<int64_t>{3}, dtype(), device());
    Variable x(data, false);

    EXPECT_THROW({
        gradcheck(f, x, 1e-6, 1e-5, 1e-3, true);
    }, GradCheckError);

    try {
        gradcheck(f, x, 1e-6, 1e-5, 1e-3, true);
    } catch (const GradCheckError& e) {
        std::string msg(e.what());
        EXPECT_TRUE(msg.find("requires_grad") != std::string::npos);
    }
}

// ============================================================================
// SECTION 3: Verbose Function Testing
// ============================================================================

TEST_P(GradCheckExtendedMultiDTypeTest, VerboseModePassingTest) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16/BFloat16 gradcheck precision");

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

    std::ostringstream capture;
    std::streambuf* old_cout = std::cout.rdbuf(capture.rdbuf());

    bool passed = gradcheck_verbose(f, x, 1e-6, 1e-5, 1e-3);

    std::cout.rdbuf(old_cout);

    EXPECT_TRUE(passed);

    std::string output = capture.str();
    EXPECT_TRUE(output.find("Gradient Check") != std::string::npos);
    EXPECT_TRUE(output.find("Input shape") != std::string::npos);
}

TEST_P(GradCheckExtendedMultiDTypeTest, VerboseModeFailingTest) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16/BFloat16 gradcheck precision");

    auto failing_func = [](const Variable& x) -> Variable {
        Variable result(x.tensor() * 3.0f, false);
        return result;
    };

    Tensor data = zeros({5}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    for (int i = 0; i < 5; ++i) {
        data_cpu.data<float>()[i] = static_cast<float>(i + 1);
    }
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    std::ostringstream capture;
    std::streambuf* old_cout = std::cout.rdbuf(capture.rdbuf());

    bool passed = gradcheck_verbose(failing_func, x, 1e-6, 1e-5, 1e-3);

    std::cout.rdbuf(old_cout);

    EXPECT_FALSE(passed);

    std::string output = capture.str();
    EXPECT_TRUE(output.find("FAILED") != std::string::npos ||
                output.find("failed") != std::string::npos);
}

// ============================================================================
// SECTION 4: Non-Scalar Output Testing
// ============================================================================

TEST_P(GradCheckExtendedMultiDTypeTest, NonScalarFunctionOutput) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16/BFloat16 gradcheck precision");

    auto f = [](const Variable& x) -> Variable {
        return x * x;
    };

    Tensor data = zeros({4}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    for (int i = 0; i < 4; ++i) {
        data_cpu.data<float>()[i] = static_cast<float>(i + 1) * 0.5f;
    }
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    bool passed = gradcheck(f, x, 1e-4, 1e-4, 1e-2);
    EXPECT_TRUE(passed);
}

// ============================================================================
// SECTION 5: Float64 Precision Testing
// ============================================================================

TEST_P(GradCheckExtendedMultiDTypeTest, HighPrecisionGradients) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16/BFloat16 gradcheck precision");

    auto f = [](const Variable& x) -> Variable {
        return x * x * x - x * x * 2.0 + x * 3.0;
    };

    Tensor data = zeros({4}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    float* ptr = data_cpu.data<float>();
    ptr[0] = 0.5f;
    ptr[1] = 1.0f;
    ptr[2] = 1.5f;
    ptr[3] = 2.0f;
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    double eps = (dtype() == DType::Float64) ? 1e-7 : 1e-4;
    double gc_at = (dtype() == DType::Float64) ? 1e-6 : 1e-3;
    double gc_rt = (dtype() == DType::Float64) ? 1e-4 : 1e-2;
    auto result = gradcheck_detailed(f, x, eps, gc_at, gc_rt);

    EXPECT_TRUE(result.passed);
    double expected_tol = (dtype() == DType::Float64) ? 1e-5 : 1e-1;
    EXPECT_LT(result.max_abs_error, expected_tol);
}

TEST_P(GradCheckExtendedMultiDTypeTest, NumericalGradientDirect) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16/BFloat16 gradcheck precision");

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

    double eps = (dtype() == DType::Float64) ? 1e-7 : 1e-6;
    Tensor num_grad = numerical_gradient(f, x, eps);

    auto num_grad_cpu = num_grad.to(Device::cpu()).to(DType::Float32);
    const float* grad_ptr = num_grad_cpu.data<float>();

    float tol = (dtype() == DType::Float64) ? 1e-4f : 1e-2f;
    EXPECT_NEAR(grad_ptr[0], 2.0f, tol);
    EXPECT_NEAR(grad_ptr[1], 4.0f, tol);
    EXPECT_NEAR(grad_ptr[2], 6.0f, tol);
}

// ============================================================================
// SECTION 6: Error Condition Testing
// ============================================================================

TEST_P(GradCheckExtendedMultiDTypeTest, ShapeMismatchError) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16/BFloat16 gradcheck precision");

    Tensor num_grad = zeros({3}, dtype(), device());
    Tensor ana_grad = zeros({4}, dtype(), device());

    GradCheckResult result = compare_gradients(num_grad, ana_grad, 1e-5, 1e-3);

    EXPECT_FALSE(result.passed);
    EXPECT_TRUE(result.error_message.find("shape") != std::string::npos);
}

TEST_P(GradCheckExtendedMultiDTypeTest, MultidimensionalShapeMismatch) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16/BFloat16 gradcheck precision");

    Tensor num_grad = zeros({2, 3}, dtype(), device());
    Tensor ana_grad = zeros({2, 4}, dtype(), device());

    GradCheckResult result = compare_gradients(num_grad, ana_grad, 1e-5, 1e-3);

    EXPECT_FALSE(result.passed);
    EXPECT_TRUE(result.error_message.find("shape") != std::string::npos);
}

TEST_P(GradCheckExtendedMultiDTypeTest, MissingGradientError) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16/BFloat16 gradcheck precision");

    auto unused_input_func = [this](const Variable& x) -> Variable {
        Tensor constant = zeros({1}, dtype(), device());
        auto constant_cpu = constant.to(Device::cpu()).to(DType::Float32);
        constant_cpu.data<float>()[0] = 5.0f;
        constant = constant_cpu.to(dtype()).to(device());
        return Variable(constant, false);
    };

    Tensor data = zeros({3}, dtype(), device());
    Variable x(data, true);

    GradCheckResult result = gradcheck_detailed(unused_input_func, x);

    EXPECT_FALSE(result.passed);
    EXPECT_TRUE(result.error_message.find("gradient") != std::string::npos ||
                result.error_message.find("input") != std::string::npos);
}

// ============================================================================
// SECTION 7: Edge Cases and Numerical Stability
// ============================================================================

TEST_P(GradCheckExtendedMultiDTypeTest, LargeTensorSampling) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16/BFloat16 gradcheck precision");

    auto f = [](const Variable& x) -> Variable {
        return x * x;
    };

    Tensor data = zeros({20}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    for (int i = 0; i < 20; ++i) {
        data_cpu.data<float>()[i] = static_cast<float>(i) * 0.1f;
    }
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    GradCheckResult result = gradcheck_detailed(f, x, 1e-4, 1e-4, 1e-2);

    EXPECT_TRUE(result.passed);
    EXPECT_EQ(result.total_elements, 20);
    EXPECT_LE(result.numerical_grad.size(), 10);
    EXPECT_LE(result.analytical_grad.size(), 10);
}

TEST_P(GradCheckExtendedMultiDTypeTest, SmallAnalyticalValueRelativeError) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16/BFloat16 gradcheck precision");

    auto f = [](const Variable& x) -> Variable {
        return x * 0.001f;
    };

    Tensor data = zeros({3}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    float* ptr = data_cpu.data<float>();
    ptr[0] = 0.01f;
    ptr[1] = 0.02f;
    ptr[2] = 0.03f;
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    GradCheckResult result = gradcheck_detailed(f, x, 1e-6, 1e-5, 1e-1);

    EXPECT_GE(result.max_rel_error, 0.0);
    EXPECT_LT(result.max_rel_error, 1e10);
}

TEST_P(GradCheckExtendedMultiDTypeTest, FailingElementsTruncation) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16/BFloat16 gradcheck precision");

    auto bad_func = [](const Variable& x) -> Variable {
        Variable result(x.tensor() * 100.0f, false);
        return result;
    };

    Tensor data = zeros({25}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    for (int i = 0; i < 25; ++i) {
        data_cpu.data<float>()[i] = static_cast<float>(i + 1);
    }
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    GradCheckResult result = gradcheck_detailed(bad_func, x, 1e-6, 1e-5, 1e-3);

    EXPECT_FALSE(result.passed);
    EXPECT_GT(result.failing_elements, 0);
    EXPECT_LE(result.fail_indices.size(), 10);
}

TEST_P(GradCheckExtendedMultiDTypeTest, CompareGradientsTyped) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16/BFloat16 gradcheck precision");

    Tensor num_grad = zeros({4}, dtype(), device());
    Tensor ana_grad = zeros({4}, dtype(), device());

    auto num_cpu = num_grad.to(Device::cpu()).to(DType::Float32);
    auto ana_cpu = ana_grad.to(Device::cpu()).to(DType::Float32);

    float* num_ptr = num_cpu.data<float>();
    float* ana_ptr = ana_cpu.data<float>();

    for (int i = 0; i < 4; ++i) {
        num_ptr[i] = static_cast<float>(i + 1);
        ana_ptr[i] = static_cast<float>(i + 1) + 1e-7f;
    }

    num_grad = num_cpu.to(dtype()).to(device());
    ana_grad = ana_cpu.to(dtype()).to(device());

    GradCheckResult result = compare_gradients(num_grad, ana_grad, 1e-6, 1e-4);

    EXPECT_TRUE(result.passed);
}

// ============================================================================
// SECTION 8: Multi-dimensional & Shape Preservation
// ============================================================================

TEST_P(GradCheckExtendedMultiDTypeTest, IndirectZerosLikeTensor) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16/BFloat16 gradcheck precision");

    auto f = [](const Variable& x) -> Variable {
        return x * x;
    };

    Tensor data = zeros({2, 3, 4}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    for (int i = 0; i < 24; ++i) {
        data_cpu.data<float>()[i] = static_cast<float>(i) * 0.05f;
    }
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    Tensor num_grad = numerical_gradient(f, x, 1e-4);

    auto num_grad_shape = num_grad.shape();
    auto data_shape = data.shape();
    EXPECT_TRUE(std::equal(num_grad_shape.begin(), num_grad_shape.end(),
                           data_shape.begin(), data_shape.end()));
    EXPECT_EQ(num_grad.dtype(), data.dtype());
    EXPECT_EQ(num_grad.device(), data.device());
}

TEST_P(GradCheckExtendedMultiDTypeTest, IndirectExtractScalarZeroDim) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16/BFloat16 gradcheck precision");

    auto f = [](const Variable& x) -> Variable {
        return tenzor::sum(x);
    };

    Tensor data = zeros({3}, dtype(), device());
    auto data_cpu = data.to(Device::cpu()).to(DType::Float32);
    data_cpu.data<float>()[0] = 1.0f;
    data_cpu.data<float>()[1] = 2.0f;
    data_cpu.data<float>()[2] = 3.0f;
    data = data_cpu.to(dtype()).to(device());

    Variable x(data, true);

    bool passed = gradcheck(f, x, 1e-4, 1e-4, 1e-2);
    EXPECT_TRUE(passed);
}

// ============================================================================
// SECTION 9: Error Message Generation
// ============================================================================

TEST_P(GradCheckExtendedMultiDTypeTest, ErrorMessageGeneration) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) SKIP_WITH_REASON(::tenzor::testing::SkipReason::GradcheckFDPrecision, "Float16/BFloat16 gradcheck precision");

    Tensor num_grad = zeros({5}, dtype(), device());
    Tensor ana_grad = zeros({5}, dtype(), device());

    auto num_cpu = num_grad.to(Device::cpu()).to(DType::Float32);
    auto ana_cpu = ana_grad.to(Device::cpu()).to(DType::Float32);

    for (int i = 0; i < 5; ++i) {
        num_cpu.data<float>()[i] = static_cast<float>(i);
        ana_cpu.data<float>()[i] = static_cast<float>(i) + 10.0f;
    }

    num_grad = num_cpu.to(dtype()).to(device());
    ana_grad = ana_cpu.to(dtype()).to(device());

    GradCheckResult result = compare_gradients(num_grad, ana_grad, 1e-5, 1e-3);

    EXPECT_FALSE(result.passed);
    EXPECT_FALSE(result.error_message.empty());

    std::string msg = result.error_message;
    EXPECT_TRUE(msg.find("failed") != std::string::npos);
    EXPECT_TRUE(msg.find("absolute error") != std::string::npos);
    EXPECT_TRUE(msg.find("relative error") != std::string::npos);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(GradCheckExtendedMultiDTypeTest);
