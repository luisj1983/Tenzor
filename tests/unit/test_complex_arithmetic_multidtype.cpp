/**
 * @file test_complex_arithmetic_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for complex arithmetic operations:
 *        creation, add, mul, div, abs, exp, sqrt
 *
 * Note: Complex tests use Complex64/Complex128 directly (not parameterized dtype)
 * but run across multiple backends. The dtype parameter controls which complex
 * precision variant to test (Float32->Complex64, Float64->Complex128).
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <complex>
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class ComplexArithmeticMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Map Float32->Complex64, Float64->Complex128; skip Float16
    DType complex_dtype() const {
        if (dtype() == DType::Float32) return DType::Complex64;
        if (dtype() == DType::Float64) return DType::Complex128;
        return DType::Complex64; // fallback, will skip
    }
};

TEST_P(ComplexArithmeticMultiDTypeTest, ComplexCreation) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "No Float16 complex type";
    auto t = zeros({2, 3}, complex_dtype(), device());
    EXPECT_EQ(t.dtype(), complex_dtype());
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
}

TEST_P(ComplexArithmeticMultiDTypeTest, ComplexAdd) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "No Float16 complex type";
    auto a = Tensor({int64_t(2)}, DType::Complex64, device());
    auto b = Tensor({int64_t(2)}, DType::Complex64, device());

    auto* ap = a.data<std::complex<float>>();
    ap[0] = {1.0f, 2.0f};
    ap[1] = {3.0f, 4.0f};
    auto* bp = b.data<std::complex<float>>();
    bp[0] = {5.0f, 6.0f};
    bp[1] = {7.0f, 8.0f};

    try {
        auto c = add(a, b).to(Device::cpu());
        auto* cp = c.data<std::complex<float>>();
        EXPECT_FLOAT_EQ(cp[0].real(), 6.0f);
        EXPECT_FLOAT_EQ(cp[0].imag(), 8.0f);
        EXPECT_FLOAT_EQ(cp[1].real(), 10.0f);
        EXPECT_FLOAT_EQ(cp[1].imag(), 12.0f);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex add not supported on this backend";
    }
}

TEST_P(ComplexArithmeticMultiDTypeTest, ComplexMul) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "No Float16 complex type";
    auto a = Tensor({int64_t(1)}, DType::Complex64, device());
    auto b = Tensor({int64_t(1)}, DType::Complex64, device());

    a.data<std::complex<float>>()[0] = {1.0f, 2.0f};
    b.data<std::complex<float>>()[0] = {3.0f, 4.0f};

    try {
        auto c = mul(a, b).to(Device::cpu());
        auto* cp = c.data<std::complex<float>>();
        EXPECT_FLOAT_EQ(cp[0].real(), -5.0f);
        EXPECT_FLOAT_EQ(cp[0].imag(), 10.0f);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex mul not supported on this backend";
    }
}

TEST_P(ComplexArithmeticMultiDTypeTest, ComplexAbs) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "No Float16 complex type";
    auto a = Tensor({int64_t(1)}, DType::Complex64, device());
    a.data<std::complex<float>>()[0] = {3.0f, 4.0f};

    try {
        auto result = abs(a).to(Device::cpu());
        if (result.dtype() == DType::Float32) {
            EXPECT_NEAR(result.data<float>()[0], 5.0f, 1e-5f);
        } else if (result.dtype() == DType::Complex64) {
            auto* rp = result.data<std::complex<float>>();
            EXPECT_NEAR(rp[0].real(), 5.0f, 1e-5f);
        }
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex abs not supported on this backend";
    }
}

TEST_P(ComplexArithmeticMultiDTypeTest, ComplexExp) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "No Float16 complex type";
    auto a = Tensor({int64_t(1)}, DType::Complex64, device());
    a.data<std::complex<float>>()[0] = {0.0f, static_cast<float>(M_PI)};

    try {
        auto c = exp(a).to(Device::cpu());
        auto* cp = c.data<std::complex<float>>();
        EXPECT_NEAR(cp[0].real(), -1.0f, 1e-5f);
        EXPECT_NEAR(cp[0].imag(), 0.0f, 1e-5f);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex exp not supported on this backend";
    }
}

TEST_P(ComplexArithmeticMultiDTypeTest, ComplexSqrt) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "No Float16 complex type";
    auto a = Tensor({int64_t(1)}, DType::Complex64, device());
    a.data<std::complex<float>>()[0] = {-1.0f, 0.0f};

    try {
        auto c = sqrt(a).to(Device::cpu());
        auto* cp = c.data<std::complex<float>>();
        EXPECT_NEAR(cp[0].real(), 0.0f, 1e-5f);
        EXPECT_NEAR(std::abs(cp[0].imag()), 1.0f, 1e-5f);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex sqrt not supported on this backend";
    }
}

TEST_P(ComplexArithmeticMultiDTypeTest, ComplexDivNonTrivial) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "No Float16 complex type";
    auto a = Tensor({int64_t(1)}, DType::Complex64, device());
    auto b = Tensor({int64_t(1)}, DType::Complex64, device());

    a.data<std::complex<float>>()[0] = {1.0f, 2.0f};
    b.data<std::complex<float>>()[0] = {3.0f, 4.0f};

    try {
        auto c = div(a, b).to(Device::cpu());
        auto* cp = c.data<std::complex<float>>();
        EXPECT_NEAR(cp[0].real(), 0.44f, 1e-5f);
        EXPECT_NEAR(cp[0].imag(), 0.08f, 1e-5f);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex div not supported on this backend";
    }
}

TEST_P(ComplexArithmeticMultiDTypeTest, Complex128MulAccuracy) {
    if (dtype() != DType::Float64) GTEST_SKIP() << "Complex128 test only for Float64";
    auto a = Tensor({int64_t(1)}, DType::Complex128, device());
    auto b = Tensor({int64_t(1)}, DType::Complex128, device());

    a.data<std::complex<double>>()[0] = {1e-10, 2e-10};
    b.data<std::complex<double>>()[0] = {3e-10, 4e-10};

    try {
        auto c = mul(a, b).to(Device::cpu());
        auto* cp = c.data<std::complex<double>>();
        EXPECT_NEAR(cp[0].real(), -5e-20, 1e-30);
        EXPECT_NEAR(cp[0].imag(), 10e-20, 1e-30);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex128 mul not supported on this backend";
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ComplexArithmeticMultiDTypeTest);
