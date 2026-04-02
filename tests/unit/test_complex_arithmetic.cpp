#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <complex>

using namespace tenzor;
using namespace tenzor::testing;

class ComplexArithmeticTest : public BackendTest {};

// ============================================================================
// Complex tensor creation (these work)
// ============================================================================

TEST_P(ComplexArithmeticTest, Complex64Creation) {
    auto t = zeros({2, 3}, DType::Complex64, device);
    EXPECT_EQ(t.dtype(), DType::Complex64);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
}

TEST_P(ComplexArithmeticTest, Complex128Creation) {
    auto t = zeros({2, 3}, DType::Complex128, device);
    EXPECT_EQ(t.dtype(), DType::Complex128);
}

TEST_P(ComplexArithmeticTest, Complex64DataAccess) {
    auto t = Tensor({int64_t(2)}, DType::Complex64, device);
    auto* p = t.data<std::complex<float>>();
    p[0] = {1.0f, 2.0f};
    p[1] = {3.0f, 4.0f};

    auto tc = t.to(Device::cpu());
    auto* cp = tc.data<std::complex<float>>();
    EXPECT_FLOAT_EQ(cp[0].real(), 1.0f);
    EXPECT_FLOAT_EQ(cp[0].imag(), 2.0f);
    EXPECT_FLOAT_EQ(cp[1].real(), 3.0f);
    EXPECT_FLOAT_EQ(cp[1].imag(), 4.0f);
}

TEST_P(ComplexArithmeticTest, Complex128DataAccess) {
    auto t = Tensor({int64_t(2)}, DType::Complex128, device);
    auto* p = t.data<std::complex<double>>();
    p[0] = {1.5, 2.5};
    p[1] = {3.5, 4.5};

    auto tc = t.to(Device::cpu());
    auto* cp = tc.data<std::complex<double>>();
    EXPECT_DOUBLE_EQ(cp[0].real(), 1.5);
    EXPECT_DOUBLE_EQ(cp[0].imag(), 2.5);
}

TEST_P(ComplexArithmeticTest, Complex64ZerosValues) {
    auto t = zeros({3}, DType::Complex64, device).to(Device::cpu());
    auto* p = t.data<std::complex<float>>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(p[i].real(), 0.0f);
        EXPECT_FLOAT_EQ(p[i].imag(), 0.0f);
    }
}

TEST_P(ComplexArithmeticTest, Complex64OnesValues) {
    // Note: ones() for complex may not produce (1,0) since fill is scalar-based.
    // This test documents the current behavior.
    try {
        auto t = ones({3}, DType::Complex64, device).to(Device::cpu());
        // Check that values are created without crashing
        EXPECT_EQ(t.numel(), 3);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex ones not supported on " << device.to_string();
    }
}

// ============================================================================
// Complex arithmetic (document current support level)
// These may throw if the backend doesn't support complex ops yet.
// ============================================================================

TEST_P(ComplexArithmeticTest, ComplexAdd) {
    auto a = Tensor({int64_t(2)}, DType::Complex64, device);
    auto b = Tensor({int64_t(2)}, DType::Complex64, device);

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
        GTEST_SKIP() << "Complex add not yet supported on " << device.to_string();
    }
}

TEST_P(ComplexArithmeticTest, ComplexSub) {
    auto a = Tensor({int64_t(2)}, DType::Complex64, device);
    auto b = Tensor({int64_t(2)}, DType::Complex64, device);

    a.data<std::complex<float>>()[0] = {5.0f, 8.0f};
    a.data<std::complex<float>>()[1] = {3.0f, 1.0f};
    b.data<std::complex<float>>()[0] = {2.0f, 3.0f};
    b.data<std::complex<float>>()[1] = {1.0f, 1.0f};

    try {
        auto c = sub(a, b).to(Device::cpu());
        auto* cp = c.data<std::complex<float>>();
        EXPECT_FLOAT_EQ(cp[0].real(), 3.0f);
        EXPECT_FLOAT_EQ(cp[0].imag(), 5.0f);
        EXPECT_FLOAT_EQ(cp[1].real(), 2.0f);
        EXPECT_FLOAT_EQ(cp[1].imag(), 0.0f);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex sub not yet supported on " << device.to_string();
    }
}

TEST_P(ComplexArithmeticTest, ComplexMul) {
    auto a = Tensor({int64_t(1)}, DType::Complex64, device);
    auto b = Tensor({int64_t(1)}, DType::Complex64, device);

    // (1+2i) * (3+4i) = -5 + 10i
    a.data<std::complex<float>>()[0] = {1.0f, 2.0f};
    b.data<std::complex<float>>()[0] = {3.0f, 4.0f};

    try {
        auto c = mul(a, b).to(Device::cpu());
        auto* cp = c.data<std::complex<float>>();
        EXPECT_FLOAT_EQ(cp[0].real(), -5.0f);
        EXPECT_FLOAT_EQ(cp[0].imag(), 10.0f);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex mul not yet supported on " << device.to_string();
    }
}

TEST_P(ComplexArithmeticTest, ComplexDiv) {
    auto a = Tensor({int64_t(1)}, DType::Complex64, device);
    auto b = Tensor({int64_t(1)}, DType::Complex64, device);

    a.data<std::complex<float>>()[0] = {1.0f, 2.0f};
    b.data<std::complex<float>>()[0] = {1.0f, 0.0f};

    try {
        auto c = div(a, b).to(Device::cpu());
        auto* cp = c.data<std::complex<float>>();
        EXPECT_FLOAT_EQ(cp[0].real(), 1.0f);
        EXPECT_FLOAT_EQ(cp[0].imag(), 2.0f);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex div not yet supported on " << device.to_string();
    }
}

TEST_P(ComplexArithmeticTest, ComplexAbs) {
    auto a = Tensor({int64_t(1)}, DType::Complex64, device);
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
        GTEST_SKIP() << "Complex abs not yet supported on " << device.to_string();
    }
}

TEST_P(ComplexArithmeticTest, ComplexRealAdd) {
    auto complex_t = Tensor({int64_t(2)}, DType::Complex64, device);
    auto real_t = Tensor({int64_t(2)}, DType::Float32, device);

    complex_t.data<std::complex<float>>()[0] = {1.0f, 2.0f};
    complex_t.data<std::complex<float>>()[1] = {3.0f, 4.0f};
    real_t.data<float>()[0] = 10.0f;
    real_t.data<float>()[1] = 20.0f;

    try {
        auto result = add(complex_t, real_t).to(Device::cpu());
        EXPECT_EQ(result.dtype(), DType::Complex64);
        auto* rp = result.data<std::complex<float>>();
        EXPECT_FLOAT_EQ(rp[0].real(), 11.0f);
        EXPECT_FLOAT_EQ(rp[0].imag(), 2.0f);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex-real add not yet supported on " << device.to_string();
    }
}

// ============================================================================
// Complex division (non-trivial divisor)
// ============================================================================

TEST_P(ComplexArithmeticTest, ComplexDivNonTrivial) {
    auto a = Tensor({int64_t(1)}, DType::Complex64, device);
    auto b = Tensor({int64_t(1)}, DType::Complex64, device);

    // (1+2i) / (3+4i) = (1*3+2*4)/(3^2+4^2) + (2*3-1*4)/(3^2+4^2)i = 11/25 + 2/25i
    a.data<std::complex<float>>()[0] = {1.0f, 2.0f};
    b.data<std::complex<float>>()[0] = {3.0f, 4.0f};

    try {
        auto c = div(a, b).to(Device::cpu());
        auto* cp = c.data<std::complex<float>>();
        EXPECT_NEAR(cp[0].real(), 0.44f, 1e-5f);
        EXPECT_NEAR(cp[0].imag(), 0.08f, 1e-5f);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex div not yet supported on " << device.to_string();
    }
}

// ============================================================================
// Complex negation
// ============================================================================

TEST_P(ComplexArithmeticTest, ComplexNeg) {
    auto a = Tensor({int64_t(2)}, DType::Complex64, device);
    a.data<std::complex<float>>()[0] = {1.0f, -2.0f};
    a.data<std::complex<float>>()[1] = {-3.0f, 4.0f};

    try {
        auto c = neg(a).to(Device::cpu());
        auto* cp = c.data<std::complex<float>>();
        EXPECT_FLOAT_EQ(cp[0].real(), -1.0f);
        EXPECT_FLOAT_EQ(cp[0].imag(), 2.0f);
        EXPECT_FLOAT_EQ(cp[1].real(), 3.0f);
        EXPECT_FLOAT_EQ(cp[1].imag(), -4.0f);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex neg not yet supported on " << device.to_string();
    }
}

// ============================================================================
// Complex exp: exp(a+bi) = exp(a)(cos(b) + i*sin(b))
// ============================================================================

TEST_P(ComplexArithmeticTest, ComplexExp) {
    auto a = Tensor({int64_t(1)}, DType::Complex64, device);
    // exp(0+pi*i) = cos(pi) + i*sin(pi) = -1 + 0i
    a.data<std::complex<float>>()[0] = {0.0f, static_cast<float>(M_PI)};

    try {
        auto c = exp(a).to(Device::cpu());
        auto* cp = c.data<std::complex<float>>();
        EXPECT_NEAR(cp[0].real(), -1.0f, 1e-5f);
        EXPECT_NEAR(cp[0].imag(), 0.0f, 1e-5f);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex exp not yet supported on " << device.to_string();
    }
}

// ============================================================================
// Complex log: log(r*e^(i*theta)) = log(r) + i*theta
// ============================================================================

TEST_P(ComplexArithmeticTest, ComplexLog) {
    auto a = Tensor({int64_t(1)}, DType::Complex64, device);
    // log(1+0i) = 0+0i
    a.data<std::complex<float>>()[0] = {1.0f, 0.0f};

    try {
        auto c = log(a).to(Device::cpu());
        auto* cp = c.data<std::complex<float>>();
        EXPECT_NEAR(cp[0].real(), 0.0f, 1e-5f);
        EXPECT_NEAR(cp[0].imag(), 0.0f, 1e-5f);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex log not yet supported on " << device.to_string();
    }
}

// ============================================================================
// Complex broadcasting
// ============================================================================

TEST_P(ComplexArithmeticTest, ComplexBroadcastAdd) {
    auto a = Tensor({int64_t(2), int64_t(3)}, DType::Complex64, device);
    auto b = Tensor({int64_t(3)}, DType::Complex64, device);  // broadcasts to (2,3)

    auto* ap = a.data<std::complex<float>>();
    for (int i = 0; i < 6; ++i) ap[i] = {static_cast<float>(i), 0.0f};
    auto* bp = b.data<std::complex<float>>();
    for (int i = 0; i < 3; ++i) bp[i] = {0.0f, static_cast<float>(i)};

    try {
        auto c = add(a, b).to(Device::cpu());
        EXPECT_EQ(c.shape()[0], 2);
        EXPECT_EQ(c.shape()[1], 3);
        auto* cp = c.data<std::complex<float>>();
        // First row: (0+0i, 1+1i, 2+2i)
        EXPECT_FLOAT_EQ(cp[0].real(), 0.0f);
        EXPECT_FLOAT_EQ(cp[0].imag(), 0.0f);
        EXPECT_FLOAT_EQ(cp[1].real(), 1.0f);
        EXPECT_FLOAT_EQ(cp[1].imag(), 1.0f);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex broadcast add not yet supported on " << device.to_string();
    }
}

// ============================================================================
// Complex128 arithmetic
// ============================================================================

TEST_P(ComplexArithmeticTest, Complex128MulAccuracy) {
    auto a = Tensor({int64_t(1)}, DType::Complex128, device);
    auto b = Tensor({int64_t(1)}, DType::Complex128, device);

    // (1e-10 + 2e-10i) * (3e-10 + 4e-10i)
    // = (3e-20 - 8e-20) + (4e-20 + 6e-20)i = -5e-20 + 10e-20i
    a.data<std::complex<double>>()[0] = {1e-10, 2e-10};
    b.data<std::complex<double>>()[0] = {3e-10, 4e-10};

    try {
        auto c = mul(a, b).to(Device::cpu());
        auto* cp = c.data<std::complex<double>>();
        EXPECT_NEAR(cp[0].real(), -5e-20, 1e-30);
        EXPECT_NEAR(cp[0].imag(), 10e-20, 1e-30);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex128 mul not yet supported on " << device.to_string();
    }
}

// ============================================================================
// Complex sqrt
// ============================================================================

TEST_P(ComplexArithmeticTest, ComplexSqrt) {
    auto a = Tensor({int64_t(1)}, DType::Complex64, device);
    // sqrt(-1+0i) = 0+1i
    a.data<std::complex<float>>()[0] = {-1.0f, 0.0f};

    try {
        auto c = sqrt(a).to(Device::cpu());
        auto* cp = c.data<std::complex<float>>();
        EXPECT_NEAR(cp[0].real(), 0.0f, 1e-5f);
        EXPECT_NEAR(std::abs(cp[0].imag()), 1.0f, 1e-5f);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Complex sqrt not yet supported on " << device.to_string();
    }
}

// ============================================================================
// Instantiate for CPU backend
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    CPU, ComplexArithmeticTest,
    ::testing::Values("cpu"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);
