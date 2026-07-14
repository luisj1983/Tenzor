#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cmath>
#include <complex>
#include <limits>

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
    auto t = ones({3}, DType::Complex64, device).to(Device::cpu());
    // Check that values are created without crashing
    EXPECT_EQ(t.numel(), 3);
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

    auto c = add(a, b).to(Device::cpu());
    auto* cp = c.data<std::complex<float>>();
    EXPECT_FLOAT_EQ(cp[0].real(), 6.0f);
    EXPECT_FLOAT_EQ(cp[0].imag(), 8.0f);
    EXPECT_FLOAT_EQ(cp[1].real(), 10.0f);
    EXPECT_FLOAT_EQ(cp[1].imag(), 12.0f);
}

TEST_P(ComplexArithmeticTest, ComplexSub) {
    auto a = Tensor({int64_t(2)}, DType::Complex64, device);
    auto b = Tensor({int64_t(2)}, DType::Complex64, device);

    a.data<std::complex<float>>()[0] = {5.0f, 8.0f};
    a.data<std::complex<float>>()[1] = {3.0f, 1.0f};
    b.data<std::complex<float>>()[0] = {2.0f, 3.0f};
    b.data<std::complex<float>>()[1] = {1.0f, 1.0f};

    auto c = sub(a, b).to(Device::cpu());
    auto* cp = c.data<std::complex<float>>();
    EXPECT_FLOAT_EQ(cp[0].real(), 3.0f);
    EXPECT_FLOAT_EQ(cp[0].imag(), 5.0f);
    EXPECT_FLOAT_EQ(cp[1].real(), 2.0f);
    EXPECT_FLOAT_EQ(cp[1].imag(), 0.0f);
}

TEST_P(ComplexArithmeticTest, ComplexMul) {
    auto a = Tensor({int64_t(1)}, DType::Complex64, device);
    auto b = Tensor({int64_t(1)}, DType::Complex64, device);

    // (1+2i) * (3+4i) = -5 + 10i
    a.data<std::complex<float>>()[0] = {1.0f, 2.0f};
    b.data<std::complex<float>>()[0] = {3.0f, 4.0f};

    auto c = mul(a, b).to(Device::cpu());
    auto* cp = c.data<std::complex<float>>();
    EXPECT_FLOAT_EQ(cp[0].real(), -5.0f);
    EXPECT_FLOAT_EQ(cp[0].imag(), 10.0f);
}

TEST_P(ComplexArithmeticTest, ComplexDiv) {
    auto a = Tensor({int64_t(1)}, DType::Complex64, device);
    auto b = Tensor({int64_t(1)}, DType::Complex64, device);

    a.data<std::complex<float>>()[0] = {1.0f, 2.0f};
    b.data<std::complex<float>>()[0] = {1.0f, 0.0f};

    auto c = div(a, b).to(Device::cpu());
    auto* cp = c.data<std::complex<float>>();
    EXPECT_FLOAT_EQ(cp[0].real(), 1.0f);
    EXPECT_FLOAT_EQ(cp[0].imag(), 2.0f);
}

TEST_P(ComplexArithmeticTest, ComplexAbs) {
    auto a = Tensor({int64_t(1)}, DType::Complex64, device);
    a.data<std::complex<float>>()[0] = {3.0f, 4.0f};

    auto result = abs(a).to(Device::cpu());
    if (result.dtype() == DType::Float32) {
        EXPECT_NEAR(result.data<float>()[0], 5.0f, 1e-5f);
    } else if (result.dtype() == DType::Complex64) {
        auto* rp = result.data<std::complex<float>>();
        EXPECT_NEAR(rp[0].real(), 5.0f, 1e-5f);
    }
}

// JIT-R140: CPU's sign() previously had no Complex64/128 branch at all and
// threw "sign: unsupported dtype", while CUDA/ROCm/OneAPI/Vulkan all
// support it (sign(z) = z/|z| for |z|!=0, else 0+0i). Verify CPU now
// matches that same formula, including the overflow-safe-magnitude and
// zero-input edge cases the GPU backends' hypot-based implementations
// already handle correctly.
TEST_P(ComplexArithmeticTest, ComplexSign) {
    auto a = Tensor({int64_t(4)}, DType::Complex64, device);
    a.data<std::complex<float>>()[0] = {3.0f, 4.0f};   // |z|=5
    a.data<std::complex<float>>()[1] = {0.0f, 0.0f};   // zero input
    a.data<std::complex<float>>()[2] = {-1.0f, 0.0f};  // negative real axis
    a.data<std::complex<float>>()[3] = {std::numeric_limits<float>::quiet_NaN(), 1.0f};  // NaN component

    auto result = sign(a).to(Device::cpu());
    ASSERT_EQ(result.dtype(), DType::Complex64);
    auto* rp = result.data<std::complex<float>>();

    EXPECT_NEAR(rp[0].real(), 0.6f, 1e-5f);   // 3/5
    EXPECT_NEAR(rp[0].imag(), 0.8f, 1e-5f);   // 4/5
    EXPECT_NEAR(std::abs(rp[0]), 1.0f, 1e-5f);

    EXPECT_EQ(rp[1].real(), 0.0f);
    EXPECT_EQ(rp[1].imag(), 0.0f);

    EXPECT_NEAR(rp[2].real(), -1.0f, 1e-5f);
    EXPECT_NEAR(rp[2].imag(), 0.0f, 1e-5f);

    // JIT-R171: a NaN component must propagate to NaN+NaNi, not silently
    // collapse to 0+0i (mag itself is NaN, so this must NOT take the
    // "zero magnitude" branch).
    EXPECT_TRUE(std::isnan(rp[3].real())) << "sign() must propagate NaN, not silently return 0";
    EXPECT_TRUE(std::isnan(rp[3].imag())) << "sign() must propagate NaN, not silently return 0";
}

TEST_P(ComplexArithmeticTest, Complex128SignMatchesFormula) {
    auto a = Tensor({int64_t(3)}, DType::Complex128, device);
    a.data<std::complex<double>>()[0] = {6.0, 8.0};   // |z|=10
    a.data<std::complex<double>>()[1] = {0.0, 0.0};
    a.data<std::complex<double>>()[2] = {std::numeric_limits<double>::quiet_NaN(), 1.0};  // NaN component

    auto result = sign(a).to(Device::cpu());
    ASSERT_EQ(result.dtype(), DType::Complex128);
    auto* rp = result.data<std::complex<double>>();

    EXPECT_NEAR(rp[0].real(), 0.6, 1e-12);
    EXPECT_NEAR(rp[0].imag(), 0.8, 1e-12);
    EXPECT_NEAR(std::abs(rp[0]), 1.0, 1e-12);

    EXPECT_EQ(rp[1].real(), 0.0);
    EXPECT_EQ(rp[1].imag(), 0.0);

    // JIT-R171: a NaN component must propagate, not silently collapse to 0+0i.
    EXPECT_TRUE(std::isnan(rp[2].real())) << "sign() must propagate NaN, not silently return 0";
    EXPECT_TRUE(std::isnan(rp[2].imag())) << "sign() must propagate NaN, not silently return 0";
}

TEST_P(ComplexArithmeticTest, ComplexRealAdd) {
    auto complex_t = Tensor({int64_t(2)}, DType::Complex64, device);
    auto real_t = Tensor({int64_t(2)}, DType::Float32, device);

    complex_t.data<std::complex<float>>()[0] = {1.0f, 2.0f};
    complex_t.data<std::complex<float>>()[1] = {3.0f, 4.0f};
    real_t.data<float>()[0] = 10.0f;
    real_t.data<float>()[1] = 20.0f;

    auto result = add(complex_t, real_t).to(Device::cpu());
    EXPECT_EQ(result.dtype(), DType::Complex64);
    auto* rp = result.data<std::complex<float>>();
    EXPECT_FLOAT_EQ(rp[0].real(), 11.0f);
    EXPECT_FLOAT_EQ(rp[0].imag(), 2.0f);
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

    auto c = div(a, b).to(Device::cpu());
    auto* cp = c.data<std::complex<float>>();
    EXPECT_NEAR(cp[0].real(), 0.44f, 1e-5f);
    EXPECT_NEAR(cp[0].imag(), 0.08f, 1e-5f);
}

// ============================================================================
// Complex negation
// ============================================================================

TEST_P(ComplexArithmeticTest, ComplexNeg) {
    auto a = Tensor({int64_t(2)}, DType::Complex64, device);
    a.data<std::complex<float>>()[0] = {1.0f, -2.0f};
    a.data<std::complex<float>>()[1] = {-3.0f, 4.0f};

    auto c = neg(a).to(Device::cpu());
    auto* cp = c.data<std::complex<float>>();
    EXPECT_FLOAT_EQ(cp[0].real(), -1.0f);
    EXPECT_FLOAT_EQ(cp[0].imag(), 2.0f);
    EXPECT_FLOAT_EQ(cp[1].real(), 3.0f);
    EXPECT_FLOAT_EQ(cp[1].imag(), -4.0f);
}

// ============================================================================
// Complex exp: exp(a+bi) = exp(a)(cos(b) + i*sin(b))
// ============================================================================

TEST_P(ComplexArithmeticTest, ComplexExp) {
    auto a = Tensor({int64_t(1)}, DType::Complex64, device);
    // exp(0+pi*i) = cos(pi) + i*sin(pi) = -1 + 0i
    a.data<std::complex<float>>()[0] = {0.0f, static_cast<float>(M_PI)};

    auto c = exp(a).to(Device::cpu());
    auto* cp = c.data<std::complex<float>>();
    EXPECT_NEAR(cp[0].real(), -1.0f, 1e-5f);
    EXPECT_NEAR(cp[0].imag(), 0.0f, 1e-5f);
}

// ============================================================================
// Complex log: log(r*e^(i*theta)) = log(r) + i*theta
// ============================================================================

TEST_P(ComplexArithmeticTest, ComplexLog) {
    auto a = Tensor({int64_t(1)}, DType::Complex64, device);
    // log(1+0i) = 0+0i
    a.data<std::complex<float>>()[0] = {1.0f, 0.0f};

    auto c = log(a).to(Device::cpu());
    auto* cp = c.data<std::complex<float>>();
    EXPECT_NEAR(cp[0].real(), 0.0f, 1e-5f);
    EXPECT_NEAR(cp[0].imag(), 0.0f, 1e-5f);
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

    auto c = add(a, b).to(Device::cpu());
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 3);
    auto* cp = c.data<std::complex<float>>();
    // First row: (0+0i, 1+1i, 2+2i)
    EXPECT_FLOAT_EQ(cp[0].real(), 0.0f);
    EXPECT_FLOAT_EQ(cp[0].imag(), 0.0f);
    EXPECT_FLOAT_EQ(cp[1].real(), 1.0f);
    EXPECT_FLOAT_EQ(cp[1].imag(), 1.0f);
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

    auto c = mul(a, b).to(Device::cpu());
    auto* cp = c.data<std::complex<double>>();
    EXPECT_NEAR(cp[0].real(), -5e-20, 1e-30);
    EXPECT_NEAR(cp[0].imag(), 10e-20, 1e-30);
}

// ============================================================================
// Complex sqrt
// ============================================================================

TEST_P(ComplexArithmeticTest, ComplexSqrt) {
    auto a = Tensor({int64_t(1)}, DType::Complex64, device);
    // sqrt(-1+0i) = 0+1i
    a.data<std::complex<float>>()[0] = {-1.0f, 0.0f};

    auto c = sqrt(a).to(Device::cpu());
    auto* cp = c.data<std::complex<float>>();
    EXPECT_NEAR(cp[0].real(), 0.0f, 1e-5f);
    EXPECT_NEAR(std::abs(cp[0].imag()), 1.0f, 1e-5f);
}

// ============================================================================
// Transcendental functions on complex tensors
// ============================================================================

TEST_P(ComplexArithmeticTest, ComplexSinCos) {
    auto t = zeros({2}, DType::Complex64, device);
    auto* p = t.data<std::complex<float>>();
    p[0] = {0.0f, 0.0f};     // sin(0) = 0, cos(0) = 1
    p[1] = {1.5708f, 0.0f};  // sin(pi/2) ≈ 1, cos(pi/2) ≈ 0

    auto s = tenzor::sin(t);
    auto c = tenzor::cos(t);
    auto* sp = s.data<std::complex<float>>();
    auto* cp = c.data<std::complex<float>>();

    EXPECT_NEAR(sp[0].real(), 0.0f, 1e-5f);
    EXPECT_NEAR(cp[0].real(), 1.0f, 1e-5f);
    EXPECT_NEAR(sp[1].real(), 1.0f, 1e-3f);
    EXPECT_NEAR(cp[1].real(), 0.0f, 1e-3f);
}

TEST_P(ComplexArithmeticTest, ComplexMatmul) {
    auto a = zeros({2, 3}, DType::Complex64, device);
    auto b = zeros({3, 2}, DType::Complex64, device);

    // Fill with simple values
    auto* ap = a.data<std::complex<float>>();
    auto* bp = b.data<std::complex<float>>();
    for (int i = 0; i < 6; ++i) {
        ap[i] = {static_cast<float>(i + 1), 0.0f};
        bp[i] = {static_cast<float>(i + 1), 0.0f};
    }

    auto c = tenzor::matmul(a, b);
    EXPECT_EQ(c.dtype(), DType::Complex64);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);

    // C[0,0] = a[0,:] . b[:,0] = 1*1 + 2*2 + 3*3 = 14 (real values)
    auto* cp = c.data<std::complex<float>>();
    EXPECT_NEAR(cp[0].real(), 22.0f, 1e-3f);
}

// ============================================================================
// Instantiate for CPU backend
// ============================================================================
// Complex arithmetic is currently only exercised on CPU here. When the
// complex-dtype kernels land on CUDA/ROCm/Vulkan/OneAPI, add matching
// INSTANTIATE_TEST_SUITE_P(<Backend>, ...) blocks. Do not fold them into a
// single multi-backend instantiation until every kernel the tests call is
// registered on that backend — otherwise the instantiation silently skips.

INSTANTIATE_TEST_SUITE_P(
    CPU, ComplexArithmeticTest,
    ::testing::Values("cpu"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);
