#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include "tenzor/tenzor.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"

namespace tenzor { void initialize(); }

namespace tz = ::tenzor;

// Audit Phase 2 Tasks 2.3+2.4+2.5+2.10:
// matmul/bmm/dot dtype coverage, pow/clamp F64 precision.

class MathDtypeCoverageEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static ::testing::Environment* const g_env =
    ::testing::AddGlobalTestEnvironment(new MathDtypeCoverageEnv);

// ============================================================================
// Task 2.3: matmul integer dtypes
// ============================================================================

TEST(MathDtypeCoverage, MatmulInt16) {
    // 4x4 @ 4x4 of value 2 * value 3 → each output = 4*2*3 = 24
    auto a = tz::full({4, 4}, 2.0, tz::DType::Int16);
    auto b = tz::full({4, 4}, 3.0, tz::DType::Int16);
    auto c = tz::matmul(a, b);
    EXPECT_EQ(c.dtype(), tz::DType::Int16);
    auto ci32 = c.cpu().to(tz::DType::Int32);
    const int32_t* p = ci32.data<int32_t>();
    for (int i = 0; i < 16; ++i)
        EXPECT_EQ(p[i], 24) << "i=" << i;
}

TEST(MathDtypeCoverage, MatmulInt64) {
    // Large values to ensure Int64 accumulator is used
    auto a = tz::full({4, 4}, 1000.0, tz::DType::Int64);
    auto b = tz::full({4, 4}, 1000.0, tz::DType::Int64);
    auto c = tz::matmul(a, b);
    EXPECT_EQ(c.dtype(), tz::DType::Int64);
    const int64_t* p = c.cpu().data<int64_t>();
    for (int i = 0; i < 16; ++i)
        EXPECT_EQ(p[i], 4LL * 1000 * 1000) << "i=" << i;
}

TEST(MathDtypeCoverage, MatmulUInt16) {
    auto a = tz::full({4, 4}, 3.0, tz::DType::UInt16);
    auto b = tz::full({4, 4}, 5.0, tz::DType::UInt16);
    auto c = tz::matmul(a, b);
    EXPECT_EQ(c.dtype(), tz::DType::UInt16);
    auto ci32 = c.cpu().to(tz::DType::Int32);
    const int32_t* p = ci32.data<int32_t>();
    for (int i = 0; i < 16; ++i)
        EXPECT_EQ(p[i], 60) << "i=" << i; // 4*3*5=60
}

TEST(MathDtypeCoverage, MatmulUInt32) {
    auto a = tz::full({4, 4}, 7.0, tz::DType::UInt32);
    auto b = tz::full({4, 4}, 8.0, tz::DType::UInt32);
    auto c = tz::matmul(a, b);
    EXPECT_EQ(c.dtype(), tz::DType::UInt32);
    auto ci64 = c.cpu().to(tz::DType::Int64);
    const int64_t* p = ci64.data<int64_t>();
    for (int i = 0; i < 16; ++i)
        EXPECT_EQ(p[i], 224LL) << "i=" << i; // 4*7*8=224
}

TEST(MathDtypeCoverage, MatmulUInt64) {
    auto a = tz::full({4, 4}, 2.0, tz::DType::UInt64);
    auto b = tz::full({4, 4}, 2.0, tz::DType::UInt64);
    auto c = tz::matmul(a, b);
    EXPECT_EQ(c.dtype(), tz::DType::UInt64);
    const uint64_t* p = c.cpu().data<uint64_t>();
    for (int i = 0; i < 16; ++i)
        EXPECT_EQ(p[i], 16ULL) << "i=" << i; // 4*2*2=16
}

// ============================================================================
// Task 2.4: bmm dtype parity with matmul
// ============================================================================

TEST(MathDtypeCoverage, BmmFloat16) {
    // (2, 3, 4) @ (2, 4, 3) → (2, 3, 3), each element = 4*2*3=24
    auto a = tz::full({2, 3, 4}, 2.0, tz::DType::Float16);
    auto b = tz::full({2, 4, 3}, 3.0, tz::DType::Float16);
    auto c = tz::bmm(a, b);
    ASSERT_EQ(c.shape()[0], 2);
    ASSERT_EQ(c.shape()[1], 3);
    ASSERT_EQ(c.shape()[2], 3);
    auto cf64 = c.cpu().to(tz::DType::Float64);
    const double* p = cf64.data<double>();
    for (int i = 0; i < c.numel(); ++i)
        EXPECT_NEAR(p[i], 24.0, 0.1) << "i=" << i;
}

TEST(MathDtypeCoverage, BmmBFloat16) {
    auto a = tz::full({2, 3, 4}, 2.0, tz::DType::BFloat16);
    auto b = tz::full({2, 4, 3}, 3.0, tz::DType::BFloat16);
    auto c = tz::bmm(a, b);
    ASSERT_EQ(c.shape()[0], 2);
    ASSERT_EQ(c.shape()[1], 3);
    ASSERT_EQ(c.shape()[2], 3);
    auto cf64 = c.cpu().to(tz::DType::Float64);
    const double* p = cf64.data<double>();
    for (int i = 0; i < c.numel(); ++i)
        EXPECT_NEAR(p[i], 24.0, 0.2) << "i=" << i;
}

TEST(MathDtypeCoverage, BmmComplex64) {
    // a[*,i,k] = (1+0i), b[*,k,j] = (0+1i) → c[*,i,j] = sum_k (1+0i)*(0+1i) = K*i = 2i
    tz::Tensor a({2, 2, 2}, tz::DType::Complex64, tz::Device::cpu());
    tz::Tensor b({2, 2, 2}, tz::DType::Complex64, tz::Device::cpu());
    {
        auto* pa = a.data<std::complex<float>>();
        auto* pb = b.data<std::complex<float>>();
        for (int i = 0; i < 8; ++i) {
            pa[i] = {1.0f, 0.0f};
            pb[i] = {0.0f, 1.0f};
        }
    }
    auto c = tz::bmm(a, b);
    ASSERT_EQ(c.shape()[0], 2);
    ASSERT_EQ(c.shape()[1], 2);
    ASSERT_EQ(c.shape()[2], 2);
    const auto* pc = c.data<std::complex<float>>();
    for (int i = 0; i < c.numel(); ++i) {
        EXPECT_NEAR(pc[i].real(), 0.0f, 1e-5f) << "re i=" << i;
        EXPECT_NEAR(pc[i].imag(), 2.0f, 1e-5f) << "im i=" << i;
    }
}

TEST(MathDtypeCoverage, BmmInt16) {
    // (2, 2, 2) @ (2, 2, 2), values 2 and 3 → each = 2*(2*3) = 12
    auto a = tz::full({2, 2, 2}, 2.0, tz::DType::Int16);
    auto b = tz::full({2, 2, 2}, 3.0, tz::DType::Int16);
    auto c = tz::bmm(a, b);
    ASSERT_EQ(c.shape()[0], 2);
    ASSERT_EQ(c.shape()[1], 2);
    ASSERT_EQ(c.shape()[2], 2);
    auto ci32 = c.cpu().to(tz::DType::Int32);
    const int32_t* p = ci32.data<int32_t>();
    for (int i = 0; i < c.numel(); ++i)
        EXPECT_EQ(p[i], 12) << "i=" << i;
}

TEST(MathDtypeCoverage, BmmUInt16) {
    auto a = tz::full({2, 2, 2}, 4.0, tz::DType::UInt16);
    auto b = tz::full({2, 2, 2}, 5.0, tz::DType::UInt16);
    auto c = tz::bmm(a, b);
    ASSERT_EQ(c.shape()[0], 2);
    ASSERT_EQ(c.shape()[1], 2);
    ASSERT_EQ(c.shape()[2], 2);
    auto ci32 = c.cpu().to(tz::DType::Int32);
    const int32_t* p = ci32.data<int32_t>();
    for (int i = 0; i < c.numel(); ++i)
        EXPECT_EQ(p[i], 40) << "i=" << i; // 2*4*5=40
}

// ============================================================================
// Task 2.5: dot kernel half/complex coverage
// ============================================================================

TEST(MathDtypeCoverage, DotFloat16) {
    // 8 elements: 2.0 * 3.0 summed = 48.0
    auto a = tz::full({8}, 2.0, tz::DType::Float16);
    auto b = tz::full({8}, 3.0, tz::DType::Float16);
    auto c = tz::dot(a, b);
    auto cf64 = c.to(tz::DType::Float64);
    EXPECT_NEAR(cf64.data<double>()[0], 48.0, 0.1);
}

TEST(MathDtypeCoverage, DotBFloat16) {
    auto a = tz::full({8}, 2.0, tz::DType::BFloat16);
    auto b = tz::full({8}, 3.0, tz::DType::BFloat16);
    auto c = tz::dot(a, b);
    auto cf64 = c.to(tz::DType::Float64);
    EXPECT_NEAR(cf64.data<double>()[0], 48.0, 0.5);
}

TEST(MathDtypeCoverage, DotComplex64) {
    // (1+2i) * (3+4i) = 3+4i+6i+8i² = (3-8)+(4+6)i = -5+10i
    // summed 4 times: -20 + 40i
    tz::Tensor a({4}, tz::DType::Complex64, tz::Device::cpu());
    tz::Tensor b({4}, tz::DType::Complex64, tz::Device::cpu());
    {
        auto* pa = a.data<std::complex<float>>();
        auto* pb = b.data<std::complex<float>>();
        for (int i = 0; i < 4; ++i) {
            pa[i] = {1.0f, 2.0f};
            pb[i] = {3.0f, 4.0f};
        }
    }
    auto c = tz::dot(a, b);
    const auto* cv = c.data<std::complex<float>>();
    EXPECT_NEAR(cv[0].real(), -20.0f, 1e-4f);
    EXPECT_NEAR(cv[0].imag(),  40.0f, 1e-4f);
}

TEST(MathDtypeCoverage, DotComplex128) {
    // (1+2i) * (3+4i) = -5+10i, summed 4 times: -20+40i
    tz::Tensor a({4}, tz::DType::Complex128, tz::Device::cpu());
    tz::Tensor b({4}, tz::DType::Complex128, tz::Device::cpu());
    {
        auto* pa = a.data<std::complex<double>>();
        auto* pb = b.data<std::complex<double>>();
        for (int i = 0; i < 4; ++i) {
            pa[i] = {1.0, 2.0};
            pb[i] = {3.0, 4.0};
        }
    }
    auto c = tz::dot(a, b);
    const auto* cv = c.data<std::complex<double>>();
    EXPECT_NEAR(cv[0].real(), -20.0, 1e-10);
    EXPECT_NEAR(cv[0].imag(),  40.0, 1e-10);
}

TEST(MathDtypeCoverage, DotInt16) {
    auto a = tz::full({4}, 3.0, tz::DType::Int16);
    auto b = tz::full({4}, 5.0, tz::DType::Int16);
    auto c = tz::dot(a, b);
    // 4 * 3 * 5 = 60
    auto ci32 = c.to(tz::DType::Int32);
    EXPECT_EQ(ci32.data<int32_t>()[0], 60);
}

TEST(MathDtypeCoverage, DotUInt16) {
    auto a = tz::full({4}, 7.0, tz::DType::UInt16);
    auto b = tz::full({4}, 8.0, tz::DType::UInt16);
    auto c = tz::dot(a, b);
    // 4 * 7 * 8 = 224
    auto ci32 = c.to(tz::DType::Int32);
    EXPECT_EQ(ci32.data<int32_t>()[0], 224);
}

TEST(MathDtypeCoverage, DotUInt32) {
    auto a = tz::full({4}, 100.0, tz::DType::UInt32);
    auto b = tz::full({4}, 200.0, tz::DType::UInt32);
    auto c = tz::dot(a, b);
    // 4 * 100 * 200 = 80000
    auto ci64 = c.to(tz::DType::Int64);
    EXPECT_EQ(ci64.data<int64_t>()[0], 80000LL);
}

TEST(MathDtypeCoverage, DotUInt64) {
    auto a = tz::full({4}, 10.0, tz::DType::UInt64);
    auto b = tz::full({4}, 10.0, tz::DType::UInt64);
    auto c = tz::dot(a, b);
    // 4 * 10 * 10 = 400
    EXPECT_EQ(c.data<uint64_t>()[0], 400ULL);
}

// ============================================================================
// Task 2.10: pow / clamp F64 precision
// ============================================================================

TEST(MathDtypeCoverage, PowFloat64FullPrecision) {
    // Pre-fix: exponent is truncated to float (~7 decimal digits).
    // Post-fix: full double precision (~15 decimal digits).
    double exp_val = 1.234567890123456;
    auto x = tz::full({4}, 2.0, tz::DType::Float64);
    auto y = tz::pow(x, exp_val);
    double expected = std::pow(2.0, exp_val);
    const double* p = y.cpu().data<double>();
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(p[i], expected, 1e-13)
            << "pow lost double precision in exponent (audit P1 2.10)";
}

TEST(MathDtypeCoverage, ClampFloat64FullPrecision) {
    double lo = 1.2345678901234567;
    double hi = 9.8765432109876543;
    // Values in range — should pass through unchanged
    auto x = tz::full({4}, 5.0, tz::DType::Float64);
    auto y = tz::clamp(x, lo, hi);
    const double* p = y.cpu().data<double>();
    for (int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(p[i], 5.0);

    // Values below lo — clamped to lo; test precision of lo is preserved
    auto x_low = tz::full({4}, 0.0, tz::DType::Float64);
    auto y_low = tz::clamp(x_low, lo, hi);
    const double* p_low = y_low.cpu().data<double>();
    for (int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(p_low[i], lo)
            << "clamp_min lost double precision (audit P1 2.10)";
}

TEST(MathDtypeCoverage, ClampMinFloat64Precision) {
    double lo = 3.141592653589793;
    auto x = tz::full({4}, 0.0, tz::DType::Float64);
    auto y = tz::clamp_min(x, lo);
    const double* p = y.cpu().data<double>();
    for (int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(p[i], lo)
            << "clamp_min lost double precision (audit P1 2.10)";
}

TEST(MathDtypeCoverage, ClampMaxFloat64Precision) {
    double hi = 2.718281828459045;
    auto x = tz::full({4}, 100.0, tz::DType::Float64);
    auto y = tz::clamp_max(x, hi);
    const double* p = y.cpu().data<double>();
    for (int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(p[i], hi)
            << "clamp_max lost double precision (audit P1 2.10)";
}
