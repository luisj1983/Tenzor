/**
 * @file test_special_math_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for special mathematical functions
 *
 * Covers OpIds 490-509: Gamma, Lgamma, Digamma, Polygamma, Beta, BetaInc,
 * BesselJ0, BesselJ1, BesselY0, BesselY1, BesselI0, BesselI1,
 * ErfInv, Sinc, Zeta
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Fixture
// ============================================================================

class SpecialMathMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Special math functions need wider tolerance, especially for Float16
    float special_tol() const {
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
            return 0.05f;
        }
        return std::max(atol(), 1e-4f);
    }

    // Helper: create a tensor from a vector of floats
    Tensor fromValues(const std::vector<float>& vals) {
        auto t = tenzor::full({static_cast<int64_t>(vals.size())},
                              0.0f, DType::Float32, Device::cpu());
        auto* d = t.data<float>();
        for (size_t i = 0; i < vals.size(); ++i) d[i] = vals[i];
        return t.to(device()).to(dtype());
    }

    // Helper: get float values from result tensor
    std::vector<float> toFloats(const Tensor& t) {
        auto f32 = t.to(Device::cpu()).to(DType::Float32);
        auto* d = f32.data<float>();
        return std::vector<float>(d, d + f32.numel());
    }
};

// ============================================================================
// Gamma Function Tests
// ============================================================================

TEST_P(SpecialMathMultiDTypeTest, GammaKnownValues) {
    // gamma(1)=1, gamma(2)=1, gamma(3)=2, gamma(5)=24
    auto input = fromValues({1.0f, 2.0f, 3.0f, 5.0f});
    auto result = tenzor::gamma(input);
    auto vals = toFloats(result);

    EXPECT_NEAR(vals[0], 1.0f, special_tol());
    EXPECT_NEAR(vals[1], 1.0f, special_tol());
    EXPECT_NEAR(vals[2], 2.0f, special_tol());
    EXPECT_NEAR(vals[3], 24.0f, special_tol() * 24.0f);  // scale-relative
    expectDevice(result);
    expectDType(result);
}

TEST_P(SpecialMathMultiDTypeTest, GammaHalfInteger) {
    // gamma(0.5) = sqrt(pi) ≈ 1.7724539
    auto input = fromValues({0.5f});
    auto result = tenzor::gamma(input);
    auto vals = toFloats(result);
    EXPECT_NEAR(vals[0], std::sqrt(static_cast<float>(M_PI)), special_tol());
}

// ============================================================================
// Lgamma Tests
// ============================================================================

TEST_P(SpecialMathMultiDTypeTest, LgammaKnownValues) {
    // lgamma(1)=0, lgamma(2)=0, lgamma(5)=ln(24)
    auto input = fromValues({1.0f, 2.0f, 5.0f});
    auto result = tenzor::lgamma(input);
    auto vals = toFloats(result);

    EXPECT_NEAR(vals[0], 0.0f, special_tol());
    EXPECT_NEAR(vals[1], 0.0f, special_tol());
    EXPECT_NEAR(vals[2], std::log(24.0f), special_tol());
}

// ============================================================================
// Digamma Tests
// ============================================================================

TEST_P(SpecialMathMultiDTypeTest, DigammaKnownValues) {
    // digamma(1) = -euler_gamma ≈ -0.5772
    // digamma(2) = 1 - euler_gamma ≈ 0.4228
    constexpr float euler_gamma = 0.5772156649f;
    auto input = fromValues({1.0f, 2.0f});
    auto result = tenzor::digamma(input);
    auto vals = toFloats(result);

    EXPECT_NEAR(vals[0], -euler_gamma, special_tol());
    EXPECT_NEAR(vals[1], 1.0f - euler_gamma, special_tol());
}

// ============================================================================
// Polygamma Tests
// ============================================================================

TEST_P(SpecialMathMultiDTypeTest, PolygammaOrder0IsDigamma) {
    // polygamma(0, x) == digamma(x)
    auto input = fromValues({1.0f, 2.0f, 3.0f});
    auto pg0 = tenzor::polygamma(0, input);
    auto dg = tenzor::digamma(input);

    auto pg0_vals = toFloats(pg0);
    auto dg_vals = toFloats(dg);
    for (size_t i = 0; i < pg0_vals.size(); ++i) {
        EXPECT_NEAR(pg0_vals[i], dg_vals[i], special_tol()) << "Index " << i;
    }
}

TEST_P(SpecialMathMultiDTypeTest, PolygammaOrder1) {
    // polygamma(1, 1) = pi^2/6 ≈ 1.6449
    auto input = fromValues({1.0f});
    auto result = tenzor::polygamma(1, input);
    auto vals = toFloats(result);
    float expected = static_cast<float>(M_PI * M_PI / 6.0);
    EXPECT_NEAR(vals[0], expected, std::max(special_tol(), 0.02f));
}

// ============================================================================
// Beta Function Tests
// ============================================================================

TEST_P(SpecialMathMultiDTypeTest, BetaKnownValues) {
    // beta(a, b) = gamma(a)*gamma(b)/gamma(a+b)
    // beta(1, 1) = 1, beta(2, 2) = 1/6
    auto a = fromValues({1.0f, 2.0f});
    auto b = fromValues({1.0f, 2.0f});
    auto result = tenzor::beta(a, b);
    auto vals = toFloats(result);

    EXPECT_NEAR(vals[0], 1.0f, special_tol());
    EXPECT_NEAR(vals[1], 1.0f / 6.0f, special_tol());
}

// ============================================================================
// BetaInc Tests
// ============================================================================

TEST_P(SpecialMathMultiDTypeTest, BetaIncBoundaries) {
    // betainc(a, b, 0) = 0, betainc(a, b, 1) = 1 (regularized)
    auto a = fromValues({2.0f, 2.0f});
    auto b = fromValues({3.0f, 3.0f});
    auto x = fromValues({0.0f, 1.0f});
    auto result = tenzor::betainc(a, b, x);
    auto vals = toFloats(result);

    EXPECT_NEAR(vals[0], 0.0f, special_tol());
    EXPECT_NEAR(vals[1], 1.0f, special_tol());
}

// ============================================================================
// Bessel Function Tests
// ============================================================================

TEST_P(SpecialMathMultiDTypeTest, BesselJ0AtZero) {
    // J0(0) = 1
    auto input = fromValues({0.0f});
    auto result = tenzor::bessel_j0(input);
    auto vals = toFloats(result);
    EXPECT_NEAR(vals[0], 1.0f, special_tol());
}

TEST_P(SpecialMathMultiDTypeTest, BesselJ1AtZero) {
    // J1(0) = 0
    auto input = fromValues({0.0f});
    auto result = tenzor::bessel_j1(input);
    auto vals = toFloats(result);
    EXPECT_NEAR(vals[0], 0.0f, special_tol());
}

TEST_P(SpecialMathMultiDTypeTest, BesselI0AtZero) {
    // I0(0) = 1
    auto input = fromValues({0.0f});
    auto result = tenzor::bessel_i0(input);
    auto vals = toFloats(result);
    EXPECT_NEAR(vals[0], 1.0f, special_tol());
}

TEST_P(SpecialMathMultiDTypeTest, BesselI1AtZero) {
    // I1(0) = 0
    auto input = fromValues({0.0f});
    auto result = tenzor::bessel_i1(input);
    auto vals = toFloats(result);
    EXPECT_NEAR(vals[0], 0.0f, special_tol());
}

TEST_P(SpecialMathMultiDTypeTest, BesselY0PositiveArgs) {
    // Y0 is defined for x > 0; test at known values
    auto input = fromValues({1.0f, 2.0f});
    auto result = tenzor::bessel_y0(input);
    auto vals = toFloats(result);
    // Y0(1) ≈ 0.0883, Y0(2) ≈ 0.5104
    EXPECT_NEAR(vals[0], 0.0882569642f, special_tol());
    EXPECT_NEAR(vals[1], 0.5103756726f, special_tol());
}

TEST_P(SpecialMathMultiDTypeTest, BesselY1PositiveArgs) {
    // Y1 defined for x > 0
    auto input = fromValues({1.0f, 2.0f});
    auto result = tenzor::bessel_y1(input);
    auto vals = toFloats(result);
    // Y1(1) ≈ -0.7812, Y1(2) ≈ -0.1070
    EXPECT_NEAR(vals[0], -0.7812128213f, special_tol());
    EXPECT_NEAR(vals[1], -0.1070324315f, special_tol());
}

// ============================================================================
// ErfInv Tests
// ============================================================================

TEST_P(SpecialMathMultiDTypeTest, ErfInvAtZero) {
    // erfinv(0) = 0
    auto input = fromValues({0.0f});
    auto result = tenzor::erfinv(input);
    auto vals = toFloats(result);
    EXPECT_NEAR(vals[0], 0.0f, special_tol());
}

TEST_P(SpecialMathMultiDTypeTest, ErfInvRoundTrip) {
    // erfinv(erf(x)) ≈ x
    auto x = fromValues({0.1f, 0.5f, -0.3f});
    auto erf_x = tenzor::erf(x);
    auto roundtrip = tenzor::erfinv(erf_x);
    auto vals_x = toFloats(x);
    auto vals_rt = toFloats(roundtrip);

    for (size_t i = 0; i < vals_x.size(); ++i) {
        EXPECT_NEAR(vals_rt[i], vals_x[i], special_tol()) << "Index " << i;
    }
}

// ============================================================================
// Sinc Tests
// ============================================================================

TEST_P(SpecialMathMultiDTypeTest, SincKnownValues) {
    // sinc(0) = 1, sinc(n) = 0 for non-zero integer n (normalized sinc)
    auto input = fromValues({0.0f, 1.0f, -1.0f, 2.0f});
    auto result = tenzor::sinc(input);
    auto vals = toFloats(result);

    EXPECT_NEAR(vals[0], 1.0f, special_tol());
    EXPECT_NEAR(vals[1], 0.0f, special_tol());
    EXPECT_NEAR(vals[2], 0.0f, special_tol());
    EXPECT_NEAR(vals[3], 0.0f, special_tol());
}

// ============================================================================
// Zeta Tests
// ============================================================================

TEST_P(SpecialMathMultiDTypeTest, ZetaRiemannKnownValues) {
    // zeta(2, 1) = pi^2/6 ≈ 1.6449 (Riemann zeta at s=2, q=1)
    auto x = fromValues({2.0f});
    auto q = fromValues({1.0f});
    auto result = tenzor::zeta(x, q);
    auto vals = toFloats(result);
    float expected = static_cast<float>(M_PI * M_PI / 6.0);
    EXPECT_NEAR(vals[0], expected, special_tol());
}

// ============================================================================
// Single Element and Empty Tensor Edge Cases
// ============================================================================

TEST_P(SpecialMathMultiDTypeTest, SingleElementGamma) {
    auto input = fromValues({3.0f});
    auto result = tenzor::gamma(input);
    expectShape(result, {1});
    auto vals = toFloats(result);
    EXPECT_NEAR(vals[0], 2.0f, special_tol());
}

TEST_P(SpecialMathMultiDTypeTest, DTypePreservation) {
    auto input = fromValues({1.0f, 2.0f});
    auto positive_input = fromValues({0.5f, 1.5f});

    expectDType(tenzor::gamma(positive_input));
    expectDType(tenzor::lgamma(positive_input));
    expectDType(tenzor::digamma(positive_input));
    expectDType(tenzor::erf(input));
    expectDType(tenzor::erfinv(fromValues({0.1f, 0.5f})));
    expectDType(tenzor::sinc(input));
    expectDType(tenzor::bessel_j0(input));
    expectDType(tenzor::bessel_i0(input));
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SpecialMathMultiDTypeTest);
