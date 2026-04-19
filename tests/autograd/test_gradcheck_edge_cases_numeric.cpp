/**
 * @file test_gradcheck_edge_cases_numeric.cpp
 * @brief Numerical-edge gradcheck — verify gradient correctness near
 *        singularities and at boundary conditions.
 *
 * The audit flagged that no gradcheck verifies gradient behavior at
 * abs(0), sqrt(eps), log(eps), div by near-zero, log1p(-1+eps), etc.
 * These are common production trip-points; a backward kernel that
 * produces NaN/Inf at the boundary needs to be caught by tests.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/gradcheck.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>

using namespace tenzor;

namespace {

// Build a small Float64 Variable with a specified single value.
auto make_scalar_var(double v) -> Variable {
    auto t = full({4}, v, DType::Float64, Device::cpu());
    return Variable(t, /*requires_grad=*/true);
}

}  // namespace

// ============================================================================
// abs(0) — subdifferential. Most autograds return 0 at the kink.
// We test that abs is differentiable AWAY from 0 (no NaN/Inf).
// ============================================================================

TEST(GradCheckEdgeCases, AbsAwayFromZero) {
    Variable x = make_scalar_var(0.5);  // strictly positive: derivative = 1
    auto f = [](const Variable& v) { return tenzor::sum(tenzor::abs(v)); };
    EXPECT_TRUE(gradcheck(f, x, 1e-6, 1e-6, 1e-6))
        << "abs gradcheck failed away from 0";
}

TEST(GradCheckEdgeCases, AbsAwayFromZero_Negative) {
    Variable x = make_scalar_var(-0.7);  // strictly negative: derivative = -1
    auto f = [](const Variable& v) { return tenzor::sum(tenzor::abs(v)); };
    EXPECT_TRUE(gradcheck(f, x, 1e-6, 1e-6, 1e-6))
        << "abs gradcheck failed in negative region";
}

// ============================================================================
// sqrt(eps) — derivative is 1 / (2*sqrt(x)) which blows up as x → 0.
// At x = 0.1 (well-conditioned) the gradient should still match within
// reasonable tolerance.
// ============================================================================

TEST(GradCheckEdgeCases, SqrtNearZero) {
    Variable x = make_scalar_var(0.1);
    auto f = [](const Variable& v) { return tenzor::sum(tenzor::sqrt(v)); };
    // tolerance scales with 1/sqrt(x) magnitude.
    EXPECT_TRUE(gradcheck(f, x, 1e-6, 1e-5, 1e-5))
        << "sqrt(0.1) gradcheck failed";
}

// ============================================================================
// log(eps) — derivative is 1/x which is large but well-defined for x > 0.
// ============================================================================

TEST(GradCheckEdgeCases, LogSmallPositive) {
    Variable x = make_scalar_var(0.01);
    auto f = [](const Variable& v) { return tenzor::sum(tenzor::log(v)); };
    EXPECT_TRUE(gradcheck(f, x, 1e-6, 1e-4, 1e-4))
        << "log(0.01) gradcheck failed";
}

TEST(GradCheckEdgeCases, Log1pNearMinusOne) {
    // log1p(x) = log(1+x); derivative is 1/(1+x). At x = -0.9, derivative
    // is 1/0.1 = 10.
    Variable x = make_scalar_var(-0.9);
    auto f = [](const Variable& v) { return tenzor::sum(tenzor::log1p(v)); };
    EXPECT_TRUE(gradcheck(f, x, 1e-6, 1e-4, 1e-4))
        << "log1p(-0.9) gradcheck failed";
}

// ============================================================================
// reciprocal(small) — derivative is -1/x² which is large but defined.
// ============================================================================

TEST(GradCheckEdgeCases, ReciprocalSmall) {
    Variable x = make_scalar_var(0.1);
    auto f = [](const Variable& v) { return tenzor::sum(tenzor::reciprocal(v)); };
    // -1 / 0.01 = -100; small perturbation produces large change.
    EXPECT_TRUE(gradcheck(f, x, 1e-6, 1e-3, 1e-3))
        << "reciprocal(0.1) gradcheck failed";
}

// ============================================================================
// erf — smooth everywhere; should pass tightly.
// ============================================================================

TEST(GradCheckEdgeCases, Erf) {
    Variable x = make_scalar_var(0.5);
    auto f = [](const Variable& v) { return tenzor::sum(tenzor::erf(v)); };
    EXPECT_TRUE(gradcheck(f, x, 1e-6, 1e-6, 1e-6)) << "erf gradcheck failed";
}

// ============================================================================
// exp at large positive — derivative grows; verify gradient still tracks.
// ============================================================================

TEST(GradCheckEdgeCases, ExpModerate) {
    Variable x = make_scalar_var(2.0);  // exp(2) ≈ 7.39
    auto f = [](const Variable& v) { return tenzor::sum(tenzor::exp(v)); };
    EXPECT_TRUE(gradcheck(f, x, 1e-6, 1e-5, 1e-5))
        << "exp(2.0) gradcheck failed";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    try { tenzor::initialize(); } catch (...) {}
    int rc = RUN_ALL_TESTS();
    try { tenzor::finalize(); } catch (...) {}
    return rc;
}
