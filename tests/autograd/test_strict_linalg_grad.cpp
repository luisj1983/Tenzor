/**
 * @file test_strict_linalg_grad.cpp
 * @brief Verifies the TENZOR_STRICT_LINALG_GRAD env var contract for
 *        linalg backward functions that return zero gradients by default.
 *
 * Without strict mode, `ldl_factor` and `householder_product` silently
 * produce zero input gradients — users see training "work" while gradients
 * through the factorization are missing. Strict mode surfaces those gaps
 * as runtime errors so CI catches accidental use in a gradient chain.
 */

#include <cstdlib>

#include <gtest/gtest.h>

#include <tenzor/autograd/ops.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/tenzor.hpp>
#include <tenzor/utils/error.hpp>

#include "../backend_test_fixture.hpp"
#include "../grad_flow_helpers.hpp"

namespace tenzor {
namespace {

class StrictLinalgGradTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
        const char* prev = std::getenv("TENZOR_STRICT_LINALG_GRAD");
        saved_strict_ = prev ? std::string(prev) : std::string();
    }
    void TearDown() override {
        if (saved_strict_.empty()) {
            unsetenv("TENZOR_STRICT_LINALG_GRAD");
        } else {
            setenv("TENZOR_STRICT_LINALG_GRAD", saved_strict_.c_str(), 1);
        }
    }
    std::string saved_strict_;

    // Build a STRONGLY diagonally-dominant SPD matrix. The closed-form LDL
    // backward is only valid for the no-pivoting case; diagonal dominance
    // guarantees Bunch-Kaufman (*sytrf) selects 1x1 pivots with no interchange
    // (trivial ipiv), so these gradient tests exercise the supported path
    // deterministically rather than relying on a random draw not pivoting.
    Tensor make_spd(int64_t n) {
        auto x = randn({n, n}, DType::Float64, device);
        auto eye_n = eye(n, std::nullopt, DType::Float64, device);
        return matmul(x, transpose(x, 0, 1)) + eye_n * 100.0;
    }

    // A symmetric INDEFINITE matrix that forces Bunch-Kaufman pivoting:
    // [[0, 1], [1, 0]] has eigenvalues +/-1 and a zero diagonal, so *sytrf must
    // interchange / use a 2x2 block (non-trivial ipiv).
    Tensor make_indefinite_pivoting() {
        auto A = zeros({2, 2}, DType::Float64, Device::cpu());
        double* d = A.data<double>();
        d[1] = 1.0;  // (0,1)
        d[2] = 1.0;  // (1,0)
        return A.to(device);
    }
};

// audit-2026-05-03 Phase 12: LDL backward now uses the closed-form
// structural-symmetric backprop derived from A = L D L^T (see
// `LinalgLDLFactorBackward::backward` in `function_new_ops.cpp`). The old
// "zero-gradient stub gated by TENZOR_STRICT_LINALG_GRAD" contract is no
// longer applicable — these tests now verify the meaningful gradient is
// produced and that the strict env var is a no-op for the LDL path
// (it stays in source as a structural marker for *other* linalg ops that
// still use the zero-grad stub: `householder_product`).

TEST_P(StrictLinalgGradTest, LDLFactor_ProducesMeaningfulGradient) {
    unsetenv("TENZOR_STRICT_LINALG_GRAD");

    Variable A(make_spd(4), /*requires_grad=*/true);
    auto [LD, pivots] = ldl_factor(A);
    auto loss = tenzor::sum(LD);
    EXPECT_NO_THROW(loss.backward());

    ASSERT_TRUE(A.has_grad());
    auto g = A.grad().value().cpu().to(DType::Float64).contiguous();
    const double* gp = g.data<double>();
    bool any_nonzero = false;
    for (int64_t i = 0; i < g.numel(); ++i) {
        if (gp[i] != 0.0) { any_nonzero = true; break; }
    }
    EXPECT_TRUE(any_nonzero)
        << "LDL backward should now produce a real (non-zero) gradient";
}

TEST_P(StrictLinalgGradTest, LDLFactor_StrictModeIsNoopForLDL) {
    setenv("TENZOR_STRICT_LINALG_GRAD", "1", /*overwrite=*/1);

    Variable A(make_spd(4), /*requires_grad=*/true);
    auto [LD, pivots] = ldl_factor(A);
    auto loss = tenzor::sum(LD);
    // LDL has a real backward; strict mode no longer throws for it.
    EXPECT_NO_THROW(loss.backward());
    EXPECT_GRAD_FLOWS(A);
}

TEST_P(StrictLinalgGradTest, LDLFactor_StrictModeFalseEquivalentToUnset) {
    setenv("TENZOR_STRICT_LINALG_GRAD", "0", /*overwrite=*/1);

    Variable A(make_spd(4), /*requires_grad=*/true);
    auto [LD, pivots] = ldl_factor(A);
    auto loss = tenzor::sum(LD);
    EXPECT_NO_THROW(loss.backward());

    ASSERT_TRUE(A.has_grad());
    auto g = A.grad().value().cpu().to(DType::Float64).contiguous();
    const double* gp = g.data<double>();
    bool any_nonzero = false;
    for (int64_t i = 0; i < g.numel(); ++i) {
        if (gp[i] != 0.0) { any_nonzero = true; break; }
    }
    EXPECT_TRUE(any_nonzero)
        << "TENZOR_STRICT_LINALG_GRAD=0 should produce the same real "
           "gradient as the unset default";
}

// The closed-form adjoint is invalid once Bunch-Kaufman pivots, so backward
// through an indefinite (pivoting) factorization must fail loud rather than
// return a silently-wrong gradient. (Mirrors PyTorch, which has no pivoted
// ldl_factor backward.)
TEST_P(StrictLinalgGradTest, LDLFactor_PivotingThrowsNonDifferentiable) {
    unsetenv("TENZOR_STRICT_LINALG_GRAD");

    Variable A(make_indefinite_pivoting(), /*requires_grad=*/true);
    auto [LD, pivots] = ldl_factor(A);
    auto loss = tenzor::sum(LD);
    EXPECT_THROW(loss.backward(), tenzor::NonDifferentiable);
}

INSTANTIATE_BACKEND_TESTS(StrictLinalgGradTest);

}  // namespace
}  // namespace tenzor
