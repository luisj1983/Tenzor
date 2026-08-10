/**
 * @file test_strict_linalg_grad_multidtype.cpp
 * @brief Multi-dtype / multi-backend companion to test_strict_linalg_grad.cpp.
 *
 * The plain file (BackendTest, Float64-only) verifies the LDL-factor backward
 * produces a meaningful (non-zero) gradient, that the TENZOR_STRICT_LINALG_GRAD
 * env var is a no-op for the LDL path, and that a pivoting (indefinite)
 * factorization throws NonDifferentiable rather than returning a silently-wrong
 * gradient. It runs in Float64 because the closed-form LDL adjoint is a
 * linalg-precision path.
 *
 * This companion re-runs the same surface across {Float32, Float64, Float16} x
 * 5 backends via MultiBackendDTypeTest. Float16 is skipped categorically
 * (DtypeUnsupportedOnBackend): no backend registers a Float16 linalg
 * factorization kernel, so ldl_factor would throw before the backward is ever
 * reached. Float32 and Float64 exercise the real closed-form adjoint in
 * non-Float64 precision and on GPU backends, catching a backend that only
 * registered the Float64 backward.
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

#include "../multi_backend_dtype_fixture.hpp"
#include "../grad_flow_helpers.hpp"

namespace tenzor {
namespace {

// No backend registers a Float16 linalg factorization kernel, so ldl_factor
// would throw before backward is reached. Skip the whole TEST_P body for F16
// (macro so GTEST_SKIP's `return` exits the test body, not a helper).
#define skip_if_half_linalg()                                            \
    do {                                                                 \
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {   \
            SKIP_WITH_REASON(                                             \
                ::tenzor::testing::SkipReason::DtypeUnsupportedOnBackend, \
                "ldl_factor has no Float16/BFloat16 linalg kernel");      \
        }                                                                \
    } while (0)

class StrictLinalgGradMultiDType : public ::tenzor::testing::MultiBackendDTypeTest {
protected:
    void SetUp() override {
        ::tenzor::testing::MultiBackendDTypeTest::SetUp();
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

    // STRONGLY diagonally-dominant SPD matrix in the test dtype. Diagonal
    // dominance guarantees Bunch-Kaufman (*sytrf) selects 1x1 pivots with no
    // interchange (trivial ipiv), so the closed-form adjoint is valid. Built in
    // Float32 first (randn/eye) then narrowed to dtype() so Float64 stays
    // full-precision and Float16 — if reached — at least has representable
    // entries (the 100.0 dominance scale fits in Float16).
    Tensor make_spd(int64_t n) {
        auto x = randn({n, n}, DType::Float32, device());
        auto eye_n = eye(n, std::nullopt, DType::Float32, device());
        auto spd = matmul(x, transpose(x, 0, 1)) + eye_n * 100.0;
        return spd.to(dtype());
    }

    // [[0, 1], [1, 0]] — indefinite, forces a 2x2 Bunch-Kaufman pivot.
    Tensor make_indefinite_pivoting() {
        auto A = zeros({2, 2}, DType::Float32, Device::cpu());
        float* d = A.data<float>();
        d[1] = 1.0f;  // (0,1)
        d[2] = 1.0f;  // (1,0)
        return A.to(device()).to(dtype());
    }
};

TEST_P(StrictLinalgGradMultiDType, LDLFactor_ProducesMeaningfulGradient) {
    skip_if_half_linalg();
    unsetenv("TENZOR_STRICT_LINALG_GRAD");

    Variable A(make_spd(4), /*requires_grad=*/true);
    auto [LD, pivots] = ldl_factor(A);
    auto loss = tenzor::sum(LD);
    EXPECT_NO_THROW(loss.backward());

    ASSERT_TRUE(A.has_grad());
    auto g = A.grad().value().to(Device::cpu()).to(DType::Float64).contiguous();
    const double* gp = g.data<double>();
    bool any_nonzero = false;
    for (int64_t i = 0; i < g.numel(); ++i) {
        if (gp[i] != 0.0) { any_nonzero = true; break; }
    }
    EXPECT_TRUE(any_nonzero)
        << "LDL backward should produce a real (non-zero) gradient";
}

TEST_P(StrictLinalgGradMultiDType, LDLFactor_StrictModeIsNoopForLDL) {
    skip_if_half_linalg();
    setenv("TENZOR_STRICT_LINALG_GRAD", "1", /*overwrite=*/1);

    Variable A(make_spd(4), /*requires_grad=*/true);
    auto [LD, pivots] = ldl_factor(A);
    auto loss = tenzor::sum(LD);
    EXPECT_NO_THROW(loss.backward());
    EXPECT_GRAD_FLOWS(A);
}

TEST_P(StrictLinalgGradMultiDType, LDLFactor_StrictModeFalseEquivalentToUnset) {
    skip_if_half_linalg();
    setenv("TENZOR_STRICT_LINALG_GRAD", "0", /*overwrite=*/1);

    Variable A(make_spd(4), /*requires_grad=*/true);
    auto [LD, pivots] = ldl_factor(A);
    auto loss = tenzor::sum(LD);
    EXPECT_NO_THROW(loss.backward());

    ASSERT_TRUE(A.has_grad());
    auto g = A.grad().value().to(Device::cpu()).to(DType::Float64).contiguous();
    const double* gp = g.data<double>();
    bool any_nonzero = false;
    for (int64_t i = 0; i < g.numel(); ++i) {
        if (gp[i] != 0.0) { any_nonzero = true; break; }
    }
    EXPECT_TRUE(any_nonzero)
        << "TENZOR_STRICT_LINALG_GRAD=0 should produce the same real "
           "gradient as the unset default";
}

TEST_P(StrictLinalgGradMultiDType, LDLFactor_PivotingThrowsNonDifferentiable) {
    skip_if_half_linalg();
    unsetenv("TENZOR_STRICT_LINALG_GRAD");

    Variable A(make_indefinite_pivoting(), /*requires_grad=*/true);
    auto [LD, pivots] = ldl_factor(A);
    auto loss = tenzor::sum(LD);
    EXPECT_THROW(loss.backward(), tenzor::NonDifferentiable);
}

// The instantiation macro expands to unqualified BackendDTypeParam /
// BackendDTypeParamName (tenzor::testing::); bring them into scope since this
// file's TEST_Ps live in an anonymous namespace.
using ::tenzor::testing::BackendDTypeParam;
using ::tenzor::testing::BackendDTypeParamName;
INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(StrictLinalgGradMultiDType);

}  // namespace
}  // namespace tenzor