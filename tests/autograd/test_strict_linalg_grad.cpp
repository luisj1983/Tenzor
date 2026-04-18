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

namespace tenzor {
namespace {

class StrictLinalgGradTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
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

    // Build an SPD-ish matrix so ldl_factor succeeds.
    Tensor make_spd(int64_t n) {
        auto x = randn({n, n}, DType::Float64, Device::cpu());
        auto eye_n = eye(n, std::nullopt, DType::Float64, Device::cpu());
        return matmul(x, transpose(x, 0, 1)) + eye_n;
    }
};

TEST_F(StrictLinalgGradTest, LDLFactor_ZeroGradByDefault) {
    unsetenv("TENZOR_STRICT_LINALG_GRAD");

    Variable A(make_spd(4), /*requires_grad=*/true);
    auto [LD, pivots] = ldl_factor(A);
    auto loss = tenzor::sum(LD);
    EXPECT_NO_THROW(loss.backward());

    ASSERT_TRUE(A.has_grad());
    auto g = A.grad().value().to(DType::Float64).contiguous();
    const double* gp = g.data<double>();
    for (int64_t i = 0; i < g.numel(); ++i) {
        EXPECT_EQ(gp[i], 0.0)
            << "LDL backward default mode should return zero gradient";
    }
}

TEST_F(StrictLinalgGradTest, LDLFactor_StrictModeThrows) {
    setenv("TENZOR_STRICT_LINALG_GRAD", "1", /*overwrite=*/1);

    Variable A(make_spd(4), /*requires_grad=*/true);
    auto [LD, pivots] = ldl_factor(A);
    auto loss = tenzor::sum(LD);
    EXPECT_THROW(loss.backward(), std::runtime_error)
        << "With TENZOR_STRICT_LINALG_GRAD=1, LDL backward must throw";
}

TEST_F(StrictLinalgGradTest, LDLFactor_StrictModeFalseStillPermissive) {
    setenv("TENZOR_STRICT_LINALG_GRAD", "0", /*overwrite=*/1);

    Variable A(make_spd(4), /*requires_grad=*/true);
    auto [LD, pivots] = ldl_factor(A);
    auto loss = tenzor::sum(LD);
    EXPECT_NO_THROW(loss.backward())
        << "TENZOR_STRICT_LINALG_GRAD=0 should behave like unset";
}

}  // namespace
}  // namespace tenzor

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    try { tenzor::initialize(); } catch (...) {}
    return RUN_ALL_TESTS();
}
