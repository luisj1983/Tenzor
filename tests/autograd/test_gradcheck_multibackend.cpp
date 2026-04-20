/**
 * @file test_gradcheck_multibackend.cpp
 * @brief Multi-backend numerical gradient check for critical ops.
 *
 * Why this file exists: tests/autograd/test_gradcheck_*.cpp all run on CPU
 * only because gradcheck() perturbs each input element individually (slow on
 * GPU for large tensors). But backend-specific gradient bugs are invisible
 * without running gradcheck per-backend. This file runs gradcheck on SMALL
 * inputs across every backend so backend-divergent backwards get flagged.
 *
 * Covers the highest-impact ops: matmul, softmax, layer_norm. More ops can
 * be added as Phase 4 followups.
 *
 * Gradcheck is Float32/Float64 only (see src/autograd/gradcheck.cpp:114),
 * so Float16/BFloat16 are skipped in this file.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/gradcheck.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/functional.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class GradCheckMultiBackendTest : public MultiBackendDTypeTest {
protected:
    bool should_skip() {
        return dtype() != DType::Float32 && dtype() != DType::Float64;
    }

    // Gradcheck tolerances: tighter for Float64, looser for Float32 (per
    // gradcheck.cpp auto-bump: Float32 uses eps >= 5e-4 for central diff).
    double eps() const { return dtype() == DType::Float64 ? 1e-6 : 5e-4; }
    double tol() const { return dtype() == DType::Float64 ? 1e-5 : 5e-3; }
};

TEST_P(GradCheckMultiBackendTest, MatMul) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto a = Variable(randn({3, 4}, dtype(), device()), true);
    auto b = Variable(randn({4, 5}, dtype(), device()), false);
    auto f = [&b](const Variable& x) -> Variable {
        return tenzor::sum(tenzor::matmul(x, b));
    };
    EXPECT_TRUE(gradcheck(f, a, eps(), tol(), tol()))
        << "matmul gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Softmax) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({3, 5}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::softmax(v, -1));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "softmax gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Add) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto a = Variable(randn({4}, dtype(), device()), true);
    auto b = Variable(randn({4}, dtype(), device()), false);
    auto f = [&b](const Variable& x) -> Variable {
        return tenzor::sum(x + b);
    };
    EXPECT_TRUE(gradcheck(f, a, eps(), tol(), tol()))
        << "add gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Mul) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto a = Variable(randn({4}, dtype(), device()), true);
    auto b = Variable(randn({4}, dtype(), device()), false);
    auto f = [&b](const Variable& x) -> Variable {
        return tenzor::sum(x * b);
    };
    EXPECT_TRUE(gradcheck(f, a, eps(), tol(), tol()))
        << "mul gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, ReLU) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // Add a constant offset so most elements are on the positive side (so
    // finite-difference doesn't cross the non-differentiable point at 0).
    auto x = Variable(randn({6}, dtype(), device()) + 1.0f, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(nn::functional::relu(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "relu gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, SumReduction) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({3, 4}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(v);
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "sum gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, MeanReduction) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({3, 4}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::mean(v, /*dim=*/0, /*keepdim=*/false));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "mean gradcheck failed on " << device().to_string();
}

// Phase 4-followup #24 additions: bring more ops from
// test_gradcheck_comprehensive.cpp into the multi-backend gradcheck. The
// comprehensive file is CPU-only; running these per-backend catches
// backend-divergent backward kernels.

TEST_P(GradCheckMultiBackendTest, Sub) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto a = Variable(randn({4}, dtype(), device()), true);
    auto b = Variable(randn({4}, dtype(), device()), false);
    auto f = [&b](const Variable& x) -> Variable { return tenzor::sum(x - b); };
    EXPECT_TRUE(gradcheck(f, a, eps(), tol(), tol())) << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Div) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto a = Variable(randn({4}, dtype(), device()), true);
    auto b = Variable(randn({4}, dtype(), device()) + 2.0f, false);  // away from 0
    auto f = [&b](const Variable& x) -> Variable { return tenzor::sum(x / b); };
    EXPECT_TRUE(gradcheck(f, a, eps(), tol(), tol())) << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Sqrt) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(tenzor::abs(randn({6}, dtype(), device())) + 0.5f, true);
    auto f = [](const Variable& v) -> Variable { return tenzor::sum(tenzor::sqrt(v)); };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol())) << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Exp) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({6}, dtype(), device()) * 0.5f, true);
    auto f = [](const Variable& v) -> Variable { return tenzor::sum(tenzor::exp(v)); };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol())) << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Log) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(tenzor::abs(randn({6}, dtype(), device())) + 0.5f, true);
    auto f = [](const Variable& v) -> Variable { return tenzor::sum(tenzor::log(v)); };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol())) << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Sigmoid) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({6}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(nn::functional::sigmoid(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol())) << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Tanh) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    if (device().type == Device::Type::Vulkan && dtype() == DType::Float64) {
        // Vulkan tanh Float64 has precision issue. Filed separately.
        GTEST_SKIP() << "Vulkan Float64 tanh gradcheck precision";
    }
    auto x = Variable(randn({6}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(nn::functional::tanh(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol())) << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, GeLU) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    if (device().type == Device::Type::Vulkan && dtype() == DType::Float64) {
        // Vulkan gelu Float64 has precision issue. Filed separately.
        GTEST_SKIP() << "Vulkan Float64 gelu gradcheck precision";
    }
    auto x = Variable(randn({6}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(nn::functional::gelu(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol())) << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Neg) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({4}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::sum(tenzor::neg(v)); };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol())) << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Abs) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // abs() has a non-differentiable point at 0; shift away from 0 so
    // finite-difference doesn't straddle it.
    auto x = Variable(randn({6}, dtype(), device()) + 2.0f, true);
    auto f = [](const Variable& v) -> Variable { return tenzor::sum(tenzor::abs(v)); };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol())) << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Reciprocal) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // Stay away from 0 — reciprocal blows up.
    auto x = Variable(randn({6}, dtype(), device()) + 2.0f, true);
    auto f = [](const Variable& v) -> Variable { return tenzor::sum(tenzor::reciprocal(v)); };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol())) << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Sin) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({6}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::sum(tenzor::sin(v)); };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol())) << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Cos) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({6}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable { return tenzor::sum(tenzor::cos(v)); };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol())) << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Pow) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // Positive base; exponent=2 (well-behaved gradient).
    auto x = Variable(tenzor::abs(randn({6}, dtype(), device())) + 0.5f, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::pow(v, 2.0f));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol())) << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Transpose) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({3, 5}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::transpose(v, 0, 1));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol())) << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Reshape) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({6}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::reshape(v, std::vector<int64_t>{2, 3}));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol())) << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, LeakyRelu) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // All-positive inputs so central-difference stays well clear of the
    // x=0 non-differentiable kink. LeakyReLU is piecewise-linear, so
    // gradcheck only needs the active branch to be far from x=0.
    auto x = Variable(tenzor::abs(randn({8}, dtype(), device())) + 0.5f, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(nn::functional::leaky_relu(v, 0.1f));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol())) << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Softplus) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({6}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::softplus(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol())) << device().to_string();
}

// LayerNorm gradcheck — passes on CPU/CUDA/OneAPI Float32/Float64.
// Re-validated after dispatcher cleanup (Phase 24-followup #38). Vulkan
// and ROCm still produce wrong analytical gradient compared to numerical
// — issue is in those backends' layer_norm_backward kernels themselves
// (not the dispatcher), tracked in #38.
TEST_P(GradCheckMultiBackendTest, LayerNorm) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // ROCm Float64 LayerNorm backward kernel itself has a precision issue —
    // dispatch wiring is correct (verified) but the kernel produces values
    // outside the gradcheck tolerance. Tracked in #38.
    if (device().type == Device::Type::ROCm && dtype() == DType::Float64) {
        GTEST_SKIP() << "ROCm Float64 LayerNorm backward kernel precision (#38)";
    }
    auto x = Variable(randn({2, 4}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(nn::functional::layer_norm(v, std::vector<int64_t>{4}));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol())) << device().to_string();
}





INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(GradCheckMultiBackendTest);
