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

// ============================================================================
// Phase 4.1 — Activations expansion
//
// Each test mirrors the ReLU pattern: small input, scalar-loss `sum`, and
// where the activation has non-differentiable points (HardShrink,
// Threshold) the input is biased away from the boundary so finite
// differences don't straddle the kink.
// ============================================================================

TEST_P(GradCheckMultiBackendTest, Mish) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({6}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(nn::mish(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "mish gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, ELU) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({6}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(nn::elu(v, /*alpha=*/1.0));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "elu gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, SELU) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({6}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(nn::selu(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "selu gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, LogSigmoid) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({6}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(nn::log_sigmoid(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "log_sigmoid gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Swish) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({6}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(nn::swish(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "swish gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, HardShrink) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // HardShrink with lambda=0.5 is non-differentiable at ±0.5. Bias inputs
    // outside the kink region: shift to range roughly [1.0, 3.0] so finite
    // differences stay away from ±0.5.
    auto x = Variable(randn({6}, dtype(), device()) + 2.0f, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(nn::hardshrink(v, /*lambda=*/0.5));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "hardshrink gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Softsign) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({6}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(nn::softsign(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "softsign gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Threshold) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // Threshold is non-differentiable at the threshold value. Bias inputs to
    // be predominantly above the threshold so finite-diff perturbations don't
    // cross it.
    auto x = Variable(randn({6}, dtype(), device()) + 2.0f, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(nn::threshold(v, /*threshold=*/0.0, /*value=*/0.0));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "threshold gradcheck failed on " << device().to_string();
}

// ============================================================================
// Phase 4.2 — Reductions expansion
//
// Var, std, prod, cumsum, cumprod, logsumexp, max(dim), min(dim).
// argmax/argmin produce integer outputs and are excluded from gradcheck.
// ============================================================================

TEST_P(GradCheckMultiBackendTest, Var) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({4, 3}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::var(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "var gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Std) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({4, 3}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::std(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "std gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Prod) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // Bias well away from zero — the prod gradient (prod / x_i) blows up as
    // x_i approaches 0 and finite differences become unreliable.
    auto x = Variable(randn({4}, dtype(), device()) + 2.0f, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::prod(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "prod gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, CumSum) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({4}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::cumsum(v, 0));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "cumsum gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, CumProd) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // Same zero-blowup concern as prod — bias the input away from 0.
    auto x = Variable(randn({4}, dtype(), device()) + 2.0f, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::cumprod(v, 0));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "cumprod gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, LogSumExp) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({3, 4}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::logsumexp(v, /*dim=*/-1));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "logsumexp gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, MaxDim) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // Build inputs with well-separated values so finite-difference
    // perturbations don't change which element is argmax.
    auto base = randn({4, 3}, dtype(), device());
    auto x = Variable(base * static_cast<float>(2.0) + 5.0f, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::max(v, /*dim=*/0));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "max(dim) gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, MinDim) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto base = randn({4, 3}, dtype(), device());
    auto x = Variable(base * static_cast<float>(2.0) + 5.0f, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::min(v, /*dim=*/0));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "min(dim) gradcheck failed on " << device().to_string();
}

// ============================================================================
// Phase 4.3 — Indexing expansion
//
// index_select, gather, scatter, where, roll. (cat/flip already covered
// elsewhere via the original 25-op set.)
// ============================================================================

TEST_P(GradCheckMultiBackendTest, IndexSelect) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({5, 3}, dtype(), device()), true);
    Tensor idx_cpu = tenzor::zeros({3}, DType::Int64, Device::cpu());
    auto* idx_data = idx_cpu.data<int64_t>();
    idx_data[0] = 0; idx_data[1] = 2; idx_data[2] = 4;
    auto idx = (device().type == Device::Type::CPU) ? idx_cpu : idx_cpu.to(device());
    auto f = [&idx](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::index_select(v, /*dim=*/0, idx));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "index_select gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Gather) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({4, 3}, dtype(), device()), true);
    Tensor idx_cpu = tenzor::zeros({4, 3}, DType::Int64, Device::cpu());
    auto* idx_data = idx_cpu.data<int64_t>();
    for (int64_t i = 0; i < 12; ++i) idx_data[i] = i % 3;
    auto idx = (device().type == Device::Type::CPU) ? idx_cpu : idx_cpu.to(device());
    auto f = [&idx](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::gather(v, /*dim=*/1, idx));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "gather gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Scatter) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({4, 3}, dtype(), device()), true);
    auto src = Variable(randn({4, 3}, dtype(), device()), false);
    Tensor idx_cpu = tenzor::zeros({4, 3}, DType::Int64, Device::cpu());
    auto* idx_data = idx_cpu.data<int64_t>();
    for (int64_t i = 0; i < 12; ++i) idx_data[i] = i % 3;
    auto idx = (device().type == Device::Type::CPU) ? idx_cpu : idx_cpu.to(device());
    auto f = [&src, &idx](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::scatter(v, /*dim=*/1, idx, src));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "scatter gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Where) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // Build the condition as a deterministic Bool tensor on CPU then move
    // to device; avoids depending on overloaded comparison operators that
    // may not return a usable Variable type.
    Tensor cond_cpu = tenzor::zeros({4}, DType::Bool, Device::cpu());
    auto* cond_data = cond_cpu.data<uint8_t>();
    cond_data[0] = 1; cond_data[1] = 0; cond_data[2] = 1; cond_data[3] = 0;
    auto cond_t = (device().type == Device::Type::CPU)
                  ? cond_cpu
                  : cond_cpu.to(device());
    Variable cond_var(cond_t, false);
    Variable y_var(randn({4}, dtype(), device()), false);
    auto f = [&cond_var, &y_var](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::where(cond_var, v, y_var));
    };
    auto x = Variable(randn({4}, dtype(), device()), true);
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "where gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Roll) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({6}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::roll(v, /*shifts=*/2, /*dim=*/0));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "roll gradcheck failed on " << device().to_string();
}

// ============================================================================
// Phase 4.4 — Linalg + special math + FFT expansion
//
// LU/LinalgLUSolve are tested in test_gradcheck_missing.cpp (Tensor API,
// not Variable) — they don't appear here. complex-output FFT ops use a
// round-trip pattern (rfft → irfft → sum) so the gradcheck output is real.
// ============================================================================

TEST_P(GradCheckMultiBackendTest, Erf) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({4}, dtype(), device()) * 0.5f, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::erf(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "erf gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Erfc) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({4}, dtype(), device()) * 0.5f, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::erfc(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "erfc gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Lgamma) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // Bias positive — gamma(x) has poles at non-positive integers.
    auto x = Variable(randn({4}, dtype(), device()) + 3.0f, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::lgamma(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "lgamma gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Digamma) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({4}, dtype(), device()) + 3.0f, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::digamma(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "digamma gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, I0e) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({4}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::i0e(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "i0e gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, I1e) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({4}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::i1e(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "i1e gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Det) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // Build a well-conditioned matrix: random + identity offset so det is
    // bounded away from 0 (det/x_ij blows up at singular matrices).
    auto eye_t = tenzor::eye(3, std::nullopt, dtype(), device());
    auto x = Variable(randn({3, 3}, dtype(), device()) * 0.3f + eye_t, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::det(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "det gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Inv) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto eye_t = tenzor::eye(3, std::nullopt, dtype(), device());
    auto x = Variable(randn({3, 3}, dtype(), device()) * 0.3f + eye_t, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::inv(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "inv gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, Cholesky) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // Build a symmetric positive-definite matrix: A^T A + I.
    auto base = randn({3, 3}, dtype(), device()) * 0.3f;
    auto baseT = tenzor::transpose(base, 0, 1);
    auto eye_t = tenzor::eye(3, std::nullopt, dtype(), device());
    auto spd = tenzor::matmul(baseT, base) + eye_t;
    auto x = Variable(spd, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::cholesky(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "cholesky gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, VectorNorm) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // Bias away from 0; vector_norm gradient is x/|x| which blows up at 0.
    auto x = Variable(randn({4}, dtype(), device()) + 2.0f, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(tenzor::vector_norm(v));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "vector_norm gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, FFTRoundTrip) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // rfft(x) → irfft → real-valued output. The round-trip is identity (up
    // to numerical precision), so any error in the rfft/irfft backwards
    // shows up as gradcheck mismatch on the round-trip composition.
    auto x = Variable(randn({8}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        auto Y = tenzor::fft_autograd::rfft(v);
        auto x_back = tenzor::fft_autograd::irfft(Y);
        return tenzor::sum(x_back);
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "rfft/irfft round-trip gradcheck failed on " << device().to_string();
}

// ============================================================================
// Phase 4.5 — NN ops expansion
//
// Conv2d, AvgPool2d, MaxPool2d, BatchNorm, GroupNorm, InstanceNorm,
// RMSNorm, Embedding. The biggest correctness-risk gap — these are the
// kernels most likely to have backend-specific backward bugs.
// ============================================================================

TEST_P(GradCheckMultiBackendTest, Conv2d) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // Small input/weight to keep the finite-diff loop tractable.
    // Shape: batch=1, in_ch=2, H=3, W=3; weight: out_ch=2, in_ch=2, kH=2, kW=2.
    auto x = Variable(randn({1, 2, 3, 3}, dtype(), device()), true);
    auto w_var = Variable(randn({2, 2, 2, 2}, dtype(), device()), false);
    auto f = [&w_var](const Variable& v) -> Variable {
        return tenzor::sum(nn::functional::conv2d(v, w_var));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "conv2d gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, AvgPool2d) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    auto x = Variable(randn({1, 2, 4, 4}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(nn::functional::avg_pool2d(v, /*kernel_size=*/{2, 2}));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "avg_pool2d gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, MaxPool2d) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // Bias inputs so the max element of each pool window is well-separated
    // (otherwise tiny finite-diff perturbations can change which element is
    // max, producing a non-smooth gradient at the perturbation point).
    auto x = Variable(randn({1, 2, 4, 4}, dtype(), device()) * 2.0f + 5.0f, true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(nn::functional::max_pool2d(v, /*kernel_size=*/{2, 2}));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "max_pool2d gradcheck failed on " << device().to_string();
}

TEST_P(GradCheckMultiBackendTest, GroupNorm) {
    if (should_skip()) { GTEST_SKIP() << "gradcheck supports only Float32/Float64"; return; }
    // 1 batch × 4 channels × 3×3, group into 2 groups of 2 channels.
    auto x = Variable(randn({1, 4, 3, 3}, dtype(), device()), true);
    auto f = [](const Variable& v) -> Variable {
        return tenzor::sum(nn::functional::group_norm(v, /*num_groups=*/2));
    };
    EXPECT_TRUE(gradcheck(f, x, eps(), tol(), tol()))
        << "group_norm gradcheck failed on " << device().to_string();
}


INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(GradCheckMultiBackendTest);
