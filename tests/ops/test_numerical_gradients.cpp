/**
 * @file test_numerical_gradients.cpp
 * @brief CPU-only numerical gradient verification for differentiable
 *        operations. Uses gradcheck() to confirm analytical gradients match
 *        finite-difference numerical gradients on small inputs.
 *
 * NOTE: This file is intentionally CPU-only and is NOT a multi-backend
 * test. The gradcheck() infrastructure runs finite-difference perturbations
 * directly on raw tensor pointers (CPU memory), so multi-backend coverage
 * lives in tests/autograd/test_gradcheck_multibackend.cpp instead. Keeping
 * this file CPU-only is the correct test design: it acts as the reference
 * the multi-backend gradchecks compare against.
 *
 * If you are adding a new op's gradcheck, prefer:
 *   - tests/autograd/test_gradcheck_multibackend.cpp (5 backends × Float32+Float64)
 *   - tests/autograd/test_gradcheck_missing.cpp (CPU-only sweep)
 * Add to this file ONLY when you specifically want to assert the CPU
 * reference gradient against finite differences without involving GPUs.
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/gradcheck.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cmath>
#include <algorithm>  // std::clamp

using namespace tenzor;

class NumericalGradientTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }

    void SetUp() override {
        set_grad_enabled(true);
    }

    /// Create a small random CPU tensor as Variable
    auto make_var(std::vector<int64_t> shape, float low = -1.0f, float high = 1.0f) -> Variable {
        auto t = tenzor::randn(shape, DType::Float32, Device::cpu());
        // Affinely map the standard-normal samples toward [low, high]. randn
        // has unbounded support, so the affine map alone can still produce
        // values OUTSIDE [low, high] -- including negatives, which are out of
        // domain for log()/sqrt() and would nondeterministically NaN the
        // gradcheck (a domain-violation failure, not an op bug). Clamp to the
        // requested bounds so domain-restricted ops always see valid inputs.
        auto range = high - low;
        auto data = t.data<float>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            data[i] = data[i] * range * 0.5f + (low + high) * 0.5f;
            data[i] = std::clamp(data[i], low, high);
        }
        return Variable(t, true);
    }
};

// ---- Element-wise math ----

TEST_F(NumericalGradientTest, Exp) {
    auto x = make_var({4}, -2.0f, 2.0f);
    auto f = [](const Variable& v) { return tenzor::exp(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_F(NumericalGradientTest, Log) {
    auto x = make_var({4}, 0.5f, 3.0f);  // positive values
    auto f = [](const Variable& v) { return tenzor::log(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_F(NumericalGradientTest, Sqrt) {
    auto x = make_var({4}, 0.5f, 3.0f);  // positive values
    auto f = [](const Variable& v) { return tenzor::sqrt(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_F(NumericalGradientTest, Sin) {
    auto x = make_var({4});
    auto f = [](const Variable& v) { return tenzor::sin(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_F(NumericalGradientTest, Cos) {
    auto x = make_var({4});
    auto f = [](const Variable& v) { return tenzor::cos(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_F(NumericalGradientTest, Tanh) {
    auto x = make_var({4});
    auto f = [](const Variable& v) { return tenzor::tanh(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_F(NumericalGradientTest, Abs) {
    auto x = make_var({4}, 0.5f, 3.0f);  // avoid zero (non-differentiable)
    auto f = [](const Variable& v) { return tenzor::abs(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_F(NumericalGradientTest, Neg) {
    auto x = make_var({4});
    auto f = [](const Variable& v) { return tenzor::neg(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

// ---- Binary operations ----

TEST_F(NumericalGradientTest, AddBroadcast) {
    auto a = make_var({3, 4});
    auto b = make_var({4});
    std::function<Variable(const Variable&)> f = [&b](const Variable& v) { return v + b; };
    EXPECT_TRUE(gradcheck(f, a));
}

TEST_F(NumericalGradientTest, MulBroadcast) {
    auto a = make_var({3, 4});
    auto b = make_var({4});
    std::function<Variable(const Variable&)> f = [&b](const Variable& v) { return v * b; };
    EXPECT_TRUE(gradcheck(f, a));
}

TEST_F(NumericalGradientTest, Div) {
    auto a = make_var({4});
    auto b = make_var({4}, 0.5f, 3.0f);  // positive denominator
    std::function<Variable(const Variable&)> f = [&b](const Variable& v) { return v / b; };
    EXPECT_TRUE(gradcheck(f, a));
}

// ---- Reductions ----

TEST_F(NumericalGradientTest, Sum) {
    auto x = make_var({3, 4});
    auto f = [](const Variable& v) { return tenzor::sum(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_F(NumericalGradientTest, Mean) {
    auto x = make_var({3, 4});
    auto f = [](const Variable& v) { return tenzor::mean(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

// ---- MatMul ----

TEST_F(NumericalGradientTest, MatMul) {
    auto x = make_var({3, 4});
    auto w = make_var({4, 2});
    std::function<Variable(const Variable&)> f = [&w](const Variable& v) {
        return tenzor::sum(tenzor::matmul(v, w));
    };
    EXPECT_TRUE(gradcheck(f, x));
}

// ---- Activations ----

TEST_F(NumericalGradientTest, Sigmoid) {
    auto x = make_var({4});
    std::function<Variable(const Variable&)> f = [](const Variable& v) {
        return tenzor::sigmoid(v);
    };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_F(NumericalGradientTest, Gelu) {
    auto x = make_var({4});
    std::function<Variable(const Variable&)> f = [](const Variable& v) {
        return tenzor::gelu(v);
    };
    EXPECT_TRUE(gradcheck(f, x));
}

// ---- Softmax ----

// LogSoftmax gradcheck is tested in tests/autograd/test_gradcheck_extended.cpp
// Omitted here due to numerical sensitivity in finite-difference approximation

// ---- Pow ----

TEST_F(NumericalGradientTest, PowSquare) {
    auto x = make_var({4}, 0.5f, 3.0f);  // positive base
    std::function<Variable(const Variable&)> f = [](const Variable& v) {
        return tenzor::sum(v * v);
    };
    EXPECT_TRUE(gradcheck(f, x));
}
