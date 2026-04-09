/**
 * @file test_numerical_gradients_multidtype.cpp
 * @brief Multi-backend numerical gradient verification for differentiable operations
 *
 * Uses gradcheck() to verify analytical gradients match finite-difference
 * numerical gradients for core operations across all backends and dtypes.
 * Uses small tensor sizes for speed.
 *
 * Float16 is skipped because finite-difference numerical gradients require
 * more precision than Float16 can provide.
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/gradcheck.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class NumericalGradientMultiTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        // Float16 lacks sufficient precision for numerical gradient checks
        if (dtype() == DType::Float16) {
            GTEST_SKIP() << "Float16 lacks precision for numerical gradient verification";
        }

        // Seed the RNG so `make_var` produces deterministic inputs across
        // runs. Without this, each subprocess invocation of ctest re-seeds
        // from time/urandom and occasionally generates inputs that land in
        // numerically unfavorable regions (e.g. row sums near zero for
        // MatMul, values near exp overflow, etc.), causing spurious flaky
        // failures for a well-conditioned analytical gradient.
        tenzor::manual_seed(0x7e4207u);

        set_grad_enabled(true);
    }

    /// Create a small random tensor as Variable on the test device.
    /// Values are uniformly distributed in [low, high]. Using `rand` (uniform
    /// on [0,1)) avoids the previous bug where `randn` (standard normal) was
    /// rescaled naively — that leaked out-of-range values (e.g. negatives
    /// into log()) which caused numerical gradients to become NaN and made
    /// tests flakily fail.
    auto make_var(std::vector<int64_t> shape, float low = -1.0f, float high = 1.0f) -> Variable {
        auto t = tenzor::rand(shape, DType::Float32, Device::cpu());
        auto data = t.data<float>();
        const float span = high - low;
        for (int64_t i = 0; i < t.numel(); ++i) {
            data[i] = data[i] * span + low;
        }
        if (dtype() != DType::Float32) {
            t = t.to(dtype());
        }
        t = t.to(device());
        return Variable(t, true);
    }
};

// ---- Element-wise math ----

TEST_P(NumericalGradientMultiTest, Exp) {
    auto x = make_var({4}, -2.0f, 2.0f);
    auto f = [](const Variable& v) { return tenzor::exp(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(NumericalGradientMultiTest, Log) {
    auto x = make_var({4}, 0.5f, 3.0f);  // positive values
    auto f = [](const Variable& v) { return tenzor::log(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(NumericalGradientMultiTest, Sqrt) {
    auto x = make_var({4}, 0.5f, 3.0f);  // positive values
    auto f = [](const Variable& v) { return tenzor::sqrt(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(NumericalGradientMultiTest, Sin) {
    auto x = make_var({4});
    auto f = [](const Variable& v) { return tenzor::sin(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(NumericalGradientMultiTest, Cos) {
    auto x = make_var({4});
    auto f = [](const Variable& v) { return tenzor::cos(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(NumericalGradientMultiTest, Tanh) {
    auto x = make_var({4});
    auto f = [](const Variable& v) { return tenzor::tanh(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(NumericalGradientMultiTest, Abs) {
    auto x = make_var({4}, 0.5f, 3.0f);  // avoid zero (non-differentiable)
    auto f = [](const Variable& v) { return tenzor::abs(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(NumericalGradientMultiTest, Neg) {
    auto x = make_var({4});
    auto f = [](const Variable& v) { return tenzor::neg(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

// ---- Binary operations ----

TEST_P(NumericalGradientMultiTest, AddBroadcast) {
    auto a = make_var({3, 4});
    auto b = make_var({4});
    std::function<Variable(const Variable&)> f = [&b](const Variable& v) { return v + b; };
    EXPECT_TRUE(gradcheck(f, a));
}

TEST_P(NumericalGradientMultiTest, MulBroadcast) {
    auto a = make_var({3, 4});
    auto b = make_var({4});
    std::function<Variable(const Variable&)> f = [&b](const Variable& v) { return v * b; };
    EXPECT_TRUE(gradcheck(f, a));
}

TEST_P(NumericalGradientMultiTest, Div) {
    auto a = make_var({4});
    auto b = make_var({4}, 0.5f, 3.0f);  // positive denominator
    std::function<Variable(const Variable&)> f = [&b](const Variable& v) { return v / b; };
    EXPECT_TRUE(gradcheck(f, a));
}

// ---- Reductions ----

TEST_P(NumericalGradientMultiTest, Sum) {
    auto x = make_var({3, 4});
    auto f = [](const Variable& v) { return tenzor::sum(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(NumericalGradientMultiTest, Mean) {
    auto x = make_var({3, 4});
    auto f = [](const Variable& v) { return tenzor::mean(v); };
    EXPECT_TRUE(gradcheck(f, x));
}

// ---- MatMul ----

TEST_P(NumericalGradientMultiTest, MatMul) {
    // Keep the shapes small and the values modest so the finite-difference
    // cancellation noise in Float32 stays within the Float32 tolerance.
    // For f(v) = sum(v @ w) with values in [-0.5, 0.5], |f| is bounded by
    // (#input_elements * max|v| * max|w| * output_size) which keeps the FD
    // noise well under gradcheck's 1e-2 relative tolerance.
    auto x = make_var({2, 3}, -0.5f, 0.5f);
    auto w = make_var({3, 2}, -0.5f, 0.5f);
    std::function<Variable(const Variable&)> f = [&w](const Variable& v) {
        return tenzor::sum(tenzor::matmul(v, w));
    };
    EXPECT_TRUE(gradcheck(f, x));
}

// ---- Activations ----

TEST_P(NumericalGradientMultiTest, Sigmoid) {
    auto x = make_var({4});
    std::function<Variable(const Variable&)> f = [](const Variable& v) {
        return tenzor::sigmoid(v);
    };
    EXPECT_TRUE(gradcheck(f, x));
}

TEST_P(NumericalGradientMultiTest, Gelu) {
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

TEST_P(NumericalGradientMultiTest, PowSquare) {
    auto x = make_var({4}, 0.5f, 3.0f);  // positive base
    std::function<Variable(const Variable&)> f = [](const Variable& v) {
        return tenzor::sum(v * v);
    };
    EXPECT_TRUE(gradcheck(f, x));
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(NumericalGradientMultiTest);
