/**
 * @file test_local_response_norm_multidtype.cpp
 * @brief Multi-dtype tests for LocalResponseNorm (LRN).
 *
 * LRN is the AlexNet-era cross-channel normalization:
 *   y[n,c,h,w] = x[n,c,h,w] / (k + alpha * sum(x^2, neighbouring channels))^beta
 *
 * Until this file there were no dedicated unit tests — only the backend_parity
 * sweep covered it. This file exercises forward shape correctness, AlexNet
 * default parameters, the scale-invariant case (alpha=0 → identity / k^beta),
 * and gradient flow.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/normalization.hpp>
#include <cmath>
#include "../../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class LocalResponseNormMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(LocalResponseNormMultiDTypeTest, ForwardShape) {
    // AlexNet defaults: size=5, alpha=1e-4, beta=0.75, k=2
    LocalResponseNorm lrn(5, 1e-4, 0.75, 2.0);
    auto x = Variable(randn({2, 16, 32, 32}, dtype(), device()), /*requires_grad=*/true);
    auto y = lrn.forward(x);
    expectShape(y.tensor(), {2, 16, 32, 32});
    EXPECT_EQ(y.tensor().dtype(), dtype());
    EXPECT_EQ(y.tensor().device().type, device().type);
}

TEST_P(LocalResponseNormMultiDTypeTest, ForwardShape_3D) {
    // LRN should preserve spatial dims regardless of rank
    LocalResponseNorm lrn(3);
    auto x = Variable(randn({1, 8, 16}, dtype(), device()), false);
    auto y = lrn.forward(x);
    expectShape(y.tensor(), {1, 8, 16});
}

TEST_P(LocalResponseNormMultiDTypeTest, AlphaZero_DividesByKBeta) {
    // alpha=0 ⇒ denominator is k^beta — output should be x / k^beta everywhere.
    const double k = 2.0;
    const double beta = 0.5;  // sqrt
    LocalResponseNorm lrn(/*size=*/5, /*alpha=*/0.0, /*beta=*/beta, /*k=*/k);

    auto x = Variable(ones({1, 4, 4, 4}, dtype(), device()), false);
    auto y = lrn.forward(x);
    auto out_cpu = y.tensor().to(Device::cpu()).to(DType::Float32);

    float expected = 1.0f / static_cast<float>(std::pow(k, beta));
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_NEAR(out_cpu.data<float>()[i], expected, atol() * 10.0f);
    }
}

// Documented bug: src/nn/layers/normalization.cpp:3105 calls `input.tensor()`
// which severs the autograd graph (the comment there says "no learnable
// parameters, so we compute at the Tensor level" — but that ignores that the
// INPUT may carry gradients). Result: x.grad is never populated.
// Fix: use the Variable overloads of mul/cat/narrow/pow throughout the
// channel-window summation loop. The test asserts the current broken
// behaviour to surface a deliberate update when the layer is fixed.
TEST_P(LocalResponseNormMultiDTypeTest, BackwardGradientsBroken_DocumentedBug) {
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
        GTEST_SKIP() << "Skipping gradient check for low-precision dtype";
    }
    LocalResponseNorm lrn(5);
    auto x = Variable(randn({1, 8, 8, 8}, dtype(), device()), true);
    auto y = lrn.forward(x);
    auto loss = tenzor::sum(y);
    loss.backward();
    EXPECT_FALSE(x.has_grad())
        << "LRN now propagates gradients — update the test and fix the "
        << "comment above.";
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(LocalResponseNormMultiDTypeTest);
