/**
 * @file test_hardshrink_threshold_multidtype.cpp
 * @brief Multi-backend, multi-dtype tests for nn::hardshrink and nn::threshold
 *
 * Converted from test_hardshrink_threshold.cpp.  Uses MultiBackendDTypeTest
 * so that each test runs across all backend + dtype combinations.
 *
 * Since the original test wrote raw float values via data<float>(), this
 * version creates tensors in Float32 on CPU, sets the values, then converts
 * to the target dtype/device.  Result verification converts back to CPU
 * Float32 before reading with data<float>().
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>

#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class HardshrinkThresholdMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(HardshrinkThresholdMultiDTypeTest, HardshrinkForwardPassThroughAboveLambda) {
    // hardshrink(x, 0.5) -- values with |x| > 0.5 pass through, others zero.
    auto x_cpu = zeros({5}, DType::Float32, Device::cpu());
    x_cpu.data<float>()[0] = -1.0f;
    x_cpu.data<float>()[1] = -0.3f;
    x_cpu.data<float>()[2] =  0.0f;
    x_cpu.data<float>()[3] =  0.4f;
    x_cpu.data<float>()[4] =  0.9f;

    auto x_t = x_cpu.to(dtype()).to(device());
    auto x = Variable(x_t, false);
    auto y = tenzor::nn::hardshrink(x, 0.5);

    auto yd = y.tensor().to(Device::cpu()).to(DType::Float32);
    const float* yp = yd.data<float>();
    EXPECT_NEAR(yp[0], -1.0f, atol());
    EXPECT_NEAR(yp[1],  0.0f, atol());
    EXPECT_NEAR(yp[2],  0.0f, atol());
    EXPECT_NEAR(yp[3],  0.0f, atol());
    EXPECT_NEAR(yp[4],  0.9f, atol());
}

TEST_P(HardshrinkThresholdMultiDTypeTest, HardshrinkBackwardPassesGradient) {
    auto x_cpu = zeros({4}, DType::Float32, Device::cpu());
    x_cpu.data<float>()[0] = -1.0f;
    x_cpu.data<float>()[1] = -0.3f;
    x_cpu.data<float>()[2] =  0.3f;
    x_cpu.data<float>()[3] =  1.0f;

    auto x_t = x_cpu.to(dtype()).to(device());
    auto x = Variable(x_t, true);
    auto y = tenzor::nn::hardshrink(x, 0.5);
    auto loss = tenzor::sum(y);
    loss.backward();

    ASSERT_TRUE(x.grad().has_value());
    auto g = x.grad().value().to(Device::cpu()).to(DType::Float32);
    const float* gp = g.data<float>();
    // Gradient is 1 where |x| > 0.5, 0 otherwise.
    EXPECT_NEAR(gp[0], 1.0f, atol());  // |-1| > 0.5
    EXPECT_NEAR(gp[1], 0.0f, atol());  // |-0.3| < 0.5
    EXPECT_NEAR(gp[2], 0.0f, atol());
    EXPECT_NEAR(gp[3], 1.0f, atol());
}

TEST_P(HardshrinkThresholdMultiDTypeTest, HardshrinkNegativeLambdaThrows) {
    auto x_t = zeros({1}, dtype(), device());
    auto x = Variable(x_t, false);
    EXPECT_THROW(tenzor::nn::hardshrink(x, -0.1), std::invalid_argument);
}

TEST_P(HardshrinkThresholdMultiDTypeTest, ThresholdForward) {
    // threshold(x, t=0.5, value=-99) -- x > 0.5 passes through, else becomes -99.
    auto x_cpu = zeros({4}, DType::Float32, Device::cpu());
    x_cpu.data<float>()[0] = -1.0f;
    x_cpu.data<float>()[1] =  0.0f;
    x_cpu.data<float>()[2] =  0.7f;
    x_cpu.data<float>()[3] =  2.0f;

    auto x_t = x_cpu.to(dtype()).to(device());
    auto x = Variable(x_t, false);
    auto y = tenzor::nn::threshold(x, 0.5, -99.0);

    auto yd = y.tensor().to(Device::cpu()).to(DType::Float32);
    const float* yp = yd.data<float>();
    EXPECT_NEAR(yp[0], -99.0f, atol());
    EXPECT_NEAR(yp[1], -99.0f, atol());
    EXPECT_NEAR(yp[2],   0.7f, atol());
    EXPECT_NEAR(yp[3],   2.0f, atol());
}

TEST_P(HardshrinkThresholdMultiDTypeTest, ThresholdBackward) {
    auto x_cpu = zeros({4}, DType::Float32, Device::cpu());
    x_cpu.data<float>()[0] = -1.0f;
    x_cpu.data<float>()[1] =  0.0f;
    x_cpu.data<float>()[2] =  0.7f;
    x_cpu.data<float>()[3] =  2.0f;

    auto x_t = x_cpu.to(dtype()).to(device());
    auto x = Variable(x_t, true);
    auto y = tenzor::nn::threshold(x, 0.5, 0.0);
    auto loss = tenzor::sum(y);
    loss.backward();

    ASSERT_TRUE(x.grad().has_value());
    auto g = x.grad().value().to(Device::cpu()).to(DType::Float32);
    const float* gp = g.data<float>();
    EXPECT_NEAR(gp[0], 0.0f, atol());
    EXPECT_NEAR(gp[1], 0.0f, atol());
    EXPECT_NEAR(gp[2], 1.0f, atol());
    EXPECT_NEAR(gp[3], 1.0f, atol());
}

// ============================================================================
// Instantiate for all available backends and dtypes
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(HardshrinkThresholdMultiDTypeTest);
