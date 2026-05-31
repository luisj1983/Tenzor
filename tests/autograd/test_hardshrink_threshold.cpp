/**
 * @file test_hardshrink_threshold.cpp
 * @brief P2.6b: verify nn::hardshrink and nn::threshold implementations
 *        (previously threw at runtime).
 */

#include <gtest/gtest.h>

#include <cmath>

#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>
#include "../grad_flow_helpers.hpp"
#include "../backend_test_fixture.hpp"

namespace tenzor {
namespace {

class HardshrinkThresholdTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

TEST_P(HardshrinkThresholdTest, HardshrinkForwardPassThroughAboveLambda) {
    // hardshrink(x, 0.5) — values with |x| > 0.5 pass through, others zero.
    auto x_host = zeros({5}, DType::Float32, Device::cpu());
    x_host.data<float>()[0] = -1.0f;
    x_host.data<float>()[1] = -0.3f;
    x_host.data<float>()[2] =  0.0f;
    x_host.data<float>()[3] =  0.4f;
    x_host.data<float>()[4] =  0.9f;
    auto x_t = x_host.to(device);

    auto x = Variable(x_t, false);
    auto y = tenzor::nn::hardshrink(x, 0.5);
    auto y_cpu = y.tensor().cpu();
    const float* yd = y_cpu.data<float>();
    EXPECT_FLOAT_EQ(yd[0], -1.0f);
    EXPECT_FLOAT_EQ(yd[1],  0.0f);
    EXPECT_FLOAT_EQ(yd[2],  0.0f);
    EXPECT_FLOAT_EQ(yd[3],  0.0f);
    EXPECT_FLOAT_EQ(yd[4],  0.9f);
}

TEST_P(HardshrinkThresholdTest, HardshrinkBackwardPassesGradient) {
    auto x_host = zeros({4}, DType::Float32, Device::cpu());
    x_host.data<float>()[0] = -1.0f;
    x_host.data<float>()[1] = -0.3f;
    x_host.data<float>()[2] =  0.3f;
    x_host.data<float>()[3] =  1.0f;
    auto x_t = x_host.to(device);

    auto x = Variable(x_t, true);
    auto y = tenzor::nn::hardshrink(x, 0.5);
    auto loss = tenzor::sum(y);
    loss.backward();

    EXPECT_GRAD_FLOWS(x);
    auto g_cpu = x.grad().value().cpu();
    const float* g = g_cpu.data<float>();
    // Gradient is 1 where |x| > 0.5, 0 otherwise.
    EXPECT_FLOAT_EQ(g[0], 1.0f);  // |-1| > 0.5
    EXPECT_FLOAT_EQ(g[1], 0.0f);  // |-0.3| < 0.5
    EXPECT_FLOAT_EQ(g[2], 0.0f);
    EXPECT_FLOAT_EQ(g[3], 1.0f);
}

TEST_P(HardshrinkThresholdTest, HardshrinkNegativeLambdaThrows) {
    auto x_t = zeros({1}, DType::Float32, device);
    auto x = Variable(x_t, false);
    EXPECT_THROW(tenzor::nn::hardshrink(x, -0.1), std::invalid_argument);
}

TEST_P(HardshrinkThresholdTest, ThresholdForward) {
    // threshold(x, t=0.5, value=-99) — x > 0.5 passes through, else becomes -99.
    auto x_host = zeros({4}, DType::Float32, Device::cpu());
    x_host.data<float>()[0] = -1.0f;
    x_host.data<float>()[1] =  0.0f;
    x_host.data<float>()[2] =  0.7f;
    x_host.data<float>()[3] =  2.0f;
    auto x_t = x_host.to(device);

    auto x = Variable(x_t, false);
    auto y = tenzor::nn::threshold(x, 0.5, -99.0);
    auto y_cpu = y.tensor().cpu();
    const float* yd = y_cpu.data<float>();
    EXPECT_FLOAT_EQ(yd[0], -99.0f);
    EXPECT_FLOAT_EQ(yd[1], -99.0f);
    EXPECT_FLOAT_EQ(yd[2],   0.7f);
    EXPECT_FLOAT_EQ(yd[3],   2.0f);
}

TEST_P(HardshrinkThresholdTest, ThresholdBackward) {
    auto x_host = zeros({4}, DType::Float32, Device::cpu());
    x_host.data<float>()[0] = -1.0f;
    x_host.data<float>()[1] =  0.0f;
    x_host.data<float>()[2] =  0.7f;
    x_host.data<float>()[3] =  2.0f;
    auto x_t = x_host.to(device);

    auto x = Variable(x_t, true);
    auto y = tenzor::nn::threshold(x, 0.5, 0.0);
    auto loss = tenzor::sum(y);
    loss.backward();

    EXPECT_GRAD_FLOWS(x);
    auto g_cpu = x.grad().value().cpu();
    const float* g = g_cpu.data<float>();
    EXPECT_FLOAT_EQ(g[0], 0.0f);
    EXPECT_FLOAT_EQ(g[1], 0.0f);
    EXPECT_FLOAT_EQ(g[2], 1.0f);
    EXPECT_FLOAT_EQ(g[3], 1.0f);
}

INSTANTIATE_BACKEND_TESTS(HardshrinkThresholdTest);

} // namespace
} // namespace tenzor
