/**
 * @file test_glu.cpp
 * @brief Tests for GLU (Gated Linear Unit) activation layer.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/activations/activations.hpp>

using namespace tenzor;
using namespace tenzor::nn;

class GLUTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
    void SetUp() override { set_grad_enabled(true); }
};

TEST_F(GLUTest, ForwardShape) {
    GLU glu(-1);
    auto input = Variable(randn({2, 8}, DType::Float32, Device::cpu()), false);
    auto output = glu.forward(input);
    // GLU splits last dim in half: 8 -> 4
    ASSERT_EQ(output.tensor().shape()[0], 2);
    ASSERT_EQ(output.tensor().shape()[1], 4);
}

TEST_F(GLUTest, ForwardDim0) {
    GLU glu(0);
    auto input = Variable(randn({6, 4}, DType::Float32, Device::cpu()), false);
    auto output = glu.forward(input);
    ASSERT_EQ(output.tensor().shape()[0], 3);
    ASSERT_EQ(output.tensor().shape()[1], 4);
}

TEST_F(GLUTest, ForwardValues) {
    GLU glu(-1);
    // Input of size [1, 4]: first half [a, b], second half [c, d]
    // Output = [a, b] * sigmoid([c, d])
    auto t = zeros({1, 4}, DType::Float32, Device::cpu());
    float* d = t.data<float>();
    d[0] = 1.0f; d[1] = 2.0f; d[2] = 0.0f; d[3] = 0.0f;
    auto input = Variable(t, false);
    auto output = glu.forward(input);
    // sigmoid(0) = 0.5, so output = [0.5, 1.0]
    auto out_data = output.tensor().data<float>();
    EXPECT_NEAR(out_data[0], 0.5f, 1e-5f);
    EXPECT_NEAR(out_data[1], 1.0f, 1e-5f);
}

TEST_F(GLUTest, Backward) {
    GLU glu(-1);
    auto input = Variable(randn({4, 8}, DType::Float32, Device::cpu()), true);
    auto output = glu.forward(input);
    auto loss = sum(output);
    loss.backward();
    auto grad = input.grad();
    ASSERT_TRUE(grad.has_value());
    ASSERT_EQ(grad.value().shape()[0], 4);
    ASSERT_EQ(grad.value().shape()[1], 8);
}
