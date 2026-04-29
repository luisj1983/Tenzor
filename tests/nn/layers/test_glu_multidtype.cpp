/**
 * @file test_glu_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for GLU activation layer
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include "../../grad_flow_helpers.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/activations/activations.hpp>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn;

class GLUMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(GLUMultiDTypeTest, ForwardShape) {
    GLU glu(-1);
    auto input = createInput({2, 8}, false);
    auto output = glu.forward(input);
    ASSERT_EQ(output.tensor().shape()[0], 2);
    ASSERT_EQ(output.tensor().shape()[1], 4);
}

TEST_P(GLUMultiDTypeTest, ForwardDim0) {
    GLU glu(0);
    auto input = createInput({6, 4}, false);
    auto output = glu.forward(input);
    ASSERT_EQ(output.tensor().shape()[0], 3);
    ASSERT_EQ(output.tensor().shape()[1], 4);
}

TEST_P(GLUMultiDTypeTest, ForwardValues) {
    GLU glu(-1);
    // Build on CPU then convert
    auto t = tenzor::zeros({1, 4}, DType::Float32, Device::cpu());
    float* d = t.data<float>();
    d[0] = 1.0f; d[1] = 2.0f; d[2] = 0.0f; d[3] = 0.0f;
    t = t.to(dtype()).to(device());
    auto input = Variable(t, false);
    auto output = glu.forward(input);
    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* out_data = out_cpu.data<float>();
    EXPECT_NEAR(out_data[0], 0.5f, atol());
    EXPECT_NEAR(out_data[1], 1.0f, atol());
}

TEST_P(GLUMultiDTypeTest, Backward) {
    GLU glu(-1);
    auto input = createInput({4, 8}, true);
    auto output = glu.forward(input);
    auto loss = sum(output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
    auto grad = input.grad();
    ASSERT_EQ(grad.value().shape()[0], 4);
    ASSERT_EQ(grad.value().shape()[1], 8);
}

TEST_P(GLUMultiDTypeTest, DTypePreserved) {
    GLU glu(-1);
    auto input = createInput({2, 8}, false);
    auto output = glu.forward(input);
    expectDType(output.tensor());
}

TEST_P(GLUMultiDTypeTest, DevicePreserved) {
    GLU glu(-1);
    auto input = createInput({2, 8}, false);
    auto output = glu.forward(input);
    expectDevice(output.tensor());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(GLUMultiDTypeTest);
