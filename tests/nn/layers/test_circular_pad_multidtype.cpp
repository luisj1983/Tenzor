/**
 * @file test_circular_pad_multidtype.cpp
 * @brief Multi-backend, multi-dtype tests for CircularPad1d/2d/3d.
 *
 * Forward shape and gradient propagation. Circular padding wraps values
 * from the opposite end, so a perturbation at position i causes changes
 * at the wrapped output positions too — backward must propagate through
 * those wrapped positions or the gradient drops contributions.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/padding.hpp>
#include "../../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class CircularPadMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(CircularPadMultiDTypeTest, CircularPad1d_ForwardShape) {
    nn::CircularPad1d pad(2, 1);  // pad left=2 right=1
    Variable input = createInput({1, 3, 4}, false);
    auto output = pad.forward(input);
    // Output W = 4 + 2 + 1 = 7
    expectShape(output.tensor(), {1, 3, 7});
}

TEST_P(CircularPadMultiDTypeTest, CircularPad1d_BackwardGradPopulated) {
    nn::CircularPad1d pad(1);  // symmetric
    Variable input = createInput({1, 2, 4}, true);
    auto output = pad.forward(input);
    sum(output).backward();
    ASSERT_TRUE(input.has_grad()) << device().to_string();
    EXPECT_EQ(input.grad()->numel(), input.tensor().numel());
    auto g_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f);
}

TEST_P(CircularPadMultiDTypeTest, CircularPad2d_ForwardShape) {
    nn::CircularPad2d pad(1, 1, 2, 2);  // L=R=1, T=B=2
    Variable input = createInput({1, 3, 4, 4}, false);
    auto output = pad.forward(input);
    // Output: H = 4+4 = 8, W = 4+2 = 6
    expectShape(output.tensor(), {1, 3, 8, 6});
}

TEST_P(CircularPadMultiDTypeTest, CircularPad2d_BackwardGradPopulated) {
    nn::CircularPad2d pad(1);
    Variable input = createInput({1, 2, 4, 4}, true);
    auto output = pad.forward(input);
    sum(output).backward();
    ASSERT_TRUE(input.has_grad()) << device().to_string();
    EXPECT_EQ(input.grad()->numel(), input.tensor().numel());
    auto g_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f);
}

TEST_P(CircularPadMultiDTypeTest, CircularPad3d_ForwardShape) {
    nn::CircularPad3d pad(1);  // symmetric on all 3 spatial dims
    Variable input = createInput({1, 2, 4, 4, 4}, false);
    auto output = pad.forward(input);
    // Output: D=H=W = 4+1+1 = 6
    expectShape(output.tensor(), {1, 2, 6, 6, 6});
}

TEST_P(CircularPadMultiDTypeTest, CircularPad3d_BackwardGradPopulated) {
    nn::CircularPad3d pad(1);
    Variable input = createInput({1, 2, 4, 4, 4}, true);
    auto output = pad.forward(input);
    sum(output).backward();
    ASSERT_TRUE(input.has_grad()) << device().to_string();
    EXPECT_EQ(input.grad()->numel(), input.tensor().numel());
    auto g_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(CircularPadMultiDTypeTest);
