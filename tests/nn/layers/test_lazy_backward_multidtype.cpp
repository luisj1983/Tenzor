/**
 * @file test_lazy_backward_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for Lazy* layer backward passes
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/lazy_linear.hpp>
#include <tenzor/nn/layers/lazy_conv.hpp>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class LazyBackwardMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        set_grad_enabled(true);
    }
};

TEST_P(LazyBackwardMultiDTypeTest, LazyLinear_ForwardShape) {
    LazyLinear layer(8);
    convert_model(layer);

    auto input = createInput({2, 16}, false);
    auto output = layer.forward(input);
    EXPECT_EQ(output.tensor().shape()[1], 8);
}

TEST_P(LazyBackwardMultiDTypeTest, LazyLinear_Backward) {
    LazyLinear layer(8);
    convert_model(layer);

    auto input = createInput({2, 16}, true);
    auto output = layer.forward(input);
    EXPECT_EQ(output.tensor().shape()[1], 8);

    auto loss = sum(output);
    loss.backward();

    ASSERT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad().value().shape()[0], 2);
    EXPECT_EQ(input.grad().value().shape()[1], 16);

    auto params = layer.parameters();
    EXPECT_FALSE(params.empty());
}

TEST_P(LazyBackwardMultiDTypeTest, LazyConv1d_Backward) {
    LazyConv1d layer(8, 3, 1, 1);
    convert_model(layer);

    auto input = createInput({1, 4, 16}, true);
    auto output = layer.forward(input);
    EXPECT_EQ(output.tensor().shape()[1], 8);

    auto loss = sum(output);
    loss.backward();

    ASSERT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad().value().shape()[0], 1);
}

TEST_P(LazyBackwardMultiDTypeTest, LazyConv2d_Backward) {
    LazyConv2d layer(16, 3, 1, 1);
    convert_model(layer);

    auto input = createInput({1, 8, 8, 8}, true);
    auto output = layer.forward(input);
    EXPECT_EQ(output.tensor().shape()[1], 16);

    auto loss = sum(output);
    loss.backward();

    ASSERT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad().value().shape()[1], 8);
}

TEST_P(LazyBackwardMultiDTypeTest, LazyConv3d_Backward) {
    LazyConv3d layer(8, 3, 1, 1);
    convert_model(layer);

    auto input = createInput({1, 4, 4, 4, 4}, true);
    auto output = layer.forward(input);
    EXPECT_EQ(output.tensor().shape()[1], 8);

    auto loss = sum(output);
    loss.backward();

    ASSERT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad().value().shape()[1], 4);
}

TEST_P(LazyBackwardMultiDTypeTest, LazyLinear_MaterializedParametersHaveGrad) {
    LazyLinear layer(4);
    convert_model(layer);

    auto input = createInput({2, 8}, true);
    auto output = layer.forward(input);
    auto loss = sum(output);
    loss.backward();

    auto params = layer.parameters();
    EXPECT_FALSE(params.empty());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(LazyBackwardMultiDTypeTest);
