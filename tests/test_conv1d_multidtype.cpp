/**
 * @file test_conv1d_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for Conv1d layer
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/conv.hpp>
#include "multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class Conv1dMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(Conv1dMultiDTypeTest, BasicForwardShape) {
    auto conv = nn::Conv1d(2, 3, 3, 1, 0, 1, 1, false);
    convert_model(conv);

    auto input = createInput({1, 2, 5}, false);
    auto output = conv.forward(input);
    expectShape(output.tensor(), {1, 3, 3});
    expectDevice(output.tensor());
    expectDType(output.tensor());
}

TEST_P(Conv1dMultiDTypeTest, WithBias) {
    auto conv = nn::Conv1d(4, 8, 3, 1, 1, 1, 1, true);
    convert_model(conv);

    auto input = createInput({2, 4, 10}, false);
    auto output = conv.forward(input);
    expectShape(output.tensor(), {2, 8, 10});
}

TEST_P(Conv1dMultiDTypeTest, WithStridePadding) {
    auto conv = nn::Conv1d(3, 6, 5, 2, 2);
    convert_model(conv);

    auto input = createInput({1, 3, 16}, false);
    auto output = conv.forward(input);
    // L_out = floor((16 + 2*2 - 5) / 2) + 1 = floor(15/2) + 1 = 8
    expectShape(output.tensor(), {1, 6, 8});
}

TEST_P(Conv1dMultiDTypeTest, MultipleKernelSizes) {
    for (int64_t k : {1, 3, 5, 7}) {
        auto conv = nn::Conv1d(2, 4, k, 1, k / 2);
        convert_model(conv);

        auto input = createInput({1, 2, 16}, false);
        auto output = conv.forward(input);
        EXPECT_EQ(output.tensor().shape()[0], 1);
        EXPECT_EQ(output.tensor().shape()[1], 4);
    }
}

// Backward — exercises Conv1dBackwardInput, Conv1dBackwardWeight,
// Conv1dBackwardBias kernels. The optimizer-style sum().backward() pattern
// populates .grad() on the input tensor and the conv's parameters.
TEST_P(Conv1dMultiDTypeTest, BackwardProducesGradients) {
    auto conv = nn::Conv1d(2, 4, 3, 1, 1, 1, 1, /*bias=*/true);
    convert_model(conv);

    auto input = createInput({1, 2, 8}, /*requires_grad=*/true);
    auto output = conv.forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();

    ASSERT_TRUE(input.has_grad()) << "input gradient must be populated";
    auto g = input.grad().value();
    EXPECT_EQ(g.shape()[0], 1);
    EXPECT_EQ(g.shape()[1], 2);
    EXPECT_EQ(g.shape()[2], 8);
    expectDevice(g);

    // Both weight and bias parameters should see gradients
    for (const auto& [name, p] : conv.named_parameters()) {
        ASSERT_TRUE(p->has_grad()) << "parameter " << name << " missing grad";
    }
}

TEST_P(Conv1dMultiDTypeTest, BackwardWithStridePadding) {
    auto conv = nn::Conv1d(3, 6, 5, 2, 2);
    convert_model(conv);

    auto input = createInput({2, 3, 12}, /*requires_grad=*/true);
    auto output = conv.forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();

    ASSERT_TRUE(input.has_grad());
    EXPECT_EQ(input.grad().value().shape()[0], 2);
    EXPECT_EQ(input.grad().value().shape()[1], 3);
    EXPECT_EQ(input.grad().value().shape()[2], 12);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(Conv1dMultiDTypeTest);
