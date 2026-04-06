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

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(Conv1dMultiDTypeTest);
