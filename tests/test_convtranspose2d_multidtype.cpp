/**
 * @file test_convtranspose2d_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for ConvTranspose2d layer
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/conv.hpp>
#include "multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class ConvTranspose2dMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(ConvTranspose2dMultiDTypeTest, UpsampleShape) {
    // stride=2 upsamples spatially
    nn::ConvTranspose2d deconv(4, 8, 4, 2, 1);
    convert_model(deconv);

    auto input = createInput({1, 4, 8, 8}, false);
    auto output = deconv.forward(input);
    expectShape(output.tensor(), {1, 8, 16, 16});
    expectDevice(output.tensor());
    expectDType(output.tensor());
}

TEST_P(ConvTranspose2dMultiDTypeTest, IdentityStride) {
    nn::ConvTranspose2d deconv(3, 6, 3, 1, 1);
    convert_model(deconv);

    auto input = createInput({1, 3, 8, 8}, false);
    auto output = deconv.forward(input);
    expectShape(output.tensor(), {1, 6, 8, 8});
}

TEST_P(ConvTranspose2dMultiDTypeTest, BatchDim) {
    nn::ConvTranspose2d deconv(4, 8, 3, 2, 1);
    convert_model(deconv);

    auto input = createInput({4, 4, 6, 6}, false);
    auto output = deconv.forward(input);
    EXPECT_EQ(output.tensor().shape()[0], 4);
    EXPECT_EQ(output.tensor().shape()[1], 8);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ConvTranspose2dMultiDTypeTest);
