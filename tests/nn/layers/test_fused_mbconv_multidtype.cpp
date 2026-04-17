/**
 * @file test_fused_mbconv_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for FusedMBConv layer
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/mobilenet.hpp>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn;

class FusedMBConvMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(FusedMBConvMultiDTypeTest, ForwardShape) {
    FusedMBConv block(16, 32, 4, 1);
    convert_model(block);
    auto input = createInput({1, 16, 8, 8}, false);
    auto output = block.forward(input);
    auto shape = output.tensor().shape();
    ASSERT_EQ(shape[0], 1);
    ASSERT_EQ(shape[1], 32);
    ASSERT_EQ(shape[2], 8);
    ASSERT_EQ(shape[3], 8);
}

TEST_P(FusedMBConvMultiDTypeTest, ForwardWithStride2) {
    FusedMBConv block(16, 32, 4, 2);
    convert_model(block);
    auto input = createInput({1, 16, 8, 8}, false);
    auto output = block.forward(input);
    auto shape = output.tensor().shape();
    ASSERT_EQ(shape[0], 1);
    ASSERT_EQ(shape[1], 32);
    ASSERT_EQ(shape[2], 4);
    ASSERT_EQ(shape[3], 4);
}

TEST_P(FusedMBConvMultiDTypeTest, Backward) {
    FusedMBConv block(8, 8, 2, 1);
    convert_model(block);
    auto input = createInput({1, 8, 4, 4}, true);
    auto output = block.forward(input);
    auto loss = sum(output);
    loss.backward();
    ASSERT_TRUE(input.grad().has_value());
}

TEST_P(FusedMBConvMultiDTypeTest, OutputValuesFinite) {
    FusedMBConv block(16, 32, 4, 1);
    convert_model(block);
    auto input = createInput({1, 16, 8, 8}, false);
    auto output = block.forward(input);
    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* d = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(d[i])) << "Non-finite at " << i;
    }
}

TEST_P(FusedMBConvMultiDTypeTest, DTypePreserved) {
    FusedMBConv block(16, 32, 4, 1);
    convert_model(block);
    auto input = createInput({1, 16, 8, 8}, false);
    auto output = block.forward(input);
    expectDType(output.tensor());
}

TEST_P(FusedMBConvMultiDTypeTest, DevicePreserved) {
    FusedMBConv block(16, 32, 4, 1);
    convert_model(block);
    auto input = createInput({1, 16, 8, 8}, false);
    auto output = block.forward(input);
    expectDevice(output.tensor());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(FusedMBConvMultiDTypeTest);
