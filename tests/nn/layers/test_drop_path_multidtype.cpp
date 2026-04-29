/**
 * @file test_drop_path_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for DropPath (stochastic depth) layer
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include "../../grad_flow_helpers.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/drop_path.hpp>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn;

class DropPathMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(DropPathMultiDTypeTest, EvalModePassthrough) {
    DropPath dp(0.5);
    convert_model(dp);
    dp.eval();
    auto input = createInput({4, 8}, false);
    auto output = dp.forward(input);

    auto in_cpu = input.tensor().to(Device::cpu()).to(DType::Float32);
    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* in_data = in_cpu.data<float>();
    auto* out_data = out_cpu.data<float>();
    for (int64_t i = 0; i < in_cpu.numel(); ++i) {
        EXPECT_NEAR(in_data[i], out_data[i], atol());
    }
}

TEST_P(DropPathMultiDTypeTest, ZeroDropRatePassthrough) {
    DropPath dp(0.0);
    convert_model(dp);
    dp.train();
    auto input = createInput({4, 8}, false);
    auto output = dp.forward(input);

    auto in_cpu = input.tensor().to(Device::cpu()).to(DType::Float32);
    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* in_data = in_cpu.data<float>();
    auto* out_data = out_cpu.data<float>();
    for (int64_t i = 0; i < in_cpu.numel(); ++i) {
        EXPECT_NEAR(in_data[i], out_data[i], atol());
    }
}

TEST_P(DropPathMultiDTypeTest, TrainingModeOutputShape) {
    DropPath dp(0.3);
    convert_model(dp);
    dp.train();
    auto input = createInput({8, 16}, false);
    auto output = dp.forward(input);
    ASSERT_EQ(output.tensor().shape()[0], 8);
    ASSERT_EQ(output.tensor().shape()[1], 16);
}

TEST_P(DropPathMultiDTypeTest, Backward) {
    // Use a larger batch so that the probability of every sample being dropped
    // is negligible — keeps EXPECT_GRAD_FLOWS deterministic with a non-zero
    // drop rate.
    DropPath dp(0.3);
    convert_model(dp);
    dp.train();
    auto input = createInput({16, 8}, true);
    auto output = dp.forward(input);
    auto loss = sum(output);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

TEST_P(DropPathMultiDTypeTest, DTypePreserved) {
    DropPath dp(0.3);
    convert_model(dp);
    dp.eval();
    auto input = createInput({4, 8}, false);
    auto output = dp.forward(input);
    expectDType(output.tensor());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DropPathMultiDTypeTest);
