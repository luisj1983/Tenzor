/**
 * @file test_dropout3d_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for Dropout3d layer
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn;

class Dropout3dMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(Dropout3dMultiDTypeTest, EvalModePassthrough) {
    Dropout3d dp(0.5);
    convert_model(dp);
    dp.eval();
    auto input = createInput({2, 3, 4, 4, 4}, false);
    auto output = dp.forward(input);

    auto in_cpu = input.tensor().to(Device::cpu()).to(DType::Float32);
    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* in_data = in_cpu.data<float>();
    auto* out_data = out_cpu.data<float>();
    for (int64_t i = 0; i < in_cpu.numel(); ++i) {
        EXPECT_NEAR(in_data[i], out_data[i], atol());
    }
}

TEST_P(Dropout3dMultiDTypeTest, OutputShape) {
    Dropout3d dp(0.5);
    convert_model(dp);
    dp.train();
    auto input = createInput({2, 3, 4, 4, 4}, false);
    auto output = dp.forward(input);
    auto shape = output.tensor().shape();
    ASSERT_EQ(shape.size(), 5);
    ASSERT_EQ(shape[0], 2);
    ASSERT_EQ(shape[1], 3);
    ASSERT_EQ(shape[2], 4);
}

TEST_P(Dropout3dMultiDTypeTest, ChannelwiseDropout) {
    Dropout3d dp(0.99);
    convert_model(dp);
    dp.train();
    auto input = Variable(createOnes({1, 10, 2, 2, 2}), false);
    auto output = dp.forward(input);
    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* out_data = out_cpu.data<float>();
    for (int64_t c = 0; c < 10; ++c) {
        bool first_val_zero = (std::abs(out_data[c * 8]) < 1e-6f);
        if (first_val_zero) {
            for (int64_t s = 0; s < 8; ++s) {
                EXPECT_NEAR(out_data[c * 8 + s], 0.0f, atol())
                    << "Channel " << c << " spatial " << s << " should be zero";
            }
        }
    }
}

TEST_P(Dropout3dMultiDTypeTest, Backward) {
    Dropout3d dp(0.3);
    convert_model(dp);
    dp.train();
    auto input = createInput({2, 3, 4, 4, 4}, true);
    auto output = dp.forward(input);
    auto loss = sum(output);
    loss.backward();
    ASSERT_TRUE(input.grad().has_value());
}

TEST_P(Dropout3dMultiDTypeTest, DTypePreserved) {
    Dropout3d dp(0.5);
    convert_model(dp);
    dp.eval();
    auto input = createInput({2, 3, 4, 4, 4}, false);
    auto output = dp.forward(input);
    expectDType(output.tensor());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(Dropout3dMultiDTypeTest);
