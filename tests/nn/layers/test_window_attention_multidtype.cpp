/**
 * @file test_window_attention_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for WindowAttention layer (Swin Transformer)
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/vision.hpp>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class WindowAttentionMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        set_grad_enabled(true);
    }
};

TEST_P(WindowAttentionMultiDTypeTest, ForwardShape) {
    WindowAttention wa(32, 7, 4);
    convert_model(wa);

    // Input: (batch * num_windows, window_size*window_size, dim)
    auto input = createInput({4, 49, 32}, false);
    auto output = wa.forward(input, Tensor{});
    auto shape = output.tensor().shape();
    ASSERT_EQ(shape.size(), 3);
    ASSERT_EQ(shape[0], 4);
    ASSERT_EQ(shape[1], 49);
    ASSERT_EQ(shape[2], 32);
}

TEST_P(WindowAttentionMultiDTypeTest, Backward) {
    WindowAttention wa(16, 4, 2);
    convert_model(wa);

    auto input = createInput({2, 16, 16}, true);
    auto output = wa.forward(input, Tensor{});
    auto loss = sum(output);
    loss.backward();
    ASSERT_TRUE(input.grad().has_value());
    ASSERT_EQ(input.grad().value().shape()[0], 2);
}

TEST_P(WindowAttentionMultiDTypeTest, OutputFinite) {
    WindowAttention wa(32, 7, 4);
    convert_model(wa);

    auto input = createInput({2, 49, 32}, false);
    auto output = wa.forward(input, Tensor{});

    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i]));
    }
}

TEST_P(WindowAttentionMultiDTypeTest, DifferentWindowSizes) {
    for (int64_t ws : {4, 7}) {
        WindowAttention wa(16, ws, 2);
        convert_model(wa);

        auto input = createInput({1, ws * ws, 16}, false);
        auto output = wa.forward(input, Tensor{});
        EXPECT_EQ(output.tensor().shape()[1], ws * ws);
        EXPECT_EQ(output.tensor().shape()[2], 16);
    }
}

TEST_P(WindowAttentionMultiDTypeTest, ParametersNonEmpty) {
    WindowAttention wa(32, 7, 4);
    convert_model(wa);

    auto params = wa.parameters();
    EXPECT_FALSE(params.empty());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(WindowAttentionMultiDTypeTest);
