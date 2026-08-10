/**
 * @file test_linear_reshape_integration_multidtype.cpp
 * @brief Multi-dtype / multi-backend companion to test_linear_reshape_integration.cpp.
 *
 * The plain file (BackendTest, Float32-only) exercises a Linear layer fed by
 * reshape / permute chains and asserts gradient flow back to the original
 * input across backends. This companion re-runs the same integration surface
 * across {Float32, Float64, Float16} x {cpu, cuda, vulkan, oneapi, rocm, mps}
 * via MultiBackendDTypeTest, so a backend that drops the autograd graph or
 * only registers a Float32 reshape/Linear kernel is caught as a per-dtype
 * failure rather than hidden behind a single-dtype sweep.
 *
 * Inputs use randn (via createInput) rather than ones so the gradient-flow
 * and non-zero-gradient assertions are non-degenerate.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <memory>
#include "multi_backend_dtype_fixture.hpp"
#include "grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class LinearReshapeIntegrationMultiDType : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
        set_grad_enabled(true);
    }
};

TEST_P(LinearReshapeIntegrationMultiDType, LinearWithReshapeInput) {
    auto linear = std::make_shared<Linear>(4, 3);
    convert_model(linear);

    // Create input with shape {2, 4} and reshape to {8} then back to {2, 4}.
    Variable x = createInput({2, 4}, true);
    auto x_flat = reshape(x, {8});
    auto x_reshaped = reshape(x_flat, {2, 4});

    auto output = linear->forward(x_reshaped);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);

    auto loss = sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(x);
    auto grad = x.grad().value();
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 4);

    auto params = linear->parameters();
    ASSERT_GE(params.size(), 2);  // At least weight and bias
    for (auto& param : params) {
        ASSERT_TRUE(param->grad().has_value());
    }
}

TEST_P(LinearReshapeIntegrationMultiDType, LinearWithPermuteInput) {
    auto linear = std::make_shared<Linear>(4, 3);
    convert_model(linear);

    // Create input with shape {2, 3, 4}, permute to {2, 4, 3}, reshape to {8, 3}.
    Variable x = createInput({2, 3, 4}, true);
    auto x_permuted = permute(x, {0, 2, 1});  // {2, 4, 3}
    auto x_reshaped = reshape(x_permuted, {8, 3});

    auto linear2 = std::make_shared<Linear>(3, 5);
    convert_model(linear2);
    auto output = linear2->forward(x_reshaped);

    EXPECT_EQ(output.shape()[0], 8);
    EXPECT_EQ(output.shape()[1], 5);

    auto loss = sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(x);
    auto grad = x.grad().value();
    EXPECT_EQ(grad.shape()[0], 2);
    EXPECT_EQ(grad.shape()[1], 3);
    EXPECT_EQ(grad.shape()[2], 4);
}

TEST_P(LinearReshapeIntegrationMultiDType, MultipleReshapeOps) {
    // Chain of reshape operations feeding a Linear layer.
    Variable x = createInput({6}, true);
    auto y1 = reshape(x, {2, 3});
    auto y2 = reshape(y1, {3, 2});
    auto y3 = reshape(y2, {6});

    auto linear = std::make_shared<Linear>(6, 4);
    convert_model(linear);
    auto output = linear->forward(y3);

    auto loss = sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(x);
    auto grad = x.grad().value();
    EXPECT_EQ(grad.shape()[0], 6);

    // Gradient should be non-zero (randn input + random Linear weight).
    auto grad_cpu = grad.to(Device::cpu()).to(DType::Float32).contiguous();
    auto grad_data = grad_cpu.data<float>();
    bool has_nonzero = false;
    for (int64_t i = 0; i < grad_cpu.numel(); ++i) {
        if (grad_data[i] != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(LinearReshapeIntegrationMultiDType);