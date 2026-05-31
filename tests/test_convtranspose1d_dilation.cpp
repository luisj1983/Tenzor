/**
 * @file test_convtranspose1d_dilation.cpp
 * @brief Audit F.17: ConvTranspose1d now accepts dilation != 1.
 *
 * Previous behaviour: the ONNX importer threw `dilation != 1 not
 * supported` and ConvTranspose1d's constructor took no dilation
 * parameter at all.  The fix threads `dilation` through the layer
 * (constructor → forward dispatch → ConvTranspose1dBackward) and
 * removes the importer guard.  This test pins three contracts:
 *
 *   1. ConvTranspose1d(dilation > 1) constructs and forward runs.
 *   2. Output length matches the closed-form spec
 *        L_out = (L_in-1)*stride - 2*pad + dilation*(k-1) + op + 1
 *   3. Backward flows through dilation > 1 with finite gradients.
 */

#include "backend_test_fixture.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/ops/reduction.hpp"

#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

namespace {

class ConvTranspose1dDilationTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

TEST_P(ConvTranspose1dDilationTest, ConstructsWithDilation) {
    EXPECT_NO_THROW({
        ConvTranspose1d layer(
            /*in_channels=*/4, /*out_channels=*/8,
            /*kernel_size=*/3, /*stride=*/1,
            /*padding=*/0, /*output_padding=*/0,
            /*groups=*/1, /*bias=*/true,
            /*dilation=*/2);
        layer.to(device);
    });
    EXPECT_NO_THROW({
        ConvTranspose1d layer(
            4, 8, 3, /*stride=*/2,
            /*padding=*/1, /*output_padding=*/0,
            1, true, /*dilation=*/3);
        layer.to(device);
    });
}

TEST_P(ConvTranspose1dDilationTest, RejectsInvalidDilation) {
    EXPECT_THROW({
        ConvTranspose1d layer(4, 8, 3, 1, 0, 0, 1, true, /*dilation=*/0);
    }, std::invalid_argument);
    EXPECT_THROW({
        ConvTranspose1d layer(4, 8, 3, 1, 0, 0, 1, true, /*dilation=*/-1);
    }, std::invalid_argument);
}

TEST_P(ConvTranspose1dDilationTest, OutputLengthMatchesSpec) {
    // L_out = (L_in - 1)*stride - 2*pad + dilation*(k-1) + op + 1
    // With L_in=10, stride=1, pad=0, k=3, dilation=2, op=0:
    //   L_out = 9 - 0 + 2*2 + 0 + 1 = 14
    ConvTranspose1d layer(
        /*in_channels=*/4, /*out_channels=*/8,
        /*kernel_size=*/3, /*stride=*/1,
        /*padding=*/0, /*output_padding=*/0,
        /*groups=*/1, /*bias=*/true,
        /*dilation=*/2);
    layer.to(device);

    auto input = randn({2, 4, 10}, DType::Float32, device);
    auto output = layer.forward(Variable(input, false));
    auto out_shape = output.shape();
    ASSERT_EQ(out_shape.size(), 3u);
    EXPECT_EQ(out_shape[0], 2);
    EXPECT_EQ(out_shape[1], 8);
    EXPECT_EQ(out_shape[2], 14) << "ConvTranspose1d dilation=2 output length";
}

TEST_P(ConvTranspose1dDilationTest, OutputLengthMatchesSpecWithStrideAndPad) {
    // L_in=8, stride=2, pad=1, k=3, dilation=2, op=0:
    //   L_out = (8-1)*2 - 2*1 + 2*(3-1) + 0 + 1 = 14 - 2 + 4 + 1 = 17
    ConvTranspose1d layer(
        /*in_channels=*/3, /*out_channels=*/5,
        /*kernel_size=*/3, /*stride=*/2,
        /*padding=*/1, /*output_padding=*/0,
        /*groups=*/1, /*bias=*/false,
        /*dilation=*/2);
    layer.to(device);

    auto input = randn({1, 3, 8}, DType::Float32, device);
    auto output = layer.forward(Variable(input, false));
    auto out_shape = output.shape();
    ASSERT_EQ(out_shape.size(), 3u);
    EXPECT_EQ(out_shape[0], 1);
    EXPECT_EQ(out_shape[1], 5);
    EXPECT_EQ(out_shape[2], 17) << "ConvTranspose1d dilation=2 stride=2 pad=1 output length";
}

TEST_P(ConvTranspose1dDilationTest, BackwardFlowsThroughDilation) {
    ConvTranspose1d layer(
        /*in_channels=*/2, /*out_channels=*/3,
        /*kernel_size=*/3, /*stride=*/1,
        /*padding=*/0, /*output_padding=*/0,
        /*groups=*/1, /*bias=*/true,
        /*dilation=*/2);
    layer.to(device);

    auto input_t = randn({1, 2, 6}, DType::Float32, device);
    Variable x(input_t, /*requires_grad=*/true);
    auto y = layer.forward(x);
    auto loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad());
    auto g = (*x.grad()).cpu();
    const auto* gp = g.data<float>();
    bool all_finite = true;
    bool any_nonzero = false;
    for (int64_t i = 0; i < g.numel(); ++i) {
        if (!std::isfinite(gp[i])) all_finite = false;
        if (std::abs(gp[i]) > 0.0f) any_nonzero = true;
    }
    EXPECT_TRUE(all_finite) << "input grads contain NaN/Inf with dilation=2";
    EXPECT_TRUE(any_nonzero) << "input grads must be non-zero (sum of strictly positive sensitivities)";

    auto weight_ptr = layer.get_parameter("weight");
    ASSERT_TRUE(weight_ptr != nullptr);
    ASSERT_TRUE(weight_ptr->has_grad());
    auto gw = (*weight_ptr->grad()).cpu();
    const auto* gwp = gw.data<float>();
    bool wfin = true;
    bool wnz = false;
    for (int64_t i = 0; i < gw.numel(); ++i) {
        if (!std::isfinite(gwp[i])) wfin = false;
        if (std::abs(gwp[i]) > 0.0f) wnz = true;
    }
    EXPECT_TRUE(wfin) << "weight grads contain NaN/Inf with dilation=2";
    EXPECT_TRUE(wnz) << "weight grads must be non-zero with dilation=2";
}

INSTANTIATE_BACKEND_TESTS(ConvTranspose1dDilationTest);

}  // namespace
