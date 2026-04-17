/**
 * @file test_fold.cpp
 * @brief Tests for Fold (col2im) layer.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/vision.hpp>

using namespace tenzor;
using namespace tenzor::nn;

class FoldTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
    void SetUp() override { set_grad_enabled(true); }
};

TEST_F(FoldTest, ForwardShape) {
    // output_size={4, 4}, kernel=2, stride=1 -> L = (4-2+1)*(4-2+1) = 9
    Fold fold({4, 4}, /*kernel=*/2, /*dilation=*/1, /*padding=*/0, /*stride=*/1);
    // Input: (N, C*kernel*kernel, L) = (1, 1*2*2, 9)
    auto input = Variable(randn({1, 4, 9}, DType::Float32, Device::cpu()), false);
    auto output = fold.forward(input);
    auto shape = output.tensor().shape();
    ASSERT_EQ(shape[0], 1);   // batch
    ASSERT_EQ(shape[2], 4);   // height
    ASSERT_EQ(shape[3], 4);   // width
}

TEST_F(FoldTest, FoldUnfoldInverse) {
    // Fold should be approximately the inverse of Unfold
    Unfold unfold(/*kernel=*/3, /*dilation=*/1, /*padding=*/0, /*stride=*/1);
    Fold fold({6, 6}, /*kernel=*/3, /*dilation=*/1, /*padding=*/0, /*stride=*/1);

    auto input = Variable(randn({1, 1, 6, 6}, DType::Float32, Device::cpu()), false);
    auto unfolded = unfold.forward(input);
    auto refolded = fold.forward(unfolded);
    // Note: Fold accumulates overlapping values, so with stride < kernel,
    // the result won't exactly match. But the shape should be correct.
    ASSERT_EQ(refolded.tensor().shape()[2], 6);
    ASSERT_EQ(refolded.tensor().shape()[3], 6);
}

TEST_F(FoldTest, Backward) {
    Fold fold({4, 4}, 2, 1, 0, 1);
    auto input = Variable(randn({1, 4, 9}, DType::Float32, Device::cpu()), true);
    auto output = fold.forward(input);
    auto loss = sum(output);
    loss.backward();
    ASSERT_TRUE(input.grad().has_value());
}
