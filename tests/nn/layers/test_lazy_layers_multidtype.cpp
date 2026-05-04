/**
 * @file test_lazy_layers_multidtype.cpp
 * @brief Multi-backend × multi-dtype tests for LazyConv1d/2d/3d and
 *        LazyLinear (audit-2026-05-03 N2).
 *
 * Each test verifies:
 *   1. `is_materialized()` is false pre-forward, parameters() empty.
 *   2. First forward triggers materialization with correct in_channels /
 *      in_features inferred from input shape.
 *   3. Backward gradient flow through the materialized internal layer.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/lazy_conv.hpp>
#include <tenzor/nn/layers/lazy_linear.hpp>
#include <tenzor/autograd/variable.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include "../../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class LazyLayersMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// LazyConv1d
// ============================================================================

TEST_P(LazyLayersMultiDTypeTest, LazyConv1d_MaterializeOnForward) {
    LazyConv1d layer(/*out_channels=*/8, /*kernel_size=*/3);
    EXPECT_FALSE(layer.is_materialized());
    EXPECT_TRUE(layer.parameters().empty());

    auto input = Variable(createRandn({2, 4, 16}), false);  // (N, C_in=4, L=16)
    auto out = layer.forward(input);
    EXPECT_TRUE(layer.is_materialized());
    EXPECT_FALSE(layer.parameters().empty());

    // Output shape should be (N, out_channels, L_out = L - kernel_size + 1)
    ASSERT_EQ(out.shape().size(), 3u);
    EXPECT_EQ(out.shape()[0], 2);
    EXPECT_EQ(out.shape()[1], 8);
    EXPECT_EQ(out.shape()[2], 14);
    expectDType(out.tensor());
}

TEST_P(LazyLayersMultiDTypeTest, LazyConv1d_BackwardGradFlow) {
    LazyConv1d layer(/*out_channels=*/4, /*kernel_size=*/3);
    auto input = Variable(createRandn({2, 3, 8}), true);
    auto out = layer.forward(input);
    auto loss = tenzor::sum(out);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

// ============================================================================
// LazyConv2d
// ============================================================================

TEST_P(LazyLayersMultiDTypeTest, LazyConv2d_MaterializeOnForward) {
    LazyConv2d layer(/*out_channels=*/8, /*kernel_size=*/3);
    EXPECT_FALSE(layer.is_materialized());
    EXPECT_TRUE(layer.parameters().empty());

    auto input = Variable(createRandn({2, 4, 8, 8}), false);
    auto out = layer.forward(input);
    EXPECT_TRUE(layer.is_materialized());
    EXPECT_FALSE(layer.parameters().empty());

    // 8 - 3 + 1 = 6
    ASSERT_EQ(out.shape().size(), 4u);
    EXPECT_EQ(out.shape()[0], 2);
    EXPECT_EQ(out.shape()[1], 8);
    EXPECT_EQ(out.shape()[2], 6);
    EXPECT_EQ(out.shape()[3], 6);
    expectDType(out.tensor());
}

TEST_P(LazyLayersMultiDTypeTest, LazyConv2d_BackwardGradFlow) {
    LazyConv2d layer(/*out_channels=*/4, /*kernel_size=*/3);
    auto input = Variable(createRandn({2, 3, 6, 6}), true);
    auto out = layer.forward(input);
    auto loss = tenzor::sum(out);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

// ============================================================================
// LazyConv3d
// ============================================================================

TEST_P(LazyLayersMultiDTypeTest, LazyConv3d_MaterializeOnForward) {
    LazyConv3d layer(/*out_channels=*/4, /*kernel_size=*/2);
    EXPECT_FALSE(layer.is_materialized());

    auto input = Variable(createRandn({1, 2, 4, 4, 4}), false);
    auto out = layer.forward(input);
    EXPECT_TRUE(layer.is_materialized());

    // 4 - 2 + 1 = 3 in each spatial dim
    ASSERT_EQ(out.shape().size(), 5u);
    EXPECT_EQ(out.shape()[0], 1);
    EXPECT_EQ(out.shape()[1], 4);
    expectDType(out.tensor());
}

TEST_P(LazyLayersMultiDTypeTest, LazyConv3d_BackwardGradFlow) {
    LazyConv3d layer(/*out_channels=*/2, /*kernel_size=*/2);
    auto input = Variable(createRandn({1, 2, 4, 4, 4}), true);
    auto out = layer.forward(input);
    auto loss = tenzor::sum(out);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

// ============================================================================
// LazyLinear
// ============================================================================

TEST_P(LazyLayersMultiDTypeTest, LazyLinear_MaterializeOnForward) {
    LazyLinear layer(/*out_features=*/16);
    EXPECT_TRUE(layer.parameters().empty());

    auto input = Variable(createRandn({4, 32}), false);
    auto out = layer.forward(input);
    EXPECT_FALSE(layer.parameters().empty());

    ASSERT_EQ(out.shape().size(), 2u);
    EXPECT_EQ(out.shape()[0], 4);
    EXPECT_EQ(out.shape()[1], 16);
    expectDType(out.tensor());
}

TEST_P(LazyLayersMultiDTypeTest, LazyLinear_BackwardGradFlow) {
    LazyLinear layer(/*out_features=*/8);
    auto input = Variable(createRandn({4, 16}), true);
    auto out = layer.forward(input);
    auto loss = tenzor::sum(out);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(LazyLayersMultiDTypeTest);
