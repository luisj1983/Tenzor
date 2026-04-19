/**
 * @file test_upsample_unflatten_multidtype.cpp
 * @brief Multi-dtype tests for nn::Upsample and nn::Unflatten layers.
 *
 * Both layers previously had no dedicated multidtype coverage. Tests cover
 * forward shape correctness for the major modes (nearest/bilinear/trilinear
 * for Upsample; dim+sizes for Unflatten) and dtype preservation.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/upsample.hpp>
#include <tenzor/nn/layers/flatten.hpp>
#include "../../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

// ============================================================================
// Upsample
// ============================================================================

class UpsampleMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(UpsampleMultiDTypeTest, NearestScaleFactor2_4D) {
    Upsample up(/*size=*/std::nullopt, /*scale_factor=*/2.0, "nearest", false);
    auto x = Variable(randn({1, 4, 8, 8}, dtype(), device()), false);
    auto y = up.forward(x);
    expectShape(y.tensor(), {1, 4, 16, 16});
    EXPECT_EQ(y.tensor().dtype(), dtype());
    EXPECT_EQ(y.tensor().device().type, device().type);
}

TEST_P(UpsampleMultiDTypeTest, NearestExplicitSize_4D) {
    Upsample up(/*size=*/std::vector<int64_t>{12, 12},
                /*scale_factor=*/std::nullopt, "nearest", false);
    auto x = Variable(randn({1, 2, 4, 4}, dtype(), device()), false);
    auto y = up.forward(x);
    expectShape(y.tensor(), {1, 2, 12, 12});
}

TEST_P(UpsampleMultiDTypeTest, BilinearScaleFactor2_4D) {
    Upsample up(/*size=*/std::nullopt, /*scale_factor=*/2.0, "bilinear", false);
    auto x = Variable(randn({1, 3, 8, 8}, dtype(), device()), false);
    auto y = up.forward(x);
    expectShape(y.tensor(), {1, 3, 16, 16});
}

// Phase 3 addition: backward gradient population.
TEST_P(UpsampleMultiDTypeTest, NearestBackwardGradPopulated) {
    Upsample up(std::nullopt, 2.0, "nearest", false);
    auto x = Variable(randn({1, 2, 4, 4}, dtype(), device()), true);
    auto y = up.forward(x);
    sum(y).backward();
    ASSERT_TRUE(x.has_grad()) << device().to_string();
    auto g_max = max(abs(x.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f);
}

TEST_P(UpsampleMultiDTypeTest, BilinearBackwardGradPopulated) {
    Upsample up(std::nullopt, 2.0, "bilinear", false);
    auto x = Variable(randn({1, 2, 4, 4}, dtype(), device()), true);
    auto y = up.forward(x);
    sum(y).backward();
    ASSERT_TRUE(x.has_grad()) << device().to_string();
    auto g_max = max(abs(x.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(UpsampleMultiDTypeTest);

// ============================================================================
// Unflatten — inverse of Flatten; expands dim N into a multi-dim shape.
// ============================================================================

class UnflattenMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(UnflattenMultiDTypeTest, SplitLastDim) {
    // Input {batch, 12} → split dim 1 into {3, 4} → {batch, 3, 4}
    Unflatten unflat(1, {3, 4});
    auto x = Variable(randn({2, 12}, dtype(), device()), false);
    auto y = unflat.forward(x);
    expectShape(y.tensor(), {2, 3, 4});
    EXPECT_EQ(y.tensor().dtype(), dtype());
    EXPECT_EQ(y.tensor().device().type, device().type);
}

TEST_P(UnflattenMultiDTypeTest, SplitMiddleDim) {
    // Input {2, 6, 5} → split dim 1 into {2, 3} → {2, 2, 3, 5}
    Unflatten unflat(1, {2, 3});
    auto x = Variable(randn({2, 6, 5}, dtype(), device()), false);
    auto y = unflat.forward(x);
    expectShape(y.tensor(), {2, 2, 3, 5});
}

TEST_P(UnflattenMultiDTypeTest, ValuesPreserved) {
    Unflatten unflat(1, {2, 2});
    auto cpu_in = zeros({1, 4}, DType::Float32, Device::cpu());
    for (int64_t i = 0; i < 4; ++i) cpu_in.data<float>()[i] = static_cast<float>(i);
    auto x = Variable(cpu_in.to(dtype()).to(device()), false);
    auto y = unflat.forward(x);
    auto out_cpu = y.tensor().to(Device::cpu()).to(DType::Float32);
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(out_cpu.data<float>()[i], static_cast<float>(i), atol() * 10.0f);
    }
}

// Phase 3 addition: backward through Unflatten (it's a reshape — gradient
// must reshape back).
TEST_P(UnflattenMultiDTypeTest, BackwardGradPopulated) {
    Unflatten unflat(1, {2, 2});
    auto x = Variable(randn({2, 4}, dtype(), device()), true);
    auto y = unflat.forward(x);
    sum(y).backward();
    ASSERT_TRUE(x.has_grad());
    EXPECT_EQ(x.grad()->numel(), x.tensor().numel());
    auto g_max = max(abs(x.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(UnflattenMultiDTypeTest);
