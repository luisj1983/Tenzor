/**
 * @file test_view_tracking.cpp
 * @brief Tests for view tracking in Tensor
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/transform.hpp>

using namespace tenzor;

class ViewTrackingTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

TEST_F(ViewTrackingTest, NonViewTensor) {
    auto t = tenzor::ones({3, 4}, DType::Float32);
    EXPECT_FALSE(t.is_view());
    EXPECT_EQ(t._view_base(), nullptr);
}

TEST_F(ViewTrackingTest, ViewCreatesIsView) {
    auto t = tenzor::ones({3, 4}, DType::Float32);
    auto v = t.view({12});
    EXPECT_TRUE(v.is_view());
    EXPECT_NE(v._view_base(), nullptr);
}

TEST_F(ViewTrackingTest, TransposeCreatesView) {
    auto t = tenzor::ones({3, 4}, DType::Float32);
    auto v = t.transpose(0, 1);
    EXPECT_TRUE(v.is_view());
    EXPECT_NE(v._view_base(), nullptr);
}

TEST_F(ViewTrackingTest, PermuteCreatesView) {
    auto t = tenzor::ones({2, 3, 4}, DType::Float32);
    auto v = t.permute({2, 0, 1});
    EXPECT_TRUE(v.is_view());
    EXPECT_NE(v._view_base(), nullptr);
}

TEST_F(ViewTrackingTest, SliceCreatesView) {
    auto t = tenzor::ones({10}, DType::Float32);
    auto v = t.slice(0, 2, 5);
    EXPECT_TRUE(v.is_view());
    EXPECT_NE(v._view_base(), nullptr);
}

TEST_F(ViewTrackingTest, SqueezeCreatesView) {
    auto t = tenzor::ones({1, 3}, DType::Float32);
    auto v = t.squeeze(0);
    EXPECT_TRUE(v.is_view());
    EXPECT_NE(v._view_base(), nullptr);
}

TEST_F(ViewTrackingTest, UnsqueezeCreatesView) {
    auto t = tenzor::ones({3}, DType::Float32);
    auto v = t.unsqueeze(0);
    EXPECT_TRUE(v.is_view());
    EXPECT_NE(v._view_base(), nullptr);
}

TEST_F(ViewTrackingTest, ChainedViewTracksOriginalBase) {
    auto t = tenzor::ones({3, 4}, DType::Float32);
    auto v1 = t.view({12});
    auto v2 = v1.view({4, 3});

    EXPECT_TRUE(v1.is_view());
    EXPECT_TRUE(v2.is_view());
    // Both views should trace back to the same base (t)
    EXPECT_EQ(v1._view_base(), v2._view_base());
}

TEST_F(ViewTrackingTest, CloneIsNotView) {
    auto t = tenzor::ones({3, 4}, DType::Float32);
    auto c = t.clone();
    EXPECT_FALSE(c.is_view());
}

TEST_F(ViewTrackingTest, DetachCreatesView) {
    auto t = tenzor::ones({3, 4}, DType::Float32);
    auto d = t.detach();
    EXPECT_TRUE(d.is_view());
}
