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

TEST_F(ViewTrackingTest, ViewSurvivesBaseDestruction) {
    // Regression test: view_base_ must keep the base TensorImpl alive.
    // Before the intrusive_ptr fix, this was a use-after-free.
    Tensor view;
    {
        auto base = tenzor::ones({3, 4}, DType::Float32);
        view = base.view({12});
        EXPECT_TRUE(view.is_view());
        EXPECT_NE(view._view_base(), nullptr);
    }
    // base is now out of scope, but the view should still be valid
    EXPECT_TRUE(view.is_view());
    EXPECT_NE(view._view_base(), nullptr);
    EXPECT_EQ(view.numel(), 12);
    // Access data to ensure it's still alive (ASan would catch UAF)
    const float* vdata = view.data<float>();
    EXPECT_FLOAT_EQ(vdata[0], 1.0f);
    EXPECT_FLOAT_EQ(vdata[11], 1.0f);
}

TEST_F(ViewTrackingTest, ChainedViewSurvivesIntermediateDestruction) {
    // Ensure chained views survive when intermediate views are destroyed
    Tensor v2;
    {
        auto base = tenzor::ones({2, 3, 4}, DType::Float32);
        auto v1 = base.view({6, 4});
        v2 = v1.view({24});
        // v1 and base go out of scope
    }
    EXPECT_TRUE(v2.is_view());
    EXPECT_EQ(v2.numel(), 24);
    const float* v2data = v2.data<float>();
    EXPECT_FLOAT_EQ(v2data[0], 1.0f);
}

TEST_F(ViewTrackingTest, TransposeViewSurvivesBaseDestruction) {
    Tensor view;
    {
        auto base = tenzor::ones({3, 4}, DType::Float32);
        view = base.transpose(0, 1);
    }
    EXPECT_TRUE(view.is_view());
    EXPECT_EQ(view.shape()[0], 4);
    EXPECT_EQ(view.shape()[1], 3);
    const float* tdata = view.data<float>();
    EXPECT_FLOAT_EQ(tdata[0], 1.0f);
}

TEST_F(ViewTrackingTest, SliceViewSurvivesBaseDestruction) {
    Tensor view;
    {
        auto base = tenzor::ones({10}, DType::Float32);
        view = base.slice(0, 2, 5);
    }
    EXPECT_TRUE(view.is_view());
    EXPECT_EQ(view.numel(), 3);
    const float* sdata = view.data<float>();
    EXPECT_FLOAT_EQ(sdata[0], 1.0f);
}
