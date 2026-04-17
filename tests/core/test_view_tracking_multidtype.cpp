/**
 * @file test_view_tracking_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for view tracking in Tensor
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/transform.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class ViewTrackingMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(ViewTrackingMultiDTypeTest, NonViewTensor) {
    auto t = tenzor::ones({3, 4}, dtype(), device());
    EXPECT_FALSE(t.is_view());
    EXPECT_EQ(t._view_base(), nullptr);
}

TEST_P(ViewTrackingMultiDTypeTest, ViewCreatesIsView) {
    auto t = tenzor::ones({3, 4}, dtype(), device());
    auto v = t.view({12});
    EXPECT_TRUE(v.is_view());
    EXPECT_NE(v._view_base(), nullptr);
}

TEST_P(ViewTrackingMultiDTypeTest, TransposeCreatesView) {
    auto t = tenzor::ones({3, 4}, dtype(), device());
    auto v = t.transpose(0, 1);
    EXPECT_TRUE(v.is_view());
    EXPECT_NE(v._view_base(), nullptr);
}

TEST_P(ViewTrackingMultiDTypeTest, SliceCreatesView) {
    auto t = tenzor::ones({10}, dtype(), device());
    auto v = t.slice(0, 2, 5);
    EXPECT_TRUE(v.is_view());
    EXPECT_NE(v._view_base(), nullptr);
}

TEST_P(ViewTrackingMultiDTypeTest, SqueezeCreatesView) {
    auto t = tenzor::ones({1, 3}, dtype(), device());
    auto v = t.squeeze(0);
    EXPECT_TRUE(v.is_view());
    EXPECT_NE(v._view_base(), nullptr);
}

TEST_P(ViewTrackingMultiDTypeTest, ChainedViewTracksOriginalBase) {
    auto t = tenzor::ones({3, 4}, dtype(), device());
    auto v1 = t.view({12});
    auto v2 = v1.view({4, 3});

    EXPECT_TRUE(v1.is_view());
    EXPECT_TRUE(v2.is_view());
    EXPECT_EQ(v1._view_base(), v2._view_base());
}

TEST_P(ViewTrackingMultiDTypeTest, CloneIsNotView) {
    auto t = tenzor::ones({3, 4}, dtype(), device());
    auto c = t.clone();
    EXPECT_FALSE(c.is_view());
}

TEST_P(ViewTrackingMultiDTypeTest, ViewSurvivesBaseDestruction) {
    Tensor view;
    {
        auto base = tenzor::ones({3, 4}, dtype(), device());
        view = base.view({12});
        EXPECT_TRUE(view.is_view());
        EXPECT_NE(view._view_base(), nullptr);
    }
    EXPECT_TRUE(view.is_view());
    EXPECT_NE(view._view_base(), nullptr);
    EXPECT_EQ(view.numel(), 12);

    // Verify data by moving to CPU + Float32 for access
    auto cpu_view = view.to(Device::cpu()).to(DType::Float32);
    const float* vdata = cpu_view.data<float>();
    EXPECT_FLOAT_EQ(vdata[0], 1.0f);
    EXPECT_FLOAT_EQ(vdata[11], 1.0f);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ViewTrackingMultiDTypeTest);
