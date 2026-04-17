/**
 * @file test_shape_info_snapshot_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for Tensor::shape_info_snapshot()
 */

#define TENZOR_INTERNAL 1

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

using namespace tenzor;
using namespace tenzor::testing;

class ShapeInfoSnapshotMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(ShapeInfoSnapshotMultiDTypeTest, CapturesCurrentShapeAndStrides) {
    auto t = tenzor::zeros({3, 4}, dtype(), device());

    auto snap = t.shape_info_snapshot();
    ASSERT_TRUE(snap);
    ASSERT_EQ(snap->shape.size(), 2u);
    EXPECT_EQ(snap->shape[0], 3);
    EXPECT_EQ(snap->shape[1], 4);

    ASSERT_EQ(snap->strides.size(), 2u);
    EXPECT_EQ(snap->strides[0], 4);
    EXPECT_EQ(snap->strides[1], 1);

    EXPECT_EQ(snap->offset, 0);
}

TEST_P(ShapeInfoSnapshotMultiDTypeTest, SurvivesSourceTensorDestruction) {
    std::shared_ptr<const ShapeInfo> snap;
    {
        auto t = tenzor::zeros({5, 6}, dtype(), device());
        snap = t.shape_info_snapshot();
        ASSERT_TRUE(snap);
    }
    ASSERT_EQ(snap->shape.size(), 2u);
    EXPECT_EQ(snap->shape[0], 5);
    EXPECT_EQ(snap->shape[1], 6);
}

TEST_P(ShapeInfoSnapshotMultiDTypeTest, RepeatedCallsReturnCachedPointer) {
    auto t = tenzor::zeros({2, 2}, dtype(), device());
    auto s1 = t.shape_info_snapshot();
    auto s2 = t.shape_info_snapshot();
    EXPECT_EQ(s1.get(), s2.get())
        << "Repeated shape_info_snapshot() calls should return the "
           "same cached shared_ptr until a mutator invalidates it";
}

TEST_P(ShapeInfoSnapshotMultiDTypeTest, MutationInvalidatesCache) {
    auto t = tenzor::zeros({2, 3}, dtype(), device());
    auto s1 = t.shape_info_snapshot();
    ASSERT_TRUE(s1);
    EXPECT_EQ(s1->shape[0], 2);

    {
        auto& shape_ref = t.mutable_shape();
        shape_ref = {6};
        auto& strides_ref = t.mutable_strides();
        strides_ref = {1};
    }

    auto s2 = t.shape_info_snapshot();
    ASSERT_TRUE(s2);
    EXPECT_NE(s1.get(), s2.get())
        << "After mutation, shape_info_snapshot must return a fresh "
           "ShapeInfo, not the stale cached pointer";
    EXPECT_EQ(s2->shape.size(), 1u);
    EXPECT_EQ(s2->shape[0], 6);

    // Old snapshot still valid
    EXPECT_EQ(s1->shape.size(), 2u);
    EXPECT_EQ(s1->shape[0], 2);
    EXPECT_EQ(s1->shape[1], 3);
}

TEST_P(ShapeInfoSnapshotMultiDTypeTest, SnapshotIsThreadSafeForReaders) {
    auto t = tenzor::zeros({8, 8}, dtype(), device());

    constexpr int kNumThreads = 4;
    constexpr int kIterations = 200;
    std::atomic<int> errors{0};
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);

    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([&]() {
            for (int k = 0; k < kIterations; ++k) {
                auto snap = t.shape_info_snapshot();
                if (!snap || snap->shape.size() != 2 ||
                    snap->shape[0] != 8 || snap->shape[1] != 8) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(errors.load(), 0);
}

TEST_P(ShapeInfoSnapshotMultiDTypeTest, SnapshotCarriesOffset) {
    auto t = tenzor::zeros({10}, dtype(), device());
    t.set_offset(3);
    auto snap = t.shape_info_snapshot();
    ASSERT_TRUE(snap);
    EXPECT_EQ(snap->offset, 3);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ShapeInfoSnapshotMultiDTypeTest);
