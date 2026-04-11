// Tests for Phase 3.1b: Tensor::shape_info_snapshot().
//
// Verifies that:
//   1. The snapshot captures the current shape/strides/offset.
//   2. The snapshot is independent of later mutations to the source.
//   3. Repeated calls return the cached pointer (same ptr identity).
//   4. A mutation invalidates the cache — the next snapshot rebuilds.
//   5. Snapshots survive destruction of the source tensor.
//   6. The snapshot is safe to read from multiple threads (smoke test).

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/ops/creation.hpp>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

namespace tenzor {
namespace {

class ShapeInfoSnapshotTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

TEST_F(ShapeInfoSnapshotTest, CapturesCurrentShapeAndStrides) {
    auto t = tenzor::zeros({3, 4}, DType::Float32, Device::cpu());

    auto snap = t.shape_info_snapshot();
    ASSERT_TRUE(snap);
    ASSERT_EQ(snap->shape.size(), 2u);
    EXPECT_EQ(snap->shape[0], 3);
    EXPECT_EQ(snap->shape[1], 4);

    ASSERT_EQ(snap->strides.size(), 2u);
    // Row-major 3x4: strides = {4, 1}.
    EXPECT_EQ(snap->strides[0], 4);
    EXPECT_EQ(snap->strides[1], 1);

    EXPECT_EQ(snap->offset, 0);
}

TEST_F(ShapeInfoSnapshotTest, SurvivesSourceTensorDestruction) {
    std::shared_ptr<const ShapeInfo> snap;
    {
        auto t = tenzor::zeros({5, 6}, DType::Float32, Device::cpu());
        snap = t.shape_info_snapshot();
        ASSERT_TRUE(snap);
    }
    // Source tensor is dead. Snapshot must still be readable — it owns
    // its own copies of the shape/strides vectors.
    ASSERT_EQ(snap->shape.size(), 2u);
    EXPECT_EQ(snap->shape[0], 5);
    EXPECT_EQ(snap->shape[1], 6);
}

TEST_F(ShapeInfoSnapshotTest, RepeatedCallsReturnCachedPointer) {
    auto t = tenzor::zeros({2, 2}, DType::Float32, Device::cpu());
    auto s1 = t.shape_info_snapshot();
    auto s2 = t.shape_info_snapshot();
    // Pointer identity — the cache should be reused.
    EXPECT_EQ(s1.get(), s2.get())
        << "Repeated shape_info_snapshot() calls should return the "
           "same cached shared_ptr until a mutator invalidates it";
}

TEST_F(ShapeInfoSnapshotTest, MutationInvalidatesCache) {
    auto t = tenzor::zeros({2, 3}, DType::Float32, Device::cpu());
    auto s1 = t.shape_info_snapshot();
    ASSERT_TRUE(s1);
    EXPECT_EQ(s1->shape[0], 2);

    // Trigger an in-place shape mutation via the internal API. This
    // must invalidate the cache so the next snapshot rebuilds from
    // the post-mutation state.
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

    // The old snapshot is still valid — it has its own copy of the data.
    EXPECT_EQ(s1->shape.size(), 2u);
    EXPECT_EQ(s1->shape[0], 2);
    EXPECT_EQ(s1->shape[1], 3);
}

TEST_F(ShapeInfoSnapshotTest, SnapshotIsThreadSafeForReaders) {
    // Smoke test: multiple reader threads can call shape_info_snapshot
    // concurrently without crashing, so long as no one is mutating.
    // This verifies the atomic cache doesn't tear under contended reads.
    auto t = tenzor::zeros({8, 8}, DType::Float32, Device::cpu());

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

TEST_F(ShapeInfoSnapshotTest, SnapshotCarriesOffset) {
    auto t = tenzor::zeros({10}, DType::Float32, Device::cpu());
    // Force a non-zero offset via the internal API.
    t.set_offset(3);
    auto snap = t.shape_info_snapshot();
    ASSERT_TRUE(snap);
    EXPECT_EQ(snap->offset, 3);
}

} // namespace
} // namespace tenzor
