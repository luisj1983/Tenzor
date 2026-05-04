/**
 * @file test_sequence_parallel.cpp
 * @brief C++ test for distributed::SequenceParallel (audit-2026-05-03 N4).
 *
 * Mirrors `tests/python/test_sequence_parallel.py` — single-rank smoke
 * coverage. The TP group degenerates to size 1 in a non-distributed
 * test environment, so each scatter/gather method becomes a shape-
 * preserving identity. This verifies the API surface compiles and the
 * single-process path is correct.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/sequence_parallel.hpp>
#include <tenzor/distributed/device_mesh.hpp>
#include <tenzor/ops/creation.hpp>
#include <memory>

using namespace tenzor;
using namespace tenzor::distributed;

class SequenceParallelTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

TEST_F(SequenceParallelTest, ConstructionWithSingleRankMesh) {
    // Single-rank TP mesh — degenerate but useful for API validation.
    auto mesh = std::make_shared<DeviceMesh>(
        Device::Type::CPU, std::vector<int64_t>{1}, std::vector<std::string>{"tp"});
    SequenceParallel sp(mesh, "tp");
    EXPECT_EQ(sp.tp_size(), 1);
    EXPECT_EQ(sp.tp_rank(), 0);
}

TEST_F(SequenceParallelTest, ScatterSequenceSingleRankIsIdentity) {
    auto mesh = std::make_shared<DeviceMesh>(
        Device::Type::CPU, std::vector<int64_t>{1}, std::vector<std::string>{"tp"});
    SequenceParallel sp(mesh, "tp");

    auto input = randn({2, 16, 8}, DType::Float32, Device::cpu());
    auto out = sp.scatter_sequence(input, /*seq_dim=*/1);

    ASSERT_EQ(out.shape().size(), 3u);
    EXPECT_EQ(out.shape()[0], 2);
    EXPECT_EQ(out.shape()[1], 16);  // single-rank: no scatter
    EXPECT_EQ(out.shape()[2], 8);
}

TEST_F(SequenceParallelTest, GatherSequenceSingleRankIsIdentity) {
    auto mesh = std::make_shared<DeviceMesh>(
        Device::Type::CPU, std::vector<int64_t>{1}, std::vector<std::string>{"tp"});
    SequenceParallel sp(mesh, "tp");

    auto input = randn({2, 16, 8}, DType::Float32, Device::cpu());
    auto out = sp.gather_sequence(input, /*seq_dim=*/1);

    ASSERT_EQ(out.shape().size(), 3u);
    EXPECT_EQ(out.shape()[1], 16);
}

TEST_F(SequenceParallelTest, PreAttentionGatherShape) {
    auto mesh = std::make_shared<DeviceMesh>(
        Device::Type::CPU, std::vector<int64_t>{1}, std::vector<std::string>{"tp"});
    SequenceParallel sp(mesh, "tp");

    auto input = randn({2, 16, 8}, DType::Float32, Device::cpu());
    auto out = sp.pre_attention_gather(input, /*seq_dim=*/1);
    EXPECT_EQ(out.shape()[1], 16);
}

TEST_F(SequenceParallelTest, PostAttentionScatterShape) {
    auto mesh = std::make_shared<DeviceMesh>(
        Device::Type::CPU, std::vector<int64_t>{1}, std::vector<std::string>{"tp"});
    SequenceParallel sp(mesh, "tp");

    auto input = randn({2, 16, 8}, DType::Float32, Device::cpu());
    auto out = sp.post_attention_scatter(input, /*seq_dim=*/1);
    EXPECT_EQ(out.shape()[1], 16);
}

TEST_F(SequenceParallelTest, LocalSeqLenSingleRank) {
    auto mesh = std::make_shared<DeviceMesh>(
        Device::Type::CPU, std::vector<int64_t>{1}, std::vector<std::string>{"tp"});
    SequenceParallel sp(mesh, "tp");
    // Single-rank: local_seq_len == global_seq_len.
    EXPECT_EQ(sp.local_seq_len(64), 64);
    EXPECT_EQ(sp.local_seq_len(128), 128);
}
