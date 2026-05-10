/**
 * @file test_zero_partitioning.cpp
 * @brief Unit tests for ZeRO element-level partition layout.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/ops/creation.hpp>

using namespace tenzor;
using namespace tenzor::optim;

namespace {

auto make_params(const std::vector<std::vector<int64_t>>& shapes)
    -> std::vector<std::shared_ptr<Variable>> {
    std::vector<std::shared_ptr<Variable>> v;
    v.reserve(shapes.size());
    for (const auto& sh : shapes) {
        v.push_back(std::make_shared<Variable>(
            ones(sh, DType::Float32, Device::cpu()), true));
    }
    return v;
}

class ZeROPartitioningTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

}  // namespace

// Layout for a single rank — every param's whole numel falls in rank 0's slice.
TEST_F(ZeROPartitioningTest, SingleRankCoversEverything) {
    auto params = make_params({{4, 4}, {8}, {2, 3, 5}});  // 16 + 8 + 30 = 54 elements
    auto base = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config;
    config.world_size = 1;
    config.rank = 0;
    config.partitioning_mode = PartitioningMode::ElementLevel;

    ZeROStage1Optimizer opt(std::move(base), config);

    const auto& L = opt.test_partition_layout();
    ASSERT_EQ(L.params.size(), 3u);
    ASSERT_GE(L.rank_starts.size(), 2u);
    EXPECT_EQ(L.params[0].global_offset, 0);
    EXPECT_EQ(L.params[0].numel, 16);
    EXPECT_EQ(L.params[1].global_offset, 16);
    EXPECT_EQ(L.params[1].numel, 8);
    EXPECT_EQ(L.params[2].global_offset, 24);
    EXPECT_EQ(L.params[2].numel, 30);
    EXPECT_EQ(L.total_elements_padded, 54);
    EXPECT_EQ(L.rank_starts, (std::vector<int64_t>{0, 54}));
    EXPECT_EQ(L.rank_size(0), 54);
}

// Even split across 4 ranks of a 60-element model.
TEST_F(ZeROPartitioningTest, FourRanksEvenSplit) {
    auto params = make_params({{60}});  // single 60-element param
    auto base = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config;
    config.world_size = 4;
    config.rank = 0;
    config.partitioning_mode = PartitioningMode::ElementLevel;

    ZeROStage1Optimizer opt(std::move(base), config);

    const auto& L = opt.test_partition_layout();
    EXPECT_EQ(L.total_elements_padded, 60);
    ASSERT_EQ(L.rank_starts.size(), 5u);
    EXPECT_EQ(L.rank_starts, (std::vector<int64_t>{0, 15, 30, 45, 60}));
    for (int r = 0; r < 4; ++r) EXPECT_EQ(L.rank_size(r), 15);
}

// Padding when total_elements is not divisible by world_size.
TEST_F(ZeROPartitioningTest, UnevenSplitPadsToWorldSizeMultiple) {
    auto params = make_params({{17}});  // 17 elements; ceil(17/4)*4 == 20 padded
    auto base = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config;
    config.world_size = 4;
    config.rank = 0;
    config.partitioning_mode = PartitioningMode::ElementLevel;

    ZeROStage1Optimizer opt(std::move(base), config);

    const auto& L = opt.test_partition_layout();
    EXPECT_EQ(L.total_elements_padded, 20);
    ASSERT_EQ(L.rank_starts.size(), 5u);
    EXPECT_EQ(L.rank_starts, (std::vector<int64_t>{0, 5, 10, 15, 20}));
    for (int r = 0; r < 4; ++r) {
        EXPECT_EQ(L.rank_size(r), 5);
    }
}

// ParamLevel mode leaves layout empty.
TEST_F(ZeROPartitioningTest, ParamLevelModeNoLayout) {
    auto params = make_params({{4, 4}});
    auto base = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config;
    config.world_size = 1;
    config.rank = 0;
    // partitioning_mode default = ParamLevel

    ZeROStage1Optimizer opt(std::move(base), config);

    const auto& L = opt.test_partition_layout();
    EXPECT_TRUE(L.params.empty());
    EXPECT_TRUE(L.rank_starts.empty());
    EXPECT_EQ(L.total_elements_padded, 0);
}
