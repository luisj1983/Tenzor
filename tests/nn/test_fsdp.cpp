/**
 * @file test_fsdp.cpp
 * @brief Tests for Fully Sharded Data Parallel (FSDP) functionality
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/fsdp.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/core/dtype.hpp>
#include <memory>
#include <cmath>
#include <numeric>

using namespace tenzor;
using namespace tenzor::nn;
namespace dist = tenzor::distributed;

// ============================================================================
// Test Environment
// ============================================================================

class FSDPTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const fsdp_env =
    ::testing::AddGlobalTestEnvironment(new FSDPTestEnvironment);

// ============================================================================
// Test Model
// ============================================================================

class FSDPTestModel : public Module {
public:
    FSDPTestModel(int64_t input_size, int64_t hidden_size, int64_t output_size) {
        fc1_ = std::make_shared<Linear>(input_size, hidden_size);
        fc2_ = std::make_shared<Linear>(hidden_size, output_size);

        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        auto x = fc1_->forward(input);
        x = nn::relu(x);
        x = fc2_->forward(x);
        return x;
    }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;
};

// ============================================================================
// FSDPUnit Tests
// ============================================================================

TEST(FSDPUnitTest, Construction) {
    auto model = std::make_shared<FSDPTestModel>(16, 32, 8);
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29600);

    dist::FSDPConfig config;
    config.strategy = dist::ShardingStrategy::FULL_SHARD;

    EXPECT_NO_THROW({
        dist::FSDPUnit unit(*model, *pg, config);
    });
}

TEST(FSDPUnitTest, ShardSize) {
    auto model = std::make_shared<Linear>(16, 32);
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29602);

    dist::FSDPConfig config;
    config.strategy = dist::ShardingStrategy::FULL_SHARD;

    dist::FSDPUnit unit(*model, *pg, config);

    // With world_size=1, shard_numel should equal total_numel
    EXPECT_EQ(unit.shard_numel(), unit.total_numel());
}

// Regression: high ranks whose shard offset lands at/beyond total_numel must
// report 0 valid elements, not underflow the unsigned length subtraction (which
// previously produced a ~1.8e19-element OOB slice). Covers ceil-division
// padding and verifies the shards exactly tile [0, total_numel).
TEST(FSDPUnitTest, ShardRangeNoUnderflowForHighRanks) {
    struct Case { size_t total; int ws; };
    const Case cases[] = {
        {5, 4}, {5, 8}, {8, 4}, {7, 3}, {1, 4}, {0, 4}, {16, 1}, {1000, 7},
    };
    for (const auto& c : cases) {
        size_t covered = 0;
        for (int rank = 0; rank < c.ws; ++rank) {
            auto r = dist::FSDPUnit::compute_shard_range(c.total, c.ws, rank);
            // valid_numel never exceeds the padded shard length...
            EXPECT_LE(r.valid_numel, r.shard_numel) << "total=" << c.total
                << " ws=" << c.ws << " rank=" << rank;
            if (r.valid_numel == 0) {
                // An empty shard is entirely padding: its offset is at or beyond
                // the end (offset can legitimately exceed total_numel here).
                EXPECT_GE(r.shard_offset, c.total) << "empty shard must be all-padding";
            } else {
                // A non-empty window stays within [0, total_numel).
                EXPECT_LT(r.shard_offset, c.total) << "total=" << c.total
                    << " ws=" << c.ws << " rank=" << rank;
                EXPECT_LE(r.shard_offset + r.valid_numel, c.total) << "total=" << c.total
                    << " ws=" << c.ws << " rank=" << rank;
            }
            covered += r.valid_numel;
        }
        // Every real element is owned by exactly one rank.
        EXPECT_EQ(covered, c.total) << "total=" << c.total << " ws=" << c.ws;
    }
}

// ============================================================================
// FullyShardedDataParallel Tests
// ============================================================================

TEST(FSDPTest, Construction) {
    auto model = std::make_shared<FSDPTestModel>(16, 32, 8);
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29603);

    dist::FSDPConfig config;
    config.strategy = dist::ShardingStrategy::FULL_SHARD;
    config.auto_wrap_min_params = 10;

    EXPECT_NO_THROW({
        dist::FullyShardedDataParallel fsdp(*model, *pg, config);
    });
}

TEST(FSDPTest, ForwardPass) {
    auto model = std::make_shared<FSDPTestModel>(8, 16, 4);
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29604);

    dist::FSDPConfig config;
    config.strategy = dist::ShardingStrategy::NO_SHARD;

    dist::FullyShardedDataParallel fsdp(*model, *pg, config);

    Variable input(rand({2, 8}), true);
    Variable output = fsdp.forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 4);
}

TEST(FSDPTest, SingleProcessNumericalEquivalence) {
    auto model = std::make_shared<FSDPTestModel>(8, 16, 4);

    Variable input(rand({2, 8}), true);
    Variable direct_output = model->forward(input);

    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29605);

    dist::FSDPConfig config;
    config.strategy = dist::ShardingStrategy::NO_SHARD;

    dist::FullyShardedDataParallel fsdp(*model, *pg, config);
    Variable fsdp_output = fsdp.forward(input);

    auto direct_data = direct_output.tensor();
    auto fsdp_data = fsdp_output.tensor();

    EXPECT_EQ(direct_data.shape().size(), fsdp_data.shape().size());
    for (size_t i = 0; i < direct_data.shape().size(); ++i) {
        EXPECT_EQ(direct_data.shape()[i], fsdp_data.shape()[i]);
    }
}

TEST(FSDPTest, ShardingStrategies) {
    auto model = std::make_shared<FSDPTestModel>(8, 16, 4);
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29606);

    for (auto strategy : {dist::ShardingStrategy::FULL_SHARD,
                          dist::ShardingStrategy::SHARD_GRAD_OP,
                          dist::ShardingStrategy::NO_SHARD}) {
        dist::FSDPConfig config;
        config.strategy = strategy;

        EXPECT_NO_THROW({
            dist::FullyShardedDataParallel fsdp(*model, *pg, config);
        });
    }
}

// Regression: FULL_SHARD must write optimizer updates back into the persistent
// shard. Simulate an optimizer step (add 1.0 to the materialized full param),
// release, then re-gather: the update must survive. Without the writeback the
// gather rebuilds the full params from the stale shard and reverts the weights,
// so training makes zero progress. This reproduces even at world_size=1.
TEST(FSDPTest, FullShardWritebackPreservesUpdates) {
    auto model = std::make_shared<FSDPTestModel>(8, 16, 4);
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29650);
    dist::FSDPConfig config;
    config.strategy = dist::ShardingStrategy::FULL_SHARD;
    dist::FullyShardedDataParallel fsdp(*model, *pg, config);

    // Materialize the full params and snapshot the first weight.
    fsdp.summon_full_params();
    auto params = model->parameters();
    ASSERT_FALSE(params.empty());
    auto& p = params[0];
    Tensor before = p->tensor().to(Device::cpu()).contiguous().clone();

    // Simulate an optimizer step: add 1.0 to the full param in place.
    auto& t = p->tensor();
    std::vector<int64_t> tshape(t.shape().begin(), t.shape().end());
    add_(t, tenzor::full(tshape, 1.0, t.dtype(), t.device()));
    fsdp.release_full_params();

    // Re-gather: the +1 update must be preserved.
    fsdp.summon_full_params();
    Tensor after = model->parameters()[0]->tensor().to(Device::cpu()).contiguous();
    fsdp.release_full_params();

    ASSERT_EQ(after.numel(), before.numel());
    const float* b = before.data<float>();
    const float* a = after.data<float>();
    for (int64_t i = 0; i < after.numel(); ++i) {
        EXPECT_NEAR(a[i], b[i] + 1.0f, 1e-4f)
            << "FULL_SHARD writeback missing: gather reverted the update at " << i;
    }
}

TEST(FSDPTest, MemoryReduction) {
    auto model = std::make_shared<FSDPTestModel>(32, 64, 16);
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29607);

    dist::FSDPConfig config;
    config.strategy = dist::ShardingStrategy::FULL_SHARD;
    config.auto_wrap_min_params = 10;

    dist::FullyShardedDataParallel fsdp(*model, *pg, config);

    EXPECT_GT(fsdp.total_params(), 0u);

    size_t sharded_bytes = fsdp.sharded_param_bytes();
    EXPECT_GT(sharded_bytes, 0u);
}

TEST(FSDPTest, SummonFullParams) {
    auto model = std::make_shared<FSDPTestModel>(8, 16, 4);
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29608);

    dist::FSDPConfig config;
    config.strategy = dist::ShardingStrategy::FULL_SHARD;

    dist::FullyShardedDataParallel fsdp(*model, *pg, config);

    EXPECT_NO_THROW({ fsdp.summon_full_params(); });
    EXPECT_NO_THROW({ fsdp.release_full_params(); });
}

TEST(FSDPTest, CPUOffloadConfig) {
    auto model = std::make_shared<FSDPTestModel>(8, 16, 4);
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29609);

    dist::FSDPConfig config;
    config.strategy = dist::ShardingStrategy::FULL_SHARD;
    config.cpu_offload = true;

    EXPECT_NO_THROW({
        dist::FullyShardedDataParallel fsdp(*model, *pg, config);
    });
}

TEST(FSDPTest, AutoWrapPolicy) {
    auto model = std::make_shared<FSDPTestModel>(8, 16, 4);
    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29610);

    // High threshold: root becomes single unit
    dist::FSDPConfig config_high;
    config_high.auto_wrap_min_params = 1000000;

    dist::FullyShardedDataParallel fsdp_high(*model, *pg, config_high);
    size_t high_units = fsdp_high.units().size();

    // Low threshold: submodules get their own units
    dist::FSDPConfig config_low;
    config_low.auto_wrap_min_params = 10;

    dist::FullyShardedDataParallel fsdp_low(*model, *pg, config_low);
    size_t low_units = fsdp_low.units().size();

    EXPECT_GE(low_units, high_units);
}
