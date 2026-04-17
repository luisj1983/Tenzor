/**
 * @file test_zero_stage1_distributed_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for ZeRO Stage 1 distributed training
 *
 * Requires distributed environment (RANK, WORLD_SIZE). Skips on CPU and single-device.
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <memory>
#include <cstdlib>

using namespace tenzor;
using namespace tenzor::optim;
using namespace tenzor::testing;

class ZeROStage1DistributedMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        if (device().type == Device::Type::CPU) {
            GTEST_SKIP() << "Distributed ZeRO tests require non-CPU multi-device backend";
        }

        auto* rank_env = std::getenv("RANK");
        auto* world_size_env = std::getenv("WORLD_SIZE");

        if (!rank_env || !world_size_env) {
            GTEST_SKIP() << "Distributed environment not available (RANK, WORLD_SIZE not set)";
        }

        int world_size = std::atoi(world_size_env);
        if (world_size < 2) {
            GTEST_SKIP() << "Need at least 2 processes for distributed ZeRO tests";
        }
    }
};

TEST_P(ZeROStage1DistributedMultiDTypeTest, DistributedEnvironmentRequired) {
    // If we reach here, distributed environment is available
    auto* rank_env = std::getenv("RANK");
    auto* world_size_env = std::getenv("WORLD_SIZE");
    ASSERT_NE(rank_env, nullptr);
    ASSERT_NE(world_size_env, nullptr);

    int rank = std::atoi(rank_env);
    int world_size = std::atoi(world_size_env);
    EXPECT_GE(rank, 0);
    EXPECT_GE(world_size, 2);
}

TEST_P(ZeROStage1DistributedMultiDTypeTest, ConstructionWithProcessGroup) {
    auto* rank_env = std::getenv("RANK");
    auto* world_size_env = std::getenv("WORLD_SIZE");
    int rank = std::atoi(rank_env);
    int world_size = std::atoi(world_size_env);

    auto params = std::vector<std::shared_ptr<Variable>>();
    auto t = tenzor::ones({64, 64}, dtype(), device());
    params.push_back(std::make_shared<Variable>(t, true));

    auto adam = std::make_unique<Adam>(params, 1e-3);

    ZeROStage1Config config;
    config.world_size = world_size;
    config.rank = rank;
    config.offload_to_cpu = false;

    EXPECT_NO_THROW({
        ZeROStage1Optimizer optimizer(std::move(adam), config);
    });
}

TEST_P(ZeROStage1DistributedMultiDTypeTest, PartitioningWithMultipleRanks) {
    auto* rank_env = std::getenv("RANK");
    auto* world_size_env = std::getenv("WORLD_SIZE");
    int rank = std::atoi(rank_env);
    int world_size = std::atoi(world_size_env);

    auto t = tenzor::ones({128, 128}, dtype(), device());
    std::vector<std::shared_ptr<Variable>> params;
    params.push_back(std::make_shared<Variable>(t, true));

    auto adam = std::make_unique<Adam>(params, 1e-3);

    ZeROStage1Config config;
    config.world_size = world_size;
    config.rank = rank;

    ZeROStage1Optimizer optimizer(std::move(adam), config);
    EXPECT_GT(optimizer.local_param_count(), 0u);
}

TEST_P(ZeROStage1DistributedMultiDTypeTest, StepWithDistributedParams) {
    auto* rank_env = std::getenv("RANK");
    auto* world_size_env = std::getenv("WORLD_SIZE");
    int rank = std::atoi(rank_env);
    int world_size = std::atoi(world_size_env);

    auto t = tenzor::ones({32, 32}, dtype(), device());
    auto param = std::make_shared<Variable>(t, true);
    param->set_grad(tenzor::ones({32, 32}, dtype(), device()) * 0.1f);

    std::vector<std::shared_ptr<Variable>> params = {param};
    auto adam = std::make_unique<Adam>(params, 1e-3);

    ZeROStage1Config config;
    config.world_size = world_size;
    config.rank = rank;

    ZeROStage1Optimizer optimizer(std::move(adam), config);
    EXPECT_NO_THROW(optimizer.step());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ZeROStage1DistributedMultiDTypeTest);
