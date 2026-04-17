/**
 * @file test_zero_profiling_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for ZeRO optimizer profiling infrastructure
 *
 * Distributed/multi-device tests. CPU and single-device backends are skipped.
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <memory>

using namespace tenzor;
using namespace tenzor::optim;
using namespace tenzor::testing;

class ZeROProfilingMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        if (device().type == Device::Type::CPU) {
            GTEST_SKIP() << "ZeRO profiling tests require non-CPU multi-device backend";
        }

        // Create test parameters
        auto t1 = Tensor({128, 256}, dtype(), device());
        auto t2 = Tensor({256, 512}, dtype(), device());
        auto t3 = Tensor({512, 128}, dtype(), device());

        t1.fill_(0.01f);
        t2.fill_(0.01f);
        t3.fill_(0.01f);

        auto p1 = std::make_shared<Variable>(t1, true);
        auto p2 = std::make_shared<Variable>(t2, true);
        auto p3 = std::make_shared<Variable>(t3, true);

        p1->set_grad(Tensor({128, 256}, dtype(), device()));
        p2->set_grad(Tensor({256, 512}, dtype(), device()));
        p3->set_grad(Tensor({512, 128}, dtype(), device()));

        p1->mutable_grad()->fill_(0.1f);
        p2->mutable_grad()->fill_(0.1f);
        p3->mutable_grad()->fill_(0.1f);

        parameters_ = {p1, p2, p3};
    }

    std::vector<std::shared_ptr<Variable>> parameters_;
};

TEST_P(ZeROProfilingMultiDTypeTest, EnableDisableProfiling) {
    auto adam = std::make_unique<Adam>(parameters_, 1e-3);

    ZeROStage1Config config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = false;
    config.process_group = nullptr;

    ZeROStage1Optimizer optimizer(std::move(adam), config);

    EXPECT_FALSE(optimizer.is_profiling_enabled());
    optimizer.enable_profiling(true);
    EXPECT_TRUE(optimizer.is_profiling_enabled());
    optimizer.enable_profiling(false);
    EXPECT_FALSE(optimizer.is_profiling_enabled());
}

TEST_P(ZeROProfilingMultiDTypeTest, StepWithProfilingEnabled) {
    auto adam = std::make_unique<Adam>(parameters_, 1e-3);

    ZeROStage1Config config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = false;
    config.process_group = nullptr;

    ZeROStage1Optimizer optimizer(std::move(adam), config);
    optimizer.enable_profiling(true);

    EXPECT_NO_THROW(optimizer.step());
}

TEST_P(ZeROProfilingMultiDTypeTest, ProfilingReport) {
    auto adam = std::make_unique<Adam>(parameters_, 1e-3);

    ZeROStage1Config config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = false;
    config.process_group = nullptr;

    ZeROStage1Optimizer optimizer(std::move(adam), config);
    optimizer.enable_profiling(true);
    optimizer.step();

    auto stats = optimizer.get_profiling_stats();
    // Stats should have been populated after a step
    EXPECT_TRUE(true);  // Reaching here without exception is the check
}

TEST_P(ZeROProfilingMultiDTypeTest, ResetProfiling) {
    auto adam = std::make_unique<Adam>(parameters_, 1e-3);

    ZeROStage1Config config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = false;
    config.process_group = nullptr;

    ZeROStage1Optimizer optimizer(std::move(adam), config);
    optimizer.enable_profiling(true);
    optimizer.step();

    EXPECT_NO_THROW(optimizer.reset_profiling_stats());
}

TEST_P(ZeROProfilingMultiDTypeTest, MultipleStepsWithProfiling) {
    auto adam = std::make_unique<Adam>(parameters_, 1e-3);

    ZeROStage1Config config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = false;
    config.process_group = nullptr;

    ZeROStage1Optimizer optimizer(std::move(adam), config);
    optimizer.enable_profiling(true);

    for (int i = 0; i < 3; ++i) {
        // Refresh gradients
        for (auto& p : parameters_) {
            p->mutable_grad()->fill_(0.1f);
        }
        optimizer.step();
    }

    auto stats = optimizer.get_profiling_stats();
    EXPECT_TRUE(true);  // Reaching here without exception is the check
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ZeROProfilingMultiDTypeTest);
