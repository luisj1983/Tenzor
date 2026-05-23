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

    // audit T.1: profiling must not perturb the optimizer step result. Each
    // parameter started at 0.01, grad=0.1, so after one Adam step the value
    // should match a CPU+F32 reference (closed-form Adam: 0.01 - lr * ~1.0).
    std::vector<std::shared_ptr<Variable>> ref_params;
    for (const auto& shape_pair : std::vector<std::vector<int64_t>>{{128, 256}, {256, 512}, {512, 128}}) {
        auto t = tenzor::full(shape_pair, 0.01, DType::Float32, Device::cpu());
        auto v = std::make_shared<Variable>(t, true);
        v->set_grad(tenzor::full(shape_pair, 0.1, DType::Float32, Device::cpu()));
        ref_params.push_back(v);
    }
    Adam ref_adam(ref_params, 1e-3);
    ref_adam.step();

    for (size_t i = 0; i < parameters_.size(); ++i) {
        expectTensorNear(parameters_[i]->tensor(), ref_params[i]->tensor(), atol_);
    }
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

    // audit T.1: assert profiling actually recorded the step (was:
    // EXPECT_TRUE(true) no-op). num_steps must have advanced and timings
    // must be non-negative.
    EXPECT_EQ(stats.num_steps, 1u) << "profiling did not record the step";
    EXPECT_GE(stats.total_step_time_ms, 0.0);
    EXPECT_GE(stats.compute_time_ms, 0.0);
    EXPECT_GE(stats.communication_time_ms, 0.0);
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

    auto stats_before = optimizer.get_profiling_stats();
    ASSERT_EQ(stats_before.num_steps, 1u);

    EXPECT_NO_THROW(optimizer.reset_profiling_stats());

    // audit T.1: after reset, num_steps must be zero (was: EXPECT_NO_THROW only).
    auto stats_after = optimizer.get_profiling_stats();
    EXPECT_EQ(stats_after.num_steps, 0u) << "reset_profiling_stats() did not clear stats";
    EXPECT_EQ(stats_after.total_step_time_ms, 0.0);
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
    // audit T.1: 3 step()s must produce num_steps == 3 (was:
    // EXPECT_TRUE(true) no-op).
    EXPECT_EQ(stats.num_steps, 3u);
    EXPECT_GE(stats.total_step_time_ms, 0.0);

    // Also assert the final param values match a CPU+F32 plain-Adam reference
    // after 3 steps with the same grad=0.1 per step (params reset between
    // each step by mutable_grad()->fill_).
    std::vector<std::shared_ptr<Variable>> ref_params;
    for (const auto& shape_pair : std::vector<std::vector<int64_t>>{{128, 256}, {256, 512}, {512, 128}}) {
        auto t = tenzor::full(shape_pair, 0.01, DType::Float32, Device::cpu());
        ref_params.push_back(std::make_shared<Variable>(t, true));
    }
    Adam ref_adam(ref_params, 1e-3);
    for (int i = 0; i < 3; ++i) {
        for (size_t k = 0; k < ref_params.size(); ++k) {
            ref_params[k]->set_grad(tenzor::full(
                {ref_params[k]->tensor().shape().begin(), ref_params[k]->tensor().shape().end()},
                0.1, DType::Float32, Device::cpu()));
        }
        ref_adam.step();
    }
    for (size_t i = 0; i < parameters_.size(); ++i) {
        expectTensorNear(parameters_[i]->tensor(), ref_params[i]->tensor(), atol_);
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ZeROProfilingMultiDTypeTest);
