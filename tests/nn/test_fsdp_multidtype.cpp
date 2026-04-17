/**
 * @file test_fsdp_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for Fully Sharded Data Parallel (FSDP)
 *
 * These tests require multi-device setups. CPU and single-device backends are skipped.
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/fsdp.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>
#include <memory>

using namespace tenzor;
using namespace tenzor::nn;
namespace dist = tenzor::distributed;
using namespace tenzor::testing;

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

class FSDPMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        // FSDP requires multi-device; skip CPU and single-device backends
        if (device().type == Device::Type::CPU) {
            GTEST_SKIP() << "FSDP tests require non-CPU multi-device backend";
        }
    }
};

TEST_P(FSDPMultiDTypeTest, Construction) {
    auto model = std::make_shared<FSDPTestModel>(16, 32, 8);
    convert_model(model);

    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29650);

    dist::FSDPConfig config;
    config.strategy = dist::ShardingStrategy::FULL_SHARD;

    EXPECT_NO_THROW({
        dist::FSDPUnit unit(*model, *pg, config);
    });
}

TEST_P(FSDPMultiDTypeTest, ForwardPass) {
    auto model = std::make_shared<FSDPTestModel>(8, 16, 4);
    convert_model(model);

    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29651);

    dist::FSDPConfig config;
    config.strategy = dist::ShardingStrategy::NO_SHARD;

    dist::FullyShardedDataParallel fsdp(*model, *pg, config);

    auto input = createInput({2, 8}, true);
    Variable output = fsdp.forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 4);
}

TEST_P(FSDPMultiDTypeTest, ShardingStrategies) {
    auto model = std::make_shared<FSDPTestModel>(8, 16, 4);
    convert_model(model);

    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29652);

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

TEST_P(FSDPMultiDTypeTest, ShardSize) {
    auto model = std::make_shared<Linear>(16, 32);
    convert_model(model);

    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29653);

    dist::FSDPConfig config;
    config.strategy = dist::ShardingStrategy::FULL_SHARD;

    dist::FSDPUnit unit(*model, *pg, config);
    EXPECT_EQ(unit.shard_numel(), unit.total_numel());
}

TEST_P(FSDPMultiDTypeTest, OutputFinite) {
    auto model = std::make_shared<FSDPTestModel>(8, 16, 4);
    convert_model(model);

    auto pg = dist::ProcessGroup::create_process_group(
        dist::Backend::GLOO, 0, 1, "localhost", 29654);

    dist::FSDPConfig config;
    config.strategy = dist::ShardingStrategy::NO_SHARD;

    dist::FullyShardedDataParallel fsdp(*model, *pg, config);

    auto input = createInput({2, 8}, false);
    auto output = fsdp.forward(input);

    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i]));
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(FSDPMultiDTypeTest);
