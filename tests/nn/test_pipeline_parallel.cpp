/**
 * @file test_pipeline_parallel.cpp
 * @brief Tests for PipelineStage and pipeline schedules
 *
 * Recreated after the original orphan was deleted in test suite cleanup.
 * Verifies that PipelineStage wraps an nn::Module correctly and reports
 * its stage metadata. Schedule tests are minimal — full schedule execution
 * requires a process group setup.
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/distributed/pipeline_parallel.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "../grad_flow_helpers.hpp"  // W.26: EXPECT_GRAD_FLOWS

using namespace tenzor;
using namespace tenzor::distributed;

class PipelineStageTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }

    std::shared_ptr<nn::Linear> make_linear(int in, int out) {
        return std::make_shared<nn::Linear>(in, out);
    }
};

// ============================================================================
// PipelineStage construction and metadata
// ============================================================================

TEST_F(PipelineStageTest, ConstructFirstStage) {
    auto module = make_linear(8, 16);
    PipelineStage stage(module, /*stage_id=*/0, /*num_stages=*/4);
    EXPECT_EQ(stage.stage_id(), 0);
    EXPECT_EQ(stage.num_stages(), 4);
    EXPECT_TRUE(stage.is_first());
    EXPECT_FALSE(stage.is_last());
}

TEST_F(PipelineStageTest, ConstructLastStage) {
    auto module = make_linear(16, 8);
    PipelineStage stage(module, /*stage_id=*/3, /*num_stages=*/4);
    EXPECT_EQ(stage.stage_id(), 3);
    EXPECT_EQ(stage.num_stages(), 4);
    EXPECT_FALSE(stage.is_first());
    EXPECT_TRUE(stage.is_last());
}

TEST_F(PipelineStageTest, ConstructMiddleStage) {
    auto module = make_linear(16, 16);
    PipelineStage stage(module, /*stage_id=*/2, /*num_stages=*/4);
    EXPECT_FALSE(stage.is_first());
    EXPECT_FALSE(stage.is_last());
}

TEST_F(PipelineStageTest, SingleStagePipelineIsBothFirstAndLast) {
    auto module = make_linear(8, 8);
    PipelineStage stage(module, /*stage_id=*/0, /*num_stages=*/1);
    EXPECT_TRUE(stage.is_first());
    EXPECT_TRUE(stage.is_last());
}

// ============================================================================
// PipelineStage forward pass
// ============================================================================

TEST_F(PipelineStageTest, ForwardProducesExpectedShape) {
    auto module = make_linear(8, 16);
    PipelineStage stage(module, 0, 2);

    Variable input(tenzor::randn({4, 8}, DType::Float32, Device::cpu()), false);
    auto output = stage.forward(input);

    auto shape = output.tensor().shape();
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 16);
}

TEST_F(PipelineStageTest, ForwardPreservesBatchDimension) {
    auto module = make_linear(10, 5);
    PipelineStage stage(module, 1, 3);

    Variable input(tenzor::randn({7, 10}, DType::Float32, Device::cpu()), false);
    auto output = stage.forward(input);

    EXPECT_EQ(output.tensor().shape()[0], 7);
    EXPECT_EQ(output.tensor().shape()[1], 5);
}

TEST_F(PipelineStageTest, ForwardSupportsAutograd) {
    auto module = make_linear(4, 4);
    PipelineStage stage(module, 0, 2);

    Variable input(tenzor::randn({2, 4}, DType::Float32, Device::cpu()), true);
    auto output = stage.forward(input);

    auto grad = tenzor::ones(std::vector<int64_t>(output.shape().begin(), output.shape().end()),
                              DType::Float32, Device::cpu());
    EXPECT_NO_THROW(output.backward(grad));
    EXPECT_GRAD_FLOWS(input);  // W.26
}

TEST_F(PipelineStageTest, MultipleStagesShareInput) {
    // Stages with the same shape should be composable.
    auto m1 = make_linear(4, 4);
    auto m2 = make_linear(4, 4);
    PipelineStage s1(m1, 0, 2);
    PipelineStage s2(m2, 1, 2);

    Variable input(tenzor::randn({3, 4}, DType::Float32, Device::cpu()), false);
    auto out1 = s1.forward(input);
    auto out2 = s2.forward(out1);

    EXPECT_EQ(out2.tensor().shape()[0], 3);
    EXPECT_EQ(out2.tensor().shape()[1], 4);
}
