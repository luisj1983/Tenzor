/**
 * @file test_pipeline_parallel.cpp
 * @brief Tests for pipeline parallelism stages and micro-batch splitting
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/pipeline_parallel.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/transform.hpp>
#include <memory>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::distributed;

// Global test environment for initialization
class PipelineTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const pipeline_env =
    ::testing::AddGlobalTestEnvironment(new PipelineTestEnvironment);

// ============================================================================
// Simple stage module for testing
// ============================================================================

class SimpleStageModule : public Module {
public:
    explicit SimpleStageModule(int64_t in_features, int64_t out_features) {
        linear_ = std::make_shared<Linear>(in_features, out_features);
        register_module("linear", linear_);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        return linear_->forward(input);
    }

private:
    std::shared_ptr<Linear> linear_;
};

// ============================================================================
// PipelineStage Tests
// ============================================================================

TEST(PipelineStageTest, Construction) {
    auto module = std::make_shared<SimpleStageModule>(10, 20);

    // Valid construction
    EXPECT_NO_THROW({
        PipelineStage stage(module, 0, 4);
    });

    EXPECT_NO_THROW({
        PipelineStage stage(module, 3, 4);
    });
}

TEST(PipelineStageTest, InvalidConstruction) {
    auto module = std::make_shared<SimpleStageModule>(10, 20);

    // Null module
    EXPECT_THROW({
        PipelineStage stage(nullptr, 0, 4);
    }, std::invalid_argument);

    // stage_id out of range
    EXPECT_THROW({
        PipelineStage stage(module, 4, 4);
    }, std::invalid_argument);

    EXPECT_THROW({
        PipelineStage stage(module, -1, 4);
    }, std::invalid_argument);

    // Invalid num_stages
    EXPECT_THROW({
        PipelineStage stage(module, 0, 0);
    }, std::invalid_argument);

    EXPECT_THROW({
        PipelineStage stage(module, 0, -1);
    }, std::invalid_argument);
}

TEST(PipelineStageTest, Accessors) {
    auto module = std::make_shared<SimpleStageModule>(10, 20);
    PipelineStage stage(module, 1, 4);

    EXPECT_EQ(stage.stage_id(), 1);
    EXPECT_EQ(stage.num_stages(), 4);
    EXPECT_FALSE(stage.is_first());
    EXPECT_FALSE(stage.is_last());
}

TEST(PipelineStageTest, FirstLastStage) {
    auto module = std::make_shared<SimpleStageModule>(10, 20);

    PipelineStage first(module, 0, 4);
    EXPECT_TRUE(first.is_first());
    EXPECT_FALSE(first.is_last());

    PipelineStage last(module, 3, 4);
    EXPECT_FALSE(last.is_first());
    EXPECT_TRUE(last.is_last());

    // Single stage is both first and last
    PipelineStage single(module, 0, 1);
    EXPECT_TRUE(single.is_first());
    EXPECT_TRUE(single.is_last());
}

TEST(PipelineStageTest, Forward) {
    auto module = std::make_shared<SimpleStageModule>(10, 20);
    PipelineStage stage(module, 0, 2);

    auto input = Variable(randn({4, 10}), /*requires_grad=*/true);
    auto output = stage.forward(input);

    // Output shape: [4, 20] (batch=4, out_features=20)
    EXPECT_EQ(output.tensor().shape()[0], 4);
    EXPECT_EQ(output.tensor().shape()[1], 20);
}

// ============================================================================
// Micro-batch splitting tests
// ============================================================================

TEST(MicrobatchSplitTest, EvenSplit) {
    // Test that chunk/split produces correct number of micro-batches
    auto tensor = randn({8, 10});  // batch_size=8
    auto chunks = split(tensor, /*split_size=*/2, /*dim=*/0);  // 4 chunks of 2

    EXPECT_EQ(chunks.size(), 4u);
    for (const auto& c : chunks) {
        EXPECT_EQ(c.shape()[0], 2);
        EXPECT_EQ(c.shape()[1], 10);
    }
}

TEST(MicrobatchSplitTest, UnevenSplit) {
    // batch_size=7, split_size=2 -> 3 chunks of 2 + 1 chunk of 1
    auto tensor = randn({7, 10});
    auto chunks = split(tensor, /*split_size=*/2, /*dim=*/0);

    EXPECT_EQ(chunks.size(), 4u);
    EXPECT_EQ(chunks[0].shape()[0], 2);
    EXPECT_EQ(chunks[1].shape()[0], 2);
    EXPECT_EQ(chunks[2].shape()[0], 2);
    EXPECT_EQ(chunks[3].shape()[0], 1);
}

TEST(MicrobatchSplitTest, ConcatReconstruction) {
    // Splitting and concatenating should give back the original tensor
    auto original = randn({8, 10});
    auto chunks = split(original, /*split_size=*/2, /*dim=*/0);

    std::vector<Tensor> chunk_vec(chunks.begin(), chunks.end());
    auto reconstructed = cat(chunk_vec, /*dim=*/0);

    EXPECT_EQ(reconstructed.shape()[0], 8);
    EXPECT_EQ(reconstructed.shape()[1], 10);
}

// ============================================================================
// GPipeSchedule unit tests (single-rank, single-stage sanity check)
// ============================================================================

TEST(GPipeScheduleTest, SingleStageSingleMicrobatch) {
    // With 1 stage and 1 micro-batch, GPipe should just forward through
    // the stage without any send/recv (since it is both first and last).
    // This requires a ProcessGroup, but for a single-stage pipeline we
    // can verify the PipelineStage + schedule types compile and link.

    auto module = std::make_shared<SimpleStageModule>(10, 5);
    PipelineStage stage(module, 0, 1);

    // Verify stage can forward
    auto input = Variable(randn({4, 10}), /*requires_grad=*/true);
    auto output = stage.forward(input);

    EXPECT_EQ(output.tensor().shape()[0], 4);
    EXPECT_EQ(output.tensor().shape()[1], 5);

    // Verify schedule types are constructible
    GPipeSchedule gpipe;
    OneFOneBSchedule ofob;
    (void)gpipe;
    (void)ofob;
}

// ============================================================================
// Pipeline stage chaining test (simulated, no real distributed)
// ============================================================================

TEST(PipelineParallelTest, TwoStageForwardChain) {
    // Simulate a 2-stage pipeline locally (no ProcessGroup needed):
    // stage0 forward -> stage1 forward
    auto module0 = std::make_shared<SimpleStageModule>(10, 20);
    auto module1 = std::make_shared<SimpleStageModule>(20, 5);

    PipelineStage stage0(module0, 0, 2);
    PipelineStage stage1(module1, 1, 2);

    auto input = Variable(randn({4, 10}), /*requires_grad=*/true);

    // Stage 0 forward
    auto act0 = stage0.forward(input);
    EXPECT_EQ(act0.tensor().shape()[0], 4);
    EXPECT_EQ(act0.tensor().shape()[1], 20);

    // Stage 1 forward (using stage 0's output as input)
    auto act1 = stage1.forward(act0);
    EXPECT_EQ(act1.tensor().shape()[0], 4);
    EXPECT_EQ(act1.tensor().shape()[1], 5);
}

TEST(PipelineParallelTest, TwoStageMicrobatchChain) {
    // Simulate a 2-stage pipeline with micro-batch splitting
    auto module0 = std::make_shared<SimpleStageModule>(10, 20);
    auto module1 = std::make_shared<SimpleStageModule>(20, 5);

    PipelineStage stage0(module0, 0, 2);
    PipelineStage stage1(module1, 1, 2);

    // Full batch: [8, 10], split into 4 micro-batches of [2, 10]
    auto full_input = randn({8, 10});
    auto micro_inputs = split(full_input, /*split_size=*/2, /*dim=*/0);

    std::vector<Variable> final_outputs;
    for (const auto& mb_tensor : micro_inputs) {
        Variable mb_input(mb_tensor, /*requires_grad=*/true);

        // Stage 0
        auto act0 = stage0.forward(mb_input);
        EXPECT_EQ(act0.tensor().shape()[0], 2);

        // Stage 1
        auto act1 = stage1.forward(act0);
        EXPECT_EQ(act1.tensor().shape()[0], 2);
        EXPECT_EQ(act1.tensor().shape()[1], 5);

        final_outputs.push_back(act1);
    }

    // Concatenate micro-batch outputs
    auto combined = autograd::cat(final_outputs, /*dim=*/0);
    EXPECT_EQ(combined.tensor().shape()[0], 8);
    EXPECT_EQ(combined.tensor().shape()[1], 5);
}
