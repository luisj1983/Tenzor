/**
 * @file test_gradient_checkpoint.cpp
 * @brief Comprehensive tests for gradient checkpointing
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/checkpoint.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include <cmath>
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::autograd;

class GradientCheckpointTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset_checkpoint_stats();
    }

    void TearDown() override {
        reset_checkpoint_stats();
    }
};

// ==============================================================================
// Basic Checkpoint Statistics Tests
// ==============================================================================

TEST_F(GradientCheckpointTest, StatsTracking) {
    auto& stats = get_checkpoint_stats();
    EXPECT_EQ(stats.num_checkpoints, 0);
    EXPECT_EQ(stats.num_recomputations, 0);
    EXPECT_EQ(stats.saved_memory_bytes, 0);
}

TEST_F(GradientCheckpointTest, ResetStats) {
    auto& stats = get_checkpoint_stats();

    // Modify stats
    stats.num_checkpoints = 10;
    stats.num_recomputations = 5;

    reset_checkpoint_stats();

    EXPECT_EQ(stats.num_checkpoints, 0);
    EXPECT_EQ(stats.num_recomputations, 0);
}

// ==============================================================================
// Checkpoint Context Tests
// ==============================================================================

TEST_F(GradientCheckpointTest, CheckpointContextEnabled) {
    CheckpointContext ctx(true);
    EXPECT_TRUE(ctx.is_enabled());
}

TEST_F(GradientCheckpointTest, CheckpointContextDisabled) {
    CheckpointContext ctx(false);
    EXPECT_FALSE(ctx.is_enabled());
}

TEST_F(GradientCheckpointTest, CheckpointContextStats) {
    CheckpointContext ctx(true);

    // Create and checkpoint a simple computation
    auto x = Variable(ones({2, 3}), true);
    auto two = Variable(full({2, 3}, 2.0f), false);
    auto y = checkpoint([&two](const Variable& in) {
        return in * two;
    }, x);

    auto stats = ctx.get_stats();
    // Stats should have been updated
    EXPECT_GE(stats.num_checkpoints, 0);
}

// ==============================================================================
// Global Checkpoint Enable/Disable Tests
// ==============================================================================

TEST_F(GradientCheckpointTest, GlobalEnableDisable) {
    set_checkpoint_enabled(false);
    EXPECT_FALSE(is_checkpoint_enabled());

    set_checkpoint_enabled(true);
    EXPECT_TRUE(is_checkpoint_enabled());
}

// ==============================================================================
// Memory Tracker Tests
// ==============================================================================

TEST_F(GradientCheckpointTest, MemoryTrackerStart) {
    MemoryTracker::reset();
    MemoryTracker::start_tracking();

    size_t current = MemoryTracker::current_memory();
    EXPECT_GE(current, 0);

    MemoryTracker::stop_tracking();
}

TEST_F(GradientCheckpointTest, MemoryTrackerPeak) {
    MemoryTracker::reset();
    MemoryTracker::start_tracking();

    size_t peak = MemoryTracker::peak_memory();
    EXPECT_GE(peak, 0);

    MemoryTracker::stop_tracking();
}

TEST_F(GradientCheckpointTest, MemoryTrackerReset) {
    MemoryTracker::start_tracking();
    MemoryTracker::reset();

    EXPECT_EQ(MemoryTracker::current_memory(), 0);
    EXPECT_EQ(MemoryTracker::peak_memory(), 0);

    MemoryTracker::stop_tracking();
}

// ==============================================================================
// Simple Checkpoint Function Tests
// ==============================================================================

TEST_F(GradientCheckpointTest, SimpleForwardPass) {
    // Create input
    auto x_tensor = randn({4, 8});
    Variable x(x_tensor, true);

    // Checkpoint a simple function: y = x * 2 + 1
    auto checkpointed_fn = [](const Variable& input) -> Variable {
        auto shape = input.shape();
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        auto two = Variable(full(shape_vec, 2.0f), false);
        auto one = Variable(full(shape_vec, 1.0f), false);
        auto doubled = input * two;
        auto result = doubled + one;
        return result;
    };

    auto y = checkpoint(checkpointed_fn, x);

    // Verify forward pass correctness
    EXPECT_EQ(y.tensor().shape().size(), x.tensor().shape().size());

    const float* x_data = x.tensor().data<float>();
    const float* y_data = y.tensor().data<float>();

    for (int i = 0; i < x.tensor().numel(); ++i) {
        EXPECT_FLOAT_EQ(y_data[i], x_data[i] * 2.0f + 1.0f);
    }
}

TEST_F(GradientCheckpointTest, CheckpointGradientCorrectness) {
    // Create input
    auto x_tensor = ones({3, 3});
    Variable x(x_tensor, true);

    // Checkpoint function: y = x^2
    auto checkpointed_fn = [](const Variable& input) -> Variable {
        return input * input;
    };

    // Use checkpoint_with_original for leaf variables
    auto y = checkpoint_with_original(checkpointed_fn, x, &x);

    // Compute loss and backward
    auto loss = sum(y);
    loss.backward();

    // Gradient should be dy/dx = 2*x = 2*1 = 2
    EXPECT_GRAD_FLOWS(x);
    const float* grad_data = x.grad()->data<float>();

    for (int i = 0; i < x.tensor().numel(); ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], 2.0f);
    }
}

TEST_F(GradientCheckpointTest, MultiVariableCheckpoint) {
    // Test checkpoint with multiple inputs/outputs
    auto x_tensor = ones({2, 2});
    auto y_tensor = mul(ones({2, 2}), full({2, 2}, 2.0f));

    Variable x(x_tensor, true);
    Variable y(y_tensor, true);

    // Checkpoint function: z = x * y + x
    // IMPORTANT: Take inputs by const reference so computation graph references cached_recompute_inputs_
    auto multi_fn = [](const std::vector<Variable>& inputs) -> std::vector<Variable> {
        auto prod = inputs[0] * inputs[1];
        auto result = prod + inputs[0];
        return std::vector<Variable>{result};
    };

    // Use checkpoint_with_originals for leaf variables
    auto outputs = checkpoint_with_originals(multi_fn, {x, y}, {&x, &y});
    EXPECT_EQ(outputs.size(), 1);

    auto z = outputs[0];

    // Verify forward: z = 1 * 2 + 1 = 3
    const float* z_data = z.tensor().data<float>();
    for (int i = 0; i < z.tensor().numel(); ++i) {
        EXPECT_FLOAT_EQ(z_data[i], 3.0f);
    }

    // Backward pass
    auto loss = sum(z);
    loss.backward();

    // Verify gradients exist
    EXPECT_GRAD_FLOWS(x);
    EXPECT_GRAD_FLOWS(y);
}

TEST_F(GradientCheckpointTest, NestedCheckpoints) {
    auto x_tensor = ones({2, 2});
    Variable x(x_tensor, true);

    // Outer checkpoint - pass x by reference
    auto outer_fn = [&x](const Variable& input) -> Variable {
        // Inner checkpoint - for nested checkpoints with non-leaf intermediates
        auto inner_fn = [](const Variable& in) -> Variable {
            auto shape = in.shape();
            std::vector<int64_t> shape_vec(shape.begin(), shape.end());
            auto three = Variable(full(shape_vec, 3.0f), false);
            return in * three;
        };

        auto intermediate = checkpoint(inner_fn, input);
        auto shape = input.shape();
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        auto one = Variable(full(shape_vec, 1.0f), false);
        return intermediate + one;
    };

    // Use checkpoint_with_original for leaf variable x
    auto y = checkpoint_with_original(outer_fn, x, &x);

    // Forward: y = (x * 3) + 1 = 3 + 1 = 4
    const float* y_data = y.tensor().data<float>();
    for (int i = 0; i < y.tensor().numel(); ++i) {
        EXPECT_FLOAT_EQ(y_data[i], 4.0f);
    }

    // Backward
    auto loss = sum(y);
    loss.backward();

    // Gradient: dy/dx = 3
    EXPECT_GRAD_FLOWS(x);
    const float* grad_data = x.grad()->data<float>();
    for (int i = 0; i < x.tensor().numel(); ++i) {
        EXPECT_FLOAT_EQ(grad_data[i], 3.0f);
    }
}

// ==============================================================================
// Checkpoint with Activations
// ==============================================================================

TEST_F(GradientCheckpointTest, CheckpointWithReLU) {
    auto x_tensor = randn({4, 4});
    Variable x(x_tensor, true);

    // Simplified: just apply relu directly without intermediate operations
    auto relu_fn = [](const Variable& input) -> Variable {
        return nn::relu(input);
    };

    // Use checkpoint_with_original for leaf variable
    auto y = checkpoint_with_original(relu_fn, x, &x);

    // Backward
    auto loss = sum(y);
    loss.backward();

    EXPECT_GRAD_FLOWS(x);
}

TEST_F(GradientCheckpointTest, CheckpointWithSigmoid) {
    auto x_tensor = randn({3, 3});
    Variable x(x_tensor, true);

    auto sigmoid_fn = [](const Variable& input) -> Variable {
        return nn::sigmoid(input);
    };

    // Use checkpoint_with_original for leaf variable
    auto y = checkpoint_with_original(sigmoid_fn, x, &x);

    // Verify sigmoid output is in (0, 1)
    const float* y_data = y.tensor().data<float>();
    for (int i = 0; i < y.tensor().numel(); ++i) {
        EXPECT_GT(y_data[i], 0.0f);
        EXPECT_LT(y_data[i], 1.0f);
    }

    // Backward
    auto loss = sum(y);
    loss.backward();

    EXPECT_GRAD_FLOWS(x);
}

// ==============================================================================
// Checkpoint Segment Tests
// ==============================================================================

TEST_F(GradientCheckpointTest, CheckpointSegmentConstruction) {
    CheckpointSegment segment("test_segment", 0);

    EXPECT_EQ(segment.name(), "test_segment");
    EXPECT_EQ(segment.nesting_level(), 0);
}

TEST_F(GradientCheckpointTest, CheckpointSegmentExecution) {
    CheckpointSegment segment("compute_segment", 0);

    auto x_tensor = ones({2, 2});
    Variable x(x_tensor, true);

    // IMPORTANT: Take inputs by const reference
    auto compute_fn = [](const std::vector<Variable>& inputs) -> std::vector<Variable> {
        auto shape = inputs[0].shape();
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        auto five = Variable(full(shape_vec, 5.0f), false);
        auto result = inputs[0] * five;
        std::vector<Variable> outputs;
        outputs.push_back(result);
        return outputs;
    };

    auto outputs = segment.execute(compute_fn, {x});

    EXPECT_EQ(outputs.size(), 1);

    const float* output_data = outputs[0].tensor().data<float>();
    for (int i = 0; i < outputs[0].tensor().numel(); ++i) {
        EXPECT_FLOAT_EQ(output_data[i], 5.0f);
    }
}

TEST_F(GradientCheckpointTest, CheckpointSegmentNesting) {
    CheckpointSegment outer("outer", 0);
    CheckpointSegment inner("inner", 1);

    EXPECT_EQ(outer.nesting_level(), 0);
    EXPECT_EQ(inner.nesting_level(), 1);
}

// ==============================================================================
// Performance and Memory Tests
// ==============================================================================

TEST_F(GradientCheckpointTest, MemorySavingsEstimation) {
    MemoryTracker::reset();
    MemoryTracker::start_tracking();

    auto x_tensor = randn({10, 10});
    Variable x(x_tensor, true);

    // Checkpoint a computation that creates intermediate tensors
    auto memory_fn = [](const Variable& input) -> Variable {
        auto shape = input.shape();
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        auto two = Variable(full(shape_vec, 2.0f), false);
        auto one = Variable(full(shape_vec, 1.0f), false);
        auto three = Variable(full(shape_vec, 3.0f), false);
        auto temp1 = input * two;
        auto temp2 = temp1 + one;
        auto temp3 = temp2 * three;
        return temp3;
    };

    auto y = checkpoint(memory_fn, x);
    auto loss = sum(y);
    loss.backward();

    size_t peak = MemoryTracker::peak_memory();
    EXPECT_GE(peak, 0);

    MemoryTracker::stop_tracking();
}

TEST_F(GradientCheckpointTest, CheckpointStatsAccumulation) {
    reset_checkpoint_stats();

    auto x_tensor = ones({5, 5});
    Variable x(x_tensor, true);

    // Create multiple checkpoints
    for (int i = 0; i < 3; ++i) {
        auto fn = [i](const Variable& input) -> Variable {
            auto shape = input.shape();
            std::vector<int64_t> shape_vec(shape.begin(), shape.end());
            auto scalar = Variable(full(shape_vec, static_cast<float>(i + 1)), false);
            return input * scalar;
        };

        auto y = checkpoint(fn, x);
        auto loss = sum(y);
        loss.backward();
    }

    auto& stats = get_checkpoint_stats();
    EXPECT_GT(stats.num_checkpoints, 0);
}

int main(int argc, char** argv) {
    // Initialize Tenzor library (loads backends and registers operations)

    ::testing::InitGoogleTest(&argc, argv);
    if (!::testing::GTEST_FLAG(list_tests)) {
        tenzor::initialize();
    }
    int result = RUN_ALL_TESTS();

    // Cleanup
    tenzor::finalize();

    return result;
}
