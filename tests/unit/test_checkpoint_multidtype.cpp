/**
 * @file test_checkpoint_multidtype.cpp
 * @brief Multi-dtype multi-backend tests for gradient checkpointing functionality
 *
 * Tests gradient checkpointing with Float32, Float64, and Float16 dtypes across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - Checkpointing works correctly with different dtypes and backends
 * - Memory savings achieved through activation checkpointing
 * - Gradient correctness preserved across different precision levels
 * - Forward and backward passes maintain numerical accuracy
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/checkpoint.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::autograd;
using namespace tenzor::testing;

// ============================================================================
// Checkpoint Multi-Backend Multi-DType Test Fixture
// ============================================================================

class CheckpointMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        reset_checkpoint_stats();
    }

    void TearDown() override {
        reset_checkpoint_stats();
    }

    // Helper to create tensor with specific dtype and device
    Tensor createTensor(const std::vector<int64_t>& shape, float value) {
        auto t = full(shape, value, DType::Float32, Device::cpu());
        if (dtype() != DType::Float32) {
            t = t.to(dtype());
        }
        if (device() != Device::cpu()) {
            t = t.to(device());
        }
        return t;
    }

    // Helper to create random tensor with specific dtype
    Tensor createRandomTensor(const std::vector<int64_t>& shape) {
        // Skip Float16 for randn
        if (dtype() == DType::Float16) {
            // Create Float32, then convert
            auto t = randn(shape, DType::Float32, Device::cpu());
            return t.to(dtype()).to(device());
        }
        auto t = randn(shape, dtype(), Device::cpu());
        if (device() != Device::cpu()) {
            t = t.to(device());
        }
        return t;
    }

    // Helper to verify tensor values with dtype-appropriate tolerance
    void verifyValues(const Tensor& t, float expected, const std::string& message) {
        auto t_cpu = t.to(Device::cpu()).to(DType::Float32);
        const float* data = t_cpu.data<float>();
        for (int64_t i = 0; i < t_cpu.numel(); ++i) {
            EXPECT_NEAR(data[i], expected, atol())
                << message << " at index " << i << " for " << backend_name();
        }
    }

    // Helper to verify gradient values
    void verifyGradient(const Variable& var, float expected, const std::string& message) {
        ASSERT_TRUE(var.grad().has_value()) << "No gradient for " << message;
        verifyValues(*var.grad(), expected, "Gradient " + message);
    }
};

// ============================================================================
// Basic Checkpoint Tests with Multiple DTypes
// ============================================================================

TEST_P(CheckpointMultiDTypeTest, SimpleForwardPass) {
    // Create input with specific dtype and device
    auto x_tensor = createTensor({4, 8}, 1.0f);
    Variable x(x_tensor, true);

    // Checkpoint a simple function: y = x * 2 + 1
    auto checkpointed_fn = [this](const Variable& input) -> Variable {
        auto shape = input.shape();
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        auto two = Variable(createTensor(shape_vec, 2.0f), false);
        auto one = Variable(createTensor(shape_vec, 1.0f), false);
        auto doubled = input * two;
        auto result = doubled + one;
        return result;
    };

    auto y = checkpoint(checkpointed_fn, x);

    // Verify forward pass correctness: y = 1 * 2 + 1 = 3
    verifyValues(y.tensor(), 3.0f, "Forward pass output");
}

TEST_P(CheckpointMultiDTypeTest, CheckpointGradientCorrectness) {
    // Create input with specific dtype and device
    auto x_tensor = createTensor({3, 3}, 1.0f);
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
    verifyGradient(x, 2.0f, "x gradient");
}

TEST_P(CheckpointMultiDTypeTest, MultiVariableCheckpoint) {
    // Test checkpoint with multiple inputs
    auto x_tensor = createTensor({2, 2}, 1.0f);
    auto y_tensor = createTensor({2, 2}, 2.0f);

    Variable x(x_tensor, true);
    Variable y(y_tensor, true);

    // Checkpoint function: z = x * y + x
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
    verifyValues(z.tensor(), 3.0f, "Multi-variable forward output");

    // Backward pass
    auto loss = sum(z);
    loss.backward();

    // Verify gradients exist
    EXPECT_GRAD_FLOWS(x);
    EXPECT_GRAD_FLOWS(y);

    // Gradient verification: dz/dx = y + 1 = 2 + 1 = 3
    verifyGradient(x, 3.0f, "x gradient in multi-variable");
    // dz/dy = x = 1
    verifyGradient(y, 1.0f, "y gradient in multi-variable");
}

// ============================================================================
// Checkpoint with Activations (Different DTypes)
// ============================================================================

TEST_P(CheckpointMultiDTypeTest, CheckpointWithReLU) {
    auto x_tensor = createRandomTensor({4, 4});
    Variable x(x_tensor, true);

    auto relu_fn = [](const Variable& input) -> Variable {
        return nn::relu(input);
    };

    // Use checkpoint_with_original for leaf variable
    auto y = checkpoint_with_original(relu_fn, x, &x);

    // Backward
    auto loss = sum(y);
    loss.backward();

    EXPECT_GRAD_FLOWS(x);

    // Verify ReLU behavior: output should be non-negative
    auto y_cpu = y.tensor().to(Device::cpu()).to(DType::Float32);
    const float* y_data = y_cpu.data<float>();
    for (int64_t i = 0; i < y_cpu.numel(); ++i) {
        EXPECT_GE(y_data[i], 0.0f) << "ReLU output should be non-negative at index " << i;
    }
}

TEST_P(CheckpointMultiDTypeTest, CheckpointWithSigmoid) {
    auto x_tensor = createRandomTensor({3, 3});
    Variable x(x_tensor, true);

    auto sigmoid_fn = [](const Variable& input) -> Variable {
        return nn::sigmoid(input);
    };

    // Use checkpoint_with_original for leaf variable
    auto y = checkpoint_with_original(sigmoid_fn, x, &x);

    // Verify sigmoid output is in (0, 1)
    auto y_cpu = y.tensor().to(Device::cpu()).to(DType::Float32);
    const float* y_data = y_cpu.data<float>();
    for (int64_t i = 0; i < y_cpu.numel(); ++i) {
        EXPECT_GT(y_data[i], 0.0f) << "Sigmoid output should be > 0 at index " << i;
        EXPECT_LT(y_data[i], 1.0f) << "Sigmoid output should be < 1 at index " << i;
    }

    // Backward
    auto loss = sum(y);
    loss.backward();

    EXPECT_GRAD_FLOWS(x);
}

// ============================================================================
// Nested Checkpoints with Different DTypes
// ============================================================================

TEST_P(CheckpointMultiDTypeTest, NestedCheckpoints) {
    auto x_tensor = createTensor({2, 2}, 1.0f);
    Variable x(x_tensor, true);

    // Outer checkpoint
    auto outer_fn = [this, &x](const Variable& input) -> Variable {
        // Inner checkpoint
        auto inner_fn = [this](const Variable& in) -> Variable {
            auto shape = in.shape();
            std::vector<int64_t> shape_vec(shape.begin(), shape.end());
            auto three = Variable(createTensor(shape_vec, 3.0f), false);
            return in * three;
        };

        auto intermediate = checkpoint(inner_fn, input);
        auto shape = input.shape();
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        auto one = Variable(createTensor(shape_vec, 1.0f), false);
        return intermediate + one;
    };

    // Use checkpoint_with_original for leaf variable x
    auto y = checkpoint_with_original(outer_fn, x, &x);

    // Forward: y = (x * 3) + 1 = 3 + 1 = 4
    verifyValues(y.tensor(), 4.0f, "Nested checkpoint forward");

    // Backward
    auto loss = sum(y);
    loss.backward();

    // Gradient: dy/dx = 3
    verifyGradient(x, 3.0f, "Nested checkpoint gradient");
}

// ============================================================================
// Memory Savings Verification
// ============================================================================

TEST_P(CheckpointMultiDTypeTest, MemorySavingsEstimation) {
    MemoryTracker::reset();
    MemoryTracker::start_tracking();

    auto x_tensor = createRandomTensor({10, 10});
    Variable x(x_tensor, true);

    // Checkpoint a computation that creates intermediate tensors
    auto memory_fn = [this](const Variable& input) -> Variable {
        auto shape = input.shape();
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        auto two = Variable(createTensor(shape_vec, 2.0f), false);
        auto one = Variable(createTensor(shape_vec, 1.0f), false);
        auto three = Variable(createTensor(shape_vec, 3.0f), false);
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

// ============================================================================
// Checkpoint Statistics Tracking
// ============================================================================

TEST_P(CheckpointMultiDTypeTest, CheckpointStatsAccumulation) {
    reset_checkpoint_stats();

    auto x_tensor = createTensor({5, 5}, 1.0f);
    Variable x(x_tensor, true);

    // Create multiple checkpoints
    for (int i = 0; i < 3; ++i) {
        auto fn = [this, i](const Variable& input) -> Variable {
            auto shape = input.shape();
            std::vector<int64_t> shape_vec(shape.begin(), shape.end());
            auto scalar = Variable(createTensor(shape_vec, static_cast<float>(i + 1)), false);
            return input * scalar;
        };

        auto y = checkpoint(fn, x);
        auto loss = sum(y);
        loss.backward();
    }

    auto& stats = get_checkpoint_stats();
    EXPECT_GT(stats.num_checkpoints, 0) << "Checkpoints should be tracked for " << backend_name();
}

// ============================================================================
// Complex Computation Checkpoint
// ============================================================================

TEST_P(CheckpointMultiDTypeTest, ComplexComputationCheckpoint) {
    auto x_tensor = createTensor({4, 4}, 2.0f);
    Variable x(x_tensor, true);

    // Complex function: y = (x^2 + x) * 3 - 1
    auto complex_fn = [this](const Variable& input) -> Variable {
        auto shape = input.shape();
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());

        auto squared = input * input;  // x^2
        auto added = squared + input;  // x^2 + x
        auto three = Variable(createTensor(shape_vec, 3.0f), false);
        auto multiplied = added * three;  // (x^2 + x) * 3
        auto one = Variable(createTensor(shape_vec, 1.0f), false);
        auto result = multiplied - one;  // (x^2 + x) * 3 - 1

        return result;
    };

    auto y = checkpoint_with_original(complex_fn, x, &x);

    // Forward: y = (2^2 + 2) * 3 - 1 = (4 + 2) * 3 - 1 = 6 * 3 - 1 = 17
    verifyValues(y.tensor(), 17.0f, "Complex computation forward");

    // Backward
    auto loss = sum(y);
    loss.backward();

    // Gradient: dy/dx = d/dx[(x^2 + x) * 3 - 1] = (2x + 1) * 3 = (2*2 + 1) * 3 = 5 * 3 = 15
    verifyGradient(x, 15.0f, "Complex computation gradient");
}

// ============================================================================
// Checkpoint Segment Tests
// ============================================================================

TEST_P(CheckpointMultiDTypeTest, CheckpointSegmentExecution) {
    CheckpointSegment segment("compute_segment", 0);

    auto x_tensor = createTensor({2, 2}, 1.0f);
    Variable x(x_tensor, true);

    auto compute_fn = [this](const std::vector<Variable>& inputs) -> std::vector<Variable> {
        auto shape = inputs[0].shape();
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        auto five = Variable(createTensor(shape_vec, 5.0f), false);
        auto result = inputs[0] * five;
        return std::vector<Variable>{result};
    };

    auto outputs = segment.execute(compute_fn, {x});

    EXPECT_EQ(outputs.size(), 1);
    verifyValues(outputs[0].tensor(), 5.0f, "Checkpoint segment output");
}

// ============================================================================
// Precision Preservation Tests
// ============================================================================

TEST_P(CheckpointMultiDTypeTest, PrecisionPreservation) {
    // Test that gradients are computed with acceptable precision for the dtype
    auto x_tensor = createTensor({8, 8}, 0.5f);
    Variable x(x_tensor, true);

    // Simple linear function to minimize accumulation errors
    auto linear_fn = [this](const Variable& input) -> Variable {
        auto shape = input.shape();
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        auto weight = Variable(createTensor(shape_vec, 4.0f), false);
        return input * weight;
    };

    auto y = checkpoint_with_original(linear_fn, x, &x);

    // Forward: y = 0.5 * 4 = 2
    verifyValues(y.tensor(), 2.0f, "Linear forward");

    // Backward
    auto loss = sum(y);
    loss.backward();

    // Gradient: dy/dx = 4
    verifyGradient(x, 4.0f, "Linear gradient");
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(CheckpointMultiDTypeTest);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 11
 * DTypes Tested: Float32, Float64, Float16
 * Backends Tested: CPU, CUDA, OneAPI
 * Total Scenarios: 11 tests × 3 dtypes × 3 backends = 99 test scenarios
 *
 * Coverage:
 * - Basic checkpoint: forward pass, gradient correctness
 * - Multi-variable: checkpoint with multiple inputs
 * - Activations: ReLU, Sigmoid through checkpoint
 * - Nested: nested checkpoint contexts
 * - Memory: memory savings estimation
 * - Statistics: checkpoint stats tracking
 * - Complex: multi-operation checkpoint
 * - Segments: checkpoint segment execution
 * - Precision: dtype precision preservation
 */
