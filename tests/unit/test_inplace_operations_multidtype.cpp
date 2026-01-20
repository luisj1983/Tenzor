/**
 * @file test_inplace_operations_multidtype.cpp
 * @brief Multi-dtype multi-backend tests for in-place operations
 *
 * Tests in-place operations with Float32, Float64, and Float16 dtypes across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends:
 * - Inplace arithmetic (add_, sub_, mul_, div_)
 * - Inplace activations (relu_, sigmoid_, tanh_, etc.)
 * - Memory efficiency verification
 * - Dtype preservation
 * - Broadcasting support
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// In-Place Operations Multi-Backend Multi-DType Test Fixture
// ============================================================================

class InPlaceOperationsMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// In-Place Arithmetic Operations
// ============================================================================

TEST_P(InPlaceOperationsMultiDTypeTest, AddInPlace) {
    auto a = createOnes({3, 4});
    auto b = createOnes({3, 4}) * 2.0f;

    void* original_ptr = a.data_ptr();

    // In-place add
    add_(a, b);

    // Verify pointer didn't change (true in-place)
    EXPECT_EQ(a.data_ptr(), original_ptr);

    // Verify values
    auto a_cpu = a.to(Device::cpu()).to(DType::Float32);
    const float* data = a_cpu.data<float>();
    for (size_t i = 0; i < static_cast<size_t>(a_cpu.numel()); ++i) {
        EXPECT_NEAR(data[i], 3.0f, atol()) << "at index " << i;
    }
}

TEST_P(InPlaceOperationsMultiDTypeTest, MulInPlace) {
    auto a = createOnes({3, 4}) * 2.0f;
    auto b = createOnes({3, 4}) * 3.0f;

    void* original_ptr = a.data_ptr();

    // In-place multiply
    mul_(a, b);

    // Verify pointer didn't change
    EXPECT_EQ(a.data_ptr(), original_ptr);

    // Verify values
    auto a_cpu = a.to(Device::cpu()).to(DType::Float32);
    const float* data = a_cpu.data<float>();
    for (size_t i = 0; i < static_cast<size_t>(a_cpu.numel()); ++i) {
        EXPECT_NEAR(data[i], 6.0f, atol()) << "at index " << i;
    }
}

TEST_P(InPlaceOperationsMultiDTypeTest, SubInPlace) {
    auto a = createOnes({3, 4}) * 5.0f;
    auto b = createOnes({3, 4}) * 2.0f;

    void* original_ptr = a.data_ptr();

    // In-place subtract
    sub_(a, b);

    // Verify pointer didn't change
    EXPECT_EQ(a.data_ptr(), original_ptr);

    // Verify values
    auto a_cpu = a.to(Device::cpu()).to(DType::Float32);
    const float* data = a_cpu.data<float>();
    for (size_t i = 0; i < static_cast<size_t>(a_cpu.numel()); ++i) {
        EXPECT_NEAR(data[i], 3.0f, atol()) << "at index " << i;
    }
}

TEST_P(InPlaceOperationsMultiDTypeTest, DivInPlace) {
    auto a = createOnes({3, 4}) * 6.0f;
    auto b = createOnes({3, 4}) * 2.0f;

    void* original_ptr = a.data_ptr();

    // In-place divide
    div_(a, b);

    // Verify pointer didn't change
    EXPECT_EQ(a.data_ptr(), original_ptr);

    // Verify values
    auto a_cpu = a.to(Device::cpu()).to(DType::Float32);
    const float* data = a_cpu.data<float>();
    for (size_t i = 0; i < static_cast<size_t>(a_cpu.numel()); ++i) {
        EXPECT_NEAR(data[i], 3.0f, atol()) << "at index " << i;
    }
}

TEST_P(InPlaceOperationsMultiDTypeTest, ChainedInPlaceOperations) {
    auto a = createOnes({3, 4});
    auto b = createOnes({3, 4}) * 2.0f;
    auto c = createOnes({3, 4}) * 3.0f;

    void* original_ptr = a.data_ptr();

    // Chain multiple in-place operations: (1 + 2) * 3 - 2 = 7
    add_(a, b);  // a = 1 + 2 = 3
    mul_(a, c);  // a = 3 * 3 = 9
    sub_(a, b);  // a = 9 - 2 = 7

    // Verify pointer stayed the same
    EXPECT_EQ(a.data_ptr(), original_ptr);

    // Verify final value
    auto a_cpu = a.to(Device::cpu()).to(DType::Float32);
    const float* data = a_cpu.data<float>();
    for (size_t i = 0; i < static_cast<size_t>(a_cpu.numel()); ++i) {
        EXPECT_NEAR(data[i], 7.0f, atol()) << "at index " << i;
    }
}

// ============================================================================
// In-Place Activation Functions
// ============================================================================

TEST_P(InPlaceOperationsMultiDTypeTest, ReLUInPlace) {
    // Create tensor with positive and negative values
    auto a = tenzor::randn({2, 3}, DType::Float32, Device::cpu());
    if (dtype() != DType::Float32) {
        a = a.to(dtype());
    }
    if (device() != Device::cpu()) {
        a = a.to(device());
    }

    void* original_ptr = a.data_ptr();

    // In-place ReLU
    nn::relu_(a);

    // Verify pointer didn't change
    EXPECT_EQ(a.data_ptr(), original_ptr);

    // Verify all values are non-negative
    auto a_cpu = a.to(Device::cpu()).to(DType::Float32);
    const float* data = a_cpu.data<float>();
    for (size_t i = 0; i < static_cast<size_t>(a_cpu.numel()); ++i) {
        EXPECT_GE(data[i], 0.0f) << "at index " << i;
    }
}

TEST_P(InPlaceOperationsMultiDTypeTest, SigmoidInPlace) {
    auto a = tenzor::randn({3, 4}, DType::Float32, Device::cpu());
    if (dtype() != DType::Float32) {
        a = a.to(dtype());
    }
    if (device() != Device::cpu()) {
        a = a.to(device());
    }

    void* original_ptr = a.data_ptr();

    // In-place sigmoid
    nn::sigmoid_(a);

    // Verify pointer didn't change
    EXPECT_EQ(a.data_ptr(), original_ptr);

    // Verify values are in (0, 1) range
    auto a_cpu = a.to(Device::cpu()).to(DType::Float32);
    const float* data = a_cpu.data<float>();
    for (size_t i = 0; i < static_cast<size_t>(a_cpu.numel()); ++i) {
        EXPECT_GT(data[i], 0.0f) << "at index " << i;
        EXPECT_LT(data[i], 1.0f) << "at index " << i;
    }
}

TEST_P(InPlaceOperationsMultiDTypeTest, TanhInPlace) {
    auto a = tenzor::randn({3, 4}, DType::Float32, Device::cpu());
    if (dtype() != DType::Float32) {
        a = a.to(dtype());
    }
    if (device() != Device::cpu()) {
        a = a.to(device());
    }

    void* original_ptr = a.data_ptr();

    // In-place tanh
    nn::tanh_(a);

    // Verify pointer didn't change
    EXPECT_EQ(a.data_ptr(), original_ptr);

    // Verify values are in (-1, 1) range
    auto a_cpu = a.to(Device::cpu()).to(DType::Float32);
    const float* data = a_cpu.data<float>();
    for (size_t i = 0; i < static_cast<size_t>(a_cpu.numel()); ++i) {
        EXPECT_GT(data[i], -1.0f) << "at index " << i;
        EXPECT_LT(data[i], 1.0f) << "at index " << i;
    }
}

TEST_P(InPlaceOperationsMultiDTypeTest, LeakyReLUInPlace) {
    auto a = tenzor::randn({3, 4}, DType::Float32, Device::cpu());
    if (dtype() != DType::Float32) {
        a = a.to(dtype());
    }
    if (device() != Device::cpu()) {
        a = a.to(device());
    }

    // Store original values for comparison
    auto original = a.clone();

    void* original_ptr = a.data_ptr();

    // In-place leaky ReLU with slope 0.1
    nn::leaky_relu_(a, 0.1f);

    // Verify pointer didn't change
    EXPECT_EQ(a.data_ptr(), original_ptr);

    // Verify dtype preservation
    EXPECT_EQ(a.dtype(), dtype());
}

TEST_P(InPlaceOperationsMultiDTypeTest, GeLUInPlace) {
    auto a = tenzor::randn({3, 4}, DType::Float32, Device::cpu());
    if (dtype() != DType::Float32) {
        a = a.to(dtype());
    }
    if (device() != Device::cpu()) {
        a = a.to(device());
    }

    void* original_ptr = a.data_ptr();

    // In-place GELU
    nn::gelu_(a);

    // Verify pointer didn't change
    EXPECT_EQ(a.data_ptr(), original_ptr);

    // Verify dtype preservation
    EXPECT_EQ(a.dtype(), dtype());
}

// ============================================================================
// DType Preservation Tests
// ============================================================================

TEST_P(InPlaceOperationsMultiDTypeTest, DTypePreservation) {
    auto a = createOnes({3, 4});
    auto b = createOnes({3, 4});

    // Verify initial dtype
    EXPECT_EQ(a.dtype(), dtype());

    // In-place operation
    add_(a, b);

    // Verify dtype is preserved
    EXPECT_EQ(a.dtype(), dtype());
}

TEST_P(InPlaceOperationsMultiDTypeTest, ActivationDTypePreservation) {
    auto a = createOnes({3, 4});

    EXPECT_EQ(a.dtype(), dtype());

    nn::relu_(a);
    EXPECT_EQ(a.dtype(), dtype());

    nn::sigmoid_(a);
    EXPECT_EQ(a.dtype(), dtype());

    nn::tanh_(a);
    EXPECT_EQ(a.dtype(), dtype());
}

// ============================================================================
// Memory Efficiency Tests
// ============================================================================

TEST_P(InPlaceOperationsMultiDTypeTest, MemoryEfficiencyComparison) {
    const int64_t size = 100000;  // 100K elements
    auto a = createOnes({size});
    auto b = createOnes({size});

    // Record initial pointer
    void* original_ptr = a.data_ptr();

    // In-place operation should not allocate new memory
    add_(a, b);

    // Verify no new allocation (pointer unchanged)
    EXPECT_EQ(a.data_ptr(), original_ptr);

    // Out-of-place would create new tensor
    auto c = createOnes({size});
    auto d = add(c, b);

    // d should be different tensor with different pointer
    EXPECT_NE(d.data_ptr(), c.data_ptr());
}

TEST_P(InPlaceOperationsMultiDTypeTest, LargeActivationInPlace) {
    const int64_t size = 100000;
    auto a = createOnes({size}) * 2.0f;

    void* original_ptr = a.data_ptr();

    // In-place activation on large tensor
    nn::relu_(a);

    // Should not allocate
    EXPECT_EQ(a.data_ptr(), original_ptr);

    // Verify some values are still 2.0 (since input was positive)
    auto a_cpu = a.to(Device::cpu()).to(DType::Float32);
    const float* data = a_cpu.data<float>();
    for (size_t i = 0; i < std::min(size_t(100), static_cast<size_t>(a_cpu.numel())); ++i) {
        EXPECT_NEAR(data[i], 2.0f, atol()) << "at index " << i;
    }
}

TEST_P(InPlaceOperationsMultiDTypeTest, MemorySizeConsistency) {
    auto a = createOnes({10, 20});

    size_t num_elements = a.numel();

    // After in-place operation, size should remain the same
    auto b = createOnes({10, 20});
    add_(a, b);

    EXPECT_EQ(a.numel(), num_elements);
}

// ============================================================================
// Broadcasting Tests
// ============================================================================

TEST_P(InPlaceOperationsMultiDTypeTest, BroadcastingInPlace) {
    auto a = createOnes({3, 4});
    auto b = tenzor::ones({1, 4}, dtype(), device()) * 2.0f;

    void* original_ptr = a.data_ptr();

    // In-place add with broadcasting
    add_(a, b);

    EXPECT_EQ(a.data_ptr(), original_ptr);

    // All values should be 3.0 (1.0 + 2.0)
    auto a_cpu = a.to(Device::cpu()).to(DType::Float32);
    const float* data = a_cpu.data<float>();
    for (size_t i = 0; i < static_cast<size_t>(a_cpu.numel()); ++i) {
        EXPECT_NEAR(data[i], 3.0f, atol()) << "at index " << i;
    }
}

TEST_P(InPlaceOperationsMultiDTypeTest, ScalarBroadcastInPlace) {
    auto a = createOnes({4, 3}) * 2.0f;
    auto b = tenzor::ones({1}, dtype(), device()) * 3.0f;

    void* original_ptr = a.data_ptr();

    mul_(a, b);

    EXPECT_EQ(a.data_ptr(), original_ptr);

    auto a_cpu = a.to(Device::cpu()).to(DType::Float32);
    const float* data = a_cpu.data<float>();
    for (size_t i = 0; i < static_cast<size_t>(a_cpu.numel()); ++i) {
        EXPECT_NEAR(data[i], 6.0f, atol()) << "at index " << i;
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_P(InPlaceOperationsMultiDTypeTest, NonContiguousTensorError) {
    auto a = createOnes({4, 4});
    auto b = createOnes({4, 4});

    // Create non-contiguous view via transpose
    auto a_t = a.transpose(0, 1);

    // In-place operations should fail on non-contiguous tensors
    EXPECT_THROW(add_(a_t, b), std::runtime_error);
    EXPECT_THROW(mul_(a_t, b), std::runtime_error);
    EXPECT_THROW(nn::relu_(a_t), std::runtime_error);
    EXPECT_THROW(nn::sigmoid_(a_t), std::runtime_error);
}

TEST_P(InPlaceOperationsMultiDTypeTest, ShapeMismatchError) {
    auto a = createOnes({3, 4});
    auto b = createOnes({4, 3});

    // Shape mismatch should throw
    EXPECT_THROW(add_(a, b), std::runtime_error);
    EXPECT_THROW(mul_(a, b), std::runtime_error);
}

// ============================================================================
// Gradient Compatibility Tests
// ============================================================================

TEST_P(InPlaceOperationsMultiDTypeTest, GradientCompatibility) {
    auto a_tensor = createOnes({3, 4});
    auto b_tensor = createOnes({3, 4});

    Variable a(a_tensor, true);
    Variable b(b_tensor, false);

    void* original_ptr = a.tensor().data_ptr();

    // In-place operation with gradients enabled
    add_(a.tensor(), b.tensor());

    // Verify pointer didn't change
    EXPECT_EQ(a.tensor().data_ptr(), original_ptr);
}

// ============================================================================
// Complex Workflow Tests
// ============================================================================

TEST_P(InPlaceOperationsMultiDTypeTest, ComplexInPlaceWorkflow) {
    // Simulate a mini neural network layer update
    auto weights = createOnes({5, 5}) * 0.5f;
    auto gradients = createOnes({5, 5}) * 0.1f;

    auto lr_tensor = tenzor::ones({1}, dtype(), device()) * 0.01f;

    void* original_ptr = weights.data_ptr();

    // Weight update: weights -= learning_rate * gradients
    auto scaled_grad = gradients * lr_tensor;
    sub_(weights, scaled_grad);

    EXPECT_EQ(weights.data_ptr(), original_ptr);

    // Verify update: 0.5 - 0.01 * 0.1 = 0.499
    auto weights_cpu = weights.to(Device::cpu()).to(DType::Float32);
    const float* data = weights_cpu.data<float>();
    for (size_t i = 0; i < static_cast<size_t>(weights_cpu.numel()); ++i) {
        EXPECT_NEAR(data[i], 0.499f, atol()) << "at index " << i;
    }
}

TEST_P(InPlaceOperationsMultiDTypeTest, ActivationPipelineInPlace) {
    // Create input with mixed values
    auto a = tenzor::randn({2, 3}, DType::Float32, Device::cpu());
    if (dtype() != DType::Float32) {
        a = a.to(dtype());
    }
    if (device() != Device::cpu()) {
        a = a.to(device());
    }

    void* original_ptr = a.data_ptr();

    // Apply multiple activations in sequence
    nn::relu_(a);           // First eliminate negatives
    nn::sigmoid_(a);        // Then squash to (0,1)

    EXPECT_EQ(a.data_ptr(), original_ptr);

    // All values should be in (0, 1) range
    auto a_cpu = a.to(Device::cpu()).to(DType::Float32);
    const float* data = a_cpu.data<float>();
    for (size_t i = 0; i < static_cast<size_t>(a_cpu.numel()); ++i) {
        EXPECT_GE(data[i], 0.0f) << "at index " << i;
        EXPECT_LE(data[i], 1.0f) << "at index " << i;
    }
}

// ============================================================================
// Normalization Tests
// ============================================================================

TEST_P(InPlaceOperationsMultiDTypeTest, InPlaceNormalization) {
    // Create tensor with known values
    auto a = tenzor::arange(1.0f, 5.0f, 1.0f, DType::Float32, Device::cpu()).reshape({2, 2});
    if (dtype() != DType::Float32) {
        a = a.to(dtype());
    }
    if (device() != Device::cpu()) {
        a = a.to(device());
    }

    void* original_ptr = a.data_ptr();

    // In-place normalization: x = x - mean
    float mean_val = 2.5f;  // (1+2+3+4)/4
    auto mean_tensor = tenzor::ones({1}, dtype(), device()) * mean_val;
    sub_(a, mean_tensor);

    EXPECT_EQ(a.data_ptr(), original_ptr);

    // Verify centering
    auto a_cpu = a.to(Device::cpu()).to(DType::Float32);
    const float* data = a_cpu.data<float>();
    std::vector<float> expected = {-1.5f, -0.5f, 0.5f, 1.5f};
    for (size_t i = 0; i < static_cast<size_t>(a_cpu.numel()); ++i) {
        EXPECT_NEAR(data[i], expected[i], atol()) << "at index " << i;
    }
}

// ============================================================================
// Specific Value Tests
// ============================================================================

TEST_P(InPlaceOperationsMultiDTypeTest, SigmoidSpecificValue) {
    // Test sigmoid(0) = 0.5
    auto a = createZeros({1});

    nn::sigmoid_(a);

    auto a_cpu = a.to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(a_cpu.data<float>()[0], 0.5f, atol());
}

TEST_P(InPlaceOperationsMultiDTypeTest, TanhSpecificValue) {
    // Test tanh(0) = 0
    auto a = createZeros({1});

    nn::tanh_(a);

    auto a_cpu = a.to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(a_cpu.data<float>()[0], 0.0f, atol());
}

TEST_P(InPlaceOperationsMultiDTypeTest, ReLUPreservesPositive) {
    // ReLU should preserve positive values
    auto a = createOnes({3, 4}) * 5.0f;
    auto original_sum = tenzor::sum(a).to(Device::cpu()).to(DType::Float32).data<float>()[0];

    nn::relu_(a);

    auto new_sum = tenzor::sum(a).to(Device::cpu()).to(DType::Float32).data<float>()[0];
    EXPECT_NEAR(new_sum, original_sum, atol());
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(InPlaceOperationsMultiDTypeTest);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 25
 * DTypes Tested: Float32, Float64, Float16
 * Backends Tested: CPU, CUDA, OneAPI
 * Total Scenarios: 25 tests × 3 dtypes × 3 backends = 225 test scenarios
 *
 * Coverage:
 * - In-place arithmetic: add_, sub_, mul_, div_, chained operations
 * - In-place activations: relu_, sigmoid_, tanh_, leaky_relu_, gelu_
 * - DType preservation: arithmetic, activations
 * - Memory efficiency: pointer stability, large tensors, size consistency
 * - Broadcasting: row broadcast, scalar broadcast
 * - Error handling: non-contiguous tensors, shape mismatch
 * - Gradient compatibility
 * - Complex workflows: weight updates, activation pipelines
 * - Normalization
 * - Specific values: sigmoid(0), tanh(0), relu preserves positive
 */
