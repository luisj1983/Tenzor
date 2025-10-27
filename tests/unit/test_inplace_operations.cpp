/**
 * @file test_inplace_operations.cpp
 * @brief Unit tests for in-place operations with memory allocation tracking
 */

#include <gtest/gtest.h>
#include "tenzor/ops/math.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/core/tensor.hpp"
#include <memory>

namespace tenzor {
namespace test {

class InPlaceOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = Device::cpu();
    }

    Device device_;
};

// ============================================================================
// In-Place Arithmetic Operations
// ============================================================================

TEST_F(InPlaceOperationsTest, AddInPlace) {
    auto a = ones({3, 4}, DType::Float32, device_);
    auto b = ones({3, 4}, DType::Float32, device_) * 2.0f;

    void* original_ptr = a.data();

    // In-place add
    add_(a, b);

    // Verify pointer didn't change (true in-place)
    EXPECT_EQ(a.data(), original_ptr);

    // Verify values
    auto data = a.to_vector<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.0f) << "at index " << i;
    }
}

TEST_F(InPlaceOperationsTest, MulInPlace) {
    auto a = ones({3, 4}, DType::Float32, device_) * 2.0f;
    auto b = ones({3, 4}, DType::Float32, device_) * 3.0f;

    void* original_ptr = a.data();

    // In-place multiply
    mul_(a, b);

    // Verify pointer didn't change
    EXPECT_EQ(a.data(), original_ptr);

    // Verify values
    auto data = a.to_vector<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 6.0f) << "at index " << i;
    }
}

TEST_F(InPlaceOperationsTest, SubInPlace) {
    auto a = ones({3, 4}, DType::Float32, device_) * 5.0f;
    auto b = ones({3, 4}, DType::Float32, device_) * 2.0f;

    void* original_ptr = a.data();

    // In-place subtract
    sub_(a, b);

    // Verify pointer didn't change
    EXPECT_EQ(a.data(), original_ptr);

    // Verify values
    auto data = a.to_vector<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.0f) << "at index " << i;
    }
}

TEST_F(InPlaceOperationsTest, DivInPlace) {
    auto a = ones({3, 4}, DType::Float32, device_) * 6.0f;
    auto b = ones({3, 4}, DType::Float32, device_) * 2.0f;

    void* original_ptr = a.data();

    // In-place divide
    div_(a, b);

    // Verify pointer didn't change
    EXPECT_EQ(a.data(), original_ptr);

    // Verify values
    auto data = a.to_vector<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.0f) << "at index " << i;
    }
}

TEST_F(InPlaceOperationsTest, ChainedInPlaceOperations) {
    auto a = ones({3, 4}, DType::Float32, device_);
    auto b = ones({3, 4}, DType::Float32, device_) * 2.0f;
    auto c = ones({3, 4}, DType::Float32, device_) * 3.0f;

    void* original_ptr = a.data();

    // Chain multiple in-place operations
    add_(a, b);  // a = 1 + 2 = 3
    mul_(a, c);  // a = 3 * 3 = 9
    sub_(a, b);  // a = 9 - 2 = 7

    // Verify pointer stayed the same
    EXPECT_EQ(a.data(), original_ptr);

    // Verify final value
    auto data = a.to_vector<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 7.0f) << "at index " << i;
    }
}

// ============================================================================
// In-Place Activation Functions
// ============================================================================

TEST_F(InPlaceOperationsTest, ReLUInPlace) {
    // Create tensor with positive and negative values
    std::vector<float> values = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    auto a = Tensor::from_vector(values, {2, 3}, DType::Float32, device_);

    void* original_ptr = a.data();

    // In-place ReLU
    nn::relu_(a);

    // Verify pointer didn't change
    EXPECT_EQ(a.data(), original_ptr);

    // Verify values: max(0, x)
    auto data = a.to_vector<float>();
    std::vector<float> expected = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    ASSERT_EQ(data.size(), expected.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(data[i], expected[i]) << "at index " << i;
    }
}

TEST_F(InPlaceOperationsTest, SigmoidInPlace) {
    std::vector<float> values = {-1.0f, 0.0f, 1.0f};
    auto a = Tensor::from_vector(values, {3}, DType::Float32, device_);

    void* original_ptr = a.data();

    // In-place sigmoid
    nn::sigmoid_(a);

    // Verify pointer didn't change
    EXPECT_EQ(a.data(), original_ptr);

    // Verify values are in (0, 1) range
    auto data = a.to_vector<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_GT(data[i], 0.0f) << "at index " << i;
        EXPECT_LT(data[i], 1.0f) << "at index " << i;
    }

    // Check specific value: sigmoid(0) = 0.5
    EXPECT_NEAR(data[1], 0.5f, 1e-5);
}

TEST_F(InPlaceOperationsTest, TanhInPlace) {
    std::vector<float> values = {-1.0f, 0.0f, 1.0f};
    auto a = Tensor::from_vector(values, {3}, DType::Float32, device_);

    void* original_ptr = a.data();

    // In-place tanh
    nn::tanh_(a);

    // Verify pointer didn't change
    EXPECT_EQ(a.data(), original_ptr);

    // Verify values are in (-1, 1) range
    auto data = a.to_vector<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_GT(data[i], -1.0f) << "at index " << i;
        EXPECT_LT(data[i], 1.0f) << "at index " << i;
    }

    // Check specific value: tanh(0) = 0
    EXPECT_NEAR(data[1], 0.0f, 1e-5);
}

TEST_F(InPlaceOperationsTest, LeakyReLUInPlace) {
    std::vector<float> values = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    auto a = Tensor::from_vector(values, {5}, DType::Float32, device_);

    void* original_ptr = a.data();

    // In-place leaky ReLU with slope 0.1
    nn::leaky_relu_(a, 0.1);

    // Verify pointer didn't change
    EXPECT_EQ(a.data(), original_ptr);

    // Verify values
    auto data = a.to_vector<float>();
    std::vector<float> expected = {-0.2f, -0.1f, 0.0f, 1.0f, 2.0f};
    ASSERT_EQ(data.size(), expected.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_NEAR(data[i], expected[i], 1e-5) << "at index " << i;
    }
}

TEST_F(InPlaceOperationsTest, GeLUInPlace) {
    std::vector<float> values = {-1.0f, 0.0f, 1.0f};
    auto a = Tensor::from_vector(values, {3}, DType::Float32, device_);

    void* original_ptr = a.data();

    // In-place GELU
    nn::gelu_(a);

    // Verify pointer didn't change
    EXPECT_EQ(a.data(), original_ptr);

    // GELU(0) should be approximately 0
    auto data = a.to_vector<float>();
    EXPECT_NEAR(data[1], 0.0f, 1e-3);

    // GELU output should be smooth
    EXPECT_LT(data[0], 0.0f);  // Negative input gives negative output
    EXPECT_GT(data[2], 0.0f);  // Positive input gives positive output
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(InPlaceOperationsTest, NonContiguousTensorError) {
    auto a = ones({4, 4}, DType::Float32, device_);
    auto b = ones({4, 4}, DType::Float32, device_);

    // Create non-contiguous view via transpose
    auto a_t = a.transpose(0, 1);

    // In-place operations should fail on non-contiguous tensors
    EXPECT_THROW(add_(a_t, b), std::runtime_error);
    EXPECT_THROW(mul_(a_t, b), std::runtime_error);
    EXPECT_THROW(nn::relu_(a_t), std::runtime_error);
    EXPECT_THROW(nn::sigmoid_(a_t), std::runtime_error);
}

// ============================================================================
// Memory Efficiency Tests
// ============================================================================

TEST_F(InPlaceOperationsTest, MemoryEfficiencyComparison) {
    const size_t size = 1000 * 1000;  // 1M elements = 4MB for float32
    auto a = ones({size}, DType::Float32, device_);
    auto b = ones({size}, DType::Float32, device_);

    // Record initial pointer
    void* original_ptr = a.data();

    // In-place operation should not allocate new memory
    add_(a, b);

    // Verify no new allocation (pointer unchanged)
    EXPECT_EQ(a.data(), original_ptr);

    // Out-of-place would create new tensor
    auto c = ones({size}, DType::Float32, device_);
    auto d = add(c, b);

    // d should be different tensor with different pointer
    EXPECT_NE(d.data(), c.data());
}

TEST_F(InPlaceOperationsTest, LargeActivationInPlace) {
    const size_t size = 1000 * 1000;
    auto a = ones({size}, DType::Float32, device_) * 2.0f;

    void* original_ptr = a.data();

    // In-place activation on large tensor
    nn::relu_(a);

    // Should not allocate
    EXPECT_EQ(a.data(), original_ptr);

    // Verify all values are still 2.0 (since input was positive)
    auto data = a.to_vector<float>();
    for (size_t i = 0; i < std::min(size_t(100), data.size()); ++i) {
        EXPECT_FLOAT_EQ(data[i], 2.0f);
    }
}

// ============================================================================
// Broadcasting Tests
// ============================================================================

TEST_F(InPlaceOperationsTest, BroadcastingInPlace) {
    auto a = ones({3, 4}, DType::Float32, device_);
    auto b = ones({1, 4}, DType::Float32, device_) * 2.0f;

    void* original_ptr = a.data();

    // In-place add with broadcasting
    add_(a, b);

    EXPECT_EQ(a.data(), original_ptr);

    // All values should be 3.0 (1.0 + 2.0)
    auto data = a.to_vector<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.0f) << "at index " << i;
    }
}

} // namespace test
} // namespace tenzor
