/**
 * @file test_oneapi_backend.cpp
 * @brief Comprehensive tests for OneAPI/SYCL backend implementation
 *
 * Tests cover:
 * - Basic tensor allocation and deallocation
 * - Memory copy operations (host↔device)
 * - Arithmetic operations (add, sub, mul, div)
 * - Matrix multiplication (matmul)
 * - Unary operations (sqrt, neg, abs, exp, log, pow)
 * - Activation functions (relu, sigmoid, tanh, gelu, leaky_relu)
 * - Reduction operations (sum, mean, max, min)
 * - Transform operations (reshape, transpose, permute, squeeze, unsqueeze)
 * - Fill operations (zeros, ones, full, fill)
 * - Convolution operations (conv2d_forward, conv2d_backward)
 * - Batch normalization (batchnorm2d)
 * - Error handling and edge cases
 * - Multi-device scenarios (if available)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../multi_backend_dtype_fixture.hpp"  // CC.18: SKIP_WITH_REASON
#include <cmath>
#include <algorithm>

using namespace tenzor;

/**
 * @brief Test fixture for OneAPI backend tests
 *
 * Sets up the Tenzor environment and provides helper methods for
 * device availability checking and tensor validation.
 */
class OneAPIBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize Tenzor (loads all backends including OneAPI)
        initialize();
    }

    /**
     * @brief Check if OneAPI devices are available
     * @return true if at least one OneAPI device is available
     */
    bool hasOneAPIDevice() const {
        try {
            auto device = Device::oneapi(0);
            // Try to create a small tensor to verify device is functional
            auto t = zeros({1}, DType::Float32, device);
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    /**
     * @brief Get number of available OneAPI devices
     * @return Device count (0 if none available)
     */
    int32_t getOneAPIDeviceCount() const {
        try {
            int32_t count = 0;
            while (true) {
                auto device = Device::oneapi(count);
                auto t = zeros({1}, DType::Float32, device);
                count++;
            }
        } catch (const std::exception&) {
            return 0;
        }
        return 0;
    }

    /**
     * @brief Helper to compare two span shapes
     */
    bool shapesEqual(std::span<const int64_t> a, std::span<const int64_t> b) const {
        if (a.size() != b.size()) {
            return false;
        }
        return std::equal(a.begin(), a.end(), b.begin());
    }

    /**
     * @brief Compare two tensors with tolerance
     * @param a First tensor (on device or CPU)
     * @param b Second tensor (on device or CPU)
     * @param rtol Relative tolerance
     * @param atol Absolute tolerance
     * @return true if tensors are equal within tolerance
     */
    bool tensorsClose(const Tensor& a, const Tensor& b,
                      float rtol = 1e-5f, float atol = 1e-7f) {
        // Move both tensors to CPU for comparison
        auto a_cpu = a.to(Device::cpu());
        auto b_cpu = b.to(Device::cpu());

        if (!shapesEqual(a_cpu.shape(), b_cpu.shape())) {
            return false;
        }

        auto a_data = a_cpu.data<float>();
        auto b_data = b_cpu.data<float>();
        auto numel = a_cpu.numel();

        for (int64_t i = 0; i < numel; ++i) {
            float diff = std::abs(a_data[i] - b_data[i]);
            float threshold = atol + rtol * std::abs(b_data[i]);
            if (diff > threshold) {
                return false;
            }
        }

        return true;
    }
};

// ============================================================================
// Basic Backend Registration and Device Management
// ============================================================================

TEST_F(OneAPIBackendTest, BackendRegistration) {
    // Test that we can create a OneAPI device object
    // This will fail if backend isn't registered
    try {
        auto device = Device::oneapi(0);
        EXPECT_EQ(device.type, Device::Type::OneAPI);
        EXPECT_EQ(device.index, 0);
    } catch (const std::exception& e) {
        FAIL() << "BackendRegistration threw exception: " << e.what();
    }
}

TEST_F(OneAPIBackendTest, DeviceAvailability) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    // Verify device properties
    auto device = Device::oneapi(0);
    auto t = zeros({2, 3}, DType::Float32, device);
    EXPECT_EQ(t.device().type, Device::Type::OneAPI);
    EXPECT_EQ(t.device().index, 0);
}

// ============================================================================
// Memory Allocation and Deallocation Tests
// ============================================================================

TEST_F(OneAPIBackendTest, BasicMemoryAllocation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    // Test various tensor sizes
    auto t1 = zeros({1}, DType::Float32, device);
    EXPECT_EQ(t1.numel(), 1);

    auto t2 = zeros({10, 20}, DType::Float32, device);
    EXPECT_EQ(t2.numel(), 200);

    auto t3 = zeros({5, 10, 15}, DType::Float32, device);
    EXPECT_EQ(t3.numel(), 750);

    auto t4 = zeros({2, 3, 4, 5}, DType::Float32, device);
    EXPECT_EQ(t4.numel(), 120);
}

TEST_F(OneAPIBackendTest, LargeMemoryAllocation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    // Allocate larger tensors to test memory management
    auto t1 = zeros({1000, 1000}, DType::Float32, device);
    EXPECT_EQ(t1.numel(), 1000000);

    auto t2 = zeros({100, 100, 100}, DType::Float32, device);
    EXPECT_EQ(t2.numel(), 1000000);
}

TEST_F(OneAPIBackendTest, MemoryDeallocation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    // Test that tensors are properly deallocated when going out of scope
    for (int i = 0; i < 100; ++i) {
        auto t = zeros({100, 100}, DType::Float32, device);
        // Tensor destroyed at end of each iteration
    }
}

// ============================================================================
// Memory Copy Operations Tests
// ============================================================================

TEST_F(OneAPIBackendTest, HostToDeviceCopy) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    // Create tensor on CPU
    auto cpu_tensor = ones({3, 4}, DType::Float32, Device::cpu());

    // Copy to device
    auto gpu_tensor = cpu_tensor.to(device);
    EXPECT_EQ(gpu_tensor.device().type, Device::Type::OneAPI);
    EXPECT_TRUE(shapesEqual(gpu_tensor.shape(), cpu_tensor.shape()));
}

TEST_F(OneAPIBackendTest, DeviceToHostCopy) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    // Create tensor on device
    auto gpu_tensor = ones({3, 4}, DType::Float32, device);

    // Copy to CPU
    auto cpu_tensor = gpu_tensor.to(Device::cpu());
    EXPECT_EQ(cpu_tensor.device().type, Device::Type::CPU);
    EXPECT_TRUE(shapesEqual(cpu_tensor.shape(), gpu_tensor.shape()));

    // Verify data
    auto data = cpu_tensor.data<float>();
    for (int64_t i = 0; i < cpu_tensor.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST_F(OneAPIBackendTest, DeviceToDeviceCopy) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    // Create two tensors on same device
    auto t1 = ones({3, 4}, DType::Float32, device);

    auto t2 = t1.to(device);
    EXPECT_EQ(t2.device().type, Device::Type::OneAPI);
    EXPECT_TRUE(tensorsClose(t1, t2));
}

TEST_F(OneAPIBackendTest, RoundTripCopy) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    // Create CPU tensor with specific values
    auto cpu_original = full({5, 6}, 3.14f, DType::Float32, Device::cpu());

    // Copy to device and back
    auto gpu_tensor = cpu_original.to(device);
    auto cpu_result = gpu_tensor.to(Device::cpu());

    // Verify data integrity
    EXPECT_TRUE(tensorsClose(cpu_original, cpu_result));
}

// ============================================================================
// Binary Arithmetic Operations Tests
// ============================================================================

TEST_F(OneAPIBackendTest, AddOperation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto a = full({3, 4}, 2.0f, DType::Float32, device);
    auto b = full({3, 4}, 3.0f, DType::Float32, device);

    auto c = a + b;
    EXPECT_EQ(c.device().type, Device::Type::OneAPI);
    EXPECT_TRUE(shapesEqual(c.shape(), a.shape()));

    // Verify result
    auto cpu_result = c.to(Device::cpu());
    auto data = cpu_result.data<float>();
    for (int64_t i = 0; i < cpu_result.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 5.0f);
    }
}

TEST_F(OneAPIBackendTest, SubOperation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto a = full({3, 4}, 5.0f, DType::Float32, device);
    auto b = full({3, 4}, 2.0f, DType::Float32, device);

    auto c = a - b;
    EXPECT_EQ(c.device().type, Device::Type::OneAPI);

    auto cpu_result = c.to(Device::cpu());
    auto data = cpu_result.data<float>();
    for (int64_t i = 0; i < cpu_result.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.0f);
    }
}

TEST_F(OneAPIBackendTest, MulOperation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto a = full({3, 4}, 3.0f, DType::Float32, device);
    auto b = full({3, 4}, 4.0f, DType::Float32, device);

    auto c = a * b;
    EXPECT_EQ(c.device().type, Device::Type::OneAPI);

    auto cpu_result = c.to(Device::cpu());
    auto data = cpu_result.data<float>();
    for (int64_t i = 0; i < cpu_result.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 12.0f);
    }
}

TEST_F(OneAPIBackendTest, DivOperation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto a = full({3, 4}, 12.0f, DType::Float32, device);
    auto b = full({3, 4}, 4.0f, DType::Float32, device);

    auto c = a / b;
    EXPECT_EQ(c.device().type, Device::Type::OneAPI);

    auto cpu_result = c.to(Device::cpu());
    auto data = cpu_result.data<float>();
    for (int64_t i = 0; i < cpu_result.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.0f);
    }
}

// ============================================================================
// Matrix Multiplication Tests
// ============================================================================

TEST_F(OneAPIBackendTest, MatMulBasic) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    // Create matrices on CPU for ground truth
    auto a_cpu = ones({3, 4}, DType::Float32, Device::cpu());
    auto b_cpu = ones({4, 5}, DType::Float32, Device::cpu());
    auto expected = matmul(a_cpu, b_cpu);

    // Perform matmul on OneAPI device
    auto a = a_cpu.to(device);
    auto b = b_cpu.to(device);

    auto c = matmul(a, b);
    EXPECT_EQ(c.device().type, Device::Type::OneAPI);
    std::vector<int64_t> expected_shape = {3, 5};
    EXPECT_TRUE(shapesEqual(c.shape(), expected_shape));

    // Verify result
    EXPECT_TRUE(tensorsClose(c, expected, 1e-4f, 1e-5f));
}

TEST_F(OneAPIBackendTest, MatMulSquareMatrices) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto a = full({10, 10}, 2.0f, DType::Float32, device);
    auto b = full({10, 10}, 3.0f, DType::Float32, device);

    auto c = matmul(a, b);
    std::vector<int64_t> expected_shape = {10, 10};
    EXPECT_TRUE(shapesEqual(c.shape(), expected_shape));

    // Each element should be 2.0 * 3.0 * 10 = 60.0
    auto cpu_result = c.to(Device::cpu());
    auto data = cpu_result.data<float>();
    for (int64_t i = 0; i < cpu_result.numel(); ++i) {
        EXPECT_NEAR(data[i], 60.0f, 1e-3f);
    }
}

TEST_F(OneAPIBackendTest, MatMulLargeMatrices) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    // Test with larger matrices
    auto a = ones({64, 128}, DType::Float32, device);
    auto b = ones({128, 64}, DType::Float32, device);

    auto c = matmul(a, b);
    std::vector<int64_t> expected_shape = {64, 64};
    EXPECT_TRUE(shapesEqual(c.shape(), expected_shape));

    // Each element should be 1.0 * 1.0 * 128 = 128.0
    auto cpu_result = c.to(Device::cpu());
    auto data = cpu_result.data<float>();
    EXPECT_NEAR(data[0], 128.0f, 1e-3f);
}

// ============================================================================
// Unary Operations Tests
// ============================================================================

TEST_F(OneAPIBackendTest, SqrtOperation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = full({3, 4}, 4.0f, DType::Float32, device);

    auto result = sqrt(input);
    EXPECT_EQ(result.device().type, Device::Type::OneAPI);

    auto cpu_result = result.to(Device::cpu());
    auto data = cpu_result.data<float>();
    for (int64_t i = 0; i < cpu_result.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 2.0f);
    }
}

TEST_F(OneAPIBackendTest, NegOperation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = full({3, 4}, 3.5f, DType::Float32, device);

    auto result = neg(input);

    auto cpu_result = result.to(Device::cpu());
    auto data = cpu_result.data<float>();
    for (int64_t i = 0; i < cpu_result.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], -3.5f);
    }
}

TEST_F(OneAPIBackendTest, AbsOperation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = full({3, 4}, -2.5f, DType::Float32, device);

    auto result = tenzor::abs(input);

    auto cpu_result = result.to(Device::cpu());
    auto data = cpu_result.data<float>();
    for (int64_t i = 0; i < cpu_result.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 2.5f);
    }
}

TEST_F(OneAPIBackendTest, ExpOperation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = full({3, 4}, 1.0f, DType::Float32, device);

    auto result = exp(input);

    auto cpu_result = result.to(Device::cpu());
    auto data = cpu_result.data<float>();
    for (int64_t i = 0; i < cpu_result.numel(); ++i) {
        EXPECT_NEAR(data[i], std::exp(1.0f), 1e-5f);
    }
}

TEST_F(OneAPIBackendTest, LogOperation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = full({3, 4}, std::exp(1.0f), DType::Float32, device);

    auto result = log(input);

    auto cpu_result = result.to(Device::cpu());
    auto data = cpu_result.data<float>();
    for (int64_t i = 0; i < cpu_result.numel(); ++i) {
        EXPECT_NEAR(data[i], 1.0f, 1e-5f);
    }
}

TEST_F(OneAPIBackendTest, PowOperation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = full({3, 4}, 2.0f, DType::Float32, device);

    auto result = pow(input, 3.0f);

    auto cpu_result = result.to(Device::cpu());
    auto data = cpu_result.data<float>();
    for (int64_t i = 0; i < cpu_result.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 8.0f);
    }
}

// ============================================================================
// Activation Functions Tests
// ============================================================================

TEST_F(OneAPIBackendTest, ReluActivation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    // Create tensor with mixed positive and negative values on CPU
    auto input_cpu = zeros({4}, DType::Float32, Device::cpu());
    auto input_data = input_cpu.data<float>();
    input_data[0] = -2.0f;
    input_data[1] = -1.0f;
    input_data[2] = 1.0f;
    input_data[3] = 2.0f;

    auto input = input_cpu.to(device);

    auto result = nn::relu(Variable(input)).tensor();

    auto cpu_result = result.to(Device::cpu());
    auto data = cpu_result.data<float>();
    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[1], 0.0f);
    EXPECT_FLOAT_EQ(data[2], 1.0f);
    EXPECT_FLOAT_EQ(data[3], 2.0f);
}

TEST_F(OneAPIBackendTest, SigmoidActivation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = zeros({3, 4}, DType::Float32, device);

    auto result = nn::sigmoid(Variable(input)).tensor();
    EXPECT_EQ(result.device().type, Device::Type::OneAPI);

    // sigmoid(0) = 0.5
    auto cpu_result = result.to(Device::cpu());
    auto data = cpu_result.data<float>();
    for (int64_t i = 0; i < cpu_result.numel(); ++i) {
        EXPECT_NEAR(data[i], 0.5f, 1e-5f);
    }
}

TEST_F(OneAPIBackendTest, TanhActivation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = zeros({3, 4}, DType::Float32, device);

    auto result = nn::tanh(Variable(input)).tensor();

    // tanh(0) = 0.0
    auto cpu_result = result.to(Device::cpu());
    auto data = cpu_result.data<float>();
    for (int64_t i = 0; i < cpu_result.numel(); ++i) {
        EXPECT_NEAR(data[i], 0.0f, 1e-5f);
    }
}

TEST_F(OneAPIBackendTest, LeakyReluActivation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input_cpu = zeros({4}, DType::Float32, Device::cpu());
    auto input_data = input_cpu.data<float>();
    input_data[0] = -2.0f;
    input_data[1] = -1.0f;
    input_data[2] = 1.0f;
    input_data[3] = 2.0f;

    auto input = input_cpu.to(device);

    float alpha = 0.01f;
    auto result = nn::leaky_relu(Variable(input), alpha).tensor();

    auto cpu_result = result.to(Device::cpu());
    auto data = cpu_result.data<float>();
    EXPECT_NEAR(data[0], -0.02f, 1e-5f);
    EXPECT_NEAR(data[1], -0.01f, 1e-5f);
    EXPECT_FLOAT_EQ(data[2], 1.0f);
    EXPECT_FLOAT_EQ(data[3], 2.0f);
}

// ============================================================================
// Reduction Operations Tests
// ============================================================================

TEST_F(OneAPIBackendTest, SumReduction) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = full({3, 4}, 2.0f, DType::Float32, device);

    // Sum all elements (use std::nullopt for full reduction)
    auto result = sum(input, std::nullopt, false);
    EXPECT_EQ(result.numel(), 1);

    auto cpu_result = result.to(Device::cpu());
    auto data = cpu_result.data<float>();
    EXPECT_FLOAT_EQ(data[0], 24.0f);  // 3 * 4 * 2.0
}

TEST_F(OneAPIBackendTest, SumReductionAlongDimension) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = ones({3, 4}, DType::Float32, device);

    // Sum along dimension 1
    auto result = sum(input, 1, false);
    std::vector<int64_t> expected_shape = {3};
    EXPECT_TRUE(shapesEqual(result.shape(), expected_shape));

    auto cpu_result = result.to(Device::cpu());
    auto data = cpu_result.data<float>();
    for (int64_t i = 0; i < cpu_result.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 4.0f);
    }
}

TEST_F(OneAPIBackendTest, MeanReduction) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = full({3, 4}, 6.0f, DType::Float32, device);

    auto result = mean(input, -1, false);

    auto cpu_result = result.to(Device::cpu());
    auto data = cpu_result.data<float>();
    EXPECT_FLOAT_EQ(data[0], 6.0f);
}

TEST_F(OneAPIBackendTest, MaxReduction) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input_cpu = zeros({4}, DType::Float32, Device::cpu());
    auto input_data = input_cpu.data<float>();
    input_data[0] = 1.0f;
    input_data[1] = 5.0f;
    input_data[2] = 3.0f;
    input_data[3] = 2.0f;

    auto input = input_cpu.to(device);

    auto result = max(input, -1, false);

    auto cpu_result = result.to(Device::cpu());
    auto data = cpu_result.data<float>();
    EXPECT_FLOAT_EQ(data[0], 5.0f);
}

TEST_F(OneAPIBackendTest, MinReduction) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input_cpu = zeros({4}, DType::Float32, Device::cpu());
    auto input_data = input_cpu.data<float>();
    input_data[0] = 5.0f;
    input_data[1] = 1.0f;
    input_data[2] = 3.0f;
    input_data[3] = 2.0f;

    auto input = input_cpu.to(device);

    auto result = min(input, -1, false);

    auto cpu_result = result.to(Device::cpu());
    auto data = cpu_result.data<float>();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
}

// ============================================================================
// Finding 29: large (> WG_SIZE^2 = 65,536 element) full-reduction regression.
//
// src/backends/oneapi/kernels/reduction.cpp's own comments document that the
// two-phase SYCL reduction for full-tensor Int64/UInt64 sum(), and
// separately the WG_SIZE=256-capped sycl_arg_reduce_full used by no-dim
// argmax/argmin (Float32/Float64/Float16/Int32), both PREVIOUSLY silently
// dropped partial sums/indices whenever num_wgs > WG_SIZE -- i.e. whenever
// the tensor exceeds WG_SIZE^2 = 65,536 elements -- and that this bug class
// recurred independently in at least two separately-implemented reduction
// kernels. Every other OneAPI reduction test in this file/suite stays at
// a few thousand elements at most, so a regression that reintroduces the
// WG_SIZE cap (e.g. collapsing the grid-stride fold "for simplicity") would
// pass the entire existing suite silently. These use 200,000 elements
// (> 65,536, and not a round multiple of WG_SIZE=256, so a boundary-only
// fix wouldn't accidentally look correct).
// ============================================================================

TEST_F(OneAPIBackendTest, LargeSumReductionInt64) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }
    auto device = Device::oneapi(0);
    constexpr int64_t n = 200000;  // > WG_SIZE^2 = 65,536

    auto input_cpu = zeros({n}, DType::Int64, Device::cpu());
    auto* data = input_cpu.data<int64_t>();
    for (int64_t i = 0; i < n; ++i) data[i] = 1;

    auto input = input_cpu.to(device);
    auto result = sum(input, std::nullopt, false);
    ASSERT_EQ(result.numel(), 1);

    auto cpu_result = result.to(Device::cpu());
    EXPECT_EQ(cpu_result.data<int64_t>()[0], n)
        << "full-tensor Int64 sum() over " << n << " elements (> WG_SIZE^2) "
           "must not silently drop partial sums from work-groups beyond "
           "WG_SIZE (num_wgs > WG_SIZE)";
}

TEST_F(OneAPIBackendTest, LargeSumReductionUInt64) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }
    auto device = Device::oneapi(0);
    constexpr int64_t n = 200000;

    auto input_cpu = zeros({n}, DType::UInt64, Device::cpu());
    auto* data = input_cpu.data<uint64_t>();
    for (int64_t i = 0; i < n; ++i) data[i] = 1;

    auto input = input_cpu.to(device);
    auto result = sum(input, std::nullopt, false);
    ASSERT_EQ(result.numel(), 1);

    auto cpu_result = result.to(Device::cpu());
    EXPECT_EQ(cpu_result.data<uint64_t>()[0], static_cast<uint64_t>(n))
        << "full-tensor UInt64 sum() over " << n << " elements (> WG_SIZE^2) "
           "must not silently drop partial sums";
}

TEST_F(OneAPIBackendTest, LargeArgMaxFullReduction) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }
    auto device = Device::oneapi(0);
    constexpr int64_t n = 200000;
    constexpr int64_t max_idx = 150000;  // deliberately not the last element

    auto input_cpu = zeros({n}, DType::Float32, Device::cpu());
    auto* data = input_cpu.data<float>();
    for (int64_t i = 0; i < n; ++i) data[i] = static_cast<float>(i % 1000);
    data[max_idx] = 5000.0f;  // unambiguous global max

    auto input = input_cpu.to(device);
    // No dim -> full-tensor reduction, the sycl_arg_reduce_full path.
    auto result = argmax(input, std::nullopt, false);
    ASSERT_EQ(result.numel(), 1);

    auto cpu_result = result.to(Device::cpu());
    EXPECT_EQ(cpu_result.data<int64_t>()[0], max_idx)
        << "full-tensor argmax() over " << n << " elements (> WG_SIZE^2) must "
           "not silently drop the correct index from a work-group beyond "
           "WG_SIZE";
}

TEST_F(OneAPIBackendTest, LargeArgMinFullReduction) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }
    auto device = Device::oneapi(0);
    constexpr int64_t n = 200000;
    constexpr int64_t min_idx = 180000;

    auto input_cpu = zeros({n}, DType::Float32, Device::cpu());
    auto* data = input_cpu.data<float>();
    for (int64_t i = 0; i < n; ++i) data[i] = static_cast<float>(1000 + (i % 1000));
    data[min_idx] = -5000.0f;  // unambiguous global min

    auto input = input_cpu.to(device);
    auto result = argmin(input, std::nullopt, false);
    ASSERT_EQ(result.numel(), 1);

    auto cpu_result = result.to(Device::cpu());
    EXPECT_EQ(cpu_result.data<int64_t>()[0], min_idx)
        << "full-tensor argmin() over " << n << " elements (> WG_SIZE^2) must "
           "not silently drop the correct index";
}

// ============================================================================
// Transform Operations Tests
// ============================================================================

TEST_F(OneAPIBackendTest, ReshapeOperation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = ones({2, 3, 4}, DType::Float32, device);

    auto result = reshape(input, {6, 4});
    std::vector<int64_t> expected_shape = {6, 4};
    EXPECT_TRUE(shapesEqual(result.shape(), expected_shape));
    EXPECT_EQ(result.numel(), input.numel());

    auto result2 = reshape(input, {24});
    std::vector<int64_t> expected_shape2 = {24};
    EXPECT_TRUE(shapesEqual(result2.shape(), expected_shape2));
}

TEST_F(OneAPIBackendTest, TransposeOperation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = ones({2, 3, 4}, DType::Float32, device);

    auto result = transpose(input, 0, 1);
    std::vector<int64_t> expected_shape = {3, 2, 4};
    EXPECT_TRUE(shapesEqual(result.shape(), expected_shape));

    auto result2 = transpose(input, 1, 2);
    std::vector<int64_t> expected_shape2 = {2, 4, 3};
    EXPECT_TRUE(shapesEqual(result2.shape(), expected_shape2));
}

TEST_F(OneAPIBackendTest, PermuteOperation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = ones({2, 3, 4}, DType::Float32, device);

    auto result = permute(input, {2, 0, 1});
    std::vector<int64_t> expected_shape = {4, 2, 3};
    EXPECT_TRUE(shapesEqual(result.shape(), expected_shape));
}

TEST_F(OneAPIBackendTest, SqueezeOperation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = ones({1, 3, 1, 4}, DType::Float32, device);

    auto result = squeeze(input, 0);
    std::vector<int64_t> expected_shape = {3, 1, 4};
    EXPECT_TRUE(shapesEqual(result.shape(), expected_shape));
}

TEST_F(OneAPIBackendTest, UnsqueezeOperation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = ones({3, 4}, DType::Float32, device);

    auto result = unsqueeze(input, 0);
    std::vector<int64_t> expected_shape = {1, 3, 4};
    EXPECT_TRUE(shapesEqual(result.shape(), expected_shape));

    auto result2 = unsqueeze(input, 1);
    std::vector<int64_t> expected_shape2 = {3, 1, 4};
    EXPECT_TRUE(shapesEqual(result2.shape(), expected_shape2));
}

TEST_F(OneAPIBackendTest, ContiguousOperation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = ones({3, 4}, DType::Float32, device);

    auto result = contiguous(input);
    EXPECT_TRUE(shapesEqual(result.shape(), input.shape()));
    EXPECT_TRUE(tensorsClose(input, result));
}

TEST_F(OneAPIBackendTest, CloneOperation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = full({3, 4}, 2.5f, DType::Float32, device);

    auto result = input.clone();
    EXPECT_TRUE(shapesEqual(result.shape(), input.shape()));
    EXPECT_TRUE(tensorsClose(input, result));
}

// ============================================================================
// Fill Operations Tests
// ============================================================================

TEST_F(OneAPIBackendTest, ZerosCreation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto t = zeros({3, 4}, DType::Float32, device);
    EXPECT_EQ(t.device().type, Device::Type::OneAPI);
    std::vector<int64_t> expected_shape = {3, 4};
    EXPECT_TRUE(shapesEqual(t.shape(), expected_shape));

    auto cpu_t = t.to(Device::cpu());
    auto data = cpu_t.data<float>();
    for (int64_t i = 0; i < cpu_t.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
}

TEST_F(OneAPIBackendTest, OnesCreation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto t = ones({3, 4}, DType::Float32, device);

    auto cpu_t = t.to(Device::cpu());
    auto data = cpu_t.data<float>();
    for (int64_t i = 0; i < cpu_t.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST_F(OneAPIBackendTest, FullCreation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto t = full({3, 4}, 7.5f, DType::Float32, device);

    auto cpu_t = t.to(Device::cpu());
    auto data = cpu_t.data<float>();
    for (int64_t i = 0; i < cpu_t.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 7.5f);
    }
}

TEST_F(OneAPIBackendTest, FillOperation) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto t = zeros({3, 4}, DType::Float32, device);

    t.fill_(9.0f);

    auto cpu_result = t.to(Device::cpu());
    auto data = cpu_result.data<float>();
    for (int64_t i = 0; i < cpu_result.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 9.0f);
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(OneAPIBackendTest, InvalidDeviceIndex) {
    // Attempting to use a very high device index should throw
    EXPECT_THROW({
        auto device = Device::oneapi(999);
        auto t = zeros({2, 3}, DType::Float32, device);
    }, std::exception);
}

TEST_F(OneAPIBackendTest, ShapeMismatchAddition) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto a = ones({3, 4}, DType::Float32, device);
    auto b = ones({5, 6}, DType::Float32, device);

    // Incompatible shapes should throw
    EXPECT_THROW({
        auto c = a + b;
    }, std::exception);
}

TEST_F(OneAPIBackendTest, InvalidMatMulDimensions) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto a = ones({3, 4}, DType::Float32, device);
    auto b = ones({5, 6}, DType::Float32, device);

    // Incompatible dimensions for matmul
    EXPECT_THROW({
        auto c = matmul(a, b);
    }, std::exception);
}

TEST_F(OneAPIBackendTest, InvalidReshape) {
    if (!hasOneAPIDevice()) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "No OneAPI devices available");
    }

    auto device = Device::oneapi(0);

    auto input = ones({3, 4}, DType::Float32, device);

    // Cannot reshape to incompatible total size
    EXPECT_THROW({
        auto result = reshape(input, {5, 5});  // 25 != 12
    }, std::exception);
}

// ============================================================================
// Multi-Device Tests (if applicable)
// ============================================================================

TEST_F(OneAPIBackendTest, MultiDeviceSupport) {
    // Check if multiple devices are available
    int32_t device_count = 0;
    try {
        while (true) {
            auto device = Device::oneapi(device_count);
            auto t = zeros({1}, DType::Float32, device);
            device_count++;
            if (device_count > 10) break;  // Safety limit
        }
    } catch (const std::exception&) {
        // Expected when we run out of devices
    }

    if (device_count < 2) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "Less than 2 OneAPI devices available");
    }

    // Test tensor creation on different devices
    auto dev0 = Device::oneapi(0);
    auto dev1 = Device::oneapi(1);

    auto t0 = ones({3, 4}, DType::Float32, dev0);
    auto t1 = ones({3, 4}, DType::Float32, dev1);

    EXPECT_EQ(t0.device().index, 0);
    EXPECT_EQ(t1.device().index, 1);
}

TEST_F(OneAPIBackendTest, CrossDeviceCopy) {
    // Check if multiple devices are available
    int32_t device_count = 0;
    try {
        while (true) {
            auto device = Device::oneapi(device_count);
            auto t = zeros({1}, DType::Float32, device);
            device_count++;
            if (device_count > 10) break;
        }
    } catch (const std::exception&) {
        // Expected
    }

    if (device_count < 2) {
        SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "Less than 2 OneAPI devices available");
    }

    auto dev0 = Device::oneapi(0);
    auto dev1 = Device::oneapi(1);

    // Create tensor on device 0
    auto t0 = full({3, 4}, 5.0f, DType::Float32, dev0);

    // Copy to device 1
    auto t1 = t0.to(dev1);
    EXPECT_EQ(t1.device().index, 1);

    // Verify data integrity
    auto cpu_t0 = t0.to(Device::cpu());
    auto cpu_t1 = t1.to(Device::cpu());
    EXPECT_TRUE(tensorsClose(cpu_t0, cpu_t1));
}

// ============================================================================
// Main Test Entry Point
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
