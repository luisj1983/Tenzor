/**
 * @file test_rocm_reduction.cpp
 * @brief Unit tests for ROCm reduction kernels
 *
 * Tests sum, mean, max, and min reduction operations on AMD GPUs using HIP.
 */

#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/device.hpp"
#include <cmath>
#include <vector>

using namespace tenzor;

// Test fixture for ROCm reduction operations
class ROCmReductionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if ROCm device is available
        try {
            Device rocm_device(DeviceType::ROCm, 0);
            rocm_available = true;
        } catch (...) {
            rocm_available = false;
            GTEST_SKIP() << "ROCm device not available, skipping tests";
        }
    }

    bool rocm_available = false;
};

// ============================================================================
// Full Reduction Tests (dim = -1)
// ============================================================================

TEST_F(ROCmReductionTest, SumFullReduction) {
    if (!rocm_available) GTEST_SKIP();

    Device rocm_device(DeviceType::ROCm, 0);

    // Create test tensor: [1, 2, 3, 4, 5, 6]
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor cpu_tensor({2, 3}, DType::Float32, Device(DeviceType::CPU));
    std::copy(data.begin(), data.end(), cpu_tensor.data<float>());

    // Transfer to ROCm
    Tensor rocm_tensor = cpu_tensor.to(rocm_device);

    // Compute sum (should be 21.0)
    Tensor result = rocm_tensor.sum();

    // Transfer back to CPU for verification
    Tensor cpu_result = result.to(Device(DeviceType::CPU));

    EXPECT_NEAR(cpu_result.data<float>()[0], 21.0f, 1e-5f);
}

TEST_F(ROCmReductionTest, MeanFullReduction) {
    if (!rocm_available) GTEST_SKIP();

    Device rocm_device(DeviceType::ROCm, 0);

    // Create test tensor: [1, 2, 3, 4, 5, 6]
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor cpu_tensor({2, 3}, DType::Float32, Device(DeviceType::CPU));
    std::copy(data.begin(), data.end(), cpu_tensor.data<float>());

    // Transfer to ROCm
    Tensor rocm_tensor = cpu_tensor.to(rocm_device);

    // Compute mean (should be 3.5)
    Tensor result = rocm_tensor.mean();

    // Transfer back to CPU for verification
    Tensor cpu_result = result.to(Device(DeviceType::CPU));

    EXPECT_NEAR(cpu_result.data<float>()[0], 3.5f, 1e-5f);
}

TEST_F(ROCmReductionTest, MaxFullReduction) {
    if (!rocm_available) GTEST_SKIP();

    Device rocm_device(DeviceType::ROCm, 0);

    // Create test tensor
    std::vector<float> data = {1.0f, 5.0f, 3.0f, 9.0f, 2.0f, 7.0f};
    Tensor cpu_tensor({2, 3}, DType::Float32, Device(DeviceType::CPU));
    std::copy(data.begin(), data.end(), cpu_tensor.data<float>());

    // Transfer to ROCm
    Tensor rocm_tensor = cpu_tensor.to(rocm_device);

    // Compute max (should be 9.0)
    Tensor result = rocm_tensor.max();

    // Transfer back to CPU for verification
    Tensor cpu_result = result.to(Device(DeviceType::CPU));

    EXPECT_NEAR(cpu_result.data<float>()[0], 9.0f, 1e-5f);
}

TEST_F(ROCmReductionTest, MinFullReduction) {
    if (!rocm_available) GTEST_SKIP();

    Device rocm_device(DeviceType::ROCm, 0);

    // Create test tensor
    std::vector<float> data = {5.0f, 2.0f, 8.0f, 1.0f, 9.0f, 3.0f};
    Tensor cpu_tensor({2, 3}, DType::Float32, Device(DeviceType::CPU));
    std::copy(data.begin(), data.end(), cpu_tensor.data<float>());

    // Transfer to ROCm
    Tensor rocm_tensor = cpu_tensor.to(rocm_device);

    // Compute min (should be 1.0)
    Tensor result = rocm_tensor.min();

    // Transfer back to CPU for verification
    Tensor cpu_result = result.to(Device(DeviceType::CPU));

    EXPECT_NEAR(cpu_result.data<float>()[0], 1.0f, 1e-5f);
}

// ============================================================================
// Dimensional Reduction Tests
// ============================================================================

TEST_F(ROCmReductionTest, SumAlongDimension) {
    if (!rocm_available) GTEST_SKIP();

    Device rocm_device(DeviceType::ROCm, 0);

    // Create 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor cpu_tensor({2, 3}, DType::Float32, Device(DeviceType::CPU));
    std::copy(data.begin(), data.end(), cpu_tensor.data<float>());

    Tensor rocm_tensor = cpu_tensor.to(rocm_device);

    // Sum along dimension 0 (should be [5, 7, 9])
    Tensor result = rocm_tensor.sum(0);
    Tensor cpu_result = result.to(Device(DeviceType::CPU));

    EXPECT_EQ(cpu_result.shape()[0], 3);
    EXPECT_NEAR(cpu_result.data<float>()[0], 5.0f, 1e-5f);
    EXPECT_NEAR(cpu_result.data<float>()[1], 7.0f, 1e-5f);
    EXPECT_NEAR(cpu_result.data<float>()[2], 9.0f, 1e-5f);

    // Sum along dimension 1 (should be [6, 15])
    result = rocm_tensor.sum(1);
    cpu_result = result.to(Device(DeviceType::CPU));

    EXPECT_EQ(cpu_result.shape()[0], 2);
    EXPECT_NEAR(cpu_result.data<float>()[0], 6.0f, 1e-5f);
    EXPECT_NEAR(cpu_result.data<float>()[1], 15.0f, 1e-5f);
}

TEST_F(ROCmReductionTest, MeanAlongDimension) {
    if (!rocm_available) GTEST_SKIP();

    Device rocm_device(DeviceType::ROCm, 0);

    // Create 2x3 tensor: [[1, 2, 3], [4, 5, 6]]
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor cpu_tensor({2, 3}, DType::Float32, Device(DeviceType::CPU));
    std::copy(data.begin(), data.end(), cpu_tensor.data<float>());

    Tensor rocm_tensor = cpu_tensor.to(rocm_device);

    // Mean along dimension 0 (should be [2.5, 3.5, 4.5])
    Tensor result = rocm_tensor.mean(0);
    Tensor cpu_result = result.to(Device(DeviceType::CPU));

    EXPECT_EQ(cpu_result.shape()[0], 3);
    EXPECT_NEAR(cpu_result.data<float>()[0], 2.5f, 1e-5f);
    EXPECT_NEAR(cpu_result.data<float>()[1], 3.5f, 1e-5f);
    EXPECT_NEAR(cpu_result.data<float>()[2], 4.5f, 1e-5f);
}

TEST_F(ROCmReductionTest, MaxAlongDimension) {
    if (!rocm_available) GTEST_SKIP();

    Device rocm_device(DeviceType::ROCm, 0);

    // Create 2x3 tensor: [[1, 5, 3], [4, 2, 6]]
    std::vector<float> data = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
    Tensor cpu_tensor({2, 3}, DType::Float32, Device(DeviceType::CPU));
    std::copy(data.begin(), data.end(), cpu_tensor.data<float>());

    Tensor rocm_tensor = cpu_tensor.to(rocm_device);

    // Max along dimension 0 (should be [4, 5, 6])
    Tensor result = rocm_tensor.max(0);
    Tensor cpu_result = result.to(Device(DeviceType::CPU));

    EXPECT_EQ(cpu_result.shape()[0], 3);
    EXPECT_NEAR(cpu_result.data<float>()[0], 4.0f, 1e-5f);
    EXPECT_NEAR(cpu_result.data<float>()[1], 5.0f, 1e-5f);
    EXPECT_NEAR(cpu_result.data<float>()[2], 6.0f, 1e-5f);
}

TEST_F(ROCmReductionTest, MinAlongDimension) {
    if (!rocm_available) GTEST_SKIP();

    Device rocm_device(DeviceType::ROCm, 0);

    // Create 2x3 tensor: [[5, 2, 8], [3, 9, 1]]
    std::vector<float> data = {5.0f, 2.0f, 8.0f, 3.0f, 9.0f, 1.0f};
    Tensor cpu_tensor({2, 3}, DType::Float32, Device(DeviceType::CPU));
    std::copy(data.begin(), data.end(), cpu_tensor.data<float>());

    Tensor rocm_tensor = cpu_tensor.to(rocm_device);

    // Min along dimension 0 (should be [3, 2, 1])
    Tensor result = rocm_tensor.min(0);
    Tensor cpu_result = result.to(Device(DeviceType::CPU));

    EXPECT_EQ(cpu_result.shape()[0], 3);
    EXPECT_NEAR(cpu_result.data<float>()[0], 3.0f, 1e-5f);
    EXPECT_NEAR(cpu_result.data<float>()[1], 2.0f, 1e-5f);
    EXPECT_NEAR(cpu_result.data<float>()[2], 1.0f, 1e-5f);
}

// ============================================================================
// KeepDim Tests
// ============================================================================

TEST_F(ROCmReductionTest, SumKeepDim) {
    if (!rocm_available) GTEST_SKIP();

    Device rocm_device(DeviceType::ROCm, 0);

    // Create 2x3 tensor
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor cpu_tensor({2, 3}, DType::Float32, Device(DeviceType::CPU));
    std::copy(data.begin(), data.end(), cpu_tensor.data<float>());

    Tensor rocm_tensor = cpu_tensor.to(rocm_device);

    // Sum with keepdim=true
    Tensor result = rocm_tensor.sum(0, true);

    // Should have shape [1, 3]
    EXPECT_EQ(result.ndim(), 2);
    EXPECT_EQ(result.shape()[0], 1);
    EXPECT_EQ(result.shape()[1], 3);

    Tensor cpu_result = result.to(Device(DeviceType::CPU));
    EXPECT_NEAR(cpu_result.data<float>()[0], 5.0f, 1e-5f);
    EXPECT_NEAR(cpu_result.data<float>()[1], 7.0f, 1e-5f);
    EXPECT_NEAR(cpu_result.data<float>()[2], 9.0f, 1e-5f);
}

// ============================================================================
// Large Tensor Tests (stress test for two-phase reduction)
// ============================================================================

TEST_F(ROCmReductionTest, LargeTensorSum) {
    if (!rocm_available) GTEST_SKIP();

    Device rocm_device(DeviceType::ROCm, 0);

    // Create large tensor (1M elements)
    int64_t n = 1024 * 1024;
    std::vector<float> data(n, 1.0f);
    Tensor cpu_tensor({n}, DType::Float32, Device(DeviceType::CPU));
    std::copy(data.begin(), data.end(), cpu_tensor.data<float>());

    Tensor rocm_tensor = cpu_tensor.to(rocm_device);

    // Compute sum
    Tensor result = rocm_tensor.sum();
    Tensor cpu_result = result.to(Device(DeviceType::CPU));

    // Should be approximately 1M (allow for floating point error)
    EXPECT_NEAR(cpu_result.data<float>()[0], static_cast<float>(n), 1e-3f);
}

// ============================================================================
// Data Type Tests
// ============================================================================

TEST_F(ROCmReductionTest, SumFloat64) {
    if (!rocm_available) GTEST_SKIP();

    Device rocm_device(DeviceType::ROCm, 0);

    // Create double precision tensor
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Tensor cpu_tensor({2, 3}, DType::Float64, Device(DeviceType::CPU));
    std::copy(data.begin(), data.end(), cpu_tensor.data<double>());

    Tensor rocm_tensor = cpu_tensor.to(rocm_device);

    // Compute sum
    Tensor result = rocm_tensor.sum();
    Tensor cpu_result = result.to(Device(DeviceType::CPU));

    EXPECT_NEAR(cpu_result.data<double>()[0], 21.0, 1e-10);
}

TEST_F(ROCmReductionTest, SumInt32) {
    if (!rocm_available) GTEST_SKIP();

    Device rocm_device(DeviceType::ROCm, 0);

    // Create int32 tensor
    std::vector<int32_t> data = {1, 2, 3, 4, 5, 6};
    Tensor cpu_tensor({2, 3}, DType::Int32, Device(DeviceType::CPU));
    std::copy(data.begin(), data.end(), cpu_tensor.data<int32_t>());

    Tensor rocm_tensor = cpu_tensor.to(rocm_device);

    // Compute sum
    Tensor result = rocm_tensor.sum();
    Tensor cpu_result = result.to(Device(DeviceType::CPU));

    EXPECT_EQ(cpu_result.data<int32_t>()[0], 21);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
