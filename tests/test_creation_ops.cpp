#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <cmath>

using namespace tenzor;

// Test zeros operation
TEST(CreationOpsTest, Zeros) {
    auto t = zeros({3, 4}, DType::Float32, Device{Device::Type::CPU, 0});

    ASSERT_EQ(t.shape()[0], 3);
    ASSERT_EQ(t.shape()[1], 4);
    ASSERT_EQ(t.dtype(), DType::Float32);

    // Check all elements are zero
    const float* data = t.data<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_EQ(data[i], 0.0f);
    }
}

TEST(CreationOpsTest, ZerosFloat64) {
    auto t = zeros({2, 3, 4}, DType::Float64, Device{Device::Type::CPU, 0});

    ASSERT_EQ(t.ndim(), 3);
    ASSERT_EQ(t.shape()[0], 2);
    ASSERT_EQ(t.shape()[1], 3);
    ASSERT_EQ(t.shape()[2], 4);

    const double* data = t.data<double>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_EQ(data[i], 0.0);
    }
}

// Test ones operation
TEST(CreationOpsTest, Ones) {
    auto t = ones({3, 4}, DType::Float32, Device{Device::Type::CPU, 0});

    ASSERT_EQ(t.shape()[0], 3);
    ASSERT_EQ(t.shape()[1], 4);
    ASSERT_EQ(t.dtype(), DType::Float32);

    // Check all elements are one
    const float* data = t.data<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_EQ(data[i], 1.0f);
    }
}

TEST(CreationOpsTest, OnesInt32) {
    auto t = ones({5, 5}, DType::Int32, Device{Device::Type::CPU, 0});

    ASSERT_EQ(t.dtype(), DType::Int32);

    const int32_t* data = t.data<int32_t>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_EQ(data[i], 1);
    }
}

// Test rand operation
TEST(CreationOpsTest, Rand) {
    auto t = rand({100, 100}, DType::Float32, Device{Device::Type::CPU, 0});

    ASSERT_EQ(t.shape()[0], 100);
    ASSERT_EQ(t.shape()[1], 100);
    ASSERT_EQ(t.dtype(), DType::Float32);

    // Check values are in [0, 1)
    const float* data = t.data<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        EXPECT_GE(data[i], 0.0f);
        EXPECT_LT(data[i], 1.0f);
    }

    // Check that not all values are the same (with very high probability)
    bool all_same = true;
    float first = data[0];
    for (int64_t i = 1; i < t.numel(); ++i) {
        if (data[i] != first) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);
}

// Test randn operation
TEST(CreationOpsTest, Randn) {
    auto t = randn({100, 100}, DType::Float32, Device{Device::Type::CPU, 0});

    ASSERT_EQ(t.shape()[0], 100);
    ASSERT_EQ(t.shape()[1], 100);
    ASSERT_EQ(t.dtype(), DType::Float32);

    // Calculate mean and std to verify it's roughly N(0,1)
    const float* data = t.data<float>();
    double sum = 0.0;
    for (int64_t i = 0; i < t.numel(); ++i) {
        sum += data[i];
    }
    double mean = sum / t.numel();

    double variance_sum = 0.0;
    for (int64_t i = 0; i < t.numel(); ++i) {
        double diff = data[i] - mean;
        variance_sum += diff * diff;
    }
    double std = std::sqrt(variance_sum / t.numel());

    // Mean should be close to 0 (within 0.1 for 10000 samples)
    EXPECT_NEAR(mean, 0.0, 0.1);

    // Std should be close to 1 (within 0.1 for 10000 samples)
    EXPECT_NEAR(std, 1.0, 0.1);
}

TEST(CreationOpsTest, RandnFloat64) {
    auto t = randn({50, 50}, DType::Float64, Device{Device::Type::CPU, 0});

    ASSERT_EQ(t.dtype(), DType::Float64);

    const double* data = t.data<double>();
    double sum = 0.0;
    for (int64_t i = 0; i < t.numel(); ++i) {
        sum += data[i];
    }
    double mean = sum / t.numel();

    EXPECT_NEAR(mean, 0.0, 0.15);
}

// Test large tensors
TEST(CreationOpsTest, LargeZeros) {
    auto t = zeros({1000, 1000}, DType::Float32, Device{Device::Type::CPU, 0});

    ASSERT_EQ(t.numel(), 1000000);

    // Spot check some values
    const float* data = t.data<float>();
    EXPECT_EQ(data[0], 0.0f);
    EXPECT_EQ(data[500000], 0.0f);
    EXPECT_EQ(data[999999], 0.0f);
}

TEST(CreationOpsTest, LargeOnes) {
    auto t = ones({500, 2000}, DType::Float32, Device{Device::Type::CPU, 0});

    ASSERT_EQ(t.numel(), 1000000);

    // Spot check some values
    const float* data = t.data<float>();
    EXPECT_EQ(data[0], 1.0f);
    EXPECT_EQ(data[500000], 1.0f);
    EXPECT_EQ(data[999999], 1.0f);
}

// Test different shapes
TEST(CreationOpsTest, ScalarShape) {
    auto t = zeros({1}, DType::Float32, Device{Device::Type::CPU, 0});
    ASSERT_EQ(t.numel(), 1);
    EXPECT_EQ(t.data<float>()[0], 0.0f);
}

TEST(CreationOpsTest, HighDimensional) {
    auto t = ones({2, 3, 4, 5}, DType::Float32, Device{Device::Type::CPU, 0});

    ASSERT_EQ(t.ndim(), 4);
    ASSERT_EQ(t.numel(), 2 * 3 * 4 * 5);

    // Spot check
    const float* data = t.data<float>();
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(data[i * 12], 1.0f);
    }
}
