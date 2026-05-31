#include <gtest/gtest.h>
#include "backend_test_fixture.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <cmath>

using namespace tenzor;

class CreationOpsTest : public ::tenzor::testing::BackendTest {};

// Test zeros operation
TEST_P(CreationOpsTest, Zeros) {
    auto t = zeros({3, 4}, DType::Float32, device);

    ASSERT_EQ(t.shape()[0], 3);
    ASSERT_EQ(t.shape()[1], 4);
    ASSERT_EQ(t.dtype(), DType::Float32);

    // Check all elements are zero
    auto t_cpu = t.cpu();
    const float* data = t_cpu.data<float>();
    for (int64_t i = 0; i < t_cpu.numel(); ++i) {
        EXPECT_EQ(data[i], 0.0f);
    }
}

TEST_P(CreationOpsTest, ZerosFloat64) {
    auto t = zeros({2, 3, 4}, DType::Float64, device);

    ASSERT_EQ(t.ndim(), 3);
    ASSERT_EQ(t.shape()[0], 2);
    ASSERT_EQ(t.shape()[1], 3);
    ASSERT_EQ(t.shape()[2], 4);

    auto t_cpu = t.cpu();
    const double* data = t_cpu.data<double>();
    for (int64_t i = 0; i < t_cpu.numel(); ++i) {
        EXPECT_EQ(data[i], 0.0);
    }
}

// Test ones operation
TEST_P(CreationOpsTest, Ones) {
    auto t = ones({3, 4}, DType::Float32, device);

    ASSERT_EQ(t.shape()[0], 3);
    ASSERT_EQ(t.shape()[1], 4);
    ASSERT_EQ(t.dtype(), DType::Float32);

    // Check all elements are one
    auto t_cpu = t.cpu();
    const float* data = t_cpu.data<float>();
    for (int64_t i = 0; i < t_cpu.numel(); ++i) {
        EXPECT_EQ(data[i], 1.0f);
    }
}

TEST_P(CreationOpsTest, OnesInt32) {
    auto t = ones({5, 5}, DType::Int32, device);

    ASSERT_EQ(t.dtype(), DType::Int32);

    auto t_cpu = t.cpu();
    const int32_t* data = t_cpu.data<int32_t>();
    for (int64_t i = 0; i < t_cpu.numel(); ++i) {
        EXPECT_EQ(data[i], 1);
    }
}

// Test rand operation
TEST_P(CreationOpsTest, Rand) {
    auto t = rand({100, 100}, DType::Float32, device);

    ASSERT_EQ(t.shape()[0], 100);
    ASSERT_EQ(t.shape()[1], 100);
    ASSERT_EQ(t.dtype(), DType::Float32);

    // Check values are in [0, 1)
    auto t_cpu = t.cpu();
    const float* data = t_cpu.data<float>();
    for (int64_t i = 0; i < t_cpu.numel(); ++i) {
        EXPECT_GE(data[i], 0.0f);
        EXPECT_LT(data[i], 1.0f);
    }

    // Check that not all values are the same (with very high probability)
    bool all_same = true;
    float first = data[0];
    for (int64_t i = 1; i < t_cpu.numel(); ++i) {
        if (data[i] != first) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);
}

// Test randn operation
TEST_P(CreationOpsTest, Randn) {
    auto t = randn({100, 100}, DType::Float32, device);

    ASSERT_EQ(t.shape()[0], 100);
    ASSERT_EQ(t.shape()[1], 100);
    ASSERT_EQ(t.dtype(), DType::Float32);

    // Calculate mean and std to verify it's roughly N(0,1)
    auto t_cpu = t.cpu();
    const float* data = t_cpu.data<float>();
    double sum = 0.0;
    for (int64_t i = 0; i < t_cpu.numel(); ++i) {
        sum += data[i];
    }
    double mean = sum / t_cpu.numel();

    double variance_sum = 0.0;
    for (int64_t i = 0; i < t_cpu.numel(); ++i) {
        double diff = data[i] - mean;
        variance_sum += diff * diff;
    }
    double std = std::sqrt(variance_sum / t_cpu.numel());

    // Mean should be close to 0 (within 0.1 for 10000 samples)
    // reason: randn finite-sample variance; 3σ for N=10000 (~3/sqrt(N)=0.03,
    // bound padded to 0.1 for CI stability)
    EXPECT_NEAR(mean, 0.0, 0.1);

    // Std should be close to 1 (within 0.1 for 10000 samples)
    // reason: randn finite-sample variance; 3σ for N=10000
    EXPECT_NEAR(std, 1.0, 0.1);
}

TEST_P(CreationOpsTest, RandnFloat64) {
    auto t = randn({50, 50}, DType::Float64, device);

    ASSERT_EQ(t.dtype(), DType::Float64);

    auto t_cpu = t.cpu();
    const double* data = t_cpu.data<double>();
    double sum = 0.0;
    for (int64_t i = 0; i < t_cpu.numel(); ++i) {
        sum += data[i];
    }
    double mean = sum / t_cpu.numel();

    EXPECT_NEAR(mean, 0.0, 0.15);
}

// Test large tensors
TEST_P(CreationOpsTest, LargeZeros) {
    auto t = zeros({1000, 1000}, DType::Float32, device);

    ASSERT_EQ(t.numel(), 1000000);

    // Spot check some values
    auto t_cpu = t.cpu();
    const float* data = t_cpu.data<float>();
    EXPECT_EQ(data[0], 0.0f);
    EXPECT_EQ(data[500000], 0.0f);
    EXPECT_EQ(data[999999], 0.0f);
}

TEST_P(CreationOpsTest, LargeOnes) {
    auto t = ones({500, 2000}, DType::Float32, device);

    ASSERT_EQ(t.numel(), 1000000);

    // Spot check some values
    auto t_cpu = t.cpu();
    const float* data = t_cpu.data<float>();
    EXPECT_EQ(data[0], 1.0f);
    EXPECT_EQ(data[500000], 1.0f);
    EXPECT_EQ(data[999999], 1.0f);
}

// Test different shapes
TEST_P(CreationOpsTest, ScalarShape) {
    auto t = zeros({1}, DType::Float32, device);
    ASSERT_EQ(t.numel(), 1);
    auto t_cpu = t.cpu();
    EXPECT_EQ(t_cpu.data<float>()[0], 0.0f);
}

TEST_P(CreationOpsTest, HighDimensional) {
    auto t = ones({2, 3, 4, 5}, DType::Float32, device);

    ASSERT_EQ(t.ndim(), 4);
    ASSERT_EQ(t.numel(), 2 * 3 * 4 * 5);

    // Spot check
    auto t_cpu = t.cpu();
    const float* data = t_cpu.data<float>();
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(data[i * 12], 1.0f);
    }
}

// Test randint operation
TEST_P(CreationOpsTest, Randint) {
    auto t = randint(0, 10, {100, 100}, DType::Int64, device);

    ASSERT_EQ(t.shape()[0], 100);
    ASSERT_EQ(t.shape()[1], 100);
    ASSERT_EQ(t.dtype(), DType::Int64);

    // Check values are in [0, 10)
    auto t_cpu = t.cpu();
    const int64_t* data = t_cpu.data<int64_t>();
    for (int64_t i = 0; i < t_cpu.numel(); ++i) {
        EXPECT_GE(data[i], 0);
        EXPECT_LT(data[i], 10);
    }

    // Check that not all values are the same (with very high probability)
    bool all_same = true;
    int64_t first = data[0];
    for (int64_t i = 1; i < t_cpu.numel(); ++i) {
        if (data[i] != first) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);
}

TEST_P(CreationOpsTest, RandintInt32) {
    auto t = randint(5, 15, {50, 50}, DType::Int32, device);

    ASSERT_EQ(t.dtype(), DType::Int32);

    auto t_cpu = t.cpu();
    const int32_t* data = t_cpu.data<int32_t>();
    for (int64_t i = 0; i < t_cpu.numel(); ++i) {
        EXPECT_GE(data[i], 5);
        EXPECT_LT(data[i], 15);
    }
}

TEST_P(CreationOpsTest, RandintNegativeRange) {
    auto t = randint(-10, 10, {100}, DType::Int64, device);

    auto t_cpu = t.cpu();
    const int64_t* data = t_cpu.data<int64_t>();
    bool has_negative = false;
    bool has_positive = false;
    for (int64_t i = 0; i < t_cpu.numel(); ++i) {
        EXPECT_GE(data[i], -10);
        EXPECT_LT(data[i], 10);
        if (data[i] < 0) has_negative = true;
        if (data[i] > 0) has_positive = true;
    }
    // With 100 samples from [-10, 10), we should almost certainly have both
    EXPECT_TRUE(has_negative);
    EXPECT_TRUE(has_positive);
}

// Test arange operation
TEST_P(CreationOpsTest, Arange) {
    auto t = arange(0.0f, 10.0f, 1.0f, DType::Float32, device);

    ASSERT_EQ(t.ndim(), 1);
    ASSERT_EQ(t.numel(), 10);
    ASSERT_EQ(t.dtype(), DType::Float32);

    auto t_cpu = t.cpu();
    const float* data = t_cpu.data<float>();
    for (int i = 0; i < 10; ++i) {
        EXPECT_NEAR(data[i], static_cast<float>(i), 1e-6f);
    }
}

TEST_P(CreationOpsTest, ArangeWithStep) {
    auto t = arange(0.0f, 10.0f, 2.0f, DType::Float32, device);

    ASSERT_EQ(t.numel(), 5);  // [0, 2, 4, 6, 8]

    auto t_cpu = t.cpu();
    const float* data = t_cpu.data<float>();
    EXPECT_NEAR(data[0], 0.0f, 1e-6f);
    EXPECT_NEAR(data[1], 2.0f, 1e-6f);
    EXPECT_NEAR(data[2], 4.0f, 1e-6f);
    EXPECT_NEAR(data[3], 6.0f, 1e-6f);
    EXPECT_NEAR(data[4], 8.0f, 1e-6f);
}

TEST_P(CreationOpsTest, ArangeFloat64) {
    auto t = arange(1.0f, 5.0f, 0.5f, DType::Float64, device);

    ASSERT_EQ(t.dtype(), DType::Float64);
    ASSERT_EQ(t.numel(), 8);  // [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

    auto t_cpu = t.cpu();
    const double* data = t_cpu.data<double>();
    for (int i = 0; i < 8; ++i) {
        double expected = 1.0 + i * 0.5;
        EXPECT_NEAR(data[i], expected, 1e-10);
    }
}

TEST_P(CreationOpsTest, ArangeNegativeStep) {
    auto t = arange(10.0f, 0.0f, -2.0f, DType::Float32, device);

    ASSERT_EQ(t.numel(), 5);  // [10, 8, 6, 4, 2]

    auto t_cpu = t.cpu();
    const float* data = t_cpu.data<float>();
    EXPECT_NEAR(data[0], 10.0f, 1e-6f);
    EXPECT_NEAR(data[1], 8.0f, 1e-6f);
    EXPECT_NEAR(data[2], 6.0f, 1e-6f);
    EXPECT_NEAR(data[3], 4.0f, 1e-6f);
    EXPECT_NEAR(data[4], 2.0f, 1e-6f);
}

TEST_P(CreationOpsTest, ArangeFractional) {
    auto t = arange(0.0f, 1.0f, 0.1f, DType::Float32, device);

    ASSERT_EQ(t.numel(), 10);  // [0.0, 0.1, 0.2, ..., 0.9]

    auto t_cpu = t.cpu();
    const float* data = t_cpu.data<float>();
    for (int i = 0; i < 10; ++i) {
        EXPECT_NEAR(data[i], i * 0.1f, 1e-5f);
    }
}

INSTANTIATE_BACKEND_TESTS(CreationOpsTest);
