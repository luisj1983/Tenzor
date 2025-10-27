#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class OpsBackendTest : public BackendTest {};

TEST_P(OpsBackendTest, Zeros) {
    auto t = zeros({2, 3}, DType::Float32, device);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);

    auto t_cpu = t.to(Device::cpu());
    const float* data = t_cpu.data<float>();
    for (int i = 0; i < t_cpu.numel(); i++) {
        EXPECT_FLOAT_EQ(data[i], 0.0f) << "Failed on " << device.to_string();
    }
}

TEST_P(OpsBackendTest, Ones) {
    auto t = ones({3, 4}, DType::Float32, device);
    EXPECT_EQ(t.numel(), 12);

    auto t_cpu = t.to(Device::cpu());
    const float* data = t_cpu.data<float>();
    for (int i = 0; i < t_cpu.numel(); i++) {
        EXPECT_FLOAT_EQ(data[i], 1.0f) << "Failed on " << device.to_string();
    }
}

TEST_P(OpsBackendTest, Full) {
    auto t = full({2, 3}, 5.0f, DType::Float32, device);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_EQ(t.numel(), 6);

    auto t_cpu = t.to(Device::cpu());
    const float* data = t_cpu.data<float>();
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(data[i], 5.0f) << "Failed on " << device.to_string();
    }
}

TEST_P(OpsBackendTest, FullInt32) {
    auto t = full({3}, 42.0f, DType::Int32, device);
    EXPECT_EQ(t.numel(), 3);

    auto t_cpu = t.to(Device::cpu());
    const int32_t* data = t_cpu.data<int32_t>();
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(data[i], 42) << "Failed on " << device.to_string();
    }
}

TEST_P(OpsBackendTest, Arange) {
    auto t = arange(0, 5, 1, DType::Float32, device);
    EXPECT_EQ(t.numel(), 5);

    auto t_cpu = t.to(Device::cpu());
    const float* data = t_cpu.data<float>();
    for (int i = 0; i < 5; i++) {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i)) << "Failed on " << device.to_string();
    }
}

TEST_P(OpsBackendTest, ArangeStep) {
    auto t = arange(0, 10, 2, DType::Float32, device);
    EXPECT_EQ(t.numel(), 5);

    auto t_cpu = t.to(Device::cpu());
    const float* data = t_cpu.data<float>();
    EXPECT_FLOAT_EQ(data[0], 0.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[1], 2.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[2], 4.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[3], 6.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[4], 8.0f) << "Failed on " << device.to_string();
}

TEST_P(OpsBackendTest, ArangeFloat) {
    auto t = arange(0, 2, 0.5f, DType::Float32, device);
    EXPECT_EQ(t.numel(), 4);

    auto t_cpu = t.to(Device::cpu());
    const float* data = t_cpu.data<float>();
    EXPECT_FLOAT_EQ(data[0], 0.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[1], 0.5f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[2], 1.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[3], 1.5f) << "Failed on " << device.to_string();
}

TEST_P(OpsBackendTest, Linspace) {
    auto t = linspace(0, 1, 5, DType::Float32, device);
    EXPECT_EQ(t.numel(), 5);

    auto t_cpu = t.to(Device::cpu());
    const float* data = t_cpu.data<float>();
    EXPECT_FLOAT_EQ(data[0], 0.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[1], 0.25f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[2], 0.5f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[3], 0.75f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[4], 1.0f) << "Failed on " << device.to_string();
}

TEST_P(OpsBackendTest, LinspaceNegative) {
    auto t = linspace(-5, 5, 11, DType::Float32, device);
    EXPECT_EQ(t.numel(), 11);

    auto t_cpu = t.to(Device::cpu());
    const float* data = t_cpu.data<float>();
    EXPECT_FLOAT_EQ(data[0], -5.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[5], 0.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[10], 5.0f) << "Failed on " << device.to_string();
}

TEST_P(OpsBackendTest, Eye) {
    auto t = eye(3, std::nullopt, DType::Float32, device);
    EXPECT_EQ(t.shape()[0], 3);
    EXPECT_EQ(t.shape()[1], 3);

    auto t_cpu = t.to(Device::cpu());
    auto* data = t_cpu.data<float>();
    // Check diagonal
    EXPECT_FLOAT_EQ(data[0], 1.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[4], 1.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[8], 1.0f) << "Failed on " << device.to_string();
    // Check off-diagonal
    EXPECT_FLOAT_EQ(data[1], 0.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[2], 0.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[3], 0.0f) << "Failed on " << device.to_string();
}

TEST_P(OpsBackendTest, EyeRectangular) {
    auto t = eye(2, 4, DType::Float32, device);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 4);

    auto t_cpu = t.to(Device::cpu());
    auto* data = t_cpu.data<float>();
    // Check diagonal
    EXPECT_FLOAT_EQ(data[0], 1.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[5], 1.0f) << "Failed on " << device.to_string();
    // Check off-diagonal
    EXPECT_FLOAT_EQ(data[1], 0.0f) << "Failed on " << device.to_string();
    EXPECT_FLOAT_EQ(data[2], 0.0f) << "Failed on " << device.to_string();
}

TEST_P(OpsBackendTest, Rand) {
    auto t = rand({100}, DType::Float32, device);
    EXPECT_EQ(t.numel(), 100);

    auto t_cpu = t.to(Device::cpu());
    const float* data = t_cpu.data<float>();
    // Check all values are in [0, 1]
    for (int i = 0; i < 100; i++) {
        EXPECT_GE(data[i], 0.0f) << "Failed on " << device.to_string();
        EXPECT_LE(data[i], 1.0f) << "Failed on " << device.to_string();
    }
}

TEST_P(OpsBackendTest, Randn) {
    auto t = randn({1000}, DType::Float32, device);
    EXPECT_EQ(t.numel(), 1000);

    auto t_cpu = t.to(Device::cpu());
    const float* data = t_cpu.data<float>();
    // Basic check: calculate mean and std
    // For N(0,1) with 1000 samples, mean should be close to 0
    // and std should be close to 1
    double sum = 0.0;
    for (int i = 0; i < 1000; i++) {
        sum += data[i];
    }
    double mean = sum / 1000.0;
    EXPECT_NEAR(mean, 0.0, 0.2) << "Failed on " << device.to_string();

    double var_sum = 0.0;
    for (int i = 0; i < 1000; i++) {
        var_sum += (data[i] - mean) * (data[i] - mean);
    }
    double std = std::sqrt(var_sum / 1000.0);
    EXPECT_NEAR(std, 1.0, 0.2) << "Failed on " << device.to_string();
}

// Instantiate tests for all backends
INSTANTIATE_BACKEND_TESTS(OpsBackendTest);
