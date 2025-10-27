#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class TensorBackendTest : public BackendTest {};

TEST_P(TensorBackendTest, Creation) {
    auto t = zeros({2, 3}, DType::Float32, device);
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_EQ(t.numel(), 6);
}

TEST_P(TensorBackendTest, Ones) {
    auto t = ones({3, 4}, DType::Float32, device);
    EXPECT_EQ(t.shape()[0], 3);
    EXPECT_EQ(t.shape()[1], 4);

    // Verify values
    auto t_cpu = t.to(Device::cpu());
    auto* data = t_cpu.data<float>();
    for (int64_t i = 0; i < t_cpu.numel(); ++i) {
        EXPECT_NEAR(data[i], 1.0f, 1e-5f) << "Failed on " << device.to_string();
    }
}

TEST_P(TensorBackendTest, DeviceProperty) {
    auto t = zeros({2, 2}, DType::Float32, device);
    EXPECT_EQ(t.device().type, device.type);
    EXPECT_EQ(t.device().index, device.index);
}

TEST_P(TensorBackendTest, Full) {
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

TEST_P(TensorBackendTest, Reshape) {
    auto t = ones({2, 3}, DType::Float32, device);
    auto reshaped = t.reshape({3, 2});

    EXPECT_EQ(reshaped.shape()[0], 3);
    EXPECT_EQ(reshaped.shape()[1], 2);
    EXPECT_EQ(reshaped.numel(), 6);
}

TEST_P(TensorBackendTest, Transpose) {
    auto t = zeros({2, 3}, DType::Float32, device);
    auto transposed = t.transpose(0, 1);

    EXPECT_EQ(transposed.shape()[0], 3);
    EXPECT_EQ(transposed.shape()[1], 2);
}

TEST_P(TensorBackendTest, DeviceTransfer) {
    // Create tensor on parameterized device
    auto t = ones({2, 2}, DType::Float32, device);

    // Transfer to CPU
    auto t_cpu = t.to(Device::cpu());
    EXPECT_EQ(t_cpu.device().type, Device::Type::CPU);

    // Verify data integrity
    auto* data = t_cpu.data<float>();
    for (int64_t i = 0; i < t_cpu.numel(); ++i) {
        EXPECT_NEAR(data[i], 1.0f, 1e-5f) << "Failed on " << device.to_string();
    }
}

// Instantiate tests for all backends
INSTANTIATE_BACKEND_TESTS(TensorBackendTest);
