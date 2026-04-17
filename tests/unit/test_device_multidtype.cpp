/**
 * @file test_device_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for Device creation and properties
 */

#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class DeviceMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(DeviceMultiDTypeTest, CPUDeviceProperties) {
    auto cpu_device = Device::cpu();
    EXPECT_EQ(cpu_device.type, Device::Type::CPU);
    EXPECT_EQ(cpu_device.index, 0);
}

TEST_P(DeviceMultiDTypeTest, ToString) {
    auto cpu = Device::cpu();
    EXPECT_EQ(cpu.to_string(), "cpu");

    auto cuda = Device::cuda(0);
    EXPECT_EQ(cuda.to_string(), "cuda:0");
}

TEST_P(DeviceMultiDTypeTest, Comparison) {
    auto cpu1 = Device::cpu();
    auto cpu2 = Device::cpu();
    EXPECT_EQ(cpu1, cpu2);

    auto cuda1 = Device::cuda(0);
    auto cuda2 = Device::cuda(1);
    EXPECT_NE(cuda1, cuda2);
}

TEST_P(DeviceMultiDTypeTest, TensorOnDevice) {
    // Create a tensor on the parameterized device with the parameterized dtype
    auto t = createRandn({4, 4});
    EXPECT_EQ(t.device().type, device().type);
    expectDType(t);
}

TEST_P(DeviceMultiDTypeTest, TensorTransferToDevice) {
    // Create on CPU then transfer
    auto cpu_tensor = randn({3, 3}, DType::Float32, Device::cpu());
    auto dev_tensor = cpu_tensor.to(dtype()).to(device());
    EXPECT_EQ(dev_tensor.device().type, device().type);
    expectDType(dev_tensor);
}

TEST_P(DeviceMultiDTypeTest, TensorRoundTrip) {
    // CPU -> device -> CPU roundtrip
    auto original = randn({2, 5}, DType::Float32, Device::cpu());
    auto on_device = original.to(dtype()).to(device());
    auto back_cpu = on_device.to(Device::cpu()).to(DType::Float32);

    EXPECT_EQ(back_cpu.shape()[0], 2);
    EXPECT_EQ(back_cpu.shape()[1], 5);

    auto* orig_data = original.data<float>();
    auto* back_data = back_cpu.data<float>();
    for (int64_t i = 0; i < original.numel(); ++i) {
        EXPECT_NEAR(back_data[i], orig_data[i], atol()) << "Mismatch at index " << i;
    }
}

TEST_P(DeviceMultiDTypeTest, ZerosOnDevice) {
    auto t = createZeros({5, 5});
    auto cpu_t = t.to(Device::cpu()).to(DType::Float32);
    auto* data = cpu_t.data<float>();
    for (int64_t i = 0; i < cpu_t.numel(); ++i) {
        EXPECT_NEAR(data[i], 0.0f, atol());
    }
}

TEST_P(DeviceMultiDTypeTest, OnesOnDevice) {
    auto t = createOnes({3, 3});
    auto cpu_t = t.to(Device::cpu()).to(DType::Float32);
    auto* data = cpu_t.data<float>();
    for (int64_t i = 0; i < cpu_t.numel(); ++i) {
        EXPECT_NEAR(data[i], 1.0f, atol());
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DeviceMultiDTypeTest);
