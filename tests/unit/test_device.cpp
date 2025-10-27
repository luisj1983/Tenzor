#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class DeviceTest : public BackendTest {};

TEST_P(DeviceTest, CPUDevice) {
    auto cpu_device = Device::cpu();
    EXPECT_EQ(cpu_device.type, Device::Type::CPU) << "Failed on " << device.to_string();
    EXPECT_EQ(cpu_device.index, 0) << "Failed on " << device.to_string();
}

TEST_P(DeviceTest, ToString) {
    auto cpu = Device::cpu();
    EXPECT_EQ(cpu.to_string(), "cpu") << "Failed on " << device.to_string();

    auto cuda = Device::cuda(0);
    EXPECT_EQ(cuda.to_string(), "cuda:0") << "Failed on " << device.to_string();
}

TEST_P(DeviceTest, Comparison) {
    auto cpu1 = Device::cpu();
    auto cpu2 = Device::cpu();
    EXPECT_EQ(cpu1, cpu2) << "Failed on " << device.to_string();

    auto cuda1 = Device::cuda(0);
    auto cuda2 = Device::cuda(1);
    EXPECT_NE(cuda1, cuda2) << "Failed on " << device.to_string();
}

INSTANTIATE_BACKEND_TESTS(DeviceTest);
