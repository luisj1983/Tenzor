#include <gtest/gtest.h>
#include <tenzor/core/device.hpp>

using namespace tenzor;

TEST(DeviceTest, CPUDevice) {
    auto device = Device::cpu();
    EXPECT_EQ(device.type, Device::Type::CPU);
    EXPECT_EQ(device.index, 0);
}

TEST(DeviceTest, ToString) {
    auto cpu = Device::cpu();
    EXPECT_EQ(cpu.to_string(), "cpu");

    auto cuda = Device::cuda(0);
    EXPECT_EQ(cuda.to_string(), "cuda:0");
}

TEST(DeviceTest, Comparison) {
    auto cpu1 = Device::cpu();
    auto cpu2 = Device::cpu();
    EXPECT_EQ(cpu1, cpu2);

    auto cuda1 = Device::cuda(0);
    auto cuda2 = Device::cuda(1);
    EXPECT_NE(cuda1, cuda2);
}
