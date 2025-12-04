#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"

using namespace tenzor;

TEST(OnesF16Simple, SingleElement) {
    auto t = ones({1}, DType::Float16, Device::vulkan(0));
    auto cpu = t.to(Device::cpu());
    auto* data = cpu.data<Float16>();
    
    std::cout << "Float16 ones([1]): " << static_cast<float>(data[0]) 
              << " (bits=0x" << std::hex << data[0].bits << std::dec << ")" << std::endl;
    
    EXPECT_EQ(data[0].bits, 0x3C00);  // 1.0 in Float16
}

TEST(OnesF16Simple, TwoElements) {
    auto t = ones({2}, DType::Float16, Device::vulkan(0));
    auto cpu = t.to(Device::cpu());
    auto* data = cpu.data<Float16>();
    
    std::cout << "Float16 ones([2]): " << static_cast<float>(data[0]) << ", " << static_cast<float>(data[1]) << std::endl;
    
    EXPECT_EQ(data[0].bits, 0x3C00);
    EXPECT_EQ(data[1].bits, 0x3C00);
}

TEST(OnesF16Simple, OnesLikeSingleElement) {
    auto input = zeros({1}, DType::Float16, Device::vulkan(0));
    auto t = ones_like(input);
    auto cpu = t.to(Device::cpu());
    auto* data = cpu.data<Float16>();
    
    std::cout << "Float16 ones_like({1}): " << static_cast<float>(data[0]) 
              << " (bits=0x" << std::hex << data[0].bits << std::dec << ")" << std::endl;
    
    EXPECT_EQ(data[0].bits, 0x3C00);
}
