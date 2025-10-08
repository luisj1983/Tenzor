#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>

using namespace tenzor;

TEST(TensorTest, Creation) {
    auto t = zeros({2, 3}, DType::Float32, Device::cpu());
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_EQ(t.numel(), 6);
}

TEST(TensorTest, Ones) {
    auto t = ones({3, 4});
    EXPECT_EQ(t.shape()[0], 3);
    EXPECT_EQ(t.shape()[1], 4);
}

TEST(TensorTest, DeviceProperty) {
    auto t = zeros({2, 2});
    EXPECT_EQ(t.device().type, Device::Type::CPU);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
