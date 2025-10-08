#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>

using namespace tenzor;

TEST(OpsTest, Zeros) {
    auto t = zeros({2, 3});
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
}

TEST(OpsTest, Ones) {
    auto t = ones({3, 4});
    EXPECT_EQ(t.numel(), 12);
}

// Add more operation tests
