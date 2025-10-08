#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>

using namespace tenzor;

TEST(AutogradTest, VariableCreation) {
    auto data = ones({2, 2});
    auto var = Variable(data, true);

    EXPECT_TRUE(var.requires_grad());
    EXPECT_FALSE(var.has_grad());
}

TEST(AutogradTest, Detach) {
    auto var = Variable(ones({2, 2}), true);
    auto detached = var.detach();

    EXPECT_FALSE(detached.requires_grad());
}

// Add more autograd tests
