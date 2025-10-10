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

TEST(OpsTest, Full) {
    auto t = full({2, 3}, 5.0f);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_EQ(t.numel(), 6);

    const float* data = t.data<float>();
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(data[i], 5.0f);
    }
}

TEST(OpsTest, FullInt32) {
    auto t = full({3}, 42.0f, DType::Int32);
    EXPECT_EQ(t.numel(), 3);

    const int32_t* data = t.data<int32_t>();
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(data[i], 42);
    }
}

TEST(OpsTest, Arange) {
    auto t = arange(0, 5, 1);
    EXPECT_EQ(t.numel(), 5);

    const float* data = t.data<float>();
    for (int i = 0; i < 5; i++) {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
    }
}

TEST(OpsTest, ArangeStep) {
    auto t = arange(0, 10, 2);
    EXPECT_EQ(t.numel(), 5);

    const float* data = t.data<float>();
    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 4.0f);
    EXPECT_FLOAT_EQ(data[3], 6.0f);
    EXPECT_FLOAT_EQ(data[4], 8.0f);
}

TEST(OpsTest, ArangeFloat) {
    auto t = arange(0, 2, 0.5f);
    EXPECT_EQ(t.numel(), 4);

    const float* data = t.data<float>();
    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[1], 0.5f);
    EXPECT_FLOAT_EQ(data[2], 1.0f);
    EXPECT_FLOAT_EQ(data[3], 1.5f);
}

TEST(OpsTest, Linspace) {
    auto t = linspace(0, 1, 5);
    EXPECT_EQ(t.numel(), 5);

    const float* data = t.data<float>();
    EXPECT_FLOAT_EQ(data[0], 0.0f);
    EXPECT_FLOAT_EQ(data[1], 0.25f);
    EXPECT_FLOAT_EQ(data[2], 0.5f);
    EXPECT_FLOAT_EQ(data[3], 0.75f);
    EXPECT_FLOAT_EQ(data[4], 1.0f);
}

TEST(OpsTest, LinspaceNegative) {
    auto t = linspace(-5, 5, 11);
    EXPECT_EQ(t.numel(), 11);

    const float* data = t.data<float>();
    EXPECT_FLOAT_EQ(data[0], -5.0f);
    EXPECT_FLOAT_EQ(data[5], 0.0f);
    EXPECT_FLOAT_EQ(data[10], 5.0f);
}

TEST(OpsTest, Eye) {
    auto t = eye(3);
    EXPECT_EQ(t.shape()[0], 3);
    EXPECT_EQ(t.shape()[1], 3);

    const float* data = t.data<float>();
    // Check diagonal
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[4], 1.0f);
    EXPECT_FLOAT_EQ(data[8], 1.0f);
    // Check off-diagonal
    EXPECT_FLOAT_EQ(data[1], 0.0f);
    EXPECT_FLOAT_EQ(data[2], 0.0f);
    EXPECT_FLOAT_EQ(data[3], 0.0f);
}

TEST(OpsTest, EyeRectangular) {
    auto t = eye(2, 4);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 4);

    const float* data = t.data<float>();
    // Check diagonal
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[5], 1.0f);
    // Check off-diagonal
    EXPECT_FLOAT_EQ(data[1], 0.0f);
    EXPECT_FLOAT_EQ(data[2], 0.0f);
}

TEST(OpsTest, Rand) {
    auto t = rand({100});
    EXPECT_EQ(t.numel(), 100);

    const float* data = t.data<float>();
    // Check all values are in [0, 1]
    for (int i = 0; i < 100; i++) {
        EXPECT_GE(data[i], 0.0f);
        EXPECT_LE(data[i], 1.0f);
    }
}

TEST(OpsTest, Randn) {
    auto t = randn({1000});
    EXPECT_EQ(t.numel(), 1000);

    const float* data = t.data<float>();
    // Basic check: calculate mean and std
    // For N(0,1) with 1000 samples, mean should be close to 0
    // and std should be close to 1
    double sum = 0.0;
    for (int i = 0; i < 1000; i++) {
        sum += data[i];
    }
    double mean = sum / 1000.0;
    EXPECT_NEAR(mean, 0.0, 0.2);  // Allow some variance

    double var_sum = 0.0;
    for (int i = 0; i < 1000; i++) {
        var_sum += (data[i] - mean) * (data[i] - mean);
    }
    double std = std::sqrt(var_sum / 1000.0);
    EXPECT_NEAR(std, 1.0, 0.2);  // Allow some variance
}

// Add more operation tests
