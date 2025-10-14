#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"

using namespace tenzor;

TEST(BmmTest, BasicFunctionality) {
    // Create test tensors
    // a: (2, 3, 4) - 2 batches of 3x4 matrices
    // b: (2, 4, 5) - 2 batches of 4x5 matrices
    // result: (2, 3, 5) - 2 batches of 3x5 matrices

    Tensor a = ones({2, 3, 4}, DType::Float32, Device::cpu());
    Tensor b = ones({2, 4, 5}, DType::Float32, Device::cpu());

    Tensor result = bmm(a, b);

    // Verify output shape
    EXPECT_EQ(result.shape().size(), 3);
    EXPECT_EQ(result.shape()[0], 2);  // batch size
    EXPECT_EQ(result.shape()[1], 3);  // n
    EXPECT_EQ(result.shape()[2], 5);  // p

    // Verify values - each element should be 4.0 (sum of 4 ones)
    auto* data = result.data<float>();
    for (int64_t i = 0; i < result.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 4.0f);
    }
}

TEST(BmmTest, DifferentBatches) {
    // Test with different values in each batch
    Tensor a = ones({2, 2, 3}, DType::Float32, Device::cpu());
    Tensor b = ones({2, 3, 2}, DType::Float32, Device::cpu());

    // Modify second batch of a to be 2.0
    auto* a_data = a.data<float>();
    for (int i = 6; i < 12; ++i) {
        a_data[i] = 2.0f;
    }

    Tensor result = bmm(a, b);

    // First batch should have sum of 3.0 (3 ones)
    auto* r_data = result.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(r_data[i], 3.0f);
    }

    // Second batch should have sum of 6.0 (3 twos)
    for (int i = 4; i < 8; ++i) {
        EXPECT_FLOAT_EQ(r_data[i], 6.0f);
    }
}

TEST(BmmTest, InvalidInputs) {
    // Test with non-3D tensors
    Tensor a = ones({2, 3}, DType::Float32, Device::cpu());
    Tensor b = ones({3, 4}, DType::Float32, Device::cpu());

    EXPECT_THROW(bmm(a, b), std::runtime_error);

    // Test with mismatched batch sizes
    Tensor a3d = ones({2, 3, 4}, DType::Float32, Device::cpu());
    Tensor b3d = ones({3, 4, 5}, DType::Float32, Device::cpu());

    EXPECT_THROW(bmm(a3d, b3d), std::runtime_error);

    // Test with mismatched inner dimensions
    Tensor a_wrong = ones({2, 3, 4}, DType::Float32, Device::cpu());
    Tensor b_wrong = ones({2, 5, 6}, DType::Float32, Device::cpu());

    EXPECT_THROW(bmm(a_wrong, b_wrong), std::runtime_error);
}

TEST(BmmTest, SingleBatch) {
    // Test with batch size of 1
    Tensor a = ones({1, 2, 3}, DType::Float32, Device::cpu());
    Tensor b = ones({1, 3, 4}, DType::Float32, Device::cpu());

    Tensor result = bmm(a, b);

    EXPECT_EQ(result.shape()[0], 1);
    EXPECT_EQ(result.shape()[1], 2);
    EXPECT_EQ(result.shape()[2], 4);

    auto* data = result.data<float>();
    for (int64_t i = 0; i < result.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.0f);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
