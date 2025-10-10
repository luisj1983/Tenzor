#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>

using namespace tenzor;

// Global test environment that initializes Tenzor before tests
class TenzorTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

// Register the environment
static ::testing::Environment* const tenzor_env =
    ::testing::AddGlobalTestEnvironment(new TenzorTestEnvironment);

//==============================================================================
// Broadcasting Tests
//==============================================================================

TEST(BroadcastingTest, AddBroadcast_ScalarToVector) {
    // Test: (3,) + (1,) -> (3,)
    auto a = ones({3}, DType::Float32);
    auto b = ones({1}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();

    a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f;
    b_data[0] = 10.0f;

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 3);

    auto c_data = c.data<float>();
    EXPECT_FLOAT_EQ(c_data[0], 11.0f);
    EXPECT_FLOAT_EQ(c_data[1], 12.0f);
    EXPECT_FLOAT_EQ(c_data[2], 13.0f);
}

TEST(BroadcastingTest, AddBroadcast_RowToMatrix) {
    // Test: (2, 3) + (1, 3) -> (2, 3)
    auto a = ones({2, 3}, DType::Float32);
    auto b = ones({1, 3}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();

    // A = [[1, 2, 3], [4, 5, 6]]
    for (int i = 0; i < 6; i++) {
        a_data[i] = static_cast<float>(i + 1);
    }

    // B = [[10, 20, 30]]
    b_data[0] = 10.0f; b_data[1] = 20.0f; b_data[2] = 30.0f;

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 3);

    auto c_data = c.data<float>();

    // Expected: [[11, 22, 33], [14, 25, 36]]
    EXPECT_FLOAT_EQ(c_data[0], 11.0f);
    EXPECT_FLOAT_EQ(c_data[1], 22.0f);
    EXPECT_FLOAT_EQ(c_data[2], 33.0f);
    EXPECT_FLOAT_EQ(c_data[3], 14.0f);
    EXPECT_FLOAT_EQ(c_data[4], 25.0f);
    EXPECT_FLOAT_EQ(c_data[5], 36.0f);
}

TEST(BroadcastingTest, AddBroadcast_ColumnToMatrix) {
    // Test: (2, 3) + (2, 1) -> (2, 3)
    auto a = ones({2, 3}, DType::Float32);
    auto b = ones({2, 1}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();

    // A = [[1, 2, 3], [4, 5, 6]]
    for (int i = 0; i < 6; i++) {
        a_data[i] = static_cast<float>(i + 1);
    }

    // B = [[100], [200]]
    b_data[0] = 100.0f;
    b_data[1] = 200.0f;

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 3);

    auto c_data = c.data<float>();

    // Expected: [[101, 102, 103], [204, 205, 206]]
    EXPECT_FLOAT_EQ(c_data[0], 101.0f);
    EXPECT_FLOAT_EQ(c_data[1], 102.0f);
    EXPECT_FLOAT_EQ(c_data[2], 103.0f);
    EXPECT_FLOAT_EQ(c_data[3], 204.0f);
    EXPECT_FLOAT_EQ(c_data[4], 205.0f);
    EXPECT_FLOAT_EQ(c_data[5], 206.0f);
}

TEST(BroadcastingTest, AddBroadcast_ScalarToMatrix) {
    // Test: (2, 3) + (1, 1) -> (2, 3)
    auto a = ones({2, 3}, DType::Float32);
    auto b = ones({1, 1}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();

    for (int i = 0; i < 6; i++) {
        a_data[i] = static_cast<float>(i + 1);
    }

    b_data[0] = 1000.0f;

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 3);

    auto c_data = c.data<float>();

    // Expected: [[1001, 1002, 1003], [1004, 1005, 1006]]
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 1000.0f + static_cast<float>(i + 1));
    }
}

TEST(BroadcastingTest, AddBroadcast_DifferentDimensions) {
    // Test: (3, 2) + (2,) -> (3, 2)
    auto a = ones({3, 2}, DType::Float32);
    auto b = ones({2}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();

    // A = [[1, 2], [3, 4], [5, 6]]
    for (int i = 0; i < 6; i++) {
        a_data[i] = static_cast<float>(i + 1);
    }

    // B = [10, 20]
    b_data[0] = 10.0f;
    b_data[1] = 20.0f;

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 3);
    EXPECT_EQ(c.shape()[1], 2);

    auto c_data = c.data<float>();

    // Expected: [[11, 22], [13, 24], [15, 26]]
    EXPECT_FLOAT_EQ(c_data[0], 11.0f);
    EXPECT_FLOAT_EQ(c_data[1], 22.0f);
    EXPECT_FLOAT_EQ(c_data[2], 13.0f);
    EXPECT_FLOAT_EQ(c_data[3], 24.0f);
    EXPECT_FLOAT_EQ(c_data[4], 15.0f);
    EXPECT_FLOAT_EQ(c_data[5], 26.0f);
}

TEST(BroadcastingTest, AddBroadcast_Int32) {
    // Test broadcasting with Int32
    auto a = ones({2, 3}, DType::Int32);
    auto b = ones({1, 3}, DType::Int32);

    auto a_data = a.data<int32_t>();
    auto b_data = b.data<int32_t>();

    for (int i = 0; i < 6; i++) {
        a_data[i] = i + 1;
    }

    b_data[0] = 10; b_data[1] = 20; b_data[2] = 30;

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 3);

    auto c_data = c.data<int32_t>();

    // Expected: [[11, 22, 33], [14, 25, 36]]
    EXPECT_EQ(c_data[0], 11);
    EXPECT_EQ(c_data[1], 22);
    EXPECT_EQ(c_data[2], 33);
    EXPECT_EQ(c_data[3], 14);
    EXPECT_EQ(c_data[4], 25);
    EXPECT_EQ(c_data[5], 36);
}

TEST(BroadcastingTest, AddNoBroadcast_SameShape) {
    // Test that same-shape operations still work (fast path)
    auto a = ones({2, 3}, DType::Float32);
    auto b = ones({2, 3}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();

    for (int i = 0; i < 6; i++) {
        a_data[i] = static_cast<float>(i + 1);
        b_data[i] = static_cast<float>((i + 1) * 10);
    }

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 3);

    auto c_data = c.data<float>();

    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(c_data[i], static_cast<float>((i + 1) + (i + 1) * 10));
    }
}
