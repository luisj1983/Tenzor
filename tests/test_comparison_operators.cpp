/**
 * @file test_comparison_operators.cpp
 * @brief Test element-wise comparison operators
 */

#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "backend_test_fixture.hpp"

using namespace tenzor;

class ComparisonOperatorTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;

        // Create test tensors with known values (host -> CPU -> device).
        std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        std::vector<float> data_b = {3.0f, 2.0f, 3.0f, 2.0f, 6.0f};
        a = from_data<float>(data_a.data(), {5}, Device::cpu()).to(device);
        b = from_data<float>(data_b.data(), {5}, Device::cpu()).to(device);
    }

    Tensor a;
    Tensor b;
};

TEST_P(ComparisonOperatorTest, EqualOperator) {
    // Test operator==
    Tensor result = a == b;

    EXPECT_EQ(result.shape().size(), 1);
    EXPECT_EQ(result.shape()[0], 5);
    EXPECT_EQ(result.dtype(), DType::Bool);

    // Expected: {false, true, true, false, false}
    auto result_cpu = result.cpu();
    auto* data = result_cpu.data<bool>();
    EXPECT_FALSE(data[0]);  // 1.0 != 3.0
    EXPECT_TRUE(data[1]);   // 2.0 == 2.0
    EXPECT_TRUE(data[2]);   // 3.0 == 3.0
    EXPECT_FALSE(data[3]);  // 4.0 != 2.0
    EXPECT_FALSE(data[4]);  // 5.0 != 6.0
}

TEST_P(ComparisonOperatorTest, NotEqualOperator) {
    // Test operator!=
    Tensor result = a != b;

    EXPECT_EQ(result.shape().size(), 1);
    EXPECT_EQ(result.shape()[0], 5);
    EXPECT_EQ(result.dtype(), DType::Bool);

    // Expected: {true, false, false, true, true}
    auto result_cpu = result.cpu();
    auto* data = result_cpu.data<bool>();
    EXPECT_TRUE(data[0]);   // 1.0 != 3.0
    EXPECT_FALSE(data[1]);  // 2.0 == 2.0
    EXPECT_FALSE(data[2]);  // 3.0 == 3.0
    EXPECT_TRUE(data[3]);   // 4.0 != 2.0
    EXPECT_TRUE(data[4]);   // 5.0 != 6.0
}

TEST_P(ComparisonOperatorTest, LessThanOperator) {
    // Test operator<
    Tensor result = a < b;

    EXPECT_EQ(result.shape().size(), 1);
    EXPECT_EQ(result.shape()[0], 5);
    EXPECT_EQ(result.dtype(), DType::Bool);

    // Expected: {true, false, false, false, true}
    auto result_cpu = result.cpu();
    auto* data = result_cpu.data<bool>();
    EXPECT_TRUE(data[0]);   // 1.0 < 3.0
    EXPECT_FALSE(data[1]);  // 2.0 < 2.0
    EXPECT_FALSE(data[2]);  // 3.0 < 3.0
    EXPECT_FALSE(data[3]);  // 4.0 < 2.0
    EXPECT_TRUE(data[4]);   // 5.0 < 6.0
}

TEST_P(ComparisonOperatorTest, LessThanOrEqualOperator) {
    // Test operator<=
    Tensor result = a <= b;

    EXPECT_EQ(result.shape().size(), 1);
    EXPECT_EQ(result.shape()[0], 5);
    EXPECT_EQ(result.dtype(), DType::Bool);

    // Expected: {true, true, true, false, true}
    auto result_cpu = result.cpu();
    auto* data = result_cpu.data<bool>();
    EXPECT_TRUE(data[0]);   // 1.0 <= 3.0
    EXPECT_TRUE(data[1]);   // 2.0 <= 2.0
    EXPECT_TRUE(data[2]);   // 3.0 <= 3.0
    EXPECT_FALSE(data[3]);  // 4.0 <= 2.0
    EXPECT_TRUE(data[4]);   // 5.0 <= 6.0
}

TEST_P(ComparisonOperatorTest, GreaterThanOperator) {
    // Test operator>
    Tensor result = a > b;

    EXPECT_EQ(result.shape().size(), 1);
    EXPECT_EQ(result.shape()[0], 5);
    EXPECT_EQ(result.dtype(), DType::Bool);

    // Expected: {false, false, false, true, false}
    auto result_cpu = result.cpu();
    auto* data = result_cpu.data<bool>();
    EXPECT_FALSE(data[0]);  // 1.0 > 3.0
    EXPECT_FALSE(data[1]);  // 2.0 > 2.0
    EXPECT_FALSE(data[2]);  // 3.0 > 3.0
    EXPECT_TRUE(data[3]);   // 4.0 > 2.0
    EXPECT_FALSE(data[4]);  // 5.0 > 6.0
}

TEST_P(ComparisonOperatorTest, GreaterThanOrEqualOperator) {
    // Test operator>=
    Tensor result = a >= b;

    EXPECT_EQ(result.shape().size(), 1);
    EXPECT_EQ(result.shape()[0], 5);
    EXPECT_EQ(result.dtype(), DType::Bool);

    // Expected: {false, true, true, true, false}
    auto result_cpu = result.cpu();
    auto* data = result_cpu.data<bool>();
    EXPECT_FALSE(data[0]);  // 1.0 >= 3.0
    EXPECT_TRUE(data[1]);   // 2.0 >= 2.0
    EXPECT_TRUE(data[2]);   // 3.0 >= 3.0
    EXPECT_TRUE(data[3]);   // 4.0 >= 2.0
    EXPECT_FALSE(data[4]);  // 5.0 >= 6.0
}

TEST_P(ComparisonOperatorTest, ComparisonWith2D) {
    // Test with 2D tensors
    std::vector<float> data_x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_y = {2.0f, 2.0f, 4.0f, 3.0f};
    auto x = from_data<float>(data_x.data(), {2, 2}, Device::cpu()).to(device);
    auto y = from_data<float>(data_y.data(), {2, 2}, Device::cpu()).to(device);

    Tensor eq_result = x == y;
    Tensor lt_result = x < y;
    Tensor gt_result = x > y;

    EXPECT_EQ(eq_result.shape().size(), 2);
    EXPECT_EQ(eq_result.shape()[0], 2);
    EXPECT_EQ(eq_result.shape()[1], 2);
    EXPECT_EQ(eq_result.dtype(), DType::Bool);

    auto eq_cpu = eq_result.cpu();
    auto* eq_data = eq_cpu.data<bool>();
    EXPECT_FALSE(eq_data[0]);  // 1.0 != 2.0
    EXPECT_TRUE(eq_data[1]);   // 2.0 == 2.0
    EXPECT_FALSE(eq_data[2]);  // 3.0 != 4.0
    EXPECT_FALSE(eq_data[3]);  // 4.0 != 3.0

    auto lt_cpu = lt_result.cpu();
    auto* lt_data = lt_cpu.data<bool>();
    EXPECT_TRUE(lt_data[0]);   // 1.0 < 2.0
    EXPECT_FALSE(lt_data[1]);  // 2.0 < 2.0
    EXPECT_TRUE(lt_data[2]);   // 3.0 < 4.0
    EXPECT_FALSE(lt_data[3]);  // 4.0 < 3.0
}

TEST_P(ComparisonOperatorTest, FunctionVersions) {
    // Test that function versions work the same as operators
    Tensor eq_op = a == b;
    Tensor eq_fn = eq(a, b);

    EXPECT_EQ(eq_op.shape().size(), eq_fn.shape().size());
    EXPECT_EQ(eq_op.dtype(), eq_fn.dtype());

    auto eq_op_cpu = eq_op.cpu();
    auto eq_fn_cpu = eq_fn.cpu();
    auto* op_data = eq_op_cpu.data<bool>();
    auto* fn_data = eq_fn_cpu.data<bool>();
    for (int64_t i = 0; i < 5; ++i) {
        EXPECT_EQ(op_data[i], fn_data[i]);
    }

    // Test other functions
    Tensor ne_result = ne(a, b);
    Tensor lt_result = lt(a, b);
    Tensor le_result = le(a, b);
    Tensor gt_result = gt(a, b);
    Tensor ge_result = ge(a, b);

    EXPECT_EQ(ne_result.dtype(), DType::Bool);
    EXPECT_EQ(lt_result.dtype(), DType::Bool);
    EXPECT_EQ(le_result.dtype(), DType::Bool);
    EXPECT_EQ(gt_result.dtype(), DType::Bool);
    EXPECT_EQ(ge_result.dtype(), DType::Bool);
}

INSTANTIATE_BACKEND_TESTS(ComparisonOperatorTest);
