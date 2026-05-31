#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include <cmath>
#include <cstring>
#include <vector>

using namespace tenzor;

// Test comparison operators across backends
class ComparisonOpsTest : public ::tenzor::testing::BackendTest {};

TEST_P(ComparisonOpsTest, EqualOperator) {
    auto a = tenzor::full({3, 3}, 5.0f, DType::Float32, device);
    auto b = tenzor::full({3, 3}, 5.0f, DType::Float32, device);
    auto c = tenzor::full({3, 3}, 3.0f, DType::Float32, device);

    auto result_true = tenzor::eq(a, b);
    auto result_false = tenzor::eq(a, c);

    // Convert to CPU for verification
    auto result_true_cpu = result_true.to(Device::cpu());
    auto result_false_cpu = result_false.to(Device::cpu());

    // Check all should be true
    for (int64_t i = 0; i < result_true_cpu.numel(); ++i) {
        EXPECT_TRUE(static_cast<const bool*>(result_true_cpu.data_ptr())[i])
            << "Failed at index " << i;
    }

    // Check all should be false
    for (int64_t i = 0; i < result_false_cpu.numel(); ++i) {
        EXPECT_FALSE(static_cast<const bool*>(result_false_cpu.data_ptr())[i])
            << "Failed at index " << i;
    }
}

TEST_P(ComparisonOpsTest, NotEqualOperator) {
    auto a = tenzor::full({3, 3}, 5.0f, DType::Float32, device);
    auto b = tenzor::full({3, 3}, 3.0f, DType::Float32, device);

    auto result = tenzor::ne(a, b);
    auto result_cpu = result.to(Device::cpu());

    // Check all should be true
    for (int64_t i = 0; i < result_cpu.numel(); ++i) {
        EXPECT_TRUE(static_cast<const bool*>(result_cpu.data_ptr())[i])
            << "Failed at index " << i;
    }
}

TEST_P(ComparisonOpsTest, LessThanOperator) {
    auto a = tenzor::full({3, 3}, 3.0f, DType::Float32, device);
    auto b = tenzor::full({3, 3}, 5.0f, DType::Float32, device);

    auto result = tenzor::lt(a, b);
    auto result_cpu = result.to(Device::cpu());

    // Check all should be true (3 < 5)
    for (int64_t i = 0; i < result_cpu.numel(); ++i) {
        EXPECT_TRUE(static_cast<const bool*>(result_cpu.data_ptr())[i])
            << "Failed at index " << i;
    }
}

TEST_P(ComparisonOpsTest, GreaterThanOperator) {
    auto a = tenzor::full({3, 3}, 7.0f, DType::Float32, device);
    auto b = tenzor::full({3, 3}, 5.0f, DType::Float32, device);

    auto result = tenzor::gt(a, b);
    auto result_cpu = result.to(Device::cpu());

    // Check all should be true (7 > 5)
    for (int64_t i = 0; i < result_cpu.numel(); ++i) {
        EXPECT_TRUE(static_cast<const bool*>(result_cpu.data_ptr())[i])
            << "Failed at index " << i;
    }
}

TEST_P(ComparisonOpsTest, LessEqualOperator) {
    auto a = tenzor::full({2, 2}, 5.0f, DType::Float32, device);
    auto b = tenzor::full({2, 2}, 5.0f, DType::Float32, device);

    auto result = tenzor::le(a, b);
    auto result_cpu = result.to(Device::cpu());

    // Check all should be true (5 <= 5)
    for (int64_t i = 0; i < result_cpu.numel(); ++i) {
        EXPECT_TRUE(static_cast<const bool*>(result_cpu.data_ptr())[i])
            << "Failed at index " << i;
    }
}

TEST_P(ComparisonOpsTest, GreaterEqualOperator) {
    auto a = tenzor::full({2, 2}, 5.0f, DType::Float32, device);
    auto b = tenzor::full({2, 2}, 5.0f, DType::Float32, device);

    auto result = tenzor::ge(a, b);
    auto result_cpu = result.to(Device::cpu());

    // Check all should be true (5 >= 5)
    for (int64_t i = 0; i < result_cpu.numel(); ++i) {
        EXPECT_TRUE(static_cast<const bool*>(result_cpu.data_ptr())[i])
            << "Failed at index " << i;
    }
}

INSTANTIATE_BACKEND_TESTS(ComparisonOpsTest);

// Utility operations (cat, clamp, sign) across backends
class UtilityOpsTest : public ::tenzor::testing::BackendTest {};

TEST_P(UtilityOpsTest, ClampOperation) {
    // Create tensor with values: -5, -3, -1, 1, 3, 5
    std::vector<float> data = {-5.0f, -3.0f, -1.0f, 1.0f, 3.0f, 5.0f};
    auto input_cpu = tenzor::zeros({6}, DType::Float32, Device::cpu());
    std::memcpy(input_cpu.data<float>(), data.data(), data.size() * sizeof(float));
    auto input = input_cpu.to(device);

    auto result = tenzor::clamp(input, -2.0f, 2.0f);
    auto result_cpu = result.to(Device::cpu());

    // Expected: [-2, -2, -1, 1, 2, 2]
    auto* ptr = result_cpu.data<float>();
    EXPECT_FLOAT_EQ(ptr[0], -2.0f);
    EXPECT_FLOAT_EQ(ptr[1], -2.0f);
    EXPECT_FLOAT_EQ(ptr[2], -1.0f);
    EXPECT_FLOAT_EQ(ptr[3], 1.0f);
    EXPECT_FLOAT_EQ(ptr[4], 2.0f);
    EXPECT_FLOAT_EQ(ptr[5], 2.0f);
}

TEST_P(UtilityOpsTest, SignOperation) {
    // Create simple test
    auto pos = tenzor::full({2, 2}, 5.0f, DType::Float32, device);
    auto result = tenzor::sign(pos);
    auto result_cpu = result.to(Device::cpu());

    // All should be 1.0
    const float* data = static_cast<const float*>(result_cpu.data_ptr());
    for (int64_t i = 0; i < result_cpu.numel(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f) << "Sign of positive should be 1.0";
    }
}

INSTANTIATE_BACKEND_TESTS(UtilityOpsTest);
