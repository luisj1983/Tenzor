#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;

// Global test environment for initialization
class ComparisonOpsTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const comparison_env =
    ::testing::AddGlobalTestEnvironment(new ComparisonOpsTestEnvironment);

// Test comparison operators across backends
class ComparisonOpsTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        // Library is initialized by global test environment

        backend_name = GetParam();
        if (backend_name == "cpu") {
            device = Device::cpu();
        } else if (backend_name == "cuda") {
            device = Device::cuda(0);
        } else if (backend_name == "oneapi") {
            device = Device::oneapi(0);
        } else if (backend_name == "vulkan") {
            device = Device::vulkan(0);
        }
    }

    std::string backend_name;
    Device device;
};

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
            << "Failed for backend: " << backend_name << " at index " << i;
    }

    // Check all should be false
    for (int64_t i = 0; i < result_false_cpu.numel(); ++i) {
        EXPECT_FALSE(static_cast<const bool*>(result_false_cpu.data_ptr())[i])
            << "Failed for backend: " << backend_name << " at index " << i;
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
            << "Failed for backend: " << backend_name << " at index " << i;
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
            << "Failed for backend: " << backend_name << " at index " << i;
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
            << "Failed for backend: " << backend_name << " at index " << i;
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
            << "Failed for backend: " << backend_name << " at index " << i;
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
            << "Failed for backend: " << backend_name << " at index " << i;
    }
}

// Only test OneAPI for utility operations (cat, clamp, sign)
TEST(UtilityOpsTest, ClampOperation) {
    auto device = Device::oneapi(0);

    // Create tensor with values: -5, -3, -1, 1, 3, 5
    std::vector<float> data = {-5.0f, -3.0f, -1.0f, 1.0f, 3.0f, 5.0f};
    auto input = tenzor::zeros({6}, DType::Float32, device);
    // TODO: Fill tensor with data

    auto result = tenzor::clamp(input, -2.0f, 2.0f);
    auto result_cpu = result.to(Device::cpu());

    // Expected: [-2, -2, -1, 1, 2, 2]
    // This test needs proper tensor initialization
    GTEST_SKIP() << "Clamp test requires proper tensor initialization";
}

TEST(UtilityOpsTest, SignOperation) {
    auto device = Device::oneapi(0);

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

// Parameterized test across all backends
INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    ComparisonOpsTest,
    ::testing::Values("cpu", "oneapi", "vulkan"),
    [](const ::testing::TestParamInfo<ComparisonOpsTest::ParamType>& info) {
        return info.param;
    }
);

// Note: GTest::gtest_main handles initialization
// No need for custom main() function when using GTest::gtest_main
