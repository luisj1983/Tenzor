#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

/**
 * @file test_broadcasting_multidtype.cpp
 * @brief Multi-dtype parameterized tests for broadcasting operations
 *
 * This file refactors test_broadcasting.cpp to test operations across multiple data types:
 * - Float32, Float64 (floating-point operations)
 * - Int32 (integer operations)
 *
 * Broadcasting operations are dtype-agnostic and work with all numeric types.
 * Coverage improvement: tests now run with 3 dtypes instead of just Float32
 */

// ============================================================================
// Multi-DType Test Fixture
// ============================================================================

struct BackendDTypeParam {
    std::string backend_name;
    DType dtype;
    std::string dtype_name;

    std::string ToString() const {
        return backend_name + "_" + dtype_name;
    }
};

// Required for gtest_discover_tests to show human-readable test names
void PrintTo(const BackendDTypeParam& param, std::ostream* os) {
    *os << param.ToString();
}

class BroadcastingMultiDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    Device device;
    DType dtype;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype = param.dtype;

        if (param.backend_name == "cpu") {
            device = Device::cpu();
        }
        else if (param.backend_name == "cuda") {
            if (!isBackendAvailable(Device::Type::CUDA)) {
                GTEST_SKIP() << "CUDA not available";
            }
            device = Device::cuda(0);
        }
        else if (param.backend_name == "vulkan") {
            if (!isBackendAvailable(Device::Type::Vulkan)) {
                GTEST_SKIP() << "Vulkan not available";
            }
            device = Device::vulkan(0);
        }
        else if (param.backend_name == "oneapi") {
            if (!isBackendAvailable(Device::Type::OneAPI)) {
                GTEST_SKIP() << "OneAPI not available";
            }
            device = Device::oneapi(0);
        }
    }

    static bool isBackendAvailable(Device::Type type) {
        try {
            Device test_device{type, 0};
            auto t = zeros({2, 2}, DType::Float32, test_device);
            return true;
        } catch (...) {
            return false;
        }
    }

    // Helper to set tensor data based on dtype
    template<typename T>
    void SetData(Tensor& t, const std::vector<T>& values) {
        auto t_cpu = t.to(Device::cpu());
        T* data = t_cpu.data<T>();
        for (size_t i = 0; i < values.size() && i < t.numel(); i++) {
            data[i] = values[i];
        }
        t = t_cpu.to(device);
    }

    // Helper to verify tensor data based on dtype
    template<typename T>
    void VerifyData(const Tensor& t, const std::vector<T>& expected) {
        auto t_cpu = t.to(Device::cpu());
        const T* data = t_cpu.data<T>();
        for (size_t i = 0; i < expected.size() && i < t.numel(); i++) {
            if constexpr (std::is_floating_point_v<T>) {
                if constexpr (std::is_same_v<T, float>) {
                    EXPECT_FLOAT_EQ(data[i], expected[i])
                        << "Failed at index " << i << " on " << device.to_string();
                } else {
                    EXPECT_DOUBLE_EQ(data[i], expected[i])
                        << "Failed at index " << i << " on " << device.to_string();
                }
            } else {
                EXPECT_EQ(data[i], expected[i])
                    << "Failed at index " << i << " on " << device.to_string();
            }
        }
    }
};

// ============================================================================
// Broadcasting Tests
// ============================================================================

TEST_P(BroadcastingMultiDTypeTest, AddBroadcast_ScalarToVector) {
    // Test: (3,) + (1,) -> (3,)
    auto a = ones({3}, dtype, device);
    auto b = ones({1}, dtype, device);

    switch(dtype) {
        case DType::Float32:
            SetData<float>(a, {1.0f, 2.0f, 3.0f});
            SetData<float>(b, {10.0f});
            break;
        case DType::Float64:
            SetData<double>(a, {1.0, 2.0, 3.0});
            SetData<double>(b, {10.0});
            break;
        case DType::Int32:
            SetData<int32_t>(a, {1, 2, 3});
            SetData<int32_t>(b, {10});
            break;
        default:
            FAIL() << "Unsupported dtype";
    }

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 3) << "Failed on " << device.to_string();

    switch(dtype) {
        case DType::Float32:
            VerifyData<float>(c, {11.0f, 12.0f, 13.0f});
            break;
        case DType::Float64:
            VerifyData<double>(c, {11.0, 12.0, 13.0});
            break;
        case DType::Int32:
            VerifyData<int32_t>(c, {11, 12, 13});
            break;
        default:
            FAIL() << "Unsupported dtype";
    }
}

TEST_P(BroadcastingMultiDTypeTest, AddBroadcast_RowToMatrix) {
    // Test: (2, 3) + (1, 3) -> (2, 3)
    auto a = ones({2, 3}, dtype, device);
    auto b = ones({1, 3}, dtype, device);

    switch(dtype) {
        case DType::Float32:
            SetData<float>(a, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
            SetData<float>(b, {10.0f, 20.0f, 30.0f});
            break;
        case DType::Float64:
            SetData<double>(a, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
            SetData<double>(b, {10.0, 20.0, 30.0});
            break;
        case DType::Int32:
            SetData<int32_t>(a, {1, 2, 3, 4, 5, 6});
            SetData<int32_t>(b, {10, 20, 30});
            break;
        default:
            FAIL() << "Unsupported dtype";
    }

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c.shape()[1], 3) << "Failed on " << device.to_string();

    // Expected: [[11, 22, 33], [14, 25, 36]]
    switch(dtype) {
        case DType::Float32:
            VerifyData<float>(c, {11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f});
            break;
        case DType::Float64:
            VerifyData<double>(c, {11.0, 22.0, 33.0, 14.0, 25.0, 36.0});
            break;
        case DType::Int32:
            VerifyData<int32_t>(c, {11, 22, 33, 14, 25, 36});
            break;
        default:
            FAIL() << "Unsupported dtype";
    }
}

TEST_P(BroadcastingMultiDTypeTest, AddBroadcast_ColumnToMatrix) {
    // Test: (2, 3) + (2, 1) -> (2, 3)
    auto a = ones({2, 3}, dtype, device);
    auto b = ones({2, 1}, dtype, device);

    switch(dtype) {
        case DType::Float32:
            SetData<float>(a, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
            SetData<float>(b, {100.0f, 200.0f});
            break;
        case DType::Float64:
            SetData<double>(a, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
            SetData<double>(b, {100.0, 200.0});
            break;
        case DType::Int32:
            SetData<int32_t>(a, {1, 2, 3, 4, 5, 6});
            SetData<int32_t>(b, {100, 200});
            break;
        default:
            FAIL() << "Unsupported dtype";
    }

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c.shape()[1], 3) << "Failed on " << device.to_string();

    // Expected: [[101, 102, 103], [204, 205, 206]]
    switch(dtype) {
        case DType::Float32:
            VerifyData<float>(c, {101.0f, 102.0f, 103.0f, 204.0f, 205.0f, 206.0f});
            break;
        case DType::Float64:
            VerifyData<double>(c, {101.0, 102.0, 103.0, 204.0, 205.0, 206.0});
            break;
        case DType::Int32:
            VerifyData<int32_t>(c, {101, 102, 103, 204, 205, 206});
            break;
        default:
            FAIL() << "Unsupported dtype";
    }
}

TEST_P(BroadcastingMultiDTypeTest, AddBroadcast_ScalarToMatrix) {
    // Test: (2, 3) + (1, 1) -> (2, 3)
    auto a = ones({2, 3}, dtype, device);
    auto b = ones({1, 1}, dtype, device);

    switch(dtype) {
        case DType::Float32:
            SetData<float>(a, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
            SetData<float>(b, {1000.0f});
            break;
        case DType::Float64:
            SetData<double>(a, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
            SetData<double>(b, {1000.0});
            break;
        case DType::Int32:
            SetData<int32_t>(a, {1, 2, 3, 4, 5, 6});
            SetData<int32_t>(b, {1000});
            break;
        default:
            FAIL() << "Unsupported dtype";
    }

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c.shape()[1], 3) << "Failed on " << device.to_string();

    // Expected: [[1001, 1002, 1003], [1004, 1005, 1006]]
    switch(dtype) {
        case DType::Float32:
            VerifyData<float>(c, {1001.0f, 1002.0f, 1003.0f, 1004.0f, 1005.0f, 1006.0f});
            break;
        case DType::Float64:
            VerifyData<double>(c, {1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0});
            break;
        case DType::Int32:
            VerifyData<int32_t>(c, {1001, 1002, 1003, 1004, 1005, 1006});
            break;
        default:
            FAIL() << "Unsupported dtype";
    }
}

TEST_P(BroadcastingMultiDTypeTest, AddBroadcast_DifferentDimensions) {
    // Test: (3, 2) + (2,) -> (3, 2)
    auto a = ones({3, 2}, dtype, device);
    auto b = ones({2}, dtype, device);

    switch(dtype) {
        case DType::Float32:
            SetData<float>(a, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
            SetData<float>(b, {10.0f, 20.0f});
            break;
        case DType::Float64:
            SetData<double>(a, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
            SetData<double>(b, {10.0, 20.0});
            break;
        case DType::Int32:
            SetData<int32_t>(a, {1, 2, 3, 4, 5, 6});
            SetData<int32_t>(b, {10, 20});
            break;
        default:
            FAIL() << "Unsupported dtype";
    }

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 3) << "Failed on " << device.to_string();
    EXPECT_EQ(c.shape()[1], 2) << "Failed on " << device.to_string();

    // Expected: [[11, 22], [13, 24], [15, 26]]
    switch(dtype) {
        case DType::Float32:
            VerifyData<float>(c, {11.0f, 22.0f, 13.0f, 24.0f, 15.0f, 26.0f});
            break;
        case DType::Float64:
            VerifyData<double>(c, {11.0, 22.0, 13.0, 24.0, 15.0, 26.0});
            break;
        case DType::Int32:
            VerifyData<int32_t>(c, {11, 22, 13, 24, 15, 26});
            break;
        default:
            FAIL() << "Unsupported dtype";
    }
}

TEST_P(BroadcastingMultiDTypeTest, AddNoBroadcast_SameShape) {
    // Test that same-shape operations still work (fast path)
    auto a = ones({2, 3}, dtype, device);
    auto b = ones({2, 3}, dtype, device);

    switch(dtype) {
        case DType::Float32: {
            std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            std::vector<float> b_data = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
            SetData<float>(a, a_data);
            SetData<float>(b, b_data);
            break;
        }
        case DType::Float64: {
            std::vector<double> a_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
            std::vector<double> b_data = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
            SetData<double>(a, a_data);
            SetData<double>(b, b_data);
            break;
        }
        case DType::Int32: {
            std::vector<int32_t> a_data = {1, 2, 3, 4, 5, 6};
            std::vector<int32_t> b_data = {10, 20, 30, 40, 50, 60};
            SetData<int32_t>(a, a_data);
            SetData<int32_t>(b, b_data);
            break;
        }
        default:
            FAIL() << "Unsupported dtype";
    }

    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c.shape()[1], 3) << "Failed on " << device.to_string();

    switch(dtype) {
        case DType::Float32:
            VerifyData<float>(c, {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f});
            break;
        case DType::Float64:
            VerifyData<double>(c, {11.0, 22.0, 33.0, 44.0, 55.0, 66.0});
            break;
        case DType::Int32:
            VerifyData<int32_t>(c, {11, 22, 33, 44, 55, 66});
            break;
        default:
            FAIL() << "Unsupported dtype";
    }
}

// ============================================================================
// Test Instantiation
// ============================================================================

std::vector<BackendDTypeParam> GenerateBackendDTypeCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi"};

    // Test with these dtypes - broadcasting works with all numeric types
    std::vector<std::pair<DType, std::string>> dtypes = {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
        {DType::Int32, "int32"},
    };

    std::vector<BackendDTypeParam> combinations;
    for (const auto& backend : backends) {
        for (const auto& [dtype, dtype_name] : dtypes) {
            combinations.push_back({backend, dtype, dtype_name});
        }
    }
    return combinations;
}

INSTANTIATE_TEST_SUITE_P(
    AllBackendsAllDTypes,
    BroadcastingMultiDTypeTest,
    ::testing::ValuesIn(GenerateBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

/*
 * COVERAGE IMPACT SUMMARY:
 *
 * Original test_broadcasting.cpp:
 * - 7 tests × 4 backends × 1 dtype (Float32) = 28 test scenarios
 * - One test already included Int32, so partial multi-dtype coverage
 *
 * Refactored test_broadcasting_multidtype.cpp:
 * - 6 tests × 4 backends × 3 dtypes (Float32, Float64, Int32) = 72 test scenarios
 * - All tests now systematically test Float32, Float64, and Int32
 *
 * Coverage increase: ~3x improvement
 *
 * Tests converted: 6/7 (removed redundant AddBroadcast_Int32 as all tests now support Int32)
 * - AddBroadcast_ScalarToVector
 * - AddBroadcast_RowToMatrix
 * - AddBroadcast_ColumnToMatrix
 * - AddBroadcast_ScalarToMatrix
 * - AddBroadcast_DifferentDimensions
 * - AddNoBroadcast_SameShape
 *
 * DTypes tested:
 * - Float32 (original)
 * - Float64 (NEW - adds double precision testing)
 * - Int32 (expanded - now tested in all scenarios instead of just one)
 *
 * Broadcasting operations are dtype-agnostic and work uniformly across numeric types.
 * This refactoring ensures comprehensive coverage across floating-point and integer types.
 *
 * Estimated coverage increase: From ~10% (mostly Float32) to ~30% (3 numeric dtypes)
 */
