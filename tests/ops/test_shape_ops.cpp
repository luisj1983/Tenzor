#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>

using namespace tenzor;

/**
 * @file test_shape_ops.cpp
 * @brief Backend-agnostic tests for shape manipulation operations
 *
 * Tests: squeeze, unsqueeze
 * All backends: CPU, CUDA, Vulkan, OneAPI, ROCm
 * All dtypes: Float32, Float64, Int32, Int64, Bool, UInt8
 *
 * Shape operations are dtype-agnostic - they only manipulate dimensions,
 * not data values. Testing with all dtypes ensures this property holds.
 */

class ShapeOpsTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const shape_env =
    ::testing::AddGlobalTestEnvironment(new ShapeOpsTestEnvironment);

// ============================================================================
// Parameterization: Backend + DType
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

class ShapeOpsTest : public ::testing::TestWithParam<BackendDTypeParam> {
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
        else if (param.backend_name == "adaptivecpp") {
            if (!isBackendAvailable(Device::Type::AdaptiveCpp)) {
                GTEST_SKIP() << "AdaptiveCpp not available";
            }
            device = Device::adaptivecpp(0);
        }
        else if (param.backend_name == "rocm") {
            if (!isBackendAvailable(Device::Type::ROCm)) {
                GTEST_SKIP() << "ROCm not available";
            }
            device = Device::rocm(0);
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

    // Helper to verify data preservation across different dtypes
    template<typename T>
    void verifySequentialData(const Tensor& t, size_t count) {
        auto t_cpu = t.to(Device::cpu());
        const T* data = t_cpu.data<T>();

        for (size_t i = 0; i < count; ++i) {
            EXPECT_EQ(data[i], static_cast<T>(i));
        }
    }

    void verifyDataPreservation(const Tensor& t, size_t count) {
        if (dtype == DType::Float32) {
            verifySequentialData<float>(t, count);
        }
        else if (dtype == DType::Float64) {
            verifySequentialData<double>(t, count);
        }
        else if (dtype == DType::Int32) {
            verifySequentialData<int32_t>(t, count);
        }
        else if (dtype == DType::Int64) {
            verifySequentialData<int64_t>(t, count);
        }
        else if (dtype == DType::Bool) {
            auto t_cpu = t.to(Device::cpu());
            const bool* data = t_cpu.data<bool>();
            for (size_t i = 0; i < count; ++i) {
                EXPECT_EQ(data[i], (i % 2 == 1));
            }
        }
        else if (dtype == DType::UInt8) {
            verifySequentialData<uint8_t>(t, count);
        }
    }
};

// ============================================================================
// Squeeze Tests
// ============================================================================

TEST_P(ShapeOpsTest, SqueezeAll) {
    // Remove all dimensions of size 1
    auto input = ones({1, 3, 1, 5, 1}, dtype, device);
    auto output = squeeze(input);

    // Should become [3, 5]
    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 3);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.device().type, device.type);
    EXPECT_EQ(output.dtype(), dtype);  // DType preserved
}

TEST_P(ShapeOpsTest, SqueezeSpecificDim) {
    // Remove specific dimension
    auto input = ones({1, 3, 1, 5}, dtype, device);
    auto output = squeeze(input, 2);  // Remove dim 2 (size 1)

    // Should become [1, 3, 5]
    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 5);
    EXPECT_EQ(output.dtype(), dtype);  // DType preserved
}

TEST_P(ShapeOpsTest, SqueezeNonSingletonDim) {
    // Squeezing a non-size-1 dimension should not change shape
    auto input = ones({2, 3, 4}, dtype, device);
    auto output = squeeze(input, 1);  // dim 1 has size 3

    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.dtype(), dtype);  // DType preserved
}

TEST_P(ShapeOpsTest, SqueezeNoSingletonDims) {
    // Tensor with no size-1 dimensions
    auto input = ones({2, 3, 4, 5}, dtype, device);
    auto output = squeeze(input);

    // Should remain unchanged
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 5);
    EXPECT_EQ(output.dtype(), dtype);  // DType preserved
}

TEST_P(ShapeOpsTest, SqueezeScalar) {
    // Edge case: all dimensions are 1 -> becomes scalar
    auto input = ones({1, 1, 1}, dtype, device);
    auto output = squeeze(input);

    EXPECT_EQ(output.shape().size(), 0);  // Scalar
    EXPECT_EQ(output.numel(), 1);
    EXPECT_EQ(output.dtype(), dtype);  // DType preserved
}

// ============================================================================
// Unsqueeze Tests
// ============================================================================

TEST_P(ShapeOpsTest, UnsqueezeBeginning) {
    // Add dimension at the start
    auto input = ones({3, 4, 5}, dtype, device);
    auto output = unsqueeze(input, 0);

    // Should become [1, 3, 4, 5]
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 5);
    EXPECT_EQ(output.device().type, device.type);
    EXPECT_EQ(output.dtype(), dtype);  // DType preserved
}

TEST_P(ShapeOpsTest, UnsqueezeMiddle) {
    // Add dimension in the middle
    auto input = ones({2, 3, 4}, dtype, device);
    auto output = unsqueeze(input, 2);

    // Should become [2, 3, 1, 4]
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 1);
    EXPECT_EQ(output.shape()[3], 4);
    EXPECT_EQ(output.dtype(), dtype);  // DType preserved
}

TEST_P(ShapeOpsTest, UnsqueezeEnd) {
    // Add dimension at the end
    auto input = ones({2, 3}, dtype, device);
    auto output = unsqueeze(input, 2);

    // Should become [2, 3, 1]
    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 1);
    EXPECT_EQ(output.dtype(), dtype);  // DType preserved
}

TEST_P(ShapeOpsTest, UnsqueezeNegativeDim) {
    // Negative dimension index
    auto input = ones({2, 3, 4}, dtype, device);
    auto output = unsqueeze(input, -1);  // Add at end

    // Should become [2, 3, 4, 1]
    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 1);
    EXPECT_EQ(output.dtype(), dtype);  // DType preserved
}

TEST_P(ShapeOpsTest, UnsqueezeScalar) {
    // Unsqueeze a scalar tensor
    auto input = ones({}, dtype, device);  // Scalar
    auto output = unsqueeze(input, 0);

    EXPECT_EQ(output.shape().size(), 1);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.dtype(), dtype);  // DType preserved
}

// ============================================================================
// Squeeze/Unsqueeze Round-trip Tests
// ============================================================================

TEST_P(ShapeOpsTest, SqueezeUnsqueezeRoundTrip) {
    // Original -> Unsqueeze -> Squeeze -> Should match original
    auto original = ones({2, 3, 4}, dtype, device);
    auto unsqueezed = unsqueeze(original, 1);  // [2, 1, 3, 4]
    auto squeezed = squeeze(unsqueezed, 1);    // [2, 3, 4]

    EXPECT_EQ(squeezed.shape().size(), 3);
    EXPECT_EQ(squeezed.shape()[0], 2);
    EXPECT_EQ(squeezed.shape()[1], 3);
    EXPECT_EQ(squeezed.shape()[2], 4);
    EXPECT_EQ(squeezed.dtype(), dtype);  // DType preserved through round trip
}

TEST_P(ShapeOpsTest, MultipleUnsqueezes) {
    // Add multiple dimensions
    auto input = ones({2, 3}, dtype, device);
    auto out1 = unsqueeze(input, 0);      // [1, 2, 3]
    auto out2 = unsqueeze(out1, 3);        // [1, 2, 3, 1]
    auto out3 = unsqueeze(out2, 2);        // [1, 2, 1, 3, 1]

    EXPECT_EQ(out3.shape().size(), 5);
    EXPECT_EQ(out3.shape()[0], 1);
    EXPECT_EQ(out3.shape()[1], 2);
    EXPECT_EQ(out3.shape()[2], 1);
    EXPECT_EQ(out3.shape()[3], 3);
    EXPECT_EQ(out3.shape()[4], 1);
    EXPECT_EQ(out3.dtype(), dtype);  // DType preserved through multiple operations
}

// ============================================================================
// Data Preservation Tests
// ============================================================================

TEST_P(ShapeOpsTest, SqueezePreservesData) {
    // Ensure data is unchanged, only shape changes
    // Create test data based on dtype
    Tensor input_cpu;
    if (dtype == DType::Float32) {
        input_cpu = arange(0.0f, 12.0f, 1.0f, dtype, Device::cpu()).reshape({1, 3, 4, 1});
    }
    else if (dtype == DType::Float64) {
        input_cpu = arange(0.0, 12.0, 1.0, dtype, Device::cpu()).reshape({1, 3, 4, 1});
    }
    else if (dtype == DType::Int32 || dtype == DType::Int64) {
        input_cpu = arange(0, 12, 1, dtype, Device::cpu()).reshape({1, 3, 4, 1});
    }
    else if (dtype == DType::UInt8) {
        input_cpu = arange(0, 12, 1, dtype, Device::cpu()).reshape({1, 3, 4, 1});
    }
    else if (dtype == DType::Bool) {
        // For Bool, create alternating true/false pattern
        auto temp = arange(0, 12, 1, DType::Int32, Device::cpu());
        input_cpu = (temp % 2).to(DType::Bool).reshape({1, 3, 4, 1});
    }

    auto input = (device.type == Device::Type::CPU) ? input_cpu : input_cpu.to(device);
    auto squeezed = squeeze(input);  // [3, 4]

    // Verify shape and data preservation
    EXPECT_EQ(squeezed.shape().size(), 2);
    EXPECT_EQ(squeezed.dtype(), dtype);
    verifyDataPreservation(squeezed, 12);
}

TEST_P(ShapeOpsTest, UnsqueezePreservesData) {
    // Create test data based on dtype
    Tensor input_cpu;
    if (dtype == DType::Float32) {
        input_cpu = arange(0.0f, 6.0f, 1.0f, dtype, Device::cpu()).reshape({2, 3});
    }
    else if (dtype == DType::Float64) {
        input_cpu = arange(0.0, 6.0, 1.0, dtype, Device::cpu()).reshape({2, 3});
    }
    else if (dtype == DType::Int32 || dtype == DType::Int64) {
        input_cpu = arange(0, 6, 1, dtype, Device::cpu()).reshape({2, 3});
    }
    else if (dtype == DType::UInt8) {
        input_cpu = arange(0, 6, 1, dtype, Device::cpu()).reshape({2, 3});
    }
    else if (dtype == DType::Bool) {
        auto temp = arange(0, 6, 1, DType::Int32, Device::cpu());
        input_cpu = (temp % 2).to(DType::Bool).reshape({2, 3});
    }

    auto input = (device.type == Device::Type::CPU) ? input_cpu : input_cpu.to(device);
    auto unsqueezed = unsqueeze(input, 1);  // [2, 1, 3]

    // Verify shape and data preservation
    EXPECT_EQ(unsqueezed.shape().size(), 3);
    EXPECT_EQ(unsqueezed.dtype(), dtype);
    verifyDataPreservation(unsqueezed, 6);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_P(ShapeOpsTest, BatchDimensionHandling) {
    // Common pattern: add batch dimension
    auto input = ones({3, 224, 224}, dtype, device);  // Single image
    auto batched = unsqueeze(input, 0);  // [1, 3, 224, 224]

    EXPECT_EQ(batched.shape()[0], 1);
    EXPECT_EQ(batched.shape()[1], 3);
    EXPECT_EQ(batched.shape()[2], 224);
    EXPECT_EQ(batched.shape()[3], 224);
    EXPECT_EQ(batched.dtype(), dtype);  // DType preserved

    // Remove batch dimension
    auto unbatched = squeeze(batched, 0);
    EXPECT_EQ(unbatched.shape().size(), 3);
    EXPECT_EQ(unbatched.shape()[0], 3);
    EXPECT_EQ(unbatched.shape()[1], 224);
    EXPECT_EQ(unbatched.shape()[2], 224);
    EXPECT_EQ(unbatched.dtype(), dtype);  // DType preserved
}

// ============================================================================
// Test Parameterization
// ============================================================================

std::vector<BackendDTypeParam> GenerateShapeOpsParams() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi", "adaptivecpp", "rocm"};

    // Shape operations work with ALL dtypes - they only manipulate dimensions
    std::vector<std::pair<DType, std::string>> dtypes = {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
        {DType::Int32, "int32"},
        {DType::Int64, "int64"},
        {DType::Bool, "bool"},
        {DType::UInt8, "uint8"},
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
    ShapeOpsTest,
    ::testing::ValuesIn(GenerateShapeOpsParams()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

/*
 * COVERAGE ANALYSIS:
 *
 * Test scenarios: 14 tests × 5 backends × 6 dtypes = 420 test scenarios
 *
 * Backends tested:
 * - CPU (always available)
 * - CUDA (if available)
 * - Vulkan (if available)
 * - OneAPI (if available)
 * - ROCm (if available)
 *
 * DTypes tested:
 * - Float32 (standard floating point)
 * - Float64 (double precision)
 * - Int32 (standard integer)
 * - Int64 (long integer)
 * - Bool (boolean values)
 * - UInt8 (unsigned byte)
 *
 * Test categories:
 * 1. Squeeze operations (5 tests)
 * 2. Unsqueeze operations (5 tests)
 * 3. Round-trip tests (2 tests)
 * 4. Data preservation (2 tests)
 * 5. Integration tests (1 test)
 *
 * Key properties verified:
 * - Shape changes are correct for all dtypes
 * - Data is preserved across operations
 * - DType is preserved through all transformations
 * - Operations work consistently across all backends
 */
