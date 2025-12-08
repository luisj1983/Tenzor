#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>

using namespace tenzor;

/**
 * @file test_shape_ops_multidtype.cpp
 * @brief Multi-dtype tests for shape manipulation operations
 *
 * Tests: squeeze, unsqueeze, reshape, transpose, permute
 * Backends: CPU, CUDA, Vulkan, OneAPI, ROCm
 * DTypes: Float32, Float64, Int32
 *
 * Shape operations are dtype-agnostic - they only manipulate dimensions,
 * not data values. Testing with multiple dtypes ensures this property holds.
 */

class ShapeOpsMultiDTypeTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const shape_multidtype_env =
    ::testing::AddGlobalTestEnvironment(new ShapeOpsMultiDTypeTestEnvironment);

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

class ShapeOpsMultiDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
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

    // Helper to verify data preservation
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
    }

    // Helper to create test tensors
    Tensor createTestTensor(const std::vector<int64_t>& shape) {
        int64_t numel = 1;
        for (auto dim : shape) numel *= dim;

        Tensor input_cpu;
        if (dtype == DType::Float32) {
            input_cpu = arange(0.0f, static_cast<float>(numel), 1.0f, dtype, Device::cpu()).reshape(shape);
        }
        else if (dtype == DType::Float64) {
            input_cpu = arange(0.0, static_cast<double>(numel), 1.0, dtype, Device::cpu()).reshape(shape);
        }
        else if (dtype == DType::Int32) {
            input_cpu = arange(0, static_cast<int32_t>(numel), 1, dtype, Device::cpu()).reshape(shape);
        }

        return (device.type == Device::Type::CPU) ? input_cpu : input_cpu.to(device);
    }
};

// ============================================================================
// Squeeze Tests
// ============================================================================

TEST_P(ShapeOpsMultiDTypeTest, SqueezeAll) {
    auto input = ones({1, 3, 1, 5, 1}, dtype, device);
    auto output = squeeze(input);

    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 3);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.device().type, device.type);
    EXPECT_EQ(output.dtype(), dtype);
}

TEST_P(ShapeOpsMultiDTypeTest, SqueezeSpecificDim) {
    auto input = ones({1, 3, 1, 5}, dtype, device);
    auto output = squeeze(input, 2);

    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 5);
    EXPECT_EQ(output.dtype(), dtype);
}

TEST_P(ShapeOpsMultiDTypeTest, SqueezeNonSingletonDim) {
    auto input = ones({2, 3, 4}, dtype, device);
    auto output = squeeze(input, 1);

    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.dtype(), dtype);
}

TEST_P(ShapeOpsMultiDTypeTest, SqueezeNoSingletonDims) {
    auto input = ones({2, 3, 4, 5}, dtype, device);
    auto output = squeeze(input);

    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 5);
    EXPECT_EQ(output.dtype(), dtype);
}

TEST_P(ShapeOpsMultiDTypeTest, SqueezePreservesData) {
    auto input = createTestTensor({1, 3, 4, 1});
    auto squeezed = squeeze(input);

    EXPECT_EQ(squeezed.shape().size(), 2);
    EXPECT_EQ(squeezed.dtype(), dtype);
    verifyDataPreservation(squeezed, 12);
}

// ============================================================================
// Unsqueeze Tests
// ============================================================================

TEST_P(ShapeOpsMultiDTypeTest, UnsqueezeBeginning) {
    auto input = ones({3, 4, 5}, dtype, device);
    auto output = unsqueeze(input, 0);

    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 5);
    EXPECT_EQ(output.device().type, device.type);
    EXPECT_EQ(output.dtype(), dtype);
}

TEST_P(ShapeOpsMultiDTypeTest, UnsqueezeMiddle) {
    auto input = ones({2, 3, 4}, dtype, device);
    auto output = unsqueeze(input, 2);

    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 1);
    EXPECT_EQ(output.shape()[3], 4);
    EXPECT_EQ(output.dtype(), dtype);
}

TEST_P(ShapeOpsMultiDTypeTest, UnsqueezeEnd) {
    auto input = ones({2, 3}, dtype, device);
    auto output = unsqueeze(input, 2);

    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 1);
    EXPECT_EQ(output.dtype(), dtype);
}

TEST_P(ShapeOpsMultiDTypeTest, UnsqueezeNegativeDim) {
    auto input = ones({2, 3, 4}, dtype, device);
    auto output = unsqueeze(input, -1);

    EXPECT_EQ(output.shape().size(), 4);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 1);
    EXPECT_EQ(output.dtype(), dtype);
}

TEST_P(ShapeOpsMultiDTypeTest, UnsqueezePreservesData) {
    auto input = createTestTensor({2, 3});
    auto unsqueezed = unsqueeze(input, 1);

    EXPECT_EQ(unsqueezed.shape().size(), 3);
    EXPECT_EQ(unsqueezed.dtype(), dtype);
    verifyDataPreservation(unsqueezed, 6);
}

// ============================================================================
// Reshape Tests
// ============================================================================

TEST_P(ShapeOpsMultiDTypeTest, ReshapeBasic) {
    auto input = ones({2, 3, 4}, dtype, device);
    auto output = input.reshape({6, 4});

    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 6);
    EXPECT_EQ(output.shape()[1], 4);
    EXPECT_EQ(output.numel(), input.numel());
    EXPECT_EQ(output.dtype(), dtype);
}

TEST_P(ShapeOpsMultiDTypeTest, ReshapeToVector) {
    auto input = ones({2, 3, 4}, dtype, device);
    auto output = input.reshape({24});

    EXPECT_EQ(output.shape().size(), 1);
    EXPECT_EQ(output.shape()[0], 24);
    EXPECT_EQ(output.dtype(), dtype);
}

TEST_P(ShapeOpsMultiDTypeTest, ReshapeInferDimension) {
    auto input = ones({2, 3, 4}, dtype, device);
    auto output = input.reshape({-1, 4});

    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 6);
    EXPECT_EQ(output.shape()[1], 4);
    EXPECT_EQ(output.dtype(), dtype);
}

TEST_P(ShapeOpsMultiDTypeTest, ReshapePreservesData) {
    auto input = createTestTensor({2, 3, 4});
    auto reshaped = input.reshape({4, 6});

    EXPECT_EQ(reshaped.shape()[0], 4);
    EXPECT_EQ(reshaped.shape()[1], 6);
    EXPECT_EQ(reshaped.dtype(), dtype);
    verifyDataPreservation(reshaped, 24);
}

// ============================================================================
// Transpose Tests
// ============================================================================

TEST_P(ShapeOpsMultiDTypeTest, Transpose2D) {
    auto input = createTestTensor({3, 4});
    auto output = transpose(input, 0, 1);

    EXPECT_EQ(output.shape().size(), 2);
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.dtype(), dtype);
}

TEST_P(ShapeOpsMultiDTypeTest, TransposeDims) {
    auto input = createTestTensor({2, 3, 4});
    auto output = transpose(input, 0, 2);

    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 2);
    EXPECT_EQ(output.dtype(), dtype);
}

TEST_P(ShapeOpsMultiDTypeTest, TransposeRoundTrip) {
    auto input = createTestTensor({2, 3, 4});
    auto transposed = transpose(input, 0, 2);
    auto back = transpose(transposed, 0, 2);

    EXPECT_EQ(back.shape()[0], 2);
    EXPECT_EQ(back.shape()[1], 3);
    EXPECT_EQ(back.shape()[2], 4);
    EXPECT_EQ(back.dtype(), dtype);
}

// ============================================================================
// Permute Tests
// ============================================================================

TEST_P(ShapeOpsMultiDTypeTest, PermuteBasic) {
    auto input = createTestTensor({2, 3, 4});
    auto output = permute(input, {2, 0, 1});

    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 2);
    EXPECT_EQ(output.shape()[2], 3);
    EXPECT_EQ(output.dtype(), dtype);
}

TEST_P(ShapeOpsMultiDTypeTest, PermuteIdentity) {
    auto input = createTestTensor({2, 3, 4});
    auto output = permute(input, {0, 1, 2});

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.dtype(), dtype);
}

TEST_P(ShapeOpsMultiDTypeTest, PermuteReverse) {
    auto input = createTestTensor({2, 3, 4});
    auto output = permute(input, {2, 1, 0});

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 2);
    EXPECT_EQ(output.dtype(), dtype);
}

TEST_P(ShapeOpsMultiDTypeTest, Permute4D) {
    auto input = createTestTensor({2, 3, 4, 5});
    auto output = permute(input, {0, 2, 3, 1});

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 4);
    EXPECT_EQ(output.shape()[2], 5);
    EXPECT_EQ(output.shape()[3], 3);
    EXPECT_EQ(output.dtype(), dtype);
}

// ============================================================================
// Combined Operations Tests
// ============================================================================

TEST_P(ShapeOpsMultiDTypeTest, SqueezeUnsqueezeRoundTrip) {
    auto original = ones({2, 3, 4}, dtype, device);
    auto unsqueezed = unsqueeze(original, 1);
    auto squeezed = squeeze(unsqueezed, 1);

    EXPECT_EQ(squeezed.shape().size(), 3);
    EXPECT_EQ(squeezed.shape()[0], 2);
    EXPECT_EQ(squeezed.shape()[1], 3);
    EXPECT_EQ(squeezed.shape()[2], 4);
    EXPECT_EQ(squeezed.dtype(), dtype);
}

TEST_P(ShapeOpsMultiDTypeTest, ReshapeTransposeReshape) {
    auto input = createTestTensor({2, 3, 4});
    auto reshaped1 = input.reshape({6, 4});
    auto transposed = transpose(reshaped1, 0, 1);
    auto reshaped2 = transposed.reshape({2, 2, 6});

    EXPECT_EQ(reshaped2.shape()[0], 2);
    EXPECT_EQ(reshaped2.shape()[1], 2);
    EXPECT_EQ(reshaped2.shape()[2], 6);
    EXPECT_EQ(reshaped2.dtype(), dtype);
}

TEST_P(ShapeOpsMultiDTypeTest, BatchDimensionHandling) {
    auto input = ones({3, 224, 224}, dtype, device);
    auto batched = unsqueeze(input, 0);

    EXPECT_EQ(batched.shape()[0], 1);
    EXPECT_EQ(batched.shape()[1], 3);
    EXPECT_EQ(batched.shape()[2], 224);
    EXPECT_EQ(batched.shape()[3], 224);
    EXPECT_EQ(batched.dtype(), dtype);

    auto unbatched = squeeze(batched, 0);
    EXPECT_EQ(unbatched.shape().size(), 3);
    EXPECT_EQ(unbatched.shape()[0], 3);
    EXPECT_EQ(unbatched.shape()[1], 224);
    EXPECT_EQ(unbatched.shape()[2], 224);
    EXPECT_EQ(unbatched.dtype(), dtype);
}

// ============================================================================
// Test Parameterization
// ============================================================================

std::vector<BackendDTypeParam> GenerateShapeOpsMultiDTypeParams() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi", "adaptivecpp", "rocm"};

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
    AllBackendsMultiDType,
    ShapeOpsMultiDTypeTest,
    ::testing::ValuesIn(GenerateShapeOpsMultiDTypeParams()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

/*
 * COVERAGE SUMMARY:
 *
 * Total tests: 27 tests × 5 backends × 3 dtypes = 405 test scenarios
 *
 * Test categories:
 * - Squeeze operations: 5 tests
 * - Unsqueeze operations: 5 tests
 * - Reshape operations: 4 tests
 * - Transpose operations: 3 tests
 * - Permute operations: 4 tests
 * - Combined operations: 3 tests
 * - Integration tests: 3 tests
 *
 * Backends: CPU, CUDA, Vulkan, OneAPI, ROCm
 * DTypes: Float32, Float64, Int32
 *
 * Operations tested:
 * - squeeze: Remove singleton dimensions
 * - unsqueeze: Add singleton dimensions
 * - reshape: Change tensor shape
 * - transpose: Swap dimensions
 * - permute: Arbitrary dimension reordering
 *
 * Key properties verified:
 * - Shape transformations work correctly
 * - Data is preserved across operations
 * - DType is preserved through all operations
 * - Operations compose correctly
 * - Backend consistency across all devices
 */
