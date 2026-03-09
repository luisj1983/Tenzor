/**
 * @file test_transforms_multidtype.cpp
 * @brief Multi-dtype tests for tensor transforms
 *
 * Coverage: Backend × DType testing for transform operations
 * - All backends: CPU, CUDA, Vulkan, OneAPI
 * - Primary dtypes: Float32, Float64, Int32
 *
 * Operations tested:
 * - Reshape, view, transpose, permute
 * - Squeeze, unsqueeze, flatten
 * - Combined operations
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>

using namespace tenzor;

// ============================================================================
// Backend + DType Parameterization
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

class TransformMultiDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    Device device;
    DType dtype;
    Tensor t2d;
    Tensor t3d;
    Tensor t4d;

    void SetUp() override {
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
        else if (param.backend_name == "rocm") {
            if (!isBackendAvailable(Device::Type::ROCm)) {
                GTEST_SKIP() << "ROCm not available";
            }
            device = Device::rocm(0);
        }

        // Create test tensors
        t2d = zeros({2, 3}, dtype, device);
        t3d = zeros({2, 3, 4}, dtype, device);
        t4d = zeros({2, 3, 4, 5}, dtype, device);
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
};

// ==============================================================================
// Reshape Tests
// ==============================================================================

TEST_P(TransformMultiDTypeTest, Reshape_Basic) {
    auto t = zeros({6}, dtype, device);
    auto reshaped = t.reshape({2, 3});

    EXPECT_EQ(reshaped.ndim(), 2);
    EXPECT_EQ(reshaped.shape()[0], 2);
    EXPECT_EQ(reshaped.shape()[1], 3);
    EXPECT_EQ(reshaped.numel(), 6);
}

TEST_P(TransformMultiDTypeTest, Reshape_InferDimension) {
    auto t = zeros({12}, dtype, device);
    auto reshaped = t.reshape({3, -1});

    EXPECT_EQ(reshaped.shape()[0], 3);
    EXPECT_EQ(reshaped.shape()[1], 4);
}

TEST_P(TransformMultiDTypeTest, Reshape_MultiDimensional) {
    auto t = zeros({2, 3, 4}, dtype, device);
    auto reshaped = t.reshape({6, 4});

    EXPECT_EQ(reshaped.ndim(), 2);
    EXPECT_EQ(reshaped.shape()[0], 6);
    EXPECT_EQ(reshaped.shape()[1], 4);
}

// ==============================================================================
// View Tests
// ==============================================================================

TEST_P(TransformMultiDTypeTest, View_Basic) {
    auto t = zeros({6}, dtype, device);
    auto viewed = t.view({2, 3});

    EXPECT_EQ(viewed.ndim(), 2);
    EXPECT_EQ(viewed.shape()[0], 2);
    EXPECT_EQ(viewed.shape()[1], 3);
}

TEST_P(TransformMultiDTypeTest, View_SharesStorage) {
    auto t = ones({6}, dtype, device);
    auto viewed = t.view({2, 3});

    EXPECT_EQ(t.storage(), viewed.storage());
}

// ==============================================================================
// Transpose Tests
// ==============================================================================

TEST_P(TransformMultiDTypeTest, Transpose_2D) {
    auto t = t2d.transpose(0, 1);

    EXPECT_EQ(t.shape()[0], 3);
    EXPECT_EQ(t.shape()[1], 2);
}

TEST_P(TransformMultiDTypeTest, Transpose_NegativeDims) {
    auto t = t2d.transpose(-2, -1);

    EXPECT_EQ(t.shape()[0], 3);
    EXPECT_EQ(t.shape()[1], 2);
}

TEST_P(TransformMultiDTypeTest, Transpose_3D) {
    auto t = t3d.transpose(0, 2);

    EXPECT_EQ(t.shape()[0], 4);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_EQ(t.shape()[2], 2);
}

// ==============================================================================
// Permute Tests
// ==============================================================================

TEST_P(TransformMultiDTypeTest, Permute_3D) {
    auto t = t3d.permute({2, 0, 1});

    EXPECT_EQ(t.shape()[0], 4);
    EXPECT_EQ(t.shape()[1], 2);
    EXPECT_EQ(t.shape()[2], 3);
}

TEST_P(TransformMultiDTypeTest, Permute_Reverse) {
    auto t = t3d.permute({2, 1, 0});

    EXPECT_EQ(t.shape()[0], 4);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_EQ(t.shape()[2], 2);
}

TEST_P(TransformMultiDTypeTest, Permute_NegativeIndices) {
    auto t = t3d.permute({-1, -2, -3});

    EXPECT_EQ(t.shape()[0], 4);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_EQ(t.shape()[2], 2);
}

// ==============================================================================
// Squeeze Tests
// ==============================================================================

TEST_P(TransformMultiDTypeTest, Squeeze_SingleDim) {
    auto t = zeros({2, 1, 3}, dtype, device);
    auto squeezed = t.squeeze(1);

    EXPECT_EQ(squeezed.ndim(), 2);
    EXPECT_EQ(squeezed.shape()[0], 2);
    EXPECT_EQ(squeezed.shape()[1], 3);
}

TEST_P(TransformMultiDTypeTest, Squeeze_All) {
    auto t = zeros({1, 2, 1, 3, 1}, dtype, device);
    auto squeezed = t.squeeze();

    EXPECT_EQ(squeezed.ndim(), 2);
    EXPECT_EQ(squeezed.shape()[0], 2);
    EXPECT_EQ(squeezed.shape()[1], 3);
}

TEST_P(TransformMultiDTypeTest, Squeeze_AllOnes) {
    auto t = zeros({1, 1, 1}, dtype, device);
    auto squeezed = t.squeeze();

    // PyTorch behavior: squeezing all singleton dims yields a 0-D scalar tensor
    EXPECT_EQ(squeezed.ndim(), 0);
    EXPECT_EQ(squeezed.numel(), 1);
}

TEST_P(TransformMultiDTypeTest, Squeeze_NegativeIndex) {
    auto t = zeros({2, 1, 3}, dtype, device);
    auto squeezed = t.squeeze(-2);

    EXPECT_EQ(squeezed.ndim(), 2);
    EXPECT_EQ(squeezed.shape()[0], 2);
    EXPECT_EQ(squeezed.shape()[1], 3);
}

// ==============================================================================
// Unsqueeze Tests
// ==============================================================================

TEST_P(TransformMultiDTypeTest, Unsqueeze_Front) {
    auto t = zeros({2, 3}, dtype, device);
    auto unsqueezed = t.unsqueeze(0);

    EXPECT_EQ(unsqueezed.ndim(), 3);
    EXPECT_EQ(unsqueezed.shape()[0], 1);
    EXPECT_EQ(unsqueezed.shape()[1], 2);
    EXPECT_EQ(unsqueezed.shape()[2], 3);
}

TEST_P(TransformMultiDTypeTest, Unsqueeze_Middle) {
    auto t = zeros({2, 3}, dtype, device);
    auto unsqueezed = t.unsqueeze(1);

    EXPECT_EQ(unsqueezed.ndim(), 3);
    EXPECT_EQ(unsqueezed.shape()[0], 2);
    EXPECT_EQ(unsqueezed.shape()[1], 1);
    EXPECT_EQ(unsqueezed.shape()[2], 3);
}

TEST_P(TransformMultiDTypeTest, Unsqueeze_End) {
    auto t = zeros({2, 3}, dtype, device);
    auto unsqueezed = t.unsqueeze(2);

    EXPECT_EQ(unsqueezed.ndim(), 3);
    EXPECT_EQ(unsqueezed.shape()[0], 2);
    EXPECT_EQ(unsqueezed.shape()[1], 3);
    EXPECT_EQ(unsqueezed.shape()[2], 1);
}

TEST_P(TransformMultiDTypeTest, Unsqueeze_NegativeIndex) {
    auto t = zeros({2, 3}, dtype, device);
    auto unsqueezed = t.unsqueeze(-1);

    EXPECT_EQ(unsqueezed.ndim(), 3);
    EXPECT_EQ(unsqueezed.shape()[0], 2);
    EXPECT_EQ(unsqueezed.shape()[1], 3);
    EXPECT_EQ(unsqueezed.shape()[2], 1);
}

// ==============================================================================
// Flatten Tests
// ==============================================================================

TEST_P(TransformMultiDTypeTest, Flatten_All) {
    auto t = zeros({2, 3, 4}, dtype, device);
    auto flattened = t.flatten();

    EXPECT_EQ(flattened.ndim(), 1);
    EXPECT_EQ(flattened.shape()[0], 24);
}

TEST_P(TransformMultiDTypeTest, Flatten_Partial) {
    auto t = zeros({2, 3, 4, 5}, dtype, device);
    auto flattened = t.flatten(1, 2);

    EXPECT_EQ(flattened.ndim(), 3);
    EXPECT_EQ(flattened.shape()[0], 2);
    EXPECT_EQ(flattened.shape()[1], 12);
    EXPECT_EQ(flattened.shape()[2], 5);
}

TEST_P(TransformMultiDTypeTest, Flatten_FirstTwoDims) {
    auto t = zeros({2, 3, 4}, dtype, device);
    auto flattened = t.flatten(0, 1);

    EXPECT_EQ(flattened.ndim(), 2);
    EXPECT_EQ(flattened.shape()[0], 6);
    EXPECT_EQ(flattened.shape()[1], 4);
}

TEST_P(TransformMultiDTypeTest, Flatten_NegativeIndices) {
    auto t = zeros({2, 3, 4}, dtype, device);
    auto flattened = t.flatten(-2, -1);

    EXPECT_EQ(flattened.ndim(), 2);
    EXPECT_EQ(flattened.shape()[0], 2);
    EXPECT_EQ(flattened.shape()[1], 12);
}

// ==============================================================================
// Combined Operations Tests
// ==============================================================================

TEST_P(TransformMultiDTypeTest, Combined_UnsqueezeSqueeze) {
    auto t = zeros({2, 3}, dtype, device);
    auto result = t.unsqueeze(1).squeeze(1);

    EXPECT_EQ(result.ndim(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 3);
}

TEST_P(TransformMultiDTypeTest, Combined_PermuteTranspose) {
    auto t = zeros({2, 3, 4}, dtype, device);
    auto permuted = t.permute({2, 1, 0});
    auto transposed = permuted.transpose(0, 2);

    EXPECT_EQ(transposed.shape()[0], 2);
    EXPECT_EQ(transposed.shape()[1], 3);
    EXPECT_EQ(transposed.shape()[2], 4);
}

// ==============================================================================
// Test Instantiation
// ==============================================================================

std::vector<BackendDTypeParam> GenerateTransformCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi", "rocm"};

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
    AllBackendsCommonDTypes,
    TransformMultiDTypeTest,
    ::testing::ValuesIn(GenerateTransformCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Test Environment Setup
// ============================================================================

class TransformTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        initialize();
    }
};

static ::testing::Environment* const transform_env =
    ::testing::AddGlobalTestEnvironment(new TransformTestEnvironment);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
