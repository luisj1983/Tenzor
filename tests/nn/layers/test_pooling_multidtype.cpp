#include <gtest/gtest.h>
#include "../../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;

/**
 * @file test_pooling_multidtype.cpp
 * @brief Multi-dtype tests for pooling layers (MaxPool2d, AvgPool2d, AdaptiveAvgPool2d)
 *
 * Tests pooling operations with Float32, Float64, and Float16 dtypes
 * for mixed precision training scenarios.
 */

// ============================================================================
// Multi-DType Parameterization
// ============================================================================

struct DTypeParam {
    DType dtype;
    std::string dtype_name;
    float tolerance;

    std::string ToString() const {
        return dtype_name;
    }
};

// Required for gtest_discover_tests to show human-readable test names
void PrintTo(const DTypeParam& param, std::ostream* os) {
    *os << param.ToString();
}

class PoolingMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    DType dtype;
    float tol;
    Device device;

    void SetUp() override {
        tenzor::initialize();
        auto param = GetParam();
        dtype = param.dtype;
        tol = param.tolerance;
        device = Device::cpu();
    }

    template<typename T>
    bool values_close(const T* a, const T* b, size_t n, float rtol, float atol) {
        for (size_t i = 0; i < n; ++i) {
            float diff = std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
            float threshold = atol + rtol * std::abs(static_cast<float>(b[i]));
            if (diff > threshold) {
                return false;
            }
        }
        return true;
    }
};

// ============================================================================
// MaxPool2d Tests
// ============================================================================

TEST_P(PoolingMultiDTypeTest, MaxPool2dForwardBasic) {
    auto param = GetParam();
    auto pool = nn::MaxPool2d(2, 2, 0);

    auto input_tensor = randn({2, 3, 32, 32}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = pool.forward(input);

    // Check shape
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(PoolingMultiDTypeTest, MaxPool2dMaxValueSelection) {
    auto param = GetParam();
    auto pool = nn::MaxPool2d(2, 2, 0);

    // Create known input
    auto input_tensor = zeros({1, 1, 4, 4}, DType::Float32, device);
    float* data = input_tensor.data<float>();

    data[0] = 1.0f;  data[1] = 2.0f;  data[2] = 3.0f;  data[3] = 4.0f;
    data[4] = 5.0f;  data[5] = 6.0f;  data[6] = 7.0f;  data[7] = 8.0f;
    data[8] = 9.0f;  data[9] = 10.0f; data[10] = 11.0f; data[11] = 12.0f;
    data[12] = 13.0f; data[13] = 14.0f; data[14] = 15.0f; data[15] = 16.0f;

    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = pool.forward(input);

    auto output_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    const float* out_data = output_cpu.data<float>();

    EXPECT_NEAR(out_data[0], 6.0f, tol);
    EXPECT_NEAR(out_data[1], 8.0f, tol);
    EXPECT_NEAR(out_data[2], 14.0f, tol);
    EXPECT_NEAR(out_data[3], 16.0f, tol);
}

TEST_P(PoolingMultiDTypeTest, MaxPool2dGradientFlow) {
    auto param = GetParam();
    auto pool = nn::MaxPool2d(2);

    auto input_tensor = randn({2, 3, 16, 16}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(out_shape_vec, dtype, device);

    EXPECT_NO_THROW({
        output.backward(grad_output);
    });

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype);
}

// ============================================================================
// AvgPool2d Tests
// ============================================================================

TEST_P(PoolingMultiDTypeTest, AvgPool2dForwardBasic) {
    auto param = GetParam();
    auto pool = nn::AvgPool2d(2, 2, 0);

    auto input_tensor = randn({2, 3, 32, 32}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(PoolingMultiDTypeTest, AvgPool2dAverageComputation) {
    auto param = GetParam();
    auto pool = nn::AvgPool2d(2, 2, 0);

    auto input_tensor = zeros({1, 1, 4, 4}, DType::Float32, device);
    float* data = input_tensor.data<float>();

    for (int i = 0; i < 16; ++i) {
        data[i] = static_cast<float>(i + 1);
    }

    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = pool.forward(input);

    auto output_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    const float* out_data = output_cpu.data<float>();

    EXPECT_NEAR(out_data[0], 3.5f, tol);   // avg of [1,2,5,6]
    EXPECT_NEAR(out_data[1], 5.5f, tol);   // avg of [3,4,7,8]
    EXPECT_NEAR(out_data[2], 11.5f, tol);  // avg of [9,10,13,14]
    EXPECT_NEAR(out_data[3], 13.5f, tol);  // avg of [11,12,15,16]
}

TEST_P(PoolingMultiDTypeTest, AvgPool2dGradientFlow) {
    auto param = GetParam();
    auto pool = nn::AvgPool2d(2);

    auto input_tensor = randn({2, 3, 16, 16}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(out_shape_vec, dtype, device);

    EXPECT_NO_THROW({
        output.backward(grad_output);
    });

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype);
}

// ============================================================================
// AdaptiveAvgPool2d Tests
// ============================================================================

TEST_P(PoolingMultiDTypeTest, AdaptiveAvgPool2dForwardBasic) {
    auto param = GetParam();
    auto pool = nn::AdaptiveAvgPool2d(7, 7);

    auto input_tensor = randn({2, 3, 32, 32}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 7);
    EXPECT_EQ(output.shape()[3], 7);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(PoolingMultiDTypeTest, AdaptiveAvgPool2dGlobalPooling) {
    auto param = GetParam();
    auto pool = nn::AdaptiveAvgPool2d(1, 1);

    auto input_tensor = randn({2, 64, 14, 14}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = pool.forward(input);

    EXPECT_EQ(output.shape()[2], 1);
    EXPECT_EQ(output.shape()[3], 1);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(PoolingMultiDTypeTest, AdaptiveAvgPool2dGradientFlow) {
    auto param = GetParam();
    auto pool = nn::AdaptiveAvgPool2d(7);

    auto input_tensor = randn({2, 3, 32, 32}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = pool.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(out_shape_vec, dtype, device);

    EXPECT_NO_THROW({
        output.backward(grad_output);
    });

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype);
}

// ============================================================================
// Mixed Precision Tests
// ============================================================================

TEST_P(PoolingMultiDTypeTest, SequentialPoolingPreservesType) {
    auto param = GetParam();
    auto pool1 = nn::MaxPool2d(2);
    auto pool2 = nn::MaxPool2d(2);

    auto input_tensor = randn({2, 3, 64, 64}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto x = pool1.forward(input);
    auto output = pool2.forward(x);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

// ============================================================================
// Test Instantiation
// ============================================================================

std::vector<DTypeParam> GeneratePoolingDTypeParams() {
    return {
        {DType::Float32, "float32", 1e-5f},
        {DType::Float64, "float64", 1e-10f},
        {DType::Float16, "float16", 1e-2f}  // Lower precision for Float16
    };
}

INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    PoolingMultiDTypeTest,
    ::testing::ValuesIn(GeneratePoolingDTypeParams()),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 11
 * DTypes Tested: Float32, Float64, Float16
 * Total Scenarios: 11 tests × 3 dtypes = 33 test scenarios
 *
 * Coverage:
 * - MaxPool2d: forward, value selection, gradient flow
 * - AvgPool2d: forward, average computation, gradient flow
 * - AdaptiveAvgPool2d: forward, global pooling, gradient flow
 * - Mixed precision: sequential pooling type preservation
 *
 * Tolerances:
 * - Float32: 1e-5 (standard precision)
 * - Float64: 1e-10 (high precision)
 * - Float16: 1e-2 (reduced precision for mixed precision training)
 */
