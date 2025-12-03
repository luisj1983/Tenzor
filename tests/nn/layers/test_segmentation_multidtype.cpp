#include <gtest/gtest.h>
#include "../../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/segmentation.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

/**
 * @file test_segmentation_multidtype.cpp
 * @brief Multi-dtype tests for segmentation layers
 *
 * Tests segmentation layers (AtrousSeparableConv2d, ASPP, upsample_bilinear)
 * with Float32 and Float64 dtypes. Segmentation operations work with floating-point types.
 */

// ============================================================================
// Test Environment Setup
// ============================================================================

class SegmentationMultiDTypeTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const seg_multidtype_env =
    ::testing::AddGlobalTestEnvironment(new SegmentationMultiDTypeTestEnvironment);

// ============================================================================
// Multi-DType Parameterization
// ============================================================================

struct DTypeParam {
    DType dtype;
    std::string dtype_name;
    float rtol;
    float atol;

    std::string ToString() const {
        return dtype_name;
    }
};

// Required for gtest_discover_tests to show human-readable test names
void PrintTo(const DTypeParam& param, std::ostream* os) {
    *os << param.ToString();
}

// ============================================================================
// Helper Functions
// ============================================================================

template<typename T>
bool tensors_close_typed(const Tensor& a, const Tensor& b, float rtol, float atol) {
    if (a.numel() != b.numel()) return false;

    auto a_cpu = a.to(Device::cpu());
    auto b_cpu = b.to(Device::cpu());

    const T* a_data = a_cpu.data<T>();
    const T* b_data = b_cpu.data<T>();
    size_t numel = a_cpu.numel();

    for (size_t i = 0; i < numel; ++i) {
        float a_val = static_cast<float>(a_data[i]);
        float b_val = static_cast<float>(b_data[i]);
        float diff = std::abs(a_val - b_val);
        float threshold = atol + rtol * std::abs(b_val);
        if (diff > threshold) {
            return false;
        }
    }
    return true;
}

bool tensors_close(const Tensor& a, const Tensor& b, float rtol = 1e-4f, float atol = 1e-5f) {
    if (a.dtype() != b.dtype()) return false;

    switch (a.dtype()) {
        case DType::Float32:
            return tensors_close_typed<float>(a, b, rtol, atol);
        case DType::Float64:
            return tensors_close_typed<double>(a, b, rtol, atol);
        default:
            return false;
    }
}

// ============================================================================
// AtrousSeparableConv2d Multi-DType Tests
// ============================================================================

class AtrousSeparableConv2dMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    Device device;
    DType dtype;
    float rtol;
    float atol;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype = param.dtype;
        rtol = param.rtol;
        atol = param.atol;
        device = Device::cpu();
    }
};

TEST_P(AtrousSeparableConv2dMultiDTypeTest, BasicForwardShape) {
    // AtrousSeparableConv2d: in_channels=16, out_channels=32, kernel=3, dilation=2
    auto layer = AtrousSeparableConv2d(16, 32, 3, 2, true);
    layer.to(device);

    auto input = Variable(randn({2, 16, 28, 28}, dtype, device), true);
    auto output = layer.forward(input);

    // Should preserve spatial dimensions with proper padding
    EXPECT_EQ(output.shape()[0], 2);   // batch
    EXPECT_EQ(output.shape()[1], 32);  // out_channels
    EXPECT_EQ(output.shape()[2], 28);  // height (preserved)
    EXPECT_EQ(output.shape()[3], 28);  // width (preserved)
    EXPECT_EQ(output.tensor().device().type, device.type);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(AtrousSeparableConv2dMultiDTypeTest, DilationRate1) {
    // With dilation=1, should behave like standard separable conv
    auto layer = AtrousSeparableConv2d(8, 16, 3, 1, false);
    layer.to(device);

    auto input = Variable(randn({1, 8, 32, 32}, dtype, device), true);
    auto output = layer.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 32);
    EXPECT_EQ(output.shape()[3], 32);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(AtrousSeparableConv2dMultiDTypeTest, DilationRate6) {
    // Large dilation rate (common in DeepLabV3)
    auto layer = AtrousSeparableConv2d(32, 64, 3, 6, true);
    layer.to(device);

    auto input = Variable(randn({2, 32, 64, 64}, dtype, device), true);
    auto output = layer.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 64);
    EXPECT_EQ(output.shape()[2], 64);
    EXPECT_EQ(output.shape()[3], 64);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(AtrousSeparableConv2dMultiDTypeTest, GradientFlow) {
    auto layer = AtrousSeparableConv2d(4, 8, 3, 2, true);
    layer.to(device);

    auto input = Variable(randn({1, 4, 16, 16}, dtype, device), true);
    auto output = layer.forward(input);

    // Backward pass
    auto grad_output = ones(std::vector<int64_t>(output.shape().begin(), output.shape().end()), dtype, device);
    output.backward(grad_output);

    // Check gradient exists
    ASSERT_TRUE(input.grad().has_value());
    auto grad = input.grad().value();
    EXPECT_EQ(grad.shape()[0], 1);
    EXPECT_EQ(grad.shape()[1], 4);
    EXPECT_EQ(grad.shape()[2], 16);
    EXPECT_EQ(grad.shape()[3], 16);
    EXPECT_EQ(grad.dtype(), dtype);
}

TEST_P(AtrousSeparableConv2dMultiDTypeTest, SmallInput) {
    // Test with small spatial dimensions
    auto layer = AtrousSeparableConv2d(4, 8, 3, 1, true);
    layer.to(device);

    auto input = Variable(randn({1, 4, 8, 8}, dtype, device), true);
    auto output = layer.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.shape()[2], 8);
    EXPECT_EQ(output.shape()[3], 8);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

// ============================================================================
// ASPP Multi-DType Tests
// ============================================================================

class ASPPMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    Device device;
    DType dtype;
    float rtol;
    float atol;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype = param.dtype;
        rtol = param.rtol;
        atol = param.atol;
        device = Device::cpu();
    }
};

TEST_P(ASPPMultiDTypeTest, BasicForwardShape) {
    // Standard ASPP configuration for DeepLabV3
    std::vector<int64_t> atrous_rates = {6, 12, 18};
    auto aspp = ASPP(256, 256, atrous_rates, false, 0.1f);
    aspp.to(device);

    auto input = Variable(randn({2, 256, 32, 32}, dtype, device), true);
    auto output = aspp.forward(input);

    // ASPP preserves spatial dimensions and outputs specified channels
    EXPECT_EQ(output.shape()[0], 2);    // batch
    EXPECT_EQ(output.shape()[1], 256);  // out_channels
    EXPECT_EQ(output.shape()[2], 32);   // height preserved
    EXPECT_EQ(output.shape()[3], 32);   // width preserved
    EXPECT_EQ(output.tensor().device().type, device.type);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(ASPPMultiDTypeTest, WithSeparableConvolution) {
    // ASPP with separable convolutions (more efficient)
    std::vector<int64_t> atrous_rates = {6, 12, 18};
    auto aspp = ASPP(128, 256, atrous_rates, true, 0.1f);
    aspp.to(device);

    auto input = Variable(randn({1, 128, 64, 64}, dtype, device), true);
    auto output = aspp.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 256);
    EXPECT_EQ(output.shape()[2], 64);
    EXPECT_EQ(output.shape()[3], 64);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(ASPPMultiDTypeTest, DifferentAtrousRates) {
    // Test with different atrous rates
    std::vector<int64_t> atrous_rates = {3, 6, 9};
    auto aspp = ASPP(64, 128, atrous_rates, false, 0.0f);
    aspp.to(device);

    auto input = Variable(randn({2, 64, 16, 16}, dtype, device), true);
    auto output = aspp.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 128);
    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(ASPPMultiDTypeTest, SmallSpatialDimensions) {
    // Test with small feature maps (common at deep layers)
    std::vector<int64_t> atrous_rates = {6, 12, 18};
    auto aspp = ASPP(512, 256, atrous_rates, false, 0.1f);
    aspp.to(device);

    auto input = Variable(randn({1, 512, 8, 8}, dtype, device), true);
    auto output = aspp.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 256);
    EXPECT_EQ(output.shape()[2], 8);
    EXPECT_EQ(output.shape()[3], 8);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(ASPPMultiDTypeTest, GradientFlow) {
    std::vector<int64_t> atrous_rates = {6, 12, 18};
    auto aspp = ASPP(32, 64, atrous_rates, false, 0.0f);
    aspp.to(device);

    auto input = Variable(randn({1, 32, 16, 16}, dtype, device), true);
    auto output = aspp.forward(input);

    auto grad_output = ones(std::vector<int64_t>(output.shape().begin(), output.shape().end()), dtype, device);
    output.backward(grad_output);

    ASSERT_TRUE(input.grad().has_value());
    auto grad = input.grad().value();
    EXPECT_EQ(grad.shape()[0], 1);
    EXPECT_EQ(grad.shape()[1], 32);
    EXPECT_EQ(grad.shape()[2], 16);
    EXPECT_EQ(grad.shape()[3], 16);
    EXPECT_EQ(grad.dtype(), dtype);
}

TEST_P(ASPPMultiDTypeTest, MultiScaleFeatureFusion) {
    // Verify all 5 branches are working (1x1, 3 atrous, global pool)
    std::vector<int64_t> atrous_rates = {6, 12, 18};
    auto aspp = ASPP(64, 128, atrous_rates, false, 0.0f);
    aspp.to(device);

    auto input = Variable(randn({2, 64, 32, 32}, dtype, device), true);
    auto output = aspp.forward(input);

    // Output should be non-zero (all branches contributing)
    auto output_cpu = output.tensor().to(Device::cpu());

    bool has_nonzero = false;
    if (dtype == DType::Float32) {
        const float* out_data = output_cpu.data<float>();
        for (int64_t i = 0; i < 100; ++i) {
            if (std::abs(out_data[i]) > atol) {
                has_nonzero = true;
                break;
            }
        }
    } else if (dtype == DType::Float64) {
        const double* out_data = output_cpu.data<double>();
        for (int64_t i = 0; i < 100; ++i) {
            if (std::abs(static_cast<float>(out_data[i])) > atol) {
                has_nonzero = true;
                break;
            }
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_P(ASPPMultiDTypeTest, MultipleBatches) {
    std::vector<int64_t> atrous_rates = {6, 12, 18};
    auto aspp = ASPP(32, 64, atrous_rates, false, 0.1f);
    aspp.to(device);

    auto input = Variable(randn({4, 32, 16, 16}, dtype, device), true);
    auto output = aspp.forward(input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 64);
    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

// ============================================================================
// Bilinear Upsampling Multi-DType Tests
// ============================================================================

class BilinearUpsamplingMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    Device device;
    DType dtype;
    float rtol;
    float atol;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype = param.dtype;
        rtol = param.rtol;
        atol = param.atol;
        device = Device::cpu();
    }
};

TEST_P(BilinearUpsamplingMultiDTypeTest, Upsample2x) {
    // Upsample from 8x8 to 16x16
    auto input = Variable(randn({1, 3, 8, 8}, dtype, device), true);
    auto output = upsample_bilinear(input, 16, 16);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
    EXPECT_EQ(output.tensor().device().type, device.type);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(BilinearUpsamplingMultiDTypeTest, Upsample4x) {
    // Upsample from 7x7 to 28x28
    auto input = Variable(randn({2, 16, 7, 7}, dtype, device), true);
    auto output = upsample_bilinear(input, 28, 28);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 28);
    EXPECT_EQ(output.shape()[3], 28);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(BilinearUpsamplingMultiDTypeTest, NonUniformScale) {
    // Different scales for H and W
    auto input = Variable(randn({1, 8, 10, 15}, dtype, device), true);
    auto output = upsample_bilinear(input, 30, 45);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.shape()[2], 30);  // 3x upscale
    EXPECT_EQ(output.shape()[3], 45);  // 3x upscale
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(BilinearUpsamplingMultiDTypeTest, SinglePixelUpsampling) {
    // Edge case: 1x1 to larger
    auto input = Variable(ones({1, 1, 1, 1}, dtype, device) * 5.0f, true);
    auto output = upsample_bilinear(input, 4, 4);

    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 4);

    // All values should be approximately 5.0
    auto output_cpu = output.tensor().to(Device::cpu());

    if (dtype == DType::Float32) {
        const float* out_data = output_cpu.data<float>();
        for (int i = 0; i < 16; ++i) {
            EXPECT_NEAR(out_data[i], 5.0f, atol * 10);
        }
    } else if (dtype == DType::Float64) {
        const double* out_data = output_cpu.data<double>();
        for (int i = 0; i < 16; ++i) {
            EXPECT_NEAR(static_cast<float>(out_data[i]), 5.0f, atol * 10);
        }
    }
}

TEST_P(BilinearUpsamplingMultiDTypeTest, InterpolationSmoothnessIdentity) {
    // Test that upsampling preserves values at certain grid points
    auto input_cpu = arange(0.0f, 4.0f, 1.0f, DType::Float32, Device::cpu()).reshape({1, 1, 2, 2});
    // Input: [[0, 1],
    //         [2, 3]]

    // Convert to target dtype
    if (dtype != DType::Float32) {
        input_cpu = input_cpu.to(dtype);
    }

    auto input_data = (device.type == Device::Type::CPU) ? input_cpu : input_cpu.to(device);
    auto input = Variable(input_data, false);

    auto output = upsample_bilinear(input, 4, 4);
    auto output_cpu = output.tensor().to(Device::cpu());

    // Check corner values are approximately preserved
    float corner_tol = atol * 20;  // Looser tolerance for interpolation

    if (dtype == DType::Float32) {
        const float* out_data = output_cpu.data<float>();
        EXPECT_NEAR(out_data[0], 0.0f, corner_tol);   // Top-left
        EXPECT_NEAR(out_data[3], 1.0f, corner_tol);   // Top-right
        EXPECT_NEAR(out_data[12], 2.0f, corner_tol);  // Bottom-left
        EXPECT_NEAR(out_data[15], 3.0f, corner_tol);  // Bottom-right
    } else if (dtype == DType::Float64) {
        const double* out_data = output_cpu.data<double>();
        EXPECT_NEAR(static_cast<float>(out_data[0]), 0.0f, corner_tol);
        EXPECT_NEAR(static_cast<float>(out_data[3]), 1.0f, corner_tol);
        EXPECT_NEAR(static_cast<float>(out_data[12]), 2.0f, corner_tol);
        EXPECT_NEAR(static_cast<float>(out_data[15]), 3.0f, corner_tol);
    }
}

TEST_P(BilinearUpsamplingMultiDTypeTest, GradientFlow) {
    auto input = Variable(randn({1, 4, 8, 8}, dtype, device), true);
    auto output = upsample_bilinear(input, 16, 16);

    auto grad_output = ones(std::vector<int64_t>(output.shape().begin(), output.shape().end()), dtype, device);
    output.backward(grad_output);

    // Gradient should exist and have correct shape
    ASSERT_TRUE(input.grad().has_value());
    auto grad = input.grad().value();
    EXPECT_EQ(grad.shape()[0], 1);
    EXPECT_EQ(grad.shape()[1], 4);
    EXPECT_EQ(grad.shape()[2], 8);
    EXPECT_EQ(grad.shape()[3], 8);
    EXPECT_EQ(grad.dtype(), dtype);
}

TEST_P(BilinearUpsamplingMultiDTypeTest, LargeBatchSize) {
    // Test with large batch
    auto input = Variable(randn({16, 64, 14, 14}, dtype, device), true);
    auto output = upsample_bilinear(input, 28, 28);

    EXPECT_EQ(output.shape()[0], 16);
    EXPECT_EQ(output.shape()[1], 64);
    EXPECT_EQ(output.shape()[2], 28);
    EXPECT_EQ(output.shape()[3], 28);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(BilinearUpsamplingMultiDTypeTest, DownsampleOperation) {
    // Edge case: downsample instead of upsample
    auto input = Variable(randn({1, 8, 32, 32}, dtype, device), true);
    auto output = upsample_bilinear(input, 16, 16);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_P(BilinearUpsamplingMultiDTypeTest, ASPPWithUpsampling) {
    // Simulate DeepLabV3+ decoder pattern: ASPP + upsample
    std::vector<int64_t> atrous_rates = {6, 12, 18};
    auto aspp = ASPP(128, 256, atrous_rates, false, 0.1f);
    aspp.to(device);

    auto input = Variable(randn({2, 128, 16, 16}, dtype, device), true);

    // ASPP processing
    auto aspp_output = aspp.forward(input);
    EXPECT_EQ(aspp_output.shape()[2], 16);
    EXPECT_EQ(aspp_output.shape()[3], 16);
    EXPECT_EQ(aspp_output.tensor().dtype(), dtype);

    // Upsample to original resolution
    auto upsampled = upsample_bilinear(aspp_output, 64, 64);
    EXPECT_EQ(upsampled.shape()[0], 2);
    EXPECT_EQ(upsampled.shape()[1], 256);
    EXPECT_EQ(upsampled.shape()[2], 64);
    EXPECT_EQ(upsampled.shape()[3], 64);
    EXPECT_EQ(upsampled.tensor().dtype(), dtype);

    // Gradient flow through both
    auto grad = ones(std::vector<int64_t>(upsampled.shape().begin(), upsampled.shape().end()), dtype, device);
    upsampled.backward(grad);

    ASSERT_TRUE(input.grad().has_value());
}

TEST_P(BilinearUpsamplingMultiDTypeTest, SequentialUpsampling) {
    // Test multiple upsampling stages
    auto input = Variable(randn({1, 16, 8, 8}, dtype, device), true);

    auto output1 = upsample_bilinear(input, 16, 16);
    EXPECT_EQ(output1.shape()[2], 16);
    EXPECT_EQ(output1.tensor().dtype(), dtype);

    auto output2 = upsample_bilinear(output1, 32, 32);
    EXPECT_EQ(output2.shape()[2], 32);
    EXPECT_EQ(output2.tensor().dtype(), dtype);

    // Gradient flow through multiple stages
    auto grad = ones(std::vector<int64_t>(output2.shape().begin(), output2.shape().end()), dtype, device);
    output2.backward(grad);

    ASSERT_TRUE(input.grad().has_value());
}

// ============================================================================
// Test Instantiation
// ============================================================================

std::vector<DTypeParam> GenerateSegmentationMultiDTypeParams() {
    return {
        {DType::Float32, "float32", 1e-4f, 1e-5f},
        {DType::Float64, "float64", 1e-5f, 1e-6f}
    };
}

INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    AtrousSeparableConv2dMultiDTypeTest,
    ::testing::ValuesIn(GenerateSegmentationMultiDTypeParams()),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    ASPPMultiDTypeTest,
    ::testing::ValuesIn(GenerateSegmentationMultiDTypeParams()),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    BilinearUpsamplingMultiDTypeTest,
    ::testing::ValuesIn(GenerateSegmentationMultiDTypeParams()),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

/*
 * COVERAGE SUMMARY:
 *
 * Test Classes: 3
 * - AtrousSeparableConv2dMultiDTypeTest: 5 tests
 * - ASPPMultiDTypeTest: 7 tests
 * - BilinearUpsamplingMultiDTypeTest: 10 tests
 *
 * Total Test Cases: 22
 * DTypes Tested: Float32, Float64
 * Total Scenarios: 22 tests × 2 dtypes = 44 test scenarios
 *
 * Coverage:
 * - AtrousSeparableConv2d: shape verification, various dilation rates, gradient flow, edge cases
 * - ASPP: standard config, separable convolution, different atrous rates, multi-scale fusion
 * - Bilinear Upsampling: 2x/4x upsampling, non-uniform scaling, interpolation accuracy, gradient flow
 * - Integration: ASPP + upsampling pipeline, sequential upsampling
 *
 * Tolerances:
 * - Float32: rtol=1e-4, atol=1e-5 (standard precision)
 * - Float64: rtol=1e-5, atol=1e-6 (higher precision)
 *
 * Note: Segmentation operations work with floating-point types for computer vision tasks.
 * Integer dtypes are not supported for these operations.
 */
