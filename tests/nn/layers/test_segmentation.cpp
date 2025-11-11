#include <gtest/gtest.h>
#include "../../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

/**
 * @file test_segmentation.cpp
 * @brief Comprehensive backend-agnostic tests for segmentation layers
 *
 * Tests cover:
 * - AtrousSeparableConv2d (depthwise separable convolution with dilation)
 * - ASPP (Atrous Spatial Pyramid Pooling)
 * - upsample_bilinear (bilinear interpolation upsampling)
 * - All backends (CPU, CUDA, Vulkan, OneAPI, ROCm)
 */

// ============================================================================
// Test Environment Setup
// ============================================================================

class SegmentationTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const seg_env =
    ::testing::AddGlobalTestEnvironment(new SegmentationTestEnvironment);

// ============================================================================
// Helper Functions
// ============================================================================

bool tensors_close(const Tensor& a, const Tensor& b, float rtol = 1e-4f, float atol = 1e-5f) {
    if (a.numel() != b.numel()) return false;

    auto a_cpu = a.to(Device::cpu());
    auto b_cpu = b.to(Device::cpu());

    const float* a_data = a_cpu.data<float>();
    const float* b_data = b_cpu.data<float>();
    size_t numel = a_cpu.numel();

    for (size_t i = 0; i < numel; ++i) {
        float diff = std::abs(a_data[i] - b_data[i]);
        float threshold = atol + rtol * std::abs(b_data[i]);
        if (diff > threshold) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// AtrousSeparableConv2d Tests
// ============================================================================

class AtrousSeparableConv2dTest : public tenzor::testing::BackendTest {};

TEST_P(AtrousSeparableConv2dTest, BasicForwardShape) {
    // AtrousSeparableConv2d: in_channels=16, out_channels=32, kernel=3, dilation=2
    auto layer = AtrousSeparableConv2d(16, 32, 3, 2, true);
    layer.to(device);

    auto input = Variable(randn({2, 16, 28, 28}, DType::Float32, device), true);
    auto output = layer.forward(input);

    // Should preserve spatial dimensions with proper padding
    EXPECT_EQ(output.shape()[0], 2);   // batch
    EXPECT_EQ(output.shape()[1], 32);  // out_channels
    EXPECT_EQ(output.shape()[2], 28);  // height (preserved)
    EXPECT_EQ(output.shape()[3], 28);  // width (preserved)
    EXPECT_EQ(output.tensor().device().type, device.type);
}

TEST_P(AtrousSeparableConv2dTest, DilationRate1) {
    // With dilation=1, should behave like standard separable conv
    auto layer = AtrousSeparableConv2d(8, 16, 3, 1, false);
    layer.to(device);

    auto input = Variable(randn({1, 8, 32, 32}, DType::Float32, device), true);
    auto output = layer.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 32);
    EXPECT_EQ(output.shape()[3], 32);
}

TEST_P(AtrousSeparableConv2dTest, DilationRate6) {
    // Large dilation rate (common in DeepLabV3)
    auto layer = AtrousSeparableConv2d(32, 64, 3, 6, true);
    layer.to(device);

    auto input = Variable(randn({2, 32, 64, 64}, DType::Float32, device), true);
    auto output = layer.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 64);
    EXPECT_EQ(output.shape()[2], 64);
    EXPECT_EQ(output.shape()[3], 64);
}

TEST_P(AtrousSeparableConv2dTest, GradientFlow) {
    auto layer = AtrousSeparableConv2d(4, 8, 3, 2, true);
    layer.to(device);

    auto input = Variable(randn({1, 4, 16, 16}, DType::Float32, device), true);
    auto output = layer.forward(input);

    // Backward pass
    auto grad_output = ones(output.shape(), DType::Float32, device);
    output.backward(grad_output);

    // Check gradient exists
    ASSERT_TRUE(input.grad().has_value());
    auto grad = input.grad().value();
    EXPECT_EQ(grad.shape()[0], 1);
    EXPECT_EQ(grad.shape()[1], 4);
    EXPECT_EQ(grad.shape()[2], 16);
    EXPECT_EQ(grad.shape()[3], 16);
}

// ============================================================================
// ASPP Tests
// ============================================================================

class ASPPTest : public tenzor::testing::BackendTest {};

TEST_P(ASPPTest, BasicForwardShape) {
    // Standard ASPP configuration for DeepLabV3
    std::vector<int64_t> atrous_rates = {6, 12, 18};
    auto aspp = ASPP(256, 256, atrous_rates, false, 0.1f);
    aspp.to(device);

    auto input = Variable(randn({2, 256, 32, 32}, DType::Float32, device), true);
    auto output = aspp.forward(input);

    // ASPP preserves spatial dimensions and outputs specified channels
    EXPECT_EQ(output.shape()[0], 2);    // batch
    EXPECT_EQ(output.shape()[1], 256);  // out_channels
    EXPECT_EQ(output.shape()[2], 32);   // height preserved
    EXPECT_EQ(output.shape()[3], 32);   // width preserved
    EXPECT_EQ(output.tensor().device().type, device.type);
}

TEST_P(ASPPTest, WithSeparableConvolution) {
    // ASPP with separable convolutions (more efficient)
    std::vector<int64_t> atrous_rates = {6, 12, 18};
    auto aspp = ASPP(128, 256, atrous_rates, true, 0.1f);
    aspp.to(device);

    auto input = Variable(randn({1, 128, 64, 64}, DType::Float32, device), true);
    auto output = aspp.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 256);
    EXPECT_EQ(output.shape()[2], 64);
    EXPECT_EQ(output.shape()[3], 64);
}

TEST_P(ASPPTest, DifferentAtrousRates) {
    // Test with different atrous rates
    std::vector<int64_t> atrous_rates = {3, 6, 9};
    auto aspp = ASPP(64, 128, atrous_rates, false, 0.0f);
    aspp.to(device);

    auto input = Variable(randn({2, 64, 16, 16}, DType::Float32, device), true);
    auto output = aspp.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 128);
    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
}

TEST_P(ASPPTest, SmallSpatialDimensions) {
    // Test with small feature maps (common at deep layers)
    std::vector<int64_t> atrous_rates = {6, 12, 18};
    auto aspp = ASPP(512, 256, atrous_rates, false, 0.1f);
    aspp.to(device);

    auto input = Variable(randn({1, 512, 8, 8}, DType::Float32, device), true);
    auto output = aspp.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 256);
    EXPECT_EQ(output.shape()[2], 8);
    EXPECT_EQ(output.shape()[3], 8);
}

TEST_P(ASPPTest, GradientFlow) {
    std::vector<int64_t> atrous_rates = {6, 12, 18};
    auto aspp = ASPP(32, 64, atrous_rates, false, 0.0f);
    aspp.to(device);

    auto input = Variable(randn({1, 32, 16, 16}, DType::Float32, device), true);
    auto output = aspp.forward(input);

    auto grad_output = ones(output.shape(), DType::Float32, device);
    output.backward(grad_output);

    ASSERT_TRUE(input.grad().has_value());
    auto grad = input.grad().value();
    EXPECT_EQ(grad.shape()[0], 1);
    EXPECT_EQ(grad.shape()[1], 32);
    EXPECT_EQ(grad.shape()[2], 16);
    EXPECT_EQ(grad.shape()[3], 16);
}

TEST_P(ASPPTest, MultiScaleFeatureFusion) {
    // Verify all 5 branches are working (1x1, 3 atrous, global pool)
    std::vector<int64_t> atrous_rates = {6, 12, 18};
    auto aspp = ASPP(64, 128, atrous_rates, false, 0.0f);
    aspp.to(device);

    auto input = Variable(randn({2, 64, 32, 32}, DType::Float32, device), true);
    auto output = aspp.forward(input);

    // Output should be non-zero (all branches contributing)
    auto output_cpu = output.tensor().to(Device::cpu());
    const float* out_data = output_cpu.data<float>();

    bool has_nonzero = false;
    for (int64_t i = 0; i < 100; ++i) {  // Check first 100 elements
        if (std::abs(out_data[i]) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// ============================================================================
// Bilinear Upsampling Tests
// ============================================================================

class BilinearUpsamplingTest : public tenzor::testing::BackendTest {};

TEST_P(BilinearUpsamplingTest, Upsample2x) {
    // Upsample from 8x8 to 16x16
    auto input = Variable(randn({1, 3, 8, 8}, DType::Float32, device), true);
    auto output = upsample_bilinear(input, 16, 16);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 16);
    EXPECT_EQ(output.shape()[3], 16);
    EXPECT_EQ(output.tensor().device().type, device.type);
}

TEST_P(BilinearUpsamplingTest, Upsample4x) {
    // Upsample from 7x7 to 28x28
    auto input = Variable(randn({2, 16, 7, 7}, DType::Float32, device), true);
    auto output = upsample_bilinear(input, 28, 28);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 28);
    EXPECT_EQ(output.shape()[3], 28);
}

TEST_P(BilinearUpsamplingTest, NonUniformScale) {
    // Different scales for H and W
    auto input = Variable(randn({1, 8, 10, 15}, DType::Float32, device), true);
    auto output = upsample_bilinear(input, 30, 45);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.shape()[2], 30);  // 3x upscale
    EXPECT_EQ(output.shape()[3], 45);  // 3x upscale
}

TEST_P(BilinearUpsamplingTest, SinglePixelUpsampling) {
    // Edge case: 1x1 to larger
    auto input = Variable(ones({1, 1, 1, 1}, DType::Float32, device) * 5.0f, true);
    auto output = upsample_bilinear(input, 4, 4);

    EXPECT_EQ(output.shape()[2], 4);
    EXPECT_EQ(output.shape()[3], 4);

    // All values should be approximately 5.0
    auto output_cpu = output.tensor().to(Device::cpu());
    const float* out_data = output_cpu.data<float>();

    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(out_data[i], 5.0f, 0.1f);
    }
}

TEST_P(BilinearUpsamplingTest, InterpolationSmoothnessIdentity) {
    // Test that upsampling preserves values at certain grid points
    auto input_cpu = arange(0.0f, 4.0f, 1.0f, DType::Float32, Device::cpu()).reshape({1, 1, 2, 2});
    // Input: [[0, 1],
    //         [2, 3]]

    auto input_data = (device.type == Device::Type::CPU) ? input_cpu : input_cpu.to(device);
    auto input = Variable(input_data, false);

    auto output = upsample_bilinear(input, 4, 4);
    auto output_cpu = output.tensor().to(Device::cpu());
    const float* out_data = output_cpu.data<float>();

    // Check corner values are approximately preserved
    EXPECT_NEAR(out_data[0], 0.0f, 0.2f);          // Top-left
    EXPECT_NEAR(out_data[3], 1.0f, 0.2f);          // Top-right
    EXPECT_NEAR(out_data[12], 2.0f, 0.2f);         // Bottom-left
    EXPECT_NEAR(out_data[15], 3.0f, 0.2f);         // Bottom-right
}

TEST_P(BilinearUpsamplingTest, GradientFlow) {
    auto input = Variable(randn({1, 4, 8, 8}, DType::Float32, device), true);
    auto output = upsample_bilinear(input, 16, 16);

    auto grad_output = ones(output.shape(), DType::Float32, device);
    output.backward(grad_output);

    // Gradient should exist and have correct shape
    ASSERT_TRUE(input.grad().has_value());
    auto grad = input.grad().value();
    EXPECT_EQ(grad.shape()[0], 1);
    EXPECT_EQ(grad.shape()[1], 4);
    EXPECT_EQ(grad.shape()[2], 8);
    EXPECT_EQ(grad.shape()[3], 8);
}

TEST_P(BilinearUpsamplingTest, LargeBatchSize) {
    // Test with large batch
    auto input = Variable(randn({16, 64, 14, 14}, DType::Float32, device), true);
    auto output = upsample_bilinear(input, 28, 28);

    EXPECT_EQ(output.shape()[0], 16);
    EXPECT_EQ(output.shape()[1], 64);
    EXPECT_EQ(output.shape()[2], 28);
    EXPECT_EQ(output.shape()[3], 28);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(ASPPErrorTest, InvalidAtrousRates) {
    // ASPP requires exactly 3 atrous rates
    std::vector<int64_t> invalid_rates = {6, 12};  // Only 2 rates

    EXPECT_THROW(
        ASPP(256, 256, invalid_rates, false, 0.1f),
        std::invalid_argument
    );
}

TEST(ASPPErrorTest, TooManyAtrousRates) {
    std::vector<int64_t> invalid_rates = {6, 12, 18, 24};  // 4 rates

    EXPECT_THROW(
        ASPP(256, 256, invalid_rates, false, 0.1f),
        std::invalid_argument
    );
}

TEST(BilinearErrorTest, Invalid3DInput) {
    // Bilinear upsample expects 4D input (N, C, H, W)
    auto input = Variable(randn({2, 16, 8}), true);  // Only 3D

    EXPECT_THROW(
        upsample_bilinear(input, 16, 16),
        std::runtime_error
    );
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_P(BilinearUpsamplingTest, ASPPWithUpsampling) {
    // Simulate DeepLabV3+ decoder pattern: ASPP + upsample
    std::vector<int64_t> atrous_rates = {6, 12, 18};
    auto aspp = ASPP(128, 256, atrous_rates, false, 0.1f);
    aspp.to(device);

    auto input = Variable(randn({2, 128, 16, 16}, DType::Float32, device), true);

    // ASPP processing
    auto aspp_output = aspp.forward(input);
    EXPECT_EQ(aspp_output.shape()[2], 16);
    EXPECT_EQ(aspp_output.shape()[3], 16);

    // Upsample to original resolution
    auto upsampled = upsample_bilinear(aspp_output, 64, 64);
    EXPECT_EQ(upsampled.shape()[0], 2);
    EXPECT_EQ(upsampled.shape()[1], 256);
    EXPECT_EQ(upsampled.shape()[2], 64);
    EXPECT_EQ(upsampled.shape()[3], 64);

    // Gradient flow through both
    auto grad = ones(upsampled.shape(), DType::Float32, device);
    upsampled.backward(grad);

    ASSERT_TRUE(input.grad().has_value());
}

// ============================================================================
// Instantiate Tests for All Backends
// ============================================================================

INSTANTIATE_BACKEND_TESTS(AtrousSeparableConv2dTest);
INSTANTIATE_BACKEND_TESTS(ASPPTest);
INSTANTIATE_BACKEND_TESTS(BilinearUpsamplingTest);
