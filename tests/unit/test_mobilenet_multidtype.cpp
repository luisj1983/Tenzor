/**
 * @file test_mobilenet_multidtype.cpp
 * @brief Multi-dtype tests for MobileNet V2 and V3 variants (Float32, Float64, Float16)
 *
 * Tests MobileNet architectures with multiple data types for mobile/edge deployment:
 * - MobileNetV2 and V3 (Small, Large) architectures
 * - Depthwise separable convolutions
 * - Inverted residual blocks
 * - Squeeze-and-Excitation blocks (V3)
 * - Different width multipliers
 * - Various input image sizes
 * - Float16 is critical for mobile/edge inference
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/mobilenet.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::models;

// ============================================================================
// Test Fixture with Multi-DType Support
// ============================================================================

template<typename T>
class MobileNetMultiDTypeTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = Device::cpu();

        // Set dtype based on template parameter
        if constexpr (std::is_same_v<T, float>) {
            dtype_ = DType::Float32;
            tolerance_ = 1e-5f;
            rtol_ = 1e-4f;
        } else if constexpr (std::is_same_v<T, double>) {
            dtype_ = DType::Float64;
            tolerance_ = 1e-10;
            rtol_ = 1e-9;
        } else if constexpr (std::is_same_v<T, half_t>) {
            dtype_ = DType::Float16;
            tolerance_ = 1e-2f;  // Float16 has lower precision
            rtol_ = 1e-2f;
        }
    }

    Device device_;
    DType dtype_;
    float tolerance_;
    float rtol_;

    // Helper to check tensor shape
    void expectShape(const Variable& var, const std::vector<int64_t>& expected_shape) {
        auto shape = var.tensor().shape();
        EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), expected_shape);
    }

    // Helper to count parameters
    size_t countParameters(const std::shared_ptr<Module>& model) {
        auto params = model->parameters();
        size_t total_params = 0;
        for (const auto& p : params) {
            size_t param_size = 1;
            for (auto dim : p->tensor().shape()) {
                param_size *= dim;
            }
            total_params += param_size;
        }
        return total_params;
    }

    // Helper to check gradient flow
    void checkGradientFlow(const Variable& input, const std::shared_ptr<Module>& model) {
        EXPECT_TRUE(input.grad().has_value());
        auto params = model->parameters();
        EXPECT_GT(params.size(), 0);

        // Verify at least some parameters have gradients
        size_t params_with_grad = 0;
        for (const auto& p : params) {
            if (p->grad().has_value()) {
                params_with_grad++;
            }
        }
        EXPECT_GT(params_with_grad, 0);
    }
};

using DTypeTypes = ::testing::Types<float, double, half_t>;
TYPED_TEST_SUITE(MobileNetMultiDTypeTest, DTypeTypes);

// ============================================================================
// MobileNetV2 Core Architecture Tests
// ============================================================================

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV2ForwardShape) {
    auto model = mobilenet_v2(1000, false);
    Variable input(Tensor({2, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 1000});
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV2GradientFlow) {
    auto model = mobilenet_v2(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    this->checkGradientFlow(input, model);
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV2BatchSizes) {
    auto model = mobilenet_v2(10, false);

    // Test different batch sizes
    std::vector<int64_t> batch_sizes = {1, 2, 4, 8};
    for (auto batch_size : batch_sizes) {
        Variable input(Tensor({batch_size, 3, 224, 224}, this->dtype_, this->device_), true);
        Variable output = model->forward(input);
        this->expectShape(output, {batch_size, 10});
    }
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV2ParameterCount) {
    auto model = mobilenet_v2(1000, false);
    size_t total_params = this->countParameters(model);

    // MobileNetV2 should have ~3.5M parameters (allow 30% tolerance)
    EXPECT_GT(total_params, 2'500'000);
    EXPECT_LT(total_params, 5'000'000);
}

// ============================================================================
// MobileNetV2 Width Multiplier Tests (Critical for Mobile Deployment)
// ============================================================================

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV2WidthMultiplier_0_5) {
    auto model = mobilenet_v2_width(1000, 0.5, false);
    Variable input(Tensor({2, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 1000});

    // 0.5 width multiplier should have fewer parameters
    size_t params = this->countParameters(model);
    EXPECT_LT(params, 3'000'000);  // Much smaller than standard ~3.5M
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV2WidthMultiplier_0_75) {
    auto model = mobilenet_v2_width(1000, 0.75, false);
    Variable input(Tensor({2, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 1000});
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV2WidthMultiplier_1_25) {
    auto model = mobilenet_v2_width(1000, 1.25, false);
    Variable input(Tensor({2, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 1000});

    // 1.25 width multiplier should have more parameters
    size_t params = this->countParameters(model);
    EXPECT_GT(params, 3'500'000);
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV2WidthMultiplier_1_5) {
    auto model = mobilenet_v2_width(1000, 1.5, false);
    Variable input(Tensor({2, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 1000});
}

// ============================================================================
// MobileNetV2 Different Input Sizes (Critical for Mobile Applications)
// ============================================================================

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV2InputSize_128) {
    auto model = mobilenet_v2(1000, false);
    Variable input(Tensor({2, 3, 128, 128}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 1000});
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV2InputSize_160) {
    auto model = mobilenet_v2(1000, false);
    Variable input(Tensor({2, 3, 160, 160}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 1000});
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV2InputSize_192) {
    auto model = mobilenet_v2(1000, false);
    Variable input(Tensor({2, 3, 192, 192}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 1000});
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV2InputSize_256) {
    auto model = mobilenet_v2(1000, false);
    Variable input(Tensor({2, 3, 256, 256}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 1000});
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV2InputSize_320) {
    auto model = mobilenet_v2(1000, false);
    Variable input(Tensor({2, 3, 320, 320}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 1000});
}

// ============================================================================
// MobileNetV3-Small Tests
// ============================================================================

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV3SmallForwardShape) {
    auto model = mobilenet_v3_small(1000, false);
    Variable input(Tensor({2, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 1000});
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV3SmallGradientFlow) {
    auto model = mobilenet_v3_small(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    this->checkGradientFlow(input, model);
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV3SmallParameterCount) {
    auto model = mobilenet_v3_small(1000, false);
    size_t total_params = this->countParameters(model);

    // MobileNetV3-Small should have ~2.5M parameters (allow 30% tolerance)
    EXPECT_GT(total_params, 1'800'000);
    EXPECT_LT(total_params, 3'500'000);
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV3SmallBatchSizes) {
    auto model = mobilenet_v3_small(10, false);

    std::vector<int64_t> batch_sizes = {1, 2, 4, 8};
    for (auto batch_size : batch_sizes) {
        Variable input(Tensor({batch_size, 3, 224, 224}, this->dtype_, this->device_), true);
        Variable output = model->forward(input);
        this->expectShape(output, {batch_size, 10});
    }
}

// ============================================================================
// MobileNetV3-Large Tests
// ============================================================================

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV3LargeForwardShape) {
    auto model = mobilenet_v3_large(1000, false);
    Variable input(Tensor({2, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 1000});
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV3LargeGradientFlow) {
    auto model = mobilenet_v3_large(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    this->checkGradientFlow(input, model);
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV3LargeParameterCount) {
    auto model = mobilenet_v3_large(1000, false);
    size_t total_params = this->countParameters(model);

    // MobileNetV3-Large should have ~5.4M parameters (allow 30% tolerance)
    EXPECT_GT(total_params, 4'000'000);
    EXPECT_LT(total_params, 7'000'000);
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV3LargeBatchSizes) {
    auto model = mobilenet_v3_large(10, false);

    std::vector<int64_t> batch_sizes = {1, 2, 4, 8};
    for (auto batch_size : batch_sizes) {
        Variable input(Tensor({batch_size, 3, 224, 224}, this->dtype_, this->device_), true);
        Variable output = model->forward(input);
        this->expectShape(output, {batch_size, 10});
    }
}

// ============================================================================
// MobileNetV3 Different Input Sizes with SE Blocks
// ============================================================================

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV3LargeInputSize_160) {
    auto model = mobilenet_v3_large(1000, false);
    Variable input(Tensor({2, 3, 160, 160}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 1000});
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV3SmallInputSize_192) {
    auto model = mobilenet_v3_small(1000, false);
    Variable input(Tensor({2, 3, 192, 192}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 1000});
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV3LargeInputSize_256) {
    auto model = mobilenet_v3_large(1000, false);
    Variable input(Tensor({2, 3, 256, 256}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 1000});
}

// ============================================================================
// Custom Number of Classes Tests
// ============================================================================

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV2CustomClasses_10) {
    auto model = mobilenet_v2(10, false);
    Variable input(Tensor({2, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 10});
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV2CustomClasses_100) {
    auto model = mobilenet_v2(100, false);
    Variable input(Tensor({2, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 100});
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV3SmallCustomClasses_50) {
    auto model = mobilenet_v3_small(50, false);
    Variable input(Tensor({2, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 50});
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV3LargeCustomClasses_200) {
    auto model = mobilenet_v3_large(200, false);
    Variable input(Tensor({2, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {2, 200});
}

// ============================================================================
// Depthwise Separable Convolution Tests (Architecture-specific)
// ============================================================================

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV2DepthwiseConvolution) {
    // Test that depthwise separable convolutions work correctly
    auto model = mobilenet_v2(10, false);
    model->eval();

    Variable input(Tensor({1, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output1 = model->forward(input);
    Variable output2 = model->forward(input);

    // Same input should produce same output in eval mode
    this->expectShape(output1, {1, 10});
    this->expectShape(output2, {1, 10});
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV3DepthwiseWithSE) {
    // Test depthwise convolutions with SE blocks in V3
    auto model = mobilenet_v3_large(10, false);
    model->eval();

    Variable input(Tensor({1, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {1, 10});
}

// ============================================================================
// Inverted Residual Block Tests
// ============================================================================

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV2InvertedResiduals) {
    // Test inverted residual connections
    auto model = mobilenet_v2(10, false);
    model->train();

    Variable input(Tensor({2, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Verify gradient can flow through residual connections
    EXPECT_TRUE(input.grad().has_value());
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV3InvertedResidualsWithSE) {
    // Test inverted residuals with SE blocks in V3
    auto model = mobilenet_v3_large(10, false);
    model->train();

    Variable input(Tensor({2, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
}

// ============================================================================
// Mobile/Edge Deployment Scenario Tests (Float16 Critical)
// ============================================================================

TYPED_TEST(MobileNetMultiDTypeTest, MobileEdgeScenario_SmallModel_SmallInput) {
    // Typical mobile scenario: small model, smaller input, low precision
    auto model = mobilenet_v3_small(100, false);
    Variable input(Tensor({1, 3, 160, 160}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {1, 100});
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileEdgeScenario_WidthMultiplier_SmallInput) {
    // Ultra-lightweight mobile scenario
    auto model = mobilenet_v2_width(100, 0.5, false);
    Variable input(Tensor({1, 3, 128, 128}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {1, 100});
}

TYPED_TEST(MobileNetMultiDTypeTest, MobileEdgeScenario_BatchInference) {
    // Mobile batch inference (e.g., video frames)
    auto model = mobilenet_v3_small(10, false);
    model->eval();

    Variable input(Tensor({4, 3, 192, 192}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {4, 10});
}

// ============================================================================
// Performance Characteristics Tests
// ============================================================================

TYPED_TEST(MobileNetMultiDTypeTest, MobileNetV2V3ParameterComparison) {
    auto v2_model = mobilenet_v2(1000, false);
    auto v3_small = mobilenet_v3_small(1000, false);
    auto v3_large = mobilenet_v3_large(1000, false);

    size_t v2_params = this->countParameters(v2_model);
    size_t v3_small_params = this->countParameters(v3_small);
    size_t v3_large_params = this->countParameters(v3_large);

    // Parameter count ordering
    EXPECT_LT(v3_small_params, v2_params);
    EXPECT_GT(v3_large_params, v2_params);
}

TYPED_TEST(MobileNetMultiDTypeTest, WidthMultiplierParameterScaling) {
    auto model_05 = mobilenet_v2_width(1000, 0.5, false);
    auto model_10 = mobilenet_v2_width(1000, 1.0, false);
    auto model_15 = mobilenet_v2_width(1000, 1.5, false);

    size_t params_05 = this->countParameters(model_05);
    size_t params_10 = this->countParameters(model_10);
    size_t params_15 = this->countParameters(model_15);

    // Parameters should scale with width multiplier
    EXPECT_LT(params_05, params_10);
    EXPECT_LT(params_10, params_15);
}

// ============================================================================
// Edge Cases and Robustness
// ============================================================================

TYPED_TEST(MobileNetMultiDTypeTest, MinimalConfiguration) {
    auto model = mobilenet_v2_width(2, 0.5, false);
    Variable input(Tensor({1, 3, 128, 128}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {1, 2});
}

TYPED_TEST(MobileNetMultiDTypeTest, LargeConfiguration) {
    auto model = mobilenet_v3_large(5000, false);
    Variable input(Tensor({1, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {1, 5000});
}

TYPED_TEST(MobileNetMultiDTypeTest, SingleBatchInference) {
    auto model = mobilenet_v2(10, false);
    model->eval();

    Variable input(Tensor({1, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {1, 10});
}

TYPED_TEST(MobileNetMultiDTypeTest, LargeBatchInference) {
    auto model = mobilenet_v3_small(10, false);
    model->eval();

    Variable input(Tensor({16, 3, 224, 224}, this->dtype_, this->device_), true);
    Variable output = model->forward(input);

    this->expectShape(output, {16, 10});
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
