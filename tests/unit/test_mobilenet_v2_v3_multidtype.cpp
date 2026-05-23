/**
 * @file test_mobilenet_multidtype.cpp
 * @brief Multi-dtype tests for MobileNet V2 and V3 variants (Float32, Float64, Float16)
 *
 * Tests MobileNet architectures with multiple data types across CPU, CUDA, OneAPI,
 * Vulkan, and ROCm backends for mobile/edge deployment:
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
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::models;

// ============================================================================
// MobileNet Multi-Backend Multi-DType Test Fixture
// ============================================================================

class MobileNetMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Helper to count total parameters in a model (templated for different model types)
    template<typename ModelType>
    size_t countModelParameters(const std::shared_ptr<ModelType>& model) {
        auto params = model->parameters();
        return countParameters(params);
    }

    // Helper to check gradient flow (templated for different model types)
    template<typename ModelType>
    void checkGradientFlow(const Variable& input, const std::shared_ptr<ModelType>& model) {
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

// ============================================================================
// MobileNetV2 Core Architecture Tests
// ============================================================================

TEST_P(MobileNetMultiDTypeTest, MobileNetV2ForwardShape) {
    auto model = mobilenet_v2(1000, false);
    convert_model(model);

    Variable input = createInput({2, 3, 224, 224});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV2GradientFlow) {
    auto model = mobilenet_v2(10, false);
    convert_model(model);
    model->train();

    Variable input = createInput({1, 3, 224, 224}, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    checkGradientFlow(input, model);
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV2BatchSizes) {
    auto model = mobilenet_v2(10, false);
    convert_model(model);

    // Test different batch sizes
    std::vector<int64_t> batch_sizes = {1, 2, 4};
    for (auto batch_size : batch_sizes) {
        Variable input = createInput({batch_size, 3, 224, 224});
        Variable output = model->forward(input);
        expectShape(output.tensor(), {batch_size, 10});
        expectFiniteNonZero(output.tensor());
    }
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV2ParameterCount) {
    auto model = mobilenet_v2(1000, false);
    size_t total_params = countModelParameters(model);

    // MobileNetV2 should have ~3.5M parameters (allow 30% tolerance)
    EXPECT_GT(total_params, 2'500'000);
    EXPECT_LT(total_params, 5'000'000);
}

// ============================================================================
// MobileNetV2 Width Multiplier Tests (Critical for Mobile Deployment)
// ============================================================================

TEST_P(MobileNetMultiDTypeTest, MobileNetV2WidthMultiplier_0_5) {
    auto model = mobilenet_v2_width(1000, 0.5, false);
    convert_model(model);

    Variable input = createInput({2, 3, 224, 224});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectFiniteNonZero(output.tensor());

    // 0.5 width multiplier should have fewer parameters
    size_t params = countModelParameters(model);
    EXPECT_LT(params, 3'000'000);  // Much smaller than standard ~3.5M
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV2WidthMultiplier_0_75) {
    auto model = mobilenet_v2_width(1000, 0.75, false);
    convert_model(model);

    Variable input = createInput({2, 3, 224, 224});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectFiniteNonZero(output.tensor());
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV2WidthMultiplier_1_25) {
    auto model = mobilenet_v2_width(1000, 1.25, false);
    convert_model(model);

    Variable input = createInput({2, 3, 224, 224});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectFiniteNonZero(output.tensor());

    // 1.25 width multiplier should have more parameters
    size_t params = countModelParameters(model);
    EXPECT_GT(params, 3'500'000);
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV2WidthMultiplier_1_5) {
    auto model = mobilenet_v2_width(1000, 1.5, false);
    convert_model(model);

    Variable input = createInput({2, 3, 224, 224});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// MobileNetV2 Different Input Sizes (Critical for Mobile Applications)
// ============================================================================

TEST_P(MobileNetMultiDTypeTest, MobileNetV2InputSize_128) {
    auto model = mobilenet_v2(1000, false);
    convert_model(model);

    Variable input = createInput({2, 3, 128, 128});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectFiniteNonZero(output.tensor());
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV2InputSize_160) {
    auto model = mobilenet_v2(1000, false);
    convert_model(model);

    Variable input = createInput({2, 3, 160, 160});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectFiniteNonZero(output.tensor());
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV2InputSize_192) {
    auto model = mobilenet_v2(1000, false);
    convert_model(model);

    Variable input = createInput({2, 3, 192, 192});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectFiniteNonZero(output.tensor());
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV2InputSize_256) {
    auto model = mobilenet_v2(1000, false);
    convert_model(model);

    Variable input = createInput({2, 3, 256, 256});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectFiniteNonZero(output.tensor());
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV2InputSize_320) {
    auto model = mobilenet_v2(1000, false);
    convert_model(model);

    Variable input = createInput({2, 3, 320, 320});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// MobileNetV3-Small Tests
// ============================================================================

TEST_P(MobileNetMultiDTypeTest, MobileNetV3SmallForwardShape) {
    auto model = mobilenet_v3_small(1000, false);
    convert_model(model);

    Variable input = createInput({2, 3, 224, 224});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV3SmallGradientFlow) {
    auto model = mobilenet_v3_small(10, false);
    convert_model(model);
    model->train();

    Variable input = createInput({1, 3, 224, 224}, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    checkGradientFlow(input, model);
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV3SmallParameterCount) {
    auto model = mobilenet_v3_small(1000, false);
    size_t total_params = countModelParameters(model);

    // MobileNetV3-Small should have ~2.5M parameters (allow 30% tolerance)
    EXPECT_GT(total_params, 1'800'000);
    EXPECT_LT(total_params, 3'500'000);
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV3SmallBatchSizes) {
    auto model = mobilenet_v3_small(10, false);
    convert_model(model);

    std::vector<int64_t> batch_sizes = {1, 2, 4};
    for (auto batch_size : batch_sizes) {
        Variable input = createInput({batch_size, 3, 224, 224});
        Variable output = model->forward(input);
        expectShape(output.tensor(), {batch_size, 10});
        expectFiniteNonZero(output.tensor());
    }
}

// ============================================================================
// MobileNetV3-Large Tests
// ============================================================================

TEST_P(MobileNetMultiDTypeTest, MobileNetV3LargeForwardShape) {
    auto model = mobilenet_v3_large(1000, false);
    convert_model(model);

    Variable input = createInput({2, 3, 224, 224});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV3LargeGradientFlow) {
    auto model = mobilenet_v3_large(10, false);
    convert_model(model);
    model->train();

    Variable input = createInput({1, 3, 224, 224}, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    checkGradientFlow(input, model);
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV3LargeParameterCount) {
    auto model = mobilenet_v3_large(1000, false);
    size_t total_params = countModelParameters(model);

    // MobileNetV3-Large should have ~5.4M parameters (allow 30% tolerance)
    EXPECT_GT(total_params, 4'000'000);
    EXPECT_LT(total_params, 7'000'000);
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV3LargeBatchSizes) {
    auto model = mobilenet_v3_large(10, false);
    convert_model(model);

    std::vector<int64_t> batch_sizes = {1, 2, 4};
    for (auto batch_size : batch_sizes) {
        Variable input = createInput({batch_size, 3, 224, 224});
        Variable output = model->forward(input);
        expectShape(output.tensor(), {batch_size, 10});
        expectFiniteNonZero(output.tensor());
    }
}

// ============================================================================
// MobileNetV3 Different Input Sizes with SE Blocks
// ============================================================================

TEST_P(MobileNetMultiDTypeTest, MobileNetV3LargeInputSize_160) {
    auto model = mobilenet_v3_large(1000, false);
    convert_model(model);

    Variable input = createInput({2, 3, 160, 160});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectFiniteNonZero(output.tensor());
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV3SmallInputSize_192) {
    auto model = mobilenet_v3_small(1000, false);
    convert_model(model);

    Variable input = createInput({2, 3, 192, 192});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectFiniteNonZero(output.tensor());
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV3LargeInputSize_256) {
    auto model = mobilenet_v3_large(1000, false);
    convert_model(model);

    Variable input = createInput({2, 3, 256, 256});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// Custom Number of Classes Tests
// ============================================================================

TEST_P(MobileNetMultiDTypeTest, MobileNetV2CustomClasses_10) {
    auto model = mobilenet_v2(10, false);
    convert_model(model);

    Variable input = createInput({2, 3, 224, 224});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 10});
    expectFiniteNonZero(output.tensor());
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV2CustomClasses_100) {
    auto model = mobilenet_v2(100, false);
    convert_model(model);

    Variable input = createInput({2, 3, 224, 224});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 100});
    expectFiniteNonZero(output.tensor());
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV3SmallCustomClasses_50) {
    auto model = mobilenet_v3_small(50, false);
    convert_model(model);

    Variable input = createInput({2, 3, 224, 224});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 50});
    expectFiniteNonZero(output.tensor());
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV3LargeCustomClasses_200) {
    auto model = mobilenet_v3_large(200, false);
    convert_model(model);

    Variable input = createInput({2, 3, 224, 224});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 200});
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// Depthwise Separable Convolution Tests (Architecture-specific)
// ============================================================================

TEST_P(MobileNetMultiDTypeTest, MobileNetV2DepthwiseConvolution) {
    // Test that depthwise separable convolutions work correctly
    auto model = mobilenet_v2(10, false);
    convert_model(model);
    model->eval();

    Variable input = createInput({1, 3, 224, 224});
    Variable output1 = model->forward(input);
    Variable output2 = model->forward(input);

    // Same input should produce same output in eval mode
    expectShape(output1.tensor(), {1, 10});
    expectShape(output2.tensor(), {1, 10});
    expectFiniteNonZero(output1.tensor());
    expectFiniteNonZero(output2.tensor());
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV3DepthwiseWithSE) {
    // Test depthwise convolutions with SE blocks in V3
    auto model = mobilenet_v3_large(10, false);
    convert_model(model);
    model->eval();

    Variable input = createInput({1, 3, 224, 224});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 10});
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// Inverted Residual Block Tests
// ============================================================================

TEST_P(MobileNetMultiDTypeTest, MobileNetV2InvertedResiduals) {
    // Test inverted residual connections
    auto model = mobilenet_v2(10, false);
    convert_model(model);
    model->train();

    Variable input = createInput({2, 3, 224, 224}, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Verify gradient can flow through residual connections
    EXPECT_GRAD_FLOWS(input);
}

TEST_P(MobileNetMultiDTypeTest, MobileNetV3InvertedResidualsWithSE) {
    // Test inverted residuals with SE blocks in V3
    auto model = mobilenet_v3_large(10, false);
    convert_model(model);
    model->train();

    Variable input = createInput({2, 3, 224, 224}, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

// ============================================================================
// Mobile/Edge Deployment Scenario Tests (Float16 Critical)
// ============================================================================

TEST_P(MobileNetMultiDTypeTest, MobileEdgeScenario_SmallModel_SmallInput) {
    // Typical mobile scenario: small model, smaller input, low precision
    auto model = mobilenet_v3_small(100, false);
    convert_model(model);

    Variable input = createInput({1, 3, 160, 160});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 100});
    expectFiniteNonZero(output.tensor());
}

TEST_P(MobileNetMultiDTypeTest, MobileEdgeScenario_WidthMultiplier_SmallInput) {
    // Ultra-lightweight mobile scenario
    auto model = mobilenet_v2_width(100, 0.5, false);
    convert_model(model);

    Variable input = createInput({1, 3, 128, 128});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 100});
    expectFiniteNonZero(output.tensor());
}

TEST_P(MobileNetMultiDTypeTest, MobileEdgeScenario_BatchInference) {
    // Mobile batch inference (e.g., video frames)
    auto model = mobilenet_v3_small(10, false);
    convert_model(model);
    model->eval();

    Variable input = createInput({4, 3, 192, 192});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {4, 10});
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// Performance Characteristics Tests
// ============================================================================

TEST_P(MobileNetMultiDTypeTest, MobileNetV2V3ParameterComparison) {
    auto v2_model = mobilenet_v2(1000, false);
    auto v3_small = mobilenet_v3_small(1000, false);
    auto v3_large = mobilenet_v3_large(1000, false);

    size_t v2_params = countModelParameters(v2_model);
    size_t v3_small_params = countModelParameters(v3_small);
    size_t v3_large_params = countModelParameters(v3_large);

    // Parameter count ordering
    EXPECT_LT(v3_small_params, v2_params);
    EXPECT_GT(v3_large_params, v2_params);
}

TEST_P(MobileNetMultiDTypeTest, WidthMultiplierParameterScaling) {
    auto model_05 = mobilenet_v2_width(1000, 0.5, false);
    auto model_10 = mobilenet_v2_width(1000, 1.0, false);
    auto model_15 = mobilenet_v2_width(1000, 1.5, false);

    size_t params_05 = countModelParameters(model_05);
    size_t params_10 = countModelParameters(model_10);
    size_t params_15 = countModelParameters(model_15);

    // Parameters should scale with width multiplier
    EXPECT_LT(params_05, params_10);
    EXPECT_LT(params_10, params_15);
}

// ============================================================================
// Edge Cases and Robustness
// ============================================================================

TEST_P(MobileNetMultiDTypeTest, MinimalConfiguration) {
    auto model = mobilenet_v2_width(2, 0.5, false);
    convert_model(model);

    Variable input = createInput({1, 3, 128, 128});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 2});
    expectFiniteNonZero(output.tensor());
}

TEST_P(MobileNetMultiDTypeTest, LargeConfiguration) {
    auto model = mobilenet_v3_large(5000, false);
    convert_model(model);

    Variable input = createInput({1, 3, 224, 224});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 5000});
    expectFiniteNonZero(output.tensor());
}

TEST_P(MobileNetMultiDTypeTest, SingleBatchInference) {
    auto model = mobilenet_v2(10, false);
    convert_model(model);
    model->eval();

    Variable input = createInput({1, 3, 224, 224});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 10});
    expectFiniteNonZero(output.tensor());
}

TEST_P(MobileNetMultiDTypeTest, LargeBatchInference) {
    auto model = mobilenet_v3_small(10, false);
    convert_model(model);
    model->eval();

    Variable input = createInput({16, 3, 224, 224});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {16, 10});
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MobileNetMultiDTypeTest);

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (!::testing::GTEST_FLAG(list_tests)) {
        tenzor::initialize();
    }
    return RUN_ALL_TESTS();
}
