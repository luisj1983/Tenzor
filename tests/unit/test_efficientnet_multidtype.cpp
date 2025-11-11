/**
 * @file test_efficientnet_multidtype.cpp
 * @brief Comprehensive multi-dtype tests for EfficientNet B0-B7 variants
 * @details Tests Float32, Float64, and Float16 support for efficient vision models
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/efficientnet.hpp"
#include "../../include/tenzor/core/tensor.hpp"
#include "../../include/tenzor/autograd/variable.hpp"

using namespace tenzor;
using namespace tenzor::models;

// ============================================================================
// Multi-DType Test Fixture
// ============================================================================

class EfficientNetMultiDTypeTest : public ::testing::TestWithParam<DType> {
protected:
    void SetUp() override {
        dtype_ = GetParam();
        device_ = Device::cpu();
    }

    DType dtype_;
    Device device_;

    // Helper to check shapes match
    void expectShapeEquals(const Tensor& tensor,
                          const std::vector<int64_t>& expected_shape) {
        auto shape = tensor.shape();
        EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), expected_shape);
    }

    // Helper to check dtype matches
    void expectDTypeEquals(const Tensor& tensor, DType expected_dtype) {
        EXPECT_EQ(tensor.dtype(), expected_dtype);
    }
};

// ============================================================================
// SqueezeExcitation Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, SqueezeExcitationForwardShape) {
    int64_t channels = 64;
    auto se = std::make_shared<EfficientNetSqueezeExcitation>(channels, 0.25);

    Variable input(Tensor({2, channels, 14, 14}, dtype_, device_), true);
    Variable output = se->forward(input);

    // SE preserves input shape and dtype
    expectShapeEquals(output.tensor(), {2, channels, 14, 14});
    expectDTypeEquals(output.tensor(), dtype_);
}

TEST_P(EfficientNetMultiDTypeTest, SqueezeExcitationDifferentChannels) {
    // Test with various channel sizes
    std::vector<int64_t> channel_sizes = {32, 64, 128, 256};

    for (int64_t channels : channel_sizes) {
        auto se = std::make_shared<EfficientNetSqueezeExcitation>(channels, 0.25);
        Variable input(Tensor({1, channels, 7, 7}, dtype_, device_), true);
        Variable output = se->forward(input);

        expectShapeEquals(output.tensor(), {1, channels, 7, 7});
        expectDTypeEquals(output.tensor(), dtype_);
    }
}

TEST_P(EfficientNetMultiDTypeTest, SqueezeExcitationGradientFlow) {
    auto se = std::make_shared<EfficientNetSqueezeExcitation>(32, 0.25);

    Variable input(Tensor({1, 32, 7, 7}, dtype_, device_), true);
    Variable output = se->forward(input);
    Variable loss = tenzor::sum(output * output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    auto params = se->parameters();
    EXPECT_GT(params.size(), 0);
    for (const auto& param : params) {
        EXPECT_TRUE(param->grad().has_value());
        expectDTypeEquals(param->grad().value(), dtype_);
    }
}

TEST_P(EfficientNetMultiDTypeTest, SqueezeExcitationDifferentReductionRatios) {
    int64_t channels = 64;
    std::vector<double> reduction_ratios = {0.25, 0.5, 0.75};

    for (double ratio : reduction_ratios) {
        auto se = std::make_shared<EfficientNetSqueezeExcitation>(channels, ratio);
        Variable input(Tensor({2, channels, 14, 14}, dtype_, device_), true);
        Variable output = se->forward(input);

        expectShapeEquals(output.tensor(), {2, channels, 14, 14});
        expectDTypeEquals(output.tensor(), dtype_);
    }
}

// ============================================================================
// MBConvBlock Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, MBConvBlockNoExpansionShape) {
    // MBConv with expand_ratio=1 (no expansion phase)
    auto block = std::make_shared<MBConvBlock>(32, 32, 1, 3, 1, true, 0.25, 0.0);

    Variable input(Tensor({2, 32, 28, 28}, dtype_, device_), true);
    Variable output = block->forward(input);

    expectShapeEquals(output.tensor(), {2, 32, 28, 28});
    expectDTypeEquals(output.tensor(), dtype_);
}

TEST_P(EfficientNetMultiDTypeTest, MBConvBlockWithExpansionShape) {
    // MBConv with expand_ratio=6
    auto block = std::make_shared<MBConvBlock>(32, 64, 6, 3, 2, true, 0.25, 0.0);

    Variable input(Tensor({2, 32, 28, 28}, dtype_, device_), true);
    Variable output = block->forward(input);

    // Stride=2 halves spatial dims, channels change to out_channels
    expectShapeEquals(output.tensor(), {2, 64, 14, 14});
    expectDTypeEquals(output.tensor(), dtype_);
}

TEST_P(EfficientNetMultiDTypeTest, MBConvBlockDifferentExpansionRatios) {
    std::vector<int> expansion_ratios = {1, 3, 6};

    for (int ratio : expansion_ratios) {
        auto block = std::make_shared<MBConvBlock>(16, 24, ratio, 3, 1, true, 0.25, 0.0);
        Variable input(Tensor({2, 16, 56, 56}, dtype_, device_), true);
        Variable output = block->forward(input);

        expectShapeEquals(output.tensor(), {2, 24, 56, 56});
        expectDTypeEquals(output.tensor(), dtype_);
    }
}

TEST_P(EfficientNetMultiDTypeTest, MBConvBlockDifferentKernelSizes) {
    std::vector<int> kernel_sizes = {3, 5};

    for (int kernel : kernel_sizes) {
        auto block = std::make_shared<MBConvBlock>(24, 40, 6, kernel, 1, true, 0.25, 0.0);
        Variable input(Tensor({2, 24, 28, 28}, dtype_, device_), true);
        Variable output = block->forward(input);

        expectShapeEquals(output.tensor(), {2, 40, 28, 28});
        expectDTypeEquals(output.tensor(), dtype_);
    }
}

TEST_P(EfficientNetMultiDTypeTest, MBConvBlockDifferentStrides) {
    // Test stride=1 (same size) and stride=2 (downsampling)
    auto block_stride1 = std::make_shared<MBConvBlock>(32, 48, 6, 3, 1, true, 0.25, 0.0);
    auto block_stride2 = std::make_shared<MBConvBlock>(32, 48, 6, 3, 2, true, 0.25, 0.0);

    Variable input(Tensor({2, 32, 56, 56}, dtype_, device_), true);

    Variable output1 = block_stride1->forward(input);
    expectShapeEquals(output1.tensor(), {2, 48, 56, 56});

    Variable output2 = block_stride2->forward(input);
    expectShapeEquals(output2.tensor(), {2, 48, 28, 28});
}

TEST_P(EfficientNetMultiDTypeTest, MBConvBlockGradientFlow) {
    auto block = std::make_shared<MBConvBlock>(16, 24, 6, 3, 1, true, 0.25, 0.0);

    Variable input(Tensor({2, 16, 56, 56}, dtype_, device_), true);
    Variable output = block->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    expectDTypeEquals(input.grad().value(), dtype_);

    auto params = block->parameters();
    EXPECT_GT(params.size(), 0);
    for (const auto& param : params) {
        EXPECT_TRUE(param->grad().has_value());
        expectDTypeEquals(param->grad().value(), dtype_);
    }
}

TEST_P(EfficientNetMultiDTypeTest, MBConvBlockWithoutSE) {
    // Test MBConv without Squeeze-and-Excitation
    auto block = std::make_shared<MBConvBlock>(32, 48, 6, 3, 1, false, 0.25, 0.0);

    Variable input(Tensor({2, 32, 28, 28}, dtype_, device_), true);
    Variable output = block->forward(input);

    expectShapeEquals(output.tensor(), {2, 48, 28, 28});
    expectDTypeEquals(output.tensor(), dtype_);
}

// ============================================================================
// EfficientNet-B0 Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB0ForwardShape) {
    auto model = efficientnet_b0(1000, false);

    Variable input(Tensor({2, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);

    expectShapeEquals(output.tensor(), {2, 1000});
    expectDTypeEquals(output.tensor(), dtype_);
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB0GradientFlow) {
    auto model = efficientnet_b0(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    expectDTypeEquals(input.grad().value(), dtype_);

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB0SmallBatch) {
    auto model = efficientnet_b0(10, false);

    // Test with batch size 1
    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);

    expectShapeEquals(output.tensor(), {1, 10});
    expectDTypeEquals(output.tensor(), dtype_);
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB0CustomClasses) {
    // Test with non-standard number of classes
    auto model = efficientnet_b0(100, false);

    Variable input(Tensor({2, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);

    expectShapeEquals(output.tensor(), {2, 100});
    expectDTypeEquals(output.tensor(), dtype_);
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB0DifferentBatchSizes) {
    auto model = efficientnet_b0(10, false);
    std::vector<int64_t> batch_sizes = {1, 2, 4, 8};

    for (int64_t batch : batch_sizes) {
        Variable input(Tensor({batch, 3, 224, 224}, dtype_, device_), true);
        Variable output = model->forward(input);

        expectShapeEquals(output.tensor(), {batch, 10});
        expectDTypeEquals(output.tensor(), dtype_);
    }
}

// ============================================================================
// EfficientNet-B1 Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB1ForwardShape) {
    auto model = efficientnet_b1(1000, false);

    Variable input(Tensor({2, 3, 240, 240}, dtype_, device_), true);
    Variable output = model->forward(input);

    expectShapeEquals(output.tensor(), {2, 1000});
    expectDTypeEquals(output.tensor(), dtype_);
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB1GradientFlow) {
    auto model = efficientnet_b1(10, false);
    model->train();

    Variable input(Tensor({1, 3, 240, 240}, dtype_, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    expectDTypeEquals(input.grad().value(), dtype_);
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB1CustomResolution) {
    auto model = efficientnet_b1(10, false);

    // Test with different resolutions
    Variable input_224(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output_224 = model->forward(input_224);
    expectShapeEquals(output_224.tensor(), {1, 10});

    Variable input_256(Tensor({1, 3, 256, 256}, dtype_, device_), true);
    Variable output_256 = model->forward(input_256);
    expectShapeEquals(output_256.tensor(), {1, 10});
}

// ============================================================================
// EfficientNet-B2 Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB2ForwardShape) {
    auto model = efficientnet_b2(1000, false);

    Variable input(Tensor({2, 3, 260, 260}, dtype_, device_), true);
    Variable output = model->forward(input);

    expectShapeEquals(output.tensor(), {2, 1000});
    expectDTypeEquals(output.tensor(), dtype_);
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB2GradientFlow) {
    auto model = efficientnet_b2(10, false);
    model->train();

    Variable input(Tensor({1, 3, 260, 260}, dtype_, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    expectDTypeEquals(input.grad().value(), dtype_);
}

// ============================================================================
// EfficientNet-B3 Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB3ForwardShape) {
    auto model = efficientnet_b3(1000, false);

    Variable input(Tensor({1, 3, 300, 300}, dtype_, device_), true);
    Variable output = model->forward(input);

    expectShapeEquals(output.tensor(), {1, 1000});
    expectDTypeEquals(output.tensor(), dtype_);
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB3BatchSizeOne) {
    auto model = efficientnet_b3(10, false);

    // Test with batch size 1
    Variable input(Tensor({1, 3, 300, 300}, dtype_, device_), true);
    Variable output = model->forward(input);

    expectShapeEquals(output.tensor(), {1, 10});
    expectDTypeEquals(output.tensor(), dtype_);
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB3GradientFlow) {
    auto model = efficientnet_b3(10, false);
    model->train();

    Variable input(Tensor({1, 3, 300, 300}, dtype_, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    expectDTypeEquals(input.grad().value(), dtype_);
}

// ============================================================================
// EfficientNet-B4 Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB4ForwardShape) {
    auto model = efficientnet_b4(1000, false);

    Variable input(Tensor({1, 3, 380, 380}, dtype_, device_), true);
    Variable output = model->forward(input);

    expectShapeEquals(output.tensor(), {1, 1000});
    expectDTypeEquals(output.tensor(), dtype_);
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB4CustomClasses) {
    auto model = efficientnet_b4(50, false);

    Variable input(Tensor({1, 3, 380, 380}, dtype_, device_), true);
    Variable output = model->forward(input);

    expectShapeEquals(output.tensor(), {1, 50});
    expectDTypeEquals(output.tensor(), dtype_);
}

// ============================================================================
// EfficientNet-B5 Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB5ForwardShape) {
    auto model = efficientnet_b5(1000, false);

    Variable input(Tensor({1, 3, 456, 456}, dtype_, device_), true);
    Variable output = model->forward(input);

    expectShapeEquals(output.tensor(), {1, 1000});
    expectDTypeEquals(output.tensor(), dtype_);
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB5GradientFlow) {
    auto model = efficientnet_b5(10, false);
    model->train();

    Variable input(Tensor({1, 3, 456, 456}, dtype_, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    expectDTypeEquals(input.grad().value(), dtype_);
}

// ============================================================================
// EfficientNet-B6 Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB6ForwardShape) {
    auto model = efficientnet_b6(1000, false);

    Variable input(Tensor({1, 3, 528, 528}, dtype_, device_), true);
    Variable output = model->forward(input);

    expectShapeEquals(output.tensor(), {1, 1000});
    expectDTypeEquals(output.tensor(), dtype_);
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB6CustomClasses) {
    auto model = efficientnet_b6(200, false);

    Variable input(Tensor({1, 3, 528, 528}, dtype_, device_), true);
    Variable output = model->forward(input);

    expectShapeEquals(output.tensor(), {1, 200});
    expectDTypeEquals(output.tensor(), dtype_);
}

// ============================================================================
// EfficientNet-B7 Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB7ForwardShape) {
    auto model = efficientnet_b7(1000, false);

    Variable input(Tensor({1, 3, 600, 600}, dtype_, device_), true);
    Variable output = model->forward(input);

    expectShapeEquals(output.tensor(), {1, 1000});
    expectDTypeEquals(output.tensor(), dtype_);
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB7GradientFlow) {
    auto model = efficientnet_b7(10, false);
    model->train();

    Variable input(Tensor({1, 3, 600, 600}, dtype_, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    expectDTypeEquals(input.grad().value(), dtype_);
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB7BatchProcessing) {
    auto model = efficientnet_b7(10, false);

    Variable input(Tensor({2, 3, 600, 600}, dtype_, device_), true);
    Variable output = model->forward(input);

    expectShapeEquals(output.tensor(), {2, 10});
    expectDTypeEquals(output.tensor(), dtype_);
}

// ============================================================================
// Compound Scaling Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, CompoundScalingConfig) {
    auto config_b0 = EfficientNetConfig::efficientnet_b0(1000);
    auto config_scaled = EfficientNetConfig::efficientnet_b0(1000);
    config_scaled.apply_compound_scaling(0.5);

    // Scaled config should have larger width and depth
    EXPECT_GT(config_scaled.width_mult, config_b0.width_mult);
    EXPECT_GT(config_scaled.depth_mult, config_b0.depth_mult);
    EXPECT_GT(config_scaled.resolution, config_b0.resolution);
}

TEST_P(EfficientNetMultiDTypeTest, CompoundScalingEffects) {
    // Test that compound scaling produces different model sizes
    auto model_b0 = efficientnet_b0(10, false);
    auto model_b3 = efficientnet_b3(10, false);

    auto params_b0 = model_b0->parameters();
    auto params_b3 = model_b3->parameters();

    // B3 should have more parameters than B0
    EXPECT_GT(params_b3.size(), params_b0.size());
}

// ============================================================================
// Different Image Sizes Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, DifferentImageSizesB0) {
    auto model = efficientnet_b0(10, false);

    // Test with various image sizes
    std::vector<int64_t> sizes = {224, 256, 384};

    for (int64_t size : sizes) {
        Variable input(Tensor({1, 3, size, size}, dtype_, device_), true);
        Variable output = model->forward(input);

        expectShapeEquals(output.tensor(), {1, 10});
        expectDTypeEquals(output.tensor(), dtype_);
    }
}

TEST_P(EfficientNetMultiDTypeTest, DifferentImageSizesB4) {
    auto model = efficientnet_b4(10, false);

    // Test with various image sizes (larger for B4)
    std::vector<int64_t> sizes = {320, 380, 512};

    for (int64_t size : sizes) {
        Variable input(Tensor({1, 3, size, size}, dtype_, device_), true);
        Variable output = model->forward(input);

        expectShapeEquals(output.tensor(), {1, 10});
        expectDTypeEquals(output.tensor(), dtype_);
    }
}

TEST_P(EfficientNetMultiDTypeTest, NonSquareInputImages) {
    auto model = efficientnet_b0(10, false);

    // Test with non-square inputs (should still work due to adaptive pooling)
    Variable input_rect1(Tensor({1, 3, 224, 256}, dtype_, device_), true);
    Variable output_rect1 = model->forward(input_rect1);
    expectShapeEquals(output_rect1.tensor(), {1, 10});

    Variable input_rect2(Tensor({1, 3, 256, 224}, dtype_, device_), true);
    Variable output_rect2 = model->forward(input_rect2);
    expectShapeEquals(output_rect2.tensor(), {1, 10});
}

// ============================================================================
// Cross-Variant Comparison Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, VariantResolutionProgression) {
    // Verify that higher variants can handle larger resolutions
    auto b0 = efficientnet_b0(10, false);
    auto b3 = efficientnet_b3(10, false);
    auto b7 = efficientnet_b7(10, false);

    Variable input_b0(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output_b0 = b0->forward(input_b0);
    expectShapeEquals(output_b0.tensor(), {1, 10});

    Variable input_b3(Tensor({1, 3, 300, 300}, dtype_, device_), true);
    Variable output_b3 = b3->forward(input_b3);
    expectShapeEquals(output_b3.tensor(), {1, 10});

    Variable input_b7(Tensor({1, 3, 600, 600}, dtype_, device_), true);
    Variable output_b7 = b7->forward(input_b7);
    expectShapeEquals(output_b7.tensor(), {1, 10});
}

TEST_P(EfficientNetMultiDTypeTest, VariantParameterScaling) {
    // Verify parameter count increases across variants
    std::vector<std::shared_ptr<EfficientNet>> models = {
        efficientnet_b0(10, false),
        efficientnet_b1(10, false),
        efficientnet_b2(10, false)
    };

    std::vector<size_t> param_counts;
    for (const auto& model : models) {
        auto params = model->parameters();
        size_t count = 0;
        for (const auto& p : params) {
            size_t param_size = 1;
            for (auto dim : p->tensor().shape()) {
                param_size *= dim;
            }
            count += param_size;
        }
        param_counts.push_back(count);
    }

    // Each variant should have more parameters than the previous
    for (size_t i = 1; i < param_counts.size(); ++i) {
        EXPECT_GT(param_counts[i], param_counts[i-1]);
    }
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, MinimalBatchSize) {
    // Test all variants with batch size 1
    std::vector<std::shared_ptr<EfficientNet>> models = {
        efficientnet_b0(10, false),
        efficientnet_b2(10, false),
        efficientnet_b4(10, false)
    };

    std::vector<int64_t> resolutions = {224, 260, 380};

    for (size_t i = 0; i < models.size(); ++i) {
        Variable input(Tensor({1, 3, resolutions[i], resolutions[i]}, dtype_, device_), true);
        Variable output = models[i]->forward(input);
        expectShapeEquals(output.tensor(), {1, 10});
        expectDTypeEquals(output.tensor(), dtype_);
    }
}

TEST_P(EfficientNetMultiDTypeTest, LargeBatchSize) {
    auto model = efficientnet_b0(10, false);

    // Test with larger batch size
    Variable input(Tensor({16, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);

    expectShapeEquals(output.tensor(), {16, 10});
    expectDTypeEquals(output.tensor(), dtype_);
}

TEST_P(EfficientNetMultiDTypeTest, SingleClassOutput) {
    // Test with single class (binary classification scenario)
    auto model = efficientnet_b0(1, false);

    Variable input(Tensor({2, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);

    expectShapeEquals(output.tensor(), {2, 1});
    expectDTypeEquals(output.tensor(), dtype_);
}

TEST_P(EfficientNetMultiDTypeTest, ManyClassesOutput) {
    // Test with many classes (fine-grained classification)
    auto model = efficientnet_b0(10000, false);

    Variable input(Tensor({2, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);

    expectShapeEquals(output.tensor(), {2, 10000});
    expectDTypeEquals(output.tensor(), dtype_);
}

// ============================================================================
// Training vs Evaluation Mode Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, TrainingVsEvalMode) {
    auto model = efficientnet_b0(10, false);
    Variable input(Tensor({2, 3, 224, 224}, dtype_, device_), true);

    // Test in evaluation mode
    model->eval();
    Variable output_eval = model->forward(input);
    expectShapeEquals(output_eval.tensor(), {2, 10});

    // Test in training mode
    model->train();
    Variable output_train = model->forward(input);
    expectShapeEquals(output_train.tensor(), {2, 10});

    // Both should produce same shape
    auto shape_eval = output_eval.tensor().shape(); auto shape_train = output_train.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_eval.begin(), shape_eval.end()), std::vector<int64_t>(shape_train.begin(), shape_train.end()));
}

// ============================================================================
// Instantiate Tests for All DTypes
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    MultiDTypeSupport,
    EfficientNetMultiDTypeTest,
    ::testing::Values(
        DType::Float32,
        DType::Float64,
        DType::Float16
    ),
    [](const ::testing::TestParamInfo<DType>& info) {
        switch (info.param) {
            case DType::Float32: return "Float32";
            case DType::Float64: return "Float64";
            case DType::Float16: return "Float16";
            default: return "Unknown";
        }
    }
);

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();  // Initialize Tenzor library and backends
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
