/**
 * @file test_efficientnet_multidtype.cpp
 * @brief Multi-backend and multi-dtype tests for EfficientNet B0-B7 variants
 * @details Tests Float32, Float64, and Float16 support across CPU, CUDA, OneAPI backends
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include "../../include/tenzor/models/efficientnet.hpp"
#include <tenzor/nn/offload.hpp>
#include <memory>
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::models;
using namespace tenzor::testing;

// ============================================================================
// Multi-Backend Multi-DType Test Fixture
// ============================================================================

class EfficientNetMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    std::unique_ptr<nn::OffloadContext> offload_ctx_;

    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
    }

    /**
     * @brief Convert model to target dtype and device
     * @note CPU-start offloading was removed due to stability issues;
     *       models run directly on GPU instead.
     */
    template <typename ModuleT>
    void convert_model_with_offload(ModuleT& model) {
        // Use standard conversion for all backends
        convert_model(model);
    }

    template <typename ModuleT>
    void enable_offloading_if_needed(ModuleT& model) { (void)model; }

    /**
     * @brief Get appropriate input size for the current backend and dtype
     * GPU backends use smaller sizes for Float64 due to memory constraints
     */
    int64_t getInputSize(int64_t default_size) {
        if (device().type == Device::Type::CPU) {
            return default_size;
        }

        bool is_float64 = (dtype() == DType::Float64);

        if (is_float64) {
            // Float64: reduce sizes significantly (2x memory)
            if (default_size >= 600) return 288;
            if (default_size >= 456) return 256;
            if (default_size >= 380) return 224;
            if (default_size >= 300) return 192;
            return std::min(default_size, int64_t(160));
        }

        // Float32/Float16: only reduce large sizes slightly for large models
        if (default_size >= 600) return 384;
        return default_size;
    }
};

// ============================================================================
// SqueezeExcitation Multi-Backend Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, SqueezeExcitationForwardShape) {
    int64_t channels = 64;
    auto se = std::make_shared<EfficientNetSqueezeExcitation>(channels, 0.25);
    convert_model_with_offload(se);

    Variable input(Tensor({2, channels, 14, 14}, dtype(), device()), true);
    Variable output = se->forward(input);

    // SE preserves input shape and dtype
    expectShape(output.tensor(), {2, channels, 14, 14});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(EfficientNetMultiDTypeTest, SqueezeExcitationDifferentChannels) {
    // Test with various channel sizes
    std::vector<int64_t> channel_sizes = {32, 64, 128, 256};

    for (int64_t channels : channel_sizes) {
        auto se = std::make_shared<EfficientNetSqueezeExcitation>(channels, 0.25);
        convert_model_with_offload(se);
        Variable input(Tensor({1, channels, 7, 7}, dtype(), device()), true);
        Variable output = se->forward(input);

        expectShape(output.tensor(), {1, channels, 7, 7});
        expectDType(output.tensor());
        expectFiniteNonZero(output.tensor());
    }
}

TEST_P(EfficientNetMultiDTypeTest, SqueezeExcitationGradientFlow) {
    auto se = std::make_shared<EfficientNetSqueezeExcitation>(32, 0.25);
    convert_model_with_offload(se);

    Variable input(Tensor({1, 32, 7, 7}, dtype(), device()), true);
    Variable output = se->forward(input);
    Variable loss = tenzor::sum(output * output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    auto params = se->parameters();
    EXPECT_GT(params.size(), 0);
    for (const auto& param : params) {
        EXPECT_TRUE(param->grad().has_value());
        expectDType(param->grad().value());
    }
}

TEST_P(EfficientNetMultiDTypeTest, SqueezeExcitationDifferentReductionRatios) {
    int64_t channels = 64;
    std::vector<double> reduction_ratios = {0.25, 0.5, 0.75};

    for (double ratio : reduction_ratios) {
        auto se = std::make_shared<EfficientNetSqueezeExcitation>(channels, ratio);
        convert_model_with_offload(se);
        Variable input(Tensor({2, channels, 14, 14}, dtype(), device()), true);
        Variable output = se->forward(input);

        expectShape(output.tensor(), {2, channels, 14, 14});
        expectDType(output.tensor());
        expectFiniteNonZero(output.tensor());
    }
}

// ============================================================================
// MBConvBlock Multi-Backend Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, MBConvBlockNoExpansionShape) {
    // MBConv with expand_ratio=1 (no expansion phase)
    auto block = std::make_shared<MBConvBlock>(32, 32, 1, 3, 1, true, 0.25, 0.0);
    convert_model_with_offload(block);

    Variable input(Tensor({2, 32, 28, 28}, dtype(), device()), true);
    Variable output = block->forward(input);

    expectShape(output.tensor(), {2, 32, 28, 28});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(EfficientNetMultiDTypeTest, MBConvBlockWithExpansionShape) {
    // MBConv with expand_ratio=6
    auto block = std::make_shared<MBConvBlock>(32, 64, 6, 3, 2, true, 0.25, 0.0);
    convert_model_with_offload(block);

    Variable input(Tensor({2, 32, 28, 28}, dtype(), device()), true);
    Variable output = block->forward(input);

    // Stride=2 halves spatial dims, channels change to out_channels
    expectShape(output.tensor(), {2, 64, 14, 14});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(EfficientNetMultiDTypeTest, MBConvBlockDifferentExpansionRatios) {
    std::vector<int> expansion_ratios = {1, 3, 6};

    for (int ratio : expansion_ratios) {
        auto block = std::make_shared<MBConvBlock>(16, 24, ratio, 3, 1, true, 0.25, 0.0);
        convert_model_with_offload(block);
        Variable input(Tensor({2, 16, 56, 56}, dtype(), device()), true);
        Variable output = block->forward(input);

        expectShape(output.tensor(), {2, 24, 56, 56});
        expectDType(output.tensor());
        expectFiniteNonZero(output.tensor());
    }
}

TEST_P(EfficientNetMultiDTypeTest, MBConvBlockDifferentKernelSizes) {
    std::vector<int> kernel_sizes = {3, 5};

    for (int kernel : kernel_sizes) {
        auto block = std::make_shared<MBConvBlock>(24, 40, 6, kernel, 1, true, 0.25, 0.0);
        convert_model_with_offload(block);
        Variable input(Tensor({2, 24, 28, 28}, dtype(), device()), true);
        Variable output = block->forward(input);

        expectShape(output.tensor(), {2, 40, 28, 28});
        expectDType(output.tensor());
        expectFiniteNonZero(output.tensor());
    }
}

TEST_P(EfficientNetMultiDTypeTest, MBConvBlockDifferentStrides) {
    // Test stride=1 (same size) and stride=2 (downsampling)
    auto block_stride1 = std::make_shared<MBConvBlock>(32, 48, 6, 3, 1, true, 0.25, 0.0);
    auto block_stride2 = std::make_shared<MBConvBlock>(32, 48, 6, 3, 2, true, 0.25, 0.0);
    convert_model_with_offload(block_stride1);
    convert_model_with_offload(block_stride2);

    Variable input(Tensor({2, 32, 56, 56}, dtype(), device()), true);

    Variable output1 = block_stride1->forward(input);
    expectShape(output1.tensor(), {2, 48, 56, 56});
    expectFiniteNonZero(output1.tensor());

    Variable output2 = block_stride2->forward(input);
    expectShape(output2.tensor(), {2, 48, 28, 28});
    expectFiniteNonZero(output2.tensor());
}

TEST_P(EfficientNetMultiDTypeTest, MBConvBlockGradientFlow) {
    auto block = std::make_shared<MBConvBlock>(16, 24, 6, 3, 1, true, 0.25, 0.0);
    convert_model_with_offload(block);

    Variable input(Tensor({2, 16, 56, 56}, dtype(), device()), true);
    Variable output = block->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    expectDType(input.grad().value());

    auto params = block->parameters();
    EXPECT_GT(params.size(), 0);
    for (const auto& param : params) {
        EXPECT_TRUE(param->grad().has_value());
        expectDType(param->grad().value());
    }
}

TEST_P(EfficientNetMultiDTypeTest, MBConvBlockWithoutSE) {
    // Test MBConv without Squeeze-and-Excitation
    auto block = std::make_shared<MBConvBlock>(32, 48, 6, 3, 1, false, 0.25, 0.0);
    convert_model_with_offload(block);

    Variable input(Tensor({2, 32, 28, 28}, dtype(), device()), true);
    Variable output = block->forward(input);

    expectShape(output.tensor(), {2, 48, 28, 28});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// EfficientNet-B0 Multi-Backend Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB0ForwardShape) {
    auto model = efficientnet_b0(1000, false);
    convert_model_with_offload(model);

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB0GradientFlow) {
    auto model = efficientnet_b0(10, false);
    convert_model_with_offload(model);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    expectDType(input.grad().value());

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB0SmallBatch) {
    auto model = efficientnet_b0(10, false);
    convert_model_with_offload(model);

    // Test with batch size 1
    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 10});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB0CustomClasses) {
    // Test with non-standard number of classes
    auto model = efficientnet_b0(100, false);
    convert_model_with_offload(model);

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 100});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB0DifferentBatchSizes) {
    auto model = efficientnet_b0(10, false);
    convert_model_with_offload(model);
    std::vector<int64_t> batch_sizes = {1, 2, 4, 8};

    for (int64_t batch : batch_sizes) {
        Variable input(Tensor({batch, 3, 224, 224}, dtype(), device()), true);
        Variable output = model->forward(input);

        expectShape(output.tensor(), {batch, 10});
        expectDType(output.tensor());
        expectFiniteNonZero(output.tensor());
    }
}

// ============================================================================
// EfficientNet-B1 Multi-Backend Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB1ForwardShape) {
    auto model = efficientnet_b1(1000, false);
    convert_model_with_offload(model);

    Variable input(Tensor({2, 3, 240, 240}, dtype(), device()), true);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB1GradientFlow) {
    auto model = efficientnet_b1(10, false);
    convert_model_with_offload(model);
    model->train();

    Variable input(Tensor({1, 3, 240, 240}, dtype(), device()), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    expectDType(input.grad().value());
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB1CustomResolution) {
    auto model = efficientnet_b1(10, false);
    convert_model_with_offload(model);

    // Test with different resolutions
    Variable input_224(Tensor({1, 3, 224, 224}, dtype(), device()), true);
    Variable output_224 = model->forward(input_224);
    expectShape(output_224.tensor(), {1, 10});
    expectFiniteNonZero(output_224.tensor());

    Variable input_256(Tensor({1, 3, 256, 256}, dtype(), device()), true);
    Variable output_256 = model->forward(input_256);
    expectShape(output_256.tensor(), {1, 10});
    expectFiniteNonZero(output_256.tensor());
}

// ============================================================================
// EfficientNet-B2 Multi-Backend Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB2ForwardShape) {
    auto model = efficientnet_b2(1000, false);
    convert_model_with_offload(model);

    Variable input(Tensor({2, 3, 260, 260}, dtype(), device()), true);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB2GradientFlow) {
    auto model = efficientnet_b2(10, false);
    convert_model_with_offload(model);
    model->train();

    Variable input(Tensor({1, 3, 260, 260}, dtype(), device()), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    expectDType(input.grad().value());
}

// ============================================================================
// EfficientNet-B3 Multi-Backend Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB3ForwardShape) {
    auto model = efficientnet_b3(1000, false);
    convert_model_with_offload(model);

    Variable input(Tensor({1, 3, 300, 300}, dtype(), device()), true);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB3BatchSizeOne) {
    auto model = efficientnet_b3(10, false);
    convert_model_with_offload(model);

    // Test with batch size 1
    Variable input(Tensor({1, 3, 300, 300}, dtype(), device()), true);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 10});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB3GradientFlow) {
    auto model = efficientnet_b3(10, false);
    convert_model_with_offload(model);
    model->train();

    Variable input(Tensor({1, 3, 300, 300}, dtype(), device()), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    expectDType(input.grad().value());
}

// ============================================================================
// EfficientNet-B4 Multi-Backend Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB4ForwardShape) {
    auto model = efficientnet_b4(1000, false);
    convert_model_with_offload(model);

    Variable input(Tensor({1, 3, 380, 380}, dtype(), device()), true);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB4CustomClasses) {
    auto model = efficientnet_b4(50, false);
    convert_model_with_offload(model);

    Variable input(Tensor({1, 3, 380, 380}, dtype(), device()), true);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 50});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// EfficientNet-B5 Multi-Backend Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB5ForwardShape) {
    auto model = efficientnet_b5(1000, false);
    convert_model_with_offload(model);

    Variable input(Tensor({1, 3, 456, 456}, dtype(), device()), true);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB5GradientFlow) {
    auto model = efficientnet_b5(10, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(456);
    Variable input(Tensor({1, 3, img_size, img_size}, dtype(), device()), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    expectDType(input.grad().value());
}

// ============================================================================
// EfficientNet-B6 Multi-Backend Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB6ForwardShape) {
    auto model = efficientnet_b6(1000, false);
    convert_model_with_offload(model);

    Variable input(Tensor({1, 3, 528, 528}, dtype(), device()), true);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB6CustomClasses) {
    auto model = efficientnet_b6(200, false);
    convert_model_with_offload(model);

    Variable input(Tensor({1, 3, 528, 528}, dtype(), device()), true);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 200});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// EfficientNet-B7 Multi-Backend Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB7ForwardShape) {
    auto model = efficientnet_b7(1000, false);
    convert_model_with_offload(model);

    int64_t img_size = getInputSize(600);
    Variable input(Tensor({1, 3, img_size, img_size}, dtype(), device()), true);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB7GradientFlow) {
    auto model = efficientnet_b7(10, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(600);
    Variable input(Tensor({1, 3, img_size, img_size}, dtype(), device()), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    expectDType(input.grad().value());
}

TEST_P(EfficientNetMultiDTypeTest, EfficientNetB7BatchProcessing) {
    auto model = efficientnet_b7(10, false);
    convert_model_with_offload(model);

    int64_t img_size = getInputSize(600);
    // Use batch size 1 for GPU with Float64 (memory constraints)
    int batch_size = (device().type != Device::Type::CPU && dtype() == DType::Float64) ? 1 : 2;
    Variable input(Tensor({batch_size, 3, img_size, img_size}, dtype(), device()), true);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {batch_size, 10});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// Compound Scaling Multi-Backend Multi-DType Tests
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
    convert_model_with_offload(model_b0);
    convert_model_with_offload(model_b3);

    auto params_b0 = model_b0->parameters();
    auto params_b3 = model_b3->parameters();

    // B3 should have more parameters than B0
    EXPECT_GT(params_b3.size(), params_b0.size());
}

// ============================================================================
// Different Image Sizes Multi-Backend Multi-DType Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, DifferentImageSizesB0) {
    auto model = efficientnet_b0(10, false);
    convert_model_with_offload(model);

    // Test with various image sizes
    std::vector<int64_t> sizes = {224, 256, 384};

    for (int64_t size : sizes) {
        Variable input(Tensor({1, 3, size, size}, dtype(), device()), true);
        Variable output = model->forward(input);

        expectShape(output.tensor(), {1, 10});
        expectDType(output.tensor());
        expectFiniteNonZero(output.tensor());
    }
}

TEST_P(EfficientNetMultiDTypeTest, DifferentImageSizesB4) {
    auto model = efficientnet_b4(10, false);
    convert_model_with_offload(model);

    // Test with various image sizes (larger for B4)
    std::vector<int64_t> sizes = {320, 380, 512};

    for (int64_t size : sizes) {
        Variable input(Tensor({1, 3, size, size}, dtype(), device()), true);
        Variable output = model->forward(input);

        expectShape(output.tensor(), {1, 10});
        expectDType(output.tensor());
        expectFiniteNonZero(output.tensor());
    }
}

TEST_P(EfficientNetMultiDTypeTest, NonSquareInputImages) {
    auto model = efficientnet_b0(10, false);
    convert_model_with_offload(model);

    // Test with non-square inputs (should still work due to adaptive pooling)
    Variable input_rect1(Tensor({1, 3, 224, 256}, dtype(), device()), true);
    Variable output_rect1 = model->forward(input_rect1);
    expectShape(output_rect1.tensor(), {1, 10});
    expectFiniteNonZero(output_rect1.tensor());

    Variable input_rect2(Tensor({1, 3, 256, 224}, dtype(), device()), true);
    Variable output_rect2 = model->forward(input_rect2);
    expectShape(output_rect2.tensor(), {1, 10});
    expectFiniteNonZero(output_rect2.tensor());
}

// ============================================================================
// Cross-Variant Comparison Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, VariantResolutionProgression) {
    // Verify that higher variants can handle larger resolutions
    auto b0 = efficientnet_b0(10, false);
    auto b3 = efficientnet_b3(10, false);
    auto b7 = efficientnet_b7(10, false);
    convert_model_with_offload(b0);
    convert_model_with_offload(b3);
    convert_model_with_offload(b7);

    int64_t size_b0 = getInputSize(224);
    Variable input_b0(Tensor({1, 3, size_b0, size_b0}, dtype(), device()), true);
    Variable output_b0 = b0->forward(input_b0);
    expectShape(output_b0.tensor(), {1, 10});
    expectFiniteNonZero(output_b0.tensor());

    int64_t size_b3 = getInputSize(300);
    Variable input_b3(Tensor({1, 3, size_b3, size_b3}, dtype(), device()), true);
    Variable output_b3 = b3->forward(input_b3);
    expectShape(output_b3.tensor(), {1, 10});
    expectFiniteNonZero(output_b3.tensor());

    int64_t size_b7 = getInputSize(600);
    Variable input_b7(Tensor({1, 3, size_b7, size_b7}, dtype(), device()), true);
    Variable output_b7 = b7->forward(input_b7);
    expectShape(output_b7.tensor(), {1, 10});
    expectFiniteNonZero(output_b7.tensor());
}

TEST_P(EfficientNetMultiDTypeTest, VariantParameterScaling) {
    // Verify parameter count increases across variants
    std::vector<std::shared_ptr<EfficientNet>> models = {
        efficientnet_b0(10, false),
        efficientnet_b1(10, false),
        efficientnet_b2(10, false)
    };

    for (auto& model : models) {
        convert_model_with_offload(model);
    }

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

    for (auto& model : models) {
        convert_model_with_offload(model);
    }

    std::vector<int64_t> resolutions = {224, 260, 380};

    for (size_t i = 0; i < models.size(); ++i) {
        Variable input(Tensor({1, 3, resolutions[i], resolutions[i]}, dtype(), device()), true);
        Variable output = models[i]->forward(input);
        expectShape(output.tensor(), {1, 10});
        expectDType(output.tensor());
    }
}

TEST_P(EfficientNetMultiDTypeTest, LargeBatchSize) {
    auto model = efficientnet_b0(10, false);
    convert_model_with_offload(model);

    // Test with larger batch size
    Variable input(Tensor({16, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {16, 10});
    expectDType(output.tensor());
}

TEST_P(EfficientNetMultiDTypeTest, SingleClassOutput) {
    // Test with single class (binary classification scenario)
    auto model = efficientnet_b0(1, false);
    convert_model_with_offload(model);

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1});
    expectDType(output.tensor());
}

TEST_P(EfficientNetMultiDTypeTest, ManyClassesOutput) {
    // Test with many classes (fine-grained classification)
    auto model = efficientnet_b0(10000, false);
    convert_model_with_offload(model);

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 10000});
    expectDType(output.tensor());
}

// ============================================================================
// Training vs Evaluation Mode Tests
// ============================================================================

TEST_P(EfficientNetMultiDTypeTest, TrainingVsEvalMode) {
    auto model = efficientnet_b0(10, false);
    convert_model_with_offload(model);
    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), true);

    // Test in evaluation mode
    model->eval();
    Variable output_eval = model->forward(input);
    expectShape(output_eval.tensor(), {2, 10});

    // Test in training mode
    model->train();
    Variable output_train = model->forward(input);
    expectShape(output_train.tensor(), {2, 10});

    // Both should produce same shape
    auto shape_eval = output_eval.tensor().shape();
    auto shape_train = output_train.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_eval.begin(), shape_eval.end()),
              std::vector<int64_t>(shape_train.begin(), shape_train.end()));
}

// ============================================================================
// Instantiate Tests for All Backends and DTypes
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(EfficientNetMultiDTypeTest);

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
