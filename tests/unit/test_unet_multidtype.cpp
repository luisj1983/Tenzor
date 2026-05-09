/**
 * @file test_unet_multidtype.cpp
 * @brief Multi-dtype tests for U-Net segmentation model
 *
 * Tests U-Net architecture with Float32, Float64, and Float16 support across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends:
 * - U-Net construction with different depths
 * - Encoder-decoder architecture validation
 * - Skip connections functionality
 * - Forward pass with different image sizes
 * - Output channel configurations
 * - Bilinear vs transposed conv upsampling
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/unet.hpp"
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::models;

// ============================================================================
// U-Net Multi-Backend Multi-DType Test Fixture
// ============================================================================

class UNetMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Helper to count total parameters in a model
    size_t countModelParameters(const std::shared_ptr<UNet>& model) {
        auto params = model->parameters();
        return countParameters(params);
    }

    // Helper to check for finite values in output
    bool checkFiniteValues(const Variable& var, size_t num_samples = 1000) {
        auto tensor_cpu = var.tensor().to(Device::cpu());
        // Convert any non-Float32 type to Float32 for data access
        if (tensor_cpu.dtype() != DType::Float32) {
            tensor_cpu = tensor_cpu.to(DType::Float32);
        }
        auto data = tensor_cpu.data<float>();
        size_t check_count = std::min(num_samples, static_cast<size_t>(tensor_cpu.numel()));
        for (size_t i = 0; i < check_count; ++i) {
            if (!std::isfinite(data[i])) {
                return false;
            }
        }
        return true;
    }
};

// ============================================================================
// U-Net Construction Tests
// ============================================================================

TEST_P(UNetMultiDTypeTest, BasicConstruction) {
    // Test basic U-Net construction
    auto model = std::make_shared<UNet>(3, 21, false);
    convert_model(model);
    EXPECT_NE(model, nullptr);

    // Check parameters exist
    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(UNetMultiDTypeTest, DifferentConfigurations) {
    // Test different configurations
    auto model_binary = std::make_shared<UNet>(3, 1, false);
    convert_model(model_binary);
    EXPECT_NE(model_binary, nullptr);

    auto model_grayscale = std::make_shared<UNet>(1, 21, false);
    convert_model(model_grayscale);
    EXPECT_NE(model_grayscale, nullptr);

    auto model_bilinear = std::make_shared<UNet>(3, 21, true);
    convert_model(model_bilinear);
    EXPECT_NE(model_bilinear, nullptr);
}

TEST_P(UNetMultiDTypeTest, ParameterCount) {
    // U-Net with transposed conv should have more parameters
    auto model_transposed = std::make_shared<UNet>(3, 21, false);
    size_t total_transposed = countModelParameters(model_transposed);

    // U-Net with bilinear should have fewer parameters
    auto model_bilinear = std::make_shared<UNet>(3, 21, true);
    size_t total_bilinear = countModelParameters(model_bilinear);

    // U-Net should have around 31M parameters for transposed conv
    EXPECT_GT(total_transposed, 20'000'000);
    EXPECT_LT(total_transposed, 45'000'000);

    // Bilinear should have fewer parameters (no learned upsampling)
    EXPECT_LT(total_bilinear, total_transposed);
}

TEST_P(UNetMultiDTypeTest, DifferentChannelConfigurations) {
    // Test various input/output channel combinations
    std::vector<std::pair<int64_t, int64_t>> configs = {
        {1, 1},    // Binary grayscale
        {1, 2},    // Two-class grayscale
        {3, 1},    // Binary RGB
        {3, 2},    // Two-class RGB
        {3, 21},   // Pascal VOC
        {4, 10},   // RGBD to 10 classes
    };

    for (const auto& [in_ch, out_ch] : configs) {
        auto model = std::make_shared<UNet>(in_ch, out_ch, false);
        EXPECT_NE(model, nullptr);
        EXPECT_EQ(model->get_in_channels(), in_ch);
        EXPECT_EQ(model->get_num_classes(), out_ch);
    }
}

// ============================================================================
// Encoder-Decoder Architecture Tests
// ============================================================================

TEST_P(UNetMultiDTypeTest, EncoderPathDimensionality) {
    auto model = std::make_shared<UNet>(3, 21, false);
    convert_model(model);

    Variable images = createInput({1, 3, 256, 256});
    Variable output = model->forward(images);

    // Output should match input spatial dimensions (encoder-decoder symmetry)
    expectShape(output.tensor(), {1, 21, 256, 256});
    expectDType(output.tensor());
}

TEST_P(UNetMultiDTypeTest, DecoderPathReconstruction) {
    // Test that decoder properly reconstructs spatial dimensions
    std::vector<int64_t> sizes = {64, 128, 256};

    for (int64_t size : sizes) {
        auto model = std::make_shared<UNet>(3, 21, false);
        convert_model(model);

        Variable images = createInput({1, 3, size, size});
        Variable output = model->forward(images);

        auto shape = output.tensor().shape();
        EXPECT_EQ(shape[2], size) << "Height not preserved at size " << size;
        EXPECT_EQ(shape[3], size) << "Width not preserved at size " << size;
        expectDType(output.tensor());
    }
}

TEST_P(UNetMultiDTypeTest, BilinearVsTransposedConv) {
    auto model_transposed = std::make_shared<UNet>(3, 21, false);
    auto model_bilinear = std::make_shared<UNet>(3, 21, true);
    convert_model(model_transposed);
    convert_model(model_bilinear);

    Variable images = createInput({1, 3, 256, 256});

    // Both should produce same output shape
    Variable output_transposed = model_transposed->forward(images);
    Variable output_bilinear = model_bilinear->forward(images);

    auto shape_t = output_transposed.tensor().shape();
    auto shape_b = output_bilinear.tensor().shape();

    EXPECT_EQ(shape_t[0], shape_b[0]);
    EXPECT_EQ(shape_t[1], shape_b[1]);
    EXPECT_EQ(shape_t[2], shape_b[2]);
    EXPECT_EQ(shape_t[3], shape_b[3]);

    // Check configuration
    EXPECT_FALSE(model_transposed->is_bilinear());
    EXPECT_TRUE(model_bilinear->is_bilinear());
}

// ============================================================================
// Skip Connection Tests
// ============================================================================

TEST_P(UNetMultiDTypeTest, GradientFlowThroughSkips) {
    auto model = std::make_shared<UNet>(3, 21, false);
    convert_model(model);
    model->train();

    Variable images = createInput({1, 3, 256, 256}, true);
    Variable output = model->forward(images);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Input should have gradients (gradient flows through skip connections)
    EXPECT_TRUE(images.grad().has_value());

    // All parameters should have gradients
    auto params = model->parameters();
    for (const auto& p : params) {
        EXPECT_TRUE(p->grad().has_value()) << "Parameter missing gradient";
    }
}

TEST_P(UNetMultiDTypeTest, SkipConnectionFeaturePreservation) {
    // Skip connections should help preserve fine details
    auto model = std::make_shared<UNet>(3, 1, false);
    convert_model(model);
    model->eval();

    Variable images = createInput({1, 3, 256, 256}, false);
    Variable output = model->forward(images);

    // Output should have finite values (skip connections prevent vanishing gradients)
    EXPECT_TRUE(checkFiniteValues(output)) << "Output contains non-finite values";
}

// ============================================================================
// Forward Pass with Different Image Sizes
// ============================================================================

TEST_P(UNetMultiDTypeTest, MultipleImageSizes) {
    auto model = std::make_shared<UNet>(3, 21, false);
    convert_model(model);

    // Test with various image sizes
    std::vector<int64_t> sizes = {64, 128, 256};

    for (int64_t size : sizes) {
        Variable images = createInput({1, 3, size, size});
        Variable output = model->forward(images);

        expectShape(output.tensor(), {1, 21, size, size});
        expectDType(output.tensor());
    }
}

TEST_P(UNetMultiDTypeTest, NonSquareImages) {
    auto model = std::make_shared<UNet>(3, 21, false);
    convert_model(model);

    // Test with non-square images
    std::vector<std::pair<int64_t, int64_t>> sizes = {
        {128, 256},
        {256, 128},
        {192, 256}
    };

    for (const auto& [h, w] : sizes) {
        Variable images = createInput({1, 3, h, w});
        Variable output = model->forward(images);

        auto shape = output.tensor().shape();
        EXPECT_EQ(shape[0], 1);
        EXPECT_EQ(shape[1], 21);
        EXPECT_EQ(shape[2], h) << "Height not preserved for " << h << "x" << w;
        EXPECT_EQ(shape[3], w) << "Width not preserved for " << h << "x" << w;
    }
}

TEST_P(UNetMultiDTypeTest, BatchSizeVariation) {
    auto model = std::make_shared<UNet>(3, 21, false);
    convert_model(model);

    // U-Net keeps full-resolution feature maps live across the encoder/
    // decoder skip connections, so the activation footprint scales as
    // batch * H * W * channels * num_skips. On 8 GB Vulkan/CUDA devices the
    // 256x256 setup overflows even at batch=1 once Float32 activations are
    // doubled by the autograd graph. Halve the spatial dims on memory-
    // constrained backends; the kernel coverage is identical.
    bool tight = (backend_name() == "vulkan" || backend_name() == "cuda");
    int64_t hw = tight ? 128 : 256;
    std::vector<int64_t> batch_sizes = {1, 2, 4};

    for (int64_t batch : batch_sizes) {
        Variable images = createInput({batch, 3, hw, hw});
        Variable output = model->forward(images);

        auto shape = output.tensor().shape();
        EXPECT_EQ(shape[0], batch) << "Batch size not preserved";
        EXPECT_EQ(shape[1], 21);
        EXPECT_EQ(shape[2], hw);
        EXPECT_EQ(shape[3], hw);
    }
}

TEST_P(UNetMultiDTypeTest, SmallImageSizes) {
    auto model = std::make_shared<UNet>(3, 21, false);
    convert_model(model);

    // Test with small images (edge case)
    // U-Net has 4 downsampling layers, so minimum size is 16x16
    std::vector<int64_t> sizes = {16, 32, 48, 64};

    for (int64_t size : sizes) {
        Variable images = createInput({1, 3, size, size});
        Variable output = model->forward(images);

        auto shape = output.tensor().shape();
        EXPECT_EQ(shape[2], size) << "Small image size " << size << " not handled correctly";
        EXPECT_EQ(shape[3], size);
    }
}

// ============================================================================
// Output Channel Configuration Tests
// ============================================================================

TEST_P(UNetMultiDTypeTest, BinarySegmentation) {
    // Binary segmentation (1 output channel)
    auto model = std::make_shared<UNet>(3, 1, false);
    convert_model(model);

    Variable images = createInput({2, 3, 256, 256});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {2, 1, 256, 256});
    expectDType(output.tensor());
}

TEST_P(UNetMultiDTypeTest, MultiClassSegmentation) {
    // Iterates 5 class-count variants; on memory-constrained backends the
    // residual graph from one iteration overlaps the model construction of
    // the next, so even 256x256 at batch=1 OOMs on 8 GB Vulkan. See
    // BatchSizeVariation above for the rationale on shrinking H/W rather
    // than skipping the test.
    bool tight = (backend_name() == "vulkan" || backend_name() == "cuda");
    int64_t hw = tight ? 128 : 256;
    std::vector<int64_t> num_classes = {2, 5, 10, 21, 50};

    for (int64_t classes : num_classes) {
        auto model = std::make_shared<UNet>(3, classes, false);
        convert_model(model);

        Variable images = createInput({1, 3, hw, hw});
        Variable output = model->forward(images);

        auto shape = output.tensor().shape();
        EXPECT_EQ(shape[1], classes) << "Wrong number of output classes for " << classes;
    }
}

TEST_P(UNetMultiDTypeTest, GrayscaleInput) {
    // Test with grayscale input (1 channel)
    auto model = std::make_shared<UNet>(1, 21, false);
    convert_model(model);

    Variable images = createInput({1, 1, 256, 256});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {1, 21, 256, 256});
    expectDType(output.tensor());
}

TEST_P(UNetMultiDTypeTest, MultiChannelInput) {
    // Test with multi-channel inputs (e.g., RGBD, multispectral)
    std::vector<int64_t> input_channels = {1, 3, 4, 6};

    for (int64_t in_ch : input_channels) {
        auto model = std::make_shared<UNet>(in_ch, 21, false);
        convert_model(model);

        Variable images = createInput({1, in_ch, 256, 256});
        Variable output = model->forward(images);

        auto shape = output.tensor().shape();
        EXPECT_EQ(shape[0], 1);
        EXPECT_EQ(shape[1], 21);
        EXPECT_EQ(shape[2], 256);
        EXPECT_EQ(shape[3], 256);
    }
}

// ============================================================================
// Upsampling Method Tests
// ============================================================================

TEST_P(UNetMultiDTypeTest, BilinearUpsampling) {
    auto model = std::make_shared<UNet>(3, 21, true);  // bilinear=true
    convert_model(model);
    EXPECT_TRUE(model->is_bilinear());

    Variable images = createInput({2, 3, 256, 256});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {2, 21, 256, 256});
    expectDType(output.tensor());
}

TEST_P(UNetMultiDTypeTest, TransposedConvUpsampling) {
    auto model = std::make_shared<UNet>(3, 21, false);  // bilinear=false
    convert_model(model);
    EXPECT_FALSE(model->is_bilinear());

    Variable images = createInput({2, 3, 256, 256});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {2, 21, 256, 256});
    expectDType(output.tensor());
}

TEST_P(UNetMultiDTypeTest, UpsamplingMethodComparison) {
    // Both methods should produce outputs of same shape
    auto model_bilinear = std::make_shared<UNet>(3, 1, true);
    auto model_transposed = std::make_shared<UNet>(3, 1, false);
    convert_model(model_bilinear);
    convert_model(model_transposed);

    Variable images = createInput({1, 3, 128, 128}, false);

    model_bilinear->eval();
    model_transposed->eval();

    Variable output_bilinear = model_bilinear->forward(images);
    Variable output_transposed = model_transposed->forward(images);

    auto shape_b = output_bilinear.tensor().shape();
    auto shape_t = output_transposed.tensor().shape();

    // Shapes should match
    EXPECT_EQ(shape_b[0], shape_t[0]);
    EXPECT_EQ(shape_b[1], shape_t[1]);
    EXPECT_EQ(shape_b[2], shape_t[2]);
    EXPECT_EQ(shape_b[3], shape_t[3]);
}

TEST_P(UNetMultiDTypeTest, UpsamplingWithGradients) {
    // Test gradient flow through different upsampling methods
    auto model_bilinear = std::make_shared<UNet>(3, 1, true);
    auto model_transposed = std::make_shared<UNet>(3, 1, false);
    convert_model(model_bilinear);
    convert_model(model_transposed);

    model_bilinear->train();
    model_transposed->train();

    Variable images_b = createInput({1, 3, 128, 128}, true);
    Variable images_t = createInput({1, 3, 128, 128}, true);

    Variable output_b = model_bilinear->forward(images_b);
    Variable output_t = model_transposed->forward(images_t);

    Variable loss_b = tenzor::sum(output_b);
    Variable loss_t = tenzor::sum(output_t);

    loss_b.backward();
    loss_t.backward();

    // Both should propagate gradients
    EXPECT_TRUE(images_b.grad().has_value());
    EXPECT_TRUE(images_t.grad().has_value());
}

// ============================================================================
// Training and Inference Mode Tests
// ============================================================================

TEST_P(UNetMultiDTypeTest, TrainingMode) {
    auto model = std::make_shared<UNet>(3, 21, false);
    convert_model(model);
    model->train();

    Variable images = createInput({1, 3, 256, 256}, true);
    Variable output = model->forward(images);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Should compute gradients in training mode
    EXPECT_TRUE(images.grad().has_value());

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(UNetMultiDTypeTest, InferenceMode) {
    auto model = std::make_shared<UNet>(3, 21, false);
    convert_model(model);
    model->eval();

    Variable images = createInput({1, 3, 256, 256}, false);
    Variable output = model->forward(images);

    // Output should be valid
    expectShape(output.tensor(), {1, 21, 256, 256});
    expectDType(output.tensor());
}

TEST_P(UNetMultiDTypeTest, ModeToggling) {
    auto model = std::make_shared<UNet>(3, 1, false);
    convert_model(model);

    // Start in eval mode
    model->eval();
    Variable images_eval = createInput({1, 3, 128, 128}, false);
    Variable output_eval = model->forward(images_eval);

    // Switch to train mode
    model->train();
    Variable images_train = createInput({1, 3, 128, 128}, true);
    Variable output_train = model->forward(images_train);
    Variable loss = tenzor::sum(output_train);
    loss.backward();

    // Should have gradients in train mode
    EXPECT_TRUE(images_train.grad().has_value());
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_P(UNetMultiDTypeTest, MinimalImageSize) {
    // U-Net has 4 downsamplings (16x reduction), minimum size is 16x16
    auto model = std::make_shared<UNet>(3, 21, false);
    convert_model(model);

    Variable images = createInput({1, 3, 16, 16});
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[2], 16);
    EXPECT_EQ(shape[3], 16);
}

TEST_P(UNetMultiDTypeTest, SingleClassSegmentation) {
    // Single output class (binary segmentation)
    auto model = std::make_shared<UNet>(3, 1, false);
    convert_model(model);

    Variable images = createInput({1, 3, 256, 256});
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[1], 1);
}

TEST_P(UNetMultiDTypeTest, LargeBatchSize) {
    // Test with larger batch size
    auto model = std::make_shared<UNet>(3, 21, true);  // Use bilinear for memory
    convert_model(model);

    Variable images = createInput({8, 3, 128, 128});
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], 8);
    EXPECT_EQ(shape[1], 21);
    EXPECT_EQ(shape[2], 128);
    EXPECT_EQ(shape[3], 128);
}

TEST_P(UNetMultiDTypeTest, OutputNumericalStability) {
    auto model = std::make_shared<UNet>(3, 1, false);
    convert_model(model);
    model->eval();

    Variable images = createInput({1, 3, 128, 128}, false);
    Variable output = model->forward(images);

    // Output should have finite values
    EXPECT_TRUE(checkFiniteValues(output)) << "Output contains non-finite values";
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(UNetMultiDTypeTest);

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
