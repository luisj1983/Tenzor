/**
 * @file test_deeplabv3plus_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for DeepLab v3+ segmentation model
 *
 * Tests DeepLabV3+ across multiple backends (CPU, CUDA, OneAPI) and
 * data types (Float32, Float64, Float16). DeepLabV3+ is a state-of-the-art
 * semantic segmentation model with:
 * - ASPP (Atrous Spatial Pyramid Pooling) module
 * - Decoder with skip connections
 * - Support for different backbone architectures (ResNet50, ResNet101, MobileNetV2)
 * - Multi-scale feature extraction
 * - Variable input sizes
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/deeplabv3plus.hpp"
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::models;
using namespace tenzor::testing;

// ============================================================================
// Test Fixture with Multi-Backend Multi-DType Support
// ============================================================================

class DeepLabV3PlusMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// ResNet50 Backbone Tests
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, ResNet50ForwardShape) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model(model);

    auto images = createInput({2, 3, 512, 512});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {2, 21, 512, 512});
    expectDType(output.tensor());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, ResNet50GradientFlow) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model(model);
    model->train();

    auto images = createInput({1, 3, 512, 512});
    Variable output = model->forward(images);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(images.grad().has_value())
        << "Gradient not computed on " << backend_name();
    auto params = model->parameters();
    EXPECT_GT(params.size(), 0)
        << "No parameters found on " << backend_name();

    // Verify gradient dtype matches
    EXPECT_EQ(images.grad()->dtype(), dtype())
        << "Gradient dtype mismatch on " << backend_name();
}

TEST_P(DeepLabV3PlusMultiDTypeTest, ResNet50SmallBatchForward) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model(model);

    auto images = createInput({1, 3, 512, 512});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {1, 21, 512, 512});
}

// ============================================================================
// ResNet101 Backbone Tests
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, ResNet101ForwardShape) {
    auto model = DeepLabV3Plus_ResNet101(21, 16, false);
    convert_model(model);

    auto images = createInput({1, 3, 512, 512});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {1, 21, 512, 512});
    expectDType(output.tensor());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, ResNet101GradientFlow) {
    auto model = DeepLabV3Plus_ResNet101(21, 16, false);
    convert_model(model);
    model->train();

    auto images = createInput({1, 3, 512, 512});
    Variable output = model->forward(images);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(images.grad().has_value())
        << "Gradient not computed on " << backend_name();
    EXPECT_EQ(images.grad()->dtype(), dtype())
        << "Gradient dtype mismatch on " << backend_name();
}

TEST_P(DeepLabV3PlusMultiDTypeTest, ResNet101BatchProcessing) {
    auto model = DeepLabV3Plus_ResNet101(21, 16, false);
    convert_model(model);

    auto images = createInput({4, 3, 512, 512});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {4, 21, 512, 512});
}

// ============================================================================
// MobileNetV2 Backbone Tests
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, MobileNetForwardShape) {
    auto model = DeepLabV3Plus_MobileNetV2(21, 16, false);
    convert_model(model);

    auto images = createInput({2, 3, 512, 512});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {2, 21, 512, 512});
    expectDType(output.tensor());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, MobileNetGradientFlow) {
    auto model = DeepLabV3Plus_MobileNetV2(21, 16, false);
    convert_model(model);
    model->train();

    auto images = createInput({1, 3, 512, 512});
    Variable output = model->forward(images);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(images.grad().has_value())
        << "Gradient not computed on " << backend_name();
    EXPECT_EQ(images.grad()->dtype(), dtype())
        << "Gradient dtype mismatch on " << backend_name();
}

// ============================================================================
// Multi-Scale Feature Extraction Tests
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, DifferentInputSizes) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model(model);

    // Test with 256x256
    auto images_256 = createInput({1, 3, 256, 256});
    Variable output_256 = model->forward(images_256);
    expectShape(output_256.tensor(), {1, 21, 256, 256});

    // Test with 1024x1024
    auto images_1024 = createInput({1, 3, 1024, 1024});
    Variable output_1024 = model->forward(images_1024);
    expectShape(output_1024.tensor(), {1, 21, 1024, 1024});

    // Verify dtype preservation
    EXPECT_EQ(output_256.tensor().dtype(), dtype());
    EXPECT_EQ(output_1024.tensor().dtype(), dtype());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, NonSquareInputs) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model(model);

    // Test with rectangular input
    auto images_rect = createInput({1, 3, 384, 512});
    Variable output_rect = model->forward(images_rect);
    expectShape(output_rect.tensor(), {1, 21, 384, 512});
}

TEST_P(DeepLabV3PlusMultiDTypeTest, SmallInputSize) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model(model);

    // Test with small input (128x128)
    auto images_small = createInput({1, 3, 128, 128});
    Variable output_small = model->forward(images_small);
    expectShape(output_small.tensor(), {1, 21, 128, 128});
}

// ============================================================================
// ASPP Module Tests (Atrous Spatial Pyramid Pooling)
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, ASPPFeatureExtraction) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model(model);
    model->eval();

    auto images = createInput({1, 3, 512, 512});
    Variable output = model->forward(images);

    // ASPP should preserve spatial dimensions
    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[2], 512) << "Height not preserved through ASPP on " << backend_name();
    EXPECT_EQ(shape[3], 512) << "Width not preserved through ASPP on " << backend_name();
}

TEST_P(DeepLabV3PlusMultiDTypeTest, ASPPWithDilation) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model(model);

    // Test that ASPP handles different input sizes (tests dilation rates)
    auto images_large = createInput({1, 3, 768, 768});
    Variable output_large = model->forward(images_large);

    EXPECT_EQ(output_large.tensor().dtype(), dtype())
        << "ASPP dtype preservation failed on " << backend_name();
}

// ============================================================================
// Decoder with Skip Connections Tests
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, DecoderSkipConnections) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model(model);
    model->train();

    auto images = createInput({1, 3, 512, 512});
    Variable output = model->forward(images);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Skip connections should allow gradient flow
    EXPECT_TRUE(images.grad().has_value())
        << "Skip connection gradient flow failed on " << backend_name();
}

TEST_P(DeepLabV3PlusMultiDTypeTest, DecoderOutputResolution) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model(model);

    auto images = createInput({1, 3, 512, 512});
    Variable output = model->forward(images);

    // Decoder should restore full input resolution
    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[2], 512) << "Decoder height restoration failed on " << backend_name();
    EXPECT_EQ(shape[3], 512) << "Decoder width restoration failed on " << backend_name();
}

// ============================================================================
// Parameter Count and Model Structure Tests
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, ResNet50ParameterCount) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    auto params = model->parameters();

    size_t total_params = countParameters(params);

    // DeepLabV3+ ResNet50 should have ~40M parameters (allow 30% tolerance)
    EXPECT_GT(total_params, 30'000'000)
        << "Too few parameters on " << backend_name();
    EXPECT_LT(total_params, 55'000'000)
        << "Too many parameters on " << backend_name();
}

TEST_P(DeepLabV3PlusMultiDTypeTest, MobileNetParameterCount) {
    auto model = DeepLabV3Plus_MobileNetV2(21, 16, false);
    auto params = model->parameters();

    size_t total_params = countParameters(params);

    // MobileNetV2 backbone should have fewer parameters than ResNet50
    EXPECT_GT(total_params, 1'000'000)
        << "Too few parameters for MobileNet on " << backend_name();
    EXPECT_LT(total_params, 15'000'000)
        << "Too many parameters for MobileNet on " << backend_name();
}

// ============================================================================
// Binary and Multi-Class Segmentation Tests
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, BinarySegmentation) {
    auto model = DeepLabV3Plus_ResNet50(1, 16, false);
    convert_model(model);

    auto images = createInput({2, 3, 512, 512});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {2, 1, 512, 512});
    EXPECT_EQ(output.tensor().dtype(), dtype());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, MultiClassSegmentation) {
    // Test with large number of classes (e.g., COCO-style)
    auto model = DeepLabV3Plus_ResNet50(80, 16, false);
    convert_model(model);

    auto images = createInput({1, 3, 512, 512});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {1, 80, 512, 512});
}

TEST_P(DeepLabV3PlusMultiDTypeTest, FewClassSegmentation) {
    // Test with few classes
    auto model = DeepLabV3Plus_ResNet50(2, 16, false);
    convert_model(model);

    auto images = createInput({1, 3, 512, 512});
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[1], 2) << "Few-class segmentation failed on " << backend_name();
}

// ============================================================================
// Training and Evaluation Mode Tests
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, TrainEvalModeConsistency) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model(model);

    auto images = createInput({1, 3, 512, 512});

    // Test in evaluation mode
    model->eval();
    Variable output_eval = model->forward(images);

    // Test in training mode
    model->train();
    Variable output_train = model->forward(images);

    // Shape should be consistent
    auto shape_eval = output_eval.tensor().shape();
    auto shape_train = output_train.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_eval.begin(), shape_eval.end()),
              std::vector<int64_t>(shape_train.begin(), shape_train.end()))
        << "Shape inconsistency between train/eval modes on " << backend_name();
}

TEST_P(DeepLabV3PlusMultiDTypeTest, BatchNormInEvalMode) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model(model);
    model->eval();

    auto images = createInput({1, 3, 512, 512});
    Variable output = model->forward(images);

    // Should produce valid output in eval mode
    EXPECT_FALSE(output.tensor().shape().empty())
        << "Eval mode output invalid on " << backend_name();
}

// ============================================================================
// Backbone Architecture Comparison Tests
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, BackboneOutputConsistency) {
    auto model_resnet50 = DeepLabV3Plus_ResNet50(21, 16, false);
    auto model_resnet101 = DeepLabV3Plus_ResNet101(21, 16, false);
    auto model_mobilenet = DeepLabV3Plus_MobileNetV2(21, 16, false);

    convert_model(model_resnet50);
    convert_model(model_resnet101);
    convert_model(model_mobilenet);

    auto images = createInput({1, 3, 512, 512});

    Variable output_resnet50 = model_resnet50->forward(images);
    Variable output_resnet101 = model_resnet101->forward(images);
    Variable output_mobilenet = model_mobilenet->forward(images);

    // All backbones should produce same shape output
    auto shape_resnet50 = output_resnet50.tensor().shape();
    auto shape_resnet101 = output_resnet101.tensor().shape();
    auto shape_mobilenet = output_mobilenet.tensor().shape();

    EXPECT_EQ(std::vector<int64_t>(shape_resnet50.begin(), shape_resnet50.end()),
              std::vector<int64_t>(shape_resnet101.begin(), shape_resnet101.end()))
        << "ResNet50/101 shape mismatch on " << backend_name();
    EXPECT_EQ(std::vector<int64_t>(shape_resnet50.begin(), shape_resnet50.end()),
              std::vector<int64_t>(shape_mobilenet.begin(), shape_mobilenet.end()))
        << "ResNet/MobileNet shape mismatch on " << backend_name();

    // All should preserve dtype
    EXPECT_EQ(output_resnet50.tensor().dtype(), dtype());
    EXPECT_EQ(output_resnet101.tensor().dtype(), dtype());
    EXPECT_EQ(output_mobilenet.tensor().dtype(), dtype());
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, LargeBatchProcessing) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model(model);

    // Test with larger batch size
    auto images = createInput({8, 3, 256, 256});
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], 8) << "Large batch processing failed on " << backend_name();
    EXPECT_EQ(output.tensor().dtype(), dtype());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, MinimalInputSize) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model(model);

    // Test with minimal viable input (64x64)
    auto images = createInput({1, 3, 64, 64});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {1, 21, 64, 64});
}

TEST_P(DeepLabV3PlusMultiDTypeTest, VeryLargeInputSize) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model(model);

    // Test with very large input (may require significant memory)
    auto images = createInput({1, 3, 2048, 2048});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {1, 21, 2048, 2048});
}

TEST_P(DeepLabV3PlusMultiDTypeTest, SequentialForwardPasses) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model(model);

    // Multiple forward passes should be consistent
    auto images = createInput({1, 3, 512, 512});

    Variable output1 = model->forward(images);
    Variable output2 = model->forward(images);
    Variable output3 = model->forward(images);

    auto shape1 = output1.tensor().shape();
    auto shape2 = output2.tensor().shape();
    auto shape3 = output3.tensor().shape();

    EXPECT_EQ(std::vector<int64_t>(shape1.begin(), shape1.end()),
              std::vector<int64_t>(shape2.begin(), shape2.end()))
        << "Sequential forward pass 1-2 shape mismatch on " << backend_name();
    EXPECT_EQ(std::vector<int64_t>(shape2.begin(), shape2.end()),
              std::vector<int64_t>(shape3.begin(), shape3.end()))
        << "Sequential forward pass 2-3 shape mismatch on " << backend_name();
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DeepLabV3PlusMultiDTypeTest);

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();  // Initialize Tenzor library and backends
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
