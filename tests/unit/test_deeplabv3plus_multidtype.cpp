/**
 * @file test_deeplabv3plus_multidtype.cpp
 * @brief Multi-dtype tests for DeepLab v3+ segmentation model
 *
 * Tests DeepLabV3+ with Float32, Float64, and Float16 support.
 * DeepLabV3+ is a state-of-the-art semantic segmentation model with:
 * - ASPP (Atrous Spatial Pyramid Pooling) module
 * - Decoder with skip connections
 * - Support for different backbone architectures (ResNet50, ResNet101, MobileNetV2)
 * - Multi-scale feature extraction
 * - Variable input sizes
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/deeplabv3plus.hpp"

using namespace tenzor;
using namespace tenzor::models;

// ============================================================================
// Multi-DType Test Fixture
// ============================================================================

template<typename DTypeParam>
class DeepLabV3PlusMultiDTypeTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = Device::cpu();
        dtype_ = DTypeParam::dtype;
    }

    Device device_;
    DType dtype_;
};

// Define dtype parameter structs
struct Float32Type { static constexpr DType dtype = DType::Float32; static constexpr const char* name = "Float32"; };
struct Float64Type { static constexpr DType dtype = DType::Float64; static constexpr const char* name = "Float64"; };
struct Float16Type { static constexpr DType dtype = DType::Float16; static constexpr const char* name = "Float16"; };

using DTypeImplementations = ::testing::Types<Float32Type, Float64Type, Float16Type>;
TYPED_TEST_SUITE(DeepLabV3PlusMultiDTypeTest, DTypeImplementations);

// ============================================================================
// ResNet50 Backbone Tests
// ============================================================================

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, ResNet50ForwardShape) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    Variable images(Tensor({2, 3, 512, 512}, this->dtype_, this->device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 21, 512, 512}))
        << "Failed for dtype: " << TypeParam::name;
    EXPECT_EQ(output.tensor().dtype(), this->dtype_)
        << "Output dtype mismatch for: " << TypeParam::name;
}

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, ResNet50GradientFlow) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    model->train();

    Variable images(Tensor({1, 3, 512, 512}, this->dtype_, this->device_), true);
    Variable output = model->forward(images);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(images.grad().has_value())
        << "Gradient not computed for dtype: " << TypeParam::name;
    auto params = model->parameters();
    EXPECT_GT(params.size(), 0)
        << "No parameters found for dtype: " << TypeParam::name;

    // Verify gradient dtype matches
    EXPECT_EQ(images.grad()->dtype(), this->dtype_)
        << "Gradient dtype mismatch for: " << TypeParam::name;
}

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, ResNet50SmallBatchForward) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    Variable images(Tensor({1, 3, 512, 512}, this->dtype_, this->device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 21, 512, 512}))
        << "Failed for dtype: " << TypeParam::name;
}

// ============================================================================
// ResNet101 Backbone Tests
// ============================================================================

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, ResNet101ForwardShape) {
    auto model = DeepLabV3Plus_ResNet101(21, 16, false);
    Variable images(Tensor({1, 3, 512, 512}, this->dtype_, this->device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 21, 512, 512}))
        << "Failed for dtype: " << TypeParam::name;
    EXPECT_EQ(output.tensor().dtype(), this->dtype_)
        << "Output dtype mismatch for: " << TypeParam::name;
}

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, ResNet101GradientFlow) {
    auto model = DeepLabV3Plus_ResNet101(21, 16, false);
    model->train();

    Variable images(Tensor({1, 3, 512, 512}, this->dtype_, this->device_), true);
    Variable output = model->forward(images);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(images.grad().has_value())
        << "Gradient not computed for dtype: " << TypeParam::name;
    EXPECT_EQ(images.grad()->dtype(), this->dtype_)
        << "Gradient dtype mismatch for: " << TypeParam::name;
}

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, ResNet101BatchProcessing) {
    auto model = DeepLabV3Plus_ResNet101(21, 16, false);
    Variable images(Tensor({4, 3, 512, 512}, this->dtype_, this->device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{4, 21, 512, 512}))
        << "Batch processing failed for dtype: " << TypeParam::name;
}

// ============================================================================
// MobileNetV2 Backbone Tests
// ============================================================================

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, MobileNetForwardShape) {
    auto model = DeepLabV3Plus_MobileNetV2(21, 16, false);
    Variable images(Tensor({2, 3, 512, 512}, this->dtype_, this->device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 21, 512, 512}))
        << "Failed for dtype: " << TypeParam::name;
    EXPECT_EQ(output.tensor().dtype(), this->dtype_)
        << "Output dtype mismatch for: " << TypeParam::name;
}

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, MobileNetGradientFlow) {
    auto model = DeepLabV3Plus_MobileNetV2(21, 16, false);
    model->train();

    Variable images(Tensor({1, 3, 512, 512}, this->dtype_, this->device_), true);
    Variable output = model->forward(images);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(images.grad().has_value())
        << "Gradient not computed for dtype: " << TypeParam::name;
    EXPECT_EQ(images.grad()->dtype(), this->dtype_)
        << "Gradient dtype mismatch for: " << TypeParam::name;
}

// ============================================================================
// Multi-Scale Feature Extraction Tests
// ============================================================================

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, DifferentInputSizes) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);

    // Test with 256x256
    Variable images_256(Tensor({1, 3, 256, 256}, this->dtype_, this->device_), true);
    Variable output_256 = model->forward(images_256);
    auto shape_256 = output_256.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_256.begin(), shape_256.end()),
              (std::vector<int64_t>{1, 21, 256, 256}))
        << "256x256 input failed for dtype: " << TypeParam::name;

    // Test with 1024x1024
    Variable images_1024(Tensor({1, 3, 1024, 1024}, this->dtype_, this->device_), true);
    Variable output_1024 = model->forward(images_1024);
    auto shape_1024 = output_1024.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_1024.begin(), shape_1024.end()),
              (std::vector<int64_t>{1, 21, 1024, 1024}))
        << "1024x1024 input failed for dtype: " << TypeParam::name;

    // Verify dtype preservation
    EXPECT_EQ(output_256.tensor().dtype(), this->dtype_);
    EXPECT_EQ(output_1024.tensor().dtype(), this->dtype_);
}

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, NonSquareInputs) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);

    // Test with rectangular input
    Variable images_rect(Tensor({1, 3, 384, 512}, this->dtype_, this->device_), true);
    Variable output_rect = model->forward(images_rect);
    auto shape_rect = output_rect.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_rect.begin(), shape_rect.end()),
              (std::vector<int64_t>{1, 21, 384, 512}))
        << "Non-square input failed for dtype: " << TypeParam::name;
}

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, SmallInputSize) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);

    // Test with small input (128x128)
    Variable images_small(Tensor({1, 3, 128, 128}, this->dtype_, this->device_), true);
    Variable output_small = model->forward(images_small);
    auto shape_small = output_small.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_small.begin(), shape_small.end()),
              (std::vector<int64_t>{1, 21, 128, 128}))
        << "Small input size failed for dtype: " << TypeParam::name;
}

// ============================================================================
// ASPP Module Tests (Atrous Spatial Pyramid Pooling)
// ============================================================================

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, ASPPFeatureExtraction) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    model->eval();

    Variable images(Tensor({1, 3, 512, 512}, this->dtype_, this->device_), true);
    Variable output = model->forward(images);

    // ASPP should preserve spatial dimensions
    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[2], 512) << "Height not preserved through ASPP for dtype: " << TypeParam::name;
    EXPECT_EQ(shape[3], 512) << "Width not preserved through ASPP for dtype: " << TypeParam::name;
}

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, ASPPWithDilation) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);

    // Test that ASPP handles different input sizes (tests dilation rates)
    Variable images_large(Tensor({1, 3, 768, 768}, this->dtype_, this->device_), true);
    Variable output_large = model->forward(images_large);

    EXPECT_EQ(output_large.tensor().dtype(), this->dtype_)
        << "ASPP dtype preservation failed for: " << TypeParam::name;
}

// ============================================================================
// Decoder with Skip Connections Tests
// ============================================================================

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, DecoderSkipConnections) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    model->train();

    Variable images(Tensor({1, 3, 512, 512}, this->dtype_, this->device_), true);
    Variable output = model->forward(images);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Skip connections should allow gradient flow
    EXPECT_TRUE(images.grad().has_value())
        << "Skip connection gradient flow failed for dtype: " << TypeParam::name;
}

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, DecoderOutputResolution) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);

    Variable images(Tensor({1, 3, 512, 512}, this->dtype_, this->device_), true);
    Variable output = model->forward(images);

    // Decoder should restore full input resolution
    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[2], 512) << "Decoder height restoration failed for dtype: " << TypeParam::name;
    EXPECT_EQ(shape[3], 512) << "Decoder width restoration failed for dtype: " << TypeParam::name;
}

// ============================================================================
// Parameter Count and Model Structure Tests
// ============================================================================

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, ResNet50ParameterCount) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // DeepLabV3+ ResNet50 should have ~40M parameters (allow 30% tolerance)
    EXPECT_GT(total_params, 30'000'000)
        << "Too few parameters for dtype: " << TypeParam::name;
    EXPECT_LT(total_params, 55'000'000)
        << "Too many parameters for dtype: " << TypeParam::name;
}

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, MobileNetParameterCount) {
    auto model = DeepLabV3Plus_MobileNetV2(21, 16, false);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // MobileNetV2 backbone should have fewer parameters than ResNet50
    EXPECT_GT(total_params, 1'000'000)
        << "Too few parameters for MobileNet dtype: " << TypeParam::name;
    EXPECT_LT(total_params, 15'000'000)
        << "Too many parameters for MobileNet dtype: " << TypeParam::name;
}

// ============================================================================
// Binary and Multi-Class Segmentation Tests
// ============================================================================

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, BinarySegmentation) {
    auto model = DeepLabV3Plus_ResNet50(1, 16, false);
    Variable images(Tensor({2, 3, 512, 512}, this->dtype_, this->device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1, 512, 512}))
        << "Binary segmentation failed for dtype: " << TypeParam::name;
    EXPECT_EQ(output.tensor().dtype(), this->dtype_);
}

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, MultiClassSegmentation) {
    // Test with large number of classes (e.g., COCO-style)
    auto model = DeepLabV3Plus_ResNet50(80, 16, false);
    Variable images(Tensor({1, 3, 512, 512}, this->dtype_, this->device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 80, 512, 512}))
        << "Multi-class (80 classes) segmentation failed for dtype: " << TypeParam::name;
}

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, FewClassSegmentation) {
    // Test with few classes
    auto model = DeepLabV3Plus_ResNet50(2, 16, false);
    Variable images(Tensor({1, 3, 512, 512}, this->dtype_, this->device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[1], 2) << "Few-class segmentation failed for dtype: " << TypeParam::name;
}

// ============================================================================
// Training and Evaluation Mode Tests
// ============================================================================

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, TrainEvalModeConsistency) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    Variable images(Tensor({1, 3, 512, 512}, this->dtype_, this->device_), true);

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
        << "Shape inconsistency between train/eval modes for dtype: " << TypeParam::name;
}

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, BatchNormInEvalMode) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    model->eval();

    Variable images(Tensor({1, 3, 512, 512}, this->dtype_, this->device_), true);
    Variable output = model->forward(images);

    // Should produce valid output in eval mode
    EXPECT_FALSE(output.tensor().shape().empty())
        << "Eval mode output invalid for dtype: " << TypeParam::name;
}

// ============================================================================
// Backbone Architecture Comparison Tests
// ============================================================================

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, BackboneOutputConsistency) {
    auto model_resnet50 = DeepLabV3Plus_ResNet50(21, 16, false);
    auto model_resnet101 = DeepLabV3Plus_ResNet101(21, 16, false);
    auto model_mobilenet = DeepLabV3Plus_MobileNetV2(21, 16, false);

    Variable images(Tensor({1, 3, 512, 512}, this->dtype_, this->device_), true);

    Variable output_resnet50 = model_resnet50->forward(images);
    Variable output_resnet101 = model_resnet101->forward(images);
    Variable output_mobilenet = model_mobilenet->forward(images);

    // All backbones should produce same shape output
    auto shape_resnet50 = output_resnet50.tensor().shape();
    auto shape_resnet101 = output_resnet101.tensor().shape();
    auto shape_mobilenet = output_mobilenet.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_resnet50.begin(), shape_resnet50.end()),
              std::vector<int64_t>(shape_resnet101.begin(), shape_resnet101.end()))
        << "ResNet50/101 shape mismatch for dtype: " << TypeParam::name;
    EXPECT_EQ(std::vector<int64_t>(shape_resnet50.begin(), shape_resnet50.end()),
              std::vector<int64_t>(shape_mobilenet.begin(), shape_mobilenet.end()))
        << "ResNet/MobileNet shape mismatch for dtype: " << TypeParam::name;

    // All should preserve dtype
    EXPECT_EQ(output_resnet50.tensor().dtype(), this->dtype_);
    EXPECT_EQ(output_resnet101.tensor().dtype(), this->dtype_);
    EXPECT_EQ(output_mobilenet.tensor().dtype(), this->dtype_);
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, LargeBatchProcessing) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);

    // Test with larger batch size
    Variable images(Tensor({8, 3, 256, 256}, this->dtype_, this->device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], 8) << "Large batch processing failed for dtype: " << TypeParam::name;
    EXPECT_EQ(output.tensor().dtype(), this->dtype_);
}

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, MinimalInputSize) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);

    // Test with minimal viable input (64x64)
    Variable images(Tensor({1, 3, 64, 64}, this->dtype_, this->device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 21, 64, 64}))
        << "Minimal input size failed for dtype: " << TypeParam::name;
}

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, VeryLargeInputSize) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);

    // Test with very large input (may require significant memory)
    Variable images(Tensor({1, 3, 2048, 2048}, this->dtype_, this->device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 21, 2048, 2048}))
        << "Very large input size failed for dtype: " << TypeParam::name;
}

TYPED_TEST(DeepLabV3PlusMultiDTypeTest, SequentialForwardPasses) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);

    // Multiple forward passes should be consistent
    Variable images(Tensor({1, 3, 512, 512}, this->dtype_, this->device_), true);

    Variable output1 = model->forward(images);
    Variable output2 = model->forward(images);
    Variable output3 = model->forward(images);

    auto shape1 = output1.tensor().shape();
    auto shape2 = output2.tensor().shape();
    auto shape3 = output3.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape1.begin(), shape1.end()),
              std::vector<int64_t>(shape2.begin(), shape2.end()))
        << "Sequential forward pass 1-2 shape mismatch for dtype: " << TypeParam::name;
    EXPECT_EQ(std::vector<int64_t>(shape2.begin(), shape2.end()),
              std::vector<int64_t>(shape3.begin(), shape3.end()))
        << "Sequential forward pass 2-3 shape mismatch for dtype: " << TypeParam::name;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();  // Initialize Tenzor library and backends
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
