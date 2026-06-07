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
#include <tenzor/nn/offload.hpp>
#include <memory>
#include "../../include/tenzor/models/deeplabv3plus.hpp"
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::models;
using namespace tenzor::testing;

// ============================================================================
// Test Fixture with Multi-Backend Multi-DType Support
// ============================================================================

class DeepLabV3PlusMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    std::vector<std::unique_ptr<nn::OffloadContext>> offload_ctxs_;

    template <typename ModuleT>
    void convert_model_with_offload(ModuleT& model) {
        if (device().type == Device::Type::CPU) {
            convert_model(model);
        } else {
            // GPU backend: CPU-start offloading
            if constexpr (requires { model->to(dtype()); }) {
                model->to(dtype());
            } else if constexpr (requires { model.to(dtype()); }) {
                model.to(dtype());
            }

            nn::OffloadContext::Config config;
            config.offload_parameters = true;
            config.offload_gradients = true;
            config.prefetch_depth = 2;
            config.pin_first_layer = true;
            config.pin_last_layer = true;
            config.target_device = device();

            std::unique_ptr<nn::OffloadContext> ctx;
            if constexpr (requires { model.get(); }) {
                ctx = std::make_unique<nn::OffloadContext>(*model, config);
            } else if constexpr (requires { *model; }) {
                ctx = std::make_unique<nn::OffloadContext>(*model, config);
            } else {
                ctx = std::make_unique<nn::OffloadContext>(model, config);
            }
            ctx->enable();
            offload_ctxs_.push_back(std::move(ctx));
        }
    }

    template <typename ModuleT>
    void enable_offloading_if_needed(ModuleT& model) { (void)model; }

    /**
     * @brief Get appropriate input size for the current backend and dtype
     * GPU backends use smaller sizes for Float64 due to memory constraints
     * DeepLabV3Plus with ResNet is very memory-intensive
     */
    int64_t getInputSize(int64_t default_size) {
        if (device().type == Device::Type::CPU) {
            // Half-precision on CPU has dtype conversion overhead even with Float32 compute
            bool is_half = (dtype() == DType::Float16 || dtype() == DType::BFloat16);
            if (is_half) {
                if (default_size >= 1024) return 384;
                if (default_size >= 512) return 256;
                return std::min(default_size, int64_t(224));
            }
            return default_size;
        }

        bool is_float64 = (dtype() == DType::Float64);
        bool is_float16 = (dtype() == DType::Float16);

        if (is_float64) {
            // Float64: DeepLabV3Plus needs very small sizes
            if (default_size >= 1024) return 128;
            if (default_size >= 512) return 128;
            if (default_size >= 256) return 96;
            return std::min(default_size, int64_t(96));
        }

        if (is_float16) {
            // Float16: smaller to avoid numerical issues
            if (default_size >= 1024) return 256;
            if (default_size >= 512) return 192;
            if (default_size >= 256) return 160;
            return std::min(default_size, int64_t(128));
        }

        // Float32: moderate reduction for GPU memory
        if (default_size >= 1024) return 384;
        if (default_size >= 512) return 256;
        return std::min(default_size, int64_t(224));
    }
};

// ============================================================================
// ResNet50 Backbone Tests
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, ResNet50ForwardShape) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model_with_offload(model);

    int64_t img_size = getInputSize(512);
    auto images = createInput({2, 3, img_size, img_size});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {2, 21, img_size, img_size});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, ResNet50GradientFlow) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model_with_offload(model);
    model->train();

    auto images = createInput({1, 3, getInputSize(512), getInputSize(512)});
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
    convert_model_with_offload(model);

    int64_t img_size = getInputSize(512);
    auto images = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {1, 21, img_size, img_size});
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// ResNet101 Backbone Tests
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, ResNet101ForwardShape) {
    auto model = DeepLabV3Plus_ResNet101(21, 16, false);
    convert_model_with_offload(model);

    int64_t img_size = getInputSize(512);
    auto images = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {1, 21, img_size, img_size});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, ResNet101GradientFlow) {
    auto model = DeepLabV3Plus_ResNet101(21, 16, false);
    convert_model_with_offload(model);
    model->train();

    auto images = createInput({1, 3, getInputSize(512), getInputSize(512)});
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
    convert_model_with_offload(model);

    int64_t img_size = getInputSize(512);
    auto images = createInput({4, 3, img_size, img_size});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {4, 21, img_size, img_size});
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// MobileNetV2 Backbone Tests
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, MobileNetForwardShape) {
    auto model = DeepLabV3Plus_MobileNetV2(21, 16, false);
    convert_model_with_offload(model);

    int64_t img_size = getInputSize(512);
    auto images = createInput({2, 3, img_size, img_size});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {2, 21, img_size, img_size});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, MobileNetGradientFlow) {
    auto model = DeepLabV3Plus_MobileNetV2(21, 16, false);
    convert_model_with_offload(model);
    model->train();

    auto images = createInput({1, 3, getInputSize(512), getInputSize(512)});
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
    convert_model_with_offload(model);

    // Test with smaller size (256 or reduced for GPU/Float64)
    int64_t size_small = getInputSize(256);
    auto images_small = createInput({1, 3, size_small, size_small});
    Variable output_small = model->forward(images_small);
    expectShape(output_small.tensor(), {1, 21, size_small, size_small});

    // Test with larger size (512 or reduced for GPU/Float64)
    int64_t size_large = getInputSize(512);
    auto images_large = createInput({1, 3, size_large, size_large});
    Variable output_large = model->forward(images_large);
    expectShape(output_large.tensor(), {1, 21, size_large, size_large});

    // Verify dtype preservation
    EXPECT_EQ(output_small.tensor().dtype(), dtype());
    EXPECT_EQ(output_large.tensor().dtype(), dtype());
    expectFiniteNonZero(output_small.tensor());
    expectFiniteNonZero(output_large.tensor());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, NonSquareInputs) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model_with_offload(model);

    // Test with rectangular input
    auto images_rect = createInput({1, 3, 384, 512});
    Variable output_rect = model->forward(images_rect);
    expectShape(output_rect.tensor(), {1, 21, 384, 512});
    expectFiniteNonZero(output_rect.tensor());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, SmallInputSize) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model_with_offload(model);

    // Test with small input (128x128)
    auto images_small = createInput({1, 3, 128, 128});
    Variable output_small = model->forward(images_small);
    expectShape(output_small.tensor(), {1, 21, 128, 128});
    expectFiniteNonZero(output_small.tensor());
}

// ============================================================================
// ASPP Module Tests (Atrous Spatial Pyramid Pooling)
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, ASPPFeatureExtraction) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model_with_offload(model);
    model->eval();

    int64_t img_size = getInputSize(512);
    auto images = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(images);

    // ASPP should preserve spatial dimensions
    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[2], img_size) << "Height not preserved through ASPP on " << backend_name();
    EXPECT_EQ(shape[3], img_size) << "Width not preserved through ASPP on " << backend_name();
    expectFiniteNonZero(output.tensor());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, ASPPWithDilation) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model_with_offload(model);

    // Test that ASPP handles different input sizes (tests dilation rates)
    auto images_large = createInput({1, 3, 768, 768});
    Variable output_large = model->forward(images_large);

    EXPECT_EQ(output_large.tensor().dtype(), dtype())
        << "ASPP dtype preservation failed on " << backend_name();
    expectFiniteNonZero(output_large.tensor());
}

// ============================================================================
// Decoder with Skip Connections Tests
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, DecoderSkipConnections) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model_with_offload(model);
    model->train();

    auto images = createInput({1, 3, getInputSize(512), getInputSize(512)});
    Variable output = model->forward(images);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Skip connections should allow gradient flow
    EXPECT_TRUE(images.grad().has_value())
        << "Skip connection gradient flow failed on " << backend_name();
    expectFiniteNonZero(output.tensor());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, DecoderOutputResolution) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model_with_offload(model);

    int64_t img_size = getInputSize(512);
    auto images = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(images);

    // Decoder should restore full input resolution
    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[2], img_size) << "Decoder height restoration failed on " << backend_name();
    EXPECT_EQ(shape[3], img_size) << "Decoder width restoration failed on " << backend_name();
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// Parameter Count and Model Structure Tests
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, ResNet50ParameterCount) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    auto params = model->parameters();

    size_t total_params = countParameters(params);

    // DeepLabV3+ ResNet50 here uses depthwise-separable atrous convolutions in
    // the ASPP (constructed with use_separable=true — the canonical DeepLabV3+
    // efficiency design from the paper). The separable atrous convs are ~10x
    // smaller than full 3x3 atrous convs, so the total is ~29.5M parameters
    // rather than the ~40M a full-conv ASPP would give. The bounds below still
    // catch a backbone/ASPP/decoder that is missing or grossly mis-sized.
    EXPECT_GT(total_params, 28'000'000)
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
    convert_model_with_offload(model);

    int64_t img_size = getInputSize(512);
    auto images = createInput({2, 3, img_size, img_size});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {2, 1, img_size, img_size});
    expectFiniteNonZero(output.tensor());
    EXPECT_EQ(output.tensor().dtype(), dtype());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, MultiClassSegmentation) {
    // Test with large number of classes (e.g., COCO-style)
    auto model = DeepLabV3Plus_ResNet50(80, 16, false);
    convert_model_with_offload(model);

    int64_t img_size = getInputSize(512);
    auto images = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {1, 80, img_size, img_size});
    expectFiniteNonZero(output.tensor());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, FewClassSegmentation) {
    // Test with few classes
    auto model = DeepLabV3Plus_ResNet50(2, 16, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(512), getInputSize(512)});
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[1], 2) << "Few-class segmentation failed on " << backend_name();
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// Training and Evaluation Mode Tests
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, TrainEvalModeConsistency) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(512), getInputSize(512)});

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
    expectFiniteNonZero(output_eval.tensor());
    expectFiniteNonZero(output_train.tensor());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, BatchNormInEvalMode) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model_with_offload(model);
    model->eval();

    auto images = createInput({1, 3, getInputSize(512), getInputSize(512)});
    Variable output = model->forward(images);

    // Should produce valid output in eval mode
    EXPECT_FALSE(output.tensor().shape().empty())
        << "Eval mode output invalid on " << backend_name();
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// Backbone Architecture Comparison Tests
// ============================================================================

TEST_P(DeepLabV3PlusMultiDTypeTest, BackboneOutputConsistency) {
    auto model_resnet50 = DeepLabV3Plus_ResNet50(21, 16, false);
    auto model_resnet101 = DeepLabV3Plus_ResNet101(21, 16, false);
    auto model_mobilenet = DeepLabV3Plus_MobileNetV2(21, 16, false);

    convert_model_with_offload(model_resnet50);
    convert_model_with_offload(model_resnet101);
    convert_model_with_offload(model_mobilenet);

    auto images = createInput({1, 3, getInputSize(512), getInputSize(512)});

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
    expectFiniteNonZero(output_resnet50.tensor());
    expectFiniteNonZero(output_resnet101.tensor());
    expectFiniteNonZero(output_mobilenet.tensor());

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
    convert_model_with_offload(model);

    // Test with larger batch size
    auto images = createInput({8, 3, 256, 256});
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], 8) << "Large batch processing failed on " << backend_name();
    EXPECT_EQ(output.tensor().dtype(), dtype());
    expectFiniteNonZero(output.tensor());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, MinimalInputSize) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model_with_offload(model);

    // Test with minimal viable input (64x64)
    auto images = createInput({1, 3, 64, 64});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {1, 21, 64, 64});
    expectFiniteNonZero(output.tensor());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, VeryLargeInputSize) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model_with_offload(model);

    // Use smaller input size for CPU to avoid timeout (2048x2048 is too slow on CPU)
    // GPU backends use reduced sizes for Float64/Float16 due to memory constraints
    int64_t size;
    if (backend_name() == "cpu") {
        size = 512;
    } else {
        // Use getInputSize to handle memory-constrained GPU configurations
        size = getInputSize(1024);
    }

    auto images = createInput({1, 3, size, size});
    Variable output = model->forward(images);

    expectShape(output.tensor(), {1, 21, size, size});
    expectFiniteNonZero(output.tensor());
}

TEST_P(DeepLabV3PlusMultiDTypeTest, SequentialForwardPasses) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    convert_model_with_offload(model);

    // Multiple forward passes should be consistent
    auto images = createInput({1, 3, getInputSize(512), getInputSize(512)});

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
    expectFiniteNonZero(output1.tensor());
    expectFiniteNonZero(output2.tensor());
    expectFiniteNonZero(output3.tensor());
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DeepLabV3PlusMultiDTypeTest);

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
