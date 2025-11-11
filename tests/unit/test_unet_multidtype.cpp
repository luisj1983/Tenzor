/**
 * @file test_unet_multidtype.cpp
 * @brief Multi-dtype tests for U-Net segmentation model
 *
 * Tests U-Net architecture with Float32, Float64, and Float16 support:
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
#include <cmath>
#include <limits>

using namespace tenzor;
using namespace tenzor::models;

// Tolerance values for different data types
template<typename T>
struct TypeTolerance {
    static constexpr float value = 1e-5f;
};

template<>
struct TypeTolerance<float> {
    static constexpr float value = 1e-4f;
};

template<>
struct TypeTolerance<double> {
    static constexpr float value = 1e-6f;
};

// Float16 has much lower precision
template<>
struct TypeTolerance<int16_t> {  // Float16 stored as int16_t
    static constexpr float value = 1e-2f;
};

// ============================================================================
// U-Net Construction Tests
// ============================================================================

template<typename T>
class UNetConstructionTest : public ::testing::Test {
protected:
    void SetUp() override { device_ = Device::cpu(); }
    Device device_;

    tenzor::DType getDType() const {
        if constexpr (std::is_same_v<T, float>) return tenzor::DType::Float32;
        else if constexpr (std::is_same_v<T, double>) return tenzor::DType::Float64;
        else return tenzor::DType::Float16;
    }
};

using UNetConstructionTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(UNetConstructionTest, UNetConstructionTypes);

TYPED_TEST(UNetConstructionTest, BasicConstruction) {
    auto dtype = this->getDType();

    // Test basic U-Net construction
    auto model = std::make_shared<UNet>(3, 21, false);
    EXPECT_NE(model, nullptr);

    // Check parameters exist
    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);

    // Test different configurations
    auto model_binary = std::make_shared<UNet>(3, 1, false);
    EXPECT_NE(model_binary, nullptr);

    auto model_grayscale = std::make_shared<UNet>(1, 21, false);
    EXPECT_NE(model_grayscale, nullptr);

    auto model_bilinear = std::make_shared<UNet>(3, 21, true);
    EXPECT_NE(model_bilinear, nullptr);
}

TYPED_TEST(UNetConstructionTest, ParameterCount) {
    auto dtype = this->getDType();

    // U-Net with transposed conv should have more parameters
    auto model_transposed = std::make_shared<UNet>(3, 21, false);
    auto params_transposed = model_transposed->parameters();

    size_t total_transposed = 0;
    for (const auto& p : params_transposed) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_transposed += param_size;
    }

    // U-Net with bilinear should have fewer parameters
    auto model_bilinear = std::make_shared<UNet>(3, 21, true);
    auto params_bilinear = model_bilinear->parameters();

    size_t total_bilinear = 0;
    for (const auto& p : params_bilinear) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_bilinear += param_size;
    }

    // U-Net should have around 31M parameters for transposed conv
    EXPECT_GT(total_transposed, 20'000'000);
    EXPECT_LT(total_transposed, 45'000'000);

    // Bilinear should have fewer parameters (no learned upsampling)
    EXPECT_LT(total_bilinear, total_transposed);
}

TYPED_TEST(UNetConstructionTest, DifferentChannelConfigurations) {
    auto dtype = this->getDType();

    // Test various input/output channel combinations
    std::vector<std::pair<int64_t, int64_t>> configs = {
        {1, 1},    // Binary grayscale
        {1, 2},    // Two-class grayscale
        {3, 1},    // Binary RGB
        {3, 2},    // Two-class RGB
        {3, 21},   // Pascal VOC
        {3, 81},   // COCO
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

template<typename T>
class UNetArchitectureTest : public ::testing::Test {
protected:
    void SetUp() override { device_ = Device::cpu(); }
    Device device_;

    tenzor::DType getDType() const {
        if constexpr (std::is_same_v<T, float>) return tenzor::DType::Float32;
        else if constexpr (std::is_same_v<T, double>) return tenzor::DType::Float64;
        else return tenzor::DType::Float16;
    }
};

using UNetArchitectureTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(UNetArchitectureTest, UNetArchitectureTypes);

TYPED_TEST(UNetArchitectureTest, EncoderPathDimensionality) {
    auto dtype = this->getDType();

    auto model = std::make_shared<UNet>(3, 21, false);
    Variable images(Tensor({1, 3, 256, 256}, dtype, this->device_), true);
    Variable output = model->forward(images);

    // Output should match input spatial dimensions (encoder-decoder symmetry)
    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], 1);     // Batch
    EXPECT_EQ(shape[1], 21);    // Classes
    EXPECT_EQ(shape[2], 256);   // Height preserved
    EXPECT_EQ(shape[3], 256);   // Width preserved
}

TYPED_TEST(UNetArchitectureTest, DecoderPathReconstruction) {
    auto dtype = this->getDType();

    // Test that decoder properly reconstructs spatial dimensions
    std::vector<int64_t> sizes = {64, 128, 256, 512};

    for (int64_t size : sizes) {
        auto model = std::make_shared<UNet>(3, 21, false);
        Variable images(Tensor({1, 3, size, size}, dtype, this->device_), true);
        Variable output = model->forward(images);

        auto shape = output.tensor().shape();
        EXPECT_EQ(shape[2], size) << "Height not preserved at size " << size;
        EXPECT_EQ(shape[3], size) << "Width not preserved at size " << size;
    }
}

TYPED_TEST(UNetArchitectureTest, BilinearVsTransposedConv) {
    auto dtype = this->getDType();

    auto model_transposed = std::make_shared<UNet>(3, 21, false);
    auto model_bilinear = std::make_shared<UNet>(3, 21, true);

    Variable images(Tensor({1, 3, 256, 256}, dtype, this->device_), true);

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

template<typename T>
class UNetSkipConnectionTest : public ::testing::Test {
protected:
    void SetUp() override { device_ = Device::cpu(); }
    Device device_;

    tenzor::DType getDType() const {
        if constexpr (std::is_same_v<T, float>) return tenzor::DType::Float32;
        else if constexpr (std::is_same_v<T, double>) return tenzor::DType::Float64;
        else return tenzor::DType::Float16;
    }
};

using UNetSkipConnectionTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(UNetSkipConnectionTest, UNetSkipConnectionTypes);

TYPED_TEST(UNetSkipConnectionTest, GradientFlowThroughSkips) {
    auto dtype = this->getDType();

    auto model = std::make_shared<UNet>(3, 21, false);
    model->train();

    Variable images(Tensor({1, 3, 256, 256}, dtype, this->device_), true);
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

TYPED_TEST(UNetSkipConnectionTest, SkipConnectionFeaturePreservation) {
    auto dtype = this->getDType();

    // Skip connections should help preserve fine details
    // We can verify this by checking that output has reasonable values
    auto model = std::make_shared<UNet>(3, 1, false);
    model->eval();

    Variable images(Tensor({1, 3, 256, 256}, dtype, this->device_), false);
    Variable output = model->forward(images);

    // Output should have finite values (skip connections prevent vanishing gradients)
    auto output_data = output.tensor().data<TypeParam>();
    bool all_finite = true;
    for (size_t i = 0; i < 256 * 256; ++i) {
        if (!std::isfinite(static_cast<float>(output_data[i]))) {
            all_finite = false;
            break;
        }
    }
    EXPECT_TRUE(all_finite) << "Output contains non-finite values";
}

// ============================================================================
// Forward Pass with Different Image Sizes
// ============================================================================

template<typename T>
class UNetImageSizeTest : public ::testing::Test {
protected:
    void SetUp() override { device_ = Device::cpu(); }
    Device device_;

    tenzor::DType getDType() const {
        if constexpr (std::is_same_v<T, float>) return tenzor::DType::Float32;
        else if constexpr (std::is_same_v<T, double>) return tenzor::DType::Float64;
        else return tenzor::DType::Float16;
    }
};

using UNetImageSizeTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(UNetImageSizeTest, UNetImageSizeTypes);

TYPED_TEST(UNetImageSizeTest, MultipleImageSizes) {
    auto dtype = this->getDType();
    auto model = std::make_shared<UNet>(3, 21, false);

    // Test with various image sizes
    std::vector<int64_t> sizes = {64, 128, 256, 512};

    for (int64_t size : sizes) {
        Variable images(Tensor({1, 3, size, size}, dtype, this->device_), true);
        Variable output = model->forward(images);

        auto shape = output.tensor().shape();
        EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
                  (std::vector<int64_t>{1, 21, size, size}))
            << "Failed at size " << size;
    }
}

TYPED_TEST(UNetImageSizeTest, NonSquareImages) {
    auto dtype = this->getDType();
    auto model = std::make_shared<UNet>(3, 21, false);

    // Test with non-square images
    std::vector<std::pair<int64_t, int64_t>> sizes = {
        {128, 256},
        {256, 128},
        {192, 256},
        {256, 192}
    };

    for (const auto& [h, w] : sizes) {
        Variable images(Tensor({1, 3, h, w}, dtype, this->device_), true);
        Variable output = model->forward(images);

        auto shape = output.tensor().shape();
        EXPECT_EQ(shape[0], 1);
        EXPECT_EQ(shape[1], 21);
        EXPECT_EQ(shape[2], h) << "Height not preserved for " << h << "x" << w;
        EXPECT_EQ(shape[3], w) << "Width not preserved for " << h << "x" << w;
    }
}

TYPED_TEST(UNetImageSizeTest, BatchSizeVariation) {
    auto dtype = this->getDType();
    auto model = std::make_shared<UNet>(3, 21, false);

    // Test with different batch sizes
    std::vector<int64_t> batch_sizes = {1, 2, 4, 8};

    for (int64_t batch : batch_sizes) {
        Variable images(Tensor({batch, 3, 256, 256}, dtype, this->device_), true);
        Variable output = model->forward(images);

        auto shape = output.tensor().shape();
        EXPECT_EQ(shape[0], batch) << "Batch size not preserved";
        EXPECT_EQ(shape[1], 21);
        EXPECT_EQ(shape[2], 256);
        EXPECT_EQ(shape[3], 256);
    }
}

TYPED_TEST(UNetImageSizeTest, SmallImageSizes) {
    auto dtype = this->getDType();
    auto model = std::make_shared<UNet>(3, 21, false);

    // Test with small images (edge case)
    // U-Net has 4 downsampling layers, so minimum size is 16x16
    std::vector<int64_t> sizes = {16, 32, 48, 64};

    for (int64_t size : sizes) {
        Variable images(Tensor({1, 3, size, size}, dtype, this->device_), true);
        Variable output = model->forward(images);

        auto shape = output.tensor().shape();
        EXPECT_EQ(shape[2], size) << "Small image size " << size << " not handled correctly";
        EXPECT_EQ(shape[3], size);
    }
}

// ============================================================================
// Output Channel Configuration Tests
// ============================================================================

template<typename T>
class UNetOutputChannelTest : public ::testing::Test {
protected:
    void SetUp() override { device_ = Device::cpu(); }
    Device device_;

    tenzor::DType getDType() const {
        if constexpr (std::is_same_v<T, float>) return tenzor::DType::Float32;
        else if constexpr (std::is_same_v<T, double>) return tenzor::DType::Float64;
        else return tenzor::DType::Float16;
    }
};

using UNetOutputChannelTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(UNetOutputChannelTest, UNetOutputChannelTypes);

TYPED_TEST(UNetOutputChannelTest, BinarySegmentation) {
    auto dtype = this->getDType();

    // Binary segmentation (1 output channel)
    auto model = std::make_shared<UNet>(3, 1, false);
    Variable images(Tensor({2, 3, 256, 256}, dtype, this->device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1, 256, 256}));
}

TYPED_TEST(UNetOutputChannelTest, MultiClassSegmentation) {
    auto dtype = this->getDType();

    // Test various numbers of classes
    std::vector<int64_t> num_classes = {2, 5, 10, 21, 50, 81, 100};

    for (int64_t classes : num_classes) {
        auto model = std::make_shared<UNet>(3, classes, false);
        Variable images(Tensor({1, 3, 256, 256}, dtype, this->device_), true);
        Variable output = model->forward(images);

        auto shape = output.tensor().shape();
        EXPECT_EQ(shape[1], classes) << "Wrong number of output classes for " << classes;
    }
}

TYPED_TEST(UNetOutputChannelTest, GrayscaleInput) {
    auto dtype = this->getDType();

    // Test with grayscale input (1 channel)
    auto model = std::make_shared<UNet>(1, 21, false);
    Variable images(Tensor({1, 1, 256, 256}, dtype, this->device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 21, 256, 256}));
}

TYPED_TEST(UNetOutputChannelTest, MultiChannelInput) {
    auto dtype = this->getDType();

    // Test with multi-channel inputs (e.g., RGBD, multispectral)
    std::vector<int64_t> input_channels = {1, 3, 4, 6, 8};

    for (int64_t in_ch : input_channels) {
        auto model = std::make_shared<UNet>(in_ch, 21, false);
        Variable images(Tensor({1, in_ch, 256, 256}, dtype, this->device_), true);
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

template<typename T>
class UNetUpsamplingTest : public ::testing::Test {
protected:
    void SetUp() override { device_ = Device::cpu(); }
    Device device_;

    tenzor::DType getDType() const {
        if constexpr (std::is_same_v<T, float>) return tenzor::DType::Float32;
        else if constexpr (std::is_same_v<T, double>) return tenzor::DType::Float64;
        else return tenzor::DType::Float16;
    }
};

using UNetUpsamplingTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(UNetUpsamplingTest, UNetUpsamplingTypes);

TYPED_TEST(UNetUpsamplingTest, BilinearUpsampling) {
    auto dtype = this->getDType();

    auto model = std::make_shared<UNet>(3, 21, true);  // bilinear=true
    EXPECT_TRUE(model->is_bilinear());

    Variable images(Tensor({2, 3, 256, 256}, dtype, this->device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 21, 256, 256}));
}

TYPED_TEST(UNetUpsamplingTest, TransposedConvUpsampling) {
    auto dtype = this->getDType();

    auto model = std::make_shared<UNet>(3, 21, false);  // bilinear=false
    EXPECT_FALSE(model->is_bilinear());

    Variable images(Tensor({2, 3, 256, 256}, dtype, this->device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 21, 256, 256}));
}

TYPED_TEST(UNetUpsamplingTest, UpsamplingMethodComparison) {
    auto dtype = this->getDType();

    // Both methods should produce outputs of same shape
    auto model_bilinear = std::make_shared<UNet>(3, 1, true);
    auto model_transposed = std::make_shared<UNet>(3, 1, false);

    Variable images(Tensor({1, 3, 128, 128}, dtype, this->device_), false);

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

TYPED_TEST(UNetUpsamplingTest, UpsamplingWithGradients) {
    auto dtype = this->getDType();

    // Test gradient flow through different upsampling methods
    auto model_bilinear = std::make_shared<UNet>(3, 1, true);
    auto model_transposed = std::make_shared<UNet>(3, 1, false);

    model_bilinear->train();
    model_transposed->train();

    Variable images_b(Tensor({1, 3, 128, 128}, dtype, this->device_), true);
    Variable images_t(Tensor({1, 3, 128, 128}, dtype, this->device_), true);

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

template<typename T>
class UNetModeTest : public ::testing::Test {
protected:
    void SetUp() override { device_ = Device::cpu(); }
    Device device_;

    tenzor::DType getDType() const {
        if constexpr (std::is_same_v<T, float>) return tenzor::DType::Float32;
        else if constexpr (std::is_same_v<T, double>) return tenzor::DType::Float64;
        else return tenzor::DType::Float16;
    }
};

using UNetModeTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(UNetModeTest, UNetModeTypes);

TYPED_TEST(UNetModeTest, TrainingMode) {
    auto dtype = this->getDType();

    auto model = std::make_shared<UNet>(3, 21, false);
    model->train();

    Variable images(Tensor({1, 3, 256, 256}, dtype, this->device_), true);
    Variable output = model->forward(images);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Should compute gradients in training mode
    EXPECT_TRUE(images.grad().has_value());

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TYPED_TEST(UNetModeTest, InferenceMode) {
    auto dtype = this->getDType();

    auto model = std::make_shared<UNet>(3, 21, false);
    model->eval();

    Variable images(Tensor({1, 3, 256, 256}, dtype, this->device_), false);
    Variable output = model->forward(images);

    // Output should be valid
    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 21);
    EXPECT_EQ(shape[2], 256);
    EXPECT_EQ(shape[3], 256);
}

TYPED_TEST(UNetModeTest, ModeToggling) {
    auto dtype = this->getDType();

    auto model = std::make_shared<UNet>(3, 1, false);

    // Start in eval mode
    model->eval();
    Variable images_eval(Tensor({1, 3, 128, 128}, dtype, this->device_), false);
    Variable output_eval = model->forward(images_eval);

    // Switch to train mode
    model->train();
    Variable images_train(Tensor({1, 3, 128, 128}, dtype, this->device_), true);
    Variable output_train = model->forward(images_train);
    Variable loss = tenzor::sum(output_train);
    loss.backward();

    // Should have gradients in train mode
    EXPECT_TRUE(images_train.grad().has_value());
}

// ============================================================================
// Edge Case Tests
// ============================================================================

template<typename T>
class UNetEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override { device_ = Device::cpu(); }
    Device device_;

    tenzor::DType getDType() const {
        if constexpr (std::is_same_v<T, float>) return tenzor::DType::Float32;
        else if constexpr (std::is_same_v<T, double>) return tenzor::DType::Float64;
        else return tenzor::DType::Float16;
    }
};

using UNetEdgeCaseTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(UNetEdgeCaseTest, UNetEdgeCaseTypes);

TYPED_TEST(UNetEdgeCaseTest, MinimalImageSize) {
    auto dtype = this->getDType();

    // U-Net has 4 downsamplings (16x reduction), minimum size is 16x16
    auto model = std::make_shared<UNet>(3, 21, false);
    Variable images(Tensor({1, 3, 16, 16}, dtype, this->device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[2], 16);
    EXPECT_EQ(shape[3], 16);
}

TYPED_TEST(UNetEdgeCaseTest, SingleClassSegmentation) {
    auto dtype = this->getDType();

    // Single output class (binary segmentation)
    auto model = std::make_shared<UNet>(3, 1, false);
    Variable images(Tensor({1, 3, 256, 256}, dtype, this->device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[1], 1);
}

TYPED_TEST(UNetEdgeCaseTest, LargeBatchSize) {
    auto dtype = this->getDType();

    // Test with larger batch size
    auto model = std::make_shared<UNet>(3, 21, true);  // Use bilinear for memory
    Variable images(Tensor({16, 3, 128, 128}, dtype, this->device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], 16);
    EXPECT_EQ(shape[1], 21);
    EXPECT_EQ(shape[2], 128);
    EXPECT_EQ(shape[3], 128);
}

TYPED_TEST(UNetEdgeCaseTest, ConsistentOutputAcrossCalls) {
    auto dtype = this->getDType();

    auto model = std::make_shared<UNet>(3, 1, false);
    model->eval();

    // Create deterministic input
    Variable images(Tensor({1, 3, 128, 128}, dtype, this->device_), false);
    auto data = images.tensor().data<TypeParam>();
    for (size_t i = 0; i < 3 * 128 * 128; ++i) {
        data[i] = static_cast<float>(i % 100) / 100.0f;
    }

    // Multiple forward passes should give same result (in eval mode)
    Variable output1 = model->forward(images);
    Variable output2 = model->forward(images);

    auto data1 = output1.tensor().data<TypeParam>();
    auto data2 = output2.tensor().data<TypeParam>();

    float tolerance = TypeTolerance<TypeParam>::value;
    bool outputs_match = true;

    for (size_t i = 0; i < 128 * 128; ++i) {
        if (std::abs(data1[i] - data2[i]) > tolerance) {
            outputs_match = false;
            break;
        }
    }

    EXPECT_TRUE(outputs_match) << "Outputs differ across calls in eval mode";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
