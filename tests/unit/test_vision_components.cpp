/**
 * @file test_vision_components.cpp
 * @brief Tests for vision components: PatchEmbedding, SE, MBConv, etc.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/vit.hpp"
#include "../../include/tenzor/models/efficientnet.hpp"
#include "../../include/tenzor/models/swin_transformer.hpp"
#include "../../include/tenzor/models/convnext.hpp"
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::models;

class VisionComponentsTest : public ::testing::Test {
protected:
    void SetUp() override { device_ = Device::cpu(); }
    Device device_;
};

// ============================================================================
// PatchEmbedding Component Tests
// ============================================================================

TEST_F(VisionComponentsTest, PatchEmbedding16x16) {
    auto patch_embed = std::make_shared<PatchEmbedding>(224, 16, 3, 768);
    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = patch_embed->forward(input);

    // (224/16) * (224/16) = 14 * 14 = 196 patches
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 196, 768}));
}

TEST_F(VisionComponentsTest, PatchEmbedding14x14) {
    auto patch_embed = std::make_shared<PatchEmbedding>(224, 14, 3, 1280);
    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = patch_embed->forward(input);

    // (224/14) * (224/14) = 16 * 16 = 256 patches
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 256, 1280}));
}

TEST_F(VisionComponentsTest, PatchEmbeddingGradientFlow) {
    auto patch_embed = std::make_shared<PatchEmbedding>(224, 16, 3, 768);
    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = patch_embed->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    auto params = patch_embed->parameters();
    EXPECT_GT(params.size(), 0);
}

// ============================================================================
// Squeeze-Excitation Component Tests
// ============================================================================

TEST_F(VisionComponentsTest, SqueezeExcitationForwardShape) {
    auto se = std::make_shared<tenzor::models::EfficientNetSqueezeExcitation>(64, 0.25);
    Variable input(Tensor({2, 64, 14, 14}, DType::Float32, device_), true);
    Variable output = se->forward(input);

    // SE preserves shape
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 64, 14, 14}));
}

TEST_F(VisionComponentsTest, SqueezeExcitationDifferentReduction) {
    auto se_025 = std::make_shared<tenzor::models::EfficientNetSqueezeExcitation>(128, 0.25);
    Variable input(Tensor({1, 128, 7, 7}, DType::Float32, device_), true);
    Variable output = se_025->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 128, 7, 7}));
}

TEST_F(VisionComponentsTest, SqueezeExcitationGradientFlow) {
    auto se = std::make_shared<tenzor::models::EfficientNetSqueezeExcitation>(32, 0.25);
    Variable input(Tensor({1, 32, 14, 14}, DType::Float32, device_), true);
    Variable output = se->forward(input);
    Variable loss = tenzor::sum(output * output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

// ============================================================================
// MBConv Component Tests (EfficientNet)
// ============================================================================

TEST_F(VisionComponentsTest, MBConvBlockExpand1Shape) {
    auto mbconv = std::make_shared<MBConvBlock>(32, 32, 1, 3, 1, true, 0.25, 0.0);
    Variable input(Tensor({2, 32, 28, 28}, DType::Float32, device_), true);
    Variable output = mbconv->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 32, 28, 28}));
}

TEST_F(VisionComponentsTest, MBConvBlockExpand6Shape) {
    auto mbconv = std::make_shared<MBConvBlock>(24, 40, 6, 3, 2, true, 0.25, 0.0);
    Variable input(Tensor({2, 24, 56, 56}, DType::Float32, device_), true);
    Variable output = mbconv->forward(input);

    // Stride=2 halves spatial dimensions
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 40, 28, 28}));
}

TEST_F(VisionComponentsTest, MBConvBlockKernel5) {
    auto mbconv = std::make_shared<MBConvBlock>(40, 80, 6, 5, 2, true, 0.25, 0.0);
    Variable input(Tensor({1, 40, 28, 28}, DType::Float32, device_), true);
    Variable output = mbconv->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 80, 14, 14}));
}

TEST_F(VisionComponentsTest, MBConvBlockGradientFlow) {
    auto mbconv = std::make_shared<MBConvBlock>(16, 24, 6, 3, 1, true, 0.25, 0.0);
    Variable input(Tensor({1, 16, 56, 56}, DType::Float32, device_), true);
    Variable output = mbconv->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

// ============================================================================
// ConvNeXt Block Component Tests
// ============================================================================

TEST_F(VisionComponentsTest, ConvNeXtBlockForwardShape) {
    auto block = std::make_shared<ConvNeXtBlock>(96, 0.0, 1e-6);
    Variable input(Tensor({2, 96, 56, 56}, DType::Float32, device_), true);
    Variable output = block->forward(input);

    // ConvNeXt block preserves shape
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 96, 56, 56}));
}

TEST_F(VisionComponentsTest, ConvNeXtBlockGradientFlow) {
    auto block = std::make_shared<ConvNeXtBlock>(96, 0.1, 1e-6);
    Variable input(Tensor({1, 96, 56, 56}, DType::Float32, device_), true);
    Variable output = block->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

TEST_F(VisionComponentsTest, ConvNeXtBlockDifferentChannels) {
    auto block_96 = std::make_shared<ConvNeXtBlock>(96, 0.0, 1e-6);
    Variable input_96(Tensor({1, 96, 56, 56}, DType::Float32, device_), true);
    Variable output_96 = block_96->forward(input_96);
    auto shape_96 = output_96.tensor().shape();
    EXPECT_EQ(shape_96[1], 96);

    auto block_192 = std::make_shared<ConvNeXtBlock>(192, 0.0, 1e-6);
    Variable input_192(Tensor({1, 192, 28, 28}, DType::Float32, device_), true);
    Variable output_192 = block_192->forward(input_192);
    auto shape_192 = output_192.tensor().shape();
    EXPECT_EQ(shape_192[1], 192);
}

// ============================================================================
// LayerScale Component Tests
// ============================================================================

TEST_F(VisionComponentsTest, LayerScaleForwardShape) {
    auto layer_scale = std::make_shared<LayerScale>(96, 1e-6);
    Variable input(Tensor({2, 96, 56, 56}, DType::Float32, device_), true);
    Variable output = layer_scale->forward(input);

    // LayerScale preserves shape
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 96, 56, 56}));
}

TEST_F(VisionComponentsTest, LayerScaleGradientFlow) {
    auto layer_scale = std::make_shared<LayerScale>(96, 1e-6);
    Variable input(Tensor({1, 96, 56, 56}, DType::Float32, device_), true);
    Variable output = layer_scale->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    auto params = layer_scale->parameters();
    EXPECT_GT(params.size(), 0);
}

// ============================================================================
// Swin MLP Component Tests
// ============================================================================

TEST_F(VisionComponentsTest, SwinMLPForwardShape) {
    auto mlp = std::make_shared<SwinMLP>(96, 384, 96, 0.0);
    Variable input(Tensor({2, 56*56, 96}, DType::Float32, device_), true);
    Variable output = mlp->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 56*56, 96}));
}

TEST_F(VisionComponentsTest, SwinMLPGradientFlow) {
    auto mlp = std::make_shared<SwinMLP>(96, 384, 96, 0.1);
    Variable input(Tensor({1, 3136, 96}, DType::Float32, device_), true);
    Variable output = mlp->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(VisionComponentsTest, PatchEmbeddingBatchSizeOne) {
    auto patch_embed = std::make_shared<PatchEmbedding>(224, 16, 3, 768);
    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = patch_embed->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 196, 768}));
}

TEST_F(VisionComponentsTest, SEBlockDifferentChannels) {
    std::vector<int64_t> channel_sizes = {16, 32, 64, 128, 256};

    for (auto channels : channel_sizes) {
        auto se = std::make_shared<tenzor::models::EfficientNetSqueezeExcitation>(channels, 0.25);
        Variable input(Tensor({1, channels, 7, 7}, DType::Float32, device_), true);
        Variable output = se->forward(input);

        auto shape = output.tensor().shape();
        EXPECT_EQ(shape[1], channels);
    }
}
