/**
 * @file test_vit.cpp
 * @brief Comprehensive tests for Vision Transformer (ViT) variants
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/vit.hpp"
#include "../../include/tenzor/core/tensor.hpp"
#include "../../include/tenzor/autograd/variable.hpp"

using namespace tenzor;
using namespace tenzor::models;

class ViTTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = Device::cpu();
    }

    Device device_;
};

// ============================================================================
// PatchEmbedding Tests
// ============================================================================

TEST_F(ViTTest, PatchEmbeddingForwardShape) {
    int64_t image_size = 224;
    int64_t patch_size = 16;
    int64_t num_channels = 3;
    int64_t hidden_size = 768;

    auto patch_embed = std::make_shared<PatchEmbedding>(
        image_size, patch_size, num_channels, hidden_size);

    Variable input(Tensor({2, num_channels, image_size, image_size},
                         DType::Float32, device_), true);
    Variable output = patch_embed->forward(input);

    // Expected: (batch, num_patches, hidden_size)
    // num_patches = (224/16) * (224/16) = 196
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 196, 768}));
}

TEST_F(ViTTest, PatchEmbeddingGradientFlow) {
    auto patch_embed = std::make_shared<PatchEmbedding>(224, 16, 3, 768);

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = patch_embed->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    auto params = patch_embed->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_F(ViTTest, PatchEmbeddingNumPatches) {
    auto patch_embed = std::make_shared<PatchEmbedding>(224, 16, 3, 768);
    EXPECT_EQ(patch_embed->num_patches(), 196);

    auto patch_embed_32 = std::make_shared<PatchEmbedding>(224, 32, 3, 768);
    EXPECT_EQ(patch_embed_32->num_patches(), 49);
}

// ============================================================================
// ViTEmbeddings Tests
// ============================================================================

TEST_F(ViTTest, ViTEmbeddingsForwardShape) {
    auto config = ViTConfig::base_patch16(224);
    auto embeddings = std::make_shared<ViTEmbeddings>(config);

    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = embeddings->forward(input);

    // Expected: (batch, num_patches + 1 (CLS), hidden_size)
    // num_patches = 196, so seq_len = 197
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 197, 768}));
}

TEST_F(ViTTest, ViTEmbeddingsGradientFlow) {
    auto config = ViTConfig::base_patch16(224);
    auto embeddings = std::make_shared<ViTEmbeddings>(config);

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = embeddings->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
}

// ============================================================================
// ViT Base/16 Tests
// ============================================================================

TEST_F(ViTTest, ViTBaseConfig) {
    auto config = ViTConfig::base_patch16(224);

    EXPECT_EQ(config.image_size, 224);
    EXPECT_EQ(config.patch_size, 16);
    EXPECT_EQ(config.hidden_size, 768);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 12);
    EXPECT_EQ(config.intermediate_size, 3072);
    EXPECT_EQ(config.num_patches(), 196);
    EXPECT_EQ(config.seq_length(), 197);
}

TEST_F(ViTTest, ViTBasePatch16ForwardShape) {
    auto model = ViT_Base_Patch16(1000, false, 224);

    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
}

TEST_F(ViTTest, ViTBasePatch16GradientFlow) {
    auto model = ViT_Base_Patch16(10, false, 224);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_F(ViTTest, ViTBasePatch16ParameterCount) {
    auto model = ViT_Base_Patch16(1000, false, 224);
    auto params = model->parameters();

    // ViT-Base should have around 86M parameters
    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // Allow 20% tolerance
    EXPECT_GT(total_params, 70'000'000);
    EXPECT_LT(total_params, 100'000'000);
}

// ============================================================================
// ViT Base/32 Tests
// ============================================================================

TEST_F(ViTTest, ViTBasePatch32Config) {
    auto config = ViTConfig::base_patch32(224);

    EXPECT_EQ(config.patch_size, 32);
    EXPECT_EQ(config.num_patches(), 49);  // 224/32 = 7, 7*7 = 49
    EXPECT_EQ(config.seq_length(), 50);   // 49 + 1 (CLS)
}

TEST_F(ViTTest, ViTBasePatch32ForwardShape) {
    auto model = ViT_Base_Patch32(1000, false, 224);

    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
}

TEST_F(ViTTest, ViTBasePatch32GradientFlow) {
    auto model = ViT_Base_Patch32(10, false, 224);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
}

// ============================================================================
// ViT Large/16 Tests
// ============================================================================

TEST_F(ViTTest, ViTLargeConfig) {
    auto config = ViTConfig::large_patch16(224);

    EXPECT_EQ(config.image_size, 224);
    EXPECT_EQ(config.patch_size, 16);
    EXPECT_EQ(config.hidden_size, 1024);
    EXPECT_EQ(config.num_hidden_layers, 24);
    EXPECT_EQ(config.num_attention_heads, 16);
    EXPECT_EQ(config.intermediate_size, 4096);
}

TEST_F(ViTTest, ViTLargePatch16ForwardShape) {
    auto model = ViT_Large_Patch16(1000, false, 224);

    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
}

TEST_F(ViTTest, ViTLargePatch16GradientFlow) {
    auto model = ViT_Large_Patch16(10, false, 224);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
}

TEST_F(ViTTest, ViTLargePatch16ParameterCount) {
    auto model = ViT_Large_Patch16(1000, false, 224);
    auto params = model->parameters();

    // ViT-Large should have around 307M parameters
    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // Allow 20% tolerance
    EXPECT_GT(total_params, 250'000'000);
    EXPECT_LT(total_params, 370'000'000);
}

// ============================================================================
// ViT Large/32 Tests
// ============================================================================

TEST_F(ViTTest, ViTLargePatch32ForwardShape) {
    auto model = ViT_Large_Patch32(1000, false, 224);

    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
}

// ============================================================================
// ViT Huge/14 Tests
// ============================================================================

TEST_F(ViTTest, ViTHugeConfig) {
    auto config = ViTConfig::huge_patch14(224);

    EXPECT_EQ(config.image_size, 224);
    EXPECT_EQ(config.patch_size, 14);
    EXPECT_EQ(config.hidden_size, 1280);
    EXPECT_EQ(config.num_hidden_layers, 32);
    EXPECT_EQ(config.num_attention_heads, 16);
    EXPECT_EQ(config.intermediate_size, 5120);
    EXPECT_EQ(config.num_patches(), 256);  // (224/14)^2 = 16*16 = 256
    EXPECT_EQ(config.seq_length(), 257);   // 256 + 1 (CLS)
}

TEST_F(ViTTest, ViTHugePatch14ForwardShape) {
    auto model = ViT_Huge_Patch14(1000, false, 224);

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 1000}));
}

TEST_F(ViTTest, ViTHugePatch14GradientFlow) {
    auto model = ViT_Huge_Patch14(10, false, 224);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
}

TEST_F(ViTTest, ViTHugePatch14ParameterCount) {
    auto model = ViT_Huge_Patch14(1000, false, 224);
    auto params = model->parameters();

    // ViT-Huge should have around 632M parameters
    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // Allow 20% tolerance
    EXPECT_GT(total_params, 500'000'000);
    EXPECT_LT(total_params, 760'000'000);
}

// ============================================================================
// ViT Huge/16 Tests
// ============================================================================

TEST_F(ViTTest, ViTHugePatch16ForwardShape) {
    auto model = ViT_Huge_Patch16(1000, false, 224);

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 1000}));
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(ViTTest, ViTBaseBatchSizeOne) {
    auto model = ViT_Base_Patch16(10, false, 224);

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 10}));
}

TEST_F(ViTTest, ViTBaseCustomClasses) {
    auto model = ViT_Base_Patch16(100, false, 224);

    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 100}));
}

TEST_F(ViTTest, ViTBaseDifferentImageSize) {
    // Test with 384x384 input (commonly used for fine-tuning)
    auto model = ViT_Base_Patch16(1000, false, 384);

    Variable input(Tensor({1, 3, 384, 384}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 1000}));
}

TEST_F(ViTTest, PatchEmbeddingDifferentPatchSizes) {
    // Test patch size 14
    auto patch_embed_14 = std::make_shared<PatchEmbedding>(224, 14, 3, 1280);
    Variable input_14(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output_14 = patch_embed_14->forward(input_14);

    auto shape_14 = output_14.tensor().shape();
    // 224/14 = 16, 16*16 = 256 patches
    EXPECT_EQ(std::vector<int64_t>(shape_14.begin(), shape_14.end()),
              (std::vector<int64_t>{1, 256, 1280}));

    // Test patch size 32
    auto patch_embed_32 = std::make_shared<PatchEmbedding>(224, 32, 3, 768);
    Variable input_32(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output_32 = patch_embed_32->forward(input_32);

    auto shape_32 = output_32.tensor().shape();
    // 224/32 = 7, 7*7 = 49 patches
    EXPECT_EQ(std::vector<int64_t>(shape_32.begin(), shape_32.end()),
              (std::vector<int64_t>{1, 49, 768}));
}

TEST_F(ViTTest, ViTConfigNumPatchesCalculation) {
    auto config_16 = ViTConfig::base_patch16(224);
    EXPECT_EQ(config_16.num_patches(), 196);

    auto config_32 = ViTConfig::base_patch32(224);
    EXPECT_EQ(config_32.num_patches(), 49);

    auto config_14 = ViTConfig::huge_patch14(224);
    EXPECT_EQ(config_14.num_patches(), 256);
}


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
