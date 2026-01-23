/**
 * @file test_vit_multidtype.cpp
 * @brief Multi-backend and multi-dtype tests for Vision Transformer (ViT) variants
 *
 * Tests ViT models across all backends (CPU, CUDA, OneAPI) and dtypes
 * (Float32, Float64, Float16) to ensure:
 * - Proper dtype propagation through patch embedding, position embeddings, and transformers
 * - Correct output shapes across all ViT variants (Tiny, Small, Base, Large, Huge)
 * - Gradient flow with different dtypes and backends
 * - Class token handling across dtypes
 * - Different image sizes and patch sizes
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/vit.hpp"
#include "../../include/tenzor/core/tensor.hpp"
#include "../../include/tenzor/autograd/variable.hpp"
#include "../multi_backend_dtype_fixture.hpp"
#include <tuple>
#include <vector>

using namespace tenzor;
using namespace tenzor::models;
using namespace tenzor::testing;

// ============================================================================
// Parameterized Test Fixture (Multi-Backend + Multi-DType)
// ============================================================================

class ViTMultiDtypeTest : public MultiBackendDTypeTest {
protected:
    /**
     * @brief Get reduced image size for memory-constrained configurations
     * For CUDA on large models with Float64/Float32, use smaller images to fit in 8GB
     * Sizes are chosen to be divisible by common patch sizes (14, 16)
     */
    int getImageSizeForMemory(int default_size, size_t param_count, bool needs_gradients, int patch_size = 16) const {
        if (backend_name() != "cuda") return default_size;

        bool is_float64 = (dtype() == DType::Float64);

        // ViT-Huge (~632M params) with Float64 + gradients needs very small images
        if (param_count > 500'000'000 && needs_gradients && is_float64) {
            return 56;  // 56 is divisible by 14, gives 4x4=16 patches
        }
        // ViT-Huge (~632M params) with Float32 + gradients
        if (param_count > 500'000'000 && needs_gradients) {
            return 112;  // 112 is divisible by 14 and 16
        }
        // ViT-Huge forward-only with Float64
        if (param_count > 500'000'000 && is_float64) {
            return 112;  // Divisible by 14 and 16
        }
        // ViT-Large with gradients and Float64
        if (param_count > 200'000'000 && needs_gradients && is_float64) {
            return 160;  // Divisible by 16
        }
        // Large image sizes (512+) need reduction with Float64
        if (default_size >= 512 && is_float64) {
            return 384;  // Divisible by 16
        }
        return default_size;
    }
};

// ============================================================================
// PatchEmbedding Tests
// ============================================================================

TEST_P(ViTMultiDtypeTest, PatchEmbeddingForwardShape) {
    int64_t image_size = 224;
    int64_t patch_size = 16;
    int64_t num_channels = 3;
    int64_t hidden_size = 768;

    auto patch_embed = std::make_shared<PatchEmbedding>(
        image_size, patch_size, num_channels, hidden_size);
    convert_model(patch_embed);

    auto input = createInput({2, num_channels, image_size, image_size});
    auto output = patch_embed->forward(input);

    // Expected: (batch, num_patches, hidden_size)
    // num_patches = (224/16) * (224/16) = 196
    expectShape(output.tensor(), {2, 196, 768});
    expectDType(output.tensor());
}

TEST_P(ViTMultiDtypeTest, PatchEmbeddingGradientFlow) {
    auto patch_embed = std::make_shared<PatchEmbedding>(224, 16, 3, 768);
    convert_model(patch_embed);

    auto input = createInput({1, 3, 224, 224});
    auto output = patch_embed->forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype());

    auto params = patch_embed->parameters();
    EXPECT_GT(params.size(), 0);

    // Check gradient dtype for parameters
    for (const auto& param : params) {
        if (param->grad().has_value()) {
            EXPECT_EQ(param->grad()->dtype(), dtype());
        }
    }
}

TEST_P(ViTMultiDtypeTest, PatchEmbeddingNumPatches) {
    auto patch_embed = std::make_shared<PatchEmbedding>(224, 16, 3, 768);
    EXPECT_EQ(patch_embed->num_patches(), 196);

    auto patch_embed_32 = std::make_shared<PatchEmbedding>(224, 32, 3, 768);
    EXPECT_EQ(patch_embed_32->num_patches(), 49);

    auto patch_embed_14 = std::make_shared<PatchEmbedding>(224, 14, 3, 1280);
    EXPECT_EQ(patch_embed_14->num_patches(), 256);
}

TEST_P(ViTMultiDtypeTest, PatchEmbeddingDifferentPatchSizes) {
    // Test patch size 14
    auto patch_embed_14 = std::make_shared<PatchEmbedding>(224, 14, 3, 1280);
    convert_model(patch_embed_14);
    auto input_14 = createInput({1, 3, 224, 224});
    auto output_14 = patch_embed_14->forward(input_14);

    expectShape(output_14.tensor(), {1, 256, 1280});
    expectDType(output_14.tensor());

    // Test patch size 32
    auto patch_embed_32 = std::make_shared<PatchEmbedding>(224, 32, 3, 768);
    convert_model(patch_embed_32);
    auto input_32 = createInput({1, 3, 224, 224});
    auto output_32 = patch_embed_32->forward(input_32);

    expectShape(output_32.tensor(), {1, 49, 768});
    expectDType(output_32.tensor());
}

// ============================================================================
// ViTEmbeddings Tests (Patch + Position + Class Token)
// ============================================================================

TEST_P(ViTMultiDtypeTest, ViTEmbeddingsForwardShape) {
    auto config = ViTConfig::base_patch16(224);
    auto embeddings = std::make_shared<ViTEmbeddings>(config);
    convert_model(embeddings);

    auto input = createInput({2, 3, 224, 224});
    auto output = embeddings->forward(input);

    // Expected: (batch, num_patches + 1 (CLS), hidden_size)
    // num_patches = 196, so seq_len = 197
    expectShape(output.tensor(), {2, 197, 768});
    expectDType(output.tensor());
}

TEST_P(ViTMultiDtypeTest, ViTEmbeddingsGradientFlow) {
    auto config = ViTConfig::base_patch16(224);
    auto embeddings = std::make_shared<ViTEmbeddings>(config);
    convert_model(embeddings);

    auto input = createInput({1, 3, 224, 224});
    auto output = embeddings->forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

TEST_P(ViTMultiDtypeTest, ViTEmbeddingsClassToken) {
    auto config = ViTConfig::base_patch16(224);
    auto embeddings = std::make_shared<ViTEmbeddings>(config);
    convert_model(embeddings);

    auto input = createInput({2, 3, 224, 224});
    auto output = embeddings->forward(input);

    // Verify class token is added (seq_len should be num_patches + 1)
    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[1], 197);  // 196 patches + 1 class token
    expectDType(output.tensor());
}

// ============================================================================
// ViT Base/16 Tests
// ============================================================================

TEST_P(ViTMultiDtypeTest, ViTBaseConfig) {
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

TEST_P(ViTMultiDtypeTest, ViTBasePatch16ForwardShape) {
    auto model = ViT_Base_Patch16(1000, false, 224);
    convert_model(model);

    auto input = createInput({2, 3, 224, 224});
    auto output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectDType(output.tensor());
}

TEST_P(ViTMultiDtypeTest, ViTBasePatch16GradientFlow) {
    auto model = ViT_Base_Patch16(10, false, 224);
    convert_model(model);
    model->train();

    auto input = createInput({1, 3, 224, 224});
    auto output = model->forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype());

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(ViTMultiDtypeTest, ViTBasePatch16ParameterCount) {
    auto model = ViT_Base_Patch16(1000, false, 224);
    convert_model(model);
    auto params = model->parameters();

    // ViT-Base should have around 86M parameters
    size_t total_params = countParameters(params);

    // Allow 20% tolerance
    EXPECT_GT(total_params, 70'000'000);
    EXPECT_LT(total_params, 100'000'000);
}

TEST_P(ViTMultiDtypeTest, ViTBasePatch16BatchSizeOne) {
    auto model = ViT_Base_Patch16(10, false, 224);
    convert_model(model);

    auto input = createInput({1, 3, 224, 224});
    auto output = model->forward(input);

    expectShape(output.tensor(), {1, 10});
    expectDType(output.tensor());
}

TEST_P(ViTMultiDtypeTest, ViTBasePatch16CustomClasses) {
    auto model = ViT_Base_Patch16(100, false, 224);
    convert_model(model);

    auto input = createInput({2, 3, 224, 224});
    auto output = model->forward(input);

    expectShape(output.tensor(), {2, 100});
    expectDType(output.tensor());
}

// ============================================================================
// ViT Base/32 Tests
// ============================================================================

TEST_P(ViTMultiDtypeTest, ViTBasePatch32Config) {
    auto config = ViTConfig::base_patch32(224);

    EXPECT_EQ(config.patch_size, 32);
    EXPECT_EQ(config.num_patches(), 49);  // 224/32 = 7, 7*7 = 49
    EXPECT_EQ(config.seq_length(), 50);   // 49 + 1 (CLS)
}

TEST_P(ViTMultiDtypeTest, ViTBasePatch32ForwardShape) {
    auto model = ViT_Base_Patch32(1000, false, 224);
    convert_model(model);

    auto input = createInput({2, 3, 224, 224});
    auto output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectDType(output.tensor());
}

TEST_P(ViTMultiDtypeTest, ViTBasePatch32GradientFlow) {
    auto model = ViT_Base_Patch32(10, false, 224);
    convert_model(model);
    model->train();

    auto input = createInput({1, 3, 224, 224});
    auto output = model->forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

// ============================================================================
// ViT Large/16 Tests
// ============================================================================

TEST_P(ViTMultiDtypeTest, ViTLargeConfig) {
    auto config = ViTConfig::large_patch16(224);

    EXPECT_EQ(config.image_size, 224);
    EXPECT_EQ(config.patch_size, 16);
    EXPECT_EQ(config.hidden_size, 1024);
    EXPECT_EQ(config.num_hidden_layers, 24);
    EXPECT_EQ(config.num_attention_heads, 16);
    EXPECT_EQ(config.intermediate_size, 4096);
}

TEST_P(ViTMultiDtypeTest, ViTLargePatch16ForwardShape) {
    auto model = ViT_Large_Patch16(1000, false, 224);
    convert_model(model);

    auto input = createInput({2, 3, 224, 224});
    auto output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectDType(output.tensor());
}

TEST_P(ViTMultiDtypeTest, ViTLargePatch16GradientFlow) {
    // Use smaller image for Float64 CUDA to fit in 8GB
    int img_size = getImageSizeForMemory(224, 307'000'000, true);

    auto model = ViT_Large_Patch16(10, false, img_size);
    convert_model(model);
    model->train();

    auto input = createInput({1, 3, img_size, img_size});
    auto output = model->forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

TEST_P(ViTMultiDtypeTest, ViTLargePatch16ParameterCount) {
    auto model = ViT_Large_Patch16(1000, false, 224);
    convert_model(model);
    auto params = model->parameters();

    // ViT-Large should have around 307M parameters
    size_t total_params = countParameters(params);

    // Allow 20% tolerance
    EXPECT_GT(total_params, 250'000'000);
    EXPECT_LT(total_params, 370'000'000);
}

// ============================================================================
// ViT Large/32 Tests
// ============================================================================

TEST_P(ViTMultiDtypeTest, ViTLargePatch32ForwardShape) {
    auto model = ViT_Large_Patch32(1000, false, 224);
    convert_model(model);

    auto input = createInput({2, 3, 224, 224});
    auto output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectDType(output.tensor());
}

// ============================================================================
// ViT Huge/14 Tests
// ============================================================================

TEST_P(ViTMultiDtypeTest, ViTHugeConfig) {
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

TEST_P(ViTMultiDtypeTest, ViTHugePatch14ForwardShape) {
    // Use smaller image for Float64 CUDA to fit in 8GB
    int img_size = getImageSizeForMemory(224, 632'000'000, false);

    auto model = ViT_Huge_Patch14(1000, false, img_size);
    convert_model(model);

    auto input = createInput({1, 3, img_size, img_size});
    auto output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectDType(output.tensor());
}

TEST_P(ViTMultiDtypeTest, ViTHugePatch14GradientFlow) {
    // For CUDA + Float64, use a reduced model (8 layers instead of 32)
    // to fit in 8GB GPU memory (~160M params instead of 632M)
    bool use_reduced_model = (backend_name() == "cuda" && dtype() == DType::Float64);
    int img_size = use_reduced_model ? 112 : getImageSizeForMemory(224, 632'000'000, true);

    std::shared_ptr<ViTForImageClassification> model;
    if (use_reduced_model) {
        // Create reduced ViT-Huge config: same hidden size but 8 layers
        ViTConfig config;
        config.image_size = img_size;
        config.patch_size = 14;
        config.hidden_size = 1280;        // Same as Huge
        config.num_hidden_layers = 8;     // Reduced from 32 to 8
        config.num_attention_heads = 16;  // Same as Huge
        config.intermediate_size = 5120;  // Same as Huge
        model = std::make_shared<ViTForImageClassification>(config, 10);
    } else {
        model = ViT_Huge_Patch14(10, false, img_size);
    }

    convert_model(model);
    model->train();

    auto input = createInput({1, 3, img_size, img_size});
    auto output = model->forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

TEST_P(ViTMultiDtypeTest, ViTHugePatch14ParameterCount) {
    auto model = ViT_Huge_Patch14(1000, false, 224);
    convert_model(model);
    auto params = model->parameters();

    // ViT-Huge should have around 632M parameters
    size_t total_params = countParameters(params);

    // Allow 20% tolerance
    EXPECT_GT(total_params, 500'000'000);
    EXPECT_LT(total_params, 760'000'000);
}

// ============================================================================
// ViT Huge/16 Tests
// ============================================================================

TEST_P(ViTMultiDtypeTest, ViTHugePatch16ForwardShape) {
    // Use smaller image for Float64 CUDA to fit in 8GB
    int img_size = getImageSizeForMemory(224, 632'000'000, false);

    auto model = ViT_Huge_Patch16(1000, false, img_size);
    convert_model(model);

    auto input = createInput({1, 3, img_size, img_size});
    auto output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectDType(output.tensor());
}

// ============================================================================
// Different Image Sizes Tests
// ============================================================================

TEST_P(ViTMultiDtypeTest, ViTBaseDifferentImageSize384) {
    // Test with 384x384 input (commonly used for fine-tuning)
    auto model = ViT_Base_Patch16(1000, false, 384);
    convert_model(model);

    auto input = createInput({1, 3, 384, 384});
    auto output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectDType(output.tensor());
}

TEST_P(ViTMultiDtypeTest, ViTBaseDifferentImageSize512) {
    // Use smaller image for Float64 CUDA to fit in 8GB (384 instead of 512)
    int img_size = getImageSizeForMemory(512, 86'000'000, false);

    auto model = ViT_Base_Patch16(1000, false, img_size);
    convert_model(model);

    auto input = createInput({1, 3, img_size, img_size});
    auto output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectDType(output.tensor());

    // Verify number of patches based on image size
    auto config = ViTConfig::base_patch16(img_size);
    int expected_patches = (img_size / 16) * (img_size / 16);
    EXPECT_EQ(config.num_patches(), expected_patches);
}

// ============================================================================
// Config Tests (dtype-independent but comprehensive)
// ============================================================================

TEST_P(ViTMultiDtypeTest, ViTConfigNumPatchesCalculation) {
    auto config_16 = ViTConfig::base_patch16(224);
    EXPECT_EQ(config_16.num_patches(), 196);

    auto config_32 = ViTConfig::base_patch32(224);
    EXPECT_EQ(config_32.num_patches(), 49);

    auto config_14 = ViTConfig::huge_patch14(224);
    EXPECT_EQ(config_14.num_patches(), 256);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST_P(ViTMultiDtypeTest, ViTBaseLargeBatchSize) {
    auto model = ViT_Base_Patch16(10, false, 224);
    convert_model(model);

    auto input = createInput({8, 3, 224, 224});
    auto output = model->forward(input);

    expectShape(output.tensor(), {8, 10});
    expectDType(output.tensor());
}

TEST_P(ViTMultiDtypeTest, ViTBaseSingleChannel) {
    // Test with grayscale input (1 channel)
    auto patch_embed = std::make_shared<PatchEmbedding>(224, 16, 1, 768);
    convert_model(patch_embed);

    auto input = createInput({2, 1, 224, 224});
    auto output = patch_embed->forward(input);

    expectShape(output.tensor(), {2, 196, 768});
    expectDType(output.tensor());
}

TEST_P(ViTMultiDtypeTest, ViTBaseMultiChannel) {
    // Test with hyperspectral input (4 channels)
    auto patch_embed = std::make_shared<PatchEmbedding>(224, 16, 4, 768);
    convert_model(patch_embed);

    auto input = createInput({2, 4, 224, 224});
    auto output = patch_embed->forward(input);

    expectShape(output.tensor(), {2, 196, 768});
    expectDType(output.tensor());
}

// ============================================================================
// Training Mode Tests
// ============================================================================

TEST_P(ViTMultiDtypeTest, ViTBaseTrainEvalMode) {
    auto model = ViT_Base_Patch16(10, false, 224);
    convert_model(model);

    auto input = createInput({2, 3, 224, 224});

    // Test in training mode
    model->train();
    auto output_train = model->forward(input);
    expectDType(output_train.tensor());

    // Test in evaluation mode
    model->eval();
    auto output_eval = model->forward(input);
    expectDType(output_eval.tensor());

    // Both should have same shape
    auto shape_train = output_train.tensor().shape();
    auto shape_eval = output_eval.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_train.begin(), shape_train.end()),
              std::vector<int64_t>(shape_eval.begin(), shape_eval.end()));
}

// ============================================================================
// Instantiate Tests for Each Backend + DType Combination
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ViTMultiDtypeTest);

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();  // Initialize Tenzor library and backends
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
