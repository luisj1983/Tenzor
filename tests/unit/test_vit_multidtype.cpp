/**
 * @file test_vit_multidtype.cpp
 * @brief Multi-dtype tests for Vision Transformer (ViT) variants
 *
 * Tests ViT models with Float32, Float64, and Float16 data types to ensure:
 * - Proper dtype propagation through patch embedding, position embeddings, and transformers
 * - Correct output shapes across all ViT variants (Tiny, Small, Base, Large, Huge)
 * - Gradient flow with different dtypes
 * - Class token handling across dtypes
 * - Different image sizes and patch sizes
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/vit.hpp"
#include "../../include/tenzor/core/tensor.hpp"
#include "../../include/tenzor/autograd/variable.hpp"
#include <tuple>
#include <vector>

using namespace tenzor;
using namespace tenzor::models;

// ============================================================================
// Parameterized Test Fixture
// ============================================================================

class ViTMultiDtypeTest : public ::testing::TestWithParam<DType> {
protected:
    void SetUp() override {
        device_ = Device::cpu();
        dtype_ = GetParam();

        // Set tolerance based on dtype
        if (dtype_ == DType::Float16) {
            rtol_ = 1e-2f;
            atol_ = 1e-2f;
        } else if (dtype_ == DType::Float32) {
            rtol_ = 1e-4f;
            atol_ = 1e-5f;
        } else {  // Float64
            rtol_ = 1e-6f;
            atol_ = 1e-7f;
        }
    }

    Device device_;
    DType dtype_;
    float rtol_;
    float atol_;

    // Helper to create input tensor with current dtype
    Variable createInput(const std::vector<int64_t>& shape, bool requires_grad = true) {
        return Variable(Tensor(shape, dtype_, device_), requires_grad);
    }

    // Helper to count total parameters
    size_t countParameters(const std::vector<std::shared_ptr<Variable>>& params) {
        size_t total = 0;
        for (const auto& p : params) {
            size_t param_size = 1;
            for (auto dim : p->tensor().shape()) {
                param_size *= dim;
            }
            total += param_size;
        }
        return total;
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

    Variable input = createInput({2, num_channels, image_size, image_size});
    Variable output = patch_embed->forward(input);

    // Expected: (batch, num_patches, hidden_size)
    // num_patches = (224/16) * (224/16) = 196
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 196, 768}));

    // Verify output dtype matches input
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ViTMultiDtypeTest, PatchEmbeddingGradientFlow) {
    auto patch_embed = std::make_shared<PatchEmbedding>(224, 16, 3, 768);

    Variable input = createInput({1, 3, 224, 224});
    Variable output = patch_embed->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype_);

    auto params = patch_embed->parameters();
    EXPECT_GT(params.size(), 0);

    // Check gradient dtype for parameters
    for (const auto& param : params) {
        if (param->grad().has_value()) {
            EXPECT_EQ(param->grad()->dtype(), dtype_);
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
    Variable input_14 = createInput({1, 3, 224, 224});
    Variable output_14 = patch_embed_14->forward(input_14);

    auto shape_14 = output_14.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_14.begin(), shape_14.end()),
              (std::vector<int64_t>{1, 256, 1280}));
    EXPECT_EQ(output_14.tensor().dtype(), dtype_);

    // Test patch size 32
    auto patch_embed_32 = std::make_shared<PatchEmbedding>(224, 32, 3, 768);
    Variable input_32 = createInput({1, 3, 224, 224});
    Variable output_32 = patch_embed_32->forward(input_32);

    auto shape_32 = output_32.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_32.begin(), shape_32.end()),
              (std::vector<int64_t>{1, 49, 768}));
    EXPECT_EQ(output_32.tensor().dtype(), dtype_);
}

// ============================================================================
// ViTEmbeddings Tests (Patch + Position + Class Token)
// ============================================================================

TEST_P(ViTMultiDtypeTest, ViTEmbeddingsForwardShape) {
    auto config = ViTConfig::base_patch16(224);
    auto embeddings = std::make_shared<ViTEmbeddings>(config);

    Variable input = createInput({2, 3, 224, 224});
    Variable output = embeddings->forward(input);

    // Expected: (batch, num_patches + 1 (CLS), hidden_size)
    // num_patches = 196, so seq_len = 197
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 197, 768}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ViTMultiDtypeTest, ViTEmbeddingsGradientFlow) {
    auto config = ViTConfig::base_patch16(224);
    auto embeddings = std::make_shared<ViTEmbeddings>(config);

    Variable input = createInput({1, 3, 224, 224});
    Variable output = embeddings->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype_);
}

TEST_P(ViTMultiDtypeTest, ViTEmbeddingsClassToken) {
    auto config = ViTConfig::base_patch16(224);
    auto embeddings = std::make_shared<ViTEmbeddings>(config);

    Variable input = createInput({2, 3, 224, 224});
    Variable output = embeddings->forward(input);

    // Verify class token is added (seq_len should be num_patches + 1)
    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[1], 197);  // 196 patches + 1 class token
    EXPECT_EQ(output.tensor().dtype(), dtype_);
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

    Variable input = createInput({2, 3, 224, 224});
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ViTMultiDtypeTest, ViTBasePatch16GradientFlow) {
    auto model = ViT_Base_Patch16(10, false, 224);
    model->train();

    Variable input = createInput({1, 3, 224, 224});
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype_);

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(ViTMultiDtypeTest, ViTBasePatch16ParameterCount) {
    auto model = ViT_Base_Patch16(1000, false, 224);
    auto params = model->parameters();

    // ViT-Base should have around 86M parameters
    size_t total_params = countParameters(params);

    // Allow 20% tolerance
    EXPECT_GT(total_params, 70'000'000);
    EXPECT_LT(total_params, 100'000'000);
}

TEST_P(ViTMultiDtypeTest, ViTBasePatch16BatchSizeOne) {
    auto model = ViT_Base_Patch16(10, false, 224);

    Variable input = createInput({1, 3, 224, 224});
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 10}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ViTMultiDtypeTest, ViTBasePatch16CustomClasses) {
    auto model = ViT_Base_Patch16(100, false, 224);

    Variable input = createInput({2, 3, 224, 224});
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 100}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
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

    Variable input = createInput({2, 3, 224, 224});
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ViTMultiDtypeTest, ViTBasePatch32GradientFlow) {
    auto model = ViT_Base_Patch32(10, false, 224);
    model->train();

    Variable input = createInput({1, 3, 224, 224});
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype_);
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

    Variable input = createInput({2, 3, 224, 224});
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ViTMultiDtypeTest, ViTLargePatch16GradientFlow) {
    auto model = ViT_Large_Patch16(10, false, 224);
    model->train();

    Variable input = createInput({1, 3, 224, 224});
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype_);
}

TEST_P(ViTMultiDtypeTest, ViTLargePatch16ParameterCount) {
    auto model = ViT_Large_Patch16(1000, false, 224);
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

    Variable input = createInput({2, 3, 224, 224});
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
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
    auto model = ViT_Huge_Patch14(1000, false, 224);

    Variable input = createInput({1, 3, 224, 224});
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 1000}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ViTMultiDtypeTest, ViTHugePatch14GradientFlow) {
    auto model = ViT_Huge_Patch14(10, false, 224);
    model->train();

    Variable input = createInput({1, 3, 224, 224});
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype_);
}

TEST_P(ViTMultiDtypeTest, ViTHugePatch14ParameterCount) {
    auto model = ViT_Huge_Patch14(1000, false, 224);
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
    auto model = ViT_Huge_Patch16(1000, false, 224);

    Variable input = createInput({1, 3, 224, 224});
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 1000}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

// ============================================================================
// Different Image Sizes Tests
// ============================================================================

TEST_P(ViTMultiDtypeTest, ViTBaseDifferentImageSize384) {
    // Test with 384x384 input (commonly used for fine-tuning)
    auto model = ViT_Base_Patch16(1000, false, 384);

    Variable input = createInput({1, 3, 384, 384});
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 1000}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ViTMultiDtypeTest, ViTBaseDifferentImageSize512) {
    // Test with 512x512 input
    auto model = ViT_Base_Patch16(1000, false, 512);

    Variable input = createInput({1, 3, 512, 512});
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 1000}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);

    // Verify number of patches: 512/16 = 32, 32*32 = 1024 patches
    auto config = ViTConfig::base_patch16(512);
    EXPECT_EQ(config.num_patches(), 1024);
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

    Variable input = createInput({8, 3, 224, 224});
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{8, 10}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ViTMultiDtypeTest, ViTBaseSingleChannel) {
    // Test with grayscale input (1 channel)
    auto patch_embed = std::make_shared<PatchEmbedding>(224, 16, 1, 768);

    Variable input = createInput({2, 1, 224, 224});
    Variable output = patch_embed->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 196, 768}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ViTMultiDtypeTest, ViTBaseMultiChannel) {
    // Test with hyperspectral input (4 channels)
    auto patch_embed = std::make_shared<PatchEmbedding>(224, 16, 4, 768);

    Variable input = createInput({2, 4, 224, 224});
    Variable output = patch_embed->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 196, 768}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

// ============================================================================
// Training Mode Tests
// ============================================================================

TEST_P(ViTMultiDtypeTest, ViTBaseTrainEvalMode) {
    auto model = ViT_Base_Patch16(10, false, 224);

    Variable input = createInput({2, 3, 224, 224});

    // Test in training mode
    model->train();
    Variable output_train = model->forward(input);
    EXPECT_EQ(output_train.tensor().dtype(), dtype_);

    // Test in evaluation mode
    model->eval();
    Variable output_eval = model->forward(input);
    EXPECT_EQ(output_eval.tensor().dtype(), dtype_);

    // Both should have same shape
    auto shape_train = output_train.tensor().shape();
    auto shape_eval = output_eval.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_train.begin(), shape_train.end()),
              std::vector<int64_t>(shape_eval.begin(), shape_eval.end()));
}

// ============================================================================
// Instantiate Tests for Each DType
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    MultiDType,
    ViTMultiDtypeTest,
    ::testing::Values(
        DType::Float32,
        DType::Float64,
        DType::Float16
    ),
    [](const ::testing::TestParamInfo<DType>& info) {
        std::string name;
        switch (info.param) {
            case DType::Float32: name = "Float32"; break;
            case DType::Float64: name = "Float64"; break;
            case DType::Float16: name = "Float16"; break;
            default: name = "Unknown"; break;
        }
        return name;
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
