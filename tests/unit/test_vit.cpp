/**
 * @file test_vit.cpp
 * @brief Comprehensive tests for Vision Transformer (ViT) variants
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/offload.hpp>
#include <optional>
#include "../../include/tenzor/models/vit.hpp"
#include "../../include/tenzor/core/tensor.hpp"
#include "../../include/tenzor/autograd/variable.hpp"
#include "../backend_test_fixture.hpp"
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::models;

class ViTTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ============================================================================
// PatchEmbedding Tests
// ============================================================================

TEST_P(ViTTest, PatchEmbeddingForwardShape) {
    int64_t image_size = 224;
    int64_t patch_size = 16;
    int64_t num_channels = 3;
    int64_t hidden_size = 768;

    auto patch_embed = std::make_shared<PatchEmbedding>(
        image_size, patch_size, num_channels, hidden_size);
    patch_embed->to(device);

    Variable input(randn({2, num_channels, image_size, image_size},
                         DType::Float32, device), true);
    Variable output = patch_embed->forward(input);

    // Expected: (batch, num_patches, hidden_size)
    // num_patches = (224/16) * (224/16) = 196
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 196, 768}));
}

TEST_P(ViTTest, PatchEmbeddingGradientFlow) {
    auto patch_embed = std::make_shared<PatchEmbedding>(224, 16, 3, 768);
    patch_embed->to(device);

    Variable input(randn({1, 3, 224, 224}, DType::Float32, device), true);
    Variable output = patch_embed->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    auto params = patch_embed->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(ViTTest, PatchEmbeddingNumPatches) {
    auto patch_embed = std::make_shared<PatchEmbedding>(224, 16, 3, 768);
    EXPECT_EQ(patch_embed->num_patches(), 196);

    auto patch_embed_32 = std::make_shared<PatchEmbedding>(224, 32, 3, 768);
    EXPECT_EQ(patch_embed_32->num_patches(), 49);
}

// ============================================================================
// ViTEmbeddings Tests
// ============================================================================

TEST_P(ViTTest, ViTEmbeddingsForwardShape) {
    auto config = ViTConfig::base_patch16(224);
    auto embeddings = std::make_shared<ViTEmbeddings>(config);
    embeddings->to(device);

    Variable input(randn({2, 3, 224, 224}, DType::Float32, device), true);
    Variable output = embeddings->forward(input);

    // Expected: (batch, num_patches + 1 (CLS), hidden_size)
    // num_patches = 196, so seq_len = 197
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 197, 768}));
}

TEST_P(ViTTest, ViTEmbeddingsGradientFlow) {
    auto config = ViTConfig::base_patch16(224);
    auto embeddings = std::make_shared<ViTEmbeddings>(config);
    embeddings->to(device);

    Variable input(randn({1, 3, 224, 224}, DType::Float32, device), true);
    Variable output = embeddings->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

// ============================================================================
// ViT Base/16 Tests
// ============================================================================

TEST_P(ViTTest, ViTBaseConfig) {
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

TEST_P(ViTTest, ViTBasePatch16ForwardShape) {
    auto model = ViT_Base_Patch16(1000, false, 224);
    model->to(device);

    Variable input(randn({2, 3, 224, 224}, DType::Float32, device), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
}

TEST_P(ViTTest, ViTBasePatch16GradientFlow) {
    auto model = ViT_Base_Patch16(10, false, 224);
    model->to(device);
    model->train();

    Variable input(randn({1, 3, 224, 224}, DType::Float32, device), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(ViTTest, ViTBasePatch16ParameterCount) {
    auto model = ViT_Base_Patch16(1000, false, 224);
    model->to(device);
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

TEST_P(ViTTest, ViTBasePatch32Config) {
    auto config = ViTConfig::base_patch32(224);

    EXPECT_EQ(config.patch_size, 32);
    EXPECT_EQ(config.num_patches(), 49);  // 224/32 = 7, 7*7 = 49
    EXPECT_EQ(config.seq_length(), 50);   // 49 + 1 (CLS)
}

TEST_P(ViTTest, ViTBasePatch32ForwardShape) {
    auto model = ViT_Base_Patch32(1000, false, 224);
    model->to(device);

    Variable input(randn({2, 3, 224, 224}, DType::Float32, device), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
}

TEST_P(ViTTest, ViTBasePatch32GradientFlow) {
    auto model = ViT_Base_Patch32(10, false, 224);
    model->to(device);
    model->train();

    Variable input(randn({1, 3, 224, 224}, DType::Float32, device), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

// ============================================================================
// ViT Large/16 Tests
// ============================================================================

TEST_P(ViTTest, ViTLargeConfig) {
    auto config = ViTConfig::large_patch16(224);

    EXPECT_EQ(config.image_size, 224);
    EXPECT_EQ(config.patch_size, 16);
    EXPECT_EQ(config.hidden_size, 1024);
    EXPECT_EQ(config.num_hidden_layers, 24);
    EXPECT_EQ(config.num_attention_heads, 16);
    EXPECT_EQ(config.intermediate_size, 4096);
}

TEST_P(ViTTest, ViTLargePatch16ForwardShape) {
    auto model = ViT_Large_Patch16(1000, false, 224);
    model->to(device);

    Variable input(randn({2, 3, 224, 224}, DType::Float32, device), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
}

TEST_P(ViTTest, ViTLargePatch16GradientFlow) {
    auto model = ViT_Large_Patch16(10, false, 224);
    model->to(device);
    model->train();

    Variable input(randn({1, 3, 224, 224}, DType::Float32, device), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

TEST_P(ViTTest, ViTLargePatch16ParameterCount) {
    auto model = ViT_Large_Patch16(1000, false, 224);
    model->to(device);
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

TEST_P(ViTTest, ViTLargePatch32ForwardShape) {
    auto model = ViT_Large_Patch32(1000, false, 224);
    model->to(device);

    Variable input(randn({2, 3, 224, 224}, DType::Float32, device), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
}

// ============================================================================
// ViT Huge/14 Tests
// ============================================================================

TEST_P(ViTTest, ViTHugeConfig) {
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

TEST_P(ViTTest, ViTHugePatch14ForwardShape) {
    auto model = ViT_Huge_Patch14(1000, false, 224);
    model->to(device);

    // Forward-only shape check: run under no_grad so the autograd engine does
    // not retain all 32 encoder layers' activations for a backward that never
    // happens. ViT-Huge/14 is ~632M params (~2.5GB FP32); retaining the full
    // graph on top of the weights OOMs an 8GB laptop GPU even at batch 1.
    // no_grad is the correct inference pattern for a shape check (mirrors
    // torch.no_grad()), not a workaround.
    NoGradGuard no_grad;
    Variable input(randn({1, 3, 224, 224}, DType::Float32, device), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 1000}));
}

TEST_P(ViTTest, ViTHugePatch14GradientFlow) {
    auto model = ViT_Huge_Patch14(10, false, 224);
    model->train();
    // ViT-Huge/14 (~632M params) forward+backward in Float32 exceeds an 8GB
    // GPU when every layer's activations are retained. Two memory techniques
    // make the full model fit:
    //   - Gradient checkpointing (set_gradient_checkpointing): each encoder
    //     layer recomputes its activations during backward instead of storing
    //     them, dropping activation memory from O(num_layers) to O(1).
    //     Gradients are identical (checkpoint captures/replays RNG), so the
    //     grad-flow check still validates the real model.
    //   - ZeRO-Phase-2 offload (cuda only): on the 8GB CUDA device the 632M
    //     Float32 params + grads alone (~5GB) plus one checkpointed layer's
    //     working set still OOMs under checkpointing alone. OffloadContext
    //     keeps params and grads on the CPU host and stages each layer onto
    //     the GPU just-in-time (prefetch_depth=2), so peak GPU memory is one
    //     layer's working set. Compute stays on the GPU — this is the
    //     library's documented ZeRO-Phase-2 path (nn/offload.hpp), not a CPU
    //     compute fallback. CPU/ROCm/OneAPI/Vulkan have enough device memory
    //     for the full model under checkpointing alone, so they take the
    //     plain model->to(device) path.
    model->set_gradient_checkpointing(true);

    std::optional<nn::OffloadContext> offload_ctx;
    if (device.type == Device::Type::CUDA) {
        nn::OffloadContext::Config config;
        config.offload_parameters = true;
        config.offload_gradients = true;
        config.prefetch_depth = 2;
        config.pin_first_layer = true;
        config.pin_last_layer = true;
        config.target_device = device;
        // The model is destroyed together with this OffloadContext at end of
        // test, so the default restore-at-shutdown is pointless — and for a
        // model offloaded precisely because it exceeds device capacity it
        // OOMs anyway (see warnings above). Worse, the restored GPU tensors
        // are freed only when the model destructs, which is *after* the
        // context's TransferEngine (and its CUDA streams) are torn down,
        // stranding the freed CachingAllocator blocks' completion events on
        // dead streams. Those blocks become permanently un-reusable and
        // un-evictable, so the next test in this process (e.g.
        // ViTHugePatch16ForwardShape) OOMs on a device full of stuck cache.
        config.restore_on_destruction = false;
        offload_ctx.emplace(*model, config);
        offload_ctx->enable();
    } else {
        model->to(device);
    }

    Variable input(randn({1, 3, 224, 224}, DType::Float32, device), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

TEST_P(ViTTest, ViTHugePatch14ParameterCount) {
    auto model = ViT_Huge_Patch14(1000, false, 224);
    model->to(device);
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

TEST_P(ViTTest, ViTHugePatch16ForwardShape) {
    auto model = ViT_Huge_Patch16(1000, false, 224);
    model->to(device);

    Variable input(randn({1, 3, 224, 224}, DType::Float32, device), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 1000}));
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_P(ViTTest, ViTBaseBatchSizeOne) {
    auto model = ViT_Base_Patch16(10, false, 224);
    model->to(device);

    Variable input(randn({1, 3, 224, 224}, DType::Float32, device), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 10}));
}

TEST_P(ViTTest, ViTBaseCustomClasses) {
    auto model = ViT_Base_Patch16(100, false, 224);
    model->to(device);

    Variable input(randn({2, 3, 224, 224}, DType::Float32, device), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 100}));
}

TEST_P(ViTTest, ViTBaseDifferentImageSize) {
    // Test with 384x384 input (commonly used for fine-tuning)
    auto model = ViT_Base_Patch16(1000, false, 384);
    model->to(device);

    Variable input(randn({1, 3, 384, 384}, DType::Float32, device), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 1000}));
}

TEST_P(ViTTest, PatchEmbeddingDifferentPatchSizes) {
    // Test patch size 14
    auto patch_embed_14 = std::make_shared<PatchEmbedding>(224, 14, 3, 1280);
    patch_embed_14->to(device);
    Variable input_14(randn({1, 3, 224, 224}, DType::Float32, device), true);
    Variable output_14 = patch_embed_14->forward(input_14);

    auto shape_14 = output_14.tensor().shape();
    // 224/14 = 16, 16*16 = 256 patches
    EXPECT_EQ(std::vector<int64_t>(shape_14.begin(), shape_14.end()),
              (std::vector<int64_t>{1, 256, 1280}));

    // Test patch size 32
    auto patch_embed_32 = std::make_shared<PatchEmbedding>(224, 32, 3, 768);
    patch_embed_32->to(device);
    Variable input_32(randn({1, 3, 224, 224}, DType::Float32, device), true);
    Variable output_32 = patch_embed_32->forward(input_32);

    auto shape_32 = output_32.tensor().shape();
    // 224/32 = 7, 7*7 = 49 patches
    EXPECT_EQ(std::vector<int64_t>(shape_32.begin(), shape_32.end()),
              (std::vector<int64_t>{1, 49, 768}));
}

TEST_P(ViTTest, ViTConfigNumPatchesCalculation) {
    auto config_16 = ViTConfig::base_patch16(224);
    EXPECT_EQ(config_16.num_patches(), 196);

    auto config_32 = ViTConfig::base_patch32(224);
    EXPECT_EQ(config_32.num_patches(), 49);

    auto config_14 = ViTConfig::huge_patch14(224);
    EXPECT_EQ(config_14.num_patches(), 256);
}

INSTANTIATE_BACKEND_TESTS(ViTTest);
