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
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/math.hpp>
#include "../../include/tenzor/models/vit.hpp"
#include "../../include/tenzor/core/tensor.hpp"
#include "../../include/tenzor/autograd/variable.hpp"
#include "../../include/tenzor/autograd/function.hpp"
#include "../multi_backend_dtype_fixture.hpp"
#include <tuple>
#include <vector>
#include <cmath>
#include "../grad_flow_helpers.hpp"

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
        (void)patch_size;
        if (backend_name() != "cuda" && backend_name() != "vulkan") return default_size;

        bool is_float64 = (dtype() == DType::Float64);
        // Vulkan uses unfused attention (separate BMM ops) while CUDA uses cuDNN SDPA,
        // so Vulkan needs more memory for attention intermediates in the autograd graph.
        bool is_vulkan = (backend_name() == "vulkan");
        // Vulkan F16/BF16 widens to F32 inside several ops (LayerNorm/SDPA/...) and
        // also stores activations packed into F32 words for the autograd graph, so
        // memory usage tracks F32, not F16. Treat them like F32 here.
        bool is_vulkan_half = (is_vulkan && (dtype() == DType::Float16 ||
                                             dtype() == DType::BFloat16));

        // ViT-Huge (~632M params) with Float64 + gradients needs very small images
        if (param_count > 500'000'000 && needs_gradients && is_float64) {
            return 56;  // 56 is divisible by 14, gives 4x4=16 patches
        }
        // ViT-Huge (~632M params) with Float32 + gradients
        if (param_count > 500'000'000 && needs_gradients) {
            return is_vulkan ? 56 : 112;  // 112 is divisible by 14 and 16
        }
        // ViT-Huge forward-only with Float64 — Vulkan's 8 GB VRAM can't hold
        // a 5 GB Float64 weight set plus activations. Pick a per-patch-size
        // image dim that gives a small-enough sequence length:
        //   * Patch14: 56 (=4 patches/side, seq_len=17)
        //   * Patch16: 64 (=4 patches/side, seq_len=17)
        // Both leave headroom for the 32-layer attention working set.
        if (param_count > 500'000'000 && is_float64) {
            if (is_vulkan) return (patch_size == 14) ? 56 : 64;
            return 112;
        }
        // ViT-Huge forward-only with Float16/BFloat16 on Vulkan: still hits OOM at 224
        // because the attention intermediates are stored in F32 representation.
        if (param_count > 500'000'000 && is_vulkan_half) {
            return 112;
        }
        // ViT-Large with gradients and Float64
        // Vulkan needs smaller images due to unfused attention intermediates
        if (param_count > 200'000'000 && needs_gradients && is_float64) {
            return is_vulkan ? 64 : 160;  // 64 divisible by 16, smaller for Vulkan F64 memory
        }
        // ViT-Large with gradients on Vulkan F16/BF16/F32: 224 is fine for CUDA's
        // fused SDPA but Vulkan's unfused-attention activation graph blows the
        // 8 GB device budget. Halve for Vulkan.
        if (param_count > 200'000'000 && needs_gradients && is_vulkan) {
            return 112;
        }
        // ViT-Large forward-only with Float64 on Vulkan: 307M * 8B = 2.5 GB
        // weights + 224x224 activations exceeds 8 GB. Reduce to 128.
        if (param_count > 200'000'000 && is_float64 && is_vulkan) {
            return 128;
        }
        // Large image sizes (512+) need reduction with Float64
        if (default_size >= 512 && is_float64) {
            return 384;  // Divisible by 16
        }
        return default_size;
    }

    /**
     * @brief Pick a batch size that fits in 8 GB GPU memory for parametrized
     * "large batch" tests. Returns `default_batch` on CPU/oneapi/rocm; on
     * memory-constrained CUDA/Vulkan it shrinks the batch when the model is
     * big enough to push activations past the device budget.
     */
    int getBatchSizeForMemory(int default_batch, size_t param_count) const {
        if (backend_name() != "cuda" && backend_name() != "vulkan") return default_batch;
        bool is_float64 = (dtype() == DType::Float64);
        bool is_vulkan = (backend_name() == "vulkan");
        if (param_count > 50'000'000 && is_float64) {
            // ViT-Base: 86M F64 params x 8 B = 700 MB; 8 batches at 224x224
            // pushes a 32-layer attention graph past 8 GB. Halve.
            return std::max(1, default_batch / 2);
        }
        if (param_count > 50'000'000 && is_vulkan) {
            // Vulkan stores half activations in F32-equivalent buffers; 8x224
            // ViT-Base activations ~6 GB. Quarter on Vulkan to be safe.
            return std::max(1, default_batch / 4);
        }
        return default_batch;
    }

    // Audit-T.1: cheap "the backend produced something meaningful" check
    // that's safe to apply to every ViT/PatchEmbedding forward output —
    // even the 600M-param ones where running a CPU reference would be
    // prohibitively expensive. We assert:
    //   * no NaN/Inf anywhere
    //   * not all-zero (would silently pass on an uninitialised buffer)
    //   * absolute max under a generous ceiling (catches garbage 1e20 values)
    void expectOutputSane(const Tensor& output, float max_ceiling = 1.0e6f) {
        auto cpu = output.to(Device::cpu());
        if (cpu.dtype() != DType::Float32) cpu = cpu.to(DType::Float32);
        cpu = cpu.contiguous();
        const float* p = cpu.data<float>();
        bool any_finite_nonzero = false;
        float max_abs = 0.0f;
        for (int64_t i = 0; i < cpu.numel(); ++i) {
            float v = p[i];
            ASSERT_FALSE(std::isnan(v))
                << "NaN at index " << i << " on " << device().to_string()
                << " / dtype " << static_cast<int>(dtype());
            ASSERT_FALSE(std::isinf(v))
                << "Inf at index " << i << " on " << device().to_string()
                << " / dtype " << static_cast<int>(dtype());
            if (v != 0.0f) any_finite_nonzero = true;
            max_abs = std::max(max_abs, std::abs(v));
        }
        EXPECT_TRUE(any_finite_nonzero)
            << "Output is identically zero on " << device().to_string()
            << " — uninitialised buffer?";
        EXPECT_LT(max_abs, max_ceiling)
            << "Output has absurd magnitude " << max_abs
            << " on " << device().to_string()
            << " — likely uninitialised or NaN-poisoned buffer";
    }

    // Audit-T.1: stricter check — match a CPU reference computed by
    // running the same op on a Float32 CPU clone of the input.
    void expectMatchesCpu(const Tensor& actual, const Tensor& cpu_ref,
                          float tol) {
        auto actual_cpu = actual.to(Device::cpu()).to(DType::Float32).contiguous();
        auto ref = cpu_ref.to(DType::Float32).contiguous();
        ASSERT_EQ(actual_cpu.numel(), ref.numel());
        const float* a = actual_cpu.data<float>();
        const float* r = ref.data<float>();
        for (int64_t i = 0; i < actual_cpu.numel(); ++i)
            EXPECT_NEAR(a[i], r[i], tol)
                << "Mismatch at index " << i
                << " on " << device().to_string();
    }

    // audit-4 U.16: helper that builds {CPU/Float32 reference output,
    // parameterized-backend input} for the small-config ViT TEST_Ps.
    //
    // Pattern: the model is freshly constructed in CPU/Float32 state. We:
    //   1. create a CPU/Float32 input (seeded once),
    //   2. run the model on CPU/F32 to capture a deterministic reference,
    //   3. convert_model() to the parameterized backend/dtype,
    //   4. produce the device/dtype counterpart of the same input.
    //
    // Returns (cpu_ref_output, device_input). Caller runs the device forward
    // on `device_input` and asserts `expectMatchesCpu(actual, cpu_ref_output,
    // vitAtol())`. This replaces the loose `expectOutputSane` ceiling for
    // small-config models where a CPU reference is cheap.
    template <typename ModelT, typename ForwardFn>
    std::pair<Tensor, Variable> captureCpuReferenceAndConvert(
            ModelT& model,
            const std::vector<int64_t>& input_shape,
            ForwardFn&& forward_fn,
            bool input_requires_grad = true) {
        // Step 1: CPU/Float32 input.
        auto cpu_input_t = tenzor::randn(input_shape, DType::Float32, Device::cpu());
        Variable cpu_input(cpu_input_t, input_requires_grad);

        // Step 2: CPU reference output — model is fresh CPU/Float32 here.
        Tensor cpu_ref = forward_fn(model, cpu_input).tensor()
                              .to(DType::Float32).contiguous();

        // Step 3: migrate model to the parameterized backend/dtype.
        convert_model(model);

        // Step 4: same input, moved to (device_, dtype_).
        auto dev_t = cpu_input_t.to(device_);
        if (dtype_ != DType::Float32) dev_t = dev_t.to(dtype_);
        Variable device_input(dev_t, input_requires_grad);

        return {std::move(cpu_ref), std::move(device_input)};
    }

    // Audit-T.1: tolerance picked per dtype for the device-vs-CPU
    // PatchEmbedding diff after the Float32 round-trip.
    float vitAtol() const {
        switch (dtype()) {
            case DType::Float16:
            case DType::BFloat16:
                return 1.0e-1f;
            case DType::Float64:
                return 1.0e-5f;
            default:
                return 5.0e-3f;
        }
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
    // Audit-T.1: sanity-check values (no NaN/Inf, not all-zero, sane scale).
    expectOutputSane(output.tensor());
}

TEST_P(ViTMultiDtypeTest, PatchEmbeddingGradientFlow) {
    auto patch_embed = std::make_shared<PatchEmbedding>(224, 16, 3, 768);
    convert_model(patch_embed);

    auto input = createInput({1, 3, 224, 224});
    auto output = patch_embed->forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
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
    expectOutputSane(output_14.tensor());  // audit-T.1

    // Test patch size 32
    auto patch_embed_32 = std::make_shared<PatchEmbedding>(224, 32, 3, 768);
    convert_model(patch_embed_32);
    auto input_32 = createInput({1, 3, 224, 224});
    auto output_32 = patch_embed_32->forward(input_32);

    expectShape(output_32.tensor(), {1, 49, 768});
    expectDType(output_32.tensor());
    expectOutputSane(output_32.tensor());  // audit-T.1
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
    expectOutputSane(output.tensor());  // audit-T.1
}

TEST_P(ViTMultiDtypeTest, ViTEmbeddingsGradientFlow) {
    auto config = ViTConfig::base_patch16(224);
    auto embeddings = std::make_shared<ViTEmbeddings>(config);
    convert_model(embeddings);

    auto input = createInput({1, 3, 224, 224});
    auto output = embeddings->forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
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

    // Audit-T.1: the prepended CLS token row (position 0 in dim 1) must
    // not be identically zero — a backend that silently allocates an
    // empty CLS slot would pass shape checks but produce useless
    // downstream features. Pull row 0 out and assert it has any nonzero
    // entry.
    auto cls_row = output.tensor().to(Device::cpu()).to(DType::Float32)
        .contiguous();
    const float* p = cls_row.data<float>();
    // Layout (B=2, S=197, H=768): the CLS row of sample 0 occupies the
    // first 768 floats.
    bool any_nonzero = false;
    for (int i = 0; i < 768; ++i)
        if (p[i] != 0.0f) { any_nonzero = true; break; }
    EXPECT_TRUE(any_nonzero) << "CLS token row is identically zero on "
                             << device().to_string();
    expectOutputSane(output.tensor());
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
    // audit-4 U.16: small-config TEST_P — diff against CPU/Float32 reference
    // instead of the loose `expectOutputSane` ceiling. The helper runs the
    // CPU forward BEFORE `convert_model` migrates the weights.
    auto [cpu_ref, input] = captureCpuReferenceAndConvert(
        model, {2, 3, 224, 224},
        [](auto& m, const Variable& x) { return m->forward(x); });

    auto output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectDType(output.tensor());
    expectMatchesCpu(output.tensor(), cpu_ref, vitAtol());
}

TEST_P(ViTMultiDtypeTest, ViTBasePatch16GradientFlow) {
    auto model = ViT_Base_Patch16(10, false, 224);
    convert_model(model);
    model->train();

    auto input = createInput({1, 3, 224, 224});
    auto output = model->forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
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
    // audit-4 U.16: small-config TEST_P — diff against CPU/Float32 reference.
    auto [cpu_ref, input] = captureCpuReferenceAndConvert(
        model, {1, 3, 224, 224},
        [](auto& m, const Variable& x) { return m->forward(x); });

    auto output = model->forward(input);

    expectShape(output.tensor(), {1, 10});
    expectDType(output.tensor());
    expectMatchesCpu(output.tensor(), cpu_ref, vitAtol());
}

TEST_P(ViTMultiDtypeTest, ViTBasePatch16CustomClasses) {
    auto model = ViT_Base_Patch16(100, false, 224);
    // audit-4 U.16: small-config TEST_P — diff against CPU/Float32 reference.
    auto [cpu_ref, input] = captureCpuReferenceAndConvert(
        model, {2, 3, 224, 224},
        [](auto& m, const Variable& x) { return m->forward(x); });

    auto output = model->forward(input);

    expectShape(output.tensor(), {2, 100});
    expectDType(output.tensor());
    expectMatchesCpu(output.tensor(), cpu_ref, vitAtol());
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
    // audit-4 U.16: small-config TEST_P — diff against CPU/Float32 reference.
    auto [cpu_ref, input] = captureCpuReferenceAndConvert(
        model, {2, 3, 224, 224},
        [](auto& m, const Variable& x) { return m->forward(x); });

    auto output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectDType(output.tensor());
    expectMatchesCpu(output.tensor(), cpu_ref, vitAtol());
}

TEST_P(ViTMultiDtypeTest, ViTBasePatch32GradientFlow) {
    auto model = ViT_Base_Patch32(10, false, 224);
    convert_model(model);
    model->train();

    auto input = createInput({1, 3, 224, 224});
    auto output = model->forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
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
    // 307M-param model + 224x224 activations exceeds an 8 GB Vulkan device
    // at Float64 (and at Float16/BFloat16 because Vulkan's half-precision
    // attention path widens to F32 internally). Use the memory helper so
    // memory-constrained backends shrink the input but CPU/CUDA-with-headroom
    // still exercise the full model.
    int img_size = getImageSizeForMemory(224, 307'000'000, false);
    auto model = ViT_Large_Patch16(1000, false, img_size);
    convert_model(model);

    auto input = createInput({2, 3, img_size, img_size});
    auto output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectDType(output.tensor());
    // reason: ViT-Large-Patch16 forward is ~307M params; running a CPU/Float32
    // reference (2 * 1000 ≈ 2e3 output floats but 307M weights to multiply
    // against) doubles peak host RAM at no value-level signal we don't already
    // get from the parameterized CPU/Float32 run of this same TEST_P.
    expectOutputSane(output.tensor());  // audit-T.1
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

    EXPECT_GRAD_FLOWS(input);
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
    // reason: ViT-Large-Patch32 ~307M params; capturing a CPU/F32 reference
    // doubles host memory and the parameterized CPU/F32 run already covers
    // the value-level signal.
    expectOutputSane(output.tensor());  // audit-T.1
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
    int img_size = getImageSizeForMemory(224, 632'000'000, false, /*patch_size=*/14);

    auto model = ViT_Huge_Patch14(1000, false, img_size);
    convert_model(model);

    auto input = createInput({1, 3, img_size, img_size});
    auto output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectDType(output.tensor());
    // reason: ViT-Huge-Patch14 forward is ~632M params; a CPU/F32 reference
    // forward would push host RAM into multi-GB territory for no extra
    // signal beyond the parameterized CPU/F32 run.
    expectOutputSane(output.tensor());  // audit-T.1
}

TEST_P(ViTMultiDtypeTest, ViTHugePatch14GradientFlow) {
    // Use a reduced model (8 layers instead of 32) to fit in 8GB GPU memory
    // (~160M params instead of 632M) for memory-constrained configurations:
    // - Float64 on any GPU backend (doubles memory vs Float32)
    // - Vulkan backend (unfused attention stores more intermediates than CUDA's cuDNN SDPA)
    bool use_reduced_model = (backend_name() != "cpu" &&
        (dtype() == DType::Float64 ||
         backend_name() == "vulkan"));
    int img_size = use_reduced_model ? 112 : getImageSizeForMemory(224, 632'000'000, true, /*patch_size=*/14);

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

    EXPECT_GRAD_FLOWS(input);
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
    int img_size = getImageSizeForMemory(224, 632'000'000, false, /*patch_size=*/16);

    auto model = ViT_Huge_Patch16(1000, false, img_size);
    convert_model(model);

    auto input = createInput({1, 3, img_size, img_size});
    auto output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectDType(output.tensor());
    // reason: ViT-Huge-Patch16 forward is ~632M params; CPU reference would
    // be multi-GB host RAM with no extra signal over the parameterized
    // CPU/F32 run.
    expectOutputSane(output.tensor());  // audit-T.1
}

// ============================================================================
// Different Image Sizes Tests
// ============================================================================

TEST_P(ViTMultiDtypeTest, ViTBaseDifferentImageSize384) {
    // Test with 384x384 input (commonly used for fine-tuning)
    auto model = ViT_Base_Patch16(1000, false, 384);
    // audit-4 U.16: small-config TEST_P — diff against CPU/Float32 reference.
    auto [cpu_ref, input] = captureCpuReferenceAndConvert(
        model, {1, 3, 384, 384},
        [](auto& m, const Variable& x) { return m->forward(x); });

    auto output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectDType(output.tensor());
    expectMatchesCpu(output.tensor(), cpu_ref, vitAtol());
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
    // reason: 512x512 input may shrink to 384x384 on memory-constrained
    // backends so the input shape diverges between CPU and device runs —
    // can't compare against a static CPU reference.
    expectOutputSane(output.tensor());  // audit-T.1

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
    // ViT-Base is ~86M params; 8 batches at 224x224 in Float64 push the
    // autograd-graph activations past an 8 GB Vulkan/CUDA budget. Shrink
    // the batch on memory-constrained backends.
    int batch = getBatchSizeForMemory(8, 86'000'000);
    auto model = ViT_Base_Patch16(10, false, 224);
    convert_model(model);

    auto input = createInput({batch, 3, 224, 224});
    auto output = model->forward(input);

    expectShape(output.tensor(), {batch, 10});
    expectDType(output.tensor());
    expectOutputSane(output.tensor());  // audit-T.1
}

TEST_P(ViTMultiDtypeTest, ViTBaseSingleChannel) {
    // Test with grayscale input (1 channel)
    auto patch_embed = std::make_shared<PatchEmbedding>(224, 16, 1, 768);
    convert_model(patch_embed);

    auto input = createInput({2, 1, 224, 224});
    auto output = patch_embed->forward(input);

    expectShape(output.tensor(), {2, 196, 768});
    expectDType(output.tensor());
    expectOutputSane(output.tensor());  // audit-T.1
}

TEST_P(ViTMultiDtypeTest, ViTBaseMultiChannel) {
    // Test with hyperspectral input (4 channels)
    auto patch_embed = std::make_shared<PatchEmbedding>(224, 16, 4, 768);
    convert_model(patch_embed);

    auto input = createInput({2, 4, 224, 224});
    auto output = patch_embed->forward(input);

    expectShape(output.tensor(), {2, 196, 768});
    expectDType(output.tensor());
    expectOutputSane(output.tensor());  // audit-T.1
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
    expectOutputSane(output_train.tensor());  // audit-T.1

    // Test in evaluation mode
    model->eval();
    auto output_eval = model->forward(input);
    expectDType(output_eval.tensor());
    expectOutputSane(output_eval.tensor());  // audit-T.1

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
    ::testing::InitGoogleTest(&argc, argv);
    if (!::testing::GTEST_FLAG(list_tests)) {
        tenzor::initialize();
    }
    return RUN_ALL_TESTS();
}
