/**
 * @file test_swin_transformer_multidtype.cpp
 * @brief Multi-dtype tests for Swin Transformer variants
 *
 * Tests Swin Transformer models with Float32, Float64, and Float16 data types across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - Proper dtype propagation through shifted window attention
 * - Correct output shapes for Swin-Tiny, Small, Base, and Large variants
 * - Gradient flow through patch merging and window partitioning
 * - Hierarchical feature map handling across dtypes
 * - Relative position bias with different dtypes
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/swin_transformer.hpp"
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::models;

// ============================================================================
// Swin Transformer Multi-Backend Multi-DType Test Fixture
// ============================================================================

class SwinMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // For Float16, use relaxed tolerances
    float param_count_tol_;

    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        // Swin has complex attention mechanisms, need relaxed tolerances
        if (dtype() == DType::Float16) {
            param_count_tol_ = 0.10f;  // 10% tolerance
        } else {
            param_count_tol_ = 0.02f;  // 2% tolerance
        }
    }

    // Helper to check parameter count within tolerance
    bool CheckParameterCount(int64_t actual, int64_t expected) {
        int64_t tolerance = static_cast<int64_t>(expected * param_count_tol_);
        return std::abs(actual - expected) <= tolerance;
    }

    // Image size must satisfy multiple constraints:
    // 1. (img_size / patch_size) % window_size == 0
    // 2. img_size / patch_size / 2^(num_stages-1) must be integer
    // For Swin Tiny: patch_size=4, window_size=7, num_stages=4
    // Minimum valid size is 224 (224/4=56, 56%7=0, 56/8=7)
    int GetImageSize() {
        return 224;
    }

    // Helper to check for finite values
    bool checkFiniteValues(const Variable& var, size_t num_samples = 100) {
        auto tensor_cpu = var.tensor().to(Device::cpu()).to(DType::Float32);
        auto data = tensor_cpu.data<float>();
        size_t check_count = std::min(num_samples, static_cast<size_t>(tensor_cpu.numel()));
        for (size_t i = 0; i < check_count; ++i) {
            if (!std::isfinite(data[i])) {
                return false;
            }
        }
        return true;
    }

    // Helper to create and convert a Swin model
    template<typename ModelFactory>
    auto createSwinModel(ModelFactory factory) {
        auto model = factory();
        model->to(dtype());
        model->to(device());
        return model;
    }
};

// ============================================================================
// Swin-Tiny Tests
// ============================================================================

TEST_P(SwinMultiDTypeTest, SwinTinyForwardShape) {
    int img_size = GetImageSize();
    auto model = swin_tiny(1000, img_size, false);
    model->to(dtype());
    model->to(device());

    Variable input = createInput({2, 3, img_size, img_size});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(SwinMultiDTypeTest, SwinTinyGradientFlow) {
    int img_size = GetImageSize();
    auto model = swin_tiny(10, img_size, false);
    model->to(dtype());
    model->to(device());
    model->train();

    Variable input = createInput({1, 3, img_size, img_size}, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    if (input.grad().has_value()) {
        auto grad = input.grad().value().to(Device::cpu()).to(DType::Float32);
        auto grad_data = grad.data<float>();
        int64_t total = grad.numel();
        int non_finite_count = 0;
        float max_abs = 0;
        for (int64_t i = 0; i < total; ++i) {
            if (!std::isfinite(grad_data[i])) non_finite_count++;
            else max_abs = std::max(max_abs, std::abs(grad_data[i]));
        }
        std::cerr << "SWIN_TINY GRAD STATS: total=" << total << " non_finite=" << non_finite_count
                  << " max_abs_finite=" << max_abs << std::endl;
    }

    EXPECT_GRAD_FLOWS(input);
    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);

    // Verify gradients exist and have correct dtype
    for (const auto& p : params) {
        if (p->grad().has_value()) {
            EXPECT_EQ(p->grad()->dtype(), dtype());
        }
    }
}

TEST_P(SwinMultiDTypeTest, SwinTinyParameterCount) {
    auto model = swin_tiny(1000, 224, false);
    auto params = model->parameters();

    size_t total_params = countParameters(params);

    // Swin-Tiny should have ~29M parameters (allow 20% tolerance)
    EXPECT_GT(total_params, 23'000'000);
    EXPECT_LT(total_params, 35'000'000);
}

TEST_P(SwinMultiDTypeTest, SwinTinyBatchSizeOne) {
    int img_size = GetImageSize();
    auto model = swin_tiny(10, img_size, false);
    model->to(dtype());
    model->to(device());

    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 10});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(SwinMultiDTypeTest, SwinTinyCustomClasses) {
    int img_size = GetImageSize();
    auto model = swin_tiny(100, img_size, false);
    model->to(dtype());
    model->to(device());

    Variable input = createInput({2, 3, img_size, img_size});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 100});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// Swin-Small Tests
// ============================================================================

TEST_P(SwinMultiDTypeTest, SwinSmallForwardShape) {
    int img_size = GetImageSize();
    auto model = swin_small(1000, img_size, false);
    model->to(dtype());
    model->to(device());

    Variable input = createInput({2, 3, img_size, img_size});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(SwinMultiDTypeTest, SwinSmallGradientFlow) {
    int img_size = GetImageSize();
    auto model = swin_small(10, img_size, false);
    model->to(dtype());
    model->to(device());
    model->train();

    Variable input = createInput({1, 3, img_size, img_size}, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    if (input.grad().has_value()) {
        auto grad = input.grad().value().to(Device::cpu()).to(DType::Float32);
        auto grad_data = grad.data<float>();
        int64_t total = grad.numel();
        int non_finite_count = 0;
        float max_abs = 0;
        for (int64_t i = 0; i < total; ++i) {
            if (!std::isfinite(grad_data[i])) non_finite_count++;
            else max_abs = std::max(max_abs, std::abs(grad_data[i]));
        }
        std::cerr << "SWIN_SMALL GRAD STATS: total=" << total << " non_finite=" << non_finite_count
                  << " max_abs_finite=" << max_abs << std::endl;
    }

    EXPECT_GRAD_FLOWS(input);
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

TEST_P(SwinMultiDTypeTest, SwinSmallParameterCount) {
    auto model = swin_small(1000, 224, false);
    auto params = model->parameters();

    size_t total_params = countParameters(params);

    // Swin-Small should have ~50M parameters (allow 20% tolerance)
    EXPECT_GT(total_params, 40'000'000);
    EXPECT_LT(total_params, 60'000'000);
}

// ============================================================================
// Swin-Base Tests
// ============================================================================

TEST_P(SwinMultiDTypeTest, SwinBaseForwardShape) {
    int img_size = GetImageSize();
    auto model = swin_base(1000, img_size, false);
    model->to(dtype());
    model->to(device());

    Variable input = createInput({2, 3, img_size, img_size});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(SwinMultiDTypeTest, SwinBaseGradientFlow) {
    int img_size = GetImageSize();
    auto model = swin_base(10, img_size, false);
    model->to(dtype());
    model->to(device());
    model->train();

    Variable input = createInput({1, 3, img_size, img_size}, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

TEST_P(SwinMultiDTypeTest, SwinBaseParameterCount) {
    auto model = swin_base(1000, 224, false);
    auto params = model->parameters();

    size_t total_params = countParameters(params);

    // Swin-Base should have ~88M parameters (allow 20% tolerance)
    EXPECT_GT(total_params, 70'000'000);
    EXPECT_LT(total_params, 105'000'000);
}

// ============================================================================
// Swin-Large Tests
// ============================================================================

TEST_P(SwinMultiDTypeTest, SwinLargeForwardShape) {
    int img_size = GetImageSize();
    auto model = swin_large(1000, img_size, false);
    model->to(dtype());
    model->to(device());

    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(SwinMultiDTypeTest, SwinLargeGradientFlow) {
    int img_size = GetImageSize();

    // With gradient checkpointing, activations are not saved during forward pass
    auto model = swin_large(10, img_size, false, true);  // use_checkpoint=true
    model->to(dtype());
    model->to(device());
    model->train();

    Variable input = createInput({1, 3, img_size, img_size}, true);
    Variable output = (*model)(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

TEST_P(SwinMultiDTypeTest, SwinLargeParameterCount) {
    auto model = swin_large(1000, 224, false);
    auto params = model->parameters();

    size_t total_params = countParameters(params);

    // Swin-Large should have ~197M parameters (allow 20% tolerance)
    EXPECT_GT(total_params, 155'000'000);
    EXPECT_LT(total_params, 235'000'000);
}

// ============================================================================
// Different Image Sizes Tests
// ============================================================================

TEST_P(SwinMultiDTypeTest, SwinTinyImageSize448) {
    auto model = swin_tiny(1000, 448, false);
    model->to(dtype());
    model->to(device());

    Variable input = createInput({1, 3, 448, 448});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(SwinMultiDTypeTest, SwinSmallImageSize448) {
    auto model = swin_small(1000, 448, false);
    model->to(dtype());
    model->to(device());

    Variable input = createInput({1, 3, 448, 448});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(SwinMultiDTypeTest, SwinBaseImageSize672) {
    // Swin-Base at 672x672 needs >8 GB on Vulkan/CUDA (window-attention
    // tensors and the patch-merging activations dominate even with
    // gradient checkpointing). Pick the largest "valid" Swin size for the
    // device memory budget — windows require img_size / patch / 2^stages
    // to be divisible by window_size=7. For Swin-Base (patch=4, 4 stages):
    //   * 672 / 4 / 8 = 21 ✔     (default; needs ~12 GB)
    //   * 448 / 4 / 8 = 14 ✔     (~6 GB; fits 8 GB Vulkan F32)
    //   * 224 / 4 / 8 =  7 ✔     (~3 GB; fits 8 GB Vulkan F64)
    bool tight = (backend_name() == "vulkan" || backend_name() == "cuda");
    bool tight_f64 = tight && (dtype() == DType::Float64);
    int img_size = tight_f64 ? 224 : (tight ? 448 : 672);
    auto model = swin_base(1000, img_size, false, true);  // use_checkpoint=true for memory
    model->to(dtype());
    model->to(device());

    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// Patch Merging Tests
// ============================================================================

TEST_P(SwinMultiDTypeTest, SwinTinyPatchMergingFeatures) {
    int img_size = GetImageSize();
    auto model = swin_tiny(1000, img_size, false);
    model->to(dtype());
    model->to(device());

    // Test that model processes patches correctly at different scales
    Variable input1 = createInput({1, 3, img_size, img_size});
    Variable output1 = model->forward(input1);

    Variable input2 = createInput({2, 3, img_size, img_size});
    Variable output2 = model->forward(input2);

    // Batch size should scale linearly
    EXPECT_EQ(output2.tensor().shape()[0], 2 * output1.tensor().shape()[0]);
    expectDType(output1.tensor());
    expectDType(output2.tensor());
    expectFiniteNonZero(output1.tensor());
    expectFiniteNonZero(output2.tensor());
}

TEST_P(SwinMultiDTypeTest, SwinSmallPatchMergingConsistency) {
    int img_size = GetImageSize();
    auto model = swin_small(100, img_size, false);
    model->to(dtype());
    model->to(device());
    model->eval();

    // Same input should give same output (deterministic forward pass)
    Variable input1 = createInput({1, 3, img_size, img_size}, false);
    Variable output1 = model->forward(input1);

    Variable input2 = createInput({1, 3, img_size, img_size}, false);
    Variable output2 = model->forward(input2);

    // Outputs should have same shape
    auto shape1 = output1.tensor().shape();
    auto shape2 = output2.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape1.begin(), shape1.end()),
              std::vector<int64_t>(shape2.begin(), shape2.end()));
    expectDType(output1.tensor());
}

// ============================================================================
// Shifted Window Attention Tests
// ============================================================================

TEST_P(SwinMultiDTypeTest, SwinTinyShiftedWindowGradients) {
    int img_size = GetImageSize();
    auto model = swin_tiny(10, img_size, false);
    model->to(dtype());
    model->to(device());
    model->train();

    // Initialize input with small non-zero values to ensure non-zero gradients
    Variable input(tenzor::randn({1, 3, img_size, img_size}, dtype(), device()) * 0.01f, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::mean(output);
    loss.backward();

    // Verify gradients flow through shifted window attention
    EXPECT_GRAD_FLOWS(input);

    auto grad = input.grad().value().to(Device::cpu()).to(DType::Float32);
    auto grad_data = grad.data<float>();

    // Check for non-zero gradients
    // BFloat16 gradients here are empirically as large/accurate as
    // Float32/Float16's (~1e-3 to 2e-3 magnitude, verified via a
    // standalone diagnostic comparing raw gradient values across dtypes
    // with the same seed) -- autograd flows correctly for BFloat16.
    // atol()'s general-purpose BFloat16 value (1e-2, tuned for broader
    // numerical-comparison contexts) is coincidentally looser than
    // Float16's own carve-out below and was misclassifying a genuine,
    // correctly-sized gradient signal as "no gradient". Match Float16's
    // carve-out rather than falling through to atol().
    float grad_tol = (dtype() == DType::Float16 || dtype() == DType::BFloat16) ? 1e-3f : atol();
    bool has_nonzero_grad = false;
    for (size_t i = 0; i < std::min(size_t(100), static_cast<size_t>(grad.numel())); ++i) {
        if (std::abs(grad_data[i]) > grad_tol) {
            has_nonzero_grad = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero_grad);
}

TEST_P(SwinMultiDTypeTest, SwinBaseShiftedWindowAttention) {
    int img_size = GetImageSize();
    auto model = swin_base(100, img_size, false);
    model->to(dtype());
    model->to(device());
    model->train();

    Variable input = createInput({2, 3, img_size, img_size});
    Variable output = model->forward(input);

    // Test that attention mechanism produces finite outputs
    EXPECT_TRUE(checkFiniteValues(output));
}

// ============================================================================
// Hierarchical Feature Extraction Tests
// ============================================================================

TEST_P(SwinMultiDTypeTest, SwinTinyHierarchicalFeatures) {
    int img_size = GetImageSize();
    auto model = swin_tiny(1000, img_size, false);
    model->to(dtype());
    model->to(device());

    // Test feature extraction at different batch sizes
    Variable input_small = createInput({1, 3, img_size, img_size});
    Variable output_small = model->forward(input_small);

    Variable input_large = createInput({4, 3, img_size, img_size});
    Variable output_large = model->forward(input_large);

    // Features should scale with batch size
    EXPECT_EQ(output_small.tensor().shape()[1], output_large.tensor().shape()[1]);
    EXPECT_EQ(output_large.tensor().shape()[0], 4);
    expectDType(output_small.tensor());
    expectDType(output_large.tensor());
}

TEST_P(SwinMultiDTypeTest, SwinSmallHierarchicalGradients) {
    int img_size = GetImageSize();
    auto model = swin_small(50, img_size, false);
    model->to(dtype());
    model->to(device());
    model->train();

    Variable input = createInput({2, 3, img_size, img_size}, true);
    Variable output = model->forward(input);
    Variable squared = output * output;
    Variable loss = tenzor::sum(squared);
    loss.backward();

    // Verify hierarchical gradients
    EXPECT_GRAD_FLOWS(input);
    EXPECT_EQ(input.grad()->dtype(), dtype());

    auto params = model->parameters();
    int params_with_grad = 0;
    for (const auto& p : params) {
        if (p->grad().has_value()) {
            params_with_grad++;
        }
    }
    EXPECT_GT(params_with_grad, 0);
}

TEST_P(SwinMultiDTypeTest, SwinBaseHierarchicalFeatureExtraction) {
    int img_size = GetImageSize();
    auto model = swin_base(200, img_size, false);
    model->to(dtype());
    model->to(device());
    model->eval();

    // Initialize with random values
    Variable input(tenzor::randn({1, 3, img_size, img_size}, dtype(), device()) * 0.01f, false);
    Variable output = model->forward(input);

    // Verify output features are properly scaled
    auto output_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto output_data = output_cpu.data<float>();
    double sum = 0.0;
    size_t check_count = std::min(size_t(200), static_cast<size_t>(output_cpu.numel()));
    for (size_t i = 0; i < check_count; ++i) {
        sum += std::abs(static_cast<double>(output_data[i]));
    }
    double mean_abs = sum / check_count;

    // Features should be in reasonable range (not all zeros, not extreme)
    EXPECT_GT(mean_abs, atol());
    EXPECT_LT(mean_abs, 1000.0);
}

TEST_P(SwinMultiDTypeTest, SwinLargeHierarchicalMultiScale) {
    int img_size = GetImageSize();
    auto model = swin_large(100, img_size, false);
    model->to(dtype());
    model->to(device());

    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);

    // Output should capture multi-scale hierarchical information
    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 100);
    expectDType(output.tensor());
}

// ============================================================================
// Variant Comparison Tests
// ============================================================================

TEST_P(SwinMultiDTypeTest, VariantParameterProgression) {
    // Verify that parameter counts increase: Tiny < Small < Base < Large
    auto tiny = swin_tiny(1000, 224, false);
    auto small = swin_small(1000, 224, false);
    auto base = swin_base(1000, 224, false);
    auto large = swin_large(1000, 224, false);

    size_t tiny_params = countParameters(tiny->parameters());
    size_t small_params = countParameters(small->parameters());
    size_t base_params = countParameters(base->parameters());
    size_t large_params = countParameters(large->parameters());

    EXPECT_LT(tiny_params, small_params);
    EXPECT_LT(small_params, base_params);
    EXPECT_LT(base_params, large_params);
}

TEST_P(SwinMultiDTypeTest, VariantOutputConsistency) {
    int img_size = GetImageSize();

    // All variants should produce same output shape for same input/output config
    auto tiny = swin_tiny(50, 224, false);
    tiny->to(dtype());
    tiny->to(device());

    auto small = swin_small(50, 224, false);
    small->to(dtype());
    small->to(device());

    auto base = swin_base(50, 224, false);
    base->to(dtype());
    base->to(device());

    Variable input = createInput({1, 3, img_size, img_size}, false);

    Variable output_tiny = tiny->forward(input);
    Variable output_small = small->forward(input);
    Variable output_base = base->forward(input);

    auto shape_tiny = output_tiny.tensor().shape();
    auto shape_small = output_small.tensor().shape();
    auto shape_base = output_base.tensor().shape();

    EXPECT_EQ(std::vector<int64_t>(shape_tiny.begin(), shape_tiny.end()),
              std::vector<int64_t>(shape_small.begin(), shape_small.end()));
    EXPECT_EQ(std::vector<int64_t>(shape_small.begin(), shape_small.end()),
              std::vector<int64_t>(shape_base.begin(), shape_base.end()));

    expectDType(output_tiny.tensor());
    expectDType(output_small.tensor());
    expectDType(output_base.tensor());
}

// ============================================================================
// Edge Cases and Robustness Tests
// ============================================================================

TEST_P(SwinMultiDTypeTest, SwinTinyMinimalBatch) {
    int img_size = GetImageSize();
    auto model = swin_tiny(10, img_size, false);
    model->to(dtype());
    model->to(device());

    // Test with minimal batch size
    Variable input = createInput({1, 3, img_size, img_size}, false);
    Variable output = model->forward(input);

    EXPECT_EQ(output.tensor().shape()[0], 1);
    expectDType(output.tensor());
}

TEST_P(SwinMultiDTypeTest, SwinSmallLargeBatch) {
    int img_size = GetImageSize();
    auto model = swin_small(10, img_size, false, true);  // use_checkpoint=true for memory
    model->to(dtype());
    model->to(device());

    // batch=8 at the default img_size pushes past 8 GB on Vulkan/CUDA when
    // the dtype is Float64 (every weight + activation tensor doubles).
    // Halve the batch on memory-constrained backends; the test still
    // exercises "more than one sample" through the window-attention path.
    bool tight = (backend_name() == "vulkan" || backend_name() == "cuda")
                  && dtype() == DType::Float64;
    int batch = tight ? 4 : 8;

    // Test with larger batch
    Variable input = createInput({batch, 3, img_size, img_size}, false);
    Variable output = model->forward(input);

    EXPECT_EQ(output.tensor().shape()[0], batch);
    expectDType(output.tensor());
}

TEST_P(SwinMultiDTypeTest, SwinBaseNumericalStability) {
    int img_size = GetImageSize();
    auto model = swin_base(10, img_size, false);
    model->to(dtype());
    model->to(device());
    model->train();

    Variable input = createInput({1, 3, img_size, img_size}, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Check for numerical stability (no NaN or Inf)
    {
        auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
        auto* out_data = out_cpu.data<float>();
        int64_t n = out_cpu.numel();
        float max_out = 0;
        int non_fin = 0;
        for (int64_t i = 0; i < n; i++) {
            if (!std::isfinite(out_data[i])) non_fin++;
            else max_out = std::max(max_out, std::abs(out_data[i]));
        }
        std::cerr << "OUTPUT STATS: numel=" << n << " non_finite=" << non_fin
                  << " max_abs=" << max_out << std::endl;
    }
    EXPECT_TRUE(checkFiniteValues(output));

    if (input.grad().has_value()) {
        auto grad = input.grad().value().to(Device::cpu()).to(DType::Float32);
        auto grad_data = grad.data<float>();
        int64_t total = grad.numel();
        int non_finite_count = 0;
        float max_abs = 0;
        for (int64_t i = 0; i < total; ++i) {
            if (!std::isfinite(grad_data[i])) {
                non_finite_count++;
                if (non_finite_count <= 10) {
                    std::cerr << "  NON-FINITE at index " << i << ": " << grad_data[i] << std::endl;
                }
            } else {
                max_abs = std::max(max_abs, std::abs(grad_data[i]));
            }
        }
        std::cerr << "GRAD STATS: total=" << total << " non_finite=" << non_finite_count
                  << " max_abs_finite=" << max_abs << std::endl;
        for (size_t i = 0; i < std::min(size_t(100), static_cast<size_t>(grad.numel())); ++i) {
            EXPECT_TRUE(std::isfinite(grad_data[i]));
        }
    }
}

TEST_P(SwinMultiDTypeTest, SwinLargeTrainEvalModeConsistency) {
    int img_size = GetImageSize();
    auto model = swin_large(50, img_size, false);
    model->to(dtype());
    model->to(device());

    Variable input = createInput({1, 3, img_size, img_size}, false);

    // Test in eval mode
    model->eval();
    Variable output_eval = model->forward(input);

    // Test in train mode
    model->train();
    Variable output_train = model->forward(input);

    // Both should produce outputs with same shape
    auto shape_eval = output_eval.tensor().shape();
    auto shape_train = output_train.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_eval.begin(), shape_eval.end()),
              std::vector<int64_t>(shape_train.begin(), shape_train.end()));
    expectDType(output_eval.tensor());
    expectDType(output_train.tensor());
}

// ============================================================================
// DType Conversion Tests
// ============================================================================

TEST_P(SwinMultiDTypeTest, SwinTinyDTypePreservation) {
    int img_size = GetImageSize();
    auto model = swin_tiny(10, img_size, false);
    model->to(dtype());
    model->to(device());

    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);

    // Verify dtype is preserved throughout forward pass
    expectDType(output.tensor());
}

TEST_P(SwinMultiDTypeTest, SwinSmallGradientDTypeConsistency) {
    int img_size = GetImageSize();
    auto model = swin_small(10, img_size, false);
    model->to(dtype());
    model->to(device());
    model->train();

    Variable input = createInput({1, 3, img_size, img_size}, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Verify gradient dtype matches input dtype
    if (input.grad().has_value()) {
        EXPECT_EQ(input.grad()->dtype(), dtype());
    }
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SwinMultiDTypeTest);

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
