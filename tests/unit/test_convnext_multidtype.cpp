/**
 * @file test_convnext_multidtype.cpp
 * @brief Multi-backend and multi-dtype tests for ConvNeXt variants
 * @details Tests ConvNeXt architectures across CPU, CUDA, OneAPI backends
 *          with Float32, Float64, and Float16 data types
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include "../../include/tenzor/models/convnext.hpp"
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::models;
using namespace tenzor::testing;

// ============================================================================
// Test Fixture with Backend + DType Parameterization
// ============================================================================

class ConvNeXtMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
    }

    /**
     * @brief Get appropriate input size for the current backend and dtype
     * GPU backends use smaller sizes for Float64 due to memory constraints
     */
    int64_t getInputSize(int64_t default_size) {
        if (device().type == Device::Type::CPU) {
            return default_size;
        }

        bool is_float64 = (dtype() == DType::Float64);
        bool is_float16 = (dtype() == DType::Float16);

        if (is_float64) {
            // Float64: ConvNeXt needs smaller sizes for gradients
            if (default_size >= 224) return 128;
            return std::min(default_size, int64_t(112));
        }

        if (is_float16) {
            // Float16: slightly smaller to avoid numerical issues
            if (default_size >= 224) return 160;
            return std::min(default_size, int64_t(128));
        }

        // Float32: moderate reduction for GPU
        return default_size;
    }

    /**
     * @brief Get input size for large models (ConvNeXt-Large, XLarge)
     * These models need smaller input sizes to fit in 8GB VRAM
     * Sizes are very conservative to handle GPU memory fragmentation from prior tests
     * @param model_size "large" or "xlarge"
     * @param needs_gradients true if backward pass will be run
     */
    int64_t getLargeModelInputSize(const std::string& model_size, bool needs_gradients) {
        if (device().type == Device::Type::CPU) {
            return 224;
        }

        bool is_float64 = (dtype() == DType::Float64);
        bool is_float32 = (dtype() == DType::Float32);

        if (model_size == "xlarge") {
            if (needs_gradients) {
                // XLarge (~350M params) with gradients needs very small input
                // Very conservative sizes for 8GB VRAM with fragmentation
                if (is_float64) return 32;   // Minimal size
                if (is_float32) return 48;   // Very conservative for fragmented memory
                return 64;   // Float16
            } else {
                // XLarge forward only - also conservative
                if (is_float64) return 48;
                return 96;
            }
        } else if (model_size == "large") {
            if (needs_gradients) {
                // Large (~198M params) with gradients
                // Very conservative sizes for fragmented 8GB VRAM
                if (is_float64) return 32;   // Very conservative
                if (is_float32) return 64;
                return 64;   // Float16
            } else {
                // Large forward only
                if (is_float64) return 64;
                return 96;
            }
        }

        // Fallback to standard sizing
        return getInputSize(224);
    }

    /**
     * @brief Check if this is a memory-constrained configuration
     */
    bool isMemoryConstrained() {
        return (device().type != Device::Type::CPU) &&
               (dtype() == DType::Float64 || dtype() == DType::Float16);
    }
};

// ============================================================================
// ConvNeXt-Tiny Tests (Multi-Backend Multi-DType)
// ============================================================================

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyForwardShape) {
    auto model = convnext_tiny(1000, false);
    convert_model(model);
    int64_t img_size = getInputSize(224);
    Variable input = createInput({2, 3, img_size, img_size});
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
    EXPECT_EQ(output.tensor().dtype(), dtype());
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyGradientFlow) {
    auto model = convnext_tiny(10, false);
    convert_model(model);
    model->train();

    int64_t img_size = getInputSize(224);
    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output * output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyParameterCount) {
    auto model = convnext_tiny(1000, false);
    convert_model(model);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // ConvNeXt-Tiny should have ~28M parameters (allow 20% tolerance)
    EXPECT_GT(total_params, 22'000'000);
    EXPECT_LT(total_params, 34'000'000);
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyDepthwiseConvolution) {
    // Test that depthwise convolutions work correctly
    auto model = convnext_tiny(10, false);
    convert_model(model);
    int64_t img_size = getInputSize(224);
    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);

    EXPECT_EQ(output.tensor().dtype(), dtype());
    // Check for NaN via reduction to CPU
    auto cpu_output = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto output_data = cpu_output.data<float>();
    bool has_nan = false;
    for (size_t i = 0; i < cpu_output.numel(); ++i) {
        if (std::isnan(output_data[i])) {
            has_nan = true;
            break;
        }
    }
    EXPECT_FALSE(has_nan);
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyLayerScaling) {
    // Test layer scaling functionality
    // Use train mode since that's the normal use case and exercises layer scaling
    auto model = convnext_tiny(10, false);
    convert_model(model);
    model->train();

    int64_t img_size = getInputSize(224);
    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output * output);
    loss.backward();

    // Verify gradients flow through layer scaling - just check gradient exists
    EXPECT_GRAD_FLOWS(input);
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyGELUActivation) {
    // Test GELU activation through forward pass
    auto model = convnext_tiny(10, false);
    convert_model(model);
    int64_t img_size = getInputSize(224);
    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);

    // GELU should produce bounded outputs for reasonable inputs
    auto cpu_output = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto output_data = cpu_output.data<float>();
    for (size_t i = 0; i < std::min(size_t(100), static_cast<size_t>(cpu_output.numel())); ++i) {
        EXPECT_FALSE(std::isnan(output_data[i]));
        EXPECT_FALSE(std::isinf(output_data[i]));
    }
}

// ============================================================================
// ConvNeXt-Small Tests (Multi-Backend Multi-DType)
// ============================================================================

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtSmallForwardShape) {
    auto model = convnext_small(1000, false);
    convert_model(model);
    int64_t img_size = getInputSize(224);
    Variable input = createInput({2, 3, img_size, img_size});
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
    EXPECT_EQ(output.tensor().dtype(), dtype());
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtSmallGradientFlow) {
    auto model = convnext_small(10, false);
    convert_model(model);
    model->train();

    int64_t img_size = getInputSize(224);
    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output * output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtSmallDepthwiseConvolution) {
    auto model = convnext_small(10, false);
    convert_model(model);
    int64_t img_size = getInputSize(224);
    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);

    EXPECT_EQ(output.tensor().dtype(), dtype());
    auto cpu_output = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto output_data = cpu_output.data<float>();
    bool has_nan = false;
    for (size_t i = 0; i < cpu_output.numel(); ++i) {
        if (std::isnan(output_data[i])) {
            has_nan = true;
            break;
        }
    }
    EXPECT_FALSE(has_nan);
}

// ============================================================================
// ConvNeXt-Base Tests (Multi-Backend Multi-DType)
// ============================================================================

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtBaseForwardShape) {
    auto model = convnext_base(1000, false);
    convert_model(model);
    int64_t img_size = getInputSize(224);
    Variable input = createInput({2, 3, img_size, img_size});
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
    EXPECT_EQ(output.tensor().dtype(), dtype());
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtBaseGradientFlow) {
    auto model = convnext_base(10, false);
    convert_model(model);
    model->train();

    int64_t img_size = getInputSize(224);
    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output * output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtBaseParameterCount) {
    auto model = convnext_base(1000, false);
    convert_model(model);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // ConvNeXt-Base should have ~89M parameters (allow 20% tolerance)
    EXPECT_GT(total_params, 70'000'000);
    EXPECT_LT(total_params, 107'000'000);
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtBaseLayerScaling) {
    // Test layer scaling functionality
    // Use train mode since that's the normal use case and exercises layer scaling
    auto model = convnext_base(10, false);
    convert_model(model);
    model->train();

    int64_t img_size = getInputSize(224);
    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output * output);
    loss.backward();

    // Verify gradients flow through layer scaling - just check gradient exists
    EXPECT_GRAD_FLOWS(input);
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

// ============================================================================
// ConvNeXt-Large Tests (Multi-Backend Multi-DType)
// ============================================================================

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtLargeForwardShape) {
    auto model = convnext_large(1000, false);
    convert_model(model);
    int64_t img_size = getLargeModelInputSize("large", false);
    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 1000}));
    EXPECT_EQ(output.tensor().dtype(), dtype());
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtLargeGradientFlow) {
    auto model = convnext_large(10, false);
    convert_model(model);
    model->train();

    int64_t img_size = getLargeModelInputSize("large", true);
    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output * output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtLargeParameterCount) {
    auto model = convnext_large(1000, false);
    convert_model(model);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // ConvNeXt-Large should have ~198M parameters (allow 20% tolerance)
    EXPECT_GT(total_params, 160'000'000);
    EXPECT_LT(total_params, 240'000'000);
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtLargeGELUActivation) {
    auto model = convnext_large(10, false);
    convert_model(model);
    int64_t img_size = getLargeModelInputSize("large", false);
    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);

    auto cpu_output = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto output_data = cpu_output.data<float>();
    for (size_t i = 0; i < std::min(size_t(100), static_cast<size_t>(cpu_output.numel())); ++i) {
        EXPECT_FALSE(std::isnan(output_data[i]));
        EXPECT_FALSE(std::isinf(output_data[i]));
    }
}

// ============================================================================
// ConvNeXt-XLarge Tests (Multi-Backend Multi-DType)
// ============================================================================

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtXLargeForwardShape) {
    auto model = convnext_xlarge(1000, false);
    convert_model(model);
    int64_t img_size = getLargeModelInputSize("xlarge", false);
    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 1000}));
    EXPECT_EQ(output.tensor().dtype(), dtype());
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtXLargeGradientFlow) {
    auto model = convnext_xlarge(10, false);
    convert_model(model);
    model->train();

    int64_t img_size = getLargeModelInputSize("xlarge", true);
    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output * output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

// ============================================================================
// Edge Case Tests (Multi-Backend Multi-DType)
// ============================================================================

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyBatchSizeOne) {
    auto model = convnext_tiny(10, false);
    convert_model(model);
    int64_t img_size = getInputSize(224);
    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 10}));
    EXPECT_EQ(output.tensor().dtype(), dtype());
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyCustomClasses) {
    auto model = convnext_tiny(100, false);
    convert_model(model);
    int64_t img_size = getInputSize(224);
    Variable input = createInput({2, 3, img_size, img_size});
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 100}));
    EXPECT_EQ(output.tensor().dtype(), dtype());
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyMultipleBatchSizes) {
    auto model = convnext_tiny(10, false);
    convert_model(model);

    int64_t img_size = getInputSize(224);
    // Test with different batch sizes (fewer for memory-constrained configs)
    std::vector<int> batch_sizes = isMemoryConstrained() ? std::vector<int>{1, 2} : std::vector<int>{1, 2, 4, 8};
    for (int batch_size : batch_sizes) {
        Variable input = createInput({batch_size, 3, img_size, img_size});
        Variable output = model->forward(input);

        auto shape = output.tensor().shape();
        EXPECT_EQ(shape[0], batch_size);
        EXPECT_EQ(shape[1], 10);
        EXPECT_EQ(output.tensor().dtype(), dtype());
    }
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtSmallNumericalStability) {
    auto model = convnext_small(10, false);
    convert_model(model);
    int64_t img_size = getInputSize(224);
    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);

    // Check for numerical stability (no NaN or Inf)
    auto cpu_output = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto data = cpu_output.data<float>();
    for (size_t i = 0; i < cpu_output.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i])) << "NaN detected at index " << i;
        EXPECT_FALSE(std::isinf(data[i])) << "Inf detected at index " << i;
    }
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtBaseDepthwiseConsistency) {
    auto model = convnext_base(10, false);
    convert_model(model);
    int64_t img_size = getInputSize(224);
    Variable input = createInput({1, 3, img_size, img_size});

    // Run forward pass twice with same input
    Variable output1 = model->forward(input);
    Variable output2 = model->forward(input);

    // Outputs should be identical (deterministic)
    auto cpu_out1 = output1.tensor().to(Device::cpu()).to(DType::Float32);
    auto cpu_out2 = output2.tensor().to(Device::cpu()).to(DType::Float32);
    auto data1 = cpu_out1.data<float>();
    auto data2 = cpu_out2.data<float>();

    // CUDA can have slight non-determinism due to cuDNN algorithms and atomic
    // operations; oneMKL/oneDNN likewise do not promise bitwise run-to-run
    // reproducibility without CNR mode (results vary with buffer alignment,
    // which the caching allocator changes between two forward passes of the
    // same model). Component-level probes are bit-identical; the ~1-ulp GEMM
    // jitter only becomes visible through ConvNeXt-Base's 27-block stage 3,
    // where it amplifies to ~3e-6. Use the same relaxed Float64 tolerance for
    // both library-backed GPU backends.
    float consistency_tol = atol();
    if ((device().type == Device::Type::CUDA ||
         device().type == Device::Type::OneAPI ||
         device().type == Device::Type::Vulkan ||
         device().type == Device::Type::ROCm) &&
        (dtype() == DType::Float64 || dtype() == DType::Float32)) {
        // Library-level run-to-run jitter in deep networks. cuDNN/oneDNN pick
        // convolution algorithms heuristically and use atomic reductions, so
        // two forward passes of the same model are not bit-identical — the
        // ~1-ulp GEMM jitter amplifies through ConvNeXt-Base's 27-block stage 3.
        // Float32's default 1e-5 atol sits right at that amplified jitter
        // (~1e-5 observed), so it needs the same relaxed bound already used for
        // Float64. (Float16's 1e-2 atol is already loose enough.)
        // Vulkan's tiled matmul/conv use the same atomic-reduction reductions
        // and the caching allocator changes buffer alignment between the two
        // passes, so it exhibits the identical ~4e-6 (Float64) jitter — measured
        // 4.35e-6, well within 5e-5. (Not a skip: the 5e-5 bound still catches
        // any real >5e-5 nondeterminism.)
        consistency_tol = 5e-5f;
    }

    for (size_t i = 0; i < cpu_out1.numel(); ++i) {
        EXPECT_NEAR(data1[i], data2[i], consistency_tol);
    }
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtLargeGradientNumericalStability) {
    auto model = convnext_large(10, false);
    convert_model(model);
    model->train();

    int64_t img_size = getLargeModelInputSize("large", true);
    Variable input = createInput({1, 3, img_size, img_size});
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output * output);
    loss.backward();

    // Check gradient numerical stability
    EXPECT_GRAD_FLOWS(input);
    auto grad_cpu = input.grad().value().to(Device::cpu()).to(DType::Float32);
    auto grad_data = grad_cpu.data<float>();
    for (size_t i = 0; i < grad_cpu.numel(); ++i) {
        EXPECT_FALSE(std::isnan(grad_data[i])) << "NaN gradient at index " << i;
        EXPECT_FALSE(std::isinf(grad_data[i])) << "Inf gradient at index " << i;
    }
}

// ============================================================================
// Cross-Variant Comparison Tests
// ============================================================================

TEST_P(ConvNeXtMultiDTypeTest, CrossVariantOutputShapeConsistency) {
    // All variants should produce same output shape for same num_classes
    int num_classes = 100;
    int64_t img_size = getInputSize(224);
    Variable input = createInput({1, 3, img_size, img_size});

    auto tiny = convnext_tiny(num_classes, false);
    auto small = convnext_small(num_classes, false);
    auto base = convnext_base(num_classes, false);
    convert_model(tiny);
    convert_model(small);
    convert_model(base);

    auto output_tiny = tiny->forward(input);
    auto output_small = small->forward(input);
    auto output_base = base->forward(input);

    EXPECT_EQ(output_tiny.tensor().shape()[1], num_classes);
    EXPECT_EQ(output_small.tensor().shape()[1], num_classes);
    EXPECT_EQ(output_base.tensor().shape()[1], num_classes);
}

TEST_P(ConvNeXtMultiDTypeTest, CrossVariantDTypePreservation) {
    // All variants should preserve dtype
    int64_t img_size = getInputSize(224);
    Variable input = createInput({1, 3, img_size, img_size});

    auto tiny = convnext_tiny(10, false);
    auto small = convnext_small(10, false);
    convert_model(tiny);
    convert_model(small);

    auto output_tiny = tiny->forward(input);
    auto output_small = small->forward(input);

    EXPECT_EQ(output_tiny.tensor().dtype(), dtype());
    EXPECT_EQ(output_small.tensor().dtype(), dtype());
}

// ============================================================================
// Instantiate Tests for Multiple Backends and DTypes
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ConvNeXtMultiDTypeTest);

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
