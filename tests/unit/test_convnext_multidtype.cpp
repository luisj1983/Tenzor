/**
 * @file test_convnext_multidtype.cpp
 * @brief Multi-backend and multi-dtype tests for ConvNeXt variants
 * @details Tests ConvNeXt architectures across CPU, CUDA, OneAPI backends
 *          with Float32, Float64, and Float16 data types
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include "../../include/tenzor/models/convnext.hpp"

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
};

// ============================================================================
// ConvNeXt-Tiny Tests (Multi-Backend Multi-DType)
// ============================================================================

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyForwardShape) {
    auto model = convnext_tiny(1000, false);
    convert_model(model);
    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), true);
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

    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
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
    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);
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

    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Verify gradients flow through layer scaling - just check gradient exists
    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyGELUActivation) {
    // Test GELU activation through forward pass
    auto model = convnext_tiny(10, false);
    convert_model(model);
    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);
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
    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), true);
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

    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtSmallDepthwiseConvolution) {
    auto model = convnext_small(10, false);
    convert_model(model);
    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);
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
    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), true);
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

    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
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

    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Verify gradients flow through layer scaling - just check gradient exists
    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

// ============================================================================
// ConvNeXt-Large Tests (Multi-Backend Multi-DType)
// ============================================================================

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtLargeForwardShape) {
    auto model = convnext_large(1000, false);
    convert_model(model);
    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);
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

    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
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
    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);
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
    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);
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

    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

// ============================================================================
// Edge Case Tests (Multi-Backend Multi-DType)
// ============================================================================

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyBatchSizeOne) {
    auto model = convnext_tiny(10, false);
    convert_model(model);
    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 10}));
    EXPECT_EQ(output.tensor().dtype(), dtype());
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyCustomClasses) {
    auto model = convnext_tiny(100, false);
    convert_model(model);
    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 100}));
    EXPECT_EQ(output.tensor().dtype(), dtype());
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyMultipleBatchSizes) {
    auto model = convnext_tiny(10, false);
    convert_model(model);

    // Test with different batch sizes
    for (int batch_size : {1, 2, 4, 8}) {
        Variable input(Tensor({batch_size, 3, 224, 224}, dtype(), device()), true);
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
    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);
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
    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);

    // Run forward pass twice with same input
    Variable output1 = model->forward(input);
    Variable output2 = model->forward(input);

    // Outputs should be identical (deterministic)
    auto cpu_out1 = output1.tensor().to(Device::cpu()).to(DType::Float32);
    auto cpu_out2 = output2.tensor().to(Device::cpu()).to(DType::Float32);
    auto data1 = cpu_out1.data<float>();
    auto data2 = cpu_out2.data<float>();

    for (size_t i = 0; i < cpu_out1.numel(); ++i) {
        EXPECT_NEAR(data1[i], data2[i], atol());
    }
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtLargeGradientNumericalStability) {
    auto model = convnext_large(10, false);
    convert_model(model);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Check gradient numerical stability
    EXPECT_TRUE(input.grad().has_value());
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
    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);

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
    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), true);

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
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
