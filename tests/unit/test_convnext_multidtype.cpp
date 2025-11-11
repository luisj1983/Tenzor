/**
 * @file test_convnext_multidtype.cpp
 * @brief Comprehensive multi-dtype tests for ConvNeXt variants
 * @details Tests ConvNeXt architectures with Float32, Float64, and Float16 data types
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/convnext.hpp"

using namespace tenzor;
using namespace tenzor::models;

// ============================================================================
// Test Fixture with Parameterized DType
// ============================================================================

class ConvNeXtMultiDTypeTest : public ::testing::TestWithParam<DType> {
protected:
    void SetUp() override {
        device_ = Device::cpu();
        dtype_ = GetParam();

        // Set tolerance based on dtype
        switch (dtype_) {
            case DType::Float16:
                rel_tolerance_ = 1e-2;
                abs_tolerance_ = 1e-3;
                break;
            case DType::Float32:
                rel_tolerance_ = 1e-5;
                abs_tolerance_ = 1e-6;
                break;
            case DType::Float64:
                rel_tolerance_ = 1e-10;
                abs_tolerance_ = 1e-11;
                break;
            default:
                rel_tolerance_ = 1e-5;
                abs_tolerance_ = 1e-6;
        }
    }

    Device device_;
    DType dtype_;
    double rel_tolerance_;
    double abs_tolerance_;
};

// ============================================================================
// ConvNeXt-Tiny Tests (Multi-DType)
// ============================================================================

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyForwardShape) {
    auto model = convnext_tiny(1000, false);
    Variable input(Tensor({2, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyGradientFlow) {
    auto model = convnext_tiny(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype_);

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);

    // Verify all parameter gradients exist and have correct dtype
    for (const auto& param : params) {
        if (param->requires_grad()) {
            EXPECT_TRUE(param->grad().has_value());
            EXPECT_EQ(param->grad()->dtype(), dtype_);
        }
    }
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyParameterCount) {
    auto model = convnext_tiny(1000, false);
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
    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);

    EXPECT_EQ(output.tensor().dtype(), dtype_);
    EXPECT_FALSE(std::isnan(output.tensor().item<float>()));
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyLayerScaling) {
    // Test layer scaling functionality
    auto model = convnext_tiny(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Verify gradients are properly scaled
    EXPECT_TRUE(input.grad().has_value());
    auto grad_data = input.grad().value().data<float>();
    bool has_nonzero_grad = false;
    for (size_t i = 0; i < input.grad().value().numel(); ++i) {
        if (std::abs(grad_data[i]) > abs_tolerance_) {
            has_nonzero_grad = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero_grad);
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyGELUActivation) {
    // Test GELU activation through forward pass
    auto model = convnext_tiny(10, false);
    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);

    // GELU should produce bounded outputs for reasonable inputs
    auto output_data = output.tensor().data<float>();
    for (size_t i = 0; i < std::min(size_t(100), static_cast<size_t>(output.tensor().numel())); ++i) {
        EXPECT_FALSE(std::isnan(output_data[i]));
        EXPECT_FALSE(std::isinf(output_data[i]));
    }
}

// ============================================================================
// ConvNeXt-Small Tests (Multi-DType)
// ============================================================================

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtSmallForwardShape) {
    auto model = convnext_small(1000, false);
    Variable input(Tensor({2, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtSmallGradientFlow) {
    auto model = convnext_small(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype_);
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtSmallDepthwiseConvolution) {
    auto model = convnext_small(10, false);
    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);

    EXPECT_EQ(output.tensor().dtype(), dtype_);
    EXPECT_FALSE(std::isnan(output.tensor().item<float>()));
}

// ============================================================================
// ConvNeXt-Base Tests (Multi-DType)
// ============================================================================

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtBaseForwardShape) {
    auto model = convnext_base(1000, false);
    Variable input(Tensor({2, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtBaseGradientFlow) {
    auto model = convnext_base(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype_);
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtBaseParameterCount) {
    auto model = convnext_base(1000, false);
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
    auto model = convnext_base(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    auto grad_data = input.grad().value().data<float>();
    bool has_nonzero_grad = false;
    for (size_t i = 0; i < input.grad().value().numel(); ++i) {
        if (std::abs(grad_data[i]) > abs_tolerance_) {
            has_nonzero_grad = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero_grad);
}

// ============================================================================
// ConvNeXt-Large Tests (Multi-DType)
// ============================================================================

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtLargeForwardShape) {
    auto model = convnext_large(1000, false);
    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 1000}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtLargeGradientFlow) {
    auto model = convnext_large(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype_);
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtLargeParameterCount) {
    auto model = convnext_large(1000, false);
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
    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);

    auto output_data = output.tensor().data<float>();
    for (size_t i = 0; i < std::min(size_t(100), static_cast<size_t>(output.tensor().numel())); ++i) {
        EXPECT_FALSE(std::isnan(output_data[i]));
        EXPECT_FALSE(std::isinf(output_data[i]));
    }
}

// ============================================================================
// ConvNeXt-XLarge Tests (Multi-DType)
// ============================================================================

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtXLargeForwardShape) {
    auto model = convnext_xlarge(1000, false);
    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 1000}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtXLargeGradientFlow) {
    auto model = convnext_xlarge(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype_);
}

// ============================================================================
// Edge Case Tests (Multi-DType)
// ============================================================================

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyBatchSizeOne) {
    auto model = convnext_tiny(10, false);
    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 10}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyCustomClasses) {
    auto model = convnext_tiny(100, false);
    Variable input(Tensor({2, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 100}));
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtTinyMultipleBatchSizes) {
    auto model = convnext_tiny(10, false);

    // Test with different batch sizes
    for (int batch_size : {1, 2, 4, 8}) {
        Variable input(Tensor({batch_size, 3, 224, 224}, dtype_, device_), true);
        Variable output = model->forward(input);

        auto shape = output.tensor().shape();
        EXPECT_EQ(shape[0], batch_size);
        EXPECT_EQ(shape[1], 10);
        EXPECT_EQ(output.tensor().dtype(), dtype_);
    }
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtSmallNumericalStability) {
    auto model = convnext_small(10, false);
    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);

    // Check for numerical stability (no NaN or Inf)
    auto data = output.tensor().data<float>();
    for (size_t i = 0; i < output.tensor().numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i])) << "NaN detected at index " << i;
        EXPECT_FALSE(std::isinf(data[i])) << "Inf detected at index " << i;
    }
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtBaseDepthwiseConsistency) {
    auto model = convnext_base(10, false);
    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);

    // Run forward pass twice with same input
    Variable output1 = model->forward(input);
    Variable output2 = model->forward(input);

    // Outputs should be identical (deterministic)
    auto data1 = output1.tensor().data<float>();
    auto data2 = output2.tensor().data<float>();

    for (size_t i = 0; i < output1.tensor().numel(); ++i) {
        EXPECT_NEAR(data1[i], data2[i], abs_tolerance_);
    }
}

TEST_P(ConvNeXtMultiDTypeTest, ConvNeXtLargeGradientNumericalStability) {
    auto model = convnext_large(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Check gradient numerical stability
    EXPECT_TRUE(input.grad().has_value());
    auto grad_data = input.grad().value().data<float>();
    for (size_t i = 0; i < input.grad().value().numel(); ++i) {
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
    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);

    auto tiny = convnext_tiny(num_classes, false);
    auto small = convnext_small(num_classes, false);
    auto base = convnext_base(num_classes, false);

    auto output_tiny = tiny->forward(input);
    auto output_small = small->forward(input);
    auto output_base = base->forward(input);

    EXPECT_EQ(output_tiny.tensor().shape()[1], num_classes);
    EXPECT_EQ(output_small.tensor().shape()[1], num_classes);
    EXPECT_EQ(output_base.tensor().shape()[1], num_classes);
}

TEST_P(ConvNeXtMultiDTypeTest, CrossVariantDTypePreservation) {
    // All variants should preserve dtype
    Variable input(Tensor({1, 3, 224, 224}, dtype_, device_), true);

    auto tiny = convnext_tiny(10, false);
    auto small = convnext_small(10, false);

    auto output_tiny = tiny->forward(input);
    auto output_small = small->forward(input);

    EXPECT_EQ(output_tiny.tensor().dtype(), dtype_);
    EXPECT_EQ(output_small.tensor().dtype(), dtype_);
}

// ============================================================================
// Instantiate Tests for Multiple DTypes
// ============================================================================

INSTANTIATE_TEST_SUITE_P(
    MultiDType,
    ConvNeXtMultiDTypeTest,
    ::testing::Values(DType::Float32, DType::Float64, DType::Float16),
    [](const ::testing::TestParamInfo<DType>& info) {
        switch (info.param) {
            case DType::Float32: return "Float32";
            case DType::Float64: return "Float64";
            case DType::Float16: return "Float16";
            default: return "Unknown";
        }
    }
);

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
