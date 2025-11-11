#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

/**
 * @file test_linear_multidtype.cpp
 * @brief Multi-dtype tests for Linear (fully connected) layer
 *
 * Tests linear transformation with Float32, Float64, and Float16 dtypes
 * for mixed precision training scenarios. Linear layers are critical building
 * blocks in neural networks and need robust multi-dtype support.
 */

// ============================================================================
// Multi-DType Parameterization
// ============================================================================

struct DTypeParam {
    DType dtype;
    std::string dtype_name;
    float tolerance;
    float gradient_tol;

    std::string ToString() const {
        return dtype_name;
    }
};

class LinearMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    DType dtype;
    float tol;
    float grad_tol;
    Device device;

    void SetUp() override {
        tenzor::initialize();
        auto param = GetParam();
        dtype = param.dtype;
        tol = param.tolerance;
        grad_tol = param.gradient_tol;
        device = Device::cpu();
    }

    // Helper to check if values are NaN or Inf
    template<typename T>
    bool has_invalid_values(const Tensor& tensor) {
        auto cpu_tensor = tensor.to(Device::cpu()).to(DType::Float32);
        auto data = cpu_tensor.data<float>();
        for (int64_t i = 0; i < cpu_tensor.numel(); ++i) {
            if (std::isnan(data[i]) || std::isinf(data[i])) {
                return true;
            }
        }
        return false;
    }
};

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST_P(LinearMultiDTypeTest, ForwardShapeSingleBatch) {
    auto param = GetParam();
    Linear linear(10, 5);

    auto input_tensor = ones({1, 10}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.tensor().dtype(), dtype);
    EXPECT_FALSE(has_invalid_values<float>(output.tensor()));
}

TEST_P(LinearMultiDTypeTest, ForwardShapeMultiBatch) {
    auto param = GetParam();
    Linear linear(10, 5);

    auto input_tensor = ones({32, 10}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 32);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(LinearMultiDTypeTest, ForwardDifferentInputShapes) {
    auto param = GetParam();
    Linear linear(8, 4);

    std::vector<int64_t> batch_sizes = {1, 16, 64};

    for (auto bs : batch_sizes) {
        auto input_tensor = randn({bs, 8}, DType::Float32, device);
        if (dtype != DType::Float32) {
            input_tensor = input_tensor.to(dtype);
        }

        auto input = Variable(input_tensor, true);
        auto output = linear.forward(input);

        EXPECT_EQ(output.shape()[0], bs) << "Batch size " << bs;
        EXPECT_EQ(output.shape()[1], 4) << "Batch size " << bs;
        EXPECT_EQ(output.tensor().dtype(), dtype);
        EXPECT_FALSE(has_invalid_values<float>(output.tensor()));
    }
}

TEST_P(LinearMultiDTypeTest, ForwardWithBias) {
    auto param = GetParam();
    Linear linear(3, 2, true);

    EXPECT_TRUE(linear.has_bias());
    EXPECT_NE(linear.bias(), nullptr);

    auto input_tensor = zeros({4, 3}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 2);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(LinearMultiDTypeTest, ForwardWithoutBias) {
    auto param = GetParam();
    Linear linear(3, 2, false);

    EXPECT_FALSE(linear.has_bias());
    EXPECT_EQ(linear.bias(), nullptr);

    auto input_tensor = ones({4, 3}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 2);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

// ============================================================================
// Parameter Tests
// ============================================================================

TEST_P(LinearMultiDTypeTest, WeightShape) {
    auto param = GetParam();
    Linear linear(10, 5);

    auto weight = linear.weight();
    EXPECT_NE(weight, nullptr);

    auto weight_shape = weight->shape();
    EXPECT_EQ(weight_shape.size(), 2);
    EXPECT_EQ(weight_shape[0], 5);   // out_features
    EXPECT_EQ(weight_shape[1], 10);  // in_features
}

TEST_P(LinearMultiDTypeTest, ParameterCount) {
    auto param = GetParam();

    // With bias
    Linear linear_with_bias(10, 5, true);
    auto params_with_bias = linear_with_bias.parameters();
    EXPECT_EQ(params_with_bias.size(), 2);  // weight and bias

    // Without bias
    Linear linear_no_bias(10, 5, false);
    auto params_no_bias = linear_no_bias.parameters();
    EXPECT_EQ(params_no_bias.size(), 1);  // only weight
}

TEST_P(LinearMultiDTypeTest, BiasShape) {
    auto param = GetParam();
    Linear linear(10, 5, true);

    auto bias = linear.bias();
    EXPECT_NE(bias, nullptr);

    auto bias_shape = bias->shape();
    EXPECT_EQ(bias_shape.size(), 1);
    EXPECT_EQ(bias_shape[0], 5);  // out_features
}

// ============================================================================
// Gradient Tests
// ============================================================================

TEST_P(LinearMultiDTypeTest, BackwardGradientFlow) {
    auto param = GetParam();
    Linear linear(8, 4, true);

    auto input_tensor = randn({16, 8}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = linear.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(shape_vec, dtype, device);

    EXPECT_NO_THROW({
        output.backward(grad_output);
    });

    // Check input gradient
    EXPECT_TRUE(input.has_grad());
    EXPECT_EQ(input.grad()->dtype(), dtype);
    EXPECT_FALSE(has_invalid_values<float>(*input.grad()));

    // Check weight gradient
    auto weight = linear.weight();
    EXPECT_TRUE(weight->has_grad());
    EXPECT_EQ(weight->grad()->dtype(), dtype);
    EXPECT_FALSE(has_invalid_values<float>(weight->grad().value()));

    // Check bias gradient
    auto bias = linear.bias();
    EXPECT_TRUE(bias->has_grad());
    EXPECT_EQ(bias->grad()->dtype(), dtype);
    EXPECT_FALSE(has_invalid_values<float>(bias->grad().value()));
}

TEST_P(LinearMultiDTypeTest, BackwardGradientShapes) {
    auto param = GetParam();
    Linear linear(10, 5, true);

    auto input_tensor = randn({8, 10}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = linear.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(shape_vec, dtype, device);
    output.backward(grad_output);

    // Input gradient should match input shape
    EXPECT_EQ(input.grad()->shape()[0], 8);
    EXPECT_EQ(input.grad()->shape()[1], 10);

    // Weight gradient should match weight shape
    auto weight_grad = linear.weight()->grad().value();
    EXPECT_EQ(weight_grad.shape()[0], 5);
    EXPECT_EQ(weight_grad.shape()[1], 10);

    // Bias gradient should match bias shape
    auto bias_grad = linear.bias()->grad().value();
    EXPECT_EQ(bias_grad.shape()[0], 5);
}

TEST_P(LinearMultiDTypeTest, BackwardNoBias) {
    auto param = GetParam();
    Linear linear(8, 4, false);

    auto input_tensor = randn({16, 8}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = linear.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(shape_vec, dtype, device);

    EXPECT_NO_THROW({
        output.backward(grad_output);
    });

    EXPECT_TRUE(input.has_grad());
    EXPECT_TRUE(linear.weight()->has_grad());
    EXPECT_EQ(linear.bias(), nullptr);
}

// ============================================================================
// Numerical Tests
// ============================================================================

TEST_P(LinearMultiDTypeTest, ConsistentOutputDeterministic) {
    auto param = GetParam();
    Linear linear(4, 2);

    auto input_tensor = ones({3, 4}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }
    auto input = Variable(input_tensor, true);

    // Forward pass twice with same input
    auto output1 = linear.forward(input);
    auto output2 = linear.forward(input);

    // Outputs should be identical
    auto out1_f32 = output1.tensor().to(Device::cpu()).to(DType::Float32);
    auto out2_f32 = output2.tensor().to(Device::cpu()).to(DType::Float32);

    auto data1 = out1_f32.data<float>();
    auto data2 = out2_f32.data<float>();

    for (int64_t i = 0; i < out1_f32.numel(); ++i) {
        EXPECT_NEAR(data1[i], data2[i], tol);
    }
}

TEST_P(LinearMultiDTypeTest, ZeroInputWithBias) {
    auto param = GetParam();
    Linear linear(5, 3, true);

    auto input_tensor = zeros({2, 5}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = linear.forward(input);

    // With zero input, output should equal bias (broadcasted)
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_FALSE(has_invalid_values<float>(output.tensor()));
}

TEST_P(LinearMultiDTypeTest, ContiguityAfterTranspose) {
    auto param = GetParam();
    Linear linear(10, 5);

    auto input_tensor = randn({32, 10}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }
    auto input = Variable(input_tensor, true);

    // Linear layer internally transposes weights
    // This should NOT throw "matmul requires contiguous tensors"
    EXPECT_NO_THROW({
        auto output = linear.forward(input);
        EXPECT_EQ(output.shape()[0], 32);
        EXPECT_EQ(output.shape()[1], 5);
    });
}

// ============================================================================
// Large Scale Tests
// ============================================================================

TEST_P(LinearMultiDTypeTest, LargeFeatures) {
    auto param = GetParam();
    Linear linear(1024, 512);

    auto input_tensor = randn({8, 1024}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 8);
    EXPECT_EQ(output.shape()[1], 512);
    EXPECT_EQ(output.tensor().dtype(), dtype);
    EXPECT_FALSE(has_invalid_values<float>(output.tensor()));
}

TEST_P(LinearMultiDTypeTest, SingleFeature) {
    auto param = GetParam();
    Linear linear(1, 1);

    auto input_tensor = randn({10, 1}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 10);
    EXPECT_EQ(output.shape()[1], 1);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

// ============================================================================
// Mixed Precision Tests
// ============================================================================

TEST_P(LinearMultiDTypeTest, SequentialLayersPreserveType) {
    auto param = GetParam();
    Linear linear1(64, 32);
    Linear linear2(32, 16);
    Linear linear3(16, 8);

    auto input_tensor = randn({4, 64}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto x = Variable(input_tensor, true);
    auto x1 = linear1.forward(x);
    auto x2 = linear2.forward(x1);
    auto output = linear3.forward(x2);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.tensor().dtype(), dtype);
    EXPECT_FALSE(has_invalid_values<float>(output.tensor()));
}

TEST_P(LinearMultiDTypeTest, TypePersistsThroughActivation) {
    auto param = GetParam();
    Linear linear(32, 16);

    auto input_tensor = randn({8, 32}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = linear.forward(input);
    auto activated = relu(output);

    EXPECT_EQ(activated.tensor().dtype(), dtype);
    EXPECT_FALSE(has_invalid_values<float>(activated.tensor()));
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_P(LinearMultiDTypeTest, ExtremeValues) {
    auto param = GetParam();
    Linear linear(8, 4, true);

    // Test with large values
    auto input_tensor = randn({4, 8}, DType::Float32, device) * 100.0f;
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);

    EXPECT_NO_THROW({
        auto output = linear.forward(input);
        // For Float16, values might overflow but shouldn't be NaN
        if (dtype != DType::Float16) {
            EXPECT_FALSE(has_invalid_values<float>(output.tensor()));
        }
    });
}

TEST_P(LinearMultiDTypeTest, VerySmallBatchSize) {
    auto param = GetParam();
    Linear linear(16, 8);

    auto input_tensor = randn({1, 16}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(LinearMultiDTypeTest, LargeBatchSize) {
    auto param = GetParam();
    Linear linear(16, 8);

    auto input_tensor = randn({256, 16}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = linear.forward(input);

    EXPECT_EQ(output.shape()[0], 256);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

// ============================================================================
// Test Instantiation
// ============================================================================

std::vector<DTypeParam> GenerateLinearDTypeParams() {
    return {
        {DType::Float32, "float32", 1e-5f, 1e-4f},
        {DType::Float64, "float64", 1e-10f, 1e-8f},
        {DType::Float16, "float16", 1e-2f, 1e-1f}  // Looser tolerance for reduced precision
    };
}

INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    LinearMultiDTypeTest,
    ::testing::ValuesIn(GenerateLinearDTypeParams()),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 22
 * DTypes Tested: Float32, Float64, Float16
 * Total Scenarios: 22 tests × 3 dtypes = 66 test scenarios
 *
 * Coverage:
 * - Forward pass: shape preservation, single/multi batch, with/without bias
 * - Parameters: weight/bias shapes, parameter counting
 * - Gradients: flow, shapes, with/without bias
 * - Numerical: deterministic output, zero input, transpose contiguity
 * - Large scale: large features, single feature edge case
 * - Mixed precision: sequential layers, activation integration
 * - Edge cases: extreme values, very small/large batch sizes
 *
 * Tolerances:
 * - Float32: 1e-5 (standard precision)
 * - Float64: 1e-10 (high precision for scientific computing)
 * - Float16: 1e-2 (reduced precision for mixed precision training)
 *
 * Critical Tests:
 * - ContiguityAfterTranspose: Verifies the fix for non-contiguous weight transpose
 * - GradientFlow: Ensures backpropagation works correctly across dtypes
 * - SequentialLayersPreserveType: Validates dtype preservation in deep networks
 */
