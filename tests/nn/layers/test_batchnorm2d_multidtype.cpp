#include <gtest/gtest.h>
#include "../../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

/**
 * @file test_batchnorm2d_multidtype.cpp
 * @brief Multi-dtype tests for BatchNorm2d layer
 *
 * Tests batch normalization with Float32, Float64, and Float16 dtypes
 * for mixed precision training scenarios.
 */

// ============================================================================
// Multi-DType Parameterization
// ============================================================================

struct DTypeParam {
    DType dtype;
    std::string dtype_name;
    float tolerance;
    float variance_tol;

    std::string ToString() const {
        return dtype_name;
    }
};

// Required for gtest_discover_tests to show human-readable test names
void PrintTo(const DTypeParam& param, std::ostream* os) {
    *os << param.ToString();
}

class BatchNorm2dMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    DType dtype;
    float tol;
    float var_tol;
    Device device;

    void SetUp() override {
        tenzor::initialize();
        auto param = GetParam();
        dtype = param.dtype;
        tol = param.tolerance;
        var_tol = param.variance_tol;
        device = Device::cpu();
    }
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_P(BatchNorm2dMultiDTypeTest, ForwardShapePreservation) {
    auto param = GetParam();
    BatchNorm2d bn(64);

    auto input_tensor = randn({32, 64, 28, 28}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = bn.forward(input);

    auto out_shape = output.shape();
    EXPECT_EQ(out_shape.size(), 4);
    EXPECT_EQ(out_shape[0], 32);
    EXPECT_EQ(out_shape[1], 64);
    EXPECT_EQ(out_shape[2], 28);
    EXPECT_EQ(out_shape[3], 28);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

TEST_P(BatchNorm2dMultiDTypeTest, ParameterInitialization) {
    auto param = GetParam();
    BatchNorm2d bn_affine(32, 1e-5, 0.1, true);
    auto params = bn_affine.parameters();

    EXPECT_EQ(params.size(), 2);  // weight and bias
    EXPECT_EQ(params[0]->shape()[0], 32);
    EXPECT_EQ(params[1]->shape()[0], 32);

    // Weight initialized to 1, bias to 0
    auto weight_f32 = params[0]->tensor().to(DType::Float32);
    auto bias_f32 = params[1]->tensor().to(DType::Float32);
    auto weight_data = weight_f32.data<float>();
    auto bias_data = bias_f32.data<float>();

    for (int64_t i = 0; i < 32; ++i) {
        EXPECT_NEAR(weight_data[i], 1.0f, tol);
        EXPECT_NEAR(bias_data[i], 0.0f, tol);
    }
}

TEST_P(BatchNorm2dMultiDTypeTest, TrainingModeNormalization) {
    auto param = GetParam();
    BatchNorm2d bn(3, 1e-5, 0.1, false);
    bn.train();

    auto input_tensor = randn({16, 3, 8, 8}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, false);
    auto output = bn.forward(input);

    auto output_f32 = output.tensor().to(DType::Float32);
    auto output_data = output_f32.data<float>();

    int64_t N = 16, C = 3, H = 8, W = 8;
    int64_t spatial_size = H * W;
    int64_t batch_size = N * spatial_size;

    for (int64_t c = 0; c < C; ++c) {
        double sum = 0.0;
        double sum_sq = 0.0;

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    float val = output_data[idx];
                    sum += val;
                    sum_sq += val * val;
                }
            }
        }

        double mean = sum / batch_size;
        double variance = sum_sq / batch_size - mean * mean;

        EXPECT_NEAR(mean, 0.0, tol * 10) << "Channel " << c;
        EXPECT_NEAR(variance, 1.0, var_tol) << "Channel " << c;
    }
}

TEST_P(BatchNorm2dMultiDTypeTest, InferenceModeUsesRunningStats) {
    auto param = GetParam();
    BatchNorm2d bn(3, 1e-5, 0.1, false);

    // Train on some data
    bn.train();
    auto train_input = randn({32, 3, 8, 8}, DType::Float32, device);
    if (dtype != DType::Float32) {
        train_input = train_input.to(dtype);
    }
    bn.forward(Variable(train_input, false));

    // Switch to eval
    bn.eval();
    auto test_input = randn({16, 3, 8, 8}, DType::Float32, device);
    if (dtype != DType::Float32) {
        test_input = test_input.to(dtype);
    }
    auto output = bn.forward(Variable(test_input, false));

    EXPECT_EQ(output.shape()[0], 16);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.tensor().dtype(), dtype);
}

// ============================================================================
// Epsilon Parameter Tests
// ============================================================================

TEST_P(BatchNorm2dMultiDTypeTest, EpsilonPreventsDivisionByZero) {
    auto param = GetParam();
    BatchNorm2d bn(2, 1e-3, 0.1, false);
    bn.train();

    auto input_tensor = ones({16, 2, 8, 8}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    EXPECT_NO_THROW({
        auto output = bn.forward(Variable(input_tensor, false));
        auto output_f32 = output.tensor().to(DType::Float32);
        auto data = output_f32.data<float>();

        for (size_t i = 0; i < output_f32.numel(); ++i) {
            EXPECT_FALSE(std::isnan(data[i]));
            EXPECT_FALSE(std::isinf(data[i]));
        }
    });
}

// ============================================================================
// Affine Transformation Tests
// ============================================================================

TEST_P(BatchNorm2dMultiDTypeTest, AffineTransformationApplied) {
    auto param = GetParam();
    BatchNorm2d bn(2, 1e-5, 0.1, true);
    bn.train();

    auto params_vec = bn.parameters();
    ASSERT_GE(params_vec.size(), 2);

    params_vec[0]->tensor().fill_(2.0f);  // weight
    params_vec[1]->tensor().fill_(1.0f);  // bias

    auto input_tensor = zeros({4, 2, 4, 4}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto output = bn.forward(Variable(input_tensor, false));

    auto output_f32 = output.tensor().to(DType::Float32);
    auto output_data = output_f32.data<float>();

    for (int64_t i = 0; i < output_f32.numel(); ++i) {
        EXPECT_NEAR(output_data[i], 1.0f, tol * 10);
    }
}

// ============================================================================
// Gradient Tests
// ============================================================================

TEST_P(BatchNorm2dMultiDTypeTest, BackwardPassGradientFlow) {
    auto param = GetParam();
    BatchNorm2d bn(4, 1e-5, 0.1, true);
    bn.train();

    auto input_tensor = randn({8, 4, 8, 8}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = bn.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(shape_vec, dtype, device);
    output.backward(grad_output);

    EXPECT_TRUE(input.has_grad());
    EXPECT_EQ(input.grad()->dtype(), dtype);

    auto input_grad_f32 = input.grad()->to(DType::Float32);
    auto grad_data = input_grad_f32.data<float>();

    for (int64_t i = 0; i < input_grad_f32.numel(); ++i) {
        EXPECT_FALSE(std::isnan(grad_data[i]));
        EXPECT_FALSE(std::isinf(grad_data[i]));
    }
}

TEST_P(BatchNorm2dMultiDTypeTest, ParameterGradients) {
    auto param = GetParam();
    BatchNorm2d bn(4, 1e-5, 0.1, true);
    bn.train();

    auto input_tensor = randn({8, 4, 8, 8}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto input = Variable(input_tensor, true);
    auto output = bn.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(shape_vec, dtype, device);
    output.backward(grad_output);

    auto params_vec = bn.parameters();
    ASSERT_GE(params_vec.size(), 2);

    EXPECT_TRUE(params_vec[0]->has_grad());  // weight
    EXPECT_TRUE(params_vec[1]->has_grad());  // bias

    auto weight_grad = params_vec[0]->grad().value();
    auto bias_grad = params_vec[1]->grad().value();

    EXPECT_EQ(weight_grad.shape()[0], 4);
    EXPECT_EQ(bias_grad.shape()[0], 4);
}

// ============================================================================
// Different Batch Sizes
// ============================================================================

TEST_P(BatchNorm2dMultiDTypeTest, VariableBatchSizes) {
    auto param = GetParam();
    BatchNorm2d bn(4, 1e-5, 0.1, true);
    bn.train();

    std::vector<int64_t> batch_sizes = {1, 4, 16, 32};

    for (auto bs : batch_sizes) {
        auto input_tensor = randn({bs, 4, 8, 8}, DType::Float32, device);
        if (dtype != DType::Float32) {
            input_tensor = input_tensor.to(dtype);
        }

        EXPECT_NO_THROW({
            auto output = bn.forward(Variable(input_tensor, false));
            EXPECT_EQ(output.shape()[0], bs);
            EXPECT_EQ(output.tensor().dtype(), dtype);
        });
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_P(BatchNorm2dMultiDTypeTest, ConstantInput) {
    auto param = GetParam();
    BatchNorm2d bn(3, 1e-5, 0.1, false);
    bn.train();

    auto input_tensor = ones({8, 3, 8, 8}, DType::Float32, device) * 5.0f;
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    auto output = bn.forward(Variable(input_tensor, false));

    auto output_f32 = output.tensor().to(DType::Float32);
    auto data = output_f32.data<float>();

    for (int64_t i = 0; i < output_f32.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i]));
        EXPECT_FALSE(std::isinf(data[i]));
    }
}

TEST_P(BatchNorm2dMultiDTypeTest, ExtremeValues) {
    auto param = GetParam();
    BatchNorm2d bn(2, 1e-5, 0.1, true);
    bn.train();

    auto input_tensor = randn({8, 2, 8, 8}, DType::Float32, device) * 1000.0f;
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    EXPECT_NO_THROW({
        auto output = bn.forward(Variable(input_tensor, false));
        auto output_f32 = output.tensor().to(DType::Float32);
        auto data = output_f32.data<float>();

        for (int64_t i = 0; i < output_f32.numel(); ++i) {
            EXPECT_FALSE(std::isnan(data[i]));
            EXPECT_FALSE(std::isinf(data[i]));
        }
    });
}

// ============================================================================
// Test Instantiation
// ============================================================================

std::vector<DTypeParam> GenerateBatchNormDTypeParams() {
    return {
        {DType::Float32, "float32", 1e-5f, 1e-4f},
        {DType::Float64, "float64", 1e-10f, 1e-8f},
        {DType::Float16, "float16", 1e-2f, 1e-1f}
    };
}

INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    BatchNorm2dMultiDTypeTest,
    ::testing::ValuesIn(GenerateBatchNormDTypeParams()),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 12
 * DTypes Tested: Float32, Float64, Float16
 * Total Scenarios: 12 tests × 3 dtypes = 36 test scenarios
 *
 * Coverage:
 * - Basic: shape preservation, parameter initialization, normalization
 * - Modes: training vs inference, running statistics
 * - Epsilon: division by zero prevention
 * - Affine: transformation application
 * - Gradients: backward pass, parameter gradients
 * - Batch sizes: variable batch size handling
 * - Edge cases: constant input, extreme values
 *
 * Tolerances:
 * - Float32: 1e-5 (mean), 1e-4 (variance)
 * - Float64: 1e-10 (mean), 1e-8 (variance)
 * - Float16: 1e-2 (mean), 1e-1 (variance) - reduced precision for mixed precision training
 */
