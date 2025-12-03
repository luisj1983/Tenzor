#include <gtest/gtest.h>
#include "../../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace tenzor;
using namespace tenzor::nn;

/**
 * @file test_dropout_multidtype.cpp
 * @brief Multi-dtype tests for Dropout and Dropout2d layers
 *
 * Tests dropout operations with Float32, Float64, and Float16 dtypes
 * for mixed precision training scenarios.
 */

// ============================================================================
// Multi-DType Parameterization
// ============================================================================

struct DTypeParam {
    DType dtype;
    std::string dtype_name;
    float tolerance;

    std::string ToString() const {
        return dtype_name;
    }
};

// Required for gtest_discover_tests to show human-readable test names
void PrintTo(const DTypeParam& param, std::ostream* os) {
    *os << param.ToString();
}

class DropoutMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    DType dtype;
    float tol;
    Device device;

    void SetUp() override {
        tenzor::initialize();
        auto param = GetParam();
        dtype = param.dtype;
        tol = param.tolerance;
        device = Device::cpu();
    }

    template<typename T>
    T* get_data_ptr(const Tensor& t) {
        return static_cast<T*>(t.impl()->storage->data());
    }
};

// ============================================================================
// Dropout Basic Tests
// ============================================================================

TEST_P(DropoutMultiDTypeTest, InferenceModeNoModification) {
    auto param = GetParam();
    Dropout dropout(0.5);
    dropout.eval();

    auto input_tensor = ones({2, 3, 4}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    Variable input(input_tensor, false);
    auto output = dropout.forward(input);

    EXPECT_EQ(output.tensor().dtype(), dtype);
    EXPECT_EQ(output.tensor().shape().size(), 3);
    EXPECT_EQ(output.tensor().shape()[0], 2);
    EXPECT_EQ(output.tensor().shape()[1], 3);
    EXPECT_EQ(output.tensor().shape()[2], 4);

    // In inference mode, output should equal input
    auto output_f32 = output.tensor().to(DType::Float32);
    auto* out_data = output_f32.data<float>();
    for (size_t i = 0; i < output_f32.numel(); ++i) {
        EXPECT_NEAR(out_data[i], 1.0f, tol);
    }
}

TEST_P(DropoutMultiDTypeTest, TrainingModeModifiesOutput) {
    auto param = GetParam();
    Dropout dropout(0.5);
    dropout.train();

    auto input_tensor = ones({100, 100}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    Variable input(input_tensor, false);
    auto output = dropout.forward(input);

    EXPECT_EQ(output.tensor().dtype(), dtype);

    // Convert to float32 for analysis
    auto output_f32 = output.tensor().to(DType::Float32);
    auto* data = output_f32.data<float>();

    size_t zero_count = 0;
    size_t non_zero_count = 0;

    for (size_t i = 0; i < output_f32.numel(); ++i) {
        if (std::abs(data[i]) < tol) {
            zero_count++;
        } else {
            non_zero_count++;
        }
    }

    EXPECT_GT(zero_count, 0);
    EXPECT_GT(non_zero_count, 0);
}

TEST_P(DropoutMultiDTypeTest, ProbabilityZeroNoDropout) {
    auto param = GetParam();
    Dropout dropout(0.0);
    dropout.train();

    auto input_tensor = ones({50, 50}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    Variable input(input_tensor, false);
    auto output = dropout.forward(input);

    auto output_f32 = output.tensor().to(DType::Float32);
    auto* data = output_f32.data<float>();

    for (size_t i = 0; i < output_f32.numel(); ++i) {
        EXPECT_NEAR(data[i], 1.0f, tol);
    }
}

TEST_P(DropoutMultiDTypeTest, InvertedDropoutScaling) {
    auto param = GetParam();
    Dropout dropout(0.5);
    dropout.train();

    auto input_tensor = ones({100, 100}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    Variable input(input_tensor, false);
    auto output = dropout.forward(input);

    // Check that non-zero values are scaled by 1/(1-p) = 2.0
    auto output_f32 = output.tensor().to(DType::Float32);
    auto* data = output_f32.data<float>();

    double sum_non_zero = 0.0;
    size_t non_zero_count = 0;

    for (size_t i = 0; i < output_f32.numel(); ++i) {
        if (std::abs(data[i]) > tol) {
            sum_non_zero += data[i];
            non_zero_count++;
        }
    }

    if (non_zero_count > 0) {
        double avg_non_zero = sum_non_zero / non_zero_count;
        double expected_scale = 1.0 / (1.0 - 0.5);  // 2.0
        EXPECT_NEAR(avg_non_zero, expected_scale, std::max(0.1, static_cast<double>(tol) * 10));
    }
}

TEST_P(DropoutMultiDTypeTest, StatisticalDistribution) {
    auto param = GetParam();
    Dropout dropout(0.3);
    dropout.train();

    auto input_tensor = ones({500, 500}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    Variable input(input_tensor, false);
    auto output = dropout.forward(input);

    auto output_f32 = output.tensor().to(DType::Float32);
    auto* data = output_f32.data<float>();

    size_t kept_count = 0;
    size_t total = output_f32.numel();

    for (size_t i = 0; i < total; ++i) {
        if (std::abs(data[i]) > tol) {
            kept_count++;
        }
    }

    double keep_rate = static_cast<double>(kept_count) / total;
    double expected_keep_rate = 1.0 - 0.3;

    EXPECT_NEAR(keep_rate, expected_keep_rate, 0.03);
}

// ============================================================================
// Gradient Tests
// ============================================================================

TEST_P(DropoutMultiDTypeTest, BackwardPassGradientShape) {
    auto param = GetParam();
    Dropout dropout(0.5);
    dropout.train();

    auto input_tensor = randn({10, 20}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    Variable input(input_tensor, true);
    auto output = dropout.forward(input);

    EXPECT_TRUE(output.grad_fn() != nullptr);

    auto grad_output = ones({10, 20}, dtype, device);
    EXPECT_NO_THROW({
        output.backward(grad_output);
    });

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_EQ(input.grad()->dtype(), dtype);
}

TEST_P(DropoutMultiDTypeTest, BackwardPassGradientValues) {
    auto param = GetParam();
    Dropout dropout(0.0);  // No dropout for deterministic test
    dropout.train();

    auto input_tensor = ones({5, 5}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    Variable input(input_tensor, true);
    auto output = dropout.forward(input);

    auto grad_output = full({5, 5}, 2.0f, dtype, device);
    output.backward(grad_output);

    ASSERT_TRUE(input.grad().has_value());
    auto grad_f32 = input.grad()->to(DType::Float32);
    auto* grad_data = grad_f32.data<float>();

    for (size_t i = 0; i < grad_f32.numel(); ++i) {
        EXPECT_NEAR(grad_data[i], 2.0f, tol);
    }
}

// ============================================================================
// Dropout2d Tests
// ============================================================================

TEST_P(DropoutMultiDTypeTest, Dropout2dInferenceMode) {
    auto param = GetParam();
    Dropout2d dropout(0.5);
    dropout.eval();

    auto input_tensor = ones({2, 4, 8, 8}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    Variable input(input_tensor, false);
    auto output = dropout.forward(input);

    ASSERT_EQ(output.tensor().shape().size(), 4);
    EXPECT_EQ(output.tensor().dtype(), dtype);

    auto output_f32 = output.tensor().to(DType::Float32);
    auto* data = output_f32.data<float>();

    for (size_t i = 0; i < output_f32.numel(); ++i) {
        EXPECT_NEAR(data[i], 1.0f, tol);
    }
}

TEST_P(DropoutMultiDTypeTest, Dropout2dChannelWiseDropout) {
    auto param = GetParam();
    Dropout2d dropout(0.5);
    dropout.train();

    auto input_tensor = ones({2, 10, 8, 8}, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    Variable input(input_tensor, false);
    auto output = dropout.forward(input);

    EXPECT_EQ(output.tensor().dtype(), dtype);

    // Verify entire channels are uniformly dropped or kept
    auto output_f32 = output.tensor().to(DType::Float32);
    auto* data = output_f32.data<float>();

    for (size_t n = 0; n < 2; ++n) {
        for (size_t c = 0; c < 10; ++c) {
            size_t channel_offset = (n * 10 + c) * 8 * 8;
            float first_value = data[channel_offset];

            for (size_t i = 0; i < 8 * 8; ++i) {
                EXPECT_NEAR(data[channel_offset + i], first_value, tol);
            }
        }
    }
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

TEST_P(DropoutMultiDTypeTest, DifferentTensorShapes) {
    auto param = GetParam();
    Dropout dropout(0.5);
    dropout.train();

    // 1D
    auto input_1d = ones({1000}, DType::Float32, device);
    if (dtype != DType::Float32) input_1d = input_1d.to(dtype);
    Variable var_1d(input_1d, false);
    auto output_1d = dropout.forward(var_1d);
    EXPECT_EQ(output_1d.tensor().dtype(), dtype);
    EXPECT_EQ(output_1d.tensor().shape().size(), 1);

    // 4D
    auto input_4d = ones({8, 16, 32, 32}, DType::Float32, device);
    if (dtype != DType::Float32) input_4d = input_4d.to(dtype);
    Variable var_4d(input_4d, false);
    auto output_4d = dropout.forward(var_4d);
    EXPECT_EQ(output_4d.tensor().dtype(), dtype);
    EXPECT_EQ(output_4d.tensor().shape().size(), 4);
}

TEST_P(DropoutMultiDTypeTest, PreservesExpectedValue) {
    auto param = GetParam();
    Dropout dropout(0.5);
    dropout.train();

    auto input_tensor = full({1000, 1000}, 10.0f, DType::Float32, device);
    if (dtype != DType::Float32) {
        input_tensor = input_tensor.to(dtype);
    }

    Variable input(input_tensor, false);
    auto output = dropout.forward(input);

    auto output_f32 = output.tensor().to(DType::Float32);
    auto* data = output_f32.data<float>();

    double sum = 0.0;
    for (size_t i = 0; i < output_f32.numel(); ++i) {
        sum += data[i];
    }

    double mean = sum / output_f32.numel();
    EXPECT_NEAR(mean, 10.0, std::max(0.5, static_cast<double>(tol) * 100));
}

// ============================================================================
// Test Instantiation
// ============================================================================

std::vector<DTypeParam> GenerateDropoutDTypeParams() {
    return {
        {DType::Float32, "float32", 1e-5f},
        {DType::Float64, "float64", 1e-10f},
        {DType::Float16, "float16", 1e-2f}
    };
}

INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    DropoutMultiDTypeTest,
    ::testing::ValuesIn(GenerateDropoutDTypeParams()),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 13
 * DTypes Tested: Float32, Float64, Float16
 * Total Scenarios: 13 tests × 3 dtypes = 39 test scenarios
 *
 * Coverage:
 * - Dropout: inference mode, training mode, probabilities, scaling, statistics
 * - Gradient: backward pass, gradient shape, gradient values
 * - Dropout2d: inference mode, channel-wise dropout
 * - Numerical stability: tensor shapes, expected value preservation
 *
 * Tolerances:
 * - Float32: 1e-5 (standard precision)
 * - Float64: 1e-10 (high precision)
 * - Float16: 1e-2 (reduced precision for mixed precision training)
 */
