/**
 * @file test_batchnorm1d.cpp
 * @brief Tests for BatchNorm1d layer (N, C) and (N, C, L) inputs.
 *
 * Mirrors the structure of test_batchnorm2d.cpp for the 1D case. BatchNorm1d
 * normalises over (N,) for 2D input or (N, L) for 3D input, per channel.
 *
 * Parameterized across all backends via BackendTest: every TEST_P creates its
 * tensors on the fixture's `device` and moves the layer there via bn.to(device).
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../backend_test_fixture.hpp"
#include "../../grad_flow_helpers.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace tenzor;
using namespace tenzor::nn;

class BatchNorm1dTest : public ::tenzor::testing::BackendTest {};

// ==================== Forward Shape Preservation ====================

TEST_P(BatchNorm1dTest, ForwardShapePreservation2D) {
    BatchNorm1d bn(64);
    bn.to(device);

    auto input = Variable(randn({32, 64}, DType::Float32, device), true);
    auto output = bn.forward(input);

    auto out_shape = output.shape();
    ASSERT_EQ(out_shape.size(), 2);
    EXPECT_EQ(out_shape[0], 32);  // N
    EXPECT_EQ(out_shape[1], 64);  // C
}

TEST_P(BatchNorm1dTest, ForwardShapePreservation3D) {
    BatchNorm1d bn(8);
    bn.to(device);

    auto input = Variable(randn({16, 8, 32}, DType::Float32, device), true);
    auto output = bn.forward(input);

    auto out_shape = output.shape();
    ASSERT_EQ(out_shape.size(), 3);
    EXPECT_EQ(out_shape[0], 16);  // N
    EXPECT_EQ(out_shape[1], 8);   // C
    EXPECT_EQ(out_shape[2], 32);  // L
}

// ==================== Parameter Initialization ====================

TEST_P(BatchNorm1dTest, ParameterInitialization) {
    BatchNorm1d bn_affine(32, 1e-5, 0.1, true);
    bn_affine.to(device);
    auto params = bn_affine.parameters();

    ASSERT_EQ(params.size(), 2);  // weight and bias
    EXPECT_EQ(params[0]->shape()[0], 32);
    EXPECT_EQ(params[1]->shape()[0], 32);

    auto weight_cpu = params[0]->tensor().cpu();
    auto bias_cpu = params[1]->tensor().cpu();
    auto weight_data = weight_cpu.data<float>();
    auto bias_data = bias_cpu.data<float>();

    for (int64_t i = 0; i < 32; ++i) {
        EXPECT_FLOAT_EQ(weight_data[i], 1.0f);
        EXPECT_FLOAT_EQ(bias_data[i], 0.0f);
    }
}

TEST_P(BatchNorm1dTest, NoAffineParameters) {
    BatchNorm1d bn_no_affine(32, 1e-5, 0.1, false);
    bn_no_affine.to(device);
    EXPECT_EQ(bn_no_affine.parameters().size(), 0);
}

// ==================== Training Mode Normalization ====================

TEST_P(BatchNorm1dTest, TrainingModeNormalization2D) {
    // Each channel should have ~zero mean, ~unit variance after forward.
    BatchNorm1d bn(3, 1e-5, 0.1, false);  // no affine
    bn.to(device);
    bn.train();

    auto input = Variable(randn({64, 3}, DType::Float32, device), false);
    auto output = bn.forward(input);

    auto output_cpu = output.tensor().cpu();
    auto output_data = output_cpu.data<float>();
    const int64_t N = 64, C = 3;

    for (int64_t c = 0; c < C; ++c) {
        double sum = 0.0;
        double sum_sq = 0.0;
        for (int64_t n = 0; n < N; ++n) {
            float v = output_data[n * C + c];
            sum += v;
            sum_sq += v * v;
        }
        double mean = sum / N;
        double variance = sum_sq / N - mean * mean;
        EXPECT_NEAR(mean, 0.0, 1e-4) << "channel " << c;
        EXPECT_NEAR(variance, 1.0, 1e-3) << "channel " << c;
    }
}

TEST_P(BatchNorm1dTest, TrainingModeNormalization3D) {
    BatchNorm1d bn(3, 1e-5, 0.1, false);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({32, 3, 32}, DType::Float32, device), false);
    auto output = bn.forward(input);

    auto output_cpu = output.tensor().cpu();
    auto output_data = output_cpu.data<float>();
    const int64_t N = 32, C = 3, L = 32;
    const int64_t batch = N * L;

    for (int64_t c = 0; c < C; ++c) {
        double sum = 0.0;
        double sum_sq = 0.0;
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t l = 0; l < L; ++l) {
                int64_t idx = (n * C + c) * L + l;
                float v = output_data[idx];
                sum += v;
                sum_sq += v * v;
            }
        }
        double mean = sum / batch;
        double variance = sum_sq / batch - mean * mean;
        EXPECT_NEAR(mean, 0.0, 1e-4) << "channel " << c;
        EXPECT_NEAR(variance, 1.0, 1e-3) << "channel " << c;
    }
}

// ==================== Backward (Gradient Flow) ====================

TEST_P(BatchNorm1dTest, BackwardPassGradientFlow2D) {
    BatchNorm1d bn(4, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({16, 4}, DType::Float32, device), true);
    auto output = bn.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    output.backward(ones(shape_vec, DType::Float32, device));

    EXPECT_GRAD_FLOWS(input);

    auto input_grad = input.grad().value().cpu();
    auto grad_data = input_grad.data<float>();
    for (int64_t i = 0; i < input_grad.numel(); ++i) {
        EXPECT_FALSE(std::isnan(grad_data[i]));
        EXPECT_FALSE(std::isinf(grad_data[i]));
    }

    auto params = bn.parameters();
    ASSERT_GE(params.size(), 2);
    EXPECT_GRAD_FLOWS(*params[0]);
    EXPECT_GRAD_FLOWS(*params[1]);
    EXPECT_EQ(params[0]->grad().value().shape()[0], 4);
    EXPECT_EQ(params[1]->grad().value().shape()[0], 4);
}

TEST_P(BatchNorm1dTest, BackwardPassGradientFlow3D) {
    BatchNorm1d bn(4, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({8, 4, 16}, DType::Float32, device), true);
    auto output = bn.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    output.backward(ones(shape_vec, DType::Float32, device));

    EXPECT_GRAD_FLOWS(input);

    auto input_grad = input.grad().value().cpu();
    auto grad_data = input_grad.data<float>();
    for (int64_t i = 0; i < input_grad.numel(); ++i) {
        EXPECT_FALSE(std::isnan(grad_data[i]));
        EXPECT_FALSE(std::isinf(grad_data[i]));
    }
}

// ==================== Edge Cases ====================

TEST_P(BatchNorm1dTest, ConstantInput) {
    BatchNorm1d bn(3, 1e-5, 0.1, false);
    bn.to(device);
    bn.train();

    auto input = Variable(ones({8, 3, 16}, DType::Float32, device) * 5.0f, false);
    auto output = bn.forward(input);

    auto output_cpu = output.tensor().cpu();
    auto data = output_cpu.data<float>();
    for (int64_t i = 0; i < output_cpu.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i]));
        EXPECT_FALSE(std::isinf(data[i]));
    }
}

TEST_P(BatchNorm1dTest, InferenceAfterTraining) {
    BatchNorm1d bn(4, 1e-5, 0.1, false);
    bn.to(device);

    bn.train();
    auto train_input = Variable(randn({32, 4, 8}, DType::Float32, device) * 2.0f + 5.0f, false);
    bn.forward(train_input);

    bn.eval();
    auto test_input = Variable(randn({8, 4, 8}, DType::Float32, device), false);
    auto output = bn.forward(test_input);

    ASSERT_EQ(output.shape()[0], 8);
    ASSERT_EQ(output.shape()[1], 4);
    ASSERT_EQ(output.shape()[2], 8);

    auto output_cpu = output.tensor().cpu();
    auto data = output_cpu.data<float>();
    for (int64_t i = 0; i < output_cpu.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i]));
    }
}

INSTANTIATE_BACKEND_TESTS(BatchNorm1dTest);
