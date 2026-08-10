/**
 * @file test_batchnorm3d.cpp
 * @brief Tests for BatchNorm3d layer (N, C, D, H, W) inputs.
 *
 * Mirrors the structure of test_batchnorm2d.cpp for the 3D case. BatchNorm3d
 * normalises over (N, D, H, W) per channel, and is implemented as a reshape
 * delegating to BatchNorm2d under the hood.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../backend_test_fixture.hpp"
#include "../../grad_flow_helpers.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

class BatchNorm3dTest : public ::tenzor::testing::BackendTest {};

// ==================== Forward Shape Preservation ====================

TEST_P(BatchNorm3dTest, ForwardShapePreservation) {
    BatchNorm3d bn(16);
    bn.to(device);

    auto input = Variable(randn({8, 16, 4, 8, 8}, DType::Float32, device), true);
    auto output = bn.forward(input);

    auto out_shape = output.shape();
    ASSERT_EQ(out_shape.size(), 5);
    EXPECT_EQ(out_shape[0], 8);   // N
    EXPECT_EQ(out_shape[1], 16);  // C
    EXPECT_EQ(out_shape[2], 4);   // D
    EXPECT_EQ(out_shape[3], 8);   // H
    EXPECT_EQ(out_shape[4], 8);   // W
}

TEST_P(BatchNorm3dTest, ForwardShapeLarger) {
    BatchNorm3d bn(32);
    bn.to(device);

    auto input = Variable(randn({4, 32, 8, 16, 16}, DType::Float32, device), true);
    auto output = bn.forward(input);

    auto out_shape = output.shape();
    ASSERT_EQ(out_shape.size(), 5);
    EXPECT_EQ(out_shape[0], 4);
    EXPECT_EQ(out_shape[1], 32);
    EXPECT_EQ(out_shape[2], 8);
    EXPECT_EQ(out_shape[3], 16);
    EXPECT_EQ(out_shape[4], 16);
}

// ==================== Parameter Initialization ====================

TEST_P(BatchNorm3dTest, ParameterInitialization) {
    BatchNorm3d bn_affine(32, 1e-5, 0.1, true);
    bn_affine.to(device);
    auto params = bn_affine.parameters();

    ASSERT_EQ(params.size(), 2);
    EXPECT_EQ(params[0]->shape()[0], 32);
    EXPECT_EQ(params[1]->shape()[0], 32);

    auto weight_data = params[0]->tensor().cpu().data<float>();
    auto bias_data = params[1]->tensor().cpu().data<float>();

    for (int64_t i = 0; i < 32; ++i) {
        EXPECT_FLOAT_EQ(weight_data[i], 1.0f);
        EXPECT_FLOAT_EQ(bias_data[i], 0.0f);
    }
}

TEST_P(BatchNorm3dTest, NoAffineParameters) {
    BatchNorm3d bn_no_affine(32, 1e-5, 0.1, false);
    bn_no_affine.to(device);
    EXPECT_EQ(bn_no_affine.parameters().size(), 0);
}

// ==================== Training Mode Normalization ====================

TEST_P(BatchNorm3dTest, TrainingModeNormalization) {
    BatchNorm3d bn(3, 1e-5, 0.1, false);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({16, 3, 4, 4, 4}, DType::Float32, device), false);
    auto output = bn.forward(input);

    auto output_data = output.tensor().cpu().data<float>();
    const int64_t N = 16, C = 3, D = 4, H = 4, W = 4;
    const int64_t spatial = D * H * W;
    const int64_t batch = N * spatial;

    for (int64_t c = 0; c < C; ++c) {
        double sum = 0.0;
        double sum_sq = 0.0;
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t d = 0; d < D; ++d) {
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t w = 0; w < W; ++w) {
                        int64_t idx = (((n * C + c) * D + d) * H + h) * W + w;
                        float v = output_data[idx];
                        sum += v;
                        sum_sq += v * v;
                    }
                }
            }
        }
        double mean = sum / batch;
        double variance = sum_sq / batch - mean * mean;
        EXPECT_NEAR(mean, 0.0, 1e-4) << "channel " << c;
        EXPECT_NEAR(variance, 1.0, 1e-3) << "channel " << c;
    }
}

// ==================== Backward (Gradient Flow) ====================

TEST_P(BatchNorm3dTest, BackwardPassGradientFlow) {
    BatchNorm3d bn(4, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({4, 4, 2, 4, 4}, DType::Float32, device), true);
    auto output = bn.forward(input);

    // A uniform seed gradient (ones, equivalently sum(out).backward()) is
    // degenerate for BatchNorm in train mode: with affine=true and
    // weight=ones/bias=zeros the output is zero-mean, so the true input
    // gradient is identically 0 (the backward projects out the constant
    // component) and the weight gradient collapses too — a broken backward
    // kernel would pass identically to a correct one. Use a non-degenerate
    // scalar loss, sum(out*out + out); its seed gradient 2*out+1 is not in
    // span{1, x_hat} (the eps in invstd leaves a residual), so input, weight
    // and bias gradients all flow. Mirrors the note in
    // test_grad_nn_parity.cpp::BatchNorm2dBackward.
    auto loss = tenzor::sum(output * output + output);
    loss.backward();

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

// ==================== Edge Cases ====================

TEST_P(BatchNorm3dTest, ConstantInput) {
    BatchNorm3d bn(3, 1e-5, 0.1, false);
    bn.to(device);
    bn.train();

    auto input = Variable(ones({8, 3, 2, 4, 4}, DType::Float32, device) * 5.0f, false);
    auto output = bn.forward(input);

    auto data = output.tensor().cpu().data<float>();
    for (int64_t i = 0; i < output.tensor().numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i]));
        EXPECT_FALSE(std::isinf(data[i]));
    }
}

TEST_P(BatchNorm3dTest, InferenceAfterTraining) {
    BatchNorm3d bn(4, 1e-5, 0.1, false);
    bn.to(device);

    bn.train();
    auto train_input = Variable(randn({16, 4, 2, 4, 4}, DType::Float32, device) * 2.0f + 5.0f, false);
    bn.forward(train_input);

    bn.eval();
    auto test_input = Variable(randn({8, 4, 2, 4, 4}, DType::Float32, device), false);
    auto output = bn.forward(test_input);

    auto data = output.tensor().cpu().data<float>();
    for (int64_t i = 0; i < output.tensor().numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i]));
    }
}

INSTANTIATE_BACKEND_TESTS(BatchNorm3dTest);
