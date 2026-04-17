/**
 * @file test_batchnorm3d_multidtype.cpp
 * @brief Multi-dtype multi-backend tests for BatchNorm3d layer.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class BatchNorm3dMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    float variance_tolerance() const {
        if (dtype() == DType::Float16) return 0.2f;
        return 1e-3f;
    }
};

TEST_P(BatchNorm3dMultiDTypeTest, ForwardShapePreservation) {
    BatchNorm3d bn(16);
    convert_model(bn);

    Variable input = createInput({8, 16, 4, 8, 8}, true);
    auto output = bn.forward(input);

    expectShape(output.tensor(), {8, 16, 4, 8, 8});
    expectDType(output.tensor());
}

TEST_P(BatchNorm3dMultiDTypeTest, ParameterInitialization) {
    BatchNorm3d bn_affine(32, 1e-5, 0.1, true);
    auto params = bn_affine.parameters();

    ASSERT_EQ(params.size(), 2);
    EXPECT_EQ(params[0]->shape()[0], 32);
    EXPECT_EQ(params[1]->shape()[0], 32);

    auto weight_f32 = params[0]->tensor().to(Device::cpu()).to(DType::Float32);
    auto bias_f32 = params[1]->tensor().to(Device::cpu()).to(DType::Float32);
    auto weight_data = weight_f32.data<float>();
    auto bias_data = bias_f32.data<float>();

    for (int64_t i = 0; i < 32; ++i) {
        EXPECT_NEAR(weight_data[i], 1.0f, atol());
        EXPECT_NEAR(bias_data[i], 0.0f, atol());
    }
}

TEST_P(BatchNorm3dMultiDTypeTest, TrainingModeNormalization) {
    BatchNorm3d bn(3, 1e-5, 0.1, false);
    convert_model(bn);
    bn.train();

    Variable input = createInput({16, 3, 4, 4, 4}, false);
    auto output = bn.forward(input);

    auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto output_data = output_f32.data<float>();

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
        EXPECT_NEAR(mean, 0.0, rtol() + atol());
        EXPECT_NEAR(variance, 1.0, variance_tolerance());
    }
}

TEST_P(BatchNorm3dMultiDTypeTest, BackwardPassGradientFlow) {
    BatchNorm3d bn(4, 1e-5, 0.1, true);
    convert_model(bn);
    bn.train();

    Variable input = createInput({4, 4, 2, 4, 4}, true);
    auto output = bn.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    auto grad_seed_cpu = ones(shape_vec, DType::Float32, Device::cpu());
    auto grad_seed = grad_seed_cpu.to(device()).to(dtype());
    output.backward(grad_seed);

    ASSERT_TRUE(input.has_grad());
    auto input_grad_f32 = input.grad().value().to(Device::cpu()).to(DType::Float32);
    auto grad_data = input_grad_f32.data<float>();
    for (int64_t i = 0; i < input_grad_f32.numel(); ++i) {
        EXPECT_FALSE(std::isnan(grad_data[i]));
        EXPECT_FALSE(std::isinf(grad_data[i]));
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(BatchNorm3dMultiDTypeTest);
