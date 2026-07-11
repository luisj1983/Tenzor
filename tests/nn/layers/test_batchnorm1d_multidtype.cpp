/**
 * @file test_batchnorm1d_multidtype.cpp
 * @brief Multi-dtype multi-backend tests for BatchNorm1d layer.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include "../../grad_flow_helpers.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class BatchNorm1dMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    float variance_tolerance() const {
        if (dtype() == DType::Float16) return 0.2f;
        // BFloat16 has only 8 mantissa bits (~2-3 decimal digits); the CPU
        // kernel already accumulates mean/variance in Float32 with Kahan
        // summation (src/backends/cpu/kernels/batchnorm.cpp) and narrows
        // only the final stored result, so the residual ~1e-3 error here is
        // expected BFloat16 rounding noise on the narrowed output, not
        // accumulated computational error. Empirically observed ~1.3e-3.
        if (dtype() == DType::BFloat16) return 5e-3f;
        return 1e-3f;
    }
};

TEST_P(BatchNorm1dMultiDTypeTest, ForwardShapePreservation2D) {
    BatchNorm1d bn(64);
    convert_model(bn);

    Variable input = createInput({32, 64}, true);
    auto output = bn.forward(input);

    expectShape(output.tensor(), {32, 64});
    expectDType(output.tensor());
}

TEST_P(BatchNorm1dMultiDTypeTest, ForwardShapePreservation3D) {
    BatchNorm1d bn(16);
    convert_model(bn);

    Variable input = createInput({8, 16, 32}, true);
    auto output = bn.forward(input);

    expectShape(output.tensor(), {8, 16, 32});
    expectDType(output.tensor());
}

TEST_P(BatchNorm1dMultiDTypeTest, ParameterInitialization) {
    BatchNorm1d bn_affine(32, 1e-5, 0.1, true);
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

TEST_P(BatchNorm1dMultiDTypeTest, TrainingModeNormalization3D) {
    BatchNorm1d bn(3, 1e-5, 0.1, false);
    convert_model(bn);
    bn.train();

    Variable input = createInput({16, 3, 8}, false);
    auto output = bn.forward(input);

    auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto output_data = output_f32.data<float>();

    const int64_t N = 16, C = 3, L = 8;
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
        EXPECT_NEAR(mean, 0.0, rtol() + atol());
        EXPECT_NEAR(variance, 1.0, variance_tolerance());
    }
}

TEST_P(BatchNorm1dMultiDTypeTest, BackwardPassGradientFlow) {
    BatchNorm1d bn(4, 1e-5, 0.1, true);
    convert_model(bn);
    bn.train();

    Variable input = createInput({8, 4, 16}, true);
    auto output = bn.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    // Non-uniform grad seed: BatchNorm input/weight grads vanish under a uniform
    // (all-ones) upstream gradient, so use randn to actually exercise grad flow.
    auto grad_seed = createRandn(shape_vec);
    output.backward(grad_seed);

    EXPECT_GRAD_FLOWS(input);
    auto input_grad_f32 = input.grad().value().to(Device::cpu()).to(DType::Float32);
    auto grad_data = input_grad_f32.data<float>();
    for (int64_t i = 0; i < input_grad_f32.numel(); ++i) {
        EXPECT_FALSE(std::isnan(grad_data[i]));
        EXPECT_FALSE(std::isinf(grad_data[i]));
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(BatchNorm1dMultiDTypeTest);
