/**
 * @file test_instance_norm_multidtype.cpp
 * @brief Multi-dtype tests for InstanceNorm1d, InstanceNorm2d, and InstanceNorm3d layers
 *
 * Tests instance normalization with Float32, Float64, and Float16 dtypes across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - Correct shape preservation for 1d, 2d, and 3d variants
 * - Proper per-instance per-channel normalization over spatial dims
 * - Parameter initialization and affine/non-affine modes
 * - Gradient flow through instance normalization
 * - Numerical stability with constant inputs (epsilon)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

// ============================================================================
// InstanceNorm Multi-Backend Multi-DType Test Fixture
// ============================================================================

class InstanceNormMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    float variance_tolerance() const {
        if (dtype() == DType::Float16) {
            return 0.2f;
        } else if (dtype() == DType::Float64) {
            return 1e-4f;
        }
        return 1e-4f;
    }
};

// ============================================================================
// InstanceNorm2d Tests
// ============================================================================

TEST_P(InstanceNormMultiDTypeTest, InstanceNorm2d_ForwardShapePreservation) {
    InstanceNorm2d in(64);
    convert_model(in);

    Variable input = createInput({8, 64, 16, 16}, true);
    auto output = in.forward(input);

    expectShape(output.tensor(), {8, 64, 16, 16});
    expectDType(output.tensor());
}

TEST_P(InstanceNormMultiDTypeTest, InstanceNorm2d_NormalizesPerInstance) {
    InstanceNorm2d in(3, 1e-5, false);
    convert_model(in);
    in.train();

    Variable input = createInput({4, 3, 8, 8}, false);
    auto output = in.forward(input);

    auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto output_data = output_f32.data<float>();

    int64_t N = 4, C = 3, H = 8, W = 8;
    int64_t spatial_size = H * W;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            double sum = 0.0;
            double sum_sq = 0.0;

            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    float val = output_data[idx];
                    sum += val;
                    sum_sq += val * val;
                }
            }

            double mean = sum / spatial_size;
            double variance = sum_sq / spatial_size - mean * mean;

            EXPECT_NEAR(mean, 0.0, atol() * 10)
                << "Instance " << n << ", Channel " << c;
            EXPECT_NEAR(variance, 1.0, variance_tolerance())
                << "Instance " << n << ", Channel " << c;
        }
    }
}

TEST_P(InstanceNormMultiDTypeTest, InstanceNorm2d_AffineParameters) {
    InstanceNorm2d in(4, 1e-5, true);

    auto params = in.parameters();
    EXPECT_EQ(params.size(), 2);

    EXPECT_EQ(params[0]->shape()[0], 4);
    EXPECT_EQ(params[1]->shape()[0], 4);

    auto weight_f32 = params[0]->tensor().to(Device::cpu()).to(DType::Float32);
    auto bias_f32 = params[1]->tensor().to(Device::cpu()).to(DType::Float32);
    auto weight_data = weight_f32.data<float>();
    auto bias_data = bias_f32.data<float>();

    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(weight_data[i], 1.0f, atol());
        EXPECT_NEAR(bias_data[i], 0.0f, atol());
    }
}

TEST_P(InstanceNormMultiDTypeTest, InstanceNorm2d_NoAffine) {
    InstanceNorm2d in(4, 1e-5, false);

    auto params = in.parameters();
    EXPECT_EQ(params.size(), 0);
}

TEST_P(InstanceNormMultiDTypeTest, InstanceNorm2d_BackwardGradientFlow) {
    InstanceNorm2d in(4, 1e-5, true);
    convert_model(in);
    in.train();

    Variable input = createInput({8, 4, 8, 8}, true);
    auto output = in.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(shape_vec, dtype(), device());
    output.backward(grad_output);

    EXPECT_TRUE(input.has_grad());
    EXPECT_EQ(input.grad()->dtype(), dtype());

    auto input_grad_f32 = input.grad()->to(Device::cpu()).to(DType::Float32);
    auto grad_data = input_grad_f32.data<float>();

    for (int64_t i = 0; i < input_grad_f32.numel(); ++i) {
        EXPECT_FALSE(std::isnan(grad_data[i]));
        EXPECT_FALSE(std::isinf(grad_data[i]));
    }
}

TEST_P(InstanceNormMultiDTypeTest, InstanceNorm2d_EpsilonPreventsDivByZero) {
    InstanceNorm2d in(2, 1e-3, false);
    convert_model(in);
    in.train();

    auto input_tensor = createOnes({8, 2, 8, 8});
    Variable input(input_tensor, false);

    EXPECT_NO_THROW({
        auto output = in.forward(input);
        auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
        auto data = output_f32.data<float>();

        for (int64_t i = 0; i < output_f32.numel(); ++i) {
            EXPECT_FALSE(std::isnan(data[i]));
            EXPECT_FALSE(std::isinf(data[i]));
        }
    });
}

// ============================================================================
// InstanceNorm1d Tests
// ============================================================================

TEST_P(InstanceNormMultiDTypeTest, InstanceNorm1d_ForwardShapePreservation) {
    InstanceNorm1d in(32);
    convert_model(in);

    Variable input = createInput({8, 32, 64}, true);
    auto output = in.forward(input);

    expectShape(output.tensor(), {8, 32, 64});
    expectDType(output.tensor());
}

TEST_P(InstanceNormMultiDTypeTest, InstanceNorm1d_BackwardGradientFlow) {
    InstanceNorm1d in(16, 1e-5, true);
    convert_model(in);
    in.train();

    Variable input = createInput({4, 16, 32}, true);
    auto output = in.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(shape_vec, dtype(), device());
    output.backward(grad_output);

    EXPECT_TRUE(input.has_grad());
    EXPECT_EQ(input.grad()->dtype(), dtype());

    auto input_grad_f32 = input.grad()->to(Device::cpu()).to(DType::Float32);
    auto grad_data = input_grad_f32.data<float>();

    for (int64_t i = 0; i < input_grad_f32.numel(); ++i) {
        EXPECT_FALSE(std::isnan(grad_data[i]));
        EXPECT_FALSE(std::isinf(grad_data[i]));
    }
}

// ============================================================================
// InstanceNorm3d Tests
// ============================================================================

TEST_P(InstanceNormMultiDTypeTest, InstanceNorm3d_ForwardShapePreservation) {
    InstanceNorm3d in(16);
    convert_model(in);

    Variable input = createInput({4, 16, 8, 8, 8}, true);
    auto output = in.forward(input);

    expectShape(output.tensor(), {4, 16, 8, 8, 8});
    expectDType(output.tensor());
}

TEST_P(InstanceNormMultiDTypeTest, InstanceNorm3d_BackwardGradientFlow) {
    InstanceNorm3d in(8, 1e-5, true);
    convert_model(in);
    in.train();

    Variable input = createInput({2, 8, 4, 4, 4}, true);
    auto output = in.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(shape_vec, dtype(), device());
    output.backward(grad_output);

    EXPECT_TRUE(input.has_grad());
    EXPECT_EQ(input.grad()->dtype(), dtype());

    auto input_grad_f32 = input.grad()->to(Device::cpu()).to(DType::Float32);
    auto grad_data = input_grad_f32.data<float>();

    for (int64_t i = 0; i < input_grad_f32.numel(); ++i) {
        EXPECT_FALSE(std::isnan(grad_data[i]));
        EXPECT_FALSE(std::isinf(grad_data[i]));
    }
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(InstanceNormMultiDTypeTest);
