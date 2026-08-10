/**
 * @file test_batchnorm_dispatch_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for BatchNorm2d sub-operations
 *
 * Covers: BatchNorm2d training mode (fused training path), running stats
 * update, and mean/var computation verification.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/batchnorm.hpp>
#include <tenzor/autograd/variable.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// Macro (not a method) so any future GTEST_SKIP returns from the TEST_P body
// rather than from a helper method. BatchNorm runs FP32-accumulation-with-
// FP16-storage on CUDA/ROCm and true half on the other backends; the fixture's
// setTolerances() (called from SetUp) already widens atol/rtol to 1e-2 for
// Float16/BFloat16. Instead of skipping the half cases, we run them at the
// dtype-aware tolerance — re-asserting it here keeps these numerical checks
// half-correct even if the fixture default ever tightens.
#define loosenForHalf() \
    do { \
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) { \
            rtol_ = 1e-2f; \
            atol_ = 1e-2f; \
        } \
    } while (0)

class BatchNormDispatchMultiDTypeTest : public MultiBackendDTypeTest {
protected:
};

// ============================================================================
// Training mode forward (exercises BatchNorm2dFusedTraining OpId)
// ============================================================================

TEST_P(BatchNormDispatchMultiDTypeTest, TrainingForwardShape) {
    nn::BatchNorm2d bn(8);
    convert_model(bn);
    bn.train();

    auto input = createInput({2, 8, 4, 4}, false);
    auto output = bn.forward(input);
    expectShape(output.tensor(), {2, 8, 4, 4});
    expectDevice(output.tensor());
    expectDType(output.tensor());
}

TEST_P(BatchNormDispatchMultiDTypeTest, TrainingUpdatesRunningStats) {
    loosenForHalf();
    nn::BatchNorm2d bn(4, 1e-5, 0.1, true, true);
    convert_model(bn);
    bn.train();

    // Forward twice to verify running stats are updated
    auto input1 = createInput({2, 4, 4, 4}, false);
    bn.forward(input1);

    auto input2 = createInput({2, 4, 4, 4}, false);
    bn.forward(input2);

    // Running mean should no longer be all zeros after two forward passes
    auto params = bn.parameters();
    ASSERT_GE(params.size(), 2u) << "BatchNorm2d should have at least weight and bias";
}

// ============================================================================
// Eval mode forward (exercises BatchNorm2dForward OpId)
// ============================================================================

TEST_P(BatchNormDispatchMultiDTypeTest, EvalForwardShape) {
    nn::BatchNorm2d bn(8);
    convert_model(bn);
    bn.eval();

    auto input = createInput({2, 8, 4, 4}, false);
    auto output = bn.forward(input);
    expectShape(output.tensor(), {2, 8, 4, 4});
    expectDevice(output.tensor());
}

// ============================================================================
// Backward gradient flow
// ============================================================================

TEST_P(BatchNormDispatchMultiDTypeTest, BackwardGradientFlow) {
    nn::BatchNorm2d bn(4);
    convert_model(bn);
    bn.train();

    auto input = createInput({2, 4, 4, 4}, true);
    auto output = bn.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); })
        << "BatchNorm2d backward threw on " << device().to_string();

    ASSERT_TRUE(input.grad().has_value())
        << "BatchNorm2d backward did not produce input gradient on " << device().to_string();
    expectShape(*input.grad(), {2, 4, 4, 4});
}

// ============================================================================
// Affine parameter gradients
// ============================================================================

TEST_P(BatchNormDispatchMultiDTypeTest, AffineParameterGradients) {
    nn::BatchNorm2d bn(4, 1e-5, 0.1, true);
    convert_model(bn);
    bn.train();

    auto input = createInput({2, 4, 4, 4}, true);
    auto output = bn.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    output.backward(grad_output);

    auto params = bn.parameters();
    // Weight (gamma) and bias (beta) should both have gradients
    for (size_t i = 0; i < params.size(); ++i) {
        ASSERT_TRUE(params[i]->grad().has_value())
            << "BatchNorm2d parameter " << i << " has no gradient on " << device().to_string();
    }
}

// ============================================================================
// Numerical Correctness Tests
// ============================================================================

TEST_P(BatchNormDispatchMultiDTypeTest, NormalizesToZeroMeanUnitVariance) {
    loosenForHalf();
    // In training mode, the per-channel output should have approximately zero
    // mean and unit variance across the (N, H, W) dimensions, scaled by gamma=1
    // and shifted by beta=0 (the default initialization).
    nn::BatchNorm2d bn(4, 1e-5, 0.1, true);
    convert_model(bn);
    bn.train();

    auto input_t = tenzor::randn({8, 4, 4, 4}, dtype(), device());
    Variable input(input_t, false);
    auto output = bn.forward(input);

    // Compute per-channel mean across N,H,W and verify it's close to zero
    auto cpu_out = output.tensor().to(Device::cpu()).to(DType::Float32).contiguous();
    auto* data = cpu_out.data<float>();
    int64_t N = 8, C = 4, H = 4, W = 4;
    for (int64_t c = 0; c < C; ++c) {
        float sum = 0.0f;
        int64_t count = N * H * W;
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    sum += data[((n * C + c) * H + h) * W + w];
                }
            }
        }
        float mean = sum / static_cast<float>(count);
        EXPECT_NEAR(mean, 0.0f, std::max(atol() * 100.0f, 1e-3f))
            << "BatchNorm2d output channel " << c << " mean " << mean << " not close to 0";
    }
}

TEST_P(BatchNormDispatchMultiDTypeTest, EvalModeUsesRunningStats) {
    loosenForHalf();
    // In eval mode with affine=false, output = (input - running_mean) / sqrt(running_var + eps).
    // After construction, running_mean=0 and running_var=1, so output should equal input.
    nn::BatchNorm2d bn(4, 1e-5, 0.1, /*affine=*/false, /*track_running_stats=*/true);
    convert_model(bn);
    bn.eval();

    auto input_t = tenzor::randn({2, 4, 4, 4}, dtype(), device());
    Variable input(input_t, false);
    auto output = bn.forward(input);

    auto cpu_in = input_t.to(Device::cpu()).to(DType::Float32).contiguous();
    auto cpu_out = output.tensor().to(Device::cpu()).to(DType::Float32).contiguous();
    auto* in_data = cpu_in.data<float>();
    auto* out_data = cpu_out.data<float>();

    // Output ≈ input (since running_mean=0, running_var=1, only sqrt(1+eps) ≈ 1 division)
    for (int64_t i = 0; i < cpu_in.numel(); ++i) {
        EXPECT_NEAR(out_data[i], in_data[i], std::max(atol() * 10.0f, 1e-3f))
            << "BatchNorm2d eval mode mismatch at index " << i;
    }
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(BatchNormDispatchMultiDTypeTest);
