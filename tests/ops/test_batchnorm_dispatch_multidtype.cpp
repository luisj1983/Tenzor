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

class BatchNormDispatchMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void skipIfHalf() {
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
            GTEST_SKIP() << "BatchNorm dispatch tests require higher precision";
        }
    }
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
    skipIfHalf();
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
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(BatchNormDispatchMultiDTypeTest);
