/**
 * @file test_spectral_norm_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for spectral normalization
 *
 * Covers: SpectralNorm::apply, remove, sigma
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/utils/spectral_norm.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Fixture
// ============================================================================

class SpectralNormMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Apply Tests
// ============================================================================

TEST_P(SpectralNormMultiDTypeTest, ApplySucceeds) {
    auto linear = std::make_shared<nn::Linear>(16, 8);
    convert_model(linear);

    auto sn = nn::utils::SpectralNorm::apply(linear, "weight", 1);
    ASSERT_NE(sn, nullptr);
}

TEST_P(SpectralNormMultiDTypeTest, SigmaPositive) {
    auto linear = std::make_shared<nn::Linear>(16, 8);
    convert_model(linear);

    auto sn = nn::utils::SpectralNorm::apply(linear, "weight", 1);

    // Run a forward pass to trigger power iteration
    Variable input = createInput({2, 16}, false);
    auto output = linear->forward(input);

    auto sigma = sn->sigma();
    auto sigma_f32 = sigma.to(Device::cpu()).to(DType::Float32);
    EXPECT_GT(sigma_f32.data<float>()[0], 0.0f);
}

TEST_P(SpectralNormMultiDTypeTest, RemoveRestoresWeight) {
    auto linear = std::make_shared<nn::Linear>(16, 8);
    convert_model(linear);

    auto sn = nn::utils::SpectralNorm::apply(linear, "weight", 1);

    // Forward pass with SN
    Variable input1 = createInput({2, 16}, false);
    linear->forward(input1);

    // Remove SN
    EXPECT_NO_THROW(sn->remove());

    // Should still work after removal
    Variable input2 = createInput({2, 16}, false);
    auto output = linear->forward(input2);
    expectShape(output.tensor(), {2, 8});
}

TEST_P(SpectralNormMultiDTypeTest, MultipleForwardPasses) {
    auto linear = std::make_shared<nn::Linear>(16, 8);
    convert_model(linear);

    auto sn = nn::utils::SpectralNorm::apply(linear, "weight", 3);

    // Multiple forward passes should refine the sigma estimate
    for (int i = 0; i < 5; ++i) {
        Variable input = createInput({2, 16}, false);
        linear->forward(input);
    }

    auto sigma = sn->sigma();
    auto sigma_f32 = sigma.to(Device::cpu()).to(DType::Float32);
    EXPECT_GT(sigma_f32.data<float>()[0], 0.0f);
}

// ============================================================================
// PP.2 regression: training actually mutates the layer's weight slot
// ============================================================================
//
// Pre-PP.2 SpectralNorm registered u/v/sigma as bare Variable/Tensor
// members and the pre-hook overwrote `*param_` with a raw-Tensor result.
// Optimiser updates landed on the post-normalised weight Variable, then the
// next forward wiped them. Training was a no-op.
//
// This test verifies that weight_orig (the registered trainable leaf) is
// updated by SGD and that the layer's `weight` slot reflects the update on
// the next forward.
TEST_P(SpectralNormMultiDTypeTest, OptimizerStepActuallyTrains) {
    auto linear = std::make_shared<nn::Linear>(16, 8);
    convert_model(linear);

    auto sn = nn::utils::SpectralNorm::apply(linear, "weight", 1);
    ASSERT_NE(sn, nullptr);
    ASSERT_NE(sn->weight_orig(), nullptr);

    auto params = linear->parameters();
    ASSERT_GE(params.size(), 2u) << "expected weight + bias + weight_orig (and weight slot)";

    Tensor wo_before = sn->weight_orig()->tensor()
                          .to(Device::cpu()).to(DType::Float32).clone();
    Tensor w_slot_before =
        linear->get_parameter("weight")->tensor()
            .to(Device::cpu()).to(DType::Float32).clone();

    Variable input = createInput({4, 16}, /*requires_grad=*/false);
    auto output = linear->forward(input);
    Variable loss = sum(output * output);
    loss.backward();

    ASSERT_TRUE(sn->weight_orig()->has_grad())
        << "SpectralNorm's weight_orig received no gradient — backward did not flow through W/sigma";

    optim::SGD sgd(params, /*lr=*/0.1);
    sgd.step();

    Tensor wo_after = sn->weight_orig()->tensor()
                         .to(Device::cpu()).to(DType::Float32).clone();
    auto* wob = wo_before.data<float>();
    auto* woa = wo_after.data<float>();
    bool wo_changed = false;
    for (int64_t i = 0; i < wo_before.numel(); ++i) {
        if (wob[i] != woa[i]) { wo_changed = true; break; }
    }
    EXPECT_TRUE(wo_changed)
        << "SpectralNorm::weight_orig unchanged after SGD step — optimiser did "
           "not see weight_orig as a parameter";

    Variable input2 = createInput({4, 16}, /*requires_grad=*/false);
    linear->forward(input2);
    Tensor w_slot_after =
        linear->get_parameter("weight")->tensor()
            .to(Device::cpu()).to(DType::Float32).clone();

    auto* wb = w_slot_before.data<float>();
    auto* wa = w_slot_after.data<float>();
    bool w_changed = false;
    for (int64_t i = 0; i < w_slot_before.numel(); ++i) {
        if (wb[i] != wa[i]) { w_changed = true; break; }
    }
    EXPECT_TRUE(w_changed)
        << "Layer's weight slot did not change after step+forward — the pre-hook "
           "either lost the link to weight_orig or the optimiser updated the wrong leaves";
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SpectralNormMultiDTypeTest);
