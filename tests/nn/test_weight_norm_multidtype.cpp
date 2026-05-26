/**
 * @file test_weight_norm_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for weight normalization
 *
 * Covers: WeightNorm::apply, remove, reparameterization
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/utils/weight_norm.hpp>
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

class WeightNormMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Apply Tests
// ============================================================================

TEST_P(WeightNormMultiDTypeTest, ApplySucceeds) {
    auto linear = std::make_shared<nn::Linear>(16, 8);
    convert_model(linear);

    auto wn = nn::utils::WeightNorm::apply(linear, "weight", 0);
    ASSERT_NE(wn, nullptr);
}

TEST_P(WeightNormMultiDTypeTest, ForwardPassEquivalent) {
    // Output with weight norm should be valid (not NaN, correct shape)
    auto linear = std::make_shared<nn::Linear>(16, 8);
    convert_model(linear);

    auto wn = nn::utils::WeightNorm::apply(linear, "weight", 0);

    Variable input = createInput({2, 16}, false);
    auto output = linear->forward(input);
    expectShape(output.tensor(), {2, 8});
    expectDevice(output.tensor());
    expectDType(output.tensor());
}

TEST_P(WeightNormMultiDTypeTest, RemoveRestoresWeight) {
    auto linear = std::make_shared<nn::Linear>(16, 8);
    convert_model(linear);

    auto wn = nn::utils::WeightNorm::apply(linear, "weight", 0);

    // Forward with weight norm
    Variable input1 = createInput({2, 16}, false);
    linear->forward(input1);

    // Remove weight norm
    EXPECT_NO_THROW(wn->remove());

    // Should still work after removal
    Variable input2 = createInput({2, 16}, false);
    auto output = linear->forward(input2);
    expectShape(output.tensor(), {2, 8});
}

TEST_P(WeightNormMultiDTypeTest, MultipleForwardPasses) {
    auto linear = std::make_shared<nn::Linear>(32, 16);
    convert_model(linear);

    auto wn = nn::utils::WeightNorm::apply(linear, "weight", 0);

    // Multiple forward passes should work consistently
    for (int i = 0; i < 5; ++i) {
        Variable input = createInput({4, 32}, false);
        auto output = linear->forward(input);
        expectShape(output.tensor(), {4, 16});
    }
}

// ============================================================================
// PP.1 regression: training actually mutates the layer's weight slot
// ============================================================================
//
// Pre-PP.1, WeightNorm registered g/v as bare Variable members (not actual
// module parameters) and the pre-hook overwrote the layer's `weight` slot
// with a raw-Tensor result. Optimiser updates landed on the post-normalised
// weight Variable, then the next forward wiped them. Training was a no-op.
//
// This test runs forward -> loss -> backward -> SGD step -> re-forward, and
// asserts:
//   1) the registered g (the trainable magnitude) actually changed after
//      the step (proves the optimiser sees the WeightNorm parameters)
//   2) the layer's `weight` slot tensor changed after the step (proves the
//      pre-hook recomputes from the updated g/v, instead of returning a
//      stale value frozen at apply() time)
TEST_P(WeightNormMultiDTypeTest, OptimizerStepActuallyTrains) {
    auto linear = std::make_shared<nn::Linear>(16, 8);
    convert_model(linear);

    auto wn = nn::utils::WeightNorm::apply(linear, "weight", 0);
    ASSERT_NE(wn, nullptr);
    ASSERT_NE(wn->weight_g(), nullptr);

    // Collect parameters (the registered g and v are now visible via the
    // module's named_parameters()).
    auto params = linear->parameters();
    ASSERT_GE(params.size(), 3u) << "expected weight + bias + g + v";

    // Snapshot g and the layer's weight slot before the step.
    Tensor g_before = wn->weight_g()->tensor().to(Device::cpu()).to(DType::Float32).clone();
    Tensor w_slot_before =
        linear->get_parameter("weight")->tensor().to(Device::cpu()).to(DType::Float32).clone();

    // Forward + scalar loss + backward.
    Variable input = createInput({4, 16}, /*requires_grad=*/false);
    auto output = linear->forward(input);
    Variable loss = sum(output * output);   // L = sum(y^2)
    loss.backward();

    // The trainable g must have received a gradient.
    ASSERT_TRUE(wn->weight_g()->has_grad())
        << "WeightNorm's g received no gradient — backward did not flow through the reparameterisation";

    // Step.
    optim::SGD sgd(params, /*lr=*/0.1);
    sgd.step();

    // g should have changed.
    Tensor g_after = wn->weight_g()->tensor().to(Device::cpu()).to(DType::Float32).clone();
    auto* gb = g_before.data<float>();
    auto* ga = g_after.data<float>();
    bool g_changed = false;
    for (int64_t i = 0; i < g_before.numel(); ++i) {
        if (gb[i] != ga[i]) { g_changed = true; break; }
    }
    EXPECT_TRUE(g_changed)
        << "WeightNorm::g unchanged after SGD step — optimiser did not see g as a parameter";

    // Re-forward — the pre-hook should recompute the weight slot from the
    // updated g/v. Snapshot the slot AFTER the new forward.
    Variable input2 = createInput({4, 16}, /*requires_grad=*/false);
    linear->forward(input2);
    Tensor w_slot_after =
        linear->get_parameter("weight")->tensor().to(Device::cpu()).to(DType::Float32).clone();

    auto* wb = w_slot_before.data<float>();
    auto* wa = w_slot_after.data<float>();
    bool w_changed = false;
    for (int64_t i = 0; i < w_slot_before.numel(); ++i) {
        if (wb[i] != wa[i]) { w_changed = true; break; }
    }
    EXPECT_TRUE(w_changed)
        << "Layer's weight slot did not change after step+forward — the pre-hook "
           "either lost the link to g/v or the optimiser updated the wrong leaves";
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(WeightNormMultiDTypeTest);
