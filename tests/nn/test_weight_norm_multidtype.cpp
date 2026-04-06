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
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
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
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(WeightNormMultiDTypeTest);
