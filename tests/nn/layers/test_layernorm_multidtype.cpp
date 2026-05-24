/**
 * @file test_layernorm_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for nn::LayerNorm layer
 *
 * Standalone layer-level tests (constructor, state_dict, forward, backward,
 * numerical properties). Complements the broader normalization tests in
 * test_normalization_multidtype.cpp.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/normalization.hpp>
#include <tenzor/autograd/variable.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include "../../grad_flow_helpers.hpp"  // W.26: EXPECT_GRAD_FLOWS

using namespace tenzor;
using namespace tenzor::testing;

class LayerNormMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Constructor / parameters
// ============================================================================

TEST_P(LayerNormMultiDTypeTest, ConstructorRegistersAffineParameters) {
    nn::LayerNorm ln({16});
    convert_model(ln);
    auto params = ln.parameters();
    // elementwise_affine=true by default → weight + bias.
    EXPECT_EQ(params.size(), 2u);
    EXPECT_EQ(countParameters(params), static_cast<size_t>(32));
}

TEST_P(LayerNormMultiDTypeTest, NoAffineSkipsParameters) {
    nn::LayerNorm ln({16}, 1e-5, /*elementwise_affine=*/false);
    convert_model(ln);
    auto params = ln.parameters();
    EXPECT_EQ(params.size(), 0u);
}

// ============================================================================
// Forward pass & statistics
// ============================================================================

TEST_P(LayerNormMultiDTypeTest, ForwardShape) {
    nn::LayerNorm ln({16});
    convert_model(ln);
    Variable x = createInput({2, 8, 16}, false);
    auto y = ln.forward(x);
    expectShape(y.tensor(), {2, 8, 16});
    expectDevice(y.tensor());
    expectDType(y.tensor());
}

TEST_P(LayerNormMultiDTypeTest, NormalizesToZeroMeanUnitVar) {
    // Affine parameters are defaulted to gamma=1, beta=0, so output per
    // normalized slab should be ~mean=0, std≈1.
    nn::LayerNorm ln({16});
    convert_model(ln);
    Variable x = createInput({2, 16}, false);
    auto y = ln.forward(x);

    auto y_cpu = y.tensor().to(Device::cpu()).to(DType::Float32);
    const auto* p = y_cpu.data<float>();
    // Check each row independently.
    float row_tol = (dtype_ == DType::Float16 || dtype_ == DType::BFloat16)
                        ? 2e-1f : 2e-2f;
    for (int64_t r = 0; r < 2; ++r) {
        float sum = 0.0f, sumsq = 0.0f;
        for (int64_t c = 0; c < 16; ++c) {
            float v = p[r * 16 + c];
            sum += v;
            sumsq += v * v;
        }
        float mean = sum / 16.0f;
        float var = sumsq / 16.0f - mean * mean;
        EXPECT_NEAR(mean, 0.0f, row_tol) << "row " << r << " mean";
        EXPECT_NEAR(var, 1.0f, row_tol) << "row " << r << " var";
    }
}

// ============================================================================
// Backward / gradient flow
// ============================================================================

TEST_P(LayerNormMultiDTypeTest, GradientFlow) {
    nn::LayerNorm ln({16});
    convert_model(ln);
    Variable x = createInput({2, 16}, true);
    auto y = ln.forward(x);
    auto grad = tenzor::ones({2, 16}, dtype_, device_);
    EXPECT_NO_THROW({ y.backward(grad); });
    EXPECT_GRAD_FLOWS(x);  // W.26
    ASSERT_TRUE(x.grad().has_value())
        << "LayerNorm backward did not populate input grad on "
        << device().to_string();
    expectShape(*x.grad(), {2, 16});
}

// ============================================================================
// state_dict round-trip
// ============================================================================

TEST_P(LayerNormMultiDTypeTest, StateDictRoundtrip) {
    nn::LayerNorm ln({16});
    convert_model(ln);
    auto state = ln.state_dict();
    EXPECT_EQ(state.size(), 2u);

    nn::LayerNorm ln2({16});
    convert_model(ln2);
    ln2.load_state_dict(state);
    EXPECT_EQ(ln2.state_dict().size(), 2u);
}

// ============================================================================
// Different normalized-shape arities
// ============================================================================

TEST_P(LayerNormMultiDTypeTest, MultiDimNormalizedShape) {
    nn::LayerNorm ln({4, 8});
    convert_model(ln);
    Variable x = createInput({2, 3, 4, 8}, false);
    auto y = ln.forward(x);
    expectShape(y.tensor(), {2, 3, 4, 8});
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(LayerNormMultiDTypeTest);
