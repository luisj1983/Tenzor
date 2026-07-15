/**
 * @file test_layernorm_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for nn::LayerNorm layer
 *
 * Standalone layer-level tests (constructor, state_dict, forward, backward,
 * numerical properties). Complements the broader normalization tests in
 * test_normalization_multidtype.cpp.
 */

#include <gtest/gtest.h>
#include <cmath>
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
    // Non-uniform upstream grad: d(sum(LayerNorm(x)))/dx is ~0 (the LN output
    // sums to a constant), so an all-ones seed yields a legitimately-zero input
    // grad — only rounding noise keeps it >0 on F32/F64, and F16 rounds to 0.
    // A varied seed exercises a genuinely non-zero gradient on every dtype.
    auto grad = tenzor::randn({2, 16}, DType::Float32, device_).to(dtype_);
    EXPECT_NO_THROW({ y.backward(grad); });
    EXPECT_GRAD_FLOWS(x);  // W.26
    expectShape(*x.grad(), {2, 16});
}

TEST_P(LayerNormMultiDTypeTest, TinyVarianceGradientFinite) {
    // M4: cuDNN's LayerNorm forward (CUDA F16/BF16, cuDNN build) used to
    // narrow the saved mean/inv_std to the input's half-precision dtype
    // instead of keeping them at Float32. rstd = 1/sqrt(var+eps) can exceed
    // FP16's max (65504) for a near-constant input with a tiny eps,
    // saturating to Inf when narrowed.
    //
    // Both x and the upstream grad_seed are constant (bit-identical across
    // all elements) so the TRUE (mathematically exact) grad_input is
    // IDENTICALLY ZERO regardless of rstd's magnitude: x_hat=(x-mean)*rstd
    // = 0*rstd = 0 for every element (x-mean is exactly 0, independent of
    // rstd), and dy*w - mean(dy*w) = 0 for every element too (dy*w is the
    // same constant everywhere). grad_input = rstd*(dy*w-mean(dy*w)-x_hat*
    // ...) = rstd*0 = 0 — exactly representable in any float dtype, no
    // matter how large rstd is, UNLESS an implementation detail (like
    // narrowing rstd to F16 before this multiply) turns that 0*rstd into
    // 0*Inf = NaN. This isolates the narrowing bug from the (expected,
    // not-a-bug) case where a genuinely large gradient just doesn't fit in
    // FP16 — here the true answer is always 0, so any NaN/Inf is an
    // artifact of internal precision, not a real overflow.
    nn::LayerNorm ln({16}, /*eps=*/1e-12);
    convert_model(ln);
    auto x_t = tenzor::full({2, 16}, 1.0, dtype_, device_);
    Variable x(x_t, true);

    auto y = ln.forward(x);
    auto y_cpu = y.tensor().to(Device::cpu()).to(DType::Float32);
    const auto* yp = y_cpu.data<float>();
    for (int64_t i = 0; i < y_cpu.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(yp[i])) << "output[" << i << "] = " << yp[i];
    }

    auto grad_seed = tenzor::full({2, 16}, 1.0, dtype_, device_);
    y.backward(grad_seed);
    ASSERT_TRUE(x.grad().has_value());
    auto grad_cpu = x.grad()->to(Device::cpu()).to(DType::Float32);
    const auto* gp = grad_cpu.data<float>();
    for (int64_t i = 0; i < grad_cpu.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(gp[i])) << "grad[" << i << "] = " << gp[i];
    }
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
