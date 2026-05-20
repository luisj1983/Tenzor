/**
 * @file test_fp8_quantize_grad.cpp
 * @brief Regression test for fp8_quantize_dequantize STE backward.
 *
 * Audit item E.15: the public fp8 quantize/dequantize functions previously
 * had no autograd Function wrapper, so any training pipeline that
 * routed activations through FP8 silently severed the gradient graph.
 * fp8_quantize_dequantize now ships an STE-backward Variable wrapper.
 */

#include <gtest/gtest.h>

#include <cmath>

#include "tenzor/autograd/variable.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/fp8_scaling.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

using namespace tenzor;

namespace {

class Fp8QuantDequantGradTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

TEST_F(Fp8QuantDequantGradTest, GradFlowsThroughInRangeValues) {
    // Build an input well within FP8_E4M3's representable range
    // (E4M3 max ≈ 448).  Every position is in-range so STE backward
    // passes the upstream gradient through unchanged.
    Tensor data({4}, DType::Float32, Device::cpu());
    auto* p = data.data<float>();
    p[0] = 1.0f; p[1] = -2.0f; p[2] = 3.0f; p[3] = -4.0f;

    Variable x(data, /*requires_grad=*/true);
    auto y = fp8_quantize_dequantize(x, DType::FP8_E4M3);
    auto loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad());
    auto g = *x.grad();
    // Expected: ones (STE pass-through within range).
    const auto* gp = g.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(gp[i], 1.0f) << "i=" << i;
    }
}

TEST_F(Fp8QuantDequantGradTest, GradZeroedOutsideRangeFixedScale) {
    // With an explicit FIXED scale, some values fall outside the
    // representable FP8 range (|x| > scale * fp8_max) and STE clips
    // their gradients to zero.  Pick scale = 1.0 so |x| > fp8_max ≈ 448
    // triggers clipping.
    Tensor data({4}, DType::Float32, Device::cpu());
    auto* p = data.data<float>();
    p[0] = 1.0f;       // in range
    p[1] = 1000.0f;    // out of range (|x| > 448)
    p[2] = -2.0f;      // in range
    p[3] = -1000.0f;   // out of range

    Variable x(data, /*requires_grad=*/true);
    // Explicit scale=1.0 means in-range threshold = fp8_max ≈ 448.
    auto y = fp8_quantize_dequantize(x, DType::FP8_E4M3, /*scale=*/1.0f);
    auto loss = sum(y);
    loss.backward();

    ASSERT_TRUE(x.has_grad());
    auto g = *x.grad();
    const auto* gp = g.data<float>();
    EXPECT_FLOAT_EQ(gp[0], 1.0f) << "in-range entry must pass STE grad";
    EXPECT_FLOAT_EQ(gp[1], 0.0f) << "out-of-range entry must be clipped to 0";
    EXPECT_FLOAT_EQ(gp[2], 1.0f);
    EXPECT_FLOAT_EQ(gp[3], 0.0f);
}

TEST_F(Fp8QuantDequantGradTest, ForwardPreservesShapeAndSign) {
    // FP8_E4M3 round-trip quality is a property of quantize_to_fp8
    // itself; this test only verifies that fp8_quantize_dequantize
    // preserves the output shape and that the sign of each in-range
    // value is preserved (E4M3 has a true zero so we test non-zero
    // entries).
    Tensor data({3}, DType::Float32, Device::cpu());
    auto* p = data.data<float>();
    p[0] = 5.0f; p[1] = 12.0f; p[2] = -7.0f;

    Variable x(data, false);
    auto y = fp8_quantize_dequantize(x, DType::FP8_E4M3);

    ASSERT_EQ(y.tensor().shape().size(), 1u);
    EXPECT_EQ(y.tensor().shape()[0], 3);

    const auto* yp = y.tensor().data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::signbit(yp[i]) == std::signbit(p[i]))
            << "sign not preserved at i=" << i
            << " (input=" << p[i] << ", output=" << yp[i] << ")";
    }
}

}  // namespace
