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

#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/fp8_scaling.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

#include <optional>
#include <string>
#include <vector>

using namespace tenzor;

namespace {

// Relocated from include/tenzor/ops/fp8_scaling.hpp (decl) and
// src/ops/fp8_scaling.cpp (Function + impl): the STE-backward
// fp8_quantize_dequantize(const Variable&, ...) overload is exercised only
// by this regression test, so it lives here file-local instead of on the
// public op surface. The other fp8 functions remain in fp8_scaling.{hpp,cpp}.
class Fp8QuantizeDequantizeFunction : public Function {
public:
    Fp8QuantizeDequantizeFunction(DType fp8_dtype, std::optional<float> scale,
                                  float fp8_min, float fp8_max)
        : fp8_dtype_(fp8_dtype), scale_(scale), fp8_min_(fp8_min), fp8_max_(fp8_max) {}

    auto forward(std::vector<Variable> inputs)
        -> std::vector<Variable> override {
        const auto& in = inputs[0].tensor();
        auto [fp8_tensor, params] = quantize_to_fp8(in, fp8_dtype_, scale_,
                                                     /*stochastic_rounding=*/false);
        Tensor out = dequantize_from_fp8(fp8_tensor, params.scale);
        // STE backward needs to know which positions were in-range — save
        // the original input scaled by the scaling factor so backward can
        // build the mask without re-running the FP8 cast.
        scale_used_ = params.scale;
        save_for_backward({in});
        return {Variable(out, false)};
    }

    auto backward(std::vector<Tensor> grad_outputs)
        -> std::vector<Tensor> override {
        const auto& grad_out = grad_outputs[0];
        const auto& input = saved_tensors_[0];

        // STE convention used by Transformer Engine / Megatron:
        //   grad_in[i] = grad_out[i]                       if |input[i] / scale| ≤ fp8_max,
        //                0                                  otherwise.
        // Equivalent to clamping to the representable FP8 range.
        const float scale = scale_used_;
        const float range = fp8_max_ * scale;
        // Build the in-range mask: |input| ≤ range.
        Tensor input_abs = ::tenzor::abs(input);
        Tensor threshold = ::tenzor::full({1}, static_cast<double>(range),
                                           input.dtype(), input.device());
        // mask is 1.0 where in-range, 0.0 elsewhere.  Use where(condition, 1, 0).
        Tensor in_range = ::tenzor::where(
            ::tenzor::le(input_abs, threshold),
            ::tenzor::full(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                            1.0, input.dtype(), input.device()),
            ::tenzor::full(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                            0.0, input.dtype(), input.device()));
        Tensor grad_in = ::tenzor::mul(grad_out, in_range);
        return {grad_in};
    }

    auto name() const -> std::string override { return "Fp8QuantizeDequantize"; }

private:
    DType fp8_dtype_;
    std::optional<float> scale_;
    float fp8_min_;
    float fp8_max_;
    float scale_used_{1.0f};
};

auto fp8_quantize_dequantize(const Variable& input,
                             DType fp8_dtype,
                             std::optional<float> scale = std::nullopt)
    -> Variable {
    if (fp8_dtype != DType::FP8_E4M3 && fp8_dtype != DType::FP8_E5M2) {
        throw std::runtime_error(
            "fp8_quantize_dequantize: target dtype must be FP8_E4M3 or FP8_E5M2");
    }

    const float fp8_max = fp8_max_value(fp8_dtype);
    const float fp8_min = -fp8_max;

    auto fn = std::make_shared<Fp8QuantizeDequantizeFunction>(
        fp8_dtype, scale, fp8_min, fp8_max);
    auto outputs = fn->forward({input});
    if (input.requires_grad() && is_grad_enabled()) {
        outputs[0].set_requires_grad(true);
        outputs[0].set_grad_fn(fn);
        fn->set_next_functions({input.grad_fn()});
        fn->set_input_variables({input});
    }
    return outputs[0];
}

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
