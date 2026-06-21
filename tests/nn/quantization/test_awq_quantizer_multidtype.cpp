/**
 * @file test_awq_quantizer_multidtype.cpp
 * @brief OO.19 — Multi-backend + multi-dtype companion for AWQQuantizer.
 *
 * Sweeps the AWQ quantization pipeline (activation-scale computation +
 * layer quantization) across {Float16, BFloat16, Float32} weights for every
 * available backend. The internal quantization math runs on Float32 inside
 * the quantizer; this test verifies that:
 *   1. compute_act_scales() accepts inputs in each float dtype and produces
 *      a well-shaped, non-negative scale vector.
 *   2. quantize_layer() round-trips dtype-converted weights without
 *      crashing, and the dequantized error stays within a dtype-dependent
 *      bound (1e-2 for Float16, 1e-3 for BFloat16, 1e-5 for Float32).
 *
 * The AWQ implementation upcasts to Float32 internally, so the error bound
 * reflects the input-side rounding error of the dtype conversion alone,
 * not the quantizer's quantization error (which is measured separately by
 * the base test_awq_quantizer.cpp).
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/quantization/awq.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include "awq_dequant_helper.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn::quantization;

namespace {

// Per-dtype tolerance for the dequant round-trip vs the original Float32
// reference. The AWQ algorithm itself quantizes to INT4/INT8 internally
// regardless of the input dtype, so the tolerance encodes the lower bound
// imposed by dtype conversion rather than the quantization-error bound.
float awq_atol_for(DType d) {
    switch (d) {
        case DType::Float16:  return 1e-2f;
        case DType::BFloat16: return 1e-2f;
        case DType::Float32:  return 1e-5f;
        default:              return 1e-3f;
    }
}

} // namespace

class AWQQuantizerMultiDTypeTest : public MultiBackendDTypeTest {};

// Activation-scale computation must accept each float dtype and produce a
// 1-D non-negative tensor of length == feature_dim. We do not enforce
// bit-exact parity across backends — the quantizer's internal reductions
// are Float32-accumulated regardless of the input dtype, so cross-backend
// parity for the scales is delegated to the per-op reduction parity tests.
TEST_P(AWQQuantizerMultiDTypeTest, ComputeActScalesShape) {
    // Skip integer dtypes the fixture may parameterize over — quantization
    // of integer activations is not part of the AWQ surface.
    if (dtype_ != DType::Float32 && dtype_ != DType::Float64 &&
        dtype_ != DType::Float16 && dtype_ != DType::BFloat16) {
        SKIP_WITH_REASON(tenzor::testing::SkipReason::DtypeUnsupportedOnBackend,
            "AWQ activation scales are float-only.");
        return;
    }
    // Audit-5: removed a `try { ... } catch (const std::exception&) {
    // GTEST_SKIP(); }` wrapper that buried compute_act_scales dispatch
    // failures as "unsupported on this backend/dtype". The dtype filter above
    // is the legitimate precondition; let exceptions from the work propagate.
    auto act_cpu = tenzor::randn({8, 64}, DType::Float32, Device::cpu());
    auto act = act_cpu.to(dtype_).to(device_);

    auto scales = AWQQuantizer::compute_act_scales(act);
    ASSERT_EQ(scales.dim(), 1);
    EXPECT_EQ(scales.size(0), 64);

    auto scales_cpu = scales.to(Device::cpu()).to(DType::Float32);
    const float* data = scales_cpu.data<float>();
    for (int64_t i = 0; i < scales_cpu.numel(); ++i) {
        EXPECT_GE(data[i], 0.0f) << "scale[" << i << "] is negative";
    }
}

// Layer quantization end-to-end: build a Float32 reference weight, convert
// to dtype_, quantize, and verify the dequantized weight stays within a
// dtype-dependent tolerance of the Float32 reference. The internal Float32
// math means the dominant error term is the input-side conversion to dtype_.
TEST_P(AWQQuantizerMultiDTypeTest, QuantizeLayerRoundTrip) {
    if (dtype_ != DType::Float32 && dtype_ != DType::Float64 &&
        dtype_ != DType::Float16 && dtype_ != DType::BFloat16) {
        SKIP_WITH_REASON(tenzor::testing::SkipReason::DtypeUnsupportedOnBackend,
            "AWQ quantize_layer is float-only.");
        return;
    }

    AWQConfig config;
    config.bits = 4;
    config.group_size = 64;
    AWQQuantizer q(config);

    // Use modest values to keep the INT4 quantization error well-bounded —
    // the test verifies dtype conversion / dispatch correctness, not
    // quantization-noise minimisation (covered by test_awq_quantizer.cpp).
    auto weight_cpu = tenzor::randn({32, 64}, DType::Float32, Device::cpu());
    auto scales_cpu = tenzor::abs(tenzor::randn({64}, DType::Float32,
                                                Device::cpu()));

    // Audit-5: removed a `try { ... } catch (const std::exception&) {
    // GTEST_SKIP(); }` wrapper that hid quantize_layer dispatch failures as
    // "unsupported on this backend/dtype". The dtype filter above is the
    // legitimate precondition; let exceptions from the work propagate.
    auto weight = weight_cpu.to(dtype_).to(device_);
    auto scales = scales_cpu.to(dtype_).to(device_);

    auto result = q.quantize_layer(weight, scales);
    ASSERT_EQ(result.quantized_weight.dim(), 2);
    EXPECT_EQ(result.quantized_weight.size(0), 32);
    EXPECT_GT(result.scales.numel(), 0);
    ASSERT_EQ(result.in_features, 64);

    // Reconstruct the dequantized weight from the packed INT4 codes and verify
    // it round-trips to within the analytic INT4 half-quant-step PLUS the
    // dtype-conversion slack. The quantizer upcasts to Float32 internally, so
    // the only additional error over the pure-Float32 case is the input-side
    // rounding from converting the reference to dtype_ — captured by
    // awq_atol_for(dtype_). Both terms are real; a quantizer emitting garbage
    // of the right shape (the previous shape-only check) fails this.
    auto recon = ::tenzor::testing::awq_detail::reconstruct_awq(result,
                                                                config.group_size);

    auto ref_cpu = weight_cpu.to(DType::Float32);  // original Float32 reference
    const float* ref_p = ref_cpu.data<float>();
    auto scales_out = result.scales.to(Device::cpu()).to(DType::Float32);
    auto act_out    = result.act_scales.to(Device::cpu()).to(DType::Float32);
    const float* scales_p = scales_out.data<float>();
    const float* act_p    = act_out.data<float>();
    const int64_t out_features = result.scales.size(0);
    const int64_t num_groups   = result.scales.size(1);
    const int64_t in_features  = result.in_features;

    const float dtype_slack = awq_atol_for(dtype_);
    double max_excess = 0.0, max_abs_err = 0.0;
    for (int64_t o = 0; o < out_features; ++o) {
        for (int64_t j = 0; j < in_features; ++j) {
            int64_t g = j / config.group_size;
            float scale = scales_p[o * num_groups + g];
            float aj    = act_p[j];
            double step = 0.5 * static_cast<double>(scale) /
                std::max(static_cast<double>(std::abs(aj)), 1e-12);
            double bound = step + dtype_slack;
            double err = std::abs(
                static_cast<double>(recon[o * in_features + j]) -
                static_cast<double>(ref_p[o * in_features + j]));
            max_abs_err = std::max(max_abs_err, err);
            max_excess = std::max(max_excess, err - bound);
        }
    }
    EXPECT_LE(max_excess, 1e-4)
        << "AWQ dequant round-trip for dtype " << dtype_name(dtype_)
        << " exceeds half-quant-step+conversion bound by " << max_excess
        << " (max abs err " << max_abs_err << ")";
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(AWQQuantizerMultiDTypeTest);
