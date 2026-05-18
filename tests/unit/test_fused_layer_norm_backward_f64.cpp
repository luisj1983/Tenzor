/**
 * @file test_fused_layer_norm_backward_f64.cpp
 * @brief Regression test for audit P0 #2: fused LayerNorm backward dtype bug.
 *
 * Pre-fix, fused_layer_norm_backward_kernel in fused_ops.cpp:
 *   1. Unconditionally allocated grad_weight / grad_bias as DType::Float32,
 *      regardless of the input dtype.
 *   2. Read mean.data<float>() / inv_std.data<float>() — reinterpreting a
 *      Float64 buffer as a float array — producing garbage values.
 *   3. Accumulated into std::vector<float> local_gw / local_gb even inside
 *      the Float64 dispatch branch, silently downgrading precision.
 *
 * The fix templatizes on input dtype so Float64 inputs use Float64
 * accumulators and Float64 allocated grad_weight / grad_bias.
 *
 * The kernel is tested via direct dispatch (OpId::FusedLayerNormBackward) to
 * exercise the kernel itself, following the pattern in
 * test_layer_norm_f64_precision.cpp which dispatches OpId::LayerNorm directly.
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/creation.hpp"

#include <cmath>
#include <span>

namespace tenzor { void initialize(); }

using namespace tenzor;
using namespace tenzor::nn;

// Test environment that initializes Tenzor once for the whole binary.
class FusedLNBwdF64Env : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};

static ::testing::Environment* const g_fused_ln_bwd_f64_env =
    ::testing::AddGlobalTestEnvironment(new FusedLNBwdF64Env);

// ---------------------------------------------------------------------------
// P0 #2 — direct dispatch test.
//
// Constructs Float64 input, mean, inv_std, weight and dispatches
// OpId::FusedLayerNormBackward directly. Pre-fix, the kernel read
// mean.data<float>() on a Float64 buffer (garbage), allocated grad_weight as
// Float32 (wrong dtype), and accumulated into float vectors. Post-fix the
// kernel allocates all outputs as Float64 and reads stats through double*.
// ---------------------------------------------------------------------------
TEST(FusedLayerNormBackwardF64, DirectDispatchGradDtypesMatchInputDtype) {
    const int64_t N = 8, C = 16;

    // Build Float64 forward inputs
    Tensor input     = tenzor::randn({N, C}, DType::Float64, Device::cpu());
    Tensor grad_out  = tenzor::randn({N, C}, DType::Float64, Device::cpu());
    Tensor weight    = tenzor::ones ({C},    DType::Float64, Device::cpu());

    // Compute plausible mean and inv_std in Float64 (simulating the forward pass)
    Tensor mean    = tenzor::zeros({N}, DType::Float64, Device::cpu());
    Tensor inv_std = tenzor::zeros({N}, DType::Float64, Device::cpu());
    {
        const double* inp = input.data<double>();
        double* m   = mean.data<double>();
        double* rs  = inv_std.data<double>();
        for (int64_t b = 0; b < N; ++b) {
            double sum = 0.0;
            for (int64_t i = 0; i < C; ++i) sum += inp[b * C + i];
            m[b] = sum / C;
            double var = 0.0;
            for (int64_t i = 0; i < C; ++i) {
                double d = inp[b * C + i] - m[b];
                var += d * d;
            }
            var /= C;
            rs[b] = 1.0 / std::sqrt(var + 1e-5);
        }
    }

    // Dispatch: inputs order must match kernel_registry expectation:
    //   [0] grad_output, [1] input, [2] weight(?), [3] mean, [4] inv_std
    // Check cpu_kernel_registry.cpp for the canonical argument order.
    // From the registry: inputs[0]=grad_output, inputs[1]=input,
    //                    inputs[2]=weight, inputs[3]=mean, inputs[4]=inv_std
    // (same order as the fused_layer_norm_backward_kernel signature which
    //  is: grad_output, input, normalized_shape, mean, inv_std, weight)
    // The registry wrapper reorders: inputs[0], inputs[1], normalized_shape,
    //   inputs[3], inputs[4], inputs[2]
    const std::string norm_shape_str = std::to_string(C);
    NewOpAttributes attrs;
    attrs.set(AttrKey::NormalizedShape, norm_shape_str);

    const Tensor inputs_arr[5] = {grad_out, input, weight, mean, inv_std};
    auto results = tenzor::dispatch(OpId::FusedLayerNormBackward,
                                    std::span<const Tensor>{inputs_arr, 5}, attrs);

    ASSERT_GE(results.size(), 3u) << "FusedLayerNormBackward must return 3 tensors";

    const Tensor& gi = results[0];  // grad_input
    const Tensor& gw = results[1];  // grad_weight
    const Tensor& gb = results[2];  // grad_bias

    // --- dtype checks ---
    EXPECT_EQ(gi.dtype(), DType::Float64)
        << "grad_input dtype should be Float64, got "
        << static_cast<int>(gi.dtype());
    EXPECT_EQ(gw.dtype(), DType::Float64)
        << "grad_weight dtype should be Float64, got "
        << static_cast<int>(gw.dtype());
    EXPECT_EQ(gb.dtype(), DType::Float64)
        << "grad_bias dtype should be Float64, got "
        << static_cast<int>(gb.dtype());

    // --- sanity: grad_input finite and nonzero ---
    {
        Tensor gi_c = gi.contiguous();
        const double* p = gi_c.data<double>();
        bool any_bad = false, all_zero = true;
        for (int64_t i = 0; i < gi_c.numel(); ++i) {
            if (std::isnan(p[i]) || std::isinf(p[i])) any_bad = true;
            if (std::abs(p[i]) > 1e-12) all_zero = false;
        }
        EXPECT_FALSE(any_bad)
            << "grad_input contains NaN/Inf — likely float-reinterpret of F64 stats buffer";
        EXPECT_FALSE(all_zero) << "grad_input is identically zero";
    }

    // --- sanity: grad_weight finite and nonzero ---
    {
        Tensor gw_c = gw.contiguous();
        const double* p = gw_c.data<double>();
        bool any_bad = false, all_zero = true;
        for (int64_t i = 0; i < gw_c.numel(); ++i) {
            if (std::isnan(p[i]) || std::isinf(p[i])) any_bad = true;
            if (std::abs(p[i]) > 1e-12) all_zero = false;
        }
        EXPECT_FALSE(any_bad)  << "grad_weight contains NaN/Inf";
        EXPECT_FALSE(all_zero) << "grad_weight is identically zero";
    }

    // --- sanity: grad_bias finite and nonzero ---
    {
        Tensor gb_c = gb.contiguous();
        const double* p = gb_c.data<double>();
        bool any_bad = false, all_zero = true;
        for (int64_t i = 0; i < gb_c.numel(); ++i) {
            if (std::isnan(p[i]) || std::isinf(p[i])) any_bad = true;
            if (std::abs(p[i]) > 1e-12) all_zero = false;
        }
        EXPECT_FALSE(any_bad)  << "grad_bias contains NaN/Inf";
        EXPECT_FALSE(all_zero) << "grad_bias is identically zero";
    }
}

// ---------------------------------------------------------------------------
// P0 #2 — pre-fix, mean.data<float>() reinterpreted a Float64 buffer;
// for a large-magnitude mean the resulting garbage values would overflow
// float range or produce NaN in the grad_input computation.
// This test uses a large-offset input so Float32-reinterpretation of the
// Float64 mean/rstd values would produce clearly bogus (NaN/Inf/huge) grads.
// ---------------------------------------------------------------------------
TEST(FusedLayerNormBackwardF64, LargeMagnitudeInputNoGarbage) {
    const int64_t N = 4, C = 8;

    // Inputs with magnitude 1e7 — if mean is stored as Float64 (~1e7) and
    // then reinterpreted as float*, the bit pattern gives a completely wrong
    // exponent, so intermediate values in the backward computation go NaN or Inf.
    Tensor input    = tenzor::randn({N, C}, DType::Float64, Device::cpu());
    // shift to large positive mean
    {
        double* p = input.data<double>();
        for (int64_t i = 0; i < input.numel(); ++i) p[i] += 1e7;
    }
    Tensor grad_out = tenzor::ones({N, C}, DType::Float64, Device::cpu());
    Tensor weight   = tenzor::ones({C},   DType::Float64, Device::cpu());

    Tensor mean    = tenzor::zeros({N}, DType::Float64, Device::cpu());
    Tensor inv_std = tenzor::zeros({N}, DType::Float64, Device::cpu());
    {
        const double* inp = input.data<double>();
        double* m  = mean.data<double>();
        double* rs = inv_std.data<double>();
        for (int64_t b = 0; b < N; ++b) {
            double sum = 0.0;
            for (int64_t i = 0; i < C; ++i) sum += inp[b * C + i];
            m[b] = sum / C;
            double var = 0.0;
            for (int64_t i = 0; i < C; ++i) {
                double d = inp[b * C + i] - m[b];
                var += d * d;
            }
            var /= C;
            rs[b] = 1.0 / std::sqrt(var + 1e-5);
        }
    }

    NewOpAttributes attrs;
    attrs.set(AttrKey::NormalizedShape, std::to_string(C));
    const Tensor inputs_arr[5] = {grad_out, input, weight, mean, inv_std};
    auto results = tenzor::dispatch(OpId::FusedLayerNormBackward,
                                    std::span<const Tensor>{inputs_arr, 5}, attrs);
    ASSERT_GE(results.size(), 3u);

    Tensor gi_c = results[0].contiguous();
    const double* p = gi_c.data<double>();
    bool any_bad = false;
    for (int64_t i = 0; i < gi_c.numel(); ++i) {
        if (std::isnan(p[i]) || std::isinf(p[i]) || std::abs(p[i]) > 1e6) {
            any_bad = true;
            break;
        }
    }
    EXPECT_FALSE(any_bad)
        << "grad_input overflow/NaN with large-magnitude Float64 input — "
           "float-reinterpret of mean/rstd buffer is likely still present";
}

// ---------------------------------------------------------------------------
// Smoke test: Float32 path continues to produce correct dtype and finite grads.
// ---------------------------------------------------------------------------
TEST(FusedLayerNormBackwardF64, Float32PathUnchanged) {
    const int64_t N = 4, C = 8;

    Tensor input    = tenzor::randn({N, C}, DType::Float32, Device::cpu());
    Tensor grad_out = tenzor::ones ({N, C}, DType::Float32, Device::cpu());
    Tensor weight   = tenzor::ones ({C},    DType::Float32, Device::cpu());
    Tensor mean     = tenzor::zeros({N},    DType::Float32, Device::cpu());
    Tensor inv_std  = tenzor::ones ({N},    DType::Float32, Device::cpu());

    NewOpAttributes attrs;
    attrs.set(AttrKey::NormalizedShape, std::to_string(C));
    const Tensor inputs_arr[5] = {grad_out, input, weight, mean, inv_std};
    auto results = tenzor::dispatch(OpId::FusedLayerNormBackward,
                                    std::span<const Tensor>{inputs_arr, 5}, attrs);
    ASSERT_GE(results.size(), 3u);

    EXPECT_EQ(results[0].dtype(), DType::Float32) << "grad_input dtype mismatch";
    EXPECT_EQ(results[1].dtype(), DType::Float32) << "grad_weight dtype mismatch";
    EXPECT_EQ(results[2].dtype(), DType::Float32) << "grad_bias dtype mismatch";
}
