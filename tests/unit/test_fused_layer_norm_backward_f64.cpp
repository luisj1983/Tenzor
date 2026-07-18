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
 *
 * Ported to cross-backend via BackendTest (TEST_P fanned over all backends).
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/creation.hpp"
#include "../backend_test_fixture.hpp"

#include <cmath>
#include <span>

using namespace tenzor;
using namespace tenzor::nn;

class FusedLayerNormBackwardF64 : public ::tenzor::testing::BackendTest {};

// ---------------------------------------------------------------------------
// P0 #2 — direct dispatch test.
//
// Constructs Float64 input, mean, inv_std, weight and dispatches
// OpId::FusedLayerNormBackward directly. Pre-fix, the kernel read
// mean.data<float>() on a Float64 buffer (garbage), allocated grad_weight as
// Float32 (wrong dtype), and accumulated into float vectors. Post-fix the
// kernel allocates all outputs as Float64 and reads stats through double*.
// ---------------------------------------------------------------------------
TEST_P(FusedLayerNormBackwardF64, DirectDispatchGradDtypesMatchInputDtype) {
    const int64_t N = 8, C = 16;

    // Build Float64 forward inputs on the target device
    Tensor input     = tenzor::randn({N, C}, DType::Float64, device);
    Tensor grad_out  = tenzor::randn({N, C}, DType::Float64, device);
    Tensor weight    = tenzor::ones ({C},    DType::Float64, device);

    // Compute plausible mean and inv_std in Float64 (simulating the forward pass).
    // Read input on the host, compute stats on the host, then move to device.
    Tensor mean_cpu    = tenzor::zeros({N}, DType::Float64, Device::cpu());
    Tensor inv_std_cpu = tenzor::zeros({N}, DType::Float64, Device::cpu());
    {
        Tensor input_cpu = input.cpu();
        const double* inp = input_cpu.data<double>();
        double* m   = mean_cpu.data<double>();
        double* rs  = inv_std_cpu.data<double>();
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
    Tensor mean    = mean_cpu.to(device);
    Tensor inv_std = inv_std_cpu.to(device);

    // Dispatch: inputs order must match kernel_registry expectation:
    //   [0] grad_output, [1] input, [2] weight(?), [3] mean, [4] inv_std
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
        Tensor gi_c = gi.contiguous().cpu();
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
        Tensor gw_c = gw.contiguous().cpu();
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
        Tensor gb_c = gb.contiguous().cpu();
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
TEST_P(FusedLayerNormBackwardF64, LargeMagnitudeInputNoGarbage) {
    const int64_t N = 4, C = 8;

    // Inputs with magnitude 1e7 — if mean is stored as Float64 (~1e7) and
    // then reinterpreted as float*, the bit pattern gives a completely wrong
    // exponent, so intermediate values in the backward computation go NaN or Inf.
    Tensor input = tenzor::randn({N, C}, DType::Float64, device);
    // shift to large positive mean (host write: edit on CPU, move back to device)
    {
        Tensor input_cpu = input.cpu();
        double* p = input_cpu.data<double>();
        for (int64_t i = 0; i < input_cpu.numel(); ++i) p[i] += 1e7;
        input = input_cpu.to(device);
    }
    Tensor grad_out = tenzor::ones({N, C}, DType::Float64, device);
    Tensor weight   = tenzor::ones({C},   DType::Float64, device);

    // Compute stats on the host from the device input, then move to device.
    Tensor mean_cpu    = tenzor::zeros({N}, DType::Float64, Device::cpu());
    Tensor inv_std_cpu = tenzor::zeros({N}, DType::Float64, Device::cpu());
    {
        Tensor input_cpu = input.cpu();
        const double* inp = input_cpu.data<double>();
        double* m  = mean_cpu.data<double>();
        double* rs = inv_std_cpu.data<double>();
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
    Tensor mean    = mean_cpu.to(device);
    Tensor inv_std = inv_std_cpu.to(device);

    NewOpAttributes attrs;
    attrs.set(AttrKey::NormalizedShape, std::to_string(C));
    const Tensor inputs_arr[5] = {grad_out, input, weight, mean, inv_std};
    auto results = tenzor::dispatch(OpId::FusedLayerNormBackward,
                                    std::span<const Tensor>{inputs_arr, 5}, attrs);
    ASSERT_GE(results.size(), 3u);

    Tensor gi_c = results[0].contiguous().cpu();
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
TEST_P(FusedLayerNormBackwardF64, Float32PathUnchanged) {
    const int64_t N = 4, C = 8;

    Tensor input    = tenzor::randn({N, C}, DType::Float32, device);
    Tensor grad_out = tenzor::ones ({N, C}, DType::Float32, device);
    Tensor weight   = tenzor::ones ({C},    DType::Float32, device);
    Tensor mean     = tenzor::zeros({N},    DType::Float32, device);
    Tensor inv_std  = tenzor::ones ({N},    DType::Float32, device);

    NewOpAttributes attrs;
    attrs.set(AttrKey::NormalizedShape, std::to_string(C));
    const Tensor inputs_arr[5] = {grad_out, input, weight, mean, inv_std};
    auto results = tenzor::dispatch(OpId::FusedLayerNormBackward,
                                    std::span<const Tensor>{inputs_arr, 5}, attrs);
    ASSERT_GE(results.size(), 3u);

    const Tensor& gi = results[0];  // grad_input
    const Tensor& gw = results[1];  // grad_weight
    const Tensor& gb = results[2];  // grad_bias

    // --- dtype checks ---
    EXPECT_EQ(gi.dtype(), DType::Float32) << "grad_input dtype mismatch";
    EXPECT_EQ(gw.dtype(), DType::Float32) << "grad_weight dtype mismatch";
    EXPECT_EQ(gb.dtype(), DType::Float32) << "grad_bias dtype mismatch";

    // --- value-sanity: grad_weight non-zero ---
    {
        Tensor gw_c = gw.contiguous().cpu();
        const float* p = gw_c.data<float>();
        float max_abs = 0.0f;
        for (int64_t i = 0; i < gw_c.numel(); ++i)
            max_abs = std::max(max_abs, std::abs(p[i]));
        EXPECT_GT(max_abs, 1e-6f)
            << "grad_weight is effectively zero — regression may have zeroed it";
    }

    // --- value-sanity: grad_bias non-zero ---
    {
        Tensor gb_c = gb.contiguous().cpu();
        const float* p = gb_c.data<float>();
        float max_abs = 0.0f;
        for (int64_t i = 0; i < gb_c.numel(); ++i)
            max_abs = std::max(max_abs, std::abs(p[i]));
        EXPECT_GT(max_abs, 1e-6f)
            << "grad_bias is effectively zero — grad_out=ones*N should sum to N per channel";
    }

    // --- value-sanity: grad_input finite (no NaN/Inf) ---
    {
        Tensor gi_c = gi.contiguous().cpu();
        const float* p = gi_c.data<float>();
        bool any_bad = false;
        for (int64_t i = 0; i < gi_c.numel(); ++i) {
            if (std::isnan(p[i]) || std::isinf(p[i])) { any_bad = true; break; }
        }
        EXPECT_FALSE(any_bad)
            << "grad_input contains NaN/Inf in the Float32 path";
    }
}

// ---------------------------------------------------------------------------
// Float16 path: ROCm's fused_layer_norm_backward_hip previously had NO
// Float16 handling at all (only BFloat16 was widened) and unconditionally
// threw "Only Float32 and Float64 supported" for Float16 input, unlike every
// other activation/fused kernel in the backend (GELU, Sigmoid, Softmax,
// FusedAddReLU, FusedGelu, FusedRMSNorm all support Float16 via widen-narrow).
// ---------------------------------------------------------------------------
TEST_P(FusedLayerNormBackwardF64, Float16PathWidenNarrow) {
    const int64_t N = 4, C = 8;

    Tensor input    = tenzor::randn({N, C}, DType::Float16, device);
    Tensor grad_out = tenzor::ones ({N, C}, DType::Float16, device);
    Tensor weight   = tenzor::ones ({C},    DType::Float16, device);
    Tensor mean     = tenzor::zeros({N},    DType::Float16, device);
    Tensor inv_std  = tenzor::ones ({N},    DType::Float16, device);

    NewOpAttributes attrs;
    attrs.set(AttrKey::NormalizedShape, std::to_string(C));
    const Tensor inputs_arr[5] = {grad_out, input, weight, mean, inv_std};
    std::vector<Tensor> results;
    ASSERT_NO_THROW({
        results = tenzor::dispatch(OpId::FusedLayerNormBackward,
                                   std::span<const Tensor>{inputs_arr, 5}, attrs);
    }) << "FusedLayerNormBackward should widen-narrow Float16 like every other "
       << "fused kernel, not throw, on " << device.to_string();
    ASSERT_GE(results.size(), 3u);

    const Tensor& gi = results[0];
    const Tensor& gw = results[1];
    const Tensor& gb = results[2];

    EXPECT_EQ(gi.dtype(), DType::Float16) << "grad_input dtype mismatch";
    EXPECT_EQ(gw.dtype(), DType::Float16) << "grad_weight dtype mismatch";
    EXPECT_EQ(gb.dtype(), DType::Float16) << "grad_bias dtype mismatch";

    Tensor gb_c = gb.contiguous().cpu().to(DType::Float32);
    const float* p = gb_c.data<float>();
    float max_abs = 0.0f;
    for (int64_t i = 0; i < gb_c.numel(); ++i) max_abs = std::max(max_abs, std::abs(p[i]));
    EXPECT_GT(max_abs, 1e-3f)
        << "grad_bias is effectively zero — grad_out=ones*N should sum to N per channel";
}

INSTANTIATE_BACKEND_TESTS(FusedLayerNormBackwardF64);
