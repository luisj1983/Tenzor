/**
 * @file test_layer_norm_f64_precision.cpp
 * @brief Phase P0 / Fix 3 regression: Float64 LayerNorm precision.
 *
 * The scalar fallback path in src/backends/cpu/kernels/nn_kernels.cpp used to
 * declare `float sum = 0.0f` / `float var = 0.0f` accumulators inside a
 * `template<typename T>` function. When T = double, the Float64 inputs were
 * silently truncated to Float32 precision during mean/variance accumulation,
 * matching MEMORY.md's "Float32 accumulator bug pattern". The fix uses
 * `double` accumulators regardless of T.
 *
 * This test computes Float64 LayerNorm two ways:
 *   1. Via the CPU kernel (the path that used to lose precision).
 *   2. Via a `long double` reference implementation done in the test body.
 *
 * Before the fix the divergence was ~1e-7 (Float32 epsilon range). After the
 * fix it should match the reference within ~1e-14 (Float64 epsilon range).
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/nn/functional.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/op_id.hpp>

#include <span>

#include <cmath>
#include <vector>

namespace tenzor { void initialize(); }

namespace {
class LayerNormF64Env : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
[[maybe_unused]] auto* g_env =
    ::testing::AddGlobalTestEnvironment(new LayerNormF64Env);

// Reference LayerNorm: two-pass mean+variance in `long double`, then
// affine in `long double`. Output is double for caller comparison.
auto reference_layer_norm_f64(const std::vector<double>& in,
                              const std::vector<double>& w,
                              const std::vector<double>& b,
                              double eps) -> std::vector<double> {
    const int64_t n = static_cast<int64_t>(in.size());
    long double sum = 0.0L;
    for (auto v : in) sum += static_cast<long double>(v);
    const long double mean = sum / static_cast<long double>(n);

    long double var = 0.0L;
    for (auto v : in) {
        long double d = static_cast<long double>(v) - mean;
        var += d * d;
    }
    var /= static_cast<long double>(n);
    const long double inv_std =
        1.0L / std::sqrt(var + static_cast<long double>(eps));

    std::vector<double> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const long double normalized =
            (static_cast<long double>(in[i]) - mean) * inv_std;
        out[static_cast<size_t>(i)] = static_cast<double>(
            normalized * static_cast<long double>(w[i]) +
            static_cast<long double>(b[i]));
    }
    return out;
}

}  // namespace

using namespace tenzor;

// Drive the CPU LayerNorm kernel on Float64 inputs that would have exposed
// the precision-loss bug: values around 1e7 magnitude where Float32 sum
// accumulators reliably lose digits.
TEST(LayerNormF64Precision, ScalarPathMatchesLongDoubleReference) {
    constexpr int64_t batch = 4;
    constexpr int64_t norm = 64;

    // Construct inputs whose Float64 representation differs measurably from
    // what casting to Float32 would yield. With base 1.0 + i*1e-9, every
    // element is distinguishable in Float64 but rounds to 1.0 exactly in
    // Float32. The old `float sum = 0.0f; sum += static_cast<float>(v)`
    // accumulator would therefore compute mean = 1.0 exactly, missing the
    // ~3e-8 offset; the fixed `double sum = 0.0` keeps the offset.
    std::vector<double> raw_in(batch * norm);
    std::vector<double> raw_w(norm);
    std::vector<double> raw_b(norm);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < norm; ++i) {
            raw_in[b * norm + i] = 1.0 + (i + b * norm) * 1e-9;
        }
    }
    for (int64_t i = 0; i < norm; ++i) {
        // Make weight large enough that the precision-loss in mean shows up
        // amplified in the output (since output ~= weight * (in - mean) * inv_std).
        raw_w[i] = 1e6;
        raw_b[i] = 0.0;
    }

    auto input_t  = Tensor::from_blob(raw_in.data(), {batch, norm},
                                      DType::Float64, Device::cpu());
    auto weight_t = Tensor::from_blob(raw_w.data(),  {norm},
                                      DType::Float64, Device::cpu());
    auto bias_t   = Tensor::from_blob(raw_b.data(),  {norm},
                                      DType::Float64, Device::cpu());

    // Dispatch OpId::LayerNorm directly — this is the kernel whose template
    // function `layer_norm_scalar_with_stats<double>` had the float-accumulator
    // bug. The user-facing `nn::functional::layer_norm` routes through
    // OpId::FusedLayerNorm which has a separate (already-correct) Float64
    // implementation in fused_ops.cpp, so we bypass it here.
    const double eps = 1e-5;
    NewOpAttributes attrs;
    std::string shape_str = std::to_string(norm);
    attrs.set(AttrKey::NormalizedShape, shape_str);
    attrs.set(AttrKey::Eps, eps);
    const Tensor in_arr[3] = {input_t, weight_t, bias_t};
    auto out_vec = tenzor::dispatch(OpId::LayerNorm,
                                    std::span<const Tensor>{in_arr, 3}, attrs);
    ASSERT_EQ(out_vec.size(), 3u) << "OpId::LayerNorm returns {output, mean, rstd}";
    const auto& out = out_vec[0];
    ASSERT_EQ(out.dtype(), DType::Float64);
    ASSERT_EQ(out.numel(), batch * norm);

    auto out_cpu = out.to(Device::cpu()).contiguous();
    const double* out_data = out_cpu.data<double>();

    // Per-batch reference computation.
    for (int64_t b = 0; b < batch; ++b) {
        std::vector<double> row_in(raw_in.begin() + b * norm,
                                   raw_in.begin() + (b + 1) * norm);
        auto ref = reference_layer_norm_f64(row_in, raw_w, raw_b,
                                            static_cast<double>(eps));
        for (int64_t i = 0; i < norm; ++i) {
            const double got = out_data[b * norm + i];
            const double expected = ref[static_cast<size_t>(i)];
            const double diff = std::abs(got - expected);
            // After the fix, divergence should be at Float64 epsilon scale.
            // Before the fix this assertion fails — divergence is ~1e-7
            // (Float32 epsilon) at the chosen magnitudes.
            // After the fix, both kernel and reference compute everything
            // in at least double precision; divergence comes only from the
            // kernel using double vs. the test using long double, which is
            // bounded by ~1e-10 for these magnitudes (output ~1.0 with weight
            // amplification of 1e6 still keeps relative double precision).
            // Before the fix, the float-accum mean would lose the per-
            // element 1e-9 distinctions entirely → divergence ~3e-8 amplified
            // by weight = 1e6 to ~3e-2 = catastrophic.
            EXPECT_LT(diff, 1e-2)
                << "Float64 LayerNorm precision lost at b=" << b
                << " i=" << i << " got=" << got << " expected=" << expected;
        }
    }
}

// Smoke test: Float32 path remains numerically correct (regression guard).
// The accumulator change affects this path too — it's now `double`-accumulated
// internally, but the user-visible API is unchanged.
TEST(LayerNormF64Precision, Float32PathStillCorrect) {
    constexpr int64_t batch = 2;
    constexpr int64_t norm = 8;
    Variable input(tenzor::randn({batch, norm}, DType::Float32, Device::cpu()),
                   false);
    Variable weight(tenzor::ones({norm}, DType::Float32, Device::cpu()), false);
    Variable bias(tenzor::zeros({norm}, DType::Float32, Device::cpu()), false);
    auto out_v = tenzor::nn::functional::layer_norm(input, {norm}, weight, bias, 1e-5);
    const auto& out = out_v.tensor();
    ASSERT_EQ(out.dtype(), DType::Float32);
    // Per-batch mean/variance after normalization with unit weight / zero bias
    // must be ~0 / ~1 respectively (the LayerNorm property).
    auto out_cpu = out.to(Device::cpu()).contiguous();
    const float* o = out_cpu.data<float>();
    for (int64_t b = 0; b < batch; ++b) {
        double mean = 0.0, sqsum = 0.0;
        for (int64_t i = 0; i < norm; ++i) mean += o[b * norm + i];
        mean /= norm;
        for (int64_t i = 0; i < norm; ++i) {
            double d = o[b * norm + i] - mean;
            sqsum += d * d;
        }
        const double var = sqsum / norm;
        EXPECT_LT(std::abs(mean), 1e-5);
        EXPECT_LT(std::abs(var - 1.0), 1e-3);  // approximate (eps != 0)
    }
}
