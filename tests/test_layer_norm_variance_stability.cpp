/**
 * @file test_layer_norm_variance_stability.cpp
 * @brief Numerical-precision fix: layer_norm_simd and
 *        layer_norm_simd_with_stats used the catastrophic-cancellation
 *        formula `var = sum_sq / n - mean^2`. For inputs with large mean
 *        (the classic pre-trained-embedding scenario) this can produce
 *        negative variance and a NaN inv_std. Two-pass `E[(x - mean)^2]`
 *        is unconditionally stable.
 *
 * The regression test constructs an input with a small but non-zero
 * standard deviation around a very large mean (e.g. std=1.0, mean=1e6).
 * Pre-fix: var = (~1e12 + epsilon) - (~1e12) catastrophically cancels.
 * Post-fix: var ≈ 1.0 exactly.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/nn/layers/normalization.hpp>
#include <tenzor/ops/op_id.hpp>

#include <cmath>
#include <span>
#include <vector>

namespace tenzor { void initialize(); }

namespace {
class LayerNormStabilityEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
[[maybe_unused]] auto* g_env =
    ::testing::AddGlobalTestEnvironment(new LayerNormStabilityEnv);
}  // namespace

using namespace tenzor;

namespace {

// Helper: dispatch OpId::LayerNorm and return the per-batch mean / rstd
// saved-stats Tensors so we can directly assert their values.
struct LNResult { Tensor output, mean, rstd; };

auto run_layer_norm_f32(const std::vector<float>& in_data,
                         int64_t batch, int64_t norm_size,
                         float eps = 1e-5f) -> LNResult {
    auto x = Tensor::from_blob(const_cast<float*>(in_data.data()),
                               {batch, norm_size}, DType::Float32,
                               Device::cpu()).clone();
    std::vector<float> w_data(static_cast<size_t>(norm_size), 1.0f);
    std::vector<float> b_data(static_cast<size_t>(norm_size), 0.0f);
    auto w = Tensor::from_blob(w_data.data(), {norm_size}, DType::Float32,
                               Device::cpu()).clone();
    auto b = Tensor::from_blob(b_data.data(), {norm_size}, DType::Float32,
                               Device::cpu()).clone();
    NewOpAttributes attrs;
    attrs.set(AttrKey::NormalizedShape, std::to_string(norm_size));
    attrs.set(AttrKey::Eps, static_cast<double>(eps));
    const Tensor in_arr[3] = {x, w, b};
    auto out = tenzor::dispatch(OpId::LayerNorm,
                                std::span<const Tensor>{in_arr, 3}, attrs);
    EXPECT_EQ(out.size(), 3u);
    return {out[0], out[1], out[2]};
}

}  // namespace

TEST(LayerNormVarianceStability, LargeMeanProducesValidVariance) {
    // Generate input: each row has 64 values, all near 1e6 with std ~1.0.
    // Pre-fix: (sum_sq / n - mean^2) = (~1e12 + 1) - (~1e12) = noise around 1.
    // For values like 1e6 + small_offset, Float32 has only 7 digits of
    // precision; sum_sq computed in Float32 loses the "+ small_offset"
    // entirely → var goes negative → inv_std = NaN.
    constexpr int64_t batch = 4;
    constexpr int64_t norm_size = 64;
    std::vector<float> data(batch * norm_size);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < norm_size; ++i) {
            // Mean = 1e6, std ≈ 1 (alternating +1/-1 offsets).
            data[b * norm_size + i] = 1.0e6f + ((i % 2) ? 1.0f : -1.0f);
        }
    }

    auto r = run_layer_norm_f32(data, batch, norm_size);
    auto rstd_cpu = r.rstd.to(Device::cpu()).contiguous();
    const float* rstd = rstd_cpu.data<float>();

    for (int64_t b = 0; b < batch; ++b) {
        EXPECT_FALSE(std::isnan(rstd[b]))
            << "rstd is NaN at batch " << b
            << " — catastrophic cancellation in one-pass variance regressed";
        EXPECT_TRUE(std::isfinite(rstd[b]))
            << "rstd is non-finite at batch " << b;
        EXPECT_GT(rstd[b], 0.0f)
            << "rstd is non-positive at batch " << b
            << " (var went negative pre-fix)";
        // True variance is 1.0, so inv_std ≈ 1.0 (modulo eps).
        EXPECT_NEAR(rstd[b], 1.0f, 0.01f)
            << "rstd diverges from expected 1.0 at batch " << b;
    }
}

TEST(LayerNormVarianceStability, OutputMatchesNormalizedReference) {
    // For a row with values 0..norm_size-1, mean = (n-1)/2, var = (n^2-1)/12.
    // Verify the kernel produces normalized output matching this reference
    // within Float32 tolerance.
    constexpr int64_t norm_size = 16;
    constexpr int64_t batch = 1;
    std::vector<float> data(static_cast<size_t>(norm_size));
    for (int64_t i = 0; i < norm_size; ++i) data[i] = static_cast<float>(i);

    auto r = run_layer_norm_f32(data, batch, norm_size);
    auto out_cpu = r.output.to(Device::cpu()).contiguous();
    const float* od = out_cpu.data<float>();

    const double mean_ref = (norm_size - 1) / 2.0;
    const double var_ref = (norm_size * norm_size - 1) / 12.0;
    const double inv_std_ref = 1.0 / std::sqrt(var_ref + 1e-5);
    for (int64_t i = 0; i < norm_size; ++i) {
        const double expected = (i - mean_ref) * inv_std_ref;
        EXPECT_NEAR(od[i], expected, 1e-5)
            << "LayerNorm output diverges at i=" << i;
    }
}

// =========================================================================
// 5th-audit A1 extension: BACKWARD path on the same large-mean input.
//
// Pre-fix `LayerNormBackward::backward` recomputes variance using the
// same catastrophic-cancellation form `(sum_sq * inv_n) - mu*mu`, then
// guards on `var < eps` and silently zeros grad_input. The two-pass
// double-precision rewrite fixes the variance estimate so the gradient
// flows. Sibling to the forward fix already on main (commit 2ee72b5b).
// =========================================================================

TEST(LayerNormVarianceStability, BackwardLargeMeanProducesFiniteNonZeroGradient) {
    using namespace tenzor;
    using namespace tenzor::nn;

    const int64_t B = 2;
    const int64_t N = 64;
    auto x_t = randn({B, N}, DType::Float32, Device::cpu()) +
               full({1}, 1e6f, DType::Float32, Device::cpu());
    auto x = Variable(x_t, /*requires_grad=*/true);

    LayerNorm ln(/*normalized_shape=*/std::vector<int64_t>{N}, /*eps=*/1e-5);
    auto y = ln.forward(x);
    y.backward(ones_like(y.tensor()));

    ASSERT_TRUE(x.grad().has_value()) << "grad_input must be populated";
    auto g = x.grad().value();
    const float* p = g.data<float>();

    bool any_nan = false, any_inf = false, all_zero = true;
    for (int64_t i = 0; i < g.numel(); ++i) {
        if (std::isnan(p[i])) any_nan = true;
        if (std::isinf(p[i])) any_inf = true;
        if (p[i] != 0.0f) all_zero = false;
    }
    EXPECT_FALSE(any_nan) << "grad_input contained NaN — variance produced "
                              "non-finite intermediates (A1)";
    EXPECT_FALSE(any_inf) << "grad_input contained Inf";
    EXPECT_FALSE(all_zero) << "grad_input collapsed to zero — the "
                               "zero_variance branch fired due to F32 "
                               "catastrophic cancellation in backward (A1)";
}

TEST(LayerNormVarianceStability, BackwardSmallVarianceStable) {
    using namespace tenzor;
    using namespace tenzor::nn;
    const int64_t B = 2;
    const int64_t N = 64;
    auto x_t = randn({B, N}, DType::Float32, Device::cpu()) *
               full({1}, 1e-4f, DType::Float32, Device::cpu());
    auto x = Variable(x_t, true);
    LayerNorm ln(std::vector<int64_t>{N}, /*eps=*/1e-5);
    auto y = ln.forward(x);
    y.backward(ones_like(y.tensor()));

    ASSERT_TRUE(x.grad().has_value());
    auto g = x.grad().value();
    const float* p = g.data<float>();
    for (int64_t i = 0; i < g.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(p[i]))
            << "Non-finite grad at i=" << i << ", v=" << p[i];
    }
}
