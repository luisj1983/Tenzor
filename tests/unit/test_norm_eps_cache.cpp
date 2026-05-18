/**
 * @file test_norm_eps_cache.cpp
 * @brief Regression test for audit P0 #6: oneDNN primitive cache keys must
 *        include eps.
 *
 * Pre-fix, LayerNormCacheKey{batch_size, norm_size} and
 * BatchNormCacheKey{N, C, H, W, dtype} both omitted the eps field.
 * oneDNN bakes eps into the primitive descriptor at creation time, so two
 * LayerNorm / BatchNorm2d modules with identical shapes but different eps
 * values would silently share a cached primitive — the second module ran
 * with the first module's eps, corrupting the normalization denominator.
 *
 * IMPORTANT: oneDNN is only engaged above a size threshold (LayerNorm >=8M
 * elements, BatchNorm >10M elements). Tests below use tensors that exceed
 * these thresholds so the oneDNN code-path — and therefore the cache — is
 * actually exercised.
 *
 * Input design: x = randn * 1e-3 so variance ≈ 1e-6. Then:
 *   eps_default=1e-5:  inv_std ≈ 1/sqrt(1.1e-5) ≈ 301
 *   eps_tiny=1e-12:    inv_std ≈ 1/sqrt(1e-6)   ≈ 1000
 * The two modules produce outputs differing by ~3× — well above the 1e-3
 * threshold used in the EXPECT_GT assertions.
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"

namespace tenzor { void initialize(); }

using namespace tenzor;
using namespace tenzor::nn;

// ---------------------------------------------------------------------------
// Test environment: initialise backends once per binary
// ---------------------------------------------------------------------------
class NormEpsCacheEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};

static ::testing::Environment* const g_env =
    ::testing::AddGlobalTestEnvironment(new NormEpsCacheEnv);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

// Max absolute element-wise difference, cast to float64 for accuracy.
double max_abs_diff(const Tensor& a, const Tensor& b) {
    auto diff = abs(a.to(DType::Float64) - b.to(DType::Float64));
    return max(diff).item<double>();
}

// Mean absolute element-wise difference.
double mean_abs_diff(const Tensor& a, const Tensor& b) {
    auto diff = abs(a.to(DType::Float64) - b.to(DType::Float64));
    return mean(diff).item<double>();
}

} // namespace

// ---------------------------------------------------------------------------
// Audit P0 #6 — LayerNorm
//
// Shape: [1, 8388608] = 8 Mi elements — exceeds the oneDNN LN threshold of
// 8*1024*1024 so the primitive cache is actually exercised.
// Input: randn scaled by 1e-3 → variance ≈ 1e-6 so eps dominates.
// ---------------------------------------------------------------------------
TEST(NormEpsCache, LayerNormDistinctEpsProducesDistinctOutput) {
    // 8M elements — just at/above oneDNN threshold
    const int64_t batch_size = 1;
    const int64_t norm_size  = 8 * 1024 * 1024;

    // Small-variance input: randn * 1e-3 so var ≈ 1e-6
    auto x_t = randn({batch_size, norm_size}, DType::Float32) * 1e-3f;
    auto x = Variable(x_t, false);

    LayerNorm ln_default({norm_size}, /*eps=*/1e-5);
    LayerNorm ln_tiny   ({norm_size}, /*eps=*/1e-12);

    // Run default first so it populates the cache; then run tiny.
    // Pre-fix: tiny would reuse default's primitive (same {batch, norm} key)
    // and produce output based on eps=1e-5 instead of eps=1e-12.
    auto y_default = ln_default.forward(x).tensor();
    auto y_tiny    = ln_tiny.forward(x).tensor();

    double diff = mean_abs_diff(y_default, y_tiny);
    EXPECT_GT(diff, 1e-3)
        << "LayerNorm mean abs diff across eps=1e-5 vs eps=1e-12 is " << diff
        << " — outputs should differ significantly when eps is much larger than"
        << " variance. oneDNN primitive cache key missing eps (audit P0 #6).";
}

// ---------------------------------------------------------------------------
// Audit P0 #6 — BatchNorm2d
//
// Shape: [2, 4, 1024, 1280] = 10,485,760 elements — exceeds the oneDNN BN
// threshold of 10M elements so the cache is exercised.
// ---------------------------------------------------------------------------
TEST(NormEpsCache, BatchNormDistinctEpsProducesDistinctOutput) {
    // 2 * 4 * 1024 * 1280 = 10,485,760 > 10M → oneDNN path
    const int64_t N = 2, C = 4, H = 1024, W = 1280;

    auto x_t = randn({N, C, H, W}, DType::Float32) * 1e-3f;
    auto x = Variable(x_t, false);

    BatchNorm2d bn_default(C, /*eps=*/1e-5);
    BatchNorm2d bn_tiny   (C, /*eps=*/1e-12);

    auto y_default = bn_default.forward(x).tensor();
    auto y_tiny    = bn_tiny.forward(x).tensor();

    double diff = mean_abs_diff(y_default, y_tiny);
    EXPECT_GT(diff, 1e-3)
        << "BatchNorm mean abs diff across eps=1e-5 vs eps=1e-12 is " << diff
        << " — cache key missing eps (audit P0 #6).";
}

// ---------------------------------------------------------------------------
// Order independence: the cached primitive for ln_tiny must not be replaced
// or corrupted when ln_default runs between two ln_tiny.forward() calls.
// Uses the same large tensor shape so the oneDNN path is exercised.
// ---------------------------------------------------------------------------
TEST(NormEpsCache, OrderIndependence) {
    const int64_t batch_size = 1;
    const int64_t norm_size  = 8 * 1024 * 1024;

    auto x_t = randn({batch_size, norm_size}, DType::Float32) * 1e-3f;
    auto x = Variable(x_t, false);

    LayerNorm ln_a({norm_size}, /*eps=*/1e-5);
    LayerNorm ln_b({norm_size}, /*eps=*/1e-12);

    // Run b first, then a, then b again — b's output must be identical both times.
    auto y_b1 = ln_b.forward(x).tensor();
    (void)ln_a.forward(x);
    auto y_b2 = ln_b.forward(x).tensor();

    double diff = max_abs_diff(y_b1, y_b2);
    EXPECT_LT(diff, 1e-9)
        << "ln_b output changed after ln_a forward — cache invalidation / "
        << "collision bug (audit P0 #6). diff=" << diff;
}
