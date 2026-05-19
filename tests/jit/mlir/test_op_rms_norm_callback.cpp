// Phase 13 / Group D.4.2 — tenzor_rms_norm custom_call dispatcher.
//
// Dispatcher: parses `eps=<f>` from backend_config, materializes an
// all-ones weight when caller omitted it, and dispatches to
// OpId::RMSNorm. Validated against hand-computed references.

#include "tenzor/jit/mlir/iree_customcalls.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace cc = ::tenzor::jit::mlir_jit::customcalls;

namespace {

auto ensure_core_init() -> void {
    static const bool inited = []() {
        ::tenzor::initialize();
        return true;
    }();
    (void)inited;
}

}  // namespace

TEST(RMSNormCallback, MatchesHandComputed_UnitWeight_EpsTinyNotZero) {
    ensure_core_init();
    // x = [3, 4, 0, 0]; weight = 1; eps = 1e-8 (effectively zero).
    // mean(x^2) = 25/4 = 6.25; rms = 2.5
    // out = x / 2.5 = [1.2, 1.6, 0, 0]
    const std::vector<int64_t> x_shape{1, 1, 4};
    const std::vector<int64_t> w_shape{4};
    auto x = ::tenzor::full(x_shape, 0.f, ::tenzor::DType::Float32);
    auto w = ::tenzor::full(w_shape, 1.f, ::tenzor::DType::Float32);
    {
        auto* p = x.data<float>();
        p[0] = 3.f; p[1] = 4.f; p[2] = 0.f; p[3] = 0.f;
    }
    auto out = cc::dispatch_rms_norm({x, w}, "eps=1e-8");
    const float* op = out.data<float>();
    EXPECT_NEAR(op[0], 1.2f, 1e-4f);
    EXPECT_NEAR(op[1], 1.6f, 1e-4f);
    EXPECT_NEAR(op[2], 0.0f, 1e-4f);
    EXPECT_NEAR(op[3], 0.0f, 1e-4f);
}

TEST(RMSNormCallback, MatchesHandComputed_NonUnitWeight) {
    ensure_core_init();
    // x = [1, 2, 3, 4]; weight = [2, 2, 2, 2]; eps = 1e-8
    // mean(x^2) = 30/4 = 7.5; rms = sqrt(7.5)
    // out = x / sqrt(7.5) * 2
    const std::vector<int64_t> x_shape{1, 1, 4};
    const std::vector<int64_t> w_shape{4};
    auto x = ::tenzor::full(x_shape, 0.f, ::tenzor::DType::Float32);
    auto w = ::tenzor::full(w_shape, 2.f, ::tenzor::DType::Float32);
    {
        auto* p = x.data<float>();
        p[0] = 1.f; p[1] = 2.f; p[2] = 3.f; p[3] = 4.f;
    }
    auto out = cc::dispatch_rms_norm({x, w}, "eps=1e-8");
    const float* op = out.data<float>();
    const float rms = std::sqrt(7.5f);
    EXPECT_NEAR(op[0], (1.f / rms) * 2.f, 1e-4f);
    EXPECT_NEAR(op[1], (2.f / rms) * 2.f, 1e-4f);
    EXPECT_NEAR(op[2], (3.f / rms) * 2.f, 1e-4f);
    EXPECT_NEAR(op[3], (4.f / rms) * 2.f, 1e-4f);
}

TEST(RMSNormCallback, NoWeightDefaultsToOnes) {
    ensure_core_init();
    // Pass only x (no weight); dispatcher must materialize all-ones weight.
    // Expect same numerical output as passing explicit ones weight.
    const std::vector<int64_t> x_shape{1, 1, 4};
    auto x = ::tenzor::full(x_shape, 0.f, ::tenzor::DType::Float32);
    {
        auto* p = x.data<float>();
        p[0] = 3.f; p[1] = 4.f; p[2] = 0.f; p[3] = 0.f;
    }
    auto out1 = cc::dispatch_rms_norm({x}, "eps=1e-8");
    auto ones = ::tenzor::full({4}, 1.f, ::tenzor::DType::Float32);
    auto out2 = cc::dispatch_rms_norm({x, ones}, "eps=1e-8");
    auto diff = ::tenzor::max(::tenzor::abs(out1 - out2)).item<float>();
    EXPECT_LT(diff, 1e-6f);
}

TEST(RMSNormCallback, RejectsEmptyInputs) {
    ensure_core_init();
    EXPECT_THROW(cc::dispatch_rms_norm({}, "eps=1e-6"), std::runtime_error);
}

TEST(RMSNormCallback, EpsAffectsResultWhenXNearZero) {
    ensure_core_init();
    // With x = tiny vector and a large eps, the rms term is dominated by
    // sqrt(eps) so the output should differ substantially from the no-eps
    // case.
    auto x = ::tenzor::full({1, 1, 4}, 1e-3f, ::tenzor::DType::Float32);
    auto w = ::tenzor::full({4},       1.0f,   ::tenzor::DType::Float32);
    auto small_eps = cc::dispatch_rms_norm({x, w}, "eps=1e-12");
    auto big_eps   = cc::dispatch_rms_norm({x, w}, "eps=1.0");
    auto diff = ::tenzor::max(::tenzor::abs(small_eps - big_eps))
                    .item<float>();
    EXPECT_GT(diff, 1e-4f) << "eps changes must propagate through dispatch";
}
