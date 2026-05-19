// Phase 13 / Group D.3.2 — tenzor_rope_apply custom_call dispatcher.
//
// Dispatcher: out = x * cos + rotate_half(x) * sin, computed with raw
// Tensor ops (narrow + neg + cat + multiply + add). The `offset` int attr
// is parsed but not used — the eager precomputation bakes the offset into
// the cos/sin tables.

#include "tenzor/jit/mlir/iree_customcalls.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

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

TEST(RoPECallback, MatchesHandComputed_Position1WithRotation) {
    ensure_core_init();
    // x: (1, 1, 2, 4) = [[1,2,3,4], [5,6,7,8]]
    // cos = [[1,1,1,1],[0.5,0.5,0.5,0.5]]; sin = [[0,0,0,0],[0.5,0.5,0.5,0.5]]
    // Position 0: out == x
    // Position 1: rotate_half([5,6,7,8]) = [-7,-8,5,6]
    //   out = [2.5, 3, 3.5, 4] + [-3.5, -4, 2.5, 3] = [-1, -1, 6, 7]
    const std::vector<int64_t> x_shape{1, 1, 2, 4};
    const std::vector<int64_t> tab_shape{2, 4};
    auto x = ::tenzor::full(x_shape, 0.0f, ::tenzor::DType::Float32);
    {
        auto* p = x.data<float>();
        for (int64_t i = 0; i < 8; ++i) p[i] = static_cast<float>(i + 1);
    }
    auto cos = ::tenzor::full(tab_shape, 0.0f, ::tenzor::DType::Float32);
    auto sin = ::tenzor::full(tab_shape, 0.0f, ::tenzor::DType::Float32);
    {
        auto* cp = cos.data<float>();
        auto* sp = sin.data<float>();
        for (int64_t i = 0; i < 4; ++i) { cp[i] = 1.0f; sp[i] = 0.0f; }
        for (int64_t i = 0; i < 4; ++i) { cp[4 + i] = 0.5f; sp[4 + i] = 0.5f; }
    }

    auto out = cc::dispatch_rope_apply({x, cos, sin}, "offset=0");
    const float* op = out.data<float>();
    EXPECT_NEAR(op[0],  1.0f, 1e-5f);
    EXPECT_NEAR(op[1],  2.0f, 1e-5f);
    EXPECT_NEAR(op[2],  3.0f, 1e-5f);
    EXPECT_NEAR(op[3],  4.0f, 1e-5f);
    EXPECT_NEAR(op[4], -1.0f, 1e-5f);
    EXPECT_NEAR(op[5], -1.0f, 1e-5f);
    EXPECT_NEAR(op[6],  6.0f, 1e-5f);
    EXPECT_NEAR(op[7],  7.0f, 1e-5f);
}

TEST(RoPECallback, IdentityWhenCosOneSinZero) {
    ensure_core_init();
    // cos = ones, sin = zeros => out == x (RoPE no-op).
    const std::vector<int64_t> x_shape{2, 1, 4, 8};
    const std::vector<int64_t> tab_shape{4, 8};
    auto x = ::tenzor::full(x_shape, 0.0f, ::tenzor::DType::Float32);
    {
        auto* p = x.data<float>();
        const auto n = x.numel();
        for (int64_t i = 0; i < n; ++i) p[i] = static_cast<float>(i) * 0.1f;
    }
    auto cos = ::tenzor::full(tab_shape, 1.0f, ::tenzor::DType::Float32);
    auto sin = ::tenzor::full(tab_shape, 0.0f, ::tenzor::DType::Float32);

    auto out = cc::dispatch_rope_apply({x, cos, sin}, "");
    auto diff = ::tenzor::max(::tenzor::abs(x - out)).item<float>();
    EXPECT_LT(diff, 1e-6f);
}

TEST(RoPECallback, RejectsOddLastDim) {
    ensure_core_init();
    auto x = ::tenzor::full({1, 1, 2, 3}, 1.0f, ::tenzor::DType::Float32);
    auto cos = ::tenzor::full({2, 3}, 1.0f, ::tenzor::DType::Float32);
    auto sin = ::tenzor::full({2, 3}, 0.0f, ::tenzor::DType::Float32);
    EXPECT_THROW(cc::dispatch_rope_apply({x, cos, sin}, "offset=0"),
                 std::runtime_error);
}

TEST(RoPECallback, RejectsWrongInputCount) {
    ensure_core_init();
    auto x = ::tenzor::full({1, 1, 2, 4}, 1.0f, ::tenzor::DType::Float32);
    EXPECT_THROW(cc::dispatch_rope_apply({x, x}, ""),
                 std::runtime_error);
}
