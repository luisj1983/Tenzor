// Phase 13 / Group D.1.2 — tenzor_flash_attention custom_call dispatcher.
//
// The dispatcher (customcalls::dispatch_flash_attention) is what the IREE
// runtime plugin invokes once the iree/runtime/api.h headers are restored
// in the distribution. It takes a vector of Tensors plus the backend_config
// string and dispatches to OpId::FlashAttention via the existing fast-
// dispatch table.
//
// This test exercises the dispatcher directly with constructed tensors and
// compares its output to the eager `::tenzor::flash_attention` autograd
// helper.

#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
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

::tenzor::Tensor make_filled(const std::vector<int64_t>& shape, float seed,
                             float step) {
    auto t = ::tenzor::full(shape, 0.0f, ::tenzor::DType::Float32);
    auto* p = t.data<float>();
    const auto n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        p[i] = seed + step * static_cast<float>(i % 17) * 0.01f;
    }
    return t;
}

}  // namespace

TEST(FlashAttentionCallback, MatchesEager_NonCausal_ImplicitScale) {
    ensure_core_init();
    const std::vector<int64_t> shape{1, 2, 4, 16};

    auto q = make_filled(shape, 0.1f, 1.0f);
    auto k = make_filled(shape, 0.2f, 1.5f);
    auto v = make_filled(shape, 0.3f, 2.0f);

    // Eager: scale defaults to 1/sqrt(D) when backend_config has scale=0.
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.back()));
    auto eager = ::tenzor::flash_attention(
        ::tenzor::Variable(q, false),
        ::tenzor::Variable(k, false),
        ::tenzor::Variable(v, false),
        scale, /*causal=*/false);

    // Dispatcher: backend_config="causal=false,scale=0" (scale=0 means the
    // dispatcher picks the implicit default).
    auto out = cc::dispatch_flash_attention({q, k, v},
                                            "causal=false,scale=0");

    auto diff = ::tenzor::max(::tenzor::abs(eager.tensor() - out))
                    .item<float>();
    EXPECT_LT(diff, 1e-5f) << "callback diverged from eager by " << diff;
}

TEST(FlashAttentionCallback, MatchesEager_Causal_ExplicitScale) {
    ensure_core_init();
    const std::vector<int64_t> shape{1, 2, 8, 16};

    auto q = make_filled(shape, 0.0f, 0.5f);
    auto k = make_filled(shape, 0.0f, 0.7f);
    auto v = make_filled(shape, 0.0f, 0.9f);

    const float scale = 0.125f;
    auto eager = ::tenzor::flash_attention(
        ::tenzor::Variable(q, false),
        ::tenzor::Variable(k, false),
        ::tenzor::Variable(v, false),
        scale, /*causal=*/true);
    auto out = cc::dispatch_flash_attention({q, k, v},
                                            "causal=true,scale=0.125");

    auto diff = ::tenzor::max(::tenzor::abs(eager.tensor() - out))
                    .item<float>();
    EXPECT_LT(diff, 1e-5f) << "callback diverged from eager by " << diff;
}

TEST(FlashAttentionCallback, RejectsWrongInputCount) {
    ensure_core_init();
    auto q = ::tenzor::full({1, 2, 4, 16}, 0.1f, ::tenzor::DType::Float32);
    EXPECT_THROW(cc::dispatch_flash_attention({q, q}, "causal=false"),
                 std::runtime_error);
}

TEST(FlashAttentionCallback, BackendConfigParserHandlesWhitespaceFreeForm) {
    ensure_core_init();
    const std::vector<int64_t> shape{1, 1, 4, 8};
    auto q = make_filled(shape, 0.0f, 0.1f);
    auto k = make_filled(shape, 0.1f, 0.2f);
    auto v = make_filled(shape, 0.2f, 0.3f);

    // Empty backend_config: causal=false, scale=0 (implicit default).
    auto r1 = cc::dispatch_flash_attention({q, k, v}, "");
    // Just causal=true: scale falls back to default.
    auto r2 = cc::dispatch_flash_attention({q, k, v}, "causal=true");

    // r1 (non-causal) and r2 (causal) must differ — the causal mask
    // changes the attention pattern at every non-final row.
    auto diff = ::tenzor::max(::tenzor::abs(r1 - r2)).item<float>();
    EXPECT_GT(diff, 1e-4f) << "causal vs non-causal results should differ";
}
