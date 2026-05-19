// Phase 13 / Group D.2.2 — tenzor_gqa custom_call dispatcher.
//
// The dispatcher broadcasts K and V from H_kv heads to H_q heads (via
// unsqueeze + expand + reshape — same pattern as
// GroupedQueryAttention::repeat_kv), then forwards to the FlashAttention
// dispatcher. Validated against a hand-built eager reference (broadcast
// KV via the same op chain, then call autograd::flash_attention).

#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/jit/mlir/iree_customcalls.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
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
        p[i] = seed + step * static_cast<float>(i % 19) * 0.01f;
    }
    return t;
}

// Manual KV-repeat for the eager reference, mirroring the dispatcher.
::tenzor::Tensor repeat_kv(const ::tenzor::Tensor& x, int64_t Hq) {
    const auto sh = x.shape();
    const int64_t B   = sh[0];
    const int64_t Hkv = sh[1];
    const int64_t S   = sh[2];
    const int64_t D   = sh[3];
    if (Hq == Hkv) return x;
    const int64_t G = Hq / Hkv;
    auto u = ::tenzor::unsqueeze(x, 2);
    auto e = ::tenzor::expand(u, {B, Hkv, G, S, D});
    return ::tenzor::reshape(e, {B, Hq, S, D});
}

}  // namespace

TEST(GQACallback, MatchesEager_GroupSize4_NonCausal) {
    ensure_core_init();
    const std::vector<int64_t> q_shape{1, 8, 8, 16};
    const std::vector<int64_t> kv_shape{1, 2, 8, 16};

    auto q = make_filled(q_shape,  0.1f, 1.0f);
    auto k = make_filled(kv_shape, 0.2f, 1.5f);
    auto v = make_filled(kv_shape, 0.3f, 2.0f);

    const float scale = 1.0f / std::sqrt(static_cast<float>(q_shape.back()));
    auto k_full = repeat_kv(k, q_shape[1]);
    auto v_full = repeat_kv(v, q_shape[1]);
    auto eager = ::tenzor::flash_attention(
        ::tenzor::Variable(q,      false),
        ::tenzor::Variable(k_full, false),
        ::tenzor::Variable(v_full, false),
        scale, /*causal=*/false);

    auto out = cc::dispatch_gqa({q, k, v}, "causal=false,scale=0");
    auto diff = ::tenzor::max(::tenzor::abs(eager.tensor() - out))
                    .item<float>();
    EXPECT_LT(diff, 1e-5f) << "GQA callback diverged by " << diff;
}

TEST(GQACallback, MatchesEager_MQA_Causal) {
    ensure_core_init();
    // Multi-Query Attention: H_kv = 1.
    const std::vector<int64_t> q_shape{1, 4, 8, 16};
    const std::vector<int64_t> kv_shape{1, 1, 8, 16};

    auto q = make_filled(q_shape,  0.0f, 0.4f);
    auto k = make_filled(kv_shape, 0.0f, 0.6f);
    auto v = make_filled(kv_shape, 0.0f, 0.8f);

    const float scale = 0.25f;
    auto k_full = repeat_kv(k, q_shape[1]);
    auto v_full = repeat_kv(v, q_shape[1]);
    auto eager = ::tenzor::flash_attention(
        ::tenzor::Variable(q,      false),
        ::tenzor::Variable(k_full, false),
        ::tenzor::Variable(v_full, false),
        scale, /*causal=*/true);

    auto out = cc::dispatch_gqa({q, k, v}, "causal=true,scale=0.25");
    auto diff = ::tenzor::max(::tenzor::abs(eager.tensor() - out))
                    .item<float>();
    EXPECT_LT(diff, 1e-5f) << "GQA(MQA causal) diverged by " << diff;
}

TEST(GQACallback, MHAPassthrough_HeadsEqual) {
    ensure_core_init();
    // When H_kv == H_q, the dispatcher must NOT introduce a broadcast — it
    // forwards to FlashAttention with the original KV tensors.
    const std::vector<int64_t> shape{1, 4, 8, 16};
    auto q = make_filled(shape, 0.0f, 0.3f);
    auto k = make_filled(shape, 0.1f, 0.5f);
    auto v = make_filled(shape, 0.2f, 0.7f);

    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.back()));
    auto eager = ::tenzor::flash_attention(
        ::tenzor::Variable(q, false),
        ::tenzor::Variable(k, false),
        ::tenzor::Variable(v, false),
        scale, /*causal=*/false);
    auto out_gqa = cc::dispatch_gqa({q, k, v}, "causal=false,scale=0");
    auto out_fa  = cc::dispatch_flash_attention({q, k, v}, "causal=false,scale=0");

    auto diff_eager = ::tenzor::max(::tenzor::abs(eager.tensor() - out_gqa))
                          .item<float>();
    auto diff_fa = ::tenzor::max(::tenzor::abs(out_fa - out_gqa)).item<float>();
    EXPECT_LT(diff_eager, 1e-5f);
    EXPECT_LT(diff_fa,    1e-5f);
}

TEST(GQACallback, RejectsBadHeadsRatio) {
    ensure_core_init();
    // H_q = 5, H_kv = 2: 5 % 2 != 0, must reject.
    auto q = ::tenzor::full({1, 5, 4, 8}, 0.1f, ::tenzor::DType::Float32);
    auto k = ::tenzor::full({1, 2, 4, 8}, 0.2f, ::tenzor::DType::Float32);
    auto v = ::tenzor::full({1, 2, 4, 8}, 0.3f, ::tenzor::DType::Float32);
    EXPECT_THROW(cc::dispatch_gqa({q, k, v}, "causal=false"),
                 std::runtime_error);
}
