// Phase 13 / Group D.3.2 — tenzor_rope_apply custom_call dispatcher.
//
// Dispatcher: out = x * cos + rotate_half(x) * sin, computed with raw
// Tensor ops (narrow + neg + cat + multiply + add). The `offset` int attr
// is parsed but not used — the eager precomputation bakes the offset into
// the cos/sin tables.

#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_customcalls.hpp"
#include "tenzor/jit/mlir/iree_runtime.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#include "tenzor/jit/tracer.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>
#include <filesystem>
#include <random>

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

// JIT-R042 regression: dispatch_rope_apply (the plugin/custom-call path's
// C++ reimplementation, distinct from JIT-R007's MLIR expand-path lowering)
// was computing the rotation entirely in storage dtype instead of widening
// F16/BF16 to F32 like eager's rope.cpp ("F121") does. Same hand-computed
// case as MatchesHandComputed_Position1WithRotation above, run in Float16 —
// asserts (a) the OUTPUT dtype is still Float16 (narrowed back, not left at
// the widened F32), and (b) the widen/narrow round-trip doesn't drift beyond
// F16 precision.
TEST(RoPECallback, MatchesHandComputed_Position1WithRotation_Float16Widens) {
    ensure_core_init();
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
    auto x16   = x.to(::tenzor::DType::Float16);
    auto cos16 = cos.to(::tenzor::DType::Float16);
    auto sin16 = sin.to(::tenzor::DType::Float16);

    auto out = cc::dispatch_rope_apply({x16, cos16, sin16}, "offset=0");
    ASSERT_EQ(out.dtype(), ::tenzor::DType::Float16)
        << "dispatch_rope_apply must narrow the widened F32 compute back to "
           "the input's Float16 dtype";
    auto out32 = out.to(::tenzor::DType::Float32);
    const float* op = out32.data<float>();
    EXPECT_NEAR(op[0],  1.0f, 1e-2f);
    EXPECT_NEAR(op[1],  2.0f, 1e-2f);
    EXPECT_NEAR(op[2],  3.0f, 1e-2f);
    EXPECT_NEAR(op[3],  4.0f, 1e-2f);
    EXPECT_NEAR(op[4], -1.0f, 1e-2f);
    EXPECT_NEAR(op[5], -1.0f, 1e-2f);
    EXPECT_NEAR(op[6],  6.0f, 1e-2f);
    EXPECT_NEAR(op[7],  7.0f, 1e-2f);
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

// ─── End-to-end plugin path: lower → compile → invoke in-process ──────────

namespace {
namespace tzj = ::tenzor::jit;
namespace tzm = ::tenzor::jit::mlir_jit;

auto make_tmp_cache_dir_rope() -> std::filesystem::path {
    auto now =
        std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(static_cast<std::uint64_t>(now));
    std::filesystem::path d = std::filesystem::temp_directory_path() /
        ("tenzor_jit_plugin_rope_" + std::to_string(rng()));
    std::filesystem::create_directories(d);
    return d;
}
}  // namespace

TEST(OpRoPECallback, EndToEndPluginPathMatchesEager) {
    ensure_core_init();
    const std::vector<int64_t> x_shape{1, 2, 4, 16};
    // cos/sin are broadcast to x's leading dims; emit them at (S,D).
    const std::vector<int64_t> tab_shape{4, 16};

    auto x_t   = ::tenzor::full(x_shape,   0.0f, ::tenzor::DType::Float32);
    auto cos_t = ::tenzor::full(tab_shape, 0.0f, ::tenzor::DType::Float32);
    auto sin_t = ::tenzor::full(tab_shape, 0.0f, ::tenzor::DType::Float32);
    {
        auto* xp = x_t.data<float>();
        auto* cp = cos_t.data<float>();
        auto* sp = sin_t.data<float>();
        for (int64_t i = 0; i < x_t.numel(); ++i)
            xp[i] = 0.01f * static_cast<float>(i % 19);
        for (int64_t i = 0; i < cos_t.numel(); ++i) {
            cp[i] = std::cos(0.07f * static_cast<float>(i));
            sp[i] = std::sin(0.07f * static_cast<float>(i));
        }
    }

    // Eager reference via dispatcher (already validated against the formula
    // in MatchesHandComputed_*).
    auto eager_out = cc::dispatch_rope_apply({x_t, cos_t, sin_t}, "");

    // Build graph: RoPE(x, cos, sin).
    tzj::Graph g;
    auto x_v   = g.create_value("x",   x_shape,   ::tenzor::DType::Float32,
                                ::tenzor::Device::cpu());
    auto cos_v = g.create_value("cos", tab_shape, ::tenzor::DType::Float32,
                                ::tenzor::Device::cpu());
    auto sin_v = g.create_value("sin", tab_shape, ::tenzor::DType::Float32,
                                ::tenzor::Device::cpu());
    g.set_inputs({x_v, cos_v, sin_v});
    auto node = g.create_node(tzj::OpType::RoPE);
    node->add_input(x_v); node->add_input(cos_v); node->add_input(sin_v);
    auto out = g.create_value("o", x_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    node->add_output(out);
    g.add_node(node);
    g.set_outputs({out});

    tzm::GraphToMLIR lowerer;
    lowerer.set_plugin_enabled(true);
    const std::string mlir = lowerer.lower(g);
    ASSERT_NE(mlir.find("call @tenzor_plugin.rope_apply"),
              std::string::npos) << mlir;

    tzm::CompileOptions opts;
    opts.target         = "llvm-cpu";
    opts.plugin_enabled = true;
    opts.cache_dir      = make_tmp_cache_dir_rope();
    auto artifact = tzm::compile_mlir(mlir, opts);
    auto invoker  = tzm::IreeInvoker::load(
        artifact, tzm::IreeInvoker::Mode::InProcess);

    auto outs = invoker->invoke({x_t, cos_t, sin_t});
    ASSERT_EQ(outs.size(), 1u);
    auto diff = ::tenzor::max(::tenzor::abs(eager_out - outs[0]))
                    .item<float>();
    EXPECT_LT(diff, 1e-5f)
        << "plugin-path RoPE diverged from eager-dispatcher by " << diff;

    std::error_code _ec;
    std::filesystem::remove_all(opts.cache_dir, _ec);
}
