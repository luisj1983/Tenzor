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
#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_customcalls.hpp"
#include "tenzor/jit/mlir/iree_runtime.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#include "tenzor/jit/tracer.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

#include <filesystem>
#include <random>

#include <gtest/gtest.h>

#include <cmath>

#include "mlir_target_util.hpp"

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

// ─── End-to-end plugin path: lower → compile → invoke in-process ──────────
// Validates the *full* MVP-1 Path A pipeline: a Graph containing a
// FlashAttention dialect node is lowered to `call @tenzor_plugin.flash_attention`
// MLIR text, compiled to .vmfb by iree-compile, and invoked through the
// in-process IreeInvoker. The plugin VM module satisfies the import at
// session load time, so the callback runs and the output must match eager.

namespace {

namespace tzj = ::tenzor::jit;
namespace tzm = ::tenzor::jit::mlir_jit;

auto make_tmp_cache_dir() -> std::filesystem::path {
    auto now =
        std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(static_cast<std::uint64_t>(now));
    std::filesystem::path d = std::filesystem::temp_directory_path() /
        ("tenzor_jit_plugin_fa_" + std::to_string(rng()));
    std::filesystem::create_directories(d);
    return d;
}

}  // namespace

TEST(OpFlashAttentionCallback, EndToEndPluginPathMatchesEager) {
    ensure_core_init();
    const std::vector<int64_t> shape{1, 2, 4, 16};

    auto q_t = make_filled(shape, 0.05f, 0.7f);
    auto k_t = make_filled(shape, 0.1f,  0.9f);
    auto v_t = make_filled(shape, 0.15f, 1.1f);

    // Eager reference. scale=0 -> autograd op picks 1/sqrt(D).
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.back()));
    auto eager_out = ::tenzor::flash_attention(
        ::tenzor::Variable(q_t, false),
        ::tenzor::Variable(k_t, false),
        ::tenzor::Variable(v_t, false),
        scale, /*causal=*/true);

    // Build graph: FlashAttention(Q,K,V) {causal=true}.
    tzj::Graph g;
    auto q = g.create_value("q", shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto k = g.create_value("k", shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto v = g.create_value("v", shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({q, k, v});
    auto node = g.create_node(tzj::OpType::FlashAttention);
    node->add_input(q); node->add_input(k); node->add_input(v);
    auto out = g.create_value("o", shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    node->add_output(out);
    node->set_bool_attr("causal", true);
    g.add_node(node);
    g.set_outputs({out});

    // Plugin-path lower → compile → in-process invoke.
    tzm::GraphToMLIR lowerer;
    lowerer.set_plugin_enabled(true);
    const std::string mlir = lowerer.lower(g);
    // Sanity: the call site went through the plugin path.
    ASSERT_NE(mlir.find("call @tenzor_plugin.flash_attention"),
              std::string::npos) << mlir;

    // JIT-R128: fan out over every available IREE target, mirroring the
    // expand-path sibling test_op_flash_attention_expand.cpp -- the plugin
    // callback itself always runs the computation on the host, but the
    // surrounding compiled module's HAL buffer marshaling is genuinely
    // target-specific, so this was previously zero-coverage on CUDA/ROCm/
    // Vulkan.
    namespace mt = ::tenzor::testing::mlir;
    mt::ensure_core_init();
    for (const auto& target : mt::available_iree_targets()) {
        tzm::CompileOptions opts;
        opts.target         = target;
        opts.plugin_enabled = true;
        opts.cache_dir      = make_tmp_cache_dir();
        auto artifact = tzm::compile_mlir(mlir, opts);
        std::unique_ptr<tzm::IreeInvoker> invoker;
        try {
            invoker = tzm::IreeInvoker::load(
                artifact, tzm::IreeInvoker::Mode::InProcess);
        } catch (const std::exception& e) {
            // See test_op_flash_attention_expand.cpp / test_op_rms_norm_
            // expand.cpp for why this is a runtime-availability fact of this
            // host, not a lowering defect.
            std::error_code _ec;
            std::filesystem::remove_all(opts.cache_dir, _ec);
            if (std::string(e.what()).find("no driver") != std::string::npos ||
                std::string(e.what()).find("NOT_FOUND") != std::string::npos) {
                continue;
            }
            throw;
        }

        auto outs = invoker->invoke({q_t, k_t, v_t});
        ASSERT_EQ(outs.size(), 1u) << "target=" << target;
        auto diff = ::tenzor::max(::tenzor::abs(eager_out.tensor() - outs[0]))
                        .item<float>();
        EXPECT_LT(diff, 1e-4f)
            << "plugin-path FA diverged from eager by " << diff << " on target=" << target;

        std::error_code _ec;
        std::filesystem::remove_all(opts.cache_dir, _ec);
    }
}
