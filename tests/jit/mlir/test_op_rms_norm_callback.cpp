// Phase 13 / Group D.4.2 — tenzor_rms_norm custom_call dispatcher.
//
// Dispatcher: parses `eps=<f>` from backend_config, materializes an
// all-ones weight when caller omitted it, and dispatches to
// OpId::RMSNorm. Validated against hand-computed references.

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

#include <gtest/gtest.h>

#include <cmath>
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

// ─── End-to-end plugin path: lower → compile → invoke in-process ──────────

namespace {
namespace tzj = ::tenzor::jit;
namespace tzm = ::tenzor::jit::mlir_jit;

auto make_tmp_cache_dir_rms() -> std::filesystem::path {
    auto now =
        std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(static_cast<std::uint64_t>(now));
    std::filesystem::path d = std::filesystem::temp_directory_path() /
        ("tenzor_jit_plugin_rms_" + std::to_string(rng()));
    std::filesystem::create_directories(d);
    return d;
}
}  // namespace

TEST(OpRMSNormCallback, EndToEndPluginPathMatchesEager) {
    ensure_core_init();
    const std::vector<int64_t> x_shape{2, 4, 32};
    const std::vector<int64_t> w_shape{32};

    auto x_t = ::tenzor::full(x_shape, 0.0f, ::tenzor::DType::Float32);
    auto w_t = ::tenzor::full(w_shape, 0.0f, ::tenzor::DType::Float32);
    {
        auto* xp = x_t.data<float>();
        auto* wp = w_t.data<float>();
        for (int64_t i = 0; i < x_t.numel(); ++i)
            xp[i] = 0.05f * static_cast<float>(i % 13) - 0.2f;
        for (int64_t i = 0; i < w_t.numel(); ++i)
            wp[i] = 0.7f + 0.01f * static_cast<float>(i);
    }
    const float eps = 1e-6f;

    auto eager_out = cc::dispatch_rms_norm({x_t, w_t},
                                            "eps=0.000001");

    // Build graph: RMSNorm(x, weight) {eps=1e-6}.
    tzj::Graph g;
    auto x_v = g.create_value("x", x_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    auto w_v = g.create_value("w", w_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    g.set_inputs({x_v, w_v});
    auto node = g.create_node(tzj::OpType::RMSNorm);
    node->add_input(x_v); node->add_input(w_v);
    auto out = g.create_value("o", x_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    node->add_output(out);
    node->set_attr("eps", eps);
    g.add_node(node);
    g.set_outputs({out});

    tzm::GraphToMLIR lowerer;
    lowerer.set_plugin_enabled(true);
    const std::string mlir = lowerer.lower(g);
    ASSERT_NE(mlir.find("call @tenzor_plugin.rms_norm"),
              std::string::npos) << mlir;

    tzm::CompileOptions opts;
    opts.target         = "llvm-cpu";
    opts.plugin_enabled = true;
    opts.cache_dir      = make_tmp_cache_dir_rms();
    auto artifact = tzm::compile_mlir(mlir, opts);
    auto invoker  = tzm::IreeInvoker::load(
        artifact, tzm::IreeInvoker::Mode::InProcess);

    auto outs = invoker->invoke({x_t, w_t});
    ASSERT_EQ(outs.size(), 1u);
    auto diff = ::tenzor::max(::tenzor::abs(eager_out - outs[0]))
                    .item<float>();
    EXPECT_LT(diff, 1e-5f)
        << "plugin-path RMSNorm diverged from eager-dispatcher by " << diff;

    std::error_code _ec;
    std::filesystem::remove_all(opts.cache_dir, _ec);
}
