// Phase 13 / Group D.1.3 — FlashAttention expand-to-stablehlo + roundtrip.
//
// Two lowering modes for OpType::FlashAttention:
//   plugin_enabled=true  -> stablehlo.custom_call @tenzor_flash_attention
//   plugin_enabled=false -> pure StableHLO primitives (dot_general → scale →
//                           [+ causal mask] → softmax → dot_general)
//
// Test asserts (a) both modes produce different MLIR text, (b) the expand
// path is accepted by iree-compile (so deploy targets without the Tenzor
// plugin can still build a vmfb), and (c) the result is invokable through
// iree-run-module subprocess and matches the eager flash_attention kernel
// to within 1e-3.

#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_paths.hpp"
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
#include <fstream>
#include <string>

namespace tzj = ::tenzor::jit;
namespace tzm = ::tenzor::jit::mlir_jit;

namespace {

auto ensure_core_init() -> void {
    static const bool inited = []() {
        ::tenzor::initialize();
        return true;
    }();
    (void)inited;
}

void assert_iree_compile_accepts(const std::string& mlir_text,
                                 const std::string& tag) {
    auto tmp = std::filesystem::temp_directory_path() /
               ("tenzor_fa_expand_" + std::to_string(::getpid()) + "_" + tag);
    std::filesystem::create_directories(tmp);
    auto in_path  = tmp / "mod.mlir";
    auto out_path = tmp / "mod.vmfb";
    { std::ofstream of(in_path); of << mlir_text; }
    const std::string& iree_compile =
        ::tenzor::jit::mlir_jit::resolve_iree_compile();
    std::string cmd = iree_compile +
        " --iree-hal-target-backends=llvm-cpu \"" + in_path.string() +
        "\" -o \"" + out_path.string() + "\" 2>&1";
    const int rc = std::system(cmd.c_str());
    EXPECT_EQ(rc, 0) << "iree-compile rejected emitted MLIR (" << tag << "):\n"
                     << mlir_text;
    std::filesystem::remove_all(tmp);
}

tzj::Graph make_fa_graph(const std::vector<int64_t>& shape, bool causal) {
    tzj::Graph g;
    auto q = g.create_value("q", shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto k = g.create_value("k", shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto v = g.create_value("v", shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({q, k, v});
    auto node = g.create_node(tzj::OpType::FlashAttention);
    node->add_input(q);
    node->add_input(k);
    node->add_input(v);
    auto out = g.create_value("o", shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    node->add_output(out);
    node->set_bool_attr("causal", causal);
    // scale=0 means the expand path picks 1/sqrt(D); leaving it unset is
    // exactly equivalent to scale=0.0f under get_attr_float.
    g.add_node(node);
    g.set_outputs({out});
    return g;
}

}  // namespace

TEST(OpFlashAttentionExpand, EmitsDistinctTextFromCustomCall) {
    ensure_core_init();
    auto g = make_fa_graph({2, 4, 16, 64}, /*causal=*/false);

    tzm::GraphToMLIR plugin_lower;
    plugin_lower.set_plugin_enabled(true);
    const std::string plugin_mlir = plugin_lower.lower(g);

    tzm::GraphToMLIR expand_lower;
    expand_lower.set_plugin_enabled(false);
    const std::string expand_mlir = expand_lower.lower(g);

    EXPECT_NE(plugin_mlir.find("@tenzor_flash_attention"),
              std::string::npos) << plugin_mlir;
    EXPECT_EQ(expand_mlir.find("@tenzor_flash_attention"),
              std::string::npos) << expand_mlir;
    // The expand path must contain the canonical attention primitives.
    EXPECT_NE(expand_mlir.find("stablehlo.dot_general"),
              std::string::npos) << expand_mlir;
    EXPECT_NE(expand_mlir.find("stablehlo.exponential"),
              std::string::npos) << expand_mlir;
}

TEST(OpFlashAttentionExpand, ExpandPathIsIreeCompileClean_NonCausal) {
    ensure_core_init();
    auto g = make_fa_graph({2, 4, 16, 64}, /*causal=*/false);
    tzm::GraphToMLIR lowerer;
    lowerer.set_plugin_enabled(false);
    const std::string mlir = lowerer.lower(g);
    assert_iree_compile_accepts(mlir, "noncausal");
}

TEST(OpFlashAttentionExpand, ExpandPathIsIreeCompileClean_Causal) {
    ensure_core_init();
    auto g = make_fa_graph({2, 4, 16, 64}, /*causal=*/true);
    tzm::GraphToMLIR lowerer;
    lowerer.set_plugin_enabled(false);
    const std::string mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.iota"),    std::string::npos) << mlir;
    EXPECT_NE(mlir.find("stablehlo.compare"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir, "causal");
}

TEST(OpFlashAttentionExpand, ExpandResultMatchesEager_NonCausal) {
    ensure_core_init();
    const std::vector<int64_t> shape{1, 2, 4, 16};

    // Build deterministic Q, K, V.
    auto q_t = ::tenzor::full(shape, 0.1f, ::tenzor::DType::Float32);
    auto k_t = ::tenzor::full(shape, 0.2f, ::tenzor::DType::Float32);
    auto v_t = ::tenzor::full(shape, 0.3f, ::tenzor::DType::Float32);
    // Add small per-element variation so softmax has signal.
    {
        auto* qp = q_t.data<float>();
        auto* kp = k_t.data<float>();
        auto* vp = v_t.data<float>();
        const auto n = q_t.numel();
        for (int64_t i = 0; i < n; ++i) {
            qp[i] += 0.001f * static_cast<float>(i % 7);
            kp[i] += 0.001f * static_cast<float>((i * 3) % 11);
            vp[i] += 0.001f * static_cast<float>((i * 5) % 13);
        }
    }

    // Eager reference via autograd::flash_attention.
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.back()));
    auto eager_out = ::tenzor::flash_attention(
        ::tenzor::Variable(q_t, false),
        ::tenzor::Variable(k_t, false),
        ::tenzor::Variable(v_t, false),
        scale, /*causal=*/false);

    // JIT path: build graph, lower with expand, compile, run.
    auto g = make_fa_graph(shape, /*causal=*/false);
    tzm::GraphToMLIR lowerer;
    lowerer.set_plugin_enabled(false);
    const std::string mlir = lowerer.lower(g);

    tzm::CompileOptions opts;
    opts.target         = "llvm-cpu";
    opts.plugin_enabled = false;
    auto artifact = tzm::compile_mlir(mlir, opts);
    auto invoker  = tzm::IreeInvoker::load(artifact);
    auto outs     = invoker->invoke({q_t, k_t, v_t});
    ASSERT_EQ(outs.size(), 1u);

    auto diff = ::tenzor::max(::tenzor::abs(eager_out.tensor() - outs[0]))
                    .item<float>();
    EXPECT_LT(diff, 1e-3f)
        << "FlashAttention expand path diverged from eager by " << diff;
}
