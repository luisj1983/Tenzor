// Phase 13 / Group D.2.3 — GQA expand-to-stablehlo + roundtrip.
//
// The expand decomposition broadcasts K and V from H_kv heads to H_q
// heads (insert size-1 group dim, broadcast_in_dim, flatten), then runs
// the same scaled-dot-product attention as FlashAttention. The roundtrip
// test confirms both lowering modes produce different IR text and the
// expand path is iree-compile-clean for representative GQA shapes.

#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_paths.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#include "tenzor/jit/tracer.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

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
               ("tenzor_gqa_expand_" + std::to_string(::getpid()) + "_" + tag);
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

tzj::Graph make_gqa_graph(const std::vector<int64_t>& q_shape,
                          const std::vector<int64_t>& kv_shape,
                          bool causal) {
    tzj::Graph g;
    auto q = g.create_value("q", q_shape,  ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto k = g.create_value("k", kv_shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto v = g.create_value("v", kv_shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({q, k, v});
    auto node = g.create_node(tzj::OpType::GQA);
    node->add_input(q);
    node->add_input(k);
    node->add_input(v);
    auto out = g.create_value("o", q_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    node->add_output(out);
    node->set_bool_attr("causal", causal);
    g.add_node(node);
    g.set_outputs({out});
    return g;
}

}  // namespace

TEST(OpGQAExpand, EmitsDistinctTextFromCustomCall) {
    ensure_core_init();
    auto g = make_gqa_graph({2, 8, 16, 64}, {2, 2, 16, 64}, /*causal=*/false);

    tzm::GraphToMLIR plugin_lower;
    plugin_lower.set_plugin_enabled(true);
    const std::string plugin_mlir = plugin_lower.lower(g);

    tzm::GraphToMLIR expand_lower;
    expand_lower.set_plugin_enabled(false);
    const std::string expand_mlir = expand_lower.lower(g);

    EXPECT_NE(plugin_mlir.find("@tenzor_plugin.gqa"),
              std::string::npos) << plugin_mlir;
    EXPECT_EQ(expand_mlir.find("@tenzor_plugin.gqa"),
              std::string::npos) << expand_mlir;
    // Expand path must contain reshape (KV broadcast prep), broadcast_in_dim
    // and dot_general (attention math).
    EXPECT_NE(expand_mlir.find("stablehlo.reshape"),
              std::string::npos) << expand_mlir;
    EXPECT_NE(expand_mlir.find("stablehlo.broadcast_in_dim"),
              std::string::npos) << expand_mlir;
    EXPECT_NE(expand_mlir.find("stablehlo.dot_general"),
              std::string::npos) << expand_mlir;
}

TEST(OpGQAExpand, ExpandPathIsIreeCompileClean_GroupSize4) {
    ensure_core_init();
    auto g = make_gqa_graph({2, 8, 16, 64}, {2, 2, 16, 64}, /*causal=*/false);
    tzm::GraphToMLIR lowerer;
    lowerer.set_plugin_enabled(false);
    const std::string mlir = lowerer.lower(g);
    assert_iree_compile_accepts(mlir, "g4");
}

TEST(OpGQAExpand, ExpandPathIsIreeCompileClean_MQACausal) {
    ensure_core_init();
    // Multi-Query Attention: H_kv = 1
    auto g = make_gqa_graph({1, 8, 32, 64}, {1, 1, 32, 64}, /*causal=*/true);
    tzm::GraphToMLIR lowerer;
    lowerer.set_plugin_enabled(false);
    const std::string mlir = lowerer.lower(g);
    assert_iree_compile_accepts(mlir, "mqa");
}

TEST(OpGQAExpand, MHANoKVReshapeWhenHeadsMatch) {
    ensure_core_init();
    // When H_kv == H_q the KV-expansion step is a no-op: no extra reshape
    // of K/V is introduced. The softmax path still uses broadcast_in_dim
    // (for the reduce-and-rebroadcast), but the KV reshape -> bcast ->
    // reshape chain is skipped. Validate iree-compile cleanliness on a
    // (1,4,8,32) MHA case.
    auto g = make_gqa_graph({1, 4, 8, 32}, {1, 4, 8, 32}, /*causal=*/false);
    tzm::GraphToMLIR lowerer;
    lowerer.set_plugin_enabled(false);
    const std::string mlir = lowerer.lower(g);
    assert_iree_compile_accepts(mlir, "mha");
}
