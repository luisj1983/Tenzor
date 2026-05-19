// Phase 13 / Group D.3.3 — RoPE expand-to-stablehlo + roundtrip.
//
// Expand decomposition: out = x * cos + rotate_half(x) * sin where
// rotate_half(x) = concat([-x[..., D/2:], x[..., :D/2]], dim=-1). The
// cos/sin tables (shape (S, D)) are broadcast across batch + head dims
// to (B, H, S, D). The integer `offset` attr is absorbed into the table
// values by the eager precomputation, so the expand path does not need
// to handle it explicitly.

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
               ("tenzor_rope_expand_" + std::to_string(::getpid()) + "_" + tag);
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

tzj::Graph make_rope_graph(const std::vector<int64_t>& x_shape,
                           const std::vector<int64_t>& tab_shape) {
    tzj::Graph g;
    auto x   = g.create_value("x",   x_shape,   ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    auto cos = g.create_value("cos", tab_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    auto sin = g.create_value("sin", tab_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    g.set_inputs({x, cos, sin});
    auto node = g.create_node(tzj::OpType::RoPE);
    node->add_input(x);
    node->add_input(cos);
    node->add_input(sin);
    auto out = g.create_value("y", x_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    node->add_output(out);
    g.add_node(node);
    g.set_outputs({out});
    return g;
}

}  // namespace

TEST(OpRoPEExpand, EmitsDistinctTextFromCustomCall) {
    ensure_core_init();
    auto g = make_rope_graph({2, 4, 16, 64}, {16, 64});

    tzm::GraphToMLIR plugin_lower;
    plugin_lower.set_plugin_enabled(true);
    const std::string plugin_mlir = plugin_lower.lower(g);

    tzm::GraphToMLIR expand_lower;
    expand_lower.set_plugin_enabled(false);
    const std::string expand_mlir = expand_lower.lower(g);

    EXPECT_NE(plugin_mlir.find("@tenzor_rope_apply"),
              std::string::npos) << plugin_mlir;
    EXPECT_EQ(expand_mlir.find("@tenzor_rope_apply"),
              std::string::npos) << expand_mlir;
    // Expand path must contain the rotate-half pattern: slice + negate
    // + concatenate.
    EXPECT_NE(expand_mlir.find("stablehlo.slice"),
              std::string::npos) << expand_mlir;
    EXPECT_NE(expand_mlir.find("stablehlo.negate"),
              std::string::npos) << expand_mlir;
    EXPECT_NE(expand_mlir.find("stablehlo.concatenate"),
              std::string::npos) << expand_mlir;
}

TEST(OpRoPEExpand, ExpandPathIsIreeCompileClean) {
    ensure_core_init();
    auto g = make_rope_graph({2, 4, 16, 64}, {16, 64});
    tzm::GraphToMLIR lowerer;
    lowerer.set_plugin_enabled(false);
    const std::string mlir = lowerer.lower(g);
    assert_iree_compile_accepts(mlir, "default");
}

TEST(OpRoPEExpand, ExpandResultMatchesHandComputed) {
    ensure_core_init();
    // x: (B=1, H=1, S=2, D=4). cos/sin: (S=2, D=4).
    // x = [[[1, 2, 3, 4], [5, 6, 7, 8]]]  (one batch, one head)
    // cos = [[1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5]]
    // sin = [[0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]]
    // rotate_half(x) at pos 0 = [-3, -4, 1, 2]; at pos 1 = [-7, -8, 5, 6]
    // out[0] = x[0]*cos[0] + rh[0]*sin[0] = x[0]
    // out[1] = x[1]*cos[1] + rh[1]*sin[1]
    //        = [2.5, 3, 3.5, 4] + [-3.5, -4, 2.5, 3]
    //        = [-1, -1, 6, 7]
    const std::vector<int64_t> x_shape{1, 1, 2, 4};
    const std::vector<int64_t> tab_shape{2, 4};

    auto x_t = ::tenzor::full(x_shape, 0.f, ::tenzor::DType::Float32);
    {
        auto* p = x_t.data<float>();
        for (int64_t i = 0; i < 8; ++i) p[i] = static_cast<float>(i + 1);
    }
    auto cos_t = ::tenzor::full(tab_shape, 0.f, ::tenzor::DType::Float32);
    auto sin_t = ::tenzor::full(tab_shape, 0.f, ::tenzor::DType::Float32);
    {
        auto* cp = cos_t.data<float>();
        auto* sp = sin_t.data<float>();
        for (int64_t i = 0; i < 4; ++i) { cp[i]     = 1.0f; sp[i]     = 0.0f; }
        for (int64_t i = 0; i < 4; ++i) { cp[4 + i] = 0.5f; sp[4 + i] = 0.5f; }
    }

    auto g = make_rope_graph(x_shape, tab_shape);
    tzm::GraphToMLIR lowerer;
    lowerer.set_plugin_enabled(false);
    const std::string mlir = lowerer.lower(g);

    tzm::CompileOptions opts;
    opts.target         = "llvm-cpu";
    opts.plugin_enabled = false;
    auto artifact = tzm::compile_mlir(mlir, opts);
    auto invoker  = tzm::IreeInvoker::load(artifact);
    auto outs     = invoker->invoke({x_t, cos_t, sin_t});
    ASSERT_EQ(outs.size(), 1u);
    const float* op = outs[0].data<float>();

    // Position 0: out == x  (cos=1, sin=0)
    EXPECT_NEAR(op[0], 1.0f, 1e-5f);
    EXPECT_NEAR(op[1], 2.0f, 1e-5f);
    EXPECT_NEAR(op[2], 3.0f, 1e-5f);
    EXPECT_NEAR(op[3], 4.0f, 1e-5f);
    // Position 1: out = x*0.5 + rotate_half(x)*0.5
    //   x[1] = [5,6,7,8]; rotate_half = [-7,-8,5,6]
    //   out = [2.5-3.5, 3-4, 3.5+2.5, 4+3] = [-1, -1, 6, 7]
    EXPECT_NEAR(op[4], -1.0f, 1e-5f);
    EXPECT_NEAR(op[5], -1.0f, 1e-5f);
    EXPECT_NEAR(op[6],  6.0f, 1e-5f);
    EXPECT_NEAR(op[7],  7.0f, 1e-5f);
}
