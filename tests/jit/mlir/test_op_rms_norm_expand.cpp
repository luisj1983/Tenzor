// Phase 13 / Group D.4.3 — RMSNorm expand-to-stablehlo + roundtrip.
//
// Expand decomposition: out = x / sqrt(mean(x^2, dim=-1) + eps) * weight.
// Reuses stablehlo.reduce (sum), stablehlo.divide / multiply / add / sqrt
// primitives. Validated against a hand-computed reference.

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
               ("tenzor_rmsn_expand_" + std::to_string(::getpid()) + "_" + tag);
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

tzj::Graph make_rms_graph_with_weight(const std::vector<int64_t>& x_shape,
                                      const std::vector<int64_t>& w_shape,
                                      float eps) {
    tzj::Graph g;
    auto x = g.create_value("x", x_shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto w = g.create_value("w", w_shape, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x, w});
    auto node = g.create_node(tzj::OpType::RMSNorm);
    node->add_input(x);
    node->add_input(w);
    auto out = g.create_value("y", x_shape, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    node->add_output(out);
    node->set_attr("eps", eps);
    g.add_node(node);
    g.set_outputs({out});
    return g;
}

}  // namespace

TEST(OpRMSNormExpand, EmitsDistinctTextFromCustomCall) {
    ensure_core_init();
    auto g = make_rms_graph_with_weight({2, 16, 768}, {768}, 1e-6f);

    tzm::GraphToMLIR plugin_lower;
    plugin_lower.set_plugin_enabled(true);
    const std::string plugin_mlir = plugin_lower.lower(g);

    tzm::GraphToMLIR expand_lower;
    expand_lower.set_plugin_enabled(false);
    const std::string expand_mlir = expand_lower.lower(g);

    EXPECT_NE(plugin_mlir.find("@tenzor_rms_norm"),
              std::string::npos) << plugin_mlir;
    EXPECT_EQ(expand_mlir.find("@tenzor_rms_norm"),
              std::string::npos) << expand_mlir;
    EXPECT_NE(expand_mlir.find("stablehlo.reduce"),
              std::string::npos) << expand_mlir;
    EXPECT_NE(expand_mlir.find("stablehlo.sqrt"),
              std::string::npos) << expand_mlir;
    EXPECT_NE(expand_mlir.find("stablehlo.divide"),
              std::string::npos) << expand_mlir;
}

TEST(OpRMSNormExpand, ExpandPathIsIreeCompileClean) {
    ensure_core_init();
    auto g = make_rms_graph_with_weight({2, 16, 768}, {768}, 1e-6f);
    tzm::GraphToMLIR lowerer;
    lowerer.set_plugin_enabled(false);
    const std::string mlir = lowerer.lower(g);
    assert_iree_compile_accepts(mlir, "default");
}

TEST(OpRMSNormExpand, ExpandResultMatchesHandComputed) {
    ensure_core_init();
    // x: (1, 1, 4) = [3, 4, 0, 0]; weight = [1, 1, 1, 1]; eps = 0
    // mean(x^2) = (9+16+0+0)/4 = 6.25; rms = 2.5
    // out = x / 2.5 * 1 = [1.2, 1.6, 0, 0]
    const std::vector<int64_t> x_shape{1, 1, 4};
    const std::vector<int64_t> w_shape{4};

    auto x_t = ::tenzor::full(x_shape, 0.0f, ::tenzor::DType::Float32);
    auto w_t = ::tenzor::full(w_shape, 1.0f, ::tenzor::DType::Float32);
    {
        auto* p = x_t.data<float>();
        p[0] = 3.0f; p[1] = 4.0f; p[2] = 0.0f; p[3] = 0.0f;
    }

    auto g = make_rms_graph_with_weight(x_shape, w_shape, /*eps=*/0.0f);
    tzm::GraphToMLIR lowerer;
    lowerer.set_plugin_enabled(false);
    const std::string mlir = lowerer.lower(g);

    tzm::CompileOptions opts;
    opts.target         = "llvm-cpu";
    opts.plugin_enabled = false;
    auto artifact = tzm::compile_mlir(mlir, opts);
    auto invoker  = tzm::IreeInvoker::load(artifact);
    auto outs     = invoker->invoke({x_t, w_t});
    ASSERT_EQ(outs.size(), 1u);
    const float* op = outs[0].data<float>();
    EXPECT_NEAR(op[0], 1.2f, 1e-5f);
    EXPECT_NEAR(op[1], 1.6f, 1e-5f);
    EXPECT_NEAR(op[2], 0.0f, 1e-5f);
    EXPECT_NEAR(op[3], 0.0f, 1e-5f);
}

TEST(OpRMSNormExpand, WithNonUnitWeight) {
    ensure_core_init();
    // x = [1, 2, 3, 4]; weight = [2, 2, 2, 2]; eps = 0
    // mean(x^2) = 30/4 = 7.5; rms = sqrt(7.5)
    // out = x / sqrt(7.5) * 2
    const std::vector<int64_t> x_shape{1, 1, 4};
    const std::vector<int64_t> w_shape{4};

    auto x_t = ::tenzor::full(x_shape, 0.0f, ::tenzor::DType::Float32);
    auto w_t = ::tenzor::full(w_shape, 2.0f, ::tenzor::DType::Float32);
    {
        auto* p = x_t.data<float>();
        p[0] = 1.0f; p[1] = 2.0f; p[2] = 3.0f; p[3] = 4.0f;
    }
    auto g = make_rms_graph_with_weight(x_shape, w_shape, 0.0f);
    tzm::GraphToMLIR lowerer;
    lowerer.set_plugin_enabled(false);
    const std::string mlir = lowerer.lower(g);
    tzm::CompileOptions opts;
    opts.target         = "llvm-cpu";
    opts.plugin_enabled = false;
    auto artifact = tzm::compile_mlir(mlir, opts);
    auto invoker  = tzm::IreeInvoker::load(artifact);
    auto outs     = invoker->invoke({x_t, w_t});
    ASSERT_EQ(outs.size(), 1u);
    const float* op = outs[0].data<float>();
    const float rms = std::sqrt(7.5f);
    EXPECT_NEAR(op[0], (1.0f / rms) * 2.0f, 1e-5f);
    EXPECT_NEAR(op[1], (2.0f / rms) * 2.0f, 1e-5f);
    EXPECT_NEAR(op[2], (3.0f / rms) * 2.0f, 1e-5f);
    EXPECT_NEAR(op[3], (4.0f / rms) * 2.0f, 1e-5f);
}
