// Phase 13 / Group C.7 — Norm ops via the MLIR backend.
//
// Covered ops:
//   LayerNorm   → mean/var/scale/shift decomposition over stablehlo
//                 primitives (no fused stablehlo op)
//   BatchNorm2d → stablehlo.batch_norm_inference (inference path)
//
// Tenzor's nn::LayerNorm / nn::BatchNorm2d dispatch through fused-op
// IDs that aren't in the OpType mapping, so a trace doesn't surface
// OpType::LayerNorm / BatchNorm2d. These tests therefore build graphs
// by hand and validate the emitted MLIR via iree-compile.

#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#include "tenzor/jit/mlir/iree_paths.hpp"
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

void assert_iree_compile_accepts(const std::string& mlir_text) {
    auto tmp = std::filesystem::temp_directory_path() /
               ("tenzor_op_norms_" + std::to_string(::getpid()) + "_" +
                std::to_string(reinterpret_cast<std::uintptr_t>(&mlir_text)));
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
    EXPECT_EQ(rc, 0) << "iree-compile rejected emitted MLIR:\n" << mlir_text;
    std::filesystem::remove_all(tmp);
}

}  // namespace

TEST(OpNorms, LayerNormEmitsAndParses) {
    ensure_core_init();
    // x: (4, 8), normalize last 1 dim. weight & bias: (8,)
    tzj::Graph g;
    auto x = g.create_value("x", {4, 8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto w = g.create_value("w", {8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto b = g.create_value("b", {8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x, w, b});
    auto node = g.create_node(tzj::OpType::LayerNorm);
    node->add_input(x);
    node->add_input(w);
    node->add_input(b);
    auto z = g.create_value("z", {4, 8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    node->set_vec_attr("normalized_shape", {8});
    node->set_attr("eps", 1e-5f);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.reduce"), std::string::npos) << mlir;
    EXPECT_NE(mlir.find("stablehlo.sqrt"),   std::string::npos) << mlir;
    EXPECT_NE(mlir.find("stablehlo.divide"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpNorms, BatchNorm2dEmitsAndParses) {
    ensure_core_init();
    // x: (N=2, C=3, H=4, W=4); weight/bias/running_mean/running_var: (3,)
    tzj::Graph g;
    auto x  = g.create_value("x",  {2, 3, 4, 4}, ::tenzor::DType::Float32,
                             ::tenzor::Device::cpu());
    auto w  = g.create_value("w",  {3}, ::tenzor::DType::Float32,
                             ::tenzor::Device::cpu());
    auto b  = g.create_value("b",  {3}, ::tenzor::DType::Float32,
                             ::tenzor::Device::cpu());
    auto rm = g.create_value("rm", {3}, ::tenzor::DType::Float32,
                             ::tenzor::Device::cpu());
    auto rv = g.create_value("rv", {3}, ::tenzor::DType::Float32,
                             ::tenzor::Device::cpu());
    g.set_inputs({x, w, b, rm, rv});
    auto node = g.create_node(tzj::OpType::BatchNorm2d);
    node->add_input(x);
    node->add_input(w);
    node->add_input(b);
    node->add_input(rm);
    node->add_input(rv);
    auto z = g.create_value("z", {2, 3, 4, 4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    node->set_attr("eps", 1e-5f);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.batch_norm_inference"),
              std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}
