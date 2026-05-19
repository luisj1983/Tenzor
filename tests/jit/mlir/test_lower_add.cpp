// Phase 13 / Task B.1 — Graph → MLIR text lowering tests for Add.
//
// These tests cover two layers:
//   1. Static string assertions: the emitted text contains the canonical
//      `stablehlo.add` op, `func.func @main`, and the `module {` wrapper.
//   2. Round-trip through `iree-compile` to prove the emitted MLIR is
//      actually syntactically valid (caught early by an external parser
//      rather than by waiting for the runtime test in B.2).

#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#include "tenzor/jit/tracer.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

namespace tzm = ::tenzor::jit::mlir_jit;
namespace tzj = ::tenzor::jit;
namespace fs = std::filesystem;

namespace {

auto ensure_core_init() -> void {
    static const bool inited = []() {
        ::tenzor::initialize();
        return true;
    }();
    (void)inited;
}

auto build_add_graph() -> tzj::Graph {
    tzj::Graph g;
    auto x = g.create_value("x", {4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto y = g.create_value("y", {4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x, y});

    auto node = g.create_node(tzj::OpType::Add, "add");
    node->add_input(x);
    node->add_input(y);
    auto z = g.create_value("z", {4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    g.add_node(node);

    g.set_outputs({z});
    return g;
}

auto make_tmp_dir(const std::string& tag) -> fs::path {
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(static_cast<uint64_t>(now));
    fs::path dir = fs::temp_directory_path() /
                   ("tenzor_lower_add_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(dir);
    return dir;
}

}  // namespace

TEST(LowerAdd, EmitsStableHLOAdd) {
    ensure_core_init();
    auto g = build_add_graph();
    tzm::GraphToMLIR lowerer;
    const std::string mlir_text = lowerer.lower(g);

    EXPECT_NE(mlir_text.find("stablehlo.add"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("func.func @main"), std::string::npos)
        << mlir_text;
    EXPECT_NE(mlir_text.find("module {"), std::string::npos) << mlir_text;
    EXPECT_NE(mlir_text.find("tensor<4xf32>"), std::string::npos) << mlir_text;
}

TEST(LowerAdd, UnsupportedOpThrowsWithName) {
    ensure_core_init();
    tzj::Graph g;
    auto x = g.create_value("x", {4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(tzj::OpType::Mul, "mul");
    node->add_input(x);
    node->add_input(x);
    auto z = g.create_value("z", {4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    g.add_node(node);
    g.set_outputs({z});

    tzm::GraphToMLIR lowerer;
    try {
        (void)lowerer.lower(g);
        FAIL() << "expected std::runtime_error for unsupported OpType";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("Mul"), std::string::npos) << msg;
        EXPECT_NE(msg.find("not yet supported"), std::string::npos) << msg;
    }
}

TEST(LowerAdd, EmittedMLIRParsesWithIreeCompile) {
    // Validates that the text we produce is actually accepted by the
    // upstream iree-compile binary on the llvm-cpu target. This catches
    // syntactic regressions in the emit helpers before any runtime test.
    ensure_core_init();
    auto g = build_add_graph();
    tzm::GraphToMLIR lowerer;
    const std::string mlir_text = lowerer.lower(g);

    const fs::path tmp = make_tmp_dir("ireeparse");
    const fs::path in_path  = tmp / "add.mlir";
    const fs::path out_path = tmp / "add.vmfb";
    {
        std::ofstream of(in_path);
        of << mlir_text;
    }

    // Resolve iree-compile path: env override → PATH.
    std::string iree_compile = "iree-compile";
    if (const char* env = std::getenv("TENZOR_IREE_COMPILE"); env && *env) {
        iree_compile = env;
    }

    std::ostringstream cmd;
    cmd << iree_compile
        << " --iree-hal-target-backends=llvm-cpu "
        << "\"" << in_path.string() << "\" "
        << "-o \"" << out_path.string() << "\" 2>&1";
    const std::string cmd_str = cmd.str();

    const int rc = std::system(cmd_str.c_str());
    EXPECT_EQ(rc, 0)
        << "iree-compile rejected the emitted MLIR. cmd=" << cmd_str
        << "\n--- emitted MLIR ---\n"
        << mlir_text;
    EXPECT_TRUE(fs::exists(out_path)) << out_path;

    fs::remove_all(tmp);
}
