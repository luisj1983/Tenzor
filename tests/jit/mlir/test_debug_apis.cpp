// Phase 13 / Group F — Debug UX tests for show_graph / show_mlir /
// show_stablehlo / show_iree / cache_stats + TENZOR_JIT_DUMP env var.
//
// These tests exercise the C++ side of the API surface used by
// python/tenzor/jit.py's show_* functions. The bindings layer is a thin
// wrapper; the heavy lifting (trace, lower, dump) lives in
// CompiledFunction::dump_*.

#include "tenzor/autograd/variable.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include <string>

namespace {

auto ensure_core_init() -> void {
    static const bool inited = []() {
        ::tenzor::initialize();
        return true;
    }();
    (void)inited;
}

}  // namespace

// ---------------------------------------------------------------------------
// F.1 — show_graph dumps the Graph IR
// ---------------------------------------------------------------------------

TEST(DebugAPIs, ShowGraphDumpsAddNode) {
    ensure_core_init();
    auto fn = [](const ::tenzor::Variable& x)
        -> ::tenzor::Variable { return x + x; };
    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = "llvm-cpu";
    ::tenzor::jit::CompiledFunction compiled(fn, cfg);

    auto x = ::tenzor::Variable(
        ::tenzor::full({4}, 1.5f, ::tenzor::DType::Float32),
        /*requires_grad=*/false);

    // Force at least one invocation so the user-facing path mirrors what
    // python/tenzor/jit.py does on first call.
    auto _trigger = compiled(x);
    auto text = compiled.dump_graph(x);
    EXPECT_NE(text.find("Add"), std::string::npos) << text;
}

// ---------------------------------------------------------------------------
// F.2 — show_mlir contains stablehlo.add and func.func @main
// ---------------------------------------------------------------------------

TEST(DebugAPIs, ShowMlirContainsStablehlo) {
    ensure_core_init();
    auto fn = [](const ::tenzor::Variable& x)
        -> ::tenzor::Variable { return x + x; };
    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = "llvm-cpu";
    ::tenzor::jit::CompiledFunction compiled(fn, cfg);

    auto x = ::tenzor::Variable(
        ::tenzor::full({4}, 1.5f, ::tenzor::DType::Float32),
        /*requires_grad=*/false);

    auto _trigger = compiled(x);
    auto text = compiled.dump_mlir(x);
    EXPECT_NE(text.find("stablehlo.add"), std::string::npos) << text;
    EXPECT_NE(text.find("func.func @main"), std::string::npos);
}
