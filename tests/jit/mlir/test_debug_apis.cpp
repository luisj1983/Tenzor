// Phase 13 / Group F — Debug UX tests for show_graph / show_mlir /
// show_stablehlo / show_iree / cache_stats + TENZOR_JIT_DUMP env var.
//
// These tests exercise the C++ side of the API surface used by
// python/tenzor/jit.py's show_* functions. The bindings layer is a thin
// wrapper; the heavy lifting (trace, lower, dump) lives in
// CompiledFunction::dump_*.

#include "tenzor/autograd/variable.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

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

// ---------------------------------------------------------------------------
// F.3 — show_stablehlo: pure-StableHLO output decomposes Tenzor custom_calls
// ---------------------------------------------------------------------------
//
// For an Add-only graph there are no @tenzor_* custom_calls in either the
// plugin-on or plugin-off form, so the pure-StableHLO assertion is that
// the expanded text NEVER contains "@tenzor_". We also confirm that the
// underlying stablehlo.add op is still present.

TEST(DebugAPIs, ShowStableHLODecomposesCustomCalls) {
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
    auto sh = compiled.dump_stablehlo(x);
    EXPECT_EQ(sh.find("@tenzor_"), std::string::npos)
        << "expanded StableHLO must not contain any @tenzor_* custom_calls\n"
        << sh;
    EXPECT_NE(sh.find("stablehlo.add"), std::string::npos) << sh;
}

// ---------------------------------------------------------------------------
// F.4 — show_iree captures the iree-compile pipeline dump
// ---------------------------------------------------------------------------

TEST(DebugAPIs, ShowIreeCapturesPipelineDump) {
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
    auto text = compiled.dump_iree(x);
    EXPECT_GT(text.size(), 100u);
    // iree-compile prints "// -----// IR Dump After XYZPass" between
    // pass dumps when --mlir-print-ir-after-all is enabled. Either the
    // "IR Dump" preamble or the "// -----//" pass separator is sufficient
    // evidence that the dump made it through the popen pipeline.
    const bool has_dump_header =
        text.find("IR Dump") != std::string::npos ||
        text.find("// -----//") != std::string::npos;
    EXPECT_TRUE(has_dump_header)
        << "expected --mlir-print-ir-after-all pipeline trace; first 200 "
        << "bytes were: " << text.substr(0, 200);
}

// ---------------------------------------------------------------------------
// F.5 — cache_stats counters increment on miss / hit and TENZOR_JIT_DUMP
// writes the full pipeline artifacts to disk.
// ---------------------------------------------------------------------------

TEST(DebugAPIs, CacheStatsIncrementOnHitAndMiss) {
    ensure_core_init();
    ::tenzor::jit::mlir_jit::reset_cache_stats();
    const auto s0 = ::tenzor::jit::mlir_jit::cache_stats();
    EXPECT_EQ(s0.hits, 0u);
    EXPECT_EQ(s0.misses, 0u);

    auto fn = [](const ::tenzor::Variable& x)
        -> ::tenzor::Variable { return x + x; };
    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = "llvm-cpu";
    ::tenzor::jit::CompiledFunction compiled(fn, cfg);

    auto x = ::tenzor::Variable(
        ::tenzor::full({4}, 1.5f, ::tenzor::DType::Float32),
        /*requires_grad=*/false);

    auto _r1 = compiled(x);
    const auto s1 = ::tenzor::jit::mlir_jit::cache_stats();
    EXPECT_EQ(s1.misses, 1u);
    EXPECT_EQ(s1.hits, 0u);
    // total_compile_ms may be 0 if the iree-compile vmfb cache hit on disk
    // (the cache is keyed by sha256(text)+target+revision), so we accept
    // any non-negative value. The hit/miss counters are the definitive
    // signal that the wiring is correct.
    EXPECT_GE(s1.total_compile_ms, 0.0);

    auto _r2 = compiled(x);
    const auto s2 = ::tenzor::jit::mlir_jit::cache_stats();
    EXPECT_GE(s2.hits, 1u);
}

TEST(DebugAPIs, JitDumpEnvWritesArtifactsToDir) {
    ensure_core_init();
    namespace fs = std::filesystem;

    const auto tmp = fs::temp_directory_path() / "tz-jit-dump-test";
    std::error_code ec;
    fs::remove_all(tmp, ec);
    fs::create_directories(tmp);

    ::setenv("TENZOR_JIT_DUMP", tmp.string().c_str(), 1);

    auto fn = [](const ::tenzor::Variable& x)
        -> ::tenzor::Variable { return x + x; };
    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = "llvm-cpu";
    ::tenzor::jit::CompiledFunction compiled(fn, cfg);

    auto x = ::tenzor::Variable(
        ::tenzor::full({8}, 2.25f, ::tenzor::DType::Float32),
        /*requires_grad=*/false);
    auto _trigger = compiled(x);

    ::unsetenv("TENZOR_JIT_DUMP");

    // tmp should now have at least one subdirectory containing the dump.
    bool found = false;
    for (const auto& entry : fs::directory_iterator(tmp)) {
        if (!fs::is_directory(entry.path())) continue;
        for (const auto* fname :
             {"graph.txt", "mlir.txt", "stablehlo.txt", "iree.log"}) {
            EXPECT_TRUE(fs::exists(entry.path() / fname))
                << "missing dump artifact: "
                << (entry.path() / fname).string();
        }
        found = true;
        break;
    }
    EXPECT_TRUE(found)
        << "no dump subdirectory was created under " << tmp.string();
}
