// Phase 13 / Task A.7 — IREE Compiler embedding API integration tests.

#include "tenzor/jit/mlir/iree_compile.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

namespace tj = ::tenzor::jit::mlir_jit;
namespace fs = std::filesystem;

namespace {

auto make_tmp_dir() -> fs::path {
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(static_cast<uint64_t>(now));
    fs::path dir =
        fs::temp_directory_path() /
        ("tenzor_jit_compile_test_" + std::to_string(rng()));
    fs::create_directories(dir);
    return dir;
}

constexpr const char* kTrivialModule = R"MLIR(module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
}
)MLIR";

}  // namespace

TEST(IreeCompile, CacheKey_Deterministic) {
    const std::string a = tj::compute_cache_key("module { }", "llvm-cpu");
    const std::string b = tj::compute_cache_key("module { }", "llvm-cpu");
    EXPECT_EQ(a, b);
    EXPECT_FALSE(a.empty());
    // SHA-256 hex digest is exactly 64 characters.
    EXPECT_EQ(a.size(), 64U);
}

TEST(IreeCompile, CacheKey_ChangesWithTarget) {
    const std::string a = tj::compute_cache_key("module { }", "llvm-cpu");
    const std::string b = tj::compute_cache_key("module { }", "cuda");
    EXPECT_NE(a, b);
}

TEST(IreeCompile, CacheKey_ChangesWithText) {
    const std::string a = tj::compute_cache_key("module { }", "llvm-cpu");
    const std::string b =
        tj::compute_cache_key("module { /* tweak */ }", "llvm-cpu");
    EXPECT_NE(a, b);
}

TEST(IreeCompile, CompilerVersion_NonEmpty) {
    const std::string v = tj::iree_compiler_version();
    EXPECT_FALSE(v.empty());
}

TEST(IreeCompile, TrivialIdentity_LlvmCpu) {
    const fs::path tmp = make_tmp_dir();
    tj::CompileOptions opts;
    opts.target = "llvm-cpu";
    opts.cache_dir = tmp;

    tj::CompiledArtifact artifact;
    ASSERT_NO_THROW({ artifact = tj::compile_mlir(kTrivialModule, opts); });

    EXPECT_TRUE(fs::exists(artifact.vmfb_path))
        << "vmfb not produced at " << artifact.vmfb_path;
    EXPECT_GT(fs::file_size(artifact.vmfb_path), 0U);
    EXPECT_EQ(artifact.target, "llvm-cpu");

    // Cleanup
    fs::remove_all(tmp);
}

TEST(IreeCompile, CacheHitReusesArtifact) {
    const fs::path tmp = make_tmp_dir();
    tj::CompileOptions opts;
    opts.target = "llvm-cpu";
    opts.cache_dir = tmp;

    const tj::CompiledArtifact first = tj::compile_mlir(kTrivialModule, opts);
    ASSERT_TRUE(fs::exists(first.vmfb_path));
    const auto first_mtime = fs::last_write_time(first.vmfb_path);

    // Second call should return the same path and NOT re-write the file.
    const tj::CompiledArtifact second = tj::compile_mlir(kTrivialModule, opts);
    EXPECT_EQ(first.vmfb_path, second.vmfb_path);
    EXPECT_EQ(first_mtime, fs::last_write_time(second.vmfb_path));

    fs::remove_all(tmp);
}

TEST(IreeCompile, ThrowsOnSyntaxError) {
    const fs::path tmp = make_tmp_dir();
    tj::CompileOptions opts;
    opts.target = "llvm-cpu";
    opts.cache_dir = tmp;

    const std::string bad = "this is not valid MLIR at all";
    try {
        (void)tj::compile_mlir(bad, opts);
        FAIL() << "Expected JitCompileError";
    } catch (const tj::JitCompileError& e) {
        // Dump path should exist and contain the bad source.
        EXPECT_TRUE(fs::exists(e.mlir_dump_path()));
        std::ifstream f(e.mlir_dump_path());
        std::ostringstream ss;
        ss << f.rdbuf();
        EXPECT_EQ(ss.str(), bad);
    }

    fs::remove_all(tmp);
}

TEST(IreeCompile, ThrowsOnEmptyTarget) {
    tj::CompileOptions opts;
    opts.target = "";
    EXPECT_THROW((void)tj::compile_mlir(kTrivialModule, opts),
                 tj::JitCompileError);
}
