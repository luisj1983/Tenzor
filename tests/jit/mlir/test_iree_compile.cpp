// Phase 13 / Task A.7 — IREE Compiler embedding API integration tests.

#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/backend/runtime_simd.hpp"

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

// JIT-R009 regression: the llvm-cpu cache key must fold in the actual
// detected host SIMD feature set, not just the bare "llvm-cpu" target
// string -- otherwise two machines with different CPU ISAs (one with
// AVX-512, one without) would alias to the SAME cache key even though
// --iree-llvmcpu-target-cpu-features=host bakes a DIFFERENT, incompatible
// ISA into each machine's .vmfb (SIGILL risk on a shared cache dir).
TEST(IreeCompile, CacheKey_LlvmCpuFoldsHostSimdFeatures) {
    const std::string bare = tj::compute_cache_key("module { }", "llvm-cpu");
    const std::string with_simd = tj::compute_cache_key(
        "module { }",
        std::string("llvm-cpu:") +
            ::tenzor::backend::get_simd_features().to_string());
    EXPECT_NE(bare, with_simd)
        << "llvm-cpu cache key is unaffected by host SIMD features";

    // End-to-end: the actual compiled artifact's vmfb filename must be the
    // SIMD-folded key, not the bare-target key.
    const fs::path tmp = make_tmp_dir();
    tj::CompileOptions opts;
    opts.target = "llvm-cpu";
    opts.cache_dir = tmp;
    const tj::CompiledArtifact artifact = tj::compile_mlir(kTrivialModule, opts);

    const std::string expected_key = tj::compute_cache_key(
        kTrivialModule,
        std::string("llvm-cpu:") +
            ::tenzor::backend::get_simd_features().to_string());
    EXPECT_EQ(artifact.vmfb_path.filename().string(), expected_key + ".vmfb");

    fs::remove_all(tmp);
}

// JIT-R008 regression: compiling an F16/BF16 graph for Vulkan with no
// resolvable target architecture must refuse (throw) rather than silently
// compile against IREE's conservative default SPIR-V env (which lacks
// shaderFloat16 and would compute garbage/NaN on-device with no error).
// This machine has no TENZOR_DEFAULT_VULKAN_TARGET build macro and no
// TENZOR_VULKAN_TARGET env override set, so vulkan_arch resolves empty --
// exactly the scenario the fix targets. The throw fires during arch
// resolution, before any real Vulkan device/runtime is touched, so this is
// testable without a physical Vulkan device.
TEST(IreeCompile, VulkanFloat16WithNoResolvedArchThrows) {
    ASSERT_EQ(std::getenv("TENZOR_VULKAN_TARGET"), nullptr)
        << "test assumes no TENZOR_VULKAN_TARGET override is set";
    constexpr const char* kF16Module = R"MLIR(module {
  func.func @main(%arg0: tensor<4xf16>) -> tensor<4xf16> {
    return %arg0 : tensor<4xf16>
  }
}
)MLIR";
    const fs::path tmp = make_tmp_dir();
    tj::CompileOptions opts;
    opts.target = "vulkan";
    opts.cache_dir = tmp;
    EXPECT_THROW(tj::compile_mlir(kF16Module, opts), std::runtime_error);
    fs::remove_all(tmp);
}

// Sanity: an F32-only Vulkan graph with no resolved arch must NOT be
// rejected by the new guard (only F16/BF16 graphs are affected) -- it will
// still fail further down for lack of a real Vulkan device/runtime in this
// test environment, but that failure must not be the R008 guard's
// std::runtime_error with the "shaderFloat16" message.
TEST(IreeCompile, VulkanFloat32WithNoResolvedArchDoesNotHitF16Guard) {
    ASSERT_EQ(std::getenv("TENZOR_VULKAN_TARGET"), nullptr)
        << "test assumes no TENZOR_VULKAN_TARGET override is set";
    const fs::path tmp = make_tmp_dir();
    tj::CompileOptions opts;
    opts.target = "vulkan";
    opts.cache_dir = tmp;
    try {
        tj::compile_mlir(kTrivialModule, opts);
    } catch (const std::exception& e) {
        EXPECT_EQ(std::string(e.what()).find("shaderFloat16"), std::string::npos)
            << "F32-only graph incorrectly hit the F16/BF16 arch guard: "
            << e.what();
    }
    fs::remove_all(tmp);
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
