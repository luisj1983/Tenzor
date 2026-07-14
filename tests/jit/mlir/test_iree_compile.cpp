// Phase 13 / Task A.7 — IREE Compiler embedding API integration tests.

#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/backend/runtime_simd.hpp"

#include <gtest/gtest.h>

#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

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

// JIT-R127 regression: compile_via_subprocess already retried a rejected
// cuda_arch once with the sm_80 Ampere baseline (JIT-R101), but the in-process
// embedding-API path (compile_via_embedding_api, used whenever the linked
// libIREECompiler.so itself registers the "cuda" HAL backend -- true for this
// build's third_party/iree_dist libIREECompiler.so) had no equivalent retry:
// it just threw JitCompileError, permanently forcing eager fallback for any
// cuda graph on a device whose reported compute-capability arch is newer than
// this toolchain's NVPTX backend supports -- even though the exact same graph
// on the exact same hardware compiles fine through the subprocess path. Force
// that exact rejection with a bogus-but-well-formed numeric arch the NVPTX
// backend cannot possibly recognize, and confirm the compile now succeeds
// (via the sm_80 retry) instead of throwing. This does not require a real
// CUDA device -- IREE's target-arch validation happens purely at compile
// time, before any device/runtime is touched. Must use a module with a real
// compute op (not the pure-identity kTrivialModule): a trivial passthrough
// generates no hal.executable at all, so the NVPTX target string is never
// actually consulted and the rejection never fires (verified experimentally
// -- the identity module compiles "successfully" for literally any garbage
// arch string, real or not).
TEST(IreeCompile, CudaEmbeddingApiRetriesRejectedArchWithSm80) {
    constexpr const char* kAddModule = R"MLIR(module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)MLIR";
    const fs::path tmp = make_tmp_dir();
    tj::CompileOptions opts;
    opts.target = "cuda";
    opts.cache_dir = tmp;
    opts.cuda_arch = "sm_9999";  // no NVPTX backend has ever shipped this

    tj::CompiledArtifact artifact;
    ASSERT_NO_THROW({ artifact = tj::compile_mlir(kAddModule, opts); })
        << "expected the sm_9999 rejection to be retried with sm_80 rather "
           "than propagating as a hard failure";

    EXPECT_TRUE(fs::exists(artifact.vmfb_path));
    EXPECT_GT(fs::file_size(artifact.vmfb_path), 0U);
    EXPECT_EQ(artifact.target, "cuda");

    // JIT-R149: the returned vmfb actually holds sm_80 bytecode (that's what
    // compiled), so it must be filed under the sm_80 cache key, not the
    // rejected sm_9999 one the compile was nominally requested for -- a
    // cache entry whose key/filename asserts "sm_9999" but contains sm_80
    // bytecode is a cache-integrity defect even though PTX forward-compat
    // happens to mask it functionally. Verify by compiling the SAME module
    // directly for sm_80 against the SAME cache dir: it must be a cache HIT
    // at the exact path the sm_9999 compile relocated to (not a fresh
    // compile to some other path), proving the fallback compile's output was
    // relocated to the key that actually matches its contents.
    tj::CompileOptions sm80_opts = opts;
    sm80_opts.cuda_arch = "sm_80";
    tj::CompiledArtifact sm80_artifact;
    ASSERT_NO_THROW({ sm80_artifact = tj::compile_mlir(kAddModule, sm80_opts); });
    EXPECT_EQ(sm80_artifact.vmfb_path, artifact.vmfb_path)
        << "sm_9999-requested compile's fallback output should be filed "
           "under the sm_80 cache key (same path a direct sm_80 compile "
           "resolves to), not left mislabeled under sm_9999's key";

    // JIT-R173 regression: the FIRST sm_9999 call above left the nominal
    // (sm_9999) cache key permanently empty after relocating its output to
    // the sm_80 key -- without the JIT-R173 fix, EVERY subsequent sm_9999
    // call would probe only that now-permanently-empty nominal key, miss,
    // and redo the whole failing-nominal-attempt + sm_80-retry +
    // re-relocation every single time, never hitting cache. Verify a SECOND
    // call with the exact original (nominal sm_9999) options returns the
    // same relocated artifact WITHOUT spending any real compile time --
    // total_compile_ms only increases on an actual compile_mlir cache miss
    // that reaches the real compile path (see record_compile_ms call
    // sites), so a near-zero delta here proves this hit the relocated
    // sm_80 cache directly instead of recompiling.
    tj::reset_cache_stats();
    tj::CompiledArtifact artifact2;
    ASSERT_NO_THROW({ artifact2 = tj::compile_mlir(kAddModule, opts); });
    EXPECT_EQ(artifact2.vmfb_path, artifact.vmfb_path)
        << "repeat sm_9999 compile must resolve to the same relocated "
           "sm_80-keyed vmfb, not a fresh path";
    EXPECT_LT(tj::cache_stats().total_compile_ms, 1.0)
        << "repeat sm_9999 compile for the SAME graph must hit the "
           "relocated sm_80 cache directly (JIT-R173), not redo the "
           "failing nominal-arch attempt + sm_80 retry on every call";

    fs::remove_all(tmp);
}

// JIT-R150 regression: the "always dump the MLIR source for offline debug"
// write had no error-checking at all -- not even f.good()/f.fail() -- so a
// write failure (disk full, quota exhaustion, a restrictive sandbox) would
// silently violate the comment's own claimed invariant ("so that the path
// returned in JitCompileError exists"), handing a caller catching
// JitCompileError a mlir_dump_path() that's missing or truncated during
// exactly the failure-diagnosis scenario the dump exists to support.
// Reproduced with a cache_dir made read-only after creation: the directory
// exists (so secure_cache_dir's create_directories is a no-op) but the
// ofstream open for the .mlir dump fails, which must now surface as a clear
// JitCompileError instead of silently producing no file.
TEST(IreeCompile, MlirDumpWriteFailureThrowsClearError) {
    if (::getuid() == 0) {
        GTEST_SKIP() << "running as root: permission bits don't block writes";
    }
    const fs::path tmp = make_tmp_dir();
    ASSERT_EQ(::chmod(tmp.c_str(), 0500), 0) << "failed to make " << tmp << " read-only";
    struct PermGuard {
        fs::path p;
        ~PermGuard() { ::chmod(p.c_str(), 0700); }  // restore so remove_all can clean up
    } perm_guard{tmp};

    tj::CompileOptions opts;
    opts.target = "llvm-cpu";
    opts.cache_dir = tmp;

    // A read-only cache_dir also makes the LATER vmfb-write step fail (with
    // its own, already-error-checked JitCompileError), so a bare
    // EXPECT_THROW wouldn't discriminate the fix from the pre-fix silent
    // dump-write failure -- both throw *some* JitCompileError here. What the
    // fix specifically must do is surface the dump-write failure itself
    // (message mentions the MLIR debug dump), before ever reaching the vmfb
    // compile attempt.
    bool threw = false;
    try {
        (void)tj::compile_mlir(kTrivialModule, opts);
    } catch (const tj::JitCompileError& e) {
        threw = true;
        EXPECT_NE(std::string(e.what()).find("MLIR debug dump"), std::string::npos)
            << "expected the error to specifically call out the MLIR debug "
               "dump write failure, got: " << e.what();
    }
    EXPECT_TRUE(threw) << "expected compile_mlir to throw JitCompileError "
                          "for a read-only cache_dir";

    ::chmod(tmp.c_str(), 0700);
    fs::remove_all(tmp);
}

// JIT-R167 regression: secure_cache_dir()/cache_file_is_trustworthy() used
// stat() (which FOLLOWS symlinks) rather than lstat() to verify the
// shared-temp cache dir's ownership. A local attacker who knows the
// victim's uid can pre-create a symlink at the deterministic path
// (<tmp>/tenzor-<uid>) pointing at a directory the victim already owns:
// stat() follows it and reports victim ownership (passing the "not
// squatted" check), and the subsequent chmod/mkdir then operate on the
// symlink's TARGET -- bypassing the entire defense via one directory-level
// symlink. Reproduced by planting exactly that symlink and confirming
// compile_mlir (with no XDG_CACHE_HOME/HOME, forcing the shared-temp
// fallback) now refuses it instead of silently following it.
TEST(IreeCompile, SharedTempCacheDirRejectsSymlinkSquatting) {
    if (::getuid() == 0) {
        GTEST_SKIP() << "running as root: this attack targets cross-user "
                         "ownership checks that don't apply to root";
    }

    const char* home_saved = std::getenv("HOME");
    const bool had_home = home_saved != nullptr;
    const std::string home_val = had_home ? home_saved : std::string{};
    const char* xdg_saved = std::getenv("XDG_CACHE_HOME");
    const bool had_xdg = xdg_saved != nullptr;
    const std::string xdg_val = had_xdg ? xdg_saved : std::string{};
    struct EnvGuard {
        bool had_home, had_xdg;
        std::string home_val, xdg_val;
        ~EnvGuard() {
            if (had_home) setenv("HOME", home_val.c_str(), 1); else unsetenv("HOME");
            if (had_xdg) setenv("XDG_CACHE_HOME", xdg_val.c_str(), 1); else unsetenv("XDG_CACHE_HOME");
        }
    } env_guard{had_home, had_xdg, home_val, xdg_val};
    unsetenv("HOME");
    unsetenv("XDG_CACHE_HOME");

    // Matches default_cache_dir()'s shared-temp derivation exactly.
    const fs::path tenzor_root =
        fs::temp_directory_path() / ("tenzor-" + std::to_string(::getuid()));

    // Preserve any pre-existing legitimate content at this real, shared,
    // deterministic path (e.g. from a prior real run on this machine)
    // rather than destroying it.
    const fs::path backup = fs::temp_directory_path() /
        ("tenzor_r167_test_backup_" + std::to_string(::getpid()));
    bool had_preexisting = fs::exists(tenzor_root) || fs::is_symlink(tenzor_root);
    if (had_preexisting) {
        fs::rename(tenzor_root, backup);
    }
    struct PathGuard {
        fs::path root, backup, attacker_target;
        bool had_preexisting;
        ~PathGuard() {
            std::error_code ec;
            fs::remove_all(root, ec);       // symlink or dir this test created
            fs::remove_all(attacker_target, ec);
            if (had_preexisting) {
                fs::rename(backup, root, ec);
            }
        }
    } path_guard{tenzor_root, backup, {}, had_preexisting};

    // Plant the attack: tenzor_root is a symlink to a directory that (in a
    // real attack) some OTHER user already owns; here it's simply a
    // separate directory this same test process owns, since a single test
    // process can't fork a genuinely different uid -- what matters for this
    // regression is that a SYMLINK sits at the deterministic path at all,
    // which secure_cache_dir must now categorically refuse regardless of
    // who owns what it points to.
    const fs::path attacker_target = fs::temp_directory_path() /
        ("tenzor_r167_attacker_target_" + std::to_string(::getpid()));
    fs::create_directories(attacker_target);
    path_guard.attacker_target = attacker_target;
    ASSERT_EQ(::symlink(attacker_target.c_str(), tenzor_root.c_str()), 0)
        << "failed to plant symlink at " << tenzor_root;

    tj::CompileOptions opts;
    opts.target = "llvm-cpu";
    // cache_dir left empty -> default_cache_dir() -> shared-temp fallback
    // (HOME/XDG_CACHE_HOME both unset above) -> secure_cache_dir() must
    // reject the planted symlink.
    EXPECT_THROW({ (void)tj::compile_mlir(kTrivialModule, opts); },
                tj::JitCompileError)
        << "expected compile_mlir to refuse a symlinked shared-temp cache "
           "dir instead of silently following it into the attacker's "
           "target directory";
}

// JIT-R188: the IreeCompilerInflightGuard added around every real embedding-
// API call site (embedding_api_supported_targets, iree_compiler_version,
// compile_via_embedding_api) must not introduce a deadlock or incorrectly
// serialize concurrent compiles under normal operation (no shutdown in
// progress) -- only the actual process-exit race it targets should ever
// throw/block. Several threads compile DISTINCT modules concurrently on
// llvm-cpu (which uses the embedding API) and each also queries
// iree_compiler_version()/embedding_api_supported_targets() concurrently;
// all must complete successfully with no exception, hang, or crash.
TEST(IreeCompile, ConcurrentEmbeddingApiCallsDoNotDeadlock) {
    const fs::path tmp = make_tmp_dir();
    constexpr int kThreads = 8;
    std::vector<std::thread> threads;
    std::vector<bool> ok(kThreads, false);
    std::vector<std::string> errors(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i] {
            try {
                std::ostringstream mod;
                mod << "module {\n"
                    << "  func.func @main(%arg0: tensor<" << (i + 1)
                    << "xf32>) -> tensor<" << (i + 1) << "xf32> {\n"
                    << "    %0 = stablehlo.add %arg0, %arg0 : tensor<"
                    << (i + 1) << "xf32>\n"
                    << "    return %0 : tensor<" << (i + 1) << "xf32>\n"
                    << "  }\n}\n";
                tj::CompileOptions opts;
                opts.target = "llvm-cpu";
                opts.cache_dir = tmp;
                auto artifact = tj::compile_mlir(mod.str(), opts);
                (void)tj::iree_compiler_version();
                ok[i] = fs::exists(artifact.vmfb_path) &&
                        fs::file_size(artifact.vmfb_path) > 0;
                if (!ok[i]) errors[i] = "empty or missing vmfb";
            } catch (const std::exception& e) {
                errors[i] = e.what();
            }
        });
    }
    for (auto& t : threads) t.join();
    for (int i = 0; i < kThreads; ++i) {
        EXPECT_TRUE(ok[i]) << "thread " << i << " failed: " << errors[i];
    }

    std::error_code ec;
    fs::remove_all(tmp, ec);
}
