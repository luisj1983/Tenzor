// JIT-R125 regression: a multi-output subprocess invocation whose LATER
// (not first) --output=@path temp file cannot be created must never
// silently return fewer tensors than @main actually produces.
//
// This lives in its own executable (rather than alongside
// test_jit_eager_fallback.cpp's broader suite) because the repro requires
// unshare(CLONE_NEWUSER | CLONE_NEWNS) to deterministically force the
// partial mkstemps() failure, and unshare(CLONE_NEWUSER) only succeeds in a
// single-threaded process. tenzor::initialize() (loading every backend) and
// several other tests in this file's original suite spin up threads that
// persist for the rest of the process, so this test only gets a genuine
// single-threaded process — and thus real regression-catching power,
// instead of a graceful skip — by being the first and only test to run.

#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_paths.hpp"
#include "tenzor/jit/mlir/iree_runtime.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/tracer.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

#include <sched.h>
#include <sys/mount.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

namespace tj = ::tenzor::jit::mlir_jit;
namespace fs = std::filesystem;

auto make_tmp_dir() -> fs::path {
    const auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(static_cast<uint64_t>(now));
    fs::path dir = fs::temp_directory_path() /
                   ("tenzor_jit_r125_test_" + std::to_string(rng()));
    fs::create_directories(dir);
    return dir;
}

}  // namespace

// JIT-R125 regression: force a LATER (not the first) --output=@path temp
// file to fail creation during a multi-output subprocess invocation, by
// entering a private user+mount namespace and mounting a tmpfs whose inode
// quota permits creating exactly ONE file. The first mkstemps() call
// (npy_path, made unconditionally before the multi-output branch) succeeds
// and consumes the one available inode; the per-extra-output loop's
// mkstemps() call then fails with ENOSPC. Before the fix, out_paths ended
// up silently shorter than expected_outputs and invoke_subprocess still
// built argv and trusted the (mismatched --output= count) result -- on this
// host that happens to surface as an opaque "iree-run-module exit=1" /
// OUT_OF_RANGE error from iree-run-module's own CLI validation, but per the
// original finding that behavior is NOT guaranteed across IREE versions.
// After the fix, invoke_subprocess abandons the bit-exact npy attempt
// entirely in that case and falls through to the ASCII-stdout path, which
// itself throws the JIT-R128 precision-loss guard for these two real-valued
// (Float64) outputs -- a clean, specific, well-diagnosed error instead of
// ever invoking iree-run-module with a malformed argv.
TEST(JitR125OutputTempExhaustion, MultiOutputSubprocessThrowsWhenLaterOutputTempFileUnavailable) {
    // unshare(CLONE_NEWUSER) requires the calling process to be single-
    // threaded, so this MUST happen before tenzor::initialize() (which loads
    // every backend and their driver/thread-pool init) or any other tenzor
    // API call. Enter a private user+mount namespace (unprivileged; no root
    // required) and mount a tmpfs whose inode quota permits exactly one file
    // to be created, then point TMPDIR at it -- this affects this process
    // (and its later iree-run-module child, which inherits the namespace
    // across fork/exec) only, and unwinds automatically when the process
    // exits. Skip gracefully rather than fail if namespaces aren't available.
    const uid_t outer_uid = ::getuid();
    const gid_t outer_gid = ::getgid();
    if (::unshare(CLONE_NEWUSER | CLONE_NEWNS) != 0) {
        GTEST_SKIP() << "user/mount namespaces unavailable in this "
                        "environment -- cannot deterministically force a "
                        "partial mkstemps failure (errno=" << errno << ")";
    }
    {
        std::ofstream setgroups_f("/proc/self/setgroups");
        setgroups_f << "deny";
    }
    {
        std::ofstream uid_map_f("/proc/self/uid_map");
        uid_map_f << "0 " << outer_uid << " 1";
    }
    {
        std::ofstream gid_map_f("/proc/self/gid_map");
        gid_map_f << "0 " << outer_gid << " 1";
    }
    const fs::path constrained_dir = fs::temp_directory_path() /
        ("tenzor_r125_constrained_" +
         std::to_string(std::chrono::system_clock::now().time_since_epoch().count()));
    fs::create_directories(constrained_dir);
    if (::mount("tmpfs", constrained_dir.c_str(), "tmpfs", 0,
                "size=1m,nr_inodes=2,mode=1777") != 0) {
        GTEST_SKIP() << "tmpfs mount unavailable in this environment -- "
                        "cannot deterministically force a partial mkstemps "
                        "failure (errno=" << errno << ")";
    }

    ::tenzor::initialize();

    ::tenzor::jit::Graph g;
    auto x_v = g.create_value("x", {4}, ::tenzor::DType::Float64,
                              ::tenzor::Device::cpu());
    g.set_inputs({x_v});

    auto add_node_ = g.create_node(::tenzor::jit::OpType::Add);
    add_node_->add_input(x_v);
    add_node_->add_input(x_v);
    auto out0 = g.create_value("out0", {4}, ::tenzor::DType::Float64,
                               ::tenzor::Device::cpu());
    add_node_->add_output(out0);
    g.add_node(add_node_);

    auto mul_node_ = g.create_node(::tenzor::jit::OpType::Mul);
    mul_node_->add_input(x_v);
    mul_node_->add_input(x_v);
    auto out1 = g.create_value("out1", {4}, ::tenzor::DType::Float64,
                               ::tenzor::Device::cpu());
    mul_node_->add_output(out1);
    g.add_node(mul_node_);

    g.set_outputs({out0, out1});

    tj::GraphToMLIR lowerer;
    const std::string mlir = lowerer.lower(g);
    ASSERT_NE(mlir.find("func.func @main"), std::string::npos) << mlir;

    tj::CompileOptions opts;
    opts.target    = "llvm-cpu";
    opts.cache_dir = make_tmp_dir();
    tj::CompiledArtifact artifact;
    ASSERT_NO_THROW({ artifact = tj::compile_mlir(mlir, opts); });

    auto invoker =
        tj::IreeInvoker::load(artifact, tj::IreeInvoker::Mode::Subprocess);
    invoker->set_expected_outputs(2);

    auto x_t = ::tenzor::full({4}, 0.1, ::tenzor::DType::Float64);

    const char* saved_tmpdir_raw = std::getenv("TMPDIR");
    const bool had_saved = saved_tmpdir_raw != nullptr;
    const std::string saved = had_saved ? saved_tmpdir_raw : std::string{};
    struct EnvGuard {
        bool had;
        std::string val;
        ~EnvGuard() {
            if (had) setenv("TMPDIR", val.c_str(), 1);
            else unsetenv("TMPDIR");
        }
    } env_guard{had_saved, saved};
    setenv("TMPDIR", constrained_dir.string().c_str(), 1);

    try {
        (void)invoker->invoke({x_t});
        FAIL() << "expected invoke() to throw when a later output's temp "
                  "file cannot be created for a multi-output subprocess "
                  "invocation (must not silently return a truncated result)";
    } catch (const std::exception& e) {
        // With the fix, invoke_subprocess abandons the bit-exact npy attempt
        // BEFORE ever invoking iree-run-module with a mismatched --output=
        // count, falling through cleanly to the ASCII-stdout path -- which
        // then throws the JIT-R128 precision-loss guard for these two
        // real-valued (Float64) outputs. Asserting on that specific message
        // (rather than accepting any exception) matters: on this host,
        // iree-run-module happens to also reject a mismatched --output=
        // count on its own (exit!=0, surfaced as a generic "iree-run-module
        // exit=1" error) -- which would make a same-shaped bug (the
        // all_output_temps_created gate silently removed) look like it still
        // "throws correctly" by sheer luck of that separate, unguaranteed
        // (per this finding) CLI behavior. The precise message ties this
        // test to the actual fix rather than an incidental downstream error.
        EXPECT_NE(std::string(e.what()).find("precision-degraded"), std::string::npos)
            << "expected the JIT-R128 precision-loss guard's message (reached "
               "via the ASCII fallback after abandoning the mismatched-count "
               "npy attempt), got: " << e.what();
    }

    std::error_code ec;
    fs::remove_all(opts.cache_dir, ec);
}
