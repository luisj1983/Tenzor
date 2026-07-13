// JIT-R109 regression: shared_iree_runtime_instance()/shared_iree_hal_device()
// used to hand back a raw pointer BEFORE retaining it -- the retain happened
// in the caller, one or two statements later, with no lock held in between.
// cleanup_shared_iree_state() (std::atexit-registered, runs at real process
// exit) releases every cached object under g_shared_iree_device_mu. If it ran
// in that gap -- a concurrent thread mid-JIT-call while the process exits --
// the object could drop to refcount 0 and be destroyed before the calling
// thread's retain(), which then operates on freed memory (SIGSEGV or silent
// heap corruption depending on allocator luck).
//
// The fix moves the retain for BOTH shared objects inside their accessor
// functions, under the SAME g_shared_iree_device_mu lock cleanup_shared_
// iree_state() uses for its release loop, so a caller's retain and
// cleanup's release can never interleave.
//
// This lives in its own executable because testing_force_shared_iree_state_
// cleanup() (exposed purely for this test) PERMANENTLY disables the shared
// IREE runtime instance for the rest of the process -- by design, since it
// exercises the exact std::atexit teardown path -- so it cannot share a
// binary with any other MLIR JIT test.

#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_runtime.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#include "tenzor/jit/tracer.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <random>
#include <thread>
#include <vector>

namespace {

namespace tj = ::tenzor::jit::mlir_jit;
namespace fs = std::filesystem;

auto make_tmp_dir() -> fs::path {
    const auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(static_cast<uint64_t>(now));
    fs::path dir = fs::temp_directory_path() /
                   ("tenzor_jit_r109_test_" + std::to_string(rng()));
    fs::create_directories(dir);
    return dir;
}

}  // namespace

TEST(JitR109SharedStateTeardownRace, ConcurrentInProcessUseSurvivesTeardown) {
    ::tenzor::initialize();

    ::tenzor::jit::Graph g;
    auto x_v = g.create_value("x", {4}, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    g.set_inputs({x_v});
    auto add_node_ = g.create_node(::tenzor::jit::OpType::Add);
    add_node_->add_input(x_v);
    add_node_->add_input(x_v);
    auto out = g.create_value("out", {4}, ::tenzor::DType::Float32,
                              ::tenzor::Device::cpu());
    add_node_->add_output(out);
    g.add_node(add_node_);
    g.set_outputs({out});

    tj::GraphToMLIR lowerer;
    const std::string mlir = lowerer.lower(g);

    tj::CompileOptions opts;
    opts.target    = "llvm-cpu";
    opts.cache_dir = make_tmp_dir();
    tj::CompiledArtifact artifact;
    ASSERT_NO_THROW({ artifact = tj::compile_mlir(mlir, opts); });

    // Warm the shared instance/device cache with one successful load first,
    // matching real usage (the shared objects are lazily created on first
    // InProcess call).
    {
        auto warm = tj::IreeInvoker::load(artifact, tj::IreeInvoker::Mode::InProcess, 0);
        ASSERT_NE(warm, nullptr);
    }

    std::atomic<bool> stop{false};
    std::atomic<int> post_cleanup_throws{0};
    std::atomic<int> iterations{0};

    // Background threads hammer the InProcess load path -- each call goes
    // through shared_iree_hal_device()'s cache-hit path (the exact code this
    // fix changed) for the whole run.
    constexpr int kThreads = 4;
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&]() {
            while (!stop.load(std::memory_order_relaxed)) {
                try {
                    auto inv = tj::IreeInvoker::load(
                        artifact, tj::IreeInvoker::Mode::InProcess, 0);
                    if (inv) iterations.fetch_add(1, std::memory_order_relaxed);
                } catch (const std::exception&) {
                    // Expected once cleanup has run: shared_iree_runtime_
                    // instance() now correctly throws instead of handing
                    // out a dangling reference.
                    post_cleanup_throws.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    // Let the workers run for a bit under normal (pre-teardown) conditions,
    // then force the exact teardown-while-concurrently-in-use race this fix
    // targets, then let them keep running so some calls land squarely in
    // (and after) the teardown window.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    tj::testing_force_shared_iree_state_cleanup();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stop.store(true, std::memory_order_relaxed);
    for (auto& w : workers) w.join();

    EXPECT_GT(iterations.load(), 0)
        << "no successful InProcess loads completed before cleanup -- this "
           "test didn't actually exercise the concurrent-use path";
    EXPECT_GT(post_cleanup_throws.load(), 0)
        << "no post-cleanup call ever observed the torn-down state -- "
           "either the teardown race window was too short or cleanup never "
           "ran";

    // Reaching this line at all -- kThreads threads racing testing_force_
    // shared_iree_state_cleanup() with no SIGSEGV / heap corruption -- IS
    // the actual JIT-R109 regression check: the pre-fix code retained the
    // shared instance/device OUTSIDE the lock cleanup_shared_iree_state()
    // uses to release them, so an unlucky scheduling here could dereference
    // freed memory. A crash (this process not reaching this point) is the
    // failure mode, not a normal EXPECT_*/ASSERT_* mismatch.
    SUCCEED();

    // shared_iree_runtime_instance() must now be permanently torn down.
    // Unlike the crash-based check above, this is a deterministic (non-
    // racy) assertion of the post-cleanup contract.
    EXPECT_THROW(
        {
            (void)tj::IreeInvoker::load(artifact, tj::IreeInvoker::Mode::InProcess, 0);
        },
        std::exception)
        << "shared_iree_runtime_instance() must throw after "
           "cleanup_shared_iree_state() has run, not silently succeed with "
           "a stale/dangling instance";

    std::error_code ec;
    fs::remove_all(opts.cache_dir, ec);
}
