// JIT-R168 regression: shared_iree_hal_device()'s cache-MISS/insert path had
// no guard against racing cleanup_shared_iree_state() -- unlike the sibling
// shared_iree_runtime_instance(), which the JIT-R109 fix already made throw
// if g_shared_iree_instance is null (already torn down). The device above is
// created OUTSIDE g_shared_iree_device_mu (JIT-R130, so a slow/wedged driver
// on one URI can't block unrelated compiles); if cleanup_shared_iree_state()
// runs in that window, the pre-fix code would try_emplace the freshly
// created device into the now-cleared map AFTER cleanup already ran -- no
// second cleanup pass is coming, so that device (and any background thread
// its HAL driver started, e.g. Vulkan's completion-watcher) is never
// released, reintroducing exactly the dlclose-vs-background-thread teardown
// race JIT-R109 eliminated for the cache-HIT path.
//
// test_jit_r109_shared_state_teardown_race.cpp's own comment notes it warms
// the device cache ONCE before racing threads start, so every racing call
// goes through the cache-HIT path for the whole run -- the cache-miss/insert
// branch this fix targets was never exercised by any test. This test forces
// genuine cache misses on every iteration (not just the first) by folding a
// unique, never-before-seen value into MESA_VK_DEVICE_SELECT each time --
// device_selection_env_key() (JIT-R141) folds that env var into
// shared_iree_hal_device()'s cache key for the "vulkan" driver, so each
// iteration gets a fresh key regardless of how many physical GPUs this host
// actually has, without needing multiple real devices to exercise the race
// repeatedly.
//
// Lives in its own executable for the same reason as the JIT-R109 test:
// testing_force_shared_iree_state_cleanup() permanently disables the shared
// IREE runtime state for the rest of the process.

#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_runtime.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#include "tenzor/jit/tracer.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/tenzor.hpp"

#include "../../backend_parity/parity_test_utils.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
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
                   ("tenzor_jit_r168_test_" + std::to_string(rng()));
    fs::create_directories(dir);
    return dir;
}

}  // namespace

TEST(JitR168CacheMissTeardownRace, CacheMissRacingCleanupDoesNotLeakOrCrash) {
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
    opts.target    = "vulkan-spirv";
    opts.cache_dir = make_tmp_dir();
    tj::CompiledArtifact artifact;
    try {
        artifact = tj::compile_mlir(mlir, opts);
    } catch (const std::exception& e) {
        // JIT-R177: escalate to FAIL() under TENZOR_REQUIRE_MULTI_BACKEND=1
        // instead of a bare skip with no signal -- a CI host that's supposed
        // to have Vulkan but whose driver silently failed would otherwise
        // report this dedicated cache-miss/teardown-race regression test as
        // a clean SKIPPED, with zero indication the race is actually
        // uncovered on that run.
        if (::tenzor::testing::golden::require_multi_backend()) {
            FAIL() << "Vulkan IREE target required by "
                      "TENZOR_REQUIRE_MULTI_BACKEND but not available: " << e.what();
        }
        GTEST_SKIP() << "no Vulkan IREE target available: " << e.what();
    }

    // Deliberately NO warm-up call: every load below must be a genuine cache
    // miss (see the unique-env-var trick below), unlike the JIT-R109 test
    // which warms the cache first and stays on the cache-HIT path.
    std::unique_ptr<tj::IreeInvoker> probe;
    try {
        probe = tj::IreeInvoker::load(artifact, tj::IreeInvoker::Mode::InProcess, 0);
    } catch (const std::exception& e) {
        if (::tenzor::testing::golden::require_multi_backend()) {
            FAIL() << "in-process Vulkan device required by "
                      "TENZOR_REQUIRE_MULTI_BACKEND but not available: " << e.what();
        }
        GTEST_SKIP() << "no in-process Vulkan device available: " << e.what();
    }
    if (!probe || probe->raw_device_handle_for_testing() == nullptr) {
        if (::tenzor::testing::golden::require_multi_backend()) {
            FAIL() << "in-process Vulkan device required by "
                      "TENZOR_REQUIRE_MULTI_BACKEND but not available";
        }
        GTEST_SKIP() << "no in-process Vulkan device available";
    }
    probe.reset();

    const char* saved_raw = std::getenv("MESA_VK_DEVICE_SELECT");
    const bool had_saved = saved_raw != nullptr;
    const std::string saved = had_saved ? saved_raw : std::string{};
    struct EnvGuard {
        bool had;
        std::string val;
        ~EnvGuard() {
            if (had) setenv("MESA_VK_DEVICE_SELECT", val.c_str(), 1);
            else unsetenv("MESA_VK_DEVICE_SELECT");
        }
    } env_guard{had_saved, saved};

    std::atomic<bool> stop{false};
    std::atomic<int> pre_cleanup_misses{0};
    std::atomic<int> post_cleanup_throws{0};
    std::atomic<int> post_cleanup_phantom_successes{0};
    std::atomic<bool> cleanup_started{false};
    std::atomic<uint64_t> counter{0};

    // Single background thread: setenv is not safe to call concurrently from
    // multiple threads racing each other (glibc's env functions share global
    // state with no internal locking against each other), so one thread
    // serializes the "fresh cache key every iteration" trick while the main
    // thread races cleanup against it below.
    std::thread worker([&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            const uint64_t i = counter.fetch_add(1, std::memory_order_relaxed);
            // A unique value each iteration -> device_selection_env_key()
            // computes a distinct cache key -> shared_iree_hal_device() MUST
            // take the cache-miss/create/insert branch every time, not just
            // on the first call.
            setenv("MESA_VK_DEVICE_SELECT", ("r168_fake_" + std::to_string(i)).c_str(), 1);
            // Sampled BEFORE the call: the bug being tested is a device
            // creation that STARTS before cleanup but whose try_emplace
            // lands after -- pre-fix this silently "succeeds" (a phantom,
            // never-to-be-released device) instead of throwing, so a crash
            // is NOT the expected symptom here (unlike the JIT-R109 test,
            // whose bug is a use-after-free). What must never happen is
            // "this call returned a live device AFTER cleanup had already
            // been requested".
            const bool cleanup_was_started_before_call =
                cleanup_started.load(std::memory_order_relaxed);
            try {
                auto inv = tj::IreeInvoker::load(
                    artifact, tj::IreeInvoker::Mode::InProcess, 0);
                if (inv) {
                    pre_cleanup_misses.fetch_add(1, std::memory_order_relaxed);
                    if (cleanup_was_started_before_call) {
                        post_cleanup_phantom_successes.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                }
            } catch (const std::exception&) {
                // Expected once cleanup has run: the JIT-R168 fix throws
                // instead of caching a device the map can no longer track.
                post_cleanup_throws.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    // Each iteration creates a brand-new Vulkan device (a genuine cache
    // miss every time by construction), which is measurably slower than the
    // JIT-R109 test's cache-HIT-only loop -- give the first few iterations
    // enough time to actually complete before forcing teardown, so this
    // test exercises "some misses succeed, then teardown races a later one"
    // rather than "teardown wins before anything ever succeeds" (still a
    // valid code path, but not the specific interleaving this test targets).
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    cleanup_started.store(true, std::memory_order_relaxed);
    tj::testing_force_shared_iree_state_cleanup();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    stop.store(true, std::memory_order_relaxed);
    worker.join();

    EXPECT_GT(pre_cleanup_misses.load(), 0)
        << "no successful cache-miss load completed before cleanup -- this "
           "test didn't actually exercise the concurrent cache-miss path";
    EXPECT_GT(post_cleanup_throws.load(), 0)
        << "no post-cleanup call ever observed the torn-down state -- "
           "either the teardown race window was too short or cleanup never "
           "ran";
    // This is the actual JIT-R168 regression check: pre-fix, a device
    // creation racing cleanup can silently succeed (try_emplace into the
    // just-cleared map) instead of throwing -- a phantom device the cache
    // no longer tracks and cleanup_shared_iree_state() will never release
    // again. Any such success is the bug, not a timing fluke.
    EXPECT_EQ(post_cleanup_phantom_successes.load(), 0)
        << "a cache-miss load returned a \"successful\" device AFTER "
           "cleanup_shared_iree_state() had already been invoked -- this is "
           "a phantom device that bypassed the already-torn-down guard and "
           "will never be released";

    // Reaching this line at all -- the worker thread racing cache-miss
    // device creation against testing_force_shared_iree_state_cleanup()
    // with no SIGSEGV / heap corruption / leaked-forever device -- IS the
    // actual JIT-R168 regression check. A crash (this process not reaching
    // this point) is the failure mode, not a normal EXPECT_*/ASSERT_*
    // mismatch.
    SUCCEED();

    // Deterministic (non-racy) post-cleanup contract: every subsequent
    // cache-miss attempt must throw, never silently cache/leak a device.
    setenv("MESA_VK_DEVICE_SELECT", "r168_after_cleanup", 1);
    EXPECT_THROW(
        {
            (void)tj::IreeInvoker::load(artifact, tj::IreeInvoker::Mode::InProcess, 0);
        },
        std::exception)
        << "shared_iree_hal_device()'s cache-miss path must throw after "
           "cleanup_shared_iree_state() has run, not silently cache a "
           "device that will never be released";

    std::error_code ec;
    fs::remove_all(opts.cache_dir, ec);
}
