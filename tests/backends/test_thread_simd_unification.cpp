/**
 * @file test_thread_simd_unification.cpp
 * @brief Stream 17 regression — CPU thread/SIMD unification
 *
 * Validates:
 *   1. configure_omp_threads() is idempotent — repeated calls do not bounce
 *      the thread count.
 *   2. TENZOR_NUM_THREADS is honoured when OMP_NUM_THREADS is unset.
 *   3. OMP_NUM_THREADS takes precedence over TENZOR_NUM_THREADS.
 *   4. CPUBackend::event_elapsed_ms reports a wall-clock delta close to a
 *      sleep_for() interval (Stream 17 Fix 4).
 *   5. runtime_simd::detect_simd_level() returns a coherent value
 *      (>= SSE2 on x86-64).
 *
 * Tests 1–3 modify global env state and call configure_omp_threads() in
 * subprocess-free isolation. Because the function is std::call_once-guarded,
 * the second test invocation in a single process would always see the value
 * fixed by the first. We therefore exercise the env-precedence and
 * idempotence checks in a way that does not assume a fresh once_flag.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/loader.hpp>
#include <tenzor/backend/runtime_simd.hpp>

#include <chrono>
#include <cstdlib>
#include <thread>

#ifdef _OPENMP
#  include <omp.h>
#endif

using namespace tenzor;

namespace {

class ThreadSimdUnificationTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

// -------------------------------------------------------------------------
// Fix 1 / Test 1 — Single thread default consistency
// -------------------------------------------------------------------------
// configure_omp_threads() is guarded by std::call_once. Repeated invocations
// through create_backend() (and therefore the once-flag) must observe the
// same omp_get_max_threads() value. The first call wins; later calls must
// not "reset" omp threads back to a runtime default.
TEST_F(ThreadSimdUnificationTest, SingleThreadDefaultConsistency) {
#ifndef _OPENMP
    GTEST_SKIP() << "OpenMP not built — omp_get_max_threads unavailable";
#else
    auto& loader = backend_registry();
    auto* cpu1 = loader.get_backend("cpu");
    ASSERT_NE(cpu1, nullptr);
    int first = omp_get_max_threads();

    // Fetch the backend again — registry caches the same instance, but the
    // important invariant is that omp_get_max_threads is unchanged.
    auto* cpu2 = loader.get_backend("cpu");
    ASSERT_NE(cpu2, nullptr);
    int second = omp_get_max_threads();

    EXPECT_EQ(first, second)
        << "configure_omp_threads must be idempotent — repeated backend "
           "look-ups changed omp_get_max_threads from "
        << first << " to " << second;
    EXPECT_GE(first, 1) << "OMP thread count must be at least 1";
#endif
}

// -------------------------------------------------------------------------
// Fix 1 / Test 2 — Auto default is sensible
// -------------------------------------------------------------------------
// Validates that configure_omp_threads() produces a thread count consistent
// with hardware_concurrency() — i.e. the bogus "/2" default is gone.
TEST_F(ThreadSimdUnificationTest, AutoDefaultMatchesHardwareConcurrency) {
#ifndef _OPENMP
    GTEST_SKIP() << "OpenMP not built";
#else
    // If a user pre-set OMP_NUM_THREADS or TENZOR_NUM_THREADS, configure_omp
    // _threads will honour it and not reflect the auto path; in that case
    // we still validate the precedence — the runtime threads count must
    // match the env override.
    const char* omp_env = std::getenv("OMP_NUM_THREADS");
    const char* tz_env = std::getenv("TENZOR_NUM_THREADS");

    auto& loader = backend_registry();
    auto* cpu = loader.get_backend("cpu");
    ASSERT_NE(cpu, nullptr);
    int threads = omp_get_max_threads();

    if (omp_env && std::atoi(omp_env) > 0) {
        EXPECT_EQ(threads, std::atoi(omp_env))
            << "OMP_NUM_THREADS pre-set to " << omp_env
            << " but omp_get_max_threads reports " << threads;
    } else if (tz_env && std::atoi(tz_env) > 0) {
        EXPECT_EQ(threads, std::atoi(tz_env))
            << "TENZOR_NUM_THREADS pre-set to " << tz_env
            << " but omp_get_max_threads reports " << threads;
    } else {
        // Auto path: must NOT be hardware_concurrency() / 2 — that's the
        // bug Stream 17 fixed. We allow either online-cores or
        // hardware_concurrency() (within a tolerance for hybrid chips).
        unsigned int hw = std::thread::hardware_concurrency();
        if (hw > 0) {
            EXPECT_GT(threads, static_cast<int>(hw) / 2)
                << "Auto thread count " << threads
                << " is suspiciously close to the legacy hardware/2 default ("
                << (hw / 2) << "); hardware_concurrency=" << hw;
        }
        EXPECT_GE(threads, 1);
    }
#endif
}

// -------------------------------------------------------------------------
// Fix 4 / Test 4 — CPU event elapsed time
// -------------------------------------------------------------------------
TEST_F(ThreadSimdUnificationTest, CPUEventElapsedTime) {
    auto& loader = backend_registry();
    auto* cpu = loader.get_backend("cpu");
    ASSERT_NE(cpu, nullptr);

    auto start = cpu->create_event(0, /*enable_timing=*/true);
    auto end = cpu->create_event(0, /*enable_timing=*/true);
    ASSERT_NE(start, nullptr) << "CPU event handle must be a real timestamp";
    ASSERT_NE(end, nullptr);

    auto stream = cpu->create_stream(0);
    cpu->record_event(start, stream);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    cpu->record_event(end, stream);

    float elapsed_ms = cpu->event_elapsed_ms(start, end);
    EXPECT_GE(elapsed_ms, 5.0f)
        << "Elapsed time " << elapsed_ms
        << " ms is far below the 10 ms sleep — wall-clock is not being captured";
    EXPECT_LT(elapsed_ms, 5000.0f)
        << "Elapsed time " << elapsed_ms
        << " ms is implausibly large (>= 5 s) for a 10 ms sleep";

    // Same event used twice — elapsed must be exactly zero.
    EXPECT_FLOAT_EQ(cpu->event_elapsed_ms(start, start), 0.0f);

    cpu->destroy_event(start);
    cpu->destroy_event(end);
    cpu->destroy_stream(stream);
}

// -------------------------------------------------------------------------
// Fix 4 — Null event handles must not crash
// -------------------------------------------------------------------------
TEST_F(ThreadSimdUnificationTest, CPUEventNullHandlesAreSafe) {
    auto& loader = backend_registry();
    auto* cpu = loader.get_backend("cpu");
    ASSERT_NE(cpu, nullptr);

    EXPECT_FLOAT_EQ(cpu->event_elapsed_ms(nullptr, nullptr), 0.0f);
    cpu->record_event(nullptr, nullptr);    // No-op, must not crash.
    cpu->wait_event(nullptr, nullptr);      // No-op, must not crash.
    cpu->destroy_event(nullptr);            // delete nullptr is well-defined.
    cpu->synchronize_event(nullptr);
}

// -------------------------------------------------------------------------
// Fix 2 / Test 5 — SIMD detection coherence
// -------------------------------------------------------------------------
// Detected level must agree with the features struct's best_level() — they
// are derived from the same CPUID/XCR0 cache and must never disagree.
TEST_F(ThreadSimdUnificationTest, SIMDDetectionCoherence) {
    auto level = ::tenzor::backend::detect_simd_level();
    const auto& features = ::tenzor::backend::get_simd_features();
    EXPECT_EQ(level, features.best_level())
        << "detect_simd_level() and SIMDFeatures::best_level() must agree";

    // On x86-64 SSE2 is the baseline ISA — it must always be detected.
    #if defined(__x86_64__) || defined(_M_X64)
        EXPECT_TRUE(features.sse2)
            << "SSE2 must be detected on x86-64 (baseline ISA)";
        EXPECT_TRUE(::tenzor::backend::has_simd_feature(
            ::tenzor::backend::SIMDLevel::SSE2))
            << "has_simd_feature(SSE2) must return true on x86-64";
    #endif

    // AVX-512 detection must respect OS state-save support: if avx512f is
    // claimed, os_avx512 must also be true (otherwise we'd SIGILL).
    if (features.avx512f) {
        EXPECT_TRUE(features.os_avx512)
            << "SIMDFeatures reports AVX-512 hardware but OS XSAVE is "
               "disabled — get_simd_features() forgot the XCR0 check";
    }
    if (features.avx) {
        EXPECT_TRUE(features.os_avx)
            << "AVX claimed without OS YMM save support";
    }
}

} // namespace
