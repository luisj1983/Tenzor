/**
 * @file test_profiler.cpp
 * @brief Unit tests for AutogradProfiler infrastructure
 *
 * CPU-only tests exercising the profiler singleton, enable/disable,
 * recording, summary output, reset, and Chrome trace export.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/profiler.hpp>

#include <chrono>
#include <filesystem>
#include <mutex>
#include <string>
#include <unistd.h>  // getpid — CC.17 per-test, per-process temp path

using namespace tenzor;
using namespace std::chrono_literals;

class ProfilerTest : public ::testing::Test {
protected:
    void SetUp() override {
        static std::once_flag init_flag;
        std::call_once(init_flag, []() { tenzor::initialize(); });

        // Always start from a clean, disabled state.
        auto& prof = AutogradProfiler::instance();
        prof.disable();
        prof.reset();
    }

    void TearDown() override {
        auto& prof = AutogradProfiler::instance();
        prof.disable();
        prof.reset();
    }
};

// ============================================================================
// 1. DefaultDisabled
// ============================================================================

TEST_F(ProfilerTest, DefaultDisabled) {
    auto& prof = AutogradProfiler::instance();
    EXPECT_FALSE(prof.is_enabled());
    EXPECT_FALSE(prof.is_trace_enabled());
}

// ============================================================================
// 2. EnableDisable
// ============================================================================

TEST_F(ProfilerTest, EnableDisable) {
    auto& prof = AutogradProfiler::instance();

    prof.enable();
    EXPECT_TRUE(prof.is_enabled());

    prof.disable();
    EXPECT_FALSE(prof.is_enabled());
}

// ============================================================================
// 3. EnableTrace
// ============================================================================

TEST_F(ProfilerTest, EnableTrace) {
    auto& prof = AutogradProfiler::instance();

    prof.enable_trace();
    EXPECT_TRUE(prof.is_trace_enabled());
    // enable_trace() also enables profiling
    EXPECT_TRUE(prof.is_enabled());

    prof.disable();
    EXPECT_FALSE(prof.is_trace_enabled());
}

// ============================================================================
// 4. RecordAndSummary
// ============================================================================

TEST_F(ProfilerTest, RecordAndSummary) {
    auto& prof = AutogradProfiler::instance();
    prof.enable();

    prof.record("test_op", std::chrono::nanoseconds(1'000'000));  // 1 ms

    auto s = prof.summary();
    EXPECT_FALSE(s.empty());
    EXPECT_NE(s.find("test_op"), std::string::npos)
        << "Summary should contain the recorded operation name. Got:\n" << s;
}

// ============================================================================
// 5. Reset
// ============================================================================

TEST_F(ProfilerTest, Reset) {
    auto& prof = AutogradProfiler::instance();
    prof.enable();

    prof.record("old_op", std::chrono::nanoseconds(500'000));
    // Verify it was recorded
    ASSERT_NE(prof.summary().find("old_op"), std::string::npos);

    prof.reset();

    auto s = prof.summary();
    EXPECT_EQ(s.find("old_op"), std::string::npos)
        << "After reset, old operations should not appear in summary. Got:\n" << s;
}

// ============================================================================
// 6. ExportChromeTrace
// ============================================================================

TEST_F(ProfilerTest, ExportChromeTrace) {
    auto& prof = AutogradProfiler::instance();
    prof.enable_trace();

    // Record a trace event with explicit start/duration for the trace log.
    auto start = std::chrono::steady_clock::now();
    prof.record("traced_op", std::chrono::nanoseconds(2'000'000));
    prof.record_trace("traced_op", start, std::chrono::nanoseconds(2'000'000),
                      ProfilePhase::Backward);

    // CC.17: per-process, per-test temp path so parallel ctest runs don't race.
    const std::string path = (std::filesystem::temp_directory_path() /
        ("tenzor_test_trace_" + std::to_string(::getpid()) + "_" +
         ::testing::UnitTest::GetInstance()->current_test_info()->name() +
         ".json")).string();

    // Remove pre-existing file to ensure we test fresh creation.
    std::filesystem::remove(path);

    prof.export_chrome_trace(path);

    EXPECT_TRUE(std::filesystem::exists(path))
        << "export_chrome_trace should create " << path;

    // Cleanup
    std::filesystem::remove(path);
}

// ============================================================================
// 7. Singleton
// ============================================================================

TEST_F(ProfilerTest, Singleton) {
    auto* a = &AutogradProfiler::instance();
    auto* b = &AutogradProfiler::instance();
    EXPECT_EQ(a, b) << "instance() must always return the same object";
}
