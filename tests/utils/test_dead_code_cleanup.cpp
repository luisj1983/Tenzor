/**
 * @file test_dead_code_cleanup.cpp
 * @brief Coverage for Stream 25 / Audit-12 dead-code cleanups.
 *
 * The S25 stream removed/consolidated several long-dormant pieces of
 * infrastructure:
 *
 *   1. Stale forward declarations in `cpu_backend.cpp`.
 *   2. The 4070-line `register_cpu_kernels` function split into per-category
 *      helpers.
 *   3. `NotImplementedError` exception type that was declared but never
 *      thrown — now used at known-stub sites.
 *   4. Two parallel logging stacks deduplicated; `tenzor::Logger` now
 *      forwards to the spdlog facade.
 *   5/6. Trivial syntax cleanups (double semicolon, VLA → std::vector).
 *
 * These tests anchor the behavioural contract of (3) and (4) so that
 * future refactors don't accidentally regress them.
 */

#include <gtest/gtest.h>

#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/utils/log.hpp"
#include "tenzor/utils/logging.hpp"

#include <memory>

#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unistd.h>  // getpid

namespace {

class DeadCodeCleanupTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

// ---------------------------------------------------------------------------
// Fix 3: `NotImplementedError` is constructible, is a `std::exception`, and
// is now thrown by at least one production path that used to throw a bare
// `std::runtime_error`.
// ---------------------------------------------------------------------------

TEST_F(DeadCodeCleanupTest, NotImplementedErrorIsAStdException) {
    static_assert(std::is_base_of_v<std::exception, tenzor::NotImplementedError>,
                  "NotImplementedError must derive from std::exception");
    static_assert(std::is_base_of_v<std::runtime_error, tenzor::NotImplementedError>,
                  "NotImplementedError derives via TenzorException → runtime_error");

    try {
        throw tenzor::NotImplementedError("S25 test stub");
    } catch (const std::exception& e) {
        // TenzorException prefixes the message with source location info,
        // so we only assert that the original payload survived somewhere.
        const std::string msg{e.what()};
        EXPECT_NE(msg.find("S25 test stub"), std::string::npos)
            << "NotImplementedError::what() lost the message payload: " << msg;
    }
}

TEST_F(DeadCodeCleanupTest, NotImplementedErrorIsThrownBySomeProductionPath) {
    // The S25 sweep converted ~38 throw sites. The most cleanly observable
    // ones are the abstract container forwards (`ModuleList::forward()`,
    // `ModuleDict::forward()`) in `src/nn/module.cpp`. Each used to throw
    // `std::runtime_error("... does not implement forward(). ...")` and now
    // throws `NotImplementedError`.
    //
    // Invoke the REAL production path: a bare ModuleList has no forward, so
    // calling forward() must throw NotImplementedError (not a bare
    // runtime_error). This is the actual S25 conversion under test — not an
    // in-test rethrow.
    using tenzor::nn::ModuleList;
    using tenzor::nn::ModuleDict;

    tenzor::Variable input(tenzor::ones({2, 3}, tenzor::DType::Float32),
                           /*requires_grad=*/false);

    {
        ModuleList list;
        bool caught_typed = false;
        bool caught_other = false;
        try {
            (void)list.forward(input);
        } catch (const tenzor::NotImplementedError& e) {
            caught_typed = true;
            const std::string msg{e.what()};
            EXPECT_NE(msg.find("does not implement forward"), std::string::npos)
                << "ModuleList::forward() threw NotImplementedError but with an "
                   "unexpected message: " << msg;
        } catch (const std::exception&) {
            caught_other = true;
        }
        EXPECT_TRUE(caught_typed)
            << "ModuleList::forward() must throw the converted NotImplementedError.";
        EXPECT_FALSE(caught_other)
            << "ModuleList::forward() threw a non-NotImplementedError exception.";
    }

    {
        ModuleDict dict;
        bool caught_typed = false;
        try {
            (void)dict.forward(input);
        } catch (const tenzor::NotImplementedError&) {
            caught_typed = true;
        } catch (const std::exception&) {
            // fall through; caught_typed stays false
        }
        EXPECT_TRUE(caught_typed)
            << "ModuleDict::forward() must throw the converted NotImplementedError.";
    }

    // NotImplementedError must be catchable as its concrete type, distinct from
    // a bare std::runtime_error — a converted site that fell back to
    // runtime_error would be caught by the runtime_error handler instead.
    bool caught_typed = false;
    bool caught_generic_runtime = false;
    try {
        ModuleList list;
        (void)list.forward(input);
    } catch (const tenzor::NotImplementedError&) {
        caught_typed = true;
    } catch (const std::runtime_error&) {
        caught_generic_runtime = true;
    }
    EXPECT_TRUE(caught_typed);
    EXPECT_FALSE(caught_generic_runtime);
}

// ---------------------------------------------------------------------------
// Fix 4: tenzor::Logger (legacy facade) is callable and routes through the
// unified spdlog logger without throwing. Also verifies the spdlog-facade
// (`tenzor::utils::logger`) is alive and shares state.
// ---------------------------------------------------------------------------

TEST_F(DeadCodeCleanupTest, LegacyLoggerIsCallableAfterDedup) {
    auto& legacy = tenzor::Logger::instance();

    // Route legacy output to a private temp file so we can verify the message
    // was actually ROUTED, not just that the call didn't throw.
    const std::string log_path =
        (std::filesystem::temp_directory_path() /
         ("tenzor_deadcode_legacy_" + std::to_string(::getpid()) + ".log"))
            .string();
    if (std::filesystem::exists(log_path)) std::filesystem::remove(log_path);

    // Set a known level — this also mirrors onto the spdlog logger.
    legacy.set_level(tenzor::LogLevel::Warning);
    EXPECT_EQ(legacy.get_level(), tenzor::LogLevel::Warning);
    legacy.set_output_file(log_path);
    legacy.enable_console(false);

    // Each severity method should be a no-throw forward to spdlog.
    EXPECT_NO_THROW(legacy.debug("DEADCODE_DEBUG_should_be_filtered"));
    EXPECT_NO_THROW(legacy.info("DEADCODE_INFO_should_be_filtered"));
    EXPECT_NO_THROW(legacy.warning("DEADCODE_WARNING_should_appear"));
    EXPECT_NO_THROW(legacy.error("DEADCODE_ERROR_should_appear"));
    EXPECT_NO_THROW(legacy.fatal("DEADCODE_FATAL_should_appear"));

    // Restore default routing before reading back.
    legacy.enable_console(true);

    std::ifstream f(log_path);
    ASSERT_TRUE(f.good()) << "legacy logger never created its output file";
    const std::string contents((std::istreambuf_iterator<char>(f)),
                               std::istreambuf_iterator<char>());

    // Warning/error/fatal (>= Warning threshold) must have been routed.
    EXPECT_NE(contents.find("DEADCODE_WARNING_should_appear"), std::string::npos)
        << "warning-level message was not routed to the legacy logger sink";
    EXPECT_NE(contents.find("DEADCODE_ERROR_should_appear"), std::string::npos);
    EXPECT_NE(contents.find("DEADCODE_FATAL_should_appear"), std::string::npos);
    // Debug/info (below the Warning threshold) must have been filtered out.
    EXPECT_EQ(contents.find("DEADCODE_DEBUG_should_be_filtered"), std::string::npos)
        << "debug message leaked past the Warning level filter";
    EXPECT_EQ(contents.find("DEADCODE_INFO_should_be_filtered"), std::string::npos);

    if (std::filesystem::exists(log_path)) std::filesystem::remove(log_path);
}

TEST_F(DeadCodeCleanupTest, UnifiedSpdlogFacadeIsCallable) {
    auto lg = tenzor::utils::logger();
    ASSERT_NE(lg, nullptr) << "tenzor::utils::logger() must return a valid logger";

    EXPECT_NO_THROW(TENZOR_LOG_INFO("S25: smoke-test the spdlog facade"));
    EXPECT_NO_THROW(TENZOR_LOG_WARN("S25: smoke-test warn-level path"));
    EXPECT_NO_THROW(TENZOR_LOG_ERROR("S25: smoke-test error-level path"));
}

TEST_F(DeadCodeCleanupTest, LegacyMacrosOnlyDefinedHereStillWork) {
    // Macros that exist only in `logging.hpp` (not `log.hpp`): WARNING,
    // FATAL, WARN_ONCE. These should still expand to legacy Logger calls.
    EXPECT_NO_THROW(TENZOR_LOG_WARNING("S25: legacy WARNING macro"));
    EXPECT_NO_THROW(TENZOR_LOG_FATAL("S25: legacy FATAL macro"));
    EXPECT_NO_THROW(TENZOR_WARN_ONCE("S25: legacy WARN_ONCE macro"));
    // Calling WARN_ONCE again should be a silent no-op (covered by the
    // call_once flag); just verify it doesn't throw.
    EXPECT_NO_THROW(TENZOR_WARN_ONCE("S25: legacy WARN_ONCE macro"));
}

}  // namespace
