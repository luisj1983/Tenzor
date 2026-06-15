/**
 * @file test_logging.cpp
 * @brief Unit tests for logging system
 */

#include <gtest/gtest.h>
#include <tenzor/utils/logging.hpp>
#include <fstream>
#include <filesystem>
#include <string>
#include <unistd.h>  // getpid — audit-5 Y.33

using namespace tenzor;

class LoggingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Audit-5 Y.33: per-test, per-process temp path. Mirrors the
        // tests/nn/test_safetensors.cpp pattern (audit-3 T.11). The old
        // shared "/tmp/tenzor_test.log" raced when two `bin/test_logging`
        // processes ran on the same host (CI matrix, parallel re-runs) and
        // also stomped any developer's hand-rolled log at that path.
        const auto* info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        const std::string test_name = info ? info->name() : "unknown";
        test_log_file_ = (std::filesystem::temp_directory_path() /
                          ("tenzor_logging_" + std::to_string(::getpid()) +
                           "_" + test_name + ".log")).string();

        // Reset logger to default state
        auto& logger = Logger::instance();
        logger.set_level(LogLevel::Info);
        logger.enable_console(true);

        // Clean up test log file if it exists
        if (std::filesystem::exists(test_log_file_)) {
            std::filesystem::remove(test_log_file_);
        }
    }

    void TearDown() override {
        // Clean up test log file
        if (std::filesystem::exists(test_log_file_)) {
            std::filesystem::remove(test_log_file_);
        }
    }

    std::string test_log_file_;
};

// Test 1: Singleton instance
TEST_F(LoggingTest, SingletonInstance) {
    auto& logger1 = Logger::instance();
    auto& logger2 = Logger::instance();

    // Should be same instance
    EXPECT_EQ(&logger1, &logger2);
}

// Test 2: Log level setting and getting
TEST_F(LoggingTest, LogLevelGetSet) {
    auto& logger = Logger::instance();

    logger.set_level(LogLevel::Debug);
    EXPECT_EQ(logger.get_level(), LogLevel::Debug);

    logger.set_level(LogLevel::Warning);
    EXPECT_EQ(logger.get_level(), LogLevel::Warning);

    logger.set_level(LogLevel::Fatal);
    EXPECT_EQ(logger.get_level(), LogLevel::Fatal);
}

// Test 3: Log level filtering - Debug level
TEST_F(LoggingTest, LogLevelFilteringDebug) {
    auto& logger = Logger::instance();
    logger.set_level(LogLevel::Debug);
    logger.set_output_file(test_log_file_);
    logger.enable_console(false);

    logger.debug("Debug message");
    logger.info("Info message");
    logger.warning("Warning message");
    logger.error("Error message");

    // All messages should be logged
    std::ifstream file(test_log_file_);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    EXPECT_NE(content.find("[DEBUG]"), std::string::npos);
    EXPECT_NE(content.find("[INFO]"), std::string::npos);
    EXPECT_NE(content.find("[WARNING]"), std::string::npos);
    EXPECT_NE(content.find("[ERROR]"), std::string::npos);
}

// Test 4: Log level filtering - Info level
TEST_F(LoggingTest, LogLevelFilteringInfo) {
    auto& logger = Logger::instance();
    logger.set_level(LogLevel::Info);
    logger.set_output_file(test_log_file_);
    logger.enable_console(false);

    logger.debug("Debug message");
    logger.info("Info message");
    logger.warning("Warning message");

    std::ifstream file(test_log_file_);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    // Debug should be filtered
    EXPECT_EQ(content.find("[DEBUG]"), std::string::npos);
    EXPECT_NE(content.find("[INFO]"), std::string::npos);
    EXPECT_NE(content.find("[WARNING]"), std::string::npos);
}

// Test 5: Log level filtering - Warning level
TEST_F(LoggingTest, LogLevelFilteringWarning) {
    auto& logger = Logger::instance();
    logger.set_level(LogLevel::Warning);
    logger.set_output_file(test_log_file_);
    logger.enable_console(false);

    logger.debug("Debug message");
    logger.info("Info message");
    logger.warning("Warning message");
    logger.error("Error message");

    std::ifstream file(test_log_file_);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    // Debug and Info should be filtered
    EXPECT_EQ(content.find("[DEBUG]"), std::string::npos);
    EXPECT_EQ(content.find("[INFO]"), std::string::npos);
    EXPECT_NE(content.find("[WARNING]"), std::string::npos);
    EXPECT_NE(content.find("[ERROR]"), std::string::npos);
}

// Test 6: File output
TEST_F(LoggingTest, FileOutput) {
    auto& logger = Logger::instance();
    logger.set_output_file(test_log_file_);
    logger.enable_console(false);

    logger.info("Test message 1");
    logger.warning("Test message 2");

    ASSERT_TRUE(std::filesystem::exists(test_log_file_));

    std::ifstream file(test_log_file_);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    EXPECT_NE(content.find("Test message 1"), std::string::npos);
    EXPECT_NE(content.find("Test message 2"), std::string::npos);
}

// Test 7: Console enable/disable
TEST_F(LoggingTest, ConsoleEnableDisable) {
    auto& logger = Logger::instance();

    logger.enable_console(true);
    // Console logging is enabled by default, hard to test output capture
    // Just verify no crash
    logger.info("Console enabled message");

    logger.enable_console(false);
    logger.info("Console disabled message");

    // No exception = pass
    SUCCEED();
}

// Test 8: Multiple log calls append to file
TEST_F(LoggingTest, MultipleLogCallsAppend) {
    auto& logger = Logger::instance();
    logger.set_output_file(test_log_file_);
    logger.enable_console(false);

    logger.info("Message 1");
    logger.info("Message 2");
    logger.info("Message 3");

    std::ifstream file(test_log_file_);
    int line_count = 0;
    std::string line;
    while (std::getline(file, line)) {
        line_count++;
    }

    EXPECT_EQ(line_count, 3);
}

// Test 9: Fatal log level
TEST_F(LoggingTest, FatalLogLevel) {
    auto& logger = Logger::instance();
    logger.set_output_file(test_log_file_);
    logger.enable_console(false);
    logger.set_level(LogLevel::Fatal);

    logger.debug("Debug");
    logger.info("Info");
    logger.warning("Warning");
    logger.error("Error");
    logger.fatal("Fatal");

    std::ifstream file(test_log_file_);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    // Only Fatal should be logged
    EXPECT_EQ(content.find("[DEBUG]"), std::string::npos);
    EXPECT_EQ(content.find("[INFO]"), std::string::npos);
    EXPECT_EQ(content.find("[WARNING]"), std::string::npos);
    EXPECT_EQ(content.find("[ERROR]"), std::string::npos);
    EXPECT_NE(content.find("[FATAL]"), std::string::npos);
}

// Test 10: Generic log function
TEST_F(LoggingTest, GenericLogFunction) {
    auto& logger = Logger::instance();
    logger.set_output_file(test_log_file_);
    logger.enable_console(false);
    logger.set_level(LogLevel::Debug);

    logger.log(LogLevel::Debug, "Debug via log()");
    logger.log(LogLevel::Info, "Info via log()");
    logger.log(LogLevel::Warning, "Warning via log()");
    logger.log(LogLevel::Error, "Error via log()");
    logger.log(LogLevel::Fatal, "Fatal via log()");

    std::ifstream file(test_log_file_);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    EXPECT_NE(content.find("Debug via log()"), std::string::npos);
    EXPECT_NE(content.find("Info via log()"), std::string::npos);
    EXPECT_NE(content.find("Warning via log()"), std::string::npos);
    EXPECT_NE(content.find("Error via log()"), std::string::npos);
    EXPECT_NE(content.find("Fatal via log()"), std::string::npos);
}

// Test 11: Macros
//
// The overlapping severity names (TENZOR_LOG_DEBUG/INFO/ERROR) are owned
// exclusively by log.hpp's spdlog facade — they do NOT route through the legacy
// Logger or its output file. Only the legacy-only macros (WARNING / FATAL /
// WARN_ONCE) and direct Logger calls land in the Logger file. This deliberate
// split fixes the prior per-TU divergence where the same macro name routed to a
// different sink depending on include order.
TEST_F(LoggingTest, ConvenienceMacros) {
    auto& logger = Logger::instance();
    logger.set_output_file(test_log_file_);
    logger.enable_console(false);
    logger.set_level(LogLevel::Debug);

    // These route to spdlog (stderr/registry), not the Logger file — just
    // verify they compile and don't throw.
    TENZOR_LOG_DEBUG("Debug macro");
    TENZOR_LOG_INFO("Info macro");
    TENZOR_LOG_ERROR("Error macro");

    // These are legacy-only macros backed by the Logger file sink.
    TENZOR_LOG_WARNING("Warning macro");
    TENZOR_LOG_FATAL("Fatal macro");

    std::ifstream file(test_log_file_);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    // DEBUG/INFO/ERROR macros must NOT appear in the legacy Logger file —
    // they belong to the spdlog facade now.
    EXPECT_EQ(content.find("Debug macro"), std::string::npos);
    EXPECT_EQ(content.find("Info macro"), std::string::npos);
    EXPECT_EQ(content.find("Error macro"), std::string::npos);

    // Legacy-only macros do land in the Logger file.
    EXPECT_NE(content.find("Warning macro"), std::string::npos);
    EXPECT_NE(content.find("Fatal macro"), std::string::npos);
}
