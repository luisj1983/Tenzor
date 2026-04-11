/**
 * @file logging.hpp
 * @brief Logging system for debugging and diagnostics
 *
 * Provides structured logging with multiple severity levels,
 * automatic source location tracking, and configurable outputs.
 */

#pragma once

#include <mutex>
#include <string>
#include <string_view>
#include <source_location>
#include <format>

namespace tenzor {

/**
 * @brief Logging severity levels
 *
 * - Debug: Detailed debugging information
 * - Info: General informational messages
 * - Warning: Warning messages (non-critical issues)
 * - Error: Error messages (recoverable errors)
 * - Fatal: Fatal errors (program termination)
 */
enum class LogLevel {
    Debug,    ///< Detailed debugging information
    Info,     ///< General informational messages
    Warning,  ///< Warning messages
    Error,    ///< Error messages
    Fatal     ///< Fatal errors
};

/**
 * @brief Singleton logger for library diagnostics
 *
 * Provides structured logging with automatic source location tracking.
 * Supports console and file output with configurable log levels.
 *
 * **Default Behavior:**
 * - Log level: Info (Debug messages filtered)
 * - Console output: Enabled
 * - File output: Disabled
 *
 * @par Thread Safety
 * Thread-safe for concurrent logging from multiple threads
 *
 * @code
 * auto& logger = Logger::instance();
 * logger.set_level(LogLevel::Debug);
 * logger.set_output_file("tenzor.log");
 *
 * TENZOR_LOG_INFO("Model loaded successfully");
 * TENZOR_LOG_WARNING("Using CPU fallback");
 * @endcode
 */
class Logger {
public:
    static auto instance() -> Logger&;

    // Log functions
    auto log(LogLevel level,
            std::string_view message,
            const std::source_location& location = std::source_location::current()) -> void;

    auto debug(std::string_view message,
              const std::source_location& location = std::source_location::current()) -> void;

    auto info(std::string_view message,
             const std::source_location& location = std::source_location::current()) -> void;

    auto warning(std::string_view message,
                const std::source_location& location = std::source_location::current()) -> void;

    auto error(std::string_view message,
              const std::source_location& location = std::source_location::current()) -> void;

    auto fatal(std::string_view message,
              const std::source_location& location = std::source_location::current()) -> void;

    // Configuration
    auto set_level(LogLevel level) -> void;
    auto get_level() const -> LogLevel;

    auto set_output_file(std::string_view path) -> void;
    auto enable_console(bool enable) -> void;

private:
    Logger() = default;

    LogLevel level_{LogLevel::Info};
    bool console_enabled_{true};
    std::string output_file_;
};

// Convenience macros
#define TENZOR_LOG_DEBUG(msg) ::tenzor::Logger::instance().debug(msg)
#define TENZOR_LOG_INFO(msg) ::tenzor::Logger::instance().info(msg)
#define TENZOR_LOG_WARNING(msg) ::tenzor::Logger::instance().warning(msg)
#define TENZOR_LOG_ERROR(msg) ::tenzor::Logger::instance().error(msg)
#define TENZOR_LOG_FATAL(msg) ::tenzor::Logger::instance().fatal(msg)

// Emit a warning at most once for a given call site over the lifetime of
// the process. Safe under concurrent calls (std::call_once is thread-safe).
// The flag is a function-local static so the macro is self-contained and
// requires no header for <mutex> at the call site.
#define TENZOR_WARN_ONCE(msg)                                           \
    do {                                                                \
        static ::std::once_flag _tenzor_warn_once_flag;                 \
        ::std::call_once(_tenzor_warn_once_flag, []() {                 \
            ::tenzor::Logger::instance().warning(msg);                  \
        });                                                             \
    } while (0)

} // namespace tenzor
