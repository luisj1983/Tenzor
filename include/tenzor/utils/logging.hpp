/**
 * @file logging.hpp
 * @brief Compatibility shim over the unified spdlog facade (`log.hpp`).
 *
 * Stream 25 / Audit-12 deduplicated the two parallel logging stacks that
 * historically coexisted in Tenzor:
 *
 *   - `tenzor/utils/log.hpp`     (modern, spdlog-based, `TENZOR_LOG_WARN` etc.)
 *   - `tenzor/utils/logging.hpp` (legacy, custom Logger class)
 *
 * The legacy `Logger` class still exists for source-compatibility with the
 * 22 callers that use `TENZOR_LOG_WARNING`, `TENZOR_LOG_FATAL`, and
 * `TENZOR_WARN_ONCE` (none of which `log.hpp` provides). Its methods are
 * routed through the unified spdlog logger — see `src/utils/logging.cpp`.
 *
 * The legacy macro names that overlap with `log.hpp` (TENZOR_LOG_DEBUG /
 * INFO / ERROR) now forward directly to the spdlog facade so there is only
 * one definition site each.
 */

#pragma once

#include "tenzor/utils/log.hpp"

#include <mutex>
#include <string>
#include <string_view>
#include <source_location>

namespace tenzor {

/**
 * @brief Logging severity levels (compat with the legacy Logger API).
 */
enum class LogLevel {
    Debug,    ///< Detailed debugging information
    Info,     ///< General informational messages
    Warning,  ///< Warning messages
    Error,    ///< Error messages
    Fatal     ///< Fatal errors
};

/**
 * @brief Singleton logger — compat facade over `tenzor::utils::logger()`.
 *
 * Stream 25: the implementation forwards to spdlog; this class exists for
 * source-compat with callers that took a reference to `Logger::instance()`
 * and called `set_level`, `set_output_file`, `enable_console`, or the
 * per-severity methods (`warning(msg)`, `fatal(msg)`, ...).
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

// ---------------------------------------------------------------------------
// Macros.
//
// The overlapping severities (DEBUG/INFO/ERROR) are now defined exclusively
// in `log.hpp` — including this header (which #includes log.hpp at the top)
// already brings them in. We only define the legacy-only macros here:
// WARNING, FATAL, and WARN_ONCE. They are routed through `tenzor::Logger`
// so the existing legacy callers keep working unchanged.
// ---------------------------------------------------------------------------

// String-only severity macros that the unified facade does not provide.
#define TENZOR_LOG_WARNING(msg) ::tenzor::Logger::instance().warning(msg)
#define TENZOR_LOG_FATAL(msg)   ::tenzor::Logger::instance().fatal(msg)

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
