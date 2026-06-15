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
 * callers that use `TENZOR_LOG_WARNING`, `TENZOR_LOG_FATAL`,
 * `TENZOR_WARN_ONCE`, and direct `Logger::instance()` access (none of which
 * `log.hpp` provides). It writes timestamped, uppercase-[LEVEL]-tagged lines
 * to std::cout and/or a configured output file — see `src/utils/logging.cpp`.
 *
 * The macro names that overlap with `log.hpp` (TENZOR_LOG_DEBUG / INFO /
 * ERROR) are NOT redefined here: they stay owned by `log.hpp`'s spdlog facade
 * so each has exactly one expansion target process-wide, regardless of include
 * order. (A previous version re-pointed them at `Logger`, which made the same
 * macro route to a different sink/format/level per translation unit.)
 */

#pragma once

#include "tenzor/utils/log.hpp"

#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <source_location>
#include <utility>

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
    std::ofstream file_stream_;
    std::mutex mutex_;
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

namespace detail {

/// Format a legacy log message. Mirrors spdlog's call contract: the first
/// argument is treated as a format string and the rest as fmt arguments.
/// A lone pre-formatted string (the common case for `TENZOR_LOG_INFO(s)`)
/// is returned verbatim, so callers passing a `std::format`/`std::string`
/// result with incidental braces are never re-parsed.
template <typename... Args>
inline auto legacy_log_format(std::string_view fmt_str, Args&&... args) -> std::string {
    if constexpr (sizeof...(Args) == 0) {
        return std::string(fmt_str);
    } else {
        return ::fmt::format(::fmt::runtime(fmt_str), std::forward<Args>(args)...);
    }
}

}  // namespace detail

// Legacy-only severity macros that the unified facade does not provide. These
// route through `tenzor::Logger` (std::cout / Logger::set_output_file with
// uppercase [LEVEL] tags) and have no name collision with `log.hpp`.
#define TENZOR_LOG_WARNING(msg) ::tenzor::Logger::instance().warning(msg)
#define TENZOR_LOG_FATAL(msg)   ::tenzor::Logger::instance().fatal(msg)

// IMPORTANT: do NOT redefine TENZOR_LOG_DEBUG / INFO / ERROR here. They are
// owned exclusively by `log.hpp` (the spdlog facade, included at the top of
// this header), so they expand to a single target process-wide. A previous
// version `#undef`'d and re-pointed them at `tenzor::Logger`, which made the
// same macro name route to a different sink/format/level depending on whether
// a TU transitively included this header — a per-TU divergence bug. Leaving
// them to log.hpp guarantees one expansion target for those three names
// regardless of include order. (WARNING/FATAL/WARN_ONCE are distinct names
// that log.hpp does not define, so they stay on the legacy Logger.)

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
