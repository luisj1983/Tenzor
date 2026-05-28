/**
 * @file logging.cpp
 * @brief Stream 25 / Audit-12: deduplicated logger implementation.
 *
 * Historically Tenzor had two parallel logging implementations:
 *
 *   include/tenzor/utils/log.hpp     +  src/utils/log.cpp      (spdlog facade)
 *   include/tenzor/utils/logging.hpp +  src/utils/logging.cpp  (custom Logger)
 *
 * The two systems used different macro names (`TENZOR_LOG_WARN` vs
 * `TENZOR_LOG_WARNING`, `TENZOR_LOG_CRITICAL` vs `TENZOR_LOG_FATAL`), so
 * they did not actively conflict, but every diagnostic message was being
 * formatted twice — once by spdlog, once by the hand-rolled `Logger::log`
 * that wrote to `std::cout` with an `ostringstream` per call.
 *
 * The spdlog-based facade (`log.hpp`) is more featureful (fmt-style
 * variadic formatting, env-var-controlled levels, file/stderr sinks,
 * coloured terminal output, registry-based access). It is the survivor.
 *
 * `logging.hpp` and its `Logger` class continue to exist as a thin shim
 * routed through `log.hpp`, so the 22 callers using `TENZOR_LOG_WARNING`,
 * `TENZOR_LOG_FATAL`, `TENZOR_WARN_ONCE`, and direct `Logger::instance()`
 * keep compiling without churn. All output now flows through a single
 * spdlog logger.
 */

#include "tenzor/utils/logging.hpp"
#include "tenzor/utils/log.hpp"

#include <spdlog/sinks/basic_file_sink.h>

#include <string>

namespace tenzor {

namespace {

auto to_spdlog_level(LogLevel level) -> spdlog::level::level_enum {
    switch (level) {
        case LogLevel::Debug:   return spdlog::level::debug;
        case LogLevel::Info:    return spdlog::level::info;
        case LogLevel::Warning: return spdlog::level::warn;
        case LogLevel::Error:   return spdlog::level::err;
        case LogLevel::Fatal:   return spdlog::level::critical;
    }
    return spdlog::level::info;
}

}  // namespace

auto Logger::instance() -> Logger& {
    static Logger logger;
    return logger;
}

auto Logger::log(LogLevel level, std::string_view message,
                [[maybe_unused]] const std::source_location& location) -> void {
    if (level < level_) return;
    // Forward to the unified spdlog facade. The "%v" pattern in log.cpp
    // already prefixes timestamp + severity + logger name, so we just emit
    // the raw message payload.
    auto lg = ::tenzor::utils::logger();
    lg->log(to_spdlog_level(level), "{}", message);
}

auto Logger::debug(std::string_view message, const std::source_location& location) -> void {
    log(LogLevel::Debug, message, location);
}

auto Logger::info(std::string_view message, const std::source_location& location) -> void {
    log(LogLevel::Info, message, location);
}

auto Logger::warning(std::string_view message, const std::source_location& location) -> void {
    log(LogLevel::Warning, message, location);
}

auto Logger::error(std::string_view message, const std::source_location& location) -> void {
    log(LogLevel::Error, message, location);
}

auto Logger::fatal(std::string_view message, const std::source_location& location) -> void {
    log(LogLevel::Fatal, message, location);
}

auto Logger::set_level(LogLevel level) -> void {
    level_ = level;
    // Mirror onto the spdlog logger so env-driven filters and Logger-driven
    // filters stay in sync. The minimum of the two wins (spdlog's filter is
    // checked first; if it passes, our LogLevel `level_` is the second
    // gate inside Logger::log).
    ::tenzor::utils::logger()->set_level(to_spdlog_level(level));
}

auto Logger::get_level() const -> LogLevel {
    return level_;
}

auto Logger::set_output_file(std::string_view path) -> void {
    output_file_ = path;
    // Attach a file sink to the unified spdlog logger. Idempotent in spirit:
    // adding a sink with the same path twice will duplicate writes, so this
    // is best-effort and mostly used by tests.
    try {
        auto lg = ::tenzor::utils::logger();
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
            std::string(path), /*truncate=*/false);
        lg->sinks().push_back(std::move(file_sink));
    } catch (const std::exception&) {
        // Permission denied or path unreachable — silently fall back to the
        // existing sinks (stderr remains available).
    }
}

auto Logger::enable_console(bool enable) -> void {
    console_enabled_ = enable;
    // The unified spdlog logger always keeps a stderr sink (created in
    // log.cpp), so this flag is now an API-compat hint only. It is stored
    // but no longer drives output routing — disabling stderr globally would
    // affect every other consumer of the unified logger.
}

} // namespace tenzor
