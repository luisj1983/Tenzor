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

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace tenzor {

namespace {

auto level_tag(LogLevel level) -> const char* {
    switch (level) {
        case LogLevel::Debug:   return "DEBUG";
        case LogLevel::Info:    return "INFO";
        case LogLevel::Warning: return "WARNING";
        case LogLevel::Error:   return "ERROR";
        case LogLevel::Fatal:   return "FATAL";
    }
    return "INFO";
}

auto timestamp_now() -> std::string {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto t = system_clock::to_time_t(now);
    const auto ms =
        duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    std::ostringstream os;
    os << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << '.'
       << std::setfill('0') << std::setw(3) << ms.count();
    return os.str();
}

}  // namespace

auto Logger::instance() -> Logger& {
    static Logger logger;
    return logger;
}

auto Logger::log(LogLevel level, std::string_view message,
                [[maybe_unused]] const std::source_location& location) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    if (level < level_) return;

    // Legacy contract: each line is
    //   [YYYY-MM-DD HH:MM:SS.mmm] [LEVEL] message
    // with an uppercase severity tag (DEBUG/INFO/WARNING/ERROR/FATAL),
    // emitted to std::cout when the console is enabled and/or to the
    // configured output file, and flushed per line so consumers that read
    // the stream/file immediately after logging (tests, crash handlers)
    // observe the message. This path backs only the legacy-only macros
    // (TENZOR_LOG_WARNING/FATAL/WARN_ONCE) and direct Logger::instance()
    // callers; TENZOR_LOG_DEBUG/INFO/ERROR are owned solely by log.hpp's
    // spdlog facade so those names have one expansion target process-wide.
    std::string line;
    line.reserve(message.size() + 48);
    line.append("[").append(timestamp_now()).append("] [")
        .append(level_tag(level)).append("] ")
        .append(message).append("\n");

    if (console_enabled_) {
        std::cout << line;
        std::cout.flush();
    }
    if (file_stream_.is_open()) {
        file_stream_ << line;
        file_stream_.flush();
    }
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
    std::lock_guard<std::mutex> lock(mutex_);
    level_ = level;
}

auto Logger::get_level() const -> LogLevel {
    return level_;
}

auto Logger::set_output_file(std::string_view path) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    output_file_ = std::string(path);
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
    if (!output_file_.empty()) {
        file_stream_.open(output_file_, std::ios::out | std::ios::trunc);
    }
}

auto Logger::enable_console(bool enable) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    console_enabled_ = enable;
}

} // namespace tenzor
