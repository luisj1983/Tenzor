#pragma once

#include <string>
#include <string_view>
#include <source_location>
#include <format>

namespace tenzor {

// Log levels
enum class LogLevel {
    Debug,
    Info,
    Warning,
    Error,
    Fatal
};

// Logger class
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

} // namespace tenzor
