#include "tenzor/utils/logging.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>

namespace tenzor {

auto Logger::instance() -> Logger& {
    static Logger logger;
    return logger;
}

auto Logger::log(LogLevel level, std::string_view message,
                [[maybe_unused]] const std::source_location& location) -> void {
    if (level < level_) return;

    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);

    std::ostringstream oss;
    oss << "[" << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << "] ";

    switch (level) {
        case LogLevel::Debug: oss << "[DEBUG] "; break;
        case LogLevel::Info: oss << "[INFO] "; break;
        case LogLevel::Warning: oss << "[WARNING] "; break;
        case LogLevel::Error: oss << "[ERROR] "; break;
        case LogLevel::Fatal: oss << "[FATAL] "; break;
    }

    oss << message;

    if (console_enabled_) {
        std::cout << oss.str() << std::endl;
    }

    if (!output_file_.empty()) {
        std::ofstream file(output_file_, std::ios::app);
        file << oss.str() << std::endl;
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
    level_ = level;
}

auto Logger::get_level() const -> LogLevel {
    return level_;
}

auto Logger::set_output_file(std::string_view path) -> void {
    output_file_ = path;
}

auto Logger::enable_console(bool enable) -> void {
    console_enabled_ = enable;
}

} // namespace tenzor
