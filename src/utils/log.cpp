/**
 * @file log.cpp
 * @brief Implementation of the tenzor::utils::logger() singleton (D.1).
 *
 * Initialises a multi-sink spdlog logger ("tenzor") that always writes to
 * stderr, plus optionally tees to TENZOR_LOG_FILE when that env var is set.
 * The runtime level is taken from TENZOR_LOG_LEVEL (default: info).
 *
 * Initialisation is guarded by std::call_once so concurrent first-callers
 * don't race on the spdlog registry.
 */

#include "tenzor/utils/log.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace tenzor::utils {

namespace {

auto parse_level(const char* s) -> spdlog::level::level_enum {
    if (!s) return spdlog::level::info;
    std::string v = s;
    for (auto& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (v == "trace")    return spdlog::level::trace;
    if (v == "debug")    return spdlog::level::debug;
    if (v == "info")     return spdlog::level::info;
    if (v == "warn"  ||
        v == "warning")  return spdlog::level::warn;
    if (v == "error" ||
        v == "err")      return spdlog::level::err;
    if (v == "critical"||
        v == "crit")     return spdlog::level::critical;
    if (v == "off")      return spdlog::level::off;
    return spdlog::level::info;
}

}  // namespace

auto logger() -> std::shared_ptr<spdlog::logger> {
    static std::once_flag init_flag;
    static std::shared_ptr<spdlog::logger> lg;
    std::call_once(init_flag, []() {
        std::vector<spdlog::sink_ptr> sinks;
        sinks.push_back(std::make_shared<spdlog::sinks::stderr_color_sink_mt>());

        if (const char* path = std::getenv("TENZOR_LOG_FILE");
            path && std::strlen(path) > 0) {
            try {
                sinks.push_back(
                    std::make_shared<spdlog::sinks::basic_file_sink_mt>(
                        path, /*truncate=*/false));
            } catch (const std::exception&) {
                // Filesystem unwritable / permission denied — stderr sink
                // still works; ignore the file sink and continue.
            }
        }

        lg = std::make_shared<spdlog::logger>("tenzor",
                                              sinks.begin(), sinks.end());
        lg->set_level(parse_level(std::getenv("TENZOR_LOG_LEVEL")));
        lg->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v");
        lg->flush_on(spdlog::level::warn);

        // Register so other code can fetch it by name via spdlog::get("tenzor").
        try {
            spdlog::register_logger(lg);
        } catch (const spdlog::spdlog_ex&) {
            // Already registered (idempotent — fine).
        }
    });
    return lg;
}

}  // namespace tenzor::utils
