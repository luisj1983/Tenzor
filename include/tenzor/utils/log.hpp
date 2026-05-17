/**
 * @file log.hpp
 * @brief Tenzor structured logging facade over spdlog (D.1).
 *
 * Provides a single global logger named "tenzor" with level macros:
 *   TENZOR_LOG_TRACE / DEBUG / INFO / WARN / ERROR / CRITICAL.
 *
 * Sinks default to stderr; set the env var TENZOR_LOG_FILE=/path/to/file
 * before first use to also tee to a file. Set TENZOR_LOG_LEVEL=debug to
 * enable lower severities at runtime (default: info).
 *
 * Usage:
 *   #include "tenzor/utils/log.hpp"
 *   TENZOR_LOG_WARN("collective failed on rank {}: {}", rank, e.what());
 *
 * The macros expand to a single function call when severity is enabled
 * and to nothing when compiled out — same cost model as spdlog directly.
 */

#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>
#include <memory>

namespace tenzor::utils {

/**
 * @brief Get the global "tenzor" logger.
 *
 * Lazily initialised on first call (thread-safe via spdlog's registry).
 * Routes to stderr by default; respects TENZOR_LOG_FILE and TENZOR_LOG_LEVEL
 * env vars at first-call time.
 */
auto logger() -> std::shared_ptr<spdlog::logger>;

}  // namespace tenzor::utils

// Severity macros — argument forwarding through spdlog's compile-time
// format checker. Calls are zero-cost when the logger's level filters
// them out.
#define TENZOR_LOG_TRACE(...)    ::tenzor::utils::logger()->trace(__VA_ARGS__)
#define TENZOR_LOG_DEBUG(...)    ::tenzor::utils::logger()->debug(__VA_ARGS__)
#define TENZOR_LOG_INFO(...)     ::tenzor::utils::logger()->info(__VA_ARGS__)
#define TENZOR_LOG_WARN(...)     ::tenzor::utils::logger()->warn(__VA_ARGS__)
#define TENZOR_LOG_ERROR(...)    ::tenzor::utils::logger()->error(__VA_ARGS__)
#define TENZOR_LOG_CRITICAL(...) ::tenzor::utils::logger()->critical(__VA_ARGS__)
