/**
 * @file config.hpp
 * @brief Runtime configuration management
 *
 * Provides global configuration system for library behavior customization.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <optional>

namespace tenzor {

/**
 * @brief Singleton configuration manager
 *
 * Manages runtime configuration settings for the library.
 *
 * @par Scope
 * The production surface is intentionally minimal: only the boolean-flag
 * accessors needed by the global deterministic-mode helpers
 * (`set_deterministic`/`is_deterministic`) live here. The previous generic
 * string/int/float get/set API plus `load_from_file`/`save_to_file` had no
 * production callers and were relocated into the unit test
 * (`tests/utils/test_config.cpp`).
 *
 * @par Thread Safety
 * Not thread-safe. Configure before parallel operations begin.
 *
 * @code
 * auto& config = Config::instance();
 * config.set_bool("deterministic", true);
 * @endcode
 */
class Config {
public:
    static auto instance() -> Config&;

    auto get_bool(const std::string& key) const -> std::optional<bool>;
    auto set_bool(const std::string& key, bool value) -> void;

private:
    Config() = default;

    std::unordered_map<std::string, std::string> config_;
};

/// Set deterministic mode globally
auto set_deterministic(bool deterministic) -> void;

/// Check if deterministic mode is enabled
auto is_deterministic() -> bool;

} // namespace tenzor
