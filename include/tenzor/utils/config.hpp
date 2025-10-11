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
 * Settings can be loaded from files or set programmatically.
 *
 * **Common Settings:**
 * - "num_threads": Number of CPU threads
 * - "device": Default device ("cpu", "cuda")
 * - "cudnn_benchmark": Enable cuDNN benchmarking
 * - "deterministic": Enable deterministic operations
 *
 * @par Thread Safety
 * Not thread-safe. Configure before parallel operations begin.
 *
 * @code
 * auto& config = Config::instance();
 * config.set_int("num_threads", 4);
 * config.set_bool("deterministic", true);
 * @endcode
 */
class Config {
public:
    static auto instance() -> Config&;

    // Get configuration values
    auto get_string(const std::string& key) const -> std::optional<std::string>;
    auto get_int(const std::string& key) const -> std::optional<int64_t>;
    auto get_float(const std::string& key) const -> std::optional<double>;
    auto get_bool(const std::string& key) const -> std::optional<bool>;

    // Set configuration values
    auto set_string(const std::string& key, const std::string& value) -> void;
    auto set_int(const std::string& key, int64_t value) -> void;
    auto set_float(const std::string& key, double value) -> void;
    auto set_bool(const std::string& key, bool value) -> void;

    // Load from file
    auto load_from_file(const std::string& path) -> bool;

    // Save to file
    auto save_to_file(const std::string& path) -> bool;

private:
    Config() = default;

    std::unordered_map<std::string, std::string> config_;
};

} // namespace tenzor
