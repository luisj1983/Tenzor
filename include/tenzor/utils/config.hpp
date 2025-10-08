#pragma once

#include <string>
#include <unordered_map>
#include <optional>

namespace tenzor {

// Runtime configuration
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
