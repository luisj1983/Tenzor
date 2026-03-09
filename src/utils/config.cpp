#include "tenzor/utils/config.hpp"
#include <fstream>
#include <sstream>

namespace tenzor {

auto Config::instance() -> Config& {
    static Config config;
    return config;
}

auto Config::get_string(const std::string& key) const -> std::optional<std::string> {
    auto it = config_.find(key);
    if (it != config_.end()) {
        return it->second;
    }
    return std::nullopt;
}

auto Config::get_int(const std::string& key) const -> std::optional<int64_t> {
    auto value = get_string(key);
    if (value) {
        return std::stoll(*value);
    }
    return std::nullopt;
}

auto Config::get_float(const std::string& key) const -> std::optional<double> {
    auto value = get_string(key);
    if (value) {
        return std::stod(*value);
    }
    return std::nullopt;
}

auto Config::get_bool(const std::string& key) const -> std::optional<bool> {
    auto value = get_string(key);
    if (value) {
        return (*value == "true" || *value == "1");
    }
    return std::nullopt;
}

auto Config::set_string(const std::string& key, const std::string& value) -> void {
    config_[key] = value;
}

auto Config::set_int(const std::string& key, int64_t value) -> void {
    config_[key] = std::to_string(value);
}

auto Config::set_float(const std::string& key, double value) -> void {
    config_[key] = std::to_string(value);
}

auto Config::set_bool(const std::string& key, bool value) -> void {
    config_[key] = value ? "true" : "false";
}

auto Config::load_from_file(const std::string& path) -> bool {
    std::ifstream file(path);
    if (!file) return false;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        auto pos = line.find('=');
        if (pos != std::string::npos) {
            auto key = line.substr(0, pos);
            auto value = line.substr(pos + 1);
            config_[key] = value;
        }
    }

    return true;
}

auto Config::save_to_file(const std::string& path) -> bool {
    std::ofstream file(path);
    if (!file) return false;

    for (const auto& [key, value] : config_) {
        file << key << "=" << value << "\n";
    }

    return true;
}

auto set_deterministic(bool deterministic) -> void {
    Config::instance().set_bool("deterministic", deterministic);
}

auto is_deterministic() -> bool {
    return Config::instance().get_bool("deterministic").value_or(false);
}

} // namespace tenzor
