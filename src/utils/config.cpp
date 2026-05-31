#include "tenzor/utils/config.hpp"

namespace tenzor {

auto Config::instance() -> Config& {
    static Config config;
    return config;
}

auto Config::get_bool(const std::string& key) const -> std::optional<bool> {
    auto it = config_.find(key);
    if (it != config_.end()) {
        return (it->second == "true" || it->second == "1");
    }
    return std::nullopt;
}

auto Config::set_bool(const std::string& key, bool value) -> void {
    config_[key] = value ? "true" : "false";
}

auto set_deterministic(bool deterministic) -> void {
    Config::instance().set_bool("deterministic", deterministic);
}

auto is_deterministic() -> bool {
    return Config::instance().get_bool("deterministic").value_or(false);
}

} // namespace tenzor
