/**
 * @file auth.hpp
 * @brief API key authentication middleware for inference serving
 */
#pragma once

#include <string>
#include <vector>
#include <algorithm>

namespace tenzor::serving {

struct AuthConfig {
    bool enabled{false};
    std::vector<std::string> api_keys;
    std::string header_name{"Authorization"};
};

/// Validate a Bearer token against the configured API keys
inline bool validate_token(const AuthConfig& config, const std::string& header_value) {
    if (!config.enabled || config.api_keys.empty()) return true;

    std::string token = header_value;
    if (token.size() > 7 && token.substr(0, 7) == "Bearer ") {
        token = token.substr(7);
    }
    return std::find(config.api_keys.begin(), config.api_keys.end(), token)
           != config.api_keys.end();
}

} // namespace tenzor::serving
