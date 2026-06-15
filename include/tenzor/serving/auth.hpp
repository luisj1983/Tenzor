/**
 * @file auth.hpp
 * @brief API key authentication middleware for inference serving
 */
#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <cstddef>

namespace tenzor::serving {

struct AuthConfig {
    bool enabled{false};
    std::vector<std::string> api_keys;
    std::string header_name{"Authorization"};
};

/// Constant-time string comparison: the running time depends only on the input
/// lengths, never on how many leading bytes match. Mirrors the server's live
/// auth path (server.cpp ct_eq) so the two implementations cannot diverge.
inline bool ct_eq(const std::string& a, const std::string& b) {
    unsigned diff = static_cast<unsigned>(a.size() ^ b.size());
    std::size_t n = std::max(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        unsigned ca = i < a.size() ? static_cast<unsigned char>(a[i]) : 0u;
        unsigned cb = i < b.size() ? static_cast<unsigned char>(b[i]) : 0u;
        diff |= ca ^ cb;
    }
    return diff == 0;
}

/// Validate a Bearer token against the configured API keys.
///
/// Uses a constant-time comparison against EVERY configured key (no early break)
/// so the response time does not leak how many leading bytes of a guessed key
/// are correct — matching the server's live auth path rather than the
/// short-circuiting std::find it previously used.
inline bool validate_token(const AuthConfig& config, const std::string& header_value) {
    if (!config.enabled || config.api_keys.empty()) return true;

    std::string token = header_value;
    if (token.size() > 7 && token.substr(0, 7) == "Bearer ") {
        token = token.substr(7);
    }
    bool valid = false;
    for (const auto& key : config.api_keys) {
        valid |= ct_eq(token, key);
    }
    return valid;
}

} // namespace tenzor::serving
