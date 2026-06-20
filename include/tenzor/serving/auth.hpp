/**
 * @file auth.hpp
 * @brief API key authentication middleware for inference serving
 */
#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

// When the serving target is linked against OpenSSL (TENZOR_SERVING_HAS_OPENSSL),
// token comparison is performed on fixed-length HMAC-SHA256 digests. This makes
// the comparison work independent of BOTH the attacker-supplied token length and
// the configured key length: every comparison hashes its inputs down to exactly
// 32 bytes before the constant-time compare, so neither a matching prefix nor a
// length difference is observable via timing. Without OpenSSL the header falls
// back to a length-folding constant-time byte compare (still safe against
// prefix-timing; only the length of the longer operand is in principle
// observable, which is the documented limitation of the fallback).
#if defined(TENZOR_SERVING_HAS_OPENSSL)
#include <openssl/hmac.h>
#include <openssl/evp.h>
#endif

namespace tenzor::serving {

struct AuthConfig {
    bool enabled{false};
    std::vector<std::string> api_keys;
    std::string header_name{"Authorization"};
};

/// Constant-time comparison of two fixed-length byte buffers of equal size.
/// The running time depends only on the (compile-time-known) length N and never
/// on the contents, so no information about a matching prefix is leaked.
inline bool ct_eq_fixed(const unsigned char* a, const unsigned char* b, std::size_t n) {
    unsigned diff = 0u;
    for (std::size_t i = 0; i < n; ++i) {
        diff |= static_cast<unsigned>(a[i] ^ b[i]);
    }
    return diff == 0;
}

#if defined(TENZOR_SERVING_HAS_OPENSSL)
/// HMAC-SHA256 of `msg` under a fixed library key, written into a 32-byte digest.
/// The HMAC key here is NOT a secret — it only serves to map arbitrary-length
/// inputs to a fixed-length 32-byte value so the subsequent compare runs in time
/// independent of the input length. Both the candidate token and each configured
/// key are passed through the identical transform before comparison.
inline std::array<unsigned char, 32> auth_digest(const std::string& msg) {
    static constexpr unsigned char kHmacKey[] = "tenzor.serving.auth.v1";
    std::array<unsigned char, 32> out{};
    unsigned int out_len = 0;
    ::HMAC(::EVP_sha256(), kHmacKey, static_cast<int>(sizeof(kHmacKey) - 1),
           reinterpret_cast<const unsigned char*>(msg.data()), msg.size(),
           out.data(), &out_len);
    return out;
}
#endif

/// Constant-time string comparison. Mirrors the server's live auth path so the
/// two implementations cannot diverge.
///
/// With OpenSSL available, both operands are first reduced to fixed-length
/// HMAC-SHA256 digests and the 32-byte digests are compared in constant time, so
/// the comparison work is independent of both the attacker-supplied token length
/// and the configured key length. Without OpenSSL it folds the length difference
/// into the running diff and loops over the longer operand; this remains safe
/// against prefix-timing attacks.
inline bool ct_eq(const std::string& a, const std::string& b) {
#if defined(TENZOR_SERVING_HAS_OPENSSL)
    const auto da = auth_digest(a);
    const auto db = auth_digest(b);
    return ct_eq_fixed(da.data(), db.data(), da.size());
#else
    unsigned diff = static_cast<unsigned>(a.size() ^ b.size());
    std::size_t n = std::max(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        unsigned ca = i < a.size() ? static_cast<unsigned char>(a[i]) : 0u;
        unsigned cb = i < b.size() ? static_cast<unsigned char>(b[i]) : 0u;
        diff |= ca ^ cb;
    }
    return diff == 0;
#endif
}

/// Strip a leading "Bearer " prefix from an Authorization header value.
/// substr(0,7) is well-defined on strings shorter than 7 chars (it yields the
/// whole string), so the comparison is safe without a length guard. Shared by
/// validate_token and the server's live auth path so both treat the exact
/// "Bearer " (7-char) boundary identically.
inline std::string strip_bearer(const std::string& header_value) {
    if (header_value.substr(0, 7) == "Bearer ") {
        return header_value.substr(7);
    }
    return header_value;
}

/// Validate a Bearer token against the configured API keys.
///
/// Uses a constant-time comparison against EVERY configured key (no early break)
/// so the response time does not leak how many leading bytes of a guessed key
/// are correct — matching the server's live auth path rather than the
/// short-circuiting std::find it previously used.
///
/// Secure default: when auth is *enabled* the request must present a valid key.
/// If auth is enabled but no keys are configured this is an operator
/// misconfiguration, and we FAIL CLOSED (deny every request) rather than fail
/// open — an enabled-but-keyless server must never silently accept traffic. Only
/// when auth is explicitly disabled do we short-circuit to allow.
inline bool validate_token(const AuthConfig& config, const std::string& header_value) {
    if (!config.enabled) return true;       // auth disabled: allow
    if (config.api_keys.empty()) return false;  // enabled but misconfigured: deny-all

    // Strip the "Bearer " prefix whenever present. substr(0,7) is safe on
    // shorter strings (returns the whole string), so no length guard is needed —
    // and adding one would diverge from server.cpp, which strips for exactly
    // "Bearer " (7 chars) too.
    std::string token = strip_bearer(header_value);
    bool valid = false;
    for (const auto& key : config.api_keys) {
        valid |= ct_eq(token, key);
    }
    return valid;
}

} // namespace tenzor::serving
