/**
 * @file autotune.cpp
 * @brief Implementation of kernel autotuning cache
 */

#include "../../include/tenzor/jit/autotune.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <thread>
#include <sstream>
#include <stdexcept>

namespace tenzor {
namespace jit {

// ============================================================================
// Singleton
// ============================================================================

auto AutotuneCache::instance() -> AutotuneCache& {
    static AutotuneCache cache;
    return cache;
}

// ============================================================================
// Lookup
// ============================================================================

auto AutotuneCache::lookup(const std::string& key, int num_candidates) const
    -> std::optional<int> {
    std::shared_lock lock(mutex_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        return std::nullopt;
    }
    int id = it->second.algorithm_id;
    // Reject a cached id that cannot address the current candidate list. A cache
    // autotuned on another architecture (or an older build with more candidates)
    // could otherwise return an id that indexes past the current candidates into
    // an invalid/undefined kernel — a correctness bug, not merely a slow choice.
    if (num_candidates >= 0 && (id < 0 || id >= num_candidates)) {
        return std::nullopt;
    }
    return id;
}

// ============================================================================
// Record
// ============================================================================

auto AutotuneCache::record(const std::string& key, int algorithm_id, double time_ms) -> void {
    // Reject non-positive timings. A degenerate/failed benchmark reporting 0.0
    // (or a negative value) would otherwise become permanently sticky — no
    // genuinely-measured positive time could ever displace it — and would be
    // persisted across restarts via save()/load().
    if (!(time_ms > 0.0)) return;  // also rejects NaN
    std::unique_lock lock(mutex_);
    auto it = cache_.find(key);
    if (it == cache_.end() || time_ms < it->second.time_ms) {
        cache_[key] = CacheEntry{algorithm_id, time_ms};
    }
}

// ============================================================================
// Clear / Size
// ============================================================================

auto AutotuneCache::clear() -> void {
    std::unique_lock lock(mutex_);
    cache_.clear();
}

auto AutotuneCache::size() const -> size_t {
    std::shared_lock lock(mutex_);
    return cache_.size();
}

// ============================================================================
// Default Path
// ============================================================================

auto AutotuneCache::default_path() -> std::string {
    std::string home;
    if (const char* h = std::getenv("HOME")) {
        home = h;
    } else if (const char* h2 = std::getenv("USERPROFILE")) {
        home = h2;
    } else {
        home = ".";
    }
    return home + "/.tenzor/autotune_cache.json";
}

// ============================================================================
// Save to JSON
// ============================================================================

auto AutotuneCache::save(const std::string& path) const -> void {
    std::shared_lock lock(mutex_);

    // Ensure parent directory exists
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }

    // Write to a temp file first, then rename for atomicity. The temp name must
    // be unique per thread: two threads calling save() concurrently both hold
    // only a shared_lock, so a fixed "path.tmp" would be written by both at once
    // and truncate/interleave before the rename, corrupting the file (JIT-053).
    // A per-thread temp name lets each write independently; the final atomic
    // rename to `path` is last-writer-wins (both wrote valid content).
    const std::string tmp_path = path + ".tmp." +
        std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    try {
        std::ofstream ofs(tmp_path);
        if (!ofs) {
            throw std::runtime_error("AutotuneCache::save: cannot open " + tmp_path);
        }

        ofs << "{\n";
        bool first = true;
        for (const auto& [key, entry] : cache_) {
            if (!first) {
                ofs << ",\n";
            }
            first = false;

            // Escape key for JSON (simple: replace \ and " )
            std::string escaped_key;
            escaped_key.reserve(key.size());
            for (char c : key) {
                if (c == '"') {
                    escaped_key += "\\\"";
                } else if (c == '\\') {
                    escaped_key += "\\\\";
                } else {
                    escaped_key += c;
                }
            }

            ofs << "  \"" << escaped_key << "\": {"
                << "\"algorithm_id\": " << entry.algorithm_id
                << ", \"time_ms\": " << entry.time_ms
                << "}";
        }
        ofs << "\n}\n";
        ofs.flush();
        ofs.close();
        // Atomic rename (inside the try so a failure also cleans up the temp).
        std::filesystem::rename(tmp_path, path);
    } catch (...) {
        // Remove the (per-thread) temp so a write/rename failure does not leave
        // an orphan next to the cache — best-effort, never masks the real error.
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
        throw;
    }
}

// ============================================================================
// Load from JSON
// ============================================================================

auto AutotuneCache::load(const std::string& path) -> void {
    std::ifstream ifs(path);
    if (!ifs) {
        // File doesn't exist yet - not an error
        return;
    }

    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());

    std::unique_lock lock(mutex_);
    cache_.clear();

    // Minimal JSON parser for our specific format:
    // { "key": {"algorithm_id": N, "time_ms": F}, ... }
    size_t pos = 0;
    auto skip_ws = [&]() {
        while (pos < content.size() &&
               std::isspace(static_cast<unsigned char>(content[pos]))) ++pos;
    };

    auto expect_char = [&](char c) -> bool {
        skip_ws();
        if (pos < content.size() && content[pos] == c) {
            ++pos;
            return true;
        }
        return false;
    };

    auto parse_string = [&]() -> std::string {
        skip_ws();
        if (pos >= content.size() || content[pos] != '"') return "";
        ++pos;
        std::string result;
        while (pos < content.size() && content[pos] != '"') {
            if (content[pos] == '\\' && pos + 1 < content.size()) {
                ++pos;
                result += content[pos];
            } else {
                result += content[pos];
            }
            ++pos;
        }
        if (pos < content.size()) ++pos;  // skip closing "
        return result;
    };

    auto parse_number = [&]() -> double {
        skip_ws();
        size_t start = pos;
        while (pos < content.size() &&
               (std::isdigit(static_cast<unsigned char>(content[pos])) ||
                content[pos] == '.' ||
                content[pos] == '-' || content[pos] == 'e' || content[pos] == 'E' ||
                content[pos] == '+')) {
            ++pos;
        }
        if (start == pos) return 0.0;
        // The character filter above admits substrings that are not valid
        // doubles (e.g. "-", ".", "e", "+", "1e", "--"); std::stod would throw
        // on those. A corrupted/truncated cache file must degrade to an empty
        // cache rather than crash, so swallow the parse error and treat the
        // value as 0.0.
        try {
            return std::stod(content.substr(start, pos - start));
        } catch (const std::exception&) {
            return 0.0;
        }
    };

    if (!expect_char('{')) return;

    while (true) {
        skip_ws();
        if (pos >= content.size() || content[pos] == '}') break;

        // Parse key
        std::string key = parse_string();
        if (key.empty()) break;

        if (!expect_char(':')) break;
        if (!expect_char('{')) break;

        // Parse inner object
        int algo_id = 0;
        double time = 0.0;

        for (int field = 0; field < 2; ++field) {
            std::string field_name = parse_string();
            expect_char(':');
            double val = parse_number();

            if (field_name == "algorithm_id") {
                algo_id = static_cast<int>(val);
            } else if (field_name == "time_ms") {
                time = val;
            }

            skip_ws();
            if (pos < content.size() && content[pos] == ',') ++pos;
        }

        expect_char('}');

        // Apply the SAME validity contract as record(): a non-positive/NaN time
        // (parse_number coerces a malformed/truncated field to 0.0) would install
        // a permanently-sticky entry that no genuinely-measured positive time can
        // ever displace, and a negative algorithm_id could index an invalid
        // candidate list. Skip such entries rather than loading a poisoned cache.
        if (time > 0.0 && algo_id >= 0) {
            cache_[key] = CacheEntry{algo_id, time};
        }

        skip_ws();
        if (pos < content.size() && content[pos] == ',') ++pos;
    }
}

// ============================================================================
// Default save/load
// ============================================================================

auto AutotuneCache::save_default() const -> void {
    save(default_path());
}

auto AutotuneCache::load_default() -> void {
    load(default_path());
}

// ============================================================================
// Key Builder
// ============================================================================

auto AutotuneCache::make_key(const std::string& op_name,
                              const std::string& dtype,
                              const std::string& device_arch,
                              const std::vector<std::vector<int64_t>>& shapes) -> std::string {
    std::ostringstream oss;
    // Include the device ARCHITECTURE in the key: this cache is persisted
    // (save/load) and served across machines/GPUs, so a key without the true
    // arch would return a config autotuned for a different architecture. The
    // caller must pass an arch-qualified identifier (see the header contract);
    // an index-only "cuda:0" would alias distinct GPU generations.
    oss << op_name << ":" << dtype << ":" << device_arch;

    for (size_t i = 0; i < shapes.size(); ++i) {
        oss << ":";
        const auto& shape = shapes[i];
        for (size_t j = 0; j < shape.size(); ++j) {
            if (j > 0) oss << "x";
            oss << shape[j];
        }
    }

    return oss.str();
}

// ============================================================================
// Autotune-mode propagation (R1-11)
// ============================================================================

namespace {
thread_local bool g_autotune_mode_active = false;
}

auto autotune_mode_active() -> bool {
    return g_autotune_mode_active;
}

AutotuneModeGuard::AutotuneModeGuard(bool enable) : previous_(g_autotune_mode_active) {
    g_autotune_mode_active = enable;
}

AutotuneModeGuard::~AutotuneModeGuard() {
    g_autotune_mode_active = previous_;
}

} // namespace jit
} // namespace tenzor
