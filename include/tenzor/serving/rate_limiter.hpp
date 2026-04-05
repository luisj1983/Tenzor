/**
 * @file rate_limiter.hpp
 * @brief Token bucket rate limiter for inference serving
 */
#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

namespace tenzor::serving {

struct RateLimitConfig {
    bool enabled{false};
    double requests_per_second{100.0};
    int32_t burst_size{200};
};

class TokenBucketRateLimiter {
public:
    explicit TokenBucketRateLimiter(RateLimitConfig config) : config_(config) {}

    /// Check if a request from client_id should be allowed
    auto allow(const std::string& client_id) -> bool {
        if (!config_.enabled) return true;

        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        auto& bucket = buckets_[client_id];

        // Initialize new buckets
        if (bucket.last_refill == std::chrono::steady_clock::time_point{}) {
            bucket.tokens = static_cast<double>(config_.burst_size);
            bucket.last_refill = now;
        }

        // Refill tokens based on elapsed time
        auto elapsed = std::chrono::duration<double>(now - bucket.last_refill).count();
        bucket.tokens = std::min(
            static_cast<double>(config_.burst_size),
            bucket.tokens + elapsed * config_.requests_per_second);
        bucket.last_refill = now;

        if (bucket.tokens < 1.0) return false;
        bucket.tokens -= 1.0;
        return true;
    }

private:
    struct Bucket {
        double tokens{0};
        std::chrono::steady_clock::time_point last_refill{};
    };

    RateLimitConfig config_;
    std::mutex mutex_;
    std::unordered_map<std::string, Bucket> buckets_;
};

} // namespace tenzor::serving
