/**
 * @file rate_limiter.hpp
 * @brief Token bucket rate limiter for inference serving
 */
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

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

        // Bound memory: a client spoofing many distinct client_ids would
        // otherwise grow this map without limit (memory-exhaustion DoS). When
        // the map gets large, evict buckets idle past a TTL — they will have
        // fully refilled anyway, so eviction is behaviour-neutral. Matches the
        // server's live path (server.cpp) so the two cannot drift.
        if (buckets_.size() > kMaxBuckets) {
            const auto ttl = std::chrono::seconds(300);
            for (auto it = buckets_.begin(); it != buckets_.end();) {
                if (now - it->second.last_refill > ttl) {
                    it = buckets_.erase(it);
                } else {
                    ++it;
                }
            }
            // The TTL sweep only reclaims *idle* buckets. An attacker issuing one
            // request per distinct client_id every <TTL keeps every bucket
            // "recently refilled", so the sweep frees nothing and the map grows
            // without bound — an active-id memory-exhaustion DoS. Enforce a hard
            // absolute cap: if the map is still over kHardMaxBuckets after the
            // sweep, evict the least-recently-refilled (oldest last_refill)
            // entries until back under the cap. Evicting a stale bucket only
            // resets its tokens to full burst, so it is conservative — it can
            // never let a throttled client exceed its rate.
            if (buckets_.size() > kHardMaxBuckets) {
                std::vector<std::chrono::steady_clock::time_point> refills;
                refills.reserve(buckets_.size());
                for (const auto& kv : buckets_) {
                    refills.push_back(kv.second.last_refill);
                }
                const std::size_t to_evict = buckets_.size() - kHardMaxBuckets;
                std::nth_element(refills.begin(), refills.begin() + to_evict,
                                 refills.end());
                const auto cutoff = refills[to_evict];
                std::size_t evicted = 0;
                for (auto it = buckets_.begin();
                     it != buckets_.end() && evicted < to_evict;) {
                    if (it->second.last_refill < cutoff) {
                        it = buckets_.erase(it);
                        ++evicted;
                    } else {
                        ++it;
                    }
                }
                // Ties at exactly `cutoff` may remain; remove them too until the
                // map is bounded so the absolute cap always holds.
                for (auto it = buckets_.begin();
                     it != buckets_.end() && buckets_.size() > kHardMaxBuckets;) {
                    it = buckets_.erase(it);
                }
            }
        }

        auto& bucket = buckets_[client_id];

        // Initialize new buckets. Match the server's live guard exactly
        // (tokens == 0 AND last_refill unset) so the two implementations of this
        // security-sensitive feature cannot drift.
        if (bucket.tokens == 0 &&
            bucket.last_refill == std::chrono::steady_clock::time_point{}) {
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

    /// Map-size threshold above which idle buckets are swept (matches server).
    static constexpr std::size_t kMaxBuckets = 10000;
    /// Absolute upper bound on tracked buckets. After the idle-TTL sweep, the
    /// least-recently-refilled entries are evicted down to this cap so the map
    /// is bounded even under an all-active-client flood. Sits above kMaxBuckets
    /// so normal traffic never triggers LRU eviction.
    static constexpr std::size_t kHardMaxBuckets = 100000;

    RateLimitConfig config_;
    std::mutex mutex_;
    std::unordered_map<std::string, Bucket> buckets_;
};

} // namespace tenzor::serving
