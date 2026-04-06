/**
 * @file test_rate_limiter.cpp
 * @brief Tests for TokenBucketRateLimiter
 */

#include <gtest/gtest.h>
#include <tenzor/serving/rate_limiter.hpp>

using namespace tenzor::serving;

TEST(RateLimiterTest, ConfigDefaults) {
    RateLimitConfig config;
    EXPECT_FALSE(config.enabled);
    EXPECT_DOUBLE_EQ(config.requests_per_second, 100.0);
    EXPECT_EQ(config.burst_size, 200);
}

TEST(RateLimiterTest, DisabledAllowsAll) {
    RateLimitConfig config;
    config.enabled = false;
    TokenBucketRateLimiter limiter(config);

    for (int i = 0; i < 1000; ++i) {
        EXPECT_TRUE(limiter.allow("client_a"));
    }
}

TEST(RateLimiterTest, BurstUpToBurstSize) {
    RateLimitConfig config;
    config.enabled = true;
    config.requests_per_second = 1.0;  // Very slow refill
    config.burst_size = 10;
    TokenBucketRateLimiter limiter(config);

    // First burst_size requests should succeed (bucket starts full)
    int allowed = 0;
    for (int i = 0; i < 20; ++i) {
        if (limiter.allow("client_burst")) ++allowed;
    }
    EXPECT_EQ(allowed, 10);
}

TEST(RateLimiterTest, PerClientIsolation) {
    RateLimitConfig config;
    config.enabled = true;
    config.requests_per_second = 1.0;
    config.burst_size = 5;
    TokenBucketRateLimiter limiter(config);

    // Exhaust client_a's bucket
    for (int i = 0; i < 10; ++i) {
        limiter.allow("client_a");
    }

    // client_b should still have a full bucket
    EXPECT_TRUE(limiter.allow("client_b"));
}

TEST(RateLimiterTest, ExhaustedBucketDenies) {
    RateLimitConfig config;
    config.enabled = true;
    config.requests_per_second = 0.001;  // Near-zero refill
    config.burst_size = 3;
    TokenBucketRateLimiter limiter(config);

    // Drain the bucket
    EXPECT_TRUE(limiter.allow("client_x"));
    EXPECT_TRUE(limiter.allow("client_x"));
    EXPECT_TRUE(limiter.allow("client_x"));

    // Should be denied now
    EXPECT_FALSE(limiter.allow("client_x"));
}
