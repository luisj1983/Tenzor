/**
 * @file test_auth.cpp
 * @brief Tests for authentication configuration and token validation
 */

#include <gtest/gtest.h>
#include <tenzor/serving/auth.hpp>

using namespace tenzor::serving;

TEST(AuthTest, ConfigDefaults) {
    AuthConfig config;
    EXPECT_FALSE(config.enabled);
    EXPECT_TRUE(config.api_keys.empty());
    EXPECT_EQ(config.header_name, "Authorization");
}

TEST(AuthTest, DisabledAllowsAll) {
    AuthConfig config;
    config.enabled = false;
    EXPECT_TRUE(validate_token(config, ""));
    EXPECT_TRUE(validate_token(config, "anything"));
    EXPECT_TRUE(validate_token(config, "Bearer garbage"));
}

TEST(AuthTest, EnabledButEmptyKeysDeniesAll) {
    AuthConfig config;
    config.enabled = true;
    config.api_keys = {};
    // Secure fail-closed default (auth.hpp): an enabled-but-keyless server is an
    // operator misconfiguration, so validate_token DENIES every request rather
    // than silently accepting traffic. Auth is only bypassed when explicitly
    // disabled (config.enabled = false, see DisabledAllowsAll).
    EXPECT_FALSE(validate_token(config, "any_token"));
    EXPECT_FALSE(validate_token(config, ""));
    EXPECT_FALSE(validate_token(config, "Bearer whatever"));
}

TEST(AuthTest, ValidBearerToken) {
    AuthConfig config;
    config.enabled = true;
    config.api_keys = {"secret123", "key456"};

    EXPECT_TRUE(validate_token(config, "Bearer secret123"));
    EXPECT_TRUE(validate_token(config, "Bearer key456"));
}

TEST(AuthTest, ValidRawToken) {
    AuthConfig config;
    config.enabled = true;
    config.api_keys = {"secret123"};

    // Raw token without Bearer prefix should also work
    EXPECT_TRUE(validate_token(config, "secret123"));
}

TEST(AuthTest, InvalidToken) {
    AuthConfig config;
    config.enabled = true;
    config.api_keys = {"secret123"};

    EXPECT_FALSE(validate_token(config, "Bearer wrong_key"));
    EXPECT_FALSE(validate_token(config, "wrong_key"));
    EXPECT_FALSE(validate_token(config, ""));
}

TEST(AuthTest, MultipleKeysAnyValid) {
    AuthConfig config;
    config.enabled = true;
    config.api_keys = {"aaa", "bbb", "ccc"};

    EXPECT_TRUE(validate_token(config, "Bearer aaa"));
    EXPECT_TRUE(validate_token(config, "Bearer bbb"));
    EXPECT_TRUE(validate_token(config, "Bearer ccc"));
    EXPECT_FALSE(validate_token(config, "Bearer ddd"));
}
