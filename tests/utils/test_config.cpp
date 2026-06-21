/**
 * @file test_config.cpp
 * @brief Unit tests for the production configuration system
 *
 * Exercises the ACTUAL shipped API in include/tenzor/utils/config.hpp:
 *   - tenzor::Config::instance() / get_bool() / set_bool()
 *   - tenzor::set_deterministic() / is_deterministic()
 *
 * The previous version of this file tested a test-local `TestConfig`
 * reimplementation (a generic string/int/float get/set + file load/save API
 * that ships nowhere in production), so all of its assertions were
 * tautological — they verified a std::unordered_map living inside the test and
 * covered zero production code. That dead surface has been removed; these
 * tests now cover the real Config and the deterministic-mode free functions
 * that actually gate kernel behavior (see the backend reduction kernels).
 */

#include <gtest/gtest.h>
#include <optional>

#include "tenzor/utils/config.hpp"

namespace {

// The production Config is a process-global singleton with no reset API, so
// each test restores the keys it touches to a known state on teardown to keep
// tests independent regardless of execution order.
class ConfigTest : public ::testing::Test {
protected:
    void TearDown() override {
        // Restore deterministic flag to its default (false) so a later test
        // (or a later run of this one) sees a clean slate.
        tenzor::set_deterministic(false);
    }
};

// ---------------------------------------------------------------------------
// Config singleton + get_bool/set_bool (the real production surface)
// ---------------------------------------------------------------------------

// Test 1: instance() returns the same singleton.
TEST_F(ConfigTest, SingletonInstance) {
    auto& config1 = tenzor::Config::instance();
    auto& config2 = tenzor::Config::instance();
    EXPECT_EQ(&config1, &config2);
}

// Test 2: an unset key returns nullopt, not a default-constructed false.
TEST_F(ConfigTest, GetBoolMissingKeyReturnsNullopt) {
    auto& config = tenzor::Config::instance();
    auto value = config.get_bool("config_test_definitely_unset_key");
    EXPECT_FALSE(value.has_value());
}

// Test 3: set_bool(true)/get_bool round-trip.
TEST_F(ConfigTest, SetBoolTrueRoundtrips) {
    auto& config = tenzor::Config::instance();
    config.set_bool("config_test_flag", true);

    auto value = config.get_bool("config_test_flag");
    ASSERT_TRUE(value.has_value());
    EXPECT_TRUE(*value);
}

// Test 4: set_bool(false)/get_bool round-trip, and overwriting an existing
// key flips the stored value (catches a set that appends instead of replaces).
TEST_F(ConfigTest, SetBoolFalseAndOverwrite) {
    auto& config = tenzor::Config::instance();

    config.set_bool("config_test_overwrite", true);
    {
        auto value = config.get_bool("config_test_overwrite");
        ASSERT_TRUE(value.has_value());
        EXPECT_TRUE(*value);
    }

    config.set_bool("config_test_overwrite", false);
    {
        auto value = config.get_bool("config_test_overwrite");
        ASSERT_TRUE(value.has_value());
        EXPECT_FALSE(*value);
    }
}

// ---------------------------------------------------------------------------
// Deterministic-mode free functions (set_deterministic / is_deterministic)
// ---------------------------------------------------------------------------

// Test 5: default state is non-deterministic.
TEST_F(ConfigTest, DeterministicDefaultsFalse) {
    // TearDown resets to false; assert the documented default.
    tenzor::set_deterministic(false);
    EXPECT_FALSE(tenzor::is_deterministic());
}

// Test 6: set_deterministic(true) is observable via is_deterministic().
TEST_F(ConfigTest, SetDeterministicTrueIsObservable) {
    tenzor::set_deterministic(true);
    EXPECT_TRUE(tenzor::is_deterministic());
}

// Test 7: set_deterministic toggles back and forth.
TEST_F(ConfigTest, SetDeterministicToggles) {
    tenzor::set_deterministic(true);
    EXPECT_TRUE(tenzor::is_deterministic());

    tenzor::set_deterministic(false);
    EXPECT_FALSE(tenzor::is_deterministic());

    tenzor::set_deterministic(true);
    EXPECT_TRUE(tenzor::is_deterministic());
}

// Test 8: the free functions and Config share state — set_deterministic(true)
// writes the "deterministic" bool key that get_bool reads back. This is the
// contract the kernels rely on (is_deterministic() == get_bool("deterministic")).
TEST_F(ConfigTest, DeterministicSharesConfigState) {
    tenzor::set_deterministic(true);

    auto via_config = tenzor::Config::instance().get_bool("deterministic");
    ASSERT_TRUE(via_config.has_value());
    EXPECT_TRUE(*via_config);
    EXPECT_EQ(*via_config, tenzor::is_deterministic());

    tenzor::set_deterministic(false);
    via_config = tenzor::Config::instance().get_bool("deterministic");
    ASSERT_TRUE(via_config.has_value());
    EXPECT_FALSE(*via_config);
    EXPECT_EQ(*via_config, tenzor::is_deterministic());
}

}  // namespace
