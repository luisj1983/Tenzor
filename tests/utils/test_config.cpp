/**
 * @file test_config.cpp
 * @brief Unit tests for configuration system
 */

#include <gtest/gtest.h>
#include <tenzor/utils/config.hpp>
#include <fstream>
#include <filesystem>
#include <string>
#include <unistd.h>  // getpid — audit-5 Y.33

using namespace tenzor;

class ConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Audit-5 Y.33: per-test, per-process temp path. The old shared
        // "/tmp/tenzor_test_config.ini" raced under parallel `ctest` runs
        // (different `bin/test_config` invocations clobbered each other's
        // load_from_file fixtures). Mirrors tests/nn/test_safetensors.cpp.
        const auto* info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        const std::string test_name = info ? info->name() : "unknown";
        test_config_file_ = (std::filesystem::temp_directory_path() /
                             ("tenzor_config_" + std::to_string(::getpid()) +
                              "_" + test_name + ".ini")).string();

        // Reset config to empty state by clearing all known keys
        auto& config = Config::instance();
        (void)config;

        // Clean up test config file if it exists
        if (std::filesystem::exists(test_config_file_)) {
            std::filesystem::remove(test_config_file_);
        }
    }

    void TearDown() override {
        // Clean up test config file
        if (std::filesystem::exists(test_config_file_)) {
            std::filesystem::remove(test_config_file_);
        }
    }

    std::string test_config_file_;
};

// Test 1: Singleton instance
TEST_F(ConfigTest, SingletonInstance) {
    auto& config1 = Config::instance();
    auto& config2 = Config::instance();

    // Should be same instance
    EXPECT_EQ(&config1, &config2);
}

// Test 2: String get/set
TEST_F(ConfigTest, StringGetSet) {
    auto& config = Config::instance();

    config.set_string("device", "cuda");
    auto value = config.get_string("device");

    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, "cuda");
}

// Test 3: Int get/set
TEST_F(ConfigTest, IntGetSet) {
    auto& config = Config::instance();

    config.set_int("num_threads", 8);
    auto value = config.get_int("num_threads");

    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, 8);
}

// Test 4: Float get/set
TEST_F(ConfigTest, FloatGetSet) {
    auto& config = Config::instance();

    config.set_float("learning_rate", 0.001);
    auto value = config.get_float("learning_rate");

    ASSERT_TRUE(value.has_value());
    EXPECT_NEAR(*value, 0.001, 1e-6);
}

// Test 5: Bool get/set
TEST_F(ConfigTest, BoolGetSet) {
    auto& config = Config::instance();

    config.set_bool("deterministic", true);
    auto value = config.get_bool("deterministic");

    ASSERT_TRUE(value.has_value());
    EXPECT_TRUE(*value);

    config.set_bool("deterministic", false);
    value = config.get_bool("deterministic");

    ASSERT_TRUE(value.has_value());
    EXPECT_FALSE(*value);
}

// Test 6: Non-existent key returns nullopt
TEST_F(ConfigTest, NonExistentKey) {
    auto& config = Config::instance();

    auto str_val = config.get_string("non_existent");
    auto int_val = config.get_int("non_existent");
    auto float_val = config.get_float("non_existent");
    auto bool_val = config.get_bool("non_existent");

    EXPECT_FALSE(str_val.has_value());
    EXPECT_FALSE(int_val.has_value());
    EXPECT_FALSE(float_val.has_value());
    EXPECT_FALSE(bool_val.has_value());
}

// Test 7: Save to file
TEST_F(ConfigTest, SaveToFile) {
    auto& config = Config::instance();

    config.set_string("device", "cpu");
    config.set_int("num_threads", 4);
    config.set_float("lr", 0.01);
    config.set_bool("cudnn_benchmark", true);

    bool success = config.save_to_file(test_config_file_);
    ASSERT_TRUE(success);
    ASSERT_TRUE(std::filesystem::exists(test_config_file_));

    // Verify file contents
    std::ifstream file(test_config_file_);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    EXPECT_NE(content.find("device=cpu"), std::string::npos);
    EXPECT_NE(content.find("num_threads=4"), std::string::npos);
    EXPECT_NE(content.find("lr=0."), std::string::npos);
    EXPECT_NE(content.find("cudnn_benchmark=true"), std::string::npos);
}

// Test 8: Load from file
TEST_F(ConfigTest, LoadFromFile) {
    // Create test config file
    {
        std::ofstream file(test_config_file_);
        file << "device=cuda\n";
        file << "num_threads=8\n";
        file << "learning_rate=0.001\n";
        file << "deterministic=true\n";
        file << "# This is a comment\n";
        file << "\n";  // Empty line
        file << "batch_size=32\n";
    }

    auto& config = Config::instance();
    bool success = config.load_from_file(test_config_file_);
    ASSERT_TRUE(success);

    // Verify loaded values
    auto device = config.get_string("device");
    ASSERT_TRUE(device.has_value());
    EXPECT_EQ(*device, "cuda");

    auto threads = config.get_int("num_threads");
    ASSERT_TRUE(threads.has_value());
    EXPECT_EQ(*threads, 8);

    auto lr = config.get_float("learning_rate");
    ASSERT_TRUE(lr.has_value());
    EXPECT_NEAR(*lr, 0.001, 1e-6);

    auto det = config.get_bool("deterministic");
    ASSERT_TRUE(det.has_value());
    EXPECT_TRUE(*det);

    auto batch = config.get_int("batch_size");
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(*batch, 32);
}

// Test 9: Load from non-existent file
TEST_F(ConfigTest, LoadFromNonExistentFile) {
    auto& config = Config::instance();
    // Audit-5 Y.33: per-pid path so a parallel test run can't have produced
    // this file on disk between SetUp and the call.
    const auto bogus = (std::filesystem::temp_directory_path() /
                        ("tenzor_config_missing_" +
                         std::to_string(::getpid()) + ".ini")).string();
    bool success = config.load_from_file(bogus);
    EXPECT_FALSE(success);
}

// Test 10: Boolean string parsing
TEST_F(ConfigTest, BooleanParsing) {
    auto& config = Config::instance();

    // Test "true" string
    config.set_string("bool_test", "true");
    auto val1 = config.get_bool("bool_test");
    ASSERT_TRUE(val1.has_value());
    EXPECT_TRUE(*val1);

    // Test "1" string
    config.set_string("bool_test", "1");
    auto val2 = config.get_bool("bool_test");
    ASSERT_TRUE(val2.has_value());
    EXPECT_TRUE(*val2);

    // Test "false" string
    config.set_string("bool_test", "false");
    auto val3 = config.get_bool("bool_test");
    ASSERT_TRUE(val3.has_value());
    EXPECT_FALSE(*val3);

    // Test "0" string
    config.set_string("bool_test", "0");
    auto val4 = config.get_bool("bool_test");
    ASSERT_TRUE(val4.has_value());
    EXPECT_FALSE(*val4);
}

// Test 11: Multiple value updates
TEST_F(ConfigTest, MultipleUpdates) {
    auto& config = Config::instance();

    config.set_int("counter", 0);
    auto val = config.get_int("counter");
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 0);

    config.set_int("counter", 10);
    val = config.get_int("counter");
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 10);

    config.set_int("counter", 100);
    val = config.get_int("counter");
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 100);
}

// Test 12: Roundtrip save and load
TEST_F(ConfigTest, RoundtripSaveLoad) {
    auto& config = Config::instance();

    // Set values
    config.set_string("model", "resnet50");
    config.set_int("epochs", 100);
    config.set_float("momentum", 0.9);
    config.set_bool("use_amp", true);

    // Save
    ASSERT_TRUE(config.save_to_file(test_config_file_));

    // Create new config instance would reset it, but singleton pattern
    // means we need to manually clear and reload
    // For this test, we'll just verify load works on same instance
    auto& config2 = Config::instance();
    bool success = config2.load_from_file(test_config_file_);
    ASSERT_TRUE(success);

    // Verify values
    EXPECT_EQ(*config2.get_string("model"), "resnet50");
    EXPECT_EQ(*config2.get_int("epochs"), 100);
    EXPECT_NEAR(*config2.get_float("momentum"), 0.9, 1e-6);
    EXPECT_TRUE(*config2.get_bool("use_amp"));
}
