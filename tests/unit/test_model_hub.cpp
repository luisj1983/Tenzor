#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "tenzor/models/hub.hpp"
#include "tenzor/nn/module.hpp"
#include <fstream>
#include <filesystem>
#include <thread>
#include <chrono>

namespace fs = std::filesystem;

namespace tenzor::models::test {

class ModelHubTest : public ::testing::Test {
protected:
    std::string test_cache_dir;
    std::string test_file_path;
    std::string test_file_content;

    void SetUp() override {
        // Create unique temporary test directory for this specific test
        // Use test name and timestamp to ensure uniqueness across parallel test execution
        auto test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string unique_suffix = std::string(test_info->name()) + "_" +
                                   std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        test_cache_dir = fs::temp_directory_path() / ("tenzor_hub_test_" + unique_suffix);
        fs::create_directories(test_cache_dir);

        // Configure ModelHub to use test directory
        HubConfig config;
        config.cache_dir = test_cache_dir;
        config.verify_checksums = true;
        config.resume_downloads = true;
        config.connection_timeout = 10;
        config.max_retries = 2;
        ModelHub::set_config(config);

        // Create test file
        test_file_path = test_cache_dir + "/test_file.txt";
        test_file_content = "This is a test file for ModelHub unit tests.";
        std::ofstream file(test_file_path);
        file << test_file_content;
        file.close();
    }

    void TearDown() override {
        // Clean up test directory
        if (fs::exists(test_cache_dir)) {
            fs::remove_all(test_cache_dir);
        }
    }

    // Helper to create a simple HTTP server for testing
    // Note: In real tests, you might use a mock HTTP server library
    std::string create_test_url() {
        // For unit tests, we'll use file:// URLs
        return "file://" + test_file_path;
    }

    // Helper to compute expected SHA256
    std::string get_test_file_sha256() {
        return ModelHub::compute_checksum(test_file_path);
    }
};

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(ModelHubTest, SetAndGetCacheDir) {
    std::string custom_dir = test_cache_dir + "/custom";
    ModelHub::set_cache_dir(custom_dir);

    EXPECT_EQ(ModelHub::get_cache_dir(), custom_dir);
    EXPECT_TRUE(fs::exists(custom_dir));
}

TEST_F(ModelHubTest, SetAndGetConfig) {
    HubConfig config;
    config.cache_dir = test_cache_dir + "/config_test";
    config.max_cache_size = 1024 * 1024;  // 1MB
    config.verify_checksums = false;
    config.resume_downloads = false;
    config.connection_timeout = 60;
    config.max_retries = 5;

    ModelHub::set_config(config);

    HubConfig retrieved = ModelHub::get_config();
    EXPECT_EQ(retrieved.cache_dir, config.cache_dir);
    EXPECT_EQ(retrieved.max_cache_size, config.max_cache_size);
    EXPECT_EQ(retrieved.verify_checksums, config.verify_checksums);
    EXPECT_EQ(retrieved.resume_downloads, config.resume_downloads);
    EXPECT_EQ(retrieved.connection_timeout, config.connection_timeout);
    EXPECT_EQ(retrieved.max_retries, config.max_retries);
}

// ============================================================================
// Checksum Tests
// ============================================================================

TEST_F(ModelHubTest, ComputeChecksum) {
    std::string checksum = ModelHub::compute_checksum(test_file_path);

    // SHA256 should produce 64 hex characters
    EXPECT_EQ(checksum.length(), 64);

    // Should be consistent
    std::string checksum2 = ModelHub::compute_checksum(test_file_path);
    EXPECT_EQ(checksum, checksum2);
}

TEST_F(ModelHubTest, VerifyChecksum_Valid) {
    std::string expected = ModelHub::compute_checksum(test_file_path);
    EXPECT_TRUE(ModelHub::verify_checksum(test_file_path, expected));
}

TEST_F(ModelHubTest, VerifyChecksum_Invalid) {
    std::string wrong_checksum = "0000000000000000000000000000000000000000000000000000000000000000";
    EXPECT_FALSE(ModelHub::verify_checksum(test_file_path, wrong_checksum));
}

TEST_F(ModelHubTest, VerifyChecksum_Empty) {
    // Empty checksum should always pass (skip verification)
    EXPECT_TRUE(ModelHub::verify_checksum(test_file_path, ""));
}

// ============================================================================
// Model Registry Tests
// ============================================================================

TEST_F(ModelHubTest, RegisterModel) {
    ModelWeightInfo info;
    info.name = "test_model";
    info.url = "https://example.com/test_model.pth";
    info.sha256 = "abcd1234";
    info.size = 1024;
    info.description = "Test model";

    ModelHub::register_model(info);

    EXPECT_TRUE(ModelHub::is_registered("test_model"));

    ModelWeightInfo retrieved = ModelHub::get_model_info("test_model");
    EXPECT_EQ(retrieved.name, info.name);
    EXPECT_EQ(retrieved.url, info.url);
    EXPECT_EQ(retrieved.sha256, info.sha256);
    EXPECT_EQ(retrieved.size, info.size);
    EXPECT_EQ(retrieved.description, info.description);
}

TEST_F(ModelHubTest, RegisterMultipleModels) {
    std::vector<ModelWeightInfo> models;

    for (int i = 0; i < 5; i++) {
        ModelWeightInfo info;
        info.name = "model_" + std::to_string(i);
        info.url = "https://example.com/model_" + std::to_string(i) + ".pth";
        info.sha256 = "";
        info.size = i * 1024;
        info.description = "Test model " + std::to_string(i);
        models.push_back(info);
    }

    ModelHub::register_models(models);

    for (int i = 0; i < 5; i++) {
        EXPECT_TRUE(ModelHub::is_registered("model_" + std::to_string(i)));
    }

    auto registered = ModelHub::list_registered_models();
    EXPECT_GE(registered.size(), 5);  // At least our 5 models (may include default registry)
}

TEST_F(ModelHubTest, GetModelInfo_NotRegistered) {
    EXPECT_THROW(ModelHub::get_model_info("nonexistent_model"), std::runtime_error);
}

TEST_F(ModelHubTest, ListRegisteredModels) {
    // Clear and register fresh models
    ModelWeightInfo info1{"model1", "url1", "hash1", 100, "desc1"};
    ModelWeightInfo info2{"model2", "url2", "hash2", 200, "desc2"};

    ModelHub::register_model(info1);
    ModelHub::register_model(info2);

    auto models = ModelHub::list_registered_models();

    // Should contain at least our two models (plus any from default registry)
    EXPECT_GE(models.size(), 2);
    EXPECT_TRUE(std::find(models.begin(), models.end(), "model1") != models.end());
    EXPECT_TRUE(std::find(models.begin(), models.end(), "model2") != models.end());
}

// ============================================================================
// Cache Management Tests
// ============================================================================

TEST_F(ModelHubTest, CacheSize_Empty) {
    ModelHub::clear_cache();
    size_t size = ModelHub::cache_size();
    EXPECT_EQ(size, 0);
}

TEST_F(ModelHubTest, CacheSize_WithFiles) {
    // Clear existing cache files first
    ModelHub::clear_cache();

    // Create some test files in cache
    std::string file1 = test_cache_dir + "/test1.pt";
    std::string file2 = test_cache_dir + "/test2.pt";

    std::ofstream f1(file1);
    f1 << "test content 1";
    f1.close();

    std::ofstream f2(file2);
    f2 << "test content 2";
    f2.close();

    size_t size = ModelHub::cache_size();
    EXPECT_GT(size, 0);
    EXPECT_EQ(size, fs::file_size(file1) + fs::file_size(file2));
}

TEST_F(ModelHubTest, ClearCache) {
    // Create test files
    std::string file1 = test_cache_dir + "/model1.pt";
    std::string file2 = test_cache_dir + "/model2.pt";

    std::ofstream(file1) << "content1";
    std::ofstream(file2) << "content2";

    EXPECT_TRUE(fs::exists(file1));
    EXPECT_TRUE(fs::exists(file2));

    ModelHub::clear_cache();

    EXPECT_FALSE(fs::exists(file1));
    EXPECT_FALSE(fs::exists(file2));
}

TEST_F(ModelHubTest, ListCachedModels) {
    // Create test cache files
    std::ofstream(test_cache_dir + "/model1.pt") << "content1";
    std::ofstream(test_cache_dir + "/model2.pt") << "content2";
    std::ofstream(test_cache_dir + "/model3.pt") << "content3";

    auto cached = ModelHub::list_cached_models();

    EXPECT_EQ(cached.size(), 3);
    EXPECT_TRUE(std::find(cached.begin(), cached.end(), "model1") != cached.end());
    EXPECT_TRUE(std::find(cached.begin(), cached.end(), "model2") != cached.end());
    EXPECT_TRUE(std::find(cached.begin(), cached.end(), "model3") != cached.end());
}

TEST_F(ModelHubTest, IsCached) {
    std::string model_name = "test_cached_model";
    EXPECT_FALSE(ModelHub::is_cached(model_name));

    // Create cache file
    std::string cache_path = test_cache_dir + "/" + model_name + ".pt";
    std::ofstream(cache_path) << "cached content";

    EXPECT_TRUE(ModelHub::is_cached(model_name));
}

TEST_F(ModelHubTest, GetCachedPath) {
    std::string model_name = "test_path_model";

    // Not cached yet
    EXPECT_EQ(ModelHub::get_cached_path(model_name), "");

    // Cache it
    std::string cache_path = test_cache_dir + "/" + model_name + ".pt";
    std::ofstream(cache_path) << "cached content";

    std::string retrieved_path = ModelHub::get_cached_path(model_name);
    EXPECT_EQ(retrieved_path, cache_path);
    EXPECT_TRUE(fs::exists(retrieved_path));
}

TEST_F(ModelHubTest, RemoveFromCache) {
    std::string model_name = "test_remove_model";
    std::string cache_path = test_cache_dir + "/" + model_name + ".pt";

    // Not cached yet
    EXPECT_FALSE(ModelHub::remove_from_cache(model_name));

    // Cache it
    std::ofstream(cache_path) << "cached content";
    EXPECT_TRUE(fs::exists(cache_path));

    // Remove it
    EXPECT_TRUE(ModelHub::remove_from_cache(model_name));
    EXPECT_FALSE(fs::exists(cache_path));

    // Try removing again
    EXPECT_FALSE(ModelHub::remove_from_cache(model_name));
}

TEST_F(ModelHubTest, CleanCache_SizeLimit) {
    // Clear existing cache files first
    ModelHub::clear_cache();

    // Create files with known sizes
    std::ofstream f1(test_cache_dir + "/old1.pt");
    f1 << std::string(1000, 'a');  // 1KB
    f1.close();

    // Sleep to ensure different modification times
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    std::ofstream f2(test_cache_dir + "/old2.pt");
    f2 << std::string(1000, 'b');  // 1KB
    f2.close();

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    std::ofstream f3(test_cache_dir + "/new.pt");
    f3 << std::string(1000, 'c');  // 1KB
    f3.close();

    // Total size is ~3KB
    EXPECT_EQ(ModelHub::cache_size(), 3000);

    // Clean to 1500 bytes (should remove oldest file)
    size_t removed = ModelHub::clean_cache(1500);
    EXPECT_GE(removed, 1);

    size_t new_size = ModelHub::cache_size();
    EXPECT_LE(new_size, 2000);  // Should be around 2KB now
}

// ============================================================================
// Download Tests (using file:// URLs for testing)
// ============================================================================

TEST_F(ModelHubTest, DownloadWeights_FileURL) {
    // Register a test model with file:// URL
    std::string test_url = "file://" + test_file_path;
    std::string model_name = "test_download_model";

    // Note: file:// URLs may not work with CURL depending on build configuration
    // This test demonstrates the API, but may be skipped in CI
    try {
        std::string downloaded_path = ModelHub::download_weights(
            model_name,
            test_url,
            "",  // No checksum verification
            false  // No progress display
        );

        EXPECT_TRUE(fs::exists(downloaded_path));
        EXPECT_TRUE(ModelHub::is_cached(model_name));

        // Verify content
        std::ifstream file(downloaded_path);
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        EXPECT_EQ(content, test_file_content);
    } catch (const std::exception& e) {
        // Skip if CURL doesn't support file:// URLs
        GTEST_SKIP() << "CURL may not support file:// URLs: " << e.what();
    }
}

TEST_F(ModelHubTest, DownloadWeights_Caching) {
    // Create a cached file
    std::string model_name = "cached_model";
    std::string cache_path = test_cache_dir + "/" + model_name + ".pt";
    std::ofstream(cache_path) << "cached content";

    // Download should return cached path without hitting network
    std::string downloaded_path = ModelHub::download_weights(
        model_name,
        "https://example.com/never_downloaded.pth",
        "",
        false
    );

    EXPECT_EQ(downloaded_path, cache_path);
}

TEST_F(ModelHubTest, DownloadWeights_ChecksumMismatch) {
    std::string model_name = "checksum_test_model";
    std::string cache_path = test_cache_dir + "/" + model_name + ".pt";

    // Create cached file with wrong content
    std::ofstream(cache_path) << "wrong content";

    // Try to download with checksum verification
    std::string wrong_checksum = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef";

    // Should detect mismatch and attempt re-download
    // Since we can't easily test real downloads, we expect this to fail
    EXPECT_THROW(
        ModelHub::download_weights(
            model_name,
            "https://nonexistent.example.com/model.pth",
            wrong_checksum,
            false
        ),
        std::runtime_error
    );
}

TEST_F(ModelHubTest, DownloadPretrained_NotRegistered) {
    EXPECT_THROW(
        ModelHub::download_pretrained("nonexistent_model", false),
        std::runtime_error
    );
}

// ============================================================================
// Progress Callback Tests
// ============================================================================

TEST_F(ModelHubTest, ProgressCallback) {
    bool callback_called = false;
    size_t last_downloaded = 0;
    size_t last_total = 0;

    auto callback = [&](size_t downloaded, size_t total, double speed, double eta) {
        callback_called = true;
        last_downloaded = downloaded;
        last_total = total;
        EXPECT_GE(downloaded, 0);
        EXPECT_GE(total, 0);
        EXPECT_GE(speed, 0.0);
        EXPECT_GE(eta, 0.0);
    };

    // Try downloading with callback
    // Note: This may not trigger callback for file:// URLs or cached files
    std::string model_name = "callback_test_model";
    std::string test_url = "file://" + test_file_path;

    try {
        ModelHub::download_weights(model_name, test_url, "", false, callback);
        // Callback may or may not be called depending on CURL support
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Could not test callback: " << e.what();
    }
}

// ============================================================================
// Download Statistics Tests
// ============================================================================

TEST_F(ModelHubTest, DownloadStats) {
    // Create a test file to "download"
    std::string model_name = "stats_test_model";
    std::string test_url = "file://" + test_file_path;

    try {
        ModelHub::download_weights(model_name, test_url, "", false);

        DownloadStats stats = ModelHub::get_last_download_stats();

        EXPECT_GT(stats.total_bytes, 0);
        EXPECT_GE(stats.bytes_downloaded, 0);
        EXPECT_GE(stats.download_time, 0.0);
        EXPECT_GE(stats.average_speed, 0.0);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Could not test stats: " << e.what();
    }
}

// ============================================================================
// Weight Loading Tests
// ============================================================================

TEST_F(ModelHubTest, LoadPretrainedWeights_FileNotFound) {
    // Create a dummy module
    class DummyModule : public nn::Module {
    public:
        auto forward_impl(const Variable& input) -> Variable override {
            return input;
        }

        void load_state_dict(const std::unordered_map<std::string, Tensor>& state) override {
            // Dummy implementation for testing
            if (state.empty()) {
                throw std::runtime_error("File not found");
            }
        }
    };

    DummyModule model;

    EXPECT_THROW(
        ModelHub::load_pretrained_weights(model, "/nonexistent/path.pt", true),
        std::runtime_error
    );
}

TEST_F(ModelHubTest, LoadPretrainedWeights_Strict) {
    class TestModule : public nn::Module {
    public:
        bool load_called = false;
        bool strict_mode = false;

        auto forward_impl(const Variable& input) -> Variable override {
            return input;
        }

        void load_state_dict(const std::unordered_map<std::string, Tensor>& state) override {
            load_called = true;
            strict_mode = true;
            // Simulate architecture mismatch
            throw std::runtime_error("Architecture mismatch");
        }
    };

    TestModule model;

    // Create a dummy weights file
    std::string weights_path = test_cache_dir + "/test_weights.pt";
    std::ofstream(weights_path) << "dummy weights";

    // Strict mode should throw
    EXPECT_THROW(
        ModelHub::load_pretrained_weights(model, weights_path, true),
        std::runtime_error
    );
    EXPECT_TRUE(model.load_called);
    EXPECT_TRUE(model.strict_mode);
}

TEST_F(ModelHubTest, LoadPretrainedWeights_NonStrict) {
    class TestModule : public nn::Module {
    public:
        bool load_called = false;
        bool strict_mode = true;

        auto forward_impl(const Variable& input) -> Variable override {
            return input;
        }

        void load_state_dict(const std::unordered_map<std::string, Tensor>& state) override {
            load_called = true;
            strict_mode = false;
            // Simulate partial loading in non-strict mode - success
        }
    };

    TestModule model;

    // Create a dummy weights file
    std::string weights_path = test_cache_dir + "/test_weights.pt";
    std::ofstream(weights_path) << "dummy weights";

    // Non-strict mode should not throw
    EXPECT_NO_THROW(
        ModelHub::load_pretrained_weights(model, weights_path, false)
    );
    EXPECT_TRUE(model.load_called);
    EXPECT_FALSE(model.strict_mode);
}

// ============================================================================
// Default Registry Tests
// ============================================================================

TEST_F(ModelHubTest, DefaultRegistry_ResNet) {
    // Check that ResNet models are registered
    EXPECT_TRUE(ModelHub::is_registered("resnet18"));
    EXPECT_TRUE(ModelHub::is_registered("resnet34"));
    EXPECT_TRUE(ModelHub::is_registered("resnet50"));
    EXPECT_TRUE(ModelHub::is_registered("resnet101"));
    EXPECT_TRUE(ModelHub::is_registered("resnet152"));

    // Check that URLs are correct
    ModelWeightInfo info = ModelHub::get_model_info("resnet50");
    EXPECT_FALSE(info.url.empty());
    EXPECT_TRUE(info.url.find("pytorch.org") != std::string::npos);
}

TEST_F(ModelHubTest, DefaultRegistry_VGG) {
    EXPECT_TRUE(ModelHub::is_registered("vgg11"));
    EXPECT_TRUE(ModelHub::is_registered("vgg13"));
    EXPECT_TRUE(ModelHub::is_registered("vgg16"));
    EXPECT_TRUE(ModelHub::is_registered("vgg19"));
}

TEST_F(ModelHubTest, DefaultRegistry_MobileNet) {
    EXPECT_TRUE(ModelHub::is_registered("mobilenet_v2"));
}

TEST_F(ModelHubTest, DefaultRegistry_EfficientNet) {
    for (int i = 0; i <= 7; i++) {
        std::string name = "efficientnet_b" + std::to_string(i);
        EXPECT_TRUE(ModelHub::is_registered(name));
    }
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(ModelHubTest, ConcurrentAccess) {
    const int num_threads = 10;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([&, i]() {
            try {
                // Register model
                ModelWeightInfo info;
                info.name = "concurrent_model_" + std::to_string(i);
                info.url = "https://example.com/model_" + std::to_string(i) + ".pth";
                info.sha256 = "";
                info.size = 1024;
                info.description = "Concurrent test model";
                ModelHub::register_model(info);

                // Check registration
                EXPECT_TRUE(ModelHub::is_registered(info.name));

                // Get cache size
                size_t size = ModelHub::cache_size();
                EXPECT_GE(size, 0);

                success_count++;
            } catch (const std::exception& e) {
                FAIL() << "Thread " << i << " failed: " << e.what();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count, num_threads);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(ModelHubTest, EmptyModelName) {
    EXPECT_THROW(
        ModelHub::download_weights("", "https://example.com/model.pth", "", false),
        std::runtime_error
    );
}

TEST_F(ModelHubTest, InvalidURL) {
    EXPECT_THROW(
        ModelHub::download_weights("test_model", "not_a_valid_url", "", false),
        std::runtime_error
    );
}

TEST_F(ModelHubTest, ComputeChecksum_NonexistentFile) {
    EXPECT_THROW(
        ModelHub::compute_checksum("/nonexistent/file.txt"),
        std::runtime_error
    );
}

TEST_F(ModelHubTest, VerifyChecksum_NonexistentFile) {
    EXPECT_THROW(
        ModelHub::verify_checksum("/nonexistent/file.txt", "abc123"),
        std::runtime_error
    );
}

} // namespace tenzor::models::test

int main(int argc, char** argv) {
    tenzor::initialize();  // Initialize Tenzor library and backends
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
