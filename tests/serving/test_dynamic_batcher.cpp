/**
 * @file test_dynamic_batcher.cpp
 * @brief Tests for DynamicBatcher and BatchConfig
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/serving/server.hpp>

class TestEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static ::testing::Environment* const env = ::testing::AddGlobalTestEnvironment(new TestEnv);

using namespace tenzor::serving;

TEST(DynamicBatcherTest, BatchConfigDefaults) {
    BatchConfig config;
    EXPECT_EQ(config.max_batch_size, 32);
    EXPECT_EQ(config.max_latency_us, 10000);
}

TEST(DynamicBatcherTest, BatchConfigCustom) {
    BatchConfig config;
    config.max_batch_size = 64;
    config.max_latency_us = 5000;
    EXPECT_EQ(config.max_batch_size, 64);
    EXPECT_EQ(config.max_latency_us, 5000);
}

TEST(DynamicBatcherTest, InferRequestCreation) {
    auto tensor = tenzor::zeros({2, 3});
    auto before = std::chrono::steady_clock::now();
    InferRequest req(tensor);
    auto after = std::chrono::steady_clock::now();

    EXPECT_EQ(req.input.shape()[0], 2);
    EXPECT_EQ(req.input.shape()[1], 3);
    EXPECT_GE(req.arrival, before);
    EXPECT_LE(req.arrival, after);
}

TEST(DynamicBatcherTest, InferRequestMovesTensor) {
    auto tensor = tenzor::ones({4, 4});
    InferRequest req(std::move(tensor));
    // The moved-into request should have the correct shape
    EXPECT_EQ(req.input.shape()[0], 4);
    EXPECT_EQ(req.input.shape()[1], 4);
}

TEST(DynamicBatcherTest, ServerConfigRateLimitDefaults) {
    ServerConfig config;
    EXPECT_FALSE(config.enable_rate_limit);
    EXPECT_DOUBLE_EQ(config.rate_limit_rps, 100.0);
    EXPECT_EQ(config.rate_limit_burst, 200);
}

TEST(DynamicBatcherTest, ServerConfigAuthDefaults) {
    ServerConfig config;
    EXPECT_FALSE(config.enable_auth);
    EXPECT_TRUE(config.api_keys.empty());
    EXPECT_EQ(config.auth_header, "Authorization");
}
