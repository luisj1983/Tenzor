/**
 * @file test_model_repository.cpp
 * @brief Tests for inference serving model repository and batcher
 */

#include <gtest/gtest.h>
#include <tenzor/serving/server.hpp>

using namespace tenzor::serving;

TEST(ServingTest, ServerConfigDefaults) {
    ServerConfig config;
    EXPECT_EQ(config.http_port, 8080);
    EXPECT_EQ(config.grpc_port, 8081);
    EXPECT_EQ(config.num_workers, 4);
    EXPECT_TRUE(config.enable_metrics);
}

TEST(ServingTest, BatchConfigDefaults) {
    BatchConfig config;
    EXPECT_EQ(config.max_batch_size, 32);
    EXPECT_EQ(config.max_latency_us, 10000);
}

TEST(ServingTest, ModelRepositoryEmpty) {
    ModelRepository repo;
    auto models = repo.list_models();
    EXPECT_TRUE(models.empty());
    EXPECT_EQ(repo.get_model("nonexistent"), nullptr);
}

TEST(ServingTest, ModelRepositoryUnloadNonexistent) {
    ModelRepository repo;
    EXPECT_NO_THROW(repo.unload_model("nonexistent"));
}

TEST(ServingTest, MetricsRegistrySingleton) {
    auto& metrics = MetricsRegistry::instance();
    auto& m = metrics.get_metrics("test_model");
    EXPECT_EQ(m.total_requests.load(), 0);
    EXPECT_EQ(m.error_count.load(), 0);
}

TEST(ServingTest, MetricsPrometheusFormat) {
    auto& metrics = MetricsRegistry::instance();
    auto& m = metrics.get_metrics("test_prometheus");
    m.total_requests.store(100);
    m.total_latency_us.store(50000);

    auto text = metrics.format_prometheus();
    EXPECT_NE(text.find("tenzor_requests_total"), std::string::npos);
    EXPECT_NE(text.find("test_prometheus"), std::string::npos);
}

TEST(ServingTest, InferenceServerConstruction) {
    ServerConfig config;
    config.http_port = 0;  // Don't actually bind
    InferenceServer server(config);
    // Just verify construction doesn't crash
}

TEST(ServingTest, ModelStateEnum) {
    EXPECT_NE(static_cast<int>(ModelState::LOADING), static_cast<int>(ModelState::READY));
    EXPECT_NE(static_cast<int>(ModelState::READY), static_cast<int>(ModelState::FAILED));
}
