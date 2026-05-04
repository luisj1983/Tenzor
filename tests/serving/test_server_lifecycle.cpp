/**
 * @file test_server_lifecycle.cpp
 * @brief Tests for serving::InferenceServer lifecycle (audit-2026-05-03 N4).
 *
 * Verifies the start/stop lifecycle, basic config validation, and that
 * the repository/traffic_router accessors return functional handles.
 * Concurrent-request testing is left to integration suites that need
 * a real model on disk; this file focuses on the lifecycle invariants.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/serving/server.hpp>
#include <chrono>
#include <thread>

using namespace tenzor;
using namespace tenzor::serving;

class ServerLifecycleTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

TEST_F(ServerLifecycleTest, ConstructWithDefaultConfig) {
    ServerConfig config;
    config.http_port = 0;  // 0 = OS-assigned, avoids port collisions in tests
    config.grpc_port = 0;
    config.num_workers = 1;
    InferenceServer server(config);
    // Construction shouldn't throw and accessors should return references.
    EXPECT_NO_THROW(server.repository());
    EXPECT_NO_THROW(server.traffic_router());
}

TEST_F(ServerLifecycleTest, StartStopLifecycle) {
    ServerConfig config;
    config.http_port = 0;
    config.grpc_port = 0;
    config.num_workers = 1;
    config.enable_metrics = false;
    config.enable_health_check = false;
    InferenceServer server(config);

    EXPECT_NO_THROW(server.start());
    // Give the server thread a moment to enter its serve_loop.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_NO_THROW(server.stop());
}

TEST_F(ServerLifecycleTest, RepeatedStartStopIsSafe) {
    ServerConfig config;
    config.http_port = 0;
    config.grpc_port = 0;
    config.num_workers = 1;
    InferenceServer server(config);

    // Two start/stop cycles must not deadlock or leave dangling threads.
    server.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    server.stop();
    server.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    server.stop();
}

TEST_F(ServerLifecycleTest, ModelRepositoryAccessible) {
    ServerConfig config;
    config.http_port = 0;
    config.grpc_port = 0;
    config.num_workers = 1;
    InferenceServer server(config);

    auto& repo = server.repository();
    // Empty repo: list_models() returns empty vector.
    auto models = repo.list_models();
    EXPECT_TRUE(models.empty());
}
