/**
 * @file test_health_monitor.cpp
 * @brief Tests for health monitoring
 */

#include <gtest/gtest.h>
#include <tenzor/distributed/elastic/health_monitor.hpp>
#include <tenzor/distributed/rpc/rpc_agent.hpp>

using namespace tenzor::distributed::elastic;
using namespace tenzor::distributed::rpc;

TEST(HealthMonitorTest, ConfigDefaults) {
    HealthMonitorConfig config;
    EXPECT_EQ(config.suspect_threshold, 1);
    EXPECT_EQ(config.dead_threshold, 3);
}

TEST(HealthMonitorTest, Construction) {
    WorkerInfo self;
    self.id = 0;
    auto agent = std::make_shared<TcpRpcAgent>(self);
    HealthMonitor monitor(agent);
    // Should not be running before start()
    auto dead = monitor.dead_workers();
    EXPECT_TRUE(dead.empty());
}

TEST(HealthMonitorTest, WorkerStateEnum) {
    EXPECT_NE(static_cast<int>(WorkerState::ALIVE), static_cast<int>(WorkerState::SUSPECTED));
    EXPECT_NE(static_cast<int>(WorkerState::SUSPECTED), static_cast<int>(WorkerState::DEAD));
}
