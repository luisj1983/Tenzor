/**
 * @file test_rendezvous.cpp
 * @brief Tests for elastic rendezvous protocol
 */

#include <gtest/gtest.h>
#include <tenzor/distributed/elastic/rendezvous.hpp>
#include <chrono>

using namespace tenzor::distributed::elastic;

TEST(RendezvousTest, ConfigDefaults) {
    RendezvousConfig config;
    EXPECT_EQ(config.min_workers, 1);
    EXPECT_EQ(config.max_workers, 256);
    EXPECT_EQ(config.store_port, 29400);
}

TEST(RendezvousTest, Construction) {
    RendezvousConfig config;
    config.run_id = "test_run";
    C10dRendezvous rendezvous(config);
    EXPECT_EQ(rendezvous.rank(), -1);
    EXPECT_EQ(rendezvous.world_size(), 0);
}

TEST(RendezvousTest, LeaveResetsState) {
    RendezvousConfig config;
    config.run_id = "test_leave";
    C10dRendezvous rendezvous(config);
    rendezvous.leave();
    EXPECT_EQ(rendezvous.rank(), -1);
    EXPECT_EQ(rendezvous.world_size(), 0);
}

// Real store-based join (WS17): a single worker with min_workers=1 must get
// rank 0 / world_size 1 (previously join() was a stub returning rank=-1).
TEST(RendezvousTest, SingleWorkerJoinAssignsRankZero) {
    RendezvousConfig config;
    config.run_id = "test_single";
    config.min_workers = 1;
    config.max_workers = 4;
    config.store_port = 29621;
    config.timeout = std::chrono::seconds(5);
    C10dRendezvous rendezvous(config);

    auto result = rendezvous.join();
    EXPECT_EQ(result.rank, 0);
    EXPECT_EQ(result.world_size, 1);
    EXPECT_EQ(rendezvous.rank(), 0);
    EXPECT_EQ(rendezvous.world_size(), 1);
}

// join() must THROW on timeout when min_workers cannot be met, rather than
// silently returning rank=-1 / world_size=0 (the old stub behaviour).
TEST(RendezvousTest, JoinThrowsOnTimeoutBelowMinWorkers) {
    RendezvousConfig config;
    config.run_id = "test_timeout";
    config.min_workers = 2;   // never met by a single worker
    config.max_workers = 4;
    config.store_port = 29622;
    config.timeout = std::chrono::seconds(1);
    C10dRendezvous rendezvous(config);

    EXPECT_THROW(rendezvous.join(), std::runtime_error);
}
