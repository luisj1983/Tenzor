/**
 * @file test_rendezvous.cpp
 * @brief Tests for elastic rendezvous protocol
 */

#include <gtest/gtest.h>
#include <tenzor/distributed/elastic/rendezvous.hpp>

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
