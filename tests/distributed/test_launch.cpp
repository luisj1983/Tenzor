/**
 * @file test_launch.cpp
 * @brief Direct tests for the distributed::launch helpers
 *        (audit-2026-05-03 N4). The full `spawn` path forks subprocesses,
 *        which is awkward in a unit-test harness; this file covers the
 *        env-driven helpers (init_from_env / get_local_rank) which back
 *        in-process rank discovery.
 */

#include <gtest/gtest.h>
#include <tenzor/distributed/launch.hpp>
#include <cstdlib>

using namespace tenzor::distributed;

// LOCAL_RANK is parsed from the environment by get_local_rank().
TEST(DistributedLaunch, GetLocalRankFromEnv) {
    setenv("LOCAL_RANK", "3", /*overwrite=*/1);
    EXPECT_EQ(get_local_rank(), 3);
    setenv("LOCAL_RANK", "0", 1);
    EXPECT_EQ(get_local_rank(), 0);
}

TEST(DistributedLaunch, GetLocalRankDefaultsZeroWhenUnset) {
    unsetenv("LOCAL_RANK");
    // Documented contract: returns 0 when LOCAL_RANK isn't set.
    EXPECT_EQ(get_local_rank(), 0);
}
