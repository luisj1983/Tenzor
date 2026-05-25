/**
 * @file test_elastic_trainer.cpp
 * @brief Tests for elastic training orchestrator
 */

#include <gtest/gtest.h>
#include <tenzor/distributed/elastic/elastic_trainer.hpp>

using namespace tenzor::distributed::elastic;

TEST(ElasticTrainerTest, ConfigDefaults) {
    ElasticConfig config;
    EXPECT_EQ(config.max_restarts, 3);
    EXPECT_TRUE(config.auto_checkpoint);
    // FF.25: default is now `<tmpdir>/tenzor_elastic_<pid>`; assert the prefix
    // rather than the exact path so the per-pid suffix and the platform tmpdir
    // (`/tmp` on Linux, `$TMPDIR` elsewhere) are both accepted.
    EXPECT_TRUE(config.checkpoint_dir.find("tenzor_elastic_") != std::string::npos)
        << "checkpoint_dir = " << config.checkpoint_dir;
}

TEST(ElasticTrainerTest, Construction) {
    ElasticConfig config;
    config.rendezvous.run_id = "test";
    ElasticTrainer trainer(config);
    EXPECT_EQ(trainer.rank(), -1);
    EXPECT_EQ(trainer.world_size(), 0);
    EXPECT_EQ(trainer.restart_count(), 0);
}

TEST(ElasticTrainerTest, ElasticLaunchFunction) {
    // elastic_launch is a free function that wraps ElasticTrainer::run
    // Just verify it compiles and the function pointer exists
    auto fn_ptr = &elastic_launch;
    EXPECT_NE(fn_ptr, nullptr);
}
