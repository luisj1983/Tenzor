/**
 * @file test_traffic_router.cpp
 * @brief Tests for TrafficRouter A/B experiment routing
 */

#include <gtest/gtest.h>
#include <tenzor/serving/traffic_router.hpp>

using namespace tenzor::serving;

TEST(TrafficRouterTest, EmptyListExperiments) {
    TrafficRouter router;
    auto exps = router.list_experiments();
    EXPECT_TRUE(exps.empty());
}

TEST(TrafficRouterTest, SetAndListExperiment) {
    TrafficRouter router;
    TrafficRule rule{"model_v1", "model_v2", 0.2};
    router.set_experiment("exp_001", rule);

    auto exps = router.list_experiments();
    ASSERT_EQ(exps.size(), 1);
    EXPECT_EQ(exps[0], "exp_001");
}

TEST(TrafficRouterTest, RemoveExperiment) {
    TrafficRouter router;
    router.set_experiment("exp_rm", TrafficRule{"a", "b", 0.5});
    router.remove_experiment("exp_rm");

    auto exps = router.list_experiments();
    EXPECT_TRUE(exps.empty());
}

TEST(TrafficRouterTest, RemoveNonexistentNoThrow) {
    TrafficRouter router;
    EXPECT_NO_THROW(router.remove_experiment("does_not_exist"));
}

TEST(TrafficRouterTest, SelectModelReturnsValidModel) {
    TrafficRouter router;
    TrafficRule rule{"primary", "variant", 0.5};
    router.set_experiment("exp_select", rule);

    // Run many selections; both models should appear
    bool saw_primary = false, saw_variant = false;
    for (int i = 0; i < 200; ++i) {
        auto model = router.select_model("exp_select");
        if (model == "primary") saw_primary = true;
        if (model == "variant") saw_variant = true;
    }
    EXPECT_TRUE(saw_primary);
    EXPECT_TRUE(saw_variant);
}

TEST(TrafficRouterTest, SelectModelUnknownExperiment) {
    TrafficRouter router;
    auto model = router.select_model("nonexistent");
    EXPECT_TRUE(model.empty());
}

TEST(TrafficRouterTest, AllTrafficToA) {
    TrafficRouter router;
    TrafficRule rule{"only_a", "never_b", 0.0};
    router.set_experiment("exp_all_a", rule);

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(router.select_model("exp_all_a"), "only_a");
    }
}

TEST(TrafficRouterTest, MetricsAccumulate) {
    TrafficRouter router;
    TrafficRule rule{"ma", "mb", 0.5};
    router.set_experiment("exp_metrics", rule);

    for (int i = 0; i < 100; ++i) {
        router.select_model("exp_metrics");
    }

    auto [a, b] = router.get_metrics("exp_metrics");
    EXPECT_EQ(a + b, 100u);
    EXPECT_GT(a, 0u);
    EXPECT_GT(b, 0u);
}
