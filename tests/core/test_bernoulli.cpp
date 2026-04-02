/**
 * @file test_bernoulli.cpp
 * @brief Tests for Bernoulli distribution sampling
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"

using namespace tenzor;

class BernoulliTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

TEST_F(BernoulliTest, AllZeroProb) {
    auto probs = tenzor::zeros({10});
    auto samples = tenzor::bernoulli(probs);

    EXPECT_EQ(samples.shape()[0], 10);

    // All should be 0
    auto* data = samples.data<float>();
    for (int64_t i = 0; i < 10; i++) {
        EXPECT_EQ(data[i], 0.0f);
    }
}

TEST_F(BernoulliTest, AllOneProb) {
    auto probs = tenzor::ones({10});
    auto samples = tenzor::bernoulli(probs);

    // All should be 1
    auto* data = samples.data<float>();
    for (int64_t i = 0; i < 10; i++) {
        EXPECT_EQ(data[i], 1.0f);
    }
}

TEST_F(BernoulliTest, OutputShape) {
    auto probs = tenzor::full({3, 4}, 0.5f);
    auto samples = tenzor::bernoulli(probs);

    EXPECT_EQ(samples.ndim(), 2);
    EXPECT_EQ(samples.shape()[0], 3);
    EXPECT_EQ(samples.shape()[1], 4);
}

TEST_F(BernoulliTest, ValuesAreZeroOrOne) {
    auto probs = tenzor::full({100}, 0.5f);
    auto samples = tenzor::bernoulli(probs);

    auto* data = samples.data<float>();
    for (int64_t i = 0; i < 100; i++) {
        EXPECT_TRUE(data[i] == 0.0f || data[i] == 1.0f);
    }
}

TEST_F(BernoulliTest, StatisticalCorrectness) {
    // With p=0.7, over many samples, mean should be ~0.7
    int64_t n = 10000;
    auto probs = tenzor::full({n}, 0.7f);
    auto samples = tenzor::bernoulli(probs);

    float mean = tenzor::mean(samples).data<float>()[0];
    EXPECT_NEAR(mean, 0.7f, 0.05f);  // Allow 5% deviation
}
