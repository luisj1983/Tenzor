/**
 * @file test_multinomial.cpp
 * @brief Tests for multinomial sampling
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"

using namespace tenzor;

class MultinomialTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

TEST_F(MultinomialTest, BasicSampling) {
    // Uniform weights
    auto probs = tenzor::ones({4});
    auto samples = tenzor::multinomial(probs, 2, /*replacement=*/true);

    EXPECT_EQ(samples.ndim(), 1);
    EXPECT_EQ(samples.shape()[0], 2);
    EXPECT_EQ(samples.dtype(), DType::Int64);

    // All samples should be in [0, 4)
    auto* data = samples.data<int64_t>();
    for (int64_t i = 0; i < 2; i++) {
        EXPECT_GE(data[i], 0);
        EXPECT_LT(data[i], 4);
    }
}

TEST_F(MultinomialTest, DeterministicWeight) {
    // All weight on index 2
    auto probs = tenzor::zeros({5});
    probs.data<float>()[2] = 1.0f;

    auto samples = tenzor::multinomial(probs, 10, /*replacement=*/true);

    // All samples should be 2
    auto* data = samples.data<int64_t>();
    for (int64_t i = 0; i < 10; i++) {
        EXPECT_EQ(data[i], 2);
    }
}

TEST_F(MultinomialTest, WithoutReplacement) {
    auto probs = tenzor::ones({5});
    auto samples = tenzor::multinomial(probs, 5, /*replacement=*/false);

    EXPECT_EQ(samples.shape()[0], 5);

    // All indices should be unique
    auto* data = samples.data<int64_t>();
    std::set<int64_t> seen;
    for (int64_t i = 0; i < 5; i++) {
        EXPECT_TRUE(seen.insert(data[i]).second) << "Duplicate index: " << data[i];
    }
}

TEST_F(MultinomialTest, BatchedSampling) {
    auto probs = tenzor::ones({3, 4});
    auto samples = tenzor::multinomial(probs, 2, /*replacement=*/true);

    EXPECT_EQ(samples.ndim(), 2);
    EXPECT_EQ(samples.shape()[0], 3);
    EXPECT_EQ(samples.shape()[1], 2);
}
