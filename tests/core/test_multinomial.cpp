/**
 * @file test_multinomial.cpp
 * @brief Tests for multinomial sampling
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "../backend_test_fixture.hpp"

#include <set>

using namespace tenzor;

class MultinomialTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

TEST_P(MultinomialTest, BasicSampling) {
    // Uniform weights
    auto probs = tenzor::ones({4}, DType::Float32, device);
    auto samples = tenzor::multinomial(probs, 2, /*replacement=*/true);

    EXPECT_EQ(samples.ndim(), 1);
    EXPECT_EQ(samples.shape()[0], 2);
    EXPECT_EQ(samples.dtype(), DType::Int64);

    // All samples should be in [0, 4)
    auto samples_cpu = samples.cpu();
    auto* data = samples_cpu.data<int64_t>();
    for (int64_t i = 0; i < 2; i++) {
        EXPECT_GE(data[i], 0);
        EXPECT_LT(data[i], 4);
    }
}

TEST_P(MultinomialTest, DeterministicWeight) {
    // All weight on index 2
    auto probs_cpu = tenzor::zeros({5}, DType::Float32);
    probs_cpu.data<float>()[2] = 1.0f;
    auto probs = probs_cpu.to(device);

    auto samples = tenzor::multinomial(probs, 10, /*replacement=*/true);

    // All samples should be 2
    auto samples_cpu = samples.cpu();
    auto* data = samples_cpu.data<int64_t>();
    for (int64_t i = 0; i < 10; i++) {
        EXPECT_EQ(data[i], 2);
    }
}

TEST_P(MultinomialTest, LargeVocabularyWithReplacement) {
    // >1024 categories must work with replacement (LM-vocab sampling); the old
    // CUDA single-block CDF scan capped at 1024 and threw.
    const int64_t V = 5000;
    auto probs_cpu = tenzor::zeros({V}, DType::Float32);
    probs_cpu.data<float>()[4321] = 1.0f;  // all mass on one category
    auto probs = probs_cpu.to(device);
    auto samples = tenzor::multinomial(probs, 8, /*replacement=*/true).cpu();
    auto* d = samples.data<int64_t>();
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(d[i], 4321) << "Failed on " << device.to_string();
}

TEST_P(MultinomialTest, AllZeroWeightsThrows) {
    // A row with zero total probability must throw on every backend (not return
    // garbage indices).
    auto probs = tenzor::zeros({6}, DType::Float32).to(device);
    EXPECT_ANY_THROW(tenzor::multinomial(probs, 3, /*replacement=*/true))
        << "Failed on " << device.to_string();
    EXPECT_ANY_THROW(tenzor::multinomial(probs, 3, /*replacement=*/false))
        << "Failed on " << device.to_string();
}

TEST_P(MultinomialTest, WithoutReplacement) {
    auto probs = tenzor::ones({5}, DType::Float32, device);
    auto samples = tenzor::multinomial(probs, 5, /*replacement=*/false);

    EXPECT_EQ(samples.shape()[0], 5);

    // All indices should be unique
    auto samples_cpu = samples.cpu();
    auto* data = samples_cpu.data<int64_t>();
    std::set<int64_t> seen;
    for (int64_t i = 0; i < 5; i++) {
        EXPECT_TRUE(seen.insert(data[i]).second) << "Duplicate index: " << data[i];
    }
}

TEST_P(MultinomialTest, BatchedSampling) {
    auto probs = tenzor::ones({3, 4}, DType::Float32, device);
    auto samples = tenzor::multinomial(probs, 2, /*replacement=*/true);

    EXPECT_EQ(samples.ndim(), 2);
    EXPECT_EQ(samples.shape()[0], 3);
    EXPECT_EQ(samples.shape()[1], 2);
}

INSTANTIATE_BACKEND_TESTS(MultinomialTest);
