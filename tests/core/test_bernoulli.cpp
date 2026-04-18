/**
 * @file test_bernoulli.cpp
 * @brief Tests for Bernoulli distribution sampling — parameterized across
 *        every available backend. Previously CPU-only.
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "../backend_test_fixture.hpp"

using namespace tenzor;

class BernoulliTest : public tenzor::testing::BackendTest {};

TEST_P(BernoulliTest, AllZeroProb) {
    auto probs = tenzor::zeros({10}, DType::Float32, device);
    auto samples = tenzor::bernoulli(probs);

    EXPECT_EQ(samples.shape()[0], 10);

    auto s_cpu = samples.to(Device::cpu()).to(DType::Float32).contiguous();
    const float* data = s_cpu.data<float>();
    for (int64_t i = 0; i < 10; i++) {
        EXPECT_EQ(data[i], 0.0f);
    }
}

TEST_P(BernoulliTest, AllOneProb) {
    auto probs = tenzor::ones({10}, DType::Float32, device);
    auto samples = tenzor::bernoulli(probs);

    auto s_cpu = samples.to(Device::cpu()).to(DType::Float32).contiguous();
    const float* data = s_cpu.data<float>();
    for (int64_t i = 0; i < 10; i++) {
        EXPECT_EQ(data[i], 1.0f);
    }
}

TEST_P(BernoulliTest, OutputShape) {
    auto probs = tenzor::full({3, 4}, 0.5f, DType::Float32, device);
    auto samples = tenzor::bernoulli(probs);

    EXPECT_EQ(samples.ndim(), 2);
    EXPECT_EQ(samples.shape()[0], 3);
    EXPECT_EQ(samples.shape()[1], 4);
}

TEST_P(BernoulliTest, ValuesAreZeroOrOne) {
    auto probs = tenzor::full({100}, 0.5f, DType::Float32, device);
    auto samples = tenzor::bernoulli(probs);

    auto s_cpu = samples.to(Device::cpu()).to(DType::Float32).contiguous();
    const float* data = s_cpu.data<float>();
    for (int64_t i = 0; i < 100; i++) {
        EXPECT_TRUE(data[i] == 0.0f || data[i] == 1.0f);
    }
}

TEST_P(BernoulliTest, StatisticalCorrectness) {
    // With p=0.7, over many samples, mean should be ~0.7. Every backend's
    // RNG must produce statistically valid Bernoulli samples.
    int64_t n = 10000;
    auto probs = tenzor::full({n}, 0.7f, DType::Float32, device);
    auto samples = tenzor::bernoulli(probs);

    auto m = tenzor::mean(samples).to(Device::cpu()).to(DType::Float32).contiguous();
    float mean = m.data<float>()[0];
    EXPECT_NEAR(mean, 0.7f, 0.05f);
}

INSTANTIATE_BACKEND_TESTS(BernoulliTest);
