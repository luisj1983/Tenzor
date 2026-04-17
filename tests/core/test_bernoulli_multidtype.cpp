/**
 * @file test_bernoulli_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for Bernoulli distribution sampling
 *
 * Sampling tests verify shapes and dtype, not exact statistical properties.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class BernoulliMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(BernoulliMultiDTypeTest, AllZeroProbOutputShape) {
    auto probs = createZeros({10});
    auto samples = tenzor::bernoulli(probs);

    EXPECT_EQ(samples.shape()[0], 10);
    expectDevice(samples);
}

TEST_P(BernoulliMultiDTypeTest, AllZeroProbValues) {
    auto probs = createZeros({10});
    auto samples = tenzor::bernoulli(probs);

    auto s_cpu = samples.to(Device::cpu()).to(DType::Float32);
    auto* data = s_cpu.data<float>();
    for (int64_t i = 0; i < 10; i++) {
        EXPECT_NEAR(data[i], 0.0f, atol());
    }
}

TEST_P(BernoulliMultiDTypeTest, AllOneProbValues) {
    auto probs = createOnes({10});
    auto samples = tenzor::bernoulli(probs);

    auto s_cpu = samples.to(Device::cpu()).to(DType::Float32);
    auto* data = s_cpu.data<float>();
    for (int64_t i = 0; i < 10; i++) {
        EXPECT_NEAR(data[i], 1.0f, atol());
    }
}

TEST_P(BernoulliMultiDTypeTest, OutputShape2D) {
    // Build probs = 0.5 via ones * 0.5 pattern
    auto probs = createOnes({3, 4});
    // Scale to 0.5: multiply by scalar
    auto half = tenzor::zeros({1}, dtype(), device());
    auto h_cpu = half.to(Device::cpu()).to(DType::Float32);
    h_cpu.data<float>()[0] = 0.5f;
    half = h_cpu.to(dtype()).to(device());
    probs = probs * half;

    auto samples = tenzor::bernoulli(probs);

    EXPECT_EQ(samples.ndim(), 2);
    EXPECT_EQ(samples.shape()[0], 3);
    EXPECT_EQ(samples.shape()[1], 4);
    expectDevice(samples);
}

TEST_P(BernoulliMultiDTypeTest, ValuesAreZeroOrOne) {
    // Build probs = 0.5 via ones * 0.5
    auto probs = createOnes({100});
    auto half = tenzor::zeros({1}, dtype(), device());
    auto h_cpu = half.to(Device::cpu()).to(DType::Float32);
    h_cpu.data<float>()[0] = 0.5f;
    half = h_cpu.to(dtype()).to(device());
    probs = probs * half;

    auto samples = tenzor::bernoulli(probs);

    auto s_cpu = samples.to(Device::cpu()).to(DType::Float32);
    auto* data = s_cpu.data<float>();
    for (int64_t i = 0; i < 100; i++) {
        EXPECT_TRUE(data[i] < 0.5f || data[i] > 0.5f)
            << "Sample should be 0 or 1, got " << data[i];
        EXPECT_TRUE(std::abs(data[i]) < atol() || std::abs(data[i] - 1.0f) < atol())
            << "Sample should be 0 or 1, got " << data[i];
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(BernoulliMultiDTypeTest);
