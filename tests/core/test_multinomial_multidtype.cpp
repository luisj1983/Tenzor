/**
 * @file test_multinomial_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for multinomial sampling
 *
 * Sampling tests verify shapes and dtypes, not exact values.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class MultinomialMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(MultinomialMultiDTypeTest, BasicSamplingShape) {
    // Uniform weights
    auto probs = createOnes({4});
    auto samples = tenzor::multinomial(probs, 2, /*replacement=*/true);

    EXPECT_EQ(samples.ndim(), 1);
    EXPECT_EQ(samples.shape()[0], 2);
    EXPECT_EQ(samples.dtype(), DType::Int64);
}

TEST_P(MultinomialMultiDTypeTest, BasicSamplingValidRange) {
    auto probs = createOnes({4});
    auto samples = tenzor::multinomial(probs, 2, /*replacement=*/true);

    // All samples should be in [0, 4)
    auto s_cpu = samples.to(Device::cpu());
    auto* data = s_cpu.data<int64_t>();
    for (int64_t i = 0; i < 2; i++) {
        EXPECT_GE(data[i], 0);
        EXPECT_LT(data[i], 4);
    }
}

TEST_P(MultinomialMultiDTypeTest, DeterministicWeight) {
    // All weight on index 2
    auto probs = createZeros({5});
    // Set the weight via CPU then move back
    auto probs_cpu = probs.to(Device::cpu()).to(DType::Float32);
    probs_cpu.data<float>()[2] = 1.0f;
    probs = probs_cpu.to(dtype()).to(device());

    auto samples = tenzor::multinomial(probs, 10, /*replacement=*/true);

    auto s_cpu = samples.to(Device::cpu());
    auto* data = s_cpu.data<int64_t>();
    for (int64_t i = 0; i < 10; i++) {
        EXPECT_EQ(data[i], 2);
    }
}

TEST_P(MultinomialMultiDTypeTest, WithoutReplacementShape) {
    auto probs = createOnes({5});
    auto samples = tenzor::multinomial(probs, 5, /*replacement=*/false);

    EXPECT_EQ(samples.shape()[0], 5);
    EXPECT_EQ(samples.dtype(), DType::Int64);
}

TEST_P(MultinomialMultiDTypeTest, WithoutReplacementUnique) {
    auto probs = createOnes({5});
    auto samples = tenzor::multinomial(probs, 5, /*replacement=*/false);

    auto s_cpu = samples.to(Device::cpu());
    auto* data = s_cpu.data<int64_t>();
    std::set<int64_t> seen;
    for (int64_t i = 0; i < 5; i++) {
        EXPECT_TRUE(seen.insert(data[i]).second) << "Duplicate index: " << data[i];
    }
}

TEST_P(MultinomialMultiDTypeTest, BatchedSamplingShape) {
    auto probs = createOnes({3, 4});
    auto samples = tenzor::multinomial(probs, 2, /*replacement=*/true);

    EXPECT_EQ(samples.ndim(), 2);
    EXPECT_EQ(samples.shape()[0], 3);
    EXPECT_EQ(samples.shape()[1], 2);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MultinomialMultiDTypeTest);
