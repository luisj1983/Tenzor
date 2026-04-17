/**
 * @file test_laplace_distribution_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for Laplace distribution
 *
 * Sampling tests verify shapes and finiteness, not exact statistical properties.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributions/distribution.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"

#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class LaplaceDistributionMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(LaplaceDistributionMultiDTypeTest, MeanMatchesLoc) {
    auto loc = zeros({1}, dtype(), device());
    auto scale = ones({1}, dtype(), device());
    distributions::Laplace dist(loc, scale);

    auto m = dist.mean();
    auto m_cpu = m.to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(m_cpu.data<float>()[0], 0.0f, atol());
}

TEST_P(LaplaceDistributionMultiDTypeTest, VarianceIs2ScaleSquared) {
    auto loc = zeros({1}, dtype(), device());
    auto scale = ones({1}, dtype(), device());
    distributions::Laplace dist(loc, scale);

    auto v = dist.variance();
    auto v_cpu = v.to(Device::cpu()).to(DType::Float32);
    // Laplace variance = 2*scale^2 = 2
    EXPECT_NEAR(v_cpu.data<float>()[0], 2.0f, atol() + 1e-2f);
}

TEST_P(LaplaceDistributionMultiDTypeTest, LogProbAtMean) {
    auto loc = zeros({1}, dtype(), device());
    auto scale = ones({1}, dtype(), device());
    distributions::Laplace dist(loc, scale);

    auto lp = dist.log_prob(loc);
    auto lp_cpu = lp.to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(lp_cpu.data<float>()[0], -std::log(2.0f), atol() + 1e-2f);
}

TEST_P(LaplaceDistributionMultiDTypeTest, EntropyClosedForm) {
    auto loc = zeros({1}, dtype(), device());
    auto scale = ones({1}, dtype(), device());
    distributions::Laplace dist(loc, scale);

    auto h = dist.entropy();
    auto h_cpu = h.to(Device::cpu()).to(DType::Float32);
    // Entropy = 1 + log(2b) with b=1 is 1 + log(2)
    EXPECT_NEAR(h_cpu.data<float>()[0], 1.0f + std::log(2.0f), atol() + 1e-2f);
}

TEST_P(LaplaceDistributionMultiDTypeTest, SampleHasRequestedShape) {
    auto loc = zeros({1}, dtype(), device());
    auto scale = ones({1}, dtype(), device());
    distributions::Laplace dist(loc, scale);

    auto s = dist.sample({32});
    EXPECT_EQ(s.shape().size(), 1u);
    EXPECT_EQ(s.shape()[0], 32);
}

TEST_P(LaplaceDistributionMultiDTypeTest, SamplesAreFinite) {
    auto loc = zeros({1}, dtype(), device());
    auto scale = ones({1}, dtype(), device());
    distributions::Laplace dist(loc, scale);

    auto s = dist.sample({64});
    auto s_cpu = s.to(Device::cpu()).to(DType::Float32);
    const auto* d = s_cpu.data<float>();
    for (int64_t i = 0; i < s.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(d[i]))
            << "Laplace sample[" << i << "] = " << d[i] << " is not finite";
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(LaplaceDistributionMultiDTypeTest);
