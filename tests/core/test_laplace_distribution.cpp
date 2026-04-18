// Laplace distribution sanity tests — parameterized across every available
// backend. Previously CPU-only. Verifies identity properties and that
// sample() produces finite tensors of the requested shape on every backend.

#include <cmath>

#include <gtest/gtest.h>

#include <tenzor/distributions/distribution.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/tenzor.hpp>

#include "../backend_test_fixture.hpp"

namespace tenzor {
namespace {

class LaplaceDistributionTest : public tenzor::testing::BackendTest {};

TEST_P(LaplaceDistributionTest, MeanAndVariance) {
    auto loc = zeros({1}, DType::Float32, device);
    auto scale = ones({1}, DType::Float32, device);
    distributions::Laplace dist(loc, scale);

    auto m_cpu = dist.mean().to(Device::cpu()).contiguous();
    EXPECT_FLOAT_EQ(m_cpu.data<float>()[0], 0.0f);

    auto v_cpu = dist.variance().to(Device::cpu()).contiguous();
    // Laplace variance = 2*scale^2 = 2.
    EXPECT_NEAR(v_cpu.data<float>()[0], 2.0f, 1e-5f);
}

TEST_P(LaplaceDistributionTest, LogProbAtMeanIsMinusLog2Scale) {
    // log_prob(mu) = log(1 / (2b)) - 0 / b = -log(2b).
    auto loc = zeros({1}, DType::Float32, device);
    auto scale = ones({1}, DType::Float32, device);
    distributions::Laplace dist(loc, scale);

    auto lp = dist.log_prob(loc).to(Device::cpu()).contiguous();
    EXPECT_NEAR(lp.data<float>()[0], -std::log(2.0f), 1e-5f);
}

TEST_P(LaplaceDistributionTest, EntropyMatchesClosedForm) {
    auto loc = zeros({1}, DType::Float32, device);
    auto scale = ones({1}, DType::Float32, device);
    distributions::Laplace dist(loc, scale);

    auto h_cpu = dist.entropy().to(Device::cpu()).contiguous();
    // Entropy = 1 + log(2b) with b=1 is 1 + log(2).
    EXPECT_NEAR(h_cpu.data<float>()[0], 1.0f + std::log(2.0f), 1e-5f);
}

TEST_P(LaplaceDistributionTest, SampleHasRequestedShape) {
    auto loc = zeros({1}, DType::Float32, device);
    auto scale = ones({1}, DType::Float32, device);
    distributions::Laplace dist(loc, scale);

    auto s = dist.sample({32});
    EXPECT_EQ(s.shape().size(), 1u);
    EXPECT_EQ(s.shape()[0], 32);

    auto s_cpu = s.to(Device::cpu()).contiguous();
    const auto* d = s_cpu.data<float>();
    for (int64_t i = 0; i < s_cpu.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(d[i]))
            << "Laplace sample[" << i << "] = " << d[i] << " is not finite";
    }
}

TEST_P(LaplaceDistributionTest, SampleMeanApproximatesLoc) {
    auto loc = full({1}, 2.0, DType::Float32, device);
    auto scale = ones({1}, DType::Float32, device);
    distributions::Laplace dist(loc, scale);

    constexpr int64_t N = 8192;
    auto s_cpu = dist.sample({N}).to(Device::cpu()).contiguous();
    const auto* d = s_cpu.data<float>();
    double sum = 0.0;
    for (int64_t i = 0; i < N; ++i) sum += d[i];
    double sample_mean = sum / static_cast<double>(N);

    // With N=8192 and variance 2, MC error on the mean is ~sqrt(2/8192) ~=
    // 0.016. Tolerance 0.15 is comfortably above that and catches gross bias.
    EXPECT_NEAR(sample_mean, 2.0, 0.15)
        << "Laplace sample mean " << sample_mean
        << " diverges from loc=2.0 beyond Monte Carlo noise";
}

INSTANTIATE_BACKEND_TESTS(LaplaceDistributionTest);

}  // namespace
}  // namespace tenzor
