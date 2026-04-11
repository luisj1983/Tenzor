// Phase 6.1: Laplace distribution sanity tests.
// Verifies basic identity properties (mean = loc, variance = 2*scale^2,
// log_prob(mean) has the expected magnitude) and that sample() produces
// tensors of the requested shape.

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/distributions/distribution.hpp>
#include <tenzor/ops/creation.hpp>

#include <cmath>

namespace tenzor {
namespace {

class LaplaceDistributionTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

TEST_F(LaplaceDistributionTest, MeanAndVariance) {
    auto loc = zeros({1}, DType::Float32, Device::cpu());
    auto scale = ones({1}, DType::Float32, Device::cpu());
    distributions::Laplace dist(loc, scale);

    auto m = dist.mean();
    EXPECT_FLOAT_EQ(m.data<float>()[0], 0.0f);

    auto v = dist.variance();
    // Laplace variance = 2*scale^2 = 2.
    EXPECT_NEAR(v.data<float>()[0], 2.0f, 1e-5f);
}

TEST_F(LaplaceDistributionTest, LogProbAtMeanIsMinusLog2Scale) {
    // log_prob(mu) = log(1 / (2b)) - 0 / b = -log(2b).
    auto loc = zeros({1}, DType::Float32, Device::cpu());
    auto scale = ones({1}, DType::Float32, Device::cpu());
    distributions::Laplace dist(loc, scale);

    auto lp = dist.log_prob(loc);
    EXPECT_NEAR(lp.data<float>()[0], -std::log(2.0f), 1e-5f);
}

TEST_F(LaplaceDistributionTest, EntropyMatchesClosedForm) {
    auto loc = zeros({1}, DType::Float32, Device::cpu());
    auto scale = ones({1}, DType::Float32, Device::cpu());
    distributions::Laplace dist(loc, scale);

    auto h = dist.entropy();
    // Entropy = 1 + log(2b) with b=1 is 1 + log(2).
    EXPECT_NEAR(h.data<float>()[0], 1.0f + std::log(2.0f), 1e-5f);
}

TEST_F(LaplaceDistributionTest, SampleHasRequestedShape) {
    auto loc = zeros({1}, DType::Float32, Device::cpu());
    auto scale = ones({1}, DType::Float32, Device::cpu());
    distributions::Laplace dist(loc, scale);

    auto s = dist.sample({32});
    EXPECT_EQ(s.shape().size(), 1u);
    EXPECT_EQ(s.shape()[0], 32);

    // None of the samples should be NaN or infinite (tail clamping guards
    // against log(0)).
    const auto* d = s.data<float>();
    for (int64_t i = 0; i < s.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(d[i]))
            << "Laplace sample[" << i << "] = " << d[i] << " is not finite";
    }
}

TEST_F(LaplaceDistributionTest, SampleMeanApproximatesLoc) {
    // With a large sample and scale=1, the sample mean should be near loc=2.
    auto loc = full({1}, 2.0, DType::Float32, Device::cpu());
    auto scale = ones({1}, DType::Float32, Device::cpu());
    distributions::Laplace dist(loc, scale);

    constexpr int64_t N = 8192;
    auto s = dist.sample({N});
    const auto* d = s.data<float>();
    double sum = 0.0;
    for (int64_t i = 0; i < N; ++i) sum += d[i];
    double sample_mean = sum / static_cast<double>(N);

    // With N=8192 and variance 2, the Monte Carlo error on the mean is
    // ~sqrt(2/8192) ~= 0.016. A loose tolerance of 0.15 is comfortably
    // above that and catches gross bias.
    EXPECT_NEAR(sample_mean, 2.0, 0.15)
        << "Laplace sample mean " << sample_mean
        << " diverges from loc=2.0 beyond Monte Carlo noise";
}

} // namespace
} // namespace tenzor
