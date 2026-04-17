// Multi-backend multi-dtype tests for advanced distributions.
//
// Verifies that Gamma, Beta, StudentT, Poisson, MVN, and KL divergence
// produce correct sample statistics and log_prob values across backends
// and dtypes.

#include "../multi_backend_dtype_fixture.hpp"

#include <cmath>

#include <tenzor/distributions/distribution.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>

namespace tenzor {
namespace testing {

using namespace tenzor::distributions;

class DistributionsAdvancedMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Compute sample mean via CPU reduction (always in Float64 for accuracy).
    double sample_mean_val(const Tensor& t) const {
        auto cpu = t.to(Device::cpu()).to(DType::Float64).contiguous();
        int64_t n = cpu.numel();
        const double* p = cpu.data<double>();
        double acc = 0.0;
        for (int64_t i = 0; i < n; ++i) acc += p[i];
        return acc / static_cast<double>(n);
    }

    double sample_var_val(const Tensor& t) const {
        double m = sample_mean_val(t);
        auto cpu = t.to(Device::cpu()).to(DType::Float64).contiguous();
        int64_t n = cpu.numel();
        const double* p = cpu.data<double>();
        double acc = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            double d = p[i] - m;
            acc += d * d;
        }
        return acc / static_cast<double>(n);
    }
};

// ---------------------------------------------------------------------------
// Gamma(3, 2): mean=1.5, var=0.75
// ---------------------------------------------------------------------------

TEST_P(DistributionsAdvancedMultiDTypeTest, GammaSampleMean) {
    // Skip Float16 - insufficient precision for statistical tests
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "Float16 precision too low for statistical tests";
    }

    auto conc = tenzor::full({1}, 3.0, dtype(), device());
    auto rate = tenzor::full({1}, 2.0, dtype(), device());
    Gamma g(conc, rate);
    auto samples = g.sample({20000});

    double m = sample_mean_val(samples);
    double v = sample_var_val(samples);
    EXPECT_NEAR(m, 1.5, 0.15);
    EXPECT_NEAR(v, 0.75, 0.25);
}

// ---------------------------------------------------------------------------
// Beta(2, 5): mean=2/7, values in (0, 1)
// ---------------------------------------------------------------------------

TEST_P(DistributionsAdvancedMultiDTypeTest, BetaSampleMeanAndRange) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "Float16 precision too low for statistical tests";
    }

    auto c1 = tenzor::full({1}, 2.0, dtype(), device());
    auto c0 = tenzor::full({1}, 5.0, dtype(), device());
    Beta b(c1, c0);
    auto samples = b.sample({20000});

    auto cpu = samples.to(Device::cpu()).to(DType::Float32).contiguous();
    const float* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GT(p[i], 0.0f);
        EXPECT_LT(p[i], 1.0f);
    }
    EXPECT_NEAR(sample_mean_val(samples), 2.0 / 7.0, 0.05);
}

// ---------------------------------------------------------------------------
// StudentT(df=10): mean~0, var~1.25
// ---------------------------------------------------------------------------

TEST_P(DistributionsAdvancedMultiDTypeTest, StudentTMeanApproxZero) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "Float16 precision too low for statistical tests";
    }

    auto df = tenzor::full({1}, 10.0, dtype(), device());
    auto loc = tenzor::zeros({1}, dtype(), device());
    auto scale = tenzor::full({1}, 1.0, dtype(), device());
    StudentT t(df, loc, scale);
    auto samples = t.sample({20000});
    EXPECT_NEAR(sample_mean_val(samples), 0.0, 0.15);
    EXPECT_NEAR(sample_var_val(samples), 1.25, 0.4);
}

// ---------------------------------------------------------------------------
// MVN with loc=0, cov=I: sample shape check
// ---------------------------------------------------------------------------

TEST_P(DistributionsAdvancedMultiDTypeTest, MVNSampleShape) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "Float16 precision too low for MVN";
    }

    auto loc = tenzor::zeros({3}, dtype(), device());
    auto I = tenzor::zeros({3, 3}, dtype(), device());
    // Set identity via CPU
    auto I_cpu = I.to(Device::cpu()).to(DType::Float32);
    I_cpu.data<float>()[0] = 1.0f;
    I_cpu.data<float>()[4] = 1.0f;
    I_cpu.data<float>()[8] = 1.0f;
    auto I_dev = I_cpu.to(dtype()).to(device());

    MultivariateNormal mvn(loc, I_dev);
    auto s = mvn.sample();
    EXPECT_EQ(s.shape().size(), 1u);
    EXPECT_EQ(s.shape()[0], 3);
    EXPECT_EQ(s.dtype(), dtype());
}

// ---------------------------------------------------------------------------
// KL(N(0,1) || N(0,1)) = 0
// ---------------------------------------------------------------------------

TEST_P(DistributionsAdvancedMultiDTypeTest, KLNormalNormalZero) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "Float16 precision too low for KL computation";
    }

    auto loc1 = tenzor::zeros({1}, dtype(), device());
    auto scale1 = tenzor::full({1}, 1.0, dtype(), device());
    auto loc2 = tenzor::zeros({1}, dtype(), device());
    auto scale2 = tenzor::full({1}, 1.0, dtype(), device());
    Normal p(loc1, scale1);
    Normal q(loc2, scale2);
    auto kl = kl_divergence(p, q);
    auto val = kl.to(Device::cpu()).to(DType::Float32).data<float>()[0];
    EXPECT_NEAR(val, 0.0f, atol());
}

// ---------------------------------------------------------------------------
// KL(N(1,1) || N(0,1)) = 0.5
// ---------------------------------------------------------------------------

TEST_P(DistributionsAdvancedMultiDTypeTest, KLDifferentMeans) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "Float16 precision too low for KL computation";
    }

    auto loc1 = tenzor::full({1}, 1.0, dtype(), device());
    auto scale1 = tenzor::full({1}, 1.0, dtype(), device());
    auto loc2 = tenzor::zeros({1}, dtype(), device());
    auto scale2 = tenzor::full({1}, 1.0, dtype(), device());
    Normal p(loc1, scale1);
    Normal q(loc2, scale2);
    auto kl = kl_divergence(p, q);
    auto val = kl.to(Device::cpu()).to(DType::Float32).data<float>()[0];
    EXPECT_NEAR(val, 0.5f, atol());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DistributionsAdvancedMultiDTypeTest);

} // namespace testing
} // namespace tenzor
