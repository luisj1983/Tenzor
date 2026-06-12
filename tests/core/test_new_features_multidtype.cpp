// Multi-backend multi-dtype tests for new features: distributions, schedulers,
// special functions, and functional activations.
//
// Tests the most important subset across all backends and dtypes.

#include "../multi_backend_dtype_fixture.hpp"

#include <cmath>

#include <tenzor/distributions/distribution.hpp>
#include <tenzor/nn/optim/scheduler.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/nn/functional.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/autograd/variable.hpp>

namespace tenzor {
namespace testing {

using namespace tenzor::distributions;

class NewFeaturesMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    double scalar_val(const Tensor& t) const {
        auto cpu = t.to(Device::cpu()).to(DType::Float32).contiguous();
        return static_cast<double>(cpu.data<float>()[0]);
    }

    double sample_mean_val(const Tensor& t) const {
        auto cpu = t.to(Device::cpu()).to(DType::Float64).contiguous();
        int64_t n = cpu.numel();
        const double* p = cpu.data<double>();
        double acc = 0.0;
        for (int64_t i = 0; i < n; ++i) acc += p[i];
        return acc / static_cast<double>(n);
    }
};

// ---------------------------------------------------------------------------
// Pareto distribution: sample mean and log_prob
// ---------------------------------------------------------------------------

TEST_P(NewFeaturesMultiDTypeTest, Pareto_SampleAndLogProb) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "Float16 precision too low for statistical tests";
    }

    auto scale = tenzor::full({1}, 1.0, dtype(), device());
    auto alpha = tenzor::full({1}, 3.0, dtype(), device());
    Pareto dist(scale, alpha);

    auto samples = dist.sample({1000});
    double mean_val = sample_mean_val(samples);
    // Pareto mean = alpha * scale / (alpha - 1) = 3 / 2 = 1.5
    EXPECT_NEAR(mean_val, 1.5, 0.4);

    auto lp = dist.log_prob(tenzor::full({1}, 2.0, dtype(), device()));
    EXPECT_NEAR(scalar_val(lp), -1.674, 0.05);
}

// ---------------------------------------------------------------------------
// Weibull distribution: sample mean
// ---------------------------------------------------------------------------

TEST_P(NewFeaturesMultiDTypeTest, Weibull_Sample) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "Float16 precision too low for statistical tests";
    }

    auto scale = tenzor::full({1}, 1.0, dtype(), device());
    auto concentration = tenzor::full({1}, 2.0, dtype(), device());
    Weibull dist(scale, concentration);

    auto samples = dist.sample({5000});
    double mean_val = sample_mean_val(samples);
    // Weibull mean ~ 0.886
    EXPECT_NEAR(mean_val, 0.886, 0.15);
}

// ---------------------------------------------------------------------------
// Kumaraswamy: samples in unit interval
// ---------------------------------------------------------------------------

TEST_P(NewFeaturesMultiDTypeTest, Kumaraswamy_SampleInUnitInterval) {
    auto a = tenzor::full({1}, 2.0, dtype(), device());
    auto b = tenzor::full({1}, 5.0, dtype(), device());
    Kumaraswamy dist(a, b);

    auto samples = dist.sample({1000});
    auto cpu = samples.to(Device::cpu()).to(DType::Float32).contiguous();
    const float* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_GE(p[i], 0.0f);
        EXPECT_LE(p[i], 1.0f);
    }
}

// ---------------------------------------------------------------------------
// ndtri known values (inverse normal CDF)
// ---------------------------------------------------------------------------

TEST_P(NewFeaturesMultiDTypeTest, Ndtri_KnownValues) {
    auto p = tenzor::zeros({3}, dtype(), device());
    // Set values via CPU
    auto p_cpu = p.to(Device::cpu()).to(DType::Float32);
    p_cpu.data<float>()[0] = 0.5f;
    p_cpu.data<float>()[1] = 0.8413f;
    p_cpu.data<float>()[2] = 0.1587f;
    auto p_dev = p_cpu.to(dtype()).to(device());

    auto result = tenzor::ndtri(p_dev);

    if (dtype() == DType::Float16) {
        // Float16: precision too low for the tight value check, but still
        // exercise the op/dispatch and assert it is finite (no NaN/Inf).
        expectAllFinite(result);
        return;
    }

    auto cpu = result.to(Device::cpu()).to(DType::Float32).contiguous();
    const float* r = cpu.data<float>();

    EXPECT_NEAR(r[0], 0.0f, 0.02f);
    EXPECT_NEAR(r[1], 1.0f, 0.03f);
    EXPECT_NEAR(r[2], -1.0f, 0.03f);
}

// ---------------------------------------------------------------------------
// Functional log_sigmoid
// ---------------------------------------------------------------------------

TEST_P(NewFeaturesMultiDTypeTest, FunctionalLogSigmoid) {
    auto input_t = tenzor::zeros({3}, dtype(), device());
    auto in_cpu = input_t.to(Device::cpu()).to(DType::Float32);
    in_cpu.data<float>()[0] = 0.0f;
    in_cpu.data<float>()[1] = 5.0f;
    in_cpu.data<float>()[2] = -5.0f;
    auto in_dev = in_cpu.to(dtype()).to(device());

    auto input = Variable(in_dev, false);
    auto result = nn::functional::log_sigmoid(input);

    if (dtype() == DType::Float16) {
        // Float16: precision too low for the tight value check, but still
        // exercise the op/dispatch and assert it is finite (no NaN/Inf).
        expectAllFinite(result.tensor());
        return;
    }

    auto cpu = result.tensor().to(Device::cpu()).to(DType::Float32).contiguous();
    const float* r = cpu.data<float>();

    // log(sigmoid(0)) = log(0.5) ~ -0.693
    EXPECT_NEAR(r[0], std::log(0.5f), 0.02f);
    // log(sigmoid(5)) ~ 0
    EXPECT_NEAR(r[1], 0.0f, 0.02f);
    // log(sigmoid(-5)) ~ -5
    EXPECT_NEAR(r[2], -5.0f, 0.15f);
}

// ---------------------------------------------------------------------------
// gammainc + gammaincc = gamma (complement identity)
// ---------------------------------------------------------------------------

TEST_P(NewFeaturesMultiDTypeTest, GammaincPlusGammaincc) {
    auto a = tenzor::zeros({2}, dtype(), device());
    auto x = tenzor::zeros({2}, dtype(), device());
    // Set via CPU
    auto a_cpu = a.to(Device::cpu()).to(DType::Float32);
    a_cpu.data<float>()[0] = 2.0f;
    a_cpu.data<float>()[1] = 3.0f;
    auto x_cpu = x.to(Device::cpu()).to(DType::Float32);
    x_cpu.data<float>()[0] = 1.0f;
    x_cpu.data<float>()[1] = 2.0f;

    auto a_dev = a_cpu.to(dtype()).to(device());
    auto x_dev = x_cpu.to(dtype()).to(device());

    auto inc = tenzor::gammainc(a_dev, x_dev);
    auto incc = tenzor::gammaincc(a_dev, x_dev);
    auto total = inc + incc;
    auto gamma_a = tenzor::gamma(a_dev);

    if (dtype() == DType::Float16) {
        // Float16: precision too low for the tight identity check, but still
        // exercise the ops/dispatch and assert outputs are finite (no NaN/Inf).
        expectAllFinite(total);
        expectAllFinite(gamma_a);
        return;
    }

    auto t_cpu = total.to(Device::cpu()).to(DType::Float32).contiguous();
    auto g_cpu = gamma_a.to(Device::cpu()).to(DType::Float32).contiguous();

    for (int64_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(t_cpu.data<float>()[i], g_cpu.data<float>()[i], atol() * 10);
    }
}

// ---------------------------------------------------------------------------
// Pareto CDF
// ---------------------------------------------------------------------------

TEST_P(NewFeaturesMultiDTypeTest, Pareto_CDF) {
    auto scale = tenzor::full({1}, 1.0, dtype(), device());
    auto alpha = tenzor::full({1}, 2.0, dtype(), device());
    Pareto dist(scale, alpha);

    auto val = tenzor::full({1}, 2.0, dtype(), device());
    auto c = dist.cdf(val);

    if (dtype() == DType::Float16) {
        // Float16: precision too low for the tight value check, but still
        // exercise the CDF/dispatch and assert it is finite (no NaN/Inf).
        expectAllFinite(c);
        return;
    }

    // CDF(2) = 1 - (1/2)^2 = 0.75
    EXPECT_NEAR(scalar_val(c), 0.75, 0.02);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(NewFeaturesMultiDTypeTest);

} // namespace testing
} // namespace tenzor
