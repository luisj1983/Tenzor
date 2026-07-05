/**
 * @file test_sampling_parity.cpp
 * @brief Backend parity tests for sampling ops (Phase 3.8).
 *
 * Three-tier strategy (chosen because cross-backend RNG bit-identical
 * reproducibility is neither achievable nor desired):
 *
 *   1. Deterministic ops (histogram, bucketize) — test exact equality.
 *   2. Stochastic ops with known distribution (bernoulli, multinomial,
 *      poisson, exponential) — draw large sample, compare empirical moments
 *      to the analytic expectation with a generous tolerance.
 *   3. Where even empirical moments are fragile, assert only that the output
 *      shape and dtype are correct and that values lie in the valid support.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/advanced.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;


class SamplingParity : public BackendTest {};
// ============================================================================
// Tier 1 — deterministic
// ============================================================================

TEST_P(SamplingParity, Bucketize_Deterministic) {
    // Deterministic: bucketize(input, boundaries) -> bucket indices.
    auto input = randn({4, 16}, DType::Float32, Device::cpu());
    auto boundaries_cpu = zeros({5}, DType::Float32, Device::cpu());
    // Strictly increasing bucket boundaries at {-1, -0.5, 0, 0.5, 1}.
    auto* b = boundaries_cpu.data<float>();
    b[0] = -1.0f; b[1] = -0.5f; b[2] = 0.0f; b[3] = 0.5f; b[4] = 1.0f;

    test_operation_parity(
        [&boundaries_cpu](const std::vector<Tensor>& ins) {
            // Boundaries follow the same device as input via the second arg.
            return tenzor::bucketize(ins[0], boundaries_cpu.to(ins[0].device()),
                                      /*right=*/false);
        },
        {input}, 0.0f, 0.0f, "bucketize");
}

TEST_P(SamplingParity, Histogram_Deterministic) {
    auto input = randn({1024}, DType::Float32, Device::cpu());

    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            // histogram returns the bin counts — deterministic per input.
            auto [counts, edges] = tenzor::histogram(ins[0], 10, -3.0, 3.0);
            return counts;
        },
        {input}, 0.0f, 0.0f, "histogram");
}

// ============================================================================
// Tier 2 — stochastic, statistical-moment parity
// ============================================================================

TEST_P(SamplingParity, Bernoulli_EmpiricalProbability) {
    // With p=0.3, empirical mean should approach 0.3 for N=10000.
    // Std of mean = sqrt(p*(1-p)/N) ~= 0.0046; use 4-sigma tolerance ~= 0.02.
    auto probs = Tensor({1, 10000}, DType::Float32, Device::cpu());
    auto* p = probs.data<float>();
    for (int64_t i = 0; i < probs.numel(); ++i) p[i] = 0.3f;

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("sampling parity");

    for (const auto& dev : backends) {
        try {
            auto probs_dev = probs.to(dev);
            auto samples = tenzor::bernoulli(probs_dev);
            dev.synchronize();
            float mean = tenzor::mean(samples.to(Device::cpu())).item<float>();
            SCOPED_TRACE(std::string("Bernoulli on ") + backend_name(dev));
            EXPECT_NEAR(mean, 0.3f, 0.02f)
                << "Empirical mean deviates more than 4-sigma from analytic";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Bernoulli failed on " << backend_name(dev)
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(SamplingParity, Poisson_EmpiricalMean) {
    // For Poisson(rate), E[X] = rate. With rate=5, N=4000 samples, std of
    // mean = sqrt(5/4000) ~= 0.035. Use 4-sigma ~= 0.14.
    auto rates = Tensor({1, 4000}, DType::Float32, Device::cpu());
    auto* r = rates.data<float>();
    for (int64_t i = 0; i < rates.numel(); ++i) r[i] = 5.0f;

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("sampling parity");

    for (const auto& dev : backends) {
        try {
            auto rates_dev = rates.to(dev);
            auto samples = tenzor::poisson(rates_dev);
            dev.synchronize();
            float mean = tenzor::mean(samples.to(Device::cpu())).item<float>();
            SCOPED_TRACE(std::string("Poisson on ") + backend_name(dev));
            EXPECT_NEAR(mean, 5.0f, 0.14f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Poisson failed on " << backend_name(dev)
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(SamplingParity, Exponential_EmpiricalMean) {
    // For Exponential(rate), E[X] = 1/rate. With rate=2, N=4000, std of
    // mean = 1/(rate*sqrt(N)) = 1/(2*63.2) ~= 0.0079. Use 4-sigma ~= 0.03.
    auto rates = Tensor({1, 4000}, DType::Float32, Device::cpu());
    auto* r = rates.data<float>();
    for (int64_t i = 0; i < rates.numel(); ++i) r[i] = 2.0f;

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("sampling parity");

    for (const auto& dev : backends) {
        try {
            auto rates_dev = rates.to(dev);
            auto samples = tenzor::exponential(rates_dev);
            dev.synchronize();
            float mean = tenzor::mean(samples.to(Device::cpu())).item<float>();
            SCOPED_TRACE(std::string("Exponential on ") + backend_name(dev));
            EXPECT_NEAR(mean, 0.5f, 0.03f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Exponential failed on " << backend_name(dev)
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(SamplingParity, Multinomial_CategoryFrequencies) {
    // Drawing 10000 samples from a 4-category distribution with weights
    // [0.1, 0.2, 0.3, 0.4]. Empirical frequency of each category should
    // approach its probability within a few sigmas.
    auto probs = zeros({4}, DType::Float32, Device::cpu());
    auto* pp = probs.data<float>();
    pp[0] = 0.1f; pp[1] = 0.2f; pp[2] = 0.3f; pp[3] = 0.4f;

    const int64_t N = 10000;
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("sampling parity");

    for (const auto& dev : backends) {
        try {
            auto probs_dev = probs.to(dev);
            auto samples = tenzor::multinomial(probs_dev, N, /*replacement=*/true);
            dev.synchronize();
            auto samples_cpu = samples.to(Device::cpu());
            // samples shape is [N]; count category frequencies.
            const auto* s = samples_cpu.data<int64_t>();
            int64_t counts[4] = {0, 0, 0, 0};
            for (int64_t i = 0; i < N; ++i) {
                ASSERT_GE(s[i], 0);
                ASSERT_LT(s[i], 4);
                counts[s[i]]++;
            }
            SCOPED_TRACE(std::string("Multinomial on ") + backend_name(dev));
            // 4-sigma tolerance ~= 4 * sqrt(p*(1-p)/N)
            for (int k = 0; k < 4; ++k) {
                float p_true = pp[k];
                float p_emp = static_cast<float>(counts[k]) / N;
                float tol = 4.0f * std::sqrt(p_true * (1.0f - p_true) / N);
                EXPECT_NEAR(p_emp, p_true, tol)
                    << "category " << k << " frequency off: emp=" << p_emp
                    << ", true=" << p_true;
            }
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Multinomial failed on " << backend_name(dev)
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(SamplingParity, NormalSample_EmpiricalMomentss) {
    // randn produces N(0, 1). With N=10000, std of mean ≈ 0.01; 4-sigma ≈ 0.04.
    // Variance should be close to 1 with tolerance 4*sqrt(2/(N-1)) ≈ 0.057.
    const int64_t N = 10000;
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("sampling parity");

    for (const auto& dev : backends) {
        try {
            auto samples = randn({N}, DType::Float32, dev);
            dev.synchronize();
            auto samples_cpu = samples.to(Device::cpu());
            float mean = tenzor::mean(samples_cpu).item<float>();
            float var_val = tenzor::var(samples_cpu).item<float>();
            SCOPED_TRACE(std::string("randn on ") + backend_name(dev));
            EXPECT_NEAR(mean, 0.0f, 0.04f);
            EXPECT_NEAR(var_val, 1.0f, 0.06f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "randn failed on " << backend_name(dev) << ": "
                      << e.what() << std::endl;
        }
    }
}

TEST_P(SamplingParity, Uniform_EmpiricalMoments) {
    // rand() produces U(0, 1). E[X] = 0.5, Var[X] = 1/12 ≈ 0.0833.
    // Std of mean = sqrt((1/12)/N) ≈ 0.0029 for N=10000 → 4-sigma ≈ 0.012.
    const int64_t N = 10000;
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("sampling parity");

    for (const auto& dev : backends) {
        try {
            auto samples = rand({N}, DType::Float32, dev);
            dev.synchronize();
            auto samples_cpu = samples.to(Device::cpu());
            float mean = tenzor::mean(samples_cpu).item<float>();
            float min_val = tenzor::min(samples_cpu).item<float>();
            float max_val = tenzor::max(samples_cpu).item<float>();
            SCOPED_TRACE(std::string("rand (uniform) on ") + backend_name(dev));
            EXPECT_NEAR(mean, 0.5f, 0.02f);
            EXPECT_GE(min_val, 0.0f);
            EXPECT_LE(max_val, 1.0f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "rand failed on " << backend_name(dev) << ": "
                      << e.what() << std::endl;
        }
    }
}

// F042: rand/randn/randint now share a Philox RNG stream across CPU and GPU.
// For the same manual_seed, rand and randint are BIT-IDENTICAL; randn matches to
// within Box-Muller transcendental ULP (host libm vs device intrinsics).
TEST_P(SamplingParity, RandStreamMatchesCPU) {
    if (device.type == Device::Type::CPU) return;
    manual_seed(20240705);
    auto ref32 = rand({128}, DType::Float32, Device::cpu());
    manual_seed(20240705);
    auto out32 = rand({128}, DType::Float32, device).to(Device::cpu());
    ASSERT_EQ(ref32.numel(), out32.numel());
    for (int64_t i = 0; i < ref32.numel(); ++i)
        EXPECT_EQ(ref32.data<float>()[i], out32.data<float>()[i]) << "rand f32 " << i;

    manual_seed(20240705);
    auto ref64 = rand({128}, DType::Float64, Device::cpu());
    manual_seed(20240705);
    auto out64 = rand({128}, DType::Float64, device).to(Device::cpu());
    for (int64_t i = 0; i < ref64.numel(); ++i)
        EXPECT_EQ(ref64.data<double>()[i], out64.data<double>()[i]) << "rand f64 " << i;
}

TEST_P(SamplingParity, RandintStreamMatchesCPU) {
    if (device.type == Device::Type::CPU) return;
    manual_seed(31337);
    auto ref = randint(0, 100000, {128}, DType::Int64, Device::cpu());
    manual_seed(31337);
    auto out = randint(0, 100000, {128}, DType::Int64, device).to(Device::cpu());
    ASSERT_EQ(ref.numel(), out.numel());
    for (int64_t i = 0; i < ref.numel(); ++i)
        EXPECT_EQ(ref.data<int64_t>()[i], out.data<int64_t>()[i]) << "randint " << i;
}

TEST_P(SamplingParity, RandnStreamMatchesCPU) {
    if (device.type == Device::Type::CPU) return;
    manual_seed(4242);
    auto ref = randn({128}, DType::Float32, Device::cpu());
    manual_seed(4242);
    auto out = randn({128}, DType::Float32, device).to(Device::cpu());
    ASSERT_EQ(ref.numel(), out.numel());
    for (int64_t i = 0; i < ref.numel(); ++i)
        EXPECT_NEAR(ref.data<float>()[i], out.data<float>()[i], 1e-3f) << "randn " << i;
}

INSTANTIATE_BACKEND_TESTS(SamplingParity);


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }
    int result = RUN_ALL_TESTS();
    try { tenzor::finalize(); } catch (...) {}
    return result;
}
