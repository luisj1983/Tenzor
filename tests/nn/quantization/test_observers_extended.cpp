#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/quantization/observer.hpp>
#include <tenzor/nn/quantization/quantize.hpp>
#include "observers_extended_test_support.hpp"
#include "../../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn::quantization;

class ObserversExtendedTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ===========================================================================
// KLDivergenceObserver
// ===========================================================================

TEST_P(ObserversExtendedTest, KLDivergenceConstruction) {
    KLDivergenceObserver obs;
    // Default: num_bins=2048, dtype=Float32
}

TEST_P(ObserversExtendedTest, KLDivergenceCustomBins) {
    KLDivergenceObserver obs(1024, 128);
}

TEST_P(ObserversExtendedTest, KLDivergenceObserveDoesNotCrash) {
    KLDivergenceObserver obs;
    auto input = tenzor::randn({32, 64}, DType::Float32, device);
    EXPECT_NO_THROW(obs.observe(input));
}

TEST_P(ObserversExtendedTest, KLDivergenceQParamsValid) {
    KLDivergenceObserver obs;
    obs.observe(tenzor::randn({64, 128}, DType::Float32, device));

    auto qparams = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_cpu = qparams.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_val = scale_cpu.data<float>()[0];
    EXPECT_GT(scale_val, 0.0f) << "Scale must be positive";
}

TEST_P(ObserversExtendedTest, KLDivergenceMultipleObserves) {
    KLDivergenceObserver obs;
    obs.observe(tenzor::randn({16, 32}, DType::Float32, device));
    obs.observe(tenzor::randn({16, 32}, DType::Float32, device) * 2.0f);
    obs.observe(tenzor::randn({16, 32}, DType::Float32, device) * 5.0f);

    auto qparams = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_cpu = qparams.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_val = scale_cpu.data<float>()[0];
    EXPECT_GT(scale_val, 0.0f) << "Scale must be positive after multiple observations";
}

TEST_P(ObserversExtendedTest, KLDivergenceReset) {
    KLDivergenceObserver obs;
    obs.observe(tenzor::randn({32, 64}, DType::Float32, device) * 10.0f);

    auto qp_before = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_before_cpu = qp_before.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_before = scale_before_cpu.data<float>()[0];

    obs.reset();
    obs.observe(tenzor::randn({32, 64}, DType::Float32, device) * 0.01f);

    auto qp_after = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_after_cpu = qp_after.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_after = scale_after_cpu.data<float>()[0];
    EXPECT_LT(scale_after, scale_before)
        << "After reset with smaller data, scale should decrease";
}

// ===========================================================================
// PercentileObserver
// ===========================================================================

TEST_P(ObserversExtendedTest, PercentileConstruction) {
    PercentileObserver obs;
}

TEST_P(ObserversExtendedTest, PercentileCustomPercentile) {
    PercentileObserver obs(0.001, 0.999);
}

TEST_P(ObserversExtendedTest, PercentileObserveDoesNotCrash) {
    PercentileObserver obs;
    auto input = tenzor::randn({32, 64}, DType::Float32, device);
    EXPECT_NO_THROW(obs.observe(input));
}

TEST_P(ObserversExtendedTest, PercentileQParamsValid) {
    PercentileObserver obs;
    obs.observe(tenzor::randn({64, 128}, DType::Float32, device));

    auto qparams = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_cpu = qparams.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_val = scale_cpu.data<float>()[0];
    EXPECT_GT(scale_val, 0.0f) << "Scale must be positive";
}

TEST_P(ObserversExtendedTest, PercentileMultipleObserves) {
    PercentileObserver obs;
    obs.observe(tenzor::randn({16, 32}, DType::Float32, device));
    obs.observe(tenzor::randn({16, 32}, DType::Float32, device) * 3.0f);

    auto qparams = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_cpu = qparams.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_val = scale_cpu.data<float>()[0];
    EXPECT_GT(scale_val, 0.0f);
}

TEST_P(ObserversExtendedTest, PercentileReset) {
    PercentileObserver obs;
    obs.observe(tenzor::randn({32, 64}, DType::Float32, device) * 10.0f);

    auto qp_before = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_before_cpu = qp_before.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_before = scale_before_cpu.data<float>()[0];

    obs.reset();
    obs.observe(tenzor::randn({32, 64}, DType::Float32, device) * 0.01f);

    auto qp_after = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_after_cpu = qp_after.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_after = scale_after_cpu.data<float>()[0];
    EXPECT_LT(scale_after, scale_before)
        << "After reset with smaller data, scale should decrease";
}

TEST_P(ObserversExtendedTest, PercentileTighterPercentileSmallerScale) {
    PercentileObserver obs_wide(0.0001, 0.9999);
    PercentileObserver obs_tight(0.05, 0.95);

    auto input = tenzor::randn({256, 128}, DType::Float32, device);
    obs_wide.observe(input);
    obs_tight.observe(input);

    auto qp_wide = obs_wide.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto qp_tight = obs_tight.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_wide_cpu = qp_wide.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_tight_cpu = qp_tight.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_wide = scale_wide_cpu.data<float>()[0];
    auto scale_tight = scale_tight_cpu.data<float>()[0];

    EXPECT_LE(scale_tight, scale_wide)
        << "Tighter percentile should produce smaller or equal scale";
}

// ===========================================================================
// MSEObserver
// ===========================================================================

TEST_P(ObserversExtendedTest, MSEConstruction) {
    MSEObserver obs;
}

TEST_P(ObserversExtendedTest, MSECustomCandidates) {
    MSEObserver obs(50);
}

TEST_P(ObserversExtendedTest, MSEObserveDoesNotCrash) {
    MSEObserver obs;
    auto input = tenzor::randn({32, 64}, DType::Float32, device);
    EXPECT_NO_THROW(obs.observe(input));
}

TEST_P(ObserversExtendedTest, MSEQParamsValid) {
    MSEObserver obs;
    obs.observe(tenzor::randn({64, 128}, DType::Float32, device));

    auto qparams = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_cpu = qparams.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_val = scale_cpu.data<float>()[0];
    EXPECT_GT(scale_val, 0.0f) << "Scale must be positive";
}

TEST_P(ObserversExtendedTest, MSEMultipleObserves) {
    MSEObserver obs;
    obs.observe(tenzor::randn({16, 32}, DType::Float32, device));
    obs.observe(tenzor::randn({16, 32}, DType::Float32, device) * 4.0f);

    auto qparams = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_cpu = qparams.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_val = scale_cpu.data<float>()[0];
    EXPECT_GT(scale_val, 0.0f);
}

TEST_P(ObserversExtendedTest, MSEReset) {
    MSEObserver obs;
    obs.observe(tenzor::randn({32, 64}, DType::Float32, device) * 10.0f);

    auto qp_before = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_before_cpu = qp_before.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_before = scale_before_cpu.data<float>()[0];

    obs.reset();
    obs.observe(tenzor::randn({32, 64}, DType::Float32, device) * 0.01f);

    auto qp_after = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_after_cpu = qp_after.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_after = scale_after_cpu.data<float>()[0];
    EXPECT_LT(scale_after, scale_before)
        << "After reset with smaller data, scale should decrease";
}

// ===========================================================================
// Per-channel MinMaxObserver (per_channel_ = true)
// ===========================================================================

TEST_P(ObserversExtendedTest, PerChannelMinMaxObserveDoesNotCrash) {
    MinMaxObserver obs(/*per_channel=*/true);
    auto input = tenzor::randn({8, 16, 4, 4}, DType::Float32, device);
    EXPECT_NO_THROW(obs.observe(input));
}

TEST_P(ObserversExtendedTest, PerChannelMinMaxQParamsValid) {
    MinMaxObserver obs(/*per_channel=*/true);
    obs.observe(tenzor::randn({8, 16, 4, 4}, DType::Float32, device));

    auto qparams = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_cpu = qparams.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_val = scale_cpu.data<float>()[0];
    EXPECT_GT(scale_val, 0.0f) << "Scale must be positive";
}

TEST_P(ObserversExtendedTest, PerChannelMinMaxReset) {
    MinMaxObserver obs(/*per_channel=*/true);
    obs.observe(tenzor::randn({4, 8, 3, 3}, DType::Float32, device) * 10.0f);

    auto qp_before = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_before_cpu = qp_before.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_before = scale_before_cpu.data<float>()[0];

    obs.reset();
    obs.observe(tenzor::randn({4, 8, 3, 3}, DType::Float32, device) * 0.01f);

    auto qp_after = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_after_cpu = qp_after.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_after = scale_after_cpu.data<float>()[0];
    EXPECT_LT(scale_after, scale_before)
        << "After reset with smaller data, scale should decrease";
}

TEST_P(ObserversExtendedTest, PerChannelMinMaxMultipleObservesAccumulate) {
    MinMaxObserver obs(/*per_channel=*/true);

    obs.observe(tenzor::randn({4, 8}, DType::Float32, device) * 0.1f);
    auto qp_small = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_small_cpu = qp_small.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_small = scale_small_cpu.data<float>()[0];

    obs.observe(tenzor::randn({4, 8}, DType::Float32, device) * 10.0f);
    auto qp_large = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_large_cpu = qp_large.scale.to(Device::cpu()).to(DType::Float32);
    auto scale_large = scale_large_cpu.data<float>()[0];

    EXPECT_GE(scale_large, scale_small)
        << "Scale should grow as observed range expands";
}

INSTANTIATE_BACKEND_TESTS(ObserversExtendedTest);
