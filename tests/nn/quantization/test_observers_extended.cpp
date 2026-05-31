#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/quantization/observer.hpp>
#include <tenzor/nn/quantization/quantize.hpp>
#include "observers_extended_test_support.hpp"
#include <mutex>

using namespace tenzor;
using namespace tenzor::nn::quantization;

class ObserversExtendedTest : public ::testing::Test {
protected:
    void SetUp() override {
        static std::once_flag init_flag;
        std::call_once(init_flag, []() { tenzor::initialize(); });
    }
};

// ===========================================================================
// KLDivergenceObserver
// ===========================================================================

TEST_F(ObserversExtendedTest, KLDivergenceConstruction) {
    KLDivergenceObserver obs;
    // Default: num_bins=2048, dtype=Float32
}

TEST_F(ObserversExtendedTest, KLDivergenceCustomBins) {
    KLDivergenceObserver obs(1024, 128);
}

TEST_F(ObserversExtendedTest, KLDivergenceObserveDoesNotCrash) {
    KLDivergenceObserver obs;
    auto input = tenzor::randn({32, 64});
    EXPECT_NO_THROW(obs.observe(input));
}

TEST_F(ObserversExtendedTest, KLDivergenceQParamsValid) {
    KLDivergenceObserver obs;
    obs.observe(tenzor::randn({64, 128}));

    auto qparams = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_val = qparams.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];
    EXPECT_GT(scale_val, 0.0f) << "Scale must be positive";
}

TEST_F(ObserversExtendedTest, KLDivergenceMultipleObserves) {
    KLDivergenceObserver obs;
    obs.observe(tenzor::randn({16, 32}));
    obs.observe(tenzor::randn({16, 32}) * 2.0f);
    obs.observe(tenzor::randn({16, 32}) * 5.0f);

    auto qparams = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_val = qparams.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];
    EXPECT_GT(scale_val, 0.0f) << "Scale must be positive after multiple observations";
}

TEST_F(ObserversExtendedTest, KLDivergenceReset) {
    KLDivergenceObserver obs;
    obs.observe(tenzor::randn({32, 64}) * 10.0f);

    auto qp_before = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_before = qp_before.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];

    obs.reset();
    obs.observe(tenzor::randn({32, 64}) * 0.01f);

    auto qp_after = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_after = qp_after.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];
    EXPECT_LT(scale_after, scale_before)
        << "After reset with smaller data, scale should decrease";
}

// ===========================================================================
// PercentileObserver
// ===========================================================================

TEST_F(ObserversExtendedTest, PercentileConstruction) {
    PercentileObserver obs;
}

TEST_F(ObserversExtendedTest, PercentileCustomPercentile) {
    PercentileObserver obs(0.001, 0.999);
}

TEST_F(ObserversExtendedTest, PercentileObserveDoesNotCrash) {
    PercentileObserver obs;
    auto input = tenzor::randn({32, 64});
    EXPECT_NO_THROW(obs.observe(input));
}

TEST_F(ObserversExtendedTest, PercentileQParamsValid) {
    PercentileObserver obs;
    obs.observe(tenzor::randn({64, 128}));

    auto qparams = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_val = qparams.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];
    EXPECT_GT(scale_val, 0.0f) << "Scale must be positive";
}

TEST_F(ObserversExtendedTest, PercentileMultipleObserves) {
    PercentileObserver obs;
    obs.observe(tenzor::randn({16, 32}));
    obs.observe(tenzor::randn({16, 32}) * 3.0f);

    auto qparams = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_val = qparams.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];
    EXPECT_GT(scale_val, 0.0f);
}

TEST_F(ObserversExtendedTest, PercentileReset) {
    PercentileObserver obs;
    obs.observe(tenzor::randn({32, 64}) * 10.0f);

    auto qp_before = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_before = qp_before.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];

    obs.reset();
    obs.observe(tenzor::randn({32, 64}) * 0.01f);

    auto qp_after = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_after = qp_after.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];
    EXPECT_LT(scale_after, scale_before)
        << "After reset with smaller data, scale should decrease";
}

TEST_F(ObserversExtendedTest, PercentileTighterPercentileSmallerScale) {
    PercentileObserver obs_wide(0.0001, 0.9999);
    PercentileObserver obs_tight(0.05, 0.95);

    auto input = tenzor::randn({256, 128});
    obs_wide.observe(input);
    obs_tight.observe(input);

    auto qp_wide = obs_wide.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto qp_tight = obs_tight.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_wide = qp_wide.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];
    auto scale_tight = qp_tight.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];

    EXPECT_LE(scale_tight, scale_wide)
        << "Tighter percentile should produce smaller or equal scale";
}

// ===========================================================================
// MSEObserver
// ===========================================================================

TEST_F(ObserversExtendedTest, MSEConstruction) {
    MSEObserver obs;
}

TEST_F(ObserversExtendedTest, MSECustomCandidates) {
    MSEObserver obs(50);
}

TEST_F(ObserversExtendedTest, MSEObserveDoesNotCrash) {
    MSEObserver obs;
    auto input = tenzor::randn({32, 64});
    EXPECT_NO_THROW(obs.observe(input));
}

TEST_F(ObserversExtendedTest, MSEQParamsValid) {
    MSEObserver obs;
    obs.observe(tenzor::randn({64, 128}));

    auto qparams = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_val = qparams.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];
    EXPECT_GT(scale_val, 0.0f) << "Scale must be positive";
}

TEST_F(ObserversExtendedTest, MSEMultipleObserves) {
    MSEObserver obs;
    obs.observe(tenzor::randn({16, 32}));
    obs.observe(tenzor::randn({16, 32}) * 4.0f);

    auto qparams = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_val = qparams.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];
    EXPECT_GT(scale_val, 0.0f);
}

TEST_F(ObserversExtendedTest, MSEReset) {
    MSEObserver obs;
    obs.observe(tenzor::randn({32, 64}) * 10.0f);

    auto qp_before = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_before = qp_before.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];

    obs.reset();
    obs.observe(tenzor::randn({32, 64}) * 0.01f);

    auto qp_after = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_after = qp_after.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];
    EXPECT_LT(scale_after, scale_before)
        << "After reset with smaller data, scale should decrease";
}

// ===========================================================================
// Per-channel MinMaxObserver (per_channel_ = true)
// ===========================================================================

TEST_F(ObserversExtendedTest, PerChannelMinMaxObserveDoesNotCrash) {
    MinMaxObserver obs(/*per_channel=*/true);
    auto input = tenzor::randn({8, 16, 4, 4});
    EXPECT_NO_THROW(obs.observe(input));
}

TEST_F(ObserversExtendedTest, PerChannelMinMaxQParamsValid) {
    MinMaxObserver obs(/*per_channel=*/true);
    obs.observe(tenzor::randn({8, 16, 4, 4}));

    auto qparams = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_val = qparams.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];
    EXPECT_GT(scale_val, 0.0f) << "Scale must be positive";
}

TEST_F(ObserversExtendedTest, PerChannelMinMaxReset) {
    MinMaxObserver obs(/*per_channel=*/true);
    obs.observe(tenzor::randn({4, 8, 3, 3}) * 10.0f);

    auto qp_before = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_before = qp_before.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];

    obs.reset();
    obs.observe(tenzor::randn({4, 8, 3, 3}) * 0.01f);

    auto qp_after = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_after = qp_after.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];
    EXPECT_LT(scale_after, scale_before)
        << "After reset with smaller data, scale should decrease";
}

TEST_F(ObserversExtendedTest, PerChannelMinMaxMultipleObservesAccumulate) {
    MinMaxObserver obs(/*per_channel=*/true);

    obs.observe(tenzor::randn({4, 8}) * 0.1f);
    auto qp_small = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_small = qp_small.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];

    obs.observe(tenzor::randn({4, 8}) * 10.0f);
    auto qp_large = obs.calculate_qparams(QuantDType::INT8, QuantizationScheme::PerTensorSymmetric);
    auto scale_large = qp_large.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];

    EXPECT_GE(scale_large, scale_small)
        << "Scale should grow as observed range expands";
}
