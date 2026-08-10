/**
 * @file test_observers_extended_multidtype.cpp
 * @brief Multi-dtype / multi-backend companion to test_observers_extended.cpp.
 *
 * The plain file (BackendTest, Float32-only) exercises the extended quantization
 * observer set — KLDivergenceObserver, PercentileObserver, MSEObserver, and
 * per-channel MinMaxObserver — across backends: construction, observe-does-not-
 * crash, qparam validity (scale > 0), multi-observe accumulation, reset
 * behaviour, and (for Percentile) tighter-percentile-smaller-scale.
 *
 * This companion adds the dtype axis across {Float32, Float64, Float16} x
 * {cpu, cuda, vulkan, oneapi, rocm, mps} via MultiBackendDTypeTest. The input
 * tensor observed is created in the test dtype on the test device, so the
 * companion exercises each backend's dtype-conversion + reduction dispatch on
 * the observe() path.
 *
 * Dtype coverage is split by observer, matching what each observer actually
 * supports:
 *   - KLDivergenceObserver::observe and MinMaxObserver::observe upcast any
 *     non-Float32 input to Float32 (observer.cpp:26-29, support header:109),
 *     so they accept Float64 / Float16 inputs. These tests run the FULL
 *     {Float32, Float64, Float16} sweep — the new coverage is the device-side
 *     .to(Float32) cast kernel on each backend for F16/F64 inputs.
 *   - PercentileObserver::observe and MSEObserver::observe (test-only, in
 *     observers_extended_test_support.hpp) call .data<float>() on the tensor
 *     with NO dtype upcast, so a non-Float32 input is a type mismatch. These
 *     observe-based tests skip non-Float32 with DtypeUnsupportedOnBackend
 *     (the observers are Float32-only by construction; the F32 x backend
 *     axis is already covered by the plain file, so the skip loses nothing).
 *   - The pure-construction tests (KLDivergenceConstruction, *CustomBins,
 *     PercentileConstruction, *CustomPercentile, MSEConstruction,
 *     MSECustomCandidates) build an observer with no tensor input and are
 *     dtype-orthogonal; they run on every combo trivially.
 *
 * Qparam scales are read back via a dtype-safe helper (cast to Float32 before
 * .data<float>()[0]), matching the plain file's .to(Float32) readback.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/quantization/observer.hpp>
#include <tenzor/nn/quantization/quantize.hpp>
#include "observers_extended_test_support.hpp"
#include "../../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn::quantization;
using namespace tenzor::testing;

// PercentileObserver / MSEObserver are test-only Float32 observers with no
// dtype upcast in observe() — skip non-Float32 dtypes categorically.
#define skip_if_not_f32_test_observer()                                   \
    do {                                                                  \
        if (dtype() != DType::Float32) {                                  \
            SKIP_WITH_REASON(                                             \
                ::tenzor::testing::SkipReason::DtypeUnsupportedOnBackend, \
                "PercentileObserver/MSEObserver are test-only Float32 "   \
                "observers (no dtype upcast in observe)");                 \
        }                                                                 \
    } while (0)

class ObserversExtendedMultiDType : public MultiBackendDTypeTest {
protected:
    // Build a randn input in the test dtype on the test device, scaled.
    Tensor make_input(const std::vector<int64_t>& shape, float scale = 1.0f) {
        auto t = tenzor::randn(shape, DType::Float32, device()) * scale;
        if (dtype() != DType::Float32) {
            t = t.to(dtype());
        }
        return t;
    }

    // Read qparams.scale[0] as float regardless of the test dtype.
    static float scale_of(const QuantizationParams& qparams) {
        return qparams.scale.to(Device::cpu()).to(DType::Float32).data<float>()[0];
    }

    static const QuantizationScheme kSym = QuantizationScheme::PerTensorSymmetric;
};

// ===========================================================================
// KLDivergenceObserver  (upcasts in observe -> full dtype sweep)
// ===========================================================================

TEST_P(ObserversExtendedMultiDType, KLDivergenceConstruction) {
    KLDivergenceObserver obs;  // default num_bins=2048
}

TEST_P(ObserversExtendedMultiDType, KLDivergenceCustomBins) {
    KLDivergenceObserver obs(1024, 128);
}

TEST_P(ObserversExtendedMultiDType, KLDivergenceObserveDoesNotCrash) {
    KLDivergenceObserver obs;
    EXPECT_NO_THROW(obs.observe(make_input({32, 64})));
}

TEST_P(ObserversExtendedMultiDType, KLDivergenceQParamsValid) {
    KLDivergenceObserver obs;
    obs.observe(make_input({64, 128}));
    EXPECT_GT(scale_of(obs.calculate_qparams(QuantDType::INT8, kSym)), 0.0f)
        << "Scale must be positive";
}

TEST_P(ObserversExtendedMultiDType, KLDivergenceMultipleObserves) {
    KLDivergenceObserver obs;
    obs.observe(make_input({16, 32}));
    obs.observe(make_input({16, 32}, 2.0f));
    obs.observe(make_input({16, 32}, 5.0f));
    EXPECT_GT(scale_of(obs.calculate_qparams(QuantDType::INT8, kSym)), 0.0f)
        << "Scale must be positive after multiple observations";
}

TEST_P(ObserversExtendedMultiDType, KLDivergenceReset) {
    KLDivergenceObserver obs;
    obs.observe(make_input({32, 64}, 10.0f));
    float scale_before = scale_of(obs.calculate_qparams(QuantDType::INT8, kSym));

    obs.reset();
    obs.observe(make_input({32, 64}, 0.01f));
    float scale_after = scale_of(obs.calculate_qparams(QuantDType::INT8, kSym));
    EXPECT_LT(scale_after, scale_before)
        << "After reset with smaller data, scale should decrease";
}

// ===========================================================================
// PercentileObserver  (no upcast in observe -> F32 only)
// ===========================================================================

TEST_P(ObserversExtendedMultiDType, PercentileConstruction) {
    PercentileObserver obs;
}

TEST_P(ObserversExtendedMultiDType, PercentileCustomPercentile) {
    PercentileObserver obs(0.001, 0.999);
}

TEST_P(ObserversExtendedMultiDType, PercentileObserveDoesNotCrash) {
    skip_if_not_f32_test_observer();
    PercentileObserver obs;
    EXPECT_NO_THROW(obs.observe(make_input({32, 64})));
}

TEST_P(ObserversExtendedMultiDType, PercentileQParamsValid) {
    skip_if_not_f32_test_observer();
    PercentileObserver obs;
    obs.observe(make_input({64, 128}));
    EXPECT_GT(scale_of(obs.calculate_qparams(QuantDType::INT8, kSym)), 0.0f)
        << "Scale must be positive";
}

TEST_P(ObserversExtendedMultiDType, PercentileMultipleObserves) {
    skip_if_not_f32_test_observer();
    PercentileObserver obs;
    obs.observe(make_input({16, 32}));
    obs.observe(make_input({16, 32}, 3.0f));
    EXPECT_GT(scale_of(obs.calculate_qparams(QuantDType::INT8, kSym)), 0.0f);
}

TEST_P(ObserversExtendedMultiDType, PercentileReset) {
    skip_if_not_f32_test_observer();
    PercentileObserver obs;
    obs.observe(make_input({32, 64}, 10.0f));
    float scale_before = scale_of(obs.calculate_qparams(QuantDType::INT8, kSym));

    obs.reset();
    obs.observe(make_input({32, 64}, 0.01f));
    float scale_after = scale_of(obs.calculate_qparams(QuantDType::INT8, kSym));
    EXPECT_LT(scale_after, scale_before)
        << "After reset with smaller data, scale should decrease";
}

TEST_P(ObserversExtendedMultiDType, PercentileTighterPercentileSmallerScale) {
    skip_if_not_f32_test_observer();
    PercentileObserver obs_wide(0.0001, 0.9999);
    PercentileObserver obs_tight(0.05, 0.95);

    auto input = make_input({256, 128});
    obs_wide.observe(input);
    obs_tight.observe(input);

    float scale_wide = scale_of(obs_wide.calculate_qparams(QuantDType::INT8, kSym));
    float scale_tight = scale_of(obs_tight.calculate_qparams(QuantDType::INT8, kSym));
    EXPECT_LE(scale_tight, scale_wide)
        << "Tighter percentile should produce smaller or equal scale";
}

// ===========================================================================
// MSEObserver  (no upcast in observe -> F32 only)
// ===========================================================================

TEST_P(ObserversExtendedMultiDType, MSEConstruction) {
    MSEObserver obs;
}

TEST_P(ObserversExtendedMultiDType, MSECustomCandidates) {
    MSEObserver obs(50);
}

TEST_P(ObserversExtendedMultiDType, MSEObserveDoesNotCrash) {
    skip_if_not_f32_test_observer();
    MSEObserver obs;
    EXPECT_NO_THROW(obs.observe(make_input({32, 64})));
}

TEST_P(ObserversExtendedMultiDType, MSEQParamsValid) {
    skip_if_not_f32_test_observer();
    MSEObserver obs;
    obs.observe(make_input({64, 128}));
    EXPECT_GT(scale_of(obs.calculate_qparams(QuantDType::INT8, kSym)), 0.0f)
        << "Scale must be positive";
}

TEST_P(ObserversExtendedMultiDType, MSEMultipleObserves) {
    skip_if_not_f32_test_observer();
    MSEObserver obs;
    obs.observe(make_input({16, 32}));
    obs.observe(make_input({16, 32}, 4.0f));
    EXPECT_GT(scale_of(obs.calculate_qparams(QuantDType::INT8, kSym)), 0.0f);
}

TEST_P(ObserversExtendedMultiDType, MSEReset) {
    skip_if_not_f32_test_observer();
    MSEObserver obs;
    obs.observe(make_input({32, 64}, 10.0f));
    float scale_before = scale_of(obs.calculate_qparams(QuantDType::INT8, kSym));

    obs.reset();
    obs.observe(make_input({32, 64}, 0.01f));
    float scale_after = scale_of(obs.calculate_qparams(QuantDType::INT8, kSym));
    EXPECT_LT(scale_after, scale_before)
        << "After reset with smaller data, scale should decrease";
}

// ===========================================================================
// Per-channel MinMaxObserver (upcasts in observe -> full dtype sweep)
// ===========================================================================

TEST_P(ObserversExtendedMultiDType, PerChannelMinMaxObserveDoesNotCrash) {
    MinMaxObserver obs(/*per_channel=*/true);
    EXPECT_NO_THROW(obs.observe(make_input({8, 16, 4, 4})));
}

TEST_P(ObserversExtendedMultiDType, PerChannelMinMaxQParamsValid) {
    MinMaxObserver obs(/*per_channel=*/true);
    obs.observe(make_input({8, 16, 4, 4}));
    EXPECT_GT(scale_of(obs.calculate_qparams(QuantDType::INT8, kSym)), 0.0f)
        << "Scale must be positive";
}

TEST_P(ObserversExtendedMultiDType, PerChannelMinMaxReset) {
    MinMaxObserver obs(/*per_channel=*/true);
    obs.observe(make_input({4, 8, 3, 3}, 10.0f));
    float scale_before = scale_of(obs.calculate_qparams(QuantDType::INT8, kSym));

    obs.reset();
    obs.observe(make_input({4, 8, 3, 3}, 0.01f));
    float scale_after = scale_of(obs.calculate_qparams(QuantDType::INT8, kSym));
    EXPECT_LT(scale_after, scale_before)
        << "After reset with smaller data, scale should decrease";
}

TEST_P(ObserversExtendedMultiDType, PerChannelMinMaxMultipleObservesAccumulate) {
    MinMaxObserver obs(/*per_channel=*/true);

    obs.observe(make_input({4, 8}, 0.1f));
    float scale_small = scale_of(obs.calculate_qparams(QuantDType::INT8, kSym));

    obs.observe(make_input({4, 8}, 10.0f));
    float scale_large = scale_of(obs.calculate_qparams(QuantDType::INT8, kSym));

    EXPECT_GE(scale_large, scale_small)
        << "Scale should grow as observed range expands";
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ObserversExtendedMultiDType);