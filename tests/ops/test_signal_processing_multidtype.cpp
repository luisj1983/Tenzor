/**
 * @file test_signal_processing_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for signal processing operations
 *
 * Covers: STFT, ISTFT, CDist
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/fft.hpp>
#include <tenzor/ops/windows.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Fixture
// ============================================================================

class SignalProcessingMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void skipIfHalf() {
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
            GTEST_SKIP() << "STFT/ISTFT not supported for half precision";
        }
    }
};

// ============================================================================
// STFT Tests
// ============================================================================

TEST_P(SignalProcessingMultiDTypeTest, STFTOutputShape) {
    skipIfHalf();
    // Input: (batch=1, signal_length=256), n_fft=64, hop_length=16
    // Expected output: (1, n_fft/2+1, num_frames) where num_frames depends on settings
    auto signal = createRandn({1, 256});
    int64_t n_fft = 64;
    int64_t hop_length = 16;

    auto result = tenzor::fft::stft(signal, n_fft, hop_length, n_fft);
    auto shape = result.shape();

    EXPECT_EQ(shape[0], 1);              // batch
    EXPECT_EQ(shape[1], n_fft / 2 + 1); // frequency bins
    EXPECT_GT(shape[2], 0);              // time frames
    expectDevice(result);
}

TEST_P(SignalProcessingMultiDTypeTest, STFTWithWindow) {
    skipIfHalf();
    auto signal = createRandn({2, 512});
    int64_t n_fft = 128;
    int64_t hop_length = 32;
    auto window = tenzor::hann_window(n_fft, true, DType::Float32, device());
    if (dtype() != DType::Float32) window = window.to(dtype());

    auto result = tenzor::fft::stft(signal, n_fft, hop_length, n_fft, window);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], n_fft / 2 + 1);
}

// ============================================================================
// ISTFT Round-Trip Tests
// ============================================================================

TEST_P(SignalProcessingMultiDTypeTest, ISTFTRoundTrip) {
    skipIfHalf();
    // istft(stft(x)) should approximately recover x
    int64_t signal_length = 256;
    int64_t n_fft = 64;
    int64_t hop_length = 16;

    auto signal = createRandn({1, signal_length});
    auto window = tenzor::hann_window(n_fft, true, DType::Float32, device());
    if (dtype() != DType::Float32) window = window.to(dtype());

    auto stft_result = tenzor::fft::stft(signal, n_fft, hop_length, n_fft, window);
    auto reconstructed = tenzor::fft::istft(stft_result, n_fft, hop_length, n_fft, window,
                                        true, false, true, signal_length);

    // Reconstructed should be close to original
    auto sig_f32 = signal.to(Device::cpu()).to(DType::Float32);
    auto rec_f32 = reconstructed.to(Device::cpu()).to(DType::Float32);

    float max_diff = compute_max_abs(tenzor::sub(sig_f32, rec_f32));
    EXPECT_LT(max_diff, std::max(atol() * 10.0f, 0.1f))
        << "STFT-ISTFT round trip reconstruction error too large";
}

// ============================================================================
// CDist Tests
// ============================================================================

TEST_P(SignalProcessingMultiDTypeTest, CDistEuclidean) {
    // cdist computes pairwise distances between two sets of points
    // Input: x1=(B, P, D), x2=(B, R, D), Output: (B, P, R)
    auto x1 = createRandn({1, 3, 4});  // 3 points in 4D
    auto x2 = createRandn({1, 5, 4});  // 5 points in 4D

    auto result = tenzor::cdist(x1, x2);
    expectShape(result, {1, 3, 5});
    expectDevice(result);

    // All distances should be non-negative
    float min_val = compute_min(result);
    EXPECT_GE(min_val, -atol());
}

TEST_P(SignalProcessingMultiDTypeTest, CDistSelfDistance) {
    // Distance of a point to itself should be ~0
    auto x = createRandn({1, 2, 3});
    auto result = tenzor::cdist(x, x);

    // Diagonal entries (distance from point to itself) should be ~0
    auto result_f32 = result.to(Device::cpu()).to(DType::Float32);
    auto* r = result_f32.data<float>();
    EXPECT_NEAR(r[0], 0.0f, std::max(atol(), 1e-3f));  // d(p0, p0)
    // r[3] = d(p1, p1) for shape (1, 2, 2)
    EXPECT_NEAR(r[3], 0.0f, std::max(atol(), 1e-3f));
}

TEST_P(SignalProcessingMultiDTypeTest, CDistBatched) {
    auto x1 = createRandn({4, 6, 8});
    auto x2 = createRandn({4, 10, 8});
    auto result = tenzor::cdist(x1, x2);
    expectShape(result, {4, 6, 10});
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SignalProcessingMultiDTypeTest);
