/**
 * @file test_mfcc_multidtype.cpp
 * @brief Tests for MelScale and MFCC operations
 *
 * Verifies:
 *  - MelScale output shape: (batch, n_mels, time_frames)
 *  - MFCC output shape: (batch, n_mfcc, time_frames)
 *  - MelScale filterbank shape sanity (n_mels rows, n_freqs cols)
 *  - Non-negative mel spectrogram values
 */

#include <gtest/gtest.h>
#include "backend_test_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/fft.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// audit-2 P.9 — This file uses BackendTest (not MultiBackendDTypeTest)
// deliberately: MFCC / Mel-scale tests cover shape, batching, and sign
// invariants — Float32 only. The audio-feature kernels do not
// expose Float64 / Float16 paths.
class MFCCTest : public BackendTest {};

// audit-3 T.1 helper: float-tensor element comparison against a CPU reference.
// numpy.allclose-style comparison: |a-b| <= atol + rtol*|expected|. The pure
// absolute form is wrong for large-magnitude outputs — an MFCC c0 coefficient
// is ~-145 for a constant signal (every mel bin pinned at the log(1e-10) floor),
// so a 1e-2 absolute bound demands 5-digit agreement that two different float32
// FFT backends (cuFFT vs CPU) cannot reach on the ~1e-9 leakage bins. rtol
// defaults to 0 so existing absolute-only callers are unchanged.
static void expect_tensors_near(const Tensor& actual, const Tensor& expected,
                                float tol = 1e-3f, float rtol = 0.0f) {
    ASSERT_EQ(actual.numel(), expected.numel());
    auto a = actual.to(Device::cpu()).to(DType::Float32);
    auto b = expected.to(Device::cpu()).to(DType::Float32);
    const float* ad = a.data<float>();
    const float* bd = b.data<float>();
    for (int64_t i = 0; i < a.numel(); ++i) {
        EXPECT_LE(std::abs(ad[i] - bd[i]), tol + rtol * std::abs(bd[i]))
            << "Mismatch at " << i << ": " << ad[i] << " vs " << bd[i];
    }
}

// MelScale: verify output shape matches (batch, n_mels, time_frames)
TEST_P(MFCCTest, MelScaleOutputShape) {
    const int64_t n_fft = 512;
    const int64_t n_freqs = n_fft / 2 + 1;  // 257
    const int64_t time_frames = 10;
    const int64_t n_mels = 40;

    // Create a fake spectrogram: shape (n_freqs, time_frames)
    auto spec_cpu = tenzor::ones({n_freqs, time_frames}, DType::Float32, Device::cpu());
    auto ref = fft::mel_scale(spec_cpu, n_mels, 0.0, 8000.0, 16000);

    auto spec = tenzor::ones({n_freqs, time_frames}, DType::Float32, device);
    auto result = fft::mel_scale(spec, n_mels, 0.0, 8000.0, 16000);

    ASSERT_EQ(result.ndim(), 2);
    EXPECT_EQ(result.shape()[0], n_mels);
    EXPECT_EQ(result.shape()[1], time_frames);
    expect_tensors_near(result, ref);
}

// MelScale: batched input shape
TEST_P(MFCCTest, MelScaleBatchedShape) {
    const int64_t batch = 3;
    const int64_t n_fft = 256;
    const int64_t n_freqs = n_fft / 2 + 1;  // 129
    const int64_t time_frames = 8;
    const int64_t n_mels = 64;

    auto spec_cpu = tenzor::ones({batch, n_freqs, time_frames}, DType::Float32, Device::cpu());
    auto ref = fft::mel_scale(spec_cpu, n_mels, 0.0, 8000.0, 16000);
    auto spec = tenzor::ones({batch, n_freqs, time_frames}, DType::Float32, device);
    auto result = fft::mel_scale(spec, n_mels, 0.0, 8000.0, 16000);

    ASSERT_EQ(result.ndim(), 3);
    EXPECT_EQ(result.shape()[0], batch);
    EXPECT_EQ(result.shape()[1], n_mels);
    EXPECT_EQ(result.shape()[2], time_frames);
    expect_tensors_near(result, ref);
}

// MelScale: output values should be non-negative for non-negative input
TEST_P(MFCCTest, MelScaleNonNegative) {
    const int64_t n_freqs = 257;
    const int64_t time_frames = 5;
    const int64_t n_mels = 40;

    auto spec = tenzor::ones({n_freqs, time_frames}, DType::Float32, device);
    auto result = fft::mel_scale(spec, n_mels, 0.0, 8000.0, 16000);

    auto result_cpu = result.to(Device::cpu());
    const float* data = result_cpu.data<float>();
    for (int64_t i = 0; i < result_cpu.numel(); ++i) {
        EXPECT_GE(data[i], 0.0f) << "Negative mel value at index " << i;
    }
}

// MelScale: default f_max (Nyquist)
TEST_P(MFCCTest, MelScaleDefaultFMax) {
    const int64_t n_freqs = 257;
    const int64_t time_frames = 5;
    const int64_t n_mels = 40;

    // f_max=0.0 should default to sample_rate/2
    auto spec_cpu = tenzor::ones({n_freqs, time_frames}, DType::Float32, Device::cpu());
    auto ref = fft::mel_scale(spec_cpu, n_mels, 0.0, 0.0, 16000);
    auto spec = tenzor::ones({n_freqs, time_frames}, DType::Float32, device);
    auto result = fft::mel_scale(spec, n_mels, 0.0, 0.0, 16000);

    ASSERT_EQ(result.ndim(), 2);
    EXPECT_EQ(result.shape()[0], n_mels);
    EXPECT_EQ(result.shape()[1], time_frames);
    expect_tensors_near(result, ref);
}

// MFCC: verify output shape
TEST_P(MFCCTest, MFCCOutputShape) {
    const int64_t signal_length = 16000;  // 1 second at 16kHz
    const int64_t n_fft = 400;
    const int64_t hop_length = 160;
    const int64_t n_mfcc = 13;
    const int64_t n_mels = 40;
    const int64_t sample_rate = 16000;

    // Use a spectrally-rich waveform (the SAME samples on CPU and the target
    // device) for a meaningful value comparison. A constant/DC waveform is
    // degenerate here: every mel bin lands on the log(1e-10) floor, so the MFCC
    // is pure floor-vs-FFT-noise and the comparison measures each backend's FFT
    // noise floor rather than MFCC correctness — white noise fills every mel bin
    // with real energy and the backends then agree to floating-point precision.
    auto wf_cpu = tenzor::randn({signal_length}, DType::Float32, Device::cpu());
    auto ref = fft::mfcc(wf_cpu, sample_rate, n_mfcc, n_mels, n_fft, hop_length);
    auto waveform = wf_cpu.to(device);
    auto result = fft::mfcc(waveform, sample_rate, n_mfcc, n_mels, n_fft, hop_length);

    ASSERT_EQ(result.ndim(), 2);
    EXPECT_EQ(result.shape()[0], n_mfcc);
    EXPECT_GT(result.shape()[1], 0);
    expect_tensors_near(result, ref, 1e-2f, /*rtol=*/1e-3f);
}

// MFCC: batched waveform
TEST_P(MFCCTest, MFCCBatchedShape) {
    const int64_t batch = 2;
    const int64_t signal_length = 8000;
    const int64_t n_fft = 400;
    const int64_t hop_length = 160;
    const int64_t n_mfcc = 20;
    const int64_t n_mels = 64;
    const int64_t sample_rate = 16000;

    // Spectrally-rich waveform shared between CPU reference and target device
    // (see MFCCOutputShape: a constant signal degenerates to a log-floor noise
    // comparison rather than validating MFCC).
    auto wf_cpu = tenzor::randn({batch, signal_length}, DType::Float32, Device::cpu());
    auto ref = fft::mfcc(wf_cpu, sample_rate, n_mfcc, n_mels, n_fft, hop_length);
    auto waveform = wf_cpu.to(device);
    auto result = fft::mfcc(waveform, sample_rate, n_mfcc, n_mels, n_fft, hop_length);

    ASSERT_EQ(result.ndim(), 3);
    EXPECT_EQ(result.shape()[0], batch);
    EXPECT_EQ(result.shape()[1], n_mfcc);
    EXPECT_GT(result.shape()[2], 0);
    expect_tensors_near(result, ref, 1e-2f, /*rtol=*/1e-3f);
}

// MFCC: n_mfcc must be <= n_mels
TEST_P(MFCCTest, MFCCInvalidNMFCC) {
    auto waveform = tenzor::zeros({8000}, DType::Float32, device);

    EXPECT_THROW(
        fft::mfcc(waveform, 16000, /*n_mfcc=*/100, /*n_mels=*/40),
        std::runtime_error
    );
}

// MFCC: output contains finite values
TEST_P(MFCCTest, MFCCFiniteValues) {
    const int64_t signal_length = 4000;
    const int64_t n_mfcc = 13;

    // Use a non-zero waveform to avoid log(0)
    auto waveform = tenzor::ones({signal_length}, DType::Float32, device);
    auto result = fft::mfcc(waveform, 16000, n_mfcc, 40, 400, 160);

    auto result_cpu = result.to(Device::cpu());
    const float* data = result_cpu.data<float>();
    for (int64_t i = 0; i < result_cpu.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(data[i]))
            << "Non-finite MFCC value at index " << i << ": " << data[i];
    }
}

INSTANTIATE_BACKEND_TESTS(MFCCTest);
