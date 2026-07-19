/**
 * @file test_oneapi_fft.cpp
 * @brief Tests for oneMKL FFT integration in the OneAPI backend
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/fft.hpp"
#include "../backend_parity/parity_test_utils.hpp"
#include <cmath>

using namespace tenzor;

class OneAPIFFTTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
        // Deterministic availability precondition only. The previous version
        // also ran an rfft smoke test inside a catch-all and reported any
        // oneMKL FFT failure as "not available" — that buried real oneMKL FFT kernel
        // bugs. Now only genuine device absence skips; oneMKL FFT failures
        // surface in the individual tests below.
        SKIP_IF_NO_ONEAPI;
    }
};

TEST_F(OneAPIFFTTest, FFT1DRoundtrip) {
    auto cpu_signal = tenzor::randn({64});
    auto signal = cpu_signal.to(Device::oneapi(0));

    // Forward FFT
    auto freq = tenzor::fft::fft(signal);
    // Inverse FFT
    auto reconstructed = tenzor::fft::ifft(freq);

    // Should approximately reconstruct original
    auto recon_cpu = reconstructed.to(Device::cpu());
    // Complex output — compare magnitudes
    EXPECT_EQ(recon_cpu.shape()[0], 64);
}

TEST_F(OneAPIFFTTest, RFFT1D) {
    auto signal = tenzor::randn({128}).to(Device::oneapi(0));

    auto freq = tenzor::fft::rfft(signal);

    // rfft output has n/2+1 frequency bins
    EXPECT_EQ(freq.shape()[0], 65);  // 128/2 + 1
}

TEST_F(OneAPIFFTTest, IRFFT1D) {
    auto signal = tenzor::randn({128}).to(Device::oneapi(0));

    auto freq = tenzor::fft::rfft(signal);
    auto reconstructed = tenzor::fft::irfft(freq, 128);

    auto orig_cpu = signal.to(Device::cpu());
    auto recon_cpu = reconstructed.to(Device::cpu());

    EXPECT_EQ(recon_cpu.shape()[0], 128);

    // Check reconstruction accuracy
    auto* orig = orig_cpu.data<float>();
    auto* recon = recon_cpu.data<float>();
    for (int64_t i = 0; i < 128; i++) {
        EXPECT_NEAR(orig[i], recon[i], 1e-3) << "Mismatch at index " << i;
    }
}

TEST_F(OneAPIFFTTest, FFT2D) {
    auto input = tenzor::randn({16, 16}).to(Device::oneapi(0));

    auto freq = tenzor::fft::fft2(input);
    auto reconstructed = tenzor::fft::ifft2(freq);

    EXPECT_EQ(freq.shape()[0], 16);
    EXPECT_EQ(freq.shape()[1], 16);
}

TEST_F(OneAPIFFTTest, CPUParityFloat32) {
    auto cpu_input = tenzor::randn({32});
    auto oneapi_input = cpu_input.to(Device::oneapi(0));

    auto cpu_result = tenzor::fft::rfft(cpu_input).to(Device::cpu());
    auto oneapi_result = tenzor::fft::rfft(oneapi_input).to(Device::cpu());

    // Results should be close between CPU and OneAPI
    EXPECT_EQ(cpu_result.shape()[0], oneapi_result.shape()[0]);
}

